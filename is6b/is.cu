#include <cmath>
#include <vector>
#include <algorithm> 
#include <iostream>
#include <limits>
#include <vector>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};
/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

static inline void check(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void find_best_rectangle_kernel(int ny, int nx, const float *d_sum_d,
                                        float *d_errors_per_wh, int *d_coords_per_wh,
                                        float total_sum, int total_pix) {

    int w = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int h = threadIdx.y + blockIdx.y * blockDim.y + 1;

    if (w > nx || h > ny) return;

    float min_cost_for_this_wh = 0.0f;
    int best_x0_this_wh = 0;
    int best_y0_this_wh = 0;
    int best_x1_this_wh = 0;
    int best_y1_this_wh = 0;

    int rect_area = w * h;
    int out_area = total_pix - rect_area;

    if (rect_area == 0 || out_area <= 0) {
        int wh_idx = (h - 1) * nx + (w - 1);
        d_errors_per_wh[wh_idx] = 1.0f;
        d_coords_per_wh[4 * wh_idx + 0] = 0;
        d_coords_per_wh[4 * wh_idx + 1] = 0;
        d_coords_per_wh[4 * wh_idx + 2] = 0;
        d_coords_per_wh[4 * wh_idx + 3] = 0;
        return;
    }

    float inv_inner_area = 1.0f / static_cast<float>(rect_area);
    float inv_outer_area = 1.0f / static_cast<float>(out_area);

    for (int y0 = 0; y0 <= ny - h; ++y0) {
        for (int x0 = 0; x0 <= nx - w; ++x0) {
            int x1 = x0 + w;
            int y1 = y0 + h;

            float inner_sum = d_sum_d[x1 + y1 * (nx+1)]
                              - d_sum_d[x0 + y1 * (nx+1)]
                              - d_sum_d[x1 + y0 * (nx+1)]
                              + d_sum_d[x0 + y0 * (nx+1)];
            
            float outer_sum = total_sum - inner_sum;

            float inner_avg = inner_sum * inv_inner_area;
            float outer_avg = outer_sum * inv_outer_area;
            
            float current_cost = -(inner_sum * inner_avg + outer_sum * outer_avg);

            if (current_cost < min_cost_for_this_wh) {
                min_cost_for_this_wh = current_cost;
                best_x0_this_wh = x0;
                best_y0_this_wh = y0;
                best_x1_this_wh = x1;
                best_y1_this_wh = y1;
            }
        }
    }

    int wh_idx = (h - 1) * nx + (w - 1); 
    d_errors_per_wh[wh_idx] = min_cost_for_this_wh;
    d_coords_per_wh[4 * wh_idx + 0] = best_y0_this_wh;
    d_coords_per_wh[4 * wh_idx + 1] = best_x0_this_wh;
    d_coords_per_wh[4 * wh_idx + 2] = best_y1_this_wh;
    d_coords_per_wh[4 * wh_idx + 3] = best_x1_this_wh;
}

void compute_integral(int ny, int nx, const float* data, std::vector<float>& sum) {
    for(int x = 0; x <= nx; x++) {
        sum[x] = 0.0f;
    }
    
    for(int y = 0; y <= ny; y++) {
        sum[(nx+1) * y] = 0.0f;
    }
    
    for(int y1 = 1; y1 <= ny; y1++) {
        float temp = 0.0f;
        for(int x1 = 1; x1 <= nx; x1++) {
            int x = x1 - 1;
            int y = y1 - 1;
            temp += data[3 * x + 3 * nx * y];
            sum[x1 + (nx+1) * y1] = sum[x1 + (nx+1) * y] + temp;
        }
    }
}

Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};

    std::vector<float> sum_d((ny+1) * (nx+1));
    
    compute_integral(ny, nx, data, sum_d);

    float total_sum = sum_d[nx + (nx+1) * ny];
    int total_pix = nx * ny;

    float *d_sum_GPU = nullptr;
    float *d_errors_per_wh_GPU = nullptr;
    int   *d_coords_per_wh_GPU = nullptr;

    CHECK(cudaMalloc((void**)&d_sum_GPU, (ny+1) * (nx+1) * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_errors_per_wh_GPU, ny * nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_coords_per_wh_GPU, 4 * ny * nx * sizeof(int)));

    CHECK(cudaMemcpy(d_sum_GPU, sum_d.data(), (ny+1) * (nx+1) * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 32);
    dim3 dimGrid(divup(nx, dimBlock.x), divup(ny, dimBlock.y));

    find_best_rectangle_kernel<<<dimGrid, dimBlock>>>(
        ny, nx, d_sum_GPU,
        d_errors_per_wh_GPU, d_coords_per_wh_GPU,
        total_sum, total_pix);
    CHECK(cudaGetLastError());

    std::vector<float> h_errors_per_wh(ny * nx);
    std::vector<int> h_coords_per_wh(4 * ny * nx);

    CHECK(cudaMemcpy(h_errors_per_wh.data(), d_errors_per_wh_GPU, ny * nx * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_coords_per_wh.data(), d_coords_per_wh_GPU, 4 * ny * nx * sizeof(int), cudaMemcpyDeviceToHost));

    auto min_error_iterator = std::min_element(h_errors_per_wh.begin(), h_errors_per_wh.end());
    int best_wh_pair_idx = std::distance(h_errors_per_wh.begin(), min_error_iterator);
    
    if (*min_error_iterator < 0.0f && total_pix > 0) {
        result.y0 = h_coords_per_wh[4 * best_wh_pair_idx + 0];
        result.x0 = h_coords_per_wh[4 * best_wh_pair_idx + 1];
        result.y1 = h_coords_per_wh[4 * best_wh_pair_idx + 2];
        result.x1 = h_coords_per_wh[4 * best_wh_pair_idx + 3];

        float final_inner_sum = sum_d[result.x1 + result.y1 * (nx+1)] -
                                sum_d[result.x0 + result.y1 * (nx+1)] -
                                sum_d[result.x1 + result.y0 * (nx+1)] +
                                sum_d[result.x0 + result.y0 * (nx+1)];
        
        float final_outer_sum = total_sum - final_inner_sum;

        int final_rect_area = (result.y1 - result.y0) * (result.x1 - result.x0);
        int final_out_area = total_pix - final_rect_area;

        if (final_rect_area > 0 && final_out_area > 0) {
            for (int c = 0; c < 3; ++c) {
                result.inner[c] = final_inner_sum / static_cast<float>(final_rect_area);
                result.outer[c] = final_outer_sum / static_cast<float>(final_out_area);
            }
        } else if (final_rect_area == total_pix && total_pix > 0) {
             for (int c = 0; c < 3; ++c) {
                result.inner[c] = final_inner_sum / static_cast<float>(final_rect_area);
                result.outer[c] = 0.0f;
            }
        } else if (final_out_area == total_pix && total_pix > 0) {
             for (int c = 0; c < 3; ++c) {
                result.inner[c] = 0.0f;
                result.outer[c] = final_outer_sum / static_cast<float>(final_out_area);
            }
        }
    }

    CHECK(cudaFree(d_sum_GPU));
    CHECK(cudaFree(d_errors_per_wh_GPU));
    CHECK(cudaFree(d_coords_per_wh_GPU));

    return result;
}