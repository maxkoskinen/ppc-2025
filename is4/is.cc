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

#include <cmath>
#include <vector>
#include <omp.h>
#include <x86intrin.h>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
constexpr double4_t d4zero = {0,0,0,0};
const int PF = 20;

void compute_integral(int ny, int nx, double4_t* vd,
        double4_t* sum) {
            
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {

            __builtin_prefetch(&sum[(x+1) + PF + (nx+1) * (y+1)],0,0);
            __builtin_prefetch(&sum[x + PF + (nx+1) * (y+1)],0,0);
            __builtin_prefetch(&vd[x + PF + (nx+1) * y],0,0);

            double4_t val = vd[x + nx * y];

            sum[(x+1) + (nx+1) * (y+1)] = val
                + sum[x + (nx+1) * (y+1)] 
                + sum[(x+1) + (nx+1) * y]
                - sum[x + (nx+1) * y];
        }
    }
}


Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    double min_cost = std::numeric_limits<double>::infinity();
    int total_pix = ny * nx;

    double4_t* vd = static_cast<double4_t*>(_mm_malloc(sizeof(double4_t) * ny * nx, 64));
    double4_t* vsum = static_cast<double4_t*>(_mm_malloc(sizeof(double4_t) * (ny+1) * (nx+1), 64));

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            double4_t vec = {
                data[0+3 * x + 3 * nx * y],
                data[1+3 * x + 3 * nx * y],
                data[2+3 * x + 3 * nx * y],
                0.0
            };
            vd[x + nx * y] = vec; 
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= ny; ++i) {
        vsum[i * (nx + 1)] = d4zero;
    }
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= nx; ++j) {
        vsum[j] = d4zero;
    }

    compute_integral(ny, nx, vd, vsum);

    double4_t total_sum_vec = vsum[nx + (nx+1) * ny];

    #pragma omp parallel
    {   

        Result thread_local_best_result = {0, 0, 0, 0, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
        double thread_local_min_cost = std::numeric_limits<double>::infinity();
        bool thread_found_candidate = false;

        #pragma omp for schedule(dynamic)
        for (int h = 0; h <= ny; ++h) {

            double h_iter_min_cost = std::numeric_limits<double>::infinity();
            int h_iter_y0 = 0, h_iter_x0 = 0, h_iter_w = 0;
            double4_t h_iter_inner_avg = d4zero;
            double4_t h_iter_outer_avg = d4zero;

            for (int w = 1; w <= nx; ++w) {
                if (total_pix == h*w) continue;

                double w_iter_min_cost = std::numeric_limits<double>::infinity();
                int w_iter_y0 = 0, w_iter_x0 = 0;
                double4_t w_iter_inner_avg = d4zero;
                double4_t w_iter_outer_avg = d4zero;

                for (int y0 = 0; y0 <= ny-h; ++y0) {
                    for (int x0 = 0; x0 <= nx-w; ++x0) {
                        int y1 = y0 + h;
                        int x1 = x0 + w;

                        int rect_area = h * w;
                        int outer_area = total_pix - rect_area;

                        double inv_inner = static_cast<double>(1) / rect_area;
                        double inv_outer = static_cast<double>(1) / outer_area;

                        double4_t inner_sum = vsum[x1 + (nx+1) * y1]
                                            - vsum[x0 + (nx+1) * y1]
                                            - vsum[x1 + (nx+1) * y0]
                                            + vsum[x0 + (nx+1) * y0];
                        
                        double4_t outer_sum = total_sum_vec - inner_sum;

                        double4_t inner_avg_candidate = inv_inner * inner_sum;
                        double4_t outer_avg_candidate = inv_outer * outer_sum;


                        double4_t error_vec = -(inner_sum * inner_avg_candidate + outer_sum * outer_avg_candidate);
                        double current_cost = error_vec[0] + error_vec[1] + error_vec[2];


                        if (current_cost < w_iter_min_cost) {
                            w_iter_min_cost = current_cost;
                            w_iter_y0 = y0;
                            w_iter_x0 = x0;
                            w_iter_inner_avg = inner_avg_candidate;
                            w_iter_outer_avg = outer_avg_candidate;
                        }
                    }
                }
                if (w_iter_min_cost < h_iter_min_cost) {
                    h_iter_min_cost = w_iter_min_cost;
                    h_iter_y0 = w_iter_y0;
                    h_iter_x0 = w_iter_x0;
                    h_iter_w = w;
                    h_iter_inner_avg = w_iter_inner_avg;
                    h_iter_outer_avg = w_iter_outer_avg;
                }
            }
            if (h_iter_min_cost < thread_local_min_cost) {
                thread_local_min_cost = h_iter_min_cost;
                thread_local_best_result.y0 = h_iter_y0;
                thread_local_best_result.x0 = h_iter_x0;
                thread_local_best_result.y1 = h_iter_y0 + h;
                thread_local_best_result.x1 = h_iter_x0 + h_iter_w;
                for (int c = 0; c < 3; ++c) {
                    thread_local_best_result.inner[c] = static_cast<float>(h_iter_inner_avg[c]);
                    thread_local_best_result.outer[c] = static_cast<float>(h_iter_outer_avg[c]);
                }
                thread_found_candidate = true;
            }
        }
        if (thread_found_candidate) {
            #pragma omp critical
            {
                if (thread_local_min_cost < min_cost) {
                    min_cost = thread_local_min_cost;
                    result = thread_local_best_result;
                }
            }
        }

    }
    
    _mm_free(vd);
    _mm_free(vsum);

    return result;
}
