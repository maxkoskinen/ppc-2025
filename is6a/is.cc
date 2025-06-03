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

constexpr int parallel = 8;
typedef float float8_t __attribute__ ((vector_size (parallel * sizeof(float))));
constexpr float8_t f8zero = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

static inline float min8(float8_t vec){
    float min_val = vec[0];
    for (int i = 1; i < parallel; ++i) {
        if (vec[i] < min_val) min_val = vec[i];
    }
    return min_val;
}

void compute_integral(int ny, int nx, const float* data, float* sum) {
    #pragma omp parallel for
    for(int x = 0; x <= nx; x++) {
        sum[x] = 0.0f;
    }
    
    #pragma omp parallel for
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
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    
    float* sum_d = static_cast<float*>(_mm_malloc(sizeof(float) * (ny+1) * (nx+1), 64));
    
    compute_integral(ny, nx, data, sum_d);
    
    float total_sum = sum_d[nx + (nx+1) * ny];
    int total_pix = nx * ny;
    
    float err_best = 0.0f;
    int best_width = 0, best_height = 0;
    
    #pragma omp parallel
    {
        float err_local = 0.0f;
        float8_t err_8w = f8zero;
        float err_w = 0.0f;
        int w_local = 0, h_local = 0;
        
        #pragma omp for schedule(dynamic)
        for(int h = 1; h <= ny; h++) {
            for(int w = 1; w <= nx; w++) {
                int rect_area = w * h;
                int out = total_pix - rect_area;
                
                if (rect_area == total_pix) continue;
                
                float inv_inner = 1.0f / rect_area;
                float inv_outer = 1.0f / out;

                for(int y0 = 0; y0 <= (ny - h); y0++) {
                    int x0 = 0;
                    for(; x0 <= nx - w - (parallel - 1); x0 += parallel) {
                        int y1 = y0 + h;
                        int x1 = x0 + w;
                        
                        // load 8 consecutive windows
                        __m256 x1y1 = _mm256_loadu_ps(&sum_d[x1 + (nx+1) * y1]);
                        __m256 x0y1 = _mm256_loadu_ps(&sum_d[x0 + (nx+1) * y1]);
                        __m256 x1y0 = _mm256_loadu_ps(&sum_d[x1 + (nx+1) * y0]);
                        __m256 x0y0 = _mm256_loadu_ps(&sum_d[x0 + (nx+1) * y0]);

                        float8_t inner_sum = x1y1 - x0y1 - x1y0 + x0y0;
                        float8_t outer_sum = total_sum - inner_sum;
                        
                        float8_t inner_avg = inner_sum * inv_inner;
                        float8_t outer_avg = outer_sum * inv_outer;
                        
                        float8_t cost = -(inner_sum * inner_avg + outer_sum * outer_avg);

                        err_8w = cost < err_8w ? cost : err_8w;
                    }
                    for (; x0 <= nx - w; ++x0) {
                        int y1 = y0 + h;
                        int x1 = x0 + w;

                        float inner_sum = sum_d[x1 + (nx+1) * y1]
                                        - sum_d[x0 + (nx+1) * y1]
                                        - sum_d[x1 + (nx+1) * y0]
                                        + sum_d[x0 + (nx+1) * y0];
                        
                        float outer_sum = total_sum - inner_sum;

                        float inner_avg = inner_sum * inv_inner;
                        float outer_avg = outer_sum * inv_outer;
                        
                        float cost = -(inner_sum * inner_avg + outer_sum * outer_avg);

                        if (cost < err_local) {
                            err_local = cost;
                            w_local = w;
                            h_local = h;
                        }
                    }

                }
                
                err_w = min8(err_8w);
                
                if(err_w < err_local) {
                    err_local = err_w;
                    w_local = w;
                    h_local = h;
                }
            }
        }
        
        #pragma omp critical
        {
            if(err_local < err_best) {
                err_best = err_local;
                best_width = w_local;
                best_height = h_local;
            }
        }
    }
    
    int rect_area = best_width * best_height;
    int out_area = total_pix - rect_area;
    float inv_inner = 1.0f / rect_area;
    float inv_outer = 1.0f / out_area;
    
    float final_best_err = 0.0f;
    
    for(int y0 = 0; y0 <= (ny - best_height); y0++) {
        for(int x0 = 0; x0 <= (nx - best_width); x0++) {
            int y1 = y0 + best_height;
            int x1 = x0 + best_width;
            
            float inner_sum = sum_d[x1 + (nx+1) * y1]
                         - sum_d[x0 + (nx+1) * y1]
                         - sum_d[x1 + (nx+1) * y0]
                         + sum_d[x0 + (nx+1) * y0];
            
            float outer_sum = total_sum - inner_sum;
            float inner_avg = inner_sum * inv_inner;
            float outer_avg = outer_sum * inv_outer;
            
            float cost = -(inner_sum * inner_avg + outer_sum * outer_avg);
            
            if(cost < final_best_err) {
                final_best_err = cost;
                result.x0 = x0;
                result.y0 = y0;
                result.x1 = x1;
                result.y1 = y1;
            }
        }
    }
    
    float inner_sum = sum_d[result.x1 + (nx+1) * result.y1]
                 - sum_d[result.x0 + (nx+1) * result.y1]
                 - sum_d[result.x1 + (nx+1) * result.y0]
                 + sum_d[result.x0 + (nx+1) * result.y0];
                 
    float outer_sum = total_sum - inner_sum;
    
    for(int i = 0; i < 3; i++) {
        result.inner[i] = inner_sum / rect_area;
        result.outer[i] = outer_sum / out_area;
    }
    
    _mm_free(sum_d);
    return result;
}