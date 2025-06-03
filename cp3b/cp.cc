/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <cmath>
#include <vector>
#include <omp.h>
#include <x86intrin.h>
#include <algorithm>
#include <tuple>


typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));
const float8_t float8_0 = {0,0,0,0,0,0,0,0};

// helper functions
inline double horizontal_sum(__m256d vec) {
    __m128d hi = _mm256_extractf128_pd(vec, 1);
    __m128d lo = _mm256_castpd256_pd128(vec);
    __m128d sum1 = _mm_add_pd(hi, lo);
    __m128d sum2 = _mm_hadd_pd(sum1, sum1);
    return _mm_cvtsd_f64(sum2);
}

double r_mean(const float* row, int nx) {
    __m256d sum_vec = _mm256_setzero_pd();
    
    int j = 0;
    for (; j + 4 <= nx; j += 4) {
        __m128 float_vals = _mm_loadu_ps(&row[j]);
        __m256d double_vals = _mm256_cvtps_pd(float_vals);
        sum_vec = _mm256_add_pd(sum_vec, double_vals);
    }

    double sum = horizontal_sum(sum_vec);

    // rem elements
    for (; j < nx; ++j) {
        sum += static_cast<double>(row[j]);
    }

    return sum / static_cast<double>(nx);
}

double r_std(const float* row, int nx, double mean) {
    __m256d square_sum_vec = _mm256_setzero_pd();
    const __m256d mean_vec = _mm256_set1_pd(mean);
    int j = 0;
    
    for (; j + 4 <= nx; j += 4) {
        __m128 float_vals = _mm_loadu_ps(&row[j]);
        __m256d double_vals = _mm256_cvtps_pd(float_vals);
        __m256d diff = _mm256_sub_pd(double_vals, mean_vec);
        square_sum_vec = _mm256_fmadd_pd(diff, diff, square_sum_vec); // diff^2 + sum
    }

    double square_sum = horizontal_sum(square_sum_vec);

    // rem elements
    for (; j < nx; ++j) {
        double diff = static_cast<double>(row[j]) - mean;
        square_sum += diff * diff;
    }

    return std::sqrt(square_sum);
}

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }


void correlate(int ny, int nx, const float *data, float *result) {
    //std::vector<double> normalized_data(ny * nx, 0.0);
    double* normalized_data = static_cast<double*>(_mm_malloc(sizeof(double) * ny * nx+64, 32));
    int nb = 8;
    int na = (ny + nb - 1) / nb;
    std::vector<float8_t> vd(na * nx);
    std::vector<float> nd(ny*nx);

    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < ny; ++i) {
        double mean = r_mean(&data[i * nx], nx);
        double stddev = r_std(&data[i * nx], nx, mean);
        
        double inv_stddev = 1 /stddev;

        for (int j = 0; j < nx; ++j) {
            normalized_data[i * nx + j] = (data[i * nx + j] - mean) * inv_stddev;
        }
    }
    
    #pragma omp parallel for schedule(static,1)
    for (int ja = 0; ja < na; ++ja) {
        for (int i = 0; i < nx; ++i) {
            for (int jb = 0; jb < nb; ++jb) {
                int j = nb * ja + jb;
                const int PF = 20;
                __builtin_prefetch(&normalized_data[nx * j + i + PF]);
                vd[nx * ja + i][jb] = (j < ny) ? normalized_data[nx * j + i] : 0.0;
            }
        }
    }

    constexpr int kBlockSize = 64;
    #pragma omp parallel for schedule(dynamic)
    for (int ia = 0; ia < na; ++ia) {
        for (int ja = ia; ja < na; ++ja) {

            float8_t z000 = float8_0;
            float8_t z001 = float8_0;
            float8_t z010 = float8_0;
            float8_t z011 = float8_0;
            float8_t z100 = float8_0;
            float8_t z101 = float8_0;
            float8_t z110 = float8_0;
            float8_t z111 = float8_0;

            for (int k0 = 0; k0 < nx; k0 += kBlockSize) {
                int kend = std::min(k0 + kBlockSize, nx);
                #pragma omp simd 
                for (int k = k0; k < kend; ++k) {
                        const int PF = 20;
                    __builtin_prefetch(&vd[nx * ia + k + PF]);
                    __builtin_prefetch(&vd[nx * ja + k + PF]);

                    float8_t a000 = vd[nx * ia + k];
                    float8_t b000 = vd[nx * ja + k];
                    float8_t a100 = swap4(a000);
                    float8_t a010 = swap2(a000);
                    float8_t a110 = swap2(a100);
                    float8_t b001 = swap1(b000);

                    z000 = z000 + (a000 * b000);
                    z001 = z001 + (a000 * b001);
                    z010 = z010 + (a010 * b000);
                    z011 = z011 + (a010 * b001);
                    z100 = z100 + (a100 * b000);
                    z101 = z101 + (a100 * b001);
                    z110 = z110 + (a110 * b000);
                    z111 = z111 + (a110 * b001);
                }
            }

            float8_t vv[8] = {z000, z001, z010, z011, z100, z101, z110, z111};
            for(int kb=1; kb < nb; kb += 2){
                vv[kb] = swap1(vv[kb]);
            }

            for (int jb = 0; jb < nb; ++jb) {
                for (int ib = 0; ib < nb; ++ib) {
                    int i = ib + nb * ia;
                    int j = jb + nb * ja;
                    if (j < ny && i < ny && i <= j) {
                        result[ny * i + j] = vv[ib ^ jb][jb];
                    }
                }
            }
        }
    }
    _mm_free(normalized_data);

}