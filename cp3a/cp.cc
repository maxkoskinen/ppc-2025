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

typedef double double8_t __attribute__ ((vector_size (8 * sizeof(double))));
const double8_t double8_0 = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

__attribute__((target("avx512f")))
double r_mean(const float* row, int nx) {
    __m512d sum_vec = _mm512_setzero_pd();
    
    int j = 0;
    for (; j + 8 <= nx; j += 8) {
        __m256 float_vals = _mm256_loadu_ps(&row[j]);
        __m512d double_vals = _mm512_cvtps_pd(float_vals);
        sum_vec = _mm512_add_pd(sum_vec, double_vals);
    }

    double sum = _mm512_reduce_add_pd(sum_vec);
    for (; j < nx; ++j) {
        sum += static_cast<double>(row[j]);
    }

    return sum / static_cast<double>(nx);
}
__attribute__((target("avx512f")))
double r_std(const float* row, int nx, double mean) {
    __m512d square_sum_vec = _mm512_setzero_pd();
    const __m512d mean_vec = _mm512_set1_pd(mean);
    int j = 0;
    
    for (; j + 8 <= nx; j += 8) {
        __m256 float_vals = _mm256_loadu_ps(&row[j]);
        __m512d double_vals = _mm512_cvtps_pd(float_vals);
        __m512d diff = _mm512_sub_pd(double_vals, mean_vec);
        square_sum_vec = _mm512_fmadd_pd(diff, diff, square_sum_vec); // diff^2 + sum
    }

    double square_sum = _mm512_reduce_add_pd(square_sum_vec);
    for (; j < nx; ++j) {
        double diff = static_cast<double>(row[j]) - mean;
        square_sum += diff * diff;
    }

    return std::sqrt(square_sum);
}

__attribute__((target("avx512f")))
static inline double8_t swap4(double8_t x) { 
    return (double8_t)_mm512_permutexvar_pd(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), (__m512d)x);
}
__attribute__((target("avx512f")))
static inline double8_t swap2(double8_t x) {
    return (double8_t)_mm512_permutexvar_pd(_mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2), (__m512d)x);
}
__attribute__((target("avx512f")))
static inline double8_t swap1(double8_t x) {
    return (double8_t)_mm512_permutexvar_pd(_mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1), (__m512d)x);
}

__attribute__((target("avx512f")))
void correlate(int ny, int nx, const float *data, float *result) {

    double* normalized_data = static_cast<double*>(_mm_malloc(sizeof(double) * ny * nx, 64));
    
    int nb = 8;
    int na = (ny + nb - 1) / nb;

    double8_t* vd = static_cast<double8_t*>(_mm_malloc(sizeof(double8_t) * na * nx, 64));

    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < ny; ++i) {
        double mean = r_mean(&data[i * nx], nx);
        double stddev = r_std(&data[i * nx], nx, mean);
        double inv_stddev = 1.0 / stddev;

        for (int j = 0; j < nx; ++j) {
            normalized_data[i * nx + j] = (data[i * nx + j] - mean) * inv_stddev;
        }
    }

    #pragma omp parallel for schedule(static,1)
    for (int ja = 0; ja < na; ++ja) {
        for (int i = 0; i < nx; ++i) {
            #pragma omd simd
            for (int jb = 0; jb < nb; ++jb) {
                int j = nb * ja + jb;
                const int PF = 32;
                __builtin_prefetch(&normalized_data[nx * j + i + PF],0,0);
                vd[nx * ja + i][jb] = (j < ny) ? normalized_data[nx * j + i] : 0.0;
            }
        }
    }

    constexpr int kBlockSize = 512;
    #pragma omp parallel for schedule(dynamic,2)
    for (int ia = 0; ia < na; ++ia) {
        for (int ja = ia; ja < na; ++ja) {

            double8_t z000 = double8_0;
            double8_t z001 = double8_0;
            double8_t z010 = double8_0;
            double8_t z011 = double8_0;
            double8_t z100 = double8_0;
            double8_t z101 = double8_0;
            double8_t z110 = double8_0;
            double8_t z111 = double8_0;

            for (int k0 = 0; k0 < nx; k0 += kBlockSize) {
                int kend = std::min(k0 + kBlockSize, nx);
                for (int k = k0; k < kend; ++k) {
                    //const int PF = 20;
                    //__builtin_prefetch(&vd[nx * ia + k + PF]);
                    //__builtin_prefetch(&vd[nx * ja + k + PF]);

                    double8_t a000 = vd[nx * ia + k];
                    double8_t b000 = vd[nx * ja + k];
                    
                    double8_t a100 = swap4(a000);
                    double8_t a010 = swap2(a000);
                    double8_t a110 = swap2(a100);
                    double8_t b001 = swap1(b000);

                    z000 = _mm512_fmadd_pd(a000, b000, z000);
                    z001 = _mm512_fmadd_pd(a000, b001, z001);
                    z010 = _mm512_fmadd_pd(a010, b000, z010);
                    z011 = _mm512_fmadd_pd(a010, b001, z011);
                    z100 = _mm512_fmadd_pd(a100, b000, z100);
                    z101 = _mm512_fmadd_pd(a100, b001, z101);
                    z110 = _mm512_fmadd_pd(a110, b000, z110);
                    z111 = _mm512_fmadd_pd(a110, b001, z111);
                }
            }

            double8_t vv[8] = {z000, z001, z010, z011, z100, z101, z110, z111};
            
            for(int kb = 1; kb < nb; kb += 2){
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
    _mm_free(vd);
}