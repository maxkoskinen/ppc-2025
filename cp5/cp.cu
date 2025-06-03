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
#include <cuda_runtime.h>
#include <iostream>

static inline void check(cudaError_t err, const char *context)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)


__global__ void preprocess(int ny, int nx, int nn, float* data, float* transpose) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= nn) return;

    if (y >= ny) return;

    float sum = 0;
    float sq_sum = 0;

    for (int i = 0; i < nx; i++) {
        float val = data[i + y * nx];
        sum += val;
    }
    float mean = sum / nx;

    for (int i = 0; i < nx; i++) {
        float val = data[i + y * nx];
        float diff = val - mean;
        sq_sum += diff * diff;
    }

    float stddev = std::sqrt(sq_sum);
    
    if (stddev == 0) stddev = 1.0f;

    for (int i = 0; i < nx; ++i) {
        data[i + y * nx] = (data[i + y * nx] - mean) / stddev;
        transpose[i * nn + y] = data[i + y * nx];
    }

}



__global__ void corr_kernel(int nn, int ny, int nx, const float* transpose, float* result) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;
    
    if (jc > ic) return;

    __shared__ float xx[4][64];
    __shared__ float yy[4][64];
    
    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = 0.0f;
        }
    }
    
    for (int ks = 0; ks < nx; ks += 4) {
        int ija = ja * 8 + ia;
        
        int i = ic * 64 + ija;
        int j = jc * 64 + ija;
        
        for (int f = 0; f < 4; ++f) {
            int k = ks + f;
            if (k < nx) {
                if (i < ny) {
                    xx[f][ija] = transpose[nn*k + i];
                }
                if (j < ny) {
                    yy[f][ija] = transpose[nn*k + j];
                }
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int f = 0; f < 4; ++f) {
            if (ks + f >= nx) break;
            
            float y[8];
            for (int jb = 0; jb < 8; ++jb) {
                y[jb] = yy[f][jb * 8 + ja];
            }
            
            for (int ib = 0; ib < 8; ++ib) {
                float x = xx[f][ib * 8 + ia];
                for (int jb = 0; jb < 8; ++jb) {
                    v[ib][jb] += x * y[jb];
                }
            }
        }
        __syncthreads();
    }
    
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j < ny && j <= i) {
                result[j * ny + i] = v[ib][jb];
            }
        }
    }
}


static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}


void correlate(int ny, int nx, const float *data, float *result) {
    float* d_GPU;
    float* t_GPU;
    float* r_GPU = NULL;
    
    int nn = roundup(ny, 64);
    
    CHECK(cudaMalloc(&d_GPU, nx * ny * sizeof(float)));
    CHECK(cudaMemcpy(d_GPU, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&t_GPU, nx * nn * sizeof(float)));
    CHECK(cudaMalloc((void**)&r_GPU, ny * ny * sizeof(float)));
    CHECK(cudaMemset(r_GPU, 0, ny * ny * sizeof(float)));
    
    dim3 preBlock(64);
    dim3 preGrid(divup(ny, 64));
    preprocess<<<preGrid, preBlock>>>(ny, nx, nn, d_GPU, t_GPU);
    CHECK(cudaGetLastError());
    
    dim3 dimBlock(8, 8);
    dim3 dimGrid(divup(ny, 64), divup(ny, 64));
    corr_kernel<<<dimGrid, dimBlock>>>(nn, ny, nx, t_GPU, r_GPU);
    CHECK(cudaGetLastError());
    
    CHECK(cudaMemcpy(result, r_GPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_GPU));
    CHECK(cudaFree(t_GPU));
    CHECK(cudaFree(r_GPU));
}
