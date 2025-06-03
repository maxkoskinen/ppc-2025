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

__global__ void corr_kernel(const float* norm_data, float* result, int ny, int nx) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= ny || j > i) return;

    float sum = 0.0;
    for (int k = 0; k < nx; ++k) {
        float a = norm_data[i * nx + k];
        float b = norm_data[j * nx + k];
        sum += a * b;
    }

    result[i + j * ny] = sum;
}

double r_mean(const float* row, int nx) {
    float sum = 0;

    for (int j = 0; j < nx; ++j) {
        sum += row[j];
    }
    return sum / nx;
}

double r_stddev(const float* row, int nx, float mean) {
    float sq_sum = 0;

    for (int j = 0; j < nx; ++j) {
        double diff = row[j] - mean;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum);
}

void correlate(int ny, int nx, const float *data, float *result) {

    std::vector<float> nd(ny * nx, 0.0f);

    for(int i = 0; i < ny; ++i) {
        const float* row = &data[i * nx];
        float mean = r_mean(row, nx);
        float stddev = r_stddev(row, nx, mean);

        for (int j = 0; j < nx; ++j) {
            nd[i * nx + j] = (row[j] - mean) / stddev;
        }
    }

    float *nd_G = NULL, *res_G = NULL;
    CHECK(cudaMalloc((void**)&nd_G, ny * nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&res_G, ny * ny * sizeof(float)));
    CHECK(cudaMemset(res_G, 0, ny * ny * sizeof(float)));

    CHECK(cudaMemcpy(nd_G, nd.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((ny + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y);

    corr_kernel<<<gridSize, blockSize>>>(nd_G, res_G, ny, nx);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, res_G, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(nd_G));
    CHECK(cudaFree(res_G));

}
