#include <algorithm>
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

__host__ __device__ static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

typedef unsigned long long data_t;

#define RADIX_BITS 8
#define NUM_BUCKETS (1 << RADIX_BITS)  
#define THREADS_PER_BLOCK 256

__global__ void radix_histogram_kernel(const data_t* d_in_data, int n,
                                       int bit_offset,
                                       unsigned int* d_block_histograms,
                                       int tile_size) {
    extern __shared__ unsigned int s_hist[];

    unsigned int tid = threadIdx.x;
    unsigned int tile_idx = blockIdx.x;
    unsigned int threads_in_block = blockDim.x;

    for (unsigned int i = tid; i < NUM_BUCKETS; i += threads_in_block) {
        s_hist[i] = 0;
    }
    __syncthreads();

    int start_idx = tile_idx * tile_size;
    int end_idx = min(start_idx + tile_size, n);

    for (int i = start_idx + tid; i < end_idx; i += threads_in_block) {
        data_t val = d_in_data[i];
        unsigned int bucket = (val >> bit_offset) & (NUM_BUCKETS - 1);
        atomicAdd(&s_hist[bucket], 1);
    }
    __syncthreads();

    for (unsigned int i = tid; i < NUM_BUCKETS; i += threads_in_block) {
        d_block_histograms[tile_idx * NUM_BUCKETS + i] = s_hist[i];
    }
}

__global__ void sum_block_histograms_kernel(unsigned int* d_global_hist,
                                            const unsigned int* d_block_histograms,
                                            int num_tiles) {
    unsigned int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bucket_idx < NUM_BUCKETS) {
        unsigned int sum = 0;
        for (int i = 0; i < num_tiles; ++i) {
            sum += d_block_histograms[i * NUM_BUCKETS + bucket_idx];
        }
        d_global_hist[bucket_idx] = sum;
    }
}

__global__ void exclusive_scan_kernel(unsigned int* d_data) {
    __shared__ unsigned int temp[NUM_BUCKETS];
    int tid = threadIdx.x;

    if (tid < NUM_BUCKETS) {
        temp[tid] = d_data[tid];
    }
    __syncthreads();

    for (int offset = 1; offset < NUM_BUCKETS; offset *= 2) {
        unsigned int val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp[tid] += val;
        }
        __syncthreads();
    }

    if (tid < NUM_BUCKETS) {
        d_data[tid] = (tid == 0) ? 0 : temp[tid - 1];
    }
}

__global__ void calculate_tile_bucket_write_offsets_kernel(
    const unsigned int* d_block_histograms,
    unsigned int* d_tile_bucket_write_offsets,
    int num_tiles) {
    int bucket_idx = blockIdx.x;

    unsigned int prefix_sum = 0;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        d_tile_bucket_write_offsets[tile_idx * NUM_BUCKETS + bucket_idx] = prefix_sum;
        prefix_sum += d_block_histograms[tile_idx * NUM_BUCKETS + bucket_idx];
    }
}

__global__ void radix_reorder_kernel(data_t* d_out_data, const data_t* d_in_data, int n,
                                        int bit_offset,
                                        const unsigned int* d_global_offsets,
                                        const unsigned int* d_tile_bucket_write_offsets,
                                        int tile_size) {
    extern __shared__ char s_mem[];
    data_t* s_data = (data_t*)s_mem;
    unsigned int* s_keys = (unsigned int*)(s_data + tile_size);

    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;
    int start_idx = tile_idx * tile_size;
    int end_idx = min(start_idx + tile_size, n);
    int num_elements = end_idx - start_idx;

    if (tid < num_elements) {
        data_t val = d_in_data[start_idx + tid];
        s_data[tid] = val;
        s_keys[tid] = (val >> bit_offset) & (NUM_BUCKETS - 1);
    }
    __syncthreads();

    if (tid < num_elements) {
        unsigned int my_key = s_keys[tid];
        unsigned int local_rank = 0;
        for (int i = 0; i < tid; ++i) {
            if (s_keys[i] == my_key) {
                local_rank++;
            }
        }

        unsigned int offset = d_tile_bucket_write_offsets[tile_idx * NUM_BUCKETS + my_key];
        unsigned int out_data_idx = d_global_offsets[my_key] + offset + local_rank;
        d_out_data[out_data_idx] = s_data[tid];
    }
}



void psort(int n, data_t *data) {
    if (n <= 1) return;

    data_t *d_data, *d_buffer;
    unsigned int *d_block_histograms, *d_global_histogram, *d_bucket_offsets;
    unsigned int *d_tile_bucket_write_offsets;

    CHECK(cudaMalloc(&d_data, n * sizeof(data_t)));
    CHECK(cudaMalloc(&d_buffer, n * sizeof(data_t)));

    int tile_size = THREADS_PER_BLOCK;
    int num_tiles = divup(n, tile_size);

    CHECK(cudaMalloc(&d_block_histograms, num_tiles * NUM_BUCKETS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_global_histogram, NUM_BUCKETS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_bucket_offsets, NUM_BUCKETS * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_tile_bucket_write_offsets, num_tiles * NUM_BUCKETS * sizeof(unsigned int)));

    CHECK(cudaMemcpy(d_data, data, n * sizeof(data_t), cudaMemcpyHostToDevice));

    data_t *d_current_in = d_data;
    data_t *d_current_out = d_buffer;

    int num_total_bits = sizeof(data_t) * 8;

    for (int bit_offset = 0; bit_offset < num_total_bits; bit_offset += RADIX_BITS) {
        size_t shared_mem_size = NUM_BUCKETS * sizeof(unsigned int);
        radix_histogram_kernel<<<num_tiles, THREADS_PER_BLOCK, shared_mem_size>>>(d_current_in, n, bit_offset, d_block_histograms, tile_size);
        CHECK(cudaGetLastError());

        int sum_threads = THREADS_PER_BLOCK;
        int sum_blocks = divup(NUM_BUCKETS, sum_threads);
        sum_block_histograms_kernel<<<sum_blocks, sum_threads>>>(d_global_histogram, d_block_histograms, num_tiles);
        CHECK(cudaGetLastError());

        exclusive_scan_kernel<<<1, NUM_BUCKETS>>>(d_global_histogram);
        CHECK(cudaGetLastError());

        calculate_tile_bucket_write_offsets_kernel<<<NUM_BUCKETS, 1>>>(d_block_histograms, d_tile_bucket_write_offsets, num_tiles);
        CHECK(cudaGetLastError());

        size_t reorder_shared_mem = tile_size * sizeof(data_t) + tile_size * sizeof(unsigned int);
        radix_reorder_kernel<<<num_tiles, THREADS_PER_BLOCK, reorder_shared_mem>>>(
            d_current_out, d_current_in, n, bit_offset,
            d_global_histogram, d_tile_bucket_write_offsets, tile_size);
        CHECK(cudaGetLastError());

        CHECK(cudaDeviceSynchronize());

        std::swap(d_current_in, d_current_out);
    }

    CHECK(cudaMemcpy(data, d_current_in, n * sizeof(data_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_buffer));
    CHECK(cudaFree(d_block_histograms));
    CHECK(cudaFree(d_global_histogram));
    CHECK(cudaFree(d_bucket_offsets));
    CHECK(cudaFree(d_tile_bucket_write_offsets));
}