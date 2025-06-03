#include <algorithm> 
#include <vector>    
#include <cstdio>    
#include <cstdlib>   
#include <cuda_runtime.h>

typedef unsigned long long data_t; 

static const int RDX_BITS = 8;
static const int RDX_BKTS = 1 << RDX_BITS; 

#ifndef CUDA_CHECK_PSORT 
#define CUDA_CHECK_PSORT(err)                                     \
    do {                                                          \
        cudaError_t err_ = (err);                                 \
        if (err_ != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in psort: %s at %s:%d\n", \
                    cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)
#endif

__global__ void rdx_block_hist_k_tiled(
    const data_t*__restrict__ din_global, 
    int N_total, 
    int bs, 
    unsigned int*__restrict__ d_block_hists,
    int tile_size 
) {
    extern __shared__ unsigned int slh[]; 
    int tid = threadIdx.x;
    int bth = blockDim.x; 
    
    for (int i = tid; i < RDX_BKTS; i += bth) {
        slh[i] = 0;
    }
    __syncthreads(); 
    
    int current_tile_idx = blockIdx.x;
    int tile_data_start_idx = current_tile_idx * tile_size;
    int item_idx_in_tile = tid;
    int global_item_idx = tile_data_start_idx + item_idx_in_tile;

    if (global_item_idx < N_total) {
        data_t v = din_global[global_item_idx];
        unsigned int k = (v >> bs) & (RDX_BKTS - 1);
        atomicAdd(&slh[k], 1); 
    }
    __syncthreads(); 
    
    unsigned int* my_block_hist_output_ptr = d_block_hists + current_tile_idx * RDX_BKTS;
    for (int i = tid; i < RDX_BKTS; i += bth) {
        my_block_hist_output_ptr[i] = slh[i];
    }
}

__global__ void aggregate_block_hists_to_global_hist(
    const unsigned int*__restrict__ d_block_hists, 
    unsigned int*__restrict__ d_global_hist,       
    int num_tiles
) {
    int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bucket_idx < RDX_BKTS) {
        unsigned int sum_for_bucket = 0;
        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            sum_for_bucket += d_block_hists[tile_idx * RDX_BKTS + bucket_idx];
        }
        d_global_hist[bucket_idx] = sum_for_bucket;
    }
}

__global__ void exclusive_scan_rdx_bkts(
    unsigned int*__restrict__ d_data, 
                                      
    unsigned int*__restrict__ d_scratch_for_sum 
) {
    extern __shared__ unsigned int s_scan_data[];
    int tid = threadIdx.x;

    if (tid < RDX_BKTS) {
        s_scan_data[tid] = d_data[tid];
    }
    __syncthreads();

    for (int offset = 1; offset < RDX_BKTS; offset *= 2) {
        unsigned int val = 0;
        if (tid >= offset) {
            val = s_scan_data[tid - offset];
        }
        __syncthreads(); 
        if (tid >= offset) {
            s_scan_data[tid] += val;
        }
        __syncthreads(); 
    }

    if (tid < RDX_BKTS) {
        d_data[tid] = (tid == 0) ? 0 : s_scan_data[tid - 1];
    }
}


__global__ void calculate_tile_bucket_write_offsets_gpu(
    const unsigned int*__restrict__ d_block_hists,               
    unsigned int*__restrict__ d_tile_bucket_write_offsets, 
    int num_tiles
) {
    int bucket_idx = blockIdx.x; 
        unsigned int prefix_sum_for_this_bucket = 0;
        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            d_tile_bucket_write_offsets[tile_idx * RDX_BKTS + bucket_idx] = prefix_sum_for_this_bucket;
            prefix_sum_for_this_bucket += d_block_hists[tile_idx * RDX_BKTS + bucket_idx];
        }
}

__global__ void rdx_scat_k_stable_tiled(
    const data_t*__restrict__ din_global, 
    int N_total, 
    int bs, 
    const unsigned int*__restrict__ global_bucket_starts,       
    const unsigned int*__restrict__ tile_bucket_write_offsets,  
    data_t*__restrict__ dout_global,
    int tile_size 
) {
    extern __shared__ char s_mem_extern[]; 
    data_t* s_data = (data_t*)s_mem_extern; 
    unsigned int* s_keys = (unsigned int*)(s_data + tile_size); 

    int current_tile_idx = blockIdx.x;
    int tile_data_start_idx = current_tile_idx * tile_size;
    int item_idx_in_tile = threadIdx.x;
    int global_item_idx = tile_data_start_idx + item_idx_in_tile;

    int num_elements_in_this_tile = tile_size;
    if (current_tile_idx == gridDim.x - 1) { 
        num_elements_in_this_tile = N_total - tile_data_start_idx;
        if (num_elements_in_this_tile < 0) num_elements_in_this_tile = 0; 
    }
    
    if (item_idx_in_tile < num_elements_in_this_tile) { 
        s_data[item_idx_in_tile] = din_global[global_item_idx];
        s_keys[item_idx_in_tile] = (s_data[item_idx_in_tile] >> bs) & (RDX_BKTS - 1);
    }
    __syncthreads(); 

    if (item_idx_in_tile < num_elements_in_this_tile) {
        unsigned int my_key = s_keys[item_idx_in_tile];
        data_t my_val = s_data[item_idx_in_tile];

        unsigned int local_rank_in_tile = 0;
        for (int j = 0; j < item_idx_in_tile; ++j) {
            if (s_keys[j] == my_key) {
                local_rank_in_tile++;
            }
        }
        
        unsigned int offset_for_this_tile_in_bucket_k = tile_bucket_write_offsets[current_tile_idx * RDX_BKTS + my_key];
        
        unsigned int dest_idx = global_bucket_starts[my_key] + offset_for_this_tile_in_bucket_k + local_rank_in_tile;
        dout_global[dest_idx] = my_val;
    }
}


void psort(int n_items, data_t* d_ptr) {
    if (n_items <= 1) return;

    data_t *d_s, *d_d;         
    unsigned int *d_global_hist_gpu; 
    unsigned int *d_tbo_gpu;         

    unsigned int *d_block_hists_gpu;                
    unsigned int *d_tile_bucket_write_offsets_gpu;  

    CUDA_CHECK_PSORT(cudaMalloc(&d_s, (size_t)n_items * sizeof(data_t)));
    CUDA_CHECK_PSORT(cudaMalloc(&d_d, (size_t)n_items * sizeof(data_t)));
    
    CUDA_CHECK_PSORT(cudaMalloc(&d_global_hist_gpu, (size_t)RDX_BKTS * sizeof(unsigned int)));
    CUDA_CHECK_PSORT(cudaMalloc(&d_tbo_gpu, (size_t)RDX_BKTS * sizeof(unsigned int)));


    CUDA_CHECK_PSORT(cudaMemcpy(d_s, d_ptr, (size_t)n_items * sizeof(data_t), cudaMemcpyHostToDevice));

    data_t *d_ci = d_s; 
    data_t *d_co = d_d; 

    int n_passes = (sizeof(data_t) * 8) / RDX_BITS; 
    
    int tpb = 256; 
    
    int num_tiles = (n_items + tpb - 1) / tpb;

    CUDA_CHECK_PSORT(cudaMalloc(&d_block_hists_gpu, (size_t)num_tiles * RDX_BKTS * sizeof(unsigned int)));
    CUDA_CHECK_PSORT(cudaMalloc(&d_tile_bucket_write_offsets_gpu, (size_t)num_tiles * RDX_BKTS * sizeof(unsigned int)));

    size_t hist_kernel_shared_mem = (size_t)RDX_BKTS * sizeof(unsigned int); 
    size_t scan_kernel_shared_mem = (size_t)RDX_BKTS * sizeof(unsigned int);
    size_t scatter_kernel_shared_mem = (size_t)tpb * sizeof(data_t) + (size_t)tpb * sizeof(unsigned int);


    for (int p_idx = 0; p_idx < n_passes; ++p_idx) {
        int bs = p_idx * RDX_BITS; 

        rdx_block_hist_k_tiled<<<num_tiles, tpb, hist_kernel_shared_mem>>>(d_ci, n_items, bs, d_block_hists_gpu, tpb);
        CUDA_CHECK_PSORT(cudaGetLastError()); 
        int agg_threads = RDX_BKTS; 
        int agg_blocks = 1;
        if (RDX_BKTS > 1024) { 
             agg_threads = 1024;
             agg_blocks = (RDX_BKTS + agg_threads -1) / agg_threads;
        }
        CUDA_CHECK_PSORT(cudaMemset(d_global_hist_gpu, 0, (size_t)RDX_BKTS * sizeof(unsigned int))); // Zero out before sum
        aggregate_block_hists_to_global_hist<<<agg_blocks, agg_threads>>>(d_block_hists_gpu, d_global_hist_gpu, num_tiles);
        CUDA_CHECK_PSORT(cudaGetLastError());

        CUDA_CHECK_PSORT(cudaMemcpy(d_tbo_gpu, d_global_hist_gpu, (size_t)RDX_BKTS * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
        exclusive_scan_rdx_bkts<<<1, tpb, scan_kernel_shared_mem>>>(d_tbo_gpu, nullptr); 
        CUDA_CHECK_PSORT(cudaGetLastError());

        int tbwo_blocks = RDX_BKTS;
        int tbwo_threads = 1; 
        calculate_tile_bucket_write_offsets_gpu<<<tbwo_blocks, tbwo_threads>>>(d_block_hists_gpu, d_tile_bucket_write_offsets_gpu, num_tiles);
        CUDA_CHECK_PSORT(cudaGetLastError());
        
        rdx_scat_k_stable_tiled<<<num_tiles, tpb, scatter_kernel_shared_mem>>>(
            d_ci, n_items, bs, 
            d_tbo_gpu, d_tile_bucket_write_offsets_gpu, 
            d_co, tpb
        );
        CUDA_CHECK_PSORT(cudaGetLastError());
        CUDA_CHECK_PSORT(cudaDeviceSynchronize());

        std::swap(d_ci, d_co); 
    }

    CUDA_CHECK_PSORT(cudaMemcpy(d_ptr, d_ci, (size_t)n_items * sizeof(data_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_PSORT(cudaFree(d_s));
    CUDA_CHECK_PSORT(cudaFree(d_d));
    CUDA_CHECK_PSORT(cudaFree(d_global_hist_gpu));
    CUDA_CHECK_PSORT(cudaFree(d_tbo_gpu));
    CUDA_CHECK_PSORT(cudaFree(d_block_hists_gpu));
    CUDA_CHECK_PSORT(cudaFree(d_tile_bucket_write_offsets_gpu));
}