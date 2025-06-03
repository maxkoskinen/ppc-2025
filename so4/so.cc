#include <algorithm>
#include <vector>
#include <omp.h>
#include <cstring>

const int MIN_PARALLEL_N = 65536;

typedef unsigned long long data_t;

void merge(data_t* data, int start, int mid, int end, data_t* buffer) {
    int i = start;
    int j = mid;
    int k = start;
    
    while (i < mid && j < end) {
        if (data[i] <= data[j]) {
            buffer[k++] = data[i++];
        } else {
            buffer[k++] = data[j++];
        }
    }

    while (i < mid) {
        buffer[k++] = data[i++];
    }
    
    while (j < end) {
        buffer[k++] = data[j++];
    }
    
    std::memcpy(data + start, buffer + start, (end - start) * sizeof(data_t));
}


void parallel_merge_sort(data_t* data, int n, int num_threads) {
    data_t* buffer = new data_t[n];
    
    int chunk_size = (n + num_threads - 1) / num_threads;
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < num_threads; i++) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, n);
            if (start < n) {
                std::sort(data + start, data + end);
            }
        }
    }
    
    for (int size = chunk_size; size < n; size *= 2) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i += 2 * size) {
            int start = i;
            int mid = std::min(i + size, n);
            int end = std::min(i + 2 * size, n);
            if (mid < end) {
                merge(data, start, mid, end, buffer);
            }
        }
    }
    
    delete[] buffer;
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    //std::sort(data, data + n);

    if (n <= 1) return;
    
    int num_threads = omp_get_max_threads();
    parallel_merge_sort(data, n, num_threads);
}