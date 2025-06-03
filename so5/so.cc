#include <algorithm>
#include <omp.h>
#include <vector>
#include <math.h>

typedef unsigned long long data_t;

const int MIN_PAR = 10000;
const int CUTOFF = 10000;


int partition(data_t* data, int low, int high) {
    int mid = low + (high - low) / 2;
    
    if (data[low] > data[mid])
        std::swap(data[low], data[mid]);
    if (data[low] > data[high])
        std::swap(data[low], data[high]);
    if (data[mid] > data[high])
        std::swap(data[mid], data[high]);
    
    std::swap(data[mid], data[high - 1]);
    data_t pivot = data[high - 1];

    int i = low;
    int j = high - 1;
    
    while (true) {
        while (data[++i] < pivot) {
            if (i >= high - 1) break;
        }
        
        while (data[--j] > pivot) {
            if (j <= low) break;
        }
        
        if (i >= j) break;
        std::swap(data[i], data[j]);
    }
    
    std::swap(data[i], data[high - 1]);
    return i;
}


void s_quicksort(data_t* data, int low, int high) {
    if (high - low < 8) {
        std::sort(data + low, data + high + 1);
        return;
    }
    
    if (low < high) {
        int pivot_idx = partition(data, low, high);
        s_quicksort(data, low, pivot_idx - 1);
        s_quicksort(data, pivot_idx + 1, high);
    }
}

void p_quicksort(data_t* data, int low, int high, int depth) {
    if (high - low < MIN_PAR) {
        std::sort(data + low, data + high + 1);
        return;
    }
    
    int pivot_idx = partition(data, low, high);
    
    if ((high - low) > CUTOFF && depth > 0) {
        #pragma omp task default(none) shared(data) firstprivate(low, pivot_idx, depth)
        p_quicksort(data, low, pivot_idx - 1, depth - 1);
        
        #pragma omp task default(none) shared(data) firstprivate(pivot_idx, high, depth)
        p_quicksort(data, pivot_idx + 1, high, depth - 1);
        
        #pragma omp taskwait
    } else {
        s_quicksort(data, low, pivot_idx - 1);
        s_quicksort(data, pivot_idx + 1, high);
    }
}


void psort(int n, data_t *data) {
    if (n <= 1) return;

    int num_threads = omp_get_max_threads();
    int depth_limit = (n + num_threads - 1) / num_threads;;
    
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            p_quicksort(data, 0, n - 1, depth_limit);
        }
    }
    
}