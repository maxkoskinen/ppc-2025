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
#include <algorithm>
#include <cstring>

// Threshold for switching to standard multiplication
const int STRASSEN_THRESHOLD = 64;

// Helper functions for data normalization
double r_mean(const float* row, int nx) {
    double sum = 0.0;

    #pragma omp simd reduction(+:sum)
    for (int j = 0; j < nx; ++j) {
        sum += static_cast<double>(row[j]);
    }

    return sum / static_cast<double>(nx);
}

double r_std(const float* row, int nx, double mean) {
    double square_sum = 0.0;

    #pragma omp simd reduction(+:square_sum)
    for (int j = 0; j < nx; ++j) {
        double diff = static_cast<double>(row[j]) - mean;
        square_sum += diff * diff;
    }

    return std::sqrt(square_sum);
}


void correlate(int ny, int nx, const float *data, float *result) {
    // Normalize the data
    std::vector<double> normalized_data(ny * nx);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ny; ++i) {
        double mean = r_mean(&data[i * nx], nx);
        double stddev = r_std(&data[i * nx], nx, mean);
        double inv_stddev = 1.0 / stddev;

        for (int j = 0; j < nx; ++j) {
            normalized_data[i * nx + j] = (data[i * nx + j] - mean) * inv_stddev;
        }
    }

    
}