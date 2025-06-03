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

void correlate(int ny, int nx, const float *data, float *result) {

    std::vector<double> means(ny, (double)0.0);
    std::vector<double> stddevs(ny, (double)0.0);
    
    #pragma omp parallel for
    for (int i = 0; i < ny; ++i) {
        double sum = 0.0;
        for (int k = 0; k < nx; ++k) {
            sum += data[k + i * nx];
        }
        double mean = sum / nx;
        means[i] = mean;

        double sum_sq = 0.0;
        for (int k = 0; k < nx; ++k) {
            double diff = data[k + i * nx] - mean;
            sum_sq += diff * diff;
        }
        stddevs[i] = std::sqrt(sum_sq);
    }


    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum0 = 0.0;
            double mean_i = means[i];
            double mean_j = means[j];
            
            for (int k = 0; k < nx; ++k) {
                double a0 = data[k + i * nx] - mean_i; 
                double b0 = data[k + j * nx] - mean_j; 

                sum0 += a0 * b0;

            }

            double denom = stddevs[i] * stddevs[j];
            float corr = (denom != 0.0) ? static_cast<float>(sum0 / denom) : 0.0f;
            result[i + j * ny] = corr;
        }
    }

}
