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

void correlate(int ny, int nx, const float *data, float *result) {

    std::vector<double> means(ny, (double)0.0);
    std::vector<double> stddevs(ny, (double)0.0);
    
    for (int i = 0; i < ny; ++i) {
        double sum = 0.0;
        for (int k = 0; k < nx; ++k) {
            sum += data[k + i * nx];
        }
        means[i] = sum / nx;
    }

    for (int i = 0; i < ny; ++i) {
        double sum_sq = 0.0;
        for (int k = 0; k < nx; ++k) {
            double diff = data[k + i * nx] - means[i];
            sum_sq += diff * diff;
        }
        stddevs[i] = std::sqrt(sum_sq);
    }
 
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            double mean_i = means[i];
            double mean_j = means[j];
            int k = 0;
            for (; k + 4 < nx; k += 4) {
                double a0 = data[k + i * nx] - mean_i; 
                double b0 = data[k + j * nx] - mean_j; 

                double a1 = data[k + 1 + i * nx] - mean_i; 
                double b1 = data[k + 1 + j * nx] - mean_j; 

                double a00 = data[k + 2 + i * nx] - mean_i; 
                double b00 = data[k + 2 + j * nx] - mean_j; 

                double a11 = data[k + 3 + i * nx] - mean_i; 
                double b11 = data[k + 3 + j * nx] - mean_j; 

                sum0 += a0 * b0;
                sum1 += a1 * b1;
                sum2 += a00 * b00;
                sum3 += a11 * b11;
            }
            for (; k < nx; ++k) {
                double a = data[k + i * nx] - mean_i;
                double b = data[k + j * nx] - mean_j;
                sum0 += a * b;
            }

            double denom = stddevs[i] * stddevs[j];
            float corr = (denom != 0.0) ? (float)((sum0 + sum1 + sum2 + sum3) / denom) : 0.0f;
            result[i + j * ny] = corr;
        }
    }

}
