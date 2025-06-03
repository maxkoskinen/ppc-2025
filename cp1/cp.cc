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
    
    std::vector<double> means(ny);
    std::vector<double> stddevs(ny);

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
            double sum = 0.0;
            for (int k = 0; k < nx; ++k) {
                double a = data[k + i * nx] - means[i];
                double b = data[k + j * nx] - means[j];
                sum += a * b;
            }
            double denom = stddevs[i] * stddevs[j];
            float corr = (denom != 0.0) ? (float)(sum / denom) : 0.0f;
            result[i + j * ny] = corr;
        }
    }
}