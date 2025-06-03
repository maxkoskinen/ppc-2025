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


typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));



void correlate(int ny, int nx, const float *data, float *result) {
    int nv = 4;
    int nvec = (nx + nv - 1) / 4;

    std::vector<double4_t> vectors(ny * nvec);
    std::vector<double> stddevs(ny);

    for (int i = 0; i < ny; ++i) {
        double sum = 0.0;
        for (int k = 0; k < nx; ++k) {
            sum += data[k + i * nx];
        }
        double mean = sum / nx;

        double sum_sq = 0.0;
        for (int k = 0; k < nx; ++k) {
            double diff = data[k + i * nx] - mean;
            sum_sq += diff * diff;
        }
        double stddev = std::sqrt(sum_sq);
        stddevs[i] = stddev;

        for (int v = 0; v < nvec; ++v) {
            double4_t vec = {0.0, 0.0, 0.0, 0.0};
            for (int l = 0; l < nv; ++l) {
                int idx = v * nv + l;
                if (idx < nx) {
                    vec[l] = (double)data[idx + i * nx] - mean;
                }
            }
            vectors[i * nvec + v] = vec;
        }
    }

    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j <= i; ++j) {
            double4_t sum_vec = {0.0, 0.0, 0.0, 0.0};
            for (int v = 0; v < nvec; ++v) {
                double4_t a = vectors[i * nvec + v];
                double4_t b = vectors[j * nvec + v];
                sum_vec += a * b;
            }

            double sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
            double denom = stddevs[i] * stddevs[j];
            float corr = (denom != 0.0) ? (float)(sum / denom) : 0.0f;
            result[i + j * ny] = corr;
        }
    }
}