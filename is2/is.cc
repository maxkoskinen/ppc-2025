struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

#include <cmath>
#include <vector>


void compute_integral(int ny, int nx, const float* data,
        std::vector<std::vector<std::vector<double>>> &sum,
        std::vector<std::vector<std::vector<double>>> &sum_sq) {
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                double pix = static_cast<double>(data[c + 3 * x + 3 * nx * y]);
                double pix_sq = pix * pix;

                sum[c][y + 1][x + 1] = pix
                                    + sum[c][y + 1][x]
                                    + sum[c][y][x + 1]
                                    - sum[c][y][x];

                sum_sq[c][y + 1][x + 1] = pix_sq
                                        + sum_sq[c][y + 1][x]
                                        + sum_sq[c][y][x + 1]
                                        - sum_sq[c][y][x];
            }
        }
    }
}




Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    std::vector<std::vector<std::vector<double>>> sum(3, std::vector<std::vector<double>>(ny + 1, std::vector<double>(nx + 1, 0.0)));
    std::vector<std::vector<std::vector<double>>> sum_sq(3, std::vector<std::vector<double>>(ny + 1, std::vector<double>(nx + 1, 0.0)));

    compute_integral(ny, nx, data, sum, sum_sq);

    double min_cost = std::numeric_limits<double>::infinity();
    int total_pix = ny * nx;

    for (int y0 = 0; y0 < ny; ++y0) {
        for (int y1 = y0 + 1; y1 <= ny; ++y1) {
            for (int x0 = 0; x0 < nx; ++x0) {
                for (int x1 = x0 + 1; x1 <= nx; ++x1) {
                    int rect_area = (y1 - y0) * (x1 - x0);
                    int outer_area = total_pix - rect_area;
                    if (rect_area <= 0 || outer_area <= 0) continue;

                    double inner[3] = {0.0, 0.0, 0.0}, outer[3] = {0.0, 0.0, 0.0}, cost = 0.0;

                    for (int c = 0; c < 3; ++c) {
                        double in_sum = sum[c][y1][x1] - sum[c][y1][x0] - sum[c][y0][x1] + sum[c][y0][x0];
                        double in_sum_sq = sum_sq[c][y1][x1] - sum_sq[c][y1][x0] - sum_sq[c][y0][x1] + sum_sq[c][y0][x0];

                        double total_sum = sum[c][ny][nx];
                        double total_sum_sq = sum_sq[c][ny][nx];

                        double out_sum = total_sum - in_sum;
                        double out_sum_sq = total_sum_sq - in_sum_sq;

                        inner[c] = in_sum / rect_area;
                        outer[c] = out_sum / outer_area;

                        cost += in_sum_sq - rect_area * inner[c] * inner[c];
                        cost += out_sum_sq - outer_area * outer[c] * outer[c];
                    }

                    if (cost < min_cost) {
                        min_cost = cost;
                        result.y0 = y0;
                        result.y1 = y1;
                        result.x0 = x0;
                        result.x1 = x1;
                        for (int c = 0; c < 3; ++c) {
                            result.inner[c] = inner[c];
                            result.outer[c] = outer[c];
                        }
                    }
                }
            }
        }
    }

    return result;
}
