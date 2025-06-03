/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

#include <vector>
#include <algorithm>

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
  for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
          std::vector<float> window;

          int y_start = std::max(0, y - hy);
          int y_end   = std::min(ny, y + hy + 1);
          int x_start = std::max(0, x - hx);
          int x_end   = std::min(nx, x + hx + 1);

          for (int j = y_start; j < y_end; ++j) {
              for (int i = x_start; i < x_end; ++i) {
                  window.push_back(in[i + j * nx]);
              }
          }

          size_t mid = window.size() / 2;
          std::nth_element(window.begin(), window.begin() + mid, window.end());
          float median = window[mid];

          if (window.size() % 2 == 0) {
              std::nth_element(window.begin(), window.begin() + mid - 1, window.end());
              median = 0.5f * (median + window[mid - 1]);
          }

          out[x + y * nx] = median;
      }
  }
}