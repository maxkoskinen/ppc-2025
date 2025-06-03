#include "pngio.h"
#include <cstdlib>
#include <iostream>

#include "is.h"
#include "tests.h"

inline bool boundary(int v, int a, int b) {
    return v == a || v == b - 1;
}

bool at_boundary(const Result &r, int y, int x, int d) {
    if (boundary(y, r.y0 - d, r.y1 + d)) {
        return inside(x, r.x0 - d, r.x1 + d);
    } else if (boundary(x, r.x0 - d, r.x1 + d)) {
        return inside(y, r.y0 - d, r.y1 + d);
    } else {
        return false;
    }
}

bool at_inner_boundary(const Result &r, int y, int x) {
    return at_boundary(r, y, x, 0);
}

bool at_outer_boundary(const Result &r, int y, int x) {
    return at_boundary(r, y, x, 1);
}

ppc::Image8 visualize(TestCaseInstance &instance) {
    int ny = instance.Ny;
    int nx = instance.Nx;
    Result &r = instance.Expected;

    ppc::Image8 result;
    result.resize(ny, nx * 3 + 4, 3);

    // draw separating lines
    for (int y = 0; y < ny; ++y) {
        result.setlin3(y, nx + 1, 1.0, 1.0, 1.0);
        result.setlin3(y, 2 * nx + 3, 1.0, 1.0, 1.0);
    }

    find_avgs(ny, nx, instance.Data.data(), r);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            for (int c = 0; c < 3; ++c) {
                result.setlin(y, x, c, instance.Data[c + 3 * (x + nx * y)]);
            }

            if (is_inside(r, y, x)) {
                result.setlin3(y, x + nx + 2, r.inner[0], r.inner[1], r.inner[2]);
            } else {
                result.setlin3(y, x + nx + 2, r.outer[0], r.outer[1], r.outer[2]);
            }

            if (at_inner_boundary(r, y, x)) {
                result.setlin3(y, x + 2 * nx + 4, 1.0f, 1.0f, 1.0f);
            } else if (at_outer_boundary(r, y, x)) {
                result.setlin3(y, x + 2 * nx + 4, 0.0f, 0.0f, 0.0f);
            } else {
                for (int c = 0; c < 3; ++c) {
                    result.setlin(y, x + 2 * nx + 4, c, instance.Data[c + 3 * (x + nx * y)]);
                }
            }
        }
    }
    return result;
}

ppc::Image8 handle_test_case(const char *testfile) {
    std::ifstream input_file(testfile);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        std::exit(1);
    }

    std::string input_type;
    CHECK_READ(input_file >> input_type);
    if (input_type == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        CHECK_READ(input_file >> input_type);
    }

    auto test_case = make_test_case(input_type, input_file);
    TestCaseInstance instance = test_case->generate();
    instance.Expected = segment(instance.Ny, instance.Nx, instance.Data.data());
    return visualize(instance);
}

ppc::Image8 handle_image(const char *testfile) {
    ppc::Image8 im_input;
    ppc::read_image(im_input, testfile);

    if (im_input.nc != 3) {
        std::cerr << "Image " << testfile << " should have 3 color components." << std::endl;
        std::exit(1);
    }

    TestCaseInstance instance;
    instance.Nx = im_input.nx;
    instance.Ny = im_input.ny;
    instance.Data.resize(im_input.data.size());
    std::transform(im_input.data.begin(), im_input.data.end(), instance.Data.begin(),
                   [](unsigned char cc) {
                       return static_cast<float>(cc) / 255.f;
                   });

    instance.Expected = segment(instance.Ny, instance.Nx, instance.Data.data());
    return visualize(instance);
}

void help(const char *prog) {
    std::cout << "usage: " << prog << " test.txt [output]" << std::endl;
    std::cout << "       " << prog << " test.png [output]" << std::endl;
    std::cout << "  If test is a png image, run the segmentation on that image." << std::endl;
    std::cout << "  If test is a test case, generate the image and expected output." << std::endl;
    std::exit(1);
}

int main(int argc, const char **argv) {
    if (argc != 2 && argc != 3) {
        help(argv[0]);
    }
    const char *testfile = argv[1];
    std::size_t len = strlen(testfile);
    if (len < 4) {
        std::cerr << "Input file name too short. Please use either a '*.txt' or a '*.png' file." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const char *imfile_output = "out.png";
    if (argc == 3) {
        imfile_output = argv[2];
    }

    auto result = [&]() {
        if (strcmp(testfile + len - 4, ".png") == 0) {
            return handle_image(testfile);
        } else if (strcmp(testfile + len - 4, ".txt") == 0) {
            return handle_test_case(testfile);
        } else {
            help(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }();

    write_image(result, imfile_output);

    std::cout << "Generated result '" << imfile_output << "'\n";

    return EXIT_SUCCESS;
}
