#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Noise Addition
py::bytes add_uniform_noise(const py::bytes &data, int low, int high);
py::bytes add_gaussian_noise(const py::bytes &data, double mean, double stddev);
py::bytes add_salt_pepper_noise(const py::bytes &data, double salt_prob, double pepper_prob);

// Spatial Low-Pass Filters
py::bytes apply_average_filter(const py::bytes &data, int kernel_size);
py::bytes apply_gaussian_filter(const py::bytes &data, int kernel_size);
py::bytes apply_median_filter(const py::bytes &data, int kernel_size);

// Edge Detection
py::bytes apply_canny(const py::bytes &data, double t_low, double t_high, int kernel_size);
py::bytes apply_sobel(const py::bytes &data, int direction);
py::bytes apply_prewitt(const py::bytes &data, int direction);
py::bytes apply_roberts(const py::bytes &data, int direction);

// --- BINDINGS ---
PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "C++ OpenCV Backend";

    // --- Noise Addition ---
    m.def("add_uniform_noise", &add_uniform_noise,
          "Add uniform noise to an image (PNG bytes). "
          "low/high: additive range [-128, 128]",
          py::arg("data"), py::arg("low"), py::arg("high"));

    m.def("add_gaussian_noise", &add_gaussian_noise,
          "Add Gaussian noise to an image (PNG bytes). "
          "mean in [-50,50], stddev in [1,80]",
          py::arg("data"), py::arg("mean"), py::arg("stddev"));

    m.def("add_salt_pepper_noise", &add_salt_pepper_noise,
          "Add salt & pepper noise to an image (PNG bytes). "
          "Probabilities in [0, 0.3]",
          py::arg("data"), py::arg("salt_prob"), py::arg("pepper_prob"));

    // --- Spatial Low-Pass Filters ---
    m.def("apply_average_filter", &apply_average_filter,
          "Apply average (box) filter. kernel_size must be odd, in [3, 21]",
          py::arg("data"), py::arg("kernel_size"));

    m.def("apply_gaussian_filter", &apply_gaussian_filter,
          "Apply Gaussian blur filter. kernel_size must be odd, in [3, 21]",
          py::arg("data"), py::arg("kernel_size"));

    m.def("apply_median_filter", &apply_median_filter,
          "Apply median filter. kernel_size must be odd, in [3, 21]",
          py::arg("data"), py::arg("kernel_size"));

    // --- Edge Detection ---
    m.def("apply_canny", &apply_canny,
          "Canny edge detection. "
          "t_low in [0, 100], t_high in [t_low, 300], kernel_size odd in [3, 7]",
          py::arg("data"), py::arg("t_low"), py::arg("t_high"), py::arg("kernel_size"));
    
    m.def("apply_sobel", &apply_sobel,
          "Sobel edge detection (from scratch). "
          "direction: 0 = X only, 1 = Y only, 2 = Both (magnitude)",
          py::arg("data"), py::arg("direction"));

    m.def("apply_prewitt", &apply_prewitt,
          "Prewitt edge detection (from scratch). "
          "direction: 0 = X only, 1 = Y only, 2 = Both (magnitude)",
          py::arg("data"), py::arg("direction"));
          
    m.def("apply_roberts", &apply_roberts,
          "Roberts edge detection (from scratch, 2x2 kernels). "
          "direction: 0 = X only, 1 = Y only, 2 = Both (magnitude)",
          py::arg("data"), py::arg("direction"));
}
