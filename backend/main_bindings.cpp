#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "image_utils.hpp"

namespace py = pybind11;

/* =========================================================
   FORWARD DECLARATIONS — noise_filters.cpp
   ========================================================= */

py::bytes add_uniform_noise(const py::bytes &data, int low, int high);
py::bytes add_gaussian_noise(const py::bytes &data, double mean, double stddev);
py::bytes add_salt_pepper_noise(const py::bytes &data, double salt_prob, double pepper_prob);

py::bytes apply_average_filter(const py::bytes &data, int kernel_size);
py::bytes apply_gaussian_filter(const py::bytes &data, int kernel_size);
py::bytes apply_median_filter(const py::bytes &data, int kernel_size);

/* =========================================================
   FORWARD DECLARATIONS — edge_freq.cpp
   ========================================================= */

py::bytes apply_canny(const py::bytes &data, double t_low, double t_high, int kernel_size);
py::bytes apply_sobel(const py::bytes &data, int direction);
py::bytes apply_prewitt(const py::bytes &data, int direction);
py::bytes apply_roberts(const py::bytes &data, int direction);

/* =========================================================
   FORWARD DECLARATIONS — hist_contrast.cpp
   ========================================================= */

struct ImageStats {
    float mean;
    float stddev;
    float min_val;
    float max_val;
};

std::vector<float>                           compute_histogram(const py::bytes &data);
std::vector<std::vector<float>>              compute_bgr_histograms(const py::bytes &data);
std::vector<float>                           compute_cdf(const std::vector<float> &hist);
std::vector<float>                           compute_pdf(const std::vector<float> &hist);
std::vector<std::vector<float>>              get_gray_histogram_and_cdf(const py::bytes &data);
std::vector<std::vector<std::vector<float>>> get_bgr_histograms_and_cdfs(const py::bytes &data);
py::bytes                                    equalize_image(const py::bytes &data);
py::bytes                                    equalize_bgr(const py::bytes &data);
py::bytes                                    normalize_image(const py::bytes &data);
py::bytes                                    normalize_bgr(const py::bytes &data);
py::bytes                                    color_to_gray(const py::bytes &data);
ImageStats                                   compute_stats(const py::bytes &data);
py::bytes                                    apply_mapping_curve(const py::bytes &data, const std::vector<float> &mapping);

/* =========================================================
   FORWARD DECLARATIONS — color_hybrid.cpp
   ========================================================= */

py::bytes get_spectrum(const py::bytes &data);
py::bytes lowpass_filter(const py::bytes &data, float cutoff);
py::bytes highpass_filter(const py::bytes &data, float cutoff);
py::bytes create_hybrid_image(const py::bytes &low_data, const py::bytes &high_data, float cutoff);
py::bytes adjust_filter(const py::bytes &data, const std::string &filter_type, float cutoff);

/* =========================================================
   BINDINGS
   ========================================================= */

PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "C++ OpenCV Backend — noise, filters, edge detection, histogram/contrast, frequency filters";

    // ── Noise Addition ────────────────────────────────────────────────────
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

    // ── Spatial Low-Pass Filters ──────────────────────────────────────────
    m.def("apply_average_filter", &apply_average_filter,
          "Apply average (box) filter. kernel_size must be odd, in [3, 21]",
          py::arg("data"), py::arg("kernel_size"));

    m.def("apply_gaussian_filter", &apply_gaussian_filter,
          "Apply Gaussian blur filter. kernel_size must be odd, in [3, 21]",
          py::arg("data"), py::arg("kernel_size"));

    m.def("apply_median_filter", &apply_median_filter,
          "Apply median filter. kernel_size must be odd, in [3, 21]",
          py::arg("data"), py::arg("kernel_size"));

    // ── Edge Detection ────────────────────────────────────────────────────
    m.def("apply_canny", &apply_canny,
          "Canny edge detection. "
          "t_low in [0, 100], t_high in [t_low, 300], kernel_size odd in [3, 7]",
          py::arg("data"), py::arg("t_low"), py::arg("t_high"), py::arg("kernel_size"));

    m.def("apply_sobel", &apply_sobel,
          "Sobel edge detection. "
          "direction: 0 = X only, 1 = Y only, 2 = Both (magnitude)",
          py::arg("data"), py::arg("direction"));

    m.def("apply_prewitt", &apply_prewitt,
          "Prewitt edge detection. "
          "direction: 0 = X only, 1 = Y only, 2 = Both (magnitude)",
          py::arg("data"), py::arg("direction"));

    m.def("apply_roberts", &apply_roberts,
          "Roberts edge detection (2x2 kernels). "
          "direction: 0 = X only, 1 = Y only, 2 = Both (magnitude)",
          py::arg("data"), py::arg("direction"));

    // ── ImageStats ────────────────────────────────────────────────────────
    py::class_<ImageStats>(m, "ImageStats")
        .def_readonly("mean",    &ImageStats::mean)
        .def_readonly("stddev",  &ImageStats::stddev)
        .def_readonly("min_val", &ImageStats::min_val)
        .def_readonly("max_val", &ImageStats::max_val)
        .def("__repr__", [](const ImageStats &s) {
            return "<ImageStats mean="  + std::to_string(s.mean)
                 + " stddev="          + std::to_string(s.stddev)
                 + " min="             + std::to_string(s.min_val)
                 + " max="             + std::to_string(s.max_val) + ">";
        });

    // ── Histogram ─────────────────────────────────────────────────────────
    m.def("compute_histogram", &compute_histogram,
          "Raw histogram counts (256 bins) for a grayscale image (PNG bytes).",
          py::arg("data"));

    m.def("compute_bgr_histograms", &compute_bgr_histograms,
          "Returns [B_hist, G_hist, R_hist] — each a 256-element float vector.",
          py::arg("data"));

    // ── CDF / PDF ─────────────────────────────────────────────────────────
    m.def("compute_cdf", &compute_cdf,
          "Normalised CDF [0..1] computed from a histogram vector.",
          py::arg("hist"));

    m.def("compute_pdf", &compute_pdf,
          "Normalised PDF [0..1] computed from a histogram vector.",
          py::arg("hist"));

    // ── All-in-one helpers ────────────────────────────────────────────────
    m.def("get_gray_histogram_and_cdf", &get_gray_histogram_and_cdf,
          "Returns [[hist], [cdf], [pdf]] for a grayscale image (PNG bytes).",
          py::arg("data"));

    m.def("get_bgr_histograms_and_cdfs", &get_bgr_histograms_and_cdfs,
          "Returns [[[B_hist],[B_cdf],[B_pdf]], [[G_...]], [[R_...]]] (PNG bytes).",
          py::arg("data"));

    // ── Equalization ──────────────────────────────────────────────────────
    m.def("equalize_image", &equalize_image,
          "Histogram equalization on a grayscale image (PNG bytes).",
          py::arg("data"));

    m.def("equalize_bgr", &equalize_bgr,
          "Per-channel histogram equalization on a BGR image (PNG bytes).",
          py::arg("data"));

    // ── Normalization ─────────────────────────────────────────────────────
    m.def("normalize_image", &normalize_image,
          "Min-max normalization on a grayscale image (PNG bytes).",
          py::arg("data"));

    m.def("normalize_bgr", &normalize_bgr,
          "Per-channel min-max normalization on a BGR image (PNG bytes).",
          py::arg("data"));

    // ── Color → Gray ──────────────────────────────────────────────────────
    m.def("color_to_gray", &color_to_gray,
          "Convert a BGR image to grayscale (PNG bytes).",
          py::arg("data"));

    // ── Statistics ────────────────────────────────────────────────────────
    m.def("compute_stats", &compute_stats,
          "Returns ImageStats (mean, stddev, min_val, max_val) for a grayscale image (PNG bytes).",
          py::arg("data"));

    // ── Custom mapping curve (LUT) ────────────────────────────────────────
    m.def("apply_mapping_curve", &apply_mapping_curve,
          "Apply a custom 256-entry LUT/mapping to a grayscale image (PNG bytes).",
          py::arg("data"), py::arg("mapping"));

    // ── Frequency domain — spectrum ───────────────────────────────────────
    m.def("get_spectrum", &get_spectrum,
          "Log-magnitude spectrum of a grayscale image (PNG bytes). "
          "Returns a PNG with DC at centre, suitable for display.",
          py::arg("data"));

    // ── Frequency domain — filters ────────────────────────────────────────
    m.def("lowpass_filter", &lowpass_filter,
          "Low-pass filter (blurs / smooths). "
          "cutoff: frequency radius in pixels — smaller = more blur.",
          py::arg("data"), py::arg("cutoff") = 30.0f);

    m.def("highpass_filter", &highpass_filter,
          "High-pass filter (edges / detail). "
          "cutoff: frequency radius in pixels — larger = less sharpening.",
          py::arg("data"), py::arg("cutoff") = 30.0f);

    m.def("adjust_filter", &adjust_filter,
          "Dispatcher: filter_type = 'lowpass' | 'highpass'.",
          py::arg("data"), py::arg("filter_type"), py::arg("cutoff") = 30.0f);

    // ── Hybrid image ──────────────────────────────────────────────────────
    m.def("create_hybrid_image", &create_hybrid_image,
          "Hybrid image: low_data contributes blurry structure (seen from far), "
          "high_data contributes fine edges (seen up close). "
          "Both must be grayscale PNG bytes and the same size.",
          py::arg("low_data"), py::arg("high_data"), py::arg("cutoff") = 30.0f);
}