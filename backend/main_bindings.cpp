#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
namespace py = pybind11;

/* =========================================================
   FORWARD DECLARATIONS — histcontrastcode.cpp
   ========================================================= */

struct ImageStats {
    float mean;
    float stddev;
    float min_val;
    float max_val;
};

std::vector<float>                           compute_histogram(const Mat&);
std::vector<std::vector<float>>              compute_bgr_histograms(const Mat&);
std::vector<float>                           compute_cdf(const std::vector<float>&);
std::vector<float>                           compute_pdf(const std::vector<float>&);
std::vector<std::vector<float>>              get_gray_histogram_and_cdf(const Mat&);
std::vector<std::vector<std::vector<float>>> get_bgr_histograms_and_cdfs(const Mat&);
Mat                                          equalize_image(const Mat&);
Mat                                          equalize_bgr(const Mat&);
Mat                                          normalize_image(const Mat&);
Mat                                          normalize_bgr(const Mat&);
Mat                                          color_to_gray(const Mat&);
ImageStats                                   compute_stats(const Mat&);
Mat                                          apply_mapping_curve(const Mat&, const std::vector<float>&);

/* =========================================================
   FORWARD DECLARATIONS — frequency_filters.cpp
   ========================================================= */

Mat compute_fft(const Mat&);
Mat get_spectrum(const Mat&);
Mat lowpass_filter(const Mat&, float);
Mat highpass_filter(const Mat&, float);
Mat create_hybrid_image(const Mat&, const Mat&, float);
Mat adjust_filter(const Mat&, const std::string&, float);

/* =========================================================
   NUMPY ↔ MAT CONVERTERS
   ========================================================= */

static Mat numpy_to_mat_1c(py::array_t<uint8_t> input)
{
    py::buffer_info buf = input.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected a 2D (grayscale) array");
    return Mat(buf.shape[0], buf.shape[1], CV_8UC1,
               static_cast<uint8_t*>(buf.ptr)).clone();
}

static Mat numpy_to_mat_3c(py::array_t<uint8_t> input)
{
    py::buffer_info buf = input.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("Expected a 3D (H x W x 3) BGR array");
    return Mat(buf.shape[0], buf.shape[1], CV_8UC3,
               static_cast<uint8_t*>(buf.ptr)).clone();
}

static py::array_t<uint8_t> mat_to_numpy_1c(const Mat& mat)
{
    return py::array_t<uint8_t>(
        { mat.rows, mat.cols },
        { (size_t)mat.step[0], (size_t)mat.step[1] },
        mat.data
    );
}

static py::array_t<uint8_t> mat_to_numpy_3c(const Mat& mat)
{
    return py::array_t<uint8_t>(
        { mat.rows, mat.cols, 3 },
        { (size_t)mat.step[0], (size_t)mat.step[1], (size_t)mat.elemSize1() },
        mat.data
    );
}

/* =========================================================
   BINDINGS
   ========================================================= */

PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "C++ OpenCV Backend — histogram/contrast + frequency filters";

    // ── ImageStats ────────────────────────────────────────────────────────
    py::class_<ImageStats>(m, "ImageStats")
        .def_readonly("mean",    &ImageStats::mean)
        .def_readonly("stddev",  &ImageStats::stddev)
        .def_readonly("min_val", &ImageStats::min_val)
        .def_readonly("max_val", &ImageStats::max_val)
        .def("__repr__", [](const ImageStats& s) {
            return "<ImageStats mean="   + std::to_string(s.mean)
                 + " stddev="           + std::to_string(s.stddev)
                 + " min="             + std::to_string(s.min_val)
                 + " max="             + std::to_string(s.max_val) + ">";
        });

    // ── Histogram ─────────────────────────────────────────────────────────
    m.def("compute_histogram", [](py::array_t<uint8_t> img) {
        return compute_histogram(numpy_to_mat_1c(img));
    }, "Raw histogram counts (256 bins) for a grayscale image.");

    m.def("compute_bgr_histograms", [](py::array_t<uint8_t> img) {
        return compute_bgr_histograms(numpy_to_mat_3c(img));
    }, "Returns [B_hist, G_hist, R_hist] — each a 256-element float vector.");

    // ── CDF / PDF ─────────────────────────────────────────────────────────
    m.def("compute_cdf", &compute_cdf,
        "Normalised CDF [0..1] computed from a histogram vector.");

    m.def("compute_pdf", &compute_pdf,
        "Normalised PDF [0..1] computed from a histogram vector.");

    // ── All-in-one helpers ────────────────────────────────────────────────
    m.def("get_gray_histogram_and_cdf", [](py::array_t<uint8_t> img) {
        return get_gray_histogram_and_cdf(numpy_to_mat_1c(img));
    }, "Returns [[hist], [cdf], [pdf]] for a grayscale image.");

    m.def("get_bgr_histograms_and_cdfs", [](py::array_t<uint8_t> img) {
        return get_bgr_histograms_and_cdfs(numpy_to_mat_3c(img));
    }, "Returns [[[B_hist],[B_cdf],[B_pdf]], [[G_...]], [[R_...]]].");

    // ── Equalization ──────────────────────────────────────────────────────
    m.def("equalize_image", [](py::array_t<uint8_t> img) {
        return mat_to_numpy_1c(equalize_image(numpy_to_mat_1c(img)));
    }, "Histogram equalization on a grayscale image.");

    m.def("equalize_bgr", [](py::array_t<uint8_t> img) {
        return mat_to_numpy_3c(equalize_bgr(numpy_to_mat_3c(img)));
    }, "Per-channel histogram equalization on a BGR image.");

    // ── Normalization ─────────────────────────────────────────────────────
    m.def("normalize_image", [](py::array_t<uint8_t> img) {
        return mat_to_numpy_1c(normalize_image(numpy_to_mat_1c(img)));
    }, "Min-max normalization on a grayscale image.");

    m.def("normalize_bgr", [](py::array_t<uint8_t> img) {
        return mat_to_numpy_3c(normalize_bgr(numpy_to_mat_3c(img)));
    }, "Per-channel min-max normalization on a BGR image.");

    // ── Color → Gray ──────────────────────────────────────────────────────
    m.def("color_to_gray", [](py::array_t<uint8_t> img) {
        return mat_to_numpy_1c(color_to_gray(numpy_to_mat_3c(img)));
    }, "Convert a BGR image to grayscale.");

    // ── Statistics ────────────────────────────────────────────────────────
    m.def("compute_stats", [](py::array_t<uint8_t> img) {
        return compute_stats(numpy_to_mat_1c(img));
    }, "Returns ImageStats (mean, stddev, min_val, max_val) for a grayscale image.");

    // ── Custom mapping curve (LUT) ────────────────────────────────────────
    m.def("apply_mapping_curve", [](py::array_t<uint8_t> img,
                                    const std::vector<float>& mapping) {
        return mat_to_numpy_1c(apply_mapping_curve(numpy_to_mat_1c(img), mapping));
    }, "Apply a custom 256-entry LUT/mapping to a grayscale image.");

    // ── Frequency domain — spectrum ───────────────────────────────────────
    m.def("get_spectrum", [](py::array_t<uint8_t> img) {
        return mat_to_numpy_1c(get_spectrum(numpy_to_mat_1c(img)));
    }, "Log-magnitude spectrum of a grayscale image (DC at centre). "
       "Returns a CV_8U grayscale image suitable for display.");

    // ── Frequency domain — filters ────────────────────────────────────────
    m.def("lowpass_filter", [](py::array_t<uint8_t> img, float cutoff) {
        return mat_to_numpy_1c(lowpass_filter(numpy_to_mat_1c(img), cutoff));
    }, py::arg("img"), py::arg("cutoff") = 30.0f,
       "Low-pass filter (blurs / smooths). "
       "cutoff: frequency radius in pixels — smaller = more blur.");

    m.def("highpass_filter", [](py::array_t<uint8_t> img, float cutoff) {
        return mat_to_numpy_1c(highpass_filter(numpy_to_mat_1c(img), cutoff));
    }, py::arg("img"), py::arg("cutoff") = 30.0f,
       "High-pass filter (edges / detail). "
       "cutoff: frequency radius in pixels — larger = less sharpening.");

    m.def("adjust_filter", [](py::array_t<uint8_t> img,
                               const std::string& filter_type,
                               float cutoff) {
        return mat_to_numpy_1c(adjust_filter(numpy_to_mat_1c(img), filter_type, cutoff));
    }, py::arg("img"), py::arg("filter_type"), py::arg("cutoff") = 30.0f,
       "Dispatcher for the PyQt UI. filter_type: 'lowpass' | 'highpass'.");

    // ── Hybrid image ──────────────────────────────────────────────────────
    m.def("create_hybrid_image", [](py::array_t<uint8_t> low_img,
                                    py::array_t<uint8_t> high_img,
                                    float cutoff) {
        return mat_to_numpy_1c(
            create_hybrid_image(
                numpy_to_mat_1c(low_img),
                numpy_to_mat_1c(high_img),
                cutoff
            )
        );
    }, py::arg("low_img"), py::arg("high_img"), py::arg("cutoff") = 30.0f,
       "Hybrid image: low_img contributes blurry structure (seen from far), "
       "high_img contributes fine edges (seen up close). "
       "Both must be grayscale and the same size.");
}