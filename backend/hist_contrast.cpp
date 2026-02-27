#include "image_utils.hpp"
#include <vector>
#include <string>

using namespace cv;

/* =========================================================
   TYPES
   ========================================================= */

struct ImageStats {
    float mean;
    float stddev;
    float min_val;
    float max_val;
};

/* =========================================================
   HISTOGRAM COMPUTATION
   ========================================================= */

// Raw bin counts (256 bins) for a grayscale image
std::vector<float> compute_histogram(const py::bytes &data)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    std::vector<float> hist(256, 0.0f);
    for (int y = 0; y < gray.rows; ++y) {
        const uchar *row = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x)
            hist[row[x]]++;
    }
    return hist;
}

// Raw bin counts per channel — order is BGR (OpenCV native)
std::vector<std::vector<float>> compute_bgr_histograms(const py::bytes &data)
{
    Mat img = decode_image(data);

    std::vector<std::vector<float>> histograms(3, std::vector<float>(256, 0.0f));
    for (int y = 0; y < img.rows; ++y) {
        const Vec3b *row = img.ptr<Vec3b>(y);
        for (int x = 0; x < img.cols; ++x) {
            histograms[0][row[x][0]]++;   // B
            histograms[1][row[x][1]]++;   // G
            histograms[2][row[x][2]]++;   // R
        }
    }
    return histograms;
}

/* =========================================================
   CDF / PDF
   ========================================================= */

// Cumulative distribution function, normalised to [0, 1]
std::vector<float> compute_cdf(const std::vector<float> &hist)
{
    std::vector<float> cdf(256, 0.0f);
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + hist[i];

    float total = cdf[255];
    if (total > 0)
        for (float &v : cdf)
            v /= total;

    return cdf;
}

// Probability density function (normalised histogram)
std::vector<float> compute_pdf(const std::vector<float> &hist)
{
    float total = 0.0f;
    for (float v : hist) total += v;

    std::vector<float> pdf(256, 0.0f);
    if (total > 0)
        for (int i = 0; i < 256; ++i)
            pdf[i] = hist[i] / total;

    return pdf;
}

/* =========================================================
   ALL-IN-ONE HELPERS
   Returns { hist, cdf, pdf } in one call to avoid redundant
   histogram traversals when the caller needs all three.
   ========================================================= */

// Grayscale: returns [hist, cdf, pdf]
std::vector<std::vector<float>> get_gray_histogram_and_cdf(const py::bytes &data)
{
    auto hist = compute_histogram(data);
    return { hist, compute_cdf(hist), compute_pdf(hist) };
}

// Color: returns [[B_hist, B_cdf, B_pdf], [G_...], [R_...]]
std::vector<std::vector<std::vector<float>>> get_bgr_histograms_and_cdfs(const py::bytes &data)
{
    auto hists = compute_bgr_histograms(data);
    std::vector<std::vector<std::vector<float>>> result;
    result.reserve(3);
    for (const auto &hist : hists)
        result.push_back({ hist, compute_cdf(hist), compute_pdf(hist) });
    return result;
}

/* =========================================================
   HISTOGRAM EQUALIZATION
   ========================================================= */

py::bytes equalize_image(const py::bytes &data)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat equalized;
    equalizeHist(gray, equalized);

    Mat result;
    cvtColor(equalized, result, COLOR_GRAY2BGR);
    return encode_image(result);
}

py::bytes equalize_bgr(const py::bytes &data)
{
    Mat img = decode_image(data);

    std::vector<Mat> channels(3);
    split(img, channels);
    for (Mat &ch : channels)
        equalizeHist(ch, ch);

    Mat result;
    merge(channels, result);
    return encode_image(result);
}

/* =========================================================
   NORMALIZATION
   ========================================================= */

py::bytes normalize_image(const py::bytes &data)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat normalized;
    normalize(gray, normalized, 0, 255, NORM_MINMAX);

    Mat result;
    cvtColor(normalized, result, COLOR_GRAY2BGR);
    return encode_image(result);
}

py::bytes normalize_bgr(const py::bytes &data)
{
    Mat img = decode_image(data);

    std::vector<Mat> channels(3);
    split(img, channels);
    for (Mat &ch : channels)
        normalize(ch, ch, 0, 255, NORM_MINMAX);

    Mat result;
    merge(channels, result);
    return encode_image(result);
}

/* =========================================================
   COLOR CONVERSION
   ========================================================= */

py::bytes color_to_gray(const py::bytes &data)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat result;
    cvtColor(gray, result, COLOR_GRAY2BGR);
    return encode_image(result);
}

/* =========================================================
   IMAGE STATISTICS
   ========================================================= */

ImageStats compute_stats(const py::bytes &data)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Scalar mean, stddev;
    meanStdDev(gray, mean, stddev);

    double min_val, max_val;
    minMaxLoc(gray, &min_val, &max_val);

    return {
        static_cast<float>(mean[0]),
        static_cast<float>(stddev[0]),
        static_cast<float>(min_val),
        static_cast<float>(max_val)
    };
}

/* =========================================================
   CUSTOM MAPPING CURVE (LUT)
   ========================================================= */

py::bytes apply_mapping_curve(const py::bytes &data, const std::vector<float> &mapping)
{
    if (mapping.size() != 256)
        throw std::runtime_error("mapping must have exactly 256 entries");

    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat mapped(gray.size(), CV_8UC1);
    for (int y = 0; y < gray.rows; ++y) {
        const uchar *src = gray.ptr<uchar>(y);
        uchar       *dst = mapped.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x)
            dst[x] = static_cast<uchar>(
                std::min(255.0f, std::max(0.0f, mapping[src[x]])));
    }

    Mat result;
    cvtColor(mapped, result, COLOR_GRAY2BGR);
    return encode_image(result);
}