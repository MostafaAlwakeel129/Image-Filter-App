#pragma once

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// Decode PNG bytes coming from Python into an OpenCV BGR matrix
inline cv::Mat decode_image(const py::bytes &data) {
    std::string raw = data;
    std::vector<uchar> buf(raw.begin(), raw.end());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Failed to decode image");
    return img;
}

// Encode an OpenCV matrix back into PNG bytes to send to Python
inline py::bytes encode_image(const cv::Mat &img) {
    std::vector<uchar> buf;
    cv::imencode(".png", img, buf);
    return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
}

// Convert a BGR image to grayscale using the standard luminance equation:
//   Y = 0.299·R + 0.587·G + 0.114·B
// This matches the ITU-R BT.601 luma coefficients that cvtColor uses
// internally, but is applied explicitly here so the conversion rule is
// visible and consistent across every call-site.
inline cv::Mat to_grayscale(const cv::Mat &bgr) {
    cv::Mat gray(bgr.size(), CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y) {
        const cv::Vec3b *src = bgr.ptr<cv::Vec3b>(y);
        uchar           *dst = gray.ptr<uchar>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            dst[x] = static_cast<uchar>(
                0.114f * src[x][0] +   // B
                0.587f * src[x][1] +   // G
                0.299f * src[x][2]);   // R
        }
    }
    return gray;
}