#include "image_utils.hpp"
#include <random>

// ---------- noise addition ---------------------------------------------------

// Uniform noise: each pixel channel += random int in [low, high]
py::bytes add_uniform_noise(const py::bytes &data, int low, int high) {
    cv::Mat img = decode_image(data);
    cv::Mat noise(img.size(), img.type());
    // Generate uniform random noise
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(low, high);
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            cv::Vec3b &px = img.at<cv::Vec3b>(r, c);
            for (int ch = 0; ch < 3; ++ch) {
                int val = static_cast<int>(px[ch]) + dist(rng);
                px[ch] = static_cast<uchar>(std::clamp(val, 0, 255));
            }
        }
    }
    return encode_image(img);
}

// Gaussian noise: per-channel additive noise sampled from N(mean, stddev^2)
py::bytes add_gaussian_noise(const py::bytes &data, double mean, double stddev) {
    cv::Mat img = decode_image(data);
    cv::Mat noise(img.size(), CV_16SC3);
    cv::randn(noise, cv::Scalar(mean, mean, mean), cv::Scalar(stddev, stddev, stddev));
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            cv::Vec3b &px = img.at<cv::Vec3b>(r, c);
            const cv::Vec3s &n  = noise.at<cv::Vec3s>(r, c);
            for (int ch = 0; ch < 3; ++ch) {
                int val = static_cast<int>(px[ch]) + n[ch];
                px[ch] = static_cast<uchar>(std::clamp(val, 0, 255));
            }
        }
    }
    return encode_image(img);
}

// Salt & Pepper noise: randomly set pixels to 0 or 255
py::bytes add_salt_pepper_noise(const py::bytes &data, double salt_prob, double pepper_prob) {
    cv::Mat img = decode_image(data);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            double p = dist(rng);
            if (p < salt_prob) {
                img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
            } else if (p < salt_prob + pepper_prob) {
                img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return encode_image(img);
}

// ---------- spatial domain low-pass filters ----------------------------------

// Average (box) filter
py::bytes apply_average_filter(const py::bytes &data, int kernel_size) {
    cv::Mat img = decode_image(data);
    cv::Mat result;
    cv::blur(img, result, cv::Size(kernel_size, kernel_size));
    return encode_image(result);
}

// Gaussian filter
py::bytes apply_gaussian_filter(const py::bytes &data, int kernel_size) {
    cv::Mat img = decode_image(data);
    cv::Mat result;
    // sigma = 0 lets OpenCV compute it from kernel_size
    cv::GaussianBlur(img, result, cv::Size(kernel_size, kernel_size), 0, 0);
    return encode_image(result);
}

// Median filter
py::bytes apply_median_filter(const py::bytes &data, int kernel_size) {
    cv::Mat img = decode_image(data);
    cv::Mat result;
    cv::medianBlur(img, result, kernel_size);
    return encode_image(result);
}
