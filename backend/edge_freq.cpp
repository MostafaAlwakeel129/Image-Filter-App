#include "image_utils.hpp"

// Canny Edge Detection

py::bytes apply_canny(const py::bytes &data,
                      double t_low,
                      double t_high,
                      int    kernel_size)
{
    cv::Mat img = decode_image(data);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(kernel_size, kernel_size), 0);
    cv::Mat edges;
    cv::Canny(blurred, edges, t_low, t_high);
    cv::Mat result;
    cv::cvtColor(edges, result, cv::COLOR_GRAY2BGR);

    return encode_image(result);
}

// Sobel,Prewitt,Roberts Edge Detection

static cv::Mat apply_gradient_kernels(const cv::Mat &gray,
                                      const cv::Mat &Kx,
                                      const cv::Mat &Ky,
                                      int direction)
{

    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F);
    cv::Mat Gx, Gy;
    cv::filter2D(gray_f, Gx, CV_32F, Kx);
    cv::filter2D(gray_f, Gy, CV_32F, Ky);

    cv::Mat result_f;
    if (direction == 0) {
        result_f = cv::abs(Gx);           // vertical edges only
    } else if (direction == 1) {
        result_f = cv::abs(Gy);           // horizontal edges only
    } else {
        cv::magnitude(Gx, Gy, result_f);  // full magnitude sqrt(Gx²+Gy²)
    }

    //cv::Mat result_8u;
    //v::normalize(result_f, result_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat result_8u;
    result_f.convertTo(result_8u, CV_8U);
    
    return result_8u;
}

py::bytes apply_sobel(const py::bytes &data, int direction)
{
    cv::Mat img = decode_image(data);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat Kx = (cv::Mat_<float>(3, 3) <<
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1);

    cv::Mat Ky = (cv::Mat_<float>(3, 3) <<
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1);

    cv::Mat result_8u = apply_gradient_kernels(gray, Kx, Ky, direction);

    cv::Mat result_bgr;
    cv::cvtColor(result_8u, result_bgr, cv::COLOR_GRAY2BGR);

    return encode_image(result_bgr);
}

py::bytes apply_prewitt(const py::bytes &data, int direction)
{
    cv::Mat img = decode_image(data);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat Kx = (cv::Mat_<float>(3, 3) <<
        -1,  0,  1,
        -1,  0,  1,
        -1,  0,  1);

    cv::Mat Ky = (cv::Mat_<float>(3, 3) <<
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1);

    cv::Mat result_8u = apply_gradient_kernels(gray, Kx, Ky, direction);
    cv::Mat result_bgr;
    cv::cvtColor(result_8u, result_bgr, cv::COLOR_GRAY2BGR);

    return encode_image(result_bgr);
}

py::bytes apply_roberts(const py::bytes &data, int direction)
{
    cv::Mat img = decode_image(data);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 2x2 kernels — diagonal gradient operators
    cv::Mat Kx = (cv::Mat_<float>(2, 2) <<
         1,  0,
         0, -1);

    cv::Mat Ky = (cv::Mat_<float>(2, 2) <<
         0,  1,
        -1,  0);

    cv::Mat result_8u = apply_gradient_kernels(gray, Kx, Ky, direction);
    cv::Mat result_bgr;
    cv::cvtColor(result_8u, result_bgr, cv::COLOR_GRAY2BGR);

    return encode_image(result_bgr);
}