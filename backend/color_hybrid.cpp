#include "image_utils.hpp"
#include <cmath>
#include <string>

using namespace cv;

// Smoothing width for the soft band-edge on both masks (pixels)
static constexpr float MASK_SIGMA = 10.0f;

/* =========================================================
   INTERNAL HELPERS
   ========================================================= */

/*
   fftshift — swaps the four quadrants so the DC component
   moves from the top-left corner to the centre of the image.
   Calling it twice is the identity (i.e. it also serves as
   ifftshift for even-sized images).
   Works in-place on any single- or multi-channel float Mat.
*/
static void fftshift(Mat &src)
{
    src = src(Rect(0, 0, src.cols & ~1, src.rows & ~1));

    int cx = src.cols / 2;
    int cy = src.rows / 2;

    Mat q0(src, Rect(0,  0,  cx, cy));   // top-left
    Mat q1(src, Rect(cx, 0,  cx, cy));   // top-right
    Mat q2(src, Rect(0,  cy, cx, cy));   // bottom-left
    Mat q3(src, Rect(cx, cy, cx, cy));   // bottom-right

    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

/*
   build_mask_lowpass — single-channel CV_32F mask of `size`.
   Values are 1 inside the circle of `radius`, falling off with
   a Gaussian soft edge outside.
   The mask is in SHIFTED space (DC at centre).
*/
static Mat build_mask_lowpass(Size size, float radius)
{
    Mat mask(size, CV_32F);
    const float sigma2 = 2.0f * MASK_SIGMA * MASK_SIGMA;
    int cx = size.width  / 2;
    int cy = size.height / 2;

    for (int i = 0; i < size.height; ++i)
        for (int j = 0; j < size.width; ++j) {
            float d = hypot(float(i - cy), float(j - cx));
            mask.at<float>(i, j) = (d <= radius)
                ? 1.0f
                : exp(-(d - radius) * (d - radius) / sigma2);
        }
    return mask;
}

/*
   build_mask_highpass — exact complement of the low-pass mask
   so that LPF + HPF = identity at every pixel.
*/
static Mat build_mask_highpass(Size size, float radius)
{
    Mat hp;
    subtract(Scalar::all(1.0f), build_mask_lowpass(size, radius), hp);
    return hp;
}

/*
   apply_mask_internal — core filter routine on a raw cv::Mat.
   1. Forward DFT with optimal padding
   2. fftshift  → DC at centre
   3. Multiply by mask
   4. fftshift  → DC back to corner
   5. Inverse DFT, crop, normalise → CV_8U
*/
static Mat apply_mask_internal(const Mat &gray, const Mat &mask)
{
    CV_Assert(gray.type() == CV_8UC1);

    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    Mat padded;
    copyMakeBorder(gray, padded,
                   0, m - gray.rows, 0, n - gray.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    Mat floatImg;
    padded.convertTo(floatImg, CV_32F);
    Mat planes[] = { floatImg, Mat::zeros(floatImg.size(), CV_32F) };
    Mat fft;
    merge(planes, 2, fft);

    dft(fft, fft);
    fftshift(fft);

    Mat m2;
    if (mask.size() != fft.size())
        resize(mask, m2, fft.size(), 0, 0, INTER_LINEAR);
    else
        m2 = mask;

    Mat maskPlanes[] = { m2, m2 };
    Mat maskComplex;
    merge(maskPlanes, 2, maskComplex);
    multiply(fft, maskComplex, fft);

    fftshift(fft);

    Mat result;
    dft(fft, result, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

    result = result(Rect(0, 0, gray.cols, gray.rows));
    normalize(result, result, 0, 255, NORM_MINMAX);
    Mat output;
    result.convertTo(output, CV_8U);
    return output;
}

/* =========================================================
   PUBLIC API  — all functions accept/return py::bytes
   ========================================================= */

/*
   get_spectrum — log-magnitude spectrum for display.
   Returns a normalised BGR PNG image with DC at the centre.
*/
py::bytes get_spectrum(const py::bytes &data)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    Mat padded;
    copyMakeBorder(gray, padded,
                   0, m - gray.rows, 0, n - gray.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    Mat floatImg;
    padded.convertTo(floatImg, CV_32F);
    Mat planes[] = { floatImg, Mat::zeros(floatImg.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);
    fftshift(complexImg);

    Mat splitPlanes[2];
    split(complexImg, splitPlanes);

    Mat mag;
    magnitude(splitPlanes[0], splitPlanes[1], mag);
    mag += Scalar::all(1);
    log(mag, mag);

    Mat mag8u;
    normalize(mag, mag8u, 0, 255, NORM_MINMAX, CV_8U);

    // Crop back to original size
    mag8u = mag8u(Rect(0, 0, gray.cols, gray.rows));

    Mat result;
    cvtColor(mag8u, result, COLOR_GRAY2BGR);
    return encode_image(result);
}

/*
   lowpass_filter — keeps low frequencies (blurs / smooths).
   cutoff: radius in pixels in the shifted frequency domain.
*/
py::bytes lowpass_filter(const py::bytes &data, float cutoff)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat mask   = build_mask_lowpass(gray.size(), cutoff);
    Mat filtered = apply_mask_internal(gray, mask);

    Mat result;
    cvtColor(filtered, result, COLOR_GRAY2BGR);
    return encode_image(result);
}

/*
   highpass_filter — keeps high frequencies (edges / detail).
   cutoff: radius in pixels in the shifted frequency domain.
*/
py::bytes highpass_filter(const py::bytes &data, float cutoff)
{
    Mat img  = decode_image(data);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat mask   = build_mask_highpass(gray.size(), cutoff);
    Mat filtered = apply_mask_internal(gray, mask);

    Mat result;
    cvtColor(filtered, result, COLOR_GRAY2BGR);
    return encode_image(result);
}

/*
   create_hybrid_image — classic Oliva & Torralba hybrid image.
   low_data  → contributes blurry structure  (visible from far away)
   high_data → contributes fine edges         (visible up close)
   cutoff    → shared frequency boundary (pixels)
*/
py::bytes create_hybrid_image(const py::bytes &low_data,
                              const py::bytes &high_data,
                              float cutoff)
{
    Mat lowImg  = decode_image(low_data);
    Mat highImg = decode_image(high_data);

    // Convert both to grayscale for processing
    Mat lowGray, highGray;
    cvtColor(lowImg,  lowGray,  COLOR_BGR2GRAY);
    cvtColor(highImg, highGray, COLOR_BGR2GRAY);

    // Resize if needed
    if (lowGray.size() != highGray.size())
        resize(highGray, highGray, lowGray.size());

    Mat lowFiltered  = apply_mask_internal(lowGray,  build_mask_lowpass (lowGray.size(), cutoff));
    Mat highFiltered = apply_mask_internal(highGray, build_mask_highpass(highGray.size(), cutoff));

    Mat lowF, highF;
    lowFiltered.convertTo (lowF,  CV_32F);
    highFiltered.convertTo(highF, CV_32F);

    normalize(lowF,  lowF,  0.0, 1.0, NORM_MINMAX);
    normalize(highF, highF, 0.0, 1.0, NORM_MINMAX);

    // Blend: low-pass at half weight, high-pass re-biased around 0.5
    Mat hybrid = lowF * 0.5f + (highF + 0.5f) * 0.5f;

    normalize(hybrid, hybrid, 0, 255, NORM_MINMAX);
    Mat hybrid8u;
    hybrid.convertTo(hybrid8u, CV_8U);

    Mat result;
    cvtColor(hybrid8u, result, COLOR_GRAY2BGR);
    return encode_image(result);
}

/*
   adjust_filter — convenience dispatcher for the PyQt UI.
   filterType: "lowpass" | "highpass"
*/
py::bytes adjust_filter(const py::bytes &data, const std::string &filterType, float cutoff)
{
    if      (filterType == "lowpass")  return lowpass_filter (data, cutoff);
    else if (filterType == "highpass") return highpass_filter(data, cutoff);
    else                               return data;
}