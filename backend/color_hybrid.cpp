#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

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
static void fftshift(Mat& src)
{
    // Ensure even dimensions so all four quadrants are equal
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
   apply_mask — core filter routine.
   1. Forward DFT with optimal padding
   2. fftshift  → DC at centre
   3. Multiply by mask  (mask must be in shifted space)
   4. fftshift  → DC back to corner
   5. Inverse DFT, crop, normalise → CV_8U
*/
static Mat apply_mask(const Mat& gray, const Mat& mask)
{
    CV_Assert(gray.type() == CV_8UC1);

    // 1. Pad to optimal DFT size
    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    Mat padded;
    copyMakeBorder(gray, padded,
                   0, m - gray.rows, 0, n - gray.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    // Build 2-channel complex image
    Mat floatImg;
    padded.convertTo(floatImg, CV_32F);
    Mat planes[] = { floatImg, Mat::zeros(floatImg.size(), CV_32F) };
    Mat fft;
    merge(planes, 2, fft);

    // 2. Forward DFT
    dft(fft, fft);

    // 3. Shift DC to centre
    fftshift(fft);

    // 4. Resize mask to match padded FFT size if the caller passed a
    //    mask built from the original (unpadded) image size
    Mat m2;
    if (mask.size() != fft.size())
        resize(mask, m2, fft.size(), 0, 0, INTER_LINEAR);
    else
        m2 = mask;

    // 5. Expand mask to 2 channels and multiply
    Mat maskPlanes[] = { m2, m2 };
    Mat maskComplex;
    merge(maskPlanes, 2, maskComplex);
    multiply(fft, maskComplex, fft);

    // 6. Shift DC back to corner
    fftshift(fft);

    // 7. Inverse DFT (DFT_SCALE divides by N×M → correct amplitude)
    Mat result;
    dft(fft, result, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

    // 8. Crop to original size, normalise, convert
    result = result(Rect(0, 0, gray.cols, gray.rows));
    normalize(result, result, 0, 255, NORM_MINMAX);
    Mat output;
    result.convertTo(output, CV_8U);
    return output;
}

/* =========================================================
   PUBLIC API
   ========================================================= */

/*
   compute_fft — returns the raw (unshifted) 2-channel complex
   CV_32F DFT of `gray`, padded to the optimal DFT size.
   Useful if you need direct access to the spectrum.
*/
Mat compute_fft(const Mat& gray)
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
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);
    return complexImg;
}

/*
   get_spectrum — log-magnitude spectrum for display.
   Returns a normalised CV_8U image with DC at the centre.
*/
Mat get_spectrum(const Mat& gray)
{
    Mat fft = compute_fft(gray);
    fftshift(fft);

    Mat planes[2];
    split(fft, planes);

    Mat mag;
    magnitude(planes[0], planes[1], mag);
    mag += Scalar::all(1);
    log(mag, mag);
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8U);
    return mag;
}

/*
   lowpass_filter — keeps low frequencies (blurs / smooths).
   cutoff: radius in pixels in the shifted frequency domain.
           Smaller = more aggressive blur.
*/
Mat lowpass_filter(const Mat& gray, float cutoff)
{
    CV_Assert(gray.type() == CV_8UC1);
    Mat mask = build_mask_lowpass(gray.size(), cutoff);
    return apply_mask(gray, mask);
}

/*
   highpass_filter — keeps high frequencies (edges / detail).
   cutoff: radius in pixels in the shifted frequency domain.
           Larger = less aggressive sharpening.
*/
Mat highpass_filter(const Mat& gray, float cutoff)
{
    CV_Assert(gray.type() == CV_8UC1);
    Mat mask = build_mask_highpass(gray.size(), cutoff);
    return apply_mask(gray, mask);
}

/*
   create_hybrid_image — classic Oliva & Torralba hybrid image.

   lowFreqImg  → contributes blurry structure  (visible from far away)
   highFreqImg → contributes fine edges         (visible up close)
   cutoff      → shared frequency boundary (pixels)

   The high-pass component is re-biased to 0.5 before blending
   because its mean is near zero; without this the hybrid looks dark.
*/
Mat create_hybrid_image(const Mat& lowFreqImg,
                        const Mat& highFreqImg,
                        float cutoff)
{
    CV_Assert(lowFreqImg.type()  == CV_8UC1);
    CV_Assert(highFreqImg.type() == CV_8UC1);
    CV_Assert(lowFreqImg.size()  == highFreqImg.size());

    Mat lowFreq  = lowpass_filter (lowFreqImg,  cutoff);
    Mat highFreq = highpass_filter(highFreqImg, cutoff);

    Mat lowF, highF;
    lowFreq.convertTo (lowF,  CV_32F);
    highFreq.convertTo(highF, CV_32F);

    // Normalise each component independently to [0, 1]
    normalize(lowF,  lowF,  0.0, 1.0, NORM_MINMAX);
    normalize(highF, highF, 0.0, 1.0, NORM_MINMAX);

    // Blend: low-pass at half weight, high-pass re-biased around 0.5
    Mat hybrid = lowF * 0.5f + (highF + 0.5f) * 0.5f;

    normalize(hybrid, hybrid, 0, 255, NORM_MINMAX);
    Mat result;
    hybrid.convertTo(result, CV_8U);
    return result;
}

/*
   adjust_filter — convenience dispatcher for the PyQt UI.
   filterType: "lowpass" | "highpass"
*/
Mat adjust_filter(const Mat& gray, const string& filterType, float cutoff)
{
    if      (filterType == "lowpass")  return lowpass_filter (gray, cutoff);
    else if (filterType == "highpass") return highpass_filter(gray, cutoff);
    else                               return gray.clone();
}