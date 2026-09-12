#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

std::vector<float> extract12ElementFeatureVector(const cv::Mat& imagePatch) {
    cv::Mat gray;
    if (imagePatch.channels() == 3) {
        cv::cvtColor(imagePatch, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = imagePatch.clone();
    }

    // 1. Spatial Shape Qualities
    cv::Mat gradX, gradY, absGradX, absGradY;
    cv::Sobel(gray, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_32F, 0, 1, 3);
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);
    
    cv::Mat gradSum;
    cv::add(absGradX, absGradY, gradSum);
    float edge1D = static_cast<float>(cv::mean(gradSum)[0]);

    cv::Mat canny;
    cv::Canny(gray, canny, 50, 150);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    float area2D = (!contours.empty()) ? static_cast<float>(cv::contourArea(contours[0])) : 0.0f;

    cv::Scalar meanVal, stdDevVal;
    cv::meanStdDev(gray, meanVal, stdDevVal);
    float depth3D = static_cast<float>(stdDevVal[0]);

    // 2. Crystal Layer Depth Factors (FFT)
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(grayFloat.rows);
    int n = cv::getOptimalDFTSize(grayFloat.cols);
    cv::copyMakeBorder(grayFloat, padded, 0, m - grayFloat.rows, 0, n - grayFloat.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);

    cv::split(complexI, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);

    cv::Scalar fftMean, fftStd;
    cv::meanStdDev(mag, fftMean, fftStd);
    float crystalPrimary = static_cast<float>(fftMean[0]);
    float crystalSecondary = static_cast<float>(fftStd[0]);

    double maxVal;
    cv::minMaxLoc(mag, nullptr, &maxVal);
    cv::Scalar totalSum = cv::sum(mag);
    float phasePeriodicity = static_cast<float>(maxVal / (totalSum[0] + 1e-6));

    // 3. Photometric & Opacity Dynamics
    std::vector<uchar> flatPixels;
    if (gray.isContinuous()) {
        flatPixels.assign(gray.data, gray.data + gray.total());
    } else {
        for (int i = 0; i < gray.rows; ++i) {
            flatPixels.insert(flatPixels.end(), gray.ptr<uchar>(i), gray.ptr<uchar>(i) + gray.cols);
        }
    }
    std::sort(flatPixels.begin(), flatPixels.end());

    size_t idx10 = static_cast<size_t>(flatPixels.size() * 0.10);
    size_t idx90 = static_cast<size_t>(flatPixels.size() * 0.90);
    float darkness = static_cast<float>(flatPixels[idx10]);
    float lightness = static_cast<float>(flatPixels[idx90]);
    float opacityIndex = static_cast<float>(meanVal[0] / 255.0);

    // 4. Auxiliary Spatial & Contrast Moments
    double minPixelVal, maxPixelVal;
    cv::minMaxLoc(gray, &minPixelVal, &maxPixelVal);
    float localContrast = static_cast<float>(maxPixelVal - minPixelVal);
    float densityVariance = static_cast<float>(stdDevVal[0] * stdDevVal[0]);

    cv::Mat structProduct;
    cv::multiply(gradX, gradY, structProduct);
    float structuralMoment = static_cast<float>(cv::mean(structProduct)[0]);

    return {
        edge1D, area2D, depth3D,
        crystalPrimary, crystalSecondary, phasePeriodicity,
        lightness, darkness, opacityIndex,
        localContrast, densityVariance, structuralMoment
    };
}