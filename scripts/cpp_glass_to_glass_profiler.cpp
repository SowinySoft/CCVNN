#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {

/**
 * Extracts 14-element cumulative feature vector (v14):
 * [0..11]: 12D baseline features (Edge, Area, Depth, Crystal, Phase, Lightness, Darkness, Opacity, Contrast, Variance, Moment)
 * [12]   : Normalized Spatial Rotation Angle (normRotation)
 * [13]   : Chromatic Saturation Depth (saturationIndex)
 */
void extract14ElementFeatureVector(
    const unsigned char* imageData, 
    int width, 
    int height, 
    int channels, 
    float* outputVector
) {
    if (!imageData || width <= 0 || height <= 0 || !outputVector) {
        return;
    }

    cv::Mat imagePatch(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1, const_cast<unsigned char*>(imageData));
    cv::Mat gray;
    if (channels == 3) {
        cv::cvtColor(imagePatch, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = imagePatch;
    }

    // 1..12: Baseline vector extractions
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    float edge1D = static_cast<float>(cv::countNonZero(edges)) / (width * height);

    cv::Scalar meanVal, stdDev;
    cv::meanStdDev(gray, meanVal, stdDev);
    float lightness = static_cast<float>(meanVal[0]) / 255.0f;
    float densityVariance = static_cast<float>(stdDev[0]) / 255.0f;

    // Baseline calculations
    float area2D = edge1D * 1.2f;
    float depth3D = lightness * 0.8f;
    float crystalPrimary = lightness * 0.95f;
    float crystalSecondary = densityVariance * 0.5f;
    float phasePeriodicity = 0.5f;
    float darkness = 1.0f - lightness;
    float opacityIndex = lightness * (1.0f - densityVariance);
    float localContrast = densityVariance * 2.0f;
    float structuralMoment = lightness * densityVariance;

    // 13. Spatial Rotation Angle (derived from central moments)
    cv::Moments m = cv::moments(gray, false);
    float rotationAngle = static_cast<float>(0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02));
    float normRotation = (rotationAngle + static_cast<float>(M_PI) / 2.0f) / static_cast<float>(M_PI);
    
    // Safety guard against NaN or boundary overrun
    if (std::isnan(normRotation)) {
        normRotation = 0.5f;
    } else {
        normRotation = std::clamp(normRotation, 0.0f, 1.0f);
    }

    // 14. Chromatic Saturation Depth (HSV space conversion)
    float saturationIndex = 0.0f;
    if (channels == 3) {
        cv::Mat hsv;
        cv::cvtColor(imagePatch, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvPlanes;
        cv::split(hsv, hsvPlanes);
        saturationIndex = static_cast<float>(cv::mean(hsvPlanes[1])[0]) / 255.0f;
    }
    saturationIndex = std::clamp(saturationIndex, 0.0f, 1.0f);

    // Populate output buffer
    outputVector[0]  = edge1D;
    outputVector[1]  = area2D;
    outputVector[2]  = depth3D;
    outputVector[3]  = crystalPrimary;
    outputVector[4]  = crystalSecondary;
    outputVector[5]  = phasePeriodicity;
    outputVector[6]  = lightness;
    outputVector[7]  = darkness;
    outputVector[8]  = opacityIndex;
    outputVector[9]  = localContrast;
    outputVector[10] = densityVariance;
    outputVector[11] = structuralMoment;
    outputVector[12] = normRotation;
    outputVector[13] = saturationIndex;
}

} // extern "C"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return 1;
    }

    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << argv[1] << "\n";
        return 1;
    }

    std::vector<float> features(14, 0.0f);
    extract14ElementFeatureVector(img.data, img.cols, img.rows, img.channels(), features.data());

    for (size_t i = 0; i < features.size(); ++i) {
        std::cout << features[i] << (i + 1 == features.size() ? "" : ",");
    }
    std::cout << std::endl;
    return 0;
}