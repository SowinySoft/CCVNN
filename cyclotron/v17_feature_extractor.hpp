#ifndef V17_FEATURE_EXTRACTOR_HPP
#define V17_FEATURE_EXTRACTOR_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <span>
#include <opencv2/opencv.hpp>

namespace ccvnn {

constexpr size_t VECTOR_DIM_V17 = 17;
using InputVector17 = std::array<float, VECTOR_DIM_V17>;

enum FeatureIndex : uint8_t {
    E0_S_1D       = 0,  // 1D Edge Gradient Magnitude
    E1_S_2D       = 1,  // 2D Area Morphology Ratio
    E2_S_3D       = 2,  // 3D Lattice Surface Relief Depth
    E3_C_PRI      = 3,  // Primary Texture Frequency
    E4_C_SEC      = 4,  // Secondary High-Frequency Grain Intensity
    E5_C_PHASE    = 5,  // Phase Periodicity
    E6_P_LIGHT    = 6,  // Photometric Lightness / Reflectance
    E7_P_DARK     = 7,  // Photometric Darkness / Shadow Density
    E8_P_OPACITY  = 8,  // Material Opacity Index
    E9_M_CONTRAST = 9,  // Local Neighborhood Variance (RMS)
    E10_M_DENSITY = 10, // Defect Spatial Density Variance
    E11_M_STRUCT  = 11, // Structural Tensor Eigenvalue Ratio
    E12_THETA_ROT = 12, // Principal Rotation Angle (Normalized [-1, 1])
    E13_HS_CHROMA = 13, // Hue / Saturation Chromaticity Ratio
    E14_B_LUX     = 14, // Absolute Scene Luminance (v16 addition)
    E15_C_RATIO   = 15, // Dynamic Foreground/Background Contrast (v16 addition)
    E16_R_SPECULAR= 16  // Specular Reflection / Glare Index (v17 addition)
};

// Tunable extractor hyper-parameters passed by reference (zero runtime cost)
struct ExtractorConfig {
    uint8_t light_threshold    = 180;
    uint8_t dark_threshold     = 50;
    uint8_t specular_threshold = 252;
    float   eps                = 1e-5f;
    bool    enable_openmp      = true;  // Parallel processing across dynamic batches
};

class FeatureExtractorV17 {
public:
    explicit FeatureExtractorV17(ExtractorConfig config = ExtractorConfig{})
        : config_(config) {}

    ~FeatureExtractorV17() = default;

    // Single Frame / Single ROI Extraction (Dynamic Shape)
    InputVector17 extract(const cv::Mat& frame_bgr, const cv::Rect& roi) const noexcept;

    // Dynamic Batching: Single Frame with Multiple ROIs -> Writes directly to contiguous raw buffer [N, 17]
    void extract_batch(const cv::Mat& frame_bgr, 
                       std::span<const cv::Rect> rois, 
                       float* out_tensor_buffer) const noexcept;

    // Dynamic Batching: Multiple Variable-Resolution Frames/ROIs -> Direct tensor buffer write [N, 17]
    void extract_batch(std::span<const cv::Mat> frames_bgr, 
                       std::span<const cv::Rect> rois, 
                       float* out_tensor_buffer) const noexcept;

private:
    ExtractorConfig config_;
    
    // Inlined core feature engine operating on a dynamic-resolution matrix
    void compute_vector(const cv::Mat& cropped_bgr, float* out_vec) const noexcept;
};

} // namespace ccvnn

#endif // V17_FEATURE_EXTRACTOR_HPP