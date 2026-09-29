#include "v17_feature_extractor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace ccvnn {

InputVector17 FeatureExtractorV17::extract(const cv::Mat& frame_bgr, const cv::Rect& roi) const noexcept {
    InputVector17 v_in{};
    cv::Rect safe_roi = roi & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
    if (safe_roi.width <= 0 || safe_roi.height <= 0) {
        v_in.fill(0.0f);
        return v_in;
    }
    compute_vector(frame_bgr(safe_roi), v_in.data());
    return v_in;
}

void FeatureExtractorV17::extract_batch(const cv::Mat& frame_bgr, 
                                        std::span<const cv::Rect> rois, 
                                        float* out_tensor_buffer) const noexcept {
    const int batch_size = static_cast<int>(rois.size());

    #pragma omp parallel for if(config_.enable_openmp && batch_size > 1) schedule(static)
    for (int i = 0; i < batch_size; ++i) {
        float* batch_offset = out_tensor_buffer + (i * VECTOR_DIM_V17);
        cv::Rect safe_roi = rois[i] & cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
        
        if (safe_roi.width <= 0 || safe_roi.height <= 0) {
            std::fill_n(batch_offset, VECTOR_DIM_V17, 0.0f);
            continue;
        }
        compute_vector(frame_bgr(safe_roi), batch_offset);
    }
}

void FeatureExtractorV17::extract_batch(std::span<const cv::Mat> frames_bgr, 
                                        std::span<const cv::Rect> rois, 
                                        float* out_tensor_buffer) const noexcept {
    const int batch_size = static_cast<int>(std::min(frames_bgr.size(), rois.size()));

    #pragma omp parallel for if(config_.enable_openmp && batch_size > 1) schedule(static)
    for (int i = 0; i < batch_size; ++i) {
        float* batch_offset = out_tensor_buffer + (i * VECTOR_DIM_V17);
        const cv::Mat& frame = frames_bgr[i];
        cv::Rect safe_roi = rois[i] & cv::Rect(0, 0, frame.cols, frame.rows);

        if (frame.empty() || safe_roi.width <= 0 || safe_roi.height <= 0) {
            std::fill_n(batch_offset, VECTOR_DIM_V17, 0.0f);
            continue;
        }
        compute_vector(frame(safe_roi), batch_offset);
    }
}

void FeatureExtractorV17::compute_vector(const cv::Mat& cropped_bgr, float* out_vec) const noexcept {
    // Dynamic Shape Normalization Base
    const double total_pixels = static_cast<double>(cropped_bgr.rows * cropped_bgr.cols);
    if (total_pixels <= 0.0) {
        std::fill_n(out_vec, VECTOR_DIM_V17, 0.0f);
        return;
    }

    cv::Mat gray, hsv;
    cv::cvtColor(cropped_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(cropped_bgr, hsv, cv::COLOR_BGR2HSV);

    // 1. Spatial Morphology & Dynamic Shape Gradient (e0 - e2)
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);
    out_vec[E0_S_1D] = static_cast<float>(cv::mean(grad_mag)[0] / 255.0f);

    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    out_vec[E1_S_2D] = static_cast<float>(cv::countNonZero(thresh) / total_pixels);

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_32F);
    cv::Scalar lap_mean, lap_std;
    cv::meanStdDev(laplacian, lap_mean, lap_std);
    out_vec[E2_S_3D] = static_cast<float>(lap_std[0] / 100.0f);

    // 2. Crystal & Texture Dynamics (e3 - e5)
    cv::Scalar gray_mean, gray_std;
    cv::meanStdDev(gray, gray_mean, gray_std);
    out_vec[E3_C_PRI] = static_cast<float>(gray_std[0] / 128.0f);
    out_vec[E4_C_SEC] = static_cast<float>((lap_std[0] * lap_std[0]) / 10000.0f);

    double dot_prod = cv::mean(grad_x.mul(grad_y))[0];
    out_vec[E5_C_PHASE] = static_cast<float>(std::tanh(dot_prod / 1000.0));

    // 3. Photometric Dynamics (e6 - e8)
    cv::Mat light_mask = gray > config_.light_threshold;
    cv::Mat dark_mask  = gray < config_.dark_threshold;
    out_vec[E6_P_LIGHT]   = static_cast<float>(cv::countNonZero(light_mask) / total_pixels);
    out_vec[E7_P_DARK]    = static_cast<float>(cv::countNonZero(dark_mask)  / total_pixels);
    out_vec[E8_P_OPACITY] = static_cast<float>(1.0f - (out_vec[E6_P_LIGHT] + out_vec[E7_P_DARK]));

    // 4. Auxiliary Moments & Structural Tensor (e9 - e11)
    out_vec[E9_M_CONTRAST] = static_cast<float>(gray_std[0] / (gray_mean[0] + config_.eps));
    out_vec[E10_M_DENSITY] = static_cast<float>(out_vec[E0_S_1D] * out_vec[E1_S_2D]);

    double J11 = cv::mean(grad_x.mul(grad_x))[0];
    double J22 = cv::mean(grad_y.mul(grad_y))[0];
    double J12 = cv::mean(grad_x.mul(grad_y))[0];

    double trace = J11 + J22;
    double det = J11 * J22 - J12 * J12;
    double lambda1 = 0.5 * (trace + std::sqrt(std::max(0.0, trace * trace - 4.0 * det)));
    double lambda2 = 0.5 * (trace - std::sqrt(std::max(0.0, trace * trace - 4.0 * det)));
    out_vec[E11_M_STRUCT] = static_cast<float>((lambda1 - lambda2) / (lambda1 + lambda2 + config_.eps));

    // 5. Rotation & Chromatic Invariance (e12 - e13)
    double angle = 0.5 * std::atan2(2.0 * J12, J11 - J22);
    out_vec[E12_THETA_ROT] = static_cast<float>(angle / M_PI);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    double mean_h = cv::mean(hsv_channels[0])[0];
    double mean_s = cv::mean(hsv_channels[1])[0];
    out_vec[E13_HS_CHROMA] = static_cast<float>(mean_h / (mean_s + 1.0f));

    // 6. Photometric Normalization & Glare Suppression (e14 - e16)
    out_vec[E14_B_LUX] = static_cast<float>(gray_mean[0] / 255.0f);

    double min_val, max_val;
    cv::minMaxLoc(gray, &min_val, &max_val);
    out_vec[E15_C_RATIO] = static_cast<float>((max_val - min_val) / (max_val + min_val + config_.eps));

    cv::Mat specular_mask = gray >= config_.specular_threshold;
    out_vec[E16_R_SPECULAR] = static_cast<float>(cv::countNonZero(specular_mask) / total_pixels);
}

} // namespace ccvnn