#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <numeric>
#include <chrono>
#include <cmath>
#include <algorithm>

// 5-Frame Temporal Hysteresis Filter
class HysteresisFilter {
private:
    std::deque<float> window_;
    const size_t capacity_ = 5;

public:
    bool process(float raw_score) {
        window_.push_back(raw_score);
        if (window_.size() > capacity_) {
            window_.pop_front();
        }
        float mean = std::accumulate(window_.begin(), window_.end(), 0.0f) / window_.size();
        return mean >= 0.50f; // Consensus decision threshold
    }
};

// Extracts 14-element cumulative feature vector (v14) from live camera frame
std::vector<float> extract_ccvnn_features(const cv::Mat& frame) {
    if (frame.empty()) {
        return std::vector<float>(14, 0.0f);
    }

    cv::Mat gray, blurred, thresh;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::threshold(blurred, thresh, 100, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float W = static_cast<float>(frame.cols);
    float H = static_cast<float>(frame.rows);

    cv::Scalar meanVal, stdDev;
    cv::meanStdDev(gray, meanVal, stdDev);
    float lightness = static_cast<float>(meanVal[0]) / 255.0f;
    float densityVariance = static_cast<float>(stdDev[0]) / 255.0f;

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    float edgeDensity = static_cast<float>(cv::countNonZero(edges)) / (W * H);

    if (contours.empty()) {
        std::vector<float> vec(14, 0.0f);
        vec[9] = lightness;
        vec[10] = densityVariance;
        vec[11] = edgeDensity;
        vec[12] = 0.5f;
        return vec;
    }

    // Locate primary inspection target
    auto max_it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });

    cv::Rect bbox = cv::boundingRect(*max_it);
    cv::Moments m = cv::moments(*max_it);
    float cx = (m.m00 != 0) ? static_cast<float>(m.m10 / m.m00) : 0.0f;
    float cy = (m.m00 != 0) ? static_cast<float>(m.m01 / m.m00) : 0.0f;

    // Moment-based spatial rotation angle
    float rotationAngle = static_cast<float>(0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02));
    float normRotation = (rotationAngle + static_cast<float>(M_PI) / 2.0f) / static_cast<float>(M_PI);
    normRotation = std::clamp(normRotation, 0.0f, 1.0f);

    // Saturation Index (HSV Space)
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvPlanes;
    cv::split(hsv, hsvPlanes);
    float saturationIndex = static_cast<float>(cv::mean(hsvPlanes[1])[0]) / 255.0f;

    return {
        bbox.x / W,                                              // 1. bbox_x
        bbox.y / H,                                              // 2. bbox_y
        bbox.width / W,                                          // 3. bbox_w
        bbox.height / H,                                         // 4. bbox_h
        cx / W,                                                  // 5. cx
        cy / H,                                                  // 6. cy
        static_cast<float>(cv::contourArea(*max_it)) / (W * H),  // 7. area
        static_cast<float>(cv::arcLength(*max_it, true)) / (2.0f * (W + H)), // 8. perimeter
        (bbox.height > 0) ? static_cast<float>(bbox.width) / bbox.height : 0.0f, // 9. aspect_ratio
        lightness,                                               // 10. lightness
        densityVariance,                                         // 11. densityVariance
        edgeDensity,                                             // 12. edgeDensity
        normRotation,                                            // 13. normRotation
        saturationIndex                                          // 14. saturationIndex
    };
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./ccvnn_live_inspector <path_to_model.pt> [camera_index]" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    int camera_id = (argc >= 3) ? std::stoi(argv[2]) : 0;

    // 1. Load TorchScript Model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        module.eval();
        std::cout << "[✓] Loaded V14 TorchScript model: " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "[!] Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // 2. Open Live Camera Stream
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "[!] Failed to open video camera index " << camera_id << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 120);

    HysteresisFilter filter;
    cv::Mat frame;

    std::cout << "[+] Starting real-time V14 inspection loop. Press 'q' to exit..." << std::endl;

    while (cap.read(frame)) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 3. Extract 14-element feature vector
        std::vector<float> features = extract_ccvnn_features(frame);

        // 4. Zero-copy wrapper into PyTorch Tensor [1, 14]
        torch::Tensor input_tensor = torch::from_blob(features.data(), {1, 14}, torch::kFloat32);

        // 5. Execute Inference
        std::vector<torch::IValue> inputs = {input_tensor};
        torch::Tensor output = module.forward(inputs).toTensor();
        
        float pass_score = 0.0f;
        if (output.numel() == 1) {
            pass_score = output.item<float>();
        } else {
            torch::Tensor probs = torch::softmax(output, /*dim=*/1);
            pass_score = probs[0][1].item<float>();
        }

        // 6. Apply Hysteresis Filtering
        bool pass_decision = filter.process(pass_score);

        auto t_end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // 7. Render Visual Overlay
        cv::Scalar status_color = pass_decision ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string label = (pass_decision ? "PASS" : "FAIL") + cv::format(" (Score: %.2f | Latency: %.3f ms)", pass_score, latency_ms);
        
        cv::putText(frame, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2);
        cv::imshow("CCVNN V14 Real-Time Edge Inspector", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}