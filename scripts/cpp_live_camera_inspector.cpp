// File: scripts/cpp_live_camera_inspector.cpp
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <numeric>
#include <chrono>

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

// Extracts 9 spatial/co-coordinate features from the largest contour in a frame
std::vector<float> extract_ccvnn_features(const cv::Mat& frame) {
    cv::Mat gray, blurred, thresh;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::threshold(blurred, thresh, 100, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return std::vector<float>(9, 0.0f); // Default zero-vector if no target detected
    }

    // Locate primary inspection target
    auto max_it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });

    cv::Rect bbox = cv::boundingRect(*max_it);
    cv::Moments m = cv::moments(*max_it);
    float cx = (m.m00 != 0) ? static_cast<float>(m.m10 / m.m00) : 0.0f;
    float cy = (m.m00 != 0) ? static_cast<float>(m.m01 / m.m00) : 0.0f;

    // Normalize features relative to frame dimensions (9-element co-coordinate vector)
    float W = static_cast<float>(frame.cols);
    float H = static_cast<float>(frame.rows);

    return {
        bbox.x / W,
        bbox.y / H,
        bbox.width / W,
        bbox.height / H,
        cx / W,
        cy / H,
        static_cast<float>(cv::contourArea(*max_it)) / (W * H),
        static_cast<float>(cv::arcLength(*max_it, true)) / (2.0f * (W + H)),
        (bbox.height > 0) ? static_cast<float>(bbox.width) / bbox.height : 0.0f
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
        std::cout << "[✓] Loaded TorchScript model: " << model_path << std::endl;
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

    std::cout << "[+] Starting real-time inspection loop. Press 'q' to exit..." << std::endl;

    while (cap.read(frame)) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // 3. Extract 9-element feature vector
        std::vector<float> features = extract_ccvnn_features(frame);

        // 4. Zero-copy wrapper into PyTorch Tensor [1, 9]
        torch::Tensor input_tensor = torch::from_blob(features.data(), {1, 9}, torch::kFloat32);

        // 5. Execute Inference
        std::vector<torch::IValue> inputs = {input_tensor};
        torch::Tensor output = module.forward(inputs).toTensor();
        
        // Softmax output probabilities
        torch::Tensor probs = torch::softmax(output, /*dim=*/1);
        float pass_score = probs[0][1].item<float>();

        // 6. Apply Hysteresis Filtering
        bool pass_decision = filter.process(pass_score);

        auto t_end = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // 7. Render Visual Overlay
        cv::Scalar status_color = pass_decision ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        std::string label = (pass_decision ? "PASS" : "FAIL") + cv::format(" (Score: %.2f | Latency: %.3f ms)", pass_score, latency_ms);
        
        cv::putText(frame, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2);
        cv::imshow("CCVNN Real-Time Edge Inspector", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
