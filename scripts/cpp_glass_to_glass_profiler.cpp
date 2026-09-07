// File: scripts/cpp_glass_to_glass_profiler.cpp
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <fstream>

using Clock = std::chrono::high_resolution_clock;
using DurationUs = std::chrono::duration<double, std::micro>;

struct LatencyBreakdown {
    double frame_cap_us;
    double feature_extract_us;
    double inference_us;
    double hysteresis_us;
    double gpio_trigger_us;
    double total_glass_to_glass_us;
};

// Mock/Sysfs GPIO Relay Interface
class GPIORelay {
private:
    int gpio_pin_;
    bool hardware_enabled_;

public:
    GPIORelay(int pin = 18, bool enable_hw = false) : gpio_pin_(pin), hardware_enabled_(enable_hw) {}

    void trigger(bool pass) {
        auto t0 = Clock::now();
        if (hardware_enabled_) {
            // High-speed GPIO toggle via sysfs or libgpiod
            std::ofstream gpio_val("/sys/class/gpio/gpio" + std::to_string(gpio_pin_) + "/value");
            if (gpio_val.is_open()) {
                gpio_val << (pass ? "1" : "0");
            }
        } else {
            // Microsecond volatile memory write simulation
            volatile int signal = pass ? 1 : 0;
            (void)signal;
        }
    }
};

class HysteresisFilter {
private:
    std::deque<float> window_;
    const size_t capacity_ = 5;

public:
    bool process(float raw_score) {
        window_.push_back(raw_score);
        if (window_.size() > capacity_) window_.pop_front();
        float mean = std::accumulate(window_.begin(), window_.end(), 0.0f) / window_.size();
        return mean >= 0.50f;
    }
};

std::vector<float> extract_features(const cv::Mat& frame) {
    cv::Mat gray, thresh;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, thresh, 100, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return std::vector<float>(9, 0.0f);

    auto max_it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });

    cv::Rect bbox = cv::boundingRect(*max_it);
    cv::Moments m = cv::moments(*max_it);
    float cx = (m.m00 != 0) ? static_cast<float>(m.m10 / m.m00) : 0.0f;
    float cy = (m.m00 != 0) ? static_cast<float>(m.m01 / m.m00) : 0.0f;

    float W = static_cast<float>(frame.cols);
    float H = static_cast<float>(frame.rows);

    return {
        bbox.x / W, bbox.y / H, bbox.width / W, bbox.height / H,
        cx / W, cy / H,
        static_cast<float>(cv::contourArea(*max_it)) / (W * H),
        static_cast<float>(cv::arcLength(*max_it, true)) / (2.0f * (W + H)),
        (bbox.height > 0) ? static_cast<float>(bbox.width) / bbox.height : 0.0f
    };
}

void print_percentiles(std::vector<double>& v, const std::string& name) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double p50 = v[n * 0.50];
    double p95 = v[n * 0.95];
    double p99 = v[n * 0.99];
    std::cout << "  " << name << " -> p50: " << p50 / 1000.0 << " ms | p95: " 
              << p95 / 1000.0 << " ms | p99: " << p99 / 1000.0 << " ms\n";
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./ccvnn_glass_profiler <model_path> [num_frames=500] [camera_id=0]\n";
        return -1;
    }

    std::string model_path = argv[1];
    int max_frames = (argc >= 3) ? std::stoi(argv[2]) : 500;
    int camera_id = (argc >= 4) ? std::stoi(argv[3]) : 0;

    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();

    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "[!] Camera capture failed to initialize.\n";
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    HysteresisFilter filter;
    GPIORelay relay(18, false);
    cv::Mat frame;

    std::vector<LatencyBreakdown> logs;
    logs.reserve(max_frames);

    std::cout << "[+] Warmup run (50 frames)...\n";
    for (int i = 0; i < 50; ++i) {
        cap.read(frame);
        auto feat = extract_features(frame);
        auto t = torch::from_blob(feat.data(), {1, 9}, torch::kFloat32);
        std::vector<torch::IValue> in = {t};
        module.forward(in);
    }

    std::cout << "[+] Profiling " << max_frames << " frames...\n";

    for (int i = 0; i < max_frames; ++i) {
        LatencyBreakdown lb;

        // Stage 1: Frame Capture
        auto t0 = Clock::now();
        cap.read(frame);
        auto t1 = Clock::now();
        lb.frame_cap_us = std::chrono::duration_cast<DurationUs>(t1 - t0).count();

        // Stage 2: Feature Extraction
        auto feat = extract_features(frame);
        auto t2 = Clock::now();
        lb.feature_extract_us = std::chrono::duration_cast<DurationUs>(t2 - t1).count();

        // Stage 3: Inference
        torch::Tensor input_tensor = torch::from_blob(feat.data(), {1, 9}, torch::kFloat32);
        std::vector<torch::IValue> inputs = {input_tensor};
        torch::Tensor output = module.forward(inputs).toTensor();
        torch::Tensor probs = torch::softmax(output, 1);
        float score = probs[0][1].item<float>();
        auto t3 = Clock::now();
        lb.inference_us = std::chrono::duration_cast<DurationUs>(t3 - t2).count();

        // Stage 4: Hysteresis Filter
        bool decision = filter.process(score);
        auto t4 = Clock::now();
        lb.hysteresis_us = std::chrono::duration_cast<DurationUs>(t4 - t3).count();

        // Stage 5: Relay Trigger
        relay.trigger(decision);
        auto t5 = Clock::now();
        lb.gpio_trigger_us = std::chrono::duration_cast<DurationUs>(t5 - t4).count();

        lb.total_glass_to_glass_us = std::chrono::duration_cast<DurationUs>(t5 - t0).count();
        logs.push_back(lb);
    }

    std::cout << "\n======================================================\n";
    std::cout << "        GLASS-TO-GLASS LATENCY AUDIT REPORT           \n";
    std::cout << "======================================================\n";

    std::vector<double> caps, preps, infs, hysts, gpios, totals;
    for (const auto& l : logs) {
        caps.push_back(l.frame_cap_us);
        preps.push_back(l.feature_extract_us);
        infs.push_back(l.inference_us);
        hysts.push_back(l.hysteresis_us);
        gpios.push_back(l.gpio_trigger_us);
        totals.push_back(l.total_glass_to_glass_us);
    }

    print_percentiles(caps,   "1. Frame Acquisition  ");
    print_percentiles(preps,  "2. Spatial Extraction ");
    print_percentiles(infs,   "3. CCVNN Forward Pass ");
    print_percentiles(hysts,  "4. Hysteresis Filter  ");
    print_percentiles(gpios,  "5. GPIO Relay Output  ");
    std::cout << "------------------------------------------------------\n";
    print_percentiles(totals, "TOTAL Glass-To-Glass  ");
    std::cout << "======================================================\n";

    return 0;
}
