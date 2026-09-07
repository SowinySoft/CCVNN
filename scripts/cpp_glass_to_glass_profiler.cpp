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
#include <iomanip>
#include <cctype>

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

class GPIORelay {
private:
    int gpio_pin_;
    bool hardware_enabled_;

public:
    GPIORelay(int pin = 18, bool enable_hw = false) : gpio_pin_(pin), hardware_enabled_(enable_hw) {}

    void trigger(bool pass) {
        if (hardware_enabled_) {
            std::ofstream gpio_val("/sys/class/gpio/gpio" + std::to_string(gpio_pin_) + "/value");
            if (gpio_val.is_open()) {
                gpio_val << (pass ? "1" : "0");
            }
        } else {
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
    if (frame.empty()) return std::vector<float>(9, 0.0f);

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

struct Stats { double p50; double p95; double p99; };

Stats calc_stats(std::vector<double> v) {
    if (v.empty()) return {0.0, 0.0, 0.0};
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return { v[n * 0.50] / 1000.0, v[n * 0.95] / 1000.0, v[n * 0.99] / 1000.0 };
}

void export_json(const std::vector<LatencyBreakdown>& logs, const std::string& filename) {
    std::vector<double> caps, preps, infs, hysts, gpios, totals;
    for (const auto& l : logs) {
        caps.push_back(l.frame_cap_us);
        preps.push_back(l.feature_extract_us);
        infs.push_back(l.inference_us);
        hysts.push_back(l.hysteresis_us);
        gpios.push_back(l.gpio_trigger_us);
        totals.push_back(l.total_glass_to_glass_us);
    }

    Stats s_cap = calc_stats(caps);
    Stats s_prep = calc_stats(preps);
    Stats s_inf = calc_stats(infs);
    Stats s_hyst = calc_stats(hysts);
    Stats s_gpio = calc_stats(gpios);
    Stats s_tot = calc_stats(totals);

    std::ofstream out(filename);
    out << std::fixed << std::setprecision(4);
    out << "{\n";
    out << "  \"metadata\": { \"frames_profiled\": " << logs.size() << " },\n";
    out << "  \"summary_ms\": {\n";
    out << "    \"frame_acquisition\": {\"p50\":" << s_cap.p50 << ",\"p95\":" << s_cap.p95 << ",\"p99\":" << s_cap.p99 << "},\n";
    out << "    \"spatial_extraction\": {\"p50\":" << s_prep.p50 << ",\"p95\":" << s_prep.p95 << ",\"p99\":" << s_prep.p99 << "},\n";
    out << "    \"ccvnn_inference\": {\"p50\":" << s_inf.p50 << ",\"p95\":" << s_inf.p95 << ",\"p99\":" << s_inf.p99 << "},\n";
    out << "    \"hysteresis_filter\": {\"p50\":" << s_hyst.p50 << ",\"p95\":" << s_hyst.p95 << ",\"p99\":" << s_hyst.p99 << "},\n";
    out << "    \"gpio_relay\": {\"p50\":" << s_gpio.p50 << ",\"p95\":" << s_gpio.p95 << ",\"p99\":" << s_gpio.p99 << "},\n";
    out << "    \"total_glass_to_glass\": {\"p50\":" << s_tot.p50 << ",\"p95\":" << s_tot.p95 << ",\"p99\":" << s_tot.p99 << "}\n";
    out << "  },\n";
    out << "  \"frame_totals_us\": [";
    for (size_t i = 0; i < totals.size(); ++i) {
        out << totals[i] << (i + 1 < totals.size() ? "," : "");
    }
    out << "]\n";
    out << "}\n";
    out.close();

    std::cout << "[✓] Exported benchmark JSON to: " << filename << std::endl;
}

bool is_number(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./ccvnn_glass_profiler <model_path> [num_frames=500] [camera_id_or_video_path=0]\n";
        return -1;
    }

    std::string model_path = argv[1];
    int max_frames = (argc >= 3) ? std::stoi(argv[2]) : 500;
    std::string source = (argc >= 4) ? argv[3] : "0";

    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();

    cv::VideoCapture cap;
    if (is_number(source)) {
        cap.open(std::stoi(source));
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        std::cerr << "[!] Camera/Video source failed to initialize: " << source << "\n";
        return -1;
    }

    HysteresisFilter filter;
    GPIORelay relay(18, false);
    cv::Mat frame;

    std::vector<LatencyBreakdown> logs;
    logs.reserve(max_frames);

    std::cout << "[+] Warmup run (10 frames)...\n";
    for (int i = 0; i < 10; ++i) {
        if (!cap.read(frame)) break;
        auto feat = extract_features(frame);
        auto t = torch::from_blob(feat.data(), {1, 9}, torch::kFloat32);
        std::vector<torch::IValue> in = {t};
        module.forward(in);
    }

    std::cout << "[+] Profiling " << max_frames << " frames...\n";

    for (int i = 0; i < max_frames; ++i) {
        LatencyBreakdown lb;

        auto t0 = Clock::now();
        if (!cap.read(frame)) {
            // Loop video feed if file ends before max_frames
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap.read(frame)) break;
        }
        auto t1 = Clock::now();
        lb.frame_cap_us = std::chrono::duration_cast<DurationUs>(t1 - t0).count();

        auto feat = extract_features(frame);
        auto t2 = Clock::now();
        lb.feature_extract_us = std::chrono::duration_cast<DurationUs>(t2 - t1).count();

        torch::Tensor input_tensor = torch::from_blob(feat.data(), {1, 9}, torch::kFloat32);
        std::vector<torch::IValue> inputs = {input_tensor};
        torch::Tensor output = module.forward(inputs).toTensor();
        torch::Tensor probs = torch::softmax(output, 1);
        float score = probs[0][1].item<float>();
        auto t3 = Clock::now();
        lb.inference_us = std::chrono::duration_cast<DurationUs>(t3 - t2).count();

        bool decision = filter.process(score);
        auto t4 = Clock::now();
        lb.hysteresis_us = std::chrono::duration_cast<DurationUs>(t4 - t3).count();

        relay.trigger(decision);
        auto t5 = Clock::now();
        lb.gpio_trigger_us = std::chrono::duration_cast<DurationUs>(t5 - t4).count();

        lb.total_glass_to_glass_us = std::chrono::duration_cast<DurationUs>(t5 - t0).count();
        logs.push_back(lb);
    }

    export_json(logs, "benchmark_results.json");
    return 0;
}
