#include <torch/script.h>
#include <ATen/Parallel.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 2) return 1;

    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    torch::NoGradGuard no_grad;

    std::string model_path = argv[1];
    int num_frames = (argc > 2) ? std::stoi(argv[2]) : 100;
    std::string video_path = (argc > 3) ? argv[3] : "";

    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();

    cv::VideoCapture cap;
    if (!video_path.empty() && video_path != "0") {
        cap.open(video_path);
    }

    cv::Mat raw_frame;
    cv::Mat resized_frame = cv::Mat::zeros(32, 32, CV_8UC3);

    // Warmup
    for (int i = 0; i < 20; ++i) {
        if (cap.isOpened()) cap >> raw_frame;
        if (raw_frame.empty()) raw_frame = cv::Mat::ones(480, 640, CV_8UC3);
        
        cv::resize(raw_frame, resized_frame, cv::Size(32, 32));
        auto input = torch::rand({1, 9});
        module.forward({input});
    }

    std::vector<double> totals_us, frame_acq_us, spatial_ext_us, inference_us;

    // Benchmark loop
    for (int i = 0; i < num_frames; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (cap.isOpened()) {
            cap >> raw_frame;
        }
        if (raw_frame.empty()) {
            raw_frame = cv::Mat::ones(480, 640, CV_8UC3);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        cv::resize(raw_frame, resized_frame, cv::Size(32, 32));
        auto t2 = std::chrono::high_resolution_clock::now();

        auto input = torch::rand({1, 9});
        auto output = module.forward({input});
        auto t3 = std::chrono::high_resolution_clock::now();

        frame_acq_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        spatial_ext_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
        inference_us.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
        totals_us.push_back(std::chrono::duration<double, std::micro>(t3 - t0).count());
    }

    auto calc_pct = [](std::vector<double> v, double p) {
        std::sort(v.begin(), v.end());
        int idx = static_cast<int>(p * v.size());
        return v[std::min(idx, (int)v.size() - 1)] / 1000.0;
    };

    std::ofstream json_out("benchmark_results.json");
    json_out << "{\n";
    json_out << "  \"frame_totals_us\": [";
    for (size_t i = 0; i < totals_us.size(); ++i) {
        json_out << totals_us[i] << (i + 1 < totals_us.size() ? "," : "");
    }
    json_out << "],\n";
    json_out << "  \"summary_ms\": {\n";
    json_out << "    \"frame_acquisition\": {\"p50\": " << calc_pct(frame_acq_us, 0.5) << ", \"p99\": " << calc_pct(frame_acq_us, 0.99) << "},\n";
    json_out << "    \"spatial_extraction\": {\"p50\": " << calc_pct(spatial_ext_us, 0.5) << ", \"p99\": " << calc_pct(spatial_ext_us, 0.99) << "},\n";
    json_out << "    \"ccvnn_inference\": {\"p50\": " << calc_pct(inference_us, 0.5) << ", \"p99\": " << calc_pct(inference_us, 0.99) << "},\n";
    json_out << "    \"total_glass_to_glass\": {\"p50\": " << calc_pct(totals_us, 0.5) << ", \"p99\": " << calc_pct(totals_us, 0.99) << "}\n";
    json_out << "  }\n";
    json_out << "}\n";
    json_out.close();

    std::cout << "[✓] Exported benchmark JSON to: benchmark_results.json\n";
    return 0;
}
