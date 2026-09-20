#include <iostream>
#include <vector>
#include <chrono>
#include "CCVNN_V17_Extractor.hpp"

int main() {
    constexpr int WIDTH = 64;
    constexpr int HEIGHT = 64;
    constexpr int NUM_RUNS = 10000;

    std::vector<uint8_t> roi(WIDTH * HEIGHT);
    for (size_t i = 0; i < roi.size(); ++i) {
        roi[i] = static_cast<uint8_t>((i * 7 + 13) % 256);
    }

    // Warmup pass
    auto v_out = ccvnn::CCVNNV17Extractor::extract(roi.data(), WIDTH, HEIGHT);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_RUNS; ++i) {
        v_out = ccvnn::CCVNNV17Extractor::extract(roi.data(), WIDTH, HEIGHT);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / NUM_RUNS;

    std::cout << "CCVNN V17 SIMD Extractor Execution Summary:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Average Latency per 64x64 ROI: " << avg_us << " us\n";
    std::cout << "Extracted Vector [e14 B_lux, e15 C_ratio, e16 R_specular]: ["
              << v_out[14] << ", " << v_out[15] << ", " << v_out[16] << "]\n";

    if (avg_us < 500.0) {
        std::cout << "✅ Latency Target (< 500 us) PASSED!\n";
    } else {
        std::cout << "⚠️ Latency Target EXCEEDED!\n";
    }

    return 0;
}
