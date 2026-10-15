#include "dual_mode_inference.hpp"
#include <iostream>
#include <numeric>

int main() {
    try {
        std::string engine_path = "models/dual_mode_vision_int8.engine";
        ccvnn::DualModeTRTEngine engine(engine_path);

        std::cout << "[INFO] TensorRT Engine initialized. Memory buffers mapped via zero-copy." << std::endl;

        // --- Option 1: Populate Zero-Copy Host Mapped Buffers ---
        float* host_img  = engine.get_host_image_buffer();
        float* host_vcum = engine.get_host_vcum_buffer();

        // Simulate 2D normalized image frame ingestion
        std::fill_n(host_img, ccvnn::EngineDimensions::IMAGE_BYTES / sizeof(float), 0.5f);

        // Populate V_cum telemetry vector: [S_anomaly, S_cum, I_severity, C_counter, R_plc, T_latency, H_health, Z_hash]
        host_vcum[0] = 0.85f; // S_anomaly
        host_vcum[1] = 0.78f; // S_cum
        host_vcum[2] = 2.00f; // I_severity
        host_vcum[3] = 4.00f; // C_counter
        host_vcum[4] = 1.00f; // R_plc
        host_vcum[5] = 450.0f;// T_latency (us)
        host_vcum[6] = 1.00f; // H_health
        host_vcum[7] = 1024.0f;// Z_hash

        float output_logits[ccvnn::EngineDimensions::LOGITS_SIZE] = {0};

        // Execute zero-copy inference
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = engine.infer_zero_copy(output_logits);
        
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();

        if (success) {
            std::cout << "[SUCCESS] Dual-Mode Zero-Copy Inference completed in " << elapsed_us << " us\n";
            std::cout << "Class Logits: [Nominal: " << output_logits[0] 
                      << ", Minor: " << output_logits[1] 
                      << ", Critical: " << output_logits[2] 
                      << ", Hardware_Fault: " << output_logits[3] << "]\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Runtime Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}