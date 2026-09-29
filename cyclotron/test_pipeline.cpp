#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <numeric>

#include "v17_feature_extractor.hpp"
#include "watcher_edge_daemon.hpp"

namespace ccvnn::testing {

class PipelineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Configure Extractor with production default settings
        ExtractorConfig extractor_cfg;
        extractor_cfg.light_threshold = 180;
        extractor_cfg.dark_threshold = 50;
        extractor_cfg.specular_threshold = 252;
        extractor_cfg.enable_openmp = true;
        extractor = std::make_unique<FeatureExtractorV17>(extractor_cfg);

        // Configure Daemon with a 5-frame consecutive trip threshold
        DaemonConfig daemon_cfg;
        daemon_cfg.ema_alpha = 0.80f;
        daemon_cfg.anomaly_threshold = 0.75f;
        daemon_cfg.consecutive_required = 5;
        daemon_cfg.modbus_ip = "127.0.0.1"; // Local loopback mock/daemon
        daemon_cfg.modbus_port = 1502;
        daemon = std::make_unique<WatcherEdgeDaemon>(daemon_cfg);
    }

    // Helper: Synthesize synthetic frame (nominal vs anomalous surface defect)
    cv::Mat generate_synthetic_frame(int height, int width, bool add_defect) {
        cv::Mat frame = cv::Mat(height, width, CV_8UC3, cv::Scalar(120, 120, 120));
        
        if (add_defect) {
            // Draw high-contrast specular glare and irregular dark crack
            cv::rectangle(frame, cv::Point(10, 10), cv::Point(30, 30), cv::Scalar(255, 255, 255), -1);
            cv::line(frame, cv::Point(5, 50), cv::Point(width - 5, 50), cv::Scalar(10, 10, 10), 4);
            cv::circle(frame, cv::Point(width / 2, height / 2), 15, cv::Scalar(255, 255, 255), -1);
        }
        return frame;
    }

    // Mock Inference Backend: Maps 17D feature vector to raw scalar score
    float mock_inference_engine(const InputVector17& v_in) {
        // Heuristic: weighted combination of specular, contrast, and edge density
        float score = (v_in[E16_R_SPECULAR] * 0.40f) + 
                      (v_in[E0_S_1D] * 0.30f) + 
                      (v_in[E15_C_RATIO] * 0.30f);
        return std::clamp(score * 2.5f, 0.0f, 1.0f);
    }

    std::unique_ptr<FeatureExtractorV17> extractor;
    std::unique_ptr<WatcherEdgeDaemon> daemon;
};

// ----------------------------------------------------------------------------
// Test 1: Feature Extractor Dynamic Shape & Bounds Verification
// ----------------------------------------------------------------------------
TEST_F(PipelineIntegrationTest, ExtractorDynamicShapeBounds) {
    cv::Mat frame_hd = generate_synthetic_frame(1080, 1920, false);
    cv::Rect roi(100, 100, 256, 256);

    auto start_time = std::chrono::high_resolution_clock::now();
    InputVector17 v_in = extractor->extract(frame_hd, roi);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // Verify bounded values across 17 dimensions
    for (size_t i = 0; i < VECTOR_DIM_V17; ++i) {
        EXPECT_GE(v_in[i], -1.0f) << "Feature dimension " << i << " below lower bound!";
        EXPECT_LE(v_in[i], 1.0f)  << "Feature dimension " << i << " above upper bound!";
    }

    // Performance threshold: single frame extraction must be sub-millisecond (< 1000 µs)
    EXPECT_LT(elapsed_us, 1000);
}

// ----------------------------------------------------------------------------
// Test 2: Dynamic Batch Processing Contiguous Buffer Packing
// ----------------------------------------------------------------------------
TEST_F(PipelineIntegrationTest, ExtractorDynamicBatching) {
    cv::Mat frame = generate_synthetic_frame(720, 1280, true);
    std::vector<cv::Rect> rois = {
        cv::Rect(0, 0, 128, 128),
        cv::Rect(200, 200, 256, 256),
        cv::Rect(400, 100, 64, 64),
        cv::Rect(10, 10, 500, 500)
    };

    const size_t batch_size = rois.size();
    std::vector<float> tensor_buffer(batch_size * VECTOR_DIM_V17, 0.0f);

    extractor->extract_batch(frame, rois, tensor_buffer.data());

    // Verify non-zero feature extraction across all batch offset strides
    for (size_t b = 0; b < batch_size; ++b) {
        const float* offset = tensor_buffer.data() + (b * VECTOR_DIM_V17);
        float sum = std::accumulate(offset, offset + VECTOR_DIM_V17, 0.0f);
        EXPECT_GT(std::abs(sum), 0.0f) << "Batch index " << b << " resulted in zero vector!";
    }
}

// ----------------------------------------------------------------------------
// Test 3: Nominal Pipeline Flow (No State Trip)
// ----------------------------------------------------------------------------
TEST_F(PipelineIntegrationTest, NominalPipelineNoTrip) {
    cv::Mat frame_nominal = generate_synthetic_frame(480, 640, false);
    cv::Rect roi(50, 50, 200, 200);

    for (int i = 0; i < 10; ++i) {
        InputVector17 v_in = extractor->extract(frame_nominal, roi);
        float score = mock_inference_engine(v_in);
        
        CumulativeVector8 v_cum = daemon->process_frame(score, 450.0f, 0xABC123);

        EXPECT_EQ(daemon->current_state(), DebounceState::IDLE);
        EXPECT_EQ(v_cum[V4_R_PLC], 0.0f) << "PLC Relay tripped unexpectedly on nominal frame!";
        EXPECT_LT(v_cum[V1_S_CUM], 0.75f);
    }
}

// ----------------------------------------------------------------------------
// Test 4: Anomaly Spike & State Machine Trip Execution
// ----------------------------------------------------------------------------
TEST_F(PipelineIntegrationTest, AnomalySpikeDebounceAndTrip) {
    cv::Mat frame_defect = generate_synthetic_frame(480, 640, true);
    cv::Rect roi(0, 0, 480, 480);

    InputVector17 v_in = extractor->extract(frame_defect, roi);
    float score = mock_inference_engine(v_in);
    EXPECT_GE(score, 0.75f) << "Defective frame score below anomaly threshold!";

    // Frame 1: Transitions from IDLE -> EVALUATING
    CumulativeVector8 v_cum = daemon->process_frame(score, 500.0f, 0xDEADBEEF);
    EXPECT_EQ(daemon->current_state(), DebounceState::EVALUATING);
    EXPECT_EQ(v_cum[V4_R_PLC], 0.0f);
    EXPECT_FLOAT_EQ(v_cum[V3_C_DEBOUNCE], 1.0f / 5.0f);

    // Frames 2-4: Remain in EVALUATING, incrementing debounce ratio
    for (int f = 2; f <= 4; ++f) {
        v_cum = daemon->process_frame(score, 510.0f, 0xDEADBEEF);
        EXPECT_EQ(daemon->current_state(), DebounceState::EVALUATING);
        EXPECT_EQ(v_cum[V4_R_PLC], 0.0f);
    }

    // Frame 5: Reaches N_consec = 5 -> Transitions to TRIPPED
    v_cum = daemon->process_frame(score, 495.0f, 0xDEADBEEF);
    EXPECT_EQ(daemon->current_state(), DebounceState::TRIPPED);
    EXPECT_EQ(v_cum[V4_R_PLC], 1.0f) << "PLC Relay failed to trip after N_consec frames!";
    EXPECT_FLOAT_EQ(v_cum[V3_C_DEBOUNCE], 1.0f);
    EXPECT_EQ(v_cum[V2_I_SEVERITY], 3.0f); // Critical severity
}

// ----------------------------------------------------------------------------
// Test 5: False Alarm Decay & Reset Verification
// ----------------------------------------------------------------------------
TEST_F(PipelineIntegrationTest, FalseAlarmDecayAndManualReset) {
    cv::Mat frame_defect = generate_synthetic_frame(480, 640, true);
    cv::Mat frame_nominal = generate_synthetic_frame(480, 640, false);
    cv::Rect roi(0, 0, 480, 480);

    float defect_score = mock_inference_engine(extractor->extract(frame_defect, roi));
    float nominal_score = mock_inference_engine(extractor->extract(frame_nominal, roi));

    // Push 2 defect frames (EVALUATING)
    daemon->process_frame(defect_score, 500.0f, 0x111);
    daemon->process_frame(defect_score, 500.0f, 0x111);
    EXPECT_EQ(daemon->current_state(), DebounceState::EVALUATING);

    // Push nominal frame -> Should decay state back to IDLE
    CumulativeVector8 v_cum = daemon->process_frame(nominal_score, 400.0f, 0x111);
    EXPECT_EQ(daemon->current_state(), DebounceState::IDLE);
    EXPECT_EQ(v_cum[V4_R_PLC], 0.0f);
    EXPECT_EQ(v_cum[V3_C_DEBOUNCE], 0.0f);

    // Trigger full trip and verify manual state reset
    for (int i = 0; i < 5; ++i) {
        daemon->process_frame(defect_score, 500.0f, 0x222);
    }
    EXPECT_EQ(daemon->current_state(), DebounceState::TRIPPED);

    daemon->reset_trip_state();
    EXPECT_EQ(daemon->current_state(), DebounceState::IDLE);
    EXPECT_EQ(daemon->current_v_cum()[V4_R_PLC], 0.0f);
}

} // namespace ccvnn::testing

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}