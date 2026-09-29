#ifndef WATCHER_EDGE_DAEMON_HPP
#define WATCHER_EDGE_DAEMON_HPP

#include <array>
#include <cstdint>
#include <string>
#include <atomic>
#include <modbus/modbus.h> // libmodbus standard C API

namespace ccvnn {

constexpr size_t VECTOR_DIM_VCUM = 8;
using CumulativeVector8 = std::array<float, VECTOR_DIM_VCUM>;

enum VCumIndex : uint8_t {
    V0_S_ANOMALY  = 0, // Instantaneous single-frame defect probability
    V1_S_CUM      = 1, // Exponential Moving Average (EMA) accumulated score
    V2_I_SEVERITY = 2, // Severity Tier {0: Nominal, 1: Low, 2: Moderate, 3: Critical}
    V3_C_DEBOUNCE = 3, // Consecutive frame confidence ratio [0.0, 1.0]
    V4_R_PLC      = 4, // Hardware actuation flag {0.0, 1.0}
    V5_T_LATENCY  = 5, // Glass-to-glass latency (microseconds)
    V6_H_HEALTH   = 6, // Daemon & MQTT connection health state {1.0 = healthy}
    V7_Z_HASH     = 7  // ROI spatial inspection verification hash
};

enum class DebounceState : uint8_t {
    IDLE       = 0,
    EVALUATING = 1,
    TRIPPED    = 2
};

struct DaemonConfig {
    float    ema_alpha                 = 0.80f; // EMA decay factor α ∈ [0, 1]
    float    anomaly_threshold         = 0.75f; // Threshold to enter EVALUATING state
    uint16_t consecutive_required     = 5;     // Frame count requirement N_consec
    std::string modbus_ip             = "192.168.1.100";
    int      modbus_port               = 502;
    uint16_t reg_plc_coil              = 40001; // Modbus Register 40001: Hardware Relay
    uint16_t reg_severity              = 40002; // Modbus Register 40002: Hazard Severity
};

class WatcherEdgeDaemon {
public:
    explicit WatcherEdgeDaemon(DaemonConfig config = DaemonConfig{});
    ~WatcherEdgeDaemon();

    // Disable copy/move semantics for real-time safety
    WatcherEdgeDaemon(const WatcherEdgeDaemon&) = delete;
    WatcherEdgeDaemon& operator=(const WatcherEdgeDaemon&) = delete;

    // Connect to Modbus TCP PLC Controller
    bool initialize_modbus() noexcept;

    // Core deterministic process frame step
    CumulativeVector8 process_frame(float raw_anomaly_score, 
                                    float latency_us, 
                                    uint32_t roi_hash) noexcept;

    // Force reset state machine from TRIPPED -> IDLE
    void reset_trip_state() noexcept;

    // Accessors
    DebounceState current_state() const noexcept { return state_; }
    const CumulativeVector8& current_v_cum() const noexcept { return v_cum_; }

private:
    DaemonConfig      config_;
    CumulativeVector8 v_cum_{};
    DebounceState     state_{DebounceState::IDLE};
    uint16_t          consecutive_count_{0};
    
    modbus_t*         mb_ctx_{nullptr};
    std::atomic<bool> modbus_connected_{false};

    // Internal pipeline steps
    void update_ema(float raw_anomaly_score) noexcept;
    void evaluate_state_machine() noexcept;
    void sync_modbus_hardware() noexcept;
    uint8_t calculate_severity_tier(float score) const noexcept;
	
// Inside ccvnn::WatcherEdgeDaemon class definition
private:
    std::unique_ptr<MqttTelemetryExporter> exporter_{nullptr};

public:
    void enable_telemetry(const MqttConfig& mqtt_cfg) noexcept {
        exporter_ = std::make_unique<MqttTelemetryExporter>(mqtt_cfg);
        exporter_->start();
    }

    CumulativeVector8 process_frame(float raw_anomaly_score, float latency_us, uint32_t roi_hash) noexcept {
        // ... Core Modbus & state machine evaluation ...

        if (exporter_) {
            uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();
            exporter_->enqueue_telemetry(v_cum_, state_, now_us);
        }

        return v_cum_;
    }
};

} // namespace ccvnn

#endif // WATCHER_EDGE_DAEMON_HPP