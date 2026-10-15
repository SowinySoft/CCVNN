#include "watcher_edge_daemon.hpp"
#include "circuit_breaker.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include "hardware_watchdog.hpp"
#include <csignal>
#include <chrono>

namespace ccvnn {
//


std::atomic<bool> g_shutdown(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown.store(true);
    }
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize 10-second watchdog monitor
    ccvnn::WatchdogManager watchdog(10 /* timeout_sec */, 1000 /* ping_interval_ms */);
    watchdog.start();

    std::cout << "[DAEMON] WatcherEdgeDaemon started. Processing pipeline running...\n";

    while (!g_shutdown.load()) {
        // --- Simulate DeepStream / TensorRT Pipeline Execution ---
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS

        // Reset watchdog timer on successful iteration
        watchdog.kick();
    }

    std::cout << "[DAEMON] Shutting down daemon gracefully...\n";
    watchdog.stop();
    return 0;
}

class WatcherEdgeDaemonEx {
public:
    WatcherEdgeDaemonEx(DaemonConfig cfg, CircuitBreakerConfig cb_cfg, int fallback_gpio_pin = 18)
        : config_(cfg), circuit_breaker_(cb_cfg), fallback_relay_(fallback_gpio_pin) {
        v_cum_.fill(0.0f);
        v_cum_[V6_H_HEALTH] = 1.0f;
        fallback_relay_.initialize();
    }

    CumulativeVector8 process_frame(float raw_anomaly_score, float latency_us, uint32_t roi_hash) noexcept {
        // 1. Core vector accumulation and debouncing state machine
        v_cum_[V0_S_ANOMALY] = std::clamp(raw_anomaly_score, 0.0f, 1.0f);
        v_cum_[V5_T_LATENCY] = latency_us;
        v_cum_[V7_Z_HASH]    = static_cast<float>(roi_hash);

        // EMA Recurrence: V_cum^(t) = α · V_cum^(t-1) + (1 - α) · e^(t)
        v_cum_[V1_S_CUM] = (config_.ema_alpha * v_cum_[V1_S_CUM]) + 
                           ((1.0f - config_.ema_alpha) * raw_anomaly_score);

        // Evaluate 3-state debouncer
        evaluate_state_machine();

        // 2. Hardware Actuation with Circuit Breaker Fallback
        sync_hardware_with_circuit_breaker();

        return v_cum_;
    }

private:
    void evaluate_state_machine() noexcept {
        const float score = v_cum_[V1_S_CUM];

        switch (state_) {
            case DebounceState::IDLE:
                v_cum_[V4_R_PLC] = 0.0f;
                if (score >= config_.anomaly_threshold) {
                    state_ = DebounceState::EVALUATING;
                    consecutive_count_ = 1;
                }
                break;
            case DebounceState::EVALUATING:
                v_cum_[V4_R_PLC] = 0.0f;
                if (score >= config_.anomaly_threshold) {
                    if (++consecutive_count_ >= config_.consecutive_required) {
                        state_ = DebounceState::TRIPPED;
                        v_cum_[V4_R_PLC] = 1.0f;
                    }
                } else {
                    state_ = DebounceState::IDLE;
                    consecutive_count_ = 0;
                }
                break;
            case DebounceState::TRIPPED:
                v_cum_[V4_R_PLC] = 1.0f;
                break;
        }
    }

    void sync_hardware_with_circuit_breaker() noexcept {
        const bool relay_requested = (v_cum_[V4_R_PLC] > 0.5f);
        const uint16_t plc_actuation = relay_requested ? 1 : 0;
        const uint16_t severity_code = static_cast<uint16_t>(v_cum_[V2_I_SEVERITY]);

        // Check if primary Modbus execution is allowed by Circuit Breaker
        if (circuit_breaker_.allow_primary_execution()) {
            bool modbus_ok = write_modbus_registers(plc_actuation, severity_code);

            if (modbus_ok) {
                circuit_breaker_.record_success();
                v_cum_[V6_H_HEALTH] = 1.0f;
                
                // Keep local fallback disengaged when primary network actuation succeeds
                fallback_relay_.set_relay(false);
                return;
            } else {
                // Record Modbus network/timeout fault
                circuit_breaker_.record_failure();
            }
        }

        // --- FALLBACK PATH: Modbus disconnected or Circuit Breaker OPEN ---
        v_cum_[V6_H_HEALTH] = 0.0f; // Flag health degradation
        
        // Directly drive local GPIO relay pin
        fallback_relay_.set_relay(relay_requested);
    }

    bool write_modbus_registers(uint16_t plc_actuation, uint16_t severity_code) noexcept {
        if (!mb_ctx_) return false;

        int rc1 = modbus_write_register(mb_ctx_, config_.reg_plc_coil, plc_actuation);
        int rc2 = modbus_write_register(mb_ctx_, config_.reg_severity, severity_code);

        return (rc1 != -1 && rc2 != -1);
    }

    DaemonConfig        config_;
    CumulativeVector8   v_cum_{};
    DebounceState       state_{DebounceState::IDLE};
    uint16_t            consecutive_count_{0};
    modbus_t*           mb_ctx_{nullptr};

    CircuitBreaker      circuit_breaker_;
    LocalFallbackRelay  fallback_relay_;
};
//
WatcherEdgeDaemon::WatcherEdgeDaemon(DaemonConfig config)
    : config_(config) {
    v_cum_.fill(0.0f);
    v_cum_[V6_H_HEALTH] = 1.0f; // Default health nominal
}

WatcherEdgeDaemon::~WatcherEdgeDaemon() {
    if (mb_ctx_) {
        modbus_close(mb_ctx_);
        modbus_free(mb_ctx_);
        mb_ctx_ = nullptr;
    }
}

bool WatcherEdgeDaemon::initialize_modbus() noexcept {
    mb_ctx_ = modbus_new_tcp(config_.modbus_ip.c_str(), config_.modbus_port);
    if (!mb_ctx_) {
        v_cum_[V6_H_HEALTH] = 0.0f;
        return false;
    }

    // Set connection timeout (100 ms) for ultra-low latency response
    modbus_set_response_timeout(mb_ctx_, 0, 100000);

    if (modbus_connect(mb_ctx_) == -1) {
        v_cum_[V6_H_HEALTH] = 0.0f;
        modbus_connected_.store(false);
        return false;
    }

    modbus_connected_.store(true);
    v_cum_[V6_H_HEALTH] = 1.0f;
    return true;
}

CumulativeVector8 WatcherEdgeDaemon::process_frame(float raw_anomaly_score, 
                                                float latency_us, 
                                                uint32_t roi_hash) noexcept {
    // 1. Record instantaneous inputs
    v_cum_[V0_S_ANOMALY] = std::clamp(raw_anomaly_score, 0.0f, 1.0f);
    v_cum_[V5_T_LATENCY] = latency_us;
    v_cum_[V7_Z_HASH]    = static_cast<float>(roi_hash);

    // 2. Compute Temporal EMA Recurrence: V_cum^(t) = α · V_cum^(t-1) + (1 - α) · e^(t)
    update_ema(raw_anomaly_score);

    // 3. Compute Severity Tier
    v_cum_[V2_I_SEVERITY] = static_cast<float>(calculate_severity_tier(v_cum_[V1_S_CUM]));

    // 4. Stateful Temporal Debouncing State Machine Execution
    evaluate_state_machine();

    // 5. Hardware Actuation via Modbus TCP Registers (40001 / 40002)
    sync_modbus_hardware();

    return v_cum_;
}

void WatcherEdgeDaemon::update_ema(float raw_anomaly_score) noexcept {
    const float alpha = config_.ema_alpha;
    const float prev_cum = v_cum_[V1_S_CUM];
    
    // V_cum^(t) = α · V_cum^(t-1) + (1 - α) · e^(t)
    v_cum_[V1_S_CUM] = (alpha * prev_cum) + ((1.0f - alpha) * raw_anomaly_score);
}

void WatcherEdgeDaemon::evaluate_state_machine() noexcept {
    const float current_score = v_cum_[V1_S_CUM];

    switch (state_) {
        case DebounceState::IDLE:
            v_cum_[V4_R_PLC] = 0.0f;
            if (current_score >= config_.anomaly_threshold) {
                state_ = DebounceState::EVALUATING;
                consecutive_count_ = 1;
            } else {
                consecutive_count_ = 0;
            }
            break;

        case DebounceState::EVALUATING:
            v_cum_[V4_R_PLC] = 0.0f;
            if (current_score >= config_.anomaly_threshold) {
                consecutive_count_++;
                if (consecutive_count_ >= config_.consecutive_required) {
                    state_ = DebounceState::TRIPPED;
                    v_cum_[V4_R_PLC] = 1.0f; // Trip PLC Relay
                }
            } else {
                // False alarm decay: drop back to IDLE
                consecutive_count_ = 0;
                state_ = DebounceState::IDLE;
            }
            break;

        case DebounceState::TRIPPED:
            // Remain in TRIPPED state until explicit hardware/software reset
            v_cum_[V4_R_PLC] = 1.0f;
            break;
    }

    // Update debouncing confidence ratio [0.0, 1.0]
    v_cum_[V3_C_DEBOUNCE] = static_cast<float>(consecutive_count_) / 
                             static_cast<float>(config_.consecutive_required);
}

void WatcherEdgeDaemon::sync_modbus_hardware() noexcept {
    if (!modbus_connected_.load() || !mb_ctx_) {
        v_cum_[V6_H_HEALTH] = 0.0f;
        return;
    }

    const uint16_t plc_actuation = static_cast<uint16_t>(v_cum_[V4_R_PLC]);
    const uint16_t severity_code = static_cast<uint16_t>(v_cum_[V2_I_SEVERITY]);

    // Write Register 40001: PLC Actuation Relay Coil
    int rc1 = modbus_write_register(mb_ctx_, config_.reg_plc_coil, plc_actuation);
    // Write Register 40002: Severity Tier Code
    int rc2 = modbus_write_register(mb_ctx_, config_.reg_severity, severity_code);

    if (rc1 == -1 || rc2 == -1) {
        v_cum_[V6_H_HEALTH] = 0.0f; // Flag health degradation on communication error
        modbus_connected_.store(false);
    } else {
        v_cum_[V6_H_HEALTH] = 1.0f;
    }
}

uint8_t WatcherEdgeDaemon::calculate_severity_tier(float score) const noexcept {
    if (score < 0.25f) return 0; // Nominal
    if (score < 0.50f) return 1; // Low
    if (score < 0.75f) return 2; // Moderate
    return 3;                    // Critical
}

void WatcherEdgeDaemon::reset_trip_state() noexcept {
    state_ = DebounceState::IDLE;
    consecutive_count_ = 0;
    v_cum_[V4_R_PLC] = 0.0f;
    v_cum_[V1_S_CUM] = 0.0f;
    v_cum_[V3_C_DEBOUNCE] = 0.0f;
    sync_modbus_hardware();
}

} // namespace ccvnn