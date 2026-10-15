#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <memory>
#include <functional>

#include "circuit_breaker.hpp"
#include "watcher_edge_daemon.hpp"
//GoogleTest suite
namespace ccvnn::testing {

// Mockable Edge Daemon for Hardware Failure Injection
class TestableWatcherEdgeDaemon {
public:
    using ModbusWriteHook = std::function<bool(uint16_t plc_actuation, uint16_t severity_code)>;

    TestableWatcherEdgeDaemon(DaemonConfig cfg, CircuitBreakerConfig cb_cfg)
        : config_(cfg), circuit_breaker_(cb_cfg), fallback_relay_(18) {
        v_cum_.fill(0.0f);
        v_cum_[V6_H_HEALTH] = 1.0f;
        fallback_relay_.initialize();
    }

    // Injectable hook replacing physical libmodbus socket calls
    void set_modbus_write_hook(ModbusWriteHook hook) {
        modbus_hook_ = std::move(hook);
    }

    CumulativeVector8 process_frame(float score, float latency_us, uint32_t roi_hash) noexcept {
        v_cum_[V0_S_ANOMALY] = score;
        v_cum_[V5_T_LATENCY] = latency_us;
        v_cum_[V7_Z_HASH]    = static_cast<float>(roi_hash);

        // Exponential Moving Average (EMA) score calculation
        v_cum_[V1_S_CUM] = (config_.ema_alpha * v_cum_[V1_S_CUM]) + 
                           ((1.0f - config_.ema_alpha) * score);

        // Simple threshold evaluation for testing
        if (v_cum_[V1_S_CUM] >= config_.anomaly_threshold) {
            v_cum_[V4_R_PLC] = 1.0f;
            state_ = DebounceState::TRIPPED;
        } else {
            v_cum_[V4_R_PLC] = 0.0f;
            state_ = DebounceState::IDLE;
        }

        sync_hardware();
        return v_cum_;
    }

    CircuitState circuit_state() const noexcept { return circuit_breaker_.state(); }
    bool is_fallback_relay_active() const noexcept { return fallback_relay_.is_active(); }
    const CumulativeVector8& current_v_cum() const noexcept { return v_cum_; }

private:
    void sync_hardware() noexcept {
        const bool relay_requested = (v_cum_[V4_R_PLC] > 0.5f);
        const uint16_t plc_actuation = relay_requested ? 1 : 0;
        const uint16_t severity_code = static_cast<uint16_t>(v_cum_[V2_I_SEVERITY]);

        if (circuit_breaker_.allow_primary_execution()) {
            bool modbus_ok = false;
            if (modbus_hook_) {
                modbus_ok = modbus_hook_(plc_actuation, severity_code);
            }

            if (modbus_ok) {
                circuit_breaker_.record_success();
                v_cum_[V6_H_HEALTH] = 1.0f;
                fallback_relay_.set_relay(false); // Disengage fallback on primary success
                return;
            } else {
                circuit_breaker_.record_failure();
            }
        }

        // --- Hardware Fallback Path ---
        v_cum_[V6_H_HEALTH] = 0.0f;
        fallback_relay_.set_relay(relay_requested);
    }

    DaemonConfig          config_;
    CumulativeVector8     v_cum_{};
    DebounceState         state_{DebounceState::IDLE};
    CircuitBreaker        circuit_breaker_;
    LocalFallbackRelay    fallback_relay_;
    ModbusWriteHook       modbus_hook_{nullptr};
};

// Fixture with short cooldown duration for fast test execution
class CircuitBreakerTest : public ::testing::Test {
protected:
    void SetUp() override {
        daemon_cfg.ema_alpha = 0.0f; // Direct score passthrough
        daemon_cfg.anomaly_threshold = 0.75f;

        // Configure fast trip parameters: 3 failures trip OPEN, 50ms cooldown, 2 success probes to CLOSE
        cb_cfg.failure_threshold = 3;
        cb_cfg.success_threshold = 2;
        cb_cfg.cooldown_period_ms = 50; 

        daemon = std::make_unique<TestableWatcherEdgeDaemon>(daemon_cfg, cb_cfg);
    }

    DaemonConfig daemon_cfg;
    CircuitBreakerConfig cb_cfg;
    std::unique_ptr<TestableWatcherEdgeDaemon> daemon;
};

// ----------------------------------------------------------------------------
// Test 1: Isolated CircuitBreaker Class Unit Verification
// ----------------------------------------------------------------------------
TEST(CircuitBreakerUnitTest, DirectStateTransitions) {
    CircuitBreakerConfig config;
    config.failure_threshold = 2;
    config.success_threshold = 2;
    config.cooldown_period_ms = 20;

    CircuitBreaker cb(config);
    EXPECT_EQ(cb.state(), CircuitState::CLOSED);

    // Single failure does not trip
    cb.record_failure();
    EXPECT_EQ(cb.state(), CircuitState::CLOSED);

    // Second failure trips to OPEN
    cb.record_failure();
    EXPECT_EQ(cb.state(), CircuitState::OPEN);
    EXPECT_FALSE(cb.allow_primary_execution());

    // Wait for cooldown
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    EXPECT_TRUE(cb.allow_primary_execution()); // Triggers transition to HALF_OPEN
    EXPECT_EQ(cb.state(), CircuitState::HALF_OPEN);

    // Two consecutive successes recover to CLOSED
    cb.record_success();
    EXPECT_EQ(cb.state(), CircuitState::HALF_OPEN);
    cb.record_success();
    EXPECT_EQ(cb.state(), CircuitState::CLOSED);
}

// ----------------------------------------------------------------------------
// Test 2: Failure Injection, Fallback Relay Activation & Health Signal
// ----------------------------------------------------------------------------
TEST_F(CircuitBreakerTest, ModbusFailureTriggersLocalFallbackAndHealthDegradation) {
    // Inject Modbus network failure (always returns false)
    daemon->set_modbus_write_hook([](uint16_t, uint16_t) { return false; });

    // Frames 1-2: Failures accumulating, Breaker remains CLOSED
    for (int i = 0; i < 2; ++i) {
        auto v_cum = daemon->process_frame(0.9f, 400.0f, 0xA1);
        EXPECT_EQ(daemon->circuit_state(), CircuitState::CLOSED);
        EXPECT_FLOAT_EQ(v_cum[V6_H_HEALTH], 0.0f); // Health dropped due to Modbus failure
        EXPECT_TRUE(daemon->is_fallback_relay_active()); // Local GPIO activated instantly
    }

    // Frame 3: Reaches failure_threshold = 3 -> Circuit Breaker trips OPEN
    auto v_cum = daemon->process_frame(0.9f, 410.0f, 0xA1);
    EXPECT_EQ(daemon->circuit_state(), CircuitState::OPEN);
    EXPECT_FLOAT_EQ(v_cum[V6_H_HEALTH], 0.0f);
    EXPECT_TRUE(daemon->is_fallback_relay_active());
}

// ----------------------------------------------------------------------------
// Test 3: HALF_OPEN Probe Failure Triggers Immediate Re-trip
// ----------------------------------------------------------------------------
TEST_F(CircuitBreakerTest, HalfOpenProbeFailureReTripsOpen) {
    bool modbus_online = false;
    daemon->set_modbus_write_hook([&](uint16_t, uint16_t) { return modbus_online; });

    // Force trip to OPEN (3 failures)
    for (int i = 0; i < 3; ++i) {
        daemon->process_frame(0.85f, 350.0f, 0xB2);
    }
    EXPECT_EQ(daemon->circuit_state(), CircuitState::OPEN);

    // Wait for cooldown period (50ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(60));

    // Next frame allows probe execution, transitioning to HALF_OPEN. Probe fails.
    daemon->process_frame(0.85f, 360.0f, 0xB2);
    
    // Failed probe in HALF_OPEN must instantly re-trip to OPEN
    EXPECT_EQ(daemon->circuit_state(), CircuitState::OPEN);
    EXPECT_TRUE(daemon->is_fallback_relay_active());
}

// ----------------------------------------------------------------------------
// Test 4: Full Recovery Flow Back to CLOSED State
// ----------------------------------------------------------------------------
TEST_F(CircuitBreakerTest, CompleteRecoveryToClosedState) {
    bool modbus_online = false;
    daemon->set_modbus_write_hook([&](uint16_t, uint16_t) { return modbus_online; });

    // Trip breaker to OPEN
    for (int i = 0; i < 3; ++i) {
        daemon->process_frame(0.95f, 300.0f, 0xC3);
    }
    EXPECT_EQ(daemon->circuit_state(), CircuitState::OPEN);

    // Wait for cooldown
    std::this_thread::sleep_for(std::chrono::milliseconds(60));

    // Restore Modbus network health
    modbus_online = true;

    // Frame 1 in HALF_OPEN: Probe Success 1/2
    auto v_cum1 = daemon->process_frame(0.95f, 300.0f, 0xC3);
    EXPECT_EQ(daemon->circuit_state(), CircuitState::HALF_OPEN);
    EXPECT_FLOAT_EQ(v_cum1[V6_H_HEALTH], 1.0f);
    EXPECT_FALSE(daemon->is_fallback_relay_active()); // Fallback disengaged

    // Frame 2 in HALF_OPEN: Probe Success 2/2 -> Fully Recovers to CLOSED
    auto v_cum2 = daemon->process_frame(0.95f, 300.0f, 0xC3);
    EXPECT_EQ(daemon->circuit_state(), CircuitState::CLOSED);
    EXPECT_FLOAT_EQ(v_cum2[V6_H_HEALTH], 1.0f);
    EXPECT_FALSE(daemon->is_fallback_relay_active());
}

} // namespace ccvnn::testing

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}