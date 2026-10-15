#ifndef CIRCUIT_BREAKER_HPP
#define CIRCUIT_BREAKER_HPP

#include <chrono>
#include <cstdint>
#include <atomic>
#include <functional>

namespace ccvnn {

enum class CircuitState : uint8_t {
    CLOSED    = 0, // Normal operation: Route requests to primary Modbus TCP
    OPEN      = 1, // Fault state: Skip Modbus, execute local hardware fallback
    HALF_OPEN = 2  // Recovery probe: Test primary Modbus with limited requests
};

struct CircuitBreakerConfig {
    uint32_t failure_threshold     = 3;     // Consecutive Modbus errors before tripping OPEN
    uint32_t success_threshold     = 2;     // Consecutive successful probes in HALF_OPEN to close
    uint32_t cooldown_period_ms   = 5000;  // Time to remain in OPEN before probing (HALF_OPEN)
};

class CircuitBreaker {
public:
    explicit CircuitBreaker(CircuitBreakerConfig config = CircuitBreakerConfig{})
        : config_(config), state_(CircuitState::CLOSED) {}

    // Determines whether primary execution (Modbus TCP) should be attempted
    bool allow_primary_execution() noexcept {
        const auto now = std::chrono::steady_clock::now();

        if (state_ == CircuitState::OPEN) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_state_change_
            ).count();

            if (elapsed_ms >= config_.cooldown_period_ms) {
                transition_to(CircuitState::HALF_OPEN);
                return true; // Allow probe execution
            }
            return false; // Skip primary execution, use local fallback
        }
        return true; // CLOSED or HALF_OPEN
    }

    // Call on primary execution success
    void record_success() noexcept {
        consecutive_failures_ = 0;
        if (state_ == CircuitState::HALF_OPEN) {
            consecutive_successes_++;
            if (consecutive_successes_ >= config_.success_threshold) {
                transition_to(CircuitState::CLOSED);
            }
        }
    }

    // Call on primary execution failure
    void record_failure() noexcept {
        consecutive_successes_ = 0;
        consecutive_failures_++;

        if (state_ == CircuitState::CLOSED && consecutive_failures_ >= config_.failure_threshold) {
            transition_to(CircuitState::OPEN);
        } else if (state_ == CircuitState::HALF_OPEN) {
            // Immediate re-trip to OPEN if probe fails
            transition_to(CircuitState::OPEN);
        }
    }

    CircuitState state() const noexcept { return state_; }

private:
    void transition_to(CircuitState new_state) noexcept {
        state_ = new_state;
        last_state_change_ = std::chrono::steady_clock::now();
        consecutive_failures_ = 0;
        consecutive_successes_ = 0;
    }

    CircuitBreakerConfig config_;
    CircuitState state_{CircuitState::CLOSED};
    uint32_t consecutive_failures_{0};
    uint32_t consecutive_successes_{0};
    std::chrono::steady_clock::time_point last_state_change_{std::chrono::steady_clock::now()};
};

// Interface for hardware-level local fallback relay actuation (e.g. sysfs GPIO / libgpiod)
class LocalFallbackRelay {
public:
    explicit LocalFallbackRelay(int gpio_pin = 18) : gpio_pin_(gpio_pin) {}

    bool initialize() noexcept {
        // Platform specific: Initialize sysfs GPIO or libgpiod line for sub-millisecond local switching
        initialized_ = true;
        return true;
    }

    void set_relay(bool active) noexcept {
        if (!initialized_) return;
        // Low-latency direct pin state write
        relay_active_ = active;
    }

    bool is_active() const noexcept { return relay_active_; }

private:
    int gpio_pin_;
    bool initialized_{false};
    bool relay_active_{false};
};

} // namespace ccvnn

#endif // CIRCUIT_BREAKER_HPP