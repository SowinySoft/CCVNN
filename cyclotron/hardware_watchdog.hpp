#ifndef HARDWARE_WATCHDOG_HPP
#define HARDWARE_WATCHDOG_HPP

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/watchdog.h>
#include <systemd/sd-daemon.h>

namespace ccvnn {

class WatchdogManager {
public:
    explicit WatchdogManager(int timeout_seconds = 10, int ping_interval_ms = 1000)
        : timeout_sec_(timeout_seconds),
          ping_interval_ms_(ping_interval_ms),
          running_(false),
          watchdog_fd_(-1) {
        last_heartbeat_ms_ = get_current_time_ms();
    }

    ~WatchdogManager() {
        stop();
    }

    bool start() {
        // 1. Initialize Linux Hardware Watchdog (/dev/watchdog)
        watchdog_fd_ = open("/dev/watchdog", O_WRONLY);
        if (watchdog_fd_ < 0) {
            std::cerr << "[WATCHDOG] Warning: Failed to open /dev/watchdog. Operating in systemd-only mode.\n";
        } else {
            // Set hardware watchdog timeout
            if (ioctl(watchdog_fd_, WDIOC_SETTIMEOUT, &timeout_sec_) < 0) {
                std::cerr << "[WATCHDOG] Warning: Failed to set hardware watchdog timeout.\n";
            } else {
                std::cout << "[WATCHDOG] Hardware watchdog initialized with " << timeout_sec_ << "s timeout.\n";
            }
        }

        // 2. Notify systemd that service is ready
        sd_notify(0, "READY=1");

        running_ = true;
        monitor_thread_ = std::thread(&WatchdogManager::monitor_loop, this);
        return true;
    }

    void stop() {
        if (!running_) return;
        running_ = false;

        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }

        // Clean shutdown: Write 'V' to /dev/watchdog to disable hardware reset before exiting gracefully
        if (watchdog_fd_ >= 0) {
            write(watchdog_fd_, "V", 1);
            close(watchdog_fd_);
            watchdog_fd_ = -1;
            std::cout << "[WATCHDOG] Hardware watchdog safely closed.\n";
        }
    }

    // Call from main processing pipeline on every successfully processed frame or V_cum telemetry update
    void kick() noexcept {
        last_heartbeat_ms_.store(get_current_time_ms(), std::memory_order_release);
    }

private:
    static uint64_t get_current_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    void monitor_loop() {
        const uint64_t max_allowed_staleness_ms = static_cast<uint64_t>(timeout_sec_) * 1000;

        while (running_) {
            uint64_t now = get_current_time_ms();
            uint64_t last_beat = last_heartbeat_ms_.load(std::memory_order_acquire);
            uint64_t elapsed = now - last_beat;

            if (elapsed < max_allowed_staleness_ms) {
                // Pipeline is healthy: Ping Linux Hardware Watchdog
                if (watchdog_fd_ >= 0) {
                    ioctl(watchdog_fd_, WDIOC_KEEPALIVE, 0);
                }

                // Ping systemd Watchdog (Notifies systemd WatchdogSec timer)
                sd_notify(0, "WATCHDOG=1");
            } else {
                // PIPELINE FREEZE DETECTED: Intentionally withhold keep-alive ping!
                std::cerr << "[WATCHDOG] CRITICAL: Pipeline freeze detected! No heartbeat for " 
                          << elapsed << " ms. Withholding keep-alive...\n";
                
                // Allow systemd / hardware watchdog timer to expire and trigger service/board reset
                sd_notify(0, "STATUS=Pipeline freeze detected - awaiting restart");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(ping_interval_ms_));
        }
    }

    int timeout_sec_;
    int ping_interval_ms_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> last_heartbeat_ms_;
    int watchdog_fd_;
    std::thread monitor_thread_;
};

} // namespace ccvnn

#endif // HARDWARE_WATCHDOG_HPP