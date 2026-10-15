#ifndef MQTT_TELEMETRY_EXPORTER_HPP
#define MQTT_TELEMETRY_EXPORTER_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <mosquitto.h>

#include "watcher_edge_daemon.hpp"

namespace ccvnn {

struct MqttConfig {
    std::string broker_host      = "127.0.0.1";
    int         broker_port      = 1883;
    std::string client_id        = "watcher_edge_node_01";
    std::string topic_prefix     = "ccvnn/edge/v17";
    int         qos              = 0; // QoS 0 for zero-latency telemetry overhead
    size_t      max_queue_size   = 1000;
    uint32_t    publish_rate_hz  = 30; // Throttled telemetry rate to conserve bandwidth
};

class MqttTelemetryExporter {
public:
    explicit MqttTelemetryExporter(MqttConfig config = MqttConfig{});
    ~MqttTelemetryExporter();

    // Non-blocking queue insertion called directly from process_frame loop
    void enqueue_telemetry(const CumulativeVector8& v_cum, 
                           DebounceState state, 
                           uint64_t timestamp_us) noexcept;

    bool start() noexcept;
    void stop() noexcept;

private:
    struct TelemetryPayload {
        CumulativeVector8 v_cum;
        DebounceState     state;
        uint64_t          timestamp_us;
    };

    MqttConfig              config_;
    struct mosquitto*       mosq_{nullptr};
    std::atomic<bool>       running_{false};
    std::thread             worker_thread_;

    std::queue<TelemetryPayload> queue_;
    std::mutex                   queue_mutex_;
    std::condition_variable      cv_;

    void worker_loop() noexcept;
    std::string serialize_json(const TelemetryPayload& payload) const noexcept;
};

} // namespace ccvnn

#endif // MQTT_TELEMETRY_EXPORTER_HPP