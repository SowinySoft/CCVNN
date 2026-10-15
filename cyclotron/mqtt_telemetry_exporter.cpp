#include "mqtt_telemetry_exporter.hpp"
#include <cstdio>
#include <iostream>

namespace ccvnn {

MqttTelemetryExporter::MqttTelemetryExporter(MqttConfig config)
    : config_(config) {
    mosquitto_lib_init();
    mosq_ = mosquitto_new(config_.client_id.c_str(), true, nullptr);
}

MqttTelemetryExporter::~MqttTelemetryExporter() {
    stop();
    if (mosq_) {
        mosquitto_destroy(mosq_);
        mosq_ = nullptr;
    }
    mosquitto_lib_cleanup();
}

bool MqttTelemetryExporter::start() noexcept {
    if (!mosq_) return false;

    int rc = mosquitto_connect(mosq_, config_.broker_host.c_str(), config_.broker_port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        std::cerr << "[MQTT Exporter] Failed to connect to broker: " 
                  << mosquitto_strerror(rc) << std::endl;
        return false;
    }

    running_.store(true);
    worker_thread_ = std::thread(&MqttTelemetryExporter::worker_loop, this);
    return true;
}

void MqttTelemetryExporter::stop() noexcept {
    if (running_.exchange(false)) {
        cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        if (mosq_) {
            mosquitto_disconnect(mosq_);
        }
    }
}

void MqttTelemetryExporter::enqueue_telemetry(const CumulativeVector8& v_cum, 
                                             DebounceState state, 
                                             uint64_t timestamp_us) noexcept {
    if (!running_.load()) return;

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (queue_.size() >= config_.max_queue_size) {
        queue_.pop(); // Drop oldest payload to maintain bounded memory latency
    }
    queue_.push({v_cum, state, timestamp_us});
    cv_.notify_one();
}

void MqttTelemetryExporter::worker_loop() noexcept {
    const std::string topic = config_.topic_prefix + "/" + config_.client_id + "/v_cum";

    while (running_.load()) {
        TelemetryPayload payload;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { 
                return !queue_.empty() || !running_.load(); 
            });

            if (!running_.load() && queue_.empty()) break;

            payload = queue_.front();
            queue_.pop();
        }

        std::string json_str = serialize_json(payload);

        int rc = mosquitto_publish(mosq_, nullptr, topic.c_str(),
                                   static_cast<int>(json_str.size()),
                                   json_str.c_str(), config_.qos, false);

        if (rc != MOSQ_ERR_SUCCESS) {
            std::cerr << "[MQTT Exporter] Publish failed: " << mosquitto_strerror(rc) << std::endl;
            mosquitto_reconnect(mosq_);
        }

        mosquitto_loop(mosq_, 0, 1);
    }
}

std::string MqttTelemetryExporter::serialize_json(const TelemetryPayload& payload) const noexcept {
    char buffer[512];
    const auto& v = payload.v_cum;

    // Fast zero-allocation JSON serialization
    int n = snprintf(buffer, sizeof(buffer),
        "{"
          "\"ts_us\":%lu,"
          "\"state\":%u,"
          "\"v_cum\":{"
            "\"S_anomaly\":%.4f,"
            "\"S_cum\":%.4f,"
            "\"I_severity\":%u,"
            "\"C_debounce\":%.4f,"
            "\"R_plc\":%u,"
            "\"T_latency_us\":%.1f,"
            "\"H_health\":%.1f,"
            "\"Z_hash\":%u"
          "}"
        "}",
        payload.timestamp_us,
        static_cast<uint8_t>(payload.state),
        v[V0_S_ANOMALY],
        v[V1_S_CUM],
        static_cast<uint8_t>(v[V2_I_SEVERITY]),
        v[V3_C_DEBOUNCE],
        static_cast<uint8_t>(v[V4_R_PLC]),
        v[V5_T_LATENCY],
        v[V6_H_HEALTH],
        static_cast<uint32_t>(v[V7_Z_HASH])
    );

    return std::string(buffer, n > 0 ? n : 0);
}

} // namespace ccvnn