import json
import time
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from watcher.src.engine import DynamicSeverityEngine
from watcher.src.publisher import MQTTPublisher

received_messages = []

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    received_messages.append(payload)
    print(f"\n[SUBSCRIBER RECEIVED] Topic: {msg.topic}")
    print(json.dumps(payload, indent=2))

def test_mqtt_pipeline():
    # 1. Start test subscriber

# Update subscriber line:
    sub_client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    sub_client.on_message = on_message
    sub_client.connect("127.0.0.1", 1883)
    sub_client.subscribe("factory/#")
    sub_client.loop_start()

    # 2. Initialize Engine & Publisher
    engine = DynamicSeverityEngine("configs/severity_config.yaml")
    publisher = MQTTPublisher(host="127.0.0.1", port=1883)

    time.sleep(0.5)  # Allow MQTT connections to settle

    # 3. Simulate frame stream
    frames = [
        {"class": "fire_smoke", "conf": 0.78, "bbox": [100, 150, 200, 250]},
        {"class": "fire_smoke", "conf": 0.82, "bbox": [102, 152, 202, 252]},
        {"class": "fire_smoke", "conf": 0.85, "bbox": [105, 155, 205, 255]},  # Trigger frame
    ]

    for frame in frames:
        rule = engine.process_detection(frame["class"], frame["conf"], "cam_zone1_01")
        if rule:
            publisher.publish_event(
                rule=rule,
                sensor_id="cam_zone1_01",
                zone_id="zone_01",
                confidence=frame["conf"],
                bounding_box=frame["bbox"],
                tracking_id="cam_zone1_01"
            )

    time.sleep(1.0)  # Wait for broker message transmission

    assert len(received_messages) == 1
    assert received_messages[0]["severity"] == "CRITICAL"
    assert received_messages[0]["sensor_id"] == "cam_zone1_01"
    print("\n✓ End-to-end MQTT integration test passed successfully.")

    publisher.disconnect()
    sub_client.loop_stop()
    sub_client.disconnect()

if __name__ == "__main__":
    test_mqtt_pipeline()