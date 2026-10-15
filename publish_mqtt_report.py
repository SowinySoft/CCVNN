import json
import os
import sys
import paho.mqtt.client as mqtt

# Configuration
JSON_FILE = "batch_verifier_report.json"
MQTT_BROKER = "127.0.0.1"  # Or your broker IP / Mosquitto host
MQTT_PORT = 1883
MQTT_TOPIC = "ccvnn/v17/industrial/audit_report"

def publish_audit_report():
    if not os.path.exists(JSON_FILE):
        print(f"[X] Error: {JSON_FILE} not found. Run 'test_batch_verifier.py' first.")
        sys.exit(1)

    # Read the JSON payload
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    payload_str = json.dumps(report_data, indent=2)

    # Initialize MQTT client (compatible with paho-mqtt v1.x and v2.x)
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="CCVNN_Audit_Publisher")
    except AttributeError:
        # Fallback for paho-mqtt v1.x
        client = mqtt.Client(client_id="CCVNN_Audit_Publisher")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[✓] Connected successfully to MQTT Broker ({MQTT_BROKER}:{MQTT_PORT})")
        else:
            print(f"[X] Failed to connect, return code: {rc}")

    def on_publish(client, userdata, mid, reason_code=None, properties=None):
        print(f"[✓] Report payload successfully published to topic: '{MQTT_TOPIC}' (MID: {mid})")

    client.on_connect = on_connect
    client.on_publish = on_publish

    print(f"Connecting to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()

        # Publish payload with Quality of Service (QoS) level 1
        info = client.publish(MQTT_TOPIC, payload_str, qos=1)
        info.wait_for_publish(timeout=5)

        client.loop_stop()
        client.disconnect()
        print("[✓] MQTT Client disconnected cleanly.")

    except Exception as e:
        print(f"[X] MQTT Publishing Error: {e}")

if __name__ == "__main__":
    publish_audit_report()
