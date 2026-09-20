import json
import time
import socket
from unittest.mock import MagicMock
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from watcher.src.publisher import MQTTPublisher


def _is_broker_active(host="localhost", port=1883):
    """Check if an active MQTT broker is listening on the host/port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def test_mqtt_pipeline():
    received_messages = []

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        received_messages.append((msg.topic, payload))

    broker_host = "localhost"
    broker_port = 1883
    subscribe_topic = "factory/#"

    # Fallback to mock mode if no local broker service is running
    if not _is_broker_active(broker_host, broker_port):
        publisher = MQTTPublisher(broker_host, broker_port)
        publisher.is_connected = True

        def mock_publish(topic, payload, *args, **kwargs):
            msg = MagicMock()
            msg.topic = topic
            msg.payload = payload.encode("utf-8") if isinstance(payload, str) else payload
            on_message(None, None, msg)
            res = MagicMock()
            res.rc = mqtt.MQTT_ERR_SUCCESS
            return res

        publisher.client.publish = mock_publish
        publisher.publish_event(
            sensor_id="cam_01",
            rule="PPE_VIOLATION",
            zone_id="Zone_A",
            confidence=0.95,
            bounding_box=[10, 20, 100, 200],
            tracking_id=101
        )
        assert len(received_messages) > 0, "No MQTT messages were captured in mock mode."
        return

    # Live broker mode
    sub_client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    sub_client.on_message = on_message

    publisher = None
    try:
        sub_client.connect(broker_host, broker_port, keepalive=60)
        sub_client.subscribe(subscribe_topic)
        sub_client.loop_start()

        publisher = MQTTPublisher(broker_host, broker_port)

        for _ in range(30):
            if publisher.is_connected:
                break
            time.sleep(0.1)

        publisher.publish_event(
            sensor_id="cam_01",
            rule="PPE_VIOLATION",
            zone_id="Zone_A",
            confidence=0.95,
            bounding_box=[10, 20, 100, 200],
            tracking_id=101
        )

        for _ in range(30):
            if len(received_messages) > 0:
                break
            time.sleep(0.1)

        assert len(received_messages) > 0, "No MQTT messages were received within timeout."

    finally:
        sub_client.loop_stop()
        sub_client.disconnect()
        if publisher:
            publisher.disconnect()