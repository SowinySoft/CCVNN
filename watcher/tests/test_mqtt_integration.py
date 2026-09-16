import json
import time
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from watcher.src.publisher import MQTTPublisher


def test_mqtt_pipeline():
    received_messages = []

    def on_message(client, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        received_messages.append((msg.topic, payload))

    sub_client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    sub_client.on_message = on_message

    broker_host = "localhost"
    broker_port = 1883
    
    # Use wildcard to capture dynamically routed alert topics (e.g. factory/Zone_A/alerts/info)
    subscribe_topic = "factory/#"

    publisher = None
    try:
        sub_client.connect(broker_host, broker_port, keepalive=60)
        sub_client.subscribe(subscribe_topic)
        sub_client.loop_start()

        publisher = MQTTPublisher(broker_host, broker_port)
        
        # Allow connection handshake to complete so publisher doesn't trigger Store-and-Forward
        time.sleep(0.5)

        publisher.publish_event(
            sensor_id="cam_01",
            rule="PPE_VIOLATION",
            zone_id="Zone_A",
            confidence=0.95,
            bounding_box=[10, 20, 100, 200],
            tracking_id=101
        )

        time.sleep(0.5)

        assert len(received_messages) > 0

    finally:
        sub_client.loop_stop()
        sub_client.disconnect()
        if publisher is not None and hasattr(publisher, "close"):
            publisher.close()