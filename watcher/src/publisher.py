import os
import json
import sqlite3
import logging
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

logger = logging.getLogger("MQTTPublisher")


class MQTTPublisher:
    def __init__(self, host="localhost", port=1883, db_path="offline_events.db"):
        self.host = host
        self.port = port
        self.db_path = db_path
        self.is_connected = False

        self._init_sqlite_db()

        self.client = mqtt.Client(CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker ({self.host}:{self.port}): {e}")

    def _init_sqlite_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS offline_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize offline SQLite database: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.is_connected = True
            logger.info(f"Connected to MQTT broker at {self.host}:{self.port}")
        else:
            self.is_connected = False
            logger.warning(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self.is_connected = False
        logger.warning("Disconnected from MQTT broker.")

    def publish_event(self, sensor_id, rule, zone_id, confidence, bounding_box, tracking_id):
        topic = f"factory/{zone_id}/alerts/info"
        payload = {
            "sensor_id": sensor_id,
            "rule": rule,
            "zone_id": zone_id,
            "confidence": confidence,
            "bounding_box": bounding_box,
            "tracking_id": tracking_id
        }
        self.publish_event_dict(payload, topic=topic)

    def publish_event_dict(self, event_dict, topic="factory/alerts"):
        payload_str = json.dumps(event_dict)
        if self.is_connected:
            try:
                res = self.client.publish(topic, payload_str)
                if res.rc == mqtt.MQTT_ERR_SUCCESS:
                    return
            except Exception as e:
                logger.error(f"Publish error: {e}")

        # Store in SQLite if offline or publish fails
        self._buffer_offline_message(topic, payload_str)

    def _buffer_offline_message(self, topic, payload):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO offline_messages (topic, payload) VALUES (?, ?)", (topic, payload))
            conn.commit()
            conn.close()
            logger.info("Message buffered to offline SQLite database.")
        except Exception as e:
            logger.error(f"Failed to buffer message to SQLite: {e}")

    def reconnect_and_flush(self, host=None, port=None):
        """Reconnects to the specified MQTT broker and flushes buffered SQLite offline messages."""
        if host:
            self.host = host
        if port:
            self.port = port

        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            self.is_connected = True
            logger.info(f"Successfully reconnected to MQTT broker at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to reconnect to MQTT broker ({self.host}:{self.port}): {e}")
            return False

        if os.path.exists(self.db_path):
            self._flush_sqlite_buffer()

        return True

    def _flush_sqlite_buffer(self):
        """Flushes messages stored in SQLite offline_messages table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, topic, payload FROM offline_messages")
            rows = cursor.fetchall()

            for msg_id, topic, payload in rows:
                self.client.publish(topic, payload)
                cursor.execute("DELETE FROM offline_messages WHERE id = ?", (msg_id,))

            conn.commit()
            conn.close()
            logger.info(f"Flushed {len(rows)} offline messages to broker.")
        except Exception as e:
            logger.error(f"Error flushing SQLite buffer: {e}")