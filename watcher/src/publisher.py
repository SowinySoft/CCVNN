import json
import logging
import sqlite3
import time
from typing import Any, Dict, Optional

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError:
    mqtt = None
    CallbackAPIVersion = None

from watcher.src.engine import SeverityRule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQTTPublisher")


class MQTTPublisher:
    """Thread-safe MQTT publisher with SQLite offline store-and-forward buffering."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        db_path: str = "watcher_offline_queue.db",
    ):
        self.host = host
        self.port = port
        self.db_path = db_path
        self._init_db()

        self.client: Optional[mqtt.Client] = None
        if mqtt:
            self.client = mqtt.Client(
                CallbackAPIVersion.VERSION2 if CallbackAPIVersion else None,
                protocol=mqtt.MQTTv5,
            )
            self._connect()

    def _init_db(self) -> None:
        """Ensures the SQLite offline message queue table is available."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        finally:
            conn.close()

    def _buffer_message(self, topic: str, payload: Dict[str, Any]) -> None:
        """Buffers an unsent MQTT event to SQLite storage."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO offline_messages (topic, payload) VALUES (?, ?)",
                    (topic, json.dumps(payload)),
                )
            logger.info(f"Buffered offline event to SQLite for topic [{topic}]")
        finally:
            conn.close()

    def _connect(self) -> None:
        """Initiates background MQTT broker connection loop."""
        try:
            if self.client:
                self.client.connect(self.host, self.port, keepalive=60)
                self.client.loop_start()
                logger.info(
                    f"Connected to MQTT broker at {self.host}:{self.port}"
                )
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker ({self.host}:{self.port}): {e}")

    def publish_event_dict(
        self, payload: Dict[str, Any], topic: str = "watcher/safety/alerts"
    ) -> bool:
        """Publishes raw payload dictionary or buffers offline if unavailable."""
        if self.client and self.client.is_connected():
            try:
                res = self.client.publish(topic, json.dumps(payload), qos=1)
                if res.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Published MQTT event directly to [{topic}]")
                    self.flush_offline_buffer()
                    return True
            except Exception as e:
                logger.warning(f"Error publishing directly via MQTT: {e}")

        self._buffer_message(topic, payload)
        return False

    def publish_event(
        self,
        rule: SeverityRule,
        sensor_id: str,
        zone_id: str,
        confidence: float,
        bounding_box: list,
        tracking_id: str,
    ) -> bool:
        """Formats a structured safety alert event and dispatches via MQTT."""
        severity_val = getattr(rule, "severity", "INFO")
        severity_str = (
            str(severity_val.value)
            if hasattr(severity_val, "value")
            else str(severity_val)
        )

        payload = {
            "event_id": f"evt_{int(time.time() * 1000)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sensor_id": sensor_id,
            "zone_id": zone_id,
            "tracking_id": tracking_id,
            "event_type": getattr(rule, "display_name", str(rule)),
            "severity": severity_str,
            "confidence": round(confidence, 4),
            "bounding_box": bounding_box,
            "action_type": getattr(rule, "action_type", None),
            "plc_config": getattr(rule, "plc_config", None),
        }

        rule_topics = getattr(rule, "topics", None)
        target_topics = (
            rule_topics
            if rule_topics
            else [f"factory/{zone_id}/alerts/{severity_str.lower()}"]
        )

        success = True
        for topic in target_topics:
            if not self.publish_event_dict(payload, topic=topic):
                success = False

        return success

    def flush_offline_buffer(self) -> None:
        """Flushes cached SQLite offline messages sequentially once connected."""
        if not self.client or not self.client.is_connected():
            return

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, topic, payload FROM offline_messages ORDER BY id ASC"
            )
            rows = cursor.fetchall()

            if not rows:
                return

            flushed_ids = []
            for row_id, topic, payload_str in rows:
                try:
                    res = self.client.publish(topic, payload_str, qos=1)
                    if res.rc == mqtt.MQTT_ERR_SUCCESS:
                        flushed_ids.append(row_id)
                    else:
                        break
                except Exception as e:
                    logger.error(f"Failed to flush queued message ID {row_id}: {e}")
                    break

            if flushed_ids:
                with conn:
                    conn.executemany(
                        "DELETE FROM offline_messages WHERE id = ?",
                        [(i,) for i in flushed_ids],
                    )
                logger.info(
                    f"Flushed {len(flushed_ids)} offline messages from SQLite queue."
                )
        finally:
            conn.close()

    def disconnect(self) -> None:
        """Stops event loop and closes MQTT network connection."""
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting MQTT client: {e}")