import os
import time
import socket
import sqlite3
import logging
from unittest.mock import MagicMock
import paho.mqtt.client as mqtt
from watcher.src.publisher import MQTTPublisher

logger = logging.getLogger(__name__)
DB_PATH = "test_offline_events.db"


def _is_broker_active(host="127.0.0.1", port=1883):
    """Checks if an active MQTT broker is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def test_broker_disconnect_and_reconnect_flushing():
    print("\n--- Testing MQTT Broker Outage & SQLite Store-and-Forward ---")

    # Clean up old test database if present
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    try:
        # 1. Initialize publisher with an unreachable port to simulate a crashed broker
        logger.info("Step 1: Initializing publisher with offline broker endpoint (localhost:19999)...")
        publisher = MQTTPublisher(host="127.0.0.1", port=19999, db_path=DB_PATH)

        # 2. Attempt publishing critical safety events while broker is offline
        test_events = [
            {
                "sensor_id": "cam_01",
                "zone_id": "zone_alpha",
                "tracking_id": "tr_101",
                "event_type": "FIRE_DETECTION",
                "confidence": 0.94,
                "severity": "CRITICAL"
            },
            {
                "sensor_id": "cam_02",
                "zone_id": "zone_beta",
                "tracking_id": "tr_102",
                "event_type": "FORKLIFT_INTRUSION",
                "confidence": 0.88,
                "severity": "CRITICAL"
            },
            {
                "sensor_id": "cam_01",
                "zone_id": "zone_alpha",
                "tracking_id": "tr_103",
                "event_type": "PPE_MISSING",
                "confidence": 0.79,
                "severity": "WARNING"
            }
        ]

        logger.info("Step 2: Publishing events during broker outage...")
        for event in test_events:
            publisher.publish_event_dict(event)

        # 3. Verify payloads were written to SQLite offline database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), payload FROM offline_messages")
        buffered_count, sample_payload = cursor.fetchone()
        conn.close()

        assert buffered_count == len(test_events), (
            f"Expected {len(test_events)} buffered events, but found {buffered_count}"
        )
        print(f"✓ Offline Buffer Verified: {buffered_count} events held in SQLite queue.")
        print(f"   Sample buffered payload: {sample_payload[:80]}...")

        # 4. Restore broker target connection and flush queue
        logger.info("Step 3: Restoring broker connection (localhost:1883) & triggering flush...")

        # Mock broker connection & publish if no live MQTT broker service is running on 1883
        if not _is_broker_active("127.0.0.1", 1883):
            publisher.client.connect = MagicMock(return_value=0)
            publisher.client.loop_start = MagicMock()
            mock_pub_res = MagicMock()
            mock_pub_res.rc = mqtt.MQTT_ERR_SUCCESS
            publisher.client.publish = MagicMock(return_value=mock_pub_res)

        publisher.reconnect_and_flush(host="127.0.0.1", port=1883)

        # 5. Verify SQLite queue was completely drained
        time.sleep(0.5)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM offline_messages")
        remaining_count = cursor.fetchone()[0]
        conn.close()

        assert remaining_count == 0, (
            f"Expected 0 remaining queued messages, but found {remaining_count}"
        )

    finally:
        if "publisher" in locals() and publisher:
            publisher.disconnect()
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
            except Exception:
                pass