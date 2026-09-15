import json
import logging
import os
import threading
import time
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WatcherDaemon")

# Configuration via Environment Variables or Defaults
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_DETECTIONS_TOPIC = os.getenv("MQTT_DETECTIONS_TOPIC", "ccvnn/detections")
MQTT_STATUS_TOPIC = os.getenv("MQTT_STATUS_TOPIC", "ccvnn/status")

PLC_HOST = os.getenv("PLC_HOST", "ccvnn-plc-sim")
PLC_PORT = int(os.getenv("PLC_PORT", 5020))
PLC_REGISTER_ADDRESS = 0  # Holding Register 40001

DEBOUNCE_THRESHOLD = 5
COOLDOWN_SECONDS = 5.0  # Seconds without hazards before returning to NORMAL

# State Management
state_lock = threading.Lock()
consecutive_hazards = 0
last_hazard_time = 0.0
hazard_active = False
mqtt_client_instance = None


def write_plc_register(address: int, value: int):
    """Executes Modbus TCP write operation to PLC holding register."""
    try:
        client = ModbusTcpClient(PLC_HOST, port=PLC_PORT)
        if client.connect():
            client.write_register(address=address, value=value)
            client.close()
            logger.info(f"Modbus Write Success: Register {40001 + address} -> {value}")
        else:
            logger.error(f"Failed to connect to PLC at {PLC_HOST}:{PLC_PORT}")
    except Exception as e:
        logger.error(f"Error communicating with PLC: {e}")


def broadcast_status(status: str, reasoning: str):
    """Publishes state change events (ALARM / NORMAL) to MQTT status topic."""
    if mqtt_client_instance is None:
        return

    payload = {
        "status": status,
        "timestamp": time.time(),
        "reasoning": reasoning,
        "plc_register_40001": 1 if status == "ALARM" else 0
    }
    try:
        mqtt_client_instance.publish(MQTT_STATUS_TOPIC, json.dumps(payload), qos=1)
        logger.info(f"Broadcasted status event to '{MQTT_STATUS_TOPIC}': {status}")
    except Exception as e:
        logger.error(f"Failed to broadcast status message: {e}")


def cooldown_watchdog():
    """Background thread monitoring hazard timeout and auto-clearing PLC active states."""
    global hazard_active, consecutive_hazards, last_hazard_time

    while True:
        time.sleep(1.0)
        with state_lock:
            if hazard_active and (time.time() - last_hazard_time >= COOLDOWN_SECONDS):
                logger.warning(
                    f"No hazards detected for {COOLDOWN_SECONDS}s. Cooldown triggered. Resetting system..."
                )
                write_plc_register(address=PLC_REGISTER_ADDRESS, value=0)
                hazard_active = False
                consecutive_hazards = 0
                broadcast_status("NORMAL", f"Cooldown period reached ({COOLDOWN_SECONDS}s without hazards).")


def on_connect(client, userdata, flags, reason_code, properties=None):
    """MQTT On-Connect Callback (Paho v2 API)."""
    if reason_code == 0:
        logger.info(f"Connected to MQTT broker. Subscribing to: {MQTT_DETECTIONS_TOPIC}")
        client.subscribe(MQTT_DETECTIONS_TOPIC)
        # Broadcast initial operational state on startup/reconnect
        current_state = "ALARM" if hazard_active else "NORMAL"
        broadcast_status(current_state, "Watcher daemon connected to MQTT broker.")
    else:
        logger.error(f"Failed to connect to MQTT broker, return code: {reason_code}")


def on_message(client, userdata, msg):
    """MQTT On-Message Handler for detection processing and debouncing."""
    global consecutive_hazards, last_hazard_time, hazard_active

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        event_type = payload.get("event_type")
        confidence = payload.get("confidence", 0.0)

        if event_type == "fire_hazard" and confidence >= 0.8:
            with state_lock:
                last_hazard_time = time.time()
                consecutive_hazards += 1
                logger.info(f"Hazard frame received ({consecutive_hazards}/{DEBOUNCE_THRESHOLD})")

                if consecutive_hazards >= DEBOUNCE_THRESHOLD and not hazard_active:
                    logger.critical("Hazard threshold reached! Triggering PLC Alarm State -> 1")
                    write_plc_register(address=PLC_REGISTER_ADDRESS, value=1)
                    hazard_active = True
                    broadcast_status("ALARM", f"Debounce threshold reached ({DEBOUNCE_THRESHOLD} consecutive fire hazards).")
    except Exception as e:
        logger.error(f"Failed to process detection payload: {e}")


def run():
    global mqtt_client_instance

    # Initialize MQTT Client using Callback API Version 2
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    mqtt_client_instance = client

    # Start Cooldown Watchdog Daemon Thread
    watchdog_thread = threading.Thread(target=cooldown_watchdog, daemon=True)
    watchdog_thread.start()

    logger.info("Watcher Safety Service daemon running with status broadcasting...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    run()