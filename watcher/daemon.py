import json
import logging
import os
import threading
import time
import paho.mqtt.client as mqtt
import psycopg2
from pymodbus.client import ModbusTcpClient

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WatcherDaemon")

# Configuration via Environment Variables
MQTT_BROKER = os.getenv("MQTT_BROKER", "ccvnn-mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_DETECTIONS_TOPIC = os.getenv("MQTT_DETECTIONS_TOPIC", "ccvnn/detections")
MQTT_STATUS_TOPIC = os.getenv("MQTT_STATUS_TOPIC", "ccvnn/status")

PLC_HOST = os.getenv("PLC_HOST", "ccvnn-plc-sim")
PLC_PORT = int(os.getenv("PLC_PORT", 5020))
PLC_REGISTER_ADDRESS = 0  # Holding Register 40001

DB_HOST = os.getenv("DB_HOST", "ccvnn-db")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "ccvnn_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgrespassword")

DEBOUNCE_THRESHOLD = 5
COOLDOWN_SECONDS = 5.0

# State Management
state_lock = threading.Lock()
consecutive_hazards = 0
last_hazard_time = 0.0
hazard_active = False
mqtt_client_instance = None


def log_to_database_async(status: str, reasoning: str, plc_value: int):
    """Executes DB insert asynchronously to keep hardware and MQTT actions zero-latency."""
    def _insert():
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                connect_timeout=3
            )
            with conn:
                with conn.cursor() as cursor:
                    query = """
                        INSERT INTO safety_audit_logs (timestamp, state, reasoning, plc_register_40001)
                        VALUES (NOW(), %s, %s, %s);
                    """
                    cursor.execute(query, (status, reasoning, plc_value))
            conn.close()
            logger.info(f"Audit log persisted to DB: [{status}]")
        except Exception as e:
            logger.error(f"Database audit write failed (non-blocking): {e}")

    threading.Thread(target=_insert, daemon=True).start()


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


def handle_state_transition(status: str, reasoning: str):
    """Coordinates Hardware Actuation, MQTT Status Broadcast, and DB Auditing."""
    plc_value = 1 if status == "ALARM" else 0

    # 1. Critical Path: Hardware Actuation
    write_plc_register(address=PLC_REGISTER_ADDRESS, value=plc_value)

    # 2. Critical Path: MQTT Broadcast
    if mqtt_client_instance is not None:
        payload = {
            "status": status,
            "timestamp": time.time(),
            "reasoning": reasoning,
            "plc_register_40001": plc_value
        }
        try:
            mqtt_client_instance.publish(MQTT_STATUS_TOPIC, json.dumps(payload), qos=1)
            logger.info(f"Broadcasted status event to '{MQTT_STATUS_TOPIC}': {status}")
        except Exception as e:
            logger.error(f"Failed to broadcast status message: {e}")

    # 3. Non-Blocking Path: TimescaleDB Audit Record
    log_to_database_async(status=status, reasoning=reasoning, plc_value=plc_value)


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
                hazard_active = False
                consecutive_hazards = 0
                handle_state_transition("NORMAL", f"Cooldown period reached ({COOLDOWN_SECONDS}s without hazards).")


def on_connect(client, userdata, flags, reason_code, properties=None):
    """MQTT On-Connect Callback (Paho v2 API)."""
    if reason_code == 0:
        logger.info(f"Connected to MQTT broker. Subscribing to: {MQTT_DETECTIONS_TOPIC}")
        client.subscribe(MQTT_DETECTIONS_TOPIC)
        # Broadcast and log initial state on boot
        handle_state_transition("NORMAL", "Watcher Safety Daemon initialized.")
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
                    logger.critical("Hazard threshold reached! Triggering ALARM State.")
                    hazard_active = True
                    handle_state_transition("ALARM", f"Debounce threshold reached ({DEBOUNCE_THRESHOLD} consecutive fire hazards).")
    except Exception as e:
        logger.error(f"Failed to process detection payload: {e}")


def run():
    global mqtt_client_instance

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    mqtt_client_instance = client

    watchdog_thread = threading.Thread(target=cooldown_watchdog, daemon=True)
    watchdog_thread.start()

    logger.info("Watcher Safety Service daemon running with async TimescaleDB auditing...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

# Insert into watcher/daemon.py loop
from watcher.ingestion.buffer import MultiVectorIngestionBuffer
from watcher.rules.multi_factor_engine import MultiFactorRuleEngine

# Initialize within WatcherDaemon
self.buffer = MultiVectorIngestionBuffer(max_capacity=1000)
self.rule_engine = MultiFactorRuleEngine()

async def process_cycle(self):
    batch = await self.buffer.get_batch()
    if not batch:
        return

    for vector in batch.vectors:
        actions = self.rule_engine.evaluate_vector(vector)
        for act in actions:
            await self.execute_action(act)
            
            
if __name__ == "__main__":
    run()