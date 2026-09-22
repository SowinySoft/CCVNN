import logging
import os
import time
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContainerizedE2ETest")

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
PLC_HOST = os.getenv("PLC_HOST", "127.0.0.1")
PLC_PORT = int(os.getenv("PLC_PORT", 5020))
PLC_SLAVE_ID = int(os.getenv("PLC_SLAVE_ID", 1))


def safe_write_register(client, address, values, slave_id=PLC_SLAVE_ID, attempts=5):
    last_result = None
    if isinstance(values, int):
        values = [values]

    for attempt in range(attempts):
        try:
            if not client.connected and not client.connect():
                raise ConnectionError("PLC connection could not be established")

            result = client.write_registers(
                address=address,
                values=values,
                slave=slave_id,
            )
            last_result = result

            if not result.isError():
                return result, address, slave_id

        except Exception as exc:
            last_result = exc

        # The simulator may reset a socket during startup. Reconnect before
        # retrying rather than reusing the broken Modbus connection.
        try:
            client.close()
        except Exception:
            pass

        time.sleep(1 + attempt)

    raise AssertionError(
        f"Unable to write registers {values} to address {address} using slave ID {slave_id}. "
        f"Last Modbus result: {last_result!r}"
    )


def read_holding_registers(client, address, count):
    response = client.read_holding_registers(
        address=address,
        count=count,
        slave=PLC_SLAVE_ID,
    )

    if response is None or response.isError():
        raise AssertionError(
            f"Unable to read register {address} using slave ID "
            f"{PLC_SLAVE_ID}: {response!r}"
        )

    return response


def verify_e2e_stack():
    logger.info("--- Starting V14 Cumulative Vector Vision Containerized E2E Pipeline Verification ---")

    logger.info(f"Connecting to PLC Simulator at {PLC_HOST}:{PLC_PORT}...")
    plc_client = ModbusTcpClient(PLC_HOST, port=PLC_PORT)

    plc_connected = False
    for i in range(10):
        if plc_client.connect():
            plc_connected = True
            break
        logger.info(f"Waiting for PLC ({i+1}/10)...")
        time.sleep(1)

    assert plc_connected, f"Failed to connect to Modbus PLC simulator at {PLC_HOST}:{PLC_PORT}!"
    logger.info("✓ Connected to Modbus PLC.")

    alerts_received = []

    def on_message(client, userdata, msg):
        payload = msg.payload.decode()
        logger.info(f"[MQTT Received] Topic: {msg.topic} | Payload: {payload}")
        alerts_received.append((msg.topic, payload))

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_message = on_message

    mqtt_connected = False
    for i in range(10):
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
            mqtt_connected = True
            break
        except Exception:
            logger.info(f"Waiting for MQTT Broker ({i+1}/10)...")
            time.sleep(1)

    assert mqtt_connected, f"Failed to connect to MQTT broker at {MQTT_HOST}:{MQTT_PORT}!"

    mqtt_client.subscribe("factory/+/hazard")
    mqtt_client.loop_start()
    logger.info("✓ Subscribed to MQTT factory hazard topics.")

    active_addr = None
    active_slave = None
    try:
        target_addresses = [0, 1]

        for addr in target_addresses:
            try:
                write_res, active_addr, active_slave = safe_write_register(
                    plc_client,
                    address=addr,
                    values=[1],
                    slave_id=PLC_SLAVE_ID,
                )
                rr = read_holding_registers(plc_client, active_addr, count=1)
                assert rr.registers[0] == 1
                break
            except Exception as e:
                if addr == target_addresses[-1]:
                    raise AssertionError(f"PLC Modbus verification failed on all target addresses: {e}")
                plc_client.close()
                time.sleep(1)

        logger.info(f"✓ Modbus Register read/write verified (Address: {active_addr}, Slave ID: {active_slave}).")

    finally:
        if plc_client and active_addr is not None:
            try:
                plc_client.write_register(
                    address=active_addr,
                    value=0,
                    slave=PLC_SLAVE_ID,
                )
            except Exception as exc:
                logger.warning("Unable to reset register %s: %s", active_addr, exc)
            plc_client.close()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

def check_cyclotron_multi_stream_pipeline():
    """Validates multi-factor vector schema and rule evaluation within E2E harness."""
    from watcher.schemas.vector_schema import VectorPayload, VectorSourceType, SpatialEntity
    from watcher.rules.multi_factor_engine import MultiFactorRuleEngine

    engine = MultiFactorRuleEngine("config/watcher_multi_factor_rules.yaml")
    vec = VectorPayload(
        vector_id="e2e_test_vec",
        source_type=VectorSourceType.CAMERA_STREAM,
        source_id="cam_e2e",
        hazard_score=0.90,
        entities=[SpatialEntity(label="person", confidence=0.88, class_id=0)]
    )
    
    actions = engine.evaluate_vector(vec)
    assert len(actions) == 1, "E2E Multi-factor rule evaluation failed!"
    print("[E2E SUCCESS] Cyclotron Multi-Stream pipeline check passed.")
    
if __name__ == "__main__":
    verify_e2e_stack()