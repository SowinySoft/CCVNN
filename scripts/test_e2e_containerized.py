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


def safe_write_register(client, addresses, value):
    last_error = None

    for addr in addresses:
        try:
            response = client.write_register(
                address=addr,
                value=value,
                slave=PLC_SLAVE_ID,
            )

            if response is not None and not response.isError():
                return response, addr, PLC_SLAVE_ID

            last_error = repr(response)
        except Exception as exc:
            last_error = repr(exc)

    raise AssertionError(
        f"Unable to write registers {addresses} using slave ID "
        f"{PLC_SLAVE_ID}. Last Modbus result: {last_error}"
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
        write_res = None
        target_addresses = [0, 1]

        for attempt in range(5):
            write_res, active_addr, active_slave = safe_write_register(
                plc_client,
                [0, 1],
                1,
            )

            rr = read_holding_registers(plc_client, active_addr, count=1)
            assert rr.registers[0] == 1, (
                f"Expected register value 1, got {rr.registers[0]}"
            )
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


if __name__ == "__main__":
    verify_e2e_stack()