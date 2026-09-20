import logging
import os
import time
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContainerizedE2ETest")

# Dynamic environment resolution with defaults
MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
PLC_HOST = os.getenv("PLC_HOST", "127.0.0.1")
PLC_PORT = int(os.getenv("PLC_PORT", 5020))
PLC_SLAVE_ID = int(os.getenv("PLC_SLAVE_ID", 1))


def verify_e2e_stack():
    logger.info("--- Starting V14 Cumulative Vector Vision Containerized E2E Pipeline Verification ---")

    # 1. Connect to Containerized Modbus PLC
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

    # 2. Setup MQTT Listener with Retries
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

    # 3. Direct register verification on PLC (Holding Register 40001 / Address 0)
    def safe_write_register(client, address, value):
        for slave_arg in [PLC_SLAVE_ID, 1, 0, None]:
            try:
                kwargs = {"slave": slave_arg} if slave_arg is not None else {}
                res = client.write_register(address, value, **kwargs)
                if not res.isError():
                    return res, slave_arg
            except TypeError:
                try:
                    kwargs = {"unit": slave_arg} if slave_arg is not None else {}
                    res = client.write_register(address, value, **kwargs)
                    if not res.isError():
                        return res, slave_arg
                except Exception:
                    continue
            except Exception:
                continue
        return client.write_register(address, value), None

    def safe_read_holding_registers(client, address, count, active_slave):
        kwargs = {}
        if active_slave is not None:
            kwargs["slave"] = active_slave
        try:
            res = client.read_holding_registers(address, count=count, **kwargs)
            if not res.isError():
                return res
        except TypeError:
            kwargs = {"unit": active_slave} if active_slave is not None else {}
            res = client.read_holding_registers(address, count=count, **kwargs)
        return res

    active_slave = None
    try:
        write_res = None
        for attempt in range(5):
            write_res, active_slave = safe_write_register(plc_client, 0, 1)
            if write_res and not write_res.isError():
                break
            time.sleep(1)

        assert write_res and not write_res.isError(), f"Failed to write register 0! Error: {write_res}"

        rr = safe_read_holding_registers(plc_client, 0, count=1, active_slave=active_slave)
        assert not rr.isError(), f"Failed to read register 0! Error: {rr}"
        assert rr.registers[0] == 1, f"Expected register value 1, got {rr.registers[0]}"
        logger.info(f"✓ Modbus Register read/write verified (Active Slave ID: {active_slave}).")

    finally:
        if plc_client:
            try:
                safe_write_register(plc_client, 0, 0)
            except Exception:
                pass
            plc_client.close()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    verify_e2e_stack()