import logging
import time
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContainerizedE2ETest")

MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
PLC_HOST = "127.0.0.1"
PLC_PORT = 5020


def verify_e2e_stack():
    logger.info("--- Starting Phase 7 Containerized E2E Pipeline Verification ---")

    # 1. Connect to Containerized Modbus PLC
    logger.info(f"Connecting to PLC Simulator at {PLC_HOST}:{PLC_PORT}...")
    plc_client = ModbusTcpClient(PLC_HOST, port=PLC_PORT)
    
    connected = False
    for _ in range(10):
        if plc_client.connect():
            connected = True
            break
        time.sleep(1)

    assert connected, "Failed to connect to containerized Modbus PLC simulator!"
    logger.info("✓ Connected to Modbus PLC.")

    # 2. Setup MQTT Listener
    alerts_received = []

    def on_message(client, userdata, msg):
        payload = msg.payload.decode()
        logger.info(f"[MQTT Received] Topic: {msg.topic} | Payload: {payload}")
        alerts_received.append((msg.topic, payload))

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
    mqtt_client.subscribe("factory/+/hazard")
    mqtt_client.loop_start()

    logger.info("✓ Subscribed to MQTT topics.")

    # 3. Direct register verification on PLC (Holding Register 40001 / Address 0)
    try:
        write_res = plc_client.write_register(0, 1)
        assert not write_res.isError(), "Failed to write register 0"

        rr = plc_client.read_holding_registers(0, count=1)
        assert not rr.isError(), "Failed to read register 0"
        assert rr.registers[0] == 1, f"Expected 1, got {rr.registers[0]}"
        logger.info("✓ Modbus Register read/write verified.")
    finally:
        if plc_client:
            plc_client.write_register(0, 0)
            plc_client.close()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    verify_e2e_stack()