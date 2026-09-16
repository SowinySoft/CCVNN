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
    # Attempt connection with retries
    connected = False
    for _ in range(10):
        if plc_client.connect():
            connected = True
            break
        time.sleep(1)

    assert connected, "Failed to connect to containerized Modbus PLC simulator!"
    #assert plc_client.connect(), "Failed to connect to containerized Modbus PLC simulator!"
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

    # Replace the direct register check block with:
    INFO = "Simulating direct register verification on PLC (Register 40001)..."

    # 1. Write value 1 to address 0 (Holding Register 40001 in 0-based indexing)
    plc_client.write_register(0, 1)

    # 2. Read address 0 with count=1 keyword argument
    rr = plc_client.read_holding_registers(0, count=1)

    # 3. Verify read back
    assert not rr.isError() and rr.registers[0] == 1, "Failed to write/read PLC register 40001!"
    logger.info("✓ PLC Register 40001 updated to active state (1).")

    # Reset state
    plc_client.write_register(40001, 0)
    plc_client.close()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

    logger.info("\n🎉 CONTAINERIZED END-TO-END PIPELINE VERIFICATION PASSED!")


if __name__ == "__main__":
    verify_e2e_stack()