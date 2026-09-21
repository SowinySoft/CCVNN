import json
import sys
import time
import paho.mqtt.client as mqtt
from pymodbus.client import ModbusTcpClient

# --- Configuration ---
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ccvnn/v17/industrial/audit_report"

DEFAULT_PLC_IP = "192.168.10.45"
DEFAULT_PLC_PORT = 502
MODBUS_COIL_REGISTER = 40001  # Target Register


def write_plc_register(plc_ip, plc_port, register_address, value):
    """Writes a coil value to the target Modbus TCP PLC."""
    print(f"\n[PLC ACTUATION] Connecting to Modbus PLC at {plc_ip}:{plc_port}...")
    client = ModbusTcpClient(plc_ip, port=plc_port)
    
    if not client.connect():
        print(f"[X] PLC Connection Failed: Unable to reach {plc_ip}:{plc_port}")
        return False

    try:
        # Modbus register write (Coil / Holding Register)
        # Note: 40001 is typically holding register 0 or coil 0 depending on PLC offset.
        reg_offset = register_address - 40001 if register_address >= 40001 else register_address
        
        result = client.write_coil(reg_offset, bool(value))
        if not result.isError():
            print(f"[✓] SUCCESS: Wrote Coil Value {value} to PLC {plc_ip}:{plc_port} [Register: {register_address}]")
            return True
        else:
            print(f"[X] ERROR writing to PLC register: {result}")
            return False
    except Exception as e:
        print(f"[X] Exception during Modbus communication: {e}")
        return False
    finally:
        client.close()


def on_message(client, userdata, msg):
    """Callback triggered when a new audit report is published to MQTT."""
    print(f"\n==================================================")
    print(f"[MQTT RECEIVE] Message received on topic '{msg.topic}'")
    print(f"==================================================")
    
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        summary = payload.get("metadata", {}).get("summary", {})
        hazards_detected = summary.get("hazards_detected", 0)
        audit_records = payload.get("audit_records", [])

        print(f"Audit Summary: {summary.get('total_processed', 0)} Images Processed | Hazards: {hazards_detected}")

        # Check records for hazard triggering
        has_active_hazard = hazards_detected > 0 or any(
            rec.get("register_40001_coil") == 1 for rec in audit_records
        )

        target_plc_ip = DEFAULT_PLC_IP
        target_plc_port = DEFAULT_PLC_PORT

        if has_active_hazard:
            print("\n[!] CRITICAL: Hazard detected in audit payload. Triggering E-STOP Coil (1)...")
            write_plc_register(target_plc_ip, target_plc_port, MODBUS_COIL_REGISTER, value=1)
        else:
            print("\n[✓] Audit payload CLEAR. Ensuring PLC Coil remains De-actuated (0)...")
            write_plc_register(target_plc_ip, target_plc_port, MODBUS_COIL_REGISTER, value=0)

    except json.JSONDecodeError as e:
        print(f"[X] Failed to parse incoming JSON payload: {e}")
    except Exception as e:
        print(f"[X] Unexpected error processing message: {e}")


def start_consumer():
    print("==================================================")
    print("  CCVNN V17 - MODBUS TCP / MQTT CONSUMER SERVICE")
    print("==================================================")
    print(f"Listening to MQTT Topic : {MQTT_TOPIC}")
    print(f"Target PLC Controller  : {DEFAULT_PLC_IP}:{DEFAULT_PLC_PORT}")
    print("Press Ctrl+C to stop.\n")

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="CCVNN_Modbus_Consumer")
    except AttributeError:
        client = mqtt.Client(client_id="CCVNN_Modbus_Consumer")

    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.subscribe(MQTT_TOPIC, qos=1)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[!] Stopping Consumer Service cleanly...")
        client.disconnect()
        sys.exit(0)
    except Exception as e:
        print(f"[X] MQTT Consumer Error: {e}")

if __name__ == "__main__":
    start_consumer()
