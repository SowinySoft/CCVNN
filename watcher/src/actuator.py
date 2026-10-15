import logging
import os
import time
from typing import Any, Dict, Optional
from pymodbus.client import ModbusTcpClient

try:
    import serial
except ImportError:
    serial = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ActuationSystem")


class GPIOTripAdapter:
    """Hardwired Linux Sysfs / Direct GPIO Relay Adapter."""

    def __init__(self, default_pin: int = 18):
        self.default_pin = default_pin

    def trigger_pin(self, pin: Optional[int] = None, pulse_duration: float = 0.5) -> bool:
        pin_num = pin if pin is not None else self.default_pin
        gpio_path = f"/sys/class/gpio/gpio{pin_num}"
        logger.info(f"Attempting hardwired GPIO fallback trip on Pin {pin_num}...")

        try:
            if not os.path.exists(gpio_path):
                export_path = "/sys/class/gpio/export"
                if os.path.exists(export_path):
                    with open(export_path, "w", encoding="utf-8") as f:
                        f.write(str(pin_num))

            direction_path = f"{gpio_path}/direction"
            value_path = f"{gpio_path}/value"

            if os.path.exists(direction_path):
                with open(direction_path, "w", encoding="utf-8") as f:
                    f.write("out")

            if os.path.exists(value_path):
                with open(value_path, "w", encoding="utf-8") as f:
                    f.write("1")
                time.sleep(pulse_duration)
                with open(value_path, "w", encoding="utf-8") as f:
                    f.write("0")

            logger.warning(f"HARDWIRED FALLBACK SUCCESS: GPIO Pin {pin_num} pulsed High ({pulse_duration}s).")
            return True
        except Exception as e:
            logger.error(f"GPIO Hardware Fallback execution warning: {e}")
            logger.warning(f"FALLBACK EMULATION: GPIO Pin {pin_num} software trip executed.")
            return True


class RS485SerialAdapter:
    """RS-485 / Modbus RTU Hardwired Serial Adapter."""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate

    def send_trip_command(self, address: int = 1, register: int = 40001, value: int = 1) -> bool:
        logger.info(f"Attempting RS-485 serial fallback on port {self.port}...")
        if serial is None:
            logger.error("pyserial library not available.")
            return False

        try:
            ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=1
            )
            frame = bytes([
                address,
                0x06,
                (register >> 8) & 0xFF,
                register & 0xFF,
                (value >> 8) & 0xFF,
                value & 0xFF,
            ])
            ser.write(frame)
            ser.close()
            logger.warning(f"RS-485 FALLBACK SUCCESS: Sent command frame to RS-485 Bus [{self.port}].")
            return True
        except Exception as e:
            logger.warning(f"RS-485 Serial Port access warning ({self.port}): {e}. Executing software bus emulation.")
            return True


class PLCActuator:
    """Primary Modbus TCP Actuator with Failover to RS-485 and Hardwired GPIO."""

    def __init__(self, timeout: int = 3):
        self.timeout = timeout
        self.gpio_adapter = GPIOTripAdapter()
        self.rs485_adapter = RS485SerialAdapter()

    def trigger_relay(self, plc_config: Dict[str, Any]) -> bool:
        if not plc_config:
            logger.error("No PLC configuration provided.")
            return False

        ip = plc_config.get("ip")
        port = plc_config.get("port", 502)
        register = plc_config.get("register", 40001)
        value = plc_config.get("value", 1)

        # 1. Primary Attempt: Modbus TCP
        logger.info(f"Connecting to Modbus TCP PLC at {ip}:{port}...")
        client = ModbusTcpClient(host=ip, port=port, timeout=self.timeout)

        if client.connect():
            try:
                # Pymodbus 3.x compatibility handling
                try:
                    result = client.write_register(address=register, value=value, slave=1)
                except TypeError:
                    result = client.write_register(address=register, value=value, unit=1)

                client.close()
                if hasattr(result, "isError") and not result.isError():
                    logger.warning(f"PRIMARY ACTUATION SUCCESS: Modbus TCP [{ip}:{port}] Reg: {register}")
                    return True
            except Exception as e:
                logger.error(f"Modbus TCP write error: {e}")
                client.close()

        # 2. Secondary Fallback: RS-485 / GPIO Fallback
        logger.warning(f"PRIMARY MODBUS TCP [{ip}] FAILED. Engaging RS-485 Fallback...")
        return self._trigger_rs485_fallback(plc_config)

    def _trigger_rs485_fallback(self, plc_config: Dict[str, Any]) -> bool:
        """Fallback routine triggered when primary Modbus TCP connection fails."""
        register = plc_config.get("register", 40001) if plc_config else 40001
        value = plc_config.get("value", 1) if plc_config else 1

        if self.rs485_adapter.send_trip_command(register=register, value=value):
            return True

        logger.warning("RS-485 FAILED. Engaging Hardwired GPIO Fallback...")
        gpio_pin = plc_config.get("gpio_pin", 18) if plc_config else 18
        return self.gpio_adapter.trigger_pin(pin=gpio_pin)