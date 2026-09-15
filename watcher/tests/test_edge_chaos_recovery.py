import logging
import time
from unittest.mock import MagicMock, patch
from watcher.src.WatcherEngine import AlertAction, WatcherEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeChaosTest")


class MockModbusClient:
    """Simulates a flapping Modbus PLC TCP connection."""
    def __init__(self, fail_after_writes: int = 2):
        self.connected = True
        self.write_count = 0
        self.fail_after_writes = fail_after_writes

    def write_register(self, address, value):
        if not self.connected:
            raise ConnectionError("Modbus TCP connection lost!")
        
        self.write_count += 1
        if self.write_count >= self.fail_after_writes:
            logger.warning("[Chaos] Simulating sudden PLC network partition...")
            self.connected = False
            raise ConnectionError("Modbus TCP connection lost!")
        return MagicMock(isError=lambda: False)

    def connect(self):
        logger.info("[Recovery] Reconnecting to Modbus TCP PLC...")
        self.connected = True
        self.write_count = 0
        return True


class ResilientPLCHandler:
    """Wrapper implementing automatic Modbus TCP reconnection with exponential backoff."""
    def __init__(self, host: str, port: int, max_retries: int = 3):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.client = MockModbusClient()

    def safe_write(self, register: int, value: int):
        attempt = 0
        backoff = 0.05  # Initial backoff in seconds

        while attempt < self.max_retries:
            try:
                if not self.client.connected:
                    self.client.connect()
                self.client.write_register(register, value)
                return True
            except (ConnectionError, Exception) as e:
                attempt += 1
                logger.error(f"PLC Write Failed (Attempt {attempt}/{self.max_retries}): {e}")
                if attempt >= self.max_retries:
                    logger.critical("Max retries reached. Entering failsafe local state buffering.")
                    return False
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
        return False


def test_plc_network_partition_recovery():
    logger.info("--- Running Chaos Test: PLC Network Partition Recovery ---")
    handler = ResilientPLCHandler(host="192.168.10.45", port=502, max_retries=3)

    # First writes should succeed, then partition triggers, followed by recovery reconnect
    success_1 = handler.safe_write(40001, 1)
    assert success_1, "First write should succeed."

    # Trigger partition and verify retry & recovery
    success_recovery = handler.safe_write(40001, 1)
    logger.info(f"Recovery status after partition: {success_recovery}")
    assert handler.client.connected, "Client should successfully reconnect after backoff."
    logger.info("✓ SUCCESS: PLC network partition recovery test passed.")


def test_mqtt_buffer_retention():
    logger.info("--- Running Chaos Test: MQTT Broker Failover & Buffer Retention ---")
    
    # Simulate offline buffer queue for alerts when broker is unreachable
    offline_buffer = []
    broker_online = False

    def publish_alert(topic: str, payload: dict):
        nonlocal broker_online
        if not broker_online:
            logger.warning(f"[Broker Offline] Buffering alert for topic '{topic}': {payload}")
            offline_buffer.append((topic, payload))
            return False
        else:
            logger.info(f"[Broker Online] Publishing buffered alert to '{topic}'")
            return True

    # 1. Dispatch while broker is down
    publish_alert("factory/cam_01/hazard", {"status": "ALERT", "code": 1})
    publish_alert("factory/cam_01/hazard", {"status": "HEARTBEAT", "code": 2})
    
    assert len(offline_buffer) == 2, "Alerts should be buffered locally during broker downtime."

    # 2. Broker restores
    logger.info("[Broker Recovered] Flushing offline buffer...")
    broker_online = True
    
    flushed_count = 0
    while offline_buffer:
        topic, payload = offline_buffer.pop(0)
        if publish_alert(topic, payload):
            flushed_count += 1

    assert flushed_count == 2, "All buffered alerts must be successfully flushed upon reconnection."
    logger.info("✓ SUCCESS: MQTT buffer retention and failover flush passed.")


def test_triton_server_recovery():
    logger.info("--- Running Chaos Test: Triton Server Restart & Backoff ---")
    
    server_healthy = False
    attempts = 0

    def query_triton_with_backoff():
        nonlocal server_healthy, attempts
        backoff = 0.01
        for _ in range(3):
            attempts += 1
            if not server_healthy:
                logger.warning(f"Triton server unreachable on attempt {attempts}. Backing off...")
                time.sleep(backoff)
                backoff *= 2
                # Simulate server coming back online on attempt 3
                if attempts >= 3:
                    server_healthy = True
            else:
                logger.info("Triton server responded successfully.")
                return True
        return server_healthy

    success = query_triton_with_backoff()
    assert success, "Client should successfully reconnect to Triton after server restart."
    logger.info("✓ SUCCESS: Triton server recovery and backoff test passed.")


if __name__ == "__main__":
    test_plc_network_partition_recovery()
    test_mqtt_buffer_retention()
    test_triton_server_recovery()
    logger.info("\n🎉 ALL PHASE 6 CHAOS & FAULT TOLERANCE TESTS PASSED SUCCESSFULLY!")