from unittest.mock import MagicMock, patch
from watcher.src.actuator import PLCActuator

@patch("watcher.src.actuator.ModbusTcpClient")
def test_plc_actuation_success(mock_client_cls):
    # Setup mock client behavior
    mock_client = MagicMock()
    mock_client.connect.return_value = True
    
    mock_result = MagicMock()
    mock_result.isError.return_value = False
    mock_client.write_register.return_value = mock_result
    
    mock_client_cls.return_value = mock_client

    # Test configuration
    plc_config = {
        "ip": "192.168.10.45",
        "port": 502,
        "register": 40001,
        "value": 1
    }

    actuator = PLCActuator()
    success = actuator.trigger_relay(plc_config)

    # Assertions
    assert success is True
    mock_client.connect.assert_called_once()
    mock_client.write_register.assert_called_once_with(address=40001, value=1, slave=1)
    mock_client.close.assert_called_once()
    print("✓ Modbus TCP actuation success test passed.")


@patch("watcher.src.actuator.PLCActuator._trigger_rs485_fallback", return_value=False)
@patch("watcher.src.actuator.ModbusTcpClient")
def test_plc_connection_failure(mock_client_cls, mock_fallback):
    mock_client = MagicMock()
    mock_client.connect.return_value = False
    mock_client_cls.return_value = mock_client

    plc_config = {
        "ip": "192.168.10.45",
        "port": 502,
        "register": 40001,
        "value": 1
    }

    actuator = PLCActuator()
    success = actuator.trigger_relay(plc_config)

    assert success is False
    
if __name__ == "__main__":
    test_plc_actuation_success()
    test_plc_connection_failure()
