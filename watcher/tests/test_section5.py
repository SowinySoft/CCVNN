from unittest.mock import patch, MagicMock
from watcher.src.actuator import PLCActuator
from watcher.src.notifications import NotificationDispatcher

def test_hardware_actuation_fallback():
    actuator = PLCActuator()
    plc_config = {
        "ip": "192.168.254.254",  # Unreachable IP to force primary failure
        "port": 502,
        "register": 40001,
        "value": 1,
        "gpio_pin": 21
    }

    print("\n--- Testing Hardware Failover Chaining (TCP Fail -> RS-485 -> GPIO) ---")
    success = actuator.trigger_relay(plc_config)
    assert success is True
    print("✓ Hardware actuation failover passed.")

def test_webhook_and_sms_notifications():
    dispatcher = NotificationDispatcher()
    payload = {
        "event_type": "Electrical Arc / Shock Hazard",
        "severity": "CRITICAL",
        "zone_id": "zone_power_station"
    }

    print("\n--- Testing Webhook & SMS Secondary Notifications ---")
    # Test SMS Dispatch
    sms_res = dispatcher.dispatch_sms_alert("+201000000000", "Critical Arc Flash Detected")
    assert sms_res is True

    # Test Webhook with Mock
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        webhook_res = dispatcher.dispatch_webhook("https://hooks.slack.com/services/test/mock", payload)
        assert webhook_res is True

    print("✓ Secondary notification dispatcher passed.")

if __name__ == "__main__":
    test_hardware_actuation_fallback()
    test_webhook_and_sms_notifications()