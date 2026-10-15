import time
from watcher.src.overrides import (
    MaintenanceManager,
    HysteresisCooldownManager,
    ZoneAggregationEngine
)


def test_maintenance_suppression():
    print("\n--- Testing Maintenance Window Suppression API ---")
    mm = MaintenanceManager()
    now = time.time()

    # Schedule maintenance on zone_1 for 60 seconds
    mm.schedule_window("zone_1", start_time=now - 10, end_time=now + 50, reason="Sensor Calibration")

    is_maint, reason = mm.is_suppressed(zone_id="zone_1", sensor_id="cam_01", current_time=now)
    assert is_maint is True, "Zone_1 should be suppressed"
    print(f"✓ Suppressed active maintenance window: {reason}")

    is_maint_unrelated, _ = mm.is_suppressed(zone_id="zone_2", sensor_id="cam_02", current_time=now)
    assert is_maint_unrelated is False, "Zone_2 should NOT be suppressed"
    print("✓ Unrelated zone passed through successfully.")


def test_cooldown_hysteresis():
    print("\n--- Testing Cooldown & Hysteresis Timers ---")
    cooldown_mgr = HysteresisCooldownManager(default_cooldown_seconds=10.0)
    now = time.time()

    # First alert allowed
    allow1 = cooldown_mgr.check_and_update("zone_1", "Fire Hazard", current_time=now)
    assert allow1 is True, "First alert must be allowed"

    # Immediate second alert blocked
    allow2 = cooldown_mgr.check_and_update("zone_1", "Fire Hazard", current_time=now + 2.0)
    assert allow2 is False, "Alert within cooldown must be suppressed"
    print("✓ Muted duplicate alert payload within 2-second interval.")

    # Alert after cooldown allowed
    allow3 = cooldown_mgr.check_and_update("zone_1", "Fire Hazard", current_time=now + 11.0)
    assert allow3 is True, "Alert after cooldown expiration must be allowed"
    print("✓ Allowed new alert payload after 11-second interval (cooldown expired).")


def test_zone_aggregation_multi_camera():
    print("\n--- Testing Zone Aggregation & Multi-Camera Correlation ---")
    aggregator = ZoneAggregationEngine(correlation_window_seconds=5.0)
    now = time.time()

    # Camera 1 detects HIGH severity fire
    res1 = aggregator.register_and_correlate(
        zone_id="warehouse_east",
        sensor_id="cam_east_01",
        event_type="Fire / Smoke Hazard",
        severity="HIGH",
        current_time=now
    )
    assert res1["effective_severity"] == "HIGH"
    assert res1["is_multi_camera_escalated"] is False
    print("✓ Single camera detection retained HIGH severity.")

    # Camera 2 detects SAME fire in SAME zone within 2 seconds
    res2 = aggregator.register_and_correlate(
        zone_id="warehouse_east",
        sensor_id="cam_east_02",
        event_type="Fire / Smoke Hazard",
        severity="HIGH",
        current_time=now + 2.0
    )
    assert res2["effective_severity"] == "CRITICAL"
    assert res2["is_multi_camera_escalated"] is True
    assert res2["correlated_cameras_count"] == 2
    print("✓ Multi-camera observation (cam_east_01 + cam_east_02) escalated HIGH -> CRITICAL.")


if __name__ == "__main__":
    test_maintenance_suppression()
    test_cooldown_hysteresis()
    test_zone_aggregation_multi_camera()