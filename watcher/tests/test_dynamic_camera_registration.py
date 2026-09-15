from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "watcher" / "src"
CONFIG_PATH = PROJECT_ROOT / "config" / "watcher_stress_test_rules.yaml"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from WatcherEngine import AlertAction, WatcherEngine


def test_dynamic_registration():
    engine = WatcherEngine(config_path=str(CONFIG_PATH))

    # Dynamically register streams not declared in static YAML config
    dynamic_cams = ["cam_dynamic_99", "cam_dynamic_100", "cam_dynamic_101"]

    print("\n--- Testing Dynamic Stream Onboarding ---")
    for cam in dynamic_cams:
        cfg = engine.register_camera(cam)
        print(
            f"✓ Onboarded {cam}: Reg={cfg['plc_register']}, Topic={cfg['mqtt_topic']}"
        )

    # Test state evaluation on dynamically onboarded stream
    print("\n--- Testing Rule Evaluation on Dynamic Stream ---")
    res1 = engine.evaluate_rule("cam_dynamic_99", raw_anomaly=True)
    res2 = engine.evaluate_rule("cam_dynamic_99", raw_anomaly=True)
    print(
        f"Frame 1 (Persistence 1/2): {res1.name}"
    )  # Expected: NONE (Persistence building)
    print(
        f"Frame 2 (Persistence 2/2): {res2.name}"
    )  # Expected: INITIAL_TRIGGER

    assert res2 == AlertAction.INITIAL_TRIGGER
    print(
        "\n✓ SUCCESS: Dynamic camera onboarding & rule evaluation passed without static config entries."
    )


if __name__ == "__main__":
    test_dynamic_registration()