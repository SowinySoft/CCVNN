from pathlib import Path
import sys

# Ensure src/ is added to Python Path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "watcher" / "src"
CONFIG_PATH = PROJECT_ROOT / "config" / "watcher_stress_test_rules.yaml"

sys.path.append(str(SRC_PATH))

from WatcherEngine import AlertAction, WatcherEngine


def run_suppression_test():
    engine = WatcherEngine(config_path=str(CONFIG_PATH))
    target_id = "cam_01_zone_A"

    # Simulation sequence: 14 hazard frames, followed by 4 clear frames
    frame_sequence = [True] * 14 + [False] * 4

    print(f"\nLoaded Config for target '{target_id}': {engine.rules.get(target_id, {})}\n")
    print(f"{'Frame':<7} | {'Detected':<9} | {'Action':<20} | {'Dispatch Event'}")
    print("-" * 75)

    for idx, hazard_present in enumerate(frame_sequence, start=1):
        action = engine.evaluate_rule(target_id, hazard_present)

        status_msg = "SUPPRESSED (No Action)"
        if action == AlertAction.INITIAL_TRIGGER:
            status_msg = "DISPATCH: Reg 40001 = 1 | Topic: factory/cam_01/hazard [ACTIVE]"
        elif action == AlertAction.HEARTBEAT_REMINDER:
            status_msg = "DISPATCH: Reg 40001 = 1 | Topic: factory/cam_01/hazard [HEARTBEAT]"
        elif action == AlertAction.CLEAR:
            status_msg = "DISPATCH: Reg 40001 = 0 | Topic: factory/cam_01/hazard [CLEAR]"

        print(f"Frame {idx:<2} | {str(hazard_present):<9} | {action.name:<20} | {status_msg}")


if __name__ == "__main__":
    run_suppression_test()