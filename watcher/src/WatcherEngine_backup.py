import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AlertAction(Enum):
    NONE = 0
    INITIAL_TRIGGER = 1
    HEARTBEAT_REMINDER = 2
    CLEAR = 3


class HazardStateTracker:

    def __init__(
        self, persistence_threshold: int = 2, heartbeat_interval: int = 10
    ):
        self.persistence_threshold = persistence_threshold
        self.heartbeat_interval = heartbeat_interval
        self.consecutive_detects = 0
        self.frames_since_last_alert = 0
        self.is_active = False

    def process_frame(self, hazard_detected: bool) -> Tuple[AlertAction, str]:
        if hazard_detected:
            self.consecutive_detects += 1

            if not self.is_active:
                if self.consecutive_detects >= self.persistence_threshold:
                    self.is_active = True
                    self.frames_since_last_alert = 0
                    return (
                        AlertAction.INITIAL_TRIGGER,
                        f"Hazard initial trigger ({self.consecutive_detects}/{self.persistence_threshold} frames)",
                    )
            else:
                self.frames_since_last_alert += 1
                if self.frames_since_last_alert >= self.heartbeat_interval:
                    self.frames_since_last_alert = 0
                    return (
                        AlertAction.HEARTBEAT_REMINDER,
                        f"Heartbeat reminder ({self.heartbeat_interval} frames elapsed)",
                    )
        else:
            if self.is_active:
                self.is_active = False
                self.consecutive_detects = 0
                self.frames_since_last_alert = 0
                return (
                    AlertAction.CLEAR,
                    "Hazard cleared - resetting state",
                )

            self.consecutive_detects = 0

        return AlertAction.NONE, ""


class WatcherEngine:

    def __init__(self, config_path: Optional[str] = None):
        self.trackers: Dict[str, HazardStateTracker] = {}
        self.rules = {}

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                self.rules = config_data.get("rules", {})

    def evaluate_rule(
        self, target_id: str, hazard_detected: bool
    ) -> AlertAction:
        rule_cfg = self.rules.get(target_id, {})
        persistence = rule_cfg.get("persistence_threshold", 2)
        heartbeat = rule_cfg.get("heartbeat_interval", 10)

        if target_id not in self.trackers:
            self.trackers[target_id] = HazardStateTracker(
                persistence_threshold=persistence, heartbeat_interval=heartbeat
            )

        action, reason = self.trackers[target_id].process_frame(hazard_detected)
        return action
