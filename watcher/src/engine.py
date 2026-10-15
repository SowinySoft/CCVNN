import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class SeverityRule:
    class_id: int
    display_name: str
    min_confidence: float
    consecutive_frames: int
    severity: str
    action_type: str
    plc_config: Dict[str, Any] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)


class DynamicSeverityEngine:
    """Evaluates raw neural detections against configured debouncing & severity rules."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.rules: Dict[str, SeverityRule] = {}
        self.default_rule: Optional[SeverityRule] = None
        self.frame_counters: Dict[str, int] = {}
        self.trackers: Dict[Any, Any] = {}

        if self.config_path:
            self.load_config(self.config_path)

    def load_config(self, config_path: Optional[str] = None) -> None:
        """Loads or reloads severity rule configurations from a YAML file."""
        path_to_load = config_path or self.config_path
        if not path_to_load or not os.path.exists(path_to_load):
            raise FileNotFoundError(
                f"Configuration file not found: {path_to_load}"
            )

        self.config_path = path_to_load

        with open(path_to_load, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        default_cfg = data.get("default_debouncing") or {}
        self.default_rule = SeverityRule(
            class_id=-1,
            display_name="Default Fallback",
            min_confidence=default_cfg.get("min_confidence", 0.50),
            consecutive_frames=default_cfg.get("consecutive_frames", 5),
            severity="WARNING",
            action_type="LOG_ONLY",
            plc_config=default_cfg.get("plc_config", {}),
            topics=default_cfg.get("topics", []),
        )

        self.rules.clear()
        event_classes = data.get("event_classes") or {}

        for idx, (class_name, cfg) in enumerate(event_classes.items()):
            cfg = cfg or {}
            self.rules[class_name] = SeverityRule(
                class_id=cfg.get("class_id", idx),
                display_name=cfg.get("display_name", class_name),
                min_confidence=cfg.get(
                    "min_confidence", self.default_rule.min_confidence
                ),
                consecutive_frames=cfg.get(
                    "consecutive_frames", self.default_rule.consecutive_frames
                ),
                severity=cfg.get("severity", "WARNING"),
                action_type=cfg.get("action_type", "LOG_ONLY"),
                plc_config=cfg.get("plc_config", {}),
                topics=cfg.get("topics", []),
            )

    def evaluate(self, event_type: str) -> SeverityRule:
        """Retrieves matching severity rule or returns system fallback."""
        return self.rules.get(event_type, self.default_rule)

    def process_detection(
        self,
        class_name: str,
        confidence: float,
        tracking_id: Optional[str] = None,
    ) -> Optional[SeverityRule]:
        """Evaluates detection persistence and returns rule if threshold is satisfied."""
        rule = self.evaluate(class_name)
        key = f"{class_name}:{tracking_id}" if tracking_id else class_name

        if not rule or confidence < rule.min_confidence:
            self.frame_counters[key] = 0
            return None

        self.frame_counters[key] = self.frame_counters.get(key, 0) + 1

        if self.frame_counters[key] >= rule.consecutive_frames:
            return rule

        return None

    def reset_tracker(
        self, tracking_id: str, class_name: Optional[str] = None
    ) -> None:
        """Resets debounce counters and object tracking states."""
        key_str = f"{class_name}:{tracking_id}" if class_name else tracking_id
        self.frame_counters.pop(key_str, None)

        if hasattr(self, "trackers"):
            key = (tracking_id, class_name) if class_name else tracking_id
            self.trackers.pop(key, None)
            self.trackers.pop(tracking_id, None)