import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SeverityRule:
    class_id: int
    display_name: str
    min_confidence: float
    consecutive_frames: int
    severity: str
    action_type: str

class DynamicSeverityEngine:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.rules: Dict[str, SeverityRule] = {}
        self.default_rule: Optional[SeverityRule] = None
        self.load_config()

    def load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        default_cfg = data.get("default_debouncing") or {}
        self.default_rule = SeverityRule(
            class_id=-1,
            display_name="Default Fallback",
            min_confidence=default_cfg.get("min_confidence", 0.50),
            consecutive_frames=default_cfg.get("consecutive_frames", 5),
            severity="WARNING",
            action_type="LOG_ONLY"
        )

        self.rules.clear()
        event_classes = data.get("event_classes") or {}
        
        for idx, (class_name, cfg) in enumerate(event_classes.items()):
            cfg = cfg or {}
            self.rules[class_name] = SeverityRule(
                class_id=cfg.get("class_id", idx),
                display_name=cfg.get("display_name", class_name),
                min_confidence=cfg.get("min_confidence", self.default_rule.min_confidence),
                consecutive_frames=cfg.get("consecutive_frames", self.default_rule.consecutive_frames),
                severity=cfg.get("severity", "WARNING"),
                action_type=cfg.get("action_type", "LOG_ONLY")
            )

    def evaluate(self, event_type: str) -> SeverityRule:
        return self.rules.get(event_type, self.default_rule)