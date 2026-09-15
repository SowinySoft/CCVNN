from enum import Enum
import logging
import os
import threading
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger("WatcherEngine")


class AlertAction(Enum):
    NONE = 0
    INITIAL_TRIGGER = 1
    HEARTBEAT_REMINDER = 2
    CLEAR = 3


class WatcherEngine:
    """State suppression engine supporting both static and dynamic camera onboarding."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.lock = threading.Lock()

        # Config structures
        self.rules: Dict[str, dict] = {}
        self.default_rule: dict = {}

        # Dynamic register tracking
        self.assigned_registers: Dict[int, str] = {}
        self.registered_cameras: Dict[str, dict] = {}

        # Per-camera state tracking
        self.state_history: Dict[str, int] = {}
        self.active_alerts: Dict[str, bool] = {}
        self.frames_since_last_alert: Dict[str, int] = {}

        self._load_config()

    def _load_config(self):
        """Loads rule definitions, auto-creating default config if file is missing."""
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

        if not os.path.exists(self.config_path):
            default_data = {
                "default_rule": {
                    "persistence_threshold": 2,
                    "heartbeat_interval": 10,
                    "plc_register_base": 40001,
                    "mqtt_topic_template": "factory/{cam_id}/hazard",
                },
                "rules": {
                    "cam_01": {
                        "persistence_threshold": 2,
                        "heartbeat_interval": 10,
                        "plc_register": 40001,
                        "mqtt_topic": "factory/cam_01/hazard",
                    }
                },
            }
            with open(self.config_path, "w") as f:
                yaml.dump(default_data, f, default_flow_style=False)
            logger.info(
                f"Created default configuration file at: {self.config_path}"
            )

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        self.default_rule = data.get(
            "default_rule",
            {
                "persistence_threshold": 2,
                "heartbeat_interval": 10,
                "plc_register_base": 40001,
                "mqtt_topic_template": "factory/{cam_id}/hazard",
            },
        )

        static_rules = data.get("rules", {})
        for cam_id, cfg in static_rules.items():
            self.register_camera(cam_id, custom_config=cfg)

    def register_camera(
        self, cam_id: str, custom_config: Optional[dict] = None
    ) -> dict:
        """Dynamically registers a new camera stream thread-safely."""
        with self.lock:
            if cam_id in self.registered_cameras:
                return self.registered_cameras[cam_id]

            if custom_config:
                cfg = custom_config.copy()
            else:
                # Auto-generate dynamic configuration using default_rule template
                base_reg = self.default_rule.get("plc_register_base", 40001)
                assigned_offset = len(self.registered_cameras)
                target_reg = base_reg + assigned_offset

                cfg = {
                    "persistence_threshold": self.default_rule.get(
                        "persistence_threshold", 2
                    ),
                    "heartbeat_interval": self.default_rule.get(
                        "heartbeat_interval", 10
                    ),
                    "plc_register": target_reg,
                    "mqtt_topic": self.default_rule.get(
                        "mqtt_topic_template", "factory/{cam_id}/hazard"
                    ).format(cam_id=cam_id),
                }

            # Register collision check
            plc_reg = cfg["plc_register"]
            if plc_reg in self.assigned_registers:
                existing_cam = self.assigned_registers[plc_reg]
                if existing_cam != cam_id:
                    raise ValueError(
                        f"PLC Register Collision! Register {plc_reg} requested by {cam_id} "
                        f"is already assigned to {existing_cam}."
                    )

            self.assigned_registers[plc_reg] = cam_id
            self.registered_cameras[cam_id] = cfg
            self.rules[cam_id] = cfg

            # Initialize tracking states
            self.state_history[cam_id] = 0
            self.active_alerts[cam_id] = False
            self.frames_since_last_alert[cam_id] = 0

            logger.info(
                f"Registered camera '{cam_id}' -> PLC Reg: {plc_reg} | "
                f"Topic: {cfg['mqtt_topic']} | Persistence: {cfg['persistence_threshold']}"
            )
            return cfg

    def evaluate_rule(self, cam_id: str, raw_anomaly: bool) -> AlertAction:
        """Evaluates anomaly state using state-aware suppression."""
        if cam_id not in self.registered_cameras:
            self.register_camera(cam_id)

        cfg = self.registered_cameras[cam_id]
        p_thresh = cfg["persistence_threshold"]
        h_interval = cfg["heartbeat_interval"]

        with self.lock:
            # Update persistence counter
            if raw_anomaly:
                self.state_history[cam_id] += 1
            else:
                self.state_history[cam_id] = 0

            is_persistent = self.state_history[cam_id] >= p_thresh
            is_active = self.active_alerts[cam_id]

            # 1. INITIAL TRIGGER
            if is_persistent and not is_active:
                self.active_alerts[cam_id] = True
                self.frames_since_last_alert[cam_id] = 0
                return AlertAction.INITIAL_TRIGGER

            # 2. HEARTBEAT REMINDER
            elif is_persistent and is_active:
                self.frames_since_last_alert[cam_id] += 1
                if self.frames_since_last_alert[cam_id] >= h_interval:
                    self.frames_since_last_alert[cam_id] = 0
                    return AlertAction.HEARTBEAT_REMINDER
                return AlertAction.NONE

            # 3. CLEAR
            elif not is_persistent and is_active:
                self.active_alerts[cam_id] = False
                self.frames_since_last_alert[cam_id] = 0
                return AlertAction.CLEAR

            return AlertAction.NONE