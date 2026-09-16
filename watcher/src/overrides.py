import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpatialAndOverrides")


@dataclass
class MaintenanceWindow:
    target_id: str  # Can be zone_id, sensor_id, or "GLOBAL"
    start_time: float  # POSIX timestamp
    end_time: float    # POSIX timestamp
    reason: str


class MaintenanceManager:
    """Manages scheduled maintenance windows to suppress false alarms during servicing."""

    def __init__(self):
        self._windows: List[MaintenanceWindow] = []

    def schedule_window(
        self,
        target_id: str,
        start_time: float,
        end_time: float,
        reason: str = "Scheduled Maintenance"
    ) -> MaintenanceWindow:
        window = MaintenanceWindow(
            target_id=target_id,
            start_time=start_time,
            end_time=end_time,
            reason=reason
        )
        self._windows.append(window)
        logger.info(
            f"Maintenance scheduled for '{target_id}' from "
            f"{datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()} to "
            f"{datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat()} ({reason})"
        )
        return window

    def is_suppressed(self, zone_id: str, sensor_id: str, current_time: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        now = current_time if current_time is not None else time.time()
        
        # Clean expired windows
        self._windows = [w for w in self._windows if w.end_time > now]

        for window in self._windows:
            if window.start_time <= now <= window.end_time:
                if window.target_id in (zone_id, sensor_id, "GLOBAL"):
                    return True, f"Suppressed by maintenance window on '{window.target_id}' ({window.reason})"

        return False, None


class HysteresisCooldownManager:
    """Prevents alert payload spamming using configurable per-key cooldown timers."""

    def __init__(self, default_cooldown_seconds: float = 15.0):
        self.default_cooldown = default_cooldown_seconds
        # Key: (target_id, event_type) -> last_alert_timestamp
        self._last_alert_times: Dict[Tuple[str, str], float] = {}

    def check_and_update(
        self,
        target_id: str,
        event_type: str,
        cooldown_seconds: Optional[float] = None,
        current_time: Optional[float] = None
    ) -> bool:
        """Returns True if alert is ALLOWED (cooldown expired), False if in cooldown (SPAM/MUTED)."""
        now = current_time if current_time is not None else time.time()
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.default_cooldown
        key = (target_id, event_type)

        last_time = self._last_alert_times.get(key, 0.0)
        elapsed = now - last_time

        if elapsed >= cooldown:
            self._last_alert_times[key] = now
            return True
        else:
            logger.debug(
                f"Cooldown active for key {key}: {cooldown - elapsed:.1f}s remaining. Alert muted."
            )
            return False


@dataclass
class ZoneDetectionRecord:
    sensor_id: str
    event_type: str
    severity: str
    timestamp: float


class ZoneAggregationEngine:
    """Correlates multi-camera observations in the same spatial zone to escalate severity or aggregate alerts."""

    def __init__(self, correlation_window_seconds: float = 5.0):
        self.correlation_window = correlation_window_seconds
        # Key: zone_id -> List[ZoneDetectionRecord]
        self._zone_history: Dict[str, List[ZoneDetectionRecord]] = {}

    def register_and_correlate(
        self,
        zone_id: str,
        sensor_id: str,
        event_type: str,
        severity: str,
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Registers a camera detection, checks multi-camera spatial overlap, and returns aggregated zone status."""
        now = current_time if current_time is not None else time.time()

        if zone_id not in self._zone_history:
            self._zone_history[zone_id] = []

        # Prune records older than the correlation window
        cutoff = now - self.correlation_window
        self._zone_history[zone_id] = [
            rec for rec in self._zone_history[zone_id] if rec.timestamp >= cutoff
        ]

        # Add current detection
        record = ZoneDetectionRecord(
            sensor_id=sensor_id,
            event_type=event_type,
            severity=severity,
            timestamp=now
        )
        self._zone_history[zone_id].append(record)

        # Calculate distinct active cameras in this zone reporting the same event type
        matching_sensors = {
            rec.sensor_id
            for rec in self._zone_history[zone_id]
            if rec.event_type == event_type
        }

        camera_count = len(matching_sensors)
        escalated = False
        final_severity = severity

        # Multi-camera correlation rule: 2+ cameras observing same hazard escalates HIGH -> CRITICAL
        if camera_count >= 2 and severity == "HIGH":
            final_severity = "CRITICAL"
            escalated = True
            logger.warning(
                f"MULTI-CAMERA CORRELATION ESCALATION in Zone '{zone_id}': "
                f"{camera_count} cameras ({matching_sensors}) detected '{event_type}'. Escalating HIGH -> CRITICAL."
            )

        return {
            "zone_id": zone_id,
            "event_type": event_type,
            "original_severity": severity,
            "effective_severity": final_severity,
            "correlated_cameras_count": camera_count,
            "active_sensors": list(matching_sensors),
            "is_multi_camera_escalated": escalated
        }