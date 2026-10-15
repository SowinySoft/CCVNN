import os
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

try:
    import psutil
except ImportError:
    psutil = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WatcherTelemetry")


@dataclass
class CameraHealthStatus:
    is_blurred: bool
    blur_score: float
    is_occluded: bool
    occlusion_ratio: float
    status: str  # "HEALTHY", "WARNING", "CRITICAL"


@dataclass
class SystemHardwareStatus:
    cpu_temp_c: float
    cpu_usage_pct: float
    ram_usage_pct: float
    thermal_throttling_risk: bool
    status: str  # "HEALTHY", "WARNING", "CRITICAL"


class CameraHealthMonitor:
    """Analyzes incoming video frames for lens blur, spray/fog, or physical occlusion."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        occlusion_dark_threshold: float = 30.0,
        occlusion_area_ratio: float = 0.85
    ):
        self.blur_threshold = blur_threshold
        self.occlusion_dark_threshold = occlusion_dark_threshold
        self.occlusion_area_ratio = occlusion_area_ratio

    def analyze_frame(self, frame: np.ndarray) -> CameraHealthStatus:
        if frame is None or frame.size == 0:
            return CameraHealthStatus(
                is_blurred=True, blur_score=0.0,
                is_occluded=True, occlusion_ratio=1.0,
                status="CRITICAL"
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurred = blur_score < self.blur_threshold

        dark_pixels = np.sum(gray < self.occlusion_dark_threshold)
        total_pixels = gray.size
        occlusion_ratio = float(dark_pixels / total_pixels) if total_pixels > 0 else 1.0
        is_occluded = occlusion_ratio >= self.occlusion_area_ratio

        if is_occluded or blur_score < (self.blur_threshold * 0.3):
            status = "CRITICAL"
        elif is_blurred:
            status = "WARNING"
        else:
            status = "HEALTHY"

        if status != "HEALTHY":
            logger.warning(
                f"Camera Health Anomaly ({status}): Blur Score={blur_score:.1f} (thresh={self.blur_threshold}), "
                f"Occlusion Ratio={occlusion_ratio * 100:.1f}%"
            )

        return CameraHealthStatus(
            is_blurred=is_blurred,
            blur_score=blur_score,
            is_occluded=is_occluded,
            occlusion_ratio=occlusion_ratio,
            status=status
        )


class SystemHardwareMonitor:
    """Monitors host CPU/GPU thermal sensors, memory limits, and load stats."""

    def __init__(self, warning_temp_c: float = 75.0, critical_temp_c: float = 88.0):
        self.warning_temp = warning_temp_c
        self.critical_temp = critical_temp_c

    def _read_cpu_temperature(self) -> float:
        try:
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                with open(thermal_path, "r") as f:
                    return float(f.read().strip()) / 1000.0
        except Exception:
            pass

        if psutil:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current and entry.current > 0:
                                return float(entry.current)
            except Exception:
                pass

        return 48.5

    def _read_ram_usage(self) -> float:
        if psutil:
            return float(psutil.virtual_memory().percent)
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            mem = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    mem[parts[0].strip()] = int(parts[1].split()[0])
            total = mem.get("MemTotal", 1)
            free = mem.get("MemAvailable", mem.get("MemFree", 0))
            return float(((total - free) / total) * 100.0)
        except Exception:
            return 35.0

    def _read_cpu_usage(self) -> float:
        if psutil:
            return float(psutil.cpu_percent(interval=None))
        try:
            load1, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return min(100.0, float((load1 / cpu_count) * 100.0))
        except Exception:
            return 12.0

    def get_telemetry(self) -> SystemHardwareStatus:
        cpu_temp = self._read_cpu_temperature()
        cpu_usage = self._read_cpu_usage()
        ram_usage = self._read_ram_usage()

        throttling_risk = cpu_temp >= self.warning_temp

        if cpu_temp >= self.critical_temp or ram_usage >= 95.0:
            status = "CRITICAL"
        elif cpu_temp >= self.warning_temp or ram_usage >= 85.0:
            status = "WARNING"
        else:
            status = "HEALTHY"

        if status != "HEALTHY":
            logger.warning(
                f"Hardware Telemetry Alert ({status}): Temp={cpu_temp:.1f}°C, "
                f"CPU={cpu_usage:.1f}%, RAM={ram_usage:.1f}%"
            )

        return SystemHardwareStatus(
            cpu_temp_c=cpu_temp,
            cpu_usage_pct=cpu_usage,
            ram_usage_pct=ram_usage,
            thermal_throttling_risk=throttling_risk,
            status=status
        )