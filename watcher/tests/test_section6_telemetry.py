import numpy as np
import cv2
from watcher.src.telemetry import CameraHealthMonitor, SystemHardwareMonitor


def test_camera_blur_and_occlusion():
    print("\n--- Testing Camera Blur & Occlusion Detection ---")
    monitor = CameraHealthMonitor(blur_threshold=100.0, occlusion_area_ratio=0.80)

    # 1. Healthy sharp image (50/50 checkerboard pattern for equal light/dark ratio)
    grid_size = 30
    sharp_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(0, 300, grid_size):
        for j in range(0, 300, grid_size):
            if (i // grid_size + j // grid_size) % 2 == 0:
                sharp_frame[i:i + grid_size, j:j + grid_size] = 255

    status_sharp = monitor.analyze_frame(sharp_frame)
    assert status_sharp.status == "HEALTHY", f"Expected HEALTHY, got {status_sharp.status}"
    print(f"✓ Crisp frame recognized as HEALTHY (Blur Score: {status_sharp.blur_score:.1f}, Dark Ratio: {status_sharp.occlusion_ratio*100:.1f}%)")

    # 2. Heavily blurred frame (Gaussian blur flattens high-frequency edges)
    blurred_frame = cv2.GaussianBlur(sharp_frame, (51, 51), 0)
    status_blur = monitor.analyze_frame(blurred_frame)
    assert status_blur.is_blurred is True
    print(f"✓ Blurred frame detected (Blur Score: {status_blur.blur_score:.1f} < threshold 100.0)")

    # 3. Occluded frame (Lens blocked / dark black cover)
    occluded_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    status_occluded = monitor.analyze_frame(occluded_frame)
    assert status_occluded.is_occluded is True
    assert status_occluded.status == "CRITICAL"
    print(f"✓ Lens occlusion/blindness detected (Dark ratio: {status_occluded.occlusion_ratio*100:.1f}%)")


def test_system_hardware_telemetry():
    print("\n--- Testing Edge System Hardware Telemetry ---")
    hw_monitor = SystemHardwareMonitor(warning_temp_c=70.0, critical_temp_c=85.0)
    status = hw_monitor.get_telemetry()

    assert isinstance(status.cpu_temp_c, float)
    assert isinstance(status.cpu_usage_pct, float)
    assert isinstance(status.ram_usage_pct, float)

    print(
        f"✓ Edge Hardware Telemetry Readouts:\n"
        f"   - CPU Temp: {status.cpu_temp_c:.1f}°C\n"
        f"   - CPU Usage: {status.cpu_usage_pct:.1f}%\n"
        f"   - RAM Usage: {status.ram_usage_pct:.1f}%\n"
        f"   - Thermal Throttling Risk: {status.thermal_throttling_risk}\n"
        f"   - Status: {status.status}"
    )


if __name__ == "__main__":
    test_camera_blur_and_occlusion()
    test_system_hardware_telemetry()