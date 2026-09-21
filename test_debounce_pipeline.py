import triton_client
import json
import time

IMAGE_PATH = "photos/spark_test.jpeg"

print("==================================================")
print("  CCVNN V17 - 3-FRAME DEBOUNCE & PLC TEST PIPELINE")
print("==================================================\n")

for frame_idx in range(1, 4):
    print(f"--- [Frame {frame_idx} / 3] Sending frame to Triton ---")
    
    telemetry, coil = triton_client.send_frame_to_triton(IMAGE_PATH)
    
    rt_telemetry = telemetry.get("realtime_telemetry", {})
    plc_actuation = telemetry.get("modbus_tcp_plc_actuation", {})
    
    eval_state = rt_telemetry.get("evaluation_state", "N/A")
    debounce_count = rt_telemetry.get("debounce_consecutive_frames", "N/A")
    latency = rt_telemetry.get("pipeline_latency", "N/A")
    actuation_status = plc_actuation.get("actuation_status", "N/A")
    plc_target = plc_actuation.get("target_plc", "N/A")
    
    print(f"  Evaluation State : {eval_state}")
    print(f"  Debounce Counter : {debounce_count}")
    print(f"  Pipeline Latency : {latency}")
    print(f"  PLC Target       : {plc_target}")
    print(f"  Register 40001   : {coil} ({actuation_status})\n")
    
    if frame_idx < 3:
        time.sleep(0.1)

print("==================================================")
if coil == 1 or "TRIGGERED" in actuation_status.upper() or "3 / 3" in debounce_count:
    print("SUCCESS: 3-Frame Debounce Threshold Reached! PLC Register 40001 Actuated.")
else:
    print("STATUS: Processed 3 frames. Inspect telemetry log above.")
print("==================================================")
