import os
import glob
import json
import csv
import time
from datetime import datetime
import triton_client
import paho.mqtt.client as mqtt

# --- Configurations ---
TARGET_DIR = "photos"
TXT_REPORT = "batch_verifier_report.txt"
CSV_REPORT = "batch_verifier_report.csv"
JSON_REPORT = "batch_verifier_report.json"

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "ccvnn/v17/industrial/audit_report"


def find_all_images(directory):
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(image_files)


def publish_mqtt_report(json_filepath):
    print("\n--- Initiating Automated MQTT Report Publish ---")
    if not os.path.exists(json_filepath):
        print(f"[X] Error: {json_filepath} missing. Skipping MQTT publish.")
        return

    with open(json_filepath, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    payload_str = json.dumps(report_data, indent=2)

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="CCVNN_Audit_Publisher")
    except AttributeError:
        client = mqtt.Client(client_id="CCVNN_Audit_Publisher")

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[✓] Connected to MQTT Broker ({MQTT_BROKER}:{MQTT_PORT})")
        else:
            print(f"[X] MQTT Connection failed with code: {rc}")

    def on_publish(client, userdata, mid, reason_code=None, properties=None):
        print(f"[✓] Successfully published report payload to topic '{MQTT_TOPIC}' (MID: {mid})")

    client.on_connect = on_connect
    client.on_publish = on_publish

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()

        info = client.publish(MQTT_TOPIC, payload_str, qos=1)
        info.wait_for_publish(timeout=5)

        client.loop_stop()
        client.disconnect()
        print("[✓] MQTT Client disconnected cleanly.")
    except Exception as e:
        print(f"[X] Failed to publish via MQTT: {e}")


def run_batch_verification():
    print("==================================================")
    print("  CCVNN V17 - RECURSIVE BATCH IMAGE AUDIT")
    print("==================================================\n")
    
    images = find_all_images(TARGET_DIR)
    
    if not images:
        print(f"No image files found in '{TARGET_DIR}' directory.")
        return

    print(f"Found {len(images)} image(s) in '{TARGET_DIR}'. Processing...\n")

    report_entries = []
    total_processed = 0
    hazard_count = 0
    clear_count = 0

    for idx, img_path in enumerate(images, start=1):
        filename = os.path.basename(img_path)
        print(f"[{idx}/{len(images)}] Verifying: {img_path}")
        
        try:
            telemetry, coil = triton_client.send_frame_to_triton(img_path)
            
            rt_telemetry = telemetry.get("realtime_telemetry", {})
            plc_actuation = telemetry.get("modbus_tcp_plc_actuation", {})
            
            eval_state = rt_telemetry.get("evaluation_state", "UNKNOWN")
            hazard_density = rt_telemetry.get("peak_localized_hazard_density", "N/A")
            latency = rt_telemetry.get("pipeline_latency", "N/A")
            actuation_status = plc_actuation.get("actuation_status", "N/A")
            
            if coil == 1 or "HAZARD" in str(eval_state).upper():
                status_summary = "HAZARD DETECTED"
                hazard_count += 1
            else:
                status_summary = "CLEAR / NORMAL"
                clear_count += 1

            entry = {
                "image_id": f"IMG-{idx:04d}",
                "file_name": filename,
                "relative_path": img_path,
                "classification_status": status_summary,
                "evaluation_state": eval_state,
                "hazard_density": hazard_density,
                "latency": latency,
                "register_40001_coil": coil,
                "actuation_status": actuation_status
            }
            report_entries.append(entry)
            total_processed += 1
            
        except Exception as e:
            print(f"  --> Error processing {img_path}: {e}")
            report_entries.append({
                "image_id": f"IMG-{idx:04d}",
                "file_name": filename,
                "relative_path": img_path,
                "classification_status": f"ERROR: {str(e)}",
                "evaluation_state": "FAILED",
                "hazard_density": "N/A",
                "latency": "N/A",
                "register_40001_coil": 0,
                "actuation_status": "N/A"
            })

    timestamp = datetime.now().isoformat()

    # 1. Export JSON Report
    json_payload = {
        "metadata": {
            "title": "CCVNN V17 Automated Batch Audit Report",
            "timestamp": timestamp,
            "target_directory": os.path.abspath(TARGET_DIR),
            "summary": {
                "total_processed": total_processed,
                "hazards_detected": hazard_count,
                "clear_count": clear_count
            }
        },
        "audit_records": report_entries
    }
    
    with open(JSON_REPORT, "w", encoding="utf-8") as f_json:
        json.dump(json_payload, f_json, indent=2)
    print(f"[✓] Exported JSON report to '{JSON_REPORT}'")

    # 2. Export CSV Report
    csv_headers = [
        "image_id", "file_name", "relative_path", "classification_status",
        "evaluation_state", "hazard_density", "latency", "register_40001_coil", "actuation_status"
    ]
    
    with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(report_entries)
    print(f"[✓] Exported CSV report to '{CSV_REPORT}'")

    # 3. Export TXT Report
    with open(TXT_REPORT, "w", encoding="utf-8") as f_txt:
        f_txt.write("====================================================================================================\n")
        f_txt.write(f"                        CCVNN V17 - AUTOMATED BATCH AUDIT REPORT\n")
        f_txt.write(f" Generated: {timestamp}\n")
        f_txt.write(f" Target Folder: {os.path.abspath(TARGET_DIR)}\n")
        f_txt.write(f" Total Images Processed: {total_processed} | Hazards Detected: {hazard_count} | Clear: {clear_count}\n")
        f_txt.write("====================================================================================================\n\n")
        
        f_txt.write(f"{'ID':<10} | {'FILE NAME':<25} | {'STATUS':<20} | {'HAZARD DENSITY':<15} | {'LATENCY':<10} | {'COIL 40001':<10}\n")
        f_txt.write("-" * 100 + "\n")
        
        for item in report_entries:
            f_txt.write(
                f"{item['image_id']:<10} | "
                f"{item['file_name'][:25]:<25} | "
                f"{item['classification_status'][:20]:<20} | "
                f"{str(item['hazard_density']):<15} | "
                f"{str(item['latency']):<10} | "
                f"{str(item['register_40001_coil']):<10}\n"
            )
            
        f_txt.write("\n" + "=" * 100 + "\n")
        f_txt.write("                                     END OF VERIFICATION REPORT\n")
        f_txt.write("====================================================================================================\n")
    print(f"[✓] Exported TXT report to '{TXT_REPORT}'")

    # --- Automated MQTT Trigger Step ---
    publish_mqtt_report(JSON_REPORT)

    print("\n==================================================")
    print("ALL STEPS COMPLETED: Audit -> Reports -> MQTT Publish")
    print("==================================================")


if __name__ == "__main__":
    run_batch_verification()
