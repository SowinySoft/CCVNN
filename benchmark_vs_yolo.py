import os
import re
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

SDK_IMAGE = "nvcr.io/nvidia/tritonserver:26.08-py3-sdk"

models = [
    {"name": "ccvnn", "input_name": "84", "input_shape": "1,4,64,64", "label": "CCVNN (Full)"},
    {"name": "ccvnn_v20_backbone", "input_name": "43", "input_shape": "1,4,64,64", "label": "CCVNN (Backbone)"},
    {"name": "yolov8n", "input_name": "images", "input_shape": "1,3,640,640", "label": "YOLOv8n Baseline"},
    {"name": "ccvnn_v21_dynamic", "input_name": "84", "input_shape": "1,4,64,64", "label": "CCVNN v21 Dynamic"},
    {"name": "ccvnn_v21_int8", "input_name": "84", "input_shape": "1,4,64,64", "label": "CCVNN v21 INT8"},
    {"name": "yolov8n_openvino", "input_name": "x.1", "input_shape": "1,3,640,640", "label": "YOLOv8n OpenVINO"}
]

concurrencies = [1, 3, 5, 7]

def run_single_benchmark(model, c):
    """Executes perf_analyzer inside Triton SDK container for a specific model and concurrency."""
    cmd = [
        "docker", "run", "--rm", "--net=host",
        "-v", f"{os.getcwd()}:/workspace",
        SDK_IMAGE, 
        "perf_analyzer",
        "-m", model["name"],
        "-i", "grpc",
        "-u", "localhost:8001",
        "-b", "1",
        "--async",
        "--concurrency-range", str(c),
        "--measurement-mode", "time_windows",
        "--measurement-interval", "2000",
        "--stability-percentage", "99",
        "--shape", f"{model['input_name']}:{model['input_shape']}"
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
        output = res.stdout + "\n" + res.stderr
        
        if "Request for unknown model" in output or "is not ready" in output:
            print(f"[SKIP] Model '{model['name']}' is not loaded on Triton Server.\n", flush=True)
            return None, False

        throughput_match = re.search(r"(?:throughput|inferences/second):\s+([\d\.]+)", output, re.IGNORECASE)
        latency_match = re.search(r"avg(?:erage)?\s+latency:\s+([\d\.]+)\s*(usec|ms)?", output, re.IGNORECASE)
        p95_match = re.search(r"p95\s*(?:latency)?:\s+([\d\.]+)\s*(usec|ms)?", output, re.IGNORECASE)
        if not p95_match:
            p95_match = re.search(r"latency\s*p95:\s+([\d\.]+)\s*(usec|ms)?", output, re.IGNORECASE)
        
        if throughput_match:
            throughput = float(throughput_match.group(1))
            
            lat_unit = latency_match.group(2).lower() if (latency_match and latency_match.group(2)) else "usec"
            avg_lat = float(latency_match.group(1)) / (1000.0 if lat_unit == "usec" else 1.0) if latency_match else 0.0
            
            p95_unit = p95_match.group(2).lower() if (p95_match and p95_match.group(2)) else "usec"
            p95_lat = float(p95_match.group(1)) / (1000.0 if p95_unit == "usec" else 1.0) if p95_match else 0.0
            
            print(f"DONE -> Throughput: {throughput:.2f} infer/sec | Avg Latency: {avg_lat:.2f} ms", flush=True)
            return {
                "Model": model["label"],
                "Concurrency": c,
                "Throughput": throughput,
                "Avg_Latency_ms": avg_lat,
                "p95_Latency_ms": p95_lat
            }, True
        else:
            print(f"[ERROR] Could not parse metrics for {model['label']} (C={c}). Output tail:", flush=True)
            clean_lines = [line for line in output.splitlines() if line.strip()]
            for line in clean_lines[-3:]:
                print(f"     | {line}", flush=True)
            return None, True

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] >90s limit reached for {model['label']} at C={c}. Skipping.", flush=True)
        return None, True
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return None, True


if __name__ == '__main__':
    # Force process spawning to avoid C++ thread state corruption in Python 3.14
    mp.set_start_method('spawn', force=True)

    results = []
    print("=== STARTING TRITON COMPARATIVE BENCHMARK ===", flush=True)

    # Execute benchmarks sequentially across models using ThreadPoolExecutor for IO safety
    with ThreadPoolExecutor(max_workers=1) as executor:
        for model in models:
            model_available = True
            for c in concurrencies:
                if not model_available:
                    break

                print(f"Testing {model['label']} @ Concurrency C={c}... ", end="", flush=True)
                future = executor.submit(run_single_benchmark, model, c)
                res_data, model_available = future.result()
                
                if res_data:
                    results.append(res_data)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_4way_results.csv", index=False)
    print("\n=== BENCHMARK COMPLETE: RESULTS SAVED TO benchmark_4way_results.csv ===", flush=True)
    if not df.empty:
        print(df.to_string(index=False), flush=True)