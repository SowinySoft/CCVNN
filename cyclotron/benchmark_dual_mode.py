#!/usr/bin/env python3
import os
import re
import subprocess
import argparse
import pandas as pd
from typing import Dict, Any

class TensorRTBenchmarkRunner:
    """
    Automates TensorRT engine compilation and benchmark profiling across precision modes 
    (FP32, FP16, INT8) using trtexec for the dual-mode model.
    """
    def __init__(self, onnx_path: str, calib_cache: str, duration_sec: int = 10):
        self.onnx_path = onnx_path
        self.calib_cache = calib_cache
        self.duration_sec = duration_sec
        self.input_shapes = "frame:1x3x256x256,v_cum:1x8"

    def run_benchmark(self, precision: str) -> Dict[str, Any]:
        engine_file = f"dual_mode_{precision.lower()}.engine"
        
        # Base trtexec CLI flags
        cmd = [
            "trtexec",
            f"--onnx={self.onnx_path}",
            f"--saveEngine={engine_file}",
            f"--shapes={self.input_shapes}",
            f"--duration={self.duration_sec}",
            "--warmUp=2000",
            "--avgRuns=100",
            "--useCudaGraph",  # Reduce launch overhead
            "--separateProfileRun"
        ]

        # Precision-specific flags
        if precision == "FP16":
            cmd.append("--fp16")
        elif precision == "INT8":
            cmd.extend(["--int8", "--fp16"]) # Allow FP16 fallback for non-quantized layers
            if os.path.exists(self.calib_cache):
                cmd.append(f"--calib={self.calib_cache}")

        print(f"\n[BENCHMARK] Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
            return self._parse_trtexec_output(result.stdout, precision)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] trtexec failed for precision {precision}:\n{e.output}")
            return {}

    def _parse_trtexec_output(self, output: str, precision: str) -> Dict[str, Any]:
        """Extracts latency percentiles and throughput from raw trtexec stdout logs."""
        metrics = {"Precision": precision}

        # Regex patterns for throughput and compute latency
        throughput_match = re.search(r"Throughput:\s+([\d.]+)\s+qps", output)
        latency_mean_match = re.search(r"GPU Compute Time:.*mean\s+=\s+([\d.]+)\s+ms", output)
        latency_p50_match  = re.search(r"GPU Compute Time:.*median\s+=\s+([\d.]+)\s+ms", output)
        latency_p99_match  = re.search(r"GPU Compute Time:.*percentile\(99%\)\s+=\s+([\d.]+)\s+ms", output)

        if throughput_match:
            metrics["Throughput (QPS)"] = float(throughput_match.group(1))
        if latency_mean_match:
            metrics["Mean Latency (ms)"] = float(latency_mean_match.group(1))
        if latency_p50_match:
            metrics["P50 Latency (ms)"] = float(latency_p50_match.group(1))
        if latency_p99_match:
            metrics["P99 Latency (ms)"] = float(latency_p99_match.group(1))

        return metrics

def print_summary_table(results: list[Dict[str, Any]]) -> None:
    """Renders formatted comparison matrix."""
    df = pd.DataFrame(results)
    
    if "Throughput (QPS)" in df.columns and len(df) > 1:
        base_qps = df.loc[df["Precision"] == "FP32", "Throughput (QPS)"].values
        if len(base_qps) > 0 and base_qps[0] > 0:
            df["Speedup vs FP32"] = (df["Throughput (QPS)"] / base_qps[0]).map("{:.2f}x".format)

    print("\n" + "=" * 80)
    print(" DUAL-MODE MODEL TENSORRT PERFORMANCE MATRIX ")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TensorRT Dual-Mode Engine")
    parser.add_argument("--onnx", type=str, default="models/dual_mode_vision.onnx", help="Path to ONNX model")
    parser.add_argument("--cache", type=str, default="dual_mode_int8.cache", help="INT8 Calibration cache")
    parser.add_argument("--duration", type=int, default=10, help="Benchmark run duration per precision (seconds)")
    args = parser.parse_args()

    runner = TensorRTBenchmarkRunner(onnx_path=args.onnx, calib_cache=args.cache, duration_sec=args.duration)
    
    precisions = ["FP32", "FP16", "INT8"]
    benchmark_data = []

    for precision in precisions:
        metrics = runner.run_benchmark(precision)
        if metrics:
            benchmark_data.append(metrics)

    if benchmark_data:
        print_summary_table(benchmark_data)