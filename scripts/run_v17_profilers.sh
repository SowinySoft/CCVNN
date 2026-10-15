#!/usr/bin/env bash
set -e

REPORT_DIR="reports/profiling_v17"
mkdir -p "$REPORT_DIR"

echo "=================================================="
echo "🚀 1. Installing Profiling Dependencies"
echo "=================================================="
pip install --quiet line-profiler memory-profiler psutil matplotlib

echo -e "\n=================================================="
echo "⚙️ 2. Injecting Zero-Copy Buffer Profiling Code"
echo "=================================================="
PROFILED_SCRIPT="scripts/profile_v17_target.py"

cat << 'PYEOF' > "$PROFILED_SCRIPT"
import time
import os
import psutil
import numpy as np
import onnxruntime as ort
from memory_profiler import profile as mem_profile

try:
    @profile
    def run_v17_inference_loop(session, input_name, input_buffer, dummy_source, iterations=5000):
        # High-frequency zero-allocation loop using in-place copy
        for _ in range(iterations):
            np.copyto(input_buffer, dummy_source)
            _ = session.run(None, {input_name: input_buffer})
except NameError:
    def run_v17_inference_loop(session, input_name, input_buffer, dummy_source, iterations=5000):
        for _ in range(iterations):
            np.copyto(input_buffer, dummy_source)
            _ = session.run(None, {input_name: input_buffer})

@mem_profile
def main():
    model_path = "ccvnn_v17_backbone_int8.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found.")
        return

    # Configure session for single-threaded low latency
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Pre-allocate contiguous zero-copy memory buffer
    input_buffer = np.empty((1, 17), dtype=np.float32)
    dummy_source = np.random.randn(1, 17).astype(np.float32)

    # Execute zero-copy profiling loop
    run_v17_inference_loop(session, input_name, input_buffer, dummy_source)

if __name__ == "__main__":
    main()
PYEOF

echo -e "\n=================================================="
echo "⏱️ 3. Running Line Profiler (CPU Breakdown)"
echo "=================================================="
kernprof -l "$PROFILED_SCRIPT"

python3 -m line_profiler profile_v17_target.py.lprof > "$REPORT_DIR/line_profile_summary.txt"
cat "$REPORT_DIR/line_profile_summary.txt"

echo -e "\n=================================================="
echo "🧠 4. Running Memory Profiler & Generating Plot"
echo "=================================================="
mprof run --interval 0.01 --output "$REPORT_DIR/mprofile.dat" python3 "$PROFILED_SCRIPT" > "$REPORT_DIR/memory_profile_text.txt"

# Generate memory visual graph
mprof plot "$REPORT_DIR/mprofile.dat" --output "$REPORT_DIR/v17_memory_profile.png" --title "CCVNN V17 Zero-Copy Inference Memory Profile"

echo -e "\n=================================================="
echo "✅ Profiling Complete!"
echo "=================================================="
echo "📄 Text Summary File : $REPORT_DIR/line_profile_summary.txt"
echo "📄 Memory Text Log  : $REPORT_DIR/memory_profile_text.txt"
echo "📊 Visual Summary   : $REPORT_DIR/v17_memory_profile.png"

# Cleanup temporary files
rm -f "$PROFILED_SCRIPT" profile_v17_target.py.lprof
