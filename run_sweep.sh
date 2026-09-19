#!/usr/bin/env bash
set -e

#MODEL_NAME="ccvnn_v14_model_a"
MODEL_NAME="ccvnn_v14_model_b"
# In your run_sweep.sh or config file
#MODEL_NAME="ccvnn_v14_model_c"
TRITON_URL="localhost:8001"
SDK_IMAGE="nvcr.io/nvidia/tritonserver:26.08-py3-sdk"

for batch in 1 16 32 64; do
  echo "=== Benchmarking Batch Size: ${batch} (CPU Mode) ==="
  docker run --rm --net=host -v "$(pwd):/workspace" -w /workspace "${SDK_IMAGE}" \
    perf_analyzer \
    -m "${MODEL_NAME}" \
    -i grpc \
    -u "${TRITON_URL}" \
    -b "${batch}" \
    --concurrency-range 1:8:1 \
    --measurement-interval 20000 \
    --stability-percentage 10 \
    --async \
    --shape input_vector:14 \
    -f "benchmark_batch_${batch}.csv"
done

echo "Sweep complete. Log files generated in benchmark_batch_*.csv"
