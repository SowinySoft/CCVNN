#!/usr/bin/env bash
set -e

MODEL_NAME="cyclotron_ensemble"
TRITON_CONTAINER="tritonserver_prod"
SDK_IMAGE="nvcr.io/nvidia/tritonserver:26.08-py3-sdk"
OUTPUT_CSV="perf_sla_results.csv"

cleanup() {
    echo -e "\n[!] Interrupted. Terminating container..."
    docker rm -f perf_analyzer_runner 2>/devnull || true
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "[1/3] Verifying Triton Container..."
if ! docker ps --format '{{.Names}}' | grep -q "^${TRITON_CONTAINER}$"; then
    echo "ERROR: Triton container '${TRITON_CONTAINER}' is not running."
    exit 1
fi

rm -f ${OUTPUT_CSV}

echo "[2/3] Running SLA Benchmark Sweep..."

docker run --rm \
  --name perf_analyzer_runner \
  -v $(pwd):/workspace \
  --net container:${TRITON_CONTAINER} \
  ${SDK_IMAGE} \
  perf_analyzer \
    -m ${MODEL_NAME} \
    -i gRPC \
    -u localhost:8001 \
    --async \
    --concurrency-range 1:3:1 \
    --shape camera_frame_tensor:1,3,640,640 \
    --shape telemetry_tensor:1,4 \
    --measurement-mode count_windows \
    --measurement-request-count 15 \
    --stability-percentage 999999 \
    --max-trials 10 \
    -f /workspace/${OUTPUT_CSV} || true

echo ""
echo "[3/3] Analyzing SLA Compliance Table..."
python3 parse_sla.py

echo ""
echo "SLA Benchmark execution complete."
