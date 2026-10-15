#!/usr/bin/env bash

TRITON_URL="localhost:8001"
MODELS_STR="ccvnn"
CONCURRENCY_RANGE="1:8:2"  # Tests Concurrency 1, 3, 5, 7

echo "=========================================================="
echo " Starting Dynamic Batching Benchmarks: ccvnn"
echo "=========================================================="

docker run --rm --net=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -e TRITON_URL="${TRITON_URL}" \
  -e CONCURRENCY_RANGE="${CONCURRENCY_RANGE}" \
  -e MODELS_STR="${MODELS_STR}" \
  nvcr.io/nvidia/tritonserver:26.08-py3-sdk \
  bash -c '
read -r -a MODELS <<< "$MODELS_STR"

for MODEL in "${MODELS[@]}"; do
  OUTPUT_CSV="perf_${MODEL}_dynamic_batching.csv"
  
  echo ""
  echo ">>> Benchmarking Model: ${MODEL} across Concurrency Range [ ${CONCURRENCY_RANGE} ]"
  echo "----------------------------------------------------------"
  
  perf_analyzer \
    -m "${MODEL}" \
    -i grpc \
    -u "${TRITON_URL}" \
    --async \
    -v \
    -b 1 \
    --shape input_0:1,4,64,64 \
    --concurrency-range "${CONCURRENCY_RANGE}" \
    --measurement-interval 12000 \
    --stability-percentage 50 \
    --max-trials 10 \
    -f "${OUTPUT_CSV}"

  if [ $? -eq 0 ]; then
    echo "      [SUCCESS] Report saved to ${OUTPUT_CSV}"
  else
    echo "      [ERROR] perf_analyzer failed for ${MODEL}."
  fi
done
'

echo ""
echo "=========================================================="
echo " Benchmarking Run Completed."
echo "=========================================================="