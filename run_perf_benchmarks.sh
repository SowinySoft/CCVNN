#!/usr/bin/env bash

TRITON_URL="localhost:8001"
MODELS_STR="ccvnn ccvnn_v20_backbone"
BATCH_SIZES_STR="1 4 8 16 32"
CONCURRENCY_RANGE="1:8:2"

echo "=========================================================="
echo " Starting Optimized Solution A Benchmarks [ B, 4, 64, 64 ]"
echo "=========================================================="

docker run --rm --net=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -e TRITON_URL="${TRITON_URL}" \
  -e CONCURRENCY_RANGE="${CONCURRENCY_RANGE}" \
  -e MODELS_STR="${MODELS_STR}" \
  -e BATCH_SIZES_STR="${BATCH_SIZES_STR}" \
  nvcr.io/nvidia/tritonserver:26.08-py3-sdk \
  bash -c '
read -r -a MODELS <<< "$MODELS_STR"
read -r -a BATCH_SIZES <<< "$BATCH_SIZES_STR"

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo ">>> Benchmarking Model: ${MODEL}"
  echo "----------------------------------------------------------"

  for B in "${BATCH_SIZES[@]}"; do
    OUTPUT_CSV="perf_${MODEL}_batch_${B}.csv"
    
    echo "  [+] Testing Request Batch Size: ${B} (Shape: ${B},4,64,64)..."
    
    perf_analyzer \
      -m "${MODEL}" \
      -i gRPC \
      -u "${TRITON_URL}" \
      -a \
      -v \
      --shape "input_0:${B},4,64,64" \
      --concurrency-range "${CONCURRENCY_RANGE}" \
      --measurement-interval 5000 \
      --stability-percentage 20 \
      --max-trials 3 \
      -f "${OUTPUT_CSV}"

    if [ $? -eq 0 ]; then
      echo "      [SUCCESS] Report saved to ${OUTPUT_CSV}"
    else
      echo "      [ERROR] perf_analyzer failed for ${MODEL} with batch size ${B}."
    fi
  done
done
'

echo ""
echo "=========================================================="
echo " Benchmarking Run Completed."
echo "=========================================================="