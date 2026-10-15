#!/usr/bin/env bash

# Launch Triton Server with CPU-optimized telemetry on port 8002
tritonserver \
  --model-repository=/mnt/c/DDVNN-HOME/CCVNN/model \
  --allow-metrics=true \
  --metrics-port=8002 \
  --allow-cpu-metrics=true \
  --allow-gpu-metrics=false \
  --metrics-config summary_latencies=true \
  --metrics-config summary_quantiles=0.5:0.05,0.95:0.001,0.99:0.001 \
  --metrics-interval-ms=2000