#!/usr/bin/env bash
set -e

MODE=$(echo "${1:-cpu}" | tr '[:upper:]' '[:lower:]')
MODEL_DIR="model_repository/ccvnn_v14_model_b"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Directory '$MODEL_DIR' does not exist."
  exit 1
fi

case "$MODE" in
  cpu)
    if [ -f "$MODEL_DIR/config_cpu.pbtxt" ]; then
      cp "$MODEL_DIR/config_cpu.pbtxt" "$MODEL_DIR/config.pbtxt"
      echo "[+] Successfully activated CPU config for ccvnn_v14_model_b"
    else
      echo "Error: $MODEL_DIR/config_cpu.pbtxt not found."
      exit 1
    fi
    ;;
  gpu)
    if [ -f "$MODEL_DIR/config_gpu.pbtxt" ]; then
      cp "$MODEL_DIR/config_gpu.pbtxt" "$MODEL_DIR/config.pbtxt"
      echo "[+] Successfully activated GPU config for ccvnn_v14_model_b"
    else
      echo "Error: $MODEL_DIR/config_gpu.pbtxt not found."
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [cpu|gpu]"
    exit 1
    ;;
esac

# Update timestamp to trigger Triton poll auto-reload
touch "$MODEL_DIR/config.pbtxt"