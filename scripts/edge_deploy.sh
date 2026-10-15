#!/usr/bin/env bash
set -e

echo "===================================================="
echo " CCVNN V14 Edge Deployment & Hardware Bootstrapper "
echo "===================================================="

REPO_URL="https://github.com/SowinySoft/CCVNN.git"

if [ ! -d ".git" ]; then
    echo "[+] Cloning latest CCVNN production release..."
    git clone "$REPO_URL" .
else
    echo "[+] Updating local repository to latest main..."
    git pull origin main
fi

echo "[+] Verifying V14 core artifacts..."
MODEL_PATH="model_repository/ccvnn_v14_model_b/1/model.pt"
CONFIG_PATH="model_repository/ccvnn_v14_model_b/config.pbtxt"

if [ -f "$MODEL_PATH" ] && [ -f "$CONFIG_PATH" ]; then
    echo "[✓] V14 Model binary and Triton config present."
else
    echo "[!] Artifact check failed: Missing $MODEL_PATH or $CONFIG_PATH" && exit 1
fi

echo ""
echo "CCVNN V14 Deployment Ready!"
echo "----------------------------------------------------"
echo "Option A (Triton Container): docker compose up -d"
echo "Option B (Native C++ Edge):  g++ -O3 scripts/cpp_inspection_engine.cpp -I <libtorch_path>/include -L <libtorch_path>/lib -ltorch -ltorch_cpu -lc10 -std=c++17 -o ccvnn_cpp"
echo "===================================================="