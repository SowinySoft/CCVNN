#!/usr/bin/env bash
set -e

echo "===================================================="
echo " CCVNN Edge Deployment & Hardware Bootstrapper "
echo "===================================================="

REPO_URL="https://github.com/SowinySoft/CCVNN.git"

if [ ! -d ".git" ]; then
    echo "[+] Cloning latest CCVNN production release..."
    git clone $REPO_URL .
else
    echo "[+] Updating local repository to latest main..."
    git pull origin main
fi

echo "[+] Verifying core artifacts..."
if [ -f "model_repository/ccvnn_inspector/1/model.pt" ] && [ -f "model_repository/ccvnn_inspector/config.pbtxt" ]; then
    echo "[✓] Model binary and Triton config present."
else
    echo "[!] Artifact check failed!" && exit 1
fi

echo ""
echo "Deployment Ready!"
echo "----------------------------------------------------"
echo "Option A (Triton Container): docker compose up -d"
echo "Option B (Native C++ Edge):  g++ -O3 scripts/cpp_inspection_engine.cpp -I <libtorch_path>/include -L <libtorch_path>/lib -ltorch -ltorch_cpu -lc10 -std=c++17 -o ccvnn_cpp"
echo "===================================================="
