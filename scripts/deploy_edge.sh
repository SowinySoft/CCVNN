#!/usr/bin/env bash
set -euo pipefail

TARGET_IP="${1:?Usage: $0 <edge_node_ip> [user]}"
TARGET_USER="${2:-root}"
DEST_DIR="/opt/ccvnn"

echo "[+] Preparing build directory..."
mkdir -p build

echo "[+] Building V14 release engine binary..."
LIBTORCH_PATH="${LIBTORCH_PATH:-/usr/local/libtorch}"
if [ -d "${LIBTORCH_PATH}" ]; then
    g++ -O3 -std=c++17 scripts/cpp_inspection_engine.cpp \
        -I "${LIBTORCH_PATH}/include" -I "${LIBTORCH_PATH}/include/torch/csrc/api/include" \
        -L "${LIBTORCH_PATH}/lib" -ltorch -ltorch_cpu -lc10 \
        -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -pthread \
        -o build/ccvnn_engine
else
    g++ -O3 -std=c++17 scripts/cpp_inspection_engine.cpp \
        -lopencv_core -lopencv_imgproc -pthread \
        -o build/ccvnn_engine
fi

echo "[+] Provisioning directories on target (${TARGET_IP})..."
ssh "${TARGET_USER}@${TARGET_IP}" "mkdir -p ${DEST_DIR}/{bin,config,models,logs}"

echo "[+] Syncing binary and configurations..."
rsync -avz build/ccvnn_engine "${TARGET_USER}@${TARGET_IP}:${DEST_DIR}/bin/"
if [ -f "deploy/ccvnn.service" ]; then
    rsync -avz deploy/ccvnn.service "${TARGET_USER}@${TARGET_IP}:/etc/systemd/system/"
fi

echo "[+] Enabling and starting CCVNN V14 daemon..."
ssh "${TARGET_USER}@${TARGET_IP}" "
    systemctl daemon-reload && \
    systemctl enable ccvnn.service && \
    systemctl restart ccvnn.service && \
    systemctl status ccvnn.service --no-pager
"

echo "[✓] Deployment to ${TARGET_IP} complete."