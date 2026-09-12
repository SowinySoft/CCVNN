#!/usr/bin/env bash
set -euo pipefail

TARGET_IP="${1:?Usage: $0 <edge_node_ip> [user]}"
TARGET_USER="${2:-root}"
DEST_DIR="/opt/ccvnn"

echo "[+] Building release binary..."
g++ -O3 -std=c++17 scripts/cpp_inspection_engine.cpp -o build/ccvnn_engine -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -pthread

echo "[+] Provisioning directories on target (${TARGET_IP})..."
ssh "${TARGET_USER}@${TARGET_IP}" "mkdir -p ${DEST_DIR}/{bin,config,logs}"

echo "[+] Syncing binary and configurations..."
rsync -avz build/ccvnn_engine "${TARGET_USER}@${TARGET_IP}:${DEST_DIR}/bin/"
rsync -avz deploy/ccvnn.service "${TARGET_USER}@${TARGET_IP}:/etc/systemd/system/"

echo "[+] Enabling and starting CCVNN daemon..."
ssh "${TARGET_USER}@${TARGET_IP}" "
    systemctl daemon-reload &&     systemctl enable ccvnn.service &&     systemctl restart ccvnn.service &&     systemctl status ccvnn.service --no-pager
"

echo "[✓] Deployment to ${TARGET_IP} complete."
