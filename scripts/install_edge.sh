#!/bin/bash
set -e

INSTALL_DIR="/opt/ccvnn"
TARBALL="deploy/ccvnn-edge-v1.0.0.tar.gz"

if [ ! -f "$TARBALL" ]; then
  echo "[!] Deployment package '$TARBALL' not found!"
  exit 1
fi

echo "[+] Provisioning CCVNN Edge Node..."

# 1. Extract release files to /opt/ccvnn
mkdir -p "$INSTALL_DIR"
tar -xzf "$TARBALL" -C /tmp
cp -r /tmp/ccvnn_edge_release/* "$INSTALL_DIR/"
rm -rf /tmp/ccvnn_edge_release

# 2. Grant execution permissions
chmod +x "$INSTALL_DIR/bin/"*

# 3. Configure Real-Time SCHED_FIFO & Memory Locking Limits
if [ -w /etc/security/limits.conf ] && ! grep -q "# CCVNN RT Limits" /etc/security/limits.conf; then
    cat << 'EOF' >> /etc/security/limits.conf

# CCVNN RT Limits
* soft memlock unlimited
* hard memlock unlimited
* soft rtprio 99
* hard rtprio 99
EOF
    echo "[✓] Configured Unlimited Memory Locking & RT Scheduling Limits"
fi

# 4. Set CPU Governor to Performance (if sysfs governor path exists)
if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
    echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
    echo "[✓] CPU Core Frequencies Locked to Max Performance"
fi

# 5. Conditionally Register systemd Service (Physical Host Only)
if pidof systemd >/dev/null 2>&1 || [ -d /run/systemd/system ]; then
    cp "$INSTALL_DIR/config/ccvnn.service" /etc/systemd/system/ccvnn.service
    systemctl daemon-reload
    systemctl enable ccvnn.service
    echo "[✓] Registered systemd service: ccvnn.service"
else
    echo "[!] Container environment detected (no PID 1 systemd). Skipped service registration."
fi

echo "========================================================"
echo "[✓] CCVNN Engine Installed to $INSTALL_DIR"
echo "========================================================"
