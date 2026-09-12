#!/bin/bash
set -e

RELEASE_DIR="deploy/ccvnn_edge_release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR/bin" "$RELEASE_DIR/models" "$RELEASE_DIR/config"

# Copy compiled binaries & models
cp build/cpp_glass_to_glass_profiler "$RELEASE_DIR/bin/"
[ -f build/v4l2_camera_profiler ] && cp build/v4l2_camera_profiler "$RELEASE_DIR/bin/"
cp models/ccvnn_traced.pt "$RELEASE_DIR/models/"

# Generate systemd unit file
cat << 'EOF' > "$RELEASE_DIR/config/ccvnn.service"
[Unit]
Description=CCVNN Sub-Millisecond Edge Inspection Engine
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ccvnn
ExecStart=/opt/ccvnn/bin/cpp_glass_to_glass_profiler /opt/ccvnn/models/ccvnn_traced.pt 100
Restart=always
RestartSec=1s

# Real-time process scheduling & core isolation
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=99
CPUAffinity=2 3
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

# Create release tarball
tar -czvf deploy/ccvnn-edge-v1.0.0.tar.gz -C deploy ccvnn_edge_release

echo "[✓] Created deployment package: deploy/ccvnn-edge-v1.0.0.tar.gz"
