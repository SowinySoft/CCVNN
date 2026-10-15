#!/bin/bash
set -e

RELEASE_DIR="deploy/ccvnn_edge_release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR/bin" "$RELEASE_DIR/models" "$RELEASE_DIR/config"

# Copy compiled binaries & models
[ -f build/cpp_glass_to_glass_profiler ] && cp build/cpp_glass_to_glass_profiler "$RELEASE_DIR/bin/"
[ -f build/v4l2_camera_profiler ] && cp build/v4l2_camera_profiler "$RELEASE_DIR/bin/"
[ -f build/ccvnn_engine ] && cp build/ccvnn_engine "$RELEASE_DIR/bin/"

# Copy V14 TorchScript model
if [ -f "model_repository/ccvnn_v14_model_b/1/model.pt" ]; then
    cp model_repository/ccvnn_v14_model_b/1/model.pt "$RELEASE_DIR/models/ccvnn_v14.pt"
elif [ -f "models/ccvnn_traced.pt" ]; then
    cp models/ccvnn_traced.pt "$RELEASE_DIR/models/ccvnn_v14.pt"
fi

# Generate systemd unit file
cat << 'EOF' > "$RELEASE_DIR/config/ccvnn.service"
[Unit]
Description=CCVNN V14 Sub-Millisecond Edge Inspection Engine
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ccvnn
ExecStart=/opt/ccvnn/bin/ccvnn_engine /opt/ccvnn/models/ccvnn_v14.pt
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
mkdir -p deploy
tar -czvf deploy/ccvnn-edge-v1.0.0.tar.gz -C deploy ccvnn_edge_release

echo "[✓] Created V14 deployment package: deploy/ccvnn-edge-v1.0.0.tar.gz"