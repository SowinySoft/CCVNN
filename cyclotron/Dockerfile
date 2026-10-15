# ==============================================================================
# Stage 1: C++ Toolchain & Build Environment
# ==============================================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install compiler toolchain and development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopencv-dev \
    libmodbus-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy build configuration and source files
COPY CMakeLists.txt .
COPY v17_feature_extractor.hpp v17_feature_extractor.cpp ./
COPY watcher_edge_daemon.hpp watcher_edge_daemon.cpp ./
COPY main.cpp ./

# Build optimized Release binary target
RUN mkdir build && cd build && \
    cmake -GNinja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-O3 -march=x86-64-v3 -flto -ffast-math" .. && \
    ninja watcher_edge_daemon

# ==============================================================================
# Stage 2: Minimal Production Runtime
# ==============================================================================
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime shared objects only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libmodbus5 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hardened security: Dedicated non-root service account
RUN useradd -m -u 10001 -s /bin/bash ccvnn

WORKDIR /app
COPY --from=builder /workspace/build/watcher_edge_daemon /app/watcher_edge_daemon

RUN chown -R ccvnn:ccvnn /app

USER ccvnn

# Environment defaults for runtime execution
ENV OMP_NUM_THREADS=4 \
    MODBUS_IP="192.168.1.100" \
    MODBUS_PORT=502 \
    ANOMALY_THRESHOLD=0.75 \
    CONSECUTIVE_REQUIRED=5

ENTRYPOINT ["/app/watcher_edge_daemon"]

# Stage 1: Build Environment
FROM nvcr.io/nvidia/deepstream-l4t:7.0-triton-multiarch AS builder

WORKDIR /build

# Install build toolchain and GStreamer development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy plugin source code and model directory
COPY gstdsdualmode.cpp dual_mode_inference.hpp deepstream_pipeline.cpp ./
COPY models/ ./models/

# Compile shared C++ GStreamer plugin and pipeline runtime
RUN g++ -shared -fPIC -O3 \
    gstdsdualmode.cpp \
    -o libnvds_dualmode.so \
    $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-base-1.0) \
    -I/opt/nvidia/deepstream/deepstream/sources/includes \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart -lnvinfer && \
    g++ -O3 deepstream_pipeline.cpp \
    -o deepstream_dualmode_app \
    $(pkg-config --cflags --libs gstreamer-1.0)

# Stage 2: Runtime Environment
FROM nvcr.io/nvidia/deepstream-l4t:7.0-samples-multiarch AS runtime

WORKDIR /app

# Install runtime plugin dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-v4l2 \
    && rm -rf /var/lib/apt/lists/*

# Install custom plugin into DeepStream plugin directory
COPY --from=builder /build/libnvds_dualmode.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
COPY --from=builder /build/deepstream_dualmode_app /app/deepstream_dualmode_app
COPY --from=builder /build/models /app/models

# Configure plugin search path and library paths
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH

CMD ["./deepstream_dualmode_app"]