# CCVNN — Continuous Computer Vision Neural Network

[![CCVNN V17 CI/CD Pipeline](https://github.com/SowinySoft/CCVNN/actions/workflows/v17_ci.yml/badge.svg)](https://github.com/SowinySoft/CCVNN/actions)
[![Release](https://img.shields.io/badge/release-v17.0.0-blue.svg)](https://github.com/SowinySoft/CCVNN/releases/tag/v17.0.0)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CCVNN (Continuous Computer Vision Neural Network) is an edge-optimized, high-frequency industrial vision architecture designed for low-latency state evaluation, real-time hazard detection, and automated hardware actuation.

---

## 🚀 What is New in V17.0.0

### ⚡ Zero-Copy In-Place Memory Optimization
* **Flat Memory Footprint:** Replaced dynamic NumPy array creation with contiguous, pre-allocated zero-copy memory buffers (`np.empty` + `np.copyto`), locking baseline RAM usage at a constant ~65 MiB.
* **Sub-Millisecond Latency:** Bypassed CPython allocation marshalling and garbage collection pauses during high-frequency loop execution.
* **Thread Pinning:** Single-threaded intra-op session tuning eliminates core context switching overheads for INT8 ONNX models.

### 🏭 Industrial Microservice Architecture
* **Triton Inference Server:** Containerized backend serving high-throughput quantized ONNX models.
* **Watcher Governor:** Continuous system integrity, error monitoring, and failover safety tracking.
* **Closed-Loop Actuation:** Direct MQTT messaging and Modbus TCP PLC hardware trigger pipeline.

---

## 🛠 Quick Start

### 1. Running via Docker Container
docker pull ghcr.io/sowinysoft/ccvnn-triton:v17.0.0
docker run -d --net=host --name ccvnn-v17 ghcr.io/sowinysoft/ccvnn-triton:v17.0.0

### 2. Local Setup & Profiling
# Clone and setup environment
git clone [https://github.com/SowinySoft/CCVNN.git](https://github.com/SowinySoft/CCVNN.git)
cd CCVNN
git checkout v17.0.0

# Run automated zero-copy profiling suite
chmod +x scripts/run_v17_profilers.sh
./scripts/run_v17_profilers.sh

---

## 📊 Verification & CI/CD Status
All 6 automated E2E pipelines pass on the main branch:
- ✅ ONNX INT8 Model Integrity & Static Validation
- ✅ Production Docker Container Build & Registry Push
- ✅ End-to-End Microservice Pipeline Integration Tests
- ✅ Triton Model Serving & Execution Benchmarks
- ✅ Edge Deployment Manifest & Linter Compliance

---

## 📄 License & Maintainers
Developed and maintained by **Saeed Sowiny** ([SowinySoft](mailto:sowinysoft@gmail.com)).  
Licensed under the [MIT License](LICENSE).
