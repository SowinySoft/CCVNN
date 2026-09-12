# CCVNN: Computer Vision Neural Network Microservice

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-v1.28%2B-blue.svg)](#)
[![NVIDIA
Triton](https://img.shields.io/badge/Triton-Inference%20Server-76B900.svg)](#)
[![Helm](https://img.shields.io/badge/Helm-v3.0%2B-0F1689.svg)](#)

An enterprise-grade, edge-deployable computer vision microservice
architecture leveraging **NVIDIA Triton Inference Server**, **TensorRT
INT8 acceleration**, **gRPC with Mutual TLS (mTLS)** security, and
**Kubernetes Helm** orchestration for multi-camera video stream
processing.

------------------------------------------------------------------------

## 🏗 System Architecture

                          +------------------------------------------+
                          |            Kubernetes Cluster            |
                          |               (ccvnn-edge)               |
                          |                                          |
    +------------------+  |  +----------------+  +----------------+  |
    | Multi-Camera     |  |  |  gRPC Ingress  |  |    Triton      |  |
    | RTSP Ingestion   |---->|    Service     |->|   Inference    |  |
    | Client (mTLS)    |  |  |   (Port 8001)  |  |     Server     |  |
    +------------------+  |  +----------------+  +-------+--------+  |
                          |                              |           |
    +------------------+  |  +----------------+          |           |
    | Prometheus /     |<----+  Metrics Svc   |<---------+           |
    | Grafana Loki     |  |  |   (Port 8002)  |                      |
    +------------------+  |  +----------------+                      |
                          +------------------------------------------+

The system operates across four dedicated stages:

1.  **Model Optimization:** TensorRT INT8 calibration and custom dynamic
    batching (`ccvnn_hardswish` / `ccvnn_relu6`).
2.  **Ingestion Pipeline:** Non-blocking multi-camera RTSP ingestion
    worker threads connecting over secure gRPC.
3.  **Observability:** Prometheus metric exporters
    (`nv_inference_request_success`) coupled with Grafana Loki
    structured logging.
4.  **Security & Orchestration:** Production Helm chart deployment
    backed by custom PKI mutual TLS (mTLS) certificate validation.

------------------------------------------------------------------------

## 📁 Repository Structure

    CCVNN/
    ├── certs/                        # PKI certificates & key generation scripts (Git ignored)
    │   ├── generate_certs.sh
    │   ├── ca.crt / ca.key
    │   ├── server.crt / server.key
    │   └── client.crt / client.key
    ├── helm/                         # Production Kubernetes Helm chart & rendered manifests
    │   ├── ccvnn-microservice/       # Chart templates & values.yaml
    │   ├── namespace-ccvnn-edge.yaml
    │   ├── secret-ccvnn-tls.yaml
    │   └── rendered-edge-manifests.yaml
    ├── model_repository/             # NVIDIA Triton Model Repository layout
    │   ├── ccvnn_hardswish/
    │   └── ccvnn_relu6/
    ├── camera_config.py              # Heterogeneous RTSP URL builder
    ├── rtsp_multicamera_ingestion.py # Multi-camera CPU ingestion pipeline
    ├── rtsp_nvdec_multicamera.py     # Multi-camera Zero-Copy NVDEC pipeline
    ├── rtsp_mtls_multicamera.py      # Secure mTLS multi-camera ingestion client
    └── README.md

------------------------------------------------------------------------

## ⚙️ Triton Inference Server Configuration

The microservice utilizes Triton's dynamic batching and INT8 TensorRT
engine backends.

### Model Repository Layout

    model_repository/
    ├── ccvnn_hardswish/
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.plan
    └── ccvnn_relu6/
        ├── config.pbtxt
        └── 1/
            └── model.plan

### Sample `config.pbtxt` (`ccvnn_hardswish`)

    name: "ccvnn_hardswish"
    platform: "tensorrt_plan"
    max_batch_size: 16

    input [
      {
        name: "input_0"
        data_type: TYPE_FP32
        dims: [ 3, 224, 224 ]
      }
    ]
    output [
      {
        name: "output_0"
        data_type: TYPE_FP32
        dims: [ 1000 ]
      }
    ]

    dynamic_batching {
      preferred_batch_size: [ 4, 8, 16 ]
      max_queue_delay_microseconds: 5000
    }

    instance_group [
      {
        count: 1
        kind: KIND_GPU
      }
    ]

------------------------------------------------------------------------

## 🔒 mTLS Security Setup

Communication between client workers and the Triton gRPC server is
encrypted and mutually authenticated via custom X.509 PKI certificates.

### 1. Generate PKI Certificates

    chmod +x certs/generate_certs.sh
    ./certs/generate_certs.sh

### 2. Verify Generated Certificate Assets

    ls -la certs/
    # Expected files: ca.crt, server.crt, server.key, client.crt, client.key

> **Security Note:** Private key files (`*.key`, `*.csr`, `*.srl`) are
> ignored by `.gitignore` and must never be committed to source control.

------------------------------------------------------------------------

## 🚀 Kubernetes & Helm Deployment

### Prerequisites

- `kubectl` (v1.28+)
- `helm` (v3.0+)
- Active Kubernetes Edge Cluster (K3s, MicroK8s, or Minikube)

### 1. Create Namespace & Secret

    # Create target namespace
    kubectl create namespace ccvnn-edge

    # Deploy mTLS Kubernetes Secret
    kubectl create secret generic ccvnn-microservice-tls \
      --from-file=ca.crt=certs/ca.crt \
      --from-file=server.crt=certs/server.crt \
      --from-file=server.key=certs/server.key \
      -n ccvnn-edge

### 2. Offline Helm Manifest Verification (Dry-Run)

    # Lint chart syntax
    helm lint helm/ccvnn-microservice/

    # Render dry-run manifests to file
    helm template ccvnn-edge-release helm/ccvnn-microservice/ \
      --namespace ccvnn-edge \
      > helm/rendered-edge-manifests.yaml

### 3. Deploy Helm Release to Edge Cluster

    helm upgrade --install ccvnn-edge-release helm/ccvnn-microservice/ \
      --namespace ccvnn-edge \
      --set replicaCount=1 \
      --set triton.logVerboseLevel=0 \
      --wait --timeout 2m0s

### 4. Verify Pod Readiness & Services

    # Check running pods
    kubectl get pods -n ccvnn-edge -o wide

    # Verify exposed ports (HTTP: 8000, gRPC: 8001, Metrics: 8002)
    kubectl get svc -n ccvnn-edge

    # Test internal metrics endpoint
    POD_NAME=$(kubectl get pods -n ccvnn-edge -l app.kubernetes.io/name=ccvnn-microservice -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n ccvnn-edge $POD_NAME -- curl -s http://localhost:8002/metrics | grep nv_inference_request_success

------------------------------------------------------------------------

## 🧪 End-to-End Ingestion Testing

    # Forward remote gRPC port
    kubectl port-forward -n ccvnn-edge svc/ccvnn-edge-release-ccvnn-microservice 8001:8001 &
    PF_PID=$!

    # Execute mTLS ingestion worker pipeline
    python3 rtsp_mtls_multicamera.py

    # Terminate port forwarding process
    kill $PF_PID

------------------------------------------------------------------------

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
