#!/bin/bash

echo "=========================================================="
echo " Building and Launching CCVNN Metropolis Triton Service   "
echo "=========================================================="

# 1. Build Docker Image
echo "[1/2] Building Triton Microservice Image..."
docker build -t ccvnn-triton-microservice:latest -f Dockerfile.triton .

# 2. Run Container with NVIDIA GPU Runtime
echo "[2/2] Launching Container with Shared CUDA Memory..."
docker run --gpus all --runtime nvidia -d \
  --name ccvnn_triton_container \
  --restart always \
  --shm-size=2g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  ccvnn-triton-microservice:latest

echo " Service started on ports: 8000 (HTTP), 8001 (gRPC), 8002 (Metrics)"
