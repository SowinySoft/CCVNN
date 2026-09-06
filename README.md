# Coordinate Vector Neural Network (CCVNN)

Sub-millisecond edge inspection engine for industrial AOI & robotics via NVIDIA DeepStream & Triton.

## Key Performance Metrics
- **Mean Latency**: 0.013 ms (ORT) / 0.555 ms (Jetson Orin Engine)
- **Throughput**: 1,800+ FPS (Jetson Orin) / 75,000+ FPS (IPC Benchmark)
- **Engine Footprint**: 0.32 MB TensorRT / ONNX engine

## Repository Architecture
- `deepstream/`: NVIDIA DeepStream zero-copy CUDA pipeline & configuration.
- `model_repository/`: Triton Inference Server repository hierarchy (`config.pbtxt` & ONNX model).
- `triton_grpc_client.py`: High-speed gRPC IPC benchmark client.
- `index.html`: Technical whitepaper landing page for GitHub Pages.

## Licensing
- **AGPLv3**: Open-source for non-commercial community and research use.
- **Commercial**: Closed-source OEM embedding and enterprise deployment.
