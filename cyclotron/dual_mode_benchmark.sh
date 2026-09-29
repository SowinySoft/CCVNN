# Make script executable
chmod +x benchmark_dual_mode.py

# Run benchmark across FP32, FP16, and INT8 modes
python3 benchmark_dual_mode.py \
  --onnx models/dual_mode_vision.onnx \
  --cache dual_mode_int8.cache \
  --duration 15