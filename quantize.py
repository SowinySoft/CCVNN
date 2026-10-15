import os
import torch
import torch.nn as nn
import time

# ---------------------------------------------------------
# 1. Thread Optimization for CPU
# ---------------------------------------------------------
num_cores = os.cpu_count() or 1
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
print(f"[INFO] Running on {num_cores} CPU threads")

# ---------------------------------------------------------
# 2. Import Your Custom Network Architecture
# Replace 'my_model_file' with your Python file (e.g., model.py)
# and 'MyCustomModel' with your actual class name.
# ---------------------------------------------------------
# Example: from model import MyCustomModel
# If defined in another directory:
# import sys; sys.path.append("./path_to_model_folder")

# Placeholder class structure -- REPLACE with your actual model class or import
class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm(x)
        return self.fc2(x)

# ---------------------------------------------------------
# 3. Instantiate & Load Trained FP32 Weights
# ---------------------------------------------------------
weights_path = "model_weights.pth"  # Change to your .pt / .pth path

model_fp32 = MyCustomModel()

if os.path.exists(weights_path):
    print(f"[INFO] Loading weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # Handle state_dict if wrapped in a dict key (e.g., checkpoint['state_dict'])
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_fp32.load_state_dict(checkpoint['state_dict'])
    else:
        model_fp32.load_state_dict(checkpoint)
else:
    print(f"[WARNING] {weights_path} not found. Running benchmark with un-trained initialized weights.")

model_fp32.eval()

# ---------------------------------------------------------
# 4. Apply Dynamic INT8 Quantization
# Targets Linear and Recurrent (LSTM, GRU, RNN) layers
# ---------------------------------------------------------
print("[INFO] Applying Dynamic INT8 Quantization...")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear, nn.LSTM, nn.GRU},  # Layer types to quantize
    dtype=torch.qint8
)

# ---------------------------------------------------------
# 5. Benchmark FP32 vs INT8 Inference
# Adjust dummy_input dimensions to match your model's expected input shape
# ---------------------------------------------------------
dummy_input = torch.randn(1, 10, 1024)  # (Batch, Sequence, Features)

def benchmark(model, name):
    # Warmup runs
    with torch.inference_mode():
        for _ in range(5):
            _ = model(dummy_input)
            
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(50):
            _ = model(dummy_input)
    end = time.perf_counter()
    avg_time = ((end - start) / 50) * 1000
    print(f"{name} Avg Latency: {avg_time:.2f} ms per inference")

benchmark(model_fp32, "FP32 Model")
benchmark(model_int8, "INT8 Model")

# ---------------------------------------------------------
# 6. Save Quantized Weights
# ---------------------------------------------------------
output_path = "model_int8.pt"
torch.save(model_int8.state_dict(), output_path)
print(f"[SUCCESS] Quantized INT8 weights saved to {output_path}")
