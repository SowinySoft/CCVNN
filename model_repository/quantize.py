import os
import time
import torch
import torch.nn as nn

# ---------------------------------------------------------
# 1. Hardware & Thread Optimization (CPU Environment)
# ---------------------------------------------------------
num_cores = os.cpu_count() or 1
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
print(f"[INFO] CPU Optimization: Configured {num_cores} threads for execution.")

# ---------------------------------------------------------
# 2. Base Model Definitions matching config.pbtxt specifications
# ---------------------------------------------------------
class CCVNNHardswish12D(nn.Module):
    """Model matching ccvnn_hardswish_12d (Input Dim: 12, Output Dim: 1)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 64)
        self.act1 = nn.Hardswish()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.Hardswish()
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        return self.out(self.act2(self.fc2(self.act1(self.fc1(x)))))


class CCVNNReLU69D(nn.Module):
    """Model matching ccvnn_relu6_9d (Input Dim: 9, Output Dim: 1)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.act1 = nn.ReLU6()
        self.fc2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU6()
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        return self.out(self.act2(self.fc2(self.act1(self.fc1(x)))))


# ---------------------------------------------------------
# 3. Dynamic INT8 Quantization & Benchmarking Function
# ---------------------------------------------------------
def process_and_quantize(model, name, weights_path, input_dim):
    print(f"
==================================================")
    print(f" Processing Model: {name}")
    print(f"==================================================")

    # Load Weights if present
    if os.path.exists(weights_path):
        print(f"[INFO] Loading FP32 checkpoint from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"[WARNING] Checkpoint {weights_path} not found. Proceeding with initialized weights.")

    model.eval()

    # Dynamic INT8 Quantization for Linear layers
    print("[INFO] Applying Dynamic INT8 Quantization...")
    model_int8 = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    # Benchmark payload using batch_size 1 matching config.pbtxt dims
    dummy_input = torch.randn(1, input_dim)

    def benchmark(net, label):
        # Warmup
        with torch.inference_mode():
            for _ in range(10):
                _ = net(dummy_input)
        
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(100):
                _ = net(dummy_input)
        end = time.perf_counter()
        avg_ms = ((end - start) / 100) * 1000
        print(f" -> {label} Average Latency: {avg_ms:.3f} ms per inference")

    benchmark(model, "FP32 Model")
    benchmark(model_int8, "INT8 Quantized Model")

    # Export INT8 weights directly into model repository output path
    output_dir = os.path.dirname(weights_path)
    output_file = os.path.join(output_dir, "model_int8.pt")
    torch.save(model_int8.state_dict(), output_file)
    print(f"[SUCCESS] Saved INT8 quantized checkpoint to: {output_file}")


# ---------------------------------------------------------
# 4. Sequential Batch Execution for Repository Models
# ---------------------------------------------------------
if __name__ == "__main__":
    base_repo = "model_repository"

    # 1. Quantize Hardswish 12D Version
    process_and_quantize(
        model=CCVNNHardswish12D(),
        name="ccvnn_hardswish_12d",
        weights_path=os.path.join(base_repo, "ccvnn_hardswish_12d", "1", "model.pt"),
        input_dim=12
    )

    # 2. Quantize ReLU6 9D Version
    process_and_quantize(
        model=CCVNNReLU69D(),
        name="ccvnn_relu6_9d",
        weights_path=os.path.join(base_repo, "ccvnn_relu6_9d", "1", "model.pt"),
        input_dim=9
    )
