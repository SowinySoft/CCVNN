import os
import torch
import torch.nn as nn
from src.ccvnn_model import CCVNNModel  # Ensure src/ccvnn_model.py exists

def export_dual_axis_onnx():
    print("[1/3] Instantiating CCVNN Full Vision Fusion Model and loading weights...")
    model = CCVNNModel()
    
    # Path to your trained v20 PyTorch weights
    weights_path = "weights/ccvnn_v20.pth"  
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Loaded trained weights from {weights_path}")
    else:
        print(f"[WARNING] Weights file '{weights_path}' not found. Exporting uninitialized architecture.")

    model.eval()

    # Dummy tensor matching input channels and initial spatial dimensions [B, C, H, W]
    dummy_input = torch.randn(1, 4, 64, 64, dtype=torch.float32)

    # Output path inside Triton model repository
    os.makedirs("model/ccvnn_v21_dynamic/1", exist_ok=True)
    onnx_output_path = "model/ccvnn_v21_dynamic/1/model.onnx"

    print("[2/3] Exporting to ONNX with dynamic axes: batch_size (-1) & seq_len (-1)...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=17,  # Kept at 17 for Triton/OpenVINO compatibility
        do_constant_folding=True,
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes={
            "input_0": {0: "batch_size", 1: "seq_len"},
            "output_0": {0: "batch_size", 1: "seq_len"}
        }
    )
    
    print(f"[3/3] Dynamic ONNX model successfully saved to {onnx_output_path}")

if __name__ == "__main__":
    export_dual_axis_onnx()