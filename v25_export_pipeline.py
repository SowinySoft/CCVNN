# ==============================================================================
# CCVNN V25 PRODUCTION-GRADE DUAL-MODE GEOMETRIC PIPELINE
# Framework: First-Order Jacobian & Second-Order Hessian Curvature Engine
# Dynamic Shapes: Fully Dynamic Batch [-1] & Spatial Resolution [-1, -1]
# Artifacts: PyTorch ONNX (.onnx) & OpenVINO IR (.xml / .bin)
# ==============================================================================

import os
import shutil
import warnings
import logging
from typing import Tuple

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov


# ==============================================================================
# 1. HARDWARE CONFIGURATION & DEVICE SWITCH
# ==============================================================================

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Hardware Accelerator Initialized: {device.type.upper()}")
    return device


# ==============================================================================
# 2. CCVNN V25 GEOMETRIC FEATURE EXTRACTION MODULE (E18-E25)
# ==============================================================================

class V25GeometricFeatureExtractor(nn.Module):
    """
    Transforms [Batch, 4, H, W] RGBA Tensors into [Batch, 25, H, W] Feature Space:
    - Base Stencil & Statistics (P0-P4, C_local, opacity, mu, sigma)
    - First-Order Spatial Gradients & Jacobian (Gx, Gy, grad_mag, Laplacian)
    - Complex Phase Components (Zr, Zi)
    - Color Metrics (chroma, saturation) & E18-E22 Expansion Nodes
    - Second-Order Hessian Curvature Matrix (hxx, hyy, hxy)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x_rgba: torch.Tensor) -> torch.Tensor:
        r, g, b = x_rgba[:, 0:1, :, :], x_rgba[:, 1:2, :, :], x_rgba[:, 2:3, :, :]
        opacity = x_rgba[:, 3:4, :, :]
        
        # Luminance
        I = 0.299 * r + 0.587 * g + 0.114 * b
        
        # 5-Point Cross Stencil Pads (Preserves H x W)
        P0 = I
        P1 = F.pad(I[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')
        P2 = F.pad(I[:, :, 1:, :],  (0, 0, 0, 1), mode='replicate')
        P3 = F.pad(I[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')
        P4 = F.pad(I[:, :, :, 1:],  (0, 1, 0, 0), mode='replicate')

        stencil = torch.cat([P0, P1, P2, P3, P4], dim=1)
        p_max, _ = torch.max(stencil, dim=1, keepdim=True)
        p_min, _ = torch.min(stencil, dim=1, keepdim=True)
        
        C_local = (p_max - p_min) / (p_max + p_min + self.eps)
        mu_phi = torch.mean(stencil, dim=1, keepdim=True)
        sigma_phi = torch.var(stencil, dim=1, keepdim=True, unbiased=False)
        
        # --- First-Order Jacobian & Spatial Gradients ---
        Gx = (P4 - P3) / 2.0
        Gy = (P2 - P1) / 2.0
        laplacian = (P1 + P2 + P3 + P4) - 4.0 * P0
        grad_mag = torch.sqrt(Gx**2 + Gy**2 + self.eps)
        
        # Phase components
        phase = torch.atan2(Gy, Gx)
        Zr = P0 * torch.cos(phase)
        Zi = P0 * torch.sin(phase)
        
        # Color metrics
        sum_rgb = r + g + b + self.eps
        min_rgb = torch.min(torch.cat([r, g, b], dim=1), dim=1, keepdim=True)[0]
        saturation = 1.0 - (3.0 * min_rgb / sum_rgb)
        
        # --- E18 - E22 Expansion Nodes ---
        E18 = torch.clamp((r - g) / (r + g + self.eps) * saturation, min=0.0)
        E19 = 1.0 / (1.0 + grad_mag)
        E20 = sigma_phi / (torch.abs(mu_phi) + self.eps)
        E21 = (torch.abs(Gx) + self.eps) / (torch.abs(Gy) + self.eps)
        E22 = grad_mag / (P0 + 1.0)

        # --- Second-Order Hessian Curvature Components (hxx, hyy, hxy) ---
        hxx = F.pad(Gx[:, :, :, 1:] - Gx[:, :, :, :-1], (0, 1, 0, 0), mode='replicate')
        hyy = F.pad(Gy[:, :, 1:, :] - Gy[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')
        hxy = F.pad(Gx[:, :, 1:, :] - Gx[:, :, :-1, :], (0, 0, 0, 1), mode='replicate')

        return torch.cat([
            P0, P1, P2, P3, P4,             # Channels 0-4
            C_local, opacity, mu_phi,       # Channels 5-7
            sigma_phi, Gx, Gy, laplacian,   # Channels 8-11
            grad_mag, Zr, Zi,               # Channels 12-14
            (r - g) / (b + self.eps), saturation, # Channels 15-16
            E18, E19, E20, E21, E22,        # Channels 17-21
            hxx, hyy, hxy                   # Channels 22-24 (Total 25 Channels)
        ], dim=1)


class MetaACON(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.fc1 = nn.Conv2d(channels, max(channels // 4, 1), 1)
        self.fc2 = nn.Conv2d(max(channels // 4, 1), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.fc2(F.hardswish(self.fc1(torch.mean(x, dim=(2, 3), keepdim=True)))))
        p1_p2 = self.p1 - self.p2
        return p1_p2 * x * torch.sigmoid(beta * p1_p2 * x) + self.p2 * x


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_r = self.conv_r(x_r) - self.conv_i(x_i)
        y_i = self.conv_r(x_i) + self.conv_i(x_r)
        return y_r, y_i


# ==============================================================================
# 3. CCVNN V25 DUAL-MODE PIPELINE WRAPPERS
# ==============================================================================

class CCVNNV25InferencePipeline(nn.Module):
    """Mode 1: Inference Backbone leveraging full 25-channel V25 geometric features."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.extractor = V25GeometricFeatureExtractor()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(25, hidden_dim)
        self.act1 = nn.Hardswish()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.Hardswish()
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_rgba: torch.Tensor) -> torch.Tensor:
        spatial = self.extractor(x_rgba)
        vec = self.pool(spatial).flatten(1)
        x = self.act1(self.fc1(vec))
        x = self.act2(self.fc2(x))
        return self.sigmoid(self.fc3(x))


class CCVNN2DVisionPipeline(nn.Module):
    """Mode 2: Vision Engine using complex phase features derived from V25 extractor."""
    def __init__(self, out_channels: int = 32, num_classes: int = 10):
        super().__init__()
        self.extractor = V25GeometricFeatureExtractor()
        self.complex_conv = ComplexConv2d(in_channels=1, out_channels=out_channels)
        self.acon_r = MetaACON(out_channels)
        self.acon_i = MetaACON(out_channels)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(out_channels * 2 * 8 * 8, num_classes)

    def forward(self, x_rgba: torch.Tensor) -> torch.Tensor:
        spatial = self.extractor(x_rgba)
        x_r = spatial[:, 13:14, :, :]
        x_i = spatial[:, 14:15, :, :]

        y_r, y_i = self.complex_conv(x_r, x_i)
        y_r = self.acon_r(y_r)
        y_i = self.acon_i(y_i)

        y_r = self.pool(y_r)
        y_i = self.pool(y_i)
        concat_feats = torch.cat([y_r, y_i], dim=1)
        return self.fc(torch.flatten(concat_feats, start_dim=1))


# ==============================================================================
# 4. TRITON REPOSITORY & MULTI-BACKEND CONFIGURATION
# ==============================================================================

def write_triton_repository() -> None:
    if os.path.exists("model_repository"):
        shutil.rmtree("model_repository")
        print("🧹 Cleaned legacy model_repository.")

    backbone_config = """name: "ccvnn_v25_backbone"
backend: "openvino"
max_batch_size: 0

input [
  {
    name: "input_rgba"
    data_type: TYPE_FP32
    dims: [ -1, 4, -1, -1 ]
  }
]

output [
  {
    name: "hazard_prob"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 2000
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]
"""

    vision_config = """name: "ccvnn"
backend: "openvino"
max_batch_size: 0

input [
  {
    name: "input_rgba"
    data_type: TYPE_FP32
    dims: [ -1, 4, -1, -1 ]
  }
]

output [
  {
    name: "output_logits"
    data_type: TYPE_FP32
    dims: [ -1, 10 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]
"""
    os.makedirs("model_repository/ccvnn_v25_backbone/1", exist_ok=True)
    os.makedirs("model_repository/ccvnn/1", exist_ok=True)
    
    with open("model_repository/ccvnn_v25_backbone/config.pbtxt", "w") as f:
        f.write(backbone_config)
    with open("model_repository/ccvnn/config.pbtxt", "w") as f:
        f.write(vision_config)
    print("✅ Generated Triton config.pbtxt with full dynamic bounds [-1, 4, -1, -1].")


# ==============================================================================
# 5. DUAL ARTIFACT EXPORT & FULL DYNAMIC VERIFICATION
# ==============================================================================

def export_and_verify_v25() -> None:
    device = get_device()
    write_triton_repository()

    dummy_rgba = torch.randn(1, 4, 64, 64, dtype=torch.float32, device=device)
    dynamic_axes_spec = {
        "input_rgba": {0: "batch_size", 2: "height", 3: "width"}
    }

    # --- Mode 1: Inference Backbone Export (ONNX & OpenVINO) ---
    print("\n--- Exporting Mode 1: ccvnn_v25_backbone ---")
    model_inf = CCVNNV25InferencePipeline().to(device).eval()
    
    torch.onnx.export(
        model_inf, dummy_rgba, "model_repository/ccvnn_v25_backbone/1/model.onnx",
        input_names=["input_rgba"], output_names=["hazard_prob"],
        dynamic_axes=dynamic_axes_spec, opset_version=14
    )
    print("✅ Saved PyTorch ONNX: ccvnn_v25_backbone/1/model.onnx")

    ov_inf = ov.convert_model(model_inf.cpu(), example_input=dummy_rgba.cpu())
    ov_inf.reshape({ov_inf.input(0): ov.PartialShape([-1, 4, -1, -1])})
    ov.save_model(ov_inf, "model_repository/ccvnn_v25_backbone/1/model.xml")
    print("✅ Saved OpenVINO IR: ccvnn_v25_backbone/1/model.xml & model.bin")


    # --- Mode 2: Vision Engine Export (ONNX & OpenVINO) ---
    print("\n--- Exporting Mode 2: ccvnn (Vision Engine) ---")
    model_vis = CCVNN2DVisionPipeline().to(device).eval()
    
    torch.onnx.export(
        model_vis, dummy_rgba, "model_repository/ccvnn/1/model.onnx",
        input_names=["input_rgba"], output_names=["output_logits"],
        dynamic_axes=dynamic_axes_spec, opset_version=14
    )
    print("✅ Saved PyTorch ONNX: ccvnn/1/model.onnx")

    ov_vis = ov.convert_model(model_vis.cpu(), example_input=dummy_rgba.cpu())
    ov_vis.reshape({ov_vis.input(0): ov.PartialShape([-1, 4, -1, -1])})
    ov.save_model(ov_vis, "model_repository/ccvnn/1/model.xml")
    print("✅ Saved OpenVINO IR: ccvnn/1/model.xml & model.bin")


    # ==========================================================================
    # 6. RUNTIME VERIFICATION (DYNAMIC BATCH & DYNAMIC RESOLUTION)
    # ==========================================================================
    print("\n==============================================================")
    print(" RUNNING FULLY DYNAMIC OPEN_VINO RUNTIME VALIDATION")
    print("==============================================================")
    core = ov.Core()
    test_cases = [(1, 32, 32), (4, 64, 64), (16, 128, 128), (32, 96, 96)]

    compiled_inf = core.compile_model("model_repository/ccvnn_v25_backbone/1/model.xml", "CPU")
    for b_size, h, w in test_cases:
        test_in = np.random.randn(b_size, 4, h, w).astype(np.float32)
        res = compiled_inf([test_in])[compiled_inf.output(0)]
        assert res.shape == (b_size, 1), f"Backbone shape mismatch: got {res.shape}"
        print(f" [V25 Backbone] Batch: {b_size:2d} | Res: {h}x{w} -> Output: {res.shape}")

    compiled_vis = core.compile_model("model_repository/ccvnn/1/model.xml", "CPU")
    for b_size, h, w in test_cases:
        test_in = np.random.randn(b_size, 4, h, w).astype(np.float32)
        res = compiled_vis([test_in])[compiled_vis.output(0)]
        assert res.shape == (b_size, 10), f"Vision shape mismatch: got {res.shape}"
        print(f" [V25 Vision]   Batch: {b_size:2d} | Res: {h}x{w} -> Output: {res.shape}")

    print("\n✅ V25 GEOMETRIC DUAL-MODE DYNAMIC VERIFICATION PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    export_and_verify_v25()
