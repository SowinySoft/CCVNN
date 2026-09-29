# Step 0: Install dependencies
!pip install -q torch onnx onnxruntime openvino openvino-dev numpy

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import openvino as ov

# ==============================================================================
# 1. MODEL ARCHITECTURE DEFINITIONS
# ==============================================================================

# Mode 1: Telemetry & Convergence Backbone (17-vector parameter input)
class CCVNNV17Backbone(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, output_dim=1):
        super(CCVNNV17Backbone, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.Hardswish()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act2 = nn.Hardswish()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v_input_17):
        x = self.act1(self.fc1(v_input_17))
        x = self.act2(self.fc2(x))
        return self.sigmoid(self.fc3(x))


# Mode 2: 2D Matrix / Dual-Channel Complex-Valued Vision Module
class CCVNN2DVision(nn.Module):
    def __init__(self, in_channels=2, num_classes=10):
        super(CCVNN2DVision, self).__init__()
        # Input: [Batch, 2 (Real/Imag), 64, 64]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.act1 = nn.Hardswish()
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, input_real_imag):
        x = self.act1(self.conv1(input_real_imag))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

# ==============================================================================
# 2. CONFIG BUILDER FOR TRITON MODEL REPOSITORY
# ==============================================================================

def write_triton_configs():
    # Config for Mode 1: ccvnn_v17_backbone
    backbone_config = """name: "ccvnn_v17_backbone"
backend: "openvino"
max_batch_size: 0

input [
  {
    name: "v_input_17"
    data_type: TYPE_FP32
    dims: [ -1, 17 ]
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
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 2000
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]

parameters: {
  key: "NUM_STREAMS"
  value: { string_value: "AUTO" }
}
parameters: {
  key: "INFERENCE_NUM_THREADS"
  value: { string_value: "4" }
}
"""

    # Config for Mode 2: ccvnn (2D Vision)
    vision_config = """name: "ccvnn"
backend: "openvino"
max_batch_size: 0

input [
  {
    name: "input_real_imag"
    data_type: TYPE_FP32
    dims: [ -1, 2, 64, 64 ]
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
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]

parameters: {
  key: "NUM_STREAMS"
  value: { string_value: "AUTO" }
}
parameters: {
  key: "INFERENCE_NUM_THREADS"
  value: { string_value: "4" }
}
"""

    with open("model_repository/ccvnn_v17_backbone/config.pbtxt", "w") as f:
        f.write(backbone_config)
    with open("model_repository/ccvnn/config.pbtxt", "w") as f:
        f.write(vision_config)
    print("✅ Created config.pbtxt files for both ccvnn_v17_backbone and ccvnn.")

# ==============================================================================
# 3. DUAL EXPORT & OPEN VINO CONVERSION PIPELINE
# ==============================================================================

def export_dual_mode_stack():
    # Setup directory hierarchy
    os.makedirs("model_repository/ccvnn_v17_backbone/1", exist_ok=True)
    os.makedirs("model_repository/ccvnn/1", exist_ok=True)
    
    write_triton_configs()

    # --- EXPORT MODE 1: ccvnn_v17_backbone ---
    print("\n--- Processing Mode 1: ccvnn_v17_backbone ---")
    backbone_model = CCVNNV17Backbone().eval()
    backbone_onnx = "model_repository/ccvnn_v17_backbone/1/model.onnx"
    dummy_b17 = torch.randn(1, 17, dtype=torch.float32)

    torch.onnx.export(
        backbone_model,
        dummy_b17,
        backbone_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["v_input_17"],
        output_names=["hazard_prob"],
        dynamic_axes={"v_input_17": {0: "batch_size"}, "hazard_prob": {0: "batch_size"}}
    )
    print(f"✅ Exported ONNX: {backbone_onnx}")

    # Convert to OpenVINO IR
    ov_backbone = ov.convert_model(backbone_onnx)
    ov.save_model(ov_backbone, "model_repository/ccvnn_v17_backbone/1/model.xml")
    print("✅ Converted & Saved OpenVINO IR: model.xml + model.bin")

    # --- EXPORT MODE 2: ccvnn 2D Vision ---
    print("\n--- Processing Mode 2: ccvnn (2D Vision) ---")
    vision_model = CCVNN2DVision().eval()
    vision_onnx = "model_repository/ccvnn/1/model.onnx"
    dummy_vision = torch.randn(1, 2, 64, 64, dtype=torch.float32)

    torch.onnx.export(
        vision_model,
        dummy_vision,
        vision_onnx,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_real_imag"],
        output_names=["output_logits"],
        dynamic_axes={"input_real_imag": {0: "batch_size"}, "output_logits": {0: "batch_size"}}
    )
    print(f"✅ Exported ONNX: {vision_onnx}")

    # Convert to OpenVINO IR
    ov_vision = ov.convert_model(vision_onnx)
    ov.save_model(ov_vision, "model_repository/ccvnn/1/model.xml")
    print("✅ Converted & Saved OpenVINO IR: model.xml + model.bin")

# ==============================================================================
# 4. VERIFICATION ACROSS DYNAMIC BATCH SIZES
# ==============================================================================

def verify_dual_mode_stack():
    print("\n==============================================================")
    print(" RUNNING DYNAMIC BATCH VERIFICATION (OpenVINO Engine)")
    print("==============================================================")
    core = ov.Core()

    # 1. Verify Backbone OpenVINO IR
    b_xml = "model_repository/ccvnn_v17_backbone/1/model.xml"
    b_compiled = core.compile_model(b_xml, "CPU")
    for b_size in [1, 4, 16, 32]:
        test_in = np.random.randn(b_size, 17).astype(np.float32)
        res = b_compiled([test_in])[b_compiled.output(0)]
        assert res.shape == (b_size, 1), f"Backbone mismatch at batch {b_size}"
        print(f" [Backbone] Verified Batch Size {b_size:2d} -> Output Shape: {res.shape}")

    # 2. Verify Vision OpenVINO IR
    v_xml = "model_repository/ccvnn/1/model.xml"
    v_compiled = core.compile_model(v_xml, "CPU")
    for b_size in [1, 4, 16, 32]:
        test_in = np.random.randn(b_size, 2, 64, 64).astype(np.float32)
        res = v_compiled([test_in])[v_compiled.output(0)]
        assert res.shape == (b_size, 10), f"Vision mismatch at batch {b_size}"
        print(f" [Vision]   Verified Batch Size {b_size:2d} -> Output Shape: {res.shape}")

    print("\n✅ DUAL-MODE RECONSTRUCTION AND VERIFICATION COMPLETE!")

if __name__ == "__main__":
    export_dual_mode_stack()
    verify_dual_mode_stack()