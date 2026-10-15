import os
import numpy as np
import openvino as ov
import nncf

# 1. Load FP32 OpenVINO IR
core = ov.Core()
model_path = "model/ccvnn_v21_dynamic/1/model.xml"
model = core.read_model(model_path)

input_name = model.inputs[0].any_name

# 2. Reshape input to fix channel dimension (4) while keeping batch size dynamic (-1)
# This prevents NNCF slice/quantize shape collapse to 0 channels.
print(f"Reshaping input '{input_name}' to dynamic batch shape [-1, 4, 64, 64]...")
model.reshape({input_name: ov.PartialShape([-1, 4, 64, 64])})

# 3. Calibration dataset setup
def transform_fn(data_item):
    return {input_name: data_item}

calibration_samples = [
    np.random.randn(1, 4, 64, 64).astype(np.float32)
    for _ in range(100)
]
calibration_dataset = nncf.Dataset(calibration_samples, transform_fn)

# 4. Run NNCF Post-Training Quantization
print("Starting NNCF INT8 Post-Training Quantization...")
quantized_model = nncf.quantize(
    model,
    calibration_dataset,
    preset=nncf.QuantizationPreset.PERFORMANCE
)

# 5. Save corrected INT8 model
output_dir = "model/ccvnn_v21_int8/1"
os.makedirs(output_dir, exist_ok=True)
ov.save_model(quantized_model, os.path.join(output_dir, "model.xml"))
print(f"✓ Quantized INT8 model saved successfully to {output_dir}/model.xml")