import numpy as np
import openvino as ov
import nncf

# 1. Load FP16 OpenVINO IR model
core = ov.Core()
model_path = "models/vector_vision_general_v1_ir/1/model.xml"
model = core.read_model(model_path)

# 2. Create synthetic calibration dataset (64 representative input frames)
def transform_fn(data_item):
    return { "images": data_item }

calibration_data = [
    np.random.rand(1, 3, 640, 640).astype(np.float32) for _ in range(64)
]
calibration_dataset = nncf.Dataset(calibration_data, transform_fn)

# 3. Apply INT8 Post-Training Quantization (PTQ)
quantized_model = nncf.quantize(
    model=model,
    calibration_dataset=calibration_dataset,
    preset=nncf.QuantizationPreset.PERFORMANCE  # Optimizes for latency on CPU
)

# 4. Serialize quantized INT8 model
ov.save_model(
    quantized_model, 
    "models/vector_vision_general_v1_int8/1/model.xml"
)
print("INT8 Model saved successfully to models/vector_vision_general_v1_int8/1/")