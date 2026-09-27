import openvino as ov

# Initialize OpenVINO Core
core = ov.Core()
model_path = "models/vector_vision_general_v1_cpu/1/model.xml"

# Load the XML model
model = core.read_model(model_path)

# Define dynamic shape [-1, 3, 640, 640] (-1 allows dynamic batching)
model.reshape({"images": ov.PartialShape([-1, 3, 640, 640])})

# Save the updated model IR
ov.save_model(model, model_path)
print("Successfully reshaped OpenVINO model to dynamic batch size!")