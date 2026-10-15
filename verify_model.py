import onnx

model_path = "model_repository/ccvnn_hardswish_12d/1/model.onnx"
model = onnx.load(model_path)

print(f"ONNX IR Version: {model.ir_version}")
print(f"Number of Graph Nodes: {len(model.graph.node)}")
print(f"Number of Initializer Tensors (Weights): {len(model.graph.initializer)}")

for init in model.graph.initializer:
    print(f"Tensor: {init.name} | Data Type: {init.data_type} | Shape: {list(init.dims)}")
