import os
import sys
import subprocess

# Auto-install onnx if missing in CI runner environment
try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("[!] 'onnx' library not found. Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
    import onnx
    from onnx import helper, TensorProto


def create_hazard_model():
    target_dir = "model_repository/ccvnn_hazard_v17/1"
    os.makedirs(target_dir, exist_ok=True)

    input_telemetry = helper.make_tensor_value_info("input_telemetry", TensorProto.FLOAT, ["batch", 4])
    output_hazard = helper.make_tensor_value_info("output_hazard", TensorProto.FLOAT, ["batch", 1])

    node = helper.make_node("ReduceMean", inputs=["input_telemetry"], outputs=["output_hazard"], axes=[1], keepdims=1)
    graph = helper.make_graph([node], "hazard_model", [input_telemetry], [output_hazard])
    model = helper.make_model(graph, producer_name="ccvnn")
    
    path = os.path.join(target_dir, "model.onnx")
    onnx.save(model, path)
    print(f"[✓] Generated {path}")


def create_vision_model():
    target_dir = "model_repository/vector_vision_general_v1/1"
    os.makedirs(target_dir, exist_ok=True)

    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 640, 640])
    num_detections = helper.make_tensor_value_info("num_detections", TensorProto.INT32, ["batch", 1])
    detection_boxes = helper.make_tensor_value_info("detection_boxes", TensorProto.FLOAT, ["batch", 100, 4])
    detection_scores = helper.make_tensor_value_info("detection_scores", TensorProto.FLOAT, ["batch", 100])
    detection_classes = helper.make_tensor_value_info("detection_classes", TensorProto.INT32, ["batch", 100])

    n1 = helper.make_node("Constant", inputs=[], outputs=["num_detections"], value=helper.make_tensor("c1", TensorProto.INT32, [1, 1], [0]))
    n2 = helper.make_node("Constant", inputs=[], outputs=["detection_boxes"], value=helper.make_tensor("c2", TensorProto.FLOAT, [1, 100, 4], [0.0] * 400))
    n3 = helper.make_node("Constant", inputs=[], outputs=["detection_scores"], value=helper.make_tensor("c3", TensorProto.FLOAT, [1, 100], [0.0] * 100))
    n4 = helper.make_node("Constant", inputs=[], outputs=["detection_classes"], value=helper.make_tensor("c4", TensorProto.INT32, [1, 100], [0] * 100))

    graph = helper.make_graph(
        [n1, n2, n3, n4],
        "vision_model",
        [images],
        [num_detections, detection_boxes, detection_scores, detection_classes],
    )
    model = helper.make_model(graph, producer_name="ccvnn")

    path = os.path.join(target_dir, "model.onnx")
    onnx.save(model, path)
    print(f"[✓] Generated {path}")


if __name__ == "__main__":
    create_hazard_model()
    create_vision_model()
