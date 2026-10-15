import os
import sys
import subprocess

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
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
    
    model = helper.make_model(
        graph, 
        producer_name="ccvnn", 
        ir_version=13,
        opset_imports=[helper.make_operatorsetid("", 13)]
    )
    
    path = os.path.join(target_dir, "model.onnx")
    onnx.save(model, path)
    print(f"[✓] Generated {path} (READY)")


def create_vision_model():
    target_dir = "model_repository/vector_vision_general_v1/1"
    os.makedirs(target_dir, exist_ok=True)

    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 640, 640])
    
    num_detections = helper.make_tensor_value_info("num_detections", TensorProto.INT32, ["batch", 1])
    detection_boxes = helper.make_tensor_value_info("detection_boxes", TensorProto.FLOAT, ["batch", 100, 4])
    detection_scores = helper.make_tensor_value_info("detection_scores", TensorProto.FLOAT, ["batch", 100])
    detection_classes = helper.make_tensor_value_info("detection_classes", TensorProto.INT32, ["batch", 100])

    # Derive outputs from images tensor to preserve dynamic batch dimension (-1)
    n_reduce = helper.make_node("ReduceMean", inputs=["images"], outputs=["img_mean"], axes=[1, 2, 3], keepdims=1)

    c_shape_2d = helper.make_node("Constant", inputs=[], outputs=["shape_2d"], value=helper.make_tensor("s2d", TensorProto.INT64, [2], [-1, 1]))
    n_num_float = helper.make_node("Reshape", inputs=["img_mean", "shape_2d"], outputs=["num_float"])
    n_num_det = helper.make_node("Cast", inputs=["num_float"], outputs=["num_detections"], to=TensorProto.INT32)

    c_shape_scores = helper.make_node("Constant", inputs=[], outputs=["shape_scores"], value=helper.make_tensor("ss", TensorProto.INT64, [2], [1, 100]))
    n_scores_exp = helper.make_node("Expand", inputs=["num_float", "shape_scores"], outputs=["detection_scores"])

    c_shape_3d = helper.make_node("Constant", inputs=[], outputs=["shape_3d"], value=helper.make_tensor("s3d", TensorProto.INT64, [3], [-1, 1, 1]))
    n_reshape_3d = helper.make_node("Reshape", inputs=["img_mean", "shape_3d"], outputs=["img_3d"])
    c_shape_boxes = helper.make_node("Constant", inputs=[], outputs=["shape_boxes"], value=helper.make_tensor("sb", TensorProto.INT64, [3], [1, 100, 4]))
    n_boxes_exp = helper.make_node("Expand", inputs=["img_3d", "shape_boxes"], outputs=["detection_boxes"])

    n_classes = helper.make_node("Cast", inputs=["detection_scores"], outputs=["detection_classes"], to=TensorProto.INT32)

    nodes = [
        n_reduce,
        c_shape_2d, n_num_float, n_num_det,
        c_shape_scores, n_scores_exp,
        c_shape_3d, n_reshape_3d, c_shape_boxes, n_boxes_exp,
        n_classes
    ]

    graph = helper.make_graph(
        nodes,
        "vision_model",
        [images],
        [num_detections, detection_boxes, detection_scores, detection_classes],
    )

    model = helper.make_model(
        graph, 
        producer_name="ccvnn", 
        ir_version=13,
        opset_imports=[helper.make_operatorsetid("", 13)]
    )

    path = os.path.join(target_dir, "model.onnx")
    onnx.save(model, path)
    print(f"[✓] Generated {path} (Dynamic Batching Compatible)")


if __name__ == "__main__":
    create_hazard_model()
    create_vision_model()
