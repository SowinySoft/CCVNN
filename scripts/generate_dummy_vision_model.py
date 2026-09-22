"""
Generates a minimal valid ONNX model stub for vector_vision_general_v1.
Used by local test environments and GitHub Actions CI runner.
"""

import os
import onnx
from onnx import helper, TensorProto


def generate_stub():
    target_dir = "model_repository/vector_vision_general_v1/1"
    os.makedirs(target_dir, exist_ok=True)

    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, ["batch", 3, 640, 640]
    )
    num_detections = helper.make_tensor_value_info(
        "num_detections", TensorProto.INT32, ["batch", 1]
    )
    detection_boxes = helper.make_tensor_value_info(
        "detection_boxes", TensorProto.FLOAT, ["batch", 100, 4]
    )
    detection_scores = helper.make_tensor_value_info(
        "detection_scores", TensorProto.FLOAT, ["batch", 100]
    )
    detection_classes = helper.make_tensor_value_info(
        "detection_classes", TensorProto.INT32, ["batch", 100]
    )

    node1 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["num_detections"],
        value=helper.make_tensor("c1", TensorProto.INT32, [1, 1], [0]),
    )
    node2 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["detection_boxes"],
        value=helper.make_tensor(
            "c2", TensorProto.FLOAT, [1, 100, 4], [0.0] * 400
        ),
    )
    node3 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["detection_scores"],
        value=helper.make_tensor("c3", TensorProto.FLOAT, [1, 100], [0.0] * 100),
    )
    node4 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["detection_classes"],
        value=helper.make_tensor("c4", TensorProto.INT32, [1, 100], [0] * 100),
    )

    graph = helper.make_graph(
        [node1, node2, node3, node4],
        "vector_vision_stub",
        [images],
        [num_detections, detection_boxes, detection_scores, detection_classes],
    )

    model = helper.make_model(graph, producer_name="ccvnn_ci")
    onnx_path = os.path.join(target_dir, "model.onnx")
    onnx.save(model, onnx_path)
    print(f"[✓] Successfully generated stub ONNX model at: {onnx_path}")


if __name__ == "__main__":
    generate_stub()