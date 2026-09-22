import os
import onnx
from onnx import helper, TensorProto

def main():
    os.makedirs("model_repository/vector_vision_general_v1/1", exist_ok=True)
    os.makedirs("model_repository/ccvnn_hazard_v17/1", exist_ok=True)

    # 1. Vision Stub
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 640, 640])
    num_detections = helper.make_tensor_value_info("num_detections", TensorProto.INT32, ["batch", 1])
    detection_boxes = helper.make_tensor_value_info("detection_boxes", TensorProto.FLOAT, ["batch", 100, 4])
    detection_scores = helper.make_tensor_value_info("detection_scores", TensorProto.FLOAT, ["batch", 100])
    detection_classes = helper.make_tensor_value_info("detection_classes", TensorProto.INT32, ["batch", 100])

    n1 = helper.make_node("Constant", inputs=[], outputs=["num_detections"], value=helper.make_tensor("c1", TensorProto.INT32, [1, 1], [0]))
    n2 = helper.make_node("Constant", inputs=[], outputs=["detection_boxes"], value=helper.make_tensor("c2", TensorProto.FLOAT, [1, 100, 4], [0.0] * 400))
    n3 = helper.make_node("Constant", inputs=[], outputs=["detection_scores"], value=helper.make_tensor("c3", TensorProto.FLOAT, [1, 100], [0.0] * 100))
    n4 = helper.make_node("Constant", inputs=[], outputs=["detection_classes"], value=helper.make_tensor("c4", TensorProto.INT32, [1, 100], [0] * 100))

    g_vision = helper.make_graph([n1, n2, n3, n4], "vision_stub", [images], [num_detections, detection_boxes, detection_scores, detection_classes])
    m_vision = helper.make_model(g_vision, producer_name="ccvnn_ci")
    onnx.save(m_vision, "model_repository/vector_vision_general_v1/1/model.onnx")
    print("[✓] Generated vector_vision_general_v1 model.onnx")

    # 2. Hazard Stub
    input_telemetry = helper.make_tensor_value_info("input_telemetry", TensorProto.FLOAT, ["batch", 4])
    output_hazard = helper.make_tensor_value_info("output_hazard", TensorProto.FLOAT, ["batch", 1])

    nh = helper.make_node("Constant", inputs=[], outputs=["output_hazard"], value=helper.make_tensor("ch1", TensorProto.FLOAT, [1, 1], [0.0]))
    g_hazard = helper.make_graph([nh], "hazard_stub", [input_telemetry], [output_hazard])
    m_hazard = helper.make_model(g_hazard, producer_name="ccvnn_ci")
    onnx.save(m_hazard, "model_repository/ccvnn_hazard_v17/1/model.onnx")
    print("[✓] Generated ccvnn_hazard_v17 model.onnx")

if __name__ == "__main__":
    main()
