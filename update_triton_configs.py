import os
import openvino as ov

BASE_DIR = "/mnt/c/DDVNN-HOME/CCVNN/model"
core = ov.Core()

MODEL_NAMES = ["ccvnn_v20_backbone", "ccvnn_v21_dynamic", "ccvnn"]

TYPE_MAP = {
    "f32": "TYPE_FP32",
    "f16": "TYPE_FP16",
    "i64": "TYPE_INT64",
    "i32": "TYPE_INT32",
    "boolean": "TYPE_BOOL"
}

for model_name in MODEL_NAMES:
    model_dir = os.path.join(BASE_DIR, model_name)
    xml_path = os.path.join(model_dir, "1", "model.xml")
    bin_path = os.path.join(model_dir, "1", "model.bin")
    config_path = os.path.join(model_dir, "config.pbtxt")

    if not os.path.exists(xml_path) or not os.path.exists(bin_path):
        print(f"[SKIP] {model_name}: missing IR files")
        continue

    model = core.read_model(xml_path, bin_path)

    inputs_str = ""
    for inp in model.inputs:
        names = list(inp.get_tensor().get_names())
        name = names[0] if names else "input"
        ps = inp.get_partial_shape()
        dims = [s.get_length() if s.is_static else -1 for s in ps]
        dtype = TYPE_MAP.get(inp.get_element_type().get_type_name(), "TYPE_FP32")
        
        inputs_str += f"""input [
  {{
    name: "{name}"
    data_type: {dtype}
    dims: [ {', '.join(map(str, dims))} ]
  }}
]
"""

    outputs_str = ""
    for out in model.outputs:
        names = list(out.get_tensor().get_names())
        name = names[0] if names else "output"
        ps = out.get_partial_shape()
        dims = [s.get_length() if s.is_static else -1 for s in ps]
        dtype = TYPE_MAP.get(out.get_element_type().get_type_name(), "TYPE_FP32")
        
        outputs_str += f"""output [
  {{
    name: "{name}"
    data_type: {dtype}
    dims: [ {', '.join(map(str, dims))} ]
  }}
]
"""

    pbtxt_content = f"""name: "{model_name}"
backend: "openvino"
max_batch_size: 0

{inputs_str}{outputs_str}instance_group [
  {{
    count: 2
    kind: KIND_CPU
  }}
]
parameters {{
  key: "INFERENCE_NUM_THREADS"
  value: {{ string_value: "2" }}
}}
parameters {{
  key: "CPU_THROUGHPUT_STREAMS"
  value: {{ string_value: "1" }}
}}
parameters {{
  key: "ENABLE_CPU_PINNING"
  value: {{ string_value: "NO" }}
}}
parameters {{
  key: "PERFORMANCE_HINT"
  value: {{ string_value: "THROUGHPUT" }}
}}
"""
    with open(config_path, "w") as f:
        f.write(pbtxt_content)
    print(f"[UPDATED] {config_path}")
