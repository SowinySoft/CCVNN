import os
import openvino as ov

MODEL_SPECS = {
    "ccvnn_v20_backbone": {"input": "43", "output": "hazard_prob"},
    "ccvnn_v21_dynamic": {"input": "84", "output": "output_logits"},
    "ccvnn": {"input": "84", "output": "output_logits"},
}

BASE_DIR = "/mnt/c/DDVNN-HOME/CCVNN/model"
core = ov.Core()

for model_name, names in MODEL_SPECS.items():
    xml_path = os.path.join(BASE_DIR, model_name, "1", "model.xml")
    if not os.path.exists(xml_path):
        print(f"Skipping {xml_path} (not found)")
        continue

    print(f"Patching {xml_path}...")
    model = core.read_model(xml_path)

    for idx, inp in enumerate(model.inputs):
        if not inp.get_names():
            target_name = names["input"] if idx == 0 else f"input_{idx}"
            inp.get_tensor().set_names({target_name})

    for idx, out in enumerate(model.outputs):
        if not out.get_names():
            target_name = names["output"] if idx == 0 else f"output_{idx}"
            out.get_tensor().set_names({target_name})

    ov.save_model(model, xml_path)
    print(f"Successfully updated {model_name}")
