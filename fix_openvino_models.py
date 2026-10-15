import os
import shutil
import tempfile
import openvino as ov

MODEL_SPECS = {
    "ccvnn_v20_backbone": {"input": "43", "output": "hazard_prob"},
    "ccvnn_v21_dynamic": {"input": "84", "output": "output_logits"},
    "ccvnn": {"input": "84", "output": "output_logits"},
}

BASE_DIR = "/mnt/c/DDVNN-HOME/CCVNN/model"

for model_name, names in MODEL_SPECS.items():
    model_dir = os.path.join(BASE_DIR, model_name, "1")
    xml_path = os.path.join(model_dir, "model.xml")
    bin_path = os.path.join(model_dir, "model.bin")

    if not os.path.exists(xml_path) or not os.path.exists(bin_path):
        print(f"[SKIP] {model_name}: missing model.xml or model.bin")
        continue

    if os.path.getsize(xml_path) == 0 or os.path.getsize(bin_path) == 0:
        print(f"[WARNING] {model_name}: files are 0 bytes!")
        continue

    print(f"[PROCESSING] {model_name}...")
    try:
        core = ov.Core()
        model = core.read_model(xml_path, bin_path)

        for idx, inp in enumerate(model.inputs):
            target_name = names["input"] if idx == 0 else f"input_{idx}"
            inp.get_tensor().set_names({target_name})

        for idx, out in enumerate(model.outputs):
            target_name = names["output"] if idx == 0 else f"output_{idx}"
            out.get_tensor().set_names({target_name})

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_xml = os.path.join(tmp_dir, "model.xml")
            ov.save_model(model, tmp_xml)
            shutil.move(tmp_xml, xml_path)
            shutil.move(os.path.join(tmp_dir, "model.bin"), bin_path)

        print(f"[SUCCESS] Patched {model_name}\n")
    except Exception as e:
        print(f"[ERROR] {model_name}: {e}\n")
