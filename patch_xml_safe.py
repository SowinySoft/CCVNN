import os
import re

MODEL_SPECS = {
    "ccvnn_v20_backbone": {"input": "43", "output": "hazard_prob"},
    "ccvnn_v21_dynamic": {"input": "84", "output": "output_logits"},
    "ccvnn": {"input": "84", "output": "output_logits"},
}

BASE_DIR = "/mnt/c/DDVNN-HOME/CCVNN/model"

for model_name, names in MODEL_SPECS.items():
    xml_path = os.path.join(BASE_DIR, model_name, "1", "model.xml")
    if not os.path.exists(xml_path):
        print(f"Skipping {xml_path} (not found)")
        continue

    if os.path.getsize(xml_path) == 0:
        print(f"ERROR: {xml_path} is 0 bytes! Please restore a valid model.xml file first.")
        continue

    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Inject names attribute into output port of Parameter layers (Inputs)
    content_updated = re.sub(
        r'(<layer[^>]*type="Parameter"[^>]*>.*?<output>.*?<port\s+id="\d+"[^\/>]*)(\/?>)',
        rf'\1 names="{names["input"]}"\2',
        content,
        flags=re.DOTALL
    )

    # Inject names attribute into input port of Result layers (Outputs)
    content_updated = re.sub(
        r'(<layer[^>]*type="Result"[^>]*>.*?<input>.*?<port\s+id="\d+"[^\/>]*)(\/?>)',
        rf'\1 names="{names["output"]}"\2',
        content_updated,
        flags=re.DOTALL
    )

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(content_updated)

    print(f"Successfully patched {model_name}")
