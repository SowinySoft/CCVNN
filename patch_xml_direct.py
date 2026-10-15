import os
import xml.etree.ElementTree as ET

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

    print(f"Patching XML directly: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update Parameter (Input) ports
    for layer in root.findall(".//layer[@type='Parameter']"):
        for port in layer.findall("./output/port"):
            port.set("names", names["input"])

    # Update Result (Output) ports
    for layer in root.findall(".//layer[@type='Result']"):
        for port in layer.findall("./input/port"):
            port.set("names", names["output"])

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Successfully patched {model_name}")
