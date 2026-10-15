import cv2
import numpy as np
import tritonclient.http as httpclient
import json
import os

def send_frame_to_triton(image_path, model_name="ccvnn_v17_chromatic", url="localhost:8000"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = img_rgb.astype(np.uint8)

    client = httpclient.InferenceServerClient(url=url)
    
    metadata = client.get_model_metadata(model_name)
    input_name = metadata['inputs'][0]['name']
    output_name = metadata['outputs'][0]['name']

    inputs = [httpclient.InferInput(input_name, input_data.shape, "UINT8")]
    inputs[0].set_data_from_numpy(input_data)

    response = client.infer(model_name=model_name, inputs=inputs)
    output_data = response.as_numpy(output_name)
    
    raw_str = output_data.flat[0].decode('utf-8') if isinstance(output_data.flat[0], bytes) else str(output_data.flat[0])
    
    try:
        telemetry = json.loads(raw_str)
    except json.JSONDecodeError:
        telemetry = {"raw_output": raw_str}

    plc_info = telemetry.get("modbus_tcp_plc_actuation", {})
    coil = plc_info.get("register_40001_coil", 0)
    
    return telemetry, coil
