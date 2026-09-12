import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="CCVNN Lightweight Inference Microservice")

# Model Registry Configuration
MODELS = {
    "ccvnn_relu6_9d": {
        "path": "model_repository/ccvnn_relu6_9d/1/model.onnx",
        "expected_dim": 9,
        "session": None
    },
    "ccvnn_hardswish_12d": {
        "path": "model_repository/ccvnn_hardswish_12d/1/model.onnx",
        "expected_dim": 12,
        "session": None
    }
}

# Fallback path for 12D model if not in subfolder
if not os.path.exists(MODELS["ccvnn_hardswish_12d"]["path"]) and os.path.exists("hardswish_ccvnn.onnx"):
    MODELS["ccvnn_hardswish_12d"]["path"] = "hardswish_ccvnn.onnx"

# Initialize ONNX Sessions
for name, config in MODELS.items():
    if os.path.exists(config["path"]):
        config["session"] = ort.InferenceSession(config["path"], providers=['CPUExecutionProvider'])
        print(f"[✓] Loaded {name} from {config['path']}")
    else:
        print(f"[!] Warning: Path {config['path']} not found for {name}")

class InferRequest(BaseModel):
    inputs: List[List[float]]

@app.get("/v2/health/ready")
def health_ready():
    return {"status": "READY"}

@app.get("/v2/models/{model_name}/ready")
def model_ready(model_name: str):
    if model_name in MODELS and MODELS[model_name]["session"] is not None:
        return {"name": model_name, "ready": True}
    raise HTTPException(status_code=404, detail="Model not found or not loaded")

@app.post("/v2/models/{model_name}/infer")
def infer(model_name: str, request: InferRequest):
    if model_name not in MODELS or MODELS[model_name]["session"] is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} unavailable")
    
    cfg = MODELS[model_name]
    input_data = np.array(request.inputs, dtype=np.float32)

    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data, axis=0)
        
    if input_data.shape[1] != cfg["expected_dim"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Dimension mismatch for {model_name}: expected {cfg['expected_dim']}, got {input_data.shape[1]}"
        )
    
    session = cfg["session"]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    
    return {
        "model_name": model_name,
        "outputs": outputs[0].tolist()
    }