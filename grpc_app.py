from contextlib import asynccontextmanager
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException

TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "ccvnn_v17_backbone")
INPUT_NAME = "v_input_17"
OUTPUT_NAME = "hazard_prob"

MAX_CLIENT_BATCH_SIZE = 128
TRITON_MAX_BATCH_SIZE = 16
FEATURE_DIM = 17

triton_client: Optional[grpcclient.InferenceServerClient] = None
executor: Optional[ThreadPoolExecutor] = None

def _build_numpy_array(features: List[float], shape: tuple) -> np.ndarray:
    return np.array(features, dtype=np.float32).reshape(shape)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global triton_client, executor
    executor = ThreadPoolExecutor(max_workers=8)
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)
    yield
    # Graceful Shutdown: Drain thread pool and close gRPC client channel cleanly
    if executor:
        executor.shutdown(wait=True, cancel_futures=False)
    if triton_client:
        await triton_client.close()

app = FastAPI(title="CCVNN Backbone Inference API", lifespan=lifespan)

class BatchVectorInputRequest(BaseModel):
    samples: List[List[float]]

class BatchInferenceResponse(BaseModel):
    hazard_probabilities: List[float]

@app.post("/predict-batch", response_model=BatchInferenceResponse)
async def predict_batch(payload: BatchVectorInputRequest):
    total_samples = len(payload.samples)
    if total_samples == 0 or total_samples > MAX_CLIENT_BATCH_SIZE:
        raise HTTPException(status_code=400, detail="Invalid batch size")

    loop = asyncio.get_running_loop()
    all_probs = []

    chunks = [
        payload.samples[i : i + TRITON_MAX_BATCH_SIZE]
        for i in range(0, total_samples, TRITON_MAX_BATCH_SIZE)
    ]

    for chunk_samples in chunks:
        n_samples = len(chunk_samples)
        flat_list = [item for sublist in chunk_samples for item in sublist]
        chunk_array = await loop.run_in_executor(
            executor, _build_numpy_array, flat_list, (n_samples, FEATURE_DIM)
        )
        chunk_inputs = [grpcclient.InferInput(INPUT_NAME, [n_samples, FEATURE_DIM], "FP32")]
        chunk_inputs[0].set_data_from_numpy(chunk_array)
        chunk_outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        try:
            res = await triton_client.infer(
                model_name=MODEL_NAME,
                inputs=chunk_inputs,
                outputs=chunk_outputs,
            )
            probs = res.as_numpy(OUTPUT_NAME).flatten().tolist()
            all_probs.extend(probs)
        except InferenceServerException as e:
            raise HTTPException(status_code=500, detail=str(e))

    return BatchInferenceResponse(hazard_probabilities=all_probs)
