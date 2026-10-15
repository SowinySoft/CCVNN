import numpy as np
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

MAX_CLIENT_BATCH_SIZE = 128  # Client-facing max payload limit
TRITON_MAX_BATCH_SIZE = 16   # Matches max_batch_size in config.pbtxt
FEATURE_DIM = 17

# Pre-allocated input buffer for max Triton chunk
_TRITON_INPUT_BUFFER = np.empty((TRITON_MAX_BATCH_SIZE, FEATURE_DIM), dtype=np.float32)

app = FastAPI(title="CCVNN Inference Gateway")

# Initialize Triton Client (adjust host/port if necessary)
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

class BatchInferenceRequest(BaseModel):
    samples: list[list[float]]

@app.post("/predict-batch")
async def predict_batch(request: BatchInferenceRequest):
    total_samples = len(request.samples)
    
    # 1. Guard against empty payloads
    if total_samples == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch payload cannot be empty."
        )

    # 2. Guard against oversized client requests before calling Triton
    if total_samples > MAX_CLIENT_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {total_samples} exceeds maximum client limit of {MAX_CLIENT_BATCH_SIZE}."
        )

    # 3. Validate dimension shapes
    for i, sample in enumerate(request.samples):
        if len(sample) != FEATURE_DIM:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sample at index {i} has dimension {len(sample)}, expected {FEATURE_DIM}."
            )

    results = []

    # 4. Chunk client payload into sub-batches <= TRITON_MAX_BATCH_SIZE (16)
    try:
        for start_idx in range(0, total_samples, TRITON_MAX_BATCH_SIZE):
            end_idx = min(start_idx + TRITON_MAX_BATCH_SIZE, total_samples)
            chunk = request.samples[start_idx:end_idx]
            chunk_size = len(chunk)

            # Copy chunk into slice of pre-allocated contiguous buffer
            _TRITON_INPUT_BUFFER[:chunk_size] = chunk
            input_slice = _TRITON_INPUT_BUFFER[:chunk_size]

            # Construct Triton Inputs
            triton_input = grpcclient.InferInput("v_input_17", input_slice.shape, "FP32")
            triton_input.set_data_from_numpy(input_slice)

            # Synchronous Triton Execution
            response = triton_client.infer(
                model_name="ccvnn_v17_backbone",
                inputs=[triton_input]
            )

            # Extract outputs and append to results list
            chunk_output = response.as_numpy("hazard_prob").flatten().tolist()
            results.extend(chunk_output)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triton Engine Failure: {str(e)}"
        )

    return {"hazard_probabilities": results}