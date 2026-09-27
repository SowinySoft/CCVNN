from contextlib import asynccontextmanager
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
import numpy as np
from pydantic import BaseModel, Field
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException

TRITON_SERVER_URL = "localhost:8001"
MODEL_NAME = "ccvnn_v17_backbone"
INPUT_NAME = "v_input_17"
OUTPUT_NAME = "hazard_prob"

# Global client reference to reuse gRPC channels across requests
triton_client: Optional[grpcclient.InferenceServerClient] = None


async def warm_up_triton(
    client: grpcclient.InferenceServerClient, passes: int = 5
) -> None:
    """Pre-warms Triton C++ execution handles and OpenVINO engines."""
    dummy_input = np.zeros((1, 17), dtype=np.float32)
    inputs = [grpcclient.InferInput(INPUT_NAME, dummy_input.shape, "FP32")]
    inputs[0].set_data_from_numpy(dummy_input)
    outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

    for _ in range(passes):
        await client.infer(
            model_name=MODEL_NAME, inputs=inputs, outputs=outputs
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages persistent gRPC connection pool and warm-up lifecycle."""
    global triton_client

    # Initialization / Startup
    triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)

    if not await triton_client.is_server_live():
        raise RuntimeError(
            f"Triton Inference Server at {TRITON_SERVER_URL} is unreachable."
        )

    if not await triton_client.is_model_ready(MODEL_NAME):
        raise RuntimeError(
            f"Model '{MODEL_NAME}' is not loaded or ready on Triton."
        )

    print(
        f"[+] Connected to Triton at {TRITON_SERVER_URL}. Warming up engine..."
    )
    await warm_up_triton(triton_client, passes=5)
    print(f"[+] Lifespan warm-up complete. Service ready to serve traffic.")

    yield

    # Teardown / Shutdown
    if triton_client:
        await triton_client.close()
        print("[+] Triton gRPC client channel closed successfully.")


app = FastAPI(
    title="CCVNN Backbone Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


class VectorInputRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=17,
        max_length=17,
        description="17-dimensional vector representation.",
        json_schema_extra={"example": [0.12] * 17},
    )


class SingleInferenceResponse(BaseModel):
    hazard_probability: float


class BatchVectorInputRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of 17-dimensional feature vectors.",
        json_schema_extra={"example": [[0.12] * 17, [0.45] * 17]},
    )


class BatchInferenceResponse(BaseModel):
    hazard_probabilities: List[float]


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Endpoint for readiness and liveness probes."""
    if not triton_client or not await triton_client.is_model_ready(MODEL_NAME):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton model backend unavailable",
        )
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/predict", response_model=SingleInferenceResponse)
async def predict_single(payload: VectorInputRequest):
    """Single vector hazard probability prediction."""
    try:
        input_array = (
            np.array(payload.features, dtype=np.float32)
            .reshape(1, 17)
        )

        inputs = [grpcclient.InferInput(INPUT_NAME, input_array.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_array)
        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        response = await triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        result = response.as_numpy(OUTPUT_NAME)
        return SingleInferenceResponse(
            hazard_probability=float(result.flatten()[0])
        )

    except InferenceServerException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triton Engine Failure: {str(e)}",
        )


@app.post("/predict-batch", response_model=BatchInferenceResponse)
async def predict_batch(payload: BatchVectorInputRequest):
    """Batch vector prediction leveraging dynamic shape support."""
    try:
        for idx, sample in enumerate(payload.samples):
            if len(sample) != 17:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Sample at index {idx} does not have length 17.",
                )

        input_array = np.array(payload.samples, dtype=np.float32)

        inputs = [grpcclient.InferInput(INPUT_NAME, input_array.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_array)
        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        response = await triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
        )

        result = response.as_numpy(OUTPUT_NAME)
        return BatchInferenceResponse(
            hazard_probabilities=result.flatten().tolist()
        )

    except InferenceServerException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triton Engine Failure: {str(e)}",
        )