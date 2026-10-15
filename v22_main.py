import asyncio
from contextlib import asynccontextmanager
import os
import sys
import traceback
from typing import Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
import numpy as np
import orjson
from pydantic import BaseModel, Field

# Guarantee script directory is on Python path across multiprocessing spawns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v22_coalescer import BatchCoalescer
from v22_triton_client import TritonAsyncProxy


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        # Enable native NumPy array serialization directly in C
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class PredictRequest(BaseModel):
    data: List[float] = Field(..., example=[0.1] * 16384)


class PredictResponse(BaseModel):
    predictions: List[float]
    coalesced: bool = True


# Triton Configuration
TRITON_URL = os.getenv("TRITON_URL", "127.0.0.1:8001")
DEFAULT_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "ccvnn")

# Gateway Timeout in seconds
GATEWAY_TIMEOUT = float(os.getenv("GATEWAY_TIMEOUT", "15.0"))

# Static fallback specs for known models if Triton metadata discovery is unavailable
KNOWN_MODEL_SPECS: Dict[str, dict] = {
    "ccvnn": {"shape": (4, 64, 64), "flat_dim": 16384},
    "ccvnn_v20_backbone": {"shape": (4, 64, 64), "flat_dim": 16384},
    "ccvnn_v21_dynamic": {"shape": (4, 64, 64), "flat_dim": 16384},
    "ccvnn_v21_int8": {"shape": (4, 64, 64), "flat_dim": 16384},
    "ccvnn_orbit_pipeline": {"shape": (4, 64, 64), "flat_dim": 16384},
    "orbit_sieve_core": {"shape": (4, 64, 64), "flat_dim": 16384},
    "yolov8n": {"shape": (3, 640, 640), "flat_dim": 1228800},
    "yolov8n_openvino": {"shape": (3, 640, 640), "flat_dim": 1228800},
}

triton_proxy = TritonAsyncProxy(triton_url=TRITON_URL, model_name=DEFAULT_MODEL_NAME)
coalescers: Dict[str, BatchCoalescer] = {}
model_element_shapes: Dict[str, Tuple[int, ...]] = {}
coalescer_lock = asyncio.Lock()


def resolve_element_shape(model_name: str, spec: dict) -> Tuple[int, ...]:
    """Extracts non-batch element shape from model metadata spec or fallback configuration."""
    dims = spec.get("input_dims", []) if spec else []
    
    # Strip dynamic or explicitly fixed batch dimension (-1 or dynamic leading dimension)
    if dims and len(dims) == 4:
        element_shape = tuple(dims[1:])
    elif dims and len(dims) == 3:
        element_shape = tuple(dims)
    elif model_name in KNOWN_MODEL_SPECS:
        element_shape = KNOWN_MODEL_SPECS[model_name]["shape"]
    elif "yolo" in model_name.lower():
        element_shape = (3, 640, 640)
    else:
        element_shape = (4, 64, 64)

    # Sanitize negative or wildcard dimensions (-1) returned by dynamic model definitions
    sanitized = []
    for idx, d in enumerate(element_shape):
        if d <= 0:
            if "yolo" in model_name.lower():
                fallback_dim = (3, 640, 640)[idx]
            else:
                fallback_dim = (4, 64, 64)[idx]
            sanitized.append(fallback_dim)
        else:
            sanitized.append(d)

    return tuple(sanitized)


async def get_or_create_coalescer(model_name: str) -> BatchCoalescer:
    if model_name not in coalescers:
        async with coalescer_lock:
            if model_name not in coalescers:
                # Query Triton for exact dynamic input/output names and tensor shapes
                spec = await triton_proxy.inspect_model(model_name=model_name)
                model_element_shapes[model_name] = resolve_element_shape(model_name, spec)

                async def dispatch_fn(batched_input: np.ndarray) -> np.ndarray:
                    return await triton_proxy.infer(batched_input, model_name=model_name)

                # Adjust batching parameters for heavy vision models vs pipeline models
                is_heavy_model = "yolo" in model_name.lower()
                batch_size = 4 if is_heavy_model else 16
                delay_ms = 1.0 if is_heavy_model else 3.0

                coalescer = BatchCoalescer(
                    dispatch_fn=dispatch_fn,
                    max_batch_size=batch_size,
                    max_delay_ms=delay_ms,
                )
                coalescer.start()
                coalescers[model_name] = coalescer
                print(
                    f"[GATEWAY] Dynamic BatchCoalescer started for '{model_name}' "
                    f"(Element Shape: {model_element_shapes[model_name]}, BatchSize: {batch_size})."
                )
    return coalescers[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager."""
    try:
        await triton_proxy.connect()
        # Pre-initialize coalescer and discover metadata for default and key pipeline models
        for initial_model in [DEFAULT_MODEL_NAME, "ccvnn_v20_backbone", "ccvnn_orbit_pipeline"]:
            try:
                await get_or_create_coalescer(initial_model)
            except Exception as m_exc:
                print(f"[GATEWAY WARNING] Deferred initialization for '{initial_model}': {m_exc}")

        print(f"[GATEWAY] Gateway connected to Triton at {TRITON_URL}.")
    except Exception as exc:
        print(f"[GATEWAY ERROR] Initialization failed during startup: {exc}")

    yield

    async with coalescer_lock:
        for m_name, c in coalescers.items():
            await c.stop()
        coalescers.clear()
        model_element_shapes.clear()
    await triton_proxy.close()
    print("[GATEWAY] Gateway resources cleaned up.")


app = FastAPI(
    title="v22-gateway",
    description="High-Throughput REST/gRPC ML API Gateway",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


def _parse_payload_to_array(body_bytes: bytes, elem_shape: tuple) -> np.ndarray:
    """Helper function executed off the main event loop."""
    payload = orjson.loads(body_bytes)
    raw_data = payload.get("data", []) or payload.get("inputs", [])
    data_vec = np.array(raw_data, dtype=np.float32)
    required_size = int(np.prod(elem_shape))

    if data_vec.size < required_size:
        data_vec = np.pad(data_vec, (0, required_size - data_vec.size), mode="constant")
    elif data_vec.size > required_size:
        data_vec = data_vec[:required_size]

    return data_vec.reshape(elem_shape)


@app.post("/v2/models/{model_name}/infer")
@app.post("/v1/predict/{model_name}")
async def predict(model_name: str, request: Request):
    try:
        coalescer = await get_or_create_coalescer(model_name)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gateway dynamic batcher failed for model '{model_name}': {exc}",
        )

    elem_shape = model_element_shapes.get(model_name, (4, 64, 64))

    # Parse and reshape off the main asyncio thread to avoid event-loop blocking
    try:
        body_bytes = await request.body()
        input_arr = await asyncio.to_thread(_parse_payload_to_array, body_bytes, elem_shape)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid payload format: {exc}",
        )

    try:
        async with asyncio.timeout(GATEWAY_TIMEOUT):
            output_tensor = await coalescer.enqueue(input_arr)
            # Pass 1D numpy array directly to orjson without python .tolist() conversion
            return ORJSONResponse({"predictions": output_tensor.flatten(), "coalesced": True})
    except (TimeoutError, asyncio.TimeoutError):
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Inference request timed out in queue for model '{model_name}'.",
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {type(exc).__name__} - {str(exc)}",
        )


@app.get("/health/live")
async def liveness():
    return {"status": "live"}


@app.get("/health/ready")
async def readiness(response: Response, model_name: str = DEFAULT_MODEL_NAME):
    is_ready = await triton_proxy.is_ready(model_name=model_name)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready", "model": model_name, "triton_connected": False}
    return {"status": "ready", "model": model_name, "triton_connected": True}