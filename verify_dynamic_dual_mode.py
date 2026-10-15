#!/usr/bin/env python3
import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

TRITON_URL = "localhost:8001"
MODELS = ["ccvnn_v20_backbone", "ccvnn"]

INPUT_NAME = "input_0"
OUTPUT_NAME = "output_0"

# Test configurations
BATCH_TESTS = [1, 4, 8, 16, 32]
SPATIAL_TESTS = [
    (1, 4, 64, 64),   # Standard shape
    (1, 4, 32, 32),   # Scaled down
    (1, 4, 128, 128), # Scaled up
]

def test_model(client, model_name):
    print(f"\n========================================================")
    print(f" Testing Model: {model_name}")
    print(f"========================================================")

    if not client.is_model_ready(model_name):
        print(f"[ERROR] Model '{model_name}' is NOT ready on Triton.")
        return

    print(f"[SUCCESS] Model '{model_name}' is online.\n")

    # 1. Dynamic Batch Size Validation
    print("--- Test 1: Dynamic Batching (Fixed Spatial [4, 64, 64]) ---")
    for b in BATCH_TESTS:
        shape = (b, 4, 64, 64)
        input_data = np.random.randn(*shape).astype(np.float32)

        inputs = [grpcclient.InferInput(INPUT_NAME, shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        start_t = time.perf_counter()
        try:
            response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
            latency_ms = (time.perf_counter() - start_t) * 1000
            result = response.as_numpy(OUTPUT_NAME)
            print(f"  Batch B={b:2d} | Input: {str(shape):<18} -> Output: {str(result.shape):<12} | Latency: {latency_ms:6.2f} ms [PASSED]")
        except InferenceServerException as e:
            print(f"  Batch B={b:2d} | Input: {str(shape):<18} -> [FAILED]: {e.message()}")

    # 2. Dynamic Spatial Dimensions Validation
    print("\n--- Test 2: Variable Spatial Resolution ---")
    for shape in SPATIAL_TESTS:
        input_data = np.random.randn(*shape).astype(np.float32)

        inputs = [grpcclient.InferInput(INPUT_NAME, shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        start_t = time.perf_counter()
        try:
            response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
            latency_ms = (time.perf_counter() - start_t) * 1000
            result = response.as_numpy(OUTPUT_NAME)
            print(f"  Shape: {str(shape):<18} -> Output: {str(result.shape):<12} | Latency: {latency_ms:6.2f} ms [PASSED]")
        except InferenceServerException as e:
            print(f"  Shape: {str(shape):<18} -> [REJECTED BY TRITON]: {e.message()}")

def main():
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL)
        if not client.is_server_ready():
            print(f"[ERROR] Triton server at {TRITON_URL} is not responding.")
            return
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    for model_name in MODELS:
        test_model(client, model_name)

if __name__ == "__main__":
    main()