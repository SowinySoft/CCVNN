import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# Triton gRPC default port is 8001
TRITON_URL = "localhost:8001"
MODEL_NAME = "ccvnn_v20_backbone"

def main():
    # 1. Initialize Triton gRPC Client
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL)
    except Exception as e:
        print(f"[ERROR] Failed to initialize gRPC client: {e}")
        return

    # 2. Check Server & Model Status
    if not client.is_server_ready():
        print(f"[ERROR] Triton server at {TRITON_URL} is not ready.")
        return
    if not client.is_model_ready(MODEL_NAME):
        print(f"[ERROR] Model '{MODEL_NAME}' is not loaded/ready.")
        return

    print(f"[SUCCESS] Connected to Triton gRPC server at {TRITON_URL}")

    # 3. Configure Input/Output Specifications
    # Adjust tensor names and dimensions to match your config.pbtxt
    INPUT_NAME = "input"      # Replace with your actual input tensor name
    OUTPUT_NAME = "output"    # Replace with your actual output tensor name
    
    BATCH_SIZE = 8
    VECTOR_DIM = 512
    
    # Generate dummy payload matching input dimensions and data type
    input_data = np.random.randn(BATCH_SIZE, VECTOR_DIM).astype(np.float32)

    # Prepare Triton input and output objects
    inputs = [grpcclient.InferInput(INPUT_NAME, input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

    # 4. Execute Batch Inference
    print(f"\nSending batch request (Shape: {input_data.shape}, DataType: FP32)...")
    start_time = time.perf_counter()

    try:
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract output tensor
        result = response.as_numpy(OUTPUT_NAME)

        print(f"[SUCCESS] Inference complete in {latency_ms:.2f} ms")
        print(f"Result Tensor Shape : {result.shape}")
        print(f"Result Tensor Dtype : {result.dtype}")
        print(f"First Sample Output : {result[0][:5]}...")

    except InferenceServerException as e:
        print(f"[ERROR] Inference call failed: {e}")

if __name__ == "__main__":
    main()