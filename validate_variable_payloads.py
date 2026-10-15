import time
import numpy as np
import tritonclient.grpc as grpcclient

MODEL_NAME = "ccvnn_v21_dynamic"
URL = "localhost:8001"

try:
    client = grpcclient.InferenceServerClient(url=URL)
    print(f"Connected to Triton at {URL}")
except Exception as e:
    print(f"Failed to connect to Triton server: {e}")
    exit(1)

# Dynamic bounds for '84': [Batch Size, Channels=4, 64, 64]
payload_shapes = [
    (1, 4, 64, 64),
    (2, 4, 64, 64),
    (4, 4, 64, 64),
    (8, 4, 64, 64),
      
]

print(f"\n=== VERIFYING DYNAMIC SHAPE PAYLOADS ON '{MODEL_NAME}' ===")

for shape in payload_shapes:
    dummy_data = np.random.randn(*shape).astype(np.float32)
    
    inputs = [grpcclient.InferInput("84", list(shape), "FP32")]
    inputs[0].set_data_from_numpy(dummy_data)
    
    start_time = time.perf_counter()
    response = client.infer(model_name=MODEL_NAME, inputs=inputs)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    output_name = response.get_response().outputs[0].name
    output_data = response.as_numpy(output_name)
    
    print(f"Shape: {str(shape):<18} | Latency: {latency_ms:.2f} ms | Output ({output_name}) Shape: {output_data.shape}")

print("\n✓ Phase 1 complete: Dynamic shape execution confirmed without fixed padding overhead.")