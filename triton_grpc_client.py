import argparse
import time
import numpy as np
import tritonclient.grpc as grpcclient

def run_inference(url="localhost:8001", batch_size=1, iterations=1000):
    client = grpcclient.InferenceServerClient(url=url)
    
    inputs = [grpcclient.InferInput('input_vector', [batch_size, 9], 'FP32')]
    dummy_data = np.random.randn(batch_size, 9).astype(np.float32)
    inputs[0].set_data_from_numpy(dummy_data)
    
    outputs = [grpcclient.InferRequestedOutput('output_coords')]
    
    print(f"[+] Benchmarking Triton gRPC IPC ({iterations} iterations)...")
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = client.infer(model_name='ccvnn', inputs=inputs, outputs=outputs)
        latencies.append((time.perf_counter() - t0) * 1000)
        
    avg_latency = np.mean(latencies)
    fps = (batch_size * 1000) / avg_latency
    print(f"    - Average Latency: {avg_latency:.3f} ms")
    print(f"    - Throughput:      {fps:.1f} FPS")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='localhost:8001')
    args = parser.parse_args()
    try:
        run_inference(args.url)
    except Exception as e:
        print(f"[!] Triton Connection Check: {e}")