import asyncio
import time
import numpy as np
import httpx

BASE_URL = "http://127.0.0.1:8000/v2/models"

async def send_inference_request(client: httpx.AsyncClient, model_name: str, input_dim: int, req_id: int):
    # Generate batch of 1 sample with 'input_dim' features: [[x1, x2, ..., xN]]
    dummy_input = np.random.randn(1, input_dim).astype(np.float32).tolist()
    
    payload = {
        "inputs": dummy_input
    }
    
    endpoint = f"{BASE_URL}/{model_name}/infer"
    start_time = time.perf_counter()
    
    response = await client.post(endpoint, json=payload)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    if response.status_code == 200:
        res_data = response.json()
        output = res_data.get("outputs", res_data)
        print(f"[Req #{req_id:02d}] {model_name:20s} | Latency: {latency_ms:.2f}ms | Status: OK")
        return res_data
    else:
        print(f"[Req #{req_id:02d}] {model_name:20s} | Failed ({response.status_code}): {response.text}")
        return None

async def main():
    print("--- Starting Concurrent Microservice Benchmark ---")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for i in range(10):
            if i % 2 == 0:
                tasks.append(send_inference_request(client, "ccvnn_relu6_9d", 9, i + 1))
            else:
                tasks.append(send_inference_request(client, "ccvnn_hardswish_12d", 12, i + 1))
        
        start_total = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time_ms = (time.perf_counter() - start_total) * 1000
        
        successful = sum(1 for r in results if r is not None)
        print("--------------------------------------------------")
        print(f"Executed {len(tasks)} requests ({successful} succeeded) in {total_time_ms:.2f}ms")

if __name__ == "__main__":
    asyncio.run(main())