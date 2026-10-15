import os
import sys
import numpy as np
import tritonclient.http as httpclient

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "ccvnn_v14_model_b")

def run_single_inference_test():
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)

        if not client.is_model_ready(MODEL_NAME):
            print(f"[!] Error: Model '{MODEL_NAME}' is not ready on server {TRITON_URL}.")
            sys.exit(1)

        metadata = client.get_model_metadata(MODEL_NAME)
        
        input_info = metadata["inputs"][0]
        output_info = metadata["outputs"][0]

        input_name = input_info["name"]
        output_name = output_info["name"]

        print(f"--- CCVNN V14 Single Inference Test ---")
        print(f"Target URL:            {TRITON_URL}")
        print(f"Model Name:            {MODEL_NAME}")
        print(f"Detected Input Name:   '{input_name}' ({input_info['datatype']})")
        print(f"Detected Output Name:  '{output_name}' ({output_info['datatype']})")

        # Generate 1 sample with exactly 14 features matching V14 schema
        batch_size = 1
        feature_dim = 14
        dummy_input = np.random.randn(batch_size, feature_dim).astype(np.float32)

        triton_input = httpclient.InferInput(
            name=input_name,
            shape=list(dummy_input.shape),
            datatype=input_info["datatype"]
        )
        triton_input.set_data_from_numpy(dummy_input)

        triton_output = httpclient.InferRequestedOutput(output_name)

        print(f"Sending inference request to '{MODEL_NAME}'...")
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=[triton_input],
            outputs=[triton_output]
        )

        output_data = response.as_numpy(output_name)
        print("\n[✓] Inference Successful!")
        print(f"Output Tensor: {output_name}")
        print(f"Output Shape:  {output_data.shape}")
        print(f"Output Data:   {output_data}")
        sys.exit(0)

    except Exception as e:
        print(f"\n[!] Inference failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_single_inference_test()