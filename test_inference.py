import os
import sys
import numpy as np
import tritonclient.http as httpclient

TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
TARGET_MODELS = os.getenv("TARGET_MODELS", "ccvnn_v14_model_a,ccvnn_v14_model_b").split(",")

def test_v14_inference_suite():
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)
    except Exception as e:
        print(f"[!] Failed to initialize Triton HTTP client at {TRITON_URL}: {e}")
        sys.exit(1)

    print(f"=== CCVNN V14 Multi-Model Test Suite ===")
    print(f"Server Target: {TRITON_URL}")
    print(f"Models to Test: {TARGET_MODELS}\n")

    overall_success = True

    # Standard 14D V14 Feature Vector Payload: shape (1, 14)
    dummy_input_14d = np.random.randn(1, 14).astype(np.float32)

    for model_name in TARGET_MODELS:
        model_name = model_name.strip()
        print(f"[*] Testing model: '{model_name}'...")
        
        try:
            if not client.is_model_ready(model_name):
                print(f"    [!] Error: Model '{model_name}' is not ready on Triton server.")
                overall_success = False
                continue

            # Resolve dynamic input/output tensor names from model metadata
            metadata = client.get_model_metadata(model_name)
            input_name = metadata["inputs"][0]["name"]
            output_name = metadata["outputs"][0]["name"]

            inputs = [httpclient.InferInput(input_name, dummy_input_14d.shape, "FP32")]
            inputs[0].set_data_from_numpy(dummy_input_14d)

            outputs = [httpclient.InferRequestedOutput(output_name)]

            response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
            output_data = response.as_numpy(output_name)

            print(f"    [✓] Output Tensor ('{output_name}'): Shape {output_data.shape} | Sample: {output_data.flatten()[:4]}")

        except Exception as e:
            print(f"    [!] Failed testing '{model_name}': {e}")
            overall_success = False

    print("\n========================================")
    if overall_success:
        print("[✓] All CCVNN V14 model inference tests passed successfully.")
        sys.exit(0)
    else:
        print("[!] One or more model tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_v14_inference_suite()