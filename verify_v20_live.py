import numpy as np
import tritonclient.http as httpclient

def test_live_endpoints():
    client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Test Data: Batch of 2 RGBA images [2, 4, 64, 64]
    dummy_input = np.random.randn(2, 4, 64, 64).astype(np.float32)
    
    inputs = [
        httpclient.InferInput("input_rgba", dummy_input.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(dummy_input)
    
    # 1. Query V20 Backbone
    res_backbone = client.infer("ccvnn_v20_backbone", inputs)
    print("✅ V20 Backbone Response Shape:", res_backbone.as_numpy("hazard_prob").shape)
    
    # 2. Query V20 Vision Engine
    res_vision = client.infer("ccvnn", inputs)
    print("✅ V20 Vision Engine Response Shape:", res_vision.as_numpy("output_logits").shape)

if __name__ == "__main__":
    test_live_endpoints()