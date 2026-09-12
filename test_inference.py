import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

# Test ccvnn_relu6_9d
input_data_9d = np.random.randn(1, 9).astype(np.float32)
inputs = [httpclient.InferInput("input_vector", input_data_9d.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data_9d)

response = client.infer(model_name="ccvnn_relu6_9d", inputs=inputs)
print("ccvnn_relu6_9d output:", response.as_numpy("output_coords"))

# Test ccvnn_hardswish_12d
input_data_12d = np.random.randn(1, 12).astype(np.float32)
inputs_12d = [httpclient.InferInput("input_vector", input_data_12d.shape, "FP32")]
inputs_12d[0].set_data_from_numpy(input_data_12d)

response_12d = client.infer(model_name="ccvnn_hardswish_12d", inputs=inputs_12d)
print("ccvnn_hardswish_12d output:", response_12d.as_numpy("output_prediction"))