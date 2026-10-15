import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

if client.is_server_live():
    print("Triton server is operational.\n")

# --- 1. Test ccvnn_hardswish_12d ---
model_12d_name = "ccvnn_hardswish_12d"
input_data_12d = np.random.randn(1, 12).astype(np.float32)

inputs_12d = [httpclient.InferInput("input_vector", input_data_12d.shape, "FP32")]
inputs_12d[0].set_data_from_numpy(input_data_12d)

outputs_12d = [httpclient.InferRequestedOutput("output_prediction")]

results_12d = client.infer(
    model_name=model_12d_name, inputs=inputs_12d, outputs=outputs_12d
)
print(f"{model_12d_name} output:", results_12d.as_numpy("output_prediction"))

# --- 2. Test ccvnn_relu6_9d ---
model_9d_name = "ccvnn_relu6_9d"
input_data_9d = np.random.randn(1, 9).astype(np.float32)

inputs_9d = [httpclient.InferInput("input_vector", input_data_9d.shape, "FP32")]
inputs_9d[0].set_data_from_numpy(input_data_9d)

outputs_9d = [httpclient.InferRequestedOutput("output_coords")]

results_9d = client.infer(
    model_name=model_9d_name, inputs=inputs_9d, outputs=outputs_9d
)
print(f"{model_9d_name} output:", results_9d.as_numpy("output_coords"))