import json
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "input_0")
            if in_tensor is None:
                raise pb_utils.TritonModelException("Input tensor 'input_0' not found in inference request.")

            v20_features = in_tensor.as_numpy()

            if v20_features.ndim > 1 and v20_features.shape[-1] == 1:
                v20_features = np.squeeze(v20_features, axis=-1)

            if v20_features.ndim == 1:
                v20_features = np.expand_dims(v20_features, axis=0)

            # Ensure we maintain a proper feature dimension width for downstream reshaping
            feature_dim = max(v20_features.shape[-1] // 2, 1)
            chunk_a = v20_features[..., :feature_dim]
            chunk_b = v20_features[..., feature_dim : feature_dim * 2]
            if chunk_b.shape[-1] == 0:
                chunk_b = np.ones_like(chunk_a)

            max_sift_steps = 3
            current_representation = chunk_a
            
            for step in range(max_sift_steps):
                transformed = np.tanh(current_representation * np.swapaxes(chunk_b, min(chunk_b.ndim-1, 0), -1))
                delta = np.mean(np.abs(transformed - current_representation))
                current_representation = transformed
                if delta < 1e-4:
                    break

            # Ensure output is flattened per batch item with a guaranteed non-scalar feature vector
            if current_representation.ndim > 1:
                flat_out = current_representation.reshape(current_representation.shape[0], -1)
            else:
                flat_out = np.expand_dims(current_representation, axis=0)

            output_tensor = pb_utils.Tensor("final_output", flat_out.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        pass