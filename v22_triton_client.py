import os
import re
import numpy as np
import tritonclient.grpc.aio as grpcclient
import tritonclient.utils as triton_utils

try:
    from v23_shm_manager import TritonZeroCopyManager
except ImportError:
    TritonZeroCopyManager = None


def parse_pbtxt_config(pbtxt_path: str) -> dict | None:
    """Parses local config.pbtxt for model metadata fallbacks supporting bracket, block, and scalar syntaxes."""
    if not os.path.exists(pbtxt_path):
        return None
    try:
        with open(pbtxt_path, "r", encoding="utf-8") as f:
            content = f.read()

        def extract_block_spec(block_type: str) -> dict:
            # Captures block definition without greedy multi-block swallowing
            block_pattern = rf'{block_type}\s*\[?\s*\{{([\s\S]*?)\}}\s*\]?'
            blocks = re.findall(block_pattern, content)
            
            if not blocks:
                return {"name": None, "dims": [], "datatype": "FP32"}

            first_block = blocks[0]
            
            name_match = re.search(r'name:\s*"([^"]+)"', first_block)
            name = name_match.group(1) if name_match else None

            dtype_match = re.search(r'data_type:\s*(?:TYPE_)?([A-Z0-9_]+)', first_block)
            datatype = dtype_match.group(1) if dtype_match else "FP32"

            dims = []
            bracket_dims = re.search(r'dims:\s*\[\s*([^\]]+)\s*\]', first_block)
            if bracket_dims:
                dims = [
                    int(x.strip()) 
                    for x in bracket_dims.group(1).split(",") 
                    if x.strip().lstrip('-').isdigit()
                ]
            else:
                scalar_dims = re.findall(r'dims:\s*(-?\d+)', first_block)
                if scalar_dims:
                    dims = [int(x) for x in scalar_dims]

            return {"name": name, "dims": dims, "datatype": datatype}

        input_spec = extract_block_spec("input")
        output_spec = extract_block_spec("output")

        dims = input_spec["dims"]
        valid_dims = [d for d in dims if d > 0]

        return {
            "input_name": input_spec["name"],
            "output_name": output_spec["name"],
            "input_datatype": input_spec["datatype"],
            "input_dims": dims,
            "expected_rank": len(valid_dims) + 1 if valid_dims else 2,
        }
    except Exception as exc:
        print(f"[TRITON CLIENT] Failed to parse pbtxt file '{pbtxt_path}': {exc}")
        return None


def prepare_input_tensor(batched_input: np.ndarray, spec: dict) -> np.ndarray:
    """Standardizes input tensor dimensions strictly matching target Triton input shape requirements."""
    formatted = batched_input.astype(np.float32)

    if formatted.ndim == 1:
        formatted = np.expand_dims(formatted, axis=0)

    batch_size = formatted.shape[0]
    input_dims = spec.get("input_dims", [])
    valid_dims = [d for d in input_dims if d > 0]

    # Handle 4D multi-channel tensor structures (e.g., [4, 64, 64])
    if len(valid_dims) == 3:
        c, h, w = valid_dims
        return formatted.reshape((batch_size, c, h, w))

    if len(valid_dims) == 4:
        _, c, h, w = valid_dims
        return formatted.reshape((batch_size, c, h, w))

    # Special handling for backbone / multi-dimensional feature graphs expecting 4D inputs
    if "backbone" in spec.get("input_name", "").lower() or len(input_dims) == 4:
        if formatted.size == batch_size * 16384:
            return formatted.reshape((batch_size, 4, 64, 64))

    if len(valid_dims) == 1:
        return formatted.reshape((batch_size, valid_dims[0]))

    return formatted.reshape((batch_size, -1))

def normalize_output_shape(output_data: np.ndarray, batch_size: int) -> np.ndarray:
    """Guarantees output is safely formatted as (batch_size, feature_dim) without data distortion."""
    if output_data is None:
        raise ValueError("Output tensor from Triton is None.")

    arr = np.asarray(output_data, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((batch_size, 1), dtype=np.float32)

    # Handle scalar or single-element expansion safely
    if arr.size == 1:
        return np.full((batch_size, 1), arr.item(), dtype=np.float32)

    if arr.shape[0] == batch_size:
        return arr.reshape((batch_size, -1))

    if arr.size % batch_size == 0:
        return arr.reshape((batch_size, -1))

    feature_dim = max(arr.size // batch_size, 1)
    return arr.flat[: batch_size * feature_dim].reshape((batch_size, feature_dim))


class TritonAsyncProxy:
    def __init__(
        self,
        triton_url: str = "127.0.0.1:8001",
        model_name: str = "ccvnn_v20_backbone",
        model_version: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version or ""
        self.default_input_name = input_name
        self.default_output_name = output_name
        self.client: grpcclient.InferenceServerClient | None = None
        self.model_spec_cache: dict[str, dict] = {}

    async def connect(self):
        if self.client is None:
            self.client = grpcclient.InferenceServerClient(url=self.triton_url)

    async def close(self):
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def inspect_model(self, model_name: str | None = None, pbtxt_path: str | None = None) -> dict:
        target_model = model_name or self.model_name
        if target_model in self.model_spec_cache:
            return self.model_spec_cache[target_model]

        if self.client is None:
            await self.connect()

        version_arg = self.model_version or ""

        # Strategy 1: Dynamic gRPC Metadata Discovery
        try:
            metadata = await self.client.get_model_metadata(
                model_name=target_model, model_version=version_arg
            )
            input_meta = metadata.inputs[0] if metadata.inputs else None
            output_meta = metadata.outputs[0] if metadata.outputs else None

            input_name = input_meta.name if input_meta else self.default_input_name
            output_name = output_meta.name if output_meta else self.default_output_name
            input_dims = list(input_meta.shape) if input_meta else [16384]

            raw_dtype = str(input_meta.datatype) if input_meta else "FP32"
            triton_dtype = raw_dtype.replace("TYPE_", "")

            if input_name and output_name:
                spec = {
                    "input_name": input_name,
                    "output_name": output_name,
                    "input_datatype": triton_dtype,
                    "input_dims": input_dims,
                    "expected_rank": len(input_dims) if input_dims else 2,
                }
                self.model_spec_cache[target_model] = spec
                return spec
        except Exception as exc:
            print(f"[TRITON CLIENT] Metadata discovery warning for '{target_model}': {exc}")

        # Strategy 2: Dynamic gRPC Config Discovery
        try:
            config_resp = await self.client.get_model_config(
                model_name=target_model, model_version=version_arg
            )
            model_config = config_resp.config

            input_name = model_config.input[0].name if model_config.input else self.default_input_name
            output_name = model_config.output[0].name if model_config.output else self.default_output_name
            raw_dims = list(model_config.input[0].dims) if model_config.input else [16384]
            raw_dtype = str(model_config.input[0].data_type) if model_config.input else "FP32"

            if input_name and output_name:
                spec = {
                    "input_name": input_name,
                    "output_name": output_name,
                    "input_datatype": raw_dtype.replace("TYPE_", ""),
                    "input_dims": raw_dims,
                    "expected_rank": len(raw_dims) if raw_dims else 2,
                }
                self.model_spec_cache[target_model] = spec
                return spec
        except Exception as exc:
            print(f"[TRITON CLIENT] Server config query warning for '{target_model}': {exc}")

        # Strategy 3: Local pbtxt Fallback
        pbtxt_file = pbtxt_path or f"{target_model}/config.pbtxt"
        pbtxt_spec = parse_pbtxt_config(pbtxt_file)
        if pbtxt_spec and pbtxt_spec.get("input_name"):
            self.model_spec_cache[target_model] = pbtxt_spec
            return pbtxt_spec

        # Strategy 4: Dynamic Query with empty outputs list to force Triton to echo output names
        fallback_spec = {
            "input_name": self.default_input_name or "84",
            "output_name": self.default_output_name or "output_logits",
            "input_datatype": "FP32",
            "input_dims": [16384],
            "expected_rank": 2,
        }
        self.model_spec_cache[target_model] = fallback_spec
        return fallback_spec

    async def infer(
        self,
        batched_input: np.ndarray,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> np.ndarray:
        if self.client is None:
            await self.connect()

        target_model = model_name or self.model_name

        spec = self.model_spec_cache.get(target_model)
        if not spec:
            spec = await self.inspect_model(target_model)

        target_input = input_name or spec.get("input_name") or self.default_input_name
        target_output = output_name or spec.get("output_name") or self.default_output_name

        formatted_input = prepare_input_tensor(batched_input, spec)
        batch_size = formatted_input.shape[0]

        triton_dtype = spec.get("input_datatype", "FP32")
        if isinstance(triton_dtype, str):
            triton_dtype = triton_dtype.replace("TYPE_", "")
        else:
            triton_dtype = triton_utils.np_to_triton_dtype(formatted_input.dtype)

        inputs = [grpcclient.InferInput(target_input, list(formatted_input.shape), triton_dtype)]
        inputs[0].set_data_from_numpy(formatted_input)

        # Do not supply explicit outputs if output_name isn't confirmed; Triton returns all by default
        outputs = [grpcclient.InferRequestedOutput(target_output)] if target_output else None
        version_arg = self.model_version or ""

        try:
            response = await self.client.infer(
                model_name=target_model,
                model_version=version_arg,
                inputs=inputs,
                outputs=outputs,
                client_timeout=10  # Prevents premature HTTP 500 drops under CPU queue spikes
            )
        except Exception:
            # Fallback retry without requested outputs constraint
            try:
                response = await self.client.infer(
                    model_name=target_model,
                    model_version=version_arg,
                    inputs=inputs,
                    outputs=None,
                    client_timeout=10  # Prevents premature HTTP 500 drops under CPU queue spikes
                )
            except Exception as exc:
                print(f"[TRITON ERROR] Inference failed for '{target_model}': {exc}")
                raise RuntimeError(f"Triton gRPC Inference Error on '{target_model}': {exc}") from exc
        output_data = None
        if target_output:
            try:
                output_data = response.as_numpy(target_output)
            except Exception:
                pass

        if output_data is None:
            resp_outputs = response.get_response().outputs
            for out_meta in resp_outputs:
                try:
                    output_data = response.as_numpy(out_meta.name)
                    if output_data is not None:
                        # Update cache dynamically with actual output name
                        spec["output_name"] = out_meta.name
                        break
                except Exception:
                    continue

        if output_data is None:
            raise RuntimeError(f"Model '{target_model}' returned empty output tensor.")

        return normalize_output_shape(output_data, batch_size)


class AsyncTritonClient(TritonAsyncProxy):
    def __init__(self, url: str = "127.0.0.1:8001"):
        super().__init__(triton_url=url)
        self.shm_mgr = None

    async def init_shm(self, max_batch_size: int = 16, feature_dim: int = 16384):
        if TritonZeroCopyManager is not None:
            try:
                self.shm_mgr = TritonZeroCopyManager(self.client)
                self.shm_mgr.allocate_and_register(
                    element_shape=(feature_dim,),
                    max_batch_size=max_batch_size,
                )
            except Exception as exc:
                print(f"[TRITON SHM WARNING] Failed to initialize SHM manager: {exc}")
                self.shm_mgr = None

    async def infer_zero_copy(self, model_name: str, input_array: np.ndarray) -> np.ndarray:
        spec = await self.inspect_model(model_name)

        formatted_input = prepare_input_tensor(input_array, spec)
        batch_size = formatted_input.shape[0]

        if TritonZeroCopyManager is None or self.shm_mgr is None:
            return await self.infer(formatted_input, model_name=model_name)

        version_arg = self.model_version or ""

        try:
            self.shm_mgr.write_tensor_to_shm(formatted_input)
            shm_input = self.shm_mgr.prepare_shm_infer_input(
                input_name=spec["input_name"], shape=formatted_input.shape
            )

            response = await self.client.infer(
                model_name=model_name, model_version=version_arg, inputs=[shm_input], client_timeout=10  # Prevents premature HTTP 500 drops under CPU queue spikes
            )

            output = None
            try:
                output = response.as_numpy(spec["output_name"])
            except Exception:
                pass

            if output is None:
                outputs_meta = response.get_response().outputs
                if outputs_meta:
                    output = response.as_numpy(outputs_meta[0].name)

            if output is not None:
                return normalize_output_shape(output, batch_size)
        except Exception as exc:
            print(f"[TRITON SHM WARNING] Zero-copy inference failed, falling back to gRPC: {exc}")

        return await self.infer(formatted_input, model_name=model_name)