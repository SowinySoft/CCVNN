import asyncio
import grpc
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tritonclient.grpc import service_pb2, service_pb2_grpc

workspace = Path(__file__).parent

all_onnx = list(workspace.rglob("*.onnx"))
hardswish_file = next(f for f in all_onnx if "hardswish" in str(f).lower())
relu6_file = next(f for f in all_onnx if "relu6" in str(f).lower())

class TritonServerMock(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self):
        sess_hardswish = ort.InferenceSession(str(hardswish_file))
        sess_relu6 = ort.InferenceSession(str(relu6_file))
        
        self.sessions = {
            "ccvnn_relu6": sess_relu6,
            "ccvnn_relu6_9d": sess_relu6,
            "ccvnn_hardswish": sess_hardswish,
            "ccvnn_hardswish_12d": sess_hardswish,
        }
        print(f"[Mock Triton] Loaded ReLU6: {relu6_file.name}")
        print(f"[Mock Triton] Loaded HardSwish: {hardswish_file.name}")

    async def ServerReady(self, request, context):
        return service_pb2.ServerReadyResponse(ready=True)

    async def ModelReady(self, request, context):
        return service_pb2.ModelReadyResponse(ready=True)

    async def ModelInfer(self, request, context):
        model_name = request.model_name
        if model_name not in self.sessions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{model_name}' not loaded.")
            return service_pb2.ModelInferResponse()

        session = self.sessions[model_name]

        if request.raw_input_contents:
            raw_bytes = request.raw_input_contents[0]
            dim = 9 if "relu6" in model_name else 12
            input_array = np.frombuffer(raw_bytes, dtype=np.float32).reshape(1, dim)
        else:
            contents = request.inputs[0].contents
            input_array = np.array(contents.fp32_contents, dtype=np.float32).reshape(1, -1)

        output_name = session.get_outputs()[0].name
        input_name = session.get_inputs()[0].name
        res = session.run([output_name], {input_name: input_array})[0]
        
        # Standardize array to float32 and capture actual model output shape
        res_fp32 = np.array(res, dtype=np.float32)
        out_name = request.outputs[0].name if request.outputs else "output_tensor"

        response = service_pb2.ModelInferResponse(model_name=model_name)
        response.outputs.add(
            name=out_name,
            datatype="FP32",
            shape=list(res_fp32.shape)
        )
        response.raw_output_contents.append(res_fp32.tobytes())
        return response

async def serve():
    server = grpc.aio.server()
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(TritonServerMock(), server)
    server.add_insecure_port("127.0.0.1:8001")
    print("--- Mock Triton gRPC Server Listening on 127.0.0.1:8001 ---")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())