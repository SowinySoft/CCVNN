import mmap
import os
import struct
from typing import Tuple
import numpy as np
import tritonclient.grpc as grpcclient

# Graceful import for Triton Shared Memory extension
try:
    import tritonclient.grpc.shm as triton_shm
    HAS_TRITON_SHM = True
except ModuleNotFoundError:
    from multiprocessing import shared_memory
    triton_shm = None
    HAS_TRITON_SHM = False


class TritonZeroCopyManager:
    """Manages POSIX shared memory regions for zero-copy Triton gRPC inference,
    with automatic fallback to Python's native multiprocessing.shared_memory
    when tritonclient.grpc.shm is unavailable.
    """

    def __init__(self, triton_client: grpcclient.InferenceServerClient, shm_key: str = "/ccvnn_shm_region"):
        self.client = triton_client
        self.shm_key = shm_key
        self.shm_handle = None
        self.native_shm = None
        self.shm_size = 0
        self.shm_name = "shm_input_region"

    def allocate_and_register(self, element_shape: Tuple[int, ...], max_batch_size: int = 16) -> None:
        """Allocates POSIX shared memory buffer and registers it with the Triton server."""
        element_size = int(np.prod(element_shape)) * np.dtype(np.float32).itemsize
        self.shm_size = element_size * max_batch_size

        # Unregister existing handle if present on Triton server
        try:
            self.client.unregister_system_shared_memory(self.shm_name)
        except Exception:
            pass

        if HAS_TRITON_SHM:
            self.shm_handle = triton_shm.create_shared_memory_region(
                shm_name=self.shm_name,
                shm_key=self.shm_key,
                byte_size=self.shm_size,
            )
            self.client.register_system_shared_memory(
                name=self.shm_name,
                key=self.shm_key,
                byte_size=self.shm_size,
            )
            print(f"[ZERO-COPY] Registered {self.shm_size} bytes POSIX SHM via Triton C-extension.")
        else:
            posix_name = self.shm_key.lstrip("/")

            # Clean up previous local shared memory buffer if allocated
            if self.native_shm is not None:
                try:
                    self.native_shm.close()
                    self.native_shm.unlink()
                except Exception:
                    pass
                self.native_shm = None
            else:
                try:
                    stale_shm = shared_memory.SharedMemory(name=posix_name)
                    stale_shm.close()
                    stale_shm.unlink()
                except FileNotFoundError:
                    pass

            self.native_shm = shared_memory.SharedMemory(name=posix_name, create=True, size=self.shm_size)

            self.client.register_system_shared_memory(
                name=self.shm_name,
                key=self.shm_key,
                byte_size=self.shm_size,
            )
            print(f"[ZERO-COPY] Registered {self.shm_size} bytes POSIX SHM via multiprocessing.shared_memory.")

    def write_tensor_to_shm(self, input_array: np.ndarray, offset: int = 0) -> None:
        """Direct memcpy into allocated shared memory address space."""
        if HAS_TRITON_SHM:
            if not self.shm_handle:
                raise RuntimeError("Shared memory region is not allocated.")
            triton_shm.set_shared_memory_region(
                self.shm_handle,
                [input_array.astype(np.float32)],
                offset=offset,
            )
        else:
            if self.native_shm is None:
                raise RuntimeError("Shared memory region is not allocated.")
            data = input_array.astype(np.float32)
            dest_arr = np.ndarray(data.shape, dtype=np.float32, buffer=self.native_shm.buf, offset=offset)
            dest_arr[...] = data

    def prepare_shm_infer_input(self, input_name: str, shape: Tuple[int, ...]) -> grpcclient.InferInput:
        """Constructs gRPC InferInput structure referencing registered SHM buffer."""
        infer_input = grpcclient.InferInput(input_name, list(shape), "FP32")
        infer_input.set_shared_memory(self.shm_name, self.shm_size)
        return infer_input

    def cleanup(self) -> None:
        """Unregisters and frees system shared memory handles."""
        try:
            self.client.unregister_system_shared_memory(self.shm_name)
        except Exception:
            pass

        if HAS_TRITON_SHM and self.shm_handle:
            triton_shm.destroy_shared_memory_region(self.shm_handle)
            self.shm_handle = None
        elif self.native_shm is not None:
            try:
                self.native_shm.close()
                self.native_shm.unlink()
            except Exception:
                pass
            self.native_shm = None

        print("[ZERO-COPY] Unregistered and destroyed SHM regions.")