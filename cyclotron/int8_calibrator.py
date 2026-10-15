import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
from torch.utils.data import DataLoader

class DualModeEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator for multi-input TensorRT engines (2D Image + 1D Telemetry).
    Computes activation quantization dynamic ranges using KL-Divergence minimization.
    """
    def __init__(self, data_loader: DataLoader, cache_file: str = "dual_mode_int8.cache"):
        super().__init__()
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.cache_file = cache_file
        self.batch_size = data_loader.batch_size

        # Retrieve first batch to determine memory dimensions
        sample_img, sample_vcum = next(iter(data_loader))
        self.img_bytes = sample_img.numpy().nbytes
        self.vcum_bytes = sample_vcum.numpy().nbytes

        # Allocate page-locked host memory & device memory for bindings
        self.d_image = cuda.mem_alloc(self.img_bytes)
        self.d_vcum = cuda.mem_alloc(self.vcum_bytes)

        self.current_index = 0

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        """
        Provides GPU memory device pointers for input tensor bindings.
        TensorRT calls this method iteratively until returning None.
        """
        try:
            images, vcum = next(self.data_iter)
        except StopIteration:
            return None

        # Ensure contiguous FP32 layout
        img_np = np.ascontiguousarray(images.numpy(), dtype=np.float32)
        vcum_np = np.ascontiguousarray(vcum.numpy(), dtype=np.float32)

        # Copy calibration batch from Host RAM to CUDA Device VRAM
        cuda.memcpy_htod(self.d_image, img_np)
        cuda.memcpy_htod(self.d_vcum, vcum_np)

        self.current_index += 1
        print(f"[Calibrator] Processing batch {self.current_index}/{len(self.data_loader)}")

        # Return list of raw GPU device memory pointers matching ONNX input order
        return [int(self.d_image), int(self.d_vcum)]

    def read_calibration_cache(self) -> bytes | None:
        """Reads existing calibration scale factors if available to skip re-calibration."""
        if os.path.exists(self.cache_file):
            print(f"[Calibrator] Loading cached quantization scales from: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persists calculated INT8 dynamic range scales to disk."""
        print(f"[Calibrator] Writing calibration cache to: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(onnx_path: str, engine_path: str, calibrator: DualModeEntropyCalibrator) -> None:
    """
    Parses ONNX model, applies INT8 dynamic range quantization using the calibrator,
    and serializes the compiled engine to disk.
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    print(f"[Builder] Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"[ONNX Error] {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    
    # Enable INT8 and FP16 fallback execution modes
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = calibrator

    print("[Builder] Building TensorRT Engine with INT8 Quantization (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT serialized engine.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"[Builder] TensorRT INT8 Engine successfully exported to: {engine_path}")


# --- Execution Example ---
if __name__ == "__main__":
    # Synthetic calibration dataset (Representative sample of 100 operational frames)
    class CalibrationDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            # 2D Normalised RGB Frame [3, 256, 256]
            frame = torch.randn(3, 256, 256, dtype=torch.float32)
            # 1D Telemetry Vector V_cum [8]
            vcum = torch.tensor([0.2, 0.15, 0.0, 0.0, 0.0, 320.0, 1.0, 2048.0], dtype=torch.float32)
            return frame, vcum

    dataset = CalibrationDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    calibrator = DualModeEntropyCalibrator(loader, cache_file="dual_mode_int8.cache")
    build_int8_engine("models/dual_mode_vision.onnx", "models/dual_mode_vision_int8.engine", calibrator)