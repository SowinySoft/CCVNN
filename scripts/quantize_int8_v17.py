#!/usr/bin/env python3
"""
CCVNN V17 Post-Training Quantization (PTQ) Calibration Engine
Architectural Path: ccvnn_v17_backbone.onnx -> INT8 Calibration -> ccvnn_v17_backbone.engine / ccvnn_v17_backbone_int8.onnx
"""

import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ONNX_MODEL_PATH = "ccvnn_v17_backbone.onnx" if os.path.exists("ccvnn_v17_backbone.onnx") else "scripts/ccvnn_v17_backbone.onnx"
CALIBRATION_CACHE_PATH = "scripts/ccvnn_v17.calibration.cache"
ENGINE_OUTPUT_PATH = "ccvnn_v17_backbone.engine"
INT8_ONNX_OUTPUT_PATH = "ccvnn_v17_backbone_int8.onnx"

NUM_CALIBRATION_BATCHES = 100
BATCH_SIZE = 32
VECTOR_DIM = 17

def generate_calibration_data(num_batches=100, batch_size=32) -> np.ndarray:
    """
    Generates representative synthetic 17-element calibration data matching physical bounds.
    """
    np.random.seed(42)
    calibration_dataset = []

    for _ in range(num_batches):
        batch = np.random.uniform(0.0, 1.0, size=(batch_size, VECTOR_DIM)).astype(np.float32)
        # Domain bounds for environmental vector extensions (e12 - e16)
        batch[:, 12] = np.random.uniform(-180.0, 180.0, size=(batch_size,)) # e12: theta_rot
        batch[:, 14] = np.random.uniform(0.05, 0.95, size=(batch_size,))    # e14: b_lux
        batch[:, 15] = np.random.uniform(0.10, 0.99, size=(batch_size,))    # e15: c_ratio
        batch[:, 16] = np.random.exponential(scale=0.02, size=(batch_size,)) # e16: r_specular
        batch[:, 16] = np.clip(batch[:, 16], 0.0, 1.0)
        calibration_dataset.append(batch)

    return np.array(calibration_dataset)


def run_tensorrt_ptq():
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    class CCVNNEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, calibration_data, cache_file):
            super().__init__()
            self.data = calibration_data
            self.cache_file = cache_file
            self.batch_size = calibration_data.shape[1]
            self.current_index = 0

            import pycuda.driver as cuda
            import pycuda.autoinit
            self.device_input = cuda.mem_alloc(self.data[0].nbytes)

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.current_index < len(self.data):
                import pycuda.driver as cuda
                batch = np.ascontiguousarray(self.data[self.current_index])
                cuda.memcpy_htod(self.device_input, batch)
                self.current_index += 1
                return [int(self.device_input)]
            return None

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)

    logging.info(f"Building TensorRT INT8 Engine from {ONNX_MODEL_PATH} via IInt8EntropyCalibrator2...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_MODEL_PATH, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logging.error(f"ONNX Parse Error: {parser.get_error(error)}")
            return False

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    
    calib_data = generate_calibration_data(NUM_CALIBRATION_BATCHES, BATCH_SIZE)
    calibrator = CCVNNEntropyCalibrator(calib_data, CALIBRATION_CACHE_PATH)
    config.int8_calibrator = calibrator

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is not None:
        with open(ENGINE_OUTPUT_PATH, "wb") as f:
            f.write(serialized_engine)
        logging.info(f"✅ TensorRT INT8 Engine compiled successfully: {ENGINE_OUTPUT_PATH}")
        return True
    else:
        logging.error("TensorRT Engine build failed!")
        return False


def run_onnxruntime_ptq():
    logging.info("TensorRT not detected. Falling back to ONNX Runtime INT8 Static Quantization...")
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

    class CCVNNDataReader(CalibrationDataReader):
        def __init__(self, calibration_data):
            self.data = calibration_data
            self.enum_data = None
            self.rewind()

        def get_next(self):
            if self.enum_data is None:
                return None
            # Fix input key matching exported ONNX graph input: v_input_17
            return next(self.enum_data, None)

        def rewind(self):
            self.enum_data = iter([{"v_input_17": batch} for batch in self.data])

    calib_data = generate_calibration_data(NUM_CALIBRATION_BATCHES, BATCH_SIZE)
    data_reader = CCVNNDataReader(calib_data)

    quantize_static(
        model_input=ONNX_MODEL_PATH,
        model_output=INT8_ONNX_OUTPUT_PATH,
        calibration_data_reader=data_reader,
        quant_format=QuantType.QInt8
    )
    logging.info(f"✅ ONNX Runtime Static INT8 Model generated: {INT8_ONNX_OUTPUT_PATH}")
    return True


if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        logging.error(f"ONNX Model not found at {ONNX_MODEL_PATH}. Run scripts/export_onnx_v17.py first.")
        sys.exit(1)

    try:
        import tensorrt
        import pycuda
        success = run_tensorrt_ptq()
    except ImportError:
        success = run_onnxruntime_ptq()

    if success:
        logging.info("Phase 3.3 INT8 PTQ Quantization completed successfully.")
    else:
        sys.exit(1)
