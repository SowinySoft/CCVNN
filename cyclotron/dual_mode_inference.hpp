#ifndef DUAL_MODE_INFERENCE_HPP
#define DUAL_MODE_INFERENCE_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace ccvnn {

// Custom TensorRT Logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT " << static_cast<int>(severity) << "] " << msg << std::endl;
        }
    }
};

// Input/Output Tensor Specifications
struct EngineDimensions {
    static constexpr size_t BATCH_SIZE     = 1;
    static constexpr size_t IMAGE_CHANNELS = 3;
    static constexpr size_t IMAGE_HEIGHT   = 256;
    static constexpr size_t IMAGE_WIDTH    = 256;
    static constexpr size_t V_CUM_SIZE     = 8;
    static constexpr size_t LOGITS_SIZE    = 4;

    static constexpr size_t IMAGE_BYTES  = BATCH_SIZE * IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float);
    static constexpr size_t V_CUM_BYTES  = BATCH_SIZE * V_CUM_SIZE * sizeof(float);
    static constexpr size_t LOGITS_BYTES = BATCH_SIZE * LOGITS_SIZE * sizeof(float);
};

class DualModeTRTEngine {
public:
    explicit DualModeTRTEngine(const std::string& engine_path) {
        init_cuda_resources();
        load_engine(engine_path);
        setup_zero_copy_buffers();
    }

    ~DualModeTRTEngine() {
        // Free zero-copy pinned memory
        if (host_image_ptr_) cudaFreeHost(host_image_ptr_);
        if (host_vcum_ptr_)  cudaFreeHost(host_vcum_ptr_);
        if (host_logits_ptr_) cudaFreeHost(host_logits_ptr_);

        if (stream_) cudaStreamDestroy(stream_);
    }

    // Direct GPU Device Pointer Inference (Zero CPU-GPU copy)
    // Used when preprocessing output is already residing on GPU VRAM (e.g., NVMM / NPP CUDA surface)
    bool infer_device_inputs(const float* d_frame_ptr, const float* d_vcum_ptr, float* h_out_logits) {
        // Assign device memory addresses to TensorRT bindings
        bindings_[0] = const_cast<float*>(d_frame_ptr);
        bindings_[1] = const_cast<float*>(d_vcum_ptr);
        bindings_[2] = dev_logits_ptr_;

        // Asynchronous Execution on dedicated CUDA Stream
        bool status = context_->enqueueV2(bindings_.data(), stream_, nullptr);
        if (!status) return false;

        // Async memcpy for classification output vector
        cudaMemcpyAsync(h_out_logits, dev_logits_ptr_, EngineDimensions::LOGITS_BYTES, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        return true;
    }

    // Unified Zero-Copy Memory Inference (Host-Mapped Pinned Memory)
    // Allows CPU code to write frame and V_cum directly into mapped memory regions shared with GPU
    bool infer_zero_copy(float* out_logits) {
        // Bind mapped device pointers directly
        bindings_[0] = dev_image_ptr_;
        bindings_[1] = dev_vcum_ptr_;
        bindings_[2] = dev_logits_ptr_;

        // Execute TensorRT graph directly on mapped memory addresses
        bool status = context_->enqueueV2(bindings_.data(), stream_, nullptr);
        if (!status) return false;

        cudaStreamSynchronize(stream_);

        // Copy outputs from pinned memory region
        std::memcpy(out_logits, host_logits_ptr_, EngineDimensions::LOGITS_BYTES);
        return true;
    }

    // Accessors for zero-copy host pointers (Zero-Allocation Writes)
    float* get_host_image_buffer() const noexcept { return host_image_ptr_; }
    float* get_host_vcum_buffer()  const noexcept { return host_vcum_ptr_; }

private:
    void init_cuda_resources() {
        cudaStreamCreate(&stream_);
    }

    void load_engine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open TensorRT engine file: " + engine_path);
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
        engine_  = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), size));
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

        bindings_.resize(3); // 2 Inputs ("frame", "v_cum"), 1 Output ("logits")
    }

    void setup_zero_copy_buffers() {
        // Allocate zero-copy mapped host memory for Image Input
        cudaHostAlloc(&host_image_ptr_, EngineDimensions::IMAGE_BYTES, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&dev_image_ptr_, host_image_ptr_, 0);

        // Allocate zero-copy mapped host memory for V_cum Input
        cudaHostAlloc(&host_vcum_ptr_, EngineDimensions::V_CUM_BYTES, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&dev_vcum_ptr_, host_vcum_ptr_, 0);

        // Allocate zero-copy mapped host memory for Logits Output
        cudaHostAlloc(&host_logits_ptr_, EngineDimensions::LOGITS_BYTES, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&dev_logits_ptr_, host_logits_ptr_, 0);
    }

    TRTLogger logger_;
    cudaStream_t stream_{nullptr};

    std::unique_ptr<nvinfer1::IRuntime>          runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    std::vector<void*> bindings_;

    // Pinned Host Pointers
    float* host_image_ptr_{nullptr};
    float* host_vcum_ptr_{nullptr};
    float* host_logits_ptr_{nullptr};

    // Mapped Device Pointers (Zero-Copy)
    float* dev_image_ptr_{nullptr};
    float* dev_vcum_ptr_{nullptr};
    float* dev_logits_ptr_{nullptr};
};

} // namespace ccvnn

#endif // DUAL_MODE_INFERENCE_HPP