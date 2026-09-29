#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvdsmeta.h"
#include "dual_mode_inference.hpp"

#define PACKAGE "dsdualmode"
#define VERSION "1.0.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "DeepStream Zero-Copy Dual-Mode TRT Pipeline"

typedef struct _GstDsDualMode {
    GstBaseTransform element;
    ccvnn::DualModeTRTEngine* trt_engine;
    float current_vcum[8];
    GstMutex vcum_lock;
} GstDsDualMode;

typedef struct _GstDsDualModeClass {
    GstBaseTransformClass parent_class;
} GstDsDualModeClass;

G_DEFINE_TYPE(GstDsDualMode, gst_ds_dual_mode, GST_TYPE_BASE_TRANSFORM);

static GstFlowReturn gst_ds_dual_mode_transform_ip(GstBaseTransform* trans, GstBuffer* buf) {
    GstDsDualMode* self = (GstDsDualMode*)trans;

    // 1. Retrieve NvBufSurface metadata container from GstBuffer
    GstMapInfo in_map_info;
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) {
        GST_ELEMENT_ERROR(trans, STREAM, FAILED, ("Failed to map GstBuffer"), (NULL));
        return GST_FLOW_ERROR;
    }

    NvBufSurface* surface = (NvBufSurface*)in_map_info.data;
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!surface || surface->numFilled == 0) {
        gst_buffer_unmap(buf, &in_map_info);
        return GST_FLOW_OK;
    }

    // 2. Extract GPU Device Pointer directly from NVMM Surface (Zero CPU Copy)
    NvBufSurfaceParams* frame_params = &surface->surfaceList[0];
    float* d_frame_ptr = static_cast<float*>(frame_params->dataPtr);

    // Copy local snapshot of atomic V_cum vector
    float host_vcum[8];
    g_mutex_lock(&self->vcum_lock);
    std::memcpy(host_vcum, self->current_vcum, sizeof(host_vcum));
    g_mutex_unlock(&self->vcum_lock);

    // 3. Allocate device memory for V_cum vector input binding
    float* d_vcum_ptr = nullptr;
    cudaMallocAsync(&d_vcum_ptr, sizeof(host_vcum), nullptr);
    cudaMemcpyAsync(d_vcum_ptr, host_vcum, sizeof(host_vcum), cudaMemcpyHostToDevice, nullptr);

    // 4. Run Zero-Copy TensorRT Inference
    float output_logits[4] = {0.0f};
    if (self->trt_engine) {
        self->trt_engine->infer_device_inputs(d_frame_ptr, d_vcum_ptr, output_logits);
    }

    cudaFreeAsync(d_vcum_ptr, nullptr);

    // 5. Attach Dual-Mode Classification Meta to DeepStream Frame
    if (batch_meta && batch_meta->num_frames_in_batch > 0) {
        NvDsFrameMeta* frame_meta = nvds_get_nth_frame_meta(batch_meta->frame_meta_list, 0);
        
        // Allocate User Meta container
        NvDsUserMeta* user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
        if (user_meta) {
            float* logits_copy = static_cast<float*>(g_malloc(sizeof(output_logits)));
            std::memcpy(logits_copy, output_logits, sizeof(output_logits));

            user_meta->user_meta_data = logits_copy;
            user_meta->base_meta.meta_type = (NvDsMetaType)NVDS_USER_META;
            nvds_add_user_meta_to_frame(frame_meta, user_meta);
        }
    }

    gst_buffer_unmap(buf, &in_map_info);
    return GST_FLOW_OK;
}

static void gst_ds_dual_mode_init(GstDsDualMode* self) {
    g_mutex_init(&self->vcum_lock);
    self->trt_engine = new ccvnn::DualModeTRTEngine("models/dual_mode_vision_int8.engine");
    
    // Default V_cum initialization
    std::fill_n(self->current_vcum, 8, 0.0f);
    self->current_vcum[6] = 1.0f; // H_health
}

static void gst_ds_dual_mode_class_init(GstDsDualModeClass* klass) {
    GstBaseTransformClass* base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);
    base_transform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_ds_dual_mode_transform_ip);
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "dsdualmode", GST_RANK_NONE, gst_ds_dual_mode_get_type());
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    dsdualmode,
    DESCRIPTION,
    plugin_init,
    VERSION,
    LICENSE,
    PACKAGE,
    "https://ccvnn.ai"
)