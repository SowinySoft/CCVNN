#include <iostream>
#include <vector>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvds_osd_meta.h"

// Define feature vector dimensions matching ONNX/Triton configuration
constexpr int HARDSWISH_12D_DIM = 12;
constexpr int RELU6_9D_DIM = 9;

extern "C" GstPadProbeReturn ccvnn_meta_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            
            // Extract bounding box dimensions for feature normalization
            float width = obj_meta->rect_params.width;
            float height = obj_meta->rect_params.height;
            float top = obj_meta->rect_params.top;
            float left = obj_meta->rect_params.left;

            // Generate 12D feature vector for ccvnn_hardswish_12d
            std::vector<float> feature_12d = {
                left, top, width, height,
                width / (height + 1e-5f),
                top + height / 2.0f,
                left + width / 2.0f,
                width * height,
                0.5f, 0.1f, 0.0f, 1.0f
            };

            // Attach vector pointer to User Meta payload
            NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
            user_meta->user_meta_data = (void *)feature_12d.data();
            user_meta->base_meta.meta_type = (NvDsMetaType)NVDS_USER_META;
            nvds_add_user_meta_to_obj(obj_meta, user_meta);
        }
    }

    return GST_PAD_PROBE_OK;
}