#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include "gstnvdsmeta.h"
#include "gstnvinfer_meta.h"

// Pad probe callback to intercept nvinferserver output metadata
static GstPadProbeReturn
nvinferserver_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user->data);
            
            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                NvDsInferTensorOutputMeta *tensor_meta = (NvDsInferTensorOutputMeta *)user_meta->user_meta_data;
                std::cout << "[DeepStream C++] Frame #" << frame_meta->frame_num 
                          << " | Infer Output Layers: " << tensor_meta->numOutputLayers 
                          << " | Layer Name: " << tensor_meta->outputLayersInfo[0].layerName << std::endl;
            }
        }
    }
    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *src = NULL, *convert = NULL, *caps = NULL, *nvinfer = NULL, *sink = NULL;
    GstCaps *caps_filter = NULL;
    GstPad *infer_src_pad = NULL;

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    pipeline = gst_pipeline_new("ccvnn-deepstream-pipeline");
    src = gst_element_factory_make("videotestsrc", "src");
    convert = gst_element_factory_make("nvvideoconvert", "convert");
    caps = gst_element_factory_make("capsfilter", "caps");
    nvinfer = gst_element_factory_make("nvinferserver", "nvinferserver");
    sink = gst_element_factory_make("fakesink", "sink");

    if (!pipeline || !src || !convert || !caps || !nvinfer || !sink) {
        std::cerr << "Error: Failed to create GStreamer pipeline elements." << std::endl;
        return -1;
    }

    // Configure video stream source
    g_object_set(G_OBJECT(src), "num-buffers", 50, NULL);
    caps_filter = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12, width=1280, height=720, framerate=30/1");
    g_object_set(G_OBJECT(caps), "caps", caps_filter, NULL);
    gst_caps_unref(caps_filter);

    // Link nvinferserver to declarative configuration file
    //g_object_set(G_OBJECT(nvinfer), "config-file-path", "config_infer_server_ccvnn.txt", NULL);
	// 1. Determine config path from CLI argument or default to relu6
    const char *config_file = (argc > 1) ? argv[1] : "config_infer_server_relu6.txt";

    // 2. Pass the dynamic config file path to nvinferserver
    g_object_set(G_OBJECT(nvinfer), "config-file-path", config_file, NULL);

    gst_bin_add_many(GST_BIN(pipeline), src, convert, caps, nvinfer, sink, NULL);
    if (!gst_element_link_many(src, convert, caps, nvinfer, sink, NULL)) {
        std::cerr << "Error: Failed to link pipeline elements." << std::endl;
        return -1;
    }

    // Attach pad probe on nvinferserver src pad
    infer_src_pad = gst_element_get_static_pad(nvinfer, "src");
    gst_pad_add_probe(infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER, nvinferserver_src_pad_buffer_probe, NULL, NULL);
    gst_object_unref(infer_src_pad);

    std::cout << "--- Starting DeepStream C++ Pipeline Harness (nvinferserver) ---" << std::endl;
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_main_loop_unref(loop);

    return 0;
}