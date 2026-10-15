#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"

static GstPadProbeReturn
nvinferserver_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
                NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
                if (tensor_meta) {
                    std::cout << "[Probe] Processed tensor meta with " 
                              << tensor_meta->num_output_layers << " output layers." << std::endl;
                }
            }
        }
    }
    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>" << std::endl;
        return -1;
    }

    gst_init(&argc, &argv);
    GMainLoop *loop = g_main_loop_new(NULL, FALSE);

    GstElement *pipeline = gst_pipeline_new("deepstream-ccvnn-pipeline");
    GstElement *source = gst_element_factory_make("videotestsrc", "source");
    GstElement *caps_filter = gst_element_factory_make("capsfilter", "capsfilter");
    GstElement *nvvideoconvert = gst_element_factory_make("nvvideoconvert", "nvvideoconvert");
    GstElement *nvstreammux = gst_element_factory_make("nvstreammux", "streammux");
    GstElement *nvinferserver = gst_element_factory_make("nvinferserver", "nvinferserver");
    GstElement *fakesink = gst_element_factory_make("fakesink", "fakesink");

    #define CHECK_ELEM(elem, name) if (!elem) { std::cerr << "Failed to create element: " << name << std::endl; return -1; }
    CHECK_ELEM(pipeline, "pipeline");
    CHECK_ELEM(source, "videotestsrc");
    CHECK_ELEM(caps_filter, "capsfilter");
    CHECK_ELEM(nvvideoconvert, "nvvideoconvert");
    CHECK_ELEM(nvstreammux, "nvstreammux");
    CHECK_ELEM(nvinferserver, "nvinferserver");
    CHECK_ELEM(fakesink, "fakesink");

    g_object_set(G_OBJECT(source), "num-buffers", 50, NULL);
    g_object_set(G_OBJECT(nvstreammux), "batch-size", 1, "width", 1920, "height", 1080, NULL);
    g_object_set(G_OBJECT(nvinferserver), "config-file-path", argv[1], NULL);

    gst_bin_add_many(GST_BIN(pipeline), source, caps_filter, nvvideoconvert, nvstreammux, nvinferserver, fakesink, NULL);

    GstCaps *caps = gst_caps_from_string("video/x-raw, format=NV12, width=1920, height=1080");
    g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);
    gst_caps_unref(caps);

    if (!gst_element_link(source, caps_filter) ||
        !gst_element_link(caps_filter, nvvideoconvert)) {
        std::cerr << "Failed to link source pipeline." << std::endl;
        return -1;
    }

    GstPad *sinkpad = gst_element_request_pad_simple(nvstreammux, "sink_0");
    GstPad *srcpad = gst_element_get_static_pad(nvvideoconvert, "src");
    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        std::cerr << "Failed to link nvvideoconvert to nvstreammux." << std::endl;
        return -1;
    }
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);

    if (!gst_element_link_many(nvstreammux, nvinferserver, fakesink, NULL)) {
        std::cerr << "Failed to link nvstreammux to fakesink." << std::endl;
        return -1;
    }

    GstPad *infer_src_pad = gst_element_get_static_pad(nvinferserver, "src");
    if (infer_src_pad) {
        gst_pad_add_probe(infer_src_pad, GST_PAD_PROBE_TYPE_BUFFER, nvinferserver_src_pad_buffer_probe, NULL, NULL);
        gst_object_unref(infer_src_pad);
    }

    std::cout << "Starting pipeline with config: " << argv[1] << std::endl;
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_main_loop_unref(loop);

    return 0;
}