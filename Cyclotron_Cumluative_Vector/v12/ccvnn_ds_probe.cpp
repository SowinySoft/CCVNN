#include <iostream>
#include <vector>
#include <gst/gst.h>

constexpr int HARDSWISH_12D_DIM = 12;
constexpr int RELU6_9D_DIM = 9;

extern "C" GstPadProbeReturn ccvnn_meta_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) {
        return GST_PAD_PROBE_OK;
    }

    std::vector<float> feature_12d = {
        0.0f, 0.0f, 1920.0f, 1080.0f,
        1.777f, 540.0f, 960.0f,
        2073600.0f, 0.5f, 0.1f, 0.0f, 1.0f
    };

    return GST_PAD_PROBE_OK;
}