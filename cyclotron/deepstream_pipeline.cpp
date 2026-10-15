#include <gst/gst.h>
#include <iostream>

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    // Build GStreamer Pipeline String
    std::string pipeline_str = 
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=1920, height=1080, format=YUY2, framerate=30/1 ! "
        "nvvideoconvert nvbuf-memory-type=0 ! "
        "video/x-raw(memory:NVMM), format=RGBA, width=256, height=256 ! "
        "dsdualmode name=dual_mode_node ! "
        "fakesink sync=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipeline_str.c_str(), &error);

    if (error) {
        std::cerr << "[ERROR] Pipeline Launch Failed: " << error->message << std::endl;
        g_clear_error(&error);
        return 1;
    }

    std::cout << "[INFO] DeepStream Dual-Mode Pipeline running..." << std::endl;
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, 
                        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (msg) gst_message_unref(msg);
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}