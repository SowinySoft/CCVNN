# Compile shared plugin object
g++ -shared -fPIC -O3 \
  gstdsdualmode.cpp \
  -o libnvds_dualmode.so \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-base-1.0) \
  -I/opt/nvidia/deepstream/deepstream/sources/includes \
  -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 -lcudart -lnvinfer

# Copy library to DeepStream plugin directory
sudo cp libnvds_dualmode.so /opt/nvidia/deepstream/deepstream/lib/gst-plugins/