CXX ?= g++
CFLAGS := -Wall -Werror -O3 -fPIC

CUDA_VER ?= 12.0
DEEPSTREAM_DIR ?= /opt/nvidia/deepstream/deepstream

INCS := -I$(DEEPSTREAM_DIR)/sources/includes \
        -I/usr/local/cuda/include \
        $(shell pkg-config --cflags gstreamer-1.0 gstreamer-video-1.0 2>/dev/null)

LIBS := $(shell pkg-config --libs gstreamer-1.0 gstreamer-video-1.0 2>/dev/null) \
        -L/usr/local/cuda/lib64 -lcudart \
        -L$(DEEPSTREAM_DIR)/lib -lnvds_meta -lnvds_osd

TARGET := libccvnn_ds_probe.so
SRC := ccvnn_ds_probe.cpp
OBJ := $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -shared -o $@ $(OBJ) $(LIBS)
	@echo "Build successful: $(TARGET)"

%.o: %.cpp
	$(CXX) $(CFLAGS) $(INCS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
