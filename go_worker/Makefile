CXX := g++
CXXFLAGS := -O2 -std=c++17 -fPIC -shared

ORT_ROOT := /root/packs/onnxruntime-linux-x64-gpu-1.21.0
CUDA_ROOT := /usr/local/cuda
FBGEMM_ROOT := /root/packs/FBGEMM
FBGEMM_BUILD := $(FBGEMM_ROOT)/build

INCLUDES := -I. -I$(ORT_ROOT)/include -I$(CUDA_ROOT)/include -I$(FBGEMM_ROOT)/include
LIBS := \
    -L$(ORT_ROOT)/lib -lonnxruntime \
    -L$(CUDA_ROOT)/lib64 -lcudart \
    -L$(FBGEMM_BUILD) -lfbgemm \
    -L$(FBGEMM_BUILD)/cpuinfo -lcpuinfo \
    -fopenmp

SRC := onnxruntime_cgo.cpp
TARGET := libonnx_cgo.so

all:
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(INCLUDES) $(LIBS)

clean:
	rm -f $(TARGET)
