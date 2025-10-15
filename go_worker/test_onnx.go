package main

/*
#cgo CXXFLAGS: -std=c++17 -I. -I/root/packs/onnxruntime-linux-x64-gpu-1.21.0/include -I/usr/local/cuda/include
#cgo LDFLAGS: -L. -lonnx_cgo -L/root/packs/onnxruntime-linux-x64-gpu-1.21.0/lib -lonnxruntime -L/usr/local/cuda/lib64 -lcudart
#include "onnxruntime_cgo.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math/rand"
	"unsafe"
)

func test() {
	modelPath := C.CString("/root/datas/models/resnet-152.onnx")
	defer C.free(unsafe.Pointer(modelPath))

	session := C.ORT_LoadModel(modelPath, 0)
	if session == nil {
		panic("❌ Failed to load ONNX model")
	}
	defer C.ORT_ReleaseSession(session)

	batchSize := int64(16)
	shape := []int64{batchSize, 3, 224, 224}
	inputLen := 1
	for _, d := range shape {
		inputLen *= int(d)
	}
	inputData := make([]float32, inputLen)
	for i := range inputData {
		inputData[i] = rand.Float32()
	}

	tensor := C.ORT_CreateTensor(0,
		(*C.float)(unsafe.Pointer(&inputData[0])),
		C.size_t(len(inputData)),
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.size_t(len(shape)))

	if tensor == nil {
		panic("❌ Failed to create input tensor")
	}
	defer C.ORT_ReleaseTensor(tensor)

	inputs := []*C.ORTTensor{tensor}
	outputRaw := C.ORT_RunInferenceFixedOutput(session, &inputs[0], 1, C.int64_t(batchSize))
	if outputRaw == nil {
		panic("❌ Failed to run inference")
	}

	outputSlice := (*[1]*C.ORTTensor)(unsafe.Pointer(outputRaw))[:1:1]
	defer C.ORT_ReleaseTensor(outputSlice[0])

	var outShape *C.int64_t
	var outDim C.size_t
	outData := C.ORT_GetTensorData(outputSlice[0], &outShape, &outDim)
	defer C.free(unsafe.Pointer(outData))
	defer C.free(unsafe.Pointer(outShape))

	goShape := (*[8]C.int64_t)(unsafe.Pointer(outShape))[:outDim:outDim]
	outLen := 1
	for _, d := range goShape {
		outLen *= int(d)
	}
	output := (*[1 << 30]float32)(unsafe.Pointer(outData))[:outLen:outLen]

	fmt.Println("✅ 输出 shape:", goShape)
	fmt.Println("✅ 输出前两个值:", output[0], output[1])
}
