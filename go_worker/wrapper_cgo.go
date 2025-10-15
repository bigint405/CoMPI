package main

/*
#cgo CXXFLAGS: -std=c++17 -I. -I/root/packs/onnxruntime-linux-x64-gpu-1.21.0/include -I/usr/local/cuda/include -I/root/packs/FBGEMM/include
#cgo LDFLAGS: -L. -lonnx_cgo -L/root/packs/onnxruntime-linux-x64-gpu-1.21.0/lib -lonnxruntime -L/usr/local/cuda/lib64 -lcudart
#include "onnxruntime_cgo.h"
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

type ORTSession = C.ORTSession
type ORTTensor = C.ORTTensor

func LoadModel(modelPath string, deviceID int) (*ORTSession, error) {
	cPath := C.CString(modelPath)
	sess := C.ORT_LoadModel(cPath, C.int(deviceID))
	C.free(unsafe.Pointer(cPath))
	if sess == nil {
		return nil, errors.New("load onnx model error")
	}
	return sess, nil
}

func UnloadModel(sess *ORTSession) {
	C.ORT_ReleaseSession(sess)
}

func GetTensorShape(tensor *ORTTensor) ([]int64, func(), error) {
	var shapePtr *C.int64_t
	var shapeLen C.size_t

	releaseFunc := func() {
		if shapePtr != nil {
			C.free(unsafe.Pointer(shapePtr))
		}
	}

	status := C.ORT_GetTensorShape(tensor, &shapePtr, &shapeLen)
	if status != 0 {
		return nil, releaseFunc, errors.New("failed to get shape")
	}

	shape := unsafe.Slice((*int64)(unsafe.Pointer(shapePtr)), shapeLen)
	return shape, releaseFunc, nil
}

// RunInferenceFixedOutput 封装 ORT_RunInferenceFixedOutput（返回 GPU 上的多个输出张量）
func RunInferenceFixedOutput(session *ORTSession, inputs []*ORTTensor, batchSize int64) ([]*ORTTensor, func(), error) {
	if session == nil || len(inputs) == 0 {
		return nil, nil, errors.New("invalid session or inputs")
	}

	inputCount := len(inputs)
	inputsPtr := (**ORTTensor)(unsafe.Pointer(&inputs[0]))

	// 调用 C 函数
	outputsPtr := C.ORT_RunInferenceFixedOutput(
		session,
		inputsPtr,
		C.size_t(inputCount),
		C.int64_t(batchSize),
	)

	if outputsPtr == nil {
		return nil, nil, errors.New("inference failed")
	}

	l := C.ORT_GetOutputCount(session)

	// 构造 []*ORTTensor 切片（不复制，只封装）
	outputSlice := unsafe.Slice(outputsPtr, l)

	releaseFunc := func() {
		for i := range int(l) {
			C.ORT_ReleaseTensor(outputSlice[i])
		}
	}

	return outputSlice, releaseFunc, nil
}

// QuantizeVectorsFBGEMMBytes 执行量化并返回零拷贝的 []byte（需手动释放）
func QuantizeVectorsFBGEMMBytes(vectors [][]float32, shapes [][]int64) ([]byte, func(), error) {
	numVecs := len(vectors)
	if numVecs == 0 || len(shapes) != numVecs {
		return nil, nil, errors.New("invalid input")
	}

	// C.float*[]：vecPtrs
	vecPtrsSize := unsafe.Sizeof((*C.float)(nil)) * uintptr(numVecs)
	vecPtrs := C.malloc(C.size_t(vecPtrsSize))
	defer C.free(vecPtrs)
	cVecPtrs := (*[1 << 30]*C.float)(vecPtrs)

	// C.int32_t dims[]
	dimsSize := C.size_t(C.sizeof_int32_t * numVecs)
	cDims := (*C.int32_t)(C.malloc(dimsSize))
	defer C.free(unsafe.Pointer(cDims))
	goDims := unsafe.Slice(cDims, numVecs)

	// C.int64_t*[]：shapePtrs
	shapePtrsSize := unsafe.Sizeof((*C.int64_t)(nil)) * uintptr(numVecs)
	shapePtrs := C.malloc(C.size_t(shapePtrsSize))
	defer C.free(shapePtrs)
	cShapePtrs := (*[1 << 30]*C.int64_t)(shapePtrs)

	// C.int32_t shapeLens[]
	shapeLensSize := C.size_t(C.sizeof_int32_t * numVecs)
	cShapeLens := (*C.int32_t)(C.malloc(shapeLensSize))
	defer C.free(unsafe.Pointer(cShapeLens))
	goShapeLens := unsafe.Slice(cShapeLens, numVecs)

	for i := range numVecs {
		if len(vectors[i]) == 0 || len(shapes[i]) == 0 {
			return nil, nil, errors.New("empty vector or shape")
		}
		goDims[i] = C.int32_t(len(vectors[i]))
		cVecPtrs[i] = (*C.float)(unsafe.Pointer(&vectors[i][0]))

		goShapeLens[i] = C.int32_t(len(shapes[i]))
		cShapePtrs[i] = (*C.int64_t)(unsafe.Pointer(&shapes[i][0]))
	}

	// C output buffer
	var cOut *C.uchar
	cLenPtr := C.malloc(C.size_t(unsafe.Sizeof(C.size_t(0))))
	defer C.free(cLenPtr)

	status := C.QuantizeVectorsFBGEMM(
		(**C.float)(vecPtrs),
		cDims,
		(**C.int64_t)(shapePtrs),
		cShapeLens,
		C.int32_t(numVecs),
		&cOut,
		(*C.size_t)(cLenPtr),
	)
	if status != 0 {
		return nil, nil, errors.New("quantization failed")
	}
	cLen := *(*C.size_t)(cLenPtr)

	result := unsafe.Slice((*byte)(unsafe.Pointer(cOut)), int(cLen))

	release := func() {
		C.free(unsafe.Pointer(cOut))
	}

	return result, release, nil
}

// DequantizeFBGEMM 解量化 FBGEMM 数据，返回：每个向量的 []float32、长度、shape 结构、释放函数
func DequantizeFBGEMM(data []byte) ([][]float32, []int32, [][]int64, func(), error) {
	if len(data) == 0 {
		return nil, nil, nil, nil, errors.New("empty data")
	}

	cData := (*C.uint8_t)(unsafe.Pointer(&data[0]))
	cLen := C.size_t(len(data))

	var cVecs **C.float
	var cDims *C.int32_t
	var cShapes **C.int64_t
	var cShapeLens *C.int32_t
	var cNumVecs C.int

	status := C.DequantizeVectorsFBGEMM(
		cData,
		cLen,
		&cVecs,
		&cDims,
		&cShapes,
		&cShapeLens,
		&cNumVecs,
	)
	if status != 0 {
		return nil, nil, nil, nil, errors.New(fmt.Sprintf("dequantization failed: %d", int(status)))
	}

	num := int(cNumVecs)

	// 转换输出
	floatVecs := make([][]float32, num)
	dims := unsafe.Slice((*int32)(unsafe.Pointer(cDims)), num)
	shapeLens := unsafe.Slice((*int32)(unsafe.Pointer(cShapeLens)), num)
	shapePtrs := unsafe.Slice((**int64)(unsafe.Pointer(cShapes)), num)
	vecPtrs := unsafe.Slice((**C.float)(unsafe.Pointer(cVecs)), num)

	shapes := make([][]int64, num)
	for i := 0; i < num; i++ {
		dim := int(dims[i])
		floatVecs[i] = unsafe.Slice((*float32)(unsafe.Pointer(vecPtrs[i])), dim)

		shapeLen := int(shapeLens[i])
		shapes[i] = unsafe.Slice((*int64)(unsafe.Pointer(shapePtrs[i])), shapeLen)
	}

	// 封装释放函数，交由用户调用
	release := func() {
		for i := 0; i < num; i++ {
			C.free(unsafe.Pointer(vecPtrs[i]))
			C.free(unsafe.Pointer(shapePtrs[i]))
		}
		C.free(unsafe.Pointer(cVecs))
		C.free(unsafe.Pointer(cDims))
		C.free(unsafe.Pointer(cShapes))
		C.free(unsafe.Pointer(cShapeLens))
	}

	// 如果你想也返回释放函数可修改返回值
	return floatVecs, dims, shapes, release, nil
}

// CreateTensorBatchFromGo 构造一个 GPU 上的批量 ORT Tensor（不复制原始 float32 数据）
func CreateTensorBatchFromGo(gpuID int, data [][]float32, shape []int64) (*ORTTensor, func(), error) {
	if len(data) == 0 || len(shape) == 0 {
		return nil, nil, errors.New("empty input")
	}

	singleLen := len(data[0])
	// 取消下面的验证，加速
	for i := 1; i < len(data); i++ {
		if len(data[i]) != singleLen {
			return nil, nil, errors.New("inconsistent sample lengths")
		}
	}

	// 构建 []*C.float 指针数组
	ptrs := make([]*C.float, len(data))
	for i := range data {
		ptrs[i] = (*C.float)(unsafe.Pointer(&data[i][0]))
	}
	cPtrs := (**C.float)(unsafe.Pointer(&ptrs[0]))
	cShape := (*C.int64_t)(unsafe.Pointer(&shape[0]))
	cShapeLen := C.size_t(len(shape))
	cSingleLen := C.size_t(singleLen)
	cBatchSize := C.int(len(data))

	out := C.ORT_CreateTensorBatch(
		C.int(gpuID),
		cPtrs,
		cBatchSize,
		cSingleLen,
		cShape,
		cShapeLen,
	)
	if out == nil {
		return nil, nil, errors.New("ORT_CreateTensorBatch failed")
	}
	releaseFunc := func() {
		C.ORT_ReleaseTensor(out)
	}

	return out, releaseFunc, nil
}

// GetTensorDataUnsafeSliceBatched 返回 [][]float32 和 shape（按 batch 拆分），无复制，需释放
func GetTensorDataUnsafeSliceBatched(tensor *ORTTensor) ([][]float32, []int64, func(), error) {
	if tensor == nil {
		return nil, nil, nil, errors.New("nil tensor")
	}

	var cShape *C.int64_t
	var cDim C.size_t

	cData := C.ORT_GetTensorData(tensor, &cShape, &cDim)
	if cData == nil {
		return nil, nil, nil, errors.New("ORT_GetTensorData returned null")
	}

	// 转换 shape
	shape := unsafe.Slice((*int64)(unsafe.Pointer(cShape)), cDim)

	// 校验 shape
	if len(shape) < 1 || shape[0] <= 0 {
		C.free(unsafe.Pointer(cData))
		C.free(unsafe.Pointer(cShape))
		return nil, nil, nil, errors.New("invalid shape")
	}

	batchSize := int(shape[0])
	perSize := 1
	for _, d := range shape[1:] {
		if d <= 0 {
			C.free(unsafe.Pointer(cData))
			C.free(unsafe.Pointer(cShape))
			return nil, nil, nil, errors.New("invalid shape dimension")
		}
		perSize *= int(d)
	}
	total := batchSize * perSize

	// 映射 flat data
	flat := unsafe.Slice((*float32)(unsafe.Pointer(cData)), total)

	// 拆分为 [][]float32（无复制）
	result := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		result[i] = flat[i*perSize : (i+1)*perSize]
	}

	release := func() {
		C.free(unsafe.Pointer(cData))
		C.free(unsafe.Pointer(cShape))
	}

	return result, shape, release, nil
}
