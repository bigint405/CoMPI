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

func LoadModel(modelPath string, deviceID int, enableCudaGraph bool) (*ORTSession, error) {
	cPath := C.CString(modelPath)
	ecg := 0
	if enableCudaGraph {
		ecg = 1
	}
	sess := C.ORT_LoadModel(cPath, C.int(deviceID), C.int(ecg))
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
		C.free(unsafe.Pointer(outputsPtr))
	}

	return outputSlice, releaseFunc, nil
}

// RunInferenceWithGraphSlot 使用 graph 管理的固定 input/output 缓冲区执行一次推理。
//
//	batchSizeGraph – graph 视角下的 batch 维（通常是 maxBS）
//	pingIndex      – 0 / 1，对应 ping–pong 槽
//
// 返回：长度为 session 输出个数的 []*ORTTensor 切片，这些 tensor 全部由 session 持有，
//
//	不需要（也不能）纳入 per-batch 的释放逻辑。
func RunInferenceWithGraphSlot(
	session *ORTSession,
	batchSizeGraph int64,
	pingIndex int,
) ([]*ORTTensor, error) {
	if session == nil {
		return nil, errors.New("nil session")
	}

	cBatch := C.int64_t(batchSizeGraph)
	cPing := C.int(pingIndex)

	outputsPtr := C.ORT_RunInferenceWithGraphSlot(
		session,
		cBatch,
		cPing,
	)
	if outputsPtr == nil {
		return nil, errors.New("ORT_RunInferenceWithGraphSlot failed")
	}

	// 输出个数从 C 侧查询
	l := C.ORT_GetOutputCount(session)
	if l == 0 {
		return nil, errors.New("no outputs")
	}

	// 和 RunInferenceFixedOutput 一样，用 unsafe.Slice 构造零拷贝切片
	outputSlice := unsafe.Slice(outputsPtr, l)

	// 注意：这些 ORTTensor* 由 session 持久持有，不要在 batch 结束时调用 ORT_ReleaseTensor
	return outputSlice, nil
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
		return nil, nil, nil, nil, fmt.Errorf("dequantization failed: %d", int(status))
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

// DecodeRawVectors 在非量化模式下解析一条样本的原始 bytes，
// 直接在同一块地址空间上零拷贝返回每个输入向量及其 shape。
func DecodeRawVectors(data []byte) ([][]float32, [][]int64, error) {
	if len(data) < 4 {
		return nil, nil, errors.New("raw data too short for numVecs")
	}
	p := 0

	// 1) numVecs
	numVecs := *(*int32)(unsafe.Pointer(&data[p]))
	p += 4
	if numVecs <= 0 {
		return nil, nil, errors.New("numVecs <= 0 in raw data")
	}
	n := int(numVecs)

	// 2) dims[0..n-1]
	need := 4 * n
	if len(data) < p+need {
		return nil, nil, errors.New("raw data too short for dims")
	}
	dimsSlice := unsafe.Slice((*int32)(unsafe.Pointer(&data[p])), n)
	p += need

	// 3) shapeLens[0..n-1]
	if len(data) < p+need {
		return nil, nil, errors.New("raw data too short for shapeLens")
	}
	shapeLenSlice := unsafe.Slice((*int32)(unsafe.Pointer(&data[p])), n)
	p += need

	// 4) shapes flatten
	totalShape := 0
	for i := 0; i < n; i++ {
		if shapeLenSlice[i] < 0 {
			return nil, nil, errors.New("negative shapeLen in raw data")
		}
		totalShape += int(shapeLenSlice[i])
	}
	needShapes := 8 * totalShape
	if len(data) < p+needShapes {
		return nil, nil, errors.New("raw data too short for shapes")
	}
	shapesAll := unsafe.Slice((*int64)(unsafe.Pointer(&data[p])), totalShape)
	p += needShapes

	// 5) float32 data
	totalDim := 0
	for i := 0; i < n; i++ {
		if dimsSlice[i] < 0 {
			return nil, nil, errors.New("negative dim in raw data")
		}
		totalDim += int(dimsSlice[i])
	}
	needFloats := 4 * totalDim
	if len(data) < p+needFloats {
		return nil, nil, errors.New("raw data too short for float data")
	}

	floatBase := (*float32)(unsafe.Pointer(&data[p]))

	floatVecs := make([][]float32, n)
	shapes := make([][]int64, n)

	shapeOffset := 0
	floatOffset := 0
	for i := 0; i < n; i++ {
		dim := int(dimsSlice[i])
		shapeLen := int(shapeLenSlice[i])

		// shape[i]
		shapes[i] = shapesAll[shapeOffset : shapeOffset+shapeLen]
		shapeOffset += shapeLen

		// 第 i 个向量：从 floatOffset 开始，长度 dim
		vecPtr := (*float32)(unsafe.Pointer(
			uintptr(unsafe.Pointer(floatBase)) + uintptr(floatOffset*4),
		))
		floatVecs[i] = unsafe.Slice(vecPtr, dim)
		floatOffset += dim
	}

	return floatVecs, shapes, nil
}

func DecodeVectorsFromBytes(data []byte, useQuantization bool) ([][]float32, [][]int64, func(), error) {
	if useQuantization {
		vecs, _, shapes, release, err := DequantizeFBGEMM(data)
		return vecs, shapes, release, err
	}

	vecs, shapes, err := DecodeRawVectors(data)
	if err != nil {
		return nil, nil, nil, err
	}
	return vecs, shapes, nil, nil
}

// CreateTensorBatchFromGo 构造一个 GPU 上的批量 ORT Tensor（不复制原始 float32 数据）
func CreateTensorBatchFromGo(gpuID int, data [][]float32, shape []int64) (*ORTTensor, func(), error) {
	if len(data) == 0 || len(shape) == 0 {
		return nil, nil, errors.New("empty input")
	}

	batchSize := len(data)
	singleLen := len(data[0])
	if batchSize == 0 || singleLen == 0 {
		return nil, nil, errors.New("invalid data size")
	}

	// 用 C.malloc 分配 float* 数组，避免 Go slice 里存 Go 指针再传给 C
	ptrSize := unsafe.Sizeof((*C.float)(nil))
	ptrsMem := C.malloc(C.size_t(ptrSize) * C.size_t(batchSize))
	if ptrsMem == nil {
		return nil, nil, errors.New("malloc for ptrs failed")
	}
	// 这个数组只在本次调用内使用，可以在函数结束时释放
	defer C.free(ptrsMem)

	cPtrArray := (*[1 << 30]*C.float)(ptrsMem)

	for i := 0; i < batchSize; i++ {
		if data[i] == nil || len(data[i]) == 0 {
			cPtrArray[i] = nil
		} else {
			cPtrArray[i] = (*C.float)(unsafe.Pointer(&data[i][0]))
		}
	}

	cPtrs := (**C.float)(ptrsMem)
	cShape := (*C.int64_t)(unsafe.Pointer(&shape[0]))
	cShapeLen := C.size_t(len(shape))
	cSingleLen := C.size_t(singleLen)
	cBatchSize := C.int(batchSize)

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

// CreateGraphInputBatchFromGo 将某个输入 index 的整个 batch 上传到
// graph slot 管理的固定 GPU 缓冲区中。
func CreateGraphInputBatchFromGo(
	sess *ORTSession,
	pingIndex int,
	inputIndex int,
	data [][]float32,
	shape []int64,
) (*ORTTensor, error) {
	if sess == nil {
		return nil, errors.New("nil session")
	}
	if len(data) == 0 || len(shape) == 0 {
		return nil, errors.New("empty input")
	}

	batchSize := len(data)
	singleLen := len(data[0])
	if batchSize == 0 || singleLen == 0 {
		return nil, errors.New("invalid data size")
	}

	// 同样使用 C.malloc 分配 float* 数组，避免 Go slice 的指针被直接传入 C
	ptrSize := unsafe.Sizeof((*C.float)(nil))
	ptrsMem := C.malloc(C.size_t(ptrSize) * C.size_t(batchSize))
	if ptrsMem == nil {
		return nil, errors.New("malloc for graph input ptrs failed")
	}
	defer C.free(ptrsMem)

	cPtrArray := (*[1 << 30]*C.float)(ptrsMem)
	for i := range batchSize {
		if data[i] == nil || len(data[i]) == 0 {
			cPtrArray[i] = nil
		} else {
			cPtrArray[i] = (*C.float)(unsafe.Pointer(&data[i][0]))
		}
	}

	cPtrs := (**C.float)(ptrsMem)
	cShape := (*C.int64_t)(unsafe.Pointer(&shape[0]))
	cShapeLen := C.size_t(len(shape))

	cBatchSize := C.int(batchSize)
	cSingleLen := C.size_t(singleLen)
	cPing := C.int(pingIndex)
	cInputIndex := C.int(inputIndex)

	out := C.ORT_CreateOrUpdateGraphInput(
		sess,
		cPing,
		cInputIndex,
		cPtrs,
		cBatchSize,
		cSingleLen,
		cShape,
		cShapeLen,
	)
	if out == nil {
		return nil, errors.New("ORT_CreateOrUpdateGraphInput failed")
	}

	// 返回的 ORTTensor 属于 session/graph 管理，不需要每批释放
	return out, nil
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

// SetSessionCUDAGraphEnabled 启用 / 禁用某个 session 的 CUDA graph 管理
func SetSessionCUDAGraphEnabled(sess *ORTSession, enabled bool) {
	if sess == nil {
		return
	}
	flag := C.int(0)
	if enabled {
		flag = 1
	}
	C.ORT_SetCudaGraphEnabled(sess, flag)
}
