package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	pb "compi_go_server/rpc_server"

	"github.com/google/btree"
	"github.com/rs/xid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const epsilon = 5.0 // 留一点富余，防止精度问题引起的额外计算，可手动调整。同一batch内不同RT的最大差距

type stageInfo struct {
	id             string
	stageLatency   float64
	maxBS          int32
	numOutput      uint32    // 输出向量的个数
	shapeInput     [][]int32 // 输出向量的shape
	modelSize      int64
	inputSize      int64
	runningMaxSize int64
	fillTimeOut    float64 // 凑一个batch等待的最大毫秒数

	deviceMu        sync.Mutex
	deployedDevices map[string]*deviceInfo // 已部署的设备

	modelLatency       map[string]float64
	directStageLatency map[string]float64 // modelID -> direct latency

	taskChan chan *requestInfo
}

func (s *stageInfo) getModelLatency(modelID string) float64 {
	l, _ := s.modelLatency[modelID]
	return l + s.stageLatency
}

func (s *stageInfo) getDirectLatency(modelID string) float64 {
	l, _ := s.directStageLatency[modelID]
	return l + s.stageLatency
}

type stageSequence struct {
	stageIDs    []string
	taskChan    chan *requestInfo
	latency     float64
	maxBS       int32
	fillTimeOut int64 // 一个batch超时的纳秒数

	batchMu       sync.Mutex
	queueingBatch *btree.BTree

	deviceMu        sync.Mutex
	deployedDevices map[string]*deviceInfo // 已部署的设备
}

func (sche *Scheduler) NewStageSequence(stageIDs []string, latency float64, maxBS int32, fillTimeOut float64) *stageSequence {
	sSeq := &stageSequence{
		stageIDs:    stageIDs,
		taskChan:    make(chan *requestInfo, maxBS),
		latency:     latency,
		maxBS:       maxBS,
		fillTimeOut: int64(fillTimeOut * 1e6), // 毫秒到纳秒

		queueingBatch: btree.New(3),

		deployedDevices: make(map[string]*deviceInfo),
	}
	// init deployedDevices
	for i, stageID := range stageIDs {
		stage := sche.getStage(stageID)
		stage.deviceMu.Lock()
		defer stage.deviceMu.Unlock()
		if i == 0 {
			for k, v := range stage.deployedDevices {
				sSeq.deployedDevices[k] = v
			}
		} else {
			deleteDevices := []string{}
			for k, _ := range sSeq.deployedDevices {
				_, ok := stage.deployedDevices[k]
				if !ok {
					deleteDevices = append(deleteDevices, k)
				}
			}
			for _, d := range deleteDevices {
				delete(sSeq.deployedDevices, d)
			}
		}
	}
	go sche.stageWorker(sSeq)
	return sSeq
}

type modelInfo struct {
	slo          float64
	startStageID string
}

type StageTime struct {
	recvTime              float64
	startQueueingTime     float64
	startInferTime        float64
	finishInferTime       float64
	finishPostProcessTime float64
	sendTime              float64
}

type requestInfo struct {
	id          string
	tArrive     float64
	rt          float64
	discardLine float64 // 丢弃阈值
	slo         float64
	stageIDs    []string
	modelID     string
	data        []byte

	batchID    string               // 传到哪个batch中
	action     chan pb.TargetAction // 作为中间结果，存储GetTarget的action
	recvDevice *deviceInfo          // 接收结果的设备
	startTag   uint32               // 传输中间结果的起始tag，随数量递增
	finalResp  chan int32           // 最终结果，存储状态，0代表正常，1代表丢弃

	stageTimeMu sync.Mutex
	stageTimes  map[string]StageTime // stageID -> stage time
}

func (req *requestInfo) debugInfo() string {
	return fmt.Sprintf("request %s- in %s- now infering stage %s\n", req.id, req.batchID, fmt.Sprintln(req.stageIDs))
}

type grpcClients struct {
	StartInferServiceClient      pb.StartInferServiceClient
	SetTargetDeviceServiceClient pb.SetTargetDeviceServiceClient
}

type deviceInfo struct {
	id                 string
	rankRecv           int32
	ipOfMachine        string // 所在机器的ip
	deviceNumOnMachine int32  // 在本机上的设备序号

	batchMu       sync.RWMutex
	queueingBatch *btree.BTree

	stageMu        sync.RWMutex
	deployedStages map[string]bool // 已部署的stage的id

	memUsedForAllModel   int64
	memUsedForMaxData    int64
	memUsedForMaxRunning int64
	maxMem               int64

	rpc     *grpcClients // 说是每个device一个，但是同一机器的device的rpc是相同的连接，只是指针的复制
	tcpConn net.Conn
	tcpMu   *sync.Mutex
}

func (device *deviceInfo) debugInfo() string {
	device.batchMu.Lock()
	s := fmt.Sprintf("%s (%d) with %d batches: \n", device.id, device.rankRecv, device.queueingBatch.Len())
	device.queueingBatch.Ascend(func(item btree.Item) bool {
		batch := item.(*batchInfo)
		s += batch.debugInfo()
		return true
	})
	device.batchMu.Unlock()
	s += "\n"
	return s
}

type batchInfo struct {
	id          string
	rt          float64
	discardLine float64 // 丢弃阈值
	stageSeq    *stageSequence
	stageIDs    []string
	latency     float64 // 整体执行时间（包括direct之后的stage）
	device      *deviceInfo
	reqMu       sync.RWMutex
	sendMu      sync.Mutex
	reqs        map[string]*requestInfo // req_id -> req
	hasMidReq   bool                    // 包含中间stage的请求，不可被丢弃
	// NEW —— 批级超时/收口
	createdAt time.Time
	deadline  time.Time
	timer     *time.Timer
	closed    bool      // 该批已收口，不再接收新样本
	finalBS   int32     // 收口时一次性确定
	closeOnce sync.Once // 防重复 finalize（满批+超时竞态）
}

func (a *batchInfo) Less(b btree.Item) bool {
	if a.rt == b.(*batchInfo).rt {
		return a.id < b.(*batchInfo).id
	}
	return a.rt < b.(*batchInfo).rt
}

func (b *batchInfo) isOpen() bool {
	if b == nil {
		return false
	}
	if b.closed || len(b.reqs) >= int(b.stageSeq.maxBS) {
		return false
	}
	return true
}

func (batch *batchInfo) addReq(req *requestInfo) {
	batch.reqMu.Lock()
	batch.reqs[req.id] = req
	if len(batch.reqs) > 1 {
		batch.rt = min(req.rt, batch.rt)
		batch.discardLine = min(req.discardLine, batch.discardLine)
	} else {
		batch.rt = req.rt
		batch.discardLine = req.discardLine
	}
	batch.reqMu.Unlock()
}

func (batch *batchInfo) addReqWithoutUpdateRT(req *requestInfo) {
	batch.reqMu.Lock()
	batch.reqs[req.id] = req
	batch.reqMu.Unlock()
}

func (batch *batchInfo) debugInfo() string {
	batch.reqMu.RLock()
	s := fmt.Sprintf("batch info %s- with %d reqs: \n", batch.id, len(batch.reqs))
	for k, _ := range batch.reqs {
		s += fmt.Sprintf("%s-, ", k)
	}
	batch.reqMu.RUnlock()
	s += "\n"
	return s
}

type StageTimes struct {
	PreProcessTimes    []float64 `json:"pre_process_times"`
	QueueingTimes      []float64 `json:"queueing_times"`
	InferTimes         []float64 `json:"infer_times"`
	PostProcessTimes   []float64 `json:"post_process_times"`
	WaitingToSendTimes []float64 `json:"waiting_to_send_times"`
	TransferTimes      []float64 `json:"transfer_times"`
	MaxSize            int       `json:"-"` // 不导出
	Len                int       `json:"-"` // 不导出
}

func NewStageTimes(maxSize int) *StageTimes {
	return &StageTimes{
		PreProcessTimes:    make([]float64, maxSize),
		QueueingTimes:      make([]float64, maxSize),
		InferTimes:         make([]float64, maxSize),
		PostProcessTimes:   make([]float64, maxSize),
		WaitingToSendTimes: make([]float64, maxSize),
		TransferTimes:      make([]float64, maxSize),
		MaxSize:            maxSize,
		Len:                0,
	}
}

func (st *StageTimes) append(prePorcessTime, queueingTime, inferTime, postProcessTime, waitingToSendTime, transferTime float64) {
	if st.Len >= st.MaxSize {
		st.Len = 0
	}
	st.PreProcessTimes[st.Len] = prePorcessTime
	st.QueueingTimes[st.Len] = queueingTime
	st.InferTimes[st.Len] = inferTime
	st.PostProcessTimes[st.Len] = postProcessTime
	st.WaitingToSendTimes[st.Len] = waitingToSendTime
	st.TransferTimes[st.Len] = transferTime
	st.Len++
}

func (st *StageTimes) export() *StageTimes {
	return &StageTimes{
		PreProcessTimes:    st.PreProcessTimes[:st.Len],
		QueueingTimes:      st.QueueingTimes[:st.Len],
		InferTimes:         st.InferTimes[:st.Len],
		PostProcessTimes:   st.PostProcessTimes[:st.Len],
		WaitingToSendTimes: st.WaitingToSendTimes[:st.Len],
		TransferTimes:      st.TransferTimes[:st.Len],
		MaxSize:            st.MaxSize,
		Len:                st.Len,
	}
}

type Scheduler struct {
	dag *DAG

	stages sync.Map
	// stages map[string]*stageInfo

	models sync.Map
	// models map[string]*modelInfo

	batchs sync.Map // batch_id -> *batchInfo

	requests sync.Map // req_id -> *requestInfo

	deviceMu           sync.Mutex //锁下面用于注册的计数的，不锁map
	deviceNumRegisterd int32
	deviceNum          int32

	startRank int32
	devices   sync.Map
	// devices            map[string]*deviceInfo

	staticDeployment *[][]string

	nextStageMu          sync.RWMutex
	nextStageMap         map[string]map[string][]string
	nextStageWithThisMu  sync.RWMutex
	nextStageWithThisMap map[string]map[string][]string

	newStageSeqMu        sync.Mutex
	stageSeqStageMap     sync.Map // joined stageIDs -> stageSequence
	stageSeqModelMap     sync.Map // modelID -> stageID -> stageSequence
	stageSeqModelNextMap sync.Map // modelID -> stageID -> stageSequence

	justStatic        bool
	discard           bool
	debug             bool
	debugTimeMaxSize  int
	debugStageTimesMu sync.Mutex
	debugStageTimes   map[string]map[string]*StageTimes // modelID -> stageID -> []stagetimes
	debugModelTimesMu sync.Mutex
	debugModelTimes   map[string]*StageTimes // modelID -> []stagetimes

	distTagMu sync.Mutex
	distTag   uint32
}

func (sche *Scheduler) clearDebugTimes() {
	sche.debugStageTimesMu.Lock()
	defer sche.debugStageTimesMu.Unlock()

	for _, stageMap := range sche.debugStageTimes {
		for _, stageTimes := range stageMap {
			stageTimes.Len = 0
		}
	}

	sche.debugModelTimesMu.Lock()
	defer sche.debugModelTimesMu.Unlock()

	for _, modelTimes := range sche.debugModelTimes {
		modelTimes.Len = 0
	}
	log.Printf("Debug times cleared")
}

func (sche *Scheduler) exportDebugTimes(exportDir string) {
	// 创建导出目录
	if err := os.MkdirAll(exportDir, os.ModePerm); err != nil {
		log.Printf("ERROR: Failed to create export directory: %v", err)
		return
	}

	// 导出 debugStageTimes
	sche.debugStageTimesMu.Lock()
	for modelID, stageMap := range sche.debugStageTimes {
		for stageID, stageTimes := range stageMap {
			// 构造文件路径
			filePath := filepath.Join(exportDir, modelID+"_"+stageID+".json")

			// 打开文件（如果不存在则创建）
			file, err := os.Create(filePath)
			if err != nil {
				log.Printf("ERROR: Failed to create file %s: %v", filePath, err)
				continue
			}

			// 将数据编码为 JSON 并写入文件
			encoder := json.NewEncoder(file)
			if err := encoder.Encode(stageTimes.export()); err != nil {
				log.Printf("ERROR: Failed to write JSON to file %s: %v", filePath, err)
			}

			file.Close()
		}
	}
	sche.debugStageTimesMu.Unlock()

	// 导出 debugModelTimes
	sche.debugModelTimesMu.Lock()
	for modelID, modelTimes := range sche.debugModelTimes {
		// 构造文件路径
		filePath := filepath.Join(exportDir, modelID+".json")

		// 打开文件（如果不存在则创建）
		file, err := os.Create(filePath)
		if err != nil {
			log.Printf("ERROR: Failed to create file %s: %v", filePath, err)
			continue
		}

		// 将数据编码为 JSON 并写入文件
		encoder := json.NewEncoder(file)
		if err := encoder.Encode(modelTimes.export()); err != nil {
			log.Printf("ERROR: Failed to write JSON to file %s: %v", filePath, err)
		}

		file.Close()
	}
	sche.debugModelTimesMu.Unlock()

	log.Printf("Debug times exported to directory: %s", exportDir)
}

func (sche *Scheduler) getStage(stageID string) *stageInfo {
	val, ok := sche.stages.Load(stageID)
	if !ok {
		return nil
	}
	return val.(*stageInfo)
}

func (sche *Scheduler) getDevice(deviceID string) *deviceInfo {
	val, ok := sche.devices.Load(deviceID)
	if !ok {
		return nil
	}
	return val.(*deviceInfo)
}

func (sche *Scheduler) getModel(modelID string) *modelInfo {
	val, ok := sche.models.Load(modelID)
	if !ok {
		return nil
	}
	return val.(*modelInfo)
}

func (sche *Scheduler) getBatch(batchID string) *batchInfo {
	val, ok := sche.batchs.Load(batchID)
	if !ok {
		return nil
	}
	return val.(*batchInfo)
}

func (sche *Scheduler) getReq(reqID string) *requestInfo {
	val, ok := sche.requests.Load(reqID)
	if !ok {
		return nil
	}
	return val.(*requestInfo)
}

func (sche *Scheduler) getReqAndDelete(reqID string) *requestInfo {
	val, ok := sche.requests.LoadAndDelete(reqID)
	if !ok {
		return nil
	}
	return val.(*requestInfo)
}

func (sche *Scheduler) getTagAndAdd(add uint32) uint32 {
	sche.distTagMu.Lock()
	defer sche.distTagMu.Unlock()
	a := sche.distTag
	sche.distTag += add
	return a
}

func (sche *Scheduler) debugInfo() string {
	s := "\nStages: \n"
	sche.stages.Range(func(key, value any) bool {
		stage := value.(*stageInfo)
		s += "stage " + stage.id + ":\n"
		return true
	})
	s += "\nDevices: \n"
	sche.devices.Range(func(key, value any) bool {
		device := value.(*deviceInfo)
		s += device.debugInfo()
		return true
	})
	s += "\nRequests: \n"
	sche.requests.Range(func(key, value any) bool {
		req := value.(*requestInfo)
		s += req.debugInfo()
		return true
	})
	return s
}

func (sche *Scheduler) isFirstStage(req *requestInfo) bool {
	return sche.getModel(req.modelID).startStageID == req.stageIDs[0]
}

func (sche *Scheduler) nextStage(stageID string, modelID string) []string {
	// 先获取读锁，检查是否已有缓存
	sche.nextStageMu.RLock()
	if modelMap, exists := sche.nextStageMap[stageID]; exists {
		if result, exists := modelMap[modelID]; exists {
			sche.nextStageMu.RUnlock() // 释放读锁
			return result
		}
	}
	sche.nextStageMu.RUnlock() // 释放读锁

	// 如果缓存中没有结果，获取写锁并计算结果
	sche.nextStageMu.Lock()
	defer sche.nextStageMu.Unlock()

	// 再次检查，防止其他线程已经写入
	if modelMap, exists := sche.nextStageMap[stageID]; exists {
		if result, exists := modelMap[modelID]; exists {
			return result
		}
	} else {
		// 如果 stageID 不存在，初始化对应的 map
		sche.nextStageMap[stageID] = make(map[string][]string)
	}

	// 调用 sche.dag.NextStage 并存储结果
	result := sche.dag.NextStage(stageID, modelID)
	sche.nextStageMap[stageID][modelID] = result
	return result
}

func (sche *Scheduler) nextStageWithThis(stageID string, modelID string) []string {
	// 先获取读锁，检查是否已有缓存
	sche.nextStageWithThisMu.RLock()
	if modelMap, exists := sche.nextStageWithThisMap[stageID]; exists {
		if result, exists := modelMap[modelID]; exists {
			sche.nextStageWithThisMu.RUnlock() // 释放读锁
			return result
		}
	}
	sche.nextStageWithThisMu.RUnlock() // 释放读锁

	// 如果缓存中没有结果，获取写锁并计算结果
	sche.nextStageWithThisMu.Lock()
	defer sche.nextStageWithThisMu.Unlock()

	// 再次检查，防止其他线程已经写入
	if modelMap, exists := sche.nextStageWithThisMap[stageID]; exists {
		if result, exists := modelMap[modelID]; exists {
			return result
		}
	} else {
		// 如果 stageID 不存在，初始化对应的 map
		sche.nextStageWithThisMap[stageID] = make(map[string][]string)
	}

	// 调用 sche.dag.NextStageWithThis 并存储结果
	result := sche.dag.NextStageWithThis(stageID, modelID)
	sche.nextStageWithThisMap[stageID][modelID] = result
	return result
}

func joinStageIDs(stageIDs []string) string {
	return strings.Join(stageIDs, "\x00")
}

func (sche *Scheduler) getStageSequence(modelID, stageID string) *stageSequence {
	var stageMap *sync.Map
	stageMapAny, ok := sche.stageSeqModelMap.Load(modelID)
	if ok {
		stageMap = stageMapAny.(*sync.Map)
	} else {
		stageMap = &sync.Map{}
		sche.stageSeqModelMap.Store(modelID, stageMap)
	}

	stageSeqAny, ok := stageMap.Load(stageID)
	if ok {
		return stageSeqAny.(*stageSequence)
	}

	stageIDs := sche.nextStageWithThis(stageID, modelID)
	stageKey := joinStageIDs(stageIDs)

	sche.newStageSeqMu.Lock()
	stageSeqAny2, ok := sche.stageSeqStageMap.Load(stageKey)
	var stageSeq *stageSequence
	if ok {
		stageSeq = stageSeqAny2.(*stageSequence)
	} else {
		stage := sche.getStage(stageID)
		stageSeq = sche.NewStageSequence(stageIDs, stage.getModelLatency(modelID), stage.maxBS, stage.fillTimeOut)
		sche.stageSeqStageMap.Store(stageKey, stageSeq)
	}
	sche.newStageSeqMu.Unlock()

	stageMap.Store(stageID, stageSeq)
	return stageSeq
}

func (sche *Scheduler) getNextStageSequence(modelID, stageID string) *stageSequence {
	var stageMap *sync.Map
	stageMapAny, ok := sche.stageSeqModelNextMap.Load(modelID)
	if ok {
		stageMap = stageMapAny.(*sync.Map)
	} else {
		stageMap = &sync.Map{}
		sche.stageSeqModelNextMap.Store(modelID, stageMap)
	}

	stageSeqAny, ok := stageMap.Load(stageID)
	if ok {
		return stageSeqAny.(*stageSequence)
	}

	stageIDs := sche.nextStage(stageID, modelID)
	if len(stageIDs) == 0 {
		return nil
	}
	stageKey := joinStageIDs(stageIDs)

	sche.newStageSeqMu.Lock()
	stageSeqAny2, ok := sche.stageSeqStageMap.Load(stageKey)
	var stageSeq *stageSequence
	if ok {
		stageSeq = stageSeqAny2.(*stageSequence)
	} else {
		stage := sche.getStage(stageID)
		stageSeq = sche.NewStageSequence(stageIDs, stage.getModelLatency(modelID), stage.maxBS, stage.fillTimeOut)
		sche.stageSeqStageMap.Store(stageKey, stageSeq)
	}
	sche.newStageSeqMu.Unlock()

	stageMap.Store(stageID, stageSeq)
	return stageSeq
}

func (sche *Scheduler) loadStageProfile(config *[]StageProfile) {
	sche.stages = sync.Map{}
	for _, s := range *config {
		sche.stages.Store(s.ID, &stageInfo{
			id:             s.ID,
			stageLatency:   s.Latency,
			maxBS:          s.MaxBS,
			numOutput:      s.NumOutput,
			modelSize:      s.ModelSize,
			inputSize:      s.InputSize,
			runningMaxSize: s.RunningMaxSize,
			fillTimeOut:    s.FillTimeOut,

			shapeInput:      s.ShapeInput,
			deviceMu:        sync.Mutex{},
			deployedDevices: make(map[string]*deviceInfo),
			taskChan:        make(chan *requestInfo, s.MaxBS),
		})
	}
	sche.nextStageMu.Lock()
	sche.nextStageMap = make(map[string]map[string][]string)
	sche.nextStageMu.Unlock()
	sche.nextStageWithThisMu.Lock()
	sche.nextStageWithThisMap = make(map[string]map[string][]string)
	sche.nextStageWithThisMu.Unlock()
}

func (sche *Scheduler) dfsUpdateLatency(stage *stageInfo) {
	edges := sche.dag.Edges[stage.id]
	stage.modelLatency = map[string]float64{}
	stage.directStageLatency = map[string]float64{}
	for modelID, edge := range edges {
		sub_stage := sche.getStage(edge.Target)
		if sub_stage.modelLatency == nil { // 记忆化搜索，没有搜过的进去搜索
			sche.dfsUpdateLatency(sub_stage)
		}
		stage.modelLatency[modelID] = sub_stage.getModelLatency(modelID)
		log.Printf("Updating latency of stage %s, model %s: %.3e", stage.id, modelID, stage.modelLatency[modelID])

		if edge.Type == Direct {
			stage.directStageLatency[modelID] = sub_stage.getDirectLatency(modelID)
		}

		log.Printf("Updating direct latency of stage %s, model %s: %.3e", stage.id, modelID, stage.directStageLatency[modelID])
	}
}

func (sche *Scheduler) loadModelConfig(config *[]ModelConfig) {
	sche.models = sync.Map{}
	for _, m := range *config {
		sche.models.Store(m.ID, &modelInfo{
			slo:          m.SLO,
			startStageID: m.StartStage,
		})
		sche.dfsUpdateLatency(sche.getStage(m.StartStage))
	}
}

func (sche *Scheduler) loadDeviceConfig(deviceNum int32) {
	sche.deviceNum = deviceNum
	sche.devices = sync.Map{}
}

func (sche *Scheduler) loadStaticDeployment(config *[][]string) {
	sche.staticDeployment = config
}

func (sche *Scheduler) LoadConfig(config *SchedulerConfig) {
	sche.dag = NewDAGFromConfig(&config.DAGConfig)
	sche.loadStageProfile(&config.StageProfile)
	sche.loadModelConfig(&config.ModelConfig)
	sche.loadDeviceConfig(config.DeviceNum)
	sche.loadStaticDeployment(&config.StaticDeployment)
	sche.justStatic = config.JustStatic
	sche.discard = config.Discard
	sche.debug = config.Debug
	if sche.debug {
		sche.debugTimeMaxSize = config.DebugTimeMaxSize
		sche.debugStageTimes = make(map[string]map[string]*StageTimes)
		sche.debugModelTimes = make(map[string]*StageTimes)
	}
	sche.batchs = sync.Map{}
	sche.requests = sync.Map{}
	sche.distTag = 0
	sche.distTagMu = sync.Mutex{}
}

func (sche *Scheduler) DeployStage(stageID string, deviceID string) {
	stage := sche.getStage(stageID)
	device := sche.getDevice(deviceID)

	stage.deviceMu.Lock()
	device.stageMu.Lock()
	defer device.stageMu.Unlock()
	defer stage.deviceMu.Unlock()

	device.memUsedForAllModel = max(device.memUsedForAllModel, stage.modelSize)
	device.memUsedForMaxData = max(device.memUsedForMaxData, stage.inputSize)
	device.memUsedForMaxRunning = max(device.memUsedForMaxRunning, stage.runningMaxSize)
	device.deployedStages[stageID] = true
	log.Printf("stage %s -> device %s", stageID, deviceID)
	stage.deployedDevices[deviceID] = device
}

// 注册设备，返回startRank、worldSize、静态部署策略，以及error
func (sche *Scheduler) RegisterDevice(machineIP string, deviceNum int32, maxMem []int64) (int32, int32, *[][]string, error) {
	sche.deviceMu.Lock()
	if sche.deviceNumRegisterd+deviceNum > sche.deviceNum {
		sche.deviceMu.Unlock()
		return 0, 0, nil, errors.New("no need for more devices")
	}

	// 初始化rpc连接
	var err error
	conn, err := grpc.NewClient(machineIP+":50052", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		sche.deviceMu.Unlock()
		return 0, 0, nil, fmt.Errorf("无法连接 gRPC 服务器: %v", err)
	}
	rpc := &grpcClients{}
	rpc.StartInferServiceClient = pb.NewStartInferServiceClient(conn)
	rpc.SetTargetDeviceServiceClient = pb.NewSetTargetDeviceServiceClient(conn)

	// 初始化tcp连接
	tcpConn, err := ConnectTCP(machineIP + ":50053")
	if err != nil {
		log.Fatalf("Failed to connect tcp of : %v", err)
	}
	tcpMu := &sync.Mutex{}

	deviceRank := sche.deviceNumRegisterd
	sche.deviceNumRegisterd += deviceNum
	sr := sche.startRank
	sche.startRank += deviceNum // 每个gpu占一个rank
	sche.deviceMu.Unlock()

	var staticDeployment [][]string
	l := int32(len(*sche.staticDeployment))
	for i := deviceRank; i < deviceRank+deviceNum; i++ {
		if i < l {
			staticDeployment = append(staticDeployment, (*sche.staticDeployment)[i])
		} else {
			staticDeployment = append(staticDeployment, make([]string, 0))
		}
	}

	for i := int32(0); i < deviceNum; i++ {
		deviceID := fmt.Sprintf("device-%s-%d", machineIP, i)
		sche.devices.Store(deviceID, &deviceInfo{
			id: deviceID,
			// rankSend:             sr + i*2,
			rankRecv:             sr + i,
			ipOfMachine:          machineIP,
			deviceNumOnMachine:   i,
			batchMu:              sync.RWMutex{},
			queueingBatch:        btree.New(3),
			stageMu:              sync.RWMutex{},
			deployedStages:       make(map[string]bool),
			maxMem:               maxMem[i],
			memUsedForAllModel:   0,
			memUsedForMaxData:    0,
			memUsedForMaxRunning: 0,
			rpc:                  rpc,
			tcpConn:              tcpConn,
			tcpMu:                tcpMu,
		})

		deployStages := staticDeployment[i]
		// 根据StaticDeployment部署stage
		for _, stageID := range deployStages {
			sche.DeployStage(stageID, deviceID)
		}
	}
	return sr, sche.deviceNum /*worker数量*/, &staticDeployment, nil
}

// batch 超时或满了，准备发车开始推理
func (sche *Scheduler) finalizeBatch(b *batchInfo, reason string) {
	b.closeOnce.Do(func() {
		// 停止timer并清空通道
		if b.timer != nil {
			if !b.timer.Stop() {
				select {
				case <-b.timer.C:
				default:
				}
			}
		}
		log.Printf("Finalize batch because of %s, batch %s", reason, b.id)
		// 从 stage 的 queueingBatch 下树（避免后续被误命中）
		b.stageSeq.batchMu.Lock()
		b.stageSeq.queueingBatch.Delete(b)
		b.stageSeq.batchMu.Unlock()

		// 标记关闭、确定 final_bs
		b.reqMu.Lock()
		finalBS := int32(len(b.reqs))
		b.closed = true
		b.finalBS = finalBS
		b.reqMu.Unlock()

		if b.device != nil {
			// 各req数据在stageworker中已经发过（或正在发送）发“发车信号”到 worker（复用 StartInferService）
			log.Printf("Batchsize confirm %d, of batch %s", b.finalBS, b.id)
			go func() {
				req := &pb.StartInferRequest{
					BatchId:   b.id,
					DeviceNum: b.device.deviceNumOnMachine,
					FinalBs:   finalBS,
				}
				_, err := b.device.rpc.StartInferServiceClient.StartInfer(context.Background(), req)
				if err != nil {
					log.Printf("ERROR: StartInfer %s failed: %v", b.id, err)
				}
			}()
		}
	})
}

func checkDeviceFinishQueueOnTime(now float64, di *deviceInfo) (bool, float64) {
	ok := true
	if di.queueingBatch.Len() > 0 {
		di.queueingBatch.Ascend(func(i btree.Item) bool {
			bi, ok := i.(*batchInfo)
			if !ok {
				log.Printf("ERROR: Type assertion in checkDeviceFinishQueueOnTime failed, expected *batchInfo but got %T", i)
				now = math.MaxFloat64
				return false
			}
			batchedLatency := bi.latency * (0.8 + 0.2*float64(len(bi.reqs)/int(bi.stageSeq.maxBS))) // 让请求均匀地分布于多个设备上，不要集中在一个上面
			now += batchedLatency
			log.Printf("DEVICE %s: stages %s takes %.3f, from %s, to %s, rt is %s", di.id, fmt.Sprintln(bi.stageIDs), batchedLatency, getTimeStr(now-batchedLatency), getTimeStr(now), getTimeStr(bi.rt))
			if now > bi.discardLine {
				ok = false
			}
			return true
		})
	}
	return ok, now
}

func (sche *Scheduler) deployReqOnQueueingBatch(req *requestInfo) *batchInfo {
	stageReq := sche.getStageSequence(req.modelID, req.stageIDs[0])
	var left, right *batchInfo
	stageReq.queueingBatch.DescendLessOrEqual(&batchInfo{rt: req.rt + epsilon}, func(i btree.Item) bool {
		left = i.(*batchInfo) // 这里无需给device的队列上锁，因为只有两种情况会修改该batch使得后面没法用：1.设备执行该batch，但执行时需要先申请stage的锁，所以不会出现问题；2.其它新请求抢先把该batch塞满，但只有该stage的请求会用到该batch，而该stage的请求也需要先申请stage的锁，所以还是不会有问题
		return false
	})
	dt := math.MaxFloat64
	if left != nil {
		dt = math.Abs(float64(left.rt - req.rt))
	}
	now := getTime()
	stageReq.queueingBatch.AscendGreaterOrEqual(&batchInfo{rt: req.rt}, func(i btree.Item) bool {
		bi := i.(*batchInfo)
		if bi.rt <= req.rt+epsilon { // 在可接受误差内，直接返回
			right = bi
			bi.device.batchMu.Lock() // 这里上锁是要直接对队列进行操作，而同一队列的其它batch可能也在操作该队列，所以需要上锁
			bi.device.queueingBatch.Delete(bi)
			bi.addReq(req)
			bi.device.queueingBatch.ReplaceOrInsert(bi)
			bi.device.batchMu.Unlock()
			return false
		}
		if bi.rt-req.rt >= dt { // 继续找还不如用左边的，不找了
			return false
		}

		old_rt := bi.rt
		bi.device.batchMu.Lock()
		defer bi.device.batchMu.Unlock()
		// 假设插入
		bi.device.queueingBatch.Delete(bi)
		bi.rt = req.rt
		bi.device.queueingBatch.ReplaceOrInsert(bi)
		// 验证是否可行
		ok, _ := checkDeviceFinishQueueOnTime(now, bi.device)
		if ok { // 可行，直接继续
			right = bi
			bi.addReqWithoutUpdateRT(req)
			return false // 找到了
		}
		// 不可行，还原队列
		bi.device.queueingBatch.Delete(bi)
		bi.rt = old_rt
		bi.device.queueingBatch.ReplaceOrInsert(bi)
		return true // 继续找
	})

	var batch *batchInfo
	if right != nil {
		batch = right
	} else if left != nil {
		batch = left
		batch.device.batchMu.Lock()
		batch.device.queueingBatch.Delete(batch)
		batch.addReq(req)
		batch.device.queueingBatch.ReplaceOrInsert(batch)
		batch.device.batchMu.Unlock()
	} else {
		return nil
	}
	return batch
}

func (sche *Scheduler) deployBatchOnGPU(batch *batchInfo) *deviceInfo {
	// 搜索已部署stage且能处理该batch的GPU
	batch.stageSeq.deviceMu.Lock()
	defer batch.stageSeq.deviceMu.Unlock()

	var bestMu sync.Mutex
	var device *deviceInfo
	var newFinishTime float64
	var wg sync.WaitGroup

	var backupDevice *deviceInfo
	var backupFinishTime float64

	log.Printf("Finding device for batch %s", batch.id)
	nowTime := getTime()
	for _, di := range batch.stageSeq.deployedDevices {
		wg.Add(1)
		go func() {
			defer wg.Done()
			di.batchMu.Lock()
			defer di.batchMu.Unlock()
			log.Printf("Trying device %s for batch %s", di.id, batch.id)
			// 尝试加入batch，测试是否可行
			di.queueingBatch.ReplaceOrInsert(batch)
			ok, now := checkDeviceFinishQueueOnTime(nowTime, di)
			if ok { // 找到可行的
				log.Printf("Device %s for batch %s OK!", di.id, batch.id)
				bestMu.Lock()
				defer bestMu.Unlock()
				if device == nil || now < newFinishTime { // 找到了更优的device，更新
					log.Printf("Device %s for batch %s update new finish time %s", di.id, batch.id, getTimeStr(now))
					if device != nil {
						device.batchMu.Lock()
						device.queueingBatch.Delete(batch) // 还原之前的队列
						device.batchMu.Unlock()
					}
					device = di
					newFinishTime = now
					if backupDevice != nil {
						backupDevice.batchMu.Lock()
						backupDevice.queueingBatch.Delete(batch) // 还原之前的队列
						backupDevice.batchMu.Unlock()
					}
				} else {
					log.Printf("Device %s for batch %s finish time %s later than old time %s", di.id, batch.id, getTimeStr(now), getTimeStr(newFinishTime))
					di.queueingBatch.Delete(batch) // 还原之前的队列
				}
			} else if batch.hasMidReq || !sche.discard { // 目前没找到合法的，但是这个batch有中间stage的请求，或者不允许丢弃，需要找个备用的
				log.Printf("Device %s for batch %s not OK but discard is illegal", di.id, batch.id)
				bestMu.Lock()
				defer bestMu.Unlock()
				if device == nil && (backupDevice == nil || now < backupFinishTime) {
					if backupDevice != nil {
						backupDevice.batchMu.Lock()
						backupDevice.queueingBatch.Delete(batch) // 还原之前的队列
						backupDevice.batchMu.Unlock()
					}
					backupDevice = di
					backupFinishTime = now
				} else {
					di.queueingBatch.Delete(batch) // 还原之前的队列
				}
			} else {
				log.Printf("Device %s for batch %s not OK and discard", di.id, batch.id)
				di.queueingBatch.Delete(batch)
			}
		}()
	}
	wg.Wait()
	if device == nil {
		device = backupDevice
	}
	return device
}

func (sche *Scheduler) scheduleReqOnBatch(ri *requestInfo) (*batchInfo, bool) {
	batch := sche.deployReqOnQueueingBatch(ri)
	stageSeq := sche.getStageSequence(ri.modelID, ri.stageIDs[0])
	new_batch := false
	if batch == nil {
		// 建立新batch
		new_batch = true
		batch = &batchInfo{
			id:        "batch-" + xid.New().String(),
			stageSeq:  stageSeq,
			stageIDs:  ri.stageIDs,
			reqMu:     sync.RWMutex{},
			sendMu:    sync.Mutex{},
			reqs:      make(map[string]*requestInfo),
			latency:   stageSeq.latency,
			createdAt: time.Now(),
		}
		log.Printf("Creating new batch for %s: %s", ri.stageIDs, batch.id)
		batch.addReq(ri)
		if !sche.isFirstStage(ri) {
			batch.hasMidReq = true
		}
		// 在已部署有stage的GPU上寻找batch的部署设备
		device := sche.deployBatchOnGPU(batch)
		batch.device = device
		// 添加到stage的队列里
		if device != nil {
			stageSeq.queueingBatch.ReplaceOrInsert(batch) // 在外面已经上锁了，所以这里不用上锁
		}
		// 保存batch
		sche.batchs.Store(batch.id, batch)
		// 设置batch的deadline
		batch.deadline = batch.createdAt.Add(time.Duration(stageSeq.fillTimeOut) * time.Nanosecond)
		if stageSeq.fillTimeOut > 0 {
			d := time.Until(batch.deadline)
			batch.timer = time.NewTimer(d)
			go func(b *batchInfo) {
				<-b.timer.C
				sche.finalizeBatch(b, "timeout")
			}(batch)
		}
	}
	return batch, new_batch
}

func startInfer(ri *requestInfo, batch *batchInfo, stageIDs []string, maxBs int32, fillTimeOut int64) {
	rpcReq := &pb.SetTargetRequest{
		RequestId:         ri.id,
		StageId:           stageIDs,
		BatchId:           batch.id,
		Rt:                float32(batch.rt),
		DeviceNum:         batch.device.deviceNumOnMachine,
		StartTag:          ri.startTag,
		MaxBs:             maxBs,
		FollowServerInfer: fillTimeOut != 0,
	}
	go func() {
		_, err := batch.device.rpc.SetTargetDeviceServiceClient.SetTarget(context.Background(), rpcReq)
		if err != nil {
			log.Printf("ERROR: send SetTarget request of req %s error: %v", ri.id, err)
		}
	}()

	go func() {
		batch.device.tcpMu.Lock()
		defer batch.device.tcpMu.Unlock()
		err := SendBytesWithID(batch.device.tcpConn, ri.startTag, ri.data)
		ri.data = nil
		if err != nil {
			log.Printf("ERROR: send data of req %s error: %v", ri.id, err)
		}
	}()

}

func (sche *Scheduler) stageWorker(stage *stageSequence) {
	var req *requestInfo
	for {
		startStageReqs := make([]*requestInfo, 0, stage.maxBS)
		midStageReqs := make([]*requestInfo, 0, stage.maxBS)
		ids := fmt.Sprint(stage.stageIDs)
		if req == nil { // 没有之前未处理的req，取出新req处理
			req = <-stage.taskChan
			log.Printf("stage %s get req %s", ids, req.id)
		} else {
			log.Printf("stage %s get newReq %s with new batch", ids, req.id)
		}

		if sche.isFirstStage(req) {
			startStageReqs = append(startStageReqs, req)
		} else {
			midStageReqs = append(midStageReqs, req)
		}

		stage.batchMu.Lock()
		batch, _ := sche.scheduleReqOnBatch(req)
		closed := false
		log.Printf("stage %s put req %s in batch %s", ids, req.id, batch.id)
		// 从该stage堆积的请求中找出适合一块放到该batch里的，避免每次都重新找一遍
		for {
			// batch如果满了或者不接收了就从stage的batch队列中去掉
			if !batch.isOpen() {
				closed = true
				stage.queueingBatch.Delete(batch)
				req = nil
				break
			}
			// 没有堆积的请求了，退出
			if len(stage.taskChan) == 0 {
				req = nil
				break
			}
			// 尝试将rt相近的req移送同一batch，减少计算开销
			newReq := <-stage.taskChan
			if math.Abs(newReq.rt-req.rt) <= epsilon {
				log.Printf("stage %s put newReq %s in batch %s", ids, newReq.id, batch.id)
				batch.addReqWithoutUpdateRT(newReq)
				if sche.isFirstStage(newReq) {
					startStageReqs = append(startStageReqs, newReq)
				} else {
					midStageReqs = append(midStageReqs, newReq)
				}
				// 满足放到该batch中，继续从队列中拿
				continue
			} else {
				// 不满足放到该batch中的条件，重新找一个
				req = newReq
				break
			}
		}
		stage.batchMu.Unlock()

		// 进行req的转发
		if batch.device != nil {
			log.Printf("Deploying batch %s -> device %s, rt: %s, batchsize: %d", batch.id, batch.device.id, getTimeStr(batch.rt), len(batch.reqs))
			// 分别处理起始stage的请求，将其发送到目标设备
			s := ""
			for _, ri := range startStageReqs {
				ri.startTag = sche.getTagAndAdd(1)
				go startInfer(ri, batch, ri.stageIDs, stage.maxBS, stage.fillTimeOut)
				s += ri.id + ", "
			}
			log.Printf("startStageReqs of batch %s of stage %s len: %d with reqs: %s", batch.id, ids, len(startStageReqs), s)

			// 分别处理每个中间stage的请求
			s = ""
			for _, ri := range midStageReqs {
				s += ri.id + ", "
				// ri.startTag = sche.getTagAndAdd(stage.numOutput)
				ri.startTag = sche.getTagAndAdd(1) // 所有向量打包成一个[]byte了，所以这里只用+1
				ri.recvDevice = batch.device
				ri.batchID = batch.id
				// 通知该请求的转发目标
				ri.action <- pb.TargetAction_FORWARD
			}
			log.Printf("midStageReqs of %s len: %d with reqs: %s", ids, len(midStageReqs), s)

			// batch已满，可以执行了
			if closed {
				sche.finalizeBatch(batch, "full")
			}
		} else {
			// batch没有可用设备，丢弃并通知
			log.Printf("batch %s discard", batch.id)
			for _, ri := range startStageReqs {
				ri.finalResp <- 1
			}
			for _, ri := range midStageReqs {
				ri.action <- pb.TargetAction_DISCARD // 通知 HandleMidStageReq
				ri.finalResp <- 1                    // 通知 HandleRequest
			}
		}
	}
}

// 对于新请求，根据所属stage用一个队列装下它们，然后每个stage一个协程串行地一次性处理多个对应请求，同时防止同步问题

// 返回状态和数据，状态: 0代表正常，1代表丢弃
func (sche *Scheduler) HandleRequest(req *PredictRequest) (int32, []byte) {
	mi := sche.getModel(req.ModelID)
	if mi == nil {
		log.Printf("ERROR: no model: %s", req.ModelID)
		return 1, nil
	}

	stageSeq := sche.getStageSequence(req.ModelID, mi.startStageID)

	ri := requestInfo{
		id:         "request-" + xid.New().String(),
		tArrive:    getTime(),
		slo:        mi.slo,
		stageIDs:   stageSeq.stageIDs,
		modelID:    req.ModelID,
		data:       req.Input,
		action:     make(chan pb.TargetAction, 1), // 留一个缓冲，不然会造成stageWorker死锁
		finalResp:  make(chan int32, 1),
		stageTimes: make(map[string]StageTime),
	}
	log.Printf("Get new request %s of model %s with data len %d", ri.id, ri.modelID, len(ri.data))
	ri.rt = ri.tArrive + ri.slo*0.8 - stageSeq.latency
	ri.discardLine = ri.tArrive + ri.slo*2 - stageSeq.latency
	log.Printf("Rt of request %s in stage %s: %s", ri.id, fmt.Sprint(ri.stageIDs), getTimeStr(ri.rt))
	sche.requests.Store(ri.id, &ri)
	// 放到每个阶段的处理队列中，通过 stageWorker 处理
	stageSeq.taskChan <- &ri
	// 阻塞等待结果
	state := <-ri.finalResp
	log.Printf("Finish request %s of model %s", ri.id, ri.modelID)
	sche.requests.Delete(ri.id)
	// log.Println(sche.debugInfo())
	return state, ri.data
}

func (sche *Scheduler) HandleMidStageReq(batchID string, reqIDs []string) ([]pb.TargetAction, []string, []uint32, error) {
	batch := sche.getBatch(batchID)
	if batch == nil {
		log.Printf("ERROR: no batch: %s", batchID)
		return nil, nil, nil, fmt.Errorf("no batch: %s", batchID)
	}

	// 这个batch已经开始推理了，从stage的队列中移除
	batch.stageSeq.batchMu.Lock()
	batch.stageSeq.queueingBatch.Delete(batch)
	batch.stageSeq.batchMu.Unlock()

	action := make([]pb.TargetAction, len(reqIDs))
	ip := make([]string, len(reqIDs))
	startTags := make([]uint32, len(reqIDs))
	batch.reqMu.Lock() // 此时可能有其它协程也在处理推理该batch请求的任务
	reqs := []*requestInfo{}
	for i, reqID := range reqIDs {
		// log.Printf("midstage req %s start deal (%s %d/%d)", reqID, batchID, i+1, len(reqIDs))
		req := batch.reqs[reqID]
		reqs = append(reqs, req)
		nextStageSeq := sche.getNextStageSequence(req.modelID, req.stageIDs[len(req.stageIDs)-1])
		if nextStageSeq == nil { // 最后一个stage
			action[i] = pb.TargetAction_FINISH
			log.Printf("request %s finish", req.id)
		} else {
			log.Printf("request %s from %s to %s", req.id, req.stageIDs, fmt.Sprintln(nextStageSeq.stageIDs))
			req.stageIDs = nextStageSeq.stageIDs
			req.rt = req.tArrive + req.slo - nextStageSeq.latency
			nextStageSeq.taskChan <- req
		}
	}

	batch.reqMu.Unlock()
	// log.Println(sche.debugInfo())

	// 并行阻塞等待结果
	var wg sync.WaitGroup
	for i, req := range reqs {
		if action[i] != pb.TargetAction_FINISH {
			wg.Add(1)
			go func() {
				defer wg.Done()
				action[i] = <-req.action
				if action[i] == pb.TargetAction_DISCARD {
					return // 丢弃
				}
				ip[i] = req.recvDevice.ipOfMachine + ":50053"
				startTags[i] = req.startTag
				nextStage := sche.getStage(req.stageIDs[0])
				// 通过 rpc SetTarget 通知接收端
				rpcReq := &pb.SetTargetRequest{
					RequestId:         req.id,
					StageId:           req.stageIDs,
					BatchId:           req.batchID,
					Rt:                float32(req.rt),
					DeviceNum:         req.recvDevice.deviceNumOnMachine,
					StartTag:          req.startTag,
					MaxBs:             nextStage.maxBS,
					FollowServerInfer: nextStage.fillTimeOut != 0,
				}
				go req.recvDevice.rpc.SetTargetDeviceServiceClient.SetTarget(context.Background(), rpcReq)
			}()
		}
	}
	wg.Wait()

	log.Printf("finish midstage req from batch %s with reqs: %s", batchID, fmt.Sprint(reqIDs))
	return action, ip, startTags, nil
}

func (sche *Scheduler) HandleFinishBatchRequest(rpcReq *pb.FinishBatchRequest) int32 {
	batchID := rpcReq.GetBatchId()
	reqIDs := rpcReq.GetRequestIds()
	requestTimes := rpcReq.GetRequestTimes()
	startInferTime := rpcReq.GetStartInferTime()
	finishInferTime := rpcReq.GetFinishInferTime()
	finishPostProcessTime := rpcReq.GetFinishPostProcessTime()

	batch := sche.getBatch(batchID)
	batch.reqMu.Lock()
	stageID := batch.stageSeq.stageIDs[0]
	for i, reqID := range reqIDs {
		if sche.debug {
			req := sche.getReq(reqID)
			if req != nil {
				req.stageTimeMu.Lock()
				req.stageTimes[stageID] = StageTime{
					recvTime:              requestTimes[i].GetRecvTime(),
					startQueueingTime:     requestTimes[i].GetStartQueueingTime(),
					startInferTime:        startInferTime,
					finishInferTime:       finishInferTime,
					finishPostProcessTime: finishPostProcessTime,
					sendTime:              requestTimes[i].GetSendTime(),
				}
				req.stageTimeMu.Unlock()
			}
		}
		delete(batch.reqs, reqID) // 从batch中删除对应req，因为可能由于传输同步问题，设备先只处理了一部分req，没有全部处理
	}
	if len(batch.reqs) == 0 {
		batch.device.batchMu.Lock()
		batch.device.queueingBatch.Delete(batch)
		batch.device.batchMu.Unlock()
		log.Printf("batch %s remove from device %s", batch.id, batch.device.id)

		sche.batchs.Delete(batchID)
	}
	batch.reqMu.Unlock()
	return 0
}
func (sche *Scheduler) AnalyzeReqTime(req *requestInfo, recvTime float64) {
	model := sche.getModel(req.modelID)
	stageSeqID := sche.nextStageWithThis(model.startStageID, req.modelID)
	t := req.stageTimes[stageSeqID[0]]

	req.stageTimeMu.Lock()
	sche.debugStageTimesMu.Lock()

	sum := []float64{0, 0, 0, 0, 0, 0}
	appendThreshold := 1e5 // 预设阈值
	var appendData [][]float64
	exceedsThreshold := false

	strStageSeqIDs := fmt.Sprint(stageSeqID)
	stageIDs := []string{strStageSeqIDs}

	stageMap, ok := sche.debugStageTimes[req.modelID]
	if !ok {
		stageMap = make(map[string]*StageTimes)
		sche.debugStageTimes[req.modelID] = stageMap
	}

	for {
		log.Printf("AnalyzeReqTime: req time of req %s in stage %s: %s", req.id, strStageSeqIDs, fmt.Sprintln(t))
		times, ok := stageMap[strStageSeqIDs]
		if !ok {
			times = NewStageTimes(sche.debugTimeMaxSize)
			stageMap[strStageSeqIDs] = times
		}

		nextStageIDs := sche.nextStage(stageSeqID[len(stageSeqID)-1], req.modelID)

		var data []float64
		if len(nextStageIDs) == 0 {
			data = []float64{
				t.startQueueingTime - t.recvTime,
				t.startInferTime - t.startQueueingTime,
				t.finishInferTime - t.startInferTime,
				t.finishPostProcessTime - t.finishInferTime,
				t.sendTime - t.finishPostProcessTime,
				recvTime - t.sendTime,
			}
			sum[0] += data[0]
			sum[1] += data[1]
			sum[2] += data[2]
			sum[3] += data[3]
			sum[4] += data[4]
			sum[5] += data[5]
			appendData = append(appendData, data)
			break
		} else {
			strStageSeqIDs = fmt.Sprint(nextStageIDs)
			stageIDs = append(stageIDs, strStageSeqIDs)
			nextT := req.stageTimes[nextStageIDs[0]]
			data = []float64{
				t.startQueueingTime - t.recvTime,
				t.startInferTime - t.startQueueingTime,
				t.finishInferTime - t.startInferTime,
				t.finishPostProcessTime - t.finishInferTime,
				t.sendTime - t.finishPostProcessTime,
				nextT.recvTime - t.sendTime,
			}
			sum[0] += data[0]
			sum[1] += data[1]
			sum[2] += data[2]
			sum[3] += data[3]
			sum[4] += data[4]
			sum[5] += data[5]
			appendData = append(appendData, data)
			t = nextT
			stageSeqID = nextStageIDs
		}

		// 检查是否有数据超出阈值
		for i, value := range data {
			if value > appendThreshold || value < -appendThreshold {
				log.Printf("ERROR: AnalyzeReqTime Skipping append due to value exceeding threshold: %d: %f \n %s recvTime: %f \n nextStageIDs: %s", i, value, fmt.Sprint(req.stageTimes), recvTime, fmt.Sprint(nextStageIDs))
				exceedsThreshold = true
				appendData = nil // 取消所有append计划
				break
			}
		}
		if exceedsThreshold {
			break
		}
	}

	// 如果没有超出阈值，统一append
	if !exceedsThreshold {
		for i, data := range appendData {
			times := stageMap[stageIDs[i]]
			times.append(data[0], data[1], data[2], data[3], data[4], data[5])
		}

		sche.debugModelTimesMu.Lock()
		defer sche.debugModelTimesMu.Unlock()
		modelMap, ok := sche.debugModelTimes[req.modelID]
		if !ok {
			modelMap = NewStageTimes(sche.debugTimeMaxSize)
			sche.debugModelTimes[req.modelID] = modelMap
		}
		modelMap.append(sum[0], sum[1], sum[2], sum[3], sum[4], sum[5])
	}

	req.stageTimeMu.Unlock()
	sche.debugStageTimesMu.Unlock()
}

// 返回状态：0：正常结束；1：未找到此请求
func (sche *Scheduler) HandleFinishRequest(rpcReq *pb.FinishInferRequest) int32 {
	req := sche.getReqAndDelete(rpcReq.GetRequestId())
	if req == nil {
		return 1
	}
	req.data = rpcReq.GetTensors()
	req.finalResp <- 0
	if sche.debug {
		recvTime := getTime()
		batch := sche.getBatch(rpcReq.GetBatchId())
		if batch == nil {
			log.Printf("ERROR: HandleFinishRequest fail to find batch %s of req %s", rpcReq.GetBatchId(), req.id)
			return 0
		}

		req.stageTimeMu.Lock()
		req.stageTimes[batch.stageSeq.stageIDs[0]] = StageTime{
			recvTime:              rpcReq.GetRequestTime().GetRecvTime(),
			startQueueingTime:     rpcReq.GetRequestTime().GetStartQueueingTime(),
			startInferTime:        rpcReq.GetStartInferTime(),
			finishInferTime:       rpcReq.GetFinishInferTime(),
			finishPostProcessTime: rpcReq.GetFinishPostProcessTime(),
			sendTime:              rpcReq.GetRequestTime().GetSendTime(),
		}
		req.stageTimeMu.Unlock()
		sche.AnalyzeReqTime(req, recvTime)
	}
	return 0
}

// 每个stage建立一个对应协程
func (sche *Scheduler) Start() {
	// 启动一个新的 Goroutine，每分钟调用一次 debugInfo
	go func() {
		for {
			time.Sleep(1 * time.Minute) // 每分钟执行一次
			log.Println(sche.debugInfo())
		}
	}()
}
