package main

import (
	pb "compi_go_worker/rpc_server"
	"context"
	"errors"
	"fmt"
	"log"
	"sync"

	"github.com/google/btree"
	"github.com/rs/xid"
)

type action int

const (
	LOAD action = iota
	UNLOAD
)

const worker_epsilon = 10 // 留出的误差空间

type WorkerRequestTime struct {
	recvTime          float64
	startQueueingTime float64
	sendTime          float64
}

type workerBatch struct {
	id         string
	idOnDevice string
	modelID    []string
	rt         float64
	datas      [][]byte
	reqIDs     []string
	maxBS      int
	bs         int

	finalBS int // StartInfer 带来的最终批大小；0 表示尚未收到

	ortTensors              []*ORTTensor
	releaseFuncsBeforeInfer []func()
	releaseFuncsAfterInfer  []func()

	loadMu     sync.Mutex
	loaded     bool
	loadFailed bool
	loadCond   *sync.Cond

	releaseMu sync.Mutex

	targetAction []chan pb.TargetAction
	targetIP     []string
	targetTag    []uint32

	requestTimes          []WorkerRequestTime
	startInferTime        float64
	finishInferTime       float64
	finishPostProcessTime float64

	originalBatch      []*workerBatch // 开启autoMergeBatch后，每个请求对应的原batch
	seqInOriginalBatch []int          // 开启autoMergeBatch后，每个请求对应的位于原batch的序号

	// 仅在 useCUDAGraph = true 时有效：
	// 0 或 1，表示当前 batch 绑定到哪一套 Graph slot（ping / pong）
	graphPingIndex int
}

func NewWorkerBatch(req *pb.SetTargetRequest) *workerBatch {
	b := &workerBatch{
		id:         req.GetBatchId(),
		idOnDevice: req.GetBatchId() + "-" + xid.New().String(),
		modelID:    req.GetStageId(),
		datas:      make([][]byte, 0, req.GetMaxBs()),
		reqIDs:     make([]string, 0, req.GetMaxBs()),
		maxBS:      int(req.GetMaxBs()),
		rt:         float64(req.GetRt()),

		targetAction: make([]chan pb.TargetAction, req.GetMaxBs()),
		targetIP:     make([]string, req.GetMaxBs()),
		targetTag:    make([]uint32, req.GetMaxBs()),

		requestTimes: make([]WorkerRequestTime, req.GetMaxBs()),
	}
	b.loadCond = sync.NewCond(&b.loadMu)
	return b
}

func (a *workerBatch) Less(ib btree.Item) bool {
	b := ib.(*workerBatch)
	if a.rt != b.rt {
		return a.rt < b.rt
	}
	return a.idOnDevice < b.idOnDevice
}

func (batch *workerBatch) isFull() bool {
	return batch.bs >= batch.maxBS
}

func (batch *workerBatch) addReq(reqID string, data []byte, originalBatch *workerBatch, originalSeq int) {
	batch.reqIDs[batch.bs] = reqID
	batch.datas[batch.bs] = data
	batch.originalBatch[batch.bs] = originalBatch
	batch.seqInOriginalBatch[batch.bs] = originalSeq
	if batch.targetAction[batch.bs] == nil {
		batch.targetAction[batch.bs] = make(chan pb.TargetAction, 1)
	}
	batch.bs++
}

func (batch *workerBatch) getOriginalBS() int {
	if batch.finalBS == 0 {
		return batch.bs
	}
	return batch.finalBS
}

func (batch *workerBatch) releaseBeforeInfer() {
	batch.loadMu.Lock()
	defer batch.loadMu.Unlock()
	log.Printf("batch %s start releaseBeforeInfer", batch.idOnDevice)
	for _, f := range batch.releaseFuncsBeforeInfer {
		if f != nil {
			f()
		}
	}
	batch.releaseFuncsBeforeInfer = nil
	batch.loaded = false
	batch.loadFailed = false
}

func (batch *workerBatch) addReleaseFuncAfterInfer(f func()) {
	if f != nil {
		batch.releaseMu.Lock()
		defer batch.releaseMu.Unlock()
		batch.releaseFuncsAfterInfer = append(batch.releaseFuncsAfterInfer, f)
	}
}

func (batch *workerBatch) releaseAfterInfer() {
	batch.releaseMu.Lock()
	defer batch.releaseMu.Unlock()
	log.Printf("batch %s start releaseAfterInfer", batch.idOnDevice)
	for _, f := range batch.releaseFuncsAfterInfer {
		f()
	}
	batch.releaseFuncsAfterInfer = nil
}

type ModelInput struct {
	ID   string
	Data []byte
}

type ModelControlCommand struct {
	ModelID   string
	ModelPath string
	Action    action
	WarmUp    bool
}

type GPUWorker struct {
	deviceIDOnMachine int
	rank              int
	sessionLock       sync.RWMutex
	sessions          map[string]*ORTSession

	modelControlChan chan *ModelControlCommand

	readyToInferQueueMu sync.RWMutex
	readyToInferQueue   *btree.BTree // batch_rt -> *WorkerBatch
	batchs              sync.Map     // batch_id(short) -> *WorkerBatch

	inferCond *sync.Cond
	warmUpMu  sync.Mutex

	getTargetStub   pb.GetTargetDeviceServiceClient
	finishInferStub pb.FinishInferServiceClient
	finishBatchStub pb.FinishBatchServiceClient
	sendData        SendFuncType

	// 仅在 useCUDAGraph = true 时使用：
	// 简单的 ping-pong 分配器，每次调用在 0 和 1 之间切换
	graphPingMu   sync.Mutex
	nextGraphPing int

	useQuantization   bool // 是否开启输入输出的解量化和量化
	followServerBatch bool // 是否严格按 server 批
	autoMergeBatch    bool // 是否自动合并小批，只有在 followServerBatch 开启后才有效，否则会出bug
	fixedMaxBatch     bool // 是否保证每次推理都是推固定的最大batchsize，以避免模型切换的开销
	useCUDAGraph      bool // 是否使用 CUDA Graph 减少batchsize切换的影响
	debug             bool
}

func NewGPUWorker(deviceID int, rank int, getTargetStub pb.GetTargetDeviceServiceClient, finishInferStub pb.FinishInferServiceClient,
	finishBatchStub pb.FinishBatchServiceClient, sendData SendFuncType, debug bool, followServerBatch bool, autoMergeBatch bool,
	fixedMaxBatch bool, useCUDAGraph bool, useQuantization bool) *GPUWorker {
	if useCUDAGraph && fixedMaxBatch {
		log.Printf("[GPU %d] useCUDAGraph is enabled, force fixedMaxBatch = false", deviceID)
		fixedMaxBatch = false
	}
	w := &GPUWorker{
		deviceIDOnMachine: deviceID,
		rank:              rank,
		sessions:          make(map[string]*ORTSession),

		modelControlChan:  make(chan *ModelControlCommand),
		readyToInferQueue: btree.New(3),

		getTargetStub:   getTargetStub,
		finishInferStub: finishInferStub,
		finishBatchStub: finishBatchStub,
		sendData:        sendData,

		followServerBatch: followServerBatch,
		autoMergeBatch:    autoMergeBatch,
		fixedMaxBatch:     fixedMaxBatch,
		useCUDAGraph:      useCUDAGraph,
		useQuantization:   useQuantization,

		debug: debug,
	}
	w.inferCond = sync.NewCond(&w.readyToInferQueueMu)
	return w
}

// only used when useCUDAGraph == true
func (w *GPUWorker) nextPingIndex() int {
	w.graphPingMu.Lock()
	defer w.graphPingMu.Unlock()

	idx := w.nextGraphPing
	if idx != 0 && idx != 1 {
		// 兜底，防止以后不小心改坏
		idx = 0
	}

	// 简单在 0 / 1 之间切换
	w.nextGraphPing = 1 - idx
	return idx
}

func (w *GPUWorker) getBatch(batchID string) *workerBatch {
	val, ok := w.batchs.Load(batchID)
	if !ok {
		return nil
	}
	return val.(*workerBatch)
}

// 创建或获取 batch；若不存在则创建“占位批”
func (w *GPUWorker) getOrCreateBatchLocked(batchID string) *workerBatch {
	b := w.getBatch(batchID)
	if b != nil {
		return b
	}
	// 不存在，新建
	b = &workerBatch{
		id:         batchID,
		idOnDevice: batchID + "-" + xid.New().String(),
		rt:         0, // 没有 rt 也无所谓：不可推理时不会被取走
		maxBS:      0, // 等首条 SetTarget 再补
		bs:         0,
		finalBS:    0,
	}
	b.loadCond = sync.NewCond(&b.loadMu)

	w.batchs.Store(batchID, b)
	return b
}

func (w *GPUWorker) SetFinalBS(batchID string, final int) {
	w.readyToInferQueueMu.Lock()
	defer w.readyToInferQueueMu.Unlock()

	b := w.getOrCreateBatchLocked(batchID)
	// 赋值/覆写 finalBS（一般不会变化；若变化按最新为准）
	b.finalBS = final
	// w.readyToInferQueue.ReplaceOrInsert(b) // 这里不insert，反正后面还会改的

	// 如果已经收齐，升级为可推理并重排
	w.CheckAutoMerge(b)
}

func sameModelChain(a, b *workerBatch) bool {
	if len(a.modelID) != len(b.modelID) {
		return false
	}
	for i := range a.modelID {
		if a.modelID[i] != b.modelID[i] {
			return false
		}
	}
	return true
}

func (w *GPUWorker) toDevice(batch *workerBatch, deviceID int) {
	log.Printf("batch %s start to load on device %d", batch.idOnDevice, deviceID)

	batch.loadMu.Lock()
	defer batch.loadMu.Unlock()

	// 已经成功加载过就不重复
	if batch.loaded && !batch.loadFailed {
		return
	}

	var dataBatch [][][]float32
	var shapeBatch [][]int64

	// 解码（量化 / 非量化）
	for i := range batch.bs {
		data := batch.datas[i]

		tensors, shapes, releaseFunc, err := DecodeVectorsFromBytes(data, w.useQuantization)
		if releaseFunc != nil {
			batch.releaseFuncsBeforeInfer = append(batch.releaseFuncsBeforeInfer, releaseFunc)
		}
		if err != nil {
			log.Printf("ERROR: decode input error of batch %s: %v", batch.idOnDevice, err)
			batch.loaded = true
			batch.loadFailed = true
			batch.loadCond.Broadcast()
			return
		}

		// 初始化 dataBatch / shapeBatch
		if dataBatch == nil {
			dataBatch = make([][][]float32, len(tensors))
			for j := range dataBatch {
				if w.fixedMaxBatch {
					dataBatch[j] = make([][]float32, batch.maxBS)
				} else {
					dataBatch[j] = make([][]float32, batch.bs)
				}
			}
			shapeBatch = shapes
			for j := range shapeBatch {
				if w.fixedMaxBatch {
					shapeBatch[j][0] = int64(batch.maxBS)
				} else {
					shapeBatch[j][0] = int64(batch.bs)
				}
			}
		}

		// 填入第 i 个样本的各个输入向量（只是 slice 指针赋值，不复制数据）
		for j, d := range tensors {
			dataBatch[j][i] = d
		}
	}

	// 空 batch 的兜底（理论上不会发生）
	if len(dataBatch) == 0 {
		batch.loaded = true
		batch.loadFailed = false
		batch.loadCond.Broadcast()
		return
	}

	// fixedMaxBatch 下，如果本批样本数小于 maxBS，尾部补 nil，作为哨兵
	if w.fixedMaxBatch && batch.bs < batch.maxBS {
		for j := range dataBatch {
			for k := batch.bs; k < batch.maxBS; k++ {
				dataBatch[j][k] = nil
			}
		}
	}

	// =========================
	//  分支一：CUDA Graph 模式
	// =========================
	if w.useCUDAGraph {
		// 约定：开启 CUDA Graph 的情况下只有单阶段模型
		if len(batch.modelID) == 0 {
			log.Printf("ERROR: batch %s has no modelID for CUDA Graph", batch.idOnDevice)
			batch.loaded = true
			batch.loadFailed = true
			batch.loadCond.Broadcast()
			return
		}
		modelID := batch.modelID[0]

		w.sessionLock.RLock()
		session, ok := w.sessions[modelID]
		w.sessionLock.RUnlock()
		if !ok || session == nil {
			log.Printf("ERROR: model %s not loaded for CUDA Graph on device %d", modelID, w.deviceIDOnMachine)
			batch.loaded = true
			batch.loadFailed = true
			batch.loadCond.Broadcast()
			return
		}

		// 分配 ping–pong 槽，并记录到 batch 上
		ping := w.nextPingIndex()
		batch.graphPingIndex = ping

		// 对每个输入 index 调用 CreateGraphInputBatchFromGo
		for inputIndex := range dataBatch {
			_, err := CreateGraphInputBatchFromGo(
				session,
				ping,
				inputIndex,
				dataBatch[inputIndex],
				shapeBatch[inputIndex],
			)
			if err != nil {
				log.Printf("ERROR: CreateGraphInputBatchFromGo error for batch %s, input %d: %v", batch.idOnDevice, inputIndex, err)
				batch.loaded = true
				batch.loadFailed = true
				batch.loadCond.Broadcast()
				return
			}
		}

		batch.loaded = true
		batch.loadFailed = false
		log.Printf("batch %s finish to load (CUDA Graph) on device %d ping %d", batch.idOnDevice, deviceID, ping)
		batch.loadCond.Broadcast()
		return
	}

	// =========================
	//  分支二：非 CUDA Graph 模式（原逻辑）
	// =========================
	batch.ortTensors = make([]*ORTTensor, len(dataBatch))
	for i := range dataBatch {
		data := dataBatch[i]
		t, releaseFunc, err := CreateTensorBatchFromGo(deviceID, data, shapeBatch[i])
		if releaseFunc != nil {
			batch.releaseFuncsBeforeInfer = append(batch.releaseFuncsBeforeInfer, releaseFunc)
		}
		if err != nil {
			log.Printf("ERROR: Create ort tensor error of batch %s: %v", batch.idOnDevice, err)
			batch.loaded = true
			batch.loadFailed = true
			batch.loadCond.Broadcast()
			return
		}
		batch.ortTensors[i] = t
	}

	batch.loaded = true
	batch.loadFailed = false
	log.Printf("batch %s finish to load on device %d", batch.idOnDevice, deviceID)
	batch.loadCond.Broadcast()
}

// 需要申请 w.readyToInferQueueMu 锁
func (w *GPUWorker) CheckAutoMerge(batch *workerBatch) {
	if w.followServerBatch && batch.finalBS > 0 {
		// 检查是否即收到了所有请求数据又收到了运行推理的控制信号或者已经满编
		// TODO：往已有batch里补满这个过程并没有考虑rt优先级的问题
		if (batch.finalBS > 0 && batch.bs >= batch.finalBS) || batch.bs >= batch.maxBS {
			if w.autoMergeBatch {
				i := batch.bs - 1
				w.readyToInferQueue.Ascend(func(it btree.Item) bool {
					b := it.(*workerBatch)
					// 目标batch也得是finalBS不为0的开启了follow的，如果是没开启follow的，那会影响scheduler判断
					if b != batch && b.finalBS != 0 && !b.isFull() && sameModelChain(b, batch) {
						for ; i >= 0 && !b.isFull(); i-- {
							b.addReq(batch.reqIDs[i], batch.datas[i], batch, i)
							batch.bs--
							if w.debug {
								// 转移debug信息
								b.requestTimes[b.bs-1].recvTime = batch.requestTimes[batch.bs].recvTime
								b.requestTimes[b.bs-1].startQueueingTime = batch.requestTimes[batch.bs].startQueueingTime
							}
							log.Printf("automerge: req to other batch: %s -> %s", batch.reqIDs[i], b.idOnDevice)
						}
					}
					return i >= 0
				})
			}
			// 没开启w.autoMergeBatch或者还剩下没分完的请求
			if batch.bs > 0 {
				log.Printf("batch ready to infer: %s", batch.idOnDevice)
				w.readyToInferQueue.ReplaceOrInsert(batch)
				w.inferCond.Broadcast()
			}
		}
	}
	// 没开启w.followServerBatch
	if !w.followServerBatch && batch.bs > 0 {
		log.Printf("batch ready to infer: %s", batch.idOnDevice)
		w.readyToInferQueue.ReplaceOrInsert(batch)
		w.inferCond.Broadcast()
	}
}

func (w *GPUWorker) AddReqToBatchInReadyToInferQueue(req *pb.SetTargetRequest, data []byte) {
	var tRecv float64
	if w.debug {
		tRecv = getTime()
	}

	w.readyToInferQueueMu.Lock()
	defer w.readyToInferQueueMu.Unlock()

	batch := w.getOrCreateBatchLocked(req.GetBatchId())
	if batch.bs == 0 {
		// if req.GetRt() > float32(time.Now().UnixMicro())/1000 {
		// 	// 超时丢弃 ！！不能直接丢弃，会导致scheduler以为一直在推理！！
		// 	log.Printf("req %s of batch %s timeout, discard", req.RequestId, req.GetBatchId())
		// 	return
		// }
		// 到达这一步说明要么是一个新任务，要么这个任务已经被推理完了（非followServerBatch模式下）
		batch.modelID = req.GetStageId()
		batch.datas = make([][]byte, req.GetMaxBs())
		batch.reqIDs = make([]string, req.GetMaxBs())
		batch.originalBatch = make([]*workerBatch, req.GetMaxBs())
		batch.seqInOriginalBatch = make([]int, req.GetMaxBs())
		batch.targetAction = make([]chan pb.TargetAction, req.GetMaxBs())
		batch.targetIP = make([]string, req.GetMaxBs())
		batch.targetTag = make([]uint32, req.GetMaxBs())
		batch.maxBS = int(req.GetMaxBs())
		batch.rt = float64(req.GetRt())
		batch.requestTimes = make([]WorkerRequestTime, req.GetMaxBs())

		log.Printf("creating new batch %s with req %s of stage %s and rt %s", batch.idOnDevice, req.RequestId, req.StageId, getTimeStr(batch.rt))
		batch.addReq(req.GetRequestId(), data, batch, batch.bs)

		if !w.followServerBatch || !req.GetFollowServerInfer() {
			w.readyToInferQueue.ReplaceOrInsert(batch) // 如果不是followServerBatch模式则不插入，因为并没有准备推理，得等后面判断
			w.inferCond.Broadcast()
		}
	} else {
		log.Printf("adding req %s to existing batch %s", req.RequestId, batch.idOnDevice)
		batch.addReq(req.GetRequestId(), data, batch, batch.bs)
		if batch.rt > float64(req.GetRt())+worker_epsilon { // 需要更新rt
			if !w.followServerBatch || !req.GetFollowServerInfer() {
				w.readyToInferQueue.Delete(batch)
			}
			batch.rt = float64(req.GetRt())
			if !w.followServerBatch || !req.GetFollowServerInfer() {
				w.readyToInferQueue.ReplaceOrInsert(batch)
			}
			log.Printf("updating batch rt of %s to %s", batch.idOnDevice, getTimeStr(batch.rt))
		}
	}

	// 记录新请求的debug信息
	if w.debug {
		batch.requestTimes[batch.bs-1].recvTime = tRecv
		batch.requestTimes[batch.bs-1].startQueueingTime = getTime()
	}

	w.CheckAutoMerge(batch)
}

func (w *GPUWorker) Start(ctx context.Context, wg *sync.WaitGroup) {
	wg.Add(2)
	go w.modelControlLoop(ctx, wg)
	go w.inferLoop(ctx, wg)
}

// 阻塞地放入模型控制命令
func (w *GPUWorker) PutModelControlCommand(c *ModelControlCommand) {
	w.modelControlChan <- c
}

func (w *GPUWorker) modelControlLoop(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()
	for {
		select {
		case <-ctx.Done():
			log.Printf("[GPU %d] Model control loop exit", w.deviceIDOnMachine)
			return
		case cmd := <-w.modelControlChan:
			modelID := cmd.ModelID
			if cmd.Action == LOAD {
				sess, err := LoadModel(cmd.ModelPath, w.deviceIDOnMachine, w.useCUDAGraph)
				if err != nil {
					log.Printf("[GPU %d] load Model %s error", w.deviceIDOnMachine, modelID)
					continue
				}
				if w.useCUDAGraph {
					SetSessionCUDAGraphEnabled(sess, true)
					if cmd.WarmUp {
						w.warmUpMu.Lock()
						// TODO warmup
						w.warmUpMu.Unlock()
					}
				}
				w.sessionLock.Lock()
				w.sessions[modelID] = sess
				w.sessionLock.Unlock()
				log.Printf("[GPU %d] Model loaded: %s", w.deviceIDOnMachine, modelID)
			} else if cmd.Action == UNLOAD {
				w.sessionLock.Lock()
				if s, ok := w.sessions[modelID]; ok {
					UnloadModel(s)
					delete(w.sessions, modelID)
					log.Printf("[GPU %d] Model unloaded: %s", w.deviceIDOnMachine, modelID)
				}
				w.sessionLock.Unlock()
			}
		}
	}
}

func (w *GPUWorker) postProcess(batch *workerBatch, ortResults []*ORTTensor) {
	datas := make([][][]float32, batch.bs)
	for i := range batch.bs {
		datas[i] = make([][]float32, len(ortResults))
	}
	shapes := make([][]int64, len(ortResults))

	defer batch.releaseBeforeInfer() // todo: 找出这些中不需要的提前删，直接提前删会导致一些东西释放早了
	defer batch.releaseAfterInfer()

	for j, ortResult := range ortResults {
		data, shape, f, err := GetTensorDataUnsafeSliceBatched(ortResult)
		batch.addReleaseFuncAfterInfer(f)
		if err != nil {
			log.Printf("post process infer result of batch %s error: %v", batch.idOnDevice, err)
			return
		}
		for i := range batch.bs {
			datas[i][j] = data[i] // 指针复制，不是值复制
		}
		shapes[j] = shape
		shapes[j][0] = 1
	}

	// 并行量化打包
	var wg sync.WaitGroup
	errCh := make(chan error, batch.bs) // 收集错误信息

	for i := range batch.bs {
		wg.Add(1)
		go func() {
			defer wg.Done()

			data, f, err := QuantizeVectorsFBGEMMBytes(datas[i], shapes)
			if err != nil {
				errCh <- fmt.Errorf("quantize result of batch %s error: %v", batch.idOnDevice, err)
				return
			}
			// 把结果放回原来存数据的位置，不放回原batch的原位置，因为原batch可能存有新的还没推理的数据（即代理了其它batch）
			batch.addReleaseFuncAfterInfer(f)
			batch.datas[i] = data
			// ob := batch.originalBatch[i]
			// ob.addReleaseFuncAfterInfer(f)
			// ob.datas[batch.seqInOriginalBatch[i]] = data
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		log.Printf("ERROR in post process: %v", err)
		return // 遇到错误就退出
	}
	if w.debug {
		batch.finishPostProcessTime = getTime()
	}

	//结果转发
	for i := range batch.bs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			act := <-batch.targetAction[i]
			if act == pb.TargetAction_DISCARD {
				return
			} else if act == pb.TargetAction_FINISH {
				rpcReq := &pb.FinishInferRequest{
					RequestId: batch.reqIDs[i],
					Tensors:   batch.datas[i],
				}
				if w.debug {
					batch.requestTimes[i].sendTime = getTime()
					rpcReq.RequestTime = &pb.RequestTime{
						RecvTime:          batch.requestTimes[i].recvTime,
						StartQueueingTime: batch.requestTimes[i].startQueueingTime,
						SendTime:          batch.requestTimes[i].sendTime,
					}
					rpcReq.StartInferTime = batch.startInferTime
					rpcReq.FinishInferTime = batch.finishInferTime
					rpcReq.FinishPostProcessTime = batch.finishPostProcessTime
					rpcReq.BatchId = batch.id
				}
				_, err := w.finishInferStub.FinishInfer(context.Background(), rpcReq)
				if err != nil {
					log.Printf("req %s of batch %s finish infer error: %v", batch.reqIDs[i], batch.idOnDevice, err)
				}
			} else if act == pb.TargetAction_FORWARD {
				if w.debug {
					batch.requestTimes[i].sendTime = getTime()
				}
				w.sendData(batch.targetIP[i], batch.targetTag[i], batch.datas[i])
			}
		}(i)
	}
	wg.Wait()
	w.finishBatch(batch)

	// var lastBatch *workerBatch
	// for _, b := range batch.originalBatch {
	// 	if b != lastBatch {
	// 		lastBatch = b
	// 		b.agentWG.Done()
	// 		go w.finishBatch(b)
	// 	}
	// }
}

func (w *GPUWorker) inferBatch(batch *workerBatch) error {
	batch.loadMu.Lock()
	if !batch.loaded {
		log.Printf("batch %s waiting to infer", batch.idOnDevice)
		batch.loadCond.Wait()
	}
	if batch.loadFailed {
		batch.loadMu.Unlock()
		return errors.New("ERROR: batch load failed")
	}
	log.Printf(
		"batch %s start to infer stages %s on device %d",
		batch.idOnDevice,
		fmt.Sprint(batch.modelID),
		w.deviceIDOnMachine,
	)

	batch.startInferTime = getTime()

	// =========================
	//  分支一：CUDA Graph 模式（单阶段）
	// =========================
	if w.useCUDAGraph {
		if len(batch.modelID) == 0 {
			batch.loadMu.Unlock()
			return fmt.Errorf("ERROR: no modelID for batch %s", batch.idOnDevice)
		}
		modelID := batch.modelID[0]

		w.sessionLock.RLock()
		session, ok := w.sessions[modelID]
		w.sessionLock.RUnlock()
		if !ok || session == nil {
			batch.loadMu.Unlock()
			return fmt.Errorf("ERROR: model %s not loaded on device %d",
				modelID, w.deviceIDOnMachine)
		}

		log.Printf(
			"batch %s start to infer (CUDA Graph) model %s on device %d, bs=%d, ping=%d",
			batch.idOnDevice,
			modelID,
			w.deviceIDOnMachine,
			batch.bs,
			batch.graphPingIndex,
		)

		// Graph 视角下的 batch size，这里直接使用实际 batch.bs
		// 这样不同的 bs 会对应不同的 GraphSlot（按需捕获 / 复用）

		ortResults, err := RunInferenceWithGraphSlot(
			session,
			int64(batch.bs),
			batch.graphPingIndex,
		)
		if err != nil {
			batch.loadMu.Unlock()
			return fmt.Errorf("ERROR: RunInferenceWithGraphSlot failed for batch %s: %w",
				batch.idOnDevice, err)
		}

		batch.finishInferTime = getTime()
		batch.loadMu.Unlock()

		// Graph 输出是 session 持有的 ORTTensor*，直接送去后处理
		go w.postProcess(batch, ortResults)
		return nil
	}

	// =========================
	//  分支二：非 CUDA Graph 模式（原逻辑，多阶段）
	// =========================
	ortResults := batch.ortTensors
	var f func()
	var err error

	for _, mID := range batch.modelID {
		w.sessionLock.RLock()
		session, ok := w.sessions[mID]
		w.sessionLock.RUnlock()
		if !ok || session == nil {
			batch.loadMu.Unlock()
			return fmt.Errorf("ERROR: model %s not loaded on device %d", mID, w.deviceIDOnMachine)
		}

		if w.fixedMaxBatch {
			log.Printf(
				"batch %s start to infer stage %s on device %d, with bs %d (fixedMaxBatch %d)",
				batch.idOnDevice, mID, w.deviceIDOnMachine, batch.bs, batch.maxBS,
			)
			ortResults, f, err = RunInferenceFixedOutput(session, ortResults, int64(batch.maxBS))
		} else {
			log.Printf(
				"batch %s start to infer stage %s on device %d, with bs %d",
				batch.idOnDevice, mID, w.deviceIDOnMachine, batch.bs,
			)
			ortResults, f, err = RunInferenceFixedOutput(session, ortResults, int64(batch.bs))
		}

		if f != nil {
			batch.addReleaseFuncAfterInfer(f)
		}
		if err != nil {
			batch.loadMu.Unlock()
			return err
		}
	}

	batch.finishInferTime = getTime()
	batch.loadMu.Unlock()

	go w.postProcess(batch, ortResults)
	return nil
}

// 目标查询
func (w *GPUWorker) getTarget(batch *workerBatch) {
	// 通过一个循环把请求按原batch分好进行getTarget询问，结果仍放到代理batch中
	var batchs []*workerBatch
	var lrb []int // lastReqBorder
	var lastID string
	for i := range batch.bs {
		b := batch.originalBatch[i]
		if i == 0 || b.idOnDevice != lastID {
			lastID = b.idOnDevice
			batchs = append(batchs, b)
			lrb = append(lrb, i)
		}
	}
	lrb = append(lrb, batch.bs)
	for i, b := range batchs {
		go func(i int, b *workerBatch) {
			rpcReq := &pb.GetTargetRequest{
				BatchId:    b.id,
				RequestIds: batch.reqIDs[lrb[i]:lrb[i+1]],
			}
			resp, err := w.getTargetStub.GetTarget(context.Background(), rpcReq)
			if err != nil {
				log.Printf("get target of batch %s (agent %s) error: %v", b.idOnDevice, batch.idOnDevice, err)
				// 丢弃batch
				for j := lrb[i]; j < lrb[i+1]; j++ {
					// b.targetAction[batch.seqInOriginalBatch[j]] <- pb.TargetAction_DISCARD
					batch.targetAction[j] <- pb.TargetAction_DISCARD
				}
				return
			}
			for j := lrb[i]; j < lrb[i+1]; j++ {
				k := j - lrb[i]
				// b.targetIP[batch.seqInOriginalBatch[j]] = resp.Ip[k]
				// b.targetTag[batch.seqInOriginalBatch[j]] = resp.StartTags[k]
				// b.targetAction[batch.seqInOriginalBatch[j]] <- resp.Action[k]
				batch.targetIP[j] = resp.Ip[k]
				batch.targetTag[j] = resp.StartTags[k]
				batch.targetAction[j] <- resp.Action[k]
			}
		}(i, b)
	}
}

func (w *GPUWorker) finishBatch(batch *workerBatch) {
	// 如果是代理batch，分别finish其中的每一个原batch的部分

	var batchs []*workerBatch
	var lrb []int // lastReqBorder
	var lastID string
	for i := range batch.bs {
		b := batch.originalBatch[i]
		if i == 0 || b.idOnDevice != lastID {
			lastID = b.idOnDevice
			batchs = append(batchs, b)
			lrb = append(lrb, i)
		}
	}
	lrb = append(lrb, batch.bs)

	for i, b := range batchs {
		go func(i int, b *workerBatch) {
			rpcReq := &pb.FinishBatchRequest{
				BatchId:    b.id,
				RequestIds: batch.reqIDs[lrb[i]:lrb[i+1]],
			}
			if w.debug {
				requestTimes := make([]*pb.RequestTime, lrb[i+1]-lrb[i])
				log.Printf("FinishBatch %s with client %s bs: %d, len(rt): %d", b.idOnDevice, batch.idOnDevice, batch.bs, len(batch.requestTimes))
				for j := lrb[i]; j < lrb[i+1]; j++ {
					k := j - lrb[i]
					rt := batch.requestTimes[j]
					requestTimes[k] = &pb.RequestTime{
						RecvTime:          rt.recvTime,
						StartQueueingTime: rt.startQueueingTime,
						SendTime:          rt.sendTime,
					}
				}
				rpcReq.RequestTimes = requestTimes
				rpcReq.StartInferTime = batch.startInferTime
				rpcReq.FinishInferTime = batch.finishInferTime
				rpcReq.FinishPostProcessTime = batch.finishPostProcessTime
			}
			_, err := w.finishBatchStub.FinishBatch(context.Background(), rpcReq)
			if err != nil {
				log.Printf("finish batch of batch %s error: %v", batch.idOnDevice, err)
			}
		}(i, b)
	}
}

func (w *GPUWorker) inferLoop(ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

	var nextBatch *workerBatch

	for {
		if isContextDone(ctx) {
			return
		}

		w.readyToInferQueueMu.Lock() // 保证此时没有在往batch里塞新请求，可以安心的toDevice和getTarget
		for {
			var batch *workerBatch

			if nextBatch != nil {
				batch = nextBatch
				nextBatch = nil
			} else {
				batchItem := w.readyToInferQueue.DeleteMin()
				if batchItem != nil {
					batch = batchItem.(*workerBatch)
					w.batchs.Delete(batch.id)
					go w.toDevice(batch, w.deviceIDOnMachine)
					go w.getTarget(batch)
				}
			}

			if batch != nil {
				// 检查是否可以预加载下一个
				peek := w.readyToInferQueue.Min()
				if peek != nil {
					peek := peek.(*workerBatch)
					if peek.isFull() || peek.bs >= peek.finalBS {
						nextBatch = w.readyToInferQueue.DeleteMin().(*workerBatch)
						w.batchs.Delete(nextBatch.id)
						log.Printf("device %d preloading batch %s", w.deviceIDOnMachine, nextBatch.idOnDevice)
						go w.toDevice(nextBatch, w.deviceIDOnMachine)
						go w.getTarget(nextBatch)
					}
				}
				log.Printf(
					"device %d start infer batch %s with batchsize %d and rt %s and queuelen %d",
					w.deviceIDOnMachine,
					batch.idOnDevice,
					batch.bs,
					getTimeStr(batch.rt),
					w.readyToInferQueue.Len(),
				)
				w.readyToInferQueueMu.Unlock()
				err := w.inferBatch(batch)
				if err != nil {
					log.Printf("infer batch %s failed: %v", batch.idOnDevice, err)
				} else {
					log.Printf("device %d finish infer batch %s with batchsize %d using time %.3e", w.deviceIDOnMachine, batch.idOnDevice, batch.bs, batch.finishInferTime-batch.startInferTime)
				}
				break
			}

			// 队列为空，等待条件
			w.inferCond.Wait()
			if isContextDone(ctx) {
				w.readyToInferQueueMu.Unlock()
				return
			}
		}
	}
}

func isContextDone(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}
