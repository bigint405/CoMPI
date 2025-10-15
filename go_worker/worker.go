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

	ortTensors              []*ORTTensor
	releaseFuncsBeforeInfer []func()
	releaseFuncsAfterInfer  []func()

	loadMu     sync.Mutex
	loaded     bool
	loadFailed bool
	loadCond   *sync.Cond

	releaseMu sync.Mutex

	targetAction chan []int32
	targetIP     []string
	targetTag    []uint32

	requestTimes          []WorkerRequestTime
	startInferTime        float64
	finishInferTime       float64
	finishPostProcessTime float64
	sendTime              float64
}

func NewWorkerBatch(req *pb.SetTargetRequest) *workerBatch {
	b := &workerBatch{
		id:         req.GetBatchId(),
		idOnDevice: req.GetBatchId() + "-" + xid.New().String(),
		modelID:    req.GetStageId(),
		datas:      make([][]byte, req.GetMaxBs())[:0],
		reqIDs:     make([]string, req.GetMaxBs())[:0],
		maxBS:      int(req.GetMaxBs()),
		rt:         float64(req.GetRt()),

		targetAction: make(chan []int32, 1),

		requestTimes: make([]WorkerRequestTime, req.GetMaxBs()),
	}
	b.loadCond = sync.NewCond(&b.loadMu)
	return b
}

func (a *workerBatch) Less(b btree.Item) bool {
	if a.rt == b.(*workerBatch).rt {
		return a.id < b.(*workerBatch).id
	}
	return a.rt < b.(*workerBatch).rt
}

func (batch *workerBatch) isFull() bool {
	return batch.bs >= batch.maxBS
}

func (batch *workerBatch) toDevice(deviceID int) {
	log.Printf("batch %s start to load on device %d", batch.idOnDevice, deviceID)
	batch.loadMu.Lock()
	defer batch.loadMu.Unlock()
	if batch.loaded && !batch.loadFailed {
		return
	}
	var dataBatch [][][]float32
	var shapeBatch [][]int64
	// 解量化
	for i, data := range batch.datas {
		tensors, _, shapes, releaseFunc, err := DequantizeFBGEMM(data)
		batch.releaseFuncsBeforeInfer = append(batch.releaseFuncsBeforeInfer, releaseFunc)
		if err != nil {
			log.Printf("ERROR: Dequantize error of batch %s: %v", batch.idOnDevice, err)
			batch.loaded = true
			batch.loadFailed = true
			batch.loadCond.Broadcast()
			return
		}
		if i == 0 {
			dataBatch = make([][][]float32, len(tensors))
			for j := range len(tensors) {
				dataBatch[j] = make([][]float32, batch.bs)
			}
			shapeBatch = shapes
			for j := range len(shapeBatch) {
				shapeBatch[j][0] = int64(batch.bs)
			}
		}
		for j, d := range tensors {
			dataBatch[j][i] = d // 指针复制，基本没开销
		}
	}
	// 转成 ORTTensor 切片
	batch.ortTensors = make([]*ORTTensor, len(dataBatch))
	for i, data := range dataBatch {
		t, releaseFunc, err := CreateTensorBatchFromGo(deviceID, data, shapeBatch[i])
		batch.releaseFuncsBeforeInfer = append(batch.releaseFuncsBeforeInfer, releaseFunc)
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

	getTargetStub   pb.GetTargetDeviceServiceClient
	finishInferStub pb.FinishInferServiceClient
	finishBatchStub pb.FinishBatchServiceClient
	sendData        SendFuncType

	debug bool
}

func NewGPUWorker(deviceID int, rank int, getTargetStub pb.GetTargetDeviceServiceClient, finishInferStub pb.FinishInferServiceClient, finishBatchStub pb.FinishBatchServiceClient, sendData SendFuncType, debug bool) *GPUWorker {
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

		debug: debug,
	}
	w.inferCond = sync.NewCond(&w.readyToInferQueueMu)
	return w
}

func (w *GPUWorker) getBatch(batchID string) *workerBatch {
	val, ok := w.batchs.Load(batchID)
	if !ok {
		return nil
	}
	return val.(*workerBatch)
}

func (w *GPUWorker) AddReqToBatchInReadyToInferQueue(req *pb.SetTargetRequest, data []byte) {
	var tRecv float64
	if w.debug {
		tRecv = getTime()
	}

	w.readyToInferQueueMu.Lock()
	defer w.readyToInferQueueMu.Unlock()

	batch := w.getBatch(req.GetBatchId())
	if batch == nil {
		// if req.GetRt() > float32(time.Now().UnixMicro())/1000 {
		// 	// 超时丢弃 ！！不能直接丢弃，会导致scheduler以为一直在推理！！
		// 	log.Printf("req %s of batch %s timeout, discard", req.RequestId, req.GetBatchId())
		// 	return
		// }
		// 到达这一步说明要么是一个新任务，要么这个任务已经被推理完了
		batch = NewWorkerBatch(req)

		log.Printf("creating new batch %s with req %s of stage %s and rt %s", batch.idOnDevice, req.RequestId, req.StageId, getTimeStr(batch.rt))
		batch.reqIDs = append(batch.reqIDs, req.GetRequestId())
		batch.bs++
		batch.datas = append(batch.datas, data)

		w.batchs.Store(batch.id, batch)
		w.readyToInferQueue.ReplaceOrInsert(batch)
		w.inferCond.Broadcast()
	} else {
		log.Printf("adding req %s to existing batch %s", req.RequestId, batch.idOnDevice)
		batch.reqIDs = append(batch.reqIDs, req.GetRequestId())
		batch.bs++
		batch.datas = append(batch.datas, data)
		if batch.rt > float64(req.GetRt())+worker_epsilon { // 需要更新rt
			w.readyToInferQueue.Delete(batch)
			batch.rt = float64(req.Rt)
			w.readyToInferQueue.ReplaceOrInsert(batch)
			log.Printf("updating batch %s rt to %s", batch.idOnDevice, getTimeStr(batch.rt))
		}
	}
	if w.debug {
		batch.requestTimes[batch.bs-1].recvTime = tRecv
		batch.requestTimes[batch.bs-1].startQueueingTime = getTime()
	}
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
				sess, err := LoadModel(cmd.ModelPath, w.deviceIDOnMachine)
				if err != nil {
					log.Printf("[GPU %d] load Model %s error", w.deviceIDOnMachine, modelID)
					continue
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

			batch.addReleaseFuncAfterInfer(f)
			batch.datas[i] = data
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
	actions := <-batch.targetAction

	for i, act := range actions {
		if act == int32(pb.TargetAction_DISCARD) {
			continue
		}
		wg.Add(1)
		if act == int32(pb.TargetAction_FINISH) {
			go func() {
				defer wg.Done()
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
			}()
		}
		if act == int32(pb.TargetAction_FORWARD) {
			go func() {
				defer wg.Done()
				if w.debug {
					batch.requestTimes[i].sendTime = getTime()
				}
				w.sendData(batch.targetIP[i], batch.targetTag[i], batch.datas[i])
			}()
		}
	}
	wg.Wait()
	w.finishBatch(batch)
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
	log.Printf("batch %s start to infer stages %s on device %d", batch.idOnDevice, fmt.Sprint(batch.modelID), w.deviceIDOnMachine)

	ortResults := batch.ortTensors
	var f func()
	var err error
	batch.startInferTime = getTime()
	for _, mID := range batch.modelID {
		session, ok := w.sessions[mID]
		if !ok {
			batch.loadMu.Unlock()
			return errors.New(fmt.Sprintf("ERROR: model not loaded on device %d", w.deviceIDOnMachine))
		}
		// 记录请求开始推理的时间
		log.Printf("batch %s start to infer stage %s on device %d", batch.idOnDevice, mID, w.deviceIDOnMachine)
		ortResults, f, err = RunInferenceFixedOutput(session, ortResults, int64(batch.bs))
		log.Printf("batch %s finish infering stage %s on device %d", batch.idOnDevice, mID, w.deviceIDOnMachine)
		batch.addReleaseFuncAfterInfer(f)
		if err != nil {
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
	rpcReq := &pb.GetTargetRequest{
		Rank:       int32(w.rank),
		BatchId:    batch.id,
		RequestIds: batch.reqIDs,
	}
	resp, err := w.getTargetStub.GetTarget(context.Background(), rpcReq)
	if err != nil {
		log.Printf("get target of batch %s error: %v", batch.idOnDevice, err)
		batch.targetAction <- nil // 丢弃batch
		return
	}
	batch.targetIP = resp.GetIp()
	batch.targetTag = resp.GetStartTags()
	batch.targetAction <- resp.GetAction()
}

func (w *GPUWorker) finishBatch(batch *workerBatch) {
	rpcReq := &pb.FinishBatchRequest{
		BatchId:    batch.id,
		RequestIds: batch.reqIDs,
	}
	if w.debug {
		requestTimes := make([]*pb.RequestTime, batch.bs)
		log.Printf("FinishBatch %s bs: %d, len(rt): %d", batch.idOnDevice, batch.bs, len(batch.requestTimes))
		for i := range batch.bs {
			rt := batch.requestTimes[i]
			requestTimes[i] = &pb.RequestTime{
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
					go batch.toDevice(w.deviceIDOnMachine)
					go w.getTarget(batch)
				}
			}

			if batch != nil {
				// 检查是否可以预加载下一个
				if peek := w.readyToInferQueue.Min(); peek != nil && peek.(*workerBatch).isFull() {
					nextBatch = w.readyToInferQueue.DeleteMin().(*workerBatch)
					w.batchs.Delete(nextBatch.id)
					log.Printf("device %d preloading batch %s", w.deviceIDOnMachine, nextBatch.idOnDevice)
					go nextBatch.toDevice(w.deviceIDOnMachine) // 预加载
					go w.getTarget(nextBatch)
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
