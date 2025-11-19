package main

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"os"
	"runtime"
	"sync"
	"time"

	pb "compi_go_worker/rpc_server"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type WorkerManager struct {
	serverAddr       string
	selfAddr         string
	grpcURL          string
	rpcServerWorkers int
	maxDataSize      int
	modelPath        map[string]string
	staticDep        [][]string
	deviceNum        int
	startRank        int
	maxMem           []int64
	grpcConn         *grpc.ClientConn
	registerStub     pb.RegisterDeviceServiceClient
	getTargetStub    pb.GetTargetDeviceServiceClient
	finishInferStub  pb.FinishInferServiceClient
	finishBatchStub  pb.FinishBatchServiceClient
	cancelAll        context.CancelFunc
	wg               sync.WaitGroup

	otherWorkerTCPConn   map[string]net.Conn // ip -> conn
	otherWorkerTCPConnMu map[string]*sync.Mutex
	connMapMu            sync.Mutex

	workers  []*GPUWorker
	recvMu   sync.RWMutex
	recvReq  map[uint32]*pb.SetTargetRequest
	recvData map[uint32][]byte

	followServerBatch bool
	autoMergeBatch    bool
	fixedMaxBatch     bool
	useCUDAGraph      bool
	useQuantization   bool

	debug bool
}

func NewWorkerManager(configPath string) (*WorkerManager, error) {
	cfgFile, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	cfg := struct {
		ServerAddr        string            `json:"server_addr"`
		SelfAddr          string            `json:"self_addr"`
		RPCServerWorkers  int               `json:"rpc_server_workers"`
		MaxDataSizeMB     int               `json:"max_data_size"`
		GPUNum            int               `json:"gpu_num"`
		GPUMem            []int64           `json:"gpu_mem"`
		ModelPath         map[string]string `json:"model_path"`
		FollowServerBatch bool              `json:"follow_server_batch"`
		UseQuantization   bool              `json:"use_quantization"` // 是否启用输入输出的解量化和量化
		AutoMergeBatch    bool              `json:"auto_merge_batch"` // 允许推理时自动合并能合并的小批次成一个大批次
		FixedMaxBatch     bool              `json:"fixed_max_batch"`  // 是否保证每次推理都是推固定的最大batchsize，以避免模型切换的开销
		UseCUDAGraph      bool              `json:"use_cuda_graph"`   // 是否保证每次推理都是推固定的最大batchsize，以避免模型切换的开销
		Debug             bool              `json:"debug"`
	}{
		UseQuantization:   false,
		FollowServerBatch: false,
		AutoMergeBatch:    false,
		FixedMaxBatch:     false,
		UseCUDAGraph:      false,
		Debug:             false,
	}

	err = json.Unmarshal(cfgFile, &cfg)
	if err != nil {
		return nil, err
	}

	conn, err := grpc.NewClient(cfg.ServerAddr+":50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	log.Printf("Connected to server at %s\n", cfg.ServerAddr)

	return &WorkerManager{
		serverAddr:       cfg.ServerAddr,
		selfAddr:         cfg.SelfAddr,
		grpcURL:          cfg.ServerAddr + ":50051",
		rpcServerWorkers: cfg.RPCServerWorkers,
		maxDataSize:      cfg.MaxDataSizeMB * 1024 * 1024,
		modelPath:        cfg.ModelPath,
		deviceNum:        cfg.GPUNum,
		maxMem:           cfg.GPUMem,
		grpcConn:         conn,
		registerStub:     pb.NewRegisterDeviceServiceClient(conn),
		getTargetStub:    pb.NewGetTargetDeviceServiceClient(conn),
		finishInferStub:  pb.NewFinishInferServiceClient(conn),
		finishBatchStub:  pb.NewFinishBatchServiceClient(conn),

		otherWorkerTCPConn:   make(map[string]net.Conn),
		otherWorkerTCPConnMu: make(map[string]*sync.Mutex),

		workers:  make([]*GPUWorker, cfg.GPUNum),
		recvReq:  make(map[uint32]*pb.SetTargetRequest),
		recvData: make(map[uint32][]byte),

		useQuantization:   cfg.UseQuantization,
		followServerBatch: cfg.FollowServerBatch,
		autoMergeBatch:    cfg.AutoMergeBatch,
		fixedMaxBatch:     cfg.FixedMaxBatch,
		useCUDAGraph:      cfg.UseCUDAGraph,

		debug: cfg.Debug,
	}, nil
}

func (m *WorkerManager) Register() bool {
	if m.deviceNum < 1 {
		return false
	}
	req := &pb.RegisterDeviceRequest{
		DeviceNum: int32(m.deviceNum),
		Ip:        m.selfAddr,
		MaxMem:    m.maxMem,
	}
	resp, err := m.registerStub.RegisterDevice(context.Background(), req)
	if err != nil || resp.StartRank == -1 {
		log.Println("❌ Failed to register with server:", err)
		return false
	}

	m.startRank = int(resp.StartRank)

	for _, stages := range resp.StaticDeployment {
		stageList := make([]string, len(stages.Stage))
		copy(stageList, stages.Stage)
		m.staticDep = append(m.staticDep, stageList)
	}

	log.Printf("✅ Registered: start_rank=%d\n", m.startRank)
	return true
}

func (m *WorkerManager) recvSetTargetReq(req *pb.SetTargetRequest) {
	tag := req.GetStartTag()
	log.Printf("recv SetTarget gRPC request of req %s of batch %s with tag %d", req.RequestId, req.BatchId, tag)
	m.recvMu.Lock()
	data, ok := m.recvData[tag]
	if ok {
		delete(m.recvData, tag)
		m.recvMu.Unlock()
		m.workers[req.GetDeviceNum()].AddReqToBatchInReadyToInferQueue(req, data)
	} else {
		m.recvReq[tag] = req
		m.recvMu.Unlock()
	}
}

func (m *WorkerManager) recvStartInfer(req *pb.StartInferRequest) {
	log.Printf("recv StartInfer gRPC request of batch %s", req.GetBatchId())
	dev := int(req.GetDeviceNum())
	w := m.workers[dev]
	w.SetFinalBS(req.GetBatchId(), int(req.GetFinalBs()))
}

func (m *WorkerManager) recvDataWithTag(tag uint32, data []byte) {
	log.Printf("recv tcp data with len %d of tag %d", len(data), tag)
	m.recvMu.Lock()
	req, ok := m.recvReq[tag]
	if ok {
		delete(m.recvReq, tag)
		m.recvMu.Unlock()
		m.workers[req.GetDeviceNum()].AddReqToBatchInReadyToInferQueue(req, data)
	} else {
		m.recvData[tag] = data
		m.recvMu.Unlock()
	}
}

type SendFuncType func(ip string, id uint32, data []byte)

func (m *WorkerManager) sendData(ip string, id uint32, data []byte) {
	var conn net.Conn
	var err error
	m.connMapMu.Lock()
	mu, ok := m.otherWorkerTCPConnMu[ip]
	if !ok {
		mu = &sync.Mutex{}
		m.otherWorkerTCPConnMu[ip] = mu
		// 建立新连接
		conn, err = ConnectTCP(ip)
		if err != nil {
			m.connMapMu.Unlock()
			log.Printf("ERROR: Failed to connect to device %s: %v", ip, err)
			return
		}
		m.otherWorkerTCPConn[ip] = conn
	}
	m.connMapMu.Unlock()

	mu.Lock()
	conn = m.otherWorkerTCPConn[ip]
	for range 3 { // 最多尝试3次
		if conn == nil {
			// 重新建立新连接
			conn, err = ConnectTCP(ip)
			if err != nil {
				log.Printf("ERROR: Failed to connect to device %s: %v", ip, err)
				time.Sleep(time.Millisecond * 500) // 每 500ms 尝试一次
				continue
			}
		}
		err = SendBytesWithID(conn, id, data)
		if err != nil {
			log.Printf("ERROR: send data %d to device %s failed: %v", id, ip, err)
			conn = nil
		} else {
			break
		}

	}
	mu.Unlock()
}

func (m *WorkerManager) Start() bool {
	ctx, cancel := context.WithCancel(context.Background())
	m.cancelAll = cancel

	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(m.maxDataSize * 1024 * 1024),
		grpc.MaxSendMsgSize(m.maxDataSize * 1024 * 1024),
		grpc.NumStreamWorkers(uint32(runtime.NumCPU())),
		grpc.MaxConcurrentStreams(128),
	}

	// 启动 grpc 服务器
	StartGRPCServer(ctx, &m.wg, "0.0.0.0:50052", func(s *grpc.Server) {
		serverStruct := &WorkerRpcServer{
			workerManager: m,
		}
		pb.RegisterSetTargetDeviceServiceServer(s, serverStruct)
		pb.RegisterStartInferServiceServer(s, serverStruct)
	}, opts...)

	// 启动 TCP 服务器
	StartTCPServer(ctx, &m.wg, "0.0.0.0:50053", func(c net.Conn) {
		for {
			if isContextDone(ctx) {
				break
			}
			id, data, err := ReceiveBytesWithID(c)
			if err != nil {
				log.Printf("recieve tcp data of id %d error: %v", id, err)
				break
			}
			m.recvDataWithTag(id, data)
		}
	})

	if !m.Register() {
		return false
	}

	for deviceID := range m.deviceNum {
		worker := NewGPUWorker(deviceID, m.startRank+deviceID, m.getTargetStub, m.finishInferStub, m.finishBatchStub,
			m.sendData, m.debug, m.followServerBatch, m.autoMergeBatch, m.fixedMaxBatch, m.useCUDAGraph, m.useQuantization)
		m.workers[deviceID] = worker
		worker.Start(ctx, &m.wg)
		// static 部署
		for _, stage := range m.staticDep[deviceID] {
			go worker.PutModelControlCommand(&ModelControlCommand{
				Action:    LOAD,
				ModelID:   stage,
				ModelPath: m.modelPath[stage],
				WarmUp:    true,
			})
		}
	}
	log.Println("✅ All workers started")
	return true
}

func (m *WorkerManager) Shutdown() {
	if m.cancelAll != nil {
		m.cancelAll()
	}
	log.Println("🛑 All workers shutdown requested")
}

func (m *WorkerManager) Join() {
	m.wg.Wait()
	log.Println("✅ All workers joined")
}
