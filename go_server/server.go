package main

import (
	"bytes"
	"compi_go_server/rpc_server"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"runtime"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
)

type rpcServer struct {
	rpc_server.UnimplementedGetTargetDeviceServiceServer
	rpc_server.UnimplementedFinishInferServiceServer
	rpc_server.UnimplementedFinishBatchServiceServer
	rpc_server.UnimplementedRegisterDeviceServiceServer

	sche *Scheduler
}

func (s *rpcServer) GetTarget(ctx context.Context, req *rpc_server.GetTargetRequest) (*rpc_server.GetTargetResponse, error) {
	log.Printf("GetTarget request: batch %s, req_id: %s", req.GetBatchId(), fmt.Sprintln(req.GetRequestIds()))
	action, ip, startTags, err := s.sche.HandleMidStageReq(req.GetBatchId(), req.GetRequestIds(), req.GetRank())
	resp := &rpc_server.GetTargetResponse{Action: action, Ip: ip, StartTags: startTags}
	return resp, err
}

func (s *rpcServer) FinishInfer(ctx context.Context, req *rpc_server.FinishInferRequest) (*rpc_server.FinishInferResponse, error) {
	log.Printf("FinishInfer request: request_id=%s", req.RequestId)
	status := s.sche.HandleFinishRequest(req)
	return &rpc_server.FinishInferResponse{Status: status}, nil
}

func (s *rpcServer) FinishBatch(ctx context.Context, req *rpc_server.FinishBatchRequest) (*rpc_server.FinishBatchResponse, error) {
	log.Printf("FinishBatch request: batch %s, req_id: %s", req.GetBatchId(), fmt.Sprintln(req.GetRequestIds()))
	status := s.sche.HandleFinishBatchRequest(req)
	return &rpc_server.FinishBatchResponse{Status: status}, nil
}

func (s *rpcServer) RegisterDevice(ctx context.Context, req *rpc_server.RegisterDeviceRequest) (*rpc_server.RegisterDeviceResponse, error) {
	host := req.GetIp()
	log.Printf("RegisterDevice request from %s with device_num=%d", host, req.DeviceNum)
	if len(host) == 0 {
		log.Printf("Failed to get host IP")
		return nil, errors.New("host IP is empty")
	}
	sr, ws, sd, err := s.sche.RegisterDevice(host, req.DeviceNum, req.MaxMem)
	if err != nil {
		log.Printf("Failed to register device: %v", err)
		return nil, err
	}
	responseSD := rpc_server.RegisterDeviceResponse{
		StartRank:        sr,
		WorldSize:        ws,
		StaticDeployment: make([]*rpc_server.Stages, len(*sd)),
	}
	for i, stages := range *sd {
		responseSD.StaticDeployment[i] = &rpc_server.Stages{Stage: stages}
	}
	return &responseSD, err
}

// 启动 gRPC 服务器
func RunGRPCSserver(config *ServerConfig, sche *Scheduler) {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(config.MaxDataSize * 1024 * 1024),
		grpc.MaxSendMsgSize(config.MaxDataSize * 1024 * 1024),
		grpc.NumStreamWorkers(uint32(runtime.NumCPU())),
		grpc.MaxConcurrentStreams(128),
	}

	s := grpc.NewServer(opts...)
	serverStruct := rpcServer{
		sche: sche,
	}
	rpc_server.RegisterGetTargetDeviceServiceServer(s, &serverStruct)
	rpc_server.RegisterFinishInferServiceServer(s, &serverStruct)
	rpc_server.RegisterFinishBatchServiceServer(s, &serverStruct)
	rpc_server.RegisterRegisterDeviceServiceServer(s, &serverStruct)

	log.Println("gRPC server listening on port 50051")
	s.Serve(lis)
}

// 设置gin
func setupRouter(sche *Scheduler) *gin.Engine {
	r := gin.Default()
	r.POST("/predict", func(c *gin.Context) {
		var req PredictRequest
		req.ModelID = c.PostForm("model_id")
		if req.ModelID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "缺少 model_id"})
			return
		}

		// 读取量化后的 tensor 文件（field 名为 "tensor"）
		file, err := c.FormFile("tensor")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "缺少量化向量文件 (tensor)"})
			return
		}

		f, err := file.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "无法打开 tensor 文件"})
			return
		}
		defer f.Close()

		data, err := io.ReadAll(f)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "读取 tensor 文件失败"})
			return
		}
		req.Input = data
		state, data := sche.HandleRequest(&req)
		if state == 0 {
			var buf bytes.Buffer
			writer := multipart.NewWriter(&buf)
			part, _ := writer.CreateFormFile("file", "result.bin")
			part.Write(data) // 写入数据
			writer.Close()

			// **返回 multipart/form-data**
			c.Header("Content-Type", writer.FormDataContentType())
			c.Data(http.StatusOK, writer.FormDataContentType(), buf.Bytes())
		} else if state == 1 {
			log.Printf("request discard")
			c.String(http.StatusTooManyRequests, "Request discard")
		} else {
			c.String(http.StatusInternalServerError, "Unknown return state")
		}
	})

	r.GET("/export_debug_times", func(c *gin.Context) {
		exportDir := c.Query("dir") // 从查询参数获取导出目录
		if exportDir == "" {
			exportDir = "debug_times" // 默认导出目录
		}

		// 调用 sche.exportDebugTimes
		sche.exportDebugTimes(exportDir)

		// 返回成功响应
		c.JSON(http.StatusOK, gin.H{"message": "Debug times exported successfully", "directory": exportDir})
	})

	r.GET("/clear_debug_times", func(c *gin.Context) {
		sche.clearDebugTimes()

		// 返回成功响应
		c.JSON(http.StatusOK, gin.H{"message": "Debug times clear successfully"})
	})

	return r
}

func main() {
	// 允许多核
	runtime.GOMAXPROCS(runtime.NumCPU())
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	// 解析参数
	schedulerConfigPath := flag.String("sp", "../experiments/motivation/insight1_scheduler_config_1.json", "path of scheduler config file")
	flag.Parse()
	if len(*schedulerConfigPath) == 0 {
		log.Println("No config path")
		return
	}

	//解析scheduler config
	file, err := os.Open(*schedulerConfigPath)
	if err != nil {
		log.Println("Error opening scheduler config file:", err)
		return
	}
	data, err := io.ReadAll(file)
	if err != nil {
		log.Println("Error reading scheduler config file:", err)
		return
	}
	var schedulerConfig SchedulerConfig
	if err = json.Unmarshal(data, &schedulerConfig); err != nil {
		log.Println("Error decoding scheduler config file:", err)
		return
	}
	file.Close()

	//解析server config
	if file, err = os.Open("../configs/server_config.json"); err != nil {
		log.Println("Error opening server config file:", err)
		return
	}
	if data, err = io.ReadAll(file); err != nil {
		log.Println("Error reading server config file:", err)
		return
	}
	var serverConfig ServerConfig
	if err = json.Unmarshal(data, &serverConfig); err != nil {
		log.Println("Error decoding server config file:", err)
		return
	}
	file.Close()

	// 启动调度器
	var scheduler Scheduler
	scheduler.LoadConfig(&schedulerConfig)
	scheduler.Start()

	// 启动gRPC服务器
	go RunGRPCSserver(&serverConfig, &scheduler)

	// 启动http服务器
	r := setupRouter(&scheduler)
	server := &http.Server{
		Addr:    ":" + serverConfig.ServerPort,
		Handler: r,
		// ReadTimeout:    10 * time.Second, // 允许更长的读取时间
		// IdleTimeout:    30 * time.Second,
		// WriteTimeout:   10 * time.Second, // 允许更长的响应时间
		MaxHeaderBytes: 16 * 1024 * 1024,
	}
	server.ListenAndServe()
}
