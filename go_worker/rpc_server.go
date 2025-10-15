package main

import (
	"compi_go_worker/rpc_server"
	"context"
)

type WorkerRpcServer struct {
	rpc_server.UnimplementedSetTargetDeviceServiceServer
	// 模型控制命令的tpc接口
	workerManager *WorkerManager
}

func (s *WorkerRpcServer) SetTarget(ctx context.Context, req *rpc_server.SetTargetRequest) (*rpc_server.SetTargetResponse, error) {
	s.workerManager.recvSetTargetReq(req)
	return &rpc_server.SetTargetResponse{Status: 0}, nil
}
