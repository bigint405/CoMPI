package main

import (
	"compi_go_worker/rpc_server"
	"context"
)

type WorkerRpcServer struct {
	rpc_server.UnimplementedSetTargetDeviceServiceServer
	rpc_server.UnimplementedStartInferServiceServer

	workerManager *WorkerManager
}

func (s *WorkerRpcServer) SetTarget(ctx context.Context, req *rpc_server.SetTargetRequest) (*rpc_server.SetTargetResponse, error) {
	s.workerManager.recvSetTargetReq(req)
	return &rpc_server.SetTargetResponse{Status: 0}, nil
}

func (s *WorkerRpcServer) StartInfer(ctx context.Context, req *rpc_server.StartInferRequest) (*rpc_server.StartInferResponse, error) {
	s.workerManager.recvStartInfer(req)
	return &rpc_server.StartInferResponse{Status: 0}, nil
}
