package main

import (
	"compi_go_server/rpc_server"
	"log"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
)

// ✅ 测试时启动 gRPC 服务器
func TestGRPCServer(t *testing.T) {
	lis, err := net.Listen("tcp", ":50051")
	assert.NoError(t, err)

	opts := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(100 * 1024 * 1024), // 100MB
		grpc.MaxSendMsgSize(100 * 1024 * 1024), // 100MB
	}

	s := grpc.NewServer(opts...)
	rpc_server.RegisterGetTargetDeviceServiceServer(s, &rpcServer{})
	rpc_server.RegisterFinishInferServiceServer(s, &rpcServer{})

	go func() {
		log.Println("🔥 gRPC test server running on port 50051")
		err := s.Serve(lis)
		assert.NoError(t, err)
	}()

	// ✅ 保持测试运行，避免 gRPC 服务器退出
	select {}
}
