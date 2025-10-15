package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
)

// 建立 TCP 连接（含超时、KeepAlive 设置）
func ConnectTCP(addr string) (net.Conn, error) {
	dialer := net.Dialer{
		Timeout:   5 * time.Second,  // 建立连接超时时间
		KeepAlive: 15 * time.Second, // TCP KeepAlive 间隔
		// DualStack: true,               // 可选，支持 IPv4+IPv6
	}

	conn, err := dialer.Dial("tcp", addr)
	if err != nil {
		return nil, err
	}

	// 可选：设置读/写超时（每次 I/O 的超时）
	// _ = conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	// _ = conn.SetWriteDeadline(time.Now().Add(5 * time.Second))

	log.Printf("Connected to %s with KeepAlive", addr)
	return conn, nil
}

// SendBytesWithID 发送 ID + 长度前缀 + 数据
func SendBytesWithID(conn net.Conn, id uint32, data []byte) error {
	// todo 可以加上断线重连
	// 拼接 ID 和长度字段（共 8 字节）
	if conn == nil {
		return errors.New("nil conn")
	}
	var header [8]byte
	binary.LittleEndian.PutUint32(header[0:4], id)
	binary.LittleEndian.PutUint32(header[4:8], uint32(len(data)))
	// 写入头部
	// log.Printf("SendBytesWithID: id: %d, len(data): %d", id, len(data))
	if _, err := conn.Write(header[:]); err != nil {
		return err
	}

	// 写入实际数据
	_, err := conn.Write(data)
	return err
}

// ReceiveBytesWithID 接收 ID + 长度前缀 + 数据
func ReceiveBytesWithID(conn net.Conn) (uint32, []byte, error) {
	// 读 ID（4字节）
	idBuf := make([]byte, 4)
	if _, err := io.ReadFull(conn, idBuf); err != nil {
		return 0, nil, err
	}
	id := binary.LittleEndian.Uint32(idBuf)

	// 读长度（4字节）
	lenBuf := make([]byte, 4)
	if _, err := io.ReadFull(conn, lenBuf); err != nil {
		return 0, nil, err
	}
	length := binary.LittleEndian.Uint32(lenBuf)

	// 读数据内容
	buf := make([]byte, length)
	if _, err := io.ReadFull(conn, buf); err != nil {
		return 0, nil, err
	}

	return id, buf, nil
}

// StartTCPServer 启动一个 TCP 服务器
func StartTCPServer(ctx context.Context, wg *sync.WaitGroup, addr string, handleConn func(net.Conn)) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", addr, err)
	}
	log.Printf("TCP server listening on %s", addr)

	// 等待关闭监听
	go func() {
		<-ctx.Done()
		log.Println("Shutting down TCP server...")
		_ = ln.Close()
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			conn, err := ln.Accept()
			if err != nil {
				select {
				case <-ctx.Done():
					log.Println("TCP listener closed gracefully")
					return
				default:
					log.Printf("Accept error: %v", err)
					continue
				}
			}

			log.Printf("recv tcp conn from %s", conn.RemoteAddr().String())

			// 为每个连接单独起 goroutine 处理
			wg.Add(1)
			go func(c net.Conn) {
				defer wg.Done()
				handleConn(c)
			}(conn)
		}
	}()
}

// StartGRPCServer 启动 gRPC 服务器
func StartGRPCServer(
	ctx context.Context,
	wg *sync.WaitGroup,
	addr string,
	registerFunc func(*grpc.Server), // 注册服务的回调
	opts ...grpc.ServerOption, // 可选的 grpc server 配置
) {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", addr, err)
	}

	server := grpc.NewServer(opts...)
	registerFunc(server)

	wg.Add(1)
	go func() {
		defer wg.Done()
		log.Printf("gRPC server listening on %s", addr)

		go func() {
			<-ctx.Done()
			log.Println("Shutting down gRPC server...")
			server.GracefulStop()
		}()

		if err := server.Serve(listener); err != nil {
			select {
			case <-ctx.Done():
				log.Println("gRPC server closed gracefully")
			default:
				log.Fatalf("gRPC server failed: %v", err)
			}
		}
	}()
}

func getTime() float64 {
	return float64(time.Now().UnixMicro()) / 1000 // SLO 等时间都以毫秒为单位
}

func getTimeStr(t float64) string {
	seconds := int64(t / 1000)           // 秒部分
	milliseconds := int64(t) % 1000      // 毫秒部分
	microseconds := int64(t*1000) % 1000 // 微秒部分

	return time.Unix(seconds, 0).Format("2006-01-02 15:04:05") +
		fmt.Sprintf(".%03d.%03d", milliseconds, microseconds)
}
