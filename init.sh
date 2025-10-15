#!/bin/bash

cd go_worker
make clean
make

rm rpc_server
ln -s ../go_server/rpc_server rpc_server

rm net_utils.go
ln -s ../go_server/net_utils.go net_utils.go

cd ../go_client

rm onnxruntime_cgo.h
ln -s ../go_worker/onnxruntime_cgo.h onnxruntime_cgo.h

rm wrapper_cgo.go
ln -s ../go_worker/wrapper_cgo.go wrapper_cgo.go

rm libonnx_cgo.so
ln -s ../go_worker/libonnx_cgo.so libonnx_cgo.so