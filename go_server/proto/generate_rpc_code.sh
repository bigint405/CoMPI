protoc --go_out=. --go-grpc_out=. --proto_path=. rpc_service.proto
rm -rf ../rpc_server
mv compi_go_server/rpc_server ..
rmdir compi_go_server
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. rpc_service.proto
# mv -f rpc_service_pb2_grpc.py ../../device_daemon/go_server/rpc_service_pb2_grpc.py
# mv -f rpc_service_pb2.py ../../device_daemon/go_server/rpc_service_pb2.py
# sed -i '0,/import rpc_service_pb2 as rpc__service__pb2/s/import rpc_service_pb2 as rpc__service__pb2/from . import rpc_service_pb2 as rpc__service__pb2/' ../../device_daemon/go_server/rpc_service_pb2_grpc.py
