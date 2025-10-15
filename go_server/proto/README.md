以下操作都被整合到`generate_rpc_code.sh`中了，在该目录直接执行即可

## Go接口

生成`rpc_service_grpc.pb.go`、`rpc_service.pb.go`

```sh
protoc --go_out=. --go-grpc_out=. --proto_path=. rpc_service.proto
```

然后把`rpc_server`文件夹放到`go_server`文件夹下面。

## Python接口

生成`rpc_service_pb2_grpc.py`、`rpc_service_pb2.py`

```sh
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. rpc_service.proto
```

放到`device_daemon/go_server`文件夹下面

然后把`rpc_service_pb2_grpc.py`中的

```py
import rpc_service_pb2 as rpc__service__pb2
```

改成

```py
from . import rpc_service_pb2 as rpc__service__pb2
```
