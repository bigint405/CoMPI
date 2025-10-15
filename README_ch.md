# CoMPI

#### 介绍

本项目包含了一个支持模型融合与并行化的机器学习模型推理系统，以及一个联合优化模型融合与并行化部署方案的工具。工作已包含在论文 CoMPI: Coordinated Model Merging and Parallel Inference at Edge 中并发表在 ACM Symposium on Cloud Computing 2025 上。

#### 文件夹结构

下面展示了本项目的主要文件及其含义

```
├── configs 配置文件
│   ├── dataset_configs.json imagenet数据集路径
│   ├── merge_config.json 参与融合的模型配置，此为样例
│   ├── scheduler_config.json ML推理系统服务器的调度器设置，此为样例，运行 model_merge/onnx_stage_generator.py 会生成新的
│   ├── server_config.json ML推理系统服务器设置，此为样例
│   └── worker_config.json ML推理系统工作节点设置，此为样例
├── datasets 数据集管理工具
│   ├── imagenet_preprocess.py 预处理下载好的imagenet数据集
│   └── pre_trans.py 
├── go_client 发送请求的客户端，能够发送多种分布的请求用来测试
├── go_server ML推理server，只用在server节点运行一个
├── go_worker ML推理worker，每个worker节点都要运行一个
├── model_manager 模型管理工具
│   └── generator.py 生成若干用于测试的模型
├── model_merge 联合优化模型融合与并行化部署方案的工具
│   ├── model_configs 得到模型的元数据
│   │   ├── config_simplifier.py 简化元数据到给定的粒度
│   │   └── resnet_config_generator.py 运行resnet系列模型并生成元数据文件
│   ├── model_merger.py 算法入口，得到融合方案与并行方案
│   ├── onnx_stage_generator.py 读取融合的结果，并生成供ML推理系统运行的配置文件
│   └── parallel.py 给定融合方案以及负载特征，得到并行化和部署方案
├── utils 一些工具
│   ├── draw_dag.py 读取融合方案并可视化
│   ├── gather_onnx_running_data.py 运行onnx模型并收集元数据
│   └── onnx_utils.py 提供对onnx模型操作的接口，包括拆分
└── init.sh 编译ML推理系统
```
