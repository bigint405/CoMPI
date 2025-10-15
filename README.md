# CoMPI

#### Introduction

This project provides a machine learning inference system that supports model merging and parallelization, together with a tool that jointly optimizes model merging and parallel deployment. The work is included in the paper *CoMPI: Coordinated Model Merging and Parallel Inference at Edge*, published at the ACM Symposium on Cloud Computing 2025.

#### Directory Structure

Below are the main folders and files with their purposes.

```
├── configs                                 Configuration files
│   ├── dataset_configs.json                Path settings for the ImageNet dataset
│   ├── merge_config.json                   Sample configuration for models to be merged
│   ├── scheduler_config.json               Sample scheduler settings for the ML inference server; running model_merge/onnx_stage_generator.py will generate a new one
│   ├── server_config.json                  Sample ML inference server settings
│   └── worker_config.json                  Sample ML inference worker settings
├── datasets                                Dataset utilities
│   ├── imagenet_preprocess.py              Preprocess the downloaded ImageNet dataset
│   └── pre_trans.py
├── go_client                               Request sending client that can generate workloads with multiple arrival distributions for testing
├── go_server                               ML inference server; run exactly one instance on the server node
├── go_worker                               ML inference worker; run one instance on each worker node
├── model_manager                           Model management utilities
│   └── generator.py                        Generate a set of test models
├── model_merge                             Tool for joint optimization of model merging and parallel deployment
│   ├── model_configs                       Obtain model metadata
│   │   ├── config_simplifier.py            Reduce metadata to the desired granularity
│   │   └── resnet_config_generator.py      Execute ResNet family models and produce metadata files
│   ├── model_merger.py                     Algorithm entry point that outputs the merging plan and the parallel plan
│   ├── onnx_stage_generator.py             Read the merging results and generate configuration files for the ML inference system
│   └── parallel.py                         Given the merging plan and workload features, derive the parallelization and deployment plan
├── utils                                   Misc utilities
│   ├── draw_dag.py                         Visualize a merging plan as a DAG
│   ├── gather_onnx_running_data.py         Run ONNX models and collect metadata
│   └── onnx_utils.py                       ONNX helpers, including model splitting
└── init.sh                                 Build script for the ML inference system
```
