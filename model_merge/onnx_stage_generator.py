# 读取融合的结果 weights、merge_schemes.json 和 parallel_scheme.json，生成onnx模型、worker_config.json和scheduler_config.json

import argparse
import json
import os
import shutil

import numpy as np
import torch

from model_merge.dag import ModelInfo, Stage
from model_merge.model_configs.base_merge_scheme import BaseMergeScheme
from model_merge.model_merger import export_merge_schemes_to_dag, load_merge_schemes_from_json
import model_manager.original_models as om
import utils.onnx_utils as ou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate models')
    parser.add_argument('-i','--merge_result_dir', type=str, help='Path to the merge result', default="experiments/exp/exp_overall")
    parser.add_argument('-m','--model', type=str, help='Path to save output models', default="experiments/exp/exp_overall/model_stages")
    parser.add_argument('-c','--config', type=str, help='Path to save output configs', default="experiments/exp/exp_overall/new_configs")
    parser.add_argument('-y','--model_type', type=str, help='Model type if no merge', default="resnet152")
    parser.add_argument('-n','--task_name', type=str, help='output stage name', default="test")
    args = parser.parse_args()

    step = 0

    weight_dir = os.path.join(args.merge_result_dir, "weights")

    # 读取merge_schemes.json构建dag
    merge_schemes = load_merge_schemes_from_json(os.path.join(args.merge_result_dir, "merge_schemes.json"))
    
    # 读取parallel_scheme.json
    with open(os.path.join(args.merge_result_dir, "parallel_scheme.json"), "r") as f:
        parallel_scheme = json.load(f)

    model_type = args.model_type
    if len(merge_schemes) > 0:
        model_type = merge_schemes[0].model_type

    model_infos = []
    id_of_name = {}
    for i, n in enumerate(parallel_scheme["model_names"]):
        mi = ModelInfo()
        mi.type = {"type": model_type}
        mi.name = n
        id_of_name[n] = i
        model_infos.append(mi)

    dag = export_merge_schemes_to_dag(model_infos, merge_schemes, stage_name_prefix="cluster1")

    spliters = {}

    for m, ns in enumerate(parallel_scheme["num_layer_each_stage"]):
        s = 0
        for i in ns:
            s += i
            if spliters.get(s) is None:
                spliters[s] = []
            spliters[s].append(model_infos[m].name)
    spliters = list(spliters.items())
    spliters.sort(key=lambda x:x[0])

    def split_stage(s: Stage):
        for sp in spliters:
            split = sp[0]
            if s.start_layer_index < split and s.start_layer_index + s.num_layers > split:
                union = []
                for m in s.models:
                    if m in sp[1]:
                        union.append(m)
                if len(union) == 0: # stage和该split并没有交集
                    continue
                # 分割该stage
                new_stage = s.copy(s.id+"s")
                new_stage.start_layer_index = split
                new_stage.num_layers = s.start_layer_index + s.num_layers - split
                s.num_layers = split - s.start_layer_index
                standard_scheme = BaseMergeScheme(model_type, [0], [0])
                s.split_point = standard_scheme.layer_groups[s.start_layer_index + s.num_layers - 1].split_point
                s.split_point_shape = standard_scheme.layer_groups[s.start_layer_index + s.num_layers - 1].split_point_shape

                dag.add_stage(new_stage)
                # 转移边关系，所有s接出去的边都由新stage负责
                for m in s.models:
                    ts = []
                    target_set = dag.get_target_set(s.id, m)
                    if not target_set is None:
                        for t in target_set:
                            dag.add_edge(new_stage.id, t, m)
                            ts.append(t)
                        for t in ts:
                            target_set.discard(t)
                    dag.add_edge(s.id, new_stage.id, m)
                split_stage(new_stage)
                break

    ss = list(dag.stages.values())
    for s in ss:
        split_stage(s)

    # 生成onnx模型
    tmp_model_dir = "/tmp/CoMPI/models"
    os.makedirs(tmp_model_dir, exist_ok=True)
    if step < 1:
        models = []
        for i, m in enumerate(parallel_scheme["model_names"]):
            t = parallel_scheme["model_types"]
            model_path = os.path.join(weight_dir, m + ".pth")
            model = om.get_original_model_from_config(t[i])
            model.load_state_dict(torch.load(model_path, weights_only=True))
            models.append(model)
            ou.torch2onnx(model, torch.from_numpy(np.ones((1, 3, 224, 224), dtype=np.float32)), os.path.join(tmp_model_dir, m + ".onnx"))

    # 按pipeline进一步划分，生成onnx模型
    stage_path = {}
    os.makedirs(args.model, exist_ok=True)
    seg_dir = os.path.join(tmp_model_dir, "seg")
    os.makedirs(seg_dir, exist_ok=True)
    for i, mid in enumerate(parallel_scheme["model_names"]):
        last_onnx_path = os.path.join(tmp_model_dir, mid + ".onnx")
        sid = dag.start_stages[mid]
        while not sid is None:
            stage = dag.stages[sid]
            stage_onnx_path = os.path.join(args.model, f"{sid}.onnx")
            if stage.split_point is None: # 最后一个stage
                stage_path[sid] = stage_onnx_path
                if step < 2:
                    shutil.move(last_onnx_path, stage_onnx_path)
                break
            if step < 2:
                ou.segment_onnx_with_tensor_util(last_onnx_path, [[stage.split_point]], [[stage.split_point_shape]], seg_dir, batch_size=16)
            if stage_path.get(sid) == None:
                # 记录对应模型
                tmp_stage_onnx_path = os.path.join(seg_dir, "part0.onnx")
                if step < 2:
                    shutil.move(tmp_stage_onnx_path, stage_onnx_path)
                stage_path[sid] = stage_onnx_path
            last_onnx_path = os.path.join(tmp_model_dir, "part1.onnx")
            if step < 2:
                shutil.move(os.path.join(seg_dir, "part1.onnx"), last_onnx_path)
            ns = dag.get_next_stages(sid, mid)
            if len(ns) == 0:
                break
            sid = next(iter(ns))
    shutil.rmtree(tmp_model_dir)

    # 生成 worker_config
    os.makedirs(args.config, exist_ok=True)
    if step < 3:
        worker_config = {
            "server_addr": "localhost",
            "debug": True,
            "self_addr": "localhost",
            "rpc_server_workers": 4,
            "max_data_size": 256,
            "gpu_num": 2,
            "gpu_mem": [
                15655829504,
                15655829504
            ],
            "model_path": stage_path
        }
        with open(os.path.join(args.config, "worker_config.json"), 'w') as f:
            f.write(json.dumps(worker_config))

    # 生成 scheduler_config
    if step < 4:
        SLO = 1000000
        MAX_BATCHSIZE = 16
        edges = {} # source, target, type -> models
        for sid, edge_dict in dag.edges.items():
            for m, tids in edge_dict.items():
                for tid in tids:
                    edge_type = 'direct'
                    target_stage = dag.stages[tid]
                    for (l, ms) in spliters:
                        if target_stage.start_layer_index == l and m in ms:
                            edge_type = 'normal'
                            break
                    key = (sid, tid, edge_type)
                    if edges.get(key) is None:
                        edges[key] = []
                    edges[key].append(m)
        config_edges = []
        for (sid, tid, edge_type), m in edges.items():
            config_edges.append({
                "source": sid,
                "target": tid,
                "models": m,
                "type": edge_type
            })

        layers = {} # mid, lid -> (start_layer, end_layer)
        for m, ns in enumerate(parallel_scheme["num_layer_each_stage"]):
            s = 0
            for i, n in enumerate(ns):
                layers[(m, i)] = (s, s + n)
                s += n

        num_GPU = len(parallel_scheme["deployment_of_stage"])
        static_deployment = [set() for _ in range(num_GPU)]
        for i in range(num_GPU):
            deps = parallel_scheme["deployment_of_stage"][i]
            for s in dag.stages.values():
                for (m, lid) in deps: # GPU i 上部署了模型 m 的第 lid 个 pipeline stage
                    sl, el = layers[(m, lid)]
                    if s.start_layer_index >= sl and s.start_layer_index + s.num_layers <= el:
                        static_deployment[i].add(s.id)
        config_static_deployment = [list(d) for d in static_deployment]

        scheduler_config = {
            "model_config": [{
                "id": mid,
                "SLO": SLO,
                "start_stage": dag.start_stages[mid]
            } for mid in parallel_scheme["model_names"]],
            "dag_config": {
                "stages": [{
                    "id": s.id,
                    "models": s.models
                } for s in dag.stages.values()],
                "edges": config_edges
            },
            "stage_profile": [{
                "id": s.id,
                "latency": 1,
                "max_bs": MAX_BATCHSIZE,
                "num_output": 1,
                "model_size": 0,
                "input_size": 0,
                "running_max_size": 0
            } for s in dag.stages.values()],
            "device_num": num_GPU,
            "static_deployment": config_static_deployment,
            "just_static": True,
            "discard": False,
            "debug": True,
            "debug_size": 100000,
        }

        with open(os.path.join(args.config, "scheduler_config.json"), 'w') as f:
            f.write(json.dumps(scheduler_config))
