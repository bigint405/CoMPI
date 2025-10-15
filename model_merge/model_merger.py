import gc
import json
import math
import os
import shutil
import signal
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
import model_manager.original_models as om
from sortedcontainers import SortedKeyList
from torch.utils.data import DataLoader
from datasets.imagenet_utils import CustomImageFolder
from torchvision import transforms
from model_merge.dag import DAG, GPUInfo, ModelInfo, Stage
from model_merge.model_configs.base_merge_scheme import BaseMergeScheme
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_merge.model_configs.binary_merge_scheme import BinaryMergeScheme, BinaryMergeSchemesManager, SplitScheme
from model_merge.model_configs.structures import ParallelScheme
from model_merge.parallel import ParallelManager
from utils.log_util import log_print, set_log_file
from utils.union_find import UnionFind

class MergeHistory:
    history: Dict[tuple, bool]
    path: str

    def get_schemes_id(schemes):
        return str([s.get_id() for s in schemes])

    def __init__(self, path):
        self.history = {}
        self.path = path
    
    def add_result(self, schemes, ok:bool):
        tmp_schemes = merge_adjacent_schemes(schemes)
        idx = MergeHistory.get_schemes_id(tmp_schemes)
        self.history[idx] = ok

    def check_result(self, schemes) -> Optional[bool]:
        tmp_schemes = merge_adjacent_schemes(schemes)
        idx = MergeHistory.get_schemes_id(tmp_schemes)
        return self.history.get(idx)
    
    def load(self, path=None):
        if path is None:
            path = self.path
        with open(path, 'r') as f:
            self.history = json.load(f)
    
    def export(self, path=None):
        log_print("Exporting merge history")
        if path is None:
            path = self.path
        with open(path, 'w') as f:
            json.dump(self.history, f)


def first_scheme(model_infos: List[ModelInfo], mem_calc_mode) -> BaseMergeScheme:
    if mem_calc_mode == 'simple':
        init_scheme = BaseMergeScheme(model_infos[0].type['type'], [mi.max_stage_num for mi in model_infos], list(range(len(model_infos))), mem_calc_mode=mem_calc_mode)
        init_scheme.calc_mem_saved()
    else:
        init_scheme = BinaryMergeScheme(model_infos[0].type['type'], [mi.max_stage_num for mi in model_infos], list(range(len(model_infos))), mem_calc_mode=mem_calc_mode)
        init_scheme.code = 1
    return init_scheme

def split_vertical(n: int, l: list[int]) -> int:
    best_diff = float('inf')
    best_index = -1

    left_count = 0
    i = 0

    while i < n:
        curr = l[i]
        start = i
        while i + 1 < n and l[i + 1] == curr:
            i += 1
        end = i
        block_size = end - start + 1

        left_count += block_size
        right_count = n - left_count

        if i + 1 < n:
            diff = abs(left_count - right_count)
            if diff < best_diff:
                best_diff = diff
                best_index = i + 1
            else:
                break
        i += 1

    # 特判全等：直接在中点划分
    if best_index == -1:
        best_index = n // 2
    return best_index

def get_all_vertical_spliters(code: int, n: int, l: list[int], base: int, spliters: dict[int, SplitScheme]):
    spliter = split_vertical(n, l)
    spliters[code] = SplitScheme(base, base + n, base + spliter)
    n1 = spliter
    n2 = n - spliter
    if n1 > 1:
        get_all_vertical_spliters(code << 1, n1, l[:spliter], base, spliters)
    if n2 > 1:
        get_all_vertical_spliters((code << 1) + 1, n2, l[spliter:], base + spliter, spliters)

def split_alg(old_scheme:BaseMergeScheme) -> List[BaseMergeScheme]:
    ans = []
    # 按照SLO纵向划分
    best_index = split_vertical(old_scheme.num_models, old_scheme.max_stage_num)

    ans.extend(old_scheme.newVerticalScheme(best_index))

    # 按照层组横向划分
    total = sum(lg.latency for lg in old_scheme.layer_groups)
    best_diff = float('inf')
    best_index = -1
    left_sum = 0
    for i in range(1, len(old_scheme.layer_groups)):
        left_sum += old_scheme.layer_groups[i-1].latency
        right_sum = total - left_sum
        diff = abs(left_sum - right_sum)
        if diff < best_diff:
            best_diff = diff
            best_index = i
        else:
            break
    if best_index > 0:
        ans.extend(old_scheme.newHorizontalScheme(best_index))

    return ans

def layer_from_name(model: nn.Module, name: str):
    # 支持嵌套子模块（如 "layer1.0.conv1"）
    module = model
    for attr in name.split('.'):
        module = module._modules[attr]
    return module

def weight_func(target_acc, accuracy):
    # 保证accuracy在[0,1]范围内
    accuracy = max(0.0, min(1.0, accuracy))
    if accuracy < target_acc:
        # 在[0, target_acc]区间，线性从10降到1
        return 10 - 9 * (accuracy / target_acc)
    else:
        # 在[target_acc, 1]区间，线性从1降到0.3
        return 1 - 0.7 * ((accuracy - target_acc) / (1 - target_acc))

# 根据不参加融合的layers纵向划分融合方案
def split_schemes_with_layers_not_join_merge(schemes: List[BaseMergeScheme]) -> List[BaseMergeScheme]:
    """
    将融合方案中不参加融合的层作为分界线，纵向划分成不包含不参与融合的层的方案
    """
    ans = []
    for scheme in schemes:
        start_index = -1
        for i, lg in enumerate(scheme.layer_groups):
            if lg.join_merge:
                if start_index == -1:
                    start_index = i
            else:
                # 前面没有参与融合的层组，直接继续
                if start_index == -1:
                    continue
                # 找到不参加融合的层组，进行分割
                new_scheme = scheme.copy()
                new_scheme.layer_groups = scheme.layer_groups[start_index:i]
                ans.append(new_scheme)
                start_index = -1
        if start_index != -1:
            # 最后还剩一个scheme
            new_scheme = scheme.copy()
            new_scheme.layer_groups = scheme.layer_groups[start_index:]
            ans.append(new_scheme)
    return ans

# 合并模型相同且层相邻的融合方案
def merge_adjacent_schemes(accepted_schemes: List[BaseMergeScheme]) -> List[BaseMergeScheme]:
    # 先排序已有scheme
    accepted_schemes.sort(key=lambda s: (s.models, s.start_layer_index))
    merged_schemes = []
    i = 0
    while i < len(accepted_schemes):
        current_scheme = accepted_schemes[i].copy()
        j = i + 1
        mem_saved = current_scheme.mem_saved
        while j < len(accepted_schemes) and accepted_schemes[j].models == current_scheme.models and \
              accepted_schemes[j].start_layer_index == current_scheme.start_layer_index + len(current_scheme.layer_groups):
            # 合并相邻的融合方案
            current_scheme.layer_groups.extend(accepted_schemes[j].layer_groups)
            mem_saved += accepted_schemes[j].mem_saved
            j += 1
        current_scheme.mem_saved = mem_saved
        merged_schemes.append(current_scheme)
        i = j  # 跳过已合并的方案
    return merged_schemes

def estimate_sample_mem_MB_multiple(models: List[nn.Module], sample_batch: tuple, device: str = 'cuda') -> float:
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    criterion = nn.CrossEntropyLoss()
    losses = []

    # 启用训练模式并移动模型到 GPU
    for model in models:
        model.train()
        model.to(device)

    # 构建一个 batch（B=4）
    images, labels = sample_batch
    batch_size = 4
    images = images.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    labels = labels.unsqueeze(0).repeat(batch_size).to(device)

    # 为每个模型构建独立优化器（Adam）
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4) for model in models]

    # 前向 + 反向传播 + Adam step
    for i, model in enumerate(models):
        optimizers[i].zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizers[i].step()
        losses.append(loss.item())

    # 获取显存峰值，估算每 sample 占用
    total_mem_bytes = torch.cuda.max_memory_allocated(device)
    mem_per_sample_MB = total_mem_bytes / batch_size / (1024 ** 2)

    # 清理
    for model in models:
        model.to("cpu")
    del images, labels, outputs, loss, optimizers
    gc.collect()

    return mem_per_sample_MB

def merge_and_retrain(merge_history: MergeHistory, taskId: int, model_infos: List[ModelInfo], merge_schemes: List[BaseMergeScheme], device: str, batch_size: int = 32, batch_size_per_model: int = 32) -> tuple[bool, Optional[Dict]]:
    result = merge_history.check_result(merge_schemes)
    if result is not None:
        log_print(f"Retraining {MergeHistory.get_schemes_id(merge_schemes)} has history {result}")
        return result, {}
    log_print(f"Retraining {MergeHistory.get_schemes_id(merge_schemes)} has no history")

    EPOCH1 = 3
    EPOCH2 = 5
    torch_device = torch.device(device)

    local_models = []
    for mi in model_infos:
        model = om.get_original_model_from_config(mi.type)
        model.load_state_dict(torch.load(mi.model_path, weights_only=True))
        local_models.append(model)

    uf = UnionFind(len(model_infos))
    for s in merge_schemes:
        for lg in s.layer_groups:
            if not lg.join_merge:
                continue
            for layer in lg.layers:
                source_model = local_models[s.models[0]]
                layer_obj = layer_from_name(source_model, layer)
                for i in s.models:
                    if i == s.models[0]:
                        continue
                    uf.union(s.models[0], i)
                    target_model = local_models[i]
                    for attr in layer.split('.')[:-1]:
                        target_model = target_model._modules[attr]
                    target_model._modules[layer.split('.')[-1]] = layer_obj

    model_index_need_retrain = [i for i in range(len(model_infos)) if uf.connected(merge_schemes[-1].models[0], i)]
    log_print(f"model_index_need_retrain: {model_index_need_retrain}")
    if not model_index_need_retrain:
        merge_history.add_result(merge_schemes, True)
        merge_history.export()
        return True, {}

    # === 数据与模型加载 ===
    models, parameters_all, train_dataloaders, val_dataloaders = [], [], [], []
    for idx in model_index_need_retrain:
        mi = model_infos[idx]
        mi.train_dataset.load()
        mi.val_dataset.load()
        model = local_models[idx].to(torch_device)
        models.append(model)
        parameters_all.extend(model.parameters())
        
    num_workers = 4
    if len(models) > 11:
        batch_size = 3
    elif len(models) > 5:
        batch_size = 8
        num_workers = 6
    else:
        batch_size = 16
        num_workers = 8
    log_print(f"batchsize: {batch_size}, num models: {len(models)}")
    for idx in model_index_need_retrain:
        mi = model_infos[idx]
        train_dataloaders.append(DataLoader(mi.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
        val_dataloaders.append(DataLoader(mi.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4))

    optimizer = torch.optim.Adam(list(set(parameters_all)), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    loss_weights = [1.0 for _ in models]
    acc_record = [[] for _ in models]
    crash = False

    try:
        for epoch in range(EPOCH1):
            for model in models:
                model.train()
            dataloader_iters = [iter(dl) for dl in train_dataloaders]
            finished = [False] * len(models)
            while not all(finished):
                losses = []
                for i, model in enumerate(models):
                    try:
                        batch = next(dataloader_iters[i])
                    except StopIteration:
                        finished[i] = True
                        continue
                    images, labels = batch[0].to(torch_device), batch[1].to(torch_device)
                    loss = criterion(model(images), labels)
                    losses.append(loss * loss_weights[i])

                if not losses:
                    break

                total_loss = sum(losses)
                if torch.isnan(total_loss):
                    log_print(f"Task {taskId}: Loss is nan. Abort.")
                    return False, None
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 验证准确率
            retrain_remaining = len(models)
            for i, model in enumerate(models):
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for images, labels in val_dataloaders[i]:
                        images, labels = images.to(torch_device), labels.to(torch_device)
                        outputs = model(images)
                        _, pred = outputs.max(1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()
                acc = correct / total if total else 0
                acc_record[i].append(acc)
                loss_weights[i] = weight_func(model_infos[model_index_need_retrain[i]].min_acc, acc)
                log_print(f"Task {taskId} Epoch {epoch}: {model_infos[model_index_need_retrain[i]].name} acc={acc:.4f}")
                if acc >= model_infos[model_index_need_retrain[i]].min_acc:
                    retrain_remaining -= 1

            if retrain_remaining == 0:
                break

        # 第二阶段训练：单独训练未融合部分参数
        final_models = {}
        for i in range(len(models)):
            models[i] = models[i].to("cpu")
        del models
        gc.collect()

        ok = True
        models = []
        for i, idx in enumerate(model_index_need_retrain):
            mi = model_infos[idx]
            model = local_models[idx].to(torch_device)
            for s in merge_schemes:
                if idx in s.models:
                    for lg in s.layer_groups:
                        if lg.join_merge:
                            for layer_name in lg.layers:
                                layer_obj = layer_from_name(model, layer_name)
                                for param in layer_obj.parameters():
                                    param.requires_grad = False # 冻结共享层

            train_loader = DataLoader(mi.train_dataset, batch_size=batch_size_per_model, shuffle=True, drop_last=True, num_workers=4)
            val_loader = DataLoader(mi.val_dataset, batch_size=batch_size_per_model, shuffle=False, num_workers=3)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            epoch_ok = False
            for epoch in range(EPOCH2):
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(torch_device), labels.to(torch_device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(torch_device), labels.to(torch_device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total if total > 0 else 0
                log_print(f"Task {taskId} Epoch {epoch} {mi.name} second stage acc={acc:.4f}, target={mi.min_acc:.4f}")
                if acc >= mi.min_acc:
                    final_models[idx] = model
                    epoch_ok = True
                    break

            model.to("cpu")
            del model, train_loader, val_loader, optimizer
            gc.collect()

            if not epoch_ok:
                ok = False
                log_print(f"Task {taskId} {mi.name} second stage not pass, acc={acc:.4f}, target={mi.min_acc:.4f}")
                break

    except Exception as e:
        log_print(f"Task {taskId}: Exception during training: {e}")
        crash = True

    finally:
        for model in models:
            model.to("cpu")

    if ok:
        log_print(f"Task {taskId}: Training finished and all models meet target accuracy!")
        merge_history.add_result(merge_schemes, True)
        merge_history.export()
        local_models.clear()
        gc.collect()
        return True, final_models

    log_print(f"Task {taskId}: Training finished but some models didn't meet target accuracy.")
    if not crash:
        merge_history.add_result(merge_schemes, False)
        merge_history.export()
    local_models.clear()
    gc.collect()
    return False, None

def export_merge_schemes_to_json(merge_schemes: List[BaseMergeScheme], output_path: str):
    """
    导出融合方案到JSON文件
    """
    schemes_data = []
    for scheme in merge_schemes:
        scheme_data = {
            'model_type': scheme.model_type,
            'max_stage_num': scheme.max_stage_num,
            'models': scheme.models,
            'start_layer_index': scheme.start_layer_index,
            'num_layers': len(scheme.layer_groups),
            'mem_saved': int(scheme.mem_saved),
            "param_size": sum(lg.param_size for lg in scheme.layer_groups),
            'total_latency': scheme.getLatency(),
        }
        if isinstance(scheme, BinaryMergeScheme):
            scheme_data['code'] = scheme.code
        schemes_data.append(scheme_data)

    with open(output_path, 'w') as f:
        json.dump(schemes_data, f, indent=4)

def load_merge_schemes_from_json(input_path: str) -> List[BaseMergeScheme]:
    """
    从JSON文件加载融合方案
    """
    with open(input_path, 'r') as f:
        schemes_data = json.load(f)

    schemes = []
    for scheme_data in schemes_data:
        scheme = None
        if scheme_data.get('code') is None:
            scheme = BaseMergeScheme(
                model_type=scheme_data['model_type'],
                max_stage_num=scheme_data['max_stage_num'],
                models=scheme_data['models']
            )
        else:
            scheme = BinaryMergeScheme(
                model_type=scheme_data['model_type'],
                max_stage_num=scheme_data['max_stage_num'],
                models=scheme_data['models']
            )
            scheme.code = scheme_data['code']
        scheme.start_layer_index = scheme_data['start_layer_index']
        scheme.mem_saved = scheme_data['mem_saved']
        scheme.layer_groups = scheme.layer_groups[scheme_data['start_layer_index']:scheme_data['start_layer_index'] + scheme_data['num_layers']]
        schemes.append(scheme)
    return schemes

def export_merge_schemes_to_dag(model_infos: List[ModelInfo], merge_schemes: List[BaseMergeScheme], stage_name_prefix:str="test") -> DAG:
    stage_count = 0

    def new_stage_id():
        nonlocal stage_count
        ans = ''
        if len(stage_name_prefix) == 0:
            ans = f"stage-{stage_count}"
        else:
            ans = f"stage_{stage_name_prefix}-{stage_count}"
        stage_count += 1
        return ans
    
    dag = DAG()
    stage_map = {}

    for i in range(len(model_infos)):
        # 获取与该模型相关的stage
        relative_stages = []
        for s in merge_schemes:
            if i in s.models:
                relative_stages.append(s)
        relative_stages.sort(key=lambda s: s.start_layer_index)
        dag.models.append(model_infos[i].name)
        # 构建图
        last_layer = 0
        last_stage = None
        for s in relative_stages:
            assert isinstance(s, BaseMergeScheme)
            if s.start_layer_index > last_layer:
                # 中间存在没有融合的层，单独构建为一个该模型独有的阶段
                standard_scheme = BaseMergeScheme(model_infos[i].type['type'], [0], [0])
                standard_scheme.layer_groups = standard_scheme.layer_groups[last_layer:s.start_layer_index]
                stage = Stage({ "id": new_stage_id(),
                                "models": [model_infos[i].name],
                                "split_point": standard_scheme.layer_groups[-1].split_point,
                                "latency": standard_scheme.getLatency(),
                                "start_layer_index": last_layer,
                                "num_layers": s.start_layer_index - last_layer,
                                "split_point_shape": standard_scheme.layer_groups[-1].split_point_shape
                               })
                dag.add_stage(stage)
                if last_stage is None: # 该stage是这个模型的start stage
                    dag.set_start_stage(model_infos[i].name, stage.id)
                else:
                    dag.add_edge(last_stage.id, stage.id, model_infos[i].name)
                last_stage = stage
            # 构建融合的阶段
            if s.models[0] == i: # 保证每个阶段只构建一次
                stage = Stage({ "id": new_stage_id(),
                                "models": [model_infos[j].name for j in s.models],
                                "split_point": s.layer_groups[-1].split_point,
                                "latency": s.getLatency(),
                                "start_layer_index": s.start_layer_index,
                                "num_layers": len(s.layer_groups),
                                "split_point_shape": s.layer_groups[-1].split_point_shape,
                               })
                dag.add_stage(stage)
                for j in s.models:
                    stage_map[(j, s.start_layer_index)] = stage
            else: # 获取之前构建过的阶段
                stage = stage_map[(i, s.start_layer_index)]
            last_layer = s.start_layer_index + len(s.layer_groups)
            # 连边
            if last_stage is None: # 该stage是这个模型的start stage
                dag.set_start_stage(model_infos[i].name, stage.id)
            else:
                dag.add_edge(last_stage.id, stage.id, model_infos[i].name)
            last_stage = stage

        standard_scheme = BaseMergeScheme(model_infos[i].type['type'], [0], [0])
        if len(relative_stages) == 0: # 该模型没有任何融合阶段
            last_start_index = 0
        else: # 最后一个阶段后面是否还有未融合的层
            last_start_index = last_stage.start_layer_index + last_stage.num_layers
        total_len = len(standard_scheme.layer_groups)
        if last_start_index < total_len:
            standard_scheme.layer_groups = standard_scheme.layer_groups[last_start_index:]
            stage = Stage({ "id": new_stage_id(),
                            "models": [model_infos[i].name],
                            "split_point": standard_scheme.layer_groups[-1].split_point,
                            "latency": standard_scheme.getLatency(),
                            "start_layer_index": last_start_index,
                            "num_layers": total_len - last_start_index,
                            "split_point_shape": standard_scheme.layer_groups[-1].split_point_shape,
                        })
            dag.add_stage(stage)
            if last_stage is None:
                dag.set_start_stage(model_infos[i].name, stage.id)
            else:
                dag.add_edge(last_stage.id, stage.id, model_infos[i].name)
    return dag

def merge_model(result_dir:str, merge_history: MergeHistory, model_infos: List[ModelInfo], gpu_infos: List[GPUInfo], devices: List[str], batch_size: int = 32, batch_size_per_model: int = 32, mem_calc_mode="simple") -> tuple[List[BaseMergeScheme], ParallelScheme]:
    parallel_manager = ParallelManager(model_infos, gpu_infos)

    queue = SortedKeyList(key=lambda i: -i.mem_saved)
    init_scheme = first_scheme(model_infos, mem_calc_mode)
    queue.add(init_scheme)

    merged_table = [[False for _ in init_scheme.layer_groups] for _ in range(init_scheme.num_models)]
    accepted_schemes = []

    vertical_spliters = {}
    get_all_vertical_spliters(1, init_scheme.num_models, init_scheme.max_stage_num, 0, vertical_spliters)

    log_print("Calculating original mem used.")
    origin_mem_used, _ = parallel_manager.solve([], vertical_spliters)

    log_print("Calculating mem_saved of init scheme.")
    init_mem_used, parallel_scheme = parallel_manager.solve([init_scheme], vertical_spliters)
    init_scheme.mem_saved = origin_mem_used - init_mem_used
    init_scheme.ps = parallel_scheme

    visited_scheme = set()
    visited_scheme.add(init_scheme.get_id())

    merge_count = 0

    def retrain_wrapper(taskId, scheme, device):
        tmp_accepted = accepted_schemes + [scheme]
        ok, models = merge_and_retrain(merge_history, taskId, model_infos, tmp_accepted, device, batch_size, batch_size_per_model)
        return ok, models

    while len(queue) > 0:
        # 安全地取出前 len(devices) 个方案
        schemes_to_try = []
        for _ in range(len(devices)):
            if len(queue) == 0:
                break
            schemes_to_try.append(queue.pop(0))

        if not schemes_to_try:
            break

        log_print(f"Trying {len(schemes_to_try)} schemes in parallel...")

        results = []
        accepted = False
        with ThreadPoolExecutor(max_workers=len(schemes_to_try)) as executor:
            futures = {
                executor.submit(retrain_wrapper, i, scheme, devices[i % len(devices)]): (i, scheme)
                for i, scheme in enumerate(schemes_to_try)
            }
            for future in as_completed(futures):
                idx, scheme = futures[future]
                try:
                    ok, models = future.result()
                except Exception as e:
                    log_print(f"Exception during training scheme {scheme}: {e}")
                    ok, models = False, None
                results.append((idx, scheme, ok, models))
                if ok and idx == 0:
                    accepted = True

        results.sort(key=lambda x: x[0])  # 保证队列顺序优先
        # 所有并行任务都运行完之后再清空，否则会报错
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        for idx, scheme, ok, models in results:
            if ok:
                if idx == 0: # 只接受节省最多的方案
                    log_print(f"Scheme {scheme} merged successfully.")
                    if models:
                        for id, model in models.items():
                            torch.save(model.state_dict(), model_infos[id].model_path)
                    del models

                    accepted_schemes.append(scheme)
                    parallel_scheme = scheme.ps
                    merge_count += 1

                    for i in scheme.models:
                        for j in range(scheme.start_layer_index, scheme.start_layer_index + len(scheme.layer_groups)):
                            merged_table[i][j] = True

                    # 输出中间结果
                    mid_result_dir = os.path.join(result_dir, f"{merge_count}")
                    os.makedirs(mid_result_dir, exist_ok=True)
                    parallel_scheme.to_json(model_infos, os.path.join(mid_result_dir, "parallel_scheme.json"))
                    export_merge_schemes_to_json(accepted_schemes, os.path.join(mid_result_dir, 'original_merge_schemes.json'))
                    tmp_output_schemes = merge_adjacent_schemes(accepted_schemes)
                    tmp_output_schemes = split_schemes_with_layers_not_join_merge(tmp_output_schemes)
                    export_merge_schemes_to_json(tmp_output_schemes, os.path.join(mid_result_dir, "merge_schemes.json"))
                    tmp_output_schemes = load_merge_schemes_from_json(os.path.join(mid_result_dir, "merge_schemes.json"))
                    dag = export_merge_schemes_to_dag(model_infos, tmp_output_schemes)
                    dag.visualize(save_path=os.path.join(mid_result_dir, "merge_dag.png"))
                    del tmp_output_schemes, dag
                else:
                    # 其它可行的放回队列中，因为目的是筛出不可行的然后split
                    log_print(f"Scheme {scheme} merged successfully, but not first.")
                    queue.add(scheme)
            else:
                log_print(f"Scheme {scheme} merged failed.")
                new_schemes = split_alg(scheme)
                if not accepted:
                    valid_schemes = []
                    for new_scheme in new_schemes:
                        if new_scheme.get_id() in visited_scheme:
                            continue
                        visited_scheme.add(new_scheme.get_id())
                        if any(merged_table[i][j]
                                for i in new_scheme.models
                                for j in range(new_scheme.start_layer_index,
                                                new_scheme.start_layer_index + len(new_scheme.layer_groups))):
                            log_print(f"Scheme {new_scheme} has already been merged, skipping.")
                            continue
                        if mem_calc_mode == 'multi':
                            accepted_schemes.append(new_scheme)
                            log_print(f"Get mem saved from scheme {new_scheme}")
                            new_mem, new_ps = parallel_manager.solve(accepted_schemes, vertical_spliters)
                            new_scheme.mem_saved = origin_mem_used - new_mem
                            new_scheme.ps = new_ps
                            log_print(f"Mem saved of scheme {new_scheme} is {new_scheme.mem_saved}")
                            accepted_schemes = accepted_schemes[:-1]
                        if new_scheme.mem_saved > 0:
                            valid_schemes.append(new_scheme)
                    queue.update(valid_schemes)
                else: # 之后再统一更新 mem_saved
                    queue.update(new_schemes)
        
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 更新 queue 中的 mem_saved 并跳过涉及已融合层的方案
        if accepted:
            log_print(f"Update mem_saved of all scheme in queue")
            new_queue = SortedKeyList(key=lambda i: -i.mem_saved)
            while len(queue) > 0:
                tmp_scheme = queue.pop(0)
                if any(merged_table[i][j]
                        for i in tmp_scheme.models
                        for j in range(tmp_scheme.start_layer_index,
                                        tmp_scheme.start_layer_index + len(tmp_scheme.layer_groups))):
                    log_print(f"Scheme {tmp_scheme} has already been merged, skipping.")
                    continue
                log_print(f"Scheme {tmp_scheme} is still valid.")
                if mem_calc_mode == 'multi':
                    accepted_schemes.append(tmp_scheme)
                    mem_used, new_ps = parallel_manager.solve(accepted_schemes, vertical_spliters)
                    tmp_scheme.mem_saved = origin_mem_used - mem_used
                    tmp_scheme.ps = new_ps
                    accepted_schemes = accepted_schemes[:-1]
                if tmp_scheme.mem_saved > 0:
                    new_queue.add(tmp_scheme)
            queue = new_queue

    accepted_schemes = merge_adjacent_schemes(accepted_schemes)
    accepted_schemes = split_schemes_with_layers_not_join_merge(accepted_schemes)
    return accepted_schemes, parallel_scheme

def get_dataset(config: Dict) -> torch.utils.data.Dataset:
    if config['type'] == 'image_folder':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return CustomImageFolder(root_dir=config['base_path'], target_classes=config['class'], transform=transform)


def load_model_gpu_infos_dict(merge_config) -> tuple[list[ModelInfo], list]:
    model_infos = []
    for model_config in merge_config['models']:
        mi = ModelInfo()
        mi.name = model_config['name']
        mi.type = model_config['type']
        mi.model_path = model_config['model_path']

        mi.train_dataset = get_dataset(model_config['train_dataset'])
        mi.val_dataset = get_dataset(model_config['val_dataset'])

        mi.min_acc = model_config['min_acc']
        mi.max_stage_num = model_config['max_stage_num']

        mi.max_load = model_config['max_gpu_num']
        mi.mean_load = model_config['mean_gpu_num']

        model_infos.append(mi)
    model_infos.sort(key=lambda x:x.max_stage_num)
    log_print('Sorted models:')
    for mi in model_infos:
        log_print(f"{mi.name}: {mi.max_stage_num}")
    gpu_infos = []
    for gpu_info in merge_config['GPU_infos']:
        gpu = GPUInfo()
        gpu.total_memory = gpu_info['memory']
        gpu_infos.append(gpu)
    return model_infos, gpu_infos

if __name__ == '__main__':
    log_print("Start", flush=True)
    def terminate_process(signal_num, frame):
        log_print("收到终止信号...")
        print("收到终止信号...")
        # 使用父进程的进程组ID来终止整个进程组（父进程和子进程）
        os.system("pgrep -f 'model_merger' | xargs kill")
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
    signal.signal(signal.SIGINT, terminate_process)  # 捕获 Ctrl+C

    set_log_file("test.log")

    merge_history_path = "merge_history.json"
    result_path = 'compi_results_2'
    weight_dir = os.path.join(result_path, "weights")
    merge_history = MergeHistory(merge_history_path)
    if os.path.exists(merge_history_path):
        merge_history.load()

    with open('configs/merge_config.json', 'r') as f:
        merge_config = json.load(f)
    
    model_infos, gpu_infos = load_model_gpu_infos_dict(merge_config)
    os.makedirs(weight_dir, exist_ok=True)
    for mi in model_infos:
        dst_path = os.path.join(weight_dir, f"{mi.name}.pth")
        if not os.path.exists(dst_path): # 便于读取之前运行的结果
            shutil.copyfile(mi.model_path, dst_path)
        mi.model_path = dst_path

    log_print("Start Merge", flush=True)
    accepted_schemes, parallel_scheme = merge_model(result_path, merge_history, model_infos, gpu_infos, merge_config['devices'], merge_config['batch_size'], merge_config['batch_size_per_model'], mem_calc_mode='multi')
    export_merge_schemes_to_json(accepted_schemes, os.path.join(result_path, 'merge_schemes.json'))
    parallel_scheme.to_json(model_infos, os.path.join(result_path, 'parallel_scheme.json'))

    accepted_schemes = load_merge_schemes_from_json(os.path.join(result_path, 'merge_schemes.json'))
    dag = export_merge_schemes_to_dag(model_infos, accepted_schemes)
    dag.visualize(save_path=os.path.join(result_path, 'merge_dag.png'))
    