'''
运行resnet模型并生成配置信息文件
'''
import json
import os
import time
import torchvision.models as models

from model_merge.model_configs.base_merge_scheme import LayerGroup

import torch
import torch.nn as nn
from collections import OrderedDict


def measure_resnet_block_times(model, dummy_input, device='cuda', repeat=20, warmup=5):
    model = model.to(device)
    dummy_input = dummy_input.to(device)
    model.eval()

    times = {}

    with torch.no_grad():
        # 1. conv1+bn1+relu+maxpool整体
        for _ in range(warmup):
            x = dummy_input
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeat):
            x = dummy_input
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
        torch.cuda.synchronize()
        t1 = time.time()
        times['stage0'] = (t1 - t0) / repeat

        # 2. layer1, layer2, layer3, layer4，细化到layerX.Y
        prev = x
        for lname in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, lname)
            for idx, block in enumerate(layer):
                # warmup
                for _ in range(warmup):
                    block(prev)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(repeat):
                    out = block(prev)
                torch.cuda.synchronize()
                t1 = time.time()
                times[f'{lname}.{idx}'] = (t1 - t0) / repeat
                prev = out

        # 3. avgpool+fc整体
        for _ in range(warmup):
            y = model.avgpool(prev)
            y = torch.flatten(y, 1)
            y = model.fc(y)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeat):
            y = model.avgpool(prev)
            y = torch.flatten(y, 1)
            y = model.fc(y)
        torch.cuda.synchronize()
        t1 = time.time()
        times['stage_head'] = (t1 - t0) / repeat

    return times


def register_hooks(model, dummy_input):
    summary = OrderedDict()

    def hook_fn(name):
        def hook(module, input, output):
            def extract_shapes(tensors):
                if isinstance(tensors, (tuple, list)):
                    return [(tuple(t.shape), t.numel()) for t in tensors if isinstance(t, torch.Tensor)]
                elif isinstance(tensors, torch.Tensor):
                    return [(tuple(tensors.shape), tensors.numel())]
                else:
                    return []

            input_info = extract_shapes(input)
            output_info = extract_shapes(output)

            has_params = any(p.requires_grad for p in module.parameters())
            param_size = sum(p.numel() for p in module.parameters() if p.requires_grad)

            summary[name] = {
                'input_info': input_info,     # List of (shape, size)
                'output_info': output_info,   # List of (shape, size)
                'has_params': has_params,
                'param_size': param_size
            }
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential) or module == model:
            continue
        hooks.append(module.register_forward_hook(hook_fn(name)))

    # forward pass
    with torch.no_grad():
        model.eval()
        _ = model(dummy_input)

    for hook in hooks:
        hook.remove()

    return summary

def sum_mem(names:list[str], summary:dict) -> float:
    return sum(summary[name]['param_size'] for name in names if name in summary) * 4 # 4 是float的字节大小

if __name__ == '__main__':
    ms = ['resnet50', 'resnet101', 'resnet152']
    for model_name in ms:
        model = models.__dict__[model_name]()
        dummy_input = torch.randn(16, 3, 224, 224)
        summary = register_hooks(model, dummy_input)
        avg_times = measure_resnet_block_times(model, dummy_input, device='cuda', repeat=100, warmup=10)
        config = []
        total_time = sum(avg_times.values())
        for name, t in avg_times.items():
            if name == 'stage0':
                names = ["conv1", "bn1", "relu", "maxpool"]
                shape = list(summary["maxpool"]["output_info"][0][0])
                shape[0] = 'batch_size'
                shape = tuple(shape)
                lg = LayerGroup(t*1000, names, "/maxpool/MaxPool_output_0", sum_mem(names, summary), shape)
            elif name == 'stage_head':
                names = ["avgpool", "fc"]
                shape = list(summary["fc"]["output_info"][0][0])
                shape[0] = 'batch_size'
                shape = tuple(shape)
                lg = LayerGroup(t*1000, names, None, sum_mem(names, summary), shape, join_merge=False)
            else:
                shape = list(summary[name]["output_info"][0][0])
                shape[0] = 'batch_size'
                shape = tuple(shape)
                lg = LayerGroup(t*1000, [name], f"/{name.split('.')[0]}/{name}/relu_2/Relu_output_0", sum_mem([name], summary), shape)
            config.append(lg)
        os.makedirs(os.path.join(os.path.dirname(__file__), "configs"), exist_ok=True)
        with open(os.path.join(os.path.dirname(__file__), "configs", f"{model_name}.json"), "w") as f:
            json.dump([lg.__dict__ for lg in config], f, indent=2, ensure_ascii=False)

