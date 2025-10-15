import argparse
import json
import os

from model_merge.model_configs.base_merge_scheme import LayerGroup

# 简化原始的模型config，防止过于复杂融合复杂度过高，限制每个lg的时间长度都在给定的time以上，保证每个新lg的join_merge相同
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simplify model configs')
    parser.add_argument('-f','--file', type=str, help='path to target config file', default="model_merge/model_configs/configs/resnet152_old.json")
    parser.add_argument('-t','--time', type=str, help='min time of layer groups', default=2)

    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {args.file} does not exist.")
        exit(1)

    new_config = []
    t = float(args.time)
    t0 = 0
    names = []
    p_size = 0
    j_m = None
    last_sp = None
    shape = None

    for i, lg in enumerate(model_config):
        if not j_m is None and lg["join_merge"] != j_m:
            new_lg = LayerGroup(t0, names, last_sp, p_size, shape, j_m)
            t0 = 0
            names = []
            p_size = 0
            j_m = None
            last_sp = None
            shape = None
            new_config.append(new_lg)

        t0 += lg["latency"]
        names.extend(lg["layers"])
        p_size += lg["param_size"]
        last_sp = lg["split_point"]
        j_m = lg["join_merge"]
        shape = lg["split_point_shape"]

        if t0 >= t or i == len(model_config) - 1:
            new_lg = LayerGroup(t0, names, last_sp, p_size, shape, j_m)
            t0 = 0
            names = []
            p_size = 0
            j_m = None
            last_sp = None
            shape = None
            new_config.append(new_lg)
    
    with open("new_config.json", "w") as f:
        json.dump([lg.__dict__ for lg in new_config], f, indent=2, ensure_ascii=False)

