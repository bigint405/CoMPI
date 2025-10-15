import json
import os

from model_merge.model_merger import export_merge_schemes_to_dag, load_model_gpu_infos_dict, load_merge_schemes_from_json


if __name__ == '__main__':
    with open('experiments/exp/exp_overall/merge_config.json', 'r') as f:
        merge_config = json.load(f)
    model_infos, gpu_infos = load_model_gpu_infos_dict(merge_config)
    # for mi in model_infos:
    #     mi.type['type'] = 'resnet152_old'
    merge_schemes = load_merge_schemes_from_json("experiments/exp/exp_overall/merge_schemes.json")
    dag = export_merge_schemes_to_dag(model_infos, merge_schemes, stage_name_prefix='')
    dag.visualize(save_path=os.path.join('experiments/exp/exp_overall', 'merge_visual_CoMPI.pdf'))