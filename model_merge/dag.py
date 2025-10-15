from typing import Dict, List, Set

from matplotlib import patches, pyplot as plt
import torch
import torch.nn as nn

class GPUInfo:
    total_memory: float  # GPU memory in MB
    used_memory: float = 0.0

    mean_load: float = 0.0 # 当前部署的所有stage的平均负载之和

    def __init__(self, total_memory: float = 0, used_memory: float = 0, mean_load: float = 0):
        self.total_memory = total_memory
        self.used_memory = used_memory
        self.mean_load = mean_load

    def mem_left(self):
        return self.total_memory - self.used_memory
    
    def copy(self):
        g = GPUInfo(
            self.total_memory,
            self.used_memory,
            self.mean_load
        )
        return g

class ModelInfo:
    name: str
    type: Dict
    model_path: str
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    min_acc: float
    max_stage_num: int # 最大可以划分的stage数量

    max_load: float
    mean_load: float

    model: nn.Module

class Stage:
    id: str
    model_path: str
    models: List[str] # [model_id]
    split_point: str
    split_point_shape: List[str | int]
    latency: float
    start_layer_index: int
    num_layers: int
    
    def __init__(self, stage:Dict):
        self.id = stage.get('id')
        self.model_path = stage.get('model_path')
        self.models = stage.get('models')
        self.split_point = stage.get('split_point')
        self.latency = stage.get('latency')
        self.start_layer_index = stage.get('start_layer_index')
        self.num_layers = stage.get('num_layers')
        self.split_point_shape = stage.get('split_point_shape')
    
    def copy(self, id):
        return Stage({
            "id": id,
            "model_path": self.model_path,
            "models": self.models.copy(),
            "split_point": self.split_point,
            "latency": self.latency,
            "start_layer_index": self.start_layer_index,
            "num_layers": self.num_layers,
            "split_point_shape": self.split_point_shape,
        })


class DAG:
    stages:Dict[str, Stage] # stage_id -> Stage
    edges:Dict[str, Dict[str, Set[str]]] # source_stage_id -> (model_id -> target_stage_ids)
    start_stages:Dict[str, str] # model_id -> stage_id
    models:List[str] # List of model IDs 

    def __init__(self):
        self.stages = {}
        self.edges = {}
        self.start_stages = {}
        self.models = []

    
    def get_stages(self):
        return self.stages.values()
    

    def get_stage_model_path(self, id:str):
        return self.stages[id].model_path
    
    def get_target_set(self, source_id:str, model_id:str):
        edge_dict = self.edges.get(source_id)
        if edge_dict is None:
            return None
        return edge_dict.get(model_id)

    def add_stage(self, stage:Stage):
        id = stage.id
        if self.stages.get(id) is None:
            self.stages[id] = stage
            self.edges[id] = {}


    def set_start_stage(self, model_id: str, stage_id: str):
        self.start_stages[model_id] = stage_id


    def add_edge(self, start:str, end:str, model:str):
        p = self.edges[start]
        if p.get(model) is None:
            p[model] = set()
        p[model].add(end)


    def get_next_stages(self, stage_id:str, model_id:str) -> Set[str]:
        edge_map = self.edges.get(stage_id, None)
        if edge_map is None:
            return None
        return edge_map.get(model_id, set())


    def load_json(self, json_config:Dict):
        self.__init__()
        for stage in json_config['stages']:
            self.add_stage(Stage(stage))
        for edge in json_config['edges']:
            self.add_edge(edge[0], edge[1], edge[2])
    
    def to_json(self, path):
        pass

    def visualize(self, save_path="dag.png"):
        model_gap = 2.0
        stage_width = 1.2
        v_gap = 0  # stage之间的垂直间隔
        font_size = 10
        min_rect_height = font_size * 1.5  # 最矮的矩形高度，保证能写下名字

        models = self.models
        n_models = len(models)
        fig, ax = plt.subplots(figsize=(2 + n_models * 2, 10), constrained_layout=True)

        # 计算最小latency
        min_latency = min([stage.latency for stage in self.get_stages() if stage.latency > 0])
        # 动态缩放倍数
        latency_scale = min_rect_height / min_latency

        max_chain_len = 0
        total_latency = {model: 0.0 for model in models}

        # 记录每个模型每个stage的y坐标，便于后续合并
        stage_y = {}  # (stage_id, model) -> y
        stage_height = {}  # (stage_id, model) -> height

        for col, model in enumerate(models):
            y = 0
            stage_id = self.start_stages[model]
            visited = set()
            while stage_id and stage_id not in visited:
                visited.add(stage_id)
                stage = self.stages[stage_id]
                height = stage.latency * latency_scale
                stage_y[(stage_id, model)] = y
                stage_height[(stage_id, model)] = height
                total_latency[model] += stage.latency
                y += height + v_gap
                nexts = self.get_next_stages(stage_id, model)
                stage_id = list(nexts)[0] if nexts else None
            max_chain_len = max(max_chain_len, y)

        # 合并id相同且y坐标一致的stage，画大矩形
        drawn = set()
        for row in range(int(max_chain_len) + 1):  # 遍历所有可能的y
            col = 0
            while col < n_models:
                model = models[col]
                # 找到当前y下的stage
                found = False
                for (stage_id, m), y in stage_y.items():
                    if m == model and int(y) == row and (stage_id, m) not in drawn:
                        # 向右合并所有相邻模型、id相同、y一致的stage
                        merge_cols = [col]
                        next_col = col + 1
                        while next_col < n_models:
                            next_model = models[next_col]
                            if (stage_id, next_model) in stage_y and int(stage_y[(stage_id, next_model)]) == row:
                                merge_cols.append(next_col)
                                next_col += 1
                            else:
                                break
                        # 画合并后的大矩形
                        left = merge_cols[0] * model_gap
                        width = stage_width + (model_gap * (len(merge_cols) - 1))
                        height = stage_height[(stage_id, model)]
                        y_draw = y
                        rect = patches.Rectangle((left, y_draw), width, height, edgecolor='black', facecolor='skyblue', alpha=0.7)
                        ax.add_patch(rect)
                        # stage id
                        ax.text(left + width/2, y_draw + height/2, stage_id, ha='center', va='center', fontsize=font_size)
                        # 左上角标出 start_layer_index
                        ax.text(left + 0.05, y_draw + 0.05, f"{self.stages[stage_id].start_layer_index}", ha='left', va='top', fontsize=font_size-2, color='dimgray')
                        # 左下角标出 start_layer_index + num_layers
                        ax.text(left + 0.05, y_draw + height - 0.05, f"{self.stages[stage_id].start_layer_index + self.stages[stage_id].num_layers}", ha='left', va='bottom', fontsize=font_size-2, color='dimgray')
                        # 在矩形左边标出延迟
                        ax.text(left - 0.1, y_draw + height/2, f"{self.stages[stage_id].latency:.2f}", ha='right', va='center', fontsize=font_size, color='darkred')
                        # 标记已画
                        for merge_col in merge_cols:
                            drawn.add((stage_id, models[merge_col]))
                        col = merge_cols[-1] + 1
                        found = True
                        break
                if not found:
                    col += 1

        ax.set_xlim(-1, n_models * model_gap + 1)
        ax.set_ylim(-1, max_chain_len + 2)
        ax.invert_yaxis()
        ax.axis('off')
        # plt.tight_layout()
        plt.savefig(save_path)
        plt.close()