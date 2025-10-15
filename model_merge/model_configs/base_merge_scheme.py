import json
import os
from typing import List, Optional

from model_merge.model_configs.structures import ParallelScheme


class LayerGroup:
    latency: float
    layers: List[str]  # List of layer names
    split_point: str
    split_point_shape: List[str | int]
    param_size: float
    join_merge: bool  # 是否可以参与 merge

    def __init__(self, latency:float, layers:List[str], split_point:str, param_size:float, split_point_shape:Optional[List[str | int]] = None, join_merge:bool = True):
        self.latency = latency
        self.layers = layers
        self.split_point = split_point
        self.param_size = param_size
        self.split_point_shape = split_point_shape
        self.join_merge = join_merge

class BaseMergeScheme:
    layer_groups: List[LayerGroup]
    max_stage_num: List[int]
    models: List[int]  # List of model indices
    num_models: int
    start_layer_index: int
    mem_saved: float
    mem_calc_mode: str # "simple": 只考虑单机情况下融合节省的内存; "multi": 考虑多机情况下融合节省的内存
    model_type: str
    ps: ParallelScheme

    def __init__(self, model_type: str, max_stage_num: List[int], models: List[int], mem_calc_mode: str = "simple"):
        self.layer_groups = []
        self.max_stage_num = max_stage_num
        self.models = models
        self.num_models = len(models)
        self.start_layer_index = 0
        self.mem_saved = 0.0
        self.mem_calc_mode = mem_calc_mode
        self.model_type = model_type

        json_path = os.path.join(os.path.dirname(__file__), "configs", f"{model_type}.json")
        with open(json_path, "r") as f:
            data = json.load(f)
            for lg in data:
                self.layer_groups.append(
                    LayerGroup(
                        latency=lg["latency"],
                        layers=lg["layers"],
                        split_point=lg["split_point"],
                        param_size=lg["param_size"],
                        join_merge=lg["join_merge"],
                        split_point_shape=lg.get("split_point_shape")
                    )
                )
    
    @staticmethod
    def from_dict(d: dict) -> "BaseMergeScheme":
        scheme = BaseMergeScheme(
            model_type=d["model_type"],
            max_stage_num=d["max_stage_num"],
            models=d["models"],
            mem_calc_mode=d.get("mem_calc_mode", "simple")
        )
        scheme.start_layer_index = d["start_layer_index"]
        scheme.mem_saved = d["mem_saved"]
        if "param_size" in d:
            scheme.param_size = d["param_size"]
        if "total_latency" in d:
            scheme.total_latency = d["total_latency"]
        if "ps" in d:
            scheme.ps = d["ps"]  # 可扩展为 ParallelScheme.from_dict(d["ps"])
        return scheme
    
    def __str__(self):
        return f"BaseMergeScheme(model_type={self.model_type}, max_stage_num={self.max_stage_num}, models={self.models}, " \
               f"start_layer_index={self.start_layer_index}, num_layers={self.get_num_layers()}, " \
               f"mem_saved={self.mem_saved}, mem_calc_mode={self.mem_calc_mode})"
    
    def get_num_layers(self):
        return len(self.layer_groups)

    def get_id(self):
        return (self.models[0], self.num_models, self.start_layer_index, self.get_num_layers())

    def copy(self):
        new_scheme = self.__class__(self.model_type, self.max_stage_num.copy(), self.models.copy(), self.mem_calc_mode)
        new_scheme.layer_groups = new_scheme.layer_groups[self.start_layer_index:self.start_layer_index + self.get_num_layers()]
        new_scheme.start_layer_index = self.start_layer_index
        new_scheme.mem_saved = self.mem_saved
        return new_scheme

    def calc_mem_saved(self):
        if self.mem_calc_mode == "simple":
            self.mem_saved = sum(lg.param_size if lg.join_merge else 0 for lg in self.layer_groups) * self.num_models

    def newVerticalScheme(self, index: int) -> List:
        ans = []
        if index > 1: # 融合方案必须有大于1个模型
            s = self.__class__(self.model_type, self.max_stage_num[:index], self.models[:index], self.mem_calc_mode)
            s.layer_groups = self.layer_groups
            s.start_layer_index = self.start_layer_index
            s.calc_mem_saved()
            if s.mem_saved > 0 or s.mem_calc_mode != 'simple':
                if s.mem_calc_mode == 'multi':
                    s.code = self.code * 2
                ans.append(s)

        if self.num_models - index > 1:
            s = self.__class__(self.model_type, self.max_stage_num[index:], self.models[index:], self.mem_calc_mode)
            s.layer_groups = self.layer_groups
            s.start_layer_index = self.start_layer_index
            s.calc_mem_saved()
            if s.mem_saved > 0 or s.mem_calc_mode != 'simple':
                if s.mem_calc_mode == 'multi':
                    s.code = self.code * 2 + 1
                ans.append(s)

        return ans

    def newHorizontalScheme(self, index: int) -> List:
        ans = []
        if index > 0: # 融合方案必须有大于0个层组
            s = self.__class__(self.model_type, self.max_stage_num, self.models, self.mem_calc_mode)
            s.layer_groups = self.layer_groups[:index]
            s.start_layer_index = self.start_layer_index
            s.calc_mem_saved()
            if s.mem_saved > 0 or s.mem_calc_mode != 'simple':
                if s.mem_calc_mode == 'multi':
                    s.code = self.code
                ans.append(s)

        if self.get_num_layers() - index > 0:
            s = self.__class__(self.model_type, self.max_stage_num, self.models, self.mem_calc_mode)
            s.layer_groups = self.layer_groups[index:]
            s.start_layer_index = self.start_layer_index + index
            s.calc_mem_saved()
            if s.mem_saved > 0 or s.mem_calc_mode != 'simple':
                if s.mem_calc_mode == 'multi':
                    s.code = self.code
                ans.append(s)

        return ans

    def getLatency(self) -> float:
        return sum(lg.latency for lg in self.layer_groups)

