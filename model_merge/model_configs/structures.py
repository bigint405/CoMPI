import json


class ParallelScheme:
    num_layer_each_stage: list[list[int]] # [i][j] 模型i的第j个stage有多少层
    deployment_of_stage: list[list[tuple[int, int]]] # deployment_of_stage[g] = [(i, j)] GPU g 里面部署了模型i的第j个stage
    
    start_layer_each_stage: list[list[int]] # [i][j] 模型i的第j个stage的起始层编号
    deployment_of_layer: list[list[set[int]]] # deployment_of_layer[g][l] = [m1, m2, ...] GPU g 里面部署了模型m1, m2, ... 的第l层

    def to_json(self, model_infos, path: str):
        data = {
            "model_names": [mi.name for mi in model_infos],
            "model_types": [mi.type for mi in model_infos],
            "num_layer_each_stage": self.num_layer_each_stage,
            "deployment_of_stage": [
                [[i, j] for (i, j) in g]  # tuple -> list
                for g in self.deployment_of_stage
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f)