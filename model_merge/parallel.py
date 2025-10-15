# 给定融合方案以及负载特征，得到并行化和部署方案
import json
import os
import time
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

from model_merge.dag import GPUInfo, ModelInfo
from model_merge.model_configs.binary_merge_scheme import BinaryMergeScheme, BinaryMergeSchemesManager
from model_merge.model_configs.structures import ParallelScheme
from utils.log_util import log_print

def _check_initial_solution_feasibility(model):
    eps = 1e-6
    violated = []
    for c in model.component_objects(Constraint, active=True):
        cobject = getattr(model, c.name)
        for index in cobject:
            con = cobject[index]
            try:
                val = value(con.body)
                lb = value(con.lower) if con.has_lb() else float('-inf')
                ub = value(con.upper) if con.has_ub() else float('inf')
                if val < lb - eps or val > ub + eps:
                    violated.append((c.name, index, str(con.body)[:50], val, lb, ub))
            except:
                violated.append((c.name, index, str(con.body)[:50], 'EvalError', '-', '-'))

    if not violated:
        log_print("[✓] Initial solution is feasible.")
    else:
        log_print("[✗] Initial solution violates constraints:")
        for name, index, body, val, lb, ub in violated:
            log_print(f"  - {name}[{index}] {body}: value = {val}, bounds = [{lb}, {ub}]")

class ParallelManager:
    gpu_infos: list[GPUInfo]  # GPU 信息列表
    num_GPU: int

    model_type: str
    model_infos: list[ModelInfo]

    mem_lg: np.ndarray  # list[float] 每个 lg 的显存占用
    time_lg: np.ndarray  # 每个 lg 的计算时间
    sum_time_lg: np.ndarray  # time_lg 的前缀和
    num_lg: int  # layer group 数量

    def new_parallel_scheme(self):
        ps = ParallelScheme()
        ps.num_layer_each_stage = [[] for _ in range(len(self.model_infos))]
        ps.deployment_of_stage = [[] for _ in range(self.num_GPU)]

        ps.start_layer_each_stage = [[] for _ in range(len(self.model_infos))]
        ps.deployment_of_layer = [[set() for _ in range(self.num_lg)] for _ in range(self.num_GPU)]
        return ps
    
    def update_parallel_scheme(self, ps:ParallelScheme, model_id:int, layer_num_of_stages, deployment_of_stage):
        num_stage = len(layer_num_of_stages)
        
        ps.num_layer_each_stage[model_id] = [round(layer_num_of_stages[s].value) for s in range(num_stage)]
        
        ps.start_layer_each_stage[model_id] = [0 for _ in range(num_stage)]
        for s in range(1, num_stage):
            ps.start_layer_each_stage[model_id][s] = ps.start_layer_each_stage[model_id][s - 1] + ps.num_layer_each_stage[model_id][s - 1]
        
        for g in range(self.num_GPU):
            for s in range(num_stage):
                if round(deployment_of_stage[g, s].value) == 1:
                    ps.deployment_of_stage[g].append((model_id, s))
                    for l in range(ps.start_layer_each_stage[model_id][s], ps.start_layer_each_stage[model_id][s] + ps.num_layer_each_stage[model_id][s]):
                        ps.deployment_of_layer[g][l].add(model_id)

    def update_gpu_mem(self, ps:ParallelScheme, sm:BinaryMergeSchemesManager) -> list[float]:
        add_mem = [0 for _ in range(self.num_GPU)]
        for g in range(self.num_GPU):
            tmp_deployment_of_layer = [ps.deployment_of_layer[g][l].copy() for l in range(self.num_lg)]
            for code, schemes in sm.schemes.items():
                spliter = sm.spliters[code]
                for s in schemes:
                    for l in range(s.start_layer_index, s.start_layer_index + s.get_num_layers()):
                        for m in ps.deployment_of_layer[g][l]:
                            if m >= spliter.start and m < spliter.end:
                                tmp_deployment_of_layer[l].discard(m)
                                tmp_deployment_of_layer[l].add(spliter.start)
            add_mem[g] = sum(len(tmp_deployment_of_layer[l]) * self.mem_lg[l] for l in range(self.num_lg))
        return add_mem
    
    def print_parallel_scheme(self, ps:ParallelScheme):
        log_print("Pipeline stages:")
        for m in range(len(self.model_infos)):
            scheme_str = f"model {m}: "
            for layers in ps.num_layer_each_stage[m]:
                scheme_str += f"{layers}, "
            log_print(scheme_str)
        log_print("Deployment:")
        for g in range(self.num_GPU):
            log_print(f"GPU {g}: {ps.deployment_of_stage[g]}")

    def __init__(self, model_infos:list[ModelInfo], gpu_infos:list[GPUInfo], model_type:str=None):
        self.gpu_infos = gpu_infos
        self.num_GPU = len(gpu_infos)
        if model_type is None:
            self.model_type = model_infos[0].type['type']
        else:
            self.model_type = model_type
        self.model_infos = model_infos
        self.mem_lg = []
        self.time_lg = []
        json_path = os.path.join(os.path.dirname(__file__), "model_configs/configs", f"{self.model_type}.json")
        with open(json_path, "r") as f:
            data = json.load(f)
            for lg in data:
                self.mem_lg.append(lg["param_size"])
                self.time_lg.append(lg["latency"])
        self.num_lg = len(self.mem_lg)
        self.mem_lg = np.array(self.mem_lg)
        self.time_lg = np.array(self.time_lg)
        self.sum_time_lg = np.cumsum(self.time_lg)
        self.sum_time_lg = np.insert(self.sum_time_lg, 0, 0)  # 在第一位插入0, 方便统计
    
    # 给定并行方案和融合方案，给出最终的内存使用
    def get_mem_use(self, merge_schemes:list[BinaryMergeScheme], parallel_scheme:ParallelScheme) -> tuple[float, float]:
        d = np.zeros((self.num_GPU, len(self.model_infos), self.num_lg), dtype=bool) # d[g, m, l] GPU g 上是否部署了模型 m 的第 l 个 lg
        for g in range(self.num_GPU):
            for l in range(self.num_lg):
                for m in parallel_scheme.deployment_of_layer[g][l]:
                    d[g, m, l] = 1
        total_memory_before_merge = np.sum(d * self.mem_lg[None, None, :])
        for s in merge_schemes:
            for g in range(self.num_GPU):
                for m in s.models:
                    mm = s.models[0]
                    for l in range(s.start_layer_index, s.start_layer_index + s.get_num_layers()):
                        if d[g, m, l] != 0:
                            d[g, m, l] = 0
                            d[g, mm, l] = 1
        total_memory_after_merge = np.sum(d * self.mem_lg[None, None, :])
        return float(total_memory_before_merge), float(total_memory_after_merge)

    # 在已经确认了 accepted_schemes 包含的融合方案后，所有模型的最优并行化方案，包括stage分割、部署、总显存统计
    # vertical_spliters: 生成accepted_schemes的过程中纵向将模型分割成多组的spliter
    def solve(self, accepted_schemes:list[BinaryMergeScheme], vertical_spliters:dict[int,int]) -> tuple[float, ParallelScheme]:
        sm = BinaryMergeSchemesManager(accepted_schemes, vertical_spliters)
        # 先决定高层的融合方案下整体的部署情况。
        # 高层情况下gpu的used_mem中只实际改变高层融合部分的，未融合部分的先不加进去（目的是让参与了融合的模型集中到这几张卡上，充分利用融合的优势）
        # 底层的叶子节点的已确认的具体部署方案再修改used_mem
        # load的话无论高低层都要改变
        ps = self.new_parallel_scheme()
        tmp_gpu_infos = [gi.copy() for gi in self.gpu_infos]
        self._dfs(sm, 1, np.ones((self.num_GPU, self.num_lg)), tmp_gpu_infos, ps)
        self.print_parallel_scheme(ps)
        add_mem = self.update_gpu_mem(ps, sm)
        sum_mem = sum(add_mem)
        return sum_mem, ps


    # 预处理显存消耗
    # mem_x[g][l]: 在 GPU g 上部署层组 l 所占用的显存
    # mem_xy[g][l]: 在 GPU g 上同时部署 x 和 y 的 l 层组所占用的显存
    def get_mem_tables(self, sm:BinaryMergeSchemesManager, code:int, merged_layer_deploy:np.ndarray):
        vertical_spliter = sm.spliters[code]
        x_id = (vertical_spliter.start, vertical_spliter.spliter)
        y_id = (vertical_spliter.spliter, vertical_spliter.end)

        code_x = code << 1
        code_y = code_x + 1
        mem_x = self.mem_lg * (x_id[1] - x_id[0])
        mem_y = self.mem_lg * (y_id[1] - y_id[0])

        # 先计算该融合方案下，部署 x 中的所有模型的层组 l 消耗的显存（总显存减去融合节约的）
        # 此时的 mem_x[l] 为部署层组 l 所占的显存
        for sub_code in sm.sub_code.get(code_x, []):
            for s in sm.schemes.get(sub_code, []):
                for l in range(s.start_layer_index, s.start_layer_index + s.get_num_layers()):
                    if s.layer_groups[l - s.start_layer_index].join_merge:
                        mem_x[l] -= self.mem_lg[l] * (s.num_models - 1)
        for sub_code in sm.sub_code.get(code_y, []):
            for s in sm.schemes.get(sub_code, []):
                for l in range(s.start_layer_index, s.start_layer_index + s.get_num_layers()):
                    if s.layer_groups[l - s.start_layer_index].join_merge:
                        mem_y[l] -= self.mem_lg[l] * (s.num_models - 1)

        # 计算 mem_xy, 初始为不考虑双方融合的情况
        mem_xy = np.zeros((self.num_lg), dtype=float)
        ## 将同时包含x、y的融合方案找出来，记录这些层的显存消耗
        xy_schemes = sm.schemes.get(code, [])
        for s in xy_schemes:
            assert isinstance(s, BinaryMergeScheme)
            l1 = s.start_layer_index
            l2 = l1 + s.get_num_layers()
            for i in range(l1, l2):
                if s.layer_groups[i - l1].join_merge:
                    mem_xy[i] = self.mem_lg[i] * (s.num_models - 1) # 使用该统合方案后单卡上同时部署所有模型减少的显存

        # 在 merged_layer_deploy 的最后加一个全1的列表，代表一个全新的gpu的部署情况
        merged_layer_deploy = np.vstack([merged_layer_deploy, np.ones((1, self.num_lg), dtype=merged_layer_deploy.dtype)])

        # 如果已经部署了更高级的融合层，则额外显存直接归零
        mem_x = mem_x * merged_layer_deploy
        mem_y = mem_y * merged_layer_deploy
        mem_xy = mem_xy * merged_layer_deploy
        # # 将 mem_x 变为前缀和
        # mem_x = np.cumsum(mem_x, axis=1)
        # mem_x = np.insert(mem_x, 0, 0, axis=1)
        # mem_y = np.cumsum(mem_y, axis=1)
        # mem_y = np.insert(mem_y, 0, 0, axis=1)
        # # 将 mem_xy 变为前缀和
        # mem_xy = np.cumsum(mem_xy, axis=1)
        # mem_xy = np.insert(mem_xy, 0, 0, axis=1)

        return [mem_x, mem_y, mem_xy]

    def _solution_to_str(self, model, SX, SY, G, L, len_gpu_infos, num_GPU):
        dx = []
        for s in SX:
            ones = [g for g in G if round(value(model.dx[g, s])) == 1]
            dx.append(ones)
        dy = []
        for s in SY:
            ones = [g for g in G if round(value(model.dy[g, s])) == 1]
            dy.append(ones)
        ans = value(model.obj)
        stage_x = []
        for s in SX:
            stage_x.append({
                "layers": model.nx[s].value,
                "time": value(sum(self.time_lg[l] * model.is_in_stage_x[s, l] for l in L)),
                "mem": value(sum(self.mem_lg[l] * model.is_in_stage_x[s, l] for l in L))
            })
        stage_y = []
        for s in SY:
            stage_y.append({
                "layers": model.ny[s].value,
                "time": value(sum(self.time_lg[l] * model.is_in_stage_y[s, l] for l in L)),
                "mem": value(sum(self.mem_lg[l] * model.is_in_stage_y[s, l] for l in L))
            })
        addi_gpu = 0
        for g in range(len_gpu_infos, num_GPU):
            used = False
            for i in SX:
                if round(model.dx[g, i].value) == 1:
                    used = True
                    break
            if not used:
                for j in SY:
                    if round(model.dy[g, j].value) == 1:
                        used = True
                        break
            if used:
                addi_gpu += 1
        return f"ans: {ans}\naddi_gpu: {addi_gpu}\ndx: {dx}\ndy: {dy}\nstage_x: {str(stage_x)}\nstage_y: {str(stage_y)}\nmin stage time_x: {model.min_stage_time_x.value}\nmin stage time y: {model.min_stage_time_y.value}\nmin stage score: {(model.min_stage_time_x.value + model.min_stage_time_y.value) * (0.5 / self.sum_time_lg[-1])}", addi_gpu >= 1
        
    def ILP(self, sm:BinaryMergeSchemesManager, code:int, gpu_infos:list[GPUInfo], mem_tables) -> tuple[Model, bool]:
        model = ConcreteModel()
        vertical_spliter = sm.spliters[code]
        x_id = (vertical_spliter.start, vertical_spliter.spliter)
        y_id = (vertical_spliter.spliter, vertical_spliter.end)

        # num_stage_x = max(min(self.model_infos[i].max_stage_num, self.model_infos[i].max_load) for i in range(x_id[0], x_id[1]))
        # num_stage_y = max(min(self.model_infos[i].max_stage_num, self.model_infos[i].max_load) for i in range(y_id[0], y_id[1]))
        num_stage_x = min(self.model_infos[i].max_stage_num for i in range(x_id[0], x_id[1]))
        num_stage_y = min(self.model_infos[i].max_stage_num for i in range(y_id[0], y_id[1]))

        max_load_x = max(self.model_infos[i].max_load for i in range(x_id[0], x_id[1]))
        max_load_y = max(self.model_infos[i].max_load for i in range(y_id[0], y_id[1]))
        mean_load_x = sum(self.model_infos[i].mean_load for i in range(x_id[0], x_id[1]))
        mean_load_y = sum(self.model_infos[i].mean_load for i in range(y_id[0], y_id[1]))
        max_load_x = max(max_load_x, mean_load_x)
        max_load_y = max(max_load_y, mean_load_y)

        # 得到分割初始解，尽量平均分stage
        num_layers_each_stage_x = [] # 每个stage的层组数
        start_layer_each_stage_x = [] # 每个stage的起始层编号
        start_layer = 0
        num_layers = 0
        while num_layers < num_stage_x:
            target_t = self.sum_time_lg[-1] * ((num_layers + 1) / num_stage_x)
            for l in range(start_layer, self.num_lg):
                if self.sum_time_lg[l] >= target_t or l == self.num_lg - 1: # 后者是精度原因，没有识别到最后一层
                    start_layer_each_stage_x.append(start_layer)
                    # 以 l 为最后一层更优
                    if l == 0 or self.sum_time_lg[l] - target_t < target_t - self.sum_time_lg[l - 1]:
                        num_layers_each_stage_x.append(l - start_layer + 1)
                        start_layer = l + 1
                    # 以 l-1 为最后一层更优
                    else:
                        num_layers_each_stage_x.append(l - start_layer)
                        start_layer = l
                    num_layers += 1
                    break

        num_layers_each_stage_y = []
        start_layer_each_stage_y = []
        start_layer = 0
        num_layers = 0
        while num_layers < num_stage_y:
            target_t = self.sum_time_lg[-1] * ((num_layers + 1) / num_stage_y)
            for l in range(start_layer, self.num_lg):
                if self.sum_time_lg[l] >= target_t or l == self.num_lg - 1:
                    start_layer_each_stage_y.append(start_layer)
                    if l == 0 or self.sum_time_lg[l] - target_t < target_t - self.sum_time_lg[l - 1]:
                        num_layers_each_stage_y.append(l - start_layer + 1)
                        start_layer = l + 1
                    else:
                        num_layers_each_stage_y.append(l - start_layer)
                        start_layer = l
                    num_layers += 1
                    break

        L = range(self.num_lg)
        SX = range(num_stage_x)
        SY = range(num_stage_y)
        BIG_M = self.num_lg + 10
        EPS = 1e-6

        # 设置每阶段层数的变量
        model.nx = Var(
            SX,
            domain=Integers,
            bounds=(0, self.num_lg),
            initialize={i: num_layers_each_stage_x[i] for i in SX}
        )
        model.ny = Var(
            SY,
            domain=Integers,
            bounds=(0, self.num_lg),
            initialize={i: num_layers_each_stage_y[i] for i in SY}
        )
        # 设置每阶段开始的层的编号
        # 增加约束，每个阶段开始的层+该阶段层数=下个阶段开始的层
        if num_stage_x > 2:
            model.c_stage_num_x = ConstraintList()
            model.sx = Var(
                range(2, num_stage_x), # 第一位必然是0，第二位必然是nx[0]，不设置成变量了
                domain=Integers,
                bounds=(0, self.num_lg-1),
                initialize={i: start_layer_each_stage_x[i] for i in range(2, num_stage_x)}
            )
            model.c_stage_num_x.add(model.nx[0] + model.nx[1] == model.sx[2])
            for i in range(2, num_stage_x-1):
                model.c_stage_num_x.add(model.sx[i] + model.nx[i] == model.sx[i+1])
        else:
            model.sx = None
        if num_stage_y > 2:
            model.c_stage_num_y = ConstraintList()
            model.sy = Var(
                range(2, num_stage_y),
                domain=Integers,
                bounds=(0, self.num_lg-1),
                initialize={i: start_layer_each_stage_y[i] for i in range(2, num_stage_y)}
            )
            model.c_stage_num_y.add(model.ny[0] + model.ny[1] == model.sy[2])
            for i in range(2, num_stage_y-1):
                model.c_stage_num_y.add(model.sy[i] + model.ny[i] == model.sy[i+1])
        else:
            model.sy = None
        # 增加约束，所有阶段要把所有层分完
        model.c_all_layer = ConstraintList()
        if num_stage_x == 1:
            model.c_all_layer.add(model.nx[0] == self.num_lg)
        elif num_stage_x == 2:
            model.c_all_layer.add(model.nx[0] + model.nx[1] == self.num_lg)
        else:
            model.c_all_layer.add(model.sx[num_stage_x-1] + model.nx[num_stage_x-1] == self.num_lg)
        if num_stage_y == 1:
            model.c_all_layer.add(model.ny[0] == self.num_lg)
        elif num_stage_y == 2:
            model.c_all_layer.add(model.ny[0] + model.ny[1] == self.num_lg)
        else:
            model.c_all_layer.add(model.sy[num_stage_y-1] + model.ny[num_stage_y-1] == self.num_lg)

        num_GPU = len(gpu_infos) # 记录初始解所需的 GPU 数量，可能会超过最大GPU数量
        
        # 贪心计算部署初始解，为每个stage，贪心选择额外显存占用最少且不超显存、不爆均值负载的GPU进行部署

        new_gpu_infos = [g.copy() for g in gpu_infos]

        init_dx = [[0 for _ in SX] for _ in range(num_GPU)]
        start_layer = 0
        have_stage = [False for _ in range(num_GPU)]
        for i in SX:
            if num_layers_each_stage_x[i] == 0:
                continue
            # 得到该阶段部署到每个gpu上额外消耗的显存
            end_index = start_layer + num_layers_each_stage_x[i]
            # 该 stage 为 [start_layer:end_index)
            mem_g = [(g, sum(mem_tables[0][g][l] for l in range(start_layer, end_index))) for g in range(self.num_GPU)]
            # 从小到大排序
            mem_g.sort(key=lambda i:i[1])
            # 部署该stage的负载
            stage_time = self.sum_time_lg[end_index] - self.sum_time_lg[start_layer]
            num_gpu_need = ceil(max_load_x / self.sum_time_lg[-1] * stage_time)
            add_load = (mean_load_x / self.sum_time_lg[-1] * stage_time / num_gpu_need) if num_gpu_need > 0 else 0
            num_deployed = 0
            for g, m in mem_g:
                # 判断是否超显存、超负载
                if not have_stage[g] and new_gpu_infos[g].mean_load + add_load <= 1 and new_gpu_infos[g].mem_left() >= m:
                    # 可以部署
                    have_stage[g] = True
                    init_dx[g][i] = 1
                    new_gpu_infos[g].mean_load += add_load
                    new_gpu_infos[g].used_memory += m
                    num_deployed += 1
                    if num_deployed >= num_gpu_need:
                        break
            if num_deployed < num_gpu_need: # 初始解所需的gpu数量不够，虚拟出更多gpu
                for j in range(num_gpu_need - num_deployed):
                    g = GPUInfo(new_gpu_infos[0].total_memory, m, add_load)
                    new_gpu_infos.append(g)
                    init_dx.append([0 for _ in SX])
                    have_stage.append(True)
                    num_GPU += 1
                    init_dx[num_GPU - 1][i] = 1
            start_layer = end_index
        assert start_layer == self.num_lg

        init_dy = [[0 for _ in SY] for _ in range(num_GPU)]
        start_layer = 0
        have_stage = [False for _ in range(num_GPU)]
        for i in SY:
            if num_layers_each_stage_y[i] == 0:
                continue
            # 得到该阶段部署到每个gpu上额外消耗的显存
            end_index = start_layer + num_layers_each_stage_y[i]
            mem_g = [(g, sum(mem_tables[1][g][l] for l in range(start_layer, end_index))) for g in range(self.num_GPU)]
            for g in range(self.num_GPU, num_GPU): # 把虚拟的gpu也给算上
                mem_g.append((g, sum(mem_tables[1][-1][l] for l in range(start_layer, end_index))))
            # 从小到大排序
            mem_g.sort(key=lambda i:i[1])
            # 部署该stage的负载
            stage_time = self.sum_time_lg[end_index] - self.sum_time_lg[start_layer]
            num_gpu_need = ceil(max_load_y / self.sum_time_lg[-1] * stage_time)
            add_load = (mean_load_y / self.sum_time_lg[-1] * stage_time / num_gpu_need) if num_gpu_need > 0 else 0
            num_deployed = 0
            for g, m in mem_g:
                # 判断是否超显存、超负载
                if not have_stage[g] and new_gpu_infos[g].mean_load + add_load <= 1 and new_gpu_infos[g].mem_left() >= m:
                    # 可以部署
                    have_stage[g] = True
                    init_dy[g][i] = 1
                    new_gpu_infos[g].mean_load += add_load
                    new_gpu_infos[g].used_memory += m
                    num_deployed += 1
                    if num_deployed >= num_gpu_need:
                        break
            if num_deployed < num_gpu_need: # 初始解所需的gpu数量不够，虚拟出更多gpu
                for j in range(num_gpu_need - num_deployed):
                    g = GPUInfo(new_gpu_infos[0].total_memory, m, add_load)
                    new_gpu_infos.append(g)
                    init_dy.append([0 for _ in SY])
                    init_dx.append([0 for _ in SX])
                    have_stage.append(True)
                    num_GPU += 1
                    init_dy[num_GPU - 1][i] = 1
            start_layer = end_index
        assert start_layer == self.num_lg
        
        # num_GPU 已经固定下来，可以定义变量以及进行其它操作了
        G = range(num_GPU)
        NG = range(1, num_GPU + 1)

        # 把 mem_table 的三个子表的 gpu 数量维度都延长到 num_GPU ，用补充的最后一个的虚拟 gpu 的数据
        if num_GPU - len(gpu_infos) > 1:
            for i in range(3):
                mem_tables[i] = np.vstack([mem_tables[i], np.tile(mem_tables[i][-1], (num_GPU - len(gpu_infos) - 1, 1))])

        # 设置部署方案变量
        model.dx = Var(G, SX, within=Binary) # dx[g, s] x 的 stage s 部署在了 GPU g 上
        model.dy = Var(G, SY, within=Binary)
        model.dxl = Var(G, L, within=Binary, initialize=0) # dxl[g, l] x 的第 l 层是否部署在GPU g 上
        model.dyl = Var(G, L, within=Binary, initialize=0)
        model.dxyl = Var(G, L, within=Binary, initialize=0) # dxyl[g, l] 在GPU g 上同时部署了x和y的第 l 层
        model.dxl_stage = Var(G, SX, L, within=Binary, initialize=0) # dxl_stage[g, i, l] x 的第 l 层是否部署在GPU g 上，且该层属于stage i
        model.dyl_stage = Var(G, SY, L, within=Binary, initialize=0)
        model.is_in_stage_x = Var(SX, L, within=Binary, initialize=0)
        model.is_in_stage_y = Var(SY, L, within=Binary, initialize=0)
        model.num_g_of_stage_x = Var(SX, NG, within=Binary, initialize=0) # num_g_of_stage_x[s, ng] x的stage s 部署在了一共 ng 张卡上，ng 取值[1, num_GPU]
        model.num_g_of_stage_y = Var(SY, NG, within=Binary, initialize=0)
        model.load_share_of_stage_x = Var(SX, within=NonNegativeReals, initialize=0) # load_share_of_stage_x[s] x的stage s 在每张卡上的平均load
        model.load_share_of_stage_y = Var(SY, within=NonNegativeReals, initialize=0)
        model.load_of_GPU_from_stage_x = Var(G, SX, within=NonNegativeReals, initialize=0) # load_of_GPU_from_stage_x[g, s] = dx[g, s] * load_share_of_stage_x[s]
        model.load_of_GPU_from_stage_y = Var(G, SY, within=NonNegativeReals, initialize=0)

        # 设置部署方案变量之间的约束
        ## 约束 is_in_stage
        model.c_in_stage_x = ConstraintList()
        for s in SX:
            for l in L:
                # l 不属于 [sx[s], sx[s]+nx[s]) ⇒ is_in_stage == 0
                if s == 0:
                    model.c_in_stage_x.add(l >= -BIG_M * (1 - model.is_in_stage_x[s, l]))
                    model.c_in_stage_x.add(l <= model.nx[s] + BIG_M * (1 - model.is_in_stage_x[s, l]) - 1)
                elif s == 1:
                    model.c_in_stage_x.add(l >= model.nx[0] - BIG_M * (1 - model.is_in_stage_x[s, l]))
                    model.c_in_stage_x.add(l <= model.nx[0] + model.nx[s] + BIG_M * (1 - model.is_in_stage_x[s, l]) - 1)
                else:
                    model.c_in_stage_x.add(l >= model.sx[s] - BIG_M * (1 - model.is_in_stage_x[s, l]))
                    model.c_in_stage_x.add(l <= model.sx[s] + model.nx[s] + BIG_M * (1 - model.is_in_stage_x[s, l]) - 1)
            model.c_in_stage_x.add(sum(model.is_in_stage_x[s, l] for l in L) == model.nx[s])
        model.c_in_stage_y = ConstraintList()
        for s in SY:
            for l in L:
                if s == 0:
                    model.c_in_stage_y.add(l >= -BIG_M * (1 - model.is_in_stage_y[s, l]))
                    model.c_in_stage_y.add(l <= model.ny[s] + BIG_M * (1 - model.is_in_stage_y[s, l]) - 1)
                elif s == 1:
                    model.c_in_stage_y.add(l >= model.ny[0] - BIG_M * (1 - model.is_in_stage_y[s, l]))
                    model.c_in_stage_y.add(l <= model.ny[0] + model.ny[s] + BIG_M * (1 - model.is_in_stage_y[s, l]) - 1)
                else:
                    model.c_in_stage_y.add(l >= model.sy[s] - BIG_M * (1 - model.is_in_stage_y[s, l]))
                    model.c_in_stage_y.add(l <= model.sy[s] + model.ny[s] + BIG_M * (1 - model.is_in_stage_y[s, l]) - 1)
            model.c_in_stage_y.add(sum(model.is_in_stage_y[s, l] for l in L) == model.ny[s])

        ## 约束 dxl_stage
        model.c_dl_stage = ConstraintList()
        for g in G:
            for s in SX:
                for l in L:
                    # dxl_stage[g, s, l] = dx[g, s] * is_in_stage[s, l]
                    model.c_dl_stage.add(model.dxl_stage[g, s, l] <= model.dx[g, s])
                    model.c_dl_stage.add(model.dxl_stage[g, s, l] <= model.is_in_stage_x[s, l])
                    model.c_dl_stage.add(model.dxl_stage[g, s, l] >= model.dx[g, s] + model.is_in_stage_x[s, l] - 1)
            for s in SY:
                for l in L:
                    model.c_dl_stage.add(model.dyl_stage[g, s, l] <= model.dy[g, s])
                    model.c_dl_stage.add(model.dyl_stage[g, s, l] <= model.is_in_stage_y[s, l])
                    model.c_dl_stage.add(model.dyl_stage[g, s, l] >= model.dy[g, s] + model.is_in_stage_y[s, l] - 1)

        ## 聚合：dxl[g, l] = sum(dxl_stage[g, s, l] for s in SX)
        for g in G:
            for l in L:
                model.c_dl_stage.add(model.dxl[g, l] == sum(model.dxl_stage[g, s, l] for s in SX))
                model.c_dl_stage.add(model.dyl[g, l] == sum(model.dyl_stage[g, s, l] for s in SY))
        
        ## 约束 dxyl = dxl * dyl
        model.c_dxyl_stage = ConstraintList()
        for g in G:
            for l in L:
                model.c_dxyl_stage.add(model.dxyl[g, l] <= model.dxl[g, l])
                model.c_dxyl_stage.add(model.dxyl[g, l] <= model.dyl[g, l])
                model.c_dxyl_stage.add(model.dxyl[g, l] >= model.dxl[g, l] + model.dyl[g, l] - 1)
        
        ## 约束 num_g_of_stage
        model.c_num_g_of_stage = ConstraintList()
        for s in SX:
            model.c_num_g_of_stage.add(sum(model.num_g_of_stage_x[s, ng] for ng in NG) == 1)
            model.c_num_g_of_stage.add(sum(model.num_g_of_stage_x[s, ng] * ng for ng in NG) == sum(model.dx[g, s] for g in G))
        for s in SY:
            model.c_num_g_of_stage.add(sum(model.num_g_of_stage_y[s, ng] for ng in NG) == 1)
            model.c_num_g_of_stage.add(sum(model.num_g_of_stage_y[s, ng] * ng for ng in NG) == sum(model.dy[g, s] for g in G))

        ## 约束 load_share_of_stage_x
        model.c_load_share_of_stage = ConstraintList()
        for s in SX:
            sum_load_of_stage = mean_load_x / self.sum_time_lg[-1] * sum(self.time_lg[l] * model.is_in_stage_x[s, l] for l in L)
            for ng in NG:
                model.c_load_share_of_stage.add(model.load_share_of_stage_x[s] >=
                 sum_load_of_stage/ng - BIG_M * (1 - model.num_g_of_stage_x[s, ng]) - EPS)
                model.c_load_share_of_stage.add(model.load_share_of_stage_x[s] <=
                 sum_load_of_stage/ng + BIG_M * (1 - model.num_g_of_stage_x[s, ng]) + EPS)
        for s in SY:
            sum_load_of_stage = mean_load_y / self.sum_time_lg[-1] * sum(self.time_lg[l] * model.is_in_stage_y[s, l] for l in L)
            for ng in NG:
                model.c_load_share_of_stage.add(model.load_share_of_stage_y[s] >=
                 sum_load_of_stage/ng - BIG_M * (1 - model.num_g_of_stage_y[s, ng]) - EPS)
                model.c_load_share_of_stage.add(model.load_share_of_stage_y[s] <=
                 sum_load_of_stage/ng + BIG_M * (1 - model.num_g_of_stage_y[s, ng]) + EPS)

        ## 约束 load_of_GPU_from_stage_x
        model.load_of_GPU_from_stage = ConstraintList()
        for g in G:
            for s in SX:
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_x[g, s] <= model.load_share_of_stage_x[s] + EPS)
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_x[g, s] <= BIG_M * model.dx[g, s])
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_x[g, s] >= model.load_share_of_stage_x[s] - BIG_M * (1 - model.dx[g, s]) - EPS)
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_x[g, s] >= 0)
            for s in SY:
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_y[g, s] <= model.load_share_of_stage_y[s] + EPS)
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_y[g, s] <= BIG_M * model.dy[g, s])
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_y[g, s] >= model.load_share_of_stage_y[s] - BIG_M * (1 - model.dy[g, s]) - EPS)
                model.load_of_GPU_from_stage.add(model.load_of_GPU_from_stage_y[g, s] >= 0)
                

        # 初始化
        for g in G:
            for s in SX:
                model.dx[g, s].value = init_dx[g][s]
                for l in range(start_layer_each_stage_x[s], start_layer_each_stage_x[s] + num_layers_each_stage_x[s]):
                    model.is_in_stage_x[s, l].value = 1
                    if init_dx[g][s]:
                        model.dxl_stage[g, s, l].value = 1
                        model.dxl[g, l].value = 1
            for s in SY:
                model.dy[g, s].value = init_dy[g][s]
                for l in range(start_layer_each_stage_y[s], start_layer_each_stage_y[s] + num_layers_each_stage_y[s]):
                    model.is_in_stage_y[s, l].value = 1
                    if init_dy[g][s]:
                        model.dyl_stage[g, s, l].value = 1
                        model.dyl[g, l].value = 1
            for l in L:
                model.dxyl[g, l].value = model.dxl[g, l].value * model.dyl[g, l].value
        for s in SX:
            ng = value(sum(model.dx[g, s] for g in G))
            model.num_g_of_stage_x[s, ng].value = 1
            sum_load_of_stage = value(mean_load_x / self.sum_time_lg[-1] * sum(self.time_lg[l] * model.is_in_stage_x[s, l] for l in L))
            model.load_share_of_stage_x[s] = sum_load_of_stage / ng
        for s in SY:
            ng = value(sum(model.dy[g, s] for g in G))
            model.num_g_of_stage_y[s, ng].value = 1
            sum_load_of_stage = value(mean_load_y / self.sum_time_lg[-1] * sum(self.time_lg[l] * model.is_in_stage_y[s, l] for l in L))
            model.load_share_of_stage_y[s] = sum_load_of_stage / ng
        for g in G:
            for s in SX:
                model.load_of_GPU_from_stage_x[g, s] = model.load_share_of_stage_x[s] * model.dx[g, s]
            for s in SY:
                model.load_of_GPU_from_stage_y[g, s] = model.load_share_of_stage_y[s] * model.dy[g, s]

        model.c_deploy = ConstraintList()
        # 每张卡只能部署一个stage
        for g in G:
            model.c_deploy.add(sum(model.dx[g, s] for s in SX) <= 1)
            model.c_deploy.add(sum(model.dy[g, s] for s in SY) <= 1)
        # 每个stage部署的卡数量得够
        for s in SX:
            stage_time = sum(self.time_lg[l] * model.is_in_stage_x[s, l] for l in L) / self.sum_time_lg[-1]
            model.c_deploy.add(sum(model.dx[g, s] for g in G) >= (max_load_x * stage_time - EPS))
        for s in SY:
            stage_time = sum(self.time_lg[l] * model.is_in_stage_y[s, l] for l in L) / self.sum_time_lg[-1]
            model.c_deploy.add(sum(model.dy[g, s] for g in G) >= (max_load_y * stage_time - EPS))

        # 每个gpu不能超负载
        ## 初始化 new_gpu_infos ，把新加的gpuinfo归零
        new_gpu_infos = [g.copy() for g in gpu_infos]
        for _ in range(num_GPU - len(gpu_infos)):
            new_gpu_infos.append(GPUInfo(gpu_infos[0].total_memory))
        ## 约束负载
        for g in G:
            model.c_deploy.add(sum(model.load_of_GPU_from_stage_x[g, s] for s in SX)
                            + sum(model.load_of_GPU_from_stage_y[g, s] for s in SY)
                            <= 1 - new_gpu_infos[g].mean_load)

        # 计算 total_add_mem; 增加约束：每个gpu不能超显存
        total_add_mem = 0
        for g in G:
            add_mem = 0
            for l in L:
                add_mem += model.dxl[g, l] * mem_tables[0][g][l]
                add_mem += model.dyl[g, l] * mem_tables[1][g][l]
                add_mem -= model.dxyl[g, l] * mem_tables[2][g][l]
            model.c_deploy.add(add_mem <= new_gpu_infos[g].mem_left())
            total_add_mem += add_mem
        
        # 计算额外使用的 gpu
        addi_gpu = 0
        for g in range(len(gpu_infos), num_GPU):
            for i in SX:
                addi_gpu += model.dx[g, i] * (g - len(gpu_infos) + 2)
            for j in SY:
                addi_gpu += model.dy[g, j] * (g - len(gpu_infos) + 2)
        
        # 最大化两边每个阶段的执行时间的最小值，在最小显存目标不变的情况下
        # 即分别最大化 min(sum(model.is_in_stage_x[s, l] * self.time_lg[l] for l in L) for s in SX)，min(sum(model.is_in_stage_y[s, l] * self.time_lg[l] for l in L) for s in SY)
        model.min_stage_time_x = Var(within=NonNegativeReals)
        model.min_stage_time_y = Var(within=NonNegativeReals)

        # 添加约束：min_stage_time_x ≤ 每个 stage x 的时间
        model.c_min_stage_time_x = ConstraintList()
        for s in SX:
            model.c_min_stage_time_x.add(
                model.min_stage_time_x <= sum(model.is_in_stage_x[s, l] * self.time_lg[l] for l in L)
            )
        model.min_stage_time_x.value = min(sum(model.is_in_stage_x[s, l].value * self.time_lg[l] for l in L) for s in SX)

        # 添加约束：min_stage_time_y ≤ 每个 stage y 的时间
        model.c_min_stage_time_y = ConstraintList()
        for s in SY:
            model.c_min_stage_time_y.add(
                model.min_stage_time_y <= sum(model.is_in_stage_y[s, l] * self.time_lg[l] for l in L)
            )
        model.min_stage_time_y.value = min(sum(model.is_in_stage_y[s, l].value * self.time_lg[l] for l in L) for s in SY)
        
        # 平衡系数，使得这个优化是在最小化内存之后的
        # 这俩单独的最大值都是 self.sum_time_lg[-1], 我们要保证最终值<=1，因此系数为0.5/self.sum_time_lg[-1]
        max_min_stage = (model.min_stage_time_x + model.min_stage_time_y) * (0.5 / self.sum_time_lg[-1])

        # 目标函数，最小化 total_add_mem, 并使得 addi_gpu 尽量为 0
        model.obj = Objective(expr=total_add_mem + addi_gpu * self.gpu_infos[0].total_memory - max_min_stage, sense=minimize)
        # solver = SolverFactory('bonmin')
        # solver.options['bonmin.algorithm'] = 'B-iFP'
        # solver.options['bonmin.allowable_gap'] = 1e-2               # 绝对容差
        # solver.options['bonmin.allowable_fraction_gap'] = 1e-2      # 相对容差（1%）
        # solver.options['bonmin.time_limit'] = 600

        # solver = SolverFactory('cbc')
        # solver.options['seconds'] = 60  # 时间限制
        # solver.options['ratio'] = 0.01  # 相对最优间隙
        # solver.options["usehotstart"] = "yes"  # 仅适用于 CBC 支持热启动版本

        # solver = SolverFactory("scip")
        # solver.options["limits/time"] = 300
        # solver.options["parallel/maxnthreads"] = 32  # 只能控制 presolve 线程

        # 使用 Gurobi 求解
        solver = SolverFactory('gurobi')
        # 可选设置 Gurobi 参数
        solver.options['Threads'] = 50
        # 使用 50 个线程并行求解（视 CPU 核心数调整）
        solver.options['TimeLimit'] = 300

        solver.options['MIPFocus'] = 1
        # 更关注从初始解出发寻找更好的可行解

        solver.options['Heuristics'] = 0.3
        # 恢复适度启发式，用于基于初始解找邻域解（默认是0.05~0.5）

        solver.options['ImproveStartGap'] = 0.05
        # 当 gap 小于 5% 时才允许替换初始解，避免用差解覆盖好解

        # solver.options['ImproveStartTime'] = 10
        # 求解第10秒后才开始尝试改善初始解，保留初始阶段集中处理

        # solver.options['StartNodeLimit'] = 100
        # 限制只在初始解附近探索100个节点，避免乱跳全图

        solver.options['Presolve'] = 2

        solver.options['FeasibilityTol'] = 1e-9
        solver.options['IntFeasTol'] = 1e-9
        solver.options['OptimalityTol'] = 1e-9
        # 整数可行性容差设置得更严格（默认 1e-5 已合理，但可按需再减小）

        init_solution_str, _ = self._solution_to_str(model, SX, SY, G, L, len(gpu_infos), num_GPU)
        log_print(f"----------------------------code:{code}----------------------------")
        log_print(f"init solution:\n{init_solution_str}", flush=True)
        _check_initial_solution_feasibility(model)

        t_start = time.time()
        results = solver.solve(model, tee=False)
        t_end = time.time()

        tc = results.solver.termination_condition
        if tc in [TerminationCondition.optimal, TerminationCondition.feasible, TerminationCondition.maxIterations, TerminationCondition.maxTimeLimit]:
            solution_str, invalid = self._solution_to_str(model, SX, SY, G, L, len(gpu_infos), num_GPU)
            log_print(f"time_cost: {t_end - t_start}\n{solution_str}\ntermination condition: {tc}", flush=True)
            if invalid:
                log_print(f"Imposible to handle given models with given GPUs, generating temporary solution")
                return model, False # 没有可行解
            return model, True
        else:
            log_print(f"code:{code}\ntime_cost: {t_end - t_start}\ntermination condition: {tc}", flush=True)
            return model, False # 没有可行解

    # merged_layer_deploy[i, j]: GPUi上 !!没有部署!! 融合层j，这里的融合层j是跨越了x、y以及其它模型的，只有这些层会影响x、y求解时的显存占用
    # num_layer_each_stage 和 deployment_of_stage 用来记录划分和部署方案
    def _dfs(self, sm:BinaryMergeSchemesManager, code:int, merged_layer_deploy:np.ndarray, gpu_infos:list[GPUInfo], ps: ParallelScheme):
        vertical_spliter = sm.spliters[code]
        mem_tables = self.get_mem_tables(sm, code, merged_layer_deploy)
        model, valid = self.ILP(sm, code, gpu_infos, mem_tables)
        if not valid:
            return False
        num_stage_x = len(model.nx)
        num_stage_y = len(model.ny)
        # 往下递归求解
        if vertical_spliter.end - vertical_spliter.start > 2:
            # 存在至少一边需要进一步递归，更新 merged_layer_deploy，把该 code 的 scheme 加入 merged_layer_deploy
            for s in sm.schemes.get(code, []):
                for i, lg in enumerate(s.layer_groups):
                    if not lg.join_merge:
                        continue
                    l = i + s.start_layer_index
                    for g in range(self.num_GPU):
                        if round(model.dxl[g, l].value) == 1 or round(model.dyl[g, l].value) == 1:
                            merged_layer_deploy[g, l] = 0

        if vertical_spliter.spliter - vertical_spliter.start > 1:
            # 左边需要进一步递归
            # 将右边的load记录进gpu_infos中
            for g, gi in enumerate(gpu_infos):
                for s in range(num_stage_y):
                    gi.mean_load += model.load_of_GPU_from_stage_y[g, s].value

            self._dfs(sm, code * 2, merged_layer_deploy.copy(), gpu_infos, ps) # 使用 copy 的 merged_layer_deploy，方便右边接着用

            # 去掉右边的load
            for g, gi in enumerate(gpu_infos):
                for s in range(num_stage_y):
                    gi.mean_load -= model.load_of_GPU_from_stage_y[g, s].value
        else:
            # 左边是叶子节点，直接确定其并行化部署
            m = vertical_spliter.start
            # 记录部署方案
            self.update_parallel_scheme(ps, m, model.nx, model.dx)
            # 更新 gpu_infos 的显存信息
            add_mem = self.update_gpu_mem(ps, sm)
            for g in range(self.num_GPU):
                gpu_infos[g].used_memory = add_mem[g] + self.gpu_infos[g].used_memory # 在原始的基础上增加
            # 记录左边的 load 信息
            for g, gi in enumerate(gpu_infos):
                for s in range(num_stage_x):
                    gi.mean_load += model.load_of_GPU_from_stage_x[g, s].value

        if vertical_spliter.end - vertical_spliter.spliter > 1:
            # 右边需要进一步递归，此时的 merged_layer_deploy 只包含了该层code及以上的，没有问题；gpu_infos包含了左边的所有模型，也没问题；
            self._dfs(sm, code * 2 + 1, merged_layer_deploy, gpu_infos, ps)
        else:
            # 右边是叶子节点，直接确定其并行化部署
            m = vertical_spliter.spliter
            # 记录部署方案
            self.update_parallel_scheme(ps, m, model.ny, model.dy)
            # 修改gpu_infos的显存信息
            add_mem = self.update_gpu_mem(ps, sm)
            for g in range(self.num_GPU):
                gpu_infos[g].used_memory = add_mem[g] + self.gpu_infos[g].used_memory # 在原始的基础上增加
            # 记录右边的 load 信息
            for g, gi in enumerate(gpu_infos):
                for s in range(num_stage_y):
                    gi.mean_load += model.load_of_GPU_from_stage_y[g, s].value
