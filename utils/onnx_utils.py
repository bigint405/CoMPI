import time
from typing import Dict, Iterable, List
import numpy as np
import onnx
import torch
import torch.onnx
from onnx import ModelProto, NodeProto, ValueInfoProto, TensorProto, TypeProto
import json
import os
import onnxruntime as ort



def torch2onnx(model, dummy_input, onnx_model_path, input_names=['modelInput'], output_names=['modelOutput'], dynamic_axes={'modelInput' : {0 : 'batch_size'},'modelOutput' : {0 : 'batch_size'}}):
    # 设置模型为评估模式
    model.eval()
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path, 
        export_params=True,  # 导出模型的参数
        # opset_version=12,  # ONNX opset version
        do_constant_folding=True,  # 是否进行常量折叠
        input_names=input_names,  # 输入名称
        output_names=output_names,  # 输出名称
        dynamic_axes=dynamic_axes,
        verbose=True
    )


def huggingface2onnx(model_name:str, dir_path:str, model_path):
    os.system(f"optimum-cli export onnx --model {model_name} {dir_path}")
    os.system(f"mv {os.path.join(dir_path, 'model.onnx')} {model_path}")


def search_outputs(nodes:Iterable[NodeProto]):
    # 获取所有节点的输入输出名称
    all_inputs = set()
    all_outputs = []
    for node in nodes:
        all_inputs.update(node.input)
        all_outputs.append(node.output)

    # 找出没有后续子节点的输出
    no_child_outputs = [output for output in all_outputs if output not in all_inputs]

    return no_child_outputs


def nodeProto2json(node: NodeProto):
    s = {
        'name': node.name,
        'op_type': node.op_type,
        'domain': node.domain,
        'overload': node.overload,
        'doc_string': node.doc_string,
        'input': node.input._values,
        'output': node.output._values
    }
    return json.dumps(s, indent=4)


# 转换 ValueInfoProto 为 JSON
def valueInfoProto2json(value_info: ValueInfoProto):
    # 获取基本信息
    value_info_dict = {
        "name": value_info.name,
        "type": {
            "data_type": value_info.type.tensor_type.elem_type,
            "shape": []
        },
        "doc_string": value_info.doc_string,
        "metadata_props": []
    }
    
    # 获取 metadata_props 信息
    for entry in value_info.metadata_props:
        value_info_dict["metadata_props"].append({
            "key": entry.key,
            "value": entry.value
        })

    for dim in value_info.type.tensor_type.shape.dim:
        value_info_dict["type"]["shape"].append({
            "dim_value": dim.dim_value,
            "dim_param": dim.dim_param,
            "denotation": dim.denotation
        })
    
    # 转换成 JSON 字符串
    return json.dumps(value_info_dict, indent=4)


def typeProto2json(type_proto: TypeProto):
    type_proto_dict = {
        "tensor_type": {
            "elem_type": type_proto.tensor_type.elem_type,
            "shape": [dim.dim_value if dim.HasField('dim_value') else 'None' for dim in type_proto.tensor_type.shape.dim]
        }
    }
    return json.dumps(type_proto_dict, indent=4)


# 根据模型中的给定向量进行分割。
# 返回值：
#   (True, (model1, model2)) 合法划分
#   (False, stats) 非法划分 stats: 0:没有找到属于第一个阶段的节点; 1:需要更多的输入向量
def segment_onnx_with_tensor(onnx_model: ModelProto, sep_tensor_names: Iterable[str], sep_tensor_shapes, opset_version=21):
    nodes_name1 = set()
    nodes1 = []
    nodes2 = []
    nodes1_inputs = set()
    nodes2_inputs = set() # 用来识别initializer

    nodes1_inputs.update(sep_tensor_names)

    # 寻找第一阶段的节点，需要得出所有中间向量
    for _ in range(len(onnx_model.graph.node)):
        changed = False
        for node in onnx_model.graph.node:
            if not node.name in nodes_name1:
                found = False
                for output in node.output:
                    if output in nodes1_inputs:
                        found = True
                        # 把该node所有input放进集合里
                        nodes1_inputs.update(node.input)
                        # 该node加入nodes1
                        nodes_name1.add(node.name)
                        nodes1.append(node)
                        break
                if found:
                    changed = True
        if not changed:
            break
    
    if len(nodes1) == 0: # 没有找到属于第一个阶段的节点
        return False, 0
    
    # 剩下的都属于第二阶段
    for node in onnx_model.graph.node:
        if not node.name in nodes_name1:
            nodes2.append(node)
            nodes2_inputs.update(node.input)

    initializer1 = []
    initializer2 = []

    for i in onnx_model.graph.initializer:
        if i.name in nodes1_inputs:
            initializer1.append(i)
        if i.name in nodes2_inputs:
            initializer2.append(i)
            nodes2_inputs.discard(i.name) # 除去所有参数，统计阶段2需要的输入向量
    
    for node in nodes2:
        for output in node.output:
            nodes2_inputs.discard(output) # 除去自己的输出

    opset_id = onnx.helper.make_opsetid("ai.onnx", opset_version)

    if len(nodes2_inputs) != len(sep_tensor_names):
        # 阶段2需要更多的输入向量
        return False, 1

    mid_tensors = []
    for name, shape in zip(sep_tensor_names, sep_tensor_shapes):
        mid_tensors.append(onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

    model1_graph = onnx.helper.make_graph(
        nodes=nodes1,
        name=onnx_model.graph.name + "1",
        inputs=onnx_model.graph.input,
        outputs=mid_tensors,
        initializer=initializer1
    )

    m1 = onnx.helper.make_model(model1_graph, opset_imports=[opset_id])

    model2_graph = onnx.helper.make_graph(
        nodes=nodes2,
        name=onnx_model.graph.name + "2",
        inputs=mid_tensors,
        outputs=onnx_model.graph.output,
        initializer=initializer2
    )

    m2 = onnx.helper.make_model(model2_graph, opset_imports=[opset_id])

    return (True, (m1, m2))

# 根据模型中特定NodeProto进行完全分割，后半部分包括seperator。
# 返回值：
#   (True, (model1, model2)) 合法划分
#   (False, node) 非法划分，点node同时处于两端
def segment_onnx(onnx_model: ModelProto, seperators: Iterable[NodeProto], sep_tensor_shape, opset_version=21):
    nodes_name1 = set()
    nodes_name2 = set([i.name for i in seperators])
    nodes1 = []
    nodes2 = [i for i in seperators]
    nodes1_inputs = set()
    nodes2_outputs = set()

    for node in seperators:
        nodes1_inputs.update(node.input)
        nodes2_outputs.update(node.output)

    nodes2_inputs = nodes1_inputs.copy() # 用来识别initializer
    mid_inputs = nodes1_inputs.copy() # 用来识别initializer
    # print(nodes1_inputs)

    for _ in range(len(onnx_model.graph.node)):
        if len(nodes1) + len(nodes2) > len(onnx_model.graph.node):
            return False, None # 有点同时在上方和下方，非法分割。理论上不应该进入该分支，非法分割应该在下面就能识别出来
        if len(nodes1) + len(nodes2) == len(onnx_model.graph.node):
            break
        for node in onnx_model.graph.node:
            if not node.name in nodes_name1:
                found = False
                for output in node.output:
                    if output in nodes1_inputs:
                        found = True
                        # 把该node所有input放进集合里
                        for input in node.input:
                            nodes1_inputs.add(input)
                        # 该node加入nodes1
                        nodes_name1.add(node.name)
                        nodes1.append(node)
                        break
                if found and node.name in nodes_name2: # 同时在上方和下方，非法分割，返回对应点
                    return False, node
            if not node.name in nodes_name2:
                found = False
                for input in node.input:
                    if input in nodes2_outputs:
                        found = True
                        # 把该node所有input放进集合里
                        for output in node.output:
                            nodes2_outputs.add(output)
                        for i in node.input:
                            nodes2_inputs.add(i)
                        # 该node加入nodes2
                        nodes_name2.add(node.name)
                        nodes2.append(node)
                        break
                if found and node.name in nodes_name1: # 同时在上方和下方，非法分割，返回对应点
                    return False, node

    initializer1 = []
    initializer2 = []
    
    for node in seperators: # 防止initializer1中加入seperator的参数
        for i in node.input:
            nodes1_inputs.remove(i)

    for i in onnx_model.graph.initializer:
        if i.name in nodes1_inputs:
            initializer1.append(i)
            mid_inputs.discard(i.name) # 筛选出中间变量，刨去模型参数
        if i.name in nodes2_inputs:
            initializer2.append(i)
            mid_inputs.discard(i.name)

    opset_id = onnx.helper.make_opsetid("ai.onnx", opset_version)

    mid_tensors = []
    for i in mid_inputs:
        mid_tensors.append(onnx.helper.make_tensor_value_info(i, TensorProto.FLOAT, sep_tensor_shape[i]))

    model1_graph = onnx.helper.make_graph(
        nodes=nodes1,
        name=onnx_model.graph.name + "1",
        inputs=onnx_model.graph.input,
        outputs=mid_tensors,
        initializer=initializer1
    )

    m1 = onnx.helper.make_model(model1_graph, opset_imports=[opset_id])

    model2_graph = onnx.helper.make_graph(
        nodes=nodes2,
        name=onnx_model.graph.name + "2",
        inputs=mid_tensors,
        outputs=onnx_model.graph.output,
        initializer=initializer2
    )

    m2 = onnx.helper.make_model(model2_graph, opset_imports=[opset_id])

    return (True, (m1, m2))

def segment_onnx_with_tensor_util(model_path, sep_tensor_names:List[Iterable[str]], sep_tensor_shapes, out_path="split_models", batch_size=1):
    onnx_model = onnx.load(model_path)
    part_num = 0
    os.makedirs(out_path, exist_ok=True)
    print(f"共 {len(sep_tensor_names)} 个断点：")
    for names, shapes in zip(sep_tensor_names, sep_tensor_shapes):
        valid, ans = segment_onnx_with_tensor(onnx_model, names, shapes)
        if valid:
            onnx.save(ans[0], os.path.join(out_path, f"part{part_num}.onnx"))
            if part_num == len(sep_tensor_names) - 1:
                onnx.save(ans[1], os.path.join(out_path, f"part{part_num+1}.onnx"))
                print(f"断点 {part_num+1} 已成功拆分并保存。")
            else:
                onnx_model = ans[1]
            print(f"断点 {part_num} 已成功拆分并保存。")
            del ans
        else:
            if not ans is None:
                print(f"断点 {part_num} 拆分失败，status: {ans}")
            raise Exception('ONNX 模型拆分失败')
        part_num += 1
    
    # 验证
    providers = ['CUDAExecutionProvider']
    full_session = ort.InferenceSession(model_path, providers=providers)
    full_inputs = full_session.get_inputs()
    full_input_names = [inp.name for inp in full_inputs]
    full_input_shapes = [
        [dim if isinstance(dim, int) and dim > 0 else batch_size for dim in inp.shape]
        for inp in full_inputs
    ]
    # 构造所有输入
    input_data = [np.random.rand(*shape).astype(np.float32) for shape in full_input_shapes]
    input_feed = {name: data for name, data in zip(full_input_names, input_data)}
    full_outputs = full_session.run(None, input_feed)
    del full_session

    data = input_data  # 这里 data 是 list
    split_outputs = None

    # 统计每个 part 的运行时间
    for i in range(len(sep_tensor_names) + 1):
        session = ort.InferenceSession(os.path.join(out_path, f"part{i}.onnx"), providers=providers)
        inputs = session.get_inputs()
        input_names = [inp.name for inp in inputs]
        input_shapes = [
            [dim if isinstance(dim, int) and dim > 0 else batch_size for dim in inp.shape]
            for inp in inputs
        ]
        # 如果是第一段，data 已经准备好；否则 data 需要是上一个输出
        if i == 0:
            cur_input_data = data
        else:
            # 上一个输出作为当前输入
            cur_input_data = output
            # 如果只有一个输入，output 可能是单个 ndarray，需要包装成 list
            if len(input_names) == 1 and not isinstance(cur_input_data, list):
                cur_input_data = [cur_input_data]
        input_feed = {name: d for name, d in zip(input_names, cur_input_data)}

        # 统计运行时间
        for j in range(50):
            if j == 10:
                start_time = time.time()
            output = session.run(None, input_feed)
        end_time = time.time()
        # 多输出时 output 是 list
        data = output  # 作为下一个 part 的输入

        elapsed_time = end_time - start_time
        print(f"Part {i} 运行 {40} 个 batch 的总时间: {elapsed_time * 1e3:.6f} 毫秒")
        print(f"Part {i} 平均每个 batch 的时间: {elapsed_time / 40 * 1e3:.6f} 毫秒")

        if i == len(sep_tensor_names):
            split_outputs = output

    # 比较所有输出
    max_diff = 0.0
    for full_out, split_out in zip(full_outputs, split_outputs):
        diff = np.abs(full_out - split_out).max()
        max_diff = max(max_diff, diff)
    print(f"拆分前后输出最大差值: {max_diff:.6e}")


class SplitNode:
    name: str
    input_name: str
    input_shape: List

    def __init__(self, name, input_name, input_shape):
        self.name = name
        self.input_name = input_name
        self.input_shape = input_shape


def segment_onnx_util(model_path, nodes:List[SplitNode], out_path="split_models", batch_size=1):
    onnx_model = onnx.load(model_path)
    part_num = 0
    os.makedirs(out_path, exist_ok=True)
    print(f"共 {len(nodes)} 个断点：")
    for nodeinfo in nodes:
        for node in onnx_model.graph.node:
            if node.name == nodeinfo.name:
                valid, ans = segment_onnx(onnx_model, [node], {nodeinfo.input_name: nodeinfo.input_shape}) # 大小不知道就随便写，根据报错的提示信息来
                if valid:
                    onnx.save(ans[0], os.path.join(out_path, f"part{part_num}.onnx"))
                    if part_num == len(nodes) - 1:
                        onnx.save(ans[1], os.path.join(out_path, f"part{part_num+1}.onnx"))
                        print(f"断点 {part_num+1} 已成功拆分并保存。")
                    else:
                        onnx_model = ans[1]
                    print(f"断点 {part_num} 已成功拆分并保存。")
                    del ans
                else:
                    if not ans is None:
                        print(nodeProto2json(ans))
                    f"断点 {part_num} 拆分失败。"
                    raise Exception('ONNX 模型拆分失败')
                break
        part_num += 1
    
    # 验证
    providers=['CUDAExecutionProvider']
    full_session = ort.InferenceSession(model_path, providers=providers)
    full_input_name = full_session.get_inputs()[0].name
    full_input_shape = full_session.get_inputs()[0].shape
    full_input_shape = [dim if isinstance(dim, int) and dim > 0 else batch_size for dim in full_input_shape]
    input_data = np.random.rand(*full_input_shape).astype(np.float32)
    full_outputs = full_session.run(None, {full_input_name: input_data})
    del full_session

    data = input_data
    split_outputs = None

    # 统计每个 part 的运行时间
    for i in range(len(nodes) + 1):
        session = ort.InferenceSession(os.path.join(out_path, f"part{i}.onnx"), providers=providers)
        input_name = session.get_inputs()[0].name

        # 统计运行时间
        for j in range(50):
            if j == 10:
                start_time = time.time()
            output = session.run(None, {input_name: data})
        end_time = time.time()
        data = output[0]

        elapsed_time = end_time - start_time
        print(f"Part {i} 运行 {40} 个 batch 的总时间: {elapsed_time * 1e3:.6f} 毫秒")
        print(f"Part {i} 平均每个 batch 的时间: {elapsed_time / 40 * 1e3:.6f} 毫秒")

        if i == len(nodes):
            split_outputs = output

    final_diff = np.abs(full_outputs[0] - split_outputs[0]).max()
    print(f"拆分前后输出最大差值: {final_diff:.6e}")


class OnnxSessionManager:
    sessions: List[Dict[str, ort.InferenceSession]] # device_rank -> {stage_id -> session}

    def __init__(self, device_num:int):
        self.sessions = [{} for _ in range(device_num)]

    def get_session(self, rank:int, stage_id:str):
        return self.sessions[rank].get(stage_id)

    def set_session(self, rank:int, stage_id:str, ses: ort.InferenceSession):
        self.sessions[rank][stage_id] = ses

if __name__ == '__main__':
    # segment_onnx_util("/root/datas/models/resnet152-1000/resnet152.onnx",[
    #                       SplitNode("/layer3/layer3.10/relu_2/Relu", "/layer3/layer3.10/Add_output_0", ['batchsize', 1024, 14, 14]),
    #                     #   SplitNode("/Flatten", "/avgpool/GlobalAveragePool_output_0", ['batchsize', 2048, 1, 1])
    #                       ],"test/models")
    # segment_onnx_with_tensor_util("gemel_nsdi23/test/models/resnet50.onnx", [["/layer3/layer3.3/relu_2/Relu_output_0", "/layer3/layer3.4/relu/Relu_output_0"]], [[['batchsize', 1024, 14, 14], ['batchsize', 256, 14, 14]]],"test/models_seg")
    segment_onnx_with_tensor_util("/workspace/datas/models/resnet152-1000/resnet152.onnx", [["/layer3/layer3.9/relu_2/Relu_output_0"]], [[['batchsize', 1024, 14, 14]]],"test/models_seg",batch_size=16)
