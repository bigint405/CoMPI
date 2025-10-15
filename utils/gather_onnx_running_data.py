import gc
import onnxruntime as ort
import torch
import numpy as np
import pynvml
import threading
import time
import nvtx

def get_onnx_running_data(model_path, num_instances=1, batchsize=1):
    # **8️⃣ 释放 GPU 资源**
    # print("🧹 Cleaning up GPU memory...")
    # torch.cuda.empty_cache()  # **清空 CUDA 缓存**
    # torch.cuda.synchronize()  # **确保清理完成**

    # **初始化 pynvml**
    pynvml.nvmlInit()
    device_id = 0  # **监测的 GPU**
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    # **监测 GPU 显存**
    def get_gpu_memory():
        """获取当前 GPU 显存占用（B，MB）"""
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used, info.free  # **单位：B（字节）**

    # **实时监测推理过程中的最大显存**
    def monitor_gpu_usage(stop_event, max_mem_list):
        """GPU 监测线程：持续记录最大显存占用"""
        max_mem_usage = 0
        while not stop_event.is_set():
            current_mem, _ = get_gpu_memory()
            max_mem_usage = max(max_mem_usage, current_mem)
            time.sleep(0.01)  # **10ms 轮询一次**
        max_mem_list.append(max_mem_usage)  # **存储最终的最大值**

    # **ONNX 运行设备**
    providers = [("CUDAExecutionProvider", {"device_id": device_id, "arena_extend_strategy": "kSameAsRequested"})]
    # providers = [("CUDAExecutionProvider", {"device_id": device_id, "arena_extend_strategy": "kSameAsRequested"})]

    # **1️⃣ 监测模型加载显存**
    mem_before_load, free_before_load = get_gpu_memory()
    print(f"📌 GPU memory used before load: {mem_before_load} B ({mem_before_load / 1024**2:.2f} MB)")
    print(f"📌 GPU free memory before load: {free_before_load} B ({free_before_load / 1024**2:.2f} MB)")
    
    nvtx.push_range('load_model')
    sessions = [ort.InferenceSession(model_path, providers=providers) for _ in range(num_instances)]
    torch.cuda.synchronize()
    mem_after_load, _ = get_gpu_memory()
    model_gpu_memory = mem_after_load - mem_before_load
    print(f"📌 {num_instances} models loaded. Total GPU memory used: {model_gpu_memory / 1024**2:.2f} MB")
    nvtx.pop_range()

    # **2️⃣ 解析 ONNX 输入 shape，并设置 batch_size**
    batch_size = batchsize  # **手动设置 batch_size**
    input_shapes = {}
    input_datas = {}
    input_data_size = 0

    nvtx.push_range('load_data_to_cpu')
    for input_tensor in sessions[0].get_inputs():
        name = input_tensor.name
        shape = list(input_tensor.shape)

        # **动态替换 batch 维度（如果是 None，则替换为 batch_size）**
        shape[0] = batch_size if shape[0] is None or type(shape[0]) != int or shape[0]<=0 else shape[0]
        input_shapes[name] = tuple(shape)
        d = np.random.randn(*shape).astype(np.float32)
        input_datas[name] = d
        input_data_size += d.nbytes

    print(f"📌 ONNX Input Shapes: {input_shapes}, Numpy Tensor Size: {input_data_size} B, {input_data_size / 1024**2} MB")
    nvtx.pop_range()

    # **3️⃣ 监测输入数据加载显存**
    nvtx.push_range('load_data_to_gpu')
    torch.cuda.synchronize()
    mem_before_input, _ = get_gpu_memory()

    inputs = {
        name: ort.OrtValue.ortvalue_from_numpy(data, device_type="cuda", device_id=device_id)
        for name, data in input_datas.items()
    }
    inputs_list = [inputs] * num_instances
    nvtx.pop_range()

    torch.cuda.synchronize()
    time.sleep(3)
    mem_after_input, _ = get_gpu_memory()
    input_gpu_memory = mem_after_input - mem_before_input
    print(f"📌 Input loaded. GPU memory used: {input_gpu_memory} B ({input_gpu_memory / 1024**2:.2f} MB)")

    # **4️⃣ 启动监测线程**
    stop_event = threading.Event()  # **用 Event 控制线程结束**
    max_mem_list = []  # **存储最大显存**
    
    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(stop_event, max_mem_list), daemon=True)
    monitor_thread.start()

    # **5️⃣ 运行 ONNX 推理**
    mem_before_infer, _ = get_gpu_memory()
    nvtx.push_range('infer')
    for _ in range(3):
        for session in sessions:
            output = session.run(None, inputs)
            del output  # ✅ 立即清除推理输出，防止占用累积
        time.sleep(1)

    torch.cuda.synchronize()
    nvtx.pop_range()

    # **6️⃣ 停止监测线程**
    stop_event.set()
    monitor_thread.join()

    # **7️⃣ 计算推理期间的最大显存**
    max_mem_usage = max(max_mem_list)
    peak_memory_during_inference = max_mem_usage - mem_before_infer
    print(f"📌 Peak GPU memory during inference: {peak_memory_during_inference} B ({peak_memory_during_inference / 1024**2:.2f} MB, total before load {(max_mem_usage - mem_before_load) / 1024**2:.2f} MB)")

    # **8️⃣ 释放 GPU 资源**
    print("🧹 Cleaning up GPU memory...")

    for session in sessions:
        session.set_providers([])
        del session
    del sessions

    for k in list(inputs.keys()):
        del inputs[k]
    del inputs  # ✅ 清理唯一一份共享 OrtValue

    gc.collect()
    torch.cuda.empty_cache()  # **清空 CUDA 缓存**
    torch.cuda.synchronize()  # **确保清理完成**

    # **9️⃣ 再次检查 GPU 显存**
    mem_after_cleanup, free_after_cleanup = get_gpu_memory()
    print(f"📌 GPU memory after cleanup: {mem_after_cleanup} B ({mem_after_cleanup / 1024**2:.2f} MB)")
    print(f"📌 GPU free memory after cleanup: {free_after_cleanup} B ({free_after_cleanup / 1024**2:.2f} MB)")

if __name__ == '__main__':
    model_path = ["/workspace/datas/models/resnet152-1000/resnet152.onnx"]
    # model_path = ["/workspace/co-mpi/test/models_seg/part0.onnx", "/workspace/co-mpi/test/models_seg/part1.onnx", "/workspace/datas/models/resnet152-1000/resnet152.onnx"]
    bs = [1, 2, 4, 8, 16]
    for p in model_path:
        for b in bs:
            for n in [1, 1, 16]:
                print(f"📌 Running {n} instances of Model: {p}, batchsize={b}")
                get_onnx_running_data(p, num_instances=n, batchsize=b)
                print("-" * 60)
                time.sleep(3)

