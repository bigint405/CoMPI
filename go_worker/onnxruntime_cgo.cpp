#include "onnxruntime_cgo.h"
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <fbgemm/QuantUtils.h>
#include <map>
#include <chrono>
struct ORTTensor
{
    Ort::Value value{nullptr};
    bool is_gpu = false;
    bool owns_data = false;
    void *device_ptr = nullptr;
    int gpu_id = 0;
};

struct GraphSlotKey {
    int64_t batch_size = 0;   // logical batch size used in the graph (e.g., maxBS)
    int     ping_index = 0;   // 0 or 1, for ping–pong

    bool operator<(const GraphSlotKey &other) const {
        if (batch_size != other.batch_size) {
            return batch_size < other.batch_size;
        }
        return ping_index < other.ping_index;
    }
};

struct GraphSlot {
    // 持久化的输入 / 输出张量，指向固定的 GPU 内存
    std::vector<ORTTensor*> inputs;   // 通常就一个输入
    std::vector<ORTTensor*> outputs;  // 所有输出

    // 标记输出是否已经根据当前 batch_size 初始化过
    bool outputs_initialized = false;

    // 为这个 slot 生成的 gpu_graph_id，用于 ORT 内部缓存 graph
    std::string gpu_graph_id;
};

struct ORTSession
{
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::IoBinding io_binding{nullptr};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    size_t output_count = 0;
    int gpu_id = 0;

    // global switch for this session to use CUDA graph–aware input management (ping–pong slots etc.).
    bool use_cuda_graph = false;
    // per-session graph slots, keyed by (batch_size, ping_index). At this stage we only manage input tensors inside each slot.
    std::map<GraphSlotKey, GraphSlot> graph_slots;
};

extern "C"
{
    size_t ORT_GetOutputCount(ORTSession *session)
    {
        if (!session)
            return 0;
        return session->output_count;
    }

    // Enable or disable CUDA graph integration for a given session.
    void ORT_SetCudaGraphEnabled(ORTSession *session, int enabled)
    {
        if (!session) return;
        session->use_cuda_graph = (enabled != 0);
    }

    int ORT_GetTensorShape(ORTTensor *tensor, int64_t **out_shape, size_t *out_len)
    {
        if (!tensor || !tensor->value.IsTensor())
            return 1;

        try
        {
            Ort::TensorTypeAndShapeInfo info = tensor->value.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();

            *out_len = shape.size();
            *out_shape = (int64_t *)malloc(sizeof(int64_t) * shape.size());
            memcpy(*out_shape, shape.data(), sizeof(int64_t) * shape.size());
            return 0;
        }
        catch (...)
        {
            return 2;
        }
    }

    ORTSession* ORT_LoadModel(const char* model_path, int gpu_id, int enable_cuda_graph) {
        auto ort_session = new ORTSession();
        try {
            ort_session->gpu_id = gpu_id;
            ort_session->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ort");
    
            Ort::SessionOptions session_options;
    
            // 创建 CUDA V2 options 并开启 enable_cuda_graph
            const auto& api = Ort::GetApi();
            OrtCUDAProviderOptionsV2* cuda_options = nullptr;
            Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
    
            std::unique_ptr<OrtCUDAProviderOptionsV2,
                            decltype(api.ReleaseCUDAProviderOptions)>
                cuda_guard(cuda_options, api.ReleaseCUDAProviderOptions);
    
            std::string gpu_id_str = std::to_string(gpu_id);
            int numKey;
            const char* keys[]   = {"device_id", "enable_cuda_graph"};
            const char* values[] = {gpu_id_str.c_str(), "1"};
            if (enable_cuda_graph)
            {
                numKey = 2;
            }
            else
            {
                numKey = 1;
            }
            Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
                cuda_guard.get(), keys, values, numKey));
    
            Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                static_cast<OrtSessionOptions*>(session_options),
                cuda_guard.get()));
    
            session_options.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
    
            // 创建 Session 和 IoBinding
            ort_session->session = Ort::Session(ort_session->env, model_path, session_options);
            ort_session->io_binding = Ort::IoBinding(ort_session->session);
    
            // 这里一定要保留: 填充 input_names 和 output_names
            size_t input_count  = ort_session->session.GetInputCount();
            size_t output_count = ort_session->session.GetOutputCount();
            ort_session->output_count = output_count;
    
            Ort::AllocatorWithDefaultOptions allocator;
            for (size_t i = 0; i < input_count; ++i) {
                auto name = ort_session->session.GetInputNameAllocated(i, allocator);
                ort_session->input_names.push_back(name.get());
            }
            for (size_t i = 0; i < output_count; ++i) {
                auto name = ort_session->session.GetOutputNameAllocated(i, allocator);
                ort_session->output_names.push_back(name.get());
            }
    
            return ort_session;
        } catch (const Ort::Exception& e) {
            std::cerr << "[ORT ERROR][ORT_LoadModel] " << e.what() << std::endl;
            ORT_ReleaseSession(ort_session);
            return nullptr;
        }
    }
    

    ORTTensor *ORT_CreateTensor(int gpu_id, const float *data, size_t data_len, const int64_t *shape, size_t shape_len)
    {
        try
        {
            cudaError err = cudaSetDevice(gpu_id);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR][ORT_CreateTensor] cudaSetDevice fail [" << err << "] of gpu " << gpu_id << std::endl;
                return nullptr;
            }

            size_t total_len = 1;
            for (size_t i = 0; i < shape_len; ++i)
                total_len *= static_cast<size_t>(shape[i]);
            if (total_len != data_len)
                return nullptr;

            size_t total_bytes = total_len * sizeof(float);
            void *device_ptr = nullptr;
            if (cudaMalloc(&device_ptr, total_bytes) != cudaSuccess)
                return nullptr;
            if (cudaMemcpy(device_ptr, data, total_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            {
                cudaFree(device_ptr);
                return nullptr;
            }

            Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, gpu_id, OrtMemTypeDefault);
            Ort::Value tensor = Ort::Value::CreateTensor(memory_info, device_ptr, total_bytes,
                                                         shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

            auto *t = new ORTTensor();
            t->value = std::move(tensor);
            t->is_gpu = true;
            t->owns_data = true;
            t->device_ptr = device_ptr;
            t->gpu_id = gpu_id;
            return t;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "[ORT ERROR] " << e.what() << std::endl;
            return nullptr;
        }
    }

    ORTTensor *ORT_CreateTensorBatch(
        int gpu_id,
        const float **data_ptrs, // 指向 batch_size 个 float 数组的指针
        int batch_size,
        size_t single_len,    // 样本的数据长度（元素数）
        const int64_t *shape, // 样本的 shape（含 batch 维度）
        size_t shape_len)
    {
        try
        {
            cudaError err = cudaSetDevice(gpu_id);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR][ORT_CreateTensorBatch] cudaSetDevice fail [" << err << "] of gpu " << gpu_id << std::endl;
                return nullptr;
            }

            size_t total_len = batch_size * single_len;
            size_t total_bytes = total_len * sizeof(float);

            // 分配 GPU 显存
            void *device_ptr = nullptr;
            err = cudaMalloc(&device_ptr, total_bytes);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR][ORT_CreateTensorBatch] cudaMalloc fail [" << err << "] of gpu " << gpu_id << " with bytes " << total_bytes << std::endl;
                return nullptr;
            }

            // 每个样本逐个拷贝到 GPU 连续内存中
            for (int i = 0; i < batch_size; ++i)
            {
                const float *src = data_ptrs[i];
                void *dst = static_cast<float *>(device_ptr) + i * single_len;
                if (src == nullptr) {
                    // 填充位：不拷贝也不清零，保持未初始化即可
                    continue;
                }
                err = cudaMemcpy(dst, src, single_len * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    std::cerr << "[CUDA ERROR][ORT_CreateTensorBatch] cudaMemcpy fail [" << err << "] of gpu " << gpu_id << " of data  " << i << std::endl;
                    cudaFree(device_ptr);
                    return nullptr;
                }
            }
            std::cout << "tensor shape: len: " << shape_len << ", shape: ";
            for (int i = 0; i < shape_len; i++)
            {
                std::cout << shape[i] << ", ";
            }
            std::cout << std::endl;
            Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, gpu_id, OrtMemTypeDefault);
            Ort::Value tensor = Ort::Value::CreateTensor(
                memory_info, device_ptr, total_bytes,
                shape, shape_len,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

            ORTTensor *t = new ORTTensor();
            t->value = std::move(tensor);
            t->is_gpu = true;
            t->owns_data = true;
            t->device_ptr = device_ptr;
            t->gpu_id = gpu_id;
            return t;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "[ORT ERROR][ORT_CreateTensorBatch] " << e.what() << std::endl;
            return nullptr;
        }
    }

    ORTTensor **ORT_RunInferenceFixedOutput(ORTSession *session, ORTTensor *const *inputs, size_t input_count, int64_t batch_size)
    {
        try
        {
            if (!session || !inputs)
                return nullptr;

            cudaError err = cudaSetDevice(session->gpu_id);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR][ORT_CreateTensorBatch] cudaSetDevice fail [" << err << "] of gpu " << session->gpu_id << std::endl;
                return nullptr;
            }

            session->io_binding.ClearBoundInputs(); // todo 清空输入和输出不应该现在做，在把输出导出到cpu后就得清空输出了
            session->io_binding.ClearBoundOutputs();

            // 绑定输入张量
            for (size_t i = 0; i < input_count; ++i)
            {

                Ort::TensorTypeAndShapeInfo tensor_info = inputs[i]->value.GetTensorTypeAndShapeInfo();

                std::vector<int64_t> input_shape = tensor_info.GetShape();

                std::cout << "Input " << i << " shape: ";
                for (size_t j = 0; j < input_shape.size(); ++j)
                {
                    std::cout << input_shape[j] << " ";
                }
                std::cout << std::endl;

                session->io_binding.BindInput(session->input_names[i].c_str(), inputs[i]->value);
            }

            // 准备输出张量指针数组
            ORTTensor **result = (ORTTensor **)malloc(session->output_count * sizeof(ORTTensor *));

            // 遍历所有的输出张量
            for (size_t i = 0; i < session->output_count; ++i)
            {
                // 获取输出张量的类型信息
                Ort::TypeInfo output_type_info = session->session.GetOutputTypeInfo(i);

                // 从 TypeInfo 获取 TensorTypeAndShapeInfo
                Ort::ConstTensorTypeAndShapeInfo tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

                // 获取张量的形状
                std::vector<int64_t> output_shape = tensor_info.GetShape();

                // 处理 batch_size 为 -1 的情况
                if (output_shape[0] == -1)
                {
                    output_shape[0] = batch_size; // 替换 batch_size 为实际的值
                }
                // 动态计算输出张量总大小
                size_t total_len = 1;
                for (int64_t dim : output_shape)
                {
                    total_len *= dim;
                }

                size_t total_bytes = total_len * sizeof(float);

                void *device_ptr = nullptr;
                cudaError err = cudaMalloc(&device_ptr, total_bytes);
                if (err != cudaSuccess)
                {
                    std::cerr << "[CUDA ERROR][ORT_RunInferenceFixedOutput] cudaMalloc fail [" << err << "] of gpu " << session->gpu_id << " of data  " << i << std::endl;
                    return nullptr;
                }
                Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, session->gpu_id, OrtMemTypeDefault);

                // 创建输出张量
                Ort::Value output_tensor = Ort::Value::CreateTensor(memory_info, device_ptr, total_bytes,
                                                                    output_shape.data(), output_shape.size(),
                                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

                // 绑定输出张量
                session->io_binding.BindOutput(session->output_names[i].c_str(), output_tensor);

                auto *tensor = new ORTTensor();
                tensor->value = std::move(output_tensor);
                tensor->is_gpu = true;
                tensor->owns_data = true;
                tensor->device_ptr = device_ptr;

                result[i] = tensor;
            }

            // 执行推理
            session->session.Run(Ort::RunOptions{nullptr}, session->io_binding);
            return result;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "[ORT ERROR][ORT_RunInferenceFixedOutput] " << e.what() << std::endl;
            return nullptr;
        }
    }

    float *ORT_GetTensorData(ORTTensor *tensor, int64_t **out_shape, size_t *out_dim_count)
    {
        try
        {
            if (!tensor || !tensor->value)
                return nullptr;

            cudaError err = cudaSetDevice(tensor->gpu_id);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR][ORT_GetTensorData] cudaSetDevice fail [" << err << "] of gpu " << tensor->gpu_id << std::endl;
                return nullptr;
            }

            Ort::TensorTypeAndShapeInfo info = tensor->value.GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            size_t total_len = 1;
            for (size_t i = 0; i < shape.size(); ++i)
                total_len *= shape[i];

            float *host_data = (float *)malloc(total_len * sizeof(float));
            void *device_ptr = tensor->value.GetTensorMutableData<void>();
            err = cudaMemcpy(host_data, device_ptr, total_len * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR][ORT_GetTensorData] cudaMemcpy fail [" << err << "] of gpu " << tensor->gpu_id << std::endl;
                free(host_data);
                return nullptr;
            }

            *out_dim_count = shape.size();
            *out_shape = (int64_t *)malloc(shape.size() * sizeof(int64_t));
            memcpy(*out_shape, shape.data(), shape.size() * sizeof(int64_t));
            return host_data;
        }
        catch (const Ort::Exception &e)
        {
            std::cerr << "[ORT ERROR] " << e.what() << std::endl;
            return nullptr;
        }
    }

    void ORT_ReleaseTensor(ORTTensor *tensor)
    {
        if (!tensor)
            return;
        if (tensor->is_gpu && tensor->owns_data && tensor->device_ptr)
        {
            cudaFree(tensor->device_ptr);
        }
        delete tensor;
    }

    void ORT_ReleaseSession(ORTSession *session)
    {
        if (!session)
            return;

        // 释放所有 graph slot 里的 input / output GPU 内存
        for (auto &kv : session->graph_slots) {
            GraphSlot &slot = kv.second;

            auto free_tensor = [](ORTTensor *t) {
                if (!t) return;
                if (t->is_gpu && t->owns_data && t->device_ptr) {
                    cudaError_t err = cudaSetDevice(t->gpu_id);
                    if (err == cudaSuccess) {
                        cudaFree(t->device_ptr);
                    }
                }
                delete t;
            };

            for (ORTTensor *t : slot.inputs) {
                free_tensor(t);
            }
            for (ORTTensor *t : slot.outputs) {
                free_tensor(t);
            }
        }

        delete session;
    }


    int QuantizeVectorsFBGEMM(
        const float **vectors,
        const int32_t *dims,
        const int64_t **shapes,
        const int32_t *shape_lens,
        int32_t num_vecs,
        uint8_t **out_data,
        size_t *out_len)
    {
        if (!vectors || !dims || !shapes || !shape_lens || num_vecs <= 0 || !out_data || !out_len)
            return 1;

        // 预估总空间
        size_t total_bytes = 0;
        for (int i = 0; i < num_vecs; ++i)
        {
            total_bytes += sizeof(int32_t)                   // dim
                           + sizeof(int32_t)                 // shape_len
                           + sizeof(int64_t) * shape_lens[i] // shape[i]
                           + sizeof(float)                   // scale
                           + sizeof(int32_t)                 // zero_point
                           + dims[i];                        // quantized data
        }

        *out_data = (uint8_t *)malloc(total_bytes);
        if (!*out_data)
            return 2;

        uint8_t *cursor = *out_data;

        for (int i = 0; i < num_vecs; ++i)
        {
            int dim = dims[i];
            const float *vec = vectors[i];
            const int64_t *shape_ptr = shapes[i];
            int32_t shape_len = shape_lens[i];

            if (!vec || !shape_ptr || shape_len <= 0)
                return 3;

            float min_val = vec[0];
            float max_val = vec[0];
            for (int j = 1; j < dim; ++j)
            {
                if (vec[j] < min_val)
                    min_val = vec[j];
                if (vec[j] > max_val)
                    max_val = vec[j];
            }

            auto qparams = fbgemm::ChooseQuantizationParams(min_val, max_val, -128, 127);

            memcpy(cursor, &dim, sizeof(int32_t));
            cursor += sizeof(int32_t);
            memcpy(cursor, &shape_len, sizeof(int32_t));
            cursor += sizeof(int32_t);
            memcpy(cursor, shape_ptr, sizeof(int64_t) * shape_len);
            cursor += sizeof(int64_t) * shape_len;
            memcpy(cursor, &qparams.scale, sizeof(float));
            cursor += sizeof(float);
            memcpy(cursor, &qparams.zero_point, sizeof(int32_t));
            cursor += sizeof(int32_t);

            fbgemm::Quantize<int8_t>(vec, reinterpret_cast<int8_t *>(cursor), dim, qparams);
            cursor += dim;
        }

        *out_len = cursor - *out_data;
        return 0;
    }

    int DequantizeVectorsFBGEMM(
        const uint8_t *data,
        size_t len,
        float ***out_vectors,
        int32_t **out_dims,
        int64_t ***out_shapes,
        int32_t **out_shape_lens,
        int *out_num_vecs)
    {
        if (!data || len == 0 || !out_vectors || !out_dims || !out_shapes || !out_shape_lens || !out_num_vecs)
            return 1;

        const uint8_t *cursor = data;
        const uint8_t *end = data + len;

        std::vector<float *> vecs;
        std::vector<int32_t> dims;
        std::vector<int64_t *> shapes;
        std::vector<int32_t> shape_lens;

        while (cursor < end)
        {
            if ((end - cursor) < (int)(sizeof(int32_t) * 2))
                return 2;

            int32_t dim, shape_len;
            memcpy(&dim, cursor, sizeof(int32_t));
            cursor += sizeof(int32_t);
            memcpy(&shape_len, cursor, sizeof(int32_t));
            cursor += sizeof(int32_t);

            if ((end - cursor) < (int)(sizeof(int64_t) * shape_len + sizeof(float) + sizeof(int32_t) + dim))
                return 3;

            int64_t *shape_ptr = (int64_t *)malloc(sizeof(int64_t) * shape_len);
            if (!shape_ptr)
                return 4;
            memcpy(shape_ptr, cursor, sizeof(int64_t) * shape_len);
            cursor += sizeof(int64_t) * shape_len;

            float scale;
            int32_t zp;
            memcpy(&scale, cursor, sizeof(float));
            cursor += sizeof(float);
            memcpy(&zp, cursor, sizeof(int32_t));
            cursor += sizeof(int32_t);

            const int8_t *qdata = reinterpret_cast<const int8_t *>(cursor);
            cursor += dim;

            float *vec_ptr = (float *)malloc(sizeof(float) * dim);
            if (!vec_ptr)
                return 5;

            fbgemm::TensorQuantizationParams qparams;
            qparams.scale = scale;
            qparams.zero_point = zp;
            qparams.precision = 8;

            fbgemm::Dequantize<int8_t>(qdata, vec_ptr, dim, qparams);

            vecs.push_back(vec_ptr);
            dims.push_back(dim);
            shapes.push_back(shape_ptr);
            shape_lens.push_back(shape_len);
        }

        int n = vecs.size();
        *out_num_vecs = n;

        *out_vectors = (float **)malloc(sizeof(float *) * n);
        *out_dims = (int32_t *)malloc(sizeof(int32_t) * n);
        *out_shapes = (int64_t **)malloc(sizeof(int64_t *) * n);
        *out_shape_lens = (int32_t *)malloc(sizeof(int32_t) * n);

        if (!*out_vectors || !*out_dims || !*out_shapes || !*out_shape_lens)
            return 6;

        for (int i = 0; i < n; ++i)
        {
            (*out_vectors)[i] = vecs[i];
            (*out_dims)[i] = dims[i];
            (*out_shapes)[i] = shapes[i];
            (*out_shape_lens)[i] = shape_lens[i];
        }

        return 0;
    }

    // Internal helper: find or create a GraphSlot for (batch_size, ping_index).
    static GraphSlot &GetOrCreateGraphSlot(ORTSession *session, int64_t batch_size, int ping_index)
    {
        GraphSlotKey key;
        key.batch_size = batch_size;
        key.ping_index = ping_index;

        auto it = session->graph_slots.find(key);
        if (it != session->graph_slots.end()) {
            return it->second;
        }
        std::cout << "[GetOrCreateGraphSlot] creating graph slot, bs: " << batch_size << ", ping: " << ping_index << std::endl;

        GraphSlot slot;
        auto res = session->graph_slots.emplace(key, std::move(slot));
        return res.first->second;
    }

    // 为给定 session 的 (batch_size_graph, ping_index, input_index) 创建 / 更新
    // graph 管理的输入张量。
    //
    // 参数:
    //   session        – ORT session
    //   ping_index     – 0 或 1 (ping–pong)
    //   input_index    – 第几个输入 (0 .. input_names.size()-1)
    //   data_ptrs      – 长度为 batch_size 的 host 指针数组; 每个指向一条样本
    //   batch_size     – 本次实际有效样本数
    //   single_len     – 每条样本的 float 元素数
    //   shape          – 该输入在 ONNX 中的完整 shape (shape[0] = graph batch)
    //   shape_len      – rank
    //
    // 返回:
    //   指向该 slot 的第 input_index 个输入 ORTTensor 的指针; GPU 内存持久化.
    ORTTensor *ORT_CreateOrUpdateGraphInput(
        ORTSession *session,
        int ping_index,
        int input_index,
        const float **data_ptrs,
        int batch_size,
        size_t single_len,
        const int64_t *shape,
        size_t shape_len)
    {
        if (!session || !data_ptrs || !shape || shape_len == 0 || batch_size <= 0) {
            return nullptr;
        }
        if (!session->use_cuda_graph) {
            // 只在开启 CUDA graph 时允许调用
            return nullptr;
        }

        cudaError_t err = cudaSetDevice(session->gpu_id);
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR][ORT_CreateOrUpdateGraphInput] cudaSetDevice fail ["
                    << err << "] of gpu " << session->gpu_id << std::endl;
            return nullptr;
        }

        const int64_t graph_batch_size = shape[0];
        if (graph_batch_size <= 0) {
            std::cerr << "[ORT][ORT_CreateOrUpdateGraphInput] invalid graph_batch_size "
                    << graph_batch_size << std::endl;
            return nullptr;
        }

        if (input_index < 0 ||
            static_cast<size_t>(input_index) >= session->input_names.size()) {
            std::cerr << "[ORT][ORT_CreateOrUpdateGraphInput] invalid input_index "
                    << input_index << std::endl;
            return nullptr;
        }

        GraphSlot &slot = GetOrCreateGraphSlot(session, graph_batch_size, ping_index);

        // 确保 inputs 向量长度足够
        if (slot.inputs.size() <= static_cast<size_t>(input_index)) {
            slot.inputs.resize(static_cast<size_t>(input_index) + 1, nullptr);
        }

        ORTTensor *tensor = slot.inputs[input_index];

        if (!tensor) {
            // 第一次为该 input_index 分配 GPU 内存并封装为 ORTTensor
            size_t total_len = static_cast<size_t>(graph_batch_size) * single_len;
            size_t total_bytes = total_len * sizeof(float);

            void *device_ptr = nullptr;
            if (cudaMalloc(&device_ptr, total_bytes) != cudaSuccess) {
                std::cerr << "[CUDA ERROR][ORT_CreateOrUpdateGraphInput] cudaMalloc fail of gpu "
                        << session->gpu_id << " with bytes " << total_bytes << std::endl;
                return nullptr;
            }

            Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator,
                                        session->gpu_id, OrtMemTypeDefault);
            Ort::Value value = Ort::Value::CreateTensor(
                memory_info,
                device_ptr,
                total_bytes,
                shape,
                shape_len,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

            tensor = new ORTTensor();
            tensor->value = std::move(value);
            tensor->is_gpu = true;
            tensor->owns_data = true;
            tensor->device_ptr = device_ptr;
            tensor->gpu_id = session->gpu_id;

            slot.inputs[input_index] = tensor;
        }

        // 将本次 batch 的 host 数据写入到固定 GPU buffer
        float *base = static_cast<float *>(tensor->device_ptr);
        for (int i = 0; i < batch_size; ++i) {
            const float *src = data_ptrs[i];
            if (!src) {
                // 允许某些样本为空 (fixedMaxBatch 填充 nil)
                continue;
            }
            float *dst = base + static_cast<size_t>(i) * single_len;
            if (cudaMemcpy(dst, src, single_len * sizeof(float),
                        cudaMemcpyHostToDevice) != cudaSuccess) {
                std::cerr << "[CUDA ERROR][ORT_CreateOrUpdateGraphInput] cudaMemcpy fail of gpu "
                        << session->gpu_id << " at sample " << i << std::endl;
                return nullptr;
            }
        }

        return tensor;
    }

    // 使用 graph slot 中固定的输入 / 输出内存执行推理。
    //   batch_size_graph – graph 所看到的 batch 维大小 (如 maxBS)
    //   ping_index       – 0 / 1 (ping–pong)
    //
    // 返回: 持久化输出 ORTTensor* 数组 (长度 = session->output_count)
    ORTTensor **ORT_RunInferenceWithGraphSlot(ORTSession *session, int64_t batch_size_graph, int ping_index)
    {
        try {
            if (!session) return nullptr;

            cudaError_t err = cudaSetDevice(session->gpu_id);
            if (err != cudaSuccess) {
                std::cerr << "[CUDA ERROR][ORT_RunInferenceWithGraphSlot] cudaSetDevice fail ["
                        << err << "] of gpu " << session->gpu_id << std::endl;
                return nullptr;
            }

            if (!session->use_cuda_graph) {
                std::cerr << "[ORT][ORT_RunInferenceWithGraphSlot] use_cuda_graph is false" << std::endl;
                return nullptr;
            }

            GraphSlot &slot = GetOrCreateGraphSlot(session, batch_size_graph, ping_index);

            if (slot.inputs.empty()) {
                std::cerr << "[ORT][ORT_RunInferenceWithGraphSlot] "
                        << "graph slot has no inputs, call ORT_CreateOrUpdateGraphInput first."
                        << std::endl;
                return nullptr;
            }

            const size_t input_count = session->input_names.size();
            // 确保每个模型输入都有对应的 ORTTensor
            for (size_t i = 0; i < input_count; ++i) {
                if (i >= slot.inputs.size() || slot.inputs[i] == nullptr) {
                    std::cerr << "[ORT][ORT_RunInferenceWithGraphSlot] "
                            << "missing input tensor for index " << i << std::endl;
                    return nullptr;
                }
            }

            // 如未初始化输出: 按 batch_size_graph 分配一次
            if (!slot.outputs_initialized) {
                slot.outputs.clear();
                slot.outputs.reserve(session->output_count);

                for (size_t i = 0; i < session->output_count; ++i) {
                    Ort::TypeInfo output_type_info = session->session.GetOutputTypeInfo(i);
                    Ort::ConstTensorTypeAndShapeInfo tensor_info =
                        output_type_info.GetTensorTypeAndShapeInfo();

                    std::vector<int64_t> output_shape = tensor_info.GetShape();
                    if (!output_shape.empty() && output_shape[0] == -1) {
                        output_shape[0] = batch_size_graph;
                    }

                    size_t total_len = 1;
                    for (int64_t dim : output_shape) {
                        total_len *= static_cast<size_t>(dim);
                    }
                    size_t total_bytes = total_len * sizeof(float);

                    void *device_ptr = nullptr;
                    cudaError_t e2 = cudaMalloc(&device_ptr, total_bytes);
                    if (e2 != cudaSuccess) {
                        std::cerr << "[CUDA ERROR][ORT_RunInferenceWithGraphSlot] cudaMalloc fail ["
                                << e2 << "] of gpu " << session->gpu_id
                                << " for output " << i << std::endl;
                        return nullptr;
                    }

                    Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator,
                                                session->gpu_id, OrtMemTypeDefault);
                    Ort::Value out_tensor = Ort::Value::CreateTensor(
                        memory_info,
                        device_ptr,
                        total_bytes,
                        output_shape.data(),
                        output_shape.size(),
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

                    ORTTensor *t = new ORTTensor();
                    t->value = std::move(out_tensor);
                    t->is_gpu = true;
                    t->owns_data = true;
                    t->device_ptr = device_ptr;
                    t->gpu_id = session->gpu_id;

                    slot.outputs.push_back(t);
                }

                int64_t graph_id = batch_size_graph * 2 + ping_index + 1; // 保证结果大于 0
                slot.gpu_graph_id = std::to_string(graph_id);

                slot.outputs_initialized = true;
            }

            // 绑定输入 / 输出
            session->io_binding.ClearBoundInputs();
            session->io_binding.ClearBoundOutputs();

            for (size_t i = 0; i < input_count; ++i) {
                session->io_binding.BindInput(
                    session->input_names[i].c_str(),
                    slot.inputs[i]->value);
            }

            for (size_t i = 0; i < slot.outputs.size(); ++i) {
                session->io_binding.BindOutput(
                    session->output_names[i].c_str(),
                    slot.outputs[i]->value);
            }

            // RunOptions: 挂上 gpu_graph_id，交给 ORT 内部做 CUDA graph 缓存
            Ort::RunOptions run_options;
            run_options.SetRunLogVerbosityLevel(0);
            run_options.AddConfigEntry("gpu_graph_id", slot.gpu_graph_id.c_str());

            session->session.Run(run_options, session->io_binding);
            return slot.outputs.data();
        }
        catch (const Ort::Exception &e) {
            std::cerr << "[ORT ERROR][ORT_RunInferenceWithGraphSlot] "
                    << e.what() << std::endl;
            return nullptr;
        }
    }
} // end extern "C"
