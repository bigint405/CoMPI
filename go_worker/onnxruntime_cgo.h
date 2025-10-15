#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // 表示 ONNX 推理会话
    typedef struct ORTSession ORTSession;

    // 表示 GPU 或 CPU 上的张量
    typedef struct ORTTensor ORTTensor;

    size_t ORT_GetOutputCount(ORTSession *session);

    int ORT_GetTensorShape(ORTTensor *tensor, int64_t **out_shape, size_t *out_len);

    /**
     * @brief 加载 ONNX 模型并创建推理会话
     * @param model_path 模型路径（.onnx 文件）
     * @param gpu_id 使用的 GPU 设备编号
     * @return 成功返回 ORTSession*，失败返回 nullptr
     */
    ORTSession *ORT_LoadModel(const char *model_path, int gpu_id);

    /**
     * @brief 创建 GPU 上的输入张量
     * @param gpu_id 使用的 GPU 设备编号
     * @param data CPU 上的 float32 原始数据指针
     * @param data_len 数据元素数量（float 个数）
     * @param shape 张量形状数组（如 [1, 3, 224, 224]）
     * @param shape_len 形状维度个数（如 4）
     * @return 成功返回 ORTTensor*，失败返回 nullptr
     */
    ORTTensor *ORT_CreateTensor(int gpu_id, const float *data, size_t data_len,
                                const int64_t *shape, size_t shape_len);

    /**
     * @brief 创建一个 Batch 的 ORT GPU Tensor（将多个输入拼接为一个 batch 维度）
     *
     * @param gpu_id       GPU 设备编号（从 0 开始）
     * @param data_ptrs    一个包含 batch_size 个 float* 指针的数组，每个指针指向一个样本的 float32 数据（在 CPU 上）
     * @param batch_size   样本数量（即 batch 大小）
     * @param single_len   样本的元素数（即 shape 的乘积，例如 3x224x224=150528）
     * @param shape        样本的形状（含 batch 维度，例如 {16, 3, 224, 224}）
     * @param shape_len    shape 的长度（例如 4）
     *
     * @return ORTTensor*  成功返回 GPU 上的批量 ORT Tensor，失败返回 nullptr
     *
     * @note 输出的 Tensor 维度为 [batch_size, ...shape]，在 GPU 上。
     *       内部会分配并管理 GPU 显存，返回值需要通过适当接口释放。
     */
    ORTTensor *ORT_CreateTensorBatch(
        int gpu_id,
        const float **data_ptrs,
        int batch_size,
        size_t single_len,
        const int64_t *shape,
        size_t shape_len);

    /**
     * @brief 执行推理，自动构建输出张量（输出在 GPU 上）
     * @param session 已创建的 ORTSession
     * @param inputs 输入张量数组（长度为 input_count）
     * @param input_count 输入张量数量
     * @param batch_size 用于推理时自动设置输出 shape（如 [batch_size, 1000]）
     * @return 成功返回 ORTTensor**，每个输出为 GPU 上的张量；失败返回 nullptr
     */
    ORTTensor **ORT_RunInferenceFixedOutput(ORTSession *session,
                                            ORTTensor *const *inputs,
                                            size_t input_count,
                                            int64_t batch_size);

    /**
     * @brief 将 GPU 上的 ORTTensor 拷贝到 CPU，并返回 float32 数据
     * @param tensor 推理得到的 ORTTensor
     * @param out_shape 返回 shape 数组指针（需 free）
     * @param out_dim_count 返回 shape 的维度个数
     * @return 成功返回 float*（需 free），失败返回 nullptr
     */
    float *ORT_GetTensorData(ORTTensor *tensor, int64_t **out_shape, size_t *out_dim_count);

    /**
     * @brief 释放张量及其 GPU 内存（如果持有）
     * @param tensor 需释放的 ORTTensor
     */
    void ORT_ReleaseTensor(ORTTensor *tensor);

    /**
     * @brief 释放推理会话
     * @param session 已创建的 ORTSession
     */
    void ORT_ReleaseSession(ORTSession *session);

    /**
     * @brief 对多个 float32 向量进行 int8 线性量化（FBGEMM），支持每个向量不同 shape。
     * 编码格式：
     *   [int32 dim][int32 shape_len][int64 shape[shape_len]][float scale][int32 zero_point][int8 data[dim]]
     *   重复 num_vecs 次
     *
     * @param vectors      每个向量的 float* 指针数组（共 num_vecs 个）
     * @param dims         每个向量的长度（元素数），长度为 num_vecs
     * @param shapes       每个向量的 shape 指针数组（共 num_vecs 个），每个 shape 是 int64_t*
     * @param shape_lens   每个向量的 shape 维度数，长度为 num_vecs
     * @param num_vecs     向量数量
     * @param out_data     输出量化字节数据（需 free）
     * @param out_len      输出数据长度（字节数）
     *
     * @return 0 表示成功，非 0 表示失败
     */
    int QuantizeVectorsFBGEMM(
        const float **vectors,
        const int32_t *dims,
        const int64_t **shapes,
        const int32_t *shape_lens,
        int32_t num_vecs,
        uint8_t **out_data,
        size_t *out_len);

    /**
     * @brief 解量化多个 FBGEMM 编码向量，每个向量包含 dim + shape + scale + zp + int8 数据。
     *
     * 编码格式：
     *   [int32_t dim][int32_t shape_len][int64_t shape[shape_len]][float scale][int32_t zero_point][int8 data[dim]]
     *
     * @param data           输入字节数据
     * @param len            数据长度（单位：字节）
     * @param out_vectors    输出：每个向量的 float* 指针数组（需逐个 free）
     * @param out_dims       输出：每个向量的长度（float 数）
     * @param out_shapes     输出：每个向量的 shape 数组指针（int64_t*）（需逐个 free）
     * @param out_shape_lens 输出：每个向量的 shape 长度
     * @param out_num_vecs   输出：总向量数
     *
     * @return 0 表示成功，非 0 表示失败
     */
    int DequantizeVectorsFBGEMM(
        const uint8_t *data,
        size_t len,
        float ***out_vectors,
        int32_t **out_dims,
        int64_t ***out_shapes,
        int32_t **out_shape_lens,
        int *out_num_vecs);

#ifdef __cplusplus
}
#endif
