import numpy as np
import io
import ast

def parse_npy_header(byte_stream):
    """
    解析 `.npy` 文件的头部信息（shape 和 dtype）。
    - byte_stream: `io.BytesIO` 或 文件对象
    返回:
    - shape: tuple, 解析出的形状
    - dtype: numpy.dtype, 解析出的数据类型
    - header_size: int, 头部大小（字节数）
    """
    byte_stream.seek(0)  # **重置到文件开头**
    magic = byte_stream.read(6)  # 读取 magic number
    if not magic.startswith(b"\x93NUMPY"):
        raise ValueError("Invalid .npy file format")
    
    byte_stream.read(2)  # 跳过版本号
    header_len = int.from_bytes(byte_stream.read(2), "little")  # 读取头部大小
    header = byte_stream.read(header_len).decode("utf-8").strip()  # 读取头部

    # 解析 shape
    shape_start = header.index("'shape':") + len("'shape':")
    shape_end = header.index("}", shape_start)
    shape_str = header[shape_start:shape_end].strip(", ")
    shape = ast.literal_eval(shape_str)

    # 解析 dtype
    dtype_start = header.index("'descr':") + len("'descr':")
    dtype_end = header.index(",", dtype_start)
    dtype_str = header[dtype_start:dtype_end].strip(" '")
    dtype = np.dtype(dtype_str)

    return shape, dtype, 6 + 2 + 2 + header_len  # 返回 shape, dtype 和 头部大小

if __name__ == '__main__':
    # **创建 .npy 测试数据**
    array = np.random.randn(1, 3, 224, 224).astype(np.float32)
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)

    # **解析头部**
    shape, dtype, header_size = parse_npy_header(buffer)
    print(f"Shape: {shape}, Dtype: {dtype}, Header Size: {header_size} bytes")

    # **创建空数组并使用 `readinto` 读取数据**
    buffer.seek(header_size)  # **跳过头部**
    loaded_array = np.empty(shape, dtype=dtype)
    buffer.readinto(loaded_array)

    # **比较差异**
    difference = np.abs(array - loaded_array).max()
    print(f"Max Difference: {difference}")  # **应接近 0**
    print(f"Arrays equal: {np.allclose(array, loaded_array)}")  # **应为 True**
