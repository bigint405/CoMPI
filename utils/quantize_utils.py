import torch


def linear_quantize(tensor):
    """线性 int8 量化 (Min-Max 归一化)"""
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / 255
    quantized = ((tensor - min_val) / scale).round().to(torch.uint8)
    return quantized, scale, min_val

def linear_dequantize(quantized, scale, min_val):
    """线性 int8 反量化"""
    dequantized = quantized.float() * scale + min_val
    return dequantized