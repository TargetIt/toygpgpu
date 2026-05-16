"""
Mixed Precision — Multi-Format Tensor Operations (Phase 21)
=============================================================
对标 NVIDIA Hopper FP8 (E4M3/E5M2), 支持多种精度:
  - FP8 E4M3 (1 sign, 4 exp, 3 mantissa) — 推理
  - FP8 E5M2 (1 sign, 5 exp, 2 mantissa) — 训练
  - FP16 (1 sign, 5 exp, 10 mantissa)
  - INT8 signed/unsigned
  - FP32 accumulator (高精度累加)

FP8 是 LLM 推理的事实标准: H100 FP8 吞吐 = 2x INT8 吞吐。

参考: NVIDIA FP8 formats (docs.nvidia.com/deeplearning/transformer-engine)
      IEEE 754 half-precision
"""

import struct
from typing import Tuple


# Precision format identifiers
FMT_FP32 = 0
FMT_FP16 = 1
FMT_BF16 = 2
FMT_FP8_E4M3 = 3
FMT_FP8_E5M2 = 4
FMT_INT8 = 5
FMT_UINT8 = 6

FMT_NAMES = {
    0: "FP32", 1: "FP16", 2: "BF16",
    3: "FP8_E4M3", 4: "FP8_E5M2",
    5: "INT8", 6: "UINT8",
}


def float_to_fp16(value: float) -> int:
    """FP32 → FP16 (binary16, IEEE 754). Returns 16-bit pattern."""
    if value == 0:
        return 0
    # Use Python struct for accurate conversion
    f32 = struct.pack('f', float(value))
    i32 = struct.unpack('I', f32)[0]
    sign = (i32 >> 31) & 1
    exp = (i32 >> 23) & 0xFF
    mant = i32 & 0x7FFFFF

    if exp == 0:
        return sign << 15
    if exp == 0xFF:
        new_exp = 0x1F
        return (sign << 15) | (new_exp << 10) | 0x3FF

    new_exp = exp - 127 + 15
    if new_exp <= 0:
        return sign << 15
    if new_exp >= 0x1F:
        return (sign << 15) | (0x1F << 10)

    new_mant = mant >> 13
    return (sign << 15) | (new_exp << 10) | new_mant


def fp16_to_float(half: int) -> float:
    """FP16 → FP32."""
    sign = (half >> 15) & 1
    exp = (half >> 10) & 0x1F
    mant = half & 0x3FF

    if exp == 0:
        return 0.0 if mant == 0 else (-1)**sign * 2**(-14) * (mant / 1024.0)
    if exp == 0x1F:
        return float('inf') if mant == 0 else float('nan')

    return (-1)**sign * 2**(exp - 15) * (1 + mant / 1024.0)


def float_to_fp8_e4m3(value: float) -> int:
    """FP32 → FP8 E4M3 (1-4-3, range ~ ±448)."""
    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    v = abs(value)

    exp = 0
    while v >= 2.0:
        v /= 2.0
        exp += 1
    while v < 1.0 and exp > -7:
        v *= 2.0
        exp -= 1

    exp_biased = exp + 7  # bias = 7 for E4M3
    if exp_biased <= 0:
        return sign << 7
    if exp_biased >= 15:
        return (sign << 7) | (0x7 << 3) | 0x6  # max value (NaN/inf)

    mant = int((v - 1.0) * 8) & 0x7
    return (sign << 7) | (exp_biased << 3) | mant


def fp8_e4m3_to_float(fp8: int) -> float:
    """FP8 E4M3 → FP32."""
    sign = -1 if (fp8 >> 7) & 1 else 1
    exp = (fp8 >> 3) & 0xF
    mant = fp8 & 0x7

    if exp == 0:
        if mant == 0:
            return 0.0
        return sign * 2**(-6) * (mant / 8.0)
    if exp == 0xF:
        return float('nan')
    return sign * 2**(exp - 7) * (1 + mant / 8.0)


def float_to_fp8_e5m2(value: float) -> int:
    """FP32 → FP8 E5M2 (1-5-2, wider range ~ ±57344)."""
    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    v = abs(value)

    exp = 0
    while v >= 2.0:
        v /= 2.0
        exp += 1
    while v < 1.0 and exp > -15:
        v *= 2.0
        exp -= 1

    exp_biased = exp + 15  # bias = 15 for E5M2
    if exp_biased <= 0:
        return sign << 7
    if exp_biased >= 31:
        return (sign << 7) | (0x1F << 2) | 0x3

    mant = int((v - 1.0) * 4) & 0x3
    return (sign << 7) | (exp_biased << 2) | mant


def fp8_e5m2_to_float(fp8: int) -> float:
    """FP8 E5M2 → FP32."""
    sign = -1 if (fp8 >> 7) & 1 else 1
    exp = (fp8 >> 2) & 0x1F
    mant = fp8 & 0x3

    if exp == 0:
        if mant == 0:
            return 0.0
        return sign * 2**(-14) * (mant / 4.0)
    if exp == 0x1F:
        return float('inf') if mant == 0 else float('nan')
    return sign * 2**(exp - 15) * (1 + mant / 4.0)


def pack_fp8_pair(a: int, b: int) -> int:
    """将两个 FP8 值打包到 32-bit 寄存器中 (低位 a, 高位 b)。"""
    return (a & 0xFF) | ((b & 0xFF) << 16)


def unpack_fp8_pair(packed: int) -> Tuple[int, int]:
    """从 32-bit 寄存器中解包两个 FP8 值。"""
    a = packed & 0xFFFF
    b = (packed >> 16) & 0xFFFF
    a_signed = a if a < 0x8000 else a - 0x10000
    b_signed = b if b < 0x8000 else b - 0x10000
    return a_signed, b_signed


def convert(value: int, from_fmt: int, to_fmt: int) -> int:
    """精度转换 (模拟硬件转换指令 CNVT)。

    Args:
        value: 源格式下的整数值
        from_fmt: 源精度格式
        to_fmt: 目标精度格式

    Returns:
        目标格式下的整数值
    """
    # Convert to FP32 first
    if from_fmt == FMT_FP32:
        f32 = struct.unpack('f', struct.pack('I', value))[0]
    elif from_fmt == FMT_FP16:
        f32 = fp16_to_float(value & 0xFFFF)
    elif from_fmt == FMT_FP8_E4M3:
        f32 = fp8_e4m3_to_float(value & 0xFF)
    elif from_fmt == FMT_FP8_E5M2:
        f32 = fp8_e5m2_to_float(value & 0xFF)
    else:
        f32 = float(value)

    # Convert to target format
    if to_fmt == FMT_FP32:
        return struct.unpack('I', struct.pack('f', f32))[0]
    elif to_fmt == FMT_FP16:
        return float_to_fp16(f32)
    elif to_fmt == FMT_FP8_E4M3:
        return float_to_fp8_e4m3(f32)
    elif to_fmt == FMT_FP8_E5M2:
        return float_to_fp8_e5m2(f32)
    else:
        return int(f32)


def fp8_mma(a_packed: int, b_packed: int, c_val: int,
            fmt: int = FMT_FP8_E4M3) -> int:
    """FP8 张量核心 MMA (2x2 dot product with FP8 operands).

    对标 H100 FP8 MMA 指令。

    Args:
        a_packed: 两个 FP8 A 值打包
        b_packed: 两个 FP8 B 值打包
        c_val: FP32 累加器
        fmt: FP8 格式 (E4M3 or E5M2)

    Returns:
        FP32 结果
    """
    a0, a1 = unpack_fp8_pair(a_packed)
    b0, b1 = unpack_fp8_pair(b_packed)

    if fmt == FMT_FP8_E4M3:
        af0 = fp8_e4m3_to_float(a0 & 0xFF)
        af1 = fp8_e4m3_to_float(a1 & 0xFF)
        bf0 = fp8_e4m3_to_float(b0 & 0xFF)
        bf1 = fp8_e4m3_to_float(b1 & 0xFF)
    else:
        af0 = fp8_e5m2_to_float(a0 & 0xFF)
        af1 = fp8_e5m2_to_float(a1 & 0xFF)
        bf0 = fp8_e5m2_to_float(b0 & 0xFF)
        bf1 = fp8_e5m2_to_float(b1 & 0xFF)

    cf = struct.unpack('f', struct.pack('I', c_val & 0xFFFFFFFF))[0]
    result = af0 * bf0 + af1 * bf1 + cf
    return struct.unpack('I', struct.pack('f', result))[0]


class PrecisionStats:
    """精度统计 — 用于分析精度损失。"""

    def __init__(self):
        self.total_conversions = 0
        self.max_error = 0.0
        self.sum_error = 0.0

    def record(self, original: float, converted: float):
        """记录一次转换的精度损失。"""
        error = abs(original - converted)
        self.total_conversions += 1
        self.max_error = max(self.max_error, error)
        self.sum_error += error

    @property
    def avg_error(self) -> float:
        if self.total_conversions == 0:
            return 0.0
        return self.sum_error / self.total_conversions

    def report(self) -> str:
        return (f"PrecisionStats: conversions={self.total_conversions}, "
                f"avg_error={self.avg_error:.6f}, "
                f"max_error={self.max_error:.6f}")
