"""
ALU (Arithmetic Logic Unit) — 算术逻辑单元
===========================================
对标 GPGPU-Sim 中 simd_function_unit 的 SP (Single Precision) 管线。

执行 32-bit 有符号整数运算: ADD, SUB, MUL, DIV。

参考: GPGPU-Sim gpgpu-sim/shader.cc 中 simd_function_unit::cycle()
      的 SP 管线实现
"""


class ALU:
    """32-bit 算术逻辑单元

    GPGPU-Sim 有 SP/SFU/DP/INT 四种执行单元。
    Phase 0 只实现 INT（整数运算）部分。
    """

    @staticmethod
    def add(a: int, b: int) -> int:
        """有符号 32-bit 加法

        结果截断到 32-bit，模拟硬件溢出回绕。
        """
        return (a + b) & 0xFFFFFFFF

    @staticmethod
    def sub(a: int, b: int) -> int:
        """有符号 32-bit 减法"""
        return (a - b) & 0xFFFFFFFF

    @staticmethod
    def mul(a: int, b: int) -> int:
        """有符号 32-bit 乘法

        取乘积的低 32-bit（模拟硬件乘法器输出截断）。
        """
        return (a * b) & 0xFFFFFFFF

    @staticmethod
    def div(a: int, b: int) -> int:
        """有符号 32-bit 除法

        除零返回 0（模仿某些 GPU 的除零行为）。
        使用 Python 整数除法的向零舍入语义。
        """
        if b == 0:
            return 0  # 除零保护
        # 转为有符号运算
        a_signed = ALU._s32(a)
        b_signed = ALU._s32(b)
        result = int(a_signed / b_signed)  # Python 向零舍入
        return result & 0xFFFFFFFF

    @staticmethod
    def _s32(val: int) -> int:
        """将 32-bit 无符号值转为有符号 Python 整数"""
        if val & 0x80000000:
            return val - 0x100000000
        return val
