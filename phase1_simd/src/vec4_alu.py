"""
Vec4ALU — Vec4 复合数据类型算术逻辑单元
=========================================
对标 GPU shader 中 float4/uchar4 类型的打包 SIMD 运算。
将 4 个 8-bit 值打包到一个 32-bit 寄存器中执行逐分量运算。

布局: [byte3:24-31] [byte2:16-23] [byte1:8-15] [byte0:0-7]
"""


class Vec4ALU:
    """4×8-bit SIMD within a Register (SWAR) ALU

    每个 32-bit 值视为 4 个独立的 8-bit lane:
      lane 0: bits [7:0]
      lane 1: bits [15:8]
      lane 2: bits [23:16]
      lane 3: bits [31:24]
    """

    @staticmethod
    def _byte(word: int, lane: int) -> int:
        """提取第 lane 个字节 (0-3)"""
        return (word >> (lane * 8)) & 0xFF

    @staticmethod
    def pack(a: int, b: int) -> int:
        """将 4 个 8-bit 值打包到 32-bit 字

        从 rs1 取 byte0, byte1, 从 rs2 取 byte2, byte3:
          byte0 = a[7:0]
          byte1 = a[15:8]
          byte2 = b[7:0]
          byte3 = b[15:8]

        Returns:
            32-bit 打包值
        """
        b0 = a & 0xFF
        b1 = (a >> 8) & 0xFF
        b2 = b & 0xFF
        b3 = (b >> 8) & 0xFF
        return (b0) | (b1 << 8) | (b2 << 16) | (b3 << 24)

    @staticmethod
    def add(a: int, b: int) -> int:
        """4×8-bit SIMD 加法（每个字节独立相加，结果截断到 8-bit）"""
        r0 = (Vec4ALU._byte(a, 0) + Vec4ALU._byte(b, 0)) & 0xFF
        r1 = (Vec4ALU._byte(a, 1) + Vec4ALU._byte(b, 1)) & 0xFF
        r2 = (Vec4ALU._byte(a, 2) + Vec4ALU._byte(b, 2)) & 0xFF
        r3 = (Vec4ALU._byte(a, 3) + Vec4ALU._byte(b, 3)) & 0xFF
        return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24)

    @staticmethod
    def mul(a: int, b: int) -> int:
        """4×8-bit SIMD 乘法（每个字节独立相乘，结果截断到 8-bit）"""
        r0 = (Vec4ALU._byte(a, 0) * Vec4ALU._byte(b, 0)) & 0xFF
        r1 = (Vec4ALU._byte(a, 1) * Vec4ALU._byte(b, 1)) & 0xFF
        r2 = (Vec4ALU._byte(a, 2) * Vec4ALU._byte(b, 2)) & 0xFF
        r3 = (Vec4ALU._byte(a, 3) * Vec4ALU._byte(b, 3)) & 0xFF
        return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24)

    @staticmethod
    def unpack(word: int, lane: int) -> int:
        """提取第 lane 个字节 (0-3)，零扩展到 32-bit"""
        return Vec4ALU._byte(word, lane & 3)
