"""
Vec4ALU — Vec4 复合数据类型算术逻辑单元
"""
class Vec4ALU:
    @staticmethod
    def _byte(word: int, lane: int) -> int:
        return (word >> (lane * 8)) & 0xFF

    @staticmethod
    def pack(a: int, b: int) -> int:
        b0 = a & 0xFF
        b1 = (a >> 8) & 0xFF
        b2 = b & 0xFF
        b3 = (b >> 8) & 0xFF
        return (b0) | (b1 << 8) | (b2 << 16) | (b3 << 24)

    @staticmethod
    def add(a: int, b: int) -> int:
        r0 = (Vec4ALU._byte(a, 0) + Vec4ALU._byte(b, 0)) & 0xFF
        r1 = (Vec4ALU._byte(a, 1) + Vec4ALU._byte(b, 1)) & 0xFF
        r2 = (Vec4ALU._byte(a, 2) + Vec4ALU._byte(b, 2)) & 0xFF
        r3 = (Vec4ALU._byte(a, 3) + Vec4ALU._byte(b, 3)) & 0xFF
        return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24)

    @staticmethod
    def mul(a: int, b: int) -> int:
        r0 = (Vec4ALU._byte(a, 0) * Vec4ALU._byte(b, 0)) & 0xFF
        r1 = (Vec4ALU._byte(a, 1) * Vec4ALU._byte(b, 1)) & 0xFF
        r2 = (Vec4ALU._byte(a, 2) * Vec4ALU._byte(b, 2)) & 0xFF
        r3 = (Vec4ALU._byte(a, 3) * Vec4ALU._byte(b, 3)) & 0xFF
        return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24)

    @staticmethod
    def unpack(word: int, lane: int) -> int:
        return Vec4ALU._byte(word, lane & 3)
