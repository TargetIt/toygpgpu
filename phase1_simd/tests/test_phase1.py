#!/usr/bin/env python3
"""
Phase 1 测试套件
=================
测试 SIMD 向量处理器所有模块及 Phase 0 向后兼容性。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isa import *
from register_file import RegisterFile
from vector_register_file import VectorRegisterFile
from alu import ALU
from vector_alu import VectorALU
from memory import Memory
from cpu import CPU
from assembler import assemble

passed = 0
failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")


def run_asm_test(asm_file, checks):
    global passed, failed
    path = os.path.join(os.path.dirname(__file__), 'programs', asm_file)
    with open(path) as f:
        prog = assemble(f.read())
    cpu = CPU()
    cpu.load_program(prog)
    cpu.run()
    if callable(checks):
        ok, msg = checks(cpu)
        if ok:
            passed += 1
            print(f"  ✅ {asm_file}: {msg}")
        else:
            failed += 1
            print(f"  ❌ {asm_file}: {msg}")
    else:
        for addr, expected in checks.items():
            actual = cpu.memory.read_word(addr)
            if actual == (expected & 0xFFFFFFFF):
                passed += 1
                print(f"  ✅ {asm_file}: mem[{addr}] = {expected}")
            else:
                failed += 1
                print(f"  ❌ {asm_file}: mem[{addr}] = {actual}, expected {expected}")


# ============================================================
# Unit Tests
# ============================================================

def test_vector_register_file():
    print("\n--- VectorRegisterFile Tests ---")
    vrf = VectorRegisterFile(vlen=8, num_regs=8)
    vrf.write(1, [10, 20, 30, 40, 50, 60, 70, 80])
    vals = vrf.read(1)
    check(vals == [10, 20, 30, 40, 50, 60, 70, 80], "write/read full vector")
    check(vrf.read_lane(1, 0) == 10, "read lane 0")
    check(vrf.read_lane(1, 7) == 80, "read lane 7")
    vrf.broadcast(2, 42)
    check(vrf.read(2) == [42] * 8, "broadcast")
    try:
        vrf.write(8, [0])
        check(False, "invalid reg should raise")
    except IndexError:
        check(True, "invalid reg raises IndexError")


def test_vector_alu():
    print("\n--- VectorALU Tests ---")
    valu = VectorALU(vlen=8)
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [10, 20, 30, 40, 50, 60, 70, 80]
    check(valu.vadd(a, b) == [11, 22, 33, 44, 55, 66, 77, 88], "VADD")
    check(valu.vsub(b, a) == [9, 18, 27, 36, 45, 54, 63, 72], "VSUB")
    check(valu.vmul(a, b) == [10, 40, 90, 160, 250, 360, 490, 640], "VMUL")
    check(valu.vdiv(b, a) == [10, 10, 10, 10, 10, 10, 10, 10], "VDIV")
    # Negative
    neg_a = [(-x) & 0xFFFFFFFF for x in a]
    check(valu.vadd(neg_a, b) == [9, 18, 27, 36, 45, 54, 63, 72], "VADD negative")


def test_isa_vector():
    print("\n--- ISA Vector Tests ---")
    for opcode, name in [(OP_VADD, "VADD"), (OP_VSUB, "VSUB"),
                          (OP_VMUL, "VMUL"), (OP_VDIV, "VDIV")]:
        w = encode_rtype(opcode, 1, 2, 3)
        instr = decode(w)
        check(instr.opcode == opcode, f"{name} encode/decode")
        check(is_vector(instr.opcode), f"{name} is_vector")
    w = encode_itype(OP_VLD, 1, 100)
    check(decode(w).opcode == OP_VLD, "VLD encode/decode")
    w = encode_stype(OP_VST, 2, 200)
    check(decode(w).opcode == OP_VST, "VST encode/decode")
    w = encode_itype(OP_VMOV, 3, 42)
    check(decode(w).opcode == OP_VMOV, "VMOV encode/decode")


def test_assembler_vector():
    print("\n--- Assembler Vector Tests ---")
    src = """
    VADD v1, v2, v3
    VMOV v0, 42
    VLD v4, [16]
    VST v5, [32]
    HALT
    """
    prog = assemble(src)
    check(len(prog) == 5, "5 instructions")
    check(decode(prog[0]).opcode == OP_VADD, "asm VADD")
    check(decode(prog[1]).opcode == OP_VMOV, "asm VMOV")
    check(decode(prog[2]).opcode == OP_VLD, "asm VLD")
    check(decode(prog[3]).opcode == OP_VST, "asm VST")


# ============================================================
# Integration Tests
# ============================================================

def test_programs():
    print("\n--- Assembly Program Tests ---")

    # Test 01: Vector Add
    def check_vadd(cpu):
        expected = [11, 22, 33, 44, 55, 66, 77, 88]
        for i, e in enumerate(expected):
            if cpu.memory.read_word(16 + i) != e:
                return False, f"mem[{16 + i}] != {e}"
        return True, "C = A + B correct"
    run_asm_test("01_vector_add.asm", check_vadd)

    # Test 02: Vector Multiply
    def check_vmul(cpu):
        expected = [3, 6, 9, 12, 15, 18, 21, 24]
        for i, e in enumerate(expected):
            if cpu.memory.read_word(10 + i) != e:
                return False, f"mem[{10 + i}] != {e}"
        return True, "C = A * 3 correct"
    run_asm_test("02_vector_mul.asm", check_vmul)

    # Test 03: Vector Sub + Div
    run_asm_test("03_vector_sub_div.asm", {
        10: 95, 11: 75, 12: 55, 13: 35, 14: 15, 15: 5, 16: 3,
        17: (-1) & 0xFFFFFFFF,
        20: 47, 21: 37, 22: 27, 23: 17, 24: 7, 25: 2, 26: 1, 27: 0,
    })

    # Test 04: Mixed Scalar + Vector
    def check_mixed(cpu):
        expected = [33, 66, 99, 132, 165, 198, 231, 264]
        for i, e in enumerate(expected):
            if cpu.memory.read_word(20 + i) != e:
                return False, f"mem[{20 + i}] != {e}"
        return True, "mixed scalar+vector correct"
    run_asm_test("04_mixed_scalar_vector.asm", check_mixed)

    # Test 05: Phase 0 backward compatibility
    run_asm_test("05_phase0_compat.asm", {0: 8, 1: 42, 2: 50, 3: 50})


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 1: SIMD Vector Processor — Test Suite")
    print("=" * 60)

    test_vector_register_file()
    test_vector_alu()
    test_isa_vector()
    test_assembler_vector()
    test_programs()

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    if failed > 0:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
