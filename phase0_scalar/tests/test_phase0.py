#!/usr/bin/env python3
"""
Phase 0 测试套件
=================
测试标量处理器的所有模块：ALU, RegisterFile, Memory, ISA, Assembler, CPU

对标 GPGPU-Sim 的回归测试 (gpgpu-sim_simulations/)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from isa import (
    decode, encode_rtype, encode_itype, encode_stype,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_LD, OP_ST, OP_MOV, OP_HALT,
    Instruction
)
from register_file import RegisterFile
from alu import ALU
from memory import Memory
from cpu import CPU
from assembler import assemble


# ============================================================
# Test Framework
# ============================================================
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
    """运行一个汇编测试并验证结果

    checks: dict of {mem_addr: expected_value} 或 callable(cpu)
    """
    global passed, failed
    asm_path = os.path.join(os.path.dirname(__file__), 'programs', asm_file)
    with open(asm_path) as f:
        program = assemble(f.read())

    cpu = CPU()
    cpu.load_program(program)
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
            if actual == expected:
                passed += 1
                print(f"  ✅ {asm_file}: mem[{addr}] = {expected}")
            else:
                failed += 1
                print(f"  ❌ {asm_file}: mem[{addr}] = {actual}, expected {expected}")


# ============================================================
# Unit Tests
# ============================================================

def test_alu():
    print("\n--- ALU Tests ---")
    alu = ALU()
    check(alu.add(5, 3) == 8, "ADD 5+3=8")
    check(alu.sub(10, 7) == 3, "SUB 10-7=3")
    check(alu.mul(6, 7) == 42, "MUL 6*7=42")
    check(alu.div(42, 6) == 7, "DIV 42/6=7")
    check(alu.div(10, 0) == 0, "DIV by zero → 0")
    check(alu.add(0xFFFFFFFF, 1) == 0, "ADD overflow wrap")
    check(alu.sub(0, 1) == 0xFFFFFFFF, "SUB underflow wrap")
    # 负数
    check(alu.add(-10 & 0xFFFFFFFF, -5 & 0xFFFFFFFF) == (-15 & 0xFFFFFFFF), "ADD negative")
    check(alu.mul(-3 & 0xFFFFFFFF, 7) == (-21 & 0xFFFFFFFF), "MUL negative")


def test_register_file():
    print("\n--- RegisterFile Tests ---")
    rf = RegisterFile(16)
    check(rf.read(0) == 0, "r0 reads as 0")
    rf.write(0, 42)
    check(rf.read(0) == 0, "r0 write ignored")
    rf.write(1, 0xDEADBEEF)
    check(rf.read(1) == 0xDEADBEEF, "write/read r1")
    rf.write(2, 0xFFFFFFFF)
    check(rf.read(2) == 0xFFFFFFFF, "32-bit value preserved")
    try:
        rf.read(16)
        check(False, "out-of-range read should raise")
    except IndexError:
        check(True, "out-of-range read raises IndexError")


def test_memory():
    print("\n--- Memory Tests ---")
    mem = Memory(256)
    mem.write_word(0, 42)
    check(mem.read_word(0) == 42, "write/read word")
    mem.write_word(255, 0xDEADBEEF)
    check(mem.read_word(255) == 0xDEADBEEF, "last address")
    try:
        mem.read_word(256)
        check(False, "out-of-range read should raise")
    except IndexError:
        check(True, "out-of-range read raises IndexError")
    mem.write_word(10, 0x12345678)
    check(mem.read_word(10) == 0x12345678, "byte order preserved")


def test_isa():
    print("\n--- ISA Tests ---")
    # R-type encoding/decoding
    w = encode_rtype(OP_ADD, 1, 2, 3)
    instr = decode(w)
    check(instr.opcode == OP_ADD, "R-type opcode")
    check(instr.rd == 1, "R-type rd")
    check(instr.rs1 == 2, "R-type rs1")
    check(instr.rs2 == 3, "R-type rs2")

    # I-type
    w = encode_itype(OP_MOV, 5, 42)
    instr = decode(w)
    check(instr.opcode == OP_MOV, "I-type opcode")
    check(instr.rd == 5, "I-type rd")
    check(instr.imm == 42, "I-type imm")

    # S-type
    w = encode_stype(OP_ST, 3, 100)
    instr = decode(w)
    check(instr.opcode == OP_ST, "S-type opcode")
    check(instr.rs1 == 3, "S-type rs1")
    check(instr.imm == 100, "S-type imm")

    # HALT
    w = encode_rtype(OP_HALT, 0, 0, 0)
    instr = decode(w)
    check(instr.opcode == OP_HALT, "HALT opcode")

    # Negative immediate sign-extension
    w = encode_itype(OP_MOV, 1, 0xFFF)  # -1 in 12-bit
    instr = decode(w)
    check(instr.imm == -1 & 0xFFFFFFFF, "negative imm sign-extended")


def test_assembler():
    print("\n--- Assembler Tests ---")
    src = """
    MOV r1, 5
    MOV r2, 3
    ADD r3, r1, r2
    ST r3, [10]
    HALT
    """
    prog = assemble(src)
    check(len(prog) == 5, "5 instructions assembled")
    # Verify first instruction: MOV r1, 5
    instr = decode(prog[0])
    check(instr.opcode == OP_MOV, "asm MOV opcode")
    check(instr.rd == 1, "asm MOV rd=r1")
    check(instr.imm == 5, "asm MOV imm=5")
    # Verify ADD
    instr = decode(prog[2])
    check(instr.opcode == OP_ADD, "asm ADD opcode")
    # Verify HALT
    instr = decode(prog[4])
    check(instr.opcode == OP_HALT, "asm HALT")


# ============================================================
# Integration Tests (Assembly Programs)
# ============================================================

def test_programs():
    print("\n--- Assembly Program Tests ---")
    run_asm_test("01_basic_arith.asm", {0: 8})
    run_asm_test("02_mul_div.asm", {0: 42, 1: 7})
    run_asm_test("03_memory.asm", {30: 300})
    run_asm_test("04_r0_protect.asm", {0: 42})
    run_asm_test("05_negative.asm", {
        0: -30 & 0xFFFFFFFF,
        1: -15 & 0xFFFFFFFF,
        2: -5 & 0xFFFFFFFF,
    })
    run_asm_test("06_complex.asm", {0: 45})


# ============================================================
# Main
# ============================================================

def main():
    global passed, failed
    print("=" * 60)
    print("Phase 0: Scalar Processor — Test Suite")
    print("=" * 60)

    test_alu()
    test_register_file()
    test_memory()
    test_isa()
    test_assembler()
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
