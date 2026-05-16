#!/usr/bin/env python3
"""Phase 21 Test Suite — Mixed Precision Tensor"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mix_precision import *
import struct

passed = 0; failed = 0
def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"  OK {name}")
    else: failed += 1; print(f"  FAIL {name}")


def test_fp16():
    print("\n--- FP16 Conversion Tests ---")
    # 1.0 → FP16 → 1.0
    f16 = float_to_fp16(1.0)
    f32 = fp16_to_float(f16)
    check(abs(f32 - 1.0) < 0.001, f"1.0 roundtrip: {f32}")

    # 0 → FP16
    check(float_to_fp16(0) == 0, "0 → FP16 = 0")

    # Negative
    f16_neg = float_to_fp16(-2.5)
    f32_neg = fp16_to_float(f16_neg)
    check(abs(f32_neg - (-2.5)) < 0.01, f"-2.5 roundtrip: {f32_neg}")


def test_fp8_e4m3():
    print("\n--- FP8 E4M3 Conversion Tests ---")
    # 1.0 → E4M3 → back
    f8 = float_to_fp8_e4m3(1.0)
    check(f8 & 0xFF > 0, f"1.0 non-zero encoded: 0x{f8:02X}")
    f32 = fp8_e4m3_to_float(f8)
    check(abs(f32 - 1.0) < 0.15, f"1.0 E4M3 roundtrip: {f32} (error < 0.15)")

    # 0
    check(float_to_fp8_e4m3(0) == 0, "0 → E4M3 = 0")


def test_fp8_e5m2():
    print("\n--- FP8 E5M2 Conversion Tests ---")
    # Larger range than E4M3
    f8 = float_to_fp8_e5m2(100.0)
    f32 = fp8_e5m2_to_float(f8)
    check(abs(f32 - 100.0) < 10.0, f"100 E5M2 roundtrip: {f32} (wider range, less precision)")


def test_pack_unpack():
    print("\n--- FP8 Pack/Unpack Tests ---")
    a = float_to_fp8_e4m3(2.0)
    b = float_to_fp8_e4m3(3.0)
    packed = pack_fp8_pair(a, b)
    a2, b2 = unpack_fp8_pair(packed)
    check(abs(fp8_e4m3_to_float(a2) - 2.0) < 0.3, "pack/unpack a=2.0")
    check(abs(fp8_e4m3_to_float(b2) - 3.0) < 0.5, "pack/unpack b=3.0")


def test_fp8_mma():
    print("\n--- FP8 MMA Tests ---")
    a_packed = pack_fp8_pair(float_to_fp8_e4m3(1.0), float_to_fp8_e4m3(2.0))
    b_packed = pack_fp8_pair(float_to_fp8_e4m3(3.0), float_to_fp8_e4m3(4.0))
    # 1*3 + 2*4 + 0 = 11
    result = fp8_mma(a_packed, b_packed, 0, FMT_FP8_E4M3)
    f32 = struct.unpack('f', struct.pack('I', result))[0]
    check(abs(f32 - 11.0) < 0.5, f"FP8 MMA: 1*3+2*4 = 11 (got {f32})")


def test_convert():
    print("\n--- Precision Convert Tests ---")
    f32_bits = struct.unpack('I', struct.pack('f', 3.14159))[0]
    f16 = convert(f32_bits, FMT_FP32, FMT_FP16)
    back = fp16_to_float(f16)
    check(abs(back - 3.14159) < 0.01, f"FP32→FP16: {back}")


def test_precision_stats():
    print("\n--- Precision Stats Tests ---")
    ps = PrecisionStats()
    ps.record(1.0, 1.05)
    ps.record(2.0, 1.98)
    check(ps.total_conversions == 2, "2 conversions recorded")
    check(ps.max_error > 0, "max_error tracked")
    rpt = ps.report()
    check("PrecisionStats" in rpt, "report has title")


def test_backward_compat():
    print("\n--- Backward Compat Tests ---")
    from tma import TMAEngine, TensorDescriptor
    tma = TMAEngine()
    desc = TensorDescriptor([2,2])
    tma.register_descriptor(0, desc)
    check(len(tma.compute_addresses(0,[0,0],[2,2])) == 4, "Phase 20 TMA OK")
    from l2_cache import L2Cache
    check(L2Cache().total_lines == 256, "Phase 19 L2 OK")


def main():
    global passed, failed
    print("=" * 60)
    print("Phase 21: Mixed Precision — Test Suite")
    print("=" * 60)
    test_fp16()
    test_fp8_e4m3()
    test_fp8_e5m2()
    test_pack_unpack()
    test_fp8_mma()
    test_convert()
    test_precision_stats()
    test_backward_compat()
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
