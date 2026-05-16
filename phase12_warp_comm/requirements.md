# Phase 12: 学习控制台需求 + Warp 通信

## New Features (2026-05-16) — Phase 12

The following warp-level communication primitives were added in Phase 12:

- **SHFL (Warp Shuffle, opcode 0x30)**: Direct cross-thread register read within a warp. Supports 4 modes via the imm field:
  - `mode=0` (IDX): Read register from a specific lane — `rd = warp.threads[lane].rs1`
  - `mode=1` (UP): Read from `(tid - delta)` — cyclic shift upward
  - `mode=2` (DOWN): Read from `(tid + delta)` — cyclic shift downward
  - `mode=3` (XOR): Read from `(tid ^ mask)` — butterfly/pairwise exchange
  - Assembly syntax: `SHFL rd, rs1, src, mode`

- **VOTE (Warp Vote, opcode 0x33)**: Warp-wide logical reduction on predicate values. Two sub-modes via `imm[0]`:
  - `VOTE.ANY rd, rs1` (imm[0]=0): Sets rd=1 if ANY active thread has rs1 != 0, else rd=0
  - `VOTE.ALL rd, rs1` (imm[0]=1): Sets rd=1 if ALL active threads have rs1 != 0, else rd=0
  - Implements early-exit optimization: VOTE.ANY breaks on first non-zero; VOTE.ALL breaks on first zero

- **BALLOT (Warp Ballot, opcode 0x34)**: Creates a warp-wide bitmask where bit N is set if thread N's rs1 is non-zero. All active threads receive the same mask value.
  - Assembly syntax: `BALLOT rd, rs1`
  - Example (warp_size=8, even threads non-zero): result = 0b01010101 = 0x55

- **Two new demo programs**:
  - `tests/programs/09_warp_shfl.asm`: Demonstrates IDX, DOWN(1), XOR(1) modes. Each thread computes `tid*10`, then exchanges values via all three SHFL modes.
  - `tests/programs/10_warp_vote.asm`: Demonstrates VOTE.ANY, VOTE.ALL, BALLOT with even/odd thread value distribution.

- **Comprehensive Phase 12 test suite** (`tests/test_phase12.py`): 5 test categories, all passing:
  1. ISA unit tests: opcode values and OPCODE_NAMES entries
  2. Assembler tests: SHFL encoding with modes, VOTE.ANY/ALL imm[0] bit, BALLOT
  3. SHFL program test: full end-to-end verification of all 3 SHFL modes via memory checks
  4. VOTE/BALLOT program test: ANY, ALL, BALLOT results verified, including all-ones edge case
  5. Backward compatibility test: divergence, basic ALU, predication, warp regs all unchanged

- **Zero pipeline changes**: SHFL/VOTE/BALLOT execute in the existing EXEC stage with 1-cycle latency (same as ALU instructions). No new pipeline stages, scoreboard modifications, or module additions needed.

- **CUDA semantic correspondence**: SHFL=__shfl_sync, VOTE.ANY=__any_sync, VOTE.ALL=__all_sync, BALLOT=__ballot_sync. Primitive names chosen for clarity in an educational context.

## New Features (2026-05-15) — Phase 11 (Keep from Phase 11)

The following features were added after the initial release:

- **Fixed IBuffer peek and reconvergence bug**: Added `peek()` method to IBuffer for non-destructive instruction inspection, and fixed the SIMT stack reconvergence check logic (same fix as Phase 7).
- **PRED (predication) support**: Added `OP_SETP` instruction (opcode 0x24), `@p0` prefix syntax for conditional execution, and per-thread predication bit tracking (same as Phase 3).
- **Warp-level uniform registers**: Added `WREAD` (opcode 0x2A) and `WWRITE` (opcode 0x2B) instructions for warp-level register read/write operations (same as Phase 3).
- **vec4_alu.py and V4PACK/V4ADD/V4MUL/V4UNPACK instructions**: Added the Vec4ALU 4x8-bit SIMD composite data type and associated packed sub-word SIMD operations (opcodes 0x26-0x29).
- **--trace/--auto mode to learning_console.py**: Added command-line flags `--trace` for verbose instruction tracing and `--auto` for unattended continuous execution mode in the learning console.
- **run.sh --trace support**: Added `--trace` flag support to the `run.sh` launch script, propagating trace mode to the underlying simulator.

## 1. 目标
为初学者提供逐周期、逐流水级的 GPU 内部状态观察工具。

## 2. 功能
- FR-01: 交互式单步执行 (Enter 键)
- FR-02: 五级流水线可视化
- FR-03: 寄存器变化追踪
- FR-04: Scoreboard/I-Buffer/SIMT Stack 实时显示
- FR-05: 断点支持
- FR-06: 自动运行模式

## 3. 验收标准
- AC-01: 单步模式每周期输出完整状态
- AC-02: 流水线、Scoreboard、I-Buffer、Stack 同时可见
- AC-03: 断点命中时暂停
- AC-04: 分支发散程序的 SIMT Stack push/pop 可见
