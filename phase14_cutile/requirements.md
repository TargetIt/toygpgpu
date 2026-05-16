# Phase 14: 学习控制台需求 + Warp 通信 + 分块策略 + CuTile 编程模型

## New Features (2026-05-16) — Phase 14

The following CuTile Programming Model features were added in Phase 14:

- **CuTile DSL Parser** (`src/cutile_parser.py`, ~237 lines): A high-level tile-oriented domain-specific language that auto-generates toygpgpu ISA assembly from concise tile descriptions. Inspired by NVIDIA CUTLASS and Triton DSL.
  - `parse_cutile(source: str) -> CuTileKernel`: Parses CuTile DSL source into an IR (Intermediate Representation) containing TileConfig, parameters, and operation sequence.
  - `generate_asm(kernel, matrix_data) -> str`: Generates toygpgpu assembly (TLCONF + TLDS + SHLD/MUL/ADD + TLSTS + HALT) from the CuTileKernel IR.
  - `assemble_cutile(source, matrix_data) -> (machine_code, asm_text)`: Convenience function performing full parse → generate → assemble pipeline.
  - Supported comment styles: `//` C++ style and `;` assembly style.

- **CuTile DSL Syntax**: Three operation types describe a tile computation:
  - `tile M=<int>, N=<int>, K=<int>`: Declares tile dimensions (must appear before kernel body).
  - `kernel <name>(<param>:[M,K], <param>:[K,N], <param>:[M,N]) { ... }`: Declares kernel with tile-shaped parameters.
  - `load <matrix> -> smem[<offset>]`: Loads a tile from global memory to shared memory (generates TLDS).
  - `mma smem[<A>], smem[<B>] -> smem[<C>]`: Computes matrix multiply-accumulate on shared memory tiles (generates expanded SHLD + MUL + ADD loop).
  - `store smem[<off>] -> <matrix>`: Stores a result tile from shared memory to global memory (generates TLSTS).

- **OP_WGMMA (opcode 0x38) — Warp-Group MMA**: A hardware-accelerated matrix multiply-accumulate instruction operating directly on shared memory tiles.
  - Syntax: `WGMMA smem_a, smem_b, smem_c`
  - Semantics: `smem[smem_c] += A × B` where A at `smem[smem_a]` (M×K), B at `smem[smem_b]` (K×N), C at `smem[smem_c]` (M×N)
  - Uses tile dimensions (M, N, K) from the preceding TLCONF instruction
  - Accumulates into existing C values rather than overwriting
  - Encoding: `encode_rtype(OP_WGMMA, smem_a, smem_b, smem_c)`

- **New ISA additions**:
  - `OP_WGMMA = 0x38` in `isa.py`
  - `OPCODE_NAMES[OP_WGMMA] = "WGMMA"` in `isa.py`
  - `WGMMA` mnemonic in `assembler.py`: `WGMMA smem_a, smem_b, smem_c` → `encode_rtype(0x38, smem_a, smem_b, smem_c)`

- **WGMMA Execution in simt_core.py**: `_execute_warp()` handles OP_WGMMA by reading A/B tiles from shared memory, computing the full M×N×K matrix multiplication, and writing C tile back to shared memory. Accumulates into existing C values:
  ```python
  if op == OP_WGMMA:
      for m in range(M):
          for n in range(N):
              acc = smem[smem_c + m*N + n]  # existing C
              for k in range(K):
                  acc += smem[smem_a + m*K + k] * smem[smem_b + k*N + n]
              smem[smem_c + m*N + n] = acc
  ```

- **New demo program** (`tests/programs/13_cutile_matmul.cutile`):
  - CuTile DSL source for 2×2 matrix multiplication
  - Tile: M=2, N=2, K=2
  - Operations: load A → smem[0], load B → smem[4], mma → smem[8], store smem[8] → C
  - Auto-generates ~40 lines of assembly from 4 DSL statements

- **Two programming approaches**:
  1. **CuTile DSL path**: Write `.cutile` file → `parse_cutile()` + `generate_asm()` → assembled machine code. High-level, auto-generated, educational.
  2. **Direct WGMMA path**: Write `.asm` with `TLCONF` + `TLDS` + `WGMMA` + `TLSTS`. Low-level, compact, hardware-accelerated.

- **Comprehensive Phase 14 test suite** (`tests/test_phase14.py`): 6 test categories, all passing:
  1. ISA unit tests: OP_WGMMA value (0x38) and OPCODE_NAMES entry verification
  2. CuTile Parser tests: kernel name, tile M/N/K values, 4 operations parsed (load/load/mma/store)
  3. CuTile Code Generation tests: generated TLCONF, TLDS, SHLD, MUL, ADD, HALT present in output
  4. WGMMA instruction test: direct WGMMA execution via assembly (TLCONF + TLDS + WGMMA + TLSTS)
  5. CuTile End-to-End test: full DSL→asm→assemble→execute→verify pipeline, correct 2×2 matmul result
  6. Backward compatibility test: Phase 13 TLDS/SHLD and Phase 12 SHFL still work unchanged

- **Mathematical verification**: All WGMMA and CuTile DSL results match expected:
  - A=[[1,2],[3,4]], B=[[5,6],[7,8]]
  - C[0][0]=19, C[0][1]=22, C[1][0]=43, C[1][1]=50

- **Clean module separation**: `cutile_parser.py` is a standalone module with no dependencies on simt_core or other execution modules. It only imports from `assembler.py` for `assemble_cutile()`. Can be tested independently.

- **Full backward compatibility**: All existing Phase 0-13 programs run unchanged. Existing TLCONF/TLDS/TLSTS/TLSTS, SHFL/VOTE/BALLOT, and all earlier instructions remain identical.

## New Features (2026-05-16) — Phase 13

The following tiling instructions were added in Phase 13:

- **TLCONF (Tile Config, opcode 0x35)**: Configures tile dimensions for subsequent TLDS/TLSTS operations.
  - `rd` encodes tile_M (rows), `rs1` encodes tile_N (columns), `imm` encodes tile_K (inner dimension)
  - Updates SIMTCore state: `tile_m`, `tile_n`, `tile_k`
  - Default values at SIMTCore init: tile_m=8, tile_n=8, tile_k=8
  - Assembly syntax: `TLCONF M, N, K`
  - Example: `TLCONF 2, 2, 2` configures a 2x2 tile with inner dimension 2

- **TLDS (Tile Load, opcode 0x36)**: Loads a tile from global memory into shared memory.
  - `rd` encodes shared memory offset, `rs1` encodes global memory base address
  - Each active thread maps `thread_id` to (row, col) within the tile: `row = tid // tile_n`, `col = tid % tile_n`
  - Global memory address is contiguous: `glob_base + tid`
  - Shared memory address is row-major tiled: `smem_off + row * tile_n + col`
  - Assembly syntax: `TLDS smem_offset, glob_base`
  - Example: `TLDS 0, 0` loads data from mem[0..3] to smem[0..3] (for tile_M=2, tile_N=2)

- **TLSTS (Tile Store, opcode 0x37)**: Stores a tile from shared memory back to global memory.
  - Inverse of TLDS: reads from shared memory, writes to global memory
  - Same (row, col) mapping as TLDS
  - Assembly syntax: `TLSTS smem_offset, glob_base`
  - Example: `TLSTS 0, 100` stores smem[0..3] to mem[100..103]

- **SIMTCore tile state**: Three new fields in SIMTCore:
  - `self.tile_m`: Tile row count (M dimension)
  - `self.tile_n`: Tile column count (N dimension)
  - `self.tile_k`: Tile inner dimension (K dimension, shared dimension for matmul)

- **Two new demo programs**:
  - `tests/programs/11_tiled_matmul.asm`: 2x2 tiled matrix multiplication using TLDS to load A and B tiles into shared memory, SHLD to read from tile buffer, and ALU operations for multiply-accumulate. Verifies C = A x B for 2x2 matrices.
  - `tests/programs/12_tile_double_buffer.asm`: Ping-pong double buffering pattern. Two tile buffers in shared memory (buf_A at smem[0..7], buf_B at smem[8..15]) demonstrate the concept of processing one tile while the other is loaded.

- **Comprehensive Phase 13 test suite** (`tests/test_phase13.py`): 6 test categories, all passing:
  1. ISA unit tests: opcode values for TLCONF(0x35), TLDS(0x36), TLSTS(0x37) and OPCODE_NAMES entries
  2. Assembler tests: TLCONF encoding with rd/rs1/imm fields, TLDS and TLSTS mnenonic encoding
  3. Tile config state test: default tile_m/n/k values and TLCONF state update verification
  4. Tiled matmul program test: full end-to-end 2x2 matrix multiplication via shared memory tiles
  5. Double buffer program test: end-to-end ping-pong pattern with two tile buffers
  6. Backward compatibility test: SHFL, VOTE, PRED all still work from Phase 12 and earlier

- **Inter-module data flow**: TLDS connects memory.read_word() to shared_memory.write_word(); TLSTS is the reverse. This is the first phase where memory and shared_memory modules are explicitly linked in instruction execution.

- **Full backward compatibility**: All existing Phase 0-12 programs run unchanged. Existing SHLD/SHST shared memory instructions, warp communication primitives, and ALU operations remain identical.

- **Educational focus**: Demonstrates the fundamental tiling pattern used in GPU matrix multiplication libraries (CUTLASS, cuBLAS) and high-performance computing kernels.

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
