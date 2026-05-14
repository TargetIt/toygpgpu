## Quick Start

Interactive debugging with learning_console.py (scoreboard hazard visualization):

```bash
# Interactive debugging with RAW hazard detection
python src/learning_console.py tests/programs/01_raw_hazard.asm --warp-size 4

# Batch trace mode
python src/learning_console.py tests/programs/01_raw_hazard.asm --trace --warp-size 4

# WAW hazard demo
python src/learning_console.py tests/programs/02_waw_hazard.asm --warp-size 4

# Load latency demo
python src/learning_console.py tests/programs/03_ld_latency.asm --warp-size 4
```

## New in this update

- **learning_console.py**: Interactive scoreboard debugger with hazard detection display, pending write countdown visualization
- **Interactive commands**: `Enter`=step, `r`=run, `i`=state, `m`=memory, `reg`=registers, `sb`=scoreboard, `q`=quit
- **PRED/Predication**: Full `@p0` predicate support in assembler, SETP instruction, per-thread predicate register
- **vec4/float4 instructions**: V4PACK, V4ADD, V4MUL, V4UNPACK with scoreboard hazard tracking
- **Warp-level registers**: WREAD/WWRITE instructions for warp-uniform register access
- **Trace mode**: `--trace` for batch execution with scoreboard state tracking

# Phase 4: Scoreboard (流水线冒险)

实现记分板检测 RAW/WAW 数据冒险。对标 GPGPU-Sim `scoreboard`。

## 新增概念
- Scoreboard: 每寄存器 pending write 倒计时
- RAW (读后写) 冒险检测
- WAW (写后写) 冒险检测
- 流水线延迟: ALU=1cycle, LD/ST=4cycles
- warp stall + 恢复机制

## 运行

```bash
cd phase4_scoreboard && bash run.sh
```

## 测试
15 项测试，含 RAW stall、WAW stall、LD 延迟后正确读回。

## 对标 GPGPU-Sim
`scoreboard` (register hazard tracking)
