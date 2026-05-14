## Quick Start

Full interactive debugging experience with ANSI-colored pipeline visualization:

```bash
# Interactive debugging with divergence demo (recommended)
python src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4

# Auto-play mode (0.5s per cycle)
python src/learning_console.py tests/programs/demo_basic.asm --auto-interval 0.5

# Batch trace mode
python src/learning_console.py tests/programs/demo_divergence.asm --trace

# Custom warp/thread config
python src/learning_console.py tests/programs/demo_basic.asm --warp-size 8 --num-warps 2 --max-cycles 200
```

## New in this update

- **Enhanced interactive debugger**: Full ANSI color output with rich pipeline state rendering
- **console_display.py**: Dedicated formatting module for cycle rendering, register/memory diff, pipeline stage display
- **Breakpoint support**: `b <pc>` to set breakpoints, `b list` to list all, `b clear` to clear
- **I-Buffer visualization**: Real-time instruction buffer slot status (valid/ready/PC)
- **Scoreboard visualization**: Pending write countdown display per register
- **SIMT Stack visualization**: Branch divergence entries with reconvergence PC, masks
- **Memory delta tracking**: Automatic detection and display of memory changes between cycles
- **Warp-level register inspection**: `wreg` command to display all warp-uniform registers
- **PRED/Predication display**: Predicate state shown in per-thread register views
- **vec4/float4 support**: Vec4 composite data type operations fully displayed in pipeline view
- **Interactive commands**: `Enter`=step, `s`=step, `r`=run, `r N`=run N cycles, `i`=state, `m`=memory, `reg`=registers, `wreg`=warp regs, `sb`=scoreboard, `ib`=I-Buffer, `stack`=SIMT Stack, `b`=breakpoint, `q`=quit

# Phase 11: 交互式学习控制台

GDB 风格的 GPU 流水线单步调试器。每周期显示完整内部状态。

## 特性
- 五级流水线: FETCH/DECODE/ISSUE/EXEC/WB
- 寄存器变化追踪 (old→new 含符号解释)
- Scoreboard 倒计时可视化
- I-Buffer 槽位状态
- SIMT Stack 栈内容
- 内存变化 delta
- ANSI 颜色标注
- 断点支持

## 使用

```bash
# 交互式单步
python3 src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4

# 自动播放
python3 src/learning_console.py tests/programs/demo_basic.asm --auto-interval 0.5
```

## 交互命令
- `Enter` 单步 | `r` 自动运行 | `b 5` 断点 | `q` 退出
- `reg/sb/ib/stack/m` 查看状态

## 对标
GDB `stepi` 模式 + GPGPU-Sim 无等价物（教学定制）
