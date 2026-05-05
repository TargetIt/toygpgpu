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
