
# 20260514
1. - [x] ✅ **已完成 (2026-05-15)** — 为每个 phase 添加 learning_console.py
   - Phase 0: 267 行, scalar CPU 逐步调试 (PC/寄存器/内存)
   - Phase 1: 350 行, SIMD 向量寄存器 + vreg 命令
   - Phase 2: 370 行, SIMT 多 warp 视图 + 每线程寄存器
   - Phase 3: 491 行, SIMT stack 发散/重汇聚追踪
   - Phase 4: 538 行, scoreboard 冲突检测 + sb 命令
   - Phase 5: 653 行, L1 cache + 共享内存 + coalescing
   - Phase 6: 656 行, 多 block kernel launch + perf 计数器
   - Phase 7: 542 行, 流水线 pipeline 视图 + I-Buffer/OC
   - Phase 8: 578 行, PTX 前端 + ptx 命令
   - Phase 9: 588 行, tensor/MMA 指令显示
   - Phase 10: 657 行, trace/timeline/report 可视化
   - Phase 11: 已有完整版 (329 行 + console_display.py)
2. - [x] ✅ **已验证 (2026-05-15)** — 所有 phase learning_console.py 输出正确
   - Phase 0: scalar 逐步执行，寄存器/内存变化清晰
   - Phase 3: SIMT stack 发散(DIVERGE)/重汇聚(RECONVERGE)完整追踪
   - Phase 7: pipeline FETCH/ISSUE/EXEC 各阶段 + I-Buffer/Scoreboard 状态
   - PRED demo: 偶线程写入100, SIMT stack空(无发散)
   - Warp regs demo: ntid广播写入, warp统一寄存器读写正常
# 20260509
1. - [x] ✅ **已修复 (2026-05-14)** — 添加 PRED 谓词分支支持
   - 在 phase3-11 添加 OP_SETP (0x24) 指令: SETP.EQ / SETP.NE
   - 每个线程增加 pred 位, @p0 前缀实现谓词化执行
   - PRED_FLAG (bit 31) 编码在 opcode 高位, decode() 用 0x7F 掩码剥离
   - 新增 06_predication.asm demo 程序展示 PRED vs DPC
2. - [x] ✅ **已修复 (2026-05-15)** — asm/ptx 程序注释 + 流程图
   - 48 个 .asm/.ptx 文件全部添加了双语(中/英)注释头
   - 每个程序顶部添加 ASCII 文本流程图
   - 修复 phase8 ptx_parser 行内 `;` 注释处理

# 20260507
完善如下几点：
1. - [x] ✅ **已修复 (2026-05-15)** — run.sh/CLI 可开关 trace
   - 所有 phase cpu.py/simt_core.py 的 `run(trace=True)` 实际输出详细信息
   - 所有 phase learning_console.py 添加 `--auto` / `--trace` 非交互模式
   - 所有 phase run.sh 支持 `--trace` 参数
2. - [x] ✅ **已修复** — 更详细的 trace
   - 每周期输出: PC, 指令, 寄存器变化, 内存读写
   - pipeline phase 显示 FETCH/ISSUE/EXEC/WB 各阶段状态
   - SIMT stack 发散/重汇聚时显示 stack push/pop
   - Scoreboard, I-Buffer, OperandCollector 状态实时可见
3. - [x] ✅ **已修复** — asm 程序注释
   - 见 #20260509 item 2
4. - [x] ✅ **已修复 (2026-05-15)** — float4/vec4 复合数据类型
   - 新增 V4PACK/V4ADD/V4MUL/V4UNPACK 指令 (opcodes 0x26-0x29)
   - 新增 vec4_alu.py (Vec4ALU) 支持 4×8-bit SIMD 字节操作
   - phase1+ 的 assembler/cpu/simt_core 支持 vec4 指令
   - 新增 06_vec4_demo.asm 演示程序
5. - [x] ✅ **已修复 (2026-05-15)** — warp level register
   - Warp 类新增 warp_regs 字典: wid(0), ntid(1) 等统一寄存器
   - 新增 WREAD/WWRITE 指令 (opcodes 0x2A-0x2B)
   - 新增 WREG_NAMES 映射表
   - 新增 08_warp_regs.asm 演示程序

  

# 20260507
可能多个phase都存在这个问题

- [x] ✅ **已修复 (2026-05-14)** — SIMT Stack reconv_pc 重汇聚 bug
  - 在所有 phase7-11 的 ibuffer.py 中增加了 `peek()` 方法
  - 在所有 phase7-11 的 simt_core.py 中改为优先用 `ibuffer.peek().pc` 判断重汇聚
  - 修复了所有 phase 测试文件中的 Windows GBK 编码问题 (统一使用 `encoding='utf-8'`)
  - 全部 11 个 phase 测试通过，demo_divergence 结果正确

定位到了，根因在 src/simt_core.py (line 116)。

分歧之后，SIMT stack 里记录的 reconv_pc 是”下一条应该切换/重汇聚的指令地址”。但原来的代码用 warp.pc 去检查是否到达重汇聚点，而这个 warp.pc 在 fetch 阶段已经自增过了，所以会从 12 直接变成 13，把重汇聚点跳过去，导致另一半线程永远没有被切回来执行。

我做了两个改动：

在 src/ibuffer.py (line 1) 增加了 peek()，用于查看”下一条真正准备 issue 的指令”。
在 src/simt_core.py (line 116) 把重汇聚判断改成优先看 ibuffer.peek().pc，只有 buffer 为空时才退回 warp.pc。
修完后我用这个命令跑通了：

python src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4

现在另一半线程已经会执行，最终结果也对上了：

mem[100]=2, mem[102]=2：偶数线程路径
mem[101]=1, mem[103]=1：奇数线程路径
mem[201]=2, mem[202]=4, mem[203]=6：重汇聚后统一执行尾部代码
补充一点：自动验证时还撞到了一个独立的 Windows 控制台编码问题，输出 ✓ 时可能报 UnicodeEncodeError。我这次是用 PYTHONIOENCODING=utf-8 绕过去的；如果你愿意，我可以下一步把这个控制台输出也顺手改成在 GBK 终端下更稳的版本。
