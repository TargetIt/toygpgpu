
20260507
可能多个phase都存在这个问题

定位到了，根因在 src/simt_core.py (line 116)。

分歧之后，SIMT stack 里记录的 reconv_pc 是“下一条应该切换/重汇聚的指令地址”。但原来的代码用 warp.pc 去检查是否到达重汇聚点，而这个 warp.pc 在 fetch 阶段已经自增过了，所以会从 12 直接变成 13，把重汇聚点跳过去，导致另一半线程永远没有被切回来执行。

我做了两个改动：

在 src/ibuffer.py (line 1) 增加了 peek()，用于查看“下一条真正准备 issue 的指令”。
在 src/simt_core.py (line 116) 把重汇聚判断改成优先看 ibuffer.peek().pc，只有 buffer 为空时才退回 warp.pc。
修完后我用这个命令跑通了：

python src/learning_console.py tests/programs/demo_divergence.asm --warp-size 4

现在另一半线程已经会执行，最终结果也对上了：

mem[100]=2, mem[102]=2：偶数线程路径
mem[101]=1, mem[103]=1：奇数线程路径
mem[201]=2, mem[202]=4, mem[203]=6：重汇聚后统一执行尾部代码
补充一点：自动验证时还撞到了一个独立的 Windows 控制台编码问题，输出 ✓ 时可能报 UnicodeEncodeError。我这次是用 PYTHONIOENCODING=utf-8 绕过去的；如果你愿意，我可以下一步把这个控制台输出也顺手改成在 GBK 终端下更稳的版本。
