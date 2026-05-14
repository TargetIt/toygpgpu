
# 20260514

## Q10：PC10 执行完后，怎么知道要从 stack 找 PC5 执行？

PC00 -> PC01 -> PC02 -> PC03 -> PC04
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              偶数线程跳转                  奇数线程顺序执行
                    │                           │
                    v                           v
              PC08 -> PC09 -> PC10        PC05 -> PC06 -> PC07
                    │                           │
                    └─────────────┬─────────────┘
                                  v
                           PC11 -> PC12 -> PC13

不是 PC10 直接找 PC5，而是分两步：

PC10 JMP 到 PC11
下一轮 step 发现 PC11 是栈顶 reconv_pc
于是触发 _handle_reconvergence()
PC10 执行后：

warp.pc = 11
stack.top.reconv_pc = 11
下一轮 step() 开头检查：

if warp.simt_stack.at_reconvergence(warp.pc):
    self._handle_reconvergence(warp)
匹配成功，于是从 stack entry 里取出：

fallthrough_pc = 5
remaining_mask = 10101010
然后切换到：

PC05
active_mask = 10101010
也就是让奇数线程继续执行未执行路径。

## SIMT stack entry 里几个字段怎么记？
reconv_pc
  重汇聚点。执行到这里时，要检查是否还有另一条路径没跑。

orig_mask
  发散前的完整活跃线程集合。

taken_mask
  已经执行过的线程集合。

fallthrough_pc
  没跳转那条路径的入口，也就是之后要补跑的 PC。


# 20260507 

## warp是怎么创建的？ 
我看了后面的几个 phase，结论是：

**主流创建方式基本没变，仍然是“先确定数量，再批量创建 warp”**，没有看到“运行过程中动态新建一个 warp”这种做法。

可以分成两段看。

**Phase 3 到 Phase 5**
还是和你现在 `phase2` 很像，`SIMTCore` 初始化时直接建好 warps：

- [phase3_simt_stack/src/simt_core.py](/d:/lab/toygpgpu-main/toygpgpu-main/phase3_simt_stack/src/simt_core.py:59)
- [phase4_scoreboard/src/simt_core.py](/d:/lab/toygpgpu-main/toygpgpu-main/phase4_scoreboard/src/simt_core.py:61)
- [phase5_memory/src/simt_core.py](/d:/lab/toygpgpu-main/toygpgpu-main/phase5_memory/src/simt_core.py:67)

核心模式都是这一类：

```python
self.warps = [Warp(wid, warp_size) for wid in range(num_warps)]
```

也就是：
- `num_warps` 先给定
- `Warp(0)`, `Warp(1)`, ... 一次性创建
- `load_program()` 时把同一份程序装进去并复位状态

**Phase 6 以后**
多了一层更接近 CUDA 的 `kernel launch`，但 warp 依然不是动态生出来的，而是**按 block 计算数量后批量创建**。

关键在 [phase6_kernel/src/gpu_sim.py](/d:/lab/toygpgpu-main/toygpgpu-main/phase6_kernel/src/gpu_sim.py:58)：

- `launch_kernel(program, grid_dim, block_dim)`
- 先算总线程数
- 再算 `num_warps_per_block`
- 然后为每个 block 创建一个 `SIMTCore`

对应代码在：

- [phase6_kernel/src/gpu_sim.py](/d:/lab/toygpgpu-main/toygpu-main/phase6_kernel/src/gpu_sim.py:84)
- [phase6_kernel/src/gpu_sim.py](/d:/lab/toygpgpu-main/toygpu-main/phase6_kernel/src/gpu_sim.py:87)

里面是：

```python
num_warps_per_block = max(1, total_threads // self.warp_size)

for block_id in range(total_blocks):
    core = SIMTCore(
        warp_size=self.warp_size,
        num_warps=num_warps_per_block,
        memory_size=self.memory_size
    )
```

而 `SIMTCore` 里面仍然还是：

- [phase6_kernel/src/simt_core.py](/d:/lab/toygpu-main/toygpu-main/phase6_kernel/src/simt_core.py:67)

```python
self.warps = [Warp(wid, warp_size) for wid in range(num_warps)]
```

后面的 `phase7_pipeline`、`phase8_ptx`、`phase9_tensor`、`phase10_viz` 也延续这个模式。

所以更准确地总结是：

- `phase2-5`：`SIMTCore` 直接按 `num_warps` 预创建
- `phase6+`：先通过 `grid/block` 计算出每个 block 该有多少 warp，再由每个 `SIMTCore` 预创建
- 我没看到“执行时按需 spawn 一个新 warp”的实现

如果你愿意，我可以下一步把这个项目里“`thread / warp / block / core` 是怎么一层层创建出来的”画成一张小结构图。
## 这里的barrier_count的应用场景是？

我先顺着 warp 和 BAR 的实现看一下 barrier_count 现在有没有真正参与逻辑，再给你讲它在更真实的 SIMT 模型里应该用在什么场景。

barrier_count 在你当前这版里，还没有实际用上。我刚查了一下，Warp 里只是定义和重置了它 warp.py (line 66) warp.py (line 91)，但 simt_core.py 和 scheduler.py 里没有谁根据它做判断；当前调度只看 warp.at_barrier scheduler.py (line 49)。

它真正的应用场景通常是：统计“有多少线程已经到达 barrier”。典型有两类。

第一类是你以后做更真实的 SIMT 发散时会用到。现在你这版一个 warp 共享 PC，所以同一个 warp 里的 active threads 会一起执行到 BAR，这时 barrier_count 基本没意义，因为大家是“整 warp 同步到达”的。但如果后面加了 branch divergence / SIMT stack，不同 lane 可能分批次到达同一个 barrier，这时就要靠 barrier_count 判断：

到达 barrier 的线程数还没凑齐，warp 继续处于 wait 状态
当 barrier_count == 参与同步的线程数 时，才解除 barrier
第二类是做 block/CTA 级 barrier 时会用到。真实 GPU 里的 __syncthreads() 本质上不是“warp 内 barrier”，而是“同一个 thread block 里的所有线程都到齐才放行”。这时常见做法是：

每个 warp 记录自己有多少线程到达了 barrier
再由 block 级状态汇总所有 warp 的到达数
总数等于 block 内应参与的线程数后，统一 release
所以一句话总结：

在当前 phase2 里，barrier_count 更像是给后续演化预留的字段；只有 at_barrier 真正在起作用。

如果你愿意，我可以下一步直接帮你判断一件事：这个字段在 phase2 里应该先删掉保持模型干净，还是保留并补注释说明“为 Phase 3 的 divergence / CTA barrier 预留”。



## 真实的芯片场景是抓一条指令，遍历不同的warp么，还是抓一批指令遍历不同的warp，还是说抓一个warp 然后抓对应的指令？



我先看一下这段 SIMT core 的调度语义，结合你当前的实现来对照“真实芯片”里到底是谁在驱动谁：是先看 warp、还是先看 instruction。

Ran Get-Content -Path 'd:\lab\toygpgpu-main\toygpgpu-main\phase2_simt\src\simt_core.py' | Select-Object -First 220 for 1s


Ran Get-Content -Path 'd:\lab\toygpgpu-main\toygpgpu-main\phase2_simt\src\register_file.py' | Select-Object -First 220 for 1s


Worked for 20s
更接近真实芯片的思路是：

先选一个“就绪的 warp”，再取这个 warp 当前 PC 对应的那一条指令，然后把这条指令广播给这个 warp 里所有 active lanes 执行。

也就是说，主视角通常是 warp-centric，不是“先抓一批指令再到处套 warp”。

结合你这段实现 simt_core.py (line 83) 来看，流程其实很像真实 GPU 的简化版：

scheduler.select_warp() 先挑一个可发射的 warp
用这个 warp 的 warp.pc 去取指
decode(raw_word) 解码这一条指令
_execute_warp(warp, instr) 对这个 warp 的 active threads 宽执行
warp.pc += 1
所以你现在这版不是“抓一条指令遍历不同 warp”，而是：

抓一个 warp，然后抓它对应的指令。

补一个更贴近真实硬件的细节：

单个 SM 里通常有多个 warp scheduler / issue slot
所以“一个周期”里可能会同时从多个 warp 各发一条指令
但粒度仍然是“每次 issue 是某个 warp 的当前指令”，不是“同一条指令拿去扫所有 warp”
你可以把它想成：

CPU 更像“线程 -> 当前 PC -> 发一条指令”
GPU 是“warp -> 当前 PC -> 发一条 warp instruction”
只是 GPU 会在很多 warp 之间切换，靠这种切换去隐藏延迟。

如果你愿意，我下一步可以继续帮你把这段 step() 标成更准确的“教学版 GPU 调度语义”，顺手说明它和真实 GPGPU-Sim / NVIDIA SM 的差别。


