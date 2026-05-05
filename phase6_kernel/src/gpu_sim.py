"""
GPUSim — GPU 模拟器顶层 + Kernel Launch + Performance Monitor
===============================================================
对标 GPGPU-Sim 中 gpgpu_sim 顶层类的 kernel launch 和性能统计功能。

Phase 6: 整合 Phase 0-5 的所有模块，提供 CUDA 风格的 grid/block launch。

参考: GPGPU-Sim gpgpu-sim/gpu-sim.cc 中 gpgpu_sim::launch()
"""

from typing import List
from simt_core import SIMTCore


class PerfCounters:
    """性能计数器

    对应 GPGPU-Sim 的统计模块 (stat-tool)。
    """
    def __init__(self):
        self.total_cycles = 0
        self.total_instructions = 0
        self.stall_scoreboard = 0
        self.stall_barrier = 0
        self.stall_branch = 0
        self.active_cycles = 0

    @property
    def ipc(self) -> float:
        return self.total_instructions / self.total_cycles if self.total_cycles > 0 else 0.0

    def report(self) -> str:
        c = self.total_cycles
        if c == 0: return "No cycles executed"
        return (
            f"Performance Report:\n"
            f"  Total cycles:      {c}\n"
            f"  Total instructions: {self.total_instructions}\n"
            f"  IPC:               {self.ipc:.3f}\n"
            f"  Active cycles:     {self.active_cycles} ({self.active_cycles/c*100:.1f}%)\n"
            f"  Stalls:\n"
            f"    Scoreboard:      {self.stall_scoreboard} ({self.stall_scoreboard/c*100:.1f}%)\n"
            f"    Barrier:         {self.stall_barrier} ({self.stall_barrier/c*100:.1f}%)\n"
        )


class GPUSim:
    """GPU 模拟器顶层

    对应 GPGPU-Sim 的 gpgpu_sim 类。

    Attributes:
        cores: SIMT 核心列表 (每个 SM 一个)
        perf: 性能计数器
    """

    def __init__(self, num_sms: int = 1, warp_size: int = 8,
                 memory_size: int = 1024):
        self.num_sms = num_sms
        self.warp_size = warp_size
        self.memory_size = memory_size
        self.cores: List[SIMTCore] = []
        self.perf = PerfCounters()

    def launch_kernel(self, program: list[int], grid_dim: tuple = (1,),
                      block_dim: tuple = (8,)):
        """启动 kernel（对标 CUDA kernel launch）

        GPGPU-Sim 对应: gpgpu_sim::launch() 创建 grid/block 结构。
        Phase 6: 顺序执行所有 blocks（简化，不模拟并行 SM）。

        Args:
            program: 机器码列表
            grid_dim: grid 维度，如 (2,) 表示 2 个 block
            block_dim: block 维度，如 (8,) 表示 8 个线程
        """
        total_blocks = 1
        for d in grid_dim:
            total_blocks *= d
        total_threads = 1
        for d in block_dim:
            total_threads *= d

        num_warps_per_block = max(1, total_threads // self.warp_size)
        self.cores = []

        for block_id in range(total_blocks):
            core = SIMTCore(
                warp_size=self.warp_size,
                num_warps=num_warps_per_block,
                memory_size=self.memory_size
            )
            core.load_program(program)
            self.cores.append(core)

        print(f"Kernel launched: {total_blocks} block(s) x "
              f"{num_warps_per_block} warp(s) x "
              f"{self.warp_size} threads/warp = "
              f"{total_blocks * num_warps_per_block * self.warp_size} threads")

    def run(self):
        """运行所有 core 直到完成，收集性能数据"""
        self.perf = PerfCounters()
        for core in self.cores:
            while True:
                self.perf.total_cycles += 1
                # Check if any warp is stalled
                any_stalled = False
                for w in core.warps:
                    w.scoreboard.advance()
                    if w.scoreboard.stalled:
                        w.scoreboard_stalled = False
                    else:
                        w.scoreboard_stalled = w.scoreboard.stalled
                        if w.scoreboard_stalled:
                            self.perf.stall_scoreboard += 1
                            any_stalled = True
                    if w.at_barrier:
                        self.perf.stall_barrier += 1
                        any_stalled = True

                warp = core.scheduler.select_warp()
                if warp is None:
                    if not core.scheduler.has_active_warps():
                        break
                    continue

                if not any_stalled:
                    self.perf.active_cycles += 1

                # Reconvergence check
                if warp.simt_stack.at_reconvergence(warp.pc):
                    core._handle_reconvergence(warp)

                if warp.pc >= len(core.program):
                    warp.done = True
                    continue

                # Issue: fetch + decode + execute
                raw_word = core.program[warp.pc]
                instr = decode_wrapper(raw_word)

                sb = warp.scoreboard
                if sb.check_waw(instr.rd) or sb.check_raw(instr.rs1, instr.rs2):
                    warp.scoreboard_stalled = True
                    self.perf.stall_scoreboard += 1
                    continue

                old_pc = warp.pc
                warp.pc += 1
                core._execute_warp(warp, instr, old_pc)
                self.perf.total_instructions += 1

                latency = core._get_latency(instr.opcode)
                sb.reserve(instr.rd, latency)

    def report(self):
        """打印性能报告"""
        print(self.perf.report())
        for i, core in enumerate(self.cores):
            print(f"\nBlock {i}:")
            print(f"  {core.l1_cache.stats()}")
            if core.total_mem_reqs > 0:
                eff = core.coalesce_count / core.total_mem_reqs * 100
                print(f"  Coalescing: {core.coalesce_count}/{core.total_mem_reqs} ({eff:.0f}%)")


def decode_wrapper(word):
    from isa import decode
    return decode(word)
