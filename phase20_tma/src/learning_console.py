#!/usr/bin/env python3
"""
Learning Console — 交互式 GPU 流水线学习控制台
================================================
对标 GDB stepi 模式，逐周期观察 GPU 内部状态。

使用方法:
  python3 learning_console.py <program.asm> [options]

选项:
  --warp-size N      warp 大小 (默认 4)
  --num-warps N      warp 数量 (默认 1)
  --auto-interval N  自动执行间隔 (秒, 默认 0=手动)
  --max-cycles N     最大周期数 (默认 500)

交互命令:
  Enter / s  单步执行一个周期
  r          运行到 HALT
  r N        运行 N 个周期
  i          打印当前完整状态
  m          打印内存非零区域
  reg        打印所有寄存器
  sb         打印 scoreboard
  ib         打印 I-Buffer
  stack      打印 SIMT Stack
  shfl       打印 warp shuffle 拓扑 (Phase 12)
  vote       打印每线程谓词/寄存器 (Phase 12)
  pred       打印每线程谓词位 (Phase 12)
  tile       打印 tile 配置 (Phase 13)
  smem       打印 shared memory 内容 (Phase 13)
  cutile     打印 CuTile 模型信息 (Phase 14)
  graph      打印 Compute Graph IR 信息 (Phase 15)
  sched       打印 Graph Scheduler 信息 (Phase 16)
  tma         打印 TMA 引擎统计 (Phase 20)
  b <pc>     在指定 PC 设置断点
  b list     列出所有断点
  b clear    清除所有断点
  q          退出
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simt_core import SIMTCore
from assembler import assemble
from isa import OPCODE_NAMES, decode
from console_display import (
    render_cycle, snapshot_regs, snapshot_mem, mem_diff, c
)


def run_console(simt, program_text, args):
    """启动交互式控制台"""
    warp_size = args.get('warp_size', 4)
    num_warps = args.get('num_warps', 1)
    auto_interval = args.get('auto_interval', 0)
    max_cycles = args.get('max_cycles', 500)
    breakpoints = set()

    # 加载程序
    prog = assemble(program_text)
    simt.load_program(prog)

    print(c("╔══════════════════════════════════════════════════╗", 'cyan'))
    print(c("║     toygpgpu Learning Console — 交互式调试器      ║", 'cyan'))
    print(c("╠══════════════════════════════════════════════════╣", 'cyan'))
    print(f"║  Program: {len(prog)} instructions                     ║")
    print(f"║  Config:  {num_warps} warp(s) × {warp_size} threads/warp          ║")
    print(f"║  Commands: Enter=step, r=run, i=info, q=quit      ║")
    print(c("╚══════════════════════════════════════════════════╝", 'cyan'))
    print()

    # 打印初始程序
    print(c("─── Program ───", 'bold'))
    for pc, word in enumerate(prog):
        instr = decode(word)
        opname = OPCODE_NAMES.get(instr.opcode, '?')
        print(f"  PC {pc:2d}: {c(opname, 'yellow'):6s} "
              f"rd=r{instr.rd} rs1=r{instr.rs1} rs2=r{instr.rs2} imm={instr.imm}")
    print()

    cycle = 0
    prev_regs = snapshot_regs(simt)
    prev_mem = snapshot_mem(simt)
    auto_step = False

    while cycle < max_cycles:
        # 自动模式
        if auto_interval > 0 or auto_step:
            time.sleep(auto_interval)
            auto_step = True if auto_interval > 0 else auto_step
        else:
            # 等待用户输入
            try:
                cmd = input(c(f"[{cycle}] > ", 'green')).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            # 解析命令
            if cmd == '' or cmd.lower() == 's':
                pass  # 单步
            elif cmd.lower() == 'q':
                break
            elif cmd.lower() == 'r':
                auto_step = True
            elif cmd.lower().startswith('r '):
                n = int(cmd.split()[1])
                for _ in range(n):
                    if not _do_step(simt, cycle, prev_regs, prev_mem, breakpoints):
                        break
                    cycle += 1
                prev_regs = snapshot_regs(simt)
                prev_mem = snapshot_mem(simt)
                continue
            elif cmd.lower() == 'i':
                print_state(simt)
                continue
            elif cmd.lower() == 'm':
                print_memory(simt)
                continue
            elif cmd.lower() == 'reg':
                print_registers(simt)
                continue
            elif cmd.lower() == 'wreg':
                print_warp_regs(simt)
                continue
            elif cmd.lower() == 'sb':
                print_scoreboard(simt)
                continue
            elif cmd.lower() == 'ib':
                print_ibuffer(simt)
                continue
            elif cmd.lower() == 'stack':
                print_simt_stack(simt)
                continue
            elif cmd.lower() == 'shfl':
                print_shfl_info(simt)
                continue
            elif cmd.lower() == 'vote':
                print_vote_info(simt)
                continue
            elif cmd.lower() == 'pred':
                print_predicates(simt)
                continue
            elif cmd.lower() == 'tile':
                print_tile_config(simt)
                continue
            elif cmd.lower() == 'smem':
                print_shared_mem(simt)
                continue
            elif cmd.lower() == 'cutile':
                print_cutile_info(simt)
                continue
            elif cmd.lower() == 'graph':
                print_graph_info()
                continue
            elif cmd.lower() == 'sched':
                print_sched_info()
                continue
            elif cmd.lower() == 'tma':
                print_tma(simt)
                continue
            elif cmd.lower().startswith('b '):
                sub = cmd[2:].strip()
                if sub == 'list':
                    print(f"Breakpoints: {sorted(breakpoints) if breakpoints else 'none'}")
                elif sub == 'clear':
                    breakpoints.clear()
                    print("Breakpoints cleared.")
                else:
                    try:
                        breakpoints.add(int(sub))
                        print(f"Breakpoint set at PC={sub}")
                    except ValueError:
                        print(f"Invalid PC: {sub}")
                continue

        # 执行一个周期
        bp_hit = _do_step(simt, cycle, prev_regs, prev_mem, breakpoints)

        prev_regs = snapshot_regs(simt)
        prev_mem = snapshot_mem(simt)
        cycle += 1

        if bp_hit:
            auto_step = False
            print(c(f"  ● Breakpoint hit", 'red'))

        # 检查是否完成
        if not simt.scheduler.has_active_warps():
            print(c(f"\n  ✓ All warps completed at cycle {cycle}", 'green'))
            break

        # 尾页暂停
        if auto_step and not bp_hit and cycle % 5 == 0:
            # 每 5 个周期暂停一下（方便阅读）
            pass

    # 最终状态
    print(f"\n{c('─── Final State ───', 'bold')}")
    print(f"Cycles executed: {cycle}")
    print(f"Memory (non-zero):")
    print_memory(simt)
    print(f"\n{c('L1Cache', 'cyan')}: {simt.l1_cache.stats()}")
    print(f"{c('OpCollector', 'cyan')}: {simt.op_collector.stats()}")
    tma_stats = simt.tma_engine.stats()
    print(f"{c('TMAEngine', 'cyan')}: loads={tma_stats['loads']} "
          f"stores={tma_stats['stores']} boundary={tma_stats['boundary_events']}")


def _do_step(simt, cycle, prev_regs, prev_mem, breakpoints):
    """执行一个周期并渲染显示

    Returns:
        True if breakpoint hit
    """
    from isa import decode, OPCODE_NAMES

    # 记录执行前的状态
    old_regs = snapshot_regs(simt)
    old_mem = snapshot_mem(simt)

    # 收集流水线信息
    warp = None
    instr = None
    stage_info = {'fetch': '', 'decode': '', 'issue': '', 'exec': '', 'wb': ''}

    # 尝试获取即将执行的指令
    for w in simt.warps:
        if not w.done and not w.scoreboard_stalled and not w.at_barrier:
            for e in w.ibuffer.entries:
                if e.valid and e.ready:
                    instr = decode(e.instruction_word)
                    warp = w
                    opname = OPCODE_NAMES.get(instr.opcode, '?')
                    stage_info['fetch'] = f"PC={e.pc} → IBuffer"
                    stage_info['decode'] = f"{opname} decoded"
                    stage_info['issue'] = f"Scoreboard check, bank check"
                    stage_info['exec'] = f"{opname} r{instr.rd}=r{instr.rs1} op r{instr.rs2}"
                    stage_info['wb'] = f"r{instr.rd} ← result (latency)"
                    break
            if instr:
                break

    # 执行一步
    simt.step()

    # 计算变化
    new_regs = snapshot_regs(simt)
    new_mem = snapshot_mem(simt)
    mem_changes = mem_diff(old_mem, new_mem)

    # 格式化指令信息
    instr_info = None
    if instr:
        instr_info = {
            'op': OPCODE_NAMES.get(instr.opcode, '?'),
            'rd': instr.rd, 'rs1': instr.rs1, 'rs2': instr.rs2,
            'pc': warp.pc - 1 if warp else -1,
            'active': bin(warp.active_mask).count('1') if warp else 0,
            'imm': instr.imm,
        }

    # 渲染
    print(render_cycle(cycle, simt, instr_info, old_regs, mem_changes, stage_info))

    # 检查断点
    if warp and warp.pc - 1 in breakpoints:
        return True
    return False


# ─── 独立状态打印函数 ───

def print_state(simt):
    """打印当前状态摘要"""
    print(c("─── Current State ───", 'bold'))
    for w in simt.warps:
        status = 'DONE' if w.done else 'ACTIVE'
        print(f"W{w.warp_id}: PC={w.pc}, mask=0b{w.active_mask:0{simt.warp_size}b}, {status}")
        print(f"  Scoreboard: {w.scoreboard}")
        print(f"  I-Buffer: {w.ibuffer}")
        if not w.simt_stack.empty:
            print(f"  SIMT Stack depth: {len(w.simt_stack)}")


def print_memory(simt):
    """打印非零内存"""
    non_zero = []
    for i in range(min(256, simt.memory.size_words)):
        v = simt.memory.read_word(i)
        if v != 0:
            non_zero.append(f"mem[{i:3d}]=0x{v:08X}({v})")
    print('  ' + '\n  '.join(non_zero[:20] or ['(all zero)']))


def print_registers(simt):
    """打印所有寄存器"""
    for w in simt.warps:
        print(f"Warp {w.warp_id}:")
        for t in w.active_threads():
            vals = [(i, t.read_reg(i)) for i in range(16) if t.read_reg(i) != 0]
            if vals:
                reg_str = ' '.join(f"r{i}={v}" for i, v in vals)
                print(f"  T{t.thread_id}: {reg_str}")


def print_scoreboard(simt):
    for w in simt.warps:
        print(f"W{w.warp_id} Scoreboard: {w.scoreboard}")


def print_ibuffer(simt):
    for w in simt.warps:
        print(f"W{w.warp_id} I-Buffer: {w.ibuffer}")


def print_warp_regs(simt):
    """Print warp-level uniform registers"""
    from isa import WREG_NAMES
    rev = {v: k for k, v in WREG_NAMES.items()}
    print(c("--- Warp Registers ---", 'bold'))
    for w in simt.warps:
        wregs = []
        for idx, val in sorted(w.warp_regs.items()):
            name = rev.get(idx, 'wreg%d' % idx)
            wregs.append("%s=%d" % (name, val))
        print("  Warp %d: %s" % (w.warp_id, ', '.join(wregs)))


def print_simt_stack(simt):
    for w in simt.warps:
        print(f"W{w.warp_id} SIMT Stack ({len(w.simt_stack)}):")
        for e in w.simt_stack.entries:
            print(f"  reconv={e.reconv_pc} orig=0b{e.orig_mask:08b} taken=0b{e.taken_mask:08b}")


def print_shfl_info(simt):
    """Print warp shuffle topology (Phase 12)"""
    print(c("--- Warp Shuffle Info (Phase 12) ---", 'bold'))
    for w in simt.warps:
        if w.done:
            continue
        warp_size = w.warp_size
        print(f"Warp {w.warp_id} ({warp_size} threads):")
        print(f"  Active mask: 0b{w.active_mask:0{warp_size}b}")
        print(f"  Modes: 0=IDX(src_lane), 1=UP(delta), 2=DOWN(delta), 3=XOR(mask)")
        print(f"  Example SHFL modes for each thread:")
        for tid in range(warp_size):
            if w.is_active(tid):
                markers = []
                if tid == 0:
                    markers.append("lane=0")
                print(f"    T{tid}: IDX→0, UP(1)→{(tid-1)%warp_size}, DOWN(1)→{(tid+1)%warp_size}, XOR(1)→{tid^1}")


def print_vote_info(simt):
    """Print warp vote/bot state (Phase 12)"""
    print(c("--- Warp Vote Info (Phase 12) ---", 'bold'))
    for w in simt.warps:
        if w.done:
            continue
        print(f"Warp {w.warp_id}:")
        for t in w.active_threads():
            vals = [(i, t.read_reg(i)) for i in range(16) if t.read_reg(i) != 0]
            reg_str = ' '.join(f"r{i}={v}" for i, v in vals)
            pred_str = "pred=T" if t.pred else "pred=F"
            print(f"  T{t.thread_id}: {pred_str} | {reg_str if reg_str else 'all zero'}")


def print_predicates(simt):
    """Print per-thread predicate state (Phase 12)"""
    print(c("--- Predicate Registers ---", 'bold'))
    for w in simt.warps:
        print(f"Warp {w.warp_id}:")
        for t in w.threads:
            status = "ACTIVE" if w.is_active(t.thread_id) else "inactive"
            pred_str = "pred=TRUE" if t.pred else "pred=false"
            print(f"  T{t.thread_id}: {pred_str} ({status})")


def print_tile_config(simt):
    """Print tile configuration (Phase 13)"""
    print(c("--- Tile Configuration (Phase 13) ---", 'bold'))
    print(f"  Tile shape: M={simt.tile_m}, N={simt.tile_n}, K={simt.tile_k}")
    print(f"  Tile A size: {simt.tile_m} x {simt.tile_k} = {simt.tile_m * simt.tile_k} elements")
    print(f"  Tile B size: {simt.tile_k} x {simt.tile_n} = {simt.tile_k * simt.tile_n} elements")
    print(f"  Tile C size: {simt.tile_m} x {simt.tile_n} = {simt.tile_m * simt.tile_n} elements")


def print_shared_mem(simt):
    """Print shared memory non-zero contents (Phase 13)"""
    print(c("--- Shared Memory (non-zero) ---", 'bold'))
    smem = simt.thread_block.shared_memory
    non_zero = []
    for i in range(min(256, smem.size_words)):
        v = smem.read_word(i)
        if v != 0:
            non_zero.append(f"smem[{i:3d}]=0x{v:08X}({v})")
    print('  ' + '\n  '.join(non_zero[:20] or ['(all zero)']))


def print_cutile_info(simt):
    """Print CuTile model info (Phase 14)"""
    print(c("--- CuTile Model (Phase 14) ---", 'bold'))
    print(f"  Tile shape: M={simt.tile_m}, N={simt.tile_n}, K={simt.tile_k}")
    print(f"  CuTile DSL: tile M={simt.tile_m}, N={simt.tile_n}, K={simt.tile_k}")
    print(f"  kernel name(params) {{")
    print(f"      load A[0:{simt.tile_m}, 0:{simt.tile_k}] -> smem[a]")
    print(f"      load B[0:{simt.tile_k}, 0:{simt.tile_n}] -> smem[b]")
    print(f"      mma smem[a], smem[b] -> smem[c]  (uses WGMMA)")
    print(f"      store smem[c] -> C[0:{simt.tile_m}, 0:{simt.tile_n}]")
    print(f"  }}")
    print(f"  Generated ISA: TLCONF + TLDS + WGMMA + TLSTS")



def print_graph_info():
    """Print Compute Graph IR info (Phase 15)"""
    from graph_ir import build_example_graph, ComputeGraph

    print(c("--- Compute Graph IR (Phase 15) ---", 'bold'))

    # 示例图 / Example graph
    g = build_example_graph()
    valid, msg = g.validate()
    topo = g.topological_order()

    print(f"  Graph: '{g.name}', nodes={len(g.nodes)}")
    print(f"  Validate: {valid} ({msg})")
    print(f"  Topological order: {topo}")
    print()

    # 节点列表 / Node listing
    print(f"  Nodes:")
    for n in g.nodes.values():
        deps_str = f", deps={n.dependencies}" if n.dependencies else ", deps=[]"
        print(f"    [{n.node_id}] {n.op_type}:{n.name}{deps_str}")
    print()

    # DOT 输出 / DOT output
    print(f"  DOT representation:")
    for line in g.to_dot().split('\n'):
        print(f"    {line}")
    print()

    # JSON 输出 / JSON output
    print(f"  JSON export: {len(g.to_json())} chars")
    print()

    # 环检测演示 / Cycle detection demo
    print(f"  Cycle detection demo:")
    g2 = ComputeGraph("cyclic_demo")
    a = g2.add_kernel("A", block_dim=(4,))
    b = g2.add_kernel("B", block_dim=(4,), dependencies=[a])
    g2.nodes[a].dependencies.append(b)  # 引入环 / introduce cycle
    v2, m2 = g2.validate()
    print(f"    Cyclic graph: {v2} ({m2})")
    print(f"    Partial topo: {g2.topological_order()} (node 1 missing = cycle)")


def print_sched_info():
    """Print Graph Scheduling info (Phase 16)"""
    from graph_ir import ComputeGraph
    from graph_executor import GraphExecutor, fuse_kernels, plan_memory
    print("--- Graph Scheduling (Phase 16) ---")
    g = ComputeGraph("demo")
    a = g.add_kernel("A"); b = g.add_kernel("B", dependencies=[a])
    c = g.add_kernel("C", dependencies=[a]); g.add_kernel("D", dependencies=[b,c])
    exec = GraphExecutor(g)
    print(f"  Concurrent groups: {exec.concurrent_groups()}")
    print(f"  Critical path: {exec.get_critical_path()}")
    fused = fuse_kernels(g)
    print(f"  Fusion: {len(g.nodes)} -> {len(fused.nodes)} nodes")
    mem = plan_memory(g, 64)
    print(f"  Memory plan: {mem}")


def print_tma(simt):
    """Print TMA engine stats (Phase 20)"""
    from tma_engine import TMAEngine
    print(c("--- TMA Engine (Phase 20) ---", 'bold'))
    stats = simt.tma_engine.stats()
    print(f"  Loads issued: {stats.get('loads', 0)}")
    print(f"  Stores issued: {stats.get('stores', 0)}")
    print(f"  Boundary handling events: {stats.get('boundary_events', 0)}")
    print(f"  Tensor map base: {stats.get('tensor_map_base', 'N/A')}")
    print(f"  Active descriptors: {stats.get('active_descs', 0)}")


# ─── Main ───

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 learning_console.py <program.asm> [options]")
        print("Options: --warp-size N --num-warps N --auto-interval N --max-cycles N")
        print("         --graph-info    Print Compute Graph IR info and exit")
        sys.exit(1)

    # 处理 --graph-info 模式
    if sys.argv[1] == '--graph-info':
        print_graph_info()
        return

    asm_file = sys.argv[1]
    args = {
        'warp_size': 4, 'num_warps': 1,
        'auto_interval': 0, 'max_cycles': 500,
        'auto': False,
    }

    # 解析选项
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--warp-size':
            args['warp_size'] = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--num-warps':
            args['num_warps'] = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--auto-interval':
            args['auto_interval'] = float(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--max-cycles':
            args['max_cycles'] = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--auto':
            args['auto'] = True
            i += 1
        elif sys.argv[i] == '--trace':
            args['auto'] = True
            i += 1
        else:
            i += 1

    with open(asm_file, encoding='utf-8') as f:
        program_text = f.read()

    simt = SIMTCore(
        warp_size=args['warp_size'],
        num_warps=args['num_warps'],
        memory_size=1024
    )

    if args.get('auto'):
        prog = assemble(program_text)
        simt.load_program(prog)
        simt.run(trace=True)
        return

    run_console(simt, program_text, args)


if __name__ == '__main__':
    main()
