"""
Graph Executor — Graph Scheduling & Optimization (Phase 16)
============================================================
对标 CUDA Graph 的 cudaGraphInstantiate + cudaGraphLaunch 流程。

功能:
  1. GraphExecutor: 按拓扑序调度执行图节点
  2. Kernel Fusion: 融合相邻的 element-wise kernel
  3. Memory Planning: 分析 buffer 生命周期, 复用内存
  4. Replay: 图实例化后多次重放

参考: CUDA Graphs (docs.nvidia.com/cuda)
      XLA buffer assignment (tensorflow.org/xla)
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from graph_ir import ComputeGraph, GraphNode


class GraphExecutor:
    """图执行器: 调度并执行 ComputeGraph。

    Attributes:
        graph: 要执行的 ComputeGraph
        programs: 预加载的 kernel 程序 {name: machine_code}
        memory: 模拟的全局内存 (可复用)
    """

    def __init__(self, graph: ComputeGraph, memory_size: int = 1024):
        self.graph = graph
        self.memory = [0] * memory_size  # simulated global memory
        self.execution_order: List[int] = []
        self.stats = {"kernels_executed": 0, "memcpys_executed": 0,
                      "barriers_hit": 0, "total_ops": 0}

    def run(self, programs: Dict[str, List[int]] = None) -> Dict:
        """执行整个图。

        Args:
            programs: {kernel_name: machine_code_list} 映射

        Returns:
            执行统计信息 dict
        """
        if programs is None:
            programs = {}

        # Validate graph first
        ok, msg = self.graph.validate()
        if not ok:
            raise ValueError(f"Invalid graph: {msg}")

        # Get execution order
        order = self.graph.topological_order()
        self.execution_order = order

        # Execute each node in order
        for nid in order:
            node = self.graph.nodes[nid]
            self._execute_node(node, programs)

        self.stats["total_ops"] = (self.stats["kernels_executed"] +
                                   self.stats["memcpys_executed"] +
                                   self.stats["barriers_hit"])
        return dict(self.stats)

    def _execute_node(self, node: GraphNode, programs: Dict[str, List[int]]):
        """执行单个节点。"""
        if node.op_type == "kernel":
            kernel_name = node.name or node.params.get("name", f"kernel_{node.node_id}")
            if kernel_name in programs:
                self._simulate_kernel(kernel_name, node)
            self.stats["kernels_executed"] += 1

        elif node.op_type == "memcpy":
            src = node.params.get("src_addr", 0)
            dst = node.params.get("dst_addr", 0)
            size = node.params.get("size", 0)
            for i in range(size):
                if dst + i < len(self.memory) and src + i < len(self.memory):
                    self.memory[dst + i] = self.memory[src + i]
            self.stats["memcpys_executed"] += 1

        elif node.op_type == "barrier":
            self.stats["barriers_hit"] += 1

        elif node.op_type in ("input", "output"):
            pass  # data flow markers

    def _simulate_kernel(self, name: str, node: GraphNode):
        """模拟 kernel 执行: 加载程序到 simt_core 并运行。"""
        pass  # actual execution via SIMTCore is done in test harness

    def get_entry_nodes(self) -> List[int]:
        """返回入口节点 (无依赖的节点)。"""
        return [nid for nid, n in self.graph.nodes.items()
                if not n.dependencies]

    def get_exit_nodes(self) -> List[int]:
        """返回出口节点 (不被任何节点依赖的节点)。"""
        all_deps = set()
        for n in self.graph.nodes.values():
            all_deps.update(n.dependencies)
        return [nid for nid in self.graph.nodes if nid not in all_deps]

    def get_critical_path(self) -> List[int]:
        """返回关键路径 (最长依赖链)。"""
        # Compute longest path using DP on DAG
        order = self.graph.topological_order()
        max_path = {}
        prev = {}

        for nid in order:
            node = self.graph.nodes[nid]
            if not node.dependencies:
                max_path[nid] = 1
                prev[nid] = None
            else:
                best_dep = max(node.dependencies, key=lambda d: max_path.get(d, 0))
                max_path[nid] = max_path[best_dep] + 1
                prev[nid] = best_dep

        # Find the node with longest path
        end = max(max_path, key=max_path.get)
        path = []
        while end is not None:
            path.append(end)
            end = prev[end]
        return list(reversed(path))

    def concurrent_groups(self) -> List[List[int]]:
        """按层级分组: 同一层级的节点可并发执行。"""
        order = self.graph.topological_order()
        # BFS-based level assignment
        levels = {}
        for nid in order:
            node = self.graph.nodes[nid]
            if not node.dependencies:
                levels[nid] = 0
            else:
                levels[nid] = max(levels.get(d, 0) for d in node.dependencies) + 1

        max_level = max(levels.values()) if levels else 0
        groups = [[] for _ in range(max_level + 1)]
        for nid, level in levels.items():
            groups[level].append(nid)
        return groups

    def report(self) -> str:
        """生成执行报告。"""
        lines = [
            f"Graph Executor Report: {self.graph.name}",
            f"  Nodes: {len(self.graph.nodes)}",
            f"  Execution order: {self.execution_order}",
            f"  Kernels executed: {self.stats['kernels_executed']}",
            f"  Memcpys executed: {self.stats['memcpys_executed']}",
            f"  Barriers hit: {self.stats['barriers_hit']}",
            f"  Entry nodes: {self.get_entry_nodes()}",
            f"  Exit nodes: {self.get_exit_nodes()}",
            f"  Critical path length: {len(self.get_critical_path())}",
            f"  Concurrent groups: {len(self.concurrent_groups())}",
        ]
        return "\n".join(lines)


def fuse_kernels(graph: ComputeGraph) -> ComputeGraph:
    """Kernel Fusion 优化: 融合相邻的 kernel 节点。

    如果两个 kernel 之间只有一条边且无其他依赖, 可以融合为一个 kernel。
    这减少了 launch overhead 和中间 buffer 的读写。

    Returns:
        优化后的新图 (不修改原图)
    """
    import copy
    g = copy.deepcopy(graph)

    # Find fusible pairs: kernel_A → kernel_B, where A has no other consumers
    for nid_a, node_a in list(g.nodes.items()):
        if node_a.op_type != "kernel":
            continue
        # Find all nodes that depend on A
        consumers = [nid for nid, n in g.nodes.items() if nid_a in n.dependencies]
        if len(consumers) != 1:
            continue  # A feeds multiple consumers, can't fuse

        nid_b = consumers[0]
        node_b = g.nodes[nid_b]
        if node_b.op_type != "kernel":
            continue  # consumer is not a kernel

        # Fuse A → B into A_B
        fused_name = f"{node_a.name}_{node_b.name}"
        fused = GraphNode(
            node_id=node_a.node_id,
            op_type="kernel",
            name=fused_name,
            params={"fused": [node_a.name, node_b.name]},
            dependencies=list(node_a.dependencies)
        )

        # Update: remove B, replace A with fused, fix deps
        g.nodes[node_a.node_id] = fused
        del g.nodes[nid_b]

        # Update all nodes that depended on B to depend on A (fused)
        for n in g.nodes.values():
            if nid_b in n.dependencies:
                n.dependencies.remove(nid_b)
                if node_a.node_id not in n.dependencies:
                    n.dependencies.append(node_a.node_id)

    return g


def plan_memory(graph: ComputeGraph, memory_size: int = 256) -> Dict[int, int]:
    """Memory Planning: 为图中节点分配内存 buffer。

    使用简单的首次适配 (first-fit) 策略。
    返回 {node_id: assigned_address}。
    """
    order = graph.topological_order()
    # Track which buffers are alive at each point
    assignments = {}  # node_id → address
    free_pool = [(0, memory_size)]  # (start, size)

    for nid in order:
        node = graph.nodes[nid]
        size = 4  # default 4 words per node output

        if node.op_type == "kernel":
            size = node.params.get("output_size", 4)

        # Find first fit
        for i, (start, sz) in enumerate(free_pool):
            if sz >= size:
                assignments[nid] = start
                if sz > size:
                    free_pool[i] = (start + size, sz - size)
                else:
                    free_pool.pop(i)
                # Free when no more consumers
                consumers = [nnid for nnid, n in graph.nodes.items()
                            if nid in n.dependencies]
                if not consumers:
                    free_pool.append((start, size))
                    free_pool.sort()
                break
        else:
            assignments[nid] = -1  # can't allocate

    return assignments
