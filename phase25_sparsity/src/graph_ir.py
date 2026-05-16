"""
Compute Graph IR — DAG-based Operation Graph (Phase 15)
=========================================================
对标 CUDA Graphs (cudaGraph) 和 MLIR/XLA 的 HLO graph IR。

Graph IR 是一种有向无环图 (DAG)，节点表示计算操作
(kernel, memcpy, barrier)，边表示数据/执行依赖。
图可以被验证、优化、序列化和重放。

参考: CUDA Graphs API (docs.nvidia.com/cuda/cuda-c-programming-guide)
      XLA HLO (tensorflow.org/xla)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import json


@dataclass
class GraphNode:
    """计算图中的一个节点。

    Attributes:
        node_id: 唯一节点 ID
        op_type: "kernel" | "memcpy" | "barrier" | "input" | "output"
        name: 节点名称
        params: 节点参数 (如 kernel 名称, memcpy 的 src/dst 地址)
        dependencies: 前置依赖节点 ID 列表
    """
    node_id: int
    op_type: str
    name: str = ""
    params: Dict = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)

    def __repr__(self):
        return f"Node({self.node_id}, {self.op_type}, '{self.name}', deps={self.dependencies})"


class ComputeGraph:
    """计算图 DAG。

    Attributes:
        nodes: 节点列表 (按 ID 索引)
        name: 图名称
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.nodes: Dict[int, GraphNode] = {}
        self._next_id = 0

    def add_node(self, op_type: str, name: str = "", params: Dict = None,
                 dependencies: List[int] = None) -> int:
        """添加节点，返回 node_id。"""
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = GraphNode(
            node_id=nid, op_type=op_type, name=name,
            params=params or {}, dependencies=dependencies or []
        )
        return nid

    def add_kernel(self, name: str, program: List[int] = None,
                   grid_dim: Tuple[int] = (1,), block_dim: Tuple[int] = (1,),
                   dependencies: List[int] = None) -> int:
        """添加 kernel 节点。"""
        return self.add_node("kernel", name, {
            "program": program,
            "grid_dim": list(grid_dim),
            "block_dim": list(block_dim),
        }, dependencies)

    def add_memcpy(self, name: str, src_addr: int, dst_addr: int, size: int,
                   dependencies: List[int] = None) -> int:
        """添加 memcpy 节点 (device-to-device copy)。"""
        return self.add_node("memcpy", name, {
            "src_addr": src_addr, "dst_addr": dst_addr, "size": size,
        }, dependencies)

    def add_barrier(self, name: str = "", dependencies: List[int] = None) -> int:
        """添加 barrier 同步节点。"""
        return self.add_node("barrier", name, {}, dependencies)

    def add_input(self, name: str, addr: int, size: int) -> int:
        """添加输入节点 (图入口数据)。"""
        return self.add_node("input", name, {"addr": addr, "size": size})

    def add_output(self, name: str, addr: int, size: int, dependencies: List[int]) -> int:
        """添加输出节点 (图出口数据)。"""
        return self.add_node("output", name, {"addr": addr, "size": size}, dependencies)

    def validate(self) -> Tuple[bool, str]:
        """验证图的正确性。

        检查:
        1. 无环 (DAG)
        2. 所有依赖节点存在
        3. 至少有一个节点
        """
        if not self.nodes:
            return False, "Graph is empty"

        # 检查依赖节点存在
        for n in self.nodes.values():
            for dep in n.dependencies:
                if dep not in self.nodes:
                    return False, f"Node {n.node_id} depends on missing node {dep}"

        # 检测环路 (DFS)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes}

        def dfs(nid):
            color[nid] = GRAY
            for dep in self.nodes[nid].dependencies:
                if color[dep] == GRAY:
                    return False  # cycle detected
                if color[dep] == WHITE:
                    if not dfs(dep):
                        return False
            color[nid] = BLACK
            return True

        for nid in self.nodes:
            if color[nid] == WHITE:
                if not dfs(nid):
                    return False, f"Cycle detected at node {nid}"

        return True, "Valid DAG"

    def topological_order(self) -> List[int]:
        """返回拓扑排序的节点 ID 列表。"""
        in_degree = {nid: len(n.dependencies) for nid, n in self.nodes.items()}
        queue = [nid for nid, d in in_degree.items() if d == 0]
        order = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for other in self.nodes.values():
                if nid in other.dependencies:
                    in_degree[other.node_id] -= 1
                    if in_degree[other.node_id] == 0:
                        queue.append(other.node_id)

        return order

    def to_dict(self) -> Dict:
        """序列化为 dict (用于 JSON 导出)。"""
        nodes_list = []
        for n in self.nodes.values():
            nodes_list.append({
                "id": n.node_id,
                "type": n.op_type,
                "name": n.name,
                "params": {k: (v.hex() if isinstance(v, bytes) else
                               [x.hex() if isinstance(x, bytes) else x for x in v]
                               if isinstance(v, list) and v and isinstance(v[0], bytes)
                               else v)
                            for k, v in n.params.items()},
                "deps": n.dependencies,
            })
        return {"name": self.name, "nodes": nodes_list}

    def to_json(self) -> str:
        """导出为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=2)

    def to_dot(self) -> str:
        """导出为 DOT 格式 (Graphviz 可视化)。"""
        lines = [f'digraph "{self.name}" {{', '  rankdir=TB;']
        for n in self.nodes.values():
            label = f"{n.op_type}:{n.name}" if n.name else n.op_type
            shape = {"kernel": "box", "memcpy": "ellipse", "barrier": "diamond",
                     "input": "invhouse", "output": "house"}.get(n.op_type, "box")
            lines.append(f'  n{n.node_id} [label="{label}" shape={shape}];')
        for n in self.nodes.values():
            for dep in n.dependencies:
                lines.append(f'  n{dep} -> n{n.node_id};')
        lines.append('}')
        return '\n'.join(lines)

    @staticmethod
    def from_dict(data: Dict) -> 'ComputeGraph':
        """从 dict 反序列化。"""
        g = ComputeGraph(data["name"])
        for nd in data["nodes"]:
            g.nodes[nd["id"]] = GraphNode(
                node_id=nd["id"], op_type=nd["type"], name=nd["name"],
                params=nd.get("params", {}), dependencies=nd.get("deps", [])
            )
            g._next_id = max(g._next_id, nd["id"] + 1)
        return g

    @staticmethod
    def from_json(json_str: str) -> 'ComputeGraph':
        """从 JSON 字符串反序列化。"""
        return ComputeGraph.from_dict(json.loads(json_str))

    def __repr__(self):
        return f"Graph('{self.name}', nodes={len(self.nodes)})"


def build_example_graph() -> ComputeGraph:
    """Build a simple example graph: kernel_A → kernel_B → kernel_C."""
    g = ComputeGraph("example_pipeline")
    n_a = g.add_kernel("kernel_A", grid_dim=(1,), block_dim=(4,))
    n_b = g.add_kernel("kernel_B", grid_dim=(1,), block_dim=(4,), dependencies=[n_a])
    n_barrier = g.add_barrier("sync", dependencies=[n_b])
    g.add_kernel("kernel_C", grid_dim=(1,), block_dim=(4,), dependencies=[n_barrier])
    return g
