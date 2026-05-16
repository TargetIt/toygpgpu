"""
CuTile Parser — CuTile DSL → toygpgpu Internal ISA (Phase 14)
==============================================================
对标 CUTLASS 的 Tile 抽象和 CUDA C++ 模板风格的 tile 描述。

CuTile 语法 (简化):
  tile M=<int>, N=<int>, K=<int>
  kernel <name>(A:[M,K], B:[K,N], C:[M,N]) {
      load A[0:M, 0:K] -> smem[<offset>]
      load B[0:K, 0:N] -> smem[<offset>]
      mma smem[<A_off>], smem[<B_off>] -> smem[<C_off>]
      store smem[<C_off>] -> C[0:M, 0:N]
  }

参考: CUTLASS (github.com/NVIDIA/cutlass)
      Triton DSL (github.com/openai/triton)
"""

import re
from typing import List, Tuple, Dict, Optional


class TileConfig:
    """Tile shape configuration."""
    def __init__(self, m: int = 8, n: int = 8, k: int = 8):
        self.M = m
        self.N = n
        self.K = k

    def __repr__(self):
        return f"Tile({self.M}x{self.N}x{self.K})"


class CuTileKernel:
    """Parsed CuTile kernel representation."""
    def __init__(self, name: str):
        self.name = name
        self.params: Dict[str, Tuple[str, str]] = {}  # name -> (shape_str, role)
        self.tile: Optional[TileConfig] = None
        self.ops: List[Dict] = []  # list of operations

    def __repr__(self):
        return f"Kernel({self.name}, tile={self.tile}, ops={len(self.ops)})"


def parse_cutile(source: str) -> CuTileKernel:
    """Parse CuTile DSL source → CuTileKernel IR.

    Returns a CuTileKernel object containing the parsed tile config,
    kernel parameters, and operation sequence.
    """
    kernel = None
    lines = _clean_source(source)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse tile declaration
        m = re.match(r'tile\s+M\s*=\s*(\d+)\s*,\s*N\s*=\s*(\d+)\s*,\s*K\s*=\s*(\d+)', line, re.I)
        if m:
            kernel = kernel or CuTileKernel("unnamed")
            kernel.tile = TileConfig(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        # Parse kernel declaration
        m = re.match(r'kernel\s+(\w+)\s*\((.*?)\)\s*\{?', line, re.I)
        if m:
            kernel = kernel or CuTileKernel("unnamed")
            kernel.name = m.group(1)
            params_str = m.group(2)
            for param in params_str.split(','):
                param = param.strip()
                pm = re.match(r'(\w+)\s*:\s*\[(\w+)\s*,\s*(\w+)\]', param)
                if pm:
                    kernel.params[pm.group(1)] = (pm.group(2), pm.group(3))
            continue

        # Parse operations
        # load A[0:M, 0:K] -> smem[<offset>]
        m = re.match(r'load\s+(\w+)\[.*?\]\s*->\s*smem\[(\d+)\]', line, re.I)
        if m:
            kernel = kernel or CuTileKernel("unnamed")
            kernel.ops.append({'op': 'load', 'matrix': m.group(1), 'smem_off': int(m.group(2))})
            continue

        # store smem[<offset>] -> C[0:M, 0:N]
        m = re.match(r'store\s+smem\[(\d+)\]\s*->\s*(\w+)\[.*?\]', line, re.I)
        if m:
            kernel = kernel or CuTileKernel("unnamed")
            kernel.ops.append({'op': 'store', 'smem_off': int(m.group(1)), 'matrix': m.group(2)})
            continue

        # mma smem[<A>], smem[<B>] -> smem[<C>]
        m = re.match(r'mma\s+smem\[(\d+)\]\s*,\s*smem\[(\d+)\]\s*->\s*smem\[(\d+)\]', line, re.I)
        if m:
            kernel = kernel or CuTileKernel("unnamed")
            kernel.ops.append({
                'op': 'mma',
                'smem_a': int(m.group(1)),
                'smem_b': int(m.group(2)),
                'smem_c': int(m.group(3))
            })
            continue

        # closing brace (ignore)
        if line == '}' or line == '{':
            continue

    if kernel is None:
        raise ValueError("No kernel definition found in CuTile source")
    if kernel.tile is None:
        kernel.tile = TileConfig()  # default

    return kernel


def generate_asm(kernel: CuTileKernel, matrix_data: Dict[str, Dict] = None) -> str:
    """Generate toygpgpu assembly from CuTileKernel IR.

    Args:
        kernel: Parsed CuTile kernel
        matrix_data: Dict of matrix_name -> {'base': global_addr, 'M': rows, 'N': cols}

    Returns:
        Assembly source code string
    """
    tile = kernel.tile
    lines = []
    lines.append(f"; Auto-generated from CuTile: {kernel.name}")
    lines.append(f"; Tile shape: M={tile.M}, N={tile.N}, K={tile.K}")
    lines.append("")

    # Configure tile
    lines.append(f"; Configure tile dimensions")
    lines.append(f"TLCONF {tile.M}, {tile.N}, {tile.K}")
    lines.append("")

    # Process operations
    for op in kernel.ops:
        if op['op'] == 'load':
            mat_name = op['matrix']
            smem_off = op['smem_off']
            if matrix_data and mat_name in matrix_data:
                glob_base = matrix_data[mat_name]['base']
            else:
                glob_base = 0  # default
            lines.append(f"; Load {mat_name} tile → shared memory[{smem_off}]")
            lines.append(f"TLDS {smem_off}, {glob_base}")
            lines.append("")

        elif op['op'] == 'store':
            mat_name = op['matrix']
            smem_off = op['smem_off']
            if matrix_data and mat_name in matrix_data:
                glob_base = matrix_data[mat_name]['base']
            else:
                glob_base = 0
            lines.append(f"; Store result from shared memory[{smem_off}] → {mat_name}")
            lines.append(f"TLSTS {smem_off}, {glob_base}")
            lines.append("")

        elif op['op'] == 'mma':
            smem_a = op['smem_a']
            smem_b = op['smem_b']
            smem_c = op['smem_c']
            lines.append(f"; Warp-group MMA: smem[{smem_a}] × smem[{smem_b}] → smem[{smem_c}]")
            lines.append(f"; Computing {tile.M}x{tile.N} output, K={tile.K}")
            lines.append(f"; Using shared memory tiles:")
            lines.append(f";   A tile: {tile.M}x{tile.K} at smem[{smem_a}]")
            lines.append(f";   B tile: {tile.K}x{tile.N} at smem[{smem_b}]")

            # For each output element C[i][j], compute sum_k(A[i][k] * B[k][j])
            # Using shared memory loads (SHLD) and scalar ALU
            c_offset = smem_c
            for i in range(tile.M):
                for j in range(tile.N):
                    lines.append(f"; C[{i}][{j}] = sum over k")
                    # First term: A[i][0] * B[0][j]
                    a_idx = smem_a + i * tile.K + 0
                    b_idx = smem_b + 0 * tile.N + j
                    lines.append(f"SHLD r10, {a_idx}     ; A[{i}][0]")
                    lines.append(f"SHLD r11, {b_idx}     ; B[0][{j}]")
                    lines.append(f"MUL r12, r10, r11     ; A[{i}][0] * B[0][{j}]")

                    # Remaining K-1 terms
                    for k in range(1, tile.K):
                        a_idx = smem_a + i * tile.K + k
                        b_idx = smem_b + k * tile.N + j
                        lines.append(f"SHLD r10, {a_idx}     ; A[{i}][{k}]")
                        lines.append(f"SHLD r11, {b_idx}     ; B[{k}][{j}]")
                        lines.append(f"MUL r13, r10, r11     ; A[{i}][{k}] * B[{k}][{j}]")
                        lines.append(f"ADD r12, r12, r13     ; accumulate")

                    # Store to output tile in shared memory
                    c_addr = c_offset + i * tile.N + j
                    lines.append(f"; Store C[{i}][{j}] to smem[{c_addr}]")
                    lines.append(f"ST r12, [{c_addr}]")
                    lines.append("")

        lines.append("HALT")

    return '\n'.join(lines)


def _clean_source(source: str) -> List[str]:
    """Strip comments and blank lines from CuTile source."""
    clean = []
    for line in source.split('\n'):
        # Strip // comments and ; comments
        line = line.split('//')[0].split(';')[0].strip()
        if line:
            clean.append(line)
    return clean


def assemble_cutile(source: str, matrix_data: Dict[str, Dict] = None) -> Tuple[List[int], str]:
    """Full CuTile pipeline: parse → generate → assemble.

    Args:
        source: CuTile DSL source code
        matrix_data: Dict of matrix_name -> {'base': addr, 'M': rows, 'N': cols}

    Returns:
        (machine_code, assembly_text) tuple
    """
    try:
        from assembler import assemble
    except ImportError:
        from .assembler import assemble

    kernel = parse_cutile(source)
    asm_text = generate_asm(kernel, matrix_data)
    machine_code = assemble(asm_text)
    return machine_code, asm_text
