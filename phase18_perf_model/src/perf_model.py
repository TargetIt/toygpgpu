"""
Performance Model — Roofline & Bottleneck Analysis (Phase 18)
===============================================================
对标 NVIDIA Nsight Compute 和 Roofline Model 的性能分析。

Roofline Model 帮助判断 kernel 是 compute-bound 还是 memory-bound:
  - Attainable GFLOP/s = min(Peak GFLOP/s, Peak GB/s × Operational Intensity)
  - Operational Intensity = FLOP / Bytes accessed

参考: Williams, Waterman, Patterson (2009) "Roofline: An Insightful Visual
      Performance Model for Multicore Architectures"
      NVIDIA Nsight Compute (docs.nvidia.com/nsight-compute)
"""

from typing import Dict, List, Tuple


class RooflineModel:
    """Roofline 性能模型。

    Attributes:
        peak_flops: 峰值计算能力 (GFLOPS)
        peak_bandwidth: 峰值带宽 (GB/s)
        compute_bound_threshold: 超过此 OI 值为 compute-bound
    """

    def __init__(self, peak_flops: float = 100.0, peak_bandwidth: float = 50.0):
        self.peak_flops = peak_flops  # GFLOPS
        self.peak_bandwidth = peak_bandwidth  # GB/s
        self.compute_bound_threshold = peak_flops / peak_bandwidth  # FLOP/Byte

    def attainable_performance(self, operational_intensity: float) -> float:
        """根据 operational intensity 计算可达性能。

        Args:
            operational_intensity: FLOP/Byte

        Returns:
            可达 GFLOPS
        """
        return min(self.peak_flops,
                   self.peak_bandwidth * operational_intensity)

    def classify(self, operational_intensity: float) -> str:
        """分类 kernel 是 compute-bound 还是 memory-bound。"""
        ridge_point = self.peak_flops / self.peak_bandwidth
        if operational_intensity >= ridge_point:
            return "compute-bound"
        else:
            return "memory-bound"

    def roofline_data(self, oi_range: Tuple[float, float] = None,
                      points: int = 100) -> List[Tuple[float, float]]:
        """生成 Roofline 曲线数据点。

        Returns:
            [(oi, attainable_gflops)] 列表
        """
        if oi_range is None:
            oi_range = (0.01, self.compute_bound_threshold * 3)
        data = []
        for i in range(points):
            oi = oi_range[0] + (oi_range[1] - oi_range[0]) * i / (points - 1)
            data.append((oi, self.attainable_performance(oi)))
        return data

    def ascii_chart(self, kernels: Dict[str, float] = None,
                    width: int = 50, height: int = 20) -> str:
        """生成 ASCII Roofline 图表。

        Args:
            kernels: {name: operational_intensity} 映射
        """
        if kernels is None:
            kernels = {}
        lines = []
        lines.append("Roofline Model Chart")
        lines.append(f"  Peak: {self.peak_flops} GFLOPS, "
                     f"{self.peak_bandwidth} GB/s")
        lines.append(f"  Ridge Point: {self.compute_bound_threshold:.1f} FLOP/Byte")
        lines.append("  " + "-" * width)

        max_oi = max(list(kernels.values()) + [self.compute_bound_threshold * 2])
        max_perf = self.peak_flops * 1.1

        # Y-axis
        for y in range(height, 0, -1):
            perf = max_perf * y / height
            line = f"  {perf:5.0f} |"
            for x in range(width):
                oi = max_oi * x / width
                roof = self.attainable_performance(oi)
                # Check if near a kernel point
                pt = ''
                for name, k_oi in kernels.items():
                    kx = int(k_oi / max_oi * width) if max_oi > 0 else 0
                    ky = int(self.attainable_performance(k_oi) / max_perf * height)
                    if x == kx and y == ky:
                        pt = 'X'
                        break
                if pt:
                    line += pt
                elif perf <= roof:
                    line += '.'
                else:
                    line += ' '
            lines.append(line + "|")

        # X-axis
        lines.append("  " + " " * 6 + "-" * width)
        lines.append("  " + " " * 6 + f"0{' ' * (width//2)}OI (FLOP/Byte){' ' * (width//2-8)}{max_oi:.1f}")

        if kernels:
            lines.append("\n  Kernel Points:")
            for name, oi in kernels.items():
                perf = self.attainable_performance(oi)
                cls = self.classify(oi)
                lines.append(f"    X {name}: OI={oi:.2f}, "
                           f"Perf={perf:.1f} GFLOPS ({cls})")
        return '\n'.join(lines)


class PerfAnalyzer:
    """Kernel 性能分析器。

    分析 kernel 的:
      - 操作强度 (Operational Intensity)
      - 瓶颈类型
      - 优化建议
    """

    def __init__(self, roofline: RooflineModel = None):
        self.roofline = roofline or RooflineModel()

    def analyze(self, name: str, total_flops: int, bytes_read: int,
                bytes_written: int) -> Dict:
        """分析单个 kernel 的性能。

        Args:
            name: kernel 名称
            total_flops: 总浮点/整数操作数
            bytes_read: 读取字节数
            bytes_written: 写入字节数

        Returns:
            分析结果 dict
        """
        total_bytes = bytes_read + bytes_written
        oi = total_flops / total_bytes if total_bytes > 0 else float('inf')
        attainable = self.roofline.attainable_performance(oi)
        classification = self.roofline.classify(oi)
        suggestions = self._suggest(classification, oi, total_bytes)

        return {
            "name": name,
            "total_flops": total_flops,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "total_bytes": total_bytes,
            "operational_intensity": oi,
            "attainable_gflops": attainable,
            "classification": classification,
            "suggestions": suggestions,
        }

    def _suggest(self, classification: str, oi: float,
                 total_bytes: int) -> List[str]:
        """根据分类生成优化建议。"""
        suggestions = []
        if classification == "memory-bound":
            suggestions.append(
                "Memory-bound: 考虑使用 tiling 提高数据复用")
            suggestions.append(
                "使用 shared memory 减少 global memory 访问")
            suggestions.append(
                f"当前 OI={oi:.1f}, 需达到 "
                f"{self.roofline.compute_bound_threshold:.1f} 才能转为 compute-bound")
        else:
            suggestions.append(
                "Compute-bound: 考虑使用更高效的算法或 MMA 指令")
            suggestions.append("检查是否有不必要的计算可以消除")
        if total_bytes > 1024:
            suggestions.append("数据量较大, 检查 coalescing 是否充分")
        return suggestions

    def report(self, results: List[Dict]) -> str:
        """生成多 kernel 分析报告。"""
        lines = ["=" * 60,
                 "Performance Analysis Report / 性能分析报告",
                 "=" * 60]
        for r in results:
            lines.append(f"\n  Kernel: {r['name']}")
            lines.append(f"    FLOPs: {r['total_flops']}, "
                        f"Bytes: R={r['bytes_read']}/W={r['bytes_written']}")
            lines.append(f"    OI: {r['operational_intensity']:.2f} FLOP/Byte")
            lines.append(f"    Attainable: {r['attainable_gflops']:.1f} GFLOPS")
            lines.append(f"    Type: {r['classification']}")
            for s in r['suggestions']:
                lines.append(f"    → {s}")

        # Summary
        compute_bound = sum(1 for r in results
                           if r['classification'] == 'compute-bound')
        memory_bound = len(results) - compute_bound
        lines.append(f"\n  Summary: {compute_bound} compute-bound, "
                    f"{memory_bound} memory-bound out of {len(results)} kernels")
        return '\n'.join(lines)
