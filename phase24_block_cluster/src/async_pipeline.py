"""
Async Pipeline — Software-Managed Pipelining (Phase 22)
=========================================================
对标 NVIDIA Hopper 异步 warp 级流水线和 TMA 加载流水线。

核心抽象:
  - AsyncTransactionBarrier: 跟踪待处理的异步事务 (TMA 加载),
    所有事务完成后释放 barrier
  - PipelineStage: 表示软件流水线的一个阶段 (加载 → 计算 → 存储)
  - Producer/Consumer warp pattern: 一个 warp 产生数据到共享内存,
    另一个 warp 消费数据

参考:
  - CUDA Asynchronous Barriers (docs.nvidia.com/cuda/cuda-c-programming-guide)
  - Hopper TMA + async pipeline
  - GPGPU-Sim's pipeline model
"""

from enum import IntEnum
from typing import Callable, List, Optional


class PipelineStageType(IntEnum):
    """流水线阶段类型。"""
    LOAD = 0   # 从全局内存加载数据
    COMPUTE = 1  # 计算
    STORE = 2  # 写回结果
    IDLE = 3   # 空闲


class AsyncTransactionBarrier:
    """异步事务 Barrier — 跟踪待处理的异步事务。

    对标 CUDA __pipeline_wait() 和 mbarrier 机制。

    Attributes:
        pending_count: 当前待处理的异步事务数
        phase: 当前 phase (用于 double-buffering)
        completed: 是否已完成
    """

    def __init__(self):
        self.pending_count = 0
        self.phase = 0
        self.completed = False

    def increment(self, count: int = 1):
        """增加待处理事务计数 (模拟 arrive)。"""
        self.pending_count += count
        self.completed = False

    def decrement(self, count: int = 1):
        """减少待处理事务计数 (模拟 complete)。"""
        self.pending_count = max(0, self.pending_count - count)
        if self.pending_count == 0:
            self.completed = True

    def wait(self) -> bool:
        """等待所有异步事务完成。返回是否已完成。"""
        return self.completed

    def reset(self):
        """重置 barrier。"""
        self.pending_count = 0
        self.completed = False
        self.phase += 1

    def __repr__(self):
        return (f"AsyncTransactionBarrier(pending={self.pending_count}, "
                f"phase={self.phase}, completed={self.completed})")


class PipelineStage:
    """软件流水线的一个阶段。

    每个阶段包含:
      - type: 阶段类型 (LOAD, COMPUTE, STORE)
      - shared_mem_slot: 使用的共享内存槽位索引
      - data_size: 数据大小 (以 word 为单位)
      - barrier: 关联的异步 barrier

    支持 2-3 级软件流水线:
      Stage 0: LOAD
      Stage 1: COMPUTE
      Stage 2: STORE
    """

    def __init__(self, stage_type: PipelineStageType,
                 shared_mem_slot: int = 0, data_size: int = 0):
        self.type = stage_type
        self.shared_mem_slot = shared_mem_slot
        self.data_size = data_size
        self.barrier = AsyncTransactionBarrier()
        self.committed = False
        self.iteration = 0

    def start_load(self) -> AsyncTransactionBarrier:
        """开始一个加载阶段。返回关联的 async barrier。"""
        if self.type != PipelineStageType.LOAD:
            raise RuntimeError("start_load() called on non-LOAD stage")
        self.barrier.increment()
        self.committed = False
        return self.barrier

    def complete_load(self):
        """完成加载阶段。"""
        if self.type != PipelineStageType.LOAD:
            raise RuntimeError("complete_load() called on non-LOAD stage")
        self.barrier.decrement()

    def commit(self):
        """提交阶段完成 (相当于 pipeline commit)。"""
        self.committed = True

    def is_ready(self) -> bool:
        """检查阶段是否已就绪 (加载完成且已提交)。"""
        if self.type == PipelineStageType.COMPUTE:
            return True  # compute 阶段始终可用
        return self.barrier.completed and self.committed

    def execute_compute(self, data: List[int]) -> List[int]:
        """执行计算阶段 (模拟 warp 级计算)。"""
        if self.type != PipelineStageType.COMPUTE:
            raise RuntimeError("execute_compute() called on non-COMPUTE stage")
        return [x * 2 for x in data]

    def advance(self):
        """推进到下一轮迭代。"""
        self.iteration += 1
        self.committed = False
        self.barrier.reset()

    def __repr__(self):
        return (f"PipelineStage(type={self.type.name}, slot={self.shared_mem_slot}, "
                f"iter={self.iteration}, ready={self.is_ready()})")


class ProducerConsumerPipeline:
    """Producer-Consumer 流水线模式。

    一个 warp 生产数据到共享内存 (Producer warp),
    另一个 warp 消费数据 (Consumer warp)。

    Producer/Consumer pattern::
      Producer:   LOAD → barrier.arrive → (next iteration)
      Consumer:   barrier.wait → COMPUTE → STORE
    """

    def __init__(self, num_stages: int = 2):
        if num_stages < 2 or num_stages > 3:
            raise ValueError("Pipeline supports 2 or 3 stages")
        self.num_stages = num_stages
        self.stages: List[PipelineStage] = []
        self.shared_memory: List[List[int]] = [[] for _ in range(num_stages)]
        self.iteration = 0

        for i in range(num_stages):
            if i == 0:
                st = PipelineStage(PipelineStageType.LOAD, i, 16)
            elif i == num_stages - 1:
                st = PipelineStage(PipelineStageType.STORE, i, 16)
            else:
                st = PipelineStage(PipelineStageType.COMPUTE, i, 16)
            self.stages.append(st)

    def producer_load(self, stage_idx: int,
                      source_data: List[int]) -> AsyncTransactionBarrier:
        """Producer warp 加载数据到共享内存。

        Args:
            stage_idx: 阶段索引
            source_data: 从全局内存加载的数据

        Returns:
            关联的 async barrier
        """
        stage = self.stages[stage_idx]
        if stage.type != PipelineStageType.LOAD:
            raise RuntimeError("Producer can only write to LOAD stages")

        barrier = stage.start_load()
        self.shared_memory[stage_idx] = list(source_data)
        stage.complete_load()
        return barrier

    def producer_commit(self, stage_idx: int):
        """Producer warp 提交已完成的事务。"""
        self.stages[stage_idx].commit()

    def consumer_wait(self, stage_idx: int) -> bool:
        """Consumer warp 等待数据就绪。"""
        stage = self.stages[stage_idx]
        if stage.type not in (PipelineStageType.COMPUTE,
                              PipelineStageType.STORE):
            return False
        return stage.barrier.wait()

    def consumer_compute(self, stage_idx: int,
                         compute_fn: Optional[Callable] = None) -> List[int]:
        """Consumer warp 在数据上执行计算。"""
        stage = self.stages[stage_idx]
        data = self.shared_memory[stage.shared_mem_slot]

        if compute_fn:
            result = compute_fn(data)
        else:
            result = stage.execute_compute(data)

        # Store result back to shared memory for subsequent STORE stage
        self.shared_memory[stage_idx] = result
        return result

    def advance(self):
        """推进到流水线的下一轮迭代。"""
        for stage in self.stages:
            stage.advance()
        self.iteration += 1

    def run_pipeline(self, data_batches: List[List[int]],
                     compute_fn: Optional[Callable] = None) -> List[List[int]]:
        """运行完整的流水线处理: 对多个数据批次执行 load-compute-store。

        Args:
            data_batches: 多个数据批次列表
            compute_fn: 可选的用户定义计算函数

        Returns:
            计算结果列表
        """
        results = []
        num_batches = len(data_batches)

        for i in range(num_batches):
            batch = data_batches[i]
            stage_idx = i % self.num_stages

            if stage_idx < len(self.stages) and self.stages[stage_idx].type == PipelineStageType.LOAD:
                # Producer: load data
                self.producer_load(0, batch)
                self.producer_commit(0)

            # Consumer: wait and compute
            if self.consumer_wait(stage_idx):
                # Advance pipeline stages
                compute_stage = self.stages[1] if self.num_stages > 1 else self.stages[0]
                if compute_stage.type in (PipelineStageType.COMPUTE, PipelineStageType.STORE):
                    result = self.consumer_compute(1, compute_fn)
                    results.append(result)

            self.advance()

        return results
