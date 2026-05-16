"""
Stream — Multi-Stream Abstraction (Phase 17)
=============================================
对标 CUDA Stream (cudaStream_t) 的异步执行模型。

每个 Stream 有独立的命令队列，不同 Stream 的命令可以并发执行。
Event 用于跨 Stream 同步。

参考: CUDA Streams (docs.nvidia.com/cuda)
      GPGPU-Sim stream_manager
"""

from typing import Dict, List, Optional


class Stream:
    """异步执行流。

    Attributes:
        stream_id: 流 ID
        commands: 命令队列 [(op_type, params)]
        events: 该流上记录的 event
    """

    def __init__(self, stream_id: int):
        self.stream_id = stream_id
        self.commands: List[tuple] = []  # (op_type, params_dict)
        self.events: List['Event'] = []
        self.active = True

    def submit(self, op_type: str, params: Dict = None):
        """向流提交命令。"""
        self.commands.append((op_type, params or {}))

    def record_event(self, event: 'Event'):
        """在流上记录 event。"""
        event.record(self.stream_id)
        self.events.append(event)

    def wait_event(self, event: 'Event'):
        """等待其他流的 event。"""
        self.commands.append(("wait_event", {"event": event}))

    def pending(self) -> int:
        """返回待执行命令数。"""
        return len(self.commands)

    def pop(self) -> Optional[tuple]:
        """取出下一个命令。"""
        if self.commands:
            return self.commands.pop(0)
        return None

    def __repr__(self):
        return f"Stream({self.stream_id}, pending={len(self.commands)})"


class Event:
    """跨 Stream 同步事件。

    Attributes:
        event_id: 事件 ID
        recorded: 是否已记录
        stream_id: 记录此事件的流 ID
        timestamp: 记录时的周期计数
    """

    _next_id = 0

    def __init__(self):
        self.event_id = Event._next_id
        Event._next_id += 1
        self.recorded = False
        self.stream_id = -1
        self.timestamp = 0

    def record(self, stream_id: int, timestamp: int = 0):
        """标记事件为已记录。"""
        self.recorded = True
        self.stream_id = stream_id
        self.timestamp = timestamp

    def is_ready(self) -> bool:
        """事件是否已完成。"""
        return self.recorded

    def __repr__(self):
        return f"Event({self.event_id}, recorded={self.recorded})"


class StreamManager:
    """管理多个 Stream 和 Event 的调度器。

    Attributes:
        streams: Stream 列表
        events: Event 列表
        cycle: 当前周期计数
    """

    def __init__(self, num_streams: int = 3):
        self.streams = [Stream(i) for i in range(num_streams)]
        self.events: List[Event] = []
        self.cycle = 0
        self.stats = {"commands_executed": 0, "events_recorded": 0,
                      "overlapped_cycles": 0}

    def create_event(self) -> Event:
        """创建新事件。"""
        ev = Event()
        self.events.append(ev)
        return ev

    def step(self) -> bool:
        """执行一个周期: 从各 stream 尝试取一个命令执行。"""
        executed = False
        for stream in self.streams:
            if not stream.active or not stream.commands:
                continue
            cmd_type, params = stream.commands[0]

            if cmd_type == "wait_event":
                ev = params["event"]
                if ev.is_ready():
                    stream.commands.pop(0)  # consume wait
                    executed = True
                continue  # blocked

            # Execute command
            stream.commands.pop(0)
            self.stats["commands_executed"] += 1
            executed = True

            if cmd_type == "record_event":
                ev = params.get("event")
                if ev:
                    ev.record(stream.stream_id, self.cycle)
                    self.stats["events_recorded"] += 1

        # Check for concurrent execution
        active_count = sum(1 for s in self.streams
                          if s.active and s.commands)
        if active_count > 1:
            self.stats["overlapped_cycles"] += 1

        self.cycle += 1
        return any(s.commands for s in self.streams)

    def run_all(self):
        """执行所有 pending 命令。"""
        while self.step():
            pass

    def report(self) -> str:
        """生成执行报告。"""
        return (
            f"Stream Manager Report:\n"
            f"  Streams: {len(self.streams)}\n"
            f"  Commands executed: {self.stats['commands_executed']}\n"
            f"  Events recorded: {self.stats['events_recorded']}\n"
            f"  Overlapped cycles: {self.stats['overlapped_cycles']}\n"
            f"  Total cycles: {self.cycle}"
        )
