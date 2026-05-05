# Phase 4: Scoreboard (流水线冒险)

实现记分板检测 RAW/WAW 数据冒险。对标 GPGPU-Sim `scoreboard`。

## 新增概念
- Scoreboard: 每寄存器 pending write 倒计时
- RAW (读后写) 冒险检测
- WAW (写后写) 冒险检测
- 流水线延迟: ALU=1cycle, LD/ST=4cycles
- warp stall + 恢复机制

## 运行

```bash
cd phase4_scoreboard && bash run.sh
```

## 测试
15 项测试，含 RAW stall、WAW stall、LD 延迟后正确读回。

## 对标 GPGPU-Sim
`scoreboard` (register hazard tracking)
