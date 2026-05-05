# Phase 2: SIMT 核心 (Warp/Thread)

引入 SIMT 执行模型——多线程共享 PC 的 Warp 概念。对标 GPGPU-Sim 的 `shd_warp_t`。

## 新增概念
- Thread: 每线程独立寄存器堆
- Warp: 一组线程共享 PC + active mask
- Warp Scheduler: Round-Robin 调度
- TID/WID/BAR 指令
- base+thread_id 访存模式

## 运行

```bash
cd phase2_simt && bash run.sh
```

## 测试
43 项测试，含多线程向量加法、barrier 同步、多 warp 并发。

## 对标 GPGPU-Sim
`shd_warp_t` + `scheduler_unit` + `shader_core_ctx`
