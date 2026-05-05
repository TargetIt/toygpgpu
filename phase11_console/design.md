# Phase 11 设计: 学习控制台

## 架构
- `learning_console.py`: 交互循环 + 命令解析
- `console_display.py`: 渲染函数 (纯逻辑, 无状态)

## 渲染布局 (每周期)
```
╔══ Cycle N ═══════════════════════════════════╗
  W0:PC=N mask=0b1111 [ACT]
  ┌─ Pipeline (5 stages) ──────────────────────┐
  ┌─ Register Changes ─────────────────────────┐
  ┌─ Scoreboard ───────────────────────────────┐
  ┌─ I-Buffer ─────────────────────────────────┐
  ┌─ SIMT Stack (if active) ───────────────────┐
  ┌─ Memory Changes ───────────────────────────┐
  OpCollector stats | L1Cache stats
╚══════════════════════════════════════════════╝
```

## 命令
- Enter/s: 步进 1 cycle
- r/r N: 自动运行 / 运行 N cycles
- i/m/reg/sb/ib/stack: 状态查询
- b PC/b list/b clear: 断点管理
- q: 退出

## ANSI 着色
- 青色: 框架线
- 绿色: 活跃/正确
- 红色: stall/错误
- 黄色: 数据变化
- 灰色: 空闲/无数据
