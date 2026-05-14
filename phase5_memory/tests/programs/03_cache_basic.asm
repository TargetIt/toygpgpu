; ============================================================
; Phase 5: Memory Subsystem — Cache Basic Operation
; 阶段 5：内存子系统 — 缓存基本操作
;
; Purpose / 目的:
;   Demonstrate cache behavior: first load of an address
;   is a cache miss, subsequent loads of the same address
;   are cache hits. Cache reduces memory latency.
;   演示缓存行为：首次加载某地址是缓存未命中 (miss)，
;   后续加载同一地址是缓存命中 (hit)。缓存减少延迟。
;
; Expected Result / 预期结果:
;   mem[10] = 84  (42 + 42)
;   (Cache loads r2 and r3 from same address, ADD sums them)
;
; Key Concepts / 关键概念:
;   - Cache miss: first LD to an address goes to global memory
;   - Cache hit: second LD to same address returns from cache
;   - Cache is transparent to software (same instructions)
;   - Cache improves performance without changing program logic
;   - 缓存透明: 程序无需修改即可受益于缓存
;
; Flow Diagram / 流程图:
;
;   START
;     |
;     v
;   MOV r1, 42
;   ST r1, [0]      ← mem[0] = 42  (写到全局内存)
;     |
;     v
;   LD r2, [0]      ← 第一次读取: cache MISS (从全局内存加载)
;     |                (First read: cache miss, load from global)
;     |                数据被缓存 (Data is cached)
;     v
;   LD r3, [0]      ← 第二次读取: cache HIT (从缓存加载)
;     |                (Second read: cache hit, load from cache)
;     v
;   ADD r4, r2, r3  ← r4 = 42 + 42 = 84
;     |
;     v
;   ST r4, [10]     ← mem[10] = 84
;     |
;     v
;   HALT
; ============================================================

; 准备缓存测试数据 / Prepare cache test data
MOV r1, 42       ; r1 = 42
ST r1, [0]       ; mem[0] = 42

; 第一次加载: 缓存未命中 / First load: cache miss
; 地址 [0] 不在缓存中, 需要从全局内存加载
; Address [0] is not in cache, must load from global memory
; 加载同时将数据存入缓存行
; Load also fills the cache line
LD r2, [0]       ; r2 = 42  (cache miss — 从全局内存加载)

; 第二次加载: 缓存命中 / Second load: cache hit
; 地址 [0] 已在缓存中, 直接从缓存读取 (低延迟)
; Address [0] is now cached, read directly from cache (low latency)
LD r3, [0]       ; r3 = 42  (cache hit — 从缓存加载)

; 对两次加载的值求和 / Sum the two loaded values
ADD r4, r2, r3   ; r4 = 42 + 42 = 84

; 存储到结果区 / Store result
ST r4, [10]      ; mem[10] = 84

; 程序终止 / Terminate program
HALT
