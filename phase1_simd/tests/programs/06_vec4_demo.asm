; Phase 1 Vec4 Demo: Pack/Add/Mul/Unpack
; ==========================================
; 演示 float4/vec4 复合数据类型模拟:
;   将 4 个 8-bit 值打包到 32-bit 寄存器中
;   执行逐分量 SIMD 运算 (V4ADD, V4MUL)
;   提取单个分量 (V4UNPACK)
;
; 准备两个 vec4 值:
;   v1 = (3, 5, 7, 9)
;   v2 = (1, 2, 3, 4)

; ---- 构建 v1 = (3, 5, 7, 9) ----
; 先将 byte0, byte1 配对装入 r5
MOV r5, 3            ; byte0 = 3
MOV r10, 256         ; 移位乘数
MOV r11, 5
MUL r11, r11, r10    ; r11 = 5 * 256 = 1280
ADD r5, r5, r11      ; r5 = 3 + 1280 = 1283 = 0x0503

; 再将 byte2, byte3 配对装入 r6
MOV r6, 7            ; byte2 = 7
MOV r11, 9
MUL r11, r11, r10    ; r11 = 9 * 256 = 2304
ADD r6, r6, r11      ; r6 = 7 + 2304 = 2311 = 0x0907

; V4PACK: 将 4 字节打包到 r7 = (9, 7, 5, 3) = 0x09070503
V4PACK r7, r5, r6

; ---- 构建 v2 = (1, 2, 3, 4) ----
MOV r8, 1
MOV r11, 2
MUL r11, r11, r10
ADD r8, r8, r11      ; r8 = 1 + 512 = 513 = 0x0201

MOV r9, 3
MOV r11, 4
MUL r11, r11, r10
ADD r9, r9, r11      ; r9 = 3 + 1024 = 1027 = 0x0403

V4PACK r12, r8, r9   ; r12 = (4, 3, 2, 1) = 0x04030201

; ---- V4ADD: 逐字节加法 ----
; r13 = v1 + v2 = (3+1, 5+2, 7+3, 9+4) = (4, 7, 10, 13) = 0x0D0A0704
V4ADD r13, r7, r12

; ---- V4MUL: 逐字节乘法 ----
; r14 = v1 * v2 = (3*1, 5*2, 7*3, 9*4) = (3, 10, 21, 36) = 0x24150A03
V4MUL r14, r7, r12

; ---- V4UNPACK: 提取 r13 的各个 lane ----
V4UNPACK r1, r13, 0  ; r1 = lane 0 = 4
V4UNPACK r2, r13, 1  ; r2 = lane 1 = 7
V4UNPACK r3, r13, 2  ; r3 = lane 2 = 10
V4UNPACK r4, r13, 3  ; r4 = lane 3 = 13

; ---- 存储结果到内存 ----
ST r7, [0]           ; mem[0] = packed v1 = 0x09070503
ST r12, [1]          ; mem[1] = packed v2 = 0x04030201
ST r13, [2]          ; mem[2] = V4ADD result = 0x0D0A0704
ST r14, [3]          ; mem[3] = V4MUL result = 0x24150A03
ST r1, [4]           ; mem[4] = lane 0 = 4
ST r2, [5]           ; mem[5] = lane 1 = 7
ST r3, [6]           ; mem[6] = lane 2 = 10
ST r4, [7]           ; mem[7] = lane 3 = 13

HALT
