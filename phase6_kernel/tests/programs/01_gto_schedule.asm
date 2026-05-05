; Test 01: GTO vs RR scheduling — both produce correct results
; All warps execute same kernel: C[tid] = tid * warp_id + 10
WID r1
MOV r2, 10
MUL r3, r1, r2      ; r3 = wid * 10
TID r4
ADD r5, r3, r4       ; r5 = wid*10 + tid
ST r5, [100]         ; mem[100+tid] = r5
HALT
