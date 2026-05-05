; Test 04: Multi-Warp Shared Memory
; All warps access the same shared memory space
WID r1
MOV r2, 10
MUL r3, r1, r2      ; r3 = wid*10 (unique per warp)
TID r4
ADD r5, r3, r4       ; r5 = wid*10 + tid

SHST r5, [0]         ; each thread writes to shared_mem[tid]

; All threads read back
SHLD r6, [0]         ; r6 = shared_mem[tid] (from last writing warp)
ST r6, [200]         ; mem[200+tid] = r6
HALT
