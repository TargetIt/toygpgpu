; Test 05: Multi-Warp Execution
; Test that multiple warps can coexist and execute independently
; Warp 0 writes to mem[0..7], Warp 1 to mem[100..107]
; (all warps run the same kernel but WID distinguishes them)
WID r1
MOV r2, 100
MUL r3, r1, r2    ; r3 = wid * 100 (base offset)

TID r4
ADD r5, r3, r4    ; r5 = wid*100 + tid (unique address per thread)
ST r5, [0]        ; store thread address to mem[tid]
HALT
