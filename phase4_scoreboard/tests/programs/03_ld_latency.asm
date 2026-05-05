; Test 03: LD Pipeline Latency
; LD takes 4 cycles — subsequent instructions using loaded reg must stall
MOV r1, 42
ST r1, [50]       ; mem[50] = 42
LD r2, [50]       ; r2 = 42 (4 cycle latency)
ADD r3, r2, r2    ; RAW: r2 still pending! Must stall and retry
ST r3, [0]        ; mem[0] = 84
HALT
