; Warp Register Demo (Phase 3: SIMT Stack)
; =========================================
; Demonstrates warp-level uniform register read/write.
;
; WREAD rd, wid   — read warp_id into rd (broadcast to all threads)
; WREAD rd, ntid  — read warp_size into rd (broadcast to all threads)
; WWRITE rs1, wid — write rs1 to warp_id register
;
; Strategy:
;   1. Read 'wid' warp register into r0 for all threads
;   2. Read 'ntid' warp register into r1 for all threads
;   3. Store r0 and r1 to memory for verification
;   4. Write r0+10 to wid warp register (modify it)
;   5. Read wid again into r2 to verify the write
;   6. Store r2 to memory

; Read warp_id into r0 for all active threads
WREAD r0, wid

; Read ntid (warp_size) into r1 for all active threads
WREAD r1, ntid

; Store results: mem[0] = warp_id, mem[1] = ntid
ST r0, [0]
ST r1, [1]

; Test WWRITE: set wid to 99
MOV r3, 99
WWRITE r3, wid

; Read back modified wid into r2
WREAD r2, wid

; Store: mem[2] = modified wid (should be 99)
ST r2, [2]

HALT
