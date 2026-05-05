; Test 04: r0 Write Protection
; Verify that r0 is always 0, regardless of writes
MOV r1, 42
ADD r0, r1, r1   ; try to write r0 = 84 (should be ignored)
ADD r2, r0, r1   ; r2 = 0 + 42 = 42 (r0 reads as 0)
ST r2, [0]       ; mem[0] = 42
HALT
