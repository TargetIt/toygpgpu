; Test 02: Bank Conflict
; r1=bank1, r5=bank1 → same bank → bank conflict → +1 cycle
; r2=bank2, r6=bank2 → same bank → bank conflict
MOV r1, 10       ; r1 → bank 1
MOV r5, 20       ; r5 → bank 1 (same bank!)
ADD r2, r1, r5   ; r1(bank1) + r5(bank1) → conflict!
ST r2, [0]       ; mem[0] = 30

MOV r2, 5        ; r2 → bank 2
MOV r3, 3        ; r3 → bank 3
ADD r4, r2, r3   ; r2(bank2) + r3(bank3) → no conflict!
ST r4, [1]       ; mem[1] = 8
HALT
