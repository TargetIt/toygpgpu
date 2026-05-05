; Test 03: Memory Load/Store
; Store values, then load them back through different registers
MOV r1, 100
ST r1, [10]      ; mem[10] = 100

MOV r2, 200
ST r2, [20]      ; mem[20] = 200

LD r3, [10]      ; r3 = mem[10] = 100
LD r4, [20]      ; r4 = mem[20] = 200

ADD r5, r3, r4   ; r5 = 300
ST r5, [30]      ; mem[30] = 300
HALT
