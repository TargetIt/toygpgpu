; Test 03: Barrier Synchronization
; All threads compute something, then sync, then compute more
TID r1
MOV r2, 10
MUL r3, r1, r2      ; r3 = tid * 10
ST r3, [300]         ; phase 1: store before barrier

BAR                  ; all threads wait here

ADD r3, r3, r2       ; r3 += 10 (now tid*10 + 10)
ST r3, [310]         ; phase 2: store after barrier
HALT
