; Test 01: Shared Memory Read/Write
; Each thread writes tid*10 to shared_mem[tid], then reads back
TID r1
MOV r2, 10
MUL r3, r1, r2      ; r3 = tid * 10
SHST r3, [0]         ; shared_mem[tid] = tid*10
SHLD r4, [0]         ; r4 = shared_mem[tid]
ST r4, [100]         ; mem[100+tid] = r4
HALT
