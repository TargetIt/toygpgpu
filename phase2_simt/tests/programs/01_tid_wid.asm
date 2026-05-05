; Test 01: TID and WID
; Each thread stores its thread_id to mem[100+tid]
; Warp 0: tid 0..7 → mem[100..107] = 0..7
TID r1
ST r1, [100]
HALT
