// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test for the `microgpu` dialect (Phase 8 / Task 73):
// the verifier rejects a `microgpu.load_global` whose `addr` operand
// carries the wrong width (i64 instead of i32). The check fires from
// the op's `verify_` hook and reads as
// `microgpu.load_global operand 'addr' must be i32, got i64`.

func.func @bad_load_global_addr_width(%addr: i64) -> i64 {
    // CHECK-ERR: microgpu.load_global operand 'addr' must be i32
    %0 = microgpu.load_global %addr : (i64) -> i64
    return %0 : i64
}