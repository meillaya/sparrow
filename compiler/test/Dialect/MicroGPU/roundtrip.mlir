// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// Round-trip golden test for the `microgpu` dialect (Phase 8 / Task 73):
// parse each of the 6 ops + verify + re-print + FileCheck the structural
// skeleton. All operands / results are typed `i32` (the canonical
// register + address width for the single-warp backend — see
// `compiler/lib/Dialect/MicroGPU/__init__.py`).
//
// FileCheck expectations:
//   * function name preserved (@microgpu_primitives)
//   * all 6 ops appear in the printed IR
//   * i32 types ride through the assembly format unchanged

// CHECK-LABEL: func.func @microgpu_primitives
func.func @microgpu_primitives(%addr: i32, %x: i32, %y: i32) -> i32 {
    // CHECK: microgpu.load_global
    // CHECK-SAME: (i32) -> i32
    %a = microgpu.load_global %addr : (i32) -> i32

    // CHECK: microgpu.add
    // CHECK-SAME: (i32, i32) -> i32
    %b = microgpu.add %x, %a : (i32, i32) -> i32

    // CHECK: microgpu.mul
    // CHECK-SAME: (i32, i32) -> i32
    %c = microgpu.mul %b, %y : (i32, i32) -> i32

    // CHECK: microgpu.sigmoid
    // CHECK-SAME: (i32) -> i32
    %d = microgpu.sigmoid %c : (i32) -> i32

    // CHECK: microgpu.store_global
    // CHECK-SAME: (i32, i32)
    microgpu.store_global %d, %addr : (i32, i32)

    // CHECK: microgpu.return
    microgpu.return

    // CHECK: return
    return %d : i32
}