// RUN: %forge-opt %s --action=roundtrip | %FileCheck %s

// Round-trip golden test for `forge.schedule` (Wave 0 / Task 2):
// parse a single `forge.schedule.tiled_gemm` op, verify it, re-print it,
// and FileCheck the structural skeleton. The `loop` and `stages` properties
// take defaults (`kmn` and `3`) so the printed form elides them.

func.func @scheduled_gemm(
    %a: !forge.tensor<64x32xf32>,
    %b: !forge.tensor<32x128xf32>
) -> !forge.tensor<64x128xf32> {
    // CHECK-LABEL: func.func @scheduled_gemm
    // CHECK-SAME: !forge.tensor<64x32xf32>
    // CHECK-SAME: !forge.tensor<32x128xf32>
    %0 = forge.schedule.tiled_gemm %a, %b {tile = #forge.tile_shape<64x128x32>, loop = #forge.loop_order<kmn>, stages = #forge.pipeline_stages<3>} : (!forge.tensor<64x32xf32>, !forge.tensor<32x128xf32>) -> !forge.tensor<64x128xf32>
    // CHECK: forge.schedule.tiled_gemm
    // CHECK-SAME: tile = #forge.tile_shape<64x128x32>
    // CHECK-SAME: !forge.tensor<64x32xf32>
    // CHECK-SAME: !forge.tensor<32x128xf32>
    // CHECK-SAME: -> !forge.tensor<64x128xf32>
    // CHECK: return
    return %0 : !forge.tensor<64x128xf32>
}
