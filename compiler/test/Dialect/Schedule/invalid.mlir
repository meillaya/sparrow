// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test for `forge.schedule`: the `pipeline_stages` verifier rejects
// `<= 0` because software-pipelining needs at least one stage. The original
// Wave 0 brief asked for `#forge.tile_shape<0x0x0>`, but Wave 0 runs BEFORE
// Wave 4 and `forge.tensor.constant` is not yet wired in (the constant op
// lands in Task 8); we therefore trip the `pipeline_stages` verifier instead
// via `stages = #forge.pipeline_stages<0>`. The MLIR lexer interprets `0x0`
// as a hex literal, so `<0x0x0>` would only yield two parseable dims — not
// three zeros — and would mask the tile-dim verifier.

func.func @bad_pipeline_stages(
    %a: !forge.tensor<2x2xf32>,
    %b: !forge.tensor<2x2xf32>
) -> !forge.tensor<2x2xf32> {
    // CHECK-ERR: pipeline stages must be at least 1
    %0 = forge.schedule.tiled_gemm %a, %b {tile = #forge.tile_shape<1x1x1>, stages = #forge.pipeline_stages<0>} : (!forge.tensor<2x2xf32>, !forge.tensor<2x2xf32>) -> !forge.tensor<2x2xf32>
    return %0 : !forge.tensor<2x2xf32>
}
