// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s
func.func @bad_matmul(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<2048x11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    // CHECK-ERR: error: matmul inner dims mismatch
    %0 = forge.graph.matmul %x, %w : (!forge.tensor<2x512x4096xf32>, !forge.tensor<2048x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %0 : !forge.tensor<2x512x11008xf32>
}
