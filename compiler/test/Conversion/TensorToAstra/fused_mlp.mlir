// RUN: not %forge-opt %s --action=lower-astra 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// The TensorToAstra pass is unavailable in 0.2.0. The dialect cannot
// represent SRAM addresses, the conversion invents gemm/epilogue shapes
// after lowering, and the 15-bit DMA immediate cannot carry the real byte
// counts. The CLI must reject the request, not emit wrong IR.

// CHECK-ERR: TensorToAstra is unavailable

func.func @fused_mlp_lowered(
    %x: !forge.tensor<2x512x4096xf32>,
    %w: !forge.tensor<4096x11008xf32>,
    %b: !forge.tensor<11008xf32>
) -> !forge.tensor<2x512x11008xf32> {
    %y = forge.tensor.gemm %x, %w {transpose_a = false, transpose_b = false}
        : (!forge.tensor<2x512x4096xf32>, !forge.tensor<4096x11008xf32>) -> !forge.tensor<2x512x11008xf32>
    %z = forge.tensor.fused_epilogue %y, %b {activation = #forge.activation<gelu>}
        : (!forge.tensor<2x512x11008xf32>, !forge.tensor<11008xf32>) -> !forge.tensor<2x512x11008xf32>
    return %z : !forge.tensor<2x512x11008xf32>
}
