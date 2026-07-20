// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test: `forge.tensor.layer_norm` must reject a gamma whose
// shape[0] does not equal the input's last dim. The verifier's exact
// wording is part of the contract — pin it here so future refactors
// don't accidentally rename it.

// CHECK-ERR: gamma shape[0] must equal input.shape[-1]
func.func @bad_layer_norm_gamma(
    %x: !forge.tensor<2x512xf32>,
    %g: !forge.tensor<256xf32>,
    %b: !forge.tensor<512xf32>
) -> !forge.tensor<2x512xf32> {
    %0 = forge.tensor.layer_norm %x, %g, %b {epsilon = 1.000000e-05 : f32}
        : (!forge.tensor<2x512xf32>, !forge.tensor<256xf32>, !forge.tensor<512xf32>) -> !forge.tensor<2x512xf32>
    return %0 : !forge.tensor<2x512xf32>
}