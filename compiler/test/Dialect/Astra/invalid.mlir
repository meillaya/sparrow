// RUN: not %forge-opt %s --action=roundtrip 2>&1 | %FileCheck --check-prefix=CHECK-ERR %s

// Negative test for the `astra` dialect (Wave 9 / Task 40): the verifier
// rejects `k = 0` on `astra.microcode.matmul` because non-positive
// reduction dims are nonsensical (the systolic array needs at least
// one column of K). The check fires from the op's `verify_` hook and
// reads as `astra microcode.matmul k must be positive, got 0`.

func.func @bad_microcode_matmul_k(
    %sram_a: !astra.ptr<#astra.memory_space<sram0>>,
    %sram_b: !astra.ptr<#astra.memory_space<sram0>>
) -> !astra.ptr<#astra.memory_space<sram0>> {
    // CHECK-ERR: astra microcode.matmul k must be positive
    %0 = astra.microcode.matmul %sram_a, %sram_b {m = #builtin.int<16>, n = #builtin.int<16>, k = #builtin.int<0>} : (!astra.ptr<#astra.memory_space<sram0>>, !astra.ptr<#astra.memory_space<sram0>>) -> !astra.ptr<#astra.memory_space<sram0>>
    return %0 : !astra.ptr<#astra.memory_space<sram0>>
}
