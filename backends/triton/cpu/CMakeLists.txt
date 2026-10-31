# backends/triton/cpu/ — Triton backend Path A, CPU reference kernel.
#
# Spec §7 Path A: hand-tuned C++ kernel for fused_gemm_bias_gelu on CPU.
# Shipped scope: naive triple-loop in fp32. Spec §7 Path B codegen is
# not exercised by this kernel — the path-A reference kernel covers
# the full pipeline end-to-end on a CPU-only host.
#
# The kernel is intentionally built as a SEPARATE static library so it can
# also be linked into standalone micro-benchmarks without dragging the full
# runtime in. The runtime links against this library to get
# `run_fused_gemm_bias_gelu`.
add_library(sparrow_triton_cpu STATIC
    fused_gemm_bias_gelu.cpp
)

target_include_directories(sparrow_triton_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
)

target_compile_features(sparrow_triton_cpu PUBLIC cxx_std_20)
target_compile_options(sparrow_triton_cpu PRIVATE -Werror)

# The shipped kernel only depends on the ABI headers (for Result<void> /
# Error) and <cmath>. No nlohmann_json, no CUDA, no Triton. The include
# path above supplies the header; the runtime links this target, not the
# other way around.

add_library(sparrow::triton_cpu ALIAS sparrow_triton_cpu)
add_library(forgestack::triton_cpu ALIAS sparrow_triton_cpu)