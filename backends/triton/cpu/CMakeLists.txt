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
add_library(forgestack_triton_cpu STATIC
    fused_gemm_bias_gelu.cpp
)

target_include_directories(forgestack_triton_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
)

target_compile_features(forgestack_triton_cpu PUBLIC cxx_std_20)

# The shipped kernel only depends on the ABI headers (for Result<void> /
# Error) and <cmath>. No nlohmann_json, no CUDA, no Triton. The PUBLIC
# link against forgestack::runtime is *only* for the header dependency
# on Result<void>; because Result is header-only the runtime provides
# no symbols at link time, but the include-path coupling keeps the
# kernel header-only from accidently reaching into an internal lib
# path.
target_link_libraries(forgestack_triton_cpu PUBLIC forgestack::runtime)

add_library(forgestack::triton_cpu ALIAS forgestack_triton_cpu)