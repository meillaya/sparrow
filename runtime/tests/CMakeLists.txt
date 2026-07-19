# runtime/tests/ — GoogleTest harness for ForgeRT.
# Task 10 ships test_abi (ABI round-trip). Tasks 11/12 add
# test_triton_provider and test_scheduler respectively.
add_executable(test_abi test_abi.cpp)
target_link_libraries(test_abi
    PRIVATE
        forgestack::runtime
        GTest::gtest
        GTest::gtest_main
)
add_test(NAME test_abi COMMAND test_abi)

# Task 11: TritonDeviceProvider + fused_gemm_bias_gelu CPU kernel.
# Verifies allocate/release, kernel-vs-reference correctness, and the
# fp32-only dtype gate.
add_executable(test_triton_provider test_triton_provider.cpp)
target_link_libraries(test_triton_provider
    PRIVATE
        forgestack::runtime
        forgestack::triton_cpu
        GTest::gtest
        GTest::gtest_main
)
add_test(NAME test_triton_provider COMMAND test_triton_provider)

# Task 12 — ExecutionPlan JSON round-trip + topological scheduler
# exercised via MockDeviceProvider (no real provider dependency).
add_executable(test_scheduler test_scheduler.cpp)
target_link_libraries(test_scheduler
    PRIVATE
        forgestack::runtime
        GTest::gtest
        GTest::gtest_main
)
add_test(NAME test_scheduler COMMAND test_scheduler)

# Task 26 — LivenessAnalyzer (backward dataflow over ExecutionPlan).
# Verifies the per-tensor [first_use, last_use] intervals on linear
# and diamond plans. Header-only dep on execution_plan.hpp.
add_executable(test_liveness test_liveness.cpp)
target_link_libraries(test_liveness
    PRIVATE
        forgestack::runtime
        GTest::gtest
        GTest::gtest_main
)
add_test(NAME test_liveness COMMAND test_liveness)

# Task 27 — BufferReuseAllocator (first-fit-decreasing layout).
# Verifies non-overlapping ranges share offsets and overlapping ranges
# get distinct offsets. Depends on liveness.hpp for LiveRange.
add_executable(test_buffer_reuse test_buffer_reuse.cpp)
target_link_libraries(test_buffer_reuse
    PRIVATE
        forgestack::runtime
        GTest::gtest
        GTest::gtest_main
)
add_test(NAME test_buffer_reuse COMMAND test_buffer_reuse)
