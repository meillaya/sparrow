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
