# runtime/tests/ — GoogleTest harness for ForgeRT.
# Task 10 ships test_abi (ABI round-trip). Tasks 11/12 add
# test_triton_provider and test_scheduler respectively.
include(GoogleTest)

add_executable(test_abi test_abi.cpp)
target_link_libraries(test_abi
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_abi WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Task 11: TritonDeviceProvider + fused_gemm_bias_gelu CPU kernel.
# Verifies allocate/release, kernel-vs-reference correctness, and the
# fp32-only dtype gate.
add_executable(test_triton_provider test_triton_provider.cpp)
target_link_libraries(test_triton_provider
    PRIVATE
        sparrow::runtime
        sparrow::triton_cpu
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_triton_provider WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# CPU reference provider and kernel registry.
add_executable(test_cpu_provider test_cpu_provider.cpp)
target_link_libraries(test_cpu_provider
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_cpu_provider WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Task 12 — ExecutionPlan JSON round-trip + topological scheduler
# exercised via MockDeviceProvider (no real provider dependency).
add_executable(test_scheduler test_scheduler.cpp)
target_link_libraries(test_scheduler
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_scheduler WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Task 26 — LivenessAnalyzer (backward dataflow over ExecutionPlan).
# Verifies the per-tensor [first_use, last_use] intervals on linear
# and diamond plans. Header-only dep on execution_plan.hpp.
add_executable(test_liveness test_liveness.cpp)
target_link_libraries(test_liveness
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_liveness WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Task 27 — BufferReuseAllocator (first-fit-decreasing layout).
# Verifies non-overlapping ranges share offsets and overlapping ranges
# get distinct offsets. Depends on liveness.hpp for LiveRange.
add_executable(test_buffer_reuse test_buffer_reuse.cpp)
target_link_libraries(test_buffer_reuse
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_buffer_reuse WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Task 28 — MemoryPlanner (orchestrates Liveness + BufferReuse, mutates
# plan.allocations). Single end-to-end test on a 4-node diamond plan
# verifying alignment, no-overlap, and pool-bounded high-water mark.
add_executable(test_memory_planner test_memory_planner.cpp)
target_link_libraries(test_memory_planner
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_memory_planner WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Task 32 — AllocationDefinition's byte layout survives ExecutionPlan JSON.
add_executable(test_execution_plan test_execution_plan.cpp)
target_link_libraries(test_execution_plan
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_execution_plan WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Wave 14 — AstraDeviceProvider wraps the in-process Python simulator.
# Verifies the launch path that fires a `python3 -c <script>` subprocess
# against the project's .venv and parses a base64 tensor payload back.
add_executable(test_astra_provider test_astra_provider.cpp)
target_link_libraries(test_astra_provider
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_astra_provider WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Wave 17 — MicroGpuDeviceProvider wraps the in-process Python
# interpreter (single-warp SIMT bytecode). Verifies the launch path
# that fires a `python3 -c <script>` subprocess against the project's
# .venv and parses a base64 register-file payload back.
add_executable(test_microgpu_provider test_microgpu_provider.cpp)
target_link_libraries(test_microgpu_provider
    PRIVATE
        sparrow::runtime
        GTest::gtest
        GTest::gtest_main
)
gtest_discover_tests(test_microgpu_provider WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
