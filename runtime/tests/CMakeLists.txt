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
