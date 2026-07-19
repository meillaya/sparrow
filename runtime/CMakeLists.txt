# runtime/ — ForgeRT C++ execution runtime.
# Task 10 wires the ABI: TensorDescriptor / ArtifactDescriptor / Buffer /
# Event / DeviceProvider + GoogleTest round-trip tests. Tasks 11+ add the
# Triton provider, ExecutionPlan, and scheduler on top of this surface.
add_library(forgestack_runtime STATIC
    lib/abi_types.cpp
    lib/tensor_descriptor.cpp
    lib/artifact_descriptor.cpp
    lib/buffer.cpp
    lib/event.cpp
    lib/result.cpp
    lib/forge_rt.cpp
)

target_include_directories(forgestack_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_features(forgestack_runtime PUBLIC cxx_std_20)

target_link_libraries(forgestack_runtime PUBLIC nlohmann_json::nlohmann_json)

add_library(forgestack::runtime ALIAS forgestack_runtime)

enable_testing()
add_subdirectory(tests)
