# runtime/ — ForgeRT C++ execution runtime.
# Task 10 wires the ABI: TensorDescriptor / ArtifactDescriptor / Buffer /
# Event / DeviceProvider + GoogleTest round-trip tests. Tasks 11+ add the
# Triton provider, ExecutionPlan, and scheduler on top of this surface.
# Task 12 adds ExecutionPlan + JSON (de)serialization + the topological
# scheduler on top of the ABI surfaces.
add_library(forgestack_runtime STATIC
    lib/abi_types.cpp
    lib/tensor_descriptor.cpp
    lib/artifact_descriptor.cpp
    lib/buffer.cpp
    lib/event.cpp
    lib/result.cpp
    lib/forge_rt.cpp
    lib/providers/triton_provider.cpp
    lib/providers/astra_provider.cpp
    lib/execution_plan.cpp
    lib/liveness.cpp
    lib/buffer_reuse.cpp
    lib/memory_planner.cpp
    lib/scheduler.cpp
)

target_include_directories(forgestack_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # TritonProvider::launch calls into backends/triton/cpu/. The kernel
    # header is a self-contained .hpp pair with no external deps, so the
    # runtime can reach in directly without dragging the kernel CMake
    # target into the runtime's link line.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../backends/triton/cpu>
)

target_compile_features(forgestack_runtime PUBLIC cxx_std_20)

target_link_libraries(forgestack_runtime PUBLIC nlohmann_json::nlohmann_json)

add_library(forgestack::runtime ALIAS forgestack_runtime)

enable_testing()
add_subdirectory(tests)
