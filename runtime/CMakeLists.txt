# runtime/ — ForgeRT C++ execution runtime. Filled by Task 10.
# Until then, this directory compiles to an INTERFACE library so the
# top-level build can succeed.
add_library(forgestack_runtime INTERFACE)
target_include_directories(forgestack_runtime INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(forgestack_runtime INTERFACE cxx_std_20)
target_link_libraries(forgestack_runtime INTERFACE nlohmann_json::nlohmann_json)
add_library(forgestack::runtime ALIAS forgestack_runtime)