cmake_minimum_required(VERSION 3.25)
project(sparrow
    VERSION 0.3.0
    DESCRIPTION "MLIR-based heterogeneous ML compiler and runtime (Phase 1 vertical slice)"
    LANGUAGES CXX
)

# C++20 — required for std::span, std::expected, concepts
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings. `-Werror` is applied to our own targets below, not to
# FetchContent dependencies such as googletest.
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Optional sanitizers. Address and thread sanitizers are mutually
# exclusive; undefined-behavior checking can be combined with address.
option(SPARROW_SANITIZE_ADDRESS "Build with AddressSanitizer" OFF)
option(SPARROW_SANITIZE_UNDEFINED "Build with UndefinedBehaviorSanitizer" OFF)
option(SPARROW_SANITIZE_THREAD "Build with ThreadSanitizer" OFF)
if(SPARROW_SANITIZE_ADDRESS AND SPARROW_SANITIZE_THREAD)
    message(FATAL_ERROR "SPARROW_SANITIZE_ADDRESS and SPARROW_SANITIZE_THREAD are mutually exclusive")
endif()
if(SPARROW_SANITIZE_ADDRESS)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(SPARROW_SANITIZE_UNDEFINED)
    add_compile_options(-fsanitize=undefined -fno-sanitize-recover=undefined)
    add_link_options(-fsanitize=undefined)
endif()
if(SPARROW_SANITIZE_THREAD)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

# Testing — GoogleTest via CMake's FetchContent (works without system gtest)
include(CTest)
enable_testing()

# Find system packages (gtest + nlohmann_json); fall back to FetchContent
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        URL_HASH SHA256=40d4ec942217dcc84a9ebe2a68584ada7d4a33a8ee958755763278ea1c5e18ff
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Subprojects. The Triton CPU kernel is a dependency of the runtime's
# TritonProvider, so it is defined first.
add_subdirectory(backends/triton/cpu)
add_subdirectory(runtime)
# compiler/ and backends/ come online in later tasks; gate to avoid empty-subdir errors
option(SPARROW_BUILD_PYTHON "Build the pybind11 runtime bridge" ON)
option(SPARROW_BUILD_COMPILER "Build the Python compiler package" OFF)
option(SPARROW_BUILD_BACKENDS "Build the target backend libraries" OFF)

# Deprecated spellings kept for one release.
if(DEFINED FORGESTACK_BUILD_COMPILER)
    set(SPARROW_BUILD_COMPILER "${FORGESTACK_BUILD_COMPILER}")
endif()
if(DEFINED FORGESTACK_BUILD_BACKENDS)
    set(SPARROW_BUILD_BACKENDS "${FORGESTACK_BUILD_BACKENDS}")
endif()

if(SPARROW_BUILD_COMPILER)
    add_subdirectory(compiler)
endif()
if(SPARROW_BUILD_BACKENDS)
    add_subdirectory(backends)
endif()
add_subdirectory(bindings)
