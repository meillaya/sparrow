cmake_minimum_required(VERSION 3.25)
project(forgestack
    VERSION 0.1.0
    DESCRIPTION "MLIR-based heterogeneous ML compiler and runtime (Phase 1 vertical slice)"
    LANGUAGES CXX
)

# C++20 — required for std::span, std::expected, concepts
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings — strict for Phase 1 so we surface real bugs early
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

# Build with Ninja if available; fall back to Make
if(NOT CMAKE_GENERATOR)
    if(EXISTS /home/mei/projects/forgestack/.venv/bin/ninja)
        set(CMAKE_GENERATOR "Ninja" CACHE STRING "Build generator" FORCE)
    endif()
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
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Subprojects
add_subdirectory(runtime)
# compiler/ and backends/ come online in later tasks; gate to avoid empty-subdir errors
option(FORGESTACK_BUILD_COMPILER "Build the ForgeMLIR Python compiler package" OFF)
option(FORGESTACK_BUILD_BACKENDS "Build the target backend libraries" OFF)
if(FORGESTACK_BUILD_COMPILER)
    add_subdirectory(compiler)
endif()
if(FORGESTACK_BUILD_BACKENDS)
    add_subdirectory(backends)
endif()
# Triton backend Path A CPU kernel (spec §7 Path A) — Task 11 wires the
# TritonDeviceProvider on top of this. Not gated: the kernel is required
# by the runtime today, so it has to land in the default build.
add_subdirectory(backends/triton/cpu)