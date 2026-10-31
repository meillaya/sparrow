# bindings/ — Python bridge for the fused MLP vertical slice.
if(NOT SPARROW_BUILD_PYTHON)
    return()
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_LOOKUP
)
if(NOT PYBIND11_LOOKUP EQUAL 0)
    message(FATAL_ERROR "python -m pybind11 --cmakedir failed; install pybind11")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_sparrow_rt python/module.cpp)
target_link_libraries(_sparrow_rt PRIVATE sparrow::runtime sparrow::triton_cpu)
target_compile_features(_sparrow_rt PRIVATE cxx_std_20)
set_target_properties(_sparrow_rt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
