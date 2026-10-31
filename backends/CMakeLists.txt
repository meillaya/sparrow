# backends/ — Target backend libraries (CPU, GPU, etc.). Skeleton placeholder.
#
# This directory will host per-target backend implementations of the
# ForgeRT runtime API. Until SPARROW_BUILD_BACKENDS is enabled (top-level
# option), this file is not included in the build graph.
if(SPARROW_BUILD_BACKENDS)
    # Real build steps land in a later task.
    message(STATUS "backends/ subdirectory included (SPARROW_BUILD_BACKENDS=ON)")
endif()