# compiler/ — ForgeMLIR Python compiler package. Skeleton placeholder.
#
# This directory will house the xDSL/MLIR Python compiler front-end. Until
# FORGESTACK_BUILD_COMPILER is enabled (top-level option), this file is not
# included in the build graph, so it is safe to leave it as a stub.
if(FORGESTACK_BUILD_COMPILER)
    # Real build steps land in a later task (see .omo/plans/forgestack.md).
    message(STATUS "compiler/ subdirectory included (FORGESTACK_BUILD_COMPILER=ON)")
endif()