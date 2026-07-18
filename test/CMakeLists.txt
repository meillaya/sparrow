# test/ — End-to-end Python and C++ integration tests for Phase 1.
#
# Phase 1 currently has no real tests (those land in Task 13+). The directory
# exists so pytest has a `testpaths` target and CTest can pick up GoogleTest
# binaries once the runtime + compiler are wired up.
#
# No CMake targets are defined here yet; this file is intentionally a
# near-empty stub. Add `add_test(...)` calls as suites come online.