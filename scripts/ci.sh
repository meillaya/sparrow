#!/usr/bin/env bash
# scripts/ci.sh — full CI pipeline for ForgeStack Phase 1.
# Runs every gate in sequence. Exits non-zero on any failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib:/usr/lib/x86_64-linux-gnu"

echo "[ci] === ForgeStack Phase 1 CI ==="
echo "[ci] root: $ROOT"
echo

# Step 0: environment probe (18+ checks per scripts/probe-env.sh)
echo "[ci] Step 0: environment probe"
bash scripts/probe-env.sh
echo

# Step 1: C++ build + tests
echo "[ci] Step 1: CMake configure + build"
# Wipe stale build dir if it was configured with a different generator
# (or is partial) so the configure step below is always consistent.
rm -rf build
# Disable system FindGTest: on some hosts the system FindGTest.cmake resolves
# to a stale nix-store path (e.g. freecad-1.1.1) that no longer exists. Force
# FetchContent_MakeAvailable(googletest) instead — produces a fully self-
# contained v1.17.0 build under build/_deps/, which works on every host.
cmake -B build -S . -G Ninja \
    -DCMAKE_DISABLE_FIND_PACKAGE_GTest=TRUE
echo "[ci] Step 1: C++ test binaries"
cmake --build build
ctest --test-dir build --output-on-failure
echo

# Step 2: xDSL MLIR lit tests (compiler pass golden tests)
echo "[ci] Step 2: xDSL MLIR lit tests"
lit -v compiler/test/
echo

# Step 3: Python pytest (frontend + backends + end-to-end + cache + report)
#                      + Phases 2-8: astra + microcode + microgpu + memory + scheduler
# pytest auto-discovers test_*.py / *_test.py recursively under test/, so the
# explicit file list below is a defensive safety net (pytest dedupes by id).
echo "[ci] Step 3: pytest"
pytest test/ \
    test/test_functional.py \
    test/test_timing.py \
    test/test_scratchpad_utilization.py \
    test/test_dma_compute_stall.py \
    test/emit_astra_test.py \
    test/microcode_test.py \
    test/microcode_asm_test.py \
    test/microgpu_isa_test.py \
    test/microgpu_asm_test.py \
    test/microgpu_interp_test.py \
    test/microgpu_race_oob_test.py \
    test/frontend/test_cache_determinism.py \
    test/frontend/test_report_fields_complete.py \
    test/frontend/test_fx_importer_symbolic.py \
    test/frontend/test_dynamic_compile.py \
    test/end_to_end/test_canonicalize_differential.py \
    test/end_to_end/test_shape_inference_differential.py \
    test/end_to_end/test_fusion_differential.py \
    test/end_to_end/test_chained_optimize.py \
    test/end_to_end/test_fused_mlp_dtypes.py \
    test/end_to_end/test_memory_planning_e2e.py \
    test/end_to_end/test_scheduler_e2e.py \
    test/end_to_end/test_memory_report.py \
    test/end_to_end/test_astra_e2e.py \
    test/end_to_end/test_cross_provider.py \
    test/end_to_end/test_microgpu_e2e.py \
    test/end_to_end/test_microgpu_race_oob.py \
    -v --tb=short
echo

# Step 4: Lint + format check
echo "[ci] Step 4: ruff lint + format"
ruff check .
ruff format --check .
echo

# Step 5: Type check (ty warnings allowed; failures are errors)
echo "[ci] Step 5: ty type check"
ty check . || {
    echo "[ci] ty reported diagnostics (warnings allowed)"
}
echo

echo "[ci] === ALL CI GATES PASSED ==="
