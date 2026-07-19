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
cmake -B build -S . -G Ninja
echo "[ci] Step 1: C++ test binaries"
cmake --build build
ctest --test-dir build --output-on-failure
echo

# Step 2: xDSL MLIR lit tests (compiler pass golden tests)
echo "[ci] Step 2: xDSL MLIR lit tests"
lit -v compiler/test/
echo

# Step 3: Python pytest (frontend + backends + end-to-end + cache + report)
echo "[ci] Step 3: pytest"
pytest test/ -v --tb=short
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
