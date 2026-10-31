#!/usr/bin/env bash
# scripts/ci.sh — the local Sparrow gate.
#
# Runs the environment probe, the C++ build and ctest, lit, pytest, ruff,
# and ty. Any failure exits non-zero. Optional gates do not get a
# fallback: if the tool is missing, the probe says so before the gate runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
    PYTHON="$ROOT/.venv/bin/python3"
fi
export PATH="$ROOT/.venv/bin:$PATH"
export SPARROW_BUILD_DIR="$ROOT/build"

echo "[ci] === Sparrow gate ==="
echo "[ci] root: $ROOT"
echo "[ci] python: $PYTHON"
echo

echo "[ci] Step 0: environment probe"
bash scripts/probe-env.sh
echo

echo "[ci] Step 1: CMake configure + build + ctest"
rm -rf build
cmake -B build -S . -G Ninja -DPython_EXECUTABLE="$PYTHON"
cmake --build build
ctest --test-dir build --output-on-failure --no-tests=error
echo

echo "[ci] Step 2: xDSL lit tests"
if [[ -x "$ROOT/.venv/bin/lit" ]]; then
    "$ROOT/.venv/bin/lit" -v compiler/test/
else
    lit -v compiler/test/
fi
echo

echo "[ci] Step 3: pytest"
export PYTHONPATH="$ROOT/build/python:${PYTHONPATH:-}"
"$PYTHON" -m pytest -q test/
echo

echo "[ci] Step 4: ruff lint + format"
ruff check .
ruff format --check .
echo

echo "[ci] Step 5: ty type check"
ty check . --python "$PYTHON"
echo

if [[ "${SPARROW_SANITIZERS:-0}" == "1" ]]; then
    echo "[ci] Step 6: sanitizer builds"
    bash scripts/ci-sanitizers.sh
    echo
fi

echo "[ci] === ALL GATES PASSED ==="
