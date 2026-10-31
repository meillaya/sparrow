#!/usr/bin/env bash
# scripts/ci-sanitizers.sh — sanitizer jobs for the C++ runtime.
#
# Three isolated builds: ASan, UBSan, and TSan. The TSan job covers the
# event and provider paths, which are the only concurrently touched
# runtime surfaces today.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
    PYTHON="$ROOT/.venv/bin/python3"
fi

run_build() {
    local name="$1"
    local option="$2"
    local test_filter="${3:-}"
    local build_dir="build-$name"
    echo "[sanitizers] === $name ==="
    rm -rf "$build_dir"
    cmake -B "$build_dir" -S . -G Ninja \
        -DPython_EXECUTABLE="$PYTHON" \
        "-D$option=ON"
    cmake --build "$build_dir"
    if [[ -n "$test_filter" ]]; then
        ctest --test-dir "$build_dir" --output-on-failure -R "$test_filter"
    else
        ctest --test-dir "$build_dir" --output-on-failure --no-tests=error
    fi
}

run_build asan SPARROW_SANITIZE_ADDRESS
run_build ubsan SPARROW_SANITIZE_UNDEFINED
run_build tsan SPARROW_SANITIZE_THREAD "Event|Scheduler|CpuProvider|TritonProvider"

echo "[sanitizers] === ALL SANITIZER JOBS PASSED ==="
