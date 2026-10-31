#!/usr/bin/env bash
# scripts/probe-env.sh — report whether the tools the gate needs exist.
#
# Works inside a `devenv shell` (where the packages are on PATH) and
# inside a uv-managed `.venv` (where the Python packages live in the
# venv). Exits non-zero when a required tool or import is missing.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"
if [[ -x "$VENV/bin/python3" ]]; then
    PYTHON="$VENV/bin/python3"
fi
export PATH="$VENV/bin:$PATH"
if [[ "$(uname -s)" == "Linux" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib:/usr/lib/x86_64-linux-gnu"
fi

PASS=0
FAIL=0
NOTE=0

pass() {
    printf "  [PASS] %-24s %s\n" "$1" "$2"
    PASS=$((PASS + 1))
}

fail() {
    printf "  [FAIL] %-24s %s\n" "$1" "$2"
    FAIL=$((FAIL + 1))
}

note() {
    printf "  [NOTE] %-24s %s\n" "$1" "$2"
    NOTE=$((NOTE + 1))
}

echo "=== Sparrow environment probe ==="
echo "root:   $ROOT"
echo "python: $PYTHON"
echo

echo "-- Native toolchain --"
for tool in cmake ninja; do
    if command -v "$tool" >/dev/null 2>&1; then
        pass "$tool" "$(command -v "$tool")"
    else
        fail "$tool" "not found on PATH"
    fi
done
if command -v clang++ >/dev/null 2>&1; then
    pass "clang++" "$(command -v clang++)"
elif command -v g++ >/dev/null 2>&1; then
    pass "g++" "$(command -v g++)"
else
    fail "C++ compiler" "neither clang++ nor g++ found"
fi
if command -v pkg-config >/dev/null 2>&1; then
    pass "pkg-config" "$(command -v pkg-config)"
else
    note "pkg-config" "missing; CMake will use find_package paths"
fi

echo
echo "-- C++ test dependencies --"
if [[ -f /usr/include/gtest/gtest.h || -f /usr/local/include/gtest/gtest.h || -f "$VENV/include/gtest/gtest.h" ]]; then
    pass "gtest headers" "found"
else
    note "gtest headers" "not in the usual paths; CMake can fetch googletest"
fi
if [[ -f /usr/include/nlohmann/json.hpp || -f /usr/local/include/nlohmann/json.hpp || -f "$VENV/include/nlohmann/json.hpp" ]]; then
    pass "nlohmann_json" "found"
else
    note "nlohmann_json" "not in the usual paths; CMake can fetch it"
fi

echo
echo "-- Python runtime --"
if [[ -d "$VENV" ]]; then
    pass ".venv" "$VENV"
else
    note ".venv" "absent; using $PYTHON from PATH"
fi
for module in torch xdsl pytest numpy pybind11; do
    if "$PYTHON" -c "import $module" >/dev/null 2>&1; then
        version="$("$PYTHON" -c "import $module; print(getattr($module, '__version__', 'installed'))" 2>/dev/null)"
        pass "$module" "$version"
    else
        fail "$module" "import failed with $PYTHON"
    fi
done

echo
echo "-- Test and build tools --"
if command -v lit >/dev/null 2>&1; then
    pass "lit" "$(command -v lit)"
else
    fail "lit" "not found; install the lit package"
fi
if command -v FileCheck >/dev/null 2>&1; then
    pass "FileCheck" "$(command -v FileCheck)"
else
    fail "FileCheck" "not found; install LLVM or add it to PATH"
fi
for tool in ruff ty; do
    if command -v "$tool" >/dev/null 2>&1; then
        pass "$tool" "$("$tool" --version 2>&1 | head -1)"
    else
        fail "$tool" "not found; install it or use the dev shell"
    fi
done
if command -v uv >/dev/null 2>&1; then
    pass "uv" "$(uv --version 2>&1 | head -1)"
else
    note "uv" "missing; only needed for the venv bootstrap"
fi

echo
echo "=== Result: $PASS pass, $FAIL fail, $NOTE note ==="
exit "$FAIL"
