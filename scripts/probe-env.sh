#!/usr/bin/env bash
# scripts/probe-env.sh — verify every tool the ForgeStack plan needs is present.
# Run from any working directory; resolves paths relative to repo root.
#
# This is the canonical Phase 1 verification command (Task 2 of the plan).
# Exits 0 only if every tool/version probe passes. The output is captured by
# the task's evidence path under .omo/evidence/task-2-probe.log.
#
# 18-probe layout (2026-07-18, after the pip→uv migration):
#   1-5:  native toolchain (cmake, ninja, clang++, g++, pkg-config)
#   6-7:  C++ test deps headers (gtest, nlohmann_json)
#   8:    .venv directory
#   9:    torch import
#   10:   xdsl import
#   11-14: pytest, numpy, pybind11, lit imports
#   15:   lit FileCheck binary on PATH
#   16:   uv on PATH
#   17:   ruff on PATH
#   18:   ty on PATH

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib:/usr/lib/x86_64-linux-gnu"
export PATH="$VENV/bin:$PATH"

PASS=0
FAIL=0
NOTE=0

probe() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    local out
    out=$("$@" 2>&1 | head -1 || true)
    printf "  [PASS] %-22s %s\n" "$label" "$out"
    PASS=$((PASS+1))
  else
    printf "  [FAIL] %-22s command not found: %s\n" "$label" "$*"
    FAIL=$((FAIL+1))
  fi
}

note() {
  local label="$1"; shift
  printf "  [NOTE] %-22s %s\n" "$label" "$*"
  NOTE=$((NOTE+1))
}

echo "=== ForgeStack Phase 1 environment probe ==="
echo "root: $ROOT"
echo

# --- Native toolchain ---
echo "-- Native toolchain --"
for tool in cmake ninja clang++ g++ pkg-config; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf "  [PASS] %-22s %s\n" "$tool" "$(command -v "$tool")"
    PASS=$((PASS+1))
  else
    printf "  [FAIL] %-22s not found\n" "$tool"
    FAIL=$((FAIL+1))
  fi
done
cmake --version 2>/dev/null | head -1 | sed 's/^/         /'
ninja --version 2>/dev/null | sed 's/^/         /'

# --- gtest / nlohmann_json headers ---
echo
echo "-- C++ test deps --"
if [[ -f /usr/include/gtest/gtest.h ]]; then
  printf "  [PASS] %-22s /usr/include/gtest/gtest.h\n" "gtest headers"
  PASS=$((PASS+1))
elif [[ -f /usr/local/include/gtest/gtest.h ]]; then
  printf "  [PASS] %-22s /usr/local/include/gtest/gtest.h\n" "gtest headers"
  PASS=$((PASS+1))
elif [[ -f "$VENV/include/gtest/gtest.h" ]]; then
  printf "  [PASS] %-22s %s/include/gtest/gtest.h\n" "gtest headers" "$VENV"
  PASS=$((PASS+1))
else
  printf "  [FAIL] %-22s not found (need libgtest-dev)\n" "gtest headers"
  FAIL=$((FAIL+1))
fi
if [[ -f /usr/include/nlohmann/json.hpp ]]; then
  printf "  [PASS] %-22s /usr/include/nlohmann/json.hpp\n" "nlohmann_json"
  PASS=$((PASS+1))
elif [[ -f /usr/local/include/nlohmann/json.hpp ]]; then
  printf "  [PASS] %-22s /usr/local/include/nlohmann/json.hpp\n" "nlohmann_json"
  PASS=$((PASS+1))
elif [[ -f "$VENV/include/nlohmann/json.hpp" ]]; then
  printf "  [PASS] %-22s %s/include/nlohmann/json.hpp\n" "nlohmann_json" "$VENV"
  PASS=$((PASS+1))
else
  printf "  [FAIL] %-22s not found (need nlohmann-json3-dev)\n" "nlohmann_json"
  FAIL=$((FAIL+1))
fi

# --- Python + venv ---
echo
echo "-- Python + xDSL + torch --"
if [[ ! -d "$VENV" ]]; then
  printf "  [FAIL] %-22s missing — run scripts/setup-env.sh\n" ".venv"
  FAIL=$((FAIL+1))
else
  printf "  [PASS] %-22s %s\n" ".venv" "$VENV"
  PASS=$((PASS+1))
  if "$VENV/bin/python3" -c "import torch, xdsl; from xdsl.context import Context; Context()" >/dev/null 2>&1; then
    TORCH_VER=$("$VENV/bin/python3" -c "import torch; print(torch.__version__)")
    XDSL_VER=$("$VENV/bin/python3" -c "import xdsl; print(xdsl.__version__)")
    printf "  [PASS] %-22s %s\n" "torch" "$TORCH_VER"
    printf "  [PASS] %-22s %s\n" "xdsl"  "$XDSL_VER"
    PASS=$((PASS+2))
  else
    printf "  [FAIL] %-22s torch/xdsl/Context import failed\n" "python imports"
    FAIL=$((FAIL+1))
  fi
  for mod in pytest numpy pybind11 lit; do
    if "$VENV/bin/python3" -c "import $mod" >/dev/null 2>&1; then
      printf "  [PASS] %-22s importable\n" "$mod"
      PASS=$((PASS+1))
    else
      printf "  [FAIL] %-22s import failed\n" "$mod"
      FAIL=$((FAIL+1))
    fi
  done
fi

# --- lit (FileCheck) ---
echo
echo "-- lit FileCheck --"
if [[ -x "$VENV/bin/lit" ]]; then
  printf "  [PASS] %-22s %s\n" "lit" "$VENV/bin/lit"
  PASS=$((PASS+1))
else
  printf "  [FAIL] %-22s not found in venv\n" "lit"
  FAIL=$((FAIL+1))
fi

# --- uv + ruff + ty (Phase 1 dev toolchain, 2026-07-18) ---
# These three are installed globally via `uv tool install` and live on
# ~/.local/bin (outside the venv). They are NOT in the venv because CI/IDE
# integrations call them as standalone binaries.
echo
echo "-- Python dev toolchain (uv + ruff + ty) --"
for tool in uv ruff ty; do
  if command -v "$tool" >/dev/null 2>&1; then
    VER=$("$tool" --version 2>&1 | head -1)
    printf "  [PASS] %-22s %s (%s)\n" "$tool" "$(command -v "$tool")" "$VER"
    PASS=$((PASS+1))
  else
    printf "  [FAIL] %-22s not found — install with: uv tool install %s\n" "$tool" "$tool"
    FAIL=$((FAIL+1))
  fi
done

echo
echo "=== Result: $PASS pass, $FAIL fail, $NOTE note ==="
exit "$FAIL"