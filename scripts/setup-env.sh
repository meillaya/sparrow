#!/usr/bin/env bash
# scripts/setup-env.sh — bootstrap the ForgeStack Phase 1 dev environment.
#
# Two-tier strategy (matches ForgeStack env constraints 2026-07-18):
#   1. Try `devenv shell` (canonical; works on hosts with trusted-user Nix config)
#   2. Fall back to a uv-managed venv at .venv (works everywhere; what CI / containers use)
#
# After this script returns 0, every command in the plan should run inside
# the activated environment. For devenv users: `devenv shell`. For venv users:
# `source .venv/bin/activate`. CI uses scripts/ci.sh which sources the venv
# and then runs the test suites.
#
# Toolchain (2026-07-18): Python deps are installed via `uv pip install`,
# lint/format via `ruff`, type-checking via `ty`. Both ruff and ty are
# installed globally with `uv tool install` so the binaries are on PATH
# (`~/.local/bin`); see .omo/notepads/forgestack/learnings.md §(d).
#
# Usage:
#   bash scripts/setup-env.sh        # bootstrap
#   source .venv/bin/activate        # activate for current shell (venv users)
#   devenv shell                     # activate for current shell (devenv users)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---------------------------------------------------------------------------
# Tier 1: prefer `devenv shell` when the host actually allows it. This path
# delegates package management to Nix and is the canonical Phase 1 surface.
# ---------------------------------------------------------------------------
if command -v devenv >/dev/null 2>&1; then
  if nix show-config 2>/dev/null | grep -qE '^trusted-users = .*\b'$(id -un)'\b'; then
    echo "[setup-env] devenv shell usable on this host (current user is Nix trusted-user); skipping venv bootstrap"
    echo "[setup-env] OK — activate with: devenv shell"
    exit 0
  else
    echo "[setup-env] devenv present but current user is not a Nix trusted-user; using venv fallback"
  fi
fi

# ---------------------------------------------------------------------------
# Tier 2: uv-managed venv fallback. Mirrors the previous pip+venv behaviour
# but uses `uv venv` to create the environment and `uv pip install` to put
# Python deps into it. The system libstdc++ is reached via LD_LIBRARY_PATH
# (same quirk as the pip version) and `uv venv --system-site-packages`
# keeps access to system Python packages.
# ---------------------------------------------------------------------------

# Step 0: require uv. It ships as a single static binary; if it's missing we
# bail loudly rather than falling back to plain pip (the whole point of this
# script is to migrate off pip).
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup-env] ERROR: uv not found on PATH. Install from https://docs.astral.sh/uv/ or" >&2
  echo "              via devenv.nix (uv ruff ty in `packages`), then re-run." >&2
  exit 1
fi

# Step 1: create the venv if missing. `--system-site-packages` lets the venv
# see system-installed Python packages (the LD_LIBRARY_PATH workaround needs
# libstdc++ from /usr/lib at dlopen time, not Python packages). `--relocatable`
# copies the interpreter into the venv instead of symlinking it back to the
# Nix store, which keeps the venv usable after `devenv shell` exits.
if [[ ! -d "$VENV" ]]; then
  echo "[setup-env] creating venv at $VENV (uv $(uv --version | awk '{print $2}'))"
  uv venv "$VENV" \
    --python "$PYTHON_BIN" \
    --system-site-packages \
    --relocatable
else
  echo "[setup-env] reusing existing venv at $VENV"
fi

# Step 2: install Phase 1 Python dependencies via `uv pip install`. We pin
# xDSL >=0.68 for the dialect API; torch CPU-only wheels are ~250 MB vs
# ~3 GB CUDA. `uv pip install` is idempotent — running it again is a no-op
# when the lock state already matches the requested set.
echo "[setup-env] installing Python deps into $VENV via uv pip (this may take ~5 minutes on first run)"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib:/usr/lib/x86_64-linux-gnu" \
  uv pip install --python "$VENV/bin/python3" \
    "torch" \
    "xdsl>=0.68" \
    "pytest" \
    "lit" \
    "numpy" \
    "pybind11"

# Step 3: verify the install. We require torch + xdsl to both import and the
# xDSL Context to be constructible.
echo "[setup-env] verifying imports"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib:/usr/lib/x86_64-linux-gnu" \
  "$VENV/bin/python3" - <<'PY'
import sys
import torch
import xdsl
from xdsl.context import Context
print(f"  torch  {torch.__version__}")
print(f"  xdsl   {xdsl.__version__}")
print(f"  python {sys.version.split()[0]}")
ctx = Context()
print(f"  Context OK ({type(ctx).__name__})")
PY

# Step 4: wire up native build tools. On hosts where devenv shell cannot
# evaluate (this host), symlink the binaries the Nix daemon already fetched
# from nixpkgs into the venv's bin/ directory so they're on PATH when the
# venv is activated — avoids needing a system-wide install.
echo "[setup-env] wiring native build tools into $VENV/bin"
NIX_CMAKE=$( (find /nix/store -maxdepth 3 -path '*-cmake-*/bin/cmake' -type f -executable 2>/dev/null || true) | grep -v 'python3.12-cmake' | head -1 || true)
NIX_NINJA=$( (find /nix/store -maxdepth 3 -path '*-ninja-*/bin/ninja' -type f -executable 2>/dev/null || true) | grep -v 'python3.12-ninja' | head -1 || true)
NIX_NLOHMANN_INCLUDE=$( (find /nix/store -maxdepth 5 -path '*/include/nlohmann/json.hpp' 2>/dev/null || true) | head -1 | xargs -r dirname | xargs -r dirname)
for pair in "cmake:$NIX_CMAKE" "ninja:$NIX_NINJA"; do
  name="${pair%%:*}"; src="${pair#*:}"
  if [[ -n "$src" && -x "$src" && ! -e "$VENV/bin/$name" ]]; then
    ln -s "$src" "$VENV/bin/$name"
  fi
done
mkdir -p "$VENV/include"
if [[ -n "$NIX_NLOHMANN_INCLUDE" && -d "$NIX_NLOHMANN_INCLUDE/nlohmann" ]]; then
  cp -rn "$NIX_NLOHMANN_INCLUDE/nlohmann" "$VENV/include/"
fi

# Step 5: verify native tools (cmake, ninja, gtest, nlohmann_json). These come
# from system packages on most distros; we probe by name and header existence.
echo "[setup-env] verifying native toolchain"
export PATH="$VENV/bin:$PATH"
for tool in cmake ninja clang++ g++; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "  $tool  $(command -v "$tool")"
  else
    echo "  $tool  MISSING" >&2
    exit 1
  fi
done

if [[ -f /usr/include/gtest/gtest.h ]] || [[ -f /usr/local/include/gtest/gtest.h ]] || [[ -f "$VENV/include/gtest/gtest.h" ]]; then
  echo "  gtest  headers found"
else
  echo "  gtest  headers NOT found — install libgtest-dev or rely on devenv"
fi
if [[ -f /usr/include/nlohmann/json.hpp ]] || [[ -f /usr/local/include/nlohmann/json.hpp ]] || [[ -f "$VENV/include/nlohmann/json.hpp" ]]; then
  echo "  nlohmann_json  headers found"
else
  echo "  nlohmann_json  headers NOT found — install nlohmann-json3-dev or rely on devenv"
fi

# Step 6: warn if ruff / ty are not on PATH (they ship globally via
# `uv tool install`, not via the venv). The Phase 1 plan calls them in CI;
# missing here is a soft failure so setup-env.sh stays usable on hosts
# without ruff/ty.
for tool in uv ruff ty; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "  $tool  $(command -v "$tool") ($($tool --version 2>&1 | head -1))"
  else
    echo "  $tool  MISSING — install with: uv tool install $tool" >&2
  fi
done

echo "[setup-env] OK — activate with: source $VENV/bin/activate"