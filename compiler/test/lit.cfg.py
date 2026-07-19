# lit.cfg.py — ForgeStack compiler/test lit configuration.
#
# Runs the FileCheck-driven `.mlir` tests under `compiler/test/`. The
# `test_source_root = HERE` setting below makes lit recursively
# discover every ``.mlir`` file under the test tree — currently
# ``Dialect/{Graph,Tensor,Schedule}``, ``Conversion/GraphToTensor``,
# and (Phase 2) ``Transforms/{Canonicalize,ShapeInference,Fuse}``.
# Each test file uses `// RUN:` lines to invoke `forge-opt` and pipe
# its output through FileCheck.

# ruff: noqa: F821  (lit injects `config` at load time; F821 is the
# "undefined name" rule that fires on `config.x = ...` assignments below)

from __future__ import annotations

import os
import sys
from pathlib import Path

import lit.formats

HERE = Path(__file__).resolve().parent
COMPILER_ROOT = HERE.parent
REPO_ROOT = COMPILER_ROOT.parent
FORGE_OPT = COMPILER_ROOT / "tools" / "forge-opt" / "forge_opt.py"

# Lit injects `config` (a lit.TestingConfig.TestingConfig instance) into this
# module's globals at load time. We declare it as an opaque `object` so
# ruff + ty both have a symbol to point at; the real attribute set is
# populated by lit below.
config: object  # type: ignore[unresolved-reference]


def _resolve_python() -> str:
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python3",
        REPO_ROOT / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


config.name = "forgestack-compiler"
config.suffixes = [".mlir"]
config.test_format = lit.formats.ShTest()
config.test_source_root = str(HERE)
config.test_exec_root = str(HERE)
config.environment = dict(os.environ)
config.environment.update(
    {
        "TZ": "UTC",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "UTF-8",
    }
)
config.pipefail = True
config.excludes = []
config.python_executable = _resolve_python()

# `%forge-opt` runs our driver on the test file; `%FileCheck` uses the
# system binary. Phase 1 installs FileCheck via apt / Nix; if it is not on
# PATH, lit's resolver surfaces an UNSUPPORTED status rather than silently
# passing the test.
config.substitutions = [
    ("%forge-opt", f'"{config.python_executable}" "{FORGE_OPT}"'),
    ("%FileCheck", "FileCheck"),
]
