"""Chained-optimize op-count pytest (Phase 2, Task 12).

Asserts that the ``forge-opt --action=optimize`` driver — whose default
pass pipeline is ``canonicalize,shape-inference,fuse`` — runs end-to-end
on a FusedMLP ``forge.graph`` IR and the resulting ``func.func`` body
contains at most 2 ``forge.tensor.*`` ops (i.e. ``gemm +
bias-bearing fused_epilogue``; per the Wave-1 deviation note in
:mod:`compiler.lib.Transforms.Fuse`, fusion reduces 3 → 2 not 3 → 1).

The driver is invoked via :mod:`subprocess` against
``/home/mei/projects/forgestack/compiler/tools/forge-opt/forge_opt.py``
so the test exercises the same entry point end-users will hit from the
CLI. The textual output is regex-scanned for the ``func.func`` body and
``forge.tensor.\\w+`` occurrences; the count is bounded by the
post-fusion canonical form (gemm + bias-bearing fused_epilogue).

Pipeline contract pinned by this test
-------------------------------------

* The driver must successfully parse the FusedMLP ``forge.graph`` IR,
  run the chained pass pipeline, re-verify the module after every
  pass, and print back the resulting textual MLIR.
* The ``func.func`` body's ``forge.tensor.*`` op count must be ≤ 2;
  the chained pipeline is engineered to elide exactly one op from the
  triple.
"""

from __future__ import annotations

import re
import subprocess

import pytest
import torch
from forge_torch.backend import compile
from forge_torch.conftest import FusedMLP

SHAPES = [
    # (B, S, K, N) — Phase 1 canonical (spec §19)
    (2, 512, 4096, 11008),
    # Small smoke
    (1, 128, 2048, 4096),
    # Tiny smoke
    (4, 64, 512, 1024),
]


def _run_chained_optimize(mlir_text: str) -> str:
    """Invoke ``forge-opt --action=optimize`` on ``mlir_text`` and return stdout.

    Mirrors the exact command from the Task-12 deliverable: hard-coded
    paths to ``/home/mei/projects/forgestack/.venv/bin/python3`` and
    the ``forge_opt.py`` driver, stdin input, captured stdout.
    """
    proc = subprocess.run(
        [
            "/home/mei/projects/forgestack/.venv/bin/python3",
            "/home/mei/projects/forgestack/compiler/tools/forge-opt/forge_opt.py",
            "-",
            "--action=optimize",
        ],
        input=mlir_text,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _count_forge_tensor_ops_in_func_body(optimized_mlir: str) -> list[str]:
    """Return the list of ``forge.tensor.\\w+`` op names inside the ``func.func`` body.

    The textual MLIR is parsed with a regex that captures everything
    between the ``func.func`` opening brace and its matching closing
    brace. ``re.findall`` on the body then yields every distinct
    ``forge.tensor.<name>`` occurrence in source order. The list length
    is what Task 12 bounds (≤ 2).
    """
    # Greedy enough to span the whole body but not the surrounding
    # builtin.module { … } wrapper; the `[^}]` exclusion inside the
    # placeholder list is not necessary because MLIR doesn't nest
    # `}`-bearing constructs that the regex would over-consume.
    match = re.search(
        r"func\.func\s+\S+\s*\([^)]*\)[^{]*\{(?P<body>.*?)\n\s*\}",
        optimized_mlir,
        re.DOTALL,
    )
    assert match is not None, (
        f"could not locate func.func body in optimized MLIR:\n{optimized_mlir}"
    )
    body = match.group("body")
    return re.findall(r"forge\.tensor\.\w+", body)


@pytest.mark.parametrize("B,S,K,N", SHAPES)
def test_chained_optimize_collapses_fused_mlp_to_at_most_two_ops(
    B: int, S: int, K: int, N: int
) -> None:
    """Pipeline reduce: the chained ``canonicalize → shape-inference → fuse``
    pipeline leaves at most 2 ``forge.tensor.*`` ops in the FusedMLP IR.

    For each of the 3 Phase-1 shapes, builds a FusedMLP, runs the
    chained pass pipeline via ``forge-opt --action=optimize``, and
    asserts the ``func.func`` body contains ≤ 2 ``forge.tensor.*`` ops
    — i.e. the canonical post-fusion form is ``gemm +
    bias-bearing fused_epilogue`` (per the 3 → 2 deviation noted in
    :mod:`compiler.lib.Transforms.Fuse`).
    """
    torch.manual_seed(0)
    model = FusedMLP()
    x = torch.randn(B, S, K, dtype=torch.float32)
    w = torch.randn(K, N, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)

    compiled = compile(model, [x, w, b])
    optimized_mlir = _run_chained_optimize(compiled.mlir_text)

    ops = _count_forge_tensor_ops_in_func_body(optimized_mlir)
    assert len(ops) <= 2, (
        f"chained optimize did not collapse to ≤ 2 forge.tensor.* ops; "
        f"got {len(ops)}: {ops}\n--- optimized MLIR ---\n{optimized_mlir}"
    )
