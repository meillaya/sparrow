"""Dynamic-shape-aware compilation entry point (Task 17, spec §3 + §13).

``dynamic_compile`` mirrors :func:`forge_torch.backend.compile` for the
**symbolic-shape** case: when an FX node carries
``tensor_meta['symbolic_dim']`` (the canonical ``torch.export``
metadata), the per-shape invariant set is folded into the cache key so
a single model can be re-compiled once per distinct constraint signature
and re-served from disk thereafter.

Phase 4 ships this as the user-facing API for dynamic-shape code. The
lowered execution path (Triton codegen, ForgeRT scheduler) isn't
plumbed through yet — :func:`dynamic_compile` returns a
:class:`DynamicCompiledFusedMLP` whose ``__call__`` delegates to eager
PyTorch, just like :class:`forge_torch.backend.CompiledFusedMLP`.
Phase 5+ replaces the stub with the real backend.

Cache layout (per spec §13 example ``dynamic_m/``)
------------------------------------------------
``<cache_root>/dynamic_<hexdigest>/`` for each unique constraint
tuple. The on-disk directory holds:

* ``artifact.json`` — the lowered MLIR + the symbolic-constraint tuple
  used to derive the bucket.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
from torch.fx import GraphModule

from forge_torch.cache import CacheKey, make_cache_key
from forge_torch.fx_importer import FxImporter

_logger = logging.getLogger("forgestack.dynamic_compile")


DEFAULT_CACHE_ROOT = Path(".cache/forgestack")
DYNAMIC_BUCKET_PREFIX = "dynamic_"


def _sym_dim_value(meta_obj: object) -> object | None:
    """Read ``symbolic_dim`` from a torch.fx ``tensor_meta`` (dict or dataclass)."""
    if meta_obj is None:
        return None
    if isinstance(meta_obj, dict):
        return meta_obj.get("symbolic_dim")
    return getattr(meta_obj, "symbolic_dim", None)


def _discover_constraint_signature(gm: GraphModule) -> tuple[str, ...]:
    """Walk ``gm`` and collect one ``"<name> >= 1"`` entry per named symbolic dim.

    Reads ``tensor_meta['symbolic_dim']`` (and the top-level
    ``meta['symbolic_dim']`` fallback) on every FX node. Names are
    de-duplicated across nodes — an FMA-style graph with several uses
    of ``B`` produces one entry, not N — and rendered with the spec §3
    default ``>= 1`` invariant when no explicit bound is given.

    Returns ``()`` for fully-concrete shapes (the canonical "static"
    bucket per spec §13).
    """
    constraints: list[str] = []
    seen: set[str] = set()
    for node in gm.graph.nodes:
        meta = getattr(node, "meta", None) or {}
        sym = _sym_dim_value(meta.get("tensor_meta"))
        if sym is None:
            sym = meta.get("symbolic_dim") if isinstance(meta, dict) else None
        if sym is None:
            continue
        if isinstance(sym, str):
            names = [sym]
        elif isinstance(sym, (list, tuple)):
            names = [s for s in sym if isinstance(s, str)]
        else:
            continue
        for name in names:
            if name and name not in seen:
                seen.add(name)
                constraints.append(f"{name} >= 1")
    return tuple(constraints)


def _discover_artifact_path(root: Path, hex_digest: str) -> Path:
    """Locate the per-bucket directory for ``hex_digest`` under ``root``."""
    return root / f"{DYNAMIC_BUCKET_PREFIX}{hex_digest}"


def _run_forge_opt(mlir_text: str) -> str:
    """Best-effort: lower ``mlir_text`` via ``forge-opt --action=optimize``.

    Falls back to the input text on any failure (driver not on
    :envvar:`PATH`, missing dep, etc.) so the dynamic-compile pipeline
    never hard-errors on the cold path — the caller still gets a
    usable callable, and the artifact written to disk is whatever the
    driver produced (or the original MLIR on failure).
    """
    driver = (
        Path(__file__).resolve().parents[3]
        / "compiler"
        / "tools"
        / "forge-opt"
        / "forge_opt.py"
    )
    if not driver.is_file():
        _logger.debug("forge-opt driver not found at %s; skipping optimize pass", driver)
        return mlir_text
    try:
        proc = subprocess.run(
            [sys.executable, str(driver), "-", "--action=optimize"],
            input=mlir_text.encode("utf-8"),
            capture_output=True,
            check=True,
            env=os.environ.copy(),
        )
        return proc.stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        _logger.debug("forge-opt --action=optimize failed (%s); keeping raw MLIR", exc)
        return mlir_text


@dataclass
class DynamicCompiledFusedMLP:
    """Stub callable returned by :func:`dynamic_compile`.

    Mirrors :class:`forge_torch.backend.CompiledFusedMLP`: carries the
    captured graph, lowered MLIR, and constraint tuple for
    diagnostics. ``__call__`` delegates to eager PyTorch — Phase 5+
    swaps it for a Triton/ForgeRT path.
    """

    model: torch.nn.Module
    example_inputs: list[torch.Tensor]
    mlir_text: str
    cache_key: CacheKey | None = None
    cache_hit: bool = False
    symbolic_constraints: tuple[str, ...] = ()
    constraint_signature: tuple[str, ...] = field(default_factory=tuple)

    #: True iff the instance came back from :func:`dynamic_compile`.
    compiled: ClassVar[bool] = True

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        return self.model(*args)


def dynamic_compile(
    model: torch.nn.Module,
    example_inputs: list[torch.Tensor],
    *,
    target: str = "triton",
    arch: str = "x86_64_v3",
    cache_root: Path | None = None,
) -> Callable:
    """Compile a model with symbolic-shape support, caching per constraint set.

    Captures the FX graph, renders :class:`forge.graph` MLIR (mirrors
    :func:`forge_torch.backend.compile`), inspects every node's
    ``tensor_meta['symbolic_dim']`` to derive the constraint signature,
    and stores the artifact under
    ``<cache_root>/dynamic_<hexdigest>/``. Repeated calls with the
    same constraint signature hit the same bucket (warm path); a
    different signature produces a fresh bucket.

    On the cold path: runs ``forge-opt --action=optimize`` (best-effort
    — falls back to the unoptimized MLIR if the driver is unavailable)
    and persists the artifact + constraint tuple.

    On the warm path: loads the persisted artifact instead of re-running
    the pipeline.
    """
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)

    # Step 1 — capture FX + render MLIR. Accept either a raw ``nn.Module``
    # (we ``symbolic_trace`` it) or a pre-built ``GraphModule`` (skip
    # re-trace so per-placeholder metadata mutations survive into the
    # captured graph and reach the importer — the path tests use to
    # inject ``tensor_meta['symbolic_dim']``).
    if isinstance(model, GraphModule):
        gm = model
    else:
        gm = torch.fx.symbolic_trace(model)
    importer = FxImporter(example_inputs)
    mlir_text = importer.import_graph(gm)

    # Step 2 — derive the constraint signature from FX metadata.
    constraint_signature = _discover_constraint_signature(gm)

    # Step 3 — bucket on the constraint signature's SHA-256 prefix.
    cache_seed = "\n".join(constraint_signature).encode("utf-8")
    bucket_hex = hashlib.sha256(cache_seed).hexdigest()[:16]
    bucket_dir = _discover_artifact_path(root, bucket_hex)
    artifact_path = bucket_dir / "artifact.json"

    cache_hit = artifact_path.is_file()
    if cache_hit:
        artifact = json.loads(artifact_path.read_text())
        optimized_mlir = artifact.get("mlir_text", mlir_text)
    else:
        optimized_mlir = _run_forge_opt(mlir_text)
        artifact = {
            "model_name": model.__class__.__name__,
            "mlir_text": optimized_mlir,
            "symbolic_constraints": list(constraint_signature),
            "schedule": {
                "tile_m": 64,
                "tile_n": 128,
                "tile_k": 32,
                "warps": 8,
                "stages": 4,
            },
        }
        bucket_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))

    cache_key = make_cache_key(
        graph_hash=hashlib.sha256(optimized_mlir.encode("utf-8")).hexdigest(),
        target=target,
        arch=arch,
        dtype="fp32",
        shape_constraints=(
            "static" if not constraint_signature else f"dynamic_{bucket_hex[:8]}"
        ),
        schedule="schedule_64x128x32",
        symbolic_constraints=constraint_signature,
    )

    return DynamicCompiledFusedMLP(
        model=model,
        example_inputs=list(example_inputs),
        mlir_text=optimized_mlir,
        cache_key=cache_key,
        cache_hit=cache_hit,
        symbolic_constraints=constraint_signature,
        constraint_signature=constraint_signature,
    )