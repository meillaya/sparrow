"""ForgeStack dynamic_compile pytest tests (Task 17, spec §3 + §13).

Three tests pin the dynamic-shape compilation pipeline:

1. ``test_dynamic_compile_cold_call_writes_artifact`` — first
   :func:`dynamic_compile` invocation with symbolic metadata on the
   captured FX graph writes ``<root>/dynamic_<hexdigest>/artifact.json``.

2. ``test_dynamic_compile_warm_call_hits_cache`` — second call with
   the same metadata reads from the existing bucket
   (``cache_hit=True``); no new directory or ``artifact.json``
   appears.

3. ``test_dynamic_compile_different_constraints_separate_buckets``
   — different ``tensor_meta['symbolic_dim']`` metadata produces a
   different bucket hexdigest and a different on-disk directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from forge_torch.conftest import FusedMLP
from forge_torch.dynamic_compile import dynamic_compile
from torch.fx.experimental.proxy_tensor import make_fx


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase 1 canonical FusedMLP example inputs."""
    return (
        torch.randn(2, 512, 4096),
        torch.randn(4096, 11008),
        torch.randn(11008),
    )


def _inject_symbolic_dim(gm, symbolic_dim: tuple[str, ...]) -> None:
    """Mutate the first placeholder's ``tensor_meta`` to carry symbolic_dim.

    ``tensor_meta`` is a frozen dataclass; replace it wholesale via a
    shallow dict (mirrors ``test_fx_importer_symbolic.py``).
    """
    ph_x = next(n for n in gm.graph.nodes if n.op == "placeholder")
    orig_meta = ph_x.meta["tensor_meta"]
    ph_x.meta["tensor_meta"] = {
        "shape": orig_meta.shape,
        "dtype": orig_meta.dtype,
        "symbolic_dim": symbolic_dim,
    }


def test_dynamic_compile_cold_call_writes_artifact(tmp_path: Path) -> None:
    """Cold path: first ``dynamic_compile`` writes a ``dynamic_<hex>/`` bucket."""
    cache_root = tmp_path / "cache"
    x, w, b = _inputs()
    gm = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    _inject_symbolic_dim(gm, ("B", "S", "K"))

    # Pass the captured GraphModule directly — ``dynamic_compile`` would
    # re-trace a raw ``nn.Module`` and lose the per-placeholder mutation.
    compiled = dynamic_compile(gm, [x, w, b], cache_root=cache_root)

    # Cold path ⇒ the bucket must exist on disk.
    buckets = [p for p in cache_root.iterdir() if p.is_dir()]
    assert len(buckets) == 1, f"expected one dynamic bucket, got {buckets}"
    bucket = buckets[0]
    assert bucket.name.startswith("dynamic_"), bucket.name
    artifact = bucket / "artifact.json"
    assert artifact.is_file(), "artifact.json must exist on cold path"

    payload = json.loads(artifact.read_text())
    # ``make_fx`` wraps the model, so the recorded name may be ``<lambda>``
    # or the original class name depending on how it was constructed.
    assert payload["model_name"] in {"FusedMLP", "<lambda>"}
    assert payload["symbolic_constraints"] == ["B >= 1", "S >= 1", "K >= 1"]
    assert "mlir_text" in payload

    # The returned object's signature matches the bucket.
    assert compiled.constraint_signature == ("B >= 1", "S >= 1", "K >= 1")
    assert compiled.cache_hit is False


def test_dynamic_compile_warm_call_hits_cache(tmp_path: Path) -> None:
    """Warm path: second invocation with same metadata hits the existing bucket."""
    cache_root = tmp_path / "cache"
    x, w, b = _inputs()
    gm = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    _inject_symbolic_dim(gm, ("B", "S", "K"))

    cold = dynamic_compile(gm, [x, w, b], cache_root=cache_root)
    cold_paths = sorted(p.name for p in cache_root.iterdir() if p.is_dir())

    # Second call — same constraints, fresh FX module so the importer
    # is exercised again, but the warm path must not allocate a new
    # directory.
    gm2 = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    _inject_symbolic_dim(gm2, ("B", "S", "K"))
    warm = dynamic_compile(gm2, [x, w, b], cache_root=cache_root)
    warm_paths = sorted(p.name for p in cache_root.iterdir() if p.is_dir())

    assert cold_paths == warm_paths, (
        f"warm call must not allocate a new directory; before={cold_paths} after={warm_paths}"
    )
    assert warm.cache_hit is True, "second call must register as cache hit"
    # And the cached MLIR text must round-trip unchanged.
    assert warm.mlir_text == cold.mlir_text


def test_dynamic_compile_different_constraints_separate_buckets(tmp_path: Path) -> None:
    """Different constraint signatures get separate cache buckets."""
    cache_root = tmp_path / "cache"
    x, w, b = _inputs()

    gm1 = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    _inject_symbolic_dim(gm1, ("B", "S", "K"))
    compiled1 = dynamic_compile(gm1, [x, w, b], cache_root=cache_root)

    gm2 = make_fx(FusedMLP(), tracing_mode="fake")(x, w, b)
    _inject_symbolic_dim(gm2, ("M", "N", "K"))
    compiled2 = dynamic_compile(gm2, [x, w, b], cache_root=cache_root)

    # Two distinct buckets on disk.
    buckets = sorted(p.name for p in cache_root.iterdir() if p.is_dir())
    assert len(buckets) == 2, f"expected two buckets, got {buckets}"
    # Different constraint signatures → different hexdigests via
    # the cache key path.
    assert compiled1.cache_key is not None
    assert compiled2.cache_key is not None
    assert compiled1.cache_key.symbolic_constraints != compiled2.cache_key.symbolic_constraints
    assert compiled1.cache_key.hexdigest() != compiled2.cache_key.hexdigest()
