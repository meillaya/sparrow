"""Cache-key + artifact-byte determinism pytest (Phase 2, Task 21, spec §13).

Compiles the same ``FusedMLP`` 5 times in a row with the same seed,
sharing one :class:`~forge_torch.cache_store.CacheStore`. Asserts:

1. All 5 :attr:`CompiledFusedMLP.cache_key` instances hash to the same
   16-hex-char digest — i.e. graph capture, MLIR import, and key
   composition are deterministic across runs.
2. Every ``artifact.json`` written to the cache is byte-identical — the
   artifact body, the MLIR text inside it, and the schedule dict all
   match across the 5 invocations.
3. The first compile is a cache miss and the remainder are cache hits —
   mirrors the ``test_dynamic_compile`` cold/warm contract.

Both contracts catch nondeterminism in op_mapping, FX capture, or the
cache-key JSON canonical form (e.g. a stray set/dict ordering that
flips a digest between runs).
"""

from __future__ import annotations

from pathlib import Path

import torch
from forge_torch.cache import make_cache_key
from forge_torch.cache_store import CacheStore
from forge_torch.conftest import FusedMLP


def _seeded_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase 1 canonical FusedMLP example inputs (deterministic via manual_seed)."""
    torch.manual_seed(0)
    return (
        torch.randn(2, 16, 128, dtype=torch.float32),
        torch.randn(128, 256, dtype=torch.float32),
        torch.randn(256, dtype=torch.float32),
    )


def test_cache_keys_stable_across_five_runs(tmp_path: Path) -> None:
    """5x compile() → all 5 cache keys hash to the same digest."""
    from forge_torch.backend import compile

    store = CacheStore(tmp_path / "cache")
    model = FusedMLP()

    digests: list[str] = []
    for _ in range(5):
        x, w, b = _seeded_inputs()
        compiled = compile(model, [x, w, b], cache_store=store)
        assert compiled.cache_key is not None
        digests.append(compiled.cache_key.hexdigest())

    assert len(set(digests)) == 1, (
        f"cache-key digest drifted across 5 runs: {digests}"
    )


def test_artifacts_byte_identical_across_five_runs(tmp_path: Path) -> None:
    """5x compile() → all 5 written artifact.json files are byte-equal."""
    from forge_torch.backend import compile

    store = CacheStore(tmp_path / "cache")
    model = FusedMLP()

    expected_key = make_cache_key(
        graph_hash="placeholder",
        dtype="fp32",
    )
    artifact_path: Path | None = None

    for _ in range(5):
        x, w, b = _seeded_inputs()
        compiled = compile(model, [x, w, b], cache_store=store)
        assert compiled.cache_key is not None
        path = compiled.cache_key.to_path(tmp_path / "cache") / "artifact.json"
        assert path.is_file(), f"artifact.json missing at {path}"
        if artifact_path is None:
            artifact_path = path
            continue
        assert path.read_bytes() == artifact_path.read_bytes(), (
            "artifact.json bytes drifted across the 5 compile() runs"
        )


def test_first_compile_miss_then_warm_hits(tmp_path: Path) -> None:
    """First compile() is a cache miss, subsequent runs hit the cache."""
    from forge_torch.backend import compile

    store = CacheStore(tmp_path / "cache")
    model = FusedMLP()

    cache_hits: list[bool] = []
    for _ in range(5):
        x, w, b = _seeded_inputs()
        compiled = compile(model, [x, w, b], cache_store=store)
        cache_hits.append(bool(compiled.cache_hit))

    assert cache_hits[0] is False, f"first compile should be a miss, got {cache_hits}"
    assert all(hit is True for hit in cache_hits[1:]), (
        f"warm calls should hit; got {cache_hits[1:]}"
    )


def test_mlir_text_byte_identical_across_five_runs(tmp_path: Path) -> None:
    """5x compile() → the captured MLIR text is byte-equal (no FX drift)."""
    from forge_torch.backend import compile

    store = CacheStore(tmp_path / "cache")
    model = FusedMLP()

    mlir_texts: list[str] = []
    for _ in range(5):
        x, w, b = _seeded_inputs()
        compiled = compile(model, [x, w, b], cache_store=store)
        mlir_texts.append(compiled.mlir_text)

    assert len(set(mlir_texts)) == 1, (
        f"mlir_text drifted across 5 runs:\n"
        + "\n----\n".join(mlir_texts[:3])
    )


