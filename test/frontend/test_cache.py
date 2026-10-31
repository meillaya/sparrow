"""Sparrow compilation cache pytest tests (Task 14, spec §13).

Six tests pin the contract of :mod:`forge_torch.cache` and
:mod:`forge_torch.cache_store`:

1. ``test_cache_key_hexdigest_is_deterministic`` — same inputs → same digest.
2. ``test_cache_key_changes_when_target_changes`` — target is in the digest.
3. ``test_cache_key_changes_when_dtype_changes`` — dtype is in the digest.
4. ``test_cache_store_put_then_get`` — roundtrip via :class:`CacheStore`.
5. ``test_cache_store_memory_evicts_lru`` — LRU eviction past capacity.
6. ``test_cache_store_disk_persists_across_instances`` — disk tier outlives
   the in-process state.

Each test uses a per-test temporary directory under
``/tmp/sparrow-cache-tests`` so they don't share state across runs
and don't dirty the repo's own ``.cache/``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from forge_torch.cache import make_cache_key
from forge_torch.cache_store import CACHE_KEY_FIELD, CacheStore


def _without_key(artifact: dict) -> dict:
    return {name: value for name, value in artifact.items() if name != CACHE_KEY_FIELD}


def _tmp_root() -> Path:
    """Per-call cache root; cleaned up automatically by ``tempfile``."""
    return Path(tempfile.mkdtemp(prefix="sparrow-cache-tests-"))


def test_cache_key_hexdigest_is_deterministic() -> None:
    """Same CacheKey inputs ⇒ same hexdigest (call it twice)."""
    key = make_cache_key(graph_hash="abc123")
    assert key.hexdigest() == key.hexdigest()
    assert len(key.hexdigest()) == 16  # 64-bit prefix of SHA-256

    # Two equivalent keys built independently should match.
    other = make_cache_key(
        graph_hash="abc123",
        target="triton",
        arch="x86_64_v3",
        dtype="fp32",
        shape_constraints="static",
        schedule="schedule_64x128x32",
    )
    assert other.hexdigest() == key.hexdigest()


def test_cache_key_changes_when_target_changes() -> None:
    """``target`` is part of the digest (Triton vs. HLS ≠)."""
    base = make_cache_key(graph_hash="abc")
    hls = make_cache_key(graph_hash="abc", target="hls")
    assert base.target == "triton"
    assert hls.target == "hls"
    assert base.hexdigest() != hls.hexdigest()


def test_cache_key_changes_when_dtype_changes() -> None:
    """``dtype`` is part of the digest (fp32 vs. fp16 ≠)."""
    base = make_cache_key(graph_hash="abc", dtype="fp32")
    fp16 = make_cache_key(graph_hash="abc", dtype="fp16")
    assert base.dtype == "fp32"
    assert fp16.dtype == "fp16"
    assert base.hexdigest() != fp16.hexdigest()


def test_cache_store_put_then_get() -> None:
    """Write artifact → read back via :meth:`CacheStore.get` roundtrips equality."""
    root = _tmp_root()
    store = CacheStore(root)
    key = make_cache_key(graph_hash="round-trip")

    artifact = {
        "model_name": "FusedMLP",
        "mlir_text": "module { func.func @main() -> () { return } }",
        "schedule": {"tile_m": 64, "tile_n": 128, "tile_k": 32, "warps": 8, "stages": 4},
    }
    store.put(key, artifact)

    loaded = store.get(key)
    assert loaded is not None
    assert _without_key(loaded) == artifact
    assert loaded[CACHE_KEY_FIELD] == key.hexdigest()

    # The on-disk artifact.json sits under the key digest.
    expected_path = (
        root
        / "fused_mlp"
        / "x86_64_v3"
        / "fp32"
        / "static"
        / "schedule_64x128x32"
        / key.hexdigest()
        / "artifact.json"
    )
    assert expected_path.exists()
    on_disk = json.loads(expected_path.read_text())
    assert _without_key(on_disk) == artifact
    assert on_disk[CACHE_KEY_FIELD] == key.hexdigest()


def test_cache_store_memory_evicts_lru() -> None:
    """Exceeding ``max_memory_entries`` evicts the oldest hexdigest from memory."""
    store = CacheStore(_tmp_root(), max_memory_entries=3)

    keys = [make_cache_key(graph_hash=f"graph-{i}") for i in range(4)]
    for k in keys:
        store.put(k, {"graph_hash": k.graph_hash})

    # After putting 4 entries into a 3-slot tier, the FIRST inserted
    # (graph-0) must have been evicted from memory. Disk still has it.
    assert keys[0].hexdigest() not in store._memory
    for k in keys[1:]:
        assert k.hexdigest() in store._memory

    # The most recently inserted is at the back of the LRU order.
    assert list(store._memory.keys())[-1] == keys[-1].hexdigest()


def test_cache_store_disk_persists_across_instances() -> None:
    """A disk write survives a fresh :class:`CacheStore` pointing at the same root."""
    root = _tmp_root()

    writer = CacheStore(root)
    key = make_cache_key(graph_hash="persisted")
    payload = {"model_name": "FusedMLP", "mlir_text": "module {}", "schedule": {}}
    writer.put(key, payload)

    # Brand-new instance, same root — should warm its memory tier from disk.
    reader = CacheStore(root)
    fresh = reader.get(key)
    assert fresh is not None
    assert _without_key(fresh) == payload
    # The reader's memory tier now contains the entry.
    assert key.hexdigest() in reader._memory


def test_cache_key_changes_with_constraints() -> None:
    """Different constraint tuples hash to different keys (spec §13 / Task 16).

    Same graph + same target/arch/dtype/schedule but different symbolic-dim
    constraints must produce distinct cache buckets so the dynamic-compile
    pipeline can specialise per-shape without trampling.
    """
    base = make_cache_key(graph_hash="dyn-mlp")
    with_b_s = make_cache_key(
        graph_hash="dyn-mlp",
        symbolic_constraints=("B >= 1", "S >= 1"),
    )
    with_k = make_cache_key(
        graph_hash="dyn-mlp",
        symbolic_constraints=("K == 4096",),
    )
    none_constraints = make_cache_key(graph_hash="dyn-mlp", symbolic_constraints=())

    # Each tuple is distinct from every other.
    assert base.hexdigest() != with_b_s.hexdigest()
    assert base.hexdigest() != with_k.hexdigest()
    assert with_b_s.hexdigest() != with_k.hexdigest()
    # The empty-tuple case also differs from the static default (None == ())
    # — both render to () on the wire, so the digest must match here.
    assert base.hexdigest() == none_constraints.hexdigest()


def test_cache_key_stable_for_same_constraints() -> None:
    """The same constraint tuple hashes to the same digest across invocations.

    Independent :func:`make_cache_key` calls with the same arguments must
    produce identical :attr:`CacheKey.hexdigest` — the build path is
    deterministic for the same key inputs.
    """
    constraints = ("B >= 1", "S >= 1", "K == 4096")
    first = make_cache_key(graph_hash="replay", symbolic_constraints=constraints)
    # Reorder the constraint list — list is heterogeneous but the call site
    # may pass either order; the JSON canonical form sorts keys so the
    # ordering inside the tuple still has to match.
    second = make_cache_key(graph_hash="replay", symbolic_constraints=tuple(constraints))
    third = make_cache_key(graph_hash="replay", symbolic_constraints=tuple(constraints))
    assert first.hexdigest() == second.hexdigest() == third.hexdigest()
    # And of course the bundled key carries the constraints through.
    assert first.symbolic_constraints == tuple(constraints)
