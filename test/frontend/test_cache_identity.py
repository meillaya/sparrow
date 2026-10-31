"""Cache identity regression tests.

The old cache path omitted the graph hash and target, so two different
models could read the same ``artifact.json``. These tests pin the new
contract: every key owns a distinct directory, every artifact carries its
key, and a corrupt or mismatched file is a miss.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from forge_torch.cache import make_cache_key
from forge_torch.cache_store import CACHE_KEY_FIELD, CacheStore


def _root() -> Path:
    return Path(tempfile.mkdtemp(prefix="sparrow-cache-identity-"))


def _artifact(name: str) -> dict:
    return {"model_name": name, "mlir_text": f"module {{ // {name} }}", "schedule": {}}


def test_different_graphs_do_not_share_a_path() -> None:
    root = _root()
    store = CacheStore(root)
    first = make_cache_key(graph_hash="graph-a")
    second = make_cache_key(graph_hash="graph-b")

    assert first.to_path(root) != second.to_path(root)
    store.put(first, _artifact("a"))
    store.put(second, _artifact("b"))

    first_loaded = store.get(first)
    second_loaded = store.get(second)
    assert first_loaded is not None and second_loaded is not None
    assert first_loaded["model_name"] == "a"
    assert second_loaded["model_name"] == "b"


def test_different_targets_do_not_share_a_path() -> None:
    root = _root()
    store = CacheStore(root)
    triton = make_cache_key(graph_hash="same-graph", target="triton")
    astra = make_cache_key(graph_hash="same-graph", target="astra")

    assert triton.to_path(root) != astra.to_path(root)
    store.put(triton, _artifact("triton"))
    store.put(astra, _artifact("astra"))

    triton_loaded = store.get(triton)
    astra_loaded = store.get(astra)
    assert triton_loaded is not None and astra_loaded is not None
    assert triton_loaded["model_name"] == "triton"
    assert astra_loaded["model_name"] == "astra"


def test_artifact_carries_the_key_that_wrote_it() -> None:
    root = _root()
    store = CacheStore(root)
    key = make_cache_key(graph_hash="manifest")
    store.put(key, _artifact("manifest"))

    loaded = store.get(key)
    assert loaded is not None
    assert loaded[CACHE_KEY_FIELD] == key.hexdigest()


def test_key_mismatch_is_a_miss() -> None:
    root = _root()
    store = CacheStore(root)
    key = make_cache_key(graph_hash="expected")
    path = key.to_path(root) / "artifact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_name": "wrong", CACHE_KEY_FIELD: "0000000000000000"}))

    assert store.get(key) is None


def test_corrupt_artifact_is_quarantined() -> None:
    root = _root()
    store = CacheStore(root)
    key = make_cache_key(graph_hash="corrupt")
    path = key.to_path(root) / "artifact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")

    assert store.get(key) is None
    assert not path.exists()
    assert path.with_name("artifact.json.corrupt").exists()


def test_path_components_cannot_escape_the_root() -> None:
    root = _root().resolve()
    key = make_cache_key(graph_hash="escape", arch="../../outside", dtype="f/32")
    path = key.to_path(root).resolve()

    assert path.is_relative_to(root)
    assert ".." not in path.parts
