"""Two-tier compilation cache store (per spec §13).

Tier 1 — in-memory LRU keyed by :meth:`CacheKey.hexdigest`. Holds the
most recent artifacts to avoid disk roundtrips inside a single
``torch.compile`` session (e.g. autotuning repeats the same lookup).

Tier 2 — disk under ``<root>/fused_mlp/<arch>/<dtype>/<shape>/<schedule>/``.
``artifact.json`` is the canonical MLIR / schedule / kernel-metrics
bundle the compiler emits; the directory layout matches the spec §13
example so that operators can introspect the cache with plain ``ls``.

Eviction
---------
The memory tier evicts in LRU order when ``max_memory_entries`` is
exceeded. The disk tier is unbounded in Phase 1 — operator-driven
GC (``rm -rf .cache/forgestack``) is sufficient until the autotuner
(Task 15) starts filling real ``kernel.bin`` artifacts.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from forge_torch.cache import CacheKey

DEFAULT_CACHE_ROOT = Path(".cache/forgestack")


class CacheStore:
    """Two-tier (memory LRU + disk JSON) cache for compilation artifacts."""

    def __init__(self, root: Path = DEFAULT_CACHE_ROOT, *, max_memory_entries: int = 32) -> None:
        self.root = Path(root)
        self.max_memory_entries = max_memory_entries
        self._memory: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: CacheKey) -> dict | None:
        """Return the cached artifact for ``key`` or ``None`` on miss.

        Memory hit promotes the entry to the back of the LRU. Disk hit
        warms the memory tier with the same entry so subsequent lookups
        within the same session stay in-process.
        """
        h = key.hexdigest()
        if h in self._memory:
            self._memory.move_to_end(h)
            return self._memory[h]
        artifact_path = key.to_path(self.root) / "artifact.json"
        if artifact_path.exists():
            data = json.loads(artifact_path.read_text())
            self._put_memory(h, data)
            return data
        return None

    def put(self, key: CacheKey, artifact: dict) -> None:
        """Persist ``artifact`` under ``key`` and warm the memory tier."""
        h = key.hexdigest()
        dir_path = key.to_path(self.root)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True))
        self._put_memory(h, artifact)

    def _put_memory(self, h: str, data: dict) -> None:
        self._memory[h] = data
        self._memory.move_to_end(h)
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)
