"""Two-tier compilation cache store (per spec §13).

Tier 1 — in-memory LRU keyed by :meth:`CacheKey.hexdigest`. Holds the
most recent artifacts to avoid disk roundtrips inside one compile run.

Tier 2 — disk under
``<root>/fused_mlp/<arch>/<dtype>/<shape>/<schedule>/<hexdigest>/``.
``artifact.json`` carries the MLIR text, the schedule, and the key that
produced it. The digest is the final path component, so two different
keys never share a directory. ``CacheStore.get`` also checks the stored
key when the artifact has one.

Writes go through a temporary file in the same directory and
``os.replace``, so a crash cannot leave a half-written ``artifact.json``.
A malformed artifact is renamed to ``artifact.json.corrupt`` and treated
as a miss.

Eviction
---------
The memory tier evicts in LRU order when ``max_memory_entries`` is
exceeded. The disk tier is unbounded; operators can remove
``.cache/sparrow`` at any time.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

from forge_torch.cache import CacheKey

DEFAULT_CACHE_ROOT = Path(".cache/sparrow")
CACHE_KEY_FIELD = "cache_key"


class CacheStore:
    """Two-tier (memory LRU + disk JSON) cache for compilation artifacts."""

    def __init__(self, root: Path = DEFAULT_CACHE_ROOT, *, max_memory_entries: int = 32) -> None:
        self.root = Path(root)
        self.max_memory_entries = max_memory_entries
        self._memory: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: CacheKey) -> dict | None:
        """Return the cached artifact for ``key`` or ``None`` on miss.

        A memory hit promotes the entry to the back of the LRU. A disk hit
        warms the memory tier. A corrupt file or a key mismatch is a miss;
        the caller rebuilds the artifact instead of reading wrong data.
        """
        digest = key.hexdigest()
        if digest in self._memory:
            self._memory.move_to_end(digest)
            return self._memory[digest]

        artifact_path = self._artifact_path(key)
        try:
            text = artifact_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._quarantine(artifact_path)
            return None
        if not isinstance(data, dict):
            self._quarantine(artifact_path)
            return None

        stored_key = data.get(CACHE_KEY_FIELD)
        if stored_key is not None and stored_key != digest:
            return None

        self._put_memory(digest, data)
        return data

    def put(self, key: CacheKey, artifact: dict) -> None:
        """Persist ``artifact`` under ``key`` and warm the memory tier.

        The artifact is copied and stamped with the key digest. The write
        is atomic: the final path appears only after the full JSON payload
        is on disk.
        """
        digest = key.hexdigest()
        payload = dict(artifact)
        payload[CACHE_KEY_FIELD] = digest
        artifact_path = self._artifact_path(key)
        _write_json_atomic(artifact_path, payload)
        self._put_memory(digest, payload)

    def _artifact_path(self, key: CacheKey) -> Path:
        return key.to_path(self.root) / "artifact.json"

    @staticmethod
    def _quarantine(artifact_path: Path) -> None:
        """Move a corrupt artifact aside so the next lookup can rebuild it."""
        corrupt = artifact_path.with_name(artifact_path.name + ".corrupt")
        try:
            os.replace(artifact_path, corrupt)
        except OSError:
            return

    def _put_memory(self, digest: str, data: dict) -> None:
        self._memory[digest] = data
        self._memory.move_to_end(digest)
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` through a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise
