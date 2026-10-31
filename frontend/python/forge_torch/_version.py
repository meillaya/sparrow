"""Sparrow compiler version (per spec §13 cache key).

The hex digest of every :class:`forge_torch.cache.CacheKey` folds in
this string so that a compiler upgrade invalidates the on-disk cache
automatically. Bump it on any commit that changes the forge.graph /
forge.tensor dialects, the FX importer, the canonicalization pass, or
the codegen templates.
"""

from __future__ import annotations

__version__ = "0.2.0"
