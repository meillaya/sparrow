"""Hierarchical compilation cache key (per spec §13).

Cache key shape:
    graph_hash + target + arch + dtype + shape_constraints + schedule
        + compiler_version

The on-disk path layout mirrors those tiers so that ``ls`` on the cache
root shows exactly what the compiler emitted and why::

    fused_mlp/
      nvidia_sm90/
        fp16/
          dynamic_m/
            schedule_64x128x32/
              artifact.json
              kernel.bin
              metrics.json

The :meth:`CacheKey.hexdigest` folds every tier (plus the compiler
version) into a 16-hex-character SHA-256 prefix; two keys are equal
iff their on-disk paths are equal. The hex digest is what the
in-memory tier of :mod:`forge_torch.cache_store` keys on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from forge_torch._version import __version__ as COMPILER_VERSION


@dataclass(frozen=True)
class CacheKey:
    """Hierarchical compilation cache key (spec §13)."""

    graph_hash: str
    target: str  # "triton", "hls", "astra", "microgpu", "cpu"
    arch: str  # "x86_64_v3", "nvidia_sm90", etc.
    dtype: str  # "fp32", "fp16", "bf16"
    shape_constraints: str  # "static" or "dynamic_m" / "dynamic_n" etc.
    schedule: str  # "schedule_64x128x32"

    def to_path(self, root: Path) -> Path:
        """Return the on-disk directory for this key under ``root``.

        Layout::

            <root>/fused_mlp/<arch>/<dtype>/<shape_constraints>/<schedule>
        """
        return root / "fused_mlp" / self.arch / self.dtype / self.shape_constraints / self.schedule

    def hexdigest(self) -> str:
        """Stable 16-hex-char digest over all key tiers + compiler version.

        The canonical form is ``json.dumps(..., sort_keys=True)`` so that
        any reordering of dataclass fields produces the same digest.
        """
        canonical = json.dumps(
            {
                "graph_hash": self.graph_hash,
                "target": self.target,
                "arch": self.arch,
                "dtype": self.dtype,
                "shape_constraints": self.shape_constraints,
                "schedule": self.schedule,
                "compiler_version": COMPILER_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def make_cache_key(
    *,
    graph_hash: str,
    target: str = "triton",
    arch: str = "x86_64_v3",
    dtype: str = "fp32",
    shape_constraints: str = "static",
    schedule: str = "schedule_64x128x32",
) -> CacheKey:
    """Convenience builder that defaults to Phase-1 triton CPU parameters."""
    return CacheKey(
        graph_hash=graph_hash,
        target=target,
        arch=arch,
        dtype=dtype,
        shape_constraints=shape_constraints,
        schedule=schedule,
    )
