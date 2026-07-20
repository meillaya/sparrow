"""BackendCapabilities registry per spec §6 (Phase 7 Wave 1)."""

# frontend/python/forge_torch/capabilities.py
#
# BackendCapabilities registry for the Phase 7 heterogeneous partitioner
# (spec §6, Tasks 3-6). The partitioner (Task 18) consults a backend's
# :class:`BackendCapabilities` before considering it for a region: a backend
# whose :attr:`supported_ops` does not cover every op in the region is
# rejected without cost estimation. The ``compile()`` entry point (Task 5)
# performs the same check on the captured FX graph for the user-supplied
# ``target=`` argument and emits a :class:`UserWarning` when a requested
# target cannot run the model as-is.
#
# Three shipped backends:
#
# * **Triton-CPU** (Path A kernel template, spec §5) — the canonical Phase 1
#   target. Supports every forge.tensor op the Phase 7 dialect extension
#   (Tasks 11-13) ships, but only fp32 today; async execution and dynamic
#   shapes are out of scope.
# * **Astra-sim** (Phase 4 microcode ISA, spec §9) — a systolic-array
#   simulator. Has DMA + matmul + vector_gelu, but the Phase 4 microcode ISA
#   does NOT include a softmax or scaled-dot-product-attention op, so Astra
#   cannot run regions that need them. Local scratchpad = 512 KiB.
# * **MicroGPU-sim** (Phase 8 SIMT interpreter, spec §10) — a scalar
#   SIMT bytecode interpreter. Has all the activation / reduction ops
#   (gelu / layer_norm / softmax / sdpa via add+mul+sigmoid), but the ISA
#   has no reshape microcode, so MicroGPU cannot lower ``forge.tensor.reshape``
#   in isolation — the partitioner must route reshape regions to Triton-CPU.
#   Local scratchpad = 64 KiB (per-thread register file; shared memory is
#   0 because the interpreter has no shared-memory model).

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can run (spec §6 capabilities schema).

    The fields are exactly the spec §6 surface — adding a field requires a
    spec update. Frozen so callers can hash the dataclass (the partitioner's
    memoization relies on this). Op/type spellings are the canonical
    ``forge.tensor.<op>`` / canonical lowercase-dtype-token forms; the same
    spellings are emitted by the FX importer and consumed by the dialect
    verifier.
    """

    name: str
    supported_ops: frozenset[str]
    supported_types: frozenset[str]
    local_memory_bytes: int
    supports_dynamic_shapes: bool
    supports_async_execution: bool
    supports_fused_epilogues: bool


# ---------------------------------------------------------------------------
# Triton-CPU — naive triple-loop kernel, Path A reference (spec §5).
# All seven Phase 7 forge.tensor ops are supported; fp32 only because the
# shipped C++ kernel is fp32-only and the bf16/fp16 dtypes are a Phase 2
# follow-on. ``supports_fused_epilogues=True`` because the canonical
# ``fused_gemm_bias_gelu`` kernel combines gemm+bias+activation into one
# launch (Task 9 of Phase 1, mirrored in ``fused_gemm_bias_gelu.cpp``).
# Async execution is not supported (the kernel is synchronous); dynamic
# shapes are not supported (Phase 2 symbolic-guards work doesn't apply to
# this kernel).
# ---------------------------------------------------------------------------

TRITON_CAPABILITIES: Final[BackendCapabilities] = BackendCapabilities(
    name="triton",
    supported_ops=frozenset(
        {
            "forge.tensor.gemm",
            "forge.tensor.fused_epilogue",
            "forge.tensor.broadcast_add",
            "forge.tensor.gelu",
            "forge.tensor.layer_norm",
            "forge.tensor.softmax",
            "forge.tensor.scaled_dot_product_attention",
            "forge.tensor.reshape",
        }
    ),
    supported_types=frozenset({"fp32"}),
    local_memory_bytes=0,
    supports_dynamic_shapes=False,
    supports_async_execution=False,
    supports_fused_epilogues=True,
)


# ---------------------------------------------------------------------------
# Astra-sim — systolic-array simulator with DMA + matmul + vector_gelu
# (Phase 4 microcode ISA). The ISA does NOT include softmax or
# scaled-dot-product-attention, so Astra is rejected for regions containing
# them (the partitioner emits the warning at compile() time). 512 KiB
# scratchpad matches ``backends/astra/sim/timing.py::SCRATCHPAD_BYTES``.
# Async execution is supported because the simulator enqueues commands and
# returns completion events asynchronously.
# ---------------------------------------------------------------------------

ASTRA_CAPABILITIES: Final[BackendCapabilities] = BackendCapabilities(
    name="astra",
    supported_ops=frozenset(
        {
            "forge.tensor.gemm",
            "forge.tensor.fused_epilogue",
            "forge.tensor.broadcast_add",
            "forge.tensor.gelu",
            "forge.tensor.layer_norm",
        }
    ),
    supported_types=frozenset({"fp32"}),
    local_memory_bytes=512 * 1024,
    supports_dynamic_shapes=False,
    supports_async_execution=True,
    supports_fused_epilogues=True,
)


# ---------------------------------------------------------------------------
# MicroGPU-sim — SIMT bytecode interpreter (Phase 8). Has all the activation
# ops (gelu / layer_norm / softmax / sdpa composed of add+mul+sigmoid), but
# the ISA has no reshape microcode — the partitioner must route reshape
# regions to Triton-CPU. 64 KiB local memory per warp's register file
# (spec §10). Async execution is not supported (the interpreter is fully
# synchronous in the shipped reference impl).
# ---------------------------------------------------------------------------

MICROGPU_CAPABILITIES: Final[BackendCapabilities] = BackendCapabilities(
    name="microgpu",
    supported_ops=frozenset(
        {
            "forge.tensor.gemm",
            "forge.tensor.fused_epilogue",
            "forge.tensor.broadcast_add",
            "forge.tensor.gelu",
            "forge.tensor.layer_norm",
            "forge.tensor.softmax",
            "forge.tensor.scaled_dot_product_attention",
        }
    ),
    supported_types=frozenset({"fp32"}),
    local_memory_bytes=64 * 1024,
    supports_dynamic_shapes=False,
    supports_async_execution=False,
    supports_fused_epilogues=True,
)


# ---------------------------------------------------------------------------
# Registry — single source of truth for ``compile(target=...)`` and the
# partitioner (Task 18). Adding a backend = adding an entry here + an
# instantiation above. The dict is typed ``str`` so ``BACKEND_CAPABILITIES``
# is also the lookup table the partitioner's memoized ``is_capable_of()``
# check indexes by name.
# ---------------------------------------------------------------------------

BACKEND_CAPABILITIES: Final[dict[str, BackendCapabilities]] = {
    "triton": TRITON_CAPABILITIES,
    "astra": ASTRA_CAPABILITIES,
    "microgpu": MICROGPU_CAPABILITIES,
}


__all__ = (
    "ASTRA_CAPABILITIES",
    "BACKEND_CAPABILITIES",
    "MICROGPU_CAPABILITIES",
    "TRITON_CAPABILITIES",
    "BackendCapabilities",
)
