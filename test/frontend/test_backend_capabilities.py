"""Phase 7 BackendCapabilities pytest (Task 4, plan §Todos).

Pins the spec §6 capability schema for each shipped backend. These are
the exact predicates the Phase 7 partitioner (Task 18) will check before
considering a backend for a region; if any test regresses the partitioner
silently picks a wrong backend.

Coverage (Task 4 acceptance = 4 tests; we ship exactly those 4):

  * Triton supports every shipped forge.tensor op (8 total: gemm,
    fused_epilogue, broadcast_add, gelu, layer_norm, softmax, sdpa,
    reshape) but no async execution. fp32 only.
  * Astra does NOT support softmax / sdpa (Phase 4 microcode ISA lacks
    them).
  * MicroGPU does NOT support reshape (the SIMT ISA has no reshape
    microcode op).
  * The ``BACKEND_CAPABILITIES`` registry exposes the static instances
    under their canonical names (``"triton"``, ``"astra"``,
    ``"microgpu"``) by identity.
"""

from __future__ import annotations

from forge_torch.capabilities import (
    ASTRA_CAPABILITIES,
    BACKEND_CAPABILITIES,
    MICROGPU_CAPABILITIES,
    TRITON_CAPABILITIES,
)


def test_triton_supports_all_tensor_ops_except_async() -> None:
    """Triton-CPU supports every Phase 7 forge.tensor op (8 total).

    The partitioner relies on ``TRITON_CAPABILITIES`` being the superset
    so it can always fall back to Triton when no other backend is
    available for an op. ``supports_async_execution=False`` reflects the
    synchronous shipped kernel (``fused_gemm_bias_gelu.cpp`` runs
    in-process and returns once the loop is done).
    """
    expected_ops = {
        "forge.tensor.gemm",
        "forge.tensor.fused_epilogue",
        "forge.tensor.broadcast_add",
        "forge.tensor.gelu",
        "forge.tensor.layer_norm",
        "forge.tensor.softmax",
        "forge.tensor.scaled_dot_product_attention",
        "forge.tensor.reshape",
    }
    assert TRITON_CAPABILITIES.supported_ops == expected_ops
    assert TRITON_CAPABILITIES.supported_ops >= expected_ops  # exact-equality sanity
    assert TRITON_CAPABILITIES.supports_async_execution is False
    assert TRITON_CAPABILITIES.supported_types == frozenset({"fp32"})


def test_astra_does_not_support_softmax_or_sdpa() -> None:
    """Astra's microcode ISA has no softmax / sdpa op (Phase 4 limitation).

    The partitioner must reject Astra for any region that contains
    either of these two ops; the compile() entry point uses the same
    negative assertion to emit the UserWarning at the user-visible API
    (Task 5).
    """
    assert "forge.tensor.softmax" not in ASTRA_CAPABILITIES.supported_ops
    assert "forge.tensor.scaled_dot_product_attention" not in ASTRA_CAPABILITIES.supported_ops


def test_microgpu_does_not_support_reshape() -> None:
    """MicroGPU's SIMT ISA has no reshape microcode.

    The Phase 8 interpreter only implements load / store / add / mul /
    sigmoid / return; reshape must be lowered by the Triton path before
    MicroGPU can run the kernel. The partitioner routes any region
    containing ``forge.tensor.reshape`` to Triton-CPU via this negative
    predicate.
    """
    assert "forge.tensor.reshape" not in MICROGPU_CAPABILITIES.supported_ops


def test_registry_lookup_by_name() -> None:
    """``BACKEND_CAPABILITIES[<name>]`` returns the matching static instance.

    Identity (``is``) — not equality — is the contract: the partitioner
    hashes the dataclass instance for memoization, so a fresh copy from
    the dict would defeat the cache. Adding a backend = adding one entry
    here + one assertion below; the registry is intentionally exhaustive
    over the three shipped backends.
    """
    assert BACKEND_CAPABILITIES["triton"] is TRITON_CAPABILITIES
    assert BACKEND_CAPABILITIES["astra"] is ASTRA_CAPABILITIES
    assert BACKEND_CAPABILITIES["microgpu"] is MICROGPU_CAPABILITIES
    assert set(BACKEND_CAPABILITIES) == {"triton", "astra", "microgpu"}
