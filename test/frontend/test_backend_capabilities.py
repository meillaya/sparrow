"""Phase 7 BackendCapabilities pytest (Task 4, plan §Todos).

Pins the spec §6 capability schema for each shipped backend. These are
the exact predicates the Phase 7 partitioner (Task 18) will check before
considering a backend for a region; if any test regresses the partitioner
silently picks a wrong backend.

Coverage (Task 4 acceptance = 4 tests; we ship exactly those 4):

  * Triton supports every forge.tensor op that has a CPU kernel
    (gemm, fused_epilogue, broadcast_add, mul, relu, transpose, select,
    reshape, gelu, layer_norm, softmax, sdpa) but no async execution.
    fp32 only.
  * Astra does NOT support layer_norm / softmax / sdpa (the microcode
    ISA lacks them).
  * MicroGPU does NOT support gemm, fused_epilogue, sdpa, or reshape
    (the SIMT ISA has no matmul or reshape microcode).
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


def test_triton_supports_the_lowerable_tensor_ops() -> None:
    """Triton-CPU lists every op that has a dialect definition and CPU kernel.

    A capability is not a lowering guarantee; the partitioner also
    consults ``COMPILER_LOWERINGS``. ``supports_async_execution=False``
    reflects the synchronous shipped kernel.
    """
    expected_ops = {
        "forge.tensor.gemm",
        "forge.tensor.fused_epilogue",
        "forge.tensor.broadcast_add",
        "forge.tensor.mul",
        "forge.tensor.relu",
        "forge.tensor.transpose",
        "forge.tensor.select",
        "forge.tensor.reshape",
        "forge.tensor.gelu",
        "forge.tensor.layer_norm",
        "forge.tensor.softmax",
        "forge.tensor.scaled_dot_product_attention",
    }
    assert TRITON_CAPABILITIES.supported_ops == expected_ops
    assert TRITON_CAPABILITIES.supported_ops >= expected_ops  # exact-equality sanity
    assert TRITON_CAPABILITIES.supports_async_execution is False
    assert TRITON_CAPABILITIES.supported_types == frozenset({"fp32"})


def test_astra_does_not_support_reduction_or_attention_ops() -> None:
    """Astra's microcode ISA has no layer_norm, softmax, or sdpa."""
    for op in (
        "forge.tensor.layer_norm",
        "forge.tensor.softmax",
        "forge.tensor.scaled_dot_product_attention",
    ):
        assert op not in ASTRA_CAPABILITIES.supported_ops, op


def test_microgpu_does_not_support_matmul_or_attention() -> None:
    """The SIMT ISA has no matmul, fused epilogue, sdpa, or reshape op."""
    for op in (
        "forge.tensor.gemm",
        "forge.tensor.fused_epilogue",
        "forge.tensor.scaled_dot_product_attention",
        "forge.tensor.reshape",
    ):
        assert op not in MICROGPU_CAPABILITIES.supported_ops, op


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


def test_compiler_lowerings_are_explicit() -> None:
    """The compiler lowering map exists and starts empty.

    Empty means the partitioner fails closed until a region codegen
    lands. It must not silently fall back to Triton.
    """
    from forge_torch.lowerings import COMPILER_LOWERINGS

    assert set(COMPILER_LOWERINGS) == {"triton", "astra", "microgpu"}
    assert all(ops == frozenset() for ops in COMPILER_LOWERINGS.values())
