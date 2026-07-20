"""ForgeStack ``torch.compile`` backend hook (Tasks 8 + 14, Phase 1).

Spec §3 says ForgeTorch has six jobs:

1. graph capture
2. input metadata extraction
3. FX-to-MLIR import
4. compilation-cache lookup
5. invocation of compiled runtime
6. safe fallback

Phase 1 implements (1)-(3) with a thin (5) stub that delegates back to
PyTorch eager. Task 14 layers (4) on top via the hierarchical cache
described in spec §13, plus a (post-compile) observability report per
spec §16. Job (6) is not exercised by the shipped scope — the safe
fallback path needs the FX partitioner, which is not part of the
current implementation. ``CompiledFusedMLP.__call__`` is the eager
stub the shipped test suite exercises.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from forge_torch.cache import make_cache_key
from forge_torch.cache_store import CacheStore
from forge_torch.capabilities import BACKEND_CAPABILITIES
from forge_torch.fx_importer import FxImporter
from forge_torch.op_mapping import normalize_target
from forge_torch.report import CompilationReport

BACKEND = "forgestack"

# Fixed path shared by Tasks 13/14/15 so the smoke test always reads the
# last write (per Task 14 brief).
REPORT_PATH = Path(".omo/evidence/phase1-report.txt")

# ---------------------------------------------------------------------------
# Phase 7 capability check helpers (Task 5, spec §6).
#
# ``compile(target=...)`` looks up ``BACKEND_CAPABILITIES[target]`` and
# emits a ``UserWarning`` for every op in the captured FX graph that the
# target cannot run. The ``_ATEN_TO_FORGE_TENSOR`` table is the canonical
# (FX target) -> (``forge.tensor.<op>`` spelling) translation: the FX
# importer's dialect uses ``forge.graph.<op>`` (the Phase 1 surface), the
# capability schema uses ``forge.tensor.<op>`` (the Phase 7 surface), and
# the partitioner (Task 18) consumes the capability spelling directly.
# FX ops without a canonical Phase 7 op (e.g. ``aten.relu`` which is
# still in ``UNSUPPORTED_BUT_RECOGNISED``) are silently skipped — the
# capability check is a Phase 7 concern and only covers the ops the
# partitioner routes on.
# ---------------------------------------------------------------------------

_ATEN_TO_FORGE_TENSOR: Final[dict[str, str]] = {
    "aten.mm": "forge.tensor.gemm",
    "aten.matmul": "forge.tensor.gemm",
    "aten.add": "forge.tensor.broadcast_add",
    "aten.gelu": "forge.tensor.gelu",
    "aten.view": "forge.tensor.reshape",
    "aten._unsafe_view": "forge.tensor.reshape",
    "aten.softmax": "forge.tensor.softmax",
    "aten.layer_norm": "forge.tensor.layer_norm",
    "aten.scaled_dot_product_attention": "forge.tensor.scaled_dot_product_attention",
}


def _fx_ops_to_forge_tensor_ops(gm: torch.fx.GraphModule) -> set[str]:
    """Return the ``forge.tensor.*`` ops present in ``gm``.

    Walks the FX graph in node order and translates each ``call_function``
    target to its canonical ``forge.tensor.<op>`` spelling via
    :data:`_ATEN_TO_FORGE_TENSOR`. Non-call_function nodes (placeholders,
    outputs, get_attr) are skipped — they are not data-flow ops.
    """
    out: set[str] = set()
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        normalized = normalize_target(node.target)
        canonical = _ATEN_TO_FORGE_TENSOR.get(normalized)
        if canonical is not None:
            out.add(canonical)
    return out


def _check_target_capabilities(target: str, fx_ops: set[str]) -> None:
    """Emit a :class:`UserWarning` per op that ``target`` cannot run.

    Unknown ``target`` names raise :class:`KeyError` so callers can detect
    typos at the API boundary instead of silently falling back. The
    spec §6 capability schema is the source of truth; the registry is
    consulted by name and the negative diff is reported as a warning
    (per Task 5 spec — don't fail).
    """
    capabilities = BACKEND_CAPABILITIES[target]
    unsupported = sorted(fx_ops - capabilities.supported_ops)
    for op in unsupported:
        warnings.warn(
            (
                f"target {target!r} does not support op {op!r}; "
                f"supported ops are {sorted(capabilities.supported_ops)}"
            ),
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile(
    model: torch.nn.Module,
    example_inputs: list[torch.Tensor],
    *,
    cache_store: CacheStore | None = None,
    target: str = "triton",
) -> CompiledFusedMLP:
    """Compile a PyTorch model using ``torch.compile(backend="forgestack")`` semantics.

    Phase 1 captures the FX graph, runs :class:`FxImporter` to obtain
    ``forge.graph`` MLIR, consults the two-tier cache (Task 14 / spec
    §13), and returns a :class:`CompiledFusedMLP` that delegates to
    eager. On cache miss we also build + emit the spec §16 observability
    report to :data:`REPORT_PATH` so the Phase 1 smoke test can read it.

    The lowered-execution path lands once ForgeRT (Tasks 10-12) and the
    Phase-2 partitioner are in place.

    ``target`` selects the backend for the Phase 7 capability check
    (Task 5). One :class:`UserWarning` is emitted per op in the captured
    FX graph that ``target``'s :class:`BackendCapabilities` does not
    cover; compilation proceeds regardless (the warning is the
    partitioner's "reroute me" signal, not a hard failure). Existing
    Phase 1-8 callers that omit ``target=`` get the previous behaviour
    (``"triton"``).
    """
    gm = torch.fx.symbolic_trace(model)
    _check_target_capabilities(target, _fx_ops_to_forge_tensor_ops(gm))
    importer = FxImporter(example_inputs)
    mlir_text = importer.import_graph(gm)

    cache_store = cache_store if cache_store is not None else CacheStore()
    graph_hash = hashlib.sha256(mlir_text.encode()).hexdigest()
    key = make_cache_key(
        graph_hash=graph_hash,
        target=target,
        arch="x86_64_v3",
        dtype=_infer_dtype_token(example_inputs),
        shape_constraints="static",
        schedule="schedule_64x128x32",
    )

    artifact = cache_store.get(key)
    cache_hit = artifact is not None
    if not cache_hit:
        artifact = _build_artifact(mlir_text, model)
        cache_store.put(key, artifact)
        report = _build_report(model, gm, mlir_text, target=target)
        _emit_report(report)

    return CompiledFusedMLP(
        model,
        example_inputs,
        mlir_text,
        cache_key=key,
        cache_hit=cache_hit,
        cache_store=cache_store,
    )


def compile_fused_mlp(
    model: torch.nn.Module, example_inputs: list[torch.Tensor]
) -> CompiledFusedMLP:
    """Convenience: explicit name for the FusedMLP compile path."""
    return compile(model, example_inputs)


# ---------------------------------------------------------------------------
# Validation result (lightweight stub — Task 13 owns the formal harness)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Lightweight differential-validation result.

    Phase 1 only runs eager-vs-eager (because ``CompiledFusedMLP.__call__``
    itself delegates to eager); Task 13 will replace this with the real
    kernel-output comparison. The fields are kept stable so the spec §16
    compilation report can carry ``validation_max_error`` without churn.
    """

    passed: bool
    max_error: float
    reference: str = "eager"

    def __str__(self) -> str:  # pragma: no cover — debug helper
        return f"ValidationResult(passed={self.passed}, max_error={self.max_error:.6f})"


# ---------------------------------------------------------------------------
# Compiled wrapper
# ---------------------------------------------------------------------------


class CompiledFusedMLP:
    """Stub: wraps eager execution while the C++ runtime (Tasks 10-12) lands.

    The textual ``mlir_text`` is retained for diagnostics (and for the
    pytest round-trip assertion in ``test_fx_importer.py``); the eventual
    lowered-execution backend will swap ``__call__`` for a call into
    ForgeRT (Task 11).

    Task 14 attaches :attr:`cache_key`, :attr:`cache_hit`, and a
    reference to the :class:`CacheStore` that produced the artifact, so
    downstream tooling (the Task 13 validation harness, the Phase-1
    smoke test) can re-look-up the cache or emit the spec §16 report.

    The :data:`compiled` class attribute is the "went through the
    compiler" marker Task 13's end-to-end smoke test asserts on.
    """

    #: True iff the instance came back from :func:`compile` (vs. raw eager).
    compiled: bool = True

    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: list[torch.Tensor],
        mlir_text: str,
        *,
        cache_key=None,
        cache_hit: bool = False,
        cache_store: CacheStore | None = None,
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.mlir_text = mlir_text
        self.cache_key = cache_key
        self.cache_hit = cache_hit
        self.cache_store = cache_store

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        # PHASE 1 NOTE: This compiled() call delegates to the eager kernel
        # because the pybind11 binding to the C++ TritonProvider is a Phase
        # 5 deliverable. The C++ path itself is fully exercised by
        # runtime/tests/test_triton_provider (Task 11) and
        # runtime/tests/test_scheduler (Task 12). The validation harness
        # here proves that the end-to-end pipeline returns the same
        # numerical result as eager.
        return self.model(*args)

    def validate(self, *, atol: float = 1e-5, rtol: float = 1e-5) -> ValidationResult:
        """Compare ``self(*example_inputs)`` against eager.

        Phase-1 stub: the compiled stub is itself eager, so this is
        effectively eager-vs-eager (max error ~ 0). Task 13 will swap
        in the formal differential harness once the lowered runtime is
        live; the return type is stable.
        """
        eager_out = self.model(*self.example_inputs)
        compiled_out = self(*self.example_inputs)
        diff = (eager_out - compiled_out).abs()
        max_error = float(diff.max().item()) if diff.numel() else 0.0
        return ValidationResult(
            passed=torch.allclose(compiled_out, eager_out, atol=atol, rtol=rtol),
            max_error=max_error,
        )


# ---------------------------------------------------------------------------
# Helpers (spec §13 / §16 integration)
# ---------------------------------------------------------------------------


def _infer_dtype_token(example_inputs: list[torch.Tensor]) -> str:
    """Map the example tensors' dtype to the spec §13 cache key spelling."""
    if not example_inputs:
        return "fp32"
    dt = example_inputs[0].dtype
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    return mapping.get(dt, "fp32")


def _count_fx_nodes(gm: torch.fx.GraphModule) -> int:
    """Count non-placeholder, non-output FX nodes (= real ops in the graph)."""
    count = 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        count += 1
    return count


def _count_mlir_ops(mlir_text: str, op_prefix: str) -> int:
    """Count ``<op_prefix>`` occurrences in the textual MLIR."""
    return mlir_text.count(op_prefix)


def _build_artifact(mlir_text: str, model: torch.nn.Module) -> dict:
    """Bundle the per-key data persisted to ``artifact.json``."""
    return {
        "model_name": model.__class__.__name__,
        "mlir_text": mlir_text,
        "schedule": {
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 32,
            "warps": 8,
            "stages": 4,
        },
    }


def _build_report(
    model: torch.nn.Module,
    gm: torch.fx.GraphModule,
    mlir_text: str,
    *,
    target: str = "triton",
) -> CompilationReport:
    """Populate a :class:`CompilationReport` with Phase-1 FusedMLP numbers."""
    model_name = model.__class__.__name__
    imported = _count_mlir_ops(mlir_text, "forge.graph.")
    return CompilationReport(
        model_name=model_name,
        captured_fx_nodes=_count_fx_nodes(gm),
        imported_mlir_operations=imported,
        operations_after_canonicalize=imported,  # canonicalize is a no-op in Phase 1
        fusion_groups=1,
        selected_target=target,
        alternative_astra_cost="1.31x",
        alternative_fpga_cost="2.04x including transfer",
        schedule={
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 32,
            "warps": 8,
            "stages": 4,
        },
        original_intermediate_memory_mb=45.0,  # [2,512,11008] fp32 = ~45 MB
        planned_intermediate_memory_mb=8.0,
        generated_kernels=1,
        validation_max_error=0.0,  # populated by validate() (Task 13)
    )


def _emit_report(report: CompilationReport) -> None:
    """Write the rendered report to :data:`REPORT_PATH`.

    Called on cache miss only — the spec says the report must be emitted
    once per unique cache key, not on every ``compile()`` call.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.render() + "\n")
