"""Sparrow ``torch.compile`` backend hook (Tasks 8 + 14, Phase 1).

Spec §3 says Sparrow frontend has six jobs:

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
import importlib
import os
import platform
import re
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

BACKEND = "sparrow"

# The C++ fused kernel is a naive triple loop. Above this many
# multiply-adds the native path falls back to eager instead of turning a
# quick demo into a multi-minute stall. 1 << 27 is about 134M MACs.
_NATIVE_MAX_MULTIPLY_ADDS: Final = 1 << 27

# Report output path. It lives under the ignored cache root because the
# report is diagnostic output, not a tracked artifact.
REPORT_PATH = Path(".cache/sparrow/reports/phase1-report.txt")

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
    """Import a PyTorch model, cache its MLIR, and return a callable.

    This is not a registered ``torch.compile`` backend. The function
    captures the FX graph, runs :class:`FxImporter` to obtain MLIR,
    consults the two-tier cache, and returns a :class:`CompiledFusedMLP`.
    When the optional ``_sparrow_rt`` extension is importable and the graph
    is the fused MLP pattern, the callable runs the native kernel and
    reports ``lowered=True``. Otherwise it delegates to eager PyTorch and
    reports ``lowered=False``. On a cache miss the function also emits the
    observability report to :data:`REPORT_PATH` and attaches it to the
    callable.

    ``target`` selects the backend for the capability check. One
    :class:`UserWarning` is emitted per captured op that the target's
    :class:`BackendCapabilities` does not cover. The warning is
    advisory; compilation proceeds.
    """
    gm = torch.fx.symbolic_trace(model)
    _check_target_capabilities(target, _fx_ops_to_forge_tensor_ops(gm))
    importer = FxImporter(example_inputs)
    mlir_text = importer.import_graph(gm)

    native_runner = _native_runner_for(gm, example_inputs)
    execution_path = "native" if native_runner is not None else "eager"

    cache_store = cache_store if cache_store is not None else CacheStore()
    graph_hash = hashlib.sha256(mlir_text.encode()).hexdigest()
    key = make_cache_key(
        graph_hash=graph_hash,
        target=target,
        arch=_detect_arch(),
        dtype=_infer_dtype_token(example_inputs),
        shape_constraints="static",
        schedule="schedule_64x128x32",
    )

    artifact = cache_store.get(key)
    cache_hit = artifact is not None
    report: CompilationReport | None = None
    if not cache_hit:
        artifact = _build_artifact(
            mlir_text,
            model,
            target=target,
            execution_path=execution_path,
        )
        cache_store.put(key, artifact)
        report = _build_report(
            model,
            gm,
            mlir_text,
            target=target,
            execution_path=execution_path,
            generated_kernels=1 if native_runner is not None else None,
        )
        _emit_report(report)

    return CompiledFusedMLP(
        model,
        example_inputs,
        mlir_text,
        cache_key=key,
        cache_hit=cache_hit,
        cache_store=cache_store,
        report=report,
        native_runner=native_runner,
    )


def compile_fused_mlp(
    model: torch.nn.Module, example_inputs: list[torch.Tensor]
) -> CompiledFusedMLP:
    """Convenience: explicit name for the FusedMLP compile path."""
    return compile(model, example_inputs)


# ---------------------------------------------------------------------------
# Validation result (shared by the eager and native paths)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Differential-validation result with an explicit execution path.

    The shipped eager path compares eager output to eager output. The
    fields say so; callers must not read ``passed`` as proof that a
    lowered kernel ran.
    """

    passed: bool
    max_error: float
    reference: str = "eager"
    execution: str = "eager"
    lowered: bool = False

    def __str__(self) -> str:  # pragma: no cover — debug helper
        return (
            f"ValidationResult(passed={self.passed}, max_error={self.max_error:.6f}, "
            f"execution={self.execution!r})"
        )


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
    #: ``eager`` until a lowered callable replaces ``__call__``.
    execution: str = "eager"
    #: True only when ``__call__`` runs a lowered artifact.
    lowered: bool = False

    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: list[torch.Tensor],
        mlir_text: str,
        *,
        cache_key=None,
        cache_hit: bool = False,
        cache_store: CacheStore | None = None,
        report: CompilationReport | None = None,
        native_runner=None,
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.mlir_text = mlir_text
        self.cache_key = cache_key
        self.cache_hit = cache_hit
        self.cache_store = cache_store
        self.report = report
        self._native_runner = native_runner
        self.lowered = native_runner is not None
        self.execution = "native" if self.lowered else "eager"

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        if self._native_runner is not None and len(args) == 3:
            x, weight, bias = args
            if (
                x.dim() == 3
                and weight.dim() == 2
                and bias.dim() == 1
                and x.dtype == torch.float32
                and weight.dtype == torch.float32
                and bias.dtype == torch.float32
                and x.shape[-1] == weight.shape[0]
                and bias.shape[0] == weight.shape[1]
            ):
                flat = x.reshape(-1, x.shape[-1]).contiguous()
                out = self._native_runner(
                    flat.detach().cpu().numpy(),
                    weight.detach().cpu().numpy(),
                    bias.detach().cpu().numpy(),
                )
                return torch.from_numpy(out).reshape(*x.shape[:-1], weight.shape[1])
            warnings.warn(
                "native fused MLP path requires rank-3 fp32 x, rank-2 fp32 weight, "
                "and rank-1 fp32 bias; running eager instead",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.model(*args)

    def validate(self, *, atol: float = 1e-5, rtol: float = 1e-5) -> ValidationResult:
        """Compare ``self(*example_inputs)`` against eager.

        When ``lowered`` is false this compares eager to eager and the
        result names the eager execution path. The native path replaces
        this comparison with a real kernel-vs-eager check.
        """
        eager_out = self.model(*self.example_inputs)
        compiled_out = self(*self.example_inputs)
        diff = (eager_out - compiled_out).abs()
        max_error = float(diff.max().item()) if diff.numel() else 0.0
        if self.report is not None:
            self.report.validation_max_error = max_error
            self.report.execution_path = self.execution
        return ValidationResult(
            passed=torch.allclose(compiled_out, eager_out, atol=atol, rtol=rtol),
            max_error=max_error,
            execution=self.execution,
            lowered=self.lowered,
        )


# ---------------------------------------------------------------------------
# Helpers (spec §13 / §16 integration)
# ---------------------------------------------------------------------------


def _detect_arch() -> str:
    """Return the host architecture token used in the cache key.

    The compiler does not emit machine code yet, but the key must still
    differ between hosts. An explicit ``SPARROW_ARCH`` wins.
    """
    override = os.environ.get("SPARROW_ARCH")
    if override:
        return override
    machine = platform.machine().lower()
    aliases = {
        "x86_64": "x86_64_v3",
        "amd64": "x86_64_v3",
        "arm64": "aarch64",
    }
    return aliases.get(machine, machine or "unknown")


def _infer_dtype_token(example_inputs: list[torch.Tensor]) -> str:
    """Map every example tensor's dtype to the spec §13 cache key spelling.

    A mixed-precision call must not share a cache entry with a uniform
    one, so the token is the ordered unique set joined by ``+``.
    """
    mapping = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    tokens: list[str] = []
    for tensor in example_inputs:
        token = mapping.get(tensor.dtype, "fp32")
        if token not in tokens:
            tokens.append(token)
    return "+".join(tokens) if tokens else "fp32"


def _count_fx_nodes(gm: torch.fx.GraphModule) -> int:
    """Count non-placeholder, non-output FX nodes (= real ops in the graph)."""
    count = 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        count += 1
    return count


_FORGE_OP_RE: Final = re.compile(r"\bforge\.(?:graph|tensor)\.[a-z_]+")


def _count_mlir_ops(mlir_text: str) -> int:
    """Count forge operation names in the textual MLIR.

    Types such as ``!forge.tensor<...>`` are excluded, so the count is
    operations, not mentions of the dialect.
    """
    return len(_FORGE_OP_RE.findall(mlir_text))


def _build_artifact(
    mlir_text: str,
    model: torch.nn.Module,
    *,
    target: str = "triton",
    execution_path: str = "eager",
) -> dict:
    """Bundle the per-key data persisted to ``artifact.json``."""
    return {
        "model_name": model.__class__.__name__,
        "execution_path": execution_path,
        "target": target,
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
    execution_path: str = "eager",
    generated_kernels: int | None = None,
) -> CompilationReport:
    """Populate the measured fields; leave unmeasured fields as ``None``."""
    model_name = model.__class__.__name__
    imported = _count_mlir_ops(mlir_text)
    return CompilationReport(
        model_name=model_name,
        execution_path=execution_path,
        captured_fx_nodes=_count_fx_nodes(gm),
        imported_mlir_operations=imported,
        operations_after_canonicalize=imported,  # canonicalize is a no-op in Phase 1
        fusion_groups=None,
        selected_target=target,
        alternative_astra_cost=None,
        alternative_fpga_cost=None,
        schedule={
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 32,
            "warps": 8,
            "stages": 4,
        },
        original_intermediate_memory_mb=None,
        planned_intermediate_memory_mb=None,
        generated_kernels=generated_kernels,
        validation_max_error=None,
    )


def _emit_report(report: CompilationReport) -> None:
    """Write the rendered report to :data:`REPORT_PATH`.

    Called on cache miss only — the spec says the report must be emitted
    once per unique cache key, not on every ``compile()`` call.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report.render() + "\n")


def _native_runner_for(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
):
    """Return ``_sparrow_rt.run_fused_mlp`` for the fused MLP shape only.

    The pattern is one matmul, one add, and one gelu with a rank-3 x, a
    rank-2 weight, and a rank-1 bias, all fp32. Anything else returns
    ``None`` and keeps the eager path.
    """
    if len(example_inputs) != 3:
        return None
    x, weight, bias = example_inputs
    if (
        x.dim() != 3
        or weight.dim() != 2
        or bias.dim() != 1
        or x.dtype != torch.float32
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or x.shape[-1] != weight.shape[0]
        or bias.shape[0] != weight.shape[1]
    ):
        return None

    multiply_adds = int(x.shape[0]) * int(x.shape[1]) * int(weight.shape[0]) * int(weight.shape[1])
    if multiply_adds > _NATIVE_MAX_MULTIPLY_ADDS:
        return None

    targets: list[str] = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            targets.append(normalize_target(node.target))
    if sorted(targets) != ["aten.add", "aten.gelu", "aten.matmul"]:
        return None

    try:
        module = importlib.import_module("_sparrow_rt")
    except ImportError:
        return None
    runner = getattr(module, "run_fused_mlp", None)
    return runner if callable(runner) else None
