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
from xdsl.utils.exceptions import VerifyException

from forge_torch.cache import make_cache_key
from forge_torch.cache_store import CacheStore
from forge_torch.capabilities import BACKEND_CAPABILITIES
from forge_torch.fx_importer import FxImporter
from forge_torch.ops import (
    ATEN_OPS,
    is_shape_metadata_node,
    normalize_target,
    spec_for_aten,
    spec_for_method,
)
from forge_torch.plan import BuiltPlan, PlanBuildError, build_plan
from forge_torch.report import CompilationReport

BACKEND = "sparrow"

# The C++ fused kernel is a naive triple loop. Above this many
# multiply-adds the native path falls back to eager instead of turning a
# quick demo into a multi-minute stall. 1 << 27 is about 134M MACs.
_NATIVE_MAX_MULTIPLY_ADDS: Final = 1 << 27

# The CPU plan kernels are naive loops too. Skip the plan for graphs above
# the same work budget so a canonical-shape test does not stall for minutes.
_PLAN_MAX_MULTIPLY_ADDS: Final = 1 << 27

# Report output path. It lives under the ignored cache root because the
# report is diagnostic output, not a tracked artifact.
REPORT_PATH = Path(".cache/sparrow/reports/phase1-report.txt")

# ---------------------------------------------------------------------------
# Phase 7 capability check helpers (Task 5, spec §6).
#
# ``compile(target=...)`` looks up ``BACKEND_CAPABILITIES[target]`` and
# emits a ``UserWarning`` for every op in the captured FX graph that the
# target cannot run. The FX target -> ``forge.tensor.<op>`` translation
# now comes from the canonical table in :mod:`forge_torch.ops`; the
# capability schema consumes the same ``forge.tensor.<op>`` spellings as
# the partitioner. Targets the frontend does not lower are skipped — the
# capability check is advisory and the importer owns the hard error.
# ---------------------------------------------------------------------------

_ATEN_TO_FORGE_TENSOR: Final[dict[str, str]] = {name: spec.op for name, spec in ATEN_OPS.items()}


def _module_for_node(gm: torch.fx.GraphModule, node: torch.fx.Node):
    if node.op != "call_module":
        return None
    try:
        return gm.get_submodule(str(node.target))
    except AttributeError:
        return None


def _fx_ops_to_forge_tensor_ops(gm: torch.fx.GraphModule) -> set[str]:
    """Return the ``forge.tensor.*`` ops present in ``gm``.

    Walks the FX graph in node order and translates call_function,
    call_method, and call_module nodes to their canonical
    ``forge.tensor.<op>`` spelling. Non-data nodes (placeholders, outputs,
    shape bookkeeping) are skipped because they have no spec entry.
    """
    out: set[str] = set()
    for node in gm.graph.nodes:
        if is_shape_metadata_node(node):
            continue
        if node.op == "call_function":
            try:
                out.add(spec_for_aten(node.target).op)
            except (KeyError, NotImplementedError):
                continue
        elif node.op == "call_method":
            try:
                out.add(spec_for_method(str(node.target)).op)
            except KeyError:
                continue
        elif node.op == "call_module":
            module = _module_for_node(gm, node)
            if isinstance(module, torch.nn.LayerNorm):
                out.add("forge.tensor.layer_norm")
            elif isinstance(module, torch.nn.Linear):
                out.add("forge.tensor.gemm")
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
    The callable has three execution paths. The fused MLP pattern uses the
    native kernel when ``_sparrow_rt`` is built. Other graphs that lower to
    the CPU kernel registry run an execution plan through
    ``_sparrow_rt.execute_plan``. Graphs the registry cannot lower fall back
    to eager, and ``planning_error`` records why. Parameter tensors found
    through ``get_attr`` are lifted into plan inputs and bound to the live
    module tensors on each call. On a cache miss the function also writes
    the observability report and attaches it to the callable.

    ``target`` selects the backend for the capability check. One
    :class:`UserWarning` is emitted per captured op that the target's
    :class:`BackendCapabilities` does not cover. The warning is
    advisory; compilation proceeds.
    """
    gm = torch.fx.symbolic_trace(model)
    _check_target_capabilities(target, _fx_ops_to_forge_tensor_ops(gm))
    importer = FxImporter(example_inputs)
    module = importer.import_module(gm)
    mlir_text = str(module)
    lifted_inputs = tuple(importer.lifted_tensors)

    native_runner = _native_runner_for(gm, example_inputs)
    built_plan: BuiltPlan | None = None
    plan_runtime = None
    planning_error: str | None = None
    if native_runner is None and target == "triton":
        built_plan, plan_runtime, planning_error = _build_cpu_plan(module)

    if native_runner is not None:
        execution_path = "native"
    elif built_plan is not None:
        execution_path = "cpu"
    else:
        execution_path = "eager"

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
            plan_json=built_plan.to_json() if built_plan is not None else None,
        )
        cache_store.put(key, artifact)
        report = _build_report(
            model,
            gm,
            mlir_text,
            target=target,
            execution_path=execution_path,
            generated_kernels=(
                1
                if native_runner is not None
                else (len(built_plan.plan["nodes"]) if built_plan is not None else None)
            ),
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
        plan_json=built_plan.to_json() if built_plan is not None else None,
        plan_runtime=plan_runtime,
        planning_error=planning_error,
        lifted_inputs=lifted_inputs,
    )


def compile_graph(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    cache_store: CacheStore | None = None,
    target: str = "triton",
) -> CompiledFusedMLP:
    """Compile an already-captured FX graph.

    ``torch.compile`` backends receive this form, with parameters lifted
    into graph placeholders by Dynamo. Re-tracing a GraphModule preserves
    those placeholders, so the general compile path handles it.
    """
    return compile(gm, example_inputs, cache_store=cache_store, target=target)


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

    The fields name the path that ran. Callers must not read ``passed``
    as proof that a lowered kernel ran unless ``lowered`` is true.
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
    """Compiled graph wrapper.

    The textual ``mlir_text`` is retained for diagnostics (and for the
    pytest round-trip assertion in ``test_fx_importer.py``). The callable
    prefers the native fused-MLP kernel, then a CPU execution plan, then
    eager for graphs the CPU registry cannot lower. Lifted parameter
    tensors are appended to the caller's arguments when the plan runs. The eventual
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
        plan_json: str | None = None,
        plan_runtime=None,
        planning_error: str | None = None,
        lifted_inputs: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.mlir_text = mlir_text
        self.cache_key = cache_key
        self.cache_hit = cache_hit
        self.cache_store = cache_store
        self.report = report
        self._native_runner = native_runner
        self._plan_json = plan_json
        self._plan_runtime = plan_runtime
        self.planning_error = planning_error
        self._lifted_inputs = lifted_inputs
        self.lowered = native_runner is not None or plan_json is not None
        if native_runner is not None:
            self.execution = "native"
        elif plan_json is not None:
            self.execution = "cpu"
        else:
            self.execution = "eager"

    def __call__(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
        if self._plan_json is not None and self._plan_runtime is not None:
            arrays = [arg.detach().cpu().numpy() for arg in args]
            arrays.extend(tensor.detach().cpu().numpy() for tensor in self._lifted_inputs)
            outputs = self._plan_runtime.execute_plan(self._plan_json, arrays)
            if len(outputs) == 1:
                return torch.from_numpy(outputs[0])
            return tuple(torch.from_numpy(output) for output in outputs)
        return self.model(*args)

    def validate(self, *, atol: float = 1e-4, rtol: float = 1e-4) -> ValidationResult:
        """Compare ``self(*example_inputs)`` against eager.

        When ``lowered`` is false this compares eager to eager and the
        result names the eager execution path. The native path replaces
        this comparison with a real kernel-vs-eager check.
        """
        eager_out = self.model(*self.example_inputs)
        compiled_out = self(*self.example_inputs)
        if isinstance(compiled_out, tuple):
            raise TypeError("validate() supports single-output graphs")
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
    plan_json: str | None = None,
) -> dict:
    """Bundle the per-key data persisted to ``artifact.json``."""
    artifact = {
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
    if plan_json is not None:
        artifact["plan_json"] = plan_json
    return artifact


def _shape_numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _plan_cost(built: BuiltPlan) -> int:
    """Estimate the plan's dominant work in multiply-adds or elements."""
    artifacts = built.plan["artifacts"]
    total = 0
    for node in built.plan["nodes"]:
        artifact = artifacts[node["artifact"]]
        arguments = artifact["arguments"]
        shapes = [argument["shape"] for argument in arguments]
        if artifact["entry_point"] == "forge.tensor.gemm":
            if len(shapes) < 3:
                return _PLAN_MAX_MULTIPLY_ADDS + 1
            m = shapes[0][-2] if not artifact.get("transpose_a") else shapes[0][-1]
            n = shapes[1][-1] if not artifact.get("transpose_b") else shapes[1][-2]
            total += _shape_numel(shapes[0]) * int(n)
            if m <= 0 or n <= 0:
                return _PLAN_MAX_MULTIPLY_ADDS + 1
        elif artifact["entry_point"] == "forge.tensor.scaled_dot_product_attention":
            if shapes:
                shape = shapes[0]
                total += _shape_numel(shape) * int(shape[-2]) if len(shape) >= 2 else 0
        else:
            if shapes:
                total += _shape_numel(shapes[-1])
    return total


def _build_cpu_plan(
    module,
) -> tuple[BuiltPlan | None, object | None, str | None]:
    """Try to lower ``module`` and build a CPU plan.

    Returns ``(plan, runtime, error)``. A missing ``_sparrow_rt`` is not
    an error: the pure-Python frontend still works without the optional
    native module. A graph the CPU registry cannot lower returns a
    planning error and keeps the eager fallback.
    """
    try:
        runtime = importlib.import_module("_sparrow_rt")
    except ImportError:
        return None, None, None
    from compiler.lib.pipeline import lower_to_tensor

    try:
        lowered = lower_to_tensor(module)
        plan = build_plan(lowered)
        if _plan_cost(plan) > _PLAN_MAX_MULTIPLY_ADDS:
            return None, runtime, "graph exceeds the CPU kernel work budget"
        return plan, runtime, None
    except (PlanBuildError, NotImplementedError, KeyError, VerifyException) as exc:
        return None, runtime, str(exc)


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
        if is_shape_metadata_node(node):
            continue
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


#: Public alias for the general compiled graph.
CompiledGraph = CompiledFusedMLP
