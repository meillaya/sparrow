"""Map PyTorch FX aten ops to forge.graph ops. Single source of truth for Task 8.

The :func:`map_node_to_forge` helper centralises the (FX node target string) →
(``forge.graph.<op>`` name + shape inference callable) translation.
``fx_importer.FxImporter`` (which walks a captured ``torch.fx.GraphModule``)
consults this table; nothing else in Sparrow frontend should re-derive the mapping.

Supported ops (shipped scope)
-----------------------------
``aten.mm``  / ``aten.matmul`` → ``forge.graph.matmul``
``aten.add``                   → ``forge.graph.add``
``aten.gelu``                  → ``forge.graph.gelu``
``aten.view`` / ``aten._unsafe_view`` → ``forge.graph.view``
(the latter is the stride-bypassing variant FX emits around ``aten.mm``)

``aten.layer_norm``                       → ``forge.tensor.layer_norm``
``aten.softmax``                          → ``forge.tensor.softmax``
``aten.scaled_dot_product_attention``     → ``forge.tensor.scaled_dot_product_attention``
(these three are Task 14 additions — they route directly into the
``forge.tensor`` dialect rather than ``forge.graph`` because they were
not modelled in the Phase 1 graph dialect.)

Recognised-but-not-yet-implemented (raise :class:`NotImplementedError`)
-----------------------------------------------------------------------
``aten.relu`` / ``aten.silu`` / ``aten.mul`` — activations/elementwise;
  the graph dialect currently stops at the four ops above.
``aten.reshape`` / ``aten.permute`` / ``aten.transpose`` — not lowered today.
``aten.sum`` / ``aten.mean`` — reductions, not lowered today.
``aten.bmm`` / ``aten.baddbmm`` — batched matmul, not lowered today.

DType translation
-----------------
``DTYPE_TO_FORGE`` mirrors the order in
:mod:`compiler.lib.Dialect.Graph.ForgeDType`. i4 has no equivalent ``torch.dtype``
yet — when PyTorch ships it, add the line here and the same string is forward-compatible.
"""

from __future__ import annotations

import torch

# Maps torch.fx node.op strings to the corresponding forge op name. Phase 1
# ops route to ``forge.graph.*``; Task 14's tensor-dialect additions route
# directly to ``forge.tensor.*`` (the FX importer emits the latter without
# going through the GraphToTensor pass).
SUPPORTED_ATEN_OPS: dict[str, str] = {
    "aten.mm": "forge.graph.matmul",
    "aten.matmul": "forge.graph.matmul",
    "aten.add": "forge.graph.add",
    "aten.gelu": "forge.graph.gelu",
    "aten.view": "forge.graph.view",
    "aten._unsafe_view": "forge.graph.view",
    "aten.layer_norm": "forge.tensor.layer_norm",
    "aten.softmax": "forge.tensor.softmax",
    "aten.scaled_dot_product_attention": "forge.tensor.scaled_dot_product_attention",
}

# Operations that are recognised but NOT YET IMPLEMENTED in Phase 1 forge.graph.
# The importer raises NotImplementedError when it encounters any of these.
UNSUPPORTED_BUT_RECOGNISED: frozenset[str] = frozenset(
    {
        "aten.relu",
        "aten.silu",
        "aten.mul",
        "aten.reshape",
        "aten.permute",
        "aten.transpose",
        "aten.sum",
        "aten.mean",
        "aten.bmm",
        "aten.baddbmm",  # batched matmul, not lowered today
    }
)

DTYPE_TO_FORGE: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.int32: "i32",
    torch.int8: "i8",
    # i4 not yet representable as torch.dtype
}


def normalize_target(target: object) -> str:
    """Reduce a ``node.target`` to the canonical ``aten.<op>`` form.

    ``torch.fx.symbolic_trace`` stores targets as free-function objects
    (``operator.matmul``, ``torch._C._nn.gelu``); ``make_fx`` stores them
    as ``OpOverload`` instances (``torch.ops.aten.matmul.default``). This
    helper reduces both to the substring the mapping tables are keyed on.
    """
    if isinstance(target, str):
        return target
    if hasattr(target, "__str__"):
        try:
            raw = str(target)
        except Exception:  # pragma: no cover
            raw = ""
        if raw.startswith("aten.") and raw.count(".") >= 2:
            namespace, op_name, _overload = raw.split(".", 2)
            return f"{namespace}.{op_name}"
        if raw.startswith("aten."):
            return raw
    name = getattr(target, "__name__", None) or getattr(target, "_name", None)
    if name is None:
        return ""
    if name.startswith("aten."):
        return name
    builtins_alias = {
        "matmul": "aten.matmul",
        "add": "aten.add",
        "mul": "aten.mul",
    }.get(name)
    if builtins_alias is not None:
        return builtins_alias
    c_alias = {
        "gelu": "aten.gelu",
        "relu": "aten.relu",
        "silu": "aten.silu",
    }.get(name)
    if c_alias is not None:
        return c_alias
    return f"aten.{name}"


def map_node_to_forge(target: object) -> str:
    """Return the ``forge.graph.<op>`` name for an FX node target.

    Raises :class:`NotImplementedError` (with a message naming the missing op)
    if the target maps to one of :data:`UNSUPPORTED_BUT_RECOGNISED`.
    Raises :class:`KeyError` for unknown targets — the caller is expected
    to wrap with a more contextual message.
    """
    normalized = normalize_target(target)
    if normalized in SUPPORTED_ATEN_OPS:
        return SUPPORTED_ATEN_OPS[normalized]
    if normalized in UNSUPPORTED_BUT_RECOGNISED:
        msg = (
            f"{normalized!s} is recognised by the Sparrow frontend but "
            f"is not yet lowered by the shipped FX importer."
        )
        raise NotImplementedError(msg)
    raise KeyError(f"unknown FX op target: {target!r} (normalised to {normalized!r})")
