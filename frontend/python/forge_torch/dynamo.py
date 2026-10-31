"""``torch.compile`` backend entry point for Sparrow."""

from __future__ import annotations

from collections.abc import Callable

import torch

from forge_torch.backend import compile_graph


def backend(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Compile a Dynamo graph through the Sparrow frontend.

    Dynamo lifts model parameters into graph placeholders, so the
    returned callable expects the same argument order as the graph. The
    Dynamo wrapper expects the graph's tuple-shaped output, even when the
    graph has one value.
    """
    compiled = compile_graph(gm, list(example_inputs))

    def call(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = compiled(*args)
        if isinstance(output, tuple):
            return output
        return (output,)

    return call


__all__ = ("backend",)
