"""Lower Astra microcode IR to instruction words.

The codegen walks verified ``astra`` dialect operations and encodes each
one with the assembler. It no longer scrapes textual IR with a regex:
the input is a parsed ``ModuleOp`` and every address/size comes from the
op's properties.
"""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func as FuncDialect
from xdsl.ir import Operation
from xdsl.parser import Parser

from compiler.lib.Dialect.Astra import (
    Astra,
    ComputeMatmulOp,
    MicrocodeDmaLoadOp,
    MicrocodeDmaStoreOp,
    MicrocodeMatmulOp,
    MicrocodeVectorAddOp,
    MicrocodeVectorGeluOp,
)
from compiler.lib.Dialect.Astra.assembler import assemble_from_dialect

AstraOp = (
    ComputeMatmulOp
    | MicrocodeDmaLoadOp
    | MicrocodeDmaStoreOp
    | MicrocodeMatmulOp
    | MicrocodeVectorGeluOp
    | MicrocodeVectorAddOp
)


def _walk(operation: Operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for nested in block.ops:
                yield from _walk(nested)


def emit_astra_module(module: ModuleOp) -> list[int]:
    """Encode every Astra op in ``module`` in program order."""
    operations: list[AstraOp] = []
    for operation in _walk(module):
        if isinstance(operation, AstraOp):
            operations.append(operation)
    return assemble_from_dialect(operations)


def _context() -> Context:
    context = Context(allow_unregistered=False)
    context.load_dialect(Builtin)
    context.load_dialect(FuncDialect)
    context.load_dialect(Astra)
    return context


def emit_astra(region: str) -> list[int]:
    """Parse ``region`` as MLIR and encode its Astra operations.

    Kept as the textual entry point used by the CLI and tests; the
    parsing step is the only thing the string form adds. No regex is
    involved in decoding operation fields.
    """
    module = Parser(_context(), region, "<astra-region>").parse_module()
    module.verify()
    return emit_astra_module(module)


__all__ = ("emit_astra", "emit_astra_module")
