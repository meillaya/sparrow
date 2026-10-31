"""MicroGPU backend (spec §11).

The microgpu backend exposes a single-warp SIMT bytecode interpreter
running on the host. The package is split into:

* :mod:`backends.microgpu.isa` — typed 32-bit opcode enum + encode/decode.
* :mod:`backends.microgpu.asm` — text-format assembler + disassembler,
  plus the dialect ↔ bytecode round-trip used by the codegen pass.
* :mod:`backends.microgpu.interp` — Python reference interpreter with
  race + out-of-bounds detection and a dep DAG emitter for the
  Phase 7 partitioner.

The C++ side (``runtime/include/forge_rt/providers/microgpu_provider.hpp``
+ ``runtime/lib/providers/microgpu_provider.cpp``) wires this package as
a :class:`forge_rt::DeviceProvider` so the scheduler can dispatch
``TargetKind::MicroGPU`` artifacts through the existing ABI.
"""

from __future__ import annotations

from backends.microgpu.isa import MicroOp, OpCode, decode, encode

__all__ = ("MicroOp", "OpCode", "decode", "encode")
