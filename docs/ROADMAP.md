# Sparrow roadmap

The audit and [SPEC.md](SPEC.md) describe the contracts. This file tracks the
remaining milestones from the completion plan. M1 through M7 are implemented.

## Landed

### M1: general CPU execution path

- `forge.tensor.reshape` lowers `forge.graph.view`.
- `compiler.lib.context` and `compiler.lib.pipeline` run the pass chain in
  process.
- `forge_torch.plan` builds a version 2 plan from lowered MLIR, with one node
  per tensor op and JSON attributes in each artifact.
- `CpuProvider` executes gemm, broadcast add, GELU, fused epilogue, softmax,
  layer norm, attention, reshape, and copy with naive fp32 kernels.
- `_sparrow_rt.execute_plan` runs a plan from Python.
- `compile()` prefers the native fused MLP, then a CPU plan, then eager. The
  CPU path records `execution="cpu"` and `lowered=True`.

### M2: torch.compile

- `aten.linear` lowers to a transposed gemm plus a bias add.
- gemm verification and shape inference apply transpose flags.
- `compile_graph` accepts a graph already captured by Dynamo.
- `forge_torch.dynamo:backend` is a `torch_dynamo_backends` entry point named
  `sparrow`.
- The backend wraps a single tensor output in the tuple Dynamo expects.

### M3: op coverage and canonical mapping

- `forge_torch.ops` is the single `OpSpec` table; `op_mapping`, `region`,
  `partitioner`, and `backend` derive their views from it.
- `forge.tensor.relu`, `forge.tensor.mul`, `forge.tensor.transpose`, and
  `forge.tensor.select` join the dialect, CPU registry, and plan builder.
- `aten.addmm` and `aten.bmm` lower to gemm; `aten.permute`, `aten.transpose`,
  and call-function reshape lower through the movement ops.
- The importer lifts `get_attr` parameters into plan inputs, resolves shape
  arithmetic from the example inputs, and lowers `nn.Linear` /
  `nn.LayerNorm` `call_module` nodes.
- `models/transformer_block.py` uses ordinary `nn.Linear` / `nn.LayerNorm`
  parameters and runs through a CPU plan.

### M4: devices, selection, and cost

- Plan version 3 carries a device table (`id`, `name`, `target`,
  `memory_space`); version 2 plans still parse and use provider-index device
  ids.
- The scheduler resolves v3 nodes through the device table, rejects unknown
  devices, mismatched artifact targets, and missing providers.
- `CPU_CAPABILITIES` and `COMPILER_LOWERINGS["cpu"]` cover every plan entry
  point. `build_plan(..., devices=..., fuse_triton=True)` collapses a
  `gemm → broadcast_add → gelu` group into one Triton node, and the mixed plan
  runs through the scheduler.
- `_phase7_cycle_tables` derives Astra and MicroGPU cycle counts from the
  simulator tables and shares the Triton/CPU kernel cost model. Synthetic
  transfer multipliers are gone.
- `compile_partitioned(config="A")` runs the CPU plan; B and C raise
  `NotImplementedError` naming the missing lowering until those backends pass.

### M5: Astra v2

- `forge_torch.astra` walks lowered tensor IR, lays buffers out in DRAM and
  SRAM, tiles K with accumulation, chunks DMA below the 15-bit immediate, and
  emits a JSON artifact header binding every input to a DRAM offset.
- The microcode ops carry address and size properties; `astra.microcode.
  vector_add` adds elementwise and bias-broadcast adds. `MATMUL` accumulates
  K tiles through the accumulator tile.
- `AstraProvider` executes the artifact's words directly against simulated
  DRAM and SRAM. The Python child process and base64 protocol are gone.
- `emit_astra` parses IR and encodes dialect ops; the regex scraper is gone.
- Acceptance: compiler-generated 16x16 tiles match eager for K=4, K=20, and
  K=512 with bias and GELU.

### M6: MicroGPU v2

- The instruction word is 32 bits (`opcode:4 | dst:6 | src:6 | imm:16`), so
  addresses cover 65,536 words instead of 16 and the register field covers the
  full 32-slot file.
- `MATMUL` computes a 4x4x4 tile from a base word address; `FADD`, `FMUL`, and
  `FRELU` add float elementwise operations alongside the integer core.
- `TensorToMicroGPU` lowers one small elementwise op or one 4x4 GEMM per module
  and keeps reductions unsupported until the ISA has loops.
- Acceptance: compiler-generated elementwise and 4x4 GEMM programs match
  eager.

### M7: release hygiene

- `compile(..., report_path=...)` writes the report only when asked; there is
  no implicit global report file.
- CMake exposes `SPARROW_SANITIZE_ADDRESS`, `SPARROW_SANITIZE_UNDEFINED`, and
  `SPARROW_SANITIZE_THREAD`; `scripts/ci-sanitizers.sh` runs ASan, UBSan, and a
  TSan job covering the event and provider paths.
- `uv.lock` is committed, `devenv.yaml` pins nixpkgs, and `shell.nix` carries
  the tarball hash.
- `sparrow_torch` and `sparrow_rt` re-export the current names.
- The version is 0.3.0 and SPEC/AUDIT describe the shipped surface.

## Open

None. The milestones above are the completion plan.

## Gates

1. Every op added to the CPU registry needs a verifier test, a C++ kernel test,
   and a Python plan test.
2. A backend does not become selectable until its lowering map and provider
   acceptance tests pass.
3. A graph outside the supported set must not be described as compiled.
   `compile()` keeps the eager fallback and records the reason in
   `planning_error`.
