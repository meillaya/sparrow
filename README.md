# ForgeStack

MLIR-based heterogeneous ML compiler and C++ runtime for PyTorch. Phase 1 ships a complete vertical slice (PyTorch FX → xDSL MLIR → Triton-style kernel → C++ execution → matches PyTorch eager).

See `.omo/plans/forgestack.md` for the full Phase 1 plan.

## Shipped (Phases 1-8 + Phase 7)

### Phase 1 — end-to-end vertical slice (PyTorch FX → MLIR → C++ CPU)

- **Frontend** — `torch.compile(backend="forgestack")` hook + FX → MLIR importer — `733ec70`
- **Compiler** — `forge.graph` + `forge.tensor` xDSL dialects (`b4124da`) + GraphToTensor conversion pass + lit/FileCheck goldens (`c71dfed`)
- **Backend** — Triton path-A kernel template + codegen + JIT (`5eb2918`)
- **Runtime** — ForgeRT C++ ABI (TensorDescriptor/ArtifactDescriptor/Buffer/Event/DeviceProvider) (`ce97653`) + Triton CPU provider + `fused_gemm_bias_gelu` (`ed95d04`) + ExecutionPlan + JSON + topological scheduler (`6439a77`)
- **Cache + Report** — hierarchical compilation cache (per spec §13) + observability report (per spec §16) — `483b398`
- **Validation** — differential harness (`torch.allclose` + NaN/Inf/aliasing/mutation) + end-to-end FusedMLP pytest — `7418aa0`
- **CI** — `scripts/ci.sh` master gate — `56adaa8`

### Phase 2 — xDSL optimize + dynamic-shape guards

- **Passes** — canonicalization (DCE/CSE/const-fold/attr-canon) (`2b0f990`), shape inference (static dim + symbolic pass-through) (`d98951a`), fusion (gemm+bias+activation → `fused_epilogue`, FusedMLP triple) (`74d53c1`), chained `--pass-pipeline` + `--action=optimize` (`4983e3a`)
- **Dynamic shapes** — `SymbolicConstraint` + `symbolic_guard` on `forge.graph` (`672a822`), `dynamic_compile` API with constraint-bucketed cache (`74d3b2c`), symbolic constraints in cache key (`b8da1fe`), schedule dialect stub (TileShape/LoopOrder/PipelineStages) (`17225da`)
- **Differential** — canonicalize / shape-inference / fusion preserve numerics (`e507e92`, `19cf4cf`, `fa08067`), chained-optimize op-count (`1e9af03`), cache-key + artifact-byte determinism (`df0dda6`), compilation-report completeness (`38cd042`), 6-shape + fp16/bf16 differential (`91f22cf`)

### Phase 3 — ForgeRT memory planning

- **Analysis + allocators** — `LivenessAnalyzer` (backward dataflow, per-tensor LiveRange) (`5ec010e`), `BufferReuseAllocator` (first-fit decreasing by lifetime length) (`e7d3022`), MemoryPlanner wiring (`d558c5e`)
- **Runtime integration** — ExecutionPlan.allocation JSON with byte_offset/size/alignment/space (`6585093`), Scheduler consumes MemoryPlanner allocations (`015b4d9`), original/planned intermediate memory MB in compilation report (`709a21f`)
- **e2e** — memory-planning e2e (`beb8f96`), Scheduler+MemoryPlanner e2e (`3f7a3ca`), memory-report correctness (`d990191`)

### Phase 4 — `astra` dialect + microcode ISA + simulators + provider

- **Dialect** — `astra` (compute.matmul + microcode.{dma_load,dma_store,matmul,vector_gelu}) + lit golden (`9e0dbb6`), `TensorToAstra` conversion pass + `lower-astra` CLI action (`04f3aab`)
- **ISA + codegen** — microcode ISA + encode/decode round-trip (`318bc69`), text assembler/disassembler (`1563748`), dialect ↔ microcode byte round-trip (`2fa1113`), `astra_codegen` microcode bytes from region (`774728e`)
- **Simulators** — functional (correctness only) (`05b40e3`), cycle-accurate timing (per-instruction cycle table, per spec §9) (`4a2a4af`), peak scratchpad utilization (`3a61988`), DMA-stalled vs compute-stalled cycle decomposition (`757af64`), total bytes transferred (`a523e76`)
- **Provider + e2e** — entry-point cross-check (`1992a25`), random-microcode functional/timing equivalence (`40062c9`), `AstraProvider` (wraps timing simulator) (`56c8af9`), FusedMLP-on-astra-sim e2e (`3b3aad7`), cross-provider FusedMLP equivalence (`5aafd69`)

### Phase 8 — microgpu dialect + SIMT bytecode interpreter + provider

- **Dialect** — `microgpu` (load/store/add/mul/sigmoid/return) + lit golden (`2435726`), dialect ↔ bytecode byte round-trip (`7540ec5`)
- **Bytecode** — SIMT bytecode ISA + encode/decode round-trip (`ea13554`), text assembler/disassembler (`f3e1e11`)
- **Interpreter** — Python reference impl (`5c3de82`), SIMT race detection (WAR/WAW, RaceWarning to stderr) (`bf3f5d5`), LOAD/STORE out-of-bounds detection (`5a5eb4f`), dep DAG emission (`a83e78e`)
- **Provider + e2e** — `MicroGpuProvider` (wraps Python interpreter via subprocess) (`990ba49`), FusedMLP-on-microgpu e2e (`1fe7df3`), race + OOB pytest (`7605752`)

### Cross-cutting

- **CI** — `scripts/ci.sh` extended for Phases 2-8 pytest + lit (`be24810`); pytest test/ collects 211 tests across Phases 1-8
- **Lint + format** — ruff lint + format enforced on every commit (`589a29b`, `15f984b`, `74e36d8`)

## Next

- **Phase 5** — Triton path B (elementwise codegen) + quantized GEMM + autotuning — separate plan
- **Phase 6** — HLS template + C-simulation + FPGA synthesis — separate plan
- **Phase 7** — heterogeneous partitioning — separate plan

## Quick start

```bash
# One-time bootstrap (uv venv + torch + xdsl + ruff + ty; ~5 min on first run)
bash scripts/setup-env.sh

# Verify the environment (should report 18 PASS)
source .venv/bin/activate
bash scripts/probe-env.sh

# Hello FusedMLP — five lines
python3 -c "
import torch
from forge_torch.backend import compile
from forge_torch.conftest import FusedMLP
m = FusedMLP()
x, w, b = torch.randn(2, 512, 4096), torch.randn(4096, 11008), torch.randn(11008)
c = compile(m, [x, w, b])
out = c(x, w, b)
assert torch.allclose(out, m(x, w, b), atol=1e-4)
print('hello forgestack: output shape', tuple(out.shape))
"

# Run the full CI (lit + pytest + ctest + ruff + ty)
bash scripts/ci.sh
```

## Repository layout

```
forgestack/
├── compiler/                # xDSL dialects + conversion pass + forge-opt driver
│   ├── lib/Dialect/         # forge.graph, forge.tensor (Python, xDSL)
│   ├── lib/Conversion/      # GraphToTensor pass
│   ├── tools/forge-opt/     # CLI driver (parse / print / lower)
│   └── test/                # lit FileCheck golden tests
├── frontend/python/forge_torch/  # torch.compile(backend="forgestack") + FX importer
├── backends/triton/         # Path A kernel + codegen + JIT
│   └── cpu/                 # hand-tuned C++ CPU fused_gemm_bias_gelu
├── runtime/                 # ForgeRT C++: ABI + providers + scheduler
│   ├── include/forge_rt/
│   ├── lib/
│   └── tests/               # GoogleTest unit tests
├── models/                  # (placeholder for Phase 2+ transformer models)
├── test/                    # pytest (frontend + backends + end_to_end)
├── benchmarks/              # (placeholder for Phase 2+ benchmarks)
├── scripts/                 # setup-env.sh, probe-env.sh, ci.sh
├── devenv.nix               # canonical Nix dev env (Task 1)
├── shell.nix                # fallback for hosts without devenv shell access
├── pyproject.toml           # uv + ruff + ty + pytest config
├── CMakeLists.txt           # C++ build (C++20, nlohmann_json, GTest)
└── README.md
```

## License

Apache-2.0.
