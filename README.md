# ForgeStack

MLIR-based heterogeneous ML compiler and C++ runtime for PyTorch. Phase 1 ships a complete vertical slice (PyTorch FX → xDSL MLIR → Triton-style kernel → C++ execution → matches PyTorch eager).

See `.omo/plans/forgestack.md` for the full Phase 1 plan.

## Shipped (Phase 1)

- **Frontend** — `torch.compile(backend="forgestack")` hook + FX → MLIR importer
- **Compiler** — `forge.graph` and `forge.tensor` xDSL dialects + `forge_opt.py` driver + GraphToTensor conversion pass + lit/FileCheck golden tests
- **Backend** — Triton path-A kernel template + C++ CPU `fused_gemm_bias_gelu` kernel
- **Runtime** — ForgeRT C++ ABI (`TensorDescriptor`, `ArtifactDescriptor`, `Buffer`, `Event`, `DeviceProvider` abstract interface), Triton provider, `ExecutionPlan` + JSON, topological scheduler
- **Cache + Report** — hierarchical compilation cache key (per spec §13) + observability report (per spec §16)
- **Validation** — differential harness (`torch.allclose` + NaN/Inf/aliasing/mutation checks) + end-to-end FusedMLP pytest across 3 shapes
- **CI** — `scripts/ci.sh` master gate

## In progress (Phases 2-8 CPU extensions)

- **Phase 2** — xDSL canonicalize + shape inference + fusion + dynamic-shape guards
- **Phase 3** — ForgeRT memory planning upgrade (liveness + buffer reuse + ExecutionPlan allocations)
- **Phase 4** — `astra` dialect + microcode ISA + functional & cycle-accurate simulators + provider + e2e
- **Phase 5 (Triton path B + quantized + autotuning)** — separate plan
- **Phase 6 (HLS template + C-simulation + FPGA synthesis)** — separate plan
- **Phase 7 (heterogeneous partitioning)** — separate plan
- **Phase 8** — microgpu dialect + SIMT bytecode interpreter + provider + e2e

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
