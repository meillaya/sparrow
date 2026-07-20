# ForgeStack

MLIR-based heterogeneous ML compiler and C++ runtime for PyTorch — PyTorch FX → xDSL MLIR → backend providers → C++ execution.

![CI](https://img.shields.io/badge/ci-bash%20scripts%2Fci.sh-green)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

Three backends live today: **Triton-CPU** (Phase 1), **Astra-sim** (Phase 4 microcode dialect + cycle-accurate simulator), and **MicroGPU-sim** (Phase 8 SIMT interpreter). Phase 7 adds a heterogeneous partitioner that dispatches regions across them per spec §19.

## Status

| Phase | What                                                        | State |
| ----- | ----------------------------------------------------------- | :---: |
| 1     | Vertical slice — PyTorch FX → xDSL → Triton-CPU, matches eager |   ✓   |
| 2     | xDSL optimize, dynamic-shape guards, epilogue fusion         |   ✓   |
| 3     | ForgeRT memory planner (liveness + buffer reuse)             |   ✓   |
| 4     | `astra` dialect + microcode ISA + simulators + provider     |   ✓   |
| 7     | Heterogeneous partitioner + spec §19 transformer demo       |   ✓   |
| 8     | `microgpu` dialect + SIMT bytecode interpreter + provider   |   ✓   |
| 5/6   | Triton perf (path B / quantized GEMM) + HLS → FPGA          |  deferred |

`bash scripts/ci.sh` is the gate — lit + pytest (~298 tests) + ctest + ruff + ty.

## Quick start

```bash
# One-time bootstrap (uv venv + torch + xdsl + ruff + ty)
bash scripts/setup-env.sh

# Activate + verify the environment
source .venv/bin/activate
bash scripts/probe-env.sh

# Hello FusedMLP
python3 -c "
import torch
from forge_torch.backend import compile
from forge_torch.conftest import FusedMLP
m = FusedMLP()
x, w, b = torch.randn(2, 512, 4096), torch.randn(4096, 11008), torch.randn(11008)
c = compile(m, [x, w, b])
out = c(x, w, b)
assert torch.allclose(out, m(x, w, b), atol=1e-4)
print('hello forgestack:', tuple(out.shape))
"

# Run the full gate (lit + pytest + ctest + ruff + ty)
bash scripts/ci.sh
```

## Repository layout

```
forgestack/
├── compiler/                # xDSL dialects, conversion passes, forge-opt driver
├── frontend/                # torch.compile hook + FX → MLIR importer
├── runtime/                 # ForgeRT C++ ABI + scheduler + memory planner
├── backends/
│   ├── triton/cpu/          # Phase 1 — fused_gemm_bias_gelu CPU kernel
│   ├── astra/               # Phase 4 — microcode dialect + cycle-accurate sim
│   └── microgpu/            # Phase 8 — SIMT bytecode interpreter
├── models/                  # transformer_block.py (spec §19)
├── test/                    # pytest (frontend + backends + e2e)
├── scripts/                 # setup-env.sh, probe-env.sh, ci.sh
├── README.md
├── LICENSE                  # Apache-2.0
└── pyproject.toml
```

## References

- Spec sections (in `.omo/plans/forgestack-cpu.md`): [§3 FX importer](.omo/plans/forgestack-cpu.md) · [§6 partitioner capabilities](.omo/plans/forgestack-cpu.md) · [§19 transformer demo](.omo/plans/forgestack-cpu.md)
- Architecture plans: [`.omo/plans/forgestack.md`](.omo/plans/forgestack.md) (Phase 1) · [`.omo/plans/forgestack-cpu.md`](.omo/plans/forgestack-cpu.md) (Phases 2–8 + 7)
- Adding a backend: [`backend/CONTRIBUTING.md`](backend/CONTRIBUTING.md)
- Upstream dependencies: [xDSL](https://xdsl.dev/) · [torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)

## License

Apache-2.0.
