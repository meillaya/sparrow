# ForgeStack

MLIR-based heterogeneous ML compiler and runtime for PyTorch. See `.omo/plans/forgestack.md` for the Phase 1 vertical-slice plan.

## Quick start
```bash
bash scripts/setup-env.sh   # one-time bootstrap (uv venv + torch + xdsl + ruff + ty)
source .venv/bin/activate
bash scripts/probe-env.sh   # verify the environment (should report 18 PASS)
```