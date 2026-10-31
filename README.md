<br>
<br>

<h1 align="center">sparrow</h1>

<br>
<br>

Sparrow is a PyTorch-to-MLIR compiler and C++ runtime for CPU backends and a
reference CPU executor.
It is a research-scale stack for compiler and runtime engineers who need to
read every layer.

Version 0.2.0 is a baseline. The frontend imports and caches graphs (including
ordinary `nn.Linear` parameters, lifted into plan inputs), lowers supported
tensor graphs to an execution plan, and runs that plan through ForgeRT when the
pybind11 module is built. The fused MLP keeps a native fast path. Graphs outside
the CPU kernel registry fall back to eager and record why in `planning_error`.
[docs/SPEC.md](docs/SPEC.md) states each contract and marks each one met,
partial, or gap; [docs/AUDIT.md](docs/AUDIT.md) records the evidence and the
remediation order.

## Quick start

```bash
devenv shell
python3 -c "import torch, xdsl; print(torch.__version__, xdsl.__version__)"
```

On a host without `devenv`, use the venv path:

```bash
bash scripts/setup-env.sh
source .venv/bin/activate
python3 -c "import torch, xdsl; print(torch.__version__, xdsl.__version__)"
```

Run the frontend on the fused MLP fixture:

```bash
python3 - <<'PY'
import torch
from forge_torch.backend import compile


class FusedMLP(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.nn.functional.gelu(x @ weight + bias)


model = FusedMLP()
x = torch.randn(2, 8, 128)
w = torch.randn(128, 256)
b = torch.randn(256)

compiled = compile(model, [x, w, b])
out = compiled(x, w, b)
print("shape", tuple(out.shape), "lowered", compiled.lowered)
PY
```

```text
shape (2, 8, 256) lowered False
```

`lowered` is `False` in a plain dev shell. The callable then executes the eager
model, and the import, cache, and report steps still run. Build the optional
native module (`cmake -B build -S . -G Ninja && cmake --build build`, then set
`PYTHONPATH=build/python`) to run this fused MLP through the C++ kernel and get
`lowered True`. With the module built, a supported non-fused graph also lowers
to a CPU plan and runs through the scheduler; `planning_error` is empty for that
path. The kernel is a naive triple loop, so large shapes fall back to
eager.

## Backends

| Backend | Implemented | Limit |
| --- | --- | --- |
| CPU reference | fp32 kernels for gemm, add, mul, relu, GELU, softmax, layer norm, attention, transpose, select, reshape | Static shapes; naive loops |
| Triton-CPU | fp32 fused GEMM, bias, and GELU C++ kernel, provider | 2D input, one output |
| Astra-sim | 64-bit microcode ISA, assembler, functional and timing simulators, provider | 16x16 tile; codegen is not usable |
| MicroGPU-sim | 16-bit SIMT ISA, assembler, interpreter, provider | No lowering pass; one input |

The partitioner registry is narrowed to ops with a dialect definition. The
compiler lowering map covers the CPU plan and the fused Triton group; Astra and
MicroGPU lowerings are still empty, so `partition()` fails closed for them and
`compile_partitioned` raises for configs B and C. Execution plans are version 3
with a device table, and a plan can mix CPU nodes with a fused Triton node.
The fused MLP keeps the native Triton-CPU kernel.

## Architecture

Sparrow separates the compile path from the runtime path.

![Compilation path](docs/figures/compilation.png)

The frontend captures an FX graph, imports it as MLIR, and partitions FX
regions with a capability and cost model. The compiler lowers and optimizes the
MLIR. Backend artifacts and the cache sit at the end of that path.

![Runtime path](docs/figures/runtime.png)

ForgeRT takes an execution plan, computes liveness and buffer reuse, orders the
nodes, and dispatches each launch to a provider. The providers are Host-only
and synchronous.

The frontend builds a multi-node execution plan for supported tensor graphs and
runs it through the scheduler. The fused MLP takes a one-node native path. Astra
and MicroGPU still have no compiler lowering, so the two figures describe one
design whose halves meet only for the supported CPU path. The specification
marks the rest as gaps.

## torch.compile

Install the project so PyTorch discovers the backend entry point:

```bash
uv pip install -e .
```

Then select it by name:

```python
compiled = torch.compile(model, backend="sparrow")
```

Dynamo lifts model parameters into graph placeholders and hands the graph to
`compile_graph`. The backend returns the tuple shape Dynamo expects. The same
CPU plan and native paths handle execution, so an unsupported op still falls
back to eager and records the reason.

## Checks

```bash
bash scripts/probe-env.sh
bash scripts/ci.sh
```

The probe names every missing tool. The gate chains the C++ build and ctest,
lit, pytest, ruff, and ty. It expects the environment from
`scripts/setup-env.sh` and is not hermetic yet. Read
[docs/AUDIT.md](docs/AUDIT.md) before relying on it.
