# backend/CONTRIBUTING.md

How to add a new execution backend to ForgeStack (Phase 7 onboarding).
Closes the F1 reviewer finding that flagged its absence in Phases 2-8.
**Onboarding scoped**: covers plugging a new target into the existing
compiler/runtime/test surface. Does NOT cover the Phase 7 partitioner.

## Architecture overview

ForgeStack is a PyTorch compiler split into five cooperating layers:

1. **xDSL compiler** — `compiler/lib/Dialect/` defines `forge.graph`
   (PyTorch-importable) and `forge.tensor` (shape+layout explicit),
   plus the conversion pass that walks one into the other.
2. **Backend dialects** — each backend owns an xDSL dialect (`astra`,
   `microgpu`, ...); a conversion pass in
   `compiler/lib/Conversion/YourTensorToYourBackend/` lowers
   `forge.tensor.*` into the backend's dialect.
3. **`DeviceProvider` interface** (runtime ABI, spec §10) — every
   backend implements `allocate` / `copy` / `launch` / `wait` plus
   `memory_space()` and `target()`. C++ impls subclass the abstract base
   in `runtime/include/forge_rt/device_provider.hpp`; Python simulators
   wrap a subprocess bridge.
4. **ForgeRT scheduler + MemoryPlanner** — walks the dep DAG the chosen
   dialect's assembler emits, allocates buffers via
   `BufferReuseAllocator`, and dispatches each launch through the owning
   provider.
5. **Cache + report** — every `compile()` writes the `CompilationReport`
   (spec §16) and the cached `artifact.json` (spec §13), keyed off the
   `CacheKey` (see below).

The three shipping CPU-only backends are Triton-CPU, Astra-sim, and
MicroGPU-sim, each with a `*Provider` runtime shim.

## Adding a new target

Adding a backend is five steps, in order. Skipping the dialect step
turns the backend into a closed box the rest of the system cannot reason
about.

1. **Register a new xDSL dialect + ops** under
   `compiler/lib/Dialect/YourBackend/`. Define every op the codegen
   pass will emit (memory, compute, control-flow) and export a
   `YourBackend = Dialect(...)` instance.
2. **Add a `BackendCapabilities` entry** in
   `frontend/python/forge_torch/capabilities.py`: supported ops, dtypes,
   local-memory byte budget, dynamic-shapes / async / fused-epilogue
   flags.
3. **Write a `DeviceProvider` impl** under
   `runtime/lib/providers/your_backend_provider.{hpp,cpp}` (or `.py`).
4. **Write a `CostEstimator` hook** in
   `frontend/python/forge_torch/_phase7_cycle_tables.py`: per-op
   per-backend cycle counts so the partitioner ranks without measuring.
5. **Add the backend's contribution to the cache key** (below).

## Writing a DeviceProvider

A `DeviceProvider` owns allocation, copy, and launch for ONE
(memory_space, target) pair. The scheduler holds a registry of providers
keyed by `MemorySpace` and dispatches via this interface. Every method
**returns `Result<T>`** rather than throwing; C++ callers check the
`Error` payload and Python callers raise only at the C++ FFI boundary,
never inside the provider. Internal control flow uses
`if (auto err = provider.allocate(...); err) return err;` — the
scheduler dispatches hundreds of ops per `compile()` and exception
unwinding across that loop has measurable cost. For pure-Python
simulators the provider is a thin C++ shim that spawns the interpreter
as a subprocess, feeds it a JSON-serialised microcode byte stream, and
reads the result back. Mirror `runtime/lib/providers/astra_provider.cpp`
(C++) and `astra_subprocess.py` (Python); re-using the existing bridge
is cheaper than rolling your own IPC.

## Lowering forge.tensor → your dialect

The pass lives at `compiler/lib/Conversion/YourTensorToYourBackend/__init__.py`
and exposes a `ModulePass` named `tensor-to-your-backend`. It walks the
module with a `PatternRewriteWalker` and one `RewritePattern` per
`forge.tensor.*` op you support. Each pattern inspects operand shapes /
dtypes via the xDSL verifier, constructs the backend-specific ops in
the same block (operand reuse, not value remap), and returns the new
op's result so the walker rewires SSA uses. Mirror
`compiler/lib/Conversion/TensorToAstra/__init__.py` for the shape of the
pass. Add a lit/FileCheck golden in
`compiler/test/Conversion/YourTensorToYourBackend/` covering both the
round-trip case and a negative case (unsupported op → `VerifyException`).
`forge-opt --action=lower-...` loads the pass via `discover_passes()` —
do NOT register it manually.

## Cache key contributions

The `CacheKey` (spec §13) hashes every field that affects codegen or
correctness. Six fields make up the key today: `target` (chosen backend
name; MUST invalidate the cache), `arch` (micro-architecture revision,
e.g. `"astra-sim-v1"`; bump when you change the ISA), `dtype` (element
type kernels were compiled for), `shape_constraints` (symbolic bucket
for dynamic shapes, Phase 2+; only relevant when the backend supports
them), `schedule` (tile/warp/stage knobs; re-tuning without bumping
`arch` is fine), `compiler_version` (bumped on every release). Add a
new field only when codegen genuinely depends on it; do not turn the
cache key into a fingerprint of every codegen knob. When in doubt,
prefer a new `arch` value over a new key field.