# backend/CONTRIBUTING.md

How to add a new execution backend to Sparrow (Phase 7 onboarding).
Closes the F1 reviewer finding that flagged its absence in Phases 2-8.
**Onboarding scoped**: covers plugging a new target into the existing
compiler/runtime/test surface. Does NOT cover the Phase 7 partitioner.

## Architecture overview

Sparrow is a PyTorch compiler split into five cooperating layers:

1. **xDSL compiler**: `compiler/lib/Dialect/` defines `forge.graph`
   (PyTorch-importable) and `forge.tensor` (shape+layout explicit),
   plus the conversion pass that walks one into the other.
2. **Backend dialects**: each backend owns an xDSL dialect (`astra`,
   `microgpu`, ...); a conversion pass in
   `compiler/lib/Conversion/YourTensorToYourBackend/` lowers
   `forge.tensor.*` into the backend's dialect.
3. **`DeviceProvider` interface** (runtime ABI, spec section 7): every
   backend implements `allocate` / `copy` / `launch` / `wait` plus
   `memory_space()` and `target()`. C++ impls subclass the abstract base
   in `runtime/include/forge_rt/device_provider.hpp`; Python simulators
   wrap a subprocess bridge.
4. **ForgeRT scheduler + MemoryPlanner**: walks the dep DAG the chosen
   dialect's assembler emits, allocates buffers via
   `BufferReuseAllocator`, and dispatches each launch through the owning
   provider.
5. **Cache + report**: `compile()` writes the `artifact.json` and the
   `CompilationReport` on a cache miss, keyed off the `CacheKey` (spec
   sections 4.6 and 4.7; see below).

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
`if (auto err = provider.allocate(...); err) return err;`: the
scheduler dispatches hundreds of ops per `compile()` and exception
unwinding across that loop has measurable cost. For pure-Python
simulators the provider is a thin C++ shim that spawns the interpreter
as a subprocess, feeds it a JSON-serialised microcode byte stream, and
reads the result back. Mirror `runtime/lib/providers/astra_provider.cpp`
(C++); its Python child script is embedded in the same file, so there is
no separate `astra_subprocess.py` to copy.

## Lowering forge.tensor → your dialect

The pass lives at `compiler/lib/Conversion/YourTensorToYourBackend/__init__.py`
and exposes a `ModulePass` named `tensor-to-your-backend`. It walks the
module with a `PatternRewriteWalker` and one `RewritePattern` per
`forge.tensor.*` op you support. Each pattern inspects operand shapes /
dtypes via the xDSL verifier, constructs the backend-specific ops in
the same block (operand reuse, not value remap), and returns the new
op's result so the walker rewires SSA uses. `TensorToAstra` is disabled
in 0.2.0; use the dialect definitions and `GraphToTensor` as style
references, not as a working lowering. Add a lit/FileCheck golden in
`compiler/test/Conversion/YourTensorToYourBackend/` covering both the
round-trip case and a negative case (unsupported op → `VerifyException`).
Register the pass in `compiler/tools/forge-opt/forge_opt.py` and add an
action or pipeline name. Do not rely on a wildcard discovery hook; there
is none for conversions.

## Cache key contributions

The `CacheKey` (spec section 4.6) hashes every field that affects codegen
or correctness. The key carries `graph_hash`, `target` (chosen backend
name; MUST invalidate the cache), `arch` (host or ISA revision; bump
when you change the ISA), `dtype` (one ordered token per input dtype),
`shape_constraints`, `schedule`, and `symbolic_constraints`. The
compiler version is folded into the digest. Add a new field only when
codegen genuinely depends on it; do not turn the cache key into a
fingerprint of every codegen knob. When in doubt, prefer a new `arch`
value over a new key field.