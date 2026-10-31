# Sparrow audit, 2026-10-31

This document records a read-only audit of the repository at commit `55f64d2`,
before the documentation rewrite. It covers the Python frontend, the xDSL
compiler, the three CPU backends, the C++ runtime, the build files, and the
tests. The audit looked for correctness defects, claims that the code does not
support, portability failures, and gaps in the verification gates.

## Method

The audit combined five checks:

1. Read every source file under `frontend/`, `compiler/`, `backends/`,
   `runtime/`, `models/`, `scripts/`, and `test/`. The tree holds 191 tracked
   files.
2. Counted tests and inventory markers with `grep` and file walks.
3. Ran the tools that the host provides: `ruff check .`, `ruff format --check .`,
   and `python3 -m compileall` over the Python trees.
4. Ran `bash scripts/probe-env.sh` to record the host toolchain state.
5. Rendered the two new architecture scenes with the Excalidraw CLI and
   inspected the PNG output.

The host lacks `cmake`, `ninja`, a `.venv`, and the Python runtime dependencies
(`torch`, `xdsl`). No CMake build, pytest run, or ctest run was possible. A
separate compiler probe installed xDSL 0.74 in a temporary environment and ran
`forge-opt` over the lit fixtures. Findings from that probe are marked as
verified. The project pins `xdsl>=0.68`, so a version difference remains
possible. Runtime findings come from reading the code, not from executing it.
Each finding cites the line that supports it. The audit did not modify source
files.

## Summary

| Severity | Count | Meaning |
| --- | ---: | --- |
| Critical | 10 | Silent wrong results, memory corruption, or a cache that serves the wrong artifact |
| High | 18 | An advertised path cannot work, or fails on any host but the author's |
| Medium | 23 | Correctness risk under specific inputs, misleading claims, or a gate that cannot fail |
| Low | 7 | Maintainability, dead surface, and naming drift |

The dialect layer and the runtime ABI are solid. The weak points are the
conversion and codegen passes, the seam between Python and the runtime, and
the scheduler: the frontend never builds a runtime plan, and the C++ scheduler
cannot execute a multi-node plan with real buffers. Several documents describe
the intended system as if those parts were finished.

## What holds up

- The xDSL dialects parse, print, and verify for the static-shape fixtures in
  `compiler/test/`. `forge.graph`, `forge.tensor`,
  `forge.schedule`, `astra`, and `microgpu` all have lit coverage.
- The FX importer emits verified MLIR for the supported op set. It rejects
  `get_attr` instead of emitting invalid IR.
- The MicroGPU assembler round-trips exactly. The Astra assembler is
  deterministic but lossy (AUD-58). The MicroGPU interpreter has race and
  out-of-bounds diagnostics.
- The C++ ABI is small and readable. `Buffer` is move-only, the provider
  interface is explicit, and most error paths return `Result<T>`.
- `ruff check` and `ruff format --check` pass on all 102 Python files.
  `compileall` succeeds under Python 3.14.
- The repository has 183 test functions in 47 files, 26 lit tests, and 22 C++
  `TEST` macros across nine GoogleTest binaries. The tests are broad even where
  individual assertions are weak.

## Critical findings

### AUD-01 Cache path ignores the graph and target

`CacheKey.to_path` builds `<root>/fused_mlp/<arch>/<dtype>/<shape>/<schedule>`
(`frontend/python/forge_torch/cache.py:58-65`). The graph hash, target, and
compiler version appear in `hexdigest()` but not in the path. `CacheStore.get`
returns any `artifact.json` at that path (`cache_store.py:50-54`), and
`compile()` treats that as a hit (`backend.py:161-166`). Two different models,
or the same model compiled for Triton and Astra, read the first model's MLIR.
The report is also skipped on the false hit because report emission is tied to
the miss path.

Fix: include `hexdigest()` in the path, or store a key manifest inside
`artifact.json` and verify it on read.

### AUD-02 Scheduler indexes tables by unchecked node ids

The scheduler sizes its dense tables from the number of tensors and nodes
(`runtime/lib/scheduler.cpp:47-55`), then indexes them with `NodeId` values:
`in_degree[nid]++` (`:182`), `in_degree[nid]` (`:191`), and
`successors[nid]` (`:202`). Only dependency ids are range-checked (`:175-180`).
A plan with two nodes and id 100 writes past the end of both vectors. The
test fixtures use contiguous ids from zero, so the bug stays hidden.

Fix: size tables from the maximum id plus one, or reject ids outside
`[0, plan.nodes.size())` at ingestion.

### AUD-03 Scheduler never allocates intermediate buffers

`Scheduler::execute` calls `MemoryPlanner::plan`, which writes offsets and
sizes (`scheduler.cpp:70`). It never calls `DeviceProvider::allocate`. The
buffer table starts as default `Buffer` objects with null data. For every node,
`apply_planned_spec` copies metadata into the buffer (`scheduler.cpp:86-109`),
then the provider receives a null pointer. Triton, Astra, and MicroGPU all
reject null buffers (`triton_provider.cpp:193-196`,
`astra_provider.cpp:432-435`, `microgpu_provider.cpp:303-304`). Only a plan
whose inputs and outputs cover every tensor can run, and only because the
caller supplied those buffers. The planner's offsets are dead metadata.

Fix: allocate a pool through the owning provider and carve planned offsets
from it, or require callers to pre-allocate and check sizes before launch.

### AUD-04 In-place nodes destroy the live buffer

When a node lists the same tensor as input and output, the scheduler moves the
buffer out for the input span (`scheduler.cpp:264`) and then moves the same
slot again for the output span (`:289`). The second move sees the empty slot.
Stage C restores the input first and then overwrites it with the empty output
buffer (`:323-328`). The tensor is freed. `test_scheduler.cpp:153` builds a
`{3} -> {3}` node, but `MockDeviceProvider` ignores buffers, so the test does
not observe the loss.

Fix: deduplicate in-place tensor ids before staging, or stage by reference
until the launch completes.

### AUD-05 Liveness treats node ids as time

`LivenessAnalyzer` computes intervals as the minimum and maximum `NodeId` that
touch a tensor (`runtime/lib/liveness.cpp:64-76`). The allocator decides
overlap from those numeric intervals (`buffer_reuse.cpp:140-146`). The
scheduler, however, orders nodes with Kahn's algorithm and never requires ids
to match topological position (`scheduler.cpp:167-216`). A plan with id 50
scheduled before id 2 can alias two tensors that are live at the same time.

Fix: compute liveness over the topological order, then map intervals back to
node ids, or require plan producers to emit ids in topological order and
validate that contract.

### AUD-06 Dynamic cache bucket ignores the model

`dynamic_compile` hashes only the constraint signature to choose a bucket
(`frontend/python/forge_torch/dynamic_compile.py:194-200`). Every static model
hashes the empty string, so all static models share one directory. A warm call
loads the previous model's MLIR (`:200-203`) and returns it. Target, arch, and
dtype are also absent from the bucket.

Fix: hash the graph hash, target, arch, dtype, and constraint tuple together.

## Critical compiler findings

### AUD-47 `TensorToAstra` corrupts the fused epilogue

When a `gemm` lowers first, the following `fused_epilogue` sees `i64` operands
and invents `M=N=16`, dtype 4, and `K=1`
(`compiler/lib/Conversion/TensorToAstra/__init__.py:324-339`). It loads 64
bias bytes (`:357`), runs `matmul` on `[sram_b,sram_b,sram_b]` (`:362-366`),
and applies GELU and a store to `sram_z`, which no op writes (`:369-381`).
Running `--action=lower-astra` on the fused MLP fixture prints `m=16 n=16 k=1`
and `bytes=64`, and replaces the function arguments with constants at
`0x1000..0x6000` (`:250-256,335-339`). Two GEMMs receive the same addresses.
Verified with xDSL 0.74.

### AUD-48 Astra codegen does not compile the IR

`backends/astra/codegen/astra_codegen.py` scrapes `astra.microcode.*` text with
regular expressions, including from comments (`:31,62-71`). It hardcodes
`SRAM[0]` and `DRAM[0]` (`:44-54`), drops `m` and `n`, emits `elements=0` for
every `VECTOR_GELU` (`:53-54`), and ignores `astra.compute.matmul`. The
five-op roundtrip fixture produces eight words, and the GELU word decodes with
`imm=0`. The codegen path cannot execute the semantics the dialect expresses.
Verified with xDSL 0.74.

### AUD-49 Real DMA sizes do not fit the microcode immediate

`TensorToAstra` emits byte counts such as 8,388,608, 180,355,072, and
22,544,384 (`TensorToAstra/__init__.py:258-280`). The microcode immediate is 15
bits (`compiler/lib/Dialect/Astra/microcode.py:56-65`). `emit_astra` raises
`imm must fit in 16 bits, got 8388608`. Only toy tile sizes can be encoded.
Verified.

### AUD-50 There is no working Graph to Astra path

The default `--action=optimize` chain omits `graph-to-tensor`
(`forge_opt.py:290`), and `--action=lower-astra` appends `tensor-to-astra`
without inserting it (`:188-218`). Running `lower-astra` on graph IR fails
after `_rewrite_func_signatures` converts every argument to `i64`
(`TensorToAstra/__init__.py:585-600`), which the tensor verifier rejects.
Running it on the output of `graph-to-tensor` also fails: `Fuse` only matches
`fused_epilogue` (`Transforms/Fuse/__init__.py:61-89`), and `broadcast_add` is
erased only when its user is a `fused_epilogue`
(`TensorToAstra/__init__.py:406-412`). Both paths were reproduced. The only
working lowering is a hand-written tensor-dialect fixture.

## Compiler findings

### AUD-51 (high) Named symbolic dimensions cannot be parsed

`_parse_dim_token` consumes the lexer-glued token `Bx4xf32` as one identifier
(`compiler/lib/Dialect/Graph/__init__.py:316-349`). Parsing
`!forge.tensor<Bx4xf32>` fails with `unknown forge.dtype 'Bx4xf32'`. Every
shape check then falls back to `_static_shape`, which maps symbols to -1
(`Tensor/__init__.py:125-135`), so `B` and `S` compare equal. Graph broadcast
rejects a symbolic dim against a static one
(`Graph/__init__.py:407-429`), contrary to the docstring. Verified.

### AUD-52 (high) Graph verifiers ignore result dtypes

`MatmulOp.verify_` checks the two operand dtypes and never the result
(`Graph/__init__.py:490-493`). `AddOp.verify_` does the same (`:566-569`). A
roundtrip test accepts an `f16` result from `f32` operands, and the mismatch
surfaces later in the tensor verifier. Verified.

### AUD-53 (high) `Fuse` crashes on a multi-use `broadcast_add`

The pass checks that the `gemm` result has one use (`Fuse/__init__.py:75-77`)
but not that the `broadcast_add` result has one use before erasing it (`:89`).
A graph that reads the add result twice raises
`ValueError: Attempting to delete SSA value that still has uses`, exit 1.
Verified.

### AUD-54 (high) `TensorToAstra` drops attributes and batch bytes

`GemmPattern` never reads `transpose_a` or `transpose_b`
(`TensorToAstra/__init__.py:230-288`). The epilogue pattern emits GELU only for
`#forge.activation<gelu>` (`:369`), so RELU and SILU are silently dropped.
Rank greater than 2 byte counts use only the last two dimensions (`:242-267`),
so batch bytes are omitted. Verified by inspection.

### AUD-55 (medium) Compiler tests check verifiers more than passes

The negative lit cases for `fuse`, `canonicalize`, and `shape-inference` feed
IR that the dialect verifier rejects before the pass runs. The TensorToAstra
golden never checks `M`, `N`, `K`, or byte counts, so the placeholder `16x16`
output passes (`compiler/test/Conversion/TensorToAstra/fused_mlp.mlir:31-43`).
The shape-inference test cites a Python smoke test that does not exist
(`compiler/test/Transforms/ShapeInference/static_fused_mlp.mlir:11-15`).

### AUD-56 (medium) `Canonicalize` documents a subset it does not control

The pass describes a conservative GELU-only dead-code subset
(`compiler/lib/Transforms/Canonicalize/__init__.py:13-20`). xDSL's
`GreedyRewritePatternApplier` removes every trivially dead op before the custom
patterns run, so a dead `fused_epilogue` disappears while `DeadGeluPattern`
matches only `gelu` (`:70-96`). The result is broader than the documented
contract.

### AUD-57 (medium) MicroGPU arithmetic and dependency edges have defects

`_sigmoid(INT32_MIN)` raises `OverflowError`
(`backends/microgpu/interp.py:107-117`). `SIGMOID` rounds to 0 or 1, so
`sigmoid(-10)` returns 0. `ADD` and `MUL` wrap as unsigned while `LOAD` reads
signed (`:220-267`), so `sigmoid(0xFFFFFFFF)` returns 1. The dependency DAG
omits the destination read for `ADD` and `MUL` (`:310-316`) and emits duplicate
and self edges (`:345-367`). A `LOAD; SIGMOID; RETURN` program produces
`((0,1),(0,1),(1,1))`.

### AUD-58 (low) Astra roundtrip is lossy and the dialect is incomplete

Assembling and disassembling Astra text resets addresses and element counts to
zero (`compiler/lib/Dialect/Astra/assembler.py:175-202`). The ISA defines
`HALT`, but the dialect has no `astra.halt` op (`Astra/__init__.py:252-253`).
The roundtrip fixture cannot preserve the fields the simulator needs.

## High findings

### AUD-07 The compiled callable executes eager PyTorch

`CompiledFusedMLP.__call__` returns `self.model(*args)` (`backend.py:251-259`).
`validate()` compares that eager call to an eager call (`:261-276`), so the
"matches eager" result is tautological. The e2e tests that assert numerical
agreement exercise the stub, not a lowered path. The README claim that the
vertical slice "matches eager" is true only in this trivial sense.

### AUD-08 The Python frontend never reaches the C++ runtime

No Python module imports `forge_rt`, builds an `ExecutionPlan`, or calls
`Scheduler`. The only references are comments and protocols
(`backend.py:252-255`, `report.py:41-44`). The pybind11 boundary is described
as a later phase. The C++ runtime is exercised by its own tests only.

### AUD-09 Capabilities advertise lowerings that do not exist

The capability registry lists `forge.tensor.reshape` for Triton
(`capabilities.py:82`), but no such op exists in the Tensor dialect, and
`GraphToTensorPass` deliberately leaves `forge.graph.view` untouched
(`GraphToTensor/__init__.py:30-33,168-172`). MicroGPU claims `gemm`,
`fused_epilogue`, and `scaled_dot_product_attention`
(`capabilities.py:135-142`) while there is no `TensorToMicroGPU` pass and the
ISA has no matmul. Astra claims `layer_norm` (`capabilities.py:103-119`) while
`TensorToAstra` has no pattern for it (`TensorToAstra/__init__.py:213-526`).
The partitioner can therefore select a backend for an op that no pass can
lower.

### AUD-10 Runtime artifacts are not plumbed through plans

`ExecutableNode` carries an `ArtifactId` (`execution_plan.hpp:70-78`), but
`ExecutionPlan` has no artifact table. The scheduler fabricates a descriptor
with only the target and an entry point (`scheduler.cpp:292-299`). The binary,
argument list, shape guards, workspace hint, and launch geometry are lost.
Astra and MicroGPU need `artifact.binary` to launch, so a scheduler-driven
heterogeneous run cannot pass real microcode.

### AUD-11 Providers do not check buffer capacity

`AstraProvider` base64-encodes `M*K*4` input bytes, checks the child payload
against `M*N*4`, and then writes that payload to `outputs[0]` without comparing
`outputs[0].spec().byte_size` (`astra_provider.cpp:453-458,485-493`).
`TritonProvider` indexes by shape only (`triton_provider.cpp:184-198`). The
scheduler replaces the caller's `byte_size` with the planned size before launch
(`scheduler.cpp:101-107`), so an undersized caller buffer passes the metadata
check and is written past its allocation.

### AUD-12 Differential tests hardcode the author's checkout

Four end-to-end tests invoke
`/home/mei/projects/forgestack/.venv/bin/python3` and the matching
`forge_opt.py` path (`test_canonicalize_differential.py:51-52`,
`test_chained_optimize.py:57-58`, `test_fusion_differential.py:54-55`,
`test_shape_inference_differential.py:51-52`). The paths do not exist on this
host or in CI. `test_phase1_smoke.py` uses `REPO` correctly, so the pattern is
inconsistent as well as broken.

### AUD-13 Metrics and reports are populated with constants

`_build_report` hardcodes `fusion_groups=1`, both alternative costs,
`original_intermediate_memory_mb=45.0`, `planned_intermediate_memory_mb=8.0`,
`generated_kernels=1`, and `validation_max_error=0.0`
(`backend.py:337-357`). `_count_mlir_ops` counts only `forge.graph.`
(`:336`), so tensor ops are omitted. `phase7_metrics` measures wall clock but
derives memory from the first input and applies invented transfer multipliers
(`phase7_metrics.py:192-237`). The numerical error is always zero because both
sides run eager (`:271-274`).

### AUD-14 Classic FX graphs are unsupported

The importer skips `call_module` nodes (`fx_importer.py:134,170`) and raises on
`get_attr` (`:172-173`). A traced module with `nn.Linear` therefore either
raises `KeyError` downstream or silently loses operations, depending on where
the module node is consumed. `TransformerBlock` works around this by storing
weights in closures (`models/transformer_block.py:38-58`). The test named
`test_transformer_block_compile_matches_eager` never calls `compile`
(`test/test_transformer_block.py:24-38`).

### AUD-15 The default cost model can crash and mis-ranks backends

`_DIALECT_TO_BACKEND_OP` maps `forge.tensor.reshape` to Triton `matmul`
(`cost_estimator.py:116-119`). The partitioner's default shape for reshape is
`[8, 8]` (`partitioner.py:59`), while the Triton matmul table indexes
`shape[2]` (`_phase7_cycle_tables.py:42`). A reshape region without tensor
metadata raises `IndexError`. Separately, MicroGPU scores `gemm` as its `add`
op at one cycle (`cost_estimator.py:81-85`), so the default greedy pass can
choose MicroGPU for a matmul even though MicroGPU has no matmul lowering. The
partitioner tests exclude MicroGPU to avoid this result
(`test/test_partitioner.py:30-40`).

### AUD-16 Cycle tables disagree with the simulators

The frontend Astra table charges four cycles per DMA byte and four per GELU
element (`_phase7_cycle_tables.py:67-74`). The simulator uses flat per
instruction costs: DMA 4, matmul 16, GELU 4 (`backends/astra/sim/timing.py:21-28`).
The MicroGPU table cites `interp.py` constants that have no cycle model at all.
Tests lock the invented numbers (`test/frontend/test_phase7_cycle_tables.py`).
The partition ranking therefore has no measured basis.

### AUD-17 Events cannot represent asynchronous completion

`Event` stores a plain `bool` (`event.hpp:28-29`) read in a spin loop
(`event.cpp:21`). The scheduler copies one event into every output slot
(`scheduler.cpp:313-319,340`), so an asynchronous provider would complete one
copy. `test_abi.cpp:183-186` marks an event from another thread, which is a
data race under TSan. The shipped providers are synchronous, so the defect is
latent.

### AUD-18 Error paths consume caller buffers

The scheduler moves caller inputs into its table before most validation
(`scheduler.cpp:124-132`) and outputs as well (`:143-151`). Any later
`return Error` (`:259-262`, `:284-287`, `:305-309`) leaves the caller's
vectors moved-from and their buffers freed. `execute()` is single-shot and
cannot be retried after a failed launch.

### AUD-19 No `torch.compile` backend is registered

`backend.py` describes `torch.compile(backend="forgestack")` semantics
(`:126`) and the README repeated the claim. No `torch.library` registration,
`register_backend` call, or Dynamo hook exists. `compile()` is a plain Python
function, and `torch.compile(backend="forgestack")` would raise an unknown
backend error.

### AUD-20 CI gates can pass without checking

`ty check . || { echo ...; }` always exits zero (`scripts/ci.sh:86-88`), and
`error-on-warning=false` (`pyproject.toml:138`) removes the remaining signal.
`ctest` runs without `--no-tests=error` (`ci.sh:34`). lit resolves
`%FileCheck` from `PATH` (`compiler/test/lit.cfg.py:68`), so a missing binary
reports UNSUPPORTED and still exits zero. `probe-env.sh` never checks
FileCheck. The README CI badge is static text; no workflow file exists.

## Medium findings

Three findings in this section were closed by the documentation change that
accompanies this audit: AUD-21, AUD-22 for the README, and AUD-23. The rows
stay as evidence of the audited commit.


| ID | Finding | Evidence |
| --- | --- | --- |
| AUD-21 | No `LICENSE` file, although `pyproject.toml` and the old README declare Apache-2.0. | `pyproject.toml:11`; tree |
| AUD-22 | README links to `.omo/plans/*.md`; the whole `.omo` tree is gitignored and absent. The claimed spec sections have no source. | `README.md:79-80`; `.gitignore:32` |
| AUD-23 | The quickstart runs `setup-env.sh` and then sources `.venv/bin/activate`. On a host where the user is a Nix trusted user, `setup-env.sh` exits zero without creating the venv. | `README.md:27-33`; `scripts/setup-env.sh:35-43` |
| AUD-24 | Naming is split four ways: repository `sparrow`, distribution `forgestack`, package `forge_torch`, runtime `forge_rt`. Only the repository uses the new name. | `pyproject.toml:9`; `CMakeLists.txt:3`; `frontend/`; `runtime/` |
| AUD-25 | `pyproject.toml` has no package discovery configuration. A flat layout with five top-level packages cannot be installed reliably with setuptools defaults. | `pyproject.toml:46-48` |
| AUD-26 | Dependencies are unpinned or loosely bounded: `torch`, `pytest`, `lit`, `numpy`, `pybind11`; `xdsl>=0.68`; `uv.lock` is ignored; `nixos-unstable` is unpinned; shell.nix uses an unhashed `fetchTarball`. | `pyproject.toml:27-33`; `devenv.yaml:4`; `shell.nix:1`; `.gitignore:22-25` |
| AUD-27 | Environment scripts assume Linux: `LD_LIBRARY_PATH` includes `/usr/lib/x86_64-linux-gnu`; probes look in `/usr/include`; `.clangd` points at `/usr/bin/c++`. | `ci.sh:10`; `probe-env.sh:27,74-99`; `.clangd:7` |
| AUD-28 | CMake topology has gaps: `test/CMakeLists.txt` is never added; `forgestack_runtime` calls the Triton kernel without linking `forgestack_triton_cpu`; `FORGESTACK_BUILD_*` only prints messages; the Ninja probe is a hardcoded home path. | `CMakeLists.txt:17-21,52-63`; `runtime/CMakeLists.txt:7-36` |
| AUD-29 | The buffer-reuse allocator loops forever on `alignment == 0` when a conflict exists, because the offset advance is zero. | `buffer_reuse.cpp:135-136`; `memory_planner.hpp:32-34` |
| AUD-30 | The planner can assign an output tensor the same offset as a dead input, and it hardcodes `MemorySpace::Host` even though the header says spaces are preserved. Plan I/O is planned like any other tensor. | `buffer_reuse.cpp:122,150`; `buffer_reuse.hpp:17-21`; `liveness.hpp:35-40` |
| AUD-31 | Subprocess providers ignore `SIGPIPE` while writing stdin. The Astra child exits on non-zero bias, so a late write can terminate the host. | `astra_provider.cpp:132-139,226-231`; `microgpu_provider.cpp:125-132` |
| AUD-32 | `ExecutionPlan` JSON has no version, no unknown-field policy, and `from_json` throws through `at()`, against the no-exception contract. | `execution_plan.cpp:23-155`; `result.hpp:5-6` |
| AUD-33 | Four copies of the FX to forge op map diverge: `relu` becomes GELU in two copies, `addmm` and `bmm` are only in some, and the importer rejects them. | `op_mapping.py:46-56`; `region.py:38-51`; `partitioner.py:64-77`; `backend.py:60-70` |
| AUD-34 | Test quality varies: one test never compiles, e2e numerics pass through eager, differential tests wrap ignored MLIR, and the report test checks constants. | AUD-07; `test_canonicalize_differential.py:90-92`; `test_frontend/test_report.py` |
| AUD-35 | Cache writes are not atomic; malformed `artifact.json` raises an uncaught `JSONDecodeError`; the report path is inside gitignored `.omo` and is overwritten on every miss. | `cache_store.py:61-62`; `dynamic_compile.py:202,218-219`; `backend.py:42,366-367` |
| AUD-36 | `forge.schedule` is a metadata stub with one op, and no pass consumes it. No `TensorToMicroGPU` conversion exists. | `compiler/lib/Dialect/Schedule/__init__.py:8-12`; conversion tree |
| AUD-37 | `compile_partitioned` suppresses every `UserWarning` and calls `backend.compile` with the full model once per region, so per-region callables are identical and capability warnings vanish. | `compile_partitioned.py:250-261` |
| AUD-38 | Provider shapes are narrower than the frontend supports: Triton requires `x` rank 2, Astra requires a 16x16 tile, MicroGPU accepts one input and one output. | `triton_provider.cpp:150-180`; `astra_provider.cpp:412-430`; `microgpu_provider.cpp:287-305` |
| AUD-39 | MicroGPU computes output elements as `byte_size / 4` regardless of dtype, so a non-fp32 buffer is misinterpreted. | `microgpu_provider.cpp:341-342` |
| AUD-40 | No sanitizer step, no coverage report, and no benchmark code; `benchmarks/` is empty. | `ci.sh`; `benchmarks/.gitkeep` |

## Low findings

| ID | Finding | Evidence |
| --- | --- | --- |
| AUD-41 | Two `ValidationResult` classes with different fields. | `backend.py:191-206`; `validation.py:36-49` |
| AUD-42 | Version strings disagree: `0.1.0-phase1` in code, `0.1.0` in metadata. | `_version.py:12`; `pyproject.toml:8` |
| AUD-43 | Dead code and dead surface: `_GRAPH_OPS`, `_build_context`, `_ssa_type`, `_classify_region_op` is only exported, `ValidationResult` in `backend.py` is unused. | `fx_importer.py:68-73,428-438,462`; `compile_partitioned.py:72-83` |
| AUD-44 | Stale section references to a missing spec (`.omo`), "Phase 1-8" comments, and `spec §` citations in 16 files. | tree-wide `grep` |
| AUD-45 | `.clangd` hardcodes `/usr/bin/c++`, which is not the compiler on every host. | `.clangd:7` |
| AUD-46 | The report path `.omo/evidence/phase1-report.txt` is not a stable public output and disappears with the ignored directory. | `backend.py:42`; `.gitignore:32` |

## Test and build inventory

| Suite | Count | Notes |
| --- | ---: | --- |
| pytest files | 47 | 183 test functions, about 231 static parameterized cases |
| lit tests | 26 | 28 `RUN` lines across the five dialects, conversions, and transforms |
| GoogleTest binaries | 9 | 22 `TEST` macros |
| CI gates | 6 | probe, cmake/ctest, lit, pytest, ruff, ty |

Static checks on this host:

```text
ruff check .            all checks passed
ruff format --check .   102 files already formatted
python3 -m compileall   success
scripts/probe-env.sh    4 pass, 8 fail (no cmake, ninja, pkg-config, venv, lit, ty)
```

The two new Excalidraw scenes render at 2860x1635 and 2980x1075 pixels and were
inspected at full size. Their geometry is reproducible from
`docs/figures/generate.py`.

## Remediation order

1. Fix the cache path and the scheduler table indexing. These two defects can
   return the wrong artifact or corrupt memory without a warning.
2. Repair `TensorToAstra` and the Astra codegen: real shapes, real addresses,
   real activation attributes, and an immediate width that fits the data. Until
   then, do not describe `--action=lower-astra` as a working path.
3. Make the scheduler allocate intermediate buffers, handle in-place nodes,
   and plumb artifacts into the plan. Until then, do not describe the runtime
   as able to execute compiled graphs.
4. Make the frontend honest. Either implement the lowered execution path or
   label `compile()` as an import and cache layer that returns an eager
   callable. Remove the "matches eager" claim and the synthetic metrics until
   they measure something.
5. Remove capabilities that have no lowering, or add the missing conversion
   passes. Start with `forge.tensor.reshape` and MicroGPU GEMM.
6. Fix the symbolic-dim parser, result-type verifiers, and the `Fuse` use
   check. These are small changes that turn three silent failures into either
   correct IR or an explicit error.
7. Replace absolute test paths with repository-relative paths and make the CI
   gates fail closed.
8. Add the license file, fix the naming in documentation, and replace the
   dead `.omo` links with `docs/SPEC.md`.

## Changes made in 0.2.0

The documentation set adds `LICENSE`, rewrites `README.md`, and adds
`docs/SPEC.md`, `docs/AUDIT.md`, and `docs/figures/`. The implementation that
followed the audit closes the safety findings and gates the broken lowering
paths. The resolution table below records what changed. The findings still
describe commit `55f64d2`; the status column says what 0.2.0 did about each
one.

## Resolution status, 0.2.0

The implementation that follows this audit closes the safety findings and
gates the features that were producing wrong output. It does not complete the
Astra backend or general heterogeneous execution.

| Status | Findings | Evidence |
| --- | --- | --- |
| Fixed | AUD-01, AUD-06, AUD-35 | Digest path, key manifest, atomic writes, collision tests |
| Fixed | AUD-02, AUD-03, AUD-04, AUD-05 | Dense slots, provider allocation, alias-based in-place, schedule-order liveness |
| Fixed | AUD-10, AUD-11, AUD-18, AUD-39 | Artifact table, capacity checks, non-owning caller buffers, dtype gate |
| Fixed | AUD-12, AUD-20, AUD-21, AUD-22, AUD-23 | Test paths, fail-closed CI, license, README links, quickstart |
| Fixed | AUD-29, AUD-30, AUD-31 | Zero alignment, external I/O aliasing, SIGPIPE |
| Fixed | AUD-19, AUD-25, AUD-42 | torch.compile entry point, package discovery, unified 0.2.0 version |
| Mitigated | AUD-27, AUD-28 | Project fallback and link order fixed; Linux paths and orphan test stub remain |
| Fixed | AUD-51, AUD-52, AUD-53 | Symbolic dim parser, result dtype checks, multi-use fuse guard |
| Mitigated | AUD-07, AUD-08, AUD-13 | Native fused-MLP and CPU plan paths execute supported graphs; unsupported graphs stay eager; reports use optional fields and metrics are marked estimates |
| Mitigated | AUD-09, AUD-36 | Capabilities narrowed; lowering map empty; partitioner raises |
| Mitigated | AUD-47 through AUD-50 | Astra lowering disabled; CLI rejects the request |
| Mitigated | AUD-17, AUD-32 | Shared atomic events; plan version and validating parser |
| Open | AUD-14, AUD-15, AUD-16 | Classic FX importer, cost model, cycle-table drift |
| Open | AUD-24, AUD-26 | Package path rename, pins |
| Open | AUD-33, AUD-34, AUD-37, AUD-38 | Duplicate maps, test gaps, full-model region calls, provider shapes |
| Open | AUD-40, AUD-41, AUD-43 through AUD-46 | Sanitizers, duplicate result types, dead code, stale comments, report durability |
| Open | AUD-54 through AUD-58 | Astra attributes, compiler test depth, DCE, MicroGPU numerics, roundtrip |

## Limits of this audit

The original audit ran without a CMake build, pytest, or ctest. The 0.2.0
follow-up ran the full local gate on macOS with CMake 4.4, Ninja 1.13, xDSL
0.74, PyTorch 2.15, gtest 1.18, and pybind11 3.1: 29 ctest cases, 30 lit tests,
262 pytest cases (1 skipped for missing Triton), ruff, and ty all passed. The
compiler probe used xDSL 0.74, while the project allows 0.68 through 0.74. The
Nix development shell and flake were not evaluated.
