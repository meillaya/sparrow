# Sparrow system specification

| Field | Value |
| --- | --- |
| Version | 0.2.0 |
| Status | Baseline, as built |
| Date | 2026-10-31 |
| Repository | [github.com/meillaya/sparrow](https://github.com/meillaya/sparrow) |
| Companion documents | [README.md](../README.md), [AUDIT.md](AUDIT.md) |
| Normative language | [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt) |

This specification states what Sparrow is, what each layer must do, and which
parts of that contract the shipped code meets. It is a baseline for review and
for later changes, not a description of an aspirational system. The status
column records the state on 2026-10-31. The audit in [AUDIT.md](AUDIT.md)
holds the evidence for every gap.

## 1. Scope

Sparrow compiles a PyTorch model into MLIR, selects a CPU backend for each
region of the captured graph, and executes a plan through a C++ runtime with a
provider per backend. The shipped backends are Triton-CPU, Astra-sim, and
MicroGPU-sim. All three run on a CPU. The shipped implementation covers capture
through MLIR, one native fused-MLP handoff, and the runtime in isolation for
general plans. General region execution remains a Gap in section 9. The project
targets Linux and macOS development hosts, and the Nix development shell
targets Linux.

### 1.1 In scope

- The Python frontend in `frontend/python/forge_torch/`.
- The xDSL dialects, conversions, and passes in `compiler/`.
- The three backends in `backends/`.
- The C++ runtime in `runtime/`.
- The model fixture in `models/transformer_block.py`.
- The build, test, and CI commands that verify the above.

### 1.2 Out of scope

- GPU execution, CUDA, HIP, and ROCm.
- HLS and FPGA bitstream generation.
- Training, autograd, and optimizer support.
- Autotuning and performance measurement against hardware.
- A distributed runtime or multi-host execution.
- Quantized kernels beyond the dtype tokens the IR can express.

### 1.3 Intended readers

Maintainers who change a layer and need the contract of its neighbors, reviewers
who check whether a change preserves the design, and backend authors who add a
provider or a dialect. The specification assumes familiarity with MLIR, xDSL,
`torch.fx`, and C++20.

### 1.4 Relationship to the audit

The audit is a point-in-time record with evidence and remediation order. This
specification states the contract. When they disagree about the shipped
behavior, the audit is the more detailed record. When they disagree about the
intended behavior, this document wins until it is revised.

## 2. Conformance

### 2.1 Requirement keywords

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY carry the meaning given
in RFC 2119. A requirement marked Met is enforced by the shipped code and
covered by a test or a static check. A requirement marked Partial is
implemented with a named deviation. A requirement marked Gap is stated but not
implemented, or is implemented in a way that makes the contract unreliable.

### 2.2 Status values

| Status | Meaning |
| --- | --- |
| Met | The shipped code satisfies the requirement on the supported path, and a check exists |
| Partial | The code implements part of the requirement, or satisfies it only under conditions named in the deviation |
| Gap | The code does not satisfy the requirement, or the path fails in the shipped configuration |
| Declared | The interface exists but has no working implementation behind it |

A component conforms to this specification when every MUST requirement that
applies to it is Met. A component with Partial or Gap requirements does not
conform. The deviation register in section 9 lists the open items.

### 2.3 Verification

Each requirement belongs to a layer with a named verification surface:

| Layer | Verification |
| --- | --- |
| Frontend | pytest under `test/frontend/`, `test/end_to_end/`, and `test/` |
| IR | lit under `compiler/test/` and `forge-opt` exit codes |
| Backends | pytest under `test/backends/` plus the C++ provider tests |
| Runtime | GoogleTest binaries under `runtime/tests/` |
| Operation | `scripts/ci.sh`, `scripts/probe-env.sh`, `ruff`, and `ty` |

A requirement that cites a test which cannot run on a clean host is marked
Partial even when the test body asserts the right thing. The hardcoded paths in
AUD-12 are the main example.

## 3. System model

### 3.1 Layers

| Layer | Directory | Language | Responsibility |
| --- | --- | --- | --- |
| Frontend | `frontend/python/forge_torch/` | Python 3.12 | Capture, import, cache, report, partition, dispatch |
| Compiler | `compiler/lib/` | Python 3.12 | IR definitions, conversions, transforms |
| Driver | `compiler/tools/forge-opt/` | Python 3.12 | Parse, verify, run passes, print |
| Backends | `backends/` | Python + C++ | Kernel templates, codegen, assemblers, simulators |
| Runtime | `runtime/` | C++20 | ABI, plans, scheduler, memory planner, providers |
| Models | `models/` | Python | A transformer block fixture |

The layers communicate through four data products:

| Product | Producer | Consumer | Format |
| --- | --- | --- | --- |
| Captured graph | `torch.fx.symbolic_trace` | Importer, region detector | `torch.fx.GraphModule` |
| MLIR module | `FxImporter`, `forge-opt` | `forge-opt`, backends | Textual MLIR |
| Cache artifact | `backend.compile` | `CacheStore` | JSON with `model_name`, `mlir_text`, `schedule` |
| Execution plan | Frontend native slice and runtime tests | `Scheduler` | JSON with artifacts, tensors, allocations, nodes, I/O |

### 3.2 Process and trust boundaries

The frontend and the compiler run in one Python process. The runtime is a C++
library. The Triton provider calls the CPU kernel in process. The Astra and
MicroGPU providers spawn a Python child with `fork` and `execvp`, write a JSON
spec to its stdin, and read base64 output from its stdout. No layer opens a
network connection at runtime. The subprocess boundary is the only place where
untrusted bytes cross into the runtime, and the provider MUST validate the
child's exit code and output shape.

### 3.3 Naming and compatibility

The project name is Sparrow. The code still uses four historical identifiers:

| Identifier | Where it appears | Status |
| --- | --- | --- |
| `forgestack` | Compatibility CMake aliases | Stable for now |
| `forge_torch` | Python import package | Stable for now |
| `forge_rt` | C++ namespace and include directory | Stable for now |
| ForgeStack | Comments, docstrings, and scripts | Documentation debt |

A rename of an import path or a CMake target is a breaking change and MUST be
done in one change with a compatibility shim. A documentation change MUST use
Sparrow for the project and MAY name the historical identifier once.

### 3.4 Architecture figures

- [Compilation path](figures/compilation.png): PyTorch module, FX graph, MLIR
  passes, partitioner, capabilities, artifacts, cache.
- [Runtime path](figures/runtime.png): plan, liveness, memory planner,
  scheduler, providers, events.

The scenes are in `docs/figures/*.excalidraw` and are generated by
`docs/figures/generate.py`.

## 4. Frontend

### 4.1 Public API

| Function | Signature | Returns | Execution |
| --- | --- | --- | --- |
| `compile` | `(model, example_inputs, *, cache_store=None, target="triton")` | `CompiledFusedMLP` | Native fused MLP when built, otherwise eager |
| `compile_fused_mlp` | `(model, example_inputs)` | `CompiledFusedMLP` | Same as `compile` |
| `dynamic_compile` | `(model, example_inputs, *, target="triton", arch="x86_64_v3", cache_root=None)` | `DynamicCompiledFusedMLP` | Eager |
| `compile_partitioned` | `(model, example_inputs, *, config)` | `CompiledModel` | FX Interpreter |
| `partition` | `(regions, capabilities, cost_estimator=estimate_cost)` | list of `(nodes, backend)` | None |

`compile()` is not a registered `torch.compile` backend. It is a plain Python
function. A caller who writes `torch.compile(backend="forgestack")` gets an
unknown-backend error from PyTorch. The docstring language about
`torch.compile` semantics describes the intended shape of the API, not a
registered hook.

### 4.2 Capture

The frontend captures with `torch.fx.symbolic_trace`. The captured graph MUST
contain no `get_attr` node. A graph with `get_attr` raises
`NotImplementedError` naming the unsupported node. A `call_module` node is
skipped by the importer, which can leave a later consumer without an SSA value
and raise `KeyError`. A model that stores weights as module attributes, such as
a standard `nn.Linear`, is outside the supported set. The `TransformerBlock`
fixture works around this by binding weights in a closure.

### 4.3 Supported operations

| PyTorch target | Forge op | Notes |
| --- | --- | --- |
| `aten.mm`, `aten.matmul` | `forge.graph.matmul` | 2D or batched, dtype must match |
| `aten.add` | `forge.graph.add` | NumPy broadcasting |
| `aten.gelu` | `forge.graph.gelu` | Exact or tanh approximation |
| `aten.view`, `aten._unsafe_view` | `forge.graph.view` | No data movement; left in place by graph-to-tensor |
| `aten.layer_norm` | `forge.tensor.layer_norm` | Emitted directly in the tensor dialect |
| `aten.softmax` | `forge.tensor.softmax` | Emitted directly in the tensor dialect |
| `aten.scaled_dot_product_attention` | `forge.tensor.scaled_dot_product_attention` | Emitted directly in the tensor dialect |

The importer recognises `aten.relu`, `aten.silu`, `aten.mul`, `aten.reshape`,
`aten.permute`, `aten.transpose`, `aten.sum`, `aten.mean`, `aten.bmm`, and
`aten.baddbmm`, and raises `NotImplementedError` for them. The region detector
and partitioner map `aten.relu` to `forge.tensor.gelu`, so the three mapping
tables disagree. The importer table is authoritative for what can be imported.

### 4.4 Dtypes and shapes

The type carries f32, f16, bf16, i32, i8, or i4. The frontend imports the
PyTorch dtypes that map to the first five. The IR accepts symbolic dimensions,
but named dimensions such as `B` and `S` do not parse in the shipped dialect
(see AUD-51). Anonymous `?` dimensions parse. Every shape rule in the tensor
dialect maps a symbolic dimension to -1, so two different symbols compare
equal.

### 4.5 Dynamic shapes

`dynamic_compile` reads `symbolic_dim` from `node.meta` and builds a constraint
signature of the form `"<name> >= 1"`. The signature is stored in the artifact
and folded into the cache key. The bucket directory is
`<root>/dynamic_<sha256(signature)[:16]>`. The bucket does not include the graph,
target, arch, or dtype, so two models with the same constraint signature share
a bucket and the second call loads the first model's MLIR. The `forge-opt`
optimization step is best effort: a missing driver or a failed subprocess
leaves the raw MLIR in place.

### 4.6 Compilation cache

The in-process `CacheKey` has seven fields:

| Field | Meaning |
| --- | --- |
| `graph_hash` | SHA-256 of the MLIR text |
| `target` | `triton`, `astra`, `microgpu`, `hls`, or `cpu` |
| `arch` | Host token from `SPARROW_ARCH` or `platform.machine()` |
| `dtype` | Ordered unique input dtype tokens, joined by `+` |
| `shape_constraints` | `static` or a dynamic bucket label |
| `schedule` | `schedule_64x128x32` |
| `symbolic_constraints` | Tuple of `"<name> >= 1"` strings |

`hexdigest()` folds all seven fields plus the compiler version into a 16-hex
SHA-256 prefix. The disk path is
`<root>/fused_mlp/<arch>/<dtype>/<shape_constraints>/<schedule>/<hexdigest>/artifact.json`.
`CacheStore.put` stamps the digest into `artifact.json`, and `CacheStore.get`
treats a missing or mismatched stamp as a miss. Path components are sanitised
so a target or arch string cannot escape the cache root.

The memory tier is an LRU of 32 entries. The disk tier is unbounded and has no
eviction. Writes go through a temporary sibling file and `os.replace`, so a
crash cannot leave a partial `artifact.json`. A malformed file is renamed to
`artifact.json.corrupt` and treated as a miss.

### 4.7 Compilation report

`CompilationReport` has 14 fields. `compile()` renders the report to
`.cache/sparrow/reports/phase1-report.txt` on a cache miss only and attaches the
dataclass to the returned callable. The path is diagnostic, not durable.

Measured fields carry values: model name, execution path, FX node count, MLIR
operation count, selected target, and schedule. Fields the compile step cannot
measure stay `None` and render as `unavailable`: fusion groups, alternative
costs, intermediate memory, generated kernels (except on the native path), and
validation error. `CompiledFusedMLP.validate()` fills `validation_max_error`
after a real comparison. The MLIR count covers `forge.graph.*` and
`forge.tensor.*` operations.

### 4.8 Validation

`validate()` compares two tensors with dtype-specific tolerances:

| Dtype | atol | rtol |
| --- | ---: | ---: |
| f32 | 1e-4 | 1e-4 |
| f16 | 1e-2 | 1e-2 |
| bf16 | 5e-2 | 5e-2 |

The checks cover shape, dtype, NaN, Inf, and `torch.allclose`. They do not
cover aliasing or input mutation; both flags are hardcoded true.

On the native fused-MLP path, `CompiledFusedMLP.validate()` compares the C++
kernel output to eager and returns the real error. On the falling-back path it
compares eager to eager and labels the result `execution="eager"`.

### 4.9 Capability check

`compile(target=...)` maps the captured FX operations to `forge.tensor.*`
spellings and compares them with `BACKEND_CAPABILITIES[target].supported_ops`.
Each missing op emits one `UserWarning`. An unknown target raises `KeyError`.
The check is advisory: compilation continues and, in `compile_partitioned`,
every `UserWarning` is suppressed.

### 4.10 Region detection and partitioning

`partition_into_regions` walks the FX graph in topological order and merges
adjacent nodes that map to the same `forge.tensor.*` op. Nodes with no mapping
are skipped and never break a region. The result is a list of regions in
topological order. The merge is a heuristic for "one conceptual operation", not
a data-flow fusion analysis.

`partition()` scores each region against every backend whose
`supported_ops` covers the region's op set. The score is
`estimate_cost(region, backend).compute_cycles`. Candidates sort by
`(cycles, backend_name)`, and the first wins. An empty region or a region with
no candidate falls back to `triton`. The partitioner's default cost model maps
`forge.tensor.reshape` to a Triton matmul with a 2D default shape, which can
raise `IndexError`, and it scores MicroGPU GEMM as a one-cycle `add`, which can
make MicroGPU win a region it cannot lower.

### 4.11 Heterogeneous configurations

| Config | Elementwise regions | Matmul regions | Anything else |
| --- | --- | --- | --- |
| A | Triton | Triton | Triton |
| B | Triton | Astra | Triton |
| C | Triton | MicroGPU | Triton |

A region is "matmul" when every mapped op is `forge.tensor.gemm` and no
elementwise or other op is present. A mixed region goes to Triton. The
configuration is a label on the returned `CompiledModel`; execution still runs
the full FX graph through `torch.fx.Interpreter`.

### 4.12 Frontend requirements

| ID | Level | Requirement | Status |
| --- | --- | --- | --- |
| SPA-FE-01 | MUST | `compile()` capture the graph, import MLIR, consult the cache, and return a callable marked compiled | Met |
| SPA-FE-02 | MUST | Capture use `torch.fx.symbolic_trace` and reject `get_attr` with a named error | Met |
| SPA-FE-03 | MUST | An unsupported node raise a diagnostic that names the target | Partial (AUD-14) |
| SPA-FE-04 | MUST | The importer emit MLIR that passes `ModuleOp.verify()` | Met |
| SPA-FE-05 | MUST | The importer accept the seven op rows in 4.3 | Met |
| SPA-FE-06 | MUST | A cache key include graph hash, target, arch, dtype, shape constraints, schedule, and compiler version | Met |
| SPA-FE-07 | MUST | The cache path or artifact identity prevent a lookup from returning another key's artifact | Met |
| SPA-FE-08 | MUST | A dynamic bucket include the graph identity, target, arch, and dtype | Met |
| SPA-FE-09 | MUST | A compiled callable execute the lowered artifact | Partial (native fused MLP only) |
| SPA-FE-10 | MUST | Validation compare the lowered output to eager, including aliasing and mutation | Partial (native fused MLP only) |
| SPA-FE-11 | MUST | The report carry measured counts and costs, not constants | Partial (optional fields, no constants) |
| SPA-FE-12 | MUST | The report be written to a durable, documented path | Gap (AUD-46) |
| SPA-FE-13 | MUST | Region detection preserve topological order and merge only adjacent same-op nodes | Met |
| SPA-FE-14 | MUST | The partitioner filter by capability before scoring | Met |
| SPA-FE-15 | MUST | Ties break deterministically by backend name | Met |
| SPA-FE-16 | MUST | Config A, B, and C follow the table in 4.11 | Met |
| SPA-FE-17 | MUST | Per-region dispatch execute the per-region callable on the region's tensors | Gap (planning labels only) |
| SPA-FE-18 | MUST | Metrics report measured latency, memory, transfers, and error from the executed path | Partial (marked estimate) |
| SPA-FE-19 | SHOULD | A function that claims `torch.compile` semantics register a backend | Gap (AUD-19) |
| SPA-FE-20 | MUST | Cache writes be atomic and malformed artifacts fail closed | Met |

## 5. IR specification

### 5.1 Tensor type grammar

```text
!forge.tensor<<dim>x<dim>...x<dtype>>
!forge.tensor<<dim>x<dim>...x<dtype>, [<constraint>, ...]>
```

A dimension is an integer for a static extent or an identifier for a symbolic
extent. `?` is the anonymous symbolic marker. The dtype is one of `f32`, `f16`,
`bf16`, `i32`, `i8`, or `i4`. The optional constraint list has one entry per
symbolic dimension, in dimension order. Static dims MUST be non-negative. A
tensor MUST have rank at least one.

Example:

```mlir
%x: !forge.tensor<2x512x4096xf32>
%y: !forge.tensor<?x?x11008xf32, [#forge.symbolic_constraint<"B", greater_equal, 1>, #forge.symbolic_constraint<"S", greater_equal, 1>]>
```

Named dimensions in the first form do not parse in the shipped dialect. The
parser consumes `Bx4xf32` as one token and reports an unknown dtype.

### 5.2 Symbolic constraints

```text
#forge.symbolic_constraint<"<name>", <kind>, <value>>
```

| Kind | Meaning |
| --- | --- |
| `greater_equal` | `dim >= value` |
| `equal` | `dim == value` |
| `divisible` | `dim % value == 0` |
| `less_equal` | `dim <= value` |
| `range` | Placeholder; identical to `equal` today |

A constraint list MUST have no more entries than the tensor rank. A `divisible`
value of zero is accepted by the parser, which makes the predicate undefined;
the verifier SHOULD reject it.

### 5.3 Dialects and operations

| Dialect | Operations | Types and attributes |
| --- | --- | --- |
| `forge.graph` | `matmul`, `add`, `gelu`, `view`, `symbolic_guard` | `!forge.tensor`, `#forge.dtype`, `#forge.symbolic_constraint` |
| `forge.tensor` | `gemm`, `fused_epilogue`, `broadcast_add`, `gelu`, `layer_norm`, `softmax`, `scaled_dot_product_attention` | `#forge.activation` (gelu, relu, silu, none) |
| `forge.schedule` | `tiled_gemm` | `#forge.tile_shape<MxNxK>`, `#forge.loop_order`, `#forge.pipeline_stages` |
| `astra` | `compute.matmul`, `microcode.dma_load`, `microcode.dma_store`, `microcode.matmul`, `microcode.vector_gelu` | `!astra.ptr` (dram, sram0, sram1, scratch) |
| `microgpu` | `load_global`, `store_global`, `add`, `mul`, `sigmoid`, `return` | `i32` operands and results |

`forge.graph.matmul` checks that the operand dtypes match and infers the batch,
M, and N extents. `forge.graph.add` checks operand dtypes and computes the
broadcast shape. `forge.graph.gelu` and `symbolic_guard` preserve shape and
dtype. `forge.graph.view` checks the element count when all dimensions are
static. The matmul and add verifiers do not check the result dtype, so an
`f16` result over `f32` operands passes the graph verifier and fails later.

`forge.tensor.gemm` carries `transpose_a` and `transpose_b`.
`fused_epilogue` carries an activation. `gelu` carries a boolean approximation
flag. `layer_norm` carries epsilon. `softmax` carries a dimension. SDPA carries
`num_heads`. All shape checks map a symbolic dimension to -1.

### 5.4 Pass pipeline

| Pass | Contract |
| --- | --- |
| `graph-to-tensor` | One-to-one rewrite of matmul to gemm, add to broadcast_add, gelu to gelu. Leaves `view` and `symbolic_guard` in place. Idempotent on tensor IR. |
| `canonicalize` | Remove dead pure tensor ops and apply local simplifications. The custom DCE pattern documents a GELU-only subset, but the rewrite driver removes every trivially dead op first. |
| `shape-inference` | Recompute result types for gemm, broadcast_add, gelu, and fused_epilogue from operand shapes. Symbols propagate unchanged. |
| `fuse` | Fold `broadcast_add` plus a bare gelu or relu `fused_epilogue` into one bias-bearing `fused_epilogue`. Requires a single-use gemm and crashes on a multi-use add. |
| `tensor-to-astra` | Disabled in 0.2.0. The pass raises `VerifyException` because the dialect cannot represent SRAM addresses, the conversion cannot recover shapes after gemm lowering, and the 15-bit DMA immediate cannot carry real byte counts. |

The default `--action=optimize` chain is `canonicalize, shape-inference, fuse`.
`--action=lower-astra` appends `tensor-to-astra` to that chain without inserting
`graph-to-tensor`, so it fails on graph IR. It also fails on the output of
`graph-to-tensor`, because `fuse` only matches a `fused_epilogue` and the
`broadcast_add` then has no consumer pattern.

### 5.5 Driver

```text
forge-opt [--action=roundtrip|lower|optimize|lower-astra]
          [--pass-pipeline=name,name]
          [--no-auto-dialects] [--all-dialects]
          [--load-dialect FQN] [--split-input-file]
          [file|-]
```

| Exit code | Meaning |
| ---: | --- |
| 0 | Success |
| 1 | Parse error |
| 2 | Verification error |
| 3 | Diagnostic error |
| 4 | Unknown pass name |

`--split-input-file` splits on `// -----` and runs each chunk independently.
The default pipeline silently skips an unregistered pass name; an explicit
name in `--pass-pipeline` exits 4.

### 5.6 IR requirements

| ID | Level | Requirement | Status |
| --- | --- | --- | --- |
| SPA-IR-01 | MUST | Static tensor types parse, print, and verify | Met |
| SPA-IR-02 | MUST | The dtype set in 5.1 parse and print | Met |
| SPA-IR-03 | MUST | Named symbolic dimensions parse and retain their names | Met |
| SPA-IR-04 | MUST | Symbolic constraints parse and attach in dimension order | Met |
| SPA-IR-05 | MUST | `forge.graph` operations enforce their shape and dtype rules | Met |
| SPA-IR-06 | MUST | Result dtypes be checked against operand dtypes | Met |
| SPA-IR-07 | MUST | `forge.tensor` operations expose the attributes listed in 5.3 | Met |
| SPA-IR-08 | MUST | `forge.schedule` attributes enforce their positivity rules | Met |
| SPA-IR-09 | MUST | The default pipeline run canonicalize, shape-inference, and fuse in order | Met |
| SPA-IR-10 | MUST | Every pass be followed by `ModuleOp.verify()` | Met |
| SPA-IR-11 | MUST | `graph-to-tensor` lower every op it claims to lower, including view | Partial (AUD-09) |
| SPA-IR-12 | MUST | `fuse` preserve SSA uses and handle a multi-use add | Met |
| SPA-IR-13 | MUST | `tensor-to-astra` produce correct M, N, K, byte counts, and activation semantics | Gap (AUD-47, AUD-54) |
| SPA-IR-14 | MUST | An op with no lowering be rejected with a verifier error | Met (Astra request rejected) |
| SPA-IR-15 | SHOULD | The IR carry a version so a consumer can reject an incompatible module | Gap (no version field) |

## 6. Backends

### 6.1 Capability registry

Every backend is described by a frozen `BackendCapabilities` record:

| Field | Meaning |
| --- | --- |
| `name` | Registry key |
| `supported_ops` | Set of `forge.tensor.*` names |
| `supported_types` | Set of dtype tokens |
| `local_memory_bytes` | Scratch budget |
| `supports_dynamic_shapes` | Boolean |
| `supports_async_execution` | Boolean |
| `supports_fused_epilogues` | Boolean |

| Backend | Ops | Types | Local memory | Dynamic | Async | Fused |
| --- | --- | --- | ---: | --- | --- | --- |
| `triton` | gemm, fused_epilogue, broadcast_add, gelu, layer_norm, softmax, SDPA | fp32 | 0 | No | No | Yes |
| `astra` | gemm, fused_epilogue, broadcast_add, gelu | fp32 | 512 KiB | No | Yes | Yes |
| `microgpu` | broadcast_add, gelu, layer_norm, softmax | fp32 | 64 KiB | No | No | Yes |

The registry is data. The lowering passes are separate code. A capability MUST
have a matching lowering before the partitioner may select it. The registry no
longer lists ops that have no dialect definition or no hardware path:
`forge.tensor.reshape`, MicroGPU GEMM, and Astra `layer_norm` are gone. The
compiler lowering map is still empty, so `partition()` raises
`UnsupportedRegionError` for every production region until a region codegen
lands.

### 6.2 Triton-CPU

The shipped Triton-CPU path has two implementations:

- A C++ kernel `run_fused_gemm_bias_gelu` that computes `gelu(x @ w + b)` for
  fp32 `x=[M,K]`, `w=[K,N]`, and `b=[N]`. It is a naive triple loop with error
  codes 1001 and 1002. The runtime provider calls it in process.
- A Python kernel module for Triton on CUDA. It is not exercised on a CPU-only
  host. `JITLauncher.compile` calls `triton.compile` with a source string and a
  target, and drops the signature argument.

`emit_triton(region)` ignores the region and returns the source of the fused
kernel template. The C++ path supports fp32 only, 2D input, and one output.

The optional `_sparrow_rt` pybind11 module exposes `run_fused_mlp(x, w, b)`.
It builds a one-node version 2 plan, aliases the NumPy buffers as external
inputs and output, and runs the same `Scheduler` and `TritonProvider`. The
frontend flattens a rank-3 x to `[B*S, K]` and reshapes the result to
`[B, S, N]`. The path is limited to the fused MLP pattern and about 134M
multiply-adds; larger shapes fall back to eager.

### 6.3 Astra-sim

The Astra model is a 16 by 16 systolic tile with a 512 KiB scratchpad and a
64 MiB DRAM. The functional simulator executes a parsed microcode program and
returns the last stored tensor. It records DMA writes so the host can observe
the result.

The microcode word is 64 bits, packed as four 16-bit fields:
`opcode | reg_a | reg_b | imm_or_flags`. The final field is either a 15-bit
immediate or 15 bits of flags; bit 15 marks the flags form. The opcodes are
`NOP=0x00`, `DMA_LOAD=0x01`, `DMA_STORE=0x02`, `MATMUL=0x03`,
`VECTOR_GELU=0x04`, and `HALT=0xFF`.

The timing model charges 4 cycles per DMA instruction, 16 per matmul, 4 per
GELU, and 1 per NOP or HALT. It also tracks bytes resident in SRAM. It does not
model bank conflicts, stalls, or overlap.

The codegen path in `backends/astra/codegen/astra_codegen.py` scrapes textual
IR with regular expressions. It hardcodes `SRAM[0]` and `DRAM[0]`, drops the
M and N attributes, and always emits `elements=0` for GELU. It does not compile
the IR it receives. The compiler pass is disabled in 0.2.0 and raises
`VerifyException`. The provider accepts only the explicit
`astra.reference.tile16` reference program; any other artifact returns error
`2133`.

### 6.4 MicroGPU-sim

MicroGPU is a single-warp SIMT model. An instruction is 16 bits, packed as four
4-bit fields: `opcode | dst | src | imm`. The opcodes are `RETURN=0x00`,
`LOAD_GLOBAL=0x01`, `STORE_GLOBAL=0x02`, `ADD=0x03`, `MUL=0x04`,
`SIGMOID=0x05`, and `NOP=0xFF`. `NOP` is encoded on the wire as `0x0F` so it
fits the 4-bit field.

The interpreter has 32 registers, 4-byte words, a flat memory, WAR and WAW
warnings, and an out-of-bounds exception. It returns an `InterpreterState` with
the register file and a dependency DAG. The arithmetic has defects: `SIGMOID`
rounds its result to 0 or 1, `ADD` and `MUL` wrap as unsigned while loads are
signed, and the DAG omits the destination read for `ADD` and `MUL`.

There is no `TensorToMicroGPU` conversion. The dialect, assembler, interpreter,
and provider exist, but no pass turns `forge.tensor` operations into
`microgpu` operations.

### 6.5 Backend requirements

| ID | Level | Requirement | Status |
| --- | --- | --- | --- |
| SPA-BE-01 | MUST | Every backend expose the seven capability fields | Met |
| SPA-BE-02 | MUST | Triton-CPU compute fused GEMM, bias, and GELU for the documented shapes and return a `Result` | Met |
| SPA-BE-03 | MUST | Astra-sim execute the microcode ISA on a 16x16 tile and return the stored tensor | Met |
| SPA-BE-04 | MUST | MicroGPU-sim encode, decode, assemble, and interpret the 16-bit ISA | Met |
| SPA-BE-05 | MUST | Every capability have a lowering path | Partial (narrowed, no lowerings) |
| SPA-BE-06 | MUST | Backend artifacts carry target, entry point, binary, arguments, guards, and workspace | Met |
| SPA-BE-07 | MUST | An unsupported dtype or shape return an `Error`, not a crash or a partial write | Met |
| SPA-BE-08 | MUST | Simulators produce the same result for the same program and inputs | Met |
| SPA-BE-09 | MUST | Codegen consume IR values and attributes, not textual regex matches | Gap (AUD-48) |
| SPA-BE-10 | MUST | The MicroGPU backend have a tensor-to-microgpu conversion | Gap (AUD-36) |

## 7. Runtime

### 7.1 ABI types

| Enum | Values | Wire rule |
| --- | --- | --- |
| `DataType` | `Float32=0`, `Float16=1`, `BFloat16=2`, `Int32=3`, `Int8=4`, `Int4=5` | Append only |
| `MemorySpace` | `Host=0`, `Device0=1`, `Device1=2` | Append only |
| `TargetKind` | `Triton=0`, `HLS=1`, `Astra=2`, `MicroGPU=3`, `CPU=4` | Append only |

`kByteSize` returns 4, 2, 2, 4, 1, and 1 for the six dtypes. Int4 packs two
values per byte, and the tensor descriptor rounds the element count up.

`TensorDescriptor` holds dtype, shape, strides in elements, memory space, and a
byte offset. `numel` treats a scalar or empty shape as 1 and a non-positive
dimension as 0. `is_contiguous` requires exact row-major strides.

### 7.2 Result and error

`Result<T>` is a `std::variant<T, Error>`. `Error` holds a message and an
integer code; zero means unspecified. Callers MUST check `ok()` before
`value()`. `value()` on an error throws `std::bad_variant_access`, so the
"no exceptions" rule holds only for callers that follow the check.

`Buffer` is move-only. A buffer constructed with `owns=true` frees its data with
`std::free` in the destructor.

`Event` holds an id and shared completion state. Copies share one atomic flag,
so the scheduler can publish a launch to several outputs and a provider can
mark completion from another thread. `wait(timeout)` returns ok when the flag
is set before the deadline; a zero timeout polls once.

### 7.3 Artifact descriptor

| Field | Type | Meaning |
| --- | --- | --- |
| `target` | `TargetKind` | Backend selector |
| `entry_point` | string | Kernel symbol or node label |
| `binary` | bytes | Opaque artifact: microcode, bytecode, or source |
| `arguments` | list | Name, dtype, input flag, shape |
| `guards` | list | Shape guard kind, indices, value |
| `workspace_bytes` | uint64 | Scratch the kernel needs |

`ShapeGuardKind` is `Equal=0`, `Divisible=1`, `GreaterEqual=2`, or
`LessEqual=3`. The descriptor serializes as part of the plan's artifact table.
`ExecutableNode.artifact` indexes that table; the scheduler copies the node's
launch geometry into the descriptor before calling the provider.

### 7.4 Execution plan

The plan is a flat JSON document:

```json
{
  "version": 2,
  "name": "fused_mlp",
  "artifacts": [
    {
      "target": 0,
      "entry_point": "fused_gemm_bias_gelu",
      "binary": [],
      "arguments": [],
      "guards": [],
      "workspace_bytes": 0,
      "launch": {"grid": [], "block": [], "shared_memory_bytes": 0}
    }
  ],
  "tensors": [
    {"id": 0, "name": "x", "dtype": 0, "shape": [2, 512, 4096], "memory_space": 0},
    {"id": 1, "name": "y", "dtype": 0, "shape": [2, 512, 11008], "memory_space": 0}
  ],
  "allocations": [
    {"tensor_id": 0, "byte_offset": 0, "byte_size": 16777216, "alignment": 64, "memory_space": 0},
    {"tensor_id": 1, "byte_offset": 0, "byte_size": 45088768, "alignment": 64, "memory_space": 0}
  ],
  "nodes": [
    {
      "id": 0,
      "device": 0,
      "artifact": 0,
      "inputs": [0],
      "outputs": [1],
      "dependencies": [],
      "launch": {"grid": [], "block": [], "shared_memory_bytes": 0}
    }
  ],
  "inputs": [{"tensor_id": 0, "external_name": "x"}],
  "outputs": [{"tensor_id": 1, "external_name": "y"}]
}
```

Enums serialize as their integer values. `parse_plan_json` and
`parse_plan_text` return a `Result<ExecutionPlan>` and reject an unsupported
version, a duplicate id, or a node that names a missing artifact. The ADL
`from_json` used by round-trip tests still throws on a malformed payload.

### 7.5 Scheduler

`Scheduler::execute(plan, inputs, outputs)` runs these steps:

1. Validate the plan version (`-42`), arity (`-11`, `-13`), duplicate ids
   (`-28`), node artifact references (`-18`), and the input and output tensor
   ids (`-12`, `-14`).
2. Build a dense tensor slot per declared tensor. The live table is sized by
   tensor count, so a sparse id cannot force a large allocation.
3. Compute the topological order with Kahn's algorithm. An unknown dependency
   is error `-15`; a cycle is error `-20`.
4. Run `MemoryPlanner::plan(plan, order)` with the default 512 MiB pool and
   64-byte alignment.
5. Alias the caller's input and output buffers into the table as non-owning
   `Buffer` views, then allocate every intermediate through the provider whose
   `memory_space()` matches the allocation (`-26`). A failed allocation is
   `-26`.
6. Mark every input event complete.
7. For each node in order, check the device index (`-22`), resolve the real
   artifact, copy the node launch geometry, check each output capacity (`-27`),
   call `DeviceProvider::launch`, and publish the returned event to every
   output.
8. Return one event per `plan.outputs` entry, in order.

The scheduler never moves from the caller's vectors. A failed run leaves every
caller buffer valid. In-place nodes work because the input and output aliases
point at the same table buffer. The scheduler does not take ownership of
external buffers; only intermediates are owned by the table.

### 7.6 Memory planner

`MemoryPlanner::plan(plan, order, total_bytes=512 MiB, alignment=64)` runs
liveness in schedule order, stamps each range with its byte size, and lays the
internal ranges out with first-fit decreasing:

1. Sort ranges by lifetime descending, then byte size descending, then id
   ascending.
2. Walk candidate offsets in alignment steps from zero.
3. Place a range at the first offset where it does not overlap, in bytes, a
   range whose lifetime overlaps.
4. Return an empty allocation list if a range does not fit, which becomes error
   `-40`.

The result has one allocation per tensor, in tensor order, with offsets that
are multiples of `alignment`. Liveness intervals are inclusive schedule
positions, not raw node ids. External inputs and outputs are excluded from the
reuse pool and get offset 0 with their real size; their `MemorySpace` comes
from the tensor definition. An alignment of zero is rejected with error `-41`
before the offset walk starts.

### 7.7 Device provider

A provider owns allocation, copy, and launch for one memory space and target:

```cpp
Result<Buffer> allocate(const BufferSpec& spec);
Result<Event> copy(Buffer& dst, const Buffer& src, std::span<const Event> deps);
Result<Event> launch(const ArtifactDescriptor& artifact,
                     std::span<const Buffer> inputs,
                     std::span<Buffer> outputs,
                     std::span<const Event> deps);
Result<void> wait(const Event& event);
MemorySpace memory_space() const;
TargetKind target() const;
```

`allocate` MUST honor `spec.alignment`. `copy` is host-to-host in the shipped
providers and synchronous. `launch` MUST validate its input and output counts,
dtypes, and shapes before writing. `wait` MUST return ok only when the event is
complete.

| Provider | Invocation | Shape contract | Notes |
| --- | --- | --- | --- |
| `TritonProvider` | In-process C++ call | `x=[M,K]`, `w=[K,N]`, `b=[N]`, `y=[M,N]`, fp32 | Allocates with `std::aligned_alloc` |
| `AstraProvider` | `fork` and `execvp` of `python3 -c` | `x=[16,K]`, `w=[K,16]`, `b=[16]`, `y=[16,16]`, fp32 | Reference entry `astra.reference.tile16`; mode byte selects timing |
| `MicroGpuProvider` | `fork` and `execvp` of `python3 -c` | One input, one output, little-endian 16-bit words | Output element count is `byte_size / 4` |

The subprocess providers write a JSON spec to stdin and read base64 data from
stdout. They ignore `SIGPIPE` before the write, so a child that exits early
returns an error instead of terminating the host. They locate the project root
by walking up from the current directory; the fallback is the current
directory.

### 7.8 Runtime requirements

| ID | Level | Requirement | Status |
| --- | --- | --- | --- |
| SPA-RT-01 | MUST | ABI enum values stay stable and new values append | Met |
| SPA-RT-02 | MUST | Provider and scheduler paths report errors without throwing | Met (parse returns Result) |
| SPA-RT-03 | MUST | `Buffer` remain move-only and free owned storage once | Met |
| SPA-RT-04 | MUST | The plan JSON match the schema in 7.4 | Met |
| SPA-RT-05 | SHOULD | The plan carry a version and reject unknown fields | Met |
| SPA-RT-06 | MUST | The scheduler topologically order nodes and detect cycles | Met |
| SPA-RT-07 | MUST | The scheduler validate every node id before indexing a table | Met |
| SPA-RT-08 | MUST | The scheduler allocate or require backing storage for every intermediate tensor | Met |
| SPA-RT-09 | MUST | In-place nodes preserve the live buffer | Met |
| SPA-RT-10 | MUST | The scheduler resolve a real `ArtifactDescriptor` for every node | Met |
| SPA-RT-11 | MUST | Liveness use the scheduler's topological order, not raw node ids | Met |
| SPA-RT-12 | MUST | The planner reject a zero alignment and preserve caller allocations | Met |
| SPA-RT-13 | MUST | External inputs and outputs not alias each other through reuse | Met |
| SPA-RT-14 | MUST | The scheduler return one event per output and define its completion | Met |
| SPA-RT-15 | MUST | Every provider implement the six methods in 7.7 | Met |
| SPA-RT-16 | MUST | A subprocess provider validate exit code, output shape, and capacity | Partial (Astra artifact mode gated) |
| SPA-RT-17 | MUST | Shared events and provider counters be safe for the documented threading model | Met |
| SPA-RT-18 | MUST | A failed `execute()` leave caller buffers valid | Met |
| SPA-RT-19 | MUST | A provider check `byte_size` before writing any output | Met |

## 8. Operation

### 8.1 Build

| Target | Build system | Output |
| --- | --- | --- |
| `sparrow_runtime` | CMake 3.25, C++20 | Static library with the ABI, scheduler, planner, providers |
| `sparrow_triton_cpu` | CMake | Static library with the fused kernel |
| `_sparrow_rt` | CMake + pybind11 | Native bridge for the fused MLP slice |
| `compiler/` | Python, no build step | xDSL packages |
| `backends/` | Python, no build step | Codegen, assemblers, simulators |
| `frontend/` | Python, no build step | `forge_torch` package |

The top-level build adds `backends/triton/cpu/`, `runtime/`, and `bindings/`.
`SPARROW_BUILD_COMPILER`, `SPARROW_BUILD_BACKENDS`, and
`SPARROW_BUILD_PYTHON` control the optional targets. The first two default to
off; the Python bridge defaults to on. The deprecated `FORGESTACK_BUILD_*`
spellings are accepted for one release. There is no CMake target for `test/`;
pytest runs outside CMake.

`sparrow_runtime` links `sparrow_triton_cpu`, so a consumer that links only the
runtime resolves the kernel symbol. `forgestack::runtime` and
`forgestack::triton_cpu` remain as compatibility aliases.

### 8.2 Test and CI

```bash
bash scripts/ci.sh
```

The gate runs six steps:

1. `scripts/probe-env.sh` checks the native toolchain, C++ headers, the Python
   venv, the Python imports, lit, uv, ruff, and ty.
2. `cmake -B build -S . -G Ninja -DPython_EXECUTABLE="$PYTHON"`, then
   `cmake --build build`, then
   `ctest --test-dir build --output-on-failure --no-tests=error`. The build
   produces the runtime, the tests, and the optional `_sparrow_rt` module.
3. `lit -v compiler/test/`.
4. `pytest -q test/` with `build/python` on `PYTHONPATH`.
5. `ruff check .` and `ruff format --check .`.
6. `ty check . --python "$PYTHON"`. The step has no fallback.

`ctest` fails when it collects no tests. `lit.cfg.py` resolves `FileCheck`
with `shutil.which` and fails the test when it is missing. There is no workflow
file in the repository and no sanitizer or coverage step.

### 8.3 Environment assumptions

- A repository checkout with a writable working directory.
- Python 3.12 or 3.13, `torch`, `xdsl`, `numpy`, `pybind11`, `lit`, `pytest`,
  `ruff`, and `ty`.
- A C++20 compiler, CMake 3.25, and Ninja for the runtime tests.
- Linux for the Nix development shell and the `/usr/lib` linker path. macOS
  works for the Python, native, and Excalidraw steps when CMake, Ninja,
  pybind11, and the Python development headers are available.

### 8.4 Operation requirements

| ID | Level | Requirement | Status |
| --- | --- | --- | --- |
| SPA-OPS-01 | MUST | `scripts/probe-env.sh` fail when a required tool is missing | Met |
| SPA-OPS-02 | MUST | `scripts/ci.sh` build the runtime and run ctest | Met |
| SPA-OPS-03 | MUST | CI run every lit test | Met |
| SPA-OPS-04 | MUST | CI run every pytest file | Met |
| SPA-OPS-05 | MUST | CI run ruff lint and format checks | Met |
| SPA-OPS-06 | MUST | CI fail on a type error | Met |
| SPA-OPS-07 | MUST | Tests run from a clean checkout on any supported host | Met |
| SPA-OPS-08 | MUST | CI fail when ctest collects no tests or FileCheck is missing | Met |
| SPA-OPS-09 | MUST | A `LICENSE` file match the declared Apache-2.0 license | Met (added with this document set) |
| SPA-OPS-10 | MUST | Documentation links resolve inside the repository | Met (README and specification links) |
| SPA-OPS-11 | SHOULD | The test count in the README be reproducible from one command | Gap (AUD-34) |
| SPA-OPS-12 | SHOULD | The default dev shell pin every dependency or record a lock file | Partial (AUD-26) |

## 9. Deviations and change control

### 9.1 Deviation register

| Area | Requirements | Audit evidence | Effect |
| --- | --- | --- | --- |
| Execution | SPA-FE-09, SPA-FE-10, SPA-FE-17, SPA-FE-18 | AUD-07, AUD-08, AUD-13, AUD-37 | Native only for the fused MLP; general graphs run eager; region assignments are labels |
| Astra lowering | SPA-IR-13, SPA-BE-09, SPA-BE-10 | AUD-47 through AUD-50, AUD-36 | The CLI rejects the path; the dialect and simulator remain research code |
| Capability parity | SPA-BE-05 | AUD-09 | The registry is narrowed and the lowering map is empty; the partitioner fails closed |
| Report durability | SPA-FE-12 | AUD-46 | The report lives under the ignored cache root |
| torch.compile hook | SPA-FE-19 | AUD-19 | No backend is registered with PyTorch |
| Test count | SPA-OPS-11 | AUD-34 | No reproducible count is published |
| Dependency pins | SPA-OPS-12 | AUD-26 | Bounds exist; no lock file or pinned nixpkgs |

### 9.2 Change control

A change that alters a requirement MUST update this document and the
corresponding status. A change that closes a Gap SHOULD link the audit finding
it closes and add a test that fails before the change. A change that adds a
backend MUST add the capability record, a conversion pass, a provider, a cache
key contribution, and a test.

A change to the ABI MUST append enum values and MUST NOT reorder them. A change
to the plan JSON MUST add a version field and a reader that rejects a future
version. A change to a pass MUST add or update a lit case that checks the
transformed IR, and it must not stop at the verifier.

## Appendix A: file map

```text
compiler/
  lib/Dialect/{Graph,Tensor,Schedule,Astra,MicroGPU}/   IR definitions
  lib/Conversion/{GraphToTensor,TensorToAstra}/        conversions
  lib/Transforms/{Canonicalize,ShapeInference,Fuse}/   passes
  tools/forge-opt/                                     driver
  test/                                                lit cases
frontend/python/forge_torch/                           Python frontend
backends/triton/                                       kernel template and C++ kernel
backends/astra/                                        microcode assembler and simulators
backends/microgpu/                                     ISA, assembler, interpreter
runtime/include/forge_rt/                              C++ ABI
runtime/lib/                                           scheduler, planner, providers
runtime/tests/                                         GoogleTest binaries
models/transformer_block.py                            fixture
test/                                                  pytest suites
scripts/                                               environment, CI
docs/                                                  specification, audit, figures
```

## Appendix B: glossary

| Term | Meaning |
| --- | --- |
| Artifact | The opaque output of a backend for one kernel |
| Capture | The `torch.fx` trace of a model |
| Capability | The op, dtype, memory, and feature record a backend declares |
| Config A/B/C | The three heterogeneous assignments in 4.11 |
| Execution plan | The flat runtime description of tensors, allocations, and nodes |
| ForgeRT | The C++ runtime layer (`forge_rt`) |
| Gap | A requirement the shipped code does not meet |
| Lowering | Rewriting IR from a higher dialect to a lower one |
| Provider | A `DeviceProvider` implementation for one target |
| Region | A run of adjacent FX nodes with the same mapped op |
| Symbolic constraint | A `#forge.symbolic_constraint` attached to a dimension |
