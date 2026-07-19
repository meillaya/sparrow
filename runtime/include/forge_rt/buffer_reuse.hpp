#pragma once

// ForgeRT C++ ABI — buffer-reuse allocator (spec §10, Phase 3).
// Given a vector of LiveRanges and a pool of `total_bytes`, lay out
// each range into the pool such that no two simultaneously-live
// ranges overlap, and reuse the smallest possible set of offsets.
//
// Algorithm: first-fit-decreasing. We sort ranges by
// `(last_use - first_use)` DESC, then by `byte_size` DESC. The longest
// lifetime (and within that, the largest byte_size) goes first, so
// the short-lived / small allocations have the largest pool of free
// gaps to fall into. We walk a sorted free-gap list and place each
// range in the first gap whose aligned size is large enough. If no
// gap fits we return an error — the caller decides whether to retry
// with a bigger pool or fall back to per-tensor alloc.
//
// All offsets honour `alignment`; the returned Allocation::offset is
// always a multiple of `alignment`. MemorySpace is copied from the
// caller's LiveRange so the planner can preserve device vs host
// distinctions later (Phase 3 uses a single space; Phase 7 wires the
// partitioner).
#include <cstdint>
#include <utility>
#include <vector>

#include "forge_rt/abi_types.hpp"
#include "forge_rt/execution_plan.hpp"
#include "forge_rt/liveness.hpp"

namespace forge_rt {

// A pool-relative allocation slot. `offset` is byte-offset from the
// start of the backing buffer; `byte_size` is the requested footprint
// (already padded to `alignment` upstream). `space` mirrors the
// tensor's memory space so the runtime can resolve providers without
// a separate lookup.
struct Allocation {
    TensorId id = 0;
    uint64_t offset = 0;
    uint64_t byte_size = 0;
    uint64_t alignment = 0;
    MemorySpace space = MemorySpace::Host;
};

class BufferReuseAllocator {
public:
    // Compute the byte size for every tensor in the plan, in
    // plan.tensors order. numel * kByteSize(dtype), with Int4's
    // packing handled like TensorDescriptor::byte_size. Phase 3's
    // executor doesn't reshape tensors so this is purely a metadata
    // helper for the allocator.
    static std::vector<std::pair<TensorId, uint64_t>>
        compute_byte_sizes(const ExecutionPlan& plan);

    // Lay out `ranges` into a `total_bytes` pool honouring `alignment`.
    // Returns the per-range allocations in input order, OR an empty
    // vector with an error message propagated via `Error{...}`.
    // Phase 3 doesn't have a Result<std::vector<Allocation>> path —
    // a non-ok return is signalled by returning an empty vector and
    // setting the optional out-parameter. To keep the public surface
    // narrow, the planner passes ranges that already know they fit;
    // out-of-pool failures land in tests + MemoryPlanner::plan only.
    //
    // Signature is `static std::vector<Allocation> allocate(...)` per
    // the task spec; we keep it as specified and rely on the caller to
    // size `total_bytes` correctly. Empty return vector means "did
    // not fit".
    static std::vector<Allocation> allocate(
        const std::vector<LiveRange>& ranges,
        uint64_t total_bytes,
        uint64_t alignment);
};

}  // namespace forge_rt
