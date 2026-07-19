// ForgeRT — BufferReuseAllocator (spec §10, Phase 3).
// First-fit-decreasing layout for LiveRanges inside a fixed pool,
// lifetime-aware.
//
// Algorithm:
//   1. Sort ranges by (last_use - first_use) DESC, then byte_size DESC.
//      FFD's anchor heuristic: long-lived ranges settle first so the
//      short-lived siblings have the largest pool of "free at this
//      offset" slots to pick from.
//   2. For each range, walk offsets 0, alignment, 2*alignment, ... and
//      place at the first offset where no placed range P has both
//      overlapping lifetime AND [P.offset, P.offset + P.byte_size)
//      intersecting the candidate slot. This is the temporal-aware
//      variant of first-fit: an offset is "free for R" iff no range
//      alive at any of R's nodes already occupies that address range.
//   3. If a range doesn't fit (offset + byte_size > total_bytes or
//      every candidate conflicts), return an empty vector — the
//      caller is expected to size `total_bytes` generously.
//
// Complexity: O(N^2 * total_bytes / alignment) in the worst case; for
// Phase 3 plans N is small and alignment is coarse, so the inner loop
// is cheap. A heap-based variant is out of scope.
//
// The returned vector is in ascending TensorId order so callers can
// zip it back onto plan.tensors without an extra index lookup.
#include "forge_rt/buffer_reuse.hpp"

#include "forge_rt/abi_types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace forge_rt {

namespace {

inline uint64_t align_up(uint64_t offset, uint64_t alignment) {
    if (alignment <= 1) return offset;
    const uint64_t rem = offset % alignment;
    return rem == 0 ? offset : offset + (alignment - rem);
}

// True iff two inclusive NodeId intervals overlap.
inline bool intervals_overlap(uint32_t a_first, uint32_t a_last,
                              uint32_t b_first, uint32_t b_last) {
    return a_first <= b_last && b_first <= a_last;
}

// True iff two half-open byte ranges [a_off, a_off+a_sz) and
// [b_off, b_off+b_sz) intersect.
inline bool ranges_intersect(uint64_t a_off, uint64_t a_sz,
                             uint64_t b_off, uint64_t b_sz) {
    return a_off < b_off + b_sz && b_off < a_off + a_sz;
}

}  // namespace

std::vector<std::pair<TensorId, uint64_t>>
BufferReuseAllocator::compute_byte_sizes(const ExecutionPlan& plan) {
    std::vector<std::pair<TensorId, uint64_t>> out;
    out.reserve(plan.tensors.size());
    for (const TensorDefinition& td : plan.tensors) {
        uint64_t numel = 1;
        bool zero_dim = false;
        for (int64_t d : td.shape) {
            if (d <= 0) {
                zero_dim = true;
                break;
            }
            numel *= static_cast<uint64_t>(d);
        }
        uint64_t bytes = 0;
        if (!zero_dim) {
            const uint64_t per = kByteSize(td.dtype);
            if (td.dtype == DataType::Int4) {
                bytes = (numel + 1) / 2;
            } else {
                bytes = numel * per;
            }
        }
        out.emplace_back(td.id, bytes);
    }
    return out;
}

std::vector<Allocation> BufferReuseAllocator::allocate(
    const std::vector<LiveRange>& ranges,
    uint64_t total_bytes,
    uint64_t alignment) {
    std::vector<Allocation> out;
    if (ranges.empty()) return out;

    struct Placed {
        TensorId id;
        uint64_t offset;
        uint64_t byte_size;
        uint32_t first_use;
        uint32_t last_use;
    };
    std::vector<Placed> placed;
    placed.reserve(ranges.size());
    out.reserve(ranges.size());

    std::vector<LiveRange> sorted = ranges;
    std::sort(sorted.begin(), sorted.end(),
              [](const LiveRange& a, const LiveRange& b) {
                  const uint64_t la = static_cast<uint64_t>(a.last_use) -
                                      static_cast<uint64_t>(a.first_use);
                  const uint64_t lb = static_cast<uint64_t>(b.last_use) -
                                      static_cast<uint64_t>(b.first_use);
                  if (la != lb) return la > lb;
                  if (a.byte_size != b.byte_size) return a.byte_size > b.byte_size;
                  return a.id < b.id;
              });

    for (const LiveRange& r : sorted) {
        if (r.byte_size == 0) {
            // Zero-byte ranges — pin at offset 0 with no consumption.
            out.push_back(Allocation{r.id, 0, 0, alignment, MemorySpace::Host});
            placed.push_back(Placed{r.id, 0, 0, r.first_use, r.last_use});
            continue;
        }
        if (r.byte_size > total_bytes) {
            return std::vector<Allocation>{};
        }

        bool placed_ok = false;
        // Walk offsets in alignment steps. For each, check every
        // already-placed range: a placement conflicts iff the placed
        // range overlaps R in lifetime AND its byte footprint
        // intersects [offset, offset + R.byte_size).
        for (uint64_t offset = 0; offset + r.byte_size <= total_bytes;
             offset += alignment) {
            bool conflict = false;
            for (const Placed& p : placed) {
                if (p.byte_size == 0) continue;
                const bool life_overlap = intervals_overlap(
                    p.first_use, p.last_use, r.first_use, r.last_use);
                if (!life_overlap) continue;
                if (ranges_intersect(p.offset, p.byte_size, offset, r.byte_size)) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                out.push_back(Allocation{r.id, offset, r.byte_size,
                                         alignment, MemorySpace::Host});
                placed.push_back(Placed{r.id, offset, r.byte_size,
                                        r.first_use, r.last_use});
                placed_ok = true;
                break;
            }
        }
        if (!placed_ok) {
            return std::vector<Allocation>{};
        }
    }

    std::sort(out.begin(), out.end(),
              [](const Allocation& a, const Allocation& b) { return a.id < b.id; });
    return out;
}

}  // namespace forge_rt
