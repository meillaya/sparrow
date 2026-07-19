// ForgeRT — GoogleTest suite for BufferReuseAllocator (spec §10, Phase 3).
// Two test cases:
//   1. Linear_Plan — three ranges where ranges 2 and 3 are temporally
//      non-overlapping. The allocator must place them at the same
//      offset because the earlier one has gone out of scope by the
//      time the later one is placed.
//   2. No_Overlap — every pair of ranges overlaps (e.g. one long
//      lifetime running the entire plan) so no two allocations share
//      an offset.
#include "forge_rt/buffer_reuse.hpp"
#include "forge_rt/liveness.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

using forge_rt::Allocation;
using forge_rt::BufferReuseAllocator;
using forge_rt::LiveRange;
using forge_rt::TensorId;

namespace {

LiveRange MakeRange(TensorId id, uint32_t first_use, uint32_t last_use,
                    uint64_t byte_size) {
    LiveRange r;
    r.id        = id;
    r.first_use = first_use;
    r.last_use  = last_use;
    r.byte_size = byte_size;
    return r;
}

// Resolve an allocation by tensor id; the test asserts on offsets
// without caring about order.
const Allocation* FindAlloc(const std::vector<Allocation>& allocs, TensorId id) {
    for (const auto& a : allocs) {
        if (a.id == id) return &a;
    }
    return nullptr;
}

}  // namespace

// ===========================================================================
// 1. Linear plan reuse — range 1 (live 0..1) and range 2 (live 2..3)
//    do not overlap. Both are smaller than range 0 (live 0..3), so
//    under first-fit-decreasing, range 0 anchors at offset 0 and the
//    two short-lived ranges can land at offset 0 too once range 0's
//    slot frees up. We assert that ranges 1 and 2 share the same
//    offset (whichever offset the allocator picks is fine — we just
//    want reuse).
// ===========================================================================
TEST(BufferReuseAllocator_linear, NonOverlappingRangesShareOffset) {
    std::vector<LiveRange> ranges = {
        MakeRange(/*id=*/0, /*first=*/0, /*last=*/3, /*bytes=*/1024),
        MakeRange(/*id=*/1, /*first=*/0, /*last=*/1, /*bytes=*/256),
        MakeRange(/*id=*/2, /*first=*/2, /*last=*/3, /*bytes=*/256),
    };

    const auto allocs = BufferReuseAllocator::allocate(ranges, /*total=*/4096,
                                                       /*align=*/64);
    ASSERT_EQ(allocs.size(), 3u);

    const Allocation* a1 = FindAlloc(allocs, 1);
    const Allocation* a2 = FindAlloc(allocs, 2);
    ASSERT_NE(a1, nullptr);
    ASSERT_NE(a2, nullptr);

    // Range 1 and Range 2 do not overlap → they must reuse storage.
    EXPECT_EQ(a1->offset, a2->offset)
        << "expected ranges 1 and 2 to share an offset, got "
        << a1->offset << " and " << a2->offset;

    // Both must be 64-byte aligned.
    EXPECT_EQ(a1->offset % 64u, 0u);
    EXPECT_EQ(a2->offset % 64u, 0u);

    // And the byte_size we asked for was preserved.
    EXPECT_EQ(a1->byte_size, 256u);
    EXPECT_EQ(a2->byte_size, 256u);
}

// ===========================================================================
// 2. No overlap — every range's interval covers the entire plan, so
//    no two allocations may share an offset. The allocator must hand
//    out strictly distinct offsets in this case.
// ===========================================================================
TEST(BufferReuseAllocator_no_overlap, OverlappingRangesGetDistinctOffsets) {
    std::vector<LiveRange> ranges = {
        MakeRange(0, 0, 5, 64),
        MakeRange(1, 0, 5, 64),
        MakeRange(2, 0, 5, 64),
        MakeRange(3, 0, 5, 64),
    };

    const auto allocs = BufferReuseAllocator::allocate(ranges, /*total=*/4096,
                                                       /*align=*/64);
    ASSERT_EQ(allocs.size(), 4u);

    // Pairwise distinct offsets.
    for (std::size_t i = 0; i < allocs.size(); ++i) {
        for (std::size_t j = i + 1; j < allocs.size(); ++j) {
            EXPECT_NE(allocs[i].offset, allocs[j].offset)
                << "ranges " << allocs[i].id << " and " << allocs[j].id
                << " share offset " << allocs[i].offset
                << " despite overlapping lifetimes";
        }
    }

    // Every offset is alignment-honouring.
    for (const auto& a : allocs) {
        EXPECT_EQ(a.offset % 64u, 0u);
        EXPECT_EQ(a.byte_size, 64u);
    }

    // No allocation exceeds the pool.
    for (const auto& a : allocs) {
        EXPECT_LE(a.offset + a.byte_size, 4096u);
    }
}
