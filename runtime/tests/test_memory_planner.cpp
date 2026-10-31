// ForgeRT — GoogleTest suite for MemoryPlanner (spec §10, Phase 3).
// Single end-to-end test: build a 4-node diamond plan with mixed-shape
// float32 tensors, run MemoryPlanner::plan, and verify the resulting
// plan.allocations satisfies:
//   - exactly one entry per tensor,
//   - every offset is alignment-honouring,
//   - no two overlapping-lifetime tensors share any byte in [offset, offset+size),
//   - the total high-water mark is strictly less than the total pool size.
#include "forge_rt/execution_plan.hpp"
#include "forge_rt/liveness.hpp"
#include "forge_rt/memory_planner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using forge_rt::AllocationDefinition;
using forge_rt::DataType;
using forge_rt::ExecutionPlan;
using forge_rt::ExecutableNode;
using forge_rt::LaunchParameters;
using forge_rt::LivenessAnalyzer;
using forge_rt::LiveRange;
using forge_rt::MemoryPlanner;
using forge_rt::MemorySpace;
using forge_rt::NodeId;
using forge_rt::Result;
using forge_rt::TensorDefinition;
using forge_rt::TensorId;

namespace {

constexpr uint64_t kAlign = 64;
constexpr uint64_t kTotal = 4ULL * 1024 * 1024;  // 4 MiB pool — plenty for these shapes

TensorDefinition MakeTensor(TensorId id, const std::string& name,
                            std::vector<int64_t> shape) {
    TensorDefinition t;
    t.id = id;
    t.name = name;
    t.dtype = DataType::Float32;
    t.shape = std::move(shape);
    t.memory_space = MemorySpace::Host;
    return t;
}

ExecutableNode MakeNode(NodeId id, std::vector<TensorId> in,
                        std::vector<TensorId> out,
                        std::vector<NodeId> deps) {
    ExecutableNode n;
    n.id = id;
    n.device = 0;
    n.artifact = id;
    n.inputs = std::move(in);
    n.outputs = std::move(out);
    n.dependencies = std::move(deps);
    n.launch = LaunchParameters{};
    return n;
}

const AllocationDefinition* FindAlloc(const ExecutionPlan& p, TensorId id) {
    for (const auto& a : p.allocations) {
        if (a.tensor_id == id) return &a;
    }
    return nullptr;
}

}  // namespace

// ===========================================================================
// End-to-end: 4-node diamond plan with heterogeneous shapes (1K, 4K, 8K,
// 16K elements). After MemoryPlanner::plan:
//   - 5 allocations present (one per tensor),
//   - every offset is a multiple of 64,
//   - no pair of tensors with overlapping liveness has overlapping bytes,
//   - the high-water mark (max(offset + byte_size)) fits in the pool.
// ===========================================================================
TEST(MemoryPlanner_end_to_end, DiamondPlanGetsNonOverlappingAllocations) {
    ExecutionPlan p;
    p.name = "memory_planner_diamond";

    // Mix of sizes so the FFD sort has a non-trivial byte_size tiebreak.
    p.tensors = {
        MakeTensor(0, "x", {1024}),
        MakeTensor(1, "a", {4096}),
        MakeTensor(2, "b", {2048}),
        MakeTensor(3, "c", {8192}),
        MakeTensor(4, "y", {16384}),
    };
    p.nodes = {
        MakeNode(0, {0},    {1},    {}),
        MakeNode(1, {1},    {2},    {0}),
        MakeNode(2, {1},    {3},    {0}),
        MakeNode(3, {2, 3}, {4},    {1, 2}),
    };
    p.inputs  = {{0, "x"}};
    p.outputs = {{4, "y"}};

    Result<void> r = MemoryPlanner::plan(p, kTotal, kAlign);
    ASSERT_TRUE(r.ok()) << r.error().message;

    // One AllocationDefinition per tensor.
    ASSERT_EQ(p.allocations.size(), p.tensors.size());

    // Every offset is alignment-honouring and the row mirrors its tensor.
    for (const auto& a : p.allocations) {
        EXPECT_EQ(a.alignment, kAlign);
        EXPECT_EQ(a.byte_offset % kAlign, 0u);
        EXPECT_EQ(a.memory_space, MemorySpace::Host);
        // Sizes match numel * 4 (Float32).
        const TensorDefinition* td = nullptr;
        for (const auto& t : p.tensors) {
            if (t.id == a.tensor_id) { td = &t; break; }
        }
        ASSERT_NE(td, nullptr) << "allocation references unknown tensor "
                                << a.tensor_id;
        uint64_t expected_bytes = 1;
        for (int64_t d : td->shape) expected_bytes *= static_cast<uint64_t>(d);
        expected_bytes *= 4u;
        EXPECT_EQ(a.byte_size, expected_bytes);
    }

    // External inputs and outputs are caller-owned and excluded from the
    // reuse pool. Their rows carry offset 0; internal tensors are the only
    // rows whose offsets can alias.
    const std::unordered_set<TensorId> external = {0, 4};
    for (const auto& a : p.allocations) {
        if (external.contains(a.tensor_id)) {
            EXPECT_EQ(a.byte_offset, 0u);
        }
    }

    // Re-derive live ranges and assert no two overlapping-lifetime
    // internal tensors share any byte in their [offset, offset+byte_size) range.
    const std::vector<LiveRange> ranges = LivenessAnalyzer::analyze(p);
    std::unordered_map<TensorId, LiveRange> range_of;
    for (const auto& r : ranges) range_of.emplace(r.id, r);
    for (std::size_t i = 0; i < p.tensors.size(); ++i) {
        for (std::size_t j = i + 1; j < p.tensors.size(); ++j) {
            const auto* ai = FindAlloc(p, p.tensors[i].id);
            const auto* aj = FindAlloc(p, p.tensors[j].id);
            ASSERT_NE(ai, nullptr);
            ASSERT_NE(aj, nullptr);
            const auto ri_it = range_of.find(p.tensors[i].id);
            const auto rj_it = range_of.find(p.tensors[j].id);
            ASSERT_NE(ri_it, range_of.end());
            ASSERT_NE(rj_it, range_of.end());
            if (external.contains(p.tensors[i].id) ||
                external.contains(p.tensors[j].id)) {
                continue;
            }
            const LiveRange& ri = ri_it->second;
            const LiveRange& rj = rj_it->second;
            const bool life_overlap = (ri.first_use <= rj.last_use)
                                    && (rj.first_use <= ri.last_use);
            if (!life_overlap) continue;
            // Spatial overlap check.
            const bool bytes_overlap =
                ai->byte_offset < aj->byte_offset + aj->byte_size
             && aj->byte_offset < ai->byte_offset + ai->byte_size;
            EXPECT_FALSE(bytes_overlap)
                << "tensors " << p.tensors[i].id << " and "
                << p.tensors[j].id << " have overlapping lifetimes and "
                << "overlapping byte ranges ["
                << ai->byte_offset << ", " << ai->byte_offset + ai->byte_size
                << ") and ["
                << aj->byte_offset << ", " << aj->byte_offset + aj->byte_size
                << ")";
        }
    }

    // High-water mark fits inside the pool.
    uint64_t hwm = 0;
    for (const auto& a : p.allocations) {
        if (external.contains(a.tensor_id)) {
            continue;
        }
        hwm = std::max(hwm, a.byte_offset + a.byte_size);
    }
    EXPECT_LE(hwm, kTotal);
}


TEST(MemoryPlanner_alignment, ZeroAlignmentIsRejected) {
    ExecutionPlan plan;
    plan.name = "zero_alignment";
    plan.tensors = {MakeTensor(0, "x", {4})};

    Result<void> result = MemoryPlanner::plan(plan, 1024, 0);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, -41);
    EXPECT_TRUE(plan.allocations.empty());
}
