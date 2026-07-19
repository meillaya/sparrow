// ForgeRT — GoogleTest suite for LivenessAnalyzer (spec §10, Phase 3).
// Two test cases, mirroring the layouts that drive the scheduler tests:
//   1. Linear_Plan — 3-node chain. first_use of the input is node 0;
//      first_use of the output is node 1; etc. The intermediate
//      tensor's live interval must span node 0 (written) through node 1
//      (consumed).
//   2. Diamond_Plan — node0's output is consumed by both 1 and 2, so
//      its interval must extend to node 3 (the join). All other
//      intervals are point-shaped (single-producer single-consumer).
#include "forge_rt/execution_plan.hpp"
#include "forge_rt/liveness.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using forge_rt::DataType;
using forge_rt::ExecutionPlan;
using forge_rt::ExecutableNode;
using forge_rt::LaunchParameters;
using forge_rt::LivenessAnalyzer;
using forge_rt::LiveRange;
using forge_rt::MemorySpace;
using forge_rt::NodeId;
using forge_rt::TensorDefinition;
using forge_rt::TensorId;

namespace {

// Builders — kept local so the test bodies read as assertions, not
// fixture scaffolding.
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

// Look up the LiveRange for `tid` in a vector indexed by plan.tensors
// position. We return a sentinel with id=tid and dummy nodes if not
// found, so callers can use EXPECT without a separate "found" branch.
const LiveRange& FindRange(const std::vector<LiveRange>& ranges, TensorId tid) {
    static const LiveRange kMissing{0, 0, 0};
    for (const auto& r : ranges) {
        if (r.id == tid) return r;
    }
    return kMissing;
}

}  // namespace

// ===========================================================================
// 1. Linear plan — input → node0 → node1 → output. The intermediate
//    tensor (id=1) is written at node 0 and read at node 1, so its
//    interval must be [0, 1]. Inputs are only read once and outputs
//    only written once, so their intervals collapse to a single node.
// ===========================================================================
TEST(LivenessAnalyzer_linear, ThreeNodeChainSpansWrittenAndReadNodes) {
    ExecutionPlan p;
    p.name = "linear";
    p.tensors = {
        MakeTensor(0, "x",   {4}),
        MakeTensor(1, "mid", {4}),
        MakeTensor(2, "y",   {4}),
    };
    p.nodes = {
        MakeNode(0, {0}, {1}, {}),
        MakeNode(1, {1}, {2}, {0}),
    };

    const auto ranges = LivenessAnalyzer::analyze(p);
    ASSERT_EQ(ranges.size(), 3u);

    // Tensor 0 is read only at node 0 → point interval at 0.
    const auto& r0 = FindRange(ranges, 0);
    EXPECT_EQ(r0.id,        0u);
    EXPECT_EQ(r0.first_use, 0u);
    EXPECT_EQ(r0.last_use,  0u);

    // Tensor 1 is written at node 0 and read at node 1 → [0, 1].
    const auto& r1 = FindRange(ranges, 1);
    EXPECT_EQ(r1.id,        1u);
    EXPECT_EQ(r1.first_use, 0u);
    EXPECT_EQ(r1.last_use,  1u);

    // Tensor 2 is written only at node 1 → point interval at 1.
    const auto& r2 = FindRange(ranges, 2);
    EXPECT_EQ(r2.id,        2u);
    EXPECT_EQ(r2.first_use, 1u);
    EXPECT_EQ(r2.last_use,  1u);
}

// ===========================================================================
// 2. Diamond plan — 0 → {1, 2} → 3. Node 0's output is consumed by both
//    branches and joins at node 3, so its interval must reach node 3
//    even though the producer is node 0.
// ===========================================================================
TEST(LivenessAnalyzer_diamond, FirstNodeOutputLivesUntilJoinNode) {
    ExecutionPlan p;
    p.name = "diamond";
    p.tensors = {
        MakeTensor(0, "x", {4}),
        MakeTensor(1, "a", {4}),
        MakeTensor(2, "b", {4}),
        MakeTensor(3, "c", {4}),
        MakeTensor(4, "y", {4}),
    };
    p.nodes = {
        MakeNode(0, {0},    {1},    {}),
        MakeNode(1, {1},    {2},    {0}),
        MakeNode(2, {1},    {3},    {0}),
        MakeNode(3, {2, 3}, {4},    {1, 2}),
    };

    const auto ranges = LivenessAnalyzer::analyze(p);
    ASSERT_EQ(ranges.size(), 5u);

    // Tensor 0 (input) is read only at node 0.
    EXPECT_EQ(FindRange(ranges, 0).first_use, 0u);
    EXPECT_EQ(FindRange(ranges, 0).last_use,  0u);

    // Tensor 1 (a) is written at node 0 and read at nodes 1 + 2 →
    // [0, 2]. This is the interval the buffer-reuse allocator uses to
    // decide whether `a` can share storage with sibling tensors.
    EXPECT_EQ(FindRange(ranges, 1).first_use, 0u);
    EXPECT_EQ(FindRange(ranges, 1).last_use,  2u);

    // Tensor 2 (b) is written at node 1 and consumed only at node 3.
    EXPECT_EQ(FindRange(ranges, 2).first_use, 1u);
    EXPECT_EQ(FindRange(ranges, 2).last_use,  3u);

    // Tensor 3 (c) is written at node 2 and consumed only at node 3.
    EXPECT_EQ(FindRange(ranges, 3).first_use, 2u);
    EXPECT_EQ(FindRange(ranges, 3).last_use,  3u);

    // Tensor 4 (y) is written only at node 3.
    EXPECT_EQ(FindRange(ranges, 4).first_use, 3u);
    EXPECT_EQ(FindRange(ranges, 4).last_use,  3u);
}
