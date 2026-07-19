// ForgeRT — GoogleTest suite for the topological scheduler (spec §10).
// Four test cases:
//   1. ExecutionPlan_json_roundtrip — every field of a populated
//      ExecutionPlan survives to_json → from_json verbatim.
//   2. Scheduler_linear_plan — a 2-node chain. Calls land in topo
//      order (the only legal order).
//   3. Scheduler_diamond_plan — 0 → {1, 2} → 3. Sibling order is
//      unconstrained, but the diamond invariants hold.
//   4. Scheduler_cycle_detected — cyclic plans return an Error and
//      never reach the provider's launch.
// We drive the scheduler via MockDeviceProvider (mock_provider.hpp) so
// this file does not depend on Task 11's real TritonProvider.
#include "forge_rt/execution_plan.hpp"
#include "forge_rt/scheduler.hpp"

#include "mock_provider.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <vector>

using forge_rt::AllocationDefinition;
using forge_rt::Buffer;
using forge_rt::DataType;
using forge_rt::ExecutionPlan;
using forge_rt::ExecutableNode;
using forge_rt::LaunchParameters;
using forge_rt::MemorySpace;
using forge_rt::MockDeviceProvider;
using forge_rt::NodeId;
using forge_rt::Scheduler;
using forge_rt::TensorDefinition;
using forge_rt::TensorId;

// ---------------------------------------------------------------------------
// Small builders — keep the test bodies focused on assertions.
// ---------------------------------------------------------------------------
namespace {

// Default-constructed vector<Buffer> for tests that don't care about
// real allocation. Buffer is move-only so we can't use brace-init
// with multiple copies; default-construct N entries, then resize.
template <std::size_t N>
std::vector<Buffer> MakeBuffers() {
    std::vector<Buffer> v;
    v.resize(N);
    return v;
}

// Default-empty LaunchParameters for tests that don't care about
// grid/block (which is most of them — we're testing order, not
// numerics).
LaunchParameters DefaultLaunch() { return LaunchParameters{}; }

// Build a tensor definition with all fields populated. Used by every
// test in this file.
TensorDefinition MakeTensor(TensorId id, const std::string& name,
                            std::vector<int64_t> shape) {
    TensorDefinition t;
    t.id   = id;
    t.name = name;
    t.dtype = DataType::Float32;
    t.shape = std::move(shape);
    t.memory_space = MemorySpace::Host;
    return t;
}

AllocationDefinition MakeAlloc(TensorId id, std::size_t bytes) {
    AllocationDefinition a;
    a.tensor_id = id;
    a.byte_offset = 0;
    a.byte_size = bytes;
    a.alignment = 64;
    a.memory_space = MemorySpace::Host;
    return a;
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
    n.launch = DefaultLaunch();
    return n;
}

class OffsetRecordingProvider final : public MockDeviceProvider {
public:
    const std::vector<std::string>& offset_log() const noexcept {
        return offset_log_;
    }

    forge_rt::Result<forge_rt::Event> launch(
        const forge_rt::ArtifactDescriptor& artifact,
        std::span<const Buffer> inputs,
        std::span<Buffer> outputs,
        std::span<const forge_rt::Event> deps) override {
        std::string entry = artifact.entry_point + " inputs=";
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            if (index > 0) entry += ",";
            entry += std::to_string(inputs[index].spec().descriptor.byte_offset);
        }
        entry += " outputs=";
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            if (index > 0) entry += ",";
            entry += std::to_string(outputs[index].spec().descriptor.byte_offset);
        }
        offset_log_.push_back(std::move(entry));
        return MockDeviceProvider::launch(artifact, inputs, outputs, deps);
    }

private:
    std::vector<std::string> offset_log_;
};

}  // namespace

// ===========================================================================
// 1. ExecutionPlan JSON round-trip — every field survives serialize/parse.
// ===========================================================================
TEST(ExecutionPlan_json_roundtrip, PlanSurvivesFieldByField) {
    ExecutionPlan p;
    p.name = "fused_mlp_phase1";

    // 4 tensors — two externals (x, y), two intermediates.
    p.tensors = {
        MakeTensor(1, "x",  {2, 512, 4096}),
        MakeTensor(2, "w",  {2, 4096, 11008}),
        MakeTensor(3, "a",  {2, 512, 11008}),
        MakeTensor(4, "y",  {2, 512, 11008}),
    };

    // 2 allocations — only x and y. Intermediates get default-allocated
    // by the scheduler at run time.
    p.allocations = {
        MakeAlloc(1, 2 * 512 * 4096 * 4),
        MakeAlloc(4, 2 * 512 * 11008 * 4),
    };

    // 3 nodes: matmul, bias+gelu, copy-out.
    p.nodes = {
        MakeNode(0, {1, 2}, {3}, {}),
        MakeNode(1, {3},    {3}, {0}),  // in-place bias+gelu on tensor 3
        MakeNode(2, {3},    {4}, {1}),
    };
    // Tweak node 0's launch to a recognisable value so we can prove
    // the LaunchParameters sub-struct was preserved.
    p.nodes[0].launch.grid = {512, 2, 1};
    p.nodes[0].launch.block = {64, 1, 1};
    p.nodes[0].launch.shared_memory_bytes = 4096;

    p.inputs = {{1, "x"}};
    p.outputs = {{4, "y"}};

    // Serialize.
    nlohmann::json j;
    forge_rt::to_json(j, p);

    // Re-parse.
    ExecutionPlan q;
    forge_rt::from_json(j, q);

    // Structural matches.
    EXPECT_EQ(q.name, p.name);
    EXPECT_EQ(q.tensors.size(),     p.tensors.size());
    EXPECT_EQ(q.allocations.size(), p.allocations.size());
    EXPECT_EQ(q.nodes.size(),       p.nodes.size());
    EXPECT_EQ(q.inputs.size(),      p.inputs.size());
    EXPECT_EQ(q.outputs.size(),     p.outputs.size());

    // Field-by-field equality.
    for (std::size_t i = 0; i < p.tensors.size(); ++i) {
        EXPECT_EQ(q.tensors[i].id,            p.tensors[i].id);
        EXPECT_EQ(q.tensors[i].name,          p.tensors[i].name);
        EXPECT_EQ(q.tensors[i].dtype,         p.tensors[i].dtype);
        EXPECT_EQ(q.tensors[i].shape,         p.tensors[i].shape);
        EXPECT_EQ(q.tensors[i].memory_space,  p.tensors[i].memory_space);
    }
    for (std::size_t i = 0; i < p.allocations.size(); ++i) {
        EXPECT_EQ(q.allocations[i].tensor_id,    p.allocations[i].tensor_id);
        EXPECT_EQ(q.allocations[i].byte_offset,  p.allocations[i].byte_offset);
        EXPECT_EQ(q.allocations[i].byte_size,    p.allocations[i].byte_size);
        EXPECT_EQ(q.allocations[i].alignment,    p.allocations[i].alignment);
        EXPECT_EQ(q.allocations[i].memory_space, p.allocations[i].memory_space);
    }
    for (std::size_t i = 0; i < p.nodes.size(); ++i) {
        EXPECT_EQ(q.nodes[i].id,           p.nodes[i].id);
        EXPECT_EQ(q.nodes[i].device,       p.nodes[i].device);
        EXPECT_EQ(q.nodes[i].artifact,     p.nodes[i].artifact);
        EXPECT_EQ(q.nodes[i].inputs,       p.nodes[i].inputs);
        EXPECT_EQ(q.nodes[i].outputs,      p.nodes[i].outputs);
        EXPECT_EQ(q.nodes[i].dependencies, p.nodes[i].dependencies);
        EXPECT_EQ(q.nodes[i].launch.grid,                p.nodes[i].launch.grid);
        EXPECT_EQ(q.nodes[i].launch.block,               p.nodes[i].launch.block);
        EXPECT_EQ(q.nodes[i].launch.shared_memory_bytes, p.nodes[i].launch.shared_memory_bytes);
    }
    for (std::size_t i = 0; i < p.inputs.size(); ++i) {
        EXPECT_EQ(q.inputs[i].tensor_id,     p.inputs[i].tensor_id);
        EXPECT_EQ(q.inputs[i].external_name, p.inputs[i].external_name);
    }
    for (std::size_t i = 0; i < p.outputs.size(); ++i) {
        EXPECT_EQ(q.outputs[i].tensor_id,     p.outputs[i].tensor_id);
        EXPECT_EQ(q.outputs[i].external_name, p.outputs[i].external_name);
    }
}

// ===========================================================================
// 2. Linear plan — every entry point fires in the only legal order.
// ===========================================================================
TEST(Scheduler_linear, TwoNodeChainFiresInTopologicalOrder) {
    // Tensor 0 = external x, tensor 1 = intermediate, tensor 2 = external y.
    // node 0 reads 0 and writes 1; node 1 reads 1 and writes 2.
    ExecutionPlan p;
    p.name = "linear";
    p.tensors = {
        MakeTensor(0, "x",   {4}),
        MakeTensor(1, "mid", {4}),
        MakeTensor(2, "y",   {4}),
    };
    p.allocations = {MakeAlloc(0, 16), MakeAlloc(2, 16)};
    p.nodes = {
        MakeNode(0, {0}, {1}, {}),
        MakeNode(1, {1}, {2}, {0}),
    };
    p.inputs  = {{0, "x"}};
    p.outputs = {{2, "y"}};

    auto mock = std::make_shared<MockDeviceProvider>();
    Scheduler s({mock});

    std::vector<Buffer> inputs  = MakeBuffers<1>();
    std::vector<Buffer> outputs = MakeBuffers<1>();

    auto r = s.execute(p, inputs, outputs);
    ASSERT_TRUE(r.ok()) << r.error().message;

    const auto& log = mock->call_log();
    ASSERT_EQ(log.size(), 2u);
    EXPECT_EQ(log[0], "launch node 0");
    EXPECT_EQ(log[1], "launch node 1");
}

// ===========================================================================
// 3. Diamond plan — 0 → {1, 2} → 3. Sibling order is not asserted; the
//    diamond invariants are (0 first, 3 last, mid-nodes in the middle).
// ===========================================================================
TEST(Scheduler_diamond, TopologicalInvariantsOnDiamondDAG) {
    // 5 tensors: x (input), a (out of 0), b (out of 1), c (out of 2), y (out of 3).
    ExecutionPlan p;
    p.name = "diamond";
    p.tensors = {
        MakeTensor(0, "x", {4}),
        MakeTensor(1, "a", {4}),
        MakeTensor(2, "b", {4}),
        MakeTensor(3, "c", {4}),
        MakeTensor(4, "y", {4}),
    };
    p.allocations = {MakeAlloc(0, 16), MakeAlloc(4, 16)};
    p.nodes = {
        MakeNode(0, {0},       {1},    {}),
        MakeNode(1, {1},       {2},    {0}),
        MakeNode(2, {1},       {3},    {0}),
        MakeNode(3, {2, 3},    {4},    {1, 2}),
    };
    p.inputs  = {{0, "x"}};
    p.outputs = {{4, "y"}};

    auto mock = std::make_shared<MockDeviceProvider>();
    Scheduler s({mock});

    std::vector<Buffer> inputs  = MakeBuffers<1>();
    std::vector<Buffer> outputs = MakeBuffers<1>();

    auto r = s.execute(p, inputs, outputs);
    ASSERT_TRUE(r.ok()) << r.error().message;

    const auto& log = mock->call_log();
    ASSERT_EQ(log.size(), 4u);

    // Convert "launch node N" entries to integers.
    auto node_id = [](const std::string& s) -> int {
        const std::string prefix = "launch node ";
        EXPECT_EQ(s.rfind(prefix, 0), 0u) << "log entry doesn't match prefix: " << s;
        return std::stoi(s.substr(prefix.size()));
    };
    std::vector<int> order;
    for (const auto& entry : log) {
        order.push_back(node_id(entry));
    }

    // Diamond invariants.
    auto pos_of = [&](int nid) {
        auto it = std::find(order.begin(), order.end(), nid);
        EXPECT_NE(it, order.end()) << "node " << nid << " not in log";
        return static_cast<int>(it - order.begin());
    };
    EXPECT_LT(pos_of(0), pos_of(1));
    EXPECT_LT(pos_of(0), pos_of(2));
    EXPECT_LT(pos_of(1), pos_of(3));
    EXPECT_LT(pos_of(2), pos_of(3));
}

// ===========================================================================
// 4. Cycle detected — plans with cycles return Error without launching.
// ===========================================================================
TEST(Scheduler_cycle, ReturnsErrorAndSkipsLaunchOnCyclicPlan) {
    // 2 nodes depending on each other.
    ExecutionPlan p;
    p.name = "cycle";
    p.tensors = {
        MakeTensor(0, "x", {4}),
        MakeTensor(1, "y", {4}),
    };
    // (No allocations needed — caller passes them.)
    p.nodes = {
        MakeNode(0, {0}, {1}, {1}),  // depends on 1
        MakeNode(1, {1}, {0}, {0}),  // depends on 0
    };
    p.inputs  = {{0, "x"}};
    p.outputs = {{1, "y"}};

    auto mock = std::make_shared<MockDeviceProvider>();
    Scheduler s({mock});

    std::vector<Buffer> inputs  = MakeBuffers<1>();
    std::vector<Buffer> outputs = MakeBuffers<1>();

    auto r = s.execute(p, inputs, outputs);
    EXPECT_FALSE(r.ok());

    // The Error message should let callers recognise the failure mode.
    // We accept either "cycle" or "topological" — both terms are used
    // in the scheduler's diagnostic.
    const std::string& msg = r.error().message;
    const bool mentions_cycle  = msg.find("cycle")        != std::string::npos;
    const bool mentions_topo   = msg.find("topological")  != std::string::npos;
    EXPECT_TRUE(mentions_cycle || mentions_topo)
        << "expected diagnostic to mention cycle or topological, got: " << msg;

    // No provider launches should have happened.
    EXPECT_TRUE(mock->call_log().empty());
}

TEST(Scheduler, UsesMemoryPlannerAllocations) {
    ExecutionPlan plan;
    plan.name = "fused_mlp_offsets";
    plan.tensors = {
        MakeTensor(0, "x", {4}),
        MakeTensor(1, "weight", {8}),
        MakeTensor(2, "bias", {2}),
        MakeTensor(3, "matmul", {8}),
        MakeTensor(4, "gelu", {8}),
        MakeTensor(5, "y", {8}),
    };
    plan.allocations = {
        MakeAlloc(0, 16),
        MakeAlloc(1, 32),
        MakeAlloc(2, 8),
        MakeAlloc(5, 32),
    };
    plan.nodes = {
        MakeNode(0, {0, 1, 2}, {3}, {}),
        MakeNode(1, {3}, {4}, {0}),
        MakeNode(2, {4}, {5}, {1}),
    };
    plan.inputs = {{0, "x"}, {1, "weight"}, {2, "bias"}};
    plan.outputs = {{5, "y"}};

    auto provider = std::make_shared<OffsetRecordingProvider>();
    Scheduler scheduler({provider});
    std::vector<Buffer> inputs = MakeBuffers<3>();
    std::vector<Buffer> outputs = MakeBuffers<1>();

    auto result = scheduler.execute(plan, inputs, outputs);

    ASSERT_TRUE(result.ok()) << result.error().message;
    const std::vector<std::string> expected = {
        "node 0 inputs=128,64,192 outputs=0",
        "node 1 inputs=0 outputs=64",
        "node 2 inputs=64 outputs=0",
    };
    EXPECT_EQ(provider->offset_log(), expected);
    RecordProperty("offset_log", "128,64,192->0;0->64;64->0");
}
