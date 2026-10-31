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
using forge_rt::ArtifactDescriptor;
using forge_rt::Buffer;
using forge_rt::BufferSpec;
using forge_rt::DataType;
using forge_rt::DeviceDefinition;
using forge_rt::Error;
using forge_rt::Event;
using forge_rt::ExecutionPlan;
using forge_rt::ExecutableNode;
using forge_rt::LaunchParameters;
using forge_rt::MemorySpace;
using forge_rt::MockDeviceProvider;
using forge_rt::NodeId;
using forge_rt::Result;
using forge_rt::Scheduler;
using forge_rt::TargetKind;
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

std::vector<ArtifactDescriptor> MakeArtifacts(std::size_t count) {
    std::vector<ArtifactDescriptor> artifacts;
    artifacts.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        ArtifactDescriptor artifact;
        artifact.target = TargetKind::CPU;
        artifact.entry_point = "node " + std::to_string(index);
        artifacts.push_back(std::move(artifact));
    }
    return artifacts;
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

std::vector<std::string> launch_entries(const std::vector<std::string>& log) {
    std::vector<std::string> launches;
    for (const std::string& entry : log) {
        if (entry.rfind("launch ", 0) == 0) {
            launches.push_back(entry);
        }
    }
    return launches;
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
    p.version = 2;
    p.name = "fused_mlp_phase1";
    p.artifacts = MakeArtifacts(8);

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
    EXPECT_EQ(q.version, p.version);
    EXPECT_EQ(q.artifacts.size(), p.artifacts.size());
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
    p.version = 2;
    p.name = "linear";
    p.artifacts = MakeArtifacts(8);
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

    const std::vector<std::string> launches = launch_entries(mock->call_log());
    ASSERT_EQ(launches.size(), 2u);
    EXPECT_EQ(launches[0], "launch node 0");
    EXPECT_EQ(launches[1], "launch node 1");
}

// ===========================================================================
// 3. Diamond plan — 0 → {1, 2} → 3. Sibling order is not asserted; the
//    diamond invariants are (0 first, 3 last, mid-nodes in the middle).
// ===========================================================================
TEST(Scheduler_diamond, TopologicalInvariantsOnDiamondDAG) {
    // 5 tensors: x (input), a (out of 0), b (out of 1), c (out of 2), y (out of 3).
    ExecutionPlan p;
    p.version = 2;
    p.name = "diamond";
    p.artifacts = MakeArtifacts(8);
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

    const std::vector<std::string> launches = launch_entries(mock->call_log());
    ASSERT_EQ(launches.size(), 4u);

    // Convert "launch node N" entries to integers.
    auto node_id = [](const std::string& s) -> int {
        const std::string prefix = "launch node ";
        EXPECT_EQ(s.rfind(prefix, 0), 0u) << "log entry doesn't match prefix: " << s;
        return std::stoi(s.substr(prefix.size()));
    };
    std::vector<int> order;
    for (const auto& entry : launches) {
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
    p.version = 2;
    p.name = "cycle";
    p.artifacts = MakeArtifacts(8);
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
    plan.version = 2;  // provider-index compatibility path
    plan.name = "fused_mlp_offsets";
    plan.artifacts = MakeArtifacts(8);
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
        "node 0 inputs=0,0,0 outputs=0",
        "node 1 inputs=0 outputs=64",
        "node 2 inputs=64 outputs=0",
    };
    EXPECT_EQ(provider->offset_log(), expected);
    RecordProperty("offset_log", "0,0,0->0;0->64;64->0");
}

// ===========================================================================
// 6. Sparse node ids are legal. Tables are keyed by dense tensor slots and
//    node ids only appear in maps, so a large id cannot index out of bounds.
// ===========================================================================
TEST(Scheduler_sparse_ids, LargeNodeIdsRunInDependencyOrder) {
    ExecutionPlan plan;
    plan.version = 2;  // provider-index compatibility path
    plan.name = "sparse_ids";
    plan.artifacts = MakeArtifacts(21);
    plan.tensors = {
        MakeTensor(0, "x", {4}),
        MakeTensor(1, "mid", {4}),
        MakeTensor(2, "y", {4}),
    };
    plan.nodes = {
        MakeNode(10, {0}, {1}, {}),
        MakeNode(20, {1}, {2}, {10}),
    };
    plan.inputs = {{0, "x"}};
    plan.outputs = {{2, "y"}};

    auto provider = std::make_shared<MockDeviceProvider>();
    Scheduler scheduler({provider});
    std::vector<Buffer> inputs = MakeBuffers<1>();
    std::vector<Buffer> outputs = MakeBuffers<1>();

    auto result = scheduler.execute(plan, inputs, outputs);
    ASSERT_TRUE(result.ok()) << result.error().message;

    const std::vector<std::string> launches = launch_entries(provider->call_log());
    ASSERT_EQ(launches.size(), 2u);
    EXPECT_EQ(launches[0], "launch node 10");
    EXPECT_EQ(launches[1], "launch node 20");
}

// ===========================================================================
// 7. An artifact reference must resolve. A plan without an artifact table
//    fails before any allocation or launch.
// ===========================================================================
TEST(Scheduler_artifacts, MissingArtifactIsRejected) {
    ExecutionPlan plan;
    plan.version = 2;  // provider-index compatibility path
    plan.name = "missing_artifact";
    plan.tensors = {MakeTensor(0, "x", {4}), MakeTensor(1, "y", {4})};
    plan.nodes = {MakeNode(0, {0}, {1}, {})};
    plan.inputs = {{0, "x"}};
    plan.outputs = {{1, "y"}};

    auto provider = std::make_shared<MockDeviceProvider>();
    Scheduler scheduler({provider});
    std::vector<Buffer> inputs = MakeBuffers<1>();
    std::vector<Buffer> outputs = MakeBuffers<1>();

    auto result = scheduler.execute(plan, inputs, outputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, -18);
    EXPECT_TRUE(provider->call_log().empty());
}

// ===========================================================================
// 8. An undersized external output is rejected before the provider writes.
// ===========================================================================
TEST(Scheduler_capacity, UndersizedOutputIsRejected) {
    ExecutionPlan plan;
    plan.version = 2;  // provider-index compatibility path
    plan.name = "undersized";
    plan.artifacts = MakeArtifacts(2);
    plan.tensors = {MakeTensor(0, "x", {4}), MakeTensor(1, "y", {4})};
    plan.nodes = {MakeNode(0, {0}, {1}, {})};
    plan.inputs = {{0, "x"}};
    plan.outputs = {{1, "y"}};

    BufferSpec in_spec;
    in_spec.descriptor.dtype = DataType::Float32;
    in_spec.descriptor.shape = {4};
    in_spec.byte_size = 16;
    in_spec.alignment = 4;
    BufferSpec out_spec = in_spec;
    out_spec.byte_size = 4;

    std::vector<float> input_storage(4, 1.0f);
    std::vector<float> output_storage(4, 0.0f);
    std::vector<Buffer> inputs;
    inputs.emplace_back(input_storage.data(), in_spec, false);
    std::vector<Buffer> outputs;
    outputs.emplace_back(output_storage.data(), out_spec, false);

    auto provider = std::make_shared<MockDeviceProvider>();
    Scheduler scheduler({provider});
    auto result = scheduler.execute(plan, inputs, outputs);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, -27);
    EXPECT_TRUE(provider->call_log().empty());
}

// ===========================================================================
// 9. In-place nodes keep one buffer. The provider writes through the output
//    alias and the caller's storage still holds the bytes.
// ===========================================================================
class InPlaceWritingProvider final : public MockDeviceProvider {
public:
    Result<Event> launch(const ArtifactDescriptor& artifact,
                         std::span<const Buffer> inputs,
                         std::span<Buffer> outputs,
                         std::span<const Event> deps) override {
        (void)artifact;
        (void)inputs;
        (void)deps;
        if (!outputs.empty() && outputs[0].data() != nullptr) {
            static_cast<float*>(outputs[0].data())[0] = 7.0f;
        }
        Event event{999};
        event.mark_completed();
        return event;
    }
};

TEST(Scheduler_in_place, OutputAliasKeepsCallerStorage) {
    ExecutionPlan plan;
    plan.version = 2;  // provider-index compatibility path
    plan.name = "in_place";
    plan.artifacts = MakeArtifacts(1);
    plan.tensors = {MakeTensor(0, "x", {1})};
    plan.nodes = {MakeNode(0, {0}, {0}, {})};
    plan.inputs = {{0, "x"}};
    plan.outputs = {{0, "x"}};

    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = {1};
    spec.byte_size = 4;
    spec.alignment = 4;

    std::vector<float> storage = {1.0f};
    std::vector<Buffer> inputs;
    inputs.emplace_back(storage.data(), spec, false);
    std::vector<Buffer> outputs;
    outputs.emplace_back(storage.data(), spec, false);

    auto provider = std::make_shared<InPlaceWritingProvider>();
    Scheduler scheduler({provider});
    auto result = scheduler.execute(plan, inputs, outputs);

    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_FLOAT_EQ(storage[0], 7.0f);
    EXPECT_EQ(inputs[0].data(), storage.data());
    EXPECT_EQ(outputs[0].data(), storage.data());
}

// ===========================================================================
// 10. A failed launch leaves caller buffers valid and unmoved.
// ===========================================================================
class FailingProvider final : public MockDeviceProvider {
public:
    Result<Event> launch(const ArtifactDescriptor&, std::span<const Buffer>,
                         std::span<Buffer>, std::span<const Event>) override {
        return Error{"intentional launch failure", 999};
    }
};

TEST(Scheduler_failure, FailedLaunchPreservesCallerBuffers) {
    ExecutionPlan plan;
    plan.version = 2;  // provider-index compatibility path
    plan.name = "failing_launch";
    plan.artifacts = MakeArtifacts(2);
    plan.tensors = {MakeTensor(0, "x", {1}), MakeTensor(1, "y", {1})};
    plan.nodes = {MakeNode(0, {0}, {1}, {})};
    plan.inputs = {{0, "x"}};
    plan.outputs = {{1, "y"}};

    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = {1};
    spec.byte_size = 4;
    spec.alignment = 4;

    std::vector<float> input_storage = {1.0f};
    std::vector<float> output_storage = {0.0f};
    std::vector<Buffer> inputs;
    inputs.emplace_back(input_storage.data(), spec, false);
    std::vector<Buffer> outputs;
    outputs.emplace_back(output_storage.data(), spec, false);

    auto provider = std::make_shared<FailingProvider>();
    Scheduler scheduler({provider});
    auto result = scheduler.execute(plan, inputs, outputs);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, -30);
    EXPECT_EQ(inputs[0].data(), input_storage.data());
    EXPECT_EQ(outputs[0].data(), output_storage.data());
    EXPECT_EQ(inputs[0].spec().byte_size, 4u);
    EXPECT_EQ(outputs[0].spec().byte_size, 4u);
}

// ---------------------------------------------------------------------------
// Version 3 device table.
// ---------------------------------------------------------------------------

namespace {

ExecutionPlan MakeDevicePlan() {
    ExecutionPlan plan;
    plan.name = "device_table";
    plan.version = 3;
    plan.devices = {
        DeviceDefinition{0, "cpu", TargetKind::CPU, MemorySpace::Host},
        DeviceDefinition{1, "triton", TargetKind::Triton, MemorySpace::Host},
    };
    plan.artifacts = MakeArtifacts(2);
    plan.artifacts[1].target = TargetKind::Triton;
    plan.tensors = {MakeTensor(0, "x", {1}), MakeTensor(1, "y", {1}),
                    MakeTensor(2, "z", {1})};
    plan.nodes = {
        ExecutableNode{0, 0, 0, {0}, {1}, {}, DefaultLaunch()},
        ExecutableNode{1, 1, 1, {1}, {2}, {0}, DefaultLaunch()},
    };
    plan.inputs = {{0, "x"}};
    plan.outputs = {{2, "z"}};
    return plan;
}

std::vector<Buffer> OneBuffer(std::vector<float>& storage) {
    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = {1};
    spec.byte_size = 4;
    spec.alignment = 4;
    std::vector<Buffer> buffers;
    buffers.emplace_back(storage.data(), spec, false);
    return buffers;
}

}  // namespace

TEST(Scheduler_devices, DispatchesThroughDeviceTableTargets) {
    ExecutionPlan plan = MakeDevicePlan();
    auto cpu = std::make_shared<MockDeviceProvider>(MemorySpace::Host, TargetKind::CPU);
    auto triton = std::make_shared<MockDeviceProvider>(MemorySpace::Host, TargetKind::Triton);
    Scheduler scheduler({cpu, triton});

    std::vector<float> input_storage = {1.0F};
    std::vector<float> output_storage = {0.0F};
    auto inputs = OneBuffer(input_storage);
    auto outputs = OneBuffer(output_storage);
    auto result = scheduler.execute(plan, inputs, outputs);

    ASSERT_TRUE(result.ok()) << result.error().message;
    const std::vector<std::string> cpu_launches = launch_entries(cpu->call_log());
    const std::vector<std::string> triton_launches = launch_entries(triton->call_log());
    ASSERT_EQ(cpu_launches.size(), 1u);
    ASSERT_EQ(triton_launches.size(), 1u);
    EXPECT_EQ(cpu_launches[0], "launch node 0");
    EXPECT_EQ(triton_launches[0], "launch node 1");
}

TEST(Scheduler_devices, UnknownDeviceIsRejected) {
    ExecutionPlan plan = MakeDevicePlan();
    plan.nodes[0].device = 9;
    auto provider = std::make_shared<MockDeviceProvider>();
    Scheduler scheduler({provider});
    std::vector<float> input_storage = {1.0F};
    std::vector<float> output_storage = {0.0F};
    auto inputs = OneBuffer(input_storage);
    auto outputs = OneBuffer(output_storage);

    auto result = scheduler.execute(plan, inputs, outputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, -22);
    EXPECT_NE(result.error().message.find("unknown device"), std::string::npos);
}

TEST(Scheduler_devices, ArtifactTargetMustMatchDeviceTarget) {
    ExecutionPlan plan = MakeDevicePlan();
    plan.artifacts[0].target = TargetKind::Triton;
    auto provider = std::make_shared<MockDeviceProvider>();
    Scheduler scheduler({provider});
    std::vector<float> input_storage = {1.0F};
    std::vector<float> output_storage = {0.0F};
    auto inputs = OneBuffer(input_storage);
    auto outputs = OneBuffer(output_storage);

    auto result = scheduler.execute(plan, inputs, outputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, -22);
    EXPECT_NE(result.error().message.find("device"), std::string::npos);
}

TEST(Scheduler_devices, MissingProviderForDeviceTargetIsRejected) {
    ExecutionPlan plan = MakeDevicePlan();
    plan.devices[1].target = TargetKind::Astra;
    plan.artifacts[1].target = TargetKind::Astra;
    auto provider = std::make_shared<MockDeviceProvider>(MemorySpace::Host, TargetKind::CPU);
    Scheduler scheduler({provider});
    std::vector<float> input_storage = {1.0F};
    std::vector<float> output_storage = {0.0F};
    auto inputs = OneBuffer(input_storage);
    auto outputs = OneBuffer(output_storage);

    auto result = scheduler.execute(plan, inputs, outputs);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, -22);
    EXPECT_NE(result.error().message.find("provider"), std::string::npos);
}
