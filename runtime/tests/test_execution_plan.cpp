// ForgeRT — focused ExecutionPlan allocation JSON contract tests (Task 32).
#include "../include/forge_rt/execution_plan.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using forge_rt::AllocationDefinition;
using forge_rt::ArgumentDescriptor;
using forge_rt::ArtifactDescriptor;
using forge_rt::DataType;
using forge_rt::ExecutionPlan;
using forge_rt::LaunchParameters;
using forge_rt::MemorySpace;
using forge_rt::ShapeGuard;
using forge_rt::ShapeGuardKind;
using forge_rt::TargetKind;

TEST(ExecutionPlanJsonRoundtrip, AllocationFields) {
    ExecutionPlan plan;
    plan.name = "allocation_roundtrip";
    ArtifactDescriptor artifact;
    artifact.target = TargetKind::Triton;
    artifact.entry_point = "fused";
    artifact.binary = {std::byte{0x01}, std::byte{0x02}};
    artifact.arguments = {ArgumentDescriptor{"x", DataType::Float32, true, {2, 4}}};
    artifact.guards = {ShapeGuard{{1}, ShapeGuardKind::Equal, 4}};
    artifact.workspace_bytes = 128;
    artifact.launch.grid = {2, 1, 1};
    artifact.launch.block = {32, 1, 1};
    artifact.launch.shared_memory_bytes = 64;
    plan.artifacts = {artifact};
    plan.allocations = {
        AllocationDefinition{
            .tensor_id = 3,
            .byte_offset = 128,
            .byte_size = 4096,
            .alignment = 64,
            .memory_space = MemorySpace::Host,
        },
        AllocationDefinition{
            .tensor_id = 9,
            .byte_offset = 8192,
            .byte_size = 16384,
            .alignment = 256,
            .memory_space = MemorySpace::Device1,
        },
    };

    nlohmann::json json;
    forge_rt::to_json(json, plan);
    ExecutionPlan restored;
    forge_rt::from_json(json, restored);

    EXPECT_EQ(restored.version, plan.version);
    ASSERT_EQ(restored.artifacts.size(), 1u);
    EXPECT_EQ(restored.artifacts[0].target, artifact.target);
    EXPECT_EQ(restored.artifacts[0].entry_point, artifact.entry_point);
    EXPECT_EQ(restored.artifacts[0].binary, artifact.binary);
    ASSERT_EQ(restored.artifacts[0].arguments.size(), 1u);
    EXPECT_EQ(restored.artifacts[0].arguments[0].name, "x");
    EXPECT_EQ(restored.artifacts[0].guards[0].value, 4);
    EXPECT_EQ(restored.artifacts[0].workspace_bytes, 128u);
    EXPECT_EQ(restored.artifacts[0].launch.grid, artifact.launch.grid);
    ASSERT_EQ(restored.allocations.size(), plan.allocations.size());
    for (std::size_t index = 0; index < plan.allocations.size(); ++index) {
        const AllocationDefinition& expected = plan.allocations[index];
        const AllocationDefinition& actual = restored.allocations[index];
        EXPECT_EQ(actual.tensor_id, expected.tensor_id);
        EXPECT_EQ(actual.byte_offset, expected.byte_offset);
        EXPECT_EQ(actual.byte_size, expected.byte_size);
        EXPECT_EQ(actual.alignment, expected.alignment);
        EXPECT_EQ(actual.memory_space, expected.memory_space);
    }
}
