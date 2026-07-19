// ForgeRT — focused ExecutionPlan allocation JSON contract tests (Task 32).
#include "../include/forge_rt/execution_plan.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using forge_rt::AllocationDefinition;
using forge_rt::ExecutionPlan;
using forge_rt::MemorySpace;

TEST(ExecutionPlanJsonRoundtrip, AllocationFields) {
    ExecutionPlan plan;
    plan.name = "allocation_roundtrip";
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
