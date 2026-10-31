#pragma once

// ForgeRT C++ ABI — artifact descriptor (spec §10).
// A compiled kernel artifact: target, entry point, raw binary blob, argument
// list, shape guards (specialization constraints), workspace hint, and launch
// geometry. The scheduler consumes ArtifactDescriptors to materialize launch
// calls.
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "abi_types.hpp"

namespace forge_rt {

enum class ShapeGuardKind : uint8_t {
    Equal       = 0,
    Divisible   = 1,
    GreaterEqual = 2,
    LessEqual    = 3,
};

struct ShapeGuard {
    std::vector<int64_t> indices;  // which input dims the constraint applies to
    ShapeGuardKind kind = ShapeGuardKind::Equal;
    int64_t value = 0;             // comparison target
};

struct ArgumentDescriptor {
    std::string name;
    DataType dtype = DataType::Float32;
    bool is_input = true;
    std::vector<int64_t> shape;
};

// Kernel launch geometry. Copied from the executable node when the
// scheduler resolves an artifact, so a provider can see grid/block/shared
// memory without a second plan lookup.
struct LaunchParameters {
    std::vector<int64_t> grid;
    std::vector<int64_t> block;
    uint64_t shared_memory_bytes = 0;
};

struct ArtifactDescriptor {
    TargetKind target = TargetKind::CPU;
    std::string entry_point;        // kernel symbol
    std::vector<std::byte> binary;  // opaque kernel artifact (HLS bitstream, triton src, etc.)
    std::vector<ArgumentDescriptor> arguments;
    std::vector<ShapeGuard> guards;
    uint64_t workspace_bytes = 0;   // scratch the kernel needs
    LaunchParameters launch;
};

// ADL hooks for nlohmann_json. Binary bytes serialize as an integer
// array so the plan stays a single JSON document. The functions live in
// runtime/lib/artifact_descriptor.cpp.
void to_json(nlohmann::json& j, const ShapeGuard& g);
void from_json(const nlohmann::json& j, ShapeGuard& g);
void to_json(nlohmann::json& j, const ArgumentDescriptor& a);
void from_json(const nlohmann::json& j, ArgumentDescriptor& a);
void to_json(nlohmann::json& j, const LaunchParameters& l);
void from_json(const nlohmann::json& j, LaunchParameters& l);
void to_json(nlohmann::json& j, const ArtifactDescriptor& a);
void from_json(const nlohmann::json& j, ArtifactDescriptor& a);

}  // namespace forge_rt
