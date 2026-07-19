#pragma once

// ForgeRT C++ ABI — artifact descriptor (spec §10).
// A compiled kernel artifact: target, entry point, raw binary blob, argument
// list, shape guards (specialization constraints), and workspace hint. The
// scheduler consumes ArtifactDescriptors to materialize launch calls.
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
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

struct ArtifactDescriptor {
    TargetKind target = TargetKind::CPU;
    std::string entry_point;        // kernel symbol
    std::vector<std::byte> binary;  // opaque kernel artifact (HLS bitstream, triton src, etc.)
    std::vector<ArgumentDescriptor> arguments;
    std::vector<ShapeGuard> guards;
    uint64_t workspace_bytes = 0;   // scratch the kernel needs
};

}  // namespace forge_rt
