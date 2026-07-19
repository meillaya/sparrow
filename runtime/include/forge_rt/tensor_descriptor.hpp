#pragma once

// ForgeRT C++ ABI — tensor descriptor (spec §10).
// Describes the *view* into a backing buffer: dtype, shape, element strides,
// memory space, and offset into the owning buffer. Storage stays in the
// owning Buffer; this struct is metadata only.
#include <cstdint>
#include <vector>
#include "abi_types.hpp"

namespace forge_rt {

struct TensorDescriptor {
    DataType dtype = DataType::Float32;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;  // in elements (not bytes)
    MemorySpace memory_space = MemorySpace::Host;
    uint64_t byte_offset = 0;      // into the owning buffer
};

// Helpers (implemented in lib/tensor_descriptor.cpp).
std::size_t numel(const TensorDescriptor& td);
std::size_t byte_size(const TensorDescriptor& td);
bool is_contiguous(const TensorDescriptor& td);
bool operator==(const TensorDescriptor& a, const TensorDescriptor& b);

}  // namespace forge_rt
