// ForgeRT C++ ABI — TensorDescriptor helpers (spec §10).
#include "forge_rt/tensor_descriptor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace forge_rt {

// numel = product(shape[i]); empty shape yields 1 (a scalar tensor).
std::size_t numel(const TensorDescriptor& td) {
    if (td.shape.empty()) return 1;
    std::size_t n = 1;
    for (int64_t d : td.shape) {
        if (d <= 0) return 0;  // symbolic / zero-sized rank
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

// byte_size = numel * kByteSize(dtype) + ceil-pad for Int4 packing.
// Int4 uses ceil(numel/2) since two elements pack into one byte.
std::size_t byte_size(const TensorDescriptor& td) {
    const std::size_t n = numel(td);
    if (td.dtype == DataType::Int4) {
        return (n + 1) / 2;
    }
    return n * kByteSize(td.dtype);
}

// Contiguity: strides are valid, descending, last stride is 1, and the
// expected row-major stride matches. A scalar (rank 0) is trivially
// contiguous. Empty strides are treated as not-contiguous (caller should
// have filled them in).
bool is_contiguous(const TensorDescriptor& td) {
    if (td.shape.empty()) return true;
    if (td.strides.size() != td.shape.size()) return false;
    int64_t expected = 1;
    for (std::size_t i = td.shape.size(); i-- > 0;) {
        const int64_t dim = td.shape[i];
        if (td.strides[i] != expected) return false;
        if (dim == 0) {
            expected = 0;
        } else {
            expected *= dim;
        }
    }
    return true;
}

bool operator==(const TensorDescriptor& a, const TensorDescriptor& b) {
    return a.dtype == b.dtype
        && a.memory_space == b.memory_space
        && a.byte_offset == b.byte_offset
        && a.shape == b.shape
        && a.strides == b.strides;
}

}  // namespace forge_rt
