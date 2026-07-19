// ForgeRT C++ ABI — Buffer (spec §10).
#include "forge_rt/buffer.hpp"

#include <cstdlib>

namespace forge_rt {

Buffer::~Buffer() {
    if (owns_ && data_ != nullptr) {
        // Phase 1: CPU provider uses posix_memalign/malloc; the default
        // deleter is std::free. Phase 5 (Triton/CUDA provider) wraps the
        // raw pointer in a cuda-allocated Buffer and is responsible for
        // setting owns_=false — the CUDA provider's release path will
        // call cudaFree. Spec §20 forbids replacing the CUDA allocator.
        std::free(data_);
        data_ = nullptr;
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), spec_(std::move(other.spec_)), owns_(other.owns_) {
    other.data_ = nullptr;
    other.spec_ = BufferSpec{};
    other.owns_ = false;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (owns_ && data_ != nullptr) {
            std::free(data_);
        }
        data_ = other.data_;
        spec_ = std::move(other.spec_);
        owns_ = other.owns_;
        other.data_ = nullptr;
        other.spec_ = BufferSpec{};
        other.owns_ = false;
    }
    return *this;
}

void* Buffer::release() noexcept {
    void* p = data_;
    data_ = nullptr;
    spec_ = BufferSpec{};
    owns_ = false;
    return p;
}

}  // namespace forge_rt
