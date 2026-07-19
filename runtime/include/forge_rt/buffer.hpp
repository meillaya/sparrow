#pragma once

// ForgeRT C++ ABI — Buffer (spec §10).
// Wraps a region of memory (host or device) plus a BufferSpec describing the
// tensor view. Move-only: transferring ownership is explicit, no accidental
// double-free when a Buffer migrates between providers.
#include <cstdint>
#include <utility>
#include "tensor_descriptor.hpp"

namespace forge_rt {

struct BufferSpec {
    TensorDescriptor descriptor;
    uint64_t byte_size = 0;
    uint64_t alignment = 0;
};

class Buffer {
public:
    Buffer() = default;

    // Construct from external memory. If `owns == true`, ~Buffer calls
    // free() on `data` — callers using CUDA / HLS allocators MUST register
    // a custom provider (Phase 5) rather than passing owns=true.
    Buffer(void* data, BufferSpec spec, bool owns = false)
        : data_(data), spec_(std::move(spec)), owns_(owns) {}

    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void* data() const noexcept { return data_; }
    const BufferSpec& spec() const noexcept { return spec_; }
    BufferSpec& spec() noexcept { return spec_; }
    bool owns() const noexcept { return owns_; }

    // Release ownership without freeing (e.g. when handing the buffer to a
    // downstream provider that takes the raw pointer).
    void* release() noexcept;

private:
    void* data_ = nullptr;
    BufferSpec spec_{};
    bool owns_ = false;
};

}  // namespace forge_rt
