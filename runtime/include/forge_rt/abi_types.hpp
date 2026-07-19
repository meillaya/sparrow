#pragma once

// ForgeRT C++ ABI — POD enums (spec §12 "Common ABI").
// These are the wire-format identifiers: stable numeric values are part of
// the ABI and must NOT be reordered. Append new variants at the END of each
// enum and bump the runtime's serialization version.
#include <cstdint>
#include <cstddef>

namespace forge_rt {

enum class DataType : uint8_t {
    Float32  = 0,
    Float16  = 1,
    BFloat16 = 2,
    Int32    = 3,
    Int8     = 4,
    Int4     = 5,
};

enum class MemorySpace : uint8_t {
    Host    = 0,
    Device0 = 1,
    Device1 = 2,
};

enum class TargetKind : uint8_t {
    Triton   = 0,
    HLS      = 1,
    Astra    = 2,
    MicroGPU = 3,
    CPU      = 4,
};

// Returns the byte size of a single element of `d`. Int4 packs two elements
// per byte; the caller is responsible for ceiling in the tensor descriptor.
inline constexpr std::size_t kByteSize(DataType d) {
    switch (d) {
        case DataType::Float32:  return 4;
        case DataType::Float16:  return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int32:    return 4;
        case DataType::Int8:     return 1;
        case DataType::Int4:     return 1;  // packed: 2 elements per byte
    }
    return 0;
}

inline constexpr const char* to_string(DataType d) {
    switch (d) {
        case DataType::Float32:  return "Float32";
        case DataType::Float16:  return "Float16";
        case DataType::BFloat16: return "BFloat16";
        case DataType::Int32:    return "Int32";
        case DataType::Int8:     return "Int8";
        case DataType::Int4:     return "Int4";
    }
    return "Unknown";
}

inline constexpr const char* to_string(MemorySpace m) {
    switch (m) {
        case MemorySpace::Host:    return "Host";
        case MemorySpace::Device0: return "Device0";
        case MemorySpace::Device1: return "Device1";
    }
    return "Unknown";
}

inline constexpr const char* to_string(TargetKind t) {
    switch (t) {
        case TargetKind::Triton:   return "Triton";
        case TargetKind::HLS:      return "HLS";
        case TargetKind::Astra:    return "Astra";
        case TargetKind::MicroGPU: return "MicroGPU";
        case TargetKind::CPU:      return "CPU";
    }
    return "Unknown";
}

}  // namespace forge_rt
