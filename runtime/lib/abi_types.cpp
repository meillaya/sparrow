// ForgeRT C++ ABI — linker anchor for abi_types.
// The to_string / kByteSize helpers live in the header as `inline constexpr`
// so every TU sees the same definition (ODR-safe). This .cpp exists so the
// CMake STATIC library has at least one source per ABI header and so future
// ABI-versioned helpers (e.g. abi_v2::to_string) can land here without
// touching the headers.
#include "forge_rt/abi_types.hpp"
