// ForgeRT C++ ABI — Result<T> is a header-only template; this .cpp is a
// linker anchor so the CMake STATIC library has a translation unit per
// header (and so future ABI-versioned result types can land here without
// touching the headers).
#include "forge_rt/result.hpp"
