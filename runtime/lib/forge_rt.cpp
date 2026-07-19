// ForgeRT C++ ABI — umbrella translation unit.
// Pulls in every public header so the STATIC library exports their
// non-inline symbols (none today; everything is either inline constexpr
// or header-only templates). The .cpp exists as a build-system anchor
// and as a single TU to add ODR-checking TUs in future ABI versions.
#include "forge_rt/forge_rt.hpp"
