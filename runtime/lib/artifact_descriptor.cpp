// ForgeRT C++ ABI — ArtifactDescriptor JSON (de)serialization.
#include "forge_rt/artifact_descriptor.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace forge_rt {

void to_json(nlohmann::json& j, const ShapeGuard& g) {
    j = nlohmann::json{
        {"indices", g.indices},
        {"kind", static_cast<uint8_t>(g.kind)},
        {"value", g.value},
    };
}

void from_json(const nlohmann::json& j, ShapeGuard& g) {
    j.at("indices").get_to(g.indices);
    g.kind = static_cast<ShapeGuardKind>(j.at("kind").get<uint8_t>());
    j.at("value").get_to(g.value);
}

void to_json(nlohmann::json& j, const ArgumentDescriptor& a) {
    j = nlohmann::json{
        {"name", a.name},
        {"dtype", static_cast<uint8_t>(a.dtype)},
        {"is_input", a.is_input},
        {"shape", a.shape},
    };
}

void from_json(const nlohmann::json& j, ArgumentDescriptor& a) {
    j.at("name").get_to(a.name);
    a.dtype = static_cast<DataType>(j.at("dtype").get<uint8_t>());
    j.at("is_input").get_to(a.is_input);
    j.at("shape").get_to(a.shape);
}

void to_json(nlohmann::json& j, const LaunchParameters& l) {
    j = nlohmann::json{
        {"grid", l.grid},
        {"block", l.block},
        {"shared_memory_bytes", l.shared_memory_bytes},
    };
}

void from_json(const nlohmann::json& j, LaunchParameters& l) {
    j.at("grid").get_to(l.grid);
    j.at("block").get_to(l.block);
    j.at("shared_memory_bytes").get_to(l.shared_memory_bytes);
}

void to_json(nlohmann::json& j, const ArtifactDescriptor& a) {
    std::vector<uint8_t> binary;
    binary.reserve(a.binary.size());
    for (const std::byte value : a.binary) {
        binary.push_back(std::to_integer<uint8_t>(value));
    }
    j = nlohmann::json{
        {"target", static_cast<uint8_t>(a.target)},
        {"entry_point", a.entry_point},
        {"binary", std::move(binary)},
        {"arguments", a.arguments},
        {"guards", a.guards},
        {"workspace_bytes", a.workspace_bytes},
        {"launch", a.launch},
    };
}

void from_json(const nlohmann::json& j, ArtifactDescriptor& a) {
    a.target = static_cast<TargetKind>(j.at("target").get<uint8_t>());
    j.at("entry_point").get_to(a.entry_point);
    std::vector<uint8_t> binary;
    j.at("binary").get_to(binary);
    a.binary.clear();
    a.binary.reserve(binary.size());
    for (const uint8_t value : binary) {
        a.binary.push_back(std::byte{value});
    }
    j.at("arguments").get_to(a.arguments);
    j.at("guards").get_to(a.guards);
    j.at("workspace_bytes").get_to(a.workspace_bytes);
    j.at("launch").get_to(a.launch);
}

}  // namespace forge_rt
