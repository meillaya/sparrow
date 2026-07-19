#pragma once

// ForgeRT C++ ABI — minimal Result<T> (replaces std::expected, which is C++23).
// We pin C++20 in CMake (see top-level CMakeLists.txt) so we ship a tiny
// variant-based result type here. NO exceptions — callers MUST check .ok()
// before touching .value(). Errors carry a message + numeric code for
// structured logging in higher layers.
#include <string>
#include <utility>
#include <variant>

namespace forge_rt {

struct Error {
    std::string message;
    int code = 0;  // ABI-stable error class; 0 = unspecified
};

// Result<void> specialization lives at the bottom of this file.
namespace detail {
struct Unit {};
}  // namespace detail

template <typename T>
class Result {
public:
    Result(T value) : v_(std::move(value)) {}
    Result(Error e) : v_(std::move(e)) {}

    bool ok() const noexcept { return std::holds_alternative<T>(v_); }
    explicit operator bool() const noexcept { return ok(); }

    const T& value() const& { return std::get<T>(v_); }
    T& value() & { return std::get<T>(v_); }
    T&& value() && { return std::get<T>(std::move(v_)); }

    const Error& error() const& { return std::get<Error>(v_); }

private:
    std::variant<T, Error> v_;
};

// Result<void>: type-erased ok/error using a Unit sentinel.
template <>
class Result<void> {
public:
    Result() : has_value_(true) {}
    Result(Error e) : error_(std::move(e)), has_value_(false) {}

    bool ok() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }

    void value() const noexcept {}  // no-op; callers branch on .ok()
    const Error& error() const noexcept { return error_; }

private:
    Error error_{};
    bool has_value_ = false;
};

}  // namespace forge_rt
