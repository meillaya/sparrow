// ForgeRT — MicroGpuDeviceProvider implementation (spec §11).
//
// Shipped scope: every operation runs on the host. allocate uses
// std::aligned_alloc, copy is std::memcpy, launch fires a Python
// child process that runs `backends.microgpu.interp.run` on the
// caller's input bytes + the artifact's bytecode. The DeviceProvider
// ABI is the stable contract; the MicroGPU target slots into the
// same registry as Triton / Astra.
#include "forge_rt/providers/microgpu_provider.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "forge_rt/abi_types.hpp"

namespace forge_rt::providers {

namespace {

// Round `n` up to the next multiple of `align`. `align` must be > 0.
std::size_t round_up_to(std::size_t n, std::size_t align) {
    return (n + (align - 1)) & ~(align - 1);
}

// Clamp alignment to a power of two >= alignof(void*); mirrors Triton.
std::size_t clamp_alignment(uint64_t requested) {
    if (requested == 0) return alignof(void*);
    if ((requested & (requested - 1)) != 0) return alignof(void*);
    return static_cast<std::size_t>(requested);
}

// base64-encode a contiguous byte buffer. The microgpu child process
// accepts fp32 tensor payloads + bytecode word arrays as base64
// strings to keep stdout free of arbitrary binary bytes (which would
// collide with line buffering and JSON framing).
std::string base64_encode(const void* data, std::size_t nbytes) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::string out;
    out.reserve(((nbytes + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= nbytes) {
        std::uint32_t v = (std::uint32_t(bytes[i]) << 16) |
                          (std::uint32_t(bytes[i + 1]) << 8) |
                          std::uint32_t(bytes[i + 2]);
        out.push_back(kAlphabet[(v >> 18) & 0x3F]);
        out.push_back(kAlphabet[(v >> 12) & 0x3F]);
        out.push_back(kAlphabet[(v >> 6) & 0x3F]);
        out.push_back(kAlphabet[v & 0x3F]);
        i += 3;
    }
    if (i < nbytes) {
        std::uint32_t v = std::uint32_t(bytes[i]) << 16;
        if (i + 1 < nbytes) v |= std::uint32_t(bytes[i + 1]) << 8;
        out.push_back(kAlphabet[(v >> 18) & 0x3F]);
        out.push_back(kAlphabet[(v >> 12) & 0x3F]);
        if (i + 1 < nbytes) {
            out.push_back(kAlphabet[(v >> 6) & 0x3F]);
        } else {
            out.push_back('=');
        }
        out.push_back('=');
    }
    return out;
}

// Minimal subprocess helper: fork+exec, pipe stdin/stdout, block
// until the child exits and capture its stdout. c++20 has no
// std::process so we roll the POSIX primitives here — single-file, no
// external deps.
struct SubprocessResult {
    int exit_code;            // -1 if the child did not exit normally
    std::string stdout_data;  // raw stdout capture
};

SubprocessResult run_subprocess(const std::vector<std::string>& argv,
                                const std::string& stdin_data,
                                const std::string& cwd) {
    int in_pipe[2];
    int out_pipe[2];
    if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0) {
        return SubprocessResult{-1, {}};
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        return SubprocessResult{-1, {}};
    }

    if (pid == 0) {
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);

        if (!cwd.empty()) {
            (void)::chdir(cwd.c_str());
        }

        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& s : argv) {
            cargv.push_back(const_cast<char*>(s.c_str()));
        }
        cargv.push_back(nullptr);

        ::execvp(cargv[0], cargv.data());
        std::_Exit(127);
    }

    ::close(in_pipe[0]);
    ::close(out_pipe[1]);

    std::size_t total_written = 0;
    while (total_written < stdin_data.size()) {
        const ssize_t n = ::write(in_pipe[1], stdin_data.data() + total_written,
                                  stdin_data.size() - total_written);
        if (n <= 0) break;
        total_written += static_cast<std::size_t>(n);
    }
    ::close(in_pipe[1]);

    std::string out;
    std::array<char, 4096> buf{};
    while (true) {
        const ssize_t r = ::read(out_pipe[0], buf.data(), buf.size());
        if (r <= 0) break;
        out.append(buf.data(), static_cast<std::size_t>(r));
    }
    ::close(out_pipe[0]);

    int status = 0;
    ::waitpid(pid, &status, 0);
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return SubprocessResult{code, std::move(out)};
}

// The Python body handed to `python -c`. It reads a JSON spec from
// stdin (one input tensor as base64 + one bytecode word array as a
// JSON list of ints + the requested output dtype), runs
// `backends.microgpu.interp.run` directly, and emits a
// base64-encoded tensor payload followed by a JSON metadata line on
// stdout. The single-warp model uses one i32 register file + one flat
// memory buffer; the caller is responsible for choosing a launch
// program whose memory layout matches the input/output shapes.
constexpr const char* kMicroGpuLaunchScript = R"PY(
import base64
import json
import sys

from backends.microgpu.interp import (
    BYTES_PER_WORD,
    REGISTER_COUNT,
    run,
)

spec = json.loads(sys.stdin.read())
words = [int(w) & 0xFFFF for w in spec["words"]]
input_b64 = spec.get("input", "")
output_elements = int(spec.get("output_elements", 1))

if input_b64:
    payload = base64.b64decode(input_b64)
else:
    payload = b""

memory = bytearray(payload)

# Pre-seed registers from a base64 list-of-ints if the caller passed
# one. Default to all zeros.
regs = [0] * REGISTER_COUNT
if "registers" in spec:
    seed = base64.b64decode(spec["registers"])
    for index in range(min(REGISTER_COUNT, len(seed) // BYTES_PER_WORD)):
        start = index * BYTES_PER_WORD
        end = start + BYTES_PER_WORD
        regs[index] = int.from_bytes(seed[start:end], "little", signed=True)

state = run(words, memory=memory, registers=regs)
out_bytes = b"".join(
    (state.registers[i] & 0xFFFFFFFF).to_bytes(BYTES_PER_WORD, "little", signed=False)
    for i in range(output_elements)
)
sys.stdout.write(base64.b64encode(out_bytes).decode("ascii"))
sys.stdout.write("\n")
sys.stdout.write(json.dumps({"elements": output_elements}))
sys.stdout.write("\n")
sys.stdout.flush()
)PY";

// Locate the project root for the child process's cwd (so the
// `backends.*` imports resolve from the workspace's editable
// install).
std::string find_project_root(const std::string& venv_path) {
    if (!venv_path.empty()) {
        const std::string suffix = "/.venv/bin/python";
        const auto pos = venv_path.find(suffix);
        if (pos != std::string::npos) {
            return venv_path.substr(0, pos);
        }
    }

    std::array<char, 4096> cwd{};
    if (::getcwd(cwd.data(), cwd.size()) != nullptr) {
        std::string path(cwd.data());
        for (int i = 0; i < 10; ++i) {
            const std::string pyp = path + "/pyproject.toml";
            const std::string bck = path + "/backends";
            if (::access(pyp.c_str(), F_OK) == 0 &&
                ::access(bck.c_str(), F_OK) == 0) {
                return path;
            }
            const auto slash = path.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            path.erase(slash);
        }
        return std::string(cwd.data());
    }

    return {};
}

}  // namespace

MicroGpuProvider::MicroGpuProvider(std::string python_executable)
    : python_executable_(std::move(python_executable)) {}

MicroGpuProvider::~MicroGpuProvider() = default;

Result<Buffer> MicroGpuProvider::allocate(const BufferSpec& spec) {
    if (spec.byte_size == 0) {
        return Error{"MicroGpuProvider::allocate: zero byte_size", 2201};
    }
    const std::size_t align = clamp_alignment(spec.alignment);
    const std::size_t bytes = round_up_to(static_cast<std::size_t>(spec.byte_size),
                                          align);
    const std::size_t alloc_size = std::max(bytes, align);
    void* raw = std::aligned_alloc(align, alloc_size);
    if (raw == nullptr) {
        return Error{"MicroGpuProvider::allocate: std::aligned_alloc failed",
                     2202};
    }
    BufferSpec out_spec = spec;
    out_spec.byte_size = bytes;
    out_spec.alignment = align;
    return Buffer{raw, out_spec, /*owns=*/true};
}

Result<Event> MicroGpuProvider::copy(Buffer& dst, const Buffer& src,
                                    std::span<const Event> /*deps*/) {
    if (src.spec().descriptor.memory_space != MemorySpace::Host ||
        dst.spec().descriptor.memory_space != MemorySpace::Host) {
        return Error{
            "MicroGpuProvider::copy: only Host<->Host copies are supported "
            "(got src=" +
                std::string(to_string(src.spec().descriptor.memory_space)) +
                " dst=" +
                std::string(to_string(dst.spec().descriptor.memory_space)) +
                ")",
            2210};
    }
    if (src.data() == nullptr || dst.data() == nullptr) {
        return Error{"MicroGpuProvider::copy: null buffer pointer", 2211};
    }
    const std::size_t nbytes =
        std::min(src.spec().byte_size, dst.spec().byte_size);
    if (nbytes == 0) {
        return Error{"MicroGpuProvider::copy: zero-length copy", 2212};
    }
    std::memcpy(dst.data(), src.data(), nbytes);
    Event ev{next_event_id_++};
    ev.mark_completed();
    return ev;
}

Result<Event> MicroGpuProvider::launch(const ArtifactDescriptor& artifact,
                                      std::span<const Buffer> inputs,
                                      std::span<Buffer> outputs,
                                      std::span<const Event> /*deps*/) {
    if (artifact.target != TargetKind::MicroGPU) {
        return Error{
            "MicroGpuProvider::launch: artifact.target must be "
            "TargetKind::MicroGPU (got " +
                std::string(to_string(artifact.target)) + ")",
            2220};
    }
    if (outputs.size() < 1) {
        return Error{"MicroGpuProvider::launch: need 1 output (y); got " +
                         std::to_string(outputs.size()),
                     2221};
    }
    if (outputs[0].data() == nullptr) {
        return Error{"MicroGpuProvider::launch: null output buffer", 2222};
    }
    if (kByteSize(outputs[0].spec().descriptor.dtype) != sizeof(std::int32_t)) {
        return Error{"MicroGpuProvider::launch: output dtype must be a 4-byte "
                     "integer or float, got " +
                         std::string(to_string(outputs[0].spec().descriptor.dtype)),
                     2227};
    }
    if (outputs[0].spec().byte_size % sizeof(std::int32_t) != 0) {
        return Error{"MicroGpuProvider::launch: output byte_size must be a "
                     "multiple of 4, got " +
                         std::to_string(outputs[0].spec().byte_size),
                     2227};
    }

    // Build the bytecode word list. The artifact.binary carries the
    // packed 16-bit words as little-endian bytes (two bytes per
    // word). Reject malformed artifacts before forking.
    if ((artifact.binary.size() % 2) != 0) {
        return Error{
            "MicroGpuProvider::launch: artifact.binary size must be even "
            "(16-bit word stream), got " +
                std::to_string(artifact.binary.size()),
            2223};
    }
    std::ostringstream words_json;
    words_json << "[";
    bool first = true;
    for (std::size_t offset = 0; offset < artifact.binary.size();
         offset += 2) {
        const auto lo = std::to_integer<std::uint16_t>(artifact.binary[offset]);
        const auto hi = std::to_integer<std::uint16_t>(artifact.binary[offset + 1]);
        const std::uint16_t word =
            static_cast<std::uint16_t>(lo | (hi << 8));
        if (!first) words_json << ",";
        first = false;
        words_json << word;
    }
    words_json << "]";

    // The launched program reads at most one input buffer; the
    // interpreter's flat memory model treats that buffer as the
    // entire backing store.
    std::string input_b64;
    if (!inputs.empty() && inputs[0].data() != nullptr) {
        input_b64 =
            base64_encode(inputs[0].data(), inputs[0].spec().byte_size);
    }

    const std::size_t output_elements =
        static_cast<std::size_t>(outputs[0].spec().byte_size / 4);

    std::ostringstream spec_stream;
    spec_stream << "{"
                << "\"words\":" << words_json.str() << ","
                << "\"input\":\"" << input_b64 << "\","
                << "\"output_elements\":" << output_elements
                << "}";
    const std::string json_spec = spec_stream.str();

    std::vector<std::string> argv;
    argv.push_back(python_executable_.empty() ? std::string("python3")
                                              : python_executable_);
    argv.push_back("-c");
    argv.push_back(kMicroGpuLaunchScript);

    SubprocessResult sub = run_subprocess(argv, json_spec, find_project_root(python_executable_));
    if (sub.exit_code != 0) {
        return Error{
            "MicroGpuProvider::launch: python child exited " +
                std::to_string(sub.exit_code) +
                " (stdout prefix: " + sub.stdout_data.substr(0, 200) + ")",
            2224};
    }

    // Parse the first line as base64, the second as JSON metadata.
    auto first_nl = sub.stdout_data.find('\n');
    if (first_nl == std::string::npos) {
        return Error{"MicroGpuProvider::launch: child output malformed",
                     2225};
    }
    const std::string b64 = sub.stdout_data.substr(0, first_nl);
    static constexpr int kTable[256] = {
        // populated below by init guard.
        0,
    };
    static bool kReady = false;
    static int kDecTable[256];
    if (!kReady) {
        for (int i = 0; i < 256; ++i) kDecTable[i] = -1;
        const char* alpha =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) {
            kDecTable[static_cast<unsigned char>(alpha[i])] = i;
        }
        kReady = true;
        (void)kTable;
    }
    std::vector<unsigned char> raw;
    {
        int quartet = 0;
        int bits = 0;
        for (char c : b64) {
            if (c == '=' || c == '\n' || c == '\r') break;
            const int v = kDecTable[static_cast<unsigned char>(c)];
            if (v < 0) continue;
            quartet = (quartet << 6) | v;
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                raw.push_back(static_cast<unsigned char>((quartet >> bits) & 0xFF));
                quartet &= (1 << bits) - 1;
            }
        }
    }
    const std::size_t expected = output_elements * 4;
    if (raw.size() != expected) {
        return Error{
            "MicroGpuProvider::launch: decoded payload size " +
                std::to_string(raw.size()) + " != expected " +
                std::to_string(expected),
            2226};
    }
    std::memcpy(outputs[0].data(), raw.data(), expected);

    Event ev{next_event_id_++};
    ev.mark_completed();
    return ev;
}

Result<void> MicroGpuProvider::wait(const Event& /*event*/) {
    return Result<void>{};
}

}  // namespace forge_rt::providers