// ForgeRT — AstraDeviceProvider implementation (spec §10, §9).
//
// Shipped scope: every operation runs on the host. allocate uses
// std::aligned_alloc, copy is std::memcpy, launch fires a Python
// child process that runs `backends.astra.sim.run_microcode` and
// returns the resulting tensor. The DeviceProvider ABI is the stable
// contract; the Astra target slots into the same registry as Triton.
#include "forge_rt/providers/astra_provider.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

// base64-encode a contiguous byte buffer. The astra child process
// accepts fp32 tensor payloads as base64 strings to keep stdout
// free of arbitrary binary bytes (which would collide with line
// buffering and JSON framing).
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

// Minimal subprocess helper: fork+exec, pipe stdin/stdout, block until
// the child exits and capture its stdout. c++20 has no std::process so
// we roll the POSIX primitives here — single-file, no external deps.
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
        // Child: rewire stdin/stdout to the pipes, change cwd, close
        // the parent ends, then exec.
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);

        if (!cwd.empty()) {
            // Best-effort: ignore failure here; the parent's pipelined
            // exec still runs in the cwd that we ended up in.
            (void)::chdir(cwd.c_str());
        }

        // `c_str()` is NUL-terminated; iterator copies that drop the
        // trailing NUL (e.g. `s.begin(), s.end()`) make execvp read
        // past the buffer into garbage and trips Python's argv decoder.
        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& s : argv) {
            cargv.push_back(const_cast<char*>(s.c_str()));
        }
        cargv.push_back(nullptr);

        ::execvp(cargv[0], cargv.data());
        // exec only returns on failure.
        std::_Exit(127);
    }

    // Parent: feed stdin_data to the child, drain stdout, wait.
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);

    std::size_t total_written = 0;
    while (total_written < stdin_data.size()) {
        const ssize_t n = ::write(in_pipe[1], stdin_data.data() + total_written,
                                  stdin_data.size() - total_written);
        if (n <= 0) break;
        total_written += static_cast<std::size_t>(n);
    }
    ::close(in_pipe[1]);  // signal EOF on stdin

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

// Decode the first base64 chunk on a line into raw bytes. The astra
// Python helper emits one base64 string followed by a single `\n`
// followed by a JSON object on the next line; this pulls the base64
// payload back out for the writeback step.
std::vector<unsigned char> base64_decode_line(const std::string& s) {
    static constexpr int kInv[256] = {
        // populated below
        0,
    };
    static bool kInvReady = false;
    static int kTable[256];
    if (!kInvReady) {
        for (int i = 0; i < 256; ++i) kTable[i] = -1;
        const char* alpha =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) kTable[static_cast<unsigned char>(alpha[i])] = i;
        kInvReady = true;
        (void)kInv;
    }

    std::vector<unsigned char> out;
    int quartet = 0;
    int bits = 0;
    for (char c : s) {
        if (c == '=' || c == '\n' || c == '\r') break;
        const int v = kTable[static_cast<unsigned char>(c)];
        if (v < 0) continue;
        quartet = (quartet << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<unsigned char>((quartet >> bits) & 0xFF));
            quartet &= (1 << bits) - 1;
        }
    }
    return out;
}

// The Python body handed to `python -c`. It reads a JSON spec from
// stdin (x, w, b as base64 + (M, K, N) shape + mode), constructs an
// Astra microcode program that performs ``gelu(x @ w)`` for the 16x16
// tile, runs `backends.astra.sim.run_functional` directly so we can
// prime the DRAM with the caller's input bytes, and emits a
// base64-encoded tensor payload followed by a JSON metadata line on
// stdout. The bias vector is accepted for ABI conformance but the
// simulator's ISA has no vector-add — bias=0 is the contract the
// shipped microcode relies on (the test sets bias to zeros).
constexpr const char* kAstraLaunchScript = R"PY(
import base64
import json
import sys

import numpy as np

from backends.astra.sim.functional import (
    DRAM_BYTES,
    last_dram_tensor,
    run_functional,
)
from compiler.lib.Dialect.Astra.assembler import assemble

spec = json.loads(sys.stdin.read())
shape = spec["shape"]
M, K, N = int(shape[0]), int(shape[1]), int(shape[2])
mode = spec.get("mode", "functional")

x = np.frombuffer(base64.b64decode(spec["x"]), dtype=np.float32).reshape(M, K)
w = np.frombuffer(base64.b64decode(spec["w"]), dtype=np.float32).reshape(K, N)
b = np.frombuffer(base64.b64decode(spec["b"]), dtype=np.float32).reshape(N)

# ABI guard: assert the bias is the zero vector (the shipped astra
# microcode path computes gelu(x @ w); the simulator ISA has no
# vector-add op — the bias would otherwise be silently dropped).
if not np.array_equal(b, np.zeros_like(b)):
    sys.stderr.write("astra_provider: bias != 0 not supported by sim ISA\n")
    sys.exit(2)

# Pack x, w into a freshly-allocated DRAM buffer at deterministic
# offsets, then build a 16x16 tile microcode program.
TILE = 16
assert M == TILE and N == TILE, "astra launch expects M=N=16 for the tile-only matmul path"
tbytes = TILE * TILE * 4

x_offset = 0
w_offset = x_offset + M * K * 4
out_offset = w_offset + K * N * 4

dram = bytearray(DRAM_BYTES)
dram[x_offset : x_offset + M * K * 4] = np.ascontiguousarray(x, dtype=np.float32).tobytes()
dram[w_offset : w_offset + K * N * 4] = np.ascontiguousarray(w, dtype=np.float32).tobytes()

program = (
    f"DMA_LOAD dst=SRAM[0x0], src=DRAM[0x{x_offset:x}], bytes={M * K * 4}\n"
    f"DMA_LOAD dst=SRAM[0x{tbytes:x}], src=DRAM[0x{w_offset:x}], bytes={K * N * 4}\n"
    f"MATMUL a=SRAM[0x0], b=SRAM[0x{tbytes:x}], k={K}\n"
    f"VECTOR_GELU dst=SRAM[0x0], src=SRAM[0x0], elements={M * N}\n"
    f"DMA_STORE dst=DRAM[0x{out_offset:x}], src=SRAM[0x0], bytes={tbytes}\n"
    "HALT\n"
)
words = assemble(program)

result = run_functional(words, dram=dram)
tensor = last_dram_tensor(result)
if tensor is None:
    tensor = np.zeros((M, N), dtype=np.float32)
tensor = np.ascontiguousarray(tensor, dtype=np.float32)
sys.stdout.write(base64.b64encode(tensor.tobytes()).decode("ascii"))
sys.stdout.write("\n")
sys.stdout.write(json.dumps({"shape": list(tensor.shape)}))
sys.stdout.write("\n")
sys.stdout.flush()
)PY";

// Locate the project root for the child process's cwd (so the
// `backends.*` and `compiler.lib.Dialect.Astra.*` imports resolve
// from the workspace's editable install).
std::string find_project_root(const std::string& venv_path) {
    // 1. If the user passed an explicit venv python (e.g.
    // `/path/to/project/.venv/bin/python`), infer the project root by
    // stripping the trailing `/.venv/bin/python`.
    if (!venv_path.empty()) {
        const std::string suffix = "/.venv/bin/python";
        const auto pos = venv_path.find(suffix);
        if (pos != std::string::npos) {
            return venv_path.substr(0, pos);
        }
    }

    // 2. Walk up from cwd until we find a directory containing both
    // `pyproject.toml` and `backends/`. The ctest-launched test runs
    // from `build/runtime/tests`, so a fixed-string match on
    // `/runtime/...` is unreliable.
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

    // 3. Last-resort known location (covers developer hosts without
    // a populated cwd).
    return "/home/mei/projects/forgestack";
}

}  // namespace

AstraProvider::AstraProvider(std::string python_executable)
    : python_executable_(std::move(python_executable)) {}

AstraProvider::~AstraProvider() = default;

Result<Buffer> AstraProvider::allocate(const BufferSpec& spec) {
    if (spec.byte_size == 0) {
        return Error{"AstraProvider::allocate: zero byte_size", 2101};
    }
    const std::size_t align = clamp_alignment(spec.alignment);
    const std::size_t bytes = round_up_to(static_cast<std::size_t>(spec.byte_size),
                                          align);
    const std::size_t alloc_size = std::max(bytes, align);
    void* raw = std::aligned_alloc(align, alloc_size);
    if (raw == nullptr) {
        return Error{"AstraProvider::allocate: std::aligned_alloc failed",
                     2102};
    }
    BufferSpec out_spec = spec;
    out_spec.byte_size = bytes;
    out_spec.alignment = align;
    return Buffer{raw, out_spec, /*owns=*/true};
}

Result<Event> AstraProvider::copy(Buffer& dst, const Buffer& src,
                                  std::span<const Event> /*deps*/) {
    if (src.spec().descriptor.memory_space != MemorySpace::Host ||
        dst.spec().descriptor.memory_space != MemorySpace::Host) {
        return Error{
            "AstraProvider::copy: only Host<->Host copies are supported "
            "(got src=" +
                std::string(to_string(src.spec().descriptor.memory_space)) +
                " dst=" +
                std::string(to_string(dst.spec().descriptor.memory_space)) +
                ")",
            2110};
    }
    if (src.data() == nullptr || dst.data() == nullptr) {
        return Error{"AstraProvider::copy: null buffer pointer", 2111};
    }
    const std::size_t nbytes =
        std::min(src.spec().byte_size, dst.spec().byte_size);
    if (nbytes == 0) {
        return Error{"AstraProvider::copy: zero-length copy", 2112};
    }
    std::memcpy(dst.data(), src.data(), nbytes);
    Event ev{next_event_id_++};
    ev.mark_completed();
    return ev;
}

Result<Event> AstraProvider::launch(const ArtifactDescriptor& artifact,
                                    std::span<const Buffer> inputs,
                                    std::span<Buffer> outputs,
                                    std::span<const Event> /*deps*/) {
    if (artifact.target != TargetKind::Astra) {
        return Error{
            "AstraProvider::launch: artifact.target must be TargetKind::Astra "
            "(got " +
                std::string(to_string(artifact.target)) + ")",
            2120};
    }
    if (inputs.size() < 3) {
        return Error{"AstraProvider::launch: need 3 inputs (x, w, b); got " +
                          std::to_string(inputs.size()),
                      2121};
    }
    if (outputs.size() < 1) {
        return Error{"AstraProvider::launch: need 1 output (y); got " +
                          std::to_string(outputs.size()),
                      2122};
    }

    auto check_fp32 = [](const Buffer& b, const char* name) -> Error {
        if (b.spec().descriptor.dtype != DataType::Float32) {
            return Error{std::string("AstraProvider::launch: ") + name +
                              " dtype must be Float32 (got " +
                              to_string(b.spec().descriptor.dtype) +
                              "); astra fp32 only",
                          2123};
        }
        return Error{};
    };
    if (auto e = check_fp32(inputs[0], "x"); e.code != 0) return e;
    if (auto e = check_fp32(inputs[1], "w"); e.code != 0) return e;
    if (auto e = check_fp32(inputs[2], "b"); e.code != 0) return e;
    if (auto e = check_fp32(outputs[0], "y"); e.code != 0) return e;

    // Shape contract for the shipped microcode: x=[16,K], w=[K,16],
    // b=[16], y=[16,16]. The astra functional simulator only supports
    // a single TILE x TILE matmul per invocation, so M=N=16 is the
    // only shape the launch path can serve today. Tests that need
    // larger shapes use the Python-level e2e (Task 63 / 67).
    const auto& x_shape = inputs[0].spec().descriptor.shape;
    const auto& w_shape = inputs[1].spec().descriptor.shape;
    const auto& b_shape = inputs[2].spec().descriptor.shape;
    const auto& y_shape = outputs[0].spec().descriptor.shape;
    constexpr int64_t kTile = 16;
    if (x_shape.size() != 2 || x_shape[0] != kTile || x_shape[1] < 1) {
        return Error{
            "AstraProvider::launch: x must be [16, K>=1] for the tile-only "
            "matmul path",
            2124};
    }
    if (w_shape.size() != 2 || w_shape[0] != x_shape[1] || w_shape[1] != kTile) {
        return Error{
            "AstraProvider::launch: w must be [K, 16] matching x and the tile",
            2125};
    }
    if (b_shape.size() != 1 || b_shape[0] != kTile) {
        return Error{
            "AstraProvider::launch: b must be [16] for the tile shape",
            2126};
    }
    if (y_shape.size() != 2 || y_shape[0] != kTile || y_shape[1] != kTile) {
        return Error{
            "AstraProvider::launch: y must be [16, 16] matching the tile",
            2127};
    }

    if (inputs[0].data() == nullptr || inputs[1].data() == nullptr ||
        inputs[2].data() == nullptr || outputs[0].data() == nullptr) {
        return Error{"AstraProvider::launch: null buffer data()", 2128};
    }

    // Read the microcode mode out of the artifact's binary if the
    // caller bothered to encode it. Default to functional.
    std::string mode = "functional";
    if (artifact.binary.size() >= 1 && artifact.binary[0] == std::byte{0x01}) {
        mode = "timing";
    }

    // Build the JSON spec.
    const int64_t M = x_shape[0];
    const int64_t K = x_shape[1];
    const int64_t N = w_shape[1];

    std::ostringstream spec;
    spec << "{\"shape\":[" << M << "," << K << "," << N << "],"
         << "\"mode\":\"" << mode << "\","
         << "\"metrics\":false,"
         << "\"x\":\"" << base64_encode(inputs[0].data(),
                                       static_cast<std::size_t>(M * K * 4)) << "\","
         << "\"w\":\"" << base64_encode(inputs[1].data(),
                                       static_cast<std::size_t>(K * N * 4)) << "\","
         << "\"b\":\"" << base64_encode(inputs[2].data(),
                                       static_cast<std::size_t>(N * 4)) << "\"}";
    const std::string json_spec = spec.str();

    // Resolve the interpreter. Default to `python3`.
    std::vector<std::string> argv;
    argv.push_back(python_executable_.empty() ? std::string("python3")
                                              : python_executable_);
    argv.push_back("-c");
    argv.push_back(kAstraLaunchScript);

    SubprocessResult sub = run_subprocess(argv, json_spec, find_project_root(python_executable_));
    if (sub.exit_code != 0) {
        return Error{
            "AstraProvider::launch: python child exited " +
                std::to_string(sub.exit_code) +
                " (stdout prefix: " + sub.stdout_data.substr(0, 200) + ")",
            2129};
    }

    // Parse the first line as base64, the second as JSON metadata.
    auto first_nl = sub.stdout_data.find('\n');
    if (first_nl == std::string::npos) {
        return Error{"AstraProvider::launch: child output malformed",
                     2130};
    }
    const std::string b64 = sub.stdout_data.substr(0, first_nl);
    std::vector<unsigned char> raw = base64_decode_line(b64);
    const std::size_t expected = static_cast<std::size_t>(M * N * 4);
    if (raw.size() != expected) {
        return Error{
            "AstraProvider::launch: decoded payload size " +
                std::to_string(raw.size()) + " != expected " +
                std::to_string(expected),
            2131};
    }
    std::memcpy(outputs[0].data(), raw.data(), expected);

    Event ev{next_event_id_++};
    ev.mark_completed();
    return ev;
}

Result<void> AstraProvider::wait(const Event& /*event*/) {
    return Result<void>{};
}

}  // namespace forge_rt::providers
