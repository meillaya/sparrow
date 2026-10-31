// Sparrow native bridge.
//
// One narrow entry point for the fused MLP slice: build a version 2
// ExecutionPlan with one Triton node, run it through Scheduler and
// TritonProvider, and return the output tensor. The point is to exercise
// the real C++ kernel and the real runtime contract from Python without
// pretending that general region codegen exists.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "forge_rt/artifact_descriptor.hpp"
#include "forge_rt/execution_plan.hpp"
#include "forge_rt/scheduler.hpp"
#include "forge_rt/providers/triton_provider.hpp"

namespace py = pybind11;
using namespace forge_rt;

namespace {

BufferSpec make_spec(const std::vector<int64_t>& shape, const std::size_t elements) {
    BufferSpec spec;
    spec.descriptor.dtype = DataType::Float32;
    spec.descriptor.shape = shape;
    spec.descriptor.memory_space = MemorySpace::Host;
    spec.descriptor.byte_offset = 0;
    spec.byte_size = elements * sizeof(float);
    spec.alignment = 64;
    return spec;
}

TensorDefinition make_tensor(TensorId id, const char* name, std::vector<int64_t> shape) {
    TensorDefinition tensor;
    tensor.id = id;
    tensor.name = name;
    tensor.dtype = DataType::Float32;
    tensor.shape = std::move(shape);
    tensor.memory_space = MemorySpace::Host;
    return tensor;
}

py::array_t<float> run_fused_mlp(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> w,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> b) {
    if (x.ndim() != 2 || w.ndim() != 2 || b.ndim() != 1) {
        throw py::value_error("run_fused_mlp expects x=[M,K], w=[K,N], b=[N]");
    }
    const int64_t M = x.shape(0);
    const int64_t K = x.shape(1);
    const int64_t N = w.shape(1);
    if (w.shape(0) != K) {
        throw py::value_error("run_fused_mlp: x.shape[1] must equal w.shape[0]");
    }
    if (b.shape(0) != N) {
        throw py::value_error("run_fused_mlp: b.shape[0] must equal w.shape[1]");
    }
    if (M <= 0 || K <= 0 || N <= 0) {
        throw py::value_error("run_fused_mlp: every dimension must be positive");
    }

    std::vector<float> output(static_cast<std::size_t>(M * N), 0.0f);

    ExecutionPlan plan;
    plan.name = "fused_mlp_native";
    ArtifactDescriptor artifact;
    artifact.target = TargetKind::Triton;
    artifact.entry_point = "fused_gemm_bias_gelu";
    plan.artifacts.push_back(std::move(artifact));
    plan.tensors = {
        make_tensor(0, "x", {M, K}),
        make_tensor(1, "w", {K, N}),
        make_tensor(2, "b", {N}),
        make_tensor(3, "y", {M, N}),
    };
    ExecutableNode node;
    node.id = 0;
    node.device = 0;
    node.artifact = 0;
    node.inputs = {0, 1, 2};
    node.outputs = {3};
    plan.nodes.push_back(std::move(node));
    plan.inputs = {{0, "x"}, {1, "w"}, {2, "b"}};
    plan.outputs = {{3, "y"}};

    BufferSpec x_spec = make_spec({M, K}, static_cast<std::size_t>(M * K));
    BufferSpec w_spec = make_spec({K, N}, static_cast<std::size_t>(K * N));
    BufferSpec b_spec = make_spec({N}, static_cast<std::size_t>(N));
    BufferSpec y_spec = make_spec({M, N}, static_cast<std::size_t>(M * N));

    std::vector<Buffer> inputs;
    inputs.emplace_back(x.mutable_data(0, 0), x_spec, false);
    inputs.emplace_back(w.mutable_data(0, 0), w_spec, false);
    inputs.emplace_back(b.mutable_data(0), b_spec, false);
    std::vector<Buffer> outputs;
    outputs.emplace_back(output.data(), y_spec, false);

    auto provider = std::make_shared<providers::TritonProvider>();
    Scheduler scheduler({provider});
    auto result = scheduler.execute(plan, inputs, outputs);
    if (!result.ok()) {
        throw std::runtime_error("run_fused_mlp: " + result.error().message);
    }

    py::array_t<float> result_array({static_cast<py::ssize_t>(M), static_cast<py::ssize_t>(N)});
    std::copy(output.begin(), output.end(), result_array.mutable_data());
    return result_array;
}

}  // namespace

PYBIND11_MODULE(_sparrow_rt, module) {
    module.doc() = "Sparrow runtime bridge for the fused MLP slice";
    module.def("run_fused_mlp", &run_fused_mlp,
               "Run the fp32 fused MLP kernel through ForgeRT",
               py::arg("x"), py::arg("w"), py::arg("b"));
}
