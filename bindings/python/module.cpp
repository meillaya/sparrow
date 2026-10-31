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
#include "forge_rt/providers/cpu_provider.hpp"
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
    spec.descriptor.strides.assign(shape.size(), 1);
    for (std::size_t i = shape.size(); i-- > 1;) {
        spec.descriptor.strides[i - 1] = spec.descriptor.strides[i] * shape[i];
    }
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

const TensorDefinition& find_tensor(const ExecutionPlan& plan, TensorId id) {
    for (const TensorDefinition& tensor : plan.tensors) {
        if (tensor.id == id) return tensor;
    }
    throw py::value_error("execute_plan: plan references an unknown tensor id");
}

BufferSpec spec_for(const TensorDefinition& tensor, std::size_t elements) {
    BufferSpec spec;
    spec.descriptor.dtype = tensor.dtype;
    spec.descriptor.shape = tensor.shape;
    spec.descriptor.memory_space = tensor.memory_space;
    spec.descriptor.byte_offset = 0;
    spec.descriptor.strides.assign(tensor.shape.size(), 1);
    for (std::size_t i = tensor.shape.size(); i-- > 1;) {
        spec.descriptor.strides[i - 1] = spec.descriptor.strides[i] * tensor.shape[i];
    }
    spec.byte_size = elements * sizeof(float);
    spec.alignment = 64;
    return spec;
}

void check_array_shape(const py::array& array, const TensorDefinition& tensor) {
    if (array.ndim() != static_cast<py::ssize_t>(tensor.shape.size())) {
        throw py::value_error("execute_plan: input rank does not match the plan");
    }
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
        if (array.shape(static_cast<py::ssize_t>(i)) != tensor.shape[i]) {
            throw py::value_error("execute_plan: input shape does not match the plan");
        }
    }
}

py::list execute_plan(const std::string& plan_json, py::list inputs) {
    auto parsed = parse_plan_text(plan_json);
    if (!parsed.ok()) {
        throw py::value_error("execute_plan: " + parsed.error().message);
    }
    ExecutionPlan& plan = parsed.value();
    if (plan.inputs.size() != static_cast<std::size_t>(inputs.size())) {
        throw py::value_error("execute_plan: input count does not match the plan");
    }
    for (const ArtifactDescriptor& artifact : plan.artifacts) {
        if (artifact.target != TargetKind::CPU) {
            throw py::value_error(
                "execute_plan: only CPU artifacts are supported by this binding");
        }
    }

    std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> input_arrays;
    input_arrays.reserve(plan.inputs.size());
    std::vector<Buffer> input_buffers;
    input_buffers.reserve(plan.inputs.size());
    for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
        auto array = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
            inputs[static_cast<py::ssize_t>(i)]);
        if (!array) {
            throw py::value_error("execute_plan: inputs must be numeric arrays");
        }
        const TensorDefinition& tensor = find_tensor(plan, plan.inputs[i].tensor_id);
        if (tensor.dtype != DataType::Float32) {
            throw py::value_error("execute_plan: inputs must be fp32");
        }
        check_array_shape(array, tensor);
        input_arrays.push_back(array);
        BufferSpec spec = spec_for(tensor, static_cast<std::size_t>(array.size()));
        input_buffers.emplace_back(array.mutable_data(), spec, /*owns=*/false);
    }

    std::vector<py::array_t<float>> output_arrays;
    output_arrays.reserve(plan.outputs.size());
    std::vector<Buffer> output_buffers;
    output_buffers.reserve(plan.outputs.size());
    for (const PlanOutput& output : plan.outputs) {
        const TensorDefinition& tensor = find_tensor(plan, output.tensor_id);
        if (tensor.dtype != DataType::Float32) {
            throw py::value_error("execute_plan: outputs must be fp32");
        }
        std::vector<py::ssize_t> shape;
        std::size_t elements = 1;
        for (const int64_t dim : tensor.shape) {
            shape.push_back(static_cast<py::ssize_t>(dim));
            elements *= static_cast<std::size_t>(dim);
        }
        output_arrays.emplace_back(shape);
        output_buffers.emplace_back(output_arrays.back().mutable_data(),
                                    spec_for(tensor, elements), /*owns=*/false);
    }

    auto provider = std::make_shared<providers::CpuProvider>();
    Scheduler scheduler({provider});
    auto result = scheduler.execute(plan, input_buffers, output_buffers);
    if (!result.ok()) {
        throw std::runtime_error("execute_plan: " + result.error().message);
    }

    py::list outputs;
    for (py::array_t<float>& array : output_arrays) {
        outputs.append(std::move(array));
    }
    return outputs;
}

}  // namespace

PYBIND11_MODULE(_sparrow_rt, module) {
    module.doc() = "Sparrow runtime bridge for the fused MLP slice";
    module.def("run_fused_mlp", &run_fused_mlp,
               "Run the fp32 fused MLP kernel through ForgeRT",
               py::arg("x"), py::arg("w"), py::arg("b"));
    module.def("execute_plan", &execute_plan,
               "Execute a version 2 CPU plan with the given input arrays",
               py::arg("plan_json"), py::arg("inputs"));
}
