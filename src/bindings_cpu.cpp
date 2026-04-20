// bindings_cpu.cpp - CPU-only Python bindings (no nvcc/CUDA required)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define _PUFFER_STRINGIFY(x) #x
#define PUFFER_STRINGIFY(x) _PUFFER_STRINGIFY(x)
#include <cstring>

// vecenv.h header section gives us StaticVec, Dict, cudaStream_t typedef
#include "vecenv.h"

namespace py = pybind11;

// Stub out CUDA functions that the static lib references (dead code when gpu=0)
extern "C" {
typedef int cudaError_t;
typedef int cudaMemcpyKind;
cudaError_t cudaHostAlloc(void**, size_t, unsigned int) { return 0; }
cudaError_t cudaMalloc(void**, size_t) { return 0; }
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { return 0; }
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return 0; }
cudaError_t cudaMemset(void*, int, size_t) { return 0; }
cudaError_t cudaFree(void*) { return 0; }
cudaError_t cudaFreeHost(void*) { return 0; }
cudaError_t cudaSetDevice(int) { return 0; }
cudaError_t cudaDeviceSynchronize(void) { return 0; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned int) { return 0; }
cudaError_t cudaStreamQuery(cudaStream_t) { return 0; }
const char* cudaGetErrorString(cudaError_t) { return "stub"; }
}

// ============================================================================
// CPU advantage (same as puff_advantage_row_scalar but plain C++)
// ============================================================================

static void py_puff_advantage_cpu(
        long long values_ptr, long long rewards_ptr,
        long long dones_ptr, long long importance_ptr,
        long long advantages_ptr,
        int num_steps, int horizon,
        float gamma, float lambda, float rho_clip, float c_clip) {
    const float* values = (const float*)values_ptr;
    const float* rewards = (const float*)rewards_ptr;
    const float* dones = (const float*)dones_ptr;
    const float* importance = (const float*)importance_ptr;
    float* advantages = (float*)advantages_ptr;
    for (int row = 0; row < num_steps; row++) {
        int off = row * horizon;
        float lastpufferlam = 0;
        for (int t = horizon - 2; t >= 0; t--) {
            int t_next = t + 1;
            float nextnonterminal = 1.0f - dones[off + t_next];
            float imp = importance[off + t];
            float rho_t = imp < rho_clip ? imp : rho_clip;
            float c_t = imp < c_clip ? imp : c_clip;
            float r_nxt = rewards[off + t_next];
            float v = values[off + t];
            float v_nxt = values[off + t_next];
            float delta = rho_t * r_nxt + gamma * v_nxt * nextnonterminal - v;
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            advantages[off + t] = lastpufferlam;
        }
    }
}

// ============================================================================
// Dict helpers (same as bindings.cu)
// ============================================================================

static double get_config(py::dict& kwargs, const char* key) {
    if (!kwargs.contains(key))
        throw std::runtime_error(std::string("Missing config key: ") + key);
    return kwargs[key].cast<double>();
}

static Dict* py_dict_to_c_dict(py::dict py_dict) {
    Dict* c_dict = create_dict(py_dict.size());
    for (auto item : py_dict) {
        const char* key = PyUnicode_AsUTF8(item.first.ptr());
        try { dict_set(c_dict, key, item.second.cast<double>()); }
        catch (const py::cast_error&) {}
    }
    return c_dict;
}

// ============================================================================
// VecEnv wrapper
// ============================================================================

struct VecEnv {
    StaticVec* vec;
    int total_agents;
    int obs_size;
    int num_atns;
    std::vector<int> act_sizes;
    std::string obs_dtype;
    size_t obs_elem_size;
};

static std::unique_ptr<VecEnv> create_vec(py::dict args, int gpu = 0) {
    (void)gpu;
    py::dict vec_kwargs = args["vec"].cast<py::dict>();
    py::dict env_kwargs = args["env"].cast<py::dict>();
    int total_agents = (int)get_config(vec_kwargs, "total_agents");
    int num_buffers = (int)get_config(vec_kwargs, "num_buffers");
    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs);
    Dict* env_dict = py_dict_to_c_dict(env_kwargs);

    auto ve = std::make_unique<VecEnv>();
    {
        py::gil_scoped_release no_gil;
        ve->vec = create_static_vec(total_agents, num_buffers, 0, vec_dict, env_dict);
    }
    ve->total_agents = total_agents;
    ve->obs_size = get_obs_size();
    ve->num_atns = get_num_atns();
    {
        int* raw = get_act_sizes();
        int n = get_num_act_sizes();
        ve->act_sizes = std::vector<int>(raw, raw + n);
    }
    ve->obs_dtype = std::string(get_obs_dtype());
    ve->obs_elem_size = get_obs_elem_size();
    return ve;
}

static void vec_reset(VecEnv& ve) {
    py::gil_scoped_release no_gil;
    static_vec_reset(ve.vec);
}

static void cpu_vec_step_py(VecEnv& ve, long long actions_ptr) {
    memcpy(ve.vec->actions, (void*)actions_ptr,
        (size_t)ve.total_agents * ve.num_atns * sizeof(float));
    {
        py::gil_scoped_release no_gil;
        cpu_vec_step(ve.vec);
    }
}

static py::dict vec_log(VecEnv& ve) {
    Dict* out = create_dict(32);
    static_vec_log(ve.vec, out);
    py::dict result;
    for (int i = 0; i < out->size; i++)
        result[out->items[i].key] = out->items[i].value;
    free(out->items);
    free(out);
    return result;
}

static void vec_close(VecEnv& ve) {
    static_vec_close(ve.vec);
    ve.vec = nullptr;
}

// ============================================================================
// Module
// ============================================================================

PYBIND11_MODULE(_C, m) {
    m.attr("precision_bytes") = 4;
    m.attr("env_name") = PUFFER_STRINGIFY(ENV_NAME);
    m.attr("gpu") = 0;

    m.def("puff_advantage_cpu", &py_puff_advantage_cpu);
    m.def("create_vec", &create_vec, py::arg("args"), py::arg("gpu") = 0);

    py::class_<VecEnv, std::unique_ptr<VecEnv>>(m, "VecEnv")
        .def_readonly("total_agents", &VecEnv::total_agents)
        .def_readonly("obs_size", &VecEnv::obs_size)
        .def_readonly("num_atns", &VecEnv::num_atns)
        .def_readonly("act_sizes", &VecEnv::act_sizes)
        .def_readonly("obs_dtype", &VecEnv::obs_dtype)
        .def_readonly("obs_elem_size", &VecEnv::obs_elem_size)
        .def_property_readonly("gpu", [](VecEnv&) { return 0; })
        .def_property_readonly("obs_ptr", [](VecEnv& ve) { return (long long)ve.vec->observations; })
        .def_property_readonly("rewards_ptr", [](VecEnv& ve) { return (long long)ve.vec->rewards; })
        .def_property_readonly("terminals_ptr", [](VecEnv& ve) { return (long long)ve.vec->terminals; })
        .def("reset", &vec_reset)
        .def("cpu_step", &cpu_vec_step_py)
        .def("render", [](VecEnv& ve, int env_id) { static_vec_render(ve.vec, env_id); })
        .def("log", &vec_log)
        .def("close", &vec_close);
}
