// bindings.cpp - Python bindings for pufferlib (torch-free)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pufferlib.cu"

#define _PUFFER_STRINGIFY(x) #x
#define PUFFER_STRINGIFY(x) _PUFFER_STRINGIFY(x)

namespace py = pybind11;

// Wrapper functions for Python bindings
pybind11::dict puf_log(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;

    // Summary
    int gpus = pufferl.hypers.world_size;
    long global_step = pufferl.global_step;
    long epoch = pufferl.epoch;
    double now = wall_clock();
    double dt = now - pufferl.last_log_time;
    long sps = dt > 0 ? (long)((global_step - pufferl.last_log_step) / dt) : 0;
    pufferl.last_log_time = now;
    pufferl.last_log_step = global_step;

    result["SPS"] = sps * gpus;
    result["agent_steps"] = global_step * gpus;
    result["uptime"] = now - pufferl.start_time;
    result["epoch"] = epoch;

    // Environment stats
    pybind11::dict env_dict;
    Dict* env_out = log_environments_impl(pufferl);
    for (int i = 0; i < env_out->size; i++) {
        env_dict[env_out->items[i].key] = env_out->items[i].value;
    }
    result["env"] = env_dict;

    // Losses
    pybind11::dict losses_dict;
    float losses_host[NUM_LOSSES];
    cudaMemcpy(losses_host, pufferl.losses_puf.data, sizeof(losses_host), cudaMemcpyDeviceToHost);
    float n = losses_host[LOSS_N];
    if (n > 0) {
        float inv_n = 1.0f / n;
        losses_dict["policy"] = losses_host[LOSS_PG] * inv_n;
        losses_dict["value"] = losses_host[LOSS_VF] * inv_n;
        losses_dict["entropy"] = losses_host[LOSS_ENT] * inv_n;
        losses_dict["total"] = losses_host[LOSS_TOTAL] * inv_n;
        losses_dict["old_kl"] = losses_host[LOSS_OLD_APPROX_KL] * inv_n;
        losses_dict["kl"] = losses_host[LOSS_APPROX_KL] * inv_n;
        losses_dict["clipfrac"] = losses_host[LOSS_CLIPFRAC] * inv_n;
    }
    cudaMemset(pufferl.losses_puf.data, 0, numel(pufferl.losses_puf.shape) * sizeof(float));
    result["loss"] = losses_dict;

    // Profile
    pybind11::dict perf_dict;
    float train_total = 0;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = pufferl.profile.accum[i] / 1000.0f;
        perf_dict[PROF_NAMES[i]] = sec;
        if (i >= PROF_TRAIN_MISC) train_total += sec;
    }
    perf_dict["train"] = train_total;
    memset(pufferl.profile.accum, 0, sizeof(pufferl.profile.accum));
    result["perf"] = perf_dict;

    // Utilization
    pybind11::dict util_dict;
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(pufferl.nvml_device, &util);
    util_dict["gpu_percent"] = (float)util.gpu;

    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(pufferl.nvml_device, &mem);
    util_dict["gpu_mem"] = 100.0f * (float)mem.used / (float)mem.total;

    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    util_dict["vram_used_gb"] = (float)(cuda_total - cuda_free) / (1024.0f * 1024.0f * 1024.0f);
    util_dict["vram_total_gb"] = (float)cuda_total / (1024.0f * 1024.0f * 1024.0f);

    long rss_kb = 0;
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) break;
        }
        fclose(f);
    }
    util_dict["cpu_mem_gb"] = (float)rss_kb / (1024.0f * 1024.0f);
    result["util"] = util_dict;

    return result;
}

pybind11::dict puf_eval_log(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;

    double now = wall_clock();
    pufferl.last_log_time = now;
    pufferl.last_log_step = pufferl.global_step;
 
    pybind11::dict env_dict;
    Dict* env_out = create_dict(32);
    static_vec_eval_log(pufferl.vec, env_out);
    for (int i = 0; i < env_out->size; i++) {
        env_dict[env_out->items[i].key] = env_out->items[i].value;
    }
    result["env"] = env_dict;

    return result;
}

void python_vec_recv(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

void python_vec_send(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

void render(pybind11::object pufferl_obj, int env_id) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    static_vec_render(pufferl.vec, env_id);
}

void rollouts(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::gil_scoped_release no_gil;
    double t0 = wall_clock();

    // Zero state buffers
    if (pufferl.hypers.reset_state) {
        for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
            puf_zero(&pufferl.buffer_states[i], pufferl.default_stream);
        }
    }

    static_vec_omp_step(pufferl.vec);
    float sec = (float)(wall_clock() - t0);
    pufferl.profile.accum[PROF_ROLLOUT] += sec * 1000.0f;  // store as ms

    float eval_prof[NUM_EVAL_PROF];
    static_vec_read_profile(pufferl.vec, eval_prof);
    pufferl.profile.accum[PROF_EVAL_GPU] += eval_prof[EVAL_GPU];
    pufferl.profile.accum[PROF_EVAL_ENV] += eval_prof[EVAL_ENV_STEP];
    pufferl.global_step += pufferl.hypers.horizon * pufferl.hypers.total_agents;
}

pybind11::dict train(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    {
        pybind11::gil_scoped_release no_gil;
        train_impl(pufferl);
    }
    pybind11::dict losses;
    return losses;
}

void puf_close(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    close_impl(pufferl);
}

void save_weights(pybind11::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    int64_t nbytes = numel(pufferl.master_weights.shape) * sizeof(float);
    std::vector<char> buf(nbytes);
    cudaMemcpy(buf.data(), pufferl.master_weights.data, nbytes, cudaMemcpyDeviceToHost);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Failed to open " + path + " for writing");
    fwrite(buf.data(), 1, nbytes, f);
    fclose(f);
}

void load_weights(pybind11::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    int64_t nbytes = numel(pufferl.master_weights.shape) * sizeof(float);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Failed to open " + path + " for reading");
    // Verify file size matches
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size != nbytes) {
        fclose(f);
        throw std::runtime_error("Weight file size mismatch: expected " +
            std::to_string(nbytes) + " bytes, got " + std::to_string(file_size));
    }
    std::vector<char> buf(nbytes);
    size_t nread = fread(buf.data(), 1, nbytes, f);
    if ((int64_t)nread != nbytes) {
        fclose(f);
        throw std::runtime_error("Failed to read weight file");
    }
    fclose(f);
    cudaMemcpy(pufferl.master_weights.data, buf.data(), nbytes, cudaMemcpyHostToDevice);
    if (USE_BF16) {
        int n = numel(pufferl.param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl.default_stream>>>(
            pufferl.param_puf.data, pufferl.master_weights.data, n);
    }
}

void py_puff_advantage(
        long long values_ptr, long long rewards_ptr,
        long long dones_ptr,  long long importance_ptr,
        long long advantages_ptr,
        int num_steps, int horizon,
        float gamma, float lambda, float rho_clip, float c_clip) {
    constexpr int N = 16 / sizeof(precision_t);
    int blocks = grid_size(num_steps);
    auto kernel = (horizon % N == 0) ? puff_advantage : puff_advantage_scalar;
    kernel<<<blocks, 256>>>(
        (const precision_t*)values_ptr, (const precision_t*)rewards_ptr,
        (const precision_t*)dones_ptr,  (const precision_t*)importance_ptr,
        (precision_t*)advantages_ptr,
        gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

double get_config(py::dict& kwargs, const char* key) {
    if (!kwargs.contains(key)) {
        throw std::runtime_error(std::string("Missing config key: ") + key);
    }
    try {
        return kwargs[key].cast<double>();
    } catch (const py::cast_error& e) {
        throw std::runtime_error(std::string("Failed to cast config key '") + key + "': " + e.what());
    }
}

Dict* py_dict_to_c_dict(py::dict py_dict) {
    Dict* c_dict = create_dict(py_dict.size());
    for (auto item : py_dict) {
        const char* key = PyUnicode_AsUTF8(item.first.ptr());
        try {
            dict_set(c_dict, key, item.second.cast<double>());
        } catch (const py::cast_error&) {
            // Skip non-numeric values
        }
    }
    return c_dict;
}

// ============================================================================
// Python-facing VecEnv: wraps StaticVec for use from python_pufferl.py.
// After vec_step(), GPU buffers are current — Python wraps them zero-copy
// with torch.from_blob(ptr, shape, dtype, device='cuda').
// ============================================================================

struct VecEnv {
    StaticVec* vec;
    int total_agents;
    int obs_size;
    int num_atns;
    std::vector<int> act_sizes;
    std::string obs_dtype;
    size_t obs_elem_size;
    int gpu;
};

std::unique_ptr<VecEnv> create_vec(py::dict args, int gpu) {
    py::dict vec_kwargs = args["vec"].cast<py::dict>();
    py::dict env_kwargs = args["env"].cast<py::dict>();

    int total_agents = (int)get_config(vec_kwargs, "total_agents");
    int num_buffers  = (int)get_config(vec_kwargs, "num_buffers");

    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs);
    Dict* env_dict = py_dict_to_c_dict(env_kwargs);

    auto ve = std::make_unique<VecEnv>();
    ve->gpu = gpu;
    {
        py::gil_scoped_release no_gil;
        ve->vec = create_static_vec(total_agents, num_buffers, gpu, vec_dict, env_dict);
    }
    ve->total_agents  = total_agents;
    ve->obs_size      = get_obs_size();
    ve->num_atns      = get_num_atns();
    {
        int* raw = get_act_sizes();
        int  n   = get_num_act_sizes();
        ve->act_sizes = std::vector<int>(raw, raw + n);
    }
    ve->obs_dtype     = std::string(get_obs_dtype());
    ve->obs_elem_size = get_obs_elem_size();
    return ve;
}

void vec_reset(VecEnv& ve) {
    py::gil_scoped_release no_gil;
    static_vec_reset(ve.vec);
}

void gpu_vec_step_py(VecEnv& ve, long long actions_ptr) {
    cudaMemcpy(ve.vec->gpu_actions, (void*)actions_ptr,
        (size_t)ve.total_agents * ve.num_atns * sizeof(float),
        cudaMemcpyDeviceToDevice);
    {
        py::gil_scoped_release no_gil;
        gpu_vec_step(ve.vec);
    }
}

void cpu_vec_step_py(VecEnv& ve, long long actions_ptr) {
    memcpy(ve.vec->actions, (void*)actions_ptr,
        (size_t)ve.total_agents * ve.num_atns * sizeof(float));
    {
        py::gil_scoped_release no_gil;
        cpu_vec_step(ve.vec);
    }
}

py::dict vec_log(VecEnv& ve) {
    Dict* out = create_dict(32);
    static_vec_log(ve.vec, out);
    py::dict result;
    for (int i = 0; i < out->size; i++) {
        result[out->items[i].key] = out->items[i].value;
    }
    free(out->items);
    free(out);
    return result;
}

void vec_close(VecEnv& ve) {
    static_vec_close(ve.vec);
    ve.vec = nullptr;
}

std::unique_ptr<PuffeRL> create_pufferl(py::dict args) {
    py::dict train_kwargs = args["train"].cast<py::dict>();
    py::dict vec_kwargs = args["vec"].cast<py::dict>();
    py::dict env_kwargs = args["env"].cast<py::dict>();
    py::dict policy_kwargs = args["policy"].cast<py::dict>();

    HypersT hypers;
    // Layout (total_agents and num_buffers come from vec config)
    hypers.total_agents = get_config(vec_kwargs, "total_agents");
    hypers.num_buffers = get_config(vec_kwargs, "num_buffers");
    hypers.num_threads = get_config(vec_kwargs, "num_threads");
    hypers.horizon = get_config(train_kwargs, "horizon");
    // Model architecture (num_atns computed from env in C++)
    hypers.hidden_size = get_config(policy_kwargs, "hidden_size");
    hypers.num_layers = get_config(policy_kwargs, "num_layers");
    // Learning rate
    hypers.lr = get_config(train_kwargs, "learning_rate");
    hypers.min_lr_ratio = get_config(train_kwargs, "min_lr_ratio");
    hypers.anneal_lr = get_config(train_kwargs, "anneal_lr");
    // Optimizer
    hypers.beta1 = get_config(train_kwargs, "beta1");
    hypers.beta2 = get_config(train_kwargs, "beta2");
    hypers.eps = get_config(train_kwargs, "eps");
    // Training
    hypers.minibatch_size = get_config(train_kwargs, "minibatch_size");
    hypers.replay_ratio = get_config(train_kwargs, "replay_ratio");
    hypers.total_timesteps = get_config(train_kwargs, "total_timesteps");
    hypers.max_grad_norm = get_config(train_kwargs, "max_grad_norm");
    // PPO
    hypers.clip_coef = get_config(train_kwargs, "clip_coef");
    hypers.vf_clip_coef = get_config(train_kwargs, "vf_clip_coef");
    hypers.vf_coef = get_config(train_kwargs, "vf_coef");
    hypers.ent_coef = get_config(train_kwargs, "ent_coef");
    // GAE
    hypers.gamma = get_config(train_kwargs, "gamma");
    hypers.gae_lambda = get_config(train_kwargs, "gae_lambda");
    // VTrace
    hypers.vtrace_rho_clip = get_config(train_kwargs, "vtrace_rho_clip");
    hypers.vtrace_c_clip = get_config(train_kwargs, "vtrace_c_clip");
    // Priority
    hypers.prio_alpha = get_config(train_kwargs, "prio_alpha");
    hypers.prio_beta0 = get_config(train_kwargs, "prio_beta0");
    hypers.reset_state = get_config(args, "reset_state");
    // Base-level config ([base] section becomes top-level in args)
    hypers.cudagraphs = get_config(args, "cudagraphs");
    hypers.profile = get_config(args, "profile");
    // Multi-GPU / device selection
    hypers.rank = get_config(args, "rank");
    hypers.world_size = get_config(args, "world_size");
    hypers.gpu_id = get_config(args, "gpu_id");
    hypers.nccl_id = args["nccl_id"].cast<std::string>();
    // Seed
    hypers.seed = get_config(args, "seed");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    assert(device_count > 0 && "CUDA is not available");

    std::string env_name = args["env_name"].cast<std::string>();
    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs.cast<py::dict>());
    Dict* env_dict = py_dict_to_c_dict(env_kwargs.cast<py::dict>());

    std::unique_ptr<PuffeRL> pufferl;
    {
        pybind11::gil_scoped_release no_gil;
        pufferl = create_pufferl_impl(hypers, env_name, vec_dict, env_dict);
    }

    if (!pufferl) {
        throw std::runtime_error("CUDA OOM: failed to allocate training buffers");
    }

    return pufferl;
}

PYBIND11_MODULE(_C, m) {
    // Multi-GPU: generate NCCL unique ID (call on rank 0, pass bytes to all ranks)
    m.def("get_nccl_id", []() {
        ncclUniqueId id;
        ncclGetUniqueId(&id);
        return py::bytes(reinterpret_cast<char*>(&id), sizeof(id));
    });
    // Standalone utilization monitor (no PuffeRL instance needed)
    m.def("get_utilization", [](int gpu_id) {
        static bool nvml_inited = false;
        if (!nvml_inited) { nvmlInit(); nvml_inited = true; }

        py::dict util_dict;
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(gpu_id, &device);

        nvmlUtilization_t util;
        nvmlDeviceGetUtilizationRates(device, &util);
        util_dict["gpu_percent"] = (float)util.gpu;

        nvmlMemory_t mem;
        nvmlDeviceGetMemoryInfo(device, &mem);
        util_dict["gpu_mem"] = 100.0f * (float)mem.used / (float)mem.total;

        size_t cuda_free, cuda_total;
        cudaMemGetInfo(&cuda_free, &cuda_total);
        util_dict["vram_used_gb"] = (float)(cuda_total - cuda_free) / (1024.0f * 1024.0f * 1024.0f);
        util_dict["vram_total_gb"] = (float)cuda_total / (1024.0f * 1024.0f * 1024.0f);

        long rss_kb = 0;
        FILE* f = fopen("/proc/self/status", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) break;
            }
            fclose(f);
        }
        util_dict["cpu_mem_gb"] = (float)rss_kb / (1024.0f * 1024.0f);

        return util_dict;
    });

    m.attr("precision_bytes") = (int)sizeof(precision_t);
    m.attr("env_name") = PUFFER_STRINGIFY(ENV_NAME);
    m.attr("gpu") = 1;

    // Core functions
    m.def("log", &puf_log);
    m.def("eval_log", &puf_eval_log);
    m.def("render", &render);
    m.def("rollouts", &rollouts);
    m.def("train", &train);
    m.def("close", &puf_close);
    m.def("save_weights", &save_weights);
    m.def("load_weights", &load_weights);
    m.def("python_vec_recv", &python_vec_recv);
    m.def("python_vec_send", &python_vec_send);
    py::class_<Policy>(m, "Policy");
    py::class_<Muon>(m, "Muon");
    py::class_<Allocator>(m, "Allocator")
        .def(py::init<>());

    py::class_<HypersT>(m, "HypersT")
        .def_readwrite("horizon", &HypersT::horizon)
        .def_readwrite("total_agents", &HypersT::total_agents)
        .def_readwrite("num_buffers", &HypersT::num_buffers)
        .def_readwrite("num_atns", &HypersT::num_atns)
        .def_readwrite("hidden_size", &HypersT::hidden_size)

        .def_readwrite("replay_ratio", &HypersT::replay_ratio)
        .def_readwrite("num_layers", &HypersT::num_layers)
        .def_readwrite("lr", &HypersT::lr)
        .def_readwrite("min_lr_ratio", &HypersT::min_lr_ratio)
        .def_readwrite("anneal_lr", &HypersT::anneal_lr)
        .def_readwrite("beta1", &HypersT::beta1)
        .def_readwrite("beta2", &HypersT::beta2)
        .def_readwrite("eps", &HypersT::eps)
        .def_readwrite("total_timesteps", &HypersT::total_timesteps)
        .def_readwrite("max_grad_norm", &HypersT::max_grad_norm)
        .def_readwrite("clip_coef", &HypersT::clip_coef)
        .def_readwrite("vf_clip_coef", &HypersT::vf_clip_coef)
        .def_readwrite("vf_coef", &HypersT::vf_coef)
        .def_readwrite("ent_coef", &HypersT::ent_coef)
        .def_readwrite("gamma", &HypersT::gamma)
        .def_readwrite("gae_lambda", &HypersT::gae_lambda)
        .def_readwrite("vtrace_rho_clip", &HypersT::vtrace_rho_clip)
        .def_readwrite("vtrace_c_clip", &HypersT::vtrace_c_clip)
        .def_readwrite("prio_alpha", &HypersT::prio_alpha)
        .def_readwrite("prio_beta0", &HypersT::prio_beta0)
        .def_readwrite("cudagraphs", &HypersT::cudagraphs)
        .def_readwrite("profile", &HypersT::profile)
        .def_readwrite("rank", &HypersT::rank)
        .def_readwrite("world_size", &HypersT::world_size)
        .def_readwrite("gpu_id", &HypersT::gpu_id)
        .def_readwrite("nccl_id", &HypersT::nccl_id);

    py::class_<PrecisionTensor>(m, "PrecisionTensor")
        .def("__repr__", [](const PrecisionTensor& t) { return std::string(puf_repr(&t)); })
        .def("ndim", [](const PrecisionTensor& t) { return ndim(t.shape); })
        .def("numel", [](const PrecisionTensor& t) { return numel(t.shape); });
    py::class_<FloatTensor>(m, "FloatTensor")
        .def("__repr__", [](const FloatTensor& t) { return std::string(puf_repr(&t)); })
        .def("ndim", [](const FloatTensor& t) { return ndim(t.shape); })
        .def("numel", [](const FloatTensor& t) { return numel(t.shape); });

    py::class_<RolloutBuf>(m, "RolloutBuf")
        .def_readwrite("observations", &RolloutBuf::observations)
        .def_readwrite("actions", &RolloutBuf::actions)
        .def_readwrite("values", &RolloutBuf::values)
        .def_readwrite("logprobs", &RolloutBuf::logprobs)
        .def_readwrite("rewards", &RolloutBuf::rewards)
        .def_readwrite("terminals", &RolloutBuf::terminals)
        .def_readwrite("ratio", &RolloutBuf::ratio)
        .def_readwrite("importance", &RolloutBuf::importance);

    m.def("uptime", [](py::object pufferl_obj) -> double {
        PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
        double now = wall_clock();
        return now - pufferl.start_time;
    });
    m.def("puff_advantage", &py_puff_advantage);
    m.def("create_vec", &create_vec, py::arg("args"), py::arg("gpu") = 1);
    py::class_<VecEnv, std::unique_ptr<VecEnv>>(m, "VecEnv")
        .def_readonly("total_agents",  &VecEnv::total_agents)
        .def_readonly("obs_size",      &VecEnv::obs_size)
        .def_readonly("num_atns",      &VecEnv::num_atns)
        .def_readonly("act_sizes",     &VecEnv::act_sizes)
        .def_readonly("obs_dtype",     &VecEnv::obs_dtype)
        .def_readonly("obs_elem_size", &VecEnv::obs_elem_size)
        .def_readonly("gpu",           &VecEnv::gpu)
        // GPU buffer pointers — wrap with torch.from_blob(..., device='cuda')
        .def_property_readonly("gpu_obs_ptr",       [](VecEnv& ve) { return (long long)ve.vec->gpu_observations; })
        .def_property_readonly("gpu_rewards_ptr",   [](VecEnv& ve) { return (long long)ve.vec->gpu_rewards; })
        .def_property_readonly("gpu_terminals_ptr", [](VecEnv& ve) { return (long long)ve.vec->gpu_terminals; })
        // CPU buffer pointers (same as gpu_ in CPU mode since they alias)
        .def_property_readonly("obs_ptr",       [](VecEnv& ve) { return (long long)ve.vec->observations; })
        .def_property_readonly("rewards_ptr",   [](VecEnv& ve) { return (long long)ve.vec->rewards; })
        .def_property_readonly("terminals_ptr", [](VecEnv& ve) { return (long long)ve.vec->terminals; })
        .def("reset", &vec_reset)
        .def("gpu_step", &gpu_vec_step_py)
        .def("cpu_step", &cpu_vec_step_py)
        .def("render", [](VecEnv& ve, int env_id) { static_vec_render(ve.vec, env_id); })
        .def("log",   &vec_log)
        .def("close", &vec_close);

    m.def("create_pufferl", &create_pufferl);
    py::class_<PuffeRL, std::unique_ptr<PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy", &PuffeRL::policy)
        .def_readwrite("muon", &PuffeRL::muon)
        .def_readwrite("hypers", &PuffeRL::hypers)
        .def_readwrite("rollouts", &PuffeRL::rollouts)
        .def_readonly("epoch", &PuffeRL::epoch)
        .def_readonly("global_step", &PuffeRL::global_step)
        .def_readonly("last_log_time", &PuffeRL::last_log_time)
        .def("num_params", [](PuffeRL& self) -> int64_t {
            return numel(self.master_weights.shape);
        });
}
