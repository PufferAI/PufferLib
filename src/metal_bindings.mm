#include "metal_pufferlib.mm"

#include <chrono>
#include <mach/mach.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <sys/time.h>

namespace py = pybind11;

#define _PUFFER_STRINGIFY(x) #x
#define PUFFER_STRINGIFY(x) _PUFFER_STRINGIFY(x)

static double wall_clock() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static std::string tensor_repr(const FloatTensor& tensor) {
    char buffer[256];
    int position = snprintf(buffer, sizeof(buffer), "FloatTensor([");
    int dims = puf_ndim(tensor.shape);
    for (int i = 0; i < dims && position < (int)sizeof(buffer) - 32; i++) {
        position += snprintf(
            buffer + position,
            sizeof(buffer) - position,
            "%s%lld",
            i ? ", " : "",
            (long long)tensor.shape[i]);
    }
    snprintf(
        buffer + position,
        sizeof(buffer) - position,
        "], %lld elems)",
        (long long)puf_numel(tensor.shape));
    return std::string(buffer);
}

static py::dict get_utilization(int gpu_id) {
    (void)gpu_id;
    py::dict result;

    MetalContext* ctx = mtl_ctx();
    if (ctx->device) {
        uint64_t gpu_budget = [ctx->device recommendedMaxWorkingSetSize];
        uint64_t gpu_current = [ctx->device currentAllocatedSize];
        result["gpu_percent"] = 0.0f;
        if (gpu_budget > 0) {
            result["gpu_mem"] = 100.0f * (float)gpu_current / (float)gpu_budget;
        } else {
            result["gpu_mem"] = 0.0f;
        }
        result["vram_used_gb"] = (float)gpu_current / (1024.0f * 1024.0f * 1024.0f);
        result["vram_total_gb"] = (float)gpu_budget / (1024.0f * 1024.0f * 1024.0f);
    }

    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
            (task_info_t)&info, &count) == KERN_SUCCESS) {
        result["cpu_mem_gb"] = (float)info.resident_size / (1024.0f * 1024.0f * 1024.0f);
    }

    return result;
}

static py::dict puf_log(py::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    py::dict result;

    if (pufferl.train_pending) {
        auto wait_start = std::chrono::high_resolution_clock::now();
        sync_pending_train(pufferl);
        float wait_ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - wait_start).count();
        pufferl.profile.accum[PROF_TRAIN_SYNC] += wait_ms;
    }
    mtl_ensure_stream_synced((cudaStream_t)mtl_stream());

    long global_step = pufferl.global_step;
    int epoch = pufferl.epoch;
    double now = wall_clock();
    double dt = now - pufferl.last_log_time;
    long sps = dt > 0 ? (long)((global_step - pufferl.last_log_step) / dt) : 0;
    pufferl.last_log_time = now;
    pufferl.last_log_step = global_step;

    result["SPS"] = sps;
    result["agent_steps"] = global_step;
    result["uptime"] = now - pufferl.start_time;
    result["epoch"] = epoch;

    py::dict env_dict;
    Dict* env_out = log_environments_impl(pufferl);
    for (int i = 0; i < env_out->size; i++) {
        env_dict[env_out->items[i].key] = env_out->items[i].value;
    }
    result["env"] = env_dict;

    py::dict loss_dict;
    float* losses = pufferl.losses_puf.data;
    float n = losses[LOSS_N];
    if (n > 0) {
        float inv_n = 1.0f / n;
        loss_dict["policy"] = losses[LOSS_PG] * inv_n;
        loss_dict["value"] = losses[LOSS_VF] * inv_n;
        loss_dict["entropy"] = losses[LOSS_ENT] * inv_n;
        loss_dict["total"] = losses[LOSS_TOTAL] * inv_n;
        loss_dict["old_kl"] = losses[LOSS_OLD_APPROX_KL] * inv_n;
        loss_dict["kl"] = losses[LOSS_APPROX_KL] * inv_n;
        loss_dict["clipfrac"] = losses[LOSS_CLIPFRAC] * inv_n;
    }
    cudaStream_t loss_stream = pufferl.overlap_enabled
        ? (cudaStream_t)mtl_train_stream()
        : (cudaStream_t)mtl_stream();
    mtl_fill_f32(losses, 0.0f, (int)puf_numel(pufferl.losses_puf.shape), loss_stream);
    result["loss"] = loss_dict;

    py::dict perf_dict;
    float train_ms = 0.0f;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = pufferl.profile.accum[i] / 1000.0f;
        perf_dict[PROF_NAMES[i]] = sec;
        if (i >= PROF_TRAIN_PRELOOP) {
            train_ms += pufferl.profile.accum[i];
        }
    }
    perf_dict["train"] = train_ms / 1000.0f;
    memset(pufferl.profile.accum, 0, sizeof(pufferl.profile.accum));
    pufferl.rollout_sync_count = 0;
    pufferl.rollout_sync_ms = 0;
    pufferl.train_sync_count = 0;
    pufferl.train_sync_ms = 0;
    result["perf"] = perf_dict;

    result["util"] = get_utilization(0);
    return result;
}

static py::dict puf_eval_log(py::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    py::dict result;

    double now = wall_clock();
    pufferl.last_log_time = now;
    pufferl.last_log_step = pufferl.global_step;

    py::dict env_dict;
    Dict* env_out = create_dict(32);
    static_vec_eval_log(pufferl.vec, env_out);
    for (int i = 0; i < env_out->size; i++) {
        env_dict[env_out->items[i].key] = env_out->items[i].value;
    }
    result["env"] = env_dict;
    return result;
}

static void render(py::object pufferl_obj, int env_id) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    static_vec_render(pufferl.vec, env_id);
}

static void rollouts(py::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();

    { int count; double ms; mtl_sync_stats(&count, &ms); }

    py::gil_scoped_release no_gil;

    if (pufferl.train_pending) {
        auto wait_start = std::chrono::high_resolution_clock::now();
        sync_pending_train(pufferl);
        float wait_ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - wait_start).count();
        pufferl.profile.accum[PROF_TRAIN_SYNC] += wait_ms;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!pufferl.cpu_inference) puf_set_gpu_training(true);
    static_vec_omp_step(pufferl.vec);
    if (!pufferl.cpu_inference) puf_set_gpu_training(false);
    float sec = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0).count();
    pufferl.profile.accum[PROF_ROLLOUT] += sec * 1000.0f;

    float eval_prof[NUM_EVAL_PROF];
    static_vec_read_profile(pufferl.vec, eval_prof);
    pufferl.profile.accum[PROF_EVAL_GPU] += eval_prof[EVAL_GPU];
    pufferl.profile.accum[PROF_EVAL_ENV] += eval_prof[EVAL_ENV_STEP];

    mtl_sync_stats(&pufferl.rollout_sync_count, &pufferl.rollout_sync_ms);
    pufferl.global_step += pufferl.hypers.horizon * pufferl.hypers.total_agents;
}

static py::dict train(py::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    { int count; double ms; mtl_sync_stats(&count, &ms); }

    {
        py::gil_scoped_release no_gil;
        train_impl(pufferl);
    }

    mtl_sync_stats(&pufferl.train_sync_count, &pufferl.train_sync_ms);
    return py::dict();
}

static void puf_close(py::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    close_impl(pufferl);
}

static void save_weights(py::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    int64_t nbytes = pufferl.alloc_fp32.params.total_elems * sizeof(float);

    sync_pending_train(pufferl);
    mtl_ensure_stream_synced((cudaStream_t)mtl_stream());

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("Failed to open " + path + " for writing");
    }
    fwrite(pufferl.alloc_fp32.params.mem, 1, nbytes, f);
    fclose(f);
}

static void load_weights(py::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    int64_t param_count = pufferl.alloc_fp32.params.total_elems;
    int64_t nbytes = param_count * sizeof(float);

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Failed to open " + path + " for reading");
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size != nbytes) {
        fclose(f);
        throw std::runtime_error(
            "Weight file size mismatch: expected " + std::to_string(nbytes)
            + " bytes, got " + std::to_string(file_size));
    }

    sync_pending_train(pufferl);
    mtl_ensure_stream_synced((cudaStream_t)mtl_stream());

    size_t nread = fread(pufferl.alloc_fp32.params.mem, 1, nbytes, f);
    if ((int64_t)nread != nbytes) {
        fclose(f);
        throw std::runtime_error("Failed to read weight file");
    }
    fclose(f);

    copy_weights_to_infer(pufferl);
    if (pufferl.train_fp16) {
        cudaStream_t stream = (cudaStream_t)mtl_stream();
        mtl_cast_f32_to_f16(
            pufferl.param_fp16_puf.bytes,
            (const float*)pufferl.alloc_fp32.params.mem,
            (int)param_count,
            stream);
        mtl_ensure_stream_synced(stream);
    }
}

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
        int offset = row * horizon;
        float last = 0.0f;
        for (int t = horizon - 2; t >= 0; t--) {
            int next_t = t + 1;
            float next_nonterminal = 1.0f - dones[offset + next_t];
            float imp = importance[offset + t];
            float rho_t = imp < rho_clip ? imp : rho_clip;
            float c_t = imp < c_clip ? imp : c_clip;
            float delta = rho_t * rewards[offset + next_t]
                + gamma * values[offset + next_t] * next_nonterminal
                - values[offset + t];
            last = delta + gamma * lambda * c_t * last * next_nonterminal;
            advantages[offset + t] = last;
        }
    }
}

static double get_config(py::dict& kwargs, const char* key) {
    assert(kwargs.contains(key) && "Missing config key");
    return kwargs[key].cast<double>();
}

static Dict* py_dict_to_c_dict(py::dict py_dict) {
    Dict* c_dict = create_dict(py_dict.size());
    for (auto item : py_dict) {
        const char* key = PyUnicode_AsUTF8(item.first.ptr());
        try {
            dict_set(c_dict, key, item.second.cast<double>());
        } catch (const py::cast_error&) {
        }
    }
    return c_dict;
}

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
    memcpy(
        ve.vec->actions,
        (void*)actions_ptr,
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
    for (int i = 0; i < out->size; i++) {
        result[out->items[i].key] = out->items[i].value;
    }
    free(out->items);
    free(out);
    return result;
}

static void vec_close(VecEnv& ve) {
    static_vec_close(ve.vec);
    ve.vec = nullptr;
}

static std::unique_ptr<PuffeRL> create_pufferl(py::dict args) {
    py::dict train_kwargs = args["train"].cast<py::dict>();
    py::dict vec_kwargs = args["vec"].cast<py::dict>();
    py::dict env_kwargs = args["env"].cast<py::dict>();
    py::dict policy_kwargs = args["policy"].cast<py::dict>();

    HypersT hypers;
    hypers.total_agents = get_config(vec_kwargs, "total_agents");
    hypers.num_buffers = get_config(vec_kwargs, "num_buffers");
    hypers.num_threads = get_config(vec_kwargs, "num_threads");
    hypers.horizon = get_config(train_kwargs, "horizon");
    hypers.hidden_size = get_config(policy_kwargs, "hidden_size");
    hypers.num_layers = get_config(policy_kwargs, "num_layers");
    hypers.seed = args.contains("seed") ? (uint64_t)get_config(args, "seed")
        : train_kwargs.contains("seed") ? (uint64_t)get_config(train_kwargs, "seed") : 42;
    hypers.lr = get_config(train_kwargs, "learning_rate");
    hypers.min_lr_ratio = get_config(train_kwargs, "min_lr_ratio");
    hypers.anneal_lr = get_config(train_kwargs, "anneal_lr");
    hypers.beta1 = get_config(train_kwargs, "beta1");
    hypers.minibatch_size = get_config(train_kwargs, "minibatch_size");
    hypers.replay_ratio = get_config(train_kwargs, "replay_ratio");
    hypers.total_timesteps = get_config(train_kwargs, "total_timesteps");
    hypers.max_grad_norm = get_config(train_kwargs, "max_grad_norm");
    hypers.clip_coef = get_config(train_kwargs, "clip_coef");
    hypers.vf_clip_coef = get_config(train_kwargs, "vf_clip_coef");
    hypers.vf_coef = get_config(train_kwargs, "vf_coef");
    hypers.ent_coef = get_config(train_kwargs, "ent_coef");
    hypers.gamma = get_config(train_kwargs, "gamma");
    hypers.gae_lambda = get_config(train_kwargs, "gae_lambda");
    hypers.vtrace_rho_clip = get_config(train_kwargs, "vtrace_rho_clip");
    hypers.vtrace_c_clip = get_config(train_kwargs, "vtrace_c_clip");
    hypers.prio_alpha = get_config(train_kwargs, "prio_alpha");
    hypers.prio_beta0 = get_config(train_kwargs, "prio_beta0");
    hypers.reset_state =
        (args.contains("reset_state") && get_config(args, "reset_state") > 0) ||
        (train_kwargs.contains("reset_state") && get_config(train_kwargs, "reset_state") > 0);
    hypers.profile = train_kwargs.contains("profile") ? get_config(train_kwargs, "profile")
        : args.contains("profile") ? get_config(args, "profile") : 0;
    hypers.overlap =
        (train_kwargs.contains("overlap") && get_config(train_kwargs, "overlap") > 0) ||
        (args.contains("overlap") && get_config(args, "overlap") > 0);
    hypers.cpu_inference =
        (train_kwargs.contains("cpu_inference") && get_config(train_kwargs, "cpu_inference") > 0) ||
        (args.contains("cpu_inference") && get_config(args, "cpu_inference") > 0);
    hypers.train_fp16 =
        (train_kwargs.contains("train_fp16") && get_config(train_kwargs, "train_fp16") > 0) ||
        (args.contains("train_fp16") && get_config(args, "train_fp16") > 0);
    hypers.gpu_id = args.contains("gpu_id") ? (int)get_config(args, "gpu_id") : 0;

    mtl_enable_gpu_timing(hypers.profile);

    std::string env_name = args["env_name"].cast<std::string>();
    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs);
    Dict* env_dict = py_dict_to_c_dict(env_kwargs);

    std::unique_ptr<PuffeRL> pufferl;
    {
        py::gil_scoped_release no_gil;
        pufferl = create_pufferl_impl(hypers, env_name, vec_dict, env_dict);
    }

    return pufferl;
}

PYBIND11_MODULE(_C, m) {
    m.def("get_nccl_id", []() -> py::bytes {
        throw std::runtime_error("Metal backend does not support multi-GPU");
    });
    m.def("get_utilization", &get_utilization);

    m.attr("precision_bytes") = 4;
    m.attr("env_name") = PUFFER_STRINGIFY(ENV_NAME);
    m.attr("gpu") = 0;

    m.def("log", &puf_log);
    m.def("eval_log", &puf_eval_log);
    m.def("render", &render);
    m.def("rollouts", &rollouts);
    m.def("train", &train);
    m.def("close", &puf_close);
    m.def("save_weights", &save_weights);
    m.def("load_weights", &load_weights);
    m.def("uptime", [](py::object pufferl_obj) -> double {
        PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
        return wall_clock() - pufferl.start_time;
    });

    m.def("puff_advantage_cpu", &py_puff_advantage_cpu);
    m.def("create_vec", &create_vec, py::arg("args"), py::arg("gpu") = 0);

    py::class_<Policy>(m, "Policy");
    py::class_<Muon>(m, "Muon");
    py::class_<Allocator>(m, "Allocator").def(py::init<>());

    py::class_<HypersT>(m, "HypersT")
        .def_readwrite("horizon", &HypersT::horizon)
        .def_readwrite("total_agents", &HypersT::total_agents)
        .def_readwrite("num_buffers", &HypersT::num_buffers)
        .def_readwrite("num_atns", &HypersT::num_atns)
        .def_readwrite("hidden_size", &HypersT::hidden_size)
        .def_readwrite("replay_ratio", &HypersT::replay_ratio)
        .def_readwrite("num_layers", &HypersT::num_layers)
        .def_readwrite("seed", &HypersT::seed)
        .def_readwrite("lr", &HypersT::lr)
        .def_readwrite("min_lr_ratio", &HypersT::min_lr_ratio)
        .def_readwrite("anneal_lr", &HypersT::anneal_lr)
        .def_readwrite("beta1", &HypersT::beta1)
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
        .def_readwrite("reset_state", &HypersT::reset_state)
        .def_readwrite("profile", &HypersT::profile)
        .def_readwrite("overlap", &HypersT::overlap)
        .def_readwrite("cpu_inference", &HypersT::cpu_inference)
        .def_readwrite("train_fp16", &HypersT::train_fp16)
        .def_readwrite("gpu_id", &HypersT::gpu_id);

    py::class_<FloatTensor>(m, "FloatTensor")
        .def("__repr__", [](const FloatTensor& t) { return tensor_repr(t); })
        .def("ndim", [](const FloatTensor& t) { return puf_ndim(t.shape); })
        .def("numel", [](const FloatTensor& t) { return puf_numel(t.shape); });
    m.attr("PrecisionTensor") = m.attr("FloatTensor");

    py::class_<RolloutBuf>(m, "RolloutBuf")
        .def_readwrite("observations", &RolloutBuf::observations)
        .def_readwrite("actions", &RolloutBuf::actions)
        .def_readwrite("values", &RolloutBuf::values)
        .def_readwrite("logprobs", &RolloutBuf::logprobs)
        .def_readwrite("rewards", &RolloutBuf::rewards)
        .def_readwrite("terminals", &RolloutBuf::terminals)
        .def_readwrite("ratio", &RolloutBuf::ratio)
        .def_readwrite("importance", &RolloutBuf::importance);

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
            return self.alloc_fp32.params.total_elems;
        });
}
