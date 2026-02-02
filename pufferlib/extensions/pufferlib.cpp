//TODO:clamped
//5.6% cat overhead from grad clip. Preallocate?
//11% seqwise overhead from fused scan
//30% elemwise form random ops
//5% on log_coeffs_and_values

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <atomic>
#include "vecenv.h"
#include <dlfcn.h>
#include "muon.h"

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
//#include <c10/cuda/CUDAGuard.h>

#include <nvToolsExt.h>

#include <functional>
#include <iostream>
#include <vector>

typedef torch::Tensor Tensor;

create_environments_fn create_envs;
create_threads_fn create_threads;
env_init_fn env_init;
vec_reset_fn vec_reset;
vec_step_fn vec_step;
vec_send_fn vec_send;
vec_recv_fn vec_recv;
env_close_fn env_close;
vec_close_fn vec_close;
vec_log_fn vec_log;
vec_render_fn vec_render;

torch::Dtype to_torch_dtype(int dtype) {
    if (dtype == FLOAT) {
        return torch::kFloat32;
    } else if (dtype == INT) {
        return torch::kInt32;
    } else if (dtype == UNSIGNED_CHAR) {
        return torch::kUInt8;
    } else if (dtype == DOUBLE) {
        return torch::kFloat64;
    } else {
        assert(false && "to_torch_dtype failed to convert dtype");
    }
    return torch::kFloat32;
}

// Torch is stupid. Had to clip out a redundant cuda sync.
void clip_grad_norm_(
    const std::vector<Tensor>& parameters,
    double max_norm,
    double norm_type = 2.0
    ) {
  std::vector<Tensor> params_with_grad;

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(param);
    }
  }

  if (params_with_grad.empty()) {
    return;
  }

  Tensor total_norm_tensor;
  if (norm_type == std::numeric_limits<double>::infinity()) {
    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().abs().max());
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::max(torch::stack(norms));
  } else if (norm_type == 0) {
    total_norm_tensor =
        torch::full({}, static_cast<double>(params_with_grad.size()));
  } else {
    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().norm(norm_type));
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(norm_type);
  }

  Tensor clip_coef = max_norm / (total_norm_tensor + 1e-6);
  Tensor clip_coef_clamped =
      torch::clamp(clip_coef, std::nullopt /* min */, 1.0 /* max */);
  for (auto& param : params_with_grad) {
    param.grad().data().mul_(clip_coef_clamped);
  }
}

float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;  // avoid division by zero
    float ratio = static_cast<float>(t) / static_cast<float>(T);
    ratio = std::max(0.0f, std::min(1.0f, ratio));  // clamp to [0, 1]
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}


std::tuple<VecEnv*, Tensor, Tensor, Tensor, Tensor>
create_environments(int64_t num_envs, const std::string& env_name, Dict* env_kwargs) {
    // Load the function pointer
    create_envs = (create_environments_fn)dlsym(handle, "create_environments");
    create_threads = (create_threads_fn)dlsym(handle, "create_threads");
    env_init = (env_init_fn)dlsym(handle, "env_init");
    vec_reset = (vec_reset_fn)dlsym(handle, "vec_reset");
    vec_step = (vec_step_fn)dlsym(handle, "vec_step");
    vec_send = (vec_send_fn)dlsym(handle, "vec_send");
    vec_recv = (vec_recv_fn)dlsym(handle, "vec_recv");
    env_close = (env_close_fn)dlsym(handle, "env_close");
    vec_close = (vec_close_fn)dlsym(handle, "vec_close");
    vec_log = (vec_log_fn)dlsym(handle, "vec_log");
    vec_render = (vec_render_fn)dlsym(handle, "vec_render");
    int obs_n = *(int*)dlsym(handle, "OBS_N");
    int act_n = *(int*)dlsym(handle, "ACT_N");
    int obs_t = *(int*)dlsym(handle, "OBS_T");
    int act_t = *(int*)dlsym(handle, "ACT_T");

    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "dlsym error: %s\n", dlsym_error);
        dlclose(handle);
        exit(1);
    }

    VecEnv* vec = create_envs(num_envs, 2, true, 0, env_kwargs);
    printf("Created VecEnv with %d environments\n", vec->size);

    // Close the library
    //dlclose(handle);
 
    auto obs_dtype = to_torch_dtype(obs_t);
    auto atn_dtype = to_torch_dtype(act_t);

    Tensor obs = torch::from_blob(vec->gpu_observations, {num_envs, obs_n}, torch::dtype(obs_dtype).device(torch::kCUDA));
    Tensor actions = torch::from_blob(vec->gpu_actions, {num_envs}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    Tensor rewards = torch::from_blob(vec->gpu_rewards, {num_envs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    Tensor terminals = torch::from_blob(vec->gpu_terminals, {num_envs}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // TODO: RESET
    return std::make_tuple(vec, obs, actions, rewards, terminals);
}

// CUDA kernel wrappers
#include "modules.cpp"

auto DTYPE = torch::kFloat32;

namespace pufferlib {

// Advantage computation is in advantage.cpp
#include "advantage.cpp"

// Model classes are in models.cpp
#include "models.cpp"

typedef struct {
    Tensor obs;
    Tensor actions;
    Tensor rewards;
    Tensor terminals;
} EnvBuf;

typedef struct {
    // Rollout tensors
    Tensor obs;
    Tensor actions;
    Tensor state;
    Tensor state_out;
    Tensor value;
    Tensor logprobs;
    // Train tensors
    Tensor mb_obs;
    Tensor mb_state;
    Tensor mb_actions;
    Tensor mb_logprobs;
    Tensor mb_advantages;
    Tensor mb_prio;
    Tensor mb_values;
    Tensor mb_returns;
    Tensor mb_ratio;
    Tensor mb_newvalue;
} GraphBuf;

typedef struct {
    Tensor observations;
    Tensor actions;
    Tensor values;
    Tensor logprobs;
    Tensor rewards;
    Tensor terminals;
    Tensor ratio;
    Tensor importance;
} RolloutBuf;

typedef struct {
    // Layout
    int segments;
    int horizon;
    int num_envs;
    int num_buffers;
    int minibatch_segments;
    int total_minibatches;
    int accumulate_minibatches;
    // Model architecture
    int num_atns;
    int hidden_size;
    int expansion_factor;
    int num_layers;
    // Learning rate
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    // Optimizer
    float beta1;
    float beta2;
    float eps;
    // Training
    int max_epochs;
    float max_grad_norm;
    // PPO
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    // GAE
    float gamma;
    float gae_lambda;
    // VTrace
    float vtrace_rho_clip;
    float vtrace_c_clip;
    // Priority
    float prio_alpha;
    float prio_beta0;
    // Flags
    bool use_rnn;
    bool cudagraphs;
    bool kernels;
    bool profile;
} HypersT;

typedef struct {
    PolicyMinGRU* policy;
    VecEnv* vec;
    torch::optim::Muon* muon;
    HypersT hypers;
    std::vector<Tensor> buffer_states;  // Per-buffer states for contiguous access
    RolloutBuf rollouts;
    EnvBuf env;
    GraphBuf graph;
    at::cuda::CUDAGraph rollout_graph;
    at::cuda::CUDAGraph train_forward_graph;
    at::cuda::CUDAGraph rollout_copy_graphs[64][2];
    bool captured;
    Tensor adv_mean;
    Tensor adv_std;
    int epoch;
    uint64_t rng_seed;
    Tensor rng_offset;  // CUDA tensor so increment is graphable
} PuffeRL;

RolloutBuf create_rollouts(int horizon, int segments, int input_size) {
    RolloutBuf r;
    r.observations = torch::zeros({horizon, segments, input_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.actions = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    r.values = torch::zeros({horizon, segments}, torch::dtype(DTYPE).device(torch::kCUDA));
    r.logprobs = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.rewards = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.terminals = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.ratio = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.importance = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return r;
}

EnvBuf create_env(int num_envs, int input_size) {
    EnvBuf e;
    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    e.obs = torch::zeros({num_envs, input_size}, opts);
    e.actions = torch::zeros({num_envs}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    e.rewards = torch::zeros({num_envs}, opts);
    e.terminals = torch::zeros({num_envs}, opts);
    return e;
}

GraphBuf create_graph(int batch, int input_size, int minibatch_segments, int horizon,
        int num_layers, int hidden_size, int expansion_factor, PolicyMinGRU* policy) {
    GraphBuf g;
    auto options = torch::TensorOptions().dtype(DTYPE).device(torch::kCUDA);

    // Rollout tensors
    g.obs = torch::zeros({batch, input_size}, options);
    g.actions = torch::zeros(batch, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    g.value = torch::zeros(batch, options);
    g.logprobs = torch::zeros(batch, options);
    g.state = policy->initial_state(batch, torch::kCUDA);
    g.state_out = policy->initial_state(batch, torch::kCUDA);

    // Train tensors
    g.mb_obs = torch::zeros({minibatch_segments, horizon, input_size}, options);
    g.mb_state = torch::zeros({num_layers, minibatch_segments, 1, hidden_size * expansion_factor}, options);
    g.mb_newvalue = torch::zeros({minibatch_segments, horizon, 1}, options);
    g.mb_ratio = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_actions = torch::zeros({minibatch_segments, horizon}, options).to(torch::kInt64);
    g.mb_logprobs = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_advantages = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_prio = torch::zeros({minibatch_segments, 1}, options);
    g.mb_values = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_returns = torch::zeros({minibatch_segments, horizon}, options);
    return g;
}

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(32);
    vec_log(pufferl.vec, out);
    return out;
}

Tensor initial_state_impl(PuffeRL& pufferl, int64_t batch_size, torch::Device device) {
    return pufferl.policy->initial_state(batch_size, device);
}

void forward_call(GraphBuf& graph, PolicyMinGRU* policy, bool kernels,
        uint64_t rng_seed, Tensor& rng_offset) {
    torch::NoGradGuard no_grad;

    auto [logits, value, state_out] = policy->forward(graph.obs, graph.state);

    if (kernels) {
        sample_logits(logits, value, graph.actions, graph.logprobs,
            graph.value, rng_seed, rng_offset);
    } else {
        logits = torch::nan_to_num(logits, 1e-8, 1e-8, 1e-8);
        Tensor logprobs = torch::log_softmax(logits, 1);
        Tensor action = at::multinomial(logprobs.exp(), 1, true).squeeze(1);
        Tensor logprob = logprobs.gather(1, action.unsqueeze(1)).squeeze(1);
        graph.actions.copy_(action, false);
        graph.logprobs.copy_(logprob, false);
        graph.value.copy_(value.flatten(), false);
    }
    graph.state.copy_(state_out, false);
    graph.state_out.copy_(state_out, false);
}

void rollout_copy_call(RolloutBuf& rollouts, EnvBuf& env, GraphBuf& graph,
        HypersT& hypers, int h, int buf) {
    int block_size = hypers.num_envs / hypers.num_buffers;

    // Store with non-blocking copies
    // Layout is {horizon, segments, ...}, so select(0, h) gives contiguous {segments, ...}
    rollouts.observations.select(0, h).narrow(0, buf*block_size, block_size).copy_(graph.obs, true);
    rollouts.actions.select(0, h).narrow(0, buf*block_size, block_size).copy_(graph.actions, true);
    rollouts.logprobs.select(0, h).narrow(0, buf*block_size, block_size).copy_(graph.logprobs, true);
    rollouts.values.select(0, h).narrow(0, buf*block_size, block_size).copy_(graph.value, true);

    Tensor rewards_batch = env.rewards.narrow(0, buf*block_size, block_size);
    rollouts.rewards.select(0, h).narrow(0, buf*block_size, block_size).copy_(rewards_batch, true);

    Tensor terminals_batch = env.terminals.narrow(0, buf*block_size, block_size);
    rollouts.terminals.select(0, h).narrow(0, buf*block_size, block_size).copy_(terminals_batch, true);

    env.actions.narrow(0, buf*block_size, block_size).copy_(graph.actions, true);
}

void train_forward_call(GraphBuf& graph, PolicyMinGRU* policy,
        torch::optim::Muon* muon, HypersT& hypers, Tensor& adv_mean, Tensor& adv_std) {
    auto [logits, newvalue] = policy->forward_train(graph.mb_obs.to(DTYPE), graph.mb_state);

    Tensor loss;
    //if (kernels) {
    if (false) {
        loss = fused_ppo_loss(
            logits,
            newvalue,
            graph.mb_actions,
            graph.mb_logprobs.to(logits.dtype()),
            graph.mb_advantages.to(logits.dtype()),
            graph.mb_prio.to(logits.dtype()),
            graph.mb_values.to(logits.dtype()),
            graph.mb_returns.to(logits.dtype()),
            adv_mean,
            adv_std,
            hypers.clip_coef,
            hypers.vf_clip_coef,
            hypers.vf_coef,
            hypers.ent_coef
        )[0];
    } else {
        // Flatten for action lookup
        Tensor flat_logits = logits.reshape({-1, logits.size(-1)});
        Tensor flat_actions = graph.mb_actions.reshape({-1});
        Tensor logprobs_new = torch::log_softmax(flat_logits, 1);
        Tensor probs_new = logprobs_new.exp();

        // Gather logprobs for taken actions
        Tensor newlogprob_flat = logprobs_new.gather(1, flat_actions.unsqueeze(1)).squeeze(1);
        Tensor newlogprob = newlogprob_flat.reshape({hypers.minibatch_segments, hypers.horizon});
        Tensor entropy = - (probs_new * logprobs_new).sum(1).mean();

        // Compute ratio
        Tensor logratio = newlogprob - graph.mb_logprobs;
        Tensor ratio_new = logratio.exp();
        graph.mb_ratio.copy_(ratio_new, false);
        graph.mb_newvalue.copy_(newvalue, false);

        // Normalize advantages: (adv - mean) / std, then weight
        Tensor adv_normalized = graph.mb_advantages;
        adv_normalized = graph.mb_prio * (adv_normalized - adv_normalized.mean()) / (adv_normalized.std() + 1e-8);

        // Policy loss
        Tensor pg_loss1 = -adv_normalized * ratio_new;
        Tensor pg_loss2 = -adv_normalized * torch::clamp(ratio_new, 1.0 - hypers.clip_coef, 1.0 + hypers.clip_coef);
        Tensor pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        // Value loss
        newvalue = newvalue.view(graph.mb_returns.sizes());
        Tensor v_clipped = graph.mb_values + torch::clamp(newvalue - graph.mb_values,
            -hypers.vf_clip_coef, hypers.vf_clip_coef);
        Tensor v_loss_unclipped = (newvalue - graph.mb_returns).pow(2);
        Tensor v_loss_clipped = (v_clipped - graph.mb_returns).pow(2);
        Tensor v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();

        // Total loss
        loss = pg_loss + hypers.vf_coef*v_loss - hypers.ent_coef*entropy;
        /*
        {
            torch::NoGradGuard no_grad;

            // Accumulate stats
            pg_sum += pg_loss.detach();
            v_sum += v_loss.detach();
            ent_sum += entropy.detach();
            total_sum += loss.detach();

            // KL and clipping diagnostics (matches Python)
            auto old_kl = (-logratio).mean();
            auto kl = ((ratio_new - 1) - logratio).mean();
            auto cf = (ratio_new - 1.0).abs().gt(hypers.clip_coef).to(torch::kFloat32).mean();
            auto imp = ratio_new.mean();

            old_approx_kl_sum += old_kl.detach();
            approx_kl_sum += kl.detach();
            clipfrac_sum += cf.detach();
            importance_sum += imp.detach();
        }
        */
    }

    loss.backward();
    clip_grad_norm_(policy->parameters(), hypers.max_grad_norm);
    muon->step();
    muon->zero_grad();
}

// Capture
void capture_graph(at::cuda::CUDAGraph* graph, std::function<void()> func) {
    /* Checklist for avoiding diabolical capture bugs:
     * 1. Don't start separate streams before tracing (i.e. env gpu buffers)
     * 2. Make sure input/output buffer pointers don't change
     * 3. Make sure to restore the original stream after tracing
     * 4. All custom kernels need to use the default torch stream
     * 5. Make sure you are using the torch stream fns, not the c10 ones.
     * 6. Scalars get captured by value. They cannot change between calls.
     */
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    for (int i = 0; i < 10; ++i) {
        func();
    }
    warmup_stream.synchronize();

    auto cap_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(cap_stream);
    graph->capture_begin();
    func();
    graph->capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();

    at::cuda::setCurrentCUDAStream(current_stream);
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl_impl(HypersT& hypers, const std::string& env_name, Dict* env_kwargs) {
    auto pufferl = std::make_unique<pufferlib::PuffeRL>();
    pufferl->hypers = hypers;

    // Seeding
    torch::manual_seed(42);
    torch::cuda::manual_seed(42);
    pufferl->rng_seed = 42;
    pufferl->rng_offset = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));

    // Enable cuDNN benchmarking
    torch::globalContext().setBenchmarkCuDNN(true);
    torch::globalContext().setDeterministicCuDNN(false);
    torch::globalContext().setBenchmarkLimitCuDNN(32);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090)
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);

    // Enable faster FP16 reductions
    torch::globalContext().setAllowFP16ReductionCuBLAS(true);

    // BF16 reduction (if using bfloat16)
    torch::globalContext().setAllowBF16ReductionCuBLAS(true);

    // Load environment first to get input_size from obs tensor
    auto [vec, obs, actions, rewards, terminals] = create_environments(hypers.num_envs, env_name, env_kwargs);
    pufferl->vec = vec;
    pufferl->env.obs = obs;
    pufferl->env.actions = actions;
    pufferl->env.rewards = rewards;
    pufferl->env.terminals = terminals;

    int input_size = obs.size(1);
    int num_atns = hypers.num_atns;
    int hidden_size = hypers.hidden_size;
    int expansion_factor = hypers.expansion_factor;
    int num_layers = hypers.num_layers;
    bool kernels = hypers.kernels;
    PolicyMinGRU* policy = new PolicyMinGRU(input_size, num_atns, hidden_size, expansion_factor, num_layers, kernels);
    policy->to(torch::kCUDA);
    policy->to(DTYPE);
    pufferl->policy = policy;

    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    float eps = hypers.eps;
    pufferl->muon = new torch::optim::Muon(policy->parameters(),
        torch::optim::MuonOptions(lr).momentum(beta1).eps(eps));

    // Allocate buffers
    int segments = hypers.segments;
    int horizon = hypers.horizon;
    int batch = hypers.num_envs / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;
    int minibatch_segments = hypers.minibatch_segments;

    pufferl->rollouts = create_rollouts(horizon, segments, input_size);
    pufferl->graph = create_graph(batch, input_size, minibatch_segments, horizon,
        policy->num_layers, policy->hidden_size, policy->expansion_factor, policy);

    pufferl->adv_mean = torch::zeros({1}, torch::dtype(DTYPE).device(torch::kCUDA));
    pufferl->adv_std = torch::ones({1}, torch::dtype(DTYPE).device(torch::kCUDA));

    // Per-buffer states: each is {num_layers, block_size, hidden} for contiguous access
    pufferl->buffer_states.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_states[i] = policy->initial_state(batch, torch::kCUDA);
    }

    if (hypers.cudagraphs) {
        pufferl->rollout_graph = at::cuda::CUDAGraph();
        pufferl->train_forward_graph = at::cuda::CUDAGraph();

        auto* p = pufferl.get();
        capture_graph(&pufferl->rollout_graph, [p]() {
            forward_call(p->graph, p->policy, p->hypers.kernels, p->rng_seed, p->rng_offset);
        });
        capture_graph(&pufferl->train_forward_graph, [p]() {
            train_forward_call(p->graph, p->policy, p->muon,
                p->hypers, p->adv_mean, p->adv_std);
        });

        for (int i = 0; i < hypers.horizon; ++i) {
            for (int j = 0; j < hypers.num_buffers; ++j) {
                pufferl->rollout_copy_graphs[i][j] = at::cuda::CUDAGraph();
                capture_graph(&pufferl->rollout_copy_graphs[i][j], [p, i, j]() {
                    rollout_copy_call(p->rollouts, p->env, p->graph, p->hypers, i, j);
                });
            }
        }
    }

    // FAILS IF DONE AFTER CREATE_ENVIRONMENTS
    create_threads(vec, 8, 256);
    vec_reset(vec);

    return pufferl;
}

void python_vec_recv_impl(PuffeRL& pufferl, int buf) {
    vec_recv(pufferl.vec, buf);
}

void python_vec_send_impl(PuffeRL& pufferl, int buf) {
    vec_send(pufferl.vec, buf);
}

torch::autograd::tensor_list env_buffers_impl(PuffeRL& pufferl) {
    return {pufferl.env.obs, pufferl.env.actions, pufferl.env.rewards, pufferl.env.terminals};
}

// ============================================================================
// Rollout and train section functions
// ============================================================================

inline void profile_begin(const char* tag, bool enable) {
    if (enable) { cudaDeviceSynchronize(); nvtxRangePushA(tag); }
}

inline void profile_end(bool enable) {
    if (enable) { cudaDeviceSynchronize(); nvtxRangePop(); }
}

void env_recv(PuffeRL& pufferl, int buf) {
    vec_recv(pufferl.vec, buf);
}

void rollout_copy_inputs(PuffeRL& pufferl, int buf, int block_size) {
    auto& buf_state = pufferl.buffer_states[buf];
    pufferl.graph.obs.copy_(pufferl.env.obs.narrow(0, buf*block_size, block_size), true);
    pufferl.graph.state.copy_(buf_state, false);
}

void env_send(PuffeRL& pufferl, int buf) {
    vec_send(pufferl.vec, buf);
}

void compute_advantage(RolloutBuf& rollouts, Tensor& advantages, HypersT& hypers) {
    compute_puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
        rollouts.ratio, advantages, hypers.gamma, hypers.gae_lambda,
        hypers.vtrace_rho_clip, hypers.vtrace_c_clip);
}

std::tuple<Tensor, Tensor> compute_prio(Tensor& advantages,
        int minibatch_segments, int segments,
        float prio_alpha, float anneal_beta) {
    Tensor adv = advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, minibatch_segments, true);
    Tensor mb_prio = torch::pow(segments*prio_probs.index_select(0, idx).unsqueeze(1), -anneal_beta);
    return {idx, mb_prio};
}

void train_select_and_copy(GraphBuf& graph, RolloutBuf& rollouts,
        Tensor& advantages, Tensor& idx, Tensor& mb_state, Tensor& mb_prio) {
    Tensor mb_obs = rollouts.observations.index_select(0, idx);
    Tensor mb_actions = rollouts.actions.index_select(0, idx);
    Tensor mb_logprobs = rollouts.logprobs.index_select(0, idx);
    Tensor mb_values = rollouts.values.index_select(0, idx);
    Tensor mb_advantages = advantages.index_select(0, idx);
    Tensor mb_returns = mb_advantages + mb_values;

    mb_state.zero_();
    graph.mb_obs.copy_(mb_obs, false);
    graph.mb_state.copy_(mb_state, false);
    graph.mb_actions.copy_(mb_actions, false);
    graph.mb_logprobs.copy_(mb_logprobs, false);
    graph.mb_advantages.copy_(mb_advantages, false);
    graph.mb_prio.copy_(mb_prio, false);
    graph.mb_values.copy_(mb_values, false);
    graph.mb_returns.copy_(mb_returns, false);
}

void rollouts_impl(PuffeRL& pufferl) {
    torch::NoGradGuard no_grad;
    HypersT& hypers = pufferl.hypers;

    int horizon = hypers.horizon;
    int num_envs = hypers.num_envs;
    int num_buffers = hypers.num_buffers;
    int block_size = num_envs / num_buffers;
    // TODO: You removed state zeros and reward clamping

    for (int i = 0; i < num_buffers*horizon; ++i) {
        int buf = i % num_buffers;
        int h = i / num_buffers;

        profile_begin("env_recv", hypers.profile);
        env_recv(pufferl, buf);
        profile_end(hypers.profile);

        profile_begin("rollout_copy_inputs", hypers.profile);
        rollout_copy_inputs(pufferl, buf, block_size);
        profile_end(hypers.profile);

        profile_begin("rollout_graph", hypers.profile);
        if (hypers.cudagraphs) {
            pufferl.rollout_graph.replay();
        } else {
            forward_call(pufferl.graph, pufferl.policy, hypers.kernels,
                pufferl.rng_seed, pufferl.rng_offset);
        }
        profile_end(hypers.profile);

        profile_begin("rollout_copy_outputs", hypers.profile);
        auto& buf_state = pufferl.buffer_states[buf];
        buf_state.copy_(pufferl.graph.state_out, false);
        if (hypers.cudagraphs) {
            pufferl.rollout_copy_graphs[h][buf].replay();
        } else {
            rollout_copy_call(pufferl.rollouts, pufferl.env, pufferl.graph, hypers, h, buf);
        }
        profile_end(hypers.profile);

        // TODO: There should be a lighter way to sync. You need to make sure the torch data streams
        // are ready because puffer vec uses different streams. Setting to non-blocking is not enough.
        cudaDeviceSynchronize();

        profile_begin("env_send", hypers.profile);
        env_send(pufferl, buf);
        profile_end(hypers.profile);
    }
}

void train_impl(PuffeRL& pufferl) {
    HypersT& hypers = pufferl.hypers;

    // Buffers are stored as {horizon, segments, ...} for contiguous rollout writes
    // Transpose to {segments, horizon, ...} for train logic
    // Need .contiguous() because compute_puff_advantage_cuda uses raw data pointers
    RolloutBuf rollouts;
    rollouts.observations = pufferl.rollouts.observations.permute({1, 0, 2}).contiguous();
    rollouts.actions = pufferl.rollouts.actions.transpose(0, 1).contiguous();
    rollouts.logprobs = pufferl.rollouts.logprobs.transpose(0, 1).contiguous();
    rollouts.rewards = pufferl.rollouts.rewards.transpose(0, 1).contiguous();
    rollouts.rewards.clamp_(-1.0, 1.0);  // Clamp rewards here instead of in eval to save a kernel call per step
    rollouts.terminals = pufferl.rollouts.terminals.transpose(0, 1).contiguous();
    rollouts.ratio = pufferl.rollouts.ratio.transpose(0, 1).contiguous();
    rollouts.values = pufferl.rollouts.values.transpose(0, 1).contiguous();

    int total_minibatches = hypers.total_minibatches;
    int minibatch_segments = hypers.minibatch_segments;
    int segments = hypers.segments;
    int accumulate_minibatches = hypers.accumulate_minibatches;
    int horizon = hypers.horizon;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    float clip_coef = hypers.clip_coef;
    float vf_clip_coef = hypers.vf_clip_coef;
    float gamma = hypers.gamma;
    float gae_lambda = hypers.gae_lambda;
    float vtrace_rho_clip = hypers.vtrace_rho_clip;
    float vtrace_c_clip = hypers.vtrace_c_clip;
    float vf_coef = hypers.vf_coef;
    float ent_coef = hypers.ent_coef;
    float max_grad_norm = hypers.max_grad_norm;
    bool use_rnn = hypers.use_rnn;
    bool anneal_lr = hypers.anneal_lr;
    int total_epochs = hypers.max_epochs;
    int current_epoch = pufferl.epoch;

    // Accumulators
    torch::Device device = rollouts.values.device();
    torch::TensorOptions scalar_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    Tensor pg_sum = torch::zeros({}, scalar_opts);
    Tensor v_sum = torch::zeros({}, scalar_opts);
    Tensor ent_sum = torch::zeros({}, scalar_opts);
    Tensor total_sum = torch::zeros({}, scalar_opts);
    Tensor old_approx_kl_sum = torch::zeros({}, scalar_opts);
    Tensor approx_kl_sum = torch::zeros({}, scalar_opts);
    Tensor clipfrac_sum = torch::zeros({}, scalar_opts);
    Tensor importance_sum = torch::zeros({}, scalar_opts);

    PolicyMinGRU* policy = pufferl.policy;
    torch::optim::Muon* muon = pufferl.muon;

    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, hypers.max_epochs);
        muon->lr.fill_(lr);
    }

    // Annealed priority exponent - TODO: graphed?
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;

    // Zero out ratio at start of epoch (matches Python: self.ratio[:] = 1)
    rollouts.ratio.fill_(1.0);

    Tensor advantages = torch::zeros_like(rollouts.values);
    compute_advantage(rollouts, advantages, hypers);

    pufferl.adv_mean.copy_(advantages.mean().detach());
    pufferl.adv_std.copy_(advantages.std().detach());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Tensor mb_state = torch::zeros(
        {policy->num_layers, minibatch_segments, 1, (int64_t)(policy->hidden_size*policy->expansion_factor)},
        torch::dtype(DTYPE).device(rollouts.values.device())
    );

    // Temporary: random indices and uniform weights
    /*
    auto idx = torch::randint(0, segments, {minibatch_segments}, torch::dtype(torch::kInt64).device(device));
    auto mb_prio = torch::ones({minibatch_segments, 1}, torch::dtype(torch::kFloat32).device(device));
    */

    for (int mb = 0; mb < total_minibatches; ++mb) {
        advantages.fill_(0.0);

        profile_begin("compute_advantage", hypers.profile);
        compute_advantage(rollouts, advantages, hypers);
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        auto [idx, mb_prio] = compute_prio(advantages, minibatch_segments, segments,
            prio_alpha, anneal_beta);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        train_select_and_copy(pufferl.graph, rollouts, advantages, idx, mb_state, mb_prio);
        profile_end(hypers.profile);

        profile_begin("train_forward_graph", hypers.profile);
        if (hypers.cudagraphs) {
            pufferl.train_forward_graph.replay();
        } else {
            train_forward_call(pufferl.graph, pufferl.policy, pufferl.muon,
                hypers, pufferl.adv_mean, pufferl.adv_std);
        }
        profile_end(hypers.profile);

        // Update global ratio and values in-place (matches Python)
        // Buffers are {horizon, segments}, so index_copy_ along dim 1 (segments)
        // Source is {minibatch_segments, horizon}, need to transpose to {horizon, minibatch_segments}
        // Temporary: use slice instead of index_copy_ for contiguous test
        /*
        pufferl.rollouts.ratio.slice(1, 0, minibatch_segments).copy_(pufferl.graph.ratio.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));
        pufferl.rollouts.values.slice(1, 0, minibatch_segments).copy_(pufferl.graph.newvalue.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));
        */
        // Original index_copy_ version:
        pufferl.rollouts.ratio.index_copy_(1, idx, pufferl.graph.mb_ratio.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));
        pufferl.rollouts.values.index_copy_(1, idx, pufferl.graph.mb_newvalue.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));

    }
    pufferl.epoch += 1;

    // Compute explained variance at end of epoch
    /*
    auto y_true = advantages.flatten() + values.flatten();
    auto y_pred = values.flatten();
    auto var_y = y_true.var();
    */
    //double explained_var = (var_y.abs() < 1e-8) ? NAN : (1 - (y_true - y_pred).var() / var_y).item<double>();
}

// Profiler control for nsys --capture-range=cudaProfilerApi
void profiler_start() {
    cudaDeviceSynchronize();
    printf("cudaProfilerStart()\n");
    cudaProfilerStart();
}

void profiler_stop() {
    cudaDeviceSynchronize();
    cudaProfilerStop();
    printf("cudaProfilerStop()\n");
}

} // namespace pufferlib
