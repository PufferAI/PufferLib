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
#include <dlfcn.h>
#include "muon.h"

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>

#include <nvtx3/nvToolsExt.h>

#include <functional>
#include <iostream>
#include <vector>

#include "env_binding.h"

typedef torch::Tensor Tensor;

#if defined(PUFFER_DEBUG)

inline static void PUFFER_ASSERT_BREAK()
{
#if defined(_MSC_VER)
  // assert (abort) does not break into the debugger in VS 2022 ! It's insane, so we have to use this weird contraption that's cross platform.
  __debugbreak();
#elif defined(__clang__) || defined(__GNUC__)
  __builtin_trap();
#else
  /* Fallback method */
  *((volatile int*)0) = 0; /* This will cause a segmentation fault */
#endif
}
#endif

// LibTorch throws exceptions on errors, log them correctly in debug mode only.
#if PUFFER_DEBUG
#define BEGIN_LIBTORCH_CATCH try {
#else
#define BEGIN_LIBTORCH_CATCH
#endif

#if PUFFER_DEBUG
#define END_LIBTORCH_CATCH                                                                                             \
  }                                                                                                                    \
  catch (const c10::Error& e)                                                                                          \
  {                                                                                                                    \
    std::cerr << "Error from libtorch: " << e.what() << std::endl;                                                     \
    PUFFER_ASSERT_BREAK();                                                                                             \
    throw;                                                                                                             \
  }

#else
#define END_LIBTORCH_CATCH
#endif


// CUDA kernel wrappers
#include "modules.cpp"

// get dtype based on bf16 flag
inline torch::ScalarType get_dtype(bool bf16) {
    return bf16 ? torch::kBFloat16 : torch::kFloat32;
}

namespace pufferlib {

// Advantage computation is in advantage.cpp
#include "advantage.cpp"

// Model classes are in models.cpp
#include "models.cpp"

torch::Dtype to_torch_dtype(int dtype) {
    if (dtype == FLOAT) {
        return torch::kFloat32;
    } else if (dtype == INT) {
        return torch::kInt32;
    } else if (dtype == UNSIGNED_CHAR) {
        return torch::kUInt8;
    } else if (dtype == DOUBLE) {
        return torch::kFloat64;
    } else if (dtype == CHAR) {
        return torch::kInt8;
    } else {
        assert(false && "to_torch_dtype failed to convert dtype");
    }
    return torch::kFloat32;
}

// Fast clip_grad_norm_ for contiguous weights
// Cats all grads for one-shot norm computation, then scales each grad
void clip_grad_norm_(
    const std::vector<Tensor>& parameters,
    double max_norm
    ) {
  // Collect flattened grads
  std::vector<Tensor> flat_grads;
  flat_grads.reserve(parameters.size());

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      flat_grads.push_back(grad.flatten());
    }
  }

  if (flat_grads.empty()) {
    return;
  }

  // Single cat + norm (avoids per-param norm calls)
  Tensor all_grads = torch::cat(flat_grads);
  // Getting errors here? See if your net is definint a layeyr and not using it.
  // TODO: That shouldn't error
  Tensor total_norm = all_grads.to(torch::kFloat32).norm(2);

  // Compute clip coefficient
  Tensor clip_coef = torch::clamp_max(max_norm / (total_norm + 1e-6), 1.0);

  // Scale each grad in-place
  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      grad.mul_(clip_coef);
    }
  }
}

float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;  // avoid division by zero
    float ratio = static_cast<float>(t) / static_cast<float>(T);
    ratio = std::max(0.0f, std::min(1.0f, ratio));  // clamp to [0, 1]
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}


typedef struct {
    Tensor obs;
    Tensor actions;
    Tensor rewards;
    Tensor terminals;
} EnvBuf;

std::tuple<StaticVec*, Tensor>
create_environments(int num_buffers, int total_agents, const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    StaticVec* vec = create_static_vec(total_agents, num_buffers, vec_kwargs, env_kwargs);
    printf("DEBUG create_environments: vec->size=%d, vec->total_agents=%d\n",
        vec->size, vec->total_agents);

    int obs_size = get_obs_size();
    int num_atns = get_num_atns();

    env.obs = torch::from_blob(vec->gpu_observations, {total_agents, obs_size}, torch::dtype(to_torch_dtype(get_obs_type())).device(torch::kCUDA));
    env.actions = torch::from_blob(vec->gpu_actions, {total_agents, num_atns}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    env.rewards = torch::from_blob(vec->gpu_rewards, {total_agents}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    env.terminals = torch::from_blob(vec->gpu_terminals, {total_agents}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Create act_sizes tensor on CUDA (needed for sample_logits kernel)
    Tensor act_sizes = torch::from_blob(get_act_sizes(), {num_atns}, torch::dtype(torch::kInt32)).to(torch::kCUDA);

    return std::make_tuple(vec, act_sizes);
}

typedef struct {
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
} TrainGraph;

TrainGraph create_train_graph(int minibatch_segments, int horizon, int input_size,
        int num_layers, int hidden_size, int expansion_factor, int num_atns, bool bf16) {
    TrainGraph g;
    auto dtype = get_dtype(bf16);
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    g.mb_obs = torch::zeros({minibatch_segments, horizon, input_size}, options);
    g.mb_state = torch::zeros({num_layers, minibatch_segments, 1, hidden_size * expansion_factor}, options);
    g.mb_newvalue = torch::zeros({minibatch_segments, horizon, 1}, options);
    g.mb_ratio = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_actions = torch::zeros({minibatch_segments, horizon, num_atns}, options).to(torch::kInt64);
    g.mb_logprobs = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_advantages = torch::zeros({minibatch_segments, horizon}, options.dtype(torch::kFloat32));  // always fp32 for precision
    g.mb_prio = torch::zeros({minibatch_segments, 1}, options);
    g.mb_values = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_returns = torch::zeros({minibatch_segments, horizon}, options);
    return g;
}

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

RolloutBuf create_rollouts(int horizon, int segments, int input_size, int num_atns, bool bf16) {
    RolloutBuf r;
    auto dtype = get_dtype(bf16);
    r.observations = torch::zeros({horizon, segments, input_size}, torch::dtype(dtype).device(torch::kCUDA));
    r.actions = torch::zeros({horizon, segments, num_atns}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    r.values = torch::zeros({horizon, segments}, torch::dtype(dtype).device(torch::kCUDA));
    r.logprobs = torch::zeros({horizon, segments}, torch::dtype(dtype).device(torch::kCUDA));
    r.rewards = torch::zeros({horizon, segments}, torch::dtype(dtype).device(torch::kCUDA));
    r.terminals = torch::zeros({horizon, segments}, torch::dtype(dtype).device(torch::kCUDA));
    r.ratio = torch::zeros({horizon, segments}, torch::dtype(dtype).device(torch::kCUDA));
    r.importance = torch::zeros({horizon, segments}, torch::dtype(dtype).device(torch::kCUDA));
    return r;
}

typedef struct {
    // Layout
    int horizon;
    int total_agents;
    int num_buffers;
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
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
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
    bool use_omp;
    bool bf16;  // bfloat16 mixed precision training
} HypersT;

typedef struct {
    PolicyMinGRU* policy_bf16;  // Working weights (bf16) - used for forward/backward
    PolicyMinGRU* policy_fp32;  // Master weights (fp32) - used for optimizer
    StaticVec* vec;
    torch::optim::Muon* muon;
    HypersT hypers;
    std::vector<Tensor> buffer_states;  // Per-buffer states for contiguous access
    RolloutBuf rollouts;
    EnvBuf env;
    TrainGraph train_buf;
    std::vector<std::vector<at::cuda::CUDAGraph>> fused_rollout_cudagraphs;  // [horizon][num_buffers]
    at::cuda::CUDAGraph train_cudagraph;
    std::vector<at::cuda::CUDAStream> torch_streams;  // PyTorch-managed streams for OMP
    Tensor adv_mean;
    Tensor adv_std;
    Tensor act_sizes;      // CUDA int32 tensor of action head sizes for MultiDiscrete
    Tensor act_sizes_cpu;  // CPU int64 tensor (pre-computed to avoid alloc during graph replay)
    int epoch;
    uint64_t rng_seed;
    Tensor rng_offset;  // CUDA tensor so increment is graphable
} PuffeRL;

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(32);
    static_vec_log(pufferl.vec, out);
    return out;
}

// Fused rollout step: reads from env, runs forward, writes directly to rollouts storage
// Eliminates intermediate RolloutGraph buffers when cudagraphed
void fused_rollout_step(PuffeRL& pufferl, int h, int buf) {
    torch::NoGradGuard no_grad;
    HypersT& hypers = pufferl.hypers;
    int total_agents = pufferl.vec->total_agents;
    int num_buffers = hypers.num_buffers;
    int block_size = total_agents / num_buffers;

    // Get slices for this buffer
    Tensor obs_slice = pufferl.env.obs.narrow(0, buf*block_size, block_size);
    Tensor& state = pufferl.buffer_states[buf];

    // Run policy forward using bf16 working weights
    auto [logits, value, state_out] = pufferl.policy_bf16->forward(obs_slice, state);

    // Get output slices in rollouts storage
    Tensor actions_out = pufferl.rollouts.actions.select(0, h).narrow(0, buf*block_size, block_size);
    Tensor logprobs_out = pufferl.rollouts.logprobs.select(0, h).narrow(0, buf*block_size, block_size);
    Tensor values_out = pufferl.rollouts.values.select(0, h).narrow(0, buf*block_size, block_size);

    // Sample actions and write directly to rollouts
    if (hypers.kernels) {
        sample_logits(logits, value, actions_out, logprobs_out,
            values_out, pufferl.act_sizes, pufferl.rng_seed, pufferl.rng_offset);
    } else {
        int num_action_heads = actions_out.size(1);
        logits = torch::nan_to_num(logits, 1e-8, 1e-8, 1e-8);

        auto split_logits = torch::split(logits, c10::IntArrayRef(pufferl.act_sizes_cpu.data_ptr<int64_t>(), num_action_heads), 1);
        std::vector<Tensor> actions_vec;
        std::vector<Tensor> logprobs_vec;

        for (int i = 0; i < num_action_heads; i++) {
            Tensor head_logits = split_logits[i];
            Tensor log_probs = torch::log_softmax(head_logits, 1);
            Tensor action = at::multinomial(log_probs.exp(), 1, true);
            Tensor logprob = log_probs.gather(1, action);
            actions_vec.push_back(action);
            logprobs_vec.push_back(logprob);
        }
        actions_out.copy_(torch::cat(actions_vec, 1).to(torch::kFloat64), false);
        logprobs_out.copy_(torch::cat(logprobs_vec, 1).sum(1), false);
        values_out.copy_(value.flatten(), false);
    }

    // Update state
    state.copy_(state_out, false);

    // Copy obs to rollouts
    pufferl.rollouts.observations.select(0, h).narrow(0, buf*block_size, block_size).copy_(obs_slice, true);

    // Copy rewards and terminals from env to rollouts
    pufferl.rollouts.rewards.select(0, h).narrow(0, buf*block_size, block_size).copy_(
        pufferl.env.rewards.narrow(0, buf*block_size, block_size), true);
    pufferl.rollouts.terminals.select(0, h).narrow(0, buf*block_size, block_size).copy_(
        pufferl.env.terminals.narrow(0, buf*block_size, block_size), true);

    // Copy actions to env for next step
    pufferl.env.actions.narrow(0, buf*block_size, block_size).copy_(actions_out, true);
}

void train_forward_call(TrainGraph& graph, PolicyMinGRU* policy_bf16, PolicyMinGRU* policy_fp32,
        torch::optim::Muon* muon, HypersT& hypers, Tensor& adv_mean, Tensor& adv_std, Tensor& act_sizes_cpu, bool kernels) {
    auto [logits, newvalue] = policy_bf16->forward_train(graph.mb_obs, graph.mb_state);

    Tensor loss;
    if (kernels) {
        auto [mb_adv_var, mb_adv_mean] = torch::var_mean(graph.mb_advantages);  // single kernel launch
        loss = fused_ppo_loss_optimized(
            logits,
            newvalue,
            graph.mb_actions,
            graph.mb_logprobs,
            graph.mb_advantages,
            graph.mb_prio,
            graph.mb_values,
            graph.mb_returns,
            mb_adv_mean,
            mb_adv_var,  // variance, not std - kernel does sqrtf to avoid second kernel launch here
            graph.mb_ratio,
            graph.mb_newvalue.view({graph.mb_ratio.size(0), graph.mb_ratio.size(1)}),
            hypers.clip_coef,
            hypers.vf_clip_coef,
            hypers.vf_coef,
            hypers.ent_coef
        )[0];
    } else {
        int num_action_heads = graph.mb_actions.size(-1);
        int batch = hypers.minibatch_size;
        int minibatch_segments = batch / hypers.horizon;

        // Split logits by action head sizes and compute log probs for each head
        Tensor flat_logits = logits.reshape({batch, -1});
        flat_logits = torch::nan_to_num(flat_logits, 1e-8, 1e-8, 1e-8);
        auto split_logits = torch::split(flat_logits, c10::IntArrayRef(act_sizes_cpu.data_ptr<int64_t>(), num_action_heads), 1);

        std::vector<Tensor> logprobs_vec;
        std::vector<Tensor> entropies_vec;

        for (int h = 0; h < num_action_heads; h++) {
            Tensor head_logits = split_logits[h];
            Tensor log_probs = torch::log_softmax(head_logits, 1);
            Tensor probs = log_probs.exp();
            Tensor head_actions = graph.mb_actions.select(-1, h).reshape({batch}).to(torch::kInt64);
            Tensor logprob = log_probs.gather(1, head_actions.unsqueeze(1));
            logprobs_vec.push_back(logprob);
            entropies_vec.push_back(-(probs * log_probs).sum(1, true));
        }

        // Stack and reduce - no per-iteration allocations
        Tensor newlogprob = torch::cat(logprobs_vec, 1).sum(1).reshape({minibatch_segments, hypers.horizon});
        Tensor entropy = torch::cat(entropies_vec, 1).sum(1).mean();

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
    }

    // computes gradients on bf16 weights (or fp32 if not using bf16)
    loss.backward();
    
    // copy gradients from bf16 to fp32, then optimizer step on fp32 master weights
    if (hypers.bf16) {
        copy_gradients_to_fp32(policy_bf16, policy_fp32);
    }
    clip_grad_norm_(policy_fp32->parameters(), hypers.max_grad_norm);
    muon->step();
    muon->zero_grad();
    if (hypers.bf16) {
        policy_bf16->zero_grad();  // also need to clear bf16 gradients
        // sync updated fp32 weights back to bf16 for next forward pass
        sync_policy_weights(policy_bf16, policy_fp32);
    }
}

// Capture with shared memory pool
void capture_graph(at::cuda::CUDAGraph* graph, std::function<void()> func,
                   at::cuda::MempoolId_t pool) {
  BEGIN_LIBTORCH_CATCH
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
    graph->capture_begin(pool);
    func();
    graph->capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();

    at::cuda::setCurrentCUDAStream(current_stream);
  END_LIBTORCH_CATCH
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
    // Not used in static/OMP path
}

void env_send(PuffeRL& pufferl, int buf) {
    // Not used in static/OMP path
}

void compute_advantage(RolloutBuf& rollouts, Tensor& advantages, HypersT& hypers) {
    compute_puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
        rollouts.ratio, advantages, hypers.gamma, hypers.gae_lambda,
        hypers.vtrace_rho_clip, hypers.vtrace_c_clip);
}

// Thread initialization callback - sets CUDA stream once per thread
extern "C" void thread_init_wrapper(void* ctx, int buf) {
  BEGIN_LIBTORCH_CATCH
    PuffeRL* pufferl = (PuffeRL*)ctx;
    at::cuda::setCurrentCUDAStream(pufferl->torch_streams[buf]);
  END_LIBTORCH_CATCH
}

// Callback for OMP threadmanager - runs policy forward for one (buf, t) step
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
  BEGIN_LIBTORCH_CATCH
    torch::NoGradGuard no_grad;
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;

    profile_begin("fused_rollout", hypers.profile);
    if (hypers.cudagraphs) {
        // Fused cudagraph: input copy + forward + output copy in one shot
        pufferl->fused_rollout_cudagraphs[t][buf].replay();
    } else {
        fused_rollout_step(*pufferl, t, buf);
    }
    profile_end(hypers.profile);
  END_LIBTORCH_CATCH
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl_impl(HypersT& hypers, const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
  BEGIN_LIBTORCH_CATCH
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

    // Load environment first to get input_size and action info from env
    // act_sizes: 1D tensor of action space sizes per head
    // num_action_heads: number of action heads (for MultiDiscrete)
    // act_n: sum of action space sizes (decoder output dim)
    auto [vec, act_sizes] = create_environments(hypers.num_buffers, hypers.total_agents, env_name, vec_kwargs, env_kwargs, pufferl->env);
    int num_action_heads = pufferl->env.actions.size(1);
    int act_n = act_sizes.sum().item<int>();

    pufferl->vec = vec;
    pufferl->act_sizes = act_sizes;
    pufferl->act_sizes_cpu = act_sizes.cpu().to(torch::kInt64).contiguous();

    int input_size = pufferl->env.obs.size(1);
    int hidden_size = hypers.hidden_size;
    int expansion_factor = hypers.expansion_factor;
    int num_layers = hypers.num_layers;
    bool kernels = hypers.kernels;

    // Create encoder/decoder based on env_name
    // Decoder output size is act_n (sum of all action space sizes)
    // We need two sets for mixed-precision: fp32 (master) and bf16 (working)
    auto create_encoder_decoder = [&]() -> std::pair<std::shared_ptr<Encoder>, std::shared_ptr<Decoder>> {
        std::shared_ptr<Encoder> enc;
        std::shared_ptr<Decoder> dec;
        if (env_name == "puffer_snake") {
            enc = std::make_shared<SnakeEncoder>(input_size, hidden_size, 8);
            dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
        } else if (env_name == "puffer_g2048") {
            enc = std::make_shared<G2048Encoder>(input_size, hidden_size);
            dec = std::make_shared<G2048Decoder>(hidden_size, act_n);
        } else if (env_name == "puffer_nmmo3") {
            enc = std::make_shared<NMMO3Encoder>(input_size, hidden_size);
            dec = std::make_shared<NMMO3Decoder>(hidden_size, act_n);
        } else if (env_name == "puffer_drive") {
            enc = std::make_shared<DriveEncoder>(input_size, hidden_size);
            dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
        } else {
            enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
            dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
        }
        return {enc, dec};
    };

    // Create fp32 master policy (for optimizer - precise gradient accumulation)
    auto [enc_fp32, dec_fp32] = create_encoder_decoder();
    PolicyMinGRU* policy_fp32 = new PolicyMinGRU(enc_fp32, dec_fp32, input_size, act_n, hidden_size, expansion_factor, num_layers, kernels);
    policy_fp32->to(torch::kCUDA);
    policy_fp32->to(torch::kFloat32);
    pufferl->policy_fp32 = policy_fp32;

    if (hypers.bf16) {
        // create bf16 working policy (for fwd/bwd)
        auto [enc_bf16, dec_bf16] = create_encoder_decoder();
        PolicyMinGRU* policy_bf16 = new PolicyMinGRU(enc_bf16, dec_bf16, input_size, act_n, hidden_size, expansion_factor, num_layers, kernels);
        policy_bf16->to(torch::kCUDA);
        policy_bf16->to(torch::kBFloat16);
        pufferl->policy_bf16 = policy_bf16;
        sync_policy_weights(policy_bf16, policy_fp32); // initial sync
    } else {
        // just use same policy for both
        pufferl->policy_bf16 = policy_fp32;
    }

    // Optimizer uses fp32 master weights for precise gradient accumulation
    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    float eps = hypers.eps;
    pufferl->muon = new torch::optim::Muon(policy_fp32->parameters(),
        torch::optim::MuonOptions(lr).momentum(beta1).eps(eps));
    pufferl->muon->init_contiguous_weights();
    printf("DEBUG: Contiguous weight buffer: %ld elements\n", pufferl->muon->weight_buffer.numel());


    // Allocate buffers
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    printf("DEBUG: num_envs=%d, total_agents=%d, batch=%d, num_buffers=%d\n",
        vec->size, total_agents, batch, num_buffers);

    int minibatch_segments = hypers.minibatch_size / horizon;

    pufferl->rollouts = create_rollouts(horizon, total_agents, input_size, num_action_heads, hypers.bf16);
    pufferl->train_buf = create_train_graph(minibatch_segments, horizon, input_size,
        policy_fp32->num_layers, policy_fp32->hidden_size, policy_fp32->expansion_factor, num_action_heads, hypers.bf16);

    // always fp32 since advantages are computed in fp32
    pufferl->adv_mean = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    pufferl->adv_std = torch::ones({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Per-buffer states: each is {num_layers, block_size, hidden} for contiguous access
    pufferl->buffer_states.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_states[i] = pufferl->policy_bf16->initial_state(batch, torch::kCUDA);
    }

    if (hypers.cudagraphs) {
        pufferl->train_cudagraph = at::cuda::CUDAGraph();

        auto* p = pufferl.get();
        auto train_pool = at::cuda::graph_pool_handle();
        capture_graph(&pufferl->train_cudagraph, [p]() {
            train_forward_call(p->train_buf, p->policy_bf16, p->policy_fp32, p->muon,
                p->hypers, p->adv_mean, p->adv_std, p->act_sizes_cpu, p->hypers.kernels);
        }, train_pool);

        // Fused rollout cudagraphs: [horizon][num_buffers]
        // Each graph does input copy + forward + output copy in one shot
        // Use shared memory pool to reduce memory usage across graphs
        auto rollout_pool = at::cuda::graph_pool_handle();
        pufferl->fused_rollout_cudagraphs.resize(horizon);
        for (int h = 0; h < horizon; ++h) {
            pufferl->fused_rollout_cudagraphs[h].resize(num_buffers);
            for (int b = 0; b < num_buffers; ++b) {
                pufferl->fused_rollout_cudagraphs[h][b] = at::cuda::CUDAGraph();
                capture_graph(&pufferl->fused_rollout_cudagraphs[h][b], [p, h, b]() {
                    fused_rollout_step(*p, h, b);
                }, rollout_pool);
            }
        }

    }

    // Create PyTorch-managed streams and replace vec->streams with their raw cudaStream_t
    // This ensures PyTorch properly recognizes the streams for all operations
    for (int i = 0; i < num_buffers; i++) {
        pufferl->torch_streams.push_back(at::cuda::getStreamFromPool(false));
        vec->streams[i] = pufferl->torch_streams[i].stream();
    }


    // Static breakout - OMP only
    int num_threads = 16;
    if (hypers.use_omp) {
        create_static_threads(vec, num_threads, horizon, pufferl.get(), net_callback_wrapper, thread_init_wrapper);

    }
    static_vec_reset(vec);
    
    return pufferl;
  END_LIBTORCH_CATCH
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

void train_select_and_copy(TrainGraph& graph, RolloutBuf& rollouts,
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
  BEGIN_LIBTORCH_CATCH
    torch::NoGradGuard no_grad;
    HypersT& hypers = pufferl.hypers;

    int horizon = hypers.horizon;
    int num_buffers = hypers.num_buffers;
    // TODO: You removed state zeros and reward clamping

    for (int i = 0; i < num_buffers*horizon; ++i) {
        int buf = i % num_buffers;
        int h = i / num_buffers;

        profile_begin("env_recv", hypers.profile);
        env_recv(pufferl, buf);
        profile_end(hypers.profile);

        net_callback_wrapper(&pufferl, buf, h);

        // TODO: There should be a lighter way to sync. You need to make sure the torch data streams
        // are ready because puffer vec uses different streams. Setting to non-blocking is not enough.
        cudaDeviceSynchronize();

        profile_begin("env_send", hypers.profile);
        env_send(pufferl, buf);
        profile_end(hypers.profile);
    }
  END_LIBTORCH_CATCH
}


void train_impl(PuffeRL& pufferl) {
  BEGIN_LIBTORCH_CATCH
    // Update to HypersT& p
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

    // Inline any of these only used once
    int minibatch_size = hypers.minibatch_size;
    int batch_size = hypers.total_agents * hypers.horizon;
    int minibatch_segments = minibatch_size / hypers.horizon;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    bool anneal_lr = hypers.anneal_lr;
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

    PolicyMinGRU* policy_bf16 = pufferl.policy_bf16;
    // PolicyMinGRU* policy_fp32 = pufferl.policy_fp32;
    torch::optim::Muon* muon = pufferl.muon;

    int total_epochs = hypers.total_timesteps / batch_size;

    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, total_epochs);
        muon->lr.fill_(lr);
    }

    // Annealed priority exponent - TODO: graphed?
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;

    // Zero out ratio at start of epoch (matches Python: self.ratio[:] = 1)
    rollouts.ratio.fill_(1.0);

    Tensor advantages = torch::zeros_like(rollouts.values, torch::kFloat32);  // fp32 precision

    compute_advantage(rollouts, advantages, hypers);
    pufferl.adv_mean.copy_(advantages.mean().detach());
    pufferl.adv_std.copy_(advantages.std().detach());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto dtype = get_dtype(hypers.bf16);
    Tensor mb_state = torch::zeros(
        {policy_bf16->num_layers, minibatch_segments, 1, (int64_t)(policy_bf16->hidden_size*policy_bf16->expansion_factor)},
        torch::dtype(dtype).device(rollouts.values.device())
    );

    // Temporary: random indices and uniform weights
    /*
    auto idx = torch::randint(0, segments, {minibatch_segments}, torch::dtype(torch::kInt64).device(device));
    auto mb_prio = torch::ones({minibatch_segments, 1}, torch::dtype(torch::kFloat32).device(device));
    */

    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;

    for (int mb = 0; mb < total_minibatches; ++mb) {
        advantages.fill_(0.0);

        profile_begin("compute_advantage", hypers.profile);
        compute_advantage(rollouts, advantages, hypers);
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        auto [idx, mb_prio] = compute_prio(advantages, minibatch_segments, hypers.total_agents,
            prio_alpha, anneal_beta);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        TrainGraph& graph = pufferl.train_buf;
        train_select_and_copy(graph, rollouts, advantages, idx, mb_state, mb_prio);
        profile_end(hypers.profile);

        profile_begin("train_forward_graph", hypers.profile);
        if (hypers.cudagraphs) {
            pufferl.train_cudagraph.replay();
        } else {
            train_forward_call(graph, pufferl.policy_bf16, pufferl.policy_fp32, pufferl.muon,
                hypers, pufferl.adv_mean, pufferl.adv_std, pufferl.act_sizes_cpu, hypers.kernels);
        }
        profile_end(hypers.profile);

        // Update global ratio and values in-place (matches Python)
        // Buffers are {horizon, segments}, so index_copy_ along dim 1 (segments)
        // Source is {minibatch_segments, horizon}, need to transpose to {horizon, minibatch_segments}
        pufferl.rollouts.ratio.index_copy_(1, idx, graph.mb_ratio.detach().squeeze(-1).to(dtype).transpose(0, 1));
        pufferl.rollouts.values.index_copy_(1, idx, graph.mb_newvalue.detach().squeeze(-1).to(dtype).transpose(0, 1));

    }
    pufferl.epoch += 1;

    // Compute explained variance at end of epoch
    /*
    auto y_true = advantages.flatten() + values.flatten();
    auto y_pred = values.flatten();
    auto var_y = y_true.var();
    */
    //double explained_var = (var_y.abs() < 1e-8) ? NAN : (1 - (y_true - y_pred).var() / var_y).item<double>();
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  END_LIBTORCH_CATCH  
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
