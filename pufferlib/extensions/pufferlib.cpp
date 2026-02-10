/* Checklist for avoiding diabolical capture bugs:
 * 1. Don't start separate streams before tracing (i.e. env gpu buffers)
 * 2. Make sure input/output buffer pointers don't change
 * 3. Make sure to restore the original stream after tracing
 * 4. All custom kernels need to use the default torch stream
 * 5. Make sure you are using the torch stream fns, not the c10 ones.
 * 6. Scalars get captured by value. They cannot change between calls.
 */

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nccl.h>
#include <unistd.h>
#include <vector>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <nvtx3/nvToolsExt.h>

#include "muon.h"
#include "env_binding.h"

typedef torch::Tensor Tensor;

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

TrainGraph create_train_graph(int mb_segments, int horizon, int input_size,
        int num_layers, int hidden_size, int num_atns) {
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    return {
        .mb_obs = torch::zeros({mb_segments, horizon, input_size}, opts),
        .mb_state = torch::zeros({num_layers, mb_segments, 1, hidden_size}, opts),
        .mb_actions = torch::zeros({mb_segments, horizon, num_atns}, cuda_f64),
        .mb_logprobs = torch::zeros({mb_segments, horizon}, opts),
        .mb_advantages = torch::zeros({mb_segments, horizon}, cuda_f32),  // always fp32 for precision
        .mb_prio = torch::zeros({mb_segments, 1}, opts),
        .mb_values = torch::zeros({mb_segments, horizon}, opts),
        .mb_returns = torch::zeros({mb_segments, horizon}, opts),
        .mb_ratio = torch::zeros({mb_segments, horizon}, opts),
        .mb_newvalue = torch::zeros({mb_segments, horizon, 1}, opts),
    };
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

RolloutBuf create_rollouts(int horizon, int segments, int input_size, int num_atns) {
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    return {
        .observations = torch::zeros({horizon, segments, input_size}, opts),
        .actions = torch::zeros({horizon, segments, num_atns}, cuda_f64),
        .values = torch::zeros({horizon, segments}, opts),
        .logprobs = torch::zeros({horizon, segments}, opts),
        .rewards = torch::zeros({horizon, segments}, opts),
        .terminals = torch::zeros({horizon, segments}, opts),
        .ratio = torch::zeros({horizon, segments}, opts),
        .importance = torch::zeros({horizon, segments}, opts),
    };
}

typedef struct {
    // Layout
    int horizon;
    int total_agents;
    int num_buffers;
    // Model architecture
    int num_atns;
    int hidden_size;
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
    int cudagraphs;  // epoch at which to capture graph, -1 to disable
    bool kernels;
    bool profile;
    bool use_omp;
    // Multi-GPU
    int rank;
    int world_size;
    std::string nccl_id_path;
    // Threading
    int num_threads;
} HypersT;

typedef struct {
    Policy* policy_bf16;  // Working weights (bf16) - used for forward/backward
    Policy* policy_fp32;  // Master weights (fp32) - used for optimizer
    StaticVec* vec;
    torch::optim::Muon* muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    vector<Tensor> buffer_states;  // Per-buffer states for contiguous access
    RolloutBuf rollouts;
    EnvBuf env;
    TrainGraph train_buf;
    vector<vector<at::cuda::CUDAGraph>> fused_rollout_cudagraphs;  // [horizon][num_buffers]
    at::cuda::CUDAGraph train_cudagraph;
    at::cuda::MempoolId_t train_pool_id;     // Pool ID for releasing graph memory
    at::cuda::MempoolId_t rollout_pool_id;   // Pool ID for releasing graph memory
    vector<at::cuda::CUDAStream> torch_streams;  // PyTorch-managed streams for OMP
    Tensor act_sizes;      // CUDA int32 tensor of action head sizes for MultiDiscrete
    Tensor act_sizes_cpu;  // CPU int64 tensor (pre-computed to avoid alloc during graph replay)
    int epoch;
    int train_warmup;
    bool rollout_captured;
    bool train_captured;
    uint64_t rng_seed;
    Tensor rng_offset;  // CUDA tensor so increment is graphable
} PuffeRL;

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(32);
    static_vec_log(pufferl.vec, out);
    return out;
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

void compute_advantage(RolloutBuf& rollouts, Tensor& advantages, HypersT& hypers) {
    compute_puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
        rollouts.ratio, advantages, hypers.gamma, hypers.gae_lambda,
        hypers.vtrace_rho_clip, hypers.vtrace_c_clip);
}

// Thread initialization callback - sets CUDA stream once per thread
extern "C" void thread_init_wrapper(void* ctx, int buf) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    at::cuda::setCurrentCUDAStream(pufferl->torch_streams[buf]);
}

// Callback for OMP threadmanager - also called at init for warmup + capture
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
    torch::NoGradGuard no_grad;
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;

    profile_begin("fused_rollout", hypers.profile);
    if (pufferl->rollout_captured) {
        pufferl->fused_rollout_cudagraphs[t][buf].replay();
    } else {
        bool capturing = pufferl->epoch == hypers.cudagraphs;
        auto saved_stream = at::cuda::getCurrentCUDAStream();
        auto cap_stream = capturing ? at::cuda::getStreamFromPool() : saved_stream;
        if (capturing) {
            at::cuda::setCurrentCUDAStream(cap_stream);
            pufferl->fused_rollout_cudagraphs[t][buf].capture_begin(pufferl->rollout_pool_id);
        }

        int total_agents = pufferl->vec->total_agents;
        int num_buffers = hypers.num_buffers;
        int block_size = total_agents / num_buffers;

        Tensor obs_slice = pufferl->env.obs.narrow(0, buf*block_size, block_size);
        Tensor& state = pufferl->buffer_states[buf];

        auto [logits, value, state_out] = pufferl->policy_bf16->forward(obs_slice, state);

        RolloutBuf& rollouts = pufferl->rollouts;
        Tensor actions_out = rollouts.actions.select(0, t).narrow(0, buf*block_size, block_size);
        Tensor logprobs_out = rollouts.logprobs.select(0, t).narrow(0, buf*block_size, block_size);
        Tensor values_out = rollouts.values.select(0, t).narrow(0, buf*block_size, block_size);

        sample_actions(logits, value, actions_out, logprobs_out, values_out,
            pufferl->act_sizes, pufferl->act_sizes_cpu,
            pufferl->is_continuous, hypers.kernels, pufferl->rng_seed, pufferl->rng_offset);

        state.copy_(state_out, false);

        rollouts.observations.select(0, t).narrow(0, buf*block_size, block_size).copy_(obs_slice, true);
        rollouts.rewards.select(0, t).narrow(0, buf*block_size, block_size).copy_(
            pufferl->env.rewards.narrow(0, buf*block_size, block_size), true);
        rollouts.terminals.select(0, t).narrow(0, buf*block_size, block_size).copy_(
            pufferl->env.terminals.narrow(0, buf*block_size, block_size), true);

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
    PuffeRL* pufferl = (PuffeRL*)ctx;
    at::cuda::setCurrentCUDAStream(pufferl->torch_streams[buf]);
}

// Callback for OMP threadmanager - runs policy forward for one (buf, t) step
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
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
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl_impl(HypersT& hypers, const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
  BEGIN_LIBTORCH_CATCH
    auto pufferl = std::make_unique<pufferlib::PuffeRL>();
    pufferl->hypers = hypers;
    pufferl->nccl_comm = nullptr;

    // Multi-GPU: initialize NCCL (device already set by Python)
    if (hypers.world_size > 1) {
        ncclUniqueId nccl_id;
        if (hypers.rank == 0) {
            ncclGetUniqueId(&nccl_id);
            FILE* f = fopen(hypers.nccl_id_path.c_str(), "wb");
            fwrite(&nccl_id, sizeof(nccl_id), 1, f);
            fclose(f);
        }
        // Wait for rank 0 to write the ID file
        while (access(hypers.nccl_id_path.c_str(), F_OK) != 0) {
            usleep(10000);  // 10ms
        }
        if (hypers.rank != 0) {
            // Small delay to ensure file is fully written
            usleep(50000);
            FILE* f = fopen(hypers.nccl_id_path.c_str(), "rb");
            fread(&nccl_id, sizeof(nccl_id), 1, f);
            fclose(f);
        }

        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    // Seeding (vary by rank for different random exploration)
    // CC: Base seed should come from train config
    int seed = 42 + hypers.rank;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    pufferl->rng_seed = seed;
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

    // Determine if action space is continuous or discrete
    // Continuous: all action dimensions have size 1
    // Discrete: all action dimensions have size > 1
    // Mixed: not supported (assert)
    int* act_sizes_ptr = get_act_sizes();
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        if (act_sizes_ptr[i] == 1) {
            num_continuous++;
        } else {
            num_discrete++;
        }
    }
    TORCH_CHECK(num_continuous == 0 || num_discrete == 0,
        "Mixed continuous/discrete action spaces not supported. "
        "All action dimensions must be either continuous (size==1) or discrete (size>1). "
        "Got ", num_continuous, " continuous and ", num_discrete, " discrete.");
    pufferl->is_continuous = (num_continuous > 0);
    if (pufferl->is_continuous) {
        printf("Detected continuous action space with %d dimensions\n", num_action_heads);
    } else {
        printf("Detected discrete action space with %d heads\n", num_action_heads);
    }

    int input_size = pufferl->env.obs.size(1);
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    bool kernels = hypers.kernels;

    // Decoder output size: discrete = act_n (sum of action sizes), continuous = num_action_heads
    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;

    // Create fp32 master policy (for optimizer - precise gradient accumulation)
    Policy* policy_fp32 = create_policy(env_name, input_size, hidden_size,
        decoder_output_size, num_layers, act_n, is_continuous, kernels);
    policy_fp32->to(torch::kCUDA);
    policy_fp32->to(torch::kFloat32);
    pufferl->policy_fp32 = policy_fp32;

    if (USE_BF16) {
        // create bf16 working policy (for fwd/bwd)
        Policy* policy_bf16 = create_policy(env_name, input_size, hidden_size,
            decoder_output_size, num_layers, act_n, is_continuous, kernels);
        policy_bf16->to(torch::kCUDA);
        policy_bf16->to(torch::kBFloat16);
        pufferl->policy_bf16 = policy_bf16;
        sync_policy_weights(policy_bf16, policy_fp32); // initial sync
    } else {
        pufferl->policy_bf16 = policy_fp32;
    }

    // Optimizer uses fp32 master weights for precise gradient accumulation
    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    float eps = hypers.eps;
    pufferl->muon = new torch::optim::Muon(policy_fp32->parameters(),
        torch::optim::MuonOptions(lr).momentum(beta1).eps(eps));
    pufferl->muon->init_contiguous_weights();
    pufferl->muon->nccl_comm = pufferl->nccl_comm;
    pufferl->muon->world_size = hypers.world_size;
    printf("DEBUG: Contiguous weight buffer: %ld elements\n", pufferl->muon->weight_buffer.numel());


    // Allocate buffers
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    printf("DEBUG: num_envs=%d, total_agents=%d, batch=%d, num_buffers=%d\n",
        vec->size, total_agents, batch, num_buffers);

    int minibatch_segments = hypers.minibatch_size / horizon;

    pufferl->rollouts = create_rollouts(horizon, total_agents, input_size, num_action_heads);
    pufferl->train_buf = create_train_graph(minibatch_segments, horizon, input_size,
        num_layers, hidden_size, num_action_heads);

    // Per-buffer states: each is {num_layers, block_size, hidden} for contiguous access
    pufferl->buffer_states.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_states[i] = pufferl->policy_bf16->initial_state(batch, torch::kCUDA);
    }

    if (hypers.cudagraphs >= 0) {
        pufferl->train_cudagraph = at::cuda::CUDAGraph();
        pufferl->train_pool_id = at::cuda::graph_pool_handle();
        pufferl->train_warmup = 0;

        // Fused rollout cudagraphs: [horizon][num_buffers]
        pufferl->rollout_pool_id = at::cuda::graph_pool_handle();
        pufferl->fused_rollout_cudagraphs.resize(horizon);
        for (int h = 0; h < horizon; ++h) {
            pufferl->fused_rollout_cudagraphs[h].resize(num_buffers);
            for (int b = 0; b < num_buffers; ++b) {
                pufferl->fused_rollout_cudagraphs[h][b] = at::cuda::CUDAGraph();
            }
        }

        // Snapshot weights + optimizer state before init-time capture
        Tensor saved_weights = pufferl->muon->weight_buffer.clone();
        Tensor saved_momentum;
        if (pufferl->muon->momentum_buffer.defined()) {
            saved_momentum = pufferl->muon->momentum_buffer.clone();
        }

        // Run warmup + capture on a fresh stream (matching original capture_graph).
        // Tensors get associated with warmup_stream, not the default stream.
        // Captured graphs' event-waits reference warmup_stream which is dead at runtime.
        auto saved_stream = at::cuda::getCurrentCUDAStream();
        auto warmup_stream = at::cuda::getStreamFromPool();
        at::cuda::setCurrentCUDAStream(warmup_stream);

        // Init-time warmup + capture BEFORE creating streams/threads.
        // No per-buffer streams exist yet = no cross-stream deps baked into graphs.
        for (pufferl->epoch = 0; pufferl->epoch <= hypers.cudagraphs; pufferl->epoch++) {
            rollouts_impl(*pufferl);
        }
        pufferl->rollout_captured = true;

        for (int i = 0; i <= hypers.cudagraphs; i++) {
            train_impl(*pufferl);
        }

        warmup_stream.synchronize();
        cudaDeviceSynchronize();
        at::cuda::setCurrentCUDAStream(saved_stream);

        // Restore weights + optimizer state corrupted by warmup/capture
        {
        torch::NoGradGuard no_grad;
        pufferl->muon->weight_buffer.copy_(saved_weights);
        if (saved_momentum.defined()) {
            pufferl->muon->momentum_buffer.copy_(saved_momentum);
        } else {
            pufferl->muon->momentum_buffer = Tensor();
        }
        if (USE_BF16) {
            sync_policy_weights(pufferl->policy_bf16, pufferl->policy_fp32);
        }
        pufferl->muon->zero_grad();
        if (USE_BF16) {
            pufferl->policy_bf16->zero_grad();
        }
        } // end NoGradGuard

        pufferl->epoch = 0;
    }

    // Create PyTorch-managed streams and assign to vec
    for (int i = 0; i < num_buffers; i++) {
        pufferl->torch_streams.push_back(at::cuda::getStreamFromPool(false));
        vec->streams[i] = pufferl->torch_streams[i].stream();
    }

    // Static breakout - OMP only
    if (hypers.use_omp) {
        create_static_threads(vec, hypers.num_threads, horizon, pufferl.get(), net_callback_wrapper, thread_init_wrapper);
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
}


void train_impl(PuffeRL& pufferl) {
    // Update to HypersT& p
    HypersT& hypers = pufferl.hypers;

    // Clear buffer states (releases CUDA tensors)
    pufferl.buffer_states.clear();

    // Clear rollout buffers (releases CUDA tensors)
    pufferl.rollouts.observations = Tensor();
    pufferl.rollouts.actions = Tensor();
    pufferl.rollouts.values = Tensor();
    pufferl.rollouts.logprobs = Tensor();
    pufferl.rollouts.rewards = Tensor();
    pufferl.rollouts.terminals = Tensor();
    pufferl.rollouts.ratio = Tensor();
    pufferl.rollouts.importance = Tensor();

    // Clear train buffers (releases CUDA tensors)
    pufferl.train_buf.mb_obs = Tensor();
    pufferl.train_buf.mb_state = Tensor();
    pufferl.train_buf.mb_actions = Tensor();
    pufferl.train_buf.mb_logprobs = Tensor();
    pufferl.train_buf.mb_advantages = Tensor();
    pufferl.train_buf.mb_prio = Tensor();
    pufferl.train_buf.mb_values = Tensor();
    pufferl.train_buf.mb_returns = Tensor();
    pufferl.train_buf.mb_ratio = Tensor();
    pufferl.train_buf.mb_newvalue = Tensor();

    // Clear misc tensors
    pufferl.act_sizes = Tensor();
    pufferl.act_sizes_cpu = Tensor();
    pufferl.rng_offset = Tensor();

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
}

// nsys capture control (--capture-range=cudaProfilerApi). Different from profile_begin/end which are nvtx ranges.
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
