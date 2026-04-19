#import "metal_platform.h"
#include "metal_kernels.mm"
#include "cpu_inference.h"
#include "vecenv.h"


#include <cstring>
#include <memory>
#include <mutex>
#include <sys/time.h>
#include <vector>

static thread_local cudaStream_t tl_rollout_stream = 0;
static std::mutex g_rollout_profile_mutex;

static inline float prof_ms(uint64_t t0, uint64_t t1) {
    static mach_timebase_info_data_t tb;
    static std::once_flag tb_once;
    std::call_once(tb_once, []() { mach_timebase_info(&tb); });
    return (float)((double)(t1 - t0) * tb.numer / tb.denom / 1e6);
}

int obs_dtype_size(int dtype) {
    switch (dtype) {
    case FLOAT:
        return sizeof(float);
    case INT:
        return sizeof(int32_t);
    case DOUBLE:
        return sizeof(double);
    case UNSIGNED_CHAR:
    case CHAR:
        return sizeof(char);
    default:
        assert(false && "Unsupported observation dtype");
        return 0;
    }
}

template <typename T>
static inline void cpu_cast_to_f32(float* dst, const T* src, int count) {
    for (int i = 0; i < count; i++) {
        dst[i] = (float)src[i];
    }
}

// ============================================================================
// Environment creation — unified memory, no GPU copy needed
// ============================================================================

StaticVec* create_environments(int num_buffers, int total_agents,
        const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    StaticVec* vec = create_static_vec(total_agents, num_buffers, /*gpu=*/0, vec_kwargs, env_kwargs);

    int obs_size = get_obs_size();
    int num_atns = get_num_atns();
    int obs_type = get_obs_type();

    // Unified memory: env obs/actions/rewards/terminals point directly at vecenv buffers
    env.obs = {.bytes = (char*)vec->gpu_observations, .shape = {total_agents, obs_size}, .dtype_size = obs_dtype_size(obs_type)};
    env.obs_raw_dtype = obs_type;
    env.actions = {.bytes = (char*)vec->gpu_actions, .shape = {total_agents, num_atns}, .dtype_size = (int)sizeof(float)};
    env.rewards = {.data = vec->gpu_rewards, .shape = {total_agents}};
    env.terminals = {.data = vec->gpu_terminals, .shape = {total_agents}};

    return vec;
}

// ============================================================================
// Hyperparameters — single GPU only, no NCCL
// ============================================================================

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
    // Optimizer (Muon only — Adam removed)
    float beta1;
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
    bool reset_state;
    bool profile;
    bool overlap;  // async training overlap: train on separate GPU queue
    bool cpu_inference;  // CPU forward pass during rollout (no GPU sync)
    bool train_fp16;     // fp16 activations/grads during training (rollout stays fp32)
    // Single GPU (Metal has no multi-GPU, but kept for upstream compat)
    int gpu_id;
    // Threading
    int num_threads;
    // RNG seed
    uint64_t seed;
} HypersT;

// ============================================================================
// Profiling — CPU-based timing via mach_absolute_time
// ============================================================================

enum ProfileIdx {
    PROF_ROLLOUT = 0,
    PROF_EVAL_GPU,
    PROF_EVAL_ENV,
    // Fine-grained rollout sub-phases
    PROF_ROLLOUT_OBS_COPY,
    PROF_ROLLOUT_FWD,
    PROF_ROLLOUT_ACT_COPY,
    // Fine-grained training sub-phases
    PROF_TRAIN_PRELOOP,
    PROF_TRAIN_PRIO,
    PROF_TRAIN_SELECT,
    PROF_TRAIN_FWD,
    PROF_TRAIN_PPO,
    PROF_TRAIN_BACKWARD,
    PROF_TRAIN_GRAD_COPY,
    PROF_TRAIN_GRAD_CLIP,
    PROF_TRAIN_MUON,
    PROF_TRAIN_SYNC,
    NUM_PROF,
};

static const char* PROF_NAMES[NUM_PROF] = {
    "rollout",
    "eval_gpu",
    "eval_env",
    "rollout_obs_copy",
    "rollout_fwd",
    "rollout_act_copy",
    "train_preloop",
    "train_prio",
    "train_select",
    "train_fwd",
    "train_ppo",
    "train_backward",
    "train_grad_copy",
    "train_grad_clip",
    "train_muon",
    "train_sync",
};

typedef struct {
    float accum[NUM_PROF];
} ProfileT;

// ============================================================================
// PuffeRL state — Metal version (no CUDA graphs, NCCL, nvml, multi-stream)
// ============================================================================

struct PuffeRL {
    Policy* policy;
    PolicyWeights weights_fp32;
    PolicyWeights weights_fp16;  // fp16 training weights
    // Double-buffered inference weights for rollout/training overlap.
    // Rollout reads weights_infer (GPU compute), training writes weights_fp32 (GPU).
    // After each training sync, weights_fp32 is memcpy'd to weights_infer.
    PolicyWeights weights_infer;
    Allocator infer_params_alloc;
    bool overlap_enabled = false;
    bool train_pending = false;  // GPU training dispatched but not yet synced
    PolicyActivations train_activations;
    AllocSet alloc_fp32;
    AllocSet alloc_fp16;  // fp16 training: weights, activations, gradients
    Allocator pufferl_alloc;
    StaticVec* vec;
    Muon* muon;
    HypersT hypers;
    bool is_continuous;
    std::vector<cudaStream_t> rollout_streams;
    std::vector<PufTensor> buffer_states;
    std::vector<FloatTensor> sample_act_f32_buffers;
    std::vector<PolicyActivations> buffer_activations;
    std::vector<Allocator> buffer_allocs;
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;
    EnvBuf env;
    TrainGraph train_buf;
    FloatTensor old_values_puf;
    FloatTensor advantages_puf;
    IntTensor act_sizes_puf;
    FloatTensor losses_puf;
    PPOBuffersPuf ppo_bufs_puf;
    PrioBuffers prio_bufs;
    FloatTensor param_fp32_puf;
    PufTensor param_fp16_puf;   // fp16 weight buffer (flat view)
    PufTensor grad_fp16_puf;  // gradient buffer (fp16 when train_fp16, else fp32)
    FloatTensor grad_norm_puf;
    LongTensor rng_offset_puf;
    // fp16 boundary buffers: obs cast (fp32->fp16), dec_out cast (fp16->fp32),
    // state (zeroed fp16 for scan initial state)
    PufTensor fp16_obs_buf;
    PufTensor fp32_dec_out_buf;
    PufTensor fp16_state_buf;
    Allocator fp16_boundary_alloc;
    ProfileT profile;
    int rollout_sync_count;
    double rollout_sync_ms;
    int train_sync_count;
    double train_sync_ms;
    int epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    uint64_t rng_seed;
    // Action mask: true if obs embeds a mask in the last act_n columns.
    // When false, a static all-ones buffer is used instead.
    bool has_mask = false;
    int env_obs_width = 0;  // raw obs width from env (e.g. 1096 = features + mask)
    int mask_width = 0;     // total action mask width = sum of all action head sizes (e.g. 79)
    FloatTensor ones_mask;  // (act_n) all 1.0f, fallback mask when !has_mask
    // External mask path: when has_mask, masks are split from obs at rollout time.
    FloatTensor rollout_masks;
    FloatTensor train_masks;
    FloatTensor mb_masks;
    bool cpu_inference = false;  // CPU forward pass for rollout (no GPU sync)
    bool train_fp16 = false;     // fp16 training activations/grads
    // Decoder logits + f32 actions for GPU logprob recompute (cpu_inference only).
    FloatTensor rollout_logits;    // (horizon, total_agents, fused_cols)
    FloatTensor train_logits;      // (total_agents, horizon, fused_cols)
    FloatTensor rollout_actions_f32; // (horizon, total_agents, num_atns)
    FloatTensor train_actions_f32;   // (total_agents, horizon, num_atns)
};

// ============================================================================
// Logging
// ============================================================================

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(128);
    static_vec_log(pufferl.vec, out);
    return out;
}

// ============================================================================
// Per-buffer thread init — called once per buffer thread at creation
// ============================================================================

extern "C" void thread_init_metal(void* ctx, int buf) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    assert(buf >= 0 && buf < (int)pufferl->rollout_streams.size());
    tl_rollout_stream = pufferl->rollout_streams[buf];
    assert(tl_rollout_stream && "thread_init_metal requires per-buffer stream");
}

// ============================================================================
// Rollout callback — called per buffer per horizon step
// ============================================================================

extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
  @autoreleasepool {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;

    RolloutBuf& rollouts = pufferl->rollouts;
    EnvBuf& env = pufferl->env;
    int block_size = pufferl->vec->total_agents / hypers.num_buffers;
    int start = buf * block_size;
    cudaStream_t stream = tl_rollout_stream;
    assert(stream && "rollout callback requires thread-local stream");

    uint64_t tp0 = mach_absolute_time();

    // Copy env obs to rollout buffer, splitting features and mask.
    // env writes [features | mask] at env_obs_width stride.
    // rollout obs stores only features at input_size stride.
    // rollout masks stores only mask at act_n stride.
    PufTensor& obs_env = env.obs;
    int env_obs_width = pufferl->env_obs_width;
    int input_size = (int)rollouts.observations.shape[2];
    int mask_w = pufferl->mask_width;

    FloatTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);

    if (pufferl->has_mask) {
        // split copy: features prefix + mask suffix, row by row
        FloatTensor mask_dst = puf_slice(pufferl->rollout_masks, t, start, block_size);
        assert(pufferl->env.obs_raw_dtype == FLOAT && "mask split only supports float32 obs");
        const float* src_base = (const float*)(obs_env.bytes + (int64_t)start * env_obs_width * sizeof(float));
        float* feat_base = obs_dst.data;
        float* mask_base = mask_dst.data;
        for (int b = 0; b < block_size; b++) {
            memcpy(feat_base + b * input_size, src_base + b * env_obs_width,
                   input_size * sizeof(float));
            memcpy(mask_base + b * mask_w, src_base + b * env_obs_width + input_size,
                   mask_w * sizeof(float));
        }
    } else {
        // no mask: copy full obs (env_obs_width == input_size)
        PufTensor obs_src = {
            .bytes = obs_env.bytes + (int64_t)start * env_obs_width * obs_env.dtype_size,
            .shape = {block_size, env_obs_width},
            .dtype_size = obs_env.dtype_size
        };
        int count = (int)obs_src.numel();
        switch (pufferl->env.obs_raw_dtype) {
        case UNSIGNED_CHAR:
            cpu_cast_u8_to_f32(obs_dst.data, (const uint8_t*)obs_src.bytes,
                count);
            break;
        case CHAR:
            cpu_cast_to_f32(obs_dst.data, (const int8_t*)obs_src.bytes, count);
            break;
        case FLOAT:
            memcpy(obs_dst.data, obs_src.bytes, obs_src.numel() * obs_src.dtype_size);
            break;
        case INT:
            cpu_cast_to_f32(obs_dst.data, (const int32_t*)obs_src.bytes, count);
            break;
        case DOUBLE:
            cpu_cast_to_f32(obs_dst.data, (const double*)obs_src.bytes, count);
            break;
        default:
            assert(false && "Unsupported observation dtype");
        }
    }

    // Rewards + terminals -- direct memcpy, no sync check needed
    FloatTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    memcpy(rew_dst.data, env.rewards.data + start, block_size * sizeof(float));

    FloatTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    memcpy(term_dst.data, env.terminals.data + start, block_size * sizeof(float));

    uint64_t tp1 = mach_absolute_time();

    // Forward pass + sampling
    FloatTensor act_slice = puf_slice(rollouts.actions, t, start, block_size);
    FloatTensor lp_slice = puf_slice(rollouts.logprobs, t, start, block_size);
    FloatTensor val_slice = puf_slice(rollouts.values, t, start, block_size);
    int num_atns = (int)puf_numel(pufferl->act_sizes_puf.shape);
    uint32_t* buf_rng_offset = (uint32_t*)(pufferl->rng_offset_puf.data + buf);
    uint64_t buf_rng_seed = pufferl->rng_seed + buf;

    PufTensor state_puf = pufferl->buffer_states[buf];
    PolicyWeights& infer_weights = pufferl->overlap_enabled
        ? pufferl->weights_infer : pufferl->weights_fp32;
    Policy* p = pufferl->policy;
    PolicyActivations& acts = pufferl->buffer_activations[buf];
    FloatTensor& act_f32_buf = pufferl->sample_act_f32_buffers[buf];
    // Mask pointer setup for sampling
    int fused_cols = ((DecoderWeights *)infer_weights.decoder)->output_dim + 1;
    const float* mask_ptr;
    int mask_stride;
    if (pufferl->has_mask) {
        FloatTensor mask_slice = puf_slice(pufferl->rollout_masks, t, start, block_size);
        mask_ptr = mask_slice.data;
        mask_stride = pufferl->mask_width;
    } else {
        mask_ptr = pufferl->ones_mask.data;
        mask_stride = 0;
    }

    if (pufferl->cpu_inference) {
        // CPU path: cblas_sgemm + scalar gate + CPU sampling. No GPU, no sync.
        // cpu_forward_and_sample still takes PufTensor for obs_dst -- wrap FloatTensor
        PufTensor obs_puf = {.bytes = (char*)obs_dst.data, .shape = {obs_dst.shape[0], obs_dst.shape[1]}, .dtype_size = (int)sizeof(float)};
        cpu_forward_and_sample(
            obs_puf, state_puf, infer_weights, hypers.hidden_size, acts,
            pufferl->act_sizes_puf, act_f32_buf,
            lp_slice.data, val_slice.data,
            mask_ptr, mask_stride,
            buf_rng_seed, buf_rng_offset);

        // Store decoder logits + f32 actions for GPU logprob recompute at
        // training start. CPU sampling uses IEEE expf, PPO uses GPU fast::exp.
        DecoderActivations *da = (DecoderActivations *)acts.decoder;
        FloatTensor logits_dst = puf_slice(pufferl->rollout_logits, t, start, block_size);
        memcpy(logits_dst.data, da->out.data, block_size * fused_cols * sizeof(float));
        FloatTensor acts_f32_dst = puf_slice(pufferl->rollout_actions_f32, t, start, block_size);
        memcpy(acts_f32_dst.data, act_f32_buf.data, block_size * num_atns * sizeof(float));

        memcpy(act_slice.data, act_f32_buf.data, block_size * num_atns * sizeof(float));
    } else {
        // GPU path: Metal dispatch + sync (original behavior)
        PrecisionTensor obs_pt = {
            .data = obs_dst.data,
            .shape = {obs_dst.shape[0], obs_dst.shape[1]},
            .dtype_size = (int)sizeof(float),
        };
        PrecisionTensor state_pt = {
            .data = (float*)state_puf.bytes,
            .shape = {state_puf.shape[0], state_puf.shape[1], state_puf.shape[2]},
            .dtype_size = state_puf.dtype_size,
        };
        PrecisionTensor mingru_input = p->encoder.forward(infer_weights.encoder, acts.encoder, obs_pt, stream);
        PrecisionTensor h = p->network.forward(infer_weights.network, mingru_input, state_pt, acts.network, stream);
        PrecisionTensor dec_pt = p->decoder.forward(infer_weights.decoder, acts.decoder, h, stream);

        mtl_sample_logits_dispatch_to(
            dec_pt, pufferl->act_sizes_puf,
            act_f32_buf.data, lp_slice.data, val_slice.data,
            mask_ptr, mask_stride,
            buf_rng_seed, buf_rng_offset, stream);

        mtl_ensure_stream_synced(stream);

        // Stash logits + f32 actions for logprob recompute when train_fp16=1.
        // Rollout uses fp32 weights; training uses fp16 → precision mismatch in
        // PPO ratio unless we recompute old_logprobs in fp16 at training start.
        if (pufferl->train_fp16 && pufferl->rollout_logits.data) {
            DecoderActivations *da = (DecoderActivations *)acts.decoder;
            FloatTensor logits_dst = puf_slice(pufferl->rollout_logits, t, start, block_size);
            memcpy(logits_dst.data, da->out.data, block_size * fused_cols * sizeof(float));
            FloatTensor acts_f32_dst = puf_slice(pufferl->rollout_actions_f32, t, start, block_size);
            memcpy(acts_f32_dst.data, act_f32_buf.data, block_size * num_atns * sizeof(float));
        }

        memcpy(act_slice.data, act_f32_buf.data, block_size * num_atns * sizeof(float));
    }

    // Match upstream: do not zero RNN state on terminal.

    uint64_t tp2 = mach_absolute_time();

    /* copy float32 actions to env buffer (actions are float after upstream 4.0 migration).
       use act_f32_buf (already float) instead of act_slice (which stores doubles in rollout buf). */
    int64_t act_cols = env.actions.shape[1];
    memcpy(
        env.actions.bytes + start * act_cols * sizeof(float),
        act_f32_buf.data,
        block_size * act_cols * sizeof(float));

    uint64_t tp3 = mach_absolute_time();

    // Accumulate fine-grained rollout timing (callbacks run concurrently).
    {
        std::lock_guard<std::mutex> lk(g_rollout_profile_mutex);
        pufferl->profile.accum[PROF_ROLLOUT_OBS_COPY] += prof_ms(tp0, tp1);
        pufferl->profile.accum[PROF_ROLLOUT_FWD] += prof_ms(tp1, tp2);
        pufferl->profile.accum[PROF_ROLLOUT_ACT_COPY] += prof_ms(tp2, tp3);
    }
  } // @autoreleasepool
}

// ============================================================================
// Weight copy: weights_fp32 → weights_infer (for rollout/training overlap)
// ============================================================================

static void copy_weights_to_infer(PuffeRL& pufferl) {
    int64_t nbytes = pufferl.alloc_fp32.params.total_elems * sizeof(float);
    memcpy(pufferl.infer_params_alloc.mem, pufferl.alloc_fp32.params.mem, nbytes);
}

// ============================================================================
// Forward declaration: waits for async GPU training to complete.
static void sync_pending_train(PuffeRL& pufferl);

// ============================================================================
// Training loop
// ============================================================================

void train_impl(PuffeRL& pufferl) {
    HypersT& hypers = pufferl.hypers;
    uint64_t tp_preloop0 = mach_absolute_time();

    cudaStream_t train_stream = pufferl.overlap_enabled
        ? (cudaStream_t)mtl_train_stream()
        : (cudaStream_t)mtl_stream();

    // GPU training: keep all ops on the Metal encoder (GEMM, copy, zero, add).
    puf_set_gpu_training(true);

    // Transpose rollouts from (horizon, segments, ...) to (segments, horizon, ...)
    RolloutBuf& src = pufferl.rollouts;
    RolloutBuf& rollouts = pufferl.train_rollouts;

    puf_transpose_01(rollouts.observations, src.observations, train_stream);
    puf_transpose_01(rollouts.actions, src.actions, train_stream);
    puf_transpose_01(rollouts.logprobs, src.logprobs, train_stream);
    puf_transpose_01(rollouts.rewards, src.rewards, train_stream);
    puf_transpose_01(rollouts.terminals, src.terminals, train_stream);
    puf_transpose_01(rollouts.ratio, src.ratio, train_stream);
    puf_transpose_01(rollouts.values, src.values, train_stream);
    if (pufferl.has_mask)
        puf_transpose_01(pufferl.train_masks, pufferl.rollout_masks, train_stream);

    // Metal 4: ensure all rollout transposes are visible before consumers read them.
    mtl_barrier((MetalStream*)train_stream);

    // Recompute old logprobs when rollout and training use different math.
    if (pufferl.cpu_inference || pufferl.train_fp16) {
        puf_transpose_01(pufferl.train_logits, pufferl.rollout_logits, train_stream);
        puf_transpose_01(pufferl.train_actions_f32, pufferl.rollout_actions_f32, train_stream);
        mtl_barrier((MetalStream*)train_stream);

        int total_samples = hypers.total_agents * hypers.horizon;
        int fused_cols = (int)pufferl.train_logits.shape[2];
        int num_atns = (int)pufferl.train_actions_f32.shape[2];

        // Mask: embedded in obs or all-ones fallback
        const float *mask_ptr;
        int mask_stride;
        if (pufferl.has_mask) {
            mask_ptr = pufferl.train_masks.data;
            mask_stride = pufferl.mask_width;
        } else {
            mask_ptr = pufferl.ones_mask.data;
            mask_stride = 0;
        }

        mtl_recompute_logprobs(
            rollouts.logprobs.data,
            pufferl.train_logits.data,
            pufferl.train_actions_f32.data,
            pufferl.act_sizes_puf.data,
            mask_ptr, mask_stride,
            total_samples, num_atns, fused_cols, train_stream);
    }

    // Clamp rewards and fill ratio (f32 path only, no bf16)
    mtl_clamp_f32(rollouts.rewards.data, -1.0f, 1.0f,
                  (int)puf_numel(rollouts.rewards.shape), train_stream);
    mtl_fill_f32(rollouts.ratio.data, 1.0f,
                 (int)puf_numel(rollouts.ratio.shape), train_stream);

    // old_values = values.clone()
    puf_copy(pufferl.old_values_puf, rollouts.values, train_stream);
    // Metal 4 visibility boundary before minibatch loop consumes transposed rollouts.
    mtl_barrier((MetalStream*)train_stream);

    int batch_size = hypers.total_agents * hypers.horizon;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    float prio_alpha = hypers.prio_alpha;
    int current_epoch = pufferl.epoch;
    int total_epochs = hypers.total_timesteps / batch_size;
    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;

    if (hypers.anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, total_epochs);
        float* lr_ptr = pufferl.muon->lr_ptr;
        *lr_ptr = lr;
    }

    float anneal_beta = hypers.prio_beta0 + (1.0f - hypers.prio_beta0)
        * prio_alpha * (float)current_epoch / (float)total_epochs;

    uint64_t tp_preloop1 = mach_absolute_time();
    pufferl.profile.accum[PROF_TRAIN_PRELOOP] += prof_ms(tp_preloop0, tp_preloop1);

    // Single minibatch step shared by overlap and non-overlap.
    auto run_minibatch = [&](cudaStream_t s, uint32_t* rng_offset, bool gpu_profile) {
        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp0 = mach_absolute_time();

        puf_zero(&pufferl.advantages_puf, s);
        puff_advantage(rollouts.values, rollouts.rewards, rollouts.terminals,
            rollouts.ratio, pufferl.advantages_puf, hypers.gamma, hypers.gae_lambda,
            hypers.vtrace_rho_clip, hypers.vtrace_c_clip, s);

        prio_precompute(pufferl.advantages_puf, prio_alpha, pufferl.prio_bufs, s);
        prio_sample(minibatch_segments, hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.rng_seed, rng_offset, s);
        mtl_barrier((MetalStream*)s);

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp2 = mach_absolute_time();

        if (hypers.reset_state) puf_zero(&pufferl.train_buf.mb_state, s);
        {
            RolloutBuf sel_src = rollouts;
            sel_src.values = pufferl.old_values_puf;
            mtl_select_copy(sel_src, pufferl.train_buf,
                (const int64_t*)pufferl.prio_bufs.idx.data,
                pufferl.advantages_puf.data,
                pufferl.prio_bufs.mb_prio.data,
                minibatch_segments,
                pufferl.fp16_obs_buf.bytes, s);
            // gather masks from train_masks into mb_masks using same priority indices.
            // reuses index_copy_kernel as a gather: dst[i] = src[idx[i]].
            if (pufferl.has_mask) {
                MetalStream *ms2 = mtl_resolve_stream(s);
                ms2->compute_encoder();
                auto pso = mtl_pipeline("index_gather_kernel");
                mtl_set_pso(ms2, pso);
                int mw = pufferl.mask_width;
                int mask_seg_bytes = hypers.horizon * mw * (int)sizeof(float);
                mtl_set_ptr(ms2, pufferl.mb_masks.data, 0);
                mtl_set_ptr(ms2, (void*)pufferl.prio_bufs.idx.data, 1);
                mtl_set_ptr(ms2, pufferl.train_masks.data, 2);
                struct { int num_idx; int row_bytes; } mp = {minibatch_segments, mask_seg_bytes};
                mtl_set_params(ms2, mp, 3);
                mtl_dispatch_groups(ms2, pso, (minibatch_segments + 255) / 256, 256);
            }
        }
        mtl_barrier((MetalStream*)s);

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp3 = mach_absolute_time();

        PolicyWeights& train_weights = pufferl.train_fp16 ? pufferl.weights_fp16 : pufferl.weights_fp32;
        PrecisionTensor obs_pt;
        PrecisionTensor state_pt;
        if (pufferl.train_fp16) {
            obs_pt = {
                .data = (float*)pufferl.fp16_obs_buf.bytes,
                .shape = {pufferl.fp16_obs_buf.shape[0], pufferl.fp16_obs_buf.shape[1], pufferl.fp16_obs_buf.shape[2]},
                .dtype_size = pufferl.fp16_obs_buf.dtype_size,
            };
            state_pt = {
                .data = (float*)pufferl.fp16_state_buf.bytes,
                .shape = {pufferl.fp16_state_buf.shape[0], pufferl.fp16_state_buf.shape[1], pufferl.fp16_state_buf.shape[2], pufferl.fp16_state_buf.shape[3]},
                .dtype_size = pufferl.fp16_state_buf.dtype_size,
            };
        } else {
            FloatTensor &mo = pufferl.train_buf.mb_obs;
            obs_pt = {.data = mo.data, .shape = {mo.shape[0], mo.shape[1], mo.shape[2]}, .dtype_size = (int)sizeof(float)};
            FloatTensor &ms = pufferl.train_buf.mb_state;
            state_pt = {.data = ms.data, .shape = {ms.shape[0], ms.shape[1], ms.shape[2], ms.shape[3]}, .dtype_size = (int)sizeof(float)};
        }
        if (pufferl.train_fp16 && hypers.reset_state) puf_zero(&pufferl.fp16_state_buf, s);

        PrecisionTensor dec_pt = policy_forward_train(pufferl.policy, train_weights,
            pufferl.train_activations, obs_pt, state_pt, s);

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp4 = mach_absolute_time();

        PufTensor dec_puf = to_puf(dec_pt);
        if (dec_puf.dtype_size == 2) {
            mtl_cast_f16_to_f32((float*)pufferl.fp32_dec_out_buf.bytes,
                                dec_puf.bytes,
                                (int)dec_puf.numel(), s);
            mtl_barrier((MetalStream*)s);
            dec_puf = pufferl.fp32_dec_out_buf;
        }

        PrecisionTensor p_logstd = {.dtype_size = (int)sizeof(float)};
        if (pufferl.is_continuous) {
            p_logstd = ((DecoderWeights*)pufferl.weights_fp32.decoder)->logstd;
        }

        {
            const float* ppo_mask_ptr;
            int ppo_mask_stride;
            if (pufferl.has_mask) {
                ppo_mask_ptr = pufferl.mb_masks.data;
                ppo_mask_stride = pufferl.mask_width;
            } else {
                ppo_mask_ptr = pufferl.ones_mask.data;
                ppo_mask_stride = 0;
            }
            // When PER is active, pass full-batch advantages for unbiased var/mean.
            const FloatTensor *full_adv = (prio_alpha > 0.0f) ? &pufferl.advantages_puf : nullptr;
            PufTensor logstd_puf = to_puf(p_logstd);
            ppo_loss_fwd_bwd(dec_puf, logstd_puf, pufferl.train_buf,
                pufferl.act_sizes_puf, pufferl.losses_puf,
                hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef, hypers.ent_coef,
                pufferl.ppo_bufs_puf, pufferl.is_continuous,
                ppo_mask_ptr, ppo_mask_stride, full_adv, s);
        }
        mtl_barrier((MetalStream*)s);

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp5 = mach_absolute_time();

        // policy_backward now takes FloatTensor grads directly (matching upstream)
        FloatTensor grad_logstd_ft = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : FloatTensor();
        policy_backward(pufferl.policy, train_weights, pufferl.train_activations,
            pufferl.ppo_bufs_puf.grad_logits, grad_logstd_ft,
            pufferl.ppo_bufs_puf.grad_values, s);

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp6 = mach_absolute_time();

        FloatTensor& gc = pufferl.muon->gc_puf;
        mtl_barrier((MetalStream*)s);  // policy_backward writes grads, copy/cast reads them
        if (pufferl.grad_fp16_puf.dtype_size == 2) {
            mtl_cast_f16_to_f32(gc.data,
                                pufferl.grad_fp16_puf.bytes,
                                (int)pufferl.grad_fp16_puf.numel(), s);
        } else {
            // grad_fp16_puf is fp32 when !train_fp16, copy to gc
            mtl_copy_f32(gc.data, (const float*)pufferl.grad_fp16_puf.bytes,
                         (int)pufferl.grad_fp16_puf.numel(), s);
        }

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp7 = mach_absolute_time();

        mtl_barrier((MetalStream*)s);
        {
            float* scratch = pufferl.grad_norm_puf.data;
            clip_grad_norm_f32(gc, scratch, hypers.max_grad_norm, 1e-6f, s);
        }

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp8 = mach_absolute_time();

        mtl_barrier((MetalStream*)s);
        muon_step(pufferl.muon, s);


        if (pufferl.train_fp16) {
            mtl_cast_f32_to_f16(pufferl.param_fp16_puf.bytes,
                                (const float*)pufferl.alloc_fp32.params.mem,
                                (int)pufferl.alloc_fp32.params.total_elems, s);
        }

        mtl_barrier((MetalStream*)s);

        // Scatter mb_ratio and mb_newvalue back into rollout buffers so
        // subsequent minibatches see updated importance weights and values.
        // Matches CUDA upstream (pufferlib.cu:1416-1428).
        mtl_scatter_ppo_outputs(pufferl.train_buf, rollouts,
            (const int64_t*)pufferl.prio_bufs.idx.data, s);
        mtl_barrier((MetalStream*)s);

        if (gpu_profile) mtl_ensure_stream_synced(s);
        uint64_t tp9 = mach_absolute_time();

        pufferl.profile.accum[PROF_TRAIN_PRIO] += prof_ms(tp0, tp2);
        pufferl.profile.accum[PROF_TRAIN_SELECT] += prof_ms(tp2, tp3);
        pufferl.profile.accum[PROF_TRAIN_FWD] += prof_ms(tp3, tp4);
        pufferl.profile.accum[PROF_TRAIN_PPO] += prof_ms(tp4, tp5);
        pufferl.profile.accum[PROF_TRAIN_BACKWARD] += prof_ms(tp5, tp6);
        pufferl.profile.accum[PROF_TRAIN_GRAD_COPY] += prof_ms(tp6, tp7);
        pufferl.profile.accum[PROF_TRAIN_GRAD_CLIP] += prof_ms(tp7, tp8);
        pufferl.profile.accum[PROF_TRAIN_MUON] += prof_ms(tp8, tp9);
    };

    uint32_t* train_rng_offset = (uint32_t*)(pufferl.rng_offset_puf.data + hypers.num_buffers);

    puf_set_gpu_training(false);

    if (pufferl.overlap_enabled) {
        // Overlap: dispatch all minibatches on train_stream (separate Metal command queue).
        // GPU executes async during next rollout. 1-iteration policy lag — V-trace compensates.
        cudaStream_t ts = train_stream;

        if (pufferl.train_pending) {
            sync_pending_train(pufferl);
        }

        // Copy trained weights to inference buffer so the NEXT rollout sees them.
        {
            int64_t total_elems = pufferl.alloc_fp32.params.total_elems;
            PufTensor fp32_all = {.bytes = (char*)pufferl.alloc_fp32.params.mem,
                                  .shape = {total_elems}, .dtype_size = sizeof(float)};
            PufTensor infer_all = {.bytes = (char*)pufferl.infer_params_alloc.mem,
                                   .shape = {total_elems}, .dtype_size = sizeof(float)};
            puf_copy(infer_all, fp32_all, ts);
            mtl_barrier((MetalStream*)ts);
        }

        puf_set_gpu_training(true);
        MetalStream* mts = (MetalStream*)ts;
        for (int mb = 0; mb < total_minibatches; ++mb) {
            run_minibatch(ts, train_rng_offset, false);
            // Commit current command buffer when ring is >75% full to prevent
            // overflow on high replay_ratio configs. Metal queue serial execution
            // guarantees the GPU finishes reading ring data before we overwrite it.
            if (mb + 1 < total_minibatches &&
                mts->const_ring_offset > MTL_CONST_RING_SIZE * 3 / 4) {
                mts->commit_chunk();
            }
        }
        puf_set_gpu_training(false);

        ((MetalStream*)train_stream)->flush();
        pufferl.train_pending = true;
        pufferl.epoch += 1;
        return;
    }

    // Non-overlap: run minibatch loop synchronously.
    bool gpu_profile = hypers.profile;
    puf_set_gpu_training(true);
    MetalStream* mts_sync = (MetalStream*)train_stream;
    for (int mb = 0; mb < total_minibatches; ++mb) {
        run_minibatch(train_stream, train_rng_offset, gpu_profile);
        if (mb + 1 < total_minibatches &&
            mts_sync->const_ring_offset > MTL_CONST_RING_SIZE * 3 / 4) {
            mts_sync->commit_chunk();
        }
    }

    pufferl.epoch += 1;

    uint64_t tp_sync0 = mach_absolute_time();
    puf_set_gpu_training(false);
    mtl_ensure_stream_synced(train_stream);
    uint64_t tp_sync1 = mach_absolute_time();
    pufferl.profile.accum[PROF_TRAIN_SYNC] += prof_ms(tp_sync0, tp_sync1);

}

// Wait for async GPU training to complete, then snapshot weights for inference.
// Called at the end of rollouts() before the next iteration needs updated weights.
static void sync_pending_train(PuffeRL& pufferl) {
    if (!pufferl.train_pending) return;
    // Wait for async training on train_stream (separate queue) to complete.
    // After this, weights_infer and fused weight are up-to-date.
    MetalStream* ts = (MetalStream*)mtl_train_stream();
    ts->wait_completed();
    pufferl.train_pending = false;
}

// ============================================================================
// Initialization
// ============================================================================

std::unique_ptr<PuffeRL> create_pufferl_impl(HypersT& hypers,
        const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
    auto pufferl = std::make_unique<PuffeRL>();
    pufferl->hypers = hypers;

    mtl_init();

    pufferl->rng_seed = hypers.seed;

    StaticVec* vec = create_environments(hypers.num_buffers, hypers.total_agents,
        env_name, vec_kwargs, env_kwargs, pufferl->env);
    pufferl->vec = vec;

    int num_action_heads = pufferl->env.actions.shape[1];
    int* raw_act_sizes = get_act_sizes();
    int act_n = 0;
    for (int i = 0; i < num_action_heads; i++) {
        act_n += raw_act_sizes[i];
    }

    // CPU-based profiling (no CUDA events)
    memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));

    // Determine action space type
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        if (raw_act_sizes[i] == 1) {
            num_continuous++;
        } else {
            num_discrete++;
        }
    }
    if (num_continuous > 0 && num_discrete > 0) {
        assert(false && "Mixed continuous/discrete action spaces not supported");
    }
    pufferl->is_continuous = (num_continuous > 0);
    assert(!(hypers.train_fp16 && pufferl->is_continuous) &&
        "train_fp16 currently supports discrete action spaces only");

    int env_obs_width = pufferl->env.obs.shape[1];
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;

    // Action mask: env_config "mask_in_obs" > 0 means mask is embedded in obs.
    {
        DictItem* mask_entry = dict_get_unsafe(env_kwargs, "mask_in_obs");
        pufferl->has_mask = (mask_entry && mask_entry->value > 0.0f);
    }
    pufferl->env_obs_width = env_obs_width;
    pufferl->mask_width = act_n;  // total mask width = sum of action head sizes
    // when mask is embedded, split it: encoder sees only the feature prefix
    int input_size = pufferl->has_mask ? (env_obs_width - act_n) : env_obs_width;

    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;

    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int inf_batch = vec->total_agents / hypers.num_buffers;

    // ========================================================================
    // fp32 master weights (for optimizer)
    // ========================================================================

    int esz_fp32 = sizeof(float);
    pufferl->alloc_fp32.esz = esz_fp32;
    Allocator& fp32_params = pufferl->alloc_fp32.params;

    Encoder encoder = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
    };
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
    };
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
    };

    Policy* policy = new Policy{
        .encoder = encoder, .decoder = decoder, .network = network,
        .input_dim = input_size, .hidden_dim = hidden_size, .output_dim = decoder_output_size,
        .num_atns = act_n,
    };
    pufferl->policy = policy;

    // fp32 master weights
    auto new_weights = [&]() -> PolicyWeights {
        PolicyWeights w;
        w.encoder = new EncoderWeights{.in_dim = input_size, .out_dim = hidden_size};
        w.decoder = new DecoderWeights{.hidden_dim = hidden_size, .output_dim = decoder_output_size, .continuous = is_continuous};
        w.network = new MinGRUWeights{.hidden = hidden_size, .num_layers = num_layers, .horizon = hypers.horizon};
        ((MinGRUWeights*)w.network)->weights.resize(num_layers);
        return w;
    };

    pufferl->weights_fp32 = new_weights();
    PolicyWeights& wfp32 = pufferl->weights_fp32;
    encoder.reg_params(wfp32.encoder, &fp32_params, esz_fp32);
    decoder.reg_params(wfp32.decoder, &fp32_params, esz_fp32);
    network.reg_params(wfp32.network, &fp32_params, esz_fp32);

    pufferl->alloc_fp32.create();

    // Wrap fp32 params allocator for Metal GPU access
    mtl_wrap_allocator(&fp32_params);

    pufferl->param_fp32_puf = {.data = (float*)fp32_params.mem, .shape = {fp32_params.total_elems}};

    // Init weights on fp32 master
    {
        cudaStream_t default_stream = (cudaStream_t)mtl_stream();
        uint64_t init_seed = hypers.seed;
        encoder.init_weights(wfp32.encoder, &init_seed, default_stream);
        decoder.init_weights(wfp32.decoder, &init_seed, default_stream);
        network.init_weights(wfp32.network, &init_seed, default_stream);
        mtl_ensure_stream_synced(default_stream);
    }

    // Fused encoder+layer0 is disabled.
    // The null guard in mingru_forward falls back to encoder.forward().

    // ========================================================================
    // Double-buffered inference weights (for rollout/training overlap)
    // Same shapes as weights_fp32, separate allocator for isolation.
    // ========================================================================

    {
        Allocator& infer_alloc = pufferl->infer_params_alloc;
        pufferl->weights_infer = new_weights();
        PolicyWeights& wi = pufferl->weights_infer;
        encoder.reg_params(wi.encoder, &infer_alloc, esz_fp32);
        decoder.reg_params(wi.decoder, &infer_alloc, esz_fp32);
        network.reg_params(wi.network, &infer_alloc, esz_fp32);
        infer_alloc.create();
        mtl_wrap_allocator(&infer_alloc);
        // Initial copy: weights_fp32 → weights_infer
        copy_weights_to_infer(*pufferl);
        // Fused encoder+layer0 disabled (nonlinear encoder)
    }
    pufferl->overlap_enabled = hypers.overlap;
    pufferl->cpu_inference = hypers.cpu_inference;
    pufferl->train_fp16 = hypers.train_fp16;

    // fp16 training weights, activations, and gradients.

    int B_TT = minibatch_segments * hypers.horizon;
    int esz_fp16 = 2;

    // fp16 training weights (separate allocation from fp32 master)
    pufferl->alloc_fp16.esz = esz_fp16;
    Allocator& fp16_params = pufferl->alloc_fp16.params;
    Allocator& acts = pufferl->alloc_fp16.acts;
    Allocator& grads = pufferl->alloc_fp16.grads;

    pufferl->weights_fp16 = new_weights();
    PolicyWeights& wfp16 = pufferl->weights_fp16;

    encoder.reg_params(wfp16.encoder, &fp16_params, esz_fp16);
    decoder.reg_params(wfp16.decoder, &fp16_params, esz_fp16);
    network.reg_params(wfp16.network, &fp16_params, esz_fp16);

    // Register train activations/grads.
    // train_fp16: activations/grads use fp16 (esz_fp16=2), enabling fp16 GEMM paths.
    // Otherwise: activations/grads stay fp32 (PRECISION_SIZE=4) even in fp16 allocator.
    int train_precision = pufferl->train_fp16 ? esz_fp16 : PRECISION_SIZE;
    PolicyActivations& tb = pufferl->train_activations;
    tb.encoder = new EncoderActivations{};
    tb.decoder = new DecoderActivations{};
    tb.network = new MinGRUActivations{};
    encoder.reg_train(wfp16.encoder, tb.encoder, &acts, &grads, B_TT, train_precision);
    decoder.reg_train(wfp16.decoder, tb.decoder, &acts, &grads, B_TT, train_precision);
    network.reg_train(wfp16.network, tb.network, &acts, &grads, B_TT, train_precision);

    pufferl->alloc_fp16.create();

    // Wrap fp16 allocators for Metal GPU access
    mtl_wrap_allocator(&fp16_params);
    mtl_wrap_allocator(&acts);
    mtl_wrap_allocator(&grads);

    pufferl->param_fp16_puf = {.bytes = (char*)fp16_params.mem, .shape = {fp16_params.total_elems}, .dtype_size = esz_fp16};
    // When train_fp16: grads are fp16, need cast to fp32 before muon.
    // When fp32: grads are fp32, just copy (no cast needed).
    int grad_dtype = pufferl->train_fp16 ? esz_fp16 : esz_fp32;
    pufferl->grad_fp16_puf = {.bytes = (char*)grads.mem, .shape = {grads.total_elems}, .dtype_size = grad_dtype};

    // Cast fp32 master weights → fp16 training weights
    {
        cudaStream_t s = (cudaStream_t)mtl_stream();
        mtl_cast_f32_to_f16(pufferl->param_fp16_puf.bytes,
                            (const float*)fp32_params.mem,
                            (int)fp32_params.total_elems, s);
        mtl_ensure_stream_synced(s);
    }

    // Boundary buffers: fp16 obs (encoder input), fp32 dec_out (PPO input),
    // fp16 state (scan initial state — mb_state is fp32 but scan reads half*)
    {
        int dec_fused = decoder_output_size + 1;
        pufferl->fp16_obs_buf = {.shape = {minibatch_segments, hypers.horizon, input_size}, .dtype_size = esz_fp16};
        pufferl->fp32_dec_out_buf = {.shape = {minibatch_segments, hypers.horizon, dec_fused}, .dtype_size = esz_fp32};
        pufferl->fp16_state_buf = {.shape = {num_layers, minibatch_segments, 1, hidden_size}, .dtype_size = esz_fp16};
        alloc_register(&pufferl->fp16_boundary_alloc, &pufferl->fp16_obs_buf);
        alloc_register(&pufferl->fp16_boundary_alloc, &pufferl->fp32_dec_out_buf);
        alloc_register(&pufferl->fp16_boundary_alloc, &pufferl->fp16_state_buf);
        pufferl->fp16_boundary_alloc.create();
        mtl_wrap_allocator(&pufferl->fp16_boundary_alloc);
    }

    // ========================================================================
    // Optimizer (Muon) — operates on fp32 master weights
    // ========================================================================

    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    pufferl->muon = new Muon{};
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    // ========================================================================
    // Register all init-to-close buffers into pufferl_alloc, then create once
    // ========================================================================
    Allocator& alloc = pufferl->pufferl_alloc;

    int p = PRECISION_SIZE;
    pufferl->rng_offset_puf = {.shape = {num_buffers + 1}};
    pufferl->act_sizes_puf = {.shape = {num_action_heads}};
    pufferl->losses_puf = {.shape = {NUM_LOSSES}};
    pufferl->grad_norm_puf = {.shape = {1}};
    alloc_register(&alloc, &pufferl->rng_offset_puf);
    alloc_register(&alloc, &pufferl->act_sizes_puf);
    alloc_register(&alloc, &pufferl->losses_puf);
    alloc_register(&alloc, &pufferl->grad_norm_puf);

    // Per-buffer RNN states
    pufferl->buffer_states.resize(num_buffers);
    pufferl->sample_act_f32_buffers.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_states[i] = {.shape = {num_layers, batch, hidden_size}, .dtype_size = p};
        alloc_register(&alloc, &pufferl->buffer_states[i]);
        pufferl->sample_act_f32_buffers[i] = {.shape = {batch, num_action_heads}};
        alloc_register(&alloc, &pufferl->sample_act_f32_buffers[i]);
    }

    // Rollout buffers (horizon, total_agents, ...)
    register_rollout_buffers(pufferl->rollouts, alloc, horizon, total_agents, input_size, num_action_heads);

    // Train graph buffers
    register_train_buffers(pufferl->train_buf, alloc, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers);

    // Pre-allocated transposed rollouts for train_impl (total_agents, horizon, ...)
    register_rollout_buffers(pufferl->train_rollouts, alloc, total_agents, horizon, input_size, num_action_heads);

    // Pre-allocated train temporaries
    pufferl->old_values_puf = {.shape = {total_agents, horizon}};
    pufferl->advantages_puf = {.shape = {total_agents, horizon}};
    alloc_register(&alloc, &pufferl->old_values_puf);
    alloc_register(&alloc, &pufferl->advantages_puf);

    // PPO loss buffers
    register_ppo_buffers(pufferl->ppo_bufs_puf, alloc, minibatch_segments, hypers.horizon, decoder_output_size, is_continuous);

    // Priority replay buffers
    register_prio_buffers(pufferl->prio_bufs, alloc, hypers.total_agents, minibatch_segments);

    // Mask buffers: when has_mask, masks are split from obs at rollout time.
    if (pufferl->has_mask) {
        pufferl->rollout_masks = {.shape = {horizon, total_agents, act_n}};
        pufferl->train_masks = {.shape = {total_agents, horizon, act_n}};
        pufferl->mb_masks = {.shape = {minibatch_segments, hypers.horizon, act_n}};
        alloc_register(&alloc, &pufferl->rollout_masks);
        alloc_register(&alloc, &pufferl->train_masks);
        alloc_register(&alloc, &pufferl->mb_masks);
    } else {
        pufferl->ones_mask = {.shape = {act_n}};
        alloc_register(&alloc, &pufferl->ones_mask);
    }

    // Decoder logits + f32 actions for logprob recompute at training start.
    if (pufferl->cpu_inference || hypers.train_fp16) {
        int fused = decoder_output_size + 1;
        int na = num_action_heads;
        pufferl->rollout_logits = {.shape = {horizon, total_agents, fused}};
        pufferl->train_logits = {.shape = {total_agents, horizon, fused}};
        pufferl->rollout_actions_f32 = {.shape = {horizon, total_agents, na}};
        pufferl->train_actions_f32 = {.shape = {total_agents, horizon, na}};
        alloc_register(&alloc, &pufferl->rollout_logits);
        alloc_register(&alloc, &pufferl->train_logits);
        alloc_register(&alloc, &pufferl->rollout_actions_f32);
        alloc_register(&alloc, &pufferl->train_actions_f32);
    }

    // Optimizer init (register buffers with shared allocator)
    muon_init(pufferl->muon, &fp32_params,
        pufferl->param_fp32_puf, lr, beta1, alloc);
    // Single allocation for all registered buffers
    alloc.create();

    // Wrap pufferl_alloc for Metal GPU access
    mtl_wrap_allocator(&alloc);

    // Post-create initialization: unified memory, write directly
    memset(pufferl->rng_offset_puf.data, 0, (num_buffers + 1) * sizeof(long));
    memcpy(pufferl->act_sizes_puf.data, raw_act_sizes, num_action_heads * sizeof(int32_t));
    memset(pufferl->losses_puf.data, 0, NUM_LOSSES * sizeof(float));

    // Fill all-ones mask (after alloc.create + mtl_wrap)
    if (!pufferl->has_mask) {
        float* ones = pufferl->ones_mask.data;
        for (int i = 0; i < act_n; i++) ones[i] = 1.0f;
    }


    // muon_post_create: write lr and zero momentum (unified memory)
    pufferl->muon->lr_ptr = pufferl->muon->lr_puf.data;
    pufferl->muon->lr_derived_ptr = pufferl->muon->lr_derived_puf.data;
    if (pufferl->muon->ns_norm_puf.data)
        pufferl->muon->ns.norm_ptr = pufferl->muon->ns_norm_puf.data;
    *pufferl->muon->lr_ptr = pufferl->muon->lr_val_init;
    memset(pufferl->muon->lr_derived_ptr, 0, 2 * sizeof(float));
    memset(pufferl->muon->mb_puf.data, 0, puf_numel(pufferl->muon->mb_puf.shape) * sizeof(float));

    // Per-buffer inference activations (separate allocators)
    pufferl->buffer_activations.resize(num_buffers);
    pufferl->buffer_allocs.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        PolicyActivations& rbuf = pufferl->buffer_activations[i];
        Allocator& ralloc = pufferl->buffer_allocs[i];
        rbuf.encoder = new EncoderActivations{};
        rbuf.decoder = new DecoderActivations{};
        rbuf.network = new MinGRUActivations{};
        // Rollout uses fp32 — register with fp32 weights (dimensions only, dtype from PRECISION_SIZE)
        encoder.reg_rollout(pufferl->weights_fp32.encoder, rbuf.encoder, &ralloc, inf_batch);
        decoder.reg_rollout(pufferl->weights_fp32.decoder, rbuf.decoder, &ralloc, inf_batch);
        network.reg_rollout(pufferl->weights_fp32.network, rbuf.network, &ralloc, inf_batch);
        ralloc.create();
        // Wrap each per-buffer allocator for Metal GPU access
        mtl_wrap_allocator(&ralloc);
    }

    // No CUDA graph warmup on Metal (cudagraphs always -1)

    // Create per-buffer Metal streams for rollout callback workers.
    pufferl->rollout_streams.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        cudaStream_t s = (cudaStream_t)mtl_create_stream();
        pufferl->rollout_streams[i] = s;
        vec->streams[i] = s;
    }

    // Create threads for vecenv — thread_init binds per-thread rollout stream
    create_static_threads(vec, hypers.num_threads, horizon, pufferl.get(),
        net_callback_wrapper, thread_init_metal);
    static_vec_reset(vec);

    pufferl->epoch = 0;
    pufferl->global_step = 0;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double now = tv.tv_sec + tv.tv_usec * 1e-6;
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    return pufferl;
}

// ============================================================================
// Cleanup
// ============================================================================

void close_impl(PuffeRL& pufferl) {
    sync_pending_train(pufferl);
    mtl_ensure_stream_synced((cudaStream_t)mtl_stream());

    delete pufferl.muon;

    auto delete_weights = [](PolicyWeights& w) {
        delete (EncoderWeights*)w.encoder;
        delete (DecoderWeights*)w.decoder;
        delete (MinGRUWeights*)w.network;
    };
    delete (EncoderActivations*)pufferl.train_activations.encoder;
    delete (DecoderActivations*)pufferl.train_activations.decoder;
    delete (MinGRUActivations*)pufferl.train_activations.network;
    delete_weights(pufferl.weights_fp32);
    delete_weights(pufferl.weights_fp16);
    delete_weights(pufferl.weights_infer);
    for (auto& rbuf : pufferl.buffer_activations) {
        delete (EncoderActivations*)rbuf.encoder;
        delete (DecoderActivations*)rbuf.decoder;
        delete (MinGRUActivations*)rbuf.network;
    }
    delete pufferl.policy;

    static_vec_close(pufferl.vec);

    for (cudaStream_t s : pufferl.rollout_streams) {
        mtl_destroy_stream((void*)s);
    }
    pufferl.rollout_streams.clear();

    // Release MTLBuffers BEFORE freeing the underlying memory they reference.
    // MTLBuffers created with newBufferWithBytesNoCopy need their backing pages
    // still mapped when ARC releases them (Metal unmaps the GPU address space).
    mtl_destroy();

    pufferl.alloc_fp32.destroy();
    pufferl.alloc_fp16.destroy();
    pufferl.fp16_boundary_alloc.destroy();
    pufferl.infer_params_alloc.destroy();
    pufferl.pufferl_alloc.destroy();
    for (auto& a : pufferl.buffer_allocs) {
        a.destroy();
    }
}
