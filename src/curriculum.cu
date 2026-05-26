// Curriculum state-set and prioritized replay implementation.
// Included from pufferlib.cu in two phases so the buffer types are
// visible inside PuffeRL while functions that dereference PuffeRL see
// the complete struct definition.

#include <cub/device/device_scan.cuh>

#ifdef PUFFER_CURRICULUM_TYPES

// Prioritized replay over single-epoch data. These kernels are
// the least cleaned because we will likely have a better method in 5.0
struct PrioBuffers {
    FloatTensor prio_weights, cdf;
    ByteTensor cdf_temp;
    PrecisionTensor mb_prio;
    IntTensor idx, sample_done;
};

void register_prio_buffers(PrioBuffers& bufs, Allocator* alloc, int B, int minibatch_segments) {
    size_t cdf_temp_bytes = 0;
    cudaError_t err = cub::DeviceScan::InclusiveSum(
        NULL, cdf_temp_bytes, (float*)NULL, (float*)NULL, B);
    assert(err == cudaSuccess);

    bufs = (PrioBuffers){
        .prio_weights = {.shape = {B}},
        .cdf = {.shape = {B}},
        .cdf_temp = {.shape = {(int64_t)cdf_temp_bytes}},
        .mb_prio = {.shape = {minibatch_segments}},
        .idx = {.shape = {minibatch_segments}},
        .sample_done = {.shape = {1}},
    };
    alloc_register(alloc, &bufs.prio_weights);
    alloc_register(alloc, &bufs.cdf);
    alloc_register(alloc, &bufs.cdf_temp);
    alloc_register(alloc, &bufs.idx);
    alloc_register(alloc, &bufs.sample_done);
    alloc_register(alloc, &bufs.mb_prio);
}

struct StateBuffer {
    PufferState* states;       // CPU state_buffer_size entries
    PufferState* candidate_states; // CPU scratch, length candidate_capacity
    float* priorities;         // CPU priority per persistent slot
    precision_t* priorities_host; // CPU scratch for copying priorities to GPU
    precision_t* env_scores_host; // CPU scratch, length candidate_capacity
    int* heap;                 // CPU min-heap of persistent slot ids
    int* heap_pos;             // CPU inverse heap position per slot
    int capacity;
    int size;
    int num_envs;
    int num_checkpoints;
    int checkpoint_interval;
    int candidate_capacity;
    int agents_per_env;
    int num_cl_envs;
    int num_fresh_envs;
    int* env_state_inds_host;  // CPU scratch, length num_envs
    PrecisionTensor advantages; // GPU, shape {state_buffer_size}
    PrecisionTensor env_scores; // GPU scratch, shape {candidate_capacity}
    PrecisionTensor importance; // GPU, shape {total_agents}; fresh=1, CL=PER IS weight
    PrioBuffers prio_bufs;      // GPU CDF/weights/idx/importance for curriculum
};

void register_state_buffer(StateBuffer* buf, Allocator* alloc,
        int capacity, int total_agents, int num_envs, int agents_per_env,
        int num_cl_envs, int horizon, int checkpoint_interval) {
    buf->capacity = capacity;
    buf->size = 0;
    buf->num_envs = num_envs;
    buf->checkpoint_interval = checkpoint_interval;
    buf->num_checkpoints = (horizon + checkpoint_interval - 1) / checkpoint_interval;
    buf->candidate_capacity = num_envs * buf->num_checkpoints;
    buf->agents_per_env = agents_per_env;
    buf->advantages = {.shape = {capacity}};
    buf->env_scores = {.shape = {buf->candidate_capacity}};
    buf->importance = {.shape = {total_agents}};
    alloc_register(alloc, &buf->advantages);
    alloc_register(alloc, &buf->env_scores);
    alloc_register(alloc, &buf->importance);
    if (num_cl_envs > 0) {
        register_prio_buffers(buf->prio_bufs, alloc, capacity, num_cl_envs);
    }
}

int init_state_buffer(StateBuffer* buf, int total_agents) {
    size_t capacity = (size_t)buf->capacity;
    size_t state_size = sizeof(PufferState);
    size_t state_bytes = capacity * state_size;
    size_t candidate_bytes = (size_t)buf->candidate_capacity * state_size;
    buf->states = (PufferState*)malloc(state_bytes);
    buf->candidate_states = (PufferState*)malloc(candidate_bytes);
    buf->priorities = (float*)malloc(capacity * sizeof(float));
    buf->priorities_host = (precision_t*)malloc(capacity * sizeof(precision_t));
    buf->env_scores_host = (precision_t*)malloc(
        (size_t)buf->candidate_capacity * sizeof(precision_t));
    buf->heap = (int*)malloc(capacity * sizeof(int));
    buf->heap_pos = (int*)malloc(capacity * sizeof(int));
    buf->env_state_inds_host = (int*)malloc((size_t)buf->num_envs * sizeof(int));
    if (buf->states == NULL || buf->candidate_states == NULL
            || buf->priorities == NULL || buf->priorities_host == NULL
            || buf->env_scores_host == NULL || buf->heap == NULL || buf->heap_pos == NULL
            || buf->env_state_inds_host == NULL) {
        fprintf(stderr,
            "Failed to allocate curriculum state buffer: capacity=%d state_size=%d bytes=%zu\n",
            buf->capacity, (int)state_size, state_bytes);
        return 0;
    }
    return 1;
}

void close_state_buffer(StateBuffer* buf) {
    free(buf->states);
    free(buf->candidate_states);
    free(buf->priorities);
    free(buf->priorities_host);
    free(buf->env_scores_host);
    free(buf->heap);
    free(buf->heap_pos);
    free(buf->env_state_inds_host);
}

#endif

#ifdef PUFFER_CURRICULUM_IMPL

#define PRIO_WARP_SIZE 32
#ifdef USE_ROCM
using PufWarpMask = unsigned long long;
#else
using PufWarpMask = unsigned int;
#endif
#define PRIO_BLOCK_SIZE 256
#define PRIO_NUM_WARPS (PRIO_BLOCK_SIZE / PRIO_WARP_SIZE)

__device__ __forceinline__ float prio_warp_sum(float val, int width = PRIO_WARP_SIZE) {
    PufWarpMask mask = (PufWarpMask)__activemask();
    for (int s = width / 2; s >= 1; s /= 2) {
        val += __shfl_down_sync(mask, val, s, width);
    }
    return val;
}

__device__ __forceinline__ float priority_power(float value, float alpha) {
    if (alpha == 0.0f) {
        return 1.0f;
    }
    if (value <= 0.0f) {
        return 0.0f;
    }
    value = __powf(value, alpha);
    if (isnan(value) || isinf(value)) {
        return 0.0f;
    }
    return value;
}

__global__ void compute_prio_abs(
        const precision_t* __restrict__ advantages,
        float* prio_weights, float prio_alpha, float eps, int rows, int stride) {
    if (stride == 1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows) {
            prio_weights[idx] = priority_power(fabsf(to_float(advantages[idx])), prio_alpha) + eps;
        }
        return;
    }

    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;

    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(to_float(advantages[offset + t]));
    }

    local_sum = prio_warp_sum(local_sum);
    if (tx == 0) {
        prio_weights[row] = priority_power(local_sum, prio_alpha) + eps;
    }
}

__global__ void compute_curriculum_checkpoint_scores(
        precision_t* __restrict__ dst,
        const precision_t* __restrict__ advantages_bt,
        int num_fresh_envs, int num_cl_envs,
        int num_checkpoints, int checkpoint_interval,
        int agents_per_env, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int fresh_rows = num_checkpoints * num_fresh_envs;
    int total_rows = fresh_rows + num_cl_envs;
    if (row >= total_rows) {
        return;
    }

    int env_idx;
    int start_t = 0;
    int end_t = horizon;
    if (row < fresh_rows) {
        int c = row / num_fresh_envs;
        env_idx = row - c * num_fresh_envs;
        start_t = c * checkpoint_interval;
        end_t = start_t + checkpoint_interval;
        if (end_t > horizon) {
            end_t = horizon;
        }
    } else {
        int i = row - fresh_rows;
        env_idx = num_fresh_envs + i;
    }

    int agent_start = env_idx * agents_per_env;
    float sum_agent_abs = 0.0f;
    for (int a = 0; a < agents_per_env; a++) {
        int offset = (agent_start + a) * horizon;
        float agent_abs = 0.0f;
        for (int t = start_t; t < end_t; t++) {
            agent_abs += fabsf(to_float(advantages_bt[offset + t]));
        }
        sum_agent_abs += agent_abs;
    }
    dst[row] = from_float(sum_agent_abs / (float)agents_per_env);
}

// Multinomial with replacement (uses cuRAND)
__global__ void multinomial_sample_advance(int* __restrict__ out_idx,
        precision_t* __restrict__ out_importance,
        precision_t* __restrict__ row_importance,
        const float* __restrict__ prio_weights, const float* __restrict__ cdf,
        int B, int num_samples, uint64_t seed, float beta,
        int row_importance_offset, int agents_per_sample,
        int64_t* __restrict__ offset_ptr, int* __restrict__ done_counter,
        int launch_blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base_off = *offset_ptr;
    float total_weight = cdf[B - 1];
    if (tid < num_samples) {
        curandStatePhilox4_32_10_t rng_state;
        curand_init(seed, (uint64_t)base_off + tid, 0, &rng_state);
        float u = curand_uniform(&rng_state);

        int lo;
        int use_uniform = total_weight <= 0.0f || isnan(total_weight) || isinf(total_weight);
        if (use_uniform) {
            lo = (int)(u * (float)B);
            if (lo >= B) {
                lo = B - 1;
            }
        } else {
            float target = u * total_weight;
            lo = 0;
            int hi = B - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (cdf[mid] < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
        }
        if (out_importance != NULL || row_importance != NULL) {
            float weight = 1.0f;
            if (!use_uniform) {
                float value = prio_weights[lo] * (float)B / total_weight;
                weight = __powf(value, -beta);
                if (isnan(weight) || isinf(weight)) {
                    weight = 1.0f;
                }
            }
            precision_t value = from_float(weight);
            if (out_importance != NULL) {
                out_importance[tid] = value;
            }
            if (row_importance != NULL) {
                int base = row_importance_offset + tid * agents_per_sample;
                for (int a = 0; a < agents_per_sample; a++) {
                    row_importance[base + a] = value;
                }
            }
        }
        out_idx[tid] = lo;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        __threadfence();
        int ticket = atomicAdd(done_counter, 1);
        if (ticket == launch_blocks - 1) {
            *offset_ptr = base_off + num_samples;
            *done_counter = 0;
        }
    }
}

static inline void sample_prio_indices(PrioBuffers* bufs, int population,
        int samples, ulong seed, long* offset_ptr, precision_t* out_importance,
        precision_t* row_importance, int row_importance_offset,
        int agents_per_sample, float beta, cudaStream_t stream) {
    size_t cdf_temp_bytes = (size_t)bufs->cdf_temp.shape[0];
    cub::DeviceScan::InclusiveSum(bufs->cdf_temp.data, cdf_temp_bytes,
        bufs->prio_weights.data, bufs->cdf.data, population, stream);
    int blocks = (samples + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    multinomial_sample_advance<<<blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        bufs->idx.data, out_importance, row_importance,
        bufs->prio_weights.data, bufs->cdf.data, population, samples, seed, beta,
        row_importance_offset, agents_per_sample,
        offset_ptr, bufs->sample_done.data, blocks);
}

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static int fixed_agents_per_env(StaticVec* vec) {
    assert(vec->size > 0 && "state curriculum requires at least one env");
    int agents_per_env = vec->envs[0].num_agents;
    assert(agents_per_env > 0 && "env num_agents must be positive");
    for (int i = 0; i < vec->size; i++) {
        assert(vec->envs[i].num_agents == agents_per_env
            && "state curriculum currently requires fixed num_agents per env");
    }
    assert(vec->size * agents_per_env == vec->total_agents
        && "env agent counts must sum to total_agents");
    assert(vec->agents_per_buffer % agents_per_env == 0
        && "state curriculum requires agents_per_buffer to be divisible by num_agents");
    return agents_per_env;
}

static inline float clean_state_priority(float priority) {
    if (priority < 0.0f || isnan(priority) || isinf(priority)) {
        return 0.0f;
    }
    return priority;
}

static inline void state_heap_swap(StateBuffer* buf, int a, int b) {
    int slot_a = buf->heap[a];
    int slot_b = buf->heap[b];
    buf->heap[a] = slot_b;
    buf->heap[b] = slot_a;
    buf->heap_pos[slot_a] = b;
    buf->heap_pos[slot_b] = a;
}

static inline void state_heap_sift_up(StateBuffer* buf, int pos) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (buf->priorities[buf->heap[parent]] <= buf->priorities[buf->heap[pos]]) {
            break;
        }
        state_heap_swap(buf, parent, pos);
        pos = parent;
    }
}

static inline void state_heap_sift_down(StateBuffer* buf, int pos) {
    while (true) {
        int left = 2 * pos + 1;
        int right = left + 1;
        int best = pos;
        if (left < buf->size
                && buf->priorities[buf->heap[left]] < buf->priorities[buf->heap[best]]) {
            best = left;
        }
        if (right < buf->size
                && buf->priorities[buf->heap[right]] < buf->priorities[buf->heap[best]]) {
            best = right;
        }
        if (best == pos) {
            break;
        }
        state_heap_swap(buf, pos, best);
        pos = best;
    }
}

static inline void state_heap_update_slot(StateBuffer* buf, int slot, float priority,
        float decay) {
    priority = clean_state_priority(priority);
    float old_priority = buf->priorities[slot];
    if (priority < old_priority) {
        priority = decay * old_priority;
    }
    buf->priorities[slot] = priority;
    buf->priorities_host[slot] = from_float(priority);
    int pos = buf->heap_pos[slot];
    if (priority < old_priority) {
        state_heap_sift_up(buf, pos);
    } else {
        state_heap_sift_down(buf, pos);
    }
}

static inline void state_heap_insert(StateBuffer* buf, const PufferState* state, float priority) {
    priority = clean_state_priority(priority);
    if (buf->size < buf->capacity) {
        int slot = buf->size;
        buf->states[slot] = *state;
        buf->priorities[slot] = priority;
        buf->priorities_host[slot] = from_float(priority);
        buf->heap[slot] = slot;
        buf->heap_pos[slot] = slot;
        buf->size++;
        state_heap_sift_up(buf, slot);
        return;
    }

    int min_slot = buf->heap[0];
    if (priority <= buf->priorities[min_slot]) {
        return;
    }
    buf->states[min_slot] = *state;
    buf->priorities[min_slot] = priority;
    buf->priorities_host[min_slot] = from_float(priority);
    state_heap_sift_down(buf, 0);
}

static inline void capture_curriculum_checkpoint(PuffeRL* pufferl, int buffer_idx, int t) {
    StateBuffer* buf = &pufferl->state_buf;
    int interval = buf->checkpoint_interval;
    if ((t % interval) != 0) {
        return;
    }
    int checkpoint_idx = t / interval;

    StaticVec* vec = pufferl->vec;
    int env_start = vec->buffer_env_starts[buffer_idx];
    int env_end = env_start + vec->buffer_env_counts[buffer_idx];
    if (env_start >= buf->num_fresh_envs) {
        return;
    }
    if (env_end > buf->num_fresh_envs) {
        env_end = buf->num_fresh_envs;
    }

    Env* envs = vec->envs;
    PufferState* dst = buf->candidate_states + checkpoint_idx * buf->num_envs;
    for (int env_idx = env_start; env_idx < env_end; env_idx++) {
        dst[env_idx] = envs[env_idx].state;
    }
}

void curriculum_rollout_begin(PuffeRL* pufferl) {
    HypersT* h = &pufferl->hypers;
    StateBuffer* buf = &pufferl->state_buf;
    StaticVec* vec = pufferl->vec;
    cudaStream_t stream = pufferl->default_stream;
    int total_envs = vec->size;
    int agents_per_env = buf->agents_per_env;
    int total_epochs = h->total_timesteps / (h->total_agents * h->horizon);
    float progress = total_epochs > 0 ? (float)pufferl->epoch / (float)total_epochs : 1.0f;
    progress = fminf(1.0f, fmaxf(0.0f, progress));
    float current_cl_frac = h->cl_frac;
    if (h->anneal_cl) {
        current_cl_frac *= 1.0f - progress;
    }
    int num_cl_envs = (buf->size == 0 || buf->size < h->warmup_states) ? 0 :
        clamp_int((int)(current_cl_frac * (float)total_envs), 0, total_envs);
    int num_fresh_envs = total_envs - num_cl_envs;

    buf->num_cl_envs = num_cl_envs;
    buf->num_fresh_envs = num_fresh_envs;
    vec->log_env_limit = (num_cl_envs > 0) ? num_fresh_envs : 0;

    int fresh_agents = num_fresh_envs * agents_per_env;
    fill_precision_kernel<<<grid_size(fresh_agents), BLOCK_SIZE, 0, stream>>>(
        buf->importance.data, from_float(1.0f), fresh_agents);

    if (num_cl_envs > 0) {
        compute_prio_abs<<<grid_size(buf->size), BLOCK_SIZE, 0, stream>>>(
            buf->advantages.data, buf->prio_bufs.prio_weights.data,
            h->explore_alpha, 0.0f, buf->size, 1);
        long* rng_offset = pufferl->rng_offset_puf.data + h->num_buffers + 1;
        sample_prio_indices(&buf->prio_bufs, buf->size, num_cl_envs,
            pufferl->seed, rng_offset, NULL, buf->importance.data,
            fresh_agents, agents_per_env, h->explore_beta, stream);
        cudaMemcpyAsync(buf->env_state_inds_host + num_fresh_envs, buf->prio_bufs.idx.data,
            num_cl_envs * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        Env* envs = vec->envs;
        int* state_inds = buf->env_state_inds_host + num_fresh_envs;
        for (int i = 0; i < num_cl_envs; i++) {
            Env* env = &envs[num_fresh_envs + i];
            env->state = buf->states[state_inds[i]];
            puffer_state_refresh(env);
        }
        cudaMemcpy(vec->gpu_observations.data, vec->observations.data,
            (size_t)vec->total_agents * get_obs_size() * get_obs_elem_size(),
            cudaMemcpyHostToDevice);
        if (vec->action_mask_size > 0) {
            cudaMemcpy(vec->gpu_action_mask, vec->action_mask,
                (size_t)vec->total_agents * vec->action_mask_size * sizeof(unsigned char),
                cudaMemcpyHostToDevice);
        }
    }
}

void curriculum_update_advantages(PuffeRL* pufferl, PrecisionTensor* advantages,
        cudaStream_t stream) {
    StateBuffer* buf = &pufferl->state_buf;
    int horizon = advantages->shape[1];
    int num_fresh_envs = buf->num_fresh_envs;
    int num_cl_envs = buf->num_cl_envs;
    int agents_per_env = buf->agents_per_env;

    int score_rows = buf->num_checkpoints * num_fresh_envs + num_cl_envs;
    compute_curriculum_checkpoint_scores<<<grid_size(score_rows), BLOCK_SIZE, 0, stream>>>(
        buf->env_scores.data, advantages->data, num_fresh_envs,
        num_cl_envs, buf->num_checkpoints,
        buf->checkpoint_interval, agents_per_env, horizon);
    int cl_score_offset = buf->num_checkpoints * num_fresh_envs;
    cudaMemcpyAsync(buf->env_scores_host, buf->env_scores.data,
        (size_t)score_rows * sizeof(precision_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < num_cl_envs; i++) {
        int env_idx = num_fresh_envs + i;
        int slot = buf->env_state_inds_host[env_idx];
        float priority = to_float(buf->env_scores_host[cl_score_offset + i]);
        state_heap_update_slot(buf, slot, priority, pufferl->hypers.explore_decay);
    }

    for (int c = 0; c < buf->num_checkpoints; c++) {
        int candidate_offset = c * buf->num_envs;
        for (int i = 0; i < num_fresh_envs; i++) {
            int candidate_idx = candidate_offset + i;
            float priority = to_float(buf->env_scores_host[c * num_fresh_envs + i]);
            state_heap_insert(buf, &buf->candidate_states[candidate_idx], priority);
        }
    }

    cudaMemcpyAsync(buf->advantages.data, buf->priorities_host,
        (size_t)buf->size * sizeof(precision_t), cudaMemcpyHostToDevice, stream);
}

#endif
