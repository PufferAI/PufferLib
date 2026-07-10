#pragma once

// fbr: Breakout's host Env contains pointers to brick and Agent buffers. Rollouts
// instead keep one pointer-free state per environment on the GPU. A CUDA graph
// step writes observations/rewards/terminals directly into the next BF16
// rollout slice, avoiding both the host step and the float H2D conversion.
static constexpr int BC_MAX_BRICKS = OBS_SIZE - 10;
static constexpr int BC_BRICK_WORDS = (BC_MAX_BRICKS + 31) / 32;
static constexpr float BC_PI = 3.14159265358979323846f;

struct BreakoutCudaState {
    Log log;
    float paddle_x, paddle_y;
    float ball_x, ball_y, ball_vx, ball_vy;
    float initial_paddle_width, paddle_width, paddle_height, paddle_speed;
    float ball_speed, initial_ball_speed, max_ball_speed;
    // fbr: Pack 108 brick flags into four words to keep each environment state compact.
    unsigned int brick_bits[BC_BRICK_WORDS];
    int score, balls_fired, hits;
    int width, height, num_bricks, brick_rows, brick_cols;
    int ball_width, ball_height, brick_width, brick_height;
    int num_balls, max_score, half_max_score, tick, frameskip, continuous;
    unsigned int rng;
    unsigned char hit_brick;
};

struct BreakoutCudaCollision {
    float t, overlap, x, y, vx, vy;
    int brick_index;
};

static_assert(sizeof(BreakoutCudaState) % sizeof(unsigned int) == 0,
    "Breakout CUDA state must remain naturally word-aligned");

__host__ __device__ __forceinline__ bool bc_brick_destroyed(
        const BreakoutCudaState& state, int brick_idx) {
    return (state.brick_bits[brick_idx >> 5] >> (brick_idx & 31)) & 1u;
}

__host__ __device__ __forceinline__ void bc_destroy_brick_bit(
        BreakoutCudaState& state, int brick_idx) {
    state.brick_bits[brick_idx >> 5] |= 1u << (brick_idx & 31);
}

__host__ __device__ __forceinline__ void bc_clear_bricks(
        BreakoutCudaState& state) {
    for (int i = 0; i < BC_BRICK_WORDS; ++i) state.brick_bits[i] = 0;
}

template <typename T>
__device__ __forceinline__ void bc_store(T* dst, int idx, float value) {
    dst[idx] = value;
}

template <>
__device__ __forceinline__ void bc_store<precision_t>(
        precision_t* dst, int idx, float value) {
    dst[idx] = from_float(value);
}

// fbr: Match glibc rand_r so a CPU and CUDA environment initialized from the same
// state choose the same launch direction.
__device__ __forceinline__ int bc_rand_r(unsigned int* seed) {
    unsigned int next = *seed;
    int result;
    next = next * 1103515245u + 12345u;
    result = (int)((next / 65536u) % 2048u);
    next = next * 1103515245u + 12345u;
    result = (result << 10) ^ (int)((next / 65536u) % 1024u);
    next = next * 1103515245u + 12345u;
    result = (result << 10) ^ (int)((next / 65536u) % 1024u);
    *seed = next;
    return result;
}

template <typename T>
__device__ __forceinline__ void bc_observe(
        const BreakoutCudaState& state, int env_idx, T* observations) {
    int idx = env_idx * OBS_SIZE;
    bc_store(observations, idx++, state.paddle_x / state.width);
    bc_store(observations, idx++, state.paddle_y / state.height);
    bc_store(observations, idx++, state.ball_x / state.width);
    bc_store(observations, idx++, state.ball_y / state.height);
    bc_store(observations, idx++, state.ball_vx / 512.0f);
    bc_store(observations, idx++, state.ball_vy / 512.0f);
    bc_store(observations, idx++, state.balls_fired / 5.0f);
    bc_store(observations, idx++, state.score / 864.0f);
    bc_store(observations, idx++, state.num_balls / 5.0f);
    bc_store(observations, idx++, state.paddle_width / (2.0f * HALF_PADDLE_WIDTH));
    for (int i = 0; i < state.num_bricks; ++i) {
        bc_store(observations, idx++, bc_brick_destroyed(state, i) ? 1.0f : 0.0f);
    }
}

__device__ __forceinline__ bool bc_vline_collision(float xw, float yw, float hw,
        float x, float y, float vx, float vy, float h, BreakoutCudaCollision* col) {
    float t = (xw - x) / vx;
    float top = fminf(yw + hw, y + h + vy * t);
    float bottom = fmaxf(yw, y + vy * t);
    float overlap = top - bottom;
    if (overlap > 0.0f && t > 0.0f && t <= 1.0f
            && (t < col->t || (t == col->t && overlap > col->overlap))) {
        col->t = t;
        col->overlap = overlap;
        col->x = xw;
        col->y = y + vy * t;
        col->vx = -vx;
        col->vy = vy;
        return true;
    }
    return false;
}

__device__ __forceinline__ bool bc_hline_collision(float xw, float yw, float ww,
        float x, float y, float vx, float vy, float w, BreakoutCudaCollision* col) {
    float t = (yw - y) / vy;
    float right = fminf(xw + ww, x + w + vx * t);
    float left = fmaxf(xw, x + vx * t);
    float overlap = right - left;
    if (overlap > 0.0f && t > 0.0f && t <= 1.0f
            && (t < col->t || (t == col->t && overlap > col->overlap))) {
        col->t = t;
        col->overlap = overlap;
        col->x = x + vx * t;
        col->y = yw;
        col->vx = vx;
        col->vy = -vy;
        return true;
    }
    return false;
}

__device__ __forceinline__ void bc_brick_collision(
        BreakoutCudaState& state, int brick_idx, BreakoutCudaCollision* col) {
    int row = brick_idx / state.brick_cols;
    int column = brick_idx - row * state.brick_cols;
    float brick_x = column * state.brick_width;
    float brick_y = row * state.brick_height + Y_OFFSET;
    bool collision = false;
    if (state.ball_vx > 0.0f && bc_vline_collision(
            brick_x, brick_y, state.brick_height,
            state.ball_x + state.ball_width, state.ball_y,
            state.ball_vx, state.ball_vy, state.ball_height, col)) {
        collision = true;
        col->x -= state.ball_width;
    }
    if (state.ball_vx < 0.0f && bc_vline_collision(
            brick_x + state.brick_width, brick_y, state.brick_height,
            state.ball_x, state.ball_y, state.ball_vx, state.ball_vy,
            state.ball_height, col)) {
        collision = true;
    }
    if (state.ball_vy > 0.0f && bc_hline_collision(
            brick_x, brick_y, state.brick_width,
            state.ball_x, state.ball_y + state.ball_height,
            state.ball_vx, state.ball_vy, state.ball_width, col)) {
        collision = true;
        col->y -= state.ball_height;
    }
    if (state.ball_vy < 0.0f && bc_hline_collision(
            brick_x, brick_y + state.brick_height, state.brick_width,
            state.ball_x, state.ball_y, state.ball_vx, state.ball_vy,
            state.ball_width, col)) {
        collision = true;
    }
    if (collision) col->brick_index = brick_idx;
}

__device__ __forceinline__ int bc_column(const BreakoutCudaState& state, float x) {
    return (int)(x / state.brick_width);
}

__device__ __forceinline__ int bc_row(const BreakoutCudaState& state, float y) {
    return (int)((y - Y_OFFSET) / state.brick_height);
}

__device__ __forceinline__ void bc_all_brick_collisions(
        BreakoutCudaState& state, BreakoutCudaCollision* col) {
    float ball_x_dst = state.ball_x + state.ball_vx;
    float ball_y_dst = state.ball_y + state.ball_vy;
    int row_from = bc_row(state, fminf(state.ball_y, ball_y_dst));
    row_from = max(row_from, 0);
    if (row_from > state.brick_rows) return;
    int column_from = max(bc_column(state, fminf(state.ball_x, ball_x_dst)), 0);
    int column_to = bc_column(state,
        fmaxf(ball_x_dst + state.ball_width, state.ball_x + state.ball_width));
    column_to = min(column_to, state.brick_cols - 1);
    int row_to = bc_row(state,
        fmaxf(ball_y_dst + state.ball_height, state.ball_y + state.ball_height));
    row_to = min(row_to, state.brick_rows - 1);
    for (int row = row_from; row <= row_to; ++row) {
        for (int column = column_from; column <= column_to; ++column) {
            int idx = row * state.brick_cols + column;
            if (!bc_brick_destroyed(state, idx)) bc_brick_collision(state, idx, col);
        }
    }
}

__device__ __forceinline__ bool bc_paddle_collision(
        BreakoutCudaState& state, BreakoutCudaCollision* col) {
    if (state.ball_y + state.ball_height + state.ball_vy < state.paddle_y) return false;
    if (!bc_hline_collision(state.paddle_x, state.paddle_y, state.paddle_width,
            state.ball_x, state.ball_y + state.ball_height,
            state.ball_vx, state.ball_vy, state.ball_width, col)
            || col->t > 1.0f) return false;
    col->y -= state.ball_height;
    col->brick_index = BRICK_INDEX_PADDLE_COLLISION;
    state.hit_brick = false;
    float relative = ((state.ball_x + state.ball_width / 2.0f) - state.paddle_x)
        / state.paddle_width;
    float angle = -BC_PI / 4.0f + relative * BC_PI / 2.0f;
    state.ball_vx = sinf(angle) * state.ball_speed * TICK_RATE;
    state.ball_vy = -cosf(angle) * state.ball_speed * TICK_RATE;
    state.hits++;
    if (state.hits % 4 == 0 && state.ball_speed < state.max_ball_speed) {
        state.ball_speed += 64.0f;
    }
    if (state.score == state.half_max_score) {
        bc_clear_bricks(state);
    }
    return true;
}

__device__ __forceinline__ void bc_wall_collisions(
        BreakoutCudaState& state, BreakoutCudaCollision* col) {
    if (state.ball_vx < 0.0f && bc_vline_collision(0, 0, state.height,
            state.ball_x, state.ball_y, state.ball_vx, state.ball_vy,
            state.ball_height, col)) {
        col->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
    }
    if (state.ball_vx > 0.0f && bc_vline_collision(state.width, 0, state.height,
            state.ball_x + state.ball_width, state.ball_y,
            state.ball_vx, state.ball_vy, state.ball_height, col)) {
        col->x -= state.ball_width;
        col->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
    }
    if (state.ball_vy < 0.0f && bc_hline_collision(0, 0, state.width,
            state.ball_x, state.ball_y, state.ball_vx, state.ball_vy,
            state.ball_width, col)) {
        col->brick_index = BRICK_INDEX_BACKWALL_COLLISION;
    }
}

__device__ __forceinline__ void bc_destroy_brick(
        BreakoutCudaState& state, int brick_idx, float* reward) {
    int points = 7 - 3 * ((brick_idx / state.brick_cols) / 2);
    state.score += points;
    bc_destroy_brick_bit(state, brick_idx);
    *reward += points;
    if (brick_idx / state.brick_cols < 3) state.ball_speed = state.max_ball_speed;
}

__device__ __forceinline__ bool bc_handle_collisions(
        BreakoutCudaState& state, float* reward) {
    BreakoutCudaCollision col = {
        2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, BRICK_INDEX_NO_COLLISION
    };
    float offset = state.max_ball_speed * 1.1f * TICK_RATE;
    if (state.ball_x < 0.0f) state.ball_x += offset;
    if (state.ball_x > state.width) state.ball_x -= offset;
    if (state.ball_y < 0.0f) state.ball_y += offset;
    bc_all_brick_collisions(state, &col);
    bc_wall_collisions(state, &col);
    bc_paddle_collision(state, &col);
    if (col.brick_index != BRICK_INDEX_PADDLE_COLLISION && col.t <= 1.0f) {
        state.ball_x = col.x;
        state.ball_y = col.y;
        state.ball_vx = col.vx;
        state.ball_vy = col.vy;
        if (col.brick_index >= 0) bc_destroy_brick(state, col.brick_index, reward);
        if (col.brick_index == BRICK_INDEX_BACKWALL_COLLISION) {
            state.paddle_width = HALF_PADDLE_WIDTH;
        }
    }
    return col.brick_index != BRICK_INDEX_NO_COLLISION;
}

__device__ __forceinline__ void bc_reset_round(BreakoutCudaState& state) {
    state.balls_fired = 0;
    state.hit_brick = false;
    state.hits = 0;
    state.ball_speed = state.initial_ball_speed;
    state.paddle_width = state.initial_paddle_width;
    state.paddle_x = state.width / 2.0f - state.paddle_width / 2.0f;
    state.paddle_y = state.height - state.paddle_height - 10.0f;
    state.ball_x = state.paddle_x
        + (state.paddle_width / 2.0f - state.ball_width / 2.0f);
    state.ball_y = state.height / 2.0f - 30.0f;
    state.ball_vx = 0.0f;
    state.ball_vy = 0.0f;
}

__device__ __forceinline__ void bc_reset(BreakoutCudaState& state) {
    state.score = 0;
    state.num_balls = 5;
    bc_clear_bricks(state);
    bc_reset_round(state);
    state.tick = 0;
}

__device__ __forceinline__ void bc_add_log(BreakoutCudaState& state) {
    state.log.episode_length += state.tick;
    state.log.episode_return += state.score;
    state.log.score += state.score;
    state.log.perf += state.score / (float)state.max_score;
    state.log.n += 1.0f;
}

__device__ __forceinline__ void bc_step_frame(
        BreakoutCudaState& state, float action, float* reward, float* terminal) {
    float act = 0.0f;
    if (state.balls_fired == 0) {
        state.balls_fired = 1;
        float direction = BC_PI / 3.25f;
        state.ball_vy = cosf(direction) * state.ball_speed * TICK_RATE;
        state.ball_vx = sinf(direction) * state.ball_speed * TICK_RATE;
        if (bc_rand_r(&state.rng) % 2 == 0) state.ball_vx = -state.ball_vx;
    } else if (action == LEFT) {
        act = -1.0f;
    } else if (action == RIGHT) {
        act = 1.0f;
    }
    if (state.continuous) act = action;
    state.paddle_x += act * state.paddle_speed * TICK_RATE;
    state.paddle_x = fminf(fmaxf(state.paddle_x, 0.0f),
        state.width - state.paddle_width);
    if (!bc_handle_collisions(state, reward)) {
        state.ball_x += state.ball_vx;
        state.ball_y += state.ball_vy;
    }
    if (state.ball_y >= state.paddle_y + state.paddle_height) {
        state.num_balls--;
        bc_reset_round(state);
    }
    if (state.num_balls < 0 || state.score == state.max_score) {
        *terminal = 1.0f;
        bc_add_log(state);
        bc_reset(state);
    }
}

template <typename T>
__global__ void bc_init_output(
        const BreakoutCudaState* states, int count, T* obs, T* rewards, T* terminals) {
    int env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= count) return;
    bc_observe(states[env_idx], env_idx, obs);
    bc_store(rewards, env_idx, 0.0f);
    bc_store(terminals, env_idx, 0.0f);
}

template <typename T>
__global__ void bc_step(BreakoutCudaState* states, int count,
        const precision_t* actions,
        T* observations, T* rewards, T* terminals) {
    int env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool valid = env_idx < count;
    extern __shared__ unsigned int shared_words[];
    BreakoutCudaState* local_states = (BreakoutCudaState*)shared_words;
    BreakoutCudaState* local = &local_states[threadIdx.x];
    if (valid) *local = states[env_idx];
    __syncthreads();

    if (valid) {
        float reward = 0.0f;
        float terminal = 0.0f;
        float action = to_float(actions[env_idx * NUM_ATNS]);
        for (int frame = 0; frame < local->frameskip; ++frame) {
            local->tick++;
            bc_step_frame(*local, action, &reward, &terminal);
        }
        (void)observations;
        bc_store(rewards, env_idx, reward);
        bc_store(terminals, env_idx, terminal);
    }
    if (valid) states[env_idx] = *local;
}

// fbr: Physics is most efficient with one lane per independent environment. The
// observation has the opposite layout: one warp per environment makes its 118
// contiguous feature stores coalesced and supplies enough warps to fill every
// SM. A separate graph node avoids running scalar physics on only one lane.
static constexpr int BC_OBSERVE_WARPS_PER_BLOCK = 4;
static constexpr int BC_OBSERVE_BLOCK_SIZE = 32 * BC_OBSERVE_WARPS_PER_BLOCK;

template <typename T>
__global__ void bc_observe_warp(const BreakoutCudaState* states, int count,
        T* observations) {
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int env_idx = blockIdx.x * BC_OBSERVE_WARPS_PER_BLOCK + warp_in_block;
    if (env_idx >= count) return;

    const BreakoutCudaState& state = states[env_idx];
    int base = env_idx * OBS_SIZE;
    if (lane < 10) {
        float value;
        switch (lane) {
            case 0: value = state.paddle_x / state.width; break;
            case 1: value = state.paddle_y / state.height; break;
            case 2: value = state.ball_x / state.width; break;
            case 3: value = state.ball_y / state.height; break;
            case 4: value = state.ball_vx / 512.0f; break;
            case 5: value = state.ball_vy / 512.0f; break;
            case 6: value = state.balls_fired / 5.0f; break;
            case 7: value = state.score / 864.0f; break;
            case 8: value = state.num_balls / 5.0f; break;
            default:
                value = state.paddle_width / (2.0f * HALF_PADDLE_WIDTH);
                break;
        }
        bc_store(observations, base + lane, value);
    }
    for (int brick = lane; brick < state.num_bricks; brick += 32) {
        bc_store(observations, base + 10 + brick,
            bc_brick_destroyed(state, brick) ? 1.0f : 0.0f);
    }
}

__global__ void bc_clear_logs(BreakoutCudaState* states, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) states[idx].log = {};
}

static int bc_block_size = 32;

static BreakoutCudaState bc_from_host(const Breakout& env) {
    BreakoutCudaState state = {};
    state.log = env.log;
    state.paddle_x = env.paddle_x; state.paddle_y = env.paddle_y;
    state.ball_x = env.ball_x; state.ball_y = env.ball_y;
    state.ball_vx = env.ball_vx; state.ball_vy = env.ball_vy;
    state.initial_paddle_width = env.initial_paddle_width;
    state.paddle_width = env.paddle_width; state.paddle_height = env.paddle_height;
    state.paddle_speed = env.paddle_speed; state.ball_speed = env.ball_speed;
    state.initial_ball_speed = env.initial_ball_speed;
    state.max_ball_speed = env.max_ball_speed;
    for (int i = 0; i < env.num_bricks; ++i) {
        if (env.brick_states[i] != 0.0f) bc_destroy_brick_bit(state, i);
    }
    state.score = env.score; state.balls_fired = env.balls_fired; state.hits = env.hits;
    state.width = env.width; state.height = env.height; state.num_bricks = env.num_bricks;
    state.brick_rows = env.brick_rows; state.brick_cols = env.brick_cols;
    state.ball_width = env.ball_width; state.ball_height = env.ball_height;
    state.brick_width = env.brick_width; state.brick_height = env.brick_height;
    state.num_balls = env.num_balls; state.max_score = env.max_score;
    state.half_max_score = env.half_max_score; state.tick = env.tick;
    state.frameskip = env.frameskip; state.continuous = env.continuous;
    state.rng = env.rng; state.hit_brick = env.hit_brick;
    return state;
}

static void puf_cuda_env_init(VecEnv* vec, cudaStream_t stream) {
    int count = vec->size;
    if (!vec->gpu_env_state) {
        cudaMalloc(&vec->gpu_env_state, (size_t)count * sizeof(BreakoutCudaState));
    }
    BreakoutCudaState* host = (BreakoutCudaState*)calloc(count, sizeof(*host));
    for (int i = 0; i < count; ++i) {
        Breakout& env = ((Breakout*)vec->envs)[i];
        assert(env.num_bricks <= BC_MAX_BRICKS);
        host[i] = bc_from_host(env);
    }
    cudaMemcpyAsync(vec->gpu_env_state, host, (size_t)count * sizeof(*host),
        cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    free(host);

    size_t shared_bytes = (size_t)bc_block_size * sizeof(BreakoutCudaState);
    cudaFuncSetAttribute(bc_step<float>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shared_bytes);
    cudaFuncSetAttribute(bc_step<precision_t>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shared_bytes);
    bc_init_output<float><<<grid_size(count), BLOCK_SIZE, 0, stream>>>(
        (BreakoutCudaState*)vec->gpu_env_state, count,
        vec->gpu_observations, vec->gpu_rewards, vec->gpu_terminals);
}

template <typename T>
static void bc_launch_step(VecEnv* vec, int env_start, int env_count,
        const precision_t* actions, T* observations, T* rewards, T* terminals,
        cudaStream_t stream) {
    size_t shared_bytes = (size_t)bc_block_size * sizeof(BreakoutCudaState);
    bc_step<T><<<(env_count + bc_block_size - 1) / bc_block_size,
        bc_block_size, shared_bytes, stream>>>(
        (BreakoutCudaState*)vec->gpu_env_state + env_start, env_count,
        actions, observations, rewards, terminals);
    bc_observe_warp<T><<<(env_count + BC_OBSERVE_WARPS_PER_BLOCK - 1)
            / BC_OBSERVE_WARPS_PER_BLOCK,
        BC_OBSERVE_BLOCK_SIZE, 0, stream>>>(
        (BreakoutCudaState*)vec->gpu_env_state + env_start, env_count,
        observations);
}

static void puf_cuda_env_step(VecEnv* vec, int env_start, int env_count,
        int agent_start, int agent_count, const precision_t* actions,
        float* observations, float* rewards, float* terminals, cudaStream_t stream) {
    (void)agent_start; (void)agent_count;
    bc_launch_step(vec, env_start, env_count, actions,
        observations, rewards, terminals, stream);
}

static void puf_cuda_env_step_direct(VecEnv* vec, int env_start, int env_count,
        int agent_start, int agent_count, const precision_t* actions,
        precision_t* observations, precision_t* rewards, precision_t* terminals,
        cudaStream_t stream) {
    (void)agent_start; (void)agent_count;
    bc_launch_step(vec, env_start, env_count, actions,
        observations, rewards, terminals, stream);
}

static void puf_cuda_env_sync_logs(VecEnv* vec, int clear, cudaStream_t stream) {
    int count = vec->size;
    BreakoutCudaState* host = (BreakoutCudaState*)malloc((size_t)count * sizeof(*host));
    cudaMemcpy(host, vec->gpu_env_state, (size_t)count * sizeof(*host),
        cudaMemcpyDeviceToHost);
    for (int i = 0; i < count; ++i) ((Breakout*)vec->envs)[i].log = host[i].log;
    free(host);
    if (clear) {
        bc_clear_logs<<<grid_size(count), BLOCK_SIZE, 0, stream>>>(
            (BreakoutCudaState*)vec->gpu_env_state, count);
    }
}

static void puf_cuda_env_close(VecEnv* vec) {
    if (vec->gpu_env_state) cudaFree(vec->gpu_env_state);
    vec->gpu_env_state = nullptr;
}
