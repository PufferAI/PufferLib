#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../affine_lock.cu"

#define EXPECT_TRUE(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: expected true: %s\n", \
            __FILE__, __LINE__, #condition); \
        std::exit(1); \
    } \
} while (0)

#define EXPECT_EQ(actual, expected) do { \
    auto actual_value = (actual); \
    auto expected_value = (expected); \
    if (actual_value != expected_value) { \
        std::fprintf(stderr, "%s:%d: expected %s == %s, got %lld != %lld\n", \
            __FILE__, __LINE__, #actual, #expected, \
            (long long)actual_value, (long long)expected_value); \
        std::exit(1); \
    } \
} while (0)

#define EXPECT_NEAR(actual, expected, tolerance) do { \
    float actual_value = (float)(actual); \
    float expected_value = (float)(expected); \
    if (!std::isfinite(actual_value) || !std::isfinite(expected_value) || \
            std::fabs(actual_value - expected_value) > (tolerance)) { \
        std::fprintf(stderr, "%s:%d: expected %s ~= %.9g, got %.9g\n", \
            __FILE__, __LINE__, #actual, expected_value, actual_value); \
        std::exit(1); \
    } \
} while (0)

static void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

typedef struct OracleState {
    uint32_t rng;
    uint16_t state;
    uint16_t target;
    int step_count;
    int max_steps;
    int scramble_depth;
    int curriculum_depth;
    int target_distance;
    float episode_return;
    Log log;
} OracleState;

static uint32_t oracle_random_mixed_u32(OracleState* env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    uint32_t x = env->rng;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static int oracle_random_bounded(OracleState* env, int bound) {
    uint32_t ubound = (uint32_t)bound;
    uint32_t limit = UINT32_MAX - UINT32_MAX % ubound;
    uint32_t value = oracle_random_mixed_u32(env);
    while (value >= limit) {
        value = oracle_random_mixed_u32(env);
    }
    return (int)(value % ubound);
}

static const VisibleTargetDepth* oracle_depth(
        const VisibleTargetTable* table, int requested_depth) {
    for (uint32_t i = 0; i < table->depth_count; i++) {
        if ((int)table->depths[i].depth == requested_depth) {
            return &table->depths[i];
        }
    }
    return nullptr;
}

static void oracle_reset_state(OracleState* env,
        const VisibleTargetTable* table, int step_grace) {
    env->scramble_depth = env->curriculum_depth;
    env->step_count = 0;
    env->episode_return = 0.0f;
    const VisibleTargetDepth* depth = oracle_depth(table, env->scramble_depth);
    EXPECT_TRUE(depth != nullptr);
    int choice = oracle_random_bounded(env, (int)depth->stored_count);
    const VisibleTargetRecord* record =
        &table->records[depth->first_record + (uint32_t)choice];
    env->state = record->start;
    env->target = record->target;
    env->target_distance = record->depth;
    env->max_steps = env->target_distance + step_grace;
}

static uint16_t oracle_apply_action(uint16_t state, int action) {
    uint32_t next = state;
    switch (action) {
        case 0: next = (state >> 1) | ((state & 1u) << 15); break;
        case 1: next = ((state << 1) & 0xffffu) | ((state >> 15) & 1u); break;
        case 2: next = state ^ 0xfe00u; break;
        case 3: next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1); break;
        case 4: next = ((state & 0x3333u) << 2) | ((state & 0xccccu) >> 2); break;
        case 5: next = ((state & 0x0f0fu) << 4) | ((state & 0xf0f0u) >> 4); break;
        case 6:
            next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
            next = ((next & 0x3333u) << 2) | ((next & 0xccccu) >> 2);
            break;
        case 7:
            next = ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
            next = ((next & 0x3333u) << 2) | ((next & 0xccccu) >> 2);
            next = ((next & 0x0f0fu) << 4) | ((next & 0xf0f0u) >> 4);
            break;
    }
    return (uint16_t)(next & 0xffffu);
}

static int oracle_next_curriculum_depth(int current_depth, int max_depth) {
    static const int curriculum_depths[] = {2, 4, 5, 6, 8, 16};
    for (int depth : curriculum_depths) {
        if (depth > current_depth) {
            return depth < max_depth ? depth : max_depth;
        }
    }
    return max_depth;
}

static void oracle_add_log(OracleState* env, int solved,
        int max_depth, int perf_weighting) {
    int log_depth = env->target_distance;
    int at_max_depth = log_depth == max_depth;
    float ratio = log_depth / (float)max_depth;
    float solve_credit = 0.0f;
    if (solved) {
        solve_credit = perf_weighting == PERF_WEIGHTING_QUADRATIC
            ? ratio * ratio : ratio;
    }
    env->log.perf += solve_credit;
    env->log.score += solve_credit;
    env->log.solve_rate += solved;
    env->log.max_depth_solve += solved && at_max_depth;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->step_count;
    env->log.solve_steps += solved ? env->step_count : 0;
    env->log.timeout_rate += !solved;
    env->log.solve_efficiency += solved
        ? env->step_count / (float)log_depth : 0.0f;
    env->log.target_distance += env->target_distance;
    env->log.solved_target_distance += solved ? env->target_distance : 0;
    env->log.d6_rate += log_depth == 6;
    env->log.d6_solve_rate += solved && log_depth == 6;
    env->log.d8_rate += log_depth == 8;
    env->log.d8_solve_rate += solved && log_depth == 8;
    env->log.d16_rate += log_depth == 16;
    env->log.d16_solve_rate += solved && log_depth == 16;
    env->log.n += 1;
}

static void oracle_step(OracleState* env, float action,
        const VisibleTargetTable* table, int start_depth, int max_depth,
        int step_grace, int perf_weighting, float* reward, float* terminal) {
    *reward = STEP_REWARD;
    *terminal = 0.0f;
    int solved = 0;
    env->step_count += 1;
    int invalid = !std::isfinite(action) || action < 0.0f || action > 7.0f;
    if (invalid) {
        *reward = -1.0f;
        *terminal = 1.0f;
    } else {
        env->state = oracle_apply_action(env->state, (int)action);
        if (env->state == env->target) {
            *reward = 1.0f;
            *terminal = 1.0f;
            solved = 1;
        } else if (env->step_count >= env->max_steps) {
            *reward = -1.0f;
            *terminal = 1.0f;
        }
    }
    env->episode_return += *reward;
    if (*terminal != 0.0f) {
        oracle_add_log(env, solved, max_depth, perf_weighting);
        env->curriculum_depth = solved
            ? oracle_next_curriculum_depth(env->scramble_depth, max_depth)
            : start_depth;
        oracle_reset_state(env, table, step_grace);
    }
}

static uint16_t obs_bits(obs_t value) {
    uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void expect_log_equal(const Log& actual, const Log& expected) {
    const float* a = (const float*)&actual;
    const float* e = (const float*)&expected;
    for (size_t i = 0; i < sizeof(Log) / sizeof(float); i++) {
        EXPECT_EQ(float_bits(a[i]), float_bits(e[i]));
    }
}

static void expect_state_equal(const GpuAffineLockState& actual,
        const OracleState& expected) {
    EXPECT_EQ(actual.rng, expected.rng);
    EXPECT_EQ(actual.state, expected.state);
    EXPECT_EQ(actual.target, expected.target);
    EXPECT_EQ(actual.step_count, expected.step_count);
    EXPECT_EQ(actual.max_steps, expected.max_steps);
    EXPECT_EQ(actual.scramble_depth, expected.scramble_depth);
    EXPECT_EQ(actual.curriculum_depth, expected.curriculum_depth);
    EXPECT_EQ(actual.target_distance, expected.target_distance);
    EXPECT_EQ(float_bits(actual.episode_return),
        float_bits(expected.episode_return));
}

static void expect_observation_equal(const obs_t* actual,
        const OracleState& expected) {
    uint32_t bits = (uint32_t)expected.state | ((uint32_t)expected.target << 16);
    for (int bit = 0; bit < 32; bit++) {
        float value = (bits & (1u << bit)) ? 1.0f : -1.0f;
        EXPECT_EQ(obs_bits(actual[bit]), obs_bits(__float2bfloat16(value)));
    }
    float timer = expected.step_count / (float)expected.max_steps;
    EXPECT_EQ(obs_bits(actual[TIMER_INDEX]),
        obs_bits(__float2bfloat16(timer)));
}

static void fill_kwargs(Dict* kwargs, int seed, int step_grace,
        int perf_weighting) {
    std::memset(kwargs, 0, sizeof(*kwargs));
    dict_set(kwargs, "seed", seed);
    dict_set(kwargs, "start_depth", 2);
    dict_set(kwargs, "max_depth", 16);
    dict_set(kwargs, "step_grace", step_grace);
    dict_set(kwargs, "perf_weighting", perf_weighting);
}

static void test_deterministic_reset_and_step_parity() {
    constexpr int n = 257;
    constexpr int seed = 42;
    constexpr int step_grace = 2;
    constexpr int perf_weighting = PERF_WEIGHTING_QUADRATIC;

    VisibleTargetTable table = {};
    EXPECT_EQ(visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
        VISIBLE_TARGET_8ACTION_V1_HASH, &table), 0);

    obs_t* observations = nullptr;
    float* actions = nullptr;
    float* rewards = nullptr;
    float* terminals = nullptr;
    check_cuda(cudaMalloc(&observations, (size_t)n * OBS_SIZE * sizeof(obs_t)), "cudaMalloc observations");
    check_cuda(cudaMalloc(&actions, (size_t)n * sizeof(float)), "cudaMalloc actions");
    check_cuda(cudaMalloc(&rewards, (size_t)n * sizeof(float)), "cudaMalloc rewards");
    check_cuda(cudaMalloc(&terminals, (size_t)n * sizeof(float)), "cudaMalloc terminals");

    Dict kwargs;
    fill_kwargs(&kwargs, seed, step_grace, perf_weighting);
    Env* envs = puf_vec_create(n, &kwargs, observations, actions, rewards, terminals);
    EXPECT_TRUE(envs != nullptr);
    puf_reset(envs);
    check_cuda(cudaDeviceSynchronize(), "initial reset");

    std::vector<OracleState> oracle(n);
    unsigned int running_seed = seed;
    for (int i = 0; i < n; i++) {
        oracle[i].rng = (uint32_t)rand_r(&running_seed);
        oracle[i].curriculum_depth = 2;
        oracle_reset_state(&oracle[i], &table, step_grace);
    }

    std::vector<GpuAffineLockState> states(n);
    std::vector<Env> host_envs(n);
    std::vector<obs_t> host_obs((size_t)n * OBS_SIZE);
    std::vector<float> host_rewards(n), host_terminals(n), host_actions(n);
    check_cuda(cudaMemcpy(states.data(), g_gpu.states,
        n * sizeof(GpuAffineLockState), cudaMemcpyDeviceToHost), "copy reset states");
    check_cuda(cudaMemcpy(host_obs.data(), observations,
        host_obs.size() * sizeof(obs_t), cudaMemcpyDeviceToHost), "copy reset observations");
    check_cuda(cudaMemcpy(host_rewards.data(), rewards,
        n * sizeof(float), cudaMemcpyDeviceToHost), "copy reset rewards");
    check_cuda(cudaMemcpy(host_terminals.data(), terminals,
        n * sizeof(float), cudaMemcpyDeviceToHost), "copy reset terminals");
    for (int i = 0; i < n; i++) {
        expect_state_equal(states[i], oracle[i]);
        expect_observation_equal(&host_obs[(size_t)i * OBS_SIZE], oracle[i]);
        EXPECT_NEAR(host_rewards[i], 0.0f, 0.0f);
        EXPECT_NEAR(host_terminals[i], 0.0f, 0.0f);
    }

    for (int step = 0; step < 96; step++) {
        for (int i = 0; i < n; i++) {
            int selector = (step * 17 + i * 13) % 41;
            if (selector == 0) host_actions[i] = std::numeric_limits<float>::quiet_NaN();
            else if (selector == 1) host_actions[i] = -0.25f;
            else if (selector == 2) host_actions[i] = 8.0f;
            else host_actions[i] = (float)((step + 3 * i) & 7) + (selector == 3 ? 0.75f : 0.0f);
        }
        check_cuda(cudaMemcpy(actions, host_actions.data(),
            n * sizeof(float), cudaMemcpyHostToDevice), "copy actions");
        puf_step(envs);
        check_cuda(cudaDeviceSynchronize(), "step");

        for (int i = 0; i < n; i++) {
            oracle_step(&oracle[i], host_actions[i], &table,
                2, 16, step_grace, perf_weighting,
                &host_rewards[i], &host_terminals[i]);
        }

        check_cuda(cudaMemcpy(states.data(), g_gpu.states,
            n * sizeof(GpuAffineLockState), cudaMemcpyDeviceToHost), "copy states");
        check_cuda(cudaMemcpy(host_envs.data(), envs,
            n * sizeof(Env), cudaMemcpyDeviceToHost), "copy logs");
        check_cuda(cudaMemcpy(host_obs.data(), observations,
            host_obs.size() * sizeof(obs_t), cudaMemcpyDeviceToHost), "copy observations");
        std::vector<float> actual_rewards(n), actual_terminals(n);
        check_cuda(cudaMemcpy(actual_rewards.data(), rewards,
            n * sizeof(float), cudaMemcpyDeviceToHost), "copy rewards");
        check_cuda(cudaMemcpy(actual_terminals.data(), terminals,
            n * sizeof(float), cudaMemcpyDeviceToHost), "copy terminals");
        for (int i = 0; i < n; i++) {
            expect_state_equal(states[i], oracle[i]);
            expect_log_equal(host_envs[i].log, oracle[i].log);
            expect_observation_equal(&host_obs[(size_t)i * OBS_SIZE], oracle[i]);
            EXPECT_NEAR(actual_rewards[i], host_rewards[i], 0.0f);
            EXPECT_NEAR(actual_terminals[i], host_terminals[i], 0.0f);
        }
    }

    puf_close(envs);
    dict_clear(&kwargs);
    visible_targets_free(&table);
    check_cuda(cudaFree(observations), "cudaFree observations");
    check_cuda(cudaFree(actions), "cudaFree actions");
    check_cuda(cudaFree(rewards), "cudaFree rewards");
    check_cuda(cudaFree(terminals), "cudaFree terminals");
}

static void test_reset_rejection_sampling() {
    constexpr uint32_t rejection_seed = 24481u;
    constexpr uint32_t expected_final_rng = 3424986747u;
    static const int depths[] = {2, 16};
    static const int expected_choices[] = {44338, 15778};

    VisibleTargetTable table = {};
    EXPECT_EQ(visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
        VISIBLE_TARGET_8ACTION_V1_HASH, &table), 0);

    obs_t* observations = nullptr;
    float* actions = nullptr;
    float* rewards = nullptr;
    float* terminals = nullptr;
    check_cuda(cudaMalloc(&observations, OBS_SIZE * sizeof(obs_t)),
        "cudaMalloc rejection observations");
    check_cuda(cudaMalloc(&actions, sizeof(float)),
        "cudaMalloc rejection action");
    check_cuda(cudaMalloc(&rewards, sizeof(float)),
        "cudaMalloc rejection reward");
    check_cuda(cudaMalloc(&terminals, sizeof(float)),
        "cudaMalloc rejection terminal");

    Dict kwargs;
    fill_kwargs(&kwargs, 1, 0, PERF_WEIGHTING_LINEAR);
    Env* envs = puf_vec_create(1, &kwargs,
        observations, actions, rewards, terminals);

    for (int case_index = 0; case_index < 2; case_index++) {
        int depth = depths[case_index];
        GpuAffineLockState injected = {};
        injected.rng = rejection_seed;
        injected.curriculum_depth = depth;
        check_cuda(cudaMemcpy(g_gpu.states, &injected, sizeof(injected),
            cudaMemcpyHostToDevice), "inject rejection state");

        puf_reset(envs);
        check_cuda(cudaDeviceSynchronize(), "rejection reset");

        OracleState expected = {};
        expected.rng = rejection_seed;
        expected.curriculum_depth = depth;
        oracle_reset_state(&expected, &table, 0);
        EXPECT_EQ(expected.rng, expected_final_rng);
        const VisibleTargetDepth* table_depth = oracle_depth(&table, depth);
        EXPECT_TRUE(table_depth != nullptr);
        const VisibleTargetRecord* selected = &table.records[
            table_depth->first_record + (uint32_t)expected_choices[case_index]];
        EXPECT_EQ(expected.state, selected->start);
        EXPECT_EQ(expected.target, selected->target);

        GpuAffineLockState actual = {};
        obs_t actual_obs[OBS_SIZE];
        float actual_reward = 123.0f;
        float actual_terminal = 123.0f;
        check_cuda(cudaMemcpy(&actual, g_gpu.states, sizeof(actual),
            cudaMemcpyDeviceToHost), "copy rejection state");
        check_cuda(cudaMemcpy(actual_obs, observations, sizeof(actual_obs),
            cudaMemcpyDeviceToHost), "copy rejection observations");
        check_cuda(cudaMemcpy(&actual_reward, rewards, sizeof(actual_reward),
            cudaMemcpyDeviceToHost), "copy rejection reward");
        check_cuda(cudaMemcpy(&actual_terminal, terminals, sizeof(actual_terminal),
            cudaMemcpyDeviceToHost), "copy rejection terminal");
        expect_state_equal(actual, expected);
        expect_observation_equal(actual_obs, expected);
        EXPECT_EQ(float_bits(actual_reward), float_bits(0.0f));
        EXPECT_EQ(float_bits(actual_terminal), float_bits(0.0f));
    }

    puf_close(envs);
    dict_clear(&kwargs);
    visible_targets_free(&table);
    check_cuda(cudaFree(observations), "cudaFree rejection observations");
    check_cuda(cudaFree(actions), "cudaFree rejection action");
    check_cuda(cudaFree(rewards), "cudaFree rejection reward");
    check_cuda(cudaFree(terminals), "cudaFree rejection terminal");
}

static void test_puf_log_exports_cpu_contract() {
    Log log = {};
    log.perf = 1.25f;
    log.score = 2.5f;
    log.solve_rate = 2.0f;
    log.max_depth_solve = 1.0f;
    log.episode_return = 3.5f;
    log.episode_length = 8.0f;
    log.solve_steps = 5.0f;
    log.timeout_rate = 1.0f;
    log.solve_efficiency = 1.75f;
    log.target_distance = 20.0f;
    log.solved_target_distance = 12.0f;
    log.d6_rate = 2.0f;
    log.d6_solve_rate = 1.0f;
    log.d8_rate = 4.0f;
    log.d8_solve_rate = 3.0f;
    log.d16_rate = 1.0f;
    log.d16_solve_rate = 1.0f;
    log.n = 3.0f;
    Dict out = {};
    puf_log(&log, &out);
    EXPECT_EQ(out.size, 15);
    EXPECT_NEAR(dict_get(&out, "perf"), 1.25f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "score"), 2.5f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "solve_rate"), 2.0f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "max_depth_solve"), 1.0f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "episode_return"), 3.5f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "episode_length"), 8.0f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "timeout_rate"), 1.0f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "min_win_moves"), 20.0f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "solved_min_win_moves"), 6.0f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "conditional_solve_steps"), 2.5f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "conditional_solve_efficiency"), 0.875f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "d6_solve_rate"), 0.5f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "d8_solve_rate"), 0.75f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "d16_solve_rate"), 1.0f, 0.0f);
    EXPECT_NEAR(dict_get(&out, "n"), 3.0f, 0.0f);
    dict_clear(&out);

    Log zero_denominators = {};
    zero_denominators.solved_target_distance = 12.0f;
    zero_denominators.solve_steps = 5.0f;
    zero_denominators.solve_efficiency = 1.75f;
    zero_denominators.d6_solve_rate = 1.0f;
    zero_denominators.d8_solve_rate = 1.0f;
    zero_denominators.d16_solve_rate = 1.0f;
    Dict zero_out = {};
    puf_log(&zero_denominators, &zero_out);
    EXPECT_EQ(zero_out.size, 15);
    EXPECT_NEAR(dict_get(&zero_out, "solved_min_win_moves"), 0.0f, 0.0f);
    EXPECT_NEAR(dict_get(&zero_out, "conditional_solve_steps"), 0.0f, 0.0f);
    EXPECT_NEAR(dict_get(&zero_out, "conditional_solve_efficiency"), 0.0f, 0.0f);
    EXPECT_NEAR(dict_get(&zero_out, "d6_solve_rate"), 0.0f, 0.0f);
    EXPECT_NEAR(dict_get(&zero_out, "d8_solve_rate"), 0.0f, 0.0f);
    EXPECT_NEAR(dict_get(&zero_out, "d16_solve_rate"), 0.0f, 0.0f);
    dict_clear(&zero_out);
}

static void test_exhaustive_action_transforms() {
    constexpr int n = 1 << BITS;
    obs_t* observations = nullptr;
    float* actions = nullptr;
    float* rewards = nullptr;
    float* terminals = nullptr;
    check_cuda(cudaMalloc(&observations, (size_t)n * OBS_SIZE * sizeof(obs_t)),
        "cudaMalloc exhaustive observations");
    check_cuda(cudaMalloc(&actions, (size_t)n * sizeof(float)),
        "cudaMalloc exhaustive actions");
    check_cuda(cudaMalloc(&rewards, (size_t)n * sizeof(float)),
        "cudaMalloc exhaustive rewards");
    check_cuda(cudaMalloc(&terminals, (size_t)n * sizeof(float)),
        "cudaMalloc exhaustive terminals");

    Dict kwargs;
    fill_kwargs(&kwargs, 7, 100, PERF_WEIGHTING_LINEAR);
    Env* envs = puf_vec_create(n, &kwargs,
        observations, actions, rewards, terminals);
    std::vector<GpuAffineLockState> states(n);
    std::vector<float> host_actions(n), host_rewards(n), host_terminals(n);

    for (int action = 0; action < NUM_ACTIONS; action++) {
        for (int value = 0; value < n; value++) {
            uint16_t expected = oracle_apply_action((uint16_t)value, action);
            states[value] = {};
            states[value].rng = (uint32_t)(value + 1);
            states[value].state = (uint16_t)value;
            states[value].target = expected ^ 1u;
            states[value].max_steps = 100;
            states[value].scramble_depth = 16;
            states[value].curriculum_depth = 16;
            states[value].target_distance = 16;
            host_actions[value] = (float)action;
        }
        check_cuda(cudaMemcpy(g_gpu.states, states.data(),
            n * sizeof(GpuAffineLockState), cudaMemcpyHostToDevice),
            "copy exhaustive states");
        check_cuda(cudaMemcpy(actions, host_actions.data(),
            n * sizeof(float), cudaMemcpyHostToDevice),
            "copy exhaustive actions");
        puf_step(envs);
        check_cuda(cudaDeviceSynchronize(), "exhaustive action step");
        check_cuda(cudaMemcpy(states.data(), g_gpu.states,
            n * sizeof(GpuAffineLockState), cudaMemcpyDeviceToHost),
            "copy exhaustive results");
        check_cuda(cudaMemcpy(host_rewards.data(), rewards,
            n * sizeof(float), cudaMemcpyDeviceToHost),
            "copy exhaustive rewards");
        check_cuda(cudaMemcpy(host_terminals.data(), terminals,
            n * sizeof(float), cudaMemcpyDeviceToHost),
            "copy exhaustive terminals");
        for (int value = 0; value < n; value++) {
            EXPECT_EQ(states[value].state,
                oracle_apply_action((uint16_t)value, action));
            EXPECT_EQ(states[value].step_count, 1);
            EXPECT_NEAR(states[value].episode_return, STEP_REWARD, 0.0f);
            EXPECT_NEAR(host_rewards[value], STEP_REWARD, 0.0f);
            EXPECT_NEAR(host_terminals[value], 0.0f, 0.0f);
        }
    }

    puf_close(envs);
    dict_clear(&kwargs);
    check_cuda(cudaFree(observations), "cudaFree exhaustive observations");
    check_cuda(cudaFree(actions), "cudaFree exhaustive actions");
    check_cuda(cudaFree(rewards), "cudaFree exhaustive rewards");
    check_cuda(cudaFree(terminals), "cudaFree exhaustive terminals");
}

static void test_action_boundaries() {
    const float infinity = std::numeric_limits<float>::infinity();
    const float actions_under_test[] = {
        -infinity,
        -0.25f,
        -std::numeric_limits<float>::denorm_min(),
        -0.0f,
        0.0f,
        0.5f,
        0.999f,
        1.0f,
        1.5f,
        6.999f,
        7.0f,
        std::nextafter(7.0f, infinity),
        8.0f,
        infinity,
        std::numeric_limits<float>::quiet_NaN(),
    };
    constexpr int n = sizeof(actions_under_test) / sizeof(actions_under_test[0]);

    VisibleTargetTable table = {};
    EXPECT_EQ(visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
        VISIBLE_TARGET_8ACTION_V1_HASH, &table), 0);
    obs_t* observations = nullptr;
    float* actions = nullptr;
    float* rewards = nullptr;
    float* terminals = nullptr;
    check_cuda(cudaMalloc(&observations, (size_t)n * OBS_SIZE * sizeof(obs_t)),
        "cudaMalloc boundary observations");
    check_cuda(cudaMalloc(&actions, (size_t)n * sizeof(float)),
        "cudaMalloc boundary actions");
    check_cuda(cudaMalloc(&rewards, (size_t)n * sizeof(float)),
        "cudaMalloc boundary rewards");
    check_cuda(cudaMalloc(&terminals, (size_t)n * sizeof(float)),
        "cudaMalloc boundary terminals");

    Dict kwargs;
    fill_kwargs(&kwargs, 11, 100, PERF_WEIGHTING_LINEAR);
    Env* envs = puf_vec_create(n, &kwargs,
        observations, actions, rewards, terminals);

    std::vector<GpuAffineLockState> injected(n);
    std::vector<OracleState> expected(n);
    std::vector<float> expected_rewards(n), expected_terminals(n);
    for (int i = 0; i < n; i++) {
        injected[i] = {};
        injected[i].rng = (uint32_t)(1000 + i);
        injected[i].state = 0x1234u;
        injected[i].target = 0xbeefu;
        injected[i].max_steps = 100;
        injected[i].scramble_depth = 2;
        injected[i].curriculum_depth = 2;
        injected[i].target_distance = 2;

        expected[i].rng = injected[i].rng;
        expected[i].state = injected[i].state;
        expected[i].target = injected[i].target;
        expected[i].max_steps = injected[i].max_steps;
        expected[i].scramble_depth = injected[i].scramble_depth;
        expected[i].curriculum_depth = injected[i].curriculum_depth;
        expected[i].target_distance = injected[i].target_distance;
        oracle_step(&expected[i], actions_under_test[i], &table,
            2, 16, 100, PERF_WEIGHTING_LINEAR,
            &expected_rewards[i], &expected_terminals[i]);
    }
    check_cuda(cudaMemcpy(g_gpu.states, injected.data(),
        (size_t)n * sizeof(GpuAffineLockState), cudaMemcpyHostToDevice),
        "copy boundary states");
    check_cuda(cudaMemcpy(actions, actions_under_test,
        sizeof(actions_under_test), cudaMemcpyHostToDevice),
        "copy boundary actions");
    puf_step(envs);
    check_cuda(cudaDeviceSynchronize(), "boundary step");

    std::vector<GpuAffineLockState> actual_states(n);
    std::vector<Env> actual_envs(n);
    std::vector<obs_t> actual_obs((size_t)n * OBS_SIZE);
    std::vector<float> actual_rewards(n), actual_terminals(n);
    check_cuda(cudaMemcpy(actual_states.data(), g_gpu.states,
        (size_t)n * sizeof(GpuAffineLockState), cudaMemcpyDeviceToHost),
        "copy boundary results");
    check_cuda(cudaMemcpy(actual_envs.data(), envs,
        (size_t)n * sizeof(Env), cudaMemcpyDeviceToHost),
        "copy boundary logs");
    check_cuda(cudaMemcpy(actual_obs.data(), observations,
        actual_obs.size() * sizeof(obs_t), cudaMemcpyDeviceToHost),
        "copy boundary observations");
    check_cuda(cudaMemcpy(actual_rewards.data(), rewards,
        (size_t)n * sizeof(float), cudaMemcpyDeviceToHost),
        "copy boundary rewards");
    check_cuda(cudaMemcpy(actual_terminals.data(), terminals,
        (size_t)n * sizeof(float), cudaMemcpyDeviceToHost),
        "copy boundary terminals");

    for (int i = 0; i < n; i++) {
        bool invalid = !std::isfinite(actions_under_test[i])
            || actions_under_test[i] < 0.0f || actions_under_test[i] > 7.0f;
        expect_state_equal(actual_states[i], expected[i]);
        expect_log_equal(actual_envs[i].log, expected[i].log);
        expect_observation_equal(
            &actual_obs[(size_t)i * OBS_SIZE], expected[i]);
        EXPECT_EQ(float_bits(actual_rewards[i]),
            float_bits(expected_rewards[i]));
        EXPECT_EQ(float_bits(actual_terminals[i]),
            float_bits(expected_terminals[i]));
        EXPECT_EQ(float_bits(actual_terminals[i]),
            float_bits(invalid ? 1.0f : 0.0f));
        EXPECT_EQ(float_bits(actual_envs[i].log.n),
            float_bits(invalid ? 1.0f : 0.0f));
    }

    puf_close(envs);
    dict_clear(&kwargs);
    visible_targets_free(&table);
    check_cuda(cudaFree(observations), "cudaFree boundary observations");
    check_cuda(cudaFree(actions), "cudaFree boundary actions");
    check_cuda(cudaFree(rewards), "cudaFree boundary rewards");
    check_cuda(cudaFree(terminals), "cudaFree boundary terminals");
}

static const VisibleTargetRecord* find_solution_record(
        const VisibleTargetTable* table, const OracleState& state) {
    const VisibleTargetDepth* depth = oracle_depth(table, state.scramble_depth);
    EXPECT_TRUE(depth != nullptr);
    for (uint32_t i = 0; i < depth->stored_count; i++) {
        const VisibleTargetRecord* record =
            &table->records[depth->first_record + i];
        if (record->start == state.state && record->target == state.target) {
            return record;
        }
    }
    return nullptr;
}

static void test_solution_curriculum_and_logs() {
    constexpr int seed = 69;
    obs_t* observations = nullptr;
    float* actions = nullptr;
    float* rewards = nullptr;
    float* terminals = nullptr;
    check_cuda(cudaMalloc(&observations, OBS_SIZE * sizeof(obs_t)),
        "cudaMalloc solution observations");
    check_cuda(cudaMalloc(&actions, sizeof(float)), "cudaMalloc solution action");
    check_cuda(cudaMalloc(&rewards, sizeof(float)), "cudaMalloc solution reward");
    check_cuda(cudaMalloc(&terminals, sizeof(float)), "cudaMalloc solution terminal");

    VisibleTargetTable table = {};
    EXPECT_EQ(visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
        VISIBLE_TARGET_8ACTION_V1_HASH, &table), 0);
    Dict kwargs;
    fill_kwargs(&kwargs, seed, 0, PERF_WEIGHTING_QUADRATIC);
    Env* envs = puf_vec_create(1, &kwargs,
        observations, actions, rewards, terminals);
    puf_reset(envs);
    check_cuda(cudaDeviceSynchronize(), "solution reset");

    OracleState oracle = {};
    unsigned int running_seed = seed;
    oracle.rng = (uint32_t)rand_r(&running_seed);
    oracle.curriculum_depth = 2;
    oracle_reset_state(&oracle, &table, 0);
    static const int expected_depths[] = {2, 4, 5, 6, 8, 16};
    for (int expected_depth : expected_depths) {
        EXPECT_EQ(oracle.scramble_depth, expected_depth);
        const VisibleTargetRecord* record = find_solution_record(&table, oracle);
        EXPECT_TRUE(record != nullptr);
        for (int move = 0; move < record->solution_length; move++) {
            float action = (float)((record->packed_actions >> (3 * move)) & 7u);
            check_cuda(cudaMemcpy(actions, &action, sizeof(float),
                cudaMemcpyHostToDevice), "copy solution action");
            puf_step(envs);
            check_cuda(cudaDeviceSynchronize(), "solution step");

            float expected_reward = 0.0f;
            float expected_terminal = 0.0f;
            oracle_step(&oracle, action, &table, 2, 16, 0,
                PERF_WEIGHTING_QUADRATIC,
                &expected_reward, &expected_terminal);
            GpuAffineLockState actual_state;
            Env actual_env;
            obs_t actual_obs[OBS_SIZE];
            float actual_reward = 0.0f;
            float actual_terminal = 0.0f;
            check_cuda(cudaMemcpy(&actual_state, g_gpu.states,
                sizeof(actual_state), cudaMemcpyDeviceToHost),
                "copy solution state");
            check_cuda(cudaMemcpy(&actual_env, envs,
                sizeof(actual_env), cudaMemcpyDeviceToHost),
                "copy solution log");
            check_cuda(cudaMemcpy(actual_obs, observations,
                sizeof(actual_obs), cudaMemcpyDeviceToHost),
                "copy solution observations");
            check_cuda(cudaMemcpy(&actual_reward, rewards,
                sizeof(float), cudaMemcpyDeviceToHost),
                "copy solution reward");
            check_cuda(cudaMemcpy(&actual_terminal, terminals,
                sizeof(float), cudaMemcpyDeviceToHost),
                "copy solution terminal");
            expect_state_equal(actual_state, oracle);
            expect_log_equal(actual_env.log, oracle.log);
            expect_observation_equal(actual_obs, oracle);
            EXPECT_NEAR(actual_reward, expected_reward, 0.0f);
            EXPECT_NEAR(actual_terminal, expected_terminal, 0.0f);
            EXPECT_NEAR(actual_terminal,
                move + 1 == record->solution_length ? 1.0f : 0.0f, 0.0f);
        }
    }
    EXPECT_NEAR(oracle.log.solve_rate, 6.0f, 0.0f);
    EXPECT_NEAR(oracle.log.d6_solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(oracle.log.d8_solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(oracle.log.d16_solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(oracle.log.max_depth_solve, 1.0f, 0.0f);

    puf_close(envs);
    dict_clear(&kwargs);
    visible_targets_free(&table);
    check_cuda(cudaFree(observations), "cudaFree solution observations");
    check_cuda(cudaFree(actions), "cudaFree solution action");
    check_cuda(cudaFree(rewards), "cudaFree solution reward");
    check_cuda(cudaFree(terminals), "cudaFree solution terminal");
}

static void test_linear_solve_scoring() {
    constexpr int seed = 17;
    obs_t* observations = nullptr;
    float* actions = nullptr;
    float* rewards = nullptr;
    float* terminals = nullptr;
    check_cuda(cudaMalloc(&observations, OBS_SIZE * sizeof(obs_t)),
        "cudaMalloc linear observations");
    check_cuda(cudaMalloc(&actions, sizeof(float)),
        "cudaMalloc linear action");
    check_cuda(cudaMalloc(&rewards, sizeof(float)),
        "cudaMalloc linear reward");
    check_cuda(cudaMalloc(&terminals, sizeof(float)),
        "cudaMalloc linear terminal");

    VisibleTargetTable table = {};
    EXPECT_EQ(visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
        VISIBLE_TARGET_8ACTION_V1_HASH, &table), 0);
    Dict kwargs;
    fill_kwargs(&kwargs, seed, 0, PERF_WEIGHTING_LINEAR);
    Env* envs = puf_vec_create(1, &kwargs,
        observations, actions, rewards, terminals);
    puf_reset(envs);
    check_cuda(cudaDeviceSynchronize(), "linear reset");

    OracleState oracle = {};
    unsigned int running_seed = seed;
    oracle.rng = (uint32_t)rand_r(&running_seed);
    oracle.curriculum_depth = 2;
    oracle_reset_state(&oracle, &table, 0);
    const VisibleTargetRecord* record = find_solution_record(&table, oracle);
    EXPECT_TRUE(record != nullptr);
    for (int move = 0; move < record->solution_length; move++) {
        float action = (float)((record->packed_actions >> (3 * move)) & 7u);
        check_cuda(cudaMemcpy(actions, &action, sizeof(action),
            cudaMemcpyHostToDevice), "copy linear solution action");
        puf_step(envs);
    }
    check_cuda(cudaDeviceSynchronize(), "linear solution");

    Env actual_env = {};
    float actual_reward = 0.0f;
    float actual_terminal = 0.0f;
    check_cuda(cudaMemcpy(&actual_env, envs, sizeof(actual_env),
        cudaMemcpyDeviceToHost), "copy linear log");
    check_cuda(cudaMemcpy(&actual_reward, rewards, sizeof(actual_reward),
        cudaMemcpyDeviceToHost), "copy linear reward");
    check_cuda(cudaMemcpy(&actual_terminal, terminals, sizeof(actual_terminal),
        cudaMemcpyDeviceToHost), "copy linear terminal");
    EXPECT_NEAR(actual_env.log.perf, 2.0f / 16.0f, 0.0f);
    EXPECT_NEAR(actual_env.log.score, 2.0f / 16.0f, 0.0f);
    EXPECT_NEAR(actual_env.log.solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(actual_env.log.n, 1.0f, 0.0f);
    EXPECT_NEAR(actual_reward, 1.0f, 0.0f);
    EXPECT_NEAR(actual_terminal, 1.0f, 0.0f);

    puf_close(envs);
    dict_clear(&kwargs);
    visible_targets_free(&table);
    check_cuda(cudaFree(observations), "cudaFree linear observations");
    check_cuda(cudaFree(actions), "cudaFree linear action");
    check_cuda(cudaFree(rewards), "cudaFree linear reward");
    check_cuda(cudaFree(terminals), "cudaFree linear terminal");
}

static void expect_device_canaries(const unsigned char* device_storage,
        size_t prefix_bytes, size_t payload_bytes, size_t suffix_bytes,
        unsigned char canary, const char* label) {
    size_t total_bytes = prefix_bytes + payload_bytes + suffix_bytes;
    std::vector<unsigned char> host_storage(total_bytes);
    check_cuda(cudaMemcpy(host_storage.data(), device_storage, total_bytes,
        cudaMemcpyDeviceToHost), "copy canary storage");
    for (size_t i = 0; i < prefix_bytes; i++) {
        if (host_storage[i] != canary) {
            std::fprintf(stderr,
                "%s prefix canary overwritten at byte %zu: 0x%02x != 0x%02x\n",
                label, i, host_storage[i], canary);
            std::exit(1);
        }
    }
    size_t suffix_start = prefix_bytes + payload_bytes;
    for (size_t i = suffix_start; i < total_bytes; i++) {
        if (host_storage[i] != canary) {
            std::fprintf(stderr,
                "%s suffix canary overwritten at byte %zu: 0x%02x != 0x%02x\n",
                label, i - suffix_start, host_storage[i], canary);
            std::exit(1);
        }
    }
}

static void run_io_canary_case(int n) {
    constexpr size_t guard_bytes = 64;
    constexpr size_t observation_prefix = guard_bytes + sizeof(obs_t);
    constexpr unsigned char canary = 0xa5u;
    size_t observation_bytes = (size_t)n * OBS_SIZE * sizeof(obs_t);
    size_t scalar_bytes = (size_t)n * sizeof(float);
    size_t observation_storage_bytes = observation_prefix
        + observation_bytes + guard_bytes;
    size_t scalar_storage_bytes = guard_bytes + scalar_bytes + guard_bytes;

    unsigned char* observation_storage = nullptr;
    unsigned char* reward_storage = nullptr;
    unsigned char* terminal_storage = nullptr;
    float* actions = nullptr;
    check_cuda(cudaMalloc(&observation_storage, observation_storage_bytes),
        "cudaMalloc guarded observations");
    check_cuda(cudaMalloc(&reward_storage, scalar_storage_bytes),
        "cudaMalloc guarded rewards");
    check_cuda(cudaMalloc(&terminal_storage, scalar_storage_bytes),
        "cudaMalloc guarded terminals");
    check_cuda(cudaMalloc(&actions, scalar_bytes),
        "cudaMalloc guarded actions");

    obs_t* observations = reinterpret_cast<obs_t*>(
        observation_storage + observation_prefix);
    float* rewards = reinterpret_cast<float*>(reward_storage + guard_bytes);
    float* terminals = reinterpret_cast<float*>(terminal_storage + guard_bytes);
    EXPECT_EQ((uintptr_t)observations & 1u, 0u);
    EXPECT_EQ((uintptr_t)observations & 3u, 2u);
    EXPECT_EQ((uintptr_t)rewards & 3u, 0u);
    EXPECT_EQ((uintptr_t)terminals & 3u, 0u);

    check_cuda(cudaMemset(observation_storage, canary,
        observation_storage_bytes), "initialize observation canaries");
    check_cuda(cudaMemset(reward_storage, canary, scalar_storage_bytes),
        "initialize reward canaries");
    check_cuda(cudaMemset(terminal_storage, canary, scalar_storage_bytes),
        "initialize terminal canaries");
    check_cuda(cudaMemset(actions, 0, scalar_bytes),
        "initialize guarded actions");

    Dict kwargs;
    fill_kwargs(&kwargs, 31 + n, 1, PERF_WEIGHTING_LINEAR);
    Env* envs = puf_vec_create(n, &kwargs,
        observations, actions, rewards, terminals);
    puf_reset(envs);
    check_cuda(cudaDeviceSynchronize(), "guarded reset");
    expect_device_canaries(observation_storage, observation_prefix,
        observation_bytes, guard_bytes, canary, "reset observations");
    expect_device_canaries(reward_storage, guard_bytes,
        scalar_bytes, guard_bytes, canary, "reset rewards");
    expect_device_canaries(terminal_storage, guard_bytes,
        scalar_bytes, guard_bytes, canary, "reset terminals");

    check_cuda(cudaMemset(observation_storage, canary,
        observation_storage_bytes), "reinitialize observation canaries");
    check_cuda(cudaMemset(reward_storage, canary, scalar_storage_bytes),
        "reinitialize reward canaries");
    check_cuda(cudaMemset(terminal_storage, canary, scalar_storage_bytes),
        "reinitialize terminal canaries");
    puf_step(envs);
    check_cuda(cudaDeviceSynchronize(), "guarded step");
    expect_device_canaries(observation_storage, observation_prefix,
        observation_bytes, guard_bytes, canary, "step observations");
    expect_device_canaries(reward_storage, guard_bytes,
        scalar_bytes, guard_bytes, canary, "step rewards");
    expect_device_canaries(terminal_storage, guard_bytes,
        scalar_bytes, guard_bytes, canary, "step terminals");

    puf_close(envs);
    dict_clear(&kwargs);
    check_cuda(cudaFree(observation_storage),
        "cudaFree guarded observations");
    check_cuda(cudaFree(reward_storage), "cudaFree guarded rewards");
    check_cuda(cudaFree(terminal_storage), "cudaFree guarded terminals");
    check_cuda(cudaFree(actions), "cudaFree guarded actions");
}

static void test_io_canaries_and_observation_alignment() {
#if AFFINE_LOCK_GPU_SHARED_OBS
    constexpr int environments_per_block = AFFINE_LOCK_GPU_SHARED_BLOCK;
#else
    constexpr int environments_per_block =
        AFFINE_LOCK_GPU_BLOCK / AFFINE_LOCK_GPU_LANES;
#endif
    run_io_canary_case(environments_per_block);
    run_io_canary_case(environments_per_block + 1);
}

static void test_nondefault_stream_and_cuda_graph() {
    constexpr int n = 4099;
    obs_t* observations = nullptr;
    float* actions = nullptr;
    float* rewards = nullptr;
    float* terminals = nullptr;
    check_cuda(cudaMalloc(&observations, (size_t)n * OBS_SIZE * sizeof(obs_t)),
        "cudaMalloc graph observations");
    check_cuda(cudaMalloc(&actions, (size_t)n * sizeof(float)),
        "cudaMalloc graph actions");
    check_cuda(cudaMalloc(&rewards, (size_t)n * sizeof(float)),
        "cudaMalloc graph rewards");
    check_cuda(cudaMalloc(&terminals, (size_t)n * sizeof(float)),
        "cudaMalloc graph terminals");
    check_cuda(cudaMemset(actions, 0, (size_t)n * sizeof(float)),
        "clear graph actions");

    Dict kwargs;
    fill_kwargs(&kwargs, 123, 3, PERF_WEIGHTING_LINEAR);
    Env* envs = puf_vec_create(n, &kwargs,
        observations, actions, rewards, terminals);
    cudaStream_t stream;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create graph stream");
    puf_bind_stream(stream);
    puf_reset(envs);
    check_cuda(cudaStreamSynchronize(stream), "graph reset");

    VisibleTargetTable table = {};
    EXPECT_EQ(visible_targets_load(VISIBLE_TARGET_TABLE_PATH,
        VISIBLE_TARGET_8ACTION_V1_HASH, &table), 0);
    std::vector<OracleState> oracle(n);
    unsigned int running_seed = 123;
    for (int i = 0; i < n; i++) {
        oracle[i].rng = (uint32_t)rand_r(&running_seed);
        oracle[i].curriculum_depth = 2;
        oracle_reset_state(&oracle[i], &table, 3);
    }

    GpuAffineLockState* actual_states = nullptr;
    Env* actual_envs = nullptr;
    obs_t* actual_obs = nullptr;
    float* actual_rewards = nullptr;
    float* actual_terminals = nullptr;
    check_cuda(cudaMallocHost((void**)&actual_states,
        (size_t)n * sizeof(GpuAffineLockState)),
        "cudaMallocHost graph states");
    check_cuda(cudaMallocHost((void**)&actual_envs,
        (size_t)n * sizeof(Env)), "cudaMallocHost graph envs");
    check_cuda(cudaMallocHost((void**)&actual_obs,
        (size_t)n * OBS_SIZE * sizeof(obs_t)),
        "cudaMallocHost graph observations");
    check_cuda(cudaMallocHost((void**)&actual_rewards,
        (size_t)n * sizeof(float)), "cudaMallocHost graph rewards");
    check_cuda(cudaMallocHost((void**)&actual_terminals,
        (size_t)n * sizeof(float)), "cudaMallocHost graph terminals");

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "begin graph capture");
    for (int i = 0; i < 3; i++) {
        puf_step(envs);
    }
    check_cuda(cudaStreamEndCapture(stream, &graph), "end graph capture");
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        "instantiate graph");
    check_cuda(cudaGraphLaunch(graph_exec, stream), "launch graph first");
    check_cuda(cudaGraphLaunch(graph_exec, stream), "launch graph second");

    check_cuda(cudaMemcpyAsync(actual_states, g_gpu.states,
        (size_t)n * sizeof(GpuAffineLockState), cudaMemcpyDeviceToHost, stream),
        "queue graph states D2H");
    check_cuda(cudaMemcpyAsync(actual_envs, envs,
        (size_t)n * sizeof(Env), cudaMemcpyDeviceToHost, stream),
        "queue graph logs D2H");
    check_cuda(cudaMemcpyAsync(actual_obs, observations,
        (size_t)n * OBS_SIZE * sizeof(obs_t), cudaMemcpyDeviceToHost, stream),
        "queue graph observations D2H");
    check_cuda(cudaMemcpyAsync(actual_rewards, rewards,
        (size_t)n * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "queue graph rewards D2H");
    check_cuda(cudaMemcpyAsync(actual_terminals, terminals,
        (size_t)n * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "queue graph terminals D2H");
    check_cuda(cudaStreamSynchronize(stream), "synchronize graph and D2H");

    std::vector<float> expected_rewards(n), expected_terminals(n);
    for (int step = 0; step < 6; step++) {
        for (int i = 0; i < n; i++) {
            oracle_step(&oracle[i], 0.0f, &table,
                2, 16, 3, PERF_WEIGHTING_LINEAR,
                &expected_rewards[i], &expected_terminals[i]);
        }
    }
    for (int i = 0; i < n; i++) {
        expect_state_equal(actual_states[i], oracle[i]);
        expect_log_equal(actual_envs[i].log, oracle[i].log);
        expect_observation_equal(
            &actual_obs[(size_t)i * OBS_SIZE], oracle[i]);
        EXPECT_EQ(float_bits(actual_rewards[i]),
            float_bits(expected_rewards[i]));
        EXPECT_EQ(float_bits(actual_terminals[i]),
            float_bits(expected_terminals[i]));
    }

    check_cuda(cudaGraphExecDestroy(graph_exec), "destroy graph exec");
    check_cuda(cudaGraphDestroy(graph), "destroy graph");
    check_cuda(cudaFreeHost(actual_states), "cudaFreeHost graph states");
    check_cuda(cudaFreeHost(actual_envs), "cudaFreeHost graph envs");
    check_cuda(cudaFreeHost(actual_obs), "cudaFreeHost graph observations");
    check_cuda(cudaFreeHost(actual_rewards), "cudaFreeHost graph rewards");
    check_cuda(cudaFreeHost(actual_terminals), "cudaFreeHost graph terminals");
    check_cuda(cudaStreamDestroy(stream), "destroy graph stream");
    puf_bind_stream(nullptr);
    puf_close(envs);
    dict_clear(&kwargs);
    visible_targets_free(&table);
    check_cuda(cudaFree(observations), "cudaFree graph observations");
    check_cuda(cudaFree(actions), "cudaFree graph actions");
    check_cuda(cudaFree(rewards), "cudaFree graph rewards");
    check_cuda(cudaFree(terminals), "cudaFree graph terminals");
}

int main() {
    test_deterministic_reset_and_step_parity();
    test_reset_rejection_sampling();
    test_exhaustive_action_transforms();
    test_action_boundaries();
    test_solution_curriculum_and_logs();
    test_linear_solve_scoring();
    test_io_canaries_and_observation_alignment();
    test_nondefault_stream_and_cuda_graph();
    test_puf_log_exports_cpu_contract();
    std::puts("affine_lock CUDA tests passed");
    return 0;
}
