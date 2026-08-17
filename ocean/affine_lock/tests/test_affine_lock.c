#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../affine_lock.h"

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: expected true: %s\n", __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

#define EXPECT_EQ_INT(actual, expected) do { \
    int _a = (int)(actual); \
    int _e = (int)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "%s:%d: expected %s == %d, got %d\n", \
            __FILE__, __LINE__, #actual, _e, _a); \
        exit(1); \
    } \
} while (0)

#define EXPECT_EQ_U32(actual, expected) do { \
    uint32_t _a = (uint32_t)(actual); \
    uint32_t _e = (uint32_t)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "%s:%d: expected %s == 0x%x, got 0x%x\n", \
            __FILE__, __LINE__, #actual, _e, _a); \
        exit(1); \
    } \
} while (0)

#define EXPECT_EQ_U64(actual, expected) do { \
    uint64_t _a = (uint64_t)(actual); \
    uint64_t _e = (uint64_t)(expected); \
    if (_a != _e) { \
        fprintf(stderr, "%s:%d: expected %s == 0x%llx, got 0x%llx\n", \
            __FILE__, __LINE__, #actual, \
            (unsigned long long)_e, (unsigned long long)_a); \
        exit(1); \
    } \
} while (0)

#define EXPECT_NE_U32(actual, expected) do { \
    uint32_t _a = (uint32_t)(actual); \
    uint32_t _e = (uint32_t)(expected); \
    if (_a == _e) { \
        fprintf(stderr, "%s:%d: expected %s != 0x%x\n", \
            __FILE__, __LINE__, #actual, _e); \
        exit(1); \
    } \
} while (0)

#define EXPECT_NEAR(actual, expected, eps) do { \
    float _a = (float)(actual); \
    float _e = (float)(expected); \
    if (fabsf(_a - _e) > (eps)) { \
        fprintf(stderr, "%s:%d: expected %s ~= %.6f, got %.6f\n", \
            __FILE__, __LINE__, #actual, _e, _a); \
        exit(1); \
    } \
} while (0)

static AffineLockShared make_shared(
        int start_depth, int max_depth,
        int step_grace) {
    AffineLockShared shared;
    memset(&shared, 0, sizeof(shared));
    init_shared(&shared, start_depth, max_depth, step_grace, PERF_WEIGHTING_LINEAR);
    return shared;
}

static void make_env(
        AffineLock* env,
        AffineLockShared* shared,
        unsigned int seed,
        float observations[OBS_SIZE],
        float actions[NUM_ATNS],
        float rewards[1],
        float terminals[1]) {
    memset(env, 0, sizeof(*env));
    memset(observations, 0, OBS_SIZE * sizeof(float));
    actions[0] = 0.0f;
    rewards[0] = 0.0f;
    terminals[0] = 0.0f;
    init_env(env, shared, seed);
    env->agents[0].observations = observations;
    env->agents[0].actions = actions;
    env->agents[0].rewards = rewards;
    env->agents[0].terminals = terminals;
}

static uint32_t bits_from_text(const char* bits) {
    EXPECT_EQ_INT(strlen(bits), BITS);
    uint32_t value = 0u;
    for (int i = 0; i < BITS; i++) {
        EXPECT_TRUE(bits[i] == '0' || bits[i] == '1');
        if (bits[i] == '1') {
            value |= 1u << i;
        }
    }
    return value;
}

static uint32_t test_apply_action(uint32_t state, int action) {
    state &= 0xffffu;
    switch (action) {
        case 0: {
            uint32_t first = state & 1u;
            return ((state >> 1) | (first << 15)) & 0xffffu;
        }
        case 1: {
            uint32_t last = (state >> 15) & 1u;
            return ((state << 1) & 0xffffu) | last;
        }
        case 2:
            return state ^ 0xfe00u;
        case 3:
            return ((state & 0x5555u) << 1) | ((state & 0xaaaau) >> 1);
        case 4:
            return ((state & 0x3333u) << 2) | ((state & 0xccccu) >> 2);
        case 5:
            return ((state & 0x0f0fu) << 4) | ((state & 0xf0f0u) >> 4);
        case 6:
            return test_apply_action(test_apply_action(state, 3), 4);
        case 7:
            return test_apply_action(test_apply_action(state, 6), 5);
        default:
            return state;
    }
}

typedef struct TestBfsStats {
    int reachable_count;
    int distance_histogram[128];
    int farthest_distance;
    int shortest_distance;
} TestBfsStats;

static void compute_test_bfs_stats(
        const AffineLockShared* shared,
        uint32_t start,
        uint32_t target,
        TestBfsStats* stats) {
    memset(stats, 0, sizeof(*stats));
    stats->shortest_distance = -1;

    int num_states = 1 << BITS;
    int* distances = (int*)malloc((size_t)num_states * sizeof(int));
    uint32_t* queue =
        (uint32_t*)malloc((size_t)num_states * sizeof(uint32_t));
    EXPECT_TRUE(distances != NULL);
    EXPECT_TRUE(queue != NULL);

    for (int i = 0; i < num_states; i++) {
        distances[i] = -1;
    }

    int head = 0;
    int tail = 0;
    start &= shared->mask;
    target &= shared->mask;
    distances[start] = 0;
    queue[tail++] = start;

    while (head < tail) {
        uint32_t state = queue[head++];
        int distance = distances[state];
        stats->reachable_count += 1;
        if (distance >= 0 && distance < (int)(sizeof(stats->distance_histogram) /
                sizeof(stats->distance_histogram[0]))) {
            stats->distance_histogram[distance] += 1;
        }
        if (distance > stats->farthest_distance) {
            stats->farthest_distance = distance;
        }

        for (int action = 0; action < NUM_ACTIONS; action++) {
            uint32_t next = test_apply_action(state, action) & shared->mask;
            if (distances[next] >= 0) {
                continue;
            }
            distances[next] = distance + 1;
            queue[tail++] = next;
        }
    }

    stats->shortest_distance = distances[target];
    free(distances);
    free(queue);
}

static float expected_solve_credit(const AffineLockShared* shared, int depth);

static void test_log_solve_credit_uses_known_target_distance(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    AffineLock env;
    memset(&env, 0, sizeof(env));
    env.shared = &shared;
    env.scramble_depth = 16;
    env.target_distance = 8;
    env.step_count = 8;

    add_log(&env, 1);

    EXPECT_NEAR(env.log.perf, expected_solve_credit(&shared, 8), 0.0f);
    EXPECT_NEAR(env.log.score, expected_solve_credit(&shared, 8), 0.0f);
    EXPECT_NEAR(env.log.max_depth_solve, 0.0f, 0.0f);
    EXPECT_NEAR(env.log.solve_efficiency, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.target_distance, 8.0f, 0.0f);
    EXPECT_NEAR(env.log.solved_target_distance, 8.0f, 0.0f);
    EXPECT_NEAR(env.log.d6_rate, 0.0f, 0.0f);
    EXPECT_NEAR(env.log.d6_solve_rate, 0.0f, 0.0f);
    EXPECT_NEAR(env.log.d8_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.d8_solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.d16_rate, 0.0f, 0.0f);
    EXPECT_NEAR(env.log.d16_solve_rate, 0.0f, 0.0f);

    free_shared(&shared);
}

static void test_log_solve_credit_uses_quadratic_perf_weighting(void) {
    AffineLockShared shared;
    memset(&shared, 0, sizeof(shared));
    init_shared(&shared, 2, 16, 0, PERF_WEIGHTING_QUADRATIC);
    AffineLock env;
    memset(&env, 0, sizeof(env));
    env.shared = &shared;
    env.scramble_depth = 16;
    env.target_distance = 8;
    env.step_count = 8;

    add_log(&env, 1);

    float linear_ratio = expected_solve_credit(&shared, 8);
    EXPECT_NEAR(env.log.perf, linear_ratio * linear_ratio, 0.0f);
    EXPECT_NEAR(env.log.score, linear_ratio * linear_ratio, 0.0f);

    free_shared(&shared);
}

static void expect_observation_matches(const AffineLock* env) {
    float* obs = env->agents[0].observations;
    for (int bit = 0; bit < BITS; bit++) {
        uint32_t bit_mask = 1u << bit;
        float expected_current = (env->state & bit_mask) ? 1.0f : -1.0f;
        float expected_target = (env->target & bit_mask) ? 1.0f : -1.0f;
        EXPECT_NEAR(obs[bit], expected_current, 0.0f);
        EXPECT_NEAR(obs[BITS + bit], expected_target, 0.0f);
    }

    for (int i = 0; i < TIMER_INDEX; i++) {
        EXPECT_TRUE(obs[i] == -1.0f || obs[i] == 1.0f);
    }

    float expected_timer = env->max_steps > 0 ?
        (float)env->step_count / (float)env->max_steps : 0.0f;
    EXPECT_TRUE(obs[TIMER_INDEX] >= 0.0f);
    EXPECT_TRUE(obs[TIMER_INDEX] <= 1.0f);
    EXPECT_NEAR(obs[TIMER_INDEX], expected_timer, 0.000001f);
}

static int find_non_solving_action(AffineLock* env) {
    for (int action = 0; action < NUM_ACTIONS; action++) {
        uint32_t next = env->shared->next[env->state * NUM_ACTIONS + action];
        if (next != env->target) {
            return action;
        }
    }
    return -1;
}

static float expected_solve_credit(const AffineLockShared* shared, int depth) {
    return (float)depth / (float)shared->max_depth;
}

static uint64_t mix_u64(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t mix_float(uint64_t hash, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return mix_u64(hash, bits);
}

static uint64_t log_snapshot_checksum(uint64_t hash, const Log* log) {
    hash = mix_float(hash, log->perf);
    hash = mix_float(hash, log->score);
    hash = mix_float(hash, log->solve_rate);
    hash = mix_float(hash, log->max_depth_solve);
    hash = mix_float(hash, log->episode_return);
    hash = mix_float(hash, log->episode_length);
    hash = mix_float(hash, log->solve_steps);
    hash = mix_float(hash, log->timeout_rate);
    hash = mix_float(hash, log->solve_efficiency);
    hash = mix_float(hash, log->target_distance);
    hash = mix_float(hash, log->solved_target_distance);
    hash = mix_float(hash, log->d6_rate);
    hash = mix_float(hash, log->d6_solve_rate);
    hash = mix_float(hash, log->d8_rate);
    hash = mix_float(hash, log->d8_solve_rate);
    hash = mix_float(hash, log->d16_rate);
    hash = mix_float(hash, log->d16_solve_rate);
    hash = mix_float(hash, log->n);
    return hash;
}

static uint64_t reset_snapshot_checksum(const AffineLock* env) {
    uint64_t hash = 1469598103934665603ull;
    hash = mix_u64(hash, env->state);
    hash = mix_u64(hash, env->target);
    hash = mix_u64(hash, (uint64_t)env->step_count);
    hash = mix_u64(hash, (uint64_t)env->max_steps);
    hash = mix_u64(hash, (uint64_t)env->scramble_depth);
    hash = mix_u64(hash, (uint64_t)env->curriculum_depth);
    hash = mix_u64(hash, (uint64_t)env->solution_length);
    hash = mix_u64(hash, (uint64_t)(env->target_distance + 1));
    hash = mix_float(hash, env->agents[0].rewards[0]);
    hash = mix_float(hash, env->agents[0].terminals[0]);
    float* obs = env->agents[0].observations;
    for (int i = 0; i < OBS_SIZE; i++) {
        hash = mix_float(hash, obs[i]);
    }
    hash = log_snapshot_checksum(hash, &env->log);
    for (int i = 0; i < MAX_SOLUTION_DEPTH; i++) {
        hash = mix_u64(hash, (uint64_t)(env->solution_actions[i] + 1));
    }
    return hash;
}

static void expect_env_snapshots_equal(
        const AffineLock* a,
        const AffineLock* b,
        const float obs_a[OBS_SIZE],
        const float obs_b[OBS_SIZE]) {
    EXPECT_EQ_U64(reset_snapshot_checksum(a), reset_snapshot_checksum(b));
    EXPECT_TRUE(memcmp(obs_a, obs_b, OBS_SIZE * sizeof(float)) == 0);
    EXPECT_EQ_U32(a->state, b->state);
    EXPECT_EQ_U32(a->target, b->target);
    EXPECT_EQ_INT(a->scramble_depth, b->scramble_depth);
    EXPECT_EQ_INT(a->max_steps, b->max_steps);
    EXPECT_EQ_INT(a->solution_length, b->solution_length);
    EXPECT_TRUE(memcmp(a->solution_actions, b->solution_actions,
        sizeof(a->solution_actions)) == 0);
}

static void expect_solution_reaches_target(
        const AffineLockShared* shared,
        const AffineLock* env) {
    TestBfsStats stats;
    compute_test_bfs_stats(shared, env->state, env->target, &stats);
    EXPECT_EQ_INT(env->target_distance, stats.shortest_distance);
    EXPECT_EQ_INT(env->solution_length, stats.shortest_distance);
    EXPECT_EQ_INT(env->max_steps, stats.shortest_distance + shared->step_grace);
    EXPECT_TRUE(stats.reachable_count > 0);
    EXPECT_TRUE(stats.farthest_distance >= stats.shortest_distance);
    EXPECT_TRUE(stats.distance_histogram[env->target_distance] > 0);

    uint32_t simulated = env->state;
    for (int i = 0; i < env->solution_length; i++) {
        int action = env->solution_actions[i];
        EXPECT_TRUE(action >= 0 && action < NUM_ACTIONS);
        simulated = test_apply_action(simulated, action) & shared->mask;
    }
    EXPECT_EQ_U32(simulated, env->target);
}

static void solve_with_stored_solution(AffineLock* env) {
    int length = env->solution_length;
    for (int step = 0; step < length; step++) {
        env->agents[0].actions[0] = (float)env->solution_actions[step];
        puf_step(env);
        if (env->agents[0].terminals[0] != 0.0f) {
            return;
        }
    }
}

static void expect_depth_log_delta(
        const Log* before,
        const Log* after,
        int depth,
        int solved) {
    EXPECT_NEAR(after->d6_rate,
        before->d6_rate + (depth == 6 ? 1.0f : 0.0f), 0.0f);
    EXPECT_NEAR(after->d6_solve_rate,
        before->d6_solve_rate + (solved && depth == 6 ? 1.0f : 0.0f), 0.0f);
    EXPECT_NEAR(after->d8_rate,
        before->d8_rate + (depth == 8 ? 1.0f : 0.0f), 0.0f);
    EXPECT_NEAR(after->d8_solve_rate,
        before->d8_solve_rate + (solved && depth == 8 ? 1.0f : 0.0f), 0.0f);
    EXPECT_NEAR(after->d16_rate,
        before->d16_rate + (depth == 16 ? 1.0f : 0.0f), 0.0f);
    EXPECT_NEAR(after->d16_solve_rate,
        before->d16_solve_rate + (solved && depth == 16 ? 1.0f : 0.0f), 0.0f);
}

static void expect_oracle_episode_win(AffineLock* env, int depth) {
    AffineLockShared* shared = env->shared;
    EXPECT_EQ_INT(env->scramble_depth, depth);
    EXPECT_TRUE(env->solution_length > 0);
    expect_solution_reaches_target(shared, env);

    Log before = env->log;
    int target_distance = env->target_distance;
    int solution_length = env->solution_length;
    EXPECT_TRUE(solution_length > 0);
    EXPECT_EQ_INT(env->max_steps, target_distance + shared->step_grace);

    for (int step = 0; step < solution_length; step++) {
        env->agents[0].actions[0] = (float)env->solution_actions[step];
        puf_step(env);
        if (step + 1 < solution_length) {
            EXPECT_NEAR(env->agents[0].rewards[0], STEP_REWARD, 0.0f);
            EXPECT_NEAR(env->agents[0].terminals[0], 0.0f, 0.0f);
            EXPECT_EQ_INT(env->step_count, step + 1);
            expect_observation_matches(env);
        }
    }

    EXPECT_NEAR(env->agents[0].rewards[0], 1.0f, 0.0f);
    EXPECT_NEAR(env->agents[0].terminals[0], 1.0f, 0.0f);
    EXPECT_EQ_INT(env->step_count, 0);
    EXPECT_NEAR(env->log.n, before.n + 1.0f, 0.0f);
    EXPECT_NEAR(env->log.perf,
        before.perf + expected_solve_credit(shared, depth), 0.0f);
    EXPECT_NEAR(env->log.score,
        before.score + expected_solve_credit(shared, depth), 0.0f);
    EXPECT_NEAR(env->log.solve_rate, before.solve_rate + 1.0f, 0.0f);
    EXPECT_NEAR(env->log.timeout_rate, before.timeout_rate, 0.0f);
    EXPECT_NEAR(env->log.episode_length,
        before.episode_length + (float)solution_length, 0.0f);
    EXPECT_NEAR(env->log.solve_steps,
        before.solve_steps + (float)solution_length, 0.0f);
    EXPECT_NEAR(env->log.target_distance,
        before.target_distance + (float)target_distance, 0.0f);
    EXPECT_NEAR(env->log.solved_target_distance,
        before.solved_target_distance + (float)target_distance, 0.0f);
    expect_depth_log_delta(&before, &env->log, depth, 1);

    int next_depth = next_curriculum_depth(shared, depth);
    EXPECT_EQ_INT(env->scramble_depth, next_depth);
    expect_observation_matches(env);
}

static void expect_non_solving_episode_timeout(AffineLock* env, int depth) {
    AffineLockShared* shared = env->shared;
    EXPECT_EQ_INT(env->scramble_depth, depth);
    EXPECT_TRUE(env->solution_length > 0);
    expect_solution_reaches_target(shared, env);

    Log before = env->log;
    int target_distance = env->target_distance;
    int max_steps = env->max_steps;
    EXPECT_TRUE(max_steps > 0);

    for (int step = 0; step < max_steps; step++) {
        int action = find_non_solving_action(env);
        EXPECT_TRUE(action >= 0);
        env->agents[0].actions[0] = (float)action;
        puf_step(env);
        if (step + 1 < max_steps) {
            EXPECT_NEAR(env->agents[0].rewards[0], STEP_REWARD, 0.0f);
            EXPECT_NEAR(env->agents[0].terminals[0], 0.0f, 0.0f);
            EXPECT_EQ_INT(env->step_count, step + 1);
            expect_observation_matches(env);
        }
    }

    EXPECT_NEAR(env->agents[0].rewards[0], -1.0f, 0.0f);
    EXPECT_NEAR(env->agents[0].terminals[0], 1.0f, 0.0f);
    EXPECT_EQ_INT(env->step_count, 0);
    EXPECT_NEAR(env->log.n, before.n + 1.0f, 0.0f);
    EXPECT_NEAR(env->log.perf, before.perf, 0.0f);
    EXPECT_NEAR(env->log.score, before.score, 0.0f);
    EXPECT_NEAR(env->log.solve_rate, before.solve_rate, 0.0f);
    EXPECT_NEAR(env->log.timeout_rate, before.timeout_rate + 1.0f, 0.0f);
    EXPECT_NEAR(env->log.episode_length,
        before.episode_length + (float)max_steps, 0.0f);
    EXPECT_NEAR(env->log.solve_steps, before.solve_steps, 0.0f);
    EXPECT_NEAR(env->log.target_distance,
        before.target_distance + (float)target_distance, 0.0f);
    EXPECT_NEAR(env->log.solved_target_distance,
        before.solved_target_distance, 0.0f);
    expect_depth_log_delta(&before, &env->log, depth, 0);
    EXPECT_EQ_INT(env->scramble_depth, shared->start_depth);
    expect_observation_matches(env);
}

static size_t read_text_file(const char* path, char* buffer, size_t capacity) {
    FILE* file = fopen(path, "r");
    EXPECT_TRUE(file != NULL);
    size_t nread = fread(buffer, 1, capacity - 1, file);
    buffer[nread] = '\0';
    fclose(file);
    return nread;
}

static void test_metadata_contract(void) {
    EXPECT_EQ_INT(BITS, 16);
    EXPECT_EQ_INT(TIMER_INDEX, 32);
    EXPECT_EQ_INT(OBS_SIZE, 33);
    EXPECT_EQ_INT(NUM_ATNS, 1);
    EXPECT_EQ_INT(NUM_ACTIONS, 8);
}

static void test_config_and_binding_metadata_contract(void) {
    char config[16384];
    read_text_file("config/affine_lock.ini", config, sizeof(config));
    EXPECT_TRUE(strstr(config, "[base]") != NULL);
    EXPECT_TRUE(strstr(config, "env_name = affine_lock") != NULL);
    EXPECT_TRUE(strstr(config, "[env]") != NULL);
    EXPECT_TRUE(strstr(config, "start_depth = 2") != NULL);
    EXPECT_TRUE(strstr(config, "max_depth = 16") != NULL);
    EXPECT_TRUE(strstr(config, "[sweep]") != NULL);
    EXPECT_TRUE(strstr(config, "metric = perf") != NULL);
    EXPECT_TRUE(strstr(config, "goal = maximize") != NULL);
    EXPECT_TRUE(strstr(config, "min = 100_000_000") != NULL);
    EXPECT_TRUE(strstr(config, "max = 200_000_000") != NULL);
    EXPECT_TRUE(strstr(config, "[sweep.policy.num_layers]") != NULL);
    EXPECT_TRUE(strstr(config, "max = 131072") != NULL);
    EXPECT_TRUE(strstr(config, "max = 4.0") != NULL);

    char header[65536];
    read_text_file("ocean/affine_lock/affine_lock.h", header, sizeof(header));
    EXPECT_TRUE(strstr(header, "#define OBS_SIZE (TIMER_INDEX + 1)") != NULL);
    EXPECT_TRUE(strstr(header, "#define ACT_SIZES {NUM_ACTIONS}") != NULL);
    EXPECT_TRUE(strstr(header, "#define NUM_ATNS 1") != NULL);
    EXPECT_TRUE(strstr(header, "typedef") != NULL && strstr(header, "obs_t") != NULL);
}

static void test_global_action_examples(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    uint32_t start = bits_from_text("0011011000010111");

    const char* expected[NUM_ACTIONS] = {
        "0110110000101110",
        "1001101100001011",
        "0011011001101000",
        "0011100100101011",
        "1100100101001101",
        "0110001101110001",
        "1100011010001110",
        "0110110011101000",
    };

    for (int action = 0; action < NUM_ACTIONS; action++) {
        uint32_t next = shared.next[start * NUM_ACTIONS + action];
        EXPECT_EQ_U32(next, bits_from_text(expected[action]));
    }

    free_shared(&shared);
}

static void test_actions_round_trip_for_all_states(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    const int inverse_actions[NUM_ACTIONS] = {
        ACTION_SHIFT_RIGHT,
        ACTION_SHIFT_LEFT,
        ACTION_INVERT_RIGHT_7,
        ACTION_SWAP_ADJACENT_BITS,
        ACTION_SWAP_ADJACENT_PAIRS,
        ACTION_SWAP_NIBBLES_EACH_BYTE,
        ACTION_REVERSE_EACH_NIBBLE,
        ACTION_REVERSE_EACH_BYTE,
    };
    EXPECT_EQ_U32(shared.mask, 0xffffu);

    for (int action = 0; action < NUM_ACTIONS; action++) {
        int inverse = inverse_actions[action];
        EXPECT_TRUE(inverse >= 0 && inverse < NUM_ACTIONS);
        EXPECT_EQ_INT(inverse_actions[inverse], action);

        for (uint32_t state = 0; state < (1u << BITS); state++) {
            uint32_t next = shared.next[state * NUM_ACTIONS + action];
            EXPECT_EQ_U32(next & ~shared.mask, 0u);
            uint32_t round_trip = shared.next[next * NUM_ACTIONS + inverse];
            EXPECT_EQ_U32(round_trip, state);
        }
    }

    free_shared(&shared);
}

static void test_reset_randomizes_target_and_current(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 123, observations, actions, rewards, terminals);

    uint32_t first_target = 0;
    uint32_t first_state = 0;
    int target_changed = 0;
    int state_changed = 0;

    for (int i = 0; i < 16; i++) {
        puf_reset(&env);
        EXPECT_EQ_INT(env.scramble_depth, shared.start_depth);
        EXPECT_EQ_INT(env.max_steps, shared.start_depth);
        EXPECT_EQ_U32(env.target & ~shared.mask, 0u);
        EXPECT_EQ_U32(env.state & ~shared.mask, 0u);
        EXPECT_NE_U32(env.state, env.target);
        expect_observation_matches(&env);

        if (i == 0) {
            first_target = env.target;
            first_state = env.state;
        } else {
            if (env.target != first_target) {
                target_changed = 1;
            }
            if (env.state != first_state) {
                state_changed = 1;
            }
        }
    }

    EXPECT_TRUE(target_changed);
    EXPECT_TRUE(state_changed);
    free_shared(&shared);
}

static void test_visible_target_table_initialization_samples_reachable_target(void) {
    AffineLockShared shared = make_shared(8, 16, 0);

    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 777, observations, actions, rewards, terminals);
    puf_reset(&env);

    EXPECT_EQ_INT(env.scramble_depth, shared.start_depth);
    EXPECT_EQ_INT(env.target_distance, shared.start_depth);
    EXPECT_EQ_INT(env.max_steps, env.target_distance);
    EXPECT_EQ_INT(env.solution_length, env.target_distance);
    EXPECT_NE_U32(env.state, env.target);
    expect_solution_reaches_target(&shared, &env);
    expect_observation_matches(&env);

    free_shared(&shared);
}

static void test_visible_target_table_depths_have_expected_distances(void) {
    const int depths[] = {2, 4, 5, 6, 8, 16};
    for (int i = 0; i < 6; i++) {
        int depth = depths[i];
        AffineLockShared shared = make_shared(depth, 16, 0);

        AffineLock env;
        float observations[OBS_SIZE];
        float actions[NUM_ATNS];
        float rewards[1];
        float terminals[1];
        make_env(&env, &shared, (unsigned int)(1900 + depth), observations,
            actions, rewards, terminals);
        puf_reset(&env);

        TestBfsStats stats;
        compute_test_bfs_stats(&shared, env.state, env.target, &stats);
        int expected_distance = depth <= stats.farthest_distance ?
            depth : stats.farthest_distance;
        EXPECT_EQ_INT(env.target_distance, expected_distance);
        EXPECT_EQ_INT(env.solution_length, expected_distance);
        EXPECT_EQ_INT(env.max_steps, expected_distance);
        expect_solution_reaches_target(&shared, &env);

        solve_with_stored_solution(&env);
        EXPECT_NEAR(rewards[0], 1.0f, 0.0f);
        EXPECT_NEAR(terminals[0], 1.0f, 0.0f);

        free_shared(&shared);
    }
}

static void test_visible_target_table_reset_uses_stored_records(void) {
    const int requested_depths[] = {2, 4, 5, 6, 8, 16};
    const int expected_pool_sizes[] = {65536, 65536, 65536, 65536, 65536, 100548};

    for (int depth_index = 0; depth_index < 6; depth_index++) {
        int requested_depth = requested_depths[depth_index];
        AffineLockShared shared = make_shared(requested_depth, 16, 0);
        const VisibleTargetDepth* table_depth =
            visible_target_depth(&shared, requested_depth);
        EXPECT_TRUE(table_depth != NULL);
        EXPECT_EQ_INT((int)table_depth->stored_count,
            expected_pool_sizes[depth_index]);

        AffineLock env;
        float observations[OBS_SIZE];
        float actions[NUM_ATNS];
        float rewards[1];
        float terminals[1];
        make_env(&env, &shared, (unsigned int)(2500 + requested_depth),
            observations, actions, rewards, terminals);

        for (int reset = 0; reset < 8; reset++) {
            puf_reset(&env);

            EXPECT_EQ_INT(env.target_distance, requested_depth);
            EXPECT_EQ_INT(env.solution_length, requested_depth);
            EXPECT_EQ_INT(env.max_steps, requested_depth);
            TestBfsStats stats;
            compute_test_bfs_stats(&shared, env.state, env.target, &stats);
            EXPECT_EQ_INT(stats.shortest_distance, requested_depth);
            expect_solution_reaches_target(&shared, &env);
        }

        free_shared(&shared);
    }
}

static void test_visible_target_table_matches_independent_bfs_over_repeated_resets(void) {
    const int depths[] = {2, 4, 5, 6, 8, 16};

    for (int depth_index = 0; depth_index < 6; depth_index++) {
        int depth = depths[depth_index];
        AffineLockShared shared = make_shared(depth, 16, 0);

        AffineLock env;
        float observations[OBS_SIZE];
        float actions[NUM_ATNS];
        float rewards[1];
        float terminals[1];
        make_env(&env, &shared, (unsigned int)(1000 + depth),
            observations, actions, rewards, terminals);

        for (int reset = 0; reset < 12; reset++) {
            puf_reset(&env);
            EXPECT_TRUE(env.target_distance > 0);
            EXPECT_TRUE(env.solution_length > 0);
            expect_solution_reaches_target(&shared, &env);
            expect_observation_matches(&env);
        }

        free_shared(&shared);
    }
}

static void test_observation_encoding_is_32_signed_bit_floats_plus_timer(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 7, observations, actions, rewards, terminals);

    env.state = 0xa55au;
    env.target = 0x0f0fu;
    env.step_count = 3;
    env.max_steps = 12;
    compute_observations(&env);

    expect_observation_matches(&env);
    free_shared(&shared);
}

static void test_timer_observation_progresses_and_resets_after_timeout(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 19, observations, actions, rewards, terminals);
    puf_reset(&env);
    EXPECT_NEAR(observations[TIMER_INDEX], 0.0f, 0.0f);

    env.target = 0u;
    env.state = shared.mask;
    env.step_count = 0;
    env.max_steps = 4;
    compute_observations(&env);
    EXPECT_NEAR(observations[TIMER_INDEX], 0.0f, 0.0f);

    actions[0] = 1.0f;
    puf_step(&env);
    EXPECT_NEAR(terminals[0], 0.0f, 0.0f);
    EXPECT_NEAR(observations[TIMER_INDEX], 0.25f, 0.000001f);

    puf_step(&env);
    EXPECT_NEAR(terminals[0], 0.0f, 0.0f);
    EXPECT_NEAR(observations[TIMER_INDEX], 0.5f, 0.000001f);

    puf_step(&env);
    EXPECT_NEAR(terminals[0], 0.0f, 0.0f);
    EXPECT_NEAR(observations[TIMER_INDEX], 0.75f, 0.000001f);

    puf_step(&env);
    EXPECT_NEAR(rewards[0], -1.0f, 0.0f);
    EXPECT_NEAR(terminals[0], 1.0f, 0.0f);
    EXPECT_EQ_INT(env.step_count, 0);
    EXPECT_NEAR(observations[TIMER_INDEX], 0.0f, 0.0f);

    free_shared(&shared);
}

static void test_actions_apply_to_current_state_directly(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 55, observations, actions, rewards, terminals);
    puf_reset(&env);

    uint32_t target = bits_from_text("1111000011110000");
    uint32_t state = bits_from_text("0011011000010111");
    int action = 1;
    uint32_t expected_state = shared.next[state * NUM_ACTIONS + action];
    EXPECT_NE_U32(expected_state, target);

    env.target = target;
    env.state = state;
    env.step_count = 0;
    env.max_steps = 16;
    env.agents[0].actions[0] = (float)action;
    puf_step(&env);

    EXPECT_NEAR(rewards[0], -0.01f, 0.0f);
    EXPECT_NEAR(terminals[0], 0.0f, 0.0f);
    EXPECT_EQ_U32(env.target, target);
    EXPECT_EQ_U32(env.state, expected_state);

    free_shared(&shared);
}

static void test_action_float_validation_rejects_out_of_range_values(void) {
    AffineLockShared shared = make_shared(2, 16, 0);
    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 57, observations, actions, rewards, terminals);

    const float invalid_actions[] = {
        -1.0f,
        8.0f,
        NAN,
        INFINITY,
        -INFINITY,
    };
    int count = (int)(sizeof(invalid_actions) / sizeof(invalid_actions[0]));
    for (int i = 0; i < count; i++) {
        puf_reset(&env);
        float prev_timeout = env.log.timeout_rate;
        float prev_n = env.log.n;

        actions[0] = invalid_actions[i];
        puf_step(&env);

        EXPECT_NEAR(rewards[0], -1.0f, 0.0f);
        EXPECT_NEAR(terminals[0], 1.0f, 0.0f);
        EXPECT_NEAR(env.log.timeout_rate, prev_timeout + 1.0f, 0.0f);
        EXPECT_NEAR(env.log.n, prev_n + 1.0f, 0.0f);
        EXPECT_EQ_INT(env.step_count, 0);
    }

    free_shared(&shared);
}

static void test_visible_target_table_curriculum_and_logging(void) {
    AffineLockShared shared = make_shared(2, 16, 0);

    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 42, observations, actions, rewards, terminals);
    puf_reset(&env);

    const int expected_depths[] = {2, 4, 5, 6, 8, 16};
    for (int episode = 0; episode < 6; episode++) {
        int depth = expected_depths[episode];
        EXPECT_EQ_INT(env.scramble_depth, depth);
        expect_solution_reaches_target(&shared, &env);

        float prev_n = env.log.n;
        float prev_perf = env.log.perf;
        float prev_max_depth_solve = env.log.max_depth_solve;
        float prev_target_distance = env.log.target_distance;
        float prev_solved_target_distance = env.log.solved_target_distance;
        float prev_depth_6 = env.log.d6_rate;
        float prev_depth_6_solve = env.log.d6_solve_rate;
        float prev_depth_8 = env.log.d8_rate;
        float prev_depth_8_solve = env.log.d8_solve_rate;
        float prev_depth_16 = env.log.d16_rate;
        float prev_depth_16_solve = env.log.d16_solve_rate;
        int target_distance = env.target_distance;
        int metric_depth = target_distance > 0 ? target_distance : depth;

        solve_with_stored_solution(&env);
        EXPECT_NEAR(rewards[0], 1.0f, 0.0f);
        EXPECT_NEAR(terminals[0], 1.0f, 0.0f);
        EXPECT_NEAR(env.log.n, prev_n + 1.0f, 0.0f);
        EXPECT_NEAR(env.log.perf,
            prev_perf + expected_solve_credit(&shared, metric_depth), 0.0f);
        EXPECT_NEAR(env.log.max_depth_solve,
            prev_max_depth_solve + (metric_depth == shared.max_depth ? 1.0f : 0.0f),
            0.0f);
        EXPECT_NEAR(env.log.target_distance,
            prev_target_distance + (float)target_distance, 0.0f);
        EXPECT_NEAR(env.log.solved_target_distance,
            prev_solved_target_distance + (float)target_distance, 0.0f);
        EXPECT_NEAR(env.log.d6_rate,
            prev_depth_6 + (metric_depth == 6 ? 1.0f : 0.0f), 0.0f);
        EXPECT_NEAR(env.log.d6_solve_rate,
            prev_depth_6_solve + (metric_depth == 6 ? 1.0f : 0.0f), 0.0f);
        EXPECT_NEAR(env.log.d8_rate,
            prev_depth_8 + (metric_depth == 8 ? 1.0f : 0.0f), 0.0f);
        EXPECT_NEAR(env.log.d8_solve_rate,
            prev_depth_8_solve + (metric_depth == 8 ? 1.0f : 0.0f), 0.0f);
        EXPECT_NEAR(env.log.d16_rate,
            prev_depth_16 + (metric_depth == 16 ? 1.0f : 0.0f), 0.0f);
        EXPECT_NEAR(env.log.d16_solve_rate,
            prev_depth_16_solve + (metric_depth == 16 ? 1.0f : 0.0f), 0.0f);

        int next_depth = episode < 5 ? expected_depths[episode + 1] : 16;
        EXPECT_EQ_INT(env.scramble_depth, next_depth);
    }

    float prev_n = env.log.n;
    float prev_perf = env.log.perf;
    float prev_max_depth_solve = env.log.max_depth_solve;
    float prev_timeout = env.log.timeout_rate;
    EXPECT_EQ_INT(env.scramble_depth, shared.max_depth);
    actions[0] = 999.0f;
    puf_step(&env);
    EXPECT_NEAR(rewards[0], -1.0f, 0.0f);
    EXPECT_NEAR(terminals[0], 1.0f, 0.0f);
    EXPECT_NEAR(env.log.n, prev_n + 1.0f, 0.0f);
    EXPECT_NEAR(env.log.perf, prev_perf, 0.0f);
    EXPECT_NEAR(env.log.max_depth_solve, prev_max_depth_solve, 0.0f);
    EXPECT_NEAR(env.log.timeout_rate, prev_timeout + 1.0f, 0.0f);
    EXPECT_EQ_INT(env.scramble_depth, shared.start_depth);

    free_shared(&shared);
}

static void test_visible_target_table_oracle_wins_all_curriculum_depths_end_to_end(void) {
    AffineLockShared shared = make_shared(2, 16, 0);

    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 4242, observations, actions, rewards, terminals);
    puf_reset(&env);

    const int depths[] = {2, 4, 5, 6, 8, 16};
    for (int i = 0; i < 6; i++) {
        expect_oracle_episode_win(&env, depths[i]);
    }

    EXPECT_EQ_INT(env.scramble_depth, shared.max_depth);
    EXPECT_NEAR(env.log.n, 6.0f, 0.0f);
    EXPECT_NEAR(env.log.d6_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.d6_solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.d8_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.d8_solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.d16_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.d16_solve_rate, 1.0f, 0.0f);
    EXPECT_NEAR(env.log.timeout_rate, 0.0f, 0.0f);

    free_shared(&shared);
}

static void test_visible_target_table_timeouts_at_all_curriculum_depths_end_to_end(void) {
    const int loss_depths[] = {2, 4, 5, 6, 8, 16};

    for (int i = 0; i < 6; i++) {
        int loss_depth = loss_depths[i];
        AffineLockShared shared = make_shared(2, 16, 0);

        AffineLock env;
        float observations[OBS_SIZE];
        float actions[NUM_ATNS];
        float rewards[1];
        float terminals[1];
        make_env(&env, &shared, (unsigned int)(5200 + loss_depth),
            observations, actions, rewards, terminals);
        puf_reset(&env);

        while (env.scramble_depth < loss_depth) {
            expect_oracle_episode_win(&env, env.scramble_depth);
        }
        expect_non_solving_episode_timeout(&env, loss_depth);

        EXPECT_EQ_INT(env.scramble_depth, shared.start_depth);
        EXPECT_TRUE(env.log.timeout_rate >= 1.0f);
        EXPECT_TRUE(env.log.solve_rate >= 0.0f);

        free_shared(&shared);
    }
}

static int deterministic_stream_action(int episode, int step) {
    return (episode * 3 + step * 7) % NUM_ACTIONS;
}

static uint64_t run_seed_sequence_checksum(unsigned int seed) {
    AffineLockShared shared = make_shared(2, 16, 0);

    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, seed, observations, actions, rewards, terminals);

    uint64_t checksum = 1469598103934665603ull;
    for (int episode = 0; episode < 16; episode++) {
        puf_reset(&env);
        checksum = mix_u64(checksum, reset_snapshot_checksum(&env));
        int max_steps = env.max_steps;
        for (int step = 0; step < max_steps + 1; step++) {
            int action = deterministic_stream_action(episode, step);
            if (step < env.solution_length) {
                action = env.solution_actions[step];
            }
            actions[0] = (float)action;
            puf_step(&env);
            checksum = mix_u64(checksum, reset_snapshot_checksum(&env));
            if (terminals[0] != 0.0f) {
                break;
            }
        }
    }

    free_shared(&shared);
    return checksum;
}

static void test_deterministic_seed_sequences(void) {
    AffineLockShared shared = make_shared(2, 16, 0);

    AffineLock env_a;
    AffineLock env_b;
    float obs_a[OBS_SIZE], obs_b[OBS_SIZE];
    float atn_a[NUM_ATNS], atn_b[NUM_ATNS];
    float rew_a[1], rew_b[1];
    float term_a[1], term_b[1];
    make_env(&env_a, &shared, 12345, obs_a, atn_a, rew_a, term_a);
    make_env(&env_b, &shared, 12345, obs_b, atn_b, rew_b, term_b);

    for (int episode = 0; episode < 16; episode++) {
        puf_reset(&env_a);
        puf_reset(&env_b);
        expect_env_snapshots_equal(&env_a, &env_b, obs_a, obs_b);
        int max_steps = env_a.max_steps;
        for (int step = 0; step < max_steps + 1; step++) {
            int action = deterministic_stream_action(episode, step);
            if (step < env_a.solution_length) {
                action = env_a.solution_actions[step];
            }
            atn_a[0] = (float)action;
            atn_b[0] = (float)action;
            puf_step(&env_a);
            puf_step(&env_b);
            EXPECT_NEAR(rew_a[0], rew_b[0], 0.0f);
            EXPECT_NEAR(term_a[0], term_b[0], 0.0f);
            expect_env_snapshots_equal(&env_a, &env_b, obs_a, obs_b);
            if (term_a[0] != 0.0f) {
                break;
            }
        }
    }

    free_shared(&shared);

    uint64_t seed_1 = run_seed_sequence_checksum(1);
    uint64_t seed_1_repeat = run_seed_sequence_checksum(1);
    uint64_t seed_2 = run_seed_sequence_checksum(2);
    uint64_t seed_2_repeat = run_seed_sequence_checksum(2);
    EXPECT_EQ_U64(seed_1, seed_1_repeat);
    EXPECT_EQ_U64(seed_2, seed_2_repeat);
    EXPECT_TRUE(seed_1 != seed_2);
}

static uint64_t run_visible_table_seed_42_golden_sequence(void) {
    AffineLockShared shared = make_shared(2, 16, 0);

    AffineLock env;
    float observations[OBS_SIZE];
    float actions[NUM_ATNS];
    float rewards[1];
    float terminals[1];
    make_env(&env, &shared, 42, observations, actions, rewards, terminals);

    uint64_t checksum = 1469598103934665603ull;
    puf_reset(&env);
    checksum = mix_u64(checksum, reset_snapshot_checksum(&env));
    for (int episode = 0; episode < 5; episode++) {
        int length = env.solution_length;
        for (int step = 0; step < length; step++) {
            actions[0] = (float)env.solution_actions[step];
            puf_step(&env);
            checksum = mix_u64(checksum, reset_snapshot_checksum(&env));
            if (terminals[0] != 0.0f) {
                break;
            }
        }
    }
    EXPECT_EQ_INT(env.scramble_depth, 16);
    actions[0] = 999.0f;
    puf_step(&env);
    checksum = mix_u64(checksum, reset_snapshot_checksum(&env));
    EXPECT_EQ_INT(env.scramble_depth, 2);

    free_shared(&shared);
    return checksum;
}

static void test_visible_table_seed_42_golden_checksum(void) {
    uint64_t checksum = run_visible_table_seed_42_golden_sequence();
    EXPECT_EQ_U64(checksum, 0x733eb55fe141e600ull);
}

static void test_deterministic_seed_sequences_and_distinct_env_ids(void) {
    AffineLockShared shared = make_shared(2, 16, 0);

    AffineLock env_a;
    AffineLock env_b;
    float obs_a[OBS_SIZE], obs_b[OBS_SIZE];
    float atn_a[NUM_ATNS], atn_b[NUM_ATNS];
    float rew_a[1], rew_b[1];
    float term_a[1], term_b[1];
    make_env(&env_a, &shared, 12345, obs_a, atn_a, rew_a, term_a);
    make_env(&env_b, &shared, 12345, obs_b, atn_b, rew_b, term_b);

    for (int episode = 0; episode < 8; episode++) {
        puf_reset(&env_a);
        puf_reset(&env_b);
        EXPECT_EQ_U32(env_a.target, env_b.target);
        EXPECT_EQ_U32(env_a.state, env_b.state);
        EXPECT_EQ_INT(env_a.scramble_depth, env_b.scramble_depth);
        EXPECT_EQ_INT(env_a.solution_length, env_b.solution_length);
        EXPECT_TRUE(memcmp(env_a.solution_actions, env_b.solution_actions,
            sizeof(env_a.solution_actions)) == 0);
        EXPECT_TRUE(memcmp(obs_a, obs_b, sizeof(obs_a)) == 0);

        solve_with_stored_solution(&env_a);
        solve_with_stored_solution(&env_b);
        EXPECT_EQ_U32(env_a.target, env_b.target);
        EXPECT_EQ_U32(env_a.state, env_b.state);
        EXPECT_NEAR(rew_a[0], rew_b[0], 0.0f);
        EXPECT_NEAR(term_a[0], term_b[0], 0.0f);
        EXPECT_TRUE(memcmp(obs_a, obs_b, sizeof(obs_a)) == 0);
    }

    AffineLock env_1;
    AffineLock env_2;
    AffineLock env_1_repeat;
    AffineLock env_2_repeat;
    float obs_1[OBS_SIZE], obs_2[OBS_SIZE];
    float obs_1r[OBS_SIZE], obs_2r[OBS_SIZE];
    float atn_1[NUM_ATNS], atn_2[NUM_ATNS];
    float atn_1r[NUM_ATNS], atn_2r[NUM_ATNS];
    float rew_1[1], rew_2[1], rew_1r[1], rew_2r[1];
    float term_1[1], term_2[1], term_1r[1], term_2r[1];
    make_env(&env_1, &shared, 1, obs_1, atn_1, rew_1, term_1);
    make_env(&env_2, &shared, 2, obs_2, atn_2, rew_2, term_2);
    make_env(&env_1_repeat, &shared, 1, obs_1r, atn_1r, rew_1r, term_1r);
    make_env(&env_2_repeat, &shared, 2, obs_2r, atn_2r, rew_2r, term_2r);

    int differs = 0;
    for (int i = 0; i < 8; i++) {
        puf_reset(&env_1);
        puf_reset(&env_2);
        puf_reset(&env_1_repeat);
        puf_reset(&env_2_repeat);

        EXPECT_EQ_U32(env_1.target, env_1_repeat.target);
        EXPECT_EQ_U32(env_1.state, env_1_repeat.state);
        EXPECT_EQ_U32(env_2.target, env_2_repeat.target);
        EXPECT_EQ_U32(env_2.state, env_2_repeat.state);
        EXPECT_TRUE(memcmp(env_1.solution_actions, env_1_repeat.solution_actions,
            sizeof(env_1.solution_actions)) == 0);
        EXPECT_TRUE(memcmp(env_2.solution_actions, env_2_repeat.solution_actions,
            sizeof(env_2.solution_actions)) == 0);

        if (env_1.target != env_2.target || env_1.state != env_2.state ||
                memcmp(env_1.solution_actions, env_2.solution_actions,
                    sizeof(env_1.solution_actions)) != 0) {
            differs = 1;
        }
    }
    EXPECT_TRUE(differs);

    free_shared(&shared);
}

int main(void) {
    test_metadata_contract();
    test_config_and_binding_metadata_contract();
    test_global_action_examples();
    test_actions_round_trip_for_all_states();
    test_reset_randomizes_target_and_current();
    test_visible_target_table_initialization_samples_reachable_target();
    test_visible_target_table_depths_have_expected_distances();
    test_visible_target_table_reset_uses_stored_records();
    test_visible_target_table_matches_independent_bfs_over_repeated_resets();
    test_log_solve_credit_uses_known_target_distance();
    test_log_solve_credit_uses_quadratic_perf_weighting();
    test_observation_encoding_is_32_signed_bit_floats_plus_timer();
    test_timer_observation_progresses_and_resets_after_timeout();
    test_actions_apply_to_current_state_directly();
    test_action_float_validation_rejects_out_of_range_values();
    test_visible_target_table_curriculum_and_logging();
    test_visible_target_table_oracle_wins_all_curriculum_depths_end_to_end();
    test_visible_target_table_timeouts_at_all_curriculum_depths_end_to_end();
    test_deterministic_seed_sequences();
    test_visible_table_seed_42_golden_checksum();
    test_deterministic_seed_sequences_and_distinct_env_ids();
    printf("affine_lock tests passed\n");
    return 0;
}
