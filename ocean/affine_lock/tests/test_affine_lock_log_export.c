#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../affine_lock.h"

#define EXPECT_NEAR(actual, expected, tolerance) do { \
    double _actual = (double)(actual); \
    double _expected = (double)(expected); \
    double _tolerance = (double)(tolerance); \
    if (fabs(_actual - _expected) > _tolerance) { \
        fprintf(stderr, "%s:%d: expected %.9f ~= %.9f\n", \
            __FILE__, __LINE__, _actual, _expected); \
        exit(1); \
    } \
} while (0)

#define EXPECT_EQ_INT(actual, expected) do { \
    int _actual = (int)(actual); \
    int _expected = (int)(expected); \
    if (_actual != _expected) { \
        fprintf(stderr, "%s:%d: expected %d == %d\n", \
            __FILE__, __LINE__, _actual, _expected); \
        exit(1); \
    } \
} while (0)

#define EXPECT_EQ_U32(actual, expected) do { \
    uint32_t _actual = (uint32_t)(actual); \
    uint32_t _expected = (uint32_t)(expected); \
    if (_actual != _expected) { \
        fprintf(stderr, "%s:%d: expected 0x%x == 0x%x\n", \
            __FILE__, __LINE__, _actual, _expected); \
        exit(1); \
    } \
} while (0)

#define EXPECT_NE_U32(actual, expected) do { \
    uint32_t _actual = (uint32_t)(actual); \
    uint32_t _expected = (uint32_t)(expected); \
    if (_actual == _expected) { \
        fprintf(stderr, "%s:%d: expected 0x%x != 0x%x\n", \
            __FILE__, __LINE__, _actual, _expected); \
        exit(1); \
    } \
} while (0)

#define EXPECT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: expected true: %s\n", \
            __FILE__, __LINE__, #cond); \
        exit(1); \
    } \
} while (0)

static double dict_value(Dict* dict, const char* key) {
    return dict_get(dict, key);
}

static int dict_has_key(Dict* dict, const char* key) {
    return dict_find(dict, key) != NULL;
}

static void fill_vec_kwargs(Dict* vec_kwargs, int total_agents) {
    memset(vec_kwargs, 0, sizeof(*vec_kwargs));
    dict_set(vec_kwargs, "total_agents", total_agents);
    dict_set(vec_kwargs, "num_buffers", 1);
}

static void fill_env_kwargs(Dict* env_kwargs, int seed) {
    memset(env_kwargs, 0, sizeof(*env_kwargs));
    dict_set(env_kwargs, "start_depth", 2);
    dict_set(env_kwargs, "max_depth", 16);
    dict_set(env_kwargs, "step_grace", 0);
    dict_set(env_kwargs, "perf_weighting", PERF_WEIGHTING_LINEAR);
    dict_set(env_kwargs, "seed", seed);
}

static Env* make_binding_envs(int seed) {
    Dict vec_kwargs;
    Dict env_kwargs;
    fill_vec_kwargs(&vec_kwargs, 2);
    fill_env_kwargs(&env_kwargs, seed);
    int starts[1] = {0};
    int counts[1] = {0};
    int num_envs = 0;
    Env* envs = my_vec_init(&num_envs, starts, counts, &vec_kwargs, &env_kwargs);
    EXPECT_EQ_INT(num_envs, 2);
    EXPECT_EQ_INT(starts[0], 0);
    EXPECT_EQ_INT(counts[0], 2);
    dict_clear(&vec_kwargs);
    dict_clear(&env_kwargs);
    return envs;
}

static Env* make_binding_env_batch(int seed, int total_agents) {
    Dict vec_kwargs;
    Dict env_kwargs;
    fill_vec_kwargs(&vec_kwargs, total_agents);
    fill_env_kwargs(&env_kwargs, seed);
    int starts[1] = {0};
    int counts[1] = {0};
    int num_envs = 0;
    Env* envs = my_vec_init(&num_envs, starts, counts, &vec_kwargs, &env_kwargs);
    EXPECT_EQ_INT(num_envs, total_agents);
    EXPECT_EQ_INT(starts[0], 0);
    EXPECT_EQ_INT(counts[0], total_agents);
    dict_clear(&vec_kwargs);
    dict_clear(&env_kwargs);
    return envs;
}

static void free_binding_envs(Env* envs) {
    my_vec_close(envs);
    free(envs);
}

static void test_vec_init_mixes_base_seed_and_env_id(void) {
    Env* base = make_binding_envs(123);
    Env* repeat = make_binding_envs(123);
    Env* different_seed = make_binding_envs(124);

    EXPECT_EQ_U32(base[0].rng, repeat[0].rng);
    EXPECT_EQ_U32(base[1].rng, repeat[1].rng);
    EXPECT_NE_U32(base[0].rng, base[1].rng);
    EXPECT_NE_U32(base[0].rng, different_seed[0].rng);

    free_binding_envs(base);
    free_binding_envs(repeat);
    free_binding_envs(different_seed);
}

static uint64_t mix_u64_for_binding_test(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t binding_reset_checksum(const Env* env) {
    uint64_t hash = 1469598103934665603ull;
    hash = mix_u64_for_binding_test(hash, env->state);
    hash = mix_u64_for_binding_test(hash, env->target);
    hash = mix_u64_for_binding_test(hash, (uint64_t)(env->target_distance + 1));
    hash = mix_u64_for_binding_test(hash, (uint64_t)env->solution_length);
    for (int i = 0; i < MAX_SOLUTION_DEPTH; i++) {
        hash = mix_u64_for_binding_test(
            hash, (uint64_t)(env->solution_actions[i] + 1));
    }
    return hash;
}

static void assign_binding_env_buffers(
        Env* envs,
        int total_agents,
        float observations[][OBS_SIZE],
        float actions[],
        float rewards[],
        float terminals[]) {
    memset(observations, 0,
        (size_t)total_agents * OBS_SIZE * sizeof(float));
    memset(actions, 0, (size_t)total_agents * sizeof(float));
    memset(rewards, 0, (size_t)total_agents * sizeof(float));
    memset(terminals, 0, (size_t)total_agents * sizeof(float));
    for (int i = 0; i < total_agents; i++) {
        envs[i].agents[0].observations = observations[i];
        envs[i].agents[0].actions = &actions[i];
        envs[i].agents[0].rewards = &rewards[i];
        envs[i].agents[0].terminals = &terminals[i];
    }
}

static void test_vec_init_visible_targets_repeat_across_runs_and_vary_by_env_id(void) {
    const int total_agents = 64;
    Env* run_a = make_binding_env_batch(42, total_agents);
    Env* run_b = make_binding_env_batch(42, total_agents);

    float obs_a[64][OBS_SIZE];
    float obs_b[64][OBS_SIZE];
    float actions_a[64], actions_b[64];
    float rewards_a[64], rewards_b[64];
    float terminals_a[64], terminals_b[64];
    assign_binding_env_buffers(
        run_a, total_agents, obs_a, actions_a, rewards_a, terminals_a);
    assign_binding_env_buffers(
        run_b, total_agents, obs_b, actions_b, rewards_b, terminals_b);

    int saw_different_puzzle = 0;
    uint64_t first_checksum = 0;
    for (int i = 0; i < total_agents; i++) {
        puf_reset(&run_a[i]);
        puf_reset(&run_b[i]);

        uint64_t checksum_a = binding_reset_checksum(&run_a[i]);
        uint64_t checksum_b = binding_reset_checksum(&run_b[i]);
        EXPECT_EQ_U32(run_a[i].rng, run_b[i].rng);
        EXPECT_EQ_U32(run_a[i].state, run_b[i].state);
        EXPECT_EQ_U32(run_a[i].target, run_b[i].target);
        EXPECT_EQ_INT(run_a[i].target_distance, run_b[i].target_distance);
        EXPECT_EQ_INT(run_a[i].solution_length, run_b[i].solution_length);
        EXPECT_EQ_INT(run_a[i].target_distance, 2);
        EXPECT_EQ_INT(run_a[i].solution_length, 2);
        EXPECT_TRUE(checksum_a == checksum_b);
        EXPECT_TRUE(memcmp(obs_a[i], obs_b[i], sizeof(obs_a[i])) == 0);

        if (i == 0) {
            first_checksum = checksum_a;
        } else if (checksum_a != first_checksum) {
            saw_different_puzzle = 1;
        }
    }
    EXPECT_TRUE(saw_different_puzzle);

    free_binding_envs(run_a);
    free_binding_envs(run_b);
}

static void test_depth_solve_rates_are_conditional_on_depth_attempts(void) {
    Log log = {0};
    log.d6_rate = 0.25f;
    log.d6_solve_rate = 0.125f;
    log.d8_rate = 0.0f;
    log.d8_solve_rate = 0.0f;
    log.d16_rate = 0.125f;
    log.d16_solve_rate = 0.0f;
    log.score = 0.75f;
    log.target_distance = 4.0f;
    log.solved_target_distance = 2.0f;
    log.solve_rate = 0.5f;

    Dict out = {0};
    puf_log(&log, &out);

    EXPECT_EQ_INT(out.size, 15);
    EXPECT_NEAR(dict_value(&out, "score"), 0.75, 0.0);
    EXPECT_TRUE(!dict_has_key(&out, "solve_steps"));
    EXPECT_TRUE(!dict_has_key(&out, "solve_efficiency"));
    EXPECT_TRUE(!dict_has_key(&out, "scramble_unique_states"));
    EXPECT_NEAR(dict_value(&out, "min_win_moves"), 4.0, 0.0);
    EXPECT_NEAR(dict_value(&out, "solved_min_win_moves"), 4.0, 0.0);
    EXPECT_TRUE(!dict_has_key(&out, "d6_rate"));
    EXPECT_NEAR(dict_value(&out, "d6_solve_rate"), 0.5, 0.0);
    EXPECT_TRUE(!dict_has_key(&out, "d8_rate"));
    EXPECT_NEAR(dict_value(&out, "d8_solve_rate"), 0.0, 0.0);
    EXPECT_TRUE(!dict_has_key(&out, "d16_rate"));
    EXPECT_NEAR(dict_value(&out, "d16_solve_rate"), 0.0, 0.0);

    dict_clear(&out);
}

int main(void) {
    test_vec_init_mixes_base_seed_and_env_id();
    test_vec_init_visible_targets_repeat_across_runs_and_vary_by_env_id();
    test_depth_solve_rates_are_conditional_on_depth_attempts();
    return 0;
}
