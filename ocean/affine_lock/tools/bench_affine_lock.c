#define _POSIX_C_SOURCE 199309L
#define AFFINE_LOCK_NO_RENDER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "affine_lock.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int parse_int_arg(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return fallback;
}

static void bench_finish_episode(
        AffineLock* env,
        int solved,
        int invalid,
        int reward_state_mismatch,
        int write_log) {
    affine_lock_trace_episode_end(env, solved, invalid, reward_state_mismatch);
    if (write_log) {
        affine_lock_add_log(env, solved, invalid, reward_state_mismatch);
    }
    affine_lock_advance_curriculum(env, solved);
    affine_lock_reset_state(env);
}

static void bench_step_config(
        AffineLock* env,
        int write_observations,
        int write_log) {
    AffineLockShared* shared = env->shared;
    int action = (int)env->actions[0];
    float reward = AFFINE_LOCK_STEP_REWARD;
    int terminal = 0;
    int solved = 0;
    int invalid = 0;
    int reward_state_mismatch = 0;
    int step_before = env->step_count;
    uint32_t state_before = env->state;

    env->terminals[0] = 0.0f;
    env->step_count += 1;

    if (action < 0 || action >= AFFINE_LOCK_NUM_ACTIONS) {
        reward = -1.0f;
        terminal = 1;
        invalid = 1;
    } else {
        env->state = affine_lock_apply_action(shared, env->state, action);
        if (env->state == env->target) {
            reward = 1.0f;
            terminal = 1;
            solved = 1;
        } else if (env->step_count >= env->max_steps) {
            reward = -1.0f;
            terminal = 1;
        }
    }
    reward_state_mismatch = (reward == 1.0f && env->state != env->target);

    env->rewards[0] = reward;
    env->episode_return += reward;
    env->last_reward = reward;
    affine_lock_trace_policy_step(
        env, step_before, action, state_before, env->state,
        reward, terminal, solved, invalid, reward_state_mismatch);

    if (terminal) {
        env->terminals[0] = 1.0f;
        env->last_terminal = 1;
        env->last_solved = solved;
        bench_finish_episode(env, solved, invalid, reward_state_mismatch, write_log);
    }

    if (write_observations) {
        compute_observations(env);
    }
}

static void bench_step_full(AffineLock* env) {
    c_step(env);
}

static void bench_step_no_obs(AffineLock* env) {
    bench_step_config(env, 0, 1);
}

static void bench_step_no_obs_no_log(AffineLock* env) {
    bench_step_config(env, 0, 0);
}

typedef void (*BenchStepFn)(AffineLock* env);

static double bench_run_steps(
        const char* label,
        AffineLock* envs,
        int num_envs,
        int iters,
        int report,
        BenchStepFn step_fn,
        long long* terminal_count_out) {
    long long terminal_count = 0;
    double start = now_seconds();
    for (int iter = 0; iter < iters; iter++) {
#pragma omp parallel for schedule(static) reduction(+:terminal_count)
        for (int i = 0; i < num_envs; i++) {
            envs[i].actions[0] = (float)((i + iter) & 7);
            step_fn(&envs[i]);
            terminal_count += envs[i].terminals[0] != 0.0f;
        }
    }
    double elapsed = now_seconds() - start;
    if (terminal_count_out != NULL) {
        *terminal_count_out = terminal_count;
    }
    if (report) {
        double total_steps = (double)num_envs * (double)iters;
        printf("%s_sps=%.0f elapsed=%.6f terminals=%lld terminal_rate=%.4f\n",
            label, total_steps / elapsed, elapsed, terminal_count,
            (double)terminal_count / total_steps);
    }
    return elapsed;
}

static double bench_run_resets(
        const char* label,
        AffineLock* envs,
        int num_envs,
        int iters,
        int depth,
        int report) {
    double start = now_seconds();
    for (int iter = 0; iter < iters; iter++) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_envs; i++) {
            envs[i].curriculum_depth = depth;
            c_reset(&envs[i]);
        }
    }
    double elapsed = now_seconds() - start;
    if (report) {
        double total_resets = (double)num_envs * (double)iters;
        printf("%s_reset_rate=%.0f elapsed=%.6f depth=%d\n",
            label, total_resets / elapsed, elapsed, depth);
    }
    return elapsed;
}

static void bench_reset_envs_to_depth(AffineLock* envs, int num_envs, int depth) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        envs[i].curriculum_depth = depth;
        c_reset(&envs[i]);
    }
}

static void bench_prepare_nonterminal(AffineLock* envs, int num_envs) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs; i++) {
        envs[i].target = UINT32_MAX;
        envs[i].max_steps = 1 << 28;
        envs[i].step_count = 0;
        envs[i].terminals[0] = 0.0f;
    }
}

static void bench_validate_shell_distances(
        AffineLockShared* shared,
        int samples,
        unsigned int seed) {
    AffineLock env;
    memset(&env, 0, sizeof(env));
    env.shared = shared;
    env.rng = seed;

    const int depths[] = {1, 2, 4, 8, 16};
    for (int depth_idx = 0; depth_idx < 5; depth_idx++) {
        int depth = depths[depth_idx];
        if (depth > AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE) {
            continue;
        }
        uint32_t shell_start =
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[depth];
        uint32_t shell_end =
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[depth + 1];
        int mismatches = 0;
        int shorter = 0;
        int longer_or_unreachable = 0;
        int checked = 0;
        for (int i = 0; i < samples; i++) {
            uint32_t state = affine_lock_random_state_bits(&env, shared);
            uint32_t offset = (uint32_t)affine_lock_random_bounded(
                &env, (int)(shell_end - shell_start));
            const AffineLockPrecomputedTransform* transform =
                &AFFINE_LOCK_PRECOMPUTED_TRANSFORMS[shell_start + offset];
            uint32_t target =
                affine_lock_apply_precomputed_transform(shared, state, transform);
            int exact = affine_lock_shortest_distance(shared, state, target);
            checked += 1;
            if (exact != depth) {
                mismatches += 1;
                if (exact >= 0 && exact < depth) {
                    shorter += 1;
                } else {
                    longer_or_unreachable += 1;
                }
            }
        }
        printf("shell_depth=%d checked=%d mismatches=%d shorter=%d longer_or_unreachable=%d mismatch_rate=%.6f\n",
            depth, checked, mismatches, shorter, longer_or_unreachable,
            checked > 0 ? (double)mismatches / (double)checked : 0.0);
    }
}

int main(int argc, char** argv) {
    int num_envs = parse_int_arg(argc, argv, "--envs", 196608);
    int iters = parse_int_arg(argc, argv, "--iters", 128);
    int warmups = parse_int_arg(argc, argv, "--warmups", 8);
    int init_mode = parse_int_arg(argc, argv, "--init-mode", 4);
    int start_depth = parse_int_arg(argc, argv, "--start-depth", 2);
    int max_depth = parse_int_arg(argc, argv, "--max-depth", 16);
    int depth_multiplier = parse_int_arg(argc, argv, "--depth-multiplier", 2);
    int step_grace = parse_int_arg(argc, argv, "--step-grace", 0);
    int threads = parse_int_arg(argc, argv, "--threads", 1);
    int breakdown = parse_int_arg(argc, argv, "--breakdown", 0);
    int validate_shells = parse_int_arg(argc, argv, "--validate-shells", 0);

#ifdef _OPENMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
#else
    (void)threads;
#endif

    AffineLockShared* shared =
        (AffineLockShared*)calloc(1, sizeof(AffineLockShared));
    if (shared == NULL || affine_lock_init_shared(
            shared, 16, start_depth, max_depth, depth_multiplier, step_grace) != 0) {
        fprintf(stderr, "failed to initialize affine_lock shared state\n");
        free(shared);
        return 1;
    }
    if (affine_lock_configure_initialization(shared, init_mode) != 0) {
        affine_lock_free_shared(shared);
        free(shared);
        return 1;
    }

    if (validate_shells > 0) {
        bench_validate_shell_distances(shared, validate_shells, 1u);
        affine_lock_free_shared(shared);
        free(shared);
        return 0;
    }

    AffineLock* envs = (AffineLock*)calloc((size_t)num_envs, sizeof(AffineLock));
    float* observations = (float*)calloc(
        (size_t)num_envs * AFFINE_LOCK_OBS_SIZE, sizeof(float));
    float* actions = (float*)calloc(
        (size_t)num_envs * AFFINE_LOCK_NUM_ATNS, sizeof(float));
    float* rewards = (float*)calloc((size_t)num_envs, sizeof(float));
    float* terminals = (float*)calloc((size_t)num_envs, sizeof(float));
    if (envs == NULL || observations == NULL || actions == NULL ||
            rewards == NULL || terminals == NULL) {
        fprintf(stderr, "failed to allocate benchmark buffers\n");
        free(envs);
        free(observations);
        free(actions);
        free(rewards);
        free(terminals);
        affine_lock_free_shared(shared);
        free(shared);
        return 1;
    }

    for (int i = 0; i < num_envs; i++) {
        AffineLock* env = &envs[i];
        affine_lock_init_env(env, shared, (unsigned int)(i + 1), i);
        env->observations = &observations[(size_t)i * AFFINE_LOCK_OBS_SIZE];
        env->actions = &actions[(size_t)i * AFFINE_LOCK_NUM_ATNS];
        env->rewards = &rewards[i];
        env->terminals = &terminals[i];
        c_reset(env);
    }

    for (int iter = 0; iter < warmups; iter++) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < num_envs; i++) {
            envs[i].actions[0] = (float)((i + iter) & 7);
            c_step(&envs[i]);
        }
    }

    long long terminal_count = 0;
    double elapsed = bench_run_steps(
        "step", envs, num_envs, iters, 0, bench_step_full, &terminal_count);
    double total_steps = (double)num_envs * (double)iters;

    double reset_elapsed =
        bench_run_resets("reset", envs, num_envs, iters, start_depth, 0);

    printf("envs=%d iters=%d init_mode=%d threads=%d\n",
        num_envs, iters, init_mode, threads);
    printf("step_sps=%.0f elapsed=%.6f terminals=%lld terminal_rate=%.4f\n",
        total_steps / elapsed, elapsed, terminal_count,
        (double)terminal_count / total_steps);
    printf("reset_rate=%.0f reset_elapsed=%.6f\n",
        total_steps / reset_elapsed, reset_elapsed);

    if (breakdown) {
        printf("breakdown=1\n");

        bench_reset_envs_to_depth(envs, num_envs, start_depth);
        bench_run_steps(
            "normal_full", envs, num_envs, iters,
            1, bench_step_full, NULL);

        bench_reset_envs_to_depth(envs, num_envs, start_depth);
        bench_run_steps(
            "normal_no_obs", envs, num_envs, iters,
            1, bench_step_no_obs, NULL);

        bench_reset_envs_to_depth(envs, num_envs, start_depth);
        bench_run_steps(
            "normal_no_obs_no_log", envs, num_envs, iters,
            1, bench_step_no_obs_no_log, NULL);

        bench_reset_envs_to_depth(envs, num_envs, start_depth);
        bench_prepare_nonterminal(envs, num_envs);
        bench_run_steps(
            "nonterminal_full", envs, num_envs, iters,
            1, bench_step_full, NULL);

        bench_reset_envs_to_depth(envs, num_envs, start_depth);
        bench_prepare_nonterminal(envs, num_envs);
        bench_run_steps(
            "nonterminal_no_obs", envs, num_envs, iters,
            1, bench_step_no_obs, NULL);

        const int reset_depths[] = {2, 4, 8, 16};
        for (int i = 0; i < 4; i++) {
            int depth = reset_depths[i];
            if (depth >= start_depth && depth <= max_depth) {
                bench_run_resets(
                    "depth", envs, num_envs, iters, depth, 1);
            }
        }
    }

    for (int i = 0; i < num_envs; i++) {
        c_close(&envs[i]);
    }
    free(envs);
    free(observations);
    free(actions);
    free(rewards);
    free(terminals);
    affine_lock_free_shared(shared);
    free(shared);
    return 0;
}
