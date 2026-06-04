#define _POSIX_C_SOURCE 200809L
#define PATHFINDER_NO_RENDER

#include <stdio.h>
#include <time.h>
#include "../pathfinder.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void setup_env(Pathfinder* env, float* obs, float* actions,
        float* rewards, float* terminals) {
    memset(env, 0, sizeof(*env));
    env->observations = obs;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
    env->num_agents = 1;
    env->branch_prob = 0.35f;
    env->loop_prob = 0.10f;
    env->extra_entry_prob = 0.0f;
    env->min_solution_len = 1;
    env->max_solution_len = 4;
    env->max_steps = 128;
    env->rng = 12345;
    init(env);
}

static long parse_long(const char* value, long fallback) {
    char* end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return fallback;
    }
    return parsed;
}

static void bench_resets(Pathfinder* env, long resets, int curriculum_level) {
    env->curriculum_level = curriculum_level;
    double t0 = now_seconds();
    long shortest_sum = 0;
    int min_shortest = PATHFINDER_MAX_SOLUTION_LEN;
    int max_shortest = 0;
    for (long i = 0; i < resets; i++) {
        c_reset(env);
        int sp = env->state.shortest_path_len;
        shortest_sum += sp;
        if (sp < min_shortest) min_shortest = sp;
        if (sp > max_shortest) max_shortest = sp;
    }
    double elapsed = now_seconds() - t0;
    double reset_sps = (double)resets / elapsed;
    double avg_shortest = (double)shortest_sum / (double)resets;
    printf("reset_bench curriculum_level=%d resets=%ld seconds=%.6f reset_sps=%.2f avg_shortest=%.3f min_shortest=%d max_shortest=%d\n",
        curriculum_level, resets, elapsed, reset_sps, avg_shortest, min_shortest, max_shortest);
}

static void bench_steps(Pathfinder* env, long steps) {
    c_reset(env);
    double t0 = now_seconds();
    double reward_sum = 0.0;
    for (long i = 0; i < steps; i++) {
        env->actions[0] = (float)(pathfinder_rand(env) % PATHFINDER_NUM_ACTIONS);
        c_step(env);
        reward_sum += env->rewards[0];
    }
    double elapsed = now_seconds() - t0;
    double step_sps = (double)steps / elapsed;
    printf("step_bench steps=%ld seconds=%.6f step_sps=%.2f episodes=%.0f success=%.6f reward_sum=%.3f curriculum_level=%d curriculum_max_solution_len=%d\n",
        steps, elapsed, step_sps, env->log.n,
        env->log.n > 0.0f ? env->log.success / env->log.n : 0.0f,
        reward_sum, env->curriculum_level, pathfinder_curriculum_max_solution_len(env));
}

static void bench_reset_components(Pathfinder* env, long iters) {
    double t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        memset(&env->state, 0, sizeof(env->state));
    }
    double memset_sec = now_seconds() - t0;

    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        memset(&env->state, 0, sizeof(env->state));
        env->state.agent_row = 0;
        env->state.agent_col = 0;
        pathfinder_generate_maze(env);
    }
    double gen_sec = now_seconds() - t0;

    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        pathfinder_update_observations(env);
    }
    double obs_sec = now_seconds() - t0;

    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        pathfinder_update_action_mask(env);
    }
    double mask_sec = now_seconds() - t0;

    printf("reset_components iters=%ld memset_ns=%.2f gen_plus_memset_ns=%.2f obs_ns=%.2f mask_ns=%.2f\n",
        iters,
        1e9 * memset_sec / (double)iters,
        1e9 * gen_sec / (double)iters,
        1e9 * obs_sec / (double)iters,
        1e9 * mask_sec / (double)iters);

    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        pathfinder_init_walls(&env->state);
    }
    double init_walls_sec = now_seconds() - t0;

    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        pathfinder_choose_goal_at_distance(env, 4);
    }
    double choose_goal_sec = now_seconds() - t0;

    pathfinder_init_walls(&env->state);
    pathfinder_choose_goal_at_distance(env, 4);
    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        pathfinder_init_walls(&env->state);
        env->state.goal_row = 2;
        env->state.goal_col = 2;
        pathfinder_carve_solution(env);
    }
    double carve_sec = now_seconds() - t0;

    pathfinder_init_walls(&env->state);
    env->state.goal_row = 2;
    env->state.goal_col = 2;
    pathfinder_carve_solution(env);
    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        pathfinder_open_random_edges(env);
    }
    double random_edges_sec = now_seconds() - t0;

    printf("generation_components iters=%ld init_walls_ns=%.2f choose_goal_ns=%.2f init_plus_carve_ns=%.2f random_edges_ns=%.2f\n",
        iters,
        1e9 * init_walls_sec / (double)iters,
        1e9 * choose_goal_sec / (double)iters,
        1e9 * carve_sec / (double)iters,
        1e9 * random_edges_sec / (double)iters);
}

static void setup_open_line(Pathfinder* env) {
    State* s = &env->state;
    memset(s, 0, sizeof(*s));
    memset(s->true_walls, 1, sizeof(s->true_walls));
    pathfinder_reset_known(s);
    s->agent_row = 0;
    s->agent_col = 0;
    s->goal_row = 5;
    s->goal_col = 5;
    for (int col = 0; col < PATHFINDER_COLS - 1; col++) {
        pathfinder_open_edge(s, 0, col, 0, col + 1);
    }
    pathfinder_mark_visited(s, 0, 0);
    pathfinder_update_observations(env);
}

static void bench_forced_steps(Pathfinder* env, long iters) {
    setup_open_line(env);
    double t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        if (env->state.agent_col == PATHFINDER_COLS - 1) {
            setup_open_line(env);
        }
        env->actions[0] = PATHFINDER_ACT_EAST;
        c_step(env);
    }
    double move_sec = now_seconds() - t0;

    memset(&env->log, 0, sizeof(env->log));
    setup_open_line(env);
    int east_wall = pathfinder_wall_between(0, 0, 0, 1);
    env->state.true_walls[east_wall] = 1;
    pathfinder_update_observations(env);
    t0 = now_seconds();
    for (long i = 0; i < iters; i++) {
        env->actions[0] = PATHFINDER_ACT_EAST;
        c_step(env);
        if (env->terminals[0] == 0.0f) {
            c_step(env);
        }
        setup_open_line(env);
        env->state.true_walls[east_wall] = 1;
        pathfinder_update_observations(env);
    }
    double known_wall_death_sec = now_seconds() - t0;

    printf("forced_step_components iters=%ld open_move_ns=%.2f known_wall_death_cycle_ns=%.2f\n",
        iters,
        1e9 * move_sec / (double)iters,
        1e9 * known_wall_death_sec / (double)iters);
}

int main(int argc, char** argv) {
    long resets = 200000;
    long steps = 5000000;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--resets") == 0 && i + 1 < argc) {
            resets = parse_long(argv[++i], resets);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = parse_long(argv[++i], steps);
        } else {
            fprintf(stderr, "Usage: %s [--resets N] [--steps N]\n", argv[0]);
            return 1;
        }
    }

    Pathfinder env;
    float obs[PATHFINDER_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[PATHFINDER_NUM_ACTIONS] = {0};
    setup_env(&env, obs, actions, rewards, terminals);
    env.action_mask = action_mask;

    bench_resets(&env, resets, 0);
    bench_resets(&env, resets, PATHFINDER_MAX_SOLUTION_LEN);
    bench_reset_components(&env, resets);
    bench_forced_steps(&env, resets);

    memset(&env.log, 0, sizeof(env.log));
    env.curriculum_level = 0;
    env.curriculum_episodes = 0;
    env.curriculum_window_episodes = 0;
    env.curriculum_window_successes = 0;
    bench_steps(&env, steps);
    c_close(&env);
    return 0;
}
