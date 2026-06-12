#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../bat.h"

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("ASSERT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

#define ASSERT_FLOAT_NEAR(actual, expected, eps) do { \
    float _a = (actual); \
    float _e = (expected); \
    if (fabsf(_a - _e) > (eps)) { \
        printf("ASSERT_FLOAT_NEAR failed at %s:%d: got %.6f expected %.6f\n", \
            __FILE__, __LINE__, _a, _e); \
        return 1; \
    } \
} while (0)

static Bat make_test_env(void) {
    Bat env = {
        .num_agents = 1,
        .num_obstacles = 1,
        .ear_separation_scale = 0.75f,
        .ear_rear_gain = 0.20f,
        .ear_front_gain = 0.55f,
        .ear_side_gain = 0.35f,
        .max_speed = 12.0f,
        .min_speed = 2.4f,
        .accel = 30.0f,
        .turn_rate = 3.1415926f,
        .sound_speed = 100.0f,
        .reflector_strength = 2.0f,
        .chirp_cooldown_ticks = 12,
        .chirp_efficiency_reward = 1.0f,
        .step_cost = 0.001f,
        .progress_reward_scale = 0.05f,
        .collision_penalty = 1.0f,
        .valid_chirp_reward = 0.0005f,
        .early_chirp_penalty = 0.001f,
        .bug_echo_farther_penalty_scale = 0.10f,
        .bug_wing_sideband_gain = 0.10f,
        .curriculum_obstacle_step = 8,
        .curriculum_successes_per_level = 1,
        .curriculum_start_bug_distance = 14.0f,
        .rng = 1,
    };
    allocate(&env);
    return env;
}

static int test_chirp_metadata_and_observation_size(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.actions[0] = 0.0f;
    env.actions[1] = 0.0f;
    env.actions[2] = 7.0f;
    env.actions[3] = 0.0f;
    env.actions[4] = 3.0f;
    env.actions[5] = 1.0f;
    c_step(&env);

    ASSERT_FLOAT_NEAR(env.observations[CHIRP_START_OBS], 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[CHIRP_END_OBS], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[CHIRP_DURATION_OBS], 1.0f, 0.0001f);
    ASSERT_TRUE(env.observations[CHIRP_AGE_OBS] <= 1.0f);
    ASSERT_TRUE(env.observations[CHIRP_AGE_OBS] >= 0.0f);

    free_allocated(&env);
    return 0;
}

static int test_chirps_used_observation_tracks_emitted_chirps(void) {
    Bat env = make_test_env();
    c_reset(&env);

    ASSERT_FLOAT_NEAR(env.observations[CHIRPS_USED_OBS], 0.0f, 0.0001f);

    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 1.0f;
    c_step(&env);

    ASSERT_TRUE(env.chirps_emitted == 1);
    ASSERT_FLOAT_NEAR(env.observations[CHIRPS_USED_OBS],
        1.0f / (float)MAX_CHIRPS_PER_EPISODE, 0.0001f);

    env.chirps_emitted = MAX_CHIRPS_PER_EPISODE + 1;
    compute_observations(&env);
    ASSERT_FLOAT_NEAR(env.observations[CHIRPS_USED_OBS], 1.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_max_chirps_stays_fixed_with_curriculum_level(void) {
    Bat env = make_test_env();
    env.curriculum_initial_level = 8;
    c_reset(&env);

    ASSERT_TRUE(env.curriculum_level == 8);
    env.chirps_emitted = 1;
    compute_observations(&env);
    ASSERT_FLOAT_NEAR(env.observations[CHIRPS_USED_OBS],
        1.0f / (float)MAX_CHIRPS_PER_EPISODE, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_chirping_after_budget_terminates_with_penalty(void) {
    Bat env = make_test_env();
    env.chirp_cooldown_ticks = 5;
    env.early_chirp_penalty = 0.0f;
    c_reset(&env);
    env.chirps_emitted = MAX_CHIRPS_PER_EPISODE - 1;
    compute_observations(&env);

    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 1.0f;
    c_step(&env);
    ASSERT_TRUE(env.terminals[0] == 0.0f);
    ASSERT_TRUE(env.chirps_emitted == MAX_CHIRPS_PER_EPISODE);
    ASSERT_FLOAT_NEAR(env.observations[CHIRPS_USED_OBS], 1.0f, 0.0001f);

    c_step(&env);

    ASSERT_TRUE(env.terminals[0] == 1.0f);
    ASSERT_FLOAT_NEAR(env.rewards[0], -1.0f, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted == 0);

    free_allocated(&env);
    return 0;
}

static int test_timer_observation_tracks_elapsed_fraction(void) {
    Bat env = make_test_env();
    c_reset(&env);

    ASSERT_TRUE(OBS_SIZE == 41);
    ASSERT_FLOAT_NEAR(env.observations[TIMER_OBS], 0.0f, 0.0001f);

    env.actions[0] = NOOP;
    env.actions[1] = TURN_NONE;
    env.actions[5] = 0.0f;
    c_step(&env);

    ASSERT_FLOAT_NEAR(env.observations[TIMER_OBS], 1.0f / (float)MAX_STEPS, 0.0001f);

    env.tick = MAX_STEPS / 2;
    compute_observations(&env);
    ASSERT_FLOAT_NEAR(env.observations[TIMER_OBS], 0.5f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_timeout_terminates_with_minus_one_reward(void) {
    Bat env = make_test_env();
    env.num_obstacles = 0;
    env.progress_reward_scale = 0.0f;
    env.step_cost = 0.0f;
    c_reset(&env);
    env.tick = MAX_STEPS - 1;

    env.actions[0] = NOOP;
    env.actions[1] = TURN_NONE;
    env.actions[5] = 0.0f;
    c_step(&env);

    ASSERT_TRUE(env.terminals[0] == 1.0f);
    ASSERT_FLOAT_NEAR(env.rewards[0], -1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.timeout, 1.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_efficiency_scores_low_usage_above_full_budget(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.chirps_emitted = 1;
    ASSERT_FLOAT_NEAR(chirp_efficiency(&env), 0.9666667f, 0.0001f);

    env.chirps_emitted = MAX_CHIRPS_PER_EPISODE;
    ASSERT_FLOAT_NEAR(chirp_efficiency(&env), 0.50f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_perf_uses_fixed_fifteen_chirp_reference(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.chirps_emitted = 0;
    ASSERT_FLOAT_NEAR(chirp_perf(&env), 1.0f, 0.0001f);

    env.chirps_emitted = 6;
    ASSERT_FLOAT_NEAR(chirp_perf(&env), 0.60f, 0.0001f);

    env.chirps_emitted = 8;
    ASSERT_FLOAT_NEAR(chirp_perf(&env), 0.4666667f, 0.0001f);

    env.chirps_emitted = 15;
    ASSERT_FLOAT_NEAR(chirp_perf(&env), 0.05f, 0.0001f);

    env.chirps_emitted = 30;
    ASSERT_FLOAT_NEAR(chirp_perf(&env), 0.05f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_success_reward_includes_chirp_efficiency_bonus(void) {
    Bat env = make_test_env();
    env.chirp_efficiency_reward = 1.0f;
    c_reset(&env);

    env.chirps_emitted = 2;
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], 0.9333333f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_perf_uses_distance_and_obstacle_difficulty(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.curriculum_start_bug_distance = 8.0f;
    env.num_obstacles = 2;
    env.start_bug_dist = 32.0f;

    ASSERT_FLOAT_NEAR(curriculum_distance_difficulty(&env), 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(curriculum_obstacle_difficulty(&env), 0.6666667f, 0.0001f);
    ASSERT_FLOAT_NEAR(curriculum_motion_difficulty(&env), 0.0000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(curriculum_difficulty(&env), 0.3888889f, 0.0001f);
    add_log(&env, 1.0f, 0.0f, 0.0f);
    ASSERT_FLOAT_NEAR(env.log.base_perf, 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_difficulty, 0.3888889f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_perf, 0.3888889f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.num_obstacles, 2.0f, 0.0001f);

    memset(&env.log, 0, sizeof(env.log));
    add_log(&env, 0.0f, 1.0f, 0.0f);
    ASSERT_FLOAT_NEAR(env.log.base_perf, 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_difficulty, 0.3888889f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_perf, 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_perf_composes_base_perf_curriculum_difficulty_and_chirp_perf(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.curriculum_start_bug_distance = 8.0f;
    env.num_obstacles = 2;
    env.chirps_emitted = 7;
    env.start_bug_dist = 32.0f;

    add_log(&env, 1.0f, 0.0f, 0.0f);

    ASSERT_FLOAT_NEAR(env.log.base_perf, 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.chirp_perf, 0.5333334f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_difficulty, 0.3888889f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.perf, 0.2074074f, 0.0001f);

    memset(&env.log, 0, sizeof(env.log));
    add_log(&env, 0.0f, 1.0f, 0.0f);
    ASSERT_FLOAT_NEAR(env.log.base_perf, 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.perf, 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_left_right_echo_asymmetry(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.x = 20.0f;
    env.y = 20.0f;
    env.heading = 0.0f;
    env.bug_x = 35.0f;
    env.bug_y = 10.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    clear_echo_queue(&env);
    env.tick = 0;

    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 1.0f,
        .end_freq = 1.0f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    schedule_echo(&env, &chirp, 0.0f, 1.0f,
        env.bug_x, env.bug_y, env.bug_vx, env.bug_vy, 8.0f, ECHO_BUG);

    float left_energy = 0.0f;
    float right_energy = 0.0f;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick < 0) continue;
        for (int bin = 0; bin < FREQ_BINS; bin++) {
            left_energy += env.echo_queue[i].energy[0][bin];
            right_energy += env.echo_queue[i].energy[1][bin];
        }
    }

    ASSERT_TRUE(left_energy > right_energy);

    free_allocated(&env);
    return 0;
}

typedef struct BatEchoProbe {
    float left_energy;
    float right_energy;
    float left_tick;
    float right_tick;
} BatEchoProbe;

static BatEchoProbe test_probe_echo_from_relative_source(float dx, float dy) {
    Bat env = make_test_env();
    c_reset(&env);

    env.x = 24.0f;
    env.y = 24.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.heading = 0.0f;
    env.sound_speed = 40.0f;
    env.ear_separation_scale = 2.0f;
    env.ear_rear_gain = 0.20f;
    env.ear_front_gain = 0.55f;
    env.ear_side_gain = 0.35f;
    env.tick = 0;
    clear_echo_queue(&env);

    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 0.5f,
        .end_freq = 0.5f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    schedule_echo(&env, &chirp, 0.0f, 0.5f,
        env.x + dx, env.y + dy, 0.0f, 0.0f, 8.0f, ECHO_BUG);

    BatEchoProbe probe = {
        .left_tick = -1.0f,
        .right_tick = -1.0f,
    };
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick < 0) continue;
        float left_energy = 0.0f;
        float right_energy = 0.0f;
        for (int bin = 0; bin < FREQ_BINS; bin++) {
            left_energy += env.echo_queue[i].energy[0][bin];
            right_energy += env.echo_queue[i].energy[1][bin];
        }
        if (left_energy > 0.0f) {
            probe.left_energy += left_energy;
            probe.left_tick = env.echo_queue[i].tick;
        }
        if (right_energy > 0.0f) {
            probe.right_energy += right_energy;
            probe.right_tick = env.echo_queue[i].tick;
        }
    }

    free_allocated(&env);
    return probe;
}

static int test_directional_echo_arrival_and_gain_by_side(void) {
    const float left_sources[3][2] = {
        {0.0f, -18.0f},
        {18.0f, -18.0f},
        {24.0f, -8.0f},
    };
    const float right_sources[3][2] = {
        {0.0f, 18.0f},
        {18.0f, 18.0f},
        {24.0f, 8.0f},
    };

    for (int i = 0; i < 3; i++) {
        BatEchoProbe left = test_probe_echo_from_relative_source(
            left_sources[i][0], left_sources[i][1]);
        ASSERT_TRUE(left.left_tick > 0.0f);
        ASSERT_TRUE(left.right_tick > 0.0f);
        ASSERT_TRUE(left.left_tick < left.right_tick);
        ASSERT_TRUE(left.left_energy > left.right_energy);

        BatEchoProbe right = test_probe_echo_from_relative_source(
            right_sources[i][0], right_sources[i][1]);
        ASSERT_TRUE(right.left_tick > 0.0f);
        ASSERT_TRUE(right.right_tick > 0.0f);
        ASSERT_TRUE(right.right_tick < right.left_tick);
        ASSERT_TRUE(right.right_energy > right.left_energy);
    }

    BatEchoProbe front = test_probe_echo_from_relative_source(18.0f, 0.0f);
    ASSERT_TRUE(front.left_tick > 0.0f);
    ASSERT_TRUE(front.right_tick > 0.0f);
    ASSERT_FLOAT_NEAR(front.left_tick, front.right_tick, 0.0001f);
    ASSERT_FLOAT_NEAR(front.left_energy, front.right_energy, 0.0001f);

    return 0;
}

static int test_ear_directivity_gains_control_echo_energy(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.x = 20.0f;
    env.y = 20.0f;
    env.heading = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.ear_rear_gain = 0.0f;
    env.ear_front_gain = 1.0f;
    env.ear_side_gain = 0.0f;
    env.tick = 0;

    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 1.0f,
        .end_freq = 1.0f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };

    clear_echo_queue(&env);
    schedule_echo(&env, &chirp, 0.0f, 1.0f,
        env.x + 16.0f, env.y, 0.0f, 0.0f, 8.0f, ECHO_BUG);
    float front_energy = 0.0f;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        for (int ear = 0; ear < 2; ear++) {
            for (int bin = 0; bin < FREQ_BINS; bin++) {
                front_energy += env.echo_queue[i].energy[ear][bin];
            }
        }
    }

    clear_echo_queue(&env);
    schedule_echo(&env, &chirp, 0.0f, 1.0f,
        env.x, env.y - 16.0f, 0.0f, 0.0f, 8.0f, ECHO_BUG);
    float side_energy = 0.0f;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        for (int ear = 0; ear < 2; ear++) {
            for (int bin = 0; bin < FREQ_BINS; bin++) {
                side_energy += env.echo_queue[i].energy[ear][bin];
            }
        }
    }

    ASSERT_TRUE(front_energy > 0.0f);
    ASSERT_FLOAT_NEAR(side_energy, 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_default_sound_speed_allows_one_tick_interaural_delay(void) {
    Bat env = {
        .num_agents = 1,
        .num_obstacles = 0,
        .ear_separation_scale = 0.75f,
        .ear_rear_gain = 0.20f,
        .ear_front_gain = 0.55f,
        .ear_side_gain = 0.35f,
        .max_speed = 12.0f,
        .accel = 30.0f,
        .turn_rate = 3.1415926f,
        .sound_speed = 60.0f,
        .rng = 1,
    };
    allocate(&env);

    env.x = 20.0f;
    env.y = 20.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.heading = 0.0f;
    env.tick = 0;
    clear_echo_queue(&env);

    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 0.5f,
        .end_freq = 0.5f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    schedule_echo(&env, &chirp, 0.0f, 0.5f,
        env.x, env.y - 12.0f, 0.0f, 0.0f, 8.0f, ECHO_BUG);

    float left_tick = -1.0f;
    float right_tick = -1.0f;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick < 0) continue;
        float left_energy = 0.0f;
        float right_energy = 0.0f;
        for (int bin = 0; bin < FREQ_BINS; bin++) {
            left_energy += env.echo_queue[i].energy[0][bin];
            right_energy += env.echo_queue[i].energy[1][bin];
        }
        if (left_energy > 0.0f) left_tick = env.echo_queue[i].tick;
        if (right_energy > 0.0f) right_tick = env.echo_queue[i].tick;
    }

    ASSERT_TRUE(left_tick > 0.0f);
    ASSERT_TRUE(right_tick > 0.0f);
    ASSERT_TRUE(fabsf(left_tick - right_tick) >= 1.0f);

    free_allocated(&env);
    return 0;
}

static int test_echo_scheduling_uses_tick_bucket_accumulator(void) {
    Bat env = make_test_env();
    c_reset(&env);

    clear_echo_queue(&env);
    env.tick = 7;
    add_echo_event(&env, 0, 9.25f, 1.0f, 0.4f, 18.0f, ECHO_BUG);
    add_echo_event(&env, 0, 9.75f, 1.0f, 0.7f, 12.0f, ECHO_BUG);

    int slot = 10 % ECHO_QUEUE_TICKS;
    ASSERT_TRUE(env.echo_queue[slot].tick == 10);
    ASSERT_FLOAT_NEAR(env.echo_queue[slot].energy[0][FREQ_BINS - 1], 1.1f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.echo_queue[slot].bug_energy, 1.1f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.echo_queue[slot].closest_bug_echo_path, 12.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bug_wing_sidebands_spill_adjacent_bins_without_reward_inflation(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.tick = 0;
    env.bug_wing_sideband_gain = 0.25f;
    clear_echo_queue(&env);

    int bin = freq_bin_index(0.5f);
    add_echo_event(&env, 0, 1.0f, 0.5f, 0.4f, 12.0f, ECHO_BUG);
    EchoBucket* bug_bucket = &env.echo_queue[1 % ECHO_QUEUE_TICKS];
    ASSERT_FLOAT_NEAR(bug_bucket->energy[0][bin], 0.4f, 0.0001f);
    ASSERT_FLOAT_NEAR(bug_bucket->energy[0][bin - 1], 0.1f, 0.0001f);
    ASSERT_FLOAT_NEAR(bug_bucket->energy[0][bin + 1], 0.1f, 0.0001f);
    ASSERT_FLOAT_NEAR(bug_bucket->bug_energy, 0.4f, 0.0001f);

    clear_echo_queue(&env);
    add_echo_event(&env, 0, 1.0f, 0.5f, 0.4f, 12.0f, ECHO_STATIC);
    EchoBucket* static_bucket = &env.echo_queue[1 % ECHO_QUEUE_TICKS];
    ASSERT_FLOAT_NEAR(static_bucket->energy[0][bin], 0.4f, 0.0001f);
    ASSERT_FLOAT_NEAR(static_bucket->energy[0][bin - 1], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(static_bucket->energy[0][bin + 1], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(static_bucket->bug_energy, 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static float test_side_echo_receive_tick_gap(float ear_separation_scale) {
    Bat env = make_test_env();
    c_reset(&env);

    env.ear_separation_scale = ear_separation_scale;
    env.x = 20.0f;
    env.y = 20.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.heading = 0.0f;
    env.tick = 0;
    clear_echo_queue(&env);

    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 0.5f,
        .end_freq = 0.5f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    schedule_echo(&env, &chirp, 0.0f, 0.5f,
        env.x, env.y - 12.0f, 0.0f, 0.0f, 8.0f, ECHO_BUG);

    float left_tick = -1.0f;
    float right_tick = -1.0f;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick < 0) continue;
        float left_energy = 0.0f;
        float right_energy = 0.0f;
        for (int bin = 0; bin < FREQ_BINS; bin++) {
            left_energy += env.echo_queue[i].energy[0][bin];
            right_energy += env.echo_queue[i].energy[1][bin];
        }
        if (left_energy > 0.0f) left_tick = env.echo_queue[i].tick;
        if (right_energy > 0.0f) right_tick = env.echo_queue[i].tick;
    }

    ASSERT_TRUE(left_tick > 0.0f);
    ASSERT_TRUE(right_tick > 0.0f);
    float gap = fabsf(left_tick - right_tick);

    free_allocated(&env);
    return gap;
}

static int test_ear_separation_scale_controls_arrival_gap(void) {
    float narrow_gap = test_side_echo_receive_tick_gap(0.75f);
    float wide_gap = test_side_echo_receive_tick_gap(1.50f);

    ASSERT_TRUE(narrow_gap > 0.0f);
    ASSERT_TRUE(wide_gap > narrow_gap * 1.75f);

    return 0;
}

static int test_doppler_sign_for_approaching_bug(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.x = 20.0f;
    env.y = 20.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.bug_x = 42.0f;
    env.bug_y = 20.0f;
    env.bug_vx = -16.0f;
    env.bug_vy = 0.0f;
    env.heading = 0.0f;
    memset(env.observations, 0, OBS_SIZE * sizeof(float));
    clear_echo_queue(&env);
    env.tick = 0;

    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 0.5f,
        .end_freq = 0.5f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    schedule_echo(&env, &chirp, 0.0f, 0.5f,
        env.bug_x, env.bug_y, env.bug_vx, env.bug_vy, 8.0f, ECHO_BUG);

    env.tick = 27;
    compute_observations(&env);

    float low_energy = 0.0f;
    float high_energy = 0.0f;
    for (int i = 0; i < FREQ_BINS; i++) {
        float energy = env.observations[LEFT_FREQ_OFFSET + i]
            + env.observations[RIGHT_FREQ_OFFSET + i];
        if (i < FREQ_BINS / 2) {
            low_energy += energy;
        } else {
            high_energy += energy;
        }
    }

    ASSERT_TRUE(high_energy > low_energy);

    free_allocated(&env);
    return 0;
}

static int test_wall_collision_is_terminal_minus_one(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.x = ARENA_WIDTH - AGENT_RADIUS - 0.1f;
    env.y = ARENA_HEIGHT * 0.5f;
    env.heading = 0.0f;
    env.vx = env.max_speed;
    env.vy = 0.0f;
    env.actions[0] = 1.0f;
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], -1.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_catch_bug_is_terminal_plus_one(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 1.0f, 0.0001f);
    ASSERT_TRUE(env.rewards[0] > 0.9f);

    free_allocated(&env);
    return 0;
}

static int test_progress_reward_sign(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 40.0f;
    env.bug_y = 20.0f;
    env.prev_bug_dist = 25.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;

    env.actions[0] = 1.0f;
    env.actions[1] = 0.0f;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;
    c_step(&env);

    ASSERT_TRUE(env.rewards[0] > 0.0f);

    free_allocated(&env);
    return 0;
}

static int test_bat_cannot_accelerate_backward_from_brake(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.heading = 0.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.actions[0] = BRAKE;
    env.actions[1] = TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    float forward = env.vx * cosf(env.heading) + env.vy * sinf(env.heading);
    ASSERT_TRUE(forward >= -0.0001f);
    ASSERT_TRUE(env.observations[FORWARD_SPEED_OBS] >= -0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_reset_starts_with_forward_stall_speed(void) {
    Bat env = make_test_env();
    c_reset(&env);

    float forward = env.vx * cosf(env.heading) + env.vy * sinf(env.heading);
    ASSERT_TRUE(forward >= 0.19f * env.max_speed);
    ASSERT_FLOAT_NEAR(env.observations[FORWARD_SPEED_OBS], forward / env.max_speed, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_brake_clamps_to_forward_stall_speed(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.heading = 0.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.actions[0] = BRAKE;
    env.actions[1] = TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    float forward = env.vx * cosf(env.heading) + env.vy * sinf(env.heading);
    ASSERT_TRUE(forward >= 0.19f * env.max_speed);
    ASSERT_TRUE(env.x > 20.0f);

    free_allocated(&env);
    return 0;
}

static int test_bat_velocity_is_locked_to_heading(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.heading = 0.0f;
    env.vx = -env.max_speed * 0.5f;
    env.vy = 3.0f;
    env.actions[0] = NOOP;
    env.actions[1] = TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    float forward = env.vx * cosf(env.heading) + env.vy * sinf(env.heading);
    float lateral = env.vx * -sinf(env.heading) + env.vy * cosf(env.heading);
    ASSERT_TRUE(forward >= -0.0001f);
    ASSERT_FLOAT_NEAR(lateral, 0.0f, 0.0001f);
    ASSERT_TRUE(env.observations[FORWARD_SPEED_OBS] >= -0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_zero_speed_recovers_to_forward_arc(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.heading = 0.25f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.actions[0] = NOOP;
    env.actions[1] = TURN_LEFT;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    float start_x = env.x;
    float start_y = env.y;
    c_step(&env);

    float forward = env.vx * cosf(env.heading) + env.vy * sinf(env.heading);
    ASSERT_TRUE(forward >= 0.19f * env.max_speed);
    ASSERT_TRUE(dist(start_x, start_y, env.x, env.y) > 0.0f);
    ASSERT_TRUE(fabsf(env.heading - 0.25f) > 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_turn_rate_scales_with_forward_speed(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.heading = 0.0f;
    env.vx = env.max_speed * 0.5f;
    env.vy = 0.0f;
    env.actions[0] = NOOP;
    env.actions[1] = TURN_RIGHT;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.turn_velocity, env.turn_rate * 0.5f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.heading, env.turn_rate * 0.5f * TICK_RATE, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_speed_action_space_has_no_strafe(void) {
    ASSERT_TRUE(MOVE_ACTIONS == 3);
    ASSERT_TRUE(NOOP == 0);
    ASSERT_TRUE(THRUST_FORWARD == 1);
    ASSERT_TRUE(BRAKE == 2);
    return 0;
}

static int test_chirp_audio_maps_norm_freq_to_audible_sweep(void) {
    ASSERT_FLOAT_NEAR(chirp_audio_frequency_hz(0.0f), 600.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(chirp_audio_frequency_hz(1.0f), 3600.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(chirp_audio_sample_f32(0.0f, 1.0f, 0.20f, -1, 48000), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(chirp_audio_sample_f32(0.0f, 1.0f, 0.20f, 9600, 48000), 0.0f, 0.0001f);
    float sample = chirp_audio_sample_f32(0.0f, 1.0f, 0.20f, 2400, 48000);
    ASSERT_TRUE(sample >= -0.25f);
    ASSERT_TRUE(sample <= 0.25f);
    return 0;
}

static int test_chirp_audio_duration_scales_with_render_fps(void) {
    Bat env = make_test_env();
    float base_duration = chirp_duration_seconds(0.0f);
    env.render_target_fps = 60;
    ASSERT_FLOAT_NEAR(chirp_audio_duration_seconds(&env, 0.0f), base_duration, 0.0001f);
    env.render_target_fps = 30;
    ASSERT_FLOAT_NEAR(chirp_audio_duration_seconds(&env, 0.0f), base_duration * 2.0f, 0.0001f);
    env.render_target_fps = 15;
    ASSERT_FLOAT_NEAR(chirp_audio_duration_seconds(&env, 0.0f), base_duration * 4.0f, 0.0001f);
    free_allocated(&env);
    return 0;
}

static int test_chirp_cooldown_accepts_only_after_delay(void) {
    Bat env = make_test_env();
    c_reset(&env);
    env.chirp_cooldown_ticks = 12;

    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 1.0f;
    ASSERT_TRUE(try_emit_chirp(&env));
    ASSERT_TRUE(!try_emit_chirp(&env));

    env.tick += 12;
    ASSERT_TRUE(try_emit_chirp(&env));

    free_allocated(&env);
    return 0;
}

static void test_place_safe_stationary_scene(Bat* env) {
    env->num_obstacles = 0;
    env->x = 20.0f;
    env->y = 20.0f;
    env->vx = 0.0f;
    env->vy = 0.0f;
    env->heading = 0.0f;
    env->bug_x = 48.0f;
    env->bug_y = 48.0f;
    env->bug_vx = 0.0f;
    env->bug_vy = 0.0f;
    env->prev_bug_dist = dist(env->x, env->y, env->bug_x, env->bug_y);
}

static void test_set_emit_chirp_action(Bat* env) {
    env->actions[0] = NOOP;
    env->actions[1] = TURN_NONE;
    env->actions[2] = 0.0f;
    env->actions[3] = 7.0f;
    env->actions[4] = 1.0f;
    env->actions[5] = 1.0f;
}

static int test_valid_chirp_gets_reward(void) {
    Bat env = make_test_env();
    c_reset(&env);
    test_place_safe_stationary_scene(&env);
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.bug_echo_reward_scale = 0.0f;
    env.valid_chirp_reward = 0.0005f;
    env.early_chirp_penalty = 0.0020f;
    test_set_emit_chirp_action(&env);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], env.valid_chirp_reward, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted == 1);

    free_allocated(&env);
    return 0;
}

static int test_early_chirp_gets_penalty_and_emits_nothing(void) {
    Bat env = make_test_env();
    c_reset(&env);
    test_place_safe_stationary_scene(&env);
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.bug_echo_reward_scale = 0.0f;
    env.valid_chirp_reward = 0.0005f;
    env.early_chirp_penalty = 0.0020f;
    env.chirp_cooldown_ticks = 12;
    test_set_emit_chirp_action(&env);
    c_step(&env);
    test_place_safe_stationary_scene(&env);
    test_set_emit_chirp_action(&env);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], -env.early_chirp_penalty, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted == 1);

    free_allocated(&env);
    return 0;
}

static int test_chirp_before_bug_echo_arrives_gets_scaled_overlap_penalty(void) {
    Bat env = make_test_env();
    c_reset(&env);
    test_place_safe_stationary_scene(&env);
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.bug_echo_reward_scale = 0.0f;
    env.valid_chirp_reward = 0.0005f;
    env.early_chirp_penalty = 0.0020f;
    env.chirp_overlap_penalty = 0.0040f;
    env.chirp_cooldown_ticks = 1;
    test_set_emit_chirp_action(&env);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], env.valid_chirp_reward, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted == 1);

    env.last_chirp_tick = 0;
    env.last_bug_echo_expected_tick = 10.0f;
    env.tick = 5;
    test_place_safe_stationary_scene(&env);
    test_set_emit_chirp_action(&env);
    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0],
        env.valid_chirp_reward - 0.5f * env.chirp_overlap_penalty, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted == 2);

    free_allocated(&env);
    return 0;
}

static int test_chirp_after_bug_echo_arrives_ignores_static_echo_window(void) {
    Bat env = make_test_env();
    c_reset(&env);
    test_place_safe_stationary_scene(&env);
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.bug_echo_reward_scale = 0.0f;
    env.valid_chirp_reward = 0.0005f;
    env.chirp_overlap_penalty = 0.0040f;
    env.chirp_cooldown_ticks = 1;
    env.chirps_emitted = 1;
    env.last_chirp_tick = 0;
    env.last_bug_echo_expected_tick = 3.0f;
    env.tick = 4;
    test_set_emit_chirp_action(&env);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], env.valid_chirp_reward, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted == 2);

    free_allocated(&env);
    return 0;
}

static int test_reflection_arrives_at_two_way_travel_time(void) {
    float sound_speed = 100.0f;
    float distance = 25.0f;
    float echo_time = 2.0f * distance / sound_speed;

    ASSERT_FLOAT_NEAR(echo_time, 0.5f, 0.0001f);
    ASSERT_TRUE(fabsf((echo_time + 0.005f) - echo_time) <= 0.02f);
    ASSERT_TRUE(fabsf((echo_time + 0.050f) - echo_time) > 0.02f);

    return 0;
}

static float test_sum_obs(Bat* env, int offset, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        sum += env->observations[offset + i];
    }
    return sum;
}

static int test_bins_only_observation_layout(void) {
    ASSERT_TRUE(OBS_SIZE == 41);
    ASSERT_TRUE(FREQ_BINS == 16);
    ASSERT_TRUE(LEFT_FREQ_OFFSET == 0);
    ASSERT_TRUE(RIGHT_FREQ_OFFSET == 16);
    ASSERT_TRUE(CHIRP_AGE_OBS == 32);
    ASSERT_TRUE(CHIRP_COOLDOWN_OBS == 33);
    ASSERT_TRUE(CHIRP_START_OBS == 34);
    ASSERT_TRUE(CHIRP_END_OBS == 35);
    ASSERT_TRUE(CHIRP_DURATION_OBS == 36);
    ASSERT_TRUE(CHIRPS_USED_OBS == 37);
    ASSERT_TRUE(FORWARD_SPEED_OBS == 38);
    ASSERT_TRUE(TURN_RATE_OBS == 39);
    ASSERT_TRUE(TIMER_OBS == 40);
    return 0;
}

static int test_no_chirp_produces_silent_frequency_bins(void) {
    Bat env = make_test_env();
    c_reset(&env);

    ASSERT_FLOAT_NEAR(test_sum_obs(&env, LEFT_FREQ_OFFSET, FREQ_BINS), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(test_sum_obs(&env, RIGHT_FREQ_OFFSET, FREQ_BINS), 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_observations_stay_normalized_after_chirp(void) {
    Bat env = make_test_env();
    c_reset(&env);

    ASSERT_FLOAT_NEAR(env.observations[CHIRP_AGE_OBS], 1.0f, 0.0001f);
    for (int i = 0; i < OBS_SIZE; i++) {
        ASSERT_TRUE(env.observations[i] >= -1.0f);
        ASSERT_TRUE(env.observations[i] <= 1.0f);
    }

    env.actions[0] = NOOP;
    env.actions[1] = TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 1.0f;
    c_step(&env);

    float age_denom = chirp_age_norm_denominator(&env);
    ASSERT_FLOAT_NEAR(env.observations[CHIRP_AGE_OBS], 1.0f / age_denom, 0.0001f);
    for (int i = 0; i < OBS_SIZE; i++) {
        ASSERT_TRUE(env.observations[i] >= -1.0f);
        ASSERT_TRUE(env.observations[i] <= 1.0f);
    }

    free_allocated(&env);
    return 0;
}

static int test_curriculum_level_zero_starts_close_with_no_obstacles(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_obstacle_step = 1;
    env.curriculum_start_bug_distance = 12.0f;
    c_reset(&env);

    ASSERT_TRUE(env.num_obstacles == 0);
    ASSERT_TRUE(dist(env.x, env.y, env.bug_x, env.bug_y) <= 14.0f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_adds_first_obstacle_after_level_zero(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_obstacle_step = 4;

    env.curriculum_initial_level = 1;
    c_reset(&env);
    ASSERT_TRUE(env.num_obstacles == 1);

    env.curriculum_initial_level = 5;
    env.curriculum_level = 0;
    c_reset(&env);
    ASSERT_TRUE(env.num_obstacles == 2);

    env.curriculum_initial_level = 9;
    env.curriculum_level = 0;
    c_reset(&env);
    ASSERT_TRUE(env.num_obstacles == 3);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_advances_after_catch(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_obstacle_step = 1;
    env.curriculum_start_bug_distance = 12.0f;
    c_reset(&env);
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_TRUE(env.curriculum_level == 1);
    ASSERT_TRUE(env.num_obstacles == 1);
    ASSERT_TRUE(dist(env.x, env.y, env.bug_x, env.bug_y) <= 16.0f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_waits_for_required_catches(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_obstacle_step = 1;
    env.curriculum_start_bug_distance = 12.0f;
    env.curriculum_successes_per_level = 2;
    c_reset(&env);
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_TRUE(env.curriculum_level == 0);
    ASSERT_TRUE(env.curriculum_successes_at_level == 1);

    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_TRUE(env.curriculum_level == 1);
    ASSERT_TRUE(env.curriculum_successes_at_level == 0);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_initial_level_sets_first_reset_difficulty(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_initial_level = 4;
    env.curriculum_obstacle_step = 2;
    env.curriculum_start_bug_distance = 8.0f;
    c_reset(&env);

    ASSERT_TRUE(env.curriculum_level == 4);
    ASSERT_TRUE(env.num_obstacles == 2);
    float distance = dist(env.x, env.y, env.bug_x, env.bug_y);
    ASSERT_TRUE(distance >= 15.0f);
    ASSERT_TRUE(distance <= 17.0f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_initial_level_does_not_reset_progress(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_initial_level = 2;
    env.curriculum_obstacle_step = 1;
    env.curriculum_successes_per_level = 1;
    env.curriculum_start_bug_distance = 8.0f;
    c_reset(&env);
    env.x = 20.0f;
    env.y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_TRUE(env.curriculum_level == 3);
    ASSERT_TRUE(env.curriculum_successes_at_level == 0);

    free_allocated(&env);
    return 0;
}

static int test_bug_bounces_off_arena_walls(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.bug_x = ARENA_WIDTH - BUG_RADIUS + 0.1f;
    env.bug_y = ARENA_HEIGHT * 0.5f;
    env.bug_vx = 3.0f;
    env.bug_vy = 1.0f;
    update_bug(&env, 0.0f);
    ASSERT_TRUE(env.bug_x == ARENA_WIDTH - BUG_RADIUS);
    ASSERT_TRUE(env.bug_vx < 0.0f);
    ASSERT_TRUE(env.bug_vy == 1.0f);

    env.bug_x = ARENA_WIDTH * 0.5f;
    env.bug_y = BUG_RADIUS - 0.1f;
    env.bug_vx = 2.0f;
    env.bug_vy = -4.0f;
    update_bug(&env, 0.0f);
    ASSERT_TRUE(env.bug_y == BUG_RADIUS);
    ASSERT_TRUE(env.bug_vx == 2.0f);
    ASSERT_TRUE(env.bug_vy > 0.0f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_echo_arrives_after_two_way_travel_not_immediately(void) {
    Bat env = make_test_env();
    env.num_obstacles = 0;
    env.sound_speed = 60.0f;
    c_reset(&env);

    env.x = 32.0f;
    env.y = 32.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.heading = 0.0f;
    env.bug_x = 38.0f;
    env.bug_y = 32.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    compute_observations(&env);

    env.actions[0] = NOOP;
    env.actions[1] = TURN_NONE;
    env.actions[2] = 7;
    env.actions[3] = 7;
    env.actions[4] = 0;
    env.actions[5] = 1;
    c_step(&env);

    for (int i = 0; i < 6; i++) {
        ASSERT_FLOAT_NEAR(test_sum_obs(&env, LEFT_FREQ_OFFSET, FREQ_BINS), 0.0f, 0.0001f);
        ASSERT_FLOAT_NEAR(test_sum_obs(&env, RIGHT_FREQ_OFFSET, FREQ_BINS), 0.0f, 0.0001f);
        env.actions[5] = 0;
        c_step(&env);
    }

    float max_energy = 0.0f;
    for (int i = 0; i < 32; i++) {
        float energy = test_sum_obs(&env, LEFT_FREQ_OFFSET, FREQ_BINS)
            + test_sum_obs(&env, RIGHT_FREQ_OFFSET, FREQ_BINS);
        if (energy > max_energy) max_energy = energy;
        c_step(&env);
    }

    ASSERT_TRUE(max_energy > 0.01f);

    free_allocated(&env);
    return 0;
}

static int test_default_echo_range_reaches_curriculum_max_bug_distance(void) {
    Bat env = {
        .num_agents = 1,
        .num_obstacles = 0,
        .max_speed = 22.0f,
        .min_speed = 2.0f,
        .accel = 45.0f,
        .turn_rate = 9.424778f,
        .ear_rear_gain = 0.20f,
        .ear_front_gain = 0.55f,
        .ear_side_gain = 0.35f,
        .sound_speed = 180.0f,
        .rng = 1,
    };
    allocate(&env);
    c_reset(&env);

    env.tick = 0;
    env.x = 4.0f;
    env.y = 32.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.heading = 0.0f;
    env.bug_x = env.x + CURRICULUM_INBOUND_MAX_BUG_DISTANCE;
    env.bug_y = env.y;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    clear_echo_queue(&env);

    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 0.0f,
        .end_freq = 1.0f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = env.tick,
        .active = 1,
    };
    chirp.slice_count = (int)ceilf(chirp.duration / TICK_RATE);
    while (chirp.slices_scheduled < chirp.slice_count) {
        int slice_idx = chirp.slices_scheduled;
        schedule_chirp_slice_echoes(&env, &chirp, slice_idx);
        chirp.slices_scheduled += 1;
    }

    float bug_energy = 0.0f;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        bug_energy += env.echo_queue[i].bug_energy;
    }

    ASSERT_TRUE(bug_energy > 0.0f);

    free_allocated(&env);
    return 0;
}

static float test_sum_queued_echo_energy(Bat* env) {
    float energy = 0.0f;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        for (int ear = 0; ear < 2; ear++) {
            for (int bin = 0; bin < FREQ_BINS; bin++) {
                energy += env->echo_queue[i].energy[ear][bin];
            }
        }
    }
    return energy;
}

static int test_corner_reflectors_enabled_schedule_stable_echo_events(void) {
    Bat env = make_test_env();
    env.num_obstacles = 0;
    env.sound_speed = 180.0f;
    c_reset(&env);

    env.tick = 0;
    env.x = 32.0f;
    env.y = 32.0f;
    env.heading = 0.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    clear_echo_queue(&env);
    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 0.0f,
        .end_freq = 1.0f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = env.tick,
        .active = 1,
    };

    schedule_corner_reflector_echoes(&env, &chirp, 0.0f, 0.5f);

    ASSERT_TRUE(test_sum_queued_echo_energy(&env) > 0.0f);

    free_allocated(&env);
    return 0;
}

static int test_corner_reflector_echo_observations_stay_normalized(void) {
    Bat env = make_test_env();
    env.num_obstacles = 0;
    env.sound_speed = 180.0f;
    c_reset(&env);

    env.tick = 0;
    env.x = 32.0f;
    env.y = 32.0f;
    env.heading = 0.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    clear_echo_queue(&env);
    ChirpEvent chirp = {
        .x = env.x,
        .y = env.y,
        .start_freq = 0.0f,
        .end_freq = 1.0f,
        .duration = chirp_duration_seconds(0.0f),
        .birth_tick = env.tick,
        .active = 1,
    };
    schedule_corner_reflector_echoes(&env, &chirp, 0.0f, 0.5f);

    int arrival_tick = -1;
    for (int i = 0; i < ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick > 0 && test_sum_queued_echo_energy(&env) > 0.0f) {
            arrival_tick = env.echo_queue[i].tick;
            break;
        }
    }
    ASSERT_TRUE(arrival_tick > 0);

    env.tick = arrival_tick;
    compute_observations(&env);
    ASSERT_TRUE(test_sum_obs(&env, LEFT_FREQ_OFFSET, FREQ_BINS) > 0.0f ||
        test_sum_obs(&env, RIGHT_FREQ_OFFSET, FREQ_BINS) > 0.0f);
    for (int i = 0; i < OBS_SIZE; i++) {
        ASSERT_TRUE(env.observations[i] >= -1.0f);
        ASSERT_TRUE(env.observations[i] <= 1.0f);
    }

    free_allocated(&env);
    return 0;
}

static int test_frequency_bin_energy_sums_and_caps(void) {
    Bat env = make_test_env();
    memset(env.observations, 0, OBS_SIZE * sizeof(float));

    int high_bin = freq_bin_index(1.0f);
    int low_bin = freq_bin_index(0.0f);
    env.observations[LEFT_FREQ_OFFSET + high_bin] = bat_clampf(
        env.observations[LEFT_FREQ_OFFSET + high_bin] + 0.75f, 0.0f, 1.0f);
    env.observations[LEFT_FREQ_OFFSET + high_bin] = bat_clampf(
        env.observations[LEFT_FREQ_OFFSET + high_bin] + 0.75f, 0.0f, 1.0f);
    env.observations[RIGHT_FREQ_OFFSET + low_bin] = bat_clampf(
        env.observations[RIGHT_FREQ_OFFSET + low_bin] + 0.35f, 0.0f, 1.0f);

    ASSERT_FLOAT_NEAR(env.observations[LEFT_FREQ_OFFSET + FREQ_BINS - 1], 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[RIGHT_FREQ_OFFSET], 0.35f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bug_echo_reward_is_added_when_bug_echo_is_closer(void) {
    Bat env = make_test_env();
    c_reset(&env);
    env.bug_echo_reward_scale = 0.05f;
    env.last_bug_echo_path = 20.0f;
    env.last_bug_echo_x = 8.0f;
    env.last_bug_echo_y = 10.0f;
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 10.0f;
    env.y = 10.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    clear_echo_queue(&env);
    add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 15.0f, ECHO_BUG);

    c_step(&env);

    ASSERT_TRUE(env.rewards[0] > 0.0015f);
    ASSERT_FLOAT_NEAR(env.observations[LEFT_FREQ_OFFSET + 8], 0.6f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bug_echo_reward_requires_bat_displacement(void) {
    Bat env = make_test_env();
    c_reset(&env);
    env.bug_echo_reward_scale = 0.05f;
    env.last_bug_echo_path = 20.0f;
    env.last_bug_echo_x = 10.0f;
    env.last_bug_echo_y = 10.0f;
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 10.0f;
    env.y = 10.0f;
    env.heading = 0.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    clear_echo_queue(&env);
    add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 15.0f, ECHO_BUG);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.rewards[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.last_bug_echo_path, 15.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bug_echo_reward_penalizes_farther_bug_echo_weakly(void) {
    Bat env = make_test_env();
    c_reset(&env);
    env.bug_echo_reward_scale = 0.05f;
    env.last_bug_echo_path = 20.0f;
    env.last_bug_echo_x = 8.0f;
    env.last_bug_echo_y = 10.0f;
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 10.0f;
    env.y = 10.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    clear_echo_queue(&env);
    add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 25.0f, ECHO_BUG);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.rewards[0], -0.0001953f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.last_bug_echo_path, 25.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_static_echo_does_not_get_bug_echo_reward(void) {
    Bat env = make_test_env();
    c_reset(&env);
    env.bug_echo_reward_scale = 0.05f;
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.x = 10.0f;
    env.y = 10.0f;
    env.vx = 0.0f;
    env.vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    clear_echo_queue(&env);
    add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 15.0f, ECHO_STATIC);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.rewards[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[LEFT_FREQ_OFFSET + 8], 0.6f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_spawns_use_curriculum_distance_with_random_positions(void) {
    Bat env = make_test_env();
    float first_x = 0.0f;
    float first_y = 0.0f;
    float first_bug_x = 0.0f;
    float first_bug_y = 0.0f;
    float max_bat_delta = 0.0f;
    float max_bug_delta = 0.0f;

    for (int i = 0; i < 48; i++) {
        c_reset(&env);
        ASSERT_FLOAT_NEAR(dist(env.x, env.y, env.bug_x, env.bug_y),
            env.curriculum_start_bug_distance, 0.001f);
        if (i == 0) {
            first_x = env.x;
            first_y = env.y;
            first_bug_x = env.bug_x;
            first_bug_y = env.bug_y;
        } else {
            float bat_delta = dist(first_x, first_y, env.x, env.y);
            float bug_delta = dist(first_bug_x, first_bug_y, env.bug_x, env.bug_y);
            if (bat_delta > max_bat_delta) max_bat_delta = bat_delta;
            if (bug_delta > max_bug_delta) max_bug_delta = bug_delta;
        }
    }

    ASSERT_TRUE(max_bat_delta > 8.0f);
    ASSERT_TRUE(max_bug_delta > 8.0f);

    free_allocated(&env);
    return 0;
}

static int test_spawns_keep_minimum_separation_and_avoid_obstacles(void) {
    Bat env = make_test_env();
    env.curriculum_initial_level = 1;
    float expected_distance = env.curriculum_start_bug_distance + CURRICULUM_BUG_DISTANCE_STEP;

    for (int reset = 0; reset < 32; reset++) {
        c_reset(&env);
        ASSERT_FLOAT_NEAR(dist(env.x, env.y, env.bug_x, env.bug_y),
            expected_distance, 0.001f);
        for (int i = 0; i < env.num_obstacles; i++) {
            ASSERT_TRUE(!circle_rect_collision(env.x, env.y, AGENT_RADIUS + 1.0f,
                env.obstacle_x[i], env.obstacle_y[i], env.obstacle_w[i], env.obstacle_h[i]));
            ASSERT_TRUE(!circle_rect_collision(env.bug_x, env.bug_y, BUG_RADIUS + 1.0f,
                env.obstacle_x[i], env.obstacle_y[i], env.obstacle_w[i], env.obstacle_h[i]));
        }
    }

    free_allocated(&env);
    return 0;
}

static int test_obstacles_move_substantially_across_resets(void) {
    Bat env = make_test_env();
    env.curriculum_initial_level = 1;
    c_reset(&env);
    float first_x = env.obstacle_x[0];
    float first_y = env.obstacle_y[0];
    float max_delta = 0.0f;

    for (int i = 0; i < 32; i++) {
        c_reset(&env);
        float delta = dist(first_x, first_y, env.obstacle_x[0], env.obstacle_y[0]);
        if (delta > max_delta) max_delta = delta;
    }

    ASSERT_TRUE(max_delta > 16.0f);

    free_allocated(&env);
    return 0;
}

static int test_obstacles_are_small_enough_for_trainability(void) {
    Bat env = make_test_env();
    env.curriculum_initial_level = 1;

    for (int reset = 0; reset < 64; reset++) {
        c_reset(&env);
        for (int i = 0; i < env.num_obstacles; i++) {
            ASSERT_TRUE(env.obstacle_w[i] >= 3.0f);
            ASSERT_TRUE(env.obstacle_h[i] >= 3.0f);
            ASSERT_TRUE(env.obstacle_w[i] <= 8.0f);
            ASSERT_TRUE(env.obstacle_h[i] <= 8.0f);
            ASSERT_TRUE(env.obstacle_w[i] * env.obstacle_h[i] <= 64.0f);
        }
    }

    free_allocated(&env);
    return 0;
}

int main(void) {
    if (test_chirp_metadata_and_observation_size()) return 1;
    if (test_chirps_used_observation_tracks_emitted_chirps()) return 1;
    if (test_max_chirps_stays_fixed_with_curriculum_level()) return 1;
    if (test_chirping_after_budget_terminates_with_penalty()) return 1;
    if (test_timer_observation_tracks_elapsed_fraction()) return 1;
    if (test_timeout_terminates_with_minus_one_reward()) return 1;
    if (test_chirp_efficiency_scores_low_usage_above_full_budget()) return 1;
    if (test_chirp_perf_uses_fixed_fifteen_chirp_reference()) return 1;
    if (test_success_reward_includes_chirp_efficiency_bonus()) return 1;
    if (test_curriculum_perf_uses_distance_and_obstacle_difficulty()) return 1;
    if (test_perf_composes_base_perf_curriculum_difficulty_and_chirp_perf()) return 1;
    if (test_left_right_echo_asymmetry()) return 1;
    if (test_directional_echo_arrival_and_gain_by_side()) return 1;
    if (test_ear_directivity_gains_control_echo_energy()) return 1;
    if (test_default_sound_speed_allows_one_tick_interaural_delay()) return 1;
    if (test_echo_scheduling_uses_tick_bucket_accumulator()) return 1;
    if (test_bug_wing_sidebands_spill_adjacent_bins_without_reward_inflation()) return 1;
    if (test_ear_separation_scale_controls_arrival_gap()) return 1;
    if (test_doppler_sign_for_approaching_bug()) return 1;
    if (test_wall_collision_is_terminal_minus_one()) return 1;
    if (test_catch_bug_is_terminal_plus_one()) return 1;
    if (test_progress_reward_sign()) return 1;
    if (test_bat_cannot_accelerate_backward_from_brake()) return 1;
    if (test_bat_reset_starts_with_forward_stall_speed()) return 1;
    if (test_bat_brake_clamps_to_forward_stall_speed()) return 1;
    if (test_bat_velocity_is_locked_to_heading()) return 1;
    if (test_bat_zero_speed_recovers_to_forward_arc()) return 1;
    if (test_bat_turn_rate_scales_with_forward_speed()) return 1;
    if (test_bat_speed_action_space_has_no_strafe()) return 1;
    if (test_chirp_audio_maps_norm_freq_to_audible_sweep()) return 1;
    if (test_chirp_audio_duration_scales_with_render_fps()) return 1;
    if (test_chirp_cooldown_accepts_only_after_delay()) return 1;
    if (test_valid_chirp_gets_reward()) return 1;
    if (test_early_chirp_gets_penalty_and_emits_nothing()) return 1;
    if (test_chirp_before_bug_echo_arrives_gets_scaled_overlap_penalty()) return 1;
    if (test_chirp_after_bug_echo_arrives_ignores_static_echo_window()) return 1;
    if (test_reflection_arrives_at_two_way_travel_time()) return 1;
    if (test_bins_only_observation_layout()) return 1;
    if (test_no_chirp_produces_silent_frequency_bins()) return 1;
    if (test_observations_stay_normalized_after_chirp()) return 1;
    if (test_curriculum_level_zero_starts_close_with_no_obstacles()) return 1;
    if (test_curriculum_adds_first_obstacle_after_level_zero()) return 1;
    if (test_curriculum_advances_after_catch()) return 1;
    if (test_curriculum_waits_for_required_catches()) return 1;
    if (test_curriculum_initial_level_sets_first_reset_difficulty()) return 1;
    if (test_curriculum_initial_level_does_not_reset_progress()) return 1;
    if (test_bug_bounces_off_arena_walls()) return 1;
    if (test_chirp_echo_arrives_after_two_way_travel_not_immediately()) return 1;
    if (test_default_echo_range_reaches_curriculum_max_bug_distance()) return 1;
    if (test_corner_reflectors_enabled_schedule_stable_echo_events()) return 1;
    if (test_corner_reflector_echo_observations_stay_normalized()) return 1;
    if (test_frequency_bin_energy_sums_and_caps()) return 1;
    if (test_bug_echo_reward_is_added_when_bug_echo_is_closer()) return 1;
    if (test_bug_echo_reward_requires_bat_displacement()) return 1;
    if (test_bug_echo_reward_penalizes_farther_bug_echo_weakly()) return 1;
    if (test_static_echo_does_not_get_bug_echo_reward()) return 1;
    if (test_spawns_use_curriculum_distance_with_random_positions()) return 1;
    if (test_spawns_keep_minimum_separation_and_avoid_obstacles()) return 1;
    if (test_obstacles_move_substantially_across_resets()) return 1;
    if (test_obstacles_are_small_enough_for_trainability()) return 1;

    printf("bat core tests passed\n");
    return 0;
}
