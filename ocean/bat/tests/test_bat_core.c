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
        .frameskip = 1,
        .width = 64,
        .height = 64,
        .num_obstacles = 1,
        .bat_radius = 2.0f,
        .bug_radius = 1.5f,
        .bat_max_speed = 12.0f,
        .bat_accel = 30.0f,
        .bat_turn_rate = 3.1415926f,
        .bug_speed = 4.0f,
        .max_steps = 512,
        .freq_bins_per_ear = BAT_FREQ_BINS,
        .max_echo_range = 80.0f,
        .sound_speed = 100.0f,
        .reflector_spacing = 8.0f,
        .chirp_cost = 0.0005f,
        .step_cost = 0.001f,
        .progress_reward_scale = 0.05f,
        .collision_penalty = 1.0f,
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

    ASSERT_FLOAT_NEAR(env.observations[BAT_CHIRP_START_OBS], 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[BAT_CHIRP_END_OBS], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[BAT_CHIRP_DURATION_OBS], 1.0f, 0.0001f);
    ASSERT_TRUE(env.observations[BAT_CHIRP_AGE_OBS] <= 1.0f);
    ASSERT_TRUE(env.observations[BAT_CHIRP_AGE_OBS] >= 0.0f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_budget_observation_tracks_used_chirps(void) {
    Bat env = make_test_env();
    env.max_chirps_per_episode = 4;
    env.min_chirps_per_episode = 2;
    env.chirp_budget_decay_levels = 4;
    c_reset(&env);

    ASSERT_TRUE(env.chirp_budget == 4);
    ASSERT_FLOAT_NEAR(env.observations[BAT_CHIRPS_USED_OBS], 0.0f, 0.0001f);

    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 1.0f;
    c_step(&env);

    ASSERT_TRUE(env.chirps_emitted_episode == 1);
    ASSERT_FLOAT_NEAR(env.observations[BAT_CHIRPS_USED_OBS], 0.25f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_budget_stays_fixed_with_curriculum_level(void) {
    Bat env = make_test_env();
    env.curriculum_enabled = 1;
    env.curriculum_initial_level = 8;
    env.max_chirps_per_episode = 20;
    env.min_chirps_per_episode = 10;
    env.chirp_budget_decay_levels = 4;
    c_reset(&env);

    ASSERT_TRUE(env.curriculum_level == 8);
    ASSERT_TRUE(env.chirp_budget == 20);

    free_allocated(&env);
    return 0;
}

static int test_chirping_after_budget_terminates_with_penalty(void) {
    Bat env = make_test_env();
    env.max_chirps_per_episode = 1;
    env.min_chirps_per_episode = 1;
    env.chirp_budget_decay_levels = 4;
    env.chirp_cooldown_ticks = 1;
    env.early_chirp_penalty = 0.0f;
    c_reset(&env);

    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 1.0f;
    c_step(&env);
    ASSERT_TRUE(env.terminals[0] == 0.0f);
    ASSERT_TRUE(env.chirps_emitted_episode == 1);

    env.tick = env.last_chirp_tick + env.chirp_cooldown_ticks;
    c_step(&env);

    ASSERT_TRUE(env.terminals[0] == 1.0f);
    ASSERT_FLOAT_NEAR(env.rewards[0], -1.0f, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted_episode == 0);

    free_allocated(&env);
    return 0;
}

static int test_chirp_efficiency_scores_low_usage_above_full_budget(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.chirp_budget = 10;
    env.chirps_emitted_episode = 1;
    ASSERT_FLOAT_NEAR(bat_chirp_efficiency(&env), 0.95f, 0.0001f);

    env.chirps_emitted_episode = 10;
    ASSERT_FLOAT_NEAR(bat_chirp_efficiency(&env), 0.50f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_perf_uses_fixed_fifteen_chirp_reference(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.chirps_emitted_episode = 0;
    ASSERT_FLOAT_NEAR(bat_chirp_perf(&env), 1.0f, 0.0001f);

    env.chirps_emitted_episode = 6;
    ASSERT_FLOAT_NEAR(bat_chirp_perf(&env), 0.60f, 0.0001f);

    env.chirps_emitted_episode = 8;
    ASSERT_FLOAT_NEAR(bat_chirp_perf(&env), 0.4666667f, 0.0001f);

    env.chirps_emitted_episode = 15;
    ASSERT_FLOAT_NEAR(bat_chirp_perf(&env), 0.05f, 0.0001f);

    env.chirps_emitted_episode = 30;
    ASSERT_FLOAT_NEAR(bat_chirp_perf(&env), 0.05f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_success_reward_includes_chirp_efficiency_bonus(void) {
    Bat env = make_test_env();
    env.chirp_efficiency_reward = 1.0f;
    c_reset(&env);

    env.chirp_budget = 10;
    env.chirps_emitted_episode = 2;
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], 1.90f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_budget_logs_ratios_for_wandb(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.chirp_budget = 10;
    env.chirps_emitted_episode = 4;
    add_log(&env, 1.0f, 0.0f, 0.0f);

    ASSERT_FLOAT_NEAR(env.log.chirp_budget, 10.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.chirps_used_ratio, 0.40f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.chirps_remaining_ratio, 0.60f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.chirp_efficiency, 0.80f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_perf_logs_distance_and_obstacle_difficulty_components(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.curriculum_start_bug_distance = 8.0f;
    env.curriculum_max_bug_distance = 56.0f;
    env.curriculum_start_obstacles = 1;
    env.curriculum_max_obstacles = 3;
    env.num_obstacles = 2;
    env.max_chirps_per_episode = 15;
    env.min_chirps_per_episode = 6;
    env.chirp_budget = 12;
    env.start_bug_dist = 32.0f;

    ASSERT_FLOAT_NEAR(bat_curriculum_distance_difficulty(&env), 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(bat_curriculum_obstacle_difficulty(&env), 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(bat_curriculum_chirp_budget_difficulty(&env), 0.0000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(bat_curriculum_motion_difficulty(&env), 0.0000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(bat_curriculum_difficulty(&env), 0.5000000f, 0.0001f);
    add_log(&env, 1.0f, 0.0f, 0.0f);
    ASSERT_FLOAT_NEAR(env.log.base_perf, 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_distance_difficulty, 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_obstacle_difficulty, 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_chirp_budget_difficulty, 0.0000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_motion_difficulty, 0.0000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_difficulty, 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_perf, 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.num_obstacles, 2.0f, 0.0001f);

    memset(&env.log, 0, sizeof(env.log));
    add_log(&env, 0.0f, 1.0f, 0.0f);
    ASSERT_FLOAT_NEAR(env.log.base_perf, 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_difficulty, 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_perf, 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_budget_difficulty_uses_hard_edge_below_six_chirps(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.max_chirps_per_episode = 15;
    ASSERT_FLOAT_NEAR(bat_budget_difficulty(&env), 0.50f, 0.0001f);

    env.max_chirps_per_episode = 10;
    ASSERT_FLOAT_NEAR(bat_budget_difficulty(&env), 0.75f, 0.0001f);

    env.max_chirps_per_episode = 6;
    ASSERT_FLOAT_NEAR(bat_budget_difficulty(&env), 0.95f, 0.0001f);

    env.max_chirps_per_episode = 5;
    ASSERT_FLOAT_NEAR(bat_budget_difficulty(&env), 1.0f, 0.0001f);

    env.max_chirps_per_episode = 4;
    ASSERT_FLOAT_NEAR(bat_budget_difficulty(&env), 1.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_perf_composes_base_perf_curriculum_difficulty_and_chirp_perf(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.curriculum_start_bug_distance = 8.0f;
    env.curriculum_max_bug_distance = 56.0f;
    env.curriculum_start_obstacles = 1;
    env.curriculum_max_obstacles = 3;
    env.num_obstacles = 2;
    env.max_chirps_per_episode = 14;
    env.min_chirps_per_episode = 4;
    env.chirp_budget = 14;
    env.chirps_emitted_episode = 7;
    env.start_bug_dist = 32.0f;

    add_log(&env, 1.0f, 0.0f, 0.0f);

    ASSERT_FLOAT_NEAR(env.log.base_perf, 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.budget_difficulty, 0.55f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.chirp_efficiency, 0.75f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.chirp_perf, 0.5333334f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.curriculum_difficulty, 0.5000000f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.perf, 0.2666667f, 0.0001f);

    memset(&env.log, 0, sizeof(env.log));
    add_log(&env, 0.0f, 1.0f, 0.0f);
    ASSERT_FLOAT_NEAR(env.log.base_perf, 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.perf, 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_tempo_logs_far_and_near_rates(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.chirps_emitted_episode = 4;
    env.chirps_far = 2.0f;
    env.chirps_mid = 1.0f;
    env.chirps_near = 1.0f;
    env.ticks_far = 40.0f;
    env.ticks_mid = 20.0f;
    env.ticks_near = 10.0f;
    env.first_chirp_tick = 12.0f;
    env.chirp_tick_sum = 120.0f;
    env.max_steps = 120;

    add_log(&env, 1.0f, 0.0f, 0.0f);

    ASSERT_FLOAT_NEAR(env.log.far_chirp_fraction, 0.50f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.near_chirp_fraction, 0.25f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.far_chirp_rate, 0.05f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.near_chirp_rate, 0.10f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.chirp_tempo_ratio, 2.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.first_chirp_tick_norm, 0.10f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.log.mean_chirp_tick_norm, 0.25f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_left_right_echo_asymmetry(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bat_heading = 0.0f;
    env.bug_x = 35.0f;
    env.bug_y = 10.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    bat_clear_echo_queue(&env);
    env.tick = 0;

    ChirpEvent chirp = {
        .x = env.bat_x,
        .y = env.bat_y,
        .start_freq = 1.0f,
        .end_freq = 1.0f,
        .duration = bat_chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    bat_schedule_echo(&env, &chirp, 0.0f, 1.0f,
        env.bug_x, env.bug_y, env.bug_vx, env.bug_vy, 8.0f, BAT_ECHO_BUG);

    float left_energy = 0.0f;
    float right_energy = 0.0f;
    for (int i = 0; i < BAT_ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick < 0) continue;
        for (int bin = 0; bin < BAT_FREQ_BINS; bin++) {
            left_energy += env.echo_queue[i].energy[0][bin];
            right_energy += env.echo_queue[i].energy[1][bin];
        }
    }

    ASSERT_TRUE(left_energy > right_energy);

    free_allocated(&env);
    return 0;
}

static int test_default_sound_speed_allows_one_tick_interaural_delay(void) {
    Bat env = {
        .num_agents = 1,
        .frameskip = 1,
        .width = 64,
        .height = 64,
        .num_obstacles = 0,
        .bat_radius = 2.0f,
        .bug_radius = 1.5f,
        .bat_max_speed = 12.0f,
        .bat_accel = 30.0f,
        .bat_turn_rate = 3.1415926f,
        .bug_speed = 4.0f,
        .max_steps = 512,
        .freq_bins_per_ear = BAT_FREQ_BINS,
        .max_echo_range = 80.0f,
        .reflector_spacing = 8.0f,
        .rng = 1,
    };
    allocate(&env);

    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bat_heading = 0.0f;
    env.tick = 0;
    bat_clear_echo_queue(&env);

    ChirpEvent chirp = {
        .x = env.bat_x,
        .y = env.bat_y,
        .start_freq = 0.5f,
        .end_freq = 0.5f,
        .duration = bat_chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    bat_schedule_echo(&env, &chirp, 0.0f, 0.5f,
        env.bat_x, env.bat_y - 12.0f, 0.0f, 0.0f, 8.0f, BAT_ECHO_BUG);

    float left_tick = -1.0f;
    float right_tick = -1.0f;
    for (int i = 0; i < BAT_ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick < 0) continue;
        float left_energy = 0.0f;
        float right_energy = 0.0f;
        for (int bin = 0; bin < BAT_FREQ_BINS; bin++) {
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

    bat_clear_echo_queue(&env);
    env.tick = 7;
    bat_add_echo_event(&env, 0, 9.25f, 1.0f, 0.4f, 18.0f, BAT_ECHO_BUG);
    bat_add_echo_event(&env, 0, 9.75f, 1.0f, 0.7f, 12.0f, BAT_ECHO_BUG);

    int slot = 10 % BAT_ECHO_QUEUE_TICKS;
    ASSERT_TRUE(env.echo_queue[slot].tick == 10);
    ASSERT_FLOAT_NEAR(env.echo_queue[slot].energy[0][BAT_FREQ_BINS - 1], 1.1f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.echo_queue[slot].bug_energy, 1.1f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.echo_queue[slot].bug_path, 12.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static float test_side_echo_receive_tick_gap(float ear_separation_scale) {
    Bat env = make_test_env();
    c_reset(&env);

    env.ear_separation_scale = ear_separation_scale;
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bat_heading = 0.0f;
    env.tick = 0;
    bat_clear_echo_queue(&env);

    ChirpEvent chirp = {
        .x = env.bat_x,
        .y = env.bat_y,
        .start_freq = 0.5f,
        .end_freq = 0.5f,
        .duration = bat_chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    bat_schedule_echo(&env, &chirp, 0.0f, 0.5f,
        env.bat_x, env.bat_y - 12.0f, 0.0f, 0.0f, 8.0f, BAT_ECHO_BUG);

    float left_tick = -1.0f;
    float right_tick = -1.0f;
    for (int i = 0; i < BAT_ECHO_QUEUE_TICKS; i++) {
        if (env.echo_queue[i].tick < 0) continue;
        float left_energy = 0.0f;
        float right_energy = 0.0f;
        for (int bin = 0; bin < BAT_FREQ_BINS; bin++) {
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

    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bug_x = 42.0f;
    env.bug_y = 20.0f;
    env.bug_vx = -16.0f;
    env.bug_vy = 0.0f;
    env.bat_heading = 0.0f;
    memset(env.observations, 0, BAT_OBS_SIZE * sizeof(float));
    bat_clear_echo_queue(&env);
    env.tick = 0;

    ChirpEvent chirp = {
        .x = env.bat_x,
        .y = env.bat_y,
        .start_freq = 0.5f,
        .end_freq = 0.5f,
        .duration = bat_chirp_duration_seconds(0.0f),
        .birth_tick = 0,
        .active = 1,
    };
    bat_schedule_echo(&env, &chirp, 0.0f, 0.5f,
        env.bug_x, env.bug_y, env.bug_vx, env.bug_vy, 8.0f, BAT_ECHO_BUG);

    env.tick = 27;
    compute_observations(&env);

    float low_energy = 0.0f;
    float high_energy = 0.0f;
    for (int i = 0; i < BAT_FREQ_BINS; i++) {
        float energy = env.observations[BAT_LEFT_FREQ_OFFSET + i]
            + env.observations[BAT_RIGHT_FREQ_OFFSET + i];
        if (i < BAT_FREQ_BINS / 2) {
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

    env.bat_x = env.width - env.bat_radius - 0.1f;
    env.bat_y = env.height * 0.5f;
    env.bat_heading = 0.0f;
    env.bat_vx = env.bat_max_speed;
    env.bat_vy = 0.0f;
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

    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
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

    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 40.0f;
    env.bug_y = 20.0f;
    env.prev_bug_dist = 25.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;

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
    env.chirp_cost = 0.0f;
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.bat_heading = 0.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.actions[0] = BAT_BRAKE;
    env.actions[1] = BAT_TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    float forward = env.bat_vx * cosf(env.bat_heading) + env.bat_vy * sinf(env.bat_heading);
    ASSERT_TRUE(forward >= -0.0001f);
    ASSERT_TRUE(env.observations[BAT_FORWARD_SPEED_OBS] >= -0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_reset_starts_with_forward_stall_speed(void) {
    Bat env = make_test_env();
    c_reset(&env);

    float forward = env.bat_vx * cosf(env.bat_heading) + env.bat_vy * sinf(env.bat_heading);
    ASSERT_TRUE(forward >= 0.19f * env.bat_max_speed);
    ASSERT_FLOAT_NEAR(env.observations[BAT_FORWARD_SPEED_OBS], forward / env.bat_max_speed, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_brake_clamps_to_forward_stall_speed(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.bat_heading = 0.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.actions[0] = BAT_BRAKE;
    env.actions[1] = BAT_TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    float forward = env.bat_vx * cosf(env.bat_heading) + env.bat_vy * sinf(env.bat_heading);
    ASSERT_TRUE(forward >= 0.19f * env.bat_max_speed);
    ASSERT_TRUE(env.bat_x > 20.0f);

    free_allocated(&env);
    return 0;
}

static int test_bat_velocity_is_locked_to_heading(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.bat_heading = 0.0f;
    env.bat_vx = -env.bat_max_speed * 0.5f;
    env.bat_vy = 3.0f;
    env.actions[0] = BAT_NOOP;
    env.actions[1] = BAT_TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    float forward = env.bat_vx * cosf(env.bat_heading) + env.bat_vy * sinf(env.bat_heading);
    float lateral = env.bat_vx * -sinf(env.bat_heading) + env.bat_vy * cosf(env.bat_heading);
    ASSERT_TRUE(forward >= -0.0001f);
    ASSERT_FLOAT_NEAR(lateral, 0.0f, 0.0001f);
    ASSERT_TRUE(env.observations[BAT_FORWARD_SPEED_OBS] >= -0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_zero_speed_recovers_to_forward_arc(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.bat_heading = 0.25f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.actions[0] = BAT_NOOP;
    env.actions[1] = BAT_TURN_LEFT;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    float start_x = env.bat_x;
    float start_y = env.bat_y;
    c_step(&env);

    float forward = env.bat_vx * cosf(env.bat_heading) + env.bat_vy * sinf(env.bat_heading);
    ASSERT_TRUE(forward >= 0.19f * env.bat_max_speed);
    ASSERT_TRUE(bat_dist(start_x, start_y, env.bat_x, env.bat_y) > 0.0f);
    ASSERT_TRUE(fabsf(env.bat_heading - 0.25f) > 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_turn_rate_scales_with_forward_speed(void) {
    Bat env = make_test_env();
    c_reset(&env);

    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    env.bat_heading = 0.0f;
    env.bat_vx = env.bat_max_speed * 0.5f;
    env.bat_vy = 0.0f;
    env.actions[0] = BAT_NOOP;
    env.actions[1] = BAT_TURN_RIGHT;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 0.0f;

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.bat_turn_velocity, env.bat_turn_rate * 0.5f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.bat_heading, env.bat_turn_rate * 0.5f * BAT_TICK_RATE, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bat_speed_action_space_has_no_strafe(void) {
    ASSERT_TRUE(BAT_MOVE_ACTIONS == 3);
    ASSERT_TRUE(BAT_NOOP == 0);
    ASSERT_TRUE(BAT_THRUST_FORWARD == 1);
    ASSERT_TRUE(BAT_BRAKE == 2);
    return 0;
}

static int test_chirp_ring_physical_ordering(void) {
    float duration = bat_chirp_duration_seconds(1.0f);
    float outer = bat_chirp_ring_radius(1.0f, 0.0f, duration, 100.0f);
    float inner = bat_chirp_ring_radius(1.0f, 1.0f, duration, 100.0f);

    ASSERT_TRUE(outer > inner);
    ASSERT_FLOAT_NEAR(outer, 100.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(inner, 100.0f * (1.0f - duration), 0.0001f);

    return 0;
}

static int test_chirp_color_maps_low_to_red_high_to_blue(void) {
    BatColor low = bat_freq_color(0.0f, 1.0f);
    BatColor mid = bat_freq_color(0.5f, 1.0f);
    BatColor high = bat_freq_color(1.0f, 1.0f);

    ASSERT_TRUE(low.r > low.b);
    ASSERT_TRUE(high.b > high.r);
    ASSERT_TRUE(mid.g >= low.g);
    ASSERT_TRUE(mid.g >= high.g);

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
    ASSERT_TRUE(bat_try_emit_chirp(&env));
    ASSERT_TRUE(!bat_try_emit_chirp(&env));

    env.tick += 12;
    ASSERT_TRUE(bat_try_emit_chirp(&env));

    free_allocated(&env);
    return 0;
}

static void test_place_safe_stationary_scene(Bat* env) {
    env->num_obstacles = 0;
    env->bat_x = 20.0f;
    env->bat_y = 20.0f;
    env->bat_vx = 0.0f;
    env->bat_vy = 0.0f;
    env->bat_heading = 0.0f;
    env->bug_x = 48.0f;
    env->bug_y = 48.0f;
    env->bug_vx = 0.0f;
    env->bug_vy = 0.0f;
    env->prev_bug_dist = bat_dist(env->bat_x, env->bat_y, env->bug_x, env->bug_y);
}

static void test_set_emit_chirp_action(Bat* env) {
    env->actions[0] = BAT_NOOP;
    env->actions[1] = BAT_TURN_NONE;
    env->actions[2] = 0.0f;
    env->actions[3] = 7.0f;
    env->actions[4] = 1.0f;
    env->actions[5] = 1.0f;
}

static int test_valid_chirp_gets_reward_without_legacy_cost(void) {
    Bat env = make_test_env();
    c_reset(&env);
    test_place_safe_stationary_scene(&env);
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.bug_echo_reward_scale = 0.0f;
    env.chirp_cost = 10.0f;
    env.valid_chirp_reward = 0.0005f;
    env.early_chirp_penalty = 0.0020f;
    test_set_emit_chirp_action(&env);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], env.valid_chirp_reward, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted_episode == 1);

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
    env.chirp_cost = 0.0f;
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
    ASSERT_TRUE(env.chirps_emitted_episode == 1);

    free_allocated(&env);
    return 0;
}

static int test_chirp_before_echo_window_clears_gets_overlap_penalty(void) {
    Bat env = make_test_env();
    c_reset(&env);
    test_place_safe_stationary_scene(&env);
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.bug_echo_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.valid_chirp_reward = 0.0005f;
    env.early_chirp_penalty = 0.0020f;
    env.chirp_overlap_penalty = 0.0040f;
    env.chirp_cooldown_ticks = 1;
    env.max_chirp_age_ticks = 8;
    test_set_emit_chirp_action(&env);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0], env.valid_chirp_reward, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted_episode == 1);
    ASSERT_TRUE(env.chirps_overlapped == 0);

    test_place_safe_stationary_scene(&env);
    test_set_emit_chirp_action(&env);
    c_step(&env);

    ASSERT_FLOAT_NEAR(env.terminals[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.rewards[0],
        env.valid_chirp_reward - env.chirp_overlap_penalty, 0.0001f);
    ASSERT_TRUE(env.chirps_emitted_episode == 2);
    ASSERT_TRUE(env.chirps_overlapped == 1);

    free_allocated(&env);
    return 0;
}

static int test_reflection_arrives_at_two_way_travel_time(void) {
    float sound_speed = 100.0f;
    float distance = 25.0f;
    float echo_time = bat_echo_time_seconds(distance, sound_speed);

    ASSERT_FLOAT_NEAR(echo_time, 0.5f, 0.0001f);
    ASSERT_TRUE(bat_echo_is_arriving(echo_time, echo_time + 0.005f, 0.02f));
    ASSERT_TRUE(!bat_echo_is_arriving(echo_time, echo_time + 0.050f, 0.02f));

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
    ASSERT_TRUE(BAT_OBS_SIZE == 40);
    ASSERT_TRUE(BAT_FREQ_BINS == 16);
    ASSERT_TRUE(BAT_LEFT_FREQ_OFFSET == 0);
    ASSERT_TRUE(BAT_RIGHT_FREQ_OFFSET == 16);
    ASSERT_TRUE(BAT_CHIRP_AGE_OBS == 32);
    ASSERT_TRUE(BAT_CHIRP_COOLDOWN_OBS == 33);
    ASSERT_TRUE(BAT_CHIRP_START_OBS == 34);
    ASSERT_TRUE(BAT_CHIRP_END_OBS == 35);
    ASSERT_TRUE(BAT_CHIRP_DURATION_OBS == 36);
    ASSERT_TRUE(BAT_CHIRPS_USED_OBS == 37);
    ASSERT_TRUE(BAT_FORWARD_SPEED_OBS == 38);
    ASSERT_TRUE(BAT_TURN_RATE_OBS == 39);
    return 0;
}

static int test_no_chirp_produces_silent_frequency_bins(void) {
    Bat env = make_test_env();
    c_reset(&env);

    ASSERT_FLOAT_NEAR(test_sum_obs(&env, BAT_LEFT_FREQ_OFFSET, BAT_FREQ_BINS), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(test_sum_obs(&env, BAT_RIGHT_FREQ_OFFSET, BAT_FREQ_BINS), 0.0f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_observations_stay_normalized_after_chirp(void) {
    Bat env = make_test_env();
    env.max_steps = 1000;
    c_reset(&env);

    ASSERT_FLOAT_NEAR(env.observations[BAT_CHIRP_AGE_OBS], 1.0f, 0.0001f);
    for (int i = 0; i < BAT_OBS_SIZE; i++) {
        ASSERT_TRUE(env.observations[i] >= -1.0f);
        ASSERT_TRUE(env.observations[i] <= 1.0f);
    }

    env.actions[0] = BAT_NOOP;
    env.actions[1] = BAT_TURN_NONE;
    env.actions[2] = 0.0f;
    env.actions[3] = 7.0f;
    env.actions[4] = 1.0f;
    env.actions[5] = 1.0f;
    c_step(&env);

    float age_denom = bat_chirp_age_norm_denominator(&env);
    ASSERT_FLOAT_NEAR(env.observations[BAT_CHIRP_AGE_OBS], 1.0f / age_denom, 0.0001f);
    for (int i = 0; i < BAT_OBS_SIZE; i++) {
        ASSERT_TRUE(env.observations[i] >= -1.0f);
        ASSERT_TRUE(env.observations[i] <= 1.0f);
    }

    free_allocated(&env);
    return 0;
}

static int test_curriculum_level_zero_starts_close_with_no_obstacles(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_enabled = 1;
    env.curriculum_start_obstacles = 0;
    env.curriculum_max_obstacles = 3;
    env.curriculum_obstacle_step = 1;
    env.curriculum_start_bug_distance = 12.0f;
    env.curriculum_max_bug_distance = 40.0f;
    env.curriculum_bug_distance_step = 6.0f;
    c_reset(&env);

    ASSERT_TRUE(env.num_obstacles == 0);
    ASSERT_TRUE(bat_dist(env.bat_x, env.bat_y, env.bug_x, env.bug_y) <= 14.0f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_adds_first_obstacle_after_level_zero(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_enabled = 1;
    env.curriculum_start_obstacles = 0;
    env.curriculum_max_obstacles = 3;
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
    env.curriculum_enabled = 1;
    env.curriculum_start_obstacles = 1;
    env.curriculum_max_obstacles = 3;
    env.curriculum_obstacle_step = 1;
    env.curriculum_start_bug_distance = 12.0f;
    env.curriculum_max_bug_distance = 40.0f;
    env.curriculum_bug_distance_step = 6.0f;
    c_reset(&env);
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_TRUE(env.curriculum_level == 1);
    ASSERT_TRUE(env.num_obstacles == 2);
    ASSERT_TRUE(bat_dist(env.bat_x, env.bat_y, env.bug_x, env.bug_y) <= 20.0f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_waits_for_required_catches(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_enabled = 1;
    env.curriculum_start_obstacles = 1;
    env.curriculum_max_obstacles = 3;
    env.curriculum_obstacle_step = 1;
    env.curriculum_start_bug_distance = 12.0f;
    env.curriculum_max_bug_distance = 40.0f;
    env.curriculum_bug_distance_step = 6.0f;
    env.curriculum_successes_per_level = 2;
    c_reset(&env);
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
    env.bug_x = 20.5f;
    env.bug_y = 20.0f;

    c_step(&env);

    ASSERT_TRUE(env.curriculum_level == 0);
    ASSERT_TRUE(env.curriculum_successes_at_level == 1);

    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
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
    env.curriculum_enabled = 1;
    env.curriculum_initial_level = 4;
    env.curriculum_start_obstacles = 1;
    env.curriculum_max_obstacles = 3;
    env.curriculum_obstacle_step = 2;
    env.curriculum_start_bug_distance = 8.0f;
    env.curriculum_max_bug_distance = 56.0f;
    env.curriculum_bug_distance_step = 4.0f;
    c_reset(&env);

    ASSERT_TRUE(env.curriculum_level == 4);
    ASSERT_TRUE(env.num_obstacles == 3);
    float dist = bat_dist(env.bat_x, env.bat_y, env.bug_x, env.bug_y);
    ASSERT_TRUE(dist >= 20.0f);
    ASSERT_TRUE(dist <= 28.0f);

    free_allocated(&env);
    return 0;
}

static int test_curriculum_initial_level_does_not_reset_progress(void) {
    Bat env = make_test_env();
    env.num_obstacles = 3;
    env.curriculum_enabled = 1;
    env.curriculum_initial_level = 2;
    env.curriculum_start_obstacles = 1;
    env.curriculum_max_obstacles = 3;
    env.curriculum_obstacle_step = 1;
    env.curriculum_successes_per_level = 1;
    env.curriculum_start_bug_distance = 8.0f;
    env.curriculum_max_bug_distance = 56.0f;
    env.curriculum_bug_distance_step = 4.0f;
    c_reset(&env);
    env.bat_x = 20.0f;
    env.bat_y = 20.0f;
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

    env.bug_x = env.width - env.bug_radius + 0.1f;
    env.bug_y = env.height * 0.5f;
    env.bug_vx = 3.0f;
    env.bug_vy = 1.0f;
    bat_update_bug(&env, 0.0f);
    ASSERT_TRUE(env.bug_x == env.width - env.bug_radius);
    ASSERT_TRUE(env.bug_vx < 0.0f);
    ASSERT_TRUE(env.bug_vy == 1.0f);

    env.bug_x = env.width * 0.5f;
    env.bug_y = env.bug_radius - 0.1f;
    env.bug_vx = 2.0f;
    env.bug_vy = -4.0f;
    bat_update_bug(&env, 0.0f);
    ASSERT_TRUE(env.bug_y == env.bug_radius);
    ASSERT_TRUE(env.bug_vx == 2.0f);
    ASSERT_TRUE(env.bug_vy > 0.0f);

    free_allocated(&env);
    return 0;
}

static int test_chirp_echo_arrives_after_two_way_travel_not_immediately(void) {
    Bat env = make_test_env();
    env.num_obstacles = 0;
    env.sound_speed = 60.0f;
    env.max_echo_range = 128.0f;
    c_reset(&env);

    env.bat_x = 32.0f;
    env.bat_y = 32.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bat_heading = 0.0f;
    env.bug_x = 38.0f;
    env.bug_y = 32.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    compute_observations(&env);

    env.actions[0] = BAT_NOOP;
    env.actions[1] = BAT_TURN_NONE;
    env.actions[2] = 7;
    env.actions[3] = 7;
    env.actions[4] = 0;
    env.actions[5] = 1;
    c_step(&env);

    for (int i = 0; i < 6; i++) {
        ASSERT_FLOAT_NEAR(test_sum_obs(&env, BAT_LEFT_FREQ_OFFSET, BAT_FREQ_BINS), 0.0f, 0.0001f);
        ASSERT_FLOAT_NEAR(test_sum_obs(&env, BAT_RIGHT_FREQ_OFFSET, BAT_FREQ_BINS), 0.0f, 0.0001f);
        env.actions[5] = 0;
        c_step(&env);
    }

    float max_energy = 0.0f;
    for (int i = 0; i < 32; i++) {
        float energy = test_sum_obs(&env, BAT_LEFT_FREQ_OFFSET, BAT_FREQ_BINS)
            + test_sum_obs(&env, BAT_RIGHT_FREQ_OFFSET, BAT_FREQ_BINS);
        if (energy > max_energy) max_energy = energy;
        c_step(&env);
    }

    ASSERT_TRUE(max_energy > 0.01f);

    free_allocated(&env);
    return 0;
}

static int test_frequency_bin_energy_sums_and_caps(void) {
    Bat env = make_test_env();
    memset(env.observations, 0, BAT_OBS_SIZE * sizeof(float));

    bat_add_freq_energy(&env, BAT_LEFT_FREQ_OFFSET, 1.0f, 0.75f);
    bat_add_freq_energy(&env, BAT_LEFT_FREQ_OFFSET, 1.0f, 0.75f);
    bat_add_freq_energy(&env, BAT_RIGHT_FREQ_OFFSET, 0.0f, 0.35f);

    ASSERT_FLOAT_NEAR(env.observations[BAT_LEFT_FREQ_OFFSET + BAT_FREQ_BINS - 1], 1.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[BAT_RIGHT_FREQ_OFFSET], 0.35f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bug_echo_reward_is_added_when_bug_echo_is_closer(void) {
    Bat env = make_test_env();
    c_reset(&env);
    env.bug_echo_reward_scale = 0.05f;
    env.last_bug_echo_path = 20.0f;
    env.last_bug_echo_bat_x = 8.0f;
    env.last_bug_echo_bat_y = 10.0f;
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.bat_x = 10.0f;
    env.bat_y = 10.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    bat_clear_echo_queue(&env);
    bat_add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 15.0f, BAT_ECHO_BUG);

    c_step(&env);

    ASSERT_TRUE(env.rewards[0] > 0.002f);
    ASSERT_FLOAT_NEAR(env.observations[BAT_LEFT_FREQ_OFFSET + 8], 0.6f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_bug_echo_reward_requires_bat_displacement(void) {
    Bat env = make_test_env();
    c_reset(&env);
    env.bug_echo_reward_scale = 0.05f;
    env.last_bug_echo_path = 20.0f;
    env.last_bug_echo_bat_x = 10.0f;
    env.last_bug_echo_bat_y = 10.0f;
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.bat_x = 10.0f;
    env.bat_y = 10.0f;
    env.bat_heading = 0.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    bat_clear_echo_queue(&env);
    bat_add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 15.0f, BAT_ECHO_BUG);

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
    env.last_bug_echo_bat_x = 8.0f;
    env.last_bug_echo_bat_y = 10.0f;
    env.step_cost = 0.0f;
    env.progress_reward_scale = 0.0f;
    env.chirp_cost = 0.0f;
    env.bat_x = 10.0f;
    env.bat_y = 10.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    bat_clear_echo_queue(&env);
    bat_add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 25.0f, BAT_ECHO_BUG);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.rewards[0], -0.0003125f, 0.0001f);
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
    env.chirp_cost = 0.0f;
    env.bat_x = 10.0f;
    env.bat_y = 10.0f;
    env.bat_vx = 0.0f;
    env.bat_vy = 0.0f;
    env.bug_vx = 0.0f;
    env.bug_vy = 0.0f;
    env.bug_x = 50.0f;
    env.bug_y = 50.0f;
    bat_clear_echo_queue(&env);
    bat_add_echo_event(&env, 0, 1.0f, 0.5f, 0.6f, 15.0f, BAT_ECHO_STATIC);

    c_step(&env);

    ASSERT_FLOAT_NEAR(env.rewards[0], 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR(env.observations[BAT_LEFT_FREQ_OFFSET + 8], 0.6f, 0.0001f);

    free_allocated(&env);
    return 0;
}

static int test_quadrant(float x, float y, float width, float height) {
    int east = x >= width * 0.5f;
    int south = y >= height * 0.5f;
    return south * 2 + east;
}

static int test_spawns_use_different_random_quadrants(void) {
    Bat env = make_test_env();
    int seen_bat[4] = {0};
    int seen_bug[4] = {0};
    int bat_quadrants = 0;
    int bug_quadrants = 0;

    for (int i = 0; i < 48; i++) {
        c_reset(&env);
        int bq = test_quadrant(env.bat_x, env.bat_y, env.width, env.height);
        int gq = test_quadrant(env.bug_x, env.bug_y, env.width, env.height);
        ASSERT_TRUE(bq != gq);
        if (!seen_bat[bq]) {
            seen_bat[bq] = 1;
            bat_quadrants += 1;
        }
        if (!seen_bug[gq]) {
            seen_bug[gq] = 1;
            bug_quadrants += 1;
        }
    }

    ASSERT_TRUE(bat_quadrants >= 3);
    ASSERT_TRUE(bug_quadrants >= 3);

    free_allocated(&env);
    return 0;
}

static int test_spawns_keep_minimum_separation_and_avoid_obstacles(void) {
    Bat env = make_test_env();
    float min_sep = 20.0f;

    for (int reset = 0; reset < 32; reset++) {
        c_reset(&env);
        ASSERT_TRUE(bat_dist(env.bat_x, env.bat_y, env.bug_x, env.bug_y) >= min_sep);
        for (int i = 0; i < env.num_obstacles; i++) {
            ASSERT_TRUE(!bat_circle_rect_collision(env.bat_x, env.bat_y, env.bat_radius + 1.0f,
                env.obstacle_x[i], env.obstacle_y[i], env.obstacle_w[i], env.obstacle_h[i]));
            ASSERT_TRUE(!bat_circle_rect_collision(env.bug_x, env.bug_y, env.bug_radius + 1.0f,
                env.obstacle_x[i], env.obstacle_y[i], env.obstacle_w[i], env.obstacle_h[i]));
        }
    }

    free_allocated(&env);
    return 0;
}

static int test_obstacles_move_substantially_across_resets(void) {
    Bat env = make_test_env();
    c_reset(&env);
    float first_x = env.obstacle_x[0];
    float first_y = env.obstacle_y[0];
    float max_delta = 0.0f;

    for (int i = 0; i < 32; i++) {
        c_reset(&env);
        float delta = bat_dist(first_x, first_y, env.obstacle_x[0], env.obstacle_y[0]);
        if (delta > max_delta) max_delta = delta;
    }

    ASSERT_TRUE(max_delta > 16.0f);

    free_allocated(&env);
    return 0;
}

static int test_obstacles_are_small_enough_for_trainability(void) {
    Bat env = make_test_env();

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
    if (test_chirp_budget_observation_tracks_used_chirps()) return 1;
    if (test_chirp_budget_stays_fixed_with_curriculum_level()) return 1;
    if (test_chirping_after_budget_terminates_with_penalty()) return 1;
    if (test_chirp_efficiency_scores_low_usage_above_full_budget()) return 1;
    if (test_chirp_perf_uses_fixed_fifteen_chirp_reference()) return 1;
    if (test_success_reward_includes_chirp_efficiency_bonus()) return 1;
    if (test_chirp_budget_logs_ratios_for_wandb()) return 1;
    if (test_curriculum_perf_logs_distance_and_obstacle_difficulty_components()) return 1;
    if (test_budget_difficulty_uses_hard_edge_below_six_chirps()) return 1;
    if (test_perf_composes_base_perf_curriculum_difficulty_and_chirp_perf()) return 1;
    if (test_chirp_tempo_logs_far_and_near_rates()) return 1;
    if (test_left_right_echo_asymmetry()) return 1;
    if (test_default_sound_speed_allows_one_tick_interaural_delay()) return 1;
    if (test_echo_scheduling_uses_tick_bucket_accumulator()) return 1;
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
    if (test_chirp_ring_physical_ordering()) return 1;
    if (test_chirp_color_maps_low_to_red_high_to_blue()) return 1;
    if (test_chirp_cooldown_accepts_only_after_delay()) return 1;
    if (test_valid_chirp_gets_reward_without_legacy_cost()) return 1;
    if (test_early_chirp_gets_penalty_and_emits_nothing()) return 1;
    if (test_chirp_before_echo_window_clears_gets_overlap_penalty()) return 1;
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
    if (test_frequency_bin_energy_sums_and_caps()) return 1;
    if (test_bug_echo_reward_is_added_when_bug_echo_is_closer()) return 1;
    if (test_bug_echo_reward_requires_bat_displacement()) return 1;
    if (test_bug_echo_reward_penalizes_farther_bug_echo_weakly()) return 1;
    if (test_static_echo_does_not_get_bug_echo_reward()) return 1;
    if (test_spawns_use_different_random_quadrants()) return 1;
    if (test_spawns_keep_minimum_separation_and_avoid_obstacles()) return 1;
    if (test_obstacles_move_substantially_across_resets()) return 1;
    if (test_obstacles_are_small_enough_for_trainability()) return 1;

    printf("bat core tests passed\n");
    return 0;
}
