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
        .range_bins_per_ear = BAT_RANGE_BINS,
        .doppler_bins_per_ear = BAT_DOPPLER_BINS,
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
    compute_observations(&env);

    float left_energy = 0.0f;
    float right_energy = 0.0f;
    for (int i = 0; i < BAT_RANGE_BINS; i++) {
        left_energy += env.observations[BAT_LEFT_RANGE_OFFSET + i];
        right_energy += env.observations[BAT_RIGHT_RANGE_OFFSET + i];
    }

    ASSERT_TRUE(left_energy > right_energy);

    free_allocated(&env);
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
    env.bug_vx = -3.0f;
    env.bug_vy = 0.0f;
    compute_observations(&env);

    float doppler = 0.0f;
    for (int i = 0; i < BAT_DOPPLER_BINS; i++) {
        doppler += env.observations[BAT_LEFT_DOPPLER_OFFSET + i];
        doppler += env.observations[BAT_RIGHT_DOPPLER_OFFSET + i];
    }

    ASSERT_TRUE(doppler > 0.0f);

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

int main(void) {
    if (test_chirp_metadata_and_observation_size()) return 1;
    if (test_left_right_echo_asymmetry()) return 1;
    if (test_doppler_sign_for_approaching_bug()) return 1;
    if (test_wall_collision_is_terminal_minus_one()) return 1;
    if (test_catch_bug_is_terminal_plus_one()) return 1;
    if (test_progress_reward_sign()) return 1;

    printf("bat core tests passed\n");
    return 0;
}

