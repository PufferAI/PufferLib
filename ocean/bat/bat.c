#include <time.h>
#include "bat.h"

void demo() {
    Bat env = {
        .frameskip = 1,
        .width = 64,
        .height = 64,
        .num_obstacles = 3,
        .bat_radius = 2.0f,
        .bug_radius = 1.5f,
        .bat_max_speed = 12.0f,
        .bat_accel = 30.0f,
        .bat_turn_rate = BAT_PI,
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
        .rng = (unsigned int)time(NULL),
    };
    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);

    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        memset(env.actions, 0, sizeof(float) * BAT_NUM_ACTIONS);
        if (IsKeyDown(KEY_W)) env.actions[0] = BAT_THRUST_FORWARD;
        if (IsKeyDown(KEY_S)) env.actions[0] = BAT_BRAKE;
        if (IsKeyDown(KEY_A)) env.actions[0] = BAT_STRAFE_LEFT;
        if (IsKeyDown(KEY_D)) env.actions[0] = BAT_STRAFE_RIGHT;
        if (IsKeyDown(KEY_LEFT)) env.actions[1] = BAT_TURN_LEFT;
        if (IsKeyDown(KEY_RIGHT)) env.actions[1] = BAT_TURN_RIGHT;
        env.actions[2] = 0;
        env.actions[3] = 7;
        env.actions[4] = 1;
        env.actions[5] = IsKeyDown(KEY_SPACE) ? 1.0f : 0.0f;
        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
}

int main() {
    demo();
    return 0;
}

