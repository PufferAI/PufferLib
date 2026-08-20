#include <stdio.h>
#include <time.h>
#include "flappy.h"

int main(void) {
    Flappy env = {
        .width = 420,
        .height = 640,
        .max_steps = 4096,
        .gravity = 0.45f,
        .flap_velocity = -7.5f,
        .pipe_speed = 3.0f,
        .pipe_gap = 190.0f,
        .pipe_width = 58.0f,
        .pipe_spacing = 220.0f,
        .first_pipe_x = 220.0f,
        .bird_x = 96.0f,
        .bird_radius = 14.0f,
        .alive_reward = 0.01f,
        .pass_reward = 1.0f,
        .crash_reward = -1.0f,
        .center_reward = 0.03f,
        .rng = (unsigned int)time(NULL),
    };

    float observations[FLAPPY_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    init(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_UP) ? FLAPPY_FLAP : FLAPPY_NOOP;
        c_step(&env);
        c_render(&env);
    }

    c_close(&env);
    return 0;
}

