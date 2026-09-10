#include "space_onslaught.h"

int main(void) {
    float observations[SO_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    SpaceOnslaught env = {
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
        .num_agents = 1,
        .rng = 1,
        .width = 448,
        .height = 576,
        .frameskip = 1,
        .alien_width = 24,
        .alien_height = 16,
        .alien_spacing_x = 12,
        .alien_spacing_y = 16,
        .edge_margin = 16,
        .alien_step_x = 8,
        .alien_step_down = 16,
        .base_move_interval = 30,
        .min_move_interval = 3,
        .player_width = 24,
        .player_height = 16,
        .player_speed = 180,
        .player_bullet_speed = 420,
        .alien_bullet_speed = 200,
        .initial_lives = 3,
        .alien_fire_prob = 0.02f,
    };

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        int left = IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A);
        int right = IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D);
        int fire = IsKeyDown(KEY_SPACE);
        if (left && right) {
            left = 0;
            right = 0;
        }
        if (left && fire) {
            actions[0] = ACTION_LEFT_FIRE;
        } else if (right && fire) {
            actions[0] = ACTION_RIGHT_FIRE;
        } else if (left) {
            actions[0] = ACTION_LEFT;
        } else if (right) {
            actions[0] = ACTION_RIGHT;
        } else if (fire) {
            actions[0] = ACTION_FIRE;
        } else {
            actions[0] = ACTION_NOOP;
        }
        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    return 0;
}
