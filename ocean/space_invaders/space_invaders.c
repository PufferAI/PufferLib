#include <stdio.h>
#include <time.h>
#include "space_invaders.h"

void demo() {
    SpaceInvaders env = {
        .width = 600,
        .height = 480,
        .frameskip = 1,
        .player_speed = 4,
        .player_bullet_speed = 8,
        .enemy_bullet_speed = 3,
        .formation_dx = 8,
        .formation_dy = 12,
        .formation_start_interval = 30,
        .enemy_fire_interval = 45,
        .invader_w = 24,
        .invader_h = 16,
        .invader_spacing_x = 16,
        .invader_spacing_y = 12,
        .formation_margin_x = 40,
        .formation_margin_y = 50,
        .player_w = 32,
        .player_h = 16,
        .player_y_offset = 30,
        .bullet_w = 4,
        .bullet_h = 10,
        .max_lives = 3,
    };
    env.rng = (unsigned int)time(NULL);
    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);

    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        env.actions[0] = SI_NOOP;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = SI_LEFT;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = SI_RIGHT;
        if (IsKeyDown(KEY_SPACE) || IsKeyDown(KEY_UP)) env.actions[0] = SI_FIRE;
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    close_client(env.client);
}

int main(void) {
    demo();
    return 0;
}
