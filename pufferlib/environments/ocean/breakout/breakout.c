#include "breakout.h"

int main() {
    CBreakout env = {
        .frameskip = 4,
        .width = 576,
        .height = 330,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 6,
        .ball_height = 7,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
    };
    allocate(&env);
    reset(&env);
 
    Client* client = make_client(&env);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env.actions[0] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 1;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;

        step(&env);
        render(client, &env);
    }
    close_client(client);
    free_allocated(&env);
    return 0;
}

