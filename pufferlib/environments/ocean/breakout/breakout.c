#include "breakout.h"

int main() {
    int frameskip = 4;
    float dt = 1.0/60.0;
    int num_agents = 1;
    int width = 576;
    int height = 330;
    int ball_width = 6;
    int ball_height = 6;
    int num_bricks_rows = 6;
    int num_bricks_cols = 18;
    int brick_width = width / num_bricks_cols;
    int brick_height = 12;
    int paddle_width = 62;
    int paddle_height = 8;
    int obs_size = 2 + 2 + num_bricks_rows*num_bricks_cols + 1;

    CBreakout* env = allocate_cbreakout(dt, frameskip, num_agents,
        width, height, ball_width, ball_height, brick_width,
        brick_height, obs_size, num_bricks_rows,
        num_bricks_cols);
 
    Client* client = make_client(width, height, paddle_width,
        paddle_height, ball_width, ball_height);
 
    reset(env);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env->actions[0] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env->actions[0] = 1;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env->actions[0] = 2;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env->actions[0] = 3;

        step(env);
        render(client, env);
    }
    close_client(client);
    free_allocated_cbreakout(env);
    return 0;
}

