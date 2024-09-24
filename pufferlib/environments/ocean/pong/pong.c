#include "pong.h"

int main() {
    CPong env = {
        .width = 500,
        .height = 640,
        .paddle_width = 20,
        .paddle_height = 70,
        .ball_width = 10,
        .ball_height = 15,
        .paddle_speed = 8,
        .ball_initial_speed_x = 10,
        .ball_initial_speed_y = 1,
        .ball_speed_y_increment = 3,
        .ball_max_speed_y = 13,
        .max_score = 21,
        .frameskip = 4,
    };
    allocate(&env);

    Client* client = make_client(&env);

    reset(&env);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env.actions[0] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 1;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 2;

        step(&env);
        render(client, &env);
    }
    close_client(client);
    free_allocated(&env);
}

