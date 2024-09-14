#include "pong.h"

int main() {
    int width = 500;
    int height = 640;
    int paddle_width = 20;
    int paddle_height = 70;
    int ball_width = 10;
    int ball_height = 15;
    int paddle_speed = 8;
    int ball_initial_speed_x = 10;
    int ball_initial_speed_y = 1;
    int ball_speed_y_increment = 3;
    int ball_max_speed_y = 13;
    int max_score = 21;

    CMyPong* env = allocate_cmy_pong(width, height,
        paddle_width, paddle_height, ball_width, ball_height,
        paddle_speed, ball_initial_speed_x, ball_initial_speed_y,
        ball_max_speed_y, ball_speed_y_increment, max_score);

    Client* client = make_client(width, height, paddle_width,
        paddle_height, ball_width, ball_height);

    reset(env);

    while (!WindowShouldClose()) {
        // User can take control of the paddle
        env->actions[0] = 0;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env->actions[0] = 1;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env->actions[0] = 2;

        step(env);
        render(client, env);
    }
    close_client(client);
    free_allocated_cmy_pong(env);
    return 0;
}

