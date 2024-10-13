#include "pong.h"
#include "puffernet.h"

int main() {
    Weights* weights = load_weights("resources/pong_weights.bin", 133764);
    LinearLSTM* net = make_linearlstm(weights, 1, 8, 3);

    Pong env = {
        .width = 500,
        .height = 640,
        .paddle_width = 20,
        .paddle_height = 70,
        //.ball_width = 10,
        //.ball_height = 15,
        .ball_width = 32,
        .ball_height = 32,
        .paddle_speed = 8,
        .ball_initial_speed_x = 10,
        .ball_initial_speed_y = 1,
        .ball_speed_y_increment = 3,
        .ball_max_speed_y = 13,
        .max_score = 21,
        .frameskip = 1,
    };
    allocate(&env);

    Client* client = make_client(&env);

    reset(&env);
    while (!WindowShouldClose()) {
        // User can take control of the paddle
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 1;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 2;
        } else {
            forward_linearlstm(net, env.observations, env.actions);
        }

        step(&env);
        render(client, &env);
    }
    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(client);
}

