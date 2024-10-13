#include "breakout.h"
#include "puffernet.h"

int main() {
    Weights* weights = load_weights("resources/breakout_weights.bin", 148101);
    LinearLSTM* net = make_linearlstm(weights, 1, 119, 4);

    Breakout env = {
        .frameskip = 1,
        .width = 576,
        .height = 330,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
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
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 1;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;
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
    return 0;
}

