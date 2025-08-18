#include "g2048.h"
#include "puffernet.h"

int main() {
    srand(time(NULL));
    Game env;
    unsigned char observations[SIZE * SIZE] = {0};
    unsigned char terminals[1] = {0};
    int actions[1] = {0};
    float rewards[1] = {0};

    env.observations = observations;
    env.terminals = terminals;
    env.actions = actions;
    env.rewards = rewards;

    Weights* weights = load_weights("resources/g2048/g2048_weights.bin", 134917);
    int logit_sizes[1] = {4};
    LinearLSTM* net = make_linearlstm(weights, 1, 16, logit_sizes, 1);
    c_reset(&env);
    c_render(&env);

    // Main game loop
    int frame = 0;
    while (!WindowShouldClose()) {
        c_render(&env);
        frame++;

        int action = 0;
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyPressed(KEY_W) || IsKeyPressed(KEY_UP)) action = UP;
            else if (IsKeyPressed(KEY_S) || IsKeyPressed(KEY_DOWN)) action = DOWN;
            else if (IsKeyPressed(KEY_A) || IsKeyPressed(KEY_LEFT)) action = LEFT;
            else if (IsKeyPressed(KEY_D) || IsKeyPressed(KEY_RIGHT)) action = RIGHT;
            env.actions[0] = action - 1;
        } else if (frame % 10 != 0) {
            continue;
        } else {
            action = 1;
            for (int i = 0; i < 16; i++) {
                net->obs[i] = env.observations[i];
            }
            forward_linearlstm(net, net->obs, env.actions);
        }

        if (action != 0) {
            c_step(&env);
        }
    }

    free_linearlstm(net);
    c_close(&env);
    printf("Game Over! Final Max Tile: %d\n", env.score);
    return 0;
}
