#include "squared.h"
#include "puffernet.h"

int main() {
    //Weights* weights = load_weights("resources/pong_weights.bin", 133764);
    //LinearLSTM* net = make_linearlstm(weights, 1, 8, 3);

    Squared env = {.size = 11};
    allocate(&env);

    Client* client = make_client(&env);

    reset(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
        } else {
            env.actions[0] = NOOP;
            //forward_linearlstm(net, env.observations, env.actions);
        }
        step(&env);
        render(client, &env);
    }
    //free_linearlstm(net);
    //free(weights);
    free_allocated(&env);
    close_client(client);
}
