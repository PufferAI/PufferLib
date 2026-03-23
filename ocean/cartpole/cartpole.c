// local compile/eval implemented for discrete actions only
// eval with python demo.py --mode eval --env puffer_cartpole --eval-mode-path <path to model>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cartpole.h"
#include "puffernet.h"

#define NUM_WEIGHTS 133123
#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 2
#define CONTINUOUS 0

const char* WEIGHTS_PATH = "resources/cartpole/cartpole_weights.bin";

float movement(int discrete_action, int userControlMode) {
    if (userControlMode) {
        return (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) ? 1.0f : -1.0f;
    } else {
        return (discrete_action == 1) ? 1.0f : -1.0f;
    }
}

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
    LinearLSTM* net;
    
    int logit_sizes[1] = {ACTIONS_SIZE};
    net = make_linearlstm(weights, 1, OBSERVATIONS_SIZE, logit_sizes, 1);
    Cartpole env = {0};
    env.continuous = CONTINUOUS;
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        int userControlMode = IsKeyDown(KEY_LEFT_SHIFT);

        if (!userControlMode) {
            int action_value;
            forward_linearlstm(net, env.observations, &action_value);
            env.actions[0] = movement(action_value, 0);
        } else {
            env.actions[0] = movement(env.actions[0], userControlMode);
        }   

        c_step(&env);

        BeginDrawing();
        ClearBackground(RAYWHITE);
        c_render(&env);
        EndDrawing();

        if (env.terminals[0]) {
            c_reset(&env);
        }
    }

    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}
