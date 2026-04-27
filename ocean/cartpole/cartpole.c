// local compile/eval implemented for discrete actions only
// eval with python demo.py --mode eval --env puffer_cartpole --eval-mode-path <path to model>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cartpole.h"
#include "puffernet.h"

#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 2
#define CONTINUOUS 0

const char* WEIGHTS_PATH = "resources/cartpole/cartpole_weights.bin";

float movement(float action, int userControlMode) {
    if (userControlMode) {
        return (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) ? 1.0f : -1.0f;
    } else {
        return (action > 0.5f) ? 1.0f : -1.0f;
    }
}

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH);
    
    int logit_sizes[1] = {ACTIONS_SIZE};
    PufferNet* net = make_puffernet(weights, 1, OBSERVATIONS_SIZE, 32, 2, logit_sizes, 1);
    
    Cartpole env = {
        .continuous = CONTINUOUS,
        .cart_mass = 1.0f,
        .pole_mass = 0.1f,
        .pole_length = 0.5f,
        .gravity = 9.8f,
        .force_mag = 10.0f,
        .tau = 0.02f,
    };
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        int userControlMode = IsKeyDown(KEY_LEFT_SHIFT);

        if (!userControlMode) {
            forward_puffernet(net, env.observations, env.actions);
            env.actions[0] = movement(env.actions[0], 0);
        } else {
            env.actions[0] = movement(env.actions[0], userControlMode);
        }   

        c_step(&env);
        c_render(&env);

        if (env.terminals[0] > 0.5f) {
            c_reset(&env);
        }
    }

    free_puffernet(net);
    free(weights);
    free_allocated(&env);
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}
