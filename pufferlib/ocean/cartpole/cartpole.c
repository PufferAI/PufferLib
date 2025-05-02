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
const char* WEIGHTS_PATH = "/puffertank/test_newbind/pufferlib/pufferlib/resources/cartpole/cartpole_weights.bin";

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
    
    // if (CONTINUOUS) {
    //     net = make_linearlstm_float(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE);
    // } else {
    //     net = make_linearlstm_int(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE);
    // }

    net = make_linearlstm(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE);
    Cartpole env = {0};
    env.is_continuous = CONTINUOUS;
    allocate(&env);
    Client* client = make_client(&env);
    c_reset(&env);

    SetTargetFPS(60);
    int episode_steps = 0;
    float episode_return = 0.0f;

    while (!WindowShouldClose()) {
        int userControlMode = IsKeyDown(KEY_LEFT_SHIFT);

        if (!userControlMode) {
            // if (CONTINUOUS) {
            //     forward_linearlstm_float(net, env.observations, env.actions);
            //     env.actions[0] = tanhf(env.actions[0]);
            // } else {
            //     int action_value;
            //     forward_linearlstm_int(net, env.observations, &action_value);
            //     env.actions[0] = movement(action_value, 0);
            // }
            int action_value;
            forward_linearlstm(net, env.observations, &action_value);
            env.actions[0] = movement(action_value, 0);
        } else {
            env.actions[0] = movement(env.actions[0], userControlMode);
        }   

        c_step(&env);
        episode_return += env.rewards[0];
        episode_steps++;

        BeginDrawing();
        ClearBackground(RAYWHITE);
        c_render(&env);
        DrawText("Evaluating policy...", 10, 160, 20, DARKGRAY);
        EndDrawing();

        if (env.terminals[0]) {
            printf("Episode done. Steps: %d, Return: %.2f\n\n", episode_steps, episode_return);
            episode_steps = 0;
            episode_return = 0.0f;
            c_reset(&env);
        }
    }

    free_linearlstm(net);
    free(weights);
    close_client(client);
    free_allocated(&env);
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}
