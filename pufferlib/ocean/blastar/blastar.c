#include "blastar.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "puffernet.h"

const char* WEIGHTS_PATH = "resources/blastar/blastar_weights.bin";
#define OBSERVATIONS_SIZE 10
#define ACTIONS_SIZE 6
#define NUM_WEIGHTS 134407

void get_input(Blastar* env) {
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->actions[0] = 1;  // Left
    } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->actions[0] = 2;  // Right
    } else if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        env->actions[0] = 3;  // Up
    } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        env->actions[0] = 4;  // Down
    } else if (IsKeyDown(KEY_SPACE)) {
        env->actions[0] = 5;  // Fire
    } else {
        env->actions[0] = 0;  // Noop
    }
}

int demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
    LinearLSTM* net = make_linearlstm(weights, 1, OBSERVATIONS_SIZE, ACTIONS_SIZE);
    Blastar env = {
        .num_obs = OBSERVATIONS_SIZE,
    };
    allocate(&env, env.num_obs);
    Client* client = make_client(&env);
    unsigned int seed = 12345;
    srand(seed);
    c_reset(&env);
    int running = 1;
    while (running) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            get_input(&env);  // Human input
        } else {
            forward_linearlstm(net, env.observations, env.actions);  // AI input
        }
        c_step(&env);
        c_render(&env);
        if (WindowShouldClose() || env.game_over) {
            running = 0;
        }
    }
    free_linearlstm(net);
    free(weights);
    close_client(client);
    free_allocated(&env);
    return 0;
}

void perftest(float test_time) {
    Blastar env = {
        .num_obs = OBSERVATIONS_SIZE,
    };
    allocate(&env, env.num_obs);
    unsigned int seed = 12345;
    srand(seed);
    c_reset(&env);
    int start = time(NULL);
    int steps = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % ACTIONS_SIZE;  // Random actions
        c_step(&env);
        steps++;
    }
    int end = time(NULL);
    printf("Steps per second: %f\n", steps / (float)(end - start));
    free_allocated(&env);
}

int main() {
    demo();
    // perftest(10.0f);
    return 0;
}