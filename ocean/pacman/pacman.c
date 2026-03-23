#include <time.h>
#include "pacman.h"
#include "puffernet.h"

void demo() {
    // printf("OBSERVATIONS_COUNT: %d\n", OBSERVATIONS_COUNT);
    Weights* weights = load_weights("resources/pacman/pacman_weights.bin", 170117);
    int logit_sizes[1] = {4};
    LinearLSTM* net = make_linearlstm(weights, 1, OBSERVATIONS_COUNT, logit_sizes, 1);

    PacmanEnv env = {
        .randomize_starting_position = false,
        .min_start_timeout = 0, // randomized ghost delay range
        .max_start_timeout = 49,
        .frightened_time = 35,   // ghost frighten time
        .max_mode_changes = 6,
        .scatter_mode_length = 700,
        .chase_mode_length = 70,
    };
    allocate(&env);
    c_reset(&env);
 
    Client* client = make_client(&env);
    bool human_control = false;

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
            human_control = true;
        } else {
            human_control = false;
        }

        if (!human_control) {
            forward_linearlstm(net, env.observations, env.actions);
        }

        c_step(&env);
        if (env.terminals[0]) {
            c_reset(&env);
        }

        for (int i = 0; i < FRAMES; i++) {
            c_render(&env);
        }
    }
    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(client);
}

void performance_test() {
    long test_time = 10;
    PacmanEnv env = {};
    allocate(&env);
    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 4;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated(&env);
}

int main() {
    //performance_test();
    demo();
    return 0;
}
