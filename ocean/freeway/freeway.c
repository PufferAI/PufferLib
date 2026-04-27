#include <time.h>
#include "freeway.h"
#include "puffernet.h"
#include <unistd.h>

void demo() {
    Weights* weights = load_weights("resources/freeway/freeway_weights.bin");
    int logit_sizes[1] = {3};
    PufferNet* net = make_puffernet(weights, 1, 34, 128, 7, logit_sizes, 1);

    Freeway env = {
        .frameskip=4,
        .width=1216,
        .height=720,
        .player_width=64,
        .player_height=64,
        .car_width=64,
        .car_height=40,
        .lane_size=64,
        .difficulty=0,
        .level=4,
        .use_dense_rewards=1,
        .env_randomization=1,
        .enable_human_player=1,
    };
    allocate(&env);

    env.client = make_client(&env);

    c_reset(&env);
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        forward_puffernet(net, env.observations, env.actions);
        env.human_actions[0] = 0;
        if (IsKeyDown(KEY_UP)  || IsKeyDown(KEY_W)) env.human_actions[0] = 1;
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) env.human_actions[0] = 2;
        c_step(&env);
        c_render(&env);
    }
    free_puffernet(net);
    free(weights);
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
    return 0;
}
