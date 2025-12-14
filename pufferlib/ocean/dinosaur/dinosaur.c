#include "dinosaur.h"
#include "puffernet.h"

int main() {
    int max_obstacles = 10;
    int num_obs = (max_obstacles*2) + 4;

    Weights* weights = load_weights("resources/dinosaur/puffer_dinosaur_weights.bin", 544316);

    int logit_sizes[1] = {3};
    LinearLSTM* net = make_linearlstm(weights, 1, num_obs, logit_sizes, 1);

    Dinosaur env = {
        .width = 800,
        .height = 400,
        .speed_init = 6,
        .speed_max = 12,
        .spawn_rate_max = 100,
        .spawn_rate_min = 40,
        .rate_increment_rate = 400,
        .max_obstacles = 8
    };
    init(&env);

    env.observations = calloc(num_obs, sizeof(float));
    env.actions = calloc(2, sizeof(int));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        if(IsKeyDown(KEY_LEFT_SHIFT)){
            env.actions[0] = NOOP;
            if(IsKeyDown(KEY_UP)) env.actions[0] = JUMP;
            if(IsKeyDown(KEY_DOWN)) env.actions[0] = CROUCH;
        } else {
            int* actions = (int*)env.actions;
            forward_linearlstm(net, env.observations, actions);
            env.actions[0] = actions[0];
        }
        c_step(&env);
        c_render(&env);
    }

    free_linearlstm(net);
    free(weights);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
