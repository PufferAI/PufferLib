#include <time.h>
#include "artillery.h"
#include "puffernet.h"

void allocate(Artillery* env) {
    init(env);
    int obs_size = 11;
    env->observations = (float*)calloc(obs_size, sizeof(float));
    env->actions = (int*)calloc(2, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(Artillery* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void demo() {
    Artillery env = {
        .adj = 0.014364991132735087,
        .dist_fade = 0.11107816724558334,
        .frameskip = 1,
        .miss_penalty = -0.05521742600140713,
        .max_reward_dist = 44.700659812685586,
        .max_dist0 = 104.59870905085756,
        .out_bounds_penalty = -0.1,
        .turn_penalty = -0.1,
        .turn_penalty_delay = 98,
        .render = 1,
    };
    allocate(&env);

    env.client = make_client(&env);

    Weights* weights = load_weights("resources/artillery/puffer_artillery_weights.bin", 135051);
    int logit_sizes[2] = {5, 5};
    int obs_size = 11;
    LinearLSTM* net = make_linearlstm(weights, 1, obs_size, logit_sizes, 2);

    c_reset(&env);
    SetTargetFPS(30);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 4.0f;
            env.actions[1] = 4.0f;

            if (IsKeyDown(KEY_SPACE)) env.actions[0] = 0;
            if (IsKeyDown(KEY_W)) env.actions[0] = 1;
            if (IsKeyDown(KEY_S)) env.actions[0] = 2;
            if (IsKeyDown(KEY_A)) env.actions[0] = 3;
            if (IsKeyDown(KEY_D)) env.actions[0] = 4;

            if (IsKeyDown(KEY_ENTER)) env.actions[1] = 0;
            if (IsKeyDown(KEY_UP)) env.actions[1] = 1;
            if (IsKeyDown(KEY_DOWN)) env.actions[1] = 2;
            if (IsKeyDown(KEY_LEFT)) env.actions[1] = 3;
            if (IsKeyDown(KEY_RIGHT)) env.actions[1] = 4;
        } else {
            int discrete_actions[2];
            forward_linearlstm(net, env.observations, discrete_actions);
            env.actions[0] = discrete_actions[0];
            env.actions[1] = discrete_actions[1];
        }
        c_step(&env);
        c_render(&env);
    }
    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}
