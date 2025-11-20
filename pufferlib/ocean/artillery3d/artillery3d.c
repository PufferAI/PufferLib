#include "artillery3d.h"
#include "puffernet.h"

void allocate(Artillery3D* env) {
    init(env);
    env->observations = (float*)calloc(19, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(Artillery3D* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    free(env->tx);
    free(env->ty);
    free(env->tz);
    free(env->target_vx);
    free(env->target_vy);
    free(env->target_vz);
    free(env->time_target_vanish);
    c_close(env);
}

void demo() {
    Artillery3D env = {
        .debug = 0,
        .dist_fade = 0.7493789405520497,
        .max_dist0 = 127.50246246114087,
        .max_reward = 1.0,
        .max_reward_dist = 23.620013496047616,
        .miss_penalty = -0.1858434974084412,
        .out_bounds_penalty = -0.01,
        .target_size = 15,
        .turn_penalty = -0.003,
        .turn_penalty_delay = 72.37761171826367,
        .turn_penalty_ramp = 0.02,
        .rng = 7,
        .render = 1,
        .i = 1,
    };
    allocate(&env);

    env.client = make_client(&env);

    const char* weights_path = "resources/artillery3d/puffer_artillery3d_weights.bin";
    int weights_size = 135688;

    Weights* weights = load_weights(weights_path, weights_size);
    int logit_sizes[1] = {7};
    int obs_size = 19;
    LinearLSTM* net = make_linearlstm(weights, 1, obs_size, logit_sizes, 1);

    c_reset(&env);
    SetTargetFPS(30);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 1;
            if (IsKeyDown(KEY_SPACE)) env.actions[0] = 0;
            if (IsKeyDown(KEY_UP)  || IsKeyDown(KEY_W)) env.actions[0] = 1;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 2;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 3;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 4;
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
    free_allocated(&env);
    close_client(env.client);
}

int main() {
    demo();
}
