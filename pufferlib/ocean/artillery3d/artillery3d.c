#include <time.h>
#include "artillery3d.h"
#include "puffernet.h"

void demo() {
    printf("demo\n");

    Artillery3D env = {
        .x_size = 2000,
        .y_size = 2000,
        .z_size = 200,
        .debug = 0,
        .moving_target = 1,
        .dist_fade = 0.36,
        .frameskip = 1,
        .miss_penalty = -0.1,
        .max_reward = 1.0,
        .max_reward_dist = 8.5,
        .max_dist0 = 100,
        .out_bounds_penalty = -0.1,
        .target_min_x = 1000,
        .target_max_x = 1900,
        .target_min_y = 1000,
        .target_max_y = 1900,
        .target_min_z = 10,
        .target_max_z = 100,
        .target_size = 15,
        .turn_penalty = -0.03,
        .turn_penalty_delay = 64,
        .turn_penalty_ramp = 0.023,
        .render = 1,
        .rng = 7,
        .same_runs = 1,
        .i = 1,
    };
    allocate(&env);

    env.client = make_client(&env);

    const char* weights_path = "resources/artillery3d/puffer_artillery3d_weights.bin";
    int weights_size = 134664;

    Weights* weights = load_weights(weights_path, weights_size); // 133638
    int logit_sizes[1] = {7};
    int obs_size = 11;
    LinearLSTM* net = make_linearlstm(weights, 1, obs_size, logit_sizes, 1);

    c_reset(&env);
    SetTargetFPS(30);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 100;
            if (IsKeyDown(KEY_SPACE)) env.actions[0] = 0;
            if (IsKeyDown(KEY_UP)  || IsKeyDown(KEY_W)) env.actions[0] = 1;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 2;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 3;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 4;
        } else {
            int* actions = (int*)env.actions;
            //printf("C Obs: ");
            //for(int i = 0; i < obs_size; i++) {
            //    printf("%.3f ", env.observations[i]);
            //}
            //printf("\n");
            forward_linearlstm(net, env.observations, actions);
            //printf("LSTM state_h[0-3]: %.3f %.3f %.3f %.3f\n",
            //    net->lstm->state_h[0], net->lstm->state_h[1],
            //    net->lstm->state_h[2], net->lstm->state_h[3]);
            //printf("Logits: ");
            //for(int i = 0; i < 5; i++) {
            //    printf("%.3f ", net->actor->output[i]);
            //}
            //printf("\n");
            env.actions[0] = actions[0];
        }
        c_step(&env);
        c_render(&env);
    }
    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(env.client);
    printf("end demo\n");
}

int main() {
    demo();
}
