#include <time.h>
#include "artillery.h"
#include "puffernet.h"

void demo() {
    printf("demo\n");

    Artillery env = {
        .width = 1280,
        .height = 720,
        .debug = 0,
        .moving_target = 1,
        .dist_fade = 0.36,
        .frameskip = 1,
        .miss_penalty = -0.1,
        .min_aim_angle = 0.56,
        .max_aim_angle = 1.56,
        .max_reward = 1.0,
        .max_reward_dist = 8.5,
        .max_dist0 = 100,
        .out_bounds_penalty = -0.1,
        .target_min_x = 600,
        .target_max_x = 1230,
        .target_min_y = 300,
        .target_max_y = 670,
        .target_size = 15,
        .turn_penalty = -0.03,
        .turn_penalty_delay = 64,
        .turn_penalty_ramp = 0.023,
        .render = 1,
        .rng = 7,
        .same_runs = 1,
        .vm = 150.0,
        .i = 1,
    };
    allocate(&env);

    env.client = make_client(&env);

    Weights* weights = load_weights("resources/artillery/puffer_artillery_weights.bin", 134022);
    int logit_sizes[1] = {5};
    int obs_size = 8;
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
}

int main() {
    demo();
}
