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
        .timed_shell = 0,
        .dist_fade = 0.3,
        .frameskip = 1,
        .ftmp1 = 100.0,
        .ftmp2 = 50.0,
        .ftmp3 = 0.22,
        .ftmp4 = 0.1,
        .method = -1,
        .miss_penalty = -0.2,
        .min_aim_angle = 0.56,
        .max_aim_angle = 1.56,
        .max_reward = 1.0,
        .max_reward_dist = 30,
        .max_dist0 = 250.0,
        .max_score = 1.0,
        .out_bounds_penalty = -0.1,
        .target_min_x = 600,
        .target_max_x = 1230,
        .target_min_y = 300,
        .target_max_y = 670,
        .target_size = 15,
        .turn_penalty = -0.03,
        .turn_penalty_delay = 75,
        .turn_penalty_ramp = 0.015,
        .render = 1,
        .rng = 7,
        .same_runs = 1,
        .vm = 150.0,
        .continuous = 0,
        .i = 1,
    };
    printf("about to allocate\n");
    allocate(&env);

    printf("demo about to make_client\n");
    env.client = make_client(&env);

    const char* weights_path = (env.moving_target == 1) ? 
        "resources/artillery/puffer_artillery_weights_moving.bin" : 
        "resources/artillery/puffer_artillery_weights_stationary.bin";
    printf(weights_path);
    int weights_size = (env.moving_target == 1) ? 134022 : 133766;

    Weights* weights = load_weights(weights_path, weights_size); // 133638
    int logit_sizes[1] = {5};
    int obs_size = (env.moving_target == 1) ? 8 : 6;
    LinearLSTM* net = make_linearlstm(weights, 1, obs_size, logit_sizes, 1);

    printf("demo about to c_reset\n");
    c_reset(&env);
    int frame = 0;
    SetTargetFPS(30);
    while (!WindowShouldClose()) {
        // User can take control of the paddle
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if(env.continuous) {
                float move = GetMouseWheelMove();
                float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
                env.actions[0] = clamped_wheel;
            } else {
                env.actions[0] = 100;
                if (IsKeyDown(KEY_SPACE)) env.actions[0] = 0;
                if (IsKeyDown(KEY_UP)  || IsKeyDown(KEY_W)) env.actions[0] = 1;
                if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 2;
                if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 3;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 4;
            }
        } else {
            // Apply frameskip outside the env for smoother rendering
            int* actions = (int*)env.actions;
            forward_linearlstm(net, env.observations, actions);
            env.actions[0] = actions[0];
        }

        frame = (frame + 1) % 1;
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
    //test_performance(10); // found in breakout.c
}
