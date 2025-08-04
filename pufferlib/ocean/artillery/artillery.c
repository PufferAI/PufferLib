#include <time.h>
#include "artillery.h"
#include "puffernet.h"

void demo() {
    printf("demo\n");
    Weights* weights = load_weights("resources/artillery/puffer_artillery_weights.bin", 133638);
    int logit_sizes[1] = {5};
    LinearLSTM* net = make_linearlstm(weights, 1, 5, logit_sizes, 1);

    Artillery env = {
        .frameskip = 1,
        .width = 640,
        .height = 480,
        .target_min_x = 50,
        .target_max_x = 1870,
        .target_min_y = 50,
        .target_max_y = 1030,
        .min_aim_angle = 1.0,
        .max_aim_angle = 1.57,
        .max_reward = 1.0,
        .max_reward_dist = 50,
        .max_score = 1.0,
        .turn_penalty = -0.001,
        .render = 0,
        .continuous = 0,
        .ftmp1 = 300.0,
        .ftmp2 = 50.0,
        .ftmp3 = 0.1,
        .ftmp4 = 0.1,
        .render_many = 0,
        .rng = 7,
        .i = 1,
        .method = 0,
        .debug = 2,
    };
    printf("about to allocate\n");
    allocate(&env);

    printf("demo about to make_client\n");
    env.client = make_client(&env);

    printf("demo about to c_reset\n");
    c_reset(&env);
    int frame = 0;
    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        // User can take control of the paddle
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if(env.continuous) {
                float move = GetMouseWheelMove();
                float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
                env.actions[0] = clamped_wheel;
            } else {
                env.actions[0] = 0.0;
                if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 1;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 2;
            }
        } else if (frame % 4 == 0) {
            // Apply frameskip outside the env for smoother rendering
            int* actions = (int*)env.actions;
            forward_linearlstm(net, env.observations, actions);
            env.actions[0] = actions[0];
        }

        frame = (frame + 1) % 4;
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
