#include <time.h>
#include "whisker_racer.h"
#include "puffernet.h"

void demo() {
    printf("demo\n");
    Weights* weights = load_weights("/puffertank/bet/test/PufferLib/pufferlib/resources/whisker_racer/puffer_whisker_racer_weights.bin", 133124);
    int logit_sizes[1] = {3};
    LinearLSTM* net = make_linearlstm(weights, 1, 3, logit_sizes, 1);

    WhiskerRacer env = {
        .frameskip = 1,
        .width = 640,
        .height = 480,
        .llw_ang = -PI/4,
        .flw_ang = -PI/6,
        .frw_ang = PI/6,
        .rrw_ang = PI/4,
        .max_whisker_length = 100,
        .turn_pi_frac = 20,
        .maxv = 5,
        .render = 0,
        .continuous = 0,
        .reward_yellow = 0.25,
        .reward_green = 0.0,
        .gamma = 0.9,
        .track_width = 50,
        .num_radial_sectors = 16,
        .num_points = 4,
        .bezier_resolution = 16,
        .w_ang = 0.523,
        .corner_thresh = 0.5,
        .ftmp1 = 0.1,
        .ftmp2 = 0.1,
        .ftmp3 = 0.1,
        .ftmp4 = 0.1,
        .render_many = 0,
        .rng=42,
        .i = 1,
        .method = 0,
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
