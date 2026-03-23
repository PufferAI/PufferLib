#include <time.h>
#include "whisker_racer.h"
#include "puffernet.h"

void demo() {
    printf("demo\n");
    Weights* weights = load_weights("resources/whisker_racer/puffer_whisker_racer_weights.bin", 133124);
    int logit_sizes[1] = {3};
    LinearLSTM* net = make_linearlstm(weights, 1, 3, logit_sizes, 1);

    WhiskerRacer env = {
        .frameskip = 1,
        .width = 1080,
        .height = 720,
        .max_whisker_length = 100,
        .turn_pi_frac = 40,
        .maxv = 5,
        .render = 0,
        .continuous = 0,
        .reward_yellow = 0.25,
        .reward_green = -0.001,
        .track_width = 75,
        .num_radial_sectors = 180,
        .num_points = 16,
        .bezier_resolution = 4,
        .w_ang = 0.777,
        .corner_thresh = 0.5,
        .mode7 = 1, // If mode7 = 1 then 640X480 recommended
        .render_many = 0,
        .rng = 3, // rng = 3 for puffer track
        .i = 1, // i = 1 for puffer track
        .method = 1, // method = 1 for puffer track
    };

    allocate(&env);

    env.client = make_client(&env);

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
                env.actions[0] = 1.0;                                               // Straight
                if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 0.0; // Left
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 2.0; // Right
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
}

int main() {
    demo();
}
