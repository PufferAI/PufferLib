#include <time.h>
#include "breakout.h"
#include "puffernet.h"

void demo() {
    Weights* weights = load_weights("resources/breakout/breakout_weights.bin");
    int logit_sizes[1] = {3};
    PufferNet* net = make_puffernet(weights, 1, 118, 64, 2, logit_sizes, 1);

    Breakout env = {
        .frameskip = 1,
        .width = 576,
        .height = 330,
        .initial_paddle_width = 62,
        .paddle_width = 62,
        .paddle_height = 8,
        .ball_width = 32,
        .ball_height = 32,
        .brick_width = 32,
        .brick_height = 12,
        .brick_rows = 6,
        .brick_cols = 18,
        .initial_ball_speed = 256,
        .max_ball_speed = 448,
        .paddle_speed = 620,
        .continuous = 0,
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
                env.actions[0] = 0.0;
                if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 1;
                if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 2;
            }
        } else if (frame % 4 == 0) {
            // Apply frameskip outside the env for smoother rendering
            forward_puffernet(net, env.observations, env.actions);
        }

        frame = (frame + 1) % 4;
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
}
