#include <time.h>
#include "pong.h"
#include "puffernet.h"

void demo() {
    // Weight count: encoder(32x8=256) + decoder(4x32=128) + 1x mingru(3x32x32=3072) = 3456
    Weights* weights = load_weights("resources/pong/pong_weights.bin");

    int logit_sizes[1] = {3};
    PufferNet* net = make_puffernet(weights, 1, 8, 32, 1, logit_sizes, 1);

    Pong env = {
        .width = 500,
        .height = 640,
        .paddle_width = 20,
        .paddle_height = 70,
        .ball_width = 32,
        .ball_height = 32,
        .paddle_speed = 8,
        .ball_initial_speed_x = 10,
        .ball_initial_speed_y = 1,
        .ball_speed_y_increment = 3,
        .ball_max_speed_y = 13,
        .max_score = 21,
        .frameskip = 1,
        .continuous = 0,
    };

    allocate(&env);
    c_reset(&env);
    c_render(&env);
    SetTargetFPS(60);
    int frame = 0;
    while (!WindowShouldClose()) {
        // User can take control of the paddle
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if(env.continuous) {
                float move = GetMouseWheelMove();
                float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
                env.actions[0] = clamped_wheel;
                printf("Mouse wheel move: %f\n", env.actions[0]);
            } else {
                env.actions[0] = 0.0;
                if (IsKeyDown(KEY_UP)   || IsKeyDown(KEY_W)) env.actions[0] = 1.0;
                if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) env.actions[0] = 2.0;
            }
        } else if (frame == 0) {
            forward_puffernet(net, env.observations, env.actions);
        }

        frame = (frame + 1) % 8;
        c_step(&env);
        // Reset frame counter on score so policy fires immediately next
        // iteration, matching training's early-return-from-frameskip behavior
        if (env.rewards[0] != 0.0f) frame = 0;
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
