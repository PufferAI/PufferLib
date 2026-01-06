#include <time.h>
#include "pong.h"
#include "puffernet.h"

void demo() {
    Weights* weights = load_weights("resources/pong/pong_weights.bin", 133764);

    int logit_sizes[1] = {3};
    LinearLSTM* net = make_linearlstm(weights, 1, 8, logit_sizes, 1);

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
                if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 1.0;
                if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 2.0;
            }
        } else if (frame % 8 == 0) {
            // Apply frameskip outside the env for smoother rendering
            int* actions = (int*)env.actions;
            forward_linearlstm(net, env.observations, actions);
            env.actions[0] = actions[0];
        }

        frame = (frame + 1) % 8;
        c_step(&env);
        c_render(&env);
    }
    free_linearlstm(net);
    free(weights);
    free_allocated(&env);
    close_client(env.client);
}

void test_performance(int timeout) {
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

    int start = time(NULL);
    int num_steps = 0;
    while (time(NULL) - start < timeout) {
        env.actions[0] = rand() % 3;
        c_step(&env);
        num_steps++;
    }

    int end = time(NULL);
    float sps = num_steps / (end - start);
    printf("Test Environment SPS: %f\n", sps);
    free_allocated(&env);
}

int main() {
    demo();
    //test_performance(10);
}
