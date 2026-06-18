#include "flappy_bird.h"
#include "puffernet.h"

int main(void) {
    Weights* weights = load_weights("resources/flappy_bird/flappy_bird_weights.bin");
    int logit_sizes[1] = {2};
    PufferNet* net = make_puffernet(weights, 1, OBS_SIZE, 256, 1, logit_sizes, 1);

    FlappyBird env = {
        .num_agents = 1,
        .gravity = GRAVITY,
        .pipe_speed = PIPE_SPEED,
    };

    c_init(&env);
    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = (float)(rand() % 2);
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = (float)NOOP;
            if (IsKeyDown(KEY_SPACE)) {
                env.actions[0] = (float)UP;
            }
        } else {
            forward_puffernet(net, env.observations, env.actions);
        }
        c_step(&env);
        c_render(&env);
    }

    free_puffernet(net);
    free(weights);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
