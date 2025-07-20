/* FlappyBirdMini: a simple 2-block tall grid environment.
 * Agent can move up or down. Observes an 8x2 grid with moving obstacles.
 * +1 reward per obstacle passed.
 */

#include "flappy_bird_mini.h"

int main() {
    FlappyBirdMini env = {};
    env.observations = (unsigned char*)calloc(GRID_WIDTH * GRID_HEIGHT, sizeof(unsigned char)); // 8x2 grid observations
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = NOOP;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
        } else {
            env.actions[0] = rand() % 3; // 0: NOOP, 1: UP, 2: DOWN
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
