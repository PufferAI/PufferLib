#include <time.h>
#include "lock_key.h"

int main() {
    srand((unsigned int)time(NULL));

    LockKey env = {.size = 8, .num_keys = 3, .obs_dist = 2};

    int tiles = env.size * env.size;

    env.state        = (unsigned char*)calloc(tiles, sizeof(unsigned char));
    env.observations = (unsigned char*)calloc(tiles, sizeof(unsigned char));
    env.actions      = (int*)calloc(1, sizeof(int));
    env.rewards      = (float*)calloc(1, sizeof(float));
    env.terminals    = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.truncations  = (unsigned char*)calloc(1, sizeof(unsigned char)); // optional

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
                env.actions[0] = 0;
            } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
                env.actions[0] = 1;
            } else if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) {
                env.actions[0] = 2;
            } else if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) {
                env.actions[0] = 3;
            } else {
                env.actions[0] = -1; // no-op
            }
        } else {
            env.actions[0] = rand() % 5; // 4 == no-op, still fine
        }

        c_step(&env);
        c_render(&env);
    }

    free(env.state);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    if (env.truncations) free(env.truncations);

    c_close(&env);
    return 0;
}
