/* Pure C demo file for Light Thief. Build it with:
 *   bash scripts/build_ocean.sh light_thief local (debug)
 *   bash scripts/build_ocean.sh light_thief fast
 */

#include "light_thief.h"

int main() {
    LightThief env = {0};
    env.observations = (float*)calloc(23, sizeof(float));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
            env.actions[0] = ACTION_STAY;
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) env.actions[0] = ACTION_UP;
            if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) env.actions[0] = ACTION_DOWN;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = ACTION_LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = ACTION_RIGHT;
        } else {
            env.actions[0] = rand() % 5;
        }

        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
