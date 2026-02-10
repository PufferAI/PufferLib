/* Pure C demo file for Squared Continuous. Build it with:
 * bash scripts/build_ocean.sh squared_continuous local (debug)
 * bash scripts/build_ocean.sh squared_continuous fast
 */

#include "squared_continuous.h"

int main() {
    Squared env = {.size = 11};
    env.observations = (unsigned char*)calloc(env.size*env.size, sizeof(unsigned char));
    env.actions = (double*)calloc(2, sizeof(double));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.actions[0] = 0.0;
        env.actions[1] = 0.0;
        if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = -1.0;
        if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 1.0;
        if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[1] = -1.0;
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[1] = 1.0;
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
