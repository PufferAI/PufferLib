/* Pure C demo file for Nonogram. Build it with:
 * bash scripts/build_ocean.sh nonogram local (debug)
 * bash scripts/build_ocean.sh nonogram fast
 */

#include "nonogram.h"

int main() {
    Nonogram env = {.size = 8};
    int max_clues = env.size / 2;
    int obs_size = env.size * env.size + 2 * env.size * max_clues;

    env.max_steps = 4 * env.size * env.size;
    env.observations = (unsigned char*)calloc(obs_size, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        env.actions[0] = rand() % (env.size * env.size);
        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
