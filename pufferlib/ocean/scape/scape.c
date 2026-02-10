/* Pure C demo file for Scape. Build it with:
 * bash scripts/build_ocean.sh scape local (debug)
 * bash scripts/build_ocean.sh scape fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "scape.h"

int main() {
    int num_agents = 1;
    int num_obs = 1;

    Scape env = {
        .width = 1080,
        .height = 720,
        .num_agents = 1
    };
    init(&env);

    // Allocate these manually since they aren't being passed from Python
    env.observations = calloc(env.num_agents*num_obs, sizeof(float));
    env.actions = calloc(env.num_agents, sizeof(double));
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(double));

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    // while(True) will break web builds
    while (!WindowShouldClose()) {
        for (int i=0; i<env.num_agents; i++) {
            env.actions[i] = rand() % 9;
        }

        c_step(&env);
        for (int frame=0; frame<36; frame++) {
            c_render(&env);
        }
    }

    // Try to clean up after yourself
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

