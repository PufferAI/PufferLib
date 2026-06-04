#include <time.h>
#include "pathfinder.h"

int main(void) {
    Pathfinder env;
    memset(&env, 0, sizeof(env));

    float observations[PATHFINDER_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.num_agents = 1;
    env.rng = (unsigned int)time(NULL);
    env.branch_prob = 0.35f;
    env.loop_prob = 0.10f;
    env.extra_entry_prob = 0.0f;
    env.min_solution_len = 1;
    env.max_solution_len = 4;
    env.max_steps = 128;

    init(&env);
    c_reset(&env);

    for (int i = 0; i < 256; i++) {
        actions[0] = (float)(pathfinder_rand(&env) % PATHFINDER_NUM_ACTIONS);
        c_step(&env);
    }

    printf("Pathfinder random smoke: episodes=%0.0f success=%0.3f return=%0.3f\n",
        env.log.n, env.log.success, env.log.episode_return);
    c_close(&env);
    return 0;
}
