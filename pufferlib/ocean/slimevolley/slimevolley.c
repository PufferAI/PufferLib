/* Pure C demo file for SlimeVolley. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#include "slimevolley.h"
#include <stdio.h>


void abranti_simple_policy(float* obs, float* action) {
    float x_agent = obs[0];
    float x_ball = obs[4];
    float vx_ball = obs[6];
    float backward = (-23.757145f * x_agent + 23.206863f * x_ball + 0.7943352f * vx_ball) + 1.4617119f;
    float forward = -64.6463748f * backward + 22.4668393f;
    action[0] = forward;
    action[1] = backward;
    action[2] = 1.0f; // always jump
}

void random_policy(float* obs, float* action) {
    action[0] = rand() * 2 - 1;
    action[1] = rand() * 2 - 1;
    action[2] = rand() * 2 - 1;
}

int main() {
    int num_obs = 12;
    int num_actions = 3;
    SlimeVolley env = {.num_agents = 1};
    init(&env);
    env.observations = (float*)calloc(env.num_agents*num_obs, sizeof(float));
    env.actions = (float*)calloc(num_actions*env.num_agents, sizeof(float));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(env.num_agents, sizeof(unsigned char));
    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    fprintf(stderr, "num agents: %d\n", env.num_agents);

    while (!WindowShouldClose()) {
        for (int i=0; i<env.num_agents; i++) {
            if (i == 0) {
                random_policy(&env.observations[12*i], &env.actions[3*i]);
                
            } else {
                abranti_simple_policy(&env.observations[12*i], &env.actions[3*i]);
            }
        }
        c_step(&env);
        c_render(&env);
        if (env.terminals[0] || env.terminals[1]) {
            fprintf(stderr, "Episode ended. Rewards: %f, %f\n", env.rewards[0], env.rewards[1]);
            break;
        }
    }

    // Try to clean up after yourself
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
