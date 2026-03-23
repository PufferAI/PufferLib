#include "matsci.h"

int main() {
    int num_agents = 16;
    Matsci env = {.num_agents=num_agents};
    env.observations = (float*)calloc(3*num_agents, sizeof(float));
    env.actions = (float*)calloc(3*num_agents, sizeof(float));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    init(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
	for (int i=0; i<3*num_agents; i++) {
            env.actions[i] = rndf(-1.0f, 1.0f);
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

