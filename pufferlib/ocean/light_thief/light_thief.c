#include "light_thief.h"

int main() {
    LightThief env;
    env.observations = (float*)malloc(23 * sizeof(float));
    env.actions = (int*)malloc(sizeof(int));
    env.rewards = (float*)malloc(sizeof(float));
    env.terminals = (unsigned char*)malloc(sizeof(unsigned char));

    c_reset(&env);
    for (int i = 0; i < 100; i++) {
        env.actions[0] = rand() % 5;
        c_step(&env);
        if (env.terminals[0]) {
            c_reset(&env);
        }
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    return 0;
}
