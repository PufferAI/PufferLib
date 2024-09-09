#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
// #include "raylib.h"

typedef struct Tactical Tactical;

struct Tactical {
    int num_agents;
    unsigned char* observations;
    int* actions;
    float* rewards;
};

void free_tactical(Tactical* env) {
    free(env->rewards);
    free(env->observations);
    free(env->actions);
    free(env);
}
 
void compute_observations(Tactical* env) {

}

Tactical* init_tactical() {
    Tactical* env = (Tactical*)calloc(1, sizeof(Tactical));

    env->rewards = calloc(env->num_agents, sizeof(float));
    env->observations = calloc(env->num_agents*121*121*4, sizeof(unsigned char));
    env->actions = calloc(env->num_agents*6, sizeof(int));

    return env;
}

void reset(Tactical* env) {
    compute_observations(env);
}

int step(Tactical* env) {
    if (false) {
        reset(env);
        int winner = 2;
        return winner;
    }

    compute_observations(env);
    return 0;
}
