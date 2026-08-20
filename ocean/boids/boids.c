// Standalone C demo for Boids environment
// Compile using: ./build.sh boids --debug
// Run with: ./boids

#include <time.h>
#include "boids.h"

#define NUM_BOIDS_DEMO 20
#define MAX_STEPS_DEMO 500
#define ACTION_SCALE 3.0f

void generate_dummy_actions(Boids* env) {
    for (unsigned int i = 0; i < env->num_boids; ++i) {
        float rand_vx = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float rand_vy = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        env->agents[i].actions[0] = rand_vx * ACTION_SCALE;
        env->agents[i].actions[1] = rand_vy * ACTION_SCALE;
    }
}

void demo() {
    Boids env = {0};
    env.num_boids = NUM_BOIDS_DEMO;
    env.num_agents = NUM_BOIDS_DEMO;
    env.report_interval = 100;
    env.margin_turn_factor = 0.0f;
    env.centering_factor = 0.0f;
    env.avoid_factor = 1.0f;
    env.matching_factor = 1.0f;

    for (unsigned i = 0; i < env.num_boids; i++) {
        env.agents[i].observations = calloc(OBS_SIZE, sizeof(float));
        env.agents[i].actions = (float*)calloc(NUM_ATNS, sizeof(float));
        env.agents[i].rewards = (float*)calloc(1, sizeof(float));
        env.agents[i].terminals = (float*)calloc(1, sizeof(float));
        env.agents[i].action_mask = NULL;
        env.agents[i].policy = 0;
    }

    init(&env);
    Client* client = make_client(&env);

    if (client == NULL) {
        fprintf(stderr, "ERROR: Failed to create rendering client during initial setup.\n");
        puf_close(&env);
        for (unsigned i = 0; i < env.num_boids; i++) {
            free(env.agents[i].observations);
            free(env.agents[i].actions);
            free(env.agents[i].rewards);
            free(env.agents[i].terminals);
        }
        return;
    }
    env.client = client;

    puf_reset(&env);
    int total_steps = 0;

    printf("Starting Boids demo with %d boids. Press ESC to exit.\n", env.num_boids);

    while (!WindowShouldClose() && total_steps < MAX_STEPS_DEMO) {
        generate_dummy_actions(&env);
        puf_step(&env);
        puf_render(&env);
        total_steps++;
    }

    puf_close(&env);
    for (unsigned i = 0; i < env.num_boids; i++) {
        free(env.agents[i].observations);
        free(env.agents[i].actions);
        free(env.agents[i].rewards);
        free(env.agents[i].terminals);
    }
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}
