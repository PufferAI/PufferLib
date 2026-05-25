// Standalone C demo for Boids environment
// Compile using: ./scripts/build.sh boids [local|fast]
// Run with: ./boids


#include <time.h>
#include "boids.h"
#include <stdlib.h>

// --- Demo Configuration ---
#define num_agents_DEMO 32  // Number of boids for the standalone demo
#define REPORT_INTERVAL_DEMO 1000 // Report interval for the demo
#define MAX_STEPS_DEMO 10000 // Max steps per episode in the demo
#define ACTION_SCALE 3.0f   // Corresponds to action space [-3.0, 3.0]

void generate_dummy_actions(Boids* env) {
    for (unsigned int i = 0; i < env->num_agents; ++i) {
        float rand_vx = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float rand_vy = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        env->actions[i * 2 + 0] = rand_vx * ACTION_SCALE;
        env->actions[i * 2 + 1] = rand_vy * ACTION_SCALE;
    }
}

void demo() {
    Boids env = {0}; 
    env.num_agents = num_agents_DEMO;
    env.report_interval = REPORT_INTERVAL_DEMO;
    
    size_t obs_size = env.num_agents * env.num_agents * 8; // 8 = (x, y, vx, vy, dx, dy, dvx, dvy)
    size_t act_size = env.num_agents * 2; // the 2 = (dvx, dvy)
    env.observations = (float*)calloc(obs_size, sizeof(float));
    env.actions = (float*)calloc(act_size, sizeof(float));
    env.rewards = (float*)calloc(env.num_agents, sizeof(float)); // Env-level reward
    
    if (!env.observations || !env.actions || !env.rewards) {
        fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
        free(env.observations); free(env.actions); free(env.rewards);
        return;
    }

    init(&env); 
    Client* client = make_client(&env);

    if (client == NULL) {
        fprintf(stderr, "ERROR: Failed to create rendering client during initial setup.\n");
        c_close(&env);
        free(env.observations); free(env.actions); free(env.rewards);
        return;
    }
    env.client = client;
    
    // Initial reset
    c_reset(&env);
    int total_steps = 0;

    printf("Starting Boids demo with %u boids. Press ESC to exit.\n", env.num_agents);

    while (!WindowShouldClose() && total_steps < MAX_STEPS_DEMO) { // Raylib function to check if ESC is pressed or window closed
        generate_dummy_actions(&env);
        c_step(&env);
        c_render(&env);
        total_steps++;
    }

    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    // ----------------------------------------
}

int main() {
    srand(time(NULL)); // Seed random number generator
    demo();
    return 0;
}
