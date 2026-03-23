// Standalone C demo for Boids environment
// Compile using: ./scripts/build_ocean.sh boids [local|fast]
// Run with: ./boids

#include <time.h>
#include "boids.h"

// --- Demo Configuration ---
#define NUM_BOIDS_DEMO 20   // Number of boids for the standalone demo
#define MAX_STEPS_DEMO 500 // Max steps per episode in the demo
#define ACTION_SCALE 3.0f   // Corresponds to action space [-3.0, 3.0]

// Dummy action generation: random velocity changes for each boid
void generate_dummy_actions(Boids* env) {
    for (unsigned int i = 0; i < env->num_boids; ++i) {
        // Generate random floats in [-1, 1] range
        float rand_vx = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float rand_vy = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        
        // Scale to the action space [-ACTION_SCALE, ACTION_SCALE]
        env->actions[i * 2 + 0] = rand_vx * ACTION_SCALE;
        env->actions[i * 2 + 1] = rand_vy * ACTION_SCALE;
    }
}

void demo() {
    // Initialize Boids environment struct
    Boids env = {0}; 
    env.num_boids = NUM_BOIDS_DEMO;
    
    // In the Python binding, these pointers are assigned from NumPy arrays.
    // Here, we need to allocate them explicitly.
    size_t obs_size = env.num_boids * 4; // num_boids * (x, y, vx, vy)
    size_t act_size = env.num_boids * 2; // num_boids * (dvx, dvy)
    env.observations = (float*)calloc(obs_size, sizeof(float));
    env.actions = (float*)calloc(act_size, sizeof(float));
    env.rewards = (float*)calloc(env.num_boids, sizeof(float)); // Env-level reward
    
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

    printf("Starting Boids demo with %d boids. Press ESC to exit.\n", env.num_boids);

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
