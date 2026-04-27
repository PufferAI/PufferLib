#include "terraform.h"
#include "puffernet.h"
#include <time.h>

void allocate(Terraform* env) {
    env->observations = (float*)calloc(env->num_agents*319, sizeof(float));
    env->actions = (float*)calloc(3*env->num_agents, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));
    init(env);
}

void free_allocated(Terraform* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_initialized(env);
}

void demo() {
    Weights* weights = load_weights("resources/terraform/terraform_weights.bin");
    int logit_sizes[3] = {5, 5, 3};
    PufferNet* net = make_puffernet(weights, 1, 319, 512, 5, logit_sizes, 3);
    srand(time(NULL));
    Terraform env = {.size = 64, .num_agents = 1, .reset_frequency = 8192, .reward_scale = 0.04f};
    allocate(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        forward_puffernet(net, env.observations, env.actions);
        
        if(IsKeyDown(KEY_LEFT_SHIFT)) {
            // When shift is held, stop the dozer
            env.actions[0] = 2;  // Stop vertical movement
            env.actions[1] = 2;  // Stop horizontal movement
            env.actions[2] = 0;  // no scoop or drop
            // Override with keyboard controls if keys are pressed
            if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) env.actions[0] = 4;
            if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) env.actions[0] = 0;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[1] = 0;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[1] = 4;
            if (IsKeyPressed(KEY_SPACE)) env.actions[2] = 1;
            if (IsKeyPressed(KEY_ENTER)) env.actions[2] = 2;
        }
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    close_client(env.client);
    free_puffernet(net);
    free(weights);
}

int main() {
    demo();
    return 0;
}
