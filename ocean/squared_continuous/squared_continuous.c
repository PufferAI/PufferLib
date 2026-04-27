/* Pure C demo file for Squared Continuous. Build it with:
 * bash scripts/build_ocean.sh squared_continuous local (debug)
 * bash scripts/build_ocean.sh squared_continuous fast
 */

#include "squared_continuous.h"
#include "puffernet.h"

void demo() {
    Squared env = {.size = 11};
    env.observations = (unsigned char*)calloc(env.size*env.size, sizeof(unsigned char));
    env.actions = (float*)calloc(2, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    Weights* weights = load_weights("resources/squared_continuous/squared_continuous_weights.bin");
    int logit_sizes[2] = {1, 1};
    PufferNet* net = make_puffernet(weights, 1, 121, 128, 1, logit_sizes, 2);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0.0f;
            env.actions[1] = 0.0f;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = -1.0f;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 1.0f;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[1] = -1.0f;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[1] = 1.0f;
        } else {
            float obs_f[121];
            for(int i=0; i<121; i++) obs_f[i] = (float)env.observations[i];
            forward_puffernet(net, obs_f, env.actions);
        }
        c_step(&env);
        c_render(&env);
    }
    
    free_puffernet(net);
    free(weights);

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

int main() {
    demo();
    return 0;
}
