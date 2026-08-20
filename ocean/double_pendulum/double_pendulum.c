#include "double_pendulum.h"

int main(void) {
    float observations[DP_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};

    DoublePendulum env = {
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
        .num_agents = 1,
        .rng = 1,
        .cart_mass = 1.0f,
        .link1_mass = 0.1f,
        .link2_mass = 0.1f,
        .link1_length = 0.5f,
        .link2_length = 0.5f,
        .gravity = 9.8f,
        .force_mag = 10.0f,
        .dt = 0.02f,
    };

    init(&env);
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) actions[0] = 0;
        else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) actions[0] = 2;
        else actions[0] = 1;
        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    return 0;
}
