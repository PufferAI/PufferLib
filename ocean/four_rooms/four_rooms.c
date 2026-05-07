#include "four_rooms.h"

int main() {
    FourRooms env = {};
    env.size = 19;
    env.max_steps = 0;
    env.num_agents = 1;
    env.rng = 0;
    env.observations = (unsigned char*)calloc(
        FOUR_ROOMS_VIEW_SIZE * FOUR_ROOMS_VIEW_SIZE * FOUR_ROOMS_OBS_CHANNELS,
        sizeof(unsigned char)
    );
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    env.grid = (unsigned char*)calloc(env.size * env.size, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 7; // Invalid action = no-op
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) env.actions[0] = FORWARD;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
        } else {
            env.actions[0] = four_rooms_rand(&env, 3); // Only use left, right, forward
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
