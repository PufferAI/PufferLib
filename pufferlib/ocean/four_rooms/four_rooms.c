#include "four_rooms.h"

int main() {
    FourRooms env = {};
    env.size = 19;
    env.observations = (unsigned char*)calloc(7*7*3, sizeof(unsigned char)); // 7x7x3 for MinGrid encoding
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
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
            env.actions[0] = rand() % 3; // Only use left, right, forward
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.grid);
    c_close(&env);
    return 0;
}

