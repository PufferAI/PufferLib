#include "four_rooms.h"

int main() {
    FourRooms env = {};
    env.size = 19;
    env.max_steps = 0;
    env.num_agents = 1;
    env.rng = 0;
    allocate(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = DONE;
            if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) env.actions[0] = FORWARD;
            if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
        } else {
            env.actions[0] = four_rooms_rand(&env, 3);
        }
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
    return 0;
}
