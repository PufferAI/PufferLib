#include "radars.h"
#include "puffernet.h"

int main() {

    Radars env = {.initial_targets = 5};
    allocate(&env);

    Client* client = make_client(&env);

    c_reset(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if (IsKeyDown(KEY_ZERO)) env.actions[0] = SEARCH;
            if (IsKeyDown(KEY_ONE)) env.actions[0] = TRACK1;
            if (IsKeyDown(KEY_TWO)) env.actions[0] = TRACK2;
            if (IsKeyDown(KEY_THREE)) env.actions[0] = TRACK3;
            if (IsKeyDown(KEY_FOUR)) env.actions[0] = TRACK4;
            if (IsKeyDown(KEY_FIVE)) env.actions[0] = TRACK5;
        } else {
            env.actions[0] = NOOP;
        }
        c_step(&env);
        c_render(client, &env);
    }
    free_allocated(&env);
    close_client(client);
}

