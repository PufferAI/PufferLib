#include "tmaze.h"

int main() {
    TMaze env = {.size = 8};
    allocate_TMaze(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = FORWARD;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;

        } else {
            env.actions[0] = rand() % 3;
        }
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
}

