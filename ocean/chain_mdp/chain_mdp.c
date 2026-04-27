#include "chain_mdp.h"

int main() {
    Chain env = {.size = 64};
    allocate_chain(&env);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = 0;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
        } else {
            env.actions[0] = rand() % 2;
        }
        c_step(&env);
        c_render(&env);
    }
    free_allocated(&env);
}

