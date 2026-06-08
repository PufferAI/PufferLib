#include "whackamole.h"

int main() {
    Whackamole env = {0};
    env.num_agents = 1;
    env.rng = (unsigned int)time(NULL);
    
    env.observations = (float*)calloc(TOTAL_CELLS, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));
    
    c_reset(&env);
    c_render(&env);
    
    int frame = 0;
    while (1) {
        frame += 1;
        
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            env.actions[0] = NOOP;
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Vector2 mouse = GetMousePosition();
                int c = (int)(mouse.x / CELL_SIZE);
                int r = (int)(mouse.y / CELL_SIZE);
                if (r >= 0 && r < GRID_SIZE && c >= 0 && c < GRID_SIZE) {
                    env.actions[0] = (float)(r * GRID_SIZE + c);
                }
            }
            if (IsKeyPressed(KEY_R)) c_reset(&env);
        } else {
            if (frame % 10 == 0) {
                env.actions[0] = (float)(rand_r(&env.rng) % TOTAL_CELLS);
            } else {
                env.actions[0] = NOOP;
            }
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