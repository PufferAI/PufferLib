#include <stdio.h>
#include <time.h>
#include "lightsout.h"

int demo(){
    srand((unsigned)time(NULL));
    LightsOut env = {.grid_size = 7, .cell_size = 100, .client = NULL};
    env.observations = (unsigned char*)calloc(env.grid_size * env.grid_size, sizeof(unsigned char));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    c_reset(&env);
    env.client = make_client(env.cell_size, env.grid_size);

    // printf("LightsOut template ran 10 placeholder steps.\n");
    while (!WindowShouldClose()) {
        // User can take control of the first snake
        if (IsKeyPressed(KEY_UP)    || IsKeyPressed(KEY_W)) env.client->cursor_row = (env.client->cursor_row - 1 + env.grid_size) % env.grid_size;
        if (IsKeyPressed(KEY_DOWN)  || IsKeyPressed(KEY_S)) env.client->cursor_row = (env.client->cursor_row + 1) % env.grid_size;
        if (IsKeyPressed(KEY_LEFT)  || IsKeyPressed(KEY_A)) env.client->cursor_col = (env.client->cursor_col - 1 + env.grid_size) % env.grid_size;
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) env.client->cursor_col = (env.client->cursor_col + 1) % env.grid_size;
        if (IsKeyPressed(KEY_SPACE)) {
            int idx = env.client->cursor_row * env.grid_size + env.client->cursor_col;
            env.actions[0] = idx;
            c_step(&env);
        } else if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
int main(void) {
    demo();
    return 0;
}
