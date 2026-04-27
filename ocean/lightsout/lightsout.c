#include <stdio.h>
#include <time.h>
#include "lightsout.h"

static LightsOut* g_env = NULL;

static void demo_cleanup(void) {
    if (g_env == NULL) {
        return;
    }
    free(g_env->observations);
    free(g_env->actions);
    free(g_env->rewards);
    free(g_env->terminals);
    c_close(g_env);
    g_env = NULL;
}

int demo(){
    srand((unsigned)time(NULL));
    LightsOut env = {.grid_size = 5, .cell_size = 100, .client = NULL};
    g_env = &env;
    atexit(demo_cleanup);
    env.observations = (unsigned char*)calloc(env.grid_size * env.grid_size, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    env.client = make_client(env.cell_size, env.grid_size);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_UP)    || IsKeyPressed(KEY_W)) env.client->cursor_row = (env.client->cursor_row - 1 + env.grid_size) % env.grid_size;
        if (IsKeyPressed(KEY_DOWN)  || IsKeyPressed(KEY_S)) env.client->cursor_row = (env.client->cursor_row + 1) % env.grid_size;
        if (IsKeyPressed(KEY_LEFT)  || IsKeyPressed(KEY_A)) env.client->cursor_col = (env.client->cursor_col - 1 + env.grid_size) % env.grid_size;
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) env.client->cursor_col = (env.client->cursor_col + 1) % env.grid_size;
        if (IsKeyPressed(KEY_SPACE)) {
            int idx = env.client->cursor_row * env.grid_size + env.client->cursor_col;
            env.actions[0] = (float)idx;
            c_step(&env);
        } else if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        }
        c_render(&env);
    }

    demo_cleanup();
    return 0;
}
int main(void) {
    demo();
    return 0;
}
