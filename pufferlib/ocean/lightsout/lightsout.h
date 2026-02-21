#include <stdlib.h>
#include "raylib.h"

// Only use floats.
typedef struct {
    float score;
    float n; // Required as the last field.
} Log;

typedef struct Client {
    int cell_size;
    int grid_size;
    int cursor_row;
    int cursor_col;
} Client;

typedef struct {
    Log log;                     // Required field.
    unsigned char* observations; // Required field. Ensure type matches in .py and .c.
    int* actions;                // Required field. Ensure type matches in .py and .c.
    float* rewards;              // Required field.
    unsigned char* terminals;    // Required field.
    int grid_size;
    int cell_size;
    int max_steps;
    int step_count;
    float episode_return;
    unsigned char* grid;
    Client* client;
} LightsOut;

int is_solved(LightsOut* env) {
    for (int i = 0; i < env->grid_size * env->grid_size; i++) {
        if (env->grid[i] == 1) return 0; // Not solved if any light is on.
    }
    return 1; // Solved if all lights are off.
}

int count_lights_on(LightsOut* env) {
    int on = 0;
    for (int i = 0; i < env->grid_size * env->grid_size; i++) {
        on += env->grid[i] != 0;
    }
    return on;
}

void step_grid(LightsOut* env, int idx) {
    if (idx < 0 || idx >= env->grid_size * env->grid_size) return;
    int row = idx/env->grid_size;
    int col = idx%env->grid_size;
    
    static const int dirs[5][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}};
    for (int i = 0; i < 5; i++) {
        int dr = dirs[i][0];
        int dc = dirs[i][1];
        int r = row + dr;
        int c = col + dc;
        if (r >= 0 && r < env->grid_size && c >= 0 && c < env->grid_size) {
            int offset = r*env->grid_size + c;
            env->grid[offset] = !env->grid[offset];
        }
    }
}

void init_lightsout(LightsOut* env) {
    int n = env->grid_size * env->grid_size;
    if (env->grid == NULL) {
        env->grid = (unsigned char*)calloc(n, sizeof(unsigned char));
    } else {
        for (int i = 0; i < n; i++) {
            env->grid[i] = 0;
        }
    }
    env->step_count = 0;
    env->episode_return = 0.0f;

    float p = 0.5f;  // scramble probability per cell

    for (int i = 0; i < n; i++) {
        float u = (float)rand() / (float)RAND_MAX;  // ~uniform in [0,1]
        if (u < p) {
            step_grid(env, i);
        }
    }
}

void c_close(LightsOut* env) {
    free(env->grid);
    env->grid = NULL;
    if (env->client != NULL) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

void compute_observations(LightsOut* env) {
    for (int i = 0; i < env->grid_size * env->grid_size; i++) {
        env->observations[i] = env->grid[i];
    }
}

void c_reset(LightsOut* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    init_lightsout(env);
    compute_observations(env);
}

void c_step(LightsOut* env) {
    // In manual mode, keep solved screen visible until user resets.
    if (env->client != NULL && env->terminals[0]) {
        env->rewards[0] = 0.0f;
        compute_observations(env);
        return;
    }

    int num_cells = env->grid_size * env->grid_size;
    int atn = env->actions[0];
    env->terminals[0] = 0;

    float reward = -0.02f; // Base step penalty.
    int prev_on = count_lights_on(env);
    if (atn < 0 || atn >= num_cells) {
        reward -= 0.5f; // Invalid action penalty.
    } else {
        if (env->client != NULL) {
            env->client->cursor_row = atn / env->grid_size;
            env->client->cursor_col = atn % env->grid_size;
        }
        step_grid(env, atn);
        int next_on = count_lights_on(env);
        reward += 0.005f * (float)(prev_on - next_on); // Dense shaping: improve when lights decrease.
    }
    env->step_count += 1;

    if (is_solved(env)) {
        reward = 1.0f; // Solved reward.
        env->terminals[0] = 1;
    } else if (env->client == NULL && env->step_count >= env->max_steps) {
        reward -= 0.5f; // Timeout penalty during training.
        env->terminals[0] = 1;
    }

    env->rewards[0] = reward;
    env->episode_return += reward;
    if (env->terminals[0]) {
        env->log.n += 1.0f;
        env->log.score += env->episode_return;
        if (env->client == NULL) {
            init_lightsout(env);
        }
    }
    compute_observations(env);
}

// Raylib client
Color COLORS[] = {
    (Color){6, 24, 24, 255},
    (Color){0, 0, 255, 255},
    (Color){0, 128, 255, 255},
    (Color){128, 128, 128, 255},
    (Color){255, 0, 0, 255},
    (Color){255, 255, 255, 255},
    (Color){255, 85, 85, 255},
    (Color){170, 170, 170, 255},
    (Color){0, 255, 255, 255},
    (Color){255, 255, 0, 255},
};

Client* make_client(int cell_size, int grid_size) {
    Client* client= (Client*)malloc(sizeof(Client));
    client->cell_size = cell_size;
    client->grid_size = grid_size;
    client->cursor_row = 0;
    client->cursor_col = 0;
    InitWindow(grid_size*cell_size, grid_size*cell_size, "PufferLib LightsOut");
    SetTargetFPS(3);
    return client;
}

void c_render(LightsOut* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = make_client(env->cell_size, env->grid_size);
    }
    
    Client* client = env->client;
    
    BeginDrawing();
    ClearBackground(COLORS[0]);
    int sz = client->cell_size;
    for (int y = 0; y < env->grid_size; y++) {
        for (int x = 0; x < env->grid_size; x++){
            int tile = env->grid[y*env->grid_size + x];
            if (tile != 0)
                DrawRectangle(x*sz, y*sz, sz, sz, COLORS[tile]);
        }
    }
    DrawRectangleLinesEx(
        (Rectangle){client->cursor_col * sz, client->cursor_row * sz, sz, sz},
        3.0f,
        COLORS[5]
    );

    if (env->terminals[0]) {
        const char* msg = "Solved";
        int font_size = 48;
        int text_w = MeasureText(msg, font_size);
        int screen_w = env->grid_size * env->cell_size;
        int screen_h = env->grid_size * env->cell_size;

        DrawRectangle(0, 0, screen_w, screen_h, (Color){0, 0, 0, 120}); // dim overlay
        DrawText(msg, (screen_w - text_w) / 2, (screen_h - font_size) / 2, font_size, RAYWHITE);
    }

    EndDrawing();
}
