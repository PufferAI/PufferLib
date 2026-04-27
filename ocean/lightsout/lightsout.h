#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "raylib.h"

// Only use floats.
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float scramble_p;
    float n; // Required as the last field.
} Log;

typedef struct Client {
    int cell_size;
    int cursor_row;
    int cursor_col;
} Client;

typedef struct {
    Log log;                     // Required field.
    unsigned char* observations; // Required field. Ensure type matches in .py and .c.
    float* actions;              // Required field. Ensure type matches in .py and .c.
    float* rewards;              // Required field.
    float* terminals;            // Required field.
    int grid_size;
    int cell_size;
    int max_steps;
    int step_count;
    int lights_on;
    int prev_action;
    int last_action;
    float episode_return;
    float ema;
    float score_ema;
    float scramble_prob;
    unsigned char* grid;
    Client* client;
    int num_agents;
    int observation_size;
    unsigned int rng;
} LightsOut;

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
            unsigned char old = env->grid[offset];
            env->grid[offset] = (unsigned char)!old;
            env->lights_on += old ? -1 : 1;
        }
    }
}

void init_lightsout(LightsOut* env) {
    int n = env->grid_size * env->grid_size;
    if (env->grid == NULL) {
        env->grid = (unsigned char*)calloc(n, sizeof(unsigned char));
    } else {
        memset(env->grid, 0, n * sizeof(unsigned char));
    }

    if (env->ema > 0.7f && env->score_ema > 0.0f) {
        env->scramble_prob = fminf(0.5f, env->scramble_prob + 0.01f); // Increase scramble prob if EMA is high
    } else if (env->ema < 0.3f) {
        env->scramble_prob = fmaxf(0.15f, env->scramble_prob - 0.01f); // Decrease scramble prob if EMA is low
    }

    env->step_count = 0;
    env->lights_on = 0;
    env->prev_action = -1;
    env->last_action = -1;
    env->episode_return = 0.0f;

    for (int i = 0; i < n; i++) {
        float u = (float)rand_r(&env->rng) / (float)RAND_MAX;
        if (u < env->scramble_prob) {
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
    env->terminals[0] = 0.0f;
    init_lightsout(env);
    compute_observations(env);
}

void c_step(LightsOut* env) {
    int num_cells = env->grid_size * env->grid_size;
    int atn = env->actions[0];
    env->terminals[0] = 0.0f;

    float reward = -0.02 * (36.0 / (env->grid_size * env->grid_size)); // Base step penalty.
    int prev_on = env->lights_on;
    if (atn < 0 || atn >= num_cells) {
        reward -= 0.5f; // Invalid action penalty.
    } else {
        if (atn == env->last_action) {
            reward -= 0.03f; // Penalty for pressing the same cell twice in a row.
        } else if (atn == env->prev_action) {
            reward -= 0.02f; // Penalty for 2-step loop (A,B,A).
        }
        if (env->client != NULL) {
            env->client->cursor_row = atn / env->grid_size;
            env->client->cursor_col = atn % env->grid_size;
        }
        step_grid(env, atn);
        env->prev_action = env->last_action;
        env->last_action = atn;
        int next_on = env->lights_on;
        reward += 0.005f * (float)(prev_on - next_on); // Dense shaping: improve when lights decrease.
    }
    env->step_count += 1;

    if (env->lights_on == 0) {
        reward = 2.0f; // Solved reward.
        env->ema = 0.85f * env->ema + 0.15f; // Update EMA of steps to solve.
        env->terminals[0] = 1.0f;
    } else if (env->client == NULL && env->step_count >= env->max_steps) {
        reward -= 0.5f; // Timeout penalty during training.
        env->ema = 0.85f * env->ema; // Decay EMA since we failed to solve.
        env->terminals[0] = 1.0f;
    }

    env->rewards[0] = reward;
    env->episode_return += reward;

    if (env->terminals[0] > 0.0f) {
        env->log.episode_return += env->episode_return;
        env->log.episode_length += (float)env->step_count;
        env->log.n += 1.0f;
        env->log.perf += (env->lights_on == 0) ? 1.0f : 0.0f;
        env->log.score += env->episode_return;
        env->log.scramble_p += env->scramble_prob;

        env->score_ema = 0.9f * env->score_ema + 0.1f * env->episode_return;
        init_lightsout(env);
    }

    compute_observations(env);
}

// Raylib client
static const Color COLORS[] = {
    (Color){6, 24, 24, 255},
    (Color){0, 0, 255, 255},
    (Color){255, 255, 255, 255}
};

Client* make_client(int cell_size, int grid_size) {
    Client* client= (Client*)malloc(sizeof(Client));
    client->cell_size = cell_size;
    client->cursor_row = 0;
    client->cursor_col = 0;
    InitWindow(grid_size*cell_size, grid_size*cell_size, "PufferLib LightsOut");
    SetTargetFPS(5);
    return client;
}

void c_render(LightsOut* env) {
    if (IsWindowReady() && (WindowShouldClose() || IsKeyPressed(KEY_ESCAPE))) {
        c_close(env);
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
        COLORS[2]
    );

    if (env->terminals[0] > 0.0f) {
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
