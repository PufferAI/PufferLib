#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "raylib.h"

#define GRID_SIZE 5
#define TOTAL_CELLS (GRID_SIZE * GRID_SIZE)
#define CELL_SIZE 128
#define NOOP -1.0f
#define ATTEMPTS_PER_EPISODE 3

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Texture2D puffer;
    int flash_timer;      
    Color flash_color;   
    int last_action_r;   
    int last_action_c;
    bool show_flash;    
} Client;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    int mole_r;
    int mole_c;
    int hits;
    int tick;
    Client* client;
} Whackamole;

void add_log(Whackamole* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n += 1.0f;
}

void c_reset(Whackamole* env) {
    memset(env->observations, 0, sizeof(float) * TOTAL_CELLS);
    
    int mole_idx = rand_r(&env->rng) % TOTAL_CELLS;
    env->observations[mole_idx] = 1.0f;
    env->mole_r = mole_idx / GRID_SIZE;
    env->mole_c = mole_idx % GRID_SIZE;
    
    env->tick = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    
    if (env->client != NULL) {
        env->client->show_flash = false;
        env->client->flash_timer = 0;
    }
}

void c_step(Whackamole* env) {
    env->tick += 1;
    
    int action = (int)env->actions[0];
    int mole_idx = env->mole_r * GRID_SIZE + env->mole_c;
    
    if (env->client != NULL) {
        env->client->show_flash = false;
    }
    
    if (action == (int)NOOP || action < 0 || action >= TOTAL_CELLS) {
        env->rewards[0] = 0.0f;
    } else if (action == mole_idx) {
        env->rewards[0] = 1.0f;
        env->hits += 1;
        // Flash GREEN for hit
        if (env->client != NULL) {
            env->client->flash_color = (Color){0, 255, 0, 180};
            env->client->flash_timer = 15;
            env->client->show_flash = true;
            env->client->last_action_r = action / GRID_SIZE;
            env->client->last_action_c = action % GRID_SIZE;
        }
    } else {
        int action_r = action / GRID_SIZE;
        int action_c = action % GRID_SIZE;
        int dist = abs(action_r - env->mole_r) + abs(action_c - env->mole_c);
        env->rewards[0] = fmaxf(0.0f, 1.0f - dist * 0.25f);
        // Flash RED for miss
        if (env->client != NULL) {
            env->client->flash_color = (Color){255, 0, 0, 180};
            env->client->flash_timer = 15;
            env->client->show_flash = true;
            env->client->last_action_r = action_r;
            env->client->last_action_c = action_c;
        }
    }
    
    if (env->tick >= ATTEMPTS_PER_EPISODE) {
        env->terminals[0] = 1.0f;
        add_log(env);
        c_reset(env);
    } else {
        env->terminals[0] = 0.0f;
        // Move puffre for next attempt
        int new_idx = rand_r(&env->rng) % TOTAL_CELLS;
        env->observations[mole_idx] = 0.0f;
        env->observations[new_idx] = 1.0f;
        env->mole_r = new_idx / GRID_SIZE;
        env->mole_c = new_idx % GRID_SIZE;
    }
}

void c_render(Whackamole* env) {
    if (!IsWindowReady()) {
        InitWindow(CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE, "PufferLib WhacKe-a-PUFFER");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->puffer = LoadTexture("pufferfish.png");
        if (env->client->puffer.id == 0) {
            env->client->puffer = LoadTexture("ocean/whackamole/pufferfish.png");
    }
        env->client->show_flash = false;
        env->client->flash_timer = 0;
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    BeginDrawing();
    ClearBackground((Color){34, 139, 34, 255});
    
    for (int r = 0; r < GRID_SIZE; r++) {
        for (int c = 0; c < GRID_SIZE; c++) {
            int cx = c * CELL_SIZE + CELL_SIZE / 2;
            int cy = r * CELL_SIZE + CELL_SIZE / 2;
            DrawCircle(cx, cy, CELL_SIZE / 3, DARKGRAY);
        }
    }
    
    for (int i = 1; i < GRID_SIZE; i++) {
        int pos = i * CELL_SIZE;
        DrawLine(pos, 0, pos, CELL_SIZE * GRID_SIZE, BLACK);
        DrawLine(0, pos, CELL_SIZE * GRID_SIZE, pos, BLACK);
    }
    
    if (env->client->show_flash && env->client->flash_timer > 0) {
        int fx = env->client->last_action_c * CELL_SIZE;
        int fy = env->client->last_action_r * CELL_SIZE;
        DrawRectangle(fx, fy, CELL_SIZE, CELL_SIZE, env->client->flash_color);
        env->client->flash_timer--;
        if (env->client->flash_timer <= 0) {
            env->client->show_flash = false;
        }
    }
    
    int x = env->mole_c * CELL_SIZE;
    int y = env->mole_r * CELL_SIZE;
    
    if (env->client->puffer.id > 0) {
        float scale = (float)CELL_SIZE / env->client->puffer.width;
        DrawTextureEx(env->client->puffer, (Vector2){x, y}, 0.0f, scale, WHITE);
    } else {
        DrawCircle(x + CELL_SIZE/2, y + CELL_SIZE/2, CELL_SIZE/3, RED);
        DrawCircle(x + CELL_SIZE/2, y + CELL_SIZE/2, CELL_SIZE/4, YELLOW);
    }
    
    DrawText(TextFormat("Hits: %i", env->hits), 10, 10, 24, WHITE);
    DrawText(TextFormat("Return: %.1f", env->log.episode_return), 10, 40, 20, WHITE);
    DrawText(TextFormat("Attempt: %i/%i", env->tick + 1, ATTEMPTS_PER_EPISODE), 10, 70, 20, WHITE);
    
    EndDrawing();
}

void c_close(Whackamole* env) {
    if (env->client != NULL) {
        if (env->client->puffer.id > 0) {
            UnloadTexture(env->client->puffer);
        }
        free(env->client);
        env->client = NULL;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}