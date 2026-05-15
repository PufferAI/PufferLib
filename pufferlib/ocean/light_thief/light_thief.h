#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define GRID_SIZE 32
#define MAX_SEARCHLIGHTS 3
#define MAX_LOOT 5

enum {
    ACTION_UP = 0,
    ACTION_DOWN = 1,
    ACTION_LEFT = 2,
    ACTION_RIGHT = 3,
    ACTION_STAY = 4,
};

typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float n;
} Log;

typedef struct {
    float x, y;
    float vx, vy;
    float radius;
} Searchlight;

typedef struct {
    float x, y;
    bool active;
} Loot;

typedef struct LightThief {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    Log log;

    float agent_x, agent_y;
    Searchlight lights[MAX_SEARCHLIGHTS];
    Loot loot[MAX_LOOT];
    int score;
    int steps;

    float episode_return;
} LightThief;

static inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline bool is_illuminated(LightThief* state, float x, float y) {
    for (int i = 0; i < MAX_SEARCHLIGHTS; i++) {
        float dx = x - state->lights[i].x;
        float dy = y - state->lights[i].y;
        if (sqrtf(dx*dx + dy*dy) < state->lights[i].radius) {
            return true;
        }
    }
    return false;
}

static inline void add_log(LightThief* state) {
    state->log.episode_return += state->episode_return;
    state->log.episode_length += (float)state->steps;
    state->log.score += (float)state->score;
    state->log.n += 1.0f;
}

static inline void compute_observations(LightThief* state) {
    float* obs = state->observations;
    obs[0] = state->agent_x / (float)GRID_SIZE;
    obs[1] = state->agent_y / (float)GRID_SIZE;

    for (int i = 0; i < MAX_SEARCHLIGHTS; i++) {
        obs[2 + i*2] = state->lights[i].x / (float)GRID_SIZE;
        obs[3 + i*2] = state->lights[i].y / (float)GRID_SIZE;
    }

    for (int i = 0; i < MAX_LOOT; i++) {
        if (!state->loot[i].active) {
            obs[8 + i*3] = -1.0f;
            obs[9 + i*3] = -1.0f;
            obs[10 + i*3] = 0.0f;
            continue;
        }

        bool illuminated = is_illuminated(state, state->loot[i].x, state->loot[i].y);
        if (illuminated) {
            obs[8 + i*3] = state->loot[i].x / (float)GRID_SIZE;
            obs[9 + i*3] = state->loot[i].y / (float)GRID_SIZE;
        } else {
            obs[8 + i*3] = -1.0f;
            obs[9 + i*3] = -1.0f;
        }
        obs[10 + i*3] = 1.0f;
    }
}

void c_reset(LightThief* state) {
    state->agent_x = GRID_SIZE / 2.0f;
    state->agent_y = GRID_SIZE / 2.0f;
    state->score = 0;
    state->steps = 0;
    state->episode_return = 0.0f;

    for (int i = 0; i < MAX_SEARCHLIGHTS; i++) {
        state->lights[i].x = (float)(rand() % GRID_SIZE);
        state->lights[i].y = (float)(rand() % GRID_SIZE);
        state->lights[i].vx = ((float)(rand() % 100) / 100.0f) * 0.5f - 0.25f;
        state->lights[i].vy = ((float)(rand() % 100) / 100.0f) * 0.5f - 0.25f;
        state->lights[i].radius = 4.0f;
    }

    for (int i = 0; i < MAX_LOOT; i++) {
        state->loot[i].x = (float)(rand() % GRID_SIZE);
        state->loot[i].y = (float)(rand() % GRID_SIZE);
        state->loot[i].active = true;
    }

    compute_observations(state);
}

void c_step(LightThief* state) {
    state->terminals[0] = 0;

    state->steps++;
    float reward = -0.01f; // Step penalty

    // Action Space (0: up, 1: down, 2: left, 3: right, 4: stay)
    int action = state->actions[0];
    float speed = 0.5f;
    if (action == ACTION_UP) state->agent_y += speed;
    else if (action == ACTION_DOWN) state->agent_y -= speed;
    else if (action == ACTION_LEFT) state->agent_x -= speed;
    else if (action == ACTION_RIGHT) state->agent_x += speed;

    // Boundary checks
    state->agent_x = clampf(state->agent_x, 0.0f, GRID_SIZE - 1.0f);
    state->agent_y = clampf(state->agent_y, 0.0f, GRID_SIZE - 1.0f);

    // Update lights
    for (int i = 0; i < MAX_SEARCHLIGHTS; i++) {
        state->lights[i].x += state->lights[i].vx;
        state->lights[i].y += state->lights[i].vy;

        // Bounce off walls (and clamp to keep observations bounded)
        if (state->lights[i].x < 0.0f) {
            state->lights[i].x = 0.0f;
            state->lights[i].vx *= -1;
        } else if (state->lights[i].x > GRID_SIZE - 1.0f) {
            state->lights[i].x = GRID_SIZE - 1.0f;
            state->lights[i].vx *= -1;
        }

        if (state->lights[i].y < 0.0f) {
            state->lights[i].y = 0.0f;
            state->lights[i].vy *= -1;
        } else if (state->lights[i].y > GRID_SIZE - 1.0f) {
            state->lights[i].y = GRID_SIZE - 1.0f;
            state->lights[i].vy *= -1;
        }
    }

    bool illuminated = is_illuminated(state, state->agent_x, state->agent_y);

    // Reward Logic: Collect loot in the dark
    for (int i = 0; i < MAX_LOOT; i++) {
        if (!state->loot[i].active) continue;

        float dx = state->agent_x - state->loot[i].x;
        float dy = state->agent_y - state->loot[i].y;
        if (sqrtf(dx*dx + dy*dy) < 1.5f) {
            if (!illuminated) {
                state->loot[i].active = false;
                state->score++;
                reward += 1.0f;
            }
        }
    }

    if (illuminated) {
        reward -= 0.1f; // Exposure penalty
    }

    state->rewards[0] = reward;
    state->episode_return += reward;

    // Check if all loot collected or max steps
    bool any_loot = false;
    for (int i = 0; i < MAX_LOOT; i++) {
        if (state->loot[i].active) any_loot = true;
    }

    if (!any_loot || state->steps >= 500) {
        state->terminals[0] = 1;
        add_log(state);
        c_reset(state);
        return;
    }

    compute_observations(state);
}

static inline Vector2 to_screen(float x, float y, float cell_px) {
    // Invert y so env "up" maps to screen up.
    return (Vector2){
        x*cell_px + cell_px*0.5f,
        (GRID_SIZE - 1.0f - y)*cell_px + cell_px*0.5f,
    };
}

void c_render(LightThief* state) {
    const int cell_px = 20;
    const int hud_px = 60;
    const int board_px = GRID_SIZE * cell_px;

    if (!IsWindowReady()) {
        InitWindow(board_px, board_px + hud_px, "PufferLib Light Thief");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 10, 18, 255});

    // Board background
    DrawRectangle(0, 0, board_px, board_px, (Color){3, 7, 12, 255});

    // Lights (draw first, so agent/loot appear above)
    for (int i = 0; i < MAX_SEARCHLIGHTS; i++) {
        Vector2 center = to_screen(state->lights[i].x, state->lights[i].y, (float)cell_px);
        float radius_px = state->lights[i].radius * cell_px;
        DrawCircleV(center, radius_px, (Color){255, 255, 120, 28});
        DrawCircleV(center, 4.0f, (Color){255, 255, 120, 180});
    }

    // Loot (only visible when illuminated)
    for (int i = 0; i < MAX_LOOT; i++) {
        if (!state->loot[i].active) continue;
        if (!is_illuminated(state, state->loot[i].x, state->loot[i].y)) continue;

        Vector2 p = to_screen(state->loot[i].x, state->loot[i].y, (float)cell_px);
        DrawRectangle((int)(p.x - 6), (int)(p.y - 6), 12, 12, (Color){240, 200, 40, 255});
    }

    // Agent
    bool illuminated = is_illuminated(state, state->agent_x, state->agent_y);
    Vector2 agent = to_screen(state->agent_x, state->agent_y, (float)cell_px);
    Color agent_color = illuminated ? (Color){255, 90, 90, 255} : (Color){40, 200, 255, 255};
    DrawCircleV(agent, 7.0f, agent_color);

    // HUD
    DrawRectangle(0, board_px, board_px, hud_px, (Color){8, 16, 24, 255});
    DrawText(TextFormat("Score: %d   Steps: %d", state->score, state->steps), 10, board_px + 10, 20, RAYWHITE);
    DrawText("Hold SHIFT for manual control (WASD/arrows)", 10, board_px + 32, 18, (Color){200, 200, 200, 255});

    EndDrawing();
}

void c_close(LightThief* state) {
    (void)state;
    if (IsWindowReady()) {
        CloseWindow();
    }
}
