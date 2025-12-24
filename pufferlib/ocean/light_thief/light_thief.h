#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GRID_SIZE 32
#define MAX_SEARCHLIGHTS 3
#define MAX_LOOT 5

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
    bool done;
} LightThief;

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

void compute_observations(LightThief* state) {
    float* obs = state->observations;
    obs[0] = state->agent_x / (float)GRID_SIZE;
    obs[1] = state->agent_y / (float)GRID_SIZE;
    
    for (int i = 0; i < MAX_SEARCHLIGHTS; i++) {
        obs[2 + i*2] = state->lights[i].x / (float)GRID_SIZE;
        obs[3 + i*2] = state->lights[i].y / (float)GRID_SIZE;
    }
    
    for (int i = 0; i < MAX_LOOT; i++) {
        bool illuminated = is_illuminated(state, state->loot[i].x, state->loot[i].y);
        if (illuminated) {
            obs[8 + i*3] = state->loot[i].x / (float)GRID_SIZE;
            obs[9 + i*3] = state->loot[i].y / (float)GRID_SIZE;
        } else {
            obs[8 + i*3] = -1.0f;
            obs[9 + i*3] = -1.0f;
        }
        obs[10 + i*3] = state->loot[i].active ? 1.0f : 0.0f;
    }
}

void c_reset(LightThief* state) {
    state->agent_x = GRID_SIZE / 2.0f;
    state->agent_y = GRID_SIZE / 2.0f;
    state->score = 0;
    state->steps = 0;
    state->done = false;

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
    if (state->done) {
        c_reset(state);
        return;
    }

    state->steps++;
    float reward = -0.01f; // Step penalty

    // Action Space (0: up, 1: down, 2: left, 3: right, 4: stay)
    int action = state->actions[0];
    float speed = 0.5f;
    if (action == 0) state->agent_y += speed;
    else if (action == 1) state->agent_y -= speed;
    else if (action == 2) state->agent_x -= speed;
    else if (action == 3) state->agent_x += speed;

    // Boundary checks
    if (state->agent_x < 0) state->agent_x = 0;
    if (state->agent_x >= GRID_SIZE) state->agent_x = GRID_SIZE - 1;
    if (state->agent_y < 0) state->agent_y = 0;
    if (state->agent_y >= GRID_SIZE) state->agent_y = GRID_SIZE - 1;

    // Update lights
    for (int i = 0; i < MAX_SEARCHLIGHTS; i++) {
        state->lights[i].x += state->lights[i].vx;
        state->lights[i].y += state->lights[i].vy;

        // Bounce off walls
        if (state->lights[i].x < 0 || state->lights[i].x >= GRID_SIZE) state->lights[i].vx *= -1;
        if (state->lights[i].y < 0 || state->lights[i].y >= GRID_SIZE) state->lights[i].vy *= -1;
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
    state->log.episode_return += reward;

    // Check if all loot collected or max steps
    bool any_loot = false;
    for (int i = 0; i < MAX_LOOT; i++) {
        if (state->loot[i].active) any_loot = true;
    }

    if (!any_loot || state->steps >= 500) {
        state->done = true;
        state->terminals[0] = 1;
        state->log.score = state->score;
        state->log.episode_length = state->steps;
        state->log.n += 1;
    } else {
        state->terminals[0] = 0;
    }

    compute_observations(state);
}

void c_render(LightThief* state) {
    // Placeholder for now, as it requires raylib for standard ocean envs
}

void c_close(LightThief* state) {
}
