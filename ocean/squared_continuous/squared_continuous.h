/* Squared Continuous: continuous action version of squared.
 * 2 continuous action dimensions: vertical and horizontal.
 * Actions are clamped to [-1, 1] and thresholded at 0.25 magnitude.
 */

#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    unsigned char* observations; // Required. You can use any obs type, but make sure it matches in Python!
    float* actions; // Required
    float* rewards; // Required
    float* terminals; // Required
    int num_agents;
    int size;
    int tick;
    int r;
    int c;
    unsigned int rng;
} Squared;

void add_log(Squared* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(Squared* env) {
    int tiles = env->size*env->size;
    memset(env->observations, 0, tiles*sizeof(unsigned char));
    env->observations[tiles/2] = AGENT;
    env->r = env->size/2;
    env->c = env->size/2;
    env->tick = 0;
    int target_idx = 0; // Deterministic for testing
    do {
        target_idx = rand_r(&env->rng) % tiles;
    } while (target_idx == tiles/2);
    env->observations[target_idx] = TARGET;
}

// Clamp value to [-1, 1]
static inline float clamp_action(float x) {
    return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

// Required function
void c_step(Squared* env) {
    env->tick += 1;

    // Continuous actions: clamp to [-1, 1] then threshold to get discrete movement
    // action[0]: vertical (positive = down, negative = up)
    // action[1]: horizontal (positive = right, negative = left)
    float vert = clamp_action(env->actions[0]);
    float horiz = clamp_action(env->actions[1]);
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->observations[env->r*env->size + env->c] = EMPTY;

    // Threshold at 0.25 magnitude to determine direction (allows stationary)
    if (vert > 0.25) {
        env->r += 1;  // DOWN
    } else if (vert < -0.25) {
        env->r -= 1;  // UP
    }
    if (horiz > 0.25) {
        env->c += 1;  // RIGHT
    } else if (horiz < -0.25) {
        env->c -= 1;  // LEFT
    }

    if (env->tick > 3*env->size
            || env->r < 0
            || env->c < 0
            || env->r >= env->size
            || env->c >= env->size) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    int pos = env->r*env->size + env->c;
    if (env->observations[pos] == TARGET) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }

    env->observations[pos] = AGENT;
}

// Required function. Should handle creating the client on first call
void c_render(Squared* env) {
    if (!IsWindowReady()) {
        InitWindow(64*env->size, 64*env->size, "PufferLib Squared Continuous");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int px = 64;
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            int tex = env->observations[i*env->size + j];
            if (tex == EMPTY) {
                continue;
            }
            Color color = (tex == AGENT) ? (Color){0, 187, 187, 255} : (Color){187, 0, 0, 255};
            DrawRectangle(j*px, i*px, px, px, color);
        }
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Squared* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
