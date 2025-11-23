#ifndef LOCK_KEY_H
#define LOCK_KEY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

static const Color PUFF_RED        = (Color){187, 0, 0, 255};
static const Color PUFF_CYAN       = (Color){0, 187, 187, 255};
static const Color PUFF_GREEN      = (Color){0, 187, 0, 255};
static const Color PUFF_BACKGROUND = (Color){65, 30, 40, 255};
static const Color PUFF_BLACK      = (Color){0, 0, 0, 255};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;

    // observations: partial view for agent
    unsigned char* observations;
    // state: full system state
    unsigned char* state;

    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;

    int size;
    int num_keys;
    int tick;
    int x;
    int y;
    int num_keys_collected;
    int obs_dist;
} LockKey;

static inline int lk_pos(LockKey* env, int x, int y) {
    return y * env->size + x;
}

static inline int lk_visible(LockKey* env, int x, int y) {
    int dx = x - env->x; if (dx < 0) dx = -dx;
    int dy = y - env->y; if (dy < 0) dy = -dy;
    return (dx > dy ? dx : dy) <= env->obs_dist;
}

static inline void lk_update_observations(LockKey* env) {
    int tiles = env->size * env->size;
    memset(env->observations, 0, tiles * sizeof(unsigned char));

    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            if (!lk_visible(env, x, y)) continue;
            int pos = lk_pos(env, x, y);
            env->observations[pos] = env->state[pos];
        }
    }
}

void add_log(LockKey* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.n++;
}

static inline void c_reset(LockKey* env) {
    int tiles = env->size * env->size;
    memset(env->state, 0, tiles * sizeof(unsigned char));

    env->x = env->size / 2;
    env->y = env->size / 2;
    int player_pos = lk_pos(env, env->x, env->y);
    env->state[player_pos] = 1;
    env->tick = 0;

    int lock_idx;
    do lock_idx = rand() % tiles;
    while (lock_idx == player_pos);
    env->state[lock_idx] = 2;

    for (int i = 0; i < env->num_keys; i++) {
        int key_idx;
        do key_idx = rand() % tiles;
        while (env->state[key_idx] != 0);
        env->state[key_idx] = 3;
    }

    env->num_keys_collected = 0;
    lk_update_observations(env);
}

static inline void c_step(LockKey* env) {
    env->tick++;
    env->rewards[0] = -0.1f;
    env->terminals[0] = 0;
    if (env->truncations) env->truncations[0] = 0;

    int prev_pos = lk_pos(env, env->x, env->y);
    if (env->state[prev_pos] != 2)
        env->state[prev_pos] = 0;

    int a = env->actions[0];
    if (a == 0) env->x--;
    else if (a == 1) env->x++;
    else if (a == 2) env->y--;
    else if (a == 3) env->y++;

    int max_steps = 3*env->size + env->num_keys*env->num_keys;
    if (env->tick > max_steps || env->x < 0 || env->x >= env->size || env->y < 0 || env->y >= env->size) {
        env->rewards[0] = -3.0f;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    int pos = lk_pos(env, env->x, env->y);

    if (env->state[pos] == 3) {
        env->rewards[0] += 1.0f;
        env->num_keys_collected++;
    }

    if (env->state[pos] == 2 && env->num_keys_collected == env->num_keys) {
        env->rewards[0] = 3.0f;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    if (env->state[pos] != 2)
        env->state[pos] = 1;

    lk_update_observations(env);
}

static inline void c_render(LockKey* env) {
    if (!IsWindowReady()) {
        InitWindow(64*env->size, 64*env->size, "LockKey");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();

    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            Color bg = lk_visible(env, x, y) ? PUFF_BACKGROUND : PUFF_BLACK;
            DrawRectangle(x * 64, y * 64, 64, 64, bg);

            int pos = lk_pos(env, x, y);
            unsigned char v = env->observations[pos];
            if (!v) continue;

            Color color =
                (v == 1) ? PUFF_CYAN :
                (v == 2) ? PUFF_RED :
                (v == 3) ? PUFF_GREEN :
                PUFF_BACKGROUND;

            DrawRectangle(x * 64, y * 64, 64, 64, color);
        }
    }

    EndDrawing();
}

static inline void c_close(LockKey* env) {
    (void)env;
    if (IsWindowReady()) CloseWindow();
}

#endif
