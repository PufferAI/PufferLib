#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define FLAPPY_OBS_SIZE 7
#define FLAPPY_NUM_PIPES 3

#define FLAPPY_NOOP 0
#define FLAPPY_FLAP 1

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
struct Client {
    Texture2D puffer;
};

typedef struct Pipe Pipe;
struct Pipe {
    float x;
    float gap_y;
    int passed;
};

typedef struct Flappy Flappy;
struct Flappy {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    Log log;
    int num_agents;
    unsigned int rng;
    Client* client;

    int width;
    int height;
    int max_steps;
    float gravity;
    float flap_velocity;
    float pipe_speed;
    float pipe_gap;
    float pipe_width;
    float pipe_spacing;
    float first_pipe_x;
    float bird_x;
    float bird_radius;
    float alive_reward;
    float pass_reward;
    float crash_reward;
    float center_reward;

    float bird_y;
    float bird_vy;
    int tick;
    int score;
    float episode_return;
    Pipe pipes[FLAPPY_NUM_PIPES];
};

static inline float flappy_randf(Flappy* env) {
    return (float)rand_r(&env->rng) / (float)RAND_MAX;
}

static inline float flappy_clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

static inline float flappy_pipe_gap_y(Flappy* env) {
    float margin = env->pipe_gap * 0.6f;
    return margin + flappy_randf(env) * (env->height - 2.0f * margin);
}

static inline void flappy_init_pipe(Flappy* env, int i, float x) {
    env->pipes[i].x = x;
    env->pipes[i].gap_y = flappy_pipe_gap_y(env);
    env->pipes[i].passed = 0;
}

static inline void flappy_next_pipes(Flappy* env, Pipe** first, Pipe** second) {
    *first = NULL;
    *second = NULL;

    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        float dx = pipe->x + env->pipe_width - env->bird_x;
        if (dx < 0.0f) {
            continue;
        }

        if (*first == NULL || pipe->x < (*first)->x) {
            *second = *first;
            *first = pipe;
        } else if (*second == NULL || pipe->x < (*second)->x) {
            *second = pipe;
        }
    }
}

static inline void flappy_compute_observations(Flappy* env) {
    Pipe* pipe;
    Pipe* next_pipe;
    flappy_next_pipes(env, &pipe, &next_pipe);
    if (pipe == NULL) {
        pipe = &env->pipes[0];
    }
    if (next_pipe == NULL) {
        next_pipe = pipe;
    }

    float dx = (pipe->x + env->pipe_width - env->bird_x) / env->width;
    float dy = (env->bird_y - pipe->gap_y) / env->height;
    float next_dx = (next_pipe->x + env->pipe_width - env->bird_x) / env->width;

    env->observations[0] = env->bird_y / env->height;
    env->observations[1] = flappy_clampf(env->bird_vy / 16.0f, -1.0f, 1.0f);
    env->observations[2] = flappy_clampf(dx, 0.0f, 1.5f);
    env->observations[3] = pipe->gap_y / env->height;
    env->observations[4] = flappy_clampf(dy, -1.0f, 1.0f);
    env->observations[5] = flappy_clampf(next_dx, 0.0f, 2.0f);
    env->observations[6] = next_pipe->gap_y / env->height;
}

static inline void flappy_add_log(Flappy* env) {
    env->log.perf += flappy_clampf((float)env->score / 20.0f, 0.0f, 1.0f);
    env->log.score += env->score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.n += 1.0f;
}

void init(Flappy* env) {
    env->num_agents = 1;
    env->client = NULL;
    memset(&env->log, 0, sizeof(Log));
}

void c_reset(Flappy* env) {
    env->bird_y = env->height * 0.5f;
    env->bird_vy = 0.0f;
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0.0f;
    float start_x = env->first_pipe_x;
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        flappy_init_pipe(env, i, start_x + i * env->pipe_spacing);
    }

    flappy_compute_observations(env);
}

void c_step(Flappy* env) {
    env->tick += 1;
    env->rewards[0] = env->alive_reward;
    env->terminals[0] = 0.0f;

    int action = (int)env->actions[0];
    if (action == FLAPPY_FLAP) {
        env->bird_vy = env->flap_velocity;
    }

    env->bird_vy += env->gravity;
    env->bird_y += env->bird_vy;

    float max_x = 0.0f;
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        if (env->pipes[i].x > max_x) {
            max_x = env->pipes[i].x;
        }
    }

    bool done = false;
    if (env->bird_y - env->bird_radius < 0.0f || env->bird_y + env->bird_radius > env->height) {
        done = true;
    }

    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        pipe->x -= env->pipe_speed;

        if (!pipe->passed && pipe->x + env->pipe_width < env->bird_x) {
            pipe->passed = 1;
            env->score += 1;
            env->rewards[0] += env->pass_reward;
        }

        bool overlap_x = env->bird_x + env->bird_radius > pipe->x &&
            env->bird_x - env->bird_radius < pipe->x + env->pipe_width;
        bool outside_gap = env->bird_y - env->bird_radius < pipe->gap_y - env->pipe_gap * 0.5f ||
            env->bird_y + env->bird_radius > pipe->gap_y + env->pipe_gap * 0.5f;
        if (overlap_x && outside_gap) {
            done = true;
        }

        if (pipe->x + env->pipe_width < 0.0f) {
            flappy_init_pipe(env, i, max_x + env->pipe_spacing);
            max_x = env->pipes[i].x;
        }
    }

    Pipe* next;
    Pipe* ignored;
    flappy_next_pipes(env, &next, &ignored);
    if (next == NULL) {
        next = &env->pipes[0];
    }
    float center_error = fabsf(env->bird_y - next->gap_y) / (env->height * 0.5f);
    env->rewards[0] += env->center_reward * (1.0f - flappy_clampf(center_error, 0.0f, 1.0f));

    if (env->tick >= env->max_steps) {
        done = true;
    }

    if (done) {
        env->rewards[0] += env->crash_reward;
        env->terminals[0] = 1.0f;
        env->episode_return += env->rewards[0];
        flappy_add_log(env);
        c_reset(env);
        return;
    }

    env->episode_return += env->rewards[0];
    flappy_compute_observations(env);
}

void c_render(Flappy* env) {
    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        InitWindow(env->width, env->height, "PufferLib Flappy");
        SetTargetFPS(60);
        env->client->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(0, env->height - 24, env->width, 24, (Color){40, 120, 84, 255});
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        int gap_top = (int)(pipe->gap_y - env->pipe_gap * 0.5f);
        int gap_bottom = (int)(pipe->gap_y + env->pipe_gap * 0.5f);
        DrawRectangle((int)pipe->x, 0, (int)env->pipe_width, gap_top, (Color){0, 187, 187, 255});
        DrawRectangle((int)pipe->x, gap_bottom, (int)env->pipe_width,
            env->height - gap_bottom, (Color){0, 187, 187, 255});
    }

    float sprite_size = env->bird_radius * 3.0f;
    float rotation = flappy_clampf(env->bird_vy * 3.0f, -35.0f, 35.0f);
    DrawTexturePro(
        env->client->puffer,
        (Rectangle){0, 0, 128, 128},
        (Rectangle){env->bird_x, env->bird_y, sprite_size, sprite_size},
        (Vector2){sprite_size * 0.5f, sprite_size * 0.5f},
        rotation,
        WHITE
    );
    DrawText(TextFormat("Score: %i", env->score), 12, 12, 24, (Color){241, 241, 241, 255});
    EndDrawing();
}

void c_close(Flappy* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) {
            UnloadTexture(env->client->puffer);
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

