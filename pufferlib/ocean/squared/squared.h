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

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};
 
typedef struct Client Client;
typedef struct Squared Squared;
struct Squared {
    Log log;
    Client* client;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size;
    int tick;
    int r;
    int c;
};

void allocate(Squared* env) {
    env->observations = (unsigned char*)calloc(env->size*env->size, sizeof(unsigned char));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(Squared* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void add_log(Squared* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

void c_reset(Squared* env) {
    memset(env->observations, 0, env->size*env->size*sizeof(unsigned char));
    env->observations[env->size*env->size/2] = AGENT;
    env->r = env->size/2;
    env->c = env->size/2;
    env->tick = 0;
    int target_idx;
    do {
        target_idx = rand() % (env->size*env->size);
    } while (target_idx == env->size*env->size/2);
    env->observations[target_idx] = TARGET;
}

void c_step(Squared* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->observations[env->r*env->size + env->c] = EMPTY;

    if (action == DOWN) {
        env->r += 1;
    } else if (action == RIGHT) {
        env->c += 1;
    } else if (action == UP) {
        env->r -= 1;
    } else if (action == LEFT) {
        env->c -= 1;
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

typedef struct Client Client;
struct Client {
    Texture2D ball;
};

Client* make_client(Squared* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    int px = 64*env->size;
    InitWindow(px, px, "PufferLib Squared");
    SetTargetFPS(5);

    //client->agent = LoadTexture("resources/puffers_128.png");
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Squared* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

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
            Color color = (tex == AGENT) ? (Color){0, 255, 255, 255} : (Color){255, 0, 0, 255};
            DrawRectangle(j*px, i*px, px, px, color);
        }
    }
    EndDrawing();
}
