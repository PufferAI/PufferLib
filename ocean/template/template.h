#include <stdlib.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define ACT_SIZES {2}
#define OBS_SIZE 1
#define NUM_ATNS 1

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

struct Log {
    float perf;
    float score;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int size;
    int x;
    int goal;
    unsigned int rng;
};
typedef Env Template;

void puf_reset(Template* env) {
    env->goal = (rand_r(&env->rng) % 2 == 0) ? env->size : -env->size;
    env->agents[0].observations[0] = (env->goal > 0) ? 1 : 0;
    env->x = 0;
}

void puf_step(Template* env) {
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    if ((int)env->agents[0].actions[0] == 0) {
        env->x -= 1;
    } else if ((int)env->agents[0].actions[0] == 1) {
        env->x += 1;
    }
    if (env->x == env->goal) {
        puf_reset(env);
        env->agents[0].rewards[0] = 1;
        env->agents[0].terminals[0] = 1;
        env->log.perf += 1;
        env->log.score += 1;
        env->log.n += 1;
    } else if (env->x == -env->goal) {
        puf_reset(env);
        env->agents[0].rewards[0] = -1;
        env->agents[0].terminals[0] = 1;
        env->log.score -= 1;
        env->log.n += 1;
    }
    env->agents[0].observations[0] = (env->goal > 0) ? 1 : 0;
}

void puf_render(Template* env) {
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib Template");
        SetTargetFPS(5);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
            env->agents[0].actions[0] = 0;
        } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
            env->agents[0].actions[0] = 1;
        } else {
            env->agents[0].actions[0] = -1;
        }
    }
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawText("Go to the red square!", 20, 20, 20, PUFF_WHITE);
    DrawRectangle(540 - 32 + 64 * env->goal, 360 - 32, 64, 64, PUFF_RED);
    DrawRectangle(540 - 32 + 64 * env->x, 360 - 32, 64, 64, PUFF_CYAN);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Template* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->size = dict_get(kwargs, "size");
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "n", log->n);
}
