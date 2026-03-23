#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field 
} Log;

typedef struct {
    Log log;                     // Required field
    unsigned char* observations; // Required field. Ensure type matches in .py and .c
    int* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    int size;
    int x;
    int goal;
} Template;

void c_reset(Template* env) {
    env->x = 0;
    env->goal = (rand()%2 == 0) ? env->size : -env->size;
}

void c_step(Template* env) {
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    if (env->actions[0] == 0) {
        env->x -= 1;
    } else if (env->actions[0] == 1) {
        env->x += 1;
    }
    if (env->x == env->goal) {
        c_reset(env);
        env->rewards[0] = 1;
        env->terminals[0] = 1;
        env->log.score += 1;
        env->log.n += 1;
    } else if (env->x == -env->goal) {
        c_reset(env);
        env->rewards[0] = -1;
        env->terminals[0] = 1;
        env->log.score -= 1;
        env->log.n += 1;
    }
    env->observations[0] = (env->goal > 0) ? 1 : -1;
}

void c_render(Template* env) {
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib Template");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    DrawText("Go to the red square!", 20, 20, 20, PUFF_WHITE);
    DrawRectangle(540 - 32 + 64*env->goal, 360 - 32, 64, 64, PUFF_RED);
    DrawRectangle(540 - 32 + 64*env->x, 360 - 32, 64, 64, PUFF_CYAN);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    EndDrawing();
}

void c_close(Template* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
