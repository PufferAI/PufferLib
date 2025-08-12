#include "raylib.h"
#include <stdlib.h>
#include <string.h>

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char RIGHT = 2;
const unsigned char FLIP = 3;

const unsigned char OFF = 0;
const unsigned char ON = 1;
const unsigned char EMPTY = 0;
const unsigned char CURSOR = 3;

typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  // Any extra fields you add here may be exported to Python in binding.c
  float n; // Required as the last field
} Log;

typedef struct {
  Log log;
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int size;
  int pos;
  int n_correct;
  int tick;
} BitFlip;

void add_log(BitFlip *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

void c_reset(BitFlip *env) {
  memset(env->observations, OFF, env->size * 3 * sizeof(char));
  env->n_correct = 0;
  env->observations[0] = ON;
  for (int i = 1; i < env->size; i++) {
    env->observations[i] = (rand() % 2 == 1) ? ON : OFF;

    // Track how many are correct to begin with
    if (env->observations[i] == OFF) {
      env->n_correct++;
    }
  }
  env->pos = 2 * env->size + (env->size - 1) / 2;
  env->observations[env->pos] = CURSOR;
  env->tick = 0;
}

void c_step(BitFlip *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0.0;

  env->observations[env->pos] = EMPTY;

  if (action == LEFT) {
    env->pos -= 1;
  } else if (action == RIGHT) {
    env->pos += 1;
  }

  if (env->tick == 12 * env->size || env->pos < 2 * env->size ||
      env->pos >= env->size * 3) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0;
    add_log(env);
    c_reset(env);
    return;
  }

  env->observations[env->pos] = CURSOR;

  int state_idx = env->pos - env->size;
  int target_idx = env->pos - 2 * env->size;

  if (action == FLIP) {
    env->observations[state_idx] ^= 1;

    if (env->observations[state_idx] == env->observations[target_idx]) {
      env->n_correct += 1;
    } else {
      env->n_correct -= 1;
    }
  }

  if (env->n_correct == env->size) {
    env->rewards[0] = 1.0;
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
    return;
  }
}

void c_render(BitFlip *env) {
  int px = 64;

  if (!IsWindowReady()) {
    InitWindow(px * env->size, px * 3, "PufferLib BitFlip");
    SetTargetFPS(5);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(PUFF_BACKGROUND);

  for (int i = 0; i < env->size * 2; i++) {
    int tex = env->observations[i];
    if (tex == OFF) {
      continue;
    }
    DrawRectangle((i % env->size) * px, (i / env->size) * px, px, px,
                  PUFF_CYAN);
  }
  for (int i = env->size * 2; i < env->size * 3; i++) {
    int tex = env->observations[i];
    if (tex == EMPTY) {
      continue;
    }
    DrawRectangle((i % env->size) * px, (i / env->size) * px, px, px, PUFF_RED);
  }

  EndDrawing();
}

void c_close(BitFlip *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
