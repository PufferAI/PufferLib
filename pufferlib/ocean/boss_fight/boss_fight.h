#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_HALF_SIZE 5.0f
#define MAX_HP 100
#define PLAYER_SPEED_PER_TICK 0.1f
#define PLAYER_SIZE 0.3f
#define BOSS_SIZE 0.5f

const Color PLAYER_COLOR = (Color){187, 0, 0, 255};
const Color BOSS_COLOR = (Color){0, 187, 187, 255};
const Color TEXT_COLOR = (Color){241, 241, 241, 241};
const Color HITBOX_COLOR = (Color){241, 241, 241, 241};
const Color BACKGROUND_COLOR = (Color){6, 24, 24, 255};

typedef enum { PLAYER_IDLING, PLAYER_DODGING, PLAYER_ATTACKING } PlayerState;

typedef enum {
  BOSS_IDLING,
  BOSS_WINDING_UP,
  BOSS_ATTACKING,
  BOSS_RECOVERING,
} BossState;

// Only use floats!
typedef struct {
  float score;
  float n; // Required as the last field
} Log;

typedef struct {
  Log log;                  // Required field
  float *observations;      // Required field. Ensure type matches in .py and .c
  float *actions;           // Required field. Ensure type matches in .py and .c
  float *rewards;           // Required field
  unsigned char *terminals; // Required field

  int tick;
  float player_x;
  float player_y;
  float boss_x;
  float boss_y;
  float distance;

  PlayerState player_state;
  int player_hp;
  int player_dodge_cooldown;
  int player_state_ticks;

  BossState boss_state;
  int boss_hp;
  int boss_phase_ticks;

} BossFight;

float rand_uniform(float low, float high) {
  return low + (high - low) * ((float)rand() / ((float)RAND_MAX + 1.0f));
}

float distance(float x1, float y1, float x2, float y2) {
  float dx = x1 - x2;
  float dy = y1 - y2;
  return sqrtf(dx * dx + dy * dy);
}

void c_reset(BossFight *env) {
  env->tick = 0;
  env->player_x = 0;
  env->player_y = 0;
  env->boss_x = 0;
  env->boss_y = 0;
  env->player_hp = 100;
  env->boss_hp = 100;
  env->player_state = PLAYER_IDLING;
  env->player_dodge_cooldown = 0;
  env->player_state_ticks = 0;
  env->boss_state = BOSS_IDLING;
  env->boss_phase_ticks = 0;
  env->distance = 0;

  env->player_x = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
  env->player_y = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);

  while (distance(env->player_x, env->player_y, env->boss_x, env->boss_y) <
         0.1) {
    env->player_x = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
    env->player_y = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
  }

  env->distance =
      distance(env->player_x, env->player_y, env->boss_x, env->boss_y);

  int obs_idx = 0;

  env->observations[obs_idx++] = 0; // dx
  env->observations[obs_idx++] = 0; // dy
  env->observations[obs_idx++] = env->player_x;
  env->observations[obs_idx++] = env->player_y;
  env->observations[obs_idx++] = env->boss_x;
  env->observations[obs_idx++] = env->boss_y;
  env->observations[obs_idx++] = 100;
  env->observations[obs_idx++] = 100;
  env->observations[obs_idx++] = PLAYER_IDLING;
  env->observations[obs_idx++] = 0; // player_dodge_cooldown
  env->observations[obs_idx++] = 0; // player_state_ticks
  env->observations[obs_idx++] = BOSS_IDLING;
  env->observations[obs_idx++] = 0; // boss_phase_ticks
}

void c_step(BossFight *env) {
  env->rewards[0] = 0;
  env->terminals[0] = 0;
}

void c_render(BossFight *env) {
  if (!IsWindowReady()) {
    InitWindow(1080, 720, "BossFight");
    SetTargetFPS(30);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(BACKGROUND_COLOR);
  DrawText("Beat the boss!", 20, 20, 20, TEXT_COLOR);
  EndDrawing();
}

void c_close(BossFight *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
