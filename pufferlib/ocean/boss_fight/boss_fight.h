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
#define PLAYER_ATTACK_RADIUS 0.5f
#define PLAYER_ATTACK_TICKS 3
#define PLAYER_DODGE_TICKS 6
#define PLAYER_DODGE_COOLDOWN 15
#define PLAYER_ATTACK_DMG 3
#define BOSS_ATTACK_DMG 3
#define BOSS_AOE_ATTACK_RADIUS 0.7f
#define BOSS_IDLE_TICKS 12
#define BOSS_WINDUP_TICKS 18
#define BOSS_ACTIVE_TICKS 3
#define BOSS_RECOVERY_TICKS 12

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
  float perf;           // 0-1 normalized metric
  float score;          // unnormalized metric
  float episode_return; // sum of rewards
  float episode_length; // steps per episode
  float n;              // Required as last field
} Log;

typedef struct {
  Log log;                  // Required field
  float *observations;      // Required field. Ensure type matches in .py and .c
  int *actions;             // Required field. Ensure type matches in .py and .c
  float *rewards;           // Required field
  unsigned char *terminals; // Required field

  int tick;
  float player_x;
  float player_y;
  float boss_x;
  float boss_y;
  // float distance;

  PlayerState player_state;
  int player_hp;
  int player_dodge_cooldown;
  int player_state_ticks;

  BossState boss_state;
  int boss_hp;
  int boss_phase_ticks;

  float episode_return; // track within episode

} BossFight;

float rand_uniform(float low, float high) {
  return low + (high - low) * ((float)rand() / ((float)RAND_MAX + 1.0f));
}

float distance(float x1, float y1, float x2, float y2) {
  float dx = x1 - x2;
  float dy = y1 - y2;
  return sqrtf(dx * dx + dy * dy);
}

void add_log(BossFight *env) {
  env->log.episode_return += env->episode_return;
  env->log.episode_length += env->tick;
  env->log.score += env->episode_return;
  env->log.n++;
}

void update_observations(BossFight *env) {
  int obs_idx = 0;
  env->observations[obs_idx++] = env->boss_x - env->player_x;
  env->observations[obs_idx++] = env->boss_y - env->player_y;
  env->observations[obs_idx++] = env->player_x;
  env->observations[obs_idx++] = env->player_y;
  env->observations[obs_idx++] = env->boss_x;
  env->observations[obs_idx++] = env->boss_y;
  env->observations[obs_idx++] = (float)env->player_hp;
  env->observations[obs_idx++] = (float)env->boss_hp;
  env->observations[obs_idx++] = (float)env->player_state;
  env->observations[obs_idx++] = (float)env->player_dodge_cooldown;
  env->observations[obs_idx++] = (float)env->player_state_ticks;
  env->observations[obs_idx++] = (float)env->boss_state;
  env->observations[obs_idx++] = (float)env->boss_phase_ticks;
}

void c_reset(BossFight *env) {
  env->tick = 0;
  env->player_x = 0;
  env->player_y = 0;
  env->boss_x = 0;
  env->boss_y = 0;
  env->player_hp = MAX_HP;
  env->boss_hp = MAX_HP;
  env->player_state = PLAYER_IDLING;
  env->player_dodge_cooldown = 0;
  env->player_state_ticks = 0;
  env->boss_state = BOSS_IDLING;
  env->boss_phase_ticks = BOSS_IDLE_TICKS;
  env->episode_return = 0;

  env->player_x = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
  env->player_y = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);

  while (distance(env->player_x, env->player_y, env->boss_x, env->boss_y) <
         0.1) {
    env->player_x = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
    env->player_y = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
  }

  // env->distance =
  //     distance(env->player_x, env->player_y, env->boss_x, env->boss_y);

  update_observations(env);
}

void c_step(BossFight *env) {
  float reward = -0.01;
  env->terminals[0] = 0;

  int action = env->actions[0];
  float dx = 0;
  float dy = 0;

  if (action == 1) {
    dy = PLAYER_SPEED_PER_TICK;
  } else if (action == 2) {
    dy = -PLAYER_SPEED_PER_TICK;
  } else if (action == 3) {
    dx = -PLAYER_SPEED_PER_TICK;
  } else if (action == 4) {
    dx = PLAYER_SPEED_PER_TICK;
  }

  env->player_x += dx;
  env->player_y += dy;

  bool wanna_idle = action == 0;
  bool wanna_dodge = action == 5;
  bool wanna_attack = action == 6;
  bool can_dodge =
      env->player_state == PLAYER_IDLING && env->player_dodge_cooldown == 0;
  bool can_attack = env->player_state == PLAYER_IDLING;
  bool close_enough = distance(env->player_x, env->player_y, env->boss_x,
                               env->boss_y) < PLAYER_ATTACK_RADIUS;

  bool hit_wall = fabsf(env->player_x) > ARENA_HALF_SIZE ||
                  fabsf(env->player_y) > ARENA_HALF_SIZE;
  if (hit_wall) {
    reward -= 0.5;
  }
  // can't walk out of bounds
  env->player_x =
      fmaxf(-ARENA_HALF_SIZE, fminf(ARENA_HALF_SIZE, env->player_x));
  env->player_y =
      fmaxf(-ARENA_HALF_SIZE, fminf(ARENA_HALF_SIZE, env->player_y));

  if (wanna_attack && can_attack && close_enough) {
    env->boss_hp -= PLAYER_ATTACK_DMG;
    reward += 0.5;
  }

  bool in_aoe_attack = distance(env->player_x, env->player_y, env->boss_x,
                                env->boss_y) <= BOSS_AOE_ATTACK_RADIUS;
  bool boss_can_hit = env->player_state != PLAYER_DODGING && in_aoe_attack;
  bool boss_can_damage = env->boss_state == BOSS_ATTACKING && boss_can_hit;
  if (boss_can_damage) {
    env->player_hp -= BOSS_ATTACK_DMG;
  }

  bool killed_boss = env->boss_hp <= 0;
  if (killed_boss) {
    reward += 2;
    env->terminals[0] = 1;
  }

  env->rewards[0] = reward;
  env->episode_return += reward;

  bool player_died = env->player_hp <= 0;
  if (player_died) {
    env->terminals[0] = 1;
  }

  if (env->tick >= 300) {
    env->terminals[0] = 1;
  }

  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }

  if (wanna_attack && can_attack) {
    env->player_state_ticks = PLAYER_ATTACK_TICKS;
    env->player_state = PLAYER_ATTACKING;
  }
  if (wanna_dodge && can_dodge) {
    env->player_state_ticks = PLAYER_DODGE_TICKS;
    env->player_state = PLAYER_DODGING;
  }
  if (env->player_state == PLAYER_DODGING && env->player_state_ticks == 0) {
    env->player_dodge_cooldown = PLAYER_DODGE_COOLDOWN;
    env->player_state = PLAYER_IDLING;
  }
  if (env->player_state == PLAYER_ATTACKING && env->player_state_ticks == 0) {
    env->player_state = PLAYER_IDLING;
  }

  if (env->boss_phase_ticks == 0) {
    if (env->boss_state == BOSS_IDLING) {
      env->boss_state = BOSS_WINDING_UP;
      env->boss_phase_ticks = BOSS_WINDUP_TICKS;
    } else if (env->boss_state == BOSS_WINDING_UP) {
      env->boss_state = BOSS_ATTACKING;
      env->boss_phase_ticks = BOSS_ACTIVE_TICKS;
    } else if (env->boss_state == BOSS_ATTACKING) {
      env->boss_state = BOSS_RECOVERING;
      env->boss_phase_ticks = BOSS_RECOVERY_TICKS;
    } else if (env->boss_state == BOSS_RECOVERING) {
      env->boss_state = BOSS_IDLING;
      env->boss_phase_ticks = BOSS_IDLE_TICKS;
    }
  }

  env->tick++;
  if (env->boss_phase_ticks > 0) {
    env->boss_phase_ticks--;
  }
  if (env->player_state_ticks > 0) {
    env->player_state_ticks--;
  }
  if (env->player_dodge_cooldown > 0) {
    env->player_dodge_cooldown--;
  }

  update_observations(env);
}

void c_render(BossFight *env) {
  if (!IsWindowReady()) {
    InitWindow(720, 720, "BossFight");
    SetTargetFPS(30);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();

  ClearBackground(BACKGROUND_COLOR);
  DrawText("Beat the boss!", 20, 20, 20, TEXT_COLOR);

  // DrawCircle(int centerX, int centerY, float radius, Color color)

  EndDrawing();
}

void c_close(BossFight *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
