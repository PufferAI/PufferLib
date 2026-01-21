#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ARENA_HALF_SIZE 5.0f
#define MAX_HP 1.0f
#define PLAYER_SPEED_PER_TICK 0.1f
#define PLAYER_SIZE 0.3f
#define BOSS_SIZE 0.5f
#define PLAYER_ATTACK_RADIUS 0.4f
#define PLAYER_ATTACK_TICKS 3
#define PLAYER_DODGE_TICKS 4
#define PLAYER_DODGE_COOLDOWN 15
#define PLAYER_ATTACK_DMG 0.1f
#define BOSS_ATTACK_DMG 0.05f
#define BOSS_AOE_ATTACK_RADIUS 0.7f
#define BOSS_IDLE_TICKS 7
#define BOSS_WINDUP_TICKS 5
#define BOSS_ACTIVE_TICKS 5
#define BOSS_RECOVERY_TICKS 5
#define HP_BAR_WIDTH 40
#define HP_BAR_HEIGHT 5

// Rewards
#define REWARD_APPROACH 0.5f
#define REWARD_HIT_WALL -0.1f
#define REWARD_PLAYER_HIT_BOSS 5.0f
#define REWARD_BOSS_HIT_PLAYER -0.5f
#define REWARD_DODGE_SUCCESS 2.0f
#define REWARD_KILL_BOSS 50.0f
#define REWARD_PLAYER_DIED -5.0f
#define REWARD_TIMEOUT -20.0f
#define REWARD_TICK -0.001f
#define EPISODE_LENGTH 300

const Color PLAYER_COLOR = (Color){50, 100, 255, 255};
const Color BOSS_COLOR = (Color){0, 187, 187, 255};
const Color TEXT_COLOR = (Color){241, 241, 241, 255};
const Color HITBOX_COLOR = (Color){241, 241, 241, 50};
const Color BACKGROUND_COLOR = (Color){6, 24, 24, 255};
const Color HP_COLOR = (Color){0, 255, 0, 255};

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
  float wins;           // episodes where boss died
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
  float prev_distance;

  PlayerState player_state;
  float player_hp;
  int player_dodge_cooldown;
  int player_state_ticks;

  BossState boss_state;
  float boss_hp;
  int boss_phase_ticks;

  float episode_return; // track within episode

  // stats
  int player_wins;
  int boss_wins;
  int timeouts;
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
  env->log.wins += (env->boss_hp <= 0) ? 1.0f : 0.0f;
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

  while (distance(env->player_x, env->player_y, env->boss_x, env->boss_y) <=
         PLAYER_SIZE + PLAYER_ATTACK_RADIUS + BOSS_SIZE +
             BOSS_AOE_ATTACK_RADIUS) {
    env->player_x = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
    env->player_y = rand_uniform(-ARENA_HALF_SIZE, ARENA_HALF_SIZE);
  }

  env->prev_distance =
      distance(env->player_x, env->player_y, env->boss_x, env->boss_y);

  update_observations(env);
}

void c_step(BossFight *env) {
  float reward = REWARD_TICK;
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

  if (env->player_state == PLAYER_IDLING) {
    env->player_x += dx;
    env->player_y += dy;
  }

  bool wanna_idle = action == 0;
  bool wanna_dodge = action == 5;
  bool wanna_attack = action == 6;
  bool can_dodge =
      env->player_state == PLAYER_IDLING && env->player_dodge_cooldown == 0;
  bool can_attack = env->player_state == PLAYER_IDLING;

  if (wanna_attack && can_attack) {
    env->player_state_ticks = PLAYER_ATTACK_TICKS;
    env->player_state = PLAYER_ATTACKING;
  }
  if (wanna_dodge && can_dodge) {
    env->player_state_ticks = PLAYER_DODGE_TICKS;
    env->player_state = PLAYER_DODGING;
  }

  float dist = distance(env->player_x, env->player_y, env->boss_x, env->boss_y);

  reward += REWARD_APPROACH * (env->prev_distance - dist);
  env->prev_distance = dist;

  bool close_enough = dist <= BOSS_SIZE + PLAYER_ATTACK_RADIUS + PLAYER_SIZE;

  bool hit_wall = fabsf(env->player_x) > ARENA_HALF_SIZE ||
                  fabsf(env->player_y) > ARENA_HALF_SIZE;
  if (hit_wall) {
    reward += REWARD_HIT_WALL;
  }
  // can't walk out of bounds
  env->player_x =
      fmaxf(-ARENA_HALF_SIZE, fminf(ARENA_HALF_SIZE, env->player_x));
  env->player_y =
      fmaxf(-ARENA_HALF_SIZE, fminf(ARENA_HALF_SIZE, env->player_y));

  // push player out if clipping into boss
  if (dist < BOSS_SIZE + PLAYER_SIZE) {
    float overlap = BOSS_SIZE + PLAYER_SIZE - dist;
    float dx = env->player_x - env->boss_x;
    float dy = env->player_y - env->boss_y;
    env->player_x += (dx / dist) * overlap;
    env->player_y += (dy / dist) * overlap;
    // recalculate distance after push
    dist = distance(env->player_x, env->player_y, env->boss_x, env->boss_y);
  }

  if (wanna_attack && can_attack && close_enough) {
    env->boss_hp -= PLAYER_ATTACK_DMG;
    reward += REWARD_PLAYER_HIT_BOSS;
  }

  bool in_aoe_attack = dist <= BOSS_SIZE + PLAYER_SIZE + BOSS_AOE_ATTACK_RADIUS;
  bool boss_can_hit = env->player_state != PLAYER_DODGING && in_aoe_attack;
  bool boss_can_damage = env->boss_state == BOSS_ATTACKING && boss_can_hit;
  if (boss_can_damage) {
    env->player_hp -= BOSS_ATTACK_DMG;
    reward += REWARD_BOSS_HIT_PLAYER;
  }

  bool would_be_hit = env->boss_state == BOSS_ATTACKING && in_aoe_attack;

  bool successfully_dodging =
      would_be_hit && env->player_state == PLAYER_DODGING;

  if (successfully_dodging) {
    reward += REWARD_DODGE_SUCCESS;
  }

  bool killed_boss = env->boss_hp <= 0;
  bool player_died = env->player_hp <= 0;
  bool timed_out = env->tick >= EPISODE_LENGTH;

  if (killed_boss) {
    reward += REWARD_KILL_BOSS;
    env->terminals[0] = 1;
    env->player_wins++;
  } else if (player_died) {
    reward += REWARD_PLAYER_DIED;
    env->terminals[0] = 1;
    env->boss_wins++;
  } else if (timed_out) {
    reward += REWARD_TIMEOUT;
    env->terminals[0] = 1;
    env->timeouts++;
  }

  env->rewards[0] = reward;
  env->episode_return += reward;

  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }

  env->tick++;
  if (env->boss_phase_ticks > 0) {
    env->boss_phase_ticks--;
  }
  if (env->player_state_ticks > 0) {
    env->player_state_ticks--;
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
  if (env->player_state_ticks == 0) {
    if (env->player_state == PLAYER_DODGING) {
      env->player_dodge_cooldown = PLAYER_DODGE_COOLDOWN;
      env->player_state = PLAYER_IDLING;
    } else if (env->player_state == PLAYER_ATTACKING) {
      env->player_state = PLAYER_IDLING;
    }
  }
  if (env->player_dodge_cooldown > 0) {
    env->player_dodge_cooldown--;
  }

  update_observations(env);
}

int world_to_screen(float world_coord) {
  return (int)((world_coord + ARENA_HALF_SIZE) / (2 * ARENA_HALF_SIZE) *
               720.0f);
}

float radius_to_screen(float world_radius) {
  return world_radius / (2 * ARENA_HALF_SIZE) * 720.0f;
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

  // Stats top-right
  char stats[64];
  snprintf(stats, sizeof(stats), "W:%d L:%d T:%d", env->player_wins,
           env->boss_wins, env->timeouts);
  DrawText(stats, 580, 20, 20, TEXT_COLOR);

  // Player
  int player_sx = world_to_screen(env->player_x);
  int player_sy = world_to_screen(env->player_y);
  int player_hp_bar_y = player_sy + (int)radius_to_screen(PLAYER_SIZE) + 5;
  int player_hp_width = (int)((float)env->player_hp / MAX_HP * HP_BAR_WIDTH);

  Color player_color = env->player_hp <= 0 ? RED : PLAYER_COLOR;
  DrawCircle(player_sx, player_sy,
             radius_to_screen(PLAYER_SIZE + PLAYER_ATTACK_RADIUS),
             HITBOX_COLOR);
  DrawCircle(player_sx, player_sy, radius_to_screen(PLAYER_SIZE), player_color);

  // Boss
  int boss_sx = world_to_screen(env->boss_x);
  int boss_sy = world_to_screen(env->boss_y);
  int boss_hp_bar_y = boss_sy + (int)radius_to_screen(BOSS_SIZE) + 5;
  int boss_hp_width = (int)((float)env->boss_hp / MAX_HP * HP_BAR_WIDTH);

  Color boss_color = env->boss_hp <= 0 ? RED : BOSS_COLOR;
  DrawCircle(boss_sx, boss_sy,
             radius_to_screen(BOSS_SIZE + BOSS_AOE_ATTACK_RADIUS),
             HITBOX_COLOR);
  DrawCircle(boss_sx, boss_sy, radius_to_screen(BOSS_SIZE), boss_color);

  // Player HP bar - bottom left
  DrawText("Player", 20, 680, 16, TEXT_COLOR);
  DrawRectangle(20, 700, HP_BAR_WIDTH * 3, HP_BAR_HEIGHT, DARKGRAY);
  DrawRectangle(20, 700, player_hp_width * 3, HP_BAR_HEIGHT, HP_COLOR);

  // Boss HP bar - bottom right
  DrawText("Boss", 580, 680, 16, TEXT_COLOR);
  DrawRectangle(580, 700, HP_BAR_WIDTH * 3, HP_BAR_HEIGHT, DARKGRAY);
  DrawRectangle(580, 700, boss_hp_width * 3, HP_BAR_HEIGHT, HP_COLOR);

  EndDrawing();
}

void c_close(BossFight *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
