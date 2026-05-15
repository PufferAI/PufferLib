#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ARENA_HALF_SIZE 500.0f
#define MAX_HP 100.0f
#define EPSILON 1e-6f

#define PLAYER_SIZE 30.0f
#define PLAYER_SPEED_PER_TICK 25.0f
#define PLAYER_ATTACK_RADIUS 40.0f
#define PLAYER_ATTACK_TICKS 3
#define PLAYER_ATTACK_DMG 5.0f
#define PLAYER_DODGE_TICKS 4
#define PLAYER_IFRAME_TICKS 2
#define PLAYER_DODGE_COOLDOWN 15
#define PLAYER_DODGE_SPEED_PER_TICK 35.0f

#define BOSS_SIZE 50.0f
#define BOSS_ATTACK_DMG 15.0f
#define BOSS_AOE_ATTACK_RADIUS 80.0f
#define BOSS_IDLE_TICKS 7
#define BOSS_WINDUP_TICKS 5
#define BOSS_ACTIVE_TICKS 5
#define BOSS_RECOVERY_TICKS 5

#define REWARD_APPROACH 0.7f
#define REWARD_HIT_WALL -0.05f
#define REWARD_PLAYER_HIT_BOSS 0.07f
#define REWARD_BOSS_HIT_PLAYER -0.05f
#define REWARD_DODGE_SUCCESS 0.07f
#define REWARD_KILL_BOSS 1.0f
#define REWARD_PLAYER_DIED -1.0f
#define REWARD_TIMEOUT -1.0f
#define REWARD_TICK -0.01f
#define EPISODE_LENGTH 600

#define WINDOW_SIZE 720
#define TARGET_FPS 30
#define HP_BAR_WIDTH 40
#define HP_BAR_HEIGHT 5
#define UI_MARGIN 20
#define UI_RIGHT_X 580
#define UI_BOTTOM_Y 680
#define UI_HP_BAR_Y 700
#define UI_FONT_SIZE 20
#define UI_FONT_SIZE_SMALL 16

static const Color PLAYER_COLOR = (Color){50, 100, 255, 255};
static const Color BOSS_COLOR = (Color){0, 187, 187, 255};
static const Color TEXT_COLOR = (Color){241, 241, 241, 255};
static const Color HITBOX_COLOR = (Color){241, 241, 241, 50};
static const Color BACKGROUND_COLOR = (Color){6, 24, 24, 255};
static const Color HP_COLOR = (Color){0, 255, 0, 255};

static const Color ARENA_BORDER_COLOR = (Color){30, 120, 120, 255};
static const Color ARENA_GRID_COLOR = (Color){30, 70, 70, 255};

static const Color PLAYER_DODGE_COLOR = (Color){255, 215, 90, 255};
static const Color PLAYER_ATTACK_COLOR = (Color){170, 220, 255, 255};
static const Color BOSS_DANGER_COLOR = (Color){255, 80, 80, 255};

typedef enum { PLAYER_IDLING, PLAYER_DODGING, PLAYER_ATTACKING } PlayerState;

typedef enum {
  BOSS_IDLING,
  BOSS_WINDING_UP,
  BOSS_ATTACKING,
  BOSS_RECOVERING,
} BossState;

typedef struct {
  float perf;           // 0-1 normalized metric
  float score;          // Unnormalized metric
  float episode_return; // Sum of rewards
  float episode_length; // Steps per episode
  float wins;           // Episodes where boss died
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
  float dist_to_boss;

  PlayerState player_state;
  float player_hp;
  int player_dodge_cooldown;
  int player_state_ticks;
  int dodge_escape_pending;

  BossState boss_state;
  float boss_hp;
  int boss_phase_ticks;

  float episode_return;

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

  env->observations[obs_idx++] = env->player_x / ARENA_HALF_SIZE;
  env->observations[obs_idx++] = env->player_y / ARENA_HALF_SIZE;

  float dist = distance(env->player_x, env->player_y, env->boss_x, env->boss_y);
  float max_dist = sqrtf(2.0f) * ARENA_HALF_SIZE;
  env->observations[obs_idx++] = dist / max_dist;

  env->observations[obs_idx++] = env->player_hp / MAX_HP;
  env->observations[obs_idx++] = env->boss_hp / MAX_HP;

  env->observations[obs_idx++] =
      (float)env->player_dodge_cooldown / PLAYER_DODGE_COOLDOWN;

  float dodge_remaining =
      (env->player_state == PLAYER_DODGING)
          ? (float)env->player_state_ticks / PLAYER_DODGE_TICKS
          : 0.0f;
  env->observations[obs_idx++] = dodge_remaining;

  int iframe_ticks =
      env->player_state_ticks - (PLAYER_DODGE_TICKS - PLAYER_IFRAME_TICKS);
  float iframe_remaining =
      (env->player_state == PLAYER_DODGING && iframe_ticks > 0)
          ? fminf((float)iframe_ticks / PLAYER_IFRAME_TICKS, 1.0f)
          : 0.0f;
  env->observations[obs_idx++] = iframe_remaining;

  float attack_remaining =
      (env->player_state == PLAYER_ATTACKING)
          ? (float)env->player_state_ticks / PLAYER_ATTACK_TICKS
          : 0.0f;
  env->observations[obs_idx++] = attack_remaining;

  float cycle_len = BOSS_IDLE_TICKS + BOSS_WINDUP_TICKS + BOSS_ACTIVE_TICKS +
                    BOSS_RECOVERY_TICKS;
  float time_until_aoe = 0.0f;
  if (env->boss_state == BOSS_IDLING)
    time_until_aoe = env->boss_phase_ticks + BOSS_WINDUP_TICKS;
  else if (env->boss_state == BOSS_WINDING_UP)
    time_until_aoe = env->boss_phase_ticks;
  else if (env->boss_state == BOSS_RECOVERING)
    time_until_aoe =
        env->boss_phase_ticks + BOSS_IDLE_TICKS + BOSS_WINDUP_TICKS;
  env->observations[obs_idx++] = time_until_aoe / cycle_len;

  float aoe_remaining = (env->boss_state == BOSS_ATTACKING)
                            ? (float)env->boss_phase_ticks / BOSS_ACTIVE_TICKS
                            : 0.0f;
  env->observations[obs_idx++] = aoe_remaining;

  env->observations[obs_idx++] =
      (float)(EPISODE_LENGTH - env->tick) / EPISODE_LENGTH;
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
  env->dodge_escape_pending = 0;
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

  env->dist_to_boss =
      distance(env->player_x, env->player_y, env->boss_x, env->boss_y);

  update_observations(env);
}

void c_step(BossFight *env) {
  float reward = REWARD_TICK;
  env->terminals[0] = 0;

  int action = env->actions[0];
  if (IsKeyDown(KEY_LEFT_SHIFT)) {
    if (IsKeyDown(KEY_W))
      action = 1;
    else if (IsKeyDown(KEY_S))
      action = 2;
    else if (IsKeyDown(KEY_A))
      action = 3;
    else if (IsKeyDown(KEY_D))
      action = 4;
    else if (IsKeyDown(KEY_SPACE))
      action = 5;
    else if (IsKeyDown(KEY_J))
      action = 6;
    else
      action = 0;
  }

  float dx = 0;
  float dy = 0;

  if (action == 1) {
    dy = -PLAYER_SPEED_PER_TICK;
  } else if (action == 2) {
    dy = PLAYER_SPEED_PER_TICK;
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
      env->player_state != PLAYER_DODGING && env->player_dodge_cooldown == 0;
  bool can_attack = env->player_state == PLAYER_IDLING;

  float aoe_dist = BOSS_SIZE + PLAYER_SIZE + BOSS_AOE_ATTACK_RADIUS;
  bool boss_threatening =
      env->boss_state == BOSS_WINDING_UP || env->boss_state == BOSS_ATTACKING;

  float pre_dodge_dist = 0.0f;
  if (wanna_dodge && can_dodge) {
    pre_dodge_dist =
        distance(env->player_x, env->player_y, env->boss_x, env->boss_y);
    env->dodge_escape_pending =
        boss_threatening && pre_dodge_dist <= aoe_dist ? 1 : 0;

    env->player_state_ticks = PLAYER_DODGE_TICKS;
    env->player_state = PLAYER_DODGING;
  }

  // Dodge: multi-tick movement away from boss, with i-frames at start
  if (env->player_state == PLAYER_DODGING) {
    float away_x = env->player_x - env->boss_x;
    float away_y = env->player_y - env->boss_y;
    float away_norm = sqrtf(away_x * away_x + away_y * away_y);
    if (away_norm > EPSILON) {
      env->player_x += (away_x / away_norm) * PLAYER_DODGE_SPEED_PER_TICK;
      env->player_y += (away_y / away_norm) * PLAYER_DODGE_SPEED_PER_TICK;
    }
  }

  bool hit_wall = fabsf(env->player_x) > ARENA_HALF_SIZE ||
                  fabsf(env->player_y) > ARENA_HALF_SIZE;
  if (hit_wall) {
    reward += REWARD_HIT_WALL;
  }

  // Can't walk out of bounds
  env->player_x =
      fmaxf(-ARENA_HALF_SIZE, fminf(ARENA_HALF_SIZE, env->player_x));
  env->player_y =
      fmaxf(-ARENA_HALF_SIZE, fminf(ARENA_HALF_SIZE, env->player_y));

  float dist = distance(env->player_x, env->player_y, env->boss_x, env->boss_y);

  float max_dist = sqrtf(2.0f) * ARENA_HALF_SIZE;
  reward += REWARD_APPROACH * ((env->dist_to_boss - dist) / max_dist);
  env->dist_to_boss = dist;

  // Push player out if clipping into boss
  if (dist < BOSS_SIZE + PLAYER_SIZE && dist > EPSILON) {
    float overlap = BOSS_SIZE + PLAYER_SIZE - dist;
    float dx = env->player_x - env->boss_x;
    float dy = env->player_y - env->boss_y;
    env->player_x += (dx / dist) * overlap;
    env->player_y += (dy / dist) * overlap;
    dist = distance(env->player_x, env->player_y, env->boss_x, env->boss_y);
  }

  bool close_enough = dist <= BOSS_SIZE + PLAYER_ATTACK_RADIUS + PLAYER_SIZE;

  if (wanna_attack && can_attack && close_enough) {
    env->player_state_ticks = PLAYER_ATTACK_TICKS;
    env->player_state = PLAYER_ATTACKING;
    env->boss_hp -= PLAYER_ATTACK_DMG;
    reward += REWARD_PLAYER_HIT_BOSS;
  }

  bool in_aoe_attack = dist <= aoe_dist;
  bool player_iframed =
      env->player_state == PLAYER_DODGING &&
      env->player_state_ticks > (PLAYER_DODGE_TICKS - PLAYER_IFRAME_TICKS);

  // Boss deals damage every tick while player in AOE (unless i-framed)
  bool boss_can_hit = in_aoe_attack && !player_iframed;
  bool boss_can_damage = env->boss_state == BOSS_ATTACKING && boss_can_hit;
  if (boss_can_damage) {
    env->player_hp -= BOSS_ATTACK_DMG;
    reward += REWARD_BOSS_HIT_PLAYER;
  }

  // Reward dodges that actually exit the AOE during the danger window
  if (env->dodge_escape_pending) {
    if (!boss_threatening) {
      env->dodge_escape_pending = 0;
    } else if (dist > aoe_dist) {
      reward += REWARD_DODGE_SUCCESS;
      env->dodge_escape_pending = 0;
    }
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
      env->dodge_escape_pending = 0;
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
               (float)WINDOW_SIZE);
}

float radius_to_screen(float world_radius) {
  return world_radius / (2 * ARENA_HALF_SIZE) * (float)WINDOW_SIZE;
}

void c_render(BossFight *env) {
  if (!IsWindowReady()) {
    InitWindow(WINDOW_SIZE, WINDOW_SIZE, "BossFight");
    SetTargetFPS(TARGET_FPS);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();

  ClearBackground(BACKGROUND_COLOR);
  DrawText("Beat the boss!", UI_MARGIN, UI_MARGIN, UI_FONT_SIZE, TEXT_COLOR);

  // Arena (bounds + subtle grid)
  {
    const float grid_step = 100.0f;
    const float axis_step = 250.0f;
    const Color grid = Fade(ARENA_GRID_COLOR, 0.28f);
    const Color axis = Fade(ARENA_BORDER_COLOR, 0.35f);

    for (float x = -ARENA_HALF_SIZE; x <= ARENA_HALF_SIZE + 0.5f;
         x += grid_step) {
      int sx = world_to_screen(x);
      DrawLine(sx, 0, sx, WINDOW_SIZE, grid);
    }
    for (float y = -ARENA_HALF_SIZE; y <= ARENA_HALF_SIZE + 0.5f;
         y += grid_step) {
      int sy = world_to_screen(y);
      DrawLine(0, sy, WINDOW_SIZE, sy, grid);
    }

    // Crosshair axes
    DrawLine(world_to_screen(0.0f), 0, world_to_screen(0.0f), WINDOW_SIZE,
             axis);
    DrawLine(0, world_to_screen(0.0f), WINDOW_SIZE, world_to_screen(0.0f),
             axis);

    // Quadrant markers
    for (float t = -ARENA_HALF_SIZE; t <= ARENA_HALF_SIZE + 0.5f;
         t += axis_step) {
      int s = world_to_screen(t);
      DrawLineEx((Vector2){(float)s, 4.0f}, (Vector2){(float)s, 14.0f}, 2.0f,
                 Fade(ARENA_BORDER_COLOR, 0.45f));
      DrawLineEx((Vector2){4.0f, (float)s}, (Vector2){14.0f, (float)s}, 2.0f,
                 Fade(ARENA_BORDER_COLOR, 0.45f));
      DrawLineEx((Vector2){(float)s, (float)WINDOW_SIZE - 4.0f},
                 (Vector2){(float)s, (float)WINDOW_SIZE - 14.0f}, 2.0f,
                 Fade(ARENA_BORDER_COLOR, 0.45f));
      DrawLineEx((Vector2){(float)WINDOW_SIZE - 4.0f, (float)s},
                 (Vector2){(float)WINDOW_SIZE - 14.0f, (float)s}, 2.0f,
                 Fade(ARENA_BORDER_COLOR, 0.45f));
    }

    DrawRectangleLinesEx((Rectangle){0, 0, WINDOW_SIZE, WINDOW_SIZE}, 6.0f,
                         Fade(ARENA_BORDER_COLOR, 0.75f));
  }

  // Stats top-right
  char stats[64];
  snprintf(stats, sizeof(stats), "W:%d L:%d T:%d", env->player_wins,
           env->boss_wins, env->timeouts);
  DrawText(stats, UI_RIGHT_X, UI_MARGIN, UI_FONT_SIZE, TEXT_COLOR);

  // Time-left HUD (steps + approx seconds)
  {
    int steps_left = EPISODE_LENGTH - env->tick;
    if (steps_left < 0)
      steps_left = 0;
    float t = (float)steps_left / (float)EPISODE_LENGTH;

    const int bar_w = 260;
    const int bar_h = 10;
    const int bar_x = (WINDOW_SIZE - bar_w) / 2;
    const int bar_y = UI_MARGIN + UI_FONT_SIZE + 8;

    DrawText("TIME", bar_x - 50, bar_y - 4, UI_FONT_SIZE_SMALL,
             Fade(TEXT_COLOR, 0.85f));
    DrawRectangle(bar_x, bar_y, bar_w, bar_h, Fade(DARKGRAY, 0.8f));
    DrawRectangle(bar_x, bar_y, (int)((float)bar_w * t), bar_h,
                  Fade((Color){120, 210, 210, 255}, 0.95f));
    DrawRectangleLinesEx(
        (Rectangle){(float)bar_x, (float)bar_y, (float)bar_w, (float)bar_h},
        2.0f, Fade(ARENA_BORDER_COLOR, 0.7f));

    char tbuf[64];
    int secs_left = (int)ceilf((float)steps_left / (float)TARGET_FPS);
    snprintf(tbuf, sizeof(tbuf), "%d steps  (~%ds)", steps_left, secs_left);
    DrawText(tbuf, bar_x, bar_y + bar_h + 6, UI_FONT_SIZE_SMALL,
             Fade(TEXT_COLOR, 0.85f));
  }

  // Player
  int player_sx = world_to_screen(env->player_x);
  int player_sy = world_to_screen(env->player_y);
  float player_hp_ratio = fmaxf(0.0f, fminf(1.0f, env->player_hp / MAX_HP));
  int player_hp_width = (int)(player_hp_ratio * HP_BAR_WIDTH);

  float player_attack_r = radius_to_screen(PLAYER_SIZE + PLAYER_ATTACK_RADIUS);
  bool player_iframed =
      env->player_state == PLAYER_DODGING &&
      env->player_state_ticks > (PLAYER_DODGE_TICKS - PLAYER_IFRAME_TICKS);

  Color player_base = env->player_hp <= 0 ? RED : PLAYER_COLOR;
  if (env->player_state == PLAYER_DODGING)
    player_base = PLAYER_DODGE_COLOR;
  DrawCircleLines(player_sx, player_sy, player_attack_r,
                  Fade(PLAYER_ATTACK_COLOR, 0.18f));

  // Dodge trail (stateless: inferred from away-from-boss direction)
  if (env->player_state == PLAYER_DODGING) {
    float away_x = env->player_x - env->boss_x;
    float away_y = env->player_y - env->boss_y;
    float away_norm = sqrtf(away_x * away_x + away_y * away_y);
    if (away_norm > EPSILON) {
      float ux = away_x / away_norm;
      float uy = away_y / away_norm;
      for (int i = 1; i <= 4; i++) {
        float w = (float)(5 - i) / 5.0f;
        int tx = world_to_screen(env->player_x - ux * (float)i * 40.0f);
        int ty = world_to_screen(env->player_y - uy * (float)i * 40.0f);
        DrawCircle(tx, ty, radius_to_screen(PLAYER_SIZE) * (0.9f - 0.08f * i),
                   Fade(PLAYER_DODGE_COLOR, 0.08f + 0.12f * w));
      }
    }
  }

  // Player body (shadow + fill + outline)
  DrawCircle(player_sx + 3, player_sy + 4, radius_to_screen(PLAYER_SIZE),
             Fade(BLACK, 0.25f));
  DrawCircle(player_sx, player_sy, radius_to_screen(PLAYER_SIZE), player_base);
  DrawCircleLines(player_sx, player_sy, radius_to_screen(PLAYER_SIZE),
                  Fade(WHITE, 0.25f));

  // Attack effect (duration)
  if (env->player_state == PLAYER_ATTACKING) {
    float rem = (float)env->player_state_ticks / (float)PLAYER_ATTACK_TICKS;
    rem = fmaxf(0.0f, fminf(1.0f, rem));
    float pulse = 1.0f - rem;
    float outer = player_attack_r * (1.0f + 0.10f * pulse);
    float inner = player_attack_r * (0.92f + 0.04f * pulse);
    BeginBlendMode(BLEND_ADDITIVE);
    DrawRing((Vector2){(float)player_sx, (float)player_sy}, inner, outer, 0.0f,
             360.0f, 64, Fade(PLAYER_ATTACK_COLOR, 0.30f + 0.45f * rem));
    EndBlendMode();
    DrawCircleLines(player_sx, player_sy, outer,
                    Fade(PLAYER_ATTACK_COLOR, 0.25f + 0.35f * rem));
  }

  // I-frame blink
  if (player_iframed) {
    BeginBlendMode(BLEND_ADDITIVE);
    DrawCircleLines(player_sx, player_sy, radius_to_screen(PLAYER_SIZE) * 1.12f,
                    Fade(WHITE, 0.65f));
    EndBlendMode();
  }

  // Boss
  int boss_sx = world_to_screen(env->boss_x);
  int boss_sy = world_to_screen(env->boss_y);
  float boss_hp_ratio = fmaxf(0.0f, fminf(1.0f, env->boss_hp / MAX_HP));
  int boss_hp_width = (int)(boss_hp_ratio * HP_BAR_WIDTH);

  float boss_aoe_r =
      radius_to_screen(BOSS_SIZE + PLAYER_SIZE + BOSS_AOE_ATTACK_RADIUS);

  // Boss AoE telegraph/active zone
  {
    float a = 0.10f;
    if (env->boss_state == BOSS_WINDING_UP) {
      float p = 1.0f - (float)env->boss_phase_ticks / (float)BOSS_WINDUP_TICKS;
      p = fmaxf(0.0f, fminf(1.0f, p));
      a = 0.15f + 0.25f * p;
      BeginBlendMode(BLEND_ADDITIVE);
      DrawRing((Vector2){(float)boss_sx, (float)boss_sy}, boss_aoe_r * 0.93f,
               boss_aoe_r, 0.0f, 360.0f * p, 64, Fade(BOSS_DANGER_COLOR, a));
      EndBlendMode();
      DrawCircleLines(boss_sx, boss_sy, boss_aoe_r,
                      Fade(BOSS_DANGER_COLOR, 0.28f + 0.25f * p));
    } else if (env->boss_state == BOSS_ATTACKING) {
      float rem = (float)env->boss_phase_ticks / (float)BOSS_ACTIVE_TICKS;
      rem = fmaxf(0.0f, fminf(1.0f, rem));
      DrawCircle(boss_sx, boss_sy, boss_aoe_r,
                 Fade(BOSS_DANGER_COLOR, 0.22f + 0.08f * (1.0f - rem)));
      DrawCircleLines(boss_sx, boss_sy, boss_aoe_r,
                      Fade(BOSS_DANGER_COLOR, 0.95f));
    } else if (env->boss_state == BOSS_RECOVERING) {
      float rem = (float)env->boss_phase_ticks / (float)BOSS_RECOVERY_TICKS;
      rem = fmaxf(0.0f, fminf(1.0f, rem));
      DrawCircle(boss_sx, boss_sy, boss_aoe_r,
                 Fade(BOSS_DANGER_COLOR, 0.16f * rem));
      DrawCircleLines(boss_sx, boss_sy, boss_aoe_r,
                      Fade(BOSS_DANGER_COLOR, 0.55f * rem));
    } else {
      DrawCircleLines(boss_sx, boss_sy, boss_aoe_r,
                      Fade(BOSS_DANGER_COLOR, 0.12f));
    }
  }

  Color boss_color = env->boss_hp <= 0 ? RED : BOSS_COLOR;
  DrawCircleGradient(boss_sx, boss_sy, radius_to_screen(BOSS_SIZE) * 1.25f,
                     Fade(BOSS_COLOR, 0.10f), Fade(BOSS_COLOR, 0.0f));
  DrawCircle(boss_sx + 4, boss_sy + 5, radius_to_screen(BOSS_SIZE),
             Fade(BLACK, 0.22f));
  DrawCircle(boss_sx, boss_sy, radius_to_screen(BOSS_SIZE), boss_color);
  DrawCircleLines(boss_sx, boss_sy, radius_to_screen(BOSS_SIZE),
                  Fade(WHITE, 0.18f));

  // Boss state label
  {
    const char *phase = "IDLE";
    if (env->boss_state == BOSS_WINDING_UP)
      phase = "WINDUP";
    else if (env->boss_state == BOSS_ATTACKING)
      phase = "ACTIVE";
    else if (env->boss_state == BOSS_RECOVERING)
      phase = "RECOVER";

    char pbuf[32];
    snprintf(pbuf, sizeof(pbuf), "%s", phase);
    int w = MeasureText(pbuf, UI_FONT_SIZE_SMALL);
    DrawText(pbuf, boss_sx - w / 2,
             boss_sy - (int)radius_to_screen(BOSS_SIZE) - 22,
             UI_FONT_SIZE_SMALL, Fade(TEXT_COLOR, 0.85f));
  }

  // Player HP bar - bottom left
  const int hud_label_y = UI_HP_BAR_Y - 40;
  DrawText("Player", UI_MARGIN, hud_label_y, UI_FONT_SIZE_SMALL, TEXT_COLOR);
  DrawRectangle(UI_MARGIN, UI_HP_BAR_Y, HP_BAR_WIDTH * 3, HP_BAR_HEIGHT,
                DARKGRAY);
  DrawRectangle(UI_MARGIN, UI_HP_BAR_Y, player_hp_width * 3, HP_BAR_HEIGHT,
                HP_COLOR);

  // Dodge cooldown (under player hp)
  {
    float cd =
        1.0f - fmaxf(0.0f, fminf(1.0f, (float)env->player_dodge_cooldown /
                                           (float)PLAYER_DODGE_COOLDOWN));
    const int dodge_label_y = UI_HP_BAR_Y - 22;
    const int dodge_bar_y = UI_HP_BAR_Y - 18;
    DrawText("Dodge", UI_MARGIN, dodge_label_y, UI_FONT_SIZE_SMALL,
             Fade(TEXT_COLOR, 0.75f));
    DrawRectangle(UI_MARGIN + 58, dodge_bar_y, 90, 6, Fade(DARKGRAY, 0.8f));
    DrawRectangle(UI_MARGIN + 58, dodge_bar_y, (int)(90.0f * cd), 6,
                  Fade(PLAYER_DODGE_COLOR, 0.85f));
  }

  // Boss HP bar - bottom right
  DrawText("Boss", UI_RIGHT_X, hud_label_y, UI_FONT_SIZE_SMALL, TEXT_COLOR);
  DrawRectangle(UI_RIGHT_X, UI_HP_BAR_Y, HP_BAR_WIDTH * 3, HP_BAR_HEIGHT,
                DARKGRAY);
  DrawRectangle(UI_RIGHT_X, UI_HP_BAR_Y, boss_hp_width * 3, HP_BAR_HEIGHT,
                HP_COLOR);

  EndDrawing();
}

void c_close(BossFight *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
