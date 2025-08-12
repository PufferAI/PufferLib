#include <raylib.h>
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char UP = 1;
const unsigned char DOWN = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float n;
} Log;

typedef struct {
  Log log;
  int size;
  int tick;
  unsigned char *observations; // length size*size
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int agent_x, agent_y;
  int target_x, target_y;
  unsigned int rng;
} Squared;

void add_log(Squared *env);
void c_reset(Squared *env);
void c_step(Squared *env);
void c_render(Squared *env);
void c_close(Squared *env);

void add_log(Squared *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

void c_reset(Squared *env) {
  int N = env->size;

  // Zero ticks
  env->tick = 0;

  // Clear observations
  memset(env->observations, 0, N * N * sizeof(char));

  // Put agent in the middle
  env->agent_x = N / 2;
  env->agent_y = N / 2;

  // Put target in a random position
  do {
    env->target_x = rand() % N;
    env->target_y = rand() % N;
  } while (env->target_x == env->agent_x && env->target_y == env->agent_y);

  // Place agent and target on observations
  env->observations[env->agent_y * N + env->agent_x] = AGENT;
  env->observations[env->target_y * N + env->target_x] = TARGET;
}

void c_step(Squared *env) {
  int N = env->size;
  int action = env->actions[0];

  env->tick++;

  // Clear agent from observations
  env->observations[env->agent_y * N + env->agent_x] = EMPTY;

  // Update agent
  if (action == UP)
    env->agent_y--;
  if (action == DOWN)
    env->agent_y++;
  if (action == LEFT)
    env->agent_x--;
  if (action == RIGHT)
    env->agent_x++;

  // Check if not in bounds
  if (env->tick > 3 * env->size || env->agent_y < 0 || env->agent_y >= N ||
      env->agent_x < 0 || env->agent_x >= N) {
    env->rewards[0] = -1.0;
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
    return;
  }

  // Check if agent == target
  if (env->agent_x == env->target_x && env->agent_y == env->target_y) {
    env->rewards[0] = 1.0;
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
    return;
  }

  env->rewards[0] = 0;
  env->terminals[0] = 0;

  env->observations[env->agent_y * N + env->agent_x] = AGENT;
}

void c_render(Squared *env) {
  if (!IsWindowReady()) {
    InitWindow(64 * env->size, 64 * env->size, "PufferLib Squared");
    SetTargetFPS(5);
  }

  // Standard across our envs so exiting is always the same
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});

  int px = 64;
  for (int i = 0; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      int tex = env->observations[i * env->size + j];
      if (tex == EMPTY) {
        continue;
      }
      Color color =
          (tex == AGENT) ? (Color){0, 187, 187, 255} : (Color){187, 0, 0, 255};
      DrawRectangle(j * px, i * px, px, px, color);
    }
  }

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Squared *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
