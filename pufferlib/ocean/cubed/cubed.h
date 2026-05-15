#include "raylib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char UP = 1;
const unsigned char DOWN = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;
const unsigned char FRONT = 5;
const unsigned char BACK = 6;

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
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int size;
  int tick;
  int x;
  int y;
  int z;
} Cubed;

void add_log(Cubed *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

void c_reset(Cubed *env) {
  int cubes = env->size * env->size * env->size;
  memset(env->observations, 0, cubes * sizeof(unsigned char));
  env->x = env->size / 2;
  env->y = env->size / 2;
  env->z = env->size / 2;
  int agent_idx = env->x * env->size * env->size + env->y * env->size + env->z;
  env->observations[agent_idx] = AGENT;
  env->tick = 0;
  int target_idx;
  do {
    target_idx = rand() % cubes;
  } while (target_idx == agent_idx);
  env->observations[target_idx] = TARGET;
}

void c_step(Cubed *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  env->observations[env->x * env->size * env->size + env->y * env->size +
                    env->z] = EMPTY;

  if (action == UP)
    env->y += 1;
  if (action == DOWN)
    env->y -= 1;
  if (action == LEFT)
    env->x -= 1;
  if (action == RIGHT)
    env->x += 1;
  if (action == FRONT)
    env->z += 1;
  if (action == BACK)
    env->z -= 1;

  if (env->tick > 5 * env->size || env->x < 0 || env->y < 0 || env->z < 0 ||
      env->x >= env->size || env->y >= env->size || env->z >= env->size) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0;
    add_log(env);
    c_reset(env);
    return;
  }

  int pos = env->x * env->size * env->size + env->y * env->size + env->z;
  if (env->observations[pos] == TARGET) {
    env->terminals[0] = 1;
    env->rewards[0] = 1.0;
    add_log(env);
    c_reset(env);
    return;
  }

  env->observations[pos] = AGENT;
}

void c_render(Cubed *env) {
  static Camera3D camera = {0};

  if (!IsWindowReady()) {
    InitWindow(1024, 768, "PufferLib Cubed");
    SetTargetFPS(5);

    float center = env->size / 2.0f;
    camera.position =
        (Vector3){env->size * 2.5f, env->size * 2.0f, env->size * 2.5f};
    camera.target = (Vector3){center, center, center};
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    camera.fovy = 60.0f;
    camera.projection = CAMERA_PERSPECTIVE;
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
    Vector2 mouseDelta = GetMouseDelta();

    float rotSpeed = 0.003f;

    Vector3 offset = {camera.position.x - camera.target.x,
                      camera.position.y - camera.target.y,
                      camera.position.z - camera.target.z};

    float distance =
        sqrtf(offset.x * offset.x + offset.y * offset.y + offset.z * offset.z);
    float yaw = atan2f(offset.x, offset.z);
    float pitch = asinf(offset.y / distance);

    yaw -= mouseDelta.x * rotSpeed;
    pitch += mouseDelta.y * rotSpeed;

    if (pitch > 1.4f)
      pitch = 1.4f;
    if (pitch < -1.4f)
      pitch = -1.4f;

    camera.position.x = camera.target.x + distance * sinf(yaw) * cosf(pitch);
    camera.position.y = camera.target.y + distance * sinf(pitch);
    camera.position.z = camera.target.z + distance * cosf(yaw) * cosf(pitch);
  }

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});

  BeginMode3D(camera);

  float s = (float)env->size;
  Color edge = (Color){128, 128, 128, 255};
  // Bottom square
  DrawLine3D((Vector3){0, 0, 0}, (Vector3){s, 0, 0}, edge);
  DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, 0, s}, edge);
  DrawLine3D((Vector3){s, 0, 0}, (Vector3){s, 0, s}, edge);
  DrawLine3D((Vector3){0, 0, s}, (Vector3){s, 0, s}, edge);
  // Top square
  DrawLine3D((Vector3){0, s, 0}, (Vector3){s, s, 0}, edge);
  DrawLine3D((Vector3){0, s, 0}, (Vector3){0, s, s}, edge);
  DrawLine3D((Vector3){s, s, 0}, (Vector3){s, s, s}, edge);
  DrawLine3D((Vector3){0, s, s}, (Vector3){s, s, s}, edge);
  // Vertical edges
  DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, s, 0}, edge);
  DrawLine3D((Vector3){s, 0, 0}, (Vector3){s, s, 0}, edge);
  DrawLine3D((Vector3){0, 0, s}, (Vector3){0, s, s}, edge);
  DrawLine3D((Vector3){s, 0, s}, (Vector3){s, s, s}, edge);

  for (int x = 0; x < env->size; x++) {
    for (int y = 0; y < env->size; y++) {
      for (int z = 0; z < env->size; z++) {
        int idx = x * env->size * env->size + y * env->size + z;
        int tex = env->observations[idx];
        if (tex == EMPTY) {
          continue;
        }
        Color color = (tex == AGENT) ? (Color){0, 187, 187, 255}
                                     : (Color){187, 0, 0, 255};

        Vector3 pos = {(float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f};

        DrawCube(pos, 1.0f, 1.0f, 1.0f, color);
      }
    }
  }

  EndMode3D();
  EndDrawing();
}

void c_close(Cubed *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}