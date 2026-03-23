// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>

#include "drone.h"
#include "dronelib.h"
#include "raylib.h"

#define R (Color){255, 0, 0, 255}
#define W (Color){255, 255, 255, 255}
#define B (Color){0, 0, 255, 255}
Color COLORS[64] = {
    W, B, B, R, R, B, B, W,
    B, W, B, R, R, B, W, B,
    B, B, W, R, R, W, B, B,
    R, R, R, R, R, R, R, R,
    R, R, R, R, R, R, R, R,
    B, B, W, R, R, W, B, B,
    B, W, B, R, R, B, W, B,
    W, B, B, R, R, B, B, W
};
#undef R
#undef W
#undef B

typedef struct Client Client;

struct Client {
  Camera3D camera;
  float width;
  float height;

  float camera_distance;
  float camera_azimuth;
  float camera_elevation;
  bool is_dragging;
  Vector2 last_mouse_pos;

  // Trailing path buffer (for rendering only)
  Trail *trails;
};

void c_close_client(Client *client) {
  CloseWindow();
  free(client->trails);
  free(client);
}

static void update_camera_position(Client *c) {
  float r = c->camera_distance;
  float az = c->camera_azimuth;
  float el = c->camera_elevation;

  float x = r * cosf(el) * cosf(az);
  float y = r * cosf(el) * sinf(az);
  float z = r * sinf(el);

  c->camera.position = (Vector3){x, y, z};
  c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
  Vector2 mouse_pos = GetMousePosition();

  if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = true;
    client->last_mouse_pos = mouse_pos;
  }

  if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = false;
  }

  if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                           mouse_pos.y - client->last_mouse_pos.y};

    float sensitivity = 0.005f;

    client->camera_azimuth -= mouse_delta.x * sensitivity;

    client->camera_elevation += mouse_delta.y * sensitivity;
    client->camera_elevation =
        clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

    client->last_mouse_pos = mouse_pos;

    update_camera_position(client);
  }

  float wheel = GetMouseWheelMove();
  if (wheel != 0) {
    client->camera_distance -= wheel * 2.0f;
    client->camera_distance = clampf(client->camera_distance, 5.0f, 100.0f);
    update_camera_position(client);
  }
}

Client *make_client(DroneEnv *env) {
  Client *client = (Client *)calloc(1, sizeof(Client));

  client->width = WIDTH;
  client->height = HEIGHT;

  SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
  InitWindow(WIDTH, HEIGHT, "PufferLib Drone");

#ifndef __EMSCRIPTEN__
  SetTargetFPS(60);
#endif

  if (!IsWindowReady()) {
    TraceLog(LOG_ERROR, "Window failed to initialize\n");
    free(client);
    return NULL;
  }

  client->camera_distance = 40.0f;
  client->camera_azimuth = 0.0f;
  client->camera_elevation = PI / 10.0f;
  client->is_dragging = false;
  client->last_mouse_pos = (Vector2){0.0f, 0.0f};

  client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
  client->camera.fovy = 45.0f;
  client->camera.projection = CAMERA_PERSPECTIVE;

  update_camera_position(client);

  // Initialize trail buffer
  client->trails = (Trail *)calloc(env->num_agents, sizeof(Trail));
  for (int i = 0; i < env->num_agents; i++) {
    Trail *trail = &client->trails[i];
    trail->index = 0;
    trail->count = 0;
    for (int j = 0; j < TRAIL_LENGTH; j++) {
      trail->pos[j] = env->agents[i].state.pos;
    }
  }

  return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void DrawRing3D(Target ring, float thickness, Color entryColor, Color exitColor) {
  float half_thick = thickness / 2.0f;

  Vector3 center_pos = {ring.pos.x, ring.pos.y, ring.pos.z};

  Vector3 entry_start_pos = {center_pos.x - half_thick * ring.normal.x,
                             center_pos.y - half_thick * ring.normal.y,
                             center_pos.z - half_thick * ring.normal.z};

  DrawCylinderWiresEx(entry_start_pos, center_pos, ring.radius, ring.radius, 32,
                      entryColor);

  Vector3 exit_end_pos = {center_pos.x + half_thick * ring.normal.x,
                          center_pos.y + half_thick * ring.normal.y,
                          center_pos.z + half_thick * ring.normal.z};

  DrawCylinderWiresEx(center_pos, exit_end_pos, ring.radius, ring.radius, 32,
                      exitColor);
}

void c_render(DroneEnv *env) {
  if (env->client == NULL) {
    env->client = make_client(env);
    if (env->client == NULL) {
      TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
      return;
    }
  }

  if (WindowShouldClose()) {
    c_close(env);
    exit(0);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    c_close(env);
    exit(0);
  }

  if (IsKeyPressed(KEY_SPACE)) {
    env->task = (DroneTask)((env->task + 1) % TASK_N);

    if (env->task == RACE) {
      reset_rings(env->ring_buffer, env->max_rings);
    }

    for (int i = 0; i < env->num_agents; i++) {
      set_target(env->task, env->agents, i, env->num_agents);
    }
  }

  handle_camera_controls(env->client);

  Client *client = env->client;

  for (int i = 0; i < env->num_agents; i++) {
    Drone *agent = &env->agents[i];
    Trail *trail = &client->trails[i];
    trail->pos[trail->index] = agent->state.pos;
    trail->index = (trail->index + 1) % TRAIL_LENGTH;
    if (trail->count < TRAIL_LENGTH) {
      trail->count++;
    }
    if (env->terminals[i]) {
      trail->index = 0;
      trail->count = 0;
    }
  }

  BeginDrawing();
  ClearBackground(PUFF_BACKGROUND);

  BeginMode3D(client->camera);

  // draws bounding cube
  DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_X * 2.0f, GRID_Y * 2.0f,
                GRID_Z * 2.0f, WHITE);

  for (int i = 0; i < env->num_agents; i++) {
    Drone *agent = &env->agents[i];

    // draws drone body
    Color body_color = COLORS[i];
    DrawSphere(
        (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
        0.3f, body_color);

    // draws rotors according to thrust
    float T[4];
    for (int j = 0; j < 4; j++) {
      float rpm =
          (env->actions[4 * i + j] + 1.0f) * 0.5f * agent->params.max_rpm;
      T[j] = agent->params.k_thrust * rpm * rpm;
    }

    const float rotor_radius = 0.15f;
    const float visual_arm_len = agent->params.arm_len * 4.0f;

    Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                  {-visual_arm_len, 0.0f, 0.0f},
                                  {0.0f, +visual_arm_len, 0.0f},
                                  {0.0f, -visual_arm_len, 0.0f}};

    // Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};
    Color base_colors[4] = {body_color, body_color, body_color, body_color};

    for (int j = 0; j < 4; j++) {
      Vec3 world_off = quat_rotate(agent->state.quat, rotor_offsets_body[j]);

      Vector3 rotor_pos = {agent->state.pos.x + world_off.x,
                           agent->state.pos.y + world_off.y,
                           agent->state.pos.z + world_off.z};

      float rpm =
          (env->actions[4 * i + j] + 1.0f) * 0.5f * agent->params.max_rpm;
      float intensity = 0.75f + 0.25f * (rpm / agent->params.max_rpm);

      Color rotor_color =
          (Color){(unsigned char)(base_colors[j].r * intensity),
                  (unsigned char)(base_colors[j].g * intensity),
                  (unsigned char)(base_colors[j].b * intensity), 255};

      DrawSphere(rotor_pos, rotor_radius, rotor_color);

      DrawCylinderEx(
          (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
          rotor_pos, 0.02f, 0.02f, 8, BLACK);
    }

    // draws line with direction and magnitude of velocity / 10
    if (norm3(agent->state.vel) > 0.1f) {
      DrawLine3D(
          (Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
          (Vector3){agent->state.pos.x + agent->state.vel.x * 0.1f,
                    agent->state.pos.y + agent->state.vel.y * 0.1f,
                    agent->state.pos.z + agent->state.vel.z * 0.1f},
          MAGENTA);
    }

    // Draw trailing path
    Trail *trail = &client->trails[i];
    if (trail->count <= 2) {
      continue;
    }
    for (int j = 0; j < trail->count - 1; j++) {
      int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
      int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
      float alpha =
          (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f; // fade out
      Color trail_color = ColorAlpha((Color){0, 187, 187, 255}, alpha);
      DrawLine3D(
          (Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
          (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
          trail_color);
    }
  }

  // Rings
  if (env->task == RACE) {
    float ring_thickness = 0.2f;
    for (int i = 0; i < env->max_rings; i++) {
      Target ring = env->ring_buffer[i];
      DrawRing3D(ring, ring_thickness, GREEN, BLUE);
    }
  }

  if (IsKeyDown(KEY_TAB)) {
    for (int i = 0; i < env->num_agents; i++) {
      Drone *agent = &env->agents[i];
      Vec3 target_pos = agent->target->pos;
      DrawSphere((Vector3){target_pos.x, target_pos.y, target_pos.z}, 0.45f,
                 (Color){0, 255, 255, 100});
    }
  }

  EndMode3D();

  DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
  DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);
  DrawText(TextFormat("Task: %s", TASK_NAMES[env->task]), 10, 50, 16,
           PUFF_WHITE);

  EndDrawing();
}
