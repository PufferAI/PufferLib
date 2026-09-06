// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>

#include "drone.h"
#include "dronelib.h"
#include "raylib.h"
// NOTE: no raymath.h — float3 conflicts with CUDA when included via pufferl.cu

#define R (Color){255, 0, 0, 255}
#define W (Color){255, 255, 255, 255}
#define B (Color){0, 0, 255, 255}
Color COLORS[64] = {B, B, B, R, R, R, R, R,
                    B, B, B, W, W, W, W, W,
                    B, B, B, R, R, R, R, R,
                    B, B, B, W, W, W, W, W,
                    R, R, R, R, R, R, R, R,
                    W, W, W, W, W, W, W, W,
                    R, R, R, R, R, R, R, R,
                    W, W, W, W, W, W, W, W};
#undef R
#undef W
#undef B

// 3D model config
#define MODEL_SCALE_NORMAL 3.0f
#define NUM_PROPELLERS 4
static const int PROP_MESH_IDX[NUM_PROPELLERS] = {8, 6, 5, 7};
static const float PROP_DIRS[NUM_PROPELLERS] = {1.0f, -1.0f, 1.0f, -1.0f};

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

    Trail* trails;

    int selected_drone;
    bool inspect_mode;
    bool follow_mode;
    int target_fps;

    // Drone 3D model
    Model drone_model;
    bool model_loaded;
    bool use_3d_model;
    float* prop_angles;
    Vec3 prop_centers[NUM_PROPELLERS];
    float model_scale;
};

// Minimal matrix helpers (avoid raymath.h / CUDA float3 clash)
static inline Matrix drone_matrix_identity(void) {
    Matrix m = {0};
    m.m0 = m.m5 = m.m10 = m.m15 = 1.0f;
    return m;
}

static inline Matrix drone_matrix_multiply(Matrix left, Matrix right) {
    Matrix result = {0};
    result.m0 = left.m0*right.m0 + left.m1*right.m4 + left.m2*right.m8 + left.m3*right.m12;
    result.m1 = left.m0*right.m1 + left.m1*right.m5 + left.m2*right.m9 + left.m3*right.m13;
    result.m2 = left.m0*right.m2 + left.m1*right.m6 + left.m2*right.m10 + left.m3*right.m14;
    result.m3 = left.m0*right.m3 + left.m1*right.m7 + left.m2*right.m11 + left.m3*right.m15;
    result.m4 = left.m4*right.m0 + left.m5*right.m4 + left.m6*right.m8 + left.m7*right.m12;
    result.m5 = left.m4*right.m1 + left.m5*right.m5 + left.m6*right.m9 + left.m7*right.m13;
    result.m6 = left.m4*right.m2 + left.m5*right.m6 + left.m6*right.m10 + left.m7*right.m14;
    result.m7 = left.m4*right.m3 + left.m5*right.m7 + left.m6*right.m11 + left.m7*right.m15;
    result.m8 = left.m8*right.m0 + left.m9*right.m4 + left.m10*right.m8 + left.m11*right.m12;
    result.m9 = left.m8*right.m1 + left.m9*right.m5 + left.m10*right.m9 + left.m11*right.m13;
    result.m10 = left.m8*right.m2 + left.m9*right.m6 + left.m10*right.m10 + left.m11*right.m14;
    result.m11 = left.m8*right.m3 + left.m9*right.m7 + left.m10*right.m11 + left.m11*right.m15;
    result.m12 = left.m12*right.m0 + left.m13*right.m4 + left.m14*right.m8 + left.m15*right.m12;
    result.m13 = left.m12*right.m1 + left.m13*right.m5 + left.m14*right.m9 + left.m15*right.m13;
    result.m14 = left.m12*right.m2 + left.m13*right.m6 + left.m14*right.m10 + left.m15*right.m14;
    result.m15 = left.m12*right.m3 + left.m13*right.m7 + left.m14*right.m11 + left.m15*right.m15;
    return result;
}

static inline Matrix drone_matrix_translate(float x, float y, float z) {
    Matrix m = drone_matrix_identity();
    m.m12 = x; m.m13 = y; m.m14 = z;
    return m;
}

static inline Matrix drone_matrix_scale(float x, float y, float z) {
    Matrix m = {0};
    m.m0 = x; m.m5 = y; m.m10 = z; m.m15 = 1.0f;
    return m;
}

static inline Matrix drone_matrix_rotate_z(float angle) {
    float c = cosf(angle), s = sinf(angle);
    Matrix m = drone_matrix_identity();
    m.m0 = c; m.m1 = s; m.m4 = -s; m.m5 = c;
    return m;
}

// Convert dronelib Quat to raylib Matrix
static inline Matrix quat_to_matrix(Quat q) {
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    Matrix m = {0};
    m.m0 = 1.0f - 2.0f * (yy + zz);
    m.m1 = 2.0f * (xy + wz);
    m.m2 = 2.0f * (xz - wy);
    m.m4 = 2.0f * (xy - wz);
    m.m5 = 1.0f - 2.0f * (xx + zz);
    m.m6 = 2.0f * (yz + wx);
    m.m8 = 2.0f * (xz + wy);
    m.m9 = 2.0f * (yz - wx);
    m.m10 = 1.0f - 2.0f * (xx + yy);
    m.m15 = 1.0f;
    return m;
}

void close_client(Client* client) {
    if (client->model_loaded) {
        UnloadModel(client->drone_model);
    }

    if (client->prop_angles) {
        free(client->prop_angles);
    }

    CloseWindow();
    free(client->trails);
    free(client);
}

static void update_camera_position(Client* c, Vec3 target_pos) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    if (c->follow_mode) {
        c->camera.target = (Vector3){target_pos.x, target_pos.y, target_pos.z};
        c->camera.position = (Vector3){target_pos.x + x, target_pos.y + y, target_pos.z + z};
    } else {
        c->camera.target = (Vector3){0, 0, 0};
        c->camera.position = (Vector3){x, y, z};
    }
}

void handle_camera_controls(Client* client, Vec3 target_pos, float min_zoom) {
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

        update_camera_position(client, target_pos);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, min_zoom, 100.0f);
        update_camera_position(client, target_pos);
    }
}

void handle_drone_selection(Client* client, int num_agents, float dt) {
    static float repeat_timer = 0;
    static bool key_held = false;

    if (IsKeyDown(KEY_D) || IsKeyDown(KEY_A)) {
        if (!key_held || repeat_timer <= 0) {
            if (IsKeyDown(KEY_D)) {
                client->selected_drone = (client->selected_drone + 1) % num_agents;
            }

            if (IsKeyDown(KEY_A)) {
                client->selected_drone = (client->selected_drone - 1 + num_agents) % num_agents;
            }

            repeat_timer = key_held ? 0.05f : 0.3f;
            key_held = true;
        }

        repeat_timer -= dt;
    } else {
        key_held = false;
        repeat_timer = 0;
    }
}

void handle_fps_control(Client* client, float dt) {
    static float repeat_timer = 0;
    static bool key_held = false;

    if (IsKeyDown(KEY_W) || IsKeyDown(KEY_S)) {
        if (!key_held || repeat_timer <= 0) {
            if (IsKeyDown(KEY_W)) {
                client->target_fps += 10;
                if (client->target_fps > 240) client->target_fps = 240;
            }

            if (IsKeyDown(KEY_S)) {
                client->target_fps -= 10;
                if (client->target_fps < 10) client->target_fps = 10;
            }

#ifndef __EMSCRIPTEN__
            SetTargetFPS(client->target_fps);
#endif
            repeat_timer = key_held ? 0.05f : 0.3f;
            key_held = true;
        }

        repeat_timer -= dt;
    } else {
        key_held = false;
        repeat_timer = 0;
    }
}

// Compute center of mesh from vertices
static Vec3 compute_mesh_center(Mesh* mesh) {
    Vec3 center = {0, 0, 0};

    for (int v = 0; v < mesh->vertexCount; v++) {
        center.x += mesh->vertices[v * 3 + 0];
        center.y += mesh->vertices[v * 3 + 1];
        center.z += mesh->vertices[v * 3 + 2];
    }

    return scalmul3(center, 1.0f / mesh->vertexCount);
}

Client* make_client(DroneEnv* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

#if !defined(__EMSCRIPTEN__) && !defined(PLATFORM_WEB)
    SetConfigFlags(FLAG_MSAA_4X_HINT);
#endif
    InitWindow(WIDTH, HEIGHT, "PufferLib Drone");

#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif
    client->target_fps = 60;

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

    Vec3 origin = {0, 0, 0};
    update_camera_position(client, origin);

    // Initialize trail buffer
    client->trails = (Trail*)calloc(env->num_agents, sizeof(Trail));
    for (int i = 0; i < env->num_agents; i++) {
        Trail* trail = &client->trails[i];
        trail->index = 0;
        trail->count = 0;
        for (int j = 0; j < TRAIL_LENGTH; j++) {
            trail->pos[j] = env->drones[i].state.pos;
        }
    }

    client->selected_drone = 0;
    client->inspect_mode = false;
    client->follow_mode = false;
    client->target_fps = 60;
    client->model_loaded = false;
    client->model_scale = MODEL_SCALE_NORMAL;

    // Load 3D model
    const char* model_paths[] = {"resources/crazyflie.glb", "resources/drone/crazyflie.glb",
                                 "crazyflie.glb", NULL};

    for (int i = 0; model_paths[i] != NULL; i++) {
        if (FileExists(model_paths[i])) {
            client->drone_model = LoadModel(model_paths[i]);

            if (client->drone_model.meshCount > 0) {
                client->model_loaded = true;
                TraceLog(LOG_INFO, "Loaded drone model: %s", model_paths[i]);

                // Cache propeller centers
                for (int p = 0; p < NUM_PROPELLERS; p++) {
                    int idx = PROP_MESH_IDX[p];

                    if (idx < client->drone_model.meshCount) {
                        client->prop_centers[p] =
                            compute_mesh_center(&client->drone_model.meshes[idx]);
                    }
                }

                break;
            }
        }
    }

#if defined(__EMSCRIPTEN__) || defined(PLATFORM_WEB)
    client->use_3d_model = false;
#else
    client->use_3d_model = client->model_loaded;
#endif
    client->prop_angles = (float*)calloc(env->num_agents * NUM_PROPELLERS, sizeof(float));

    return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_GREEN = (Color){0, 220, 80, 255};

void DrawRing3D(Target ring, float thickness, Color entryColor, Color exitColor) {
    float half_thick = thickness / 2.0f;

    Vector3 center_pos = {ring.pos.x, ring.pos.y, ring.pos.z};

    Vector3 entry_start_pos = {center_pos.x - half_thick * ring.normal.x,
                               center_pos.y - half_thick * ring.normal.y,
                               center_pos.z - half_thick * ring.normal.z};

    DrawCylinderWiresEx(entry_start_pos, center_pos, ring.radius, ring.radius, 32, entryColor);

    Vector3 exit_end_pos = {center_pos.x + half_thick * ring.normal.x,
                            center_pos.y + half_thick * ring.normal.y,
                            center_pos.z + half_thick * ring.normal.z};

    DrawCylinderWiresEx(center_pos, exit_end_pos, ring.radius, ring.radius, 32, exitColor);
}

void DrawDroneModel(Client* client, Drone* agent, int drone_idx, float dt, Color body_color) {
    if (!client->model_loaded) return;

    Model* model = &client->drone_model;
    float* angles = &client->prop_angles[drone_idx * NUM_PROPELLERS];

    // Update propeller angles from RPM
    for (int p = 0; p < NUM_PROPELLERS; p++) {
        float rpm = agent->state.rpms[p];
        angles[p] += rpm * (2.0f * PI / 60.0f) * dt * PROP_DIRS[p];

        if (angles[p] > 2.0f * PI) angles[p] -= 2.0f * PI;
        if (angles[p] < 0.0f) angles[p] += 2.0f * PI;
    }

    // Build world transform matrices using client's model_scale
    float scale = client->model_scale;
    Matrix mScale = drone_matrix_scale(scale, scale, scale);
    Matrix mRot = quat_to_matrix(agent->state.quat);
    Matrix mTrans = drone_matrix_translate(agent->state.pos.x, agent->state.pos.y, agent->state.pos.z);

    Matrix droneWorld = drone_matrix_multiply(drone_matrix_multiply(mScale, mRot), mTrans);

    // Draw each mesh
    for (int m = 0; m < model->meshCount; m++) {
        Matrix meshWorld = droneWorld;

        // Check if this mesh is a propeller
        bool is_prop = false;
        for (int p = 0; p < NUM_PROPELLERS; p++) {
            if (m == PROP_MESH_IDX[p]) {
                is_prop = true;
                Vec3 c = client->prop_centers[p];
                Matrix toOrigin = drone_matrix_translate(-c.x, -c.y, -c.z);
                Matrix spin = drone_matrix_rotate_z(angles[p]);
                Matrix fromOrigin = drone_matrix_translate(c.x, c.y, c.z);
                Matrix propLocal = drone_matrix_multiply(drone_matrix_multiply(toOrigin, spin), fromOrigin);
                meshWorld = drone_matrix_multiply(propLocal, droneWorld);
                break;
            }
        }

        Material mat = model->materials[model->meshMaterial[m]];

        Color origColor = mat.maps[MATERIAL_MAP_DIFFUSE].color;
        int brightness = (origColor.r + origColor.g + origColor.b) / 3;

        if (is_prop || brightness > 64) {
            mat.maps[MATERIAL_MAP_DIFFUSE].color = body_color;
        } else {
            mat.maps[MATERIAL_MAP_DIFFUSE].color = origColor;
        }

        DrawMesh(model->meshes[m], mat, meshWorld);
    }
}

void DrawDronePrimitive(Client* client, Drone* agent, float* actions, Color body_color) {
    const float scale = client->model_scale;

    DrawSphere((Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z}, 0.06f * scale,
               body_color);

    const float rotor_radius = 0.03f * scale;
    const float arm_len = 0.15f * scale;
    const float diag = arm_len * 0.7071f; // 1/sqrt(2)

    Vec3 rotor_offsets[4] = {
        {+diag, +diag, 0.0f}, {+diag, -diag, 0.0f}, {-diag, -diag, 0.0f}, {-diag, +diag, 0.0f}};

    for (int j = 0; j < 4; j++) {
        Vec3 world_off = quat_rotate(agent->state.quat, rotor_offsets[j]);

        Vector3 rotor_pos = {agent->state.pos.x + world_off.x, agent->state.pos.y + world_off.y,
                             agent->state.pos.z + world_off.z};

        float rpm = (actions[j] + 1.0f) * 0.5f * agent->params.max_rpm;
        float intensity = 0.75f + 0.25f * (rpm / agent->params.max_rpm);

        Color rotor_color = (Color){(unsigned char)(body_color.r * intensity),
                                    (unsigned char)(body_color.g * intensity),
                                    (unsigned char)(body_color.b * intensity), 255};

        DrawSphere(rotor_pos, rotor_radius, rotor_color);
        DrawCylinderEx((Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
                       rotor_pos, 0.004f * scale, 0.004f * scale, 8, BLACK);
    }
}

// Task-specific overlays
static void render_task(DroneEnv* env, Client* client) {
    if (env->task != TASK_RACE) return;
    RaceConfig* cfg = (RaceConfig*)env->task_config;
    RaceState* state = (RaceState*)env->task_state;
    for (int i = 0; i < cfg->max_rings; i++)
        DrawRing3D(state->ring_buffer[i], 0.1f, GREEN, BLUE);
}

void puf_render(DroneEnv* env) {
    if (env->client == NULL) {
        env->client = make_client(env);

        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }

    // Capture modes must not hard-exit the process when the display is headless
    // or the window is closed programmatically mid-recording.
    // Raylib 5.5 WindowShouldClose() on web is emscripten_sleep(16) every call
    // (ASYNCIFY makes that ~40ms -> ~25fps). Never call it on web.
#if defined(__EMSCRIPTEN__) || defined(PLATFORM_WEB)
    if (IsKeyDown(KEY_ESCAPE)) {
        puf_close(env);
        exit(0);
    }
#else
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        if (!IsWindowReady()) {
            return;
        }
        if (IsKeyDown(KEY_ESCAPE)) {
            puf_close(env);
            exit(0);
        }
        return;
    }
#endif

    Client* client = env->client;
    float dt = ACTION_DT;

    // Get selected drone position for camera
    Vec3 drone_pos = env->drones[client->selected_drone].state.pos;

    handle_camera_controls(client, drone_pos, 1.0f);
    handle_drone_selection(client, env->num_agents, dt);
    handle_fps_control(client, dt);

    if (IsKeyPressed(KEY_I)) {
        client->inspect_mode = !client->inspect_mode;
        // When entering inspect mode, turn on follow mode by default
        if (client->inspect_mode) {
            client->follow_mode = true;
            update_camera_position(client, drone_pos);
        } else {
            // When exiting inspect mode, turn off follow mode
            client->follow_mode = false;
            update_camera_position(client, drone_pos);
        }
    }

    if (IsKeyPressed(KEY_M) && client->model_loaded) {
        client->use_3d_model = !client->use_3d_model;
    }

    if (IsKeyPressed(KEY_F)) {
        client->follow_mode = !client->follow_mode;
    }

    if (IsKeyPressed(KEY_TAB)) {
        drone_set_task(env, drone_next_task(env->task));
        drone_pos = env->drones[client->selected_drone].state.pos;
        if (client->follow_mode) {
            update_camera_position(client, drone_pos);
        }
    }

    // Update camera position every frame when in follow mode
    if (client->follow_mode) {
        update_camera_position(client, drone_pos);
    }

    bool inspect_mode = client->inspect_mode;

    // Update trails. Clear on terminal or position teleport (episode reset /
    // task swap); otherwise a line is drawn from the old path to the new spawn.
    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->drones[i];
        Trail* trail = &client->trails[i];
        Vec3 pos = agent->state.pos;
        bool teleported = false;
        if (trail->count > 0) {
            int last = (trail->index - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            // Larger than any single-frame motion under max_vel * ACTION_DT.
            teleported = norm3(sub3(pos, trail->pos[last])) > 2.0f;
        }
        bool terminal = env->agents[i].terminals && env->agents[i].terminals[0];
        if (terminal || teleported) {
            trail->index = 0;
            trail->count = 0;
            for (int j = 0; j < TRAIL_LENGTH; j++) {
                trail->pos[j] = pos;
            }
        }
        trail->pos[trail->index] = pos;
        trail->index = (trail->index + 1) % TRAIL_LENGTH;
        if (trail->count < TRAIL_LENGTH) {
            trail->count++;
        }
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    BeginMode3D(client->camera);

    // Bounding cube
    DrawCubeWires((Vector3){0, 0, 0}, GRID_X * 2.0f, GRID_Y * 2.0f, GRID_Z * 2.0f, WHITE);

    // Draw drones
    for (int i = 0; i < env->num_agents; i++) {
        Drone* agent = &env->drones[i];
        bool is_selected = (i == client->selected_drone);
        Color body_color = (inspect_mode && is_selected) ? PUFF_GREEN : COLORS[i % 64];

        if (client->use_3d_model && client->model_loaded) {
            DrawDroneModel(client, agent, i, dt, body_color);
        } else {
            DrawDronePrimitive(client, agent, env->agents[i].actions, body_color);
        }

        // Velocity vector
        if (norm3(agent->state.vel) > 0.1f) {
            Vec3 p = agent->state.pos;
            Vec3 v = scalmul3(agent->state.vel, 0.1f);
            DrawLine3D((Vector3){p.x, p.y, p.z}, (Vector3){p.x + v.x, p.y + v.y, p.z + v.z},
                       MAGENTA);
        }

        // Target line (shown in inspect mode)
        if (inspect_mode && is_selected) {
            Vec3 p = agent->state.pos;
            Vec3 t = agent->target->pos;
            DrawLine3D((Vector3){p.x, p.y, p.z}, (Vector3){t.x, t.y, t.z},
                       ColorAlpha(PUFF_GREEN, 0.5f));
        }

        // Draw trailing path for each drone
        Trail* trail = &client->trails[i];
        if (trail->count > 2) {
            Color trail_color = (inspect_mode && is_selected) ? PUFF_GREEN : PUFF_CYAN;

            for (int j = 0; j < trail->count - 1; j++) {
                int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
                int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;

                float alpha = (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f;
                Vec3 p0 = trail->pos[idx0], p1 = trail->pos[idx1];

                DrawLine3D((Vector3){p0.x, p0.y, p0.z}, (Vector3){p1.x, p1.y, p1.z},
                           ColorAlpha(trail_color, alpha));
            }
        }
    }

    // Task-specific rendering
    render_task(env, client);

    // Hover-family: cheap cubes so sphere/cube/flag formations are visible
    // without 64 DrawSphere tessellations (that path was a web FPS cliff).
    if (env->task != TASK_RACE) {
        float s = (env->task == TASK_HOVER) ? 0.12f : 0.18f;
        for (int i = 0; i < env->num_agents; i++) {
            Vec3 t = env->drones[i].target->pos;
            bool is_selected = (i == client->selected_drone);
            float size = is_selected ? s * 1.25f : s;
            Color c = is_selected ? (Color){0, 255, 100, 200} : (Color){0, 220, 255, 160};
            DrawCube((Vector3){t.x, t.y, t.z}, size, size, size, c);
        }
    } else if (inspect_mode) {
        float target_size = 0.1f;
        for (int i = 0; i < env->num_agents; i++) {
            Vec3 t = env->drones[i].target->pos;
            bool is_selected = (i == client->selected_drone);
            float size = is_selected ? target_size * 1.1f : target_size;
            DrawCube((Vector3){t.x, t.y, t.z}, size, size, size,
                     is_selected ? (Color){0, 255, 100, 180} : (Color){0, 255, 255, 100});
        }
    }

    EndMode3D();

    // Heads up display
    int y = 10;
    DrawText(TextFormat("Task: %s", task_name(env->task)), 10, y, 20, WHITE);
    y += 25;
    DrawText(TextFormat("Step: %d / %d", env->drones[client->selected_drone].episode_length,
                        task_horizon(env)), 10, y, 20, WHITE);
    y += 25;
    DrawText(TextFormat("FPS: %d (target %d, W/S)", GetFPS(), client->target_fps), 10, y, 18, WHITE);
    y += 22;
    if (client->model_loaded) {
        DrawText(TextFormat("Render: %s (M)", client->use_3d_model ? "3D Model" : "Primitive"), 10,
                 y, 18, client->use_3d_model ? PUFF_GREEN : LIGHTGRAY);
    }
    y += 22;
    DrawText(TextFormat("Follow: %s (F)", client->follow_mode ? "ON" : "OFF"), 10, y, 18,
             client->follow_mode ? PUFF_GREEN : LIGHTGRAY);
    y += 30;

    // Inspect mode stats
    if (inspect_mode) {
        int idx = client->selected_drone;
        Drone* agent = &env->drones[idx];

        DrawText(TextFormat("Drone: %d / %d (A/D to switch)", idx, env->num_agents - 1), 10, y, 20,
                 PUFF_GREEN);
        y += 30;
        DrawText(TextFormat("Pos: (%.1f, %.1f, %.1f)", agent->state.pos.x, agent->state.pos.y,
                            agent->state.pos.z),
                 10, y, 18, WHITE);
        y += 20;
        DrawText(TextFormat("Vel: %.2f m/s", norm3(agent->state.vel)), 10, y, 18, WHITE);
        y += 20;
        DrawText(TextFormat("Omega: (%.1f, %.1f, %.1f)", agent->state.omega.x, agent->state.omega.y,
                            agent->state.omega.z),
                 10, y, 18, WHITE);
        y += 25;

        // Motor RPM bars
        DrawText("Motor RPMs:", 10, y, 18, WHITE);
        y += 22;
        int bar_w = 150, bar_h = 14;
        Color motor_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};
        const char* motor_names[4] = {"M1", "M2", "M3", "M4"};

        for (int m = 0; m < 4; m++) {
            float pct = clampf(agent->state.rpms[m] / agent->params.max_rpm, 0.0f, 1.0f);
            int fill_w = (int)(pct * bar_w);

            // Label
            DrawText(motor_names[m], 10, y, 16, motor_colors[m]);

            // Bar background
            DrawRectangle(35, y, bar_w, bar_h, (Color){40, 40, 40, 255});

            // Bar fill
            DrawRectangle(35, y, fill_w, bar_h, motor_colors[m]);

            // Bar outline
            DrawRectangleLines(35, y, bar_w, bar_h, LIGHTGRAY);

            // RPM value text
            DrawText(TextFormat("%.0f", agent->state.rpms[m]), 35 + bar_w + 5, y, 14, WHITE);

            y += bar_h + 4;
        }

        y += 10;

        DrawText(TextFormat("Episode Return: %.4f", agent->episode_return), 10, y, 18, WHITE);
        y += 20;
        DrawText(TextFormat("Episode Length: %d", agent->episode_length), 10, y, 18, WHITE);
        y += 30;
    }

    // Controls (always visible)
    DrawText("Left click + drag: Rotate camera", 10, y, 16, LIGHTGRAY);
    y += 18;
    DrawText("Mouse wheel: Zoom in/out", 10, y, 16, LIGHTGRAY);
    y += 18;
    DrawText("Tab: Next task (hover/race/sphere/cube/flag)", 10, y, 16, LIGHTGRAY);
    y += 18;
    DrawText(TextFormat("I: Inspect mode [%s]", inspect_mode ? "ON" : "OFF"), 10, y, 16,
             inspect_mode ? PUFF_GREEN : LIGHTGRAY);

    EndDrawing();
    puf_web_vsync();
}