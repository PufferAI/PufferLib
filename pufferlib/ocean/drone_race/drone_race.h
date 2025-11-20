// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/stmio/drone

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"
#include "dronelib.h"

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

    Trail trail;
};

typedef struct DroneRace DroneRace;
struct DroneRace {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    Log log;
    int tick;
    int report_interval;
    int score;
    float episodic_return;

    int max_rings;
    int ring_idx;
    Ring *ring_buffer;

    int max_moves;
    int moves_left;

    Drone drone;
    Client *client;
};

void init(DroneRace *env) {
    env->log = (Log){0};
    env->tick = 0;
    env->ring_buffer = (Ring*)malloc((env->max_rings) * sizeof(Ring));
}

void add_log(DroneRace *env, float oob, float collision, float timeout) {
    env->log.score += env->score;
    env->log.episode_return += env->episodic_return;
    env->log.episode_length += env->tick;
    env->log.perf += (float)env->ring_idx / (float)env->max_rings;
    env->log.oob += oob;
    env->log.collision_rate += collision;
    env->log.timeout += timeout;
    env->log.n += 1.0f;
}

void compute_observations(DroneRace *env) {
    Drone *drone = &env->drone;

    Quat q_inv = quat_inverse(drone->state.quat);
    Ring curr_ring = env->ring_buffer[env->ring_idx];
    Ring next_ring = env->ring_buffer[env->ring_idx % env->max_rings];

    Vec3 to_curr_ring = quat_rotate(q_inv, sub3(curr_ring.pos, drone->state.pos));
    Vec3 to_next_ring = quat_rotate(q_inv, sub3(next_ring.pos, drone->state.pos));

    Vec3 curr_ring_norm = quat_rotate(q_inv, curr_ring.normal);
    Vec3 next_ring_norm = quat_rotate(q_inv, next_ring.normal);

    Vec3 linear_vel_body = quat_rotate(q_inv, drone->state.vel);
    Vec3 drone_up_world = quat_rotate(drone->state.quat, (Vec3){0.0f, 0.0f, 1.0f});

    env->observations[0] = to_curr_ring.x / GRID_X;
    env->observations[1] = to_curr_ring.y / GRID_Y;
    env->observations[2] = to_curr_ring.z / GRID_Z;

    env->observations[3] = curr_ring_norm.x;
    env->observations[4] = curr_ring_norm.y;
    env->observations[5] = curr_ring_norm.z;

    env->observations[6] = to_next_ring.x / GRID_X;
    env->observations[7] = to_next_ring.y / GRID_Y;
    env->observations[8] = to_next_ring.z / GRID_Z;

    env->observations[9] = next_ring_norm.x;
    env->observations[10] = next_ring_norm.y;
    env->observations[11] = next_ring_norm.z;

    env->observations[12] = linear_vel_body.x / drone->params.max_vel;
    env->observations[13] = linear_vel_body.y / drone->params.max_vel;
    env->observations[14] = linear_vel_body.z / drone->params.max_vel;

    env->observations[15] = drone->state.omega.x / drone->params.max_omega;
    env->observations[16] = drone->state.omega.y / drone->params.max_omega;
    env->observations[17] = drone->state.omega.z / drone->params.max_omega;

    env->observations[18] = drone_up_world.x;
    env->observations[19] = drone_up_world.y;
    env->observations[20] = drone_up_world.z;

    env->observations[21] = drone->state.quat.w;
    env->observations[22] = drone->state.quat.x;
    env->observations[23] = drone->state.quat.y;
    env->observations[24] = drone->state.quat.z;

    env->observations[25] = drone->state.rpms[0] / drone->params.max_rpm;
    env->observations[26] = drone->state.rpms[1] / drone->params.max_rpm;
    env->observations[27] = drone->state.rpms[2] / drone->params.max_rpm;
    env->observations[28] = drone->state.rpms[3] / drone->params.max_rpm;
}

void c_reset(DroneRace *env) {
    env->tick = 0;
    env->score = 0;
    env->episodic_return = 0.0f;
    env->moves_left = env->max_moves;

    // creates rings
    env->ring_idx = 0;
    float ring_radius = 2.0f;
    reset_rings(env->ring_buffer, env->max_rings, ring_radius);

    // creates drone
    Drone *drone = &env->drone;
    float size = rndf(0.05f, 0.8f);
    init_drone(drone, size, 0.1f);

    do {
        drone->state.pos = (Vec3){
            rndf(-MARGIN_X, MARGIN_X), 
            rndf(-MARGIN_Y, MARGIN_Y), 
            rndf(-MARGIN_Z, MARGIN_Z)
        };
    } while (norm3(sub3(drone->state.pos, env->ring_buffer[0].pos)) < 2.0f*ring_radius);

    drone->prev_pos = drone->state.pos;

    compute_observations(env);
}

void c_step(DroneRace *env) {
    env->tick++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->log.score = 0;

    Drone *drone = &env->drone;
    move_drone(drone, env->actions);

    // check out of bounds
    bool out_of_bounds = drone->state.pos.x < -GRID_X || drone->state.pos.x > GRID_X ||
                         drone->state.pos.y < -GRID_Y || drone->state.pos.y > GRID_Y ||
                         drone->state.pos.z < -GRID_Z || drone->state.pos.z > GRID_Z;

    if (out_of_bounds) {
        env->rewards[0] -= 1;
        env->episodic_return -= 1;
        env->terminals[0] = 1;
        add_log(env, 1.0f, 0.0f, 0.0f);
        c_reset(env);
        compute_observations(env);
        return;
    }

    // check for passing ring
    Ring *ring = &env->ring_buffer[env->ring_idx];
    float reward = check_ring(drone, ring);
    env->rewards[0] += reward;
    env->episodic_return += reward;

    if (reward > 0) {
        env->score++;
        env->ring_idx++;
    } else if (reward < 0) {
        env->terminals[0] = 1;
        add_log(env, 0.0f, 1.0f, 0.0f);
        c_reset(env);
        return;
    }

    // truncate
    env->moves_left -= 1;
    if (env->moves_left == 0 || env->ring_idx == env->max_rings) {
        env->terminals[0] = 1;
        add_log(env, 0.0f, 0.0f, env->moves_left == 0 ? 1.0f : 0.0f);
        c_reset(env);
        return;
    }

    drone->prev_pos = drone->state.pos;

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(DroneRace *env) {
    free(env->ring_buffer);

    if (env->client != NULL) {
        c_close_client(env->client);
    }
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
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

Client *make_client(DroneRace *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    InitWindow(WIDTH, HEIGHT, "PufferLib DroneRace");

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

    client->trail.index = 0;
    client->trail.count = 0;
    for (int j = 0; j < TRAIL_LENGTH; j++) {
        client->trail.pos[j] = env->drone.state.pos;
    }

    return client;
}

void DrawRing3D(Ring ring, float thickness, Color entryColor, Color exitColor) {
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

void c_render(DroneRace *env) {
    Drone *drone = &env->drone;
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

    handle_camera_controls(env->client);

    Client *client = env->client;

    client->trail.pos[client->trail.index] = env->drone.state.pos;
    client->trail.index = (client->trail.index + 1) % TRAIL_LENGTH;
    if (client->trail.count < TRAIL_LENGTH) {
        client->trail.count++;
    }
    if (env->terminals[0]) {
        client->trail.index = 0;
        client->trail.count = 0;
    }    

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    BeginMode3D(client->camera);

    // draws bounding cube
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_X * 2.0f, GRID_Y * 2.0f, GRID_Z * 2.0f,
                  WHITE);

    // draws drone body
    float r = drone->params.arm_len;
    DrawSphere((Vector3){drone->state.pos.x, drone->state.pos.y, drone->state.pos.z}, r/2.0f, RED);

    // draws rotors according to thrust
    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = (env->actions[i] + 1.0f) * 0.5f * drone->params.max_rpm;
        T[i] = drone->params.k_thrust * rpm * rpm;
    }

    const float rotor_radius = r / 4.0f;
    const float visual_arm_len = 1.0f * drone->params.arm_len;

    Vec3 rotor_offsets_body[4] = {{+r, 0.0f, 0.0f},
                                  {-r, 0.0f, 0.0f},
                                  {0.0f, +r, 0.0f},
                                  {0.0f, -r, 0.0f}};

    Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};

    for (int i = 0; i < 4; i++) {
        Vec3 world_off = quat_rotate(drone->state.quat, rotor_offsets_body[i]);

        Vector3 rotor_pos = {drone->state.pos.x + world_off.x, drone->state.pos.y + world_off.y,
                             drone->state.pos.z + world_off.z};

        float rpm = (env->actions[i] + 1.0f) * 0.5f * drone->params.max_rpm;
        float intensity = 0.75f + 0.25f * (rpm / drone->params.max_rpm);

        Color rotor_color = (Color){(unsigned char)(base_colors[i].r * intensity),
                                    (unsigned char)(base_colors[i].g * intensity),
                                    (unsigned char)(base_colors[i].b * intensity), 255};

        DrawSphere(rotor_pos, rotor_radius, rotor_color);

        DrawCylinderEx((Vector3){drone->state.pos.x, drone->state.pos.y, drone->state.pos.z}, rotor_pos, 0.02f, 0.02f, 8,
                       BLACK);
    }

    // draws line with direction and magnitude of velocity / 10
    if (norm3(drone->state.vel) > 0.1f) {
        DrawLine3D((Vector3){drone->state.pos.x, drone->state.pos.y, drone->state.pos.z},
                   (Vector3){drone->state.pos.x + drone->state.vel.x * 0.1f, drone->state.pos.y + drone->state.vel.y * 0.1f,
                             drone->state.pos.z + drone->state.vel.z * 0.1f},
                   MAGENTA);
    }

    if (client->trail.count > 2) {
        for (int j = 0; j < client->trail.count - 1; j++) {
            int idx0 = (client->trail.index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            int idx1 = (client->trail.index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
            float alpha = (float)(TRAIL_LENGTH - j) / (float)client->trail.count * 0.8f; // fade out
            Color trail_color = ColorAlpha((Color){0, 187, 187, 255}, alpha);
            DrawLine3D((Vector3){client->trail.pos[idx0].x, client->trail.pos[idx0].y, client->trail.pos[idx0].z},
                        (Vector3){client->trail.pos[idx1].x, client->trail.pos[idx1].y, client->trail.pos[idx1].z},
                        trail_color);
        }
    }

    // draws current and previous ring
    float ring_thickness = 0.2f;
    DrawRing3D(env->ring_buffer[env->ring_idx], ring_thickness, GREEN, BLUE);
    if (env->ring_idx > 0) {
        DrawRing3D(env->ring_buffer[env->ring_idx - 1], ring_thickness, GREEN, BLUE);
    }

    EndMode3D();

    // Draw 2D stats
    DrawText(TextFormat("Targets left: %d", env->max_rings - env->ring_idx), 10, 10, 20, WHITE);
    DrawText(TextFormat("Moves left: %d", env->moves_left), 10, 40, 20, WHITE);
    DrawText(TextFormat("Episode Return: %.2f", env->episodic_return), 10, 70, 20, WHITE);

    DrawText("Motor Thrusts:", 10, 110, 20, WHITE);
    DrawText(TextFormat("Front: %.3f", T[0]), 10, 135, 18, ORANGE);
    DrawText(TextFormat("Back:  %.3f", T[1]), 10, 155, 18, PURPLE);
    DrawText(TextFormat("Right: %.3f", T[2]), 10, 175, 18, LIME);
    DrawText(TextFormat("Left:  %.3f", T[3]), 10, 195, 18, SKYBLUE);

    DrawText(TextFormat("Pos: (%.1f, %.1f, %.1f)", drone->state.pos.x, drone->state.pos.y, drone->state.pos.z), 10, 225, 18,
             WHITE);
    DrawText(TextFormat("Vel: %.2f m/s", norm3(drone->state.vel)), 10, 245, 18, WHITE);

    DrawText("Left click + drag: Rotate camera", 10, 275, 16, LIGHTGRAY);
    DrawText("Mouse wheel: Zoom in/out", 10, 295, 16, LIGHTGRAY);

    EndDrawing();
}
