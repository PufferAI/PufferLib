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

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50

// Simulation properties
#define GRID_SIZE 10.0f
#define RING_RAD 2.0f
#define RING_MARGIN 4.0f
#define DT 0.02f

// Physical constants for the drone
#define MASS 1.0f       // kg
#define IXX 0.01f       // kgm^2
#define IYY 0.01f       // kgm^2
#define IZZ 0.02f       // kgm^2
#define ARM_LEN 0.1f    // m
#define K_THRUST 3e-5f  // thrust coefficient
#define K_ANG_DAMP 0.2f // angular damping coefficient
#define K_DRAG 1e-6f    // drag (torque) coefficient
#define B_DRAG 0.1f     // linear drag coefficient
#define GRAVITY 9.81f   // m/s^2
#define MAX_RPM 750.0f  // rad/s
#define MAX_VEL 50.0f   // m/s
#define MAX_OMEGA 50.0f // rad/s
#define MOTOR_TAU 0.15f // motor time constant in seconds (realistic for small quadcopter motors)

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
    float perf;
    float n;
};

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos;
    Quat orientation;
    Vec3 normal;
    float radius;
} Ring;

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }

static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }

static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline void clamp3(Vec3 *vec, float min, float max) {
    vec->x = clampf(vec->x, min, max);
    vec->y = clampf(vec->y, min, max);
    vec->z = clampf(vec->z, min, max);
}

static inline void clamp4(float a[4], float min, float max) {
    a[0] = clampf(a[0], min, max);
    a[1] = clampf(a[1], min, max);
    a[2] = clampf(a[2], min, max);
    a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
    Quat out;
    out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return out;
}

static inline void quat_normalize(Quat *q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = {0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = {q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) { return (Quat){q.w, -q.x, -q.y, -q.z}; }

Quat rndquat() {
    float u1 = rndf(0.0f, 1.0f);
    float u2 = rndf(0.0f, 1.0f);
    float u3 = rndf(0.0f, 1.0f);

    float sqrt_1_minus_u1 = sqrtf(1.0f - u1);
    float sqrt_u1 = sqrtf(u1);

    float pi_2_u2 = 2.0f * M_PI * u2;
    float pi_2_u3 = 2.0f * M_PI * u3;

    Quat q;
    q.w = sqrt_1_minus_u1 * sinf(pi_2_u2);
    q.x = sqrt_1_minus_u1 * cosf(pi_2_u2);
    q.y = sqrt_u1 * sinf(pi_2_u3);
    q.z = sqrt_u1 * cosf(pi_2_u3);

    return q;
}

Ring rndring(void) {
    Ring ring;

    ring.pos.x = rndf(-GRID_SIZE + RING_MARGIN, GRID_SIZE - RING_MARGIN);
    ring.pos.y = rndf(-GRID_SIZE + RING_MARGIN, GRID_SIZE - RING_MARGIN);
    ring.pos.z = rndf(-GRID_SIZE + RING_MARGIN, GRID_SIZE - RING_MARGIN);

    ring.orientation = rndquat();

    Vec3 base_normal = {0.0f, 0.0f, 1.0f};
    ring.normal = quat_rotate(ring.orientation, base_normal);

    ring.radius = RING_RAD;

    return ring;
}

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
    Vec3 trail[TRAIL_LENGTH];
    int trail_index;
    int trail_count;
};

typedef struct Drone Drone;
struct Drone {
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

    Vec3 pos; // global position (x, y, z)
    Vec3 prev_pos;
    Vec3 vel;   // linear velocity (u, v, w)
    Quat quat;  // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega; // angular velocity (p, q, r)

    // Motor dynamics - actual motor RPM with time constants
    float motor_rpm[4]; // actual motor speeds (rad/s)

    Client *client;
};

void init(Drone *env) {
    env->log = (Log){0};
    env->tick = 0;
    // one extra ring for observation (requires current ring, next ring)
    // max_rings and moves_left are initialised in binding.c
    env->ring_buffer = (Ring *)malloc((env->max_rings + 1) * sizeof(Ring));
}

void add_log(Drone *env) {
    env->log.score += env->score;
    env->log.episode_return += env->episodic_return;
    env->log.episode_length += env->tick;
    env->log.perf += (float)env->ring_idx / (float)env->max_rings;
    env->log.n += 1.0f;
}

void compute_observations(Drone *env) {
    Quat q_inv = quat_inverse(env->quat);
    Ring curr_ring = env->ring_buffer[env->ring_idx];
    Ring next_ring = env->ring_buffer[env->ring_idx + 1];

    Vec3 to_curr_ring = quat_rotate(q_inv, sub3(curr_ring.pos, env->pos));
    Vec3 to_next_ring = quat_rotate(q_inv, sub3(next_ring.pos, env->pos));

    Vec3 curr_ring_norm = quat_rotate(q_inv, curr_ring.normal);
    Vec3 next_ring_norm = quat_rotate(q_inv, next_ring.normal);

    Vec3 linear_vel_body = quat_rotate(q_inv, env->vel);
    Vec3 drone_up_world = quat_rotate(env->quat, (Vec3){0.0f, 0.0f, 1.0f});

    env->observations[0] = to_curr_ring.x / GRID_SIZE;
    env->observations[1] = to_curr_ring.y / GRID_SIZE;
    env->observations[2] = to_curr_ring.z / GRID_SIZE;

    env->observations[3] = curr_ring_norm.x;
    env->observations[4] = curr_ring_norm.y;
    env->observations[5] = curr_ring_norm.z;

    env->observations[6] = to_next_ring.x / GRID_SIZE;
    env->observations[7] = to_next_ring.y / GRID_SIZE;
    env->observations[8] = to_next_ring.z / GRID_SIZE;

    env->observations[9] = next_ring_norm.x;
    env->observations[10] = next_ring_norm.y;
    env->observations[11] = next_ring_norm.z;

    env->observations[12] = linear_vel_body.x / MAX_VEL;
    env->observations[13] = linear_vel_body.y / MAX_VEL;
    env->observations[14] = linear_vel_body.z / MAX_VEL;

    env->observations[15] = env->omega.x / MAX_OMEGA;
    env->observations[16] = env->omega.y / MAX_OMEGA;
    env->observations[17] = env->omega.z / MAX_OMEGA;

    env->observations[18] = drone_up_world.x;
    env->observations[19] = drone_up_world.y;
    env->observations[20] = drone_up_world.z;

    env->observations[21] = env->quat.w;
    env->observations[22] = env->quat.x;
    env->observations[23] = env->quat.y;
    env->observations[24] = env->quat.z;
}

void c_reset(Drone *env) {
    env->tick = 0;
    env->score = 0;
    env->episodic_return = 0.0f;

    env->moves_left = env->max_moves;

    env->ring_idx = 0;

    // creates rings at least MARGIN apart
    if (env->max_rings + 1 > 0) {
        env->ring_buffer[0] = rndring();
    }

    for (int i = 1; i < env->max_rings + 1; i++) {
        do {
            env->ring_buffer[i] = rndring();
        } while (norm3(sub3(env->ring_buffer[i].pos, env->ring_buffer[i - 1].pos)) < RING_MARGIN);
    }

    // start drone at least MARGIN away from the first ring
    do {
        env->pos = (Vec3){rndf(-9, 9), rndf(-9, 9), rndf(-9, 9)};
    } while (norm3(sub3(env->pos, env->ring_buffer[0].pos)) < RING_MARGIN);

    env->prev_pos = env->pos;
    env->vel = (Vec3){0.0f, 0.0f, 0.0f};
    env->omega = (Vec3){0.0f, 0.0f, 0.0f};
    env->quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};

    // Initialize motor RPMs to zero (motors start at rest)
    for (int i = 0; i < 4; i++) {
        env->motor_rpm[i] = 0.0f;
    }

    compute_observations(env);
}

void c_step(Drone *env) {
    clamp4(env->actions, -1.0f, 1.0f);

    env->tick++;
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->log.score = 0;

    // Motor dynamics with time constants
    // Convert actions [-1,1] to commanded motor RPM [0, MAX_RPM]
    float motor_rpm_cmd[4];
    for (int i = 0; i < 4; i++) {
        motor_rpm_cmd[i] = (env->actions[i] + 1.0f) * 0.5f * MAX_RPM;
    }
    
    // Update actual motor RPMs using first-order lag: τ * d(ω)/dt = ω_cmd - ω
    // Discrete integration: ω_new = ω_old + (dt/τ) * (ω_cmd - ω_old)
    float motor_gain = DT / MOTOR_TAU;
    for (int i = 0; i < 4; i++) {
        env->motor_rpm[i] += motor_gain * (motor_rpm_cmd[i] - env->motor_rpm[i]);
        // Clamp to physical limits
        env->motor_rpm[i] = clampf(env->motor_rpm[i], 0.0f, MAX_RPM);
    }

    // Calculate motor thrusts from actual RPMs (not commanded)
    float T[4];
    for (int i = 0; i < 4; i++) {
        T[i] = K_THRUST * powf(env->motor_rpm[i], 2.0f);
    }

    // body frame net force
    Vec3 F_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

    // body frame torques
    Vec3 M = {ARM_LEN * (T[1] - T[3]), ARM_LEN * (T[2] - T[0]),
              K_DRAG * (T[0] - T[1] + T[2] - T[3])};

    // applies angular damping to torques
    M.x -= K_ANG_DAMP * env->omega.x;
    M.y -= K_ANG_DAMP * env->omega.y;
    M.z -= K_ANG_DAMP * env->omega.z;

    // body frame force -> world frame force
    Vec3 F_world = quat_rotate(env->quat, F_body);

    // world frame linear drag
    F_world.x -= B_DRAG * env->vel.x;
    F_world.y -= B_DRAG * env->vel.y;
    F_world.z -= B_DRAG * env->vel.z;

    // world frame gravity
    Vec3 accel = {F_world.x / MASS, F_world.y / MASS, (F_world.z / MASS) - GRAVITY};

    // from the definition of q dot
    Quat omega_q = {0.0f, env->omega.x, env->omega.y, env->omega.z};
    Quat q_dot = quat_mul(env->quat, omega_q);

    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // integrations
    env->pos.x += env->vel.x * DT;
    env->pos.y += env->vel.y * DT;
    env->pos.z += env->vel.z * DT;

    env->vel.x += accel.x * DT;
    env->vel.y += accel.y * DT;
    env->vel.z += accel.z * DT;

    env->omega.x += (M.x / IXX) * DT;
    env->omega.y += (M.y / IYY) * DT;
    env->omega.z += (M.z / IZZ) * DT;

    clamp3(&env->vel, -MAX_VEL, MAX_VEL);
    clamp3(&env->omega, -MAX_OMEGA, MAX_OMEGA);

    env->quat.w += q_dot.w * DT;
    env->quat.x += q_dot.x * DT;
    env->quat.y += q_dot.y * DT;
    env->quat.z += q_dot.z * DT;

    quat_normalize(&env->quat);

    // check out of bounds
    bool out_of_bounds = env->pos.x < -GRID_SIZE || env->pos.x > GRID_SIZE ||
                         env->pos.y < -GRID_SIZE || env->pos.y > GRID_SIZE ||
                         env->pos.z < -GRID_SIZE || env->pos.z > GRID_SIZE;

    if (out_of_bounds) {
        env->rewards[0] -= 1;
        env->episodic_return -= 1;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        compute_observations(env);
        return;
    }

    // previous dot product negative if on the 'entry' side of the ring's plane
    float prev_dot = dot3(sub3(env->prev_pos, env->ring_buffer[env->ring_idx].pos),
                          env->ring_buffer[env->ring_idx].normal);

    // new dot product positive if on the 'exit' side of the ring's plane
    float new_dot = dot3(sub3(env->pos, env->ring_buffer[env->ring_idx].pos),
                         env->ring_buffer[env->ring_idx].normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    // if we have crossed the plane of the ring
    if (valid_dir || invalid_dir) {
        // find intesection with ring's plane
        Vec3 dir = sub3(env->pos, env->prev_pos);
        float t = -prev_dot / dot3(env->ring_buffer[env->ring_idx].normal,
                                   dir); // possible nan

        Vec3 intersection = add3(env->prev_pos, scalmul3(dir, t));
        float dist = norm3(sub3(intersection, env->ring_buffer[env->ring_idx].pos));

        // reward or terminate based on distance to ring center
        if (dist < (env->ring_buffer[env->ring_idx].radius - 0.5) && valid_dir) {
            env->rewards[0] += 1;
            env->episodic_return += 1;
            env->score++;
            env->ring_idx++;
        } else if (dist < env->ring_buffer[env->ring_idx].radius + 0.5) {
            env->rewards[0] -= 1;
            env->episodic_return -= 1;
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
            return;
        }
    }

    // truncate
    env->moves_left -= 1;
    if (env->moves_left == 0 || env->ring_idx == env->max_rings) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    env->prev_pos = env->pos;

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(Drone *env) {
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

Client *make_client(Drone *env) {
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
    client->trail_index = 0;
    client->trail_count = 0;
    for (int i = 0; i < TRAIL_LENGTH; i++) {
        client->trail[i] = env->pos;
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

void c_render(Drone *env) {
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
    client->trail[client->trail_index] = env->pos;
    client->trail_index = (client->trail_index + 1) % TRAIL_LENGTH;
    if (client->trail_count < TRAIL_LENGTH)
        client->trail_count++;

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    BeginMode3D(client->camera);

    // draws bounding cube
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_SIZE * 2.0f, GRID_SIZE * 2.0f, GRID_SIZE * 2.0f,
                  WHITE);

    // draws drone body
    DrawSphere((Vector3){env->pos.x, env->pos.y, env->pos.z}, 0.3f, RED);

    // draws rotors according to thrust
    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = (env->actions[i] + 1.0f) * 0.5f * MAX_RPM;
        T[i] = K_THRUST * rpm * rpm;
    }

    const float rotor_radius = 0.15f;
    const float visual_arm_len = ARM_LEN * 4.0f;

    Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                  {-visual_arm_len, 0.0f, 0.0f},
                                  {0.0f, +visual_arm_len, 0.0f},
                                  {0.0f, -visual_arm_len, 0.0f}};

    Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};

    for (int i = 0; i < 4; i++) {
        Vec3 world_off = quat_rotate(env->quat, rotor_offsets_body[i]);

        Vector3 rotor_pos = {env->pos.x + world_off.x, env->pos.y + world_off.y,
                             env->pos.z + world_off.z};

        float rpm = (env->actions[i] + 1.0f) * 0.5f * MAX_RPM;
        float intensity = 0.75f + 0.25f * (rpm / MAX_RPM);

        Color rotor_color = (Color){(unsigned char)(base_colors[i].r * intensity),
                                    (unsigned char)(base_colors[i].g * intensity),
                                    (unsigned char)(base_colors[i].b * intensity), 255};

        DrawSphere(rotor_pos, rotor_radius, rotor_color);

        DrawCylinderEx((Vector3){env->pos.x, env->pos.y, env->pos.z}, rotor_pos, 0.02f, 0.02f, 8,
                       BLACK);
    }

    // draws line with direction and magnitude of velocity / 10
    if (norm3(env->vel) > 0.1f) {
        DrawLine3D((Vector3){env->pos.x, env->pos.y, env->pos.z},
                   (Vector3){env->pos.x + env->vel.x * 0.1f, env->pos.y + env->vel.y * 0.1f,
                             env->pos.z + env->vel.z * 0.1f},
                   MAGENTA);
    }

    // Draw trailing path
    for (int i = 1; i < client->trail_count; i++) {
        int idx0 = (client->trail_index + i) % TRAIL_LENGTH;
        int idx1 = (client->trail_index + i - 1) % TRAIL_LENGTH;
        float alpha = (float)i / client->trail_count * 0.8f; // fade out
        Color trail_color = ColorAlpha(YELLOW, alpha);
        DrawLine3D((Vector3){client->trail[idx0].x, client->trail[idx0].y, client->trail[idx0].z},
                   (Vector3){client->trail[idx1].x, client->trail[idx1].y, client->trail[idx1].z},
                   trail_color);
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

    DrawText(TextFormat("Pos: (%.1f, %.1f, %.1f)", env->pos.x, env->pos.y, env->pos.z), 10, 225, 18,
             WHITE);
    DrawText(TextFormat("Vel: %.2f m/s", norm3(env->vel)), 10, 245, 18, WHITE);

    DrawText("Left click + drag: Rotate camera", 10, 275, 16, LIGHTGRAY);
    DrawText("Mouse wheel: Zoom in/out", 10, 295, 16, LIGHTGRAY);

    EndDrawing();
}
