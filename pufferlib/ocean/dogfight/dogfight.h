#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define DT 0.02f
#ifndef PI
#define PI 3.14159265358979f
#endif
#define WORLD_HALF_X 2000.0f
#define WORLD_HALF_Y 2000.0f
#define WORLD_MAX_Z 3000.0f
#define MAX_SPEED 250.0f
#define OBS_SIZE 19  // player(13) + rel_pos(3) + rel_vel(3)

#define MASS 3000.0f           // kg (WW2 fighter ~2500-4000)
#define WING_AREA 22.0f        // m² (P-51: 21.6, Spitfire: 22.5)
#define C_D0 0.02f             // parasitic drag coefficient
#define K 0.05f                // induced drag factor (1/(π*e*AR))
#define C_L_MAX 1.4f           // max lift coefficient (stall)
#define C_L_ALPHA 5.7f         // lift curve slope (per radian)
#define ENGINE_POWER 1000000.0f // watts (~1340 hp)
#define ETA_PROP 0.8f          // propeller efficiency
#define GRAVITY 9.81f          // m/s²
#define G_LIMIT 8.0f           // structural g limit
#define RHO 1.225f             // air density kg/m³ (sea level)

#define MAX_PITCH_RATE 2.5f    // rad/s
#define MAX_ROLL_RATE 3.0f     // rad/s
#define MAX_YAW_RATE 1.5f      // rad/s

typedef struct { float x, y, z; } Vec3;
typedef struct { float w, x, y, z; } Quat;

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float rndf(float a, float b) {
    return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline Vec3 vec3(float x, float y, float z) { return (Vec3){x, y, z}; }
static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 mul3(Vec3 a, float s) { return (Vec3){a.x * s, a.y * s, a.z * s}; }
static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline Quat quat(float w, float x, float y, float z) { return (Quat){w, x, y, z}; }

static inline Quat quat_mul(Quat a, Quat b) {
    return (Quat){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline void quat_normalize(Quat* q) {
    float n = sqrtf(q->w*q->w + q->x*q->x + q->y*q->y + q->z*q->z);
    if (n > 1e-8f) {
        float inv = 1.0f / n;
        q->w *= inv; q->x *= inv; q->y *= inv; q->z *= inv;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = {0.0f, v.x, v.y, v.z};
    Quat q_conj = {q.w, -q.x, -q.y, -q.z};
    Quat tmp = quat_mul(q, qv);
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float half = angle * 0.5f;
    float s = sinf(half);
    return (Quat){cosf(half), axis.x * s, axis.y * s, axis.z * s};
}

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Quat ori;
    float throttle;
} Plane;

typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float kills;
    float deaths;
    float shots_fired;
    float shots_hit;
    float n;
} Log;

typedef struct Client {
    Camera3D camera;
    float width;
    float height;
    // Camera orbit state (for mouse control)
    float cam_distance;
    float cam_azimuth;
    float cam_elevation;
    bool is_dragging;
    float last_mouse_x;
    float last_mouse_y;
} Client;

typedef struct Dogfight {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    Log log;
    Client *client;
    int tick;
    int max_steps;
    float episode_return;
    Plane player;
    Plane opponent;
} Dogfight;

void init(Dogfight *env) {
    env->log = (Log){0};
    env->tick = 0;
    env->episode_return = 0.0f;
    env->client = NULL;
}

void add_log(Dogfight *env) {
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.n += 1.0f;
}

void reset_plane(Plane *p, Vec3 pos, Vec3 vel) {
    p->pos = pos;
    p->vel = vel;
    p->ori = quat(1, 0, 0, 0);
    p->throttle = 0.5f;
}

void compute_observations(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    int i = 0;
    env->observations[i++] = p->pos.x / WORLD_HALF_X;
    env->observations[i++] = p->pos.y / WORLD_HALF_Y;
    env->observations[i++] = p->pos.z / WORLD_MAX_Z;
    env->observations[i++] = p->vel.x / MAX_SPEED;
    env->observations[i++] = p->vel.y / MAX_SPEED;
    env->observations[i++] = p->vel.z / MAX_SPEED;
    env->observations[i++] = p->ori.w;
    env->observations[i++] = p->ori.x;
    env->observations[i++] = p->ori.y;
    env->observations[i++] = p->ori.z;
    env->observations[i++] = up.x;
    env->observations[i++] = up.y;
    env->observations[i++] = up.z;

    // Relative position to opponent (in world frame for now)
    Vec3 rel_pos = sub3(o->pos, p->pos);
    env->observations[i++] = rel_pos.x / WORLD_HALF_X;
    env->observations[i++] = rel_pos.y / WORLD_HALF_Y;
    env->observations[i++] = rel_pos.z / WORLD_MAX_Z;

    // Relative velocity
    Vec3 rel_vel = sub3(o->vel, p->vel);
    env->observations[i++] = rel_vel.x / MAX_SPEED;
    env->observations[i++] = rel_vel.y / MAX_SPEED;
    env->observations[i++] = rel_vel.z / MAX_SPEED;
}

void c_reset(Dogfight *env) {
    env->tick = 0;
    env->episode_return = 0.0f;

    Vec3 pos = vec3(rndf(-500, 500), rndf(-500, 500), rndf(500, 1500));
    Vec3 vel = vec3(80, 0, 0);
    reset_plane(&env->player, pos, vel);

    // Spawn opponent ahead of player
    Vec3 opp_pos = vec3(
        pos.x + rndf(200, 500),
        pos.y + rndf(-100, 100),
        pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, vel);

    compute_observations(env);
}

static inline Vec3 normalize3(Vec3 v) {
    float n = norm3(v);
    if (n < 1e-8f) return vec3(0, 0, 0);
    return mul3(v, 1.0f / n);
}

static inline Vec3 cross3(Vec3 a, Vec3 b) {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

void step_plane_with_physics(Plane *p, float *actions, float dt) {
    // Body frame axes
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 right = quat_rotate(p->ori, vec3(0, 1, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Map actions to control rates
    float throttle = (actions[0] + 1.0f) * 0.5f;  // [0, 1]
    float pitch_rate = actions[1] * MAX_PITCH_RATE;
    float roll_rate = actions[2] * MAX_ROLL_RATE;
    float yaw_rate = actions[3] * MAX_YAW_RATE;

    // Integrate orientation: q_dot = 0.5 * q * omega_quat
    Vec3 omega_body = vec3(roll_rate, pitch_rate, yaw_rate);
    Quat omega_quat = quat(0, omega_body.x, omega_body.y, omega_body.z);
    Quat q_dot = quat_mul(p->ori, omega_quat);
    p->ori.w += 0.5f * q_dot.w * dt;
    p->ori.x += 0.5f * q_dot.x * dt;
    p->ori.y += 0.5f * q_dot.y * dt;
    p->ori.z += 0.5f * q_dot.z * dt;
    quat_normalize(&p->ori);

    // Velocity magnitude
    float V = norm3(p->vel);
    if (V < 1.0f) V = 1.0f;

    // Angle of attack: angle between velocity and body forward
    Vec3 vel_norm = normalize3(p->vel);
    float cos_alpha = dot3(vel_norm, forward);
    cos_alpha = clampf(cos_alpha, -1.0f, 1.0f);
    float alpha = acosf(cos_alpha);
    // Signed alpha: positive when nose up relative to velocity
    float sign_alpha = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
    alpha *= sign_alpha;

    // Lift coefficient (clamped for stall)
    float C_L = C_L_ALPHA * alpha;
    C_L = clampf(C_L, -C_L_MAX, C_L_MAX);

    // Dynamic pressure: q = 0.5 * rho * V²
    float q_dyn = 0.5f * RHO * V * V;

    // Lift magnitude: L = C_L * q * S
    float L_mag = C_L * q_dyn * WING_AREA;

    // Drag coefficient and magnitude: D = (C_D0 + K * C_L²) * q * S
    float C_D = C_D0 + K * C_L * C_L;
    float D_mag = C_D * q_dyn * WING_AREA;

    // Thrust (velocity-dependent propeller)
    float P_avail = ENGINE_POWER * throttle;
    float T_dynamic = (P_avail * ETA_PROP) / V;
    float T_static = 0.3f * P_avail;  // static thrust factor
    float T_mag = fminf(T_static, T_dynamic);

    // Force directions (world frame)
    Vec3 drag_dir = mul3(vel_norm, -1.0f);  // opposite to velocity
    Vec3 thrust_dir = forward;               // along body forward

    // Lift direction: perpendicular to velocity, in the plane of velocity and up
    Vec3 lift_dir = cross3(vel_norm, right);
    float lift_dir_mag = norm3(lift_dir);
    if (lift_dir_mag > 0.01f) {
        lift_dir = mul3(lift_dir, 1.0f / lift_dir_mag);
    } else {
        lift_dir = up;
    }

    // Weight (always down in world frame)
    Vec3 weight = vec3(0, 0, -MASS * GRAVITY);

    // Sum forces
    Vec3 F_thrust = mul3(thrust_dir, T_mag);
    Vec3 F_lift = mul3(lift_dir, L_mag);
    Vec3 F_drag = mul3(drag_dir, D_mag);
    Vec3 F_total = add3(add3(add3(F_thrust, F_lift), F_drag), weight);

    // G-limit: clamp acceleration
    Vec3 accel = mul3(F_total, 1.0f / MASS);
    float accel_mag = norm3(accel);
    float max_accel = G_LIMIT * GRAVITY;
    if (accel_mag > max_accel) {
        accel = mul3(accel, max_accel / accel_mag);
    }

    // Integrate velocity and position
    p->vel = add3(p->vel, mul3(accel, dt));
    p->pos = add3(p->pos, mul3(p->vel, dt));

    // Store throttle
    p->throttle = throttle;
}

void step_plane(Plane *p, float dt) {
    // Simple forward motion for opponent (no actions)
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    float speed = norm3(p->vel);
    if (speed < 1.0f) speed = 80.0f;
    p->vel = mul3(forward, speed);
    p->pos = add3(p->pos, mul3(p->vel, dt));
}

void c_step(Dogfight *env) {
    env->tick++;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    // Player uses full physics with actions
    step_plane_with_physics(&env->player, env->actions, DT);

    // Opponent uses simple motion (no actions)
    step_plane(&env->opponent, DT);

    // === Reward Shaping (Phase 3.5) ===
    float reward = 0.0f;
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    // 1. Base pursuit reward: closer = better
    Vec3 rel_pos = sub3(o->pos, p->pos);
    float dist = norm3(rel_pos);
    reward += -dist / 10000.0f;

    // 2. Closing velocity reward: approaching = good
    Vec3 rel_vel = sub3(p->vel, o->vel);  // player vel relative to opponent
    Vec3 rel_pos_norm = normalize3(rel_pos);
    float closing_rate = dot3(rel_vel, rel_pos_norm);  // positive when closing
    reward += closing_rate / 500.0f;  // scale: 100 m/s closing = +0.2

    // 3. Tail position reward: behind opponent = good
    Vec3 opp_forward = quat_rotate(o->ori, vec3(1, 0, 0));
    float tail_angle = dot3(rel_pos_norm, opp_forward);  // +1 when behind, -1 when in front
    reward += tail_angle * 0.02f;  // scale: behind = +0.02, in front = -0.02

    // 4. Altitude penalty: too low or too high is bad
    if (p->pos.z < 200.0f) {
        reward -= (200.0f - p->pos.z) / 2000.0f;  // max -0.1 at z=0
    } else if (p->pos.z > 2500.0f) {
        reward -= (p->pos.z - 2500.0f) / 5000.0f;  // max -0.1 at z=3000
    }

    // 5. Speed penalty: too slow is stall risk
    float speed = norm3(p->vel);
    if (speed < 50.0f) {
        reward -= (50.0f - speed) / 500.0f;  // max -0.1 at speed=0
    }

    env->rewards[0] = reward;
    env->episode_return += reward;

    // Check bounds (player only)
    bool oob = fabsf(p->pos.x) > WORLD_HALF_X ||
               fabsf(p->pos.y) > WORLD_HALF_Y ||
               p->pos.z < 0 || p->pos.z > WORLD_MAX_Z;

    if (oob || env->tick >= env->max_steps) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
}

// Forward declaration for c_close (used in c_render)
void c_close(Dogfight *env);

// Draw airplane shape using lines - shows roll/pitch/yaw clearly
// Body frame: X=forward, Y=right, Z=up
void draw_plane_shape(Vec3 pos, Quat ori, Color body_color, Color wing_color) {
    // Body frame points (scaled for visibility: ~20m wingspan, ~25m length)
    Vec3 nose = vec3(15, 0, 0);
    Vec3 tail = vec3(-10, 0, 0);
    Vec3 left_wing = vec3(0, -12, 0);
    Vec3 right_wing = vec3(0, 12, 0);
    Vec3 vtail_top = vec3(-8, 0, 8);       // Vertical stabilizer
    Vec3 htail_left = vec3(-10, -5, 0);    // Horizontal stabilizer
    Vec3 htail_right = vec3(-10, 5, 0);

    // Rotate all points by orientation and translate to world position
    Vec3 nose_w = add3(pos, quat_rotate(ori, nose));
    Vec3 tail_w = add3(pos, quat_rotate(ori, tail));
    Vec3 lwing_w = add3(pos, quat_rotate(ori, left_wing));
    Vec3 rwing_w = add3(pos, quat_rotate(ori, right_wing));
    Vec3 vtop_w = add3(pos, quat_rotate(ori, vtail_top));
    Vec3 htl_w = add3(pos, quat_rotate(ori, htail_left));
    Vec3 htr_w = add3(pos, quat_rotate(ori, htail_right));

    // Convert to Raylib Vector3
    Vector3 nose_r = {nose_w.x, nose_w.y, nose_w.z};
    Vector3 tail_r = {tail_w.x, tail_w.y, tail_w.z};
    Vector3 lwing_r = {lwing_w.x, lwing_w.y, lwing_w.z};
    Vector3 rwing_r = {rwing_w.x, rwing_w.y, rwing_w.z};
    Vector3 vtop_r = {vtop_w.x, vtop_w.y, vtop_w.z};
    Vector3 htl_r = {htl_w.x, htl_w.y, htl_w.z};
    Vector3 htr_r = {htr_w.x, htr_w.y, htr_w.z};

    // Fuselage (nose to tail)
    DrawLine3D(nose_r, tail_r, body_color);

    // Main wings (left to right, through center for visibility)
    DrawLine3D(lwing_r, rwing_r, wing_color);
    // Wing to fuselage connections (makes it look more solid)
    DrawLine3D(lwing_r, nose_r, wing_color);
    DrawLine3D(rwing_r, nose_r, wing_color);

    // Vertical stabilizer (tail to top)
    DrawLine3D(tail_r, vtop_r, body_color);

    // Horizontal stabilizer
    DrawLine3D(htl_r, htr_r, body_color);
    DrawLine3D(htl_r, tail_r, body_color);
    DrawLine3D(htr_r, tail_r, body_color);

    // Small sphere at nose to show front clearly
    DrawSphere(nose_r, 2.0f, body_color);
}

void handle_camera_controls(Client *c) {
    Vector2 mouse = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        c->is_dragging = true;
        c->last_mouse_x = mouse.x;
        c->last_mouse_y = mouse.y;
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        c->is_dragging = false;
    }

    if (c->is_dragging) {
        float sensitivity = 0.005f;
        c->cam_azimuth -= (mouse.x - c->last_mouse_x) * sensitivity;
        c->cam_elevation += (mouse.y - c->last_mouse_y) * sensitivity;
        c->cam_elevation = clampf(c->cam_elevation, -1.4f, 1.4f);  // prevent gimbal lock
        c->last_mouse_x = mouse.x;
        c->last_mouse_y = mouse.y;
    }

    // Mouse wheel zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        c->cam_distance = clampf(c->cam_distance - wheel * 10.0f, 30.0f, 300.0f);
    }
}

void c_render(Dogfight *env) {
    // 1. Lazy initialization
    if (env->client == NULL) {
        env->client = (Client *)calloc(1, sizeof(Client));
        env->client->width = 1280;
        env->client->height = 720;
        env->client->cam_distance = 80.0f;
        env->client->cam_azimuth = 0.0f;
        env->client->cam_elevation = 0.3f;
        env->client->is_dragging = false;

        InitWindow(1280, 720, "Dogfight");
        SetTargetFPS(60);

        // Z-up coordinate system
        env->client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        env->client->camera.fovy = 45.0f;
        env->client->camera.projection = CAMERA_PERSPECTIVE;
    }

    // 2. Handle window close
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    // 3. Handle mouse controls for camera orbit
    handle_camera_controls(env->client);

    // 4. Update chase camera
    Plane *p = &env->player;
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float dist = env->client->cam_distance;

    // Apply orbit offsets from mouse drag
    float az = env->client->cam_azimuth;
    float el = env->client->cam_elevation;

    // Base chase position (behind and above player)
    float cam_x = p->pos.x - fwd.x * dist * cosf(el) * cosf(az) + fwd.y * dist * sinf(az);
    float cam_y = p->pos.y - fwd.y * dist * cosf(el) * cosf(az) - fwd.x * dist * sinf(az);
    float cam_z = p->pos.z + dist * sinf(el) + 20.0f;

    env->client->camera.position = (Vector3){cam_x, cam_y, cam_z};
    env->client->camera.target = (Vector3){p->pos.x, p->pos.y, p->pos.z};

    // 5. Begin drawing
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});  // Dark blue-green sky

    BeginMode3D(env->client->camera);

    // 6. Draw ground plane at z=0
    DrawPlane((Vector3){0, 0, 0}, (Vector2){4000, 4000}, (Color){20, 60, 20, 255});

    // 7. Draw world bounds wireframe
    // Bounds: X ±2000, Y ±2000, Z 0-3000 → center at (0, 0, 1500)
    DrawCubeWires((Vector3){0, 0, 1500}, 4000, 4000, 3000, (Color){100, 100, 100, 255});

    // 8. Draw player plane (green wireframe airplane)
    draw_plane_shape(p->pos, p->ori, GREEN, LIME);

    // 9. Draw opponent plane (red wireframe airplane)
    Plane *o = &env->opponent;
    draw_plane_shape(o->pos, o->ori, RED, ORANGE);

    EndMode3D();

    // 10. Draw HUD
    float speed = norm3(p->vel);
    float dist_to_opp = norm3(sub3(o->pos, p->pos));

    DrawText(TextFormat("Speed: %.0f m/s", speed), 10, 10, 20, WHITE);
    DrawText(TextFormat("Altitude: %.0f m", p->pos.z), 10, 40, 20, WHITE);
    DrawText(TextFormat("Throttle: %.0f%%", p->throttle * 100.0f), 10, 70, 20, WHITE);
    DrawText(TextFormat("Distance: %.0f m", dist_to_opp), 10, 100, 20, WHITE);
    DrawText(TextFormat("Tick: %d / %d", env->tick, env->max_steps), 10, 130, 20, WHITE);
    DrawText(TextFormat("Return: %.2f", env->episode_return), 10, 160, 20, WHITE);

    // Controls hint
    DrawText("Mouse drag: Orbit | Scroll: Zoom | ESC: Exit", 10, (int)env->client->height - 30, 16, GRAY);

    EndDrawing();
}

void c_close(Dogfight *env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}
