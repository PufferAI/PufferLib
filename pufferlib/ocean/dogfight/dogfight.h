#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#define DEBUG 0

#define DT 0.02f
#ifndef PI
#define PI 3.14159265358979f
#endif
#define WORLD_HALF_X 2000.0f
#define WORLD_HALF_Y 2000.0f
#define WORLD_MAX_Z 3000.0f
#define MAX_SPEED 250.0f
#define OBS_SIZE 19  // player(13) + rel_pos(3) + rel_vel(3)

// ============================================================================
// AIRCRAFT PARAMETERS
// ============================================================================
// These define a WW2-era fighter aircraft (similar to P-51 Mustang / Spitfire)
//
// THEORETICAL PERFORMANCE (derived from these constants):
//   Max speed (level):  V_max = (P*eta / (0.5*rho*S*Cd0))^(1/3) ≈ 143.7 m/s
//   Stall speed:        V_stall = sqrt(2*m*g / (rho*S*Cl_max)) ≈ 39.5 m/s
//   Min sink speed:     V_minsink ≈ 1.32 * V_stall ≈ 52 m/s
//
// WING INCIDENCE:
//   The wing is mounted at +2° relative to the fuselage reference line.
//   This means at zero body AOA, the wing still generates lift (Cl ≈ 0.2).
//   Level cruise at ~100 m/s requires Cl ≈ 0.22, so nearly hands-off flight.
//
// DRAG POLAR: Cd = Cd0 + K * Cl²
//   - Cd0: parasitic/zero-lift drag (skin friction, form drag)
//   - K: induced drag factor = 1/(π * e * AR) where e≈0.8, AR≈wing²/S
// ============================================================================
#define MASS 3000.0f           // kg (WW2 fighter ~2500-4000)
#define WING_AREA 22.0f        // m² (P-51: 21.6, Spitfire: 22.5)
#define C_D0 0.02f             // parasitic drag coefficient (clean config)
#define K 0.05f                // induced drag factor: 1/(π*e*AR), e≈0.8, AR≈8
#define C_L_MAX 1.4f           // max lift coefficient before stall
#define C_L_ALPHA 5.7f         // lift curve slope dCl/dα (per radian), ≈2π for thin airfoil
#define WING_INCIDENCE 0.035f  // wing incidence angle (rad), ~2° (P-51: 2.5°, Spitfire: 2°)
                               // This is the angle between wing chord and fuselage reference.
                               // When fuselage is level (α_body=0), wing sees this AOA.
#define ENGINE_POWER 1000000.0f // watts (~1340 hp, Merlin engine class)
#define ETA_PROP 0.8f          // propeller efficiency (typical 0.7-0.85)
#define GRAVITY 9.81f          // m/s²
#define G_LIMIT 8.0f           // structural g limit (aerobatic category)
#define RHO 1.225f             // air density kg/m³ (sea level ISA)

#define MAX_PITCH_RATE 2.5f    // rad/s
#define MAX_ROLL_RATE 3.0f     // rad/s
#define MAX_YAW_RATE 1.5f      // rad/s

// Combat constants
#define GUN_RANGE 500.0f       // meters
#define GUN_CONE_ANGLE 0.087f  // ~5 degrees in radians
#define FIRE_COOLDOWN 10       // ticks (0.2 seconds at 50Hz)

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
    int fire_cooldown;  // Ticks until can fire again (0 = ready)
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
    p->fire_cooldown = 0;
}

void compute_observations(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_vel = sub3(o->vel, p->vel);

    if (DEBUG) printf("=== OBS tick=%d ===\n", env->tick);

    int i = 0;
    if (DEBUG) printf("pos_x_norm=%.3f (raw=%.1f)\n", p->pos.x / WORLD_HALF_X, p->pos.x);
    env->observations[i++] = p->pos.x / WORLD_HALF_X;
    if (DEBUG) printf("pos_y_norm=%.3f (raw=%.1f)\n", p->pos.y / WORLD_HALF_Y, p->pos.y);
    env->observations[i++] = p->pos.y / WORLD_HALF_Y;
    if (DEBUG) printf("pos_z_norm=%.3f (raw=%.1f)\n", p->pos.z / WORLD_MAX_Z, p->pos.z);
    env->observations[i++] = p->pos.z / WORLD_MAX_Z;
    if (DEBUG) printf("vel_x_norm=%.3f (raw=%.1f)\n", p->vel.x / MAX_SPEED, p->vel.x);
    env->observations[i++] = p->vel.x / MAX_SPEED;
    if (DEBUG) printf("vel_y_norm=%.3f (raw=%.1f)\n", p->vel.y / MAX_SPEED, p->vel.y);
    env->observations[i++] = p->vel.y / MAX_SPEED;
    if (DEBUG) printf("vel_z_norm=%.3f (raw=%.1f)\n", p->vel.z / MAX_SPEED, p->vel.z);
    env->observations[i++] = p->vel.z / MAX_SPEED;
    if (DEBUG) printf("ori_w=%.3f\n", p->ori.w);
    env->observations[i++] = p->ori.w;
    if (DEBUG) printf("ori_x=%.3f\n", p->ori.x);
    env->observations[i++] = p->ori.x;
    if (DEBUG) printf("ori_y=%.3f\n", p->ori.y);
    env->observations[i++] = p->ori.y;
    if (DEBUG) printf("ori_z=%.3f\n", p->ori.z);
    env->observations[i++] = p->ori.z;
    if (DEBUG) printf("up_x=%.3f\n", up.x);
    env->observations[i++] = up.x;
    if (DEBUG) printf("up_y=%.3f\n", up.y);
    env->observations[i++] = up.y;
    if (DEBUG) printf("up_z=%.3f\n", up.z);
    env->observations[i++] = up.z;
    if (DEBUG) printf("rel_pos_x_norm=%.3f (raw=%.1f)\n", rel_pos.x / WORLD_HALF_X, rel_pos.x);
    env->observations[i++] = rel_pos.x / WORLD_HALF_X;
    if (DEBUG) printf("rel_pos_y_norm=%.3f (raw=%.1f)\n", rel_pos.y / WORLD_HALF_Y, rel_pos.y);
    env->observations[i++] = rel_pos.y / WORLD_HALF_Y;
    if (DEBUG) printf("rel_pos_z_norm=%.3f (raw=%.1f)\n", rel_pos.z / WORLD_MAX_Z, rel_pos.z);
    env->observations[i++] = rel_pos.z / WORLD_MAX_Z;
    if (DEBUG) printf("rel_vel_x_norm=%.3f (raw=%.1f)\n", rel_vel.x / MAX_SPEED, rel_vel.x);
    env->observations[i++] = rel_vel.x / MAX_SPEED;
    if (DEBUG) printf("rel_vel_y_norm=%.3f (raw=%.1f)\n", rel_vel.y / MAX_SPEED, rel_vel.y);
    env->observations[i++] = rel_vel.y / MAX_SPEED;
    if (DEBUG) printf("rel_vel_z_norm=%.3f (raw=%.1f)\n", rel_vel.z / MAX_SPEED, rel_vel.z);
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

    if (DEBUG) printf("=== RESET ===\n");
    if (DEBUG) printf("player_pos=(%.1f, %.1f, %.1f)\n", pos.x, pos.y, pos.z);
    if (DEBUG) printf("player_vel=(%.1f, %.1f, %.1f) speed=%.1f\n", vel.x, vel.y, vel.z, norm3(vel));
    if (DEBUG) printf("opponent_pos=(%.1f, %.1f, %.1f)\n", opp_pos.x, opp_pos.y, opp_pos.z);
    if (DEBUG) printf("initial_dist=%.1f m\n", norm3(sub3(opp_pos, pos)));

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

// ============================================================================
// PHYSICS MODEL - step_plane_with_physics()
// ============================================================================
// This implements a simplified 6-DOF flight model with:
//   - Rate-based attitude control (not position control)
//   - Point-mass aerodynamics (no moments/stability derivatives)
//   - Propeller thrust model (T = P*eta/V, capped at static thrust)
//   - Drag polar: Cd = Cd0 + K*Cl²
//   - Wing incidence angle (built-in AOA for near-level cruise)
//
// COORDINATE SYSTEM:
//   World frame: X=East, Y=North, Z=Up (right-handed, Z-up)
//   Body frame:  X=Forward (nose), Y=Right (wing), Z=Up (canopy)
//
// WING INCIDENCE:
//   The wing is mounted at WING_INCIDENCE (~2°) relative to fuselage.
//   Effective AOA for lift = body_alpha + WING_INCIDENCE
//   This allows near-level flight at cruise speed with zero pitch input.
//
// REMAINING LIMITATIONS:
//   - No pitching moment / static stability (Cm_alpha)
//   - Rate-based controls (not position-based)
//   - Symmetric stall model (real stall is asymmetric)
// ============================================================================
void step_plane_with_physics(Plane *p, float *actions, float dt) {
    // ========================================================================
    // 1. BODY FRAME AXES (transform from body to world coordinates)
    // ========================================================================
    // These are the aircraft's body axes expressed in world coordinates
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));  // Nose direction
    Vec3 right = quat_rotate(p->ori, vec3(0, 1, 0));    // Right wing direction
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));       // Canopy direction

    // ========================================================================
    // 2. CONTROL INPUTS → ANGULAR RATES
    // ========================================================================
    // Actions are [-1, 1], mapped to physical rates
    // NOTE: These are RATE commands, not POSITION commands!
    // Holding elevator=0.5 doesn't hold 50% pitch - it pitches UP continuously
    float throttle = (actions[0] + 1.0f) * 0.5f;  // [-1,1] → [0,1]
    float pitch_rate = actions[1] * MAX_PITCH_RATE;  // rad/s, + = nose up
    float roll_rate = actions[2] * MAX_ROLL_RATE;    // rad/s, + = roll right
    float yaw_rate = actions[3] * MAX_YAW_RATE;      // rad/s, + = nose right

    // ========================================================================
    // 3. ATTITUDE INTEGRATION (Quaternion kinematics)
    // ========================================================================
    // q_dot = 0.5 * q ⊗ ω  where ω is angular velocity in body frame
    // This is the standard quaternion derivative formula
    Vec3 omega_body = vec3(roll_rate, pitch_rate, yaw_rate);  // body-frame ω
    Quat omega_quat = quat(0, omega_body.x, omega_body.y, omega_body.z);
    Quat q_dot = quat_mul(p->ori, omega_quat);
    p->ori.w += 0.5f * q_dot.w * dt;
    p->ori.x += 0.5f * q_dot.x * dt;
    p->ori.y += 0.5f * q_dot.y * dt;
    p->ori.z += 0.5f * q_dot.z * dt;
    quat_normalize(&p->ori);  // Prevent drift from numerical integration

    // ========================================================================
    // 4. ANGLE OF ATTACK (AOA, α)
    // ========================================================================
    // AOA = angle between velocity vector and body X-axis (nose)
    // Positive α = nose above flight path = generating positive lift
    //
    // SIGN CONVENTION:
    //   If velocity has component opposite to body Z (up), nose is above
    //   flight path, so α is positive.
    float V = norm3(p->vel);
    if (V < 1.0f) V = 1.0f;  // Prevent division by zero

    Vec3 vel_norm = normalize3(p->vel);
    float cos_alpha = dot3(vel_norm, forward);
    cos_alpha = clampf(cos_alpha, -1.0f, 1.0f);
    float alpha = acosf(cos_alpha);  // Always positive [0, π]

    // Determine sign: positive when nose is ABOVE velocity vector
    // If vel·up < 0, velocity is "below" the body frame → nose above → α > 0
    float sign_alpha = (dot3(p->vel, up) < 0) ? 1.0f : -1.0f;
    alpha *= sign_alpha;

    // ========================================================================
    // 5. LIFT COEFFICIENT (Linear + Stall Clamp)
    // ========================================================================
    // The wing is mounted at an incidence angle relative to the fuselage.
    // Effective AOA for lift = body AOA + wing incidence
    // This means when body is level (α=0), wing still generates lift.
    //
    // Cl = Cl_α * α_effective  (linear region)
    // Real airfoils stall around 12-15° (α ≈ 0.2-0.26 rad)
    // Cl_max = 1.4 occurs at α_eff = 1.4/5.7 ≈ 0.245 rad ≈ 14°
    float alpha_effective = alpha + WING_INCIDENCE;
    float C_L = C_L_ALPHA * alpha_effective;
    C_L = clampf(C_L, -C_L_MAX, C_L_MAX);  // Stall limiting (symmetric)

    // ========================================================================
    // 6. DYNAMIC PRESSURE
    // ========================================================================
    // q = ½ρV² [Pa or N/m²]
    // This is the "pressure" available for aerodynamic forces
    // At 100 m/s: q = 0.5 * 1.225 * 10000 = 6,125 Pa
    float q_dyn = 0.5f * RHO * V * V;

    // ========================================================================
    // 7. LIFT FORCE
    // ========================================================================
    // L = Cl * q * S  [Newtons]
    // For level flight: L = W = m*g = 29,430 N
    // Required Cl at 100 m/s: Cl = 29430 / (6125 * 22) = 0.218
    // Required α = 0.218 / 5.7 = 0.038 rad ≈ 2.2°
    float L_mag = C_L * q_dyn * WING_AREA;

    // ========================================================================
    // 8. DRAG FORCE (Drag Polar)
    // ========================================================================
    // Cd = Cd0 + K * Cl²
    //   Cd0 = parasitic drag (skin friction + form drag)
    //   K*Cl² = induced drag (vortex drag from lift generation)
    //
    // At cruise (Cl=0.22): Cd = 0.02 + 0.05*0.048 = 0.0224
    // At Cl_max (Cl=1.4):  Cd = 0.02 + 0.05*1.96 = 0.118
    float C_D = C_D0 + K * C_L * C_L;
    float D_mag = C_D * q_dyn * WING_AREA;

    // ========================================================================
    // 9. THRUST FORCE (Propeller Model)
    // ========================================================================
    // Power-based: P = T * V  →  T = P * η / V
    // At low speed, thrust is limited by static thrust capability
    //
    // At V=80 m/s, full throttle: T = 800,000 / 80 = 10,000 N
    // At V=143 m/s (max speed):   T = 800,000 / 143 = 5,594 N ≈ D
    float P_avail = ENGINE_POWER * throttle;
    float T_dynamic = (P_avail * ETA_PROP) / V;   // Thrust from power equation
    float T_static = 0.3f * P_avail;              // Static thrust limit
    float T_mag = fminf(T_static, T_dynamic);     // Can't exceed either limit

    // ========================================================================
    // 10. FORCE DIRECTIONS (All in world frame)
    // ========================================================================
    Vec3 drag_dir = mul3(vel_norm, -1.0f);  // Opposite to velocity
    Vec3 thrust_dir = forward;               // Along body X-axis (nose)

    // Lift direction: perpendicular to velocity, in plane of velocity & wing
    // lift_dir = vel × right, then normalized
    // This ensures lift is perpendicular to V and perpendicular to span
    Vec3 lift_dir = cross3(vel_norm, right);
    float lift_dir_mag = norm3(lift_dir);
    if (lift_dir_mag > 0.01f) {
        lift_dir = mul3(lift_dir, 1.0f / lift_dir_mag);
    } else {
        lift_dir = up;  // Fallback if velocity parallel to wing (rare)
    }

    // ========================================================================
    // 11. WEIGHT (Gravity)
    // ========================================================================
    Vec3 weight = vec3(0, 0, -MASS * GRAVITY);  // Always -Z in world frame

    // ========================================================================
    // 12. SUM FORCES → ACCELERATION
    // ========================================================================
    Vec3 F_thrust = mul3(thrust_dir, T_mag);
    Vec3 F_lift = mul3(lift_dir, L_mag);
    Vec3 F_drag = mul3(drag_dir, D_mag);
    Vec3 F_total = add3(add3(add3(F_thrust, F_lift), F_drag), weight);

    // ========================================================================
    // 13. G-LIMIT (Structural Load Factor)
    // ========================================================================
    // Clamp total acceleration to prevent unrealistic maneuvers
    // 8g limit: max accel = 8 * 9.81 = 78.5 m/s²
    Vec3 accel = mul3(F_total, 1.0f / MASS);
    float accel_mag = norm3(accel);
    float g_force = accel_mag / GRAVITY;
    float max_accel = G_LIMIT * GRAVITY;
    if (accel_mag > max_accel) {
        accel = mul3(accel, max_accel / accel_mag);
    }

    if (DEBUG) printf("=== PHYSICS ===\n");
    if (DEBUG) printf("speed=%.1f m/s (stall=39.5, max=143)\n", V);
    if (DEBUG) printf("throttle=%.2f\n", throttle);
    if (DEBUG) printf("alpha_body=%.2f deg, alpha_eff=%.2f deg (incidence=%.1f), C_L=%.3f\n",
                      alpha * 180.0f / PI, alpha_effective * 180.0f / PI, WING_INCIDENCE * 180.0f / PI, C_L);
    if (DEBUG) printf("thrust=%.0f N, lift=%.0f N, drag=%.0f N, weight=%.0f N\n", T_mag, L_mag, D_mag, MASS * GRAVITY);
    if (DEBUG) printf("g_force=%.2f g (limit=8)\n", g_force);

    // ========================================================================
    // 14. INTEGRATION (Semi-implicit Euler)
    // ========================================================================
    // v(t+dt) = v(t) + a * dt
    // x(t+dt) = x(t) + v(t+dt) * dt  (using NEW velocity)
    p->vel = add3(p->vel, mul3(accel, dt));
    p->pos = add3(p->pos, mul3(p->vel, dt));

    p->throttle = throttle;
}

void step_plane(Plane *p, float dt) {
    // Simple forward motion for opponent (no actions)
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    float speed = norm3(p->vel);
    if (speed < 1.0f) speed = 80.0f;
    p->vel = mul3(forward, speed);
    p->pos = add3(p->pos, mul3(p->vel, dt));

    if (DEBUG) printf("=== TARGET ===\n");
    if (DEBUG) printf("target_speed=%.1f m/s (expected=80)\n", speed);
    if (DEBUG) printf("target_pos=(%.1f, %.1f, %.1f)\n", p->pos.x, p->pos.y, p->pos.z);
    if (DEBUG) printf("target_fwd=(%.2f, %.2f, %.2f)\n", forward.x, forward.y, forward.z);
}

// Check if shooter hits target (cone-based hit detection)
bool check_hit(Plane *shooter, Plane *target) {
    Vec3 to_target = sub3(target->pos, shooter->pos);
    float dist = norm3(to_target);
    if (dist > GUN_RANGE) return false;
    if (dist < 1.0f) return false;  // Too close (avoid division issues)

    Vec3 forward = quat_rotate(shooter->ori, vec3(1, 0, 0));
    Vec3 to_target_norm = normalize3(to_target);
    float cos_angle = dot3(to_target_norm, forward);
    return cos_angle > cosf(GUN_CONE_ANGLE);
}

// Respawn opponent at random position ahead of player
void respawn_opponent(Dogfight *env) {
    Plane *p = &env->player;
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));

    // Spawn 300-600m ahead, with some lateral offset
    Vec3 opp_pos = vec3(
        p->pos.x + fwd.x * rndf(300, 600) + rndf(-100, 100),
        p->pos.y + fwd.y * rndf(300, 600) + rndf(-100, 100),
        clampf(p->pos.z + rndf(-100, 100), 200, 2500)
    );
    Vec3 vel = vec3(80, 0, 0);
    reset_plane(&env->opponent, opp_pos, vel);

    if (DEBUG) printf("=== RESPAWN ===\n");
    if (DEBUG) printf("player_pos=(%.1f, %.1f, %.1f)\n", p->pos.x, p->pos.y, p->pos.z);
    if (DEBUG) printf("player_fwd=(%.2f, %.2f, %.2f)\n", fwd.x, fwd.y, fwd.z);
    if (DEBUG) printf("new_opponent_pos=(%.1f, %.1f, %.1f)\n", opp_pos.x, opp_pos.y, opp_pos.z);
    if (DEBUG) printf("opponent_vel=(%.1f, %.1f, %.1f) NOTE: always +X!\n", vel.x, vel.y, vel.z);
    if (DEBUG) printf("respawn_dist=%.1f m\n", norm3(sub3(opp_pos, p->pos)));
}

void c_step(Dogfight *env) {
    env->tick++;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    if (DEBUG) printf("\n========== TICK %d ==========\n", env->tick);
    if (DEBUG) printf("=== ACTIONS ===\n");
    if (DEBUG) printf("throttle_raw=%.3f -> throttle=%.3f\n", env->actions[0], (env->actions[0] + 1.0f) * 0.5f);
    if (DEBUG) printf("elevator=%.3f -> pitch_rate=%.3f rad/s\n", env->actions[1], env->actions[1] * MAX_PITCH_RATE);
    if (DEBUG) printf("ailerons=%.3f -> roll_rate=%.3f rad/s\n", env->actions[2], env->actions[2] * MAX_ROLL_RATE);
    if (DEBUG) printf("rudder=%.3f -> yaw_rate=%.3f rad/s\n", env->actions[3], env->actions[3] * MAX_YAW_RATE);
    if (DEBUG) printf("trigger=%.3f (fires if >0.5)\n", env->actions[4]);

    // Player uses full physics with actions
    step_plane_with_physics(&env->player, env->actions, DT);

    // Opponent uses simple motion (no actions)
    step_plane(&env->opponent, DT);

    // === Combat (Phase 5) ===
    Plane *p = &env->player;
    Plane *o = &env->opponent;
    float reward = 0.0f;

    // Decrement fire cooldowns
    if (p->fire_cooldown > 0) p->fire_cooldown--;
    if (o->fire_cooldown > 0) o->fire_cooldown--;

    // Player fires: action[4] > 0.5 and cooldown ready
    if (env->actions[4] > 0.5f && p->fire_cooldown == 0) {
        p->fire_cooldown = FIRE_COOLDOWN;
        env->log.shots_fired += 1.0f;
        if (DEBUG) printf("=== FIRED! ===\n");

        // Check if hit
        if (check_hit(p, o)) {
            env->log.shots_hit += 1.0f;
            reward += 1.0f;  // Hit reward
            if (DEBUG) printf("*** HIT! +1.0 reward ***\n");

            // Kill: respawn opponent, big reward
            env->log.kills += 1.0f;
            reward += 10.0f;  // Kill reward
            if (DEBUG) printf("*** KILL! +10.0 reward, total kills=%.0f ***\n", env->log.kills);
            respawn_opponent(env);
        } else {
            if (DEBUG) printf("MISS\n");
        }
    }

    // === Reward Shaping (Phase 3.5) ===
    Vec3 rel_pos = sub3(o->pos, p->pos);
    float dist = norm3(rel_pos);
    float r_dist = -dist / 10000.0f;
    reward += r_dist;

    // 2. Closing velocity reward: approaching = good
    Vec3 rel_vel = sub3(p->vel, o->vel);
    Vec3 rel_pos_norm = normalize3(rel_pos);
    float closing_rate = dot3(rel_vel, rel_pos_norm);
    float r_closing = closing_rate / 500.0f;
    reward += r_closing;

    // 3. Tail position reward: behind opponent = good
    Vec3 opp_forward = quat_rotate(o->ori, vec3(1, 0, 0));
    float tail_angle = dot3(rel_pos_norm, opp_forward);
    float r_tail = tail_angle * 0.02f;
    reward += r_tail;

    // 4. Altitude penalty: too low or too high is bad
    float r_alt = 0.0f;
    if (p->pos.z < 200.0f) {
        r_alt = -(200.0f - p->pos.z) / 2000.0f;
    } else if (p->pos.z > 2500.0f) {
        r_alt = -(p->pos.z - 2500.0f) / 5000.0f;
    }
    reward += r_alt;

    // 5. Speed penalty: too slow is stall risk
    float speed = norm3(p->vel);
    float r_speed = 0.0f;
    if (speed < 50.0f) {
        r_speed = -(50.0f - speed) / 500.0f;
    }
    reward += r_speed;

    // 6. Aiming reward: feedback for gun alignment before actual hits
    Vec3 player_fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 to_opp_norm = normalize3(rel_pos);
    float aim_dot = dot3(to_opp_norm, player_fwd);  // 1.0 = perfect aim
    float aim_angle_deg = acosf(clampf(aim_dot, -1.0f, 1.0f)) * 180.0f / PI;

    float r_aim = 0.0f;
    // Reward for tracking (within 2x gun cone and in range)
    if (aim_dot > cosf(GUN_CONE_ANGLE * 2.0f) && dist < GUN_RANGE) {
        r_aim += 0.05f;
    }
    // Bonus for firing solution (within gun cone, in range)
    if (aim_dot > cosf(GUN_CONE_ANGLE) && dist < GUN_RANGE) {
        r_aim += 0.1f;
    }
    reward += r_aim;

    if (DEBUG) printf("=== REWARD ===\n");
    if (DEBUG) printf("r_dist=%.4f (dist=%.1f m)\n", r_dist, dist);
    if (DEBUG) printf("r_closing=%.4f (rate=%.1f m/s)\n", r_closing, closing_rate);
    if (DEBUG) printf("r_tail=%.4f (angle=%.2f)\n", r_tail, tail_angle);
    if (DEBUG) printf("r_alt=%.4f (z=%.1f)\n", r_alt, p->pos.z);
    if (DEBUG) printf("r_speed=%.4f (speed=%.1f)\n", r_speed, speed);
    if (DEBUG) printf("r_aim=%.4f (aim_angle=%.1f deg, dist=%.1f)\n", r_aim, aim_angle_deg, dist);
    if (DEBUG) printf("reward_total=%.4f\n", reward);

    if (DEBUG) printf("=== COMBAT ===\n");
    if (DEBUG) printf("aim_angle=%.1f deg (cone=5 deg)\n", aim_angle_deg);
    if (DEBUG) printf("dist_to_target=%.1f m (gun_range=500)\n", dist);
    if (DEBUG) printf("in_cone=%d, in_range=%d\n", aim_dot > cosf(GUN_CONE_ANGLE), dist < GUN_RANGE);

    env->rewards[0] = reward;
    env->episode_return += reward;

    // Check bounds (player only)
    bool oob = fabsf(p->pos.x) > WORLD_HALF_X ||
               fabsf(p->pos.y) > WORLD_HALF_Y ||
               p->pos.z < 0 || p->pos.z > WORLD_MAX_Z;

    if (oob || env->tick >= env->max_steps) {
        if (DEBUG) printf("=== TERMINAL ===\n");
        if (DEBUG) printf("oob=%d (x=%.1f, y=%.1f, z=%.1f)\n", oob, p->pos.x, p->pos.y, p->pos.z);
        if (DEBUG) printf("max_steps=%d, tick=%d\n", env->max_steps, env->tick);
        if (DEBUG) printf("episode_return=%.2f\n", env->episode_return);
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

    // 10. Draw tracer when firing (cooldown just set = just fired)
    if (p->fire_cooldown >= FIRE_COOLDOWN - 2) {  // Show for 2 frames
        Vec3 nose = add3(p->pos, quat_rotate(p->ori, vec3(15, 0, 0)));
        Vec3 tracer_end = add3(p->pos, quat_rotate(p->ori, vec3(GUN_RANGE, 0, 0)));
        Vector3 nose_r = {nose.x, nose.y, nose.z};
        Vector3 end_r = {tracer_end.x, tracer_end.y, tracer_end.z};
        DrawLine3D(nose_r, end_r, YELLOW);
    }

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
    DrawText(TextFormat("Kills: %.0f | Shots: %.0f/%.0f", env->log.kills, env->log.shots_hit, env->log.shots_fired), 10, 190, 20, YELLOW);

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
