// dogfight.h - WW2 aerial combat environment
// Uses flightlib.h for flight physics

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

// Define DEBUG before including flightlib.h so physics functions can use it
#define DEBUG 0

#include "flightlib.h"
#include "autopilot.h"

// Observation scheme enumeration
typedef enum {
    OBS_WORLD_FRAME = 0,    // Current baseline (19 obs)
    OBS_BODY_FRAME = 1,     // Body-frame transforms (21 obs)
    OBS_ANGLES = 2,         // Spherical coordinates (12 obs)
    OBS_CONTROL_ERROR = 3,  // Control errors to target (17 obs)
    OBS_REALISTIC = 4,      // Cockpit instruments only (10 obs)
    OBS_MAXIMALIST = 5,     // Everything combined (43 obs)
    OBS_SCHEME_COUNT
} ObsScheme;

// Observation size lookup table
static const int OBS_SIZES[OBS_SCHEME_COUNT] = {19, 21, 12, 17, 10, 43};

// Simulation timing
#define DT 0.02f

// World bounds
#define WORLD_HALF_X 2000.0f
#define WORLD_HALF_Y 2000.0f
#define WORLD_MAX_Z 3000.0f
#define MAX_SPEED 250.0f
#define OBS_SIZE 19  // player(13) + rel_pos(3) + rel_vel(3)

// Inverse constants for faster normalization (multiply instead of divide)
#define INV_WORLD_HALF_X 0.0005f       // 1/2000
#define INV_WORLD_HALF_Y 0.0005f       // 1/2000
#define INV_WORLD_MAX_Z  0.000333333f  // 1/3000
#define INV_MAX_SPEED    0.004f        // 1/250

// Combat constants
#define GUN_RANGE 500.0f       // meters
#define GUN_CONE_ANGLE 0.087f  // ~5 degrees in radians
#define FIRE_COOLDOWN 10       // ticks (0.2 seconds at 50Hz)

typedef struct Log {
    float episode_return;
    float episode_length;
    float score;           // 1.0 on kill, 0.0 on failure
    float perf;            // sweep metric (same as kills)
    float kills;           // cumulative kills
    float shots_fired;     // cumulative shots
    float accuracy;        // kills / shots_fired * 100
    float n;
} Log;

// Reward configuration (all values sweepable via INI)
typedef struct RewardConfig {
    float kill;              // +N for kill (fixed at 1.0)
    float hit;               // +N for hit
    float dist_scale;        // -N per meter distance
    float closing_scale;     // +N per m/s closing
    float tail_scale;        // ±N for tail position
    float tracking;          // +N when in 2x gun cone
    float firing_solution;   // +N when in 1x gun cone
    float alt_low;           // -N per meter below alt_min
    float alt_high;          // -N per meter above alt_max
    float stall;             // -N per m/s below speed_min
    // Thresholds (not rewards)
    float alt_min;           // 200.0
    float alt_max;           // 2500.0
    float speed_min;         // 50.0
} RewardConfig;

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
    // Per-episode precomputed values (for curriculum learning)
    float gun_cone_angle;   // Current cone angle (radians)
    float cos_gun_cone;     // cosf(gun_cone_angle)
    float cos_gun_cone_2x;  // cosf(gun_cone_angle * 2)
    // Opponent autopilot
    AutopilotState opponent_ap;
    // Observation scheme
    int obs_scheme;
    int obs_size;
    // Reward configuration (sweepable)
    RewardConfig rcfg;
    // Episode-level tracking (reset each episode)
    int kill;                   // 1 if killed this episode, 0 otherwise
    float episode_shots_fired;  // For accuracy tracking
} Dogfight;

void init(Dogfight *env, int obs_scheme, RewardConfig *rcfg) {
    env->log = (Log){0};
    env->tick = 0;
    env->episode_return = 0.0f;
    env->client = NULL;
    // Observation scheme
    env->obs_scheme = (obs_scheme >= 0 && obs_scheme < OBS_SCHEME_COUNT) ? obs_scheme : 0;
    env->obs_size = OBS_SIZES[env->obs_scheme];
    // Precompute gun cone trig (can vary per episode for curriculum)
    env->gun_cone_angle = GUN_CONE_ANGLE;
    env->cos_gun_cone = cosf(env->gun_cone_angle);
    env->cos_gun_cone_2x = cosf(env->gun_cone_angle * 2.0f);
    // Initialize opponent autopilot
    autopilot_init(&env->opponent_ap);
    // Reward configuration (copy from provided config)
    env->rcfg = *rcfg;
    // Episode tracking
    env->kill = 0;
    env->episode_shots_fired = 0.0f;
}

void add_log(Dogfight *env) {
    if (DEBUG) printf("=== ADD_LOG ===\n");
    if (DEBUG) printf("  kill=%d, episode_return=%.2f, tick=%d\n", env->kill, env->episode_return, env->tick);
    if (DEBUG) printf("  episode_shots_fired=%.0f, reward=%.2f\n", env->episode_shots_fired, env->rewards[0]);
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.perf += env->kill ? 1.0f : 0.0f;
    env->log.kills += env->kill ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.shots_fired += env->episode_shots_fired;
    env->log.accuracy = (env->log.shots_fired > 0.0f) ? (env->log.kills / env->log.shots_fired * 100.0f) : 0.0f;
    env->log.n += 1.0f;
    if (DEBUG) printf("  log.perf=%.2f, log.shots_fired=%.0f, log.n=%.0f\n", env->log.perf, env->log.shots_fired, env->log.n);
}

// Scheme 0: World frame observations (original baseline)
void compute_obs_world_frame(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_vel = sub3(o->vel, p->vel);

    if (DEBUG) printf("=== OBS tick=%d ===\n", env->tick);

    int i = 0;
    if (DEBUG) printf("pos_x_norm=%.3f (raw=%.1f)\n", p->pos.x * INV_WORLD_HALF_X, p->pos.x);
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    if (DEBUG) printf("pos_y_norm=%.3f (raw=%.1f)\n", p->pos.y * INV_WORLD_HALF_Y, p->pos.y);
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    if (DEBUG) printf("pos_z_norm=%.3f (raw=%.1f)\n", p->pos.z * INV_WORLD_MAX_Z, p->pos.z);
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    if (DEBUG) printf("vel_x_norm=%.3f (raw=%.1f)\n", p->vel.x * INV_MAX_SPEED, p->vel.x);
    env->observations[i++] = p->vel.x * INV_MAX_SPEED;
    if (DEBUG) printf("vel_y_norm=%.3f (raw=%.1f)\n", p->vel.y * INV_MAX_SPEED, p->vel.y);
    env->observations[i++] = p->vel.y * INV_MAX_SPEED;
    if (DEBUG) printf("vel_z_norm=%.3f (raw=%.1f)\n", p->vel.z * INV_MAX_SPEED, p->vel.z);
    env->observations[i++] = p->vel.z * INV_MAX_SPEED;
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
    if (DEBUG) printf("rel_pos_x_norm=%.3f (raw=%.1f)\n", rel_pos.x * INV_WORLD_HALF_X, rel_pos.x);
    env->observations[i++] = rel_pos.x * INV_WORLD_HALF_X;
    if (DEBUG) printf("rel_pos_y_norm=%.3f (raw=%.1f)\n", rel_pos.y * INV_WORLD_HALF_Y, rel_pos.y);
    env->observations[i++] = rel_pos.y * INV_WORLD_HALF_Y;
    if (DEBUG) printf("rel_pos_z_norm=%.3f (raw=%.1f)\n", rel_pos.z * INV_WORLD_MAX_Z, rel_pos.z);
    env->observations[i++] = rel_pos.z * INV_WORLD_MAX_Z;
    if (DEBUG) printf("rel_vel_x_norm=%.3f (raw=%.1f)\n", rel_vel.x * INV_MAX_SPEED, rel_vel.x);
    env->observations[i++] = rel_vel.x * INV_MAX_SPEED;
    if (DEBUG) printf("rel_vel_y_norm=%.3f (raw=%.1f)\n", rel_vel.y * INV_MAX_SPEED, rel_vel.y);
    env->observations[i++] = rel_vel.y * INV_MAX_SPEED;
    if (DEBUG) printf("rel_vel_z_norm=%.3f (raw=%.1f)\n", rel_vel.z * INV_MAX_SPEED, rel_vel.z);
    env->observations[i++] = rel_vel.z * INV_MAX_SPEED;
    // OBS_SIZE = 19
}

// Scheme 1: Body frame observations (rel_pos/vel in body frame + aim helpers)
void compute_obs_body_frame(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    // Inverse quaternion for world→body transform
    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Transform quantities to body frame
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_vel = sub3(o->vel, p->vel);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);  // rel_pos_body.x > 0 = ahead
    Vec3 rel_vel_body = quat_rotate(q_inv, rel_vel);

    // Aim helpers
    float dist = norm3(rel_pos);
    Vec3 to_target = normalize3(rel_pos_body);
    float aim_dot = to_target.x;  // In body frame, +X is forward

    // Up vector (world frame - for attitude reference)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    int i = 0;
    // Player position (world - for bounds awareness)
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    // Player velocity (body frame)
    env->observations[i++] = vel_body.x * INV_MAX_SPEED;  // Forward speed
    env->observations[i++] = vel_body.y * INV_MAX_SPEED;  // Sideslip
    env->observations[i++] = vel_body.z * INV_MAX_SPEED;  // Climb rate
    // Player orientation
    env->observations[i++] = p->ori.w;
    env->observations[i++] = p->ori.x;
    env->observations[i++] = p->ori.y;
    env->observations[i++] = p->ori.z;
    // Player up (world - for roll reference)
    env->observations[i++] = up.x;
    env->observations[i++] = up.y;
    env->observations[i++] = up.z;
    // Relative position (body frame) - THE KEY CHANGE
    env->observations[i++] = rel_pos_body.x * INV_WORLD_HALF_X;
    env->observations[i++] = rel_pos_body.y * INV_WORLD_HALF_Y;
    env->observations[i++] = rel_pos_body.z * INV_WORLD_MAX_Z;
    // Relative velocity (body frame)
    env->observations[i++] = rel_vel_body.x * INV_MAX_SPEED;
    env->observations[i++] = rel_vel_body.y * INV_MAX_SPEED;
    env->observations[i++] = rel_vel_body.z * INV_MAX_SPEED;
    // Aim helpers (NEW)
    env->observations[i++] = aim_dot;  // -1 to 1, 1 = perfect aim
    env->observations[i++] = clampf(dist / GUN_RANGE, 0.0f, 4.0f) - 2.0f;  // ~[-1,1]
    // OBS_SIZE = 21
}

// Scheme 2: Angles observations (spherical coordinates)
void compute_obs_angles(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles from quaternion
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));
    float yaw = atan2f(2.0f * (p->ori.w * p->ori.z + p->ori.x * p->ori.y),
                       1.0f - 2.0f * (p->ori.y * p->ori.y + p->ori.z * p->ori.z));

    // Target in body frame → spherical
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float azimuth = atan2f(rel_pos_body.y, rel_pos_body.x);  // -pi to pi
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float elevation = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));  // -pi/2 to pi/2

    // Closing rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closing_rate = dot3(rel_vel, normalize3(rel_pos));

    // Opponent heading relative to player
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 opp_fwd_body = quat_rotate(q_inv, opp_fwd);
    float opp_heading = atan2f(opp_fwd_body.y, opp_fwd_body.x);

    int i = 0;
    // Player state
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = norm3(p->vel) * INV_MAX_SPEED;  // Speed scalar
    env->observations[i++] = pitch / PI;      // -0.5 to 0.5
    env->observations[i++] = roll / PI;       // -1 to 1
    env->observations[i++] = yaw / PI;        // -1 to 1

    // Target angles
    env->observations[i++] = azimuth / PI;    // -1 to 1
    env->observations[i++] = elevation / (PI * 0.5f);  // -1 to 1
    env->observations[i++] = clampf(dist / GUN_RANGE, 0.0f, 4.0f) - 2.0f;  // ~[-1,1]
    env->observations[i++] = closing_rate * INV_MAX_SPEED;

    // Opponent info
    env->observations[i++] = opp_heading / PI;  // -1 to 1
    // OBS_SIZE = 12
}

// Scheme 3: Control error observations (what inputs would point at target?)
void compute_obs_control_error(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Up vector (world frame)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Target in body frame
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);
    Vec3 to_target_norm = normalize3(rel_pos_body);

    // Control errors: how to point at target
    float pitch_error = asinf(clampf(to_target_norm.z, -1.0f, 1.0f));  // + = pitch up needed
    float yaw_error = atan2f(to_target_norm.y, to_target_norm.x);      // + = yaw right needed

    // Roll to turn: if target is right (y>0), roll right helps turn toward it
    // This is the bank angle that would help turn toward target
    float roll_to_turn = atan2f(to_target_norm.y, fabsf(to_target_norm.x) + 0.1f);

    // Closing rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closing_rate = dot3(rel_vel, normalize3(rel_pos));

    // Opponent heading relative to player
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 opp_fwd_body = quat_rotate(q_inv, opp_fwd);
    float opp_heading = atan2f(opp_fwd_body.y, opp_fwd_body.x);

    int i = 0;
    // Player state (11 obs)
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = norm3(p->vel) * INV_MAX_SPEED;  // Speed scalar
    env->observations[i++] = p->ori.w;
    env->observations[i++] = p->ori.x;
    env->observations[i++] = p->ori.y;
    env->observations[i++] = p->ori.z;
    env->observations[i++] = up.x;
    env->observations[i++] = up.y;
    env->observations[i++] = up.z;

    // Control errors (4 obs) - THE KEY INFO
    env->observations[i++] = pitch_error / (PI * 0.5f);  // -1 to 1
    env->observations[i++] = yaw_error / PI;              // -1 to 1
    env->observations[i++] = roll_to_turn / (PI * 0.5f); // -1 to 1
    env->observations[i++] = clampf(dist / GUN_RANGE, 0.0f, 4.0f) - 2.0f;

    // Target info (2 obs)
    env->observations[i++] = closing_rate * INV_MAX_SPEED;
    env->observations[i++] = opp_heading / PI;
    // OBS_SIZE = 17
}

// Scheme 4: Realistic cockpit instruments only
void compute_obs_realistic(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Target apparent size (larger when closer)
    float target_size = 20.0f / fmaxf(dist, 10.0f);  // ~wingspan/distance

    // Opponent aspect (are they facing toward/away from us?)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);  // 1 = head-on, -1 = tail

    // Horizon visible (is up vector pointing up?)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;  // 1 = level, 0 = knife-edge, -1 = inverted

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = norm3(p->vel) * INV_MAX_SPEED;  // Airspeed
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;     // Altitude
    env->observations[i++] = pitch / (PI * 0.5f);            // Pitch indicator
    env->observations[i++] = roll / PI;                       // Bank indicator

    // Gunsight (3 obs)
    env->observations[i++] = target_az / PI;                  // Target azimuth in sight
    env->observations[i++] = target_el / (PI * 0.5f);         // Target elevation in sight
    env->observations[i++] = clampf(target_size, 0.0f, 2.0f) - 1.0f;  // Target size

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;                   // -1 to 1
    env->observations[i++] = horizon_visible;                 // -1 to 1
    env->observations[i++] = clampf(dist / GUN_RANGE, 0.0f, 4.0f) - 2.0f;  // Distance estimate
    // OBS_SIZE = 10
}

// Scheme 5: Maximalist - everything potentially useful
void compute_obs_maximalist(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player transforms
    Vec3 vel_body = quat_rotate(q_inv, p->vel);
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));
    float yaw = atan2f(2.0f * (p->ori.w * p->ori.z + p->ori.x * p->ori.y),
                       1.0f - 2.0f * (p->ori.y * p->ori.y + p->ori.z * p->ori.z));

    // Relative quantities
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_vel = sub3(o->vel, p->vel);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    Vec3 rel_vel_body = quat_rotate(q_inv, rel_vel);
    float dist = norm3(rel_pos);

    // Spherical coordinates
    float azimuth = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float elevation = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Aim and closing
    Vec3 to_target = normalize3(rel_pos_body);
    float aim_dot = to_target.x;
    Vec3 rel_vel_closing = sub3(p->vel, o->vel);
    float closing_rate = dot3(rel_vel_closing, normalize3(rel_pos));

    // Opponent forward vector
    Vec3 opp_fwd_world = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 opp_fwd_body = quat_rotate(q_inv, opp_fwd_world);

    int i = 0;
    // Player position (3)
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    // Player velocity world (3)
    env->observations[i++] = p->vel.x * INV_MAX_SPEED;
    env->observations[i++] = p->vel.y * INV_MAX_SPEED;
    env->observations[i++] = p->vel.z * INV_MAX_SPEED;
    // Player velocity body (3)
    env->observations[i++] = vel_body.x * INV_MAX_SPEED;
    env->observations[i++] = vel_body.y * INV_MAX_SPEED;
    env->observations[i++] = vel_body.z * INV_MAX_SPEED;
    // Player quaternion (4)
    env->observations[i++] = p->ori.w;
    env->observations[i++] = p->ori.x;
    env->observations[i++] = p->ori.y;
    env->observations[i++] = p->ori.z;
    // Player up (3)
    env->observations[i++] = up.x;
    env->observations[i++] = up.y;
    env->observations[i++] = up.z;
    // Player scalars (4)
    env->observations[i++] = norm3(p->vel) * INV_MAX_SPEED;
    env->observations[i++] = pitch / (PI * 0.5f);
    env->observations[i++] = roll / PI;
    env->observations[i++] = yaw / PI;
    // Relative position world (3)
    env->observations[i++] = rel_pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = rel_pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = rel_pos.z * INV_WORLD_MAX_Z;
    // Relative position body (3)
    env->observations[i++] = rel_pos_body.x * INV_WORLD_HALF_X;
    env->observations[i++] = rel_pos_body.y * INV_WORLD_HALF_Y;
    env->observations[i++] = rel_pos_body.z * INV_WORLD_MAX_Z;
    // Relative velocity world (3)
    env->observations[i++] = rel_vel.x * INV_MAX_SPEED;
    env->observations[i++] = rel_vel.y * INV_MAX_SPEED;
    env->observations[i++] = rel_vel.z * INV_MAX_SPEED;
    // Relative velocity body (3)
    env->observations[i++] = rel_vel_body.x * INV_MAX_SPEED;
    env->observations[i++] = rel_vel_body.y * INV_MAX_SPEED;
    env->observations[i++] = rel_vel_body.z * INV_MAX_SPEED;
    // Target angles and scalars (5)
    env->observations[i++] = azimuth / PI;
    env->observations[i++] = elevation / (PI * 0.5f);
    env->observations[i++] = clampf(dist / GUN_RANGE, 0.0f, 4.0f) - 2.0f;
    env->observations[i++] = aim_dot;
    env->observations[i++] = closing_rate * INV_MAX_SPEED;
    // Opponent forward world (3)
    env->observations[i++] = opp_fwd_world.x;
    env->observations[i++] = opp_fwd_world.y;
    env->observations[i++] = opp_fwd_world.z;
    // Opponent forward body (3)
    env->observations[i++] = opp_fwd_body.x;
    env->observations[i++] = opp_fwd_body.y;
    env->observations[i++] = opp_fwd_body.z;
    // OBS_SIZE = 43
}

// Dispatcher function
void compute_observations(Dogfight *env) {
    switch (env->obs_scheme) {
        case OBS_WORLD_FRAME:   compute_obs_world_frame(env); break;
        case OBS_BODY_FRAME:    compute_obs_body_frame(env); break;
        case OBS_ANGLES:        compute_obs_angles(env); break;
        case OBS_CONTROL_ERROR: compute_obs_control_error(env); break;
        case OBS_REALISTIC:     compute_obs_realistic(env); break;
        case OBS_MAXIMALIST:    compute_obs_maximalist(env); break;
        default:                compute_obs_world_frame(env); break;
    }
}

void c_reset(Dogfight *env) {
    env->tick = 0;
    env->episode_return = 0.0f;

    // Clear episode tracking
    env->kill = 0;
    env->episode_shots_fired = 0.0f;

    // Recompute gun cone trig (for curriculum: could vary gun_cone_angle here)
    env->cos_gun_cone = cosf(env->gun_cone_angle);
    env->cos_gun_cone_2x = cosf(env->gun_cone_angle * 2.0f);

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

    // Handle autopilot: randomize if configured, reset PID state
    if (env->opponent_ap.randomize_on_reset) {
        autopilot_randomize(&env->opponent_ap);
    }
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;

    if (DEBUG) printf("=== RESET ===\n");
    if (DEBUG) printf("kill=%d, episode_shots_fired=%.0f (now cleared)\n", env->kill, env->episode_shots_fired);
    if (DEBUG) printf("player_pos=(%.1f, %.1f, %.1f)\n", pos.x, pos.y, pos.z);
    if (DEBUG) printf("player_vel=(%.1f, %.1f, %.1f) speed=%.1f\n", vel.x, vel.y, vel.z, norm3(vel));
    if (DEBUG) printf("opponent_pos=(%.1f, %.1f, %.1f)\n", opp_pos.x, opp_pos.y, opp_pos.z);
    if (DEBUG) printf("initial_dist=%.1f m\n", norm3(sub3(opp_pos, pos)));

    compute_observations(env);
}

// Check if shooter hits target (cone-based hit detection)
bool check_hit(Plane *shooter, Plane *target, float cos_gun_cone) {
    Vec3 to_target = sub3(target->pos, shooter->pos);
    float dist = norm3(to_target);
    if (dist > GUN_RANGE) return false;
    if (dist < 1.0f) return false;  // Too close (avoid division issues)

    Vec3 forward = quat_rotate(shooter->ori, vec3(1, 0, 0));
    Vec3 to_target_norm = normalize3(to_target);
    float cos_angle = dot3(to_target_norm, forward);
    return cos_angle > cos_gun_cone;
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

    // Reset autopilot PID state on respawn
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;

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

    // Opponent uses autopilot (if not AP_STRAIGHT, uses full physics)
    if (env->opponent_ap.mode != AP_STRAIGHT) {
        float opp_actions[5];
        autopilot_step(&env->opponent_ap, &env->opponent, opp_actions, DT);
        step_plane_with_physics(&env->opponent, opp_actions, DT);
    } else {
        step_plane(&env->opponent, DT);
    }

    // === Combat (Phase 5) ===
    Plane *p = &env->player;
    Plane *o = &env->opponent;
    float reward = 0.0f;

    // Decrement fire cooldowns
    if (p->fire_cooldown > 0) p->fire_cooldown--;
    if (o->fire_cooldown > 0) o->fire_cooldown--;

    // Player fires: action[4] > 0.5 and cooldown ready
    if (DEBUG) printf("trigger=%.3f, cooldown=%d\n", env->actions[4], p->fire_cooldown);
    if (env->actions[4] > 0.5f && p->fire_cooldown == 0) {
        p->fire_cooldown = FIRE_COOLDOWN;
        env->episode_shots_fired += 1.0f;
        if (DEBUG) printf("=== FIRED! episode_shots_fired=%.0f ===\n", env->episode_shots_fired);

        // Check if hit = kill = SUCCESS = terminal
        if (check_hit(p, o, env->cos_gun_cone)) {
            if (DEBUG) printf("*** KILL! ***\n");
            env->kill = 1;
            env->rewards[0] = 1.0f;
            env->episode_return += 1.0f;
            env->terminals[0] = 1;
            add_log(env);
            c_reset(env);
            return;
        } else {
            if (DEBUG) printf("MISS (dist=%.1f, in_cone=%d)\n", norm3(sub3(o->pos, p->pos)),
                check_hit(p, o, env->cos_gun_cone));
        }
    }

    // === Reward Shaping (all values from rcfg, sweepable) ===
    Vec3 rel_pos = sub3(o->pos, p->pos);
    float dist = norm3(rel_pos);
    float r_dist = -dist * env->rcfg.dist_scale;
    reward += r_dist;

    // 2. Closing velocity reward: approaching = good
    Vec3 rel_vel = sub3(p->vel, o->vel);
    Vec3 rel_pos_norm = normalize3(rel_pos);
    float closing_rate = dot3(rel_vel, rel_pos_norm);
    float r_closing = closing_rate * env->rcfg.closing_scale;
    reward += r_closing;

    // 3. Tail position reward: behind opponent = good
    Vec3 opp_forward = quat_rotate(o->ori, vec3(1, 0, 0));
    float tail_angle = dot3(rel_pos_norm, opp_forward);
    float r_tail = tail_angle * env->rcfg.tail_scale;
    reward += r_tail;

    // 4. Altitude penalty: too low or too high is bad
    float r_alt = 0.0f;
    if (p->pos.z < env->rcfg.alt_min) {
        r_alt = -(env->rcfg.alt_min - p->pos.z) * env->rcfg.alt_low;
    } else if (p->pos.z > env->rcfg.alt_max) {
        r_alt = -(p->pos.z - env->rcfg.alt_max) * env->rcfg.alt_high;
    }
    reward += r_alt;

    // 5. Speed penalty: too slow is stall risk
    float speed = norm3(p->vel);
    float r_speed = 0.0f;
    if (speed < env->rcfg.speed_min) {
        r_speed = -(env->rcfg.speed_min - speed) * env->rcfg.stall;
    }
    reward += r_speed;

    // 6. Aiming reward: feedback for gun alignment before actual hits
    Vec3 player_fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 to_opp_norm = normalize3(rel_pos);
    float aim_dot = dot3(to_opp_norm, player_fwd);  // 1.0 = perfect aim
    float aim_angle_deg = acosf(clampf(aim_dot, -1.0f, 1.0f)) * RAD_TO_DEG;

    float r_aim = 0.0f;
    // Reward for tracking (within 2x gun cone and in range)
    if (aim_dot > env->cos_gun_cone_2x && dist < GUN_RANGE) {
        r_aim += env->rcfg.tracking;
    }
    // Bonus for firing solution (within gun cone, in range)
    if (aim_dot > env->cos_gun_cone && dist < GUN_RANGE) {
        r_aim += env->rcfg.firing_solution;
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
    if (DEBUG) printf("in_cone=%d, in_range=%d\n", aim_dot > env->cos_gun_cone, dist < GUN_RANGE);

    env->rewards[0] = reward;
    env->episode_return += reward;

    // Check bounds (player only)
    bool oob = fabsf(p->pos.x) > WORLD_HALF_X ||
               fabsf(p->pos.y) > WORLD_HALF_Y ||
               p->pos.z < 0 || p->pos.z > WORLD_MAX_Z;

    if (oob || env->tick >= env->max_steps) {
        if (DEBUG) printf("=== TERMINAL (FAILURE) ===\n");
        if (DEBUG) printf("oob=%d (x=%.1f, y=%.1f, z=%.1f)\n", oob, p->pos.x, p->pos.y, p->pos.z);
        if (DEBUG) printf("max_steps=%d, tick=%d\n", env->max_steps, env->tick);
        env->rewards[0] = 0.0f;  // No reward on failure
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
    // Bounds: X +/-2000, Y +/-2000, Z 0-3000 -> center at (0, 0, 1500)
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
    DrawText(TextFormat("Perf: %.1f%% | Shots: %.0f", env->log.perf / fmaxf(env->log.n, 1.0f) * 100.0f, env->log.shots_fired), 10, 190, 20, YELLOW);

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

// Force exact game state for testing. Defaults shown in comments are applied in Python.
void force_state(
    Dogfight *env,
    float p_px,        // = 0.0f, player pos X
    float p_py,        // = 0.0f, player pos Y
    float p_pz,        // = 1000.0f, player pos Z
    float p_vx,        // = 150.0f, player vel X (m/s)
    float p_vy,        // = 0.0f, player vel Y
    float p_vz,        // = 0.0f, player vel Z
    float p_ow,        // = 1.0f, player orientation quat W
    float p_ox,        // = 0.0f, player orientation quat X
    float p_oy,        // = 0.0f, player orientation quat Y
    float p_oz,        // = 0.0f, player orientation quat Z
    float p_throttle,  // = 1.0f, player throttle [0,1]
    float o_px,        // = -9999.0f (auto: 400m ahead), opponent pos X
    float o_py,        // = -9999.0f (auto), opponent pos Y
    float o_pz,        // = -9999.0f (auto), opponent pos Z
    float o_vx,        // = -9999.0f (auto: match player), opponent vel X
    float o_vy,        // = -9999.0f (auto), opponent vel Y
    float o_vz,        // = -9999.0f (auto), opponent vel Z
    float o_ow,        // = -9999.0f (auto: match player), opponent ori W
    float o_ox,        // = -9999.0f (auto), opponent ori X
    float o_oy,        // = -9999.0f (auto), opponent ori Y
    float o_oz,        // = -9999.0f (auto), opponent ori Z
    int tick           // = 0, environment tick
) {
    // Player state
    env->player.pos = vec3(p_px, p_py, p_pz);
    env->player.vel = vec3(p_vx, p_vy, p_vz);
    env->player.ori = quat(p_ow, p_ox, p_oy, p_oz);
    quat_normalize(&env->player.ori);
    env->player.throttle = p_throttle;
    env->player.fire_cooldown = 0;

    // Opponent position: auto = 400m ahead of player
    if (o_px < -9000.0f) {
        Vec3 fwd = quat_rotate(env->player.ori, vec3(1, 0, 0));
        env->opponent.pos = add3(env->player.pos, mul3(fwd, 400.0f));
    } else {
        env->opponent.pos = vec3(o_px, o_py, o_pz);
    }

    // Opponent velocity: auto = match player
    if (o_vx < -9000.0f) {
        env->opponent.vel = env->player.vel;
    } else {
        env->opponent.vel = vec3(o_vx, o_vy, o_vz);
    }

    // Opponent orientation: auto = match player
    if (o_ow < -9000.0f) {
        env->opponent.ori = env->player.ori;
    } else {
        env->opponent.ori = quat(o_ow, o_ox, o_oy, o_oz);
        quat_normalize(&env->opponent.ori);
    }
    env->opponent.fire_cooldown = 0;

    // Environment state
    env->tick = tick;
    env->episode_return = 0.0f;

    compute_observations(env);
}
