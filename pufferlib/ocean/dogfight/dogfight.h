// dogfight.h - WW2 aerial combat environment
// Uses flightlib.h for flight physics

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "rlgl.h"  // For rlSetClipPlanes()

// Define DEBUG before including flightlib.h so physics functions can use it
#define DEBUG 0

#include "flightlib.h"
#include "autopilot.h"

// Observation scheme enumeration
typedef enum {
    OBS_ANGLES = 0,              // Spherical coordinates (12 obs)
    OBS_CONTROL_ERROR = 1,       // Control errors to target (17 obs)
    OBS_REALISTIC = 2,           // Cockpit instruments only (10 obs)
    OBS_REALISTIC_RANGE = 3,     // REALISTIC with explicit range (10 obs)
    OBS_REALISTIC_ENEMY_STATE = 4, // + enemy pitch/roll/heading (13 obs)
    OBS_REALISTIC_FULL = 5,      // + turn rate + G-loading (15 obs)
    OBS_SCHEME_COUNT
} ObsScheme;

// Observation size lookup table
static const int OBS_SIZES[OBS_SCHEME_COUNT] = {12, 17, 10, 10, 13, 15};

// Curriculum learning stages (progressive difficulty)
typedef enum {
    CURRICULUM_TAIL_CHASE = 0,   // Easiest: opponent ahead, same heading
    CURRICULUM_HEAD_ON,          // Opponent coming toward us
    CURRICULUM_CROSSING,         // 90 degree deflection shots
    CURRICULUM_VERTICAL,         // Above or below player
    CURRICULUM_MANEUVERING,      // Opponent does turns
    CURRICULUM_FULL_RANDOM,      // Maximum difficulty
    CURRICULUM_COUNT
} CurriculumStage;

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
    float stage;           // current curriculum stage (for monitoring)
    float n;
} Log;

// Reward configuration (all values sweepable via INI)
typedef struct RewardConfig {
    float dist_scale;        // -N per meter distance
    float closing_scale;     // +N per m/s closing
    float tail_scale;        // ±N for tail position
    float tracking;          // +N when in 2x gun cone
    float firing_solution;   // +N when in 1x gun cone
    float alt_low;           // -N per meter below alt_min
    float alt_high;          // -N per meter above alt_max
    float stall;             // -N per m/s below speed_min
    float roll;              // -N per radian of bank angle (gentle level preference)
    float neg_g;             // -N per unit of negative elevator (pushing forward)
    float rudder;            // -N per unit of rudder magnitude
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
    // Curriculum learning
    int curriculum_enabled;     // 0 = off (legacy spawning), 1 = on
    int curriculum_randomize;   // 0 = progressive (training), 1 = random stage each episode (eval)
    int episodes_per_stage;     // Episodes before advancing to next stage
    int total_episodes;         // Cumulative episodes (persists across resets)
    CurriculumStage stage;      // Current difficulty stage
} Dogfight;

void init(Dogfight *env, int obs_scheme, RewardConfig *rcfg, int curriculum_enabled, int curriculum_randomize, int episodes_per_stage) {
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
    // Curriculum learning
    env->curriculum_enabled = curriculum_enabled;
    env->curriculum_randomize = curriculum_randomize;
    env->episodes_per_stage = episodes_per_stage > 0 ? episodes_per_stage : 15000;
    env->total_episodes = 0;
    env->stage = CURRICULUM_TAIL_CHASE;
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
    env->log.stage = (float)env->stage;  // Track curriculum stage
    env->log.n += 1.0f;
    if (DEBUG) printf("  log.perf=%.2f, log.shots_fired=%.0f, log.n=%.0f\n", env->log.perf, env->log.shots_fired, env->log.n);
}

// Scheme 0: Angles observations (spherical coordinates)
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

// Scheme 3: REALISTIC with explicit range (10 obs)
// Like REALISTIC but with km range + closure rate instead of target_size + distance_estimate
void compute_obs_realistic_range(Dogfight *env) {
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

    // Range in km (0 = point blank, 0.5 = 1km, 1.0 = 2km+)
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect (are they facing toward/away from us?)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);  // 1 = head-on, -1 = tail

    // Horizon visible (is up vector pointing up?)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;  // 1 = level, 0 = knife-edge, -1 = inverted

    // Closure rate (positive = closing)
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = norm3(p->vel) * INV_MAX_SPEED;  // Airspeed
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;     // Altitude
    env->observations[i++] = pitch / (PI * 0.5f);            // Pitch indicator
    env->observations[i++] = roll / PI;                       // Bank indicator

    // Gunsight (3 obs)
    env->observations[i++] = target_az / PI;                  // Target azimuth in sight
    env->observations[i++] = target_el / (PI * 0.5f);         // Target elevation in sight
    env->observations[i++] = range_km;                        // Range: 0=close, 1=2km+

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;                   // -1 to 1
    env->observations[i++] = horizon_visible;                 // -1 to 1
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);  // Closure rate
    // OBS_SIZE = 10
}

// Scheme 4: REALISTIC_ENEMY_STATE (13 obs)
// REALISTIC_RANGE + enemy pitch/roll/heading
void compute_obs_realistic_enemy_state(Dogfight *env) {
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

    // Range in km
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Horizon visible
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    // Enemy Euler angles (relative to horizon)
    float enemy_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float enemy_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                              1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));

    // Enemy heading relative to player (+1 = pointing at player, -1 = pointing away)
    float enemy_heading_rel = target_aspect;  // Already computed as dot(opp_fwd, to_player)

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = norm3(p->vel) * INV_MAX_SPEED;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = pitch / (PI * 0.5f);
    env->observations[i++] = roll / PI;

    // Gunsight (3 obs)
    env->observations[i++] = target_az / PI;
    env->observations[i++] = target_el / (PI * 0.5f);
    env->observations[i++] = range_km;

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;
    env->observations[i++] = horizon_visible;
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);

    // Enemy state (3 obs) - NEW
    env->observations[i++] = enemy_pitch / (PI * 0.5f);  // Enemy nose angle vs horizon
    env->observations[i++] = enemy_roll / PI;             // Enemy bank angle vs horizon
    env->observations[i++] = enemy_heading_rel;           // Pointing toward/away
    // OBS_SIZE = 13
}

// Scheme 5: REALISTIC_FULL (15 obs)
// REALISTIC_ENEMY_STATE + turn rate + G-loading
void compute_obs_realistic_full(Dogfight *env) {
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

    // Range in km
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Horizon visible
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    // Enemy Euler angles
    float enemy_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float enemy_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                              1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));
    float enemy_heading_rel = target_aspect;

    // Actual turn rate and G-loading from velocity change (v²/r method)
    // accel = (vel - prev_vel) / dt, centripetal = component perpendicular to vel
    float speed = norm3(p->vel);
    Vec3 accel = mul3(sub3(p->vel, p->prev_vel), 1.0f / DT);  // Actual acceleration

    // Decompose acceleration into forward (tangential) and perpendicular (centripetal)
    float turn_rate_actual = 0.0f;
    float g_loading = 1.0f;  // 1G = level flight
    if (speed > 10.0f) {
        Vec3 vel_dir = mul3(p->vel, 1.0f / speed);  // Normalized velocity
        float accel_forward = dot3(accel, vel_dir);    // Tangential component
        Vec3 accel_centripetal = sub3(accel, mul3(vel_dir, accel_forward));
        float centripetal_mag = norm3(accel_centripetal);

        // Turn rate = centripetal_accel / speed (from v²/r, so ω = a/v)
        turn_rate_actual = centripetal_mag / speed;

        // G-loading = total lateral acceleration / g (includes lift component)
        // Add 1G for gravity compensation in level flight
        g_loading = sqrtf(1.0f + (centripetal_mag * centripetal_mag) / (9.81f * 9.81f));
    }
    // Normalize turn rate: max ~0.5 rad/s (29°/s) for sustained turn
    float turn_rate_norm = clampf(turn_rate_actual / 0.5f, -1.0f, 1.0f);
    // Normalize G-loading: 0 = 1G, 1 = 9G
    float g_loading_norm = clampf((g_loading - 1.0f) / 8.0f, 0.0f, 1.0f);

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = speed * INV_MAX_SPEED;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = pitch / (PI * 0.5f);
    env->observations[i++] = roll / PI;

    // Gunsight (3 obs)
    env->observations[i++] = target_az / PI;
    env->observations[i++] = target_el / (PI * 0.5f);
    env->observations[i++] = range_km;

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;
    env->observations[i++] = horizon_visible;
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);

    // Enemy state (3 obs)
    env->observations[i++] = enemy_pitch / (PI * 0.5f);
    env->observations[i++] = enemy_roll / PI;
    env->observations[i++] = enemy_heading_rel;

    // Own state (2 obs) - NEW
    env->observations[i++] = turn_rate_norm;    // How fast am I turning?
    env->observations[i++] = g_loading_norm;    // How hard am I pulling?
    // OBS_SIZE = 15
}

// Dispatcher function
void compute_observations(Dogfight *env) {
    switch (env->obs_scheme) {
        case OBS_ANGLES:               compute_obs_angles(env); break;
        case OBS_CONTROL_ERROR:        compute_obs_control_error(env); break;
        case OBS_REALISTIC:            compute_obs_realistic(env); break;
        case OBS_REALISTIC_RANGE:      compute_obs_realistic_range(env); break;
        case OBS_REALISTIC_ENEMY_STATE: compute_obs_realistic_enemy_state(env); break;
        case OBS_REALISTIC_FULL:       compute_obs_realistic_full(env); break;
        default:                       compute_obs_angles(env); break;
    }
}

// ============================================================================
// Curriculum Learning: Stage-specific spawn functions
// ============================================================================

// Get current curriculum stage based on total episodes or random (for eval)
CurriculumStage get_curriculum_stage(Dogfight *env) {
    if (!env->curriculum_enabled) return CURRICULUM_FULL_RANDOM;
    if (env->curriculum_randomize) {
        // Random stage for eval mode - tests all difficulties
        return (CurriculumStage)(rand() % CURRICULUM_COUNT);
    }
    // Progressive stage for training
    int stage_idx = env->total_episodes / env->episodes_per_stage;
    if (stage_idx >= CURRICULUM_COUNT) stage_idx = CURRICULUM_COUNT - 1;
    return (CurriculumStage)stage_idx;
}

// Stage 0: TAIL_CHASE - Opponent ahead, same heading (easiest)
void spawn_tail_chase(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m directly ahead, same velocity direction
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-50, 50),
        player_pos.z + rndf(-30, 30)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 1: HEAD_ON - Opponent coming toward us
void spawn_head_on(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 400-600m ahead, facing us (opposite velocity)
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(400, 600),
        player_pos.y + rndf(-50, 50),
        player_pos.z + rndf(-30, 30)
    );
    Vec3 opp_vel = vec3(-player_vel.x, -player_vel.y, player_vel.z);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 2: CROSSING - 90 degree deflection shots
void spawn_crossing(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 300-500m to the side, flying perpendicular
    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(100, 200),
        player_pos.y + side * rndf(300, 500),
        player_pos.z + rndf(-50, 50)
    );
    // Perpendicular velocity (flying in Y direction)
    Vec3 opp_vel = vec3(0, -side * norm3(player_vel), 0);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 3: VERTICAL - Above or below player
void spawn_vertical(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead, 200-400m above OR below
    float vert = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float alt_offset = vert * rndf(200, 400);
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-50, 50),
        clampf(player_pos.z + alt_offset, 300, 2500)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_LEVEL;  // Maintain altitude
}

// Stage 4: MANEUVERING - Opponent does turns
void spawn_maneuvering(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Random spawn position (similar to original)
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 500),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    // Randomly choose turn direction
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = rndf(0.3f, 0.6f);  // 17-34 degrees
}

// Stage 5: FULL_RANDOM - Maximum difficulty (360° spawn + random heading)
void spawn_full_random(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Random direction in 3D sphere (300-600m from player)
    float dist = rndf(300, 600);
    float theta = rndf(0, 2.0f * M_PI);      // Azimuth: 0-360°
    float phi = rndf(-0.3f, 0.3f);           // Elevation: ±17° (keep near level)

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 2500)
    );

    // Random velocity direction (not necessarily toward/away from player)
    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);

    // Set orientation to match velocity direction (yaw rotation around Z)
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // Use autopilot randomization (if configured)
    if (env->opponent_ap.randomize_on_reset) {
        autopilot_randomize(&env->opponent_ap);
    } else {
        // Default: uniform random mode
        float r = rndf(0, 1);
        if (r < 0.2f) env->opponent_ap.mode = AP_STRAIGHT;
        else if (r < 0.4f) env->opponent_ap.mode = AP_LEVEL;
        else if (r < 0.6f) env->opponent_ap.mode = AP_TURN_LEFT;
        else if (r < 0.8f) env->opponent_ap.mode = AP_TURN_RIGHT;
        else env->opponent_ap.mode = AP_CLIMB;
    }
}

// Master spawn function: dispatches to stage-specific spawner
void spawn_by_curriculum(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    CurriculumStage new_stage = get_curriculum_stage(env);

    // Log stage transitions
    if (new_stage != env->stage) {
        printf("[Curriculum] Episode %d: Stage %d -> %d\n",
               env->total_episodes, env->stage, new_stage);
        env->stage = new_stage;
    }

    switch (env->stage) {
        case CURRICULUM_TAIL_CHASE:   spawn_tail_chase(env, player_pos, player_vel); break;
        case CURRICULUM_HEAD_ON:      spawn_head_on(env, player_pos, player_vel); break;
        case CURRICULUM_CROSSING:     spawn_crossing(env, player_pos, player_vel); break;
        case CURRICULUM_VERTICAL:     spawn_vertical(env, player_pos, player_vel); break;
        case CURRICULUM_MANEUVERING:  spawn_maneuvering(env, player_pos, player_vel); break;
        case CURRICULUM_FULL_RANDOM:
        default:                      spawn_full_random(env, player_pos, player_vel); break;
    }

    // Reset autopilot PID state after spawning
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;
}

// Legacy spawn (for curriculum_enabled=0)
void spawn_legacy(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 500),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);

    // Handle autopilot: randomize if configured, reset PID state
    if (env->opponent_ap.randomize_on_reset) {
        autopilot_randomize(&env->opponent_ap);
    }
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;
}

// ============================================================================

void c_reset(Dogfight *env) {
    // Increment total episodes BEFORE determining stage (so first episode is 0)
    env->total_episodes++;

    env->tick = 0;
    env->episode_return = 0.0f;

    // Clear episode tracking
    env->kill = 0;
    env->episode_shots_fired = 0.0f;

    // Recompute gun cone trig (for curriculum: could vary gun_cone_angle here)
    env->cos_gun_cone = cosf(env->gun_cone_angle);
    env->cos_gun_cone_2x = cosf(env->gun_cone_angle * 2.0f);

    // Spawn player at random position
    Vec3 pos = vec3(rndf(-500, 500), rndf(-500, 500), rndf(500, 1500));
    Vec3 vel = vec3(80, 0, 0);
    reset_plane(&env->player, pos, vel);

    // Spawn opponent based on curriculum stage (or legacy if disabled)
    if (env->curriculum_enabled) {
        spawn_by_curriculum(env, pos, vel);
    } else {
        spawn_legacy(env, pos, vel);
    }

    if (DEBUG) printf("=== RESET ===\n");
    if (DEBUG) printf("kill=%d, episode_shots_fired=%.0f (now cleared)\n", env->kill, env->episode_shots_fired);
    if (DEBUG) printf("player_pos=(%.1f, %.1f, %.1f)\n", pos.x, pos.y, pos.z);
    if (DEBUG) printf("player_vel=(%.1f, %.1f, %.1f) speed=%.1f\n", vel.x, vel.y, vel.z, norm3(vel));
    if (DEBUG) printf("opponent_pos=(%.1f, %.1f, %.1f)\n", env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z);
    if (DEBUG) printf("initial_dist=%.1f m, stage=%d\n", norm3(sub3(env->opponent.pos, pos)), env->stage);

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

    // 6. Roll penalty: gentle preference for level flight
    float roll_angle = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                              1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));
    float r_roll = -fabsf(roll_angle) * env->rcfg.roll;
    reward += r_roll;

    // 7. Negative G penalty: penalize low/negative G-loading
    // Threshold 0.5G: allows some slack for light maneuvers but penalizes serious neg-G
    float g_threshold = 0.5f;
    float g_deficit = fmaxf(0.0f, g_threshold - p->g_force);
    float r_neg_g = -g_deficit * env->rcfg.neg_g;
    reward += r_neg_g;

    // 8. Rudder penalty: discourage excessive rudder use
    float r_rudder = -fabsf(env->actions[3]) * env->rcfg.rudder;
    reward += r_rudder;

    // 9. Aiming reward: feedback for gun alignment before actual hits
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
    if (DEBUG) printf("r_roll=%.5f (roll=%.1f deg)\n", r_roll, roll_angle * RAD_TO_DEG);
    if (DEBUG) printf("r_neg_g=%.5f (g=%.2f)\n", r_neg_g, p->g_force);
    if (DEBUG) printf("r_rudder=%.5f (rud=%.2f)\n", r_rudder, env->actions[3]);
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

    // Set clip planes for long-range visibility (default far=1000 is too close)
    rlSetClipPlanes(1.0, 10000.0);  // near=1m, far=10km
    BeginMode3D(env->client->camera);

    // 6. Draw ground plane at z=0 (XY plane, since we use Z-up)
    // DrawPlane uses raylib's Y-up convention (XZ plane), so we draw triangles instead
    Vector3 g1 = {-2000, -2000, 0};
    Vector3 g2 = {2000, -2000, 0};
    Vector3 g3 = {2000, 2000, 0};
    Vector3 g4 = {-2000, 2000, 0};
    Color ground_color = (Color){20, 60, 20, 255};
    DrawTriangle3D(g1, g2, g3, ground_color);
    DrawTriangle3D(g1, g3, g4, ground_color);

    // 7. Draw world bounds wireframe
    // Bounds: X +/-2000, Y +/-2000, Z 0-3000 -> center at (0, 0, 1500)
    DrawCubeWires((Vector3){0, 0, 1500}, 4000, 4000, 3000, (Color){100, 100, 100, 255});

    // 8. Draw player plane (cyan wireframe airplane)
    Color cyan = {0, 255, 255, 255};
    Color light_cyan = {100, 255, 255, 255};
    draw_plane_shape(p->pos, p->ori, cyan, light_cyan);

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
