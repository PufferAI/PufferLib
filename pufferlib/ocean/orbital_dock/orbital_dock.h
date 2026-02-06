#ifndef ORBITAL_DOCK_H
#define ORBITAL_DOCK_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"

// Debug: print first N episodes (set to 0 to disable)
static int g_debug_episodes = 0;
#define DEBUG_MAX_EPISODES 0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Colors for rendering
static const Color PUFF_CYAN = {0, 187, 187, 255};
static const Color PUFF_WHITE = {241, 241, 241, 255};
static const Color PUFF_BACKGROUND = {6, 24, 24, 255};
static const Color PUFF_YELLOW = {255, 200, 0, 255};
static const Color PUFF_RED = {187, 0, 0, 255};
static const Color PUFF_ORANGE = {255, 128, 0, 255};

// ============================================================================
// Vector Math
// ============================================================================

typedef struct {
    double x, y, z;
} Vec3d;

static inline Vec3d vec3d(double x, double y, double z) {
    return (Vec3d){x, y, z};
}

static inline Vec3d add3d(Vec3d a, Vec3d b) {
    return (Vec3d){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3d sub3d(Vec3d a, Vec3d b) {
    return (Vec3d){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3d scale3d(Vec3d a, double s) {
    return (Vec3d){a.x * s, a.y * s, a.z * s};
}

static inline double dot3d(Vec3d a, Vec3d b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3d cross3d(Vec3d a, Vec3d b) {
    return (Vec3d){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline double norm3d(Vec3d a) {
    return sqrt(dot3d(a, a));
}

static inline Vec3d normalize3d(Vec3d a) {
    double n = norm3d(a);
    if (n > 1e-12) {
        return scale3d(a, 1.0 / n);
    }
    return (Vec3d){0, 0, 1};  // Default to Z-up if degenerate
}

// ============================================================================
// Log Struct (all floats, ending with n)
// ============================================================================

typedef struct {
    float episode_return;
    float episode_length;
    float dock_success;
    float crash_rate;
    float deorbit_rate;
    float escape_rate;
    float timeout_rate;
    float fuel_used;
    float final_distance;
    float final_rel_speed;
    float curriculum_stage;
    float n;  // Required as last field
} Log;

// ============================================================================
// Render Client Struct (used for visualization)
// ============================================================================

#define RENDER_WIDTH 1080
#define RENDER_HEIGHT 720
#define TRAIL_LENGTH 256

typedef struct {
    Vec3d pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct Client {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;

    Trail trail;
    float scale;  // meters per render unit
    int last_step;  // for tracking episode resets
} Client;

// ============================================================================
// Environment Struct
// ============================================================================

typedef struct {
    Log log;                     // Required field (first)
    float* observations;         // Required field - 14 floats
    float* actions;              // Required field - 3 floats (continuous velocity commands)
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    // Chaser state (inertial frame, SI units)
    double cx, cy, cz;           // Position (m)
    double cvx, cvy, cvz;        // Velocity (m/s)

    // Station state (inertial frame)
    double tx, ty, tz;           // Position (m)
    double tvx, tvy, tvz;        // Velocity (m/s)

    // Station orbital parameters (for analytical propagation)
    double t_omega;              // Angular velocity (rad/s)
    double t_hx, t_hy, t_hz;     // Angular momentum direction (unit vector)
    double t_radius;             // Station orbital radius (m)

    // Environment state
    double fuel;                 // Remaining delta-v (m/s)
    double init_dist;            // Initial distance for normalization
    double prev_dist;            // Previous distance for shaping
    double prev_speed;           // Previous relative speed for velocity shaping
    int step_count;
    int max_steps;

    // Physics config
    double mu;                   // Gravitational parameter (m^3/s^2)
    double station_radius;       // Station orbit radius (m)
    double max_thrust;           // Max thrust per axis (N)
    double mass;                 // Chaser mass (kg)
    double dt;                   // Timestep (s)
    double fuel_budget;          // Total delta-v budget (m/s)

    // Hierarchical velocity control (Hovell & Ulrich 2021)
    double kp;                   // P controller gain
    double max_cmd_vel;          // Max commanded velocity (m/s)

    // Docking conditions
    double dock_dist;            // Docking distance threshold (m)
    double dock_speed;           // Docking speed threshold (m/s)

    // Termination thresholds
    double earth_radius;         // Earth radius (m)
    double deorbit_alt;          // Deorbit altitude (m from surface)
    double escape_alt;           // Escape altitude (m from surface)

    // Difficulty / randomization
    double difficulty;           // 0.0 to 1.0
    double alt_offset_max;       // Max altitude offset (m)
    double phase_offset_max;     // Max phase offset (rad)
    double incl_offset_max;      // Max inclination offset (rad)
    double vel_perturb_max;      // Max velocity perturbation (m/s)

    // Reward weights
    double rw_dock;
    double rw_dist_shaping;
    double rw_closing;
    double rw_vel_match;
    double rw_fuel;
    double rw_crash;
    double rw_deorbit;
    double rw_escape;
    double rw_plane_align;
    double rw_node_timing;

    // RNG state
    unsigned int rng_state;

    // Curriculum state
    int curriculum_stage;        // 0=free docks, 1=easy, 2=medium, 3=full
    int curriculum_docks;        // Dock count in current window
    int curriculum_episodes;     // Episode count in current window
    int curriculum_window;       // Window size for advancement check

    // Render client (NULL if not rendering)
    Client *client;
} OrbitalDock;

// ============================================================================
// Random Number Generation (xorshift32)
// ============================================================================

static inline unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline double rndf(OrbitalDock* env) {
    // Use xorshift32 for better distribution with sequential seeds
    return (double)xorshift32(&env->rng_state) / (double)0xFFFFFFFF;
}

static inline double rndf_range(OrbitalDock* env, double min, double max) {
    return min + rndf(env) * (max - min);
}

// ============================================================================
// Physics Functions
// ============================================================================

// Rodrigues rotation: rotate vector v by angle theta around unit axis k
static Vec3d rodrigues_rotate(Vec3d v, Vec3d k, double theta) {
    double cos_t = cos(theta);
    double sin_t = sin(theta);
    Vec3d k_cross_v = cross3d(k, v);
    double k_dot_v = dot3d(k, v);
    return add3d(
        add3d(scale3d(v, cos_t), scale3d(k_cross_v, sin_t)),
        scale3d(k, k_dot_v * (1.0 - cos_t))
    );
}

// Compute LVLH basis vectors at given position/velocity
// r_hat: Radial (away from Earth)
// v_hat: Prograde (along velocity)
// h_hat: Normal (angular momentum direction)
static void compute_lvlh(Vec3d pos, Vec3d vel, Vec3d* r_hat, Vec3d* v_hat, Vec3d* h_hat) {
    *r_hat = normalize3d(pos);
    Vec3d h = cross3d(pos, vel);
    *h_hat = normalize3d(h);
    *v_hat = cross3d(*h_hat, *r_hat);
}

// Compute gravitational acceleration
static Vec3d compute_gravity(OrbitalDock* env, Vec3d pos) {
    double r = norm3d(pos);
    double r3 = r * r * r;
    return scale3d(pos, -env->mu / r3);
}

// Integrate using Velocity Verlet (Störmer-Verlet) - second order, symplectic
// Error is O(dt³) per step instead of O(dt²) for Euler
static void integrate_verlet(OrbitalDock* env,
                             double* px, double* py, double* pz,
                             double* vx, double* vy, double* vz,
                             Vec3d thrust) {
    Vec3d pos = vec3d(*px, *py, *pz);
    Vec3d vel = vec3d(*vx, *vy, *vz);

    // Acceleration at current position
    Vec3d a_old = add3d(compute_gravity(env, pos), thrust);

    // Update position: r += v*dt + 0.5*a*dt²
    pos = add3d(pos, add3d(scale3d(vel, env->dt),
                           scale3d(a_old, 0.5 * env->dt * env->dt)));

    // Acceleration at new position
    Vec3d a_new = add3d(compute_gravity(env, pos), thrust);

    // Update velocity: v += 0.5*(a_old + a_new)*dt
    vel = add3d(vel, scale3d(add3d(a_old, a_new), 0.5 * env->dt));

    *px = pos.x; *py = pos.y; *pz = pos.z;
    *vx = vel.x; *vy = vel.y; *vz = vel.z;
}

// Integrate chaser dynamics with thrust
static void integrate_chaser(OrbitalDock* env, Vec3d thrust_inertial) {
    Vec3d thrust_accel = scale3d(thrust_inertial, 1.0 / env->mass);
    integrate_verlet(env, &env->cx, &env->cy, &env->cz,
                         &env->cvx, &env->cvy, &env->cvz, thrust_accel);
}

// Integrate station dynamics (same method as chaser, so errors cancel in relative frame)
static void propagate_station(OrbitalDock* env) {
    Vec3d zero_thrust = vec3d(0, 0, 0);
    integrate_verlet(env, &env->tx, &env->ty, &env->tz,
                         &env->tvx, &env->tvy, &env->tvz, zero_thrust);
}

// ============================================================================
// Observation Computation
// ============================================================================

static void compute_observations(OrbitalDock* env) {
    // Get chaser and station states
    Vec3d c_pos = vec3d(env->cx, env->cy, env->cz);
    Vec3d c_vel = vec3d(env->cvx, env->cvy, env->cvz);
    Vec3d t_pos = vec3d(env->tx, env->ty, env->tz);
    Vec3d t_vel = vec3d(env->tvx, env->tvy, env->tvz);

    // Compute station's LVLH frame (Hill frame)
    Vec3d r_hat, v_hat, h_hat;
    compute_lvlh(t_pos, t_vel, &r_hat, &v_hat, &h_hat);

    // Relative position/velocity in inertial
    Vec3d rel_pos = sub3d(c_pos, t_pos);
    Vec3d rel_vel = sub3d(c_vel, t_vel);

    // Project to LVLH (Hill) frame
    double rel_r = dot3d(rel_pos, r_hat);  // R-bar (radial)
    double rel_v = dot3d(rel_pos, v_hat);  // V-bar (prograde)
    double rel_h = dot3d(rel_pos, h_hat);  // H-bar (normal)
    double rel_vr = dot3d(rel_vel, r_hat);
    double rel_vv = dot3d(rel_vel, v_hat);
    double rel_vh = dot3d(rel_vel, h_hat);

    // Distance and closing speed
    double dist = norm3d(rel_pos);
    double closing_speed = (dist > 1e-10) ? -dot3d(normalize3d(rel_pos), rel_vel) : 0.0;

    // Circular velocity at station orbit for normalization
    double v_circ = sqrt(env->mu / env->station_radius);

    // Altitude normalization
    double c_alt = norm3d(c_pos) - env->earth_radius;
    double t_alt = env->station_radius - env->earth_radius;
    double alt_diff = (env->alt_offset_max > 0) ? (c_alt - t_alt) / env->alt_offset_max : 0.0;

    // Phase angle (in-plane angular separation)
    // Project positions onto station's orbital plane
    Vec3d c_proj = sub3d(c_pos, scale3d(h_hat, dot3d(c_pos, h_hat)));
    Vec3d t_proj = t_pos;  // Station is already in its plane
    double phase_angle = 0.0;
    double c_proj_norm = norm3d(c_proj);
    double t_proj_norm = norm3d(t_proj);
    if (c_proj_norm > 1e-10 && t_proj_norm > 1e-10) {
        Vec3d c_proj_n = scale3d(c_proj, 1.0 / c_proj_norm);
        Vec3d t_proj_n = scale3d(t_proj, 1.0 / t_proj_norm);
        phase_angle = atan2(
            dot3d(cross3d(t_proj_n, c_proj_n), h_hat),
            dot3d(t_proj_n, c_proj_n)
        ) / M_PI;
    }

    // Inclination difference
    Vec3d c_h = normalize3d(cross3d(c_pos, c_vel));
    Vec3d t_h = vec3d(env->t_hx, env->t_hy, env->t_hz);
    double cos_incl = dot3d(c_h, t_h);
    cos_incl = fmax(-1.0, fmin(1.0, cos_incl));
    double incl_diff = acos(cos_incl);
    double incl_diff_norm = (env->incl_offset_max > 0) ? incl_diff / env->incl_offset_max : 0.0;

    // Node angle (angle to ascending/descending node)
    double node_angle = 0.0;
    Vec3d node_line = cross3d(c_h, t_h);
    double node_norm = norm3d(node_line);
    if (node_norm > 1e-10) {
        node_line = scale3d(node_line, 1.0 / node_norm);
        Vec3d c_pos_n = normalize3d(c_pos);
        node_angle = atan2(
            dot3d(cross3d(c_pos_n, node_line), c_h),
            dot3d(c_pos_n, node_line)
        ) / M_PI;
    }

    // Normalization scales - tuned for close-range docking (30-50m scenarios at d=0)
    double pos_scale = 100.0;   // 100m reference - 50m = 0.5, 100m = 1.0
    double vel_scale = 2.0;     // 2 m/s reference - 0.5 m/s = 0.25, 1 m/s = 0.5

    // Fill observation buffer (clamped to declared bounds [-10, 10])
    int idx = 0;
    env->observations[idx++] = (float)fmax(-10.0, fmin(10.0, rel_r / pos_scale));      // 0: rel_x (R-bar)
    env->observations[idx++] = (float)fmax(-10.0, fmin(10.0, rel_v / pos_scale));      // 1: rel_y (V-bar)
    env->observations[idx++] = (float)fmax(-10.0, fmin(10.0, rel_h / pos_scale));      // 2: rel_z (H-bar)
    env->observations[idx++] = (float)fmax(-10.0, fmin(10.0, rel_vr / vel_scale));     // 3: rel_vx
    env->observations[idx++] = (float)fmax(-10.0, fmin(10.0, rel_vv / vel_scale));     // 4: rel_vy
    env->observations[idx++] = (float)fmax(-10.0, fmin(10.0, rel_vh / vel_scale));     // 5: rel_vz
    env->observations[idx++] = (float)fmin(10.0, dist / pos_scale);                     // 6: dist_norm (always positive)
    env->observations[idx++] = (float)fmax(-10.0, fmin(10.0, closing_speed / vel_scale)); // 7: closing_speed
    env->observations[idx++] = (float)fmax(0.0, fmin(1.0, env->fuel / env->fuel_budget)); // 8: fuel_remaining
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, alt_diff));               // 9: orbit_alt_norm
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, phase_angle));            // 10: phase_angle
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, incl_diff_norm));         // 11: inclination_diff
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, node_angle));             // 12: node_angle
    env->observations[idx++] = (float)fmax(0.0, fmin(1.0, 1.0 - (double)env->step_count / env->max_steps)); // 13: time_remaining
}

// ============================================================================
// Curriculum Learning
// ============================================================================

// Stage parameters: {dist_min, dist_max, vel_min, vel_max, offset_max, target_dock_rate}
// 5 stages with <=2.5x distance jump between stages for smooth progression
static const double CURRICULUM_PARAMS[5][6] = {
    {5.0,   20.0,  0.2, 0.6, 3.0,  0.40},  // Stage 0: Bootstrap (5-20m)
    {15.0,  50.0,  0.3, 0.8, 5.0,  0.35},  // Stage 1: Short range (15-50m)
    {30.0, 100.0,  0.3, 1.0, 8.0,  0.30},  // Stage 2: Medium range (30-100m)
    {80.0, 250.0,  0.4, 1.5, 15.0, 0.25},  // Stage 3: Extended range (80-250m)
    {150.0,500.0,  0.5, 2.0, 20.0, 0.20},  // Stage 4: Full range (150-500m)
};

// Sample initial conditions with stage mixing (30% from previous stage)
static void sample_initial_conditions(OrbitalDock* env,
                                       double* approach_dist,
                                       double* closing_speed,
                                       double* offset_max) {
    int stage = env->curriculum_stage;

    // Stage mixing: 30% chance to sample from previous stage
    // This prevents catastrophic forgetting during transitions
    double stage_mix = rndf(env);
    if (stage > 0 && stage_mix < 0.3) {
        stage = stage - 1;  // Sample from easier stage
    }

    double dist_min = CURRICULUM_PARAMS[stage][0];
    double dist_max = CURRICULUM_PARAMS[stage][1];
    double vel_min = CURRICULUM_PARAMS[stage][2];
    double vel_max = CURRICULUM_PARAMS[stage][3];
    *offset_max = CURRICULUM_PARAMS[stage][4];

    *approach_dist = dist_min + (dist_max - dist_min) * rndf(env);
    *closing_speed = vel_min + (vel_max - vel_min) * rndf(env);
}

// Global curriculum tracking - implemented in binding.c
// Uses shared state across all envs for stable advancement
void global_curriculum_update(OrbitalDock* env, int docked);

// ============================================================================
// Reset
// ============================================================================

void c_reset(OrbitalDock* env) {
    env->step_count = 0;

    // Initialize xorshift RNG from global rand() (which has been seeded by vec_reset)
    // Use multiple rand() calls to get better entropy
    env->rng_state = (unsigned int)rand() ^ ((unsigned int)rand() << 15);
    if (env->rng_state == 0) env->rng_state = 1;  // xorshift needs non-zero state

    // Station in circular orbit at configured radius
    double r = env->station_radius;
    double v_circ = sqrt(env->mu / r);
    env->t_omega = v_circ / r;  // Angular velocity

    // Station starts at (r, 0, 0) moving in +Y direction (in XY plane)
    env->tx = r; env->ty = 0; env->tz = 0;
    env->tvx = 0; env->tvy = v_circ; env->tvz = 0;
    env->t_hx = 0; env->t_hy = 0; env->t_hz = 1.0;  // Angular momentum: +Z
    env->t_radius = r;

    // Get station's LVLH basis vectors
    Vec3d t_pos = vec3d(env->tx, env->ty, env->tz);
    Vec3d t_vel = vec3d(env->tvx, env->tvy, env->tvz);
    Vec3d r_hat, v_hat, h_hat;
    compute_lvlh(t_pos, t_vel, &r_hat, &v_hat, &h_hat);

    // === CURRICULUM-BASED INITIAL CONDITIONS ===
    double approach_dist, closing_speed, offset_max;
    sample_initial_conditions(env, &approach_dist, &closing_speed, &offset_max);

    // Random offsets in R-bar and H-bar
    double r_offset = (rndf(env) - 0.5) * 2.0 * offset_max;
    double h_offset = (rndf(env) - 0.5) * 2.0 * offset_max;

    // LVLH position: primarily V-bar (behind station), with offsets
    double lvlh_r = r_offset;
    double lvlh_v = -approach_dist;  // Negative = behind station
    double lvlh_h = h_offset;

    // Convert LVLH offset to inertial position
    Vec3d offset = add3d(add3d(scale3d(r_hat, lvlh_r), scale3d(v_hat, lvlh_v)), scale3d(h_hat, lvlh_h));
    Vec3d c_pos = add3d(t_pos, offset);
    env->cx = c_pos.x; env->cy = c_pos.y; env->cz = c_pos.z;

    // Initial velocity: co-moving + closing velocity toward station
    Vec3d c_vel = t_vel;
    // Closing velocity is along +V-bar (toward station)
    c_vel = add3d(c_vel, scale3d(v_hat, closing_speed));

    env->cvx = c_vel.x; env->cvy = c_vel.y; env->cvz = c_vel.z;

    // Initialize fuel and distance tracking
    env->fuel = env->fuel_budget;
    Vec3d rel = sub3d(vec3d(env->cx, env->cy, env->cz), vec3d(env->tx, env->ty, env->tz));
    Vec3d rel_vel = sub3d(vec3d(env->cvx, env->cvy, env->cvz), vec3d(env->tvx, env->tvy, env->tvz));
    env->init_dist = norm3d(rel);
    if (env->init_dist < 1.0) env->init_dist = 1.0;  // Prevent division by zero
    env->prev_dist = env->init_dist;
    env->prev_speed = norm3d(rel_vel);

    // === DYNAMIC MAX_STEPS ===
    // Scale max_steps with initial distance so agent has time to reach station
    // Base: 100 steps, plus 3 steps per meter of initial distance
    int base_steps = 100;
    int steps_per_meter = 3;
    env->max_steps = base_steps + (int)(env->init_dist * steps_per_meter);
    // At 20m:  100 + 60  = 160 steps
    // At 100m: 100 + 300 = 400 steps
    // At 500m: 100 + 1500 = 1600 steps

    compute_observations(env);
}

// ============================================================================
// Step
// ============================================================================

void c_step(OrbitalDock* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->step_count++;

    // 1. Get STATION'S LVLH basis (not chaser's!)
    Vec3d t_pos = vec3d(env->tx, env->ty, env->tz);
    Vec3d t_vel = vec3d(env->tvx, env->tvy, env->tvz);
    Vec3d r_hat, v_hat, h_hat;
    compute_lvlh(t_pos, t_vel, &r_hat, &v_hat, &h_hat);

    // 2. Get current relative velocity in LVLH frame
    Vec3d c_pos = vec3d(env->cx, env->cy, env->cz);
    Vec3d c_vel = vec3d(env->cvx, env->cvy, env->cvz);
    Vec3d rel_vel = sub3d(c_vel, t_vel);
    double current_vel_r = dot3d(rel_vel, r_hat);
    double current_vel_v = dot3d(rel_vel, v_hat);
    double current_vel_h = dot3d(rel_vel, h_hat);

    // 3. Read desired velocity from continuous actions (clamp to bounds)
    double desired_vel_r = fmax(-env->max_cmd_vel, fmin(env->max_cmd_vel, (double)env->actions[0]));
    double desired_vel_v = fmax(-env->max_cmd_vel, fmin(env->max_cmd_vel, (double)env->actions[1]));
    double desired_vel_h = fmax(-env->max_cmd_vel, fmin(env->max_cmd_vel, (double)env->actions[2]));

    // 4. P Controller: thrust = kp * (desired_vel - current_vel)
    // Hovell & Ulrich (2021) Eq. (14): u_t = K_p * (v_t - x_dot_t)
    double vel_error_r = desired_vel_r - current_vel_r;
    double vel_error_v = desired_vel_v - current_vel_v;
    double vel_error_h = desired_vel_h - current_vel_h;

    double thrust_r = env->kp * vel_error_r * env->mass;  // F = m * a, a = kp * error
    double thrust_v = env->kp * vel_error_v * env->mass;
    double thrust_h = env->kp * vel_error_h * env->mass;

    // Clamp thrust to physical limits
    thrust_r = fmax(-env->max_thrust, fmin(env->max_thrust, thrust_r));
    thrust_v = fmax(-env->max_thrust, fmin(env->max_thrust, thrust_v));
    thrust_h = fmax(-env->max_thrust, fmin(env->max_thrust, thrust_h));

    // LVLH thrust vector -> inertial frame
    Vec3d thrust_lvlh = add3d(
        add3d(scale3d(r_hat, thrust_r), scale3d(v_hat, thrust_v)),
        scale3d(h_hat, thrust_h)
    );

    // 3. Compute fuel usage (delta-v)
    double thrust_mag = norm3d(thrust_lvlh);
    double dv_used = (thrust_mag / env->mass) * env->dt;
    double fuel_penalty = 0.0;
    if (env->fuel > 0 && thrust_mag > 0) {
        double actual_dv = fmin(dv_used, env->fuel);
        env->fuel -= actual_dv;
        fuel_penalty = -env->rw_fuel * (actual_dv / env->fuel_budget);
        // Scale thrust if we ran out of fuel mid-burn
        if (actual_dv < dv_used) {
            double scale = actual_dv / dv_used;
            thrust_lvlh = scale3d(thrust_lvlh, scale);
        }
    } else if (env->fuel <= 0) {
        thrust_lvlh = vec3d(0, 0, 0);  // Out of fuel, no thrust
    }

    // 4. Integrate chaser dynamics
    integrate_chaser(env, thrust_lvlh);

    // 5. Propagate station analytically
    propagate_station(env);

    // 6. Compute termination conditions
    // Re-read positions after integration
    t_pos = vec3d(env->tx, env->ty, env->tz);
    t_vel = vec3d(env->tvx, env->tvy, env->tvz);
    c_pos = vec3d(env->cx, env->cy, env->cz);
    c_vel = vec3d(env->cvx, env->cvy, env->cvz);

    Vec3d rel_pos = sub3d(c_pos, t_pos);
    rel_vel = sub3d(c_vel, t_vel);
    double dist = norm3d(rel_pos);
    double rel_speed = norm3d(rel_vel);
    double c_alt = norm3d(c_pos) - env->earth_radius;

    int deorbited = (c_alt < env->deorbit_alt);
    int escaped = (c_alt > env->escape_alt);
    int timeout = (env->step_count >= env->max_steps);

    // 7. Compute rewards - two-component shaping
    double reward = 0.0;

    // === Component 1: Weak linear shaping (long-range gradient) ===
    // Provides approach signal at ALL distances, even 500m.
    // Coefficient is small so cumulative doesn't drown terminal rewards.
    // From 500m: cumulative = 0.005 * 495 = 2.5
    double distance_progress = env->prev_dist - dist;
    reward += env->rw_dist_shaping * distance_progress;

    // === Component 2: Exponential near-dock shaping (bounded, drone-style) ===
    // exp(-dist/R) potential concentrates reward near station.
    // Total cumulative bounded at ~1.0 regardless of starting distance.
    // Uses rw_vel_match parameter as coefficient (repurposed).
    double exp_R = 10.0;  // Characteristic distance — signal within ~30m
    double exp_shaping = exp(-dist / exp_R) - exp(-env->prev_dist / exp_R);
    reward += env->rw_vel_match * exp_shaping;

    // === Proximity-scaled velocity damping ===
    double eta = 0.1;
    double vel_penalty = -env->rw_closing * rel_speed / (dist + eta);
    reward += vel_penalty;



    // Terminal rewards - graduated docking quality
    // dock_clean: dist < 5m, speed < 0.5 m/s  -> +10.0
    // dock_rough: dist < 5m, speed 0.5-1.0 m/s -> +3.0
    // crash:      dist < 5m, speed >= 1.0 m/s  -> -5.0
    int dock_clean = (dist < env->dock_dist) && (rel_speed < env->dock_speed);
    int dock_rough = (dist < env->dock_dist) && (rel_speed >= env->dock_speed) && (rel_speed < 1.0);
    int crash_dock = (dist < env->dock_dist) && (rel_speed >= 1.0);

    if (dock_clean) {
        reward += env->rw_dock;  // +10.0
        env->terminals[0] = 1;
        env->log.dock_success += 1.0f;
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    } else if (dock_rough) {
        reward += 0.5;  // Rough dock - acceptable but less than clean
        env->terminals[0] = 1;
        env->log.dock_success += 0.5f;  // Count as partial success
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    } else if (crash_dock) {
        reward -= env->rw_crash;  // Crash - too fast, apply penalty
        env->terminals[0] = 1;
        env->log.crash_rate += 1.0f;
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    } else if (deorbited) {
        reward -= env->rw_deorbit;
        env->terminals[0] = 1;
        env->log.deorbit_rate += 1.0f;
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    } else if (escaped) {
        reward -= env->rw_escape;
        env->terminals[0] = 1;
        env->log.escape_rate += 1.0f;
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    } else if (timeout) {
        reward -= 1.0;  // Timeout penalty - mission failed
        env->terminals[0] = 1;
        env->log.timeout_rate += 1.0f;
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    }

    env->rewards[0] = (float)reward;
    env->prev_dist = dist;
    env->prev_speed = rel_speed;

    // DEBUG: Print first N episodes
    if (g_debug_episodes < DEBUG_MAX_EPISODES) {
        // Get raw LVLH values (before scaling) for clarity
        Vec3d rel_pos_dbg = sub3d(c_pos, t_pos);
        Vec3d rel_vel_dbg = sub3d(c_vel, t_vel);
        Vec3d r_hat_dbg, v_hat_dbg, h_hat_dbg;
        compute_lvlh(t_pos, t_vel, &r_hat_dbg, &v_hat_dbg, &h_hat_dbg);
        double rel_r_dbg = dot3d(rel_pos_dbg, r_hat_dbg);
        double rel_v_dbg = dot3d(rel_pos_dbg, v_hat_dbg);
        double rel_h_dbg = dot3d(rel_pos_dbg, h_hat_dbg);
        double rel_vr_dbg = dot3d(rel_vel_dbg, r_hat_dbg);
        double rel_vv_dbg = dot3d(rel_vel_dbg, v_hat_dbg);
        double rel_vh_dbg = dot3d(rel_vel_dbg, h_hat_dbg);

        const char* terminal_str = "";
        if (dock_clean) terminal_str = "DOCK_CLEAN";
        else if (dock_rough) terminal_str = "DOCK_ROUGH";
        else if (crash_dock) terminal_str = "CRASH";
        else if (deorbited) terminal_str = "DEORBIT";
        else if (escaped) terminal_str = "ESCAPE";
        else if (timeout) terminal_str = "TIMEOUT";

        printf("EP%d STEP%d | obs:[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f] | "
               "dist=%.2f spd=%.2f | cmd_vel=[%.2f,%.2f,%.2f] | reward=%.4f | %s\n",
               g_debug_episodes, env->step_count,
               rel_r_dbg, rel_v_dbg, rel_h_dbg,  // Position in meters
               rel_vr_dbg, rel_vv_dbg, rel_vh_dbg,  // Velocity in m/s
               dist, rel_speed,
               desired_vel_r, desired_vel_v, desired_vel_h,  // Commanded velocity
               reward,
               terminal_str);

        if (env->terminals[0]) {
            printf("--- END EPISODE %d ---\n\n", g_debug_episodes);
            g_debug_episodes++;
        }
    }

    // Update log
    env->log.episode_return += (float)reward;
    if (env->terminals[0]) {
        env->log.episode_length += (float)env->step_count;
        env->log.fuel_used += (float)(env->fuel_budget - env->fuel);
        env->log.n += 1.0f;

        // Global curriculum tracking
        int docked = dock_clean || dock_rough;
        global_curriculum_update(env, docked);
        env->log.curriculum_stage += (float)env->curriculum_stage;

        c_reset(env);
    }

    compute_observations(env);
}

// ============================================================================
// Render (implemented in render.h, included separately)
// ============================================================================

// Forward declarations - implemented in render.h
Client* make_client(OrbitalDock *env);
void close_client(Client *client);
void render_orbital_dock(OrbitalDock *env, Client *client);

void c_render(OrbitalDock* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            return;
        }
    }
    render_orbital_dock(env, env->client);
}

void c_close(OrbitalDock* env) {
    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
}

#endif // ORBITAL_DOCK_H
