#ifndef ORBITAL_DOCK_H
#define ORBITAL_DOCK_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

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
    float n;  // Required as last field
} Log;

// ============================================================================
// Environment Struct
// ============================================================================

typedef struct {
    Log log;                     // Required field (first)
    float* observations;         // Required field - 14 floats
    int* actions;                // Required field - 3 ints (MultiDiscrete)
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
    int step_count;
    int max_steps;

    // Physics config
    double mu;                   // Gravitational parameter (m^3/s^2)
    double station_radius;       // Station orbit radius (m)
    double max_thrust;           // Max thrust per axis (N)
    double mass;                 // Chaser mass (kg)
    double dt;                   // Timestep (s)
    double fuel_budget;          // Total delta-v budget (m/s)

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

// Integrate chaser dynamics using semi-implicit Euler (symplectic)
static void integrate_chaser(OrbitalDock* env, Vec3d thrust_inertial) {
    Vec3d pos = vec3d(env->cx, env->cy, env->cz);
    Vec3d vel = vec3d(env->cvx, env->cvy, env->cvz);

    // Compute acceleration: gravity + thrust/mass
    Vec3d a_grav = compute_gravity(env, pos);
    Vec3d a_thrust = scale3d(thrust_inertial, 1.0 / env->mass);
    Vec3d a_total = add3d(a_grav, a_thrust);

    // Semi-implicit Euler: update velocity first, then position
    vel = add3d(vel, scale3d(a_total, env->dt));
    pos = add3d(pos, scale3d(vel, env->dt));

    env->cx = pos.x; env->cy = pos.y; env->cz = pos.z;
    env->cvx = vel.x; env->cvy = vel.y; env->cvz = vel.z;
}

// Propagate station analytically (circular orbit)
static void propagate_station(OrbitalDock* env) {
    double angle = env->t_omega * env->dt;
    Vec3d h = vec3d(env->t_hx, env->t_hy, env->t_hz);

    // Rotate position around angular momentum axis
    Vec3d pos = vec3d(env->tx, env->ty, env->tz);
    Vec3d new_pos = rodrigues_rotate(pos, h, angle);
    env->tx = new_pos.x;
    env->ty = new_pos.y;
    env->tz = new_pos.z;

    // Rotate velocity similarly
    Vec3d vel = vec3d(env->tvx, env->tvy, env->tvz);
    Vec3d new_vel = rodrigues_rotate(vel, h, angle);
    env->tvx = new_vel.x;
    env->tvy = new_vel.y;
    env->tvz = new_vel.z;
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

    // Normalization scales - use fixed reference scales for stable observations
    double pos_scale = 10000.0;  // 10km reference - keeps observations meaningful across distances
    double vel_scale = 100.0;    // 100 m/s reference for relative velocities

    // Fill observation buffer (all normalized to approximately [-1, 1])
    int idx = 0;
    env->observations[idx++] = (float)(rel_r / pos_scale);      // 0: rel_x (R-bar) - not clamped
    env->observations[idx++] = (float)(rel_v / pos_scale);      // 1: rel_y (V-bar)
    env->observations[idx++] = (float)(rel_h / pos_scale);      // 2: rel_z (H-bar)
    env->observations[idx++] = (float)(rel_vr / vel_scale);     // 3: rel_vx
    env->observations[idx++] = (float)(rel_vv / vel_scale);     // 4: rel_vy
    env->observations[idx++] = (float)(rel_vh / vel_scale);     // 5: rel_vz
    env->observations[idx++] = (float)(dist / pos_scale);       // 6: dist_norm
    env->observations[idx++] = (float)(closing_speed / vel_scale); // 7: closing_speed
    env->observations[idx++] = (float)fmax(0.0, fmin(1.0, env->fuel / env->fuel_budget)); // 8: fuel_remaining
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, alt_diff));               // 9: orbit_alt_norm
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, phase_angle));            // 10: phase_angle
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, incl_diff_norm));         // 11: inclination_diff
    env->observations[idx++] = (float)fmax(-1.0, fmin(1.0, node_angle));             // 12: node_angle
    env->observations[idx++] = (float)fmax(0.0, fmin(1.0, 1.0 - (double)env->step_count / env->max_steps)); // 13: time_remaining
}

// ============================================================================
// Reset
// ============================================================================

void c_reset(OrbitalDock* env) {
    env->step_count = 0;

    // Station in circular orbit at configured radius
    double r = env->station_radius;
    double v_circ = sqrt(env->mu / r);
    env->t_omega = v_circ / r;  // Angular velocity

    // Station starts at (r, 0, 0) moving in +Y direction (in XY plane)
    env->tx = r; env->ty = 0; env->tz = 0;
    env->tvx = 0; env->tvy = v_circ; env->tvz = 0;
    env->t_hx = 0; env->t_hy = 0; env->t_hz = 1.0;  // Angular momentum: +Z
    env->t_radius = r;

    // Scale randomization by difficulty
    // At difficulty=0: 100m radial offset (directly above station), simple radial thrust task
    // At difficulty=1: full ranges (±50km alt, ±30° phase, ±15° incl, ±5m/s vel)
    double d = env->difficulty;

    // Minimum offsets at difficulty=0: radial separation only (simpler than phase)
    double min_phase_off = 0.0;         // No phase offset at easiest difficulty
    double min_alt_off = 100.0;         // 100m directly above station
    double min_incl_off = 0.0;
    double min_vel_perturb = 0.0;

    // Random offsets scaled by difficulty
    double rand_alt = env->alt_offset_max * (2.0 * rndf(env) - 1.0);
    double rand_phase = env->phase_offset_max * (2.0 * rndf(env) - 1.0);
    double rand_incl = env->incl_offset_max * (2.0 * rndf(env) - 1.0);
    double rand_vel = env->vel_perturb_max;

    // Interpolate between minimum and full range based on difficulty
    double alt_off = min_alt_off + d * rand_alt;
    double phase_off = min_phase_off + d * rand_phase;
    double incl_off = min_incl_off + d * rand_incl;
    double vel_perturb = min_vel_perturb + d * rand_vel;

    // Chaser orbit radius
    double c_radius = r + alt_off;
    double c_v_circ = sqrt(env->mu / c_radius);

    // Position with phase offset
    double phase = phase_off;
    env->cx = c_radius * cos(phase);
    env->cy = c_radius * sin(phase);
    env->cz = 0;

    // Apply inclination offset (rotate around X-axis)
    if (fabs(incl_off) > 1e-10) {
        Vec3d c_pos = vec3d(env->cx, env->cy, env->cz);
        Vec3d incl_axis = vec3d(1, 0, 0);
        c_pos = rodrigues_rotate(c_pos, incl_axis, incl_off);
        env->cx = c_pos.x; env->cy = c_pos.y; env->cz = c_pos.z;

        // Circular velocity with inclination
        Vec3d c_vel = vec3d(-c_v_circ * sin(phase), c_v_circ * cos(phase), 0);
        c_vel = rodrigues_rotate(c_vel, incl_axis, incl_off);
        env->cvx = c_vel.x;
        env->cvy = c_vel.y;
        env->cvz = c_vel.z;
    } else {
        env->cvx = -c_v_circ * sin(phase);
        env->cvy = c_v_circ * cos(phase);
        env->cvz = 0;
    }

    // Add velocity perturbation
    env->cvx += vel_perturb * (2.0 * rndf(env) - 1.0);
    env->cvy += vel_perturb * (2.0 * rndf(env) - 1.0);
    env->cvz += vel_perturb * (2.0 * rndf(env) - 1.0);

    // Initialize fuel and distance tracking
    env->fuel = env->fuel_budget;
    Vec3d rel = sub3d(vec3d(env->cx, env->cy, env->cz), vec3d(env->tx, env->ty, env->tz));
    env->init_dist = norm3d(rel);
    if (env->init_dist < 1.0) env->init_dist = 1.0;  // Prevent division by zero
    env->prev_dist = env->init_dist;

    compute_observations(env);
}

// ============================================================================
// Step
// ============================================================================

void c_step(OrbitalDock* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->step_count++;

    // 1. Get chaser LVLH basis
    Vec3d c_pos = vec3d(env->cx, env->cy, env->cz);
    Vec3d c_vel = vec3d(env->cvx, env->cvy, env->cvz);
    Vec3d r_hat, v_hat, h_hat;
    compute_lvlh(c_pos, c_vel, &r_hat, &v_hat, &h_hat);

    // 2. Convert discrete actions to thrust vector
    // Actions: [thrust_pro, thrust_rad, thrust_norm] each in {0,1,2,3,4}
    // Map to: {-100%, -50%, 0%, +50%, +100%}
    double thrust_levels[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    int a_pro = env->actions[0];
    int a_rad = env->actions[1];
    int a_norm = env->actions[2];

    // Clamp actions to valid range
    if (a_pro < 0) a_pro = 0; if (a_pro > 4) a_pro = 4;
    if (a_rad < 0) a_rad = 0; if (a_rad > 4) a_rad = 4;
    if (a_norm < 0) a_norm = 0; if (a_norm > 4) a_norm = 4;

    double t_pro = thrust_levels[a_pro] * env->max_thrust;
    double t_rad = thrust_levels[a_rad] * env->max_thrust;
    double t_norm = thrust_levels[a_norm] * env->max_thrust;

    // LVLH thrust vector -> inertial frame
    Vec3d thrust_lvlh = add3d(
        add3d(scale3d(v_hat, t_pro), scale3d(r_hat, t_rad)),
        scale3d(h_hat, t_norm)
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
    Vec3d t_pos = vec3d(env->tx, env->ty, env->tz);
    Vec3d t_vel = vec3d(env->tvx, env->tvy, env->tvz);
    c_pos = vec3d(env->cx, env->cy, env->cz);
    c_vel = vec3d(env->cvx, env->cvy, env->cvz);

    Vec3d rel_pos = sub3d(c_pos, t_pos);
    Vec3d rel_vel = sub3d(c_vel, t_vel);
    double dist = norm3d(rel_pos);
    double rel_speed = norm3d(rel_vel);
    double c_alt = norm3d(c_pos) - env->earth_radius;

    int docked = (dist < env->dock_dist) && (rel_speed < env->dock_speed);
    int crashed = (dist < env->dock_dist) && (rel_speed >= env->dock_speed);
    int deorbited = (c_alt < env->deorbit_alt);
    int escaped = (c_alt > env->escape_alt);
    int timeout = (env->step_count >= env->max_steps);

    // 7. Compute rewards
    double reward = fuel_penalty;

    // Distance shaping: reward for getting closer (in meters, normalized by 1000m)
    double dist_delta = env->prev_dist - dist;  // positive when closing
    reward += env->rw_dist_shaping * (dist_delta / 100.0);  // 1m closer = +0.005 reward at default

    // Closing velocity reward: reward for having velocity toward target
    double closing_speed = (dist > 1e-10) ? -dot3d(normalize3d(rel_pos), rel_vel) : 0.0;
    reward += env->rw_closing * (closing_speed / 10.0);  // 1 m/s closing = +0.01 reward at default

    // Proximity bonus: stronger reward as we get very close (exponential)
    double proximity_bonus = exp(-dist / 500.0);  // peaks at 1.0 when at target, ~0.82 at 100m, ~0.37 at 500m
    reward += env->rw_vel_match * proximity_bonus * 0.1;

    // Velocity matching shaping: reward for low relative velocity when close
    if (dist < 500.0) {  // Only matters when close
        double vel_match_bonus = (1.0 - fmin(1.0, rel_speed / 5.0)) * (1.0 - dist / 500.0);
        reward += env->rw_vel_match * vel_match_bonus * 0.05;
    }

    // Plane alignment shaping (small bonus for reducing inclination difference)
    Vec3d c_h = normalize3d(cross3d(c_pos, c_vel));
    Vec3d t_h = vec3d(env->t_hx, env->t_hy, env->t_hz);
    double cos_incl = fmax(-1.0, fmin(1.0, dot3d(c_h, t_h)));
    double incl_alignment = (1.0 + cos_incl) / 2.0;  // 0 when perpendicular, 1 when aligned
    reward += env->rw_plane_align * incl_alignment;

    // Terminal rewards
    if (docked) {
        reward += env->rw_dock;
        env->terminals[0] = 1;
        env->log.dock_success += 1.0f;
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    } else if (crashed) {
        reward -= env->rw_crash;
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
        env->terminals[0] = 1;
        env->log.timeout_rate += 1.0f;
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    }

    env->rewards[0] = (float)reward;
    env->prev_dist = dist;

    // Update log
    env->log.episode_return += (float)reward;
    if (env->terminals[0]) {
        env->log.episode_length += (float)env->step_count;
        env->log.fuel_used += (float)(env->fuel_budget - env->fuel);
        env->log.n += 1.0f;
        c_reset(env);
    }

    compute_observations(env);
}

// ============================================================================
// Render (stub for future raylib implementation)
// ============================================================================

void c_render(OrbitalDock* env) {
    // TODO: Implement raylib 3D visualization
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib Orbital Dock");
        SetTargetFPS(60);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw simple HUD for now
    char buf[256];
    Vec3d rel_pos = sub3d(vec3d(env->cx, env->cy, env->cz), vec3d(env->tx, env->ty, env->tz));
    Vec3d rel_vel = sub3d(vec3d(env->cvx, env->cvy, env->cvz), vec3d(env->tvx, env->tvy, env->tvz));
    double dist = norm3d(rel_pos);
    double rel_speed = norm3d(rel_vel);

    snprintf(buf, sizeof(buf), "Distance: %.1f m", dist);
    DrawText(buf, 20, 20, 20, PUFF_WHITE);

    snprintf(buf, sizeof(buf), "Rel Speed: %.2f m/s", rel_speed);
    DrawText(buf, 20, 45, 20, PUFF_WHITE);

    snprintf(buf, sizeof(buf), "Fuel: %.1f%%", 100.0 * env->fuel / env->fuel_budget);
    DrawText(buf, 20, 70, 20, PUFF_WHITE);

    snprintf(buf, sizeof(buf), "Step: %d / %d", env->step_count, env->max_steps);
    DrawText(buf, 20, 95, 20, PUFF_WHITE);

    snprintf(buf, sizeof(buf), "Difficulty: %.1f", env->difficulty);
    DrawText(buf, 20, 120, 20, PUFF_WHITE);

    DrawText("Orbital Dock - 3D rendering coming soon", 20, 680, 20, PUFF_CYAN);

    EndDrawing();
}

void c_close(OrbitalDock* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

#endif // ORBITAL_DOCK_H
