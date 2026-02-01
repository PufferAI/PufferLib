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
} Client;

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
    (void)env;  // unused, using global rand()
    return (double)rand() / (double)RAND_MAX;
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
    // At difficulty=0: Simple scenarios - random approach from various directions
    // At difficulty=1: Full 3D orbital mechanics with large separations
    double d = env->difficulty;

    // Get station's LVLH basis vectors
    Vec3d t_pos = vec3d(env->tx, env->ty, env->tz);
    Vec3d t_vel = vec3d(env->tvx, env->tvy, env->tvz);
    Vec3d r_hat, v_hat, h_hat;
    compute_lvlh(t_pos, t_vel, &r_hat, &v_hat, &h_hat);

    // === APPROACH DISTANCE ===
    // Design constraint: optimal first action must depend on state
    // No constant policy should achieve >50% dock rate
    // d<-0.5: 5-8m with 0-1.5 m/s closing (must brake if fast, thrust if slow)
    // d<0.05: 8-15m with 0-2.0 m/s closing (longer planning horizon)
    // d<0.15: 15-30m (2-axis corrections)
    // d<0.3:  50-200m (full 3-axis control)
    // d>=0.4: 1-5km (orbital transfers)
    double approach_dist;
    if (d < -0.5) {
        approach_dist = 30.0 + 20.0 * rndf(env);  // 30-50m at d=-1 (spec)
    } else if (d < 0.05) {
        approach_dist = 8.0 + 7.0 * rndf(env);  // 8-15m at d=0
    } else if (d < 0.15) {
        approach_dist = 15.0 + 15.0 * rndf(env);  // 15-30m at d~0.1
    } else if (d < 0.3) {
        approach_dist = 50.0 + 150.0 * rndf(env);  // 50-200m at d~0.2
    } else {
        approach_dist = 1000.0 + 4000.0 * d * rndf(env);  // 1-5km at d>=0.4
    }

    // Direction: random on sphere, but at low difficulty biased toward V-bar
    double theta = 2.0 * M_PI * rndf(env);
    double phi = acos(2.0 * rndf(env) - 1.0);

    // At low difficulty, flatten toward V-bar (equatorial plane in spherical coords)
    // d < -0.5: Pure V-bar approach (phi = 90°, along prograde direction)
    // d = 0: Pure V-bar approach
    // d = 0.5: Full random direction
    if (d < -0.5) {
        // At d=-1: Pure V-bar (prograde) approach - chaser behind station
        phi = M_PI / 2.0;  // Exactly on V-bar
        theta = M_PI;      // Negative V-bar direction (chaser behind station)
    } else if (d < 0.5) {
        phi = M_PI/2.0 + (phi - M_PI/2.0) * d * 2.0;  // At d=0, phi=90° (pure V-bar)
    }

    double lvlh_r = approach_dist * cos(phi);
    double lvlh_v = approach_dist * sin(phi) * cos(theta);
    double lvlh_h = approach_dist * sin(phi) * sin(theta);

    // Convert LVLH offset to inertial position
    Vec3d offset = add3d(add3d(scale3d(r_hat, lvlh_r), scale3d(v_hat, lvlh_v)), scale3d(h_hat, lvlh_h));
    Vec3d c_pos = add3d(t_pos, offset);
    env->cx = c_pos.x; env->cy = c_pos.y; env->cz = c_pos.z;

    // === INITIAL VELOCITY ===
    // At d<-0.5: Zero relative velocity - agent MUST thrust to dock
    // At d=0: Start with closing velocity toward station (0.1-0.3 m/s)
    //         This makes random exploration likely to stumble into dock zone
    // At higher d: Add velocity perturbations requiring correction
    Vec3d c_vel = t_vel;  // Start co-moving with station

    if (d < -0.5) {
        // At d=-1: 0-4.0 m/s closing velocity
        // Slow starts (<0.5 m/s): clean dock without braking
        // Medium starts (0.5-1.0 m/s): rough dock without braking
        // Fast starts (>1.0 m/s): crash without braking - MUST brake
        double closing_speed = 4.0 * rndf(env);  // 0-4.0 m/s
        double vr_closing = -lvlh_r / approach_dist * closing_speed;
        double vv_closing = -lvlh_v / approach_dist * closing_speed;
        double vh_closing = -lvlh_h / approach_dist * closing_speed;
        c_vel = add3d(c_vel, scale3d(r_hat, vr_closing));
        c_vel = add3d(c_vel, scale3d(v_hat, vv_closing));
        c_vel = add3d(c_vel, scale3d(h_hat, vh_closing));
    } else if (d < 0.05) {
        // At d=0: 0-2.0 m/s closing velocity, wider range
        double closing_speed = 2.0 * rndf(env);  // 0-2.0 m/s
        double vr_closing = -lvlh_r / approach_dist * closing_speed;
        double vv_closing = -lvlh_v / approach_dist * closing_speed;
        double vh_closing = -lvlh_h / approach_dist * closing_speed;
        c_vel = add3d(c_vel, scale3d(r_hat, vr_closing));
        c_vel = add3d(c_vel, scale3d(v_hat, vv_closing));
        c_vel = add3d(c_vel, scale3d(h_hat, vh_closing));
    } else if (d > 0.1) {
        // At higher difficulty, add velocity perturbations
        double vel_mag = d * 0.5 * rndf(env);  // up to 0.5 m/s at d=1
        // Random direction perturbation
        double vtheta = 2.0 * M_PI * rndf(env);
        double vphi = acos(2.0 * rndf(env) - 1.0);
        c_vel = add3d(c_vel, scale3d(r_hat, vel_mag * cos(vphi)));
        c_vel = add3d(c_vel, scale3d(v_hat, vel_mag * sin(vphi) * cos(vtheta)));
        c_vel = add3d(c_vel, scale3d(h_hat, vel_mag * sin(vphi) * sin(vtheta)));
    }

    env->cvx = c_vel.x; env->cvy = c_vel.y; env->cvz = c_vel.z;

    // Initialize fuel and distance tracking
    env->fuel = env->fuel_budget;
    Vec3d rel = sub3d(vec3d(env->cx, env->cy, env->cz), vec3d(env->tx, env->ty, env->tz));
    Vec3d rel_vel = sub3d(vec3d(env->cvx, env->cvy, env->cvz), vec3d(env->tvx, env->tvy, env->tvz));
    env->init_dist = norm3d(rel);
    if (env->init_dist < 1.0) env->init_dist = 1.0;  // Prevent division by zero
    env->prev_dist = env->init_dist;
    env->prev_speed = norm3d(rel_vel);

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

    int deorbited = (c_alt < env->deorbit_alt);
    int escaped = (c_alt > env->escape_alt);
    int timeout = (env->step_count >= env->max_steps);

    // 7. Compute rewards
    // Clean 4-component reward structure:
    // - Step penalty: encourages finishing quickly
    // - Distance shaping: guides toward target
    // - Fuel penalty: encourages efficiency
    // - Terminal rewards: graduated docking quality

    double reward = 0.0;

    // Step penalty: -0.005/step = -10.0 over 2000 steps (makes timeout costly)
    reward -= 0.005;

    // Distance shaping: reward for getting closer (potential-based)
    double dist_delta = env->prev_dist - dist;  // positive when closing
    reward += env->rw_dist_shaping * dist_delta;

    // Velocity shaping: reward for reducing relative speed (potential-based)
    // This creates a "funnel" - approach but slow down as you get close
    double speed_delta = env->prev_speed - rel_speed;  // positive when slowing
    reward += env->rw_vel_match * speed_delta;

    // Fuel penalty (already computed above)
    reward += fuel_penalty;

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
        reward += 3.0;  // Rough dock - acceptable
        env->terminals[0] = 1;
        env->log.dock_success += 0.5f;  // Count as partial success
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)rel_speed;
    } else if (crash_dock) {
        reward -= 5.0;  // Crash - too fast
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
    env->prev_speed = rel_speed;

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
