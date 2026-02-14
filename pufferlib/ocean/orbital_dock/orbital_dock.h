#ifndef ORBITAL_DOCK_H
#define ORBITAL_DOCK_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
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
// Log Struct (all floats, ending with n)
// ============================================================================

typedef struct {
    float episode_return;
    float episode_length;
    float dock_success;
    float crash_rate;
    float timeout_rate;
    float fuel_used;
    float final_distance;
    float final_rel_speed;
    float n;  // Required as last field
} Log;

// ============================================================================
// Render Client Struct
// ============================================================================

#define RENDER_WIDTH 1080
#define RENDER_HEIGHT 720
#define TRAIL_LENGTH 256

typedef struct {
    double x, y, z;
} Vec3d;

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
    float scale;
    int last_step;
} Client;

// ============================================================================
// Environment Struct
// ============================================================================

typedef struct {
    Log log;                     // Required field (first)
    float* observations;         // Required field - 6 floats [x, y, z, vx, vy, vz]
    float* actions;              // Required field - 3 floats (thrust fractions [-1,1])
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field

    // CW state in LVLH frame (meters, m/s)
    double x, y, z;             // Position relative to station
    double vx, vy, vz;          // Velocity relative to station
    double prev_dist;           // Previous distance to dock (for progress reward)

    // CW orbital parameter
    double n;                   // Mean motion = sqrt(mu/R^3) (rad/s)

    // Environment state
    double fuel;                // Remaining delta-v (m/s)
    int step_count;
    int max_steps;

    // Physics config
    double mu;                  // Gravitational parameter (m^3/s^2)
    double station_radius;      // Station orbit radius (m)
    double max_thrust;          // Max thrust per axis (N)
    double mass;                // Chaser mass (kg)
    double dt;                  // Timestep (s)
    double fuel_budget;         // Total delta-v budget (m/s)

    // Docking conditions (STELLAR: dock at [0, 60, 0])
    double dock_x, dock_y, dock_z;  // Docking point in LVLH
    double dock_dist;           // Docking distance threshold (m)
    double dock_speed;          // Final docking speed threshold (m/s)
    double dock_speed_start;    // Starting speed threshold for annealing (m/s)
    int anneal_steps;           // Per-env steps to anneal dock_speed (0 = no annealing)
    int global_step;            // Persistent step counter (never resets)

    // LOS cone (STELLAR: 60 deg total, 800m extent along +y)
    double los_half_angle;      // Half-angle in radians
    double los_extent;          // Max extent along y-axis (m)

    // Initial condition ranges (STELLAR V-bar approach)
    double init_x_center, init_y_center, init_z_center;
    double init_x_range, init_y_range, init_z_range;

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
    return (double)xorshift32(&env->rng_state) / (double)0xFFFFFFFF;
}

static inline double rndf_range(OrbitalDock* env, double min, double max) {
    return min + rndf(env) * (max - min);
}

// ============================================================================
// CW Dynamics: Clohessy-Wiltshire Linear Relative Motion
// ============================================================================

// CW derivatives: dx/dt = f(state, thrust_accel)
// state = [x, y, z, vx, vy, vz]
// accel = [ax, ay, az] (thrust/mass in LVLH)
static void cw_derivatives(double n, const double state[6],
                           const double accel[3], double deriv[6]) {
    double x  = state[0], y  = state[1], z  = state[2];
    double vx = state[3], vy = state[4], vz = state[5];

    deriv[0] = vx;
    deriv[1] = vy;
    deriv[2] = vz;
    deriv[3] = 3.0*n*n*x + 2.0*n*vy + accel[0];   // radial
    deriv[4] = -2.0*n*vx            + accel[1];     // along-track
    deriv[5] = -n*n*z               + accel[2];     // cross-track
}

// RK4 integration of CW equations over one timestep
static void cw_step_rk4(OrbitalDock* env, double ax, double ay, double az) {
    double dt = env->dt;
    double n = env->n;
    double accel[3] = {ax, ay, az};

    double s[6] = {env->x, env->y, env->z, env->vx, env->vy, env->vz};
    double k1[6], k2[6], k3[6], k4[6], tmp[6];

    // k1
    cw_derivatives(n, s, accel, k1);

    // k2
    for (int i = 0; i < 6; i++) tmp[i] = s[i] + 0.5*dt*k1[i];
    cw_derivatives(n, tmp, accel, k2);

    // k3
    for (int i = 0; i < 6; i++) tmp[i] = s[i] + 0.5*dt*k2[i];
    cw_derivatives(n, tmp, accel, k3);

    // k4
    for (int i = 0; i < 6; i++) tmp[i] = s[i] + dt*k3[i];
    cw_derivatives(n, tmp, accel, k4);

    // Update state
    for (int i = 0; i < 6; i++) {
        s[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }

    env->x  = s[0]; env->y  = s[1]; env->z  = s[2];
    env->vx = s[3]; env->vy = s[4]; env->vz = s[5];
}

// ============================================================================
// LOS Cone Check (STELLAR: 60 deg total, along +y from docking point)
// ============================================================================

static int in_los(OrbitalDock* env) {
    // Vector from docking point to chaser
    double px = env->x - env->dock_x;
    double py = env->y - env->dock_y;
    double pz = env->z - env->dock_z;

    // Must be in front of docking point (positive y relative to dock)
    if (py < 0.0 || py > env->los_extent) return 0;

    // LOS cone axis is along +y
    double p_norm = sqrt(px*px + py*py + pz*pz);
    if (p_norm < 1e-10) return 1;  // At docking point

    // cos(angle) = dot(p, cone_axis) / |p|
    // cone_axis = [0, 1, 0], so dot = py
    double cos_angle = py / p_norm;

    return (cos_angle >= cos(env->los_half_angle)) ? 1 : 0;
}

// ============================================================================
// Observation Computation (10 floats: raw LVLH state + computed features)
// ============================================================================

static void compute_observations(OrbitalDock* env) {
    // Raw LVLH state
    env->observations[0] = (float)env->x;
    env->observations[1] = (float)env->y;
    env->observations[2] = (float)env->z;
    env->observations[3] = (float)env->vx;
    env->observations[4] = (float)env->vy;
    env->observations[5] = (float)env->vz;

    // Computed features
    double dx = env->x - env->dock_x;
    double dy = env->y - env->dock_y;
    double dz = env->z - env->dock_z;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    double speed = sqrt(env->vx*env->vx + env->vy*env->vy + env->vz*env->vz);
    double closing_vel = (dist > 1e-6) ? -(dx*env->vx + dy*env->vy + dz*env->vz) / dist : 0.0;
    double time_remaining = 1.0 - (double)env->step_count / (double)env->max_steps;

    env->observations[6] = (float)dist;
    env->observations[7] = (float)speed;
    env->observations[8] = (float)closing_vel;
    env->observations[9] = (float)time_remaining;
}

// ============================================================================
// Reset
// ============================================================================

void c_reset(OrbitalDock* env) {
    env->step_count = 0;

    // Initialize xorshift RNG
    env->rng_state = (unsigned int)rand() ^ ((unsigned int)rand() << 15);
    if (env->rng_state == 0) env->rng_state = 1;

    // Compute mean motion
    env->n = sqrt(env->mu / (env->station_radius * env->station_radius * env->station_radius));

    // STELLAR V-bar approach initial conditions
    // Position: [0, 800, 0] +/- [400, 300, 400]
    double x_off = rndf_range(env, -1.0, 1.0) * env->init_x_range;
    double y_off = rndf_range(env, -1.0, 1.0) * env->init_y_range;
    double z_off = rndf_range(env, -1.0, 1.0) * env->init_z_range;

    env->x = env->init_x_center + x_off;
    env->y = env->init_y_center + y_off;
    env->z = env->init_z_center + z_off;

    // Velocity: random in ~[-2, 2] m/s per axis (STELLAR: randint(-2,2) * rand())
    env->vx = rndf_range(env, -2.0, 2.0);
    env->vy = rndf_range(env, -2.0, 2.0);
    env->vz = rndf_range(env, -2.0, 2.0);

    // Initialize fuel
    env->fuel = env->fuel_budget;

    // Compute initial distance to dock for progress reward
    double dx0 = env->x - env->dock_x;
    double dy0 = env->y - env->dock_y;
    double dz0 = env->z - env->dock_z;
    env->prev_dist = sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0);

    compute_observations(env);
}

// ============================================================================
// Step
// ============================================================================

void c_step(OrbitalDock* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->step_count++;

    // 1. Get thrust actions [-1, 1] -> force in Newtons
    double act_x = fmax(-1.0, fmin(1.0, (double)env->actions[0]));
    double act_y = fmax(-1.0, fmin(1.0, (double)env->actions[1]));
    double act_z = fmax(-1.0, fmin(1.0, (double)env->actions[2]));

    double fx = act_x * env->max_thrust;
    double fy = act_y * env->max_thrust;
    double fz = act_z * env->max_thrust;

    // 2. Fuel consumption
    double f_mag = sqrt(fx*fx + fy*fy + fz*fz);
    double dv_used = (f_mag / env->mass) * env->dt;
    double ax = fx / env->mass;
    double ay = fy / env->mass;
    double az = fz / env->mass;

    if (env->fuel > 0 && f_mag > 0) {
        double actual_dv = fmin(dv_used, env->fuel);
        env->fuel -= actual_dv;
        if (actual_dv < dv_used) {
            double scale = actual_dv / dv_used;
            ax *= scale; ay *= scale; az *= scale;
        }
    } else if (env->fuel <= 0) {
        ax = 0; ay = 0; az = 0;
    }

    // 3. Integrate CW dynamics
    cw_step_rk4(env, ax, ay, az);

    // 4. Compute distance to docking point
    double dx = env->x - env->dock_x;
    double dy = env->y - env->dock_y;
    double dz = env->z - env->dock_z;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    double speed = sqrt(env->vx*env->vx + env->vy*env->vy + env->vz*env->vz);

    // 5. Compute effective dock_speed (annealing)
    env->global_step++;
    double effective_dock_speed = env->dock_speed;
    if (env->anneal_steps > 0) {
        double frac = fmin(1.0, (double)env->global_step / (double)env->anneal_steps);
        effective_dock_speed = env->dock_speed_start + frac * (env->dock_speed - env->dock_speed_start);
    }

    // 6. Check termination conditions
    int is_in_los = in_los(env);
    int docked = is_in_los && (dist < env->dock_dist) && (speed < effective_dock_speed);
    int collision = (env->y < env->dock_y - 5.0);  // y < 55m (5m past dock point)
    int timeout = (env->step_count >= env->max_steps);

    // 6. Compute reward (Chen-style, naturally in [-1, 1])
    double reward = 0.0;
    double progress = env->prev_dist - dist;  // positive when approaching dock

    reward += 0.01 * progress;                                      // distance progress
    double prox_now = exp(-dist / 20.0);                            // proximity potential
    double prox_prev = exp(-env->prev_dist / 20.0);
    reward += 0.1 * (prox_now - prox_prev);                        // potential-based proximity
    // reward -= 0.005 * speed * speed;                             // velocity damping (disabled)
    reward -= 0.005 * (act_x*act_x + act_y*act_y + act_z*act_z);  // control cost
    reward -= 0.005;                                                // time penalty

    // Terminal rewards (replace per-step reward)
    if (docked) {
        double speed_bonus = 0.5 * fmax(0.0, 1.0 - speed / effective_dock_speed);
        reward = 0.5 + speed_bonus;  // speed=0→1.0, speed=dock_speed→0.5
        env->terminals[0] = 1;
        env->log.dock_success += 1.0f;
    } else if (collision) {
        reward = -1.0;
        env->terminals[0] = 1;
        env->log.crash_rate += 1.0f;
    } else if (timeout) {
        reward = -0.5;
        env->terminals[0] = 1;
        env->log.timeout_rate += 1.0f;
    }

    env->rewards[0] = (float)reward;  // No scaling needed — naturally in [-1, 1]
    env->prev_dist = dist;

    // Update log
    env->log.episode_return += (float)reward;
    if (env->terminals[0]) {
        env->log.episode_length += (float)env->step_count;
        env->log.fuel_used += (float)(env->fuel_budget - env->fuel);
        env->log.final_distance += (float)dist;
        env->log.final_rel_speed += (float)speed;
        env->log.n += 1.0f;

        c_reset(env);
    }

    compute_observations(env);
}

// ============================================================================
// Render (implemented in render.h)
// ============================================================================

Client* make_client(OrbitalDock *env);
void close_client(Client *client);
void render_orbital_dock(OrbitalDock *env, Client *client);

void c_render(OrbitalDock* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) return;
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
