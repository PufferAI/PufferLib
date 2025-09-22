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
#define HORIZON 1024

// Physical constants for the drone
#define BASE_MASS 1.0f       // kg
#define BASE_IXX 0.01f       // kgm^2
#define BASE_IYY 0.01f       // kgm^2
#define BASE_IZZ 0.02f       // kgm^2
#define BASE_ARM_LEN 0.1f    // m
#define BASE_K_THRUST 3e-5f  // thrust coefficient
#define BASE_K_ANG_DAMP 0.2f // angular damping coefficient
#define BASE_K_DRAG 1e-6f    // drag (torque) coefficient
#define BASE_B_DRAG 0.1f     // linear drag coefficient
#define BASE_GRAVITY 9.81f   // m/s^2
#define BASE_MAX_RPM 750.0f  // rad/s
#define BASE_MAX_VEL 50.0f   // m/s
#define BASE_MAX_OMEGA 50.0f // rad/s
#define BASE_K_MOT 0.1f      // s (Motor lag constant)
#define BASE_J_MOT 1e-5f     // kgm^2 (Motor rotational inertia)

// Simulation properties
#define GRID_X 10.0f
#define GRID_Y 10.0f
#define GRID_Z 10.0f
#define MARGIN_X (GRID_X - 1)
#define MARGIN_Y (GRID_Y - 1)
#define MARGIN_Z (GRID_Z - 1)
#define V_TARGET 0.05f
#define DT 0.05f
#define DT_RNG 0.0f

// Corner to corner distance
#define MAX_DIST sqrtf((2*GRID_X)*(2*GRID_X) + (2*GRID_Y)*(2*GRID_Y) + (2*GRID_Z)*(2*GRID_Z))

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float rings_passed;
    float collision_rate;
    float oob;
    float timeout;
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

static inline Quat add_quat(Quat a, Quat b) { return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z}; }

static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }

static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline Quat scalmul_quat(Quat a, float b) { return (Quat){a.w * b, a.x * b, a.y * b, a.z * b}; }

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

typedef struct {
    Vec3 pos;
    Quat orientation;
    Vec3 normal;
    float radius;
} Ring;

Ring rndring(float radius) {
    Ring ring;

    ring.pos.x = rndf(-GRID_X + 2*radius, GRID_X - 2*radius);
    ring.pos.y = rndf(-GRID_Y + 2*radius, GRID_Y - 2*radius);
    ring.pos.z = rndf(-GRID_Z + 2*radius, GRID_Z - 2*radius);

    ring.orientation = rndquat();

    Vec3 base_normal = {0.0f, 0.0f, 1.0f};
    ring.normal = quat_rotate(ring.orientation, base_normal);

    ring.radius = radius;

    return ring;
}

typedef struct {
    Vec3 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct {
    Vec3 pos;      // global position (x, y, z)
    Vec3 vel;      // linear velocity (u, v, w)
    Quat quat;     // roll/pitch/yaw (phi/theta/psi) as a quaternion
    Vec3 omega;    // angular velocity (p, q, r)
    float rpms[4]; // motor RPMs
} State;

typedef struct {
    Vec3 vel;       // Derivative of position
    Vec3 v_dot;       // Derivative of velocity
    Quat q_dot;       // Derivative of quaternion
    Vec3 w_dot;       // Derivative of angular velocity
    float rpm_dot[4]; // Derivative of motor RPMs
} StateDerivative;

typedef struct {
    // Physical properties. Modeled as part of the drone
    // to make domain randomization easier.
    float mass; // kg
    float ixx; // kgm^2
    float iyy; // kgm^2
    float izz; // kgm^2
    float arm_len; // m
    float k_thrust; // thrust coefficient
    float k_ang_damp; // angular damping coefficient
    float k_drag; // drag (torque) coefficient
    float b_drag; // linear drag coefficient
    float gravity; // m/s^2
    float max_rpm; // rad/s
    float max_vel; // m/s
    float max_omega; // rad/s
    float k_mot; // s
    float j_mot; // kgm^2
} Params;

typedef struct {
    // core state and parameters
    State state;
    Params params;

    // helpers for ring/swarm logic
    Vec3 spawn_pos;
    Vec3 prev_pos;
    Vec3 target_pos;
    Vec3 target_vel;

    // logging utils
    float last_abs_reward;
    float last_target_reward;
    float last_collision_reward;
    float episode_return;
    float collisions;
    int episode_length;
    float score;
    int ring_idx;
} Drone;


void init_drone(Drone* drone, float size, float dr) {
    drone->params.arm_len = size / 2.0f;

    // m ~ x^3
    float mass_scale = powf(drone->params.arm_len, 3.0f) / powf(BASE_ARM_LEN, 3.0f);
    drone->params.mass = BASE_MASS * mass_scale * rndf(1.0f - dr, 1.0f + dr);

    // I ~ mx^2
    float base_Iscale = BASE_MASS * BASE_ARM_LEN * BASE_ARM_LEN;
    float I_scale = drone->params.mass * powf(drone->params.arm_len, 2.0f) / base_Iscale;
    drone->params.ixx = BASE_IXX * I_scale * rndf(1.0f - dr, 1.0f + dr);
    drone->params.iyy = BASE_IYY * I_scale * rndf(1.0f - dr, 1.0f + dr);
    drone->params.izz = BASE_IZZ * I_scale * rndf(1.0f - dr, 1.0f + dr);

    // k_thrust ~ m/l
    float k_thrust_scale = (drone->params.mass * drone->params.arm_len) / (BASE_MASS * BASE_ARM_LEN);
    drone->params.k_thrust = BASE_K_THRUST * k_thrust_scale * rndf(1.0f - dr, 1.0f + dr);

    // k_ang_damp ~ I
    float base_avg_inertia = (BASE_IXX + BASE_IYY + BASE_IZZ) / 3.0f;
    float avg_inertia = (drone->params.ixx + drone->params.iyy + drone->params.izz) / 3.0f;
    float avg_inertia_scale = avg_inertia / base_avg_inertia;
    drone->params.k_ang_damp = BASE_K_ANG_DAMP * avg_inertia_scale * rndf(1.0f - dr, 1.0f + dr);

    // drag ~ x^2
    float drag_scale = powf(drone->params.arm_len, 2.0f) / powf(BASE_ARM_LEN, 2.0f);
    drone->params.k_drag = BASE_K_DRAG * drag_scale * rndf(1.0f - dr, 1.0f + dr);
    drone->params.b_drag = BASE_B_DRAG * drag_scale * rndf(1.0f - dr, 1.0f + dr);

    // Small gravity randomization
    drone->params.gravity = BASE_GRAVITY * rndf(0.99f, 1.01f);

    // RPM ~ 1/x
    float rpm_scale = (BASE_ARM_LEN) / (drone->params.arm_len);
    drone->params.max_rpm = BASE_MAX_RPM * rpm_scale * rndf(1.0f - dr, 1.0f + dr);

    drone->params.max_vel = BASE_MAX_VEL;
    drone->params.max_omega = BASE_MAX_OMEGA;

    drone->params.k_mot = BASE_K_MOT * rndf(1.0f - dr, 1.0f + dr);
    drone->params.j_mot = BASE_J_MOT * I_scale * rndf(1.0f - dr, 1.0f + dr);
    
    for (int i = 0; i < 4; i++) {
        drone->state.rpms[i] = 0.0f;
    }
    drone->state.pos = (Vec3){0.0f, 0.0f, 0.0f};
    drone->prev_pos = drone->state.pos;
    drone->state.vel = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.omega = (Vec3){0.0f, 0.0f, 0.0f};
    drone->state.quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
}

void compute_derivatives(State* state, Params* params, float* actions, StateDerivative* derivatives) {
    // first order rpm lag
    float target_rpms[4];
    for (int i = 0; i < 4; i++) {
        target_rpms[i] = (actions[i] + 1.0f) * 0.5f * params->max_rpm;
    }

    // rpm rates
    float rpm_dot[4];
    for (int i = 0; i < 4; i++) {
        rpm_dot[i] = (1.0f / params->k_mot) * (target_rpms[i] - state->rpms[i]);
    }

    // motor thrusts
    float T[4];
    for (int i = 0; i < 4; i++) {
        T[i] = params->k_thrust * powf(state->rpms[i], 2.0f);
    }

    // body frame net force
    Vec3 F_prop_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

    // body frame force -> world frame force
    Vec3 F_prop = quat_rotate(state->quat, F_prop_body);

    // world frame linear drag
    Vec3 F_aero;
    F_aero.x = -params->b_drag * state->vel.x;
    F_aero.y = -params->b_drag * state->vel.y;
    F_aero.z = -params->b_drag * state->vel.z;

    // velocity rates, a = F/m
    Vec3 v_dot;
    v_dot.x = (F_prop.x + F_aero.x) / params->mass;
    v_dot.y = (F_prop.y + F_aero.y) / params->mass;
    v_dot.z = ((F_prop.z + F_aero.z) / params->mass) - params->gravity;

    // quaternion rates
    Quat omega_q = {0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->quat, omega_q);
    q_dot.w *= 0.5f;
    q_dot.x *= 0.5f;
    q_dot.y *= 0.5f;
    q_dot.z *= 0.5f;

    // body frame torques
    Vec3 Tau_prop;
    Tau_prop.x = params->arm_len*(T[1] - T[3]);
    Tau_prop.y = params->arm_len*(T[2] - T[0]);
    Tau_prop.z = params->k_drag*(T[0] - T[1] + T[2] - T[3]);

    // torque from chaging motor speeds
    float Tau_mot_z = params->j_mot * (rpm_dot[0] - rpm_dot[1] + rpm_dot[2] - rpm_dot[3]);

    // torque from angular damping
    Vec3 Tau_aero;
    Tau_aero.x = -params->k_ang_damp * state->omega.x;
    Tau_aero.y = -params->k_ang_damp * state->omega.y;
    Tau_aero.z = -params->k_ang_damp * state->omega.z;
    
    // gyroscopic torque
    Vec3 Tau_iner;
    Tau_iner.x = (params->iyy - params->izz) * state->omega.y * state->omega.z;
    Tau_iner.y = (params->izz - params->ixx) * state->omega.z * state->omega.x;
    Tau_iner.z = (params->ixx - params->iyy) * state->omega.x * state->omega.y;

    // angular velocity rates
    Vec3 w_dot;
    w_dot.x = (Tau_prop.x + Tau_aero.x + Tau_iner.x) / params->ixx;
    w_dot.y = (Tau_prop.y + Tau_aero.y + Tau_iner.y) / params->iyy;
    w_dot.z = (Tau_prop.z + Tau_aero.z + Tau_iner.z + Tau_mot_z) / params->izz;

    derivatives->vel = state->vel;
    derivatives->v_dot = v_dot;
    derivatives->q_dot = q_dot;
    derivatives->w_dot = w_dot;
    for (int i = 0; i < 4; i++) {
        derivatives->rpm_dot[i] = rpm_dot[i];
    }
}

static void step(State* initial, StateDerivative* deriv, float dt, State* output) {
    output->pos = add3(initial->pos, scalmul3(deriv->vel, dt));
    output->vel = add3(initial->vel, scalmul3(deriv->v_dot, dt));
    output->quat = add_quat(initial->quat, scalmul_quat(deriv->q_dot, dt));
    output->omega = add3(initial->omega, scalmul3(deriv->w_dot, dt));
    for (int i = 0; i < 4; i++) {
        output->rpms[i] = initial->rpms[i] + deriv->rpm_dot[i] * dt;
    }
    quat_normalize(&output->quat);
}

void rk4_step(State* state, Params* params, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    State temp_state;

    compute_derivatives(state, params, actions, &k1);

    step(state, &k1, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k2);

    step(state, &k2, dt * 0.5f, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k3);

    step(state, &k3, dt, &temp_state);
    compute_derivatives(&temp_state, params, actions, &k4);

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    for (int i = 0; i < 4; i++) {
        state->rpms[i] += (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt_6;
    }

    quat_normalize(&state->quat);
}

void move_drone(Drone* drone, float* actions) {
    // clamp actions
    clamp4(actions, -1.0f, 1.0f);

    // Domain randomized dt
    float dt = DT * rndf(1.0f - DT_RNG, 1.0 + DT_RNG);

    // update drone state
    drone->prev_pos = drone->state.pos;
    rk4_step(&drone->state, &drone->params, actions, dt);

    // clamp and normalise for observations
    clamp3(&drone->state.vel, -drone->params.max_vel, drone->params.max_vel);
    clamp3(&drone->state.omega, -drone->params.max_omega, drone->params.max_omega);
}

void reset_rings(Ring* ring_buffer, int num_rings, float ring_radius) {
    ring_buffer[0] = rndring(ring_radius);
    
    // ensure rings are spaced at least 2*ring_radius apart
    for (int i = 1; i < num_rings; i++) {
        do {
            ring_buffer[i] = rndring(ring_radius);
        }  while (norm3(sub3(ring_buffer[i].pos, ring_buffer[i - 1].pos)) < 2.0f*ring_radius);
    }   
}

float check_ring(Drone* drone, Ring* ring) {
    // previous dot product negative if on the 'entry' side of the ring's plane
    float prev_dot = dot3(sub3(drone->prev_pos, ring->pos), ring->normal);

    // new dot product positive if on the 'exit' side of the ring's plane
    float new_dot = dot3(sub3(drone->state.pos, ring->pos), ring->normal);

    bool valid_dir = (prev_dot < 0.0f && new_dot > 0.0f);
    bool invalid_dir = (prev_dot > 0.0f && new_dot < 0.0f);

    // if we have crossed the plane of the ring
    if (valid_dir || invalid_dir) {
        // find intesection with ring's plane
        Vec3 dir = sub3(drone->state.pos, drone->prev_pos);
        float t = -prev_dot / dot3(ring->normal, dir); // possible nan

        Vec3 intersection = add3(drone->prev_pos, scalmul3(dir, t));
        float dist = norm3(sub3(intersection, ring->pos));

        // reward or terminate based on distance to ring center
        if (dist < (ring->radius - 0.5) && valid_dir) {
            return 1.0f;
        } else if (dist < ring->radius + 0.5) {
            return -1.0f;
        }
    }
    return 0.0f;
}
