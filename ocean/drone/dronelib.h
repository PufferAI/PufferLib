// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50

// Crazyflie Physical Constants
// https://github.com/arplaboratory/learning-to-fly
#define BASE_MASS 0.027f         // kg
#define BASE_IXX 3.85e-6f        // kgm²
#define BASE_IYY 3.85e-6f        // kgm²
#define BASE_IZZ 5.9675e-6f      // kgm²
#define BASE_ARM_LEN 0.0396f     // m
#define BASE_K_THRUST 3.16e-10f  // thrust coefficient
#define BASE_K_DRAG 0.005964552f // yaw moment constant
#define BASE_GRAVITY 9.81f       // m/s^2
#define BASE_MAX_RPM 21702.0f    // RPM
#define BASE_K_MOT 0.15f         // s (RPM time constant)

#define BASE_K_ANG_DAMP 0.0f // angular damping coefficient
#define BASE_B_DRAG 0.0f     // linear drag coefficient
#define BASE_MAX_VEL 20.0f   // m/s
#define BASE_MAX_OMEGA 20.0f // rad/s

// Simulation properties
#define GRID_X 10.0f
#define GRID_Y 10.0f
#define GRID_Z 5.0f
#define MARGIN_X (GRID_X - 1)
#define MARGIN_Y (GRID_Y - 1)
#define MARGIN_Z (GRID_Z - 1)
#define RING_RADIUS 2.0f
#define V_TARGET 0.05f

#define DRONE_OBS_SIZE 19

// Core Parameters
#define DT 0.002f // 500 Hz
#define ACTION_SUBSTEPS 5
#define ACTION_DT (DT * (float)ACTION_SUBSTEPS) // 100 Hz

#define DT_RNG 0.0f

#define MAX_DIST                                                                                   \
    sqrtf((2 * GRID_X) * (2 * GRID_X) + (2 * GRID_Y) * (2 * GRID_Y) + (2 * GRID_Z) * (2 * GRID_Z))

typedef struct {
    float w, x, y, z;
} Quat;

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Quat orientation;
    Vec3 normal;
    float radius;
} Target;

typedef struct {
    Vec3 pos[TRAIL_LENGTH];
    int index;
    int count;
} Trail;

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Quat quat;
    Vec3 omega;
    float rpms[4];
} State;

typedef struct {
    Vec3 vel;
    Vec3 v_dot;
    Quat q_dot;
    Vec3 w_dot;
    float rpm_dot[4];
} StateDerivative;

typedef struct {
    float mass;
    float ixx;
    float iyy;
    float izz;
    float arm_len;
    float k_thrust;
    float k_ang_damp;
    float k_drag;
    float b_drag;
    float gravity;
    float max_rpm;
    float max_vel;
    float max_omega;
    float k_mot;
} Params;

typedef struct {
    State state;
    Params params;
    Vec3 prev_pos;
    Target* target;

    float episode_return;
    int episode_length;
} Drone;

// math

static inline float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static inline float rndf(float a, float b, unsigned int* rng) {
    return a + ((float)rand_r(rng) / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 sub3(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 scalmul3(Vec3 a, float b) { return (Vec3){a.x * b, a.y * b, a.z * b}; }

static inline Quat add_quat(Quat a, Quat b) {
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}
static inline Quat scalmul_quat(Quat a, float b) {
    return (Quat){a.w * b, a.x * b, a.y * b, a.z * b};
}

static inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline void clamp3(Vec3* vec, float min, float max) {
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

static inline void quat_normalize(Quat* q) {
    float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        q->w /= n;
        q->x /= n;
        q->y /= n;
        q->z /= n;
    }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
    Quat qv = (Quat){0.0f, v.x, v.y, v.z};
    Quat tmp = quat_mul(q, qv);
    Quat q_conj = (Quat){q.w, -q.x, -q.y, -q.z};
    Quat res = quat_mul(tmp, q_conj);
    return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) { return (Quat){q.w, -q.x, -q.y, -q.z}; }

static inline Quat rndquat(unsigned int* rng) {
    float u1 = rndf(0.0f, 1.0f, rng);
    float u2 = rndf(0.0f, 1.0f, rng);
    float u3 = rndf(0.0f, 1.0f, rng);
    float s1 = sqrtf(1.0f - u1), s2 = sqrtf(u1);
    float a = 2.0f * (float)M_PI * u2, b = 2.0f * (float)M_PI * u3;
    return (Quat){s1 * sinf(a), s1 * cosf(a), s2 * sinf(b), s2 * cosf(b)};
}

static inline Quat quat_from_axis_angle(Vec3 axis, float angle) {
    float half = angle * 0.5f;
    float s = sinf(half);
    return (Quat){cosf(half), axis.x * s, axis.y * s, axis.z * s};
}

static inline Target rndring(unsigned int* rng, float radius) {
    Target ring = (Target){0};
    ring.pos.x = rndf(-GRID_X + 2 * radius, GRID_X - 2 * radius, rng);
    ring.pos.y = rndf(-GRID_Y + 2 * radius, GRID_Y - 2 * radius, rng);
    ring.pos.z = rndf(-GRID_Z + 2 * radius, GRID_Z - 2 * radius, rng);
    ring.orientation = rndquat(rng);
    ring.normal = quat_rotate(ring.orientation, (Vec3){0.0f, 0.0f, 1.0f});
    ring.radius = radius;
    return ring;
}

static inline Vec3 random_pos(unsigned int* rng) {
    return (Vec3){
        rndf(-MARGIN_X, MARGIN_X, rng),
        rndf(-MARGIN_Y, MARGIN_Y, rng),
        rndf(-MARGIN_Z, MARGIN_Z, rng),
    };
}

// physics

static inline float rpm_hover(const Params* p) {
    return sqrtf((p->mass * p->gravity) / (4.0f * p->k_thrust));
}

static inline float rpm_min_for_centered_hover(const Params* p) {
    float min_rpm = 2.0f * rpm_hover(p) - p->max_rpm;
    if (min_rpm < 0.0f) min_rpm = 0.0f;
    if (min_rpm > p->max_rpm) min_rpm = p->max_rpm;
    return min_rpm;
}

static inline void init_drone(Drone* drone, unsigned int* rng, float dr) {
    drone->params.arm_len = BASE_ARM_LEN * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.mass = BASE_MASS * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.ixx = BASE_IXX * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.iyy = BASE_IYY * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.izz = BASE_IZZ * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.k_thrust = BASE_K_THRUST * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.k_ang_damp = BASE_K_ANG_DAMP * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.k_drag = BASE_K_DRAG * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.b_drag = BASE_B_DRAG * rndf(1.0f - dr, 1.0f + dr, rng);
    drone->params.gravity = BASE_GRAVITY * rndf(0.99f, 1.01f, rng);
    drone->params.max_rpm = BASE_MAX_RPM;
    drone->params.max_vel = BASE_MAX_VEL;
    drone->params.max_omega = BASE_MAX_OMEGA;
    drone->params.k_mot = BASE_K_MOT * rndf(1.0f - dr, 1.0f + dr, rng);

    float hover = rpm_hover(&drone->params);
    for (int i = 0; i < 4; i++)
        drone->state.rpms[i] = hover;

    drone->state.pos = (Vec3){0, 0, 0};
    drone->prev_pos = drone->state.pos;
    drone->state.vel = (Vec3){0, 0, 0};
    drone->state.omega = (Vec3){0, 0, 0};
    drone->state.quat = (Quat){1, 0, 0, 0};
}

static inline void compute_derivatives(State* state, Params* params, float* actions,
                                       StateDerivative* d) {
    float min_rpm = rpm_min_for_centered_hover(params);

    float target_rpms[4];
    for (int i = 0; i < 4; i++) {
        float u = (actions[i] + 1.0f) * 0.5f;
        target_rpms[i] = min_rpm + u * (params->max_rpm - min_rpm);
    }

    for (int i = 0; i < 4; i++)
        d->rpm_dot[i] = (1.0f / params->k_mot) * (target_rpms[i] - state->rpms[i]);

    float T[4];
    for (int i = 0; i < 4; i++) {
        float rpm = state->rpms[i] < 0.0f ? 0.0f : state->rpms[i];
        T[i] = params->k_thrust * rpm * rpm;
    }

    Vec3 F_prop = quat_rotate(state->quat, (Vec3){0, 0, T[0] + T[1] + T[2] + T[3]});

    d->vel = state->vel;
    d->v_dot = (Vec3){
        (F_prop.x - params->b_drag * state->vel.x) / params->mass,
        (F_prop.y - params->b_drag * state->vel.y) / params->mass,
        ((F_prop.z - params->b_drag * state->vel.z) / params->mass) - params->gravity,
    };

    Quat omega_q = (Quat){0, state->omega.x, state->omega.y, state->omega.z};
    d->q_dot = scalmul_quat(quat_mul(state->quat, omega_q), 0.5f);

    float af = params->arm_len / sqrtf(2.0f);
    Vec3 tau_prop = {
        af * ((T[2] + T[3]) - (T[0] + T[1])),
        af * ((T[1] + T[2]) - (T[0] + T[3])),
        params->k_drag * (-T[0] + T[1] - T[2] + T[3]),
    };
    Vec3 tau_aero = scalmul3(state->omega, -params->k_ang_damp);
    Vec3 tau_iner = {
        (params->iyy - params->izz) * state->omega.y * state->omega.z,
        (params->izz - params->ixx) * state->omega.z * state->omega.x,
        (params->ixx - params->iyy) * state->omega.x * state->omega.y,
    };

    d->w_dot = (Vec3){
        (tau_prop.x + tau_aero.x + tau_iner.x) / params->ixx,
        (tau_prop.y + tau_aero.y + tau_iner.y) / params->iyy,
        (tau_prop.z + tau_aero.z + tau_iner.z) / params->izz,
    };
}

static inline void step(State* s, StateDerivative* d, float dt, State* out) {
    out->pos = add3(s->pos, scalmul3(d->vel, dt));
    out->vel = add3(s->vel, scalmul3(d->v_dot, dt));
    out->quat = add_quat(s->quat, scalmul_quat(d->q_dot, dt));
    out->omega = add3(s->omega, scalmul3(d->w_dot, dt));
    for (int i = 0; i < 4; i++)
        out->rpms[i] = s->rpms[i] + d->rpm_dot[i] * dt;
    quat_normalize(&out->quat);
}

static inline void rk4_step(State* state, Params* params, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    State tmp;

    compute_derivatives(state, params, actions, &k1);
    step(state, &k1, dt * 0.5f, &tmp);
    compute_derivatives(&tmp, params, actions, &k2);
    step(state, &k2, dt * 0.5f, &tmp);
    compute_derivatives(&tmp, params, actions, &k3);
    step(state, &k3, dt, &tmp);
    compute_derivatives(&tmp, params, actions, &k4);

    float dt6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2 * k2.vel.x + 2 * k3.vel.x + k4.vel.x) * dt6;
    state->pos.y += (k1.vel.y + 2 * k2.vel.y + 2 * k3.vel.y + k4.vel.y) * dt6;
    state->pos.z += (k1.vel.z + 2 * k2.vel.z + 2 * k3.vel.z + k4.vel.z) * dt6;

    state->vel.x += (k1.v_dot.x + 2 * k2.v_dot.x + 2 * k3.v_dot.x + k4.v_dot.x) * dt6;
    state->vel.y += (k1.v_dot.y + 2 * k2.v_dot.y + 2 * k3.v_dot.y + k4.v_dot.y) * dt6;
    state->vel.z += (k1.v_dot.z + 2 * k2.v_dot.z + 2 * k3.v_dot.z + k4.v_dot.z) * dt6;

    state->quat.w += (k1.q_dot.w + 2 * k2.q_dot.w + 2 * k3.q_dot.w + k4.q_dot.w) * dt6;
    state->quat.x += (k1.q_dot.x + 2 * k2.q_dot.x + 2 * k3.q_dot.x + k4.q_dot.x) * dt6;
    state->quat.y += (k1.q_dot.y + 2 * k2.q_dot.y + 2 * k3.q_dot.y + k4.q_dot.y) * dt6;
    state->quat.z += (k1.q_dot.z + 2 * k2.q_dot.z + 2 * k3.q_dot.z + k4.q_dot.z) * dt6;

    state->omega.x += (k1.w_dot.x + 2 * k2.w_dot.x + 2 * k3.w_dot.x + k4.w_dot.x) * dt6;
    state->omega.y += (k1.w_dot.y + 2 * k2.w_dot.y + 2 * k3.w_dot.y + k4.w_dot.y) * dt6;
    state->omega.z += (k1.w_dot.z + 2 * k2.w_dot.z + 2 * k3.w_dot.z + k4.w_dot.z) * dt6;

    for (int i = 0; i < 4; i++)
        state->rpms[i] +=
            (k1.rpm_dot[i] + 2 * k2.rpm_dot[i] + 2 * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt6;

    quat_normalize(&state->quat);
}

static inline void move_drone(Drone* drone, float* actions) {
    clamp4(actions, -1.0f, 1.0f);
    for (int s = 0; s < ACTION_SUBSTEPS; s++) {
        rk4_step(&drone->state, &drone->params, actions, DT);
        clamp3(&drone->state.vel, -drone->params.max_vel, drone->params.max_vel);
        clamp3(&drone->state.omega, -drone->params.max_omega, drone->params.max_omega);
        for (int i = 0; i < 4; i++)
            drone->state.rpms[i] = clampf(drone->state.rpms[i], 0.0f, drone->params.max_rpm);
    }
}

// observations

void compute_drone_observations(Drone* agent, float* observations) {
    int idx = 0;
    Quat q = agent->state.quat;
    Quat q_inv = quat_inverse(q);
    Vec3 vel_body = quat_rotate(q_inv, agent->state.vel);
    Vec3 to_target = quat_rotate(q_inv, sub3(agent->target->pos, agent->state.pos));

    float denom = agent->params.max_vel * 1.7320508f;
    observations[idx++] = vel_body.x / denom;
    observations[idx++] = vel_body.y / denom;
    observations[idx++] = vel_body.z / denom;

    observations[idx++] = agent->state.omega.x / agent->params.max_omega;
    observations[idx++] = agent->state.omega.y / agent->params.max_omega;
    observations[idx++] = agent->state.omega.z / agent->params.max_omega;

    observations[idx++] = q.w;
    observations[idx++] = q.x;
    observations[idx++] = q.y;
    observations[idx++] = q.z;

    // this is body frame so we have to be careful about scaling
    // because distances are relative to the drone orientation
    observations[idx++] = tanhf(to_target.x * 0.1f);
    observations[idx++] = tanhf(to_target.y * 0.1f);
    observations[idx++] = tanhf(to_target.z * 0.1f);

    observations[idx++] = tanhf(to_target.x * 10.0f);
    observations[idx++] = tanhf(to_target.y * 10.0f);
    observations[idx++] = tanhf(to_target.z * 10.0f);

    Vec3 normal_body = quat_rotate(q_inv, agent->target->normal);
    observations[idx++] = normal_body.x;
    observations[idx++] = normal_body.y;
    observations[idx++] = normal_body.z;
}