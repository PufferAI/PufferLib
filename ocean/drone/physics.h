// SIMD physics kernel

#pragma once

#include <string.h>

#include "dronelib.h"
#include "puffer_os.h"  // puffer_aligned_alloc/free

#define DRONE_LANES 8

// types

typedef enum {
    INTEGRATOR_RK4 = 0,
    INTEGRATOR_RK2 = 1,
} IntegratorType;

typedef float vf __attribute__((vector_size(sizeof(float) * DRONE_LANES)));

typedef struct {
    vf x, y, z;
} Vec3v;

typedef struct {
    vf w, x, y, z;
} Quatv;

typedef struct {
    Vec3v pos;
    Vec3v vel;
    Quatv quat;
    Vec3v omega;
    vf rpms[4];
} Statev;

typedef struct {
    vf mass;
    vf ixx;
    vf iyy;
    vf izz;
    vf arm_len;
    vf k_thrust;
    vf k_ang_damp;
    vf k_drag;
    vf b_drag;
    vf gravity;
    vf max_rpm;
    vf max_vel;
    vf max_omega;
    vf k_mot;
    vf inv_mass;
    vf inv_ixx;
    vf inv_iyy;
    vf inv_izz;
    vf inv_k_mot;
} Paramsv;

typedef struct {
    Vec3v vel;
    Vec3v v_dot;
    Quatv q_dot;
    Vec3v w_dot;
    vf rpm_dot[4];
} StateDerivativev;

// math

static inline Vec3v vadd3(Vec3v a, Vec3v b) { return (Vec3v){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3v vscalmul3(Vec3v a, float b) { return (Vec3v){a.x * b, a.y * b, a.z * b}; }

static inline Quatv vadd_quat(Quatv a, Quatv b) {
    return (Quatv){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Quatv vscalmul_quat(Quatv a, float b) {
    return (Quatv){a.w * b, a.x * b, a.y * b, a.z * b};
}

static inline Quatv vquat_mul(Quatv q1, Quatv q2) {
    return (Quatv){
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
    };
}

static inline void vquat_normalize(Quatv* q) {
    vf n = __builtin_elementwise_sqrt(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    q->w /= n;
    q->x /= n;
    q->y /= n;
    q->z /= n;
}

static inline vf vclampf(vf v, vf lo, vf hi) {
    return __builtin_elementwise_min(__builtin_elementwise_max(v, lo), hi);
}

static inline void vclamp3(Vec3v* v, vf lo, vf hi) {
    v->x = vclampf(v->x, lo, hi);
    v->y = vclampf(v->y, lo, hi);
    v->z = vclampf(v->z, lo, hi);
}

static inline vf vrpm_hover(const Paramsv* p) {
    return __builtin_elementwise_sqrt((p->mass * p->gravity) / (4.0f * p->k_thrust));
}

static inline vf vrpm_min_for_centered_hover(const Paramsv* p) {
    vf min_rpm = 2.0f * vrpm_hover(p) - p->max_rpm;
    return __builtin_elementwise_min(__builtin_elementwise_max(min_rpm, (vf){0}), p->max_rpm);
}

// dynamics

static inline void compute_derivatives(Statev* state, const Paramsv* params, const vf* target_rpms,
                                       StateDerivativev* d) {
    for (int i = 0; i < 4; i++)
        d->rpm_dot[i] = params->inv_k_mot * (target_rpms[i] - state->rpms[i]);

    vf T[4];
    for (int i = 0; i < 4; i++) {
        vf rpm = __builtin_elementwise_max(state->rpms[i], (vf){0});
        T[i] = params->k_thrust * rpm * rpm;
    }

    vf Tsum = T[0] + T[1] + T[2] + T[3];
    Quatv q = state->quat;
    Vec3v F_prop = {
        Tsum * 2.0f * (q.x * q.z + q.w * q.y),
        Tsum * 2.0f * (q.y * q.z - q.w * q.x),
        Tsum * (q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z),
    };

    d->vel = state->vel;
    d->v_dot = (Vec3v){
        (F_prop.x - params->b_drag * state->vel.x) * params->inv_mass,
        (F_prop.y - params->b_drag * state->vel.y) * params->inv_mass,
        ((F_prop.z - params->b_drag * state->vel.z) * params->inv_mass) - params->gravity,
    };

    Quatv omega_q = (Quatv){(vf){0}, state->omega.x, state->omega.y, state->omega.z};
    d->q_dot = vscalmul_quat(vquat_mul(state->quat, omega_q), 0.5f);

    vf af = params->arm_len * 0.70710678f; // 1/sqrt(2)
    Vec3v tau_prop = {
        af * ((T[2] + T[3]) - (T[0] + T[1])),
        af * ((T[1] + T[2]) - (T[0] + T[3])),
        params->k_drag * (-T[0] + T[1] - T[2] + T[3]),
    };
    Vec3v tau_aero = {
        -params->k_ang_damp * state->omega.x,
        -params->k_ang_damp * state->omega.y,
        -params->k_ang_damp * state->omega.z,
    };
    Vec3v tau_iner = {
        (params->iyy - params->izz) * state->omega.y * state->omega.z,
        (params->izz - params->ixx) * state->omega.z * state->omega.x,
        (params->ixx - params->iyy) * state->omega.x * state->omega.y,
    };

    d->w_dot = (Vec3v){
        (tau_prop.x + tau_aero.x + tau_iner.x) * params->inv_ixx,
        (tau_prop.y + tau_aero.y + tau_iner.y) * params->inv_iyy,
        (tau_prop.z + tau_aero.z + tau_iner.z) * params->inv_izz,
    };
}

static inline void step(Statev* s, StateDerivativev* d, float dt, Statev* out) {
    out->pos = vadd3(s->pos, vscalmul3(d->vel, dt));
    out->vel = vadd3(s->vel, vscalmul3(d->v_dot, dt));
    out->quat = vadd_quat(s->quat, vscalmul_quat(d->q_dot, dt));
    out->omega = vadd3(s->omega, vscalmul3(d->w_dot, dt));
    for (int i = 0; i < 4; i++)
        out->rpms[i] = s->rpms[i] + d->rpm_dot[i] * dt;
    vquat_normalize(&out->quat);
}

static inline void rk2_step(Statev* state, const Paramsv* params, const vf* target_rpms, float dt) {
    StateDerivativev k1, k2;
    Statev mid;

    compute_derivatives(state, params, target_rpms, &k1);
    step(state, &k1, dt * 0.5f, &mid);
    compute_derivatives(&mid, params, target_rpms, &k2);
    step(state, &k2, dt, state);
}

static inline void rk4_step(Statev* state, const Paramsv* params, const vf* target_rpms, float dt) {
    StateDerivativev k1, k2, k3, k4;
    Statev tmp;

    compute_derivatives(state, params, target_rpms, &k1);
    step(state, &k1, dt * 0.5f, &tmp);
    compute_derivatives(&tmp, params, target_rpms, &k2);
    step(state, &k2, dt * 0.5f, &tmp);
    compute_derivatives(&tmp, params, target_rpms, &k3);
    step(state, &k3, dt, &tmp);
    compute_derivatives(&tmp, params, target_rpms, &k4);

    float dt6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt6;

    state->quat.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt6;
    state->quat.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt6;
    state->quat.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt6;
    state->quat.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt6;

    for (int i = 0; i < 4; i++) {
        state->rpms[i] += (k1.rpm_dot[i] + 2.0f * k2.rpm_dot[i] + 2.0f * k3.rpm_dot[i] + k4.rpm_dot[i]) * dt6;
    }

    vquat_normalize(&state->quat);
}

// soa access

static inline void set3(Vec3v* v, int l, Vec3 a) { v->x[l] = a.x; v->y[l] = a.y; v->z[l] = a.z; }
static inline Vec3 get3(const Vec3v* v, int l) { return (Vec3){v->x[l], v->y[l], v->z[l]}; }
static inline void setq(Quatv* q, int l, Quat a) { q->w[l] = a.w; q->x[l] = a.x; q->y[l] = a.y; q->z[l] = a.z; }
static inline Quat getq(const Quatv* q, int l) { return (Quat){q->w[l], q->x[l], q->y[l], q->z[l]}; }

// engine

typedef struct {
    int num_drones;
    Statev* state;
    Paramsv* params;
    int integrator;
} Physics;

static inline void physics_init(Physics* phys, int num_drones, int integrator) {
    int num_blocks = (num_drones + DRONE_LANES - 1) / DRONE_LANES;
    phys->num_drones = num_drones;
    phys->integrator = integrator;

    phys->params = (Paramsv*)puffer_aligned_alloc(32, num_blocks * sizeof(Paramsv));
    memset(phys->params, 0, num_blocks * sizeof(Paramsv));

    phys->state = (Statev*)puffer_aligned_alloc(32, num_blocks * sizeof(Statev));
    memset(phys->state, 0, num_blocks * sizeof(Statev));
    for (int b = 0; b < num_blocks; b++)
        for (int l = 0; l < DRONE_LANES; l++)
            phys->state[b].quat.w[l] = 1.0f;
}

static inline void physics_close(Physics* phys) {
    puffer_aligned_free(phys->params);
    puffer_aligned_free(phys->state);
}

static inline void physics_set_drone(Physics* phys, int i, const Params* p, const State* st) {
    int b = i / DRONE_LANES;
    int l = i % DRONE_LANES;

    Paramsv* pv = &phys->params[b];
    pv->mass[l] = p->mass;
    pv->ixx[l] = p->ixx;
    pv->iyy[l] = p->iyy;
    pv->izz[l] = p->izz;
    pv->arm_len[l] = p->arm_len;
    pv->k_thrust[l] = p->k_thrust;
    pv->k_ang_damp[l] = p->k_ang_damp;
    pv->k_drag[l] = p->k_drag;
    pv->b_drag[l] = p->b_drag;
    pv->gravity[l] = p->gravity;
    pv->max_rpm[l] = p->max_rpm;
    pv->max_vel[l] = p->max_vel;
    pv->max_omega[l] = p->max_omega;
    pv->k_mot[l] = p->k_mot;
    pv->inv_mass[l] = p->inv_mass;
    pv->inv_ixx[l] = p->inv_ixx;
    pv->inv_iyy[l] = p->inv_iyy;
    pv->inv_izz[l] = p->inv_izz;
    pv->inv_k_mot[l] = p->inv_k_mot;

    Statev* sv = &phys->state[b];
    set3(&sv->pos, l, st->pos);
    set3(&sv->vel, l, st->vel);
    setq(&sv->quat, l, st->quat);
    set3(&sv->omega, l, st->omega);
    for (int k = 0; k < 4; k++) sv->rpms[k][l] = st->rpms[k];
}

static inline State physics_get_state(const Physics* phys, int i) {
    const Statev* sv = &phys->state[i / DRONE_LANES];
    int l = i % DRONE_LANES;
    State st;
    st.pos = get3(&sv->pos, l);
    st.vel = get3(&sv->vel, l);
    st.quat = getq(&sv->quat, l);
    st.omega = get3(&sv->omega, l);
    for (int k = 0; k < 4; k++) st.rpms[k] = sv->rpms[k][l];
    return st;
}

static inline void physics_step(Physics* phys, float* actions) {
    for (int base = 0; base < phys->num_drones; base += DRONE_LANES) {
        int lanes = phys->num_drones - base;
        if (lanes > DRONE_LANES) lanes = DRONE_LANES;

        Statev* s = &phys->state[base / DRONE_LANES];
        const Paramsv* p = &phys->params[base / DRONE_LANES];

        vf act[4];
        for (int l = 0; l < DRONE_LANES; l++) {
            int idx = base + (l < lanes ? l : 0);
            float* a = &actions[4 * idx];
            clamp4(a, -1.0f, 1.0f);
            for (int k = 0; k < 4; k++) act[k][l] = a[k];
        }

        vf min_rpm = vrpm_min_for_centered_hover(p);
        vf target_rpms[4];
        for (int k = 0; k < 4; k++) {
            vf u = (act[k] + 1.0f) * 0.5f;
            target_rpms[k] = min_rpm + u * (p->max_rpm - min_rpm);
        }

        for (int sub = 0; sub < ACTION_SUBSTEPS; sub++) {
            if (phys->integrator == INTEGRATOR_RK2) {
                rk2_step(s, p, target_rpms, DT);
            } else {
                rk4_step(s, p, target_rpms, DT);
            }
        
            vclamp3(&s->vel, -p->max_vel, p->max_vel);
            vclamp3(&s->omega, -p->max_omega, p->max_omega);
        
            for (int k = 0; k < 4; k++) {
                s->rpms[k] = vclampf(s->rpms[k], (vf){0}, p->max_rpm);
            }
        }
    }
}
