// flightlib.h - Realistic RK4 flight physics for dogfight environment
//
// Full 6-DOF flight model with:
//   - Angular momentum as state variable (omega integrated, not commanded)
//   - RK4 integration (4th-order Runge-Kutta)
//   - Aerodynamic moments from stability derivatives
//   - Control surface effectiveness (elevator, aileron, rudder)
//   - Euler's equations for rotational dynamics

#ifndef FLIGHTLIB_H
#define FLIGHTLIB_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef PI
#define PI 3.14159265358979f
#endif

// Debug control (0=off, 1+=increasingly verbose)
#ifndef DEBUG_REALISTIC
#define DEBUG_REALISTIC 0
#endif

static int _realistic_step_count = 0;
static int _realistic_rk4_stage = 0;

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

static inline Quat quat(float w, float x, float y, float z) { return (Quat){w, x, y, z}; }

static inline Quat quat_mul(Quat a, Quat b) {
    return (Quat){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline Quat quat_add(Quat a, Quat b) {
    return (Quat){a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Quat quat_scale(Quat q, float s) {
    return (Quat){q.w * s, q.x * s, q.y * s, q.z * s};
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

// Aircraft parameters - P-51D Mustang (see P51d_REFERENCE_DATA.md)

#define MASS 4082.0f           // kg
#define WING_AREA 21.65f       // m^2
#define WINGSPAN 11.28f        // m
#define CHORD 2.02f            // m

#define IXX 6500.0f    // Roll inertia
#define IYY 22000.0f   // Pitch inertia
#define IZZ 27000.0f   // Yaw inertia

#define C_D0 0.0163f
#define K 0.072f
#define K_SIDESLIP 0.7f
#define C_L_MAX 1.48f
#define C_L_ALPHA 5.56f
#define ALPHA_ZERO -0.021f
#define WING_INCIDENCE 0.026f

#define ENGINE_POWER 1112000.0f // watts
#define ETA_PROP 0.80f

#define GRAVITY 9.81f          // m/s^2
#define RHO 1.225f             // kg/m^3

#define G_LIMIT_POS 6.0f
#define G_LIMIT_NEG 1.5f

#define INV_MASS     0.000245f   // 1/4082
#define INV_GRAVITY  0.10197f    // 1/9.81
#define RAD_TO_DEG   57.2957795f // 180/PI

#define MAX_PITCH_RATE 2.5f    // rad/s
#define MAX_ROLL_RATE 3.0f     // rad/s
#define MAX_YAW_RATE 0.50f     // rad/s

typedef struct {
    Vec3 pos;
    Vec3 vel;
    Vec3 prev_vel;
    Vec3 omega;
    Quat ori;
    float throttle;
    float g_force;
    float yaw_from_rudder;
    int fire_cooldown;
    float prev_energy;  // Previous specific energy for energy management reward
} Plane;

static inline void step_plane(Plane *p, float dt) {
    p->prev_vel = p->vel;

    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    float speed = norm3(p->vel);
    if (speed < 1.0f) speed = 80.0f;
    p->vel = mul3(forward, speed);
    p->pos = add3(p->pos, mul3(p->vel, dt));

    if (DEBUG >= 10) printf("=== TARGET ===\n");
    if (DEBUG >= 10) printf("target_speed=%.1f m/s (expected=80)\n", speed);
    if (DEBUG >= 10) printf("target_pos=(%.1f, %.1f, %.1f)\n", p->pos.x, p->pos.y, p->pos.z);
    if (DEBUG >= 10) printf("target_fwd=(%.2f, %.2f, %.2f)\n", forward.x, forward.y, forward.z);
}

#define CM_0 -0.005f        // Pitch trim offset (fine-tuned for ~1.0G level flight)
#define CM_ALPHA -1.2f      // Pitch stability (negative = stable, nose-up creates nose-down moment)
#define CL_BETA -0.08f      // Dihedral effect (negative = stable, sideslip creates restoring roll)
#define CN_BETA 0.12f       // Weathervane stability (positive = stable, sideslip creates restoring yaw)

// Damping derivatives (dimensionless, multiplied by q*c/2V or p*b/2V)
#define CM_Q -10.0f         // Pitch damping (matches JSBSim P-51D)
#define CL_P -0.4f          // Roll damping (opposes roll rate)
#define CN_R -0.15f         // Yaw damping (opposes yaw rate)

// Control derivatives (per radian deflection)
#define CM_DELTA_E -0.5f    // Elevator: negative = nose UP with positive (back stick) deflection
#define CL_DELTA_A 0.20f    // Aileron: positive = roll RIGHT with positive deflection
                            // Tuning: 0.04f->19°, 0.15f->70°, need 90°, try 0.20f
#define CN_DELTA_R 0.015f   // Rudder: positive = nose RIGHT with positive (right pedal) deflection
                            // Tuning: 0.015f should give 2-20° heading change with full rudder

// Cross-coupling derivatives
#define CN_DELTA_A -0.007f  // Adverse yaw from aileron (negative = right aileron causes left yaw)
#define CL_DELTA_R -0.003f  // Roll from rudder (negative = right rudder causes left roll, rudder is above roll axis)

// Control surface deflection limits (radians)
#define MAX_ELEVATOR_DEFLECTION 0.35f   // ±20°
#define MAX_AILERON_DEFLECTION 0.35f    // ±20°
#define MAX_RUDDER_DEFLECTION 0.35f     // ±20°

// High-speed control authority scaling (prevents oscillations at high speed).
// At high speeds, control moments scale with V^2 while damping scales with V,
// causing under-damped behavior. Reduce actuator deflection at high V so the
// generated moment stays bounded -- equivalent to PX4/ArduPilot inner-loop
// q_bar / IAS^2 scheduling, applied at the plant actuator stage instead.
//
// Values from sweep_smoothness.sh brute-force search across ~5000 configs
// scoring on the (maneuver x airspeed) smoothness envelope plus full-throttle
// max-rate maneuverability. The plateau is wide; nearby values give similar
// results. V_REF=70 (vs cruise=100) cuts authority starting earlier so the
// V=80-100 band benefits too -- the worst pre-fix oscillation was at V=100
// where the prior V_REF=100 left authority untouched.
//   V=70: 1.000   V=100: 0.700   V=120: 0.500   V>=140: 0.300 (floor)
// Effect vs no-scaling baseline: 6->16 SMOOTH, 18->8 OSCILLATING on the
// 30-case envelope, peak roll rate at V=120 (full stick) ~95 deg/s.
#define CONTROL_V_REF 70.0f
#define CONTROL_SCALE_SLOPE 0.010f
#define CONTROL_SCALE_MIN 0.300f

// Runtime-configurable physics parameters for parameter sweeps + domain randomization
typedef struct {
    // Sweep knobs (untouched by randomization)
    float control_v_ref;       // Reference speed for full authority
    float control_scale_slope; // How fast authority drops with speed
    float control_scale_min;   // Floor for control authority
    float damping_scale_slope; // Extra damping scale per m/s above ref (0 = off)
    float damping_multiplier;  // Scale CM_Q, CL_P, CN_R (1.0 = normal, 2.0 = double damping)
    // Physics params (randomized per-episode when domain_randomization > 0)
    float mass, inv_mass;
    float ixx, iyy, izz;
    float gravity, inv_gravity;
    float wing_area, wingspan, chord;
    float c_d0, k, c_l_alpha, c_l_max, rho;
    float engine_power, eta_prop;
    float cm_alpha, cl_beta, cn_beta;
    float cm_q, cl_p, cn_r;
    float cm_delta_e, cl_delta_a, cn_delta_r;
    float g_limit_pos, g_limit_neg;
} FlightParams;

// Default parameters (matches current compile-time #defines)
static inline FlightParams default_flight_params(void) {
    return (FlightParams){
        .control_v_ref = CONTROL_V_REF,
        .control_scale_slope = CONTROL_SCALE_SLOPE,
        .control_scale_min = CONTROL_SCALE_MIN,
        .damping_scale_slope = 0.0f,
        .damping_multiplier = 1.15f,  // sweep_smoothness winner; 15% over JSBSim P-51D values
        .mass = MASS, .inv_mass = 1.0f / MASS,
        .ixx = IXX, .iyy = IYY, .izz = IZZ,
        .gravity = GRAVITY, .inv_gravity = 1.0f / GRAVITY,
        .wing_area = WING_AREA, .wingspan = WINGSPAN, .chord = CHORD,
        .c_d0 = C_D0, .k = K, .c_l_alpha = C_L_ALPHA, .c_l_max = C_L_MAX, .rho = RHO,
        .engine_power = ENGINE_POWER, .eta_prop = ETA_PROP,
        .cm_alpha = CM_ALPHA, .cl_beta = CL_BETA, .cn_beta = CN_BETA,
        .cm_q = CM_Q, .cl_p = CL_P, .cn_r = CN_R,
        .cm_delta_e = CM_DELTA_E, .cl_delta_a = CL_DELTA_A, .cn_delta_r = CN_DELTA_R,
        .g_limit_pos = G_LIMIT_POS, .g_limit_neg = G_LIMIT_NEG,
    };
}

// Domain randomization: called every c_reset(). When dr=0, rndf(1,1)=1.0 → exact base values.
static inline void randomize_flight_params(FlightParams* params, float dr) {
    params->mass = MASS * rndf(1.0f - dr, 1.0f + dr);
    params->inv_mass = 1.0f / params->mass;
    params->ixx = IXX * rndf(1.0f - dr, 1.0f + dr);
    params->iyy = IYY * rndf(1.0f - dr, 1.0f + dr);
    params->izz = IZZ * rndf(1.0f - dr, 1.0f + dr);

    // Gravity: tight range (always capped at +/-1%)
    float grav_dr = fminf(dr, 0.01f);
    params->gravity = GRAVITY * rndf(1.0f - grav_dr, 1.0f + grav_dr);
    params->inv_gravity = 1.0f / params->gravity;

    // Wing geometry: correlated (single scale, area ~ length^2)
    float wing_scale = rndf(1.0f - dr, 1.0f + dr);
    params->wingspan = WINGSPAN * wing_scale;
    params->chord = CHORD * wing_scale;
    params->wing_area = WING_AREA * wing_scale * wing_scale;

    params->c_d0 = C_D0 * rndf(1.0f - dr, 1.0f + dr);
    params->k = K * rndf(1.0f - dr, 1.0f + dr);
    params->c_l_alpha = C_L_ALPHA * rndf(1.0f - dr, 1.0f + dr);
    params->c_l_max = C_L_MAX * rndf(1.0f - dr, 1.0f + dr);
    params->rho = RHO * rndf(1.0f - dr, 1.0f + dr);

    params->engine_power = ENGINE_POWER * rndf(1.0f - dr, 1.0f + dr);
    params->eta_prop = ETA_PROP * rndf(1.0f - dr, 1.0f + dr);

    params->cm_alpha = CM_ALPHA * rndf(1.0f - dr, 1.0f + dr);
    params->cl_beta = CL_BETA * rndf(1.0f - dr, 1.0f + dr);
    params->cn_beta = CN_BETA * rndf(1.0f - dr, 1.0f + dr);
    params->cm_q = CM_Q * rndf(1.0f - dr, 1.0f + dr);
    params->cl_p = CL_P * rndf(1.0f - dr, 1.0f + dr);
    params->cn_r = CN_R * rndf(1.0f - dr, 1.0f + dr);

    params->cm_delta_e = CM_DELTA_E * rndf(1.0f - dr, 1.0f + dr);
    params->cl_delta_a = CL_DELTA_A * rndf(1.0f - dr, 1.0f + dr);
    params->cn_delta_r = CN_DELTA_R * rndf(1.0f - dr, 1.0f + dr);

    params->g_limit_pos = G_LIMIT_POS * rndf(1.0f - dr, 1.0f + dr);
    params->g_limit_neg = G_LIMIT_NEG * rndf(1.0f - dr, 1.0f + dr);
}

typedef struct {
    Vec3 vel;
    Vec3 v_dot;
    Quat q_dot;
    Vec3 w_dot;
} StateDerivative;

static inline float compute_aoa(Plane* p) {
    Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));

    float V = norm3(p->vel);
    if (V < 1.0f) return 0.0f;

    Vec3 vel_norm = normalize3(p->vel);

    // Alpha = atan2(-vel·up, vel·forward)
    // Continuous everywhere — no sign discontinuity at vertical pitch.
    // Positive when nose is ABOVE velocity vector (vel has component opposite to body-up).
    float vel_dot_fwd = dot3(vel_norm, forward);
    float vel_dot_up = dot3(vel_norm, up);
    float alpha = atan2f(-vel_dot_up, vel_dot_fwd);

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [AOA] forward=(%.3f,%.3f,%.3f) up=(%.3f,%.3f,%.3f)\n",
               forward.x, forward.y, forward.z, up.x, up.y, up.z);
        printf("  [AOA] vel=(%.1f,%.1f,%.1f) |vel|=%.1f\n",
               p->vel.x, p->vel.y, p->vel.z, V);
        printf("  [AOA] vel_norm=(%.4f,%.4f,%.4f)\n",
               vel_norm.x, vel_norm.y, vel_norm.z);
        printf("  [AOA] vel·fwd=%.4f, vel·up=%.4f\n", vel_dot_fwd, vel_dot_up);
        printf("  [AOA] FINAL alpha=%.4f rad = %.2f deg\n", alpha, alpha * RAD_TO_DEG);
    }

    return alpha;
}

static inline float compute_sideslip(Plane* p) {
    Vec3 right = quat_rotate(p->ori, vec3(0, 1, 0));

    float V = norm3(p->vel);
    if (V < 1.0f) return 0.0f;

    Vec3 vel_norm = normalize3(p->vel);

    // beta = arcsin(v · right / |v|) - positive when velocity has component to the right
    float sin_beta = dot3(vel_norm, right);
    float beta = asinf(clampf(sin_beta, -1.0f, 1.0f));

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [BETA] right=(%.3f,%.3f,%.3f)\n", right.x, right.y, right.z);
        printf("  [BETA] sin_beta=%.4f (vel_norm·right)\n", sin_beta);
        printf("  [BETA] FINAL beta=%.4f rad = %.2f deg\n", beta, beta * RAD_TO_DEG);
    }

    return beta;
}

static inline Vec3 compute_lift_direction(Vec3 vel_norm, Vec3 right, Vec3 body_up) {
    Vec3 lift_dir = cross3(vel_norm, right);
    float mag = norm3(lift_dir);

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [LIFT_DIR] vel_norm×right=(%.3f,%.3f,%.3f) |mag|=%.4f\n",
               lift_dir.x, lift_dir.y, lift_dir.z, mag);
    }

    if (mag > 0.01f) {
        Vec3 result = mul3(lift_dir, 1.0f / mag);
        if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
            printf("  [LIFT_DIR] normalized=(%.3f,%.3f,%.3f)\n", result.x, result.y, result.z);
        }
        return result;
    }
    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [LIFT_DIR] FALLBACK to body_up=(%.3f,%.3f,%.3f)\n", body_up.x, body_up.y, body_up.z);
    }
    return body_up;  // Fallback to body-frame up (avoids discontinuous jump to world-up)
}

static inline float compute_thrust(float throttle, float V) {
    float P_avail = ENGINE_POWER * throttle;
    float T_dynamic = (P_avail * ETA_PROP) / V;   // Thrust from power equation
    float T_static = 0.3f * P_avail;              // Static thrust limit
    float T = fminf(T_static, T_dynamic);         // Can't exceed either limit

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("  [THRUST] throttle=%.2f P_avail=%.0f W\n", throttle, P_avail);
        printf("  [THRUST] T_dynamic=%.0f N, T_static=%.0f N -> T=%.0f N\n",
               T_dynamic, T_static, T);
    }

    return T;
}

// Helper: apply derivative to state (for RK4 intermediate stages)
static inline void step_temp(Plane* state, StateDerivative* d, float dt, Plane* out) {
    out->pos = add3(state->pos, mul3(d->vel, dt));
    out->vel = add3(state->vel, mul3(d->v_dot, dt));
    out->ori = quat_add(state->ori, quat_scale(d->q_dot, dt));
    quat_normalize(&out->ori);
    out->omega = add3(state->omega, mul3(d->w_dot, dt));
    out->throttle = state->throttle;
    out->g_force = state->g_force;
    out->yaw_from_rudder = state->yaw_from_rudder;
    out->fire_cooldown = state->fire_cooldown;
    out->prev_vel = state->prev_vel;

    if (DEBUG_REALISTIC >= 5) {
        printf("    [STEP_TEMP] dt=%.4f\n", dt);
        printf("    [STEP_TEMP] d->vel=(%.2f,%.2f,%.2f) d->v_dot=(%.2f,%.2f,%.2f)\n",
               d->vel.x, d->vel.y, d->vel.z, d->v_dot.x, d->v_dot.y, d->v_dot.z);
        printf("    [STEP_TEMP] d->w_dot=(%.4f,%.4f,%.4f)\n",
               d->w_dot.x, d->w_dot.y, d->w_dot.z);
        printf("    [STEP_TEMP] out->vel=(%.2f,%.2f,%.2f)\n", out->vel.x, out->vel.y, out->vel.z);
        printf("    [STEP_TEMP] out->omega=(%.4f,%.4f,%.4f)\n",
               out->omega.x, out->omega.y, out->omega.z);
        printf("    [STEP_TEMP] out->ori=(%.4f,%.4f,%.4f,%.4f)\n",
               out->ori.w, out->ori.x, out->ori.y, out->ori.z);
    }
}

static inline void compute_derivatives(Plane* state, float* actions, float dt, StateDerivative* deriv) {

    if (DEBUG_REALISTIC >= 5) {
        const char* stage_names[] = {"k1", "k2", "k3", "k4"};
        printf("\n  === COMPUTE_DERIVATIVES (RK4 stage %s) ===\n", stage_names[_realistic_rk4_stage]);
    }

    float V = norm3(state->vel);
    if (V < 1.0f) V = 1.0f;  // Prevent div-by-zero

    Vec3 vel_norm = normalize3(state->vel);
    Vec3 forward = quat_rotate(state->ori, vec3(1, 0, 0));  // Body X-axis
    Vec3 right = quat_rotate(state->ori, vec3(0, 1, 0));    // Body Y-axis
    Vec3 body_up = quat_rotate(state->ori, vec3(0, 0, 1));  // Body Z-axis

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- STATE ---\n");
        printf("  pos=(%.1f, %.1f, %.1f)\n", state->pos.x, state->pos.y, state->pos.z);
        printf("  vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n",
               state->vel.x, state->vel.y, state->vel.z, V);
        printf("  vel_norm=(%.4f, %.4f, %.4f)\n", vel_norm.x, vel_norm.y, vel_norm.z);
        printf("  ori=(w=%.4f, x=%.4f, y=%.4f, z=%.4f) |ori|=%.6f\n",
               state->ori.w, state->ori.x, state->ori.y, state->ori.z,
               sqrtf(state->ori.w*state->ori.w + state->ori.x*state->ori.x +
                     state->ori.y*state->ori.y + state->ori.z*state->ori.z));
        printf("  omega=(%.4f, %.4f, %.4f) rad/s = (%.2f, %.2f, %.2f) deg/s\n",
               state->omega.x, state->omega.y, state->omega.z,
               state->omega.x * RAD_TO_DEG, state->omega.y * RAD_TO_DEG, state->omega.z * RAD_TO_DEG);
        printf("  forward=(%.4f, %.4f, %.4f)\n", forward.x, forward.y, forward.z);
        printf("  right=(%.4f, %.4f, %.4f)\n", right.x, right.y, right.z);
        printf("  body_up=(%.4f, %.4f, %.4f)\n", body_up.x, body_up.y, body_up.z);

        // Compute pitch angle from forward vector
        float pitch_from_forward = asinf(-forward.z) * RAD_TO_DEG;  // nose up = positive
        printf("  pitch_from_forward=%.2f deg (nose %s)\n",
               pitch_from_forward, pitch_from_forward > 0 ? "UP" : "DOWN");

        // Velocity direction
        float vel_pitch = asinf(vel_norm.z) * RAD_TO_DEG;  // climbing = positive
        printf("  vel_pitch=%.2f deg (%s)\n", vel_pitch, vel_pitch > 0 ? "CLIMBING" : "DESCENDING");
    }

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- AERODYNAMIC ANGLES ---\n");
    }
    float alpha = compute_aoa(state);
    float beta = compute_sideslip(state);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("  alpha=%.4f rad = %.2f deg (%s)\n", alpha, alpha * RAD_TO_DEG,
               alpha > 0 ? "nose ABOVE vel" : "nose BELOW vel");
        printf("  beta=%.4f rad = %.2f deg\n", beta, beta * RAD_TO_DEG);
    }

    float q_bar = 0.5f * RHO * V * V;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DYNAMIC PRESSURE ---\n");
        printf("  q_bar = 0.5 * %.4f * %.1f^2 = %.1f Pa\n", RHO, V, q_bar);
    }

    // ========================================================================
    // 4. Map actions to control surface deflections
    // ========================================================================
    // Actions are [-1, 1], mapped to deflection in radians
    // Sign conventions (M_moment is negated later for Z-up frame):
    //   - Elevator: actions[1] > 0 (push forward) → nose DOWN
    //   - Aileron: actions[2] > 0 → roll RIGHT
    //   - Rudder: actions[3] > 0 → yaw LEFT
    float throttle = clampf((actions[0] + 1.0f) * 0.5f, 0.0f, 1.0f);  // [0, 1]

    // Scale control authority at high speed to prevent over-controlling
    // At high speed, control moments scale with V² while damping scales with V,
    // causing under-damped oscillations. Reduce authority to compensate.
    float control_scale = 1.0f - fmaxf(0.0f, V - CONTROL_V_REF) * CONTROL_SCALE_SLOPE;
    control_scale = fmaxf(control_scale, CONTROL_SCALE_MIN);

    float delta_e = clampf(actions[1], -1.0f, 1.0f) * MAX_ELEVATOR_DEFLECTION * control_scale;
    float delta_a = clampf(actions[2], -1.0f, 1.0f) * MAX_AILERON_DEFLECTION * control_scale;
    float delta_r = clampf(actions[3], -1.0f, 1.0f) * MAX_RUDDER_DEFLECTION * control_scale;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- CONTROLS ---\n");
        printf("  actions=[%.3f, %.3f, %.3f, %.3f]\n",
               actions[0], actions[1], actions[2], actions[3]);
        printf("  throttle=%.3f (%.0f%%)\n", throttle, throttle * 100);
        printf("  control_scale=%.3f (V=%.1f, ref=%.1f)\n", control_scale, V, CONTROL_V_REF);
        printf("  delta_e=%.4f rad = %.2f deg (elevator, %s)\n",
               delta_e, delta_e * RAD_TO_DEG,
               delta_e > 0 ? "push=nose DOWN" : delta_e < 0 ? "pull=nose UP" : "neutral");
        printf("  delta_a=%.4f rad = %.2f deg (aileron)\n", delta_a, delta_a * RAD_TO_DEG);
        printf("  delta_r=%.4f rad = %.2f deg (rudder)\n", delta_r, delta_r * RAD_TO_DEG);
    }

    float alpha_effective = alpha + WING_INCIDENCE - ALPHA_ZERO;
    float C_L_raw = C_L_ALPHA * alpha_effective;
    float C_L = clampf(C_L_raw, -C_L_MAX, C_L_MAX);  // Stall limiting

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- LIFT COEFFICIENT ---\n");
        printf("  alpha=%.4f + WING_INCIDENCE=%.4f - ALPHA_ZERO=%.4f = alpha_eff=%.4f rad\n",
               alpha, WING_INCIDENCE, ALPHA_ZERO, alpha_effective);
        printf("  C_L_raw = C_L_ALPHA(%.2f) * alpha_eff(%.4f) = %.4f\n",
               C_L_ALPHA, alpha_effective, C_L_raw);
        printf("  C_L = clamp(%.4f, -%.2f, %.2f) = %.4f%s\n",
               C_L_raw, C_L_MAX, C_L_MAX, C_L,
               (C_L != C_L_raw) ? " (STALL CLAMPED!)" : "");
    }

    float C_D0_term = C_D0;
    float induced_term = K * C_L * C_L;
    float sideslip_term = K_SIDESLIP * beta * beta;
    float C_D = C_D0_term + induced_term + sideslip_term;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DRAG COEFFICIENT ---\n");
        printf("  C_D0=%.4f + K*C_L^2=%.4f + K_sideslip*beta^2=%.4f = C_D=%.4f\n",
               C_D0_term, induced_term, sideslip_term, C_D);
        printf("  L/D ratio = %.2f\n", (C_D > 0.0001f) ? C_L / C_D : 0.0f);
    }

    float L_mag = C_L * q_bar * WING_AREA;
    float D_mag = C_D * q_bar * WING_AREA;

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- LIFT DIRECTION ---\n");
    }
    Vec3 lift_dir = compute_lift_direction(vel_norm, right, body_up);
    Vec3 F_lift = mul3(lift_dir, L_mag);

    Vec3 F_drag = mul3(vel_norm, -D_mag);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- AERODYNAMIC FORCES ---\n");
        printf("  L_mag = C_L(%.4f) * q_bar(%.1f) * S(%.1f) = %.1f N\n",
               C_L, q_bar, WING_AREA, L_mag);
        printf("  D_mag = C_D(%.4f) * q_bar(%.1f) * S(%.1f) = %.1f N\n",
               C_D, q_bar, WING_AREA, D_mag);
        printf("  lift_dir=(%.4f, %.4f, %.4f)\n", lift_dir.x, lift_dir.y, lift_dir.z);
        printf("  F_lift=(%.1f, %.1f, %.1f) N\n", F_lift.x, F_lift.y, F_lift.z);
        printf("  F_drag=(%.1f, %.1f, %.1f) N (opposite to vel)\n", F_drag.x, F_drag.y, F_drag.z);
    }

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- THRUST ---\n");
    }
    float T_mag = compute_thrust(throttle, V);
    Vec3 F_thrust = mul3(forward, T_mag);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("  F_thrust=(%.1f, %.1f, %.1f) N (along forward)\n",
               F_thrust.x, F_thrust.y, F_thrust.z);
    }

    Vec3 F_gravity = vec3(0, 0, -MASS * GRAVITY);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- GRAVITY ---\n");
        printf("  F_gravity=(%.1f, %.1f, %.1f) N\n", F_gravity.x, F_gravity.y, F_gravity.z);
    }

    Vec3 F_aero = add3(F_lift, F_drag);
    Vec3 F_aero_thrust = add3(F_aero, F_thrust);
    Vec3 F_total = add3(F_aero_thrust, F_gravity);
    deriv->v_dot = mul3(F_total, INV_MASS);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- TOTAL FORCE & ACCELERATION ---\n");
        printf("  F_aero (lift+drag)=(%.1f, %.1f, %.1f) N\n", F_aero.x, F_aero.y, F_aero.z);
        printf("  F_aero+thrust=(%.1f, %.1f, %.1f) N\n", F_aero_thrust.x, F_aero_thrust.y, F_aero_thrust.z);
        printf("  F_total=(%.1f, %.1f, %.1f) N\n", F_total.x, F_total.y, F_total.z);
        printf("  |F_total|=%.1f N\n", norm3(F_total));
        printf("  v_dot = F/m = (%.3f, %.3f, %.3f) m/s^2\n", deriv->v_dot.x, deriv->v_dot.y, deriv->v_dot.z);
        printf("  |v_dot|=%.3f m/s^2 = %.3f g\n", norm3(deriv->v_dot), norm3(deriv->v_dot) / GRAVITY);

        // Break down vertical component
        printf("  v_dot.z=%.3f m/s^2 (%s)\n", deriv->v_dot.z,
               deriv->v_dot.z > 0 ? "accelerating UP" : "accelerating DOWN");

        // What's contributing to vertical acceleration?
        printf("  Vertical breakdown: lift_z=%.1f + drag_z=%.1f + thrust_z=%.1f + grav_z=%.1f = %.1f N\n",
               F_lift.z, F_drag.z, F_thrust.z, F_gravity.z, F_total.z);
    }

    float p = state->omega.x;  // roll rate
    float q = state->omega.y;  // pitch rate
    float r = state->omega.z;  // yaw rate

    // Non-dimensional rates for damping derivatives
    float p_hat = p * WINGSPAN / (2.0f * V);
    float q_hat = q * CHORD / (2.0f * V);
    float r_hat = r * WINGSPAN / (2.0f * V);

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- ANGULAR RATES ---\n");
        printf("  p=%.4f, q=%.4f, r=%.4f rad/s (body: roll, pitch, yaw)\n", p, q, r);
        printf("  p_hat=%.6f, q_hat=%.6f, r_hat=%.6f (non-dimensional)\n", p_hat, q_hat, r_hat);
    }

    // Rolling moment coefficient (Cl)
    // Components: dihedral effect + roll damping + aileron control + rudder coupling
    float Cl_beta = CL_BETA * beta;
    float Cl_p = CL_P * p_hat;
    float Cl_da = CL_DELTA_A * delta_a;
    float Cl_dr = CL_DELTA_R * delta_r;
    float Cl = Cl_beta + Cl_p + Cl_da + Cl_dr;

    // Pitching moment coefficient (Cm)
    // Components: static stability + pitch damping + elevator control
    float Cm_0 = CM_0;  // Trim offset
    float Cm_alpha = CM_ALPHA * alpha;
    float Cm_q = CM_Q * q_hat;
    float Cm_de = CM_DELTA_E * delta_e;
    float Cm = Cm_0 + Cm_alpha + Cm_q + Cm_de;

    // Yawing moment coefficient (Cn)
    // Components: weathervane stability + yaw damping + rudder control + adverse yaw
    float Cn_beta = CN_BETA * beta;
    float Cn_r = CN_R * r_hat;
    float Cn_dr = CN_DELTA_R * delta_r;
    float Cn_da = CN_DELTA_A * delta_a;
    float Cn = Cn_beta + Cn_r + Cn_dr + Cn_da;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- MOMENT COEFFICIENTS ---\n");
        printf("  Cl = CL_BETA*beta(%.6f) + CL_P*p_hat(%.6f) + CL_DELTA_A*da(%.6f) + CL_DELTA_R*dr(%.6f) = %.6f\n",
               Cl_beta, Cl_p, Cl_da, Cl_dr, Cl);
        printf("  Cm = CM_0(%.6f) + CM_ALPHA*alpha(%.6f) + CM_Q*q_hat(%.6f) + CM_DELTA_E*de(%.6f) = %.6f\n",
               Cm_0, Cm_alpha, Cm_q, Cm_de, Cm);
        printf("       CM_0=%.4f (trim), CM_ALPHA=%.2f, alpha=%.4f rad -> Cm_alpha=%.6f\n", CM_0, CM_ALPHA, alpha, Cm_alpha);
        printf("       (alpha>0 means nose ABOVE vel, CM_ALPHA<0 means nose-down restoring moment)\n");
        printf("       (Cm_alpha %.6f is %s)\n", Cm_alpha,
               Cm_alpha > 0 ? "nose-UP moment" : Cm_alpha < 0 ? "nose-DOWN moment" : "zero");
        printf("  Cn = CN_BETA*beta(%.6f) + CN_R*r_hat(%.6f) + CN_DELTA_R*dr(%.6f) + CN_DELTA_A*da(%.6f) = %.6f\n",
               Cn_beta, Cn_r, Cn_dr, Cn_da, Cn);
    }

    // Convert to dimensional moments (N⋅m)
    // Note: Cm sign convention is for aircraft Z-down frame (positive Cm = nose up)
    // In our Z-up frame, positive omega.y = nose DOWN, so we negate Cm
    float L_moment = Cl * q_bar * WING_AREA * WINGSPAN;  // Roll moment
    float M_moment = -Cm * q_bar * WING_AREA * CHORD;    // Pitch moment (negated for Z-up frame)
    float N_moment = Cn * q_bar * WING_AREA * WINGSPAN;  // Yaw moment

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DIMENSIONAL MOMENTS ---\n");
        printf("  L_moment (roll) = Cl(%.6f) * q_bar(%.1f) * S(%.1f) * b(%.1f) = %.1f N⋅m\n",
               Cl, q_bar, WING_AREA, WINGSPAN, L_moment);
        printf("  M_moment (pitch) = -Cm(%.6f) * q_bar(%.1f) * S(%.1f) * c(%.2f) = %.1f N⋅m\n",
               Cm, q_bar, WING_AREA, CHORD, M_moment);
        printf("       Note: M_moment negated because our Z is up (positive omega.y = nose DOWN)\n");
        printf("       Cm=%.6f -> -Cm=%.6f -> M_moment=%.1f (will cause omega.y to %s)\n",
               Cm, -Cm, M_moment, M_moment > 0 ? "INCREASE (nose DOWN)" : "DECREASE (nose UP)");
        printf("  N_moment (yaw) = Cn(%.6f) * q_bar(%.1f) * S(%.1f) * b(%.1f) = %.1f N⋅m\n",
               Cn, q_bar, WING_AREA, WINGSPAN, N_moment);
    }

    // ========================================================================
    // Angular acceleration (Euler's equations)
    // ========================================================================
    // τ = I⋅α + ω × (I⋅ω)  →  α = I⁻¹(τ - ω × (I⋅ω))
    // For diagonal inertia tensor, the gyroscopic coupling terms are:
    // (I_yy - I_zz) * q * r  for roll
    // (I_zz - I_xx) * r * p  for pitch
    // (I_xx - I_yy) * p * q  for yaw

    float gyro_roll = (IYY - IZZ) * q * r;
    float gyro_pitch = (IZZ - IXX) * r * p;
    float gyro_yaw = (IXX - IYY) * p * q;

    deriv->w_dot.x = (L_moment + gyro_roll) / IXX;
    deriv->w_dot.y = (M_moment + gyro_pitch) / IYY;
    deriv->w_dot.z = (N_moment + gyro_yaw) / IZZ;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- ANGULAR ACCELERATION (Euler's equations) ---\n");
        printf("  Gyroscopic: roll=%.3f, pitch=%.3f, yaw=%.3f N⋅m\n", gyro_roll, gyro_pitch, gyro_yaw);
        printf("  I = (Ixx=%.0f, Iyy=%.0f, Izz=%.0f) kg⋅m^2\n", IXX, IYY, IZZ);
        printf("  w_dot.x (roll)  = (L=%.1f + gyro=%.3f) / Ixx = %.6f rad/s^2 = %.3f deg/s^2\n",
               L_moment, gyro_roll, deriv->w_dot.x, deriv->w_dot.x * RAD_TO_DEG);
        printf("  w_dot.y (pitch) = (M=%.1f + gyro=%.3f) / Iyy = %.6f rad/s^2 = %.3f deg/s^2\n",
               M_moment, gyro_pitch, deriv->w_dot.y, deriv->w_dot.y * RAD_TO_DEG);
        printf("  w_dot.z (yaw)   = (N=%.1f + gyro=%.3f) / Izz = %.6f rad/s^2 = %.3f deg/s^2\n",
               N_moment, gyro_yaw, deriv->w_dot.z, deriv->w_dot.z * RAD_TO_DEG);
        printf("  w_dot.y=%.6f means omega.y will %s -> nose will pitch %s\n",
               deriv->w_dot.y,
               deriv->w_dot.y > 0 ? "INCREASE" : "DECREASE",
               deriv->w_dot.y > 0 ? "DOWN" : "UP");
    }

    // q_dot = 0.5 * q * [0, ω]  where ω is angular velocity in body frame
    Quat omega_q = {0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->ori, omega_q);
    deriv->q_dot.w = 0.5f * q_dot.w;
    deriv->q_dot.x = 0.5f * q_dot.x;
    deriv->q_dot.y = 0.5f * q_dot.y;
    deriv->q_dot.z = 0.5f * q_dot.z;

    if (DEBUG_REALISTIC >= 3 && _realistic_rk4_stage == 0) {
        printf("\n  --- QUATERNION KINEMATICS ---\n");
        printf("  omega_q=(%.4f, %.4f, %.4f, %.4f)\n", omega_q.w, omega_q.x, omega_q.y, omega_q.z);
        printf("  q_dot (before 0.5)=(%.6f, %.6f, %.6f, %.6f)\n", q_dot.w, q_dot.x, q_dot.y, q_dot.z);
        printf("  q_dot (final)=(%.6f, %.6f, %.6f, %.6f)\n",
               deriv->q_dot.w, deriv->q_dot.x, deriv->q_dot.y, deriv->q_dot.z);
    }

    deriv->vel = state->vel;

    if (DEBUG_REALISTIC >= 2 && _realistic_rk4_stage == 0) {
        printf("\n  --- DERIVATIVE SUMMARY ---\n");
        printf("  vel = (%.2f, %.2f, %.2f) m/s\n", deriv->vel.x, deriv->vel.y, deriv->vel.z);
        printf("  v_dot = (%.3f, %.3f, %.3f) m/s^2\n", deriv->v_dot.x, deriv->v_dot.y, deriv->v_dot.z);
        printf("  q_dot = (%.6f, %.6f, %.6f, %.6f)\n",
               deriv->q_dot.w, deriv->q_dot.x, deriv->q_dot.y, deriv->q_dot.z);
        printf("  w_dot = (%.6f, %.6f, %.6f) rad/s^2\n", deriv->w_dot.x, deriv->w_dot.y, deriv->w_dot.z);
    }
}

// Version with runtime-configurable parameters for sweeps + domain randomization
static inline void compute_derivatives_with_params(
    Plane* state, float* actions, float dt,
    StateDerivative* deriv, FlightParams* params)
{
    float V = norm3(state->vel);
    if (V < 1.0f) V = 1.0f;

    Vec3 vel_norm = normalize3(state->vel);
    Vec3 forward = quat_rotate(state->ori, vec3(1, 0, 0));
    Vec3 right = quat_rotate(state->ori, vec3(0, 1, 0));
    Vec3 body_up = quat_rotate(state->ori, vec3(0, 0, 1));

    float alpha = compute_aoa(state);
    float beta = compute_sideslip(state);
    float q_bar = 0.5f * params->rho * V * V;

    // Controls with runtime parameters
    float throttle = clampf((actions[0] + 1.0f) * 0.5f, 0.0f, 1.0f);

    float control_scale = 1.0f - fmaxf(0.0f, V - params->control_v_ref) * params->control_scale_slope;
    control_scale = fmaxf(control_scale, params->control_scale_min);

    float delta_e = clampf(actions[1], -1.0f, 1.0f) * MAX_ELEVATOR_DEFLECTION * control_scale;
    float delta_a = clampf(actions[2], -1.0f, 1.0f) * MAX_AILERON_DEFLECTION * control_scale;
    float delta_r = clampf(actions[3], -1.0f, 1.0f) * MAX_RUDDER_DEFLECTION * control_scale;

    // Lift and drag
    float alpha_effective = alpha + WING_INCIDENCE - ALPHA_ZERO;
    float C_L_raw = params->c_l_alpha * alpha_effective;
    float C_L = clampf(C_L_raw, -params->c_l_max, params->c_l_max);

    float C_D = params->c_d0 + params->k * C_L * C_L + K_SIDESLIP * beta * beta;

    float L_mag = C_L * q_bar * params->wing_area;
    float D_mag = C_D * q_bar * params->wing_area;

    Vec3 lift_dir = compute_lift_direction(vel_norm, right, body_up);
    Vec3 F_lift = mul3(lift_dir, L_mag);
    Vec3 F_drag = mul3(vel_norm, -D_mag);

    // Thrust (inline with params)
    float P_avail = params->engine_power * throttle;
    float T_dynamic = (P_avail * params->eta_prop) / V;
    float T_static = 0.3f * P_avail;
    float T_mag = fminf(T_static, T_dynamic);
    Vec3 F_thrust = mul3(forward, T_mag);
    Vec3 F_gravity = vec3(0, 0, -params->mass * params->gravity);

    Vec3 F_total = add3(add3(add3(F_lift, F_drag), F_thrust), F_gravity);
    deriv->v_dot = mul3(F_total, params->inv_mass);

    // Angular rates and damping
    float p = state->omega.x;
    float q = state->omega.y;
    float r = state->omega.z;

    float p_hat = p * params->wingspan / (2.0f * V);
    float q_hat = q * params->chord / (2.0f * V);
    float r_hat = r * params->wingspan / (2.0f * V);

    // Damping scaling - can boost damping at high speed and/or via multiplier
    float damping_scale = 1.0f + fmaxf(0.0f, V - params->control_v_ref) * params->damping_scale_slope;
    float total_damping = damping_scale * params->damping_multiplier;

    // Moment coefficients with scaled damping
    float Cl = params->cl_beta * beta + (params->cl_p * p_hat * total_damping) + params->cl_delta_a * delta_a + CL_DELTA_R * delta_r;
    float Cm = CM_0 + params->cm_alpha * alpha + (params->cm_q * q_hat * total_damping) + params->cm_delta_e * delta_e;
    float Cn = params->cn_beta * beta + (params->cn_r * r_hat * total_damping) + params->cn_delta_r * delta_r + CN_DELTA_A * delta_a;

    // Dimensional moments
    float L_moment = Cl * q_bar * params->wing_area * params->wingspan;
    float M_moment = -Cm * q_bar * params->wing_area * params->chord;
    float N_moment = Cn * q_bar * params->wing_area * params->wingspan;

    // Angular acceleration (Euler's equations)
    float gyro_roll = (params->iyy - params->izz) * q * r;
    float gyro_pitch = (params->izz - params->ixx) * r * p;
    float gyro_yaw = (params->ixx - params->iyy) * p * q;

    deriv->w_dot.x = (L_moment + gyro_roll) / params->ixx;
    deriv->w_dot.y = (M_moment + gyro_pitch) / params->iyy;
    deriv->w_dot.z = (N_moment + gyro_yaw) / params->izz;

    // Quaternion kinematics
    Quat omega_q = {0.0f, state->omega.x, state->omega.y, state->omega.z};
    Quat q_dot = quat_mul(state->ori, omega_q);
    deriv->q_dot.w = 0.5f * q_dot.w;
    deriv->q_dot.x = 0.5f * q_dot.x;
    deriv->q_dot.y = 0.5f * q_dot.y;
    deriv->q_dot.z = 0.5f * q_dot.z;

    deriv->vel = state->vel;
}

// RK4 step with runtime parameters
static inline void rk4_step_with_params(Plane* state, float* actions, float dt, FlightParams* params) {
    StateDerivative k1, k2, k3, k4;
    Plane temp;

    _realistic_rk4_stage = 0;
    compute_derivatives_with_params(state, actions, dt, &k1, params);

    _realistic_rk4_stage = 1;
    step_temp(state, &k1, dt * 0.5f, &temp);
    compute_derivatives_with_params(&temp, actions, dt, &k2, params);

    _realistic_rk4_stage = 2;
    step_temp(state, &k2, dt * 0.5f, &temp);
    compute_derivatives_with_params(&temp, actions, dt, &k3, params);

    _realistic_rk4_stage = 3;
    step_temp(state, &k3, dt, &temp);
    compute_derivatives_with_params(&temp, actions, dt, &k4, params);

    _realistic_rk4_stage = 0;

    float dt_6 = dt / 6.0f;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->ori.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->ori.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->ori.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->ori.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    quat_normalize(&state->ori);
}

// Step plane with runtime parameters
static inline void step_plane_with_params(Plane *p, float *actions, float dt, FlightParams* params) {
    p->prev_vel = p->vel;

    float clamped_actions[4];
    for (int i = 0; i < 4; i++) {
        clamped_actions[i] = clampf(actions[i], -1.0f, 1.0f);
    }

    rk4_step_with_params(p, clamped_actions, dt, params);

    p->throttle = (clamped_actions[0] + 1.0f) * 0.5f;

    p->omega.x = clampf(p->omega.x, -5.0f, 5.0f);
    p->omega.y = clampf(p->omega.y, -5.0f, 5.0f);
    p->omega.z = clampf(p->omega.z, -2.0f, 2.0f);

    // G-force calculation
    Vec3 dv = sub3(p->vel, p->prev_vel);
    Vec3 accel = mul3(dv, 1.0f / dt);
    Vec3 body_up = quat_rotate(p->ori, vec3(0, 0, 1));
    float accel_up = dot3(accel, body_up);
    p->g_force = accel_up * params->inv_gravity + 1.0f;

    // G-limit enforcement (same as step_plane_with_physics)
    float speed_before = norm3(p->vel);
    if (p->g_force > params->g_limit_pos) {
        float excess_g = p->g_force - params->g_limit_pos;
        float excess_accel = excess_g * params->gravity;
        Vec3 correction = mul3(body_up, excess_accel * dt);
        Vec3 vel_norm = normalize3(p->vel);
        float correction_along_vel = dot3(correction, vel_norm);
        Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));
        p->vel = sub3(p->vel, correction_perp);
        p->g_force = params->g_limit_pos;
    } else if (p->g_force < -params->g_limit_neg) {
        float deficit_g = -params->g_limit_neg - p->g_force;
        float deficit_accel = deficit_g * params->gravity;
        Vec3 correction = mul3(body_up, deficit_accel * dt);
        Vec3 vel_norm = normalize3(p->vel);
        float correction_along_vel = dot3(correction, vel_norm);
        Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));
        p->vel = add3(p->vel, correction_perp);
        p->g_force = -params->g_limit_neg;
    }

    p->yaw_from_rudder = compute_sideslip(p);
}

static inline void rk4_step(Plane* state, float* actions, float dt) {
    StateDerivative k1, k2, k3, k4;
    Plane temp;

    if (DEBUG_REALISTIC >= 5) {
        printf("\n========== RK4 STEP (dt=%.4f) ==========\n", dt);
    }

    // k1: derivative at current state
    _realistic_rk4_stage = 0;
    compute_derivatives(state, actions, dt, &k1);

    if (DEBUG_REALISTIC >= 5) {
        printf("\n  k1: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k1.v_dot.x, k1.v_dot.y, k1.v_dot.z, k1.w_dot.x, k1.w_dot.y, k1.w_dot.z);
    }

    // k2: derivative at state + k1*dt/2
    _realistic_rk4_stage = 1;
    step_temp(state, &k1, dt * 0.5f, &temp);
    compute_derivatives(&temp, actions, dt, &k2);

    if (DEBUG_REALISTIC >= 5) {
        printf("  k2: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k2.v_dot.x, k2.v_dot.y, k2.v_dot.z, k2.w_dot.x, k2.w_dot.y, k2.w_dot.z);
    }

    // k3: derivative at state + k2*dt/2
    _realistic_rk4_stage = 2;
    step_temp(state, &k2, dt * 0.5f, &temp);
    compute_derivatives(&temp, actions, dt, &k3);

    if (DEBUG_REALISTIC >= 5) {
        printf("  k3: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k3.v_dot.x, k3.v_dot.y, k3.v_dot.z, k3.w_dot.x, k3.w_dot.y, k3.w_dot.z);
    }

    // k4: derivative at state + k3*dt
    _realistic_rk4_stage = 3;
    step_temp(state, &k3, dt, &temp);
    compute_derivatives(&temp, actions, dt, &k4);

    if (DEBUG_REALISTIC >= 5) {
        printf("  k4: v_dot=(%.3f,%.3f,%.3f) w_dot=(%.6f,%.6f,%.6f)\n",
               k4.v_dot.x, k4.v_dot.y, k4.v_dot.z, k4.w_dot.x, k4.w_dot.y, k4.w_dot.z);
    }

    _realistic_rk4_stage = 0;  // Reset for next step

    float dt_6 = dt / 6.0f;

    Vec3 old_vel = state->vel;
    Vec3 old_omega = state->omega;
    Quat old_ori = state->ori;

    state->pos.x += (k1.vel.x + 2.0f * k2.vel.x + 2.0f * k3.vel.x + k4.vel.x) * dt_6;
    state->pos.y += (k1.vel.y + 2.0f * k2.vel.y + 2.0f * k3.vel.y + k4.vel.y) * dt_6;
    state->pos.z += (k1.vel.z + 2.0f * k2.vel.z + 2.0f * k3.vel.z + k4.vel.z) * dt_6;

    state->vel.x += (k1.v_dot.x + 2.0f * k2.v_dot.x + 2.0f * k3.v_dot.x + k4.v_dot.x) * dt_6;
    state->vel.y += (k1.v_dot.y + 2.0f * k2.v_dot.y + 2.0f * k3.v_dot.y + k4.v_dot.y) * dt_6;
    state->vel.z += (k1.v_dot.z + 2.0f * k2.v_dot.z + 2.0f * k3.v_dot.z + k4.v_dot.z) * dt_6;

    state->ori.w += (k1.q_dot.w + 2.0f * k2.q_dot.w + 2.0f * k3.q_dot.w + k4.q_dot.w) * dt_6;
    state->ori.x += (k1.q_dot.x + 2.0f * k2.q_dot.x + 2.0f * k3.q_dot.x + k4.q_dot.x) * dt_6;
    state->ori.y += (k1.q_dot.y + 2.0f * k2.q_dot.y + 2.0f * k3.q_dot.y + k4.q_dot.y) * dt_6;
    state->ori.z += (k1.q_dot.z + 2.0f * k2.q_dot.z + 2.0f * k3.q_dot.z + k4.q_dot.z) * dt_6;

    state->omega.x += (k1.w_dot.x + 2.0f * k2.w_dot.x + 2.0f * k3.w_dot.x + k4.w_dot.x) * dt_6;
    state->omega.y += (k1.w_dot.y + 2.0f * k2.w_dot.y + 2.0f * k3.w_dot.y + k4.w_dot.y) * dt_6;
    state->omega.z += (k1.w_dot.z + 2.0f * k2.w_dot.z + 2.0f * k3.w_dot.z + k4.w_dot.z) * dt_6;

    quat_normalize(&state->ori);

    if (DEBUG_REALISTIC >= 5) {
        printf("\n  --- RK4 WEIGHTED AVERAGE ---\n");
        printf("  vel: (%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f) delta=(%.3f,%.3f,%.3f)\n",
               old_vel.x, old_vel.y, old_vel.z,
               state->vel.x, state->vel.y, state->vel.z,
               state->vel.x - old_vel.x, state->vel.y - old_vel.y, state->vel.z - old_vel.z);
        printf("  omega: (%.4f,%.4f,%.4f) -> (%.4f,%.4f,%.4f) delta=(%.6f,%.6f,%.6f)\n",
               old_omega.x, old_omega.y, old_omega.z,
               state->omega.x, state->omega.y, state->omega.z,
               state->omega.x - old_omega.x, state->omega.y - old_omega.y, state->omega.z - old_omega.z);
        printf("  ori: (%.4f,%.4f,%.4f,%.4f) -> (%.4f,%.4f,%.4f,%.4f)\n",
               old_ori.w, old_ori.x, old_ori.y, old_ori.z,
               state->ori.w, state->ori.x, state->ori.y, state->ori.z);
    }
}


static inline void step_plane_with_physics(Plane *p, float *actions, float dt) {
    _realistic_step_count++;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║ REALISTIC PHYSICS STEP %d (dt=%.4f)                                           \n", _realistic_step_count, dt);
        printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    }

    p->prev_vel = p->vel;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== BEFORE RK4 ===\n");
        printf("pos=(%.1f, %.1f, %.1f) alt=%.1f m\n", p->pos.x, p->pos.y, p->pos.z, p->pos.z);
        printf("vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n", p->vel.x, p->vel.y, p->vel.z, norm3(p->vel));
        printf("ori=(w=%.4f, x=%.4f, y=%.4f, z=%.4f)\n", p->ori.w, p->ori.x, p->ori.y, p->ori.z);
        printf("omega=(%.4f, %.4f, %.4f) rad/s\n", p->omega.x, p->omega.y, p->omega.z);

        // Compute pitch angle
        Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
        float pitch = asinf(-forward.z) * RAD_TO_DEG;
        Vec3 vel_norm = normalize3(p->vel);
        float vel_pitch = asinf(vel_norm.z) * RAD_TO_DEG;
        float alpha = compute_aoa(p) * RAD_TO_DEG;

        printf("pitch=%.2f deg (nose %s), vel_pitch=%.2f deg (%s), alpha=%.2f deg\n",
               pitch, pitch > 0 ? "UP" : "DOWN",
               vel_pitch, vel_pitch > 0 ? "CLIMBING" : "DESCENDING",
               alpha);
        printf("actions=[thr=%.2f, elev=%.2f, ail=%.2f, rud=%.2f]\n",
               actions[0], actions[1], actions[2], actions[3]);
    }

    float clamped_actions[4];
    for (int i = 0; i < 4; i++) {
        clamped_actions[i] = clampf(actions[i], -1.0f, 1.0f);
    }

    rk4_step(p, clamped_actions, dt);

    p->throttle = (clamped_actions[0] + 1.0f) * 0.5f;

    float old_omega_y = p->omega.y;
    p->omega.x = clampf(p->omega.x, -5.0f, 5.0f);  // ~286 deg/s max roll
    p->omega.y = clampf(p->omega.y, -5.0f, 5.0f);  // ~286 deg/s max pitch
    p->omega.z = clampf(p->omega.z, -2.0f, 2.0f);  // ~115 deg/s max yaw (less authority)

    if (DEBUG_REALISTIC >= 1 && old_omega_y != p->omega.y) {
        printf("  WARNING: omega.y clamped from %.4f to %.4f\n", old_omega_y, p->omega.y);
    }

    Vec3 dv = sub3(p->vel, p->prev_vel);
    Vec3 accel = mul3(dv, 1.0f / dt);
    Vec3 body_up = quat_rotate(p->ori, vec3(0, 0, 1));

    // Total acceleration in body-up direction, converted to G
    // Add 1G because we're measuring from inertial frame (gravity already in accel)
    float accel_up = dot3(accel, body_up);
    p->g_force = accel_up * INV_GRAVITY + 1.0f;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== G-FORCE CALCULATION ===\n");
        printf("dv=(%.3f, %.3f, %.3f) over dt=%.4f\n", dv.x, dv.y, dv.z, dt);
        printf("accel=(%.3f, %.3f, %.3f) m/s^2\n", accel.x, accel.y, accel.z);
        printf("body_up=(%.4f, %.4f, %.4f)\n", body_up.x, body_up.y, body_up.z);
        printf("accel·body_up=%.3f m/s^2 / g=%.3f + 1.0 = %.3f G\n",
               accel_up, accel_up * INV_GRAVITY, p->g_force);
    }

    float speed_before_glimit = norm3(p->vel);

    if (p->g_force > G_LIMIT_POS) {
        // Positive G exceeded - reduce upward acceleration
        float excess_g = p->g_force - G_LIMIT_POS;
        float excess_accel = excess_g * GRAVITY;

        if (DEBUG_REALISTIC >= 1) {
            printf("G-LIMIT: +%.2f G exceeded limit +%.1f by %.2f G, reducing vel\n",
                   p->g_force, G_LIMIT_POS, excess_g);
        }

        Vec3 correction = mul3(body_up, excess_accel * dt);

        // Project out the component along velocity to preserve speed (energy)
        Vec3 vel_norm = normalize3(p->vel);
        float correction_along_vel = dot3(correction, vel_norm);
        Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));

        p->vel = sub3(p->vel, correction_perp);
        p->g_force = G_LIMIT_POS;

    } else if (p->g_force < -G_LIMIT_NEG) {
        // Negative G exceeded - reduce downward acceleration
        float deficit_g = -G_LIMIT_NEG - p->g_force;
        float deficit_accel = deficit_g * GRAVITY;

        if (DEBUG_REALISTIC >= 1) {
            printf("G-LIMIT: %.2f G exceeded limit -%.1f by %.2f G, reducing vel\n",
                   p->g_force, G_LIMIT_NEG, -deficit_g);
        }

        Vec3 correction = mul3(body_up, deficit_accel * dt);

        // Project out the component along velocity to preserve speed (energy)
        Vec3 vel_norm = normalize3(p->vel);
        float correction_along_vel = dot3(correction, vel_norm);
        Vec3 correction_perp = sub3(correction, mul3(vel_norm, correction_along_vel));

        p->vel = add3(p->vel, correction_perp);
        p->g_force = -G_LIMIT_NEG;
    }

    // Verify energy was preserved (speed should not have changed)
    if (DEBUG_REALISTIC >= 1) {
        float speed_after_glimit = norm3(p->vel);
        if (fabsf(speed_after_glimit - speed_before_glimit) > 0.01f) {
            printf("WARNING: G-limit changed speed from %.2f to %.2f!\n",
                   speed_before_glimit, speed_after_glimit);
        }
    }

    p->yaw_from_rudder = compute_sideslip(p);

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== AFTER RK4 ===\n");
        printf("pos=(%.1f, %.1f, %.1f) alt=%.1f m (Δalt=%.2f m)\n",
               p->pos.x, p->pos.y, p->pos.z, p->pos.z, p->pos.z - (p->pos.z - p->vel.z * dt));
        printf("vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n", p->vel.x, p->vel.y, p->vel.z, norm3(p->vel));
        printf("ori=(w=%.4f, x=%.4f, y=%.4f, z=%.4f)\n", p->ori.w, p->ori.x, p->ori.y, p->ori.z);
        printf("omega=(%.4f, %.4f, %.4f) rad/s = (%.2f, %.2f, %.2f) deg/s\n",
               p->omega.x, p->omega.y, p->omega.z,
               p->omega.x * RAD_TO_DEG, p->omega.y * RAD_TO_DEG, p->omega.z * RAD_TO_DEG);
        printf("g_force=%.2f G (limits: +%.1f/-%.1f)\n", p->g_force, G_LIMIT_POS, G_LIMIT_NEG);

        // Compute final pitch and alpha
        Vec3 forward = quat_rotate(p->ori, vec3(1, 0, 0));
        float pitch = asinf(-forward.z) * RAD_TO_DEG;
        float alpha = compute_aoa(p) * RAD_TO_DEG;
        Vec3 vel_norm = normalize3(p->vel);
        float vel_pitch = asinf(vel_norm.z) * RAD_TO_DEG;

        printf("final: pitch=%.2f deg, vel_pitch=%.2f deg, alpha=%.2f deg\n",
               pitch, vel_pitch, alpha);

        // Key insight: what's happening to orientation vs velocity?
        printf("\n=== STEP SUMMARY ===\n");
        printf("vel.z changed: %.3f -> %.3f (Δ=%.3f m/s, %s)\n",
               p->prev_vel.z, p->vel.z, p->vel.z - p->prev_vel.z,
               p->vel.z > p->prev_vel.z ? "CLIMBING MORE" : "DIVING MORE");
        printf("omega.y = %.4f rad/s = %.2f deg/s (nose pitching %s)\n",
               p->omega.y, p->omega.y * RAD_TO_DEG,
               p->omega.y > 0 ? "DOWN" : "UP");
    }

    if (DEBUG >= 10) {
        float V = norm3(p->vel);
        float alpha = compute_aoa(p) * RAD_TO_DEG;
        float beta = compute_sideslip(p) * RAD_TO_DEG;
        printf("=== REALISTIC PHYSICS ===\n");
        printf("speed=%.1f m/s\n", V);
        printf("throttle=%.2f\n", p->throttle);
        printf("alpha=%.2f deg, beta=%.2f deg\n", alpha, beta);
        printf("omega=(%.3f, %.3f, %.3f) rad/s\n", p->omega.x, p->omega.y, p->omega.z);
        printf("g_force=%.2f g (limit=+%.1f/-%.1f)\n", p->g_force, G_LIMIT_POS, G_LIMIT_NEG);
    }
}

// Calculate specific energy: Es = altitude + speed²/(2*g)
static inline float calc_specific_energy(Plane *p) {
    float speed = norm3(p->vel);
    return p->pos.z + (speed * speed) / (2.0f * GRAVITY);
}

static inline float calc_specific_energy_with_params(Plane *p, FlightParams *params) {
    float speed = norm3(p->vel);
    return p->pos.z + (speed * speed) / (2.0f * params->gravity);
}

static inline void reset_plane(Plane *p, Vec3 pos, Vec3 vel) {
    p->pos = pos;
    p->vel = vel;
    p->prev_vel = vel;
    p->omega = vec3(0, 0, 0);
    p->ori = quat(1, 0, 0, 0);
    p->throttle = 0.5f;
    p->g_force = 1.0f;
    p->yaw_from_rudder = 0.0f;
    p->fire_cooldown = 0;
    // Initialize specific energy for energy management reward
    float speed = norm3(vel);
    p->prev_energy = pos.z + (speed * speed) / (2.0f * GRAVITY);

    _realistic_step_count = 0;

    if (DEBUG_REALISTIC >= 1) {
        printf("\n=== RESET_PLANE ===\n");
        printf("pos=(%.1f, %.1f, %.1f)\n", pos.x, pos.y, pos.z);
        printf("vel=(%.2f, %.2f, %.2f) |V|=%.2f m/s\n", vel.x, vel.y, vel.z, norm3(vel));
        printf("ori=(1, 0, 0, 0) (identity)\n");
        printf("omega=(0, 0, 0)\n");
    }
}

#endif // FLIGHTLIB_H
