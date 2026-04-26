/*
 * test_common.h - Shared helpers for dogfight C tests.
 *
 * Layout (matches ocean/dogfight/test_flight_dynamics.c style):
 *   - TestEnv: a Dogfight env with statically-sized buffers wired in.
 *   - setup_env(): init + c_reset, ready to step.
 *   - run_steps(): copy actions, step N times.
 *   - plane_fwd / plane_up: forward and up world vectors.
 *   - ap_*(): C clones of pufferlib/ocean/dogfight/autopilot.py PD helpers.
 *   - py_level_flight_pitch_velocity(): faithful clone of
 *     test_flight_base.py::level_flight_pitch_velocity(), used by the
 *     speed/G tests so their ports match Python step-for-step.
 *
 * Each test_<name>.c writes plain test functions that return 0 on pass
 * and 1 on fail (matching the existing test_flight_dynamics.c pattern).
 * main() sums failures and exits with that count.
 *
 * Compile (linked to raylib because dogfight.h pulls in dogfight_render.h):
 *   gcc -O2 \
 *     -I ocean/dogfight \
 *     -I raylib-5.5_linux_amd64/include \
 *     ocean/dogfight/tests/test_<name>.c \
 *     raylib-5.5_linux_amd64/lib/libraylib.a \
 *     -lm -lpthread -ldl \
 *     -o ocean/dogfight/tests/test_<name>
 */
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
    float terminals[1];
} TestEnv;

static RewardConfig test_default_rcfg(void) {
    RewardConfig r = {0};
    r.speed_min = 50.0f;
    r.low_altitude_threshold = 1500.0f;
    return r;
}

/* Wire buffers, init the env, reset. obs_scheme 0 by default. */
static void setup_env(TestEnv* t, int obs_scheme) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    /* Match dogfight.py default. Python tests run loops up to ~1500 steps
     * and rely on the env not auto-resetting on tick timeout. */
    t->env.max_steps = 3000;
    t->env.rng = 42;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_default_rcfg();
    init(&t->env, obs_scheme, &rcfg, 0, 0, 0);
    c_reset(&t->env);
}

/* Step n times. If `action` is non-NULL, copy it into env.actions before
 * each step; otherwise use whatever the caller already wrote. */
__attribute__((unused))
static void run_steps(TestEnv* t, int n_steps, const float* action) {
    for (int i = 0; i < n_steps; i++) {
        if (action) memcpy(t->env.actions, action, TEST_NUM_ATNS * sizeof(float));
        c_step(&t->env);
    }
}

/* World-frame forward and up vectors. */
static Vec3 plane_fwd(const Plane* p) {
    return quat_rotate(p->ori, vec3(1.0f, 0.0f, 0.0f));
}
static Vec3 plane_up(const Plane* p) {
    return quat_rotate(p->ori, vec3(0.0f, 0.0f, 1.0f));
}

/* Pitch angle in degrees (positive = nose up). */
static float plane_pitch_deg(const Plane* p) {
    Vec3 f = plane_fwd(p);
    float fz = f.z;
    if (fz >  1.0f) fz =  1.0f;
    if (fz < -1.0f) fz = -1.0f;
    return asinf(fz) * 57.29577951308232f;
}

/* Bank angle in degrees (positive = right bank). */
static float plane_bank_deg(const Plane* p) {
    Vec3 u = plane_up(p);
    float uz = u.z;
    if (uz >  1.0f) uz =  1.0f;
    if (uz < -1.0f) uz = -1.0f;
    float bank = acosf(uz);
    return (u.y < 0.0f ? bank : -bank) * 57.29577951308232f;
}

static float clip_unit(float x) {
    if (x >  1.0f) return  1.0f;
    if (x < -1.0f) return -1.0f;
    return x;
}

/* PD autopilot helpers — C ports of autopilot.py. Gains from pid_tune.py.
 * Marked unused-attr because some test files don't call every helper. */
#define AP_PITCH_KP 0.2f
#define AP_PITCH_KD 0.1f
#define AP_ROLL_KP  1.0f
#define AP_ROLL_KD  0.1f
#define AP_YAW_KP   0.1f
#define AP_YAW_KD   0.02f
#define AP_RAD_TO_DEG 57.29577951308232f

__attribute__((unused))
static float ap_hold_pitch(const Plane* p, float target_deg) {
    float pitch = plane_pitch_deg(p);
    float omega_pitch_deg = p->omega.y * AP_RAD_TO_DEG;
    float err = target_deg - pitch;
    return clip_unit(-AP_PITCH_KP * err - AP_PITCH_KD * omega_pitch_deg);
}

__attribute__((unused))
static float ap_hold_vz(const Plane* p, float target_vz) {
    float omega_pitch_deg = p->omega.y * AP_RAD_TO_DEG;
    float err = target_vz - p->vel.z;
    return clip_unit(-AP_PITCH_KP * 0.6f * err - AP_PITCH_KD * omega_pitch_deg);
}

__attribute__((unused))
static float ap_hold_bank(const Plane* p, float target_bank_deg) {
    float bank = plane_bank_deg(p);
    float omega_roll_deg = p->omega.x * AP_RAD_TO_DEG;
    float err = target_bank_deg - bank;
    return clip_unit(AP_ROLL_KP * err - AP_ROLL_KD * omega_roll_deg);
}

__attribute__((unused))
static float ap_damp_yaw(const Plane* p) {
    float omega_yaw_deg = p->omega.z * AP_RAD_TO_DEG;
    return clip_unit(-AP_YAW_KP * omega_yaw_deg - AP_YAW_KD * omega_yaw_deg);
}

/* Faithful clone of test_flight_base.py::level_flight_pitch_velocity().
 *
 * Python wraps a small PD on vz in a "velocity command" with a 2/coeff
 * gain, but ctrl_elevator was never wired into env_get_state(), so the
 * `current_elevator` term in the Python is always 0. The whole thing
 * reduces to:
 *     target = clip(-(kp+kd)*vz, -0.2, 0.2)
 *     action = clip((2/coeff) * target, -1, 1)
 * With kp=kd=0.001, coeff=0.25 → action = clip(-0.016*vz, -1, 1).
 */
__attribute__((unused))
static float py_level_flight_pitch_velocity(const Plane* p) {
    const float kp = 0.001f;
    const float kd = 0.001f;
    const float coeff = 0.25f;
    float vz = p->vel.z;
    float target = -kp * vz - kd * vz;
    if (target >  0.2f) target =  0.2f;
    if (target < -0.2f) target = -0.2f;
    return clip_unit(2.0f * target / coeff);
}

/* Velocity-command wrapper. Mirrors autopilot.py:
 *   velocity_cmd = clip(2 * (target - current) / coeff, -1, 1)
 * In 3.0/4.0 the C plane has no ctrl_* state and Python's get_state never
 * exposed it, so `current` is always 0 — collapses to clip(8 * target). */
__attribute__((unused))
static float ap_to_velocity(float target_position) {
    return clip_unit(target_position * 8.0f);
}

/* Coordinated-turn elevator+aileron, mirrors autopilot.py::hold_bank_and_level. */
__attribute__((unused))
static void ap_hold_bank_and_level(const Plane* p, float target_bank_deg,
                                    float* out_elev, float* out_ail) {
    float ail = ap_hold_bank(p, target_bank_deg);
    float bank_rad = fabsf(target_bank_deg) * (3.14159265f / 180.0f);
    float extra_pitch_bias;
    if (bank_rad < 80.0f * (3.14159265f / 180.0f)) {
        extra_pitch_bias = -0.05f * (1.0f / cosf(bank_rad) - 1.0f) * 10.0f;
    } else {
        extra_pitch_bias = -0.3f;
    }
    float elev = ap_hold_vz(p, 0.0f) + extra_pitch_bias;
    *out_elev = clip_unit(elev);
    *out_ail = ail;
}

/* Variance / std-dev / zero-crossings — used by oscillation tests. */
__attribute__((unused))
static float arr_mean(const float* x, int n) {
    if (n <= 0) return 0.0f;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return (float)(s / n);
}
__attribute__((unused))
static float arr_var(const float* x, int n) {
    if (n <= 0) return 0.0f;
    float m = arr_mean(x, n);
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - m; s += d * d; }
    return (float)(s / n);
}
__attribute__((unused))
static float arr_std(const float* x, int n) {
    return sqrtf(arr_var(x, n));
}
__attribute__((unused))
static float arr_min(const float* x, int n) {
    if (n <= 0) return 0.0f;
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] < m) m = x[i];
    return m;
}
__attribute__((unused))
static float arr_max(const float* x, int n) {
    if (n <= 0) return 0.0f;
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    return m;
}
__attribute__((unused))
static int arr_zero_crossings(const float* x, int n) {
    if (n <= 1) return 0;
    int last_sign = (x[0] >= 0.0f) ? 1 : -1;
    int crossings = 0;
    for (int i = 1; i < n; i++) {
        int s = (x[i] >= 0.0f) ? 1 : -1;
        if (s != last_sign) crossings++;
        last_sign = s;
    }
    return crossings;
}

/* Heading unwrap: turn raw atan2 series into monotonically continuous angles.
 * Returns heading[n-1] - heading[0] in radians, unwrapped. */
__attribute__((unused))
static float arr_unwrap_delta(const float* h, int n) {
    if (n < 2) return 0.0f;
    float total = 0.0f;
    float prev = h[0];
    for (int i = 1; i < n; i++) {
        float d = h[i] - prev;
        while (d >  3.14159265f) d -= 2.0f * 3.14159265f;
        while (d < -3.14159265f) d += 2.0f * 3.14159265f;
        total += d;
        prev = h[i];
    }
    return total;
}

#endif /* TEST_COMMON_H */
