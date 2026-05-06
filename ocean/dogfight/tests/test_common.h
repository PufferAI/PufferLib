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

#ifndef DEG
#define DEG (3.14159265358979f / 180.0f)
#endif
#ifndef RAD
#define RAD (180.0f / 3.14159265358979f)
#endif

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
    float terminals[1];
} TestEnv;

/* Visual-render hooks. test_flight_physics.c sets these from --render/--fps;
 * other test_*.c files leave them at defaults (rendering off). */
__attribute__((unused)) static int g_visual_render = 0;
__attribute__((unused)) static int g_visual_fps = 50;
__attribute__((unused)) static const char* g_visual_only_test = NULL;
__attribute__((unused)) static FILE* g_log_csv = NULL;

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

/* Step the env once. If g_visual_render is set, also call c_render and
 * (lazily) SetTargetFPS. WindowShouldClose -> exit, so ESC bails the test. */
__attribute__((unused))
static void t_step(TestEnv* t) {
    c_step(&t->env);
    if (g_visual_render) {
        c_render(&t->env);
        static int fps_set = 0;
        if (!fps_set) { SetTargetFPS(g_visual_fps); fps_set = 1; }
        if (WindowShouldClose()) { CloseWindow(); exit(0); }
    }
}

/* Step n times. If `action` is non-NULL, copy it into env.actions before
 * each step; otherwise use whatever the caller already wrote. */
__attribute__((unused))
static void run_steps(TestEnv* t, int n_steps, const float* action) {
    for (int i = 0; i < n_steps; i++) {
        if (action) memcpy(t->env.actions, action, TEST_NUM_ATNS * sizeof(float));
        t_step(t);
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

/* Bank angle in degrees (positive = right bank).
 * Uses body-Y vector projected onto world Z for the sign — heading-independent.
 * Per this codebase's convention (aileron +1 → roll right, but right-hand rule
 * around body +X raises body +Y), body +Y points along the LEFT wing direction.
 * So body Y above horizon (r.z > 0) means left wing up = RIGHT bank. */
static float plane_bank_deg(const Plane* p) {
    Vec3 u = plane_up(p);
    Vec3 r = quat_rotate(p->ori, vec3(0.0f, 1.0f, 0.0f));
    float uz = u.z;
    if (uz >  1.0f) uz =  1.0f;
    if (uz < -1.0f) uz = -1.0f;
    float bank = acosf(uz);
    return (r.z > 0.0f ? bank : -bank) * 57.29577951308232f;
}

static float clip_unit(float x) {
    if (x >  1.0f) return  1.0f;
    if (x < -1.0f) return -1.0f;
    return x;
}

/* Build an orientation quat with given bank (positive = right wing down,
 * rotation around +X) and pitch (positive = nose above horizon, rotation
 * around -Y), both in degrees. Identity for (0, 0). Composition: bank applied
 * to pitched-forward axis (qbank * qpitch) — same convention as the existing
 * sustained_turn / turn_60 setup, but with the SIGN bug fixed.
 *
 * Verified by primitive control axis tests:
 *   - aileron actions[2] = +1 → roll_right → +X rotation → wing dips right
 *   - elevator actions[1] = -1 → pitch_up   → -Y rotation → nose rises
 */
__attribute__((unused))
static Quat attitude_quat(float bank_deg, float pitch_deg) {
    Quat qbank  = quat_from_axis_angle(vec3(1.0f, 0.0f, 0.0f),  bank_deg  * DEG);
    Quat qpitch = quat_from_axis_angle(vec3(0.0f, 1.0f, 0.0f), -pitch_deg * DEG);
    return quat_mul(qbank, qpitch);
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

/* P-controller for forward speed.
 * Returns actions[0] in [-1,1]. Engine wires throttle = (a+1)/2.
 * 50% throttle holds ~120 m/s level cruise, so target ~120 is the
 * natural neutral point. */
__attribute__((unused))
static float ap_hold_speed(const Plane* p, float target_speed) {
    float speed = sqrtf(p->vel.x * p->vel.x +
                        p->vel.y * p->vel.y +
                        p->vel.z * p->vel.z);
    float err = target_speed - speed;
    /* Aggressive gain so throttle stays at extreme until close to target;
     * at err = 0 returns 0 (50% throttle, ~120 m/s natural cruise). */
    return clip_unit(err * 0.5f);
}

/* P-controller for altitude. Returns a TARGET vz (m/s) to feed
 * into ap_hold_vz. Cap at +-15 m/s to stay within climb-rate budget. */
__attribute__((unused))
static float ap_hold_altitude_vz_target(const Plane* p, float target_alt_m) {
    float err = target_alt_m - p->pos.z;
    float vz = err * 0.1f;
    if (vz >  15.0f) vz =  15.0f;
    if (vz < -15.0f) vz = -15.0f;
    return vz;
}

/* P-controller for heading. Returns a TARGET bank angle (deg) to feed
 * into ap_hold_bank_and_level. heading is atan2(vy, vx) in deg.
 * Capped at +-45 deg bank for coordinated turning.
 *
 * Sign: positive bank (right wing down) → right turn → heading DECREASES
 * (in atan2 convention). So to drive heading TOWARD target, use negative
 * gain: bank = -err. */
__attribute__((unused))
static float ap_hold_heading_bank_target(const Plane* p, float target_heading_deg) {
    float heading = atan2f(p->vel.y, p->vel.x) * 57.29577951308232f;
    float err = target_heading_deg - heading;
    while (err >  180.0f) err -= 360.0f;
    while (err < -180.0f) err += 360.0f;
    float bank = -err * 1.0f;
    if (bank >  45.0f) bank =  45.0f;
    if (bank < -45.0f) bank = -45.0f;
    return bank;
}

/* P-controller for body roll rate (omega.x). Returns aileron action [-1,1]. */
__attribute__((unused))
static float ap_hold_roll_rate(const Plane* p, float target_omega_x_deg_s) {
    float current = p->omega.x * 57.29577951308232f;
    float err = target_omega_x_deg_s - current;
    return clip_unit(err * 0.05f);
}

/* P-controller for body pitch rate (omega.y). Returns elevator action [-1,1].
 * Note: in this env, body pitch-rate convention is omega.y > 0 → nose DOWN.
 * Elevator action positive = nose down, so sign matches: positive err → positive elev. */
__attribute__((unused))
static float ap_hold_pitch_rate(const Plane* p, float target_omega_y_deg_s) {
    float current = p->omega.y * 57.29577951308232f;
    float err = target_omega_y_deg_s - current;
    return clip_unit(err * 0.05f);
}

/* CSV telemetry writer for recovery tests. Writes a header once (lazy)
 * then a row per tick. test_name is the short label used in the CSV
 * `test` column. a is the 5-element action vector. */
__attribute__((unused))
static void recovery_log_row(const char* test_name, int step,
                             const Plane* p, const float* a) {
    if (!g_log_csv) return;
    static int header_written = 0;
    if (!header_written) {
        fprintf(g_log_csv,
            "test,tick,t_sec,bank_deg,pitch_deg,heading_deg,"
            "vx,vy,vz,speed,altitude,"
            "omega_x,omega_y,omega_z,"
            "act_throttle,act_elev,act_ail,act_rud,act_trigger\n");
        header_written = 1;
    }
    float bank = plane_bank_deg(p);
    float pitch = plane_pitch_deg(p);
    float heading = atan2f(p->vel.y, p->vel.x) * 57.29577951308232f;
    float speed = sqrtf(p->vel.x*p->vel.x + p->vel.y*p->vel.y + p->vel.z*p->vel.z);
    fprintf(g_log_csv,
        "%s,%d,%.3f,%+.3f,%+.3f,%+.3f,"
        "%+.3f,%+.3f,%+.3f,%.3f,%.2f,"
        "%+.5f,%+.5f,%+.5f,"
        "%.3f,%.3f,%.3f,%.3f,%.3f\n",
        test_name, step, step * 0.02f,
        bank, pitch, heading,
        p->vel.x, p->vel.y, p->vel.z, speed, p->pos.z,
        p->omega.x, p->omega.y, p->omega.z,
        a[0], a[1], a[2], a[3], a[4]);
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
