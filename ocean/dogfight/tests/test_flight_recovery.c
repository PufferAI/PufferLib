/*
 * test_flight_recovery.c - 1:1 port of test_flight_recovery.py.
 *
 * 18 dive-recovery scenarios, each calling run_recovery_test() with
 * a different (pitch, speed, altitude, bank) initial condition.
 *
 * All tests are soft — they print metrics + pass status but do not
 * abort. main() returns 0.
 */
#include "test_common.h"

#define G_REC 9.81f
#define DEG (3.14159265f / 180.0f)
#define RAD (180.0f / 3.14159265f)

#define RECOVERY_V_REF   100.0f
#define RECOVERY_KP_VZ   0.005f
#define RECOVERY_KD_VZ   0.05f
#define RECOVERY_KP_ROLL 1.0f

typedef struct {
    int success;
    int crashed;
    int timeout;
    float altitude_lost;
    float recovery_time;
    float min_altitude;
    float max_g;
    float max_speed;
    float pitch_rate_std;
    int zero_crossings;
} RecoveryMetrics;

/* Mirror autopilot.py::get_pitch_deg / get_bank_deg. */
static float st_get_pitch_deg(const Plane* p) {
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    return asinf(clip_unit(fwd.z)) * RAD;
}
static float st_get_bank_deg(const Plane* p) {
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float bank = acosf(clip_unit(up.z));
    return ((up.y < 0.0f) ? bank : -bank) * RAD;
}

/* test_flight_recovery.py::euler_to_quaternion — note pitch is negated. */
static void euler_to_quat(float roll_deg, float pitch_deg, float yaw_deg,
                          float* w, float* x, float* y, float* z) {
    float roll  = roll_deg  * DEG / 2.0f;
    float pitch = -pitch_deg * DEG / 2.0f;
    float yaw   = yaw_deg   * DEG / 2.0f;
    float cr = cosf(roll),  sr = sinf(roll);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cy = cosf(yaw),   sy = sinf(yaw);
    *w = cr*cp*cy + sr*sp*sy;
    *x = sr*cp*cy - cr*sp*sy;
    *y = cr*sp*cy + sr*cp*sy;
    *z = cr*cp*sy - sr*sp*cy;
}

/* Rotate (1,0,0) by quaternion to get world-frame nose direction. */
static void quat_rotate_fwd(float w, float x, float y, float z,
                             float* fx, float* fy, float* fz) {
    /* qv x v with v=(1,0,0): (0, z, -y) */
    float cx = 0.0f, cy = z, cz = -y;
    /* qv x (qv x v) */
    float cx2 = y * cz - z * cy;
    float cy2 = z * cx - x * cz;
    float cz2 = x * cy - y * cx;
    *fx = 1.0f + 2.0f * w * cx + 2.0f * cx2;
    *fy = 0.0f + 2.0f * w * cy + 2.0f * cy2;
    *fz = 0.0f + 2.0f * w * cz + 2.0f * cz2;
}

/* PID-based ideal recovery action.  Mirrors ideal_recovery_action(). */
static void recovery_action(const Plane* p, float g_limit,
                             float* out_elev, float* out_ail) {
    float bank_deg = st_get_bank_deg(p);
    float vz = p->vel.z;
    float pitch_rate = p->omega.y;
    float current_g = p->g_force;
    float speed = norm3(p->vel);
    if (speed < 50.0f) speed = 50.0f;
    float gain_scale = (RECOVERY_V_REF / speed) * (RECOVERY_V_REF / speed);
    if (gain_scale < 0.25f) gain_scale = 0.25f;
    if (gain_scale > 2.0f)  gain_scale = 2.0f;
    float kp_vz = RECOVERY_KP_VZ * gain_scale;
    float kd_vz = RECOVERY_KD_VZ * gain_scale;
    float kp_roll = RECOVERY_KP_ROLL * gain_scale;
    float bank_rad = bank_deg * DEG;
    float aileron = -kp_roll * bank_rad;
    float vz_error = -vz;
    float elevator = -kp_vz * vz_error + kd_vz * pitch_rate;
    if (current_g > g_limit - 0.5f) {
        if (elevator < -0.3f) elevator = -0.3f;
    }
    *out_elev = clip_unit(elevator);
    *out_ail = clip_unit(aileron);
}

/* Run one dive-recovery from a fixed initial condition.  Mirrors
 * test_flight_recovery.py::run_recovery_test(). */
static RecoveryMetrics run_recovery(float pitch_deg, float speed,
                                     float altitude, float bank_deg,
                                     float max_time, float g_limit) {
    TestEnv te; setup_env(&te, 0);
    float ow, ox, oy, oz;
    euler_to_quat(bank_deg, pitch_deg, 0.0f, &ow, &ox, &oy, &oz);

    float fx, fy, fz;
    quat_rotate_fwd(ow, ox, oy, oz, &fx, &fy, &fz);
    float vx = speed * fx;
    float vy = speed * fy;
    float vz = speed * fz;

    force_state(&te.env,
        0.0f, 0.0f, altitude,
        vx, vy, vz,
        ow, ox, oy, oz, 1.0f,
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f, -9999.0f,
        0, -1, -1);

    float min_alt = altitude;
    float max_g = 1.0f, max_speed = speed;
    float recovery_time = 0.0f;
    int recovered = 0, crashed = 0;

    int max_steps = (int)(max_time / 0.02f);
    if (max_steps > 1000) max_steps = 1000;
    /* Pre-allocate large enough — caller passes max_time<=20s → 1000 steps. */
    static float pitch_rates[1024];
    int n_pr = 0;
    int prev_sign = 0, zc = 0;
    int recovered_step = 0;

    for (int step = 0; step < max_steps; step++) {
        Plane* p = &te.env.player;
        float current_alt = p->pos.z;
        float current_speed = norm3(p->vel);
        float current_pitch = st_get_pitch_deg(p);
        float current_bank = st_get_bank_deg(p);
        float current_g = p->g_force;
        float current_vz = p->vel.z;

        if (current_alt < min_alt) min_alt = current_alt;
        if (fabsf(current_g) > max_g) max_g = fabsf(current_g);
        if (current_speed > max_speed) max_speed = current_speed;

        float pr = p->omega.y;
        if (n_pr < 1024) pitch_rates[n_pr++] = pr;
        if (pr != 0.0f) {
            int s = (pr > 0.0f) ? 1 : -1;
            if (prev_sign != 0 && s != prev_sign) zc++;
            prev_sign = s;
        }

        if (current_alt <= 0.0f) {
            crashed = 1;
            break;
        }

        if (current_pitch > -5.0f && fabsf(current_vz) < 5.0f && fabsf(current_bank) < 15.0f) {
            if (!recovered) {
                recovery_time = (float)step * 0.02f;
                recovered = 1;
                recovered_step = step;
            } else if (step > recovered_step + 50) {
                break;
            }
        }

        float elev, ail;
        recovery_action(p, g_limit, &elev, &ail);
        float a[5] = {1.0f, elev, ail, 0.0f, 0.0f};
        memcpy(te.env.actions, a, sizeof(a));
        c_step(&te.env);
        if (te.env.terminals[0] && !crashed) break;
    }

    RecoveryMetrics m;
    m.crashed = crashed;
    m.success = recovered && !crashed;
    m.timeout = !recovered && !crashed;
    m.altitude_lost = altitude - min_alt;
    m.recovery_time = recovered ? recovery_time : max_time;
    m.min_altitude = min_alt;
    m.max_g = max_g;
    m.max_speed = max_speed;
    m.pitch_rate_std = arr_std(pitch_rates, n_pr);
    m.zero_crossings = zc;
    return m;
}

static float theoretical_alt_loss(float speed, float g_load) {
    return (speed * speed) / (G_REC * (g_load - 1.0f));
}

typedef enum {
    PASS_NO_CRASH,
    PASS_RECOVERS,
    PASS_SUCCESS,
    PASS_ALT_LOSS_LT,
} PassMode;

static int print_result(const char* name, RecoveryMetrics m,
                         PassMode mode, float alt_limit, float theoretical) {
    int passed = 0;
    if (mode == PASS_NO_CRASH) passed = !m.crashed;
    else if (mode == PASS_RECOVERS || mode == PASS_SUCCESS) passed = m.success;
    else if (mode == PASS_ALT_LOSS_LT) passed = (m.altitude_lost < alt_limit) && !m.crashed;

    const char* status;
    if (m.crashed) status = "CRASH";
    else if (passed) status = "OK";
    else if (m.timeout) status = "TIMEOUT";
    else status = "FAIL";

    char osc[16] = "";
    if (m.pitch_rate_std > 0.3f || m.zero_crossings > 3) {
        snprintf(osc, sizeof(osc), " [OSC!]");
    }
    char theory_str[48] = "";
    if (theoretical > 0.0f) {
        snprintf(theory_str, sizeof(theory_str), ", theory: %.0fm", theoretical);
    }

    printf("%-22s alt_lost=%5.0fm  min_alt=%5.0fm  time=%4.1fs  "
           "max_g=%4.1f  max_spd=%4.0fm/s  pitch_std=%.3f  zc=%2d  [%s]%s%s\n",
           name, m.altitude_lost, m.min_altitude, m.recovery_time,
           m.max_g, m.max_speed, m.pitch_rate_std, m.zero_crossings,
           status, osc, theory_str);
    return 0;
}

/* ============================================================
 * Test cases — same parameters as test_flight_recovery.py
 * ============================================================ */

static int test_dive_30_cruise(void) {
    RecoveryMetrics m = run_recovery(-30, 120, 1500, 0, 20.0f, 6.0f);
    return print_result("dive_30_cruise", m, PASS_ALT_LOSS_LT, 300.0f,
                        theoretical_alt_loss(120, 4));
}
static int test_dive_45_cruise(void) {
    RecoveryMetrics m = run_recovery(-45, 120, 1500, 0, 20.0f, 6.0f);
    return print_result("dive_45_cruise", m, PASS_ALT_LOSS_LT, 500.0f,
                        theoretical_alt_loss(120, 4));
}
static int test_dive_60_cruise(void) {
    RecoveryMetrics m = run_recovery(-60, 120, 1500, 0, 20.0f, 6.0f);
    return print_result("dive_60_cruise", m, PASS_ALT_LOSS_LT, 700.0f,
                        theoretical_alt_loss(120, 4));
}
static int test_dive_90_vertical(void) {
    RecoveryMetrics m = run_recovery(-90, 100, 2000, 0, 20.0f, 6.0f);
    return print_result("dive_90_vertical", m, PASS_RECOVERS, 0,
                        theoretical_alt_loss(100, 4));
}
static int test_highspeed_150(void) {
    RecoveryMetrics m = run_recovery(-45, 150, 2000, 0, 20.0f, 6.0f);
    return print_result("highspeed_150", m, PASS_SUCCESS, 0,
                        theoretical_alt_loss(150, 4));
}
static int test_highspeed_175(void) {
    RecoveryMetrics m = run_recovery(-45, 175, 2000, 0, 20.0f, 6.0f);
    return print_result("highspeed_175", m, PASS_SUCCESS, 0,
                        theoretical_alt_loss(175, 4));
}
static int test_rolling_30_30(void) {
    RecoveryMetrics m = run_recovery(-30, 120, 1500, 30, 20.0f, 6.0f);
    return print_result("rolling_30_30", m, PASS_SUCCESS, 0, 0);
}
static int test_rolling_45_60(void) {
    RecoveryMetrics m = run_recovery(-45, 120, 1500, 60, 20.0f, 6.0f);
    return print_result("rolling_45_60", m, PASS_SUCCESS, 0, 0);
}
static int test_rolling_60_90(void) {
    RecoveryMetrics m = run_recovery(-60, 120, 1500, 90, 20.0f, 6.0f);
    return print_result("rolling_60_90", m, PASS_NO_CRASH, 0, 0);
}
static int test_rolling_inverted(void) {
    RecoveryMetrics m = run_recovery(-45, 120, 1500, 180, 20.0f, 6.0f);
    return print_result("rolling_inverted", m, PASS_SUCCESS, 0, 0);
}
static int test_extreme_80_150_60(void) {
    RecoveryMetrics m = run_recovery(-80, 150, 2500, 60, 20.0f, 6.0f);
    return print_result("extreme_80_150_60", m, PASS_NO_CRASH, 0,
                        theoretical_alt_loss(150, 6));
}
static int test_extreme_80_175_0(void) {
    RecoveryMetrics m = run_recovery(-80, 175, 2500, 0, 20.0f, 6.0f);
    return print_result("extreme_80_175_0", m, PASS_NO_CRASH, 0,
                        theoretical_alt_loss(175, 6));
}
static int test_extreme_80_150_90(void) {
    RecoveryMetrics m = run_recovery(-80, 150, 2500, 90, 20.0f, 6.0f);
    return print_result("extreme_80_150_90", m, PASS_NO_CRASH, 0,
                        theoretical_alt_loss(150, 6));
}
static int test_extreme_70_160_45(void) {
    RecoveryMetrics m = run_recovery(-70, 160, 2500, 45, 20.0f, 6.0f);
    return print_result("extreme_70_160_45", m, PASS_NO_CRASH, 0,
                        theoretical_alt_loss(160, 6));
}
static int test_extreme_85_140_inv(void) {
    RecoveryMetrics m = run_recovery(-85, 140, 2500, 180, 20.0f, 6.0f);
    return print_result("extreme_85_140_inv", m, PASS_NO_CRASH, 0,
                        theoretical_alt_loss(140, 6));
}
static int test_critical_500m(void) {
    RecoveryMetrics m = run_recovery(-45, 120, 500, 0, 20.0f, 6.0f);
    return print_result("critical_500m", m, PASS_NO_CRASH, 0, 0);
}
static int test_critical_300m(void) {
    RecoveryMetrics m = run_recovery(-30, 120, 300, 0, 20.0f, 6.0f);
    return print_result("critical_300m", m, PASS_NO_CRASH, 0, 0);
}
static int test_critical_extreme(void) {
    RecoveryMetrics m = run_recovery(-70, 150, 1000, 30, 20.0f, 6.0f);
    return print_result("critical_extreme", m, PASS_NO_CRASH, 0,
                        theoretical_alt_loss(150, 6));
}

int main(void) {
    int fails = 0;
    printf("\n--- Core Dive Tests ---\n");
    fails += test_dive_30_cruise();
    fails += test_dive_45_cruise();
    fails += test_dive_60_cruise();
    fails += test_dive_90_vertical();
    printf("\n--- High-Speed Tests ---\n");
    fails += test_highspeed_150();
    fails += test_highspeed_175();
    printf("\n--- Rolling Dive Tests ---\n");
    fails += test_rolling_30_30();
    fails += test_rolling_45_60();
    fails += test_rolling_60_90();
    fails += test_rolling_inverted();
    printf("\n--- Extreme Dive Tests (Combat Scenarios) ---\n");
    fails += test_extreme_80_150_60();
    fails += test_extreme_80_175_0();
    fails += test_extreme_80_150_90();
    fails += test_extreme_70_160_45();
    fails += test_extreme_85_140_inv();
    printf("\n--- Critical Altitude Tests ---\n");
    fails += test_critical_500m();
    fails += test_critical_300m();
    fails += test_critical_extreme();
    printf("\n%d hard failures\n", fails);
    return fails;
}
