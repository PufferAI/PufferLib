/*
 * sweep_smoothness.c - Brute-force harness for finding good physics params.
 *
 * Same 30-case (maneuver x airspeed) matrix as test_smoothness, but takes
 * physics knobs via CLI flags and emits ONE machine-readable line per run:
 *
 *   ctrl_slope=X ctrl_min=Y damp_slope=Z damp_mult=W \
 *     smooth=A check=B osc=C unstable=D \
 *     score=S sum_om=O sum_dev=D2 grow=G
 *
 * Driven by sweep_smoothness.sh which fans out parameter combinations via
 * xargs -P, then sorts results by `score` ascending.
 *
 * Score = sum across 30 cases of (omega_x_std + omega_y_std + 0.05*bank_dev
 *         + 0.05*pitch_dev) PLUS 5.0 per growing/crashed case.
 * Lower is better. Continuous so two near-equal configs don't tie.
 */
#include "test_common.h"

#define TICKS 250
#define SETTLE_OFF 50
#define SETTLED_N (TICKS - SETTLE_OFF)

typedef enum {
    MAN_LEVEL_HOLD, MAN_BANK_30, MAN_BANK_60,
    MAN_PITCH_UP_10, MAN_PITCH_DN_10, MAN_SNAP_ELEV,
    NUM_MAN
} ManeuverId;

static const char* MAN_NAME[NUM_MAN] = {
    "level_hold", "bank_30", "bank_60", "pitch_+10", "pitch_-10", "snap_elev"
};
static const float MAN_BANK_TARGET[NUM_MAN]  = { 0, 30, 60,  0,  0,  0 };
static const float MAN_PITCH_TARGET[NUM_MAN] = { 0,  0,  0, 10,-10,  0 };

static const int SPEEDS[] = {80, 100, 120, 140, 160};
#define NUM_SPEEDS 5

typedef struct {
    float omega_x_std, omega_y_std;
    float bank_max_dev, pitch_max_dev;
    int growing, crashed;
    const char* verdict;
} CaseResult;

static const char* classify(const CaseResult* m) {
    if (m->growing || m->crashed) return "UNSTABLE";
    int oscillating = (m->omega_x_std >= 0.3f) || (m->omega_y_std >= 0.3f);
    if (oscillating) return "OSCILLATING";
    int smooth = (m->omega_x_std < 0.1f) && (m->omega_y_std < 0.1f)
              && (m->bank_max_dev < 5.0f) && (m->pitch_max_dev < 5.0f);
    if (smooth) return "SMOOTH";
    int near = (m->omega_x_std < 0.3f) && (m->omega_y_std < 0.3f)
            && (m->bank_max_dev < 15.0f) && (m->pitch_max_dev < 15.0f);
    return near ? "CHECK" : "OSCILLATING";
}

static void run_case(ManeuverId mid, int V, CaseResult* out,
                     float ctrl_slope, float ctrl_min, float ctrl_vref,
                     float damp_slope, float damp_mult) {
    TestEnv t; setup_env(&t, 0);
    t.env.max_steps = TICKS + 100;

    /* Override physics knobs (all already plumbed in step_plane_with_params). */
    t.env.flight_params.control_scale_slope = ctrl_slope;
    t.env.flight_params.control_scale_min   = ctrl_min;
    t.env.flight_params.control_v_ref       = ctrl_vref;
    t.env.flight_params.damping_scale_slope = damp_slope;
    t.env.flight_params.damping_multiplier  = damp_mult;

    force_state(&t.env,
        0.0f, 0.0f, 1500.0f,   (float)V, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.5f,
        -9999.0f,-9999.0f,-9999.0f, -9999.0f,-9999.0f,-9999.0f,
        -9999.0f,-9999.0f,-9999.0f,-9999.0f, 0, -1, -1);

    float bank_dev[TICKS], pitch_dev[TICKS];
    float ox[TICKS], oy[TICKS];

    float bank_target  = MAN_BANK_TARGET[mid];
    float pitch_target = MAN_PITCH_TARGET[mid];

    int completed = 0;
    for (int step = 0; step < TICKS; step++) {
        Plane* p = &t.env.player;
        float thr = ap_hold_speed(p, (float)V);
        float elev = 0.0f, ail = 0.0f;

        switch (mid) {
            case MAN_LEVEL_HOLD: case MAN_BANK_30: case MAN_BANK_60:
                ap_hold_bank_and_level(p, bank_target, &elev, &ail);
                break;
            case MAN_PITCH_UP_10: case MAN_PITCH_DN_10:
                elev = ap_hold_pitch(p, pitch_target);
                ail  = ap_hold_bank(p, 0.0f);
                break;
            case MAN_SNAP_ELEV:
                if (step < 25) { elev = -0.5f; ail = ap_hold_bank(p, 0.0f); }
                else { elev = ap_hold_pitch(p, 0.0f); ail = ap_hold_bank(p, 0.0f); }
                break;
            default: break;
        }

        float a[5] = {thr, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));

        bank_dev[step]  = fabsf(plane_bank_deg(p)  - bank_target);
        pitch_dev[step] = fabsf(plane_pitch_deg(p) - pitch_target);
        ox[step] = p->omega.x; oy[step] = p->omega.y;

        t_step(&t);
        completed = step + 1;
        if (t.env.terminals[0]) break;
    }

    out->crashed = (completed < TICKS);
    if (out->crashed) {
        out->omega_x_std = out->omega_y_std = 0.0f;
        out->bank_max_dev = out->pitch_max_dev = 0.0f;
        out->growing = 0;
        out->verdict = classify(out);
        return;
    }

    const float* sox = ox + SETTLE_OFF;
    const float* soy = oy + SETTLE_OFF;
    int n = SETTLED_N;

    out->omega_x_std  = arr_std(sox, n);
    out->omega_y_std  = arr_std(soy, n);
    out->bank_max_dev = arr_max(bank_dev  + SETTLE_OFF, n);
    out->pitch_max_dev= arr_max(pitch_dev + SETTLE_OFF, n);

    int half = n / 2;
    const float MIN_GROW_STD = 0.1f;
    float vy1 = arr_var(soy, half);
    float vy2 = arr_var(soy + (n - half), half);
    float vx1 = arr_var(sox, half);
    float vx2 = arr_var(sox + (n - half), half);
    int gy = (vy1 > 1e-6f) && (vy2 > vy1 * 1.5f) && (sqrtf(vy2) > MIN_GROW_STD);
    int gx = (vx1 > 1e-6f) && (vx2 > vx1 * 1.5f) && (sqrtf(vx2) > MIN_GROW_STD);
    out->growing = (gy || gx);
    out->verdict = classify(out);
}

/* Maneuverability tests. Mimic test_flight_physics::recovery_roll_rate /
 * recovery_pitch_rate: spawn at V=120 with FULL throttle (so the plane
 * accelerates into the high-V regime where authority is reduced), apply
 * full control deflection on the target axis, run 4 seconds. Returns the
 * peak body rate in deg/s.
 *
 * axis: 0 = roll (aileron),  1 = pitch (elevator)
 */
static float run_max_rate(int axis, int V,
                          float ctrl_slope, float ctrl_min, float ctrl_vref,
                          float damp_slope, float damp_mult) {
    TestEnv t; setup_env(&t, 0);
    t.env.max_steps = 300;

    t.env.flight_params.control_scale_slope = ctrl_slope;
    t.env.flight_params.control_scale_min   = ctrl_min;
    t.env.flight_params.control_v_ref       = ctrl_vref;
    t.env.flight_params.damping_scale_slope = damp_slope;
    t.env.flight_params.damping_multiplier  = damp_mult;

    /* Full throttle, altitude 2000, identity ori. Matches recovery_*_rate. */
    force_state(&t.env,
        0.0f, 0.0f, 2000.0f,   (float)V, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        -9999.0f,-9999.0f,-9999.0f, -9999.0f,-9999.0f,-9999.0f,
        -9999.0f,-9999.0f,-9999.0f,-9999.0f, 0, -1, -1);

    float a[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (axis == 0) a[2] = +1.0f;       /* full right aileron */
    else           a[1] = -1.0f;       /* full nose-up elevator */

    float peak = 0.0f;
    for (int step = 0; step < 200; step++) {  /* 4 sec, matches recovery_*_rate */
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;

        float rate = (axis == 0) ? t.env.player.omega.x : -t.env.player.omega.y;
        rate *= AP_RAD_TO_DEG;
        if (rate > peak) peak = rate;
    }
    return peak;
}

int main(int argc, char** argv) {
    /* Defaults match flightlib.h compile-time defaults (no scaling). */
    float ctrl_slope = 0.0f, ctrl_min = 1.0f;
    float damp_slope = 0.0f, damp_mult = 1.0f;
    float ctrl_vref = 100.0f;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--ctrl-slope") && i + 1 < argc) ctrl_slope = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--ctrl-min")   && i + 1 < argc) ctrl_min   = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--ctrl-vref")  && i + 1 < argc) ctrl_vref  = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--damp-slope") && i + 1 < argc) damp_slope = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--damp-mult")  && i + 1 < argc) damp_mult  = (float)atof(argv[++i]);
    }

    int n_smooth = 0, n_check = 0, n_osc = 0, n_unstable = 0;
    double sum_om = 0.0, sum_dev = 0.0;
    int n_grow = 0;

    for (int mi = 0; mi < NUM_MAN; mi++) {
        for (int si = 0; si < NUM_SPEEDS; si++) {
            CaseResult r;
            run_case((ManeuverId)mi, SPEEDS[si], &r,
                     ctrl_slope, ctrl_min, ctrl_vref, damp_slope, damp_mult);

            if      (!strcmp(r.verdict, "SMOOTH"))      n_smooth++;
            else if (!strcmp(r.verdict, "CHECK"))       n_check++;
            else if (!strcmp(r.verdict, "OSCILLATING")) n_osc++;
            else                                        n_unstable++;

            sum_om  += r.omega_x_std + r.omega_y_std;
            sum_dev += r.bank_max_dev + r.pitch_max_dev;
            if (r.growing || r.crashed) n_grow++;
        }
    }

    /* Maneuverability tests: max roll/pitch rate at low + high speed.
     * Min targets: 60 deg/s roll, 30 deg/s pitch. Below those, penalty
     * grows linearly so the brute-force search rejects over-damped configs
     * that achieve smoothness by sacrificing combat agility. */
    float roll_80  = run_max_rate(0, 80,  ctrl_slope, ctrl_min, ctrl_vref, damp_slope, damp_mult);
    float roll_120 = run_max_rate(0, 120, ctrl_slope, ctrl_min, ctrl_vref, damp_slope, damp_mult);
    float pitch_80  = run_max_rate(1, 80,  ctrl_slope, ctrl_min, ctrl_vref, damp_slope, damp_mult);
    float pitch_120 = run_max_rate(1, 120, ctrl_slope, ctrl_min, ctrl_vref, damp_slope, damp_mult);

    const float ROLL_TARGET = 60.0f, PITCH_TARGET = 30.0f;
    double maneuver_penalty = 0.0;
    if (roll_80  < ROLL_TARGET)  maneuver_penalty += (ROLL_TARGET  - roll_80)  * 0.5;
    if (roll_120 < ROLL_TARGET)  maneuver_penalty += (ROLL_TARGET  - roll_120) * 0.5;
    if (pitch_80  < PITCH_TARGET) maneuver_penalty += (PITCH_TARGET - pitch_80)  * 0.5;
    if (pitch_120 < PITCH_TARGET) maneuver_penalty += (PITCH_TARGET - pitch_120) * 0.5;

    /* Continuous score: lower is better. Body-rate noise dominates,
     * with attitude-deviation as secondary, and a flat 5.0 hit per
     * growing/crashed case so divergence outweighs near-misses.
     * Maneuver penalty added so over-damping costs as much as under-damping. */
    double smooth_score = sum_om + 0.05 * sum_dev + 5.0 * n_grow;
    double score = smooth_score + maneuver_penalty;

    printf("ctrl_vref=%.1f ctrl_slope=%.4f ctrl_min=%.3f damp_slope=%.4f damp_mult=%.3f "
           "smooth=%d check=%d osc=%d unstable=%d "
           "score=%.4f smooth_score=%.4f maneuver_pen=%.4f "
           "roll80=%.1f roll120=%.1f pitch80=%.1f pitch120=%.1f "
           "sum_om=%.4f sum_dev=%.2f grow=%d\n",
           ctrl_vref, ctrl_slope, ctrl_min, damp_slope, damp_mult,
           n_smooth, n_check, n_osc, n_unstable,
           score, smooth_score, maneuver_penalty,
           roll_80, roll_120, pitch_80, pitch_120,
           sum_om, sum_dev, n_grow);

    return 0;
}
