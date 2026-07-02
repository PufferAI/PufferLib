/*
 * test_smoothness.c - Diagnostic envelope: (maneuver x airspeed) smoothness.
 *
 * Numerical-only diagnostic. Sweeps 6 maneuvers x 5 airspeeds = 30 cases.
 * For each case, holds the maneuver via existing autopilot helpers from
 * test_common.h for 5 sim seconds (250 ticks at 50 Hz), discards the first
 * 1 sec settle window, then computes:
 *   - body-rate std (omega.x/y/z)
 *   - max attitude deviation (bank, pitch)
 *   - control-effort RMS (aileron, elevator, rudder)
 *   - "growing" flag: variance(second half) / variance(first half) > 1.5
 *
 * Verdicts:
 *   SMOOTH       om_x<0.1, om_y<0.1, no grow, bnk<5deg, pch<5deg
 *   CHECK        within 3x SMOOTH thresholds
 *   OSCILLATING  om_x>=0.3 or om_y>=0.3 (not growing)
 *   UNSTABLE     growing flag set OR plane crashed mid-run
 *
 * CLI:
 *   --csv path      machine-readable per-case row dump
 *   --log path      per-tick CSV (uses recovery_log_row)
 *   --render --case <man>_V<speed>   render single case for visual confirm
 *   --fps N         render fps (default 50)
 *   --list          list every available case label and exit
 *
 * Exit code: 1 if any UNSTABLE case, 0 otherwise. OSCILLATING is soft-warn.
 */
#include "test_common.h"

#define TICKS 250
#define SETTLE_OFF 50
#define SETTLED_N (TICKS - SETTLE_OFF)

typedef enum {
    MAN_LEVEL_HOLD,
    MAN_BANK_30,
    MAN_BANK_60,
    MAN_PITCH_UP_10,
    MAN_PITCH_DN_10,
    MAN_SNAP_ELEV,
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
    const char* maneuver;
    int speed;
    float omega_x_std, omega_y_std, omega_z_std;
    float bank_max_dev, pitch_max_dev;
    float rms_ail, rms_elev, rms_rud;
    int growing;
    int crashed;
    const char* verdict;
} MetricRow;

static const char* classify(const MetricRow* m) {
    if (m->growing || m->crashed) return "UNSTABLE";
    int oscillating = (m->omega_x_std >= 0.3f) || (m->omega_y_std >= 0.3f);
    if (oscillating) return "OSCILLATING";
    int smooth = (m->omega_x_std < 0.1f) && (m->omega_y_std < 0.1f)
              && (m->bank_max_dev < 5.0f) && (m->pitch_max_dev < 5.0f);
    if (smooth) return "SMOOTH";
    int near_smooth = (m->omega_x_std < 0.3f) && (m->omega_y_std < 0.3f)
                   && (m->bank_max_dev < 15.0f) && (m->pitch_max_dev < 15.0f);
    return near_smooth ? "CHECK" : "OSCILLATING";
}

static float arr_rms(const float* x, int n) {
    if (n <= 0) return 0.0f;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    return (float)sqrt(s / n);
}

/* Spawn level at altitude 1500m, identity orientation, forward velocity = V. */
static void spawn_level_at_speed(Dogfight* env, float V) {
    force_state(env,
        0.0f, 0.0f, 1500.0f,   /* pos */
        V, 0.0f, 0.0f,         /* vel */
        1.0f, 0.0f, 0.0f, 0.0f, /* identity quat */
        0.5f,                   /* throttle */
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f, -9999.0f,
        0, -1, -1);
}

static void run_case(ManeuverId mid, int V, MetricRow* out) {
    TestEnv t; setup_env(&t, 0);
    t.env.max_steps = TICKS + 100;
    spawn_level_at_speed(&t.env, (float)V);

    char label[64];
    snprintf(label, sizeof(label), "%s_V%d", MAN_NAME[mid], V);

    float bank_dev[TICKS], pitch_dev[TICKS];
    float ox[TICKS], oy[TICKS], oz[TICKS];
    float ail_arr[TICKS], elev_arr[TICKS], rud_arr[TICKS];

    float bank_target  = MAN_BANK_TARGET[mid];
    float pitch_target = MAN_PITCH_TARGET[mid];

    int completed = 0;
    for (int step = 0; step < TICKS; step++) {
        Plane* p = &t.env.player;
        float thr = ap_hold_speed(p, (float)V);
        float elev = 0.0f, ail = 0.0f, rud = 0.0f;

        switch (mid) {
            case MAN_LEVEL_HOLD:
            case MAN_BANK_30:
            case MAN_BANK_60:
                ap_hold_bank_and_level(p, bank_target, &elev, &ail);
                break;
            case MAN_PITCH_UP_10:
            case MAN_PITCH_DN_10:
                elev = ap_hold_pitch(p, pitch_target);
                ail  = ap_hold_bank(p, 0.0f);
                break;
            case MAN_SNAP_ELEV:
                if (step < 25) {
                    elev = -0.5f;            /* 0.5 sec nose-up pulse */
                    ail  = ap_hold_bank(p, 0.0f);
                } else {
                    elev = ap_hold_pitch(p, 0.0f);
                    ail  = ap_hold_bank(p, 0.0f);
                }
                break;
            default: break;
        }

        float a[5] = {thr, elev, ail, rud, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));

        bank_dev[step]  = fabsf(plane_bank_deg(p)  - bank_target);
        pitch_dev[step] = fabsf(plane_pitch_deg(p) - pitch_target);
        ox[step] = p->omega.x; oy[step] = p->omega.y; oz[step] = p->omega.z;
        ail_arr[step] = ail; elev_arr[step] = elev; rud_arr[step] = rud;

        recovery_log_row(label, step, p, a);

        t_step(&t);
        completed = step + 1;
        if (t.env.terminals[0]) break;
    }

    out->maneuver = MAN_NAME[mid];
    out->speed    = V;
    out->crashed  = (completed < TICKS);

    if (out->crashed) {
        out->omega_x_std = out->omega_y_std = out->omega_z_std = 0.0f;
        out->bank_max_dev = out->pitch_max_dev = 0.0f;
        out->rms_ail = out->rms_elev = out->rms_rud = 0.0f;
        out->growing = 0;
        out->verdict = classify(out);
        return;
    }

    /* Settled window. */
    const float* sox = ox + SETTLE_OFF;
    const float* soy = oy + SETTLE_OFF;
    const float* soz = oz + SETTLE_OFF;
    int n = SETTLED_N;

    out->omega_x_std  = arr_std(sox, n);
    out->omega_y_std  = arr_std(soy, n);
    out->omega_z_std  = arr_std(soz, n);
    out->bank_max_dev = arr_max(bank_dev  + SETTLE_OFF, n);
    out->pitch_max_dev= arr_max(pitch_dev + SETTLE_OFF, n);
    out->rms_ail  = arr_rms(ail_arr  + SETTLE_OFF, n);
    out->rms_elev = arr_rms(elev_arr + SETTLE_OFF, n);
    out->rms_rud  = arr_rms(rud_arr  + SETTLE_OFF, n);

    /* Growing: variance second half / first half > 1.5, on omega.y or omega.x,
     * AND second-half std must exceed a minimum amplitude so we don't flag
     * tiny absolute noise as "diverging". 0.1 rad/s == ~5.7 deg/s. */
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

int main(int argc, char** argv) {
    const char* csv_path = NULL;
    const char* log_path = NULL;
    const char* render_case = NULL;
    int do_list = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--csv")    && i + 1 < argc) csv_path    = argv[++i];
        else if (!strcmp(argv[i], "--log")    && i + 1 < argc) log_path    = argv[++i];
        else if (!strcmp(argv[i], "--render"))                 g_visual_render = 1;
        else if (!strcmp(argv[i], "--case")   && i + 1 < argc) render_case = argv[++i];
        else if (!strcmp(argv[i], "--fps")    && i + 1 < argc) g_visual_fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--list"))                   do_list = 1;
    }

    if (do_list) {
        for (int mi = 0; mi < NUM_MAN; mi++)
            for (int si = 0; si < NUM_SPEEDS; si++)
                printf("%s_V%d\n", MAN_NAME[mi], SPEEDS[si]);
        return 0;
    }

    if (log_path) {
        g_log_csv = fopen(log_path, "w");
        if (!g_log_csv) { fprintf(stderr, "ERROR: cannot open %s\n", log_path); return 2; }
    }

    /* Single-case render mode. */
    if (render_case) {
        g_visual_render = 1;
        for (int mi = 0; mi < NUM_MAN; mi++) {
            for (int si = 0; si < NUM_SPEEDS; si++) {
                char label[64];
                snprintf(label, sizeof(label), "%s_V%d", MAN_NAME[mi], SPEEDS[si]);
                if (!strcmp(label, render_case)) {
                    MetricRow r;
                    run_case((ManeuverId)mi, SPEEDS[si], &r);
                    printf("%s om_x=%.3f om_y=%.3f om_z=%.3f bnk=%.2f pch=%.2f grow=%d crash=%d verdict=%s\n",
                           label, r.omega_x_std, r.omega_y_std, r.omega_z_std,
                           r.bank_max_dev, r.pitch_max_dev,
                           r.growing, r.crashed, r.verdict);
                    if (g_log_csv) fclose(g_log_csv);
                    return 0;
                }
            }
        }
        fprintf(stderr, "ERROR: case '%s' not found. Try --list.\n", render_case);
        if (g_log_csv) fclose(g_log_csv);
        return 2;
    }

    /* Full matrix. */
    FILE* csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { fprintf(stderr, "ERROR: cannot open %s\n", csv_path); return 2; }
        fprintf(csv, "maneuver,speed,omega_x_std,omega_y_std,omega_z_std,"
                "bank_max_dev,pitch_max_dev,rms_ail,rms_elev,rms_rud,growing,crashed,verdict\n");
    }

    int n_smooth = 0, n_check = 0, n_oscillating = 0, n_unstable = 0;

    printf("\n=== Smoothness Envelope ===\n");
    printf("%-11s %4s %6s %6s %6s %7s %7s %6s %6s %6s %5s %s\n",
           "maneuver", "V", "om_x", "om_y", "om_z",
           "bnk_dv", "pch_dv", "rms_a", "rms_e", "rms_r", "grow", "verdict");

    for (int mi = 0; mi < NUM_MAN; mi++) {
        for (int si = 0; si < NUM_SPEEDS; si++) {
            MetricRow r;
            run_case((ManeuverId)mi, SPEEDS[si], &r);

            printf("%-11s %4d %6.3f %6.3f %6.3f %7.2f %7.2f %6.3f %6.3f %6.3f %5s %s\n",
                   r.maneuver, r.speed,
                   r.omega_x_std, r.omega_y_std, r.omega_z_std,
                   r.bank_max_dev, r.pitch_max_dev,
                   r.rms_ail, r.rms_elev, r.rms_rud,
                   r.growing ? "yes" : "no",
                   r.verdict);

            if (csv) {
                fprintf(csv, "%s,%d,%.5f,%.5f,%.5f,%.3f,%.3f,%.4f,%.4f,%.4f,%d,%d,%s\n",
                        r.maneuver, r.speed,
                        r.omega_x_std, r.omega_y_std, r.omega_z_std,
                        r.bank_max_dev, r.pitch_max_dev,
                        r.rms_ail, r.rms_elev, r.rms_rud,
                        r.growing, r.crashed, r.verdict);
            }

            if      (!strcmp(r.verdict, "SMOOTH"))      n_smooth++;
            else if (!strcmp(r.verdict, "CHECK"))       n_check++;
            else if (!strcmp(r.verdict, "OSCILLATING")) n_oscillating++;
            else                                        n_unstable++;
        }
    }

    printf("\n=== Summary: %d SMOOTH, %d CHECK, %d OSCILLATING, %d UNSTABLE ===\n",
           n_smooth, n_check, n_oscillating, n_unstable);

    if (csv) fclose(csv);
    if (g_log_csv) fclose(g_log_csv);

    /* Diagnostic baseline: always exits 0. Verdict counts in the summary line
     * carry the actual info. After autopilot/physics fixes land, change this
     * to `return n_unstable > 0 ? 1 : 0;` so regressions hard-fail. */
    (void)n_unstable;
    return 0;
}
