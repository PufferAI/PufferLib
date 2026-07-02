/*
 * test_action_symmetry.c — does the env respond symmetrically to mirrored
 * left/right state? Phase 1 of the "always banks right" investigation.
 *
 * Hard tests T1-T4: action→roll direction and obs[13] (target azimuth)
 * sign for opp-left vs opp-right at curriculum_disabled defaults.
 *
 * Soft test T5: mirror-pair test. Two scenarios identical except opp
 * y-coord is flipped. Step the env identically; expected per-slot
 * relation under mirror is "NEGATE" for y-related slots and "INVARIANT"
 * for everything else. Print a per-slot asymmetry table flagging any
 * slot that doesn't obey the expected relation.
 *
 * Also writes /tmp/df_symmetry_right.jsonl and /tmp/df_symmetry_left.jsonl
 * with full obs+action+geometry per step for offline grep/analysis.
 */
#include "test_common.h"

#define DEG (3.14159265f / 180.0f)
#define N_OBS 26

/* Mirror sign per obs slot for obs_scheme=1 (OBS_OPPONENT_AWARE, 26 obs).
 *   +1 = INVARIANT under y-mirror (e.g. forward speed, altitude, range)
 *   -1 = NEGATES under y-mirror (e.g. sideslip, roll rate, target azimuth)
 *
 * Source of truth: dogfight_observations.h:690-732 (compute_obs_opponent_aware_for_plane).
 */
static const int MIRROR_SIGN[N_OBS] = {
    +1, /* [0]  fwd speed       (vel_body.x)        */
    -1, /* [1]  sideslip        (vel_body.y)        */
    +1, /* [2]  climb rate      (vel_body.z)        */
    -1, /* [3]  roll rate       (omega.x)           */
    +1, /* [4]  pitch rate      (omega.y)           */
    -1, /* [5]  yaw rate        (omega.z)           */
    +1, /* [6]  AoA                                 */
    +1, /* [7]  altitude                            */
    +1, /* [8]  g_force                             */
    +1, /* [9]  energy                              */
    +1, /* [10] up_x (world)                        */
    -1, /* [11] up_y (world)                        */
    +1, /* [12] up_z (world)                        */
    -1, /* [13] target azimuth                      */
    +1, /* [14] target elevation                    */
    +1, /* [15] range                               */
    +1, /* [16] closure                             */
    +1, /* [17] energy advantage                    */
    +1, /* [18] target aspect                       */
    +1, /* [19] opp pitch rate                      */
    -1, /* [20] opp roll rate                       */
    +1, /* [21] opp up_x                            */
    -1, /* [22] opp up_y                            */
    +1, /* [23] opp up_z                            */
    +1, /* [24] opp speed                           */
    +1, /* [25] timer                               */
};

static const char* SLOT_LABEL[N_OBS] = {
    "fwd_spd", "sideslip", "climb_rate", "roll_rate", "pitch_rate",
    "yaw_rate", "aoa", "altitude", "g_force", "energy",
    "up_x", "up_y", "up_z", "tgt_az", "tgt_el",
    "range", "closure", "E_adv", "aspect", "opp_pr",
    "opp_rr", "opp_up_x", "opp_up_y", "opp_up_z", "opp_spd",
    "timer"
};

/* Place player at origin, level. opp_y_sign = +1 (right) or -1 (left). */
static void setup_lr(TestEnv* t, float opp_y_sign) {
    setup_env(t, /*obs_scheme=*/1);
    force_state(&t->env,
        /* player level at (0,0,1000), 80 m/s +X */
        0.0f, 0.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.5f,
        /* opponent 200m ahead, ±100m on Y */
        200.0f, opp_y_sign * 100.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0, 0, 0);
}

/* T1: aileron=+1.0 should roll RIGHT (up.y goes negative, omega.x > 0). */
static int test_aileron_right(void) {
    TestEnv t; setup_lr(&t, +1.0f);
    float a[5] = {0.5f, 0.0f, +1.0f, 0.0f, 0.0f};
    run_steps(&t, 30, a);
    float up_y = plane_up(&t.env.player).y;
    float omega_x = t.env.player.omega.x;
    int pass = (up_y < -0.05f) && (omega_x > 0.01f);
    printf("T1 aileron=+1.0 => roll RIGHT: up.y=%+.3f omega.x=%+.3f [%s]\n",
           up_y, omega_x, pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* T2: aileron=-1.0 should roll LEFT (up.y > 0, omega.x < 0). */
static int test_aileron_left(void) {
    TestEnv t; setup_lr(&t, +1.0f);
    float a[5] = {0.5f, 0.0f, -1.0f, 0.0f, 0.0f};
    run_steps(&t, 30, a);
    float up_y = plane_up(&t.env.player).y;
    float omega_x = t.env.player.omega.x;
    int pass = (up_y > +0.05f) && (omega_x < -0.01f);
    printf("T2 aileron=-1.0 => roll LEFT:  up.y=%+.3f omega.x=%+.3f [%s]\n",
           up_y, omega_x, pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* T3: opp at +Y (right) ⇒ obs[13] (target azimuth) > 0. */
static int test_obs_az_right(void) {
    TestEnv t; setup_lr(&t, +1.0f);
    int pass = t.observations[13] > 0.05f;
    printf("T3 opp on RIGHT => obs[13]=%+.4f [%s]\n",
           t.observations[13], pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* T4: opp at -Y (left) ⇒ obs[13] < 0; magnitude matches T3. */
static int test_obs_az_left(void) {
    TestEnv t_r; setup_lr(&t_r, +1.0f);
    TestEnv t_l; setup_lr(&t_l, -1.0f);
    int sign_ok  = t_l.observations[13] < -0.05f;
    int mag_ok   = fabsf(fabsf(t_r.observations[13]) -
                          fabsf(t_l.observations[13])) < 1e-5f;
    int pass = sign_ok && mag_ok;
    printf("T4 opp on LEFT  => obs[13]=%+.4f (mirrors T3 |delta|=%.2e) [%s]\n",
           t_l.observations[13],
           fabsf(fabsf(t_r.observations[13]) - fabsf(t_l.observations[13])),
           pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* T5 (soft): mirror-pair. Two envs, opp at +Y vs -Y, run identical
 * forward-only actions. Per slot, accumulate |obs_R[i] - MIRROR_SIGN[i]*obs_L[i]|
 * over 200 steps. Print table; flag any slot with mean asymmetry > 1e-3.
 *
 * Also write /tmp/df_symmetry_right.jsonl and /tmp/df_symmetry_left.jsonl.
 */
static int test_mirror_pair(void) {
    TestEnv t_r; setup_lr(&t_r, +1.0f);
    TestEnv t_l; setup_lr(&t_l, -1.0f);

    float action[5] = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f};

    FILE* f_r = fopen("/tmp/df_symmetry_right.jsonl", "w");
    FILE* f_l = fopen("/tmp/df_symmetry_left.jsonl", "w");

    double sum_asym[N_OBS] = {0};
    int n = 0;
    const int STEPS = 200;

    for (int s = 0; s < STEPS; s++) {
        memcpy(t_r.env.actions, action, sizeof(action));
        memcpy(t_l.env.actions, action, sizeof(action));
        c_step(&t_r.env);
        c_step(&t_l.env);

        for (int i = 0; i < N_OBS; i++) {
            float diff = fabsf(t_r.observations[i] -
                               (float)MIRROR_SIGN[i] * t_l.observations[i]);
            sum_asym[i] += diff;
        }
        n++;

        /* Dump JSONL traces */
        for (FILE** ff = (FILE*[]){f_r, f_l, NULL}, *fp; (fp = *ff) != NULL; ff++) {
            TestEnv* tt = (fp == f_r) ? &t_r : &t_l;
            fprintf(fp, "{\"step\":%d", s);
            fprintf(fp, ",\"obs\":[");
            for (int i = 0; i < N_OBS; i++)
                fprintf(fp, "%s%.6g", i ? "," : "", tt->observations[i]);
            fprintf(fp, "],\"action\":[");
            for (int i = 0; i < 5; i++)
                fprintf(fp, "%s%.6g", i ? "," : "", tt->env.actions[i]);
            fprintf(fp, "],\"player_pos\":[%.3f,%.3f,%.3f]",
                    tt->env.player.pos.x, tt->env.player.pos.y, tt->env.player.pos.z);
            fprintf(fp, ",\"opp_pos\":[%.3f,%.3f,%.3f]",
                    tt->env.opponent.pos.x, tt->env.opponent.pos.y, tt->env.opponent.pos.z);
            fprintf(fp, "}\n");
        }
    }
    fclose(f_r); fclose(f_l);

    int n_broken = 0;
    printf("T5 mirror-pair (200 steps, neutral action) per-slot asymmetry:\n");
    printf("  Slot  Label        mean_|obs_R - sign*obs_L|   tag\n");
    for (int i = 0; i < N_OBS; i++) {
        float mean = (float)(sum_asym[i] / n);
        const char* tag = "[SYM]";
        if (mean > 1e-3f) { tag = "[BROKEN]"; n_broken++; }
        printf("  [%2d]  %-12s   %12.4e   sign=%+d %s\n",
               i, SLOT_LABEL[i], mean, MIRROR_SIGN[i], tag);
    }
    printf("T5: %d/%d slots BROKEN under y-mirror (>1e-3)\n", n_broken, N_OBS);
    printf("    traces -> /tmp/df_symmetry_{right,left}.jsonl\n");
    return 0;  /* soft */
}

int main(void) {
    int fails = 0;
    fails += test_aileron_right();
    fails += test_aileron_left();
    fails += test_obs_az_right();
    fails += test_obs_az_left();
    fails += test_mirror_pair();
    printf("\n%d hard failures\n", fails);
    return fails;
}
