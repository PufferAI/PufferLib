/*
 * test_autoace.c - 1:1 port of test_autoace.py.
 *
 * 13 behavioural tests for the AutoAce opponent (curriculum stage 20).
 * Each test sets up a specific geometric scenario, runs N ticks while the
 * player holds neutral controls, and inspects AutoAce's chosen actions
 * (env->last_opp_actions) and tactical state (env->opponent_ace).
 *
 * All tests are soft — they print metrics + [OK]/[FAIL] but never abort.
 */
#include "test_common.h"
#include "../autopilot.h"
#include "../autoace.h"

#define DEG (3.14159265f / 180.0f)
#define RAD (180.0f / 3.14159265f)

/* Spin up an env with curriculum enabled and stage forced to AUTOACE. */
static void setup_autoace_env(TestEnv* t) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 3000;
    t->env.rng = 42;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;
    RewardConfig rcfg = test_default_rcfg();
    init(&t->env, /*obs_scheme=*/0, &rcfg, /*curriculum_enabled=*/1, 0, 0);
    /* Force stage to AUTOACE so autoace_step() runs in c_step(). */
    t->env.stage = CURRICULUM_AUTOACE;
    c_reset(&t->env);
    /* spawn_by_curriculum overwrote stage based on STAGES table; reset it. */
    t->env.stage = CURRICULUM_AUTOACE;
    /* Must be != AP_STRAIGHT for opponent control branch to fire. */
    autopilot_set_mode(&t->env.opponent_ap, AP_LEVEL,
                       AP_DEFAULT_THROTTLE, 0.0f, 0.0f);
}

/* Place player ("target" in Python) and opponent ("autoace") with explicit
 * positions, velocities, and orientations. */
static void place(Dogfight* env,
                  float tpx, float tpy, float tpz,
                  float tvx, float tvy, float tvz,
                  float tow, float tox, float toy, float toz,
                  float opx, float opy, float opz,
                  float ovx, float ovy, float ovz,
                  float oow, float oox, float ooy, float ooz,
                  int target_cooldown, int autoace_cooldown) {
    force_state(env,
        tpx, tpy, tpz, tvx, tvy, tvz,
        tow, tox, toy, toz, /*throttle=*/1.0f,
        opx, opy, opz, ovx, ovy, ovz,
        oow, oox, ooy, ooz,
        0, target_cooldown, autoace_cooldown);
    /* Stage gets reset by force_state's compute_observations chain... no it
     * doesn't.  stage stays AUTOACE. */
    /* Re-arm autopilot mode (force_state's PID reset doesn't change mode). */
    if (env->opponent_ap.mode == AP_STRAIGHT) {
        autopilot_set_mode(&env->opponent_ap, AP_LEVEL,
                           AP_DEFAULT_THROTTLE, 0.0f, 0.0f);
    }
}

/* Convenience for default identity orientation on both planes. */
static void place_simple(Dogfight* env,
                          float tpx, float tpy, float tpz,
                          float tvx, float tvy, float tvz,
                          float opx, float opy, float opz,
                          float ovx, float ovy, float ovz) {
    place(env,
        tpx, tpy, tpz, tvx, tvy, tvz, 1, 0, 0, 0,
        opx, opy, opz, ovx, ovy, ovz, 1, 0, 0, 0,
        0, 0);
}

/* Player-side neutral controls (throttle 0.5, no fire). */
static void player_neutral_action(Dogfight* env) {
    float a[5] = {0.5f, 0.0f, 0.0f, 0.0f, -1.0f};
    memcpy(env->actions, a, sizeof(a));
}

static float opp_pitch_deg(const Plane* p) {
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float horiz = sqrtf(fwd.x * fwd.x + fwd.y * fwd.y);
    return atan2f(fwd.z, horiz) * RAD;
}
static float opp_heading_deg(const Plane* p) {
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    return atan2f(fwd.y, fwd.x) * RAD;
}
static float opp_bank_rad(const Plane* p) {
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float bank = acosf(clip_unit(up.z));
    return (up.y < 0.0f) ? bank : -bank;
}
static float wrap_pi(float a) {
    while (a >  3.14159265f) a -= 2.0f * 3.14159265f;
    while (a < -3.14159265f) a += 2.0f * 3.14159265f;
    return a;
}

/* ============================================================
 * Tests (order matches test_autoace.py TESTS dict)
 * ============================================================ */

static int test_target_above_ahead(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        346, 0, 1200, 80, 0, 0,
        0, 0, 1000, 100, 0, 0);
    float pitch0 = opp_pitch_deg(&t.env.opponent);
    float elev_sum = 0.0f;
    int steps = 50;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        elev_sum += t.env.last_opp_actions[1];
    }
    float pitch1 = opp_pitch_deg(&t.env.opponent);
    float dpitch = (pitch1 - pitch0) * DEG;
    float avg_elev = elev_sum / steps;
    int elev_ok = avg_elev < -0.1f;
    int pitch_ok = dpitch > 0.05f;
    int ok = elev_ok && pitch_ok;
    printf("target_above_ahead: elev_avg=%.3f dpitch=%.1fdeg [%s]\n",
           avg_elev, dpitch * RAD, ok ? "OK" : "FAIL");
    return 0;
}

static int test_target_below_ahead(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        400, 0, 600, 80, 0, 0,
        0, 0, 1000, 100, 0, 0);
    float pitch0 = opp_pitch_deg(&t.env.opponent);
    float elev_sum = 0.0f;
    int steps = 50;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        elev_sum += t.env.last_opp_actions[1];
    }
    float pitch1 = opp_pitch_deg(&t.env.opponent);
    float dpitch = (pitch1 - pitch0) * DEG;
    float avg_elev = elev_sum / steps;
    int elev_ok = avg_elev > 0.1f;
    int pitch_ok = dpitch < -0.05f;
    int ok = elev_ok && pitch_ok;
    printf("target_below_ahead: elev_avg=%.3f dpitch=%.1fdeg [%s]\n",
           avg_elev, dpitch * RAD, ok ? "OK" : "FAIL");
    return 0;
}

static int test_target_level_ahead(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        400, 0, 1000, 80, 0, 0,
        0, 0, 1000, 100, 0, 0);
    float pitch0 = opp_pitch_deg(&t.env.opponent);
    float elev_sum = 0.0f;
    int steps = 50;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        elev_sum += t.env.last_opp_actions[1];
    }
    float pitch1 = opp_pitch_deg(&t.env.opponent);
    float dpitch = (pitch1 - pitch0) * DEG;
    float avg_elev = elev_sum / steps;
    int ok = (fabsf(avg_elev) < 0.3f) && (fabsf(dpitch) < 0.2f);
    printf("target_level_ahead: elev_avg=%.3f dpitch=%.1fdeg [%s]\n",
           avg_elev, dpitch * RAD, ok ? "OK" : "FAIL");
    return 0;
}

static int test_target_left(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        283, 283, 1000, 80, 0, 0,
        0, 0, 1000, 100, 0, 0);
    float h0 = opp_heading_deg(&t.env.opponent) * DEG;
    float ail_sum = 0.0f;
    int steps = 50;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        ail_sum += t.env.last_opp_actions[2];
    }
    float h1 = opp_heading_deg(&t.env.opponent) * DEG;
    float dh = wrap_pi(h1 - h0);
    float avg_ail = ail_sum / steps;
    int ail_ok = avg_ail < -0.1f;
    int h_ok = dh > 0.05f;
    int ok = ail_ok && h_ok;
    printf("target_left:        ail_avg=%.3f dheading=%.1fdeg [%s]\n",
           avg_ail, dh * RAD, ok ? "OK" : "FAIL");
    return 0;
}

static int test_target_right(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        283, -283, 1000, 80, 0, 0,
        0, 0, 1000, 100, 0, 0);
    float h0 = opp_heading_deg(&t.env.opponent) * DEG;
    float ail_sum = 0.0f;
    int steps = 50;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        ail_sum += t.env.last_opp_actions[2];
    }
    float h1 = opp_heading_deg(&t.env.opponent) * DEG;
    float dh = wrap_pi(h1 - h0);
    float avg_ail = ail_sum / steps;
    int ail_ok = avg_ail > 0.1f;
    int h_ok = dh < -0.05f;
    int ok = ail_ok && h_ok;
    printf("target_right:       ail_avg=%.3f dheading=%.1fdeg [%s]\n",
           avg_ail, dh * RAD, ok ? "OK" : "FAIL");
    return 0;
}

static int test_break_left_from_threat(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        -300, 200, 1000, 120, 0, 0,
        0, 0, 1000, 100, 0, 0);
    float h0 = opp_heading_deg(&t.env.opponent) * DEG;
    float min_bank = 0.0f;
    int steps = 100;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        float b = opp_bank_rad(&t.env.opponent);
        if (b < min_bank) min_bank = b;
    }
    float h1 = opp_heading_deg(&t.env.opponent) * DEG;
    float dh = wrap_pi(h1 - h0);
    int bank_ok = min_bank < -0.2f;
    int h_ok = dh > 0.1f;
    int ok = bank_ok && h_ok;
    printf("break_left_threat:  min_bank=%.1fdeg dheading=%.1fdeg [%s]\n",
           min_bank * RAD, dh * RAD, ok ? "OK" : "FAIL");
    return 0;
}

static int test_break_right_from_threat(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        -300, -200, 1000, 120, 0, 0,
        0, 0, 1000, 100, 0, 0);
    float h0 = opp_heading_deg(&t.env.opponent) * DEG;
    float max_bank = 0.0f;
    int steps = 100;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        float b = opp_bank_rad(&t.env.opponent);
        if (b > max_bank) max_bank = b;
    }
    float h1 = opp_heading_deg(&t.env.opponent) * DEG;
    float dh = wrap_pi(h1 - h0);
    int bank_ok = max_bank > 0.2f;
    int h_ok = dh < -0.1f;
    int ok = bank_ok && h_ok;
    printf("break_right_threat: max_bank=%.1fdeg dheading=%.1fdeg [%s]\n",
           max_bank * RAD, dh * RAD, ok ? "OK" : "FAIL");
    return 0;
}

static int test_target_behind(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        -300, 50, 1000, 120, 0, 0,
        0, 0, 1000, 100, 0, 0);
    int steps = 30;
    int defensive = 0;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        if (t.env.opponent_ace.engagement == ENGAGE_DEFENSIVE) defensive++;
    }
    float ratio = (float)defensive / steps;
    int ok = ratio > 0.5f;
    printf("target_behind:      defensive=%d/%d (%.2f) [%s]\n",
           defensive, steps, ratio, ok ? "OK" : "FAIL");
    return 0;
}

static int test_engage_offensive(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        400, 0, 1000, 80, 0, 0,
        0, 0, 1000, 100, 0, 0);
    int steps = 30;
    int offensive = 0;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        if (t.env.opponent_ace.engagement == ENGAGE_OFFENSIVE) offensive++;
    }
    float ratio = (float)offensive / steps;
    int ok = ratio > 0.5f;
    printf("engage_offensive:   offensive=%d/%d (%.2f) [%s]\n",
           offensive, steps, ratio, ok ? "OK" : "FAIL");
    return 0;
}

static int test_fires_in_envelope(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        400, 10, 1000, 90, 0, 0,
        0, 0, 1000, 100, 0, 0);
    int steps = 300;
    int trigger_pulls = 0, in_env = 0, weapons_eng = 0;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        if (t.env.terminals[0]) {
            if (t.env.rewards[0] < 0.0f) trigger_pulls++;
            break;
        }
        if (t.env.opponent_ace.tactical.in_gun_envelope) in_env++;
        if (t.env.opponent_ace.engagement == ENGAGE_WEAPONS) weapons_eng++;
        if (t.env.last_opp_actions[4] > 0.5f) trigger_pulls++;
    }
    int ok = (trigger_pulls >= 1) || (weapons_eng >= 1);
    printf("fires_in_envelope:  triggers=%d weapons=%d in_env=%d [%s]\n",
           trigger_pulls, weapons_eng, in_env, ok ? "OK" : "FAIL");
    return 0;
}

static int test_fires_after_turn_right(void) {
    TestEnv t; setup_autoace_env(&t);
    place_simple(&t.env,
        386, -103, 1000, 90, 0, 0,
        0, 0, 1000, 100, 0, 0);
    int steps = 500;
    int trigger_pulls = 0;
    int killed = 0;
    float max_bank = 0.0f;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        if (t.env.terminals[0]) {
            if (t.env.rewards[0] < 0.0f) { killed = 1; trigger_pulls++; }
            break;
        }
        float b = opp_bank_rad(&t.env.opponent);
        if (b > max_bank) max_bank = b;
        if (t.env.last_opp_actions[4] > 0.5f) trigger_pulls++;
    }
    int banked = max_bank > 0.1f;
    int fired = (trigger_pulls >= 1) || killed;
    int ok = banked && fired;
    printf("fires_after_turn_R: max_bank=%.1fdeg triggers=%d killed=%d [%s]\n",
           max_bank * RAD, trigger_pulls, killed, ok ? "OK" : "FAIL");
    return 0;
}

static int test_no_fire_when_off_target(void) {
    TestEnv t; setup_autoace_env(&t);
    /* Target 90 deg to side, perpendicular velocity. */
    place_simple(&t.env,
        0, 300, 1000, 0, 80, 0,
        0, 0, 1000, 100, 0, 0);
    int steps = 20;
    int trigger_pulls = 0;
    for (int s = 0; s < steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        if (t.env.last_opp_actions[4] > 0.5f) trigger_pulls++;
    }
    int ok = trigger_pulls <= 1;
    printf("no_fire_off_target: triggers=%d/%d [%s]\n",
           trigger_pulls, steps, ok ? "OK" : "FAIL");
    return 0;
}

static int test_head_to_head_dogfight(void) {
    TestEnv t; setup_autoace_env(&t);
    /* Target facing -X (yaw 180): quat (0, 0, 0, 1). */
    place(&t.env,
        500, 0, 2000, -110, 0, 0, 0, 0, 0, 1,
        -500, 0, 2000, 110, 0, 0, 1, 0, 0, 0,
        300, 300);
    int max_steps = 1000;
    int kill = 0, pass_detected = 0;
    int shots_after = 0;
    for (int s = 0; s < max_steps; s++) {
        player_neutral_action(&t.env);
        c_step(&t.env);
        if (t.env.terminals[0]) {
            if (t.env.rewards[0] != 0.0f) kill = 1;
            break;
        }
        if (s == 230) {
            if (t.env.opponent_ace.tactical.range < 200.0f) pass_detected = 1;
        }
        if (s > 300 && t.env.last_opp_actions[4] > 0.5f) shots_after++;
    }
    int ok = kill || pass_detected;
    printf("head_to_head:       kill=%d pass=%d shots=%d [%s]\n",
           kill, pass_detected, shots_after, ok ? "OK" : "FAIL");
    return 0;
}

int main(void) {
    int fails = 0;
    fails += test_target_above_ahead();
    fails += test_target_below_ahead();
    fails += test_target_level_ahead();
    fails += test_target_left();
    fails += test_target_right();
    fails += test_break_left_from_threat();
    fails += test_break_right_from_threat();
    fails += test_target_behind();
    fails += test_engage_offensive();
    fails += test_fires_in_envelope();
    fails += test_fires_after_turn_right();
    fails += test_no_fire_when_off_target();
    fails += test_head_to_head_dogfight();
    printf("\n%d hard failures\n", fails);
    return fails;
}
