/*
 * test_opponent_override.c - 1:1 port of test_opponent_override.py.
 *
 * Tests external opponent action override (used by self-play in 3.0 via
 * binding.vec_enable_opponent_override / vec_set_opponent_actions).
 *
 * 4.0 has no equivalent Python bindings yet, so we manipulate the C struct
 * fields directly:
 *   env.use_opponent_override          (0/1)
 *   env.opponent_actions_override[5]   (throttle, elevator, aileron, rudder, trigger)
 *   env.last_opp_actions[5]            (read after step to confirm)
 *
 * The override branch in c_step (dogfight.h:1038) fires when
 * use_opponent_override == 1, AFTER recovery and BEFORE the normal autopilot
 * branch — so it does NOT require curriculum stage or an autopilot mode.
 *
 * Soft tests print [OK]/[FAIL]; hard test (close-range kill) returns 1 on miss.
 *
 * 7 tests, 1 SKIP (test_shape_validation is N/A — Python numpy validation).
 */
#include "test_common.h"
#include "../autopilot.h"

#define DEG (3.14159265f / 180.0f)

/* Place opponent behind player along +X, both flying +X at 80 m/s. */
static void place_tail_chase(Dogfight* env, float opp_dx_behind) {
    force_state(env,
        /* player */ 0.0f, 0.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.5f,
        /* opponent (offset along -X) */
        -opp_dx_behind, 0.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        /* tick=0, both cooldowns=0 */ 0, 0, 0);
}

/* Player neutral (no fire). Returns through env->actions. */
static void player_neutral(Dogfight* env) {
    float a[5] = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f};
    memcpy(env->actions, a, sizeof(a));
}

/* ---- 1. test_override_enable -------------------------------------------- */
static int test_override_enable(void) {
    printf("Testing opponent override enable/disable...\n");
    TestEnv t;
    setup_env(&t, 0);

    t.env.use_opponent_override = 1;
    int ok_on = (t.env.use_opponent_override == 1);
    t.env.use_opponent_override = 0;
    int ok_off = (t.env.use_opponent_override == 0);

    int pass = ok_on && ok_off;
    printf("  use_opponent_override toggle: on=%d off=%d [%s]\n",
           ok_on, ok_off, pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* ---- 2. test_set_opponent_actions --------------------------------------- */
static int test_set_opponent_actions(void) {
    printf("\nTesting set opponent actions...\n");
    TestEnv t;
    setup_env(&t, 0);

    float vals[5] = {0.5f, -0.3f, 0.2f, 0.1f, 1.0f};
    for (int i = 0; i < 5; i++) t.env.opponent_actions_override[i] = vals[i];

    int pass = 1;
    for (int i = 0; i < 5; i++) {
        if (fabsf(t.env.opponent_actions_override[i] - vals[i]) > 1e-6f) {
            pass = 0;
            printf("  slot %d: stored %.3f != expected %.3f\n",
                   i, t.env.opponent_actions_override[i], vals[i]);
        }
    }
    printf("  opponent_actions_override stores values verbatim [%s]\n",
           pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* ---- 3. test_opponent_responds_to_override ------------------------------ */
static int test_opponent_responds_to_override(void) {
    printf("\nTesting opponent responds to override...\n");
    TestEnv t;
    setup_env(&t, 0);

    /* Place both planes level at 1000m, opponent ahead. */
    force_state(&t.env,
        0.0f, 0.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.5f,
        400.0f, 0.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0, 0, 0);

    float initial_alt = t.env.opponent.pos.z;

    /* Override: full throttle, pull up (elevator=-0.5, conventional sign). */
    t.env.use_opponent_override = 1;
    t.env.opponent_actions_override[0] = 1.0f;
    t.env.opponent_actions_override[1] = -0.5f;
    t.env.opponent_actions_override[2] = 0.0f;
    t.env.opponent_actions_override[3] = 0.0f;
    t.env.opponent_actions_override[4] = 0.0f;

    player_neutral(&t.env);
    for (int i = 0; i < 100; i++) c_step(&t.env);

    float final_alt = t.env.opponent.pos.z;
    float delta = final_alt - initial_alt;
    int pass = (delta > 1.0f);  /* gained >1m altitude */
    printf("  initial_alt=%.2f final_alt=%.2f delta=%.2f [%s]\n",
           initial_alt, final_alt, delta, pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* ---- 4. test_autopilot_resumes ------------------------------------------ */
static int test_autopilot_resumes(void) {
    printf("\nTesting autopilot resumes after override disabled...\n");
    TestEnv t;
    setup_env(&t, 0);

    /* Need autopilot_ap.mode != AP_STRAIGHT for the autopilot branch in
     * c_step to run after override is disabled. AP_LEVEL just holds level. */
    autopilot_set_mode(&t.env.opponent_ap, AP_LEVEL,
                       AP_DEFAULT_THROTTLE, 0.0f, 0.0f);

    force_state(&t.env,
        0.0f, 0.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f, 0.5f,
        400.0f, 0.0f, 1000.0f, 80.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0, 0, 0);

    player_neutral(&t.env);

    /* Phase A: autopilot only. */
    for (int i = 0; i < 50; i++) c_step(&t.env);
    float alt_a = t.env.opponent.pos.z;

    /* Phase B: enable override, pull-up. */
    t.env.use_opponent_override = 1;
    t.env.opponent_actions_override[0] = 1.0f;
    t.env.opponent_actions_override[1] = -0.5f;
    t.env.opponent_actions_override[2] = 0.0f;
    t.env.opponent_actions_override[3] = 0.0f;
    t.env.opponent_actions_override[4] = 0.0f;
    for (int i = 0; i < 50; i++) c_step(&t.env);
    float alt_b = t.env.opponent.pos.z;
    /* During override, last_opp_actions must match override slot-for-slot. */
    int override_applied = 1;
    for (int i = 0; i < 5; i++) {
        if (fabsf(t.env.last_opp_actions[i] - t.env.opponent_actions_override[i]) > 1e-5f) {
            override_applied = 0;
        }
    }

    /* Phase C: disable override; autopilot resumes (mode=AP_LEVEL). */
    t.env.use_opponent_override = 0;
    /* Re-arm — c_reset may have fired during phase B if the player hit ground. */
    if (t.env.opponent_ap.mode == AP_STRAIGHT) {
        autopilot_set_mode(&t.env.opponent_ap, AP_LEVEL,
                           AP_DEFAULT_THROTTLE, 0.0f, 0.0f);
    }
    for (int i = 0; i < 50; i++) c_step(&t.env);
    float alt_c = t.env.opponent.pos.z;

    /* After override is off, last_opp_actions should reflect autopilot output,
     * not the (still-set) override values — at least one slot should differ. */
    int autopilot_active = 0;
    for (int i = 0; i < 5; i++) {
        if (fabsf(t.env.last_opp_actions[i] - t.env.opponent_actions_override[i]) > 1e-3f) {
            autopilot_active = 1;
            break;
        }
    }

    int pass = override_applied && autopilot_active;
    printf("  alt_a(autopilot)=%.2f alt_b(override)=%.2f alt_c(resumed)=%.2f\n",
           alt_a, alt_b, alt_c);
    printf("  override_applied=%d autopilot_active_after=%d [%s]\n",
           override_applied, autopilot_active, pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

/* ---- 5. test_two_way_combat --------------------------------------------- */
static int test_two_way_combat(void) {
    printf("\nTesting two-way combat (opponent shoots player)...\n");
    TestEnv t;
    setup_env(&t, 0);
    place_tail_chase(&t.env, 200.0f);  /* opponent 200m behind, aligned */

    t.env.use_opponent_override = 1;
    t.env.opponent_actions_override[0] = 1.0f;  /* full throttle */
    t.env.opponent_actions_override[1] = 0.0f;
    t.env.opponent_actions_override[2] = 0.0f;
    t.env.opponent_actions_override[3] = 0.0f;
    t.env.opponent_actions_override[4] = 1.0f;  /* fire */

    /* head_on_lockout is set from spawn type by c_reset; force_state did not
     * touch it, but our setup_env's c_reset path set it to 0 since we are
     * not in a head-on stage. Belt-and-suspenders: clear it. */
    t.env.head_on_lockout = 0;

    int shot = 0;
    int step_killed = -1;
    for (int s = 0; s < 500; s++) {
        player_neutral(&t.env);
        c_step(&t.env);
        if (t.env.terminals[0]) {
            if (t.env.opp_kill == 1 || t.env.death_reason == DEATH_KILL ||
                t.env.rewards[0] == -1.0f) {
                shot = 1;
                step_killed = s;
            }
            break;
        }
    }
    /* Mirrors Python: hit-or-not is informational; the path executing without
     * crash is what we are validating. Still print outcome. */
    if (shot) {
        printf("  player shot down at step %d [OK]\n", step_killed);
    } else {
        printf("  no kill in 500 steps (moving target — informational) [OK]\n");
    }
    return 0;  /* soft */
}

/* ---- 6. test_two_way_combat_close_range --------------------------------- */
static int test_two_way_combat_close_range(void) {
    printf("\nTesting two-way combat (close range)...\n");
    TestEnv t;
    setup_env(&t, 0);
    place_tail_chase(&t.env, 50.0f);  /* opponent 50m behind, dead-on aim */
    t.env.head_on_lockout = 0;

    t.env.use_opponent_override = 1;
    t.env.opponent_actions_override[0] = 0.5f;
    t.env.opponent_actions_override[1] = 0.0f;
    t.env.opponent_actions_override[2] = 0.0f;
    t.env.opponent_actions_override[3] = 0.0f;
    t.env.opponent_actions_override[4] = 1.0f;  /* fire */

    player_neutral(&t.env);
    c_step(&t.env);

    int term = (t.env.terminals[0] > 0.5f);
    int killed = (term && t.env.rewards[0] == -1.0f);
    printf("  term=%d rew=%.2f opp_kill=%d [%s]\n",
           term, t.env.rewards[0], t.env.opp_kill,
           killed ? "OK" : "FAIL");
    return killed ? 0 : 1;
}

/* ---- 7. test_shape_validation (SKIP) ------------------------------------ */
static int test_shape_validation(void) {
    printf("\nTesting shape validation...\n");
    printf("  Python numpy dtype/shape validation has no C equivalent [SKIP]\n");
    return 0;
}

int main(void) {
    int fails = 0;
    fails += test_override_enable();
    fails += test_set_opponent_actions();
    fails += test_opponent_responds_to_override();
    fails += test_autopilot_resumes();
    fails += test_two_way_combat();
    fails += test_two_way_combat_close_range();
    fails += test_shape_validation();
    printf("\n%d hard failures\n", fails);
    return fails;
}
