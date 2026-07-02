/*
 * test_flight_autoace.c - 1:1 port of test_flight_autoace.py.
 *
 * 5 high-level smoke tests verifying that AutoAce (curriculum stage 20)
 * runs without errors over multiple episodes and various player inputs.
 *
 * All tests are soft.
 */
#include "test_common.h"
#include "../autopilot.h"
#include "../autoace.h"

#define DEG (3.14159265f / 180.0f)
#define RAD (180.0f / 3.14159265f)

/* Spin up an env at curriculum stage AUTOACE.  Mirrors make_autoace_env(). */
static void setup_stage20_env(TestEnv* t) {
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
    t->env.stage = CURRICULUM_AUTOACE;
    c_reset(&t->env);
    t->env.stage = CURRICULUM_AUTOACE;
    autopilot_set_mode(&t->env.opponent_ap, AP_LEVEL,
                       AP_DEFAULT_THROTTLE, 0.0f, 0.0f);
}

static int test_autoace_stage_20(void) {
    TestEnv t; setup_stage20_env(&t);
    int stage = t.env.stage;
    int passed = (stage == CURRICULUM_AUTOACE);
    printf("autoace_stage:      stage=%d [%s]\n", stage, passed ? "OK" : "FAIL");
    return 0;
}

static int test_autoace_pursues(void) {
    TestEnv t; setup_stage20_env(&t);
    int steps_completed = 0;
    int steps = 250;  /* 5 seconds at 50Hz */
    for (int s = 0; s < steps; s++) {
        float a[5] = {0.5f, 0.0f, 0.0f, 0.0f, -1.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        steps_completed++;
        if (t.env.terminals[0]) break;
    }
    int passed = steps_completed > 10;
    printf("autoace_pursues:    ran %d steps at stage 20 [%s]\n",
           steps_completed, passed ? "OK" : "FAIL");
    return 0;
}

static int test_autoace_defends(void) {
    TestEnv t; setup_stage20_env(&t);
    int episodes_completed = 0;
    int total_steps = 0;
    for (int ep = 0; ep < 5; ep++) {
        c_reset(&t.env);
        t.env.stage = CURRICULUM_AUTOACE;
        autopilot_set_mode(&t.env.opponent_ap, AP_LEVEL,
                           AP_DEFAULT_THROTTLE, 0.0f, 0.0f);
        int steps = 150;  /* 3 seconds */
        for (int s = 0; s < steps; s++) {
            float a[5] = {1.0f, -0.1f, 0.2f, 0.0f, -1.0f};
            memcpy(t.env.actions, a, sizeof(a));
            c_step(&t.env);
            total_steps++;
            if (t.env.terminals[0]) break;
        }
        episodes_completed++;
    }
    int passed = (episodes_completed == 5) && (total_steps > 100);
    printf("autoace_defends:    %d episodes, %d total steps [%s]\n",
           episodes_completed, total_steps, passed ? "OK" : "FAIL");
    return 0;
}

static int test_autoace_fires(void) {
    TestEnv t; setup_stage20_env(&t);
    int player_deaths = 0;
    int episodes = 10;
    for (int ep = 0; ep < episodes; ep++) {
        c_reset(&t.env);
        t.env.stage = CURRICULUM_AUTOACE;
        autopilot_set_mode(&t.env.opponent_ap, AP_LEVEL,
                           AP_DEFAULT_THROTTLE, 0.0f, 0.0f);
        int steps = 500;  /* 10 seconds */
        for (int s = 0; s < steps; s++) {
            float a[5] = {0.5f, -0.2f, 0.5f, 0.0f, -1.0f};
            memcpy(t.env.actions, a, sizeof(a));
            c_step(&t.env);
            if (t.env.terminals[0]) {
                if (s < steps - 1) player_deaths++;
                break;
            }
        }
    }
    /* Same as Python — soft check, any result is OK. */
    printf("autoace_fires:      %d/%d early terminations [OK]\n",
           player_deaths, episodes);
    return 0;
}

static int test_autoace_energy(void) {
    TestEnv t; setup_stage20_env(&t);
    int steps = 150;
    int ran = 0;
    for (int s = 0; s < steps; s++) {
        float a[5] = {0.5f, 0.0f, 0.0f, 0.0f, -1.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        ran++;
        if (t.env.terminals[0]) break;
    }
    (void)ran;
    printf("autoace_energy:     stage 20 runs successfully [OK]\n");
    return 0;
}

int main(void) {
    int fails = 0;
    fails += test_autoace_stage_20();
    fails += test_autoace_pursues();
    fails += test_autoace_defends();
    fails += test_autoace_fires();
    fails += test_autoace_energy();
    printf("\n%d hard failures\n", fails);
    return fails;
}
