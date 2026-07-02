/*
 * test_flight_obs_static.c - 1:1 port of test_flight_obs_static.py.
 *
 * 5 tests covering observation dimensions, target azimuth/elevation,
 * edge cases, [-1,1] bounds under random states, and altitude/energy
 * clamping at extreme altitudes for both obs schemes.
 *
 * All tests are soft.
 */
#include "test_common.h"

#define OBS_ATOL 0.05f
#define OBS_RTOL 0.10f

static int close_to(float actual, float expected, float atol, float rtol) {
    float tol = atol + rtol * fabsf(expected);
    return fabsf(actual - expected) <= tol;
}

/* Set both player and opponent positions/orientations explicitly. */
static void place(Dogfight* env,
                  float ppx, float ppy, float ppz,
                  float pvx, float pvy, float pvz,
                  float pow_, float pox, float poy, float poz,
                  float opx, float opy, float opz,
                  float ovx, float ovy, float ovz) {
    force_state(env,
        ppx, ppy, ppz, pvx, pvy, pvz,
        pow_, pox, poy, poz, /*throttle=*/0.5f,
        opx, opy, opz, ovx, ovy, ovz,
        1.0f, 0.0f, 0.0f, 0.0f, 0, -1, -1);
}

/* Setup with caller-controlled obs scheme, return 22-or-26 obs buffer
 * via the wired-in observations buffer. */
static void setup_with_scheme(TestEnv* t, int scheme) {
    setup_env(t, scheme);
}

static int test_obs_scheme_dimensions(void) {
    int expected_sizes[2] = {22, 26};
    int all_ok = 1;
    for (int scheme = 0; scheme < 2; scheme++) {
        TestEnv t; setup_with_scheme(&t, scheme);
        int actual = t.env.obs_size;
        int passed = (actual == expected_sizes[scheme]);
        if (!passed) all_ok = 0;
        printf("obs_dim_%d:          %d obs (expected %d) [%s]\n",
               scheme, actual, expected_sizes[scheme], passed ? "OK" : "FAIL");
    }
    (void)all_ok;
    return 0;
}

static int test_obs_target_angles(void) {
    /* Target to the right (negative Y). */
    TestEnv t; setup_with_scheme(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        0, -400, 1000, 100, 0, 0);
    /* force_state already calls compute_observations. */
    float az_right = t.env.observations[13];

    setup_with_scheme(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        0, 0, 1400, 100, 0, 0);
    float elev_above = t.env.observations[14];

    int p1 = close_to(az_right, -0.5f, OBS_ATOL, OBS_RTOL);
    int p2 = close_to(elev_above, 1.0f, 0.1f, OBS_RTOL);
    int ok = p1 && p2;
    printf("obs_target_angles:  az_right=%.3f, elev_up=%.3f [%s]\n",
           az_right, elev_above, ok ? "OK" : "FAIL");
    return 0;
}

static int test_obs_edge_cases(void) {
    TestEnv t;
    /* Behind-left — opponent at (-400, +10, 1000). */
    setup_with_scheme(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        -400, 10, 1000, 100, 0, 0);
    float az_left = t.env.observations[13];

    /* Behind-right — opponent at (-400, -10, 1000). */
    setup_with_scheme(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        -400, -10, 1000, 100, 0, 0);
    float az_right = t.env.observations[13];

    /* Extreme distance — opponent at (5000, 0, 1000). */
    setup_with_scheme(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        5000, 0, 1000, 100, 0, 0);
    float dist_obs = t.env.observations[15];

    int p1 = az_left  >  0.9f;
    int p2 = az_right < -0.9f;
    int p3 = (dist_obs >= -1.0f) && (dist_obs <= 1.0f);
    int ok = p1 && p2 && p3;
    printf("obs_edge_cases:     az_180=%.2f/%.2f, dist_clamp=%.2f [%s]\n",
           az_left, az_right, dist_obs, ok ? "OK" : "FAIL");
    return 0;
}

/* tiny LCG for repeatable random states */
static unsigned int test_rng = 12345;
static float test_rand_uniform(float lo, float hi) {
    test_rng = test_rng * 1103515245u + 12345u;
    float u = (float)((test_rng >> 16) & 0x7FFF) / 32767.0f;
    return lo + u * (hi - lo);
}
static float test_rand_normal(void) {
    /* Simple sum-of-12 — close enough to gaussian for bounds testing. */
    float s = 0.0f;
    for (int i = 0; i < 12; i++) s += test_rand_uniform(0.0f, 1.0f);
    return s - 6.0f;
}

static int test_obs_bounds(void) {
    int passed = 1;
    int violations = 0;
    for (int trial = 0; trial < 30; trial++) {
        TestEnv t; setup_with_scheme(&t, 0);
        float px = test_rand_uniform(-4000, 4000);
        float py = test_rand_uniform(-4000, 4000);
        float pz = test_rand_uniform(100, 2900);
        float vx = test_rand_normal() * 100.0f;
        float vy = test_rand_normal() * 100.0f;
        float vz = test_rand_normal() * 100.0f;
        float ow = test_rand_normal();
        float ox = test_rand_normal();
        float oy = test_rand_normal();
        float oz = test_rand_normal();
        float n = sqrtf(ow*ow + ox*ox + oy*oy + oz*oz);
        if (n < 1e-6f) { ow = 1; ox = oy = oz = 0; n = 1; }
        ow /= n; ox /= n; oy /= n; oz /= n;
        if (ow < 0) { ow = -ow; ox = -ox; oy = -oy; oz = -oz; }
        float opx = px + test_rand_uniform(-500, 500);
        float opy = py + test_rand_uniform(-500, 500);
        float opz = pz + test_rand_uniform(-500, 500);
        place(&t.env, px, py, pz, vx, vy, vz, ow, ox, oy, oz,
              opx, opy, opz, 100, 0, 0);
        for (int i = 0; i < t.env.obs_size; i++) {
            float v = t.env.observations[i];
            if (v < -1.0f || v > 1.0f) { passed = 0; violations++; }
        }
    }
    printf("obs_bounds:         30 random states, all in [-1.0, 1.0] [%s] (%d violations)\n",
           passed ? "OK" : "FAIL", violations);
    return 0;
}

static int test_obs_altitude_energy_clamping(void) {
    int test_alts[5] = {0, 50, 100, 2500, 4999};
    const int alt_idx = 7;
    const int energy_idx = 9;
    int passed = 1;
    int n_violations = 0;
    for (int scheme = 0; scheme < 2; scheme++) {
        TestEnv t;
        for (int ai = 0; ai < 5; ai++) {
            int alt = test_alts[ai];
            float opp_alt = (alt > 100) ? (float)alt : 100.0f;
            setup_with_scheme(&t, scheme);
            place(&t.env,
                0, 0, (float)alt, 100, 0, 0, 1, 0, 0, 0,
                500, 0, opp_alt, 80, 0, 0);
            float alt_obs = t.env.observations[alt_idx];
            float e_obs = t.env.observations[energy_idx];
            if (alt_obs < -0.001f || alt_obs > 1.001f) { passed = 0; n_violations++; }
            if (e_obs < -0.001f || e_obs > 1.001f) { passed = 0; n_violations++; }
        }
        /* Opponent low altitude — verify all obs in [-1, 1]. */
        setup_with_scheme(&t, scheme);
        place(&t.env,
            0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
            500, 0, 10, 80, 0, 0);
        for (int i = 0; i < t.env.obs_size; i++) {
            float v = t.env.observations[i];
            if (v < -1.001f || v > 1.001f) { passed = 0; n_violations++; }
        }
    }
    printf("obs_alt_energy:     5 altitudes x 2 schemes, clamped to [0,1] [%s] (%d viol)\n",
           passed ? "OK" : "FAIL", n_violations);
    return 0;
}

int main(void) {
    int fails = 0;
    fails += test_obs_scheme_dimensions();
    fails += test_obs_target_angles();
    fails += test_obs_edge_cases();
    fails += test_obs_bounds();
    fails += test_obs_altitude_energy_clamping();
    printf("\n%d hard failures\n", fails);
    return fails;
}
