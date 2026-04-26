/*
 * test_flight_obs_dynamic.c - 1:1 port of test_flight_obs_dynamic.py.
 *
 * 4 tests covering azimuth +/-180 wrap, elevation extremes,
 * obs continuity through complex maneuvers, and quaternion normalization
 * drift over extended flight.
 *
 * All tests are soft.
 */
#include "test_common.h"

static void place(Dogfight* env,
                  float ppx, float ppy, float ppz,
                  float pvx, float pvy, float pvz,
                  float pow_, float pox, float poy, float poz,
                  float opx, float opy, float opz,
                  float ovx, float ovy, float ovz,
                  float throttle) {
    force_state(env,
        ppx, ppy, ppz, pvx, pvy, pvz,
        pow_, pox, poy, poz, throttle,
        opx, opy, opz, ovx, ovy, ovz,
        1.0f, 0.0f, 0.0f, 0.0f, 0, -1, -1);
}

static int test_obs_azimuth_crossover(void) {
    float azimuths[50]; int n = 0;
    for (int step = 0; step < 50; step++) {
        TestEnv t; setup_env(&t, 0);
        float y_offset = -200.0f + step * 8.0f;
        place(&t.env,
            0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
            -200, y_offset, 1000, 100, 0, 0,
            0.5f);
        azimuths[n++] = t.env.observations[13];
    }
    int jumps = 0;
    for (int i = 1; i < n; i++) {
        if (fabsf(azimuths[i] - azimuths[i - 1]) > 0.5f) jumps++;
    }
    float az_min = arr_min(azimuths, n);
    float az_max = arr_max(azimuths, n);
    int range_ok = (az_max > 0.8f) && (az_min < -0.8f);
    printf("obs_az_cross:       range=[%.2f,%.2f], discontinuities=%d [%s]\n",
           az_min, az_max, jumps, range_ok ? "OK" : "CHECK");
    return 0;
}

static int test_obs_elevation_extremes(void) {
    TestEnv t;

    setup_env(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        0, 0, 1500, 100, 0, 0, 0.5f);
    float elev_above = t.env.observations[14];

    setup_env(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        0, 0, 500, 100, 0, 0, 0.5f);
    float elev_below = t.env.observations[14];

    setup_env(&t, 0);
    place(&t.env,
        0, 0, 1000, 100, 0, 0, 1, 0, 0, 0,
        10, 0, 1500, 100, 0, 0, 0.5f);
    float elev_steep = t.env.observations[14];

    int bounded = 1;
    float vals[3] = {elev_above, elev_below, elev_steep};
    for (int i = 0; i < 3; i++) {
        if (isnan(vals[i]) || isinf(vals[i]) || vals[i] < -1.0f || vals[i] > 1.0f) bounded = 0;
    }
    int above_ok = elev_above > 0.8f;
    int below_ok = elev_below < -0.8f;
    int steep_ok = elev_steep > 0.9f;
    int ok = bounded && above_ok && below_ok && steep_ok;
    printf("obs_elev_ext:       above=%.3f, below=%.3f, steep=%.3f [%s]\n",
           elev_above, elev_below, elev_steep, ok ? "OK" : "CHECK");
    return 0;
}

static int test_obs_complex_maneuver(void) {
    TestEnv t; setup_env(&t, 0);
    place(&t.env,
        0, 0, 1500, 120, 0, 0, 1, 0, 0, 0,
        500, 0, 1500, 100, 0, 0, 1.0f);

    float prev_obs[26];
    int has_prev = 0;
    int bound_errors = 0, continuity_errors = 0;

    for (int step = 0; step < 200; step++) {
        float a[5] = {0.8f, -0.3f, 0.8f, 0.2f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        const float* obs = t.env.observations;

        for (int i = 0; i < t.env.obs_size; i++) {
            float v = obs[i];
            if (isnan(v) || isinf(v) || v < -1.0f || v > 1.0f) bound_errors++;
        }
        if (has_prev) {
            for (int i = 0; i < t.env.obs_size; i++) {
                float d = fabsf(obs[i] - prev_obs[i]);
                if (d > 0.5f) { continuity_errors++; break; }
            }
        }
        memcpy(prev_obs, obs, sizeof(float) * t.env.obs_size);
        has_prev = 1;
        if (t.env.player.pos.z < 200.0f) break;
    }
    int bounds_ok = (bound_errors == 0);
    int cont_ok = (continuity_errors <= 5);
    int ok = bounds_ok && cont_ok;
    printf("obs_complex:        bound_errors=%d, continuity_errors=%d [%s]\n",
           bound_errors, continuity_errors, ok ? "OK" : "CHECK");
    return 0;
}

static int test_quaternion_normalization(void) {
    TestEnv t; setup_env(&t, 0);
    place(&t.env,
        0, 0, 1500, 100, 0, 0, 1, 0, 0, 0,
        500, 0, 1500, 100, 0, 0, 1.0f);

    float mags[500]; int n = 0;
    float max_drift = 0.0f;
    double sum_drift = 0.0;
    for (int step = 0; step < 500; step++) {
        float t_s = (float)step * 0.02f;
        float ail = 0.5f * sinf(t_s * 2.0f);
        float elev = 0.3f * cosf(t_s * 1.5f);
        float rud = 0.2f * sinf(t_s * 0.8f);
        float a[5] = {0.7f, elev, ail, rud, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);

        Quat q = t.env.player.ori;
        float mag = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
        mags[n++] = mag;
        float drift = fabsf(mag - 1.0f);
        if (drift > max_drift) max_drift = drift;
        sum_drift += drift;
        if (t.env.player.pos.z < 200.0f) break;
    }
    float mean_drift = (float)(sum_drift / (n > 0 ? n : 1));
    float final_mag = (n > 0) ? mags[n - 1] : 1.0f;
    int ok = max_drift < 0.01f;
    printf("quat_norm:          max=%.6f, mean=%.6f, final=%.6f [%s]\n",
           max_drift, mean_drift, final_mag, ok ? "OK" : "WARN");
    return 0;
}

int main(void) {
    int fails = 0;
    fails += test_obs_azimuth_crossover();
    fails += test_obs_elevation_extremes();
    fails += test_obs_complex_maneuver();
    fails += test_quaternion_normalization();
    printf("\n%d hard failures\n", fails);
    return fails;
}
