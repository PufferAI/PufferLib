/*
 * test_flight_physics.c - 1:1 port of test_flight_physics.py.
 *
 * Each test_<name>() returns 0 on pass, 1 on fail, prints a single
 * line summarizing the result. main() sums failures and exits with
 * that count. Same style as ocean/dogfight/test_flight_dynamics.c.
 *
 * Python tests come in two flavors:
 *   - hard:  uses `assert` (only g_limit_neg, g_limit_pos in this file).
 *   - soft:  prints `[OK]`/`[CHECK]`/`[WRONG]` status; never fails the run.
 *
 * To stay 1:1 with Python, soft checks here also do not fail the run —
 * they print the status and return 0 either way. Hard checks return 1
 * on miss. The PASS/FAIL in the printed line still tells the reader.
 *
 * See test_common.h for compile recipe.
 */
#include "test_common.h"
#include "../autopilot.h"

/* P-51D references, mirrors test_flight_base.py. */
#define P51D_MAX_SPEED   159.0f
#define P51D_STALL_SPEED  45.0f
#define P51D_CLIMB_RATE   15.4f

/* identity-orientation force_state. */
static void force_level(Dogfight* env,
                        float px, float py, float pz,
                        float vx, float vy, float vz,
                        float throttle) {
    force_state(env,
        px, py, pz, vx, vy, vz,
        1.0f, 0.0f, 0.0f, 0.0f, throttle,
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f, -9999.0f,
        0, -1, -1);
}

/* Generic force_state with explicit orientation quat. */
static void force_with_ori(Dogfight* env,
                            float px, float py, float pz,
                            float vx, float vy, float vz,
                            float ow, float ox, float oy, float oz,
                            float throttle) {
    force_state(env,
        px, py, pz, vx, vy, vz,
        ow, ox, oy, oz, throttle,
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f,
        -9999.0f, -9999.0f, -9999.0f, -9999.0f,
        0, -1, -1);
}

/* Python's `velocity_cmd = clip(8 * target, -1, 1)` saturator (see comment
 * on py_level_flight_pitch_velocity in test_common.h). */
static float py_velocity_cmd(float target) {
    return clip_unit(target * 8.0f);
}

/* ============================================================
 * Speed/energy tests
 * ============================================================ */

static int test_max_speed(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, -1000, 0, 1000, 150, 0, 0, 1.0f);

    float prev = norm3(t.env.player.vel);
    int stable = 0;
    for (int step = 0; step < 1500; step++) {
        float elev = py_level_flight_pitch_velocity(&t.env.player);
        float a[5] = {1.0f, elev, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;

        float speed = norm3(t.env.player.vel);
        if (fabsf(speed - prev) < 0.05f) {
            if (++stable > 100) break;
        } else stable = 0;
        prev = speed;
    }

    float final_speed = norm3(t.env.player.vel);
    float diff = final_speed - P51D_MAX_SPEED;
    int ok = fabsf(diff) < 15.0f;
    printf("max_speed:          %.1f m/s (P-51D %.0f, diff %+.1f) [%s]\n",
           final_speed, P51D_MAX_SPEED, diff, ok ? "OK" : "CHECK");
    return 0;  /* soft */
}

static int test_acceleration(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, -1000, 0, 1000, 100, 0, 0, 1.0f);

    float v0 = norm3(t.env.player.vel);
    for (int step = 0; step < 500; step++) {
        float elev = py_level_flight_pitch_velocity(&t.env.player);
        float a[5] = {1.0f, elev, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float v1 = norm3(t.env.player.vel);
    float gain = v1 - v0;
    int ok = gain > 20.0f;
    printf("acceleration:       %.0f -> %.0f m/s (gain %+.1f) [%s]\n",
           v0, v1, gain, ok ? "OK" : "CHECK");
    return 0;  /* soft */
}

static int test_deceleration(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, -1000, 0, 1000, 150, 0, 0, 0.0f);

    float v0 = norm3(t.env.player.vel);
    for (int step = 0; step < 500; step++) {
        float elev = py_level_flight_pitch_velocity(&t.env.player);
        float a[5] = {-1.0f, elev, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float v1 = norm3(t.env.player.vel);
    float loss = v0 - v1;
    int ok = loss > 20.0f;
    printf("deceleration:       %.0f -> %.0f m/s (loss %+.1f) [%s]\n",
           v0, v1, loss, ok ? "OK" : "CHECK");
    return 0;  /* soft */
}

static int test_cruise_speed(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, -1000, 0, 1000, 120, 0, 0, 0.5f);

    float prev = norm3(t.env.player.vel);
    int stable = 0;
    for (int step = 0; step < 1500; step++) {
        float elev = py_level_flight_pitch_velocity(&t.env.player);
        float a[5] = {0.0f, elev, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;

        float speed = norm3(t.env.player.vel);
        if (fabsf(speed - prev) < 0.05f) {
            if (++stable > 100) break;
        } else stable = 0;
        prev = speed;
    }
    float final_speed = norm3(t.env.player.vel);
    printf("cruise_speed:       %.1f m/s (50%% throttle)\n", final_speed);
    return 0;  /* soft */
}

static int test_stall_speed(void) {
    /* Mirror Python: sweep V from 70 m/s down to 40 m/s in -5 steps,
     * compute pitch needed to fly level at each, force_state with that
     * pitch and zero throttle, run 100 steps and average vz over last 50.
     * If avg_vz >= -5, V is "flyable". Stall = first V where C_L_needed
     * exceeds C_L_max, or otherwise the last_flyable speed. */
    const float W = 4082.0f * 9.81f;
    const float rho = 1.225f, S = 21.65f;
    const float C_L_max = 1.48f, C_L_alpha = 5.56f;
    const float alpha_zero = -0.021f, wing_inc = 0.026f;
    const float V_stall_theory = sqrtf(2.0f * W / (rho * S * C_L_max));

    int stall_speed = 35;
    int last_flyable = -1;

    for (int V = 70; V > 35; V -= 5) {
        TestEnv t; setup_env(&t, 0);
        float q_dyn = 0.5f * rho * (float)V * (float)V;
        float C_L_needed = W / (q_dyn * S);
        if (C_L_needed > C_L_max) {
            stall_speed = V;
            break;
        }
        float alpha_needed = C_L_needed / C_L_alpha - wing_inc + alpha_zero;
        float pitch_rad = alpha_needed;
        float ow = cosf(-pitch_rad / 2.0f);
        float oy = sinf(-pitch_rad / 2.0f);
        force_with_ori(&t.env, 0, 0, 1000, (float)V, 0, 0,
                       ow, 0.0f, oy, 0.0f, 0.0f);

        float vzs[100]; int nv = 0;
        for (int s = 0; s < 100; s++) {
            vzs[nv++] = t.env.player.vel.z;
            float a[5] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            t_step(&t);
            if (t.env.terminals[0]) break;
        }
        int n_avg = (nv >= 50) ? 50 : nv;
        float avg_vz = arr_mean(vzs + (nv - n_avg), n_avg);
        if (avg_vz >= -5.0f) last_flyable = V;
    }
    if (last_flyable >= 0) stall_speed = last_flyable;

    float diff = (float)stall_speed - P51D_STALL_SPEED;
    int ok = fabsf(diff) < 10.0f;
    printf("stall_speed:        %.1f m/s (P-51D %.0f, diff %+.1f, theory %.0f) [%s]\n",
           (float)stall_speed, P51D_STALL_SPEED, diff, V_stall_theory,
           ok ? "OK" : "CHECK");
    return 0;
}

static int test_climb_rate(void) {
    const float W = 4082.0f * 9.81f;
    const float rho = 1.225f, S = 21.65f;
    const float C_L_alpha = 5.56f;
    const float alpha_zero = -0.021f, wing_inc = 0.026f;
    const float Vy = 74.0f;

    float expected_ROC = P51D_CLIMB_RATE;
    float gamma = asinf(expected_ROC / Vy);
    float L_needed = W * cosf(gamma);
    float q_dyn = 0.5f * rho * Vy * Vy;
    float C_L = L_needed / (q_dyn * S);
    float alpha = C_L / C_L_alpha - wing_inc + alpha_zero;
    float pitch = alpha + gamma;
    float target_pitch_deg = pitch * RAD;

    float ow = cosf(-pitch / 2.0f);
    float oy = sinf(-pitch / 2.0f);
    float vx = Vy * cosf(gamma);
    float vz = Vy * sinf(gamma);

    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 500, vx, 0, vz, ow, 0, oy, 0, 1.0f);

    float vzs[1000]; int nv = 0;
    float speeds[1000]; int ns = 0;
    for (int step = 0; step < 1000; step++) {
        if (step >= 250) {
            vzs[nv++] = t.env.player.vel.z;
            speeds[ns++] = norm3(t.env.player.vel);
        }
        float elev = ap_to_velocity(ap_hold_pitch(&t.env.player, target_pitch_deg));
        float ail = ap_to_velocity(ap_hold_bank(&t.env.player, 0.0f));
        float a[5] = {1.0f, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float avg_vz = nv ? arr_mean(vzs, nv) : 0.0f;
    float avg_speed = ns ? arr_mean(speeds, ns) : 0.0f;
    float diff = avg_vz - P51D_CLIMB_RATE;
    int ok = fabsf(diff) < 5.0f;
    printf("climb_rate:         %.1f m/s (P-51D %.0f, diff %+.1f, speed %.0f/%.0f) [%s]\n",
           avg_vz, P51D_CLIMB_RATE, diff, avg_speed, Vy, ok ? "OK" : "CHECK");
    return 0;
}

static int test_glide_ratio(void) {
    const float Cd0 = 0.0163f, k_glide = 0.072f;
    const float W = 4082.0f * 9.81f;
    const float rho = 1.225f, S = 21.65f;
    const float C_L_alpha = 5.56f;
    const float alpha_zero = -0.021f, wing_inc = 0.026f;

    float Cl_opt = sqrtf(Cd0 / k_glide);
    float Cd_opt = 2.0f * Cd0;
    float LD_max = Cl_opt / Cd_opt;
    float V_glide = sqrtf(2.0f * W / (rho * S * Cl_opt));
    float gamma = atanf(1.0f / LD_max);
    float sink_expected = V_glide * sinf(gamma);
    float alpha = Cl_opt / C_L_alpha - wing_inc + alpha_zero;
    float pitch = alpha - gamma;
    float target_pitch_deg = pitch * RAD;
    float ow = cosf(-pitch / 2.0f);
    float oy = sinf(-pitch / 2.0f);
    float vx = V_glide * cosf(gamma);
    float vz = -V_glide * sinf(gamma);

    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 2000, vx, 0, vz, ow, 0, oy, 0, 0.0f);

    float vzs[500]; int nv = 0;
    float speeds[500]; int ns = 0;
    for (int step = 0; step < 500; step++) {
        if (step >= 100) {
            vzs[nv++] = t.env.player.vel.z;
            speeds[ns++] = norm3(t.env.player.vel);
        }
        float elev = ap_to_velocity(ap_hold_pitch(&t.env.player, target_pitch_deg));
        float ail = ap_to_velocity(ap_hold_bank(&t.env.player, 0.0f));
        float a[5] = {-1.0f, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float avg_vz = nv ? arr_mean(vzs, nv) : 0.0f;
    float avg_sink = -avg_vz;
    float avg_speed = ns ? arr_mean(speeds, ns) : 0.0f;
    float measured_LD = (avg_sink > 0.1f) ? (avg_speed / avg_sink) : 0.0f;
    float diff = avg_sink - sink_expected;
    int ok = fabsf(diff) < 2.0f;
    printf("glide_ratio:        L/D=%.1f (theory %.1f, sink %.1f m/s, expected %.1f) [%s]\n",
           measured_LD, LD_max, avg_sink, sink_expected, ok ? "OK" : "CHECK");
    return 0;
}

/* ============================================================
 * Turn / direction / coordinated flight tests
 * ============================================================ */

static int test_sustained_turn(void) {
    const float V = 100.0f;
    const float bank_deg = 30.0f;
    float bank = bank_deg * DEG;
    float theory_turn_rate = (9.81f * tanf(bank) / V) * RAD;

    /* Spawn at +30 deg right bank, 3 deg nose-up. attitude_quat fixes the
     * sign bug that used to make this a left-bank spawn. */
    Quat q0 = attitude_quat(bank_deg, 3.0f);
    (void)bank;

    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 1500, V, 0, 0, q0.w, q0.x, q0.y, q0.z, 1.0f);

    float headings[250]; int nh = 0;
    float speeds[250]; int ns = 0;
    float alts[250]; int na = 0;
    float banks[250]; int nb = 0;

    for (int step = 0; step < 250; step++) {
        Plane* p = &t.env.player;
        float vx = p->vel.x, vy = p->vel.y;
        float heading = atan2f(vy, vx);
        float speed = norm3(p->vel);
        float alt = p->pos.z;
        Vec3 up = plane_up(p);
        float bank_actual = acosf(clip_unit(up.z)) * RAD;
        if (up.y > 0.0f) bank_actual = -bank_actual;

        if (step >= 50) {
            headings[nh++] = heading;
            speeds[ns++] = speed;
            alts[na++] = alt;
            banks[nb++] = bank_actual;
        }

        float elev_pos, ail_pos;
        ap_hold_bank_and_level(p, bank_deg, &elev_pos, &ail_pos);
        float elev = ap_to_velocity(elev_pos);
        float ail = ap_to_velocity(ail_pos);
        float a[5] = {1.0f, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }

    float turn_rate_actual = 0.0f;
    if (nh > 50) {
        float dh = arr_unwrap_delta(headings, nh);
        float time_elapsed = (float)nh * 0.02f;
        turn_rate_actual = (dh / time_elapsed) * RAD;
    }
    float avg_speed = ns ? arr_mean(speeds, ns) : 0.0f; (void)avg_speed;
    float alt_change = (na > 1) ? (alts[na - 1] - alts[0]) : 0.0f;
    float avg_bank = nb ? arr_mean(banks, nb) : 0.0f;

    int turn_ok = fabsf(turn_rate_actual) > theory_turn_rate * 0.5f;
    int alt_ok = fabsf(alt_change) < 50.0f;
    int bank_ok = fabsf(avg_bank - bank_deg) < 15.0f;
    int all_ok = turn_ok && alt_ok && bank_ok;
    printf("sustained_turn:     %.1f deg/s (theory %.1f, bank %.0f/%.0f, dalt %+.0fm) [%s]\n",
           fabsf(turn_rate_actual), theory_turn_rate, avg_bank, bank_deg,
           alt_change, all_ok ? "OK" : "CHECK");
    return 0;
}

static int test_turn_60(void) {
    const float bank_deg = 60.0f;
    const float bank_target = bank_deg * DEG;
    const float V = 100.0f;
    /* +60 deg right bank, no pitch. */
    Quat q0 = attitude_quat(bank_deg, 0.0f);
    (void)bank_target;

    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 1500, V, 0, 0, q0.w, q0.x, q0.y, q0.z, 1.0f);

    /* PID gains as in Python */
    const float coeff = 0.25f;
    const float elev_kp = -0.05f, elev_kd = 0.005f;
    const float roll_kp = -2.0f, roll_kd = -0.1f;
    float prev_vz = 0.0f, prev_bank_error = 0.0f;

    float headings[250]; int nh = 0;
    float alts[250]; int na = 0;
    float banks[250]; int nb = 0;

    for (int step = 0; step < 250; step++) {
        Plane* p = &t.env.player;
        float vz = p->vel.z;
        float alt = p->pos.z;
        float vx = p->vel.x, vy = p->vel.y;
        float heading = atan2f(vy, vx);
        Vec3 up = plane_up(p);
        float bank_actual = acosf(clip_unit(up.z));
        if (up.y < 0.0f) bank_actual = -bank_actual;

        float vz_error = -vz;
        float vz_deriv = (vz - prev_vz) / 0.02f;
        float target_elev = elev_kp * vz_error + elev_kd * vz_deriv;
        target_elev = clip_unit(target_elev);
        prev_vz = vz;
        float elev_vel = clip_unit(2.0f * target_elev / coeff);

        float bank_error = bank_target - bank_actual;
        float bank_deriv = (bank_error - prev_bank_error) / 0.02f;
        float target_ail = roll_kp * bank_error + roll_kd * bank_deriv;
        target_ail = clip_unit(target_ail);
        prev_bank_error = bank_error;
        float ail_vel = clip_unit(2.0f * target_ail / coeff);

        if (step >= 25) {
            headings[nh++] = heading;
            alts[na++] = alt;
            banks[nb++] = bank_actual * RAD;
        }

        float a[5] = {1.0f, elev_vel, ail_vel, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }

    float turn_rate = 0.0f;
    if (nh > 1) {
        float dh = arr_unwrap_delta(headings, nh);
        turn_rate = (dh / ((float)nh * 0.02f)) * RAD;
    }
    float alt_change = (na > 1) ? (alts[na - 1] - alts[0]) : 0.0f;
    float bank_mean = nb ? arr_mean(banks, nb) : 0.0f;
    float theory_rate = (9.81f * tanf(bank_target) / V) * RAD;
    float eff = (theory_rate != 0.0f) ? (100.0f * turn_rate / theory_rate) : 0.0f;

    int ok = (eff > 85.0f && eff < 105.0f) && fabsf(alt_change) < 50.0f;
    printf("turn_60:            %.1f deg/s (theory %.1f, eff %.0f%%, bank %.0f, dalt %+.0fm) [%s]\n",
           turn_rate, theory_rate, eff, bank_mean, alt_change, ok ? "OK" : "CHECK");
    return 0;
}

static int test_pitch_direction(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 1000, 80, 0, 0, 0.5f);

    float fwd_z_before = plane_fwd(&t.env.player).z;
    float a[5] = {0.5f, 1.0f, 0.0f, 0.0f, 0.0f};
    run_steps(&t, 50, a);
    float fwd_z_after = plane_fwd(&t.env.player).z;

    int nose_down = fwd_z_after < fwd_z_before;
    printf("pitch_direction:    +elev nose %s [%s]\n",
           nose_down ? "DOWN" : "UP",
           nose_down ? "OK" : "WRONG");
    return 0;  /* soft */
}

static int test_roll_direction(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 1000, 80, 0, 0, 0.5f);

    float a[5] = {0.5f, 0.0f, 1.0f, 0.0f, 0.0f};
    run_steps(&t, 50, a);
    float up_y = plane_up(&t.env.player).y;

    int rolled = fabsf(up_y) > 0.1f;
    printf("roll_direction:     |up.y|=%.3f [%s]\n",
           up_y, rolled ? "OK" : "WRONG");
    return 0;  /* soft */
}

static int test_rudder_only_turn(void) {
    const float V = 120.0f;
    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 1000, V, 0, 0, 1, 0, 0, 0, 1.0f);

    const float coeff = 0.25f;
    const float roll_kp = 1.0f, roll_kd = 0.05f;
    const float elev_kp = 0.001f, elev_kd = 0.001f;
    float prev_roll = 0.0f, prev_vz = 0.0f;
    float headings[300]; int nh = 0;

    for (int step = 0; step < 300; step++) {
        Plane* p = &t.env.player;
        float vx = p->vel.x, vy = p->vel.y, vz = p->vel.z;
        Vec3 up = plane_up(p);
        float heading = atan2f(vy, vx);
        headings[nh++] = heading;
        float roll = atan2f(up.y, up.z);

        float roll_error = -roll;
        float roll_deriv = (roll - prev_roll) / 0.02f;
        float target_ail = -(roll_kp * roll_error - roll_kd * roll_deriv);
        target_ail = clip_unit(target_ail);
        prev_roll = roll;
        float aileron_vel = clip_unit(2.0f * target_ail / coeff);

        float vz_error = -vz;
        float vz_deriv = (vz - prev_vz) / 0.02f;
        float target_elev = -elev_kp * vz_error - elev_kd * vz_deriv;
        if (target_elev >  0.3f) target_elev =  0.3f;
        if (target_elev < -0.3f) target_elev = -0.3f;
        prev_vz = vz;
        float elev_vel = clip_unit(2.0f * target_elev / coeff);

        float rudder_vel = clip_unit(2.0f * 1.0f / coeff);

        float a[5] = {1.0f, elev_vel, aileron_vel, rudder_vel, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }

    float total_dh_rad = arr_unwrap_delta(headings, nh);
    float total_dh = total_dh_rad * RAD;
    float initial_rate = 0.0f;
    if (nh > 25) initial_rate = ((headings[25] - headings[0]) / 0.5f) * RAD;
    float final_rate = 0.0f;
    if (nh > 200) final_rate = ((headings[nh - 1] - headings[nh - 100]) / 2.0f) * RAD;

    int changed = fabsf(total_dh) > 2.0f;
    int limited = fabsf(total_dh) < 20.0f;
    int ok = changed && limited;
    printf("rudder_only_turn:   heading=%.1f deg (init=%.1f deg/s, final=%.1f deg/s) [%s]\n",
           total_dh, initial_rate, final_rate, ok ? "OK" : "FAIL");
    return 0;
}

static int test_knife_edge_pull(void) {
    const float V = 150.0f;
    float roll_90 = 90.0f * DEG;
    float qw = cosf(roll_90 / 2.0f);
    float qx = -sinf(roll_90 / 2.0f);

    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 1500, V, 0, 0, qw, qx, 0, 0, 1.0f);

    float alt_start = t.env.player.pos.z;
    float headings[100]; int nh = 0;
    float alts[100]; int na = 0;
    float up_zs[100]; int nz = 0;

    for (int step = 0; step < 100; step++) {
        Plane* p = &t.env.player;
        float heading = atan2f(p->vel.y, p->vel.x);
        headings[nh++] = heading;
        alts[na++] = p->pos.z;
        up_zs[nz++] = plane_up(p).z;

        float elev_vel = clip_unit(2.0f * (-1.0f) / 0.25f);
        float a[5] = {1.0f, elev_vel, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }

    float dh = arr_unwrap_delta(headings, nh) * RAD;
    float alt_loss = alt_start - alts[na - 1];
    float avg_uz = arr_mean(up_zs, nz);
    float t_elapsed = (float)nh * 0.02f;
    float turn_rate = t_elapsed > 0.0f ? dh / t_elapsed : 0.0f;

    int heading_ok = dh > 20.0f;
    int alt_ok = alt_loss > 5.0f;
    int roll_kept = fabsf(avg_uz) < 0.3f;
    int ok = heading_ok && alt_ok && roll_kept;
    const char* dir = (dh > 0.0f) ? "LEFT" : "RIGHT";
    printf("knife_edge_pull:    turn=%.1f deg/s (%s), alt_lost=%.0fm, |up_z|=%.2f [%s]\n",
           turn_rate, dir, alt_loss, fabsf(avg_uz), ok ? "OK" : "CHECK");
    return 0;
}

static int test_knife_edge_flight(void) {
    const float V = 120.0f;
    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 1500, V, 0, 0, 1, 0, 0, 0, 1.0f);

    /* Phase 1: roll right with full aileron velocity for 30 ticks */
    for (int s = 0; s < 30; s++) {
        float ail_vel = clip_unit(2.0f * 1.0f / 0.25f);
        float a[5] = {1.0f, 0.0f, ail_vel, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
    }

    Vec3 up = plane_up(&t.env.player);
    float roll_deg = acosf(clip_unit(up.z)) * RAD;
    float alt_start = t.env.player.pos.z;
    if (fabsf(roll_deg - 90.0f) > 15.0f) {
        printf("knife_edge_flight: SKIP — failed roll to 90 deg (got %.0f)\n", roll_deg);
        return 0;
    }

    float alts[150]; int na = 0;
    for (int step = 0; step < 150; step++) {
        alts[na++] = t.env.player.pos.z;
        float rud_vel = clip_unit(2.0f * (-1.0f) / 0.25f);
        float a[5] = {1.0f, 0.0f, 0.0f, rud_vel, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float alt_end = alts[na - 1];
    float alt_loss = alt_start - alt_end;
    float t_elapsed = (float)na * 0.02f;
    float sink_rate = t_elapsed > 0.0f ? alt_loss / t_elapsed : 0.0f;
    int realistic = alt_loss > 10.0f;
    printf("knife_edge_flight:  sink=%.1f m/s, alt_lost=%.0fm in %.1fs [%s]\n",
           sink_rate, alt_loss, t_elapsed, realistic ? "OK" : "FAIL");
    return 0;
}

/* ============================================================
 * Autopilot mode / enum tests
 * ============================================================ */

static int test_mode_weights(void) {
    /* Bias 100% toward LEVEL via mode_weights, then run resets and
     * verify autopilot lands on AP_LEVEL each time.  Uses the same
     * machinery as autopilot_randomize() exposed in autopilot.h. */
    TestEnv t; setup_env(&t, 0);
    autopilot_set_mode(&t.env.opponent_ap, AP_RANDOM,
                       AP_DEFAULT_THROTTLE, AP_DEFAULT_BANK_DEG, AP_DEFAULT_CLIMB_RATE);
    for (int i = 0; i < AP_COUNT; i++) t.env.opponent_ap.mode_weights[i] = 0.0f;
    t.env.opponent_ap.mode_weights[AP_LEVEL] = 1.0f;

    int level_count = 0;
    const int trials = 50;
    for (int i = 0; i < trials; i++) {
        c_reset(&t.env);
        if (t.env.opponent_ap.mode == AP_LEVEL) level_count++;
    }
    float pct = 100.0f * level_count / (float)trials;
    int ok = (level_count == trials);
    printf("mode_weights:       %.1f%% (should be 100%% AP_LEVEL) [%s]\n",
           pct, ok ? "OK" : "CHECK");

    /* Mixed weights distribution check */
    autopilot_set_mode(&t.env.opponent_ap, AP_RANDOM,
                       AP_DEFAULT_THROTTLE, AP_DEFAULT_BANK_DEG, AP_DEFAULT_CLIMB_RATE);
    for (int i = 0; i < AP_COUNT; i++) t.env.opponent_ap.mode_weights[i] = 0.0f;
    t.env.opponent_ap.mode_weights[AP_LEVEL] = 0.5f;
    t.env.opponent_ap.mode_weights[AP_TURN_LEFT] = 0.25f;
    t.env.opponent_ap.mode_weights[AP_TURN_RIGHT] = 0.25f;
    int counts[AP_COUNT] = {0};
    const int trials2 = 200;
    for (int i = 0; i < trials2; i++) {
        c_reset(&t.env);
        AutopilotMode m = t.env.opponent_ap.mode;
        if ((int)m >= 0 && (int)m < AP_COUNT) counts[m]++;
    }
    float level_pct = 100.0f * counts[AP_LEVEL] / (float)trials2;
    float climb_pct = 100.0f * counts[AP_CLIMB] / (float)trials2;
    int dist_ok = (level_pct > 35.0f) && (climb_pct < 10.0f);
    printf("  distribution: LEVEL=%.0f%% TL=%.0f%% TR=%.0f%% CLIMB=%.0f%% [%s]\n",
           level_pct,
           100.0f * counts[AP_TURN_LEFT] / trials2,
           100.0f * counts[AP_TURN_RIGHT] / trials2,
           climb_pct, dist_ok ? "OK" : "CHECK");
    return 0;
}

static int test_autopilot_enum_sync(void) {
    int ok = (AP_STRAIGHT == 0)
          && (AP_LEVEL == 1)
          && (AP_TURN_LEFT == 2)
          && (AP_TURN_RIGHT == 3)
          && (AP_CLIMB == 4)
          && (AP_DESCEND == 5)
          && (AP_HARD_TURN_LEFT == 6)
          && (AP_HARD_TURN_RIGHT == 7)
          && (AP_WEAVE == 8)
          && (AP_EVASIVE == 9)
          && (AP_RANDOM == 10);
    printf("autopilot_enum_sync: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;  /* enum sync is a hard regression — fail if broken */
}

static int test_autopilot_random_not_hardturn(void) {
    /* Set RANDOM mode (which uses default mode_weights[1..5] from
     * autopilot_init: uniform on LEVEL, TURN_*, CLIMB, DESCEND).
     * After 30 c_resets the resulting mode must be in [1, 5]
     * — never AP_HARD_TURN_LEFT (=6). */
    TestEnv t; setup_env(&t, 0);
    autopilot_set_mode(&t.env.opponent_ap, AP_RANDOM,
                       AP_DEFAULT_THROTTLE, AP_DEFAULT_BANK_DEG, AP_DEFAULT_CLIMB_RATE);

    int seen[AP_COUNT] = {0};
    int saw_six = 0;
    int all_in_range = 1;
    for (int i = 0; i < 30; i++) {
        c_reset(&t.env);
        AutopilotMode m = t.env.opponent_ap.mode;
        if ((int)m == 6) saw_six = 1;
        if ((int)m < 1 || (int)m > 5) all_in_range = 0;
        if ((int)m >= 0 && (int)m < AP_COUNT) seen[m]++;
    }
    int unique = 0;
    for (int i = 1; i <= 5; i++) if (seen[i] > 0) unique++;
    int variety = unique >= 3;
    int ok = !saw_six && all_in_range && variety;
    printf("random_mode:        unique=%d, no_mode_6=%s [%s]\n",
           unique, saw_six ? "false" : "true", ok ? "OK" : "FAIL");
    return 0;
}

static int test_autopilot_bounds_check(void) {
    /* The Python test verifies binding.c clamps invalid mode ints to
     * AP_STRAIGHT.  The 4.0 binding does NOT expose set_autopilot at
     * all and there is no bounds check in autopilot_set_mode itself.
     * Document the Python-binding-only behavior so the suite is
     * structurally 1:1 with the .py file. */
    printf("autopilot_bounds_check: N/A — Python binding only [SKIP]\n");
    return 0;
}

static int test_force_state_pid_reset(void) {
    TestEnv t; setup_env(&t, 0);
    autopilot_set_mode(&t.env.opponent_ap, AP_LEVEL,
                       AP_DEFAULT_THROTTLE, 0.0f, 0.0f);
    float a[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    run_steps(&t, 50, a);

    force_level(&t.env, 0, 0, 2000, 150, 0, 50, 1.0f);

    int pos_ok = fabsf(t.env.player.pos.z - 2000.0f) < 1.0f;
    int vel_ok = fabsf(t.env.player.vel.z -   50.0f) < 1.0f;
    int pid_ok = fabsf(t.env.opponent_ap.prev_vz - t.env.opponent.vel.z) < 1e-3f
              && t.env.opponent_ap.prev_bank_error == 0.0f;

    int ok = pos_ok && vel_ok && pid_ok;
    printf("force_state_pid_reset: pos=%d vel=%d pid=%d [%s]\n",
           pos_ok, vel_ok, pid_ok, ok ? "OK" : "FAIL");
    return 0;  /* soft */
}

/* ============================================================
 * G-force tests
 * ============================================================ */

static int test_g_level_flight(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 1000, 120, 0, 0, 0.5f);

    float g_sum = 0.0f;
    int n = 0;
    for (int step = 0; step < 200; step++) {
        float elev = py_level_flight_pitch_velocity(&t.env.player);
        float a[5] = {0.0f, elev, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (step >= 100) { g_sum += t.env.player.g_force; n++; }
    }
    float avg_g = g_sum / (float)n;
    int ok = avg_g > 0.8f && avg_g < 1.2f;
    printf("g_level_flight:     %.2f G (target ~1.0) [%s]\n",
           avg_g, ok ? "OK" : "CHECK");
    return 0;  /* soft */
}

static int test_g_push_forward(void) {
    float min_g = 1e9f;
    float targets[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (size_t ti = 0; ti < sizeof(targets)/sizeof(targets[0]); ti++) {
        TestEnv t; setup_env(&t, 0);
        force_level(&t.env, 0, 0, 1500, 150, 0, 0, 1.0f);
        float cmd = py_velocity_cmd(targets[ti]);
        for (int step = 0; step < 25; step++) {
            float a[5] = {1.0f, cmd, 0.0f, 0.0f, 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            t_step(&t);
            if (t.env.player.g_force < min_g) min_g = t.env.player.g_force;
        }
    }
    int ok = min_g < 0.5f;
    printf("g_push_forward:     min G %+.2f (need < 0.5) [%s]\n",
           min_g, ok ? "OK" : "CHECK");
    return 0;  /* soft */
}

static int test_g_pull_back(void) {
    float max_g = -1e9f;
    float targets[] = {0.0f, -0.25f, -0.5f, -0.75f, -1.0f};
    for (size_t ti = 0; ti < sizeof(targets)/sizeof(targets[0]); ti++) {
        TestEnv t; setup_env(&t, 0);
        force_level(&t.env, 0, 0, 1500, 150, 0, 0, 1.0f);
        float cmd = py_velocity_cmd(targets[ti]);
        for (int step = 0; step < 25; step++) {
            float a[5] = {1.0f, cmd, 0.0f, 0.0f, 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            t_step(&t);
            if (t.env.player.g_force > max_g) max_g = t.env.player.g_force;
        }
    }
    int ok = max_g > 4.0f;
    printf("g_pull_back:        max G %+.2f (need > 4.0) [%s]\n",
           max_g, ok ? "OK" : "CHECK");
    return 0;  /* soft */
}

static int test_g_limit_negative(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 2000, 150, 0, 0, 1.0f);

    float g_min = 1e9f;
    float cmd = py_velocity_cmd(1.0f);  /* full forward */
    for (int step = 0; step < 150; step++) {
        float a[5] = {1.0f, cmd, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.player.g_force < g_min) g_min = t.env.player.g_force;
    }
    /* G_LIMIT_NEG is stored positive, used as -G_LIMIT_NEG. */
    int ok = g_min >= -G_LIMIT_NEG - 0.1f;
    printf("g_limit_negative:   min G %+.2f (limit %.1f) [%s]\n",
           g_min, -G_LIMIT_NEG, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_g_limit_positive(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 2000, 180, 0, 0, 1.0f);

    float g_max = -1e9f;
    float cmd = py_velocity_cmd(-1.0f);  /* full back */
    for (int step = 0; step < 150; step++) {
        float a[5] = {1.0f, cmd, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.player.g_force > g_max) g_max = t.env.player.g_force;
    }
    int ok = g_max <= G_LIMIT_POS + 0.1f;
    printf("g_limit_positive:   max G %+.2f (limit %.1f) [%s]\n",
           g_max, G_LIMIT_POS, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* ============================================================
 * Fine control / oscillation diagnostics
 * ============================================================ */

static int test_gentle_pitch_control(void) {
    float elev_targets[] = {-0.05f, -0.1f, -0.15f, -0.2f, -0.25f, -0.3f};
    int n_elev = (int)(sizeof(elev_targets) / sizeof(elev_targets[0]));
    float pitch_rates[6] = {0};

    for (int i = 0; i < n_elev; i++) {
        TestEnv t; setup_env(&t, 0);
        force_level(&t.env, 0, 0, 1500, 120, 0, 0, 0.7f);
        Vec3 fwd0 = plane_fwd(&t.env.player);
        float pitch_start = atan2f(fwd0.z, fwd0.x);
        float cmd = py_velocity_cmd(elev_targets[i]);
        for (int s = 0; s < 50; s++) {
            float a[5] = {0.4f, cmd, 0.0f, 0.0f, 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            t_step(&t);
        }
        Vec3 fwd1 = plane_fwd(&t.env.player);
        float pitch_end = atan2f(fwd1.z, fwd1.x);
        float pitch_change_deg = (pitch_end - pitch_start) * RAD;
        pitch_rates[i] = pitch_change_deg / 1.0f;
    }
    float r01  = pitch_rates[1];
    float r025 = pitch_rates[4];
    float t25 = (fabsf(r01) > 0.1f) ? (2.5f / fabsf(r01)) : 1e9f;
    float ratio = (fabsf(r01) > 0.1f) ? (r025 / r01) : 0.0f;
    int gentle  = (fabsf(r01) > 2.0f) && (fabsf(r01) < 15.0f);
    int prop    = (ratio > 1.5f) && (ratio < 4.0f);
    int can_aim = t25 < 2.0f;
    int ok = gentle && prop && can_aim;
    printf("gentle_pitch:       rate@-0.1=%.1f deg/s, 2.5deg_time=%.2fs, ratio=%.2f [%s]\n",
           r01, t25, ratio, ok ? "OK" : "CHECK");
    return 0;
}

static int test_high_speed_pitch_oscillation(void) {
    const float speed = 140.0f;
    const float bank_deg = 80.0f;
    /* +80 deg right bank, 3 deg nose-up. */
    Quat q0 = attitude_quat(bank_deg, 3.0f);

    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 2000, speed, 0, 0, q0.w, q0.x, q0.y, q0.z, 1.0f);

    float pitch_rates[250]; int npr = 0;

    for (int step = 0; step < 250; step++) {
        Plane* p = &t.env.player;
        pitch_rates[npr++] = p->omega.y;
        float elev_vel = clip_unit(2.0f * (-1.0f) / 0.25f);
        float ail = ap_to_velocity(ap_hold_bank(p, bank_deg));
        float a[5] = {1.0f, elev_vel, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }

    int settle_offset = (npr > 50) ? 50 : 0;
    int n_settled = npr - settle_offset;
    const float* settled = pitch_rates + settle_offset;

    float std_dev = arr_std(settled, n_settled);
    int crossings = arr_zero_crossings(settled, n_settled);
    float max_amp = arr_max(settled, n_settled) - arr_min(settled, n_settled);
    int growing = 0;
    if (n_settled > 100) {
        float fa = arr_max(settled, 50) - arr_min(settled, 50);
        float sa = arr_max(settled + n_settled - 50, 50) - arr_min(settled + n_settled - 50, 50);
        growing = (sa > fa * 1.5f);
    }
    int ok = (std_dev < 0.5f) && !growing;
    printf("hs_pitch_osc:       std=%.3f rad/s, crossings=%d, amp=%.3f, growing=%s [%s]\n",
           std_dev, crossings, max_amp, growing ? "true" : "false",
           ok ? "OK" : "UNSTABLE");
    return 0;
}

static int test_high_speed_roll_oscillation(void) {
    const float speed = 140.0f;
    const float pitch_deg = -75.0f;
    const float pitch_rad = pitch_deg * DEG;
    float ow = cosf(pitch_rad / 2.0f);
    float oy = sinf(pitch_rad / 2.0f);
    float vx = speed * cosf(-pitch_rad);
    float vz = speed * sinf(-pitch_rad);

    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 2500, vx, 0, vz, ow, 0, oy, 0, 0.5f);

    float roll_rates[250]; int nrr = 0;
    for (int step = 0; step < 250; step++) {
        roll_rates[nrr++] = t.env.player.omega.x;
        float ail_vel = clip_unit(2.0f * 0.5f / 0.25f);
        float a[5] = {0.0f, 0.0f, ail_vel, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        t_step(&t);
        if (t.env.terminals[0]) break;
    }

    int settle_offset = (nrr > 50) ? 50 : 0;
    int n_settled = nrr - settle_offset;
    const float* settled = roll_rates + settle_offset;

    float mean_rate = arr_mean(settled, n_settled);
    float std_dev = arr_std(settled, n_settled);
    /* zero crossings of (rate - mean) */
    float demean[250];
    for (int i = 0; i < n_settled; i++) demean[i] = settled[i] - mean_rate;
    int crossings = arr_zero_crossings(demean, n_settled);
    float max_dev = 0.0f;
    for (int i = 0; i < n_settled; i++) {
        float d = fabsf(demean[i]);
        if (d > max_dev) max_dev = d;
    }
    int growing = 0;
    if (n_settled > 100) {
        float v1 = arr_var(settled, 50);
        float v2 = arr_var(settled + n_settled - 50, 50);
        growing = (v2 > v1 * 2.0f);
    }
    int ok = (max_dev < 0.3f) && !growing;
    printf("hs_roll_osc:        std=%.3f rad/s, mean=%.3f, max_dev=%.3f, growing=%s, x=%d [%s]\n",
           std_dev, mean_rate, max_dev, growing ? "true" : "false", crossings,
           ok ? "OK" : "UNSTABLE");
    return 0;
}

static int test_speed_sweep_stability(void) {
    int speeds[] = {80, 100, 120, 140, 150};
    int n_speeds = (int)(sizeof(speeds) / sizeof(speeds[0]));
    float variances[5] = {0};

    for (int si = 0; si < n_speeds; si++) {
        TestEnv t; setup_env(&t, 0);
        const float bank_deg = 45.0f;
        Quat q0 = attitude_quat(bank_deg, 0.0f);
        force_with_ori(&t.env, 0, 0, 2000, (float)speeds[si], 0, 0,
                       q0.w, q0.x, q0.y, q0.z, 1.0f);

        float pitch_rates[150]; int npr = 0;
        for (int step = 0; step < 150; step++) {
            pitch_rates[npr++] = t.env.player.omega.y;
            float elev_vel = clip_unit(2.0f * (-0.5f) / 0.25f);
            float ail = ap_to_velocity(ap_hold_bank(&t.env.player, bank_deg));
            float a[5] = {1.0f, elev_vel, ail, 0.0f, 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            t_step(&t);
            if (t.env.terminals[0]) break;
        }
        int settle_offset = (npr > 25) ? 25 : 0;
        const float* settled = pitch_rates + settle_offset;
        int n_settled = npr - settle_offset;
        variances[si] = arr_var(settled, n_settled);
        printf("  V=%3d m/s: var=%.5f, std=%.3f rad/s\n",
               speeds[si], variances[si], sqrtf(variances[si]));
    }

    float v0 = variances[0];
    float vN = variances[n_speeds - 1];
    float ratio = (v0 > 1e-4f) ? (vN / v0) : (vN * 10000.0f);
    int ok = ratio < 10.0f;
    printf("speed_sweep:        var_ratio(150/80)=%.1fx (want <10x) [%s]\n",
           ratio, ok ? "OK" : "CHECK");
    return 0;
}

/* ============================================================
 * Attitude-recovery tests
 *
 * Each test spawns the plane at a deliberate deviation (off-bank,
 * off-pitch, descending/climbing, knife-edge) and asks the autopilot to
 * drive it back to wings-level / vz=0. We measure:
 *   - recovery time: first tick where bank/pitch error < tolerance
 *   - max overshoot past target on the opposite side
 *   - final-50-tick stability (std of bank in degrees)
 * PASS when recovery_sec is under threshold AND final_std is small.
 * ============================================================ */

typedef struct {
    const char* name;
    float spawn_bank_deg;
    float spawn_pitch_deg;
    float V;
    float pass_sec;       /* recovery threshold (sec) */
    int   max_steps;      /* sim cap (50 Hz: 250 = 5s) */
    int   pitch_recovery; /* 0 = bank recovery, 1 = pitch/vz recovery */
} RecoverySpec;

static int run_recovery_test(const RecoverySpec* s) {
    const float bank_tol = 5.0f;   /* deg */
    const float vz_tol = 2.0f;     /* m/s */
    const float target_bank = 0.0f;

    TestEnv t; setup_env(&t, 0);
    Quat q0 = attitude_quat(s->spawn_bank_deg, s->spawn_pitch_deg);
    float pitch_rad = s->spawn_pitch_deg * DEG;
    float bank_rad  = s->spawn_bank_deg  * DEG;
    /* Velocity: forward along the spawned attitude.
     * vx = V*cos(pitch)*cos(bank-component-projection)... we keep it simple:
     * use V along world-x*cos(pitch) plus world-z*sin(pitch). For bank-only
     * spawns (pitch=0) this collapses to (V, 0, 0). */
    float vx = s->V * cosf(pitch_rad) * cosf(bank_rad);
    float vy = s->V * cosf(pitch_rad) * sinf(bank_rad); (void)vy;
    float vz = s->V * sinf(pitch_rad);
    /* Keep it body-aligned with no sideslip — fwd vector in world frame
     * points along the spawned ori. We use the rotated body-X. */
    Vec3 fwd_world = quat_rotate(q0, vec3(1.0f, 0.0f, 0.0f));
    vx = s->V * fwd_world.x;
    vy = s->V * fwd_world.y;
    vz = s->V * fwd_world.z;

    force_with_ori(&t.env, 0, 0, 1500, vx, vy, vz,
                   q0.w, q0.x, q0.y, q0.z, 1.0f);

    int recovery_step = -1;
    float max_overshoot = 0.0f;
    float final_banks[60]; int nf = 0;
    int collect_start = (s->max_steps > 60) ? (s->max_steps - 60) : 0;

    for (int step = 0; step < s->max_steps; step++) {
        Plane* p = &t.env.player;
        float bank_now = plane_bank_deg(p);
        float vz_now = p->vel.z;

        int recovered;
        if (s->pitch_recovery) {
            recovered = (fabsf(vz_now) < vz_tol) && (fabsf(bank_now) < bank_tol);
        } else {
            recovered = fabsf(bank_now - target_bank) < bank_tol;
        }
        if (recovery_step < 0 && recovered) {
            recovery_step = step;
        }
        if (recovery_step >= 0) {
            /* overshoot = signed crossing past target on the opposite side */
            float past = (s->spawn_bank_deg < 0.0f) ? bank_now : -bank_now;
            if (past > max_overshoot) max_overshoot = past;
        }
        if (step >= collect_start) {
            final_banks[nf++] = bank_now;
        }

        float elev_pos, ail_pos;
        if (s->pitch_recovery) {
            ail_pos = ap_hold_bank(p, target_bank);
            elev_pos = ap_hold_vz(p, 0.0f);
        } else {
            ap_hold_bank_and_level(p, target_bank, &elev_pos, &ail_pos);
        }
        float elev = ap_to_velocity(elev_pos);
        float ail = ap_to_velocity(ail_pos);
        float a[5] = {1.0f, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));

        recovery_log_row(s->name, step, p, a);

        t_step(&t);
        if (t.env.terminals[0]) break;
    }

    float recovery_sec = (recovery_step >= 0) ? recovery_step * 0.02f : -1.0f;
    float final_std = (nf > 1) ? arr_std(final_banks, nf) : 0.0f;
    int recovered_in_time = (recovery_step >= 0) && (recovery_sec <= s->pass_sec);
    int stable = final_std < 5.0f;
    int ok = recovered_in_time && stable;

    printf("%-22s spawn(b=%+.0f,p=%+.0f) -> recovered=%s, t=%.2fs (lim %.1fs), overshoot=%.1f, final_std=%.2f [%s]\n",
           s->name,
           s->spawn_bank_deg, s->spawn_pitch_deg,
           (recovery_step >= 0) ? "yes" : "NO",
           (recovery_step >= 0) ? recovery_sec : -1.0f,
           s->pass_sec, max_overshoot, final_std,
           ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_recovery_bank_left(void) {
    RecoverySpec s = {"recovery_bank_left:",   -45.0f, 0.0f,   100.0f, 2.0f, 200, 0};
    return run_recovery_test(&s);
}
static int test_recovery_bank_right(void) {
    RecoverySpec s = {"recovery_bank_right:",  +45.0f, 0.0f,   100.0f, 2.0f, 200, 0};
    return run_recovery_test(&s);
}
static int test_recovery_pitch_dive(void) {
    RecoverySpec s = {"recovery_pitch_dive:",    0.0f, -20.0f, 100.0f, 3.5f, 300, 1};
    return run_recovery_test(&s);
}
static int test_recovery_pitch_climb(void) {
    RecoverySpec s = {"recovery_pitch_climb:",   0.0f, +20.0f, 100.0f, 3.5f, 300, 1};
    return run_recovery_test(&s);
}
static int test_recovery_knife_edge_left(void) {
    RecoverySpec s = {"recovery_knife_edge_l:", -90.0f, 0.0f,  120.0f, 4.0f, 350, 0};
    return run_recovery_test(&s);
}
static int test_recovery_knife_edge_right(void) {
    RecoverySpec s = {"recovery_knife_edge_r:", +90.0f, 0.0f,  120.0f, 4.0f, 350, 0};
    return run_recovery_test(&s);
}
static int test_recovery_inverted(void) {
    /* Spawn fully inverted (180 deg roll). AP must roll either way to level. */
    RecoverySpec s = {"recovery_inverted:    ", 180.0f, 0.0f,  130.0f, 4.0f, 400, 0};
    return run_recovery_test(&s);
}
static int test_recovery_steep_dive(void) {
    /* -45 deg pitch dive at 130 m/s. AP must pull out. */
    RecoverySpec s = {"recovery_steep_dive:  ",   0.0f, -45.0f, 130.0f, 4.0f, 400, 1};
    return run_recovery_test(&s);
}
static int test_recovery_steep_climb(void) {
    /* +45 deg pitch climb at 90 m/s. AP must lower nose to level.
     * Slow because climb-rate is steep and elevator authority is limited
     * at low speed; allow 8s. */
    RecoverySpec s = {"recovery_steep_climb: ",   0.0f, +45.0f,  90.0f, 8.0f, 600, 1};
    return run_recovery_test(&s);
}

/* ============================================================
 * Speed / altitude / heading correction tests
 * ============================================================ */

/* Forward speed correction: spawn level at start_speed, throttle to drive
 * speed -> target_speed. PASS when speed within tol of target within max_sec. */
static int run_speed_recovery(const char* name,
                              float start_speed, float target_speed,
                              float pass_sec, int max_steps) {
    const float tol = 5.0f;  /* m/s */
    TestEnv t; setup_env(&t, 0);
    t.env.max_steps = max_steps + 100;  /* avoid env auto-terminate */
    /* Level spawn, identity orientation, throttle roughly matched to start. */
    float init_throttle = (start_speed > 100.0f) ? 1.0f : 0.0f;
    force_with_ori(&t.env, 0, 0, 1500, start_speed, 0, 0,
                   1.0f, 0.0f, 0.0f, 0.0f, init_throttle);

    int recovery_step = -1;
    for (int step = 0; step < max_steps; step++) {
        Plane* p = &t.env.player;
        float speed = sqrtf(p->vel.x*p->vel.x + p->vel.y*p->vel.y + p->vel.z*p->vel.z);
        if (recovery_step < 0 && fabsf(speed - target_speed) < tol) {
            recovery_step = step;
        }
        float thr = ap_hold_speed(p, target_speed);
        float elev = ap_to_velocity(ap_hold_vz(p, 0.0f));
        float ail = ap_to_velocity(ap_hold_bank(p, 0.0f));
        float a[5] = {thr, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        recovery_log_row(name, step, p, a);
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float final_speed = sqrtf(t.env.player.vel.x * t.env.player.vel.x +
                              t.env.player.vel.y * t.env.player.vel.y +
                              t.env.player.vel.z * t.env.player.vel.z);
    float recovery_sec = (recovery_step >= 0) ? recovery_step * 0.02f : -1.0f;
    int ok = (recovery_step >= 0) && (recovery_sec <= pass_sec);
    printf("%-22s spawn=%.0f m/s -> target=%.0f, recovered=%s, t=%.2fs (lim %.1fs), final=%.1f [%s]\n",
           name, start_speed, target_speed,
           (recovery_step >= 0) ? "yes" : "NO",
           (recovery_step >= 0) ? recovery_sec : -1.0f,
           pass_sec, final_speed, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_recovery_speed_low(void) {
    /* 60 -> 120 m/s with full throttle; physics needs ~40s at low-speed end. */
    return run_speed_recovery("recovery_speed_low:   ", 60.0f, 120.0f, 45.0f, 2300);
}
static int test_recovery_speed_high(void) {
    /* 150 -> 120 m/s with idle throttle; drag-dominated decel is slow. */
    return run_speed_recovery("recovery_speed_high:  ", 150.0f, 120.0f, 30.0f, 1500);
}

/* Altitude correction: spawn level at start_alt, drive to target_alt
 * via altitude -> vz cascade. PASS when altitude within tol within max_sec. */
static int run_altitude_recovery(const char* name,
                                  float start_alt, float target_alt,
                                  float pass_sec, int max_steps) {
    const float tol = 50.0f;  /* m */
    TestEnv t; setup_env(&t, 0);
    t.env.max_steps = max_steps + 100;
    force_with_ori(&t.env, 0, 0, start_alt, 100.0f, 0, 0,
                   1.0f, 0.0f, 0.0f, 0.0f, 0.5f);

    int recovery_step = -1;
    for (int step = 0; step < max_steps; step++) {
        Plane* p = &t.env.player;
        if (recovery_step < 0 && fabsf(p->pos.z - target_alt) < tol) {
            recovery_step = step;
        }
        float target_vz = ap_hold_altitude_vz_target(p, target_alt);
        float elev = ap_to_velocity(ap_hold_vz(p, target_vz));
        float ail = ap_to_velocity(ap_hold_bank(p, 0.0f));
        float thr = ap_hold_speed(p, 100.0f);
        float a[5] = {thr, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        recovery_log_row(name, step, p, a);
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float final_alt = t.env.player.pos.z;
    float recovery_sec = (recovery_step >= 0) ? recovery_step * 0.02f : -1.0f;
    int ok = (recovery_step >= 0) && (recovery_sec <= pass_sec);
    printf("%-22s spawn=%.0f m -> target=%.0f, recovered=%s, t=%.2fs (lim %.1fs), final=%.0f [%s]\n",
           name, start_alt, target_alt,
           (recovery_step >= 0) ? "yes" : "NO",
           (recovery_step >= 0) ? recovery_sec : -1.0f,
           pass_sec, final_alt, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_recovery_altitude_low(void) {
    /* 1000 m climb at 15 m/s climb rate cap = 67s minimum, allow 90s. */
    return run_altitude_recovery("recovery_altitude_low:", 500.0f, 1500.0f, 90.0f, 5000);
}
static int test_recovery_altitude_high(void) {
    return run_altitude_recovery("recovery_altitude_hi: ", 2500.0f, 1500.0f, 90.0f, 5000);
}

/* Heading correction: spawn flying along start_hdg, drive heading -> target_hdg.
 * Uses heading->bank cascade with ap_hold_bank_and_level. */
static int run_heading_recovery(const char* name,
                                 float start_hdg_deg, float target_hdg_deg,
                                 float pass_sec, int max_steps) {
    const float tol = 5.0f;  /* deg */
    TestEnv t; setup_env(&t, 0);
    t.env.max_steps = max_steps + 100;
    /* Plane starts level, flying along start_hdg. */
    float hdg_rad = start_hdg_deg * DEG;
    float V = 100.0f;
    float vx = V * cosf(hdg_rad);
    float vy = V * sinf(hdg_rad);
    /* Body fwd should match velocity direction — yaw the plane to start_hdg. */
    Quat q0 = quat_from_axis_angle(vec3(0.0f, 0.0f, 1.0f), hdg_rad);
    force_with_ori(&t.env, 0, 0, 1500, vx, vy, 0,
                   q0.w, q0.x, q0.y, q0.z, 0.5f);

    int recovery_step = -1;
    for (int step = 0; step < max_steps; step++) {
        Plane* p = &t.env.player;
        float heading = atan2f(p->vel.y, p->vel.x) * RAD;
        float err = target_hdg_deg - heading;
        while (err >  180.0f) err -= 360.0f;
        while (err < -180.0f) err += 360.0f;
        if (recovery_step < 0 && fabsf(err) < tol) {
            recovery_step = step;
        }
        float target_bank = ap_hold_heading_bank_target(p, target_hdg_deg);
        float elev_pos, ail_pos;
        ap_hold_bank_and_level(p, target_bank, &elev_pos, &ail_pos);
        float elev = ap_to_velocity(elev_pos);
        float ail = ap_to_velocity(ail_pos);
        float thr = ap_hold_speed(p, 100.0f);
        float a[5] = {thr, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        recovery_log_row(name, step, p, a);
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float final_hdg = atan2f(t.env.player.vel.y, t.env.player.vel.x) * RAD;
    float recovery_sec = (recovery_step >= 0) ? recovery_step * 0.02f : -1.0f;
    int ok = (recovery_step >= 0) && (recovery_sec <= pass_sec);
    printf("%-22s spawn=%+.0f deg -> target=%+.0f, recovered=%s, t=%.2fs (lim %.1fs), final=%+.0f [%s]\n",
           name, start_hdg_deg, target_hdg_deg,
           (recovery_step >= 0) ? "yes" : "NO",
           (recovery_step >= 0) ? recovery_sec : -1.0f,
           pass_sec, final_hdg, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_recovery_heading_left(void) {
    return run_heading_recovery("recovery_heading_left:", -90.0f, 0.0f, 30.0f, 2000);
}
static int test_recovery_heading_right(void) {
    return run_heading_recovery("recovery_heading_right",  90.0f, 0.0f, 30.0f, 2000);
}

/* Body roll-rate hold: command a target omega.x and check we reach it
 * within tolerance and stay there. */
static int run_rate_recovery(const char* name, int axis,
                              float target_rate_deg_s,
                              float pass_sec, int max_steps) {
    /* axis: 0 = roll (omega.x via aileron), 1 = pitch (omega.y via elevator) */
    const float tol = 10.0f;  /* deg/s */
    TestEnv t; setup_env(&t, 0);
    force_with_ori(&t.env, 0, 0, 2000, 120.0f, 0, 0,
                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    int recovery_step = -1;
    for (int step = 0; step < max_steps; step++) {
        Plane* p = &t.env.player;
        float current = (axis == 0 ? p->omega.x : p->omega.y) * RAD;
        if (recovery_step < 0 && fabsf(current - target_rate_deg_s) < tol) {
            recovery_step = step;
        }
        float ail = (axis == 0) ? ap_hold_roll_rate(p, target_rate_deg_s) : 0.0f;
        float elev = (axis == 1) ? ap_hold_pitch_rate(p, target_rate_deg_s) : 0.0f;
        float a[5] = {1.0f, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        recovery_log_row(name, step, p, a);
        t_step(&t);
        if (t.env.terminals[0]) break;
    }
    float final_rate = (axis == 0 ? t.env.player.omega.x : t.env.player.omega.y) * RAD;
    float recovery_sec = (recovery_step >= 0) ? recovery_step * 0.02f : -1.0f;
    int ok = (recovery_step >= 0) && (recovery_sec <= pass_sec);
    printf("%-22s axis=%s target=%+.0f deg/s, reached=%s, t=%.2fs (lim %.1fs), final=%+.1f [%s]\n",
           name, (axis == 0 ? "roll" : "pitch"), target_rate_deg_s,
           (recovery_step >= 0) ? "yes" : "NO",
           (recovery_step >= 0) ? recovery_sec : -1.0f,
           pass_sec, final_rate, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_recovery_roll_rate(void) {
    return run_rate_recovery("recovery_roll_rate:   ", 0, +60.0f, 1.5f, 200);
}
static int test_recovery_pitch_rate(void) {
    return run_rate_recovery("recovery_pitch_rate:  ", 1, +15.0f, 1.5f, 200);
}

typedef struct { const char* name; int (*fn)(void); } TestEntry;

/* Order matches test_flight_physics.py TESTS dict (1:1). */
static const TestEntry ALL_TESTS[] = {
    {"max_speed",                    test_max_speed},
    {"acceleration",                 test_acceleration},
    {"deceleration",                 test_deceleration},
    {"cruise_speed",                 test_cruise_speed},
    {"stall_speed",                  test_stall_speed},
    {"climb_rate",                   test_climb_rate},
    {"glide_ratio",                  test_glide_ratio},
    {"sustained_turn",               test_sustained_turn},
    {"turn_60",                      test_turn_60},
    {"pitch_direction",              test_pitch_direction},
    {"roll_direction",               test_roll_direction},
    {"rudder_only_turn",             test_rudder_only_turn},
    {"knife_edge_pull",              test_knife_edge_pull},
    {"knife_edge_flight",            test_knife_edge_flight},
    {"mode_weights",                 test_mode_weights},
    {"autopilot_enum_sync",          test_autopilot_enum_sync},
    {"autopilot_random_not_hardturn", test_autopilot_random_not_hardturn},
    {"autopilot_bounds_check",       test_autopilot_bounds_check},
    {"force_state_pid_reset",        test_force_state_pid_reset},
    {"g_level_flight",               test_g_level_flight},
    {"g_push_forward",               test_g_push_forward},
    {"g_pull_back",                  test_g_pull_back},
    {"g_limit_negative",             test_g_limit_negative},
    {"g_limit_positive",             test_g_limit_positive},
    {"gentle_pitch_control",         test_gentle_pitch_control},
    {"high_speed_pitch_oscillation", test_high_speed_pitch_oscillation},
    {"high_speed_roll_oscillation",  test_high_speed_roll_oscillation},
    {"speed_sweep_stability",        test_speed_sweep_stability},
    {"recovery_bank_left",           test_recovery_bank_left},
    {"recovery_bank_right",          test_recovery_bank_right},
    {"recovery_pitch_dive",          test_recovery_pitch_dive},
    {"recovery_pitch_climb",         test_recovery_pitch_climb},
    {"recovery_knife_edge_left",     test_recovery_knife_edge_left},
    {"recovery_knife_edge_right",    test_recovery_knife_edge_right},
    {"recovery_inverted",            test_recovery_inverted},
    {"recovery_steep_dive",          test_recovery_steep_dive},
    {"recovery_steep_climb",         test_recovery_steep_climb},
    {"recovery_speed_low",           test_recovery_speed_low},
    {"recovery_speed_high",          test_recovery_speed_high},
    {"recovery_altitude_low",        test_recovery_altitude_low},
    {"recovery_altitude_high",       test_recovery_altitude_high},
    {"recovery_heading_left",        test_recovery_heading_left},
    {"recovery_heading_right",       test_recovery_heading_right},
    {"recovery_roll_rate",           test_recovery_roll_rate},
    {"recovery_pitch_rate",          test_recovery_pitch_rate},
};
static const int N_TESTS = (int)(sizeof(ALL_TESTS) / sizeof(ALL_TESTS[0]));

static void print_usage(const char* prog) {
    printf("Usage: %s [--render] [--fps N] [--test NAME] [--list]\n\n", prog);
    printf("  --render        Open a raylib window and render every step.\n");
    printf("  --fps N         Frame rate when rendering (default 50; try 5-10 for slow-mo).\n");
    printf("  --test NAME     Run only the named test (otherwise runs all).\n");
    printf("  --list          List available test names and exit.\n");
    printf("  --log FILE      Write per-tick CSV telemetry from recovery tests to FILE.\n");
}

static void print_list(void) {
    for (int i = 0; i < N_TESTS; i++) printf("%s\n", ALL_TESTS[i].name);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--render")) {
            g_visual_render = 1;
        } else if (!strcmp(argv[i], "--fps") && i + 1 < argc) {
            g_visual_fps = atoi(argv[++i]);
            if (g_visual_fps < 1) g_visual_fps = 1;
        } else if (!strcmp(argv[i], "--test") && i + 1 < argc) {
            g_visual_only_test = argv[++i];
        } else if (!strcmp(argv[i], "--log") && i + 1 < argc) {
            const char* path = argv[++i];
            g_log_csv = fopen(path, "w");
            if (!g_log_csv) {
                fprintf(stderr, "could not open log file %s\n", path);
                return 1;
            }
        } else if (!strcmp(argv[i], "--list") || !strcmp(argv[i], "-l")) {
            print_list();
            return 0;
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    int fails = 0;
    int ran = 0;
    for (int i = 0; i < N_TESTS; i++) {
        if (g_visual_only_test && strcmp(ALL_TESTS[i].name, g_visual_only_test) != 0) continue;
        fails += ALL_TESTS[i].fn();
        ran++;
    }

    if (g_visual_only_test && ran == 0) {
        fprintf(stderr, "Unknown test: %s\n", g_visual_only_test);
        fprintf(stderr, "Use --list to see available test names.\n");
        return 1;
    }

    printf("\n%d hard failures\n", fails);
    if (g_visual_render) CloseWindow();
    if (g_log_csv) fclose(g_log_csv);
    return fails;
}
