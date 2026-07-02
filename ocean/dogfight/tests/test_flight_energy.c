/*
 * test_flight_energy.c - 1:1 port of test_flight_energy.py.
 *
 * 11 tests covering energy conservation, drag bleed, and E-M theory.
 *
 * All tests are "soft" — Python uses prints with [OK]/[CHECK]/[FAIL]
 * status and never asserts. Same here: each test_<name>() returns 0.
 * The PASS/FAIL annotation in the printed line is the substantive
 * result; main() returns 0.
 */
#include "test_common.h"

#define G 9.81f
#define MASS_KG 4082.0f
#define DEG (3.14159265f / 180.0f)
#define RAD (180.0f / 3.14159265f)

typedef struct {
    float speed;
    float alt;
    float ke;
    float pe;
    float total;
    float es;     /* specific energy [m] = h + v^2/(2g) */
    float vz;
} EnergyState;

static EnergyState energy_state(const Plane* p) {
    EnergyState e;
    float V = norm3(p->vel);
    e.speed = V;
    e.alt = p->pos.z;
    e.ke = 0.5f * MASS_KG * V * V;
    e.pe = MASS_KG * G * e.alt;
    e.total = e.ke + e.pe;
    e.es = e.alt + (V * V) / (2.0f * G);
    e.vz = p->vel.z;
    return e;
}

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

/* ============================================================
 * test order matches test_flight_energy.py TESTS dict
 * ============================================================ */

static int test_sideslip_drag(void) {
    float results[2];
    const float rudder_inputs[2] = {0.0f, 1.0f};
    for (int ti = 0; ti < 2; ti++) {
        TestEnv t; setup_env(&t, 0);
        force_level(&t.env, 0, 0, 1500, 120, 0, 0, 0.0f);
        EnergyState ini = energy_state(&t.env.player);
        float prev_roll = 0.0f;
        for (int s = 0; s < 150; s++) {
            Plane* p = &t.env.player;
            Vec3 up = plane_up(p);
            float roll = atan2f(up.y, up.z);
            float aileron = 1.0f * (-roll) - 0.05f * (roll - prev_roll) / 0.02f;
            aileron = clip_unit(aileron);
            prev_roll = roll;
            float a[5] = {-1.0f, 0.0f, aileron, rudder_inputs[ti], 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            c_step(&t.env);
        }
        EnergyState fin = energy_state(&t.env.player);
        results[ti] = ini.es - fin.es;
    }
    float diff = results[1] - results[0];
    int ok = (results[1] > results[0] + 5.0f);
    printf("sideslip_drag:      no_rudder=%.0fm, full_rudder=%.0fm, diff=%+.0fm [%s]\n",
           results[0], results[1], diff, ok ? "OK" : "FAIL");
    return 0;
}

static int test_knife_edge_pull_energy(void) {
    TestEnv t; setup_env(&t, 0);
    float roll_90 = 90.0f * DEG;
    float qw = cosf(roll_90 / 2.0f);
    float qx = -sinf(roll_90 / 2.0f);
    force_with_ori(&t.env, 0, 0, 2000, 150, 0, 0, qw, qx, 0, 0, 0.0f);
    EnergyState ini = energy_state(&t.env.player);

    EnergyState fin = ini;
    for (int step = 0; step < 150; step++) {
        float a[5] = {-1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        fin = energy_state(&t.env.player);
        if (t.env.terminals[0]) break;
    }
    float ke_loss = ini.ke - fin.ke;
    float pe_loss = ini.pe - fin.pe;
    float total_loss = ini.total - fin.total;
    float es_loss = ini.es - fin.es;
    int ke_dropped = ke_loss > 0.0f;
    int pe_dropped = pe_loss > 0.0f;
    int total_dropped = total_loss > 0.0f;
    int significant = es_loss > 100.0f;
    int ok = ke_dropped && pe_dropped && total_dropped && significant;
    printf("knife_pull_E:       KE=%s, PE=%s, Es_loss=%.0fm [%s]\n",
           ke_dropped ? "DROP" : "RISE",
           pe_dropped ? "DROP" : "RISE",
           es_loss, ok ? "OK" : "FAIL");
    return 0;
}

static int test_energy_level_flight(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 1500, 120, 0, 0, 0.5f);
    EnergyState ini = energy_state(&t.env.player);

    float energies[501]; int ne = 0;
    energies[ne++] = ini.es;
    float prev_vz = 0.0f;
    const float kp = 0.001f, kd = 0.001f;
    for (int step = 0; step < 500; step++) {
        Plane* p = &t.env.player;
        float vz = p->vel.z;
        float elevator = -kp * vz - kd * (vz - prev_vz) / 0.02f;
        if (elevator >  0.2f) elevator =  0.2f;
        if (elevator < -0.2f) elevator = -0.2f;
        prev_vz = vz;
        float a[5] = {0.0f, elevator, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        energies[ne++] = energy_state(&t.env.player).es;
    }
    EnergyState fin = energy_state(&t.env.player);
    float es_change = fin.es - ini.es;
    float es_std = arr_std(energies, ne);
    int stable = (fabsf(es_change) < 50.0f) && (es_std < 30.0f);
    printf("energy_level:       Es_change=%+.1fm, std=%.1fm [%s]\n",
           es_change, es_std, stable ? "OK" : "CHECK");
    return 0;
}

static int test_energy_dive(void) {
    TestEnv t; setup_env(&t, 0);
    float pitch_down = -45.0f * DEG;
    float qw = cosf(pitch_down / 2.0f);
    float qy = -sinf(pitch_down / 2.0f);
    force_with_ori(&t.env, 0, 0, 2500, 80, 0, 0, qw, 0, qy, 0, 0.0f);
    EnergyState ini = energy_state(&t.env.player);

    for (int step = 0; step < 200; step++) {
        float a[5] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        if (t.env.terminals[0] || t.env.player.pos.z < 500.0f) break;
    }
    EnergyState fin = energy_state(&t.env.player);
    float speed_gain = fin.speed - ini.speed;
    float alt_loss = ini.alt - fin.alt;
    float ke_gain = fin.ke - ini.ke;
    float pe_loss = ini.pe - fin.pe;
    float es_loss = ini.es - fin.es;
    int ke_inc = ke_gain > 0.0f;
    int pe_dec = pe_loss > 0.0f;
    int es_ok = (es_loss > 0.0f) && (es_loss < alt_loss * 0.3f);
    int ok = ke_inc && pe_dec && es_ok;
    float pct = (alt_loss > 0.0f) ? (100.0f * es_loss / alt_loss) : 0.0f;
    printf("energy_dive:        speed+%.0f, alt-%.0f, Es_loss=%.0fm (%.0f%% to drag) [%s]\n",
           speed_gain, alt_loss, es_loss, pct, ok ? "OK" : "CHECK");
    return 0;
}

static int test_energy_climb(void) {
    TestEnv t; setup_env(&t, 0);
    float pitch_up = 30.0f * DEG;
    float qw = cosf(-pitch_up / 2.0f);
    float qy = sinf(-pitch_up / 2.0f);
    force_with_ori(&t.env, 0, 0, 1000, 140, 0, 0, qw, 0, qy, 0, 1.0f);
    EnergyState ini = energy_state(&t.env.player);

    for (int step = 0; step < 300; step++) {
        float a[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        if (t.env.terminals[0]) break;
    }
    EnergyState fin = energy_state(&t.env.player);
    float speed_loss = ini.speed - fin.speed;
    float alt_gain = fin.alt - ini.alt;
    float es_change = fin.es - ini.es;
    int energy_ok = es_change > -50.0f;
    int alt_ok = alt_gain > 100.0f;
    int ok = energy_ok && alt_ok;
    printf("energy_climb:       speed-%.0f, alt+%.0f, Es_change=%+.0fm [%s]\n",
           speed_loss, alt_gain, es_change, ok ? "OK" : "CHECK");
    return 0;
}

static int test_energy_turn_bleed(void) {
    TestEnv t; setup_env(&t, 0);
    float bank = 60.0f * DEG;
    float qw = cosf(bank / 2.0f);
    float qx = -sinf(bank / 2.0f);
    force_with_ori(&t.env, 0, 0, 1500, 120, 0, 0, qw, qx, 0, 0, 1.0f);
    EnergyState ini = energy_state(&t.env.player);

    float prev_vz = 0.0f, prev_bank_err = 0.0f;
    int ne = 0;
    for (int step = 0; step < 250; step++) {
        Plane* p = &t.env.player;
        float vz = p->vel.z;
        Vec3 up = plane_up(p);
        float bank_actual = acosf(clip_unit(up.z));

        float elev = -0.05f * (-vz) + 0.005f * (vz - prev_vz) / 0.02f;
        elev = clip_unit(elev);
        prev_vz = vz;

        float bank_err = bank - bank_actual;
        float ail = -2.0f * bank_err - 0.1f * (bank_err - prev_bank_err) / 0.02f;
        ail = clip_unit(ail);
        prev_bank_err = bank_err;

        float a[5] = {1.0f, elev, ail, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        ne++;
    }
    EnergyState fin = energy_state(&t.env.player);
    float es_loss = ini.es - fin.es;
    float t_elapsed = (float)ne * 0.02f;
    float bleed = es_loss / t_elapsed;
    int bleeding = es_loss > 10.0f;
    printf("energy_turn:        Es_loss=%.0fm in %.1fs, bleed=%.1f m/s [%s]\n",
           es_loss, t_elapsed, bleed, bleeding ? "OK" : "CHECK");
    return 0;
}

static int test_energy_loop(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 1500, 150, 0, 0, 1.0f);
    EnergyState ini = energy_state(&t.env.player);

    float g_max = 0.0f;
    for (int step = 0; step < 200; step++) {
        float a[5] = {1.0f, -0.8f, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        if (t.env.player.g_force > g_max) g_max = t.env.player.g_force;
        Vec3 fwd = plane_fwd(&t.env.player);
        if (step > 50 && fwd.z > -0.1f && fwd.x > 0.5f) break;
    }
    EnergyState fin = energy_state(&t.env.player);
    float es_loss = ini.es - fin.es;
    float pct = 100.0f * es_loss / ini.es;
    int ok = (pct > 5.0f) && (pct < 35.0f);
    printf("energy_loop:        Es_loss=%.0fm (%.1f%%), max_G=%.1f [%s]\n",
           es_loss, pct, g_max, ok ? "OK" : "CHECK");
    return 0;
}

static int test_energy_split_s(void) {
    TestEnv t; setup_env(&t, 0);
    force_level(&t.env, 0, 0, 2500, 100, 0, 0, 0.5f);
    EnergyState ini = energy_state(&t.env.player);

    /* Phase 1: half roll */
    for (int step = 0; step < 25; step++) {
        float a[5] = {0.5f, 0.0f, 1.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
    }
    /* Phase 2: pull through */
    for (int step = 0; step < 150; step++) {
        float a[5] = {1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        Vec3 fwd = plane_fwd(&t.env.player);
        if (fwd.z > 0.3f && t.env.player.pos.z < ini.alt) break;
        if (t.env.terminals[0]) break;
    }
    EnergyState fin = energy_state(&t.env.player);
    float speed_gain = fin.speed - ini.speed;
    float alt_loss = ini.alt - fin.alt;
    float es_loss = ini.es - fin.es;
    int ok = (speed_gain > 20.0f) && (alt_loss > 200.0f);
    printf("energy_split_s:     speed+%.0f, alt-%.0f, Es_loss=%.0fm [%s]\n",
           speed_gain, alt_loss, es_loss, ok ? "OK" : "CHECK");
    return 0;
}

static int test_energy_zoom(void) {
    TestEnv t; setup_env(&t, 0);
    float pitch_up = 90.0f * DEG;
    float qw = cosf(-pitch_up / 2.0f);
    float qy = sinf(-pitch_up / 2.0f);
    float V = 150.0f;
    force_with_ori(&t.env, 0, 0, 500, 0, 0, V, qw, 0, qy, 0, 0.0f);
    EnergyState ini = energy_state(&t.env.player);
    float theoretical = V * V / (2.0f * G);
    float max_alt = ini.alt;
    for (int step = 0; step < 400; step++) {
        float a[5] = {-1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        memcpy(t.env.actions, a, sizeof(a));
        c_step(&t.env);
        if (t.env.player.pos.z > max_alt) max_alt = t.env.player.pos.z;
        if (t.env.player.vel.z < 0.0f) break;
    }
    float gain = max_alt - ini.alt;
    float eff = 100.0f * gain / theoretical;
    int ok = eff > 65.0f;
    printf("energy_zoom:        alt_gain=%.0fm (theory=%.0fm, eff=%.0f%%) [%s]\n",
           gain, theoretical, eff, ok ? "OK" : "CHECK");
    return 0;
}

static int test_energy_throttle(void) {
    float results[3];
    const float throttles[3] = {1.0f, 0.0f, -1.0f};
    for (int ti = 0; ti < 3; ti++) {
        TestEnv t; setup_env(&t, 0);
        force_level(&t.env, 0, 0, 1500, 120, 0, 0, 0.5f);
        EnergyState ini = energy_state(&t.env.player);
        float prev_vz = 0.0f;
        for (int step = 0; step < 200; step++) {
            Plane* p = &t.env.player;
            float vz = p->vel.z;
            float elev = -0.001f * vz - 0.001f * (vz - prev_vz) / 0.02f;
            prev_vz = vz;
            float a[5] = {throttles[ti], elev, 0.0f, 0.0f, 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            c_step(&t.env);
        }
        EnergyState fin = energy_state(&t.env.player);
        results[ti] = fin.es - ini.es;
    }
    int full_pos = results[0] > results[1];
    int zero_neg = results[2] < results[1];
    int ok = full_pos && zero_neg;
    printf("energy_throttle:    full=%+.0fm, half=%+.0fm, zero=%+.0fm [%s]\n",
           results[0], results[1], results[2], ok ? "OK" : "CHECK");
    return 0;
}

static int test_energy_g_bleed(void) {
    float es_losses[3];
    float avg_gs[3];
    const float elevs[3] = {-0.3f, -0.6f, -1.0f};
    const char* labels[3] = {"2G", "4G", "6G"};
    for (int gi = 0; gi < 3; gi++) {
        TestEnv t; setup_env(&t, 0);
        force_level(&t.env, 0, 0, 2000, 150, 0, 0, 1.0f);
        EnergyState ini = energy_state(&t.env.player);
        float gs[50]; int ng = 0;
        for (int step = 0; step < 50; step++) {
            float a[5] = {1.0f, elevs[gi], 0.0f, 0.0f, 0.0f};
            memcpy(t.env.actions, a, sizeof(a));
            c_step(&t.env);
            gs[ng++] = t.env.player.g_force;
        }
        EnergyState fin = energy_state(&t.env.player);
        es_losses[gi] = ini.es - fin.es;
        avg_gs[gi] = arr_mean(gs, ng);
    }
    int rising = (es_losses[0] < es_losses[1]) && (es_losses[1] < es_losses[2]);
    printf("energy_g_bleed:\n");
    for (int gi = 0; gi < 3; gi++) {
        printf("  %s: avg_G=%.1f, Es_loss=%.0fm\n",
               labels[gi], avg_gs[gi], es_losses[gi]);
    }
    printf("  Higher G = more bleed: %s [%s]\n",
           rising ? "true" : "false", rising ? "OK" : "CHECK");
    return 0;
}

int main(void) {
    int fails = 0;
    fails += test_sideslip_drag();
    fails += test_knife_edge_pull_energy();
    fails += test_energy_level_flight();
    fails += test_energy_dive();
    fails += test_energy_climb();
    fails += test_energy_turn_bleed();
    fails += test_energy_loop();
    fails += test_energy_split_s();
    fails += test_energy_zoom();
    fails += test_energy_throttle();
    fails += test_energy_g_bleed();
    printf("\n%d hard failures\n", fails);
    return fails;
}
