// Structural adaptation of DrussGT 3.1.4159 (jk.mega.DrussGT)
// by Julian Kent / Skilgannon — open source (jar in third_party/).
//
// Core systems (speed-aware, not a full mega port):
//   * Go-to wave surfing with SHORT path-sim → wave-hit GF (official idea)
//   * One kNN danger profile per wave (not kNN-per-candidate)
//   * First + light second wave
//   * Visit flattener + hit-weighted surf samples
//   * Gunheat-filtered waves + imaginary pre-fire waves
//   * Visit-count DC gun (wave-pass GF logs) + cold linear/circular blend
//   * Fair last-scan aim only
//
// Intentionally omitted for train SPS: multi-buffer VCS, precise GF ranges,
// dual guns, bullet shadows, shielding.
//
// License: keep open-source; credit Skilgannon / DrussGT.

#ifndef ROBOCODE_AGENT_DRUSSGT_H
#define ROBOCODE_AGENT_DRUSSGT_H

#include "agent_common.h"

#ifndef DGT_KNN_K
#define DGT_KNN_K 6
#endif
#ifndef DGT_GUN_K
#define DGT_GUN_K 10
#endif
#ifndef DGT_GOTO_CANDS
#define DGT_GOTO_CANDS 8
#endif
#ifndef DGT_SIM_STEPS
#define DGT_SIM_STEPS 12
#endif
#ifndef DGT_BEST_DIST
#define DGT_BEST_DIST 450.0f
#endif
#ifndef DGT_DANGER_BINS
#define DGT_DANGER_BINS 17
#endif

// Manhattan weights (official metric). 8 attributes.
static const float DGT_FEAT_W[DGT_FEATS] = {
    5.00f,  // distance / 900
    4.00f,  // |lat vel| / 8
    2.00f,  // adv vel / 8
    3.00f,  // 1/(1+k*tsdc)
    2.50f,  // 1/(1+k*tsdecel)
    2.00f,  // accel
    3.00f,  // wall
    2.00f,  // dist-last-10 / 80
};

static inline float dgt_norm_time(float t, float k) {
    return 1.0f / (1.0f + k * fmaxf(t, 0.0f));
}

static inline float dgt_kernel(float dgf, float inv_two_s2) {
    float x = dgf * dgf * inv_two_s2;
    return 1.0f / (1.0f + x + 0.5f * x * x);
}

static inline void dgt_features(Robot* bot, float tgt_x, float tgt_y, Robocode* env,
                                int tick, int last_dir_change, int last_decel,
                                float last_v, float dist_last10,
                                float out[DGT_FEATS]) {
    float dx = bot->x - tgt_x, dy = bot->y - tgt_y;
    float dist = sqrtf(dx * dx + dy * dy);
    float inv = (dist > 1e-6f) ? 1.0f / dist : 1.0f;
    float ux = dx * inv, uy = dy * inv;
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float wall_min = fminf(fminf(bot->x, env->width - bot->x),
                           fminf(bot->y, env->height - bot->y));
    float wall_half = fmaxf(fminf(env->width, env->height) * 0.5f, 1.0f);
    out[0] = rb_clampf(dist * (1.0f / 900.0f), 0.0f, 1.2f);
    out[1] = fabsf(-bvx * uy + bvy * ux) * 0.125f;
    out[2] = rb_clampf((bvx * ux + bvy * uy) * 0.125f, -1.0f, 1.0f);
    out[3] = dgt_norm_time((float)(tick - last_dir_change), 0.08f);
    out[4] = dgt_norm_time((float)(tick - last_decel), 0.08f);
    out[5] = rb_clampf((bot->v - last_v) * 0.5f, -1.0f, 1.0f);
    out[6] = rb_clampf(wall_min / wall_half, 0.0f, 1.0f);
    out[7] = rb_clampf(dist_last10 * (1.0f / 80.0f), 0.0f, 1.5f);
}

static inline float dgt_feat_dist(const float* a, const float* b) {
    float d = 0.0f;
    for (int f = 0; f < DGT_FEATS; f++) {
        d += fabsf((a[f] - b[f]) * DGT_FEAT_W[f]);
    }
    return d;
}

static inline void dgt_knn_add(float knn_feats[][DGT_FEATS], float* knn_gf,
                               int* knn_n, int* knn_head, int cap,
                               const float feats[DGT_FEATS], float gf) {
    int slot = *knn_head;
    memcpy(knn_feats[slot], feats, DGT_FEATS * sizeof(float));
    knn_gf[slot] = gf;
    *knn_head = (slot + 1) % cap;
    if (*knn_n < cap) (*knn_n)++;
}

static int dgt_knn_query(const float feats[DGT_FEATS],
                         const float knn_feats[][DGT_FEATS],
                         int knn_n, int k_max,
                         float* best_d, int* best_i) {
    for (int k = 0; k < k_max; k++) {
        best_d[k] = 1e18f;
        best_i[k] = -1;
    }
    if (knn_n <= 0) return 0;
    for (int n = 0; n < knn_n; n++) {
        float d = dgt_feat_dist(feats, knn_feats[n]);
        if (d >= best_d[k_max - 1]) continue;
        int k = k_max - 1;
        while (k > 0 && d < best_d[k - 1]) {
            best_d[k] = best_d[k - 1];
            best_i[k] = best_i[k - 1];
            k--;
        }
        best_d[k] = d;
        best_i[k] = n;
    }
    int count = 0;
    for (int k = 0; k < k_max; k++) if (best_i[k] >= 0) count++;
    return count;
}

static void dgt_build_danger_profile(const float feats[DGT_FEATS],
                                     const float knn_feats[][DGT_FEATS],
                                     const float* knn_gf, int knn_n,
                                     float dens[DGT_DANGER_BINS]) {
    for (int b = 0; b < DGT_DANGER_BINS; b++) dens[b] = 0.0f;
    if (knn_n <= 0) {
        for (int b = 0; b < DGT_DANGER_BINS; b++) {
            float gf = -1.0f + 2.0f * (float)b / (float)(DGT_DANGER_BINS - 1);
            dens[b] = 0.25f / (1.0f + 8.0f * gf * gf);
        }
        return;
    }
    float best_d[DGT_KNN_K];
    int best_i[DGT_KNN_K];
    int nk = dgt_knn_query(feats, knn_feats, knn_n, DGT_KNN_K, best_d, best_i);
    const float inv_two_s2 = 1.0f / (2.0f * 0.14f * 0.14f);
    for (int k = 0; k < nk; k++) {
        float w = 1.0f / (1.0f + best_d[k]);
        float g0 = knn_gf[best_i[k]];
        for (int b = 0; b < DGT_DANGER_BINS; b++) {
            float gf = -1.0f + 2.0f * (float)b / (float)(DGT_DANGER_BINS - 1);
            dens[b] += w * dgt_kernel(gf - g0, inv_two_s2);
        }
    }
}

static inline float dgt_profile_danger(const float dens[DGT_DANGER_BINS], float gf) {
    gf = rb_clampf(gf, -1.0f, 1.0f);
    float t = (gf + 1.0f) * 0.5f * (float)(DGT_DANGER_BINS - 1);
    int i0 = (int)t;
    if (i0 < 0) i0 = 0;
    if (i0 >= DGT_DANGER_BINS - 1) return dens[DGT_DANGER_BINS - 1];
    float f = t - (float)i0;
    return dens[i0] * (1.0f - f) + dens[i0 + 1] * f;
}

static float dgt_gun_best_gf(const float feats[DGT_FEATS],
                             const float knn_feats[][DGT_FEATS],
                             const float* knn_gf, int knn_n) {
    if (knn_n <= 0) return 0.0f;
    float best_d[DGT_GUN_K];
    int best_i[DGT_GUN_K];
    int nk = dgt_knn_query(feats, knn_feats, knn_n, DGT_GUN_K, best_d, best_i);
    if (nk <= 0) return 0.0f;
    const float inv_two_s2 = 1.0f / (2.0f * 0.10f * 0.10f);
    float best_gf = 0.0f;
    float best_dens = -1.0f;
    for (int pass = 0; pass < 2; pass++) {
        int n_eval = (pass == 0) ? nk : 11;
        for (int e = 0; e < n_eval; e++) {
            float gf = (pass == 0)
                ? knn_gf[best_i[e]]
                : (-1.0f + 0.2f * (float)e);
            float dens = 0.0f;
            for (int k = 0; k < nk; k++) {
                float w = 1.0f / (1.0f + best_d[k]);
                dens += w * dgt_kernel(gf - knn_gf[best_i[k]], inv_two_s2);
            }
            if (dens > best_dens) {
                best_dens = dens;
                best_gf = gf;
            }
        }
    }
    return rb_clampf(best_gf, -1.0f, 1.0f);
}

static inline float dgt_bullet_power(Robot* bot, float enemy_energy, float dist,
                                     float enemy_fp, int bullets_hit,
                                     int bullets_fired) {
    float base = 1.95f;
    float hitrate = (bullets_fired > 4)
        ? (float)bullets_hit / (float)bullets_fired : 0.0f;
    if (hitrate > 0.33f || dist < 180.0f) base = 2.95f;
    if (dist > 600.0f && hitrate < 0.25f) base = 1.95f;
    float power = base;
    power = fminf(power, fmaxf(0.1f, ((float)bot->energy - 0.1f) * 0.25f));
    power = fminf(power, fmaxf(0.1f, enemy_energy * 0.25f));
    if (enemy_fp > 0.1f && enemy_fp < power && bot->energy < 40.0f) {
        power = fmaxf(enemy_fp, 0.15f);
    }
    if (power >= 2.4f) power = 2.95f;
    else if (power >= 1.5f) power = 1.95f;
    else if (power >= 0.9f) power = 0.95f;
    else if (power >= 0.4f) power = 0.45f;
    else power = 0.15f;
    return rb_clampf(power, 0.1f, fminf(3.0f, (float)bot->energy - 0.1f));
}

// Fair linear lead from last scan only.
static inline float dgt_linear_aim(Robot* bot, BotMem* m, float bspeed) {
    float dx = m->last_x - bot->x;
    float dy = m->last_y - bot->y;
    float dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
    float dt = dist / fmaxf(bspeed, 0.1f);
    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    float px = m->last_x + tvx * dt;
    float py = m->last_y + tvy * dt;
    // One refine pass.
    dx = px - bot->x; dy = py - bot->y;
    dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
    dt = dist / fmaxf(bspeed, 0.1f);
    return rb_abs_bearing_deg(bot->x, bot->y,
                              m->last_x + tvx * dt, m->last_y + tvy * dt);
}

// Fair circular lead from last scan (constant lateral orbit estimate).
static inline float dgt_circular_aim(Robot* bot, BotMem* m, float bspeed) {
    float dx = m->last_x - bot->x;
    float dy = m->last_y - bot->y;
    float dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
    float dt = dist / fmaxf(bspeed, 0.1f);
    float inv = 1.0f / dist;
    float ux = dx * inv, uy = dy * inv;
    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    float lat = -tvx * uy + tvy * ux;
    float ang_vel = lat / dist;
    float bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float new_bearing = bearing + ang_vel * dt * RB_R2D;
    return rb_abs_bearing_deg(bot->x, bot->y,
                              bot->x + dist * cos_deg(new_bearing),
                              bot->y + dist * sin_deg(new_bearing));
}

static inline float dgt_hist_dist10(BotMem* m) {
    if (m->dgt_hist_n < 2) return 0.0f;
    int span = m->dgt_hist_n < 10 ? m->dgt_hist_n : 10;
    int oldest = (m->dgt_hist_head - span + DGT_HIST) % DGT_HIST;
    int latest = (m->dgt_hist_head - 1 + DGT_HIST) % DGT_HIST;
    return rb_dist(m->dgt_hist_x[oldest], m->dgt_hist_y[oldest],
                   m->dgt_hist_x[latest], m->dgt_hist_y[latest]);
}

static void dgt_fallback_move(Robocode* env, Robot* bot, BotMem* m) {
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float distance = rb_dist(bot->x, bot->y, m->last_x, m->last_y);
    float best_x = bot->x, best_y = bot->y;
    float best_score = -1e18f;
    for (int di = 0; di < 2; di++) {
        float dir = (di == 0) ? 1.0f : -1.0f;
        for (int t = 0; t < 6; t++) {
            float orbit = 60.0f + 16.0f * (float)t;
            float ang = (abs_bearing + dir * orbit) * RB_D2R;
            float step = rb_clampf(distance * 0.4f + 100.0f, 90.0f, 190.0f);
            float x = bot->x + step * cosf(ang);
            float y = bot->y + step * sinf(ang);
            if (x < 30.0f || x > env->width - 30.0f ||
                    y < 30.0f || y > env->height - 30.0f) continue;
            float d = rb_dist(x, y, m->last_x, m->last_y);
            float score = -fabsf(d - DGT_BEST_DIST);
            if (distance < 220.0f) score += 0.7f * d;
            float wall = fminf(fminf(x, env->width - x), fminf(y, env->height - y));
            score += 0.35f * wall;
            if (score > best_score) {
                best_score = score;
                best_x = x;
                best_y = y;
            }
        }
    }
    rb_drive_to(env, bot, best_x, best_y);
}

static inline float dgt_point_gf(DGTWave* w, float x, float y) {
    float bearing = rb_abs_bearing_deg(w->ox, w->oy, x, y);
    float gf_raw = rb_norm_deg(bearing - w->head_on);
    float mea_deg = fmaxf(asinf(fminf(8.0f / w->speed, 1.0f)) * RB_R2D, 0.1f);
    return rb_clampf((gf_raw / mea_deg) * (float)w->lat_sign, -1.5f, 1.5f);
}

// Short path-sim: drive toward (tx,ty) until wave hits; return hit GF.
// Uses a cheap constant-speed slide (no full accel model) — captures go-to
// hit location for imminent waves without 120-step turn sim cost.
static float dgt_wave_hit_gf(Robot* bot, DGTWave* w, float tx, float ty,
                             int tick_now, float* out_hit_dist) {
    float x = bot->x, y = bot->y;
    float radius = (float)(tick_now - w->fire_tick) * w->speed;
    float dx0 = tx - x, dy0 = ty - y;
    float travel = sqrtf(dx0 * dx0 + dy0 * dy0);
    float spd = 6.5f;  // typical go-to cruise (between accel phases)
    float ux = 0.0f, uy = 0.0f;
    if (travel > 1.0f) {
        ux = dx0 / travel;
        uy = dy0 / travel;
    }
    for (int step = 0; step < DGT_SIM_STEPS; step++) {
        float ddx = x - w->ox, ddy = y - w->oy;
        float dist = sqrtf(ddx * ddx + ddy * ddy);
        if (radius + w->speed >= dist - 18.0f) {
            if (out_hit_dist) *out_hit_dist = dist;
            return dgt_point_gf(w, x, y);
        }
        // Stop sliding once near destination.
        float remx = tx - x, remy = ty - y;
        if (remx * remx + remy * remy < 64.0f) {
            x = tx;
            y = ty;
        } else {
            x += ux * spd;
            y += uy * spd;
        }
        radius += w->speed;
    }
    if (out_hit_dist) {
        float ddx = x - w->ox, ddy = y - w->oy;
        *out_hit_dist = sqrtf(ddx * ddx + ddy * ddy);
    }
    return dgt_point_gf(w, x, y);
}

// Go-to surf: path-sim hit GF + profile dens. First wave full sim; second light.
static void dgt_surf_goto(Robocode* env, Robot* bot, BotMem* m) {
    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float distance = fmaxf(rb_dist(bot->x, bot->y, m->last_x, m->last_y), 1.0f);

    // Collect active waves sorted by TTI (soonest first). Prefer real over imag.
    int idx[DGT_WAVES];
    float tti_a[DGT_WAVES];
    int n = 0;
    for (int wi = 0; wi < DGT_WAVES; wi++) {
        DGTWave* w = &m->dgt_waves[wi];
        if (!w->active) continue;
        float tti = (rb_dist(bot->x, bot->y, w->ox, w->oy)
                    - (m->tick - w->fire_tick) * w->speed) / fmaxf(w->speed, 0.1f);
        // Prefer real waves: inflate imaginary TTI slightly in sort key only.
        float key = tti + (w->imaginary ? 50.0f : 0.0f);
        idx[n] = wi;
        tti_a[n] = key;
        n++;
    }
    if (n == 0) {
        dgt_fallback_move(env, bot, m);
        return;
    }
    for (int a = 0; a < n; a++) {
        for (int b = a + 1; b < n; b++) {
            if (tti_a[b] < tti_a[a]) {
                float tt = tti_a[a]; tti_a[a] = tti_a[b]; tti_a[b] = tt;
                int ti = idx[a]; idx[a] = idx[b]; idx[b] = ti;
            }
        }
    }
    int surf_n = n > 2 ? 2 : n;
    DGTWave* w0 = &m->dgt_waves[idx[0]];
    DGTWave* w1 = (surf_n > 1) ? &m->dgt_waves[idx[1]] : NULL;

    float dens0[DGT_DANGER_BINS];
    dgt_build_danger_profile(w0->feats, m->dgt_surf_feats, m->dgt_surf_gf,
                             m->dgt_surf_n, dens0);
    // Second wave reuses dens0 (enemy gun state usually similar; saves a kNN).

    float best_x = bot->x, best_y = bot->y;
    float best_danger = 1e18f;
    float p0 = w0->power > 0.0f ? w0->power : ((20.0f - w0->speed) / 3.0f);
    float dmg0 = 1.0f + 0.12f * (4.0f * p0);
    // Path-sim only for imminent first waves; far waves use point GF.
    float tti0 = (rb_dist(bot->x, bot->y, w0->ox, w0->oy)
                 - (m->tick - w0->fire_tick) * w0->speed) / fmaxf(w0->speed, 0.1f);
    int do_sim = (tti0 < 55.0f && !w0->imaginary);

    for (int c = 0; c < DGT_GOTO_CANDS; c++) {
        float t = (float)c / (float)(DGT_GOTO_CANDS - 1);
        float gf_hint = -1.05f + 2.1f * t;
        float ea = abs_bearing + 180.0f + gf_hint * 55.0f;
        float ed = rb_clampf(0.5f * distance + 0.5f * DGT_BEST_DIST, 140.0f, 520.0f);
        if (distance < 200.0f) ed = fmaxf(ed, 280.0f);
        float x = m->last_x + ed * cos_deg(ea);
        float y = m->last_y + ed * sin_deg(ea);
        if ((c & 1) == 0) {
            float orbit = abs_bearing + 90.0f * (gf_hint >= 0.0f ? 1.0f : -1.0f)
                        + gf_hint * 30.0f;
            float rad = rb_clampf(0.4f * distance + 90.0f, 80.0f, 220.0f);
            x = bot->x + rad * cos_deg(orbit);
            y = bot->y + rad * sin_deg(orbit);
        }
        if (x < 28.0f || x > env->width - 28.0f ||
                y < 28.0f || y > env->height - 28.0f) continue;

        float hit_dist = rb_dist(x, y, w0->ox, w0->oy);
        float egf0 = do_sim
            ? dgt_wave_hit_gf(bot, w0, x, y, m->tick, &hit_dist)
            : dgt_point_gf(w0, x, y);
        float danger = dmg0 * dgt_profile_danger(dens0, egf0);
        if (w0->imaginary) danger *= 0.55f;

        if (w1) {
            float egf1 = dgt_point_gf(w1, x, y);
            float p1 = w1->power > 0.0f ? w1->power : ((20.0f - w1->speed) / 3.0f);
            float dmg1 = 1.0f + 0.12f * (4.0f * p1);
            float wgt = 0.32f * dmg1;
            if (w1->imaginary) wgt *= 0.5f;
            danger += wgt * dgt_profile_danger(dens0, egf1);
        }

        float d_en = rb_dist(x, y, m->last_x, m->last_y);
        danger += 0.001f * fabsf(d_en - DGT_BEST_DIST);
        danger += 0.0012f * fmaxf(0.0f, 300.0f - hit_dist);
        if (danger < best_danger) {
            best_danger = danger;
            best_x = x;
            best_y = y;
        }
    }
    m->dgt_goto_x = best_x;
    m->dgt_goto_y = best_y;
    rb_drive_to(env, bot, best_x, best_y);
}

static void dgt_update_gun_waves(BotMem* m) {
    for (int i = 0; i < DGT_GUN_WAVES; i++) {
        DGTGunWave* w = &m->dgt_gun_waves[i];
        if (!w->active) continue;
        w->dist_traveled += w->speed;
        float d = rb_dist(w->ox, w->oy, m->last_x, m->last_y);
        if (w->dist_traveled + w->speed >= d) {
            float bearing = rb_abs_bearing_deg(w->ox, w->oy, m->last_x, m->last_y);
            float gf = rb_clampf(
                (rb_norm_deg(bearing - w->abs_bearing) / fmaxf(w->mea, 0.1f)) * w->lat_dir,
                -1.2f, 1.2f);
            dgt_knn_add(m->dgt_gun_feats, m->dgt_gun_gf, &m->dgt_gun_n,
                        &m->dgt_gun_head, DGT_GUN_CAP, w->feats, gf);
            m->dgt_current_gf = gf;
            w->active = 0;
        } else if (w->dist_traveled > 1200.0f) {
            w->active = 0;
        }
    }
}

static void dgt_expire_and_flatten(BotMem* m, Robot* bot) {
    for (int wi = 0; wi < DGT_WAVES; wi++) {
        DGTWave* w = &m->dgt_waves[wi];
        if (!w->active) continue;
        if (w->imaginary && (m->tick - w->fire_tick) > 14) {
            w->active = 0;
            continue;
        }
        float radius = (m->tick - w->fire_tick) * w->speed;
        float dist_now = rb_dist(bot->x, bot->y, w->ox, w->oy);
        if (radius >= dist_now + 18.0f || m->tick - w->fire_tick > 350) {
            if (!w->imaginary) {
                float gf = dgt_point_gf(w, bot->x, bot->y);
                dgt_knn_add(m->dgt_surf_feats, m->dgt_surf_gf, &m->dgt_surf_n,
                            &m->dgt_surf_head, DGT_KNN_CAP, w->feats, gf);
                m->dgt_waves_passed++;
            }
            w->active = 0;
        }
    }
}

static void dgt_add_enemy_wave(BotMem* m, Robot* bot, Robocode* env,
                               float power, int imaginary) {
    if (power < 0.1f || power > 3.01f) return;
    DGTWave* w = &m->dgt_waves[m->dgt_wave_head];
    m->dgt_wave_head = (m->dgt_wave_head + 1) % DGT_WAVES;
    w->ox = m->last_x;
    w->oy = m->last_y;
    w->speed = rb_bullet_speed(power);
    w->power = power;
    w->fire_tick = imaginary ? m->tick + 1 : m->tick - 1;
    if (w->fire_tick < 0) w->fire_tick = 0;
    w->head_on = rb_abs_bearing_deg(w->ox, w->oy, bot->x, bot->y);
    float lat_sum = 0.0f;
    for (int i = 0; i < 10; i++) lat_sum += fabsf(m->dgt_lat_hist[i]);
    dgt_features(bot, w->ox, w->oy, env, m->tick, m->last_dir_change_tick,
                 m->last_dir_change_tick, m->dgt_last_v, lat_sum, w->feats);
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float dx = bot->x - w->ox, dy = bot->y - w->oy;
    float dist = fmaxf(sqrtf(dx * dx + dy * dy), 1.0f);
    float lat = (-bvx * dy + bvy * dx) / dist;
    w->lat_sign = (lat >= 0.0f) ? 1 : -1;
    w->imaginary = imaginary;
    w->active = 1;
}

static void bot_drussgt_step(Robocode* env, int bot_idx, BotMem* m) {
    Robot* bot = &env->robots[bot_idx];
    int target_idx = rb_nearest_agent(env, bot);
    if (target_idx < 0) return;
    if (!rb_scan_target(env, bot, m, target_idx)) return;

    if (!m->dgt_initialized) {
        m->dgt_initialized = 1;
        m->dgt_enemy_energy = (float)m->last_energy_seen;
        m->dgt_enemy_firepower = 2.0f;
        m->dgt_enemy_gunheat = 0.0f;
        m->dgt_lat_dir = 1.0f;
        m->dgt_enemy_lat_dir = 1.0f;
        m->dgt_last_v = bot->v;
        m->dgt_enemy_last_v = m->last_v;
        m->dgt_enemy_dir_change_tick = m->tick;
        m->dgt_enemy_decel_tick = m->tick;
        m->dgt_goto_x = bot->x;
        m->dgt_goto_y = bot->y;
        for (int i = 0; i < DGT_WAVES; i++) m->dgt_waves[i].active = 0;
        for (int i = 0; i < DGT_GUN_WAVES; i++) m->dgt_gun_waves[i].active = 0;
    }

    m->dgt_hist_x[m->dgt_hist_head] = m->last_x;
    m->dgt_hist_y[m->dgt_hist_head] = m->last_y;
    m->dgt_hist_head = (m->dgt_hist_head + 1) % DGT_HIST;
    if (m->dgt_hist_n < DGT_HIST) m->dgt_hist_n++;

    m->dgt_enemy_gunheat = fmaxf(0.0f, m->dgt_enemy_gunheat - 0.1f);

    float drop = m->dgt_enemy_energy - (float)m->last_energy_seen;
    if (drop >= 0.1f && drop <= 3.0f && m->dgt_enemy_gunheat <= 0.15f) {
        m->dgt_enemy_firepower = drop;
        m->dgt_enemy_gunheat = 1.0f + drop / 5.0f;
        int promoted = 0;
        for (int wi = 0; wi < DGT_WAVES; wi++) {
            DGTWave* w = &m->dgt_waves[wi];
            if (w->active && w->imaginary && fabsf(w->power - drop) < 0.45f) {
                w->imaginary = 0;
                w->power = drop;
                w->speed = rb_bullet_speed(drop);
                w->fire_tick = m->tick > 0 ? m->tick - 1 : 0;
                promoted = 1;
                break;
            }
        }
        if (!promoted) dgt_add_enemy_wave(m, bot, env, drop, 0);
        for (int wi = 0; wi < DGT_WAVES; wi++) {
            if (m->dgt_waves[wi].active && m->dgt_waves[wi].imaginary)
                m->dgt_waves[wi].active = 0;
        }
    }
    m->dgt_enemy_energy = (float)m->last_energy_seen;

    if (m->dgt_enemy_gunheat <= 0.0f) {
        int has_imag = 0;
        for (int wi = 0; wi < DGT_WAVES; wi++) {
            if (m->dgt_waves[wi].active && m->dgt_waves[wi].imaginary) {
                has_imag = 1;
                break;
            }
        }
        if (!has_imag) {
            float pred = m->dgt_enemy_firepower > 0.1f ? m->dgt_enemy_firepower : 2.0f;
            dgt_add_enemy_wave(m, bot, env, pred, 1);
        }
    }

    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float dxe = m->last_x - bot->x, dye = m->last_y - bot->y;
    float dist = fmaxf(sqrtf(dxe * dxe + dye * dye), 1.0f);
    float inv = 1.0f / dist;
    float ux = dxe * inv, uy = dye * inv;
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float lat_v = -bvx * uy + bvy * ux;
    m->dgt_lat_hist[m->dgt_lat_hist_i % 10] = lat_v;
    m->dgt_lat_hist_i++;
    if (fabsf(lat_v) > 0.1f) {
        float nd = lat_v > 0.0f ? 1.0f : -1.0f;
        if (nd != m->dgt_lat_dir) {
            m->dgt_lat_dir = nd;
            m->last_dir_change_tick = m->tick;
        }
    }
    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    float enemy_lat = -tvx * uy + tvy * ux;
    if (fabsf(enemy_lat) > 0.05f) {
        float ed = enemy_lat > 0.0f ? 1.0f : -1.0f;
        if (ed != m->dgt_enemy_lat_dir) {
            m->dgt_enemy_lat_dir = ed;
            m->dgt_enemy_dir_change_tick = m->tick;
        }
    }
    if (m->last_v < m->dgt_enemy_last_v - 0.5f) {
        m->dgt_enemy_decel_tick = m->tick;
    }

    dgt_expire_and_flatten(m, bot);
    dgt_update_gun_waves(m);
    dgt_surf_goto(env, bot, m);  // every tick

    // ---- Gun ----
    int can_fire = (bot->gun_heat <= 0.0f && bot->energy > 1.0f
                    && m->last_energy_seen > 0.0f);
    // Full DC while cooling into a shot so the gun is aligned at fire time.
    if (bot->gun_heat <= 0.35f) {
        float gfeats[DGT_FEATS];
        Robot fake = *bot;
        fake.x = m->last_x;
        fake.y = m->last_y;
        fake.heading = m->last_heading;
        fake.v = m->last_v;
        dgt_features(&fake, bot->x, bot->y, env, m->tick,
                     m->dgt_enemy_dir_change_tick, m->dgt_enemy_decel_tick,
                     m->dgt_enemy_last_v, dgt_hist_dist10(m), gfeats);
        float power = dgt_bullet_power(bot, (float)m->last_energy_seen, dist,
                                       m->dgt_enemy_firepower,
                                       m->dgt_bullets_hit, m->dgt_bullets_fired);
        if (power >= bot->energy) power = fmaxf(0.1f, (float)bot->energy - 0.1f);
        float bspeed = rb_bullet_speed(power);
        float max_escape = asinf(fminf(8.0f / bspeed, 1.0f)) * RB_R2D;
        float gf = dgt_gun_best_gf(gfeats, m->dgt_gun_feats, m->dgt_gun_gf, m->dgt_gun_n);
        float aim = abs_bearing + m->dgt_enemy_lat_dir * gf * max_escape;
        // Cold gun: linear + circular blend from last scan (fair).
        if (m->dgt_gun_n < 16) {
            float lin = dgt_linear_aim(bot, m, bspeed);
            float circ = dgt_circular_aim(bot, m, bspeed);
            float blend = 0.55f * lin + 0.45f * circ;
            float w = (float)m->dgt_gun_n / 16.0f;
            aim = w * aim + (1.0f - w) * blend;
        } else if (m->dgt_gun_n < 40) {
            float lin = dgt_linear_aim(bot, m, bspeed);
            aim = 0.88f * aim + 0.12f * lin;
        }
        float gun_delta = rb_turn_gun_to(bot, aim);
        if (can_fire && fabsf(gun_delta) < 2.0f && power < bot->energy) {
            fire(env, bot, bot_idx, power);
            m->dgt_bullets_fired++;
            DGTGunWave* gw = &m->dgt_gun_waves[m->dgt_gun_wave_head];
            m->dgt_gun_wave_head = (m->dgt_gun_wave_head + 1) % DGT_GUN_WAVES;
            gw->ox = bot->x;
            gw->oy = bot->y;
            gw->abs_bearing = abs_bearing;
            gw->lat_dir = m->dgt_enemy_lat_dir;
            gw->speed = bspeed;
            gw->mea = max_escape;
            gw->dist_traveled = 0.0f;
            memcpy(gw->feats, gfeats, DGT_FEATS * sizeof(float));
            gw->active = 1;
        }
    } else {
        float aim = abs_bearing + m->dgt_enemy_lat_dir * 0.2f *
                    (asinf(fminf(8.0f / 14.0f, 1.0f)) * RB_R2D);
        rb_turn_gun_to(bot, aim);
    }

    rb_turn_radar_to(bot, abs_bearing, 10.0f);
    m->dgt_last_v = bot->v;
    m->dgt_enemy_last_v = m->last_v;
}

static void dgt_on_hit_by_bullet(BotMem* m, float bullet_heading, float bullet_power) {
    float speed = rb_bullet_speed(bullet_power);
    DGTWave* best = NULL;
    int best_age = -1;
    for (int wi = 0; wi < DGT_WAVES; wi++) {
        DGTWave* w = &m->dgt_waves[wi];
        if (!w->active) continue;
        if (fabsf(w->speed - speed) > 0.6f) continue;
        int age = m->tick - w->fire_tick;
        if (age > best_age) { best_age = age; best = w; }
    }
    if (best == NULL) return;
    float gf_raw = rb_norm_deg(bullet_heading - best->head_on);
    float mea_deg = fmaxf(asinf(fminf(8.0f / best->speed, 1.0f)) * RB_R2D, 0.1f);
    float gf = rb_clampf((gf_raw / mea_deg) * (float)best->lat_sign, -1.5f, 1.5f);
    // Triple-weight hits vs flattener visits.
    for (int r = 0; r < 3; r++) {
        dgt_knn_add(m->dgt_surf_feats, m->dgt_surf_gf, &m->dgt_surf_n,
                    &m->dgt_surf_head, DGT_KNN_CAP, best->feats, gf);
    }
    m->dgt_hits_taken++;
    best->active = 0;
}

static void dgt_on_bullet_hit(BotMem* m, float hit_x, float hit_y) {
    m->dgt_bullets_hit++;
    DGTGunWave* best = NULL;
    float best_err = 1e18f;
    for (int i = 0; i < DGT_GUN_WAVES; i++) {
        DGTGunWave* w = &m->dgt_gun_waves[i];
        if (!w->active) continue;
        float err = fabsf(rb_dist(w->ox, w->oy, hit_x, hit_y) - w->dist_traveled);
        if (err < best_err) { best_err = err; best = w; }
    }
    if (best == NULL) return;
    float bearing = rb_abs_bearing_deg(best->ox, best->oy, hit_x, hit_y);
    float gf = rb_clampf(
        (rb_norm_deg(bearing - best->abs_bearing) / fmaxf(best->mea, 0.1f)) * best->lat_dir,
        -1.2f, 1.2f);
    dgt_knn_add(m->dgt_gun_feats, m->dgt_gun_gf, &m->dgt_gun_n,
                &m->dgt_gun_head, DGT_GUN_CAP, best->feats, gf);
    dgt_knn_add(m->dgt_gun_feats, m->dgt_gun_gf, &m->dgt_gun_n,
                &m->dgt_gun_head, DGT_GUN_CAP, best->feats, gf);
    best->active = 0;
}

#endif  // ROBOCODE_AGENT_DRUSSGT_H
