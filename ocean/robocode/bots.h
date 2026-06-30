// CPU bot policies for robocode. Included by robocode.h AFTER the Robocode /
// Robot / Bullet structs and the move/turn/fire helpers are defined.
//
// Information model is faithful to classic Robocode:
//   * Each tick the bot "scans" the target and reads: x, y, heading, v, energy.
//     It does NOT read gun_heading or bullet state directly.
//   * Fire events are inferred from target.energy drops in (0, 3].
//   * Training samples for the kNN danger model are only added when an enemy
//     bullet actually hits the bot — c_step calls bot_on_hit_by_bullet() and
//     we get (bullet.heading, bullet.power) from the impact event, matching
//     Robocode's onHitByBullet event.
//
// Faithful adaptation of BeepBoop's wave surfer
// (https://robowiki.net/wiki/BeepBoop/Understanding_BeepBoop) but simplified:
//   * 5 hand-picked features instead of beep-boop's learned embedding.
//   * No bullet shielding, no flattening, no virtual gun-heat waves.
//   * 3-candidate direction search (-1, 0, +1) instead of full path simulation.

#ifndef ROBOCODE_BOTS_H
#define ROBOCODE_BOTS_H

#include <string.h>
#include <math.h>

typedef enum {
    BOT_STATIONARY  = 0,
    BOT_MINIMAL     = 1,
    BOT_SURFER      = 2,
    BOT_WAVE_SURFER = 3,
} BotPolicy;

#define WS_NUM_WAVES        8
#define WS_KNN_CAP        256
#define WS_KNN_K            5
#define WS_NUM_FEATS        5
#define WS_SCAN_WIDTH_DEG   5.625f  // beep-boop's SCAN_WIDTH = π/32 in degrees
#define WS_MAX_REVERSALS    3       // reversals before falling back to full sweep

// Scale weights — bigger scale => feature contributes more to kNN distance.
// Same trick as beep-boop's embedding (no normalization to [-1,1]).
static const float WS_FEAT_W[WS_NUM_FEATS] = {
    0.01f,   // distance      (range ~0..1200 -> ~0..12)
    1.0f,    // signed lateral velocity / 8   (~-1..1)
    0.5f,    // signed advancing velocity / 8 (~-0.5..0.5)
    0.04f,   // ticks since direction change
    2.0f,    // wall proximity (0=at wall, 1=center)
};

typedef struct {
    float ox, oy;              // wave origin (target pos at fire time)
    float head_on;             // angle from origin to bot at fire time (deg)
    float speed;
    int   fire_tick;
    int   lat_sign;            // sign of bot's lateral velocity at fire time
    float feats[WS_NUM_FEATS];
    int   active;
} WSWave;

struct BotMem {
    int    tick;
    int    orbit_dir;            // -1, 0, +1
    int    last_dir_change_tick;
    // Last-scanned target snapshot. Updated only when scan_area() returns the
    // target this tick. Until first scan, last_scan_tick == 0 and decisions
    // are skipped.
    float  last_x, last_y, last_heading, last_v;
    int    last_energy_seen;
    int    last_scan_tick;
    int    radar_dir;            // ±1, the direction the radar is currently sweeping
    int    radar_reversals;      // beep-boop's lost-lock reversal counter
    int    wave_head;
    WSWave waves[WS_NUM_WAVES];
    int    knn_n;
    int    knn_head;
    float  knn_feats[WS_KNN_CAP][WS_NUM_FEATS];
    float  knn_gf[WS_KNN_CAP];
};

// ---- Lifetime ---------------------------------------------------------------
static inline void bot_mems_alloc(Robocode* env) {
    if (env->num_bots <= 0) { env->bot_mems = NULL; return; }
    env->bot_mems = (BotMem*)calloc(env->num_bots, sizeof(BotMem));
    for (int i = 0; i < env->num_bots; i++) env->bot_mems[i].orbit_dir = 1;
}
static inline void bot_mems_free(Robocode* env) {
    if (env->bot_mems) free(env->bot_mems);
}
// Called from c_reset on episode boundaries. Clears scan/wave/radar state but
// preserves the kNN model — long-term learning across episodes is the point.
static inline void bot_mems_episode_reset(Robocode* env) {
    if (env->bot_mems == NULL) return;
    for (int i = 0; i < env->num_bots; i++) {
        BotMem* m = &env->bot_mems[i];
        m->last_x = m->last_y = m->last_heading = m->last_v = 0.0f;
        m->last_energy_seen = 0;
        m->last_scan_tick = 0;
        m->radar_dir = 0;
        m->radar_reversals = 0;
        for (int wi = 0; wi < WS_NUM_WAVES; wi++) m->waves[wi].active = 0;
    }
}

// ---- Feature extraction -----------------------------------------------------
// tgt_x/tgt_y come from the bot's cached scan (BotMem.last_x / last_y), not
// the live target struct — matches Robocode's information model.
static void ws_features(Robot* bot, float tgt_x, float tgt_y, Robocode* env,
                        int tick, int last_change, float out[WS_NUM_FEATS]) {
    float dx = bot->x - tgt_x, dy = bot->y - tgt_y;
    float dist = sqrtf(dx*dx + dy*dy);
    float ux = (dist > 1e-6f) ? dx/dist : 1.0f;
    float uy = (dist > 1e-6f) ? dy/dist : 0.0f;
    float bvx = cos_deg(bot->heading) * bot->v;
    float bvy = sin_deg(bot->heading) * bot->v;
    float adv_v = bvx*ux + bvy*uy;             // along bot->target axis
    float lat_v = -bvx*uy + bvy*ux;            // perpendicular
    float wall_min = fminf(fminf(bot->x, env->width  - bot->x),
                           fminf(bot->y, env->height - bot->y));
    float wall_half = fmaxf(fminf(env->width, env->height) * 0.5f, 1.0f);
    out[0] = dist;
    out[1] = lat_v / 8.0f;
    out[2] = adv_v / 8.0f;
    out[3] = (float)(tick - last_change);
    out[4] = wall_min / wall_half;
}

// ---- kNN density estimate ---------------------------------------------------
// danger(features, gf) = sum over top-K neighbors of  w_i * N(gf - gf_i, sigma)
//   w_i = 1 / (1 + weighted_d2_i)
// Higher value => enemy's aim more likely lands at `gf` => more dangerous.
static float ws_danger(BotMem* m, const float feats[WS_NUM_FEATS], float gf) {
    if (m->knn_n == 0) return 0.0f;
    float best_d[WS_KNN_K];
    int   best_i[WS_KNN_K];
    for (int k = 0; k < WS_KNN_K; k++) { best_d[k] = 1e18f; best_i[k] = -1; }
    for (int n = 0; n < m->knn_n; n++) {
        float d2 = 0.0f;
        for (int f = 0; f < WS_NUM_FEATS; f++) {
            float diff = (feats[f] - m->knn_feats[n][f]) * WS_FEAT_W[f];
            d2 += diff * diff;
        }
        for (int k = 0; k < WS_KNN_K; k++) {
            if (d2 < best_d[k]) {
                for (int s = WS_KNN_K - 1; s > k; s--) {
                    best_d[s] = best_d[s-1]; best_i[s] = best_i[s-1];
                }
                best_d[k] = d2; best_i[k] = n;
                break;
            }
        }
    }
    const float sigma = 0.15f;
    const float two_s2 = 2.0f * sigma * sigma;
    float danger = 0.0f;
    for (int k = 0; k < WS_KNN_K; k++) {
        if (best_i[k] < 0) break;
        float w = 1.0f / (1.0f + best_d[k]);
        float dgf = gf - m->knn_gf[best_i[k]];
        danger += w * expf(-(dgf*dgf) / two_s2);
    }
    return danger;
}

static inline void ws_add_sample(BotMem* m, const float feats[WS_NUM_FEATS], float gf) {
    int slot = m->knn_head;
    memcpy(m->knn_feats[slot], feats, WS_NUM_FEATS * sizeof(float));
    m->knn_gf[slot] = gf;
    m->knn_head = (m->knn_head + 1) % WS_KNN_CAP;
    if (m->knn_n < WS_KNN_CAP) m->knn_n++;
}

// ---- onHitByBullet ----------------------------------------------------------
// Called from c_step when an enemy bullet hits a bot. The bullet's heading and
// power are provided — same info Robocode's onHitByBullet event delivers.
// We find the matching in-flight wave (by speed + age) and add a training
// sample to the kNN.
static void bot_on_hit_by_bullet(Robocode* env, int bot_idx,
                                 float bullet_heading, float bullet_power) {
    if (env->bot_policy != BOT_WAVE_SURFER) return;
    if (env->bot_mems == NULL) return;
    BotMem* m = &env->bot_mems[bot_idx - env->num_agents];
    float speed = 20.0f - 3.0f * bullet_power;
    WSWave* best = NULL;
    int best_age = -1;
    for (int wi = 0; wi < WS_NUM_WAVES; wi++) {
        WSWave* w = &m->waves[wi];
        if (!w->active) continue;
        if (fabsf(w->speed - speed) > 0.5f) continue;
        int age = m->tick - w->fire_tick;
        if (age > best_age) { best_age = age; best = w; }
    }
    if (best == NULL) return;
    float gf_raw = bullet_heading - best->head_on;
    if (gf_raw >  180.0f) gf_raw -= 360.0f;
    else if (gf_raw < -180.0f) gf_raw += 360.0f;
    float mea_rad = asinf(fminf(8.0f / best->speed, 1.0f));
    float mea_deg = fmaxf(mea_rad * (180.0f / 3.14159265358979f), 0.1f);
    float gf = (gf_raw / mea_deg) * best->lat_sign;
    if (gf >  1.5f) gf =  1.5f;
    if (gf < -1.5f) gf = -1.5f;
    ws_add_sample(m, best->feats, gf);
    best->active = 0;
}

// ---- Main entry -------------------------------------------------------------
static void bot_step(Robocode* env, int bot_idx) {
    Robot* bot = &env->robots[bot_idx];
    // Dead bots are skipped entirely; disabled bots (energy=0) are frozen
    // but the env still ticks. Same as agent rule in c_step.
    if (bot->energy < 0) return;
    if (bot->energy == 0) { bot->v = 0; return; }
    if (bot->gun_heat > 0) bot->gun_heat -= 0.1f;
    if (env->bot_policy == BOT_STATIONARY) return;

    BotMem* m = &env->bot_mems[bot_idx - env->num_agents];
    m->tick++;
    if (m->orbit_dir == 0) m->orbit_dir = 1;

    // Pick a target index. In 1v1 there's only one agent; for melee we use
    // true position to lock on — equivalent to "the only one we know about".
    int t = -1; float best = 1e18f;
    for (int j = 0; j < env->num_agents; j++) {
        Robot* a = &env->robots[j];
        if (a->energy < 0) continue;   // dead only — disabled is still a valid target
        float dx = a->x - bot->x, dy = a->y - bot->y;
        float d2 = dx*dx + dy*dy;
        if (d2 < best) { best = d2; t = j; }
    }
    if (t < 0) return;
    const float R2D = 180.0f / 3.14159265358979f;

    // ---- Radar control (faithful to beep-boop's Scanner) ------------------
    // Cold-start direction: aim toward battlefield center, max rate.
    if (m->radar_dir == 0) {
        float cx = env->width * 0.5f, cy = env->height * 0.5f;
        float center_bear = atan2f(cy - bot->y, cx - bot->x) * R2D;
        if (center_bear < 0) center_bear += 360.0f;
        float diff = center_bear - bot->radar_heading;
        if (diff >  180) diff -= 360; else if (diff < -180) diff += 360;
        m->radar_dir = (diff >= 0) ? 1 : -1;
    }
    float radar_delta;
    bool just_scanned = (m->last_scan_tick != 0 && m->last_scan_tick == m->tick - 1);
    if (just_scanned) {
        // scan(): aim radar at last seen position, then push past by SCAN_WIDTH
        // so next tick's wedge sweeps back across the target.
        float dxr = m->last_x - bot->x, dyr = m->last_y - bot->y;
        float bear = atan2f(dyr, dxr) * R2D;
        if (bear < 0) bear += 360.0f;
        float diff = bear - bot->radar_heading;
        if (diff >  180) diff -= 360; else if (diff < -180) diff += 360;
        float overshoot = (diff >= 0) ? WS_SCAN_WIDTH_DEG : -WS_SCAN_WIDTH_DEG;
        radar_delta = diff + overshoot;
        m->radar_dir = (radar_delta >= 0) ? 1 : -1;
        m->radar_reversals = 0;
    } else if (m->last_scan_tick != 0 && m->radar_reversals < WS_MAX_REVERSALS) {
        // search(): if the bearing-to-last-seen flipped sign vs our current
        // sweep direction, we overshot — reverse and bump the counter.
        float dxr = m->last_x - bot->x, dyr = m->last_y - bot->y;
        float bear = atan2f(dyr, dxr) * R2D;
        if (bear < 0) bear += 360.0f;
        float diff = bear - bot->radar_heading;
        if (diff >  180) diff -= 360; else if (diff < -180) diff += 360;
        int new_dir = (diff >= 0) ? 1 : -1;
        if (new_dir != m->radar_dir) {
            m->radar_dir = new_dir;
            m->radar_reversals++;
        }
        radar_delta = m->radar_dir * 45.0f;
    } else {
        // Cold start or out of reversals: just keep spinning at max rate.
        radar_delta = m->radar_dir * 45.0f;
    }
    bot->radar_heading_prev = bot->radar_heading;
    turn(&bot->radar_heading, radar_delta, 45.0f, 0);

    // ---- Scan: only refresh cache if the radar wedge actually crossed t ---
    int scanned = scan_area(env, bot);
    if (scanned == t) {
        Robot* tgt = &env->robots[t];
        // Detect target fire from energy drop BEFORE overwriting last_energy_seen.
        int drop = m->last_scan_tick > 0 ? (m->last_energy_seen - tgt->energy) : 0;
        bool fired = (drop > 0 && drop <= 3);
        if (env->bot_policy == BOT_SURFER && fired) {
            m->orbit_dir = -m->orbit_dir;
            m->last_dir_change_tick = m->tick;
        } else if (env->bot_policy == BOT_WAVE_SURFER && fired) {
            // Wave origin = target's PREVIOUS scanned position (where they
            // were the tick before they fired). speed inferred from drop.
            WSWave* w = &m->waves[m->wave_head];
            m->wave_head = (m->wave_head + 1) % WS_NUM_WAVES;
            w->ox = m->last_x; w->oy = m->last_y;
            w->speed = 20.0f - 3.0f * drop;
            w->fire_tick = m->tick;
            float dwx = bot->x - w->ox, dwy = bot->y - w->oy;
            w->head_on = atan2f(dwy, dwx) * R2D;
            ws_features(bot, w->ox, w->oy, env, m->tick, m->last_dir_change_tick, w->feats);
            w->lat_sign = (w->feats[1] >= 0.0f) ? 1 : -1;
            w->active = 1;
            if (m->knn_n < WS_KNN_K) {  // bootstrap while kNN is sparse
                m->orbit_dir = -m->orbit_dir;
                m->last_dir_change_tick = m->tick;
            }
        }
        // Refresh cache.
        m->last_x = tgt->x; m->last_y = tgt->y;
        m->last_heading = tgt->heading; m->last_v = tgt->v;
        m->last_energy_seen = tgt->energy;
        m->last_scan_tick = m->tick;
    }
    if (m->last_scan_tick == 0) return;  // still hunting for first contact

    // ---- Wave-surfer: expire missed waves, then choose orbit direction ---
    if (env->bot_policy == BOT_WAVE_SURFER) {
        for (int wi = 0; wi < WS_NUM_WAVES; wi++) {
            WSWave* w = &m->waves[wi];
            if (!w->active) continue;
            float radius = (m->tick - w->fire_tick) * w->speed;
            float ddx = bot->x - w->ox, ddy = bot->y - w->oy;
            float dist_now = sqrtf(ddx*ddx + ddy*ddy);
            if (radius >= dist_now + 32.0f) w->active = 0;
            else if (m->tick - w->fire_tick > 400) w->active = 0;
        }
        if (m->knn_n > 0) {
            float feats[WS_NUM_FEATS];
            ws_features(bot, m->last_x, m->last_y, env, m->tick, m->last_dir_change_tick, feats);
            int   cands[3] = {-1, 0, +1};
            float danger[3] = {0, 0, 0};
            for (int wi = 0; wi < WS_NUM_WAVES; wi++) {
                WSWave* w = &m->waves[wi];
                if (!w->active) continue;
                float ddx = bot->x - w->ox, ddy = bot->y - w->oy;
                float dist_now = fmaxf(sqrtf(ddx*ddx + ddy*ddy), 1.0f);
                float radius = (m->tick - w->fire_tick) * w->speed;
                float tti = (dist_now - radius) / w->speed;
                if (tti <= 0) continue;
                float mea_rad = asinf(fminf(8.0f / w->speed, 1.0f));
                for (int c = 0; c < 3; c++) {
                    float lat_v = cands[c] * 8.0f;
                    float dtheta = lat_v * tti / dist_now;
                    float gf = (dtheta / mea_rad) * w->lat_sign;
                    danger[c] += ws_danger(m, feats, gf);
                }
            }
            int best_c = 0;
            for (int c = 1; c < 3; c++) if (danger[c] < danger[best_c]) best_c = c;
            int new_dir = cands[best_c];
            if (new_dir == 0) new_dir = m->orbit_dir;
            if (new_dir != m->orbit_dir) {
                m->orbit_dir = new_dir;
                m->last_dir_change_tick = m->tick;
            }
        }
    }

    // ---- Shared aim/move/fire (linear-lead from cached scan values) ------
    float dx = m->last_x - bot->x, dy = m->last_y - bot->y;
    float dt = sqrtf(dx*dx + dy*dy) / 17.0f;
    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    float aim  = atan2f(dy + tvy*dt, dx + tvx*dt) * R2D;
    float bear = atan2f(dy, dx) * R2D;
    if (aim  < 0) aim  += 360.0f;
    if (bear < 0) bear += 360.0f;
    float orbit = bear + 90.0f * m->orbit_dir;
    if (orbit >= 360.0f) orbit -= 360.0f;
    else if (orbit < 0.0f) orbit += 360.0f;

    float gun_d  = aim   - bot->gun_heading;
    float body_d = orbit - bot->heading;
    if (gun_d  >  180) gun_d  -= 360; else if (gun_d  < -180) gun_d  += 360;
    if (body_d >  180) body_d -= 360; else if (body_d < -180) body_d += 360;
    float body_turned = turn(&bot->heading, body_d, 10.0f - 0.75f*fabsf(bot->v), 0);
    turn(&bot->gun_heading, gun_d, 20.0f, body_turned);
    move(env, bot, 1.0f);
    bot->x = fmaxf(16.0f, fminf(bot->x, env->width  - 16.0f));
    bot->y = fmaxf(16.0f, fminf(bot->y, env->height - 16.0f));
    if (fabsf(gun_d) < 3.0f && bot->gun_heat <= 0.0f) fire(env, bot, bot_idx, 1.0f);
}

#endif  // ROBOCODE_BOTS_H
