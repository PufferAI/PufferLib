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
    BOT_STATIONARY    = 0,
    BOT_MINIMAL       = 1,
    BOT_SURFER        = 2,
    BOT_WAVE_SURFER   = 3,
    BOT_HAWK_ON_FIRE  = 4,
    BOT_RAIKO         = 5,
    BOT_DRUSSGT       = 6,
} BotPolicy;

#define WS_NUM_WAVES        8
#define WS_KNN_CAP        256
#define WS_KNN_K            5
#define WS_NUM_FEATS        5
#define WS_SCAN_WIDTH_DEG   5.625f  // beep-boop's SCAN_WIDTH = π/32 in degrees
#define WS_MAX_REVERSALS    3       // reversals before falling back to full sweep

#define RAIKO_GF_ZERO      15
#define RAIKO_GF_BINS      31
#define RAIKO_DIST_BINS     8
#define RAIKO_WAVES        16
#define RAIKO_BEST_DISTANCE 525.0f

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

typedef struct {
    float x, y;
    float abs_bearing;
    float bearing_step;
    float speed;
    float distance;
    int dist_bin;
    int active;
} RBRaikoWave;

// DrussGT adaptation sizes (agent_drussgt.h). Moderate caps for train SPS.
#ifndef DGT_WAVES
#define DGT_WAVES          10
#define DGT_GUN_WAVES      14
#define DGT_KNN_CAP        96
#define DGT_GUN_CAP       160
#define DGT_FEATS           8
#define DGT_HIST           16
#endif
typedef struct {
    float ox, oy;
    float head_on;
    float speed;
    float power;
    int   fire_tick;
    int   lat_sign;
    float feats[DGT_FEATS];
    int   active;
    int   imaginary;
} DGTWave;

typedef struct {
    float ox, oy;
    float abs_bearing;
    float lat_dir;
    float speed;
    float mea;
    float dist_traveled;
    float feats[DGT_FEATS];
    int   active;
} DGTGunWave;

struct BotMem {
    int    tick;
    int    orbit_dir;            // -1, 0, +1
    int    last_dir_change_tick;
    // Last-scanned target snapshot. Updated only when scan_area() returns the
    // target this tick. Until first scan, last_scan_tick == 0 and decisions
    // are skipped.
    float  last_x, last_y, last_heading, last_v;
    float  last_energy_seen;
    int    last_scan_tick;
    int    radar_dir;            // ±1, the direction the radar is currently sweeping
    int    radar_reversals;      // beep-boop's lost-lock reversal counter
    int    wave_head;
    WSWave waves[WS_NUM_WAVES];
    int    knn_n;
    int    knn_head;
    float  knn_feats[WS_KNN_CAP][WS_NUM_FEATS];
    float  knn_gf[WS_KNN_CAP];

    // Shared source-port bot scratch. These are reset every episode.
    int    dest_initialized;
    float  dest_x, dest_y;
    float  dest_last_x, dest_last_y;

    // Raiko-style orbit movement + compact guess-factor gun. The original
    // Java bot uses a much larger segmented table; this keeps vector-env
    // memory bounded while preserving the online wave-update procedure.
    int    raiko_initialized;
    float  raiko_circle_dir;
    float  raiko_bearing_dir;
    float  raiko_enemy_energy;
    float  raiko_enemy_firepower;
    int    raiko_last_reverse_tick;
    int    raiko_wave_head;
    int    raiko_guess[RAIKO_DIST_BINS][RAIKO_GF_BINS];
    RBRaikoWave raiko_waves[RAIKO_WAVES];

    // DrussGT (agent_drussgt.h). kNN persists across episodes; waves clear on reset.
    int    dgt_initialized;
    float  dgt_enemy_gunheat;
    float  dgt_enemy_energy;
    float  dgt_enemy_firepower;
    float  dgt_lat_dir;
    float  dgt_enemy_lat_dir;
    float  dgt_last_v;
    float  dgt_enemy_last_v;
    int    dgt_enemy_dir_change_tick;
    int    dgt_enemy_decel_tick;
    float  dgt_goto_x, dgt_goto_y;
    int    dgt_wave_head;
    DGTWave dgt_waves[DGT_WAVES];
    int    dgt_surf_n, dgt_surf_head;
    float  dgt_surf_feats[DGT_KNN_CAP][DGT_FEATS];
    float  dgt_surf_gf[DGT_KNN_CAP];
    int    dgt_gun_n, dgt_gun_head;
    float  dgt_gun_feats[DGT_GUN_CAP][DGT_FEATS];
    float  dgt_gun_gf[DGT_GUN_CAP];
    int    dgt_gun_wave_head;
    DGTGunWave dgt_gun_waves[DGT_GUN_WAVES];
    int    dgt_bullets_hit, dgt_bullets_fired;
    int    dgt_hits_taken, dgt_waves_passed;
    int    dgt_hist_n, dgt_hist_head;
    float  dgt_hist_x[DGT_HIST];
    float  dgt_hist_y[DGT_HIST];
    float  dgt_lat_hist[10];
    int    dgt_lat_hist_i;
    float  dgt_current_gf;
};

#include "agent_hawk_on_fire.h"
#include "agent_raiko.h"
#include "agent_drussgt.h"

// Per-bot policy: bot_policy_1 overrides for the second bot in bot-vs-bot.
static inline int bot_policy_for(Robocode* env, int bot_idx) {
    int bi = bot_idx - env->num_agents;
    if (bi == 1 && env->bot_policy_1 >= 0) return env->bot_policy_1;
    return env->bot_policy;
}

// ---- Lifetime ---------------------------------------------------------------
static inline void bot_mems_alloc(Robocode* env) {
    if (env->num_bots <= 0) { env->bot_mems = NULL; return; }
    env->bot_mems = (BotMem*)calloc(env->num_bots, sizeof(BotMem));
    for (int i = 0; i < env->num_bots; i++) {
        env->bot_mems[i].orbit_dir = 1;
        env->bot_mems[i].raiko_circle_dir = 1.0f;
        env->bot_mems[i].raiko_bearing_dir = 1.0f;
    }
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
        m->dest_initialized = 0;
        m->raiko_initialized = 0;
        m->raiko_enemy_energy = 100.0f;
        m->raiko_enemy_firepower = 2.0f;
        m->raiko_circle_dir = m->raiko_circle_dir == 0.0f ? 1.0f : m->raiko_circle_dir;
        m->raiko_bearing_dir = m->raiko_bearing_dir == 0.0f ? 1.0f : m->raiko_bearing_dir;
        for (int wi = 0; wi < WS_NUM_WAVES; wi++) m->waves[wi].active = 0;
        for (int wi = 0; wi < RAIKO_WAVES; wi++) m->raiko_waves[wi].active = 0;
        // DrussGT: clear per-round waves; keep kNN across episodes.
        m->dgt_initialized = 0;
        for (int wi = 0; wi < DGT_WAVES; wi++) m->dgt_waves[wi].active = 0;
        for (int wi = 0; wi < DGT_GUN_WAVES; wi++) m->dgt_gun_waves[wi].active = 0;
        m->dgt_hist_n = 0;
        m->dgt_hist_head = 0;
        m->dgt_lat_hist_i = 0;
        for (int hi = 0; hi < 10; hi++) m->dgt_lat_hist[hi] = 0.0f;
        m->dgt_current_gf = 0.0f;
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
    if (env->bot_mems == NULL) return;
    BotMem* m = &env->bot_mems[bot_idx - env->num_agents];
    int policy = bot_policy_for(env, bot_idx);
    if (policy == BOT_DRUSSGT) {
        dgt_on_hit_by_bullet(m, bullet_heading, bullet_power);
        return;
    }
    if (policy != BOT_WAVE_SURFER) return;
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

// Curriculum random bot: sample from the same discrete action tables agents use.
// Called when rand_unit(env) < env->bot_cl_noise instead of the scripted policy.
static void bot_random_step(Robocode* env, int bot_idx) {
    Robot* bot = &env->robots[bot_idx];
    float move_atn = ACCEL_VALUES[rand_r(&env->rng) % 4];
    move(env, bot, move_atn);

    float turn_atn = TURN_VALUES[rand_r(&env->rng) % 9];
    float max_turn = 10.0f - 0.75f * fabsf(bot->v);
    if (max_turn < 0.0f) max_turn = 0.0f;
    float body = turn(&bot->heading, turn_atn, max_turn, 0.0f);

    float gun_atn = GUN_TURN_VALUES[rand_r(&env->rng) % 11];
    float gun = turn(&bot->gun_heading, gun_atn, 20.0f, body);

    float radar_atn = RADAR_TURN_VALUES[rand_r(&env->rng) % 11];
    bot->radar_heading_prev = bot->radar_heading;
    turn(&bot->radar_heading, radar_atn, 45.0f, body + gun);

    float firepower = FIREPOWER_VALUES[rand_r(&env->rng) % 6];
    if (firepower > 0.0f) {
        fire(env, bot, bot_idx, firepower);
    }

    float px = bot->x, py = bot->y;
    bot->x = fmaxf(16.0f, fminf(bot->x, env->width - 16.0f));
    bot->y = fmaxf(16.0f, fminf(bot->y, env->height - 16.0f));
    if (bot->x != px || bot->y != py) {
        float wall_dmg = fabsf(bot->v) * 0.5f - 1.0f;
        if (wall_dmg < 0.0f) wall_dmg = 0.0f;
        bot->energy -= wall_dmg;
        bot->v = 0.0f;
    }
}

// ---- Main entry -------------------------------------------------------------
static void bot_step(Robocode* env, int bot_idx) {
    Robot* bot = &env->robots[bot_idx];
    // Dead bots are skipped entirely; disabled bots (energy=0) are frozen
    // but the env still ticks. Same as agent rule in c_step.
    if (bot->energy < 0) return;
    if (bot->energy == 0) { bot->v = 0; return; }
    if (bot->gun_heat > 0) bot->gun_heat -= 0.1f;
    int policy = bot_policy_for(env, bot_idx);
    if (policy == BOT_STATIONARY) return;

    BotMem* m = &env->bot_mems[bot_idx - env->num_agents];
    m->tick++;
    if (m->orbit_dir == 0) m->orbit_dir = 1;

    // Curriculum: randomly replace scripted policy with discrete noise.
    if (env->bot_cl_noise > 0.0f && rand_unit(env) < env->bot_cl_noise) {
        bot_random_step(env, bot_idx);
        return;
    }

    if (policy == BOT_HAWK_ON_FIRE) {
        bot_hawk_on_fire_step(env, bot_idx, m);
        return;
    }
    if (policy == BOT_RAIKO) {
        bot_raiko_step(env, bot_idx, m);
        return;
    }
    if (policy == BOT_DRUSSGT) {
        bot_drussgt_step(env, bot_idx, m);
        return;
    }

    // Pick a target index. Normal training/eval bots target RL agents.
    // Temporary bot-vs-bot harnesses use num_agents=0, where bots target
    // other bots instead.
    int total_robots = env->num_agents + env->num_bots;
    int target_limit = env->num_agents > 0 ? env->num_agents : total_robots;
    int t = -1; float best = 1e18f;
    for (int j = 0; j < target_limit; j++) {
        Robot* a = &env->robots[j];
        if (j == bot_idx || a->energy <= 0.0f) continue;
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
        float drop = m->last_scan_tick > 0 ? (m->last_energy_seen - tgt->energy) : 0.0f;
        bool fired = (drop > 0.0f && drop <= 3.0f);
        if (policy == BOT_SURFER && fired) {
            m->orbit_dir = -m->orbit_dir;
            m->last_dir_change_tick = m->tick;
        } else if (policy == BOT_WAVE_SURFER && fired) {
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
    if (policy == BOT_WAVE_SURFER) {
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
