// NetHack env, obs layout mirrored by ocean/nethack/nethack.cu
// One env per agent, each owning an nle_ctx_t and a private vardir on tmpfs.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <signal.h>
#include "fs.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

// nletypes.h, not nle.h: nle.h's `settings` macro would rewrite env->settings
#include "nletypes.h"

#ifdef __cplusplus
extern "C" {
#endif
extern nle_ctx_t* nle_start(nle_obs*, FILE*, nle_settings*);
extern nle_ctx_t* nle_step(nle_ctx_t*, nle_obs*);
extern nle_ctx_t* nle_obs_refresh(nle_ctx_t*, nle_obs*);
extern int nle_path_drain(nle_ctx_t*, short*, int);
extern long nle_shop_price(nle_ctx_t*);
extern int nle_terrain_underfoot(nle_ctx_t*);
extern int nle_inside_shop(nle_ctx_t*);
extern int nle_container_at(nle_ctx_t*);
extern int nle_food_underfoot(nle_ctx_t*);
extern int nle_discoveries(nle_ctx_t*);
extern int nle_peaceful_at(nle_ctx_t*, int, int);
extern int nle_spellprot(nle_ctx_t*);
extern void nle_weight(nle_ctx_t*, int*, int*);
extern int nle_spells(nle_ctx_t*, short*, signed char*, signed char*, int*, int);
extern int nle_cast_blocked(nle_ctx_t*);
extern void nle_end(nle_ctx_t*);
extern void nle_identity(nle_ctx_t*, int*, int*, int*, int*);
#ifdef __cplusplus
}
#endif

#include "netlib.h"

#define OBS_SIZE NETHACK_OBS_SIZE
#define NUM_ATNS 20
#define ACT_SIZES {NETHACK_NUM_ACTIONS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, \
    NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, \
    NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, \
    NETHACK_SPELL_SLOTS}

typedef Env Nethack;
struct Env {
    Log log;
    Agent agents[1];
    unsigned char* action_mask;
    int num_agents;
    int pending_reset; // NLE's coroutine must reset on a stepping thread
    int tag;
    int boundary_reached;

    // engine handle
    nle_ctx_t* ctx;
    nle_obs obs;
    nle_settings settings;
    char vardir[1024];

    // NLE-written buffers
    short glyphs[NH_GRID];
    long blstats[NLE_BLSTATS_SIZE];
    unsigned char chars[NH_GRID];
    unsigned char message[NLE_MESSAGE_SIZE];
    int misc[NLE_MISC_SIZE];
    int internal[NLE_INTERNAL_SIZE];
    short inv_glyphs[NLE_INVENTORY_SIZE];
    short inv_true[NLE_INVENTORY_SIZE];
    unsigned char inv_letters[NLE_INVENTORY_SIZE];
    unsigned char inv_oclasses[NLE_INVENTORY_SIZE];
    signed char inv_state[NLE_INVENTORY_SIZE * NLE_INV_STATE_FIELDS];
    short spell_ids[8];
    signed char spell_levs[8], spell_fails[8];
    int spell_knows[8]; // retention turns, 0 = forgotten (slot-faithful)
    long stall_prev_turn; // stall watchdog: last seen game turn
    int stall_ctr; // consecutive same-turn steps
    int n_spells;

    Stats stats;

    // reward-delta trackers
    int prev_action;
    int enh_ready;
    long prev_score;
    long prev_exp;
    long prev_gold;
    long start_gold;
    float prev_ac_led; // ac-delta ledger: last ledgered AC (durable-weighted)
    float ac_account; // accrued unpaid ac-delta reward (carries)
    long prev_time;
    int prev_depth;
    unsigned prev_floor; // dnum << 8 | dlevel at last reward; guards path attribution
    int disc0; // discoveries count at reset (episode delta = types learned)
    unsigned long long engid_tested; // letters engrave-tested this episode

    // reward coefs
    float gold_coef;
    float exp_coef;
    float descent_coef;
    float floor_coef;
    float xp_coef;
    float scout_coef;
    float ac_coef;
    float scout_ready; // 0 = off; else a tile pays pro-rata to xp level vs depth
    float ac_nospell; // unpaid fraction of protection-spell AC (1 = durable AC only)
    float death_penalty;
    float mask_search20; // 1 removes SEARCH20 from the action space
    float mask_run; // 1 removes RUN from the action space
    float multi_role; // 1 = random role/race/gender/align per reset (challenge protocol)

    unsigned int rng; // required by vecenv.h
    unsigned long seed; // advanced each reset
    int role_idx, race_idx, gend_idx; // multi-role identity (read back)
};

#include "macros.h"

// init

// demo-only obs planes; NULL in training (fills skipped)
static unsigned char* nethack_color_sink;
static unsigned char* nethack_invstr_sink;
static unsigned char* nethack_tty_chars_sink;
static signed char* nethack_tty_colors_sink;
static unsigned char* nethack_tty_cursor_sink;
static const char* nethack_options_override; // demo-only; NULL = default options

static void nethack_bind_obs(Nethack* env) {
    nle_obs* o = &env->obs;
    memset(o, 0, sizeof(*o));
    o->colors = nethack_color_sink;
    o->inv_strs = nethack_invstr_sink;
    o->tty_chars = nethack_tty_chars_sink;
    o->tty_colors = nethack_tty_colors_sink;
    o->tty_cursor = nethack_tty_cursor_sink;
    o->glyphs = env->glyphs;
    o->blstats = env->blstats;
    o->chars = env->chars;
    o->message = env->message;
    o->misc = env->misc;
    o->internal = env->internal;
    o->inv_glyphs = env->inv_glyphs;
    o->inv_true_glyphs = env->inv_true;
    o->inv_letters = env->inv_letters;
    o->inv_oclasses = env->inv_oclasses;
    o->inv_state = env->inv_state;
    o->partial = 1;
}

static void nethack_init_settings(Nethack* env) {
    memset(&env->settings, 0, sizeof(env->settings));
    const char* source = getenv("NETHACKDIR");
    if (source == NULL) source = "./vendor/fast-nle/build/dat";

    if (nethack_make_vardir(source, env->vardir, sizeof(env->vardir)) != 0) {
        fprintf(stderr, "nethack: failed to create vardir from source=%s\n", source);
        strncpy(env->settings.hackdir, source, sizeof(env->settings.hackdir) - 1);
    } else {
        strncpy(env->settings.hackdir, env->vardir, sizeof(env->settings.hackdir) - 1);
    }
    env->settings.spawn_monsters = 1;
    env->settings.underfoot_glyphs = 1; // underfoot shows objects
    const char* role = getenv("NH_ROLE");
    char optbuf[512], rcbuf[512];
    const char* opts;
    if (role && role[0]) {
        snprintf(optbuf, sizeof(optbuf),
            "name:Agent,role:%s,race:random,gender:random,align:random,"
            NETHACK_OPTIONS_TAIL "!status_updates", role);
        opts = optbuf;
    } else {
        opts = nethack_options_override ? nethack_options_override
                                        : NETHACK_DEFAULT_OPTIONS;
    }
    snprintf(env->settings.options, sizeof(env->settings.options), "@%s",
             nethack_rc_path_opts(rcbuf, sizeof(rcbuf), opts));
    env->settings.fix_moon_phase = true; // moon phase from seed
}

void init(Nethack* env) {
    env->seed = 0xCAFEBEEFUL + (unsigned long)env->rng; // rng = env index
    // opt into consumed-head gating (env_head_consume_map below); =0 still disables
    setenv("PUFFER_HEAD_GATING", "1", 0);
    // nle_start deferred to first puf_reset
    nethack_init_settings(env);
}

// masking

static int nethack_slot_usable(const Nethack* env, const Verb* verb, int i) {
    if (!(verb->item_classes & (1u << env->inv_oclasses[i]))) return 0;
    // READ hygiene: blind reads refuse for free, and re-reading a still-fresh
    // book is a multi-turn re-study furnace; low-retention refresh stays legal
    if (verb->item_classes == ((1u << 9) | (1u << 10))) {
        if (env->blstats[NLE_BL_CONDITION] & 0x20L) return 0;
        if (env->inv_oclasses[i] == 10
            && env->inv_true[i] != NETHACK_PAD_GLYPH) {
            // only the discoveries channel names the spell (glyphs are shuffled
            // appearances), exactly when the agent could know it
            int otyp = env->inv_true[i] - NH_GLYPH_OBJ_OFF;
            for (int j = 0; j < NETHACK_SPELL_SLOTS; j++)
                if (env->spell_ids[j] == otyp && env->spell_knows[j] > 2000)
                    return 0;
        }
    }
    // WIELD while welded (known-cursed wield) refuses for free and the weld
    // never expires; the revealing first attempt on unknown BUC stays legal
    if (verb == &NETHACK_VERBS[NETHACK_ACT_WIELD]) {
        for (int j = 0; j < NETHACK_INV_SLOTS && env->inv_letters[j]; j++)
            if ((env->inv_state[j * NLE_INV_STATE_FIELDS + 5] & 2)
                && env->inv_state[j * NLE_INV_STATE_FIELDS + 0] == 1)
                return 0;
    }
    // TAKEOFF/REMOVE of known-cursed gear refuses for free; first attempt on
    // unknown BUC consumes a move and reveals, so it stays legal
    if ((verb == &NETHACK_VERBS[NETHACK_ACT_TAKEOFF]
         || verb == &NETHACK_VERBS[NETHACK_ACT_REMOVE])
        && env->inv_state[i * NLE_INV_STATE_FIELDS + 0] == 1)
        return 0;
    int worn = env->inv_state[i * NLE_INV_STATE_FIELDS + 5] & 1;
    if (verb->wornreq == WORN_ONLY) return worn;
    if (verb->wornreq == UNWORN_ONLY) {
        if (worn) return 0;
        // WEAR: mirror the engine's layering rules and known-cursed same-slot
        // swaps (56% of measured prompt aborts)
        if (verb->item_classes == (1u << 3)) {
            int gn = env->inv_glyphs[i] - NH_GLYPH_OBJ_OFF;
            int cat = (gn >= 0 && gn < NH_NUM_OBJECTS) ? nh_obj_armcat[gn] : -1;
            if (cat >= 0) {
                for (int j = 0; j < NETHACK_INV_SLOTS && env->inv_letters[j]; j++) {
                    if (!(env->inv_state[j * NLE_INV_STATE_FIELDS + 5] & 1)) continue;
                    int gj = env->inv_glyphs[j] - NH_GLYPH_OBJ_OFF;
                    int cj = (gj >= 0 && gj < NH_NUM_OBJECTS) ? nh_obj_armcat[gj] : -1;
                    if ((cat == 0 || cat == 6) && (cj == 5 || (cat == 6 && cj == 0))) return 0;
                    if (cj == cat
                        && env->inv_state[j * NLE_INV_STATE_FIELDS + 0] == 1) return 0;
                }
            }
        }
        return 1;
    }
    return 1;
}

// visible HOSTILE target (monster/detected/warning glyph) on the ray within
// 8 tiles. A peaceful in the line blocks the shot: ranged attacks have no
// "really attack?" prompt, so aiming at (or through) a watchman starts a war
static int nethack_ray_target(Nethack* env, int dx, int dy) {
    long hx = env->blstats[NLE_BL_X], hy = env->blstats[NLE_BL_Y];
    for (int k = 1; k <= 8; k++) {
        long x = hx + dx * k, y = hy + dy * k;
        if (x < 0 || x >= NH_COLS || y < 0 || y >= NH_ROWS) return 0;
        int gl = env->glyphs[y * NH_COLS + x];
        if ((gl >= 0 && gl < NETHACK_NUMMONS)
            || (gl >= 762 && gl < 1144)
            || (gl >= 5589 && gl < 5595))
            return !nle_peaceful_at(env->ctx, (int)x + 1, (int)y);
    }
    return 0;
}

static void nethack_compute_mask(Nethack* env) {
    unsigned char* mask = env->action_mask;
    memset(mask, 1, NETHACK_NUM_ACTIONS);
    if (env->mask_search20 != 0.0f) mask[NETHACK_ACT_SEARCH20] = 0;
    if (env->mask_run != 0.0f) mask[NETHACK_ACT_RUN] = 0;
    if (env->blstats[NLE_BL_HUNGER] == 0) mask[NETHACK_ACT_EAT] = 0; // choke gate

    // underfoot
    long hero_x = env->blstats[NLE_BL_X], hero_y = env->blstats[NLE_BL_Y];
    int underfoot = (hero_x >= 0 && hero_x < NH_COLS && hero_y >= 0 && hero_y < NH_ROWS)
           ? env->glyphs[hero_y * NH_COLS + hero_x] : -1;
    int on_object = (underfoot >= NETHACK_GLYPH_OBJ_LO && underfoot < NETHACK_GLYPH_OBJ_HI);
    int on_corpse = (underfoot >= NETHACK_GLYPH_BODY_OFF
                     && underfoot < NETHACK_GLYPH_BODY_OFF + NETHACK_NUMMONS);

    // terrain, not the map glyph: an object on the tile occludes the stairs
    // (underfoot_glyphs shows the top item), which used to mask DOWN off and
    // strand the agent on a littered staircase
    int terrain = nle_terrain_underfoot(env->ctx);
    if (terrain != NETHACK_GLYPH_DNSTAIR && terrain != NETHACK_GLYPH_DNLADDER)
        mask[NETHACK_ACT_DOWN] = 0;
    if (terrain != NETHACK_GLYPH_UPSTAIR && terrain != NETHACK_GLYPH_UPLADDER)
        mask[NETHACK_ACT_UP] = 0;
    if (env->blstats[NLE_BL_DEPTH] <= 1) mask[NETHACK_ACT_UP] = 0; // declined exit
    if (!on_object && !on_corpse) mask[NETHACK_ACT_PICKUP] = 0;
    if (terrain != NETHACK_GLYPH_ALTAR) mask[NETHACK_ACT_ALTAR_ID] = 0;
    // presence is public (the container renders); locked/empty is learnable
    if (!nle_container_at(env->ctx)) mask[NETHACK_ACT_TIP] = 0;

    // engrave-test needs an unidentified wand not yet tested this episode:
    // one test prints all its information, re-testing just drains charges
    int unid_wand = 0;
    for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i]; i++) {
        if (env->inv_oclasses[i] != 11
            || env->inv_state[i * NLE_INV_STATE_FIELDS + 6] != 0) continue;
        int lb = nethack_letter_bit(env->inv_letters[i]);
        if (lb >= 0 && (env->engid_tested & (1ULL << lb))) continue;
        unid_wand = 1;
        break;
    }
    if (!unid_wand) mask[NETHACK_ACT_ENGRAVE_ID] = 0;

    // ELBERETH: unengravable terrain (fountain/water/lava/air/cloud) and
    // levitating/engulfed both refuse for free
    int tg = terrain - 2359;
    if (tg == 31 || tg == 32 || tg == 34 || tg == 39 || tg == 40 || tg == 41)
        mask[NETHACK_ACT_ELBERETH] = 0;
    if ((env->blstats[NLE_BL_CONDITION] & 0x400L) || (env->internal[6] & 4))
        mask[NETHACK_ACT_ELBERETH] = 0;

    // spell-slot head: castable iff still known (retention > 0) and Pw covers
    // 5 * level; fail% is deliberately NOT masked (it's in the obs)
    unsigned char* sp = mask + NETHACK_NUM_ACTIONS + 12 * NETHACK_INV_SLOTS
                        + NETHACK_DIR_HEADS * NETHACK_NUM_DIRS;
    memset(sp, 0, NETHACK_SPELL_SLOTS);
    int castable = 0;
    for (int s = 0; s < env->n_spells && s < NETHACK_SPELL_SLOTS; s++) {
        if (env->spell_ids[s] > 0 && env->spell_knows[s] > 0
            && env->blstats[NLE_BL_ENE] >= 5L * (long)env->spell_levs[s]) {
            sp[s] = 1;
            castable = 1;
        }
    }
    if (!castable) sp[0] = 1; // unconsumed head still needs a legal entry
    // CAST zero-turn refusal mirror (engine predicate: stun, chant, freehand,
    // too-weak, hunger): a free refusal never advances the clock, so the
    // blocking condition can never expire -- self-sealing wedge
    if (!castable || env->internal[7] <= 10 || nle_cast_blocked(env->ctx))
        mask[NETHACK_ACT_CAST] = 0;
    // shop goods we can't pay for: picking them up incurs a bill the agent
    // has no way to settle, so gate on affordability (price is quoted to the
    // player on arrival, so this is public information)
    long shop_price = nle_shop_price(env->ctx);
    if (shop_price > env->blstats[NLE_BL_GOLD]) mask[NETHACK_ACT_PICKUP] = 0;

    // item slot heads
    for (int a = 0; a < NETHACK_NUM_ACTIONS; a++) {
        const Verb* verb = &NETHACK_VERBS[a];
        if (verb->head < 0) continue; // direct verb, no item argument
        unsigned char* slots = mask + NETHACK_NUM_ACTIONS + verb->head * NETHACK_INV_SLOTS;
        memset(slots, 0, NETHACK_INV_SLOTS);
        int has_usable = 0;
        for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i]; i++) {
            if (env->inv_oclasses[i] >= NETHACK_NUM_OCLASSES) break; // padded tail
            if (nethack_slot_usable(env, verb, i)) {
                slots[i] = 1;
                has_usable = 1;
            }
        }
        if (has_usable) continue;
        // no usable item: verb off unless EAT has actual floor food underfoot
        // (any-object gating kept EAT legal on inedible piles -- refusal loop)
        slots[0] = 1;
        int floor_food = (a == NETHACK_ACT_EAT) && (on_object || on_corpse)
                         && nle_food_underfoot(env->ctx);
        if (!floor_food) mask[a] = 0;
    }

    // per-verb dir rows: wall legality; THROW/ZAP use target rays (all-1 fallback)
    unsigned char* dirs = mask + NETHACK_NUM_ACTIONS + 12 * NETHACK_INV_SLOTS;
    memset(dirs, 1, NETHACK_NUM_DIRS);
    int legal_dirs = 0;
    for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
        long col = hero_x + NETHACK_DIR_DX[d], row = hero_y + NETHACK_DIR_DY[d];
        if (row < 0 || row >= NH_ROWS || col < 0 || col >= NH_COLS) {
            dirs[d] = 0;
            continue;
        }
        int g = env->glyphs[row * NH_COLS + col];
        if (g >= NETHACK_WALL_GLYPH_LO && g <= NETHACK_WALL_GLYPH_HI) dirs[d] = 0;
        else legal_dirs++;
    }
    if (!legal_dirs) memset(dirs, 1, NETHACK_NUM_DIRS);
    for (int h = 1; h < NETHACK_DIR_HEADS; h++)
        memcpy(dirs + h * NETHACK_NUM_DIRS, dirs, NETHACK_NUM_DIRS);

    // MOVE-head refinements: peaceful-adjacent and diagonal-door moves are void
    unsigned char keep[NETHACK_NUM_DIRS];
    memcpy(keep, dirs, NETHACK_NUM_DIRS);
    int on_door = (tg >= 12 && tg <= 14);
    for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
        if (!dirs[d]) continue;
        long col = hero_x + NETHACK_DIR_DX[d], row = hero_y + NETHACK_DIR_DY[d];
        if (row < 0 || row >= NH_ROWS || col < 0 || col >= NH_COLS) continue;
        int g = env->glyphs[row * NH_COLS + col];
        if (g >= 0 && g < 381
            && nle_peaceful_at(env->ctx, (int)col + 1, (int)row)) {
            dirs[d] = 0;
            continue;
        }
        if (d >= 4 && (on_door || (g >= 2371 && g <= 2373))) dirs[d] = 0;
    }
    int open_moves = 0;
    for (int d = 0; d < NETHACK_NUM_DIRS; d++) open_moves |= dirs[d];
    if (!open_moves) memcpy(dirs, keep, NETHACK_NUM_DIRS); // cornered: restore

    // RUN head: a run aimed at an adjacent hostile is void with p=1.0;
    // directional only -- escape dirs stay
    unsigned char* rdirs = dirs + 1 * NETHACK_NUM_DIRS;
    unsigned char rkeep[NETHACK_NUM_DIRS];
    memcpy(rkeep, rdirs, NETHACK_NUM_DIRS);
    for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
        if (!rdirs[d]) continue;
        long col = hero_x + NETHACK_DIR_DX[d], row = hero_y + NETHACK_DIR_DY[d];
        if (row < 0 || row >= NH_ROWS || col < 0 || col >= NH_COLS) continue;
        int g = env->glyphs[row * NH_COLS + col];
        if (g >= 0 && g < 381 && !nle_peaceful_at(env->ctx, (int)col + 1, (int)row))
            rdirs[d] = 0;
    }
    int open_runs = 0;
    for (int d = 0; d < NETHACK_NUM_DIRS; d++) open_runs |= rdirs[d];
    if (!open_runs) memcpy(rdirs, rkeep, NETHACK_NUM_DIRS);

    static const int ray_verbs[2] = {NETHACK_ACT_THROW, NETHACK_ACT_ZAP};
    for (int v = 0; v < 2; v++) {
        unsigned char* vdirs = dirs + nethack_dir_head(ray_verbs[v]) * NETHACK_NUM_DIRS;
        int any = 0;
        for (int d = 0; d < NETHACK_NUM_DIRS; d++) {
            vdirs[d] = nethack_ray_target(env, NETHACK_DIR_DX[d], NETHACK_DIR_DY[d]);
            any |= vdirs[d];
        }
        if (!any) memset(vdirs, 1, NETHACK_NUM_DIRS);
    }
}

// observations

static void nethack_pack_obs(Nethack* env) {
    obs_t* obs_buf = env->agents[0].observations;
    memcpy(obs_buf + NETHACK_OFF_GLYPHS, env->glyphs, sizeof(env->glyphs));
    unsigned char* bl = obs_buf + NETHACK_OFF_BLSTATS;
    for (int i = 0; i < NLE_BLSTATS_SIZE; i++) {
        uint32_t v = (uint32_t)(int32_t)env->blstats[i];
        bl[4*i + 0] = (unsigned char)(v & 0xffu);
        bl[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        bl[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        bl[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    int32_t extra[NETHACK_EXTRA_INTS] = {0};
    // engraving state 0/1/2 (bit 2 = engulfed, mask-only -- keep it out of obs)
    extra[0] = env->internal[6] & 3;
    extra[1] = env->prev_action;
    for (int i = 0; i < NLE_INVENTORY_SIZE; i++) {
        int oc = env->inv_oclasses[i];
        if (oc >= NETHACK_NUM_OCLASSES) break; // padded tail
        extra[2 + oc]++;
    }

    // 8-slot spell channel with retention (know 0 = forgotten, the re-read cue)
    env->n_spells = nle_spells(env->ctx, env->spell_ids, env->spell_levs,
                                env->spell_fails, env->spell_knows,
                                NETHACK_SPELL_SLOTS);
    extra[NETHACK_EXTRA_SPELL] = env->n_spells;
    for (int s = 0; s < NETHACK_SPELL_SLOTS; s++) {
        int* q = extra + NETHACK_EXTRA_SPELL + 1 + 4 * s;
        int known = s < env->n_spells && env->spell_ids[s] > 0;
        q[0] = known ? env->spell_ids[s] : 0;
        q[1] = known ? env->spell_levs[s] : 0;
        q[2] = known ? env->spell_fails[s] : 0;
        q[3] = known ? env->spell_knows[s] : 0;
    }

    extra[NETHACK_EXTRA_ROLEOH + env->role_idx] = 1;
    extra[NETHACK_EXTRA_ROLEOH + 13 + env->race_idx] = 1;
    extra[NETHACK_EXTRA_ROLEOH + 18 + env->gend_idx] = 1;

    int wt, wcap;
    nle_weight(env->ctx, &wt, &wcap);
    if (wcap < 1) wcap = 1;
    extra[NETHACK_EXTRA_WEIGHT + 0] = (int)(100L * wt / wcap);
    extra[NETHACK_EXTRA_WEIGHT + 1] = wcap;

    // gold/price as a percent, capped at 100 (0 = no purchase available here)
    long price = nle_shop_price(env->ctx);
    long gold = env->blstats[NLE_BL_GOLD];
    extra[NETHACK_EXTRA_SHOP] = nle_inside_shop(env->ctx);
    extra[NETHACK_EXTRA_SHOP + 1] = (price > 0)
        ? (int32_t)(gold >= price ? 100 : (gold * 100) / price) : 0;

    unsigned char* ex = obs_buf + NETHACK_OFF_EXTRA;
    for (int i = 0; i < NETHACK_EXTRA_INTS; i++) {
        uint32_t v = (uint32_t)extra[i];
        ex[4*i + 0] = (unsigned char)(v & 0xffu);
        ex[4*i + 1] = (unsigned char)((v >> 8) & 0xffu);
        ex[4*i + 2] = (unsigned char)((v >> 16) & 0xffu);
        ex[4*i + 3] = (unsigned char)((v >> 24) & 0xffu);
    }
    // slot glyphs; NH_DISC_SWAP=1: discovered identity REPLACES the appearance
    // glyph (stable rep once known; the add-channel below goes silent)
    static int dsw = -1;
    if (dsw < 0) {
        const char* e = getenv("NH_DISC_SWAP");
        dsw = e && e[0] && e[0] != '0';
    }
    unsigned char* iv = obs_buf + NETHACK_OFF_INV;
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        uint16_t g = env->inv_oclasses[i] < NETHACK_NUM_OCLASSES
                   ? (dsw && env->inv_true[i] != NETHACK_PAD_GLYPH
                      ? (uint16_t)env->inv_true[i] : (uint16_t)env->inv_glyphs[i])
                   : (uint16_t)NETHACK_PAD_GLYPH;
        iv[2*i + 0] = (unsigned char)(g & 0xffu);
        iv[2*i + 1] = (unsigned char)((g >> 8) & 0xffu);
    }
    // item state
    memcpy(obs_buf + NETHACK_OFF_INVST, env->inv_state, sizeof(env->inv_state));
    // discovered-type glyphs (engine pads with NO_GLYPH == NETHACK_PAD_GLYPH)
    unsigned char* it = obs_buf + NETHACK_OFF_INVTRUE;
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        uint16_t g = !dsw && env->inv_oclasses[i] < NETHACK_NUM_OCLASSES
                   ? (uint16_t)env->inv_true[i] : (uint16_t)NETHACK_PAD_GLYPH;
        it[2*i + 0] = (unsigned char)(g & 0xffu);
        it[2*i + 1] = (unsigned char)((g >> 8) & 0xffu);
    }
    // stall watchdog: a key-eating modal (getpos-class) can freeze the game
    // turn while swallowing every action; at 96 frozen steps fire an ESC burst
    // (ESC is a harmless no-op at the main prompt)
    long turn = env->blstats[NLE_BL_TIME];
    if (turn == env->stall_prev_turn && !env->obs.done) {
        if (++env->stall_ctr == 96) {
            for (int k = 0; k < 8 && !env->obs.done; k++) {
                env->obs.action = 27;
                nethack_engine_step(env);
            }
            env->stall_ctr = 0; // re-arm; recovery shows as turn advance
        }
    } else {
        env->stall_ctr = 0;
        env->stall_prev_turn = turn;
    }
    // topline
    unsigned char* mv = obs_buf + NETHACK_OFF_MSG;
    size_t mlen = strnlen((const char*)env->message, NETHACK_MSG_LEN);
    memcpy(mv, env->message, mlen);
    if (mlen < (size_t)NETHACK_MSG_LEN) memset(mv + mlen, 0, NETHACK_MSG_LEN - mlen);

    if (env->action_mask != NULL) nethack_compute_mask(env);
}

// logging

static void nethack_add_log(Nethack* env, int how) { // how: nle how_done, -1 = truncated
    for (int v = 0; v < NETHACK_NUM_ACTIONS; v++)
        env->log.verb_uses[v] += (float)env->stats.verb_uses[v];
    env->log.perf += (float)env->prev_score;
    env->log.score += (float)env->prev_score;
    env->log.valid_moves += (float)env->stats.valid_moves;
    env->log.illegal_actions += (float)env->stats.illegal_actions;
    env->log.new_tiles += (float)env->stats.new_tiles;
    env->log.max_depth += (float)env->stats.max_depth;
    env->log.floors += (float)env->stats.floors;
    env->log.depth_5 += env->stats.max_depth >= 5 ? 1.0f : 0.0f;
    env->log.depth_10 += env->stats.max_depth >= 10 ? 1.0f : 0.0f;
    env->log.depth_15 += env->stats.max_depth >= 15 ? 1.0f : 0.0f;
    env->log.mines_depth += (float)__builtin_popcountll(env->stats.floors_bits[2]);
    env->log.sokoban_depth += (float)__builtin_popcountll(env->stats.floors_bits[4]);
    env->log.scout_held += (float)env->stats.scout_held;
    env->log.enhances += (float)env->stats.enhances;
    env->log.floor_eats += (float)env->stats.floor_eats;
    env->log.reads_scroll += (float)env->stats.reads_scroll;
    env->log.reads_book += (float)env->stats.reads_book;
    env->log.sells += (float)env->stats.sells;
    env->log.buys += (float)env->stats.buys;
    env->log.role_ix += (float)env->role_idx;
    env->log.discoveries += (float)(nle_discoveries(env->ctx) - env->disc0);
    env->log.min_ac += (float)env->stats.min_ac;
    env->log.burdened_frac += env->stats.length > 0
        ? (float)env->stats.burdened_steps / (float)env->stats.length : 0.0f;
    env->log.game_time += (float)env->prev_time;
    env->log.max_xp_level += (float)env->stats.max_xp;
    env->log.episode_return += env->stats.ret;
    env->log.episode_length += env->stats.length;
    if (how == -1) env->log.truncated += 1.0f;
    else if (how == 0) env->log.death_combat += 1.0f;
    else if (how == 3) env->log.death_starved += 1.0f;
    else env->log.death_other += 1.0f;
    if (how >= 0 && env->stats.last_hunger >= NETHACK_HUNGER_WEAK)
        env->log.death_weak += 1.0f;
    if (how == 0)
        env->log.death_mon_level += (float)env->internal[NETHACK_INTERNAL_KILLER_MLEV];
    if (how >= 0) env->log.death_ac += (float)env->stats.last_ac;
    env->log.n += 1.0f;
}

// reset

static void nethack_do_reset(Nethack* env) {
    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }

    nethack_bind_obs(env);
    env->obs.how_done = -2; // only really_done() sets it

    // discovered types render as true glyphs on the map (engine opt-in)
    setenv("NLE_TRUE_GLYPHS", "1", 1);

    // seed advance
    env->seed = env->seed * 6364136223846793005UL + 1442695040888963407UL;
    // engine-random character per reset; identity read back after start
    if (env->multi_role != 0.0f) {
        char rcp[512];
        snprintf(env->settings.options, sizeof(env->settings.options), "@%s",
                 nethack_rc_path_opts(rcp, sizeof(rcp),
                     "name:Agent,role:random,race:random,gender:random,"
                     "align:random," NETHACK_OPTIONS_TAIL "!status_updates"));
    }
    env->settings.initial_seeds.seeds[0] = env->seed;
    env->settings.initial_seeds.seeds[1] = env->seed ^ 0x9E3779B97F4A7C15UL;
    env->settings.initial_seeds.use_init_seeds = true;
    env->settings.time_seed = env->seed;
    env->settings.time_seed_is_set = true;
    env->ctx = nle_start(&env->obs, NULL, &env->settings);

    nethack_drain_prompts(env);
    {
        int r = 0, rc = 0, g = 0, a = 0;
        nle_identity(env->ctx, &r, &rc, &g, &a);
        env->role_idx = (r >= 0 && r < 13) ? r : 0;
        env->race_idx = (rc >= 0 && rc < 5) ? rc : 0;
        env->gend_idx = (g == 1) ? 1 : 0;
    }
    nle_obs_refresh(env->ctx, &env->obs); // full fill: prev_* seeds read blstats
    if (nethack_msg_tap) nethack_msg_tap(env); // welcome arrives pre-step

    env->prev_score = 0;
    env->prev_exp = env->blstats[NLE_BL_EXP];
    env->start_gold = env->blstats[NLE_BL_GOLD];
    env->prev_gold = 0; // clamped net gold
    env->prev_depth = (int)env->blstats[NLE_BL_DEPTH];
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_action = -1;
    short scratch[2 * NETHACK_PATH_MAX]; // discard boot-walk residue
    nle_path_drain(env->ctx, scratch, NETHACK_PATH_MAX);
    env->prev_floor = (unsigned)(env->blstats[NLE_BL_DNUM] << 8 | env->blstats[NLE_BL_DLEVEL]);
    env->disc0 = nle_discoveries(env->ctx);
    env->engid_tested = 0;
    env->enh_ready = 0;
    memset(&env->stats, 0, sizeof(env->stats));
    env->stats.max_depth = env->prev_depth;
    env->stats.max_xp = (int)env->blstats[NLE_BL_XP];
    env->stats.min_ac = (int)env->blstats[NLE_BL_AC];
    env->stats.last_ac = (int)env->blstats[NLE_BL_AC];
    env->prev_ac_led = (float)env->blstats[NLE_BL_AC];
    env->ac_account = 0.0f;
    nethack_pack_obs(env);
}

void puf_reset(Nethack* env) {
    env->pending_reset = 1;
}

// reward

static void nethack_update_stats(Nethack* env) {
    if (env->blstats[NLE_BL_CAP] > 0) env->stats.burdened_steps++;
    env->stats.last_hunger = (int)env->blstats[NLE_BL_HUNGER];
    env->prev_score = env->blstats[NLE_BL_SCORE];
    env->prev_time = env->blstats[NLE_BL_TIME];
    env->prev_depth = (int)env->blstats[NLE_BL_DEPTH];
}

// Fractional scout claim, keyed by (dnum, dlevel). A tile pays its full
// scout_coef only once the hero's xp level covers depth * scout_ready; below
// that it pays pro-rata and the remainder stays claimable by a stronger
// visit. Total over all visits is capped at 1.0, so revisiting cannot farm
// it. scout_ready <= 0 restores plain first-visit semantics.
static float nethack_tile_claim(Nethack* env, long dn, long dl, long px, long py) {
    if (px < 0 || px >= NH_COLS || py < 0 || py >= NH_ROWS) return 0.0f;
    if (dn < 0 || dn > 15 || dl < 1 || dl > 64) return 0.0f;
    unsigned short key = (unsigned short)(dn << 8 | dl);
    int d = -1;
    for (int i = 0; i < env->stats.n_visited_floors; i++) {
        if (env->stats.visited_key[i] != key) continue;
        d = i;
        break;
    }
    if (d < 0) {
        if (env->stats.n_visited_floors >= NETHACK_MAX_DEPTH) return 0.0f;
        d = env->stats.n_visited_floors++;
        env->stats.visited_key[d] = key;
    }
    int idx = (int)py * NH_COLS + (int)px;
    unsigned char prev = env->stats.visited[d][idx];
    if (env->scout_ready <= 0.0f) { // plain first-visit
        if (prev) return 0.0f;
        env->stats.visited[d][idx] = 1;
        return 1.0f;
    }
    int depth = (int)env->blstats[NLE_BL_DEPTH];
    if (depth < 1) depth = 1;
    int req = (int)((float)depth * env->scout_ready + 0.5f);
    if (req < 1) req = 1;
    if (req > 255) req = 255;
    int cap = env->stats.max_xp < req ? env->stats.max_xp : req; // max_xp is monotonic
    if (cap <= (int)prev) return 0.0f;
    env->stats.visited[d][idx] = (unsigned char)cap;
    return (float)(cap - (int)prev) / (float)req;
}

static float nethack_reward(Nethack* env) {
    // death payout
    if (env->obs.done) return env->death_penalty;
    nethack_update_stats(env);

    int depth = (int)env->blstats[NLE_BL_DEPTH];

    // term order is the record run's summation order: exp gold descent xp
    // scout ac. Float adds are order-sensitive; reordering breaks bit repro.
    float r = 0.0f;

    // exp, gains only
    long exp = env->blstats[NLE_BL_EXP];
    if (exp > env->prev_exp)
        r += env->exp_coef * (float)(exp - env->prev_exp);
    env->prev_exp = exp;

    // gold, net of start
    long g = env->blstats[NLE_BL_GOLD] - env->start_gold;
    if (g < 0) g = 0;
    r += env->gold_coef * (float)(g - env->prev_gold);
    env->prev_gold = g;

    // floor: pays once per new unique (dnum, dlevel) floor entered (branches count)
    long dn = env->blstats[NLE_BL_DNUM], dl = env->blstats[NLE_BL_DLEVEL];
    if (dn >= 0 && dn < 16 && dl >= 1 && dl <= 64) {
        unsigned long long fb = 1ULL << (dl - 1);
        if (!(env->stats.floors_bits[dn] & fb)) {
            env->stats.floors_bits[dn] |= fb;
            env->stats.floors++;
            r += env->floor_coef;
        }
    }

    // descent pays per max-depth delta
    if (depth > env->stats.max_depth) {
        r += env->descent_coef * (float)(depth - env->stats.max_depth);
        env->stats.max_depth = depth;
    }

    // xp level, max only
    int xp = (int)env->blstats[NLE_BL_XP];
    if (xp > env->stats.max_xp) {
        r += env->xp_coef * (float)(xp - env->stats.max_xp);
        env->stats.max_xp = xp;
    }

    // scout: pay every tile walked this step; a rush resolves many moves in
    // one nle_step, so drain the engine's path rather than crediting only
    // where the hero stopped. A mid-step level change (trapdoor, hole)
    // leaves path coords from the old floor -- skip those.
    unsigned floor = (unsigned)(dn << 8 | dl);
    short path[2 * NETHACK_PATH_MAX];
    int n = nle_path_drain(env->ctx, path, NETHACK_PATH_MAX);
    if (floor != env->prev_floor) n = 0;
    env->prev_floor = floor;

    float fresh = 0.0f;
    int touched = 0;
    for (int i = 0; i < n; i++) {
        float c = nethack_tile_claim(env, dn, dl, path[2 * i], path[2 * i + 1]);
        fresh += c;
        touched += (c > 0.0f);
    }
    // no usable path: non-move verbs or never left the tile
    if (!n) {
        float c = nethack_tile_claim(env, dn, dl,
                      env->blstats[NLE_BL_X], env->blstats[NLE_BL_Y]);
        fresh += c;
        touched += (c > 0.0f);
    }

    if (n && fresh < (float)n - 1e-6f) env->stats.scout_held++;
    if (fresh > 0.0f) {
        r += env->scout_coef * fresh;
        env->stats.new_tiles += touched;
    }

    // ac: delta reward through a conservation ledger -- at most +-ac_coef
    // pays per step and the remainder carries, so telescoping stays exact
    // under the clamp and churn nets zero. ac_nospell is the unpaid fraction
    // of protection-spell AC (1 = durable AC only; kills cast-cycle arbitrage).
    long ac = env->blstats[NLE_BL_AC];
    env->stats.last_ac = (int)ac;
    if ((int)ac < env->stats.min_ac) env->stats.min_ac = (int)ac;

    float ac_led = env->ac_nospell != 0.0f
        ? (float)ac + env->ac_nospell * (float)nle_spellprot(env->ctx) : (float)ac;
    env->ac_account += env->ac_coef * (env->prev_ac_led - ac_led);
    env->prev_ac_led = ac_led;

    float cap = env->ac_coef;
    float pay = env->ac_account > cap ? cap
              : (env->ac_account < -cap ? -cap : env->ac_account);
    env->ac_account -= pay;
    r += pay;

    return r;
}

// stepping

static void nethack_execute(Nethack* env, int verb, int slot, int dirkey, int* bad_pick) {
    Stats* st = &env->stats;
    switch (verb) {
    case NETHACK_ACT_MOVE:
        nethack_send_key(env, dirkey);
        break;
    case NETHACK_ACT_RUN:
        st->verb_uses[verb]++;
        nethack_send_key(env, dirkey - 32); // uppercase = run
        break;
    case NETHACK_ACT_DOWN:
        nethack_send_key(env, '>');
        break;
    case NETHACK_ACT_UP:
        nethack_send_key(env, '<');
        break;
    case NETHACK_ACT_KICK:
        nethack_send_key(env, 4); // ^D
        nethack_answer_direction(env, dirkey);
        break;
    case NETHACK_ACT_SEARCH:
        st->verb_uses[verb]++;
        nethack_send_key(env, 's');
        break;
    case NETHACK_ACT_ELBERETH:
        st->verb_uses[verb]++;
        nethack_do_elbereth(env);
        break;
    case NETHACK_ACT_SEARCH20:
        st->verb_uses[verb]++;
        nethack_send_key(env, '2');
        nethack_send_key(env, '0');
        nethack_send_key(env, 's');
        break;
    case NETHACK_ACT_PICKUP: {
        int purchase = nle_shop_price(env->ctx) > 0; // 0 = own/no-charge pile, nothing to pay
        st->verb_uses[verb]++;
        nethack_send_key(env, ',');
        nethack_answer_menu(env);
        // shop pickup bills you; settle it now (the mask guarantees we can)
        if (purchase && !env->obs.done) {
            nethack_send_key(env, 'p');
            st->buys++;
        }
        break;
    }
    case NETHACK_ACT_PRAY:
        st->verb_uses[verb]++;
        nethack_send_key(env, 0x80 | 'p');
        break;
    case NETHACK_ACT_WEAR:
        nethack_wear_takeoff_conflict(env, slot);
        nethack_item_use(env, 'W', "want to wear", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_EAT:
        nethack_item_use(env, 'e', "want to eat", "eat it", slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_QUAFF:
        nethack_item_use(env, 'q', "want to drink", "rink from the", slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_THROW:
        if (nethack_item_use(env, 't', "want to throw", NULL, slot, &st->verb_uses[verb], bad_pick))
            nethack_answer_direction(env, dirkey);
        break;
    case NETHACK_ACT_ZAP:
        if (nethack_item_use(env, 'z', "want to zap", NULL, slot, &st->verb_uses[verb], bad_pick))
            nethack_answer_direction(env, dirkey);
        break;
    case NETHACK_ACT_TAKEOFF:
        nethack_item_use(env, 'T', "take off", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_PUTON:
        nethack_item_use(env, 'P', "put on", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_REMOVE:
        nethack_item_use(env, 'R', "remove", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    case NETHACK_ACT_WIELD:
        nethack_verb_wield(env, slot, bad_pick);
        break;
    case NETHACK_ACT_APPLY:
        if (nethack_item_use(env, 'a', "apply", NULL, slot, &st->verb_uses[verb], bad_pick)) {
            // diggers dig down
            int otyp = env->inv_glyphs[slot] - NH_GLYPH_OBJ_OFF;
            nethack_answer_direction(env,
                (otyp == 234 /* PICK_AXE */ || otyp == 50 /* MATTOCK */) ? '>' : dirkey);
        }
        break;
    case NETHACK_ACT_READ: {
        int oc = env->inv_oclasses[slot];
        if (nethack_item_use(env, 'r', "read", NULL, slot, &st->verb_uses[verb], bad_pick)) {
            if (oc == 9) st->reads_scroll++;
            else if (oc == 10) st->reads_book++;
            nethack_answer_menu(env);
        }
        break;
    }
    case NETHACK_ACT_ALTAR_ID: {
        // each drop onto an altar flashes the item's curse state (sets bknown),
        // so dump everything still unknown and take it straight back
        st->verb_uses[verb]++;
        for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i] && !env->obs.done; i++) {
            if (env->inv_oclasses[i] >= NETHACK_NUM_OCLASSES) break;
            if (env->inv_oclasses[i] == 12) continue; // gold: no flash
            const signed char* st8 = &env->inv_state[i * NLE_INV_STATE_FIELDS];
            if (st8[0] != 0 || (st8[5] & 1)) continue;  // BUC known, or worn
            nethack_item_use(env, 'd', "drop", NULL, i, NULL, NULL);
        }
        if (!env->obs.done) {
            nethack_send_key(env, ',');
            nethack_answer_menu(env);
        }
        break;
    }
    case NETHACK_ACT_TIP:
        // M('T') = #tip; the "tip it? [ynq]" prompt auto-commits, spillage
        // lands underfoot for PICKUP. Locked -> "It's locked." (kick first).
        st->verb_uses[verb]++;
        nethack_send_key(env, 0x80 | 'T');
        break;
    case NETHACK_ACT_ENGRAVE_ID:
        st->verb_uses[verb]++;
        nethack_do_engrave_id(env);
        break;
    case NETHACK_ACT_CAST: {
        // Z, then 'a' = first known spell; directional spells then prompt and
        // take the (ZAP-masked) direction, self-spells resolve immediately
        st->verb_uses[verb]++;
        nethack_send_key(env, 'Z');
        // proceed only if the spell chooser menu actually opened (xwait):
        // a zero-turn refusal (unmirrored rejectcasting case) leaves the
        // main prompt, where 'a' would open an apply prompt instead
        if (!env->obs.done && env->misc[NETHACK_MISC_XWAIT]) {
            nethack_send_key(env, 'a' + (slot >= 0 && slot < NETHACK_SPELL_SLOTS ? slot : 0));
            nethack_answer_direction(env, dirkey);
        }
        break;
    }
    case NETHACK_ACT_DROP:
        nethack_item_use(env, 'd', "drop", NULL, slot, &st->verb_uses[verb], bad_pick);
        break;
    }
}

// compute_mask writes through this flat alias of agents[0].action_mask
static void nethack_sync_buffers(Nethack* env) {
    env->action_mask = env->agents[0].action_mask;
}

void puf_step(Nethack* env) {
    nethack_sync_buffers(env); // agent pointers are re-dealt between epochs
    if (env->pending_reset) {
        env->pending_reset = 0;
        nethack_do_reset(env);
    }

    int verb = (int)env->agents[0].actions[0];
    int head = NETHACK_VERBS[verb].head;
    int slot = (head >= 0) ? (int)env->agents[0].actions[1 + head] : 0;
    if (verb == NETHACK_ACT_CAST) slot = (int)env->agents[0].actions[19];
    int dh = nethack_dir_head(verb);
    int dirkey = NETHACK_DIR_KEYS[dh >= 0 ? (int)env->agents[0].actions[13 + dh] : 0];

    long time_before = env->blstats[NLE_BL_TIME];
    int bad_pick = 0;
    nethack_execute(env, verb, slot, dirkey, &bad_pick);

    env->prev_action = verb;
    nethack_handle_prompts(env);
    if (!env->obs.done) nle_obs_refresh(env->ctx, &env->obs);
    nethack_auto_enhance(env);

    if (bad_pick) env->stats.illegal_actions++;
    if (env->blstats[NLE_BL_TIME] > time_before) env->stats.valid_moves++;
    env->stats.length++;

    float reward = nethack_reward(env);
    env->agents[0].rewards[0] = reward;
    env->stats.ret += reward;

    int done = env->obs.done || env->stats.length >= NETHACK_MAX_EPISODE_STEPS;
    env->agents[0].terminals[0] = done ? 1.0f : 0.0f; // truncation reported as terminal too
    if (done) {
        nethack_add_log(env, env->obs.done ? env->obs.how_done : -1);
        // eager same-thread reset: the terminal step returns the fresh obs
        nethack_do_reset(env);
    } else {
        nethack_pack_obs(env);
    }
}

void puf_close(Nethack* env) {
    if (env->ctx != NULL) {
        nle_end(env->ctx);
        env->ctx = NULL;
    }
    nethack_rm_vardir(env->vardir);
    env->vardir[0] = '\0';
}

void puf_render(Nethack* env) {
    printf("\x1b[H\x1b[2J");
    for (int r = 0; r < NH_ROWS; r++) {
        for (int c = 0; c < NH_COLS; c++) {
            unsigned char ch = env->chars[r * NH_COLS + c];
            putchar(ch ? ch : ' ');
        }
        putchar('\n');
    }
    printf("HP %ld/%ld  AC %ld  Dlvl %ld  Score %ld  T %ld\n",
           env->blstats[NLE_BL_HP], env->blstats[NLE_BL_HPMAX],
           env->blstats[NLE_BL_AC], env->blstats[NLE_BL_DEPTH],
           env->blstats[NLE_BL_SCORE], env->blstats[NLE_BL_TIME]);
    if (env->n_spells > 0)
        printf("Sp: id%d L%d fail%d%%  x%d\n", env->spell_ids[0],
               env->spell_levs[0], env->spell_fails[0], env->n_spells);
    printf("Msg: %.*s\n", NLE_MESSAGE_SIZE, env->message);
    fflush(stdout);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    init(env);
    env->gold_coef = dict_get(kwargs, "gold_coef");
    env->exp_coef = dict_get(kwargs, "exp_coef");
    env->descent_coef = dict_get(kwargs, "descent_coef");
    env->floor_coef = dict_get(kwargs, "floor_coef");
    env->scout_coef = dict_get(kwargs, "scout_coef");
    env->ac_coef = dict_get(kwargs, "ac_coef");
    env->scout_ready = dict_get(kwargs, "scout_ready");
    env->ac_nospell = dict_get(kwargs, "ac_nospell");
    env->xp_coef = dict_get(kwargs, "xp_coef");
    env->death_penalty = dict_get(kwargs, "death_penalty");
    env->mask_search20 = dict_get(kwargs, "mask_search20");
    env->mask_run = dict_get(kwargs, "mask_run");
    env->multi_role = dict_get(kwargs, "multi_role");
}

// Export order: outcomes first (score/depth/reaches/deaths), then action
// stats, then per-step diagnostics. Dict insertion order is what the
// dashboard/CSV see.
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_depth", log->max_depth);
    dict_set(out, "min_ac", log->min_ac);
    dict_set(out, "depth_5", log->depth_5);
    dict_set(out, "depth_10", log->depth_10);
    dict_set(out, "depth_15", log->depth_15);
    dict_set(out, "mines_depth", log->mines_depth);
    dict_set(out, "sokoban_depth", log->sokoban_depth);
    dict_set(out, "sells", log->sells);
    dict_set(out, "buys", log->buys);
    dict_set(out, "role_ix", log->role_ix);
    dict_set(out, "discoveries", log->discoveries);
    dict_set(out, "death_combat", log->death_combat);
    dict_set(out, "death_weak", log->death_weak);
    dict_set(out, "death_starved", log->death_starved);
    dict_set(out, "death_other", log->death_other);
    dict_set(out, "death_mon_level", log->death_mon_level);
    dict_set(out, "death_ac", log->death_ac);
    for (int v = 0; v < NETHACK_NUM_ACTIONS; v++) {
        if (NETHACK_VERB_STAT[v])
            dict_set(out, NETHACK_VERB_STAT[v], log->verb_uses[v]);
    }
    dict_set(out, "valid_moves", log->valid_moves);
    dict_set(out, "illegal_actions", log->illegal_actions);
    dict_set(out, "new_tiles", log->new_tiles);
    dict_set(out, "enhances", log->enhances);
    dict_set(out, "floor_eats", log->floor_eats);
    dict_set(out, "reads_scroll", log->reads_scroll);
    dict_set(out, "reads_book", log->reads_book);
    dict_set(out, "burdened_frac", log->burdened_frac);
    dict_set(out, "game_time", log->game_time);
    dict_set(out, "max_xp_level", log->max_xp_level);
    dict_set(out, "floors", log->floors);
    dict_set(out, "scout_held", log->scout_held);
    dict_set(out, "truncated", log->truncated);
    dict_set(out, "n", log->n);
}

// Per-(verb,head) consumption map for PPO consumed-head gating (weak symbol
// read by src/algo.cu). heads: [0]=verb, [1..12]=slot heads 0..11,
// [13..18]=per-verb dir heads, [19]=spell-slot head (CAST). A head is
// "consumed" iff the sampled verb actually uses it.
#define PUFFER_PROVIDES_HEAD_CONSUME_MAP 1
const signed char* env_head_consume_map(int* n_verbs, int* n_atns) {
    static signed char map[NETHACK_NUM_ACTIONS * NUM_ATNS];
    static int built = 0;
    if (!built) {
        memset(map, 0, sizeof(map));
        for (int v = 0; v < NETHACK_NUM_ACTIONS; v++) {
            signed char* row = map + v * NUM_ATNS;
            row[0] = 1; // verb head: always
            int sh = NETHACK_VERBS[v].head; // slot head 0..11 or -1
            if (sh >= 0) row[1 + sh] = 1;
            int dh = nethack_dir_head(v);
            if (dh >= 0) row[13 + dh] = 1;
            if (v == NETHACK_ACT_CAST) row[19] = 1; // spell-slot head
        }
        built = 1;
    }
    *n_verbs = NETHACK_NUM_ACTIONS;
    *n_atns = NUM_ATNS;
    return map;
}
