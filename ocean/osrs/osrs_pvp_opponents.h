/**
 * @fileoverview Scripted opponent policies implemented in C.
 *
 * Ports the Python opponent policies (opponents/ *.py) to C for use within
 * c_step(). Eliminates the Python round-trip for opponent action
 * generation during training with scripted opponents.
 *
 * Opponent reads game state directly from Player structs instead of parsing
 * observation arrays, which is both faster and avoids float normalization.
 *
 * Actions are direct head-value assignments: int actions[NUM_ACTION_HEADS].
 * Gear switches use loadout presets (LOADOUT_MELEE, LOADOUT_RANGE, etc.)
 * instead of per-slot equip actions.
 *
 * Phase 1 policies: TrueRandom, Panicking, WeakRandom, SemiRandom,
 * StickyPrayer, Beginner, BetterRandom, Improved.
 * Phase 2 policies: Onetick, UnpredictableImproved, UnpredictableOnetick.
 * Mixed wrappers: MixedEasy, MixedMedium, MixedHard, MixedHardBalanced.
 */

#ifndef OSRS_PVP_OPPONENTS_H
#define OSRS_PVP_OPPONENTS_H

/* This header is included from osrs_pvp.h AFTER all other headers,
 * so osrs_types.h, osrs_items.h (via gear.h), and
 * osrs_pvp_actions.h are already available. */

/* OpponentType enum and OpponentState struct are in osrs_types.h */

/* Attack style enum for opponent internal use */
#define OPP_STYLE_MAGE    0
#define OPP_STYLE_RANGED  1
#define OPP_STYLE_MELEE   2
#define OPP_STYLE_SPEC    3

/* =========================================================================
 * Utility: map OPP_STYLE_* to LOADOUT_* presets
 * ========================================================================= */

static inline int opp_style_to_loadout(int style) {
    switch (style) {
        case OPP_STYLE_MAGE:   return LOADOUT_MAGE;
        case OPP_STYLE_RANGED: return LOADOUT_RANGE;
        case OPP_STYLE_MELEE:  return LOADOUT_MELEE;
        case OPP_STYLE_SPEC:   return LOADOUT_SPEC_MELEE;
        default: return LOADOUT_KEEP;
    }
}

static inline void opp_apply_gear_switch(int* actions, int style) {
    actions[HEAD_LOADOUT] = opp_style_to_loadout(style);
}

/* Fake switch: same loadout set, no attack action follows */
static inline void opp_apply_fake_switch(int* actions, int style) {
    actions[HEAD_LOADOUT] = opp_style_to_loadout(style);
}

/* Tank gear: LOADOUT_TANK equips dhide body, rune legs, spirit shield */
static inline void opp_apply_tank_gear(int* actions) {
    actions[HEAD_LOADOUT] = LOADOUT_TANK;
}

/* =========================================================================
 * Consumable availability helpers
 * ========================================================================= */

typedef struct {
    int can_food;
    int can_brew;
    int can_karambwan;
    int can_restore;
    int can_combat_pot;
    int can_ranged_pot;
} OppConsumables;

static inline void opp_tick_cooldowns(OpponentState* opp) {
    if (opp->food_cooldown > 0) opp->food_cooldown--;
    if (opp->potion_cooldown > 0) opp->potion_cooldown--;
    if (opp->karambwan_cooldown > 0) opp->karambwan_cooldown--;
}

static inline OppConsumables opp_get_consumables(OpponentState* opp, Player* self) {
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    OppConsumables c;
    c.can_food = (opp->food_cooldown <= 0 && self->food_count > 0 && hp_pct < 1.0f);
    c.can_brew = (opp->potion_cooldown <= 0 && self->brew_doses > 0);
    c.can_karambwan = (opp->karambwan_cooldown <= 0 && self->karambwan_count > 0 && hp_pct < 1.0f);
    c.can_restore = (opp->potion_cooldown <= 0 && self->restore_doses > 0);
    c.can_combat_pot = (opp->potion_cooldown <= 0 && self->combat_potion_doses > 0);
    c.can_ranged_pot = (opp->potion_cooldown <= 0 && self->ranged_potion_doses > 0);
    return c;
}

/* (opp_apply_gear_switch is defined above as inline loadout assignment) */

/* =========================================================================
 * Prayer helpers
 * ========================================================================= */

static inline AttackStyle opp_get_gear_style(Player* p) {
    int s = get_item_attack_style(p->equipped[GEAR_SLOT_WEAPON]);
    if (s == 3) return ATTACK_STYLE_MAGIC;
    if (s == 2) return ATTACK_STYLE_RANGED;
    if (s == 1) return ATTACK_STYLE_MELEE;
    return ATTACK_STYLE_MAGIC;  /* Default */
}

static inline int opp_get_defensive_prayer(Player* target) {
    AttackStyle target_style = opp_get_gear_style(target);
    if (target_style == ATTACK_STYLE_MAGIC)  return OVERHEAD_MAGE;
    if (target_style == ATTACK_STYLE_RANGED) return OVERHEAD_RANGED;
    if (target_style == ATTACK_STYLE_MELEE)  return OVERHEAD_MELEE;
    return OVERHEAD_MAGE;  /* Default to mage */
}

static inline int opp_has_prayer_active(Player* self, int prayer_action) {
    if (prayer_action == OVERHEAD_MELEE)  return self->prayer == PRAYER_PROTECT_MELEE;
    if (prayer_action == OVERHEAD_RANGED) return self->prayer == PRAYER_PROTECT_RANGED;
    if (prayer_action == OVERHEAD_MAGE)   return self->prayer == PRAYER_PROTECT_MAGIC;
    return 0;
}

/* =========================================================================
 * Attack style helpers
 * ========================================================================= */

static inline int opp_attack_ready(Player* self) {
    return self->attack_timer <= 0;
}

static inline int opp_can_reach_melee(Player* self, Player* target) {
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    return dist <= 1 || (self->frozen_ticks == 0 && dist <= 5);
}

/**
 * Get off-prayer attack styles (styles target isn't protecting).
 * Returns bitmask: bit 0 = mage, bit 1 = ranged, bit 2 = melee
 */
static inline int opp_get_off_prayer_mask(Player* self, Player* target) {
    int mask = 0;
    if (target->prayer != PRAYER_PROTECT_MAGIC)   mask |= (1 << OPP_STYLE_MAGE);
    if (target->prayer != PRAYER_PROTECT_RANGED)  mask |= (1 << OPP_STYLE_RANGED);
    if (target->prayer != PRAYER_PROTECT_MELEE && opp_can_reach_melee(self, target))
        mask |= (1 << OPP_STYLE_MELEE);
    if (mask == 0) mask = (1 << OPP_STYLE_MAGE);  /* Fallback to mage */
    return mask;
}

static inline int opp_pick_from_mask(OsrsEnv* env, int mask) {
    /* Count set bits and pick random */
    int choices[3];
    int count = 0;
    for (int i = 0; i < 3; i++) {
        if (mask & (1 << i)) choices[count++] = i;
    }
    return choices[rand_int(env, count)];
}

static inline int opp_is_drained(Player* self) {
    // Any combat stat below base = drained (brew drain, SWH, etc.)
    return self->current_strength < self->base_strength ||
           self->current_attack < self->base_attack ||
           self->current_defence < self->base_defence ||
           self->current_ranged < self->base_ranged ||
           self->current_magic < self->base_magic;
}

static inline int opp_should_fc3(Player* self, Player* target) {
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    return target->freeze_immunity_ticks > 1 &&
           self->frozen_ticks == 0 &&
           self->attack_timer <= 2 &&
           dist > 3;
}

/* Anti-kite: update flee tracking based on distance trend */
static inline void opp_update_flee_tracking(OpponentState* opp, Player* self, Player* target) {
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    if (dist > opp->prev_dist_to_target && dist > 1) {
        opp->target_fleeing_ticks++;
    } else {
        opp->target_fleeing_ticks = 0;
    }
    opp->prev_dist_to_target = dist;
}

/* =========================================================================
 * Per-episode randomization: ranges table for all opponent types
 * ========================================================================= */

typedef struct { float base; float variance; } RandRange;

typedef struct {
    RandRange prayer_accuracy;
    RandRange off_prayer_rate;
    RandRange offensive_prayer_rate;
    RandRange action_delay_chance;
    RandRange mistake_rate;
    RandRange offensive_prayer_miss;  /* chance to attack without loadout switch (skips auto-prayer) */
} OpponentRandRanges;

#define RR(b, v) {(b), (v)}

/*                                    pray_acc      off_pray      off_pray_r    act_delay      mistake        off_pray_miss */
static const OpponentRandRanges OPP_RAND_RANGES[OPP_RANGE_KITER + 1] = {
    [OPP_NONE]                  = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_TRUE_RANDOM]           = { RR(0.33,0),   RR(0.33,0),   RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_PANICKING]             = { RR(0.33,0.1), RR(0.33,0),   RR(0,0),      RR(0.10,0.05), RR(0,0),       RR(0,0) },
    [OPP_WEAK_RANDOM]           = { RR(0.40,0.1), RR(0.33,0.1), RR(0,0),      RR(0.10,0.05), RR(0.05,0.03), RR(0,0) },
    [OPP_SEMI_RANDOM]           = { RR(0.50,0.1), RR(0.40,0.1), RR(0.05,0.03),RR(0.08,0.04), RR(0.05,0.03), RR(0,0) },
    [OPP_STICKY_PRAYER]         = { RR(0.33,0),   RR(0.33,0),   RR(0,0),      RR(0.10,0.05), RR(0,0),       RR(0,0) },
    [OPP_RANDOM_EATER]          = { RR(0.40,0.1), RR(0.33,0.1), RR(0,0),      RR(0.08,0.04), RR(0.05,0.03), RR(0,0) },
    [OPP_PRAYER_ROOKIE]         = { RR(0.30,0.1), RR(0.20,0.1), RR(0,0),      RR(0.12,0.05), RR(0.08,0.04), RR(0,0) },
    [OPP_IMPROVED]              = { RR(0.95,0.05),RR(0.95,0.05),RR(0.80,0.10),RR(0.05,0.03), RR(0.03,0.02), RR(0.05,0.03) },
    [OPP_MIXED_EASY]            = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_MIXED_MEDIUM]          = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_ONETICK]               = { RR(0.97,0.03),RR(0.97,0.03),RR(0.90,0.05),RR(0.03,0.02), RR(0.02,0.01), RR(0.03,0.02) },
    [OPP_UNPREDICTABLE_IMPROVED]= { RR(0.92,0.05),RR(0.90,0.05),RR(0.75,0.10),RR(0.08,0.04), RR(0.05,0.03), RR(0.08,0.04) },
    [OPP_UNPREDICTABLE_ONETICK] = { RR(0.95,0.03),RR(0.95,0.03),RR(0.85,0.08),RR(0.05,0.03), RR(0.03,0.02), RR(0.05,0.03) },
    [OPP_MIXED_HARD]            = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_MIXED_HARD_BALANCED]   = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_PFSP]                  = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_NOVICE_NH]             = { RR(0.60,0.10),RR(0.10,0.05),RR(0.10,0.05),RR(0.15,0.05), RR(0.10,0.05), RR(0.30,0.10) },
    [OPP_APPRENTICE_NH]         = { RR(0.60,0.10),RR(0.20,0.08),RR(0.20,0.08),RR(0.12,0.05), RR(0.08,0.04), RR(0.30,0.10) },
    [OPP_COMPETENT_NH]          = { RR(0.75,0.08),RR(0.25,0.08),RR(0.25,0.08),RR(0.10,0.04), RR(0.06,0.03), RR(0.20,0.08) },
    [OPP_INTERMEDIATE_NH]       = { RR(0.85,0.05),RR(0.70,0.08),RR(0.50,0.10),RR(0.08,0.04), RR(0.05,0.03), RR(0.20,0.08) },
    [OPP_ADVANCED_NH]           = { RR(0.95,0.05),RR(0.90,0.05),RR(0.75,0.08),RR(0.05,0.03), RR(0.03,0.02), RR(0.10,0.05) },
    [OPP_PROFICIENT_NH]         = { RR(0.95,0.03),RR(0.92,0.04),RR(0.80,0.08),RR(0.04,0.02), RR(0.03,0.02), RR(0.10,0.05) },
    [OPP_EXPERT_NH]             = { RR(0.97,0.03),RR(0.95,0.03),RR(0.85,0.05),RR(0.03,0.02), RR(0.02,0.01), RR(0.10,0.05) },
    [OPP_MASTER_NH]             = { RR(0.98,0.02),RR(0.97,0.03),RR(0.90,0.05),RR(0.02,0.01), RR(0.01,0.01), RR(0.01,0.01) },
    [OPP_SAVANT_NH]             = { RR(0.98,0.02),RR(0.97,0.03),RR(0.90,0.05),RR(0.02,0.01), RR(0.01,0.01), RR(0.01,0.01) },
    [OPP_NIGHTMARE_NH]          = { RR(0.99,0.01),RR(0.98,0.02),RR(0.95,0.03),RR(0.01,0.01), RR(0.005,0.005),RR(0.01,0.01) },
    [OPP_VENG_FIGHTER]          = { RR(0.92,0.05),RR(0.90,0.05),RR(0.85,0.10),RR(0.03,0.02), RR(0.02,0.01), RR(0.05,0.03) },
    [OPP_BLOOD_HEALER]          = { RR(0.90,0.05),RR(0.88,0.05),RR(0.80,0.10),RR(0.05,0.03), RR(0.04,0.02), RR(0.05,0.03) },
    [OPP_GMAUL_COMBO]           = { RR(0.96,0.03),RR(0.95,0.03),RR(0.90,0.05),RR(0.03,0.02), RR(0.02,0.01), RR(0.02,0.01) },
    [OPP_RANGE_KITER]           = { RR(0.93,0.04),RR(0.93,0.04),RR(0.85,0.08),RR(0.04,0.02), RR(0.03,0.02), RR(0.04,0.02) },
};

#undef RR

static inline float rand_range(OsrsEnv* env, RandRange r) {
    float v = r.base + (rand_float(env) * 2.0f - 1.0f) * r.variance;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

/* Tick-level action delay: skip prayer/attack/movement this tick (keep eating) */
static inline int opp_should_skip_offensive(OsrsEnv* env, OpponentState* opp) {
    return rand_float(env) < opp->action_delay_chance;
}

/**
 * Pick off-prayer attack style weighted by per-episode style bias.
 * Uses style_bias[3] (mage/ranged/melee weights) to sample from the off-prayer mask.
 * Falls back to uniform random if no bias styles are available off-prayer.
 */
static inline int opp_pick_off_prayer_style_biased(OsrsEnv* env, OpponentState* opp,
                                                    Player* self, Player* target) {
    int off_mask = opp_get_off_prayer_mask(self, target);
    float weights[3] = {0};
    float total = 0;
    for (int i = 0; i < 3; i++) {
        if (off_mask & (1 << i)) {
            weights[i] = opp->style_bias[i];
            total += weights[i];
        }
    }
    if (total <= 0) return opp_pick_from_mask(env, off_mask);

    float r = rand_float(env) * total;
    float cum = 0;
    for (int i = 0; i < 3; i++) {
        cum += weights[i];
        if (r < cum) return i;
    }
    return opp_pick_from_mask(env, off_mask);
}

/* Prayer mistake: small chance to pick random prayer instead of optimal */
static inline int opp_apply_prayer_mistake(OsrsEnv* env, OpponentState* opp, int correct_prayer) {
    if (rand_float(env) < opp->mistake_rate) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        return prayers[rand_int(env, 3)];
    }
    return correct_prayer;
}

/* =========================================================================
 * Phase 2: probability constants for unpredictable policies
 * ========================================================================= */

/* unpredictable_improved prayer delays: 70% instant, 20% 1-tick, 8% 2-tick, 2% 3-tick */
static const float UNPREDICTABLE_IMP_PRAYER_CUM[] = {0.70f, 0.90f, 0.98f, 1.00f};
#define UNPREDICTABLE_IMP_PRAYER_CUM_LEN 4

/* unpredictable_improved action delays: 85% instant, 12% 1-tick, 3% 2-tick */
static const float UNPREDICTABLE_IMP_ACTION_CUM[] = {0.85f, 0.97f, 1.00f};
#define UNPREDICTABLE_IMP_ACTION_CUM_LEN 3

/* unpredictable_onetick prayer delays: 80% instant, 15% 1-tick, 4% 2-tick, 1% 3-tick */
static const float UNPREDICTABLE_OT_PRAYER_CUM[] = {0.80f, 0.95f, 0.99f, 1.00f};
#define UNPREDICTABLE_OT_PRAYER_CUM_LEN 4

/* unpredictable_onetick action delays: 90% instant, 8% 1-tick, 2% 2-tick */
static const float UNPREDICTABLE_OT_ACTION_CUM[] = {0.90f, 0.98f, 1.00f};
#define UNPREDICTABLE_OT_ACTION_CUM_LEN 3

/* mistake probabilities */
#define UNPREDICTABLE_IMP_WRONG_PRAYER      0.05f
#define UNPREDICTABLE_IMP_SUBOPTIMAL_ATTACK 0.03f
#define UNPREDICTABLE_OT_FAKE_FAIL          0.12f
#define UNPREDICTABLE_OT_WRONG_PREDICT      0.08f

/* =========================================================================
 * Phase 2: helper functions for onetick + unpredictable policies
 * ========================================================================= */

/* Weighted delay sampling from cumulative weight array */
static inline int opp_sample_delay(OsrsEnv* env, const float* cum_weights, int num_weights) {
    float r = rand_float(env);
    for (int i = 0; i < num_weights; i++) {
        if (r < cum_weights[i]) return i;
    }
    return num_weights - 1;
}

/* Defensive prayer based on visible gear (uses actual weapon damage type). */
static inline int opp_get_defensive_prayer_with_spec(Player* target) {
    if (target->visible_gear == GEAR_MELEE)  return OVERHEAD_MELEE;
    if (target->visible_gear == GEAR_RANGED) return OVERHEAD_RANGED;
    if (target->visible_gear == GEAR_MAGE)   return OVERHEAD_MAGE;
    return opp_get_defensive_prayer(target);
}

/* Get opponent's current prayer style as OPP_STYLE_* (-1 if none) */
static inline int opp_get_opponent_prayer_style(Player* target) {
    if (target->prayer == PRAYER_PROTECT_MAGIC)  return OPP_STYLE_MAGE;
    if (target->prayer == PRAYER_PROTECT_RANGED) return OPP_STYLE_RANGED;
    if (target->prayer == PRAYER_PROTECT_MELEE)  return OPP_STYLE_MELEE;
    return -1;
}

/* Get target's visible gear style as GearSet value. */
static inline int opp_get_target_gear_style(Player* target) {
    return (int)target->visible_gear;
}

/* Choose ice vs blood barrage based on freeze state and HP */
static inline int opp_get_mage_attack(Player* self, Player* target) {
    int can_freeze = target->freeze_immunity_ticks <= 1 && target->frozen_ticks == 0;
    if (can_freeze) return ATTACK_ICE;
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    return (hp_pct > 0.98f) ? ATTACK_ICE : ATTACK_BLOOD;
}

/* (opp_apply_tank_gear is defined above as inline loadout assignment) */

/* Boost/restore potion logic (before attack, used by onetick+ opponents) */
static void opp_apply_boost_potion(OsrsEnv* env, OpponentState* opp, int* actions,
                                    Player* self, int attack_style, int potion_used) {
    (void)env;
    if (potion_used) return;
    if (opp->potion_cooldown > 0) return;
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    /* If drained (brew drain / SWH) and HP safe, restore before boosting.
     * 0.90 threshold ensures we finish brewing to full HP before restoring
     * (one restore dose undoes ~3 brew doses of stat drain). */
    if (opp_is_drained(self) && hp_pct > 0.90f && self->restore_doses > 0) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
        return;
    }

    if (hp_pct <= 0.90f) return;  /* eat/brew to 90%+ before boosting */

    if (attack_style == OPP_STYLE_MELEE || attack_style == OPP_STYLE_SPEC) {
        /* Boost when at or below base (covers brew-drained stats too) */
        if (self->current_strength <= self->base_strength && self->combat_potion_doses > 0) {
            actions[HEAD_POTION] = POTION_COMBAT;
            opp->potion_cooldown = 3;
        }
    } else if (attack_style == OPP_STYLE_RANGED) {
        if (self->current_ranged <= self->base_ranged && self->ranged_potion_doses > 0) {
            actions[HEAD_POTION] = POTION_RANGED;
            opp->potion_cooldown = 3;
        }
    }
}

/* Check if eating was queued in actions (food/karambwan cancel attacks) */
static inline int opp_check_eating_queued(int* actions) {
    return actions[HEAD_FOOD] != FOOD_NONE || actions[HEAD_KARAMBWAN] != KARAM_NONE;
}

/* Improved-style consumable logic. Returns 1 if potion was used (for restore/boost tracking) */
static int opp_apply_consumables(OsrsEnv* env, OpponentState* opp, int* actions,
                                  Player* self) {
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;
    OppConsumables cons = opp_get_consumables(opp, self);
    int potion_used = 0;

    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        /* Triple eat: shark + brew + karambwan */
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3;
        opp->potion_cooldown = 3;
        opp->karambwan_cooldown = 2;
        potion_used = 1;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        /* Double eat: shark + brew */
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3;
        opp->potion_cooldown = 3;
        potion_used = 1;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        /* Double eat: shark + karambwan (no brew available) */
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3;
        opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
        potion_used = 1;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        /* Karambwan as fallback food (no sharks left) */
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring (one restore undoes ~3 brews) */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
        potion_used = 1;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    (void)env;
    return potion_used;
}

/* map an OverheadPrayer to the toggle action that flips it on/off. returns
   ENCOUNTER_OVERHEAD_NO_CHANGE for PRAYER_NONE (nothing to toggle). */
static inline int opp_toggle_for_prayer(OverheadPrayer p) {
    switch (p) {
        case PRAYER_PROTECT_MAGIC:  return ENCOUNTER_OVERHEAD_TOGGLE_MAGIC;
        case PRAYER_PROTECT_RANGED: return ENCOUNTER_OVERHEAD_TOGGLE_RANGED;
        case PRAYER_PROTECT_MELEE:  return ENCOUNTER_OVERHEAD_TOGGLE_MELEE;
        case PRAYER_SMITE:          return ENCOUNTER_OVERHEAD_TOGGLE_SMITE;
        case PRAYER_REDEMPTION:     return ENCOUNTER_OVERHEAD_TOGGLE_REDEMPTION;
        default:                    return ENCOUNTER_OVERHEAD_NO_CHANGE;
    }
}

/* Convert opponent prayer intent into the toggle action required by the
   encounter overhead encoding. */
static inline void opp_emit_prayer(int* actions, Player* self, int target_overhead_action) {
    OverheadPrayer target_prayer;
    switch (target_overhead_action) {
        case OVERHEAD_NONE:       target_prayer = PRAYER_NONE;           break;
        case OVERHEAD_MAGE:       target_prayer = PRAYER_PROTECT_MAGIC;  break;
        case OVERHEAD_RANGED:     target_prayer = PRAYER_PROTECT_RANGED; break;
        case OVERHEAD_MELEE:      target_prayer = PRAYER_PROTECT_MELEE;  break;
        case OVERHEAD_SMITE:      target_prayer = PRAYER_SMITE;          break;
        case OVERHEAD_REDEMPTION: target_prayer = PRAYER_REDEMPTION;     break;
        default: return;  /* invalid: no-op */
    }
    if (self->prayer == target_prayer) return;
    /* deactivation path: toggle current-on prayer off. activation/replace path:
       emit target toggle (toggle semantics replace whatever's currently on). */
    int toggle = (target_prayer == PRAYER_NONE)
        ? opp_toggle_for_prayer(self->prayer)
        : opp_toggle_for_prayer(target_prayer);
    if (toggle != ENCOUNTER_OVERHEAD_NO_CHANGE)
        actions[HEAD_OVERHEAD] = toggle;
}

/* Process pending prayer delay: decrement, apply if ready. Returns 1 if applied. */
static inline int opp_process_pending_prayer(OpponentState* opp, int* actions, Player* self) {
    if (opp->pending_prayer_value == 0) return 0;
    if (opp->pending_prayer_delay > 0) {
        opp->pending_prayer_delay--;
        if (opp->pending_prayer_delay > 0) return 0;
    }
    OverheadPrayer target_prayer = PRAYER_NONE;
    int toggle = ENCOUNTER_OVERHEAD_NO_CHANGE;
    switch (opp->pending_prayer_value) {
        case OVERHEAD_MAGE:       target_prayer = PRAYER_PROTECT_MAGIC;  toggle = ENCOUNTER_OVERHEAD_TOGGLE_MAGIC; break;
        case OVERHEAD_RANGED:     target_prayer = PRAYER_PROTECT_RANGED; toggle = ENCOUNTER_OVERHEAD_TOGGLE_RANGED; break;
        case OVERHEAD_MELEE:      target_prayer = PRAYER_PROTECT_MELEE;  toggle = ENCOUNTER_OVERHEAD_TOGGLE_MELEE; break;
        case OVERHEAD_SMITE:      target_prayer = PRAYER_SMITE;          toggle = ENCOUNTER_OVERHEAD_TOGGLE_SMITE; break;
        case OVERHEAD_REDEMPTION: target_prayer = PRAYER_REDEMPTION;     toggle = ENCOUNTER_OVERHEAD_TOGGLE_REDEMPTION; break;
        default: break;
    }
    /* only emit toggle if we need to change — if already on target, no-op. */
    if (self->prayer != target_prayer) actions[HEAD_OVERHEAD] = toggle;
    opp->pending_prayer_value = 0;
    return 1;
}

/* Handle prayer switch with delay for unpredictable policies.
 * Detects target gear changes, samples delay, stores in pending state.
 * include_spec: if 1, also detect spec weapon (onetick/unpredictable_onetick). */
static void opp_handle_delayed_prayer(OsrsEnv* env, OpponentState* opp, int* actions,
                                       Player* self, Player* target,
                                       const float* cum_weights, int cum_len,
                                       float wrong_prayer_prob, int include_spec) {
    /* Detect target gear style change */
    int target_style = opp_get_target_gear_style(target);
    if (target_style != opp->last_target_gear_style) {
        opp->last_target_gear_style = target_style;

        /* Determine needed prayer */
        int needed_prayer = include_spec
            ? opp_get_defensive_prayer_with_spec(target)
            : opp_get_defensive_prayer(target);

        /* Check if we need to switch */
        int needs_switch = !opp_has_prayer_active(self, needed_prayer);

        if (needs_switch) {
            /* Small chance to pick wrong prayer */
            if (rand_float(env) < wrong_prayer_prob) {
                int wrong_options[2];
                int wcount = 0;
                int all_prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
                for (int i = 0; i < 3; i++) {
                    if (all_prayers[i] != needed_prayer)
                        wrong_options[wcount++] = all_prayers[i];
                }
                needed_prayer = wrong_options[rand_int(env, wcount)];
            }

            int delay = opp_sample_delay(env, cum_weights, cum_len);
            opp->pending_prayer_value = needed_prayer;
            opp->pending_prayer_delay = delay;
        }
    }

    /* Process pending prayer (may apply this tick if delay=0) */
    opp_process_pending_prayer(opp, actions, self);
}

/* =========================================================================
 * Policy implementations
 * ========================================================================= */

/* --- TrueRandom: random value per action head --- */
static void opp_true_random(OsrsEnv* env, int* actions) {
    for (int i = 0; i < NUM_ACTION_HEADS; i++) {
        actions[i] = rand_int(env, ACTION_HEAD_DIMS[i]);
    }
}

/* --- Panicking: fixed prayer, fixed style, 30% attack chance, panic eat --- */
static void opp_panicking(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* Set prayer if not already active */
    if (!opp_has_prayer_active(self, opp->chosen_prayer)) {
        opp_emit_prayer(actions, self, opp->chosen_prayer);
    }

    /* Panic eat at 25% HP */
    int eating = 0;
    if (hp_pct < 0.25f) {
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
            eating = 1;
        }
        if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
        }
    }

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 30% chance to attack */
    if (opp_attack_ready(self) && !eating && rand_float(env) < 0.30f) {
        opp_apply_gear_switch(actions, opp->chosen_style);

        if (opp->chosen_style == OPP_STYLE_MAGE) {
            int spell = (rand_int(env, 2) == 0) ? ATTACK_ICE : ATTACK_BLOOD;
            actions[HEAD_COMBAT] = spell;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    }
}

/* --- WeakRandom: random style, unreliable eating (50% skip) --- */
static void opp_weak_random(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* Random prayer each tick (includes NONE) */
    int prayers[] = {OVERHEAD_NONE, OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
    opp_emit_prayer(actions, self, prayers[rand_int(env, 4)]);

    /* Unreliable eating at 30% with 50% skip chance */
    int eating = 0;
    if (hp_pct < 0.30f && rand_float(env) > 0.50f) {
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
            eating = 1;
        } else if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
            eating = 1;
        }
    }

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* Random attack when ready */
    if (opp_attack_ready(self) && !eating) {
        int style = rand_int(env, 3);  /* 0=mage, 1=ranged, 2=melee */
        opp_apply_gear_switch(actions, style);
        if (style == OPP_STYLE_MAGE) {
            int spell = (rand_int(env, 2) == 0) ? ATTACK_ICE : ATTACK_BLOOD;
            actions[HEAD_COMBAT] = spell;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    }
}

/* --- SemiRandom: reliable eating at 30%, random everything else --- */
static void opp_semi_random(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* Random prayer each tick (no NONE) */
    int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
    opp_emit_prayer(actions, self, prayers[rand_int(env, 3)]);

    /* Reliable eating at 30% */
    int eating = 0;
    if (hp_pct < 0.30f) {
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
            eating = 1;
        } else if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
            eating = 1;
        }
    }

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* Random attack when ready */
    if (opp_attack_ready(self) && !eating) {
        int style = rand_int(env, 3);
        opp_apply_gear_switch(actions, style);
        if (style == OPP_STYLE_MAGE) {
            int spell = (rand_int(env, 2) == 0) ? ATTACK_ICE : ATTACK_BLOOD;
            actions[HEAD_COMBAT] = spell;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    }
}

/* --- StickyPrayer: sticky prayer (~12 tick avg), simple eating --- */
static void opp_sticky_prayer(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* Sticky prayer: 8% chance to switch per tick (~12 tick avg) */
    if (!opp->current_prayer_set || rand_float(env) < 0.08f) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        opp->current_prayer = prayers[rand_int(env, 3)];
        opp->current_prayer_set = 1;
    }
    if (!opp_has_prayer_active(self, opp->current_prayer)) {
        opp_emit_prayer(actions, self, opp->current_prayer);
    }

    /* Simple eating at 30% */
    int eating = 0;
    if (hp_pct < 0.30f) {
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
            eating = 1;
        } else if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
            eating = 1;
        }
    }

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* Random attack when ready */
    if (opp_attack_ready(self) && !eating) {
        int style = rand_int(env, 3);
        opp_apply_gear_switch(actions, style);
        if (style == OPP_STYLE_MAGE) {
            int spell = (rand_int(env, 2) == 0) ? ATTACK_ICE : ATTACK_BLOOD;
            actions[HEAD_COMBAT] = spell;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    }
}

/* --- Beginner: sticky prayer, multi-threshold eating, random spec --- */
static void opp_random_eater(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Sticky random prayer */
    if (!opp->current_prayer_set || rand_float(env) < 0.08f) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        opp->current_prayer = prayers[rand_int(env, 3)];
        opp->current_prayer_set = 1;
    }
    if (!opp_has_prayer_active(self, opp->current_prayer)) {
        opp_emit_prayer(actions, self, opp->current_prayer);
    }

    /* 2. Multi-threshold eating */
    int potion_used = 0;
    if (hp_pct < 0.35f) {
        /* Emergency: eat everything */
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
        }
        if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
            potion_used = 1;
        }
        if (cons.can_karambwan) {
            actions[HEAD_KARAMBWAN] = KARAM_EAT;
            opp->karambwan_cooldown = 2;
        }
    } else if (hp_pct < 0.55f) {
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
        } else if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
            potion_used = 1;
        }
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
        potion_used = 1;
    }

    /* 3. Restore if low prayer */
    if (!potion_used && prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 4. Attack when ready with random style */
    if (opp_attack_ready(self) && !eating) {
        int style = rand_int(env, 3);

        /* 30% spec chance */
        if (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) && rand_float(env) < 0.30f) {
            opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else {
            opp_apply_gear_switch(actions, style);
            if (style == OPP_STYLE_MAGE) {
                int spell = (rand_int(env, 2) == 0) ? ATTACK_ICE : ATTACK_BLOOD;
                actions[HEAD_COMBAT] = spell;
            } else {
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        }
    }

    (void)target;
}

/* --- BetterRandom: multi-threshold eating, random prayers, random spec --- */
static void opp_prayer_rookie(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    opp_emit_prayer(actions, self, def_prayer);

    /* 2. Multi-threshold eating */
    if (hp_pct < 0.35f) {
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
        }
        if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
        }
        if (cons.can_karambwan) {
            actions[HEAD_KARAMBWAN] = KARAM_EAT;
            opp->karambwan_cooldown = 2;
        }
    } else if (hp_pct < 0.55f) {
        if (cons.can_food) {
            actions[HEAD_FOOD] = FOOD_EAT;
            opp->food_cooldown = 3;
        } else if (cons.can_brew) {
            actions[HEAD_POTION] = POTION_BREW;
            opp->potion_cooldown = 3;
        }
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack with random style, random spec chance */
    if (opp_attack_ready(self) && !eating) {
        int style = rand_int(env, 3);

        if (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) && rand_float(env) < 0.30f) {
            opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else {
            opp_apply_gear_switch(actions, style);
            if (style == OPP_STYLE_MAGE) {
                int spell = (rand_int(env, 2) == 0) ? ATTACK_ICE : ATTACK_BLOOD;
                actions[HEAD_COMBAT] = spell;
            } else {
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        }
    }
}

/* --- Improved: full NH (correct prayer, off-prayer attacks, combo eating,
       spec timing, offensive prayer, movement) --- */
static void opp_improved(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer based on target's weapon */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Consumables: triple/double/single eat */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3;
        opp->potion_cooldown = 3;
        opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3;
        opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3;
        opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating was queued (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay: skip offensive actions this tick */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack decision */
    if (opp_attack_ready(self) && !eating) {
        /* Pick off-prayer style with bias */
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
        } else {
            attack_style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Check spec */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range);

        /* Anti-kite: cancel melee spec if target fleeing, force mage to freeze */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;  /* 0=ice, 1=blood, 2=atk, 3=spec */
        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;  /* blood if low HP, else ice */
        } else {
            actual_style = attack_style;
            actual_attack = 2;  /* ATK */
        }

        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack != 3 && rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        /* Attack action */
        if (actual_attack == 3) {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement when not attacking */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }
}

/* =========================================================================
 * Novice NH: learning player — 60% correct prayer, random attacks, good eating
 * Bridges easy opponents to intermediate. No off-prayer logic, no offensive
 * prayers, no movement. Just consistent attacking and sometimes-correct prayer.
 * ========================================================================= */

static void opp_novice_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as improved) */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack: off-prayer based on off_prayer_rate. Random spec. Offensive prayer. */
    if (opp_attack_ready(self) && !eating) {
        int style;
        if (rand_float(env) < opp->off_prayer_rate) {
            int off_mask = opp_get_off_prayer_mask(self, target);
            style = opp_pick_from_mask(env, off_mask);
        } else {
            style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, style, 0);

        if (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) && rand_float(env) < 0.15f) {
            int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
            if (opp->target_fleeing_ticks >= 2 && dist > 1) {
                /* Anti-kite: cancel spec, use mage */
                opp_apply_gear_switch(actions, OPP_STYLE_MAGE);
                actions[HEAD_COMBAT] = ATTACK_ICE;
            } else {
                opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        } else {
            /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
            if (rand_float(env) < opp->offensive_prayer_miss) {
                actions[HEAD_LOADOUT] = LOADOUT_KEEP;
            } else {
                opp_apply_gear_switch(actions, style);
            }

            /* Offensive prayer */
            if (rand_float(env) < opp->offensive_prayer_rate) {

            }

            if (style == OPP_STYLE_MAGE) {
                int spell = (rand_int(env, 2) == 0) ? ATTACK_ICE : ATTACK_BLOOD;
                actions[HEAD_COMBAT] = spell;
            } else {
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        }
    }
}

/* =========================================================================
 * Apprentice NH: 60% correct prayer, 20% off-prayer attacks, 20% offensive
 * prayer, random 30% spec, drain restore. Bridges novice_nh to competent_nh.
 * ========================================================================= */

static void opp_apprentice_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as improved) + drain restore */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack */
    if (opp_attack_ready(self) && !eating) {
        int style;
        if (rand_float(env) < opp->off_prayer_rate) {
            int off_mask = opp_get_off_prayer_mask(self, target);
            style = opp_pick_from_mask(env, off_mask);
        } else {
            style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, style, 0);

        if (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) && rand_float(env) < 0.30f) {
            int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
            if (opp->target_fleeing_ticks >= 2 && dist > 1) {
                opp_apply_gear_switch(actions, OPP_STYLE_MAGE);
                actions[HEAD_COMBAT] = ATTACK_ICE;
            } else {
                opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        } else {
            /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
            if (rand_float(env) < opp->offensive_prayer_miss) {
                actions[HEAD_LOADOUT] = LOADOUT_KEEP;
            } else {
                opp_apply_gear_switch(actions, style);
            }

            /* Offensive prayer */
            if (rand_float(env) < opp->offensive_prayer_rate) {

            }

            if (style == OPP_STYLE_MAGE) {
                int spell = (hp_pct < 0.30f) ? ATTACK_BLOOD : ATTACK_ICE;
                actions[HEAD_COMBAT] = spell;
            } else {
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        }
    }
}

/* =========================================================================
 * Competent NH: 75% correct prayer, 25% off-prayer attacks, 25% offensive
 * prayers, 50% conditional spec. Bridges apprentice_nh to intermediate_nh.
 * ========================================================================= */

static void opp_competent_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as improved) + drain restore */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack */
    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            int off_mask = opp_get_off_prayer_mask(self, target);
            attack_style = opp_pick_from_mask(env, off_mask);
        } else {
            attack_style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Spec check: same condition as intermediate_nh but 50% trigger rate */
        float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target_hp_pct < 0.60f &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range &&
                          rand_float(env) < 0.50f);

        /* Anti-kite: cancel melee spec if target fleeing, force mage to freeze */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;
        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;  /* blood if low, else ice */
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack != 3 && rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        /* Offensive prayer */
        if (rand_float(env) < opp->offensive_prayer_rate) {

        }

        /* Attack action */
        if (actual_attack == 3) {

            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    }
}

/* =========================================================================
 * Intermediate NH: getting the hang of it — 85% correct prayer, 70% off-prayer
 * attacks, 50% offensive prayers. No movement, no fakes. Bridges novice_nh
 * to improved.
 * ========================================================================= */

static void opp_intermediate_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as improved) */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack */
    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            int off_mask = opp_get_off_prayer_mask(self, target);
            attack_style = opp_pick_from_mask(env, off_mask);
        } else {
            attack_style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Spec check: target HP < 60%, not on melee prayer, in range */
        float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target_hp_pct < 0.60f &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range);

        /* Anti-kite: cancel melee spec if target fleeing, force mage to freeze */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;
        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;  /* blood if low, else ice */
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack != 3 && rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        /* Offensive prayer */
        if (rand_float(env) < opp->offensive_prayer_rate) {

        }

        /* Attack action */
        if (actual_attack == 3) {

            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    }
}

/* =========================================================================
 * Advanced NH: near-improved — 100% correct prayer, 90% off-prayer attacks,
 * 75% offensive prayers, same spec as improved, farcast 3 but no step under.
 * Bridges intermediate_nh to improved.
 * ========================================================================= */

static void opp_advanced_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as improved) + drain restore */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack */
    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
        } else {
            attack_style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Spec: same as improved (no HP threshold, just not praying melee + in range) */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range);

        /* Anti-kite: cancel melee spec if target fleeing, force mage to freeze */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;
        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;  /* blood if low, else ice */
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack != 3 && rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        /* Offensive prayer */
        if (rand_float(env) < opp->offensive_prayer_rate) {

        }

        /* Attack action */
        if (actual_attack == 3) {

            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement: farcast 3 only (no step under) */
        int mv_dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (opp->target_fleeing_ticks >= 2 && mv_dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }
}

/* =========================================================================
 * Proficient NH: 92% off-prayer, 80% offensive prayer, 25% step under.
 * Introduces step under at low rate between advanced_nh (no step under)
 * and expert_nh (50% step under).
 * ========================================================================= */

static void opp_proficient_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating + drain restore */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack */
    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
        } else {
            attack_style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Spec: same as improved */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range);

        /* Anti-kite: cancel melee spec if target fleeing, force mage to freeze */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;
        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack != 3 && rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        /* Offensive prayer */
        if (rand_float(env) < opp->offensive_prayer_rate) {

        }

        /* Attack action */
        if (actual_attack == 3) {

            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement: farcast 3 + 25% step under */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0 &&
            rand_float(env) < 0.25f) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }
}

/* =========================================================================
 * Expert NH: 95% off-prayer, 85% offensive prayer, 50% step under.
 * Introduces step under mechanic at reduced rate while keeping attack
 * parameters between advanced_nh and improved.
 * ========================================================================= */

static void opp_expert_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as improved) + drain restore */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    /* Check if eating (food/karambwan cancel attacks) */
    int eating = opp_check_eating_queued(actions);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack */
    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
        } else {
            attack_style = rand_int(env, 3);
        }

        /* Boost potions before attack */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Spec: same as improved */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range);

        /* Anti-kite: cancel melee spec if target fleeing, force mage to freeze */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;
        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack != 3 && rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        /* Offensive prayer */
        if (rand_float(env) < opp->offensive_prayer_rate) {

        }

        /* Attack action */
        if (actual_attack == 3) {

            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement: farcast 3 + 50% step under */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0 &&
            rand_float(env) < 0.50f) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }
}

/* =========================================================================
 * Phase 2 Policy: Onetick
 * Fake switches, tank gear, smart spec, boost pots, 1-tick attacks.
 * ========================================================================= */

static void opp_onetick(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);

    /* 0. Tank gear switch when not about to attack */
    if (!opp_attack_ready(self)) {
        opp_apply_tank_gear(actions);
    }

    /* 1. Defensive prayer with spec detection */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer_with_spec(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Consumables (same thresholds as improved) */
    int potion_used = opp_apply_consumables(env, opp, actions, self);

    /* Check if eating was queued */
    int eating_queued = opp_check_eating_queued(actions);

    /* 3. Get off-prayer mask */
    int off_mask = opp_get_off_prayer_mask(self, target);

    /* 4. Fake switch logic */
    if (opp->fake_switch_pending && opp_attack_ready(self)) {
        /* Clear fake state when attack ready */
        opp->fake_switch_pending = 0;
        opp->fake_switch_style = -1;
    } else if (!opp_attack_ready(self) && !opp->fake_switch_pending && rand_float(env) < 0.30f) {
        /* Initiate fake switch */
        int current_style = (int)self->current_gear;
        /* Don't fake melee if frozen at distance */
        int can_fake_melee = self->frozen_ticks <= 10 ||
                             chebyshev_distance(self->x, self->y, target->x, target->y) <= 1;

        /* Build fake options: off-prayer, not current style, melee only if credible */
        int fake_options[3];
        int fake_count = 0;
        for (int s = 0; s < 3; s++) {
            if (!(off_mask & (1 << s))) continue;
            if (s == current_style) continue;
            if (s == OPP_STYLE_MELEE && !can_fake_melee) continue;
            fake_options[fake_count++] = s;
        }

        if (fake_count > 0) {
            opp->fake_switch_pending = 1;
            opp->fake_switch_style = fake_options[rand_int(env, fake_count)];
            opp->opponent_prayer_at_fake = opp_get_opponent_prayer_style(target);

            /* Fake switch: set loadout but no attack */
            opp_apply_fake_switch(actions, opp->fake_switch_style);

            /* Step under if target frozen */
            int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
            if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
                actions[HEAD_COMBAT] = MOVE_UNDER;
            }

            /* Early return -- fake switch done this tick */
            return;
        }
    }

    /* 5. Determine attack style */
    /* If we just faked, anticipate opponent's prayer switch */
    int preferred_style = -1;
    if (opp->opponent_prayer_at_fake >= 0) {
        preferred_style = opp->opponent_prayer_at_fake;
        opp->opponent_prayer_at_fake = -1;
    }

    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    int can_melee_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;

    /* Spec checks: melee, ranged, magic */
    uint8_t ranged_spec = find_best_ranged_spec(self);
    uint8_t magic_spec = find_best_magic_spec(self);
    int has_ranged_or_magic_spec = (ranged_spec != ITEM_NONE || magic_spec != ITEM_NONE);

    /* If ranged/magic specs available, gate melee spec behind HP threshold too
     * so the boss saves energy for ranged/magic finishing blows */
    int should_melee_spec = opp_attack_ready(self) &&
                      self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_MELEE &&
                      can_melee_spec_range &&
                      (!has_ranged_or_magic_spec || target_hp_pct < 0.55f);

    int should_ranged_spec = opp_attack_ready(self) && ranged_spec != ITEM_NONE &&
                      self->special_energy >= get_ranged_spec_cost(self->ranged_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_RANGED &&
                      target_hp_pct < 0.55f;

    int should_magic_spec = opp_attack_ready(self) && magic_spec != ITEM_NONE &&
                      self->special_energy >= get_magic_spec_cost(self->magic_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_MAGIC &&
                      target_hp_pct < 0.55f;

    /* Anti-kite: cancel melee spec if target fleeing */
    if (should_melee_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
        should_melee_spec = 0;
    }

    int actual_style;
    int actual_attack;  /* 0=ice, 1=blood, 2=atk, 3=spec */
    int spec_loadout = LOADOUT_SPEC_MELEE;  /* default, overridden below */

    /* Spec priority: ranged at distance > magic off-prayer > melee in range */
    if (should_ranged_spec && (dist >= 3 || target->frozen_ticks > 0)) {
        actual_style = OPP_STYLE_RANGED;
        actual_attack = 3;
        spec_loadout = LOADOUT_SPEC_RANGE;
    } else if (should_magic_spec) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = 3;
        spec_loadout = LOADOUT_SPEC_MAGIC;
    } else if (should_melee_spec) {
        actual_style = OPP_STYLE_SPEC;
        actual_attack = 3;
    } else if (target->frozen_ticks == 0 && (off_mask & (1 << OPP_STYLE_MAGE))) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = (hp_pct < 0.98f)
            ? ((target->freeze_immunity_ticks <= 1 && target->frozen_ticks == 0) ? 0 : 1)
            : 0;  /* ice at full HP */
        /* Simplified: use opp_get_mage_attack for ice/blood decision */
        actual_attack = opp_get_mage_attack(self, target) == ATTACK_ICE ? 0 : 1;
    } else {
        /* Target frozen or mage not off-prayer — choose based on fake anticipation */
        int can_use_preferred = preferred_style >= 0 &&
            (preferred_style != OPP_STYLE_MELEE || self->frozen_ticks <= 10 || dist <= 1);

        if (can_use_preferred) {
            actual_style = preferred_style;
            if (preferred_style == OPP_STYLE_MAGE) {
                actual_attack = (hp_pct < 0.98f) ? 1 : 0;  /* blood if not full HP */
            } else {
                actual_attack = 2;  /* ATK */
            }
        } else if (off_mask & (1 << OPP_STYLE_MAGE)) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.98f) ? 1 : 0;
        } else {
            /* Pick non-mage from off-prayer */
            int non_mage[2];
            int nm_count = 0;
            for (int s = 1; s < 3; s++) {
                if (off_mask & (1 << s)) non_mage[nm_count++] = s;
            }
            if (nm_count == 0) {
                actual_style = OPP_STYLE_RANGED;
            } else {
                actual_style = non_mage[rand_int(env, nm_count)];
            }
            actual_attack = 2;  /* ATK */
        }
    }

    /* 6. Boost potions (before attack) */
    opp_apply_boost_potion(env, opp, actions, self, actual_style, potion_used);

    /* Tick-level action delay: skip attack but keep prayer/eating/fakes */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 7. Gear + offensive prayer + attack */
    if (opp_attack_ready(self) && !eating_queued) {
        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack == 3) {
            actions[HEAD_LOADOUT] = spec_loadout;
        } else if (rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        if (actual_attack == 3) {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement when not attacking */
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }

    (void)prayer_pct;
}

/* =========================================================================
 * Phase 2 Policy: RealisticImproved
 * Improved with prayer delays, wrong prayer chance, attack delays.
 * ========================================================================= */

static void opp_unpredictable_improved(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);

    /* 1. Handle prayer switch with delay */
    opp_handle_delayed_prayer(env, opp, actions, self, target,
                               UNPREDICTABLE_IMP_PRAYER_CUM, UNPREDICTABLE_IMP_PRAYER_CUM_LEN,
                               UNPREDICTABLE_IMP_WRONG_PRAYER, 0 /* no spec detection */);

    /* 2. Consumables (no delay — survival instinct) */
    int potion_used = opp_apply_consumables(env, opp, actions, self);

    int eating_queued = opp_check_eating_queued(actions);

    /* Tick-level action delay (additional layer on top of built-in delays) */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack style decision (with small mistake chance + style bias) */
    int attack_style;

    if (rand_float(env) < UNPREDICTABLE_IMP_SUBOPTIMAL_ATTACK) {
        /* Pick from all 3 styles (might be on-prayer) */
        attack_style = rand_int(env, 3);
    } else {
        attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
    }

    /* Boost potions before attack */
    opp_apply_boost_potion(env, opp, actions, self, attack_style, potion_used);

    /* 4. Determine actual attack */
    if (opp_attack_ready(self) && !eating_queued) {
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range;

        /* Anti-kite: cancel melee spec if target fleeing, force mage to freeze */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;

        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;  /* blood if very low */
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        /* 5. Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (actual_attack != 3 && rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        /* 6. Offensive prayer */


        /* 7. Attack with delay — sample delay, skip if > 0 */
        int action_delay = opp_sample_delay(env, UNPREDICTABLE_IMP_ACTION_CUM, UNPREDICTABLE_IMP_ACTION_CUM_LEN);
        if (action_delay == 0) {
            if (actual_attack == 3) {

                actions[HEAD_COMBAT] = ATTACK_ATK;
            } else if (actual_attack == 0) {
                actions[HEAD_COMBAT] = ATTACK_ICE;
            } else if (actual_attack == 1) {
                actions[HEAD_COMBAT] = ATTACK_BLOOD;
            } else {
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        }
        /* else: missed attack window due to delay */
    } else if (!opp_attack_ready(self)) {
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }

    (void)potion_used;
}

/* =========================================================================
 * Phase 2 Policy: RealisticOnetick
 * Onetick + prayer delays + fake execution failures + wrong prediction.
 * ========================================================================= */

static void opp_unpredictable_onetick(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);

    /* 0. Tank gear when not about to attack */
    if (!opp_attack_ready(self)) {
        opp_apply_tank_gear(actions);
    }

    /* 1. Handle prayer switch with delay (with spec detection) */
    opp_handle_delayed_prayer(env, opp, actions, self, target,
                               UNPREDICTABLE_OT_PRAYER_CUM, UNPREDICTABLE_OT_PRAYER_CUM_LEN,
                               0.0f /* no wrong prayer for onetick */, 1 /* include spec */);

    /* 2. Consumables */
    int potion_used = opp_apply_consumables(env, opp, actions, self);

    int eating_queued = opp_check_eating_queued(actions);

    /* 3. Get off-prayer mask */
    int off_mask = opp_get_off_prayer_mask(self, target);

    /* 4. Fake switch logic (same as onetick + failure chance) */
    if (opp->fake_switch_pending && opp_attack_ready(self)) {
        opp->fake_switch_pending = 0;
        opp->fake_switch_style = -1;
        opp->fake_switch_failed = 0;
    } else if (!opp_attack_ready(self) && !opp->fake_switch_pending && rand_float(env) < 0.30f) {
        int current_style = (int)self->current_gear;
        int can_fake_melee = self->frozen_ticks <= 10 ||
                             chebyshev_distance(self->x, self->y, target->x, target->y) <= 1;

        int fake_options[3];
        int fake_count = 0;
        for (int s = 0; s < 3; s++) {
            if (!(off_mask & (1 << s))) continue;
            if (s == current_style) continue;
            if (s == OPP_STYLE_MELEE && !can_fake_melee) continue;
            fake_options[fake_count++] = s;
        }

        if (fake_count > 0) {
            opp->fake_switch_pending = 1;
            opp->fake_switch_style = fake_options[rand_int(env, fake_count)];
            opp->opponent_prayer_at_fake = opp_get_opponent_prayer_style(target);

            /* Roll fake execution failure */
            opp->fake_switch_failed = (rand_float(env) < UNPREDICTABLE_OT_FAKE_FAIL) ? 1 : 0;

            /* Fake switch: set loadout but no attack */
            opp_apply_fake_switch(actions, opp->fake_switch_style);

            int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
            if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
                actions[HEAD_COMBAT] = MOVE_UNDER;
            }

            return;  /* Early return: fake switch done */
        }
    }

    /* 5. Determine attack style with fake anticipation + failure/prediction errors */
    int preferred_style = -1;

    if (opp->opponent_prayer_at_fake >= 0 && !opp->fake_switch_failed) {
        /* Fake succeeded — but small chance of wrong prediction */
        if (rand_float(env) < UNPREDICTABLE_OT_WRONG_PREDICT) {
            preferred_style = rand_int(env, 3);  /* random style */
        } else {
            preferred_style = opp->opponent_prayer_at_fake;
        }
        opp->opponent_prayer_at_fake = -1;
    } else if (opp->fake_switch_failed) {
        /* Fake failed — no preferred style */
        opp->opponent_prayer_at_fake = -1;
        opp->fake_switch_failed = 0;
    }

    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    int can_melee_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;

    /* Spec checks: melee, ranged, magic */
    uint8_t ranged_spec = find_best_ranged_spec(self);
    uint8_t magic_spec = find_best_magic_spec(self);
    int has_ranged_or_magic_spec = (ranged_spec != ITEM_NONE || magic_spec != ITEM_NONE);

    int should_melee_spec = opp_attack_ready(self) &&
                      self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_MELEE &&
                      can_melee_spec_range &&
                      (!has_ranged_or_magic_spec || target_hp_pct < 0.55f);

    int should_ranged_spec = opp_attack_ready(self) && ranged_spec != ITEM_NONE &&
                      self->special_energy >= get_ranged_spec_cost(self->ranged_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_RANGED &&
                      target_hp_pct < 0.55f;

    int should_magic_spec = opp_attack_ready(self) && magic_spec != ITEM_NONE &&
                      self->special_energy >= get_magic_spec_cost(self->magic_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_MAGIC &&
                      target_hp_pct < 0.55f;

    /* Anti-kite: cancel melee spec if target fleeing */
    if (should_melee_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
        should_melee_spec = 0;
    }

    int actual_style;
    int actual_attack;
    int spec_loadout = LOADOUT_SPEC_MELEE;

    /* Spec priority: ranged at distance > magic off-prayer > melee in range */
    if (should_ranged_spec && (dist >= 3 || target->frozen_ticks > 0)) {
        actual_style = OPP_STYLE_RANGED;
        actual_attack = 3;
        spec_loadout = LOADOUT_SPEC_RANGE;
    } else if (should_magic_spec) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = 3;
        spec_loadout = LOADOUT_SPEC_MAGIC;
    } else if (should_melee_spec) {
        actual_style = OPP_STYLE_SPEC;
        actual_attack = 3;
    } else if (target->frozen_ticks == 0 && (off_mask & (1 << OPP_STYLE_MAGE))) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = opp_get_mage_attack(self, target) == ATTACK_ICE ? 0 : 1;
    } else {
        int can_use_preferred = preferred_style >= 0 &&
            (preferred_style != OPP_STYLE_MELEE || self->frozen_ticks <= 10 || dist <= 1);

        if (can_use_preferred) {
            actual_style = preferred_style;
            actual_attack = (preferred_style == OPP_STYLE_MAGE)
                ? ((hp_pct < 0.98f) ? 1 : 0)
                : 2;
        } else if (off_mask & (1 << OPP_STYLE_MAGE)) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.98f) ? 1 : 0;
        } else {
            int non_mage[2];
            int nm_count = 0;
            for (int s = 1; s < 3; s++) {
                if (off_mask & (1 << s)) non_mage[nm_count++] = s;
            }
            actual_style = (nm_count > 0) ? non_mage[rand_int(env, nm_count)] : OPP_STYLE_RANGED;
            actual_attack = 2;
        }
    }

    /* 6. Boost potions */
    opp_apply_boost_potion(env, opp, actions, self, actual_style, potion_used);

    /* Tick-level action delay (additional layer) */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 7. Gear + attack with delay chance */
    if (opp_attack_ready(self) && !eating_queued) {
        int action_delay = opp_sample_delay(env, UNPREDICTABLE_OT_ACTION_CUM, UNPREDICTABLE_OT_ACTION_CUM_LEN);
        if (action_delay == 0) {
            /* Gear switch — spec uses spec_loadout directly */
            if (actual_attack == 3) {
                actions[HEAD_LOADOUT] = spec_loadout;
            } else if (rand_float(env) < opp->offensive_prayer_miss) {
                actions[HEAD_LOADOUT] = LOADOUT_KEEP;
            } else {
                opp_apply_gear_switch(actions, actual_style);
            }


            if (actual_attack == 3) {

                actions[HEAD_COMBAT] = ATTACK_ATK;
            } else if (actual_attack == 0) {
                actions[HEAD_COMBAT] = ATTACK_ICE;
            } else if (actual_attack == 1) {
                actions[HEAD_COMBAT] = ATTACK_BLOOD;
            } else {
                actions[HEAD_COMBAT] = ATTACK_ATK;
            }
        }
        /* else: missed attack window due to delay */
    } else if (!opp_attack_ready(self)) {
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }
}

/* =========================================================================
 * Helper: decode agent's pending action to extract attack style and prayer
 * Used by boss opponents (master_nh, savant_nh) for "reading" ability.
 * ========================================================================= */

static void opp_read_agent_action(OsrsEnv* env, OpponentState* opp) {
    opp->has_read_this_tick = 0;
    opp->read_agent_style = ATTACK_STYLE_NONE;
    opp->read_agent_prayer = PRAYER_NONE;
    opp->read_agent_moving = 0;

    if (opp->read_chance <= 0.0f || rand_float(env) >= opp->read_chance) {
        return;  /* Read failed or no read ability */
    }

    /* Read succeeded - read agent's CURRENT tick actions (player 0)
     * IMPORTANT: Read from env->actions, not pending_actions.
     * pending_actions contains PREVIOUS tick's actions.
     * env->actions is populated from ocean_acts before opponent generation. */
    int* agent_actions = &env->actions[0];

    /* Extract attack style: loadout determines weapon, so it takes priority.
     * Only fall back to attack head when loadout is KEEP/TANK (no switch). */
    int loadout = agent_actions[HEAD_LOADOUT];
    int attack = agent_actions[HEAD_COMBAT];

    if (loadout != LOADOUT_KEEP && loadout != LOADOUT_TANK) {
        /* Loadout switch — weapon determines what's physically possible */
        if (loadout == LOADOUT_MELEE || loadout == LOADOUT_SPEC_MELEE || loadout == LOADOUT_GMAUL) {
            opp->read_agent_style = ATTACK_STYLE_MELEE;
        } else if (loadout == LOADOUT_RANGE || loadout == LOADOUT_SPEC_RANGE) {
            opp->read_agent_style = ATTACK_STYLE_RANGED;
        } else if (loadout == LOADOUT_MAGE || loadout == LOADOUT_SPEC_MAGIC) {
            opp->read_agent_style = ATTACK_STYLE_MAGIC;
        }
        opp->has_read_this_tick = 1;
    } else if (attack == ATTACK_ICE || attack == ATTACK_BLOOD) {
        /* KEEP/TANK + spell cast — must already be holding a staff */
        opp->read_agent_style = ATTACK_STYLE_MAGIC;
        opp->has_read_this_tick = 1;
    } else if (attack == ATTACK_ATK) {
        /* KEEP/TANK + generic attack — use current equipped weapon */
        uint8_t weapon = env->players[0].equipped[GEAR_SLOT_WEAPON];
        int style = get_item_attack_style(weapon);
        if (style == 1) opp->read_agent_style = ATTACK_STYLE_MELEE;
        else if (style == 2) opp->read_agent_style = ATTACK_STYLE_RANGED;
        else if (style == 3) opp->read_agent_style = ATTACK_STYLE_MAGIC;
        opp->has_read_this_tick = 1;
    }

    /* Extract overhead prayer intent from agent's toggle action. since action is
       a toggle, the "intent" is the agent's target — if the toggle would activate
       or replace, we read it as intent; if it would deactivate, we read no intent.
       we approximate by looking up the target based on the toggle id; the agent's
       actual prayer state is observed separately via Player.prayer by opponent AI. */
    int overhead = agent_actions[HEAD_OVERHEAD];
    if (overhead == ENCOUNTER_OVERHEAD_TOGGLE_MELEE)       opp->read_agent_prayer = PRAYER_PROTECT_MELEE;
    else if (overhead == ENCOUNTER_OVERHEAD_TOGGLE_RANGED) opp->read_agent_prayer = PRAYER_PROTECT_RANGED;
    else if (overhead == ENCOUNTER_OVERHEAD_TOGGLE_MAGIC)  opp->read_agent_prayer = PRAYER_PROTECT_MAGIC;
    else if (overhead == ENCOUNTER_OVERHEAD_TOGGLE_SMITE)  opp->read_agent_prayer = PRAYER_SMITE;
    else if (overhead == ENCOUNTER_OVERHEAD_TOGGLE_REDEMPTION) opp->read_agent_prayer = PRAYER_REDEMPTION;

    /* Extract movement intent */
    opp->read_agent_moving = is_move_action(attack) ? 1 : 0;
}

/* Get defensive prayer against agent's read attack style */
static inline int opp_get_read_defensive_prayer(OpponentState* opp) {
    if (opp->read_agent_style == ATTACK_STYLE_MAGIC) return OVERHEAD_MAGE;
    if (opp->read_agent_style == ATTACK_STYLE_RANGED) return OVERHEAD_RANGED;
    if (opp->read_agent_style == ATTACK_STYLE_MELEE) return OVERHEAD_MELEE;
    return -1;  /* No read or unknown */
}

/* Check if a style would hit agent off-prayer (using read info) */
static inline int opp_style_off_read_prayer(OpponentState* opp, int style) {
    if (opp->read_agent_prayer == PRAYER_NONE) return 1;  /* No read, assume off */
    if (style == OPP_STYLE_MAGE && opp->read_agent_prayer != PRAYER_PROTECT_MAGIC) return 1;
    if (style == OPP_STYLE_RANGED && opp->read_agent_prayer != PRAYER_PROTECT_RANGED) return 1;
    if (style == OPP_STYLE_MELEE && opp->read_agent_prayer != PRAYER_PROTECT_MELEE) return 1;
    return 0;  /* Would hit on-prayer */
}

/* =========================================================================
 * Boss Policy: Master NH
 * Onetick-perfect mechanics + 10% chance to "read" agent's pending action.
 * When read succeeds: prays correctly against incoming attack, attacks off-prayer.
 * ========================================================================= */

static void opp_master_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);

    /* Attempt to read agent's pending action */
    opp_read_agent_action(env, opp);

    /* 0. Tank gear switch when not about to attack */
    if (!opp_attack_ready(self)) {
        opp_apply_tank_gear(actions);
    }

    /* 1. Defensive prayer - use read info if available, else detect from gear */
    int def_prayer = -1;
    if (opp->has_read_this_tick && opp->read_agent_style != ATTACK_STYLE_NONE) {
        def_prayer = opp_get_read_defensive_prayer(opp);
    }
    if (def_prayer < 0) {
        if (rand_float(env) < opp->prayer_accuracy) {
            def_prayer = opp_get_defensive_prayer_with_spec(target);
        } else {
            int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
            def_prayer = prayers[rand_int(env, 3)];
        }
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Consumables (same as onetick) */
    int potion_used = opp_apply_consumables(env, opp, actions, self);
    int eating_queued = opp_check_eating_queued(actions);

    /* 3. Get off-prayer mask (normal) and check read info for better targeting */
    int off_mask = opp_get_off_prayer_mask(self, target);

    /* 4. Fake switch logic (same as onetick) */
    if (opp->fake_switch_pending && opp_attack_ready(self)) {
        opp->fake_switch_pending = 0;
        opp->fake_switch_style = -1;
    } else if (!opp_attack_ready(self) && !opp->fake_switch_pending && rand_float(env) < 0.30f) {
        int current_style = (int)self->current_gear;
        int can_fake_melee = self->frozen_ticks <= 10 ||
                             chebyshev_distance(self->x, self->y, target->x, target->y) <= 1;

        int fake_options[3];
        int fake_count = 0;
        for (int s = 0; s < 3; s++) {
            if (!(off_mask & (1 << s))) continue;
            if (s == current_style) continue;
            if (s == OPP_STYLE_MELEE && !can_fake_melee) continue;
            fake_options[fake_count++] = s;
        }

        if (fake_count > 0) {
            opp->fake_switch_pending = 1;
            opp->fake_switch_style = fake_options[rand_int(env, fake_count)];
            opp->opponent_prayer_at_fake = opp_get_opponent_prayer_style(target);

            opp_apply_fake_switch(actions, opp->fake_switch_style);

            int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
            if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
                actions[HEAD_COMBAT] = MOVE_UNDER;
            }
            return;
        }
    }

    /* 5. Determine attack style - use read info if available */
    int preferred_style = -1;
    if (opp->opponent_prayer_at_fake >= 0) {
        preferred_style = opp->opponent_prayer_at_fake;
        opp->opponent_prayer_at_fake = -1;
    }

    /* If we read agent's prayer, pick a style they're NOT praying against */
    if (opp->has_read_this_tick && opp->read_agent_prayer != PRAYER_NONE) {
        /* Find best off-prayer style using read info */
        int read_off_styles[3];
        int read_off_count = 0;
        for (int s = 0; s < 3; s++) {
            if (!(off_mask & (1 << s))) continue;
            if (opp_style_off_read_prayer(opp, s)) {
                read_off_styles[read_off_count++] = s;
            }
        }
        if (read_off_count > 0) {
            preferred_style = read_off_styles[rand_int(env, read_off_count)];
        }
    }

    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    int can_melee_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;

    /* Spec checks: melee, ranged, magic */
    uint8_t ranged_spec = find_best_ranged_spec(self);
    uint8_t magic_spec = find_best_magic_spec(self);
    int has_ranged_or_magic_spec = (ranged_spec != ITEM_NONE || magic_spec != ITEM_NONE);

    int should_melee_spec = opp_attack_ready(self) &&
                      self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_MELEE &&
                      can_melee_spec_range &&
                      (!has_ranged_or_magic_spec || target_hp_pct < 0.55f);

    int should_ranged_spec = opp_attack_ready(self) && ranged_spec != ITEM_NONE &&
                      self->special_energy >= get_ranged_spec_cost(self->ranged_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_RANGED &&
                      target_hp_pct < 0.55f;

    int should_magic_spec = opp_attack_ready(self) && magic_spec != ITEM_NONE &&
                      self->special_energy >= get_magic_spec_cost(self->magic_spec_weapon) &&
                      target->prayer != PRAYER_PROTECT_MAGIC &&
                      target_hp_pct < 0.55f;

    /* With read, cancel specs the agent is praying against */
    if (opp->has_read_this_tick) {
        if (should_melee_spec && opp->read_agent_prayer == PRAYER_PROTECT_MELEE)
            should_melee_spec = 0;
        if (should_ranged_spec && opp->read_agent_prayer == PRAYER_PROTECT_RANGED)
            should_ranged_spec = 0;
        if (should_magic_spec && opp->read_agent_prayer == PRAYER_PROTECT_MAGIC)
            should_magic_spec = 0;
    }

    /* Anti-kite: cancel melee spec if target fleeing */
    if (should_melee_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
        should_melee_spec = 0;
    }

    /* Read-based anti-kite: if agent about to move away, cancel melee spec */
    if (should_melee_spec && opp->has_read_this_tick && opp->read_agent_moving && dist > 1) {
        should_melee_spec = 0;
    }

    int actual_style;
    int actual_attack;
    int spec_loadout = LOADOUT_SPEC_MELEE;

    /* Spec priority: ranged at distance > magic off-prayer > melee in range */
    if (should_ranged_spec && (dist >= 3 || target->frozen_ticks > 0)) {
        actual_style = OPP_STYLE_RANGED;
        actual_attack = 3;
        spec_loadout = LOADOUT_SPEC_RANGE;
    } else if (should_magic_spec) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = 3;
        spec_loadout = LOADOUT_SPEC_MAGIC;
    } else if (should_melee_spec) {
        actual_style = OPP_STYLE_SPEC;
        actual_attack = 3;
    } else if (preferred_style >= 0) {
        actual_style = preferred_style;
        actual_attack = (preferred_style == OPP_STYLE_MAGE)
            ? (opp_get_mage_attack(self, target) == ATTACK_ICE ? 0 : 1)
            : 2;
    } else if (target->frozen_ticks == 0 && (off_mask & (1 << OPP_STYLE_MAGE))) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = opp_get_mage_attack(self, target) == ATTACK_ICE ? 0 : 1;
    } else {
        actual_style = opp_pick_from_mask(env, off_mask);
        actual_attack = (actual_style == OPP_STYLE_MAGE)
            ? (opp_get_mage_attack(self, target) == ATTACK_ICE ? 0 : 1)
            : 2;
    }

    /* 6. Boost potions */
    opp_apply_boost_potion(env, opp, actions, self, actual_style, potion_used);

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 7. Gear + attack */
    if (opp_attack_ready(self) && !eating_queued) {
        /* Spec: use spec_loadout directly; normal: gear switch with prayer miss */
        if (actual_attack == 3) {
            actions[HEAD_LOADOUT] = spec_loadout;
        } else if (rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, actual_style);
        }

        if (actual_attack == 3) {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }

    (void)prayer_pct;
}

/* =========================================================================
 * Boss Policy: Savant NH
 * Onetick-perfect mechanics + 25% chance to "read" agent's pending action.
 * Same as master_nh but with higher read chance.
 * ========================================================================= */

static void opp_savant_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    /* Savant uses the same logic as master, just with higher read_chance (set in reset) */
    opp_master_nh(env, opp, actions);
}

/* =========================================================================
 * Boss Policy: Nightmare NH
 * Same as master/savant but with 50% read chance - extremely difficult.
 * ========================================================================= */

static void opp_nightmare_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    /* Nightmare uses the same logic as master, just with 50% read_chance (set in reset) */
    opp_master_nh(env, opp, actions);
}

/* =========================================================================
 * Vengeance Fighter: lunar spellbook, melee/range only, no freeze/blood.
 * Expert-level prayer/eating, veng on cooldown, melee spec only.
 * ========================================================================= */

static void opp_veng_fighter(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer (same as expert_nh) */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as expert_nh) */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    /* 3. Vengeance on cooldown */
    if (!self->veng_active && remaining_ticks(self->veng_cooldown) == 0) {
        actions[HEAD_VENG] = VENG_CAST;
    }

    /* Tick-level action delay */
    if (opp_should_skip_offensive(env, opp)) return;

    /* 4. Attack: melee/range only (no mage — lunar spellbook) */
    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            /* Off-prayer: pick melee or range based on what target ISN'T praying */
            int off_mask = opp_get_off_prayer_mask(self, target);
            /* Remove mage from mask — lunar can't cast offensive spells */
            off_mask &= ~(1 << OPP_STYLE_MAGE);
            if (off_mask == 0) off_mask = (1 << OPP_STYLE_MELEE) | (1 << OPP_STYLE_RANGED);
            attack_style = opp_pick_from_mask(env, off_mask);
        } else {
            /* Random: melee or range only (OPP_STYLE_RANGED=1, OPP_STYLE_MELEE=2) */
            attack_style = rand_int(env, 2) + 1;
        }

        /* Boost potions */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Spec: melee spec only */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range);

        /* Anti-kite: cancel spec if target fleeing */
        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
        }

        if (should_spec) {
            opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else {
            /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
            if (rand_float(env) < opp->offensive_prayer_miss) {
                actions[HEAD_LOADOUT] = LOADOUT_KEEP;
            } else {
                opp_apply_gear_switch(actions, attack_style);
            }
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement: step under frozen target */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0 &&
            rand_float(env) < 0.40f) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }
}

/* =========================================================================
 * Blood Healer: sustain fighter using blood barrage as primary healing.
 * Works at all tiers — ahrim staff can cast blood spells regardless of gear.
 * Farcast-5, reduced food reliance, blood barrage heavy when damaged.
 * ========================================================================= */

static void opp_blood_healer(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Reduced eating — relies on blood barrage for sustain above ~35%.
     * Emergency triple-eat below 35%, otherwise only brew/food below 25%. */
    if (hp_pct < 0.25f && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < 0.35f && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < 0.35f && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < 0.35f && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && hp_pct < 0.50f && cons.can_brew) {
        /* Brew-batch: blood_healer uses lower threshold (relies on blood barrage above 50%) */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack: blood barrage emphasis for sustain */
    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        int actual_attack;  /* 0=ice, 1=blood, 2=atk */

        if (hp_pct < 0.40f) {
            /* Low HP: blood barrage for heal + triple-eat */
            attack_style = OPP_STYLE_MAGE;
            actual_attack = 1;  /* blood */
        } else if (hp_pct < 0.70f) {
            /* Medium HP: strongly prefer blood barrage (~80%) for sustain */
            if (rand_float(env) < 0.80f) {
                attack_style = OPP_STYLE_MAGE;
                actual_attack = 1;  /* blood */
            } else {
                /* Off-prayer attack with style bias */
                if (rand_float(env) < opp->off_prayer_rate) {
                    attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
                } else {
                    attack_style = rand_int(env, 3);
                }
                if (attack_style == OPP_STYLE_MAGE) {
                    /* Ice to freeze, not blood (already handled blood above) */
                    actual_attack = (target->frozen_ticks == 0 && target->freeze_immunity_ticks == 0)
                                    ? 0 : 1;  /* ice if can freeze, else blood */
                } else {
                    actual_attack = 2;  /* ATK */
                }
            }
        } else {
            /* High HP: normal off-prayer targeting with ice barrage for freeze */
            if (rand_float(env) < opp->off_prayer_rate) {
                attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
            } else {
                attack_style = rand_int(env, 3);
            }
            if (attack_style == OPP_STYLE_MAGE) {
                actual_attack = (target->frozen_ticks == 0 && target->freeze_immunity_ticks == 0)
                                ? 0 : 1;  /* ice if can freeze, else blood for sustain */
            } else {
                actual_attack = 2;  /* ATK */
            }
        }

        /* Apply boost potions */
        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
        if (rand_float(env) < opp->offensive_prayer_miss) {
            actions[HEAD_LOADOUT] = LOADOUT_KEEP;
        } else {
            opp_apply_gear_switch(actions, attack_style);
        }

        /* Tank gear when critically low and not casting blood */
        if (hp_pct < 0.35f && actual_attack != 1) {
            actions[HEAD_LOADOUT] = LOADOUT_TANK;
        }

        /* Attack action */
        if (actual_attack == 0) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (actual_attack == 1) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement: maintain farcast-5 distance */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (self->frozen_ticks == 0) {
            if (target->frozen_ticks > 0 && dist < 5) {
                /* Step back to range 5 from frozen target */
                actions[HEAD_COMBAT] = MOVE_FARCAST_5;
            } else if (dist < 4 && target->frozen_ticks == 0) {
                /* Maintain distance from unfrozen target */
                actions[HEAD_COMBAT] = MOVE_FARCAST_5;
            } else if (opp->target_fleeing_ticks >= 2 && dist > 5) {
                /* Anti-kite: close to farcast-5 range */
                actions[HEAD_COMBAT] = MOVE_FARCAST_5;
            }
        }
    }
}

/* =========================================================================
 * Gmaul Combo: KO specialist with spec → gmaul instant follow-up.
 * Degrades to improved-style at tier 0 (DDS spec only, no gmaul combo).
 * At tier 1+ with gmaul available, fires spec→gmaul for burst KO.
 * ========================================================================= */

#define COMBO_IDLE       0
#define COMBO_SPEC_FIRED 1

static void opp_gmaul_combo(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;
    int has_gmaul = player_has_gmaul(self);

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating (same as improved) */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Combo state machine: follow-up gmaul after spec fired */
    if (opp->combo_state == COMBO_SPEC_FIRED && has_gmaul && !eating) {
        /* Gmaul follow-up — instant spec, bypasses attack timer */
        actions[HEAD_LOADOUT] = LOADOUT_GMAUL;
        actions[HEAD_COMBAT] = ATTACK_ATK;
        opp->combo_state = COMBO_IDLE;
        return;
    }
    /* Reset combo if we ate (can't follow up) or don't have gmaul */
    opp->combo_state = COMBO_IDLE;

    /* 4. Attack decision */
    if (opp_attack_ready(self) && !eating) {
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);

        /* KO opportunity: target in KO range and we have enough spec energy */
        int melee_spec_cost = get_melee_spec_cost(self->melee_spec_weapon);
        int gmaul_cost = 50;  /* granite maul always 50% */
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_combo = (has_gmaul &&
                           target_hp_pct < opp->ko_threshold &&
                           self->special_energy >= melee_spec_cost + gmaul_cost &&
                           target->prayer != PRAYER_PROTECT_MELEE &&
                           can_spec_range);

        /* Also check ranged spec for variety (no gmaul follow-up, just raw spec) */
        uint8_t ranged_spec = find_best_ranged_spec(self);
        int should_ranged_spec = (ranged_spec != 0 &&
                                 target_hp_pct < opp->ko_threshold &&
                                 self->special_energy >= get_ranged_spec_cost(self->ranged_spec_weapon) &&
                                 target->prayer != PRAYER_PROTECT_RANGED &&
                                 rand_float(env) < 0.25f);  /* 25% chance to use ranged spec */

        /* Anti-kite: cancel melee combo if target fleeing */
        if ((should_combo || should_ranged_spec) &&
            opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_combo = 0;
            should_ranged_spec = 0;
        }

        if (should_combo) {
            /* Fire melee spec → next tick gmaul follows */
            opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
            actions[HEAD_COMBAT] = ATTACK_ATK;
            opp->combo_state = COMBO_SPEC_FIRED;
        } else if (should_ranged_spec) {
            /* Ranged spec (no gmaul follow-up) */
            actions[HEAD_LOADOUT] = LOADOUT_SPEC_RANGE;
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else {
            /* Normal improved-style play */
            int attack_style;
            if (rand_float(env) < opp->off_prayer_rate) {
                attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
            } else {
                attack_style = rand_int(env, 3);
            }

            opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

            /* Regular melee spec (DDS at tier 0, better at higher tiers) — no gmaul combo */
            int should_regular_spec = (!has_gmaul &&
                                      self->special_energy >= melee_spec_cost &&
                                      target->prayer != PRAYER_PROTECT_MELEE &&
                                      target_hp_pct < 0.50f &&
                                      can_spec_range);
            if (should_regular_spec && opp->target_fleeing_ticks < 2) {
                opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
                actions[HEAD_COMBAT] = ATTACK_ATK;
            } else {
                /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
                if (rand_float(env) < opp->offensive_prayer_miss) {
                    actions[HEAD_LOADOUT] = LOADOUT_KEEP;
                } else {
                    opp_apply_gear_switch(actions, attack_style);
                }

                if (attack_style == OPP_STYLE_MAGE) {
                    actions[HEAD_COMBAT] = (target->frozen_ticks == 0 &&
                                           target->freeze_immunity_ticks == 0)
                                          ? ATTACK_ICE : ATTACK_BLOOD;
                } else {
                    actions[HEAD_COMBAT] = ATTACK_ATK;
                }
            }
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement: step under frozen target, farcast-3 for anti-kite */
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
            actions[HEAD_COMBAT] = MOVE_FARCAST_3;
        }
    }
}

/* =========================================================================
 * Range Kiter: ranged-dominant fighter who maintains distance.
 * Works at all tiers — rune crossbow does ranged damage at tier 0.
 * Gains spec capability at higher tiers (ACB/ZCB/dark bow/morr jav).
 * Maintains farcast-5, ice barrage to freeze, ranged primary (~60-70%).
 * ========================================================================= */

static void opp_range_kiter(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    /* 1. Defensive prayer */
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    /* 2. Multi-threshold eating + emergency blood barrage sustain */
    if (hp_pct < opp->eat_triple_threshold && cons.can_food && cons.can_brew && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->potion_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_POTION] = POTION_BREW;
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_karambwan) {
        actions[HEAD_FOOD] = FOOD_EAT;
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->food_cooldown = 3; opp->karambwan_cooldown = 2;
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        /* Brew-batch: keep eating to 90%+ before restoring */
        actions[HEAD_POTION] = POTION_BREW;
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        actions[HEAD_POTION] = POTION_RESTORE;
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    /* 3. Attack: ranged-dominant with freeze support and ranged specs */
    if (opp_attack_ready(self) && !eating) {
        /* Check ranged spec availability */
        uint8_t ranged_spec = find_best_ranged_spec(self);
        int has_ranged_spec = (ranged_spec != 0);
        int ranged_spec_cost = has_ranged_spec
                               ? get_ranged_spec_cost(self->ranged_spec_weapon) : 100;

        /* Ranged spec: freeze → spec from distance is primary KO pattern */
        int should_ranged_spec = (has_ranged_spec &&
                                 self->special_energy >= ranged_spec_cost &&
                                 target->prayer != PRAYER_PROTECT_RANGED &&
                                 target_hp_pct < 0.55f);

        /* Anti-kite not needed for ranged — we WANT distance */

        if (should_ranged_spec && (target->frozen_ticks > 0 || dist >= 3)) {
            /* Fire ranged spec from distance */
            actions[HEAD_LOADOUT] = LOADOUT_SPEC_RANGE;
            actions[HEAD_COMBAT] = ATTACK_ATK;
        } else {
            /* Style selection: ranged-biased via style_bias (initialized with range preference)
             * + force ranged at distance, force melee only when adjacent and frozen */
            int attack_style;
            int force_melee = (self->frozen_ticks > 0 && dist <= 1);
            int prefer_ranged = (dist >= 3 || target->frozen_ticks > 0);

            if (force_melee) {
                attack_style = OPP_STYLE_MELEE;
            } else if (prefer_ranged && rand_float(env) < 0.80f) {
                attack_style = OPP_STYLE_RANGED;
            } else if (rand_float(env) < opp->off_prayer_rate) {
                attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
            } else {
                attack_style = rand_int(env, 3);
            }

            /* Emergency blood barrage healing */
            int actual_attack;
            if (hp_pct < 0.30f && attack_style == OPP_STYLE_MAGE) {
                actual_attack = 1;  /* blood */
            } else if (attack_style == OPP_STYLE_MAGE) {
                actual_attack = (target->frozen_ticks == 0 &&
                                target->freeze_immunity_ticks == 0)
                               ? 0 : 2;  /* ice if can freeze, else just ATK (ranged fallback) */
                if (actual_attack == 2) attack_style = OPP_STYLE_RANGED;  /* fallback to ranged */
            } else {
                actual_attack = 2;  /* ATK */
            }

            /* Boost potions */
            opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

            /* Melee spec (DDS etc) when close — fallback */
            int melee_spec_cost = get_melee_spec_cost(self->melee_spec_weapon);
            int can_melee_spec = (self->special_energy >= melee_spec_cost &&
                                 target->prayer != PRAYER_PROTECT_MELEE &&
                                 dist <= 1 && self->frozen_ticks == 0);
            if (can_melee_spec && target_hp_pct < 0.40f && !has_ranged_spec) {
                opp_apply_gear_switch(actions, OPP_STYLE_SPEC);
                actions[HEAD_COMBAT] = ATTACK_ATK;
            } else {
                /* Gear switch — offensive_prayer_miss: skip switch to omit auto-prayer */
                if (rand_float(env) < opp->offensive_prayer_miss) {
                    actions[HEAD_LOADOUT] = LOADOUT_KEEP;
                } else {
                    opp_apply_gear_switch(actions, attack_style);
                }

                if (actual_attack == 0) {
                    actions[HEAD_COMBAT] = ATTACK_ICE;
                } else if (actual_attack == 1) {
                    actions[HEAD_COMBAT] = ATTACK_BLOOD;
                } else {
                    actions[HEAD_COMBAT] = ATTACK_ATK;
                }
            }
        }
    } else if (!opp_attack_ready(self)) {
        /* Movement: maintain farcast-5, step back after freeze */
        if (self->frozen_ticks == 0) {
            if (target->frozen_ticks > 0 && dist < 5) {
                /* Step back to range 5 from frozen target */
                actions[HEAD_COMBAT] = MOVE_FARCAST_5;
            } else if (dist < 4) {
                /* Maintain distance from approaching target */
                actions[HEAD_COMBAT] = MOVE_FARCAST_5;
            } else if (dist > 7) {
                /* Don't let them get too far — close to farcast-5 */
                actions[HEAD_COMBAT] = MOVE_FARCAST_5;
            }
        }
    }
}

/* =========================================================================
 * Mixed policy selection (MixedEasy/MixedMedium/MixedHard/MixedHardBalanced)
 * ========================================================================= */

/* MixedEasy weights: panicking=0.18, true_random=0.18, weak_random=0.18,
   semi_random=0.15, sticky_prayer=0.10, random_eater=0.10, prayer_rookie=0.06,
   improved=0.05 */
static const OpponentType MIXED_EASY_POOL[] = {
    OPP_PANICKING, OPP_TRUE_RANDOM, OPP_WEAK_RANDOM, OPP_SEMI_RANDOM,
    OPP_STICKY_PRAYER, OPP_RANDOM_EATER, OPP_PRAYER_ROOKIE, OPP_IMPROVED,
};
/* Cumulative weights * 100 for integer comparison */
static const int MIXED_EASY_CUM_WEIGHTS[] = {18, 36, 54, 69, 79, 89, 95, 100};
#define MIXED_EASY_POOL_SIZE 8

/* MixedMedium weights: random_eater=0.25, prayer_rookie=0.20, sticky_prayer=0.20,
   semi_random=0.15, improved=0.10, (patient deferred to Python) */
static const OpponentType MIXED_MEDIUM_POOL[] = {
    OPP_RANDOM_EATER, OPP_PRAYER_ROOKIE, OPP_STICKY_PRAYER,
    OPP_SEMI_RANDOM, OPP_IMPROVED,
};
static const int MIXED_MEDIUM_CUM_WEIGHTS[] = {25, 45, 65, 80, 100};
#define MIXED_MEDIUM_POOL_SIZE 5

/* MixedHard: uniform over 5 policies (20% each) */
static const OpponentType MIXED_HARD_POOL[] = {
    OPP_IMPROVED, OPP_ONETICK, OPP_UNPREDICTABLE_IMPROVED,
    OPP_UNPREDICTABLE_ONETICK, OPP_RANDOM_EATER,
};
static const int MIXED_HARD_CUM_WEIGHTS[] = {20, 40, 60, 80, 100};
#define MIXED_HARD_POOL_SIZE 5

/* MixedHardBalanced: random_eater=25%, improved=30%, unpredictable_improved=20%,
   onetick=15%, unpredictable_onetick=10% */
static const OpponentType MIXED_HARD_BALANCED_POOL[] = {
    OPP_RANDOM_EATER, OPP_IMPROVED, OPP_UNPREDICTABLE_IMPROVED,
    OPP_ONETICK, OPP_UNPREDICTABLE_ONETICK,
};
static const int MIXED_HARD_BALANCED_CUM_WEIGHTS[] = {25, 55, 75, 90, 100};
#define MIXED_HARD_BALANCED_POOL_SIZE 5

static OpponentType opp_select_from_pool(
    OsrsEnv* env, const OpponentType* pool, const int* cum_weights, int pool_size
) {
    int r = rand_int(env, 100);
    for (int i = 0; i < pool_size; i++) {
        if (r < cum_weights[i]) return pool[i];
    }
    return pool[pool_size - 1];
}

/* =========================================================================
 * Main entry point: generate opponent action
 * ========================================================================= */

static void opponent_reset(OsrsEnv* env, OpponentState* opp) {
    opp->food_cooldown = 0;
    opp->potion_cooldown = 0;
    opp->karambwan_cooldown = 0;
    opp->current_prayer_set = 0;

    /* Phase 2 state reset */
    opp->fake_switch_pending = 0;
    opp->fake_switch_style = -1;
    opp->opponent_prayer_at_fake = -1;
    opp->fake_switch_failed = 0;
    opp->pending_prayer_value = 0;
    opp->pending_prayer_delay = 0;
    opp->last_target_gear_style = -1;

    /* Per-episode eating thresholds with noise */
    opp->eat_triple_threshold = 0.30f + (rand_float(env) * 0.10f - 0.05f);
    opp->eat_double_threshold = 0.50f + (rand_float(env) * 0.10f - 0.05f);
    opp->eat_brew_threshold   = 0.70f + (rand_float(env) * 0.10f - 0.05f);

    /* Boss opponent reading ability — reset per-tick state */
    opp->has_read_this_tick = 0;
    opp->read_agent_style = ATTACK_STYLE_NONE;
    opp->read_agent_prayer = PRAYER_NONE;
    opp->read_chance = 0.0f;
    opp->read_agent_moving = 0;
    opp->prev_dist_to_target = 0;
    opp->target_fleeing_ticks = 0;

    /* Per-episode resets for specific policies */
    if (opp->type == OPP_PANICKING) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        opp->chosen_prayer = prayers[rand_int(env, 3)];
        opp->chosen_style = rand_int(env, 3);
    }

    /* Mixed policies: select sub-policy */
    if (opp->type == OPP_MIXED_EASY) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_EASY_POOL, MIXED_EASY_CUM_WEIGHTS, MIXED_EASY_POOL_SIZE);
    } else if (opp->type == OPP_MIXED_MEDIUM) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_MEDIUM_POOL, MIXED_MEDIUM_CUM_WEIGHTS, MIXED_MEDIUM_POOL_SIZE);
    } else if (opp->type == OPP_MIXED_HARD) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_HARD_POOL, MIXED_HARD_CUM_WEIGHTS, MIXED_HARD_POOL_SIZE);
    } else if (opp->type == OPP_MIXED_HARD_BALANCED) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_HARD_BALANCED_POOL, MIXED_HARD_BALANCED_CUM_WEIGHTS,
            MIXED_HARD_BALANCED_POOL_SIZE);
    } else if (opp->type == OPP_PFSP && env->pvp_runtime.pfsp.pool_size > 0) {
        int idx = 0;
        int r = rand_int(env, 1000);
        for (int i = 0; i < env->pvp_runtime.pfsp.pool_size; i++) {
            if (r < env->pvp_runtime.pfsp.cum_weights[i]) { idx = i; break; }
        }
        env->pvp_runtime.pfsp.active_pool_idx = idx;
        opp->active_sub_policy = env->pvp_runtime.pfsp.pool[idx];

        // Toggle opponent mode: selfplay uses external Python actions,
        // scripted opponents use C-generated actions
        if (opp->active_sub_policy == OPP_SELFPLAY) {
            env->pvp_runtime.use_c_opponent = 0;
            env->pvp_runtime.use_external_opponent_actions = 1;
            if (env->ocean_io.selfplay_mask) *env->ocean_io.selfplay_mask = 1;
        } else {
            env->pvp_runtime.use_c_opponent = 1;
            env->pvp_runtime.use_external_opponent_actions = 0;
            if (env->ocean_io.selfplay_mask) *env->ocean_io.selfplay_mask = 0;
        }
    } else if (opp->type == OPP_PFSP) {
        // PFSP pool not yet configured (set_pfsp_weights called after env creation).
        // Fall back to OPP_IMPROVED so the first episode isn't against a no-op opponent.
        opp->active_sub_policy = OPP_IMPROVED;
        env->pvp_runtime.pfsp.active_pool_idx = -1;  // sentinel: don't track in PFSP stats
    }

    /* Per-episode randomized decision parameters — resolved from sub-policy
     * so PFSP and mixed pools get the correct ranges. */
    OpponentType resolved = opp->active_sub_policy ? opp->active_sub_policy : opp->type;
    if (resolved > 0 && resolved <= OPP_RANGE_KITER) {
        const OpponentRandRanges* r = &OPP_RAND_RANGES[resolved];
        opp->prayer_accuracy = rand_range(env, r->prayer_accuracy);
        opp->off_prayer_rate = rand_range(env, r->off_prayer_rate);
        opp->offensive_prayer_rate = rand_range(env, r->offensive_prayer_rate);
        opp->action_delay_chance = rand_range(env, r->action_delay_chance);
        opp->mistake_rate = rand_range(env, r->mistake_rate);
        opp->offensive_prayer_miss = rand_range(env, r->offensive_prayer_miss);
    }

    /* Boss reading ability */
    if (resolved == OPP_MASTER_NH) {
        opp->read_chance = 0.10f;
    } else if (resolved == OPP_SAVANT_NH) {
        opp->read_chance = 0.25f;
    } else if (resolved == OPP_NIGHTMARE_NH) {
        opp->read_chance = 0.50f;
    }

    /* Vengeance fighter: lunar spellbook (no freeze/blood, has veng) */
    if (resolved == OPP_VENG_FIGHTER) {
        env->players[1].is_lunar_spellbook = 1;
    }

    /* Per-episode style bias: weighted preference for mage/ranged/melee.
     * Sampled for improved+ opponents that use off-prayer targeting. */
    if (resolved == OPP_IMPROVED || resolved == OPP_ONETICK ||
        resolved == OPP_UNPREDICTABLE_IMPROVED || resolved == OPP_UNPREDICTABLE_ONETICK ||
        (resolved >= OPP_ADVANCED_NH && resolved <= OPP_NIGHTMARE_NH) ||
        resolved == OPP_BLOOD_HEALER || resolved == OPP_GMAUL_COMBO ||
        resolved == OPP_RANGE_KITER) {
        float raw[3];
        for (int i = 0; i < 3; i++) raw[i] = 0.33f + (rand_float(env) - 0.5f) * 0.4f;
        float sum = raw[0] + raw[1] + raw[2];
        for (int i = 0; i < 3; i++) opp->style_bias[i] = raw[i] / sum;
    } else {
        opp->style_bias[0] = opp->style_bias[1] = opp->style_bias[2] = 0.333f;
    }

    /* gmaul_combo: per-episode KO threshold + combo state reset */
    if (resolved == OPP_GMAUL_COMBO) {
        opp->combo_state = 0;
        opp->ko_threshold = 0.45f + rand_float(env) * 0.15f;  /* 45-60% */
    }
}

static void generate_opponent_action(OsrsEnv* env, OpponentState* opp) {
    int* actions = &env->pending_actions[1 * NUM_ACTION_HEADS];

    /* Clear actions to zero (KEEP/NONE for all heads) */
    memset(actions, 0, NUM_ACTION_HEADS * sizeof(int));

    /* Update flee tracking for all opponents */
    opp_update_flee_tracking(opp, &env->players[1], &env->players[0]);

    /* Resolve active policy for mixed types */
    OpponentType active = opp->type;
    if (active == OPP_MIXED_EASY || active == OPP_MIXED_MEDIUM ||
        active == OPP_MIXED_HARD || active == OPP_MIXED_HARD_BALANCED ||
        active == OPP_PFSP) {
        active = opp->active_sub_policy;
    }

    /* Dispatch to policy implementation */
    switch (active) {
        case OPP_TRUE_RANDOM:
            opp_true_random(env, actions);
            break;
        case OPP_PANICKING:
            opp_panicking(env, opp, actions);
            break;
        case OPP_WEAK_RANDOM:
            opp_weak_random(env, opp, actions);
            break;
        case OPP_SEMI_RANDOM:
            opp_semi_random(env, opp, actions);
            break;
        case OPP_STICKY_PRAYER:
            opp_sticky_prayer(env, opp, actions);
            break;
        case OPP_RANDOM_EATER:
            opp_random_eater(env, opp, actions);
            break;
        case OPP_PRAYER_ROOKIE:
            opp_prayer_rookie(env, opp, actions);
            break;
        case OPP_IMPROVED:
            opp_improved(env, opp, actions);
            break;
        case OPP_ONETICK:
            opp_onetick(env, opp, actions);
            break;
        case OPP_UNPREDICTABLE_IMPROVED:
            opp_unpredictable_improved(env, opp, actions);
            break;
        case OPP_UNPREDICTABLE_ONETICK:
            opp_unpredictable_onetick(env, opp, actions);
            break;
        case OPP_NOVICE_NH:
            opp_novice_nh(env, opp, actions);
            break;
        case OPP_APPRENTICE_NH:
            opp_apprentice_nh(env, opp, actions);
            break;
        case OPP_COMPETENT_NH:
            opp_competent_nh(env, opp, actions);
            break;
        case OPP_INTERMEDIATE_NH:
            opp_intermediate_nh(env, opp, actions);
            break;
        case OPP_ADVANCED_NH:
            opp_advanced_nh(env, opp, actions);
            break;
        case OPP_PROFICIENT_NH:
            opp_proficient_nh(env, opp, actions);
            break;
        case OPP_EXPERT_NH:
            opp_expert_nh(env, opp, actions);
            break;
        case OPP_MASTER_NH:
            opp_master_nh(env, opp, actions);
            break;
        case OPP_SAVANT_NH:
            opp_savant_nh(env, opp, actions);
            break;
        case OPP_NIGHTMARE_NH:
            opp_nightmare_nh(env, opp, actions);
            break;
        case OPP_VENG_FIGHTER:
            opp_veng_fighter(env, opp, actions);
            break;
        case OPP_BLOOD_HEALER:
            opp_blood_healer(env, opp, actions);
            break;
        case OPP_GMAUL_COMBO:
            opp_gmaul_combo(env, opp, actions);
            break;
        case OPP_RANGE_KITER:
            opp_range_kiter(env, opp, actions);
            break;
        default:
            /* OPP_NONE or unsupported: leave NOOPs */
            break;
    }
}

static void swap_players_and_pending(OsrsEnv* env) {
    Player tmp_player = env->players[0];
    env->players[0] = env->players[1];
    env->players[1] = tmp_player;

    int tmp_actions[NUM_ACTION_HEADS];
    memcpy(tmp_actions, env->pending_actions, NUM_ACTION_HEADS * sizeof(int));
    memcpy(
        env->pending_actions,
        env->pending_actions + NUM_ACTION_HEADS,
        NUM_ACTION_HEADS * sizeof(int)
    );
    memcpy(
        env->pending_actions + NUM_ACTION_HEADS,
        tmp_actions,
        NUM_ACTION_HEADS * sizeof(int)
    );
}

static void generate_opponent_action_for_player0(OsrsEnv* env, OpponentState* opp) {
    swap_players_and_pending(env);
    generate_opponent_action(env, opp);
    swap_players_and_pending(env);
}

#endif /* OSRS_PVP_OPPONENTS_H */
