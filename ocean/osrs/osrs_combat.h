/**
 * @fileoverview osrs_combat.h — pure combat math shared by all encounters.
 *
 * stateless functions with no dependencies beyond <math.h>. use these instead
 * of reimplementing combat formulas per encounter.
 *
 * SHARED FUNCTIONS:
 *   osrs_hit_chance(att_roll, def_roll)       standard OSRS accuracy formula
 *   osrs_tbow_acc_mult(target_magic)          twisted bow accuracy multiplier
 *   osrs_tbow_dmg_mult(target_magic)          twisted bow damage multiplier
 *   osrs_barrage_resolve(targets, ...)        barrage 3x3 AoE with independent rolls
 *   osrs_npc_melee_max_hit(str, bonus)        NPC melee max hit from stats
 *   osrs_npc_ranged_max_hit(range, bonus)     NPC ranged max hit from stats
 *   osrs_npc_magic_max_hit(base, pct)         NPC magic max hit from stats
 *   osrs_npc_max_hit(style, ...)              dispatches to style-specific formula
 *   osrs_npc_attack_roll(att, bonus)          NPC attack roll
 *   osrs_player_def_roll_vs_npc(def,mag,b,s)  player defence roll vs NPC
 *   encounter_xorshift(state)                 xorshift32 RNG step
 *   encounter_rand_int(state, max)            random int in [0, max)
 *   encounter_rand_float(state)               random float in [0, 1)
 *   encounter_npc_roll_attack(att,def,mh,rng) NPC accuracy+damage in one call
 *   encounter_prayer_correct_for_style(p, s)  prayer blocks attack style check
 *   encounter_magic_hit_delay(dist, is_p)     magic projectile flight delay (ticks)
 *   encounter_ranged_hit_delay(dist, is_p)    ranged projectile flight delay (ticks)
 *   encounter_dist_to_npc(px,py,nx,ny,sz)     chebyshev dist to multi-tile NPC
 *
 * PLAYER COMBAT:
 *   osrs_player_eff_level(base,prayer,style)  effective level calculation
 *   osrs_player_att_roll(eff,bonus)           attack roll
 *   osrs_player_melee_max_hit(eff,str)        melee max hit
 *   osrs_player_ranged_max_hit(eff,str)       ranged max hit
 *   osrs_player_magic_max_hit(base,pct)       magic max hit
 *   osrs_prayer_reduce_damage(dmg,pr,st,pvp)  PvE 100% block vs PvP 40% reduction
 *   osrs_hit_chance_double(att,def)           osmumten/confliction double roll
 *   osrs_sum_equipment_bonuses(loadout,out)   sum gear stats from ITEM_DATABASE
 *
 * SEE ALSO:
 *   osrs_special_attacks.h  weapon special attack dispatch (blowpipe spec moved here)
 *   osrs_encounter.h        encounter-level abstractions (damage, movement, gear, etc.)
 *   osrs_pvp_combat.h       PvP-specific combat (prayer, veng, recoil, pending hits)
 */

#ifndef OSRS_COMBAT_H
#define OSRS_COMBAT_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include "osrs_types.h"
#include "osrs_items.h"

/* standard OSRS accuracy formula.
   att_roll and def_roll are pre-computed: eff_level * (bonus + 64).
   returns hit probability in [0, 1]. */
static inline float osrs_hit_chance(int att_roll, int def_roll) {
    if (att_roll > def_roll)
        return 1.0f - (float)(def_roll + 2) / (2.0f * (float)(att_roll + 1));
    else
        return (float)att_roll / (2.0f * (float)(def_roll + 1));
}

static inline float osrs_hit_chance_double(int att_roll, int def_roll);

/* twisted bow accuracy multiplier.
   target_magic = min(max(npc_magic_level, npc_magic_attack_bonus), 250).
   formula from RuneLite TwistedBow._accuracyMultiplier. */
static inline float osrs_tbow_acc_mult(int target_magic) {
    int m = target_magic < 250 ? target_magic : 250;
    /* ref: osrs-sdk TwistedBow.ts _accuracyMultiplier
       linear term uses 3*magic, quadratic uses 3*magic/10 */
    float lin = (float)(3 * m);
    float quad = lin / 10.0f;
    float mult = (140.0f + (lin - 10.0f) / 100.0f - (quad - 100.0f) * (quad - 100.0f) / 100.0f) / 100.0f;
    if (mult > 1.4f) mult = 1.4f;
    if (mult < 0.0f) mult = 0.0f;
    return mult;
}

/* twisted bow damage multiplier.
   same input as accuracy multiplier.
   ref: osrs-sdk TwistedBow.ts _damageMultiplier */
static inline float osrs_tbow_dmg_mult(int target_magic) {
    int m = target_magic < 250 ? target_magic : 250;
    float lin = (float)(3 * m);
    float quad = lin / 10.0f;
    float mult = (250.0f + (lin - 14.0f) / 100.0f - (quad - 140.0f) * (quad - 140.0f) / 100.0f) / 100.0f;
    if (mult > 2.5f) mult = 2.5f;
    if (mult < 0.0f) mult = 0.0f;
    return mult;
}


/* all encounters should use these instead of reimplementing.
   state must be non-zero. */
static inline uint32_t encounter_xorshift(uint32_t* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static inline int encounter_rand_int(uint32_t* rng_state, int max) {
    if (max <= 0) return 0;
    return (int)(encounter_xorshift(rng_state) % (unsigned)max);
}

static inline float encounter_rand_float(uint32_t* rng_state) {
    return (float)(encounter_xorshift(rng_state) & 0xFFFF) / 65536.0f;
}


#define BARRAGE_MAX_HITS 9
#define BARRAGE_FREEZE_TICKS 32

/* per-target info for barrage AoE. caller fills in the target array,
   osrs_barrage_resolve does accuracy/damage rolls and writes results back. */
typedef struct {
    int active;          /* in: 1 if this target slot is valid */
    int x, y;            /* in: NPC SW corner tile position */
    int def_level;       /* in: NPC defence level */
    int magic_def_bonus; /* in: NPC magic defence bonus */
    int npc_idx;         /* in: index into caller's NPC array (for callbacks) */
    int* frozen_ticks;   /* in: pointer to NPC's frozen_ticks (NULL = no freeze tracking) */
    int hit;             /* out: 1 = accuracy passed, 0 = splashed */
    int damage;          /* out: damage rolled (0 if splashed) */
} BarrageTarget;

/* result from a barrage cast */
typedef struct {
    int total_damage;    /* sum of all damage across AoE */
    int num_hits;        /* number of targets that were rolled against */
    int num_successful;  /* number that passed accuracy (hit=1) */
} BarrageResult;

/* resolve a barrage spell against a primary target + 3x3 AoE.
   - targets[0] is the primary target (always rolled first)
   - targets[1..max_targets-1] are potential AoE targets (only those within
     1 tile of primary are rolled against)
   - att_roll: pre-computed attacker magic roll (eff_level * (bonus + 64))
   - max_hit: barrage spell max hit
   - rng_state: pointer to RNG state for rolls
   - max_targets: size of targets array

   the function sets hit/damage on each target. if spell_type is ICE and
   a target's frozen_ticks pointer is set, freeze is applied immediately
   at cast time (ref: osrs-sdk IceBarrageSpell.ts). caller is responsible
   for queueing damage as pending hits with appropriate delay.

   returns aggregate result for reward/heal calculations. */
static inline BarrageResult osrs_barrage_resolve(
    BarrageTarget* targets, int max_targets,
    int att_roll, int max_hit, uint32_t* rng_state,
    int spell_type,
    int primary_use_double_accuracy
) {
    BarrageResult result = { 0, 0, 0 };

    if (max_targets < 1 || !targets[0].active) return result;

    /* primary target (index 0) always gets rolled */
    int px = targets[0].x, py = targets[0].y;
    {
        int def_roll = (targets[0].def_level + 8) * (targets[0].magic_def_bonus + 64);
        float chance = primary_use_double_accuracy
            ? osrs_hit_chance_double(att_roll, def_roll)
            : osrs_hit_chance(att_roll, def_roll);
        targets[0].hit = encounter_rand_float(rng_state) < chance;
        targets[0].damage = targets[0].hit ? encounter_rand_int(rng_state, max_hit + 1) : 0;
        result.total_damage += targets[0].damage;
        result.num_hits++;
        if (targets[0].hit) {
            result.num_successful++;
            /* ice barrage: freeze immediately at cast time */
            if (spell_type == 1 /* ENCOUNTER_SPELL_ICE */ && targets[0].frozen_ticks)
                *targets[0].frozen_ticks = BARRAGE_FREEZE_TICKS;
        }
    }

    /* AoE: roll against all other active targets within 1 tile of primary */
    for (int i = 1; i < max_targets && result.num_hits < BARRAGE_MAX_HITS; i++) {
        if (!targets[i].active) continue;
        int dx = targets[i].x - px;
        int dy = targets[i].y - py;
        if (dx < -1 || dx > 1 || dy < -1 || dy > 1) continue;

        int def_roll = (targets[i].def_level + 8) * (targets[i].magic_def_bonus + 64);
        float chance = osrs_hit_chance(att_roll, def_roll);
        targets[i].hit = encounter_rand_float(rng_state) < chance;
        targets[i].damage = targets[i].hit ? encounter_rand_int(rng_state, max_hit + 1) : 0;
        result.total_damage += targets[i].damage;
        result.num_hits++;
        if (targets[i].hit) {
            result.num_successful++;
            if (spell_type == 1 /* ENCOUNTER_SPELL_ICE */ && targets[i].frozen_ticks)
                *targets[i].frozen_ticks = BARRAGE_FREEZE_TICKS;
        }
    }

    return result;
}


/* NPC melee max hit: floor((str + 8) * (melee_str_bonus + 64) + 320) / 640) */
static inline int osrs_npc_melee_max_hit(int str_level, int melee_str_bonus) {
    return ((str_level + 8) * (melee_str_bonus + 64) + 320) / 640;
}

/* NPC ranged max hit: floor(0.5 + (range + 8) * (ranged_str_bonus + 64) / 640) */
static inline int osrs_npc_ranged_max_hit(int range_level, int ranged_str_bonus) {
    return (int)(0.5 + (double)(range_level + 8) * (ranged_str_bonus + 64) / 640.0);
}

/* NPC magic max hit: floor(base_spell_dmg * magic_dmg_pct / 100).
   magic_dmg_pct=100 means 1.0x multiplier, 175 means 1.75x. */
static inline int osrs_npc_magic_max_hit(int base_spell_dmg, int magic_dmg_pct) {
    return base_spell_dmg * magic_dmg_pct / 100;
}

/* NPC attack roll: (att_level + 9) * (att_bonus + 64).
   NPCs don't have prayer or void bonuses — just level + invisible +9. */
static inline int osrs_npc_attack_roll(int att_level, int att_bonus) {
    return (att_level + 9) * (att_bonus + 64);
}

/* player defence roll against NPC attack.
   OSRS formula: eff_def = level + stance_bonus + 8. players don't have the
   hidden +1 that NPCs get (that's why NPC attack roll uses +9).
   our sim doesn't model stance bonuses, so stance_bonus = 0.
   vs melee/ranged: (def_level + 8) * (def_bonus + 64).
   vs magic: (floor(magic_level * 0.7 + def_level * 0.3) + 8) * (def_bonus + 64).
   ref: osrs-sdk MeleeWeapon.ts:164, OSRS wiki combat formulas. */
static inline int osrs_player_def_roll_vs_npc(
    int def_level, int magic_level, int def_bonus, int attack_style
) {
    int eff_def;
    if (attack_style == 3) {  /* ATTACK_STYLE_MAGIC = 3 */
        eff_def = (int)(magic_level * 0.7 + def_level * 0.3) + 8;
    } else {
        eff_def = def_level + 8;
    }
    return eff_def * (def_bonus + 64);
}

/* pick the correct player defence bonus for an incoming NPC attack.
   attack_style: 1=melee, 2=ranged, 3=magic.
   melee_style: 0=stab, 1=slash, 2=crush (only used when attack_style == 1). */
static inline int encounter_player_def_bonus(
    int def_stab, int def_slash, int def_crush, int def_magic, int def_ranged,
    int attack_style, int melee_style
) {
    if (attack_style == 2) return def_ranged;  /* ATTACK_STYLE_RANGED */
    if (attack_style == 3) return def_magic;   /* ATTACK_STYLE_MAGIC */
    /* melee: select by sub-style */
    if (melee_style == 1) return def_slash;    /* MELEE_STYLE_SLASH */
    if (melee_style == 2) return def_crush;    /* MELEE_STYLE_CRUSH */
    return def_stab;                           /* MELEE_STYLE_STAB */
}

/* NPC max hit by style: dispatches to melee/ranged/magic formula.
   for magic, uses magic_base_dmg * magic_dmg_pct / 100. */
static inline int osrs_npc_max_hit(
    int attack_style,
    int str_level, int range_level,
    int melee_str_bonus, int ranged_str_bonus,
    int magic_base_dmg, int magic_dmg_pct
) {
    if (attack_style == 1) /* ATTACK_STYLE_MELEE = 1 */
        return osrs_npc_melee_max_hit(str_level, melee_str_bonus);
    if (attack_style == 2) /* ATTACK_STYLE_RANGED = 2 */
        return osrs_npc_ranged_max_hit(range_level, ranged_str_bonus);
    if (attack_style == 3) /* ATTACK_STYLE_MAGIC = 3 */
        return osrs_npc_magic_max_hit(magic_base_dmg, magic_dmg_pct);
    return 0;
}

/* NPC attack roll: accuracy check + damage roll in one call.
   returns damage (0 on miss). caller handles prayer separately. */
static inline int encounter_npc_roll_attack(
    int att_roll, int def_roll, int max_hit, uint32_t* rng_state
) {
    int dmg = encounter_rand_int(rng_state, max_hit + 1);
    if (encounter_rand_float(rng_state) >= osrs_hit_chance(att_roll, def_roll))
        dmg = 0;
    return dmg;
}

/* check if overhead prayer blocks the given attack style.
   uses int values directly: ATTACK_STYLE_MELEE(1) matches PRAYER_PROTECT_MELEE(3), etc.
   prayer enum: NONE=0, MAGIC=1, RANGED=2, MELEE=3.
   attack style enum: NONE=0, MELEE=1, RANGED=2, MAGIC=3.
   mapping: melee attack(1)->protect melee(3), ranged(2)->protect ranged(2), magic(3)->protect magic(1). */
static inline int encounter_prayer_correct_for_style(int prayer, int attack_style) {
    return (attack_style == 1 /* ATTACK_STYLE_MELEE */  && prayer == 3 /* PRAYER_PROTECT_MELEE */)  ||
           (attack_style == 2 /* ATTACK_STYLE_RANGED */ && prayer == 2 /* PRAYER_PROTECT_RANGED */) ||
           (attack_style == 3 /* ATTACK_STYLE_MAGIC */  && prayer == 1 /* PRAYER_PROTECT_MAGIC */);
}


/* magic hit delay: floor((1 + distance) / 3) + 1, +1 if attacker is player */
static inline int encounter_magic_hit_delay(int distance, int is_player) {
    return (1 + distance) / 3 + 1 + (is_player ? 1 : 0);
}

/* ranged hit delay: floor((3 + distance) / 6) + 1, +1 if attacker is player */
static inline int encounter_ranged_hit_delay(int distance, int is_player) {
    return (3 + distance) / 6 + 1 + (is_player ? 1 : 0);
}

/* blowpipe hit delay: floor(distance / 6) + 1, +1 if attacker is player.
   blowpipe overrides the generic ranged formula — faster at longer range.
   ref: InfernoTrainer Blowpipe.ts:56-58. */
static inline int encounter_blowpipe_hit_delay(int distance, int is_player) {
    return distance / 6 + 1 + (is_player ? 1 : 0);
}

/* chebyshev distance from point (px,py) to nearest tile of NPC footprint
   at (nx,ny) with given npc_size. accounts for multi-tile NPCs. */
static inline int encounter_dist_to_npc(int px, int py, int nx, int ny, int npc_size) {
    int cx = px < nx ? nx : (px > nx + npc_size - 1 ? nx + npc_size - 1 : px);
    int cy = py < ny ? ny : (py > ny + npc_size - 1 ? ny + npc_size - 1 : py);
    int dx = px - cx; if (dx < 0) dx = -dx;
    int dy = py - cy; if (dy < 0) dy = -dy;
    return dx > dy ? dx : dy;
}

/* fisher-yates shuffle for int arrays. used for spawn position randomization,
   snakeling placement, etc. encounters should use this instead of inlining. */
static inline void encounter_shuffle(int* arr, int n, uint32_t* rng) {
    for (int i = n - 1; i > 0; i--) {
        int j = encounter_rand_int(rng, i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* ======================================================================== */
/* player-side combat primitives                                             */
/*                                                                           */
/* pure math for player effective levels, attack rolls, and max hits.        */
/* ref: .refs/osrs-dps-calc/src/lib/PlayerVsNPCCalc.ts                      */
/*      .refs/osrs-dps-calc/src/lib/BaseCalc.ts:105-110                     */
/* ======================================================================== */

/* player effective level: floor(base * prayer_mult) + style_bonus + 8.
   prayer_mult: 1.0 (none), 1.20 (piety/rigour att), 1.23 (piety/rigour str),
   1.25 (augury). style_bonus: 0 (rapid/autocast), +3 (accurate), +1 (controlled).
   ref: PlayerVsNPCCalc.ts lines 191-208 */
static inline int osrs_player_eff_level(int base_level, float prayer_mult, int style_bonus) {
    return (int)(base_level * prayer_mult) + style_bonus + 8;
}

/* ======================================================================== */
/* stance (FightStyle) → combat modifiers.                                   */
/*                                                                           */
/* single source of truth for "what does this stance do". replaces the raw   */
/* `int style_bonus` that callers used to pass around — that discarded the   */
/* why (rapid? autocast? both 0 but speed differs), and forced each caller   */
/* to re-derive the mapping.                                                 */
/*                                                                           */
/* ref: osrs wiki "Combat Options", .refs/osrs-dps-calc PlayerVsNPCCalc.ts   */
/*      and Equipment.ts:245-270 (rapid speed -1).                           */
/* ======================================================================== */

/* attack level bonus for the stance.
   melee: accurate +3, controlled +1.
   ranged: accurate +3.
   magic powered staff: accurate +3 (wiki) — passed only when style is MAGIC.
   all autocast stances and rapid/longrange give 0. */
static inline int osrs_stance_att_bonus(FightStyle fs, AttackStyle atk) {
    switch (fs) {
        case FIGHT_STYLE_ACCURATE:   return 3;
        case FIGHT_STYLE_CONTROLLED: return atk == ATTACK_STYLE_MELEE ? 1 : 0;
        case FIGHT_STYLE_LONGRANGE:  return atk == ATTACK_STYLE_MAGIC ? 1 : 0;  /* powered staff longrange = +1 magic */
        default:                     return 0;
    }
}

/* strength level bonus (melee only). aggressive +3, controlled +1. */
static inline int osrs_stance_str_bonus(FightStyle fs) {
    switch (fs) {
        case FIGHT_STYLE_AGGRESSIVE: return 3;
        case FIGHT_STYLE_CONTROLLED: return 1;
        default:                     return 0;
    }
}

/* defence level bonus. defensive/longrange +3, controlled +1. */
static inline int osrs_stance_def_bonus(FightStyle fs) {
    switch (fs) {
        case FIGHT_STYLE_DEFENSIVE:
        case FIGHT_STYLE_LONGRANGE:  return 3;
        case FIGHT_STYLE_CONTROLLED: return 1;
        default:                     return 0;
    }
}

/* attack speed modifier (ticks to add to weapon base speed).
   rapid is the only stance that changes speed (-1 tick) per dps-calc
   Equipment.ts:248-249. everything else uses the weapon's base speed. */
static inline int osrs_stance_speed_mod(FightStyle fs) {
    return fs == FIGHT_STYLE_RAPID ? -1 : 0;
}

/* attack range modifier (tiles to add to weapon base range).
   longrange adds +2 tiles (e.g. blowpipe 5 → 7). */
static inline int osrs_stance_range_mod(FightStyle fs) {
    return fs == FIGHT_STYLE_LONGRANGE ? 2 : 0;
}

/* player attack roll: eff_level * (equipment_bonus + 64).
   ref: PlayerVsNPCCalc.ts line 212 */
static inline int osrs_player_att_roll(int eff_level, int equipment_bonus) {
    return eff_level * (equipment_bonus + 64);
}

/* player melee max hit: floor((eff_str * (str_bonus + 64) + 320) / 640).
   ref: BaseCalc.ts:107 trackMaxHitFromEffective */
static inline int osrs_player_melee_max_hit(int eff_str_level, int str_bonus) {
    return (eff_str_level * (str_bonus + 64) + 320) / 640;
}

/* player ranged max hit: same formula as melee, different input stats.
   ref: BaseCalc.ts:107 (same formula, ranged strength bonus instead of melee) */
static inline int osrs_player_ranged_max_hit(int eff_range_level, int ranged_str_bonus) {
    return (eff_range_level * (ranged_str_bonus + 64) + 320) / 640;
}

/* player magic max hit: floor(spell_base_dmg * (100 + magic_dmg_pct) / 100).
   magic_dmg_pct is the total % bonus from gear (e.g. 30 = +30%).
   spell_base_dmg: 30 for ice/blood barrage, floor(magic/3)-6 for trident, etc.
   ref: PlayerVsNPCCalc.ts lines 622-667 */
static inline int osrs_player_magic_max_hit(int spell_base_dmg, int magic_dmg_pct) {
    return spell_base_dmg * (100 + magic_dmg_pct) / 100;
}

/* prayer damage reduction.
   PvE (is_pvp=0): correct overhead prayer blocks 100% of damage → returns 0.
   PvP (is_pvp=1): correct overhead prayer reduces by 40% → returns floor(dmg * 0.6).
   wrong prayer or no prayer: returns damage unchanged.
   ref: osrs wiki "protection prayers", osrs-dps-calc */
static inline int osrs_prayer_reduce_damage(int damage, int prayer, int attack_style, int is_pvp) {
    if (damage <= 0) return 0;
    if (!encounter_prayer_correct_for_style(prayer, attack_style)) return damage;
    if (is_pvp) return (int)(damage * 0.6f);
    return 0;  /* PvE: full block */
}

/* double accuracy roll (osmumten's fang, confliction gauntlets).
   rolls accuracy twice — hit if EITHER roll succeeds.
   effective chance: 1 - (1-p)^2 where p = single roll hit chance.
   closed-form from wiki:
     if att >= def: 1 - (def+2)(2*def+3) / (6*(att+1)^2)
     if att < def:  att*(4*att+5) / (6*(att+1)*(def+1))
   ref: osrs wiki "osmumten's fang", encounter_zulrah.h:782-789 */
static inline float osrs_hit_chance_double(int att_roll, int def_roll) {
    float fa = (float)att_roll, fd = (float)def_roll;
    if (att_roll >= def_roll) {
        float num = (fd + 2.0f) * (2.0f * fd + 3.0f);
        float den = 6.0f * (fa + 1.0f) * (fa + 1.0f);
        return 1.0f - num / den;
    }
    return fa * (4.0f * fa + 5.0f) / (6.0f * (fa + 1.0f) * (fd + 1.0f));
}

/* sum equipment bonuses from a gear loadout using ITEM_DATABASE.
   iterates all slots, sums all offensive + defensive bonuses.
   attack_speed and attack_range come from the weapon slot only.
   ITEM_NONE (255) slots are skipped.
   same data as GearBonuses (osrs_types.h) but with different field naming
   convention (attack_stab vs stab_attack). see osrs_pvp_gear.h adapter. */
typedef struct {
    int attack_stab, attack_slash, attack_crush, attack_magic, attack_ranged;
    int defence_stab, defence_slash, defence_crush, defence_magic, defence_ranged;
    int melee_strength, ranged_strength, magic_damage, prayer;
    int attack_speed, attack_range;
} EquipmentBonuses;

static inline void osrs_sum_equipment_bonuses(const uint8_t loadout[NUM_GEAR_SLOTS],
                                               EquipmentBonuses* out) {
    memset(out, 0, sizeof(*out));
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t idx = loadout[slot];
        if (idx == 255) continue;  /* ITEM_NONE */
        const Item* item = &ITEM_DATABASE[idx];
        out->attack_stab += item->attack_stab;
        out->attack_slash += item->attack_slash;
        out->attack_crush += item->attack_crush;
        out->attack_magic += item->attack_magic;
        out->attack_ranged += item->attack_ranged;
        out->defence_stab += item->defence_stab;
        out->defence_slash += item->defence_slash;
        out->defence_crush += item->defence_crush;
        out->defence_magic += item->defence_magic;
        out->defence_ranged += item->defence_ranged;
        out->melee_strength += item->melee_strength;
        out->ranged_strength += item->ranged_strength;
        out->magic_damage += item->magic_damage;
        out->prayer += item->prayer;
    }
    /* weapon slot determines speed + range */
    uint8_t weapon = loadout[GEAR_SLOT_WEAPON];
    if (weapon != 255) {
        out->attack_speed = ITEM_DATABASE[weapon].attack_speed;
        out->attack_range = ITEM_DATABASE[weapon].attack_range;
    }
}

#endif /* OSRS_COMBAT_H */
