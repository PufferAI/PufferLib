/**
 * @fileoverview osrs_special_attacks.h — weapon special attack dispatch.
 *
 * pure function that resolves a special attack given pre-computed combat stats.
 * encounters call osrs_resolve_spec() and apply the returned SpecResult.
 * osrs_spec_cost() returns spec energy cost by weapon item index (0 = no spec).
 *
 * SHARED FUNCTIONS:
 *   osrs_spec_cost(weapon_idx)            spec energy cost for a weapon
 *   osrs_resolve_spec(weapon, ...)        resolve spec attack, return result
 *   osrs_blowpipe_spec_resolve(...)       blowpipe spec helper
 *
 * ref: .refs/osrs-dps-calc/src/lib/ for multipliers,
 *      .refs/osrs-sdk/src/weapons/ for behavior,
 *      osrs_pvp_combat.h for existing claws/VLS implementations,
 *      encounter_zulrah.h for eye of ayak/MSB/blowpipe specs.
 */

#ifndef OSRS_SPECIAL_ATTACKS_H
#define OSRS_SPECIAL_ATTACKS_H

#include "osrs_combat.h"
#include "osrs_items.h"


#define BLOWPIPE_SPEC_ACC_MULT  2
#define BLOWPIPE_SPEC_DMG_NUM   3   /* 1.5x = 3/2 */
#define BLOWPIPE_SPEC_DMG_DEN   2
#define BLOWPIPE_SPEC_HEAL_PCT  50
#define BLOWPIPE_SPEC_COST      50

/* Blowpipe special helper used by encounter code and focused tests. */
static inline int osrs_blowpipe_spec_resolve(
    int base_att_roll, int base_max_hit,
    int target_def_level, int target_ranged_def_bonus,
    uint32_t* rng_state
) {
    int att_roll = base_att_roll * BLOWPIPE_SPEC_ACC_MULT;
    int def_roll = (target_def_level + 8) * (target_ranged_def_bonus + 64);
    int spec_max = base_max_hit * BLOWPIPE_SPEC_DMG_NUM / BLOWPIPE_SPEC_DMG_DEN;
    if (encounter_rand_float(rng_state) < osrs_hit_chance(att_roll, def_roll))
        return encounter_rand_int(rng_state, spec_max + 1);
    return 0;
}


typedef struct {
    int num_hits;               /* number of hits (1-4) */
    int damage[4];              /* per-hit damage values */
    int total_damage;           /* sum of damage[] */
    int heal;                   /* HP healed (blowpipe, SGS) */
    int def_drain;              /* def levels to drain (DWH=30%, BGS=dmg) */
    int magic_def_drain;        /* magic def bonus drained (eye of ayak) */
    int prayer_restore;         /* prayer points restored (SGS) */
    int freeze_ticks;           /* freeze duration (ZGS) */
    int spec_cost;              /* energy consumed */
    int attack_speed_override;  /* 0 = use weapon speed, >0 = override */
} SpecResult;


static inline int osrs_spec_cost(int weapon_item_idx) {
    switch (weapon_item_idx) {
        /* melee */
        case ITEM_AGS:                  return 50;
        case ITEM_DRAGON_CLAWS:         return 50;
        case ITEM_STATIUS_WARHAMMER:    return 35;
        case ITEM_BGS:                  return 50;
        case ITEM_ZGS:                  return 50;
        case ITEM_SGS:                  return 50;
        case ITEM_ANCIENT_GS:           return 50;
        case ITEM_VESTAS:               return 25;
        case ITEM_VOIDWAKER:            return 50;
        case ITEM_GRANITE_MAUL:         return 50;
        case ITEM_DRAGON_DAGGER:        return 25;
        case ITEM_ELDER_MAUL:           return 50;
        /* ranged */
        case ITEM_TOXIC_BLOWPIPE:       return 50;
        case ITEM_MAGIC_SHORTBOW_I:     return 50;
        case ITEM_DARK_BOW:             return 55;
        case ITEM_ZARYTE_CROSSBOW:      return 75;
        case ITEM_HEAVY_BALLISTA:       return 65;
        case ITEM_MORRIGANS_JAVELIN:    return 50;
        case ITEM_ARMADYL_CROSSBOW:    return 50;
        /* magic */
        case ITEM_VOLATILE_STAFF:       return 55;
        case ITEM_EYE_OF_AYAK:          return 50;
        default:                        return 0;
    }
}
/** Resolve a special attack by weapon item index. */
static inline SpecResult osrs_resolve_spec(
    int weapon_item_idx, int att_roll, int max_hit,
    int def_roll, int target_def_level, uint32_t* rng_state
) {
    SpecResult r = {0, {0, 0, 0, 0}, 0, 0, 0, 0, 0, 0, 0, 0};

    switch (weapon_item_idx) {

    /* AGS: 2x accuracy, 1.375x max hit (godsword 1.1 * 1.25).
       ref: osrs-dps-calc [2,1] acc, [11,10]*[5,4] str */
    case ITEM_AGS: {
        int spec_att = att_roll * 2;
        int spec_max = max_hit * 11 / 8;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* dragon claws: 4-hit cascade at BASE accuracy (no multiplier).
       each successive roll uses a lower total damage range, split into 4 hitsplats.
       ref: osrs-dps-calc src/lib/dists/claws.ts dClawDist()
       generateTotals: low = floor(max * (4-accRoll) / 4), high = max + low - 1
       roll 0: total in [max, 2*max-1], split [total/2, total/4, total/8, total/8+1]
       roll 1: total in [3*max/4, 7*max/4-1], split [total/2, total/4, total/4+1, 0]
       roll 2: total in [max/2, 3*max/2-1], split [total/2, total/2+1, 0, 0]
       roll 3: total in [max/4, 5*max/4-1], split [total+1, 0, 0, 0]
       all miss: 2/3 chance [1,1,0,0], 1/3 chance [0,0,0,0] */
    case ITEM_DRAGON_CLAWS: {
        float hit_chance = osrs_hit_chance(att_roll, def_roll);  /* no acc mult */
        r.spec_cost = 50;
        r.num_hits = 4;

        int roll1 = encounter_rand_float(rng_state) < hit_chance;
        int roll2 = encounter_rand_float(rng_state) < hit_chance;
        int roll3 = encounter_rand_float(rng_state) < hit_chance;
        int roll4 = encounter_rand_float(rng_state) < hit_chance;

        if (roll1) {
            int low = max_hit;
            int high = max_hit + low - 1;
            int total = low + encounter_rand_int(rng_state, high - low + 1);
            r.damage[0] = total / 2;
            r.damage[1] = total / 4;
            r.damage[2] = total / 8;
            r.damage[3] = total / 8 + 1;
        } else if (roll2) {
            int low = max_hit * 3 / 4;
            int high = max_hit + low - 1;
            int total = low + encounter_rand_int(rng_state, high - low + 1);
            r.damage[0] = total / 2;
            r.damage[1] = total / 4;
            r.damage[2] = total / 4 + 1;
            r.damage[3] = 0;
        } else if (roll3) {
            int low = max_hit / 2;
            int high = max_hit + low - 1;
            int total = low + encounter_rand_int(rng_state, high - low + 1);
            r.damage[0] = total / 2;
            r.damage[1] = total / 2 + 1;
            r.damage[2] = 0;
            r.damage[3] = 0;
        } else if (roll4) {
            int low = max_hit / 4;
            int high = max_hit + low - 1;
            int total = low + encounter_rand_int(rng_state, high - low + 1);
            r.damage[0] = total + 1;
            r.damage[1] = 0;
            r.damage[2] = 0;
            r.damage[3] = 0;
        } else {
            /* all 4 rolls miss: 2/3 chance [1,1,0,0], 1/3 chance [0,0,0,0] */
            if (encounter_rand_int(rng_state, 3) < 2) {
                r.damage[0] = 1; r.damage[1] = 1;
            }
            r.damage[2] = 0; r.damage[3] = 0;
        }
        r.total_damage = r.damage[0] + r.damage[1] + r.damage[2] + r.damage[3];
        break;
    }

    /* statius warhammer (LMS): 1.25x accuracy, 1.25x str, 30% def drain on hit.
       ref: osrs-dps-calc PlayerVsNPCCalc.ts [5,4] acc, [5,4] str */
    case ITEM_STATIUS_WARHAMMER: {
        int spec_att = att_roll * 5 / 4;   /* 1.25x */
        int spec_max = max_hit * 5 / 4;   /* 1.25x */
        r.spec_cost = 35;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll)) {
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
            r.def_drain = target_def_level * 30 / 100;  /* 30% of current def */
        }
        r.total_damage = r.damage[0];
        break;
    }

    /* BGS: 2.0x accuracy, 1.21x str (godsword 1.1 * 1.1), drain def by damage.
       cascade drain order: def > str > atk > magic > ranged (encounter applies).
       ref: osrs-dps-calc [2,1] acc, [11,10]^2 str */
    case ITEM_BGS: {
        int spec_att = att_roll * 2;
        int spec_max = max_hit * 121 / 100;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll)) {
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
            r.def_drain = r.damage[0];  /* drain def by damage dealt */
        }
        r.total_damage = r.damage[0];
        break;
    }

    /* ZGS: 2x accuracy, 1.1x str (godsword), 32-tick freeze on hit.
       ref: osrs-dps-calc [2,1] acc, [11,10] str */
    case ITEM_ZGS: {
        int spec_att = att_roll * 2;
        int spec_max = max_hit * 11 / 10;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll)) {
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
            if (r.damage[0] > 0) r.freeze_ticks = 32;
        }
        r.total_damage = r.damage[0];
        break;
    }

    /* SGS: 2.0x accuracy, 1.1x str (godsword), heals floor(dmg/2) HP,
       restores floor(dmg/4) prayer.
       ref: osrs-dps-calc [2,1] acc, [11,10] str */
    case ITEM_SGS: {
        int spec_att = att_roll * 2;
        int spec_max = max_hit * 11 / 10;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
        r.total_damage = r.damage[0];
        r.heal = r.total_damage / 2;
        r.prayer_restore = r.total_damage / 4;
        break;
    }

    /* ancient godsword: 2x accuracy, 1.1x str (godsword).
       blood prison effect: on hit, after 8 ticks deals 25 + heals 25.
       not implemented (dps-calc also marks PARTIALLY_IMPLEMENTED).
       ref: osrs-dps-calc [2,1] acc, [11,10] str */
    case ITEM_ANCIENT_GS: {
        int spec_att = att_roll * 2;
        int spec_max = max_hit * 11 / 10;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* VLS "Feint": 20-120% of (1.2x base max), accuracy vs 25% def roll.
       ref: osrs_pvp_combat.h:928-962 */
    case ITEM_VESTAS: {
        int vls_max = max_hit * 6 / 5;  /* 1.2x */
        int vls_min = max_hit / 5;      /* 0.2x */
        int reduced_def = def_roll / 4;  /* 25% of def roll */
        r.spec_cost = 25;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(att_roll, reduced_def))
            r.damage[0] = vls_min + encounter_rand_int(rng_state, vls_max - vls_min + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* voidwaker: guaranteed magic damage at 50-150% of base max hit, vs 25% def.
       ref: osrs_pvp_combat.h:913-925, osrs wiki "voidwaker" */
    case ITEM_VOIDWAKER: {
        int vw_min = max_hit / 2;       /* 50% */
        int vw_max = max_hit * 3 / 2;   /* 150% */
        int reduced_def = def_roll / 4;  /* 25% of def roll */
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(att_roll, reduced_def))
            r.damage[0] = vw_min + encounter_rand_int(rng_state, vw_max - vw_min + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* granite maul: 1.0x accuracy, 1.0x str, instant (resets attack timer).
       ref: osrs wiki "granite maul" */
    case ITEM_GRANITE_MAUL: {
        r.spec_cost = 50;
        r.num_hits = 1;
        r.attack_speed_override = 1;  /* instant */
        if (encounter_rand_float(rng_state) < osrs_hit_chance(att_roll, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, max_hit + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* dragon dagger: [23,20] = 1.15x accuracy and 1.15x str, 2 independent hits.
       ref: osrs-dps-calc PlayerVsNPCCalc.ts:300 */
    case ITEM_DRAGON_DAGGER: {
        int spec_att = att_roll * 23 / 20;  /* 1.15x per dps-calc */
        int spec_max = max_hit * 23 / 20;   /* 1.15x */
        r.spec_cost = 25;
        r.num_hits = 2;
        for (int i = 0; i < 2; i++) {
            if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
                r.damage[i] = encounter_rand_int(rng_state, spec_max + 1);
        }
        r.total_damage = r.damage[0] + r.damage[1];
        break;
    }

    /* elder maul: 1.25x accuracy, 1.0x str, 35% def drain on hit.
       ref: osrs wiki "elder maul" */
    case ITEM_ELDER_MAUL: {
        int spec_att = att_roll * 5 / 4;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll)) {
            r.damage[0] = encounter_rand_int(rng_state, max_hit + 1);
            r.def_drain = target_def_level * 35 / 100;
        }
        r.total_damage = r.damage[0];
        break;
    }

    /* blowpipe: 2x accuracy, 1.5x max hit, heal 50% of damage.
       ref: osrs-sdk Blowpipe.ts, osrs_combat.h (moved here) */
    case ITEM_TOXIC_BLOWPIPE: {
        int spec_att = att_roll * 2;
        int spec_max = max_hit * 3 / 2;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
        r.total_damage = r.damage[0];
        r.heal = r.total_damage / 2;
        break;
    }

    /* MSB(i) Snapshot: 10/7 accuracy boost (~1.43x), 2 arrows.
       ref: encounter_zulrah.h:1052-1075 */
    case ITEM_MAGIC_SHORTBOW_I: {
        int spec_att = att_roll * 10 / 7;
        r.spec_cost = 50;
        r.num_hits = 2;
        for (int i = 0; i < 2; i++) {
            if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
                r.damage[i] = encounter_rand_int(rng_state, max_hit + 1);
        }
        r.total_damage = r.damage[0] + r.damage[1];
        break;
    }

    /* dark bow: 1.0x accuracy, 1.5x str, 2 arrows, min 8 each (dragon arrows).
       ref: osrs_pvp_combat.h:989-1015 */
    case ITEM_DARK_BOW: {
        int spec_max = max_hit * 3 / 2;
        if (spec_max > 48) spec_max = 48;
        r.spec_cost = 55;
        r.num_hits = 2;
        for (int i = 0; i < 2; i++) {
            if (encounter_rand_float(rng_state) < osrs_hit_chance(att_roll, def_roll)) {
                int dmg = encounter_rand_int(rng_state, spec_max + 1);
                r.damage[i] = dmg < 8 ? 8 : dmg;
            } else {
                r.damage[i] = 8;  /* guaranteed min on miss */
            }
        }
        r.total_damage = r.damage[0] + r.damage[1];
        break;
    }

    /* heavy ballista: 1.25x accuracy, 1.25x str.
       ref: osrs-dps-calc [5,4] acc, [5,4] str */
    case ITEM_HEAVY_BALLISTA: {
        int spec_att = att_roll * 5 / 4;
        int spec_max = max_hit * 5 / 4;
        r.spec_cost = 65;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* zaryte crossbow: 2.0x accuracy, guaranteed enhanced bolt proc.
       the bolt proc system (osrs_bolt_procs.h) handles the actual damage enhancement.
       encounters pass is_zcb_spec=1 to osrs_resolve_bolt_proc() after this spec.
       ref: osrs-dps-calc PlayerVsNPCCalc.ts:580, bolts.ts */
    case ITEM_ZARYTE_CROSSBOW: {
        int spec_att = att_roll * 2;
        r.spec_cost = 75;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, max_hit + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* morrigan's javelin: APPROXIMATE — real mechanic is initial hit + bleed
       (5x initial over 5 ticks). dps-calc doesn't implement this weapon.
       using VLS-like pattern for LMS sim: 20-120% max, vs 25% def roll.
       different LMS variant (item 22636) vs wilderness variant exists. */
    case ITEM_MORRIGANS_JAVELIN: {
        int morr_max = max_hit * 6 / 5;
        int morr_min = max_hit / 5;
        int reduced_def = def_roll / 4;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(att_roll, reduced_def))
            r.damage[0] = morr_min + encounter_rand_int(rng_state, morr_max - morr_min + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* ACB: 2x accuracy, normal damage, no special effect.
       ref: osrs wiki "armadyl crossbow", PvP-only spec (dps-calc marks UNIMPLEMENTED for PvNPC) */
    case ITEM_ARMADYL_CROSSBOW: {
        int spec_att = att_roll * 2;
        r.spec_cost = 50;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, max_hit + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* volatile nightmare staff: 1.5x accuracy, max hit = min(58, 58*floor(magic_lvl/99)+1).
       at 99 magic (our sim): max 58. below 99: max 1 (hard level gate).
       ref: osrs-dps-calc PlayerVsNPCCalc.ts:924-925 */
    case ITEM_VOLATILE_STAFF: {
        int spec_att = att_roll * 3 / 2;
        /* max hit = min(58, 58 * floor(magic_level/99) + 1).
           at 99 magic: min(58, 59) = 58. we assume 99 magic. */
        int vol_max = 58;
        r.spec_cost = 55;
        r.num_hits = 1;
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll))
            r.damage[0] = encounter_rand_int(rng_state, vol_max + 1);
        r.total_damage = r.damage[0];
        break;
    }

    /* eye of ayak Soul Rend: 2x accuracy, 1.3x max hit, drains target magic def.
       ref: encounter_zulrah.h:1098-1122 */
    case ITEM_EYE_OF_AYAK: {
        int spec_att = att_roll * 2;
        int spec_max = max_hit * 13 / 10;
        r.spec_cost = 50;
        r.num_hits = 1;
        r.attack_speed_override = 5;  /* 5-tick, slower than normal */
        if (encounter_rand_float(rng_state) < osrs_hit_chance(spec_att, def_roll)) {
            r.damage[0] = encounter_rand_int(rng_state, spec_max + 1);
            r.magic_def_drain = r.damage[0];  /* drain magic def by damage */
        }
        r.total_damage = r.damage[0];
        break;
    }

    default:
        break;
    }

    return r;
}

#endif /* OSRS_SPECIAL_ATTACKS_H */
