/**
 * @file osrs_pvp_combat.h
 * @brief PvP combat orchestration: spec dispatch, damage application, attack availability.
 *
 * delegates core math to shared modules:
 *   osrs_combat.h           — hit chance, effective levels, max hits, prayer reduction
 *   osrs_special_attacks.h  — osrs_resolve_spec(), osrs_spec_cost()
 *   osrs_damage.h           — osrs_apply_damage_pipeline(), pending hit helpers
 *   osrs_bolt_procs.h       — osrs_resolve_bolt_proc()
 *
 * what stays PvP-specific:
 *   - enum-based spec cost/multiplier tables (used by osrs_pvp_observations.h)
 *   - PvP pending hit queue (PendingHit struct with drain/heal/morr fields)
 *   - combat history tracking for RL observations
 *   - attack availability / action masking
 *   - perform_attack orchestration
 */

#ifndef OSRS_PVP_COMBAT_H
#define OSRS_PVP_COMBAT_H

#include "osrs_types.h"
#include "osrs_combat.h"
#include "osrs_special_attacks.h"
#include "osrs_damage.h"
#include "osrs_bolt_procs.h"
#include "osrs_pvp_gear.h"

static void register_hit_calculated(OsrsEnv* env, int attacker_idx, int defender_idx,
                                     AttackStyle style, int total_damage);

/* maps PvP MeleeSpecWeapon enum → item index for osrs_resolve_spec / osrs_spec_cost.
   used internally by perform_attack and availability checks. */
static inline int pvp_melee_spec_to_item(MeleeSpecWeapon w) {
    switch (w) {
        case MELEE_SPEC_AGS:              return ITEM_AGS;
        case MELEE_SPEC_DRAGON_CLAWS:     return ITEM_DRAGON_CLAWS;
        case MELEE_SPEC_GRANITE_MAUL:     return ITEM_GRANITE_MAUL;
        case MELEE_SPEC_DRAGON_DAGGER:    return ITEM_DRAGON_DAGGER;
        case MELEE_SPEC_VOIDWAKER:        return ITEM_VOIDWAKER;
        case MELEE_SPEC_DWH:              return ITEM_STATIUS_WARHAMMER;
        case MELEE_SPEC_BGS:              return ITEM_BGS;
        case MELEE_SPEC_ZGS:              return ITEM_ZGS;
        case MELEE_SPEC_SGS:              return ITEM_SGS;
        case MELEE_SPEC_ANCIENT_GS:       return ITEM_ANCIENT_GS;
        case MELEE_SPEC_VESTAS:           return ITEM_VESTAS;
        default:                          return ITEM_NONE;
    }
}

static inline int pvp_ranged_spec_to_item(RangedSpecWeapon w) {
    switch (w) {
        case RANGED_SPEC_DARK_BOW:     return ITEM_DARK_BOW;
        case RANGED_SPEC_BALLISTA:     return ITEM_HEAVY_BALLISTA;
        case RANGED_SPEC_ACB:          return ITEM_ARMADYL_CROSSBOW;
        case RANGED_SPEC_ZCB:          return ITEM_ZARYTE_CROSSBOW;
        case RANGED_SPEC_MSB:          return ITEM_MAGIC_SHORTBOW_I;
        case RANGED_SPEC_MORRIGANS:    return ITEM_MORRIGANS_JAVELIN;
        default:                       return ITEM_NONE;
    }
}

static inline int pvp_magic_spec_to_item(MagicSpecWeapon w) {
    switch (w) {
        case MAGIC_SPEC_VOLATILE_STAFF: return ITEM_VOLATILE_STAFF;
        default:                        return ITEM_NONE;
    }
}
static int get_melee_spec_cost(MeleeSpecWeapon weapon) {
    switch (weapon) {
        case MELEE_SPEC_AGS:             return 50;
        case MELEE_SPEC_DRAGON_CLAWS:    return 50;
        case MELEE_SPEC_GRANITE_MAUL:    return 50;
        case MELEE_SPEC_DRAGON_DAGGER:   return 25;
        case MELEE_SPEC_VOIDWAKER:       return 50;
        case MELEE_SPEC_DWH:             return 35;
        case MELEE_SPEC_BGS:             return 50;
        case MELEE_SPEC_ZGS:             return 50;
        case MELEE_SPEC_SGS:             return 50;
        case MELEE_SPEC_ANCIENT_GS:      return 50;
        case MELEE_SPEC_VESTAS:          return 25;
        case MELEE_SPEC_ABYSSAL_DAGGER:  return 50;
        case MELEE_SPEC_DRAGON_LONGSWORD:return 25;
        case MELEE_SPEC_DRAGON_MACE:     return 25;
        case MELEE_SPEC_ABYSSAL_BLUDGEON:return 50;
        default:                         return 50;
    }
}

static int get_ranged_spec_cost(RangedSpecWeapon weapon) {
    switch (weapon) {
        case RANGED_SPEC_DARK_BOW:    return 55;
        case RANGED_SPEC_BALLISTA:    return 65;
        case RANGED_SPEC_ACB:         return 50;
        case RANGED_SPEC_ZCB:         return 75;
        case RANGED_SPEC_DRAGON_KNIFE:return 25;
        case RANGED_SPEC_MSB:         return 50;
        case RANGED_SPEC_MORRIGANS:   return 50;
        default:                      return 50;
    }
}

static int get_magic_spec_cost(MagicSpecWeapon weapon) {
    switch (weapon) {
        case MAGIC_SPEC_VOLATILE_STAFF: return 55;
        default:                        return 50;
    }
}
static float get_melee_spec_str_mult(MeleeSpecWeapon weapon) {
    switch (weapon) {
        case MELEE_SPEC_AGS:             return 1.375f;
        case MELEE_SPEC_DRAGON_CLAWS:    return 1.0f;
        case MELEE_SPEC_GRANITE_MAUL:    return 1.0f;
        case MELEE_SPEC_DRAGON_DAGGER:   return 1.15f;
        case MELEE_SPEC_VOIDWAKER:       return 1.0f;
        case MELEE_SPEC_DWH:             return 1.25f;
        case MELEE_SPEC_BGS:             return 1.21f;
        case MELEE_SPEC_ZGS:             return 1.1f;
        case MELEE_SPEC_SGS:             return 1.1f;
        case MELEE_SPEC_ANCIENT_GS:      return 1.1f;
        case MELEE_SPEC_VESTAS:          return 1.20f;
        case MELEE_SPEC_ABYSSAL_DAGGER:  return 0.85f;
        case MELEE_SPEC_DRAGON_LONGSWORD:return 1.15f;
        case MELEE_SPEC_DRAGON_MACE:     return 1.5f;
        case MELEE_SPEC_ABYSSAL_BLUDGEON:return 1.20f;
        default:                         return 1.0f;
    }
}

static float get_melee_spec_acc_mult(MeleeSpecWeapon weapon) {
    switch (weapon) {
        case MELEE_SPEC_AGS:             return 2.0f;
        case MELEE_SPEC_DRAGON_CLAWS:    return 1.0f;
        case MELEE_SPEC_GRANITE_MAUL:    return 1.0f;
        case MELEE_SPEC_DRAGON_DAGGER:   return 1.15f;
        case MELEE_SPEC_VOIDWAKER:       return 1.0f;
        case MELEE_SPEC_DWH:             return 1.25f;
        case MELEE_SPEC_BGS:             return 2.0f;
        case MELEE_SPEC_ZGS:             return 2.0f;
        case MELEE_SPEC_SGS:             return 2.0f;
        case MELEE_SPEC_ANCIENT_GS:      return 2.0f;
        case MELEE_SPEC_VESTAS:          return 1.0f;
        case MELEE_SPEC_ABYSSAL_DAGGER:  return 1.25f;
        case MELEE_SPEC_DRAGON_LONGSWORD:return 1.25f;
        case MELEE_SPEC_DRAGON_MACE:     return 1.25f;
        case MELEE_SPEC_ABYSSAL_BLUDGEON:return 1.0f;
        default:                         return 1.0f;
    }
}

static float get_ranged_spec_str_mult(RangedSpecWeapon weapon) {
    switch (weapon) {
        case RANGED_SPEC_DARK_BOW:    return 1.5f;
        case RANGED_SPEC_BALLISTA:    return 1.25f;
        case RANGED_SPEC_ACB:         return 1.0f;
        case RANGED_SPEC_ZCB:         return 1.0f;
        case RANGED_SPEC_DRAGON_KNIFE:return 1.0f;
        case RANGED_SPEC_MSB:         return 1.0f;
        case RANGED_SPEC_MORRIGANS:   return 1.0f;
        default:                      return 1.0f;
    }
}
static float get_ranged_spec_acc_mult(RangedSpecWeapon weapon) {
    switch (weapon) {
        case RANGED_SPEC_DARK_BOW:    return 1.0f;
        case RANGED_SPEC_BALLISTA:    return 1.25f;
        case RANGED_SPEC_ACB:         return 2.0f;
        case RANGED_SPEC_ZCB:         return 2.0f;
        case RANGED_SPEC_DRAGON_KNIFE:return 1.0f;
        case RANGED_SPEC_MSB:         return 1.0f;
        case RANGED_SPEC_MORRIGANS:   return 1.0f;
        default:                      return 1.0f;
    }
}

__attribute__((unused))
static float get_magic_spec_acc_mult(MagicSpecWeapon weapon) {
    switch (weapon) {
        case MAGIC_SPEC_VOLATILE_STAFF: return 1.5f;
        default:                        return 1.0f;
    }
}

static inline float get_defence_prayer_mult(Player* p) {
    switch (p->offensive_prayer) {
        case OFFENSIVE_PRAYER_MELEE_LOW:
        case OFFENSIVE_PRAYER_RANGED_LOW:
        case OFFENSIVE_PRAYER_MAGIC_LOW:
            return 1.15f;
        case OFFENSIVE_PRAYER_PIETY:
        case OFFENSIVE_PRAYER_RIGOUR:
        case OFFENSIVE_PRAYER_AUGURY:
            return 1.25f;
        default:
            return 1.0f;
    }
}
// EFFECTIVE LEVEL ADAPTERS (delegate to osrs_player_eff_level)

static int calculate_effective_attack(Player* p, AttackStyle style) {
    int base_level;
    float prayer_mult = 1.0f;

    switch (style) {
        case ATTACK_STYLE_MELEE:
            base_level = p->current_attack;
            if (p->offensive_prayer == OFFENSIVE_PRAYER_PIETY) prayer_mult = 1.20f;
            else if (p->offensive_prayer == OFFENSIVE_PRAYER_MELEE_LOW) prayer_mult = 1.15f;
            break;
        case ATTACK_STYLE_RANGED:
            base_level = p->current_ranged;
            if (p->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) prayer_mult = 1.20f;
            else if (p->offensive_prayer == OFFENSIVE_PRAYER_RANGED_LOW) prayer_mult = 1.15f;
            break;
        case ATTACK_STYLE_MAGIC:
            base_level = p->current_magic;
            if (p->offensive_prayer == OFFENSIVE_PRAYER_AUGURY) prayer_mult = 1.25f;
            else if (p->offensive_prayer == OFFENSIVE_PRAYER_MAGIC_LOW) prayer_mult = 1.15f;
            break;
        default:
            return 0;
    }

    int style_bonus = osrs_stance_att_bonus(p->fight_style, style);

    /* magic uses +9 instead of +8 (invisible +1 for magic attack) */
    if (style == ATTACK_STYLE_MAGIC)
        return osrs_player_eff_level(base_level, prayer_mult, style_bonus) + 1;
    return osrs_player_eff_level(base_level, prayer_mult, style_bonus);
}

static int calculate_effective_strength(Player* p, AttackStyle style) {
    int base_level;
    float prayer_mult = 1.0f;

    switch (style) {
        case ATTACK_STYLE_MELEE:
            base_level = p->current_strength;
            if (p->offensive_prayer == OFFENSIVE_PRAYER_PIETY) prayer_mult = 1.23f;
            else if (p->offensive_prayer == OFFENSIVE_PRAYER_MELEE_LOW) prayer_mult = 1.15f;
            break;
        case ATTACK_STYLE_RANGED:
            base_level = p->current_ranged;
            if (p->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) prayer_mult = 1.23f;
            else if (p->offensive_prayer == OFFENSIVE_PRAYER_RANGED_LOW) prayer_mult = 1.15f;
            break;
        case ATTACK_STYLE_MAGIC:
            base_level = p->current_magic;
            break;
        default:
            return 0;
    }

    /* str bonus only applies to melee (aggressive/controlled); osrs_stance_str_bonus
       returns 0 for any non-melee stance. */
    int style_bonus = (style == ATTACK_STYLE_MELEE) ? osrs_stance_str_bonus(p->fight_style) : 0;

    return osrs_player_eff_level(base_level, prayer_mult, style_bonus);
}

static int calculate_effective_defence(Player* p, AttackStyle incoming_style) {
    int base_level = p->current_defence;
    float prayer_mult = get_defence_prayer_mult(p);
    int style_bonus = osrs_stance_def_bonus(p->fight_style);

    if (incoming_style == ATTACK_STYLE_MAGIC) {
        /* PvP magic defence: floor(magic * prayer * 0.7 + def * prayer * 0.3) + style + 8.
           augury boosts magic component via offensive prayer mult. */
        float magic_prayer_mult = 1.0f;
        if (p->offensive_prayer == OFFENSIVE_PRAYER_AUGURY) magic_prayer_mult = 1.25f;
        else if (p->offensive_prayer == OFFENSIVE_PRAYER_MAGIC_LOW) magic_prayer_mult = 1.15f;
        int magic_level = (int)floorf(p->current_magic * magic_prayer_mult);
        int def_level = (int)floorf(p->current_defence * prayer_mult);
        return (int)(magic_level * 0.7f + def_level * 0.3f) + style_bonus + 8;
    }

    return osrs_player_eff_level(base_level, prayer_mult, style_bonus);
}

static MeleeBonusType get_melee_bonus_type(Player* p) {
    if (p->current_gear == GEAR_SPEC) {
        return MELEE_SPEC_BONUS_TYPES[p->melee_spec_weapon];
    }
    GearBonuses* g = get_slot_gear_bonuses(p);
    MeleeBonusType best = MELEE_BONUS_STAB;
    int best_val = g->stab_attack;
    if (g->slash_attack > best_val) { best = MELEE_BONUS_SLASH; best_val = g->slash_attack; }
    if (g->crush_attack > best_val) { best = MELEE_BONUS_CRUSH; }
    return best;
}

static int get_attack_bonus(Player* p, AttackStyle style) {
    GearBonuses* g = get_slot_gear_bonuses(p);
    switch (style) {
        case ATTACK_STYLE_MELEE: {
            MeleeBonusType bonus = get_melee_bonus_type(p);
            switch (bonus) {
                case MELEE_BONUS_STAB: return g->stab_attack;
                case MELEE_BONUS_SLASH: return g->slash_attack;
                case MELEE_BONUS_CRUSH: return g->crush_attack;
                default: return g->slash_attack;
            }
        }
        case ATTACK_STYLE_RANGED: return g->ranged_attack;
        case ATTACK_STYLE_MAGIC: return g->magic_attack;
        default: return 0;
    }
}

static int get_defence_bonus_for_melee_type(Player* p, MeleeBonusType melee_type) {
    GearBonuses* g = get_slot_gear_bonuses(p);
    switch (melee_type) {
        case MELEE_BONUS_STAB: return g->stab_defence;
        case MELEE_BONUS_SLASH: return g->slash_defence;
        case MELEE_BONUS_CRUSH: return g->crush_defence;
        default: return g->slash_defence;
    }
}

static int get_defence_bonus(Player* defender, AttackStyle style, Player* attacker) {
    GearBonuses* g = get_slot_gear_bonuses(defender);
    switch (style) {
        case ATTACK_STYLE_MELEE: {
            MeleeBonusType bonus = get_melee_bonus_type(attacker);
            return get_defence_bonus_for_melee_type(defender, bonus);
        }
        case ATTACK_STYLE_RANGED: return g->ranged_defence;
        case ATTACK_STYLE_MAGIC: return g->magic_defence;
        default: return 0;
    }
}

static int get_strength_bonus(Player* p, AttackStyle style) {
    GearBonuses* g = get_slot_gear_bonuses(p);
    switch (style) {
        case ATTACK_STYLE_MELEE: return g->melee_strength;
        case ATTACK_STYLE_RANGED: return g->ranged_strength;
        case ATTACK_STYLE_MAGIC: return g->magic_strength;
        default: return 0;
    }
}
// HIT CHANCE AND MAX HIT (delegate to shared formulas)

static float calculate_hit_chance(OsrsEnv* env, Player* attacker, Player* defender,
                                   AttackStyle style, float acc_mult) {
    (void)env;
    int eff_attack = calculate_effective_attack(attacker, style);
    int attack_bonus = get_attack_bonus(attacker, style);
    int attack_roll = (int)(eff_attack * (attack_bonus + 64) * acc_mult);

    int eff_defence = calculate_effective_defence(defender, style);
    int defence_bonus = get_defence_bonus(defender, style, attacker);
    int defence_roll = eff_defence * (defence_bonus + 64);

    return clampf(osrs_hit_chance(attack_roll, defence_roll), 0.0f, 1.0f);
}

static int calculate_max_hit(Player* p, AttackStyle style, float str_mult, int magic_base_hit) {
    int eff_strength = calculate_effective_strength(p, style);
    int strength_bonus = get_strength_bonus(p, style);

    int max_hit;
    if (style == ATTACK_STYLE_MAGIC) {
        int base_damage = magic_base_hit;
        float magic_mult = 1.0f;
        if (p->offensive_prayer == OFFENSIVE_PRAYER_AUGURY) magic_mult = 1.04f;
        max_hit = (int)(base_damage * (1.0f + strength_bonus / 100.0f) * str_mult * magic_mult);
    } else if (style == ATTACK_STYLE_RANGED) {
        max_hit = (int)(osrs_player_ranged_max_hit(eff_strength, strength_bonus) * str_mult);
    } else {
        max_hit = (int)(osrs_player_melee_max_hit(eff_strength, strength_bonus) * str_mult);
    }

    osrs_ensure_player_equipment(p);
    if (p->equipment_effect_profile.dharok_piece_count >= 4 &&
        style == ATTACK_STYLE_MELEE) {
        float hp_ratio = 1.0f - ((float)p->current_hitpoints / p->base_hitpoints);
        max_hit = (int)(max_hit * (1.0f + hp_ratio * hp_ratio));
    }

    return max_hit;
}

static inline int get_ice_freeze_ticks(int current_magic) {
    if (current_magic >= ICE_BARRAGE_LEVEL) return 32;
    if (current_magic >= ICE_BLITZ_LEVEL) return 24;
    if (current_magic >= ICE_BURST_LEVEL) return 16;
    return 8;
}

static inline int get_ice_base_hit(int current_magic) {
    if (current_magic >= ICE_BARRAGE_LEVEL) return ICE_BARRAGE_MAX_HIT;
    if (current_magic >= ICE_BLITZ_LEVEL) return ICE_BLITZ_MAX_HIT;
    if (current_magic >= ICE_BURST_LEVEL) return ICE_BURST_MAX_HIT;
    return ICE_RUSH_MAX_HIT;
}

static inline int get_blood_base_hit(int current_magic) {
    if (current_magic >= BLOOD_BARRAGE_LEVEL) return BLOOD_BARRAGE_MAX_HIT;
    if (current_magic >= BLOOD_BLITZ_LEVEL) return BLOOD_BLITZ_MAX_HIT;
    if (current_magic >= BLOOD_BURST_LEVEL) return BLOOD_BURST_MAX_HIT;
    return BLOOD_RUSH_MAX_HIT;
}

static inline int get_blood_heal_percent(int current_magic) {
    if (current_magic >= BLOOD_BARRAGE_LEVEL) return 25;
    if (current_magic >= BLOOD_BLITZ_LEVEL) return 20;
    if (current_magic >= BLOOD_BURST_LEVEL) return 15;
    return 10;
}

/* PvP-specific: dark bow second arrow and weapon-specific ranged delays.
   standard delays use encounter_magic_hit_delay / encounter_ranged_hit_delay
   from osrs_combat.h. PvP players are always is_player=1 but PvP hit delays
   historically did NOT include the +1 player offset. keep these for PvP compat. */

static inline int pvp_magic_hit_delay(int distance) {
    return 1 + ((1 + distance) / 3);
}

static inline int pvp_ranged_hit_delay(int distance) {
    return 1 + ((3 + distance) / 6);
}

static inline int pvp_ranged_hit_delay_fast(int distance) {
    return 1 + (distance / 6);
}

static inline int pvp_ranged_hit_delay_ballista(int distance) {
    return 2 + ((1 + distance) / 6);
}

static inline int pvp_ranged_hit_delay_dbow_second(int distance) {
    return 1 + ((2 + distance) / 3);
}

static inline int pvp_ranged_hit_delay_for_weapon(int distance, int is_special, RangedSpecWeapon weapon) {
    if (!is_special) return pvp_ranged_hit_delay(distance);
    switch (weapon) {
        case RANGED_SPEC_DRAGON_KNIFE:
        case RANGED_SPEC_MORRIGANS:
            return pvp_ranged_hit_delay_fast(distance);
        case RANGED_SPEC_BALLISTA:
            return pvp_ranged_hit_delay_ballista(distance);
        default:
            return pvp_ranged_hit_delay(distance);
    }
}

static void queue_hit(Player* attacker, Player* defender, int damage,
                     AttackStyle style, int delay, int is_special, int hit_success,
                     int freeze_ticks, int heal_percent, int drain_type, int drain_percent,
                     int flat_heal) {
    if (attacker->num_pending_hits >= MAX_PENDING_HITS) return;

    PendingHit* hit = &attacker->pending_hits[attacker->num_pending_hits++];
    hit->damage = damage;
    hit->ticks_until_hit = delay;
    hit->attack_type = style;
    hit->is_special = is_special;
    hit->hit_success = hit_success;
    hit->freeze_ticks = freeze_ticks;
    hit->heal_percent = heal_percent;
    hit->drain_type = drain_type;
    hit->drain_percent = drain_percent;
    hit->flat_heal = flat_heal;
    hit->is_morr_bleed = 0;
    hit->defender_prayer_at_attack = defender->prayer;

    int actual_damage = osrs_prayer_reduce_damage(damage, defender->prayer, style, 1);
    attacker->last_queued_hit_damage += actual_damage;
}
// DAMAGE APPLICATION (uses osrs_apply_damage_pipeline for core pipeline)

static void apply_damage(OsrsEnv* env, int attacker_idx, int defender_idx,
                         PendingHit* hit) {
    Player* attacker = &env->players[attacker_idx];
    Player* defender = &env->players[defender_idx];

    osrs_ensure_player_equipment(defender);

    /* shared passive pipeline: prayer → elysian → veng → recoil → smite */
    DamageResult dr = osrs_apply_passive_damage_pipeline(
        hit->damage, hit->attack_type,
        hit->defender_prayer_at_attack,
        /* is_pvp */ 1,
        defender->veng_active,
        attacker->prayer == PRAYER_SMITE && !defender->is_lms
        ,
        &defender->equipment_effect_profile,
        &defender->item_effect_state,
        &env->rng_state
    );

    int damage = dr.final_damage;

    /* hit event recording for observations */
    defender->hit_landed_this_tick = 1;
    defender->hit_was_successful = hit->hit_success;
    defender->hit_damage += damage;
    defender->hit_style = hit->attack_type;
    defender->hit_defender_prayer = hit->defender_prayer_at_attack;
    defender->hit_was_on_prayer = dr.prayer_blocked;
    defender->hit_attacker_idx = attacker_idx;
    defender->damage_applied_this_tick = damage;

    /* apply vengeance reflection */
    if (dr.veng_damage > 0) {
        attacker->current_hitpoints -= dr.veng_damage;
        if (attacker->current_hitpoints < 0) attacker->current_hitpoints = 0;
        float reflect_scale = (float)dr.veng_damage / (float)attacker->base_hitpoints;
        attacker->total_damage_received += reflect_scale;
        defender->total_damage_dealt += reflect_scale;
        attacker->damage_received_scale += reflect_scale;
        defender->damage_dealt_scale += reflect_scale;
        defender->veng_active = 0;
    }

    /* apply recoil reflection + charge tracking */
    if (dr.recoil_damage > 0) {
        int recoil = dr.recoil_damage;
        if (defender->equipment_effect_profile.recoil_source == OSRS_RECOIL_SOURCE_RING_OF_RECOIL &&
            recoil > defender->item_effect_state.recoil_charges) {
            recoil = defender->item_effect_state.recoil_charges;
        }
        attacker->current_hitpoints -= recoil;
        if (attacker->current_hitpoints < 0) attacker->current_hitpoints = 0;
        float recoil_scale = (float)recoil / (float)attacker->base_hitpoints;
        attacker->total_damage_received += recoil_scale;
        defender->total_damage_dealt += recoil_scale;
        attacker->damage_received_scale += recoil_scale;
        defender->damage_dealt_scale += recoil_scale;
        osrs_consume_recoil_charges(defender, recoil);
    }

    /* apply damage to defender */
    defender->current_hitpoints -= damage;
    if (defender->current_hitpoints < 0) defender->current_hitpoints = 0;
    float damage_scale = (float)damage / (float)defender->base_hitpoints;
    defender->total_damage_received += damage_scale;
    attacker->total_damage_dealt += damage_scale;
    defender->damage_received_scale += damage_scale;
    attacker->damage_dealt_scale += damage_scale;
    attacker->last_target_health_percent =
        (float)defender->current_hitpoints / (float)defender->base_hitpoints;

    /* PvP-specific hit effects: drain, freeze, heal, morr bleed */
    if (hit->hit_success) {
        if (hit->drain_type == 1 && damage > 0) {
            int drain = (int)(defender->current_defence * hit->drain_percent / 100.0f);
            defender->current_defence = clamp(defender->current_defence - drain, 1, 255);
        } else if (hit->drain_type == 2 && damage > 0) {
            defender->current_defence = clamp(defender->current_defence - damage, 1, 255);
        }

        if (hit->freeze_ticks > 0 && defender->freeze_immunity_ticks == 0 && defender->frozen_ticks == 0) {
            defender->frozen_ticks = hit->freeze_ticks;
            defender->freeze_immunity_ticks = hit->freeze_ticks + 5;
            defender->freeze_applied_this_tick = 1;
        }

        if (hit->heal_percent > 0) {
            int heal = (damage * hit->heal_percent) / 100;
            attacker->current_hitpoints = clamp(attacker->current_hitpoints + heal, 0, attacker->base_hitpoints);
        }
        if (hit->flat_heal > 0) {
            attacker->current_hitpoints = clamp(attacker->current_hitpoints + hit->flat_heal, 0, attacker->base_hitpoints);
        }
    }

    if (hit->is_morr_bleed && hit->hit_success && damage > 0) {
        defender->morr_dot_remaining = damage;
    }

    /* apply smite prayer drain */
    if (dr.smite_drain > 0) {
        defender->current_prayer = clamp(defender->current_prayer - dr.smite_drain, 0, defender->base_prayer);
    }
}

static void process_pending_hits(OsrsEnv* env, int attacker_idx, int defender_idx) {
    Player* attacker = &env->players[attacker_idx];

    for (int i = 0; i < attacker->num_pending_hits; i++) {
        PendingHit* hit = &attacker->pending_hits[i];
        hit->ticks_until_hit--;

        if (hit->ticks_until_hit < 0) {
            apply_damage(env, attacker_idx, defender_idx, hit);

            for (int j = i; j < attacker->num_pending_hits - 1; j++) {
                attacker->pending_hits[j] = attacker->pending_hits[j + 1];
            }
            attacker->num_pending_hits--;
            i--;
        }
    }
}

static inline void push_recent_attack(AttackStyle* buffer, int* index, AttackStyle style) {
    buffer[*index] = style;
    *index = (*index + 1) % HISTORY_SIZE;
}

static inline void push_recent_prayer(AttackStyle* buffer, int* index, OverheadPrayer prayer) {
    AttackStyle style = ATTACK_STYLE_NONE;
    if (prayer == PRAYER_PROTECT_MAGIC) style = ATTACK_STYLE_MAGIC;
    else if (prayer == PRAYER_PROTECT_RANGED) style = ATTACK_STYLE_RANGED;
    else if (prayer == PRAYER_PROTECT_MELEE) style = ATTACK_STYLE_MELEE;
    if (style == ATTACK_STYLE_NONE) return;
    buffer[*index] = style;
    *index = (*index + 1) % HISTORY_SIZE;
}

static inline void push_recent_bool(int* buffer, int* index, int value) {
    buffer[*index] = value ? 1 : 0;
    *index = (*index + 1) % HISTORY_SIZE;
}

static void register_hit_calculated(
    OsrsEnv* env,
    int attacker_idx,
    int defender_idx,
    AttackStyle style,
    int total_damage
) {
    Player* attacker = &env->players[attacker_idx];
    Player* defender = &env->players[defender_idx];
    GearBonuses* atk_gear = get_slot_gear_bonuses(attacker);
    VisibleGearBonuses visible_buf = {
        .magic_attack = atk_gear->magic_attack,
        .magic_strength = atk_gear->magic_strength,
        .ranged_attack = atk_gear->ranged_attack,
        .ranged_strength = atk_gear->ranged_strength,
        .melee_attack = max_int(atk_gear->stab_attack, max_int(atk_gear->slash_attack, atk_gear->crush_attack)),
        .melee_strength = atk_gear->melee_strength,
        .magic_defence = atk_gear->magic_defence,
        .ranged_defence = atk_gear->ranged_defence,
        .melee_defence = max_int(atk_gear->stab_defence, max_int(atk_gear->slash_defence, atk_gear->crush_defence)),
    };
    const VisibleGearBonuses* visible = &visible_buf;

    defender->total_target_hit_count += 1;
    push_recent_attack(defender->recent_target_attack_styles, &defender->recent_target_attack_index, style);

    if (style == ATTACK_STYLE_MAGIC) {
        defender->target_hit_magic_count += 1;
        defender->target_magic_accuracy = visible->magic_attack;
        defender->target_magic_strength = visible->magic_strength;
        defender->target_magic_gear_magic_defence = visible->magic_defence;
        defender->target_magic_gear_ranged_defence = visible->ranged_defence;
        defender->target_magic_gear_melee_defence = visible->melee_defence;
    } else if (style == ATTACK_STYLE_RANGED) {
        defender->target_hit_ranged_count += 1;
        defender->target_ranged_accuracy = visible->ranged_attack;
        defender->target_ranged_strength = visible->ranged_strength;
        defender->target_ranged_gear_magic_defence = visible->magic_defence;
        defender->target_ranged_gear_ranged_defence = visible->ranged_defence;
        defender->target_ranged_gear_melee_defence = visible->melee_defence;
    } else if (style == ATTACK_STYLE_MELEE) {
        defender->target_hit_melee_count += 1;
        if (visible->melee_strength >= defender->target_melee_strength) {
            defender->target_melee_accuracy = visible->melee_attack;
            defender->target_melee_strength = visible->melee_strength;
            defender->target_melee_gear_magic_defence = visible->magic_defence;
            defender->target_melee_gear_ranged_defence = visible->ranged_defence;
            defender->target_melee_gear_melee_defence = visible->melee_defence;
        }
    }

    if (defender->prayer == PRAYER_PROTECT_MAGIC) {
        defender->player_pray_magic_count += 1;
        push_recent_prayer(defender->recent_player_prayer_styles, &defender->recent_player_prayer_index, defender->prayer);
    } else if (defender->prayer == PRAYER_PROTECT_RANGED) {
        defender->player_pray_ranged_count += 1;
        push_recent_prayer(defender->recent_player_prayer_styles, &defender->recent_player_prayer_index, defender->prayer);
    } else if (defender->prayer == PRAYER_PROTECT_MELEE) {
        defender->player_pray_melee_count += 1;
        push_recent_prayer(defender->recent_player_prayer_styles, &defender->recent_player_prayer_index, defender->prayer);
    }

    int defender_prayed_correctly = encounter_prayer_correct_for_style(defender->prayer, style);
    if (!defender_prayed_correctly) {
        defender->target_hit_correct_count += 1;
        push_recent_bool(defender->recent_target_hit_correct, &defender->recent_target_hit_correct_index, 1);
    } else {
        defender->player_prayed_correct = 1;
        push_recent_bool(defender->recent_target_hit_correct, &defender->recent_target_hit_correct_index, 0);
    }

    attacker->attack_was_on_prayer = defender_prayed_correctly;

    push_recent_attack(attacker->recent_player_attack_styles, &attacker->recent_player_attack_index, style);
    if (style == ATTACK_STYLE_MAGIC) attacker->player_hit_magic_count += 1;
    else if (style == ATTACK_STYLE_RANGED) attacker->player_hit_ranged_count += 1;
    else if (style == ATTACK_STYLE_MELEE) attacker->player_hit_melee_count += 1;
    attacker->tick_damage_scale = (float)total_damage / (float)defender->base_hitpoints;
    attacker->total_target_pray_count += 1;

    if (defender->prayer == PRAYER_PROTECT_MAGIC) {
        attacker->target_pray_magic_count += 1;
        push_recent_prayer(attacker->recent_target_prayer_styles, &attacker->recent_target_prayer_index, defender->prayer);
    } else if (defender->prayer == PRAYER_PROTECT_RANGED) {
        attacker->target_pray_ranged_count += 1;
        push_recent_prayer(attacker->recent_target_prayer_styles, &attacker->recent_target_prayer_index, defender->prayer);
    } else if (defender->prayer == PRAYER_PROTECT_MELEE) {
        attacker->target_pray_melee_count += 1;
        push_recent_prayer(attacker->recent_target_prayer_styles, &attacker->recent_target_prayer_index, defender->prayer);
    }

    if (encounter_prayer_correct_for_style(defender->prayer, style)) {
        attacker->target_pray_correct_count += 1;
        attacker->target_prayed_correct = 1;
        push_recent_bool(attacker->recent_target_prayer_correct, &attacker->recent_target_prayer_correct_index, 1);
    } else {
        push_recent_bool(attacker->recent_target_prayer_correct, &attacker->recent_target_prayer_correct_index, 0);
    }
}

static inline int is_attack_available(Player* p) {
    if (ONLY_SWITCH_GEAR_WHEN_ATTACK_SOON && remaining_ticks(p->attack_timer) > 0) return 0;
    return 1;
}

static inline int is_melee_weapon_equipped(Player* p) {
    return get_slot_weapon_attack_style(p) == ATTACK_STYLE_MELEE;
}

static inline int is_ranged_weapon_equipped(Player* p) {
    return get_slot_weapon_attack_style(p) == ATTACK_STYLE_RANGED;
}

static inline int is_melee_spec_weapon_equipped(Player* p) {
    return p->melee_spec_weapon != MELEE_SPEC_NONE;
}

static inline int is_ranged_spec_weapon_equipped(Player* p) {
    return p->ranged_spec_weapon != RANGED_SPEC_NONE;
}

static inline int is_magic_spec_weapon_equipped(Player* p) {
    return p->magic_spec_weapon != MAGIC_SPEC_NONE;
}

static inline int can_cast_ice_spell(Player* p) {
    if (p->is_lunar_spellbook) return 0;
    return p->current_magic >= ICE_RUSH_LEVEL;
}

static inline int can_cast_blood_spell(Player* p) {
    if (p->is_lunar_spellbook) return 0;
    return p->current_magic >= BLOOD_RUSH_LEVEL;
}

static inline int is_ranged_attack_available(Player* p) {
    if (!is_attack_available(p)) return 0;
    return is_ranged_weapon_equipped(p);
}

static inline int can_melee(Player* p, Player* t) {
    return is_in_melee_range(p, t) || can_move(p);
}

static inline int is_melee_attack_available(Player* p, Player* t) {
    if (!is_attack_available(p)) return 0;
    (void)t;
    return is_melee_weapon_equipped(p);
}

static inline int is_melee_spec_two_handed(MeleeSpecWeapon weapon) {
    switch (weapon) {
        case MELEE_SPEC_AGS:
        case MELEE_SPEC_DRAGON_CLAWS:
        case MELEE_SPEC_BGS:
        case MELEE_SPEC_ZGS:
        case MELEE_SPEC_SGS:
        case MELEE_SPEC_ANCIENT_GS:
        case MELEE_SPEC_ABYSSAL_BLUDGEON:
            return 1;
        default:
            return 0;
    }
}

static inline int has_free_inventory_slot(Player* p) {
    int food_slots = p->food_count + p->karambwan_count;
    int max_food_slots = MAXED_FOOD_COUNT + MAXED_KARAMBWAN_COUNT;
    return food_slots < max_food_slots;
}

static inline int can_equip_two_handed_weapon(Player* p) {
    return has_free_inventory_slot(p) || p->equipped[GEAR_SLOT_SHIELD] == ITEM_NONE;
}

static inline int can_spec(Player* p) {
    int cost = get_melee_spec_cost(p->melee_spec_weapon);
    return p->melee_spec_weapon != MELEE_SPEC_NONE && p->special_energy >= cost;
}

static inline int is_granite_maul_attack_available(Player* p) {
    if (p->melee_spec_weapon != MELEE_SPEC_GRANITE_MAUL) return 0;
    return p->special_energy >= get_melee_spec_cost(MELEE_SPEC_GRANITE_MAUL);
}

static inline int is_melee_spec_attack_available(Player* p, Player* t) {
    (void)t;
    if (!is_granite_maul_attack_available(p) && !is_attack_available(p)) return 0;
    if (is_melee_spec_two_handed(p->melee_spec_weapon) && !can_equip_two_handed_weapon(p)) return 0;
    if (!is_melee_weapon_equipped(p) || !is_melee_spec_weapon_equipped(p)) return 0;
    return can_spec(p);
}

static inline int is_ranged_spec_attack_available(Player* p) {
    if (!is_attack_available(p)) return 0;
    if (!is_ranged_attack_available(p)) return 0;
    if (p->ranged_spec_weapon == RANGED_SPEC_NONE) return 0;
    if (!is_ranged_spec_weapon_equipped(p)) return 0;
    return p->special_energy >= get_ranged_spec_cost(p->ranged_spec_weapon);
}

static inline int is_ice_attack_available(Player* p) {
    if (p->is_lunar_spellbook) return 0;
    return can_cast_ice_spell(p) && is_attack_available(p);
}

static inline int is_blood_attack_available(Player* p) {
    if (p->is_lunar_spellbook) return 0;
    return can_cast_blood_spell(p) && is_attack_available(p);
}

static inline int can_toggle_spec(Player* p) {
    if (is_melee_spec_weapon_equipped(p) && p->melee_spec_weapon != MELEE_SPEC_NONE) {
        if (is_melee_spec_two_handed(p->melee_spec_weapon) && !can_equip_two_handed_weapon(p)) return 0;
        return p->special_energy >= get_melee_spec_cost(p->melee_spec_weapon);
    }
    if (is_ranged_spec_weapon_equipped(p) && p->ranged_spec_weapon != RANGED_SPEC_NONE)
        return p->special_energy >= get_ranged_spec_cost(p->ranged_spec_weapon);
    if (is_magic_spec_weapon_equipped(p) && p->magic_spec_weapon != MAGIC_SPEC_NONE)
        return p->special_energy >= get_magic_spec_cost(p->magic_spec_weapon);
    return 0;
}

static inline int is_special_ready(Player* p, AttackStyle style) {
    switch (style) {
        case ATTACK_STYLE_MELEE:
            if (!is_melee_spec_weapon_equipped(p) || p->melee_spec_weapon == MELEE_SPEC_NONE) return 0;
            if (is_melee_spec_two_handed(p->melee_spec_weapon) && !can_equip_two_handed_weapon(p)) return 0;
            return p->special_energy >= get_melee_spec_cost(p->melee_spec_weapon);
        case ATTACK_STYLE_RANGED:
            if (!is_ranged_spec_weapon_equipped(p) || p->ranged_spec_weapon == RANGED_SPEC_NONE) return 0;
            return p->special_energy >= get_ranged_spec_cost(p->ranged_spec_weapon);
        case ATTACK_STYLE_MAGIC:
            if (!is_magic_spec_weapon_equipped(p) || p->magic_spec_weapon == MAGIC_SPEC_NONE) return 0;
            return p->special_energy >= get_magic_spec_cost(p->magic_spec_weapon);
        default:
            return 0;
    }
}

static inline int get_ticks_until_next_hit(Player* p) {
    int min_ticks = -1;
    for (int i = 0; i < p->num_pending_hits; i++) {
        if (min_ticks < 0 || p->pending_hits[i].ticks_until_hit < min_ticks) {
            min_ticks = p->pending_hits[i].ticks_until_hit;
        }
    }
    return min_ticks;
}

typedef enum {
    WEAPON_TYPE_STANDARD = 0,
    WEAPON_TYPE_HALBERD
} WeaponType;

static inline int is_halberd_weapon(MeleeSpecWeapon weapon) {
    (void)weapon;
    return 0;
}

static inline int get_attack_range(Player* p, AttackStyle style) {
    switch (style) {
        case ATTACK_STYLE_MELEE:
            if (is_halberd_weapon(p->melee_spec_weapon)) return 2;
            return 1;
        case ATTACK_STYLE_RANGED:
        case ATTACK_STYLE_MAGIC:
            return get_slot_gear_bonuses(p)->attack_range;
        default:
            return 1;
    }
}
// ATTACK EXECUTION (uses osrs_resolve_spec + osrs_resolve_bolt_proc)

static void perform_attack(OsrsEnv* env, int attacker_idx, int defender_idx,
                           AttackStyle style, int is_special, int magic_type, int distance) {
    Player* attacker = &env->players[attacker_idx];
    Player* defender = &env->players[defender_idx];

    int dx = abs_int(attacker->x - defender->x);
    int dy = abs_int(attacker->y - defender->y);
    attacker->last_attack_dx = dx;
    attacker->last_attack_dy = dy;
    attacker->last_attack_dist = (dx > dy) ? dx : dy;

    if (style == ATTACK_STYLE_MELEE && !is_in_melee_range(attacker, defender)) return;

    float acc_mult = 1.0f;
    float str_mult = 1.0f;
    int spec_cost = 0;
    int was_special_requested = is_special;
    int spec_item_idx = ITEM_NONE;

    if (is_special) {
        switch (style) {
            case ATTACK_STYLE_MELEE: {
                MeleeSpecWeapon weapon = attacker->melee_spec_weapon;
                spec_cost = get_melee_spec_cost(weapon);
                spec_item_idx = pvp_melee_spec_to_item(weapon);
                break;
            }
            case ATTACK_STYLE_RANGED: {
                RangedSpecWeapon weapon = attacker->ranged_spec_weapon;
                spec_cost = get_ranged_spec_cost(weapon);
                spec_item_idx = pvp_ranged_spec_to_item(weapon);
                break;
            }
            case ATTACK_STYLE_MAGIC: {
                MagicSpecWeapon weapon = attacker->magic_spec_weapon;
                spec_cost = get_magic_spec_cost(weapon);
                spec_item_idx = pvp_magic_spec_to_item(weapon);
                break;
            }
            default:
                break;
        }

        if (attacker->special_energy < spec_cost) {
            is_special = 0;
            spec_item_idx = ITEM_NONE;
        } else {
            /* spec energy deducted here, not via encounter_use_spec (PvP manages its own step loop) */
            attacker->special_energy -= spec_cost;
            if (!attacker->spec_regen_active && attacker->special_energy < 100) {
                attacker->spec_regen_active = 1;
                attacker->item_effect_state.special_regen_ticks = 0;
            }
        }
    }

    /* update gear based on attack style */
    if (style == ATTACK_STYLE_MELEE)
        attacker->current_gear = was_special_requested ? GEAR_SPEC : GEAR_MELEE;
    else if (style == ATTACK_STYLE_RANGED)
        attacker->current_gear = GEAR_RANGED;
    else if (style == ATTACK_STYLE_MAGIC)
        attacker->current_gear = GEAR_MAGE;

    /* === SPECIAL ATTACK DISPATCH via osrs_resolve_spec === */
    if (is_special && spec_item_idx != ITEM_NONE) {
        int eff_attack = calculate_effective_attack(attacker, style);
        int attack_bonus = get_attack_bonus(attacker, style);
        int att_roll = eff_attack * (attack_bonus + 64);

        int eff_defence = calculate_effective_defence(defender, style);
        int defence_bonus = get_defence_bonus(defender, style, attacker);
        int def_roll = eff_defence * (defence_bonus + 64);

        int magic_base_hit = 30;
        int max_hit = calculate_max_hit(attacker, style, 1.0f, magic_base_hit);

        SpecResult sr = osrs_resolve_spec(
            spec_item_idx, att_roll, max_hit, def_roll,
            defender->current_defence, &env->rng_state
        );

        /* queue each hit from the SpecResult */
        int total_damage = sr.total_damage;
        int hit_delay;
        if (style == ATTACK_STYLE_MELEE)
            hit_delay = 0;
        else if (style == ATTACK_STYLE_RANGED)
            hit_delay = pvp_ranged_hit_delay_for_weapon(distance, 1, attacker->ranged_spec_weapon);
        else
            hit_delay = pvp_magic_hit_delay(distance);

        /* determine PvP-specific hit effects */
        int drain_type = 0, drain_percent = 0;
        int freeze_ticks = sr.freeze_ticks;
        int heal_percent = 0, flat_heal = 0;

        if (sr.def_drain > 0) {
            if (spec_item_idx == ITEM_BGS) {
                drain_type = 2;  /* drain def by damage dealt */
            } else {
                drain_type = 1;
                drain_percent = (spec_item_idx == ITEM_STATIUS_WARHAMMER) ? 30 :
                                (spec_item_idx == ITEM_ELDER_MAUL) ? 35 : 0;
            }
        }
        if (sr.heal > 0 && spec_item_idx == ITEM_SGS) {
            heal_percent = 50;
        }

        /* voidwaker deals magic damage */
        AttackStyle hit_style = (spec_item_idx == ITEM_VOIDWAKER) ? ATTACK_STYLE_MAGIC : style;

        for (int i = 0; i < sr.num_hits; i++) {
            int this_delay = hit_delay;
            /* dark bow second arrow uses different delay formula */
            if (spec_item_idx == ITEM_DARK_BOW && i == 1)
                this_delay = pvp_ranged_hit_delay_dbow_second(distance);
            queue_hit(attacker, defender, sr.damage[i], hit_style, this_delay, 1,
                      sr.damage[i] > 0, freeze_ticks, heal_percent, drain_type, drain_percent, flat_heal);
        }

        register_hit_calculated(env, attacker_idx, defender_idx, hit_style, total_damage);

        /* ancient godsword blood sacrifice: 25 magic damage at 8 ticks + heal */
        if (spec_item_idx == ITEM_ANCIENT_GS && total_damage > 0) {
            int ags_heal = clamp((int)(defender->base_hitpoints * 0.15f), 0, 15);
            queue_hit(attacker, defender, 25, ATTACK_STYLE_MAGIC, 8, 1, 1, 0, 0, 0, 0, ags_heal);
        }

        /* morrigan's javelin phantom strike bleed */
        if (spec_item_idx == ITEM_MORRIGANS_JAVELIN && total_damage > 0) {
            attacker->pending_hits[attacker->num_pending_hits - 1].is_morr_bleed = 1;
            defender->morr_dot_tick_counter = 3;
        }

        goto post_attack;
    }

    /* === NORMAL ATTACK === */
    {
        /* zuriel's staff passive: 10% increased accuracy on ice spells */
        int has_zuriels = (attacker->equipped[GEAR_SLOT_WEAPON] == ITEM_ZURIELS_STAFF);
        if (has_zuriels && style == ATTACK_STYLE_MAGIC && magic_type == 1)
            acc_mult *= 1.10f;

        float hit_chance = calculate_hit_chance(env, attacker, defender, style, acc_mult);
        int magic_base_hit = 30;
        if (style == ATTACK_STYLE_MAGIC) {
            if (magic_type == 1) magic_base_hit = get_ice_base_hit(attacker->current_magic);
            else if (magic_type == 2) magic_base_hit = get_blood_base_hit(attacker->current_magic);
        }
        int max_hit = calculate_max_hit(attacker, style, str_mult, magic_base_hit);

        int hit_delay;
        if (style == ATTACK_STYLE_MELEE)
            hit_delay = 0;
        else if (style == ATTACK_STYLE_RANGED)
            hit_delay = pvp_ranged_hit_delay(distance);
        else
            hit_delay = pvp_magic_hit_delay(distance);

        int freeze_ticks = 0, heal_percent = 0;

        if (style == ATTACK_STYLE_MAGIC) {
            if (magic_type == 1) {
                freeze_ticks = get_ice_freeze_ticks(attacker->current_magic);
                if (has_zuriels) freeze_ticks = (int)(freeze_ticks * 1.10f);
            } else if (magic_type == 2) {
                heal_percent = get_blood_heal_percent(attacker->current_magic);
                if (has_zuriels) heal_percent = (int)(heal_percent * 1.50f);
            }
        }

        int total_damage = 0;
        int apply_magic_freeze_on_calc = (style == ATTACK_STYLE_MAGIC && magic_type == 1);

        /* bolt proc setup */
        int ammo_item = attacker->equipped[GEAR_SLOT_AMMO];
        int is_crossbow_ranged = (style == ATTACK_STYLE_RANGED && !is_special);

        int hit_count = 1;
        for (int i = 0; i < hit_count; i++) {
            int damage = 0;
            int hit_success = 0;

            if (rand_float(env) < hit_chance) {
                hit_success = 1;
                damage = rand_int(env, max_hit + 1);
            }

            /* bolt proc: resolve after accuracy roll, may override damage */
            if (is_crossbow_ranged) {
                BoltProcResult bp = osrs_resolve_bolt_proc(
                    ammo_item, damage, hit_success, max_hit,
                    attacker->current_ranged,
                    defender->current_hitpoints,
                    0, &env->rng_state
                );
                if (bp.proc_triggered) {
                    damage = bp.modified_damage;
                    hit_success = 1;
                }
            }

            total_damage += damage;

            int queued_freeze_ticks = freeze_ticks;
            if (apply_magic_freeze_on_calc) {
                if (hit_success && defender->freeze_immunity_ticks == 0 && defender->frozen_ticks == 0) {
                    defender->frozen_ticks = freeze_ticks;
                    defender->freeze_immunity_ticks = freeze_ticks + 5;
                    defender->freeze_applied_this_tick = 1;
                    defender->hit_attacker_idx = attacker_idx;
                }
                queued_freeze_ticks = 0;
            }
            queue_hit(attacker, defender, damage, style, hit_delay, is_special,
                      hit_success, queued_freeze_ticks, heal_percent, 0, 0, 0);
        }
        register_hit_calculated(env, attacker_idx, defender_idx, style, total_damage);
    }

post_attack:
    attacker->just_attacked = 1;
    attacker->last_attack_style = (is_special && spec_item_idx == ITEM_VOIDWAKER) ? ATTACK_STYLE_MAGIC : style;
    attacker->attack_style_this_tick = attacker->last_attack_style;
    attacker->magic_type_this_tick = magic_type;
    attacker->used_special_this_tick = is_special;

    int attack_speed = get_slot_gear_bonuses(attacker)->attack_speed;
    int is_instant = (is_special && spec_item_idx == ITEM_GRANITE_MAUL);
    if (!is_instant) {
        attacker->attack_timer = attack_speed - 1;
        attacker->attack_timer_uncapped = attack_speed - 1;
        attacker->has_attack_timer = 1;
    }
}

#endif // OSRS_PVP_COMBAT_H
