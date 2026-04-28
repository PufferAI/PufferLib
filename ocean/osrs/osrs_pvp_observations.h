/**
 * @file osrs_pvp_observations.h
 * @brief Observation generation and action mask computation
 *
 * Generates the observation vector for RL agents (334 features)
 * and computes action masks to prevent invalid actions.
 */

#ifndef OSRS_PVP_OBSERVATIONS_H
#define OSRS_PVP_OBSERVATIONS_H

#include <string.h>
#include "osrs_types.h"
#include "osrs_pvp_gear.h"
#include "osrs_pvp_combat.h"
#include "osrs_encounter.h"  // for ENCOUNTER_OVERHEAD_* / ENCOUNTER_OFFENSIVE_* encoding
#include "osrs_pvp_movement.h"

/**
 * Get relative combat skill level (strength/attack/defence).
 * Normalized to 0-1 range based on max possible boosted level.
 */
static inline float get_relative_level_combat(int current, int base) {
    int max_level = base + (int)floorf(base * 0.15f) + 5;
    return (float)current / (float)max_level;
}

/** Get relative ranged level (10% boost formula). */
static inline float get_relative_level_ranged(int current, int base) {
    int max_level = base + (int)floorf(base * 0.10f) + 4;
    return (float)current / (float)max_level;
}

/** Get relative magic level (no boost available). */
static inline float get_relative_level_magic(int current, int base) {
    return (float)current / (float)base;
}

/**
 * Check if brew/defence boost would be beneficial.
 *
 * @param p Player to check
 * @return 1 if defence not capped or HP not full
 */
static inline int can_use_brew_boost(Player* p) {
    int def_boost = (int)floorf(2.0f + (0.20f * p->base_defence));
    int def_cap = p->is_lms ? p->base_defence : p->base_defence + def_boost;
    if (p->current_defence < def_cap - 1) {
        return 1;
    }
    return p->current_hitpoints <= p->base_hitpoints;
}

/** Check if restore potion would be beneficial.
 * Stats must be drained OR prayer below 90% of base.
 */
static inline int can_restore_stats(Player* p) {
    int stats_drained = p->current_attack < p->base_attack ||
                        p->current_defence < p->base_defence ||
                        p->current_strength < p->base_strength ||
                        p->current_ranged < p->base_ranged ||
                        p->current_magic < p->base_magic;
    int prayer_low = p->current_prayer < (int)(p->base_prayer * 0.9f);
    return stats_drained || prayer_low;
}

/** Check if combat potion boost would be beneficial. */
static inline int can_boost_combat_skills(Player* p) {
    int max_att = (int)floorf(p->base_attack * 0.15f) + 5 + p->base_attack;
    int max_str = (int)floorf(p->base_strength * 0.15f) + 5 + p->base_strength;
    int def_boost = (int)floorf(p->base_defence * 0.15f) + 5;
    int max_def = p->is_lms ? p->base_defence : p->base_defence + def_boost;
    return max_att > p->current_attack + 1 ||
           max_def > p->current_defence + 1 ||
           max_str > p->current_strength + 1;
}

/** Check if ranged potion boost would be beneficial. */
static inline int can_boost_ranged(Player* p) {
    int max_ranged = (int)floorf(p->base_ranged * 0.10f) + 4 + p->base_ranged;
    return max_ranged > p->current_ranged + 1;
}

/** Check if potion type is available (timer + doses). */
static inline int can_use_potion(Player* p, int potion_type) {
    if (remaining_ticks(p->potion_timer) > 0) {
        return 0;
    }
    switch (potion_type) {
        case 1: return p->brew_doses > 0;
        case 2: return p->restore_doses > 0;
        case 3: return p->combat_potion_doses > 0;
        case 4: return p->ranged_potion_doses > 0;
        default: return 0;
    }
}

/** Check if food is available and player not at full HP. */
static inline int can_eat_food(Player* p) {
    if (remaining_ticks(p->food_timer) > 0) {
        return 0;
    }
    if (p->food_count <= 0) {
        return 0;
    }
    return p->current_hitpoints < p->base_hitpoints;
}

/** Check if karambwan is available and player not at full HP. */
static inline int can_eat_karambwan(Player* p) {
    if (remaining_ticks(p->karambwan_timer) > 0) {
        return 0;
    }
    if (p->karambwan_count <= 0) {
        return 0;
    }
    return p->current_hitpoints < p->base_hitpoints;
}

/** Check if target is about to attack (tank gear useful). */
static inline int can_switch_to_tank_gear(Player* target) {
    return remaining_ticks(target->attack_timer) <= 0;
}

/** Check if prayer switch is available (respects timing config). */
static inline int is_protected_prayer_action_available(Player* p, Player* target) {
    if (ONLY_SWITCH_PRAYER_WHEN_ABOUT_TO_ATTACK && remaining_ticks(target->attack_timer) > 0) {
        return 0;
    }
    return p->current_prayer > 0;
}

/** Check if smite prayer is available (not in LMS). */
static inline int is_smite_available(OsrsEnv* env, Player* p) {
    if (!ALLOW_SMITE || env->is_lms) {
        return 0;
    }
    if (remaining_ticks(p->attack_timer) > 0) {
        return 0;
    }
    return p->current_prayer > 0;
}

/** Check if redemption prayer is available (no supplies left). */
static inline int is_redemption_available(OsrsEnv* env, Player* p, Player* target) {
    if (!ALLOW_REDEMPTION || env->is_lms) {
        return 0;
    }
    if (p->food_count > 0 || p->karambwan_count > 0 || p->brew_doses > 0) {
        return 0;
    }
    int ticks_until_hit = get_ticks_until_next_hit(target);
    if (ticks_until_hit < 0 && remaining_ticks(target->attack_timer) > 0) {
        return 0;
    }
    return p->current_prayer > 0;
}

/** Check if melee range is possible (can move or already in range). */
static inline int is_melee_range_possible(Player* p, Player* target) {
    return can_move(p) || can_move(target) || is_in_melee_range(p, target);
}

/** Check if target can cast magic spells. */
static inline int can_target_cast_magic_spells(Player* p) {
    return !p->observed_target_lunar_spellbook;
}

/** Check if movement action is allowed. */
static inline int can_move_action(Player* p) {
    if (!ALLOW_MOVING_IF_CAN_ATTACK && remaining_ticks(p->attack_timer) == 0) {
        return 0;
    }
    return can_move(p);
}

/** Check if moving to adjacent tile is useful. */
static inline int can_move_adjacent(Player* p, Player* target, const CollisionMap* cmap) {
    (void)target;
    int dest_x = 0;
    int dest_y = 0;
    if (!select_closest_adjacent_tile(p, p->last_obs_target_x, p->last_obs_target_y, &dest_x, &dest_y, cmap)) {
        return 0;
    }
    return !(dest_x == p->x && dest_y == p->y);
}

/** Check if moving under target is useful (they're frozen). */
static inline int can_move_under(Player* p, Player* target) {
    int dist = chebyshev_distance(p->x, p->y, p->last_obs_target_x, p->last_obs_target_y);
    return remaining_ticks(target->frozen_ticks) > 0 && dist != 0;
}

/** Check if farcast tile at distance is reachable. */
static inline int can_move_to_farcast(Player* p, Player* target, int distance, const CollisionMap* cmap) {
    (void)target;
    int dest_x = 0;
    int dest_y = 0;
    if (!select_farcast_tile(p, p->last_obs_target_x, p->last_obs_target_y, distance, &dest_x, &dest_y, cmap)) {
        return 0;
    }
    return !(dest_x == p->x && dest_y == p->y);
}

/** Check if moving to diagonal tile is useful. */
static inline int can_move_diagonal(Player* p, Player* target, const CollisionMap* cmap) {
    (void)target;
    int dest_x = 0;
    int dest_y = 0;
    if (!select_closest_diagonal_tile(p, p->last_obs_target_x, p->last_obs_target_y, &dest_x, &dest_y, cmap)) {
        return 0;
    }
    return !(dest_x == p->x && dest_y == p->y);
}
// OBSERVATION NORMALIZATION DIVISORS (matches _OBS_NORM_DIVISORS in osrs_pvp.py)

static void init_obs_norm_divisors(float* d) {
    for (int i = 0; i < SLOT_NUM_OBSERVATIONS; i++) d[i] = 1.0f;

    // Spec energy (0-100)
    d[4] = 100.0f;   // player spec
    d[21] = 100.0f;  // target spec

    // Consumable counts
    d[22] = 10.0f;   // ranged potion doses
    d[23] = 10.0f;   // combat potion doses
    d[24] = 16.0f;   // restore doses
    d[25] = 20.0f;   // brew doses
    d[26] = 15.0f;   // food count (dynamic, range 1-17)
    d[27] = 4.0f;    // karambwan count

    // Frozen/immunity ticks (0-32)
    d[29] = 32.0f;   // player frozen
    d[30] = 32.0f;   // target frozen
    d[31] = 32.0f;   // player freeze immunity
    d[32] = 32.0f;   // target freeze immunity

    // Timers
    d[39] = 6.0f;    // attack timer
    d[40] = 3.0f;    // food timer
    d[41] = 3.0f;    // potion timer
    d[42] = 3.0f;    // karambwan timer
    d[43] = 4.0f;    // attack delay normalized
    d[44] = 6.0f;    // target attack timer
    d[45] = 3.0f;    // target food timer

    // Pending damage ratio (can exceed 1.0 with stacked hits)
    d[46] = 2.0f;

    // Ticks until hit
    d[47] = 6.0f;
    d[48] = 6.0f;

    // Damage scale ratio (clamped to [0.5, 2.0])
    d[65] = 2.0f;

    // Distances
    d[60] = 7.0f;
    d[61] = 7.0f;
    d[62] = 7.0f;

    // Base stats (96-102)
    for (int i = 96; i <= 102; i++) d[i] = 99.0f;

    // Spec weapon hit count
    d[106] = 4.0f;

    // Gear bonuses (119-132): player bonuses + target visible defences
    for (int i = 119; i <= 132; i++) d[i] = 170.0f;
    // attack_speed and attack_range have small values (4-6, 1-10), override
    d[123] = 6.0f;   // attack_speed
    d[124] = 10.0f;  // attack_range

    // Veng cooldowns (139-140)
    d[139] = 50.0f;
    d[140] = 50.0f;
}

static float OBS_NORM_DIVISORS[SLOT_NUM_OBSERVATIONS];
static int _obs_norm_initialized = 0;

static void ensure_obs_norm_initialized(void) {
    if (!_obs_norm_initialized) {
        init_obs_norm_divisors(OBS_NORM_DIVISORS);
        _obs_norm_initialized = 1;
    }
}

/**
 * Write normalized agent 0 observations + action mask to ocean buffer.
 *
 * Output layout: [normalized_obs(334), action_mask_as_float(39)] = 373 floats.
 */
static void ocean_write_obs(OsrsEnv* env) {
    ensure_obs_norm_initialized();
    float* dst = env->ocean_io.agent_obs;
    float* src = env->observations;  // agent 0 obs (internal buffer)

    // Normalize observations
    for (int i = 0; i < SLOT_NUM_OBSERVATIONS; i++) {
        dst[i] = src[i] / OBS_NORM_DIVISORS[i];
    }

    // Append action mask as float (agent 0 only)
    unsigned char* mask = env->action_masks;
    for (int i = 0; i < ACTION_MASK_SIZE; i++) {
        dst[SLOT_NUM_OBSERVATIONS + i] = (float)mask[i];
    }
}

/**
 * Write normalized agent 1 observations + action mask to self-play buffer.
 *
 * Mirrors ocean_write_obs() but reads from agent 1's internal buffer offsets.
 * Only called when ocean_io.agent_obs_p1 is set (self-play enabled).
 */
static void ocean_write_obs_p1(OsrsEnv* env) {
    ensure_obs_norm_initialized();
    float* dst = env->ocean_io.agent_obs_p1;
    float* src = env->observations + SLOT_NUM_OBSERVATIONS;  // agent 1 offset

    for (int i = 0; i < SLOT_NUM_OBSERVATIONS; i++) {
        dst[i] = src[i] / OBS_NORM_DIVISORS[i];
    }

    unsigned char* mask = env->action_masks + ACTION_MASK_SIZE;  // agent 1 mask offset
    for (int i = 0; i < ACTION_MASK_SIZE; i++) {
        dst[SLOT_NUM_OBSERVATIONS + i] = (float)mask[i];
    }
}

/**
 * Generate slot-mode observations with per-slot item stats.
 *
 * Observation layout (190 features):
 *   [0-118]   Core observations (gear/prayer/hp/consumables/timers/combat history/stats)
 *   [119-132] Gear bonuses (player + target visible defences)
 *   [133-149] Game mode flags, ability checks, attack_timer_ready
 *   [150-181] Slot-specific features (weapon/style/prayer/equipped per slot)
 *   [182]     Voidwaker magic damage flag
 *   [183-189] Reward shaping signals
 */
static void generate_slot_observations(OsrsEnv* env, int agent_idx) {
    Player* p = &env->players[agent_idx];
    Player* t = &env->players[1 - agent_idx];

    float* obs = env->observations + agent_idx * SLOT_NUM_OBSERVATIONS;

    // Generate base observations (0-170)
    p->last_obs_target_x = t->x;
    p->last_obs_target_y = t->y;

    obs[0] = (p->visible_gear == GEAR_MELEE) ? 1.0f : 0.0f;
    obs[1] = (p->visible_gear == GEAR_RANGED) ? 1.0f : 0.0f;
    obs[2] = (p->visible_gear == GEAR_MAGE) ? 1.0f : 0.0f;
    obs[3] = (float)p->spec_armed;  /* 1 = spec armed for next attack */
    obs[4] = (float)p->special_energy;

    obs[5] = (p->prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[6] = (p->prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[7] = (p->prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[8] = (p->prayer == PRAYER_SMITE) ? 1.0f : 0.0f;
    obs[9] = (p->prayer == PRAYER_REDEMPTION) ? 1.0f : 0.0f;

    obs[10] = (float)p->current_hitpoints / (float)p->base_hitpoints;
    obs[11] = p->last_target_health_percent;

    // Target last attack style (more reliable than gear type — can't be faked)
    obs[12] = (t->last_attack_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
    obs[13] = (t->last_attack_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[14] = (t->last_attack_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[15] = (t->last_attack_style == ATTACK_STYLE_NONE) ? 1.0f : 0.0f;

    obs[16] = (t->prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[17] = (t->prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[18] = (t->prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[19] = (t->prayer == PRAYER_SMITE) ? 1.0f : 0.0f;
    obs[20] = (t->prayer == PRAYER_REDEMPTION) ? 1.0f : 0.0f;
    obs[21] = (float)t->special_energy;

    // Consumables and stats (22-45)
    obs[22] = (float)p->ranged_potion_doses;
    obs[23] = (float)p->combat_potion_doses;
    obs[24] = (float)p->restore_doses;
    obs[25] = (float)p->brew_doses;
    obs[26] = (float)p->food_count;
    obs[27] = (float)p->karambwan_count;
    obs[28] = (float)p->current_prayer / (float)p->base_prayer;

    obs[29] = (float)remaining_ticks(p->frozen_ticks);
    obs[30] = (float)remaining_ticks(t->frozen_ticks);
    obs[31] = (float)remaining_ticks(p->freeze_immunity_ticks);
    obs[32] = (float)remaining_ticks(t->freeze_immunity_ticks);

    obs[33] = is_in_melee_range(p, t) ? 1.0f : 0.0f;

    obs[34] = get_relative_level_combat(p->current_strength, p->base_strength);
    obs[35] = get_relative_level_combat(p->current_attack, p->base_attack);
    obs[36] = get_relative_level_combat(p->current_defence, p->base_defence);
    obs[37] = get_relative_level_ranged(p->current_ranged, p->base_ranged);
    obs[38] = get_relative_level_magic(p->current_magic, p->base_magic);

    obs[39] = (float)p->attack_timer;
    obs[40] = (float)remaining_ticks(p->food_timer);
    obs[41] = (float)remaining_ticks(p->potion_timer);
    obs[42] = (float)remaining_ticks(p->karambwan_timer);

    int attack_delay = get_attack_timer_uncapped(p) - 1;
    if (attack_delay < -3) attack_delay = -3;
    else if (attack_delay > 0) attack_delay = 0;
    obs[43] = (float)(attack_delay + 3);

    obs[44] = (float)remaining_ticks(t->attack_timer);
    obs[45] = (float)remaining_ticks(t->food_timer);

    // Copy remaining base observations (46-170)
    int pending_damage = 0;
    for (int i = 0; i < p->num_pending_hits; i++) {
        pending_damage += p->pending_hits[i].damage;
    }
    obs[46] = (float)pending_damage / (float)t->base_hitpoints;

    int ticks_until_hit_on_target = get_ticks_until_next_hit(p);
    int ticks_until_hit_on_player = get_ticks_until_next_hit(t);
    obs[47] = (float)ticks_until_hit_on_target;
    obs[48] = (float)ticks_until_hit_on_player;

    obs[49] = p->just_attacked ? 1.0f : 0.0f;
    obs[50] = t->just_attacked ? 1.0f : 0.0f;

    obs[51] = p->tick_damage_scale;
    obs[52] = p->damage_received_scale;
    obs[53] = p->damage_dealt_scale;

    obs[54] = (p->last_attack_style != ATTACK_STYLE_NONE) ? 1.0f : 0.0f;
    obs[55] = p->is_moving ? 1.0f : 0.0f;
    obs[56] = t->is_moving ? 1.0f : 0.0f;

    obs[57] = (agent_idx == env->pid_holder) ? 1.0f : 0.0f;

    obs[58] = (!p->is_lunar_spellbook && p->current_magic >= 94) ? 1.0f : 0.0f;
    obs[59] = (!p->is_lunar_spellbook && p->current_magic >= 92) ? 1.0f : 0.0f;

    int dist = chebyshev_distance(p->x, p->y, t->x, t->y);
    int destination_distance = p->is_moving
        ? chebyshev_distance(p->dest_x, p->dest_y, t->x, t->y) : dist;
    int distance_to_destination = p->is_moving
        ? chebyshev_distance(p->x, p->y, p->dest_x, p->dest_y) : 0;

    if (destination_distance > 7) destination_distance = 7;
    if (distance_to_destination > 7) distance_to_destination = 7;
    if (dist > 7) dist = 7;

    obs[60] = (float)destination_distance;
    obs[61] = (float)distance_to_destination;
    obs[62] = (float)dist;

    obs[63] = p->player_prayed_correct ? 1.0f : 0.0f;
    obs[64] = p->target_prayed_correct ? 1.0f : 0.0f;

    float damage_scale = (p->total_damage_dealt + 1.0f) / (p->total_damage_received + 1.0f);
    obs[65] = clampf(damage_scale, 0.5f, 2.0f);

    // Combat history (66-95) - condensed for slot mode
    obs[66] = confidence_scale(p->total_target_hit_count);
    obs[67] = ratio_or_zero(p->target_hit_melee_count, p->total_target_hit_count);
    obs[68] = ratio_or_zero(p->target_hit_magic_count, p->total_target_hit_count);
    obs[69] = ratio_or_zero(p->target_hit_ranged_count, p->total_target_hit_count);
    obs[70] = ratio_or_zero(p->player_hit_melee_count, p->total_target_pray_count);
    obs[71] = ratio_or_zero(p->player_hit_magic_count, p->total_target_pray_count);
    obs[72] = ratio_or_zero(p->player_hit_ranged_count, p->total_target_pray_count);
    obs[73] = ratio_or_zero(p->target_hit_correct_count, p->total_target_hit_count);
    obs[74] = confidence_scale(p->total_target_pray_count);
    obs[75] = ratio_or_zero(p->target_pray_magic_count, p->total_target_pray_count);
    obs[76] = ratio_or_zero(p->target_pray_ranged_count, p->total_target_pray_count);
    obs[77] = ratio_or_zero(p->target_pray_melee_count, p->total_target_pray_count);
    obs[78] = ratio_or_zero(p->player_pray_magic_count, p->total_target_hit_count);
    obs[79] = ratio_or_zero(p->player_pray_ranged_count, p->total_target_hit_count);
    obs[80] = ratio_or_zero(p->player_pray_melee_count, p->total_target_hit_count);
    obs[81] = ratio_or_zero(p->target_pray_correct_count, p->total_target_pray_count);

    // Recent attack history (82-95)
    int recent_target_hit_melee = 0, recent_target_hit_magic = 0, recent_target_hit_ranged = 0;
    int recent_player_hit_melee = 0, recent_player_hit_magic = 0, recent_player_hit_ranged = 0;
    int recent_target_pray_magic = 0, recent_target_pray_ranged = 0, recent_target_pray_melee = 0;
    int recent_player_pray_magic = 0, recent_player_pray_ranged = 0, recent_player_pray_melee = 0;
    int recent_target_hit_correct = 0, recent_target_pray_correct = 0;

    for (int i = 0; i < HISTORY_SIZE; i++) {
        if (p->recent_target_attack_styles[i] == ATTACK_STYLE_MELEE) recent_target_hit_melee++;
        else if (p->recent_target_attack_styles[i] == ATTACK_STYLE_MAGIC) recent_target_hit_magic++;
        else if (p->recent_target_attack_styles[i] == ATTACK_STYLE_RANGED) recent_target_hit_ranged++;

        if (p->recent_player_attack_styles[i] == ATTACK_STYLE_MELEE) recent_player_hit_melee++;
        else if (p->recent_player_attack_styles[i] == ATTACK_STYLE_MAGIC) recent_player_hit_magic++;
        else if (p->recent_player_attack_styles[i] == ATTACK_STYLE_RANGED) recent_player_hit_ranged++;

        if (p->recent_target_prayer_styles[i] == ATTACK_STYLE_MAGIC) recent_target_pray_magic++;
        else if (p->recent_target_prayer_styles[i] == ATTACK_STYLE_RANGED) recent_target_pray_ranged++;
        else if (p->recent_target_prayer_styles[i] == ATTACK_STYLE_MELEE) recent_target_pray_melee++;

        if (p->recent_player_prayer_styles[i] == ATTACK_STYLE_MAGIC) recent_player_pray_magic++;
        else if (p->recent_player_prayer_styles[i] == ATTACK_STYLE_RANGED) recent_player_pray_ranged++;
        else if (p->recent_player_prayer_styles[i] == ATTACK_STYLE_MELEE) recent_player_pray_melee++;

        if (p->recent_target_hit_correct[i]) recent_target_hit_correct++;
        if (p->recent_target_prayer_correct[i]) recent_target_pray_correct++;
    }

    obs[82] = (float)recent_target_hit_melee / (float)HISTORY_SIZE;
    obs[83] = (float)recent_target_hit_magic / (float)HISTORY_SIZE;
    obs[84] = (float)recent_target_hit_ranged / (float)HISTORY_SIZE;
    obs[85] = (float)recent_player_hit_melee / (float)HISTORY_SIZE;
    obs[86] = (float)recent_player_hit_magic / (float)HISTORY_SIZE;
    obs[87] = (float)recent_player_hit_ranged / (float)HISTORY_SIZE;
    obs[88] = (float)recent_target_hit_correct / (float)HISTORY_SIZE;
    obs[89] = (float)recent_target_pray_magic / (float)HISTORY_SIZE;
    obs[90] = (float)recent_target_pray_ranged / (float)HISTORY_SIZE;
    obs[91] = (float)recent_target_pray_melee / (float)HISTORY_SIZE;
    obs[92] = (float)recent_player_pray_magic / (float)HISTORY_SIZE;
    obs[93] = (float)recent_player_pray_ranged / (float)HISTORY_SIZE;
    obs[94] = (float)recent_player_pray_melee / (float)HISTORY_SIZE;
    obs[95] = (float)recent_target_pray_correct / (float)HISTORY_SIZE;

    // Base stats (96-102)
    obs[96] = (float)p->base_attack;
    obs[97] = (float)p->base_strength;
    obs[98] = (float)p->base_defence;
    obs[99] = (float)p->base_ranged;
    obs[100] = (float)p->base_magic;
    obs[101] = (float)p->base_prayer;
    obs[102] = (float)p->base_hitpoints;

    // Spec weapon info (103-118) - same as base
    int melee_spec_cost = get_melee_spec_cost(p->melee_spec_weapon);
    obs[103] = (p->melee_spec_weapon == MELEE_SPEC_NONE) ? 0.5f : (float)melee_spec_cost / 100.0f;
    obs[104] = get_melee_spec_str_mult(p->melee_spec_weapon);
    obs[105] = get_melee_spec_acc_mult(p->melee_spec_weapon);

    int melee_hit_count = (p->melee_spec_weapon == MELEE_SPEC_DRAGON_CLAWS) ? 4 :
                          (p->melee_spec_weapon == MELEE_SPEC_DRAGON_DAGGER ||
                           p->melee_spec_weapon == MELEE_SPEC_ABYSSAL_DAGGER) ? 2 : 1;
    obs[106] = (float)melee_hit_count;
    obs[107] = (p->melee_spec_weapon == MELEE_SPEC_VOIDWAKER) ? 1.0f : 0.0f;
    obs[108] = (p->melee_spec_weapon == MELEE_SPEC_DWH ||
                p->melee_spec_weapon == MELEE_SPEC_BGS) ? 1.0f : 0.0f;
    obs[109] = (p->melee_spec_weapon == MELEE_SPEC_GRANITE_MAUL) ? 1.0f : 0.0f;

    int ranged_spec_cost = get_ranged_spec_cost(p->ranged_spec_weapon);
    obs[110] = (p->ranged_spec_weapon == RANGED_SPEC_NONE) ? 0.5f : (float)ranged_spec_cost / 100.0f;
    obs[111] = get_ranged_spec_str_mult(p->ranged_spec_weapon);
    obs[112] = get_ranged_spec_acc_mult(p->ranged_spec_weapon);
    obs[113] = p->bolt_proc_damage;
    obs[114] = p->bolt_ignores_defense ? 1.0f : 0.0f;

    obs[115] = (p->magic_spec_weapon != MAGIC_SPEC_NONE) ? 1.0f : 0.0f;
    obs[116] = (p->ranged_spec_weapon != RANGED_SPEC_NONE) ? 1.0f : 0.0f;
    obs[117] = p->has_blood_fury ? 1.0f : 0.0f;
    osrs_ensure_player_equipment(p);
    obs[118] = (p->equipment_effect_profile.dharok_piece_count >= 4) ? 1.0f : 0.0f;

    // Slot-based gear bonuses (119-129) using current equipped items
    GearBonuses* slot_bonuses = get_slot_gear_bonuses(p);
    obs[119] = (float)slot_bonuses->magic_attack;
    obs[120] = (float)slot_bonuses->magic_strength;
    obs[121] = (float)slot_bonuses->ranged_attack;
    obs[122] = (float)slot_bonuses->ranged_strength;
    obs[123] = (float)slot_bonuses->attack_speed;
    obs[124] = (float)slot_bonuses->attack_range;
    obs[125] = (float)slot_bonuses->slash_attack;
    obs[126] = (float)slot_bonuses->melee_strength;
    obs[127] = (float)slot_bonuses->ranged_defence;
    obs[128] = (float)slot_bonuses->magic_defence;
    obs[129] = (float)slot_bonuses->slash_defence;

    // Target visible gear defences (130-132) - computed from actual equipped items
    GearBonuses* target_bonuses = get_slot_gear_bonuses(t);
    obs[130] = (float)target_bonuses->ranged_defence;
    obs[131] = (float)target_bonuses->magic_defence;
    obs[132] = (float)target_bonuses->slash_defence;

    // Game mode flags and ability checks (133-148)
    obs[133] = env->is_lms ? 1.0f : 0.0f;
    obs[134] = env->pvp_runtime.is_pvp_arena ? 1.0f : 0.0f;
    obs[135] = p->veng_active ? 1.0f : 0.0f;
    obs[136] = t->veng_active ? 1.0f : 0.0f;
    obs[137] = p->is_lunar_spellbook ? 1.0f : 0.0f;
    obs[138] = p->observed_target_lunar_spellbook ? 1.0f : 0.0f;
    obs[139] = (float)remaining_ticks(p->veng_cooldown);
    obs[140] = (float)remaining_ticks(t->veng_cooldown);
    obs[141] = is_blood_attack_available(p) ? 1.0f : 0.0f;
    obs[142] = is_ice_attack_available(p) ? 1.0f : 0.0f;
    obs[143] = can_toggle_spec(p) ? 1.0f : 0.0f;
    obs[144] = is_ranged_attack_available(p) ? 1.0f : 0.0f;
    obs[145] = is_ranged_spec_attack_available(p) ? 1.0f : 0.0f;
    obs[146] = is_melee_attack_available(p, t) ? 1.0f : 0.0f;
    obs[147] = is_melee_spec_attack_available(p, t) ? 1.0f : 0.0f;
    obs[148] = (p->brew_doses > 0) ? 0.8f : 0.0f;

    // Attack timer ready boolean (149) - precomputed for agent convenience
    obs[149] = (p->attack_timer <= 0) ? 1.0f : 0.0f;

    // Slot mode specific features (150-181)
    obs[150] = (float)p->equipped[GEAR_SLOT_WEAPON] / 63.0f;  // normalized weapon index
    // actual attack style used THIS TICK (not current weapon)
    obs[151] = (p->attack_style_this_tick == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[152] = (p->attack_style_this_tick == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[153] = (p->attack_style_this_tick == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;

    AttackStyle target_style = get_slot_weapon_attack_style(t);
    obs[154] = (target_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[155] = (target_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[156] = (target_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;

    obs[157] = (p->offensive_prayer == OFFENSIVE_PRAYER_PIETY) ? 1.0f : 0.0f;
    obs[158] = (p->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    obs[159] = (p->offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;

    // Current equipped gear per slot (160-170 = 11 slots)
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        obs[160 + slot] = (float)p->equipped[slot] / 63.0f;
    }

    // Target equipped gear per slot (171-181)
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        obs[171 + slot] = (float)t->equipped[slot] / 63.0f;
    }

    // Per-slot item stats removed: 144 features (8 slots x 18 stats) were redundant
    // with gear bonuses (obs 119-132) and per-slot equipped indices (obs 160-181)

    // Voidwaker magic damage flag (182)
    uint8_t best_mspec = find_best_melee_spec(p);
    obs[182] = (best_mspec == ITEM_VOIDWAKER) ? 1.0f : 0.0f;

    // Reward shaping signals (183-189)
    obs[183] = p->used_special_this_tick ? 1.0f : 0.0f;
    obs[184] = p->ate_food_this_tick ? 1.0f : 0.0f;
    obs[185] = p->ate_karambwan_this_tick ? 1.0f : 0.0f;
    AttackStyle current_weapon_style = get_slot_weapon_attack_style(p);
    obs[186] = (current_weapon_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[187] = (current_weapon_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[188] = (current_weapon_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
    obs[189] = p->ate_brew_this_tick ? 1.0f : 0.0f;
}

/**
 * Compute action masks for loadout-based action space.
 *
 * Writes ACTION_MASK_SIZE (40) bytes: one per action value across all heads.
 * mask[i] = 1 if action is valid, 0 if invalid.
 */
static void compute_action_masks(OsrsEnv* env, int agent_idx) {
    Player* p = &env->players[agent_idx];
    Player* t = &env->players[1 - agent_idx];

    unsigned char* mask = env->action_masks + agent_idx * ACTION_MASK_SIZE;
    int offset = 0;

    // LOADOUT head (9 options: KEEP, MELEE, RANGE, MAGE, TANK, SPEC_MELEE, SPEC_RANGE, SPEC_MAGIC, GMAUL)
    mask[offset + LOADOUT_KEEP] = 1;
    // Non-spec loadouts: mask if already active
    for (int l = LOADOUT_MELEE; l <= LOADOUT_TANK; l++) {
        mask[offset + l] = is_loadout_active(p, l) ? 0 : 1;
    }

    int frozen_no_melee = !can_move(p) && !is_in_melee_range(p, t);

    /* SPEC_MELEE: available if melee spec weapon exists + enough energy.
       no timer check — spec is a toggle (arm now, fires on next attack). */
    uint8_t best_melee_spec = find_best_melee_spec(p);
    int melee_spec_cost = 25;
    if (best_melee_spec == ITEM_AGS || best_melee_spec == ITEM_ANCIENT_GS) melee_spec_cost = 50;
    if (best_melee_spec == ITEM_STATIUS_WARHAMMER) melee_spec_cost = 35;
    mask[offset + LOADOUT_SPEC_MELEE] = (best_melee_spec != ITEM_NONE) &&
        (p->special_energy >= melee_spec_cost) && !frozen_no_melee;

    /* SPEC_RANGE: available if ranged spec weapon exists + enough energy */
    uint8_t best_range_spec = find_best_ranged_spec(p);
    int range_spec_cost = 50;
    mask[offset + LOADOUT_SPEC_RANGE] = (best_range_spec != ITEM_NONE) &&
        (p->special_energy >= range_spec_cost);

    /* SPEC_MAGIC: available if magic spec weapon (volatile) exists + enough energy */
    uint8_t best_magic_spec = find_best_magic_spec(p);
    mask[offset + LOADOUT_SPEC_MAGIC] = (best_magic_spec != ITEM_NONE) &&
        (p->special_energy >= 55);

    // GMAUL: available if granite maul in inventory + enough energy, NO timer requirement (instant)
    mask[offset + LOADOUT_GMAUL] = player_has_gmaul(p) &&
        (p->special_energy >= 50) && !frozen_no_melee;

    // Mask MELEE when frozen and out of melee range
    if (frozen_no_melee) {
        mask[offset + LOADOUT_MELEE] = 0;
    }
    offset += LOADOUT_DIM;

    // COMBAT head (13 options: attacks 1-3, movement 4-12, idle 0)
    int attack_ready = remaining_ticks(p->attack_timer) == 0;
    int current_loadout = get_current_loadout(p);
    int in_mage_loadout = (current_loadout == LOADOUT_MAGE);
    int in_tank_loadout = (current_loadout == LOADOUT_TANK);
    int weapon_style = get_slot_weapon_attack_style(p);
    int melee_reachable = (weapon_style == ATTACK_STYLE_MELEE)
        ? (is_in_melee_range(p, t) || can_move(p))
        : 1;
    int can_move_now = can_move(p);
    mask[offset + ATTACK_NONE] = 1;  // NONE = idle (always valid)
    mask[offset + ATTACK_ATK] = attack_ready && !in_mage_loadout && !in_tank_loadout &&
                                 weapon_style != ATTACK_STYLE_NONE &&
                                 melee_reachable;
    mask[offset + ATTACK_ICE] = attack_ready && can_cast_ice_spell(p);
    mask[offset + ATTACK_BLOOD] = attack_ready && can_cast_blood_spell(p);
    const CollisionMap* cmap = (const CollisionMap*)env->collision_map;
    mask[offset + MOVE_ADJACENT] = can_move_now && can_move_adjacent(p, t, cmap);
    mask[offset + MOVE_UNDER] = can_move_now && can_move_under(p, t);
    mask[offset + MOVE_DIAGONAL] = can_move_now && can_move_diagonal(p, t, cmap);
    mask[offset + MOVE_FARCAST_2] = can_move_now && can_move_to_farcast(p, t, 2, cmap);
    mask[offset + MOVE_FARCAST_3] = can_move_now && can_move_to_farcast(p, t, 3, cmap);
    mask[offset + MOVE_FARCAST_4] = can_move_now && can_move_to_farcast(p, t, 4, cmap);
    mask[offset + MOVE_FARCAST_5] = can_move_now && can_move_to_farcast(p, t, 5, cmap);
    mask[offset + MOVE_FARCAST_6] = can_move_now && can_move_to_farcast(p, t, 6, cmap);
    mask[offset + MOVE_FARCAST_7] = can_move_now && can_move_to_farcast(p, t, 7, cmap);
    offset += COMBAT_DIM;

    // OVERHEAD head — 6 options in new toggle-semantic encoding:
    //   0=no_change, 1-5=toggle_{melee,ranged,magic,smite,redemption}
    // a toggle is valid if pp>0 (activation) OR if it would deactivate the currently-active prayer.
    int has_prayer = p->current_prayer > 0;
    mask[offset + ENCOUNTER_OVERHEAD_NO_CHANGE] = 1;
    mask[offset + ENCOUNTER_OVERHEAD_TOGGLE_MELEE]      = has_prayer || p->prayer == PRAYER_PROTECT_MELEE;
    mask[offset + ENCOUNTER_OVERHEAD_TOGGLE_RANGED]     = has_prayer || p->prayer == PRAYER_PROTECT_RANGED;
    mask[offset + ENCOUNTER_OVERHEAD_TOGGLE_MAGIC]      = has_prayer || p->prayer == PRAYER_PROTECT_MAGIC;
    mask[offset + ENCOUNTER_OVERHEAD_TOGGLE_SMITE]      = (has_prayer || p->prayer == PRAYER_SMITE) && !env->is_lms;
    mask[offset + ENCOUNTER_OVERHEAD_TOGGLE_REDEMPTION] = (has_prayer || p->prayer == PRAYER_REDEMPTION) && !env->is_lms;
    offset += OVERHEAD_DIM;

    // FOOD head (2 options)
    mask[offset + FOOD_NONE] = 1;
    mask[offset + FOOD_EAT] = can_eat_food(p);
    offset += FOOD_DIM;

    // POTION head (5 options)
    mask[offset + POTION_NONE] = 1;
    mask[offset + POTION_BREW] = can_use_potion(p, 1) && can_use_brew_boost(p);
    mask[offset + POTION_RESTORE] = can_use_potion(p, 2) && can_restore_stats(p);
    mask[offset + POTION_COMBAT] = can_use_potion(p, 3) && can_boost_combat_skills(p);
    mask[offset + POTION_RANGED] = can_use_potion(p, 4) && can_boost_ranged(p);
    offset += POTION_DIM;

    // KARAMBWAN head (2 options)
    mask[offset + KARAM_NONE] = 1;
    mask[offset + KARAM_EAT] = can_eat_karambwan(p);
    offset += KARAMBWAN_DIM;

    // VENG head (2 options)
    mask[offset + VENG_NONE] = 1;
    mask[offset + VENG_CAST] = !env->is_lms && p->is_lunar_spellbook && !p->veng_active &&
                                (remaining_ticks(p->veng_cooldown) == 0) && p->current_magic >= 94;
    offset += VENG_DIM;

    // OFFENSIVE head — 4 options, new toggle-semantic encoding:
    //   0=no_change, 1-3=toggle_{piety,rigour,augury}
    // valid if pp>0 (activation) OR toggle would deactivate currently-active offensive.
    mask[offset + ENCOUNTER_OFFENSIVE_NO_CHANGE] = 1;
    mask[offset + ENCOUNTER_OFFENSIVE_TOGGLE_PIETY]  = has_prayer || p->offensive_prayer == OFFENSIVE_PRAYER_PIETY;
    mask[offset + ENCOUNTER_OFFENSIVE_TOGGLE_RIGOUR] = has_prayer || p->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR;
    mask[offset + ENCOUNTER_OFFENSIVE_TOGGLE_AUGURY] = has_prayer || p->offensive_prayer == OFFENSIVE_PRAYER_AUGURY;
    offset += OFFENSIVE_DIM;
}

#endif // OSRS_PVP_OBSERVATIONS_H
