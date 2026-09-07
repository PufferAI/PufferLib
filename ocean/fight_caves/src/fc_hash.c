#include "fc_api.h"

#include <stdint.h>
#include <string.h>

/*
 * Version 2 serializes every FcState field explicitly in the documented order
 * below, including whole-tile, directional movement, and projectile collision
 * maps. Signed integers and floats are represented by 32 bits, then fed
 * least-significant byte first. Arena bytes are fed directly. This order is
 * the canonical contract: never hash struct storage, padding, or pointers.
 */

#define FC_HASH_FNV_OFFSET UINT32_C(0x811c9dc5)
#define FC_HASH_FNV_PRIME UINT32_C(0x01000193)

_Static_assert(sizeof(int) == sizeof(int32_t),
               "canonical state hash requires 32-bit int");
_Static_assert(sizeof(float) == sizeof(uint32_t),
               "canonical state hash requires 32-bit float");

static uint32_t fc_hash_u8(uint32_t hash, uint8_t value) {
    return (hash ^ value) * FC_HASH_FNV_PRIME;
}

static uint32_t fc_hash_u32(uint32_t hash, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        hash = fc_hash_u8(hash, (uint8_t)(value >> shift));
    }
    return hash;
}

static uint32_t fc_hash_i32(uint32_t hash, int value) {
    return fc_hash_u32(hash, (uint32_t)(int32_t)value);
}

static uint32_t fc_hash_f32(uint32_t hash, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return fc_hash_u32(hash, bits);
}

#define FC_HASH_I32(value) hash = fc_hash_i32(hash, (value))
#define FC_HASH_U32(value) hash = fc_hash_u32(hash, (value))
#define FC_HASH_F32(value) hash = fc_hash_f32(hash, (value))

static uint32_t fc_hash_pending_hit(uint32_t hash, const FcPendingHit* hit) {
    FC_HASH_I32(hit->active);
    FC_HASH_I32(hit->damage);
    FC_HASH_I32(hit->ticks_remaining);
    FC_HASH_I32(hit->attack_style);
    FC_HASH_I32(hit->source_npc_idx);
    FC_HASH_I32(hit->prayer_drain);
    FC_HASH_I32(hit->prayer_snapshot);
    FC_HASH_I32(hit->prayer_lock_tick);
    return hash;
}

static uint32_t fc_hash_player(uint32_t hash, const FcPlayer* player) {
    FC_HASH_I32(player->x);
    FC_HASH_I32(player->y);
    FC_HASH_I32(player->current_hp);
    FC_HASH_I32(player->max_hp);
    FC_HASH_I32(player->current_prayer);
    FC_HASH_I32(player->max_prayer);
    FC_HASH_I32(player->prayer);
    FC_HASH_I32(player->prayer_at_tick_start);
    FC_HASH_I32(player->prayer_drain_counter);
    FC_HASH_I32(player->sharks_remaining);
    FC_HASH_I32(player->prayer_doses_remaining);
    FC_HASH_I32(player->attack_timer);
    FC_HASH_I32(player->food_timer);
    FC_HASH_I32(player->potion_timer);
    FC_HASH_I32(player->combo_timer);
    FC_HASH_I32(player->run_energy);
    FC_HASH_I32(player->is_running);
    FC_HASH_I32(player->attack_level);
    FC_HASH_I32(player->strength_level);
    FC_HASH_I32(player->defence_level);
    FC_HASH_I32(player->ranged_level);
    FC_HASH_I32(player->prayer_level);
    FC_HASH_I32(player->magic_level);
    FC_HASH_I32(player->weapon_kind);
    FC_HASH_I32(player->weapon_uses_ammo);
    FC_HASH_I32(player->crystal_piece_mask);
    FC_HASH_I32(player->weapon_speed);
    FC_HASH_I32(player->weapon_range);
    FC_HASH_I32(player->ranged_attack_bonus);
    FC_HASH_I32(player->ranged_strength_bonus);
    FC_HASH_I32(player->defence_stab);
    FC_HASH_I32(player->defence_slash);
    FC_HASH_I32(player->defence_crush);
    FC_HASH_I32(player->defence_magic);
    FC_HASH_I32(player->defence_ranged);
    FC_HASH_I32(player->prayer_bonus);
    FC_HASH_I32(player->ammo_count);
    FC_HASH_I32(player->hp_regen_counter);
    for (int i = 0; i < FC_MAX_ROUTE; ++i) {
        FC_HASH_I32(player->route_x[i]);
        FC_HASH_I32(player->route_y[i]);
    }
    FC_HASH_I32(player->route_len);
    FC_HASH_I32(player->route_idx);
    FC_HASH_F32(player->facing_angle);
    FC_HASH_I32(player->attack_target_idx);
    FC_HASH_I32(player->approach_target);
    FC_HASH_I32(player->approach_target_x);
    FC_HASH_I32(player->approach_target_y);
    FC_HASH_I32(player->approach_target_size);
    for (int i = 0; i < FC_MAX_PENDING_HITS; ++i) {
        hash = fc_hash_pending_hit(hash, &player->pending_hits[i]);
    }
    FC_HASH_I32(player->num_pending_hits);
    FC_HASH_I32(player->damage_taken_this_tick);
    FC_HASH_I32(player->hit_style_this_tick);
    FC_HASH_I32(player->hit_source_npc_type);
    FC_HASH_I32(player->hit_locked_prayer_this_tick);
    FC_HASH_I32(player->hit_blocked_this_tick);
    FC_HASH_I32(player->hit_landed_this_tick);
    FC_HASH_I32(player->food_eaten_this_tick);
    FC_HASH_I32(player->potion_used_this_tick);
    FC_HASH_I32(player->prayer_changed_this_tick);
    FC_HASH_I32(player->total_damage_taken);
    FC_HASH_I32(player->total_food_eaten);
    FC_HASH_I32(player->total_potions_used);
    return hash;
}

static uint32_t fc_hash_npc(uint32_t hash, const FcNpc* npc) {
    FC_HASH_I32(npc->active);
    FC_HASH_I32(npc->npc_type);
    FC_HASH_I32(npc->spawn_index);
    FC_HASH_I32(npc->x);
    FC_HASH_I32(npc->y);
    FC_HASH_I32(npc->size);
    FC_HASH_I32(npc->current_hp);
    FC_HASH_I32(npc->max_hp);
    FC_HASH_I32(npc->is_dead);
    FC_HASH_I32(npc->death_timer);
    FC_HASH_I32(npc->attack_style);
    FC_HASH_I32(npc->attack_timer);
    FC_HASH_I32(npc->attack_speed);
    FC_HASH_I32(npc->attack_range);
    FC_HASH_I32(npc->movement_speed);
    FC_HASH_I32(npc->heal_timer);
    FC_HASH_I32(npc->heal_amount);
    FC_HASH_I32(npc->healer_distracted);
    FC_HASH_I32(npc->heal_target_idx);
    FC_HASH_I32(npc->is_respawned_jad_healer);
    FC_HASH_I32(npc->damage_taken_this_tick);
    FC_HASH_I32(npc->prayer_drain_dealt_this_tick);
    FC_HASH_I32(npc->healing_received_this_tick);
    FC_HASH_I32(npc->healing_given_this_tick);
    FC_HASH_I32(npc->healed_by_mejkot_this_tick);
    FC_HASH_I32(npc->healed_by_hurkot_this_tick);
    FC_HASH_I32(npc->healed_self_this_tick);
    FC_HASH_I32(npc->died_this_tick);
    for (int i = 0; i < FC_MAX_PENDING_HITS; ++i) {
        hash = fc_hash_pending_hit(hash, &npc->pending_hits[i]);
    }
    FC_HASH_I32(npc->num_pending_hits);
    return hash;
}

uint32_t fc_state_hash(const FcState* state) {
    uint32_t hash = FC_HASH_FNV_OFFSET;

    hash = fc_hash_player(hash, &state->player);
    for (int i = 0; i < FC_MAX_NPCS; ++i) {
        hash = fc_hash_npc(hash, &state->npcs[i]);
    }

    FC_HASH_I32(state->active_loadout);
    FC_HASH_I32(state->current_wave);
    FC_HASH_I32(state->rotation_id);
    FC_HASH_I32(state->npcs_remaining);
    FC_HASH_I32(state->total_npcs_killed);
    FC_HASH_I32(state->next_spawn_index);
    FC_HASH_I32(state->tick);
    FC_HASH_I32(state->terminal);
    FC_HASH_U32(state->rng_state);
    FC_HASH_U32(state->rng_seed);
    for (int x = 0; x < FC_ARENA_WIDTH; ++x) {
        for (int y = 0; y < FC_ARENA_HEIGHT; ++y) {
            hash = fc_hash_u8(hash, state->walkable[x][y]);
        }
    }
    for (int x = 0; x < FC_ARENA_WIDTH; ++x) {
        for (int y = 0; y < FC_ARENA_HEIGHT; ++y) {
            hash = fc_hash_u8(hash, state->movement_flags[x][y]);
        }
    }
    for (int x = 0; x < FC_ARENA_WIDTH; ++x) {
        for (int y = 0; y < FC_ARENA_HEIGHT; ++y) {
            hash = fc_hash_u8(hash, state->los_flags[x][y]);
        }
    }

    FC_HASH_I32(state->jad_healers_spawned);
    FC_HASH_I32(state->jad_healer_spawn_generations);

    FC_HASH_I32(state->damage_dealt_this_tick);
    FC_HASH_I32(state->hits_landed_this_tick);
    FC_HASH_I32(state->damage_taken_this_tick);
    FC_HASH_I32(state->prayer_lost_this_tick);
    FC_HASH_I32(state->overhead_prayer_lost_this_tick);
    FC_HASH_I32(state->tz_kih_prayer_drain_this_tick);
    FC_HASH_I32(state->npcs_killed_this_tick);
    FC_HASH_I32(state->respawned_jad_healers_killed_this_tick);
    FC_HASH_I32(state->wave_just_cleared);
    FC_HASH_I32(state->jad_damage_this_tick);
    FC_HASH_I32(state->jad_killed);
    FC_HASH_I32(state->correct_jad_prayer);
    FC_HASH_I32(state->wrong_jad_prayer);
    FC_HASH_I32(state->correct_danger_prayer);
    FC_HASH_I32(state->wrong_danger_prayer);
    FC_HASH_I32(state->attack_attempt_this_tick);
    FC_HASH_I32(state->invalid_action_this_tick);
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; ++i) {
        FC_HASH_I32(state->invalid_action_class_this_tick[i]);
    }
    FC_HASH_I32(state->movement_this_tick);
    FC_HASH_I32(state->idle_this_tick);
    FC_HASH_I32(state->food_used_this_tick);
    FC_HASH_I32(state->prayer_potion_used_this_tick);
    FC_HASH_I32(state->jad_heal_procs_this_tick);
    FC_HASH_I32(state->npc_heal_procs_this_tick);
    FC_HASH_I32(state->npc_heal_amount_this_tick);
    FC_HASH_I32(state->mejkot_heal_amount_this_tick);
    FC_HASH_I32(state->jad_heal_amount_this_tick);

    FC_HASH_F32(state->progress_required_work_start);
    FC_HASH_F32(state->progress_required_work_remaining);
    FC_HASH_F32(state->progress_current_wave_progress);
    FC_HASH_F32(state->progress_cave_progress);
    FC_HASH_I32(state->progress_ticks_since_positive);

    FC_HASH_I32(state->ep_ticks_pray_melee);
    FC_HASH_I32(state->ep_ticks_pray_range);
    FC_HASH_I32(state->ep_ticks_pray_magic);
    FC_HASH_I32(state->ep_correct_blocks);
    FC_HASH_I32(state->ep_wrong_prayer_hits);
    FC_HASH_I32(state->ep_no_prayer_hits);
    FC_HASH_I32(state->ep_damage_blocked);
    FC_HASH_I32(state->ep_prayer_switches);
    FC_HASH_I32(state->ep_pots_used);
    FC_HASH_I32(state->ep_pots_wasted);
    FC_HASH_I32(state->ep_pot_pre_prayer_sum);
    FC_HASH_I32(state->ep_food_eaten);
    FC_HASH_I32(state->ep_food_pre_hp_sum);
    FC_HASH_I32(state->ep_food_overhealed);
    FC_HASH_I32(state->ep_pots_overrestored);
    FC_HASH_I32(state->ep_tokxil_melee_ticks);
    FC_HASH_I32(state->ep_ketzek_melee_ticks);
    FC_HASH_I32(state->ep_attack_ready_ticks);
    FC_HASH_I32(state->ep_attack_attempt_ticks);
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; ++i) {
        FC_HASH_I32(state->ep_invalid_action_classes[i]);
    }
    for (int i = 0; i < NPC_TYPE_COUNT; ++i) {
        FC_HASH_I32(state->ep_damage_to_npc_type[i]);
        FC_HASH_I32(state->ep_resolved_hits_to_npc_type[i]);
        FC_HASH_I32(state->ep_damaging_hits_to_npc_type[i]);
        FC_HASH_I32(state->ep_attack_cycles_to_npc_type[i]);
        FC_HASH_I32(state->ep_target_ticks_by_npc_type[i]);
    }
    FC_HASH_I32(state->ep_target_held_ticks);
    FC_HASH_I32(state->ep_no_target_ticks);
    FC_HASH_I32(state->ep_target_in_range_los_ticks);
    FC_HASH_I32(state->ep_target_out_of_range_or_los_ticks);
    FC_HASH_I32(state->ep_attack_cooldown_wait_ticks);
    FC_HASH_I32(state->ep_ready_but_no_attack_ticks);
    FC_HASH_I32(state->ep_action_move_idle_ticks);
    FC_HASH_I32(state->ep_action_move_walk_ticks);
    FC_HASH_I32(state->ep_action_move_run_ticks);
    FC_HASH_I32(state->ep_action_attack_none_ticks);
    FC_HASH_I32(state->ep_action_attack_target_ticks);
    FC_HASH_I32(state->ep_action_prayer_noop_ticks);
    FC_HASH_I32(state->ep_action_prayer_cmd_ticks);
    FC_HASH_I32(state->ep_reached_wave_63);
    FC_HASH_I32(state->ep_jad_killed);
    FC_HASH_I32(state->wave_start_tick);
    FC_HASH_I32(state->ep_max_wave_ticks);
    FC_HASH_I32(state->ep_max_wave_ticks_wave);
    return hash;
}

#undef FC_HASH_I32
#undef FC_HASH_U32
#undef FC_HASH_F32
