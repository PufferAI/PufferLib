#include "fc_episode_summary.h"

#include <string.h>

void fc_episode_summary_build(const FcState* state, int episode_length,
                              FcEpisodeSummary* summary) {
    if (!summary) return;
    memset(summary, 0, sizeof(*summary));
    if (!state) return;

    summary->episode_length = episode_length;
    summary->wave_reached = state->current_wave;
    summary->npcs_slayed = state->total_npcs_killed;
    if (episode_length > 0) {
        summary->prayer_uptime_melee =
            (float)state->ep_ticks_pray_melee / (float)episode_length;
        summary->prayer_uptime_range =
            (float)state->ep_ticks_pray_range / (float)episode_length;
        summary->prayer_uptime_magic =
            (float)state->ep_ticks_pray_magic / (float)episode_length;
    }
    summary->correct_prayer = state->ep_correct_blocks;
    summary->wrong_prayer_hits = state->ep_wrong_prayer_hits;
    summary->no_prayer_hits = state->ep_no_prayer_hits;
    summary->prayer_switches = state->ep_prayer_switches;
    summary->damage_blocked = state->ep_damage_blocked;
    summary->damage_taken = state->player.total_damage_taken;
    if (state->ep_attack_ready_ticks > 0) {
        summary->attack_when_ready_rate =
            (float)state->ep_attack_attempt_ticks /
            (float)state->ep_attack_ready_ticks;
    }
    summary->invalid_move =
        state->ep_invalid_action_classes[FC_INVALID_ACTION_MOVE];
    summary->invalid_attack =
        state->ep_invalid_action_classes[FC_INVALID_ACTION_ATTACK];
    summary->invalid_prayer =
        state->ep_invalid_action_classes[FC_INVALID_ACTION_PRAYER];
    summary->tokxil_melee_ticks = state->ep_tokxil_melee_ticks;
    summary->ketzek_melee_ticks = state->ep_ketzek_melee_ticks;
    summary->max_wave_ticks = state->ep_max_wave_ticks;
    summary->max_wave_ticks_wave = state->ep_max_wave_ticks_wave;
    summary->reached_wave_63 = state->ep_reached_wave_63;
    summary->jad_killed = state->ep_jad_killed;
    summary->player_died = state->terminal == TERMINAL_PLAYER_DEATH;

    memcpy(summary->damage_to_npc_type, state->ep_damage_to_npc_type,
           sizeof(summary->damage_to_npc_type));
    memcpy(summary->resolved_hits_to_npc_type,
           state->ep_resolved_hits_to_npc_type,
           sizeof(summary->resolved_hits_to_npc_type));
    memcpy(summary->damaging_hits_to_npc_type,
           state->ep_damaging_hits_to_npc_type,
           sizeof(summary->damaging_hits_to_npc_type));
    memcpy(summary->attack_cycles_to_npc_type,
           state->ep_attack_cycles_to_npc_type,
           sizeof(summary->attack_cycles_to_npc_type));
    memcpy(summary->target_ticks_by_npc_type,
           state->ep_target_ticks_by_npc_type,
           sizeof(summary->target_ticks_by_npc_type));

    summary->target_held_ticks = state->ep_target_held_ticks;
    summary->no_target_ticks = state->ep_no_target_ticks;
    summary->target_in_range_los_ticks =
        state->ep_target_in_range_los_ticks;
    summary->target_out_of_range_or_los_ticks =
        state->ep_target_out_of_range_or_los_ticks;
    summary->attack_cooldown_wait_ticks =
        state->ep_attack_cooldown_wait_ticks;
    summary->ready_but_no_attack_ticks =
        state->ep_ready_but_no_attack_ticks;
    summary->action_move_idle_ticks = state->ep_action_move_idle_ticks;
    summary->action_move_walk_ticks = state->ep_action_move_walk_ticks;
    summary->action_move_run_ticks = state->ep_action_move_run_ticks;
    summary->action_attack_none_ticks = state->ep_action_attack_none_ticks;
    summary->action_attack_target_ticks =
        state->ep_action_attack_target_ticks;
    summary->action_prayer_noop_ticks = state->ep_action_prayer_noop_ticks;
    summary->action_prayer_cmd_ticks = state->ep_action_prayer_cmd_ticks;
}

const char* fc_episode_npc_metric_name(int npc_type) {
    switch (npc_type) {
        case NPC_NONE:      return "none";
        case NPC_TZ_KIH:    return "tz_kih";
        case NPC_TZ_KEK:    return "tz_kek";
        case NPC_TZ_KEK_SM: return "tz_kek_sm";
        case NPC_TOK_XIL:   return "tok_xil";
        case NPC_YT_MEJKOT: return "yt_mejkot";
        case NPC_KET_ZEK:   return "ket_zek";
        case NPC_TZTOK_JAD: return "tztok_jad";
        case NPC_YT_HURKOT: return "yt_hurkot";
        default:            return "unknown";
    }
}
