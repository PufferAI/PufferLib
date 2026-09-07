#ifndef FC_EPISODE_SUMMARY_H
#define FC_EPISODE_SUMMARY_H

#include "fc_types.h"

/* Read-only, consumer-neutral episode metrics derived from FcState. Training
 * may aggregate these values and evaluators may serialize them, but neither
 * consumer should independently reproduce their formulas. */
typedef struct {
    int episode_length;
    int wave_reached;
    int npcs_slayed;
    float prayer_uptime_melee;
    float prayer_uptime_range;
    float prayer_uptime_magic;
    int correct_prayer;
    int wrong_prayer_hits;
    int no_prayer_hits;
    int prayer_switches;
    int damage_blocked;
    int damage_taken;
    float attack_when_ready_rate;
    int invalid_move;
    int invalid_attack;
    int invalid_prayer;
    int tokxil_melee_ticks;
    int ketzek_melee_ticks;
    int max_wave_ticks;
    int max_wave_ticks_wave;
    int reached_wave_63;
    int jad_killed;
    int player_died;
    int damage_to_npc_type[NPC_TYPE_COUNT];
    int resolved_hits_to_npc_type[NPC_TYPE_COUNT];
    int damaging_hits_to_npc_type[NPC_TYPE_COUNT];
    int attack_cycles_to_npc_type[NPC_TYPE_COUNT];
    int target_ticks_by_npc_type[NPC_TYPE_COUNT];
    int target_held_ticks;
    int no_target_ticks;
    int target_in_range_los_ticks;
    int target_out_of_range_or_los_ticks;
    int attack_cooldown_wait_ticks;
    int ready_but_no_attack_ticks;
    int action_move_idle_ticks;
    int action_move_walk_ticks;
    int action_move_run_ticks;
    int action_attack_none_ticks;
    int action_attack_target_ticks;
    int action_prayer_noop_ticks;
    int action_prayer_cmd_ticks;
} FcEpisodeSummary;

/* episode_length is supplied by the consumer because standalone simulation
 * ticks and adapter step counts can intentionally differ in tests/tools. */
void fc_episode_summary_build(const FcState* state, int episode_length,
                              FcEpisodeSummary* summary);

/* Stable lowercase suffix shared by training and evaluator metric keys. */
const char* fc_episode_npc_metric_name(int npc_type);

#endif /* FC_EPISODE_SUMMARY_H */
