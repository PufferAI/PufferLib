#ifndef FC_REWARD_H
#define FC_REWARD_H

#include "fc_contracts.h"
#include "fc_types.h"

typedef struct {
    float w_damage_dealt;      /* legacy per-hit shaping; active config uses 0 */
    float w_progress;          /* reward per required-work unit removed */
    float negative_progress_multiplier;
    float w_damage_taken;
    float w_npc_kill;
    float w_wave_clear;
    float w_jad_kill;
    float w_cave_complete;
    float w_player_death;
    int scale_player_death_with_progress;
    float player_death_min_scale;
    float w_correct_jad_prayer;
    float w_correct_danger_prayer;
    float w_prayer_lost;
    float w_invalid_action;
    float w_tick_penalty;

    float shape_unnecessary_prayer_penalty;
    float shape_wave_stall_base_penalty;
    float shape_wave_stall_cap;
    float shape_jad_heal_penalty;
    float shape_npc_heal_penalty;
    float shape_no_progress_penalty_1;
    float shape_no_progress_penalty_2;
    float shape_no_progress_penalty_3;
    float shape_no_attack_base_penalty;
    float shape_no_attack_wave_scale;

    int shape_wave_stall_start;
    int shape_wave_stall_ramp_interval;
    int shape_no_progress_start_1;
    int shape_no_progress_start_2;
    int shape_no_progress_start_3;
    int shape_no_attack_start;
} FcRewardParams;

typedef struct {
    int ticks_since_attack;
    int ticks_in_wave;
    float required_work_at_wave_start;
    float cave_progress_prev;
    float last_required_work_remaining;
    float last_current_wave_progress;
    float last_cave_progress;
    float last_progress_delta;
    float last_progress_reward;
    float last_net_required_work_removed;
    int ticks_since_positive_progress;
    int positive_progress_ticks;
    int zero_progress_ticks;
    int negative_progress_ticks;
} FcRewardRuntime;

typedef struct {
    int melee_pressure_npcs;
    int any_threat;
    int tokxil_melee;
    int ketzek_melee;
} FcRewardThreatContext;

typedef struct {
    float raw[FC_REWARD_FEATURES];

    float damage_dealt;
    float progress;
    float damage_taken;
    float npc_kill;
    float wave_clear;
    float jad_kill;
    float cave_complete;
    float player_death;

    float correct_jad_prayer;
    float correct_danger_prayer;
    float prayer_lost;
    float unnecessary_prayer;
    float wave_stall;
    float no_progress;
    float no_attack;
    float jad_heal;
    float npc_heal;

    float invalid_action;
    float tick_penalty;

    float total;
    FcRewardThreatContext threat_ctx;
} FcRewardBreakdown;

/* One slot per named breakdown field, excluding raw inputs and the total. */
typedef enum {
    FC_CH_DAMAGE_DEALT = 0,
    FC_CH_PROGRESS,
    FC_CH_DAMAGE_TAKEN,
    FC_CH_NPC_KILL,
    FC_CH_WAVE_CLEAR,
    FC_CH_JAD_KILL,
    FC_CH_CAVE_COMPLETE,
    FC_CH_PLAYER_DEATH,
    FC_CH_CORRECT_JAD_PRAYER,
    FC_CH_CORRECT_DANGER_PRAYER,
    FC_CH_PRAYER_LOST,
    FC_CH_UNNECESSARY_PRAYER,
    FC_CH_WAVE_STALL,
    FC_CH_NO_PROGRESS,
    FC_CH_NO_ATTACK,
    FC_CH_JAD_HEAL,
    FC_CH_NPC_HEAL,
    FC_CH_INVALID_ACTION,
    FC_CH_TICK_PENALTY,
    FC_CH_COUNT
} FcRwdChannel;

extern const char* const FC_CH_NAMES[FC_CH_COUNT];

void fc_reward_breakdown_channels(const FcRewardBreakdown* breakdown,
                                  float out[FC_CH_COUNT]);
FcRewardParams fc_reward_default_params(void);
void fc_reward_runtime_reset(FcRewardRuntime* runtime);
float fc_reward_player_death_scale(const FcRewardParams* params,
                                   float cave_progress);
float fc_reward_required_work_remaining(const FcState* state);
void fc_reward_sync_progress_state(FcState* state,
                                   const FcRewardRuntime* runtime);
void fc_reward_runtime_begin_episode(FcRewardRuntime* runtime, FcState* state);
FcRewardBreakdown fc_reward_compute_breakdown(
    const FcState* state, const FcRewardParams* params, FcRewardRuntime* runtime);

#endif /* FC_REWARD_H */
