/*
 * binding.c — PufferLib 4.0 binding for Fight Caves environment.
 *
 * Defines the macros PufferLib needs, includes vecenv.h, and implements
 * the my_init/my_log hooks for config parsing and stat logging.
 */

#include "fight_caves.h"
#include "fc_reward.h"

/* Kept in this linked object so the selected Puffer extension exports the
 * same implementation exercised by validation. It is never called per-step. */
#include "contract_dump.c"

#define OBS_SIZE FC_PUFFER_OBS_SIZE
#define OBS_TENSOR_T FloatTensor
#define NUM_ATNS FC_PUFFER_NUM_ATNS
#define ACT_SIZES FC_PUFFER_ACT_SIZES
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE
#define MY_ACTION_MASK FC_PUFFER_MASK_SIZE

#define Env FightCaves
#include "vecenv.h"

static void fc_override_float_config(
        Dict* kwargs, const char* key, float* value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item != NULL) *value = (float)item->value;
}

static void fc_override_int_config(Dict* kwargs, const char* key, int* value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    if (item != NULL) *value = (int)item->value;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;  /* Fight Caves is single-agent */
    env->reward_params = fc_reward_default_params();

    /* Reward shaping weights (from config/fight_caves.ini [env] section) */
    fc_override_float_config(
        kwargs, "w_damage_dealt", &env->reward_params.w_damage_dealt);
    fc_override_float_config(
        kwargs, "w_progress", &env->reward_params.w_progress);
    fc_override_float_config(kwargs, "negative_progress_multiplier",
        &env->reward_params.negative_progress_multiplier);
    fc_override_float_config(
        kwargs, "w_damage_taken", &env->reward_params.w_damage_taken);
    fc_override_float_config(
        kwargs, "w_npc_kill", &env->reward_params.w_npc_kill);
    fc_override_float_config(
        kwargs, "w_wave_clear", &env->reward_params.w_wave_clear);
    fc_override_float_config(
        kwargs, "w_jad_kill", &env->reward_params.w_jad_kill);
    fc_override_float_config(
        kwargs, "w_cave_complete", &env->reward_params.w_cave_complete);
    fc_override_float_config(
        kwargs, "w_player_death", &env->reward_params.w_player_death);
    fc_override_int_config(kwargs, "scale_player_death_with_progress",
        &env->reward_params.scale_player_death_with_progress);
    fc_override_float_config(kwargs, "player_death_min_scale",
        &env->reward_params.player_death_min_scale);
    fc_override_float_config(kwargs, "w_correct_jad_prayer",
        &env->reward_params.w_correct_jad_prayer);
    fc_override_float_config(kwargs, "w_correct_danger_prayer",
        &env->reward_params.w_correct_danger_prayer);
    fc_override_float_config(
        kwargs, "w_prayer_lost", &env->reward_params.w_prayer_lost);
    fc_override_float_config(
        kwargs, "w_invalid_action", &env->reward_params.w_invalid_action);
    fc_override_float_config(
        kwargs, "w_tick_penalty", &env->reward_params.w_tick_penalty);

    /* Configurable shaping terms */
    fc_override_float_config(kwargs, "shape_unnecessary_prayer_penalty",
        &env->reward_params.shape_unnecessary_prayer_penalty);
    fc_override_float_config(kwargs, "shape_wave_stall_base_penalty",
        &env->reward_params.shape_wave_stall_base_penalty);
    fc_override_float_config(kwargs, "shape_wave_stall_cap",
        &env->reward_params.shape_wave_stall_cap);
    fc_override_float_config(kwargs, "shape_jad_heal_penalty",
        &env->reward_params.shape_jad_heal_penalty);
    fc_override_float_config(kwargs, "shape_npc_heal_penalty",
        &env->reward_params.shape_npc_heal_penalty);
    fc_override_float_config(kwargs, "shape_no_progress_penalty_1",
        &env->reward_params.shape_no_progress_penalty_1);
    fc_override_float_config(kwargs, "shape_no_progress_penalty_2",
        &env->reward_params.shape_no_progress_penalty_2);
    fc_override_float_config(kwargs, "shape_no_progress_penalty_3",
        &env->reward_params.shape_no_progress_penalty_3);
    fc_override_float_config(kwargs, "shape_no_attack_base_penalty",
        &env->reward_params.shape_no_attack_base_penalty);
    fc_override_float_config(kwargs, "shape_no_attack_wave_scale",
        &env->reward_params.shape_no_attack_wave_scale);
    fc_override_int_config(kwargs, "shape_wave_stall_start",
        &env->reward_params.shape_wave_stall_start);
    fc_override_int_config(kwargs, "shape_wave_stall_ramp_interval",
        &env->reward_params.shape_wave_stall_ramp_interval);
    fc_override_int_config(kwargs, "shape_no_progress_start_1",
        &env->reward_params.shape_no_progress_start_1);
    fc_override_int_config(kwargs, "shape_no_progress_start_2",
        &env->reward_params.shape_no_progress_start_2);
    fc_override_int_config(kwargs, "shape_no_progress_start_3",
        &env->reward_params.shape_no_progress_start_3);
    fc_override_int_config(kwargs, "shape_no_attack_start",
        &env->reward_params.shape_no_attack_start);

    DictItem* item = dict_get_unsafe(kwargs, "initial_sharks");
    env->initial_sharks = item ? (int)item->value : 0;
    item = dict_get_unsafe(kwargs, "initial_prayer_doses");
    env->initial_prayer_doses = item ? (int)item->value : 0;

    /* Obs ablation flags (default 0 — i.e. no ablation, full obs).
     * See fc_apply_obs_ablation in fc-core/src/fc_state.c for what each zeroes. */
    item = dict_get_unsafe(kwargs, "obs_ablate_npc_distance");
    env->obs_ablate_npc_distance = item ? (int)item->value : 0;
    item = dict_get_unsafe(kwargs, "obs_ablate_incoming_aggregates");
    env->obs_ablate_incoming_aggregates = item ? (int)item->value : 0;
    item = dict_get_unsafe(kwargs, "obs_ablate_npc_valid");
    env->obs_ablate_npc_valid = item ? (int)item->value : 0;

    /* Initialize game state */
    env->seed_counter = 0;
    fc_init(&env->state);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "wave_reached", log->wave_reached);
    dict_set(out, "npcs_slayed", log->npcs_slayed);
    dict_set(out, "prayer_uptime_melee", log->prayer_uptime_melee);
    dict_set(out, "prayer_uptime_range", log->prayer_uptime_range);
    dict_set(out, "prayer_uptime_magic", log->prayer_uptime_magic);
    dict_set(out, "correct_prayer", log->correct_prayer);
    dict_set(out, "wrong_prayer_hits", log->wrong_prayer_hits);
    dict_set(out, "no_prayer_hits", log->no_prayer_hits);
    dict_set(out, "prayer_switches", log->prayer_switches);
    dict_set(out, "damage_blocked", log->damage_blocked);
    dict_set(out, "dmg_taken_avg", log->dmg_taken_avg);
    dict_set(out, "attack_when_ready_rate", log->attack_when_ready_rate);
    dict_set(out, "tokxil_melee_ticks", log->tokxil_melee_ticks);
    dict_set(out, "ketzek_melee_ticks", log->ketzek_melee_ticks);
    dict_set(out, "max_wave_ticks", log->max_wave_ticks);
    dict_set(out, "max_wave_ticks_wave", log->max_wave_ticks_wave);
    dict_set(out, "reached_wave_63", log->reached_wave_63);
    dict_set(out, "jad_kill_rate", log->jad_kill_rate);
    dict_set(out, "player_death_rate", log->player_death_rate);
    dict_set(out, "target_held_ticks", log->target_held_ticks);
    dict_set(out, "no_target_ticks", log->no_target_ticks);
    dict_set(out, "target_in_range_los_ticks", log->target_in_range_los_ticks);
    dict_set(out, "target_out_of_range_or_los_ticks", log->target_out_of_range_or_los_ticks);
    dict_set(out, "attack_cooldown_wait_ticks", log->attack_cooldown_wait_ticks);
    dict_set(out, "ready_but_no_attack_ticks", log->ready_but_no_attack_ticks);
    dict_set(out, "action_move_idle_ticks", log->action_move_idle_ticks);
    dict_set(out, "action_move_walk_ticks", log->action_move_walk_ticks);
    dict_set(out, "action_move_run_ticks", log->action_move_run_ticks);
    dict_set(out, "action_attack_none_ticks", log->action_attack_none_ticks);
    dict_set(out, "action_attack_target_ticks", log->action_attack_target_ticks);
    dict_set(out, "action_prayer_noop_ticks", log->action_prayer_noop_ticks);
    dict_set(out, "action_prayer_cmd_ticks", log->action_prayer_cmd_ticks);
    dict_set(out, "no_progress_ticks", log->no_progress_ticks);
    dict_set(out, "required_work_remaining", log->required_work_remaining);
    dict_set(out, "required_work_start", log->required_work_start);
    dict_set(out, "cave_progress", log->cave_progress);
    dict_set(out, "current_wave_progress", log->current_wave_progress);
    dict_set(out, "progress_delta", log->progress_delta);
    dict_set(out, "progress_reward", log->progress_reward);
    dict_set(out, "ticks_since_positive_progress", log->ticks_since_positive_progress);
    dict_set(out, "positive_progress_ticks", log->positive_progress_ticks);
    dict_set(out, "zero_progress_ticks", log->zero_progress_ticks);
    dict_set(out, "negative_progress_ticks", log->negative_progress_ticks);
    dict_set(out, "gross_damage_dealt", log->gross_damage_dealt);
    dict_set(out, "net_required_work_removed", log->net_required_work_removed);
    dict_set(out, "gross_damage_to_net_progress_ratio", log->gross_damage_to_net_progress_ratio);
    dict_set(out, "npc_healing_total", log->npc_healing_total);
    dict_set(out, "mejkot_healing_total", log->mejkot_healing_total);
    dict_set(out, "jad_healing_total", log->jad_healing_total);
    dict_set(out, "preclip_reward", log->preclip_reward);
    dict_set(out, "postclip_reward", log->postclip_reward);
    dict_set(out, "positive_clip_count", log->positive_clip_count);
    dict_set(out, "negative_clip_count", log->negative_clip_count);

    static char npc_dmg_keys[NPC_TYPE_COUNT][48];
    static int npc_keys_built = 0;
    if (!npc_keys_built) {
        for (int i = 1; i < NPC_TYPE_COUNT; i++) {
            const char* npc_name = fc_episode_npc_metric_name(i);
            snprintf(npc_dmg_keys[i], 48, "dmg_to_%s", npc_name);
        }
        npc_keys_built = 1;
    }
    for (int i = 1; i < NPC_TYPE_COUNT; i++) {
        dict_set(out, npc_dmg_keys[i], log->dmg_to_npc_type[i]);
    }
}
