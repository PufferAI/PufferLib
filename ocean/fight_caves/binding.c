/*
 * binding.c — PufferLib 4.0 binding for Fight Caves environment.
 *
 * Defines the macros PufferLib needs, includes vecenv.h, and implements
 * the my_init/my_log hooks for config parsing and stat logging.
 */

#include "fight_caves.h"

/* Kept in this linked object so the selected Puffer extension exports the
 * same implementation exercised by validation. It is never called per-step. */
/* Cold-path machine-readable metadata exported by the compiled FC backend. */

#include <stdio.h>

#include "simulation.h"
#if defined(_WIN32)
#define FC_CONTRACT_EXPORT __declspec(dllexport)
#else
#define FC_CONTRACT_EXPORT __attribute__((visibility("default")))
#endif

#define FC_STRINGIFY_INNER(value) #value
#define FC_STRINGIFY(value) FC_STRINGIFY_INNER(value)

FC_CONTRACT_EXPORT const char* fc_training_contract_json(void) {
    static char json[2048];
    static int initialized = 0;
    if (!initialized) {
        snprintf(
            json,
            sizeof(json),
            "{"
            "\"contract_dump_schema_version\":%d,"
            "\"policy_obs_size\":%d,"
            "\"puffer_obs_size\":%d,"
            "\"puffer_action_dims\":[%d,%d,%d],"
            "\"puffer_mask_size\":%d,"
            "\"core_obs_size\":%d,"
            "\"core_action_dims\":[%d,%d,%d,%d,%d,%d,%d],"
            "\"core_action_mask\":%d,"
            "\"reward_feature_count\":%d,"
            "\"observation_version\":\"%s\","
            "\"action_version\":\"%s\","
            "\"reward_version\":\"%s\","
            "\"prayer_timing_version\":\"%s\","
            "\"state_hash_version\":%u,"
            "\"active_loadout\":\"%s\""
            "}",
            FC_CONTRACT_DUMP_SCHEMA_VERSION,
            FC_POLICY_OBS_SIZE,
            FC_PUFFER_OBS_SIZE,
            FC_PUFFER_ACTION_DIMS[0],
            FC_PUFFER_ACTION_DIMS[1],
            FC_PUFFER_ACTION_DIMS[2],
            FC_PUFFER_MASK_SIZE,
            FC_OBS_SIZE,
            FC_ACTION_DIMS[0],
            FC_ACTION_DIMS[1],
            FC_ACTION_DIMS[2],
            FC_ACTION_DIMS[3],
            FC_ACTION_DIMS[4],
            FC_ACTION_DIMS[5],
            FC_ACTION_DIMS[6],
            FC_ACTION_MASK_SIZE,
            FC_REWARD_FEATURES,
            FC_OBSERVATION_VERSION,
            FC_ACTION_VERSION,
            FC_REWARD_VERSION,
            FC_PRAYER_TIMING_VERSION,
            FC_STATE_HASH_VERSION,
            FC_STRINGIFY(FC_ACTIVE_LOADOUT));
        initialized = 1;
    }
    return json;
}


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
     * See fc_apply_obs_ablation in simulation.h for what each zeroes. */
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
    dict_set(out, "zero_progress_ticks", log->zero_progress_ticks);
    dict_set(out, "wave_reached", log->wave_reached);
    dict_set(out, "wrong_prayer_hits", log->wrong_prayer_hits);
    dict_set(out, "reached_wave_63", log->reached_wave_63);
    dict_set(out, "jad_kill_rate", log->jad_kill_rate);
    dict_set(out, "prayer_uptime_range", log->prayer_uptime_range);
    dict_set(out, "prayer_uptime_melee", log->prayer_uptime_melee);
    dict_set(out, "prayer_uptime_magic", log->prayer_uptime_magic);
    dict_set(out, "npc_healing_total", log->npc_healing_total);
    dict_set(out, "jad_healing_total", log->jad_healing_total);
    dict_set(out, "episode_length", log->episode_length);
}
