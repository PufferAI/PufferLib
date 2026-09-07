/*
 * fight_caves.h — PufferLib 4.0 environment wrapper for Fight Caves.
 *
 * Wraps the linked fc_core library into PufferLib's c_reset/c_step/c_render
 * interface. All game logic lives in fc-core.
 * This file only handles the PufferLib adapter layer:
 *   - FightCaves struct with PufferLib-required fields
 *   - c_reset: init game state, compute initial obs
 *   - c_step: read actions, step game, compute reward+obs, handle terminal
 *   - c_render: required no-op; evaluation uses the external viewer
 *   - c_close: cleanup
 *
 * Single-agent environment (num_agents=1 always for Fight Caves).
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Public fc_core interfaces used by the training adapter. */
#include "fc_types.h"
#include "fc_contracts.h"
#include "fc_api.h"
#include "fc_combat.h"
#include "fc_npc.h"
#include "fc_pathfinding.h"
#include "fc_prayer.h"
#include "fc_reward.h"
#include "fc_wave.h"

/* ======================================================================== */
/* PufferLib Log struct (required fields)                                    */
/* ======================================================================== */

typedef struct {
    float episode_length;
    float wave_reached;
    float npcs_slayed;
    float prayer_uptime_melee;
    float prayer_uptime_range;
    float prayer_uptime_magic;
    float correct_prayer;
    float wrong_prayer_hits;
    float no_prayer_hits;
    float prayer_switches;
    float damage_blocked;
    float dmg_taken_avg;
    float attack_when_ready_rate;
    float invalid_move;
    float invalid_attack;
    float invalid_prayer;
    float tokxil_melee_ticks;
    float ketzek_melee_ticks;
    float max_wave_ticks;
    float max_wave_ticks_wave;
    float reached_wave_63;
    float jad_kill_rate;
    float player_death_rate;
    float dmg_to_npc_type[NPC_TYPE_COUNT];
    float resolved_hits_to_npc_type[NPC_TYPE_COUNT];
    float damaging_hits_to_npc_type[NPC_TYPE_COUNT];
    float attack_cycles_to_npc_type[NPC_TYPE_COUNT];
    float target_ticks_by_npc_type[NPC_TYPE_COUNT];
    float target_held_ticks;
    float no_target_ticks;
    float target_in_range_los_ticks;
    float target_out_of_range_or_los_ticks;
    float attack_cooldown_wait_ticks;
    float ready_but_no_attack_ticks;
    float action_move_idle_ticks;
    float action_move_walk_ticks;
    float action_move_run_ticks;
    float action_attack_none_ticks;
    float action_attack_target_ticks;
    float action_prayer_noop_ticks;
    float action_prayer_cmd_ticks;
    float no_progress_ticks;
    float no_progress_idle_move_ticks;
    float no_progress_move_cmd_ticks;
    float no_progress_attack_none_ticks;
    float no_progress_attack_target_ticks;
    float no_progress_has_target_ticks;
    float no_progress_no_target_ticks;
    float no_progress_prayer_cmd_ticks;
    float no_progress_invalid_action_ticks;
    float required_work_remaining;
    float required_work_start;
    float cave_progress;
    float current_wave_progress;
    float progress_delta;
    float progress_reward;
    float ticks_since_positive_progress;
    float positive_progress_ticks;
    float zero_progress_ticks;
    float negative_progress_ticks;
    float gross_damage_dealt;
    float net_required_work_removed;
    float gross_damage_to_net_progress_ratio;
    float npc_healing_total;
    float mejkot_healing_total;
    float jad_healing_total;
    float preclip_reward;
    float postclip_reward;
    float positive_clip_count;
    float negative_clip_count;
    float rwd_sum[FC_CH_COUNT];
    float rwd_fires[FC_CH_COUNT];
    float n;  /* must be last */
} Log;

static void fc_puffer_accumulate_episode_summary(
    Log* log, const FcEpisodeSummary* summary) {
    log->episode_length += (float)summary->episode_length;
    log->wave_reached += (float)summary->wave_reached;
    log->npcs_slayed += (float)summary->npcs_slayed;
    log->prayer_uptime_melee += summary->prayer_uptime_melee;
    log->prayer_uptime_range += summary->prayer_uptime_range;
    log->prayer_uptime_magic += summary->prayer_uptime_magic;
    log->correct_prayer += (float)summary->correct_prayer;
    log->wrong_prayer_hits += (float)summary->wrong_prayer_hits;
    log->no_prayer_hits += (float)summary->no_prayer_hits;
    log->prayer_switches += (float)summary->prayer_switches;
    log->damage_blocked += (float)summary->damage_blocked;
    log->dmg_taken_avg += (float)summary->damage_taken;
    log->attack_when_ready_rate += summary->attack_when_ready_rate;
    log->invalid_move += (float)summary->invalid_move;
    log->invalid_attack += (float)summary->invalid_attack;
    log->invalid_prayer += (float)summary->invalid_prayer;
    log->tokxil_melee_ticks += (float)summary->tokxil_melee_ticks;
    log->ketzek_melee_ticks += (float)summary->ketzek_melee_ticks;
    log->max_wave_ticks += (float)summary->max_wave_ticks;
    log->max_wave_ticks_wave += (float)summary->max_wave_ticks_wave;
    log->reached_wave_63 += (float)summary->reached_wave_63;
    log->jad_kill_rate += (float)summary->jad_killed;
    log->player_death_rate += (float)summary->player_died;
    for (int i = 0; i < NPC_TYPE_COUNT; i++) {
        log->dmg_to_npc_type[i] +=
            (float)summary->damage_to_npc_type[i];
        log->resolved_hits_to_npc_type[i] +=
            (float)summary->resolved_hits_to_npc_type[i];
        log->damaging_hits_to_npc_type[i] +=
            (float)summary->damaging_hits_to_npc_type[i];
        log->attack_cycles_to_npc_type[i] +=
            (float)summary->attack_cycles_to_npc_type[i];
        log->target_ticks_by_npc_type[i] +=
            (float)summary->target_ticks_by_npc_type[i];
    }
    log->target_held_ticks += (float)summary->target_held_ticks;
    log->no_target_ticks += (float)summary->no_target_ticks;
    log->target_in_range_los_ticks +=
        (float)summary->target_in_range_los_ticks;
    log->target_out_of_range_or_los_ticks +=
        (float)summary->target_out_of_range_or_los_ticks;
    log->attack_cooldown_wait_ticks +=
        (float)summary->attack_cooldown_wait_ticks;
    log->ready_but_no_attack_ticks +=
        (float)summary->ready_but_no_attack_ticks;
    log->action_move_idle_ticks += (float)summary->action_move_idle_ticks;
    log->action_move_walk_ticks += (float)summary->action_move_walk_ticks;
    log->action_move_run_ticks += (float)summary->action_move_run_ticks;
    log->action_attack_none_ticks +=
        (float)summary->action_attack_none_ticks;
    log->action_attack_target_ticks +=
        (float)summary->action_attack_target_ticks;
    log->action_prayer_noop_ticks +=
        (float)summary->action_prayer_noop_ticks;
    log->action_prayer_cmd_ticks +=
        (float)summary->action_prayer_cmd_ticks;
}

/* ======================================================================== */
/* PufferLib Environment struct                                              */
/* ======================================================================== */

typedef struct FightCaves {
    Log log;                    /* required by PufferLib */
    float* observations;        /* required: FC_PUFFER_OBS_SIZE per agent */
    float* actions;             /* required: NUM_ATNS per agent (vecenv uses float*) */
    float* rewards;             /* required: 1 per agent */
    float* terminals;           /* required: 1 per agent (vecenv uses float*) */
    unsigned char* action_mask; /* required when MY_ACTION_MASK is enabled */
    int num_agents;             /* always 1 for Fight Caves */
    int rng;                    /* per-env RNG seed (set by vecenv.h) */

    /* Game state */
    FcState state;

    /* Reward weights and shaping configuration, initialized once per env. */
    FcRewardParams reward_params;
    int initial_sharks;
    int initial_prayer_doses;
    FcRewardRuntime reward_runtime;

    /* Obs ablation flags (experimental — see fc_apply_obs_ablation in fc_state.c).
     * When non-zero, the corresponding obs slots are zeroed AFTER fc_write_obs.
     * Used by the OBS Sweep / Ablation experiment to test which features the
     * policy actually relies on vs. which the GRU could re-derive from the rest. */
    int obs_ablate_npc_distance;
    int obs_ablate_incoming_aggregates;
    int obs_ablate_npc_valid;

    int ep_length;

    /* Per-episode reward-channel analytics. Reset at c_reset, transferred to
     * the per-env PufferLib Log on terminal. See FcRwdChannel enum in
     * fc_reward.h for channel indices and names. */
    float ep_rwd_sum[FC_CH_COUNT];
    int   ep_rwd_fires[FC_CH_COUNT];

    /* Puffer-action no-progress diagnostics. These are adapter-level metrics
     * for stall-like ticks: no movement, no attack cycle, no damage, no kill,
     * no wave clear, and not just normal in-range weapon cooldown waiting. */
    float ep_no_progress_ticks;
    float ep_no_progress_idle_move_ticks;
    float ep_no_progress_move_cmd_ticks;
    float ep_no_progress_attack_none_ticks;
    float ep_no_progress_attack_target_ticks;
    float ep_no_progress_has_target_ticks;
    float ep_no_progress_no_target_ticks;
    float ep_no_progress_prayer_cmd_ticks;
    float ep_no_progress_invalid_action_ticks;
    float ep_progress_delta;
    float ep_progress_reward;
    float ep_gross_damage_dealt;
    float ep_net_required_work_removed;
    float ep_npc_healing_total;
    float ep_mejkot_healing_total;
    float ep_jad_healing_total;
    float ep_preclip_reward;
    float ep_postclip_reward;
    float ep_positive_clip_count;
    float ep_negative_clip_count;

    /* RNG seed counter (increments each episode for variety) */
    uint32_t seed_counter;
} FightCaves;

/* ======================================================================== */
/* Observation writer — policy obs + action mask into flat float buffer      */
/* ======================================================================== */

static void fc_puffer_write_obs(FightCaves* env) {
    float* obs = env->observations;

    /* Policy observations */
    fc_write_obs(&env->state, obs);

    /* Optional obs ablation (zero specific feature slots in-place) */
    fc_apply_obs_ablation(obs,
                          env->obs_ablate_npc_distance,
                          env->obs_ablate_incoming_aggregates,
                          env->obs_ablate_npc_valid);

    /* Keep the float mask in observations for checkpoint compatibility, and
     * publish the same legality flags through PufferLib's native mask channel. */
    float full_mask[FC_ACTION_MASK_SIZE];
    fc_write_mask(&env->state, full_mask);
    memcpy(obs + FC_POLICY_OBS_SIZE, full_mask, sizeof(float) * FC_PUFFER_MASK_SIZE);
    if (env->action_mask != NULL) {
        for (int i = 0; i < FC_PUFFER_MASK_SIZE; i++) {
            env->action_mask[i] = (unsigned char)(full_mask[i] != 0.0f);
        }
    }
}

/* ======================================================================== */
/* Reward computation from reward features                                   */
/* ======================================================================== */

static float fc_puffer_compute_reward(FightCaves* env) {
    FcRewardBreakdown breakdown =
        fc_reward_compute_breakdown(
            &env->state, &env->reward_params, &env->reward_runtime);
    fc_reward_sync_progress_state(&env->state, &env->reward_runtime);

    if (breakdown.threat_ctx.tokxil_melee) env->state.ep_tokxil_melee_ticks++;
    if (breakdown.threat_ctx.ketzek_melee) env->state.ep_ketzek_melee_ticks++;

    /* Per-channel analytics: accumulate value and fire count per reward channel.
     * Drains into the per-env PufferLib Log on terminal; see c_step. */
    float ch[FC_CH_COUNT];
    fc_reward_breakdown_channels(&breakdown, ch);
    for (int i = 0; i < FC_CH_COUNT; i++) {
        env->ep_rwd_sum[i] += ch[i];
        if (ch[i] != 0.0f) env->ep_rwd_fires[i]++;
    }

    env->ep_progress_delta += env->reward_runtime.last_progress_delta;
    env->ep_progress_reward += breakdown.progress;
    env->ep_gross_damage_dealt += (float)env->state.damage_dealt_this_tick;
    env->ep_net_required_work_removed += env->reward_runtime.last_net_required_work_removed;
    env->ep_npc_healing_total += (float)env->state.npc_heal_amount_this_tick;
    env->ep_mejkot_healing_total += (float)env->state.mejkot_heal_amount_this_tick;
    env->ep_jad_healing_total += (float)env->state.jad_heal_amount_this_tick;
    env->ep_preclip_reward += breakdown.total;
    {
        float clipped = breakdown.total;
        if (clipped > 1.0f) {
            clipped = 1.0f;
            env->ep_positive_clip_count += 1.0f;
        } else if (clipped < -1.0f) {
            clipped = -1.0f;
            env->ep_negative_clip_count += 1.0f;
        }
        env->ep_postclip_reward += clipped;
    }

    return breakdown.total;
}

static void fc_puffer_reset_episode_action_diagnostics(FightCaves* env) {
    env->ep_no_progress_ticks = 0.0f;
    env->ep_no_progress_idle_move_ticks = 0.0f;
    env->ep_no_progress_move_cmd_ticks = 0.0f;
    env->ep_no_progress_attack_none_ticks = 0.0f;
    env->ep_no_progress_attack_target_ticks = 0.0f;
    env->ep_no_progress_has_target_ticks = 0.0f;
    env->ep_no_progress_no_target_ticks = 0.0f;
    env->ep_no_progress_prayer_cmd_ticks = 0.0f;
    env->ep_no_progress_invalid_action_ticks = 0.0f;
    env->ep_progress_delta = 0.0f;
    env->ep_progress_reward = 0.0f;
    env->ep_gross_damage_dealt = 0.0f;
    env->ep_net_required_work_removed = 0.0f;
    env->ep_npc_healing_total = 0.0f;
    env->ep_mejkot_healing_total = 0.0f;
    env->ep_jad_healing_total = 0.0f;
    env->ep_preclip_reward = 0.0f;
    env->ep_postclip_reward = 0.0f;
    env->ep_positive_clip_count = 0.0f;
    env->ep_negative_clip_count = 0.0f;
}

static void fc_puffer_record_no_progress_diagnostics(
    FightCaves* env,
    const int actions[FC_NUM_ACTION_HEADS]) {
    const FcState* state = &env->state;
    const FcPlayer* player = &state->player;
    int target_active = 0;

    if (state->terminal != TERMINAL_NONE || state->npcs_remaining <= 0) return;
    if (state->movement_this_tick || state->attack_attempt_this_tick) return;
    if (state->damage_dealt_this_tick > 0 || state->npcs_killed_this_tick > 0) return;
    if (state->wave_just_cleared) return;

    if (player->attack_target_idx >= 0) {
        const FcNpc* target = &state->npcs[player->attack_target_idx];
        target_active = target->active && !target->is_dead;
        if (target_active && player->attack_timer > 0) {
            int dist = fc_distance_to_npc(player->x, player->y, target);
            int has_los = fc_has_los_between_areas(
                player->x, player->y, 1,
                target->x, target->y, target->size, state->los_flags);
            if (dist <= player->weapon_range && has_los) return;
        }
    }

    env->ep_no_progress_ticks += 1.0f;
    if (actions[0] == FC_MOVE_IDLE) {
        env->ep_no_progress_idle_move_ticks += 1.0f;
    } else {
        env->ep_no_progress_move_cmd_ticks += 1.0f;
    }
    if (actions[1] == FC_ATTACK_NONE) {
        env->ep_no_progress_attack_none_ticks += 1.0f;
    } else {
        env->ep_no_progress_attack_target_ticks += 1.0f;
    }
    if (target_active) {
        env->ep_no_progress_has_target_ticks += 1.0f;
    } else {
        env->ep_no_progress_no_target_ticks += 1.0f;
    }
    if (actions[2] != 0) {
        env->ep_no_progress_prayer_cmd_ticks += 1.0f;
    }
    if (state->invalid_action_this_tick) {
        env->ep_no_progress_invalid_action_ticks += 1.0f;
    }
}

/* ======================================================================== */
/* PufferLib interface: c_reset, c_step, c_render, c_close                   */
/* ======================================================================== */

static uint32_t fc_puffer_mix_reset_seed(uint32_t env_rng, uint32_t episode) {
    uint32_t x = env_rng + 0x9E3779B9u * (episode + 1u);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return (x != 0u) ? x : 0x12345678u;
}

void c_reset(FightCaves* env) {
    env->seed_counter++;
    fc_reset(&env->state,
             fc_puffer_mix_reset_seed((uint32_t)env->rng, env->seed_counter));
    if (env->initial_sharks < 0) env->initial_sharks = 0;
    if (env->initial_sharks > FC_MAX_SHARKS) env->initial_sharks = FC_MAX_SHARKS;
    if (env->initial_prayer_doses < 0) env->initial_prayer_doses = 0;
    if (env->initial_prayer_doses > FC_MAX_PRAYER_DOSES)
        env->initial_prayer_doses = FC_MAX_PRAYER_DOSES;
    env->state.player.sharks_remaining = env->initial_sharks;
    env->state.player.prayer_doses_remaining = env->initial_prayer_doses;

    env->ep_length = 0;
    fc_reward_runtime_begin_episode(&env->reward_runtime, &env->state);
    for (int i = 0; i < FC_CH_COUNT; i++) {
        env->ep_rwd_sum[i] = 0.0f;
        env->ep_rwd_fires[i] = 0;
    }
    fc_puffer_reset_episode_action_diagnostics(env);

    /* Compute initial observations */
    fc_puffer_write_obs(env);
}

void c_step(FightCaves* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    /* Convert float actions from network to int action heads.
     * PufferLib sends actions as floats in a flat array.
     * Puffer-facing no-supplies policy uses only move/attack/prayer.
     * Core heads 3-6 are left as zero: no eat, no drink, no walk-to-tile. */
    int actions[FC_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    for (int h = 0; h < FC_PUFFER_NUM_ATNS; h++) {
        actions[h] = (int)env->actions[h];
    }
    /* Heads 5+6 (walk-to-tile) always 0 — not used in v1 */

    /* Step the game simulation */
    fc_step(&env->state, actions);
    fc_puffer_record_no_progress_diagnostics(env, actions);

    /* Compute reward */
    float reward = fc_puffer_compute_reward(env);
    env->rewards[0] = reward;
    env->ep_length++;

    /* Write the current tick's observation. On terminal steps, c_reset()
     * below replaces it with the next episode's initial observation. */
    fc_puffer_write_obs(env);

    /* Check terminal */
    if (fc_is_terminal(&env->state)) {
        FcEpisodeSummary summary;
        fc_episode_summary_build(&env->state, env->ep_length, &summary);
        env->terminals[0] = 1.0f;
        fc_puffer_accumulate_episode_summary(&env->log, &summary);
        env->log.no_progress_ticks += env->ep_no_progress_ticks;
        env->log.no_progress_idle_move_ticks += env->ep_no_progress_idle_move_ticks;
        env->log.no_progress_move_cmd_ticks += env->ep_no_progress_move_cmd_ticks;
        env->log.no_progress_attack_none_ticks += env->ep_no_progress_attack_none_ticks;
        env->log.no_progress_attack_target_ticks += env->ep_no_progress_attack_target_ticks;
        env->log.no_progress_has_target_ticks += env->ep_no_progress_has_target_ticks;
        env->log.no_progress_no_target_ticks += env->ep_no_progress_no_target_ticks;
        env->log.no_progress_prayer_cmd_ticks += env->ep_no_progress_prayer_cmd_ticks;
        env->log.no_progress_invalid_action_ticks += env->ep_no_progress_invalid_action_ticks;
        env->log.required_work_remaining +=
            env->reward_runtime.last_required_work_remaining;
        env->log.required_work_start +=
            env->reward_runtime.required_work_at_wave_start;
        env->log.cave_progress += env->reward_runtime.last_cave_progress;
        env->log.current_wave_progress +=
            env->reward_runtime.last_current_wave_progress;
        env->log.progress_delta += env->ep_progress_delta;
        env->log.progress_reward += env->ep_progress_reward;
        env->log.ticks_since_positive_progress +=
            (float)env->reward_runtime.ticks_since_positive_progress;
        env->log.positive_progress_ticks +=
            (float)env->reward_runtime.positive_progress_ticks;
        env->log.zero_progress_ticks +=
            (float)env->reward_runtime.zero_progress_ticks;
        env->log.negative_progress_ticks +=
            (float)env->reward_runtime.negative_progress_ticks;
        env->log.gross_damage_dealt += env->ep_gross_damage_dealt;
        env->log.net_required_work_removed += env->ep_net_required_work_removed;
        env->log.gross_damage_to_net_progress_ratio +=
            env->ep_gross_damage_dealt /
            ((env->ep_net_required_work_removed > 1.0f)
                ? env->ep_net_required_work_removed : 1.0f);
        env->log.npc_healing_total += env->ep_npc_healing_total;
        env->log.mejkot_healing_total += env->ep_mejkot_healing_total;
        env->log.jad_healing_total += env->ep_jad_healing_total;
        env->log.preclip_reward += env->ep_preclip_reward;
        env->log.postclip_reward += env->ep_postclip_reward;
        env->log.positive_clip_count += env->ep_positive_clip_count;
        env->log.negative_clip_count += env->ep_negative_clip_count;

        for (int i = 0; i < FC_CH_COUNT; i++) {
            env->log.rwd_sum[i] += env->ep_rwd_sum[i];
            env->log.rwd_fires[i] += (float)env->ep_rwd_fires[i];
        }
        env->log.n += 1.0f;

        /* Same-step autoreset: return the completed episode's reward and
         * terminal flag alongside the next episode's initial observation. */
        c_reset(env);
    }
}

void c_render(FightCaves* env) {
    /* Rendering handled by external viewer via --policy-pipe mode.
     * See fc-viewer/eval_viewer.py for the eval pipeline. */
    (void)env;
}

void c_close(FightCaves* env) {
    fc_destroy(&env->state);
}
