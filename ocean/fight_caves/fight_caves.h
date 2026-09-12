/*
 * fight_caves.h — PufferLib 4.0 environment wrapper for Fight Caves.
 *
 * Wraps simulation.h into PufferLib's c_reset/c_step/c_render interface.
 * All game logic is compiled from that shared implementation header.
 * This file only handles the PufferLib adapter layer:
 *   - FightCaves struct with PufferLib-required fields
 *   - c_reset: init game state, compute initial obs
 *   - c_step: read actions, step game, compute reward+obs, handle terminal
 *   - c_render: lazy full viewer for Puffer's standard evaluation loop
 *   - c_close: cleanup
 *
 * Single-agent environment (num_agents=1 always for Fight Caves).
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Shared simulation and contract implementation. */
#include "simulation.h"
#define FC_VIEWER_EMBEDDED
#include "viewer.c"
#undef FC_VIEWER_EMBEDDED
/* ======================================================================== */
/* PufferLib Log struct (required fields)                                    */
/* ======================================================================== */

typedef struct {
    float zero_progress_ticks;
    float wave_reached;
    float wrong_prayer_hits;
    float reached_wave_63;
    float jad_kill_rate;
    float prayer_uptime_range;
    float prayer_uptime_melee;
    float prayer_uptime_magic;
    float npc_healing_total;
    float jad_healing_total;
    float episode_length;
    float n;  /* PufferLib episode count; must be last. */
} Log;

static void fc_puffer_accumulate_episode_summary(
    Log* log, const FcEpisodeSummary* summary) {
    log->zero_progress_ticks += (float)summary->zero_progress_ticks;
    log->wave_reached += (float)summary->wave_reached;
    log->wrong_prayer_hits += (float)summary->wrong_prayer_hits;
    log->reached_wave_63 += (float)summary->reached_wave_63;
    log->jad_kill_rate += (float)summary->jad_kill_rate;
    log->prayer_uptime_range += (float)summary->prayer_uptime_range;
    log->prayer_uptime_melee += (float)summary->prayer_uptime_melee;
    log->prayer_uptime_magic += (float)summary->prayer_uptime_magic;
    log->npc_healing_total += (float)summary->npc_healing_total;
    log->jad_healing_total += (float)summary->jad_healing_total;
    log->episode_length += (float)summary->episode_length;
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
    ViewerState* viewer;         /* NULL throughout headless training */

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
    FcRewardBreakdown breakdown = fc_reward_compute_breakdown(
        &env->state, &env->reward_params, &env->reward_runtime);
    if (env->viewer) env->viewer->pending_reward_breakdown = breakdown;
    fc_reward_sync_progress_state(&env->state, &env->reward_runtime);
    return breakdown.total;
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
    fc_set_initial_supplies(&env->state, env->initial_sharks,
                            env->initial_prayer_doses);

    env->ep_length = 0;
    fc_reward_runtime_begin_episode(&env->reward_runtime, &env->state);
    /* Compute initial observations */
    fc_puffer_write_obs(env);
    if (env->viewer) env->viewer->reset_state = env->state;
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

    /* Compute reward */
    float reward = fc_puffer_compute_reward(env);
    env->rewards[0] = reward;
    env->ep_length++;

    /* Write the current tick's observation. On terminal steps, c_reset()
     * below replaces it with the next episode's initial observation. */
    fc_puffer_write_obs(env);

    /* A value snapshot survives same-step autoreset. Worker threads only copy
     * data here; all graphics and frame pacing remain inside c_render(). */
    if (env->viewer) {
        env->viewer->pending_state = env->state;
        env->viewer->pending_reward_runtime = env->reward_runtime;
        memcpy(env->viewer->pending_actions, actions, sizeof(actions));
        env->viewer->pending_frame = 1;
    }

    /* Check terminal */
    if (fc_is_terminal(&env->state)) {
        FcEpisodeSummary summary;
        fc_episode_summary_build(&env->state, &env->reward_runtime,
                                 env->ep_length, &summary);
        env->terminals[0] = 1.0f;
        fc_puffer_accumulate_episode_summary(&env->log, &summary);
        env->log.n += 1.0f;

        /* Same-step autoreset: return the completed episode's reward and
         * terminal flag alongside the next episode's initial observation. */
        c_reset(env);
    }
}

void c_render(FightCaves* env) {
    if (!env->viewer) {
        env->viewer = fc_viewer_create(1);
        if (!env->viewer) exit(EXIT_FAILURE);
        ViewerState* v = env->viewer;
        v->state = v->reset_state = env->state;
        v->reward_params = env->reward_params;
        v->reward_runtime = env->reward_runtime;
        v->active_loadout = env->state.active_loadout;
        v->obs_ablate_npc_distance = env->obs_ablate_npc_distance;
        v->obs_ablate_incoming_aggregates = env->obs_ablate_incoming_aggregates;
        v->obs_ablate_npc_valid = env->obs_ablate_npc_valid;
        snprintf(v->reward_config_path, sizeof(v->reward_config_path),
                 "Puffer environment configuration");
        v->reward_config_loaded = 1;
        fc_viewer_reset_presentation(v);
    }
    ViewerState* v = env->viewer;
    if (!v->pending_frame &&
        (v->state.rng_seed != env->state.rng_seed ||
         v->state.tick != env->state.tick)) {
        /* Also support an explicit VecEnv.reset() between render calls. */
        v->state = env->state;
        v->reward_runtime = env->reward_runtime;
        memset(&v->reward_breakdown, 0, sizeof(v->reward_breakdown));
        fc_viewer_reset_presentation(v);
    }
    fc_viewer_present_pending(v);
    int frame;
    do {
        frame = fc_viewer_frame(env->viewer, 1);
    } while (frame == 0);
    if (frame < 0) {
        fc_viewer_destroy(env->viewer);
        env->viewer = NULL;
        exit(EXIT_SUCCESS);
    }
}

void c_close(FightCaves* env) {
    fc_viewer_destroy(env->viewer);
    env->viewer = NULL;
    fc_destroy(&env->state);
}
