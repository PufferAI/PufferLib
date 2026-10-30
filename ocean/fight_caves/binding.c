/*
 * binding.c — PufferLib 5.0 adapter for Fight Caves: puf_init, puf_reset,
 * puf_step, puf_render, puf_close and puf_log. It is the one C translation
 * unit that compiles the core implementation and the viewer.
 */

/* This translation unit is the sole implementation owner in native builds. */
#define FC_CORE_IMPLEMENTATION
#include "fight_caves.h"
#undef FC_CORE_IMPLEMENTATION

/* The [env] settings puf_init and the viewer's manual play share: every
 * FC_REWARD_SETTINGS key (a missing key keeps its default) and the starting
 * supplies. */
static void fc_apply_env_config(Dict* kwargs, FcRewardParams* reward,
                                int* initial_sharks, int* initial_prayer_doses) {
    *reward = fc_reward_default_params();
    for (int i = 0; i < FC_REWARD_SETTING_COUNT; i++) {
        DictItem* item = dict_find(kwargs, FC_REWARD_SETTINGS[i].key);
        if (item != NULL)
            fc_reward_setting_store(reward, &FC_REWARD_SETTINGS[i], item->value);
    }
    DictItem* item = dict_find(kwargs, "initial_sharks");
    *initial_sharks = item ? (int)item->value : 0;
    item = dict_find(kwargs, "initial_prayer_doses");
    *initial_prayer_doses = item ? (int)item->value : 0;
}

#include "viewer.c"

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

static void fc_puffer_write_obs(FightCaves* env) {
    Agent* agent = &env->agents[0];
    float* obs = agent->observations;

    /* Policy observations */
    fc_write_obs(&env->state, obs);

    /* Keep the float mask in observations for checkpoint compatibility, and
     * publish the same legality flags through PufferLib's native mask channel. */
    float full_mask[FC_ACTION_MASK_SIZE];
    fc_write_mask(&env->state, full_mask);
    memcpy(obs + FC_POLICY_OBS_SIZE, full_mask, sizeof(float) * FC_PUFFER_MASK_SIZE);
    if (agent->action_mask != NULL) {
        for (int i = 0; i < FC_PUFFER_MASK_SIZE; i++) {
            agent->action_mask[i] = (unsigned char)(full_mask[i] != 0.0f);
        }
    }
}

/* ======================================================================== */
/* Reward computation                                                        */
/* ======================================================================== */

static float fc_puffer_compute_reward(FightCaves* env) {
    FcRewardBreakdown breakdown = fc_reward_compute_breakdown(
        &env->state, &env->reward_params, &env->reward_runtime);
    if (env->viewer) env->viewer->pending_reward_breakdown = breakdown;
    fc_reward_sync_progress_state(&env->state, &env->reward_runtime);
    return breakdown.total;
}


/* ======================================================================== */
/* PufferLib interface: puf_reset, puf_step, puf_render, puf_close           */
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

void puf_reset(Env* env) {
    env->seed_counter++;
    fc_reset(&env->state,
             fc_puffer_mix_reset_seed((uint32_t)env->rng, env->seed_counter));
    /* fc_set_initial_supplies clamps the configured supply counts. */
    fc_set_initial_supplies(&env->state, env->initial_sharks,
                            env->initial_prayer_doses);
    fc_reward_runtime_begin_episode(&env->reward_runtime, &env->state);
    /* Compute initial observations */
    fc_puffer_write_obs(env);
    if (env->viewer) env->viewer->reset_state = env->state;
}

void puf_step(Env* env) {
    Agent* agent = &env->agents[0];
    agent->terminals[0] = 0.0f;

    /* Convert float actions from network to int action heads.
     * PufferLib sends actions as floats in a flat array.
     * Puffer-facing no-supplies policy uses only move/attack/prayer.
     * Core heads 3-6 are left as zero: no eat, no drink, no walk-to-tile. */
    int actions[FC_NUM_ACTION_HEADS] = {0};
    for (int h = 0; h < FC_PUFFER_NUM_ATNS; h++) {
        actions[h] = (int)agent->actions[h];
    }

    /* Step the game simulation */
    fc_step(&env->state, actions);

    /* Compute reward */
    agent->rewards[0] = fc_puffer_compute_reward(env);

    /* Write the current tick's observation. On terminal steps, puf_reset()
     * below replaces it with the next episode's initial observation. */
    fc_puffer_write_obs(env);

    /* A value snapshot survives same-step autoreset. Worker threads only copy
     * data here; all graphics and frame pacing remain inside puf_render(). */
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
                                 env->state.tick, &summary);
        agent->terminals[0] = 1.0f;
        fc_puffer_accumulate_episode_summary(&env->log, &summary);
        env->log.n += 1.0f;

        /* Same-step autoreset: return the completed episode's reward and
         * terminal flag alongside the next episode's initial observation. */
        puf_reset(env);
    }
}

void puf_render(Env* env) {
    if (!env->viewer) {
        env->viewer = fc_viewer_create(1);
        if (!env->viewer) exit(EXIT_FAILURE);
        ViewerState* v = env->viewer;
        v->state = v->reset_state = env->state;
        v->reward_params = env->reward_params;
        v->reward_runtime = env->reward_runtime;
        snprintf(v->reward_config_path, sizeof(v->reward_config_path),
                 "Puffer environment configuration");
        v->reward_config_loaded = 1;
        fc_viewer_reset_presentation(v);
    }
    ViewerState* v = env->viewer;
    if (!v->pending_frame &&
        (v->state.rng_seed != env->state.rng_seed ||
         v->state.tick != env->state.tick)) {
        /* Also support an explicit puf_reset() between render calls. */
        v->state = env->state;
        v->reward_runtime = env->reward_runtime;
        memset(&v->reward_breakdown, 0, sizeof(v->reward_breakdown));
        fc_viewer_reset_presentation(v);
    }
    fc_viewer_present_pending(v);
    int frame;
    do {
        frame = fc_viewer_frame(env->viewer);
    } while (frame == 0);
    if (frame < 0) {
        fc_viewer_destroy(env->viewer);
        env->viewer = NULL;
        exit(EXIT_SUCCESS);
    }
}

void puf_close(Env* env) {
    fc_viewer_destroy(env->viewer);
    env->viewer = NULL;
    fc_destroy(&env->state);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;  /* Fight Caves is single-agent */
    fc_apply_env_config(kwargs, &env->reward_params, &env->initial_sharks,
                        &env->initial_prayer_doses);

    /* Callers pass a zeroed Env (Puffer callocs its envs and zeroes its
     * bufferless probe), so the other fields need no initialization. */
    fc_init(&env->state);
}

void puf_log(Log* log, Dict* out) {
    /* Native evaluation requires perf; use the existing success metric for
     * both standard aliases without adding counters or changing rewards. */
    dict_set(out, "perf", log->jad_kill_rate);
    dict_set(out, "score", log->jad_kill_rate);
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
