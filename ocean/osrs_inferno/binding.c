/**
 * @file binding.c
 * @brief Static-native binding for OSRS Inferno encounter.
 *
 * Bridges vecenv.h's contract (float actions, float terminals) with the
 * Inferno encounter's vtable interface.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "osrs_env.h"  /* pulls in osrs_types, encounter, pvp stack */

/* encounter headers + render.h have many static helpers only used by the
   standalone viewer (not c_render) — suppress unused-function noise. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "encounters/encounter_inferno.h"
#include "encounters/encounter_zulrah.h"  /* render.h references ZulrahState */
#include "osrs_render.h"
#pragma GCC diagnostic pop

#define INF_TOTAL_OBS (INF_NUM_OBS + INF_ACTION_MASK_SIZE)

typedef struct {
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int rng;
    Log log;

    EncounterState* enc_state;
    int config_start_wave;  /* the start_wave from config (not curriculum override) */

    int acts_staging[INF_NUM_ACTION_HEADS];
    unsigned char term_staging;

    /* best-episode replay recording: all envs buffer their current episode's actions.
       on terminal, if the episode reached a new global best wave, flush to disk.
       binary format: [int32 num_ticks] [uint32 rng_state] [num_heads int32 per tick] */
    int* episode_actions;    /* buffer: episode_len * NUM_ATNS ints */
    int episode_action_cap;  /* max ticks we can buffer */
    int episode_action_len;  /* ticks buffered so far this episode */
    uint32_t episode_rng_start; /* RNG state at start of current episode */

    /* replay playback: when PLAY_REPLAY=path is set, override the policy's
       actions with the recorded ones. first env only (env 0). */
    int* replay_actions;     /* full action buffer: num_ticks * NUM_ATNS */
    int replay_num_ticks;
    int replay_cursor;       /* ticks consumed so far */
    uint32_t replay_rng_seed;

    float ticks_per_second;
    double last_step_time;

    /* set by c_step on terminal-reset; consumed by c_render on its next call
       to mirror the standalone viewer's post-reset cleanup (osrs_visual.c:186). */
    int pending_render_reset;

    OsrsEnv render_env; /* minimal env wrapper for pvp_render() */
} InfernoEnv;

#define OBS_SIZE INF_TOTAL_OBS
#define NUM_ATNS INF_NUM_ACTION_HEADS
#define ACT_SIZES { ENCOUNTER_MOVE_ACTIONS, ENCOUNTER_OVERHEAD_DIM_PVE, INF_OBS_NPCS+1, 5, 2, 4, 3, 2, ENCOUNTER_OFFENSIVE_DIM }
#define OBS_TENSOR_T FloatTensor
#define Env InfernoEnv

/* global best episode tracking */
static int g_best_wave = 0;
static int g_best_ticks = 999999;
static int g_best_min_zuk_hp = 999999;  /* lowest Zuk HP reached (for Zuk-only training) */

void c_step(Env* env) {
    /* tick pacing lives in c_render — it blocks at the tick deadline calling
       pvp_render at ~60fps so sub-tile interpolation can animate between sim
       ticks. nothing to do here timing-wise. */

    int used_human_commands = 0;
    RenderClient* render_client = (RenderClient*)env->render_env.client;

    /* replay playback: if this env has a loaded replay, override policy actions */
    if (env->replay_actions && env->replay_cursor < env->replay_num_ticks) {
        int off = env->replay_cursor * NUM_ATNS;
        for (int i = 0; i < NUM_ATNS; i++)
            env->acts_staging[i] = env->replay_actions[off + i];
        env->replay_cursor++;
        if (render_client) {
            human_input_clear_pending(&render_client->human_input);
            human_input_clear_move(&render_client->human_input);
            ENCOUNTER_INFERNO.put_int(env->enc_state, "player_dest_x", -1);
            ENCOUNTER_INFERNO.put_int(env->enc_state, "player_dest_y", -1);
            ENCOUNTER_INFERNO.put_int(env->enc_state, "human_command_mode", 0);
        }
    } else if (render_client && render_client->human_input.enabled &&
               ENCOUNTER_INFERNO.step_human_commands) {
        if (env->episode_actions && render_client->human_input.commands.count > 0) {
            fprintf(stderr, "RECORD_REPLAY cannot record human command mode\n");
            abort();
        }
        ENCOUNTER_INFERNO.step_human_commands(env->enc_state, &render_client->human_input);
        used_human_commands = 1;
    } else {
        if (render_client) {
            human_input_clear_pending(&render_client->human_input);
            human_input_clear_move(&render_client->human_input);
            ENCOUNTER_INFERNO.put_int(env->enc_state, "player_dest_x", -1);
            ENCOUNTER_INFERNO.put_int(env->enc_state, "player_dest_y", -1);
            ENCOUNTER_INFERNO.put_int(env->enc_state, "human_command_mode", 0);
        }
        for (int i = 0; i < NUM_ATNS; i++)
            env->acts_staging[i] = (int)env->actions[i];
    }

    /* buffer actions for best-episode recording */
    if (env->episode_actions && !used_human_commands) {
        /* capture RNG state at the very start of the episode (before first action) */
        if (env->episode_action_len == 0)
            env->episode_rng_start = ((InfernoState*)env->enc_state)->rng_state;
        if (env->episode_action_len < env->episode_action_cap) {
            memcpy(&env->episode_actions[env->episode_action_len * NUM_ATNS],
                   env->acts_staging, NUM_ATNS * sizeof(int));
            env->episode_action_len++;
        }
    }

    if (!used_human_commands)
        ENCOUNTER_INFERNO.step(env->enc_state, env->acts_staging);

    float* obs = (float*)env->observations;
    ENCOUNTER_INFERNO.write_obs(env->enc_state, obs);
    ENCOUNTER_INFERNO.write_mask(env->enc_state, obs + INF_NUM_OBS);

    env->rewards[0] = ENCOUNTER_INFERNO.get_reward(env->enc_state);

    int is_term = ENCOUNTER_INFERNO.is_terminal(env->enc_state);
    env->term_staging = (unsigned char)is_term;
    env->terminals[0] = (float)is_term;

    /* terminal-only logging: accumulate completed episode stats into env->log.
       vecenv polls with static_vec_log() which sums across agents, divides by
       total n, then clears. this gives proper per-completed-episode averages
       instead of noisy mid-episode snapshots. */
    if (is_term) {
        InfernoState* s = (InfernoState*)env->enc_state;

        /* only count episodes that match the configured start_wave.
           curriculum agents (overridden wave) are excluded from metrics. */
        if (s->start_wave != env->config_start_wave) goto skip_log;

        env->log.episode_return += s->episode_return;
        env->log.episode_length += (float)s->tick;
        env->log.damage_dealt += s->total_damage_dealt;
        env->log.zuk_healer_damage += s->total_zuk_healer_damage;
        env->log.damage_received += s->total_damage_received;
        env->log.hp_restored += s->total_hp_restored;
        env->log.wins += (s->winner == 0) ? 1.0f : 0.0f;
        env->log.wave += (float)s->wave;
        env->log.prayer_correct += (float)s->total_prayer_correct;
        env->log.prayer_total += (float)s->total_npc_attacks;
        env->log.idle_ticks += (float)s->total_idle_ticks;
        env->log.brews_used += (float)s->total_brews_used;
        env->log.blood_healed += (float)s->total_blood_healed;
        env->log.unavoidable_off_prayer += (float)s->total_unavoidable_off;
        env->log.brews_remaining += (float)s->player.brew_doses;
        env->log.restores_remaining += (float)s->player.restore_doses;
        env->log.prayer_at_death += (float)s->player.current_prayer;
        env->log.npc_kills += (float)s->total_npc_kills;
        env->log.gear_switches += (float)s->total_gear_switches;
        env->log.current_ranged += (float)s->player.current_ranged;
        env->log.current_magic += (float)s->player.current_magic;
        env->log.start_wave += (float)env->config_start_wave;

        for (int t = 0; t < INF_NUM_NPC_TYPES; t++) {
            env->log.prayer_correct_by_type[t] += (float)s->prayer_correct_by_type[t];
            env->log.attacks_by_type[t] += (float)s->attacks_by_type[t];
            env->log.dmg_from_type[t] += s->dmg_from_type[t];
            env->log.killed_by_type[t] += (float)s->killed_by_type[t];
        }

        /* Zuk shield tracking */
        env->log.behind_shield_pct += (s->total_zuk_ticks > 0)
            ? (float)s->behind_shield_ticks / (float)s->total_zuk_ticks : 0.0f;

        /* Zuk HP remaining at episode end */
        {
            float zhp = 1200.0f;
            for (int n = 0; n < INF_MAX_NPCS; n++) {
                if (s->npcs[n].type == INF_NPC_ZUK) {
                    zhp = (float)s->npcs[n].hp;
                    break;
                }
            }
            if (s->winner == 0) zhp = 0.0f;
            env->log.zuk_hp_remaining += zhp;
        }
        env->log.min_zuk_hp_seen += (s->winner == 0)
            ? 0.0f
            : (s->min_zuk_hp_seen > 0.0f ? s->min_zuk_hp_seen : 1200.0f);

        /* action noop rates (per-episode ratios, averaged across episodes by aggregator) */
        float at = (float)s->action_total_count;
        if (at > 0.0f) {
            env->log.noop_move   += (float)s->action_noop_count[0] / at;
            env->log.noop_prayer += (float)s->action_noop_count[1] / at;
            env->log.noop_target += (float)s->action_noop_count[2] / at;
            env->log.noop_gear   += (float)s->action_noop_count[3] / at;
            env->log.noop_eat    += (float)s->action_noop_count[4] / at;
            env->log.noop_potion += (float)s->action_noop_count[5] / at;
            env->log.noop_spell  += (float)s->action_noop_count[6] / at;
            env->log.noop_spec   += (float)s->action_noop_count[7] / at;
        }

        env->log.n += 1.0f;
    skip_log:;
    }

    if (is_term) {
        /* check if this episode is a new global best — if so, flush replay to disk.
           for full runs (start_wave 0): best = highest wave reached, then fewest ticks.
           for zuk-only (start_wave 68+): best = most damage to zuk (lowest zuk HP), then fewest ticks.
           curriculum starts from mid-waves also record. */
        if (env->episode_actions && env->episode_action_len > 0) {
            InfernoState* st = (InfernoState*)env->enc_state;
            int wave = st->wave;
            int ticks = env->episode_action_len;
            int is_new_best = 0;
            if (st->start_wave == 0) {
                /* full run: best wave, then fewest ticks */
                is_new_best = (wave > g_best_wave || (wave == g_best_wave && ticks < g_best_ticks));
            } else {
                /* partial/zuk run: best = lowest Zuk HP reached.
                   if Zuk is dead (winner==0), fastest kill (fewest ticks) wins. */
                int min_zuk_hp = (st->winner == 0)
                    ? 0
                    : (st->min_zuk_hp_seen > 0.0f ? (int)st->min_zuk_hp_seen : 1200);
                is_new_best = (min_zuk_hp < g_best_min_zuk_hp ||
                              (min_zuk_hp == g_best_min_zuk_hp && min_zuk_hp == 0 && ticks < g_best_ticks));
                if (is_new_best) g_best_min_zuk_hp = min_zuk_hp;
            }
            if (is_new_best) {
                g_best_wave = wave;
                g_best_ticks = ticks;
                const char* rpath = getenv("RECORD_REPLAY");
                if (rpath && rpath[0]) {
                    FILE* fp = fopen(rpath, "wb");
                    if (!fp) {
                        fprintf(stderr, "record_best_replay_path: cannot open %s\n", rpath);
                        abort();
                    }
                    fwrite(&env->episode_action_len, sizeof(int), 1, fp);
                    fwrite(&env->episode_rng_start, sizeof(uint32_t), 1, fp);
                    fwrite(env->episode_actions, sizeof(int),
                           env->episode_action_len * NUM_ATNS, fp);
                    fclose(fp);
                    if (st->start_wave >= 68) {
                        fprintf(stderr, "replay: new best min zuk hp=%d (%d ticks, rng=%u) saved to %s\n",
                                g_best_min_zuk_hp, env->episode_action_len, env->episode_rng_start, rpath);
                    } else {
                        fprintf(stderr, "replay: new best wave %d (%d ticks, rng=%u) saved to %s\n",
                                wave, env->episode_action_len, env->episode_rng_start, rpath);
                    }
                }
            }
        }
        env->episode_action_len = 0;

        ENCOUNTER_INFERNO.reset(env->enc_state, 0);
        ENCOUNTER_INFERNO.write_obs(env->enc_state, obs);
        ENCOUNTER_INFERNO.write_mask(env->enc_state, obs + INF_NUM_OBS);

        /* render-side cleanup needs the RenderClient which lives in c_render.
           raise a flag so the next c_render call does the cleanup. */
        env->pending_render_reset = 1;
    }
}

void c_reset(Env* env) {
    /* if replaying, seed the encounter RNG to match the recording */
    uint32_t seed = env->replay_actions ? env->replay_rng_seed : 0;
    ENCOUNTER_INFERNO.reset(env->enc_state, seed);
    env->replay_cursor = 0;

    float* obs = (float*)env->observations;
    ENCOUNTER_INFERNO.write_obs(env->enc_state, obs);
    ENCOUNTER_INFERNO.write_mask(env->enc_state, obs + INF_NUM_OBS);

    env->rewards[0] = 0.0f;
    env->term_staging = 0;
    env->terminals[0] = 0.0f;
}

void c_close(Env* env) {
    free(env->episode_actions);
    env->episode_actions = NULL;
    free(env->replay_actions);
    env->replay_actions = NULL;
    if (env->enc_state) {
        ENCOUNTER_INFERNO.destroy(env->enc_state);
        env->enc_state = NULL;
    }
    if (env->render_env.client) {
        render_destroy_client((RenderClient*)env->render_env.client);
        env->render_env.client = NULL;
    }
}

void c_render(Env* env) {
    OsrsEnv* re = &env->render_env;
    re->encounter_def = (void*)&ENCOUNTER_INFERNO;
    re->encounter_state = env->enc_state;

    /* first pvp_render creates the RenderClient via lazy init. after that,
       load encounter-specific terrain/objects/models (standalone viewer does
       this in run_visual before the render loop). */
    int first_call = (re->client == NULL);
    pvp_render(re);

    if (first_call) {
        RenderClient* rc = (RenderClient*)re->client;
        rc->ticks_per_second = env->ticks_per_second;
        rc->model_cache = model_cache_load("data/equipment.models");
        if (rc->model_cache) rc->show_models = 1;
        rc->anim_cache = anim_cache_load("data/equipment.anims");
        render_init_overlay_models(rc);
        rc->terrain = terrain_load("data/inferno.terrain");
        rc->objects = objects_load("data/inferno.objects");
        rc->objects_zuk = objects_load("data/inferno_zuk.objects");
        /* inferno region (35,83) starts at world (2246, 5315) */
        if (rc->terrain) terrain_offset(rc->terrain, 2246, 5315);
        if (rc->objects) objects_offset(rc->objects, 2246, 5315);
        if (rc->objects_zuk) objects_offset(rc->objects_zuk, 2246, 5315);
        rc->npc_model_cache = model_cache_load("data/inferno.models");
        rc->npc_anim_cache = anim_cache_load("data/inferno.anims");

        /* inferno renders in encounter-local tiles, but render_make_client()
           initializes the camera to wilderness PvP world coords. mirror the
           standalone viewer's post-load bootstrap so the first live frame uses
           inferno arena bounds and entity positions. */
        render_populate_entities(rc, re);
        rc->cam_target_x = (float)rc->arena_base_x + (float)rc->arena_width / 2.0f;
        rc->cam_target_z = -((float)rc->arena_base_y + (float)rc->arena_height / 2.0f);
        for (int i = 0; i < rc->entity_count; i++) {
            int size = rc->entities[i].npc_size > 1 ? rc->entities[i].npc_size : 1;
            rc->sub_x[i] = rc->entities[i].x * 128 + size * 64;
            rc->sub_y[i] = rc->entities[i].y * 128 + size * 64;
            rc->dest_x[i] = rc->sub_x[i];
            rc->dest_y[i] = rc->sub_y[i];
        }
        env->last_step_time = GetTime();
    }

    RenderClient* rc = (RenderClient*)re->client;
    if (!rc) return;

    /* post-reset cleanup: c_step raised the flag after resetting the encounter.
       mirror osrs_visual.c:186-201 so damage splats, in-flight effects, stale
       inventory state, and last-frame sub-tile coordinates (from dead-player
       body) don't leak into the next episode. */
    if (env->pending_render_reset) {
        render_clear_history(rc);
        effect_clear_all(rc->effects);
        gui_reset_inventory_ui_state(&rc->gui);
        render_populate_entities(rc, re);
        for (int i = 0; i < rc->entity_count; i++) {
            int size = rc->entities[i].npc_size > 1 ? rc->entities[i].npc_size : 1;
            rc->sub_x[i] = rc->entities[i].x * 128 + size * 64;
            rc->sub_y[i] = rc->entities[i].y * 128 + size * 64;
            rc->dest_x[i] = rc->sub_x[i];
            rc->dest_y[i] = rc->sub_y[i];
        }
        env->pending_render_reset = 0;
    }

    /* update NPC visual positions once per tick (not per frame).
       render_post_tick snaps sub_x/sub_y/dest_x/dest_y for spawned/moved NPCs
       and resets composite state on npc_slot changes. */
    render_populate_entities(rc, re);
    render_post_tick(rc, re);

    /* match the standalone viewer's visual_frame pattern: spin pvp_render at
       ~60fps until the next sim tick is due. pvp_render uses GetFrameTime()
       to accumulate client_tick_accumulator and steps render_client_tick every
       20ms, which is what animates sub_x/sub_y interpolation smoothly between
       sim ticks. calling pvp_render only once per 600ms collapses all 30
       client-ticks into a single frame → entities snap instantly, no motion.
       so: keep rendering until the tick deadline, then return. c_step's sim
       step follows immediately and the loop repeats. */
    float tps = rc->ticks_per_second > 0.0f ? rc->ticks_per_second : 1.667f;
    double interval = 1.0 / (double)tps;
    double deadline = env->last_step_time + interval;
    while (GetTime() < deadline) {
        pvp_render(re);
    }
    env->last_step_time = GetTime();
}

#define MY_VEC_INIT
#include "vecenv.h"

/* max episode length for action buffer (INF_MAX_TICKS from encounter) */
#define REPLAY_MAX_TICKS INF_MAX_TICKS

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->enc_state = ENCOUNTER_INFERNO.create();
    memset(&env->log, 0, sizeof(Log));

    DictItem* start_wave = dict_get_unsafe(kwargs, "start_wave");
    if (start_wave)
        ENCOUNTER_INFERNO.put_int(env->enc_state, "start_wave", (int)start_wave->value);
    ENCOUNTER_INFERNO.put_float(
        env->enc_state, "damage_reward_coeff",
        (float)dict_get_unsafe(kwargs, "damage_reward_coeff")->value);
    ENCOUNTER_INFERNO.put_float(
        env->enc_state, "shield_penalty_coeff",
        (float)dict_get_unsafe(kwargs, "shield_penalty_coeff")->value);
    ENCOUNTER_INFERNO.put_float(
        env->enc_state, "tag_reward_coeff",
        (float)dict_get_unsafe(kwargs, "tag_reward_coeff")->value);
    DictItem* supply_profile_scale =
        dict_get_unsafe(kwargs, "late_start_supply_profile_scale");
    if (supply_profile_scale) {
        ENCOUNTER_INFERNO.put_float(
            env->enc_state, "late_start_supply_profile_scale",
            (float)supply_profile_scale->value);
    }
    /* match the 1-indexed → 0-indexed conversion done by encounter's put_int */
    int sw = start_wave ? (int)start_wave->value : 0;
    env->config_start_wave = (sw > 0) ? sw - 1 : 0;

    const char* record_path = getenv("RECORD_REPLAY");
    const char* play_path = getenv("PLAY_REPLAY");
    if (record_path && record_path[0] && play_path && play_path[0]) {
        fprintf(stderr, "RECORD_REPLAY and PLAY_REPLAY cannot both be set\n");
        abort();
    }

    if (record_path && record_path[0]) {
        env->episode_actions = (int*)malloc(REPLAY_MAX_TICKS * NUM_ATNS * sizeof(int));
        if (!env->episode_actions) {
            fprintf(stderr, "RECORD_REPLAY: out of memory\n");
            abort();
        }
        env->episode_action_cap = REPLAY_MAX_TICKS;
    } else {
        env->episode_actions = NULL;
        env->episode_action_cap = 0;
    }
    env->episode_action_len = 0;

    /* playback: first env (env 0) loads the replay if PLAY_REPLAY is set.
       we use g_play_replay_loaded to ensure only one env loads it. */
    env->replay_actions = NULL;
    env->replay_num_ticks = 0;
    env->replay_cursor = 0;
    env->replay_rng_seed = 0;
    env->ticks_per_second = 1.667f;
    env->last_step_time = 0.0;
    static int g_play_replay_loaded = 0;
    if (play_path && play_path[0] && !g_play_replay_loaded) {
        FILE* fp = fopen(play_path, "rb");
        if (!fp) {
            fprintf(stderr, "PLAY_REPLAY: cannot open %s\n", play_path);
            abort();
        }
        int num_ticks = 0;
        uint32_t rng_seed = 0;
        if (fread(&num_ticks, sizeof(int), 1, fp) != 1 ||
            fread(&rng_seed, sizeof(uint32_t), 1, fp) != 1 ||
            num_ticks <= 0 || num_ticks > REPLAY_MAX_TICKS) {
            fprintf(stderr, "PLAY_REPLAY: invalid replay header in %s\n", play_path);
            fclose(fp);
            abort();
        }
        int* buf = (int*)malloc(num_ticks * NUM_ATNS * sizeof(int));
        if (!buf) {
            fprintf(stderr, "PLAY_REPLAY: out of memory\n");
            fclose(fp);
            abort();
        }
        if (fread(buf, sizeof(int), num_ticks * NUM_ATNS, fp) != (size_t)(num_ticks * NUM_ATNS)) {
            fprintf(stderr, "PLAY_REPLAY: short read from %s\n", play_path);
            free(buf);
            fclose(fp);
            abort();
        }
        fclose(fp);
        env->replay_actions = buf;
        env->replay_num_ticks = num_ticks;
        env->replay_rng_seed = rng_seed;
        g_play_replay_loaded = 1;
        fprintf(stderr, "PLAY_REPLAY: loaded %d ticks, rng=%u from %s\n",
                num_ticks, rng_seed, play_path);
        ENCOUNTER_INFERNO.reset(env->enc_state, rng_seed);
    }
}

/* curriculum wave mixing: start some agents at later waves for late-game gradient signal.
   base-start agents are scored normally; curriculum agents train but don't affect sweep metric. */
#define MAX_CURRICULUM_TIERS 4

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    DictItem* base_start_wave_item = dict_get_unsafe(env_kwargs, "start_wave");
    int base_start_wave = base_start_wave_item ? (int)base_start_wave_item->value : 0;

    /* parse curriculum tiers from env config */
    static const char* wave_keys[] = {
        "curriculum_wave_1","curriculum_wave_2","curriculum_wave_3","curriculum_wave_4"
    };
    static const char* frac_keys[] = {
        "curriculum_frac_1","curriculum_frac_2","curriculum_frac_3","curriculum_frac_4"
    };
    int curriculum_waves[MAX_CURRICULUM_TIERS];
    float curriculum_fracs[MAX_CURRICULUM_TIERS];
    int num_tiers = 0;
    for (int i = 0; i < MAX_CURRICULUM_TIERS; i++) {
        DictItem* w = dict_get_unsafe(env_kwargs, wave_keys[i]);
        DictItem* f = dict_get_unsafe(env_kwargs, frac_keys[i]);
        if (w && f && f->value > 0.0) {
            curriculum_waves[num_tiers] = (int)w->value;
            curriculum_fracs[num_tiers] = (float)f->value;
            num_tiers++;
        }
    }

    /* allocate and init all envs (same as default my_vec_init) */
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));
    int num_envs = 0;
    int agents_created = 0;
    while (agents_created < total_agents) {
        srand(num_envs);
        envs[num_envs].rng = num_envs;
        my_init(&envs[num_envs], env_kwargs);
        agents_created += envs[num_envs].num_agents;
        num_envs++;
    }
    envs = (Env*)realloc(envs, num_envs * sizeof(Env));

    /* assign curriculum start_waves to agents at the end of the array */
    if (num_tiers > 0) {
        int tier_counts[MAX_CURRICULUM_TIERS];
        int curriculum_total = 0;
        for (int t = 0; t < num_tiers; t++) {
            tier_counts[t] = (int)(curriculum_fracs[t] * num_envs);
            if (tier_counts[t] < 1) tier_counts[t] = 1;
            curriculum_total += tier_counts[t];
        }
        int base_count = num_envs - curriculum_total;
        int cursor = base_count;
        for (int t = 0; t < num_tiers; t++) {
            for (int i = 0; i < tier_counts[t] && cursor < num_envs; i++, cursor++) {
                ENCOUNTER_INFERNO.put_int(envs[cursor].enc_state,
                    "start_wave", curriculum_waves[t]);
            }
        }
        fprintf(stderr, "curriculum: %d wave-%d", base_count, base_start_wave);
        for (int t = 0; t < num_tiers; t++)
            fprintf(stderr, ", %d wave-%d", tier_counts[t], curriculum_waves[t]);
        fprintf(stderr, " (%d total)\n", num_envs);
    }

    /* fill buffer info (same as default) */
    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += envs[i].num_agents;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "damage_dealt", log->damage_dealt);
    dict_set(out, "damage_received", log->damage_received);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "wins", log->wins);
    dict_set(out, "wave", log->wave);
    dict_set(out, "idle_ticks", log->idle_ticks);
    dict_set(out, "brews_used", log->brews_used);
    dict_set(out, "blood_healed", log->blood_healed);

    /* prayer analysis: correct rate + unavoidable breakdown */
    float prayer_rate = (log->prayer_total > 0.0f)
        ? log->prayer_correct / log->prayer_total : 0.0f;
    dict_set(out, "prayer_correct_rate", prayer_rate);
    /* what fraction of off-prayer hits were unavoidable (multi-style same tick) */
    float off_prayer = log->prayer_total - log->prayer_correct;
    float unavoidable_rate = (off_prayer > 0.0f)
        ? log->unavoidable_off_prayer / off_prayer : 0.0f;
    dict_set(out, "unavoidable_off_prayer_rate", unavoidable_rate);
    //dict_set(out, "unavoidable_off_prayer", log->unavoidable_off_prayer);

    dict_set(out, "brews_remaining", log->brews_remaining);
    dict_set(out, "restores_remaining", log->restores_remaining);
    dict_set(out, "prayer_at_death", log->prayer_at_death);

    //dict_set(out, "npc_kills", log->npc_kills);
    //dict_set(out, "gear_switches", log->gear_switches);
    dict_set(out, "current_ranged", log->current_ranged);
    dict_set(out, "current_magic", log->current_magic);
    dict_set(out, "behind_shield_pct", log->behind_shield_pct);
    dict_set(out, "zuk_hp_remaining", log->zuk_hp_remaining);
    dict_set(out, "min_zuk_hp_seen", log->min_zuk_hp_seen);
    dict_set(out, "hp_restored", log->hp_restored);
    dict_set(out, "zuk_healer_damage", log->zuk_healer_damage);
    dict_set(out, "deaths_to_jad", log->killed_by_type[INF_NPC_JAD] / log->n);
    float gear_switch_rate = (log->episode_length > 0.0f)
        ? log->gear_switches / log->episode_length : 0.0f;
    dict_set(out, "gear_switch_rate", gear_switch_rate);

    float wr = log->wins;
    float score;
    int start_wave = (int)(log->start_wave + 0.5f);
    if (start_wave >= 68) {
        /* Zuk-only: score = fraction of lowest Zuk HP reached (0..1), wins = 1.0 */
        score = (1200.0f - log->min_zuk_hp_seen) / 1200.0f;
    } else {
        /* full runs: wave progress (0..0.5) + win bonus (0..1) */
        float wave_frac = log->wave / (float)INF_NUM_WAVES;
        score = wr + (1.0f - wr) * wave_frac * 0.5f;
    }
    dict_set(out, "score", score);

    /* per-NPC-type prayer rates and damage (wandb only).
       keys must be string literals — dict_set stores the pointer, not a copy. */
    /*
    static const char* pray_keys[] = {
        "pray_nibbler","pray_bat","pray_blob","pray_blob_mel","pray_blob_rng","pray_blob_mag",
        "pray_meleer","pray_ranger","pray_mager","pray_jad","pray_zuk","pray_heal_jad","pray_heal_zuk","pray_shield"
    };
    static const char* dmg_keys[] = {
        "dmg_from_nibbler","dmg_from_bat","dmg_from_blob","dmg_from_blob_mel","dmg_from_blob_rng","dmg_from_blob_mag",
        "dmg_from_meleer","dmg_from_ranger","dmg_from_mager","dmg_from_jad","dmg_from_zuk","dmg_from_heal_jad","dmg_from_heal_zuk","dmg_from_shield"
    };
    static const char* kill_keys[] = {
        "killed_by_nibbler","killed_by_bat","killed_by_blob","killed_by_blob_mel","killed_by_blob_rng","killed_by_blob_mag",
        "killed_by_meleer","killed_by_ranger","killed_by_mager","killed_by_jad","killed_by_zuk","killed_by_heal_jad","killed_by_heal_zuk","killed_by_shield"
    };
    for (int t = 0; t < INF_NUM_NPC_TYPES; t++) {
        if (log->attacks_by_type[t] > 0.0f) {
            dict_set(out, pray_keys[t], log->prayer_correct_by_type[t] / log->attacks_by_type[t]);
            dict_set(out, dmg_keys[t], log->dmg_from_type[t]);
        }
        if (log->killed_by_type[t] > 0.0f)
            dict_set(out, kill_keys[t], log->killed_by_type[t]);
    }
    */
}
