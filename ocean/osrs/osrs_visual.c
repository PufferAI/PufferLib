#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include "osrs_env.h"
#include "osrs_assets.h"
#include "osrs_encounter.h"
#include "osrs_binary_io.h"
#include "encounters/encounter_nh_pvp.h"
#include "encounters/encounter_zulrah.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "encounters/encounter_inferno.h"
#include "encounters/encounter_colosseum.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef OSRS_VISUAL
#include "osrs_render.h"
#include "osrs_render_scene.h"
#include "puffercpu.c"
#include "osrs_visual_net.h"

_Static_assert(OSRS_SHARED_OBS_SIZE == 101,
    "entity encoder shared observation prefix");
_Static_assert(OSRS_ENT_INV_START == OSRS_SHARED_OBS_INVENTORY_START,
    "entity encoder inventory offset");
_Static_assert(OSRS_ENT_INV_NUM_RECS == OSRS_INVENTORY_SIZE,
    "entity encoder inventory record count");
_Static_assert(OSRS_ENT_INV_OBS_FEATS == OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES,
    "entity encoder inventory observation record width");
_Static_assert(
    OSRS_ENT_INV_START + OSRS_ENT_INV_NUM_RECS * OSRS_ENT_INV_OBS_FEATS ==
        OSRS_SHARED_OBS_EQUIPPED_START,
    "entity encoder inventory boundary");
_Static_assert(OSRS_ENT_EQUIPPED_START == OSRS_SHARED_OBS_EQUIPPED_START,
    "entity encoder equipped offset");
_Static_assert(OSRS_ENT_EQUIPPED_NUM_RECS == NUM_GEAR_SLOTS,
    "entity encoder equipped record count");
_Static_assert(
    OSRS_ENT_EQUIPPED_START +
            OSRS_ENT_EQUIPPED_NUM_RECS * OSRS_ENT_EQUIPPED_OBS_FEATS ==
        OSRS_SHARED_OBS_EFFECT_START,
    "entity encoder equipped boundary");
_Static_assert(OSRS_ENT_ITEM_FEATS == OSRS_ITEM_OBS_TABLE_COLS,
    "entity encoder expanded item record width");
_Static_assert(OSRS_ITEM_CONTENT_COUNT <= OSRS_ITEM_OBS_CODE_SCALE,
    "entity encoder item code scale");

_Static_assert(COLO_ENT_OBS_SIZE == COLO_NUM_OBS,
    "Colosseum entity encoder observation width");
_Static_assert(COLO_ENT_NPC_START == COLO_OBS_AFTER_SHARED,
    "Colosseum entity encoder NPC offset");
_Static_assert(COLO_ENT_NPC_NUM_RECS == COLO_OBS_NPCS,
    "Colosseum entity encoder NPC count");
_Static_assert(COLO_ENT_NPC_OBS_FEATS == COLO_FEATURES_PER_NPC,
    "Colosseum entity encoder NPC observation record width");
_Static_assert(
    COLO_ENT_NPC_START +
            COLO_ENT_NPC_NUM_RECS * COLO_ENT_NPC_OBS_FEATS ==
        COLO_OBS_AFTER_NPCS,
    "Colosseum entity encoder NPC boundary");
_Static_assert(
    COLO_ENT_NPC_FEATS ==
        COLO_ENT_NPC_TYPE_ONEHOT +
            (COLO_FEATURES_PER_NPC - COLO_NPC_TYPE_CODE_FEATURES),
    "Colosseum entity encoder expanded NPC record width");
_Static_assert(COLO_ENT_NPC_TYPE_ONEHOT == COLO_NUM_NPC_TYPES,
    "Colosseum entity encoder NPC type width");

_Static_assert(INF_ENT_OBS_SIZE == INF_NUM_OBS,
    "Inferno entity encoder observation width");
_Static_assert(INF_ENT_NPC_START == INF_OBS_AFTER_PILLARS,
    "Inferno entity encoder NPC offset");
_Static_assert(INF_ENT_NPC_NUM_RECS == INF_OBS_NPCS,
    "Inferno entity encoder NPC count");
_Static_assert(INF_ENT_NPC_OBS_FEATS == INF_NPC_SLOT_FEATURES,
    "Inferno entity encoder NPC observation record width");
_Static_assert(
    INF_ENT_NPC_START +
            INF_ENT_NPC_NUM_RECS * INF_ENT_NPC_OBS_FEATS ==
        INF_OBS_AFTER_NPCS,
    "Inferno entity encoder NPC boundary");
_Static_assert(
    INF_ENT_NPC_FEATS ==
        INF_ENT_NPC_TYPE_ONEHOT + (INF_NPC_SLOT_FEATURES - 1),
    "Inferno entity encoder expanded NPC record width");
_Static_assert(INF_ENT_NPC_TYPE_ONEHOT == INF_NUM_NPC_TYPES,
    "Inferno entity encoder NPC type width");
_Static_assert(INF_ENT_NPC_TYPE_SCALE == (int)INF_NPC_TYPE_CODE_SCALE,
    "Inferno entity encoder NPC type scale");

#define VISUAL_SHARED_ITEM_BRANCH(name, offset, count) { \
    .weight_name = name, \
    .start = offset, \
    .num_recs = count, \
    .feats = OSRS_ENT_ITEM_FEATS, \
    .obs_feats = 1, \
    .type_onehot = 0, \
    .code_scale = OSRS_ITEM_OBS_CODE_SCALE, \
    .bottleneck = OSRS_ENT_ITEM_BOTTLENECK, \
    .active_width = 1, \
    .expansion = ENTITY_RECORD_ITEM_TABLE, \
}

static const EntityEncoderDescriptor VISUAL_ENTITY_ENCODER_DESCRIPTORS[] = {
    {
        .env_name = "colosseum",
        .obs_size = COLO_ENT_OBS_SIZE,
        .num_branches = 3,
        .branches = {
            VISUAL_SHARED_ITEM_BRANCH(
                "inventory", OSRS_ENT_INV_START, OSRS_ENT_INV_NUM_RECS),
            VISUAL_SHARED_ITEM_BRANCH(
                "equipped", OSRS_ENT_EQUIPPED_START, OSRS_ENT_EQUIPPED_NUM_RECS),
            {
                .weight_name = "npc",
                .start = COLO_ENT_NPC_START,
                .num_recs = COLO_ENT_NPC_NUM_RECS,
                .feats = COLO_ENT_NPC_FEATS,
                .obs_feats = COLO_ENT_NPC_OBS_FEATS,
                .type_onehot = COLO_ENT_NPC_TYPE_ONEHOT,
                .code_scale = COLO_ENT_NPC_TYPE_SCALE,
                .bottleneck = COLO_ENT_NPC_BOTTLENECK,
                .active_width = COLO_ENT_NPC_TYPE_ONEHOT,
                .expansion = ENTITY_RECORD_TYPE_ONEHOT,
            },
        },
    },
    {
        .env_name = "inferno",
        .obs_size = INF_ENT_OBS_SIZE,
        .num_branches = 3,
        .branches = {
            VISUAL_SHARED_ITEM_BRANCH(
                "inventory", OSRS_ENT_INV_START, OSRS_ENT_INV_NUM_RECS),
            VISUAL_SHARED_ITEM_BRANCH(
                "equipped", OSRS_ENT_EQUIPPED_START, OSRS_ENT_EQUIPPED_NUM_RECS),
            {
                .weight_name = "npc",
                .start = INF_ENT_NPC_START,
                .num_recs = INF_ENT_NPC_NUM_RECS,
                .feats = INF_ENT_NPC_FEATS,
                .obs_feats = INF_ENT_NPC_OBS_FEATS,
                .type_onehot = INF_ENT_NPC_TYPE_ONEHOT,
                .code_scale = INF_ENT_NPC_TYPE_SCALE,
                .bottleneck = INF_ENT_NPC_BOTTLENECK,
                .active_width = INF_ENT_NPC_TYPE_ONEHOT,
                .expansion = ENTITY_RECORD_TYPE_ONEHOT,
            },
        },
    },
    {
        .env_name = "zulrah",
        .obs_size = ZUL_NUM_OBS,
        .num_branches = 2,
        .branches = {
            VISUAL_SHARED_ITEM_BRANCH(
                "inventory", OSRS_ENT_INV_START, OSRS_ENT_INV_NUM_RECS),
            VISUAL_SHARED_ITEM_BRANCH(
                "equipped", OSRS_ENT_EQUIPPED_START, OSRS_ENT_EQUIPPED_NUM_RECS),
        },
    },
    {
        .env_name = "nh_pvp",
        .obs_size = 133,
        .num_branches = 2,
        .branches = {
            VISUAL_SHARED_ITEM_BRANCH(
                "inventory", OSRS_ENT_INV_START, OSRS_ENT_INV_NUM_RECS),
            VISUAL_SHARED_ITEM_BRANCH(
                "equipped", OSRS_ENT_EQUIPPED_START, OSRS_ENT_EQUIPPED_NUM_RECS),
        },
    },
};

static const EntityEncoderDescriptor* visual_policy_entity_descriptor(
        const EncounterDef* edef) {
    for (int i = 0;
            i < (int)(sizeof(VISUAL_ENTITY_ENCODER_DESCRIPTORS) /
                sizeof(VISUAL_ENTITY_ENCODER_DESCRIPTORS[0]));
            i++) {
        const EntityEncoderDescriptor* descriptor =
            &VISUAL_ENTITY_ENCODER_DESCRIPTORS[i];
        if (strcmp(edef->name, descriptor->env_name) != 0) continue;
        if (descriptor->obs_size > 0 && descriptor->obs_size != edef->obs_size) {
            fprintf(stderr,
                "policy: %s observation width %d != descriptor width %d\n",
                edef->name, edef->obs_size, descriptor->obs_size);
            abort();
        }
        return descriptor;
    }
    return NULL;
}

#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#include <pthread.h>
#endif


static void print_player_state(Player* p, int idx) {
    printf("Player %d: HP=%d/%d Prayer=%d Gear=%d Pos=(%d,%d) Frozen=%d\n",
           idx, p->current_hitpoints, p->base_hitpoints,
           p->current_prayer, p->current_gear, p->x, p->y, p->frozen_ticks);
}

static void print_env_state(OsrsEnv* env) {
    printf("\n=== Tick %d ===\n", env->tick);
    print_player_state(&env->players[0], 0);
    print_player_state(&env->players[1], 1);
    printf("PID holder: %d\n", env->pid_holder);
}

static void run_random_episode(OsrsEnv* env, int verbose) {
    const EncounterArenaTopology* route_topology =
        pvp_route_topology_finalize((const CollisionMap*)env->collision_map);
    OsrsActorRouteCache route_cache[NUM_AGENTS] = {0};
    pvp_reset(env, route_topology);

    while (!env->episode_over) {
        for (int agent = 0; agent < NUM_AGENTS; agent++) {
            int* actions =
                env->actions + agent * OSRS_BASE_NUM_ACTION_HEADS;
            for (int h = 0; h < OSRS_BASE_NUM_ACTION_HEADS; h++) {
                actions[h] = rand() % NH_PVP_ACTION_DIMS[h];
            }
        }

        pvp_step(env, route_topology, route_cache);

        if (verbose && env->tick % 50 == 0) {
            print_env_state(env);
        }
    }

    if (verbose) {
        printf("\n=== Episode End ===\n");
        printf("Winner: Player %d\n", env->winner);
        printf("Length: %d ticks\n", env->tick);
        printf("P0 damage dealt: %.0f\n", env->players[0].total_damage_dealt);
        printf("P1 damage dealt: %.0f\n", env->players[1].total_damage_dealt);
    }
}

static void benchmark(OsrsEnv* env, int num_steps) {
    const EncounterArenaTopology* route_topology =
        pvp_route_topology_finalize((const CollisionMap*)env->collision_map);
    OsrsActorRouteCache route_cache[NUM_AGENTS] = {0};
    printf("Benchmarking %d steps...\n", num_steps);

    clock_t start = clock();
    int episodes = 0;
    int total_steps = 0;

    while (total_steps < num_steps) {
        pvp_reset(env, route_topology);
        pvp_actor_route_caches_clear(route_cache);
        episodes++;

        while (!env->episode_over && total_steps < num_steps) {
            for (int agent = 0; agent < NUM_AGENTS; agent++) {
                int* actions =
                    env->actions + agent * OSRS_BASE_NUM_ACTION_HEADS;
                for (int h = 0; h < OSRS_BASE_NUM_ACTION_HEADS; h++) {
                    actions[h] = rand() % NH_PVP_ACTION_DIMS[h];
                }
            }

            pvp_step(env, route_topology, route_cache);
            total_steps++;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Results:\n");
    printf("  Total steps: %d\n", total_steps);
    printf("  Episodes: %d\n", episodes);
    printf("  Time: %.3f seconds\n", elapsed);
    printf("  Steps/sec: %.0f\n", total_steps / elapsed);
    printf("  Avg episode length: %.1f ticks\n", (float)total_steps / episodes);
}

static EncounterContext* visual_create_encounter_context(const EncounterDef* edef) {
    if (!edef || edef->context_size == 0)
        return NULL;
    EncounterContext* context = (EncounterContext*)calloc(1, edef->context_size);
    if (!context) abort();
    if (edef->init_context)
        edef->init_context(context);
    return context;
}

static void visual_destroy_encounter_context(
    const EncounterDef* edef,
    EncounterContext** context
) {
    if (!context || !*context)
        return;
    if (edef && edef->destroy_context)
        edef->destroy_context(*context);
    free(*context);
    *context = NULL;
}

static double osrs_profile_now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static uint64_t osrs_profile_hash_bytes(uint64_t hash, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++)
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
}

#ifdef COLO_PROFILE_ENABLED
static int osrs_colosseum_profile_slot_is_counter(int slot) {
    return slot == COLO_PROF_BEST_GEAR_REQUESTS ||
        slot == COLO_PROF_BEST_GEAR_HITS ||
        slot == COLO_PROF_BEST_GEAR_BUILDS ||
        slot == COLO_PROF_VENATOR_REQUESTS ||
        slot == COLO_PROF_VENATOR_HITS ||
        slot == COLO_PROF_VENATOR_REFRESHES;
}

static void osrs_print_colosseum_profile_results(int total_steps) {
    int count = colosseum_env_profile_count();
    if (count <= 0) return;
    double values[COLO_PROF_COUNT];
    int order[COLO_PROF_COUNT];
    int order_count = 0;
    for (int i = 0; i < count; i++) {
        values[i] = colosseum_env_profile_read_reset_ms(i);
        if (!osrs_colosseum_profile_slot_is_counter(i))
            order[order_count++] = i;
    }
    for (int i = 0; i < order_count; i++) {
        int best = i;
        for (int j = i + 1; j < order_count; j++) {
            if (values[order[j]] > values[order[best]]) best = j;
        }
        int tmp = order[i];
        order[i] = order[best];
        order[best] = tmp;
    }
    double total = values[COLO_PROF_C_STEP_TOTAL];
    printf("Colosseum profile buckets:\n");
    for (int r = 0; r < order_count; r++) {
        int slot = order[r];
        double pct = total > 0.0 ? 100.0 * values[slot] / total : 0.0;
        printf("  %-28s %.3f ms  %.2f%%\n",
            colosseum_env_profile_name(slot), values[slot], pct);
    }
    double steps = total_steps > 0 ? (double)total_steps : 1.0;
    double best_gear_requests = values[COLO_PROF_BEST_GEAR_REQUESTS];
    double best_gear_hits = values[COLO_PROF_BEST_GEAR_HITS];
    printf("Colosseum cache counters:\n");
    printf("  %-28s %.0f total  %.6f per step  %.2f%% hit\n",
        "best_gear",
        best_gear_requests,
        best_gear_requests / steps,
        best_gear_requests > 0.0 ? 100.0 * best_gear_hits / best_gear_requests : 0.0);
    double venator_requests = values[COLO_PROF_VENATOR_REQUESTS];
    double venator_hits = values[COLO_PROF_VENATOR_HITS];
    printf("  %-28s %.0f total  %.6f per step  %.2f%% hit\n",
        "venator",
        venator_requests,
        venator_requests / steps,
        venator_requests > 0.0 ? 100.0 * venator_hits / venator_requests : 0.0);
    printf("  %-28s %.0f total  %.6f per step\n",
        colosseum_env_profile_name(COLO_PROF_BEST_GEAR_BUILDS),
        values[COLO_PROF_BEST_GEAR_BUILDS],
        values[COLO_PROF_BEST_GEAR_BUILDS] / steps);
    printf("  %-28s %.0f total  %.6f per step\n",
        colosseum_env_profile_name(COLO_PROF_VENATOR_REFRESHES),
        values[COLO_PROF_VENATOR_REFRESHES],
        values[COLO_PROF_VENATOR_REFRESHES] / steps);
}
#endif

#ifdef INF_PROFILE_ENABLED
static int osrs_inferno_profile_slot_is_counter(int slot) {
    return slot == INF_PROF_FORECAST_CALLS ||
        slot == INF_PROF_FORECAST_VALID_ACTIONS ||
        slot == INF_PROF_FORECAST_DISTINCT_LANDINGS;
}

static void osrs_print_inferno_profile_results(int total_steps) {
    int count = inferno_env_profile_count();
    if (count <= 0) return;
    double values[INF_PROF_COUNT];
    int order[INF_PROF_COUNT];
    int order_count = 0;
    for (int i = 0; i < count; i++) {
        values[i] = inferno_env_profile_read_reset_ms(i);
        if (!osrs_inferno_profile_slot_is_counter(i))
            order[order_count++] = i;
    }
    for (int i = 0; i < order_count; i++) {
        int best = i;
        for (int j = i + 1; j < order_count; j++) {
            if (values[order[j]] > values[order[best]]) best = j;
        }
        int tmp = order[i];
        order[i] = order[best];
        order[best] = tmp;
    }
    double total = values[INF_PROF_C_STEP_TOTAL];
    printf("Inferno profile buckets:\n");
    for (int r = 0; r < order_count; r++) {
        int slot = order[r];
        double pct = total > 0.0 ? 100.0 * values[slot] / total : 0.0;
        printf("  %-28s %.3f ms  %.2f%%\n",
            inferno_env_profile_name(slot), values[slot], pct);
    }

    double calls = values[INF_PROF_FORECAST_CALLS];
    double valid_actions = values[INF_PROF_FORECAST_VALID_ACTIONS];
    double distinct_landings = values[INF_PROF_FORECAST_DISTINCT_LANDINGS];
    double steps = total_steps > 0 ? (double)total_steps : 1.0;
    double call_divisor = calls > 0.0 ? calls : 1.0;
    printf("Inferno forecast counters:\n");
    printf("  %-28s %.0f total  %.3f per step\n",
        inferno_env_profile_name(INF_PROF_FORECAST_CALLS),
        calls,
        calls / steps);
    printf("  %-28s %.0f total  %.3f per step  %.3f per forecast\n",
        inferno_env_profile_name(INF_PROF_FORECAST_VALID_ACTIONS),
        valid_actions,
        valid_actions / steps,
        valid_actions / call_divisor);
    printf("  %-28s %.0f total  %.3f per step  %.3f per forecast\n",
        inferno_env_profile_name(INF_PROF_FORECAST_DISTINCT_LANDINGS),
        distinct_landings,
        distinct_landings / steps,
        distinct_landings / call_divisor);
}
#endif


static const EncounterDef* visual_open_encounter(OsrsEnv* env, const char* encounter_name) {
    const EncounterDef* edef = encounter_find(encounter_name);
    if (!edef) {
        fprintf(stderr, "unknown encounter: %s\n", encounter_name);
        return NULL;
    }
    env->encounter_def = (void*)edef;
    env->encounter_state = edef->create();
    env->encounter_context = visual_create_encounter_context(edef);
    return edef;
}
static void visual_finalize_encounter(
    const EncounterDef* edef,
    OsrsEnv* env
) {
    if (edef->finalize_context)
        edef->finalize_context(env->encounter_state, env->encounter_context);
}

static void run_profile(
    OsrsEnv* env,
    const char* encounter_name,
    int start_wave,
    int profile_steps,
    uint32_t profile_seed
) {
    srand(profile_seed);
    if (profile_steps > 0) {
        printf("Profiling %s for %d steps...\n",
            encounter_name ? encounter_name : "pvp",
            profile_steps);
    } else {
        printf("Profiling %s for 10 seconds...\n", encounter_name ? encounter_name : "pvp");
    }

    const EncounterArenaTopology* direct_pvp_topology = NULL;
    if (encounter_name) {
        const EncounterDef* edef = visual_open_encounter(env, encounter_name);
        if (!edef) return;

        visual_load_encounter_collision_map(edef, env, encounter_name);
        if (start_wave >= 0 && edef->put_int) {
            edef->put_int(
                env->encounter_state,
                env->encounter_context,
                "start_wave",
                start_wave);
            fprintf(stderr, "start_wave: %d\n", start_wave);
        }
        visual_finalize_encounter(edef, env);
        edef->reset(env->encounter_state, env->encounter_context, profile_seed);
    } else {
        env->pvp_runtime.use_c_opponent = 1;
        env->pvp_runtime.opponent.type = OPP_IMPROVED;
        env->has_rng_seed = 1;
        env->rng_seed = profile_seed;
        env->is_lms = 1;
        direct_pvp_topology =
            pvp_route_topology_finalize(
                (const CollisionMap*)env->collision_map);
        pvp_reset(env, direct_pvp_topology);
    }
    OsrsActorRouteCache direct_pvp_route_cache[NUM_AGENTS] = {0};

    const EncounterDef* profile_edef = (const EncounterDef*)env->encounter_def;
    float* encounter_obs = NULL;
    if (profile_edef) {
        encounter_obs = (float*)calloc(
            (size_t)(profile_edef->obs_size + profile_edef->mask_size),
            sizeof(float));
        if (!encounter_obs) abort();
        profile_edef->write_obs(
            env->encounter_state,
            (EncounterContext*)env->encounter_context,
            encounter_obs);
        profile_edef->write_mask(
            env->encounter_state,
            (EncounterContext*)env->encounter_context,
            encounter_obs + profile_edef->obs_size);
#ifdef COLO_PROFILE_ENABLED
        if (strcmp(profile_edef->name, "colosseum") == 0) {
            int count = colosseum_env_profile_count();
            for (int i = 0; i < count; i++)
                (void)colosseum_env_profile_read_reset_ms(i);
        }
#endif
#ifdef INF_PROFILE_ENABLED
        if (strcmp(profile_edef->name, "inferno") == 0) {
            int count = inferno_env_profile_count();
            for (int i = 0; i < count; i++)
                (void)inferno_env_profile_read_reset_ms(i);
        }
#endif
    }

    double start = osrs_profile_now_seconds();
    double elapsed = 0;
    int total_steps = 0;
    int enc_actions[64] = {0};
    const int pin_inventory_actions = getenv("OSRS_PROFILE_PIN_INV") != NULL;

    while ((profile_steps > 0 && total_steps < profile_steps) ||
           (profile_steps <= 0 && elapsed < 10.0)) {
        if (env->encounter_def && env->encounter_state) {
            const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
#ifdef COLO_PROFILE_ENABLED
            int col_profile_this_step = strcmp(edef->name, "colosseum") == 0;
            int col_prof_enabled = col_profile_this_step ? COLO_PROFILE_ENABLED() : 0;
            double col_prof_total_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
            double col_prof_t0 = col_prof_total_t0;
#endif
#ifdef INF_PROFILE_ENABLED
            int inf_profile_this_step = strcmp(edef->name, "inferno") == 0;
            int inf_prof_enabled = inf_profile_this_step ? INF_PROFILE_ENABLED() : 0;
            double inf_prof_total_t0 = inf_prof_enabled ? INF_PROFILE_NOW_MS() : 0.0;
            double inf_prof_t0 = inf_prof_total_t0;
#endif
            for (int h = 0; h < edef->num_action_heads; h++) {
                enc_actions[h] = rand() % edef->action_head_dims[h];
            }
            if (pin_inventory_actions) {
                for (int h = 2; h < 15 && h < edef->num_action_heads; h++)
                    enc_actions[h] = 0;
            }
#ifdef COLO_PROFILE_ENABLED
            COLO_PROFILE_MARK(COLO_PROF_C_ACTIONS);
#endif
#ifdef INF_PROFILE_ENABLED
            INF_PROFILE_MARK(INF_PROF_C_ACTIONS);
#endif
            edef->step(env->encounter_state, env->encounter_context, enc_actions);
#ifdef COLO_PROFILE_ENABLED
            COLO_PROFILE_MARK(COLO_PROF_C_ENCOUNTER_STEP);
#endif
#ifdef INF_PROFILE_ENABLED
            INF_PROFILE_MARK(INF_PROF_C_ENCOUNTER_STEP);
#endif
            edef->write_obs(
                env->encounter_state,
                (EncounterContext*)env->encounter_context,
                encounter_obs);
#ifdef COLO_PROFILE_ENABLED
            COLO_PROFILE_MARK(COLO_PROF_C_WRITE_OBS);
#endif
#ifdef INF_PROFILE_ENABLED
            INF_PROFILE_MARK(INF_PROF_C_WRITE_OBS);
#endif
            edef->write_mask(
                env->encounter_state,
                (EncounterContext*)env->encounter_context,
                encounter_obs + edef->obs_size);
#ifdef COLO_PROFILE_ENABLED
            COLO_PROFILE_MARK(COLO_PROF_C_WRITE_MASK);
#endif
#ifdef INF_PROFILE_ENABLED
            INF_PROFILE_MARK(INF_PROF_C_WRITE_MASK);
#endif
            (void)edef->get_reward(
                env->encounter_state,
                (EncounterContext*)env->encounter_context);
            if (edef->is_terminal(env->encounter_state, env->encounter_context)) {
#ifdef COLO_PROFILE_ENABLED
                COLO_PROFILE_MARK(COLO_PROF_C_REWARD_TERMINAL);
                COLO_PROFILE_MARK(COLO_PROF_C_TERMINAL_LOG);
#endif
#ifdef INF_PROFILE_ENABLED
                INF_PROFILE_MARK(INF_PROF_C_REWARD_TERMINAL);
                INF_PROFILE_MARK(INF_PROF_C_TERMINAL_LOG);
#endif
                edef->reset(
                    env->encounter_state, env->encounter_context, (uint32_t)rand());
                edef->write_obs(
                    env->encounter_state,
                    (EncounterContext*)env->encounter_context,
                    encounter_obs);
                edef->write_mask(
                    env->encounter_state,
                    (EncounterContext*)env->encounter_context,
                    encounter_obs + edef->obs_size);
#ifdef COLO_PROFILE_ENABLED
                COLO_PROFILE_MARK(COLO_PROF_C_RESET);
#endif
#ifdef INF_PROFILE_ENABLED
                INF_PROFILE_MARK(INF_PROF_C_RESET);
#endif
            } else {
#ifdef COLO_PROFILE_ENABLED
                COLO_PROFILE_MARK(COLO_PROF_C_REWARD_TERMINAL);
#endif
#ifdef INF_PROFILE_ENABLED
                INF_PROFILE_MARK(INF_PROF_C_REWARD_TERMINAL);
#endif
            }
#ifdef COLO_PROFILE_ENABLED
            if (col_prof_enabled)
                COLO_PROFILE_ADD(
                    COLO_PROF_C_STEP_TOTAL,
                    COLO_PROFILE_NOW_MS() - col_prof_total_t0);
#endif
#ifdef INF_PROFILE_ENABLED
            if (inf_prof_enabled)
                INF_PROFILE_ADD(
                    INF_PROF_C_STEP_TOTAL,
                    INF_PROFILE_NOW_MS() - inf_prof_total_t0);
#endif
        } else {
            for (int agent = 0; agent < NUM_AGENTS; agent++) {
                int* actions =
                    env->actions + agent * OSRS_BASE_NUM_ACTION_HEADS;
                for (int h = 0; h < OSRS_BASE_NUM_ACTION_HEADS; h++) {
                    actions[h] = rand() % NH_PVP_ACTION_DIMS[h];
                }
            }
            pvp_step(env, direct_pvp_topology, direct_pvp_route_cache);
            if (env->episode_over) {
                pvp_reset(env, direct_pvp_topology);
                pvp_actor_route_caches_clear(direct_pvp_route_cache);
            }
        }

        total_steps++;
        if (total_steps % 1000 == 0) {
            elapsed = osrs_profile_now_seconds() - start;
        }
    }
    elapsed = osrs_profile_now_seconds() - start;

    printf("Results:\n");
    printf("  Total steps: %d\n", total_steps);
    printf("  Time: %.3f seconds\n", elapsed);
    printf("  Steps/sec: %.0f\n", total_steps / elapsed);
#ifdef OSRS_ROUTE_PROBE
    printf(
        "  Route probes: tile=%llu cardinal=%llu attack=%llu direct=%llu overlap=%llu try_direct=%llu source=%llu reverse=%llu bfs=%llu nodes=%llu\n",
        (unsigned long long)osrs_route_probe_calls[0],
        (unsigned long long)osrs_route_probe_calls[1],
        (unsigned long long)osrs_route_probe_calls[2],
        (unsigned long long)osrs_route_probe_direct,
        (unsigned long long)osrs_route_probe_overlap,
        (unsigned long long)osrs_route_probe_try_direct,
        (unsigned long long)osrs_route_probe_source,
        (unsigned long long)osrs_route_probe_reverse,
        (unsigned long long)osrs_route_probe_bfs,
        (unsigned long long)osrs_route_probe_nodes);
    printf(
        "  Source fields: builds=%llu nodes=%llu\n",
        (unsigned long long)osrs_route_probe_source_builds,
        (unsigned long long)osrs_route_probe_source_nodes);
    printf(
        "  Route cost calls: osrs=%llu south=%llu direct=%llu south_bfs=%llu reverse=%llu target_bfs=%llu\n",
        (unsigned long long)osrs_route_probe_cost_calls[0],
        (unsigned long long)osrs_route_probe_cost_calls[1],
        (unsigned long long)osrs_route_probe_cost_calls[2],
        (unsigned long long)osrs_route_probe_cost_calls[3],
        (unsigned long long)osrs_route_probe_cost_calls[4],
        (unsigned long long)osrs_route_probe_cost_calls[5]);
    printf(
        "  Route cost BFS: osrs=%llu south=%llu direct=%llu south_bfs=%llu reverse=%llu target_bfs=%llu\n",
        (unsigned long long)osrs_route_probe_cost_bfs[0],
        (unsigned long long)osrs_route_probe_cost_bfs[1],
        (unsigned long long)osrs_route_probe_cost_bfs[2],
        (unsigned long long)osrs_route_probe_cost_bfs[3],
        (unsigned long long)osrs_route_probe_cost_bfs[4],
        (unsigned long long)osrs_route_probe_cost_bfs[5]);
#endif
#ifdef COLO_PROFILE_ENABLED
    if (encounter_name && strcmp(encounter_name, "colosseum") == 0)
        osrs_print_colosseum_profile_results(total_steps);
#endif
#ifdef INF_PROFILE_ENABLED
    if (encounter_name && strcmp(encounter_name, "inferno") == 0)
        osrs_print_inferno_profile_results(total_steps);
#endif

    if (env->encounter_def && env->encounter_state) {
        ((const EncounterDef*)env->encounter_def)->destroy(env->encounter_state);
        env->encounter_state = NULL;
        visual_destroy_encounter_context(
            (const EncounterDef*)env->encounter_def,
            (EncounterContext**)&env->encounter_context);
    }
    free(encounter_obs);
}

#ifdef OSRS_VISUAL
/* replay file: binary format for pre-recorded actions.
   header: [int32 num_ticks] [uint32 rng_state], then num_ticks * num_heads int32 values. */
typedef struct {
    int* actions;
    int  num_ticks;
    int  num_heads;
    int  current_tick;
    uint32_t rng_seed;
    void* initial_snapshot;
    size_t initial_snapshot_size;
} ReplayFile;

static ReplayFile* replay_load(const char* path, int num_heads, size_t snapshot_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "replay: can't open %s\n", path);
        abort();
    }
    int num_ticks = 0;
    uint32_t rng_seed = 12345;
    osrs_read_exact(f, &num_ticks, 4, 1, path, "replay tick count");
    osrs_read_exact(f, &rng_seed, 4, 1, path, "replay rng seed");
    if (num_ticks < 0 || num_heads <= 0) {
        fprintf(stderr, "replay: invalid shape ticks=%d heads=%d\n",
            num_ticks, num_heads);
        abort();
    }
    if ((size_t)num_ticks > SIZE_MAX / (size_t)num_heads) {
        fprintf(stderr, "replay: action count overflow ticks=%d heads=%d\n",
            num_ticks, num_heads);
        abort();
    }
    ReplayFile* rf = (ReplayFile*)osrs_calloc_or_abort(
        1, sizeof(ReplayFile), "replay file");
    rf->num_ticks = num_ticks;
    rf->num_heads = num_heads;
    rf->current_tick = 0;
    rf->rng_seed = rng_seed;
    size_t action_count = (size_t)num_ticks * (size_t)num_heads;
    rf->actions = (int*)osrs_malloc_or_abort(
        action_count * sizeof(int), "replay actions");
    osrs_read_exact(f, rf->actions, sizeof(int), action_count, path, "replay actions");
    long payload_end = ftell(f);
    if (payload_end < 0) {
        fprintf(stderr, "replay: ftell failed for %s\n", path);
        abort();
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "replay: seek failed for %s\n", path);
        abort();
    }
    long file_end = ftell(f);
    if (file_end < 0 || fseek(f, payload_end, SEEK_SET) != 0) {
        fprintf(stderr, "replay: seek failed for %s\n", path);
        abort();
    }
    long remaining = file_end - payload_end;
    if (remaining > 0) {
        if (snapshot_size == 0 || remaining != (long)snapshot_size) {
            fprintf(stderr,
                "replay: unexpected trailing bytes in %s: got %ld, expected %zu\n",
                path, remaining, snapshot_size);
            abort();
        }
        rf->initial_snapshot = osrs_malloc_or_abort(snapshot_size, "replay snapshot");
        rf->initial_snapshot_size = snapshot_size;
        osrs_read_exact(f, rf->initial_snapshot, 1, snapshot_size, path, "replay snapshot");
    }
    fclose(f);
    fprintf(stderr, "replay loaded: %d ticks, rng=%u from %s\n", num_ticks, rng_seed, path);
    return rf;
}

static int replay_get_actions(ReplayFile* rf, int* out) {
    if (rf->current_tick >= rf->num_ticks) return 0;
    int base = rf->current_tick * rf->num_heads;
    for (int h = 0; h < rf->num_heads; h++) out[h] = rf->actions[base + h];
    rf->current_tick++;
    return 1;
}

static void __attribute__((unused)) replay_free(ReplayFile* rf) {
    if (rf) { free(rf->actions); free(rf->initial_snapshot); free(rf); }
}

#define VISUAL_POLICY_MAX_ACTION_HEADS 64

typedef enum {
    VISUAL_POLICY_NONE = 0,
    VISUAL_POLICY_SAMPLE = 1,
    VISUAL_POLICY_ARGMAX = 2,
} VisualPolicyMode;

typedef struct {
    int input_size;
    int decoder_value_heads;
    int entity_encoder;
    int hidden_size;
    int num_layers;
} VisualPolicyModelShape;

typedef struct {
    int enabled;
    VisualPolicyMode mode;
    uint32_t rng_state;
    Weights* weights;
    VisualNet* net;
    float* obs;
    int obs_size;
    int mask_size;
    int action_dims[VISUAL_POLICY_MAX_ACTION_HEADS];
    int num_action_heads;
} VisualPolicy;

static uint32_t visual_policy_parse_seed(const char* value) {
    errno = 0;
    char* end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (errno || !end || *end != '\0' || parsed > UINT32_MAX) {
        fprintf(stderr, "policy: invalid policy seed: %s\n", value);
        abort();
    }
    return (uint32_t)parsed;
}

static VisualPolicyMode visual_policy_parse_mode(const char* value) {
    if (!value || strcmp(value, "sample") == 0) return VISUAL_POLICY_SAMPLE;
    if (strcmp(value, "argmax") == 0) return VISUAL_POLICY_ARGMAX;
    fprintf(stderr, "policy: invalid policy mode: %s\n", value);
    abort();
}

static int visual_policy_is_continuous(
    const int* action_dims,
    int num_action_heads
) {
    for (int h = 0; h < num_action_heads; h++) {
        if (action_dims[h] != 1) return 0;
    }
    return 1;
}

static int64_t visual_policy_advance_weight_offset(
        int64_t offset, int64_t count) {
    return (offset + count + 7) & ~(int64_t)7;
}

static int64_t visual_policy_expected_weight_count(
    int input_size,
    int hidden_size,
    int num_layers,
    const int* action_dims,
    int num_action_heads,
    int decoder_value_heads,
    int entity_encoder,
    const EntityEncoderDescriptor* descriptor
) {
    int action_sum = 0;
    for (int h = 0; h < num_action_heads; h++) {
        action_sum += action_dims[h];
    }

    int64_t total = visual_policy_advance_weight_offset(
        0, (int64_t)hidden_size * input_size);
    if (entity_encoder) {
        if (!descriptor) return -1;
        for (int i = 0; i < descriptor->num_branches; i++) {
            const EntityPoolDescriptor* branch = &descriptor->branches[i];
            total = visual_policy_advance_weight_offset(
                total, (int64_t)branch->bottleneck * branch->feats);
            total = visual_policy_advance_weight_offset(
                total, (int64_t)hidden_size * branch->bottleneck);
        }
    }
    total = visual_policy_advance_weight_offset(
        total, (int64_t)(action_sum + decoder_value_heads) * hidden_size);
    if (visual_policy_is_continuous(action_dims, num_action_heads)) {
        total = visual_policy_advance_weight_offset(total, num_action_heads);
    }
    for (int i = 0; i < num_layers; i++) {
        total = visual_policy_advance_weight_offset(
            total, (int64_t)3 * hidden_size * hidden_size);
    }
    return total;
}

static int64_t visual_policy_file_weight_count(const Weights* weights) {
    return weights->size - 7;
}

static VisualPolicyModelShape visual_policy_select_model_shape(
    const VisualPolicy* policy,
    const EncounterDef* edef,
    const EntityEncoderDescriptor* descriptor,
    int cli_hidden_size,
    int cli_num_layers,
    int cli_entity_encoder
) {
    static const int HS_GRID[] = {128, 256, 512, 1024, 2048, 4096};
    int obs_input_size = policy->obs_size;
    int full_input_size = policy->obs_size + policy->mask_size;
    int64_t file_weights = visual_policy_file_weight_count(policy->weights);
    int enc_max = descriptor ? 1 : 0;

    VisualPolicyModelShape match = {0};
    int matches = 0;
    for (int hi = 0; hi < (int)(sizeof(HS_GRID) / sizeof(HS_GRID[0])); hi++) {
        int hs = HS_GRID[hi];
        if (cli_hidden_size > 0 && hs != cli_hidden_size) continue;
        for (int layers = 1; layers <= 8; layers++) {
            if (cli_num_layers > 0 && layers != cli_num_layers) continue;
            for (int enc = 0; enc <= enc_max; enc++) {
                if (cli_entity_encoder > 0 && enc != cli_entity_encoder) continue;
                for (int value_heads = 0; value_heads <= 1; value_heads++) {
                    for (int variant = 0; variant <= 1; variant++) {
                        int input_size = variant ? full_input_size : obs_input_size;
                        if (enc && input_size != edef->obs_size) continue;
                        int64_t expected = visual_policy_expected_weight_count(
                            input_size, hs, layers, policy->action_dims,
                            policy->num_action_heads, value_heads, enc, descriptor);
                        if (expected != file_weights) continue;
                        match = (VisualPolicyModelShape){
                            input_size, value_heads, enc, hs, layers};
                        matches++;
                        if (matches <= 8) {
                            fprintf(stderr,
                                "policy: %s arch candidate hs=%d layers=%d entity=%d input=%s value_heads=%d\n",
                                edef->name, hs, layers, enc,
                                variant ? "obs+mask" : "obs", value_heads);
                        }
                    }
                }
            }
        }
    }

    if (matches == 0) {
        fprintf(stderr,
            "policy: %s model shape mismatch: file=%lld floats matches no architecture"
            " (obs=%d mask=%d, scanned hs 128..4096, layers 1..8, entity 0..%d%s)\n",
            edef->name, (long long)file_weights, policy->obs_size, policy->mask_size,
            enc_max,
            (cli_hidden_size > 0 || cli_num_layers > 0 || cli_entity_encoder > 0)
                ? " within the given CLI constraints" : "");
        abort();
    }
    if (matches > 1) {
        fprintf(stderr,
            "policy: %s model shape ambiguous: %d architectures match %lld floats"
            " (candidates above) -- pin --hidden-size/--num-layers/--entity-encoder\n",
            edef->name, matches, (long long)file_weights);
        abort();
    }
    return match;
}

static int64_t visual_policy_layout_tensor(const char* name, int64_t off, int64_t count) {
    int64_t start = off;
    int64_t end = visual_policy_advance_weight_offset(off, count);
    fprintf(stderr, "policy: tensor %-16s [%lld, %lld) (%lld floats)\n",
        name, (long long)start, (long long)end, (long long)count);
    return end;
}

static void visual_policy_assert_offset(
    const char* group, const Weights* weights, int64_t expect
) {
    if ((int64_t)weights->idx != expect) {
        fprintf(stderr, "policy: %s weight offset mismatch: cursor=%d expected=%lld\n",
            group, weights->idx, (long long)expect);
        abort();
    }
}

static VisualNet* visual_policy_make_puffernet(
    Weights* weights,
    int input_dim,
    int hidden_dim,
    int num_layers,
    int action_dims[],
    int num_action_heads,
    int decoder_value_heads,
    int entity_encoder,
    const EntityEncoderDescriptor* descriptor
) {
    VisualNet* net = (VisualNet*)calloc(1, sizeof(VisualNet));
    if (!net) {
        fprintf(stderr, "policy: failed to allocate puffer net\n");
        abort();
    }
    net->num_agents = 1;
    net->obs = (float*)calloc((size_t)input_dim, sizeof(float));
    if (!net->obs) {
        fprintf(stderr, "policy: failed to allocate puffer net obs\n");
        abort();
    }

    int action_sum = 0;
    int is_continuous = visual_policy_is_continuous(action_dims, num_action_heads);
    for (int h = 0; h < num_action_heads; h++) {
        action_sum += action_dims[h];
    }
    if (is_continuous && decoder_value_heads == 0) {
        fprintf(stderr, "policy: continuous policy-only decoder is unsupported\n");
        abort();
    }

    net->is_continuous = is_continuous;
    net->num_actions = num_action_heads;

    int64_t off = 0;
    if (entity_encoder) {
        if (!descriptor) {
            fprintf(stderr, "policy: entity encoder resolved without a descriptor\n");
            abort();
        }
        off = visual_policy_layout_tensor(
            "enc.global_w", off, (int64_t)hidden_dim * input_dim);
        for (int i = 0; i < descriptor->num_branches; i++) {
            const EntityPoolDescriptor* branch = &descriptor->branches[i];
            char name[48];
            snprintf(name, sizeof(name), "enc.%s_l1_w", branch->weight_name);
            off = visual_policy_layout_tensor(
                name, off, (int64_t)branch->bottleneck * branch->feats);
            snprintf(name, sizeof(name), "enc.%s_l2_w", branch->weight_name);
            off = visual_policy_layout_tensor(
                name, off, (int64_t)hidden_dim * branch->bottleneck);
        }
    } else {
        off = visual_policy_layout_tensor(
            "enc.weight", off, (int64_t)hidden_dim * input_dim);
    }
    int64_t off_after_encoder = off;
    off = visual_policy_layout_tensor("decoder.weight", off,
        (int64_t)(action_sum + decoder_value_heads) * hidden_dim);
    int64_t off_after_decoder = off;
    if (is_continuous) {
        off = visual_policy_layout_tensor("decoder.logstd", off, (int64_t)num_action_heads);
    }
    int64_t off_after_logstd = off;
    for (int l = 0; l < num_layers; l++) {
        char name[32];
        snprintf(name, sizeof(name), "mingru.proj[%d]", l);
        off = visual_policy_layout_tensor(name, off, (int64_t)3 * hidden_dim * hidden_dim);
    }
    int64_t off_total = off;

    if (entity_encoder) {
        net->entity_encoder = make_entity_encoder(
            weights, 1, input_dim, hidden_dim, descriptor);
    } else {
        net->encoder = make_linear(weights, 1, input_dim, hidden_dim);
    }
    visual_policy_assert_offset("encoder", weights, off_after_encoder);

    net->decoder = make_linear(weights, 1, hidden_dim, action_sum + decoder_value_heads);
    visual_policy_assert_offset("decoder", weights, off_after_decoder);

    if (net->is_continuous) {
        net->log_std = get_weights_aligned(weights, num_action_heads);
        visual_policy_assert_offset("logstd", weights, off_after_logstd);
    }

    net->mingru = make_mingru(weights, 1, hidden_dim, num_layers);
    visual_policy_assert_offset("mingru", weights, off_total);

    if (!net->is_continuous) {
        net->multidiscrete = make_multidiscrete(1, action_dims, num_action_heads);
    }
    return net;
}

static uint32_t visual_policy_next_u32(VisualPolicy* policy) {
    policy->rng_state = policy->rng_state * 1664525u + 1013904223u;
    return policy->rng_state;
}

static float visual_policy_next_uniform(VisualPolicy* policy) {
    return (float)((visual_policy_next_u32(policy) >> 8) * (1.0 / 16777216.0));
}

static int g_cli_hidden_size = -1;
static int g_cli_num_layers = -1;
static int g_cli_entity_encoder = 0;
static const char* g_cli_screenshot_path = NULL;
static int g_cli_screenshot_frame = 0;
static int g_cli_gui_tab = -1;
static float g_cli_tps = 0.0f;
static float g_cli_camera_dist = -1.0f;
static float g_cli_camera_yaw = -1000.0f;
static float g_cli_camera_pitch = -1000.0f;
static int g_cli_visual_loadout_mode = -1;
static void visual_policy_init(
    VisualPolicy* policy,
    const EncounterDef* edef,
    const char* model_path,
    VisualPolicyMode mode,
    uint32_t seed,
    int cli_hidden_size,
    int cli_num_layers,
    int cli_entity_encoder
) {
    memset(policy, 0, sizeof(*policy));
    if (!model_path || !model_path[0]) return;
    if (!edef) {
        fprintf(stderr, "policy: missing encounter definition\n");
        abort();
    }
    if (edef->num_action_heads > VISUAL_POLICY_MAX_ACTION_HEADS) {
        fprintf(stderr, "policy: too many action heads: %d\n", edef->num_action_heads);
        abort();
    }
    int action_mask_size = 0;
    for (int h = 0; h < edef->num_action_heads; h++) {
        action_mask_size += edef->action_head_dims[h];
    }
    if (action_mask_size != edef->mask_size) {
        fprintf(stderr, "policy: %s mask mismatch heads=%d mask=%d\n",
            edef->name, action_mask_size, edef->mask_size);
        abort();
    }
    policy->obs_size = edef->obs_size;
    policy->mask_size = edef->mask_size;
    policy->num_action_heads = edef->num_action_heads;
    for (int h = 0; h < edef->num_action_heads; h++) {
        policy->action_dims[h] = edef->action_head_dims[h];
    }
    const EntityEncoderDescriptor* descriptor =
        visual_policy_entity_descriptor(edef);
    policy->weights = load_weights(model_path);
    if (!policy->weights) {
        fprintf(stderr, "policy: failed to load model: %s\n", model_path);
        abort();
    }
    VisualPolicyModelShape model_shape = visual_policy_select_model_shape(
        policy, edef, descriptor,
        cli_hidden_size, cli_num_layers, cli_entity_encoder);
    fprintf(stderr,
        "policy: %s arch resolved hs=%d layers=%d entity=%d input=%d value_heads=%d\n",
        edef->name, model_shape.hidden_size, model_shape.num_layers,
        model_shape.entity_encoder, model_shape.input_size,
        model_shape.decoder_value_heads);
    policy->net = visual_policy_make_puffernet(
        policy->weights,
        model_shape.input_size,
        model_shape.hidden_size,
        model_shape.num_layers,
        policy->action_dims,
        policy->num_action_heads,
        model_shape.decoder_value_heads,
        model_shape.entity_encoder,
        descriptor);
    int64_t file_weights = visual_policy_file_weight_count(policy->weights);
    if (policy->weights->idx != file_weights) {
        fprintf(stderr,
            "policy: model shape mismatch consumed=%d floats file=%lld floats\n",
            policy->weights->idx, (long long)file_weights);
        abort();
    }
    policy->obs = (float*)osrs_calloc_or_abort(
        (size_t)(policy->obs_size + policy->mask_size),
        sizeof(float),
        "visual policy obs");
    policy->mode = mode;
    policy->rng_state = seed;
    policy->enabled = 1;
    fprintf(stderr, "policy: loaded %s mode=%s seed=%u\n",
        model_path, mode == VISUAL_POLICY_ARGMAX ? "argmax" : "sample", seed);
}

static void __attribute__((unused)) visual_policy_destroy(VisualPolicy* policy) {
    if (!policy) return;
    if (policy->net) visual_net_free(policy->net);
    free(policy->weights);
    free(policy->obs);
    memset(policy, 0, sizeof(*policy));
}

static void visual_policy_reset_recurrent(VisualPolicy* policy) {
    if (!policy || !policy->net || !policy->net->mingru) return;
    memset(policy->net->mingru->state, 0,
        (size_t)policy->net->mingru->num_layers *
        (size_t)policy->net->mingru->batch_size *
        (size_t)policy->net->mingru->hidden_size *
        sizeof(float));
}

static int visual_policy_argmax_masked(const float* logits, const float* mask, int dim) {
    int best_action = -1;
    float best_logit = -INFINITY;
    for (int a = 0; a < dim; a++) {
        if (mask[a] <= 0.5f) continue;
        if (best_action < 0 || logits[a] > best_logit) {
            best_action = a;
            best_logit = logits[a];
        }
    }
    return best_action;
}

static int visual_policy_sample_masked(
    VisualPolicy* policy,
    const float* logits,
    const float* mask,
    int dim
) {
    int best_action = visual_policy_argmax_masked(logits, mask, dim);
    if (best_action < 0) return -1;
    float max_logit = logits[best_action];
    float sum = 0.0f;
    for (int a = 0; a < dim; a++) {
        if (mask[a] <= 0.5f) continue;
        sum += expf(logits[a] - max_logit);
    }
    if (!(sum > 0.0f) || !isfinite(sum)) {
        fprintf(stderr, "policy: invalid masked softmax sum %f\n", sum);
        abort();
    }
    float threshold = visual_policy_next_uniform(policy) * sum;
    float acc = 0.0f;
    for (int a = 0; a < dim; a++) {
        if (mask[a] <= 0.5f) continue;
        acc += expf(logits[a] - max_logit);
        if (threshold <= acc) return a;
    }
    return best_action;
}

static void visual_policy_actions_from_obs(VisualPolicy* policy, int* actions) {
    if (!policy || !policy->enabled) return;
    float* encoded;
    if (policy->net->entity_encoder) {
        entity_encoder_forward(policy->net->entity_encoder, policy->obs);
        encoded = policy->net->entity_encoder->output;
    } else {
        linear(policy->net->encoder, policy->obs);
        encoded = policy->net->encoder->output;
    }
    mingru(policy->net->mingru, encoded);
    linear(policy->net->decoder, policy->net->mingru->output);

    const float* logits = policy->net->decoder->output;
    const float* mask = policy->obs + policy->obs_size;
    int logit_offset = 0;
    int mask_offset = 0;
    for (int h = 0; h < policy->num_action_heads; h++) {
        int dim = policy->action_dims[h];
        int action = policy->mode == VISUAL_POLICY_ARGMAX
            ? visual_policy_argmax_masked(logits + logit_offset, mask + mask_offset, dim)
            : visual_policy_sample_masked(
                policy, logits + logit_offset, mask + mask_offset, dim);
        if (action < 0) {
            fprintf(stderr, "policy: action head %d has no valid mask entry\n", h);
            abort();
        }
        actions[h] = action;
        logit_offset += dim;
        mask_offset += dim;
    }
}

static void visual_policy_actions(
    VisualPolicy* policy,
    const EncounterDef* edef,
    EncounterState* state,
    EncounterContext* context,
    int* actions
) {
    if (!policy || !policy->enabled) return;
    edef->write_obs(state, context, policy->obs);
    edef->write_mask(state, context, policy->obs + policy->obs_size);
    visual_policy_actions_from_obs(policy, actions);
}

static void run_policy_profile(
    OsrsEnv* env,
    const char* encounter_name,
    int start_wave,
    int profile_steps,
    const char* model_path,
    VisualPolicyMode policy_mode,
    uint32_t policy_seed,
    int loadout_mode
) {
    if (profile_steps <= 0) {
        fprintf(stderr, "policy profile requires --profile-steps > 0\n");
        abort();
    }
    const EncounterDef* edef = visual_open_encounter(env, encounter_name);
    if (!edef) abort();
    visual_load_encounter_collision_map(edef, env, encounter_name);
    if (strcmp(encounter_name, "colosseum") == 0) {
        if (start_wave >= 0)
            edef->put_int(env->encounter_state, env->encounter_context,
                "start_wave", start_wave);
        edef->put_int(env->encounter_state, env->encounter_context,
            "loadout_profile_mode", loadout_mode);
    } else if (strcmp(encounter_name, "inferno") == 0 && start_wave >= 0) {
        edef->put_int(env->encounter_state, env->encounter_context,
            "start_wave", start_wave);
    }
    visual_finalize_encounter(edef, env);
    edef->reset(env->encounter_state, env->encounter_context, policy_seed);

    VisualPolicy policy;
    visual_policy_init(&policy, edef, model_path, policy_mode, policy_seed,
        g_cli_hidden_size, g_cli_num_layers, g_cli_entity_encoder);
    if (!policy.enabled) abort();

#ifdef COLO_PROFILE_ENABLED
    int profile_count = colosseum_env_profile_count();
    for (int i = 0; i < profile_count; i++)
        (void)colosseum_env_profile_read_reset_ms(i);
#endif

    int actions[VISUAL_POLICY_MAX_ACTION_HEADS] = {0};
    int total_steps = 0;
    int reset_count = 0;
    double environment_ms = 0.0;
    uint64_t trace_hash = 1469598103934665603ULL;
    double wall_start = osrs_profile_now_seconds();
    while (total_steps < profile_steps) {
        double environment_step_ms = 0.0;
        double start_ms = osrs_profile_now_seconds() * 1000.0;
        edef->write_obs(env->encounter_state, env->encounter_context, policy.obs);
        double end_ms = osrs_profile_now_seconds() * 1000.0;
        environment_step_ms += end_ms - start_ms;
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_ADD(COLO_PROF_C_WRITE_OBS, end_ms - start_ms);
#endif

        start_ms = end_ms;
        edef->write_mask(env->encounter_state, env->encounter_context,
            policy.obs + policy.obs_size);
        end_ms = osrs_profile_now_seconds() * 1000.0;
        environment_step_ms += end_ms - start_ms;
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_ADD(COLO_PROF_C_WRITE_MASK, end_ms - start_ms);
#endif
        trace_hash = osrs_profile_hash_bytes(
            trace_hash,
            policy.obs,
            (size_t)(policy.obs_size + policy.mask_size) * sizeof(float));

        visual_policy_actions_from_obs(&policy, actions);
        trace_hash = osrs_profile_hash_bytes(
            trace_hash,
            actions,
            (size_t)policy.num_action_heads * sizeof(int));

        start_ms = osrs_profile_now_seconds() * 1000.0;
        edef->step(env->encounter_state, env->encounter_context, actions);
        end_ms = osrs_profile_now_seconds() * 1000.0;
        environment_step_ms += end_ms - start_ms;
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_ADD(COLO_PROF_C_ENCOUNTER_STEP, end_ms - start_ms);
#endif

        start_ms = end_ms;
        float reward = edef->get_reward(env->encounter_state, env->encounter_context);
        int terminal = edef->is_terminal(
            env->encounter_state, env->encounter_context);
        end_ms = osrs_profile_now_seconds() * 1000.0;
        environment_step_ms += end_ms - start_ms;
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_ADD(COLO_PROF_C_REWARD_TERMINAL, end_ms - start_ms);
#endif
        trace_hash = osrs_profile_hash_bytes(trace_hash, &reward, sizeof(reward));
        trace_hash = osrs_profile_hash_bytes(trace_hash, &terminal, sizeof(terminal));

        if (terminal) {
            start_ms = end_ms;
            reset_count++;
            edef->reset(env->encounter_state, env->encounter_context,
                policy_seed + (uint32_t)reset_count);
            visual_policy_reset_recurrent(&policy);
            end_ms = osrs_profile_now_seconds() * 1000.0;
            environment_step_ms += end_ms - start_ms;
#ifdef COLO_PROFILE_ENABLED
            COLO_PROFILE_ADD(COLO_PROF_C_RESET, end_ms - start_ms);
#endif
        }

#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_ADD(COLO_PROF_C_STEP_TOTAL, environment_step_ms);
#endif
        environment_ms += environment_step_ms;
        total_steps++;
    }
    double wall_elapsed = osrs_profile_now_seconds() - wall_start;

    printf("Policy profile results:\n");
    printf("  Total steps: %d\n", total_steps);
    printf("  Resets: %d\n", reset_count);
    printf("  Wall time: %.3f seconds\n", wall_elapsed);
    printf("  Environment time: %.3f seconds\n", environment_ms / 1000.0);
    printf("  Environment steps/sec: %.0f\n",
        environment_ms > 0.0 ? 1000.0 * total_steps / environment_ms : 0.0);
    printf("  Wall steps/sec: %.0f\n",
        wall_elapsed > 0.0 ? total_steps / wall_elapsed : 0.0);
    printf("  Trace hash: %016llx\n", (unsigned long long)trace_hash);
#ifdef COLO_PROFILE_ENABLED
    osrs_print_colosseum_profile_results(total_steps);
#endif

    visual_policy_destroy(&policy);
    edef->destroy(env->encounter_state);
    env->encounter_state = NULL;
    visual_destroy_encounter_context(
        edef, (EncounterContext**)&env->encounter_context);
}

typedef struct {
    VisualPolicy* policy;
    const EncounterDef* edef;
    EncounterState* state;
    EncounterContext* context;
    int actions[64];
#ifndef __EMSCRIPTEN__
    pthread_t thread;
#endif
    int in_flight;
    int has_actions;
} AsyncPolicy;

#ifndef __EMSCRIPTEN__
static void* async_policy_worker(void* arg) {
    AsyncPolicy* ap = (AsyncPolicy*)arg;
    visual_policy_actions(ap->policy, ap->edef, ap->state, ap->context, ap->actions);
    return NULL;
}
#endif

static void async_policy_join(AsyncPolicy* ap) {
#ifndef __EMSCRIPTEN__
    if (!ap->in_flight) return;
    if (pthread_join(ap->thread, NULL) != 0) {
        fprintf(stderr, "async policy: pthread_join failed\n");
        abort();
    }
    ap->in_flight = 0;
    ap->has_actions = 1;
#else
#endif
}

static void async_policy_spawn(
    AsyncPolicy* ap,
    VisualPolicy* policy,
    const EncounterDef* edef,
    OsrsEnv* env
) {
#ifndef __EMSCRIPTEN__
    if (ap->in_flight) {
        fprintf(stderr, "async policy: spawn while in flight\n");
        abort();
    }
    ap->policy = policy;
    ap->edef = edef;
    ap->state = env->encounter_state;
    ap->context = (EncounterContext*)env->encounter_context;
    ap->has_actions = 0;
    if (pthread_create(&ap->thread, NULL, async_policy_worker, ap) != 0) {
        fprintf(stderr, "async policy: pthread_create failed\n");
        abort();
    }
    ap->in_flight = 1;
#else
#endif
}

typedef struct {
    OsrsEnv* env;
    const char* encounter_name;
    ReplayFile* replay;
    VisualPolicy policy;
    AsyncPolicy async_policy;
    int start_wave;
    double episode_end_time;
    int episode_ended;
    int seen_lab_restore_generation;
} VisualState;

static void visual_async_policy_guard(void* ctx) {
    VisualState* vs = (VisualState*)ctx;
    async_policy_join(&vs->async_policy);
    vs->async_policy.has_actions = 0;
}

static void visual_frame(void* arg) {
    VisualState* vs = (VisualState*)arg;
    OsrsEnv* env = vs->env;
    RenderClient* rc = (RenderClient*)env->client;
    if (rc->lab_restore_generation != vs->seen_lab_restore_generation) {
        vs->seen_lab_restore_generation = rc->lab_restore_generation;
        vs->episode_ended = 0;
        async_policy_join(&vs->async_policy);
        vs->async_policy.has_actions = 0;
        visual_policy_reset_recurrent(&vs->policy);
    }

    if (rc->step_back) {
        rc->step_back = 0;
        async_policy_join(&vs->async_policy);
        vs->async_policy.has_actions = 0;
        render_restore_snapshot(rc, env);
        if (rc->history_cursor >= rc->history_count - 1) {
            rc->history_cursor = -1;
        }
        pvp_render(env);
        return;
    }

    if (rc->history_cursor >= 0) {
        pvp_render(env);
        return;
    }

    if (vs->episode_ended) {
        pvp_render(env);
        if (GetTime() - vs->episode_end_time >= 2.0) {
            vs->episode_ended = 0;
            async_policy_join(&vs->async_policy);
            vs->async_policy.has_actions = 0;
            if (env->encounter_def) {
                ((const EncounterDef*)env->encounter_def)->reset(
                    env->encounter_state,
                    (EncounterContext*)env->encounter_context,
                    (uint32_t)rand());
            } else {
                pvp_reset(env, rc->route_topology);
            }
            render_reset_episode_visual_state(rc, env);
            visual_policy_reset_recurrent(&vs->policy);
            render_save_snapshot(rc, env);
        }
        return;
    }

    if (rc->is_paused && !rc->step_once) {
        pvp_render(env);
        return;
    }
    rc->step_once = 0;

    if (rc->ticks_per_second > 0.0f) {
        double interval = 1.0 / (double)rc->ticks_per_second;
        double now = GetTime();
        if (now - rc->last_tick_time < interval) {
            pvp_render(env);
            return;
        }
        rc->last_tick_time += interval;
        if (now - rc->last_tick_time >= interval)
            rc->last_tick_time = now;
    }

    async_policy_join(&vs->async_policy);

    render_pre_tick(rc, env);

    if (env->encounter_def && env->encounter_state) {
        const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
        int enc_actions[64] = {0};
        int used_human_step = 0;

        if (rc->human_input.enabled && edef->step_human_commands) {
            edef->step_human_commands(
                env->encounter_state,
                (EncounterContext*)env->encounter_context,
                &rc->human_input);
            used_human_step = 1;
        } else if (rc->human_input.enabled) {
            if (edef->translate_human_input)
                edef->translate_human_input(&rc->human_input, enc_actions,
                                            env->encounter_state,
                                            (EncounterContext*)env->encounter_context);
            if (rc->human_input.pending_move_x >= 0 && edef->put_int) {
                edef->put_int(env->encounter_state,
                              (EncounterContext*)env->encounter_context,
                              "player_dest_x",
                              rc->human_input.pending_move_x);
                edef->put_int(env->encounter_state,
                              (EncounterContext*)env->encounter_context,
                              "player_dest_y",
                              rc->human_input.pending_move_y);
            } else if (rc->human_input.pending_attack && edef->put_int) {
                edef->put_int(
                    env->encounter_state,
                    (EncounterContext*)env->encounter_context,
                    "player_dest_x",
                    -1);
                edef->put_int(
                    env->encounter_state,
                    (EncounterContext*)env->encounter_context,
                    "player_dest_y",
                    -1);
            }
            human_input_clear_pending(&rc->human_input);
        } else if (vs->replay && replay_get_actions(vs->replay, enc_actions)) {
        } else if (vs->policy.enabled) {
            if (vs->async_policy.has_actions) {
                memcpy(enc_actions, vs->async_policy.actions,
                    sizeof(enc_actions));
            } else {
                visual_policy_actions(
                    &vs->policy,
                    edef,
                    env->encounter_state,
                    (EncounterContext*)env->encounter_context,
                    enc_actions);
            }
        } else if (strcmp(edef->name, "zulrah") == 0) {
            zul_heuristic_actions((ZulrahState*)env->encounter_state, enc_actions);
        } else {
            for (int h = 0; h < edef->num_action_heads; h++) {
                enc_actions[h] = rand() % edef->action_head_dims[h];
            }
        }
        vs->async_policy.has_actions = 0;

        if (!used_human_step) {
            edef->step(
                env->encounter_state,
                (EncounterContext*)env->encounter_context,
                enc_actions);
        }
        env->tick = edef->get_tick(
            env->encounter_state, (EncounterContext*)env->encounter_context);

        if (rc->human_input.enabled && rc->human_input.pending_move_x >= 0) {
            Player* ply = edef->get_entity(
                env->encounter_state, (EncounterContext*)env->encounter_context, 0);
            if (ply && ply->x == rc->human_input.pending_move_x &&
                ply->y == rc->human_input.pending_move_y) {
                human_input_clear_move(&rc->human_input);
            }
        }

    } else {
        if (rc->human_input.enabled) {
            human_to_pvp_actions(
                &rc->human_input, env->actions, &env->players[0]);
            int* opp = env->actions + OSRS_BASE_NUM_ACTION_HEADS;
            for (int h = 0; h < OSRS_BASE_NUM_ACTION_HEADS; h++) {
                opp[h] = rand() % NH_PVP_ACTION_DIMS[h];
            }
            human_input_clear_pending(&rc->human_input);
        } else {
            for (int agent = 0; agent < NUM_AGENTS; agent++) {
                int* actions =
                    env->actions + agent * OSRS_BASE_NUM_ACTION_HEADS;
                for (int h = 0; h < OSRS_BASE_NUM_ACTION_HEADS; h++) {
                    actions[h] = rand() % NH_PVP_ACTION_DIMS[h];
                }
            }
        }
        pvp_step(env, rc->route_topology, rc->player_route_cache);

        if (rc->human_input.enabled && rc->human_input.pending_move_x >= 0) {
            Player* p0 = &env->players[0];
            if (p0->x == rc->human_input.pending_move_x &&
                p0->y == rc->human_input.pending_move_y) {
                human_input_clear_move(&rc->human_input);
            }
        }
    }

    render_post_tick(rc, env);
    render_save_snapshot(rc, env);
    pvp_render(env);

    int is_over = env->encounter_def
        ? ((const EncounterDef*)env->encounter_def)->is_terminal(
            env->encounter_state, (EncounterContext*)env->encounter_context)
        : env->episode_over;
    if (is_over) {
        vs->episode_ended = 1;
        vs->episode_end_time = GetTime();
    } else if (env->encounter_def && vs->policy.enabled &&
               !rc->human_input.enabled && !vs->replay) {
        async_policy_spawn(
            &vs->async_policy,
            &vs->policy,
            (const EncounterDef*)env->encounter_def,
            env);
    }
}

static void run_metrics(
    OsrsEnv* env,
    const char* encounter_name,
    const char* model_path,
    VisualPolicyMode policy_mode,
    uint32_t policy_seed,
    int num_episodes,
    int loadout_mode,
    int bis_oracle,
    int start_wave
) {
    if (!encounter_name || strcmp(encounter_name, "colosseum") != 0) {
        fprintf(stderr, "metrics mode requires --encounter colosseum\n");
        return;
    }
    const EncounterDef* edef = visual_open_encounter(env, encounter_name);
    if (!edef) return;
    visual_load_encounter_collision_map(edef, env, encounter_name);
    edef->put_int(env->encounter_state, env->encounter_context, "loadout_profile_mode", loadout_mode);
    edef->put_float(env->encounter_state, env->encounter_context, "beginner_loadout_fraction", 0.5f);
    edef->put_int(env->encounter_state, env->encounter_context, "start_wave",
                  start_wave >= 0 ? start_wave : 1);
    if (bis_oracle)
        edef->put_int(env->encounter_state, env->encounter_context,
                      "bis_gear_oracle_mode", 1);
    visual_finalize_encounter(edef, env);
    edef->reset(env->encounter_state, env->encounter_context, policy_seed);

    VisualPolicy policy;
    visual_policy_init(&policy, edef, model_path, policy_mode, policy_seed,
                       g_cli_hidden_size, g_cli_num_layers, g_cli_entity_encoder);
    if (!policy.enabled) {
        fprintf(stderr, "metrics mode: failed to load model (pass --model <path>)\n");
        return;
    }

    static const char* npc_names[COLO_NUM_NPC_TYPES] = {
        "berserker", "archer", "seer", "serpent", "jaguar", "javelin",
        "shockwave", "minotaur", "manticore", "sol", "totem", "bees"
    };
    static uint64_t wpn_npc[256][COLO_NUM_NPC_TYPES];
    static uint64_t wpn_total[256];
    static uint64_t wpn_spec[256];
    static double wpn_eff_sum[256];
    static uint64_t wpn_eff_n[256];
    static double npc_eff_sum[COLO_NUM_NPC_TYPES];
    static uint64_t npc_eff_n[COLO_NUM_NPC_TYPES];
    static uint64_t wave_ticks[12], wave_visits[12], wave_reinforced[12];
    static uint64_t wave_attacks_post_reinforce[12];
    double sol_damage_by_source[COLO_NUM_SOL_DAMAGE_SOURCES] = {0};
    double javelin_damage_by_source[COLO_NUM_JAVELIN_DAMAGE_SOURCES] = {0};
    double death_by_source[COLO_NUM_DAMAGE_SOURCES] = {0};
    double doom_death_by_source[COLO_NUM_DAMAGE_SOURCES] = {0};
    memset(wpn_npc, 0, sizeof(wpn_npc));
    memset(wpn_total, 0, sizeof(wpn_total));
    memset(wpn_spec, 0, sizeof(wpn_spec));
    memset(wpn_eff_sum, 0, sizeof(wpn_eff_sum));
    memset(wpn_eff_n, 0, sizeof(wpn_eff_n));
    memset(npc_eff_sum, 0, sizeof(npc_eff_sum));
    memset(npc_eff_n, 0, sizeof(npc_eff_n));
    memset(wave_ticks, 0, sizeof(wave_ticks));
    memset(wave_visits, 0, sizeof(wave_visits));
    memset(wave_reinforced, 0, sizeof(wave_reinforced));
    memset(wave_attacks_post_reinforce, 0, sizeof(wave_attacks_post_reinforce));
    uint64_t total_attacks = 0, argmax_set_attacks = 0, argmax_evals = 0;
    long total_ticks = 0;
    int episodes = 0;
    int sol_episodes = 0;
    int prev_wave = 0, prev_reinf_timer = 0, wave_seen = 0;
    int enc_actions[64] = {0};

    enum { METRICS_MAX_EPISODES = 512 };
    if (num_episodes > METRICS_MAX_EPISODES) {
        fprintf(stderr, "metrics mode caps at %d episodes\n", METRICS_MAX_EPISODES);
        return;
    }
    static float ep_scores[METRICS_MAX_EPISODES];
    static int ep_winners[METRICS_MAX_EPISODES];

    while (episodes < num_episodes) {
        visual_policy_actions(&policy, edef, env->encounter_state,
            (EncounterContext*)env->encounter_context, enc_actions);
        edef->step(env->encounter_state,
            (EncounterContext*)env->encounter_context, enc_actions);
        total_ticks++;
        ColosseumState* cs = (ColosseumState*)env->encounter_state;

        int w_now = cs->wave >= 0 && cs->wave < 12 ? cs->wave : 11;
        if (!wave_seen || w_now != prev_wave) {
            wave_visits[w_now]++;
            wave_seen = 1;
        }
        wave_ticks[w_now]++;
        if (prev_reinf_timer > 0 &&
                cs->reinforcement_timer == COLO_REINFORCE_FIRED)
            wave_reinforced[w_now]++;
        prev_wave = w_now;
        prev_reinf_timer = cs->reinforcement_timer;

        if (cs->tick_scratch.player_attacked) {
            int slot = cs->player_attack_npc_idx;
            uint8_t w = cs->player.equipped[GEAR_SLOT_WEAPON];
            if (slot >= 0 && slot < COLO_MAX_NPCS) {
                int t = (int)cs->npcs[slot].type;
                if (t >= 0 && t < COLO_NUM_NPC_TYPES) {
                    wpn_npc[w][t]++;
                    wpn_total[w]++;
                    total_attacks++;
                    if (cs->player.used_special_this_tick) wpn_spec[w]++;

                    if (cs->reinforcement_timer == COLO_REINFORCE_FIRED)
                        wave_attacks_post_reinforce[w_now]++;

                    int am = col_attacked_with_argmax_set(cs);
                    if (am >= 0) {
                        argmax_evals++;
                        argmax_set_attacks += (uint64_t)am;
                    }
                    const ColoNPC* npc = &cs->npcs[slot];
                    if (col_npc_is_live_target(npc) &&
                            !col_type_is_hazard_entity(npc->type)) {
                        const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] =
                            col_get_best_gear_table(cs);
                        float best_dpt = 0.0f;
                        for (int st = 0; st < COLO_NUM_WEAPON_SETS; st++)
                            if (best[st][t].dpt > best_dpt)
                                best_dpt = best[st][t].dpt;
                        float cur_dpt = col_expected_dpt_for_equipment_vs_npc(
                            cs, cs->player.equipped, npc, 1);
                        if (best_dpt > 0.0f) {
                            double eff = (double)(cur_dpt / best_dpt);
                            wpn_eff_sum[w] += eff;
                            wpn_eff_n[w]++;
                            npc_eff_sum[t] += eff;
                            npc_eff_n[t]++;
                        }
                    }
                }
            }
        }
        if (edef->is_terminal(env->encounter_state,
                (EncounterContext*)env->encounter_context)) {
            ep_scores[episodes] = cs->log.outcome_score;
            ep_winners[episodes] = cs->winner;
            if (cs->sol.started) sol_episodes++;
            for (int source = 0; source < COLO_NUM_SOL_DAMAGE_SOURCES; source++)
                sol_damage_by_source[source] +=
                    (double)cs->log.sol_damage_by_source[source];
            for (int source = 0;
                    source < COLO_NUM_JAVELIN_DAMAGE_SOURCES;
                    source++)
                javelin_damage_by_source[source] +=
                    (double)cs->log.javelin_damage_by_source[source];
            for (int source = 0; source < COLO_NUM_DAMAGE_SOURCES; source++) {
                death_by_source[source] +=
                    (double)cs->log.death_by_source[source];
                doom_death_by_source[source] +=
                    (double)cs->log.doom_death_by_source[source];
            }
            episodes++;
            edef->reset(env->encounter_state,
                (EncounterContext*)env->encounter_context,
                policy_seed + (uint32_t)episodes);
            visual_policy_reset_recurrent(&policy);
            wave_seen = 0;
            prev_reinf_timer = 0;
        }
    }

    printf("# colosseum weapon behavioral metrics\n");
    printf("# episodes=%d ticks=%ld total_attacks=%llu mode=%s bis_oracle=%d\n",
        num_episodes, total_ticks, (unsigned long long)total_attacks,
        policy_mode == VISUAL_POLICY_ARGMAX ? "argmax" : "sample", bis_oracle);
    printf("# sol_episodes=%d\n", sol_episodes);
    {
        double score_sum = 0.0;
        int wins = 0;
        for (int e = 0; e < episodes; e++) {
            score_sum += (double)ep_scores[e];
            if (ep_winners[e] == COLO_OUTCOME_PLAYER_WON) wins++;
        }
        printf("# outcome scores: mean %.4f, wins %d/%d, per-episode:",
            episodes ? score_sum / (double)episodes : 0.0, wins, episodes);
        for (int e = 0; e < episodes; e++) printf(" %.3f", ep_scores[e]);
        printf("\n");
    }
    printf("# argmax-style attacks: %llu/%llu (%.1f%%)\n",
        (unsigned long long)argmax_set_attacks,
        (unsigned long long)argmax_evals,
        argmax_evals ? 100.0 * (double)argmax_set_attacks / (double)argmax_evals : 0.0);
    printf("weapon,total_attacks,per_episode,pct,spec_pct,mean_dpt_eff\n");
    for (int w = 0; w < 256; w++) {
        if (wpn_total[w] == 0) continue;
        const Item* it = get_item((uint8_t)w);
        const char* wn = (it && it->name[0]) ? it->name : "unknown";
        printf("%s,%llu,%.1f,%.1f%%,%.1f%%,%.2f\n", wn,
            (unsigned long long)wpn_total[w],
            (double)wpn_total[w] / (double)num_episodes,
            total_attacks ? 100.0 * (double)wpn_total[w] / (double)total_attacks : 0.0,
            100.0 * (double)wpn_spec[w] / (double)wpn_total[w],
            wpn_eff_n[w] ? wpn_eff_sum[w] / (double)wpn_eff_n[w] : 0.0);
    }
    printf("\nnpc,attacks,mean_dpt_eff\n");
    for (int t = 0; t < COLO_NUM_NPC_TYPES; t++) {
        if (npc_eff_n[t] == 0) continue;
        printf("%s,%llu,%.2f\n", npc_names[t],
            (unsigned long long)npc_eff_n[t],
            npc_eff_sum[t] / (double)npc_eff_n[t]);
    }
    printf("\nwave,visits,mean_ticks,reinforced_pct,attacks_post_reinforce\n");
    for (int wv = 0; wv < 12; wv++) {
        if (wave_visits[wv] == 0) continue;
        printf("%d,%llu,%.0f,%.0f%%,%llu\n", wv + 1,
            (unsigned long long)wave_visits[wv],
            (double)wave_ticks[wv] / (double)wave_visits[wv],
            100.0 * (double)wave_reinforced[wv] / (double)wave_visits[wv],
            (unsigned long long)wave_attacks_post_reinforce[wv]);
    }
    static const char* const sol_damage_source_names[COLO_NUM_SOL_DAMAGE_SOURCES] = {
        "spear_1",
        "spear_2",
        "shield_1",
        "shield_2",
        "triple_parry",
        "grapple",
        "crystal_laser",
        "molten_sand",
    };
    double sol_damage_total = 0.0;
    for (int source = 0; source < COLO_NUM_SOL_DAMAGE_SOURCES; source++)
        sol_damage_total += sol_damage_by_source[source];
    printf("\nsol_damage_source,total,per_episode,per_sol_episode,pct\n");
    for (int source = 0; source < COLO_NUM_SOL_DAMAGE_SOURCES; source++) {
        double damage = sol_damage_by_source[source];
        printf("%s,%.0f,%.3f,%.3f,%.1f%%\n",
            sol_damage_source_names[source],
            damage,
            damage / (double)num_episodes,
            sol_episodes ? damage / (double)sol_episodes : 0.0,
            sol_damage_total > 0.0 ? 100.0 * damage / sol_damage_total : 0.0);
    }
    static const char* const javelin_damage_source_names[
        COLO_NUM_JAVELIN_DAMAGE_SOURCES
    ] = {
        "basic_ranged",
        "skyfall",
        "reentry_pool",
        "reentry_volatility_pool",
    };
    double javelin_damage_total = 0.0;
    for (int source = 0;
            source < COLO_NUM_JAVELIN_DAMAGE_SOURCES;
            source++)
        javelin_damage_total += javelin_damage_by_source[source];
    printf("\njavelin_damage_source,total,per_episode,pct\n");
    for (int source = 0;
            source < COLO_NUM_JAVELIN_DAMAGE_SOURCES;
            source++) {
        double damage = javelin_damage_by_source[source];
        printf("%s,%.0f,%.3f,%.1f%%\n",
            javelin_damage_source_names[source],
            damage,
            damage / (double)num_episodes,
            javelin_damage_total > 0.0
                ? 100.0 * damage / javelin_damage_total
                : 0.0);
    }
    static const char* const damage_source_names[COLO_NUM_DAMAGE_SOURCES] = {
        "npc_attack",
        "javelin_basic_ranged",
        "manticore_venom",
        "bee_poison",
        "bee_contact",
        "javelin_skyfall",
        "reentry_pool",
        "volatility_explosion",
        "volatility_pool",
        "reentry_volatility_pool",
        "solarflare",
        "self",
        "sol_spear_1",
        "sol_spear_2",
        "sol_shield_1",
        "sol_shield_2",
        "sol_triple_parry",
        "sol_grapple",
        "sol_crystal_laser",
        "sol_molten_sand",
    };
    printf("\ndamage_source,deaths,doom_deaths\n");
    for (int source = 0; source < COLO_NUM_DAMAGE_SOURCES; source++) {
        printf("%s,%.0f,%.0f\n",
            damage_source_names[source],
            death_by_source[source],
            doom_death_by_source[source]);
    }
    printf("\nweapon\\npc");
    for (int t = 0; t < COLO_NUM_NPC_TYPES; t++) printf(",%s", npc_names[t]);
    printf("\n");
    for (int w = 0; w < 256; w++) {
        if (wpn_total[w] == 0) continue;
        const Item* it = get_item((uint8_t)w);
        const char* wn = (it && it->name[0]) ? it->name : "unknown";
        printf("%s", wn);
        for (int t = 0; t < COLO_NUM_NPC_TYPES; t++)
            printf(",%llu", (unsigned long long)wpn_npc[w][t]);
        printf("\n");
    }
    visual_policy_destroy(&policy);
}

static void run_visual(
    OsrsEnv* env,
    const char* encounter_name,
    const char* replay_path,
    int start_wave,
    int gear_tier,
    const char* model_path,
    VisualPolicyMode policy_mode,
    uint32_t policy_seed
) {
    env->client = NULL;
    const EncounterArenaTopology* direct_pvp_topology = NULL;
    if (!encounter_name) {
        const char* cmap_path = getenv("OSRS_COLLISION_MAP");
        if (cmap_path && cmap_path[0]) {
            env->collision_map = collision_map_load(cmap_path);
            if (env->collision_map) {
                fprintf(stderr, "collision map loaded: %d regions\n",
                    ((CollisionMap*)env->collision_map)->count);
            }
        }
        direct_pvp_topology =
            pvp_route_topology_finalize(
                (const CollisionMap*)env->collision_map);
    }

    if (encounter_name) {
        const EncounterDef* edef = visual_open_encounter(env, encounter_name);
        if (!edef) return;
        if (encounter_name_is_pvp(encounter_name) && edef->put_int) {
            edef->put_int(env->encounter_state, env->encounter_context, "use_c_opponent", 1);
            edef->put_int(env->encounter_state, env->encounter_context, "opponent_type", OPP_IMPROVED);
#ifdef __EMSCRIPTEN__
            edef->put_int(env->encounter_state, env->encounter_context, "use_c_opponent_p0", 0);
#else
            edef->put_int(env->encounter_state, env->encounter_context, "use_c_opponent_p0", 1);
            edef->put_int(env->encounter_state, env->encounter_context, "opponent_p0_type", OPP_IMPROVED);
#endif
            edef->put_int(env->encounter_state, env->encounter_context, "is_lms", 1);
            edef->put_int(env->encounter_state, env->encounter_context, "gear_tier", gear_tier);
        }
        if (strcmp(encounter_name, "colosseum") == 0 && edef->put_int &&
                g_cli_visual_loadout_mode >= 0) {
            edef->put_int(env->encounter_state, env->encounter_context,
                "loadout_profile_mode", g_cli_visual_loadout_mode);
        }

        VisualCollisionLoad cload = visual_load_encounter_collision_map(edef, env, encounter_name);
        if (cload.cmap) {
            fprintf(stderr, "%s collision map: %d regions, offset (%d, %d)\n",
                    encounter_name, cload.cmap->count, cload.offset_x, cload.offset_y);
        }

        if (start_wave >= 0 && edef->put_int) {
            edef->put_int(
                env->encounter_state,
                env->encounter_context,
                "start_wave",
                start_wave);
        }
        visual_finalize_encounter(edef, env);
        edef->reset(env->encounter_state, env->encounter_context, 0);
        fprintf(stderr, "encounter: %s (obs=%d, heads=%d)\n",
                edef->name, edef->obs_size, edef->num_action_heads);
        if (start_wave >= 0)
            fprintf(stderr, "start_wave: %d\n", start_wave);
    } else {
        env->pvp_runtime.use_c_opponent = 1;
        env->pvp_runtime.opponent.type = OPP_IMPROVED;
        env->is_lms = 1;
        pvp_reset(env, direct_pvp_topology);
    }


    RenderClient* rc = visual_init_render_scene(
        env,
        encounter_name,
        direct_pvp_topology);
    pvp_render(env);

    ReplayFile* replay = NULL;
    if (replay_path && env->encounter_def) {
        const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
        size_t snapshot_size = 0;
        if (edef->snapshot_size)
            snapshot_size = edef->snapshot_size(
                env->encounter_state,
                env->encounter_context);
        replay = replay_load(replay_path, edef->num_action_heads, snapshot_size);
        if (replay && replay->initial_snapshot) {
            if (!edef->restore) {
                fprintf(stderr, "replay: encounter has snapshot data but no restore hook\n");
                abort();
            }
            edef->restore(
                env->encounter_state,
                env->encounter_context,
                replay->initial_snapshot,
                replay->initial_snapshot_size);
        } else if (replay && edef->put_int) {
            edef->reset(env->encounter_state, env->encounter_context, 0);
            edef->put_int(
                env->encounter_state,
                env->encounter_context,
                "seed",
                (int)replay->rng_seed);
        }
        if (replay) {
            render_populate_entities(rc, env);
            for (int i = 0; i < rc->entity_count; i++) {
                int size = rc->entities[i].npc_size > 1 ? rc->entities[i].npc_size : 1;
                rc->sub_x[i] = rc->entities[i].x * 128 + size * 64;
                rc->sub_y[i] = rc->entities[i].y * 128 + size * 64;
                rc->dest_x[i] = rc->sub_x[i];
                rc->dest_y[i] = rc->sub_y[i];
            }
        }
    }

    VisualPolicy policy;
    visual_policy_init(
        &policy,
        (const EncounterDef*)env->encounter_def,
        model_path,
        policy_mode,
        policy_seed,
        g_cli_hidden_size,
        g_cli_num_layers,
        g_cli_entity_encoder);

    render_save_snapshot(rc, env);

#ifdef __EMSCRIPTEN__
    static VisualState web_visual_state;
    web_visual_state = (VisualState){
        .env = env,
        .encounter_name = encounter_name,
        .replay = replay,
        .policy = policy,
        .start_wave = start_wave,
        .episode_end_time = 0,
        .episode_ended = 0,
        .seen_lab_restore_generation = rc->lab_restore_generation,
    };
    rc->pre_sim_mutation_hook = visual_async_policy_guard;
    rc->pre_sim_mutation_hook_ctx = &web_visual_state;
    emscripten_set_main_loop_arg(visual_frame, &web_visual_state, 0, 1);
#else
    VisualState vs = {
        .env = env,
        .encounter_name = encounter_name,
        .replay = replay,
        .policy = policy,
        .start_wave = start_wave,
        .episode_end_time = 0,
        .episode_ended = 0,
        .seen_lab_restore_generation = rc->lab_restore_generation,
    };
    rc->pre_sim_mutation_hook = visual_async_policy_guard;
    rc->pre_sim_mutation_hook_ctx = &vs;

    if (g_cli_camera_dist > 0.0f) rc->cam_dist = g_cli_camera_dist;
    if (g_cli_camera_yaw > -999.0f) rc->cam_yaw = g_cli_camera_yaw;
    if (g_cli_camera_pitch > -999.0f) rc->cam_pitch = g_cli_camera_pitch;
    if (g_cli_gui_tab >= 0 && g_cli_gui_tab < GUI_TAB_COUNT)
        rc->gui.active_tab = (GuiTab)g_cli_gui_tab;
    if (g_cli_tps > 0.0f)
        rc->ticks_per_second = g_cli_tps;

    int frame_counter = 0;
    while (!WindowShouldClose()) {
        if (g_cli_screenshot_path && rc->entity_count > 0) {
            int eidx = rc->gui.gui_entity_idx;
            if (eidx >= 0 && eidx < rc->entity_count) {
                rc->cam_target_x = (float)rc->sub_x[eidx] / 128.0f;
                rc->cam_target_z = -(float)rc->sub_y[eidx] / 128.0f;
            }
        }
        visual_frame(&vs);
        frame_counter++;
        if (g_cli_screenshot_path && frame_counter >= g_cli_screenshot_frame) {
            TakeScreenshot(g_cli_screenshot_path);
            fprintf(stderr, "screenshot: wrote %s at frame %d\n",
                g_cli_screenshot_path, frame_counter);
            break;
        }
    }

    async_policy_join(&vs.async_policy);

    replay_free(replay);
    visual_policy_destroy(&vs.policy);

    if (env->client) {
        render_destroy_client((RenderClient*)env->client);
        env->client = NULL;
    }
    if (env->encounter_def && env->encounter_state) {
        ((const EncounterDef*)env->encounter_def)->destroy(env->encounter_state);
        env->encounter_state = NULL;
        visual_destroy_encounter_context(
            (const EncounterDef*)env->encounter_def,
            (EncounterContext**)&env->encounter_context);
    }
#endif
}
#endif

static void visual_init_env_buffers(OsrsEnv* env) {
    pvp_init(env);
    env->ocean_io.agent_actions = env->actions;
    env->ocean_io.agent_rewards = env->rewards;
    env->ocean_io.agent_terminals = env->terminals;
}

int main(int argc, char** argv) {
    int use_visual = 1;
    int use_profile = 0;
    int gear_tier = -1;
    int start_wave = -1;
    int profile_steps = 0;
    int metrics_episodes = 0;
    int metrics_bis_oracle = 0;
    int loadout_mode = 2;
    const char* encounter_name __attribute__((unused)) = NULL;
    const char* replay_path __attribute__((unused)) = NULL;
    const char* model_path __attribute__((unused)) = NULL;
    const char* policy_mode_name __attribute__((unused)) = "sample";
    uint32_t policy_seed __attribute__((unused)) = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--visual") == 0) use_visual = 1;
        else if (strcmp(argv[i], "--profile") == 0) { use_profile = 1; use_visual = 0; }
        else if (strcmp(argv[i], "--encounter") == 0 && i + 1 < argc)
            encounter_name = argv[++i];
        else if (strcmp(argv[i], "--replay") == 0 && i + 1 < argc)
            replay_path = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--policy-mode") == 0 && i + 1 < argc)
            policy_mode_name = argv[++i];
        else if (strcmp(argv[i], "--policy-seed") == 0 && i + 1 < argc)
            policy_seed = visual_policy_parse_seed(argv[++i]);
        else if (strcmp(argv[i], "--hidden-size") == 0 && i + 1 < argc)
            g_cli_hidden_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-layers") == 0 && i + 1 < argc)
            g_cli_num_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--entity-encoder") == 0)
            g_cli_entity_encoder = (i + 1 < argc && argv[i + 1][0] != '-') ? atoi(argv[++i]) : 1;
        else if (strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc)
            g_cli_screenshot_path = argv[++i];
        else if (strcmp(argv[i], "--screenshot-frame") == 0 && i + 1 < argc)
            g_cli_screenshot_frame = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tab") == 0 && i + 1 < argc)
            g_cli_gui_tab = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tps") == 0 && i + 1 < argc)
            g_cli_tps = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--camera-dist") == 0 && i + 1 < argc)
            g_cli_camera_dist = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--camera-yaw") == 0 && i + 1 < argc)
            g_cli_camera_yaw = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--camera-pitch") == 0 && i + 1 < argc)
            g_cli_camera_pitch = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--tier") == 0 && i + 1 < argc)
            gear_tier = atoi(argv[++i]);
        else if (strcmp(argv[i], "--wave") == 0 && i + 1 < argc)
            start_wave = atoi(argv[++i]);
        else if ((strcmp(argv[i], "--start-wave") == 0 ||
                  strcmp(argv[i], "--start_wave") == 0) && i + 1 < argc)
            start_wave = atoi(argv[++i]);
        else if (strcmp(argv[i], "--profile-steps") == 0 && i + 1 < argc)
            profile_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metrics") == 0 && i + 1 < argc) {
            metrics_episodes = atoi(argv[++i]);
            use_visual = 0;
        }
        else if (strcmp(argv[i], "--bis-oracle") == 0)
            metrics_bis_oracle = 1;
        else if (strcmp(argv[i], "--loadout-mode") == 0 && i + 1 < argc) {
            loadout_mode = atoi(argv[++i]);
            g_cli_visual_loadout_mode = loadout_mode;
        }
    }

#ifdef OSRS_VISUAL_DEFAULT_ENCOUNTER
    if (!encounter_name) encounter_name = OSRS_VISUAL_DEFAULT_ENCOUNTER;
#endif
    if (encounter_name && strcmp(encounter_name, "pvp") == 0) encounter_name = "nh_pvp";
    VisualPolicyMode policy_mode __attribute__((unused)) =
        visual_policy_parse_mode(policy_mode_name);

    srand((unsigned int)time(NULL));

#ifdef __EMSCRIPTEN__
    static OsrsEnv env;
#else
    OsrsEnv env;
#endif
    memset(&env, 0, sizeof(OsrsEnv));

    if (metrics_episodes > 0) {
        run_metrics(&env, encounter_name, model_path, policy_mode, policy_seed,
            metrics_episodes, loadout_mode, metrics_bis_oracle, start_wave);
        return 0;
    }

    if (use_profile) {
        visual_init_env_buffers(&env);

        if (model_path && model_path[0]) {
            run_policy_profile(&env, encounter_name, start_wave, profile_steps,
                model_path, policy_mode, policy_seed, loadout_mode);
        } else {
            run_profile(
                &env, encounter_name, start_wave, profile_steps, policy_seed);
        }

        pvp_close(&env);
        return 0;
    }

    if (use_visual) {
#ifdef OSRS_VISUAL
        visual_init_env_buffers(&env);
        if (gear_tier >= 0 && gear_tier <= 3) {
            for (int t = 0; t < 4; t++) env.pvp_runtime.gear_tier_weights[t] = 0.0f;
            env.pvp_runtime.gear_tier_weights[gear_tier] = 1.0f;
        } else {
            env.pvp_runtime.gear_tier_weights[0] = 0.60f;
            env.pvp_runtime.gear_tier_weights[1] = 0.25f;
            env.pvp_runtime.gear_tier_weights[2] = 0.10f;
            env.pvp_runtime.gear_tier_weights[3] = 0.05f;
        }
        run_visual(
            &env,
            encounter_name,
            replay_path,
            start_wave,
            gear_tier,
            model_path,
            policy_mode,
            policy_seed);
        pvp_close(&env);
#else
        fprintf(stderr, "not compiled with visual support (use: make visual)\n");
        return 1;
#endif
    } else {
        visual_init_env_buffers(&env);

        printf("OSRS PvP C Environment Demo\n\n");

        printf("Running single verbose episode...\n");
        run_random_episode(&env, 1);

        printf("\n");
        benchmark(&env, 100000);

        printf("\nVerifying observations...\n");
        pvp_reset(
            &env,
            pvp_route_topology_finalize(
                (const CollisionMap*)env.collision_map));
        float observations[NH_PVP_NUM_OBS];
        pvp_write_observations(observations, &env, 0);
        printf("Observation count per agent: %d\n", NH_PVP_NUM_OBS);
        printf("First 10 observations (agent 0): ");
        for (int i = 0; i < 10; i++) {
            printf("%.2f ", observations[i]);
        }
        printf("\n");

        printf("\nAction heads: %d\n", OSRS_BASE_NUM_ACTION_HEADS);
        printf("Action dims: [");
        for (int i = 0; i < OSRS_BASE_NUM_ACTION_HEADS; i++) {
            printf("%d", NH_PVP_ACTION_DIMS[i]);
            if (i < OSRS_BASE_NUM_ACTION_HEADS - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("\nDemo complete.\n");

        pvp_close(&env);
    }

    return 0;
}
