/**
 * @fileoverview Standalone demo for OSRS PvP C Environment
 *
 * Demonstrates environment initialization, stepping, and basic performance.
 * Compile: make
 * Run: ./osrs_pvp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "osrs_env.h"
#include "osrs_encounter.h"
#include "encounters/encounter_nh_pvp.h"
#include "encounters/encounter_zulrah.h"
#include "encounters/encounter_inferno.h"

#ifdef OSRS_VISUAL
#include "osrs_render.h"
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
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
    pvp_reset(env);

    while (!env->episode_over) {
        for (int agent = 0; agent < NUM_AGENTS; agent++) {
            int* actions = env->actions + agent * NUM_ACTION_HEADS;
            for (int h = 0; h < NUM_ACTION_HEADS; h++) {
                actions[h] = rand() % ACTION_HEAD_DIMS[h];
            }
        }

        pvp_step(env);

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
    printf("Benchmarking %d steps...\n", num_steps);

    clock_t start = clock();
    int episodes = 0;
    int total_steps = 0;

    while (total_steps < num_steps) {
        pvp_reset(env);
        episodes++;

        while (!env->episode_over && total_steps < num_steps) {
            for (int agent = 0; agent < NUM_AGENTS; agent++) {
                int* actions = env->actions + agent * NUM_ACTION_HEADS;
                for (int h = 0; h < NUM_ACTION_HEADS; h++) {
                    actions[h] = rand() % ACTION_HEAD_DIMS[h];
                }
            }

            pvp_step(env);
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

static void run_profile(OsrsEnv* env, const char* encounter_name) {
    printf("Profiling %s for 10 seconds...\n", encounter_name ? encounter_name : "pvp");

    if (encounter_name) {
        const EncounterDef* edef = encounter_find(encounter_name);
        if (!edef) {
            fprintf(stderr, "unknown encounter: %s\n", encounter_name);
            return;
        }
        env->encounter_def = (void*)edef;
        env->encounter_state = edef->create();

        if (strcmp(encounter_name, "zulrah") == 0) {
            CollisionMap* cmap = collision_map_load("data/zulrah.cmap");
            if (cmap) {
                edef->put_ptr(env->encounter_state, "collision_map", cmap);
                edef->put_int(env->encounter_state, "world_offset_x", 2256);
                edef->put_int(env->encounter_state, "world_offset_y", 3061);
                env->collision_map = cmap;
            }
        } else if (strcmp(encounter_name, "inferno") == 0) {
            CollisionMap* cmap = collision_map_load("data/inferno.cmap");
            if (cmap) {
                edef->put_ptr(env->encounter_state, "collision_map", cmap);
                edef->put_int(env->encounter_state, "world_offset_x", 2246);
                edef->put_int(env->encounter_state, "world_offset_y", 5315);
                env->collision_map = cmap;
            }
        }
        edef->reset(env->encounter_state, 0);
    } else {
        env->pvp_runtime.use_c_opponent = 1;
        env->pvp_runtime.opponent.type = OPP_IMPROVED;
        env->is_lms = 1;
        pvp_reset(env);
    }

    clock_t start = clock();
    double elapsed = 0;
    int total_steps = 0;
    int enc_actions[16] = {0};

    while (elapsed < 10.0) {
        if (env->encounter_def && env->encounter_state) {
            const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
            for (int h = 0; h < edef->num_action_heads; h++) {
                enc_actions[h] = rand() % edef->action_head_dims[h];
            }
            edef->step(env->encounter_state, enc_actions);
            if (edef->is_terminal(env->encounter_state)) {
                edef->reset(env->encounter_state, (uint32_t)rand());
            }
        } else {
            for (int agent = 0; agent < NUM_AGENTS; agent++) {
                int* actions = env->actions + agent * NUM_ACTION_HEADS;
                for (int h = 0; h < NUM_ACTION_HEADS; h++) {
                    actions[h] = rand() % ACTION_HEAD_DIMS[h];
                }
            }
            pvp_step(env);
            if (env->episode_over) {
                pvp_reset(env);
            }
        }

        total_steps++;
        if (total_steps % 1000 == 0) {
            clock_t now = clock();
            elapsed = (double)(now - start) / CLOCKS_PER_SEC;
        }
    }

    printf("Results:\n");
    printf("  Total steps: %d\n", total_steps);
    printf("  Time: %.3f seconds\n", elapsed);
    printf("  Steps/sec: %.0f\n", total_steps / elapsed);

    if (env->encounter_def && env->encounter_state) {
        ((const EncounterDef*)env->encounter_def)->destroy(env->encounter_state);
        env->encounter_state = NULL;
    }
}

#ifdef OSRS_VISUAL
/* replay file: binary format for pre-recorded actions.
   header: [int32 num_ticks] [uint32 rng_state], then num_ticks * num_heads int32 values. */
typedef struct {
    int* actions;      /* flat array: actions[tick * num_heads + head] */
    int  num_ticks;
    int  num_heads;
    int  current_tick;
    uint32_t rng_seed; /* RNG state at episode start — needed for deterministic replay */
} ReplayFile;

static ReplayFile* replay_load(const char* path, int num_heads) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "replay: can't open %s\n", path); return NULL; }
    int num_ticks = 0;
    uint32_t rng_seed = 12345;
    if (fread(&num_ticks, 4, 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&rng_seed, 4, 1, f) != 1) { fclose(f); return NULL; }
    ReplayFile* rf = (ReplayFile*)malloc(sizeof(ReplayFile));
    rf->num_ticks = num_ticks;
    rf->num_heads = num_heads;
    rf->current_tick = 0;
    rf->rng_seed = rng_seed;
    rf->actions = (int*)malloc(num_ticks * num_heads * sizeof(int));
    size_t n = fread(rf->actions, sizeof(int), num_ticks * num_heads, f);
    fclose(f);
    if ((int)n != num_ticks * num_heads) {
        fprintf(stderr, "replay: short read (%d/%d)\n", (int)n, num_ticks * num_heads);
        free(rf->actions); free(rf); return NULL;
    }
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

static void replay_free(ReplayFile* rf) {
    if (rf) { free(rf->actions); free(rf); }
}

typedef struct {
    OsrsEnv* env;
    const char* encounter_name;
    ReplayFile* replay;
    int start_wave;
    /* per-frame state */
    double episode_end_time;  /* >0 when holding final frame */
    int episode_ended;
} VisualState;

static void visual_frame(void* arg) {
    VisualState* vs = (VisualState*)arg;
    OsrsEnv* env = vs->env;
    RenderClient* rc = (RenderClient*)env->client;

    /* rewind: restore historical state and re-render */
    if (rc->step_back) {
        rc->step_back = 0;
        render_restore_snapshot(rc, env);
        /* if we restored the latest snapshot, exit rewind mode */
        if (rc->history_cursor >= rc->history_count - 1) {
            rc->history_cursor = -1;
        }
        pvp_render(env);
        return;
    }

    /* in rewind mode viewing history: just render, don't step */
    if (rc->history_cursor >= 0) {
        pvp_render(env);
        return;
    }

    /* episode ended: hold final frame for 2 seconds then reset */
    if (vs->episode_ended) {
        pvp_render(env);
        if (GetTime() - vs->episode_end_time >= 2.0) {
            vs->episode_ended = 0;
            render_clear_history(rc);
            effect_clear_all(rc->effects);
            flight_clear_all(rc);
            gui_reset_inventory_ui_state(&rc->gui);
            if (env->encounter_def) {
                ((const EncounterDef*)env->encounter_def)->reset(
                    env->encounter_state, (uint32_t)rand());
            } else {
                pvp_reset(env);
            }
            render_populate_entities(rc, env);
            for (int i = 0; i < rc->entity_count; i++) {
                rc->sub_x[i] = rc->entities[i].x * 128 + 64;
                rc->sub_y[i] = rc->entities[i].y * 128 + 64;
                rc->dest_x[i] = rc->sub_x[i];
                rc->dest_y[i] = rc->sub_y[i];
            }
            render_save_snapshot(rc, env);
        }
        return;
    }

    /* paused: render but don't step */
    if (rc->is_paused && !rc->step_once) {
        pvp_render(env);
        return;
    }
    rc->step_once = 0;

    /* tick pacing: keep rendering while waiting */
    if (rc->ticks_per_second > 0.0f) {
        double interval = 1.0 / rc->ticks_per_second;
        if (GetTime() - rc->last_tick_time < interval) {
            pvp_render(env);
            return;
        }
    }
    rc->last_tick_time = GetTime();

    /* step the simulation */
    render_pre_tick(rc, env);

    if (env->encounter_def && env->encounter_state) {
        /* encounter mode */
        const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
        int enc_actions[16] = {0};
        int used_human_step = 0;

        if (rc->human_input.enabled && edef->step_human_commands) {
            edef->step_human_commands(env->encounter_state, &rc->human_input);
            used_human_step = 1;
        } else if (rc->human_input.enabled) {
            /* human control: per-encounter translator */
            if (edef->translate_human_input)
                edef->translate_human_input(&rc->human_input, enc_actions,
                                            env->encounter_state);
            /* set encounter destination from human click for proper pathfinding.
               attacking an NPC cancels movement (OSRS: server stops walking
               to old dest and auto-walks toward target instead). */
            if (rc->human_input.pending_move_x >= 0 && edef->put_int) {
                edef->put_int(env->encounter_state, "player_dest_x",
                              rc->human_input.pending_move_x);
                edef->put_int(env->encounter_state, "player_dest_y",
                              rc->human_input.pending_move_y);
            } else if (rc->human_input.pending_attack && edef->put_int) {
                edef->put_int(env->encounter_state, "player_dest_x", -1);
                edef->put_int(env->encounter_state, "player_dest_y", -1);
            }
            human_input_clear_pending(&rc->human_input);
        } else if (vs->replay && replay_get_actions(vs->replay, enc_actions)) {
            /* replay mode: actions come from pre-recorded file */
        } else if (strcmp(edef->name, "zulrah") == 0) {
            zul_heuristic_actions((ZulrahState*)env->encounter_state, enc_actions);
        } else {
            for (int h = 0; h < edef->num_action_heads; h++) {
                enc_actions[h] = rand() % edef->action_head_dims[h];
            }
        }
        if (!used_human_step)
            edef->step(env->encounter_state, enc_actions);
        /* sync env->tick so renderer HP bars/splats use correct tick */
        env->tick = edef->get_tick(env->encounter_state);

        /* clear human move when player arrived at clicked destination */
        if (rc->human_input.enabled && rc->human_input.pending_move_x >= 0) {
            Player* ply = edef->get_entity(env->encounter_state, 0);
            if (ply && ply->x == rc->human_input.pending_move_x &&
                ply->y == rc->human_input.pending_move_y) {
                human_input_clear_move(&rc->human_input);
            }
        }

    } else {
        /* PvP mode */
        if (rc->human_input.enabled) {
            /* human control: translate staged clicks to PvP actions for agent 0 */
            human_to_pvp_actions(&rc->human_input,
                                  env->actions, &env->players[0], &env->players[1]);
            /* opponent still gets random actions */
            int* opp = env->actions + NUM_ACTION_HEADS;
            for (int h = 0; h < NUM_ACTION_HEADS; h++) {
                opp[h] = rand() % ACTION_HEAD_DIMS[h];
            }
            human_input_clear_pending(&rc->human_input);
        } else {
            for (int agent = 0; agent < NUM_AGENTS; agent++) {
                int* actions = env->actions + agent * NUM_ACTION_HEADS;
                for (int h = 0; h < NUM_ACTION_HEADS; h++) {
                    actions[h] = rand() % ACTION_HEAD_DIMS[h];
                }
            }
        }
        pvp_step(env);

        /* clear human move when player arrived at clicked destination */
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

    /* auto-reset on episode end */
    int is_over = env->encounter_def
        ? ((const EncounterDef*)env->encounter_def)->is_terminal(env->encounter_state)
        : env->episode_over;
    if (is_over) {
        vs->episode_ended = 1;
        vs->episode_end_time = GetTime();
    }
}

static void run_visual(OsrsEnv* env, const char* encounter_name, const char* replay_path, int start_wave) {
    env->client = NULL;

    /* set up encounter if specified, otherwise default to PvP */
    if (encounter_name) {
        const EncounterDef* edef = encounter_find(encounter_name);
        if (!edef) {
            fprintf(stderr, "unknown encounter: %s\n", encounter_name);
            return;
        }
        env->encounter_def = (void*)edef;
        env->encounter_state = edef->create();
        /* seed=0 matches training binding (uses default RNG, not explicit seed) */

        /* load encounter-specific collision map.
           world offset translates encounter-local (0,0) → world coords for cmap lookup.
           the Zulrah island collision data has ~69 walkable tiles forming the
           irregular island shape (narrow south, wide north, pillar alcoves). */
        if (strcmp(encounter_name, "zulrah") == 0) {
            CollisionMap* cmap = collision_map_load("data/zulrah.cmap");
            if (cmap) {
                edef->put_ptr(env->encounter_state, "collision_map", cmap);
                edef->put_int(env->encounter_state, "world_offset_x", 2256);
                edef->put_int(env->encounter_state, "world_offset_y", 3061);
                env->collision_map = cmap;
                fprintf(stderr, "zulrah collision map: %d regions, offset (2256, 3061)\n",
                        cmap->count);
            }
        } else if (strcmp(encounter_name, "inferno") == 0) {
            CollisionMap* cmap = collision_map_load("data/inferno.cmap");
            if (cmap) {
                edef->put_ptr(env->encounter_state, "collision_map", cmap);
                edef->put_int(env->encounter_state, "world_offset_x", 2246);
                edef->put_int(env->encounter_state, "world_offset_y", 5315);
                env->collision_map = cmap;
                fprintf(stderr, "inferno collision map: %d regions, offset (2246, 5315)\n",
                        cmap->count);
            }
        }

        if (start_wave >= 0 && edef->put_int)
            edef->put_int(env->encounter_state, "start_wave", start_wave);
        edef->reset(env->encounter_state, 0);
        fprintf(stderr, "encounter: %s (obs=%d, heads=%d)%s\n",
                edef->name, edef->obs_size, edef->num_action_heads,
                start_wave >= 0 ? "" : "");
        if (start_wave >= 0)
            fprintf(stderr, "start_wave: %d\n", start_wave);
    } else {
        env->pvp_runtime.use_c_opponent = 1;
        env->pvp_runtime.opponent.type = OPP_IMPROVED;
        env->is_lms = 1;
        pvp_reset(env);
    }

    /* load collision map from env var if set */
    const char* cmap_path = getenv("OSRS_COLLISION_MAP");
    if (cmap_path && cmap_path[0]) {
        env->collision_map = collision_map_load(cmap_path);
        if (env->collision_map) {
            fprintf(stderr, "collision map loaded: %d regions\n",
                    ((CollisionMap*)env->collision_map)->count);
        }
    }

    /* init window before main loop (WindowShouldClose needs a window) */
    pvp_render(env);
    RenderClient* rc = (RenderClient*)env->client;

    /* share collision map pointer with renderer for overlays */
    if (env->collision_map) {
        rc->collision_map = (const CollisionMap*)env->collision_map;
    }

    /* load 3D assets if available */
    rc->model_cache = model_cache_load("data/equipment.models");
    if (rc->model_cache) {
        rc->show_models = 1;
    }
    rc->anim_cache = anim_cache_load("data/equipment.anims");
    render_init_overlay_models(rc);
    /* load terrain/objects per encounter */
    if (!encounter_name) {
        rc->terrain = terrain_load("data/wilderness.terrain");
        rc->objects = objects_load("data/wilderness.objects");
        rc->npcs = objects_load("data/wilderness.npcs");
    } else if (strcmp(encounter_name, "zulrah") == 0) {
        rc->terrain = terrain_load("data/zulrah.terrain");
        rc->objects = objects_load("data/zulrah.objects");

        /* Zulrah coordinate alignment: three coordinate spaces are in play:

           1. OSRS world coords: absolute tile positions (e.g. 2256, 3061).
              terrain, objects, and collision maps are all authored in this space.

           2. encounter-local coords: the encounter arena uses (0,0) as origin.
              the encounter state, entity positions, and arena bounds all use this.

           3. raylib world coords: X = east, Y = up, Z = -north (right-handed).
              terrain_offset/objects_offset subtract the world origin so that
              encounter-local (0,0) maps to raylib (0,0).

           terrain/objects offset: subtract (2256, 3061) from world coords.
             regions (35,47)+(35,48) start at world (2240, 3008).
             the island platform is at world ~(2256, 3061), so offset = 2240+16, 3008+53.

           collision map offset: ADD (2254, 3060) to encounter-local coords.
             collision_get_flags expects world coords, so when the renderer or
             encounter queries tile (x, y) in local space, it looks up
             (x + 2254, y + 3060) in the collision map. */
        int zul_off_x = 2240 + 16;
        int zul_off_y = 3008 + 53;
        if (rc->terrain)
            terrain_offset(rc->terrain, zul_off_x, zul_off_y);
        if (rc->objects)
            objects_offset(rc->objects, zul_off_x, zul_off_y);

        rc->collision_map = (const CollisionMap*)env->collision_map;
        rc->collision_world_offset_x = 2256;
        rc->collision_world_offset_y = 3061;

        rc->npc_model_cache = model_cache_load("data/zulrah.models");
        rc->npc_anim_cache = anim_cache_load("data/zulrah.anims");
        fprintf(stderr, "zulrah: npc_models=%d, npc_anims=%d seqs\n",
                rc->npc_model_cache ? rc->npc_model_cache->count : 0,
                rc->npc_anim_cache ? rc->npc_anim_cache->seq_count : 0);
    } else if (encounter_name && strcmp(encounter_name, "inferno") == 0) {
        rc->terrain = terrain_load("data/inferno.terrain");
        rc->objects = objects_load("data/inferno.objects");
        rc->objects_zuk = objects_load("data/inferno_zuk.objects");
        /* inferno region (35,83) starts at world (2240, 5312).
           encounter uses region-local coords (10-40, 13-44).
           offset terrain/objects so local coord 0 maps to world 2240. */
        if (rc->terrain)
            terrain_offset(rc->terrain, 2246, 5315);
        if (rc->objects)
            objects_offset(rc->objects, 2246, 5315);
        if (rc->objects_zuk)
            objects_offset(rc->objects_zuk, 2246, 5315);

        rc->npc_model_cache = model_cache_load("data/inferno.models");
        rc->npc_anim_cache = anim_cache_load("data/inferno.anims");

        /* collision map for debug overlay (C key) */
        if (env->collision_map) {
            rc->collision_map = (const CollisionMap*)env->collision_map;
            rc->collision_world_offset_x = 2246;
            rc->collision_world_offset_y = 5315;
        }

        fprintf(stderr, "inferno: terrain=%s, cmap=%s, npc_models=%d, npc_anims=%d seqs\n",
                rc->terrain ? "loaded" : "MISSING",
                rc->collision_map ? "loaded" : "MISSING",
                rc->npc_model_cache ? rc->npc_model_cache->count : 0,
                rc->npc_anim_cache ? rc->npc_anim_cache->seq_count : 0);
    }

    /* populate entity pointers (also sets arena bounds from encounter) */
    render_populate_entities(rc, env);

    /* update camera target to center on the (possibly new) arena */
    rc->cam_target_x = (float)rc->arena_base_x + (float)rc->arena_width / 2.0f;
    rc->cam_target_z = -((float)rc->arena_base_y + (float)rc->arena_height / 2.0f);

    for (int i = 0; i < rc->entity_count; i++) {
        int size = rc->entities[i].npc_size > 1 ? rc->entities[i].npc_size : 1;
        rc->sub_x[i] = rc->entities[i].x * 128 + size * 64;
        rc->sub_y[i] = rc->entities[i].y * 128 + size * 64;
        rc->dest_x[i] = rc->sub_x[i];
        rc->dest_y[i] = rc->sub_y[i];
    }

    /* load replay file if specified */
    ReplayFile* replay = NULL;
    if (replay_path && env->encounter_def) {
        const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
        replay = replay_load(replay_path, edef->num_action_heads);
        /* restore RNG state from replay so sim matches training exactly */
        if (replay && edef->put_int) {
            edef->reset(env->encounter_state, 0);
            edef->put_int(env->encounter_state, "seed", (int)replay->rng_seed);
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

    /* save initial state as first snapshot */
    render_save_snapshot(rc, env);

    VisualState vs = {
        .env = env,
        .encounter_name = encounter_name,
        .replay = replay,
        .start_wave = start_wave,
        .episode_end_time = 0,
        .episode_ended = 0,
    };

#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop_arg(visual_frame, &vs, 0, 1);
#else
    while (!WindowShouldClose()) {
        visual_frame(&vs);
    }
#endif

    replay_free(replay);

    if (env->client) {
        render_destroy_client((RenderClient*)env->client);
        env->client = NULL;
    }
    if (env->encounter_def && env->encounter_state) {
        ((const EncounterDef*)env->encounter_def)->destroy(env->encounter_state);
        env->encounter_state = NULL;
    }
}
#endif

int main(int argc, char** argv) {
    int use_visual = 1;  /* default to visual mode */
    int use_profile = 0;
    int gear_tier = -1;  /* -1 = random (default LMS distribution) */
    int start_wave = -1; /* -1 = default (wave 0) */
    const char* encounter_name __attribute__((unused)) = NULL;
    const char* replay_path __attribute__((unused)) = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--visual") == 0) use_visual = 1;
        else if (strcmp(argv[i], "--profile") == 0) { use_profile = 1; use_visual = 0; }
        else if (strcmp(argv[i], "--encounter") == 0 && i + 1 < argc)
            encounter_name = argv[++i];
        else if (strcmp(argv[i], "--replay") == 0 && i + 1 < argc)
            replay_path = argv[++i];
        else if (strcmp(argv[i], "--tier") == 0 && i + 1 < argc)
            gear_tier = atoi(argv[++i]);
        else if (strcmp(argv[i], "--wave") == 0 && i + 1 < argc)
            start_wave = atoi(argv[++i]);
    }

#ifdef __EMSCRIPTEN__
    if (!encounter_name) encounter_name = "inferno";
#endif

    srand((unsigned int)time(NULL));

    OsrsEnv env;
    memset(&env, 0, sizeof(OsrsEnv));

    if (use_profile) {
        env.observations = (float*)calloc(NUM_AGENTS * SLOT_NUM_OBSERVATIONS, sizeof(float));
        env.actions = (int*)calloc(NUM_AGENTS * NUM_ACTION_HEADS, sizeof(int));
        env.rewards = (float*)calloc(NUM_AGENTS, sizeof(float));
        env.terminals = (unsigned char*)calloc(NUM_AGENTS, sizeof(unsigned char));
        env.action_masks = (unsigned char*)calloc(NUM_AGENTS * ACTION_MASK_SIZE, sizeof(unsigned char));
        env.action_masks_agents = (1 << NUM_AGENTS) - 1;
        env.ocean_io.agent_actions = env.actions;
        env.ocean_io.agent_obs = (float*)calloc(OCEAN_OBS_SIZE, sizeof(float));
        env.ocean_io.agent_rewards = env.rewards;
        env.ocean_io.agent_terminals = env.terminals;

        run_profile(&env, encounter_name);

        free(env.observations);
        free(env.actions);
        free(env.rewards);
        free(env.terminals);
        free(env.action_masks);
        free(env.ocean_io.agent_obs);
        pvp_close(&env);
        return 0;
    }

    if (use_visual) {
#ifdef OSRS_VISUAL
        /* pvp_init uses internal buffers — no malloc needed */
        pvp_init(&env);
        /* set gear tier: --tier N forces both players to tier N,
           otherwise default LMS distribution (mostly tier 0) */
        if (gear_tier >= 0 && gear_tier <= 3) {
            for (int t = 0; t < 4; t++) env.pvp_runtime.gear_tier_weights[t] = 0.0f;
            env.pvp_runtime.gear_tier_weights[gear_tier] = 1.0f;
        } else {
            /* default LMS: 60% tier 0, 25% tier 1, 10% tier 2, 5% tier 3 */
            env.pvp_runtime.gear_tier_weights[0] = 0.60f;
            env.pvp_runtime.gear_tier_weights[1] = 0.25f;
            env.pvp_runtime.gear_tier_weights[2] = 0.10f;
            env.pvp_runtime.gear_tier_weights[3] = 0.05f;
        }
        env.ocean_io.agent_actions = env.actions;
        env.ocean_io.agent_obs = env._obs_buf;
        env.ocean_io.agent_rewards = env.rewards;
        env.ocean_io.agent_terminals = env.terminals;
        run_visual(&env, encounter_name, replay_path, start_wave);
        pvp_close(&env);
#else
        fprintf(stderr, "not compiled with visual support (use: make visual)\n");
        return 1;
#endif
    } else {
        /* headless: allocate external buffers (matches original demo) */
        env.observations = (float*)calloc(NUM_AGENTS * SLOT_NUM_OBSERVATIONS, sizeof(float));
        env.actions = (int*)calloc(NUM_AGENTS * NUM_ACTION_HEADS, sizeof(int));
        env.rewards = (float*)calloc(NUM_AGENTS, sizeof(float));
        env.terminals = (unsigned char*)calloc(NUM_AGENTS, sizeof(unsigned char));
        env.action_masks = (unsigned char*)calloc(NUM_AGENTS * ACTION_MASK_SIZE, sizeof(unsigned char));
        env.action_masks_agents = (1 << NUM_AGENTS) - 1;
        env.ocean_io.agent_actions = env.actions;
        env.ocean_io.agent_obs = (float*)calloc(OCEAN_OBS_SIZE, sizeof(float));
        env.ocean_io.agent_rewards = env.rewards;
        env.ocean_io.agent_terminals = env.terminals;

        printf("OSRS PvP C Environment Demo\n\n");

        printf("Running single verbose episode...\n");
        run_random_episode(&env, 1);

        printf("\n");
        benchmark(&env, 100000);

        printf("\nVerifying observations...\n");
        pvp_reset(&env);
        printf("Observation count per agent: %d\n", SLOT_NUM_OBSERVATIONS);
        printf("First 10 observations (agent 0): ");
        for (int i = 0; i < 10; i++) {
            printf("%.2f ", env.observations[i]);
        }
        printf("\n");

        printf("\nAction heads: %d\n", NUM_ACTION_HEADS);
        printf("Action dims: [");
        for (int i = 0; i < NUM_ACTION_HEADS; i++) {
            printf("%d", ACTION_HEAD_DIMS[i]);
            if (i < NUM_ACTION_HEADS - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        printf("\nDemo complete.\n");

        free(env.observations);
        free(env.actions);
        free(env.rewards);
        free(env.terminals);
        free(env.action_masks);
        free(env.ocean_io.agent_obs);
        pvp_close(&env);
    }

    return 0;
}
