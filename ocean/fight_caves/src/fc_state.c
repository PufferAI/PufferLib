#include "fc_api.h"
#include "fc_combat.h"
#include "fc_npc.h"
#include "fc_wave.h"
#include "fc_pathfinding.h"
#include "fc_player_init.h"
#include "fc_action_internal.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * fc_state.c — State allocation, initialization, reset, rendering.
 *
 * FcState is caller-allocated (stack or heap). These functions
 * initialize and reset it. memset to zero is the canonical reset
 * mechanism — all fields must have safe zero defaults.
 */

/* ======================================================================== */
/* Arena collision map (from Void 634 cache, region 37,79, level 0)          */
/* ======================================================================== */

/*
 * Binary collision extracted via DumpFcCollision.kt from the Void 634 cache.
 * fightcaves.collision stores whole-tile blocking as 64*64 row-major bytes.
 * fightcaves.movement stores the corresponding directional wall bits.
 *
 * Loaded from resources/fight_caves/runtime in a packaged PufferLib checkout.
 * All three arena maps are required; running without one would change the
 * simulation's movement or line-of-sight rules.
 */
/* Cached arena data — loaded once and shared by all envs to avoid per-reset
 * file I/O. Each map retains separate storage and initialization state. */
typedef struct {
    uint8_t cells[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    int loaded;
} FcArenaMapCache;

static FcArenaMapCache g_collision_cache;
static FcArenaMapCache g_movement_cache;
static FcArenaMapCache g_los_cache;

static const char* const g_arena_asset_path_formats[] = {
    "resources/fight_caves/runtime/%s",
    NULL
};

static void fail_required_arena_asset(const char* filename,
                                      const char* override_name) {
    fprintf(stderr,
            "fatal: required Fight Caves arena asset '%s' is missing, "
            "unreadable, or not exactly %d bytes.\n"
            "Set %s to the asset's absolute path or install the pinned "
            "runtime bundle with 'bash ocean/fight_caves/scripts/"
            "setup-data.sh --core'.\n",
            filename, FC_ARENA_WIDTH * FC_ARENA_HEIGHT,
            override_name);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

static void load_required_arena_map(FcArenaMapCache* cache,
                                    const char* filename,
                                    const char* override_name) {
    FILE* file = NULL;
    const char* override_path;
    char path[256];
    uint8_t bytes[FC_ARENA_WIDTH * FC_ARENA_HEIGHT];
    size_t count;

    if (cache->loaded) return;
    override_path = getenv(override_name);
    if (override_path && override_path[0]) {
        file = fopen(override_path, "rb");
        if (!file) {
            fprintf(stderr,
                    "fatal: %s points to an unreadable Fight Caves arena "
                    "asset: %s\n",
                    override_name, override_path);
            fail_required_arena_asset(filename, override_name);
        }
    } else {
        for (int i = 0; !file && g_arena_asset_path_formats[i]; i++) {
            int length = snprintf(path, sizeof(path),
                                  g_arena_asset_path_formats[i], filename);
            if (length > 0 && (size_t)length < sizeof(path)) {
                file = fopen(path, "rb");
            }
        }
    }
    if (!file) fail_required_arena_asset(filename, override_name);

    count = fread(bytes, 1, sizeof(bytes), file);
    fclose(file);
    if (count != sizeof(bytes)) {
        fail_required_arena_asset(filename, override_name);
    }

    /* Binary files are row-major [y][x]; FcState maps are [x][y]. */
    for (int y = 0; y < FC_ARENA_HEIGHT; y++) {
        for (int x = 0; x < FC_ARENA_WIDTH; x++) {
            cache->cells[x][y] = bytes[y * FC_ARENA_WIDTH + x];
        }
    }
    cache->loaded = 1;
}

static void setup_arena(FcState* state) {
    load_required_arena_map(&g_collision_cache, "fightcaves.collision",
                            "FC_COLLISION_PATH");
    load_required_arena_map(&g_movement_cache, "fightcaves.movement",
                            "FC_MOVEMENT_PATH");
    load_required_arena_map(&g_los_cache, "fightcaves.los", "FC_LOS_PATH");
    memcpy(state->walkable, g_collision_cache.cells, sizeof(state->walkable));
    memcpy(state->movement_flags, g_movement_cache.cells,
           sizeof(state->movement_flags));
    memcpy(state->los_flags, g_los_cache.cells, sizeof(state->los_flags));
}

/* Player initialization — the loadout table is the combat-state authority. */

static void apply_loadout_combat_fields(FcPlayer* p,
                                        const FcLoadout* loadout) {
    p->max_hp = loadout->max_hp;
    p->max_prayer = loadout->max_prayer;
    p->attack_level = loadout->attack_lvl;
    p->strength_level = loadout->strength_lvl;
    p->defence_level = loadout->defence_lvl;
    p->ranged_level = loadout->ranged_lvl;
    p->prayer_level = loadout->prayer_lvl;
    p->magic_level = loadout->magic_lvl;
    p->weapon_kind = loadout->weapon_kind;
    p->weapon_uses_ammo = loadout->weapon_uses_ammo;
    p->crystal_piece_mask = loadout->crystal_piece_mask;
    p->weapon_speed = loadout->weapon_speed;
    p->weapon_range = loadout->weapon_range;
    p->ranged_attack_bonus = loadout->ranged_atk;
    p->ranged_strength_bonus = loadout->ranged_str;
    p->defence_stab = loadout->def_stab;
    p->defence_slash = loadout->def_slash;
    p->defence_crush = loadout->def_crush;
    p->defence_magic = loadout->def_magic;
    p->defence_ranged = loadout->def_ranged;
    p->prayer_bonus = loadout->prayer_bonus;
    p->ammo_count = loadout->ammo;
}

static void init_player(FcPlayer* p) {
    const FcLoadout* loadout = &FC_LOADOUTS[FC_ACTIVE_LOADOUT];
    apply_loadout_combat_fields(p, loadout);
    p->x = FC_ARENA_WIDTH / 2;
    p->y = FC_ARENA_HEIGHT / 2;
    p->current_hp = p->max_hp;
    p->current_prayer = p->max_prayer;
    p->prayer = PRAYER_NONE;
    p->prayer_at_tick_start = PRAYER_NONE;
    p->sharks_remaining = FC_MAX_SHARKS;
    p->prayer_doses_remaining = FC_MAX_PRAYER_DOSES;
    p->attack_timer = 0;
    p->food_timer = 0;
    p->potion_timer = 0;
    p->combo_timer = 0;
    p->run_energy = FC_RUN_ENERGY_MAX;
    p->is_running = 1;
    p->hp_regen_counter = 0;
    p->route_len = 0;
    p->route_idx = 0;
    p->attack_target_idx = -1;
    p->approach_target = 0;
    p->approach_target_x = -1;
    p->approach_target_y = -1;
    p->approach_target_size = 0;
}

/* ======================================================================== */
/* Lifecycle                                                                 */
/* ======================================================================== */

static void validate_npc_table_or_abort(void) {
    for (int npc_type = NPC_TZ_KIH; npc_type < NPC_TYPE_COUNT; npc_type++) {
        const FcNpcStats* stats = fc_npc_get_stats(npc_type);
        if (fc_npc_stats_valid(stats)) continue;

        fprintf(stderr,
                "fc_init: invalid NPC maxima for type %d: melee=%d ranged=%d magic=%d tenths\n",
                npc_type, stats->melee_max_hit_tenths,
                stats->ranged_max_hit_tenths,
                stats->magic_max_hit_tenths);
        abort();
    }
}

static void validate_loadout_table_or_abort(void) {
    if (FC_ACTIVE_LOADOUT < 0 || FC_ACTIVE_LOADOUT >= FC_NUM_LOADOUTS) {
        fprintf(stderr, "fc_init: active loadout %d is outside [0,%d)\n",
                FC_ACTIVE_LOADOUT, FC_NUM_LOADOUTS);
        abort();
    }

    for (int loadout_id = 0; loadout_id < FC_NUM_LOADOUTS; loadout_id++) {
        const FcLoadout* loadout = &FC_LOADOUTS[loadout_id];
        int valid = loadout->max_hp > 0 && loadout->max_prayer > 0 &&
            loadout->attack_lvl >= 1 && loadout->strength_lvl >= 1 &&
            loadout->defence_lvl >= 1 && loadout->ranged_lvl >= 1 &&
            loadout->prayer_lvl >= 1 && loadout->magic_lvl >= 1 &&
            loadout->weapon_kind >= FC_WEAPON_GENERIC_RANGED &&
            loadout->weapon_kind <= FC_WEAPON_BOW_OF_FAERDHINEN &&
            (loadout->weapon_uses_ammo == 0 ||
             loadout->weapon_uses_ammo == 1) &&
            loadout->ammo >= 0 &&
            (loadout->crystal_piece_mask & ~FC_CRYSTAL_PIECE_ALL) == 0 &&
            loadout->equipment_count >= 0 &&
            loadout->equipment_count <= FC_LOADOUT_EQUIP_MAX &&
            loadout->model_item_count >= 0 &&
            loadout->model_item_count <= FC_LOADOUT_MODEL_ITEM_MAX;

        FcPlayer player = {0};
        apply_loadout_combat_fields(&player, loadout);
        if (fc_player_ranged_base_max_hit_hp(&player) <= 0) valid = 0;
        for (int npc_type = NPC_TZ_KIH;
             valid && npc_type < NPC_TYPE_COUNT; npc_type++) {
            FcNpc target = {0};
            target.npc_type = npc_type;
            if (fc_player_ranged_final_max_hit_hp(&player, &target) <= 0)
                valid = 0;
        }

        if (valid) continue;
        fprintf(stderr,
                "fc_init: invalid loadout %d (skills=%d/%d/%d/%d/%d/%d weapon=%d ammo=%d/%d crystal=%d)\n",
                loadout_id, loadout->attack_lvl, loadout->strength_lvl,
                loadout->defence_lvl, loadout->ranged_lvl,
                loadout->prayer_lvl, loadout->magic_lvl,
                loadout->weapon_kind, loadout->weapon_uses_ammo,
                loadout->ammo, loadout->crystal_piece_mask);
        abort();
    }
}

void fc_init(FcState* state) {
    validate_npc_table_or_abort();
    validate_loadout_table_or_abort();
    memset(state, 0, sizeof(FcState));
    state->active_loadout = FC_ACTIVE_LOADOUT;
}

void fc_reset(FcState* state, uint32_t seed) {
    /* Zero everything first — ensures no stale state, padding is clean */
    memset(state, 0, sizeof(FcState));

    /* Seed RNG */
    fc_rng_seed(state, seed);

    /* Select random rotation */
    state->rotation_id = fc_rng_int(state, FC_NUM_ROTATIONS);

    /* Setup arena */
    setup_arena(state);

    /* Initialize player */
    init_player(&state->player);
    state->active_loadout = FC_ACTIVE_LOADOUT;
    state->render_events.player_attack_target_npc_slot = -1;
    state->render_events.player_move_start_x = state->player.x;
    state->render_events.player_move_start_y = state->player.y;

    /* Spawn wave 1 NPCs */
    state->current_wave = 1;
    state->next_spawn_index = 0;
    state->wave_start_tick = 0;
    fc_wave_spawn(state, 1);
}

void fc_step(FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
    if (state->terminal != TERMINAL_NONE) return;  /* episode over */
    fc_tick(state, actions);
}

void fc_request_set_running(FcState* state, int enabled) {
    if (!state || state->terminal != TERMINAL_NONE) return;
    state->player.is_running = enabled &&
        state->player.run_energy >= FC_RUN_ENERGY_MIN_START;
}

void fc_destroy(FcState* state) {
    /* Currently no heap allocations. Zero for safety. */
    memset(state, 0, sizeof(FcState));
}

/* ======================================================================== */
/* Observation / Mask / Reward                                               */
/* ======================================================================== */

/* Sort helper: indices of active NPCs, sorted by (distance, spawn_index) */
static int compare_npc_slots(const void* a, const void* b, const FcState* state) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    int da = fc_distance_to_npc(state->player.x, state->player.y, &state->npcs[ia]);
    int db = fc_distance_to_npc(state->player.x, state->player.y, &state->npcs[ib]);
    if (da != db) return da - db;
    return state->npcs[ia].spawn_index - state->npcs[ib].spawn_index;
}

/* Simple insertion sort for small arrays (max 16 elements) */
static void sort_npc_indices(int* indices, int count, const FcState* state) {
    for (int i = 1; i < count; i++) {
        int key = indices[i];
        int j = i - 1;
        while (j >= 0 && compare_npc_slots(&key, &indices[j], state) < 0) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

int fc_visible_npc_indices(const FcState* state, int out_indices[FC_VISIBLE_NPCS]) {
    int active_indices[FC_MAX_NPCS];
    int active_count = 0;

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (state->npcs[i].active && !state->npcs[i].is_dead) {
            active_indices[active_count++] = i;
        }
    }

    sort_npc_indices(active_indices, active_count, state);

    int visible = (active_count < FC_VISIBLE_NPCS) ? active_count : FC_VISIBLE_NPCS;
    for (int slot = 0; slot < visible; slot++) {
        out_indices[slot] = active_indices[slot];
    }
    return visible;
}

static int move_action_valid(const FcState* state, int action) {
    const FcPlayer* p = &state->player;

    if (action < 0 || action >= FC_MOVE_DIM) return 0;
    if (action == FC_MOVE_IDLE) return 1;
    if (action >= FC_MOVE_RUN_N && !fc_player_can_run(p)) {
        return 0;
    }

    int tx = p->x;
    int ty = p->y;
    int max_steps = (action >= FC_MOVE_RUN_N) ? 2 : 1;
    return fc_move_toward(&tx, &ty, FC_MOVE_DX[action], FC_MOVE_DY[action],
                          max_steps, state->walkable,
                          state->movement_flags) > 0;
}

static int attack_action_valid(const FcState* state, int action) {
    int visible_indices[FC_VISIBLE_NPCS];
    int visible;
    int slot;

    if (action < 0 || action >= FC_ATTACK_DIM) return 0;
    if (action == FC_ATTACK_NONE) return 1;

    visible = fc_visible_npc_indices(state, visible_indices);
    slot = action - 1;
    return slot < visible;
}

static int prayer_action_valid(int action) {
    return action >= 0 && action < FC_PRAYER_DIM;
}

int fc_eat_action_valid(const FcState* state, int action) {
    const FcPlayer* p = &state->player;

    if (action < 0 || action >= FC_EAT_DIM) return 0;
    if (action == FC_EAT_NONE) return 1;
    if (action == FC_EAT_SHARK) {
        return p->sharks_remaining > 0 &&
               p->food_timer <= 0 &&
               p->current_hp < p->max_hp;
    }
    if (action == FC_EAT_COMBO) {
        return p->sharks_remaining > 0 &&
               p->combo_timer <= 0 &&
               p->current_hp < p->max_hp;
    }
    return 0;
}

int fc_drink_action_valid(const FcState* state, int action) {
    const FcPlayer* p = &state->player;

    if (action < 0 || action >= FC_DRINK_DIM) return 0;
    if (action == FC_DRINK_NONE) return 1;
    if (action == FC_DRINK_PRAYER_POT) {
        return p->prayer_doses_remaining > 0 &&
               p->potion_timer <= 0 &&
               p->current_prayer < p->max_prayer;
    }
    return 0;
}

static int attack_style_summary_idx(int style) {
    switch (style) {
        case ATTACK_MELEE:  return 0;
        case ATTACK_RANGED: return 1;
        case ATTACK_MAGIC:  return 2;
        default:            return -1;
    }
}

static float normalize_incoming_count(int count) {
    if (count <= 0) return 0.0f;
    if (count >= 4) return 1.0f;
    return (float)count / 4.0f;
}

static float clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static float normalize_prayer_drain_counter(const FcPlayer* p) {
    int resistance = 60 + 2 * p->prayer_bonus;
    if (resistance <= 0) return 0.0f;
    float normalized = (float)p->prayer_drain_counter / (float)resistance;
    if (normalized < 0.0f) return 0.0f;
    if (normalized > 1.0f) return 1.0f;
    return normalized;
}

static float normalize_npc_prayer_drain(const FcNpc* npc) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    int maximum = stats->prayer_drain;
    if (npc->npc_type == NPC_TZ_KIH) {
        maximum += fc_npc_max_hit_tenths_for_style(stats, ATTACK_MELEE);
    }
    if (maximum <= 0) return 0.0f;
    return clamp01((float)npc->prayer_drain_dealt_this_tick / (float)maximum);
}

static float normalize_npc_heal_cooldown(const FcNpc* npc) {
    if (npc->npc_type == NPC_YT_MEJKOT) {
        if (npc->attack_speed <= 0) return 0.0f;
        return clamp01((float)npc->attack_timer / (float)npc->attack_speed);
    }

    if (npc->npc_type == NPC_YT_HURKOT && !npc->healer_distracted) {
        const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
        if (stats->heal_interval <= 0) return 0.0f;
        return clamp01((float)npc->heal_timer / (float)stats->heal_interval);
    }

    return 0.0f;
}

static int pending_hit_prayer_actionable(const FcState* state,
                                         const FcPendingHit* ph) {
    if (!ph->active) return 0;
    if (ph->prayer_snapshot >= 0) return 0;
    if (ph->prayer_lock_tick < 0) return 0;
    if (state->tick >= ph->prayer_lock_tick) return 0;
    return attack_style_summary_idx(ph->attack_style) >= 0;
}

static float pending_hit_prayer_deadline_urgency(const FcState* state,
                                                 const FcPendingHit* ph) {
    if (!pending_hit_prayer_actionable(state, ph)) return 0.0f;

    int ticks_until_lock = ph->prayer_lock_tick - state->tick;
    if (ticks_until_lock > 4) ticks_until_lock = 4;

    return (float)(5 - ticks_until_lock) / 4.0f;
}

/* Distance-only attack-style telegraph: what style would this NPC throw if it
 * attacked right now from its current position? Does NOT check LOS; the LOS
 * bit is a separate obs feature so the agent can distinguish "melee threat,
 * safespotted" (style=MELEE, LOS=0) from "melee threat, can hit me" (style=MELEE,
 * LOS=1). Returns ATTACK_NONE for empty slots, an untagged Yt-HurKot, and
 * stochastic style choices before the NPC has committed an attack. */
static int npc_telegraph_style(const FcState* state, const FcNpc* npc) {
    if (!npc->active || npc->is_dead) return ATTACK_NONE;
    if (npc->npc_type == NPC_YT_HURKOT) {
        return npc->healer_distracted ? ATTACK_MELEE : ATTACK_NONE;
    }

    if (npc->npc_type == NPC_TZTOK_JAD) {
        /* Jad alternates magic/ranged randomly at commit; read the committed
         * style from the queued pending_hit. No prediction before commit. */
        const FcPlayer* p = &state->player;
        for (int i = 0; i < p->num_pending_hits; i++) {
            const FcPendingHit* ph = &p->pending_hits[i];
            if (ph->active && ph->source_npc_idx >= 0 &&
                state->npcs[ph->source_npc_idx].npc_type == NPC_TZTOK_JAD) {
                return ph->attack_style;
            }
        }
        return ATTACK_NONE;
    }

    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    int can_melee = fc_npc_can_melee_player(state->player.x, state->player.y,
                                            npc->x, npc->y, npc->size,
                                            state->walkable,
                                            state->movement_flags);
    if (npc->npc_type == NPC_KET_ZEK && can_melee &&
        fc_has_los_between_areas(
            npc->x, npc->y, npc->size,
            state->player.x, state->player.y, 1, state->los_flags)) {
        return ATTACK_NONE;
    }
    /* Tok-Xil telegraphs melee when adjacent. Ket-Zek returns NONE above while
     * both adjacent styles remain possible. Pure melee NPCs (Yt-MejKot,
     * Tz-Kih, Tz-Kek) telegraph MELEE even when far
     * — they'll close the gap and that's what they'll hit with. */
    if (can_melee && (stats->melee_max_hit_tenths > 0 ||
                      npc->attack_style == ATTACK_MELEE)) {
        return ATTACK_MELEE;
    }
    return npc->attack_style;
}

static int npc_type_obs_offset(int npc_type) {
    switch (npc_type) {
        case NPC_TZ_KIH: return FC_NPC_TYPE_TZ_KIH;
        case NPC_TZ_KEK: return FC_NPC_TYPE_TZ_KEK;
        case NPC_TZ_KEK_SM: return FC_NPC_TYPE_TZ_KEK_SM;
        case NPC_TOK_XIL: return FC_NPC_TYPE_TOK_XIL;
        case NPC_YT_MEJKOT: return FC_NPC_TYPE_YT_MEJKOT;
        case NPC_KET_ZEK: return FC_NPC_TYPE_KET_ZEK;
        case NPC_TZTOK_JAD: return FC_NPC_TYPE_TZTOK_JAD;
        case NPC_YT_HURKOT: return FC_NPC_TYPE_YT_HURKOT;
        default: return -1;
    }
}

static int npc_kill_reward_eligible(const FcNpc* npc) {
    return npc->npc_type != NPC_YT_HURKOT ||
           !npc->is_respawned_jad_healer;
}

static int rewardable_npc_kills_this_tick(const FcState* state) {
    int count = state->npcs_killed_this_tick -
                state->respawned_jad_healers_killed_this_tick;
    return count > 0 ? count : 0;
}

void fc_write_obs(const FcState* state, float* out) {
    memset(out, 0, sizeof(float) * FC_TOTAL_OBS);

    const FcPlayer* p = &state->player;
    int incoming_counts[3][3] = {{0}};
    float prayer_deadline_urgency[3] = {0.0f, 0.0f, 0.0f};

    /* Compact incoming-hit timeline summary.
     * Counts by style for hits landing in 1, 2, and 3 ticks. This gives the
     * policy a relative timing signal without leaking absolute episode clocks.
     * Prayer deadline urgency is separate: it marks pending hits whose prayer
     * snapshot has not locked yet, which is the actual decision window. */
    for (int hi = 0; hi < p->num_pending_hits; hi++) {
        const FcPendingHit* ph = &p->pending_hits[hi];
        if (!ph->active) continue;
        int style_idx = attack_style_summary_idx(ph->attack_style);
        if (style_idx >= 0 && pending_hit_prayer_actionable(state, ph)) {
            float urgency = pending_hit_prayer_deadline_urgency(state, ph);
            if (urgency > prayer_deadline_urgency[style_idx]) {
                prayer_deadline_urgency[style_idx] = urgency;
            }
        }
        if (ph->ticks_remaining < 1 || ph->ticks_remaining > 3) continue;
        if (style_idx < 0) continue;
        int bucket = ph->ticks_remaining - 1;
        if (incoming_counts[bucket][style_idx] < 4) {
            incoming_counts[bucket][style_idx]++;
        }
    }

    /* Player features */
    float* player = out + FC_OBS_PLAYER_START;
    player[FC_OBS_PLAYER_HP]        = (p->max_hp > 0) ? (float)p->current_hp / (float)p->max_hp : 0.0f;
    player[FC_OBS_PLAYER_PRAYER]    = (p->max_prayer > 0) ? (float)p->current_prayer / (float)p->max_prayer : 0.0f;
    player[FC_OBS_PLAYER_X]         = (float)p->x / (float)FC_ARENA_WIDTH;
    player[FC_OBS_PLAYER_Y]         = (float)p->y / (float)FC_ARENA_HEIGHT;
    player[FC_OBS_PLAYER_ATK_TIMER] = (p->weapon_speed > 0)
        ? (float)p->attack_timer / (float)p->weapon_speed : 0.0f;
    player[FC_OBS_PLAYER_PRAY_MEL]  = (p->prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_PRAY_RNG]  = (p->prayer == PRAYER_PROTECT_RANGE) ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_PRAY_MAG]  = (p->prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_SHARKS]    = (float)p->sharks_remaining / (float)FC_MAX_SHARKS;
    player[FC_OBS_PLAYER_DOSES]     = (float)p->prayer_doses_remaining / (float)FC_MAX_PRAYER_DOSES;
    player[FC_OBS_PLAYER_IN_MEL_1T] = normalize_incoming_count(incoming_counts[0][0]);
    player[FC_OBS_PLAYER_IN_RNG_1T] = normalize_incoming_count(incoming_counts[0][1]);
    player[FC_OBS_PLAYER_IN_MAG_1T] = normalize_incoming_count(incoming_counts[0][2]);
    player[FC_OBS_PLAYER_IN_MEL_2T] = normalize_incoming_count(incoming_counts[1][0]);
    player[FC_OBS_PLAYER_IN_RNG_2T] = normalize_incoming_count(incoming_counts[1][1]);
    player[FC_OBS_PLAYER_IN_MAG_2T] = normalize_incoming_count(incoming_counts[1][2]);
    player[FC_OBS_PLAYER_TARGET]    = 0.0f;  /* filled after NPC slot computation below */
    player[FC_OBS_PLAYER_PRAY_DDL_MEL] = prayer_deadline_urgency[0];
    player[FC_OBS_PLAYER_PRAY_DDL_RNG] = prayer_deadline_urgency[1];
    player[FC_OBS_PLAYER_PRAY_DDL_MAG] = prayer_deadline_urgency[2];
    player[FC_OBS_PLAYER_PRAYER_LOST] = (p->max_prayer > 0)
        ? clamp01((float)state->prayer_lost_this_tick / (float)p->max_prayer)
        : 0.0f;
    player[FC_OBS_PLAYER_OVERHEAD_PRAYER_LOST] =
        state->overhead_prayer_lost_this_tick > 0 ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_RUN_ENERGY] =
        clamp01((float)p->run_energy / (float)FC_RUN_ENERGY_MAX);

    /* NPC slot selection: gather active NPCs, sort, take first 8 */
    int active_indices[FC_VISIBLE_NPCS];
    int visible = fc_visible_npc_indices(state, active_indices);
    for (int slot = 0; slot < visible; slot++) {
        const FcNpc* n = &state->npcs[active_indices[slot]];
        float* npc_out = out + FC_OBS_NPC_START + slot * FC_OBS_NPC_STRIDE;

        npc_out[FC_NPC_VALID]         = 1.0f;
        npc_out[FC_NPC_X]             = (float)n->x / (float)FC_ARENA_WIDTH;
        npc_out[FC_NPC_Y]             = (float)n->y / (float)FC_ARENA_HEIGHT;
        npc_out[FC_NPC_HP]            = (n->max_hp > 0) ? (float)n->current_hp / (float)n->max_hp : 0.0f;
        npc_out[FC_NPC_DISTANCE]      =
            (float)fc_distance_to_npc(p->x, p->y, n) / (float)FC_ARENA_WIDTH;
        float has_los = (float)fc_has_los_between_areas(
            p->x, p->y, 1, n->x, n->y, n->size, state->los_flags);
        int tele = npc_telegraph_style(state, n);
        npc_out[FC_NPC_TELE_MELEE]    = (tele == ATTACK_MELEE)  ? 1.0f : 0.0f;
        npc_out[FC_NPC_TELE_RANGED]   = (tele == ATTACK_RANGED) ? 1.0f : 0.0f;
        npc_out[FC_NPC_TELE_MAGIC]    = (tele == ATTACK_MAGIC)  ? 1.0f : 0.0f;
        npc_out[FC_NPC_ATK_TIMER]     = (n->attack_speed > 0) ? (float)n->attack_timer / (float)n->attack_speed : 0.0f;
        npc_out[FC_NPC_LOS]           = has_los;
        npc_out[FC_NPC_PRAYER_DRAIN_DEALT] = normalize_npc_prayer_drain(n);
        npc_out[FC_NPC_HEAL_RECEIVED] = (n->max_hp > 0)
            ? clamp01((float)n->healing_received_this_tick / (float)n->max_hp)
            : 0.0f;
        npc_out[FC_NPC_HEAL_GIVEN] = (n->heal_amount > 0)
            ? clamp01((float)n->healing_given_this_tick / (float)n->heal_amount)
            : 0.0f;
        npc_out[FC_NPC_HEALED_BY_MEJKOT] =
            n->healed_by_mejkot_this_tick ? 1.0f : 0.0f;
        npc_out[FC_NPC_HEALED_BY_HURKOT] =
            n->healed_by_hurkot_this_tick ? 1.0f : 0.0f;
        npc_out[FC_NPC_HEALED_SELF] = n->healed_self_this_tick ? 1.0f : 0.0f;
        npc_out[FC_NPC_TARGETS_PLAYER] =
            (n->npc_type != NPC_YT_HURKOT || n->healer_distracted) ? 1.0f : 0.0f;
        npc_out[FC_NPC_HEAL_COOLDOWN] = normalize_npc_heal_cooldown(n);
        npc_out[FC_NPC_KILL_REWARD_ELIGIBLE] =
            npc_kill_reward_eligible(n) ? 1.0f : 0.0f;
        int type_offset = npc_type_obs_offset(n->npc_type);
        if (type_offset >= 0) {
            npc_out[type_offset] = 1.0f;
        }

        /* Pending attack from this NPC — scan player's pending hits */
        npc_out[FC_NPC_PENDING_STYLE] = 0.0f;
        npc_out[FC_NPC_PENDING_TICKS] = 0.0f;
        npc_out[FC_NPC_PENDING_PRAYER_WINDOW] = 0.0f;
        npc_out[FC_NPC_PENDING_PRAYER_DEADLINE] = 0.0f;
        for (int hi = 0; hi < p->num_pending_hits; hi++) {
            const FcPendingHit* ph = &p->pending_hits[hi];
            if (ph->active && ph->source_npc_idx == active_indices[slot]) {
                npc_out[FC_NPC_PENDING_STYLE] = (float)ph->attack_style / 3.0f;
                npc_out[FC_NPC_PENDING_TICKS] = (float)ph->ticks_remaining / 10.0f;
                if (pending_hit_prayer_actionable(state, ph)) {
                    npc_out[FC_NPC_PENDING_PRAYER_WINDOW] = 1.0f;
                    npc_out[FC_NPC_PENDING_PRAYER_DEADLINE] =
                        pending_hit_prayer_deadline_urgency(state, ph);
                }
                break;  /* report first pending hit from this NPC */
            }
        }
    }
    /* Remaining NPC slots already zeroed by memset */

    /* Player target: which visible NPC slot is the current attack target */
    if (p->attack_target_idx >= 0) {
        for (int s = 0; s < visible; s++) {
            if (active_indices[s] == p->attack_target_idx) {
                player[FC_OBS_PLAYER_TARGET] = (float)(s + 1) / 8.0f;
                break;
            }
        }
    }

    /* Wave/meta features */
    float* meta = out + FC_OBS_META_START;
    meta[FC_OBS_META_WAVE]       = (float)state->current_wave / (float)FC_NUM_WAVES;
    meta[FC_OBS_META_ROTATION]   = (float)state->rotation_id / (float)FC_NUM_ROTATIONS;
    meta[FC_OBS_META_REMAINING]  = (float)state->npcs_remaining / (float)FC_MAX_NPCS;
    meta[FC_OBS_META_PRAY_DRAIN] = normalize_prayer_drain_counter(p);
    meta[FC_OBS_META_IN_MEL_3T]  = normalize_incoming_count(incoming_counts[2][0]);
    meta[FC_OBS_META_IN_RNG_3T]  = normalize_incoming_count(incoming_counts[2][1]);
    meta[FC_OBS_META_IN_MAG_3T]  = normalize_incoming_count(incoming_counts[2][2]);
    meta[FC_OBS_META_DMG_T_TICK] = (p->max_hp > 0) ? (float)state->damage_taken_this_tick / (float)p->max_hp : 0.0f;
    meta[FC_OBS_META_WAVE_CLR]   = (float)state->wave_just_cleared;
    meta[FC_OBS_META_CAVE_PROG]  = clamp01(state->progress_cave_progress);
    meta[FC_OBS_META_WAVE_PROG]  = clamp01(state->progress_current_wave_progress);
    meta[FC_OBS_META_WORK_REM]   = (state->progress_required_work_start > 0.0f)
        ? clamp01(state->progress_required_work_remaining /
                  state->progress_required_work_start)
        : 0.0f;
    meta[FC_OBS_META_NO_PROG]    = clamp01((float)state->progress_ticks_since_positive / 2400.0f);
    meta[FC_OBS_META_NPC_HEALING] = (state->progress_required_work_start > 0.0f)
        ? clamp01((float)state->npc_heal_amount_this_tick /
                  state->progress_required_work_start)
        : 0.0f;
    meta[FC_OBS_META_REWARDABLE_NPC_KILL] =
        rewardable_npc_kills_this_tick(state) > 0 ? 1.0f : 0.0f;

    /* Reward features (at offset FC_REWARD_START) — written by fc_write_reward_features */
    fc_write_reward_features(state, out + FC_REWARD_START);
}

void fc_apply_obs_ablation(float* out,
                           int ablate_npc_distance,
                           int ablate_incoming_aggregates,
                           int ablate_npc_valid) {
    if (ablate_npc_distance) {
        for (int s = 0; s < FC_OBS_NPC_SLOTS; s++) {
            out[FC_OBS_NPC_START + s * FC_OBS_NPC_STRIDE + FC_NPC_DISTANCE] = 0.0f;
        }
    }
    if (ablate_incoming_aggregates) {
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MEL_1T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_RNG_1T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MAG_1T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MEL_2T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_RNG_2T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MAG_2T] = 0.0f;
        out[FC_OBS_META_START + FC_OBS_META_IN_MEL_3T] = 0.0f;
        out[FC_OBS_META_START + FC_OBS_META_IN_RNG_3T] = 0.0f;
        out[FC_OBS_META_START + FC_OBS_META_IN_MAG_3T] = 0.0f;
    }
    if (ablate_npc_valid) {
        for (int s = 0; s < FC_OBS_NPC_SLOTS; s++) {
            out[FC_OBS_NPC_START + s * FC_OBS_NPC_STRIDE + FC_NPC_VALID] = 0.0f;
        }
    }
}

void fc_write_reward_features(const FcState* state, float* out) {
    memset(out, 0, sizeof(float) * FC_REWARD_FEATURES);

    out[FC_RWD_DAMAGE_DEALT]     = (float)state->damage_dealt_this_tick / 1000.0f;
    out[FC_RWD_DAMAGE_TAKEN]     = (state->player.max_hp > 0) ?
                                   (float)state->damage_taken_this_tick / (float)state->player.max_hp : 0.0f;
    out[FC_RWD_NPC_KILL]         = (float)rewardable_npc_kills_this_tick(state);
    out[FC_RWD_WAVE_CLEAR]       = (float)state->wave_just_cleared;
    out[FC_RWD_JAD_DAMAGE]       = (float)state->jad_damage_this_tick / 1000.0f;
    out[FC_RWD_JAD_KILL]         = (float)state->jad_killed;
    out[FC_RWD_PLAYER_DEATH]     = (state->terminal == TERMINAL_PLAYER_DEATH) ? 1.0f : 0.0f;
    out[FC_RWD_CAVE_COMPLETE]    = (state->terminal == TERMINAL_CAVE_COMPLETE) ? 1.0f : 0.0f;
    out[FC_RWD_FOOD_USED]        = (float)state->food_used_this_tick;
    out[FC_RWD_PRAYER_POT_USED]  = (float)state->prayer_potion_used_this_tick;
    out[FC_RWD_CORRECT_JAD_PRAY] = (float)state->correct_jad_prayer;
    out[FC_RWD_WRONG_JAD_PRAY]   = (float)state->wrong_jad_prayer;
    out[FC_RWD_INVALID_ACTION]   = (float)state->invalid_action_this_tick;
    out[FC_RWD_MOVEMENT]         = (float)state->movement_this_tick;
    out[FC_RWD_IDLE]             = (float)state->idle_this_tick;
    out[FC_RWD_TICK_PENALTY]     = 1.0f;  /* always fires */
    out[FC_RWD_CORRECT_DANGER_PRAY] = (float)state->correct_danger_prayer;
    out[FC_RWD_WRONG_DANGER_PRAY]   = (float)state->wrong_danger_prayer;
    out[FC_RWD_ATTACK_ATTEMPT]      = (float)state->attack_attempt_this_tick;
    out[FC_RWD_PRAYER_LOST]         = (float)state->prayer_lost_this_tick / 10.0f;
}

void fc_action_invalid_classes(const FcState* state,
                               const int actions[FC_NUM_ACTION_HEADS],
                               int out_classes[FC_INVALID_ACTION_CLASS_COUNT]) {
    /* Keep this aligned with the Puffer-facing policy mask surface:
     * move, attack, and prayer only. Consumable and path-target heads remain
     * canonical core actions, but are not emitted by the no-supplies policy. */
    out_classes[FC_INVALID_ACTION_MOVE] = !move_action_valid(state, actions[0]);
    out_classes[FC_INVALID_ACTION_ATTACK] = !attack_action_valid(state, actions[1]);
    out_classes[FC_INVALID_ACTION_PRAYER] = !prayer_action_valid(actions[2]);
}

void fc_write_mask(const FcState* state, float* out) {
    /* Set all to valid, then mask invalid */
    for (int i = 0; i < FC_ACTION_MASK_SIZE; i++) {
        out[i] = 1.0f;
    }

    /* MOVE: idle always valid. Walk/run directions masked if destination not walkable */
    for (int m = 1; m < FC_MOVE_DIM; m++) {
        if (!move_action_valid(state, m)) {
            out[FC_MASK_MOVE_START + m] = 0.0f;
        }
    }

    /* ATTACK: slot 0 (none) always valid. Slots 1-8 masked if no NPC in that slot */
    for (int attack = FC_ATTACK_NONE + 1; attack < FC_ATTACK_DIM; attack++) {
        if (!attack_action_valid(state, attack)) {
            out[FC_MASK_ATTACK_START + attack] = 0.0f;
        }
    }

    /* PRAYER: leave fully unmasked.
     * Prayer toggles may be legal even when they are redundant or no-op, and
     * the policy should learn those costs from the environment rather than
     * having them hidden by the mask. */

    /* EAT */
    if (!fc_eat_action_valid(state, FC_EAT_SHARK)) {
        out[FC_MASK_EAT_START + FC_EAT_SHARK] = 0.0f;
    }
    if (!fc_eat_action_valid(state, FC_EAT_COMBO)) {
        out[FC_MASK_EAT_START + FC_EAT_COMBO] = 0.0f;
    }

    /* DRINK */
    if (!fc_drink_action_valid(state, FC_DRINK_PRAYER_POT)) {
        out[FC_MASK_DRINK_START + FC_DRINK_PRAYER_POT] = 0.0f;
    }
}

int fc_is_terminal(const FcState* state) {
    return state->terminal != TERMINAL_NONE;
}

/* ======================================================================== */
/* Render entities                                                           */
/* ======================================================================== */

void fc_fill_render_entities(const FcState* state, FcRenderEntity* entities, int* count) {
    int idx = 0;

    /* Entity 0: player */
    FcRenderEntity* pe = &entities[idx++];
    memset(pe, 0, sizeof(FcRenderEntity));
    pe->entity_type = ENTITY_PLAYER;
    pe->x = state->player.x;
    pe->y = state->player.y;
    pe->size = 1;
    pe->current_hp = state->player.current_hp;
    pe->max_hp = state->player.max_hp;
    pe->prayer = state->player.prayer;
    pe->damage_taken_this_tick = state->player.damage_taken_this_tick;
    pe->hit_landed_this_tick = state->player.hit_landed_this_tick;

    /* Active NPCs + NPCs that just died this tick (for death hitsplat visibility) */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];
        if (!n->active && !n->died_this_tick) continue;

        FcRenderEntity* ne = &entities[idx++];
        memset(ne, 0, sizeof(FcRenderEntity));
        ne->entity_type = ENTITY_NPC;
        ne->npc_type = n->npc_type;
        ne->x = n->x;
        ne->y = n->y;
        ne->size = n->size;
        ne->current_hp = n->current_hp;
        ne->max_hp = n->max_hp;
        ne->attack_style = n->attack_style;
        ne->is_dead = n->is_dead;
        ne->damage_taken_this_tick = n->damage_taken_this_tick;
        ne->healing_received_this_tick = n->healing_received_this_tick;
        ne->died_this_tick = n->died_this_tick;
        ne->npc_slot = i;
    }

    *count = idx;
}

void fc_fill_render_events(const FcState* state, FcRenderEvents* events) {
    if (!state || !events) return;
    *events = state->render_events;
}
