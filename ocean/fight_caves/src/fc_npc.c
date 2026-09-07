#include "fc_npc.h"
#include "fc_combat.h"
#include "fc_pathfinding.h"
#include "fc_api.h"
#include "fc_spawn_internal.h"
#include <stddef.h>

/*
 * fc_npc.c — NPC framework with stat table and type-specific AI dispatch.
 *
 * PR 5: All 8 NPC types have full AI.
 *
 * NPC AI per tick (generic):
 *   1. If dead or inactive, skip.
 *   2. Decrement attack timer.
 *   3. Type-specific behavior (Jad style selection, Yt-MejKot heal, Yt-HurKot heal).
 *   4. If not in attack range, move toward player (greedy step).
 *   5. If in range and attack timer ready, roll attack and queue pending hit.
 *
 * Type-specific:
 *   Tz-Kih:     Melee + prayer drain on hit.
 *   Tz-Kek:     Melee. Splits into 2 small Tz-Kek on death.
 *   Tz-Kek-Sm:  Melee. (no special)
 *   Tok-Xil:    Ranged with projectile delay.
 *   Yt-MejKot:  Chooses one melee-cycle action: attack or heal a weak NPC.
 *   Ket-Zek:    Magic with projectile delay.
 *   TzTok-Jad:  Magic/ranged at distance; melee/magic/ranged at range 1.
 *   Yt-HurKot:  Heals Jad in range; attacks the player after being tagged.
 */

/* ======================================================================== */
/* NPC stat table                                                            */
/* ======================================================================== */

/*
 * Fight Caves stats use the reviewed OSRS parity table. Sizes and non-combat
 * behavior fields retain their existing cache/config-derived values.
 */
static const FcNpcStats NPC_STATS[NPC_TYPE_COUNT] = {
    [NPC_NONE] = {0},

    /* NPC_TZ_KIH: Lv 22 melee bat. Drains damage + 1 Prayer point.
     * Void 634: HP 100, Att 20, Str 30, Def 15, size 1, stab max 40 */
    [NPC_TZ_KIH] = {
        .max_hp = 100, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 40,
        .att_level = 20, .ranged_level = 30, .magic_level = 15,
        .def_level = 15, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_STAB,
        .size = 1, .movement_speed = 1, .prayer_drain = 10,
    },

    /* NPC_TZ_KEK: Lv 45 melee blob. Splits into 2 small on death.
     * Void 634: HP 200, Att 40, Str 60, Def 30, size 2, crush max 70 */
    [NPC_TZ_KEK] = {
        .max_hp = 200, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 70,
        .att_level = 40, .ranged_level = 60, .magic_level = 30,
        .def_level = 30, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 2, .movement_speed = 1,
    },

    /* NPC_TZ_KEK_SM: Lv 22 small blob (from split).
     * Void 634: HP 100, Att 20, Str 30, Def 15, size 1, crush max 40 */
    [NPC_TZ_KEK_SM] = {
        .max_hp = 100, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 40,
        .att_level = 20, .ranged_level = 30, .magic_level = 15,
        .def_level = 15, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 1, .movement_speed = 1,
    },

    /* NPC_TOK_XIL: Lv 90 ranged + melee (DUAL MODE).
     * Void 634: HP 400, Att 80, Str 120, Def 60, Rng 120, size 3
     * Current Fight Caves maxima are 130 for both melee and Ranged. */
    [NPC_TOK_XIL] = {
        .max_hp = 400, .attack_style = ATTACK_RANGED,
        .attack_speed = 4, .attack_range = 14,
        .melee_max_hit_tenths = 130, .ranged_max_hit_tenths = 130,
        .att_level = 80, .ranged_level = 120, .magic_level = 60,
        .def_level = 60, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 3, .movement_speed = 1,
    },

    /* NPC_YT_MEJKOT: Lv 180 melee + heals self/nearby NPCs with HP < 50% max.
     * Void 634: HP 800, Att 160, Str 240, Def 120, size 4
     * combat.toml: crush max 250. Heals 100 tenths (10 HP) as its attack. */
    [NPC_YT_MEJKOT] = {
        .max_hp = 800, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 250,
        .att_level = 160, .ranged_level = 240, .magic_level = 120,
        .def_level = 120, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 4, .movement_speed = 1, .heal_amount = 100,
    },

    /* NPC_KET_ZEK: Lv 360 magic + melee (DUAL MODE).
     * Void 634: HP 1600, Att 320, Str 480, Def 240, Mag 240, size 5
     * Current Fight Caves maxima are 550 melee and 520 Magic. The Magic
     * attack has +60 accuracy; the melee attack has no equipment bonus. */
    [NPC_KET_ZEK] = {
        .max_hp = 1600, .attack_style = ATTACK_MAGIC,
        .attack_speed = 4, .attack_range = 14,
        .melee_max_hit_tenths = 550, .magic_max_hit_tenths = 520,
        .att_level = 320, .ranged_level = 480, .magic_level = 240,
        .magic_attack_bonus = 60,
        .def_level = 240, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_STAB,
        .size = 5, .movement_speed = 1,
    },

    /* NPC_TZTOK_JAD: Lv 702 magic + ranged + melee.
     * Void 634: HP 2500, Att 640, Str 960, Def 480, Mag 480, Rng 960, size 5
     * combat.toml: melee stab max 970 (range 1), magic max 950 (range 14), ranged max 970
     * attack speed 8 (double normal), range 14. The Magic attack has +60
     * accuracy; the melee and Ranged attacks have no equipment bonus. */
    [NPC_TZTOK_JAD] = {
        .max_hp = 2500, .attack_style = ATTACK_MAGIC,
        .attack_speed = 8, .attack_range = 14,
        .melee_max_hit_tenths = 970, .ranged_max_hit_tenths = 970,
        .magic_max_hit_tenths = 950,
        .att_level = 640, .ranged_level = 960, .magic_level = 480,
        .magic_attack_bonus = 60,
        .def_level = 480, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_STAB,
        .size = 5, .movement_speed = 1,
    },

    /* NPC_YT_HURKOT: Lv 108 Jad healer. Heals Jad 50 tenths (5 HP) every 4 ticks within 5 tiles.
     * Void 634: HP 600, Att 140, Str 100, Def 60, size 1
     * combat.toml: crush max 140 */
    [NPC_YT_HURKOT] = {
        .max_hp = 600, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 140,
        .att_level = 140, .ranged_level = 120, .magic_level = 120,
        .def_level = 60, .ranged_def_bonus = 100,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 1, .movement_speed = 1,
        .heal_amount = 50, .heal_interval = 4,
    },
};

int fc_npc_max_hit_tenths_for_style(const FcNpcStats* stats, int attack_style) {
    if (stats == NULL) return 0;
    switch (attack_style) {
        case ATTACK_MELEE: return stats->melee_max_hit_tenths;
        case ATTACK_RANGED: return stats->ranged_max_hit_tenths;
        case ATTACK_MAGIC: return stats->magic_max_hit_tenths;
        default: return 0;
    }
}

int fc_npc_max_hit_hp_for_style(const FcNpcStats* stats, int attack_style) {
    int max_hit_tenths = fc_npc_max_hit_tenths_for_style(stats, attack_style);
    if (max_hit_tenths < 0 || max_hit_tenths % 10 != 0) return 0;
    return max_hit_tenths / 10;
}

int fc_npc_stats_valid(const FcNpcStats* stats) {
    if (stats == NULL) return 0;
    const int maxima[] = {
        stats->melee_max_hit_tenths,
        stats->ranged_max_hit_tenths,
        stats->magic_max_hit_tenths,
    };
    for (int i = 0; i < 3; i++) {
        if (maxima[i] < 0 || maxima[i] % 10 != 0) return 0;
    }
    return 1;
}

const FcNpcStats* fc_npc_get_stats(int npc_type) {
    if (npc_type < 0 || npc_type >= NPC_TYPE_COUNT) return &NPC_STATS[0];
    return &NPC_STATS[npc_type];
}

static int npc_attack_level_for_style(const FcNpcStats* stats,
                                      int attack_style) {
    switch (attack_style) {
        case ATTACK_MELEE: return stats->att_level;
        case ATTACK_RANGED: return stats->ranged_level;
        case ATTACK_MAGIC: return stats->magic_level;
        default: return 0;
    }
}

static int npc_attack_bonus_for_style(const FcNpcStats* stats,
                                      int attack_style) {
    switch (attack_style) {
        case ATTACK_MELEE: return stats->melee_attack_bonus;
        case ATTACK_RANGED: return stats->ranged_attack_bonus;
        case ATTACK_MAGIC: return stats->magic_attack_bonus;
        default: return 0;
    }
}

static FcAttackType npc_attack_type_for_style(const FcNpcStats* stats,
                                              int attack_style) {
    switch (attack_style) {
        case ATTACK_MELEE: return (FcAttackType)stats->melee_attack_type;
        case ATTACK_RANGED: return FC_ATTACK_TYPE_RANGED;
        case ATTACK_MAGIC: return FC_ATTACK_TYPE_MAGIC;
        default: return FC_ATTACK_TYPE_NONE;
    }
}

/* ======================================================================== */
/* Spawn                                                                     */
/* ======================================================================== */

void fc_npc_spawn(FcNpc* npc, int npc_type, int x, int y, int spawn_index) {
    const FcNpcStats* stats = fc_npc_get_stats(npc_type);

    npc->active = 1;
    npc->npc_type = npc_type;
    npc->spawn_index = spawn_index;
    npc->x = x;
    npc->y = y;
    npc->size = stats->size;
    npc->current_hp = stats->max_hp;
    npc->max_hp = stats->max_hp;
    npc->is_dead = 0;
    npc->attack_style = stats->attack_style;
    npc->attack_timer = stats->attack_speed;  /* first attack after full cooldown */
    npc->attack_speed = stats->attack_speed;
    npc->attack_range = stats->attack_range;
    npc->movement_speed = stats->movement_speed;
    npc->heal_timer = stats->heal_interval;  /* start at full cooldown */
    npc->heal_amount = stats->heal_amount;
    npc->healer_distracted = 0;
    npc->heal_target_idx = -1;
    npc->is_respawned_jad_healer = 0;
    npc->damage_taken_this_tick = 0;
    npc->prayer_drain_dealt_this_tick = 0;
    npc->healing_received_this_tick = 0;
    npc->healing_given_this_tick = 0;
    npc->healed_by_mejkot_this_tick = 0;
    npc->healed_by_hurkot_this_tick = 0;
    npc->healed_self_this_tick = 0;
    npc->died_this_tick = 0;
    npc->num_pending_hits = 0;
}

static void build_npc_movement_occupancy(
    const FcState* state,
    int npc_idx,
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    fc_build_occupancy(state, occupied, npc_idx, 0);
}

static int npc_dynamic_step_toward(FcState* state, int npc_idx,
                                   int target_x, int target_y) {
    FcNpc* npc = &state->npcs[npc_idx];
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    build_npc_movement_occupancy(state, npc_idx, occupied);
    return fc_npc_step_toward_sized_dynamic(&npc->x, &npc->y,
                                            target_x, target_y, npc->size,
                                            state->walkable,
                                            state->movement_flags,
                                            occupied);
}

static int min_i(int a, int b) {
    return a < b ? a : b;
}

static int max_i(int a, int b) {
    return a > b ? a : b;
}

static void npc_naive_player_chase_destination(const FcNpc* npc,
                                               const FcPlayer* player,
                                               int* out_x, int* out_y) {
    int source_width = npc->size;
    int source_length = npc->size;
    int target_width = 1;
    int target_length = 1;
    int diagonal = (npc->x - player->x) + (npc->y - player->y);
    int anti = (npc->x - player->x) - (npc->y - player->y);
    int south_west_clockwise = anti < 0;
    int north_west_clockwise =
        diagonal >= (target_length - 1) - (source_width - 1);
    int north_east_clockwise = anti > source_width - source_length;
    int south_east_clockwise =
        diagonal <= (target_width - 1) - (source_length - 1);

    if (south_west_clockwise && !north_west_clockwise) {
        int off_y;
        if (diagonal >= -source_width) {
            off_y = min_i(diagonal + source_width, target_length - 1);
        } else if (anti > -source_width) {
            off_y = -(source_width + anti);
        } else {
            off_y = 0;
        }
        *out_x = player->x - source_width;
        *out_y = player->y + off_y;
    } else if (north_west_clockwise && !north_east_clockwise) {
        int off_x;
        if (anti >= -target_length) {
            off_x = min_i(anti + target_length, target_width - 1);
        } else if (diagonal < target_length) {
            off_x = max_i(diagonal - target_length, -(source_width - 1));
        } else {
            off_x = 0;
        }
        *out_x = player->x + off_x;
        *out_y = player->y + target_length;
    } else if (north_east_clockwise && !south_east_clockwise) {
        int off_y;
        if (anti <= target_width) {
            off_y = target_length - anti;
        } else if (diagonal < target_width) {
            off_y = max_i(diagonal - target_width, -(source_length - 1));
        } else {
            off_y = 0;
        }
        *out_x = player->x + target_width;
        *out_y = player->y + off_y;
    } else {
        int off_x;
        if (diagonal > -source_length) {
            off_x = min_i(diagonal + source_length, target_width - 1);
        } else if (anti < source_length) {
            off_x = max_i(anti - source_length, -(source_length - 1));
        } else {
            off_x = 0;
        }
        *out_x = player->x + off_x;
        *out_y = player->y - source_length;
    }
}

int fc_npc_position_can_attack_player(const FcState* state, const FcNpc* npc,
                                      int candidate_x, int candidate_y) {
    if (!state || !npc || !npc->active || npc->is_dead) return 0;

    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    const FcPlayer* p = &state->player;

    int can_melee = fc_npc_can_melee_player(p->x, p->y,
                                            candidate_x, candidate_y,
                                            npc->size, state->walkable,
                                            state->movement_flags);
    if (can_melee &&
        (stats->melee_max_hit_tenths > 0 ||
         npc->attack_style == ATTACK_MELEE)) {
        return 1;
    }

    int distance = fc_distance_between_areas(
        p->x, p->y, 1, candidate_x, candidate_y, npc->size);
    if (npc->attack_style != ATTACK_MELEE && distance > 0 &&
        distance <= npc->attack_range) {
        return fc_has_los_between_areas(
            candidate_x, candidate_y, npc->size,
            p->x, p->y, 1, state->los_flags);
    }

    return 0;
}

static int npc_dynamic_step_toward_player_bounds(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    int target_x;
    int target_y;

    npc_naive_player_chase_destination(npc, &state->player,
                                       &target_x, &target_y);
    if (target_x == npc->x && target_y == npc->y) return 0;

    return npc_dynamic_step_toward(state, npc_idx, target_x, target_y);
}

/* ======================================================================== */
/* Tz-Kek: split on death — spawn 2 small Tz-Kek                            */
/* ======================================================================== */

void fc_npc_tz_kek_split(FcState* state, int dead_x, int dead_y) {
    /* Spawn 2 NPC_TZ_KEK_SM at/near the death position.
     * Do NOT increment npcs_remaining — the parent Tz-Kek was pre-counted as 2
     * at wave spawn time. These children inherit that count and decrement normally
     * when they die. (Matches RSPS: parent not in npcDespawn list, children are.) */
    const FcNpcStats* child_stats = fc_npc_get_stats(NPC_TZ_KEK_SM);
    for (int spawned = 0; spawned < 2; spawned++) {
        int sx = dead_x + (spawned == 0 ? 0 : 1);
        int sy = dead_y;
        /* Clamp to arena */
        if (sx >= FC_ARENA_WIDTH - 1) sx = dead_x - 1;
        if (sx < 1) sx = 1;

        if (!fc_spawn_find_available_footprint(
                state, sx, sy, child_stats->size, FC_ARENA_WIDTH - 1,
                &sx, &sy)) {
            break;
        }
        if (fc_spawn_npc_first_free(state, NPC_TZ_KEK_SM, sx, sy) < 0) {
            break;
        }
        /* No npcs_remaining++ — already pre-counted */
    }
}

/* ======================================================================== */
/* Jad direct attack selection                                               */
/* ======================================================================== */

static void record_npc_attack(FcState* state, const FcNpc* npc, int npc_idx,
                              int attack_style, int hit_delay_ticks,
                              int prayer_lock_tick, int hit_queued) {
    FcRenderEvents* events = &state->render_events;
    if (events->npc_attack_count >= FC_MAX_RENDER_NPC_ATTACKS) return;

    FcRenderNpcAttack* attack =
        &events->npc_attacks[events->npc_attack_count++];
    attack->npc_slot = npc_idx;
    attack->npc_type = npc->npc_type;
    attack->attack_style = attack_style;
    attack->source_x = npc->x;
    attack->source_y = npc->y;
    attack->source_size = npc->size;
    attack->target_x = state->player.x;
    attack->target_y = state->player.y;
    attack->hit_delay_ticks = hit_delay_ticks;
    attack->prayer_lock_tick = prayer_lock_tick;
    attack->hit_queued = hit_queued;
}

static void launch_npc_attack(FcState* state, FcNpc* npc, int npc_idx,
                              int attack_style, int hit_delay_ticks,
                              int prayer_drain, int prayer_lock_delay_ticks) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    FcPlayer* player = &state->player;
    int max_hit_hp = fc_npc_max_hit_hp_for_style(stats, attack_style);
    int attack_level = npc_attack_level_for_style(stats, attack_style);
    int attack_bonus = npc_attack_bonus_for_style(stats, attack_style);
    int attack_roll = fc_npc_attack_roll(attack_level, attack_bonus);
    FcAttackType attack_type =
        npc_attack_type_for_style(stats, attack_style);
    int defence_roll = fc_player_def_roll(player, attack_type);
    float hit_chance = fc_hit_chance(attack_roll, defence_roll);
    int hit = fc_rng_float(state) < hit_chance;
    int damage = hit ? fc_roll_npc_damage_tenths(state, max_hit_hp) : 0;

    int hit_queued = fc_queue_pending_hit(
        player->pending_hits, &player->num_pending_hits, FC_MAX_PENDING_HITS,
        damage, hit_delay_ticks, attack_style, npc_idx, prayer_drain);
    int prayer_lock_tick = -1;
    if (hit_queued) {
        FcPendingHit* queued =
            &player->pending_hits[player->num_pending_hits - 1];
        if (prayer_lock_delay_ticks > 0) {
            queued->prayer_snapshot = -1;
            queued->prayer_lock_tick =
                state->tick + prayer_lock_delay_ticks;
            prayer_lock_tick = queued->prayer_lock_tick;
        } else {
            queued->prayer_snapshot = player->prayer_at_tick_start;
        }
    }

    record_npc_attack(state, npc, npc_idx, attack_style, hit_delay_ticks,
                      prayer_lock_tick, hit_queued);
    npc->attack_timer = npc->attack_speed;
}

static void jad_attack(FcState* state, FcNpc* npc, int npc_idx) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    FcPlayer* p = &state->player;
    int dist = fc_distance_to_npc(p->x, p->y, npc);
    int can_melee = fc_npc_can_melee_player(p->x, p->y, npc->x, npc->y,
                                            npc->size, state->walkable,
                                            state->movement_flags);

    if (npc->attack_timer > 0) return;

    int use_style = ATTACK_NONE;
    int in_range = 0;

    int can_use_distance_styles =
        dist > 0 && dist <= npc->attack_range &&
        fc_has_los_between_areas(
            npc->x, npc->y, npc->size,
            p->x, p->y, 1, state->los_flags);

    if (can_melee && stats->melee_max_hit_tenths > 0) {
        /* In melee range Jad can still choose Magic or Ranged. All three
         * configured attacks have equal selection weight. */
        int choice = can_use_distance_styles ? fc_rng_int(state, 3) : 0;
        if (choice == 0) {
            use_style = ATTACK_MELEE;
        } else if (choice == 1) {
            use_style = ATTACK_MAGIC;
        } else {
            use_style = ATTACK_RANGED;
        }
        in_range = 1;
    } else if (can_use_distance_styles) {
        use_style = (fc_rng_int(state, 2) == 0) ? ATTACK_MAGIC : ATTACK_RANGED;
        in_range = 1;
    }

    if (!in_range) return;

    int delay = fc_npc_hit_delay(npc->npc_type, use_style, dist);
    if (use_style != ATTACK_MELEE && delay < 3) delay = 3;
    int prayer_lock_delay = use_style == ATTACK_MELEE ? 0 : 2;
    launch_npc_attack(state, npc, npc_idx, use_style, delay, 0,
                      prayer_lock_delay);
}

/* ======================================================================== */
/* Yt-MejKot: heal nearby NPCs                                              */
/* ======================================================================== */

static int apply_npc_heal(FcState* state, FcNpc* source, FcNpc* target,
                          int amount) {
    int before = target->current_hp;
    target->current_hp += amount;
    if (target->current_hp > target->max_hp) {
        target->current_hp = target->max_hp;
    }
    amount = target->current_hp - before;
    if (amount <= 0) return 0;
    source->healing_given_this_tick += amount;
    target->healing_received_this_tick += amount;
    if (source->npc_type == NPC_YT_MEJKOT) {
        target->healed_by_mejkot_this_tick = 1;
    } else if (source->npc_type == NPC_YT_HURKOT) {
        target->healed_by_hurkot_this_tick = 1;
    }
    if (source == target) target->healed_self_this_tick = 1;

    state->npc_heal_procs_this_tick++;
    state->npc_heal_amount_this_tick += amount;
    if (source->npc_type == NPC_YT_MEJKOT) {
        state->mejkot_heal_amount_this_tick += amount;
    }
    if (target->npc_type == NPC_TZTOK_JAD) {
        state->jad_heal_amount_this_tick += amount;
    }
    return amount;
}

static int npc_anchor_distance(const FcNpc* a, const FcNpc* b) {
    return fc_distance_between_areas(a->x, a->y, 1, b->x, b->y, 1);
}

static FcNpc* yt_mejkot_heal_target(FcState* state, FcNpc* npc) {
    if (npc->current_hp < npc->max_hp / 2) return npc;

    FcNpc* best = NULL;
    int best_distance = FC_ARENA_WIDTH;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* target = &state->npcs[i];
        if (target == npc || !target->active || target->is_dead) continue;
        if (target->current_hp >= target->max_hp / 2) continue;

        int distance = npc_anchor_distance(npc, target);
        if (distance > 8) continue;
        if (!best || distance < best_distance ||
            (distance == best_distance && target->spawn_index < best->spawn_index)) {
            best = target;
            best_distance = distance;
        }
    }
    return best;
}

static int yt_mejkot_try_heal(FcState* state, FcNpc* npc) {
    if (npc->attack_timer > 0) return 0;
    if (!fc_npc_can_melee_player(state->player.x, state->player.y,
                                 npc->x, npc->y, npc->size,
                                 state->walkable, state->movement_flags)) {
        return 0;
    }

    FcNpc* target = yt_mejkot_heal_target(state, npc);
    if (!target) return 0;

    apply_npc_heal(state, npc, target, npc->heal_amount);
    npc->heal_target_idx = (int)(target - state->npcs);
    npc->attack_timer = npc->attack_speed;
    return 1;
}

/* ======================================================================== */
/* Yt-HurKot: heal Jad until permanently tagged onto the player               */
/* ======================================================================== */

#define FC_HURKOT_HEAL_RANGE 5
static void npc_generic_attack(FcState* state, FcNpc* npc, int npc_idx);

static int find_active_jad(const FcState* state) {
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* npc = &state->npcs[i];
        if (npc->active && !npc->is_dead && npc->npc_type == NPC_TZTOK_JAD) {
            return i;
        }
    }
    return -1;
}

static void yt_hurkot_heal_cycle(FcState* state, FcNpc* npc, FcNpc* jad) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    if (npc->heal_timer > 1) {
        npc->heal_timer--;
        return;
    }

    npc->heal_timer = stats->heal_interval;
    if (npc_anchor_distance(npc, jad) > FC_HURKOT_HEAL_RANGE ||
        jad->current_hp >= jad->max_hp) {
        return;
    }

    if (apply_npc_heal(state, npc, jad, npc->heal_amount) > 0) {
        state->jad_heal_procs_this_tick++;
    }
}

static void yt_hurkot_tick(FcState* state, FcNpc* npc, int npc_idx) {
    /* Once tagged, a healer permanently pursues the player using the same
     * local, non-routing movement as every other NPC. It no longer heals Jad. */
    if (npc->healer_distracted) {
        npc->heal_target_idx = -1;
        if (!fc_npc_position_can_attack_player(state, npc, npc->x, npc->y)) {
            for (int step = 0; step < npc->movement_speed; step++) {
                if (!npc_dynamic_step_toward_player_bounds(state, npc_idx)) break;
            }
        }
        npc_generic_attack(state, npc, npc_idx);
        return;
    }

    int jad_idx = find_active_jad(state);
    FcNpc* jad = jad_idx >= 0 ? &state->npcs[jad_idx] : NULL;
    npc->heal_target_idx = jad_idx;
    if (jad) yt_hurkot_heal_cycle(state, npc, jad);

    if (jad && npc_anchor_distance(npc, jad) > FC_HURKOT_HEAL_RANGE) {
        npc_dynamic_step_toward(state, npc_idx, jad->x, jad->y);
    }
}

/* ======================================================================== */
/* Generic NPC attack (melee/ranged/magic, non-Jad)                          */
/* ======================================================================== */

/*
 * Tok-Xil switches to its weaker melee attack at contact. Ket-Zek keeps both
 * its Magic and Melee attacks valid at contact and samples between them.
 */
static void npc_generic_attack(FcState* state, FcNpc* npc, int npc_idx) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    FcPlayer* p = &state->player;
    int dist = fc_distance_to_npc(p->x, p->y, npc);
    int can_melee = fc_npc_can_melee_player(p->x, p->y, npc->x, npc->y,
                                            npc->size, state->walkable,
                                            state->movement_flags);

    if (npc->attack_timer > 0) return;

    /* Determine attack style and max hit based on distance */
    int use_style = npc->attack_style;  /* primary style */
    int in_range = 0;
    int primary_in_range =
        npc->attack_style != ATTACK_MELEE &&
        dist > 0 && dist <= npc->attack_range &&
        fc_has_los_between_areas(
            npc->x, npc->y, npc->size,
            p->x, p->y, 1, state->los_flags);

    if (can_melee && stats->melee_max_hit_tenths > 0) {
        if (npc->npc_type == NPC_KET_ZEK && primary_in_range &&
            fc_rng_int(state, 2) == 0) {
            use_style = npc->attack_style;
        } else {
            use_style = ATTACK_MELEE;
        }
        in_range = 1;
    } else if (can_melee && npc->attack_style == ATTACK_MELEE) {
        /* Pure melee NPC, in range */
        in_range = 1;
    } else if (primary_in_range) {
        in_range = 1;
    }

    if (!in_range) return;

    int delay = fc_npc_hit_delay(npc->npc_type, use_style, dist);
    launch_npc_attack(state, npc, npc_idx, use_style, delay,
                      stats->prayer_drain, 0);
}

static int npc_has_attack_position(FcState* state, FcNpc* npc) {
    return fc_npc_position_can_attack_player(state, npc, npc->x, npc->y);
}

/* ======================================================================== */
/* NPC AI tick — type dispatch                                               */
/* ======================================================================== */

void fc_npc_tick(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    if (!npc->active || npc->is_dead) return;

    if (npc->npc_type != NPC_YT_HURKOT) npc->heal_target_idx = -1;

    /* Decrement attack timer */
    if (npc->attack_timer > 0) npc->attack_timer--;

    /* --- Type-specific pre-attack behavior --- */

    /* Yt-HurKot either heals/follows Jad or permanently targets the player. */
    if (npc->npc_type == NPC_YT_HURKOT) {
        yt_hurkot_tick(state, npc, npc_idx);
        return;
    }

    /* Jad: move into range, then choose its attack style when the hit is queued */
    if (npc->npc_type == NPC_TZTOK_JAD) {
        if (!npc_has_attack_position(state, npc)) {
            for (int step = 0; step < npc->movement_speed; step++) {
                if (!npc_dynamic_step_toward_player_bounds(state, npc_idx)) break;
            }
        }
        jad_attack(state, npc, npc_idx);
        return;
    }

    /* --- Generic movement + attack for all other types --- */

    /* Movement: keep walking until this tile can actually attack. */
    if (!npc_has_attack_position(state, npc)) {
        for (int step = 0; step < npc->movement_speed; step++) {
            if (!npc_dynamic_step_toward_player_bounds(state, npc_idx)) break;
        }
    }

    /* A MejKot heal is an attack-cycle choice, not a parallel action. */
    if (npc->npc_type == NPC_YT_MEJKOT && yt_mejkot_try_heal(state, npc)) {
        return;
    }

    npc_generic_attack(state, npc, npc_idx);
}
