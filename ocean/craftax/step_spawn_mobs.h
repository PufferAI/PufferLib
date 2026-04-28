// Craftax spawn_mobs, optimized for CPU.
//
// Bitwise-equivalent to the prior JAX-transliterated baseline (verified by
// ocean/craftax_exp/parity_vs_baseline.c over 1.28M paired steps), ~6-9x
// faster per step by stripping JAX-isms:
//   - full-grid validity masks -> compact coord list collected in one pass
//   - bounding-box scan (only cells within MOB_DESPAWN_DISTANCE)
//   - early return on mob-cap / probability-roll failure (no dead writes)
//   - merged count + first_empty loops
//
// The prior reference implementation is archived at
// ocean/craftax_exp/step_spawn_mobs_baseline.h.

#pragma once

#include "step_medium.h"

#define CRAFTAX_SPAWN_MAP_CELLS (CRAFTAX_MAP_SIZE * CRAFTAX_MAP_SIZE)
#define CRAFTAX_SPAWN_BBOX_MAX_CELLS 729  // (2*DESPAWN-1)^2 at 14 = 27*27
#define CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK ( \
    (1ULL << CRAFTAX_BLOCK_GRASS) \
    | (1ULL << CRAFTAX_BLOCK_PATH) \
    | (1ULL << CRAFTAX_BLOCK_FIRE_GRASS) \
    | (1ULL << CRAFTAX_BLOCK_ICE_GRASS))
#define CRAFTAX_SPAWN_GRAVE_BLOCK_MASK ( \
    (1ULL << CRAFTAX_BLOCK_GRAVE) \
    | (1ULL << CRAFTAX_BLOCK_GRAVE2) \
    | (1ULL << CRAFTAX_BLOCK_GRAVE3))
#define CRAFTAX_SPAWN_WATER_BLOCK_MASK (1ULL << CRAFTAX_BLOCK_WATER)

typedef struct { int8_t dr, dc0, dc1; } CraftaxSpawnOffsetSpan;

static CraftaxSpawnOffsetSpan craftax_spawn_passive_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static CraftaxSpawnOffsetSpan craftax_spawn_hostile_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static CraftaxSpawnOffsetSpan craftax_spawn_boss_spans[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
static int32_t craftax_spawn_passive_span_count = 0;
static int32_t craftax_spawn_hostile_span_count = 0;
static int32_t craftax_spawn_boss_span_count = 0;
static int32_t craftax_spawn_offsets_initialized = 0;

static inline void craftax_spawn_append_span(
    CraftaxSpawnOffsetSpan* spans,
    int32_t* count,
    int32_t dr,
    int32_t dc0,
    int32_t dc1
) {
    spans[*count] = (CraftaxSpawnOffsetSpan){
        (int8_t)dr, (int8_t)dc0, (int8_t)dc1
    };
    *count += 1;
}

static inline void craftax_spawn_build_spans_for_row(
    CraftaxSpawnOffsetSpan* spans,
    int32_t* count,
    int32_t dr,
    int32_t limit,
    int32_t min_exclusive,
    int32_t max_exclusive
) {
    bool active = false;
    int32_t start = 0;
    for (int32_t dc = -limit; dc <= limit; dc++) {
        int32_t distance2 = dr * dr + dc * dc;
        bool valid = distance2 > min_exclusive && distance2 < max_exclusive;
        if (valid && !active) {
            active = true;
            start = dc;
        } else if (!valid && active) {
            craftax_spawn_append_span(spans, count, dr, start, dc - 1);
            active = false;
        }
    }
    if (active) {
        craftax_spawn_append_span(spans, count, dr, start, limit);
    }
}

static inline void craftax_spawn_init_offsets_once(void) {
    if (__atomic_load_n(
            &craftax_spawn_offsets_initialized, __ATOMIC_ACQUIRE
    )) return;

    #pragma omp critical(craftax_spawn_offsets_init)
    {
        if (!__atomic_load_n(
                &craftax_spawn_offsets_initialized, __ATOMIC_RELAXED
        )) {
            int32_t passive_count = 0;
            int32_t hostile_count = 0;
            int32_t boss_count = 0;
            int32_t limit = CRAFTAX_MOB_DESPAWN_DISTANCE - 1;
            int32_t limit2 = CRAFTAX_MOB_DESPAWN_DISTANCE
                           * CRAFTAX_MOB_DESPAWN_DISTANCE;
            for (int32_t dr = -limit; dr <= limit; dr++) {
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_passive_spans,
                    &passive_count,
                    dr,
                    limit,
                    9,
                    limit2
                );
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_hostile_spans,
                    &hostile_count,
                    dr,
                    limit,
                    81,
                    limit2
                );
                craftax_spawn_build_spans_for_row(
                    craftax_spawn_boss_spans,
                    &boss_count,
                    dr,
                    limit,
                    -1,
                    37
                );
            }
            craftax_spawn_passive_span_count = passive_count;
            craftax_spawn_hostile_span_count = hostile_count;
            craftax_spawn_boss_span_count = boss_count;
            __atomic_store_n(
                &craftax_spawn_offsets_initialized, 1, __ATOMIC_RELEASE
            );
        }
    }
}

static inline bool craftax_spawn_block_matches(uint8_t block, uint64_t mask) {
    return ((mask >> block) & 1ULL) != 0;
}

static inline uint64_t craftax_spawn_row_bits_for_mask(
    const CraftaxState* state,
    int32_t level,
    int32_t row,
    uint64_t terrain_mask
) {
    if (terrain_mask == CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK) {
        return state->spawn_all_bits[level][row];
    }
    if (terrain_mask == CRAFTAX_SPAWN_GRAVE_BLOCK_MASK) {
        return state->spawn_grave_bits[level][row];
    }
    return state->spawn_water_bits[level][row];
}

static inline uint64_t craftax_spawn_col_mask(int32_t col0, int32_t col1) {
    uint64_t hi = (1ULL << (col1 + 1)) - 1ULL;
    uint64_t lo = col0 <= 0 ? 0ULL : ((1ULL << col0) - 1ULL);
    return hi & ~lo;
}

static inline CraftaxThreefryKey craftax_spawn_next_random_key(
    CraftaxThreefryKey* rng
) {
    CraftaxThreefryKey draw;
    craftax_threefry_split(*rng, rng, &draw);
    return draw;
}

static inline int32_t craftax_spawn_floor_mob_type(
    int32_t floor, int32_t mob_class
) {
    static const int32_t mapping[CRAFTAX_NUM_LEVELS][3] = {
        {0, 0, 0}, {2, 2, 2}, {1, 1, 1}, {2, 3, 3}, {2, 4, 4},
        {1, 5, 5}, {1, 6, 6}, {1, 7, 7}, {0, 0, 0},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t class_index = craftax_step_jax_index(mob_class, 3);
    return mapping[level][class_index];
}

static inline float craftax_spawn_floor_spawn_chance(
    int32_t floor, int32_t chance_index
) {
    static const float chances[CRAFTAX_NUM_LEVELS][4] = {
        {0.1f, 0.02f, 0.05f, 0.1f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
        {0.0f, 0.06f, 0.05f, 0.0f},
        {0.1f, 0.06f, 0.05f, 0.0f},
    };
    int32_t level = craftax_step_jax_index(floor, CRAFTAX_NUM_LEVELS);
    int32_t index = craftax_step_jax_index(chance_index, 4);
    return chances[level][index];
}

static inline float craftax_spawn_mob_type_health(
    int32_t mob_type, int32_t mob_class
) {
    static const float health[CRAFTAX_NUM_MOB_TYPES][4] = {
        {3.0f, 5.0f, 3.0f, 0.0f}, {4.0f, 7.0f, 5.0f, 0.0f},
        {6.0f, 9.0f, 6.0f, 0.0f}, {8.0f, 11.0f, 8.0f, 0.0f},
        {0.0f, 12.0f, 12.0f, 0.0f}, {0.0f, 20.0f, 4.0f, 0.0f},
        {0.0f, 20.0f, 14.0f, 0.0f}, {0.0f, 24.0f, 16.0f, 0.0f},
    };
    int32_t type_index = craftax_step_jax_index(mob_type, CRAFTAX_NUM_MOB_TYPES);
    int32_t class_index = craftax_step_jax_index(mob_class, 4);
    return health[type_index][class_index];
}

static inline bool craftax_spawn_is_all_valid_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_GRASS] = 1,
        [CRAFTAX_BLOCK_PATH] = 1,
        [CRAFTAX_BLOCK_FIRE_GRASS] = 1,
        [CRAFTAX_BLOCK_ICE_GRASS] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static inline bool craftax_spawn_is_grave_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_GRAVE] = 1,
        [CRAFTAX_BLOCK_GRAVE2] = 1,
        [CRAFTAX_BLOCK_GRAVE3] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static inline bool craftax_spawn_is_water_block(int32_t block) {
    static const uint8_t flags[CRAFTAX_NUM_BLOCK_TYPES] = {
        [CRAFTAX_BLOCK_WATER] = 1,
    };
    int32_t idx = craftax_step_jax_index(block, CRAFTAX_NUM_BLOCK_TYPES);
    return flags[idx] != 0;
}

static inline int32_t craftax_spawn_player_distance_squared(
    const CraftaxState* state, int32_t row, int32_t col
) {
    int32_t dr = row - state->player_position[0];
    int32_t dc = col - state->player_position[1];
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    return dr * dr + dc * dc;
}

static inline int32_t craftax_spawn_count_mobs3(
    const CraftaxMobs3* mobs, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 3; i++) count += (int32_t)mobs->mask[level][i];
    return count;
}

static inline int32_t craftax_spawn_count_mobs2(
    const CraftaxMobs2* mobs, int32_t level
) {
    int32_t count = 0;
    for (int32_t i = 0; i < 2; i++) count += (int32_t)mobs->mask[level][i];
    return count;
}

static inline int32_t craftax_spawn_first_empty_mobs3(
    const CraftaxMobs3* mobs, int32_t level
) {
    for (int32_t i = 0; i < 3; i++) if (!mobs->mask[level][i]) return i;
    return 0;
}

static inline int32_t craftax_spawn_first_empty_mobs2(
    const CraftaxMobs2* mobs, int32_t level
) {
    for (int32_t i = 0; i < 2; i++) if (!mobs->mask[level][i]) return i;
    return 0;
}

static inline void craftax_spawn_mobs3_count_and_empty(
    const CraftaxMobs3* mobs, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 3; i++) {
        bool m = mobs->mask[level][i];
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

static inline void craftax_spawn_mobs2_count_and_empty(
    const CraftaxMobs2* mobs, int32_t level,
    int32_t* count_out, int32_t* first_empty_out
) {
    int32_t count = 0, first_empty = 0;
    bool found = false;
    for (int32_t i = 0; i < 2; i++) {
        bool m = mobs->mask[level][i];
        count += (int32_t)m;
        if (!m && !found) { first_empty = i; found = true; }
    }
    *count_out = count;
    *first_empty_out = first_empty;
}

// Baseline algorithm on a bool mask:
//   draw = valid_count * (1.0 - uniform_f32(key));
//   cum = 0;
//   for i: if valid[i] { cum += 1.0; if (cum >= draw) return i; }
// Over a compact list of length valid_count this collapses to a short loop
// using the same FP arithmetic, preserving bitwise-identical choice.
static inline int32_t craftax_spawn_pick_kth(
    int32_t valid_count, CraftaxThreefryKey key
) {
    float draw = (float)valid_count * (1.0f - craftax_threefry_uniform_f32(key));
    float cum = 0.0f;
    for (int32_t k = 0; k < valid_count; k++) {
        cum += 1.0f;
        if (cum >= draw) return k;
    }
    return valid_count - 1;
}

typedef struct { int16_t row, col; } CraftaxSpawnCoord;

typedef struct {
    CraftaxSpawnCoord passive[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    CraftaxSpawnCoord melee[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    CraftaxSpawnCoord ranged[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t passive_count;
    int32_t melee_count;
    int32_t ranged_count;
} CraftaxSpawnLists;

static inline int32_t craftax_spawn_collect_spans(
    const CraftaxState* state,
    int32_t level,
    const CraftaxSpawnOffsetSpan* spans,
    int32_t span_count,
    uint64_t terrain_mask,
    CraftaxSpawnCoord* coords
) {
    int32_t pr = state->player_position[0];
    int32_t pc = state->player_position[1];
    int32_t n = 0;
    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t candidates =
            craftax_spawn_row_bits_for_mask(state, level, row, terrain_mask)
            & ~state->mob_bits[level][row]
            & craftax_spawn_col_mask(col0, col1);
        while (candidates != 0) {
            int32_t col = __builtin_ctzll(candidates);
            coords[n].row = (int16_t)row;
            coords[n].col = (int16_t)col;
            n++;
            candidates &= candidates - 1;
        }
    }
    return n;
}

static inline bool craftax_spawn_scan_spans(
    const CraftaxState* state,
    int32_t level,
    const CraftaxSpawnOffsetSpan* spans,
    int32_t span_count,
    uint64_t terrain_mask,
    CraftaxThreefryKey pos_key,
    int32_t* out_row,
    int32_t* out_col
) {
    CraftaxSpawnCoord coords[CRAFTAX_SPAWN_BBOX_MAX_CELLS];
    int32_t n = craftax_spawn_collect_spans(
        state, level, spans, span_count, terrain_mask, coords
    );
    if (n == 0) return false;
    int32_t k = craftax_spawn_pick_kth(n, pos_key);
    *out_row = coords[k].row;
    *out_col = coords[k].col;
    return true;
}

static inline bool craftax_spawn_coord_matches(
    CraftaxSpawnCoord coord, bool exclude, int32_t row, int32_t col
) {
    return exclude && coord.row == row && coord.col == col;
}

static inline bool craftax_spawn_pick_excluding(
    const CraftaxSpawnCoord* coords, int32_t count, CraftaxThreefryKey key,
    bool exclude_a, int32_t row_a, int32_t col_a,
    bool exclude_b, int32_t row_b, int32_t col_b,
    int32_t* out_row, int32_t* out_col
) {
    int32_t valid_count = 0;
    for (int32_t i = 0; i < count; i++) {
        bool excluded = craftax_spawn_coord_matches(
            coords[i], exclude_a, row_a, col_a
        ) || craftax_spawn_coord_matches(coords[i], exclude_b, row_b, col_b);
        valid_count += excluded ? 0 : 1;
    }
    if (valid_count == 0) return false;

    int32_t k = craftax_spawn_pick_kth(valid_count, key);
    for (int32_t i = 0; i < count; i++) {
        bool excluded = craftax_spawn_coord_matches(
            coords[i], exclude_a, row_a, col_a
        ) || craftax_spawn_coord_matches(coords[i], exclude_b, row_b, col_b);
        if (excluded) continue;
        if (k == 0) {
            *out_row = coords[i].row;
            *out_col = coords[i].col;
            return true;
        }
        k--;
    }
    return false;
}

static inline void craftax_spawn_scan_all(
    const CraftaxState* state,
    int32_t level,
    int32_t ranged_type,
    bool fighting_boss,
    bool need_passive,
    bool need_melee,
    bool need_ranged,
    CraftaxSpawnLists* out
) {
    out->passive_count = 0;
    out->melee_count = 0;
    out->ranged_count = 0;

    craftax_spawn_init_offsets_once();

    if (need_passive) {
        out->passive_count = craftax_spawn_collect_spans(
            state,
            level,
            craftax_spawn_passive_spans,
            craftax_spawn_passive_span_count,
            CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK,
            out->passive
        );
    }

    if (!need_melee && !need_ranged) return;

    int32_t pr = state->player_position[0];
    int32_t pc = state->player_position[1];
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    bool ranged_water_type = (ranged_type == 5);

    uint64_t melee_terrain_mask = fighting_boss
        ? CRAFTAX_SPAWN_GRAVE_BLOCK_MASK
        : CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    uint64_t ranged_terrain_mask;
    if (fighting_boss) {
        ranged_terrain_mask = CRAFTAX_SPAWN_GRAVE_BLOCK_MASK;
    } else if (ranged_water_type) {
        ranged_terrain_mask = CRAFTAX_SPAWN_WATER_BLOCK_MASK;
    } else {
        ranged_terrain_mask = CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    }

    for (int32_t i = 0; i < span_count; i++) {
        int32_t row = pr + spans[i].dr;
        if ((uint32_t)row >= CRAFTAX_MAP_SIZE) continue;
        int32_t col0 = pc + spans[i].dc0;
        int32_t col1 = pc + spans[i].dc1;
        if (col0 < 0) col0 = 0;
        if (col1 >= CRAFTAX_MAP_SIZE) col1 = CRAFTAX_MAP_SIZE - 1;
        if (col0 > col1) continue;
        uint64_t open_bits =
            ~state->mob_bits[level][row] & craftax_spawn_col_mask(col0, col1);

        if (need_melee) {
            uint64_t melee_candidates =
                craftax_spawn_row_bits_for_mask(
                    state, level, row, melee_terrain_mask
                ) & open_bits;
            while (melee_candidates != 0) {
                int32_t col = __builtin_ctzll(melee_candidates);
                int32_t n = out->melee_count++;
                out->melee[n].row = (int16_t)row;
                out->melee[n].col = (int16_t)col;
                melee_candidates &= melee_candidates - 1;
            }
        }

        if (need_ranged) {
            uint64_t ranged_candidates =
                craftax_spawn_row_bits_for_mask(
                    state, level, row, ranged_terrain_mask
                ) & open_bits;
            while (ranged_candidates != 0) {
                int32_t col = __builtin_ctzll(ranged_candidates);
                int32_t n = out->ranged_count++;
                out->ranged[n].row = (int16_t)row;
                out->ranged[n].col = (int16_t)col;
                ranged_candidates &= ranged_candidates - 1;
            }
        }
    }
}

static inline bool craftax_spawn_scan_passive(
    const CraftaxState* state, int32_t level, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    return craftax_spawn_scan_spans(
        state,
        level,
        craftax_spawn_passive_spans,
        craftax_spawn_passive_span_count,
        CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK,
        pos_key,
        out_row,
        out_col
    );
}

static inline bool craftax_spawn_scan_melee(
    const CraftaxState* state, int32_t level, bool fighting_boss,
    CraftaxThreefryKey pos_key, int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    uint64_t terrain_mask = fighting_boss
        ? CRAFTAX_SPAWN_GRAVE_BLOCK_MASK
        : CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    return craftax_spawn_scan_spans(
        state, level, spans, span_count, terrain_mask, pos_key,
        out_row, out_col
    );
}

static inline bool craftax_spawn_scan_ranged(
    const CraftaxState* state, int32_t level, int32_t new_type,
    bool fighting_boss, CraftaxThreefryKey pos_key,
    int32_t* out_row, int32_t* out_col
) {
    craftax_spawn_init_offsets_once();
    const CraftaxSpawnOffsetSpan* spans = fighting_boss
        ? craftax_spawn_boss_spans
        : craftax_spawn_hostile_spans;
    int32_t span_count = fighting_boss
        ? craftax_spawn_boss_span_count
        : craftax_spawn_hostile_span_count;
    uint64_t terrain_mask;
    if (fighting_boss) {
        terrain_mask = CRAFTAX_SPAWN_GRAVE_BLOCK_MASK;
    } else if (new_type == 5) {
        terrain_mask = CRAFTAX_SPAWN_WATER_BLOCK_MASK;
    } else {
        terrain_mask = CRAFTAX_SPAWN_ALL_VALID_BLOCK_MASK;
    }
    return craftax_spawn_scan_spans(
        state, level, spans, span_count, terrain_mask, pos_key,
        out_row, out_col
    );
}

// Both RNG keys are always consumed (preserves baseline RNG sequence).
// Baseline quirk: type_id[level][slot] is written unconditionally, even
// when no mob spawns. We match that for bitwise parity.

static inline void craftax_spawn_passive_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(&state->passive_mobs, level, &count, &slot);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    int32_t type = craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_PASSIVE);
    state->passive_mobs.type_id[level][slot] = type;

    if (fighting_boss) return;
    if (count >= CRAFTAX_MAX_PASSIVE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 0)) return;

    int32_t row, col;
    if (!craftax_spawn_scan_passive(state, level, pos_key, &row, &col)) return;

    state->passive_mobs.position[level][slot][0] = row;
    state->passive_mobs.position[level][slot][1] = col;
    state->passive_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_PASSIVE);
    state->passive_mobs.mask[level][slot]        = true;
    state->mob_bits[level][row] |= (1ULL << col);
}

static inline void craftax_spawn_melee_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs3_count_and_empty(&state->melee_mobs, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_MELEE)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    float night_coeff = 1.0f - state->light_level;
    float spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    CraftaxThreefryKey pos_key = craftax_spawn_next_random_key(rng);

    state->melee_mobs.type_id[level][slot] = type;

    if (count >= CRAFTAX_MAX_MELEE_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= spawn_chance * (float)monster_spawn_coeff) return;

    int32_t row, col;
    if (!craftax_spawn_scan_melee(state, level, fighting_boss, pos_key, &row, &col))
        return;

    state->melee_mobs.position[level][slot][0] = row;
    state->melee_mobs.position[level][slot][1] = col;
    state->melee_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_MELEE);
    state->melee_mobs.mask[level][slot]        = true;
    state->mob_bits[level][row] |= (1ULL << col);
}

static inline void craftax_spawn_ranged_mob(
    CraftaxState* state, CraftaxThreefryKey* rng,
    int32_t level, bool fighting_boss, int32_t monster_spawn_coeff
) {
    int32_t count, slot;
    craftax_spawn_mobs2_count_and_empty(&state->ranged_mobs, level, &count, &slot);

    int32_t type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_RANGED)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);

    CraftaxThreefryKey prob_key = craftax_spawn_next_random_key(rng);
    CraftaxThreefryKey pos_key  = craftax_spawn_next_random_key(rng);

    state->ranged_mobs.type_id[level][slot] = type;

    if (count >= CRAFTAX_MAX_RANGED_MOBS) return;
    if (craftax_threefry_uniform_f32(prob_key)
        >= craftax_spawn_floor_spawn_chance(level, 2) * (float)monster_spawn_coeff)
        return;

    int32_t row, col;
    if (!craftax_spawn_scan_ranged(state, level, type, fighting_boss, pos_key,
                                    &row, &col)) return;

    state->ranged_mobs.position[level][slot][0] = row;
    state->ranged_mobs.position[level][slot][1] = col;
    state->ranged_mobs.health[level][slot]      =
        craftax_spawn_mob_type_health(type, CRAFTAX_MOB_RANGED);
    state->ranged_mobs.mask[level][slot]        = true;
    state->mob_bits[level][row] |= (1ULL << col);
}

static inline void craftax_spawn_mobs_native(
    CraftaxState* state, CraftaxThreefryKey rng
) {
    int32_t level = craftax_step_jax_index(
        state->player_level, CRAFTAX_NUM_LEVELS
    );
    bool fighting_boss = craftax_step_is_fighting_boss(state);
    int32_t monster_spawn_coeff =
        1
        + (int32_t)(state->monsters_killed[level]
                    < CRAFTAX_MONSTERS_KILLED_TO_CLEAR_LEVEL) * 2;

    bool boss_spawn_wave =
        fighting_boss && state->boss_timesteps_to_spawn_this_round >= 1;
    if (fighting_boss) {
        monster_spawn_coeff *= (int32_t)boss_spawn_wave * 1000;
    }

    int32_t passive_count, passive_slot;
    craftax_spawn_mobs3_count_and_empty(
        &state->passive_mobs, level, &passive_count, &passive_slot
    );
    CraftaxThreefryKey passive_prob_key = craftax_spawn_next_random_key(&rng);
    CraftaxThreefryKey passive_pos_key = craftax_spawn_next_random_key(&rng);
    int32_t passive_type = craftax_spawn_floor_mob_type(
        level, CRAFTAX_MOB_PASSIVE
    );
    state->passive_mobs.type_id[level][passive_slot] = passive_type;

    int32_t melee_count, melee_slot;
    craftax_spawn_mobs3_count_and_empty(
        &state->melee_mobs, level, &melee_count, &melee_slot
    );
    int32_t melee_type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_MELEE)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_MELEE);
    CraftaxThreefryKey melee_prob_key = craftax_spawn_next_random_key(&rng);
    float night_coeff = 1.0f - state->light_level;
    float melee_spawn_chance = craftax_spawn_floor_spawn_chance(level, 1)
        + craftax_spawn_floor_spawn_chance(level, 3) * night_coeff * night_coeff;
    CraftaxThreefryKey melee_pos_key = craftax_spawn_next_random_key(&rng);
    state->melee_mobs.type_id[level][melee_slot] = melee_type;

    int32_t ranged_count, ranged_slot;
    craftax_spawn_mobs2_count_and_empty(
        &state->ranged_mobs, level, &ranged_count, &ranged_slot
    );
    int32_t ranged_type = fighting_boss
        ? craftax_spawn_floor_mob_type(state->boss_progress, CRAFTAX_MOB_RANGED)
        : craftax_spawn_floor_mob_type(level, CRAFTAX_MOB_RANGED);
    CraftaxThreefryKey ranged_prob_key = craftax_spawn_next_random_key(&rng);
    CraftaxThreefryKey ranged_pos_key = craftax_spawn_next_random_key(&rng);
    state->ranged_mobs.type_id[level][ranged_slot] = ranged_type;

    bool try_passive = !fighting_boss
        && passive_count < CRAFTAX_MAX_PASSIVE_MOBS
        && craftax_threefry_uniform_f32(passive_prob_key)
            < craftax_spawn_floor_spawn_chance(level, 0);
    bool try_melee = melee_count < CRAFTAX_MAX_MELEE_MOBS
        && craftax_threefry_uniform_f32(melee_prob_key)
            < melee_spawn_chance * (float)monster_spawn_coeff;
    bool try_ranged = ranged_count < CRAFTAX_MAX_RANGED_MOBS
        && craftax_threefry_uniform_f32(ranged_prob_key)
            < craftax_spawn_floor_spawn_chance(level, 2)
                * (float)monster_spawn_coeff;

    if (!try_passive && !try_melee && !try_ranged) return;

    int32_t try_count = (int32_t)try_passive
        + (int32_t)try_melee
        + (int32_t)try_ranged;
    if (try_count == 1) {
        int32_t row, col;
        if (try_passive && craftax_spawn_scan_passive(
                state, level, passive_pos_key, &row, &col
        )) {
            state->passive_mobs.position[level][passive_slot][0] = row;
            state->passive_mobs.position[level][passive_slot][1] = col;
            state->passive_mobs.health[level][passive_slot] =
                craftax_spawn_mob_type_health(
                    passive_type, CRAFTAX_MOB_PASSIVE
                );
            state->passive_mobs.mask[level][passive_slot] = true;
            state->mob_bits[level][row] |= (1ULL << col);
        } else if (try_melee && craftax_spawn_scan_melee(
                state, level, fighting_boss, melee_pos_key, &row, &col
        )) {
            state->melee_mobs.position[level][melee_slot][0] = row;
            state->melee_mobs.position[level][melee_slot][1] = col;
            state->melee_mobs.health[level][melee_slot] =
                craftax_spawn_mob_type_health(melee_type, CRAFTAX_MOB_MELEE);
            state->melee_mobs.mask[level][melee_slot] = true;
            state->mob_bits[level][row] |= (1ULL << col);
        } else if (try_ranged && craftax_spawn_scan_ranged(
                state, level, ranged_type, fighting_boss, ranged_pos_key,
                &row, &col
        )) {
            state->ranged_mobs.position[level][ranged_slot][0] = row;
            state->ranged_mobs.position[level][ranged_slot][1] = col;
            state->ranged_mobs.health[level][ranged_slot] =
                craftax_spawn_mob_type_health(ranged_type, CRAFTAX_MOB_RANGED);
            state->ranged_mobs.mask[level][ranged_slot] = true;
            state->mob_bits[level][row] |= (1ULL << col);
        }
        return;
    }

    CraftaxSpawnLists lists;
    craftax_spawn_scan_all(
        state, level, ranged_type, fighting_boss,
        try_passive, try_melee, try_ranged, &lists
    );

    bool passive_spawned = false;
    int32_t passive_row = 0;
    int32_t passive_col = 0;
    if (try_passive && craftax_spawn_pick_excluding(
            lists.passive, lists.passive_count, passive_pos_key,
            false, 0, 0, false, 0, 0, &passive_row, &passive_col
        )) {
        state->passive_mobs.position[level][passive_slot][0] = passive_row;
        state->passive_mobs.position[level][passive_slot][1] = passive_col;
        state->passive_mobs.health[level][passive_slot] =
            craftax_spawn_mob_type_health(passive_type, CRAFTAX_MOB_PASSIVE);
        state->passive_mobs.mask[level][passive_slot] = true;
        state->mob_bits[level][passive_row] |= (1ULL << passive_col);
        passive_spawned = true;
    }

    bool melee_spawned = false;
    int32_t melee_row = 0;
    int32_t melee_col = 0;
    if (try_melee && craftax_spawn_pick_excluding(
            lists.melee, lists.melee_count, melee_pos_key,
            passive_spawned, passive_row, passive_col,
            false, 0, 0, &melee_row, &melee_col
        )) {
        state->melee_mobs.position[level][melee_slot][0] = melee_row;
        state->melee_mobs.position[level][melee_slot][1] = melee_col;
        state->melee_mobs.health[level][melee_slot] =
            craftax_spawn_mob_type_health(melee_type, CRAFTAX_MOB_MELEE);
        state->melee_mobs.mask[level][melee_slot] = true;
        state->mob_bits[level][melee_row] |= (1ULL << melee_col);
        melee_spawned = true;
    }

    int32_t ranged_row = 0;
    int32_t ranged_col = 0;
    if (try_ranged && craftax_spawn_pick_excluding(
            lists.ranged, lists.ranged_count, ranged_pos_key,
            passive_spawned, passive_row, passive_col,
            melee_spawned, melee_row, melee_col, &ranged_row, &ranged_col
        )) {
        state->ranged_mobs.position[level][ranged_slot][0] = ranged_row;
        state->ranged_mobs.position[level][ranged_slot][1] = ranged_col;
        state->ranged_mobs.health[level][ranged_slot] =
            craftax_spawn_mob_type_health(ranged_type, CRAFTAX_MOB_RANGED);
        state->ranged_mobs.mask[level][ranged_slot] = true;
        state->mob_bits[level][ranged_row] |= (1ULL << ranged_col);
    }
}
