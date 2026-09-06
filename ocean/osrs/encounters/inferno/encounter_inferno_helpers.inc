#line 862 "encounter_inferno.h"
static inline int inf_is_barrage_spell(int spell) {
    return spell == ENCOUNTER_SPELL_BLOOD || spell == ENCOUNTER_SPELL_ICE;
}

static inline int inf_spell_base_damage(int spell) {
    if (spell == ENCOUNTER_SPELL_ICE) return ICE_BARRAGE_MAX_HIT;
    if (spell == ENCOUNTER_SPELL_BLOOD) return BLOOD_BARRAGE_MAX_HIT;
    fprintf(stderr, "BUG: invalid Inferno spell %d\n", spell);
    abort();
}

static inline int inf_player_autocast_spell(const Player* p) {
    return p->autocast_spell == ENCOUNTER_SPELL_ICE
        ? ENCOUNTER_SPELL_ICE
        : ENCOUNTER_SPELL_BLOOD;
}

static inline int inf_player_magic_base_damage(const Player* p) {
    return inf_spell_base_damage(inf_player_autocast_spell(p));
}

static inline void inf_apply_spell_cast_stats(EncounterLoadoutStats* out) {
    out->attack_speed = COMBAT_SPELL_CAST_SPEED;
    out->attack_range = COMBAT_SPELL_ATTACK_RANGE;
}

static EncounterPendingHit* inf_queue_npc_pending_hit(
    InfernoState* s,
    int npc_idx,
    int ticks_remaining,
    int damage,
    AttackStyle attack_style,
    int spell_type,
    int hit_success
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) {
        fprintf(stderr, "inferno: invalid pending-hit npc slot %d\n", npc_idx);
        abort();
    }

    InfNPC* npc = &s->npcs[npc_idx];
    if (!npc->active) {
        fprintf(stderr,
            "inferno: pending-hit queued for inactive npc tick=%d slot=%d "
            "type=%d delay=%d style=%d spell=%d damage=%d\n",
            s->tick, npc_idx, npc->type, ticks_remaining, attack_style,
            spell_type, damage);
        abort();
    }

    EncounterPendingHit hit = {
        .ticks_remaining = (int8_t)ticks_remaining,
        .attack_style = (int8_t)attack_style,
        .check_prayer = 0,
        .spell_type = (int8_t)spell_type,
        .hit_success = (int8_t)hit_success,
        .elysian_reduced = 0,
        .damage = (int16_t)damage,
    };
    return encounter_pending_hit_queue_push(
        &npc->pending_hits, hit, "inferno npc", s->tick, npc_idx, npc->type);
}

static EncounterPendingHit* inf_queue_player_pending_hit(
    InfernoState* s,
    int npc_idx,
    int npc_type,
    EncounterPendingHit hit
) {
    hit.source_npc_slot = (int8_t)npc_idx;
    return encounter_pending_hit_queue_push(
        &s->player_pending_hits, hit, "inferno player", s->tick, npc_idx, npc_type);
}

static void inf_shuffle_spawns(InfernoState* s) {
    for (int i = 0; i < INF_NUM_SPAWN_POS; i++)
        s->spawn_order[i] = i;
    encounter_shuffle(s->spawn_order, INF_NUM_SPAWN_POS, &s->rng_state);
}
static const EncounterArenaTopology* inf_route_topology_for_state(
    const InfernoContext* ctx,
    const InfernoState* state
);


static int inf_pillar_footprint_blocked(
    const InfernoState* s,
    int x,
    int y,
    int size
) {
    for (int pillar_idx = 0; pillar_idx < INF_NUM_PILLARS; pillar_idx++) {
        const InfPillar* pillar = &s->pillars[pillar_idx];
        if (!pillar->active) continue;
        if (los_aabb_overlap(
                x, y, size,
                pillar->x, pillar->y, INF_PILLAR_SIZE))
            return 1;
    }
    return 0;
}
static uint8_t inf_active_pillar_mask(const InfernoState* state) {
    uint8_t mask = 0;
    for (int pillar = 0; pillar < INF_NUM_PILLARS; pillar++)
        mask |= (uint8_t)(state->pillars[pillar].active != 0) << pillar;
    return mask;
}
static int inf_pillars_at_canonical_positions(
    const InfernoState* state
) {
    for (int pillar = 0; pillar < INF_NUM_PILLARS; pillar++)
        if (state->pillars[pillar].x != INF_PILLAR_POS[pillar][0] ||
                state->pillars[pillar].y != INF_PILLAR_POS[pillar][1])
            return 0;
    return 1;
}



static uint32_t inf_pillar_los_flags(void* data, int x, int y) {
    const InfernoState* s = (const InfernoState*)data;
    return inf_pillar_footprint_blocked(s, x, y, 1)
        ? LOS_FULL_MASK
        : 0;
}

static OsrsLosQuery inf_player_los_query(const InfernoState* s) {
    return osrs_los_flags(inf_pillar_los_flags, (void*)s);
}

static int inf_footprint_blocked_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int x,
    int y,
    int size
) {
    const EncounterArenaTopology* topology =
        inf_route_topology_for_state(ctx, s);
    return encounter_arena_topology_footprint_blocked(
            topology, x, y, size) ||
        (!inf_pillars_at_canonical_positions(s) &&
         inf_pillar_footprint_blocked(s, x, y, size));
}

static int inf_los_clear_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int actor_x,
    int actor_y,
    int actor_size,
    int target_x,
    int target_y,
    int target_size,
    int attack_range
) {
    if (inf_pillars_at_canonical_positions(s))
        return encounter_arena_topology_los_clear(
            inf_route_topology_for_state(ctx, s),
            actor_x,
            actor_y,
            actor_size,
            target_x,
            target_y,
            target_size,
            attack_range);
    const EncounterArenaTopology* topology = ctx->route_topology;
    if (!encounter_arena_topology_footprint_in_bounds_assume_finalized_size_in_range(
            topology, actor_x, actor_y, actor_size) ||
            !encounter_arena_topology_footprint_in_bounds_assume_finalized_size_in_range(
                topology, target_x, target_y, target_size))
        return 0;
    return entity_has_line_of_sight_with_flags(
        inf_pillar_los_flags,
        (void*)s,
        actor_x,
        actor_y,
        actor_size,
        target_x,
        target_y,
        target_size,
        attack_range);
}

typedef struct {
    int x;
    int y;
    int size;
    int is_player;
} InfTargetArea;

static InfTargetArea inf_npc_current_target_area(const InfernoState* s, const InfNPC* npc) {
    if (npc->type == INF_NPC_NIBBLER) {
        int pillar_idx = s->nibbler_target_pillar;
        if (pillar_idx >= 0 && pillar_idx < INF_NUM_PILLARS &&
            s->pillars[pillar_idx].active) {
            return (InfTargetArea){
                .x = s->pillars[pillar_idx].x,
                .y = s->pillars[pillar_idx].y,
                .size = INF_PILLAR_SIZE,
                .is_player = 0,
            };
        }
        for (int i = 0; i < INF_NUM_PILLARS; i++) {
            if (s->pillars[i].active) {
                return (InfTargetArea){
                    .x = s->pillars[i].x,
                    .y = s->pillars[i].y,
                    .size = INF_PILLAR_SIZE,
                    .is_player = 0,
                };
            }
        }
    }

    if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
        s->npcs[npc->aggro_target].active) {
        const InfNPC* target = &s->npcs[npc->aggro_target];
        return (InfTargetArea){
            .x = target->x,
            .y = target->y,
            .size = target->size,
            .is_player = 0,
        };
    }

    return (InfTargetArea){
        .x = s->player.x,
        .y = s->player.y,
        .size = 1,
        .is_player = 1,
    };
}

static int inf_npc_has_los_to_area_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    const InfNPC* npc,
    int target_x,
    int target_y,
    int target_size
) {
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    return inf_los_clear_ctx(
        s,
        ctx,
        npc->x,
        npc->y,
        npc->size,
        target_x,
        target_y,
        target_size,
        stats->attack_range);
}


static int inf_npc_has_los_to_tile_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    const InfNPC* npc,
    int target_x,
    int target_y
) {
    return inf_npc_has_los_to_area_ctx(
        s, ctx, npc, target_x, target_y, 1);
}


static int inf_npc_has_los_direct_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    const InfNPC* npc = &s->npcs[npc_idx];
    InfTargetArea target = inf_npc_current_target_area(s, npc);
    return inf_npc_has_los_to_area_ctx(
        s, ctx, npc, target.x, target.y, target.size);
}


static void inf_reset_npc_player_los_frame(InfernoContext* ctx) {
    memset(ctx->npc_player_los_frame, -1, sizeof(ctx->npc_player_los_frame));
}

static void inf_sample_npc_player_los_frame(
    const InfernoState* s,
    InfernoContext* ctx,
    int npc_idx
) {
    ctx->npc_player_los_frame[npc_idx] =
        (int8_t)inf_npc_has_los_direct_ctx(s, ctx, npc_idx);
}

static int inf_npc_has_los_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    int8_t sampled = ctx->npc_player_los_frame[npc_idx];
    return sampled >= 0 ? sampled : inf_npc_has_los_direct_ctx(s, ctx, npc_idx);
}


static inline int inf_is_final_wave(const InfernoState* s) {
    return s->wave == INF_NUM_WAVES - 1;
}

static inline int inf_zuk_healer_tags_first_reward_gate_active(
    const InfernoState* s,
    const InfernoContext* ctx
) {
    return ctx->config.zuk_healer_reward_mode == INF_ZUK_HEALER_REWARD_MODE_TAGS_FIRST &&
        s->zuk.healer_spawned && s->total_zuk_healer_tags < 4;
}

static int inf_find_live_zuk_idx(const InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].active && s->npcs[i].type == INF_NPC_ZUK) return i;
    }
    return -1;
}

static int inf_is_live_zuk_healer_slot(const InfernoState* s, int npc_slot) {
    if (npc_slot < 0 || npc_slot >= INF_MAX_NPCS) return 0;
    const InfNPC* npc = &s->npcs[npc_slot];
    return npc->active &&
        npc->hp > 0 &&
        npc->death_ticks == 0 &&
        npc->type == INF_NPC_HEALER_ZUK;
}

static int inf_npc_type_is_set_pressure(int type) {
    return type == INF_NPC_MAGER ||
        type == INF_NPC_MELEER ||
        type == INF_NPC_RANGER;
}

enum {
    INF_STYLE_MASK_MELEE = 1 << 0,
    INF_STYLE_MASK_RANGED = 1 << 1,
    INF_STYLE_MASK_MAGIC = 1 << 2,
};

static inline int inf_attack_style_mask_bit(int style) {
    if (style == ATTACK_STYLE_MELEE) return INF_STYLE_MASK_MELEE;
    if (style == ATTACK_STYLE_RANGED) return INF_STYLE_MASK_RANGED;
    if (style == ATTACK_STYLE_MAGIC) return INF_STYLE_MASK_MAGIC;
    return 0;
}

static inline int inf_cardinal_contact_with_npc(int px, int py, int nx, int ny, int npc_size) {
    int cx = px < nx ? nx : (px > nx + npc_size - 1 ? nx + npc_size - 1 : px);
    int cy = py < ny ? ny : (py > ny + npc_size - 1 ? ny + npc_size - 1 : py);
    int dx = px - cx; if (dx < 0) dx = -dx;
    int dy = py - cy; if (dy < 0) dy = -dy;
    return dx + dy == 1;
}

static inline int inf_npc_melee_contact_allows_attack_at_tile(
    int player_x,
    int player_y,
    InfNPCType npc_type,
    int npc_x,
    int npc_y,
    int npc_size,
    int dist
) {
    if (dist != 1) return 0;
    if (npc_type == INF_NPC_RANGER || npc_type == INF_NPC_MAGER)
        return 1;
    return inf_cardinal_contact_with_npc(
        player_x, player_y, npc_x, npc_y, npc_size);
}

static inline int inf_melee_fallback_possible_at_tile(
    int player_x, int player_y, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    if (!stats->can_melee || planned_style == ATTACK_STYLE_MELEE || dist != 1)
        return 0;

    return inf_npc_melee_contact_allows_attack_at_tile(
        player_x, player_y, npc->type, npc->x, npc->y, npc->size, dist);
}


static inline int inf_attack_style_options_mask_at_tile(
    int player_x, int player_y, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    int mask = inf_attack_style_mask_bit(planned_style);
    if (inf_melee_fallback_possible_at_tile(
            player_x, player_y, npc, stats, planned_style, dist))
        mask |= INF_STYLE_MASK_MELEE;
    return mask;
}

static inline int inf_attack_style_options_mask(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    return inf_attack_style_options_mask_at_tile(
        s->player.x, s->player.y, npc, stats, planned_style, dist);
}

static inline int inf_attack_style_from_mask(int style_mask) {
    if (style_mask == INF_STYLE_MASK_MELEE) return ATTACK_STYLE_MELEE;
    if (style_mask == INF_STYLE_MASK_RANGED) return ATTACK_STYLE_RANGED;
    if (style_mask == INF_STYLE_MASK_MAGIC) return ATTACK_STYLE_MAGIC;
    return ATTACK_STYLE_NONE;
}


static inline int inf_npc_style_max_hit(
    const InfNPCStats* stats, int style
) {
    if (style == ATTACK_STYLE_NONE) return 0;
    int max_hit = osrs_npc_max_hit(style,
        stats->str_level, stats->range_level,
        stats->melee_str_bonus, stats->ranged_str_bonus,
        stats->magic_base_dmg, stats->magic_dmg_pct);
    if (stats->max_hit_cap > 0 && max_hit > stats->max_hit_cap)
        max_hit = stats->max_hit_cap;
    return max_hit;
}


static inline int inf_attack_style_telegraph_mask(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    if (npc->type == INF_NPC_JAD)
        return inf_attack_style_mask_bit(planned_style);
    return inf_attack_style_options_mask(s, npc, stats, planned_style, dist);
}

typedef struct {
    int edge_distance;
    int style_mask;
    int telegraph_style_mask;
    int can_attack_if_ready;
    int can_attack_this_tick;
} InfNpcPlayerThreat;

static inline int inf_npc_planned_style_for_current_state(const InfNPC* npc) {
    if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) {
        OverheadPrayer scanned = (OverheadPrayer)npc->blob_scanned_prayer;
        if (scanned == PRAYER_PROTECT_MAGIC) return ATTACK_STYLE_RANGED;
        if (scanned == PRAYER_PROTECT_RANGED) return ATTACK_STYLE_MAGIC;
    }
    if (npc->type == INF_NPC_JAD)
        return inf_npc_jad_const(npc)->attack_style;
    return npc->attack_style;
}

static inline int inf_npc_type_has_player_pressure(InfNPCType type) {
    return type != INF_NPC_NIBBLER &&
        type != INF_NPC_ZUK &&
        type != INF_NPC_ZUK_SHIELD &&
        type != INF_NPC_HEALER_ZUK;
}

static inline InfNpcPlayerThreat inf_npc_player_threat_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    const InfNPC* npc
) {
    InfNpcPlayerThreat threat = {0};
    if (!npc->active || npc->death_ticks > 0)
        return threat;
    if (!inf_npc_type_has_player_pressure(npc->type))
        return threat;

    InfTargetArea target = inf_npc_current_target_area(s, npc);
    if (!target.is_player)
        return threat;

    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    int dist = encounter_dist_to_npc(
        s->player.x, s->player.y, npc->x, npc->y, npc->size);
    threat.edge_distance = dist;
    if (dist == 0 || dist > stats->attack_range)
        return threat;
    if (stats->attack_range == 1 &&
            !inf_npc_melee_contact_allows_attack_at_tile(
                s->player.x, s->player.y, npc->type,
                npc->x, npc->y, npc->size, dist))
        return threat;

    if (stats->attack_range > 1 &&
            !(npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) &&
            !inf_npc_has_los_to_tile_ctx(
                s, ctx, npc, s->player.x, s->player.y))
        return threat;

    threat.can_attack_if_ready = 1;

    int planned_style = inf_npc_planned_style_for_current_state(npc);
    if (planned_style != ATTACK_STYLE_NONE) {
        threat.style_mask = inf_attack_style_options_mask(
            s, npc, stats, planned_style, dist);
        threat.telegraph_style_mask = inf_attack_style_telegraph_mask(
            s, npc, stats, planned_style, dist);
    }

    threat.can_attack_this_tick = npc->attack_timer <= 1 &&
        npc->stun_timer == 0 &&
        npc->dig_freeze_timer == 0 &&
        npc->dig_attack_delay == 0;
    return threat;
}


static inline int inf_npc_is_live_player_target(
    const InfernoState* s,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;
    const InfNPC* npc = &s->npcs[npc_idx];
    return npc->active && npc->death_ticks == 0 && npc->hp > 0 &&
        npc->type != INF_NPC_ZUK_SHIELD;
}



static inline int inf_pending_hit_obs_timer(const EncounterPendingHit* ph) {
    if (ph->check_prayer && ph->prayer_check_delay > 0)
        return ph->prayer_check_delay;
    return ph->ticks_remaining;
}

#define INF_JAD_PROJECTILE_DELAY 3

static inline EncounterProjectileDelayOptions inf_npc_projectile_options(
    InfNPCType type, int style
) {
    EncounterProjectileDelayOptions options = {0};
    if (type == INF_NPC_MAGER && style == ATTACK_STYLE_MAGIC) {
        options.visual_delay_ticks = 2;
        options.visual_hit_early_ticks = -1;
    } else if (type == INF_NPC_RANGER && style == ATTACK_STYLE_RANGED) {
        options.reduce_delay = -2;
        options.visual_delay_ticks = 3;
    } else if (type == INF_NPC_JAD && style != ATTACK_STYLE_MELEE) {
        options.reduce_delay = INF_JAD_PROJECTILE_DELAY;
        options.visual_hit_early_ticks = -1;
    } else if (type == INF_NPC_ZUK) {
        options.set_delay = 4;
        options.visual_delay_ticks = 2;
    }
    return options;
}

static inline EncounterProjectileTiming inf_npc_projectile_timing(
    InfNPCType type, int style, int distance
) {
    return encounter_projectile_timing(
        distance, 0, encounter_projectile_delay_kind_for_style((AttackStyle)style),
        inf_npc_projectile_options(type, style));
}

static inline EncounterProjectileTiming inf_player_projectile_timing(
    int style, uint8_t weapon, int is_special, int distance
) {
    EncounterProjectileDelayKind kind = encounter_projectile_delay_kind_for_style((AttackStyle)style);
    EncounterProjectileDelayOptions options = {0};
    if (style == ATTACK_STYLE_RANGED) {
        if (weapon == ITEM_TOXIC_BLOWPIPE) {
            kind = ENCOUNTER_PROJECTILE_DELAY_THROWN;
            options.visual_delay_ticks = 1;
            if (is_special) {
                options.reduce_delay = -1;
                options.visual_hit_early_ticks = 1;
            }
        } else if (weapon == ITEM_TWISTED_BOW) {
            options.visual_delay_ticks = 1;
        }
    }
    return encounter_projectile_timing(distance, 1, kind, options);
}

static inline AttackStyle inf_jad_roll_primary_style(uint32_t* rng_state) {
    return (encounter_rand_int(rng_state, 2) == 0)
        ? ATTACK_STYLE_RANGED
        : ATTACK_STYLE_MAGIC;
}

static inline int inf_npc_attack_anim_id(const InfNPC* npc, const NpcModelMapping* nm) {
    if (npc->resurrecting_this_tick)
        return INF_GEN_ANIM_MAGER_RESURRECT;
    if (npc->type == INF_NPC_RANGER &&
        npc->attack_style_this_tick == ATTACK_STYLE_MELEE)
        return INF_GEN_ANIM_RANGER_ATTACK_MELEE;
    if (npc->type == INF_NPC_MAGER &&
        npc->attack_style_this_tick == ATTACK_STYLE_MELEE)
        return INF_GEN_ANIM_MAGER_ATTACK_MELEE;

    if (npc->type == INF_NPC_JAD) {
        switch (npc->attack_style_this_tick) {
            case ATTACK_STYLE_MAGIC:
                return INF_GEN_ANIM_JALTOK_JAD_ATTACK_MAGIC;
            case ATTACK_STYLE_RANGED:
                return INF_GEN_ANIM_JALTOK_JAD_ATTACK_RANGED;
            case ATTACK_STYLE_MELEE:
                return INF_GEN_ANIM_JALTOK_JAD_ATTACK_MELEE;
            default:
                return -1;
        }
    }

    if (!nm || nm->attack_anim == 65535)
        return -1;
    return (int)nm->attack_anim;
}

static inline int inf_npc_death_anim_id(const InfNPC* npc, const NpcModelMapping* nm) {
    switch (npc->type) {
        case INF_NPC_NIBBLER:
            return INF_GEN_ANIM_NIBBLER_DEATH;
        case INF_NPC_BAT:
            return INF_GEN_ANIM_BAT_DEATH;
        case INF_NPC_BLOB:
        case INF_NPC_BLOB_MELEE:
        case INF_NPC_BLOB_RANGE:
        case INF_NPC_BLOB_MAGE:
            return INF_GEN_ANIM_BLOB_DEATH;
        case INF_NPC_MELEER:
            return INF_GEN_ANIM_MELEER_DEATH;
        case INF_NPC_RANGER:
            return INF_GEN_ANIM_RANGER_DEATH;
        case INF_NPC_MAGER:
            return INF_GEN_ANIM_MAGER_DEATH;
        case INF_NPC_JAD:
            return INF_GEN_ANIM_JALTOK_JAD_DEATH;
        case INF_NPC_ZUK:
            return INF_GEN_ANIM_TZKAL_ZUK_DEATH;
        case INF_NPC_ZUK_SHIELD:
            return INF_GEN_ANIM_ZUK_SHIELD_DEATH;
        default:
            return nm ? (int)nm->idle_anim : -1;
    }
}

static inline AttackStyle inf_choose_attack_style_for_tick(
    uint32_t* rng_state, int style_mask
) {
    AttackStyle primary_style = (AttackStyle)inf_attack_style_from_mask(
        style_mask & ~INF_STYLE_MASK_MELEE);
    if ((style_mask & INF_STYLE_MASK_MELEE) && primary_style != ATTACK_STYLE_NONE)
        return (encounter_rand_int(rng_state, 2) == 0) ? ATTACK_STYLE_MELEE : primary_style;
    return (AttackStyle)inf_attack_style_from_mask(style_mask);
}

static inline int inf_dead_mob_is_resurrectable(InfNPCType type) {
    switch (type) {
        case INF_NPC_BAT:
        case INF_NPC_BLOB:
        case INF_NPC_MELEER:
        case INF_NPC_RANGER:
        case INF_NPC_MAGER:
            return 1;
        default:
            return 0;
    }
}

static void inf_store_dead_mob(InfernoState* s, InfNPC* npc) {
    if (s->dead_mob_count >= INF_MAX_DEAD_MOBS) return;
    if (!inf_dead_mob_is_resurrectable(npc->type)) return;
    if (npc->resurrection_count != 0) return;

    InfDeadMob* dm = &s->dead_mobs[s->dead_mob_count++];
    dm->type = npc->type;
    dm->x = npc->x;
    dm->y = npc->y;
    dm->hp = npc->max_hp / 2;
    dm->max_hp = npc->max_hp;
}

static float inf_compute_reward_ctx(InfernoState* s, const InfernoContext* ctx);
static void inf_spawn_wave(InfernoState* s);
static void inf_tick_npcs_ctx(InfernoState* s, InfernoContext* ctx);
static void inf_tick_player_ctx(
    InfernoState* s,
    InfernoContext* ctx,
    const int* actions,
    int can_attack);
static void inf_apply_npc_death(InfernoState* s, int npc_idx);
static int inf_mager_resurrect_ctx(InfernoState* s, InfernoContext* ctx, int idx);
static void inf_queue_zuk_healer_sparks(InfernoState* s, const InfNPC* npc);
static void inf_resolve_pending_sparks(InfernoState* s);
static void inf_invalidate_current_obs_slots(InfernoState* s);
static void inf_refresh_current_obs_slots_ctx(InfernoState* s, const InfernoContext* ctx);
static void inf_ensure_current_obs_slots_ctx(InfernoState* s, const InfernoContext* ctx);

static InfConfig inf_default_config(void) {
    return (InfConfig){
        .start_wave = 0,
        .loadout_profile_mode = INF_LOADOUT_PROFILE_MODE_MAX_ONLY,
        .budget_loadout_fraction = 0.0f,
        .damage_reward_coeff = 0.0f,
        .offensive_prayer_reward_coeff = 0.0f,
        .shield_penalty_coeff = 0.0f,
        .tag_reward_coeff = 0.0f,
        .late_start_supply_profile_scale = 1.0f,
        .curriculum_agent = 0,
        .curriculum_supply_jitter_mode = INF_CURRICULUM_SUPPLY_MODE_OFF,
        .curriculum_supply_shared_jitter = 0.0f,
        .curriculum_supply_brew_jitter = 0.0f,
        .curriculum_supply_restore_jitter = 0.0f,
        .curriculum_no_brew_mode = INF_CURRICULUM_SUPPLY_MODE_OFF,
        .curriculum_no_brew_frac = 0.0f,
        .supply_milestone_brew_reward_coeff = 0.0f,
        .supply_milestone_restore_reward_coeff = 0.0f,
        .death_penalty_coeff = 0.0f,
        .terminal_penalty_enabled = 0,
        .phase_900_bonus = 0.0f,
        .phase_600_bonus = 0.0f,
        .phase_300_bonus = 0.0f,
        .shield_penalty_episode_cap = 0.0f,
        .jad_damage_reward_coeff = 0.0f,
        .zuk_healer_damage_reward_coeff = 0.0f,
        .set_damage_reward_coeff = 0.0f,
        .jad_kill_bonus = 0.0f,
        .zuk_healer_kill_bonus = 0.0f,
        .set_kill_bonus = 0.0f,
        .post_healer_zuk_damage_coeff = 0.0f,
        .post_healer_set_damage_reward_coeff = 0.0f,
        .post_healer_set_kill_bonus = 0.0f,
        .post_healer_set_alive_tick_penalty_coeff = 0.0f,
        .post_healer_set_alive_penalty_cap = 0.0f,
        .zuk_healer_phase_hp_delta_coeff = 0.0f,
        .zuk_untagged_healer_tick_penalty_coeff = 0.0f,
        .zuk_untagged_healer_target_bonus_coeff = 0.0f,
        .zuk_safe_untagged_healer_target_bonus_coeff = 0.0f,
        .zuk_untagged_healer_nonmagic_attack_bonus_coeff = 0.0f,
        .zuk_healer_mage_attack_penalty_coeff = 0.0f,
        .zuk_safe_untagged_healer_target_mask = 0,
        .zuk_force_safe_untagged_healer_target_mask = 0,
        .zuk_healer_reward_mode = INF_ZUK_HEALER_REWARD_MODE_BASELINE,
        .post_jad_zuk_multiplier = 1.0f,
        .jad_alive_zuk_multiplier = 1.0f,
        .shield_tag_reward_coeff = 0.0f,
        .joseph_reward_mode = INF_JOSEPH_REWARD_MODE_OFF,
    };
}

typedef struct {
    EncounterArenaTopology*
        topologies[1 << INF_NUM_PILLARS];
} InfRouteTopologyOwner;

typedef struct {
    const CollisionMap* collision_map;
    int world_offset_x;
    int world_offset_y;
    uint8_t pillar_mask;
} InfRouteTopologyBuildContext;

static InfRouteTopologyOwner inf_route_topology_owner;

static uint32_t inf_topology_collision_flags(
    const InfRouteTopologyBuildContext* build,
    int x,
    int y
) {
    if (!build->collision_map) return 0;
    return (uint32_t)collision_get_flags(
        build->collision_map,
        0,
        x + build->world_offset_x,
        y + build->world_offset_y);
}

static uint32_t inf_route_topology_flags(void* data, int x, int y) {
    const InfRouteTopologyBuildContext* build =
        (const InfRouteTopologyBuildContext*)data;
    if (x < INF_ARENA_MIN_X || x > INF_ARENA_MAX_X ||
            y < INF_ARENA_MIN_Y || y > INF_ARENA_MAX_Y)
        return COLLISION_BLOCKED;
    uint32_t flags = inf_topology_collision_flags(build, x, y);
    for (int pillar = 0; pillar < INF_NUM_PILLARS; pillar++) {
        if ((build->pillar_mask & (1 << pillar)) == 0) continue;
        if (los_aabb_overlap(
                x, y, 1,
                INF_PILLAR_POS[pillar][0],
                INF_PILLAR_POS[pillar][1],
                INF_PILLAR_SIZE))
            flags |= COLLISION_BLOCKED | LOS_FULL_MASK;
    }
    return flags;
}

static uint32_t inf_route_topology_los_flags(void* data, int x, int y) {
    const InfRouteTopologyBuildContext* build =
        (const InfRouteTopologyBuildContext*)data;
    for (int pillar = 0; pillar < INF_NUM_PILLARS; pillar++) {
        if ((build->pillar_mask & (1 << pillar)) == 0) continue;
        if (los_aabb_overlap(
                x, y, 1,
                INF_PILLAR_POS[pillar][0],
                INF_PILLAR_POS[pillar][1],
                INF_PILLAR_SIZE))
            return LOS_FULL_MASK;
    }
    return 0;
}

#define INF_ROUTE_BAKE_PATH "inferno_route.bin"
#define INF_ROUTE_REVISION_BASE UINT64_C(0x494e4645524e4f02)
#define INF_ROUTE_BAKE_VERSION 1u
#define INF_ROUTE_BAKE_COUNT (1 << INF_NUM_PILLARS)

typedef struct InfRouteBakeHeader {
    char magic[8];
    uint32_t version;
    uint32_t topology_size;
    uint32_t count;
    int32_t origin_x;
    int32_t origin_y;
    int32_t width;
    int32_t height;
    int32_t max_footprint_size;
    int32_t world_offset_x;
    int32_t world_offset_y;
    uint32_t pad;
    uint64_t revision_base;
} InfRouteBakeHeader;

_Static_assert(sizeof(InfRouteBakeHeader) == 64,
    "inferno route bake header must stay 64 bytes");

static const char INF_ROUTE_BAKE_MAGIC[8] = {
    'I', 'N', 'F', 'R', 'O', 'U', 'T', 'E'
};

static int inf_route_topologies_all_empty(void) {
    for (int i = 0; i < INF_ROUTE_BAKE_COUNT; i++) {
        if (inf_route_topology_owner.topologies[i]) return 0;
    }
    return 1;
}

static void inf_route_topologies_free_range(int count) {
    for (int i = 0; i < count; i++) {
        free(inf_route_topology_owner.topologies[i]);
        inf_route_topology_owner.topologies[i] = NULL;
    }
}

static int inf_route_bake_header_matches(
    const InfRouteBakeHeader* header,
    const InfernoContext* ctx
) {
    return header
        && memcmp(header->magic, INF_ROUTE_BAKE_MAGIC, 8) == 0
        && header->version == INF_ROUTE_BAKE_VERSION
        && header->topology_size == (uint32_t)sizeof(EncounterArenaTopology)
        && header->count == (uint32_t)INF_ROUTE_BAKE_COUNT
        && header->origin_x == INF_TOPOLOGY_MIN_X
        && header->origin_y == INF_TOPOLOGY_MIN_Y
        && header->width == INF_TOPOLOGY_WIDTH
        && header->height == INF_TOPOLOGY_HEIGHT
        && header->max_footprint_size ==
            ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE
        && header->world_offset_x == ctx->world_offset_x
        && header->world_offset_y == ctx->world_offset_y
        && header->revision_base == INF_ROUTE_REVISION_BASE;
}

static int inf_baked_topology_matches(
    const EncounterArenaTopology* topology,
    int pillar_mask
) {
    return topology
        && topology->finalized
        && topology->origin_x == INF_TOPOLOGY_MIN_X
        && topology->origin_y == INF_TOPOLOGY_MIN_Y
        && topology->width == INF_TOPOLOGY_WIDTH
        && topology->height == INF_TOPOLOGY_HEIGHT
        && topology->tile_count == INF_TOPOLOGY_WIDTH * INF_TOPOLOGY_HEIGHT
        && topology->max_footprint_size ==
            ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE
        && topology->revision ==
            INF_ROUTE_REVISION_BASE + (uint64_t)pillar_mask
        && topology->los_build_mode ==
            ENCOUNTER_ARENA_TOPOLOGY_LOS_BUILD_FLAGGED;
}

static int inf_try_load_baked_route_topologies(const InfernoContext* ctx) {
    if (!ctx || !ctx->collision_map || !inf_route_topologies_all_empty())
        return 0;
    if (!osrs_asset_exists(INF_ROUTE_BAKE_PATH)) return 0;

    const char* path = osrs_asset_path(INF_ROUTE_BAKE_PATH);
    FILE* f = osrs_asset_fopen(INF_ROUTE_BAKE_PATH, "rb");
    if (!f) return 0;

    InfRouteBakeHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1
            || !inf_route_bake_header_matches(&header, ctx)) {
        fprintf(stderr, "%s: stale or incompatible Inferno route bake\n", path);
        fclose(f);
        return 0;
    }

    for (int pillar_mask = 0; pillar_mask < INF_ROUTE_BAKE_COUNT; pillar_mask++) {
        EncounterArenaTopology* topology =
            (EncounterArenaTopology*)calloc(1, sizeof(*topology));
        if (!topology) {
            fprintf(stderr, "%s: malloc failed for baked topology %d\n",
                path, pillar_mask);
            inf_route_topologies_free_range(pillar_mask);
            fclose(f);
            abort();
        }
        if (fread(topology, sizeof(*topology), 1, f) != 1
                || !inf_baked_topology_matches(topology, pillar_mask)) {
            fprintf(stderr, "%s: bad baked topology %d\n", path, pillar_mask);
            free(topology);
            inf_route_topologies_free_range(pillar_mask);
            fclose(f);
            return 0;
        }
        inf_route_topology_owner.topologies[pillar_mask] = topology;
    }
    fclose(f);
    fprintf(stderr, "inferno route topology loaded from %s\n", path);
    return 1;
}

static void inf_write_baked_route_topologies(const InfernoContext* ctx) {
#ifndef PLATFORM_WEB
    if (!ctx || !ctx->collision_map) return;
    for (int i = 0; i < INF_ROUTE_BAKE_COUNT; i++) {
        if (!inf_baked_topology_matches(
                inf_route_topology_owner.topologies[i], i))
            return;
    }

    InfRouteBakeHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, INF_ROUTE_BAKE_MAGIC, 8);
    header.version = INF_ROUTE_BAKE_VERSION;
    header.topology_size = (uint32_t)sizeof(EncounterArenaTopology);
    header.count = (uint32_t)INF_ROUTE_BAKE_COUNT;
    header.origin_x = INF_TOPOLOGY_MIN_X;
    header.origin_y = INF_TOPOLOGY_MIN_Y;
    header.width = INF_TOPOLOGY_WIDTH;
    header.height = INF_TOPOLOGY_HEIGHT;
    header.max_footprint_size = ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE;
    header.world_offset_x = ctx->world_offset_x;
    header.world_offset_y = ctx->world_offset_y;
    header.revision_base = INF_ROUTE_REVISION_BASE;

    const char* path = osrs_asset_path(INF_ROUTE_BAKE_PATH);
    char tmp_path[2052];
    int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (n < 0 || (size_t)n >= sizeof(tmp_path)) {
        fprintf(stderr, "inferno route bake path too long: %s\n", path);
        return;
    }
    FILE* f = fopen(tmp_path, "wb");
    if (!f) {
        fprintf(stderr, "inferno route bake: cannot write %s\n", tmp_path);
        return;
    }
    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "inferno route bake: header write failed\n");
        fclose(f);
        remove(tmp_path);
        return;
    }
    for (int i = 0; i < INF_ROUTE_BAKE_COUNT; i++) {
        if (fwrite(inf_route_topology_owner.topologies[i],
                sizeof(EncounterArenaTopology), 1, f) != 1) {
            fprintf(stderr, "inferno route bake: topology %d write failed\n", i);
            fclose(f);
            remove(tmp_path);
            return;
        }
    }
    if (fclose(f) != 0 || rename(tmp_path, path) != 0) {
        fprintf(stderr, "inferno route bake: failed to publish %s\n", path);
        remove(tmp_path);
        return;
    }
    fprintf(stderr, "inferno route topology baked to %s (%zu bytes)\n",
        path,
        sizeof(header) + sizeof(EncounterArenaTopology) * (size_t)INF_ROUTE_BAKE_COUNT);
#else
    (void)ctx;
#endif
}

static void inf_finalize_route_topology(InfernoContext* ctx) {
    if (!ctx || ctx->route_topology) {
        fprintf(stderr, "inferno route topology finalized twice\n");
        abort();
    }
    inf_try_load_baked_route_topologies(ctx);
    int built = 0;
    for (int pillar_mask = 0;
            pillar_mask < (1 << INF_NUM_PILLARS);
            pillar_mask++) {
        InfRouteTopologyBuildContext build = {
            ctx->collision_map,
            ctx->world_offset_x,
            ctx->world_offset_y,
            (uint8_t)pillar_mask,
        };
        EncounterArenaTopologyBuildSpec spec = {
            .origin_x = INF_TOPOLOGY_MIN_X,
            .origin_y = INF_TOPOLOGY_MIN_Y,
            .width = INF_TOPOLOGY_WIDTH,
            .height = INF_TOPOLOGY_HEIGHT,
            .max_footprint_size =
                ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE,
            .revision =
                INF_ROUTE_REVISION_BASE + (uint64_t)pillar_mask,
            .tile_flags = inf_route_topology_flags,
            .tile_flags_ctx = &build,
            .los_tile_flags = inf_route_topology_los_flags,
            .los_tile_flags_ctx = &build,
        };
        EncounterArenaTopology** topology =
            &inf_route_topology_owner.topologies[pillar_mask];
        if (!*topology) {
            *topology = encounter_arena_topology_build(&spec);
            encounter_arena_topology_finalize(*topology);
            built = 1;
        } else {
            encounter_arena_topology_require_spec(
                *topology, &spec, "inferno");
        }
    }
    ctx->route_topology = inf_route_topology_owner.topologies[0];
    if (built) inf_write_baked_route_topologies(ctx);
}

static const EncounterArenaTopology* inf_route_topology_for_state(
    const InfernoContext* ctx,
    const InfernoState* state
) {
    encounter_arena_topology_require_finalized(ctx->route_topology);
    if (!inf_pillars_at_canonical_positions(state))
        return ctx->route_topology;
    return inf_route_topology_owner.topologies[
        inf_active_pillar_mask(state)];
}

static void inf_finalize_context(
    EncounterState* state,
    EncounterContext* context
) {
    (void)state;
    inf_finalize_route_topology((InfernoContext*)context);
}

static void inf_init_context_typed(InfernoContext* ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->config = inf_default_config();
    inf_reset_npc_player_los_frame(ctx);
}

static void inf_init_context(EncounterContext* context) {
    inf_init_context_typed((InfernoContext*)context);
}

static void inf_destroy_context(EncounterContext* context) {
}


static void inf_init_state_typed(InfernoState* s, InfernoContext* ctx) {
    inf_build_npc_stats();
    memset(s, 0, sizeof(*s));
    s->rng_state = 12345;
    s->start_wave = ctx ? ctx->config.start_wave : 0;
    s->active_loadout_profile = INF_LOADOUT_PROFILE_MAX;
}

static void inf_init_state(EncounterState* state, EncounterContext* context) {
    inf_init_state_typed((InfernoState*)state, (InfernoContext*)context);
}

static EncounterState* inf_create(void) {
    InfernoState* s = (InfernoState*)calloc(1, sizeof(InfernoState));
    inf_init_state_typed(s, NULL);
    return (EncounterState*)s;
}

static void inf_destroy(EncounterState* state) {
    free(state);
}
