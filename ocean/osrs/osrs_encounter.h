/**
 * @fileoverview osrs_encounter.h — shared encounter mechanics for the current ocean OSRS envs.
 *
 * this header holds reusable mechanics that current encounters build on.
 * encounter-specific policy should stay in the encounter header; shared helpers
 * here should stay generic enough to be reused by future envs.
 *
 * SHARED SYSTEMS (in order of appearance in this file):
 *
 *   rendering:
 *     RenderEntity                     value struct for renderer (not Player*)
 *     render_entity_from_player()      copy Player fields to RenderEntity
 *     encounter_resolve_attack_target() match npc_slot to render entity index
 *     EncounterOverlay                 visual overlay (hazards, projectiles, boss)
 *
 *   prayer (toggle-semantic, matches real OSRS click behavior):
 *     ENCOUNTER_OVERHEAD_*                canonical overhead action encoding (4/6 dim)
 *     ENCOUNTER_OFFENSIVE_*               canonical offensive action encoding (4 dim)
 *     encounter_apply_overhead_action()   apply overhead action, returns 1 on activation
 *     encounter_apply_offensive_action()  apply offensive action, returns 1 on activation
 *     encounter_drain_all_prayers()       drain both slots per tick (activation-tick skip)
 *
 *   movement:
 *     ENCOUNTER_MOVE_TARGET_DX/DY[25]  direction tables (idle + 8 walk + 16 run)
 *     encounter_move_to_target()       player movement: walk 1 tile or run 2
 *     encounter_move_toward_dest()     BFS click-to-move toward destination
 *     encounter_pathfind()             shared BFS pathfind wrapper
 *
 *   NPC pathfinding:
 *     encounter_npc_step_out_from_under()  shuffle NPC off player tile (OSRS overlap rule)
 *     encounter_npc_step_toward()      OSRS size-aware chase step
 *
 *   damage:
 *     encounter_damage_player()        apply damage to player (HP, clamp, splat, tracker)
 *     encounter_damage_npc()           apply damage to NPC (HP, splat flags)
 *
 *   per-tick flags:
 *     encounter_clear_tick_flags()     reset animation/event flags each tick
 *
 *   gear switching:
 *     encounter_apply_loadout()        memcpy loadout + set gear state
 *     encounter_populate_inventory()   dedup items from multiple loadouts for GUI
 *
 *   combat stats:
 *     EncounterLoadoutStats            derived stats (att bonus, max hit, eff level...)
 *     Player.offensive_prayer          runtime state, source of truth for prayer multipliers
 *     encounter_compute_loadout_stats() derive all stats from ITEM_DATABASE + loadout
 *
 *   hit delays:
 *     EncounterPendingHit              queued damage with tick countdown
 *
 * ALSO SEE:
 *   osrs_combat.h                     hit chance, tbow formula, barrage AoE, delay formulas
 *   osrs_pvp_combat.h                 PvP-specific damage (prayer, veng, recoil, smite)
 */

#ifndef OSRS_ENCOUNTER_H
#define OSRS_ENCOUNTER_H

#include <stdint.h>
#include <string.h>
#include "osrs_types.h"
#include "osrs_items.h"
#include "osrs_pathfinding.h"
#include "osrs_combat.h"
#include "osrs_item_effects.h"
#include "osrs_human_input_types.h"

/* opaque encounter state — each encounter defines its own struct */
typedef struct EncounterState EncounterState;

/* ======================================================================== */
/* shared pending hit system for delayed projectile damage                   */
/* ======================================================================== */

#define ENCOUNTER_MAX_PENDING_HITS 8

/* spell types for barrage freeze/heal effects on pending hits */
#define ENCOUNTER_SPELL_NONE  0
#define ENCOUNTER_SPELL_ICE   1   /* ice barrage: freeze on hit */
#define ENCOUNTER_SPELL_BLOOD 2   /* blood barrage: heal 25% of AoE damage */

typedef struct {
    int active;
    int damage;
    int ticks_remaining;   /* countdown to landing */
    int attack_style;      /* ATTACK_STYLE_* for prayer check at land time */
    int check_prayer;      /* 1 = prayer has NOT been checked yet (deferred) */
    int prayer_check_delay;/* ticks until prayer is checked (0 = check immediately on next resolve).
                              jad uses 3 to model its T+3 DelayedAction — prayer at T+3 decides
                              whether the hit is blocked, independent of projectile flight time.
                              ref: InfernoTrainer JalTokJad.ts:49-57. */
    int spell_type;        /* ENCOUNTER_SPELL_* for freeze/heal effects */
    int source_npc_type;   /* encounter-local NPC type for custom delayed rolls */
} EncounterPendingHit;

/* visual overlay data: shared between encounter and renderer.
   encounter's render_post_tick populates this, renderer reads it. */
#define ENCOUNTER_MAX_OVERLAY_TILES 16
#define ENCOUNTER_MAX_OVERLAY_ADDS 4
/* inferno can legitimately exceed single-digit projectile counts in one tick,
   especially during Zuk healer spark volleys. size this from real encounter
   volume so the renderer never silently drops visual events. */
#define ENCOUNTER_MAX_OVERLAY_PROJECTILES 48

typedef enum {
    ENCOUNTER_PROJECTILE_MOTION_OSRS_FLIGHT = 0,
    ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED = 1,
} EncounterProjectileMotionMode;

typedef struct {
    /* encounter-defined area hazards. current users write 3x3 poison clouds. */
    struct { int x, y, active; } hazards[ENCOUNTER_MAX_OVERLAY_TILES];
    int hazard_count;

    /* boss state */
    int boss_x, boss_y, boss_visible;
    int boss_form;  /* encounter-specific form/phase index */
    int boss_size;  /* NPC size in tiles (e.g. 5 for Zulrah) */

    /* encounter adds or secondary mobs. variant is encounter-defined. */
    struct { int x, y, active, variant; } adds[ENCOUNTER_MAX_OVERLAY_ADDS];
    int add_count;

    /* visual projectiles: brief flash from source to target.
       encounters fire attacks instantly, but we show a 1-tick projectile
       for visual clarity. the renderer draws these and auto-expires them. */
    struct {
        int active;
        int src_x, src_y;   /* source tile (e.g. Zulrah position) */
        int dst_x, dst_y;   /* target tile (e.g. player position) */
        int style;           /* 0=ranged, 1=magic, 2=melee, 3=cloud, 4=spawn_orb */
        int damage;          /* for hit splat at destination */
        /* flight parameters — encounters set these, renderer reads them */
        int duration_ticks;  /* flight duration in client ticks (0 = use default 35) */
        int start_h;         /* start height in OSRS units /128 (0 = use default) */
        int end_h;           /* end height in OSRS units /128 (0 = use default) */
        int curve;           /* OSRS slope param (0 = use default 16) */
        float arc_height;    /* sinusoidal arc peak in tiles (0 = quadratic/straight) */
        int tracks_target;   /* 1 = re-aim toward target each tick */
        int start_delay;     /* ticks before projectile becomes visible (0 = immediate) */
        int motion_mode;     /* EncounterProjectileMotionMode */
        float offset_x, offset_y, offset_z; /* local multi-model offset */
        int src_size;        /* source entity size for center offset (0 = use boss_size) */
        int dst_size;        /* target entity size for center offset (1 = player) */
        uint32_t model_id;   /* GFX model from cache (0 = style-based fallback) */
        int anim_id;         /* spotanim animation sequence (-1 = static model) */
        int impact_gfx_id;   /* optional landing spotanim to spawn on arrival */
    } projectiles[ENCOUNTER_MAX_OVERLAY_PROJECTILES];
    int projectile_count;

    /* melee targeting: shows which tile Zulrah is staring at */
    int melee_target_active;
    int melee_target_x, melee_target_y;
} EncounterOverlay;

/* map AttackStyle enum to overlay projectile style index.
   used by encounter_emit_projectile and render overlay systems. */
static inline int encounter_attack_style_to_proj_style(int attack_style) {
    switch (attack_style) {
        case ATTACK_STYLE_RANGED: return 0;
        case ATTACK_STYLE_MAGIC:  return 1;
        case ATTACK_STYLE_MELEE:  return 2;
        default: return 0;
    }
}

/* populate an overlay projectile slot with flight parameters.
   encounters should call this instead of filling fields manually. */
static inline int encounter_emit_projectile(
    EncounterOverlay* ov,
    int src_x, int src_y, int dst_x, int dst_y,
    int style, int damage,
    int duration_ticks, int start_h, int end_h, int curve,
    float arc_height, int tracks_target, int src_size, int dst_size,
    uint32_t model_id, int impact_gfx_id
) {
    if (ov->projectile_count >= ENCOUNTER_MAX_OVERLAY_PROJECTILES) return -1;
    int i = ov->projectile_count++;
    ov->projectiles[i].active = 1;
    ov->projectiles[i].src_x = src_x;
    ov->projectiles[i].src_y = src_y;
    ov->projectiles[i].dst_x = dst_x;
    ov->projectiles[i].dst_y = dst_y;
    ov->projectiles[i].style = style;
    ov->projectiles[i].damage = damage;
    ov->projectiles[i].duration_ticks = duration_ticks;
    ov->projectiles[i].start_h = start_h;
    ov->projectiles[i].end_h = end_h;
    ov->projectiles[i].curve = curve;
    ov->projectiles[i].arc_height = arc_height;
    ov->projectiles[i].start_delay = 0;
    ov->projectiles[i].motion_mode = ENCOUNTER_PROJECTILE_MOTION_OSRS_FLIGHT;
    ov->projectiles[i].offset_x = 0.0f;
    ov->projectiles[i].offset_y = 0.0f;
    ov->projectiles[i].offset_z = 0.0f;
    ov->projectiles[i].tracks_target = tracks_target;
    ov->projectiles[i].src_size = src_size;
    ov->projectiles[i].dst_size = dst_size;
    ov->projectiles[i].model_id = model_id;
    ov->projectiles[i].anim_id = -1;
    ov->projectiles[i].impact_gfx_id = impact_gfx_id;
    return i;
}

static inline void encounter_set_projectile_motion_mode(
    EncounterOverlay* ov, int projectile_idx, int motion_mode
) {
    if (projectile_idx < 0 || projectile_idx >= ov->projectile_count) return;
    ov->projectiles[projectile_idx].motion_mode = motion_mode;
}

static inline void encounter_set_projectile_animation(
    EncounterOverlay* ov, int projectile_idx, int anim_id
) {
    if (projectile_idx < 0 || projectile_idx >= ov->projectile_count) return;
    ov->projectiles[projectile_idx].anim_id = anim_id;
}

static inline void encounter_set_projectile_offset(
    EncounterOverlay* ov, int projectile_idx,
    float offset_x, float offset_y, float offset_z
) {
    if (projectile_idx < 0 || projectile_idx >= ov->projectile_count) return;
    ov->projectiles[projectile_idx].offset_x = offset_x;
    ov->projectiles[projectile_idx].offset_y = offset_y;
    ov->projectiles[projectile_idx].offset_z = offset_z;
}

/* ======================================================================== */
/* render entity: shared abstraction for renderer (value type, not pointer)  */
/* ======================================================================== */

typedef struct {
    EntityType entity_type;
    int npc_def_id;
    int npc_visible;
    int npc_size;
    int npc_anim_id;
    int x, y;
    int dest_x, dest_y;
    int current_hitpoints, base_hitpoints;
    int special_energy;
    OverheadPrayer prayer;
    GearSet visible_gear;
    int frozen_ticks;
    int veng_active;
    int is_running;
    AttackStyle attack_style_this_tick;
    int magic_type_this_tick;
    int hit_landed_this_tick;
    int hit_damage;
    int hit_was_successful;
    int hit_spell_type;  /* ENCOUNTER_SPELL_* for barrage impact effects on NPCs */
    int cast_veng_this_tick;
    int ate_food_this_tick;
    int ate_karambwan_this_tick;
    int used_special_this_tick;
    uint8_t equipped[NUM_GEAR_SLOTS];
    int npc_slot;  /* source slot index in encounter's NPC array; -1 for player */
    int attack_target_entity_idx;  /* render entity index of attack target, -1 = none */
} RenderEntity;

/** Fill a RenderEntity from a Player struct. */
static inline void render_entity_from_player(const Player* p, RenderEntity* out) {
    out->entity_type = p->entity_type;
    out->npc_def_id = p->npc_def_id;
    out->npc_visible = p->npc_visible;
    out->npc_size = p->npc_size;
    out->npc_anim_id = p->npc_anim_id;
    out->x = p->x;
    out->y = p->y;
    out->dest_x = p->dest_x;
    out->dest_y = p->dest_y;
    out->current_hitpoints = p->current_hitpoints;
    out->base_hitpoints = p->base_hitpoints;
    out->special_energy = p->special_energy;
    out->prayer = p->prayer;
    out->visible_gear = p->visible_gear;
    out->frozen_ticks = p->frozen_ticks;
    out->veng_active = p->veng_active;
    out->is_running = p->is_running;
    out->attack_style_this_tick = p->attack_style_this_tick;
    out->magic_type_this_tick = p->magic_type_this_tick;
    out->hit_landed_this_tick = p->hit_landed_this_tick;
    out->hit_damage = p->hit_damage;
    out->hit_was_successful = p->hit_was_successful;
    out->cast_veng_this_tick = p->cast_veng_this_tick;
    out->ate_food_this_tick = p->ate_food_this_tick;
    out->ate_karambwan_this_tick = p->ate_karambwan_this_tick;
    out->used_special_this_tick = p->used_special_this_tick;
    memcpy(out->equipped, p->equipped, NUM_GEAR_SLOTS);
    out->npc_slot = -1;  /* player, not an NPC */
    out->attack_target_entity_idx = -1;
}

/** Resolve attack_target_entity_idx for entity 0 (player) by matching npc_slot.
    call after fill_render_entities populates the entity array. any encounter with
    NPC targeting should call this so the renderer faces the correct target. */
static inline void encounter_resolve_attack_target(
    RenderEntity* entities, int count, int target_npc_slot
) {
    entities[0].attack_target_entity_idx = -1;
    if (target_npc_slot < 0) return;
    for (int i = 1; i < count; i++) {
        if (entities[i].npc_slot == target_npc_slot) {
            entities[0].attack_target_entity_idx = i;
            return;
        }
    }
}

/* ======================================================================== */
/* canonical prayer action encoding (toggle semantics, matches OSRS)         */
/*                                                                           */
/* real OSRS has no "turn off" button — clicking an active prayer icon       */
/* toggles it off. clicking a different prayer in the same slot replaces it. */
/* our encoding mirrors that exactly: agent action either no-ops or targets  */
/* a specific prayer; target-already-active → off, otherwise activate.       */
/*                                                                           */
/* each encounter chooses its action-head dim based on which prayers it      */
/* exposes — PvE uses 4 (no smite/redemption), PvP uses 6. new encounters    */
/* wire up by:                                                               */
/*   1. declaring two action heads with encounter_overhead_dim /             */
/*      ENCOUNTER_OFFENSIVE_DIM                                              */
/*   2. calling encounter_apply_overhead_action()  on pretick                */
/*   3. calling encounter_apply_offensive_action() on pretick                */
/*   4. calling encounter_drain_all_prayers() on pretick (handles both slots */
/*      + activation-tick skip + pp=0 auto-clear)                            */
/* ======================================================================== */

/* overhead action encoding. dim depends on encounter:
   - PvE (inferno/zulrah): 4 dim, actions 0-3 only
   - PvP: 6 dim, full range */
#define ENCOUNTER_OVERHEAD_NO_CHANGE          0
#define ENCOUNTER_OVERHEAD_TOGGLE_MELEE       1
#define ENCOUNTER_OVERHEAD_TOGGLE_RANGED      2
#define ENCOUNTER_OVERHEAD_TOGGLE_MAGIC       3
#define ENCOUNTER_OVERHEAD_TOGGLE_SMITE       4
#define ENCOUNTER_OVERHEAD_TOGGLE_REDEMPTION  5
#define ENCOUNTER_OVERHEAD_DIM_PVE            4
#define ENCOUNTER_OVERHEAD_DIM_PVP            6

/* offensive action encoding — 4 dim, shared by all encounters. */
#define ENCOUNTER_OFFENSIVE_NO_CHANGE         0
#define ENCOUNTER_OFFENSIVE_TOGGLE_PIETY      1
#define ENCOUNTER_OFFENSIVE_TOGGLE_RIGOUR     2
#define ENCOUNTER_OFFENSIVE_TOGGLE_AUGURY     3
#define ENCOUNTER_OFFENSIVE_DIM               4

/** apply an overhead prayer action with toggle semantics.
    target-already-active → set to PRAYER_NONE (toggle off).
    target-not-active → activate target (replacing whatever was in the slot).
    returns 1 on OFF→ON transition (caller should set prayer_just_activated),
    0 on no-op, toggle-off, or replace. */
static inline int encounter_apply_overhead_action(OverheadPrayer* overhead, int action) {
    OverheadPrayer target;
    switch (action) {
        case ENCOUNTER_OVERHEAD_NO_CHANGE:          return 0;
        case ENCOUNTER_OVERHEAD_TOGGLE_MELEE:       target = PRAYER_PROTECT_MELEE;  break;
        case ENCOUNTER_OVERHEAD_TOGGLE_RANGED:      target = PRAYER_PROTECT_RANGED; break;
        case ENCOUNTER_OVERHEAD_TOGGLE_MAGIC:       target = PRAYER_PROTECT_MAGIC;  break;
        case ENCOUNTER_OVERHEAD_TOGGLE_SMITE:       target = PRAYER_SMITE;          break;
        case ENCOUNTER_OVERHEAD_TOGGLE_REDEMPTION:  target = PRAYER_REDEMPTION;     break;
        default: return 0;
    }
    if (*overhead == target) { *overhead = PRAYER_NONE; return 0; }
    int activating = (*overhead == PRAYER_NONE) ? 1 : 0;
    *overhead = target;
    return activating;
}

/** apply an offensive prayer action with toggle semantics.
    same rules as overhead: target-active → off, target-inactive → activate.
    returns 1 on OFF→ON transition, 0 otherwise. */
static inline int encounter_apply_offensive_action(OffensivePrayer* offensive, int action) {
    OffensivePrayer target;
    switch (action) {
        case ENCOUNTER_OFFENSIVE_NO_CHANGE:       return 0;
        case ENCOUNTER_OFFENSIVE_TOGGLE_PIETY:    target = OFFENSIVE_PRAYER_PIETY;  break;
        case ENCOUNTER_OFFENSIVE_TOGGLE_RIGOUR:   target = OFFENSIVE_PRAYER_RIGOUR; break;
        case ENCOUNTER_OFFENSIVE_TOGGLE_AUGURY:   target = OFFENSIVE_PRAYER_AUGURY; break;
        default: return 0;
    }
    if (*offensive == target) { *offensive = OFFENSIVE_PRAYER_NONE; return 0; }
    int activating = (*offensive == OFFENSIVE_PRAYER_NONE) ? 1 : 0;
    *offensive = target;
    return activating;
}

/* ======================================================================== */
/* shared movement: 25-action system (idle + 8 walk + 16 run)                */
/* ======================================================================== */

/* 25 movement actions: idle(0), walk(1-8), run(9-24) */
#define ENCOUNTER_MOVE_ACTIONS 25

/* target offsets: (dx, dy) relative to player position */
static const int ENCOUNTER_MOVE_TARGET_DX[25] = {
    0,                          /* 0: idle */
    -1, -1, -1, 0, 0, 1, 1, 1, /* 1-8: walk (dist 1) */
    -2, -2, -2, -2, -2,        /* 9-13: run west edge */
    -1, -1,                     /* 14-15: run inner */
    0, 0,                       /* 16-17: run N/S 2 tiles */
    1, 1,                       /* 18-19: run inner */
    2, 2, 2, 2, 2              /* 20-24: run east edge */
};
static const int ENCOUNTER_MOVE_TARGET_DY[25] = {
    0,
    -1, 0, 1, -1, 1, -1, 0, 1,
    -2, -1, 0, 1, 2,
    -2, 2,
    -2, 2,
    -2, 2,
    -2, -1, 0, 1, 2
};

/* callback: returns 1 if tile (x, y) is walkable for the encounter.
   ctx is encounter-specific state (InfernoState*, ZulrahState*, etc.) */
typedef int (*encounter_walkable_fn)(void* ctx, int x, int y);

/** move player toward target offset via up to 2 greedy steps.
    walk actions (dist 1) take 1 step, run actions (dist 2) take up to 2.
    sets is_running = 1 if 2 steps were taken.
    returns number of tiles moved (0, 1, or 2). */
static inline int encounter_move_to_target(
    Player* p, int target_dx, int target_dy,
    encounter_walkable_fn is_walkable, void* ctx
) {
    int tx = p->x + target_dx;
    int ty = p->y + target_dy;
    int dist = abs(target_dx) > abs(target_dy) ? abs(target_dx) : abs(target_dy);
    int max_steps = dist;  /* 1 for walk, 2 for run */
    int steps = 0;

    for (int step = 0; step < max_steps; step++) {
        if (p->x == tx && p->y == ty) break;
        /* greedy step toward target */
        int dx = 0, dy = 0;
        if (tx > p->x) dx = 1; else if (tx < p->x) dx = -1;
        if (ty > p->y) dy = 1; else if (ty < p->y) dy = -1;

        /* try diagonal, x-only, y-only */
        int moved = 0;
        if (dx != 0 && dy != 0 && is_walkable(ctx, p->x + dx, p->y + dy)) {
            p->x += dx; p->y += dy; moved = 1;
        } else if (dx != 0 && is_walkable(ctx, p->x + dx, p->y)) {
            p->x += dx; moved = 1;
        } else if (dy != 0 && is_walkable(ctx, p->x, p->y + dy)) {
            p->y += dy; moved = 1;
        }
        if (!moved) break;
        steps++;
    }

    p->is_running = (steps == 2);
    p->dest_x = p->x;
    p->dest_y = p->y;
    return steps;
}

/* ======================================================================== */
/* shared BFS click-to-move (human mode + destination-based movement)        */
/* ======================================================================== */

/* shared BFS pathfind wrapper — translates local coords to world coords for pathfind_step.
   extra_blocked/blocked_ctx: optional callback for dynamic obstacles (pillars etc.).
   pass NULL/NULL for encounters with no dynamic obstacles. */
static inline PathResult encounter_pathfind(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y, int dst_x, int dst_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx
) {
    return pathfind_step(cmap, 0,
        src_x + world_offset_x, src_y + world_offset_y,
        dst_x + world_offset_x, dst_y + world_offset_y,
        extra_blocked, blocked_ctx);
}

/* arena-scoped BFS: same as encounter_pathfind but uses a smaller grid.
   arena_base_x/y: world-space origin of the arena.
   arena_w/h: arena dimensions in tiles (must be <= PATHFIND_ARENA_MAX). */
static inline PathResult encounter_pathfind_arena(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y, int dst_x, int dst_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    return pathfind_step_arena(cmap, 0,
        src_x + world_offset_x, src_y + world_offset_y,
        dst_x + world_offset_x, dst_y + world_offset_y,
        extra_blocked, blocked_ctx,
        arena_base_x + world_offset_x, arena_base_y + world_offset_y,
        arena_w, arena_h);
}

/* shared click-to-move: BFS toward destination, take up to 2 steps (run).
   call each tick when player_dest is set. clears dest when arrived.
   extra_blocked/blocked_ctx: optional dynamic obstacle callback for BFS.
   returns steps taken (0, 1, or 2). */
static inline int encounter_move_toward_dest(
    Player* p, int* dest_x, int* dest_y,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    if (*dest_x < 0 || *dest_y < 0) return 0;
    if (p->x == *dest_x && p->y == *dest_y) {
        *dest_x = -1; *dest_y = -1;
        return 0;
    }
    int steps = 0;
    for (int step = 0; step < 2; step++) {
        if (p->x == *dest_x && p->y == *dest_y) break;
        PathResult pr = (arena_w > 0)
            ? encounter_pathfind_arena(cmap, world_offset_x, world_offset_y,
                                       p->x, p->y, *dest_x, *dest_y,
                                       extra_blocked, blocked_ctx,
                                       arena_base_x, arena_base_y, arena_w, arena_h)
            : encounter_pathfind(cmap, world_offset_x, world_offset_y,
                                  p->x, p->y, *dest_x, *dest_y,
                                  extra_blocked, blocked_ctx);
        if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) break;
        int nx = p->x + pr.next_dx, ny = p->y + pr.next_dy;
        if (!is_walkable(ctx, nx, ny)) break;
        p->x = nx; p->y = ny;
        steps++;
    }
    p->is_running = (steps == 2);
    p->dest_x = p->x; p->dest_y = p->y;
    return steps;
}

/* ======================================================================== */
/* shared attack-target chase (auto-walk toward out-of-range target)         */
/* ======================================================================== */

/* footprint helpers for player-vs-target chase and range checks. */
static inline int encounter_entity_footprint_distance(
    int ax, int ay, int a_size,
    int bx, int by, int b_size
) {
    int ax1 = ax + a_size - 1;
    int ay1 = ay + a_size - 1;
    int bx1 = bx + b_size - 1;
    int by1 = by + b_size - 1;

    int dx = 0;
    if (ax1 < bx) dx = bx - ax1;
    else if (bx1 < ax) dx = ax - bx1;

    int dy = 0;
    if (ay1 < by) dy = by - ay1;
    else if (by1 < ay) dy = ay - by1;

    return dx > dy ? dx : dy;
}

static inline int encounter_entity_footprints_overlap(
    int ax, int ay, int a_size,
    int bx, int by, int b_size
) {
    return !(ax + a_size <= bx || bx + b_size <= ax ||
             ay + a_size <= by || by + b_size <= ay);
}

/* check if player can attack: in range AND has LOS (if blockers present).
   returns 1 if ready to attack, 0 if blocked or out of range.
   encounters without LOS blockers pass NULL/0 for unconditional range check. */
static inline int encounter_player_can_attack(
    int player_x, int player_y,
    int target_x, int target_y, int target_size, int attack_range,
    const LOSBlocker* los_blockers, int los_blocker_count
) {
    int dist = encounter_entity_footprint_distance(player_x, player_y, 1,
                                                   target_x, target_y, target_size);
    if (dist < 1 || dist > attack_range) return 0;
    if (!los_blockers || los_blocker_count == 0) return 1;
    return entity_has_line_of_sight(los_blockers, los_blocker_count,
                                    player_x, player_y, 1,
                                    target_x, target_y, target_size,
                                    attack_range);
}

/* auto-walk toward attack target: handles out-of-range, blocked LOS, and under-NPC.
   the caller owns the policy; this helper only computes the chase step.
   los_blockers/los_blocker_count: LOS blocking entities (pillars). NULL/0 = no LOS check.
   returns 1 if player moved (chasing), 0 if ready to attack or stuck. */
static inline int encounter_chase_attack_target(
    Player* p, int target_x, int target_y, int target_size, int attack_range,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    const LOSBlocker* los_blockers, int los_blocker_count,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    int dist = encounter_entity_footprint_distance(p->x, p->y, 1,
                                                   target_x, target_y, target_size);

    /* player under NPC (dist=0): walk to nearest tile outside the target footprint. */
    if (dist == 0) {
        int max_r = (target_size + 1) / 2 + 1;
        int best_dsq = 9999, bx = -1, by = -1;
        for (int dy = -max_r; dy <= max_r; dy++) {
            for (int dx = -max_r; dx <= max_r; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = p->x + dx, ny = p->y + dy;
                if (!is_walkable(ctx, nx, ny)) continue;
                if (encounter_entity_footprints_overlap(nx, ny, 1,
                                                        target_x, target_y, target_size))
                    continue;
                int d = dx * dx + dy * dy;
                if (d < best_dsq) { best_dsq = d; bx = nx; by = ny; }
            }
        }
        if (bx < 0) return 0;
        int steps = 0;
        for (int step = 0; step < 2; step++) {
            if (p->x == bx && p->y == by) break;
            PathResult pr = (arena_w > 0)
                ? encounter_pathfind_arena(cmap, world_offset_x, world_offset_y,
                                           p->x, p->y, bx, by,
                                           extra_blocked, blocked_ctx,
                                           arena_base_x, arena_base_y, arena_w, arena_h)
                : encounter_pathfind(cmap, world_offset_x, world_offset_y,
                                      p->x, p->y, bx, by,
                                      extra_blocked, blocked_ctx);
            if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) break;
            int nx = p->x + pr.next_dx, ny = p->y + pr.next_dy;
            if (!is_walkable(ctx, nx, ny)) break;
            p->x = nx; p->y = ny;
            steps++;
        }
        p->is_running = (steps == 2);
        p->dest_x = p->x; p->dest_y = p->y;
        return steps > 0 ? 1 : 0;
    }

    /* in range + LOS: ready to attack, no movement needed */
    if (encounter_player_can_attack(p->x, p->y, target_x, target_y,
                                     target_size, attack_range,
                                     los_blockers, los_blocker_count))
        return 0;

    /* pathfind target selection: when in range but LOS blocked by a pillar,
       seek nearest melee-adjacent tile around the NPC that isn't inside a blocker.
       ref: InfernoTrainer Player.ts:469-504 seekingTiles.
       when out of range, path toward closest NPC tile (standard OSRS behavior).
       the per-step can_attack check (below) stops the player as soon as LOS + range. */
    int cx, cy;
    int dist_now = encounter_entity_footprint_distance(p->x, p->y, 1,
                                                       target_x, target_y, target_size);
    if (dist_now > 0 && dist_now <= attack_range &&
        los_blockers && los_blocker_count > 0) {
        /* in range but no LOS — scan NPC-adjacent tiles in the same row-first
           order as osrs-sdk Player.ts seekingTiles. the destination set is
           pathability-filtered only; LOS is checked after each movement step. */
        int best_dsq = 999999;
        cx = -1; cy = -1;
        for (int xx = 0; xx < target_size; xx++) {
            int px = target_x + xx;
            int north_py = target_y + target_size;
            int south_py = target_y - 1;

            if (is_walkable(ctx, px, north_py)) {
                int ddx = px - p->x, ddy = north_py - p->y;
                int dsq = ddx * ddx + ddy * ddy;
                if (dsq < best_dsq) { best_dsq = dsq; cx = px; cy = north_py; }
            }

            if (is_walkable(ctx, px, south_py)) {
                int ddx = px - p->x, ddy = south_py - p->y;
                int dsq = ddx * ddx + ddy * ddy;
                if (dsq < best_dsq) { best_dsq = dsq; cx = px; cy = south_py; }
            }
        }
        for (int yy = 0; yy < target_size; yy++) {
            int py = target_y + yy;
            int west_px = target_x - 1;
            int east_px = target_x + target_size;

            if (is_walkable(ctx, west_px, py)) {
                int ddx = west_px - p->x, ddy = py - p->y;
                int dsq = ddx * ddx + ddy * ddy;
                if (dsq < best_dsq) { best_dsq = dsq; cx = west_px; cy = py; }
            }

            if (is_walkable(ctx, east_px, py)) {
                int ddx = east_px - p->x, ddy = py - p->y;
                int dsq = ddx * ddx + ddy * ddy;
                if (dsq < best_dsq) { best_dsq = dsq; cx = east_px; cy = py; }
            }
        }
        /* fallback: no unblocked adjacent tile, path toward closest NPC tile */
        if (cx < 0) {
            cx = p->x < target_x ? target_x :
                 (p->x > target_x + target_size - 1 ? target_x + target_size - 1 : p->x);
            cy = p->y < target_y ? target_y :
                 (p->y > target_y + target_size - 1 ? target_y + target_size - 1 : p->y);
        }
    } else {
        /* out of range: path toward closest NPC tile */
        cx = p->x < target_x ? target_x :
             (p->x > target_x + target_size - 1 ? target_x + target_size - 1 : p->x);
        cy = p->y < target_y ? target_y :
             (p->y > target_y + target_size - 1 ? target_y + target_size - 1 : p->y);
    }

    int steps = 0;
    for (int step = 0; step < 2; step++) {
        if (encounter_player_can_attack(p->x, p->y, target_x, target_y,
                                         target_size, attack_range,
                                         los_blockers, los_blocker_count))
            break;
        PathResult pr = (arena_w > 0)
            ? encounter_pathfind_arena(cmap, world_offset_x, world_offset_y,
                                       p->x, p->y, cx, cy,
                                       extra_blocked, blocked_ctx,
                                       arena_base_x, arena_base_y, arena_w, arena_h)
            : encounter_pathfind(cmap, world_offset_x, world_offset_y,
                                  p->x, p->y, cx, cy,
                                  extra_blocked, blocked_ctx);
        if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) break;
        int nx = p->x + pr.next_dx, ny = p->y + pr.next_dy;
        if (!is_walkable(ctx, nx, ny)) break;
        p->x = nx; p->y = ny;
        steps++;
    }
    p->is_running = (steps == 2);
    p->dest_x = p->x; p->dest_y = p->y;
    return steps > 0 ? 1 : 0;
}

/* ======================================================================== */
/* shared NPC step-out-from-under (OSRS: NPC shuffles off player tile)       */
/* ======================================================================== */

typedef int (*encounter_npc_blocked_fn)(void* ctx, int x, int y, int size);
typedef int (*encounter_npc_overlap_hold_fn)(void* ctx);

#define ENCOUNTER_NPC_UNDER_PLAYER_NONE  0
#define ENCOUNTER_NPC_UNDER_PLAYER_MOVED 1
#define ENCOUNTER_NPC_UNDER_PLAYER_HELD  2

/* when an NPC overlaps the player (AABB overlap), it shuffles one tile in a
   random cardinal direction. matches osrs-sdk Mob.ts:109-153 behavior:
   50% pick X-axis vs Y-axis, then 50% +1 or -1 on that axis.
   hold_overlap lets the caller preserve the one-tick "player just clicked this
   mob, so it cannot move off" rule. returns MOVED, HELD, or NONE. */
static inline int encounter_npc_step_out_from_under(
    int* npc_x, int* npc_y, int npc_size,
    int player_x, int player_y,
    encounter_npc_blocked_fn is_blocked, void* ctx,
    encounter_npc_overlap_hold_fn hold_overlap,
    uint32_t* rng
) {
    /* AABB overlap check (handles multi-tile NPCs) */
    int overlap = !(*npc_x >= player_x + 1 || *npc_x + npc_size <= player_x ||
                    *npc_y >= player_y + 1 || *npc_y + npc_size <= player_y);
    if (!overlap) return ENCOUNTER_NPC_UNDER_PLAYER_NONE;
    if (hold_overlap && hold_overlap(ctx)) return ENCOUNTER_NPC_UNDER_PLAYER_HELD;

    /* 4 cardinal directions: +x, -x, +y, -y */
    int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};

    /* random start: 50% X-axis first (dirs 0,1) vs Y-axis first (dirs 2,3),
       then 50% positive vs negative on that axis */
    int axis = encounter_rand_int(rng, 2);       /* 0=X, 1=Y */
    int sign = encounter_rand_int(rng, 2);        /* 0=positive, 1=negative */
    int order[4];
    order[0] = axis * 2 + sign;         /* primary: chosen axis+sign */
    order[1] = axis * 2 + (1 - sign);   /* secondary: chosen axis, other sign */
    order[2] = (1 - axis) * 2 + sign;   /* tertiary: other axis, same sign */
    order[3] = (1 - axis) * 2 + (1 - sign); /* last: other axis, other sign */

    for (int i = 0; i < 4; i++) {
        int nx = *npc_x + dirs[order[i]][0];
        int ny = *npc_y + dirs[order[i]][1];
        /* InfernoTrainer Mob.ts:128-142: 1-tile shuffle per tick, validated
           via normal edge-tile movement system. for size>1 NPCs, full escape
           takes multiple ticks. anchor walkability matches InfernoTrainer's
           canTileBePathedTo check on the leading edge. */
        if (!is_blocked(ctx, nx, ny, npc_size)) {
            *npc_x = nx;
            *npc_y = ny;
            return ENCOUNTER_NPC_UNDER_PLAYER_MOVED;
        }
    }
    return ENCOUNTER_NPC_UNDER_PLAYER_NONE;
}

/* ======================================================================== */
/* shared NPC greedy pathfinding                                             */
/* ======================================================================== */

/** check if the leading edge tiles are clear for an NPC moving in direction (dx, dy).
    for size>1 NPCs, OSRS checks the tiles along the leading edge that the NPC
    sweeps through — not the full destination footprint. for diagonal moves, each
    edge strip extends by 1 tile to cover the corner.
    ref: InfernoTrainer Mob.ts:229-270 getXMovementTiles/getYMovementTiles.
    is_blocked is called with size=1 for each individual edge tile. */
static inline int encounter_npc_x_edge_clear(
    int x, int y, int size, int dx, int dy,
    encounter_npc_blocked_fn is_blocked, void* ctx
) {
    if (dx == 0) return 1;
    int ex = (dx == 1) ? x + size : x - 1;
    int y_start = (dy == -1) ? y - 1 : y;
    int y_end = (dy == 1) ? y + size : y + size - 1;
    for (int ey = y_start; ey <= y_end; ey++)
        if (is_blocked(ctx, ex, ey, 1)) return 0;
    return 1;
}

static inline int encounter_npc_y_edge_clear(
    int x, int y, int size, int dx, int dy,
    encounter_npc_blocked_fn is_blocked, void* ctx
) {
    if (dy == 0) return 1;
    int ey = (dy == 1) ? y + size : y - 1;
    int x_start = (dx == -1) ? x - 1 : x;
    int x_end = (dx == 1) ? x + size : x + size - 1;
    for (int ex = x_start; ex <= x_end; ex++)
        if (is_blocked(ctx, ex, ey, 1)) return 0;
    return 1;
}

static inline int encounter_npc_axis_gap(int a, int a_size, int b, int b_size) {
    int a_max = a + a_size - 1;
    int b_max = b + b_size - 1;
    if (a_max < b) return b - a_max;
    if (b_max < a) return a - b_max;
    return 0;
}

static inline int encounter_npc_axis_dir(int a, int a_size, int b, int b_size) {
    int a_max = a + a_size - 1;
    int b_max = b + b_size - 1;
    if (a_max < b) return 1;
    if (b_max < a) return -1;
    return 0;
}

static inline int encounter_npc_try_step(
    int* x, int* y, int size, int dx, int dy,
    encounter_npc_blocked_fn is_blocked, void* ctx
) {
    if (dx == 0 && dy == 0) return 0;
    if (size <= 1) {
        if (!is_blocked(ctx, *x + dx, *y + dy, 1)) {
            *x += dx;
            *y += dy;
            return 1;
        }
        return 0;
    }

    int x_clear = encounter_npc_x_edge_clear(*x, *y, size, dx, dy, is_blocked, ctx);
    int y_clear = encounter_npc_y_edge_clear(*x, *y, size, dx, dy, is_blocked, ctx);
    if (x_clear && y_clear) {
        *x += dx;
        *y += dy;
        return 1;
    }
    return 0;
}

/** OSRS-shaped NPC step toward target. tries diagonal first, then x-only,
    then y-only when RuneLite's travel rule allows the y fallback.

    for size>1 NPCs, validates movement by checking EDGE TILES the NPC sweeps
    through, not just the destination footprint. for diagonal moves, both the
    x-edge and y-edge must be clear (each extended by 1 tile for the corner).
    ref: InfernoTrainer Mob.ts:160-270 movementStep + getX/YMovementTiles.

    stop_at_melee_distance matches RuneLite WorldArea.calculateNextTravellingPoint:
    overlap returns no normal step, cardinal melee contact returns no step,
    and diagonal contact tries x-only.

    returns 1 if moved, 0 if blocked or already at target. */
static inline int encounter_npc_step_toward(
    int* x, int* y, int tx, int ty, int npc_size,
    int target_size, int stop_at_melee_distance,
    encounter_npc_blocked_fn is_blocked, void* ctx
) {
    int size = npc_size;
    int x_gap = encounter_npc_axis_gap(*x, size, tx, target_size);
    int y_gap = encounter_npc_axis_gap(*y, size, ty, target_size);
    int dx = encounter_npc_axis_dir(*x, size, tx, target_size);
    int dy = encounter_npc_axis_dir(*y, size, ty, target_size);

    if (stop_at_melee_distance && x_gap == 0 && y_gap == 0) return 0;
    if (stop_at_melee_distance && x_gap + y_gap == 1) return 0;
    if (dx == 0 && dy == 0) return 0;

    if (stop_at_melee_distance && x_gap == 1 && y_gap == 1) {
        return encounter_npc_try_step(x, y, size, dx, 0, is_blocked, ctx);
    }

    if (dx != 0 && dy != 0 &&
        encounter_npc_try_step(x, y, size, dx, dy, is_blocked, ctx))
        return 1;
    if (dx != 0 && encounter_npc_try_step(x, y, size, dx, 0, is_blocked, ctx))
        return 1;
    int max_gap = x_gap > y_gap ? x_gap : y_gap;
    if (dy != 0 && max_gap > 1 &&
        encounter_npc_try_step(x, y, size, 0, dy, is_blocked, ctx))
        return 1;
    return 0;
}

/* ======================================================================== */
/* shared damage application helpers                                         */
/*                                                                           */
/* ENCOUNTERS: use these instead of manually subtracting HP, clamping,       */
/* and setting hit splat flags. prevents bugs from forgetting a step.        */
/* ======================================================================== */

/** apply damage to a player. updates HP (clamped to 0), sets hit splat flags,
    and accumulates damage into a per-tick tracker (for reward calculation).
    damage_tracker can be NULL if not needed.
    always sets hit_landed_this_tick so the renderer shows a splat —
    0 damage produces a blue "miss" splat (standard OSRS behavior). */
static inline void encounter_damage_player(
    Player* p, int damage, float* damage_tracker
) {
    if (damage > 0) {
        p->current_hitpoints -= damage;
        if (p->current_hitpoints < 0) p->current_hitpoints = 0;
        if (damage_tracker) *damage_tracker += (float)damage;
    }
    p->hit_landed_this_tick = 1;
    p->hit_damage = damage > 0 ? damage : 0;
}

/** apply damage to an NPC-like entity via raw field pointers.
    works with any struct that has hp/hit_landed/hit_damage int fields.
    always sets hit_landed so the renderer shows a splat —
    0 damage produces a blue "miss" splat (standard OSRS behavior). */
static inline void encounter_damage_npc(
    int* hp, int* hit_landed, int* hit_damage, int damage
) {
    if (damage > 0) {
        *hp -= damage;
    }
    *hit_landed = 1;
    *hit_damage = damage > 0 ? damage : 0;
}

/* ======================================================================== */
/* shared NPC pending hit resolution (barrage freeze + blood heal)           */
/* ======================================================================== */

/** resolve a single NPC's pending hit. tick down, apply damage when it lands.
    ice barrage: sets *frozen_ticks = BARRAGE_FREEZE_TICKS on hit.
    blood barrage: accumulates landed damage into *blood_heal_acc for 25% heal.
    returns 1 if hit landed this call, 0 otherwise. */
static inline int encounter_resolve_npc_pending_hit(
    EncounterPendingHit* ph,
    int* npc_hp, int* hit_landed, int* hit_damage,
    int* frozen_ticks, int* blood_heal_acc, float* damage_dealt_acc
) {
    (void)frozen_ticks;  /* freeze applied at cast time, not land time */
    if (!ph->active) return 0;
    ph->ticks_remaining--;
    if (ph->ticks_remaining > 0) return 0;

    /* hit landed */
    int dmg = ph->damage;
    encounter_damage_npc(npc_hp, hit_landed, hit_damage, dmg);
    if (damage_dealt_acc) *damage_dealt_acc += dmg;

    /* blood barrage: accumulate damage for 25% heal (at land time — heal depends on damage) */
    if (ph->spell_type == ENCOUNTER_SPELL_BLOOD && blood_heal_acc)
        *blood_heal_acc += dmg;

    ph->active = 0;
    return 1;
}

/** resolve player pending hits (NPC attacks landing on the player).
    ticks down each hit, applies damage when it lands, handles deferred
    prayer checks (jad-style: check_prayer=1 re-checks at land time).
    encounters MUST call this each tick for projectile-based NPC attacks.
    prayer_correct_count: incremented for each deferred prayer check that succeeds.
    multiple hits can land on the same tick (e.g. mager + ranger). */
static inline void encounter_resolve_player_pending_hits(
    EncounterPendingHit* hits, int* hit_count,
    Player* player, OverheadPrayer active_prayer,
    float* damage_received_acc, int* prayer_correct_count,
    int* off_prayer_hit_count
) {
    for (int i = 0; i < *hit_count; i++) {
        /* deferred prayer check (jad): lock in damage at T + prayer_check_delay.
           runs BEFORE ticks_remaining decrement so the check happens on the exact
           tick prayer_check_delay reaches 0, regardless of whether the hit lands
           same tick or later. after the check, damage is frozen (possibly 0)
           and further flicks don't affect this hit. */
        if (hits[i].check_prayer && hits[i].prayer_check_delay > 0) {
            hits[i].prayer_check_delay--;
            if (hits[i].prayer_check_delay == 0) {
                if (encounter_prayer_correct_for_style(active_prayer, hits[i].attack_style)) {
                    hits[i].damage = 0;
                    if (prayer_correct_count) (*prayer_correct_count)++;
                } else if (hits[i].damage > 0 && hits[i].attack_style != ATTACK_STYLE_NONE) {
                    if (off_prayer_hit_count) (*off_prayer_hit_count)++;
                }
                hits[i].check_prayer = 0;
            }
        }
        hits[i].ticks_remaining--;
        if (hits[i].ticks_remaining <= 0) {
            int dmg = hits[i].damage;
            if (hits[i].check_prayer) {
                /* legacy path: delay was 0 and check_prayer never decremented.
                   happens if encounter sets check_prayer=1 with no delay — check now. */
                if (encounter_prayer_correct_for_style(active_prayer, hits[i].attack_style)) {
                    dmg = 0;
                    if (prayer_correct_count) (*prayer_correct_count)++;
                } else if (dmg > 0 && hits[i].attack_style != ATTACK_STYLE_NONE) {
                    if (off_prayer_hit_count) (*off_prayer_hit_count)++;
                }
            } else if (dmg > 0 && hits[i].attack_style != ATTACK_STYLE_NONE) {
                /* Even if check_prayer was done at launch (e.g. non-Jad mobs), 
                   it only zeroed dmg if correctly prayed AT LAUNCH.
                   Wait, actually, for non-Jad, check_prayer is 0 and dmg is already computed.
                   If dmg > 0, it means we took a hit. Was it an off-prayer hit? 
                   Yes, because if we prayed correctly at launch, dmg would be 0.
                   BUT we don't know if we just missed the pray or if it was typeless.
                   We assume ATTACK_STYLE_NONE is typeless. */
                if (off_prayer_hit_count) (*off_prayer_hit_count)++;
            }
            
            encounter_damage_player(player, dmg, damage_received_acc);
            hits[i] = hits[--(*hit_count)];
            i--;
        }
    }
}

/* ======================================================================== */
/* shared per-tick flag clearing for encounters                              */
/* ======================================================================== */

/** clear all per-tick animation/event flags on a player.
    call at the start of each encounter tick, then set flags as events happen.
    the renderer reads these once per frame via RenderEntity. */
static inline void encounter_clear_tick_flags(Player* p) {
    p->attack_style_this_tick = ATTACK_STYLE_NONE;
    p->magic_type_this_tick = 0;
    p->hit_landed_this_tick = 0;
    p->hit_damage = 0;
    p->hit_was_successful = 0;
    p->cast_veng_this_tick = 0;
    p->ate_food_this_tick = 0;
    p->ate_karambwan_this_tick = 0;
    p->used_special_this_tick = 0;
}

/* ======================================================================== */
/* shared reset helpers                                                      */
/* ======================================================================== */

/** resolve RNG seed for encounter reset. priority: explicit seed > saved state > default.
    all encounters MUST use this to ensure consistent RNG initialization. */
static inline uint32_t encounter_resolve_seed(uint32_t saved_rng, uint32_t explicit_seed) {
    uint32_t rng = 12345;
    if (saved_rng != 0) rng = saved_rng;
    if (explicit_seed != 0) rng = explicit_seed;
    return rng;
}

/* ======================================================================== */
/* shared prayer drain                                                       */
/*                                                                           */
/* ENCOUNTERS: call encounter_drain_all_prayers() each tick to drain prayer  */
/* points at the correct OSRS rate. all encounters with overhead prayers     */
/* MUST use this — do not hand-roll prayer drain logic.                      */
/*                                                                           */
/* OSRS drain formula: each prayer has a "drain effect" value.               */
/* drain rate = 1 point per floor((18 + floor(bonus/4)) / drain_effect)      */
/* seconds. the drain counter increments each tick; when it reaches the      */
/* threshold, 1 prayer point is drained and the counter resets.              */
/*                                                                           */
/* protection prayers (melee/ranged/magic): drain_effect = 12               */
/* rigour: drain_effect = 24, augury: drain_effect = 24                     */
/* ======================================================================== */

/** drain effect values for overhead prayers.
    from the OSRS prayer table — higher values drain faster.
    used by both PvE encounters and PvP. */
static inline int encounter_overhead_drain_effect(OverheadPrayer prayer) {
    switch (prayer) {
        case PRAYER_PROTECT_MELEE:  return 12;
        case PRAYER_PROTECT_RANGED: return 12;
        case PRAYER_PROTECT_MAGIC:  return 12;
        case PRAYER_SMITE:          return 12;
        case PRAYER_REDEMPTION:     return 6;
        default: return 0;
    }
}

/** drain effect values for offensive prayers. ref: osrs wiki prayer table. */
static inline int encounter_offensive_drain_effect(OffensivePrayer prayer) {
    switch (prayer) {
        case OFFENSIVE_PRAYER_PIETY:       return 24;
        case OFFENSIVE_PRAYER_RIGOUR:      return 24;
        case OFFENSIVE_PRAYER_AUGURY:      return 24;
        case OFFENSIVE_PRAYER_MELEE_LOW:   return 6;
        case OFFENSIVE_PRAYER_RANGED_LOW:  return 6;
        case OFFENSIVE_PRAYER_MAGIC_LOW:   return 6;
        default: return 0;
    }
}

/** drain both overhead and offensive prayer for one tick.
    handles activation-tick skip (prayers activated this tick do not drain),
    the shared drain counter, and pp=0 auto-clear (all prayers off when empty).
    prayer_bonus: player's total prayer equipment bonus (typically 0-30).
    ref: osrs-sdk PrayerController.ts:44-59, wiki "Prayer flicking" section. */
static inline void encounter_drain_all_prayers(Player* p, int prayer_bonus) {
    /* active-but-not-just-activated prayers contribute to drain this tick.
       ref: wiki: "the game does not drain prayer for prayers on the tick
       they are activated". this is what makes 1-tick flicking free. */
    int overhead_effect  = p->prayer_just_activated
        ? 0 : encounter_overhead_drain_effect(p->prayer);
    int offensive_effect = p->offensive_prayer_just_activated
        ? 0 : encounter_offensive_drain_effect(p->offensive_prayer);
    int total = overhead_effect + offensive_effect;

    /* clear just-activated flags even on zero-drain paths so they don't leak
       into next tick. */
    p->prayer_just_activated = 0;
    p->offensive_prayer_just_activated = 0;

    /* pp already at/below 0 entering this tick — force prayers off and skip
       drain math. covers: external drain (smite), or activation attempted
       at pp=0 (the pretick apply_*_action helpers don't gate on pp, so the
       enum may have been set this tick before we arrived). without this,
       the prayer enum can stay active indefinitely at pp=0. */
    if (p->current_prayer <= 0) {
        p->current_prayer = 0;
        p->prayer_drain_counter = 0;
        p->prayer = PRAYER_NONE;
        p->offensive_prayer = OFFENSIVE_PRAYER_NONE;
        return;
    }
    if (total <= 0) return;

    int drain_resistance = 60 + prayer_bonus * 2;
    p->prayer_drain_counter += total;
    while (p->prayer_drain_counter > drain_resistance) {
        p->current_prayer--;
        p->prayer_drain_counter -= drain_resistance;
        if (p->current_prayer <= 0) {
            p->current_prayer = 0;
            p->prayer_drain_counter = 0;
            p->prayer = PRAYER_NONE;
            p->offensive_prayer = OFFENSIVE_PRAYER_NONE;
            break;
        }
    }
}


/* ======================================================================== */
/* shared loadout stat computation                                           */
/*                                                                           */
/* ENCOUNTERS: do NOT manually compute attack bonuses, max hits, or          */
/* effective levels. call encounter_compute_loadout_stats() with a loadout   */
/* array and it derives everything from ITEM_DATABASE automatically.         */
/*                                                                           */
/* available structs/functions:                                               */
/*   EncounterLoadoutStats — computed combat stats for one gear loadout      */
/*   OffensivePrayer       — prayer enum (NONE, PIETY, RIGOUR, AUGURY, low-tiers) */
/*   encounter_compute_loadout_stats() — derive stats from loadout + prayer  */
/* ======================================================================== */

/** combat stats derived from a gear loadout + prayer + style.
    computed once at reset, read during combat.
    prayer multipliers and style_bonus are stored for dynamic recomputation
    when stats change (brew drain, potion boost). */
typedef struct {
    int attack_bonus;     /* primary attack bonus for the style */
    int strength_bonus;   /* ranged_strength, magic_damage %, or melee_strength */
    int eff_level;        /* effective attack level (floor(base*prayer) + style + 8) */
    int max_hit;          /* base max hit (before tbow/set bonuses) */
    int attack_speed;     /* ticks between attacks (includes stance speed mod) */
    int attack_range;     /* max chebyshev distance (includes stance range mod) */
    AttackStyle style;
    FightStyle fight_style;  /* stance picked for this loadout — drives stance bonuses + speed/range mods */
    /* defence bonuses from gear */
    int def_stab, def_slash, def_crush, def_magic, def_ranged;
    /* stored for dynamic max hit recomputation after brew drain / potion boost */
    float att_prayer_mult;
    float str_prayer_mult;
    int spell_base_damage;
} EncounterLoadoutStats;

/** offensive prayer multipliers for effective level computation.
    single source of truth: the multipliers used in encounter_compute_loadout_stats()
    and encounter_update_loadout_level(). also used by PvP combat math (via
    osrs_pvp_combat.h) so all combat paths agree on prayer effects.
    ref: osrs wiki prayer table. */
static inline void encounter_offensive_prayer_mults(
    OffensivePrayer op, float* att_out, float* str_out
) {
    float att = 1.0f, str = 1.0f;
    switch (op) {
        case OFFENSIVE_PRAYER_PIETY:       att = 1.20f; str = 1.23f; break;
        case OFFENSIVE_PRAYER_RIGOUR:      att = 1.20f; str = 1.23f; break;
        case OFFENSIVE_PRAYER_AUGURY:      att = 1.25f; str = 1.00f; break;
        case OFFENSIVE_PRAYER_MELEE_LOW:   att = 1.15f; str = 1.15f; break;
        case OFFENSIVE_PRAYER_RANGED_LOW:  att = 1.15f; str = 1.15f; break;
        case OFFENSIVE_PRAYER_MAGIC_LOW:   att = 1.15f; str = 1.00f; break;
        default: break;
    }
    *att_out = att;
    *str_out = str;
}

/** augury adds +4% magic damage on top of its accuracy mult (PvP parity). */
static inline float encounter_offensive_magic_dmg_mult(OffensivePrayer op) {
    return (op == OFFENSIVE_PRAYER_AUGURY) ? 1.04f : 1.0f;
}

/** derive all combat stats from a loadout array + prayer + fight stance.
    sums equipment bonuses from ITEM_DATABASE, applies prayer multiplier,
    computes effective level and max hit. attack_speed and attack_range are
    also adjusted for the stance (rapid -1 tick, longrange +2 tiles).

    @param loadout          gear array indexed by GEAR_SLOT_* (ITEM_NONE=255 for empty)
    @param style            ATTACK_STYLE_MAGIC, ATTACK_STYLE_RANGED, or ATTACK_STYLE_MELEE
    @param offensive_prayer current offensive prayer (piety/rigour/augury/none/low tiers)
    @param base_level       base combat level (usually 99)
    @param fight_style      stance — drives attack/str/def bonuses, attack speed, range
    @param spell_base_damage 0 for ranged/melee, 30 for ice/blood barrage
    @param out              output struct to fill. prayer multipliers are stored so
                            encounter_update_loadout_level() can recompute eff/max without
                            needing the prayer arg again (callers must re-call update
                            whenever offensive prayer changes). */
static inline void encounter_compute_loadout_stats(
    const uint8_t loadout[NUM_GEAR_SLOTS],
    AttackStyle style,
    OffensivePrayer offensive_prayer,
    int base_level,
    FightStyle fight_style,
    int spell_base_damage,
    EncounterLoadoutStats* out
) {
    memset(out, 0, sizeof(*out));
    out->style = style;
    out->fight_style = fight_style;

    /* sum equipment bonuses using shared function */
    EquipmentBonuses eb;
    osrs_sum_equipment_bonuses(loadout, &eb);

    out->def_stab = eb.defence_stab;
    out->def_slash = eb.defence_slash;
    out->def_crush = eb.defence_crush;
    out->def_magic = eb.defence_magic;
    out->def_ranged = eb.defence_ranged;
    /* apply stance modifiers to weapon base speed/range. equipment.json stores
       the base (accurate/longrange speed, non-longrange range). rapid and
       longrange shift them. */
    out->attack_speed = eb.attack_speed + osrs_stance_speed_mod(fight_style);
    out->attack_range = eb.attack_range + osrs_stance_range_mod(fight_style);

    /* primary attack bonus based on style */
    if (style == ATTACK_STYLE_MAGIC) {
        out->attack_bonus = eb.attack_magic;
    } else if (style == ATTACK_STYLE_RANGED) {
        out->attack_bonus = eb.attack_ranged;
    } else {
        /* melee: best of stab/slash/crush */
        out->attack_bonus = eb.attack_stab;
        if (eb.attack_slash > out->attack_bonus) out->attack_bonus = eb.attack_slash;
        if (eb.attack_crush > out->attack_bonus) out->attack_bonus = eb.attack_crush;
    }

    /* prayer multipliers — single source of truth in encounter_offensive_prayer_mults(). */
    float att_prayer_mult, str_prayer_mult;
    encounter_offensive_prayer_mults(offensive_prayer, &att_prayer_mult, &str_prayer_mult);

    /* store for dynamic recomputation after brew drain / potion boost / prayer toggle */
    out->att_prayer_mult = att_prayer_mult;
    out->str_prayer_mult = str_prayer_mult;
    out->spell_base_damage = spell_base_damage;

    int att_stance_bonus = osrs_stance_att_bonus(fight_style, style);
    int str_stance_bonus = osrs_stance_str_bonus(fight_style);

    /* effective attack level: floor(base * prayer_mult) + stance_att_bonus + 8.
       magic uses +9 (OSRS invisible +1 boost) instead of +8. powered-staff stance
       bonus (accurate +3, longrange +1) is picked up via osrs_stance_att_bonus(MAGIC). */
    if (style == ATTACK_STYLE_MAGIC) {
        out->eff_level = (int)(base_level * att_prayer_mult) + att_stance_bonus + 9;
    } else {
        out->eff_level = (int)(base_level * att_prayer_mult) + att_stance_bonus + 8;
    }

    /* effective strength level (for max hit): floor(base * str_prayer_mult) + str_stance_bonus + 8.
       str_stance_bonus is non-zero only for melee (aggressive/controlled). */
    int eff_str_level = (int)(base_level * str_prayer_mult) + str_stance_bonus + 8;

    /* augury magic damage multiplier: +4% (matches PvP calculate_max_hit). */
    float magic_dmg_prayer_mult = encounter_offensive_magic_dmg_mult(offensive_prayer);

    /* max hit and strength bonus depend on combat style */
    if (style == ATTACK_STYLE_RANGED) {
        out->strength_bonus = eb.ranged_strength;
        out->max_hit = (int)(0.5 + eff_str_level * (eb.ranged_strength + 64) / 640.0);
    } else if (style == ATTACK_STYLE_MAGIC) {
        out->strength_bonus = eb.magic_damage;
        out->max_hit = (int)(spell_base_damage * (1.0 + eb.magic_damage / 100.0) * magic_dmg_prayer_mult);
    } else {
        out->strength_bonus = eb.melee_strength;
        out->max_hit = (int)(0.5 + eff_str_level * (eb.melee_strength + 64) / 640.0);
    }
}

/* ======================================================================== */
/* dynamic max hit recomputation (after brew drain / potion boost)            */
/*                                                                           */
/* ENCOUNTERS: call encounter_update_loadout_level() whenever the player's   */
/* current combat level changes (brew drain, restore, bastion boost).        */
/* this recomputes eff_level and max_hit using the stored prayer multiplier  */
/* and strength bonus from the initial encounter_compute_loadout_stats().    */
/* ======================================================================== */

/** recompute eff_level and max_hit for a loadout using a (possibly drained/boosted)
    current combat level AND current offensive prayer. call whenever either changes:
      - offensive prayer toggle (pretick action)
      - brew drain / super restore / bastion boost (consumable effects)
    current_att_level: the player's current attack/ranged/magic level (for accuracy).
    current_str_level: the player's current strength/ranged/magic level (for max hit).
    for ranged: both are current_ranged. for melee: att=current_attack, str=current_strength.
    for magic: max hit doesn't depend on level (spell base damage), but eff_level does.
    offensive_prayer is the current Player.offensive_prayer — mults are rewritten from it. */
static inline void encounter_update_loadout_level(
    EncounterLoadoutStats* ls, OffensivePrayer offensive_prayer,
    int current_att_level, int current_str_level
) {
    float att_prayer_mult, str_prayer_mult;
    encounter_offensive_prayer_mults(offensive_prayer, &att_prayer_mult, &str_prayer_mult);
    ls->att_prayer_mult = att_prayer_mult;
    ls->str_prayer_mult = str_prayer_mult;

    int att_stance_bonus = osrs_stance_att_bonus(ls->fight_style, ls->style);
    int str_stance_bonus = osrs_stance_str_bonus(ls->fight_style);
    /* magic uses +9 invisible boost (matches encounter_compute_loadout_stats) */
    if (ls->style == ATTACK_STYLE_MAGIC) {
        ls->eff_level = (int)(current_att_level * att_prayer_mult) + att_stance_bonus + 9;
        float magic_dmg_mult = encounter_offensive_magic_dmg_mult(offensive_prayer);
        ls->max_hit = (int)(ls->spell_base_damage * (1.0 + ls->strength_bonus / 100.0) * magic_dmg_mult);
    } else {
        ls->eff_level = (int)(current_att_level * att_prayer_mult) + att_stance_bonus + 8;
        int eff_str = (int)(current_str_level * str_prayer_mult) + str_stance_bonus + 8;
        ls->max_hit = (int)(0.5 + eff_str * (ls->strength_bonus + 64) / 640.0);
    }
}

static inline void encounter_compute_player_equipped_stats(
    Player* p,
    AttackStyle style,
    FightStyle fight_style,
    int spell_base_damage,
    EncounterLoadoutStats* out
) {
    int current_att = p->current_attack;
    int current_str = p->current_strength;
    if (style == ATTACK_STYLE_RANGED) {
        current_att = p->current_ranged;
        current_str = p->current_ranged;
    } else if (style == ATTACK_STYLE_MAGIC) {
        current_att = p->current_magic;
        current_str = p->current_magic;
    }
    encounter_compute_loadout_stats(
        p->equipped,
        style,
        p->offensive_prayer,
        current_att,
        fight_style,
        spell_base_damage,
        out);
    encounter_update_loadout_level(out, p->offensive_prayer, current_att, current_str);
}

/* ======================================================================== */
/* shared potion stat effects (brew drain, restore, bastion boost)           */
/*                                                                           */
/* ENCOUNTERS: call these when the player drinks a potion. they modify the   */
/* player's current combat levels and recompute max hit for affected loadouts.*/
/* these implement the real OSRS formulas for stat modification.             */
/*                                                                           */
/* sara brew:     heals HP, boosts def, drains att/str/ranged/magic          */
/* super restore: restores all drained stats toward base (caps at base)      */
/* bastion:       boosts ranged above base, boosts def                       */
/* ======================================================================== */

/** sara brew stat drain. call AFTER healing HP (which is encounter-specific).
    drains att/str/ranged/magic by floor(current/10)+2 each (uses CURRENT level).
    boosts defence by floor(current_def/5)+2, capped at base + max boost from base.
    floors at 0 for drained stats.
    ref: OSRS wiki Saradomin brew. */
static inline void encounter_brew_drain_stats(Player* p) {
    int att_drain = p->current_attack / 10 + 2;
    int str_drain = p->current_strength / 10 + 2;
    int rng_drain = p->current_ranged / 10 + 2;
    int mag_drain = p->current_magic / 10 + 2;
    int def_boost = p->current_defence / 5 + 2;

    p->current_attack -= att_drain;
    if (p->current_attack < 0) p->current_attack = 0;
    p->current_strength -= str_drain;
    if (p->current_strength < 0) p->current_strength = 0;
    p->current_ranged -= rng_drain;
    if (p->current_ranged < 0) p->current_ranged = 0;
    p->current_magic -= mag_drain;
    if (p->current_magic < 0) p->current_magic = 0;

    p->current_defence += def_boost;
    int def_cap = p->base_defence + (p->base_defence / 5 + 2);
    if (p->current_defence > def_cap) p->current_defence = def_cap;
}

/** super restore stat recovery. restores all combat stats toward base level.
    each dose restores floor(base * 0.25) + 8 per stat. caps at base level.
    ref: OSRS wiki Super restore. */
static inline void encounter_restore_stats(Player* p) {
    int restore = 8 + p->base_attack / 4;  /* same formula for all stats at 99 base */
    p->current_attack += restore;
    if (p->current_attack > p->base_attack) p->current_attack = p->base_attack;
    restore = 8 + p->base_strength / 4;
    p->current_strength += restore;
    if (p->current_strength > p->base_strength) p->current_strength = p->base_strength;
    restore = 8 + p->base_defence / 4;
    p->current_defence += restore;
    if (p->current_defence > p->base_defence) p->current_defence = p->base_defence;
    restore = 8 + p->base_ranged / 4;
    p->current_ranged += restore;
    if (p->current_ranged > p->base_ranged) p->current_ranged = p->base_ranged;
    restore = 8 + p->base_magic / 4;
    p->current_magic += restore;
    if (p->current_magic > p->base_magic) p->current_magic = p->base_magic;
}

/** bastion potion boost. boosts ranged by floor(base * 0.10) + 4. can exceed base.
    also boosts defence by floor(base * 0.15) + 5. can exceed base.
    ref: OSRS wiki Bastion potion. */
static inline void encounter_bastion_boost(Player* p) {
    int rng_boost = 4 + p->base_ranged / 10;
    int def_boost = 5 + p->base_defence * 15 / 100;
    p->current_ranged += rng_boost;
    int rng_cap = p->base_ranged + rng_boost;
    if (p->current_ranged > rng_cap) p->current_ranged = rng_cap;
    p->current_defence += def_boost;
    int def_cap = p->base_defence + def_boost;
    if (p->current_defence > def_cap) p->current_defence = def_cap;
}

/** recompute max hit for all loadouts after a stat change or prayer change.
    encounters should call this after brew_drain_stats, restore_stats, bastion_boost,
    or when Player.offensive_prayer toggles. ranged loadouts use current_ranged, magic
    uses current_magic, melee uses current_attack/current_strength. prayer multipliers
    are rewritten from p->offensive_prayer. */
static inline void encounter_recompute_loadout_max_hits(
    EncounterLoadoutStats* loadouts, int num_loadouts, Player* p
) {
    for (int i = 0; i < num_loadouts; i++) {
        EncounterLoadoutStats* ls = &loadouts[i];
        if (ls->style == ATTACK_STYLE_RANGED) {
            encounter_update_loadout_level(ls, p->offensive_prayer, p->current_ranged, p->current_ranged);
        } else if (ls->style == ATTACK_STYLE_MAGIC) {
            encounter_update_loadout_level(ls, p->offensive_prayer, p->current_magic, p->current_magic);
        } else {
            encounter_update_loadout_level(ls, p->offensive_prayer, p->current_attack, p->current_strength);
        }
    }
}

/* ======================================================================== */
/* shared special attack energy                                              */
/*                                                                           */
/* ENCOUNTERS: call encounter_tick_spec_regen() every game tick. call         */
/* encounter_use_spec() when the player activates a special attack.          */
/* OSRS: energy 0-100, starts at 100, regens +10 every 50 ticks (30s).      */
/* lightbearer halves regen interval to 25 ticks.                            */
/* ======================================================================== */

/** tick special attack energy regeneration from current equipped gear. */
static inline void encounter_tick_spec_regen(Player* p) {
    osrs_tick_special_regen(p);
}

/** attempt to use special attack energy. returns 1 if successful (enough energy),
    0 if not enough energy. drains on success. */
static inline int encounter_use_spec(Player* p, int cost) {
    if (p->special_energy < cost) return 0;
    p->special_energy -= cost;
    return 1;
}

/* ======================================================================== */
/* shared gear switching helpers for encounters                              */
/* ======================================================================== */

/** apply a full static loadout to player equipment and set gear state.
    used by Zulrah, Inferno, and future boss encounters with fixed loadouts. */
static inline void encounter_apply_loadout(
    Player* p, const uint8_t loadout[NUM_GEAR_SLOTS], GearSet gear_set
) {
    memcpy(p->equipped, loadout, NUM_GEAR_SLOTS);
    p->current_gear = gear_set;
    p->visible_gear = gear_set;
    osrs_refresh_player_equipment(p);
}

/** populate player inventory from multiple loadouts (deduped per slot).
    extra_items is an optional overlay array (e.g. justiciar for tank), NULL to skip.
    the GUI reads p->inventory[][] to display available gear switches. */
static void encounter_populate_inventory(
    Player* p,
    const uint8_t* const* loadouts, int num_loadouts,
    const uint8_t extra_items[NUM_GEAR_SLOTS]
) {
    memset(p->inventory, 255 /* ITEM_NONE */, sizeof(p->inventory));
    memset(p->num_items_in_slot, 0, sizeof(p->num_items_in_slot));

    for (int s = 0; s < NUM_GEAR_SLOTS; s++) {
        int n = 0;
        for (int l = 0; l < num_loadouts && n < MAX_ITEMS_PER_SLOT; l++) {
            uint8_t item = loadouts[l][s];
            if (item == 255 /* ITEM_NONE */) continue;
            int dup = 0;
            for (int j = 0; j < n; j++) { if (p->inventory[s][j] == item) { dup = 1; break; } }
            if (dup) continue;
            p->inventory[s][n++] = item;
        }
        if (extra_items && extra_items[s] != 255 /* ITEM_NONE */ && n < MAX_ITEMS_PER_SLOT) {
            int dup = 0;
            for (int j = 0; j < n; j++) { if (p->inventory[s][j] == extra_items[s]) { dup = 1; break; } }
            if (!dup) p->inventory[s][n++] = extra_items[s];
        }
        p->num_items_in_slot[s] = n;
    }
}

/* ======================================================================== */
/* shared human input translate helpers                                      */
/* ======================================================================== */

/** translate movement: convert absolute tile to 8-directional walk action.
    writes to actions[head_move]. head_move < 0 = skip. */
static inline void encounter_translate_movement(HumanInput* hi, int* actions,
                                                 int head_move,
                                                 void* (*get_entity)(void*, int),
                                                 void* state) {
    if (hi->pending_move_x < 0 || hi->pending_move_y < 0 || head_move < 0) return;
    Player* player = (Player*)get_entity(state, 0);
    if (!player) return;
    int dx = hi->pending_move_x - player->x;
    int dy = hi->pending_move_y - player->y;
    int sx = (dx > 0) ? 1 : (dx < 0) ? -1 : 0;
    int sy = (dy > 0) ? 1 : (dy < 0) ? -1 : 0;
    static const int DX8[9] = { 0, 0, 1, 1, 1, 0, -1, -1, -1 };
    static const int DY8[9] = { 0, 1, 1, 0, -1, -1, -1, 0, 1 };
    for (int m = 1; m < 9; m++) {
        if (DX8[m] == sx && DY8[m] == sy) {
            actions[head_move] = m;
            break;
        }
    }
}

/** translate overhead prayer: pending_prayer stores the new ENCOUNTER_OVERHEAD_*
    value directly (set by GUI click handlers). writes to actions[head_prayer].
    head_prayer < 0 = skip. */
static inline void encounter_translate_prayer(HumanInput* hi, int* actions, int head_prayer) {
    if (hi->pending_prayer < 0 || head_prayer < 0) return;
    actions[head_prayer] = hi->pending_prayer;
}

/** translate offensive prayer: pending_offensive_prayer stores the new
    ENCOUNTER_OFFENSIVE_* value directly. writes to actions[head_offensive].
    head_offensive < 0 = skip (encounter doesn't expose offensive as an action). */
static inline void encounter_translate_offensive_prayer(
    HumanInput* hi, int* actions, int head_offensive
) {
    if (hi->pending_offensive_prayer < 0 || head_offensive < 0) return;
    actions[head_offensive] = hi->pending_offensive_prayer;
}

/** translate NPC target: 0=none, 1+=NPC index.
    writes to actions[head_target]. head_target < 0 = skip. */
static inline void encounter_translate_target(HumanInput* hi, int* actions, int head_target) {
    if (hi->pending_target_idx < 0 || head_target < 0) return;
    actions[head_target] = hi->pending_target_idx + 1;
}

/* ======================================================================== */
/* encounter definition (vtable)                                             */
/* ======================================================================== */

typedef struct {
    const char* name;           /* "nh_pvp", "cerberus", "jad", etc. */

    /* observation/action space dimensions */
    int obs_size;               /* raw observation features (before mask) */
    int num_action_heads;
    const int* action_head_dims; /* array of per-head dimensions */
    int mask_size;              /* sum of action_head_dims */

    /* lifecycle: create/destroy encounter state */
    EncounterState* (*create)(void);
    void (*destroy)(EncounterState* state);

    /* episode lifecycle */
    void (*reset)(EncounterState* state, uint32_t seed);
    void (*step)(EncounterState* state, const int* actions);
    void (*step_human_commands)(EncounterState* state, struct HumanInput* hi);

    /* RL interface */
    void (*write_obs)(EncounterState* state, float* obs_out);
    void (*write_mask)(EncounterState* state, float* mask_out);
    float (*get_reward)(EncounterState* state);
    int (*is_terminal)(EncounterState* state);

    /* entity access for renderer (returns entity count, writes entity pointers).
       renderer uses this to draw all entities generically. */
    int (*get_entity_count)(EncounterState* state);
    void* (*get_entity)(EncounterState* state, int index);  /* returns Player* */

    /* render entity population: fills array of RenderEntity structs for the renderer.
       replaces get_entity casting for rendering. NULL = renderer falls back to get_entity. */
    void (*fill_render_entities)(EncounterState* state, RenderEntity* out, int max_entities, int* count);

    /* encounter-specific config (key-value put/get for binding kwargs) */
    void (*put_int)(EncounterState* state, const char* key, int value);
    void (*put_float)(EncounterState* state, const char* key, float value);
    void (*put_ptr)(EncounterState* state, const char* key, void* value);

    /* arena bounds for renderer (0 = use FIGHT_AREA_* defaults) */
    int arena_base_x, arena_base_y;
    int arena_width, arena_height;

    /* human mode input translation (per-encounter, NULL = no human mode).
       translates semantic HumanInput intents to encounter-specific action arrays.
       each encounter owns its own mapping since action head layouts differ. */
    void (*translate_human_input)(struct HumanInput* hi, int* actions, EncounterState* state);
    int (*is_human_targetable_npc_slot)(EncounterState* state, int npc_slot);

    /* action head indices used by shared translate helpers and renderer.
       set to -1 if the encounter doesn't have that action head. */
    int head_move;     /* movement (walk/run) */
    int head_prayer;   /* prayer switching */
    int head_target;   /* NPC target selection (index into NPC array) */

    /* render hooks (optional — NULL if not implemented).
       populates visual overlay data for the renderer. */
    void (*render_post_tick)(EncounterState* state, EncounterOverlay* overlay);

    /* logging (returns pointer to encounter's Log struct, or NULL) */
    void* (*get_log)(EncounterState* state);

    /* tick access */
    int (*get_tick)(EncounterState* state);
    int (*get_winner)(EncounterState* state);
} EncounterDef;

/* ======================================================================== */
/* encounter registry                                                        */
/* ======================================================================== */

#define MAX_ENCOUNTERS 32

typedef struct {
    const EncounterDef* defs[MAX_ENCOUNTERS];
    int count;
} EncounterRegistry;

/* WARNING: static in header — each TU gets its own copy. only works correctly
   when all encounter headers are included from a single compilation unit. */
static EncounterRegistry g_encounter_registry = { .count = 0 };

static inline void encounter_register(const EncounterDef* def) {
    if (g_encounter_registry.count < MAX_ENCOUNTERS) {
        g_encounter_registry.defs[g_encounter_registry.count++] = def;
    }
}

static inline const EncounterDef* encounter_find(const char* name) {
    for (int i = 0; i < g_encounter_registry.count; i++) {
        if (strcmp(g_encounter_registry.defs[i]->name, name) == 0) {
            return g_encounter_registry.defs[i];
        }
    }
    return NULL;
}

#endif /* OSRS_ENCOUNTER_H */
