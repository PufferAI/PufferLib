/**
 * @fileoverview Visual effect system for spell impacts and projectiles.
 *
 * Manages animated spotanim effects (ice barrage splash, blood barrage) and
 * traveling projectiles (crossbow bolts, ice barrage orb). Each effect has a
 * model, animation, position, and lifetime. Projectiles follow parabolic arcs
 * matching OSRS SceneProjectile.java trajectory math.
 *
 * Effects are spawned from game state in render_post_tick and drawn as 3D
 * models in the render pipeline. Animation advances at 50 Hz client ticks.
 */

#ifndef OSRS_PVP_EFFECTS_H
#define OSRS_PVP_EFFECTS_H

#include "osrs_models.h"
#include "osrs_anim.h"
#include <math.h>

#define MAX_ACTIVE_EFFECTS 16


/* GFX IDs from spotanim.dat */
#define GFX_BOLT            27
#define GFX_SPLASH          85     /* blue splash on spell miss */
#define GFX_ICE_BARRAGE_PROJ 368
#define GFX_ICE_BARRAGE_HIT 369
#define GFX_BLOOD_BARRAGE_HIT 377
#define GFX_TEKTON_METEOR_SPLAT 659
#define GFX_TEKTON_METEOR_PROJ  660
#define GFX_DRAGON_BOLT     1468

/* player weapon projectiles (zulrah encounter) */
#define GFX_TRIDENT_CAST    665    /* casting effect on player */
#define GFX_TRIDENT_PROJ    1040   /* trident projectile in flight */
#define GFX_TRIDENT_IMPACT  1042   /* trident hit splash on target */
#define GFX_DRAGON_ARROW    1120   /* dragon arrow projectile (tbow) */
#define GFX_RUNE_ARROW      15     /* rune arrow projectile (MSB) */
#define GFX_DRAGON_DART     1122   /* dragon dart projectile (blowpipe) */
#define GFX_RUNE_DART       231    /* rune dart projectile */
#define GFX_BLOWPIPE_SPEC   1043   /* blowpipe special attack effect */

typedef struct {
    int gfx_id;
    uint32_t model_id;
    int anim_seq_id;      /* -1 = no animation (static model) */
    int resize_xy;        /* 128 = 1.0x */
    int resize_z;
} SpotAnimMeta;

/* parsed from spotanim.dat via export_spotanims.py */
static const SpotAnimMeta SPOTANIM_TABLE[] = {
    { GFX_BOLT,              3135,  -1,   128, 128 },
    { GFX_SPLASH,            3080,  653,  128, 128 },
    { GFX_ICE_BARRAGE_PROJ,  14215, 1964, 128, 128 },
    { GFX_ICE_BARRAGE_HIT,   6381,  1965, 128, 128 },
    { GFX_BLOOD_BARRAGE_HIT, 6375,  1967, 128, 128 },
    { GFX_TEKTON_METEOR_SPLAT, 14760, 3941, 128, 128 },
    { GFX_TEKTON_METEOR_PROJ,  14759, 3942, 128, 128 },
    { GFX_DRAGON_BOLT,       0xD0001, -1, 128, 128 }, /* synthetic recolored model */
    /* player weapon projectiles (zulrah encounter) */
    { GFX_TRIDENT_CAST,      20823,  5460, 128, 128 },
    { GFX_TRIDENT_PROJ,      20825,  5462, 128, 128 },
    { GFX_TRIDENT_IMPACT,    20824,  5461, 128, 128 },
    { GFX_DRAGON_ARROW,      26377,  6622, 128, 128 },
    { GFX_RUNE_ARROW,        3136,   -1,   128, 128 },
    { GFX_DRAGON_DART,       26379,  6622, 128, 128 },
    { GFX_RUNE_DART,         3131,   -1,   128, 128 },
    { GFX_BLOWPIPE_SPEC,     29421,  876,  128, 128 },
};
#define SPOTANIM_TABLE_SIZE (sizeof(SPOTANIM_TABLE) / sizeof(SPOTANIM_TABLE[0]))

static const SpotAnimMeta* spotanim_lookup(int gfx_id) {
    for (int i = 0; i < (int)SPOTANIM_TABLE_SIZE; i++) {
        if (SPOTANIM_TABLE[i].gfx_id == gfx_id) return &SPOTANIM_TABLE[i];
    }
    return NULL;
}


typedef enum {
    EFFECT_NONE = 0,
    EFFECT_SPOTANIM,     /* plays at a fixed position (impact effects) */
    EFFECT_PROJECTILE,   /* travels from source to target with parabolic arc */
} EffectType;

typedef struct {
    EffectType type;
    int gfx_id;
    const SpotAnimMeta* meta;

    /* world position in sub-tile coords (128 units per tile) */
    double src_x, src_y;      /* start (projectiles) */
    double dst_x, dst_y;      /* end (projectiles) or position (spotanims) */
    double cur_x, cur_y;      /* current interpolated position */
    double height;             /* current height in sub-tile units */

    /* projectile trajectory (from SceneProjectile.java) */
    double x_increment;
    double y_increment;
    double diagonal_increment;
    double height_increment;
    double height_accel;       /* aDouble1578: parabolic curvature */
    int start_height;          /* sub-tile units */
    int end_height;
    int initial_slope;         /* trajectory arc angle */

    /* timing in client ticks (50 Hz) */
    int start_tick;
    int stop_tick;
    int started;               /* has calculateIncrements been called? */

    /* animation state */
    int anim_frame;
    int anim_tick_counter;
    AnimModelState* anim_state;  /* per-effect vertex transform state (heap) */

    /* orientation */
    int turn_value;            /* 0-2047 OSRS angle units */
    int tilt_angle;
} ActiveEffect;


/** Free an effect's animation state and mark it inactive. */
static void effect_free(ActiveEffect* e) {
    if (e->anim_state) {
        anim_model_state_free(e->anim_state);
        e->anim_state = NULL;
    }
    e->type = EFFECT_NONE;
}

/** Find a free effect slot, evicting the oldest if full. */
static int effect_find_slot(ActiveEffect effects[MAX_ACTIVE_EFFECTS]) {
    for (int i = 0; i < MAX_ACTIVE_EFFECTS; i++) {
        if (effects[i].type == EFFECT_NONE) return i;
    }
    /* evict oldest */
    int oldest = 0;
    for (int i = 1; i < MAX_ACTIVE_EFFECTS; i++) {
        if (effects[i].start_tick < effects[oldest].start_tick) oldest = i;
    }
    effect_free(&effects[oldest]);
    return oldest;
}

/** Create AnimModelState for an effect's model (if it has animation data). */
static void effect_init_anim_state(
    ActiveEffect* e,
    ModelCache* model_cache,
    ModelCache* secondary_model_cache
) {
    if (!e->meta || e->meta->anim_seq_id < 0) return;

    OsrsModel* om = model_cache_get(model_cache, e->meta->model_id);
    if (!om && secondary_model_cache)
        om = model_cache_get(secondary_model_cache, e->meta->model_id);
    if (!om || !om->vertex_skins || om->base_vert_count == 0) return;

    e->anim_state = anim_model_state_create(
        om->vertex_skins, om->base_vert_count);
}


/**
 * Spawn a spotanim effect at a world position (impact splash, etc).
 * Duration is determined by the animation length, or a fixed 30 client ticks
 * for static models.
 */
static int effect_spawn_spotanim_subtile(
    ActiveEffect effects[MAX_ACTIVE_EFFECTS],
    int gfx_id,
    float subtile_x, float subtile_y,
    int current_client_tick,
    AnimCache* anim_cache,
    ModelCache* model_cache,
    ModelCache* secondary_model_cache
) {
    const SpotAnimMeta* meta = spotanim_lookup(gfx_id);
    if (!meta) return -1;

    int slot = effect_find_slot(effects);
    ActiveEffect* e = &effects[slot];
    memset(e, 0, sizeof(ActiveEffect));
    e->type = EFFECT_SPOTANIM;
    e->gfx_id = gfx_id;
    e->meta = meta;

    e->cur_x = subtile_x;
    e->cur_y = subtile_y;
    e->height = 0;

    e->start_tick = current_client_tick;

    /* duration from animation, or 30 client ticks default */
    int duration = 30;
    if (meta->anim_seq_id >= 0 && anim_cache) {
        AnimSequence* seq = anim_get_sequence(anim_cache, meta->anim_seq_id);
        if (seq) {
            duration = 0;
            for (int f = 0; f < seq->frame_count; f++) {
                duration += seq->frames[f].delay;
            }
        }
    }
    e->stop_tick = current_client_tick + duration;

    effect_init_anim_state(e, model_cache, secondary_model_cache);
    return slot;
}

/* convenience wrapper: integer tile coords → sub-tile center */
static int effect_spawn_spotanim(
    ActiveEffect effects[MAX_ACTIVE_EFFECTS],
    int gfx_id, int world_x, int world_y,
    int current_client_tick, AnimCache* anim_cache,
    ModelCache* model_cache, ModelCache* secondary_model_cache
) {
    return effect_spawn_spotanim_subtile(effects, gfx_id,
        world_x * 128.0f + 64.0f, world_y * 128.0f + 64.0f,
        current_client_tick, anim_cache, model_cache, secondary_model_cache);
}

/**
 * Spawn a traveling projectile from source to target position.
 *
 * Trajectory math from SceneProjectile.java calculateIncrements/progressCycles:
 * - parabolic height arc controlled by initialSlope
 * - position advances linearly per client tick
 * - height has quadratic acceleration term
 */
static int effect_spawn_projectile(
    ActiveEffect effects[MAX_ACTIVE_EFFECTS],
    int gfx_id,
    int src_world_x, int src_world_y,
    int dst_world_x, int dst_world_y,
    int delay_client_ticks,
    int duration_client_ticks,
    int start_height_subtile,
    int end_height_subtile,
    int slope,
    int current_client_tick,
    ModelCache* model_cache,
    ModelCache* secondary_model_cache
) {
    const SpotAnimMeta* meta = spotanim_lookup(gfx_id);
    if (!meta) return -1;

    int slot = effect_find_slot(effects);
    ActiveEffect* e = &effects[slot];
    memset(e, 0, sizeof(ActiveEffect));
    e->type = EFFECT_PROJECTILE;
    e->gfx_id = gfx_id;
    e->meta = meta;

    e->src_x = src_world_x * 128.0 + 64.0;
    e->src_y = src_world_y * 128.0 + 64.0;
    e->dst_x = dst_world_x * 128.0 + 64.0;
    e->dst_y = dst_world_y * 128.0 + 64.0;
    e->cur_x = e->src_x;
    e->cur_y = e->src_y;
    e->start_height = start_height_subtile;
    e->end_height = end_height_subtile;
    e->height = start_height_subtile;
    e->initial_slope = slope;
    e->started = 0;

    e->start_tick = current_client_tick + delay_client_ticks;
    e->stop_tick = current_client_tick + delay_client_ticks + duration_client_ticks;

    effect_init_anim_state(e, model_cache, secondary_model_cache);
    return slot;
}

/**
 * Advance all active effects by one client tick (20ms).
 * Call this from the 50 Hz client-tick loop.
 */
static void effect_client_tick(
    ActiveEffect effects[MAX_ACTIVE_EFFECTS],
    int current_client_tick,
    AnimCache* anim_cache
) {
    for (int i = 0; i < MAX_ACTIVE_EFFECTS; i++) {
        ActiveEffect* e = &effects[i];
        if (e->type == EFFECT_NONE) continue;

        /* expired? */
        if (current_client_tick >= e->stop_tick) {
            effect_free(e);
            continue;
        }

        /* not started yet (delayed projectile) */
        if (current_client_tick < e->start_tick) continue;

        if (e->type == EFFECT_PROJECTILE) {
            if (!e->started) {
                /* calculateIncrements (SceneProjectile.java:37-58) */
                e->cur_x = e->src_x;
                e->cur_y = e->src_y;
                e->height = e->start_height;

                double cycles_left = (double)(e->stop_tick + 1 - current_client_tick);
                e->x_increment = (e->dst_x - e->cur_x) / cycles_left;
                e->y_increment = (e->dst_y - e->cur_y) / cycles_left;
                e->diagonal_increment = sqrt(
                    e->x_increment * e->x_increment +
                    e->y_increment * e->y_increment
                );

                e->height_increment = -e->diagonal_increment *
                    tan((double)e->initial_slope * 0.02454369);
                e->height_accel = 2.0 * (
                    (double)e->end_height - e->height -
                    e->height_increment * cycles_left
                ) / (cycles_left * cycles_left);

                e->started = 1;
            }

            /* progressCycles (SceneProjectile.java:100-118) */
            e->cur_x += e->x_increment;
            e->cur_y += e->y_increment;
            e->height += e->height_increment + 0.5 * e->height_accel;
            e->height_increment += e->height_accel;

            /* update orientation */
            e->turn_value = (int)(atan2(e->x_increment, e->y_increment) *
                325.949) + 1024;
            e->turn_value &= 0x7FF;
            e->tilt_angle = (int)(atan2(e->height_increment,
                e->diagonal_increment) * 325.949);
            e->tilt_angle &= 0x7FF;
        }

        /* advance animation */
        if (e->meta && e->meta->anim_seq_id >= 0 && anim_cache) {
            AnimSequence* seq = anim_get_sequence(anim_cache, e->meta->anim_seq_id);
            if (seq && seq->frame_count > 0) {
                e->anim_tick_counter++;
                while (e->anim_tick_counter >= seq->frames[e->anim_frame].delay) {
                    e->anim_tick_counter -= seq->frames[e->anim_frame].delay;
                    e->anim_frame++;
                    if (e->anim_frame >= seq->frame_count) {
                        e->anim_frame = 0;
                    }
                }
            }
        }
    }
}

/**
 * Clear all active effects (on episode reset).
 */
static void effect_clear_all(ActiveEffect effects[MAX_ACTIVE_EFFECTS]) {
    for (int i = 0; i < MAX_ACTIVE_EFFECTS; i++) {
        effect_free(&effects[i]);
    }
}

#endif /* OSRS_PVP_EFFECTS_H */
