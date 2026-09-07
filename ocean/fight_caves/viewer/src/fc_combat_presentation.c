#include "fc_combat_presentation.h"

#include "fc_asset_raylib.h"
#include "fc_assets.h"
#include "fc_models.h"
#include "fc_model_animation.h"
#include "fc_npc.h"
#include "fc_projectile_visual.h"
#include "raymath.h"
#include "rlgl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_HITSPLATS 32
#define MAX_PROJECTILES 16
#define MAX_VISUAL_EFFECTS 32
#define OSRS_HITSPLAT_SECONDS 1.0f
#define OSRS_HEALTHBAR_SECONDS 6.0f
#define POLICY_REPLAY_BASE_TPS (5.0f / 3.0f)

#define PROJ_JAD_MAGIC_LAUNCH 439
#define PROJ_TOK_XIL_SPINE 443
#define PROJ_TOK_XIL_IMPACT 444
#define PROJ_KET_ZEK_FIRE 445
#define PROJ_KET_ZEK_IMPACT 446
#define PROJ_JAD_MAGIC_TRAVEL 448
#define PROJ_JAD_MAGIC_IMPACT 157
#define PROJ_JAD_RANGED_IMPACT 451
#define PROJ_SPOTANIM_MODEL_BASE 0xA2000000u

typedef enum {
    HITSPLAT_DAMAGE = 0,
    HITSPLAT_HEAL = 1,
    HITSPLAT_PRAYER_DRAIN = 2,
} HitsplatKind;

typedef struct {
    int active;
    float world_x;
    float world_y;
    float world_z;
    FcVisualTargetKind actor_kind;
    int actor_slot;
    int overlay_slot;
    int damage;
    int kind;
    float seconds_left;
} Hitsplat;

typedef struct {
    int active;
    float src_x;
    float src_y;
    float src_z;
    float dst_x;
    float dst_y;
    float dst_z;
    float x;
    float y;
    float z;
    float velocity_x;
    float velocity_y;
    float velocity_z;
    float total_time;
    float elapsed;
    float launch_delay;
    int launched;
    FcVisualTargetKind source_kind;
    int source_slot;
    float source_y_offset;
    FcVisualTargetKind target_kind;
    int target_slot;
    float target_y_offset;
    int track_target;
    int attack_style;
    int launch_tick;
    int has_deferred_hitsplat;
    FcVisualTargetKind hitsplat_actor_kind;
    int hitsplat_actor_slot;
    float hitsplat_world_x;
    float hitsplat_world_y;
    float hitsplat_world_z;
    int hitsplat_damage;
    Color color;
    float radius;
    uint32_t spot_id;
    uint32_t launch_spot_id;
    uint32_t impact_spot_id;
    float projectile_angle;
    float projectile_progress;
    AnimModelState *anim_state;
    uint16_t anim_sequence;
    int anim_frame;
    float anim_timer;
} VisualProjectile;

typedef struct {
    int active;
    float x;
    float y;
    float z;
    float total_time;
    float elapsed;
    Color color;
    float radius;
    uint32_t spot_id;
    float yaw_degrees;
    int attached;
    FcVisualTargetKind attached_kind;
    int attached_slot;
    float attached_y_offset;
    FcVisualTargetKind face_kind;
    int face_slot;
    AnimModelState *anim_state;
    uint16_t anim_sequence;
    int anim_frame;
    float anim_timer;
} VisualEffect;

struct FcCombatPresentation {
    Hitsplat hitsplats[MAX_HITSPLATS];
    float player_healthbar_timer;
    float npc_healthbar_timers[FC_MAX_NPCS];
    VisualProjectile projectiles[MAX_PROJECTILES];
    VisualEffect effects[MAX_VISUAL_EFFECTS];
    NpcModelSet *projectile_models;
    SpotAnimSet *spotanims;
    Texture2D hitsplat_zero_texture;
    Texture2D hitsplat_damage_texture;
    Texture2D hitsplat_heal_texture;
    Texture2D hitsplat_prayer_drain_texture;
    Texture2D healthbar_full_texture;
    Texture2D healthbar_empty_texture;
};

static float ground_height(const FcCombatPresentationContext *context,
                           int tile_x, int tile_y) {
    return context && context->terrain && context->terrain->loaded
        ? terrain_height_at(context->terrain, tile_x, tile_y) + 0.1f : 0.0f;
}

static float smooth_ground_height(const FcCombatPresentationContext *context,
                                  float tile_x, float tile_y) {
    int x0 = (int)floorf(tile_x);
    int y0 = (int)floorf(tile_y);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= FC_ARENA_WIDTH) x0 = FC_ARENA_WIDTH - 1;
    if (y0 >= FC_ARENA_HEIGHT) y0 = FC_ARENA_HEIGHT - 1;
    int x1 = x0 + 1 < FC_ARENA_WIDTH ? x0 + 1 : x0;
    int y1 = y0 + 1 < FC_ARENA_HEIGHT ? y0 + 1 : y0;
    float tx = tile_x - floorf(tile_x);
    float ty = tile_y - floorf(tile_y);
    float h00 = ground_height(context, x0, y0);
    float h10 = ground_height(context, x1, y0);
    float h01 = ground_height(context, x0, y1);
    float h11 = ground_height(context, x1, y1);
    float h0 = h00 + (h10 - h00) * tx;
    float h1 = h01 + (h11 - h01) * tx;
    return h0 + (h1 - h0) * ty;
}

static void free_projectile(VisualProjectile *projectile) {
    if (!projectile) return;
    if (projectile->anim_state)
        anim_model_state_free(projectile->anim_state);
    memset(projectile, 0, sizeof(*projectile));
}

static void free_effect(VisualEffect *effect) {
    if (!effect) return;
    if (effect->anim_state) anim_model_state_free(effect->anim_state);
    memset(effect, 0, sizeof(*effect));
}

static void clear_visuals(FcCombatPresentation *presentation) {
    for (int i = 0; i < MAX_PROJECTILES; i++)
        free_projectile(&presentation->projectiles[i]);
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++)
        free_effect(&presentation->effects[i]);
}

FcCombatPresentation *fc_combat_presentation_create(Texture2D shared_atlas) {
    FcCombatPresentation *presentation = calloc(1, sizeof(*presentation));
    if (!presentation) return NULL;
    if (fc_asset_exists("fc_projectiles.models")) {
        presentation->projectile_models = fc_npc_models_load(
            "fc_projectiles.models", shared_atlas);
    }
    if (fc_asset_exists("fc_spotanims.bin"))
        presentation->spotanims = spotanims_load("fc_spotanims.bin");
    presentation->hitsplat_zero_texture = fc_load_texture_asset(
        "data/sprites/ui/hitsplat_zero.png");
    presentation->hitsplat_damage_texture = fc_load_texture_asset(
        "data/sprites/ui/hitsplat_damage.png");
    presentation->hitsplat_heal_texture = fc_load_texture_asset(
        "data/sprites/ui/hitsplat_heal.png");
    presentation->hitsplat_prayer_drain_texture = fc_load_texture_asset(
        "data/sprites/ui/hitsplat_prayer_drain.png");
    presentation->healthbar_full_texture = fc_load_texture_asset(
        "data/sprites/ui/healthbar_full_30.png");
    presentation->healthbar_empty_texture = fc_load_texture_asset(
        "data/sprites/ui/healthbar_empty_30.png");
    Texture2D *textures[] = {
        &presentation->hitsplat_zero_texture,
        &presentation->hitsplat_damage_texture,
        &presentation->hitsplat_heal_texture,
        &presentation->hitsplat_prayer_drain_texture,
        &presentation->healthbar_full_texture,
        &presentation->healthbar_empty_texture,
    };
    int loaded = 0;
    for (int i = 0; i < (int)(sizeof(textures) / sizeof(textures[0])); i++) {
        if (textures[i]->id > 0) {
            SetTextureFilter(*textures[i], TEXTURE_FILTER_POINT);
            loaded++;
        }
    }
    fprintf(stderr, "Actor overhead sprites loaded: %d/6\n", loaded);
    return presentation;
}

int fc_combat_presentation_ready(
        const FcCombatPresentation *presentation) {
    if (!presentation || !presentation->projectile_models ||
        !presentation->projectile_models->loaded || !presentation->spotanims ||
        !presentation->spotanims->loaded) {
        return 0;
    }
    const Texture2D textures[] = {
        presentation->hitsplat_zero_texture,
        presentation->hitsplat_damage_texture,
        presentation->hitsplat_heal_texture,
        presentation->hitsplat_prayer_drain_texture,
        presentation->healthbar_full_texture,
        presentation->healthbar_empty_texture,
    };
    for (int i = 0; i < (int)(sizeof(textures) / sizeof(textures[0])); i++) {
        if (textures[i].id == 0) return 0;
    }
    return 1;
}

void fc_combat_presentation_destroy(FcCombatPresentation *presentation) {
    if (!presentation) return;
    clear_visuals(presentation);
    Texture2D textures[] = {
        presentation->hitsplat_zero_texture,
        presentation->hitsplat_damage_texture,
        presentation->hitsplat_heal_texture,
        presentation->hitsplat_prayer_drain_texture,
        presentation->healthbar_full_texture,
        presentation->healthbar_empty_texture,
    };
    for (int i = 0; i < (int)(sizeof(textures) / sizeof(textures[0])); i++) {
        if (textures[i].id > 0) UnloadTexture(textures[i]);
    }
    if (presentation->spotanims) spotanims_free(presentation->spotanims);
    if (presentation->projectile_models)
        fc_npc_models_unload(presentation->projectile_models);
    free(presentation);
}

void fc_combat_presentation_reset(FcCombatPresentation *presentation) {
    if (!presentation) return;
    clear_visuals(presentation);
    memset(presentation->hitsplats, 0, sizeof(presentation->hitsplats));
    presentation->player_healthbar_timer = 0.0f;
    memset(presentation->npc_healthbar_timers, 0,
           sizeof(presentation->npc_healthbar_timers));
}

void fc_combat_presentation_clear_npc_healthbar(
    FcCombatPresentation *presentation, int npc_slot) {
    if (!presentation || npc_slot < 0 || npc_slot >= FC_MAX_NPCS) return;
    presentation->npc_healthbar_timers[npc_slot] = 0.0f;
}

static VisualEffect *spawn_effect(FcCombatPresentation *presentation,
                                  uint32_t spot_id, float x, float y, float z,
                                  float duration, Color color, float radius,
                                  float yaw_degrees) {
    if (!presentation || spot_id == 0) return NULL;
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++) {
        if (!presentation->effects[i].active) {
            VisualEffect *effect = &presentation->effects[i];
            memset(effect, 0, sizeof(*effect));
            effect->active = 1;
            effect->x = x;
            effect->y = y;
            effect->z = z;
            effect->total_time = duration;
            effect->color = color;
            effect->radius = radius;
            effect->spot_id = spot_id;
            effect->yaw_degrees = yaw_degrees;
            return effect;
        }
    }
    return NULL;
}

static VisualProjectile *spawn_projectile(FcCombatPresentation *presentation,
    float source_x, float source_y, float source_z,
    float target_x, float target_y, float target_z,
    float travel_seconds, Color color, float radius,
    uint32_t travel_spot, uint32_t launch_spot, uint32_t impact_spot) {
    if (!presentation ||
        (travel_spot == 0 && launch_spot == 0 && impact_spot == 0)) return NULL;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        if (!presentation->projectiles[i].active) {
            VisualProjectile *projectile = &presentation->projectiles[i];
            free_projectile(projectile);
            projectile->active = 1;
            projectile->src_x = source_x;
            projectile->src_y = source_y;
            projectile->src_z = source_z;
            projectile->dst_x = target_x;
            projectile->dst_y = target_y;
            projectile->dst_z = target_z;
            projectile->x = source_x;
            projectile->y = source_y;
            projectile->z = source_z;
            projectile->total_time = travel_seconds;
            projectile->color = color;
            projectile->radius = radius;
            projectile->spot_id = travel_spot;
            projectile->launch_spot_id = launch_spot;
            projectile->impact_spot_id = impact_spot;
            return projectile;
        }
    }
    return NULL;
}

static int actor_world_point(const FcCombatPresentationContext *context,
                             FcVisualTargetKind kind, int slot,
                             float *x, float *y, float *z) {
    if (!context || !context->scene || !x || !y || !z) return 0;
    FcVisualPose pose;
    float height;
    if (kind == FC_VISUAL_TARGET_PLAYER && context->scene->player.active) {
        pose = fc_visual_scene_player_pose(context->scene);
        height = 1.5f;
    } else if (kind == FC_VISUAL_TARGET_NPC && slot >= 0 &&
               slot < FC_MAX_NPCS && context->scene->npcs[slot].active) {
        pose = fc_visual_scene_npc_pose(context->scene, slot);
        height = 1.0f + (float)context->scene->npcs[slot].size * 0.3f;
    } else {
        return 0;
    }
    *x = pose.x;
    *z = -pose.y;
    *y = smooth_ground_height(context, pose.x, pose.y) + height;
    return 1;
}

static int tile_distance(int source_x, int source_y,
                         int target_x, int target_y) {
    int dx = abs(target_x - source_x);
    int dy = abs(target_y - source_y);
    return dx > dy ? dx : dy;
}

static float animation_client_cycles(const AnimSequence *sequence) {
    if (!sequence || sequence->frame_count == 0) return 0.0f;
    float total = 0.0f;
    for (int i = 0; i < sequence->frame_count; i++)
        total += sequence->frames[i].delay > 0 ? sequence->frames[i].delay : 1;
    return total;
}

static float effect_duration(const FcCombatPresentation *presentation,
                             const FcCombatPresentationContext *context,
                             uint32_t spot_id, float retained_cycles) {
    const SpotAnimDef *spot = presentation && presentation->spotanims
        ? spotanim_find(presentation->spotanims, (int)spot_id) : NULL;
    float cycles = 0.0f;
    if (spot && spot->animation_id >= 0 && context && context->anim_cache) {
        cycles = animation_client_cycles(anim_get_sequence(
            context->anim_cache, (uint16_t)spot->animation_id));
    }
    return fc_projectile_effect_duration_seconds(cycles, retained_cycles,
        context && context->tps > 0.0f
            ? context->tps : POLICY_REPLAY_BASE_TPS);
}

static void configure_tracking(FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context, VisualProjectile *projectile,
    FcVisualTargetKind source_kind, int source_slot,
    FcVisualTargetKind target_kind, int target_slot, int attack_style,
    float launch_cycles, float end_cycles, float angle, float progress,
    int track_target) {
    if (!presentation || !context || !projectile) return;
    projectile->source_kind = source_kind;
    projectile->source_slot = source_slot;
    projectile->target_kind = target_kind;
    projectile->target_slot = target_slot;
    projectile->track_target = track_target;
    projectile->attack_style = attack_style;
    projectile->launch_tick = context->state->tick;
    projectile->projectile_angle = angle >= 0.0f ? angle : 15.0f;
    projectile->projectile_progress = progress >= 0.0f ? progress : 0.0f;
    FcProjectileTiming timing = {0};
    if (fc_projectile_timing_from_client_cycles(
            launch_cycles, end_cycles, context->tps, &timing)) {
        projectile->launch_delay = timing.launch_delay;
        projectile->total_time = timing.total_duration;
    } else {
        projectile->launch_delay = 0.0f;
    }
    float x;
    float y;
    float z;
    if (actor_world_point(context, source_kind, source_slot, &x, &y, &z)) {
        projectile->source_y_offset = projectile->src_y - y;
        projectile->src_x = x;
        projectile->src_y = y + projectile->source_y_offset;
        projectile->src_z = z;
    }
    if (track_target &&
        actor_world_point(context, target_kind, target_slot, &x, &y, &z)) {
        projectile->target_y_offset = projectile->dst_y - y;
        projectile->dst_x = x;
        projectile->dst_y = y + projectile->target_y_offset;
        projectile->dst_z = z;
    }
    projectile->x = projectile->src_x;
    projectile->y = projectile->src_y;
    projectile->z = projectile->src_z;
    if (projectile->launch_spot_id != 0) {
        float retain = launch_cycles > 30.0f ? launch_cycles : 30.0f;
        float duration = effect_duration(presentation, context,
                                         projectile->launch_spot_id, retain);
        float yaw = atan2f(projectile->dst_x - projectile->src_x,
                           projectile->dst_z - projectile->src_z) * RAD2DEG;
        VisualEffect *effect = spawn_effect(presentation,
            projectile->launch_spot_id, projectile->src_x, projectile->src_y,
            projectile->src_z, duration, projectile->color,
            projectile->radius * 1.4f, yaw);
        if (effect) {
            effect->attached = 1;
            effect->attached_kind = source_kind;
            effect->attached_slot = source_slot;
            effect->face_kind = target_kind;
            effect->face_slot = target_slot;
            if (actor_world_point(context, source_kind, source_slot,
                                  &x, &y, &z))
                effect->attached_y_offset = effect->y - y;
        }
    }
}

static void show_healthbar(FcCombatPresentation *presentation,
                           FcVisualTargetKind kind, int slot) {
    if (kind == FC_VISUAL_TARGET_PLAYER) {
        presentation->player_healthbar_timer = OSRS_HEALTHBAR_SECONDS;
    } else if (kind == FC_VISUAL_TARGET_NPC &&
               slot >= 0 && slot < FC_MAX_NPCS) {
        presentation->npc_healthbar_timers[slot] = OSRS_HEALTHBAR_SECONDS;
    }
}

static int next_overlay_slot(const FcCombatPresentation *presentation,
                             FcVisualTargetKind kind, int actor_slot) {
    unsigned int used = 0;
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        const Hitsplat *hit = &presentation->hitsplats[i];
        if (hit->active && hit->actor_kind == kind &&
            hit->actor_slot == actor_slot &&
            hit->overlay_slot >= 0 && hit->overlay_slot < 4)
            used |= 1u << hit->overlay_slot;
    }
    for (int i = 0; i < 4; i++)
        if ((used & (1u << i)) == 0) return i;
    return 0;
}

static void spawn_status_splat(FcCombatPresentation *presentation,
    FcVisualTargetKind kind, int actor_slot,
    float x, float y, float z, int damage, HitsplatKind splat_kind) {
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        if (!presentation->hitsplats[i].active) {
            Hitsplat *hit = &presentation->hitsplats[i];
            hit->active = 1;
            hit->world_x = x;
            hit->world_y = y;
            hit->world_z = z;
            hit->actor_kind = kind;
            hit->actor_slot = actor_slot;
            hit->overlay_slot = next_overlay_slot(presentation, kind, actor_slot);
            hit->damage = damage;
            hit->kind = splat_kind;
            hit->seconds_left = OSRS_HITSPLAT_SECONDS;
            if (splat_kind != HITSPLAT_PRAYER_DRAIN)
                show_healthbar(presentation, kind, actor_slot);
            return;
        }
    }
}

static void spawn_hitsplat(FcCombatPresentation *presentation,
    FcVisualTargetKind kind, int actor_slot,
    float x, float y, float z, int damage) {
    spawn_status_splat(presentation, kind, actor_slot, x, y, z,
                       damage, HITSPLAT_DAMAGE);
}

static int defer_hitsplat(FcCombatPresentation *presentation,
                          const FcCombatPresentationContext *context,
                          const FcRenderHit *hit,
                          FcVisualTargetKind kind, int actor_slot,
                          float x, float y, float z) {
    if (!hit || hit->attack_style == ATTACK_MELEE) return 0;
    FcVisualTargetKind source_kind;
    FcVisualTargetKind target_kind;
    int source_slot;
    int target_slot;
    if (hit->target_entity_type == ENTITY_PLAYER) {
        if (hit->source_npc_slot < 0) return 0;
        source_kind = FC_VISUAL_TARGET_NPC;
        source_slot = hit->source_npc_slot;
        target_kind = FC_VISUAL_TARGET_PLAYER;
        target_slot = 0;
    } else if (hit->target_entity_type == ENTITY_NPC) {
        if (hit->target_npc_slot < 0) return 0;
        source_kind = FC_VISUAL_TARGET_PLAYER;
        source_slot = 0;
        target_kind = FC_VISUAL_TARGET_NPC;
        target_slot = hit->target_npc_slot;
    } else {
        return 0;
    }
    VisualProjectile *match = NULL;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        VisualProjectile *projectile = &presentation->projectiles[i];
        if (!projectile->active || projectile->has_deferred_hitsplat ||
            projectile->launch_tick >= context->state->tick ||
            projectile->source_kind != source_kind ||
            projectile->source_slot != source_slot ||
            projectile->target_kind != target_kind ||
            projectile->target_slot != target_slot ||
            projectile->attack_style != hit->attack_style) continue;
        if (!match || projectile->elapsed > match->elapsed) match = projectile;
    }
    if (!match) return 0;
    match->has_deferred_hitsplat = 1;
    match->hitsplat_actor_kind = kind;
    match->hitsplat_actor_slot = actor_slot;
    match->hitsplat_world_x = x;
    match->hitsplat_world_y = y;
    match->hitsplat_world_z = z;
    match->hitsplat_damage = hit->damage;
    return 1;
}

static int deferred_damage(const FcCombatPresentation *presentation,
                           FcVisualTargetKind kind, int actor_slot) {
    int damage = 0;
    if (!presentation) return 0;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        const VisualProjectile *projectile = &presentation->projectiles[i];
        if (projectile->active && projectile->has_deferred_hitsplat &&
            projectile->hitsplat_actor_kind == kind &&
            projectile->hitsplat_actor_slot == actor_slot)
            damage += projectile->hitsplat_damage;
    }
    return damage;
}

int fc_combat_presentation_npc_death_deferred(
    const FcCombatPresentation *presentation,
    const FcState *state,
    int npc_slot) {
    return presentation && state && npc_slot >= 0 && npc_slot < FC_MAX_NPCS &&
        state->npcs[npc_slot].is_dead &&
        deferred_damage(presentation, FC_VISUAL_TARGET_NPC, npc_slot) > 0;
}

void fc_combat_presentation_deferred_deaths(
    const FcCombatPresentation *presentation,
    const FcState *state,
    unsigned char deferred_deaths[FC_MAX_NPCS]) {
    if (!deferred_deaths) return;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        deferred_deaths[i] = (unsigned char)
            fc_combat_presentation_npc_death_deferred(presentation, state, i);
    }
}

static void ingest_player_attack(FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context) {
    const FcRenderEvents *events = context->events;
    const FcPlayerVisualProfile *profile = context->player_profile;
    int sx = events->player_attack_source_x;
    int sy = events->player_attack_source_y;
    int tx = events->player_attack_target_x;
    int ty = events->player_attack_target_y;
    int target_size = events->player_attack_target_size;
    float source_x = (float)sx + 0.5f;
    float source_y = ground_height(context, sx, sy) +
                     profile->projectile_start_height / 128.0f;
    float source_z = -((float)sy + 0.5f);
    float target_x = (float)tx + (float)target_size * 0.5f;
    float target_y = ground_height(context, tx, ty) +
                     profile->projectile_end_height / 128.0f;
    float target_z = -((float)ty + (float)target_size * 0.5f);
    float end_cycle = fc_projectile_profile_end_cycle(
        profile->projectile_launch_delay_client_ticks,
        profile->projectile_length_adjustment,
        profile->projectile_step_multiplier, tile_distance(sx, sy, tx, ty));
    VisualProjectile *projectile = spawn_projectile(presentation,
        source_x, source_y, source_z, target_x, target_y, target_z, 0.1f,
        profile->projectile_color, profile->projectile_radius,
        profile->projectile_travel_spot, profile->projectile_launch_spot,
        profile->projectile_impact_spot);
    configure_tracking(presentation, context, projectile,
        FC_VISUAL_TARGET_PLAYER, 0, FC_VISUAL_TARGET_NPC,
        events->player_attack_target_npc_slot, ATTACK_RANGED,
        profile->projectile_launch_delay_client_ticks, end_cycle,
        profile->projectile_angle, profile->projectile_progress, 1);
}

static void ingest_npc_attack(FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    const FcRenderNpcAttack *attack) {
    if (!attack->hit_queued || attack->attack_style == ATTACK_MELEE) return;
    int source_x = attack->source_x;
    int source_y = attack->source_y;
    if (attack->source_size > 1) {
        if (attack->target_x < attack->source_x) source_x = attack->source_x;
        else if (attack->target_x >= attack->source_x + attack->source_size)
            source_x = attack->source_x + attack->source_size - 1;
        else source_x = attack->target_x;
        if (attack->target_y < attack->source_y) source_y = attack->source_y;
        else if (attack->target_y >= attack->source_y + attack->source_size)
            source_y = attack->source_y + attack->source_size - 1;
        else source_y = attack->target_y;
    }
    float source_ground = ground_height(context, source_x, source_y);
    float source_world_x = (float)source_x + 0.5f;
    float source_world_y = source_ground + 1.0f +
                           (float)attack->source_size * 0.3f;
    float source_world_z = -((float)source_y + 0.5f);
    float target_ground = ground_height(
        context, attack->target_x, attack->target_y);
    float target_world_x = (float)attack->target_x + 0.5f;
    float target_world_y = target_ground + 1.5f;
    float target_world_z = -((float)attack->target_y + 0.5f);
    Color color = attack->attack_style == ATTACK_MAGIC
        ? CLITERAL(Color){255, 104, 36, 235}
        : CLITERAL(Color){218, 178, 92, 235};
    float radius = attack->npc_type == NPC_TZTOK_JAD ? 0.3f : 0.15f;
    uint32_t travel_spot = 0;
    uint32_t launch_spot = 0;
    uint32_t impact_spot = 0;
    float start_height = -1.0f;
    float end_height = -1.0f;
    float start_cycle = 0.0f;
    float angle = -1.0f;
    float length_adjustment = 0.0f;
    float progress = -1.0f;
    float step_multiplier = 0.0f;
    float fixed_end_cycle = -1.0f;
    int track_target = 1;
    if (attack->npc_type == NPC_TOK_XIL) {
        travel_spot = PROJ_TOK_XIL_SPINE;
        impact_spot = PROJ_TOK_XIL_IMPACT;
        start_height = 296.0f;
        end_height = 40.0f;
        start_cycle = 32.0f;
        angle = 16.0f;
        progress = 0.0f;
        step_multiplier = 5.0f;
    } else if (attack->npc_type == NPC_KET_ZEK) {
        travel_spot = PROJ_KET_ZEK_FIRE;
        impact_spot = PROJ_KET_ZEK_IMPACT;
        start_height = 192.0f;
        end_height = 40.0f;
        start_cycle = 28.0f;
        angle = 16.0f;
        length_adjustment = 8.0f;
        progress = 0.0f;
        step_multiplier = 8.0f;
    } else if (attack->npc_type == NPC_TZTOK_JAD &&
               attack->attack_style == ATTACK_MAGIC) {
        launch_spot = PROJ_JAD_MAGIC_LAUNCH;
        travel_spot = PROJ_JAD_MAGIC_TRAVEL;
        impact_spot = PROJ_JAD_MAGIC_IMPACT;
        start_height = 172.0f;
        end_height = 124.0f;
        start_cycle = 41.0f;
        angle = 16.0f;
        progress = 64.0f;
        step_multiplier = 5.0f;
    } else if (attack->npc_type == NPC_TZTOK_JAD &&
               attack->attack_style == ATTACK_RANGED) {
        impact_spot = PROJ_JAD_RANGED_IMPACT;
        start_height = 768.0f;
        end_height = 52.0f;
        angle = 0.0f;
        progress = 0.0f;
        fixed_end_cycle = 60.0f;
        track_target = 0;
    }
    if (start_height >= 0.0f)
        source_world_y = source_ground + start_height / 128.0f;
    if (end_height >= 0.0f)
        target_world_y = target_ground + end_height / 128.0f;
    float end_cycle = fixed_end_cycle >= 0.0f ? fixed_end_cycle
        : fc_projectile_profile_end_cycle(start_cycle, length_adjustment,
            step_multiplier, tile_distance(source_x, source_y,
                                             attack->target_x, attack->target_y));
    VisualProjectile *projectile = spawn_projectile(presentation,
        source_world_x, source_world_y, source_world_z,
        target_world_x, target_world_y, target_world_z,
        0.1f, color, radius, travel_spot, launch_spot, impact_spot);
    configure_tracking(presentation, context, projectile,
        FC_VISUAL_TARGET_NPC, attack->npc_slot,
        FC_VISUAL_TARGET_PLAYER, 0, attack->attack_style,
        start_cycle, end_cycle, angle, progress, track_target);
}

void fc_combat_presentation_ingest_tick(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context) {
    if (!presentation || !context || !context->state || !context->events ||
        !context->player_profile) return;
    const FcState *state = context->state;
    int player_x = state->player.x;
    int player_y = state->player.y;
    if (player_x < 0) player_x = 0;
    if (player_y < 0) player_y = 0;
    if (player_x >= FC_ARENA_WIDTH) player_x = FC_ARENA_WIDTH - 1;
    if (player_y >= FC_ARENA_HEIGHT) player_y = FC_ARENA_HEIGHT - 1;
    float player_ground = ground_height(context, player_x, player_y);
    float player_world_x = (float)state->player.x + 0.5f;
    float player_world_z = -((float)state->player.y + 0.5f);
    if (state->tz_kih_prayer_drain_this_tick > 0) {
        spawn_status_splat(presentation, FC_VISUAL_TARGET_PLAYER, 0,
            player_world_x + 0.3f, player_ground + 3.0f, player_world_z,
            state->tz_kih_prayer_drain_this_tick, HITSPLAT_PRAYER_DRAIN);
    }
    if (context->events->player_attack_fired)
        ingest_player_attack(presentation, context);
    for (int i = 0; i < context->events->hit_count; i++) {
        const FcRenderHit *hit = &context->events->hits[i];
        if (hit->target_entity_type == ENTITY_PLAYER) {
            if (!defer_hitsplat(presentation, context, hit,
                    FC_VISUAL_TARGET_PLAYER, 0, player_world_x,
                    player_ground + 2.5f, player_world_z)) {
                spawn_hitsplat(presentation, FC_VISUAL_TARGET_PLAYER, 0,
                    player_world_x, player_ground + 2.5f,
                    player_world_z, hit->damage);
            }
        } else if (hit->target_entity_type == ENTITY_NPC &&
                   hit->target_npc_slot >= 0 &&
                   hit->target_npc_slot < FC_MAX_NPCS) {
            const FcNpc *target = &state->npcs[hit->target_npc_slot];
            float ground = ground_height(context, target->x, target->y);
            float x = (float)target->x + (float)target->size * 0.5f;
            float z = -((float)target->y + (float)target->size * 0.5f);
            float y = ground + 1.0f + (float)target->size * 0.5f;
            if (!defer_hitsplat(presentation, context, hit,
                    FC_VISUAL_TARGET_NPC, hit->target_npc_slot, x, y, z)) {
                spawn_hitsplat(presentation, FC_VISUAL_TARGET_NPC,
                    hit->target_npc_slot, x, y, z, hit->damage);
            }
        }
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        if (npc->healing_received_this_tick > 0) {
            float ground = ground_height(context, npc->x, npc->y);
            float x = (float)npc->x + (float)npc->size * 0.5f;
            float z = -((float)npc->y + (float)npc->size * 0.5f);
            float y = ground + 1.4f + (float)npc->size * 0.5f;
            spawn_status_splat(presentation, FC_VISUAL_TARGET_NPC, i,
                x, y, z, npc->healing_received_this_tick, HITSPLAT_HEAL);
        }
    }
    for (int i = 0; i < context->events->npc_attack_count; i++) {
        const FcRenderNpcAttack *attack = &context->events->npc_attacks[i];
        if (attack->npc_slot >= 0 && attack->npc_slot < FC_MAX_NPCS)
            ingest_npc_attack(presentation, context, attack);
    }
}

static void refresh_actor_points(const FcCombatPresentationContext *context,
                                 VisualProjectile *projectile,
                                 int refresh_source) {
    float x;
    float y;
    float z;
    if (refresh_source && actor_world_point(context, projectile->source_kind,
            projectile->source_slot, &x, &y, &z)) {
        projectile->src_x = x;
        projectile->src_y = y + projectile->source_y_offset;
        projectile->src_z = z;
    }
    if (projectile->track_target && actor_world_point(context,
            projectile->target_kind, projectile->target_slot, &x, &y, &z)) {
        projectile->dst_x = x;
        projectile->dst_y = y + projectile->target_y_offset;
        projectile->dst_z = z;
    }
}

static int update_projectile(const FcCombatPresentationContext *context,
                             VisualProjectile *projectile, float dt) {
    float end = projectile->elapsed + dt;
    if (end > projectile->total_time) end = projectile->total_time;
    refresh_actor_points(context, projectile, !projectile->launched);
    if (end < projectile->launch_delay) {
        projectile->x = projectile->src_x;
        projectile->y = projectile->src_y;
        projectile->z = projectile->src_z;
        projectile->elapsed = end;
        return 0;
    }
    projectile->launched = 1;
    float duration = projectile->total_time - projectile->launch_delay;
    if (duration < 0.001f) duration = 0.001f;
    FcProjectilePath path = {
        .source_x = projectile->src_x,
        .source_y = projectile->src_y,
        .source_z = projectile->src_z,
        .target_x = projectile->dst_x,
        .target_y = projectile->dst_y,
        .target_z = projectile->dst_z,
        .duration = duration,
        .angle = projectile->projectile_angle,
        .progress = projectile->projectile_progress / 128.0f,
    };
    FcProjectileSample sample = {0};
    if (fc_projectile_path_sample(&path, end - projectile->launch_delay,
                                  &sample)) {
        projectile->x = sample.x;
        projectile->y = sample.y;
        projectile->z = sample.z;
        projectile->velocity_x = sample.velocity_x;
        projectile->velocity_y = sample.velocity_y;
        projectile->velocity_z = sample.velocity_z;
    }
    projectile->elapsed = end;
    if (end >= projectile->total_time) {
        projectile->x = projectile->dst_x;
        projectile->y = projectile->dst_y;
        projectile->z = projectile->dst_z;
        return 1;
    }
    return 0;
}

void fc_combat_presentation_update(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    float dt) {
    if (!presentation || !context || dt <= 0.0f) return;
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        if (presentation->hitsplats[i].active) {
            presentation->hitsplats[i].seconds_left -= dt;
            if (presentation->hitsplats[i].seconds_left <= 0.0f)
                presentation->hitsplats[i].active = 0;
        }
    }
    if (presentation->player_healthbar_timer > 0.0f) {
        presentation->player_healthbar_timer -= dt;
        if (presentation->player_healthbar_timer < 0.0f)
            presentation->player_healthbar_timer = 0.0f;
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (presentation->npc_healthbar_timers[i] > 0.0f) {
            presentation->npc_healthbar_timers[i] -= dt;
            if (presentation->npc_healthbar_timers[i] < 0.0f)
                presentation->npc_healthbar_timers[i] = 0.0f;
        }
    }
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        VisualProjectile *projectile = &presentation->projectiles[i];
        if (!projectile->active || !update_projectile(context, projectile, dt))
            continue;
        float duration = effect_duration(presentation, context,
                                         projectile->impact_spot_id, 90.0f);
        spawn_effect(presentation, projectile->impact_spot_id,
            projectile->x, projectile->y, projectile->z, duration,
            projectile->color, projectile->radius * 1.4f, 0.0f);
        if (projectile->has_deferred_hitsplat) {
            spawn_hitsplat(presentation, projectile->hitsplat_actor_kind,
                projectile->hitsplat_actor_slot, projectile->hitsplat_world_x,
                projectile->hitsplat_world_y, projectile->hitsplat_world_z,
                projectile->hitsplat_damage);
        }
        free_projectile(projectile);
    }
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++) {
        VisualEffect *effect = &presentation->effects[i];
        if (effect->active) {
            effect->elapsed += dt;
            if (effect->elapsed >= effect->total_time) free_effect(effect);
        }
    }
}

static NpcModelEntry *projectile_model_for_spot(
    FcCombatPresentation *presentation, uint32_t spot_id,
    const SpotAnimDef **out_spot) {
    const SpotAnimDef *spot = presentation && presentation->spotanims
        ? spotanim_find(presentation->spotanims, (int)spot_id) : NULL;
    if (out_spot) *out_spot = spot;
    if (!presentation || spot_id == 0 || !presentation->projectile_models)
        return NULL;
    NpcModelEntry *entry = fc_npc_model_find(presentation->projectile_models,
                                             PROJ_SPOTANIM_MODEL_BASE + spot_id);
    if (!entry && spot && spot->model_id >= 0)
        entry = fc_npc_model_find(presentation->projectile_models,
                                  (uint32_t)spot->model_id);
    if (!entry)
        entry = fc_npc_model_find(presentation->projectile_models, spot_id);
    return entry && entry->loaded ? entry : NULL;
}

void fc_combat_presentation_draw_world(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    float dt) {
    if (!presentation || !context) return;
    float animation_dt = fc_actor_animation_scaled_dt(context->tps, dt);
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        VisualProjectile *projectile = &presentation->projectiles[i];
        if (!projectile->active || projectile->spot_id == 0 ||
            !projectile->launched) continue;
        const SpotAnimDef *spot = NULL;
        NpcModelEntry *entry = projectile_model_for_spot(
            presentation, projectile->spot_id, &spot);
        if (entry) {
            float horizontal_speed = sqrtf(
                projectile->velocity_x * projectile->velocity_x +
                projectile->velocity_z * projectile->velocity_z);
            float angle = atan2f(projectile->velocity_x,
                                 projectile->velocity_z) * RAD2DEG;
            float pitch = spot ? 0.0f
                : atan2f(projectile->velocity_y, horizontal_speed);
            float scale_xy = spot && spot->resize_xy > 0
                ? (float)spot->resize_xy / 128.0f : 1.0f;
            float scale_z = spot && spot->resize_z > 0
                ? (float)spot->resize_z / 128.0f : 1.0f;
            if (spot) angle += (float)spot->rotation;
            if (spot && spot->animation_id >= 0) {
                fc_model_animation_update(entry, context->anim_cache,
                    &projectile->anim_state, &projectile->anim_sequence,
                    &projectile->anim_frame, &projectile->anim_timer,
                    spot->animation_id, animation_dt, 0.0f);
            }
            Quaternion yaw = QuaternionFromAxisAngle(
                (Vector3){0, 1, 0}, angle * DEG2RAD);
            Quaternion tilt = QuaternionFromAxisAngle(
                (Vector3){1, 0, 0}, -pitch);
            Quaternion rotation = QuaternionMultiply(yaw, tilt);
            Vector3 axis = {0, 1, 0};
            float rotation_angle = 0.0f;
            QuaternionToAxisAngle(rotation, &axis, &rotation_angle);
            rlDisableBackfaceCulling();
            DrawModelEx(entry->model,
                (Vector3){projectile->x, projectile->y, projectile->z},
                axis, rotation_angle * RAD2DEG,
                (Vector3){scale_xy, scale_z, scale_xy}, WHITE);
            rlEnableBackfaceCulling();
        } else if (projectile->radius > 0.0f) {
            DrawSphere((Vector3){projectile->x, projectile->y, projectile->z},
                       projectile->radius, projectile->color);
        }
    }
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++) {
        VisualEffect *effect = &presentation->effects[i];
        if (!effect->active) continue;
        float x = effect->x;
        float y = effect->y;
        float z = effect->z;
        if (effect->attached) {
            float actor_x;
            float actor_y;
            float actor_z;
            if (actor_world_point(context, effect->attached_kind,
                    effect->attached_slot, &actor_x, &actor_y, &actor_z)) {
                x = actor_x;
                y = actor_y + effect->attached_y_offset;
                z = actor_z;
            }
        }
        float yaw = effect->yaw_degrees;
        if (effect->face_kind != FC_VISUAL_TARGET_NONE) {
            float target_x;
            float target_y;
            float target_z;
            if (actor_world_point(context, effect->face_kind,
                    effect->face_slot, &target_x, &target_y, &target_z)) {
                (void)target_y;
                yaw = atan2f(target_x - x, target_z - z) * RAD2DEG;
            }
        }
        const SpotAnimDef *spot = NULL;
        NpcModelEntry *entry = projectile_model_for_spot(
            presentation, effect->spot_id, &spot);
        if (entry) {
            float scale_xy = spot && spot->resize_xy > 0
                ? (float)spot->resize_xy / 128.0f : 1.0f;
            float scale_z = spot && spot->resize_z > 0
                ? (float)spot->resize_z / 128.0f : 1.0f;
            if (spot && spot->animation_id >= 0) {
                fc_model_animation_update(entry, context->anim_cache,
                    &effect->anim_state, &effect->anim_sequence,
                    &effect->anim_frame, &effect->anim_timer,
                    spot->animation_id, animation_dt, 0.0f);
            }
            rlDisableBackfaceCulling();
            DrawModelEx(entry->model, (Vector3){x, y, z},
                (Vector3){0, 1, 0},
                yaw + (spot ? (float)spot->rotation : 0.0f),
                (Vector3){scale_xy, scale_z, scale_xy}, WHITE);
            rlEnableBackfaceCulling();
        } else {
            DrawSphere((Vector3){x, y, z}, effect->radius, effect->color);
        }
    }
}

static const FcRenderEntity *find_actor(
    const FcCombatPresentationDrawContext *context,
    int actor_kind, int actor_slot) {
    for (int i = 0; i < context->entity_count; i++) {
        const FcRenderEntity *entity = &context->entities[i];
        if (actor_kind == FC_VISUAL_TARGET_PLAYER &&
            entity->entity_type == ENTITY_PLAYER) return entity;
        if (actor_kind == FC_VISUAL_TARGET_NPC &&
            entity->entity_type == ENTITY_NPC &&
            entity->npc_slot == actor_slot) return entity;
    }
    return NULL;
}

static FcVisualPose entity_pose(const FcCombatPresentationDrawContext *context,
                                const FcRenderEntity *entity) {
    return entity->entity_type == ENTITY_PLAYER
        ? fc_visual_scene_player_pose(context->presentation.scene)
        : fc_visual_scene_npc_pose(context->presentation.scene,
                                   entity->npc_slot);
}

static float entity_model_top(const FcCombatPresentationDrawContext *context,
                              const FcRenderEntity *entity) {
    Model *model = NULL;
    if (entity->entity_type == ENTITY_PLAYER) {
        NpcModelEntry *entry = fc_actor_player_model_entry(
            context->player_models, context->active_loadout);
        if (entry) model = &entry->model;
    } else if (context->npc_models) {
        NpcModelEntry *entry = fc_npc_model_find(context->npc_models,
            fc_npc_type_to_model_id(entity->npc_type));
        if (entry && entry->loaded) model = &entry->model;
    }
    if (model) {
        BoundingBox bounds = GetModelBoundingBox(*model);
        if (bounds.max.y > 0.1f && bounds.max.y < 20.0f) return bounds.max.y;
    }
    return entity->entity_type == ENTITY_PLAYER
        ? 2.0f : 1.3f + (float)entity->size * 0.5f;
}

static Vector3 overlay_anchor(const FcCombatPresentationDrawContext *context,
                              const FcRenderEntity *entity,
                              float height_fraction, float extra_height) {
    FcVisualPose pose = entity_pose(context, entity);
    return (Vector3){
        pose.x,
        smooth_ground_height(&context->presentation, pose.x, pose.y) +
            entity_model_top(context, entity) * height_fraction + extra_height,
        -pose.y,
    };
}

static Texture2D hitsplat_texture(const FcCombatPresentation *presentation,
                                  const Hitsplat *hit) {
    if (hit->kind == HITSPLAT_HEAL) return presentation->hitsplat_heal_texture;
    if (hit->kind == HITSPLAT_PRAYER_DRAIN)
        return presentation->hitsplat_prayer_drain_texture;
    return hit->damage > 0 ? presentation->hitsplat_damage_texture
                           : presentation->hitsplat_zero_texture;
}

void fc_combat_presentation_draw_healthbars(
    const FcCombatPresentation *presentation,
    const FcCombatPresentationDrawContext *context) {
    if (!presentation || !context || !context->entities ||
        !context->presentation.scene) return;
    for (int i = 0; i < context->entity_count; i++) {
        const FcRenderEntity *entity = &context->entities[i];
        float timer = entity->entity_type == ENTITY_PLAYER
            ? presentation->player_healthbar_timer
            : entity->npc_slot >= 0 && entity->npc_slot < FC_MAX_NPCS
                ? presentation->npc_healthbar_timers[entity->npc_slot] : 0.0f;
        if (timer <= 0.0f || entity->max_hp <= 0) continue;
        Vector2 screen = GetWorldToScreen(
            overlay_anchor(context, entity, 1.0f, 0.12f), context->camera);
        if (screen.x < -40.0f || screen.x > GetScreenWidth() + 40.0f ||
            screen.y < -20.0f || screen.y > GetScreenHeight() + 20.0f) continue;
        FcVisualTargetKind kind = entity->entity_type == ENTITY_PLAYER
            ? FC_VISUAL_TARGET_PLAYER : FC_VISUAL_TARGET_NPC;
        int slot = entity->entity_type == ENTITY_PLAYER ? 0 : entity->npc_slot;
        int visible_hp = entity->current_hp + deferred_damage(presentation,
                                                               kind, slot);
        if (visible_hp > entity->max_hp) visible_hp = entity->max_hp;
        int fill = visible_hp * 30 / entity->max_hp;
        if (fill < 0) fill = 0;
        if (fill > 30) fill = 30;
        int x = (int)roundf(screen.x) - 15;
        int y = (int)roundf(screen.y) - 3;
        if (presentation->healthbar_empty_texture.id > 0 &&
            presentation->healthbar_full_texture.id > 0) {
            DrawTexture(presentation->healthbar_empty_texture, x, y, WHITE);
            if (fill > 0) {
                Rectangle source = {0.0f, 0.0f, (float)fill, 5.0f};
                DrawTextureRec(presentation->healthbar_full_texture, source,
                               (Vector2){(float)x, (float)y}, WHITE);
            }
        } else {
            DrawRectangle(x, y, 30, 5, RED);
            DrawRectangle(x, y, fill, 5, GREEN);
        }
    }
}

void fc_combat_presentation_draw_hitsplats(
    const FcCombatPresentation *presentation,
    const FcCombatPresentationDrawContext *context) {
    if (!presentation || !context || !context->entities ||
        !context->presentation.scene) return;
    static const int slot_x[4] = {0, 0, -15, 15};
    static const int slot_y[4] = {0, -20, -10, -10};
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        const Hitsplat *hit = &presentation->hitsplats[i];
        if (!hit->active) continue;
        Vector3 world = {hit->world_x, hit->world_y, hit->world_z};
        const FcRenderEntity *entity = find_actor(context, hit->actor_kind,
                                                   hit->actor_slot);
        if (entity) world = overlay_anchor(context, entity, 0.5f, 0.0f);
        Vector2 screen = GetWorldToScreen(world, context->camera);
        if (screen.x < -50.0f || screen.x > GetScreenWidth() + 50.0f ||
            screen.y < -50.0f || screen.y > GetScreenHeight() + 50.0f) continue;
        int slot = hit->overlay_slot;
        if (slot < 0 || slot >= 4) slot = 0;
        int center_x = (int)roundf(screen.x) + slot_x[slot];
        int center_y = (int)roundf(screen.y) + slot_y[slot];
        Texture2D texture = hitsplat_texture(presentation, hit);
        if (texture.id > 0)
            DrawTexture(texture, center_x - 12, center_y - 12, WHITE);
        int value = hit->kind == HITSPLAT_DAMAGE
            ? hit->damage / 10 : (hit->damage + 9) / 10;
        char text[16];
        snprintf(text, sizeof(text), "%d", value);
        const float font_size = 12.0f;
        Font font = runec_ui_font_for_size(context->ui_assets, font_size);
        Vector2 measured = MeasureTextEx(font, text, font_size, 0.0f);
        float text_x = floorf((float)center_x - 1.0f - measured.x * 0.5f);
        float text_y = floorf((float)center_y - 6.0f);
        DrawTextEx(font, text, (Vector2){text_x + 1.0f, text_y + 1.0f},
                   font_size, 0.0f, BLACK);
        DrawTextEx(font, text, (Vector2){text_x, text_y},
                   font_size, 0.0f, WHITE);
    }
}
