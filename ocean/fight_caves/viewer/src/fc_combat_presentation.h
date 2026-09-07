#ifndef FC_COMBAT_PRESENTATION_H
#define FC_COMBAT_PRESENTATION_H

#include "fc_actor_animation.h"
#include "fc_spotanims.h"
#include "fc_terrain_loader.h"
#include "ui_assets.h"

typedef struct FcCombatPresentation FcCombatPresentation;

typedef struct {
    const FcState *state;
    const FcRenderEvents *events;
    const FcVisualScene *scene;
    TerrainMesh *terrain;
    AnimCache *anim_cache;
    const FcPlayerVisualProfile *player_profile;
    float tps;
} FcCombatPresentationContext;

typedef struct {
    FcCombatPresentationContext presentation;
    const FcRenderEntity *entities;
    int entity_count;
    NpcModelSet *player_models;
    NpcModelSet *npc_models;
    int active_loadout;
    const RuneCUiAssets *ui_assets;
    Camera3D camera;
} FcCombatPresentationDrawContext;

FcCombatPresentation *fc_combat_presentation_create(Texture2D shared_atlas);
int fc_combat_presentation_ready(
    const FcCombatPresentation *presentation);
void fc_combat_presentation_destroy(FcCombatPresentation *presentation);
void fc_combat_presentation_reset(FcCombatPresentation *presentation);
void fc_combat_presentation_clear_npc_healthbar(
    FcCombatPresentation *presentation, int npc_slot);

void fc_combat_presentation_ingest_tick(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context);
void fc_combat_presentation_update(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    float dt);
void fc_combat_presentation_draw_world(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    float dt);
void fc_combat_presentation_draw_healthbars(
    const FcCombatPresentation *presentation,
    const FcCombatPresentationDrawContext *context);
void fc_combat_presentation_draw_hitsplats(
    const FcCombatPresentation *presentation,
    const FcCombatPresentationDrawContext *context);

int fc_combat_presentation_npc_death_deferred(
    const FcCombatPresentation *presentation,
    const FcState *state,
    int npc_slot);
void fc_combat_presentation_deferred_deaths(
    const FcCombatPresentation *presentation,
    const FcState *state,
    unsigned char deferred_deaths[FC_MAX_NPCS]);

#endif
