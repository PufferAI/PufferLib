#pragma once

#include <stdio.h>
#include <string.h>

#include "encounters/encounter_nh_pvp.h"
#include "encounters/encounter_zulrah.h"
#include "encounters/encounter_inferno.h"
#include "encounters/encounter_colosseum.h"
#include "osrs_render.h"

static int encounter_name_is_pvp(const char* encounter_name) {
    return encounter_name &&
        (strcmp(encounter_name, "pvp") == 0 ||
         strcmp(encounter_name, "nh_pvp") == 0);
}

static void visual_require_gui_item_sprite(int raw_osrs_id, void* context) {
    gui_require_sprite_by_osrs_id((GuiState*)context, raw_osrs_id);
}

typedef struct {
    CollisionMap* cmap;
    int offset_x;
    int offset_y;
} VisualCollisionLoad;

static VisualCollisionLoad visual_load_encounter_collision_map(
    const EncounterDef* encounter_def,
    OsrsEnv* env,
    const char* encounter_name
) {
    CollisionMap* collision_map = NULL;
    int offset_x = 0;
    int offset_y = 0;

    if (encounter_name_is_pvp(encounter_name)) {
        collision_map = collision_map_load(OSRS_ASSET("wilderness.cmap"));
    } else if (strcmp(encounter_name, "zulrah") == 0) {
        collision_map = collision_map_load(OSRS_ASSET("zulrah.cmap"));
        offset_x = 2256;
        offset_y = 3061;
    } else if (strcmp(encounter_name, "inferno") == 0) {
        collision_map = collision_map_load(OSRS_ASSET("inferno.cmap"));
        offset_x = 2246;
        offset_y = 5315;
    } else if (strcmp(encounter_name, "colosseum") == 0) {
        collision_map = collision_map_load(OSRS_ASSET("colosseum.cmap"));
        offset_x = 1808;
        offset_y = 3090;
    }

    VisualCollisionLoad result = {NULL, offset_x, offset_y};
    if (collision_map == NULL) return result;

    if (!encounter_name_is_pvp(encounter_name)) {
        encounter_def->put_int(
            env->encounter_state,
            env->encounter_context,
            "world_offset_x",
            offset_x);
        encounter_def->put_int(
            env->encounter_state,
            env->encounter_context,
            "world_offset_y",
            offset_y);
    }
    encounter_def->put_ptr(
        env->encounter_state,
        env->encounter_context,
        "collision_map",
        collision_map);
    env->collision_map = collision_map;
    result.cmap = collision_map;
    return result;
}

static RenderClient* visual_init_render_scene(
    OsrsEnv* env,
    const char* encounter_name,
    const EncounterArenaTopology* route_topology
) {
    RenderClient* render_client = render_make_client(env);
    env->client = render_client;
    render_client->route_topology = route_topology;
    pvp_actor_route_caches_clear(render_client->player_route_cache);
#ifdef __EMSCRIPTEN__
    if (!encounter_name || encounter_name_is_pvp(encounter_name)) {
        render_client->ticks_per_second = 15.0f;
    }
#endif

    if (!encounter_name || encounter_name_is_pvp(encounter_name)) {
        osrs_asset_require_group(OSRS_ASSET_GROUP_PVP);
    } else if (strcmp(encounter_name, "zulrah") == 0) {
        osrs_asset_require_group(OSRS_ASSET_GROUP_ZULRAH);
        osrs_asset_require_group(OSRS_ASSET_GROUP_COMBAT_VISUALS);
    } else if (strcmp(encounter_name, "inferno") == 0) {
        osrs_asset_require_group(OSRS_ASSET_GROUP_INFERNO);
        osrs_asset_require_group(OSRS_ASSET_GROUP_COMBAT_VISUALS);
    } else if (strcmp(encounter_name, "colosseum") == 0) {
        osrs_asset_require_group(OSRS_ASSET_GROUP_COLOSSEUM);
        osrs_asset_require_group(OSRS_ASSET_GROUP_COMBAT_VISUALS);
        col_for_each_display_inventory_sprite_raw_osrs_id(
            visual_require_gui_item_sprite,
            &render_client->gui);
    }

    if (env->collision_map) {
        render_client->collision_map = (const CollisionMap*)env->collision_map;
    }

    double t0 = osrs_now_ms();
    render_client->model_cache = model_cache_load(OSRS_ASSET("equipment.models"));
    if (render_client->model_cache) render_client->show_models = 1;
    render_client->anim_cache = anim_cache_load(OSRS_ASSET("equipment.anims"));
    osrs_time_log("equipment models+anims", &t0);
    render_load_projectile_assets(render_client);
    osrs_time_log("projectile assets", &t0);
    render_init_overlay_models(render_client);
    osrs_time_log("overlay models", &t0);

    if (!encounter_name || encounter_name_is_pvp(encounter_name)) {
        render_client->terrain = terrain_load(OSRS_ASSET("wilderness.terrain"));
    } else if (strcmp(encounter_name, "zulrah") == 0) {
        render_client->terrain = terrain_load(OSRS_ASSET("zulrah.terrain"));
        render_client->objects = objects_load(OSRS_ASSET("zulrah.objects"));

        int offset_x = 2256;
        int offset_y = 3061;
        if (render_client->terrain)
            terrain_offset(render_client->terrain, offset_x, offset_y);
        if (render_client->objects)
            objects_offset(render_client->objects, offset_x, offset_y);

        render_client->collision_world_offset_x = offset_x;
        render_client->collision_world_offset_y = offset_y;
        render_client->npc_model_cache = model_cache_load(OSRS_ASSET("zulrah.models"));
        render_client->npc_anim_cache = anim_cache_load(OSRS_ASSET("zulrah.anims"));
        fprintf(stderr, "zulrah: npc_models=%d, npc_anims=%d seqs\n",
            render_client->npc_model_cache ? render_client->npc_model_cache->count : 0,
            render_client->npc_anim_cache ? render_client->npc_anim_cache->seq_count : 0);
    } else if (strcmp(encounter_name, "inferno") == 0) {
        render_client->terrain = terrain_load_region(OSRS_ASSET("inferno.terrain"), 35, 83);
        render_client->objects = objects_load(OSRS_ASSET("inferno.objects"));
        render_client->objects_zuk = objects_load(OSRS_ASSET("inferno_zuk.objects"));
        if (render_client->terrain)
            terrain_offset(render_client->terrain, 2246, 5315);
        if (render_client->objects)
            objects_offset(render_client->objects, 2246, 5315);
        if (render_client->objects_zuk)
            objects_offset(render_client->objects_zuk, 2246, 5315);

        render_client->npc_model_cache = model_cache_load(OSRS_ASSET("inferno.models"));
        render_client->npc_anim_cache = anim_cache_load(OSRS_ASSET("inferno.anims"));
        if (env->collision_map) {
            render_client->collision_world_offset_x = 2246;
            render_client->collision_world_offset_y = 5315;
        }
        fprintf(stderr, "inferno: terrain=%s, cmap=%s, npc_models=%d, npc_anims=%d seqs\n",
            render_client->terrain ? "loaded" : "MISSING",
            render_client->collision_map ? "loaded" : "MISSING",
            render_client->npc_model_cache ? render_client->npc_model_cache->count : 0,
            render_client->npc_anim_cache ? render_client->npc_anim_cache->seq_count : 0);
        osrs_time_log("inferno scene meshes", &t0);
    } else if (strcmp(encounter_name, "colosseum") == 0) {
        render_client->terrain = terrain_load(OSRS_ASSET("colosseum.terrain"));
        render_client->objects = objects_load(OSRS_ASSET("colosseum.objects"));
        if (render_client->terrain)
            terrain_offset(render_client->terrain, 1808, 3090);
        if (render_client->objects)
            objects_offset(render_client->objects, 1808, 3090);
        render_client->npc_model_cache = model_cache_load(OSRS_ASSET("colosseum_npcs.models"));
        render_client->npc_anim_cache = anim_cache_load(OSRS_ASSET("colosseum_npcs.anims"));
        if (env->collision_map) {
            render_client->collision_world_offset_x = 1808;
            render_client->collision_world_offset_y = 3090;
        }
        fprintf(stderr, "colosseum: terrain=%s, cmap=%s, npc_models=%d, npc_anims=%d seqs\n",
            render_client->terrain ? "loaded" : "MISSING",
            render_client->collision_map ? "loaded" : "MISSING",
            render_client->npc_model_cache ? render_client->npc_model_cache->count : 0,
            render_client->npc_anim_cache ? render_client->npc_anim_cache->seq_count : 0);
        osrs_time_log("colosseum scene meshes", &t0);
    }

    render_populate_entities(render_client, env);
    render_client->cam_target_x = (float)render_client->arena_base_x +
        (float)render_client->arena_width / 2.0f;
    render_client->cam_target_z = -((float)render_client->arena_base_y +
        (float)render_client->arena_height / 2.0f);

    for (int i = 0; i < render_client->entity_count; i++) {
        int size = render_client->entities[i].npc_size > 1
            ? render_client->entities[i].npc_size
            : 1;
        render_client->sub_x[i] = render_client->entities[i].x * 128 + size * 64;
        render_client->sub_y[i] = render_client->entities[i].y * 128 + size * 64;
        render_client->dest_x[i] = render_client->sub_x[i];
        render_client->dest_y[i] = render_client->sub_y[i];
    }

    return render_client;
}
