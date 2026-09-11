#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#define OSRS_VISUAL

#include <stdlib.h>

#include "osrs_puffer_render.h"
#include "osrs_render_scene.h"

typedef struct {
    OsrsEnv env;
    int has_drawn;
    int last_tick;
    double tick_anchor;
} OsrsPufferRenderer;

void* osrs_puffer_render_create(
    const void* encounter_def,
    void* encounter_state,
    void* encounter_context
) {
    const EncounterDef* def = (const EncounterDef*)encounter_def;
    OsrsPufferRenderer* renderer = (OsrsPufferRenderer*)calloc(1, sizeof(*renderer));
    if (renderer == NULL) abort();

    renderer->env.encounter_def = def;
    renderer->env.encounter_state = encounter_state;
    renderer->env.encounter_context = encounter_context;
    renderer->env.tick = def->get_tick(encounter_state, encounter_context);
    renderer->last_tick = renderer->env.tick;
    visual_load_encounter_collision_map(def, &renderer->env, def->name);
    visual_init_render_scene(&renderer->env, def->name, NULL);
    return renderer;
}

void osrs_puffer_render_draw(void* opaque_renderer) {
    OsrsPufferRenderer* renderer = (OsrsPufferRenderer*)opaque_renderer;
    const EncounterDef* def = (const EncounterDef*)renderer->env.encounter_def;
    int tick = def->get_tick(
        renderer->env.encounter_state,
        renderer->env.encounter_context);
    renderer->env.tick = tick;
    RenderClient* render_client = (RenderClient*)renderer->env.client;
    if (renderer->has_drawn) {
        if (tick < renderer->last_tick) {
            render_reset_episode_visual_state(render_client, &renderer->env);
        } else if (tick > renderer->last_tick) {
            render_post_tick(render_client, &renderer->env);
        }
    }
    pvp_render(&renderer->env);
    renderer->has_drawn = 1;
    renderer->last_tick = tick;

    // Eval steps once per puf_render. Hold here at game-tick rate so the
    // next rollout matches the CPU viewer (1/0.6s, 9/0 to change speed).
    if (renderer->tick_anchor <= 0.0)
        renderer->tick_anchor = GetTime();
    while (!WindowShouldClose()) {
        double interval = 1.0 / (double)render_effective_ticks_per_second(
            render_client);
        if (GetTime() - renderer->tick_anchor >= interval)
            break;
        pvp_render(&renderer->env);
    }
    double interval = 1.0 / (double)render_effective_ticks_per_second(
        render_client);
    renderer->tick_anchor += interval;
    if (GetTime() - renderer->tick_anchor >= interval)
        renderer->tick_anchor = GetTime();
}

void osrs_puffer_render_destroy(void* opaque_renderer) {
    OsrsPufferRenderer* renderer = (OsrsPufferRenderer*)opaque_renderer;
    render_destroy_client((RenderClient*)renderer->env.client);
    collision_map_free((CollisionMap*)renderer->env.collision_map);
    free(renderer);
}
