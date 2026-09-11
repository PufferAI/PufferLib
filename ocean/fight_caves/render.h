#ifndef FIGHT_CAVES_RENDER_H
#define FIGHT_CAVES_RENDER_H
#include "simulation.h"

/* Actor Visual */

#define FC_VISUAL_LOCAL_UNITS 128.0f
#define FC_VISUAL_CLIENT_TICK_SECONDS 0.02f
#define FC_VISUAL_PATH_CAPACITY 10
#define FC_VISUAL_ACTIVE_PATH_MAX 9

typedef enum {
    FC_VISUAL_LOCOMOTION_IDLE = 0,
    FC_VISUAL_LOCOMOTION_TURN,
    FC_VISUAL_LOCOMOTION_WALK_FORWARD,
    FC_VISUAL_LOCOMOTION_WALK_BACK,
    FC_VISUAL_LOCOMOTION_WALK_LEFT,
    FC_VISUAL_LOCOMOTION_WALK_RIGHT,
    FC_VISUAL_LOCOMOTION_RUN,
} FcVisualLocomotion;

typedef enum {
    FC_VISUAL_TARGET_NONE = 0,
    FC_VISUAL_TARGET_PLAYER,
    FC_VISUAL_TARGET_NPC,
} FcVisualTargetKind;

typedef struct {
    int active;
    int size;
    int server_tile_x;
    int server_tile_y;

    /* Persistent client-local position, in 1/128-tile RuneScape units. */
    float local_x;
    float local_y;
    float previous_local_x;
    float previous_local_y;

    int path_x[FC_VISUAL_PATH_CAPACITY];
    int path_y[FC_VISUAL_PATH_CAPACITY];
    unsigned char path_running[FC_VISUAL_PATH_CAPACITY];
    int path_count;

    float yaw_degrees;
    float desired_yaw_degrees;
    FcVisualTargetKind target_kind;
    int target_slot;
    int movement_blocked;
    int moving;
    FcVisualLocomotion locomotion;
} FcVisualActor;

typedef struct {
    FcVisualActor player;
    FcVisualActor npcs[FC_MAX_NPCS];
    float client_tick_accumulator;
    float render_alpha;
} FcVisualScene;

typedef struct {
    float x;
    float y;
    float yaw_degrees;
    int moving;
    FcVisualLocomotion locomotion;
} FcVisualPose;

void fc_visual_scene_init(FcVisualScene* scene);
void fc_visual_scene_reset_player(FcVisualScene* scene, int tile_x, int tile_y,
                                  int size, float yaw_degrees);
void fc_visual_scene_reset_npc(FcVisualScene* scene, int slot, int tile_x,
                               int tile_y, int size, float yaw_degrees);
void fc_visual_scene_deactivate_npc(FcVisualScene* scene, int slot);

void fc_visual_actor_enqueue_tile(FcVisualActor* actor, int tile_x, int tile_y,
                                  int running);
void fc_visual_actor_enqueue_transition(FcVisualActor* actor, int from_x,
                                        int from_y, int to_x, int to_y,
                                        int running);
void fc_visual_actor_set_target(FcVisualActor* actor,
                                FcVisualTargetKind target_kind,
                                int target_slot);
void fc_visual_actor_set_movement_blocked(FcVisualActor* actor, int blocked);

void fc_visual_scene_update(FcVisualScene* scene, float elapsed_seconds);
FcVisualPose fc_visual_actor_pose(const FcVisualScene* scene,
                                  const FcVisualActor* actor);
FcVisualPose fc_visual_scene_player_pose(const FcVisualScene* scene);
FcVisualPose fc_visual_scene_npc_pose(const FcVisualScene* scene, int slot);

#include "assets.h"

/* Actor Animation */

#include "raylib.h"

#include <stdint.h>

typedef struct {
    uint16_t idle_anim;
    uint16_t walk_anim;
    uint16_t walk_back_anim;
    uint16_t walk_left_anim;
    uint16_t walk_right_anim;
    uint16_t turn_anim;
    uint16_t run_anim;
    uint16_t attack_anim;
    uint32_t projectile_travel_spot;
    uint32_t projectile_launch_spot;
    uint32_t projectile_impact_spot;
    Color projectile_color;
    float projectile_radius;
    float projectile_start_height;
    float projectile_end_height;
    float projectile_launch_delay_client_ticks;
    float projectile_angle;
    float projectile_length_adjustment;
    float projectile_progress;
    float projectile_step_multiplier;
} FcPlayerVisualProfile;

typedef struct {
    FcVisualScene scene;
    AnimModelState *player_state;
    uint16_t player_sequence;
    int player_frame;
    float player_timer;
    uint16_t player_pose_sequence;
    int player_pose_frame;
    float player_pose_timer;
    uint16_t player_action_sequence;
    int player_action_frame;
    float player_action_timer;
    uint16_t player_lock_sequence;
    float player_lock_timer;
    int player_attack_target;
    float prayer_flick_timer;
    AnimModelState *npc_states[FC_MAX_NPCS];
    uint16_t npc_sequences[FC_MAX_NPCS];
    int npc_frames[FC_MAX_NPCS];
    float npc_timers[FC_MAX_NPCS];
    uint16_t npc_action_sequences[FC_MAX_NPCS];
    int npc_action_frames[FC_MAX_NPCS];
    float npc_action_timers[FC_MAX_NPCS];
    int npc_attack_styles[FC_MAX_NPCS];
    float npc_attack_timers[FC_MAX_NPCS];
    float npc_prayer_indicator_timers[FC_MAX_NPCS];
    int npc_prayer_lock_ticks[FC_MAX_NPCS];
    int previous_npc_x[FC_MAX_NPCS];
    int previous_npc_y[FC_MAX_NPCS];
    int previous_npc_active[FC_MAX_NPCS];
} FcActorAnimation;

void fc_actor_animation_init(FcActorAnimation *animation);
void fc_actor_animation_reset(FcActorAnimation *animation,
                              const FcState *state,
                              NpcModelSet *player_models,
                              int active_loadout);
void fc_actor_animation_shutdown(FcActorAnimation *animation);

void fc_actor_animation_capture_tick_start(FcActorAnimation *animation,
                                           const FcState *state);
void fc_actor_animation_ingest_tick(FcActorAnimation *animation,
                                    const FcState *state,
                                    const FcRenderEvents *events);
void fc_actor_animation_ingest_events(FcActorAnimation *animation,
                                      const FcRenderEvents *events,
                                      AnimCache *cache,
                                      int active_loadout,
                                      float tps);
void fc_actor_animation_update_scene(FcActorAnimation *animation,
                                     const FcState *state,
                                     AnimCache *cache,
                                     float tps,
                                     float dt,
                                     int advance_scene,
                                     const unsigned char deferred_deaths[FC_MAX_NPCS]);
void fc_actor_animation_update_models(FcActorAnimation *animation,
                                      const FcState *state,
                                      NpcModelSet *player_models,
                                      NpcModelSet *npc_models,
                                      AnimCache *cache,
                                      int active_loadout,
                                      float tps,
                                      float dt,
                                      const unsigned char deferred_deaths[FC_MAX_NPCS]);

const FcPlayerVisualProfile *fc_player_visual_profile(int active_loadout);
int fc_player_equipment_visual_profile(const FcPlayer *player);
NpcModelEntry *fc_actor_player_model_entry(NpcModelSet *player_models,
                                           int active_loadout);
void fc_actor_animation_upload_npc(FcActorAnimation *animation,
                                   int npc_slot,
                                   NpcModelEntry *entry);
float fc_actor_animation_scaled_dt(float tps, float dt);
float fc_actor_animation_scaled_duration(float tps, float seconds);
int fc_actor_animation_render_prayer(const FcActorAnimation *animation,
                                     const FcState *state);
int fc_actor_animation_prayer_window_active(const FcActorAnimation *animation,
                                            int npc_slot,
                                            int current_tick);
int fc_actor_animation_previous_npc_active(const FcActorAnimation *animation,
                                           int npc_slot);


/* Projectile Visual */

typedef struct {
    float source_x;
    float source_y;
    float source_z;
    float target_x;
    float target_y;
    float target_z;
    float duration;
    float angle;
    float progress;
} FcProjectilePath;

typedef struct {
    float x;
    float y;
    float z;
    float velocity_x;
    float velocity_y;
    float velocity_z;
} FcProjectileSample;

typedef struct {
    float launch_delay;
    float flight_duration;
    float total_duration;
} FcProjectileTiming;

float fc_projectile_profile_end_cycle(float launch_cycle,
                                      float length_adjustment,
                                      float step_multiplier,
                                      int tile_distance);

/* Convert the client's 30-cycle-per-game-tick projectile profile to viewer
 * seconds. The extra client cycle in flight_duration matches the client
 * endpoint convention used by RuneC's projectile sampler. */
int fc_projectile_timing_from_client_cycles(float launch_cycle,
                                            float end_cycle,
                                            float ticks_per_second,
                                            FcProjectileTiming* timing);

/* Effects are retained only until either their animation ends or the client
 * retention window closes. This prevents short spot animations from looping. */
float fc_projectile_effect_duration_seconds(float animation_client_cycles,
                                            float retain_client_cycles,
                                            float ticks_per_second);

/* Sample the client projectile curve at an absolute point in its flight.
 * The target may be replaced on every render frame to reproduce the client's
 * actor-targeted homing behavior without frame-rate-dependent integration. */
int fc_projectile_path_sample(const FcProjectilePath* path,
                              float elapsed,
                              FcProjectileSample* sample);


/* Click Feedback */

#define FC_CLICK_CROSS_FRAME_COUNT 4
#define FC_CLICK_CROSS_FRAME_SECONDS 0.10f

typedef enum {
    FC_CLICK_CROSS_NONE = 0,
    FC_CLICK_CROSS_MOVE,
    FC_CLICK_CROSS_INTERACTION,
} FcClickCrossKind;

typedef struct {
    int destination_active;
    int destination_x;
    int destination_y;

    int preview_pending;
    int preview_route_x[FC_MAX_ROUTE];
    int preview_route_y[FC_MAX_ROUTE];
    int preview_route_len;

    FcClickCrossKind cross_kind;
    float cross_screen_x;
    float cross_screen_y;
    float cross_elapsed;
} FcClickFeedback;

void fc_click_feedback_reset(FcClickFeedback* feedback);

/* Build a read-only preview with the same move-near pathfinder used by
 * fc_step(). The simulation state itself is never modified. */
void fc_click_feedback_select_move(FcClickFeedback* feedback,
                                   const FcState* state,
                                   int tile_x, int tile_y,
                                   float screen_x, float screen_y);

void fc_click_feedback_select_interaction(FcClickFeedback* feedback,
                                          float screen_x, float screen_y);

/* Hand the preview over to the route produced by the authoritative tick. */
void fc_click_feedback_accept_move_tick(FcClickFeedback* feedback,
                                        const FcState* state);

/* Clear a completed or cancelled authoritative destination. */
void fc_click_feedback_sync(FcClickFeedback* feedback,
                            const FcState* state);

void fc_click_feedback_update(FcClickFeedback* feedback,
                              float elapsed_seconds);

int fc_click_feedback_cross_frame(const FcClickFeedback* feedback);

/* Return the immediate preview while the click is buffered, then the live
 * core route after the next simulation tick accepts it. */
int fc_click_feedback_route(const FcClickFeedback* feedback,
                            const FcState* state,
                            const int** out_x, const int** out_y,
                            int* out_start, int* out_len);

#include "ui.h"

/* Combat Presentation */

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


/* Debug Overlay */

#include "raylib.h"

#define DBG_COLLISION (1 << 0)
#define DBG_LOS (1 << 1)
#define DBG_PATH (1 << 2)
#define DBG_RANGE (1 << 3)
#define DBG_ALL (DBG_COLLISION | DBG_LOS | DBG_PATH | DBG_RANGE)

void dbg_log_clear(void);
void dbg_log_tick(const FcState *state);
void debug_overlay_3d(const FcState *state, int dbg_flags);
void debug_overlay_screen(const FcState *state, Camera3D cam, int dbg_flags);
void dbg_draw_prayer_window_indicator(Vector3 world_anchor, Camera3D cam);
int dbg_draw_panel_tabs(const FcState *state,
                        const FcRewardBreakdown *reward_breakdown,
                        const FcRewardRuntime *reward_runtime,
                        int reward_config_loaded,
                        const char *reward_config_path,
                        int px, int x, int by, int pw, int dbg_tab,
                        int draw_tabs, int content_height);


/* Actor Visual */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FC_VISUAL_WALK_UNITS_PER_TICK 4.0f
#define FC_VISUAL_TURN_DEGREES_PER_TICK 5.625f

static float normalize_degrees(float degrees) {
    while (degrees >= 180.0f) degrees -= 360.0f;
    while (degrees < -180.0f) degrees += 360.0f;
    return degrees;
}

static float face_angle(float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    if (fabsf(dx) < 0.0001f && fabsf(dy) < 0.0001f) return 0.0f;
    return normalize_degrees(atan2f(dx, -dy) * (180.0f / 3.14159265358979323846f));
}

static void reset_actor(FcVisualActor* actor, int tile_x, int tile_y,
                        int size, float yaw_degrees) {
    if (!actor) return;
    memset(actor, 0, sizeof(*actor));
    actor->active = 1;
    actor->size = size > 0 ? size : 1;
    actor->server_tile_x = tile_x;
    actor->server_tile_y = tile_y;
    actor->local_x = (float)tile_x * FC_VISUAL_LOCAL_UNITS +
                     (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);
    actor->local_y = (float)tile_y * FC_VISUAL_LOCAL_UNITS +
                     (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);
    actor->previous_local_x = actor->local_x;
    actor->previous_local_y = actor->local_y;
    actor->yaw_degrees = normalize_degrees(yaw_degrees);
    actor->desired_yaw_degrees = actor->yaw_degrees;
    actor->target_kind = FC_VISUAL_TARGET_NONE;
    actor->target_slot = -1;
    actor->locomotion = FC_VISUAL_LOCOMOTION_IDLE;
}

void fc_visual_scene_init(FcVisualScene* scene) {
    if (!scene) return;
    memset(scene, 0, sizeof(*scene));
    scene->player.target_slot = -1;
    for (int i = 0; i < FC_MAX_NPCS; i++) scene->npcs[i].target_slot = -1;
    scene->render_alpha = 1.0f;
}

void fc_visual_scene_reset_player(FcVisualScene* scene, int tile_x, int tile_y,
                                  int size, float yaw_degrees) {
    if (!scene) return;
    reset_actor(&scene->player, tile_x, tile_y, size, yaw_degrees);
    scene->client_tick_accumulator = 0.0f;
    scene->render_alpha = 1.0f;
}

void fc_visual_scene_reset_npc(FcVisualScene* scene, int slot, int tile_x,
                               int tile_y, int size, float yaw_degrees) {
    if (!scene || slot < 0 || slot >= FC_MAX_NPCS) return;
    reset_actor(&scene->npcs[slot], tile_x, tile_y, size, yaw_degrees);
}

void fc_visual_scene_deactivate_npc(FcVisualScene* scene, int slot) {
    if (!scene || slot < 0 || slot >= FC_MAX_NPCS) return;
    memset(&scene->npcs[slot], 0, sizeof(scene->npcs[slot]));
    scene->npcs[slot].target_slot = -1;
}

void fc_visual_actor_enqueue_tile(FcVisualActor* actor, int tile_x, int tile_y,
                                  int running) {
    if (!actor || !actor->active) return;
    actor->server_tile_x = tile_x;
    actor->server_tile_y = tile_y;

    if (actor->path_count > 0) {
        int last = actor->path_count - 1;
        if (actor->path_x[last] == tile_x && actor->path_y[last] == tile_y) {
            actor->path_running[last] = running ? 1u : 0u;
            return;
        }
    }

    if (actor->path_count >= FC_VISUAL_ACTIVE_PATH_MAX) {
        /* Native actor queues store ten entries but keep at most nine active
         * route points. A new server step drops the oldest visual waypoint. */
        memmove(actor->path_x, actor->path_x + 1,
                (FC_VISUAL_ACTIVE_PATH_MAX - 1) * sizeof(actor->path_x[0]));
        memmove(actor->path_y, actor->path_y + 1,
                (FC_VISUAL_ACTIVE_PATH_MAX - 1) * sizeof(actor->path_y[0]));
        memmove(actor->path_running, actor->path_running + 1,
                (FC_VISUAL_ACTIVE_PATH_MAX - 1) * sizeof(actor->path_running[0]));
        actor->path_count = FC_VISUAL_ACTIVE_PATH_MAX - 1;
    }

    int next = actor->path_count++;
    actor->path_x[next] = tile_x;
    actor->path_y[next] = tile_y;
    actor->path_running[next] = running ? 1u : 0u;
}

void fc_visual_actor_enqueue_transition(FcVisualActor* actor, int from_x,
                                        int from_y, int to_x, int to_y,
                                        int running) {
    if (!actor || !actor->active) return;
    int x = from_x;
    int y = from_y;
    int dx = to_x - from_x;
    int dy = to_y - from_y;
    int steps = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    int sx = (dx > 0) - (dx < 0);
    int sy = (dy > 0) - (dy < 0);

    /* Large transitions are resets/teleports, not ordinary route updates. */
    if (steps > FC_VISUAL_PATH_CAPACITY) {
        reset_actor(actor, to_x, to_y, actor->size, actor->yaw_degrees);
        return;
    }

    for (int i = 0; i < steps; i++) {
        if (x != to_x) x += sx;
        if (y != to_y) y += sy;
        fc_visual_actor_enqueue_tile(actor, x, y, running);
    }
    if (steps == 0) {
        actor->server_tile_x = to_x;
        actor->server_tile_y = to_y;
    }
}

void fc_visual_actor_set_target(FcVisualActor* actor,
                                FcVisualTargetKind target_kind,
                                int target_slot) {
    if (!actor) return;
    actor->target_kind = target_kind;
    actor->target_slot = target_slot;
}

void fc_visual_actor_set_movement_blocked(FcVisualActor* actor, int blocked) {
    if (!actor) return;
    actor->movement_blocked = blocked ? 1 : 0;
}

static const FcVisualActor* target_actor(const FcVisualScene* scene,
                                         const FcVisualActor* actor) {
    if (!scene || !actor) return NULL;
    if (actor->target_kind == FC_VISUAL_TARGET_PLAYER)
        return scene->player.active ? &scene->player : NULL;
    if (actor->target_kind == FC_VISUAL_TARGET_NPC &&
        actor->target_slot >= 0 && actor->target_slot < FC_MAX_NPCS &&
        scene->npcs[actor->target_slot].active)
        return &scene->npcs[actor->target_slot];
    return NULL;
}

static void pop_path_front(FcVisualActor* actor) {
    if (!actor || actor->path_count <= 0) return;
    actor->path_count--;
    if (actor->path_count > 0) {
        memmove(actor->path_x, actor->path_x + 1,
                actor->path_count * sizeof(actor->path_x[0]));
        memmove(actor->path_y, actor->path_y + 1,
                actor->path_count * sizeof(actor->path_y[0]));
        memmove(actor->path_running, actor->path_running + 1,
                actor->path_count * sizeof(actor->path_running[0]));
    }
}

static void move_toward(float* value, float destination, float speed) {
    if (*value < destination) {
        *value += speed;
        if (*value > destination) *value = destination;
    } else if (*value > destination) {
        *value -= speed;
        if (*value < destination) *value = destination;
    }
}

static FcVisualLocomotion directional_locomotion(float movement_yaw,
                                                  float actor_yaw,
                                                  int fast_movement) {
    float relative = normalize_degrees(movement_yaw - actor_yaw);
    if (relative >= -45.0f && relative <= 45.0f) {
        return fast_movement ? FC_VISUAL_LOCOMOTION_RUN
                             : FC_VISUAL_LOCOMOTION_WALK_FORWARD;
    }
    if (relative > 45.0f && relative < 135.0f)
        return FC_VISUAL_LOCOMOTION_WALK_RIGHT;
    if (relative < -45.0f && relative > -135.0f)
        return FC_VISUAL_LOCOMOTION_WALK_LEFT;
    return FC_VISUAL_LOCOMOTION_WALK_BACK;
}

static void update_actor_movement(FcVisualActor* actor) {
    actor->moving = 0;
    actor->locomotion = FC_VISUAL_LOCOMOTION_IDLE;
    if (!actor->active || actor->path_count <= 0 || actor->movement_blocked)
        return;

    float dst_x = (float)actor->path_x[0] * FC_VISUAL_LOCAL_UNITS +
                  (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);
    float dst_y = (float)actor->path_y[0] * FC_VISUAL_LOCAL_UNITS +
                  (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);

    /* The native client snaps to a queued waypoint when local prediction is
     * more than two tiles out of sync, rather than gliding across the gap. */
    if (fabsf(actor->local_x - dst_x) > 256.0f ||
        fabsf(actor->local_y - dst_y) > 256.0f) {
        actor->local_x = dst_x;
        actor->local_y = dst_y;
        actor->previous_local_x = dst_x;
        actor->previous_local_y = dst_y;
        return;
    }

    float movement_yaw = face_angle(actor->local_x, actor->local_y, dst_x, dst_y);
    int running = actor->path_running[0] != 0;
    float speed = FC_VISUAL_WALK_UNITS_PER_TICK;
    if (actor->target_kind == FC_VISUAL_TARGET_NONE &&
        fabsf(normalize_degrees(movement_yaw - actor->yaw_degrees)) > 0.01f)
        speed = 2.0f;
    if (actor->path_count > 2) speed = 6.0f;
    if (actor->path_count > 3) speed = 8.0f;
    if (running) speed *= 2.0f;

    actor->desired_yaw_degrees = movement_yaw;
    actor->locomotion = directional_locomotion(
        movement_yaw, actor->yaw_degrees, running || speed >= 8.0f);
    actor->moving = 1;
    move_toward(&actor->local_x, dst_x, speed);
    move_toward(&actor->local_y, dst_y, speed);
    if (fabsf(actor->local_x - dst_x) < 0.001f &&
        fabsf(actor->local_y - dst_y) < 0.001f)
        pop_path_front(actor);
}

static void update_actor_facing(const FcVisualScene* scene,
                                FcVisualActor* actor) {
    if (!actor->active) return;
    const FcVisualActor* target = target_actor(scene, actor);
    if (target) {
        actor->desired_yaw_degrees = face_angle(
            actor->local_x, actor->local_y, target->local_x, target->local_y);
    }

    float delta = normalize_degrees(actor->desired_yaw_degrees -
                                    actor->yaw_degrees);
    if (fabsf(delta) <= FC_VISUAL_TURN_DEGREES_PER_TICK) {
        actor->yaw_degrees = actor->desired_yaw_degrees;
    } else {
        actor->yaw_degrees = normalize_degrees(
            actor->yaw_degrees +
            (delta > 0.0f ? FC_VISUAL_TURN_DEGREES_PER_TICK
                          : -FC_VISUAL_TURN_DEGREES_PER_TICK));
    }

    if (!actor->moving && fabsf(delta) > 0.01f)
        actor->locomotion = FC_VISUAL_LOCOMOTION_TURN;
}

static void update_client_tick(FcVisualScene* scene) {
    scene->player.previous_local_x = scene->player.local_x;
    scene->player.previous_local_y = scene->player.local_y;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        scene->npcs[i].previous_local_x = scene->npcs[i].local_x;
        scene->npcs[i].previous_local_y = scene->npcs[i].local_y;
    }

    update_actor_movement(&scene->player);
    for (int i = 0; i < FC_MAX_NPCS; i++)
        update_actor_movement(&scene->npcs[i]);

    update_actor_facing(scene, &scene->player);
    for (int i = 0; i < FC_MAX_NPCS; i++)
        update_actor_facing(scene, &scene->npcs[i]);
}

void fc_visual_scene_update(FcVisualScene* scene, float elapsed_seconds) {
    if (!scene || elapsed_seconds <= 0.0f) return;
    scene->client_tick_accumulator += elapsed_seconds;
    /* Avoid an unbounded catch-up loop after a debugger stop or window drag. */
    if (scene->client_tick_accumulator > 0.25f)
        scene->client_tick_accumulator = 0.25f;
    while (scene->client_tick_accumulator >= FC_VISUAL_CLIENT_TICK_SECONDS) {
        update_client_tick(scene);
        scene->client_tick_accumulator -= FC_VISUAL_CLIENT_TICK_SECONDS;
    }
    scene->render_alpha = scene->client_tick_accumulator /
                          FC_VISUAL_CLIENT_TICK_SECONDS;
}

FcVisualPose fc_visual_actor_pose(const FcVisualScene* scene,
                                  const FcVisualActor* actor) {
    FcVisualPose pose = {0};
    if (!scene || !actor || !actor->active) return pose;
    float alpha = scene->render_alpha;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    float local_x = actor->previous_local_x +
                    (actor->local_x - actor->previous_local_x) * alpha;
    float local_y = actor->previous_local_y +
                    (actor->local_y - actor->previous_local_y) * alpha;
    pose.x = local_x / FC_VISUAL_LOCAL_UNITS;
    pose.y = local_y / FC_VISUAL_LOCAL_UNITS;
    pose.yaw_degrees = actor->yaw_degrees;
    pose.moving = actor->moving;
    pose.locomotion = actor->locomotion;
    return pose;
}

FcVisualPose fc_visual_scene_player_pose(const FcVisualScene* scene) {
    return scene ? fc_visual_actor_pose(scene, &scene->player)
                 : (FcVisualPose){0};
}

FcVisualPose fc_visual_scene_npc_pose(const FcVisualScene* scene, int slot) {
    if (!scene || slot < 0 || slot >= FC_MAX_NPCS)
        return (FcVisualPose){0};
    return fc_visual_actor_pose(scene, &scene->npcs[slot]);
}

#undef FC_VISUAL_WALK_UNITS_PER_TICK
#undef FC_VISUAL_TURN_DEGREES_PER_TICK

/* Actor Animation */
#include <math.h>
#include <stdio.h>
#include <string.h>

#define POLICY_REPLAY_BASE_TPS (5.0f / 3.0f)

#define PLAYER_ANIM_HUMAN_IDLE 808
#define PLAYER_ANIM_HUMAN_WALK 819
#define PLAYER_ANIM_HUMAN_WALK_BACK 820
#define PLAYER_ANIM_HUMAN_WALK_RIGHT 821
#define PLAYER_ANIM_HUMAN_WALK_LEFT 822
#define PLAYER_ANIM_HUMAN_TURN 823
#define PLAYER_ANIM_HUMAN_RUN 824
#define PLAYER_ANIM_BOW_ATTACK 426
#define PLAYER_ANIM_XBOW_IDLE 4591
#define PLAYER_ANIM_XBOW_WALK 4226
#define PLAYER_ANIM_XBOW_RUN 4228
#define PLAYER_ANIM_XBOW_ATTACK 7552
#define PLAYER_ANIM_BLOWPIPE_ATTACK 5061
#define PLAYER_ANIM_EAT 829
#define PLAYER_ANIM_DEATH 836

#define JAD_ANIM_RANGED 2652
#define JAD_ANIM_MELEE 2655
#define JAD_ANIM_MAGIC 2656

static const FcPlayerVisualProfile PLAYER_VISUALS[FC_NUM_LOADOUTS] = {
    [FC_LOADOUT_BLACK_DHIDE_RCB] = {
        PLAYER_ANIM_XBOW_IDLE, PLAYER_ANIM_XBOW_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_XBOW_RUN, PLAYER_ANIM_XBOW_ATTACK,
        27, 0, 0, {200, 200, 50, 255}, 0.12f,
        155.0f, 146.0f, 41.0f, 5.0f, 5.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_SOTA_TBOW] = {
        PLAYER_ANIM_HUMAN_IDLE, PLAYER_ANIM_HUMAN_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_HUMAN_RUN, PLAYER_ANIM_BOW_ATTACK,
        1120, 1116, 0, {190, 120, 55, 255}, 0.13f,
        163.0f, 146.0f, 41.0f, 15.0f, 5.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_LOW_DEF_RCB] = {
        PLAYER_ANIM_XBOW_IDLE, PLAYER_ANIM_XBOW_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_XBOW_RUN, PLAYER_ANIM_XBOW_ATTACK,
        27, 0, 0, {200, 200, 50, 255}, 0.12f,
        155.0f, 146.0f, 41.0f, 5.0f, 5.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_RCB_PURE] = {
        PLAYER_ANIM_XBOW_IDLE, PLAYER_ANIM_XBOW_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_XBOW_RUN, PLAYER_ANIM_XBOW_ATTACK,
        27, 0, 0, {200, 200, 50, 255}, 0.12f,
        155.0f, 146.0f, 41.0f, 5.0f, 5.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_MSBI_PURE] = {
        PLAYER_ANIM_HUMAN_IDLE, PLAYER_ANIM_HUMAN_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_HUMAN_RUN, PLAYER_ANIM_BOW_ATTACK,
        15, 24, 0, {145, 155, 165, 255}, 0.10f,
        163.0f, 146.0f, 41.0f, 15.0f, 5.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_BLOWPIPE_PURE] = {
        PLAYER_ANIM_HUMAN_IDLE, PLAYER_ANIM_HUMAN_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_HUMAN_RUN, PLAYER_ANIM_BLOWPIPE_ATTACK,
        230, 236, 0, {115, 175, 85, 255}, 0.09f,
        163.0f, 146.0f, 32.0f, 15.0f, 0.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_ACB_ARMADYL] = {
        PLAYER_ANIM_XBOW_IDLE, PLAYER_ANIM_XBOW_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_XBOW_RUN, PLAYER_ANIM_XBOW_ATTACK,
        1468, 0, 0, {165, 210, 240, 255}, 0.12f,
        155.0f, 146.0f, 41.0f, 5.0f, 5.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_BOWFA_CRYSTAL] = {
        PLAYER_ANIM_HUMAN_IDLE, PLAYER_ANIM_HUMAN_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_HUMAN_RUN, PLAYER_ANIM_BOW_ATTACK,
        1922, 1923, 0, {120, 235, 225, 255}, 0.13f,
        163.0f, 146.0f, 41.0f, 15.0f, 5.0f, 11.0f, 5.0f,
    },
    [FC_LOADOUT_TBOW_MASORI] = {
        PLAYER_ANIM_HUMAN_IDLE, PLAYER_ANIM_HUMAN_WALK,
        PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
        PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
        PLAYER_ANIM_HUMAN_RUN, PLAYER_ANIM_BOW_ATTACK,
        1120, 1116, 0, {190, 120, 55, 255}, 0.13f,
        163.0f, 146.0f, 41.0f, 15.0f, 5.0f, 11.0f, 5.0f,
    },
};

static const uint16_t NPC_ANIM_IDLE[] = {
    0, 2618, 2624, 2624, 2631, 2636, 2642, 2650, 2636
};
static const uint16_t NPC_ANIM_WALK[] = {
    0, 2619, 2623, 2623, 2632, 2634, 2643, 2651, 2634
};
static const uint16_t NPC_ANIM_ATTACK[] = {
    0, 2621, 2625, 2625, 2628, 2637, 2644, 2655, 2637
};
static const uint16_t NPC_ANIM_DEATH[] = {
    0, 2620, 2627, 2627, 2630, 2638, 2646, 2654, 2638
};

const FcPlayerVisualProfile *fc_player_visual_profile(int active_loadout) {
    static const FcPlayerVisualProfile unarmed = {
        .idle_anim=808, .walk_anim=819, .walk_back_anim=820,
        .walk_left_anim=822, .walk_right_anim=821, .turn_anim=823,
        .run_anim=824, .attack_anim=422,
    };
    if (active_loadout == -1) return &unarmed;
    if (active_loadout < 0 || active_loadout >= FC_NUM_LOADOUTS)
        active_loadout = FC_ACTIVE_LOADOUT;
    return &PLAYER_VISUALS[active_loadout];
}

int fc_player_equipment_visual_profile(const FcPlayer *player) {
    const FcItemDef *weapon = fc_item_definition(player->equipment[FC_EQUIP_SLOT_WEAPON].item_id);
    return weapon ? weapon->visual_profile : -1;
}

NpcModelEntry *fc_actor_player_model_entry(NpcModelSet *player_models,
                                           int active_loadout) {
    if (!player_models) return NULL;
    if (active_loadout < 0 || active_loadout >= FC_NUM_LOADOUTS)
        active_loadout = FC_ACTIVE_LOADOUT;
    uint32_t model_id = FC_LOADOUTS[active_loadout].player_model_id;
    NpcModelEntry *entry = fc_npc_model_find(player_models, model_id);
    return entry && entry->loaded ? entry : NULL;
}


void fc_actor_animation_upload_npc(FcActorAnimation *animation,
                                   int npc_slot,
                                   NpcModelEntry *entry) {
    if (!animation || npc_slot < 0 || npc_slot >= FC_MAX_NPCS) return;
    fc_model_animation_upload(entry, animation->npc_states[npc_slot]);
}

static AnimSequence *advance_track(AnimCache *cache, uint16_t desired,
                                   uint16_t *current, int *frame,
                                   float *timer, float dt, int play_once) {
    if (!cache || desired == 0 || !current || !frame || !timer) return NULL;
    AnimSequence *sequence = anim_get_sequence(cache, desired);
    if (!sequence || sequence->frame_count == 0) return NULL;
    if (*current != desired) {
        *current = desired;
        *frame = 0;
        *timer = (float)sequence->frames[0].delay * 0.02f;
        if (*timer < 0.016f) *timer = 0.016f;
    }
    if (*frame < 0 || *frame >= sequence->frame_count) *frame = 0;
    *timer -= dt;
    while (*timer <= 0.0f && (!play_once || *frame < sequence->frame_count - 1)) {
        (*frame)++;
        if (*frame >= sequence->frame_count) {
            if (sequence->frame_step > 0 &&
                sequence->frame_step <= sequence->frame_count) {
                *frame -= sequence->frame_step;
            } else {
                *frame = 0;
            }
        }
        float delay = (float)sequence->frames[*frame].delay * 0.02f;
        if (delay < 0.016f) delay = 0.016f;
        *timer += delay;
    }
    if (play_once && *timer <= 0.0f) *timer = 0.016f;
    return sequence;
}

static float frame_duration(const AnimSequence *sequence, int frame) {
    if (!sequence || frame < 0 || frame >= sequence->frame_count) return 0.016f;
    float duration = (float)sequence->frames[frame].delay * 0.02f;
    return duration < 0.016f ? 0.016f : duration;
}

static float track_duration(const AnimSequence *sequence) {
    if (!sequence || sequence->frame_count == 0) return 0.0f;
    float duration = 0.0f;
    for (int i = 0; i < sequence->frame_count; i++)
        duration += frame_duration(sequence, i);
    return duration;
}

static void retarget_track(AnimCache *cache, uint16_t desired,
                           uint16_t *current, int *frame, float *timer) {
    if (!cache || desired == 0 || !current || !frame || !timer ||
        *current == 0 || *current == desired) return;
    AnimSequence *old_sequence = anim_get_sequence(cache, *current);
    AnimSequence *new_sequence = anim_get_sequence(cache, desired);
    if (!old_sequence || old_sequence->frame_count == 0 ||
        !new_sequence || new_sequence->frame_count == 0) return;
    int old_frame = *frame;
    if (old_frame < 0 || old_frame >= old_sequence->frame_count) old_frame = 0;
    float old_total = track_duration(old_sequence);
    float new_total = track_duration(new_sequence);
    if (old_total <= 0.0f || new_total <= 0.0f) return;
    float old_elapsed = 0.0f;
    for (int i = 0; i < old_frame; i++)
        old_elapsed += frame_duration(old_sequence, i);
    float old_frame_duration = frame_duration(old_sequence, old_frame);
    float remaining = *timer;
    if (remaining < 0.0f) remaining = 0.0f;
    if (remaining > old_frame_duration) remaining = old_frame_duration;
    old_elapsed += old_frame_duration - remaining;
    float target = fmodf(old_elapsed, old_total) / old_total * new_total;
    float elapsed = 0.0f;
    int new_frame = 0;
    for (; new_frame < new_sequence->frame_count - 1; new_frame++) {
        float duration = frame_duration(new_sequence, new_frame);
        if (target < elapsed + duration) break;
        elapsed += duration;
    }
    *current = desired;
    *frame = new_frame;
    *timer = elapsed + frame_duration(new_sequence, new_frame) - target;
    if (*timer < 0.001f) *timer = 0.001f;
}

static int movement_sequence(const FcPlayerVisualProfile *profile,
                             uint16_t sequence) {
    return profile && sequence != 0 &&
        (sequence == profile->walk_anim ||
         sequence == profile->walk_back_anim ||
         sequence == profile->walk_left_anim ||
         sequence == profile->walk_right_anim ||
         sequence == profile->run_anim);
}

static float sequence_duration(const AnimSequence *sequence) {
    if (!sequence || sequence->frame_count == 0) return 0.45f;
    float total = track_duration(sequence);
    return total < 0.35f ? 0.35f : total;
}

static uint16_t npc_attack_sequence(int npc_type, int attack_style) {
    if (npc_type == NPC_TZTOK_JAD) {
        if (attack_style == ATTACK_MAGIC) return JAD_ANIM_MAGIC;
        if (attack_style == ATTACK_RANGED) return JAD_ANIM_RANGED;
        if (attack_style == ATTACK_MELEE) return JAD_ANIM_MELEE;
    }
    return npc_type > 0 && npc_type < 9 ? NPC_ANIM_ATTACK[npc_type] : 0;
}

float fc_actor_animation_scaled_dt(float tps, float dt) {
    if (tps <= 0.0f) return dt;
    float scale = tps / POLICY_REPLAY_BASE_TPS;
    if (scale < 0.05f) scale = 0.05f;
    if (scale > 36.0f) scale = 36.0f;
    return dt * scale;
}

float fc_actor_animation_scaled_duration(float tps, float seconds) {
    float scale = tps > 0.0f ? tps / POLICY_REPLAY_BASE_TPS : 1.0f;
    if (scale < 0.05f) scale = 0.05f;
    if (scale > 36.0f) scale = 36.0f;
    seconds /= scale;
    return seconds < 0.05f ? 0.05f : seconds;
}

static int player_lock_active(const FcActorAnimation *animation) {
    return animation && animation->player_lock_sequence != 0 &&
           animation->player_lock_timer > 0.0f;
}

static uint16_t player_action_sequence(const FcActorAnimation *animation,
                                       const FcState *state) {
    if (!animation || !state) return 0;
    if (state->terminal == TERMINAL_PLAYER_DEATH) return PLAYER_ANIM_DEATH;
    if (state->player.food_eaten_this_tick) return PLAYER_ANIM_EAT;
    return player_lock_active(animation) ? animation->player_lock_sequence : 0;
}

static int sequence_blocks_movement(AnimCache *cache, uint16_t sequence_id) {
    if (!cache || sequence_id == 0) return 0;
    AnimSequence *sequence = anim_get_sequence(cache, sequence_id);
    return sequence && sequence->postanim_move == 0;
}

static void recreate_player_state(FcActorAnimation *animation,
                                  NpcModelEntry *entry,
                                  int active_loadout) {
    if (!animation || !entry || !entry->loaded || !entry->vertex_skins) return;
    if (animation->player_state &&
        animation->player_state->vert_count == entry->base_vert_count) return;
    if (animation->player_state) anim_model_state_free(animation->player_state);
    animation->player_state = anim_model_state_create(entry->vertex_skins,
                                                       entry->base_vert_count);
    const FcPlayerVisualProfile *profile = fc_player_visual_profile(active_loadout);
    animation->player_sequence = profile->idle_anim;
    animation->player_frame = 0;
    animation->player_timer = 0.0f;
    fprintf(stderr, "Player animation state created (%d base verts, model %u)\n",
            entry->base_vert_count, entry->model_id);
}

void fc_actor_animation_init(FcActorAnimation *animation) {
    if (!animation) return;
    memset(animation, 0, sizeof(*animation));
    animation->player_attack_target = -1;
    for (int i = 0; i < FC_MAX_NPCS; i++)
        animation->npc_prayer_lock_ticks[i] = -1;
}

void fc_actor_animation_reset(FcActorAnimation *animation,
                              const FcState *state,
                              NpcModelSet *player_models,
                              int active_loadout) {
    if (!animation || !state) return;
    fc_visual_scene_init(&animation->scene);
    fc_visual_scene_reset_player(&animation->scene, state->player.x,
                                 state->player.y, 1,
                                 state->player.facing_angle);
    const FcPlayerVisualProfile *profile = fc_player_visual_profile(
        fc_player_equipment_visual_profile(&state->player));
    animation->player_pose_sequence = profile->idle_anim;
    animation->player_pose_frame = 0;
    animation->player_pose_timer = 0.0f;
    animation->player_action_sequence = 0;
    animation->player_action_frame = 0;
    animation->player_action_timer = 0.0f;
    animation->player_lock_sequence = 0;
    animation->player_lock_timer = 0.0f;
    animation->player_attack_target = -1;
    animation->prayer_flick_timer = 0.0f;
    recreate_player_state(animation,
        fc_actor_player_model_entry(player_models, active_loadout),
        active_loadout);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        animation->previous_npc_x[i] = npc->x;
        animation->previous_npc_y[i] = npc->y;
        animation->previous_npc_active[i] = npc->active;
        if (npc->active || npc->died_this_tick) {
            fc_visual_scene_reset_npc(&animation->scene, i, npc->x, npc->y,
                                      npc->size, 0.0f);
        }
        if (animation->npc_states[i]) {
            anim_model_state_free(animation->npc_states[i]);
            animation->npc_states[i] = NULL;
        }
        animation->npc_sequences[i] = 0;
        animation->npc_frames[i] = 0;
        animation->npc_timers[i] = 0.0f;
        animation->npc_action_sequences[i] = 0;
        animation->npc_action_frames[i] = 0;
        animation->npc_action_timers[i] = 0.0f;
        animation->npc_attack_styles[i] = ATTACK_NONE;
        animation->npc_attack_timers[i] = 0.0f;
        animation->npc_prayer_indicator_timers[i] = 0.0f;
        animation->npc_prayer_lock_ticks[i] = -1;
    }
}

void fc_actor_animation_shutdown(FcActorAnimation *animation) {
    if (!animation) return;
    if (animation->player_state) anim_model_state_free(animation->player_state);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (animation->npc_states[i])
            anim_model_state_free(animation->npc_states[i]);
    }
    memset(animation, 0, sizeof(*animation));
}

void fc_actor_animation_capture_tick_start(FcActorAnimation *animation,
                                           const FcState *state) {
    if (!animation || !state) return;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        animation->previous_npc_x[i] = state->npcs[i].x;
        animation->previous_npc_y[i] = state->npcs[i].y;
        animation->previous_npc_active[i] = state->npcs[i].active;
    }
}

void fc_actor_animation_ingest_tick(FcActorAnimation *animation,
                                    const FcState *state,
                                    const FcRenderEvents *events) {
    if (!animation || !state || !events) return;
    FcVisualActor *player = &animation->scene.player;
    if (!player->active) {
        fc_visual_scene_reset_player(&animation->scene,
            events->player_move_start_x, events->player_move_start_y, 1,
            state->player.facing_angle);
    }
    int waypoint_count = events->player_move_waypoint_count;
    int running = waypoint_count > 1;
    for (int i = 0; i < waypoint_count; i++) {
        fc_visual_actor_enqueue_tile(player, events->player_move_waypoint_x[i],
                                     events->player_move_waypoint_y[i], running);
    }
    if (waypoint_count == 0 &&
        (player->server_tile_x != state->player.x ||
         player->server_tile_y != state->player.y)) {
        fc_visual_actor_enqueue_transition(player, player->server_tile_x,
            player->server_tile_y, state->player.x, state->player.y,
            state->player.is_running);
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        FcVisualActor *visual = &animation->scene.npcs[i];
        if (npc->active && !animation->previous_npc_active[i]) {
            fc_visual_scene_reset_npc(&animation->scene, i, npc->x, npc->y,
                                      npc->size, 0.0f);
        } else if ((npc->active || npc->died_this_tick) && visual->active) {
            fc_visual_actor_enqueue_transition(visual,
                animation->previous_npc_x[i], animation->previous_npc_y[i],
                npc->x, npc->y, 0);
        } else if (!npc->active && !npc->died_this_tick) {
            fc_visual_scene_deactivate_npc(&animation->scene, i);
        }
    }
}

void fc_actor_animation_ingest_events(FcActorAnimation *animation,
                                      const FcRenderEvents *events,
                                      AnimCache *cache,
                                      int active_loadout,
                                      float tps) {
    if (!animation || !events) return;
    if (events->player_attack_fired) {
        const FcPlayerVisualProfile *profile =
            fc_player_visual_profile(active_loadout);
        AnimSequence *sequence = cache
            ? anim_get_sequence(cache, profile->attack_anim) : NULL;
        animation->player_lock_sequence = profile->attack_anim;
        animation->player_lock_timer = fc_actor_animation_scaled_duration(
            tps, sequence_duration(sequence));
        animation->player_attack_target =
            events->player_attack_target_npc_slot;
        animation->player_action_sequence = 0;
        animation->player_action_frame = 0;
        animation->player_action_timer = 0.0f;
    }
    if (events->prayer_flick_performed) {
        animation->prayer_flick_timer =
            fc_actor_animation_scaled_duration(tps, 0.10f);
    }
    for (int i = 0; i < events->npc_attack_count; i++) {
        const FcRenderNpcAttack *attack = &events->npc_attacks[i];
        int slot = attack->npc_slot;
        if (slot < 0 || slot >= FC_MAX_NPCS ||
            attack->attack_style == ATTACK_NONE) continue;
        animation->npc_attack_styles[slot] = attack->attack_style;
        animation->npc_attack_timers[slot] = 1.15f;
        animation->npc_prayer_indicator_timers[slot] = 0.30f;
        animation->npc_prayer_lock_ticks[slot] = attack->prayer_lock_tick;
    }
}

static void update_targets(FcActorAnimation *animation, const FcState *state) {
    int target = player_lock_active(animation)
        ? animation->player_attack_target : state->player.attack_target_idx;
    if (target >= 0 && target < FC_MAX_NPCS &&
        animation->scene.npcs[target].active) {
        fc_visual_actor_set_target(&animation->scene.player,
                                   FC_VISUAL_TARGET_NPC, target);
    } else {
        fc_visual_actor_set_target(&animation->scene.player,
                                   FC_VISUAL_TARGET_NONE, -1);
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcVisualActor *actor = &animation->scene.npcs[i];
        if (actor->active && state->npcs[i].active) {
            int heal_target = state->npcs[i].heal_target_idx;
            if (heal_target >= 0 && heal_target < FC_MAX_NPCS &&
                heal_target != i && animation->scene.npcs[heal_target].active) {
                fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_NPC,
                                           heal_target);
            } else if (heal_target == i) {
                fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_NONE, -1);
            } else {
                fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_PLAYER, 0);
            }
        } else {
            fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_NONE, -1);
        }
    }
}

void fc_actor_animation_update_scene(FcActorAnimation *animation,
                                     const FcState *state,
                                     AnimCache *cache,
                                     float tps,
                                     float dt,
                                     int advance_scene,
                                     const unsigned char deferred_deaths[FC_MAX_NPCS]) {
    if (!animation || !state) return;
    update_targets(animation, state);
    fc_visual_actor_set_movement_blocked(&animation->scene.player,
        sequence_blocks_movement(cache,
            player_action_sequence(animation, state)));
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        uint16_t action = 0;
        if ((state->npcs[i].is_dead || state->npcs[i].died_this_tick) &&
            (!deferred_deaths || !deferred_deaths[i])) {
            int type = state->npcs[i].npc_type;
            if (type > 0 && type < 9) action = NPC_ANIM_DEATH[type];
        } else if (animation->npc_attack_timers[i] > 0.0f) {
            action = npc_attack_sequence(state->npcs[i].npc_type,
                                          animation->npc_attack_styles[i]);
        }
        fc_visual_actor_set_movement_blocked(&animation->scene.npcs[i],
                                             sequence_blocks_movement(cache, action));
    }
    float visual_dt = fc_actor_animation_scaled_dt(tps, dt);
    if (advance_scene) fc_visual_scene_update(&animation->scene, visual_dt);
    if (animation->player_lock_timer > 0.0f) {
        animation->player_lock_timer -= dt;
        if (animation->player_lock_timer <= 0.0f) {
            animation->player_lock_timer = 0.0f;
            animation->player_lock_sequence = 0;
            animation->player_attack_target = -1;
            animation->player_action_sequence = 0;
            animation->player_action_frame = 0;
            animation->player_action_timer = 0.0f;
        }
    }
    if (animation->prayer_flick_timer > 0.0f) {
        animation->prayer_flick_timer -= dt;
        if (animation->prayer_flick_timer < 0.0f)
            animation->prayer_flick_timer = 0.0f;
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (animation->npc_prayer_lock_ticks[i] >= 0 &&
            state->tick >= animation->npc_prayer_lock_ticks[i]) {
            animation->npc_prayer_indicator_timers[i] = 0.0f;
            animation->npc_prayer_lock_ticks[i] = -1;
        }
        if (animation->npc_prayer_indicator_timers[i] > 0.0f) {
            animation->npc_prayer_indicator_timers[i] -= dt;
            if (animation->npc_prayer_indicator_timers[i] < 0.0f)
                animation->npc_prayer_indicator_timers[i] = 0.0f;
        }
    }
}

static uint16_t player_pose_sequence(const FcPlayerVisualProfile *profile,
                                     FcVisualLocomotion locomotion) {
    switch (locomotion) {
        case FC_VISUAL_LOCOMOTION_TURN: return profile->turn_anim;
        case FC_VISUAL_LOCOMOTION_WALK_BACK: return profile->walk_back_anim;
        case FC_VISUAL_LOCOMOTION_WALK_LEFT: return profile->walk_left_anim;
        case FC_VISUAL_LOCOMOTION_WALK_RIGHT: return profile->walk_right_anim;
        case FC_VISUAL_LOCOMOTION_RUN: return profile->run_anim;
        case FC_VISUAL_LOCOMOTION_WALK_FORWARD: return profile->walk_anim;
        default: return profile->idle_anim;
    }
}

void fc_actor_animation_update_models(FcActorAnimation *animation,
                                      const FcState *state,
                                      NpcModelSet *player_models,
                                      NpcModelSet *npc_models,
                                      AnimCache *cache,
                                      int active_loadout,
                                      float tps,
                                      float dt,
                                      const unsigned char deferred_deaths[FC_MAX_NPCS]) {
    if (!animation || !state || !cache) return;
    float anim_dt = fc_actor_animation_scaled_dt(tps, dt);
    NpcModelEntry *player_entry =
        fc_actor_player_model_entry(player_models, active_loadout);
    int visual_profile = fc_player_equipment_visual_profile(&state->player);
    recreate_player_state(animation, player_entry, visual_profile);
    if (animation->player_state && player_entry) {
        const FcPlayerVisualProfile *profile =
            fc_player_visual_profile(visual_profile);
        FcVisualPose pose = fc_visual_scene_player_pose(&animation->scene);
        uint16_t pose_sequence = player_pose_sequence(profile, pose.locomotion);
        uint16_t action_sequence = player_action_sequence(animation, state);
        if (pose_sequence != animation->player_pose_sequence &&
            movement_sequence(profile, animation->player_pose_sequence) &&
            movement_sequence(profile, pose_sequence)) {
            retarget_track(cache, pose_sequence,
                &animation->player_pose_sequence,
                &animation->player_pose_frame,
                &animation->player_pose_timer);
        }
        AnimSequence *pose_track = advance_track(cache, pose_sequence,
            &animation->player_pose_sequence, &animation->player_pose_frame,
            &animation->player_pose_timer, anim_dt, 0);
        AnimSequence *action_track = NULL;
        if (action_sequence != 0) {
            action_track = advance_track(cache, action_sequence,
                &animation->player_action_sequence,
                &animation->player_action_frame,
                &animation->player_action_timer, anim_dt,
                player_lock_active(animation));
        } else {
            animation->player_action_sequence = 0;
            animation->player_action_frame = 0;
            animation->player_action_timer = 0.0f;
        }
        animation->player_sequence = action_track ? action_sequence : pose_sequence;
        animation->player_frame = action_track
            ? animation->player_action_frame : animation->player_pose_frame;
        if (anim_mix_pose_action(cache, animation->player_state,
                player_entry->base_verts, pose_track,
                animation->player_pose_frame, action_track,
                animation->player_action_frame)) {
            fc_model_animation_upload(player_entry, animation->player_state);
        }
    }
    if (!npc_models) return;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        if (!npc->active && !npc->died_this_tick) {
            if (animation->npc_states[i]) {
                anim_model_state_free(animation->npc_states[i]);
                animation->npc_states[i] = NULL;
            }
            animation->npc_attack_styles[i] = ATTACK_NONE;
            animation->npc_attack_timers[i] = 0.0f;
            animation->npc_prayer_indicator_timers[i] = 0.0f;
            animation->npc_prayer_lock_ticks[i] = -1;
            continue;
        }
        NpcModelEntry *entry = fc_npc_model_find(
            npc_models, fc_npc_type_to_model_id(npc->npc_type));
        if (!entry || !entry->loaded || !entry->vertex_skins) continue;
        if (!animation->npc_states[i] ||
            animation->npc_states[i]->vert_count != entry->base_vert_count) {
            if (animation->npc_states[i])
                anim_model_state_free(animation->npc_states[i]);
            animation->npc_states[i] = anim_model_state_create(
                entry->vertex_skins, entry->base_vert_count);
            animation->npc_sequences[i] =
                npc->npc_type > 0 && npc->npc_type < 9
                    ? NPC_ANIM_IDLE[npc->npc_type] : 0;
            animation->npc_frames[i] = 0;
            animation->npc_timers[i] = 0.0f;
            animation->npc_action_sequences[i] = 0;
            animation->npc_action_frames[i] = 0;
            animation->npc_action_timers[i] = 0.0f;
        }
        uint16_t pose_sequence = npc->npc_type > 0 && npc->npc_type < 9
            ? NPC_ANIM_IDLE[npc->npc_type] : 0;
        if (animation->scene.npcs[i].moving &&
            npc->npc_type > 0 && npc->npc_type < 9)
            pose_sequence = NPC_ANIM_WALK[npc->npc_type];
        uint16_t action_sequence = 0;
        if ((npc->is_dead || npc->died_this_tick) &&
            (!deferred_deaths || !deferred_deaths[i])) {
            if (npc->npc_type > 0 && npc->npc_type < 9)
                action_sequence = NPC_ANIM_DEATH[npc->npc_type];
        } else if (animation->npc_attack_timers[i] > 0.0f) {
            action_sequence = npc_attack_sequence(
                npc->npc_type, animation->npc_attack_styles[i]);
        }
        AnimSequence *pose_track = advance_track(cache, pose_sequence,
            &animation->npc_sequences[i], &animation->npc_frames[i],
            &animation->npc_timers[i], anim_dt, 0);
        AnimSequence *action_track = NULL;
        if (action_sequence != 0) {
            action_track = advance_track(cache, action_sequence,
                &animation->npc_action_sequences[i],
                &animation->npc_action_frames[i],
                &animation->npc_action_timers[i], anim_dt, 1);
        } else {
            animation->npc_action_sequences[i] = 0;
            animation->npc_action_frames[i] = 0;
            animation->npc_action_timers[i] = 0.0f;
        }
        anim_mix_pose_action(cache, animation->npc_states[i],
            entry->base_verts, pose_track, animation->npc_frames[i],
            action_track, animation->npc_action_frames[i]);
        if (animation->npc_attack_timers[i] > 0.0f) {
            animation->npc_attack_timers[i] -= anim_dt;
            if (animation->npc_attack_timers[i] <= 0.0f) {
                animation->npc_attack_timers[i] = 0.0f;
                animation->npc_attack_styles[i] = ATTACK_NONE;
            }
        }
    }
}

int fc_actor_animation_render_prayer(const FcActorAnimation *animation,
                                     const FcState *state) {
    if (!animation || !state || animation->prayer_flick_timer > 0.0f)
        return PRAYER_NONE;
    return state->player.prayer;
}

int fc_actor_animation_prayer_window_active(const FcActorAnimation *animation,
                                            int npc_slot,
                                            int current_tick) {
    if (!animation || npc_slot < 0 || npc_slot >= FC_MAX_NPCS) return 0;
    return animation->npc_prayer_indicator_timers[npc_slot] > 0.0f ||
        (animation->npc_prayer_lock_ticks[npc_slot] >= 0 &&
         current_tick < animation->npc_prayer_lock_ticks[npc_slot]);
}

int fc_actor_animation_previous_npc_active(const FcActorAnimation *animation,
                                           int npc_slot) {
    return animation && npc_slot >= 0 && npc_slot < FC_MAX_NPCS
        ? animation->previous_npc_active[npc_slot] : 0;
}

#undef POLICY_REPLAY_BASE_TPS
#undef PLAYER_ANIM_HUMAN_IDLE
#undef PLAYER_ANIM_HUMAN_WALK
#undef PLAYER_ANIM_HUMAN_WALK_BACK
#undef PLAYER_ANIM_HUMAN_WALK_RIGHT
#undef PLAYER_ANIM_HUMAN_WALK_LEFT
#undef PLAYER_ANIM_HUMAN_TURN
#undef PLAYER_ANIM_HUMAN_RUN
#undef PLAYER_ANIM_BOW_ATTACK
#undef PLAYER_ANIM_XBOW_IDLE
#undef PLAYER_ANIM_XBOW_WALK
#undef PLAYER_ANIM_XBOW_RUN
#undef PLAYER_ANIM_XBOW_ATTACK
#undef PLAYER_ANIM_BLOWPIPE_ATTACK
#undef PLAYER_ANIM_EAT
#undef PLAYER_ANIM_DEATH
#undef JAD_ANIM_RANGED
#undef JAD_ANIM_MELEE
#undef JAD_ANIM_MAGIC

/* Projectile Visual */
#include <math.h>

float fc_projectile_profile_end_cycle(float launch_cycle,
                                      float length_adjustment,
                                      float step_multiplier,
                                      int tile_distance) {
    if (tile_distance < 0) tile_distance = 0;
    float end_cycle = launch_cycle + length_adjustment +
                      step_multiplier * (float)tile_distance;
    return end_cycle > launch_cycle ? end_cycle : launch_cycle + 1.0f;
}

int fc_projectile_timing_from_client_cycles(float launch_cycle,
                                            float end_cycle,
                                            float ticks_per_second,
                                            FcProjectileTiming* timing) {
    if (!timing || ticks_per_second <= 0.0f || launch_cycle < 0.0f ||
        end_cycle <= launch_cycle)
        return 0;

    float seconds_per_client_cycle = 1.0f / (30.0f * ticks_per_second);
    timing->launch_delay = launch_cycle * seconds_per_client_cycle;
    timing->flight_duration =
        (end_cycle + 1.0f - launch_cycle) * seconds_per_client_cycle;
    timing->total_duration = timing->launch_delay + timing->flight_duration;
    return 1;
}

float fc_projectile_effect_duration_seconds(float animation_client_cycles,
                                            float retain_client_cycles,
                                            float ticks_per_second) {
    if (ticks_per_second <= 0.0f || retain_client_cycles <= 0.0f)
        return 0.0f;
    if (animation_client_cycles <= 0.0f)
        animation_client_cycles = 30.0f;
    float visible_cycles = animation_client_cycles < retain_client_cycles
        ? animation_client_cycles : retain_client_cycles;
    return visible_cycles / (30.0f * ticks_per_second);
}

int fc_projectile_path_sample(const FcProjectilePath* path,
                              float elapsed,
                              FcProjectileSample* sample) {
    if (!path || !sample || path->duration <= 0.0f)
        return 0;

    float dx = path->target_x - path->source_x;
    float dz = path->target_z - path->source_z;
    float horizontal = sqrtf(dx * dx + dz * dz);
    float direction_x = 0.0f;
    float direction_z = 1.0f;
    if (horizontal > 0.00001f) {
        direction_x = dx / horizontal;
        direction_z = dz / horizontal;
    }

    float source_x = path->source_x + direction_x * path->progress;
    float source_z = path->source_z + direction_z * path->progress;
    float velocity_x = (path->target_x - source_x) / path->duration;
    float velocity_z = (path->target_z - source_z) / path->duration;
    float horizontal_speed = sqrtf(
        velocity_x * velocity_x + velocity_z * velocity_z);
    float velocity_y = horizontal_speed *
        tanf(path->angle * (3.14159265358979323846f / 128.0f));
    float acceleration_y = 2.0f *
        (path->target_y - path->source_y - velocity_y * path->duration) /
        (path->duration * path->duration);

    float t = elapsed;
    if (t < 0.0f) t = 0.0f;
    if (t > path->duration) t = path->duration;
    sample->x = source_x + velocity_x * t;
    sample->y = path->source_y + velocity_y * t +
                0.5f * acceleration_y * t * t;
    sample->z = source_z + velocity_z * t;
    sample->velocity_x = velocity_x;
    sample->velocity_y = velocity_y + acceleration_y * t;
    sample->velocity_z = velocity_z;
    return 1;
}


/* Click Feedback */
#include <string.h>

void fc_click_feedback_reset(FcClickFeedback* feedback) {
    if (!feedback) return;
    memset(feedback, 0, sizeof(*feedback));
    feedback->destination_x = -1;
    feedback->destination_y = -1;
}

static void start_cross(FcClickFeedback* feedback, FcClickCrossKind kind,
                        float screen_x, float screen_y) {
    feedback->cross_kind = kind;
    feedback->cross_screen_x = screen_x;
    feedback->cross_screen_y = screen_y;
    feedback->cross_elapsed = 0.0f;
}

void fc_click_feedback_select_move(FcClickFeedback* feedback,
                                   const FcState* state,
                                   int tile_x, int tile_y,
                                   float screen_x, float screen_y) {
    if (!feedback || !state ||
        tile_x < 0 || tile_x >= FC_ARENA_WIDTH ||
        tile_y < 0 || tile_y >= FC_ARENA_HEIGHT) {
        return;
    }

    feedback->destination_active = 1;
    feedback->destination_x = tile_x;
    feedback->destination_y = tile_y;
    feedback->preview_pending = 1;
    feedback->preview_route_len = fc_pathfind_bfs_move_near(
        state->player.x, state->player.y, tile_x, tile_y,
        state->walkable, state->movement_flags,
        feedback->preview_route_x, feedback->preview_route_y, FC_MAX_ROUTE);
    start_cross(feedback, FC_CLICK_CROSS_MOVE, screen_x, screen_y);
}

void fc_click_feedback_select_interaction(FcClickFeedback* feedback,
                                          float screen_x, float screen_y) {
    if (!feedback) return;
    feedback->destination_active = 0;
    feedback->destination_x = -1;
    feedback->destination_y = -1;
    feedback->preview_pending = 0;
    feedback->preview_route_len = 0;
    start_cross(feedback, FC_CLICK_CROSS_INTERACTION, screen_x, screen_y);
}

void fc_click_feedback_accept_move_tick(FcClickFeedback* feedback,
                                        const FcState* state) {
    if (!feedback || !state || !feedback->preview_pending) return;
    feedback->preview_pending = 0;
    feedback->preview_route_len = 0;
    fc_click_feedback_sync(feedback, state);
}

void fc_click_feedback_sync(FcClickFeedback* feedback,
                            const FcState* state) {
    if (!feedback || !state || !feedback->destination_active ||
        feedback->preview_pending) {
        return;
    }
    if (state->player.route_idx >= state->player.route_len) {
        feedback->destination_active = 0;
        feedback->destination_x = -1;
        feedback->destination_y = -1;
    }
}

void fc_click_feedback_update(FcClickFeedback* feedback,
                              float elapsed_seconds) {
    if (!feedback || feedback->cross_kind == FC_CLICK_CROSS_NONE ||
        elapsed_seconds <= 0.0f) {
        return;
    }
    feedback->cross_elapsed += elapsed_seconds;
    if (feedback->cross_elapsed >=
        FC_CLICK_CROSS_FRAME_COUNT * FC_CLICK_CROSS_FRAME_SECONDS) {
        feedback->cross_kind = FC_CLICK_CROSS_NONE;
        feedback->cross_elapsed = 0.0f;
    }
}

int fc_click_feedback_cross_frame(const FcClickFeedback* feedback) {
    if (!feedback || feedback->cross_kind == FC_CLICK_CROSS_NONE) return -1;
    int frame = (int)(feedback->cross_elapsed / FC_CLICK_CROSS_FRAME_SECONDS);
    if (frame < 0) frame = 0;
    if (frame >= FC_CLICK_CROSS_FRAME_COUNT)
        frame = FC_CLICK_CROSS_FRAME_COUNT - 1;
    return frame;
}

int fc_click_feedback_route(const FcClickFeedback* feedback,
                            const FcState* state,
                            const int** out_x, const int** out_y,
                            int* out_start, int* out_len) {
    if (!feedback || !state || !out_x || !out_y || !out_start || !out_len ||
        !feedback->destination_active) {
        return 0;
    }

    if (feedback->preview_pending) {
        *out_x = feedback->preview_route_x;
        *out_y = feedback->preview_route_y;
        *out_start = 0;
        *out_len = feedback->preview_route_len;
        return feedback->preview_route_len > 0;
    }

    *out_x = state->player.route_x;
    *out_y = state->player.route_y;
    *out_start = state->player.route_idx;
    *out_len = state->player.route_len;
    return state->player.route_idx < state->player.route_len;
}


/* Combat Presentation */
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
    if (!profile->projectile_travel_spot) return; /* unarmed: no projectile */
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

#undef MAX_HITSPLATS
#undef MAX_PROJECTILES
#undef MAX_VISUAL_EFFECTS
#undef OSRS_HITSPLAT_SECONDS
#undef OSRS_HEALTHBAR_SECONDS
#undef POLICY_REPLAY_BASE_TPS
#undef PROJ_JAD_MAGIC_LAUNCH
#undef PROJ_TOK_XIL_SPINE
#undef PROJ_TOK_XIL_IMPACT
#undef PROJ_KET_ZEK_FIRE
#undef PROJ_KET_ZEK_IMPACT
#undef PROJ_JAD_MAGIC_TRAVEL
#undef PROJ_JAD_MAGIC_IMPACT
#undef PROJ_JAD_RANGED_IMPACT
#undef PROJ_SPOTANIM_MODEL_BASE

/* Debug Overlay */
/*
 * fc_debug_overlay.c — viewer debug tooling and overlays.
 *
 * Separate from viewer.c to isolate debug visualization from core viewer code.
 * All functions are read-only — they never modify FcState or ViewerState.
 *
 * Two entry points called from viewer.c:
 *   debug_overlay_3d()  — called inside BeginMode3D/EndMode3D (tiles, rays, ranges)
 *   debug_overlay_screen() — called after EndMode3D (screen-space overlays)
 *
 * Toggle with 'O' key (debug overlay master toggle).
 * Sub-toggles cycle with Shift+O or number keys in debug mode.
 */

#include "raylib.h"
#include "rlgl.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>

/* ======================================================================== */
/* Event log ring buffer                                                    */
/* ======================================================================== */

#define DBG_LOG_MAX_ENTRIES  128
#define DBG_LOG_MAX_MSG      80

typedef struct {
    char entries[DBG_LOG_MAX_ENTRIES][DBG_LOG_MAX_MSG];
    int tick[DBG_LOG_MAX_ENTRIES];      /* game tick when event occurred */
    Color color[DBG_LOG_MAX_ENTRIES];   /* display color per entry */
    int head;                           /* next write position */
    int count;                          /* total entries (capped at MAX) */
    int scroll_offset;                  /* for scrollable display */
} DbgEventLog;

static DbgEventLog g_dbg_log = {0};

void dbg_log_clear(void) {
    g_dbg_log.head = 0;
    g_dbg_log.count = 0;
    g_dbg_log.scroll_offset = 0;
}

static void dbg_log_event(int game_tick, Color c, const char* fmt, ...) {
    int idx = g_dbg_log.head;
    g_dbg_log.tick[idx] = game_tick;
    g_dbg_log.color[idx] = c;

    va_list args;
    va_start(args, fmt);
    vsnprintf(g_dbg_log.entries[idx], DBG_LOG_MAX_MSG, fmt, args);
    va_end(args);

    g_dbg_log.head = (g_dbg_log.head + 1) % DBG_LOG_MAX_ENTRIES;
    if (g_dbg_log.count < DBG_LOG_MAX_ENTRIES) g_dbg_log.count++;
}

/* Call once per tick to auto-generate events from state changes.
 * Reads current state and emits relevant log entries. */
void dbg_log_tick(const FcState* state) {
    int t = state->tick;
    const FcPlayer* p = &state->player;
    static const char* npc_names[] = {"?","Tz-Kih","Tz-Kek","Kek-Sm","Tok-Xil",
        "MejKot","Ket-Zek","Jad","HurKot"};
    float rwd[FC_REWARD_FEATURES];

    fc_write_reward_features(state, rwd);

    /* Player took damage */
    if (p->damage_taken_this_tick > 0) {
        dbg_log_event(t, CLITERAL(Color){255,120,120,255},
                      "Player took %d damage", p->damage_taken_this_tick / 10);
    }
    if (state->tz_kih_prayer_drain_this_tick > 0) {
        dbg_log_event(t, CLITERAL(Color){100,190,255,255},
                      "Tz-Kih drained %d Prayer",
                      (state->tz_kih_prayer_drain_this_tick + 9) / 10);
    }

    /* Player ate food */
    if (p->food_eaten_this_tick) {
        dbg_log_event(t, CLITERAL(Color){180,255,180,255},
                      "Ate shark (+20 HP) [%d left]", p->sharks_remaining);
    }

    /* Player drank potion */
    if (p->potion_used_this_tick) {
        dbg_log_event(t, CLITERAL(Color){120,180,255,255},
                      "Drank ppot (+17 pray) [%d doses left]", p->prayer_doses_remaining);
    }

    /* Player switched prayer */
    if (p->prayer_changed_this_tick) {
        static const char* pray_names[] = {"OFF","Prot Melee","Prot Range","Prot Magic"};
        dbg_log_event(t, CLITERAL(Color){255,255,100,255},
                      "Prayer -> %s", pray_names[p->prayer]);
    }

    /* Player attack attempt */
    if (rwd[FC_RWD_ATTACK_ATTEMPT] > 0.0f && p->attack_target_idx >= 0) {
        const FcNpc* tgt = &state->npcs[p->attack_target_idx];
        const char* nm = (tgt->npc_type > 0 && tgt->npc_type < 9) ? npc_names[tgt->npc_type] : "?";
        dbg_log_event(t, CLITERAL(Color){200,200,200,255},
                      "Attacked %s [%d]", nm, p->attack_target_idx);
    }

    /* NPC events */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];

        /* NPC damage taken */
        if (n->damage_taken_this_tick > 0) {
            const char* nm = (n->npc_type > 0 && n->npc_type < 9) ? npc_names[n->npc_type] : "?";
            dbg_log_event(t, CLITERAL(Color){255,200,100,255},
                          "%s[%d] took %d dmg (%d HP left)",
                          nm, i, n->damage_taken_this_tick / 10, n->current_hp / 10);
        }
        if (n->healing_received_this_tick > 0) {
            const char* nm = (n->npc_type > 0 && n->npc_type < 9) ? npc_names[n->npc_type] : "?";
            dbg_log_event(t, CLITERAL(Color){120,255,140,255},
                          "%s[%d] healed %d HP",
                          nm, i, (n->healing_received_this_tick + 9) / 10);
        }

        /* NPC died */
        if (n->died_this_tick) {
            const char* nm = (n->npc_type > 0 && n->npc_type < 9) ? npc_names[n->npc_type] : "?";
            dbg_log_event(t, CLITERAL(Color){255,80,80,255},
                          "%s[%d] KILLED", nm, i);
        }

    }

    /* Movement — log new route destination (walk-to-tile or click-to-move) */
    {
        static int prev_route_dest_x = -1, prev_route_dest_y = -1;
        if (p->route_len > 0 && p->route_idx < p->route_len) {
            int dx = p->route_x[p->route_len - 1];
            int dy = p->route_y[p->route_len - 1];
            if (dx != prev_route_dest_x || dy != prev_route_dest_y) {
                dbg_log_event(t, CLITERAL(Color){180,220,255,255},
                              "Move to tile (%d,%d) [%d steps]", dx, dy, p->route_len - p->route_idx);
                prev_route_dest_x = dx;
                prev_route_dest_y = dy;
            }
        } else {
            prev_route_dest_x = -1;
            prev_route_dest_y = -1;
        }
    }

    /* Wave events */
    if (state->wave_just_cleared) {
        dbg_log_event(t, CLITERAL(Color){100,255,255,255},
                      "Wave %d CLEARED — advancing", state->current_wave - 1);
    }

    /* Jad killed */
    if (state->jad_killed) {
        dbg_log_event(t, CLITERAL(Color){255,215,0,255}, "*** JAD DEFEATED ***");
    }

    /* Player death */
    if (state->terminal == TERMINAL_PLAYER_DEATH && p->current_hp <= 0) {
        dbg_log_event(t, CLITERAL(Color){255,0,0,255}, "*** PLAYER DIED ***");
    }

    /* Cave complete */
    if (state->terminal == TERMINAL_CAVE_COMPLETE) {
        dbg_log_event(t, CLITERAL(Color){0,255,0,255}, "*** CAVE COMPLETE — FIRE CAPE ***");
    }
}

/* Colors */
#define DBG_COL_WALK     CLITERAL(Color){  30, 180,  30, 40 }
#define DBG_COL_BLOCK    CLITERAL(Color){ 200,  30,  30, 60 }
#define DBG_COL_LOS_OK   CLITERAL(Color){  30, 255,  30, 200 }
#define DBG_COL_LOS_FAIL CLITERAL(Color){ 255,  30,  30, 200 }
#define DBG_COL_PATH     CLITERAL(Color){ 255, 255,   0, 160 }
#define DBG_COL_RANGE    CLITERAL(Color){ 100, 200, 255, 120 }
#define DBG_COL_LABEL    CLITERAL(Color){ 200, 200, 200, 255 }
#define DBG_COL_VALUE    CLITERAL(Color){ 255, 255, 100, 255 }
#define DBG_COL_GOOD     CLITERAL(Color){ 100, 255, 100, 255 }
#define DBG_COL_BAD      CLITERAL(Color){ 255, 100, 100, 255 }
#define DBG_COL_DIM      CLITERAL(Color){ 120, 120, 120, 255 }

static const char* dbg_basename(const char* path) {
    const char* slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static Color dbg_reward_color(float value) {
    if (value > 0.0001f) return DBG_COL_GOOD;
    if (value < -0.0001f) return DBG_COL_BAD;
    return DBG_COL_DIM;
}

/* ======================================================================== */
/* A. Collision / LOS / Path / Range overlays (3D)                          */
/* ======================================================================== */

/* Draw walkable/blocked tile overlay */
static void dbg_draw_collision(const FcState* state) {
    for (int tx = 0; tx < FC_ARENA_WIDTH; tx++) {
        for (int ty = 0; ty < FC_ARENA_HEIGHT; ty++) {
            Color c = state->walkable[tx][ty] ? DBG_COL_WALK : DBG_COL_BLOCK;
            DrawCube((Vector3){tx + 0.5f, 0.02f, -(ty + 0.5f)}, 0.9f, 0.02f, 0.9f, c);
        }
    }
}

/* Master 3D overlay — called inside BeginMode3D/EndMode3D.
 * Only collision tiles use 3D (they work with depth test disabled). */
void debug_overlay_3d(const FcState* state, int dbg_flags) {
    rlDisableDepthTest();
    if (dbg_flags & DBG_COLLISION) dbg_draw_collision(state);
    rlEnableDepthTest();
}

/* ======================================================================== */
/* 2D screen-space overlays (LOS, path, range) — drawn after EndMode3D     */
/* Uses GetWorldToScreen() projection so nothing can occlude them.          */
/* ======================================================================== */

static Vector2 dbg_tile_to_screen(int tx, int ty, Camera3D cam) {
    Vector3 w = { tx + 0.5f, 0.5f, -(ty + 0.5f) };
    return GetWorldToScreen(w, cam);
}

/* LOS rays — 2D projected lines from player to NPCs */
static void dbg_draw_los_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    Vector2 ps = dbg_tile_to_screen(p->x, p->y, cam);

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];
        if (!n->active || n->is_dead) continue;

        int ncx = n->x + n->size / 2;
        int ncy = n->y + n->size / 2;
        Vector2 ns = dbg_tile_to_screen(ncx, ncy, cam);

        int has_los = fc_has_los_between_areas(
            p->x, p->y, 1, n->x, n->y, n->size, state->los_flags);
        Color c = has_los ? DBG_COL_LOS_OK : DBG_COL_LOS_FAIL;
        DrawLineEx(ps, ns, 3.0f, c);
        DrawCircleV(ns, 6.0f, c);
    }
}

/* Path visualization — 2D projected dots and lines along route */
static void dbg_draw_path_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    if (p->route_idx >= p->route_len) return;

    Vector2 prev = {0};
    for (int i = p->route_idx; i < p->route_len; i++) {
        Vector2 s = dbg_tile_to_screen(p->route_x[i], p->route_y[i], cam);
        DrawCircleV(s, 4.0f, DBG_COL_PATH);
        if (i > p->route_idx) {
            DrawLineEx(prev, s, 2.0f, DBG_COL_PATH);
        }
        prev = s;
    }

    /* Destination marker — larger circle */
    if (p->route_len > 0) {
        Vector2 dest = dbg_tile_to_screen(
            p->route_x[p->route_len - 1], p->route_y[p->route_len - 1], cam);
        DrawCircleV(dest, 8.0f, DBG_COL_PATH);
        DrawCircleLinesV(dest, 12.0f, DBG_COL_PATH);
    }
}

/* Attack range ring — 2D projected Chebyshev boundary */
static void dbg_draw_range_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    int range = 7;

    for (int dx = -range; dx <= range; dx++) {
        for (int dy = -range; dy <= range; dy++) {
            int dist = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);
            if (dist == range) {
                int tx = p->x + dx;
                int ty = p->y + dy;
                if (tx >= 0 && tx < FC_ARENA_WIDTH && ty >= 0 && ty < FC_ARENA_HEIGHT) {
                    Vector2 s = dbg_tile_to_screen(tx, ty, cam);
                    DrawCircleV(s, 3.0f, DBG_COL_RANGE);
                }
            }
        }
    }
}

static void dbg_draw_run_energy(const FcState* state) {
    int energy = state->player.run_energy;
    if (energy < 0) energy = 0;
    if (energy > FC_RUN_ENERGY_MAX) energy = FC_RUN_ENERGY_MAX;

    char text[32];
    snprintf(text, sizeof(text), "Run energy: %d.%02d%%",
             energy / 100, energy % 100);
    int width = fc_osrs_measure_text(text, 9);
    DrawRectangle(6, 6, width + 10, 18,
                  CLITERAL(Color){20, 18, 14, 210});
    DrawRectangleLines(6, 6, width + 10, 18,
                       CLITERAL(Color){120, 110, 90, 255});
    fc_osrs_draw_text(text, 11, 11, 9, DBG_COL_VALUE);
}

/* Master 2D overlays for LOS/path/range — called after EndMode3D */
void debug_overlay_screen(const FcState* state, Camera3D cam, int dbg_flags) {
    if (dbg_flags & DBG_LOS)   dbg_draw_los_2d(state, cam);
    if (dbg_flags & DBG_PATH)  dbg_draw_path_2d(state, cam);
    if (dbg_flags & DBG_RANGE) dbg_draw_range_2d(state, cam);
    dbg_draw_run_energy(state);
}

void dbg_draw_prayer_window_indicator(Vector3 world_anchor, Camera3D cam) {
    Vector2 screen = GetWorldToScreen(world_anchor, cam);
    /* Keep the marker visibly separate from the centered overhead HP bar. */
    screen.x += 24.0f;
    screen.y -= 12.0f;
    int screen_width = GetScreenWidth();
    int screen_height = GetScreenHeight();
    if (screen.x < -20.0f || screen.x > (float)screen_width + 20.0f ||
        screen.y < -20.0f || screen.y > (float)screen_height + 20.0f) {
        return;
    }

    DrawCircleV(screen, 15.0f, CLITERAL(Color){70, 255, 100, 70});
    DrawCircleV(screen, 9.0f, CLITERAL(Color){70, 255, 100, 250});
    DrawCircleLines((int)screen.x, (int)screen.y, 10.0f, WHITE);
}

/* ======================================================================== */
/* Compact debug tabs (usable in any viewer panel)                          */
/* ======================================================================== */

/* Draw debug info as a tabbed panel section.
 * px = panel X, x = content X, by = Y start position, pw = panel width.
 * dbg_tab: 0=player, 1=obs, 2=mask, 3=reward, 4=log.
 * draw_tabs controls whether this helper renders its own tab selector.
 * content_height limits the log viewport; zero keeps the legacy 20 rows.
 * Returns end Y position. */
int dbg_draw_panel_tabs(const FcState* state,
                        const FcRewardBreakdown* reward_breakdown,
                        const FcRewardRuntime* reward_runtime,
                        int reward_config_loaded,
                        const char* reward_config_path,
                        int px, int x, int by, int pw, int dbg_tab,
                        int draw_tabs, int content_height) {
    char buf[256];
    int lh = 14;

    if (draw_tabs) {
        DrawLine(px+4, by, px+pw-4, by,
                 CLITERAL(Color){42,36,28,255});
        by += 2;
        static const char* dtab_labels[] = {
            "Player", "Obs", "Mask", "Reward", "Log"
        };
        int num_dtabs = 5;
        int dtab_w = (pw - 12) / num_dtabs;
        int dtab_h = 16;
        for (int t = 0; t < num_dtabs; t++) {
            int tx = px + 4 + t * dtab_w;
            int selected = (t == dbg_tab);
            Color bg = selected ? CLITERAL(Color){82,73,61,255}
                                : CLITERAL(Color){42,36,28,255};
            DrawRectangle(tx, by, dtab_w, dtab_h, bg);
            Color tc = selected ? DBG_COL_VALUE : DBG_COL_DIM;
            int tw = fc_osrs_measure_text(dtab_labels[t], 8);
            fc_osrs_draw_text(dtab_labels[t], tx + (dtab_w - tw) / 2,
                     by + 4, 8, tc);
            if (selected)
                DrawLine(tx+2, by+dtab_h-1, tx+dtab_w-2,
                         by+dtab_h-1, DBG_COL_VALUE);
        }
        by += dtab_h + 3;
    }

    /* Tab content */
    if (dbg_tab == 0) {
        /* Player state */
        const FcPlayer* p = &state->player;
        static const char* pray_str[] = { "OFF", "Melee", "Range", "Magic" };

        snprintf(buf, sizeof(buf), "Pos:(%d,%d) Face:%.0f", p->x, p->y, p->facing_angle);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "HP:%d/%d Pray:%d/%d",
                 p->current_hp/10, p->max_hp/10, p->current_prayer/10, p->max_prayer/10);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Tmr atk:%d fd:%d pot:%d",
                 p->attack_timer, p->food_timer, p->potion_timer);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Prayer:%s Drain:%d Bonus:+%d",
                 pray_str[p->prayer], p->prayer_drain_counter, p->prayer_bonus);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Shark:%d Dose:%d Ammo:%d",
                 p->sharks_remaining, p->prayer_doses_remaining, p->ammo_count);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Tgt:%d Appr:%d Route:%d/%d",
                 p->attack_target_idx, p->approach_target, p->route_idx, p->route_len);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Hits:%d Run:%d Regen:%d",
                 p->num_pending_hits, p->is_running, p->hp_regen_counter);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

        /* Active NPCs compact list */
        by += 2;
        static const char* npc_names[] = {"?","Kih","Kek","KSm","Xil","Mej","Zek","Jad","Hur"};
        for (int i = 0; i < FC_MAX_NPCS; i++) {
            const FcNpc* n = &state->npcs[i];
            if (!n->active) continue;
            const char* nm = (n->npc_type>0 && n->npc_type<9) ? npc_names[n->npc_type] : "?";
            snprintf(buf, sizeof(buf), "[%d]%s hp:%d atk:%d/%d",
                     i, nm, n->current_hp/10, n->attack_timer, n->attack_speed);
            Color c = n->is_dead ? DBG_COL_BAD : DBG_COL_DIM;
            fc_osrs_draw_text(buf, x, by, 7, c); by += lh - 2;
        }

    } else if (dbg_tab == 1) {
        /* Observation values */
        float obs[FC_OBS_SIZE];
        fc_write_obs(state, obs);
        int mbase = FC_OBS_META_START;

        snprintf(buf, sizeof(buf), "HP:%.2f Pray:%.2f X:%.2f Y:%.2f",
                 obs[FC_OBS_PLAYER_HP], obs[FC_OBS_PLAYER_PRAYER],
                 obs[FC_OBS_PLAYER_X], obs[FC_OBS_PLAYER_Y]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Atk:%.2f Run:%.2f Sharks:%.2f Dose:%.2f",
                 obs[FC_OBS_PLAYER_ATK_TIMER],
                 obs[FC_OBS_PLAYER_RUN_ENERGY],
                 obs[FC_OBS_PLAYER_SHARKS],
                 obs[FC_OBS_PLAYER_DOSES]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Pray M%.0f R%.0f G%.0f T%.2f",
                 obs[FC_OBS_PLAYER_PRAY_MEL],
                 obs[FC_OBS_PLAYER_PRAY_RNG],
                 obs[FC_OBS_PLAYER_PRAY_MAG],
                 obs[FC_OBS_PLAYER_TARGET]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "PrayDDL M%.2f R%.2f G%.2f",
                 obs[FC_OBS_PLAYER_PRAY_DDL_MEL],
                 obs[FC_OBS_PLAYER_PRAY_DDL_RNG],
                 obs[FC_OBS_PLAYER_PRAY_DDL_MAG]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "In1 M%.2f R%.2f G%.2f",
                 obs[FC_OBS_PLAYER_IN_MEL_1T],
                 obs[FC_OBS_PLAYER_IN_RNG_1T],
                 obs[FC_OBS_PLAYER_IN_MAG_1T]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "In2 M%.2f R%.2f G%.2f",
                 obs[FC_OBS_PLAYER_IN_MEL_2T],
                 obs[FC_OBS_PLAYER_IN_RNG_2T],
                 obs[FC_OBS_PLAYER_IN_MAG_2T]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Meta W%.2f Rot%.2f Rem%.2f",
                 obs[mbase + FC_OBS_META_WAVE],
                 obs[mbase + FC_OBS_META_ROTATION],
                 obs[mbase + FC_OBS_META_REMAINING]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Drain:%.2f Dmg:%.2f Clear:%.0f",
                 obs[mbase + FC_OBS_META_PRAY_DRAIN],
                 obs[mbase + FC_OBS_META_DMG_T_TICK],
                 obs[mbase + FC_OBS_META_WAVE_CLR]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "In3 M%.2f R%.2f G%.2f",
                 obs[mbase + FC_OBS_META_IN_MEL_3T],
                 obs[mbase + FC_OBS_META_IN_RNG_3T],
                 obs[mbase + FC_OBS_META_IN_MAG_3T]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Prog C%.2f W%.2f Rem%.2f NoP%.2f",
                 obs[mbase + FC_OBS_META_CAVE_PROG],
                 obs[mbase + FC_OBS_META_WAVE_PROG],
                 obs[mbase + FC_OBS_META_WORK_REM],
                 obs[mbase + FC_OBS_META_NO_PROG]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

        by += 2;
        fc_osrs_draw_text("NPC slots:", x, by, 7, DBG_COL_DIM); by += lh - 1;
        for (int s = 0; s < FC_OBS_NPC_SLOTS; s++) {
            int base = FC_OBS_NPC_START + s * FC_OBS_NPC_STRIDE;
            if (obs[base + FC_NPC_VALID] < 0.5f) continue;
            char tele = '-';
            if (obs[base + FC_NPC_TELE_MELEE]  > 0.5f) tele = 'M';
            else if (obs[base + FC_NPC_TELE_RANGED] > 0.5f) tele = 'R';
            else if (obs[base + FC_NPC_TELE_MAGIC]  > 0.5f) tele = 'A';
            snprintf(buf, sizeof(buf), "[%d] hp:%.2f d:%.2f tele:%c los:%.0f pd:%.2f ddl:%.0f/%.2f",
                     s,
                     obs[base + FC_NPC_HP],
                     obs[base + FC_NPC_DISTANCE],
                     tele,
                     obs[base + FC_NPC_LOS],
                     obs[base + FC_NPC_PENDING_STYLE],
                     obs[base + FC_NPC_PENDING_PRAYER_WINDOW],
                     obs[base + FC_NPC_PENDING_PRAYER_DEADLINE]);
            fc_osrs_draw_text(buf, x, by, 7, DBG_COL_LABEL); by += lh - 2;
        }

    } else if (dbg_tab == 2) {
        /* Action mask */
        float mask[FC_ACTION_MASK_SIZE];
        fc_write_mask(state, mask);

        fc_osrs_draw_text("MOVE:", x, by, 8, DBG_COL_DIM);
        snprintf(buf, sizeof(buf), " ");
        int len = 1;
        for (int m = 0; m < FC_MOVE_DIM && len < 120; m++)
            buf[len++] = mask[FC_MASK_MOVE_START + m] > 0.5f ? '1' : '0';
        buf[len] = '\0';
        fc_osrs_draw_text(buf, x + 30, by, 8, DBG_COL_LABEL); by += lh;

        fc_osrs_draw_text("ATK:", x, by, 8, DBG_COL_DIM);
        len = 1; buf[0] = ' ';
        for (int m = 0; m < FC_ATTACK_DIM && len < 120; m++)
            buf[len++] = mask[FC_MASK_ATTACK_START + m] > 0.5f ? '1' : '0';
        buf[len] = '\0';
        fc_osrs_draw_text(buf, x + 30, by, 8, DBG_COL_LABEL); by += lh;

        fc_osrs_draw_text("PRAY:", x, by, 8, DBG_COL_DIM);
        len = 1; buf[0] = ' ';
        for (int m = 0; m < FC_PRAYER_DIM && len < 120; m++)
            buf[len++] = mask[FC_MASK_PRAYER_START + m] > 0.5f ? '1' : '0';
        buf[len] = '\0';
        fc_osrs_draw_text(buf, x + 30, by, 8, DBG_COL_LABEL); by += lh;

        snprintf(buf, sizeof(buf), "EAT:%c%c%c  DRINK:%c%c",
                 mask[FC_MASK_EAT_START+0]>0.5f?'1':'0',
                 mask[FC_MASK_EAT_START+1]>0.5f?'1':'0',
                 mask[FC_MASK_EAT_START+2]>0.5f?'1':'0',
                 mask[FC_MASK_DRINK_START+0]>0.5f?'1':'0',
                 mask[FC_MASK_DRINK_START+1]>0.5f?'1':'0');
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

        int vx = 0, vy = 0;
        for (int m = 0; m < FC_MOVE_TARGET_X_DIM; m++)
            if (mask[FC_MASK_TARGET_X_START + m] > 0.5f) vx++;
        for (int m = 0; m < FC_MOVE_TARGET_Y_DIM; m++)
            if (mask[FC_MASK_TARGET_Y_START + m] > 0.5f) vy++;
        snprintf(buf, sizeof(buf), "TGT_X:%d/65 TGT_Y:%d/65", vx, vy);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

    } else if (dbg_tab == 3) {
        const FcRewardBreakdown* b = reward_breakdown;
        int sh = 14;
        const char* cfg_name = dbg_basename(reward_config_path);

        fc_osrs_draw_text("Training reward parity", x, by, 8, DBG_COL_VALUE); by += sh + 2;
        snprintf(buf, sizeof(buf), "cfg:%s%s",
                 cfg_name,
                 reward_config_loaded ? "" : " (defaults)");
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += sh;
        snprintf(buf, sizeof(buf), "total:%+.4f noatk_t:%d",
                 b->total,
                 reward_runtime->ticks_since_attack);
        fc_osrs_draw_text(buf, x, by, 8, dbg_reward_color(b->total)); by += sh;
        snprintf(buf, sizeof(buf), "threat:%d", b->any_threat);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += sh + 2;

        {
            struct {
                const char* name;
                float value;
            } terms[] = {
                {"dmg_dealt", b->damage_dealt},
                {"progress", b->progress},
                {"dmg_taken", b->damage_taken},
                {"npc_kill", b->npc_kill},
                {"wave_clear", b->wave_clear},
                {"jad_kill", b->jad_kill},
                {"complete", b->cave_complete},
                {"death", b->player_death},
                {"jad_ok", b->correct_jad_prayer},
                {"danger_ok", b->correct_danger_prayer},
                {"pray_lost", b->prayer_lost},
                {"pray_waste", b->unnecessary_prayer},
                {"wave_stall", b->wave_stall},
                {"no_progress", b->no_progress},
                {"no_attack", b->no_attack},
                {"jad_heal", b->jad_heal},
                {"npc_heal", b->npc_heal},
                {"invalid", b->invalid_action},
                {"tick_pen", b->tick_penalty},
            };
            int term_count = (int)(sizeof(terms) / sizeof(terms[0]));

            for (int i = 0; i < term_count; i++) {
                snprintf(buf, sizeof(buf), "%-12s %+.4f",
                         terms[i].name, terms[i].value);
                fc_osrs_draw_text(buf, x, by, 7, dbg_reward_color(terms[i].value));
                by += sh;
            }
        }
    } else if (dbg_tab == 4) {
        /* Event log — scrollable with scrollbar, most recent at top */
        int entry_h = lh - 2;
        int max_visible = content_height > 0
            ? content_height / entry_h : 20;
        if (max_visible < 1) max_visible = 1;
        int total = g_dbg_log.count;
        int log_h = max_visible * entry_h;
        int scrollbar_w = 10;
        if (total == 0) {
            fc_osrs_draw_text("No events yet", x, by, 8, DBG_COL_DIM);
            by += lh;
        } else {
            int max_scroll = total - max_visible;
            if (max_scroll < 0) max_scroll = 0;

            /* Clamp scroll */
            if (g_dbg_log.scroll_offset > max_scroll)
                g_dbg_log.scroll_offset = max_scroll;
            if (g_dbg_log.scroll_offset < 0)
                g_dbg_log.scroll_offset = 0;

            /* Mouse wheel scroll */
            float wheel = GetMouseWheelMove();
            if (wheel != 0.0f) {
                Vector2 mp = GetMousePosition();
                if (mp.x >= px && mp.x < px + pw &&
                    mp.y >= by && mp.y < by + log_h) {
                    g_dbg_log.scroll_offset -= (int)wheel * 3;
                    if (g_dbg_log.scroll_offset < 0) g_dbg_log.scroll_offset = 0;
                    if (g_dbg_log.scroll_offset > max_scroll) g_dbg_log.scroll_offset = max_scroll;
                }
            }

            /* Scrollbar track */
            int sb_x = px + pw - scrollbar_w - 4;
            int sb_y = by;
            DrawRectangle(sb_x, sb_y, scrollbar_w, log_h, CLITERAL(Color){30,26,20,255});

            /* Scrollbar thumb */
            if (total > max_visible) {
                float thumb_frac = (float)max_visible / (float)total;
                int thumb_h = (int)(log_h * thumb_frac);
                if (thumb_h < 12) thumb_h = 12;
                float scroll_frac = (max_scroll > 0) ? (float)g_dbg_log.scroll_offset / (float)max_scroll : 0;
                int thumb_y = sb_y + (int)((log_h - thumb_h) * scroll_frac);
                DrawRectangle(sb_x, thumb_y, scrollbar_w, thumb_h, CLITERAL(Color){120,110,90,255});

                /* Drag scrollbar with mouse */
                if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                    Vector2 mp = GetMousePosition();
                    if (mp.x >= sb_x && mp.x < sb_x + scrollbar_w &&
                        mp.y >= sb_y && mp.y < sb_y + log_h) {
                        float click_frac = (mp.y - sb_y) / (float)log_h;
                        g_dbg_log.scroll_offset = (int)(click_frac * (float)(max_scroll + 1));
                        if (g_dbg_log.scroll_offset > max_scroll) g_dbg_log.scroll_offset = max_scroll;
                    }
                }
            }

            /* Draw entries */
            int drawn = 0;
            for (int i = g_dbg_log.scroll_offset; i < total && drawn < max_visible; i++) {
                int idx = (g_dbg_log.head - 1 - i + DBG_LOG_MAX_ENTRIES * 2) % DBG_LOG_MAX_ENTRIES;
                snprintf(buf, sizeof(buf), "t%d %s", g_dbg_log.tick[idx], g_dbg_log.entries[idx]);
                /* Truncate to fit content width */
                fc_osrs_draw_text(buf, x, by, 7, g_dbg_log.color[idx]);
                by += entry_h;
                drawn++;
            }

            /* Pad remaining space if fewer entries than max_visible */
            by += (max_visible - drawn) * entry_h;
        }
    }

    return by;
}

#undef DBG_LOG_MAX_ENTRIES
#undef DBG_LOG_MAX_MSG
#undef DBG_COL_WALK
#undef DBG_COL_BLOCK
#undef DBG_COL_LOS_OK
#undef DBG_COL_LOS_FAIL
#undef DBG_COL_PATH
#undef DBG_COL_RANGE
#undef DBG_COL_LABEL
#undef DBG_COL_VALUE
#undef DBG_COL_GOOD
#undef DBG_COL_BAD
#undef DBG_COL_DIM

#endif
