#ifndef FC_ACTOR_VISUAL_H
#define FC_ACTOR_VISUAL_H

#include "fc_types.h"

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

#endif /* FC_ACTOR_VISUAL_H */
