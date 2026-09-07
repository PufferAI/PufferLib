#ifndef FC_ACTOR_ANIMATION_H
#define FC_ACTOR_ANIMATION_H

#include "fc_actor_visual.h"
#include "fc_anim_loader.h"
#include "fc_npc_models.h"
#include "fc_types.h"
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

#endif
