#include "fc_actor_animation.h"

#include "fc_contracts.h"
#include "fc_model_animation.h"
#include "fc_models.h"

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
    if (active_loadout < 0 || active_loadout >= FC_NUM_LOADOUTS)
        active_loadout = FC_ACTIVE_LOADOUT;
    return &PLAYER_VISUALS[active_loadout];
}

NpcModelEntry *fc_actor_player_model_entry(NpcModelSet *player_models,
                                           int active_loadout) {
    if (!player_models) return NULL;
    if (active_loadout < 0 || active_loadout >= FC_NUM_LOADOUTS)
        active_loadout = FC_ACTIVE_LOADOUT;
    uint32_t model_id = FC_LOADOUTS[active_loadout].player_model_id;
    NpcModelEntry *entry = fc_npc_model_find(player_models, model_id);
    if (!entry && player_models->count > 0)
        entry = &player_models->entries[0];
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
    const FcPlayerVisualProfile *profile = fc_player_visual_profile(active_loadout);
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
    recreate_player_state(animation, player_entry, active_loadout);
    if (animation->player_state && player_entry) {
        const FcPlayerVisualProfile *profile =
            fc_player_visual_profile(active_loadout);
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
