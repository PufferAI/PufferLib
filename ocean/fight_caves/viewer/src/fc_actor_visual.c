#include "fc_actor_visual.h"

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
