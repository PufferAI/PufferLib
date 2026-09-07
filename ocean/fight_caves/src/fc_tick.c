#include "fc_api.h"
#include "fc_combat.h"
#include "fc_prayer.h"
#include "fc_pathfinding.h"
#include <math.h>
#include "fc_npc.h"
#include "fc_wave.h"
#include "fc_contracts.h"
#include "fc_action_internal.h"
#include "fc_spawn_internal.h"

/*
 * fc_tick.c — Main tick loop for Fight Caves simulation.
 *
 * Processing order (adapted from PufferLib PvP two-phase execution):
 *
 *   1. Clear per-tick event flags
 *   2. Process player actions:
 *      a. Prayer toggle (instant)
 *      b. Eat food / drink potion (if timer ready)
 *      c. Attack initiation from the pre-movement tile
 *      d. Movement (route or directional head), unless an attack fired
 *      e. Run-energy drain or restoration from actual movement
 *   3. Decrement player timers (attack, food, potion, combo)
 *   4. Prayer drain (only if prayer stayed active across the tick boundary)
 *   5. NPC AI tick (movement + attack) for all active NPCs
 *   6. Resolve pending hits (NPC → player, player → NPC)
 *   7. Check terminal conditions
 *   8. Increment tick and lock prayer snapshots due at the new boundary
 */

/* ======================================================================== */
/* Clear per-tick flags                                                      */
/* ======================================================================== */

static void clear_per_tick_flags(FcState* state) {
    state->render_events = (FcRenderEvents){0};
    state->render_events.player_attack_target_npc_slot = -1;
    state->render_events.player_move_start_x = state->player.x;
    state->render_events.player_move_start_y = state->player.y;
    state->damage_dealt_this_tick = 0;
    state->hits_landed_this_tick = 0;
    state->damage_taken_this_tick = 0;
    state->prayer_lost_this_tick = 0;
    state->overhead_prayer_lost_this_tick = 0;
    state->tz_kih_prayer_drain_this_tick = 0;
    state->npcs_killed_this_tick = 0;
    state->respawned_jad_healers_killed_this_tick = 0;
    state->wave_just_cleared = 0;
    state->jad_damage_this_tick = 0;
    state->jad_killed = 0;
    state->correct_jad_prayer = 0;
    state->wrong_jad_prayer = 0;
    state->correct_danger_prayer = 0;
    state->wrong_danger_prayer = 0;
    state->attack_attempt_this_tick = 0;
    state->invalid_action_this_tick = 0;
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; i++) {
        state->invalid_action_class_this_tick[i] = 0;
    }
    state->movement_this_tick = 0;
    state->idle_this_tick = 0;
    state->food_used_this_tick = 0;
    state->prayer_potion_used_this_tick = 0;
    state->jad_heal_procs_this_tick = 0;
    state->npc_heal_procs_this_tick = 0;
    state->npc_heal_amount_this_tick = 0;
    state->mejkot_heal_amount_this_tick = 0;
    state->jad_heal_amount_this_tick = 0;

    FcPlayer* p = &state->player;
    p->damage_taken_this_tick = 0;
    p->hit_style_this_tick = 0;
    p->hit_source_npc_type = 0;
    p->hit_locked_prayer_this_tick = 0;
    p->hit_blocked_this_tick = 0;
    p->hit_landed_this_tick = 0;
    p->food_eaten_this_tick = 0;
    p->potion_used_this_tick = 0;
    p->prayer_changed_this_tick = 0;

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        state->npcs[i].damage_taken_this_tick = 0;
        state->npcs[i].prayer_drain_dealt_this_tick = 0;
        state->npcs[i].healing_received_this_tick = 0;
        state->npcs[i].healing_given_this_tick = 0;
        state->npcs[i].healed_by_mejkot_this_tick = 0;
        state->npcs[i].healed_by_hurkot_this_tick = 0;
        state->npcs[i].healed_self_this_tick = 0;
        state->npcs[i].died_this_tick = 0;
    }
}

static void record_player_move_waypoint(FcState* state) {
    FcRenderEvents* events = &state->render_events;
    int index = events->player_move_waypoint_count;
    if (index >= FC_MAX_RENDER_MOVE_WAYPOINTS) return;
    events->player_move_waypoint_x[index] = state->player.x;
    events->player_move_waypoint_y[index] = state->player.y;
    events->player_move_waypoint_count++;
}

static void set_player_facing_from_delta(FcPlayer* player, float dx, float dy) {
    if (dx == 0.0f && dy == 0.0f) return;
    player->facing_angle = atan2f(dx, -dy) * (180.0f / 3.14159f);
}

/* ======================================================================== */
/* Resolve NPC visible-slot index to NPC array index                         */
/* ======================================================================== */

/* Same ordering as observation writer — must be identical for consistency */
static int npc_slot_to_index(const FcState* state, int slot) {
    int visible_indices[FC_VISIBLE_NPCS];
    int visible = fc_visible_npc_indices(state, visible_indices);
    if (slot < 0 || slot >= visible) return -1;
    return visible_indices[slot];
}

/* ======================================================================== */
/* Process player actions                                                    */
/* ======================================================================== */

static void record_player_action_selection(
    FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
    int act_move = actions[0];
    int act_attack = actions[1];
    int act_prayer = actions[2];
    int invalid_classes[FC_INVALID_ACTION_CLASS_COUNT];

    if (state->npcs_remaining > 0) {
        if (act_move == FC_MOVE_IDLE) {
            state->ep_action_move_idle_ticks++;
        } else if (act_move >= FC_MOVE_WALK_N && act_move < FC_MOVE_RUN_N) {
            state->ep_action_move_walk_ticks++;
        } else if (act_move >= FC_MOVE_RUN_N && act_move < FC_MOVE_DIM) {
            state->ep_action_move_run_ticks++;
        }

        if (act_attack == FC_ATTACK_NONE) {
            state->ep_action_attack_none_ticks++;
        } else {
            state->ep_action_attack_target_ticks++;
        }

        if (act_prayer == 0) {
            state->ep_action_prayer_noop_ticks++;
        } else {
            state->ep_action_prayer_cmd_ticks++;
        }
    }

    fc_action_invalid_classes(state, actions, invalid_classes);
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; i++) {
        state->invalid_action_class_this_tick[i] = invalid_classes[i];
        if (invalid_classes[i]) {
            state->invalid_action_this_tick = 1;
            state->ep_invalid_action_classes[i]++;
        }
    }
}

static void apply_player_prayer_action(
    FcState* state, int action, FcPrayerTransition* transition) {
    FcPlayer* player = &state->player;

    *transition = fc_prayer_apply_action(player, action);
    state->render_events.prayer_prior = transition->prior_prayer;
    state->render_events.prayer_final = transition->actual_final_prayer;
    state->render_events.prayer_off_performed = transition->off_performed;
    state->render_events.prayer_on_succeeded = transition->on_succeeded;
    state->render_events.prayer_flick_performed =
        transition->explicit_off_then_on &&
        transition->off_performed &&
        transition->on_succeeded;
    if (transition->final_state_changed ||
        (transition->explicit_off_then_on &&
         transition->off_performed && transition->on_succeeded)) {
        player->prayer_changed_this_tick = 1;
    }
}

static void apply_player_supplies(FcState* state, int eat_action,
                                  int drink_action) {
    FcPlayer* player = &state->player;

    if (eat_action != FC_EAT_NONE &&
        fc_eat_action_valid(state, eat_action)) {
        int heal = eat_action == FC_EAT_SHARK ? 200 : 180;
        int* cooldown_timer = eat_action == FC_EAT_SHARK
            ? &player->food_timer : &player->combo_timer;
        int cooldown = eat_action == FC_EAT_SHARK
            ? FC_FOOD_COOLDOWN_TICKS : FC_COMBO_EAT_TICKS;
        int pre_eat_hp = player->current_hp;
        state->ep_food_pre_hp_sum += pre_eat_hp;
        int hp_missing = player->max_hp - player->current_hp;
        if (heal > hp_missing) state->ep_food_overhealed++;
        state->ep_food_eaten++;
        player->total_food_eaten++;
        player->current_hp += heal;
        if (player->current_hp > player->max_hp) {
            player->current_hp = player->max_hp;
        }
        player->sharks_remaining--;
        *cooldown_timer = cooldown;
        player->food_eaten_this_tick = 1;
        state->food_used_this_tick = 1;
    }

    if (drink_action == FC_DRINK_PRAYER_POT &&
        fc_drink_action_valid(state, drink_action)) {
        int pre_drink_prayer = player->current_prayer;
        state->ep_pot_pre_prayer_sum += pre_drink_prayer;
        int prayer_missing = player->max_prayer - player->current_prayer;
        state->ep_pots_used++;
        if (player->current_prayer > player->max_prayer / 5) {
            state->ep_pots_wasted++;
        }
        player->total_potions_used++;
        int restore = fc_prayer_potion_restore(FC_PLAYER_PRAYER_LVL);
        if (restore > prayer_missing) state->ep_pots_overrestored++;
        player->current_prayer += restore;
        if (player->current_prayer > player->max_prayer) {
            player->current_prayer = player->max_prayer;
        }
        player->prayer_doses_remaining--;
        player->potion_timer = FC_POTION_COOLDOWN_TICKS;
        player->potion_used_this_tick = 1;
        state->prayer_potion_used_this_tick = 1;
    }
}

static void prepare_player_interaction(FcState* state, int explicit_move,
                                       int explicit_attack,
                                       int requested_attack_idx) {
    FcPlayer* player = &state->player;

    /* Explicit movement starts a fresh movement intent before auto-attack can
     * consume a stale route or combat approach. */
    if (explicit_move) {
        player->route_len = 0;
        player->route_idx = 0;
        player->approach_target = 0;
        player->approach_target_x = -1;
        player->approach_target_y = -1;
        player->approach_target_size = 0;
        if (!explicit_attack) {
            player->attack_target_idx = -1;
        }
    }

    /* Target selection precedes movement so movement cannot rebind a slot or
     * make the selected attack valid retroactively. */
    if (explicit_attack && requested_attack_idx >= 0 &&
        state->npcs[requested_attack_idx].active &&
        !state->npcs[requested_attack_idx].is_dead) {
        if (player->attack_target_idx != requested_attack_idx) {
            player->approach_target_x = -1;
            player->approach_target_y = -1;
            player->approach_target_size = 0;
        }
        player->attack_target_idx = requested_attack_idx;
        player->approach_target = explicit_move ? 0 : 1;
    }
}

static void launch_player_attack(FcState* state, FcNpc* target, int distance) {
    FcPlayer* player = &state->player;
    int att_roll = fc_player_ranged_attack_roll(player, target);
    const FcNpcStats* target_stats = fc_npc_get_stats(target->npc_type);
    int def_roll = fc_npc_def_roll(target_stats->def_level,
                                   target_stats->ranged_def_bonus);
    float chance = fc_hit_chance(att_roll, def_roll);
    int hit = fc_rng_float(state) < chance ? 1 : 0;
    int final_max_hit_hp = fc_player_ranged_final_max_hit_hp(player, target);
    int damage = hit
        ? fc_roll_player_damage_tenths(state, final_max_hit_hp) : 0;
    int delay = fc_ranged_hit_delay(distance);

    fc_queue_pending_hit(target->pending_hits, &target->num_pending_hits,
                         FC_MAX_PENDING_HITS, damage, delay,
                         ATTACK_RANGED, -1, 0);
    state->attack_attempt_this_tick = 1;
    state->render_events.player_attack_fired = 1;
    state->render_events.player_attack_source_x = player->x;
    state->render_events.player_attack_source_y = player->y;
    state->render_events.player_attack_target_npc_slot =
        player->attack_target_idx;
    state->render_events.player_attack_target_x = target->x;
    state->render_events.player_attack_target_y = target->y;
    state->render_events.player_attack_target_size = target->size;
    state->render_events.player_attack_hit_delay_ticks = delay;
    if (target->npc_type > NPC_NONE && target->npc_type < NPC_TYPE_COUNT) {
        state->ep_attack_cycles_to_npc_type[target->npc_type]++;
    }
    player->attack_timer = player->weapon_speed;
    if (player->weapon_uses_ammo && player->ammo_count > 0) {
        player->ammo_count--;
    }
    player->hit_landed_this_tick = 1;
}

static void record_player_target_held(FcState* state, const FcNpc* target) {
    if (target->npc_type > NPC_NONE && target->npc_type < NPC_TYPE_COUNT) {
        state->ep_target_ticks_by_npc_type[target->npc_type]++;
    }
    state->ep_target_held_ticks++;
}

static int process_player_target(FcState* state,
                                 int explicit_directional_move,
                                 int explicit_tile_move) {
    FcPlayer* player = &state->player;
    int metrics_recorded = 0;

    /* Like Void CombatMovement: approach until the current target is in range,
     * then attack on cooldown and remain stationary for this tick. */
    if (player->attack_target_idx < 0 ||
        (player->weapon_uses_ammo && player->ammo_count <= 0)) {
        return metrics_recorded;
    }

    FcNpc* target = &state->npcs[player->attack_target_idx];
    if (!target->active || target->is_dead) {
        player->attack_target_idx = -1;
        player->approach_target = 0;
        player->approach_target_x = -1;
        player->approach_target_y = -1;
        player->approach_target_size = 0;
        return metrics_recorded;
    }

    int dist = fc_distance_to_npc(player->x, player->y, target);
    int weapon_range = player->weapon_range;
    int has_los = fc_has_los_between_areas(
        player->x, player->y, 1,
        target->x, target->y, target->size, state->los_flags);
    int target_can_fire = dist > 0 && dist <= weapon_range && has_los;
    int target_ready = player->attack_timer <= 0;

    record_player_target_held(state, target);
    metrics_recorded = 1;
    if (target_can_fire) {
        state->ep_target_in_range_los_ticks++;
        if (!target_ready) {
            state->ep_attack_cooldown_wait_ticks++;
        }
    } else {
        state->ep_target_out_of_range_or_los_ticks++;
    }

    int route_endpoint_can_fire = 0;
    int target_moved =
        player->approach_target_x != target->x ||
        player->approach_target_y != target->y ||
        player->approach_target_size != target->size;
    if (player->route_idx < player->route_len) {
        int endpoint = player->route_len - 1;
        int rx = player->route_x[endpoint];
        int ry = player->route_y[endpoint];
        int route_dist = fc_distance_between_areas(
            rx, ry, 1, target->x, target->y, target->size);
        route_endpoint_can_fire = route_dist > 0 &&
            route_dist <= weapon_range &&
            fc_has_los_between_areas(
                rx, ry, 1, target->x, target->y, target->size,
                state->los_flags);
    }

    if (!target_can_fire && player->approach_target &&
        (target_moved || player->route_idx >= player->route_len ||
         !route_endpoint_can_fire) &&
        !explicit_directional_move && !explicit_tile_move) {
        /* Rebuild against the target's current rectangle whenever the queued
         * endpoint is no longer a valid firing tile. */
        player->route_len = fc_pathfind_attack_position(
            player->x, player->y, target->x, target->y, target->size,
            weapon_range, state->walkable, state->movement_flags,
            state->los_flags, player->route_x, player->route_y, FC_MAX_ROUTE);
        player->route_idx = 0;
        player->approach_target_x = target->x;
        player->approach_target_y = target->y;
        player->approach_target_size = target->size;
    }

    float target_x = (float)target->x + (float)target->size * 0.5f;
    float target_y = (float)target->y + (float)target->size * 0.5f;
    set_player_facing_from_delta(
        player, target_x - ((float)player->x + 0.5f),
        target_y - ((float)player->y + 0.5f));

    if (target_can_fire && target_ready) {
        launch_player_attack(state, target, dist);
    }

    if (target_can_fire && target_ready &&
        !state->attack_attempt_this_tick) {
        state->ep_ready_but_no_attack_ticks++;
    }
    return metrics_recorded;
}

static int process_player_movement(FcState* state, int move_action,
                                   int target_x_action, int target_y_action,
                                   int explicit_move, int explicit_attack) {
    FcPlayer* player = &state->player;
    int moved_steps = 0;

    /* If no attack fired, explicit movement replaces the combat interaction.
     * A fired attack wins the conflict and keeps its target for this tick. */
    if (explicit_move && !state->attack_attempt_this_tick) {
        player->approach_target = 0;
        if (explicit_attack) {
            player->attack_target_idx = -1;
        }
    }

    if (!state->attack_attempt_this_tick &&
        target_x_action > 0 && target_y_action > 0) {
        int target_x = target_x_action - 1;
        int target_y = target_y_action - 1;
        if (target_x < FC_ARENA_WIDTH && target_y < FC_ARENA_HEIGHT) {
            player->route_len = fc_pathfind_bfs_move_near(
                player->x, player->y, target_x, target_y,
                state->walkable, state->movement_flags,
                player->route_x, player->route_y, FC_MAX_ROUTE);
            player->route_idx = 0;
            player->attack_target_idx = -1;
            player->approach_target = 0;
            player->approach_target_x = -1;
            player->approach_target_y = -1;
            player->approach_target_size = 0;
        }
    }

    /* Routes take priority over the directional action. Attacking suppresses
     * both forms of movement for this tick only. */
    if (state->attack_attempt_this_tick) {
        player->route_len = 0;
        player->route_idx = 0;
    } else if (player->route_idx < player->route_len) {
        int steps = player->is_running && player->run_energy > 0 ? 2 : 1;
        for (int i = 0;
             i < steps && player->route_idx < player->route_len; i++) {
            int next_x = player->route_x[player->route_idx];
            int next_y = player->route_y[player->route_idx];
            int dx = next_x - player->x;
            int dy = next_y - player->y;
            if (fc_footprint_step_walkable(
                    player->x, player->y, dx, dy, 1,
                    state->walkable, state->movement_flags)) {
                set_player_facing_from_delta(player, (float)dx, (float)dy);
                player->x = next_x;
                player->y = next_y;
                state->movement_this_tick = 1;
                record_player_move_waypoint(state);
                moved_steps++;
            } else {
                player->route_len = player->route_idx;
                break;
            }
            player->route_idx++;
        }
    } else if (move_action == FC_MOVE_IDLE) {
        state->idle_this_tick = 1;
    } else if (move_action >= FC_MOVE_WALK_N &&
               move_action <= FC_MOVE_RUN_NW) {
        int dx = FC_MOVE_DX[move_action];
        int dy = FC_MOVE_DY[move_action];
        int wants_run = move_action >= FC_MOVE_RUN_N;
        int max_steps = wants_run
            ? (fc_player_can_run(player) ? 2 : 0)
            : 1;
        int old_x = player->x;
        int old_y = player->y;
        int step_x[FC_MAX_RENDER_MOVE_WAYPOINTS];
        int step_y[FC_MAX_RENDER_MOVE_WAYPOINTS];
        int moved = fc_move_toward_traced(
            &player->x, &player->y, dx, dy, max_steps,
            state->walkable, state->movement_flags,
            step_x, step_y, FC_MAX_RENDER_MOVE_WAYPOINTS);
        if (moved > 0) {
            int recorded = moved;
            if (recorded > FC_MAX_RENDER_MOVE_WAYPOINTS) {
                recorded = FC_MAX_RENDER_MOVE_WAYPOINTS;
            }
            state->render_events.player_move_waypoint_count = recorded;
            for (int i = 0; i < recorded; i++) {
                state->render_events.player_move_waypoint_x[i] = step_x[i];
                state->render_events.player_move_waypoint_y[i] = step_y[i];
            }
            set_player_facing_from_delta(
                player, (float)(player->x - old_x),
                (float)(player->y - old_y));
            state->movement_this_tick = 1;
            player->is_running = moved >= 2 ? 1 : 0;
            moved_steps = moved;
        }
    }
    return moved_steps;
}

static void update_player_run_energy(FcPlayer* player, int moved_steps) {
    if (moved_steps >= 2) {
        player->run_energy -= FC_RUN_ENERGY_DRAIN;
        if (player->run_energy <= 0) {
            player->run_energy = 0;
            player->is_running = 0;
        }
        return;
    }

    player->run_energy += FC_RUN_ENERGY_RESTORE;
    if (player->run_energy > FC_RUN_ENERGY_MAX) {
        player->run_energy = FC_RUN_ENERGY_MAX;
    }
}

static void record_player_action_outcome(FcState* state, int was_attack_ready,
                                         int target_metrics_recorded) {
    FcPlayer* player = &state->player;

    if (state->npcs_remaining > 0 && !target_metrics_recorded) {
        if (player->attack_target_idx >= 0) {
            FcNpc* target = &state->npcs[player->attack_target_idx];
            if (target->active && !target->is_dead) {
                record_player_target_held(state, target);
            } else {
                state->ep_no_target_ticks++;
            }
        } else {
            state->ep_no_target_ticks++;
        }
    }

    if (was_attack_ready) {
        state->ep_attack_ready_ticks++;
        if (state->attack_attempt_this_tick) {
            state->ep_attack_attempt_ticks++;
        }
    }
}

static void process_player_actions(FcState* state,
                                   const int actions[FC_NUM_ACTION_HEADS],
                                   FcPrayerTransition* prayer_transition) {
    FcPlayer* p = &state->player;
    int was_attack_ready = (p->attack_timer <= 0 && state->npcs_remaining > 0);

    int act_move     = actions[0];
    int act_attack   = actions[1];
    int act_prayer   = actions[2];
    int act_eat      = actions[3];
    int act_drink    = actions[4];
    int act_target_x = actions[5];
    int act_target_y = actions[6];
    int explicit_directional_move = (act_move != FC_MOVE_IDLE);
    int explicit_tile_move = (act_target_x > 0 && act_target_y > 0);
    int explicit_move = explicit_directional_move || explicit_tile_move;
    int explicit_attack = (act_attack > FC_ATTACK_NONE);
    int requested_attack_idx = -1;
    int target_metrics_recorded = 0;
    record_player_action_selection(state, actions);

    /* Resolve attack slots against the pre-action NPC slot ordering. The action
     * was chosen from the previous observation, so movement later in this tick
     * must not rebind slot N to a different NPC identity. */
    if (explicit_attack) {
        requested_attack_idx = npc_slot_to_index(state, act_attack - 1);
    }

    /* Prayer remains instant and precedes supplies. */
    apply_player_prayer_action(state, act_prayer, prayer_transition);
    apply_player_supplies(state, act_eat, act_drink);

    prepare_player_interaction(state, explicit_move, explicit_attack,
                               requested_attack_idx);

    target_metrics_recorded = process_player_target(
        state, explicit_directional_move, explicit_tile_move);

    int moved_steps = process_player_movement(
        state, act_move, act_target_x, act_target_y,
        explicit_move, explicit_attack);
    update_player_run_energy(p, moved_steps);
    record_player_action_outcome(state, was_attack_ready,
                                 target_metrics_recorded);
}

/* ======================================================================== */
/* Decrement player timers                                                   */
/* ======================================================================== */

static void decrement_player_timers(FcPlayer* p) {
    if (p->attack_timer > 0) p->attack_timer--;
    if (p->food_timer > 0) p->food_timer--;
    if (p->potion_timer > 0) p->potion_timer--;
    if (p->combo_timer > 0) p->combo_timer--;
}

/* ======================================================================== */
/* Check terminal conditions                                                 */
/* ======================================================================== */

/* ======================================================================== */
/* Jad healer auto-spawn                                                     */
/* ======================================================================== */

/*
 * Jad healer spawn (from TzhaarFightCave.kt npcLevelChanged handler):
 *   Trigger: Jad HP drops below 150 HP.
 *   Spawns up to 4 Yt-HurKot in the five Fight Cave spawn regions other than
 *   north-east (fills missing slots: 4 - currently_alive).
 *   Respawn: Only after healers restore Jad to full HP and he crosses the
 *   threshold again. Crossing back above the threshold is not enough to re-arm.
 */
static void check_jad_healers(FcState* state) {
    if (state->current_wave != FC_NUM_WAVES) return;  /* only on wave 63 */

    /* Find Jad */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* jad = &state->npcs[i];
        if (jad->npc_type != NPC_TZTOK_JAD || !jad->active || jad->is_dead) continue;

        /* Re-arm respawns only after Jad has been healed all the way to full. */
        if (jad->current_hp >= jad->max_hp) {
            state->jad_healers_spawned = 0;
            return;
        }

        if (jad->current_hp >= FC_JAD_HEALER_THRESHOLD_HP_TENTHS) return;

        /* Below threshold — spawn healers if not already spawned this cycle */
        if (state->jad_healers_spawned) return;

        /* Count currently alive healers */
        int alive_healers = 0;
        for (int h = 0; h < FC_MAX_NPCS; h++) {
            if (state->npcs[h].active && !state->npcs[h].is_dead &&
                state->npcs[h].npc_type == NPC_YT_HURKOT) {
                alive_healers++;
            }
        }

        /* Spawn up to 4 total (fill missing slots) */
        int to_spawn = FC_JAD_NUM_HEALERS - alive_healers;
        int spawn_dirs[5] = {
            SPAWN_NORTH_WEST,
            SPAWN_SOUTH_WEST,
            SPAWN_SOUTH,
            SPAWN_SOUTH_EAST,
            SPAWN_CENTER,
        };
        for (int i = 4; i > 0; i--) {
            int j = fc_rng_int(state, i + 1);
            int tmp = spawn_dirs[i];
            spawn_dirs[i] = spawn_dirs[j];
            spawn_dirs[j] = tmp;
        }
        const FcNpcStats* healer_stats = fc_npc_get_stats(NPC_YT_HURKOT);
        int is_respawn_generation = state->jad_healer_spawn_generations > 0;
        int spawned = 0;
        for (int h = 0; h < to_spawn; h++) {
            int hx, hy;
            fc_spawn_position(spawn_dirs[h], &hx, &hy);

            if (!fc_spawn_find_available_footprint(
                    state, hx, hy, healer_stats->size, FC_ARENA_WIDTH - 1,
                    &hx, &hy)) {
                continue;
            }

            int slot = fc_spawn_npc_first_free(state, NPC_YT_HURKOT, hx, hy);
            if (slot < 0) break;
            state->npcs[slot].is_respawned_jad_healer =
                is_respawn_generation;
            state->npcs_remaining++;
            spawned++;
        }
        if (spawned > 0) {
            state->jad_healers_spawned = 1;
            state->jad_healer_spawn_generations++;
        }
        return;
    }
}

/* ======================================================================== */
/* Check terminal conditions                                                 */
/* ======================================================================== */

static void check_terminal(FcState* state) {
    if (state->terminal != TERMINAL_NONE) return;

    /* Player death */
    if (state->player.current_hp <= 0) {
        state->terminal = TERMINAL_PLAYER_DEATH;
        return;
    }

    /* Wave advancement (handles wave-clear and cave-complete) */
    fc_wave_check_advance(state);

    /* Jad healer spawn check */
    check_jad_healers(state);

    /* Tick cap */
    if (state->tick >= FC_MAX_EPISODE_TICKS) {
        state->terminal = TERMINAL_TICK_CAP;
    }
}

static void lock_pending_prayers_at_boundary(FcState* state) {
    FcPlayer* p = &state->player;
    for (int i = 0; i < p->num_pending_hits; i++) {
        FcPendingHit* hit = &p->pending_hits[i];
        if (!hit->active || hit->prayer_snapshot >= 0 ||
            hit->prayer_lock_tick < 0 ||
            state->tick < hit->prayer_lock_tick) {
            continue;
        }
        hit->prayer_snapshot = p->prayer;
    }
}

/* ======================================================================== */
/* Main tick entry point                                                     */
/* ======================================================================== */

void fc_tick(FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
    state->player.prayer_at_tick_start = state->player.prayer;
    FcPrayerTransition prayer_transition = {0};

    /* 1. Clear per-tick flags */
    clear_per_tick_flags(state);
    /* 2. Process player actions */
    process_player_actions(state, actions, &prayer_transition);

    /* 3. Decrement player timers */
    decrement_player_timers(&state->player);

    /* 4. Prayer drain */
    state->overhead_prayer_lost_this_tick =
        fc_prayer_drain_tick(&state->player,
                             state->player.prayer_at_tick_start,
                             &prayer_transition);
    state->prayer_lost_this_tick += state->overhead_prayer_lost_this_tick;

    /* 4b. HP regen (1 HP = 10 tenths every FC_HP_REGEN_INTERVAL ticks) */
    if (state->player.current_hp > 0 && state->player.current_hp < state->player.max_hp) {
        state->player.hp_regen_counter++;
        if (state->player.hp_regen_counter >= FC_HP_REGEN_INTERVAL) {
            state->player.hp_regen_counter = 0;
            state->player.current_hp += 10;  /* 1 HP in tenths */
            if (state->player.current_hp > state->player.max_hp) {
                state->player.current_hp = state->player.max_hp;
            }
        }
    }

    /* 5. NPC AI tick */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        fc_npc_tick(state, i);
    }

    /* 6. Resolve pending hits */
    fc_resolve_player_pending_hits(state);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (state->npcs[i].active) {
            fc_resolve_npc_pending_hits(state, i);
        }
    }

    /* 6b. Process death timers — dead NPCs stay visible briefly */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* n = &state->npcs[i];
        if (n->is_dead && n->active) {
            if (n->death_timer > 0) {
                n->death_timer--;
            } else {
                n->active = 0;  /* fully despawn */
            }
        }
    }

    /* 7. Check terminal */
    check_terminal(state);

    /* 8. Episode analytics */
    if (state->player.prayer_at_tick_start == PRAYER_PROTECT_MELEE)
        state->ep_ticks_pray_melee++;
    if (state->player.prayer_at_tick_start == PRAYER_PROTECT_RANGE)
        state->ep_ticks_pray_range++;
    if (state->player.prayer_at_tick_start == PRAYER_PROTECT_MAGIC)
        state->ep_ticks_pray_magic++;
    if (state->player.prayer_changed_this_tick)
        state->ep_prayer_switches++;
    if (state->current_wave >= 63)
        state->ep_reached_wave_63 = 1;

    /* 9. Increment tick */
    state->tick++;

    /* This is the pre-action boundary for the next policy decision. Jad's
     * T+2 lock must be visible before that observation and cannot be changed
     * by the action selected from it. */
    lock_pending_prayers_at_boundary(state);
}
