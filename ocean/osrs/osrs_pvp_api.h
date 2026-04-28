/**
 * @file osrs_pvp_api.h
 * @brief Public API for OSRS PvP simulation
 *
 * Provides the public interface for:
 * - Player initialization
 * - Environment reset (pvp_reset)
 * - Environment step (pvp_step)
 * - Seeding for deterministic runs (pvp_seed)
 * - Cleanup (pvp_close)
 */

#ifndef OSRS_PVP_API_H
#define OSRS_PVP_API_H

#include "osrs_types.h"
#include "osrs_pvp_gear.h"
#include "osrs_pvp_combat.h"
#include "osrs_pvp_movement.h"
#include "osrs_pvp_observations.h"
#include "osrs_pvp_actions.h"

/**
 * Initialize a player with default pure build stats and gear.
 *
 * Sets up:
 * - Base stats (75 attack, 99 strength, etc.)
 * - Current stats equal to base
 * - Starting gear (mage setup)
 * - Consumables (food, brews, restores)
 * - All timers reset to 0
 * - Sequential mode equipment state
 *
 * @param p Player to initialize
 */
static void init_player(Player* p) {
    p->base_attack = MAXED_BASE_ATTACK;
    p->base_strength = MAXED_BASE_STRENGTH;
    p->base_defence = MAXED_BASE_DEFENCE;
    p->base_ranged = MAXED_BASE_RANGED;
    p->base_magic = MAXED_BASE_MAGIC;
    p->base_prayer = MAXED_BASE_PRAYER;
    p->base_hitpoints = MAXED_BASE_HITPOINTS;

    p->current_attack = p->base_attack;
    p->current_strength = p->base_strength;
    p->current_defence = p->base_defence;
    p->current_ranged = p->base_ranged;
    p->current_magic = p->base_magic;
    p->current_prayer = p->base_prayer;
    p->current_hitpoints = p->base_hitpoints;

    p->special_energy = 100;
    p->spec_regen_active = 0;
    p->spec_armed = 0;
    osrs_interaction_init(&p->interaction);
    osrs_item_effect_state_init(&p->item_effect_state);

    p->current_gear = GEAR_MAGE;
    p->visible_gear = GEAR_MAGE;

    p->food_count = MAXED_FOOD_COUNT;
    p->karambwan_count = MAXED_KARAMBWAN_COUNT;
    p->brew_doses = MAXED_BREW_DOSES;
    p->restore_doses = MAXED_RESTORE_DOSES;
    p->combat_potion_doses = MAXED_COMBAT_POTION_DOSES;
    p->ranged_potion_doses = MAXED_RANGED_POTION_DOSES;

    p->attack_timer = 0;
    p->attack_timer_uncapped = 0;
    p->has_attack_timer = 0;
    p->food_timer = 0;
    p->potion_timer = 0;
    p->karambwan_timer = 0;
    p->consumable_used_this_tick = 0;
    p->last_food_heal = 0;
    p->last_food_waste = 0;
    p->last_karambwan_heal = 0;
    p->last_karambwan_waste = 0;
    p->last_brew_heal = 0;
    p->last_brew_waste = 0;
    p->last_potion_type = 0;
    p->last_potion_was_waste = 0;

    p->frozen_ticks = 0;
    p->freeze_immunity_ticks = 0;

    p->veng_active = 0;
    p->veng_cooldown = 0;

    p->prayer = PRAYER_NONE;
    p->offensive_prayer = OFFENSIVE_PRAYER_NONE;
    p->fight_style = FIGHT_STYLE_ACCURATE;
    p->prayer_drain_counter = 0;
    p->morr_dot_remaining = 0;
    p->morr_dot_tick_counter = 0;

    p->x = 0;
    p->y = 0;
    p->dest_x = 0;
    p->dest_y = 0;
    p->is_moving = 0;
    p->is_running = 0;
    p->run_energy = 100;
    p->run_recovery_ticks = 0;
    p->last_obs_target_x = 0;
    p->last_obs_target_y = 0;

    p->just_attacked = 0;
    p->last_attack_style = ATTACK_STYLE_NONE;
    p->attack_was_on_prayer = 0;
    p->last_attack_dx = 0;
    p->last_attack_dy = 0;
    p->last_attack_dist = 0;
    p->attack_click_canceled = 0;
    p->attack_click_ready = 0;

    memset(p->pending_hits, 0, sizeof(p->pending_hits));
    p->num_pending_hits = 0;
    p->damage_applied_this_tick = 0;
    p->did_attack_auto_move = 0;

    // Hit event tracking
    p->hit_landed_this_tick = 0;
    p->hit_was_successful = 0;
    p->hit_damage = 0;
    p->hit_style = ATTACK_STYLE_NONE;
    p->hit_defender_prayer = PRAYER_NONE;
    p->hit_was_on_prayer = 0;
    p->hit_attacker_idx = -1;
    p->freeze_applied_this_tick = 0;

    p->last_target_health_percent = 0.0f;
    p->tick_damage_scale = 0.0f;
    p->damage_dealt_scale = 0.0f;
    p->damage_received_scale = 0.0f;

    p->total_target_hit_count = 0;
    p->target_hit_melee_count = 0;
    p->target_hit_ranged_count = 0;
    p->target_hit_magic_count = 0;
    p->target_hit_off_prayer_count = 0;
    p->target_hit_correct_count = 0;

    p->total_target_pray_count = 0;
    p->target_pray_melee_count = 0;
    p->target_pray_ranged_count = 0;
    p->target_pray_magic_count = 0;
    p->target_pray_correct_count = 0;

    p->player_hit_melee_count = 0;
    p->player_hit_ranged_count = 0;
    p->player_hit_magic_count = 0;

    p->player_pray_melee_count = 0;
    p->player_pray_ranged_count = 0;
    p->player_pray_magic_count = 0;

    memset(p->recent_target_attack_styles, 0, sizeof(p->recent_target_attack_styles));
    memset(p->recent_player_attack_styles, 0, sizeof(p->recent_player_attack_styles));
    memset(p->recent_target_prayer_styles, 0, sizeof(p->recent_target_prayer_styles));
    memset(p->recent_player_prayer_styles, 0, sizeof(p->recent_player_prayer_styles));
    memset(p->recent_target_prayer_correct, 0, sizeof(p->recent_target_prayer_correct));
    memset(p->recent_target_hit_correct, 0, sizeof(p->recent_target_hit_correct));
    p->recent_target_attack_index = 0;
    p->recent_player_attack_index = 0;
    p->recent_target_prayer_index = 0;
    p->recent_player_prayer_index = 0;
    p->recent_target_prayer_correct_index = 0;
    p->recent_target_hit_correct_index = 0;

    p->target_magic_accuracy = -1;
    p->target_magic_strength = -1;
    p->target_ranged_accuracy = -1;
    p->target_ranged_strength = -1;
    p->target_melee_accuracy = -1;
    p->target_melee_strength = -1;
    p->target_magic_gear_magic_defence = -1;
    p->target_magic_gear_ranged_defence = -1;
    p->target_magic_gear_melee_defence = -1;
    p->target_ranged_gear_magic_defence = -1;
    p->target_ranged_gear_ranged_defence = -1;
    p->target_ranged_gear_melee_defence = -1;
    p->target_melee_gear_magic_defence = -1;
    p->target_melee_gear_ranged_defence = -1;
    p->target_melee_gear_melee_defence = -1;

    p->player_prayed_correct = 0;
    p->target_prayed_correct = 0;

    p->total_damage_dealt = 0;
    p->total_damage_received = 0;

    p->is_lunar_spellbook = 0;
    p->observed_target_lunar_spellbook = 0;
    p->has_blood_fury = 1;

    p->melee_spec_weapon = MELEE_SPEC_NONE;
    p->ranged_spec_weapon = RANGED_SPEC_NONE;
    p->magic_spec_weapon = MAGIC_SPEC_NONE;

    p->bolt_proc_damage = 0.2f;
    p->bolt_ignores_defense = 0;

    p->prev_hp_percent = 1.0f;  // Full HP at start
}

/**
 * Set initial fight positions for both players.
 *
 * In seeded mode: deterministic positions.
 * Otherwise: random positions within fight area, nearby each other.
 *
 * @param env Environment
 */
static void set_fight_positions(OsrsEnv* env) {
    if (env->has_rng_seed) {
        int x0 = FIGHT_AREA_BASE_X;
        int y0 = FIGHT_AREA_BASE_Y;
        int x1 = FIGHT_AREA_BASE_X + FIGHT_NEARBY_RADIUS;
        int y1 = FIGHT_AREA_BASE_Y;

        env->players[0].x = x0;
        env->players[0].y = y0;
        env->players[0].dest_x = x0;
        env->players[0].dest_y = y0;
        env->players[0].is_moving = 0;

        env->players[1].x = x1;
        env->players[1].y = y1;
        env->players[1].dest_x = x1;
        env->players[1].dest_y = y1;
        env->players[1].is_moving = 0;
        return;
    }

    int base_x = FIGHT_AREA_BASE_X;
    int base_y = FIGHT_AREA_BASE_Y;
    int max_x = base_x + FIGHT_AREA_WIDTH;
    int max_y = base_y + FIGHT_AREA_HEIGHT;

    int x0 = base_x + rand_int(env, FIGHT_AREA_WIDTH);
    int y0 = base_y + rand_int(env, FIGHT_AREA_HEIGHT);

    int near_min_x = max_int(base_x, x0 - FIGHT_NEARBY_RADIUS);
    int near_min_y = max_int(base_y, y0 - FIGHT_NEARBY_RADIUS);
    int near_max_x = min_int(max_x, x0 + FIGHT_NEARBY_RADIUS);
    int near_max_y = min_int(max_y, y0 + FIGHT_NEARBY_RADIUS);

    int x1 = near_min_x + rand_int(env, near_max_x - near_min_x);
    int y1 = near_min_y + rand_int(env, near_max_y - near_min_y);

    env->players[0].x = x0;
    env->players[0].y = y0;
    env->players[0].dest_x = x0;
    env->players[0].dest_y = y0;
    env->players[0].is_moving = 0;

    env->players[1].x = x1;
    env->players[1].y = y1;
    env->players[1].dest_x = x1;
    env->players[1].dest_y = y1;
    env->players[1].is_moving = 0;
}

/**
 * Initialize internal buffer pointers for ocean pattern.
 *
 * Points observations/actions/rewards/terminals/action_masks at the internal
 * _*_buf arrays so game logic writes to local storage. PufferLib shared
 * buffers are accessed via ocean_* pointers set by the binding.
 *
 * @param env Environment to initialize
 */
void pvp_init(OsrsEnv* env) {
    env->observations = env->_obs_buf;
    env->actions = env->_acts_buf;
    env->rewards = env->_rews_buf;
    env->terminals = env->_terms_buf;
    env->action_masks = env->_masks_buf;
    env->action_masks_agents = 0x3;  // Both agents get masks

    memset(env->_obs_buf, 0, sizeof(env->_obs_buf));
    memset(env->_acts_buf, 0, sizeof(env->_acts_buf));
    memset(env->_rews_buf, 0, sizeof(env->_rews_buf));
    memset(env->_terms_buf, 0, sizeof(env->_terms_buf));
    memset(env->_masks_buf, 0, sizeof(env->_masks_buf));

    env->_episode_return = 0.0f;
    env->has_rng_seed = 0;
    env->is_lms = 1;
    env->pvp_runtime.is_pvp_arena = 0;
    env->auto_reset = 1;
    env->pvp_runtime.use_c_opponent = 0;
    env->pvp_runtime.use_c_opponent_p0 = 0;
    env->pvp_runtime.use_external_opponent_actions = 0;
    env->ocean_io.agent_obs_p1 = NULL;
    env->ocean_io.selfplay_mask = NULL;
    memset(env->pvp_runtime.external_opponent_actions, 0, sizeof(env->pvp_runtime.external_opponent_actions));
    memset(&env->pvp_runtime.opponent, 0, sizeof(env->pvp_runtime.opponent));
    memset(&env->pvp_runtime.opponent_p0, 0, sizeof(env->pvp_runtime.opponent_p0));
    memset(&env->pvp_runtime.pfsp, 0, sizeof(env->pvp_runtime.pfsp));
    memset(env->pvp_runtime.gear_tier_weights, 0, sizeof(env->pvp_runtime.gear_tier_weights));
    memset(&env->shaping, 0, sizeof(env->shaping));
    memset(&env->log, 0, sizeof(env->log));
}

/* pvp_render: forward declaration only.
   binding.c provides the stub, or osrs_render.h provides the real impl. */
void pvp_render(OsrsEnv* env);

/**
 * Reset the environment to initial state.
 *
 * Initializes both players, sets fight positions, resets tick counter,
 * and generates initial observations.
 *
 * @param env Environment to reset
 */
void pvp_reset(OsrsEnv* env) {
    if (env->has_rng_seed) {
        if (env->rng_seed == 0) {
            fprintf(stderr, "Error: seed must be non-zero (use seed=1 or higher in reset())\n");
            abort();
        }
        env->rng_state = env->rng_seed;
    } else {
        env->rng_state = (uint32_t)(size_t)env ^ 0xDEADBEEF;
    }

    init_player(&env->players[0]);
    init_player(&env->players[1]);

    // LMS overrides: defence capped at 75, prayer is 99 (no drain in LMS)
    for (int i = 0; i < NUM_AGENTS; i++) {
        env->players[i].is_lms = env->is_lms;
        if (env->is_lms) {
            env->players[i].base_defence = LMS_BASE_DEFENCE;
            env->players[i].current_defence = LMS_BASE_DEFENCE;
            env->players[i].base_prayer = 99;
            env->players[i].current_prayer = 99;
        }
    }

    set_fight_positions(env);

    // Initialize last_obs_target to actual opponent positions
    // (needed for first-tick movement commands like farcast)
    env->players[0].last_obs_target_x = env->players[1].x;
    env->players[0].last_obs_target_y = env->players[1].y;
    env->players[1].last_obs_target_x = env->players[0].x;
    env->players[1].last_obs_target_y = env->players[0].y;

    env->tick = 0;
    env->episode_over = 0;
    env->winner = -1;
    if (env->has_rng_seed) {
        env->pid_holder = 1 - (int)(env->rng_seed & 1u);
    } else {
        env->pid_holder = rand_int(env, 2);
    }
    env->pid_shuffle_countdown = 100 + rand_int(env, 51); // 100-150 ticks

    env->pvp_runtime.is_pvp_arena = 0;

    env->_episode_return = 0.0f;

    memset(env->rewards, 0, NUM_AGENTS * sizeof(float));
    memset(env->terminals, 0, NUM_AGENTS);

    memset(env->pending_actions, 0, sizeof(env->pending_actions));
    memset(env->last_executed_actions, 0, sizeof(env->last_executed_actions));

    // Initialize slot mode equipment with correlated per-episode gear randomization
    // LMS tier distribution: 80% same, 15% ±1 tier, 5% ±2 tiers
    int base_tier = sample_gear_tier(env->pvp_runtime.gear_tier_weights, &env->rng_state);
    int p1_tier = base_tier;

    float tier_roll = (float)xorshift32(&env->rng_state) / (float)UINT32_MAX;
    if (tier_roll >= 0.80f && tier_roll < 0.95f) {
        int dir = (xorshift32(&env->rng_state) & 1) ? 1 : -1;
        p1_tier = base_tier + dir;
    } else if (tier_roll >= 0.95f) {
        int dir = (xorshift32(&env->rng_state) & 1) ? 1 : -1;
        p1_tier = base_tier + dir * 2;
    }
    if (p1_tier < 0) p1_tier = 0;
    if (p1_tier > 3) p1_tier = 3;

    int tiers[NUM_AGENTS] = { base_tier, p1_tier };
    for (int i = 0; i < NUM_AGENTS; i++) {
        init_player_gear_randomized(&env->players[i], tiers[i], &env->rng_state);
        env->players[i].food_count = compute_food_count(&env->players[i]);
        osrs_refresh_player_equipment(&env->players[i]);
    }

    // Reset C-side opponent state for new episode
    // Always reset when PFSP is configured (selfplay toggle happens inside opponent_reset)
    if (env->pvp_runtime.use_c_opponent || env->pvp_runtime.opponent.type == OPP_PFSP) {
        opponent_reset(env, &env->pvp_runtime.opponent);
    }
    if (env->pvp_runtime.use_c_opponent_p0) {
        opponent_reset(env, &env->pvp_runtime.opponent_p0);
    }

    for (int i = 0; i < NUM_AGENTS; i++) {
        generate_slot_observations(env, i);
        if (env->action_masks != NULL && (env->action_masks_agents & (1 << i))) {
            compute_action_masks(env, i);
        }
    }
}

/**
 * Execute one game tick with OSRS-accurate 1-tick delay timing.
 *
 * Actions submitted on tick N are applied IMMEDIATELY in the same step,
 * producing tick N+1 state. This gives proper 1-tick delay:
 * action at tick N → effects visible at tick N+1.
 *
 * Flow:
 *   1. Copy model/external actions to env->actions
 *   2. Generate C opponent actions into env->actions
 *   3. Apply actions immediately (execute switches, then attacks)
 *   4. Increment tick
 *   5. Check win conditions
 *   6. Calculate rewards
 *   7. Generate observations
 *
 * @param env Environment
 */
void pvp_step(OsrsEnv* env) {
    memset(env->rewards, 0, NUM_AGENTS * sizeof(float));
    memset(env->terminals, 0, NUM_AGENTS);

    // Reset per-tick flags at START (clears flags from PREVIOUS tick)
    // This allows get_state() to read flags after pvp_step() returns
    for (int i = 0; i < NUM_AGENTS; i++) {
        env->players[i].hit_landed_this_tick = 0;
        env->players[i].hit_was_successful = 0;
        env->players[i].hit_damage = 0;
        env->players[i].hit_style = ATTACK_STYLE_NONE;
        env->players[i].hit_defender_prayer = PRAYER_NONE;
        env->players[i].hit_was_on_prayer = 0;
        env->players[i].hit_attacker_idx = -1;
        env->players[i].freeze_applied_this_tick = 0;
    }
    reset_tick_flags(&env->players[0]);
    reset_tick_flags(&env->players[1]);

    // Copy model's actions (player 0) or clear if C opponent controls p0
    if (env->pvp_runtime.use_c_opponent_p0) {
        memset(env->actions, 0, NUM_ACTION_HEADS * sizeof(int));
    } else {
        memcpy(env->actions, env->ocean_io.agent_actions, NUM_ACTION_HEADS * sizeof(int));
    }

    // Copy external opponent actions (player 1) or clear for C opponent
    if (env->pvp_runtime.use_external_opponent_actions) {
        memcpy(
            env->actions + NUM_ACTION_HEADS,
            env->pvp_runtime.external_opponent_actions,
            NUM_ACTION_HEADS * sizeof(int)
        );
    } else {
        memset(env->actions + NUM_ACTION_HEADS, 0, NUM_ACTION_HEADS * sizeof(int));
    }

    // Generate C opponent actions (writes to pending_actions, then copy to actions)
    if (env->pvp_runtime.use_c_opponent && !env->pvp_runtime.use_external_opponent_actions) {
        generate_opponent_action(env, &env->pvp_runtime.opponent);
        // Copy C opponent's action from pending to actions buffer
        memcpy(
            env->actions + NUM_ACTION_HEADS,
            env->pending_actions + NUM_ACTION_HEADS,
            NUM_ACTION_HEADS * sizeof(int)
        );
    }
    if (env->pvp_runtime.use_c_opponent_p0) {
        generate_opponent_action_for_player0(env, &env->pvp_runtime.opponent_p0);
        // Copy C opponent's action from pending to actions buffer
        memcpy(
            env->actions,
            env->pending_actions,
            NUM_ACTION_HEADS * sizeof(int)
        );
    }

    int first = env->pid_holder;
    int second = 1 - env->pid_holder;

    // Copy actions to local arrays for each agent
    int actions_p0[NUM_ACTION_HEADS];
    int actions_p1[NUM_ACTION_HEADS];
    memcpy(actions_p0, env->actions, NUM_ACTION_HEADS * sizeof(int));
    memcpy(actions_p1, env->actions + NUM_ACTION_HEADS, NUM_ACTION_HEADS * sizeof(int));

    // Clamp impossible cross-head combos:
    // - MELEE/RANGE/SPEC_MELEE/SPEC_RANGE/GMAUL cannot cast spells
    // - MAGE/TANK cannot use ATK (except SPEC_MAGIC which forces ATK internally)
    for (int i = 0; i < NUM_AGENTS; i++) {
        int* a = (i == 0) ? actions_p0 : actions_p1;
        int lo = a[HEAD_LOADOUT];
        int cv = a[HEAD_COMBAT];
        if (lo == LOADOUT_MAGE || lo == LOADOUT_TANK || lo == LOADOUT_SPEC_MAGIC) {
            if (cv == ATTACK_ATK) {
                a[HEAD_COMBAT] = ATTACK_NONE;
            }
        }
    }

    // Write clamped actions back for recording and read functions
    memcpy(env->actions, actions_p0, NUM_ACTION_HEADS * sizeof(int));
    memcpy(env->actions + NUM_ACTION_HEADS, actions_p1, NUM_ACTION_HEADS * sizeof(int));

    // Save executed actions for recording
    memcpy(
        env->last_executed_actions,
        env->actions,
        NUM_AGENTS * NUM_ACTION_HEADS * sizeof(int)
    );

    update_timers(&env->players[0]);
    update_timers(&env->players[1]);

    for (int i = 0; i < NUM_AGENTS; i++) {
        Player* pi = &env->players[i];
        pi->prev_hp_percent = (float)pi->current_hitpoints / (float)pi->base_hitpoints;
    }

    int* agent_actions[NUM_AGENTS];
    agent_actions[0] = actions_p0;
    agent_actions[1] = actions_p1;

    int pre_move_x[NUM_AGENTS], pre_move_y[NUM_AGENTS];
    for (int i = 0; i < NUM_AGENTS; i++) {
        pre_move_x[i] = env->players[i].x;
        pre_move_y[i] = env->players[i].y;
    }

    execute_switches(env, first, agent_actions[first]);
    execute_switches(env, second, agent_actions[second]);

    for (int i = 0; i < NUM_AGENTS; i++) {
        Player* pi = &env->players[i];
        if (pi->food_timer > 0) pi->food_timer--;
        if (pi->potion_timer > 0) pi->potion_timer--;
        if (pi->karambwan_timer > 0) pi->karambwan_timer--;
    }

    if (env->players[0].x == env->players[1].x &&
        env->players[0].y == env->players[1].y) {
        resolve_same_tile(&env->players[second], &env->players[first], (const CollisionMap*)env->collision_map);
    }

    execute_attack_movement(env, first, agent_actions[first]);
    execute_attack_movement(env, second, agent_actions[second]);

    if (env->players[0].x == env->players[1].x &&
        env->players[0].y == env->players[1].y) {
        resolve_same_tile(&env->players[second], &env->players[first], (const CollisionMap*)env->collision_map);
    }

    execute_attack_combat(env, first, agent_actions[first]);
    execute_attack_combat(env, second, agent_actions[second]);

    if (env->players[0].x == env->players[1].x &&
        env->players[0].y == env->players[1].y) {
        resolve_same_tile(&env->players[second], &env->players[first], (const CollisionMap*)env->collision_map);
    }

    for (int i = 0; i < NUM_AGENTS; i++) {
        int dx = abs(env->players[i].x - pre_move_x[i]);
        int dy = abs(env->players[i].y - pre_move_y[i]);
        int dist = (dx > dy) ? dx : dy;
        env->players[i].is_running = (dist >= 2) ? 1 : 0;
    }

    process_pending_hits(env, 0, 1);
    process_pending_hits(env, 1, 0);

    // Morrigan's javelin DoT: 5 HP every 3 ticks from calc tick
    for (int i = 0; i < NUM_AGENTS; i++) {
        Player* p = &env->players[i];
        if (p->morr_dot_remaining > 0) {
            p->morr_dot_tick_counter--;
            if (p->morr_dot_tick_counter <= 0) {
                int dot_dmg = (p->morr_dot_remaining >= 5) ? 5 : p->morr_dot_remaining;
                p->current_hitpoints -= dot_dmg;
                p->morr_dot_remaining -= dot_dmg;
                p->damage_applied_this_tick += dot_dmg;
                if (p->current_hitpoints < 0) p->current_hitpoints = 0;
                p->morr_dot_tick_counter = 3;
            }
        }
    }

    if (env->players[0].veng_active) {
        env->players[1].observed_target_lunar_spellbook = 1;
    }
    if (env->players[1].veng_active) {
        env->players[0].observed_target_lunar_spellbook = 1;
    }
    env->tick++;

    if (!env->has_rng_seed) {
        env->pid_shuffle_countdown--;
        if (env->pid_shuffle_countdown <= 0) {
            env->pid_holder = 1 - env->pid_holder;
            env->pid_shuffle_countdown = 100 + rand_int(env, 51); // 100-150 ticks
        }
    }

    memcpy(env->pending_actions, env->actions,
           NUM_AGENTS * NUM_ACTION_HEADS * sizeof(int));
    for (int i = 0; i < NUM_AGENTS; i++) {
        if (env->players[i].current_hitpoints <= 0) {
            env->episode_over = 1;
            env->winner = 1 - i;
        }
    }

    // Tick limit: treat timeout as agent 0 loss
    if (!env->episode_over && env->tick >= MAX_EPISODE_TICKS) {
        env->episode_over = 1;
        env->winner = 1;
    }
    for (int i = 0; i < NUM_AGENTS; i++) {
        env->rewards[i] = calculate_reward(env, i);

        if (env->episode_over) {
            env->terminals[i] = 1;
        }
    }

    // Accumulate agent 0's episode return (written to log at episode end)
    env->_episode_return += env->rewards[0];
    for (int i = 0; i < NUM_AGENTS; i++) {
        generate_slot_observations(env, i);
        if (env->action_masks != NULL && (env->action_masks_agents & (1 << i))) {
            compute_action_masks(env, i);
        }
    }

    // Write observations to PufferLib shared buffer
    ocean_write_obs(env);
    if (env->ocean_io.agent_obs_p1 != NULL) {
        ocean_write_obs_p1(env);
    }
    env->ocean_io.agent_rewards[0] = env->rewards[0];

    if (env->episode_over) {
        env->ocean_io.agent_terminals[0] = 1;

        // PFSP win tracking (all in C, zero Python overhead).
        // Skip if pool_idx is -1 (sentinel for pre-pool-config first episode).
        if (env->pvp_runtime.opponent.type == OPP_PFSP && env->pvp_runtime.pfsp.active_pool_idx >= 0) {
            int idx = env->pvp_runtime.pfsp.active_pool_idx;
            env->pvp_runtime.pfsp.episodes[idx] += 1.0f;
            if (env->winner == 0) {
                env->pvp_runtime.pfsp.wins[idx] += 1.0f;
            }
        }

        // Write final episode stats to log
        Player* p0 = &env->players[0];
        env->log.episode_return = env->_episode_return;
        env->log.episode_length = (float)env->tick;
        env->log.damage_dealt = p0->total_damage_dealt;
        env->log.damage_received = p0->total_damage_received;
        env->log.wins = (env->winner == 0) ? 1.0f : 0.0f;
        env->log.prayer_correct = (float)p0->target_pray_correct_count;
        env->log.prayer_total = (float)(p0->target_pray_melee_count +
            p0->target_pray_ranged_count + p0->target_pray_magic_count);
        env->log.idle_ticks = (float)(p0->food_count + p0->karambwan_count); /* food remaining */
        env->log.brews_used = (float)p0->brew_doses;    /* brews remaining */
        env->log.wave = (float)p0->special_energy;      /* spec energy remaining */
        env->log.npc_kills = (float)p0->total_target_hit_count;     /* attacks landed */
        env->log.blood_healed = (float)p0->target_hit_off_prayer_count; /* off-prayer hits */
        env->log.n = 1.0f;

        // Auto-reset for next episode
        if (env->auto_reset) {
            pvp_reset(env);
        }
    } else {
        env->ocean_io.agent_terminals[0] = 0;
    }
}

/**
 * Set RNG seed for deterministic runs.
 *
 * @param env  Environment
 * @param seed Seed value (must be non-zero)
 */
void pvp_seed(OsrsEnv* env, uint32_t seed) {
    env->rng_seed = seed;
    env->has_rng_seed = 1;
}

/**
 * Cleanup environment resources.
 *
 * Currently a no-op since all memory is statically allocated.
 *
 * @param env Environment
 */
void pvp_close(OsrsEnv* env) {
    (void)env;
}

#endif // OSRS_PVP_API_H
