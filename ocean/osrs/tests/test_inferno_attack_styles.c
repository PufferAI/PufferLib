/**
 * @file test_inferno_attack_styles.c
 * @brief regression tests for inferno NPC attack-style selection and melee
 * fallback geometry.
 *
 * BUILD:
 *   cc -std=c11 -O0 -g -I. -o /tmp/test_inferno_attack_styles \
 *       ocean/osrs/tests/test_inferno_attack_styles.c -lm
 *   /tmp/test_inferno_attack_styles
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_inferno.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_INT_EQ(label, actual, expected) do { \
    tests_run++; \
    if ((actual) == (expected)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s — got %d, expected %d\n", (label), (actual), (expected)); \
    } \
} while (0)

#define ASSERT_FLOAT_NEAR(label, actual, expected, tol) do { \
    tests_run++; \
    float diff = (float)((actual) - (expected)); \
    if (diff < 0.0f) diff = -diff; \
    if (diff <= (tol)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s — got %.6f, expected %.6f (tol %.6f)\n", \
            (label), (float)(actual), (float)(expected), (float)(tol)); \
    } \
} while (0)

static InfernoState make_test_state(int player_x, int player_y) {
    InfernoState state;
    memset(&state, 0, sizeof(state));
    memset(state.npc_los_cache, -1, sizeof(state.npc_los_cache));
    state.player.x = player_x;
    state.player.y = player_y;
    state.player_last_interaction_target_slot = -1;
    state.player_last_interaction_age = 1;
    return state;
}

static InfNPC make_test_npc(InfNPCType type, int x, int y, int size) {
    InfNPC npc;
    memset(&npc, 0, sizeof(npc));
    npc.type = type;
    npc.x = x;
    npc.y = y;
    npc.size = size;
    npc.aggro_target = -1;
    npc.attack_visual_target = -1;
    npc.jad_owner_idx = -1;
    npc.blob_scanned_prayer = -1;
    npc.jad_attack_style = ATTACK_STYLE_NONE;
    return npc;
}

static InfNPCStats make_test_stats(int default_style) {
    InfNPCStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.default_style = default_style;
    stats.can_melee = 1;
    return stats;
}

static HumanInput make_human_input(void) {
    HumanInput input;
    memset(&input, 0, sizeof(input));
    input.pending_move_x = -1;
    input.pending_move_y = -1;
    input.pending_prayer = -1;
    input.pending_offensive_prayer = -1;
    input.pending_target_idx = -1;
    return input;
}

static void init_jad_timing_test_state(InfernoState* state, int player_x, int player_y, int jad_x, int jad_y) {
    memset(state, 0, sizeof(*state));
    memset(state->npc_los_cache, -1, sizeof(state->npc_los_cache));
    state->rng_state = 12345;
    state->wave = 66;
    state->player.entity_type = ENTITY_PLAYER;
    state->player.x = player_x;
    state->player.y = player_y;
    state->player.base_hitpoints = 99;
    state->player.current_hitpoints = 99;
    state->player.base_prayer = 99;
    state->player.current_prayer = 99;
    state->player.base_attack = 99;
    state->player.base_strength = 99;
    state->player.base_defence = 99;
    state->player.base_ranged = 99;
    state->player.base_magic = 99;
    state->player.current_attack = 99;
    state->player.current_strength = 99;
    state->player.current_defence = 99;
    state->player.current_ranged = 99;
    state->player.current_magic = 99;
    state->player.prayer = PRAYER_NONE;
    state->weapon_set = INF_GEAR_MAGE;
    state->player_last_interaction_target_slot = -1;
    state->player_last_interaction_age = 1;
    state->player_dest_x = -1;
    state->player_dest_y = -1;
    osrs_interaction_init(&state->interaction);
    encounter_compute_loadout_stats(INF_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30,
        &state->loadout_stats[INF_GEAR_MAGE]);

    state->npcs[0] = make_test_npc(
        INF_NPC_JAD, jad_x, jad_y, INF_NPC_STATS[INF_NPC_JAD].size);
    state->npcs[0].active = 1;
    state->npcs[0].attack_timer = 0;
    state->npcs[0].jad_attack_style = ATTACK_STYLE_MAGIC;
    state->npcs[0].attack_style = ATTACK_STYLE_RANGED;
}

static void step_inferno_with_prayer(InfernoState* state, int prayer_action) {
    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRAYER] = prayer_action;
    inf_step((EncounterState*)state, actions);
}

static int force_mager_resurrect(InfernoState* s, int idx) {
    for (uint32_t seed = 1; seed < 100000; seed++) {
        InfernoState probe = *s;
        probe.rng_state = seed;
        if (inf_mager_resurrect(&probe, idx)) {
            s->rng_state = seed;
            return inf_mager_resurrect(s, idx);
        }
    }
    return 0;
}

static int distance_to_player(const InfernoState* state, const InfNPC* npc) {
    return encounter_dist_to_npc(
        state->player.x, state->player.y, npc->x, npc->y, npc->size);
}

static int test_profiled_supply_count(int full_doses, float profile_fraction, float scale) {
    float effective_fraction = 1.0f - scale * (1.0f - profile_fraction);
    int doses = (int)((float)full_doses * effective_fraction + 0.5f);
    if (doses < 0) doses = 0;
    if (doses > full_doses) doses = full_doses;
    return doses;
}

static void reset_inferno_at_public_wave(EncounterState* raw_state,
                                         int public_wave,
                                         float supply_profile_scale) {
    inf_put_int(raw_state, "start_wave", public_wave);
    inf_put_float(raw_state, "damage_reward_coeff", 0.01f);
    inf_put_float(raw_state, "shield_penalty_coeff", 0.01f);
    inf_put_float(raw_state, "tag_reward_coeff", 0.25f);
    inf_put_float(raw_state, "late_start_supply_profile_scale", supply_profile_scale);
    inf_reset(raw_state, 123);
}

static void assert_supply_doses(const char* label,
                                const Player* player,
                                InfSupplyDoses expected) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s brew doses", label);
    ASSERT_INT_EQ(buf, player->brew_doses, expected.brew_doses);
    snprintf(buf, sizeof(buf), "%s restore doses", label);
    ASSERT_INT_EQ(buf, player->restore_doses, expected.restore_doses);
    snprintf(buf, sizeof(buf), "%s bastion doses", label);
    ASSERT_INT_EQ(buf, player->bastion_doses, expected.bastion_doses);
    snprintf(buf, sizeof(buf), "%s stamina doses", label);
    ASSERT_INT_EQ(buf, player->stamina_doses, expected.stamina_doses);
}

static void test_reward_switches_between_healer_tags_and_damage(void) {
    printf("--- inferno reward switches between healer tags and damage ---\n");

    InfernoState healing_state = make_test_state(24, 24);
    InfernoState damage_state = make_test_state(24, 24);

    inf_put_float((EncounterState*)&healing_state, "damage_reward_coeff", 0.01f);
    inf_put_float((EncounterState*)&healing_state, "shield_penalty_coeff", 0.01f);
    inf_put_float((EncounterState*)&healing_state, "tag_reward_coeff", 0.25f);
    healing_state.wave = INF_NUM_WAVES - 1;
    healing_state.damage_dealt_this_tick = 50.0f;
    healing_state.hp_restored_this_tick = 10.0f;
    healing_state.shield_damage_this_tick = 7.0f;
    healing_state.healer_tags_this_tick = 2;
    healing_state.npcs[0] = make_test_npc(INF_NPC_HEALER_ZUK, 26, 24, 1);
    healing_state.npcs[0].active = 1;
    healing_state.npcs[0].aggro_target = 1;
    healing_state.npcs[1] = make_test_npc(INF_NPC_ZUK, 28, 24, 5);
    healing_state.npcs[1].active = 1;
    healing_state.min_zuk_hp_seen = 1200.0f;

    damage_state = healing_state;
    damage_state.wave = 0;
    damage_state.npcs[0].aggro_target = -1;

    ASSERT_FLOAT_NEAR("final-wave active healer reward uses tag path",
        inf_compute_reward(&healing_state), 0.43f, 0.0001f);
    ASSERT_FLOAT_NEAR("non-final-wave reward still uses damage path",
        inf_compute_reward(&damage_state), 0.33f, 0.0001f);
}

static void test_final_wave_reward_uses_zuk_low_watermark_progress(void) {
    printf("--- final-wave reward uses zuk low-watermark progress ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float((EncounterState*)&state, "damage_reward_coeff", 0.01f);
    inf_put_float((EncounterState*)&state, "shield_penalty_coeff", 0.01f);
    inf_put_float((EncounterState*)&state, "tag_reward_coeff", 0.25f);
    state.wave = INF_NUM_WAVES - 1;
    state.min_zuk_hp_seen = 1200.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 1150;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_JAD, 24, 32, 5);
    state.npcs[1].active = 1;

    state.damage_dealt_this_tick = 250.0f;
    state.hp_restored_this_tick = 100.0f;
    state.shield_damage_this_tick = 7.0f;
    ASSERT_FLOAT_NEAR("first zuk low watermark pays progress minus shield penalty",
        inf_compute_reward(&state), 0.43f, 0.0001f);
    ASSERT_FLOAT_NEAR("first zuk low watermark updates state",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.damage_dealt_this_tick = 400.0f;
    state.hp_restored_this_tick = 0.0f;
    state.shield_damage_this_tick = 0.0f;
    ASSERT_FLOAT_NEAR("repeated hits at same zuk hp give zero reward",
        inf_compute_reward(&state), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("same-hp hits keep low watermark",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.damage_dealt_this_tick = 600.0f;
    state.npcs[0].hp = 1180;
    ASSERT_FLOAT_NEAR("healed zuk above low watermark gives zero reward",
        inf_compute_reward(&state), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("healed zuk does not revoke low watermark",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.damage_dealt_this_tick = 900.0f;
    ASSERT_FLOAT_NEAR("non-zuk damage without new low watermark gives zero reward",
        inf_compute_reward(&state), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("non-zuk damage leaves low watermark unchanged",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.npcs[0].hp = 1140;
    state.damage_dealt_this_tick = 50.0f;
    ASSERT_FLOAT_NEAR("new lower zuk hp pays only incremental progress",
        inf_compute_reward(&state), 0.10f, 0.0001f);
    ASSERT_FLOAT_NEAR("new lower zuk hp refreshes low watermark",
        state.min_zuk_hp_seen, 1140.0f, 0.0001f);
}

static void test_inferno_reset_supplies_match_current_inventory(void) {
    printf("--- inferno reset supplies match current inventory ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    InfSupplyDoses full = inf_full_starting_supplies();

    reset_inferno_at_public_wave(raw_state, 1, 1.0f);

    assert_supply_doses("wave 1", &state->player, full);

    inf_destroy(raw_state);
}

static void test_late_start_supply_profile_anchor_waves(void) {
    printf("--- inferno late-start supply profile anchor waves ---\n");

    struct {
        int public_wave;
        float brew_fraction;
        float restore_fraction;
        float bastion_fraction;
        float stamina_fraction;
    } anchors[] = {
        { 20, 1.0000f, 0.9500f, 1.0000f, 1.0000f },
        { 40, 0.9167f, 0.8750f, 1.0000f, 1.0000f },
        { 61, 0.8333f, 0.7500f, 1.0000f, 1.0000f },
        { 64, 0.5833f, 0.5000f, 0.7500f, 1.0000f },
        { 68, 0.5833f, 0.4250f, 0.6250f, 1.0000f },
        { 69, 0.5000f, 0.3000f, 0.3750f, 1.0000f },
    };

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    InfSupplyDoses full = inf_full_starting_supplies();

    for (int i = 0; i < (int)(sizeof(anchors) / sizeof(anchors[0])); i++) {
        reset_inferno_at_public_wave(raw_state, anchors[i].public_wave, 1.0f);
        InfSupplyDoses expected = {
            .brew_doses = test_profiled_supply_count(full.brew_doses,
                anchors[i].brew_fraction, 1.0f),
            .restore_doses = test_profiled_supply_count(full.restore_doses,
                anchors[i].restore_fraction, 1.0f),
            .bastion_doses = test_profiled_supply_count(full.bastion_doses,
                anchors[i].bastion_fraction, 1.0f),
            .stamina_doses = test_profiled_supply_count(full.stamina_doses,
                anchors[i].stamina_fraction, 1.0f),
        };
        char label[64];
        snprintf(label, sizeof(label), "wave %d", anchors[i].public_wave);
        assert_supply_doses(label, &state->player, expected);
    }

    inf_destroy(raw_state);
}

static void test_late_start_supply_profile_interpolation_and_scale(void) {
    printf("--- inferno late-start supply profile interpolation and scale ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    InfSupplyDoses full = inf_full_starting_supplies();
    float t = 1.0f / 3.0f;
    float brew_fraction = 0.8333f + (0.5833f - 0.8333f) * t;
    float restore_fraction = 0.7500f + (0.5000f - 0.7500f) * t;
    float bastion_fraction = 1.0000f + (0.7500f - 1.0000f) * t;

    reset_inferno_at_public_wave(raw_state, 62, 1.0f);
    InfSupplyDoses interpolated = {
        .brew_doses = test_profiled_supply_count(full.brew_doses, brew_fraction, 1.0f),
        .restore_doses = test_profiled_supply_count(full.restore_doses, restore_fraction, 1.0f),
        .bastion_doses = test_profiled_supply_count(full.bastion_doses, bastion_fraction, 1.0f),
        .stamina_doses = full.stamina_doses,
    };
    assert_supply_doses("wave 62", &state->player, interpolated);

    reset_inferno_at_public_wave(raw_state, 69, 0.0f);
    assert_supply_doses("wave 69 scale 0", &state->player, full);

    reset_inferno_at_public_wave(raw_state, 69, 0.5f);
    InfSupplyDoses half_scale = {
        .brew_doses = test_profiled_supply_count(full.brew_doses, 0.5000f, 0.5f),
        .restore_doses = test_profiled_supply_count(full.restore_doses, 0.3000f, 0.5f),
        .bastion_doses = test_profiled_supply_count(full.bastion_doses, 0.3750f, 0.5f),
        .stamina_doses = test_profiled_supply_count(full.stamina_doses, 1.0000f, 0.5f),
    };
    assert_supply_doses("wave 69 scale 0.5", &state->player, half_scale);

    inf_destroy(raw_state);
}

static void test_late_start_supply_observations(void) {
    printf("--- inferno late-start supply observations ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    InfSupplyDoses full = inf_full_starting_supplies();
    float obs[INF_NUM_OBS];

    reset_inferno_at_public_wave(raw_state, 69, 1.0f);
    inf_write_obs(raw_state, obs);

    enum {
        INF_OBS_BREW_DOSES = 11,
        INF_OBS_RESTORE_DOSES = 12,
        INF_OBS_BASTION_DOSES = 20,
        INF_OBS_STAMINA_DOSES = 21,
    };
    ASSERT_FLOAT_NEAR("brew obs uses full-kit denominator",
        obs[INF_OBS_BREW_DOSES],
        (float)state->player.brew_doses / (float)full.brew_doses, 0.0001f);
    ASSERT_FLOAT_NEAR("restore obs uses full-kit denominator",
        obs[INF_OBS_RESTORE_DOSES],
        (float)state->player.restore_doses / (float)full.restore_doses, 0.0001f);
    ASSERT_FLOAT_NEAR("bastion obs uses full-kit denominator",
        obs[INF_OBS_BASTION_DOSES],
        (float)state->player.bastion_doses / (float)full.bastion_doses, 0.0001f);
    ASSERT_FLOAT_NEAR("stamina obs uses full-kit denominator",
        obs[INF_OBS_STAMINA_DOSES],
        (float)state->player.stamina_doses / (float)full.stamina_doses, 0.0001f);

    inf_destroy(raw_state);
}

static void test_tagged_jad_healer_melee_geometry(void) {
    printf("--- tagged jad healer melee geometry ---\n");

    InfernoState diagonal_state = make_test_state(5, 5);
    InfernoState cardinal_state = make_test_state(5, 5);

    diagonal_state.player.current_defence = 99;
    diagonal_state.player.current_magic = 99;
    diagonal_state.player.prayer = PRAYER_NONE;
    diagonal_state.weapon_set = INF_GEAR_MAGE;

    cardinal_state.player.current_defence = 99;
    cardinal_state.player.current_magic = 99;
    cardinal_state.player.prayer = PRAYER_NONE;
    cardinal_state.weapon_set = INF_GEAR_MAGE;

    diagonal_state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 6, 6, 1);
    diagonal_state.npcs[0].active = 1;
    diagonal_state.npcs[0].aggro_target = -1;

    cardinal_state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 6, 5, 1);
    cardinal_state.npcs[0].active = 1;
    cardinal_state.npcs[0].aggro_target = -1;

    inf_npc_attack(&diagonal_state, 0);
    inf_npc_attack(&cardinal_state, 0);

    ASSERT_INT_EQ("diagonal healer does not attack", diagonal_state.npcs[0].attacked_this_tick, 0);
    ASSERT_INT_EQ("diagonal healer keeps attack style none",
                  diagonal_state.npcs[0].attack_style_this_tick, ATTACK_STYLE_NONE);
    ASSERT_INT_EQ("cardinal healer attacks", cardinal_state.npcs[0].attacked_this_tick, 1);
    ASSERT_INT_EQ("cardinal healer uses melee",
                  cardinal_state.npcs[0].attack_style_this_tick, ATTACK_STYLE_MELEE);
}

static void test_overlap_shuffle_hold_after_recent_target_click(void) {
    printf("--- overlap shuffle held after recent target click ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player_last_interaction_target_slot = 0;
    state.player_last_interaction_age = 0;

    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 20, 20, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;

    inf_npc_move(&state, 0);

    ASSERT_INT_EQ("held overlap keeps x", state.npcs[0].x, 20);
    ASSERT_INT_EQ("held overlap keeps y", state.npcs[0].y, 20);
    ASSERT_INT_EQ("held overlap does not mark moved", state.npcs[0].moved_this_tick, 0);
}

static void test_overlap_shuffle_respects_npc_collision_flags(void) {
    printf("--- overlap shuffle respects npc collision flags ---\n");

    InfernoState state = make_test_state(20, 20);
    state.rng_state = 12345;

    state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 20, 20, 1);
    state.npcs[0].active = 1;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_JAD, 21, 20, 1);
    state.npcs[1].active = 1;
    state.npcs[2] = make_test_npc(INF_NPC_HEALER_JAD, 19, 20, 1);
    state.npcs[2].active = 1;
    state.npcs[3] = make_test_npc(INF_NPC_HEALER_JAD, 20, 21, 1);
    state.npcs[3].active = 1;

    inf_rebuild_entity_collision_flags(&state);
    inf_npc_move(&state, 0);

    ASSERT_INT_EQ("overlap shuffle picks the only free tile x", state.npcs[0].x, 20);
    ASSERT_INT_EQ("overlap shuffle picks the only free tile y", state.npcs[0].y, 19);
}

static void test_tagged_jad_healer_stops_at_melee_contact(void) {
    printf("--- tagged jad healer stops at melee contact ---\n");

    InfernoState state = make_test_state(20, 20);
    state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 19, 20, 1);
    state.npcs[0].active = 1;
    state.npcs[0].aggro_target = -1;

    inf_rebuild_entity_collision_flags(&state);
    inf_npc_move(&state, 0);

    ASSERT_INT_EQ("healer keeps melee contact x", state.npcs[0].x, 19);
    ASSERT_INT_EQ("healer keeps melee contact y", state.npcs[0].y, 20);
    ASSERT_INT_EQ("healer does not mark moved", state.npcs[0].moved_this_tick, 0);
}

static void test_tagged_jad_healers_queue_behind_front_healer(void) {
    printf("--- tagged jad healers queue behind front healer ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.prayer = PRAYER_PROTECT_MAGIC;
    state.weapon_set = INF_GEAR_MAGE;

    for (int i = 0; i < 5; i++) {
        state.npcs[i] = make_test_npc(INF_NPC_HEALER_JAD, 19 - i, 20, 1);
        state.npcs[i].active = 1;
        state.npcs[i].aggro_target = -1;
        state.npcs[i].attack_timer = 0;
    }

    inf_rebuild_entity_collision_flags(&state);
    inf_tick_npcs(&state);

    int attacks = 0;
    int on_player = 0;
    for (int i = 0; i < 5; i++) {
        if (state.npcs[i].attacked_this_tick) attacks++;
        if (state.npcs[i].x == state.player.x && state.npcs[i].y == state.player.y)
            on_player++;
    }

    ASSERT_INT_EQ("only front healer attacks", attacks, 1);
    ASSERT_INT_EQ("no healer steps onto player", on_player, 0);
    ASSERT_INT_EQ("front healer remains first in queue", state.npcs[0].x, 19);
    ASSERT_INT_EQ("second healer remains blocked behind front", state.npcs[1].x, 18);
}

static void test_stacked_npc_unclipping_clears_flag_when_one_leaves(void) {
    printf("--- stacked npc unclipping clears flag when one leaves ---\n");

    InfernoState state = make_test_state(25, 25);
    state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 20, 20, 1);
    state.npcs[0].active = 1;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_JAD, 20, 20, 1);
    state.npcs[1].active = 1;

    inf_rebuild_entity_collision_flags(&state);
    ASSERT_INT_EQ("stacked tile initially flagged",
                  state.npc_collision_flags[20 - INF_ARENA_MIN_X][20 - INF_ARENA_MIN_Y], 1);

    inf_update_npc_collision_flags(&state, 0, 20, 20, 21, 20, 1);

    ASSERT_INT_EQ("old stacked tile unclipped",
                  state.npc_collision_flags[20 - INF_ARENA_MIN_X][20 - INF_ARENA_MIN_Y], 0);
    ASSERT_INT_EQ("new tile flagged",
                  state.npc_collision_flags[21 - INF_ARENA_MIN_X][20 - INF_ARENA_MIN_Y], 1);
}

static void test_jad_healer_spawn_offsets_match_wave_67_reference(void) {
    printf("--- jad healer spawn offsets match wave 67 reference ---\n");

    InfernoState state = make_test_state(18, 32);
    state.rng_state = 12345;
    state.wave = 66;
    state.npcs[0] = make_test_npc(INF_NPC_JAD, 23, 30, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 100;
    state.npcs[0].max_hp = 300;

    inf_rebuild_entity_collision_flags(&state);
    inf_jad_check_healers(&state, 0);

    int healers = 0;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (!state.npcs[i].active || state.npcs[i].type != INF_NPC_HEALER_JAD) continue;
        healers++;
        int dx = state.npcs[i].x - state.npcs[0].x;
        int dy = state.npcs[i].y - state.npcs[0].y;
        ASSERT_INT_EQ("wave 67 healer owner", state.npcs[i].jad_owner_idx, 0);
        ASSERT_INT_EQ("wave 67 healer aggro", state.npcs[i].aggro_target, 0);
        ASSERT_INT_EQ("wave 67 healer x min", dx >= -5, 1);
        ASSERT_INT_EQ("wave 67 healer x max", dx <= 5, 1);
        ASSERT_INT_EQ("wave 67 healer y min", dy >= -4, 1);
        ASSERT_INT_EQ("wave 67 healer y max", dy <= 10, 1);
        ASSERT_INT_EQ("wave 67 healer outside jad footprint",
            encounter_entity_footprints_overlap(
                state.npcs[i].x, state.npcs[i].y, 1,
                state.npcs[0].x, state.npcs[0].y, state.npcs[0].size),
            0);
    }
    ASSERT_INT_EQ("wave 67 healer count", healers, 5);
}

static void test_jad_healer_spawn_offsets_match_zuk_reference(void) {
    printf("--- jad healer spawn offsets match zuk reference ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.rng_state = 67890;
    state.wave = 68;
    state.npcs[0] = make_test_npc(INF_NPC_JAD, 24, 32, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 100;
    state.npcs[0].max_hp = 300;

    inf_rebuild_entity_collision_flags(&state);
    inf_jad_check_healers(&state, 0);

    int healers = 0;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (!state.npcs[i].active || state.npcs[i].type != INF_NPC_HEALER_JAD) continue;
        healers++;
        int dx = state.npcs[i].x - state.npcs[0].x;
        int dy = state.npcs[i].y - state.npcs[0].y;
        ASSERT_INT_EQ("zuk healer x min", dx >= 0, 1);
        ASSERT_INT_EQ("zuk healer x max", dx <= 5, 1);
        ASSERT_INT_EQ("zuk healer y min", dy >= 5, 1);
        ASSERT_INT_EQ("zuk healer y max", dy <= 8, 1);
        ASSERT_INT_EQ("zuk healer outside jad footprint",
            encounter_entity_footprints_overlap(
                state.npcs[i].x, state.npcs[i].y, 1,
                state.npcs[0].x, state.npcs[0].y, state.npcs[0].size),
            0);
    }
    ASSERT_INT_EQ("zuk healer count", healers, 3);
}

static void test_meleer_dig_landing_order(void) {
    printf("--- meleer dig landing order ---\n");

    InfernoState state = make_test_state(20, 20);
    state.npcs[0] = make_test_npc(
        INF_NPC_MELEER, 5, 5, INF_NPC_STATS[INF_NPC_MELEER].size);
    state.npcs[0].active = 1;
    state.npcs[0].dig_freeze_timer = 1;

    state.pillars[0].active = 1;
    state.pillars[0].x = 17;
    state.pillars[0].y = 17;

    inf_meleer_dig_check(&state, 0);

    ASSERT_INT_EQ("blocked first landing candidate falls through to player tile x", state.npcs[0].x, 20);
    ASSERT_INT_EQ("blocked first landing candidate falls through to player tile y", state.npcs[0].y, 20);
    ASSERT_INT_EQ("dig freeze consumed", state.npcs[0].dig_freeze_timer, 0);
    ASSERT_INT_EQ("post-dig stun applied", state.npcs[0].stun_timer, 2);
    ASSERT_INT_EQ("post-dig attack delay applied", state.npcs[0].dig_attack_delay, 6);
}

static void test_melee_fallback_geometry(void) {
    printf("--- inferno melee fallback geometry ---\n");

    InfernoState diagonal_state = make_test_state(5, 5);
    InfernoState cardinal_state = make_test_state(5, 5);
    InfernoState distant_state = make_test_state(5, 5);

    InfNPC ranger_diagonal = make_test_npc(INF_NPC_RANGER, 6, 6, 1);
    InfNPC mager_diagonal = make_test_npc(INF_NPC_MAGER, 6, 6, 1);
    InfNPC blob_diagonal = make_test_npc(INF_NPC_BLOB, 6, 6, 1);
    InfNPC blob_cardinal = make_test_npc(INF_NPC_BLOB, 6, 5, 1);
    InfNPC jad_diagonal = make_test_npc(INF_NPC_JAD, 6, 6, 1);
    InfNPC jad_cardinal = make_test_npc(INF_NPC_JAD, 6, 5, 1);
    InfNPC blob_distant = make_test_npc(INF_NPC_BLOB, 7, 5, 1);

    InfNPCStats ranged_stats = make_test_stats(ATTACK_STYLE_RANGED);
    InfNPCStats magic_stats = make_test_stats(ATTACK_STYLE_MAGIC);

    ASSERT_INT_EQ(
        "ranger diagonal melee fallback",
        inf_melee_fallback_possible(
            &diagonal_state, &ranger_diagonal, &ranged_stats,
            ATTACK_STYLE_RANGED, distance_to_player(&diagonal_state, &ranger_diagonal)),
        1);
    ASSERT_INT_EQ(
        "mager diagonal melee fallback",
        inf_melee_fallback_possible(
            &diagonal_state, &mager_diagonal, &magic_stats,
            ATTACK_STYLE_MAGIC, distance_to_player(&diagonal_state, &mager_diagonal)),
        1);
    ASSERT_INT_EQ(
        "blob diagonal melee fallback blocked",
        inf_melee_fallback_possible(
            &diagonal_state, &blob_diagonal, &magic_stats,
            ATTACK_STYLE_MAGIC, distance_to_player(&diagonal_state, &blob_diagonal)),
        0);
    ASSERT_INT_EQ(
        "blob cardinal melee fallback",
        inf_melee_fallback_possible(
            &cardinal_state, &blob_cardinal, &magic_stats,
            ATTACK_STYLE_MAGIC, distance_to_player(&cardinal_state, &blob_cardinal)),
        1);
    ASSERT_INT_EQ(
        "jad diagonal melee fallback blocked",
        inf_melee_fallback_possible(
            &diagonal_state, &jad_diagonal, &ranged_stats,
            ATTACK_STYLE_RANGED, distance_to_player(&diagonal_state, &jad_diagonal)),
        0);
    ASSERT_INT_EQ(
        "jad cardinal melee fallback",
        inf_melee_fallback_possible(
            &cardinal_state, &jad_cardinal, &ranged_stats,
            ATTACK_STYLE_RANGED, distance_to_player(&cardinal_state, &jad_cardinal)),
        1);
    ASSERT_INT_EQ(
        "fallback blocked outside melee distance",
        inf_melee_fallback_possible(
            &distant_state, &blob_distant, &magic_stats,
            ATTACK_STYLE_MAGIC, distance_to_player(&distant_state, &blob_distant)),
        0);
    ASSERT_INT_EQ(
        "fallback blocked when planned style already melee",
        inf_melee_fallback_possible(
            &cardinal_state, &blob_cardinal, &magic_stats,
            ATTACK_STYLE_MELEE, distance_to_player(&cardinal_state, &blob_cardinal)),
        0);
}

static void test_style_mask_preview(void) {
    printf("--- inferno style mask preview ---\n");

    InfernoState diagonal_state = make_test_state(5, 5);
    InfernoState cardinal_state = make_test_state(5, 5);
    InfNPC ranger_diagonal = make_test_npc(INF_NPC_RANGER, 6, 6, 1);
    InfNPC blob_cardinal = make_test_npc(INF_NPC_BLOB, 6, 5, 1);
    InfNPC blob_diagonal = make_test_npc(INF_NPC_BLOB, 6, 6, 1);
    InfNPCStats ranged_stats = make_test_stats(ATTACK_STYLE_RANGED);
    InfNPCStats magic_stats = make_test_stats(ATTACK_STYLE_MAGIC);

    int ranger_mask = inf_attack_style_options_mask(
        &diagonal_state, &ranger_diagonal, &ranged_stats,
        ATTACK_STYLE_RANGED, distance_to_player(&diagonal_state, &ranger_diagonal));
    int blob_cardinal_mask = inf_attack_style_options_mask(
        &cardinal_state, &blob_cardinal, &magic_stats,
        ATTACK_STYLE_MAGIC, distance_to_player(&cardinal_state, &blob_cardinal));
    int blob_diagonal_mask = inf_attack_style_options_mask(
        &diagonal_state, &blob_diagonal, &magic_stats,
        ATTACK_STYLE_MAGIC, distance_to_player(&diagonal_state, &blob_diagonal));

    ASSERT_INT_EQ(
        "ranger diagonal preview mask",
        ranger_mask,
        INF_STYLE_MASK_MELEE | INF_STYLE_MASK_RANGED);
    ASSERT_INT_EQ(
        "ranger diagonal preview style is uncertain",
        inf_attack_style_from_mask(ranger_mask),
        ATTACK_STYLE_NONE);
    ASSERT_INT_EQ(
        "ranger diagonal obs preview keeps ranged primary",
        inf_attack_style_obs_preview(ranger_mask),
        ATTACK_STYLE_RANGED);
    ASSERT_INT_EQ(
        "blob cardinal preview mask",
        blob_cardinal_mask,
        INF_STYLE_MASK_MELEE | INF_STYLE_MASK_MAGIC);
    ASSERT_INT_EQ(
        "blob cardinal obs preview keeps magic primary",
        inf_attack_style_obs_preview(blob_cardinal_mask),
        ATTACK_STYLE_MAGIC);
    ASSERT_INT_EQ(
        "blob diagonal preview keeps magic only",
        blob_diagonal_mask,
        INF_STYLE_MASK_MAGIC);
    ASSERT_INT_EQ(
        "blob diagonal preview style is magic",
        inf_attack_style_from_mask(blob_diagonal_mask),
        ATTACK_STYLE_MAGIC);
}

static void test_style_choice_sampling(void) {
    printf("--- inferno style choice sampling ---\n");

    uint32_t rng_state = 12345;
    int saw_melee = 0;
    int saw_ranged = 0;

    for (int i = 0; i < 128; i++) {
        int style = inf_choose_attack_style_for_tick(
            &rng_state, INF_STYLE_MASK_MELEE | INF_STYLE_MASK_RANGED);
        if (style == ATTACK_STYLE_MELEE) saw_melee = 1;
        if (style == ATTACK_STYLE_RANGED) saw_ranged = 1;
    }

    ASSERT_INT_EQ("50-50 branch can emit melee", saw_melee, 1);
    ASSERT_INT_EQ("50-50 branch can emit primary style", saw_ranged, 1);
    ASSERT_INT_EQ(
        "single-style mask stays deterministic",
        inf_choose_attack_style_for_tick(&rng_state, INF_STYLE_MASK_MAGIC),
        ATTACK_STYLE_MAGIC);
}

static void test_dead_mob_store_eligibility(void) {
    printf("--- inferno dead mob resurrection eligibility ---\n");

    ASSERT_INT_EQ("bat resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BAT), 1);
    ASSERT_INT_EQ("blob parent resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB), 1);
    ASSERT_INT_EQ("meleer resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_MELEER), 1);
    ASSERT_INT_EQ("ranger resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_RANGER), 1);
    ASSERT_INT_EQ("mager resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_MAGER), 1);

    ASSERT_INT_EQ("nibbler not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_NIBBLER), 0);
    ASSERT_INT_EQ("blob melee split not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB_MELEE), 0);
    ASSERT_INT_EQ("blob range split not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB_RANGE), 0);
    ASSERT_INT_EQ("blob mage split not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB_MAGE), 0);
    ASSERT_INT_EQ("jad not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_JAD), 0);
}

static void test_resurrected_mob_does_not_reenter_dead_store(void) {
    printf("--- resurrected mob does not reenter dead store ---\n");

    InfernoState state = make_test_state(25, 16);
    state.wave = 35;

    state.npcs[0] = make_test_npc(INF_NPC_MAGER, 20, 20, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.dead_mobs[0].type = INF_NPC_RANGER;
    state.dead_mobs[0].x = 18;
    state.dead_mobs[0].y = 18;
    state.dead_mobs[0].hp = INF_NPC_STATS[INF_NPC_RANGER].hp / 2;
    state.dead_mobs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    state.dead_mob_count = 1;

    ASSERT_INT_EQ("resurrection succeeds", force_mager_resurrect(&state, 0), 1);
    ASSERT_INT_EQ("dead store consumed", state.dead_mob_count, 0);

    int resurrected_slot = -1;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (state.npcs[i].active && state.npcs[i].type == INF_NPC_RANGER) {
            resurrected_slot = i;
            break;
        }
    }
    ASSERT_INT_EQ("resurrected ranger spawned", resurrected_slot >= 0, 1);
    ASSERT_INT_EQ("respawned ranger marked resurrected",
        state.npcs[resurrected_slot].resurrection_count, 1);

    inf_store_dead_mob(&state, &state.npcs[resurrected_slot]);
    ASSERT_INT_EQ("resurrected ranger not re-added", state.dead_mob_count, 0);
}

static void test_double_mager_wave_resurrection_limit(void) {
    printf("--- double mager wave respects once-only resurrection ---\n");

    InfernoState state = make_test_state(25, 16);
    state.wave = 65;

    state.npcs[0] = make_test_npc(INF_NPC_MAGER, 18, 18, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.dead_mobs[0].type = INF_NPC_MAGER;
    state.dead_mobs[0].x = 22;
    state.dead_mobs[0].y = 22;
    state.dead_mobs[0].hp = INF_NPC_STATS[INF_NPC_MAGER].hp / 2;
    state.dead_mobs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    state.dead_mob_count = 1;

    ASSERT_INT_EQ("first mage resurrection succeeds", force_mager_resurrect(&state, 0), 1);

    int resurrected_slot = -1;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (state.npcs[i].active && state.npcs[i].type == INF_NPC_MAGER) {
            resurrected_slot = i;
            break;
        }
    }
    ASSERT_INT_EQ("second mager spawned", resurrected_slot >= 0, 1);
    ASSERT_INT_EQ("respawned mager marked resurrected",
        state.npcs[resurrected_slot].resurrection_count, 1);

    inf_store_dead_mob(&state, &state.npcs[0]);
    ASSERT_INT_EQ("original mage can still enter dead store once", state.dead_mob_count, 1);

    ASSERT_INT_EQ("resurrected mage can resurrect the original once",
        force_mager_resurrect(&state, resurrected_slot), 1);

    int original_respawn_slot = -1;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (i == resurrected_slot) continue;
        if (state.npcs[i].active && state.npcs[i].type == INF_NPC_MAGER &&
            state.npcs[i].resurrection_count == 1) {
            original_respawn_slot = i;
            break;
        }
    }
    ASSERT_INT_EQ("original mage respawned once", original_respawn_slot >= 0, 1);

    inf_store_dead_mob(&state, &state.npcs[resurrected_slot]);
    ASSERT_INT_EQ("already-resurrected mage stays out of store", state.dead_mob_count, 0);
    inf_store_dead_mob(&state, &state.npcs[original_respawn_slot]);
    ASSERT_INT_EQ("re-resurrected original mage stays out of store", state.dead_mob_count, 0);
}

static void test_pending_hit_obs_timer_prefers_prayer_window(void) {
    printf("--- pending hit obs timer prefers prayer window ---\n");

    EncounterPendingHit jad_hit = {0};
    jad_hit.check_prayer = 1;
    jad_hit.prayer_check_delay = 3;
    jad_hit.ticks_remaining = 4;

    EncounterPendingHit normal_hit = {0};
    normal_hit.check_prayer = 0;
    normal_hit.prayer_check_delay = 0;
    normal_hit.ticks_remaining = 2;

    ASSERT_INT_EQ("jad timer uses prayer window", inf_pending_hit_obs_timer(&jad_hit), 3);
    ASSERT_INT_EQ("normal timer uses travel time", inf_pending_hit_obs_timer(&normal_hit), 2);
}

static void test_jad_has_no_pre_fire_style_preview(void) {
    printf("--- jad has no pre-fire style preview ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.prayer = PRAYER_NONE;
    state.weapon_set = INF_GEAR_MAGE;
    state.wave = 66;

    state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 20, 10, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].attack_timer = 2;

    inf_npc_attack(&state, 0);

    ASSERT_INT_EQ("jad timer decrements without preview", state.npcs[0].attack_timer, 1);
    ASSERT_INT_EQ("jad style stays hidden before fire", state.npcs[0].jad_attack_style, ATTACK_STYLE_NONE);

    float obs[INF_NUM_OBS];
    inf_write_obs((EncounterState*)&state, obs);
    ASSERT_FLOAT_NEAR("prayer-critical timer ignores hidden jad style", obs[37], 1.0f, 1e-6f);
    ASSERT_INT_EQ("prayer-critical style stays zero before fire", (int)(obs[38] + obs[39] + obs[40]), 0);
}

static void test_jad_fire_tick_exposes_three_tick_prayer_deadline(void) {
    printf("--- jad fire tick exposes three tick prayer deadline ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);

    step_inferno_with_prayer(&state, 0);

    ASSERT_INT_EQ("jad attack queued one pending hit", state.player_pending_hit_count, 1);
    ASSERT_INT_EQ("jad style resets after firing", state.npcs[0].jad_attack_style, ATTACK_STYLE_NONE);
    ASSERT_INT_EQ("jad pending hit shows three tick prayer delay after fire", state.player_pending_hits[0].prayer_check_delay, 3);
    ASSERT_INT_EQ("jad close-range hit lands four ticks after fire", state.player_pending_hits[0].ticks_remaining, 4);

    float obs[INF_NUM_OBS];
    memset(obs, 0, sizeof(obs));
    inf_write_obs((EncounterState*)&state, obs);
    ASSERT_FLOAT_NEAR("prayer-critical timer exposes jad fire deadline", obs[37], 0.3f, 1e-6f);
    ASSERT_FLOAT_NEAR("prayer-critical magic style exposed after fire", obs[40], 1.0f, 1e-6f);
    int pending_start = INF_NUM_OBS - INF_FEATURES_PER_HIT * ENCOUNTER_MAX_PENDING_HITS;
    ASSERT_FLOAT_NEAR("pending hit obs timer uses prayer window", obs[pending_start + 3], 0.3f, 1e-6f);
    ASSERT_FLOAT_NEAR("pending hit pre-check damage exposes max threat", obs[pending_start + 4], 113.0f / 150.0f, 1e-6f);
}

static void test_jad_prayer_on_third_tick_blocks(void) {
    printf("--- jad prayer on third tick blocks ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);

    step_inferno_with_prayer(&state, 0);
    step_inferno_with_prayer(&state, 0);
    step_inferno_with_prayer(&state, 0);
    step_inferno_with_prayer(&state, ENCOUNTER_OVERHEAD_TOGGLE_MAGIC);

    ASSERT_INT_EQ("jad prayer check consumed pending protection", state.player_pending_hits[0].check_prayer, 0);
    ASSERT_INT_EQ("jad protected damage is frozen at zero", state.player_pending_hits[0].damage, 0);
    ASSERT_INT_EQ("jad prayer check counted correct prayer", state.prayer_correct_this_tick, 1);

    step_inferno_with_prayer(&state, 0);
    ASSERT_INT_EQ("jad protected hit removed after landing", state.player_pending_hit_count, 0);
    ASSERT_INT_EQ("jad protected hit leaves player hp unchanged", state.player.current_hitpoints, 99);
}

static void test_jad_prayer_first_on_fourth_tick_does_not_block(void) {
    printf("--- jad prayer first on fourth tick does not block ---\n");

    int saw_late_damage = 0;
    for (uint32_t seed = 1; seed < 10000 && !saw_late_damage; seed++) {
        InfernoState state;
        init_jad_timing_test_state(&state, 10, 10, 16, 10);
        state.rng_state = seed;

        step_inferno_with_prayer(&state, 0);
        step_inferno_with_prayer(&state, 0);
        step_inferno_with_prayer(&state, 0);
        step_inferno_with_prayer(&state, 0);
        ASSERT_INT_EQ("late-prayer test reaches checked pending hit", state.player_pending_hits[0].check_prayer, 0);

        step_inferno_with_prayer(&state, ENCOUNTER_OVERHEAD_TOGGLE_MAGIC);
        if (state.damage_received_this_tick > 0.0f) {
            saw_late_damage = 1;
            ASSERT_INT_EQ("late prayer did not block queued jad damage", state.player.current_hitpoints < 99, 1);
        }
    }
    ASSERT_INT_EQ("found a seed where late jad prayer takes damage", saw_late_damage, 1);
}

static void test_jad_long_distance_damage_uses_delayed_projectile_landing(void) {
    printf("--- jad long distance damage uses delayed projectile landing ---\n");

    int saw_expected_landing = 0;
    for (uint32_t seed = 1; seed < 10000 && !saw_expected_landing; seed++) {
        InfernoState state;
        init_jad_timing_test_state(&state, 10, 10, 36, 10);
        state.rng_state = seed;

        int dist = encounter_dist_to_npc(state.player.x, state.player.y,
            state.npcs[0].x, state.npcs[0].y, state.npcs[0].size);
        int hit_delay = encounter_magic_hit_delay(dist, 0);
        int expected_landing_after_fire = 3 + (hit_delay - 3);
        if (expected_landing_after_fire < 4)
            expected_landing_after_fire = 4;

        step_inferno_with_prayer(&state, 0);
        for (int t = 1; t < expected_landing_after_fire; t++) {
            step_inferno_with_prayer(&state, 0);
            ASSERT_FLOAT_NEAR("jad long-distance hit has not landed early", state.damage_received_this_tick, 0.0f, 1e-6f);
        }
        step_inferno_with_prayer(&state, 0);
        if (state.damage_received_this_tick > 0.0f) {
            saw_expected_landing = 1;
        }
    }
    ASSERT_INT_EQ("found a seed where long-distance jad damage lands on expected tick", saw_expected_landing, 1);
}

static void test_triple_jad_pending_threats_preserve_obs_shape(void) {
    printf("--- triple jad pending threats preserve obs shape ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 25, 30, 18, 33);
    state.wave = 67;
    state.npcs[1] = make_test_npc(INF_NPC_JAD, 28, 33, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[1].active = 1;
    state.npcs[1].attack_timer = 0;
    state.npcs[1].jad_attack_style = ATTACK_STYLE_RANGED;
    state.npcs[2] = make_test_npc(INF_NPC_JAD, 23, 22, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[2].active = 1;
    state.npcs[2].attack_timer = 0;
    state.npcs[2].jad_attack_style = ATTACK_STYLE_MAGIC;

    step_inferno_with_prayer(&state, 0);

    ASSERT_INT_EQ("triple jad queues three pending threats", state.player_pending_hit_count, 3);
    for (int h = 0; h < state.player_pending_hit_count; h++) {
        ASSERT_INT_EQ("each jad threat keeps three tick prayer deadline", state.player_pending_hits[h].prayer_check_delay, 3);
    }

    float obs[INF_NUM_OBS];
    inf_write_obs((EncounterState*)&state, obs);
    ASSERT_INT_EQ("inferno obs shape remains unchanged", INF_NUM_OBS, 386);
}

static void test_jad_special_wave_spawn_cadence_matches_reference(void) {
    printf("--- jad special wave spawn cadence matches reference ---\n");

    InfernoState single = make_test_state(0, 0);
    single.wave = 66;
    inf_spawn_wave(&single);

    int single_jad = -1;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (single.npcs[i].active && single.npcs[i].type == INF_NPC_JAD) {
            single_jad = i;
            break;
        }
    }
    ASSERT_INT_EQ("wave 67 spawns one jad", single_jad >= 0, 1);
    ASSERT_INT_EQ("wave 67 jad stun", single.npcs[single_jad].stun_timer, 1);
    ASSERT_INT_EQ("wave 67 jad attack speed timer", single.npcs[single_jad].attack_timer, 8);

    InfernoState triple = make_test_state(0, 0);
    triple.wave = 67;
    triple.rng_state = 12345;
    inf_spawn_wave(&triple);

    int num_jads = 0;
    int stun_sum = 0;
    int stun_product = 1;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!triple.npcs[i].active || triple.npcs[i].type != INF_NPC_JAD)
            continue;
        num_jads++;
        stun_sum += triple.npcs[i].stun_timer;
        stun_product *= triple.npcs[i].stun_timer;
        ASSERT_INT_EQ("wave 68 jad attack speed timer", triple.npcs[i].attack_timer, 9);
    }
    ASSERT_INT_EQ("wave 68 spawns three jads", num_jads, 3);
    ASSERT_INT_EQ("wave 68 shuffled stun sum", stun_sum, 12);
    ASSERT_INT_EQ("wave 68 shuffled stun product", stun_product, 28);
}

static void test_jad_melee_stays_instant_and_untelegraphed(void) {
    printf("--- jad melee stays instant and untelegraphed ---\n");

    InfernoState preview_state = make_test_state(5, 5);
    preview_state.player.current_defence = 99;
    preview_state.player.current_magic = 99;
    preview_state.player.prayer = PRAYER_NONE;
    preview_state.weapon_set = INF_GEAR_MAGE;
    preview_state.wave = 66;

    preview_state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 6, 5, INF_NPC_STATS[INF_NPC_JAD].size);
    preview_state.npcs[0].active = 1;
    preview_state.npcs[0].attack_timer = 1;
    preview_state.npcs[0].jad_attack_style = ATTACK_STYLE_RANGED;

    float obs[INF_NUM_OBS];
    inf_write_obs((EncounterState*)&preview_state, obs);

    ASSERT_FLOAT_NEAR(
        "jad prayer-critical preview does not advertise melee fallback",
        obs[41], 1.0f / 3.0f, 1e-6f);
    ASSERT_INT_EQ(
        "jad prayer-critical preview keeps ranged one-hot",
        (int)(obs[38] + obs[39] + obs[40]),
        1);
    ASSERT_FLOAT_NEAR(
        "jad preview keeps ranged as the visible style",
        obs[39], 1.0f, 1e-6f);

    int saw_melee = 0;
    for (uint32_t seed = 0; seed < 256; seed++) {
        InfernoState attack_state = make_test_state(5, 5);
        attack_state.rng_state = seed;
        attack_state.player.current_defence = 99;
        attack_state.player.current_magic = 99;
        attack_state.player.prayer = PRAYER_NONE;
        attack_state.weapon_set = INF_GEAR_MAGE;
        attack_state.wave = 66;

        attack_state.npcs[0] = make_test_npc(
            INF_NPC_JAD, 6, 5, INF_NPC_STATS[INF_NPC_JAD].size);
        attack_state.npcs[0].active = 1;
        attack_state.npcs[0].attack_timer = 0;
        attack_state.npcs[0].jad_attack_style = ATTACK_STYLE_RANGED;

        inf_npc_attack(&attack_state, 0);

        if (attack_state.npcs[0].attack_style_this_tick == ATTACK_STYLE_MELEE) {
            saw_melee = 1;
            ASSERT_INT_EQ(
                "jad melee fallback does not queue a pending hit",
                attack_state.player_pending_hit_count, 0);
            break;
        }
    }

    ASSERT_INT_EQ("jad can still choose melee instantly at fire time", saw_melee, 1);
}

static int inferno_obs_slot_type(int slot_idx) {
    if (slot_idx >= 0 && slot_idx < 2) return INF_NPC_MAGER;
    if (slot_idx >= 2 && slot_idx < 4) return INF_NPC_RANGER;
    if (slot_idx >= 4 && slot_idx < 6) return INF_NPC_MELEER;
    if (slot_idx >= 6 && slot_idx < 8) return INF_NPC_BLOB;
    if (slot_idx >= 8 && slot_idx < 10) return INF_NPC_BAT;
    if (slot_idx >= 10 && slot_idx < 12) return INF_NPC_BLOB_MAGE;
    if (slot_idx >= 12 && slot_idx < 14) return INF_NPC_BLOB_RANGE;
    if (slot_idx >= 14 && slot_idx < 16) return INF_NPC_BLOB_MELEE;
    if (slot_idx >= 16 && slot_idx < 22) return INF_NPC_NIBBLER;
    if (slot_idx >= 22 && slot_idx < 25) return INF_NPC_JAD;
    if (slot_idx == 25) return INF_NPC_ZUK;
    if (slot_idx == 26) return INF_NPC_ZUK_SHIELD;
    if (slot_idx >= 27 && slot_idx < 33) return INF_NPC_HEALER_JAD;
    if (slot_idx >= 33 && slot_idx < 37) return INF_NPC_HEALER_ZUK;
    return -1;
}

static int inferno_obs_slot_feature_count(int slot_idx) {
    int type = inferno_obs_slot_type(slot_idx);
    int has_style = (type == INF_NPC_BLOB || type == INF_NPC_JAD);
    int has_scan = (type == INF_NPC_BLOB);
    int has_los = (type != INF_NPC_NIBBLER && type != INF_NPC_MELEER &&
        type != INF_NPC_HEALER_JAD && type != INF_NPC_ZUK_SHIELD);
    int has_aggro = (type != INF_NPC_NIBBLER && type != INF_NPC_ZUK_SHIELD);
    int has_timer = (type != INF_NPC_NIBBLER && type != INF_NPC_HEALER_JAD &&
        type != INF_NPC_ZUK_SHIELD);
    int has_targeted = 1;

    return 4 + has_timer + 3 * has_style + has_los + 3 * has_scan +
        has_aggro + has_targeted;
}

static int inferno_obs_slot_start(int slot_idx) {
    int start = INF_PLAYER_OBS_SIZE + 12;
    for (int i = 0; i < slot_idx; i++) {
        start += inferno_obs_slot_feature_count(i);
    }
    return start;
}

static int inferno_target_mask_slot_offset(int slot_idx) {
    return ENCOUNTER_MOVE_ACTIONS + ENCOUNTER_OVERHEAD_DIM_PVE + 1 + slot_idx;
}

static void test_zuk_obs_tracks_shield_and_mager_aggro(void) {
    printf("--- zuk obs tracks shield hp/death and mager aggro ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.wave = 68;
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.current_ranged = 99;
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.prayer = PRAYER_NONE;
    state.weapon_set = INF_GEAR_TBOW;

    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 20, 36, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    state.npcs[0].attack_timer = 4;

    state.npcs[1] = make_test_npc(
        INF_NPC_ZUK, 22, 14, INF_NPC_STATS[INF_NPC_ZUK].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_ZUK].hp;

    state.npcs[2] = make_test_npc(
        INF_NPC_ZUK_SHIELD, 23, 44, INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = 300;
    state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].hp;

    state.zuk.shield_idx = 2;
    state.zuk.shield_dir = -1;
    state.zuk.shield_freeze = 3;
    state.npcs[0].aggro_target = 2;

    float obs[INF_NUM_OBS];
    float mask[INF_ACTION_MASK_SIZE];
    inf_write_obs((EncounterState*)&state, obs);
    inf_write_mask((EncounterState*)&state, mask);

    int mager_slot = 0;
    int shield_slot = 26;
    int mager_start = inferno_obs_slot_start(mager_slot);
    int shield_start = inferno_obs_slot_start(shield_slot);

    ASSERT_INT_EQ("first mager occupies mager slot 0", state.current_obs_slots[mager_slot], 0);
    ASSERT_INT_EQ("shield occupies dedicated shield slot", state.current_obs_slots[shield_slot], 2);
    ASSERT_FLOAT_NEAR("shield hp ratio visible in shield slot", obs[shield_start], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("mager aggro bit is 0 while on shield", obs[mager_start + 6], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("shield direction visible while alive", obs[43], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("shield freeze visible while alive", obs[44], 0.6f, 1e-6f);
    ASSERT_FLOAT_NEAR("mager target mask is valid", mask[inferno_target_mask_slot_offset(mager_slot)], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("shield target mask stays invalid", mask[inferno_target_mask_slot_offset(shield_slot)], 0.0f, 1e-6f);

    state.npcs[2].active = 0;
    state.zuk.shield_idx = -1;
    state.npcs[0].aggro_target = -1;

    memset(obs, 0, sizeof(obs));
    memset(mask, 0, sizeof(mask));
    inf_write_obs((EncounterState*)&state, obs);
    inf_write_mask((EncounterState*)&state, mask);

    ASSERT_INT_EQ("dead shield drops out of shield slot", state.current_obs_slots[shield_slot], -1);
    ASSERT_FLOAT_NEAR("dead shield slot hp zeros out", obs[shield_start], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("dead shield zeroes stale direction", obs[43], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("dead shield zeroes stale freeze", obs[44], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("mager aggro bit flips to player", obs[mager_start + 6], 1.0f, 1e-6f);
}

static void test_human_target_and_potion_translation(void) {
    printf("--- inferno human target and potion translation ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 80;
    state.player.base_prayer = 99;
    state.player.current_prayer = 60;
    state.player.current_attack = 99;
    state.player.current_strength = 99;
    state.player.current_defence = 99;
    state.player.current_ranged = 99;
    state.player.current_magic = 99;

    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 24, 24, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.npcs[1] = make_test_npc(
        INF_NPC_MAGER, 26, 24, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.npcs[2] = make_test_npc(
        INF_NPC_MAGER, 28, 24, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.npcs[3] = make_test_npc(
        INF_NPC_ZUK_SHIELD, 23, 44, INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size);
    state.npcs[3].active = 1;
    state.npcs[3].hp = state.npcs[3].max_hp = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].hp;

    {
        float obs[INF_NUM_OBS];
        inf_write_obs((EncounterState*)&state, obs);
    }

    ASSERT_INT_EQ("first visible mager is targetable",
        inf_is_human_targetable_npc_slot((EncounterState*)&state, 0), 1);
    ASSERT_INT_EQ("second visible mager is targetable",
        inf_is_human_targetable_npc_slot((EncounterState*)&state, 1), 1);
    ASSERT_INT_EQ("third capped-out mager is not targetable",
        inf_is_human_targetable_npc_slot((EncounterState*)&state, 2), 0);
    ASSERT_INT_EQ("shield is never targetable",
        inf_is_human_targetable_npc_slot((EncounterState*)&state, 3), 0);

    {
        HumanInput hi;
        int actions[INF_NUM_ACTION_HEADS];

        hi = make_human_input();
        hi.pending_target_idx = 0;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("visible mager click maps into target head",
            actions[INF_HEAD_TARGET], 1);

        hi = make_human_input();
        hi.pending_target_idx = 2;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("capped-out mager click is rejected",
            actions[INF_HEAD_TARGET], 0);

        hi = make_human_input();
        hi.pending_target_idx = 3;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("shield click is rejected",
            actions[INF_HEAD_TARGET], 0);

        hi = make_human_input();
        hi.pending_potion = POTION_BREW;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("brew still maps to eat head", actions[INF_HEAD_EAT], 1);
        ASSERT_INT_EQ("brew does not touch potion head", actions[INF_HEAD_POTION], 0);

        hi = make_human_input();
        hi.pending_potion = POTION_RESTORE;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("restore maps to potion 1", actions[INF_HEAD_POTION], 1);

        hi = make_human_input();
        hi.pending_potion = POTION_BASTION;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("bastion maps to potion 2", actions[INF_HEAD_POTION], 2);

        hi = make_human_input();
        hi.pending_potion = POTION_STAMINA;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("stamina maps to potion 3", actions[INF_HEAD_POTION], 3);

        hi = make_human_input();
        hi.pending_potion = POTION_PRAYER_POT;
        inf_translate_human_input(&hi, actions, (EncounterState*)&state);
        ASSERT_INT_EQ("prayer pot no longer aliases to restore",
            actions[INF_HEAD_POTION], 0);
    }
}

static void test_action_noop_count_matches_action_heads(void) {
    printf("--- action noop count matches inferno action heads ---\n");

    int noop_slots = (int)(
        sizeof(((InfernoState*)0)->action_noop_count) /
        sizeof(((InfernoState*)0)->action_noop_count[0]));
    ASSERT_INT_EQ("noop counter slots", noop_slots, INF_NUM_ACTION_HEADS);
}

static void test_inferno_human_equip_does_not_snap_loadout(void) {
    printf("--- inferno human equip does not snap full loadout ---\n");

    EncounterState* raw = inf_create();
    InfernoState* state = (InfernoState*)raw;
    inf_reset(raw, 123);

    HumanInput input;
    human_input_init(&input);
    input.enabled = 1;

    uint8_t old_body = state->player.equipped[GEAR_SLOT_BODY];
    human_input_queue_equip_inventory_item(
        &input, 0, ITEM_TOXIC_BLOWPIPE, GEAR_SLOT_WEAPON);

    inf_step_human_commands(raw, &input);

    ASSERT_INT_EQ("weapon changed to clicked blowpipe",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_TOXIC_BLOWPIPE);
    ASSERT_INT_EQ("body slot did not snap to ranged preset",
        state->player.equipped[GEAR_SLOT_BODY], old_body);
    ASSERT_INT_EQ("2h weapon clears shield",
        state->player.equipped[GEAR_SLOT_SHIELD], ITEM_NONE);
    ASSERT_INT_EQ("queued command drained", input.commands.count, 0);

    human_input_destroy(&input);
    inf_destroy(raw);
}

static void test_jad_render_uses_style_specific_attack_animation(void) {
    printf("--- jad render uses style-specific attack animation ---\n");

    InfernoState magic_state;
    init_jad_timing_test_state(&magic_state, 10, 10, 16, 10);
    magic_state.npcs[0].attacked_this_tick = 1;
    magic_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    RenderEntity magic_entities[4];
    int magic_count = 0;
    inf_fill_render_entities((EncounterState*)&magic_state, magic_entities, 4, &magic_count);

    InfernoState range_state;
    init_jad_timing_test_state(&range_state, 10, 10, 16, 10);
    range_state.npcs[0].attacked_this_tick = 1;
    range_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_RANGED;

    RenderEntity range_entities[4];
    int range_count = 0;
    inf_fill_render_entities((EncounterState*)&range_state, range_entities, 4, &range_count);

    ASSERT_INT_EQ("jad magic render entity count", magic_count, 2);
    ASSERT_INT_EQ("jad ranged render entity count", range_count, 2);
    ASSERT_INT_EQ("jad magic attack animation", magic_entities[1].npc_anim_id, 7592);
    ASSERT_INT_EQ("jad ranged attack animation", range_entities[1].npc_anim_id, 7593);
}

static void test_jad_magic_render_emits_three_offset_projectiles(void) {
    printf("--- jad magic render emits three offset projectiles ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);
    state.npcs[0].attacked_this_tick = 1;
    state.npcs[0].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick((EncounterState*)&state, &ov);

    ASSERT_INT_EQ("jad magic emits three projectile models", ov.projectile_count, 3);
    ASSERT_INT_EQ("jad magic front model", ov.projectiles[0].model_id, INF_GFX_448_MODEL);
    ASSERT_INT_EQ("jad magic middle model", ov.projectiles[1].model_id, INF_GFX_449_MODEL);
    ASSERT_INT_EQ("jad magic rear model", ov.projectiles[2].model_id, INF_GFX_450_MODEL);
    ASSERT_INT_EQ("jad magic front anim", ov.projectiles[0].anim_id, INF_GFX_448_ANIM);
    ASSERT_INT_EQ("jad magic middle anim", ov.projectiles[1].anim_id, INF_GFX_449_ANIM);
    ASSERT_INT_EQ("jad magic rear anim", ov.projectiles[2].anim_id, INF_GFX_450_ANIM);
    ASSERT_INT_EQ("jad magic visible duration is two ticks close range", ov.projectiles[0].duration_ticks, 2 * 30);
    ASSERT_INT_EQ("jad magic start delay is three ticks", ov.projectiles[0].start_delay, 3 * 30);
    ASSERT_FLOAT_NEAR("jad magic arc height", ov.projectiles[0].arc_height, 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("jad magic front offset", ov.projectiles[0].offset_y, 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("jad magic middle offset", ov.projectiles[1].offset_y, 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("jad magic rear offset", ov.projectiles[2].offset_y, 0.0f, 1e-6f);
}

static void test_jad_ranged_render_uses_target_anchored_two_tick_visual(void) {
    printf("--- jad ranged render uses target anchored two tick visual ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);
    state.npcs[0].attacked_this_tick = 1;
    state.npcs[0].attack_style_this_tick = ATTACK_STYLE_RANGED;

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick((EncounterState*)&state, &ov);

    ASSERT_INT_EQ("jad ranged emits one projectile", ov.projectile_count, 1);
    ASSERT_INT_EQ("jad ranged model", ov.projectiles[0].model_id, INF_GFX_451_MODEL);
    ASSERT_INT_EQ("jad ranged anim", ov.projectiles[0].anim_id, INF_GFX_451_ANIM);
    ASSERT_INT_EQ("jad ranged target-anchored motion",
        ov.projectiles[0].motion_mode, ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED);
    ASSERT_INT_EQ("jad ranged start height is player target height", ov.projectiles[0].start_h, 64);
    ASSERT_INT_EQ("jad ranged end height is player target height", ov.projectiles[0].end_h, 64);
    ASSERT_INT_EQ("jad ranged visible duration is two ticks close range", ov.projectiles[0].duration_ticks, 2 * 30);
    ASSERT_INT_EQ("jad ranged start delay is three ticks", ov.projectiles[0].start_delay, 3 * 30);
}

static void test_jad_projectile_long_distance_visual_duration_uses_reference_formula(void) {
    printf("--- jad long-distance projectile visual duration uses reference formula ---\n");

    InfernoState range_state;
    init_jad_timing_test_state(&range_state, 10, 10, 36, 10);
    range_state.npcs[0].attacked_this_tick = 1;
    range_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_RANGED;

    EncounterOverlay range_ov;
    memset(&range_ov, 0, sizeof(range_ov));
    inf_render_post_tick((EncounterState*)&range_state, &range_ov);

    int range_dist = encounter_dist_to_npc(
        range_state.player.x, range_state.player.y,
        range_state.npcs[0].x, range_state.npcs[0].y,
        range_state.npcs[0].size);
    int range_flight_ticks = encounter_ranged_hit_delay(range_dist, 0) - INF_JAD_PROJECTILE_DELAY;
    if (range_flight_ticks < 1) range_flight_ticks = 1;
    ASSERT_INT_EQ("jad ranged long-distance duration",
        range_ov.projectiles[0].duration_ticks, (range_flight_ticks + 1) * 30);

    InfernoState magic_state;
    init_jad_timing_test_state(&magic_state, 10, 10, 36, 10);
    magic_state.npcs[0].attacked_this_tick = 1;
    magic_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    EncounterOverlay magic_ov;
    memset(&magic_ov, 0, sizeof(magic_ov));
    inf_render_post_tick((EncounterState*)&magic_state, &magic_ov);

    int magic_dist = encounter_dist_to_npc(
        magic_state.player.x, magic_state.player.y,
        magic_state.npcs[0].x, magic_state.npcs[0].y,
        magic_state.npcs[0].size);
    int magic_flight_ticks = encounter_magic_hit_delay(magic_dist, 0) - INF_JAD_PROJECTILE_DELAY;
    if (magic_flight_ticks < 1) magic_flight_ticks = 1;
    ASSERT_INT_EQ("jad magic long-distance duration",
        magic_ov.projectiles[0].duration_ticks, (magic_flight_ticks + 1) * 30);
}

int main(void) {
    inf_build_npc_stats();

    test_melee_fallback_geometry();
    test_style_mask_preview();
    test_style_choice_sampling();
    test_tagged_jad_healer_melee_geometry();
    test_overlap_shuffle_hold_after_recent_target_click();
    test_overlap_shuffle_respects_npc_collision_flags();
    test_tagged_jad_healer_stops_at_melee_contact();
    test_tagged_jad_healers_queue_behind_front_healer();
    test_stacked_npc_unclipping_clears_flag_when_one_leaves();
    test_jad_healer_spawn_offsets_match_wave_67_reference();
    test_jad_healer_spawn_offsets_match_zuk_reference();
    test_meleer_dig_landing_order();
    test_reward_switches_between_healer_tags_and_damage();
    test_final_wave_reward_uses_zuk_low_watermark_progress();
    test_inferno_reset_supplies_match_current_inventory();
    test_late_start_supply_profile_anchor_waves();
    test_late_start_supply_profile_interpolation_and_scale();
    test_late_start_supply_observations();
    test_dead_mob_store_eligibility();
    test_resurrected_mob_does_not_reenter_dead_store();
    test_double_mager_wave_resurrection_limit();
    test_pending_hit_obs_timer_prefers_prayer_window();
    test_jad_has_no_pre_fire_style_preview();
    test_jad_fire_tick_exposes_three_tick_prayer_deadline();
    test_jad_prayer_on_third_tick_blocks();
    test_jad_prayer_first_on_fourth_tick_does_not_block();
    test_jad_long_distance_damage_uses_delayed_projectile_landing();
    test_triple_jad_pending_threats_preserve_obs_shape();
    test_jad_special_wave_spawn_cadence_matches_reference();
    test_jad_melee_stays_instant_and_untelegraphed();
    test_zuk_obs_tracks_shield_and_mager_aggro();
    test_human_target_and_potion_translation();
    test_action_noop_count_matches_action_heads();
    test_inferno_human_equip_does_not_snap_loadout();
    test_jad_render_uses_style_specific_attack_animation();
    test_jad_magic_render_emits_three_offset_projectiles();
    test_jad_ranged_render_uses_target_anchored_two_tick_visual();
    test_jad_projectile_long_distance_visual_duration_uses_reference_formula();

    printf("\n%d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d failed)\n", tests_failed);
        return 1;
    }
    printf("\n");
    return 0;
}
