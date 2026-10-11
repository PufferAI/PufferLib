/* Simulation, equipment, analytics and Puffer adapter snapshot tests.
 * Compiled and run by test_fight_caves_integration.py; no display needed. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "fight_caves.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char* message) {
    fprintf(stderr, "core_contract_test: %s\n", message);
    exit(EXIT_FAILURE);
}

static void wave_rotation_test(void) {
    /* FNV-1a fingerprints captured from the expanded table before factoring.
     * Each wave covers all 15 rotations and all six slots, including padding. */
    static const uint32_t expected[FC_NUM_WAVES] = {
        0x94a6ae61u, 0xb43e53bdu, 0x5ae991d9u, 0xb87f83cdu,
        0x7ae7bc55u, 0xc2eb4881u, 0x6f091049u, 0x2c80a9b9u,
        0xff5bb385u, 0x0460e561u, 0x3fd38821u, 0x70d9d565u,
        0x3033fbb1u, 0x39ff4ee5u, 0xfb88abe1u, 0x42db8009u,
        0xcd6d9305u, 0x75d5d409u, 0xecd94691u, 0xe481f925u,
        0x047e78e1u, 0xbcec175du, 0x28ee1f11u, 0xe225e405u,
        0xee2ea345u, 0x6e841145u, 0x12b64e1du, 0xf7a76349u,
        0xf41e3f01u, 0xf9d9e13du, 0x94a6ae61u, 0x7b92c141u,
        0xe6d41269u, 0xb87f83cdu, 0x4efe59e1u, 0xd0ef2aa9u,
        0x7cf6df61u, 0x2c80a9b9u, 0xf9ba9da9u, 0xffa136e9u,
        0x3fd38821u, 0x179231cdu, 0x45be2b59u, 0x1c645eddu,
        0x1ea54635u, 0x42db8009u, 0x538afaf1u, 0x8c6cad2du,
        0xecd94691u, 0x0cf73279u, 0xa339fc69u, 0xfeddd769u,
        0x28ee1f11u, 0x67f6d72du, 0x1b3ebe39u, 0x6e841145u,
        0xbc40ffe9u, 0x66ac8f75u, 0xf9552f4du, 0x3de75731u,
        0x4a20ae1du, 0xb43e53bdu, 0x5ae991d9u,
    };
    for (int wave = 1; wave <= FC_NUM_WAVES; wave++) {
        uint32_t hash = 2166136261u;
        for (int rotation = 0; rotation < FC_NUM_ROTATIONS; rotation++) {
            for (int slot = 0; slot < FC_MAX_SPAWNS_PER_WAVE; slot++) {
                int direction = fc_wave_spawn_dir(wave, rotation, slot);
                if (direction < SPAWN_SOUTH || direction > SPAWN_CENTER)
                    fail("wave spawn direction is out of range");
                hash = (hash ^ (uint32_t)direction) * 16777619u;
            }
        }
        if (hash != expected[wave - 1]) {
            fprintf(stderr, "wave_rotation_test: wave %d changed\n", wave);
            fail("wave rotation fixture mismatch");
        }
    }
    if (fc_wave_spawn_dir(0, 0, 0) != SPAWN_CENTER ||
        fc_wave_spawn_dir(FC_NUM_WAVES + 1, 0, 0) != SPAWN_CENTER ||
        fc_wave_spawn_dir(1, -1, 0) != SPAWN_CENTER ||
        fc_wave_spawn_dir(1, FC_NUM_ROTATIONS, 0) != SPAWN_CENTER ||
        fc_wave_spawn_dir(1, 0, -1) != SPAWN_CENTER ||
        fc_wave_spawn_dir(1, 0, FC_MAX_SPAWNS_PER_WAVE) != SPAWN_CENTER)
        fail("invalid wave lookup fallback changed");
    puts("wave_rotation_test: all waves, rotations, slots and invalid inputs passed");
}

static void check_observation(const FcState* state) {
    float obs[FC_TOTAL_OBS];
    float mask[FC_ACTION_MASK_SIZE];
    fc_write_obs(state, obs);
    fc_write_mask(state, mask);
    for (int i = 0; i < FC_TOTAL_OBS; i++) {
        if (!isfinite(obs[i])) fail("observation contains a non-finite value");
    }
    for (int i = 0; i < FC_ACTION_MASK_SIZE; i++) {
        if (mask[i] != 0.0f && mask[i] != 1.0f)
            fail("action mask contains a value other than zero or one");
    }
    int offset = 0;
    for (int head = 0; head < FC_NUM_ACTION_HEADS; head++) {
        int legal = 0;
        for (int action = 0; action < FC_ACTION_DIMS[head]; action++)
            legal += mask[offset + action] == 1.0f;
        if (legal == 0) fail("an action head has no legal action");
        offset += FC_ACTION_DIMS[head];
    }
}

static int core_contract_test(void) {
    _Static_assert(FC_STATE_HASH_VERSION == 6, "analytics state hash version drifted");
    _Static_assert(FC_POLICY_OBS_SIZE == 286, "policy observation contract drifted");
    _Static_assert(FC_PUFFER_OBS_SIZE == 320, "Puffer observation contract drifted");
    _Static_assert(FC_PUFFER_MASK_SIZE == 34, "Puffer mask contract drifted");
    _Static_assert(FC_PUFFER_NUM_ATNS == 3, "Puffer action-head count drifted");

    FcState first;
    FcState second;
    fc_init(&first);
    fc_init(&second);
    fc_reset(&first, 0x12345678u);
    fc_reset(&second, 0x12345678u);
    if (fc_state_hash(&first) != fc_state_hash(&second))
        fail("same-seed resets are not deterministic");
    check_observation(&first);

    int steps = 0;
    for (; steps < 4096 && !fc_is_terminal(&first); steps++) {
        int actions[FC_NUM_ACTION_HEADS] = {0};
        actions[0] = steps % FC_PUFFER_ACTION_DIMS[0];
        actions[1] = (steps / 3) % FC_PUFFER_ACTION_DIMS[1];
        actions[2] = (steps / 7) % FC_PUFFER_ACTION_DIMS[2];
        fc_step(&first, actions);
        fc_step(&second, actions);
        if (fc_state_hash(&first) != fc_state_hash(&second))
            fail("same-seed trajectories diverged");
        check_observation(&first);
    }
    if (steps == 0) fail("simulation did not advance");
    if (!fc_is_terminal(&first))
        fail("test trajectory did not exercise a terminal transition");

    if (steps != 483 || fc_state_hash(&first) != 0x5c53c26du)
        fail("fixed-seed trajectory changed; review and update the contract fixture intentionally");

    printf("core_contract_test: passed (%d steps, hash=%08x)\n",
           steps, fc_state_hash(&first));
    fc_destroy(&first);
    fc_destroy(&second);
    return EXIT_SUCCESS;
}

/* Equipment transactions and episode statistics. */

#define CHECK(test) do { if (!(test)) { \
    fprintf(stderr, "equipment: line %d: %s\n", __LINE__, #test); return 1; \
} } while (0)

static void reset(FcState *s, int empty_inventory) {
    fc_init(s);
    fc_reset(s, 101);
    if (empty_inventory) fc_set_initial_supplies(s, 0, 0);
}

static int item_slot(const FcPlayer *p, int id) {
    for (int i = 0; i < FC_INVENTORY_SLOTS; i++)
        if (p->inventory[i].item_id == id) return i;
    return -1;
}

static int episode_analytics_test(void) {
    FcState s;
    FcRewardRuntime runtime;
    FcRewardParams params = fc_reward_default_params();
    FcEpisodeSummary summary;
    reset(&s, 0);
    memset(s.npcs, 0, sizeof(s.npcs));
    s.current_wave = 63;
    int jad = fc_spawn_npc_first_free(&s, NPC_TZTOK_JAD, 10, 10);
    CHECK(jad >= 0);
    s.npcs[jad].current_hp -= 100;
    fc_reward_runtime_begin_episode(&runtime, &s);

    /* Idle, damage, healing, idle: only the two idle ticks count as zero
     * progress. Healing totals use effective HP restored, capped at max HP. */
    fc_reward_compute_breakdown(&s, &params, &runtime);
    s.npcs[jad].current_hp -= 10;
    fc_reward_compute_breakdown(&s, &params, &runtime);
    CHECK(apply_npc_heal(&s, &s.npcs[jad], &s.npcs[jad], 200) == 110);
    fc_reward_compute_breakdown(&s, &params, &runtime);
    clear_per_tick_flags(&s);
    fc_reward_compute_breakdown(&s, &params, &runtime);
    CHECK(runtime.zero_progress_ticks == 2);
    CHECK(runtime.npc_healing_total == 110 && runtime.jad_healing_total == 110);

    /* No prayer and correct prayer are excluded from wrong-prayer hits,
     * including when an attack rolls zero damage. */
    const int prayers[] = {PRAYER_NONE, PRAYER_PROTECT_MAGIC, PRAYER_PROTECT_RANGE};
    for (int i = 0; i < 3; i++) {
        CHECK(fc_queue_pending_hit(s.player.pending_hits, &s.player.num_pending_hits,
            FC_MAX_PENDING_HITS, 0, 1, ATTACK_RANGED, jad, 0));
        s.player.pending_hits[i].prayer_snapshot = prayers[i];
    }
    fc_resolve_player_pending_hits(&s);
    CHECK(s.ep_wrong_prayer_hits == 1);

    const int actions[FC_NUM_ACTION_HEADS] = {0};
    for (int prayer = PRAYER_PROTECT_MELEE; prayer <= PRAYER_PROTECT_MAGIC; prayer++) {
        s.player.prayer = prayer;
        fc_step(&s, actions);
    }
    complete_fight_caves(&s);
    fc_episode_summary_build(&s, &runtime, 3, &summary);
    CHECK(summary.wave_reached == 63 && summary.reached_wave_63 == 1);
    CHECK(summary.jad_kill_rate == 1 && summary.wrong_prayer_hits == 1);
    CHECK(summary.prayer_uptime_melee == 1.0f / 3.0f);
    CHECK(summary.prayer_uptime_range == 1.0f / 3.0f);
    CHECK(summary.prayer_uptime_magic == 1.0f / 3.0f);

    fc_reset(&s, 101);
    fc_reward_runtime_begin_episode(&runtime, &s);
    fc_episode_summary_build(&s, &runtime, 0, &summary);
    CHECK(summary.wave_reached == 1 && summary.episode_length == 0);
    CHECK(summary.zero_progress_ticks == 0 && summary.wrong_prayer_hits == 0);
    CHECK(summary.reached_wave_63 == 0 && summary.jad_kill_rate == 0);
    CHECK(summary.npc_healing_total == 0 && summary.jad_healing_total == 0);
    CHECK(summary.prayer_uptime_melee == 0 && summary.prayer_uptime_range == 0 &&
          summary.prayer_uptime_magic == 0);
    /* Multiple invalid heads still incur one invalid-action penalty. */
    const int invalid_actions[FC_NUM_ACTION_HEADS] = {999, 999, 999};
    fc_step(&s, invalid_actions);
    CHECK(s.invalid_action_this_tick == 1);
    fc_destroy(&s);
    puts("episode_analytics_test: progress, healing, prayers, Jad and reset passed");
    return 0;
}

static int loadout_totals(void) {
    /* Pre-cleanup preset totals, independent of the item definitions. */
    static const FcPlayer expected[FC_NUM_LOADOUTS] = {
        { /* Preset 0 */
            .ranged_attack_bonus = 153, .ranged_strength_bonus = 100, .defence_stab = 97,
            .defence_slash = 84, .defence_crush = 110, .defence_magic = 91,
            .defence_ranged = 90, .prayer_bonus = 0, .weapon_kind = 0,
            .weapon_speed = 5, .weapon_range = 7, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
        { /* Preset 1 */
            .ranged_attack_bonus = 215, .ranged_strength_bonus = 99, .defence_stab = 116,
            .defence_slash = 106, .defence_crush = 129, .defence_magic = 150,
            .defence_ranged = 121, .prayer_bonus = 6, .weapon_kind = 1,
            .weapon_speed = 5, .weapon_range = 10, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
        { /* Preset 2 */
            .ranged_attack_bonus = 166, .ranged_strength_bonus = 100, .defence_stab = 48,
            .defence_slash = 49, .defence_crush = 62, .defence_magic = 42,
            .defence_ranged = 46, .prayer_bonus = 8, .weapon_kind = 0,
            .weapon_speed = 5, .weapon_range = 7, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
        { /* Preset 3 */
            .ranged_attack_bonus = 166, .ranged_strength_bonus = 100, .defence_stab = 48,
            .defence_slash = 49, .defence_crush = 62, .defence_magic = 42,
            .defence_ranged = 46, .prayer_bonus = 8, .weapon_kind = 0,
            .weapon_speed = 5, .weapon_range = 7, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
        { /* Preset 4 */
            .ranged_attack_bonus = 141, .ranged_strength_bonus = 49, .defence_stab = 48,
            .defence_slash = 49, .defence_crush = 62, .defence_magic = 42,
            .defence_ranged = 46, .prayer_bonus = 3, .weapon_kind = 0,
            .weapon_speed = 3, .weapon_range = 7, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
        { /* Preset 5 */
            .ranged_attack_bonus = 101, .ranged_strength_bonus = 42, .defence_stab = 45,
            .defence_slash = 46, .defence_crush = 59, .defence_magic = 39,
            .defence_ranged = 43, .prayer_bonus = 2, .weapon_kind = 0,
            .weapon_speed = 2, .weapon_range = 5, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
        { /* Preset 6 */
            .ranged_attack_bonus = 220, .ranged_strength_bonus = 129, .defence_stab = 112,
            .defence_slash = 100, .defence_crush = 123, .defence_magic = 139,
            .defence_ranged = 117, .prayer_bonus = 11, .weapon_kind = 0,
            .weapon_speed = 5, .weapon_range = 8, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
        { /* Preset 7 */
            .ranged_attack_bonus = 233, .ranged_strength_bonus = 113, .defence_stab = 102,
            .defence_slash = 85, .defence_crush = 110, .defence_magic = 107,
            .defence_ranged = 143, .prayer_bonus = 9, .weapon_kind = 2,
            .weapon_speed = 4, .weapon_range = 10, .weapon_uses_ammo = 0,
            .crystal_piece_mask = 7, .ammo_count = 0,
        },
        { /* Preset 8 */
            .ranged_attack_bonus = 205, .ranged_strength_bonus = 97, .defence_stab = 116,
            .defence_slash = 106, .defence_crush = 129, .defence_magic = 150,
            .defence_ranged = 121, .prayer_bonus = 6, .weapon_kind = 1,
            .weapon_speed = 5, .weapon_range = 10, .weapon_uses_ammo = 1,
            .crystal_piece_mask = 0, .ammo_count = 50000,
        },
    };
    for (int i = 0; i < FC_NUM_LOADOUTS; i++) {
        FcPlayer p = {0};
        fc_items_init(&p, &FC_LOADOUTS[i]);
        CHECK(p.ranged_attack_bonus == expected[i].ranged_attack_bonus);
        CHECK(p.ranged_strength_bonus == expected[i].ranged_strength_bonus);
        CHECK(p.defence_stab == expected[i].defence_stab);
        CHECK(p.defence_slash == expected[i].defence_slash);
        CHECK(p.defence_crush == expected[i].defence_crush);
        CHECK(p.defence_magic == expected[i].defence_magic);
        CHECK(p.defence_ranged == expected[i].defence_ranged);
        CHECK(p.prayer_bonus == expected[i].prayer_bonus);
        CHECK(p.weapon_kind == expected[i].weapon_kind);
        CHECK(p.weapon_speed == expected[i].weapon_speed);
        CHECK(p.weapon_range == expected[i].weapon_range);
        CHECK(p.weapon_uses_ammo == expected[i].weapon_uses_ammo);
        CHECK(p.crystal_piece_mask == expected[i].crystal_piece_mask);
        CHECK(p.ammo_count == expected[i].ammo_count);
    }
    return 0;
}

static int transactions(void) {
    FcState s;
    reset(&s, 0);
    uint32_t before = fc_state_hash(&s);
    CHECK(fc_unequip_item(&s, FC_EQUIP_SLOT_HEAD) == FC_ITEM_NO_SPACE);
    CHECK(fc_state_hash(&s) == before);
    CHECK(fc_equip_item(&s, -1) == FC_ITEM_INVALID);
    CHECK(fc_equip_item(&s, 28) == FC_ITEM_INVALID);
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_INVALID); /* food is not equipment */
    CHECK(fc_unequip_item(&s, 6) == FC_ITEM_INVALID); /* client-only arms */
    CHECK(fc_state_hash(&s) == before);
    reset(&s, 1);
    s.player.current_hp = 400;
    s.player.current_prayer = 370;
    s.player.attack_timer = 4;
    s.player.prayer_drain_counter = 21;
    s.player.attack_target_idx = 0;
    CHECK(fc_unequip_item(&s, FC_EQUIP_SLOT_HEAD) == FC_ITEM_OK);
    CHECK(s.player.inventory[0].item_id == 27235);
    CHECK(s.player.ranged_attack_bonus == 203 && s.player.ranged_strength_bonus == 97);
    CHECK(s.player.defence_stab == 108 && s.player.prayer_bonus == 5);
    CHECK(s.player.attack_target_idx == 0);
    uint32_t rng = s.rng_state;
    s.player.defence_level = 79;
    before = fc_state_hash(&s);
    CHECK(fc_equip_item(&s, 0) == FC_ITEM_REQUIREMENTS);
    CHECK(fc_state_hash(&s) == before);
    s.player.defence_level = 99;
    CHECK(fc_equip_item(&s, 0) == FC_ITEM_OK);
    CHECK(s.player.attack_target_idx == -1);
    CHECK(s.player.ranged_attack_bonus == 215 && s.player.ranged_strength_bonus == 99);
    CHECK(s.player.current_hp == 400 && s.player.current_prayer == 370);
    CHECK(s.player.attack_timer == 4 && s.player.prayer_drain_counter == 21);
    CHECK(s.rng_state == rng && s.tick == 0);
    for (int i = 0; i < FC_EQUIPMENT_SLOTS; i++)
        if (s.player.equipment[i].item_id) CHECK(fc_unequip_item(&s, i) == FC_ITEM_OK);
    CHECK(s.player.weapon_kind == FC_WEAPON_UNARMED);
    CHECK(s.player.ranged_attack_bonus == 0 && s.player.defence_stab == 0);
    CHECK(s.player.prayer_bonus == 0 && s.player.ammo_count == 0);
    for (int i = 0; i < FC_INVENTORY_SLOTS; i++)
        if (s.player.inventory[i].item_id) CHECK(fc_equip_item(&s, i) == FC_ITEM_OK);
    CHECK(s.player.ranged_attack_bonus == 215 && s.player.ranged_strength_bonus == 99);
    CHECK(s.player.equipment[13].quantity == 50000);
    s.terminal = TERMINAL_PLAYER_DEATH;
    before = fc_state_hash(&s);
    CHECK(fc_unequip_item(&s, 0) == FC_ITEM_BUSY);
    CHECK(fc_state_hash(&s) == before);
    return 0;
}

static int two_handed_and_stacks(void) {
    FcState s;
    reset(&s, 0);
    /* Full inventory: a one-for-one shield swap may use the source slot for
     * the two-handed bow when the old shield slot is empty. */
    s.player.inventory[8] = (FcItemStack){12610, 1, 0};
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK);
    CHECK(s.player.inventory[8].item_id == 20997);
    CHECK(s.player.equipment[FC_EQUIP_SLOT_SHIELD].item_id == 12610);
    CHECK(s.player.weapon_kind == FC_WEAPON_UNARMED);
    /* With a crossbow AND shield worn, bow needs a second inventory slot. */
    s.player.inventory[9] = (FcItemStack){9185, 1, 0};
    CHECK(fc_equip_item(&s, 9) == FC_ITEM_OK);
    s.player.inventory[9] = (FcItemStack){385, 1, 0};
    uint32_t before = fc_state_hash(&s);
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_NO_SPACE);
    CHECK(fc_state_hash(&s) == before);
    s.player.inventory[14] = (FcItemStack){0};
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK);
    CHECK(s.player.inventory[8].item_id == 9185);
    CHECK(s.player.inventory[14].item_id == 12610);
    CHECK(s.player.equipment[FC_EQUIP_SLOT_SHIELD].item_id == 0);
    reset(&s, 0);
    s.player.inventory[8] = (FcItemStack){11212, 7, 0};
    CHECK(fc_unequip_item(&s, 13) == FC_ITEM_OK); /* merge despite full inventory */
    CHECK(s.player.inventory[8].quantity == 50007);
    CHECK(s.player.ammo_count == 0);
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK);
    CHECK(s.player.ammo_count == 50007);
    s.player.inventory[8] = (FcItemStack){11212, INT_MAX - 50006, 0};
    before = fc_state_hash(&s);
    CHECK(fc_unequip_item(&s, 13) == FC_ITEM_NO_SPACE); /* whole-stack overflow */
    CHECK(fc_state_hash(&s) == before);
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK); /* equip only the portion that fits */
    CHECK(s.player.equipment[13].quantity == INT_MAX);
    CHECK(s.player.inventory[8].quantity == 1);
    before = fc_state_hash(&s);
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_NO_SPACE);
    CHECK(fc_state_hash(&s) == before);
    s.player.inventory[8] = (FcItemStack){9143, 100, 0};
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK);
    CHECK(s.player.ammo_count == 0); /* bolts do not work in a bow */
    CHECK(s.player.inventory[8].item_id == 11212);
    s.player.inventory[9] = (FcItemStack){12788, 1, 0};
    CHECK(fc_equip_item(&s, 9) == FC_ITEM_OK);
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK); /* dragon arrows in quiver */
    CHECK(s.player.ammo_count == 0); /* MSB cannot fire dragon arrows */
    s.player.inventory[8] = (FcItemStack){892, 10, 0};
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK);
    CHECK(s.player.ammo_count == 10);
    s.player.inventory[9] = (FcItemStack){9185, 1, 0};
    CHECK(fc_equip_item(&s, 9) == FC_ITEM_OK);
    s.player.inventory[8] = (FcItemStack){21946, 10, 0};
    CHECK(fc_equip_item(&s, 8) == FC_ITEM_OK);
    CHECK(s.player.ammo_count == 0); /* rune crossbow cannot fire dragon bolts */
    s.player.inventory[9] = (FcItemStack){11785, 1, 0};
    CHECK(fc_equip_item(&s, 9) == FC_ITEM_OK);
    CHECK(s.player.ammo_count == 10);
    return 0;
}

static int supplies_and_loaded_weapon(void) {
    FcState s;
    reset(&s, 1);
    fc_set_initial_supplies(&s, 2, 5);
    CHECK(s.player.inventory[0].item_id == 2434 && s.player.inventory[1].item_id == 143);
    CHECK(s.player.inventory[2].item_id == 385 && s.player.inventory[3].item_id == 385);
    CHECK(fc_inventory_swap(&s, 1, 27) == FC_ITEM_OK);
    CHECK(fc_select_consumable(&s, 27) == FC_ITEM_OK);
    CHECK(fc_select_consumable(&s, 3) == FC_ITEM_OK);
    s.player.current_hp = 400;
    s.player.current_prayer = 100;
    int actions[FC_NUM_ACTION_HEADS] = {0};
    actions[3] = FC_EAT_SHARK;
    actions[4] = FC_DRINK_PRAYER_POT;
    fc_step(&s, actions);
    CHECK(s.player.inventory[27].item_id == 229);
    CHECK(s.player.inventory[0].item_id == 2434);
    CHECK(s.player.inventory[3].item_id == 0 && s.player.inventory[2].item_id == 385);
    CHECK(s.player.sharks_remaining == 1 && s.player.prayer_doses_remaining == 4);
    CHECK(s.player.selected_food_slot == -1 && s.player.selected_potion_slot == -1);
    reset(&s, 1);
    fc_items_init(&s.player, &FC_LOADOUTS[FC_LOADOUT_BLOWPIPE_PURE]);
    fc_set_initial_supplies(&s, 0, 0);
    fc_items_spend_ammo(&s.player);
    CHECK(s.player.equipment[3].charges == 49999);
    CHECK(s.player.equipment[13].item_id == 0);
    CHECK(fc_unequip_item(&s, 3) == FC_ITEM_OK);
    CHECK(s.player.inventory[0].charges == 49999);
    CHECK(fc_equip_item(&s, 0) == FC_ITEM_OK);
    CHECK(s.player.ammo_count == 49999);
    return 0;
}

static int combat(void) {
    FcState s;
    reset(&s, 1);
    memset(s.npcs, 0, sizeof(s.npcs));
    memset(s.walkable, 1, sizeof(s.walkable));
    memset(s.movement_flags, 0, sizeof(s.movement_flags));
    memset(s.los_flags, 0, sizeof(s.los_flags));
    s.player.x = 20; s.player.y = 20;
    fc_npc_spawn(&s.npcs[0], NPC_YT_MEJKOT, 20, 25, 1);
    s.npcs_remaining = 1;
    s.npcs[0].attack_timer = 100;
    s.player.attack_target_idx = 0;
    int actions[FC_NUM_ACTION_HEADS] = {0};
    fc_step(&s, actions);
    CHECK(s.render_events.player_attack_fired);
    CHECK(s.player.ammo_count == 49999 && s.player.equipment[13].quantity == 49999);
    FcPendingHit hit = s.npcs[0].pending_hits[0];
    int cooldown = s.player.attack_timer;
    CHECK(fc_unequip_item(&s, 3) == FC_ITEM_OK);
    CHECK(s.player.attack_target_idx == 0 && s.player.attack_timer == cooldown);
    CHECK(memcmp(&hit, &s.npcs[0].pending_hits[0], sizeof(hit)) == 0);
    CHECK(s.player.weapon_range == 1 && s.player.weapon_speed == 4 && !s.player.weapon_uses_ammo);
    s.player.attack_timer = 0;
    s.npcs[0].x = 21; s.npcs[0].y = 21;
    fc_step(&s, actions);
    CHECK(!s.render_events.player_attack_fired); /* diagonal is not melee contact */
    s.npcs[0].x = 21; s.npcs[0].y = 20;
    fc_step(&s, actions);
    CHECK(s.render_events.player_attack_fired);
    CHECK(s.render_events.player_attack_hit_delay_ticks == 1);
    CHECK(s.player.attack_timer == 3 && s.player.ammo_count == 0); /* end-of-tick decrement */
    int melee_hit = 0;
    for (int i = 0; i < s.render_events.hit_count; i++)
        if (s.render_events.hits[i].target_entity_type == ENTITY_NPC &&
            s.render_events.hits[i].attack_style == ATTACK_MELEE) melee_hit = 1;
    CHECK(melee_hit); /* delay-one queues resolve during this tick's hit phase */
    CHECK(fc_equip_item(&s, item_slot(&s.player, 20997)) == FC_ITEM_OK);
    CHECK(s.player.attack_timer == 3 && s.player.attack_target_idx == -1);
    int route_x[64], route_y[64];
    int len = fc_pathfind_attack_position(20, 20, 25, 25, 5, FC_ROUTE_MELEE_RANGE,
        s.walkable, s.movement_flags, s.los_flags, route_x, route_y, 64);
    CHECK(len > 0);
    CHECK(fc_npc_can_melee_player(route_x[len-1], route_y[len-1], 25, 25, 5,
                                s.walkable, s.movement_flags));
    return 0;
}

static int equipment_test(void) {
    if (loadout_totals() || transactions() || two_handed_and_stacks() ||
        supplies_and_loaded_weapon() || combat()) return 1;
    puts("equipment: preset preservation, atomic transfers, stacks, requirements, supplies and combat passed");
    return 0;
}

#undef CHECK

static FightCaves* test_env(void) {
    FightCaves* env = calloc(1, sizeof(*env));
    assert(env);
    env->num_agents = 1;
    env->rng = 73;
    env->observations = calloc(FC_PUFFER_OBS_SIZE, sizeof(float));
    env->actions = calloc(FC_PUFFER_NUM_ATNS, sizeof(float));
    env->rewards = calloc(1, sizeof(float));
    env->terminals = calloc(1, sizeof(float));
    env->action_mask = calloc(FC_PUFFER_MASK_SIZE, 1);
    env->reward_params = fc_reward_default_params();
    fc_init(&env->state);
    c_reset(env);
    return env;
}

static void test_free(FightCaves* env) {
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->action_mask);
    free(env);
}

static int adapter_snapshot_test(void) {
    FightCaves* env = test_env();
    FightCaves* rendered = test_env();
    int episodes = 0;
    /* Capture render snapshots without opening a window or loading graphics. */
    rendered->viewer = calloc(1, sizeof(*rendered->viewer));
    assert(rendered->viewer);
    for (int tick = 0; tick < 2048; tick++) {
        env->actions[0] = tick % 17;
        env->actions[1] = (tick / 3) % 9;
        env->actions[2] = (tick / 7) % 8;
        memcpy(rendered->actions, env->actions, sizeof(float) * FC_PUFFER_NUM_ATNS);
        c_step(env);
        c_step(rendered);
        assert(fc_state_hash(&env->state) == fc_state_hash(&rendered->state));
        assert(memcmp(env->observations, rendered->observations,
                      sizeof(float) * FC_PUFFER_OBS_SIZE) == 0);
        assert(memcmp(env->action_mask, rendered->action_mask,
                      FC_PUFFER_MASK_SIZE) == 0);
        assert(env->rewards[0] == rendered->rewards[0]);
        assert(env->terminals[0] == rendered->terminals[0]);
        assert(rendered->viewer->pending_frame);
        if (env->terminals[0]) {
            episodes++;
            assert(rendered->state.tick == 0);
            assert(fc_is_terminal(&rendered->viewer->pending_state));
        } else {
            assert(fc_state_hash(&rendered->viewer->pending_state) ==
                   fc_state_hash(&env->state));
        }
    }
    assert(episodes > 0);
    c_reset(env);
    c_reset(rendered);
    assert(fc_state_hash(&env->state) == fc_state_hash(&rendered->state));
    assert(fc_state_hash(&rendered->viewer->reset_state) == fc_state_hash(&env->state));
    printf("adapter_snapshot_test: passed (2048 steps, %d episodes)\n", episodes);
    free(rendered->viewer);
    rendered->viewer = NULL;
    test_free(rendered);
    test_free(env);
    return 0;
}

int main(void) {
    wave_rotation_test();
    return core_contract_test() || equipment_test() ||
           episode_analytics_test() || adapter_snapshot_test();
}
