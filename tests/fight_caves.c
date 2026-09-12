#ifdef NDEBUG
#undef NDEBUG
#endif
#include "simulation.h"
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

/* The same equipment transaction scenarios used to validate v38. */
#include <limits.h>
#include <stdio.h>
#include <string.h>

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

#ifdef FC_VIEWER_TEST
#include "render.h"
#include "ui.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static int context_menu_test(void) {
    FcMenuLayout menu = fc_menu_layout(400, 200, 800, 600, 150, 4);
    assert(menu.x == 319 && menu.y == 200 && menu.width == 162);
    assert(menu.height == FC_MENU_HEADER_HEIGHT + 4 * FC_MENU_ROW_HEIGHT + 4);
    assert(fc_menu_action_at(menu, 4, 330, 200) == -1); /* title is not a choice */
    for (int row = 0; row < 4; row++) {
        int y = menu.y + FC_MENU_HEADER_HEIGHT + row * FC_MENU_ROW_HEIGHT;
        assert(fc_menu_action_at(menu, 4, 330, y) == row);
        assert(fc_menu_action_at(menu, 4, 330, y + FC_MENU_ROW_HEIGHT - 1) == row);
    }
    assert(fc_menu_action_at(menu, 4, 330, menu.y + menu.height - 1) == -1);
    assert(fc_menu_action_at(menu, 4, menu.x - 1, 230) == -1);
    assert(fc_menu_action_at(menu, 4, menu.x + menu.width, 230) == -1);
    assert(fc_menu_contains(menu, menu.x - 9, 230, FC_MENU_DISMISS_MARGIN));
    assert(!fc_menu_contains(menu, menu.x - 11, 230, FC_MENU_DISMISS_MARGIN));
    menu = fc_menu_layout(799, 599, 800, 600, 220, 3);
    assert(menu.x + menu.width == 800 && menu.y + menu.height == 600);
    assert(fc_menu_action_at(menu, 3, 790, menu.y + FC_MENU_HEADER_HEIGHT) == 0);
    menu = fc_menu_layout(0, 0, 800, 600, 150, 4);
    assert(menu.x == 0 && menu.y == 0);
    menu = fc_menu_layout(0, 0, 80, 200, 150, 4);
    assert(menu.width == 80 && menu.x == 0);
    const int levels[] = {0, 22, 45, 22, 90, 180, 360, 702, 108};
    for (int type = 1; type <= 8; type++)
        assert(fc_menu_npc_info(type).level == levels[type]);
    assert(strcmp(fc_menu_npc_info(7).name, "TzTok-Jad") == 0);
    assert(strcmp(fc_menu_npc_info(2).name, fc_menu_npc_info(3).name) == 0);
    assert(fc_menu_npc_info(0).level == 0 && fc_menu_npc_info(-1).level == 0);
    assert(fc_menu_npc_info(9).level == 0);
    const uint32_t colors[] = {
        0xff0000, 0xff3000, 0xff3000, 0xff3000, 0xff7000, 0xff7000,
        0xff7000, 0xffb000, 0xffb000, 0xffb000, 0xffff00,
        0xc0ff00, 0xc0ff00, 0xc0ff00, 0x80ff00, 0x80ff00, 0x80ff00,
        0x40ff00, 0x40ff00, 0x40ff00, 0x00ff00
    };
    for (int difference = -10; difference <= 10; difference++)
        assert(fc_menu_level_color(100 + difference, 100) == colors[difference + 10]);
    assert(fc_menu_level_color(126, 702) == 0xff0000);
    assert(fc_menu_level_color(126, 108) == 0x00ff00);
    puts("context menu: anchoring, screen edges, rows, title and dismissal margin passed");
    return 0;
}


#include <assert.h>
#include <stdio.h>
#include <string.h>

static void make_open_state(FcState* state) {
    memset(state, 0, sizeof(*state));
    for (int x = 0; x < FC_ARENA_WIDTH; x++) {
        for (int y = 0; y < FC_ARENA_HEIGHT; y++) {
            state->walkable[x][y] = 1;
        }
    }
    state->player.x = 1;
    state->player.y = 1;
}

static int click_feedback_test(void) {
    FcState state;
    FcClickFeedback feedback;
    const int* route_x = NULL;
    const int* route_y = NULL;
    int start = -1;
    int len = -1;

    make_open_state(&state);
    FcState unchanged = state;
    fc_click_feedback_reset(&feedback);
    fc_click_feedback_select_move(&feedback, &state, 5, 3, 120.0f, 80.0f);

    assert(feedback.destination_active);
    assert(feedback.destination_x == 5 && feedback.destination_y == 3);
    assert(feedback.preview_pending);
    assert(feedback.preview_route_len > 0);
    assert(feedback.preview_route_x[feedback.preview_route_len - 1] == 5);
    assert(feedback.preview_route_y[feedback.preview_route_len - 1] == 3);
    assert(feedback.cross_kind == FC_CLICK_CROSS_MOVE);
    assert(fc_click_feedback_cross_frame(&feedback) == 0);
    assert(fc_click_feedback_route(&feedback, &state, &route_x, &route_y,
                                   &start, &len));
    assert(route_x == feedback.preview_route_x);
    assert(route_y == feedback.preview_route_y);
    assert(start == 0 && len == feedback.preview_route_len);

    fc_click_feedback_update(&feedback, 0.11f);
    assert(fc_click_feedback_cross_frame(&feedback) == 1);
    fc_click_feedback_update(&feedback, 0.30f);
    assert(feedback.cross_kind == FC_CLICK_CROSS_NONE);

    state.player.route_x[0] = 2;
    state.player.route_y[0] = 2;
    state.player.route_x[1] = 3;
    state.player.route_y[1] = 3;
    state.player.route_len = 2;
    state.player.route_idx = 1;
    fc_click_feedback_accept_move_tick(&feedback, &state);
    assert(!feedback.preview_pending);
    assert(feedback.destination_active);
    assert(fc_click_feedback_route(&feedback, &state, &route_x, &route_y,
                                   &start, &len));
    assert(route_x == state.player.route_x && route_y == state.player.route_y);
    assert(start == 1 && len == 2);

    state.player.route_idx = state.player.route_len;
    fc_click_feedback_sync(&feedback, &state);
    assert(!feedback.destination_active);

    fc_click_feedback_select_move(&feedback, &state, 7, 7, 10.0f, 20.0f);
    fc_click_feedback_select_interaction(&feedback, 30.0f, 40.0f);
    assert(!feedback.destination_active);
    assert(!feedback.preview_pending);
    assert(feedback.preview_route_len == 0);
    assert(feedback.cross_kind == FC_CLICK_CROSS_INTERACTION);
    assert(feedback.cross_screen_x == 30.0f && feedback.cross_screen_y == 40.0f);
    assert(fc_click_feedback_cross_frame(&feedback) == 0);
    fc_click_feedback_update(&feedback, 0.099f);
    assert(fc_click_feedback_cross_frame(&feedback) == 0);
    fc_click_feedback_update(&feedback, 0.002f);
    assert(fc_click_feedback_cross_frame(&feedback) == 1);
    /* Test only the presentation operation against a complete state snapshot. */
    FcState before_interaction = state;
    fc_click_feedback_select_interaction(&feedback, 70.0f, 80.0f);
    assert(memcmp(&state, &before_interaction, sizeof(state)) == 0);
    fc_click_feedback_select_move(&feedback, &unchanged, 3, 5, 10, 20);
    FcState fresh;
    make_open_state(&fresh);
    assert(memcmp(&fresh, &unchanged, sizeof(fresh)) == 0);

    puts("click feedback tests passed");
    return 0;
}

#include "raymath.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int model_picking_test(void) {
    /* Tall two-triangle actor. The uploaded mesh deliberately belongs to a
     * different actor: picking must use this instance's pose or the rest mesh. */
    float rest[] = {-0.6f,0,0, 0.6f,0,0, 0.6f,4,0, -0.6f,0,0, 0.6f,4,0, -0.6f,4,0};
    float other_actor[18] = {100,100,100};
    uint16_t faces[] = {0,1,2,0,2,3};
    int16_t pose[] = {-77,0,0, 77,0,0, 77,-512,0, -77,-512,0};
    float original_rest[18]; int16_t original_pose[12];
    memcpy(original_rest, rest, sizeof(rest));
    memcpy(original_pose, pose, sizeof(pose));
    Mesh uploaded = {.vertices=other_actor};
    ModelEntry entry = {.loaded=1, .rest_verts=rest, .face_count=2,
                       .face_indices=faces, .base_vert_count=4};
    entry.model.transform = MatrixIdentity();
    entry.model.meshes = &uploaded;
    Camera3D camera = {.position={0,4,10}, .target={0,2,0}, .up={0,1,0},
                       .fovy=50, .projection=CAMERA_PERSPECTIVE};
    Vector3 origin = {0};
    Vector2 head = GetWorldToScreenEx((Vector3){0,3.8f,0}, camera, 800, 600);
    assert(models_pick_depth(&entry, NULL, origin, 0, camera, head, 800, 600) > 0);
    assert(models_pick_depth(&entry, pose, origin, 0, camera, head, 800, 600) > 0);
    /* A head click projects onto ground far behind the actor: the old tile
     * halo would miss, even though the pointer is inside the visible model. */
    Vector3 ray = Vector3Subtract((Vector3){0,3.8f,0}, camera.position);
    float ground_z = camera.position.z + ray.z * (-camera.position.y / ray.y);
    assert(fabsf(ground_z) > 2);

    Vector2 top = GetWorldToScreenEx((Vector3){0,4,0}, camera, 800, 600);
    assert(models_pick_depth(&entry, pose, origin, 0, camera,
        (Vector2){top.x, top.y - 4}, 800, 600) > 0);
    assert(models_pick_depth(&entry, pose, origin, 0, camera,
        (Vector2){top.x, top.y - 6}, 800, 600) < 0);
    assert(models_pick_depth(&entry, pose, origin, 0, camera, (Vector2){10,10}, 800,600) < 0);

    /* Moving and turned models are picked where drawn, not at their old tile. */
    Vector3 moved = {4,0,0};
    Vector2 moved_head = GetWorldToScreenEx((Vector3){4,3.8f,0}, camera, 800,600);
    assert(models_pick_depth(&entry, pose, moved, 90, camera, moved_head, 800,600) > 0);
    assert(models_pick_depth(&entry, pose, moved, 90, camera, head, 800,600) < 0);
    /* Two instances of the same model can be in different animation poses. */
    int16_t shifted[12]; memcpy(shifted, pose, sizeof(pose));
    for (int i = 0; i < 4; i++) shifted[i*3] += 512;
    assert(models_pick_depth(&entry, shifted, origin, 0, camera, moved_head, 800,600) > 0);
    assert(models_pick_depth(&entry, shifted, origin, 0, camera, head, 800,600) < 0);
    assert(models_pick_depth(&entry, pose, origin, 0, camera, head, 800,600) > 0);

    Vector2 center = GetWorldToScreenEx((Vector3){0,2,0}, camera, 800,600);
    float front = models_pick_depth(&entry, pose, origin, 0, camera, center, 800,600);
    float back = models_pick_depth(&entry, pose, (Vector3){0,0,-4}, 0, camera, center, 800,600);
    assert(front > 0 && back > front); /* overlap ordering */
    assert(models_pick_depth(&entry, pose, (Vector3){0,0,30}, 0, camera, center, 800,600) < 0);
    assert(models_pick_depth(NULL, pose, origin, 0, camera, center, 800,600) < 0);
    assert(memcmp(rest, original_rest, sizeof(rest)) == 0);
    assert(memcmp(pose, original_pose, sizeof(pose)) == 0);
    assert(other_actor[0] == 100 && other_actor[3] == 0);
    puts("model picking: head, padding, rotation, movement, per-instance pose and depth passed");
    return 0;
}

#include <stdio.h>

/* Explicit graphics test, not part of headless CTest. Captures a contact sheet
 * in the working directory: outfit, no helmet/body/legs/gloves/boots/bow, bare. */
static int equipment_appearance_test(void) {
    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(1200, 640, "Equipment appearance validation");
    if (!IsWindowReady()) return 1;
    FcPlayerAppearance appearance;
    if (!fc_player_appearance_load(&appearance)) return 1;
    AnimCache *cache = anim_cache_load("fc_all.anims");
    if (!cache || !anim_get_sequence(cache, 422)) return 1;
    RenderTexture2D target = LoadRenderTexture(1200, 640);
    BeginTextureMode(target);
    ClearBackground((Color){45, 45, 45, 255});
    const int removed[] = {-1, 0, 4, 7, 9, 10, 3, -2};
    FcState state;
    fc_init(&state);
    for (int variant = 0; variant < 8; variant++) {
        fc_reset(&state, 101);
        fc_set_initial_supplies(&state, 0, 0);
        if (removed[variant] == -2) {
            for (int slot = 0; slot < FC_EQUIPMENT_SLOTS; slot++)
                if (state.player.equipment[slot].item_id &&
                    fc_unequip_item(&state, slot) != FC_ITEM_OK) return 1;
        } else if (removed[variant] >= 0 &&
                   fc_unequip_item(&state, removed[variant]) != FC_ITEM_OK) return 1;
        uint32_t before = fc_state_hash(&state);
        if (fc_player_appearance_sync(&appearance, &state.player, FC_PLAYER_MODEL_BASE) != 1)
            return 1;
        if (fc_player_appearance_sync(&appearance, &state.player, FC_PLAYER_MODEL_BASE) != 0)
            return 1;
        ModelEntry *entry = appearance.model->entries;
        for (int f = 0; f < entry->face_count * 3; f++)
            if (entry->face_indices[f] >= entry->base_vert_count) return 1;
        AnimModelState *pose = NULL;
        uint16_t sequence = 0;
        int frame = 0;
        float timer = 0;
        fc_model_animation_update(entry, cache, &pose, &sequence, &frame,
                                   &timer, 808, 0, 0);
        Camera3D camera = {.position={0, 1.6f, -7}, .target={0, 1.0f, 0},
                           .up={0, 1, 0}, .fovy=4.4f, .projection=CAMERA_ORTHOGRAPHIC};
        BeginScissorMode((variant % 4) * 300, (variant / 4) * 320, 300, 320);
        /* Move the model through one common camera to avoid changing meshes
         * or animation coordinates for the contact-sheet layout. */
        camera.position.x = camera.target.x = ((variant % 4) - 1.5f) * 2.0625f;
        camera.position.y += (variant / 4 ? 1 : -1) * 1.1f;
        camera.target.y += (variant / 4 ? 1 : -1) * 1.1f;
        BeginMode3D(camera);
        DrawModelEx(entry->model, (Vector3){0, 0, 0}, (Vector3){0, 1, 0},
                    180, (Vector3){1, 1, 1}, WHITE);
        EndMode3D();
        EndScissorMode();
        anim_model_state_free(pose);
        if (fc_state_hash(&state) != before) return 1;
        printf("appearance variant %d: %d vertices, %d faces\n",
               variant, entry->base_vert_count, entry->face_count);
    }
    EndTextureMode();
    Image image = LoadImageFromTexture(target.texture);
    ImageFlipVertical(&image);
    int ok = ExportImage(image, "equipment-appearance.png");
    UnloadImage(image);
    UnloadRenderTexture(target);
    anim_cache_free(cache);
    fc_player_appearance_free(&appearance);
    CloseWindow();
    return ok ? 0 : 1;
}

#endif

int main(void) {
    wave_rotation_test();
    if (core_contract_test() || equipment_test() || episode_analytics_test()) return 1;
#ifdef FC_VIEWER_TEST
    if (context_menu_test() || click_feedback_test() || model_picking_test() ||
        equipment_appearance_test()) return 1;
#endif
    return 0;
}
