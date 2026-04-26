/**
 * @file test_zulrah_human_commands.c
 * @brief tests for Zulrah encounter human command execution.
 *
 * BUILD:
 *   cc -std=c11 -O0 -g -I. -o /tmp/test_zulrah_human_commands \
 *       ocean/osrs/tests/test_zulrah_human_commands.c -lm
 *   /tmp/test_zulrah_human_commands
 */

#include <stdio.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_zulrah.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_INT_EQ(label, actual, expected) do { \
    tests_run++; \
    int _actual = (actual); \
    int _expected = (expected); \
    if (_actual == _expected) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s: got %d, expected %d\n", \
            (label), _actual, _expected); \
    } \
} while (0)

static void test_zulrah_human_equip_is_item_by_item(void) {
    printf("--- zulrah human equip is item by item ---\n");

    EncounterState* raw = zul_create();
    ZulrahState* state = (ZulrahState*)raw;
    zul_reset(raw, 123);

    HumanInput input;
    human_input_init(&input);
    input.enabled = 1;

    uint8_t old_head = state->player.equipped[GEAR_SLOT_HEAD];
    human_input_queue_equip_inventory_item(
        &input, 0, ITEM_TOXIC_BLOWPIPE, GEAR_SLOT_WEAPON);

    zul_step_human_commands(raw, &input);

    ASSERT_INT_EQ("weapon changed to clicked item",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_TOXIC_BLOWPIPE);
    ASSERT_INT_EQ("head slot did not snap to range preset",
        state->player.equipped[GEAR_SLOT_HEAD], old_head);
    ASSERT_INT_EQ("queued command drained", input.commands.count, 0);
    ASSERT_INT_EQ("human gear style follows weapon", state->player_gear, ZUL_GEAR_RANGE);

    human_input_destroy(&input);
    zul_destroy(raw);
}

static void test_zulrah_human_attack_uses_queued_weapon_style(void) {
    printf("--- zulrah human attack uses queued weapon style ---\n");

    EncounterState* raw = zul_create();
    ZulrahState* state = (ZulrahState*)raw;
    zul_reset(raw, 123);

    HumanInput input;
    human_input_init(&input);
    int actions[ZUL_NUM_ACTION_HEADS];

    human_input_queue_equip_inventory_item(
        &input, 0, ITEM_TOXIC_BLOWPIPE, GEAR_SLOT_WEAPON);
    human_input_queue_attack_npc(&input, -1);
    zul_translate_human_commands(&input, actions, state);
    ASSERT_INT_EQ("queued ranged weapon selects range attack",
        actions[ZUL_HEAD_ATTACK], ZUL_ATK_RANGE);

    human_input_clear_pending(&input);
    human_input_queue_equip_inventory_item(
        &input, 0, ITEM_TRIDENT_OF_SWAMP, GEAR_SLOT_WEAPON);
    human_input_queue_attack_npc(&input, -1);
    zul_translate_human_commands(&input, actions, state);
    ASSERT_INT_EQ("queued magic weapon selects mage attack",
        actions[ZUL_HEAD_ATTACK], ZUL_ATK_MAGE);

    human_input_destroy(&input);
    zul_destroy(raw);
}

int main(void) {
    test_zulrah_human_equip_is_item_by_item();
    test_zulrah_human_attack_uses_queued_weapon_style();

    printf("\n%d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d failed)\n", tests_failed);
        return 1;
    }
    printf("\n");
    return 0;
}
