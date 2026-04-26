/**
 * @file test_human_commands.c
 * @brief tests for human command queue staging used by encounter human mode
 *
 * BUILD:
 *   cc -std=c11 -O0 -g -I. -o /tmp/test_human_commands \
 *       ocean/osrs/tests/test_human_commands.c -lm
 *   /tmp/test_human_commands
 */

#include <stdio.h>
#include <string.h>

#include "ocean/osrs/osrs_types.h"
#include "ocean/osrs/osrs_items.h"
#include "ocean/osrs/osrs_human_input_types.h"

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

static void test_command_queue_preserves_order(void) {
    printf("--- human command queue preserves order ---\n");

    HumanInput input;
    human_input_init(&input);

    human_input_queue_walk(&input, 10, 20);
    human_input_queue_attack_npc(&input, 7);
    human_input_queue_drink(&input, POTION_BASTION, 3);

    ASSERT_INT_EQ("queue count", input.commands.count, 3);
    ASSERT_INT_EQ("first kind", input.commands.items[0].kind, HUMAN_COMMAND_WALK);
    ASSERT_INT_EQ("first x", input.commands.items[0].world_x, 10);
    ASSERT_INT_EQ("second kind", input.commands.items[1].kind, HUMAN_COMMAND_ATTACK_NPC);
    ASSERT_INT_EQ("second npc slot", input.commands.items[1].npc_slot, 7);
    ASSERT_INT_EQ("third kind", input.commands.items[2].kind, HUMAN_COMMAND_DRINK);
    ASSERT_INT_EQ("third potion", input.commands.items[2].potion, POTION_BASTION);

    human_input_destroy(&input);
}

static void test_command_queue_grows_without_silent_cap(void) {
    printf("--- human command queue grows without silent cap ---\n");

    HumanInput input;
    human_input_init(&input);

    for (int i = 0; i < 40; i++) {
        human_input_queue_equip_inventory_item(&input, i, ITEM_TOXIC_BLOWPIPE, GEAR_SLOT_WEAPON);
    }

    ASSERT_INT_EQ("queue count after growth", input.commands.count, 40);
    ASSERT_INT_EQ("first slot", input.commands.items[0].inventory_slot, 0);
    ASSERT_INT_EQ("last slot", input.commands.items[39].inventory_slot, 39);

    human_input_destroy(&input);
}

static void test_command_queue_clear_drains_commands(void) {
    printf("--- human command queue clear drains commands ---\n");

    HumanInput input;
    human_input_init(&input);

    human_input_queue_walk(&input, 1, 2);
    human_input_queue_spec_toggle(&input);
    human_input_clear_commands(&input);

    ASSERT_INT_EQ("queue count after clear", input.commands.count, 0);
    ASSERT_INT_EQ("queue capacity retained", input.commands.capacity > 0, 1);

    human_input_destroy(&input);
}

int main(void) {
    test_command_queue_preserves_order();
    test_command_queue_grows_without_silent_cap();
    test_command_queue_clear_drains_commands();

    printf("\n%d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d failed)\n", tests_failed);
        return 1;
    }
    printf("\n");
    return 0;
}
