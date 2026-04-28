/**
 * @file test_interaction.c
 * @brief tests for osrs_interaction.h: entity interaction system + spec toggle
 *
 * BUILD:
 *   cd pufferlib-metal
 *   cc -std=c11 -O0 -g -I. -o test_interaction \
 *       ocean/osrs/tests/test_interaction.c -lm
 *   ./test_interaction
 */

#include <stdio.h>
#include <stdlib.h>

#include "ocean/osrs/osrs_interaction.h"


static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_INT_EQ(label, actual, expected) do { \
    tests_run++; \
    int _a = (actual), _e = (expected); \
    if (_a == _e) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s — got %d, expected %d\n", (label), _a, _e); \
    } \
} while (0)


static void test_init(void) {
    printf("--- init ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    ASSERT_INT_EQ("target_slot is -1", ix.target_slot, -1);
    ASSERT_INT_EQ("not active", osrs_interaction_active(&ix), 0);
}


static void test_set(void) {
    printf("--- set ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    ASSERT_INT_EQ("target_slot is 5", ix.target_slot, 5);
    ASSERT_INT_EQ("active", osrs_interaction_active(&ix), 1);
}


static void test_clear(void) {
    printf("--- clear ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    osrs_interaction_clear(&ix);
    ASSERT_INT_EQ("target_slot is -1", ix.target_slot, -1);
    ASSERT_INT_EQ("not active", osrs_interaction_active(&ix), 0);
}


static void test_interrupt_move(void) {
    printf("--- interrupt: MOVE ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_MOVE);
    ASSERT_INT_EQ("returns 1", result, 1);
    ASSERT_INT_EQ("target cleared", ix.target_slot, -1);
}


static void test_interrupt_eat(void) {
    printf("--- interrupt: EAT ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_EAT);
    ASSERT_INT_EQ("returns 1", result, 1);
    ASSERT_INT_EQ("target cleared", ix.target_slot, -1);
}


static void test_interrupt_drink(void) {
    printf("--- interrupt: DRINK ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_DRINK);
    ASSERT_INT_EQ("returns 1", result, 1);
    ASSERT_INT_EQ("target cleared", ix.target_slot, -1);
}


static void test_interrupt_equip(void) {
    printf("--- interrupt: EQUIP ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_EQUIP);
    ASSERT_INT_EQ("returns 1", result, 1);
    ASSERT_INT_EQ("target cleared", ix.target_slot, -1);
}


static void test_no_interrupt_none(void) {
    printf("--- no interrupt: NONE ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_NONE);
    ASSERT_INT_EQ("returns 0", result, 0);
    ASSERT_INT_EQ("target persists", ix.target_slot, 5);
}


static void test_no_interrupt_prayer(void) {
    printf("--- no interrupt: PRAYER ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_PRAYER);
    ASSERT_INT_EQ("returns 0", result, 0);
    ASSERT_INT_EQ("target persists", ix.target_slot, 5);
}


static void test_no_interrupt_spec(void) {
    printf("--- no interrupt: SPEC ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_SPEC);
    ASSERT_INT_EQ("returns 0", result, 0);
    ASSERT_INT_EQ("target persists", ix.target_slot, 5);
}


static void test_no_interrupt_attack(void) {
    printf("--- no interrupt: ATTACK ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_ATTACK);
    ASSERT_INT_EQ("returns 0", result, 0);
    ASSERT_INT_EQ("target persists", ix.target_slot, 5);
}


static void test_interrupt_when_inactive(void) {
    printf("--- interrupt when no interaction ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    int result = osrs_interaction_check_interrupt(&ix, OSRS_IACT_MOVE);
    ASSERT_INT_EQ("returns 1 (idempotent)", result, 1);
    ASSERT_INT_EQ("target still -1", ix.target_slot, -1);
}


static void test_set_replaces(void) {
    printf("--- set replaces ---\n");
    OsrsInteraction ix;
    osrs_interaction_init(&ix);
    osrs_interaction_set(&ix, 5);
    osrs_interaction_set(&ix, 3);
    ASSERT_INT_EQ("target_slot is 3", ix.target_slot, 3);
}


static void test_spec_toggle(void) {
    printf("--- spec toggle ---\n");
    int spec_armed = 0;
    osrs_spec_toggle(&spec_armed);
    ASSERT_INT_EQ("armed after toggle", spec_armed, 1);
    osrs_spec_toggle(&spec_armed);
    ASSERT_INT_EQ("disarmed after second toggle", spec_armed, 0);
}


static void test_spec_disarm(void) {
    printf("--- spec disarm ---\n");
    int spec_armed = 1;
    osrs_spec_disarm(&spec_armed);
    ASSERT_INT_EQ("disarmed", spec_armed, 0);
}


int main(void) {
    printf("=== osrs_interaction tests ===\n\n");

    test_init();
    test_set();
    test_clear();
    test_interrupt_move();
    test_interrupt_eat();
    test_interrupt_drink();
    test_interrupt_equip();
    test_no_interrupt_none();
    test_no_interrupt_prayer();
    test_no_interrupt_spec();
    test_no_interrupt_attack();
    test_interrupt_when_inactive();
    test_set_replaces();
    test_spec_toggle();
    test_spec_disarm();

    printf("\n=== results: %d passed, %d failed, %d total ===\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}
