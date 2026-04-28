/**
 * @file test_inventory.c
 * @brief tests for osrs_inventory.h: 28-slot inventory + equipment management
 *
 * BUILD:
 *   cd pufferlib-metal
 *   cc -std=c11 -O0 -g -I. -o test_inventory \
 *       ocean/osrs/tests/test_inventory.c -lm
 *   ./test_inventory
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/osrs_inventory.h"


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

#define ASSERT_UINT8_EQ(label, actual, expected) do { \
    tests_run++; \
    uint8_t _a = (actual), _e = (expected); \
    if (_a == _e) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s — got %u, expected %u\n", (label), (unsigned)_a, (unsigned)_e); \
    } \
} while (0)


static void test_init(void) {
    printf("--- init ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    for (int i = 0; i < NUM_GEAR_SLOTS; i++) {
        ASSERT_UINT8_EQ("equipment slot empty", inv.equipment[i], ITEM_NONE);
    }
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        ASSERT_UINT8_EQ("inventory slot empty", inv.inventory[i], ITEM_NONE);
    }
    ASSERT_INT_EQ("count 0", osrs_inventory_count(&inv), 0);
    ASSERT_INT_EQ("free 28", osrs_inventory_free_slots(&inv), 28);
}


static void test_add_remove(void) {
    printf("--- inventory add/remove ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    /* add items */
    int s0 = osrs_inventory_add(&inv, ITEM_WHIP);
    int s1 = osrs_inventory_add(&inv, ITEM_DRAGON_DAGGER);
    int s2 = osrs_inventory_add(&inv, ITEM_AGS);
    ASSERT_INT_EQ("whip slot 0", s0, 0);
    ASSERT_INT_EQ("dds slot 1", s1, 1);
    ASSERT_INT_EQ("ags slot 2", s2, 2);
    ASSERT_INT_EQ("count 3", osrs_inventory_count(&inv), 3);
    ASSERT_INT_EQ("free 25", osrs_inventory_free_slots(&inv), 25);

    /* remove by slot */
    uint8_t removed = osrs_inventory_remove(&inv, 1);
    ASSERT_UINT8_EQ("removed dds", removed, ITEM_DRAGON_DAGGER);
    ASSERT_INT_EQ("count 2", osrs_inventory_count(&inv), 2);
    ASSERT_UINT8_EQ("slot 1 now empty", inv.inventory[1], ITEM_NONE);

    /* remove by item */
    int ok = osrs_inventory_remove_item(&inv, ITEM_AGS);
    ASSERT_INT_EQ("remove ags ok", ok, 1);
    ASSERT_INT_EQ("count 1", osrs_inventory_count(&inv), 1);

    /* remove nonexistent */
    ok = osrs_inventory_remove_item(&inv, ITEM_AGS);
    ASSERT_INT_EQ("remove ags again fails", ok, 0);
}


static void test_inventory_full(void) {
    printf("--- inventory full ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        int s = osrs_inventory_add(&inv, ITEM_WHIP);
        ASSERT_INT_EQ("add succeeds", s, i);
    }
    ASSERT_INT_EQ("count 28", osrs_inventory_count(&inv), 28);
    ASSERT_INT_EQ("free 0", osrs_inventory_free_slots(&inv), 0);

    /* 29th item fails */
    int s = osrs_inventory_add(&inv, ITEM_DRAGON_DAGGER);
    ASSERT_INT_EQ("29th fails", s, -1);
}


static void test_find(void) {
    printf("--- find ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    osrs_inventory_add(&inv, ITEM_WHIP);
    osrs_inventory_add(&inv, ITEM_DRAGON_DAGGER);
    osrs_inventory_add(&inv, ITEM_WHIP);

    ASSERT_INT_EQ("find whip first", osrs_inventory_find(&inv, ITEM_WHIP), 0);
    ASSERT_INT_EQ("find dds", osrs_inventory_find(&inv, ITEM_DRAGON_DAGGER), 1);
    ASSERT_INT_EQ("find missing", osrs_inventory_find(&inv, ITEM_AGS), -1);
}


static void test_equip_direct(void) {
    printf("--- equip direct ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    osrs_equip_direct(&inv, ITEM_WHIP);
    ASSERT_UINT8_EQ("weapon = whip", inv.equipment[GEAR_SLOT_WEAPON], ITEM_WHIP);
    ASSERT_INT_EQ("inventory untouched", osrs_inventory_count(&inv), 0);

    osrs_equip_direct(&inv, ITEM_HELM_NEITIZNOT);
    ASSERT_UINT8_EQ("head = neit", inv.equipment[GEAR_SLOT_HEAD], ITEM_HELM_NEITIZNOT);

    osrs_equip_direct(&inv, ITEM_DRAGON_DEFENDER);
    ASSERT_UINT8_EQ("shield = defender", inv.equipment[GEAR_SLOT_SHIELD], ITEM_DRAGON_DEFENDER);

    /* equip 2h direct: clears shield */
    osrs_equip_direct(&inv, ITEM_AGS);
    ASSERT_UINT8_EQ("weapon = ags", inv.equipment[GEAR_SLOT_WEAPON], ITEM_AGS);
    ASSERT_UINT8_EQ("shield cleared", inv.equipment[GEAR_SLOT_SHIELD], ITEM_NONE);
}


static void test_equip_from_inventory(void) {
    printf("--- equip from inventory ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    osrs_inventory_add(&inv, ITEM_WHIP);  /* slot 0 */
    int ok = osrs_equip_from_inventory(&inv, 0);
    ASSERT_INT_EQ("equip ok", ok, 1);
    ASSERT_UINT8_EQ("weapon = whip", inv.equipment[GEAR_SLOT_WEAPON], ITEM_WHIP);
    ASSERT_UINT8_EQ("inv slot 0 cleared", inv.inventory[0], ITEM_NONE);
    ASSERT_INT_EQ("inv count 0", osrs_inventory_count(&inv), 0);
}


static void test_equip_swap(void) {
    printf("--- equip swap ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    /* equip whip directly */
    osrs_equip_direct(&inv, ITEM_WHIP);
    /* put rapier in inventory */
    osrs_inventory_add(&inv, ITEM_GHRAZI_RAPIER);  /* slot 0 */

    int ok = osrs_equip_from_inventory(&inv, 0);
    ASSERT_INT_EQ("swap ok", ok, 1);
    ASSERT_UINT8_EQ("weapon = rapier", inv.equipment[GEAR_SLOT_WEAPON], ITEM_GHRAZI_RAPIER);
    ASSERT_UINT8_EQ("whip in inv slot 0", inv.inventory[0], ITEM_WHIP);
    ASSERT_INT_EQ("inv count 1", osrs_inventory_count(&inv), 1);
}


static void test_two_handed_equip(void) {
    printf("--- two-handed equip ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    /* setup: whip + defender equipped */
    osrs_equip_direct(&inv, ITEM_WHIP);
    osrs_equip_direct(&inv, ITEM_DRAGON_DEFENDER);
    /* put AGS in inventory */
    osrs_inventory_add(&inv, ITEM_AGS);  /* slot 0 */

    int ok = osrs_equip_from_inventory(&inv, 0);
    ASSERT_INT_EQ("2h equip ok", ok, 1);
    ASSERT_UINT8_EQ("weapon = ags", inv.equipment[GEAR_SLOT_WEAPON], ITEM_AGS);
    ASSERT_UINT8_EQ("shield cleared", inv.equipment[GEAR_SLOT_SHIELD], ITEM_NONE);
    /* old weapon (whip) goes to inv slot 0 (where AGS was), defender goes to next free */
    ASSERT_UINT8_EQ("whip in inv", inv.inventory[0], ITEM_WHIP);
    int def_slot = osrs_inventory_find(&inv, ITEM_DRAGON_DEFENDER);
    ASSERT_INT_EQ("defender in inv", def_slot >= 0, 1);
    ASSERT_INT_EQ("inv count 2", osrs_inventory_count(&inv), 2);
}


static void test_two_handed_no_old_weapon(void) {
    printf("--- two-handed no old weapon ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    /* setup: only defender equipped, no weapon */
    osrs_equip_direct(&inv, ITEM_DRAGON_DEFENDER);
    /* put AGS in inventory */
    osrs_inventory_add(&inv, ITEM_AGS);  /* slot 0 */

    int ok = osrs_equip_from_inventory(&inv, 0);
    ASSERT_INT_EQ("2h equip ok", ok, 1);
    ASSERT_UINT8_EQ("weapon = ags", inv.equipment[GEAR_SLOT_WEAPON], ITEM_AGS);
    ASSERT_UINT8_EQ("shield cleared", inv.equipment[GEAR_SLOT_SHIELD], ITEM_NONE);
    /* defender goes to inv slot 0 (where AGS was) */
    ASSERT_UINT8_EQ("defender in slot 0", inv.inventory[0], ITEM_DRAGON_DEFENDER);
    ASSERT_INT_EQ("inv count 1", osrs_inventory_count(&inv), 1);
}


static void test_two_handed_fail(void) {
    printf("--- two-handed fail ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    /* setup: whip + defender equipped, inventory full except AGS slot */
    osrs_equip_direct(&inv, ITEM_WHIP);
    osrs_equip_direct(&inv, ITEM_DRAGON_DEFENDER);
    /* fill 27 slots, then put AGS in slot 27 */
    for (int i = 0; i < 27; i++) {
        osrs_inventory_add(&inv, ITEM_CLIMBING_BOOTS);
    }
    osrs_inventory_add(&inv, ITEM_AGS);  /* slot 27 */
    ASSERT_INT_EQ("inv full", osrs_inventory_free_slots(&inv), 0);

    /* equipping AGS frees slot 27, but we need 2 slots (old whip + defender) — only 1 free */
    int ok = osrs_equip_from_inventory(&inv, 27);
    ASSERT_INT_EQ("2h equip fails", ok, 0);
    /* nothing changed */
    ASSERT_UINT8_EQ("weapon still whip", inv.equipment[GEAR_SLOT_WEAPON], ITEM_WHIP);
    ASSERT_UINT8_EQ("shield still defender", inv.equipment[GEAR_SLOT_SHIELD], ITEM_DRAGON_DEFENDER);
    ASSERT_UINT8_EQ("ags still in inv", inv.inventory[27], ITEM_AGS);
}


static void test_two_handed_exact_space(void) {
    printf("--- two-handed exact space ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    /* setup: whip + defender equipped, 26 items + AGS = 27 occupied, 1 free */
    osrs_equip_direct(&inv, ITEM_WHIP);
    osrs_equip_direct(&inv, ITEM_DRAGON_DEFENDER);
    for (int i = 0; i < 26; i++) {
        osrs_inventory_add(&inv, ITEM_CLIMBING_BOOTS);
    }
    osrs_inventory_add(&inv, ITEM_AGS);  /* slot 26 */
    ASSERT_INT_EQ("1 free slot", osrs_inventory_free_slots(&inv), 1);

    /* equipping AGS: frees slot 26 (+1) plus 1 existing free = 2 free, need 2 (whip + defender) */
    int ok = osrs_equip_from_inventory(&inv, 26);
    ASSERT_INT_EQ("2h equip ok", ok, 1);
    ASSERT_UINT8_EQ("weapon = ags", inv.equipment[GEAR_SLOT_WEAPON], ITEM_AGS);
    ASSERT_UINT8_EQ("shield cleared", inv.equipment[GEAR_SLOT_SHIELD], ITEM_NONE);
}


static void test_unequip(void) {
    printf("--- unequip ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    osrs_equip_direct(&inv, ITEM_WHIP);
    int ok = osrs_unequip_to_inventory(&inv, GEAR_SLOT_WEAPON);
    ASSERT_INT_EQ("unequip ok", ok, 1);
    ASSERT_UINT8_EQ("weapon cleared", inv.equipment[GEAR_SLOT_WEAPON], ITEM_NONE);
    ASSERT_INT_EQ("whip in inv", osrs_inventory_find(&inv, ITEM_WHIP), 0);

    /* unequip empty slot */
    ok = osrs_unequip_to_inventory(&inv, GEAR_SLOT_WEAPON);
    ASSERT_INT_EQ("unequip empty fails", ok, 0);

    /* unequip with full inventory */
    for (int i = 1; i < OSRS_INVENTORY_SIZE; i++) {
        osrs_inventory_add(&inv, ITEM_CLIMBING_BOOTS);
    }
    ASSERT_INT_EQ("inv full", osrs_inventory_free_slots(&inv), 0);
    osrs_equip_direct(&inv, ITEM_HELM_NEITIZNOT);
    ok = osrs_unequip_to_inventory(&inv, GEAR_SLOT_HEAD);
    ASSERT_INT_EQ("unequip full fails", ok, 0);
    ASSERT_UINT8_EQ("head still equipped", inv.equipment[GEAR_SLOT_HEAD], ITEM_HELM_NEITIZNOT);
}


static void test_gear_slot_mapping(void) {
    printf("--- gear slot mapping ---\n");

    ASSERT_INT_EQ("whip -> weapon",    osrs_item_gear_slot(ITEM_WHIP),             GEAR_SLOT_WEAPON);
    ASSERT_INT_EQ("neit -> head",      osrs_item_gear_slot(ITEM_HELM_NEITIZNOT),   GEAR_SLOT_HEAD);
    ASSERT_INT_EQ("defender -> shield", osrs_item_gear_slot(ITEM_DRAGON_DEFENDER),  GEAR_SLOT_SHIELD);
    ASSERT_INT_EQ("fury -> neck",      osrs_item_gear_slot(ITEM_FURY),             GEAR_SLOT_NECK);
    ASSERT_INT_EQ("infernal -> cape",  osrs_item_gear_slot(ITEM_INFERNAL_CAPE),    GEAR_SLOT_CAPE);
    ASSERT_INT_EQ("tassets -> legs",   osrs_item_gear_slot(ITEM_BANDOS_TASSETS),   GEAR_SLOT_LEGS);
    ASSERT_INT_EQ("bgloves -> hands",  osrs_item_gear_slot(ITEM_BARROWS_GLOVES),  GEAR_SLOT_HANDS);
    ASSERT_INT_EQ("cboots -> feet",    osrs_item_gear_slot(ITEM_CLIMBING_BOOTS),   GEAR_SLOT_FEET);
    ASSERT_INT_EQ("bring -> ring",     osrs_item_gear_slot(ITEM_BERSERKER_RING),   GEAR_SLOT_RING);
    ASSERT_INT_EQ("dbolts -> ammo",    osrs_item_gear_slot(ITEM_DIAMOND_BOLTS_E),  GEAR_SLOT_AMMO);
    ASSERT_INT_EQ("ags -> weapon",     osrs_item_gear_slot(ITEM_AGS),              GEAR_SLOT_WEAPON);
}


static void test_edge_cases(void) {
    printf("--- edge cases ---\n");
    OsrsInventory inv;
    osrs_inventory_init(&inv);

    /* ITEM_NONE operations */
    ASSERT_INT_EQ("gear slot ITEM_NONE", osrs_item_gear_slot(ITEM_NONE), -1);
    ASSERT_INT_EQ("find ITEM_NONE in empty", osrs_inventory_find(&inv, ITEM_NONE), -1);

    /* remove from empty slot */
    uint8_t r = osrs_inventory_remove(&inv, 0);
    ASSERT_UINT8_EQ("remove empty = ITEM_NONE", r, ITEM_NONE);

    /* remove out of bounds */
    r = osrs_inventory_remove(&inv, -1);
    ASSERT_UINT8_EQ("remove -1 = ITEM_NONE", r, ITEM_NONE);
    r = osrs_inventory_remove(&inv, 28);
    ASSERT_UINT8_EQ("remove 28 = ITEM_NONE", r, ITEM_NONE);

    /* equip from invalid inventory slot */
    ASSERT_INT_EQ("equip slot -1", osrs_equip_from_inventory(&inv, -1), 0);
    ASSERT_INT_EQ("equip slot 28", osrs_equip_from_inventory(&inv, 28), 0);

    /* equip from empty inventory slot */
    ASSERT_INT_EQ("equip empty slot", osrs_equip_from_inventory(&inv, 0), 0);

    /* unequip invalid gear slot */
    ASSERT_INT_EQ("unequip slot -1", osrs_unequip_to_inventory(&inv, -1), 0);
    ASSERT_INT_EQ("unequip slot 11", osrs_unequip_to_inventory(&inv, NUM_GEAR_SLOTS), 0);
}


int main(void) {
    printf("=== osrs_inventory.h tests ===\n\n");

    test_init();
    test_add_remove();
    test_inventory_full();
    test_find();
    test_equip_direct();
    test_equip_from_inventory();
    test_equip_swap();
    test_two_handed_equip();
    test_two_handed_no_old_weapon();
    test_two_handed_fail();
    test_two_handed_exact_space();
    test_unequip();
    test_gear_slot_mapping();
    test_edge_cases();

    printf("\n=== results: %d passed, %d failed, %d total ===\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}
