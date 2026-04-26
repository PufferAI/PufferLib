/**
 * @file test_gui_inventory.c
 * @brief Regression tests for GUI inventory snapshot/reset logic used by inferno human mode.
 *
 * BUILD:
 *   cc -std=c11 -O0 -g -I. -I./raylib-5.5_macos/include -o /tmp/test_gui_inventory \
 *       ocean/osrs/tests/test_gui_inventory.c ./raylib-5.5_macos/lib/libraylib.a \
 *       -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo -lm
 *   /tmp/test_gui_inventory
 */

#include <stdio.h>
#include <string.h>

#include "ocean/osrs/osrs_pvp_actions.h"
#include "ocean/osrs/osrs_gui.h"
#include "ocean/osrs/osrs_human_input.h"

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

static int find_slot_of_type(const GuiState* gs, InvSlotType type) {
    for (int i = 0; i < INV_GRID_SLOTS; i++) {
        if (gs->inv_grid[i].type == type) return i;
    }
    return -1;
}

static int count_slots_of_type(const GuiState* gs, InvSlotType type) {
    int count = 0;
    for (int i = 0; i < INV_GRID_SLOTS; i++) {
        if (gs->inv_grid[i].type == type) count++;
    }
    return count;
}

static void test_gui_populate_tracks_bastion_and_stamina(void) {
    printf("--- gui populate tracks inferno potion snapshots ---\n");

    GuiState gs;
    Player p;
    memset(&gs, 0, sizeof(gs));
    memset(&p, 0, sizeof(p));

    p.bastion_doses = 8;
    p.stamina_doses = 4;

    gui_populate_inventory(&gs, &p);

    ASSERT_INT_EQ("snapshot keeps bastion doses", gs.inv_prev_bastion_doses, 8);
    ASSERT_INT_EQ("snapshot keeps stamina doses", gs.inv_prev_stamina_doses, 4);
    ASSERT_INT_EQ("bastion vials present", count_slots_of_type(&gs, INV_SLOT_BASTION_POT), 2);
    ASSERT_INT_EQ("stamina vial present", find_slot_of_type(&gs, INV_SLOT_STAMINA_POT) >= 0, 1);
}

static void test_gui_update_tracks_bastion_and_stamina(void) {
    printf("--- gui update tracks inferno potion deltas ---\n");

    GuiState gs;
    Player p;
    memset(&gs, 0, sizeof(gs));
    memset(&p, 0, sizeof(p));

    p.bastion_doses = 8;
    p.stamina_doses = 4;
    gui_populate_inventory(&gs, &p);

    int bastion_slot = find_slot_of_type(&gs, INV_SLOT_BASTION_POT);
    int stamina_slot = find_slot_of_type(&gs, INV_SLOT_STAMINA_POT);

    gs.human_clicked_inv_slot = bastion_slot;
    p.bastion_doses = 7;
    gui_update_inventory(&gs, &p);

    ASSERT_INT_EQ("bastion click latch clears after use", gs.human_clicked_inv_slot, -1);
    ASSERT_INT_EQ("bastion snapshot updates", gs.inv_prev_bastion_doses, 7);
    ASSERT_INT_EQ("bastion keeps two vials after one sip", count_slots_of_type(&gs, INV_SLOT_BASTION_POT), 2);
    ASSERT_INT_EQ("bastion sprite downgrades to 3-dose",
        gs.inv_grid[bastion_slot].osrs_id, gui_consumable_osrs_id(INV_SLOT_BASTION_POT, 3));

    gs.human_clicked_inv_slot = stamina_slot;
    p.stamina_doses = 3;
    gui_update_inventory(&gs, &p);

    ASSERT_INT_EQ("stamina click latch clears after use", gs.human_clicked_inv_slot, -1);
    ASSERT_INT_EQ("stamina snapshot updates", gs.inv_prev_stamina_doses, 3);
    ASSERT_INT_EQ("stamina sprite downgrades to 3-dose",
        gs.inv_grid[stamina_slot].osrs_id, gui_consumable_osrs_id(INV_SLOT_STAMINA_POT, 3));
}

static void test_gui_reset_helper_clears_inventory_interaction_state(void) {
    printf("--- gui reset helper clears inventory interaction state ---\n");

    GuiState gs;
    memset(&gs, 0, sizeof(gs));
    gs.inv_grid_dirty = 0;
    gs.human_clicked_inv_slot = 7;
    gs.inv_dim_slot = 3;
    gs.inv_dim_timer = 9;
    gs.inv_drag_active = 1;
    gs.inv_drag_src_slot = 12;
    gs.inv_drag_start_x = 100;
    gs.inv_drag_start_y = 120;
    gs.inv_drag_mouse_x = 130;
    gs.inv_drag_mouse_y = 140;

    gui_reset_inventory_ui_state(&gs);

    ASSERT_INT_EQ("grid marked dirty", gs.inv_grid_dirty, 1);
    ASSERT_INT_EQ("clicked slot cleared", gs.human_clicked_inv_slot, -1);
    ASSERT_INT_EQ("dim slot cleared", gs.inv_dim_slot, -1);
    ASSERT_INT_EQ("dim timer cleared", gs.inv_dim_timer, 0);
    ASSERT_INT_EQ("drag deactivated", gs.inv_drag_active, 0);
    ASSERT_INT_EQ("drag source cleared", gs.inv_drag_src_slot, -1);
}

static void test_gui_reset_rebuild_restores_potions(void) {
    printf("--- gui reset rebuild restores inferno potions ---\n");

    GuiState gs;
    Player p;
    memset(&gs, 0, sizeof(gs));
    memset(&p, 0, sizeof(p));

    p.bastion_doses = 8;
    p.stamina_doses = 4;
    gui_populate_inventory(&gs, &p);

    p.bastion_doses = 0;
    p.stamina_doses = 0;
    gui_update_inventory(&gs, &p);
    ASSERT_INT_EQ("bastion gone after depletion", find_slot_of_type(&gs, INV_SLOT_BASTION_POT), -1);
    ASSERT_INT_EQ("stamina gone after depletion", find_slot_of_type(&gs, INV_SLOT_STAMINA_POT), -1);

    p.bastion_doses = 8;
    p.stamina_doses = 4;
    gui_reset_inventory_ui_state(&gs);
    if (gs.inv_grid_dirty) {
        gui_populate_inventory(&gs, &p);
        gs.inv_grid_dirty = 0;
    }

    ASSERT_INT_EQ("bastion restored after rebuild", count_slots_of_type(&gs, INV_SLOT_BASTION_POT), 2);
    ASSERT_INT_EQ("stamina restored after rebuild", find_slot_of_type(&gs, INV_SLOT_STAMINA_POT) >= 0, 1);
    ASSERT_INT_EQ("bastion snapshot restored", gs.inv_prev_bastion_doses, 8);
    ASSERT_INT_EQ("stamina snapshot restored", gs.inv_prev_stamina_doses, 4);
}

static void test_human_equipment_click_queues_without_mutating_player(void) {
    printf("--- human equipment click queues without mutating player ---\n");

    GuiState gs;
    Player p;
    HumanInput hi;
    memset(&gs, 0, sizeof(gs));
    memset(&p, 0, sizeof(p));
    human_input_init(&hi);

    hi.enabled = 1;
    p.equipped[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;
    gs.inv_grid[0].type = INV_SLOT_EQUIPMENT;
    gs.inv_grid[0].item_db_idx = ITEM_TOXIC_BLOWPIPE;
    gs.inv_grid[0].osrs_id = ITEM_DATABASE[ITEM_TOXIC_BLOWPIPE].item_id;

    InvAction action = gui_inv_click(&gs, &p, 0, &hi);

    ASSERT_INT_EQ("equipment click returns equip action", action, INV_ACTION_EQUIP);
    ASSERT_INT_EQ("weapon not mutated before tick",
        p.equipped[GEAR_SLOT_WEAPON], ITEM_KODAI_WAND);
    ASSERT_INT_EQ("one command queued", hi.commands.count, 1);
    ASSERT_INT_EQ("queued equip command",
        hi.commands.items[0].kind, HUMAN_COMMAND_EQUIP_INVENTORY_ITEM);
    ASSERT_INT_EQ("queued source slot", hi.commands.items[0].inventory_slot, 0);
    ASSERT_INT_EQ("queued item", hi.commands.items[0].item_db_idx, ITEM_TOXIC_BLOWPIPE);

    human_input_destroy(&hi);
}

int main(void) {
    test_gui_populate_tracks_bastion_and_stamina();
    test_gui_update_tracks_bastion_and_stamina();
    test_gui_reset_helper_clears_inventory_interaction_state();
    test_gui_reset_rebuild_restores_potions();
    test_human_equipment_click_queues_without_mutating_player();

    printf("\n%d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d failed)\n", tests_failed);
        return 1;
    }
    printf("\n");
    return 0;
}
