/**
 * @file osrs_inventory.h
 * @brief shared 28-slot inventory + 11-slot equipment management
 *
 * real OSRS inventory model: 11 equipment slots (worn gear) + 28 inventory
 * slots (bag). handles equip/unequip with two-handed weapon logic and
 * inventory<->equipment swaps.
 *
 * replaces: osrs_pvp_gear.h slot_equip_item(), item_to_gear_slot(),
 *           osrs_pvp_combat.h has_free_inventory_slot()
 */

#ifndef OSRS_INVENTORY_H
#define OSRS_INVENTORY_H

#include "osrs_types.h"
#include "osrs_items.h"

#define OSRS_INVENTORY_SIZE 28

/* a complete player equipment state: worn gear + inventory bag.
   encounters that don't need full inventory (like current inferno) can ignore
   the inventory array and just use equipment[]. */
typedef struct {
    uint8_t equipment[NUM_GEAR_SLOTS];
    uint8_t inventory[OSRS_INVENTORY_SIZE];
} OsrsInventory;


/** initialize inventory: all slots to ITEM_NONE. */
static inline void osrs_inventory_init(OsrsInventory* inv) {
    memset(inv->equipment, ITEM_NONE, NUM_GEAR_SLOTS);
    memset(inv->inventory, ITEM_NONE, OSRS_INVENTORY_SIZE);
}

/** count occupied inventory slots. */
static inline int osrs_inventory_count(const OsrsInventory* inv) {
    int count = 0;
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        if (inv->inventory[i] != ITEM_NONE) count++;
    }
    return count;
}

/** count free inventory slots. */
static inline int osrs_inventory_free_slots(const OsrsInventory* inv) {
    return OSRS_INVENTORY_SIZE - osrs_inventory_count(inv);
}

/** find first inventory slot containing item_idx. returns slot index or -1. */
static inline int osrs_inventory_find(const OsrsInventory* inv, uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return -1;
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        if (inv->inventory[i] == item_idx) return i;
    }
    return -1;
}

/** add item to first free inventory slot. returns slot index or -1 if full. */
static inline int osrs_inventory_add(OsrsInventory* inv, uint8_t item_idx) {
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        if (inv->inventory[i] == ITEM_NONE) {
            inv->inventory[i] = item_idx;
            return i;
        }
    }
    return -1;
}

/** remove item from specific inventory slot. returns the item removed (ITEM_NONE if empty). */
static inline uint8_t osrs_inventory_remove(OsrsInventory* inv, int slot) {
    if (slot < 0 || slot >= OSRS_INVENTORY_SIZE) return ITEM_NONE;
    uint8_t item = inv->inventory[slot];
    inv->inventory[slot] = ITEM_NONE;
    return item;
}

/** remove first occurrence of item_idx. returns 1 if found+removed, 0 if not found. */
static inline int osrs_inventory_remove_item(OsrsInventory* inv, uint8_t item_idx) {
    int slot = osrs_inventory_find(inv, item_idx);
    if (slot < 0) return 0;
    inv->inventory[slot] = ITEM_NONE;
    return 1;
}


/** map item index to its gear slot. returns GearSlotIndex or -1 if unmapped.
    replaces item_to_gear_slot() in osrs_pvp_gear.h:820. */
static inline int osrs_item_gear_slot(uint8_t item_idx) {
    if (item_idx >= NUM_ITEMS) return -1;
    switch (ITEM_DATABASE[item_idx].slot) {
        case SLOT_HEAD:   return GEAR_SLOT_HEAD;
        case SLOT_CAPE:   return GEAR_SLOT_CAPE;
        case SLOT_NECK:   return GEAR_SLOT_NECK;
        case SLOT_WEAPON: return GEAR_SLOT_WEAPON;
        case SLOT_BODY:   return GEAR_SLOT_BODY;
        case SLOT_SHIELD: return GEAR_SLOT_SHIELD;
        case SLOT_LEGS:   return GEAR_SLOT_LEGS;
        case SLOT_HANDS:  return GEAR_SLOT_HANDS;
        case SLOT_FEET:   return GEAR_SLOT_FEET;
        case SLOT_RING:   return GEAR_SLOT_RING;
        case SLOT_AMMO:   return GEAR_SLOT_AMMO;
        default: return -1;
    }
}


/** equip item directly (not from inventory -- used for initial setup).
    places item in correct gear slot, no inventory interaction.
    two-handed weapons clear the shield slot. */
static inline void osrs_equip_direct(OsrsInventory* inv, uint8_t item_idx) {
    int slot = osrs_item_gear_slot(item_idx);
    if (slot < 0) return;
    inv->equipment[slot] = item_idx;
    if (slot == GEAR_SLOT_WEAPON && item_is_two_handed(item_idx)) {
        inv->equipment[GEAR_SLOT_SHIELD] = ITEM_NONE;
    }
}

/** equip item from inventory slot to its gear slot.
    if gear slot occupied, old item swaps to inventory.
    two-handed weapons unequip shield to inventory first (fails if no space).
    returns 1 on success, 0 on failure.
    ref: osrs_pvp_gear.h slot_equip_item() line 283. */
static inline int osrs_equip_from_inventory(OsrsInventory* inv, int inventory_slot) {
    if (inventory_slot < 0 || inventory_slot >= OSRS_INVENTORY_SIZE) return 0;
    uint8_t item_idx = inv->inventory[inventory_slot];
    if (item_idx == ITEM_NONE) return 0;

    int gear_slot = osrs_item_gear_slot(item_idx);
    if (gear_slot < 0) return 0;

    /* two-handed: need to unequip shield if present */
    if (gear_slot == GEAR_SLOT_WEAPON && item_is_two_handed(item_idx)) {
        uint8_t shield = inv->equipment[GEAR_SLOT_SHIELD];
        if (shield != ITEM_NONE) {
            /* need a free slot for the shield (the inventory_slot will be freed
               by the equip, so count that as available) */
            int free = osrs_inventory_free_slots(inv);
            /* inventory_slot is occupied by the item we're equipping, so after
               we remove it there's +1 free. but we also need to place the old
               weapon (if any) back. */
            uint8_t old_weapon = inv->equipment[GEAR_SLOT_WEAPON];
            int slots_needed = 1;  /* shield */
            if (old_weapon != ITEM_NONE) slots_needed++;  /* old weapon swap */
            /* we free 1 slot (inventory_slot) by equipping from it */
            if (free + 1 < slots_needed) return 0;

            /* clear shield slot */
            inv->equipment[GEAR_SLOT_SHIELD] = ITEM_NONE;
            /* place shield in inventory */
            if (old_weapon != ITEM_NONE) {
                /* swap: remove item from inventory, equip it, put old weapon + shield back */
                inv->inventory[inventory_slot] = old_weapon;
                inv->equipment[gear_slot] = item_idx;
                osrs_inventory_add(inv, shield);
            } else {
                /* no old weapon: remove item, equip it, put shield in freed slot */
                inv->inventory[inventory_slot] = shield;
                inv->equipment[gear_slot] = item_idx;
            }
            return 1;
        }
    }

    /* standard equip: swap with whatever is in the gear slot */
    uint8_t old_item = inv->equipment[gear_slot];
    inv->equipment[gear_slot] = item_idx;
    inv->inventory[inventory_slot] = old_item;  /* ITEM_NONE if slot was empty */
    return 1;
}

/** unequip gear slot to inventory. returns 1 if successful (had space), 0 if full. */
static inline int osrs_unequip_to_inventory(OsrsInventory* inv, int gear_slot) {
    if (gear_slot < 0 || gear_slot >= NUM_GEAR_SLOTS) return 0;
    uint8_t item = inv->equipment[gear_slot];
    if (item == ITEM_NONE) return 0;
    int slot = osrs_inventory_add(inv, item);
    if (slot < 0) return 0;
    inv->equipment[gear_slot] = ITEM_NONE;
    return 1;
}

#endif /* OSRS_INVENTORY_H */
