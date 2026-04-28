/**
 * @file osrs_items.h
 * @brief Item database with OSRS item IDs and equipment stats
 *
 * Provides a static database of LMS items with real OSRS item IDs.
 * Each item has complete equipment stats sourced from OSRS wiki/game data.
 * Item indices and stats are auto-generated from equipment.json via
 * osrs_items_generated.h; this file owns the shared types (EquipmentSlot,
 * Item struct) and lookup tables.
 */

#ifndef OSRS_ITEMS_H
#define OSRS_ITEMS_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    SLOT_HEAD = 0,
    SLOT_CAPE = 1,
    SLOT_NECK = 2,
    SLOT_WEAPON = 3,
    SLOT_BODY = 4,
    SLOT_SHIELD = 5,
    SLOT_LEGS = 6,
    SLOT_HANDS = 7,
    SLOT_FEET = 8,
    SLOT_RING = 9,
    SLOT_AMMO = 10,
    NUM_EQUIPMENT_SLOTS = 11
} EquipmentSlot;

typedef enum {
    OSRS_ITEM_EFFECT_NONE = 0,
    OSRS_ITEM_EFFECT_TWISTED_BOW = 1u << 0,
    OSRS_ITEM_EFFECT_VIRTUS_PIECE = 1u << 1,
    OSRS_ITEM_EFFECT_CONFLICTION = 1u << 2,
    OSRS_ITEM_EFFECT_SANG_HEAL = 1u << 3,
    OSRS_ITEM_EFFECT_RECOIL_RING = 1u << 4,
    OSRS_ITEM_EFFECT_LIGHTBEARER = 1u << 5,
    OSRS_ITEM_EFFECT_DHAROK_PIECE = 1u << 6,
    OSRS_ITEM_EFFECT_ELYSIAN = 1u << 7,
} OsrsItemEffectMask;

typedef struct {
    uint16_t item_id;           // Real OSRS item ID
    char name[32];              // Human-readable name
    uint8_t slot;               // Equipment slot (EquipmentSlot enum)
    uint8_t attack_speed;       // Weapon attack speed (ticks)
    uint8_t attack_range;       // Weapon attack range (tiles)
    int16_t attack_stab;
    int16_t attack_slash;
    int16_t attack_crush;
    int16_t attack_magic;
    int16_t attack_ranged;
    int16_t defence_stab;
    int16_t defence_slash;
    int16_t defence_crush;
    int16_t defence_magic;
    int16_t defence_ranged;
    int16_t melee_strength;
    int16_t ranged_strength;
    int16_t magic_damage;       // Magic damage % bonus
    int16_t prayer;
    uint32_t effect_mask;
} Item;
// ITEM DATABASE INDICES + STATIC DATABASE (auto-generated from equipment.json)

#include "osrs_items_generated.h"

// Max items per slot (inventory width for dynamic gear)
#define MAX_ITEMS_PER_SLOT_DB 10

// Items available per slot (for masking and inventory)
// 255 = end marker (slot has fewer than MAX_ITEMS_PER_SLOT_DB options)
static const uint8_t ITEMS_BY_SLOT[NUM_EQUIPMENT_SLOTS][MAX_ITEMS_PER_SLOT_DB] = {
    [SLOT_HEAD]   = {ITEM_HELM_NEITIZNOT, ITEM_ANCESTRAL_HAT, ITEM_VIRTUS_MASK,
                     ITEM_TORAGS_HELM, ITEM_DHAROKS_HELM, ITEM_VERACS_HELM, ITEM_GUTHANS_HELM,
                     ITEM_NONE, ITEM_NONE, ITEM_NONE},
    [SLOT_CAPE]   = {ITEM_GOD_CAPE, ITEM_INFERNAL_CAPE,
                     ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE},
    [SLOT_NECK]   = {ITEM_GLORY, ITEM_FURY, ITEM_OCCULT_NECKLACE,
                     ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE},
    [SLOT_WEAPON] = {ITEM_WHIP, ITEM_RUNE_CROSSBOW, ITEM_AHRIM_STAFF, ITEM_DRAGON_DAGGER,
                     ITEM_GHRAZI_RAPIER, ITEM_INQUISITORS_MACE, ITEM_STAFF_OF_DEAD, ITEM_KODAI_WAND,
                     ITEM_VOLATILE_STAFF, ITEM_ZURIELS_STAFF},
    [SLOT_BODY]   = {ITEM_BLACK_DHIDE_BODY, ITEM_MYSTIC_TOP, ITEM_ANCESTRAL_TOP, ITEM_VIRTUS_ROBE_TOP,
                     ITEM_AHRIMS_ROBETOP, ITEM_KARILS_TOP,
                     ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE},
    [SLOT_SHIELD] = {ITEM_DRAGON_DEFENDER, ITEM_AVERNIC_DEFENDER, ITEM_ELYSIAN_SPIRIT_SHIELD,
                     ITEM_SPIRIT_SHIELD, ITEM_BLESSED_SPIRIT_SHIELD, ITEM_SPECTRAL_SPIRIT_SHIELD,
                     ITEM_CRYSTAL_SHIELD, ITEM_MAGES_BOOK, ITEM_ELIDINIS_WARD_F, ITEM_DRAGONFIRE_SHIELD},
    [SLOT_LEGS]   = {ITEM_RUNE_PLATELEGS, ITEM_MYSTIC_BOTTOM, ITEM_ANCESTRAL_BOTTOM, ITEM_VIRTUS_ROBE_BOTTOM,
                     ITEM_AHRIMS_ROBESKIRT,
                     ITEM_BANDOS_TASSETS, ITEM_TORAGS_PLATELEGS, ITEM_DHAROKS_PLATELEGS, ITEM_VERACS_PLATESKIRT,
                     ITEM_NONE},
    [SLOT_HANDS]  = {ITEM_BARROWS_GLOVES, ITEM_CONFLICTION_GAUNTLETS, ITEM_TORMENTED_BRACELET,
                     ITEM_ZARYTE_VAMBRACES, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE},
    [SLOT_FEET]   = {ITEM_CLIMBING_BOOTS, ITEM_ETERNAL_BOOTS, ITEM_AVERNIC_TREADS, ITEM_INFINITY_BOOTS,
                     ITEM_BLESSED_DHIDE_BOOTS, ITEM_MYSTIC_BOOTS, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE},
    [SLOT_RING]   = {ITEM_BERSERKER_RING, ITEM_SEERS_RING_I, ITEM_LIGHTBEARER, ITEM_VENATOR_RING,
                     ITEM_RING_OF_RECOIL, ITEM_RING_OF_SUFFERING_RI, ITEM_MAGUS_RING, ITEM_ULTOR_RING,
                     ITEM_NONE, ITEM_NONE},
    [SLOT_AMMO]   = {ITEM_DIAMOND_BOLTS_E, ITEM_DRAGON_ARROWS, ITEM_OPAL_DRAGON_BOLTS,
                     ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE, ITEM_NONE},
};

// Number of items per slot in the static DB table above
static const uint8_t NUM_ITEMS_IN_SLOT[NUM_EQUIPMENT_SLOTS] = {
    [SLOT_HEAD]   = 7,   // neitiznot, ancestral/virtus, torag/dharok/verac/guthan helms
    [SLOT_CAPE]   = 2,   // god cape, infernal
    [SLOT_NECK]   = 3,   // glory, fury, occult
    [SLOT_WEAPON] = 10,  // whip, rcb, ahrim, dds, rapier, inq mace, sotd, kodai, volatile, zuriel
    [SLOT_BODY]   = 6,   // dhide, mystic, ancestral, virtus, ahrim, karil
    [SLOT_SHIELD] = 10,  // defenders, spirit variants, crystal, elysian, ward, mages book, dfs
    [SLOT_LEGS]   = 9,   // rune, mystic, ancestral, virtus, ahrim, bandos, torag/dharok/verac legs
    [SLOT_HANDS]  = 4,   // barrows, confliction, tormented, zaryte
    [SLOT_FEET]   = 6,   // climbing, eternal, avernic, infinity, blessed d'hide, mystic
    [SLOT_RING]   = 8,   // berserker, seers (i), lightbearer, venator, recoil, suffering, magus, ultor
    [SLOT_AMMO]   = 3,   // diamond bolts (e), dragon arrows, opal dragon bolts (e)
};

/** Get item from database by index. Returns NULL if invalid. */
static inline const Item* get_item(uint8_t item_index) {
    if (item_index >= NUM_ITEMS) return NULL;
    return &ITEM_DATABASE[item_index];
}

/** Check if item is a weapon. */
static inline int item_is_weapon(uint8_t item_index) {
    if (item_index >= NUM_ITEMS) return 0;
    return ITEM_DATABASE[item_index].slot == SLOT_WEAPON;
}

/** Check if item is a shield. */
static inline int item_is_shield(uint8_t item_index) {
    if (item_index >= NUM_ITEMS) return 0;
    return ITEM_DATABASE[item_index].slot == SLOT_SHIELD;
}

/** Get attack style for a weapon item (1=melee, 2=ranged, 3=magic). */
static inline int get_item_attack_style(uint8_t item_index) {
    switch (item_index) {
        // Melee weapons
        case ITEM_WHIP:
        case ITEM_DRAGON_DAGGER:
        case ITEM_GHRAZI_RAPIER:
        case ITEM_INQUISITORS_MACE:
        case ITEM_DRAGON_CLAWS:
        case ITEM_AGS:
        case ITEM_ANCIENT_GS:
        case ITEM_GRANITE_MAUL:
        case ITEM_ELDER_MAUL:
        case ITEM_VESTAS:
        case ITEM_VOIDWAKER:
        case ITEM_STATIUS_WARHAMMER:
            return 1;  // ATTACK_STYLE_MELEE
        // Ranged weapons
        case ITEM_RUNE_CROSSBOW:
        case ITEM_ARMADYL_CROSSBOW:
        case ITEM_ZARYTE_CROSSBOW:
        case ITEM_DARK_BOW:
        case ITEM_HEAVY_BALLISTA:
        case ITEM_MORRIGANS_JAVELIN:
        case ITEM_MAGIC_SHORTBOW_I:
        case ITEM_BOW_OF_FAERDHINEN:
        case ITEM_TWISTED_BOW:
        case ITEM_TOXIC_BLOWPIPE:
            return 2;  // ATTACK_STYLE_RANGED
        // Magic weapons
        case ITEM_AHRIM_STAFF:
        case ITEM_STAFF_OF_DEAD:
        case ITEM_KODAI_WAND:
        case ITEM_VOLATILE_STAFF:
        case ITEM_ZURIELS_STAFF:
        case ITEM_TRIDENT_OF_SWAMP:
        case ITEM_SANGUINESTI_STAFF:
        case ITEM_EYE_OF_AYAK:
            return 3;  // ATTACK_STYLE_MAGIC
        default:
            return 0;  // ATTACK_STYLE_NONE
    }
}

/** Check if weapon is two-handed. */
static inline int item_is_two_handed(uint8_t item_index) {
    switch (item_index) {
        case ITEM_AGS:
        case ITEM_ANCIENT_GS:
        case ITEM_DRAGON_CLAWS:
        case ITEM_GRANITE_MAUL:
        case ITEM_ELDER_MAUL:
        case ITEM_DARK_BOW:
        case ITEM_HEAVY_BALLISTA:
        case ITEM_MAGIC_SHORTBOW_I:
        case ITEM_BOW_OF_FAERDHINEN:
        case ITEM_TWISTED_BOW:
        case ITEM_TOXIC_BLOWPIPE:
            return 1;
        default:
            return 0;
    }
}
// ITEM STATS EXTRACTION (for observations)

/** Normalization constants for item stats (max observed values in game). */
#define STAT_NORM_ATTACK 150.0f
#define STAT_NORM_DEFENCE 100.0f
#define STAT_NORM_STRENGTH 150.0f
#define STAT_NORM_MAGIC_DMG 30.0f
#define STAT_NORM_PRAYER 10.0f
#define STAT_NORM_SPEED 10.0f
#define STAT_NORM_RANGE 15.0f

/**
 * Extract normalized item stats for observations.
 *
 * Writes 18 floats to output buffer:
 *   [0-4]   attack bonuses (stab, slash, crush, magic, ranged)
 *   [5-9]   defence bonuses (stab, slash, crush, magic, ranged)
 *   [10-12] strength bonuses (melee, ranged, magic damage %)
 *   [13]    prayer bonus
 *   [14]    attack speed
 *   [15]    attack range
 *   [16]    is_weapon flag (for quick filtering)
 *   [17]    is_empty flag (1 if item_index >= NUM_ITEMS)
 *
 * @param item_index  Item database index
 * @param out         Output buffer (must have space for 18 floats)
 */
static inline void get_item_stats_normalized(uint8_t item_index, float* out) {
    if (item_index >= NUM_ITEMS) {
        // Empty slot - all zeros except is_empty flag
        for (int i = 0; i < 17; i++) out[i] = 0.0f;
        out[17] = 1.0f;  // is_empty
        return;
    }

    const Item* item = &ITEM_DATABASE[item_index];

    // Attack bonuses (normalized by max expected values)
    out[0] = (float)item->attack_stab / STAT_NORM_ATTACK;
    out[1] = (float)item->attack_slash / STAT_NORM_ATTACK;
    out[2] = (float)item->attack_crush / STAT_NORM_ATTACK;
    out[3] = (float)item->attack_magic / STAT_NORM_ATTACK;
    out[4] = (float)item->attack_ranged / STAT_NORM_ATTACK;

    // Defence bonuses
    out[5] = (float)item->defence_stab / STAT_NORM_DEFENCE;
    out[6] = (float)item->defence_slash / STAT_NORM_DEFENCE;
    out[7] = (float)item->defence_crush / STAT_NORM_DEFENCE;
    out[8] = (float)item->defence_magic / STAT_NORM_DEFENCE;
    out[9] = (float)item->defence_ranged / STAT_NORM_DEFENCE;

    // Strength bonuses
    out[10] = (float)item->melee_strength / STAT_NORM_STRENGTH;
    out[11] = (float)item->ranged_strength / STAT_NORM_STRENGTH;
    out[12] = (float)item->magic_damage / STAT_NORM_MAGIC_DMG;

    // Other bonuses
    out[13] = (float)item->prayer / STAT_NORM_PRAYER;
    out[14] = (float)item->attack_speed / STAT_NORM_SPEED;
    out[15] = (float)item->attack_range / STAT_NORM_RANGE;

    // Flags
    out[16] = (item->slot == SLOT_WEAPON) ? 1.0f : 0.0f;
    out[17] = 0.0f;  // not empty
}

/**
 * Get item index from slot inventory.
 *
 * Maps GearSlotIndex to EquipmentSlot and looks up item.
 * Returns 255 (ITEM_NONE) if slot doesn't have that item.
 */
static inline uint8_t get_item_for_slot(int gear_slot, int item_idx) {
    if (item_idx < 0 || item_idx >= MAX_ITEMS_PER_SLOT_DB) return ITEM_NONE;

    // Map GearSlotIndex to EquipmentSlot (they're slightly different)
    int eq_slot;
    switch (gear_slot) {
        case 0: eq_slot = SLOT_HEAD; break;    // GEAR_SLOT_HEAD
        case 1: eq_slot = SLOT_CAPE; break;    // GEAR_SLOT_CAPE
        case 2: eq_slot = SLOT_NECK; break;    // GEAR_SLOT_NECK
        case 3: return ITEM_NONE;              // GEAR_SLOT_AMMO (not in LMS)
        case 4: eq_slot = SLOT_WEAPON; break;  // GEAR_SLOT_WEAPON
        case 5: eq_slot = SLOT_SHIELD; break;  // GEAR_SLOT_SHIELD
        case 6: eq_slot = SLOT_BODY; break;    // GEAR_SLOT_BODY
        case 7: eq_slot = SLOT_LEGS; break;    // GEAR_SLOT_LEGS
        case 8: eq_slot = SLOT_HANDS; break;   // GEAR_SLOT_HANDS
        case 9: eq_slot = SLOT_FEET; break;    // GEAR_SLOT_FEET
        case 10: eq_slot = SLOT_RING; break;   // GEAR_SLOT_RING
        default: return ITEM_NONE;
    }

    return ITEMS_BY_SLOT[eq_slot][item_idx];
}

/**
 * Get number of available items for a gear slot.
 */
static inline int get_num_items_for_slot(int gear_slot) {
    int eq_slot;
    switch (gear_slot) {
        case 0: eq_slot = SLOT_HEAD; break;
        case 1: eq_slot = SLOT_CAPE; break;
        case 2: eq_slot = SLOT_NECK; break;
        case 3: return 0;  // AMMO
        case 4: eq_slot = SLOT_WEAPON; break;
        case 5: eq_slot = SLOT_SHIELD; break;
        case 6: eq_slot = SLOT_BODY; break;
        case 7: eq_slot = SLOT_LEGS; break;
        case 8: eq_slot = SLOT_HANDS; break;
        case 9: eq_slot = SLOT_FEET; break;
        case 10: eq_slot = SLOT_RING; break;
        default: return 0;
    }
    return NUM_ITEMS_IN_SLOT[eq_slot];
}

#endif // OSRS_ITEMS_H
