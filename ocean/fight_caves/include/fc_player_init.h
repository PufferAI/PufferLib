#ifndef FC_PLAYER_INIT_H
#define FC_PLAYER_INIT_H

#include <stdint.h>

/*
 * Player skill, equipment, and consumable configuration shared by the core,
 * training adapter, viewer, and asset tooling. The immutable table itself is
 * defined once in fc_loadouts.c.
 */

typedef enum {
    FC_LOADOUT_BLACK_DHIDE_RCB = 0,
    FC_LOADOUT_SOTA_TBOW = 1,
    FC_LOADOUT_LOW_DEF_RCB = 2,
    FC_LOADOUT_RCB_PURE = 3,
    FC_LOADOUT_MSBI_PURE = 4,
    FC_LOADOUT_BLOWPIPE_PURE = 5,
    FC_LOADOUT_ACB_ARMADYL = 6,
    FC_LOADOUT_BOWFA_CRYSTAL = 7,
    FC_LOADOUT_TBOW_MASORI = 8,
    FC_LOADOUT_COUNT
} FcLoadoutId;

#ifndef FC_ACTIVE_LOADOUT
#define FC_ACTIVE_LOADOUT FC_LOADOUT_SOTA_TBOW
#endif

#define FC_LOADOUT_EQUIP_MAX 12
#define FC_LOADOUT_MODEL_ITEM_MAX 12
#define FC_PLAYER_MODEL_BASE 0xFC000000u

typedef enum {
    FC_EQUIP_SLOT_HEAD = 0,
    FC_EQUIP_SLOT_CAPE = 1,
    FC_EQUIP_SLOT_NECK = 2,
    FC_EQUIP_SLOT_WEAPON = 3,
    FC_EQUIP_SLOT_BODY = 4,
    FC_EQUIP_SLOT_SHIELD = 5,
    FC_EQUIP_SLOT_AMMO = 6,
    FC_EQUIP_SLOT_LEGS = 7,
    FC_EQUIP_SLOT_HANDS = 9,
    FC_EQUIP_SLOT_FEET = 10,
    FC_EQUIP_SLOT_RING = 12,
} FcEquipmentSlot;

typedef struct {
    int slot;
    uint32_t item_id;
    uint32_t icon_item_id;
    const char* label;
} FcLoadoutEquipmentItem;

typedef enum {
    FC_CRYSTAL_PIECE_NONE = 0,
    FC_CRYSTAL_PIECE_HELM = 1 << 0,
    FC_CRYSTAL_PIECE_BODY = 1 << 1,
    FC_CRYSTAL_PIECE_LEGS = 1 << 2,
    FC_CRYSTAL_PIECE_ALL = FC_CRYSTAL_PIECE_HELM |
                           FC_CRYSTAL_PIECE_BODY |
                           FC_CRYSTAL_PIECE_LEGS
} FcCrystalPieceMask;

/* Exact per-piece modifiers. One percentage point is 100 basis points. */
#define FC_CRYSTAL_HELM_ACCURACY_BP  500
#define FC_CRYSTAL_HELM_DAMAGE_BP    250
#define FC_CRYSTAL_BODY_ACCURACY_BP 1500
#define FC_CRYSTAL_BODY_DAMAGE_BP    750
#define FC_CRYSTAL_LEGS_ACCURACY_BP 1000
#define FC_CRYSTAL_LEGS_DAMAGE_BP    500

typedef struct {
    const char* name;
    const char* weapon_name;
    uint32_t player_model_id;
    int combat_style_profile;
    int max_hp, max_prayer;
    int attack_lvl, strength_lvl, defence_lvl;
    int ranged_lvl, prayer_lvl, magic_lvl;
    int weapon_kind;
    int weapon_uses_ammo;
    int crystal_piece_mask;
    int weapon_speed;
    int weapon_range;
    int ranged_atk, ranged_str;
    int def_stab, def_slash, def_crush, def_magic, def_ranged;
    int prayer_bonus;
    int ammo;
    int equipment_count;
    FcLoadoutEquipmentItem equipment[FC_LOADOUT_EQUIP_MAX];
    int model_item_count;
    int model_item_ids[FC_LOADOUT_MODEL_ITEM_MAX];
} FcLoadout;

typedef enum {
    FC_WEAPON_GENERIC_RANGED = 0,
    FC_WEAPON_TWISTED_BOW = 1,
    FC_WEAPON_BOW_OF_FAERDHINEN = 2
} FcWeaponKind;

#define FC_NUM_LOADOUTS FC_LOADOUT_COUNT

extern const FcLoadout FC_LOADOUTS[FC_NUM_LOADOUTS];

#define FC_PLAYER_MAX_HP        (FC_LOADOUTS[FC_ACTIVE_LOADOUT].max_hp)
#define FC_PLAYER_MAX_PRAYER    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].max_prayer)
#define FC_PLAYER_DEFENCE_LVL   (FC_LOADOUTS[FC_ACTIVE_LOADOUT].defence_lvl)
#define FC_PLAYER_RANGED_LVL    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].ranged_lvl)
#define FC_PLAYER_PRAYER_LVL    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].prayer_lvl)
#define FC_PLAYER_MAGIC_LVL     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].magic_lvl)
#define FC_PLAYER_WEAPON_USES_AMMO \
    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].weapon_uses_ammo)
#define FC_PLAYER_WEAPON_SPEED  (FC_LOADOUTS[FC_ACTIVE_LOADOUT].weapon_speed)
#define FC_PLAYER_WEAPON_RANGE  (FC_LOADOUTS[FC_ACTIVE_LOADOUT].weapon_range)
#define FC_EQUIP_RANGED_ATK     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].ranged_atk)
#define FC_EQUIP_RANGED_STR     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].ranged_str)
#define FC_EQUIP_DEF_CRUSH      (FC_LOADOUTS[FC_ACTIVE_LOADOUT].def_crush)
#define FC_EQUIP_DEF_MAGIC      (FC_LOADOUTS[FC_ACTIVE_LOADOUT].def_magic)
#define FC_EQUIP_DEF_RANGED     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].def_ranged)

#endif /* FC_PLAYER_INIT_H */
