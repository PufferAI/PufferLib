/**
 * @fileoverview OSRS-style GUI panel system for the debug viewer.
 *
 * Renders inventory, equipment, prayer, combat, and spellbook panels
 * using real sprites exported from the OSRS cache (index 8). Tab bar
 * at the TOP matches the real OSRS fixed-mode client (7 tabs).
 *
 * Sprite sources (exported by scripts/export_sprites_modern.py):
 *   - equipment slot backgrounds: sprite IDs 156-165, 170
 *   - prayer icons (enabled/disabled): sprite IDs 115-154, 502-509, 945-951, 1420-1425
 *   - tab icons: sprite IDs 168, 898, 899, 900, 901, 779, 780
 *   - spell icons: sprite IDs 325-336, 375-386, 557, 561, 564, 607, 611, 614
 *   - special attack bar: sprite ID 657
 *
 * Layout constants derived from OSRS client widget definitions:
 *   - inventory: 4 columns x 7 rows, 36x32 item sprites
 *   - equipment: 11 slots in paperdoll layout (interface 387)
 *   - prayer: 5 columns x 6 rows grid (interface 541)
 *   - combat: 4 attack style buttons + special bar (interface 593)
 *   - spellbook: grid layout (interface 218)
 */

#ifndef OSRS_GUI_H
#define OSRS_GUI_H

#include "osrs_human_input_types.h"

#include "raylib.h"
#include "osrs_types.h"
#include "osrs_items.h"
#include "osrs_pvp_gear.h"


#define GUI_BG_DARK     CLITERAL(Color){ 62, 53, 41, 255 }
#define GUI_BG_MEDIUM   CLITERAL(Color){ 75, 67, 54, 255 }
#define GUI_BG_SLOT     CLITERAL(Color){ 56, 48, 38, 255 }
#define GUI_BG_SLOT_HL  CLITERAL(Color){ 90, 80, 60, 255 }
#define GUI_BORDER      CLITERAL(Color){ 42, 36, 28, 255 }
#define GUI_BORDER_LT   CLITERAL(Color){ 100, 90, 70, 255 }
#define GUI_TEXT_YELLOW  CLITERAL(Color){ 255, 255, 0, 255 }
#define GUI_TEXT_ORANGE  CLITERAL(Color){ 255, 152, 31, 255 }
#define GUI_TEXT_WHITE   CLITERAL(Color){ 255, 255, 255, 255 }
#define GUI_TEXT_GREEN   CLITERAL(Color){ 0, 255, 0, 255 }
#define GUI_TEXT_RED     CLITERAL(Color){ 255, 0, 0, 255 }
#define GUI_TEXT_CYAN    CLITERAL(Color){ 0, 255, 255, 255 }
#define GUI_TAB_ACTIVE   CLITERAL(Color){ 100, 90, 70, 255 }
#define GUI_TAB_INACTIVE CLITERAL(Color){ 50, 44, 35, 255 }
#define GUI_PRAYER_ON   CLITERAL(Color){ 200, 200, 100, 80 }
#define GUI_SPEC_GREEN  CLITERAL(Color){ 0, 180, 0, 255 }
#define GUI_SPEC_DARK   CLITERAL(Color){ 30, 30, 20, 255 }
#define GUI_HP_GREEN    CLITERAL(Color){ 0, 146, 0, 255 }
#define GUI_HP_RED      CLITERAL(Color){ 160, 0, 0, 255 }

/* OSRS text shadow: draw black at (+1,+1) then color on top */
#define GUI_TEXT_SHADOW CLITERAL(Color){ 0, 0, 0, 255 }


typedef enum {
    GUI_TAB_COMBAT = 0,
    GUI_TAB_STATS = 1,       /* empty (no content) */
    GUI_TAB_QUESTS = 2,      /* empty (no content) */
    GUI_TAB_INVENTORY = 3,
    GUI_TAB_EQUIPMENT = 4,
    GUI_TAB_PRAYER = 5,
    GUI_TAB_SPELLBOOK = 6,
    GUI_TAB_COUNT = 7
} GuiTab;


/* slot background sprite IDs from cache index 8:
   head=156, cape=157, neck=158, weapon=159, ring=160,
   body=161, shield=162, legs=163, hands=164, feet=165, tile=170 */
#define GUI_NUM_SLOT_SPRITES 12  /* 11 slots + tile background */


/* prayer icons — authoritative 29-entry standard book.
   enum order IS display order (left→right, top→bottom) in the 5×6 grid.
   sprite IDs match the real OSRS SpriteID.Prayeron / Prayeroff mapping. */
typedef enum {
    GUI_PRAY_THICK_SKIN = 0,      /* row 0: sprite 115 / 135 */
    GUI_PRAY_BURST_STR,           /*        sprite 116 / 136 */
    GUI_PRAY_CLARITY,             /*        sprite 117 / 137 */
    GUI_PRAY_SHARP_EYE,           /*        sprite 133 / 153 */
    GUI_PRAY_MYSTIC_WILL,         /*        sprite 134 / 154 */
    GUI_PRAY_ROCK_SKIN,           /* row 1: sprite 118 / 138 */
    GUI_PRAY_SUPERHUMAN,          /*        sprite 119 / 139 */
    GUI_PRAY_IMPROVED_REFLEX,     /*        sprite 120 / 140 */
    GUI_PRAY_RAPID_RESTORE,       /*        sprite 121 / 141 */
    GUI_PRAY_RAPID_HEAL,          /*        sprite 122 / 142 */
    GUI_PRAY_PROTECT_ITEM,        /* row 2: sprite 123 / 143 */
    GUI_PRAY_HAWK_EYE,            /*        sprite 502 / 506 */
    GUI_PRAY_MYSTIC_LORE,         /*        sprite 503 / 507 */
    GUI_PRAY_STEEL_SKIN,          /*        sprite 124 / 144 */
    GUI_PRAY_ULTIMATE_STR,        /*        sprite 125 / 145 */
    GUI_PRAY_INCREDIBLE_REFLEX,   /* row 3: sprite 126 / 146 */
    GUI_PRAY_PROTECT_MAGIC,       /*        sprite 127 / 147 */
    GUI_PRAY_PROTECT_MISSILES,    /*        sprite 128 / 148 */
    GUI_PRAY_PROTECT_MELEE,       /*        sprite 129 / 149 */
    GUI_PRAY_EAGLE_EYE,           /*        sprite 504 / 508 */
    GUI_PRAY_MYSTIC_MIGHT,        /* row 4: sprite 505 / 509 */
    GUI_PRAY_RETRIBUTION,         /*        sprite 131 / 151 */
    GUI_PRAY_REDEMPTION,          /*        sprite 130 / 150 */
    GUI_PRAY_SMITE,               /*        sprite 132 / 152 */
    GUI_PRAY_PRESERVE,            /*        sprite 947 / 951 */
    GUI_PRAY_CHIVALRY,            /* row 5: sprite 945 / 949 */
    GUI_PRAY_PIETY,               /*        sprite 946 / 950 */
    GUI_PRAY_RIGOUR,              /*        sprite 1420 / 1424 */
    GUI_PRAY_AUGURY,              /*        sprite 1421 / 1425 */
    GUI_NUM_PRAYERS               /* = 29 */
} GuiPrayerIdx;


/* Ancient spellbook sorted by level (Smoke→Shadow→Blood→Ice per row,
   Rush→Burst→Blitz→Barrage per family). Only Ice/Blood/Vengeance are
   castable in this env; Smoke/Shadow render greyed-out for authenticity. */
typedef enum {
    /* row 0: Rush spells (level 50/52/56/58) */
    GUI_SPELL_SMOKE_RUSH = 0,     /* sprite 329 / 379 */
    GUI_SPELL_SHADOW_RUSH,        /* sprite 337 / 387 */
    GUI_SPELL_BLOOD_RUSH,         /* sprite 333 / 383 */
    GUI_SPELL_ICE_RUSH,           /* sprite 325 / 375 */
    /* row 1: Burst spells (62/64/68/70) */
    GUI_SPELL_SMOKE_BURST,        /* sprite 330 / 380 */
    GUI_SPELL_SHADOW_BURST,       /* sprite 338 / 388 */
    GUI_SPELL_BLOOD_BURST,        /* sprite 334 / 384 */
    GUI_SPELL_ICE_BURST,          /* sprite 326 / 376 */
    /* row 2: Blitz spells (74/76/80/82) */
    GUI_SPELL_SMOKE_BLITZ,        /* sprite 331 / 381 */
    GUI_SPELL_SHADOW_BLITZ,       /* sprite 339 / 389 */
    GUI_SPELL_BLOOD_BLITZ,        /* sprite 335 / 385 */
    GUI_SPELL_ICE_BLITZ,          /* sprite 327 / 377 */
    /* row 3: Barrage spells (86/88/92/94) */
    GUI_SPELL_SMOKE_BARRAGE,      /* sprite 332 / 382 */
    GUI_SPELL_SHADOW_BARRAGE,     /* sprite 340 / 390 */
    GUI_SPELL_BLOOD_BARRAGE,      /* sprite 336 / 386 */
    GUI_SPELL_ICE_BARRAGE,        /* sprite 328 / 378 */
    /* lunar spell(s) we use */
    GUI_SPELL_VENGEANCE,          /* sprite 564 */
    GUI_NUM_SPELLS
} GuiSpellIdx;


/* inventory slot types: either an equipment item (ITEM_DATABASE index) or a consumable.
   consumables are tracked as counts in Player, not as individual ITEM_DATABASE entries,
   so we use dedicated types with known OSRS item IDs for sprite lookup. */
typedef enum {
    INV_SLOT_EMPTY = 0,
    INV_SLOT_EQUIPMENT,     /* item_db_idx holds ITEM_DATABASE index */
    INV_SLOT_FOOD,          /* shark (OSRS ID 385) */
    INV_SLOT_KARAMBWAN,     /* cooked karambwan (OSRS ID 3144) */
    INV_SLOT_BREW,          /* saradomin brew (OSRS IDs 6685/6687/6689/6691 for 4/3/2/1 dose) */
    INV_SLOT_RESTORE,       /* super restore (OSRS IDs 3024/3026/3028/3030) */
    INV_SLOT_COMBAT_POT,    /* super combat (OSRS IDs 12695/12697/12699/12701) */
    INV_SLOT_RANGED_POT,    /* ranging potion (OSRS IDs 2444/169/171/173) */
    INV_SLOT_ANTIVENOM,     /* anti-venom+ (OSRS IDs 12913/12915/12917/12919) */
    INV_SLOT_PRAYER_POT,    /* prayer potion (OSRS IDs 2434/139/141/143 for 4/3/2/1 dose) */
    INV_SLOT_BASTION_POT,   /* bastion potion (OSRS IDs 22461/22464/22467/22470) */
    INV_SLOT_STAMINA_POT,   /* stamina potion (OSRS IDs 12625/12627/12629/12631) */
} InvSlotType;

/* OSRS item IDs for consumable sprites (4-dose shown by default) */
#define OSRS_ID_SHARK         385
#define OSRS_ID_KARAMBWAN     3144
#define OSRS_ID_BREW_4        6685
#define OSRS_ID_BREW_3        6687
#define OSRS_ID_BREW_2        6689
#define OSRS_ID_BREW_1        6691
#define OSRS_ID_RESTORE_4     3024
#define OSRS_ID_RESTORE_3     3026
#define OSRS_ID_RESTORE_2     3028
#define OSRS_ID_RESTORE_1     3030
#define OSRS_ID_COMBAT_4      12695
#define OSRS_ID_COMBAT_3      12697
#define OSRS_ID_COMBAT_2      12699
#define OSRS_ID_COMBAT_1      12701
#define OSRS_ID_RANGED_4      2444
#define OSRS_ID_RANGED_3      169
#define OSRS_ID_RANGED_2      171
#define OSRS_ID_RANGED_1      173
#define OSRS_ID_ANTIVENOM_4   12913
#define OSRS_ID_ANTIVENOM_3   12915
#define OSRS_ID_ANTIVENOM_2   12917
#define OSRS_ID_ANTIVENOM_1   12919
#define OSRS_ID_PRAYER_POT_4  2434
#define OSRS_ID_PRAYER_POT_3  139
#define OSRS_ID_PRAYER_POT_2  141
#define OSRS_ID_PRAYER_POT_1  143
#define OSRS_ID_BASTION_4     22461
#define OSRS_ID_BASTION_3     22464
#define OSRS_ID_BASTION_2     22467
#define OSRS_ID_BASTION_1     22470
#define OSRS_ID_STAMINA_4     12625
#define OSRS_ID_STAMINA_3     12627
#define OSRS_ID_STAMINA_2     12629
#define OSRS_ID_STAMINA_1     12631

#define INV_GRID_SLOTS 28  /* 4 columns x 7 rows */

typedef struct {
    InvSlotType type;
    uint8_t     item_db_idx;   /* ITEM_DATABASE index (for INV_SLOT_EQUIPMENT) */
    int         osrs_id;       /* OSRS item ID (for sprite lookup, all types) */
} InvSlot;

/* click/drag interaction state */
#define INV_DIM_TICKS 15       /* client ticks (50 Hz) to show dim after click */
#define INV_DRAG_DEAD_ZONE 5   /* pixels before drag activates */

typedef enum {
    INV_ACTION_NONE = 0,
    INV_ACTION_EQUIP,
    INV_ACTION_EAT,
    INV_ACTION_DRINK,
} InvAction;


typedef struct {
    GuiTab active_tab;
    int panel_x, panel_y;
    int panel_w, panel_h;
    int tab_h;
    int status_bar_h;    /* compact HP/prayer/spec bar height */

    /* multi-entity cycling (G key) */
    int gui_entity_idx;
    int gui_entity_count;

    /* encounter state (for boss info display below panel) */
    void* encounter_state;
    const void* encounter_def;

    /* textures loaded from exported cache sprites */
    int sprites_loaded;

    /* equipment slot background sprites (indexed by GEAR_SLOT_*) */
    Texture2D slot_sprites[GUI_NUM_SLOT_SPRITES];
    Texture2D slot_tile_bg;   /* sprite 170: tile/background */

    /* tab icons: 7 tabs (combat, stats, quests, inventory, equipment, prayer, spellbook) */
    Texture2D tab_icons[GUI_TAB_COUNT];

    /* prayer icons: enabled and disabled variants */
    Texture2D prayer_on[GUI_NUM_PRAYERS];
    Texture2D prayer_off[GUI_NUM_PRAYERS];

    /* spell icons: enabled and disabled variants */
    Texture2D spell_on[GUI_NUM_SPELLS];
    Texture2D spell_off[GUI_NUM_SPELLS];

    /* special attack bar sprite */
    Texture2D spec_bar;
    int spec_bar_loaded;

    /* interface chrome sprites */
    Texture2D side_panel_bg;       /* 1031: stone background tile */
    Texture2D tabs_row_bottom;     /* 1032: bottom tab row strip */
    Texture2D tabs_row_top;        /* 1036: top tab row strip */
    Texture2D tab_stone_sel[5];    /* 1026-1030: selected tab corners + middle */
    Texture2D slanted_tab;         /* 952: inactive tab button */
    Texture2D slanted_tab_hover;   /* 953: hovered tab button */
    Texture2D slot_tile;           /* 170: equipment slot background */
    Texture2D slot_selected;       /* 179: equipment slot selected */
    Texture2D orb_frame;           /* 1071: minimap orb frame */
    int chrome_loaded;

    /* skill icons for stats tab (25x25 from RuneLite skill_icons) */
    #define GUI_NUM_SKILL_ICONS 7
    Texture2D skill_icons[7];  /* attack, strength, defence, ranged, prayer, magic, hitpoints */
    int skill_icons_loaded;

    /* item sprites: keyed by OSRS item ID (from data/sprites/items/{id}.png) */
    #define GUI_MAX_ITEM_SPRITES 256
    int item_sprite_ids[GUI_MAX_ITEM_SPRITES];     /* OSRS item ID, 0 = empty */
    Texture2D item_sprite_tex[GUI_MAX_ITEM_SPRITES]; /* corresponding texture */
    int item_sprite_count;

    /* inventory grid: 28 slots (4x7). initialized once at reset, then updated
       incrementally — items stay in their assigned slots (no compaction on eat).
       positions are user-rearrangeable via drag-and-drop. */
    InvSlot inv_grid[INV_GRID_SLOTS];
    int inv_grid_dirty;   /* 1 = needs full rebuild from player state */

    /* previous player state for incremental inventory updates.
       compared each tick to detect gear switches and consumable use. */
    uint8_t inv_prev_equipped[NUM_GEAR_SLOTS];
    int inv_prev_food_count;
    int inv_prev_karambwan_count;
    int inv_prev_brew_doses;
    int inv_prev_restore_doses;
    int inv_prev_prayer_pot_doses;
    int inv_prev_combat_doses;
    int inv_prev_ranged_doses;
    int inv_prev_bastion_doses;
    int inv_prev_stamina_doses;
    int inv_prev_antivenom_doses;

    /* human-clicked inventory slot: when a human clicks a consumable, this records
       the exact slot so gui_update_inventory removes from that slot instead of the
       last one. -1 = no human click pending, use default last-slot removal. */
    int human_clicked_inv_slot;

    /* click dim animation: slot index and countdown (50 Hz client ticks) */
    int inv_dim_slot;     /* -1 = none */
    int inv_dim_timer;    /* counts down from INV_DIM_TICKS */

    /* drag state */
    int inv_drag_active;       /* 1 = currently dragging */
    int inv_drag_src_slot;     /* slot being dragged */
    int inv_drag_start_x;     /* mouse position at drag start */
    int inv_drag_start_y;
    int inv_drag_mouse_x;     /* current mouse position during drag */
    int inv_drag_mouse_y;

    /* spell targeting: GuiSpellIdx of the spell awaiting an enemy click, or
       -1 when not targeting. render code sets this before calling gui_draw. */
    int pending_spell_highlight;
} GuiState;


/** Try loading a texture, returns 1 on success. */
static int gui_try_load(Texture2D* tex, const char* path) {
    if (FileExists(path)) {
        *tex = LoadTexture(path);
        return 1;
    }
    return 0;
}

/** Load all GUI sprites from data/sprites/gui/. */
static void gui_load_sprites(GuiState* gs) {
    gs->sprites_loaded = 1;
    int ok = 1;

    /* equipment slot backgrounds: sprite IDs mapped to GEAR_SLOT_* order.
       GEAR_SLOT: HEAD=0, CAPE=1, NECK=2, AMMO=3, WEAPON=4, SHIELD=5,
                  BODY=6, LEGS=7, HANDS=8, FEET=9, RING=10 */
    static const char* slot_files[] = {
        "data/sprites/gui/slot_head.png",    /* GEAR_SLOT_HEAD */
        "data/sprites/gui/slot_cape.png",    /* GEAR_SLOT_CAPE */
        "data/sprites/gui/slot_neck.png",    /* GEAR_SLOT_NECK */
        "data/sprites/gui/slot_tile.png",    /* GEAR_SLOT_AMMO (use tile bg) */
        "data/sprites/gui/slot_weapon.png",  /* GEAR_SLOT_WEAPON */
        "data/sprites/gui/slot_shield.png",  /* GEAR_SLOT_SHIELD */
        "data/sprites/gui/slot_body.png",    /* GEAR_SLOT_BODY */
        "data/sprites/gui/slot_legs.png",    /* GEAR_SLOT_LEGS */
        "data/sprites/gui/slot_hands.png",   /* GEAR_SLOT_HANDS */
        "data/sprites/gui/slot_feet.png",    /* GEAR_SLOT_FEET */
        "data/sprites/gui/slot_ring.png",    /* GEAR_SLOT_RING */
        "data/sprites/gui/slot_tile.png",    /* spare tile bg */
    };
    for (int i = 0; i < GUI_NUM_SLOT_SPRITES; i++) {
        ok &= gui_try_load(&gs->slot_sprites[i], slot_files[i]);
    }
    gui_try_load(&gs->slot_tile_bg, "data/sprites/gui/slot_tile.png");

    /* tab icons: mapped to GuiTab enum order (7 tabs) */
    static const char* tab_files[] = {
        "data/sprites/gui/tab_combat.png",    /* GUI_TAB_COMBAT */
        "data/sprites/gui/tab_stats.png",     /* GUI_TAB_STATS */
        "data/sprites/gui/tab_quests.png",    /* GUI_TAB_QUESTS */
        "data/sprites/gui/tab_inventory.png", /* GUI_TAB_INVENTORY */
        "data/sprites/gui/tab_equipment.png", /* GUI_TAB_EQUIPMENT */
        "data/sprites/gui/tab_prayer.png",    /* GUI_TAB_PRAYER */
        "data/sprites/gui/tab_magic.png",     /* GUI_TAB_SPELLBOOK */
    };
    for (int i = 0; i < GUI_TAB_COUNT; i++) {
        ok &= gui_try_load(&gs->tab_icons[i], tab_files[i]);
    }

    /* skill icons for stats tab (OSRS skill_icons from RuneLite resources) */
    static const char* skill_icon_files[] = {
        "data/sprites/gui/skill_attack.png",
        "data/sprites/gui/skill_strength.png",
        "data/sprites/gui/skill_defence.png",
        "data/sprites/gui/skill_ranged.png",
        "data/sprites/gui/skill_prayer.png",
        "data/sprites/gui/skill_magic.png",
        "data/sprites/gui/skill_hitpoints.png",
    };
    gs->skill_icons_loaded = 1;
    for (int i = 0; i < 7; i++) {
        gs->skill_icons_loaded &= gui_try_load(&gs->skill_icons[i], skill_icon_files[i]);
    }

    /* prayer icons: indexed by GuiPrayerIdx, sprite IDs match real OSRS. */
    static const int pray_on_ids[GUI_NUM_PRAYERS] = {
        115, 116, 117, 133, 134,   /* row 0: ThickSkin, Burst, Clarity, SharpEye, MysticWill */
        118, 119, 120, 121, 122,   /* row 1: RockSkin, Superhuman, ImprovedReflex, RapidRestore, RapidHeal */
        123, 502, 503, 124, 125,   /* row 2: ProtectItem, HawkEye, MysticLore, SteelSkin, UltimateStr */
        126, 127, 128, 129, 504,   /* row 3: IncredibleReflex, ProtMagic, ProtMissiles, ProtMelee, EagleEye */
        505, 131, 130, 132, 947,   /* row 4: MysticMight, Retribution, Redemption, Smite, Preserve */
        945, 946, 1420, 1421,      /* row 5: Chivalry, Piety, Rigour, Augury (1 empty cell) */
    };
    static const int pray_off_ids[GUI_NUM_PRAYERS] = {
        135, 136, 137, 153, 154,
        138, 139, 140, 141, 142,
        143, 506, 507, 144, 145,
        146, 147, 148, 149, 508,
        509, 151, 150, 152, 951,
        949, 950, 1424, 1425,
    };
    for (int i = 0; i < GUI_NUM_PRAYERS; i++) {
        const char* on_path = TextFormat("data/sprites/gui/%d.png", pray_on_ids[i]);
        const char* off_path = TextFormat("data/sprites/gui/%d.png", pray_off_ids[i]);
        gui_try_load(&gs->prayer_on[i], on_path);
        gui_try_load(&gs->prayer_off[i], off_path);
    }

    /* spell icons — indexed by GuiSpellIdx. full ancient book + vengeance. */
    static const int spell_on_ids[GUI_NUM_SPELLS] = {
        329, 337, 333, 325,   /* Rush:    Smoke, Shadow, Blood, Ice */
        330, 338, 334, 326,   /* Burst:   Smoke, Shadow, Blood, Ice */
        331, 339, 335, 327,   /* Blitz:   Smoke, Shadow, Blood, Ice */
        332, 340, 336, 328,   /* Barrage: Smoke, Shadow, Blood, Ice */
        564,                  /* Vengeance */
    };
    static const int spell_off_ids[GUI_NUM_SPELLS] = {
        379, 387, 383, 375,
        380, 388, 384, 376,
        381, 389, 385, 377,
        382, 390, 386, 378,
        614,
    };
    for (int i = 0; i < GUI_NUM_SPELLS; i++) {
        const char* on_path = TextFormat("data/sprites/gui/%d.png", spell_on_ids[i]);
        const char* off_path = TextFormat("data/sprites/gui/%d.png", spell_off_ids[i]);
        gui_try_load(&gs->spell_on[i], on_path);
        gui_try_load(&gs->spell_off[i], off_path);
    }

    /* special attack bar */
    gs->spec_bar_loaded = gui_try_load(&gs->spec_bar, "data/sprites/gui/special_attack.png");

    /* interface chrome */
    gs->chrome_loaded = 1;
    gs->chrome_loaded &= gui_try_load(&gs->side_panel_bg, "data/sprites/gui/side_panel_bg.png");
    gs->chrome_loaded &= gui_try_load(&gs->tabs_row_bottom, "data/sprites/gui/tabs_row_bottom.png");
    gs->chrome_loaded &= gui_try_load(&gs->tabs_row_top, "data/sprites/gui/tabs_row_top.png");
    gui_try_load(&gs->slanted_tab, "data/sprites/gui/slanted_tab.png");
    gui_try_load(&gs->slanted_tab_hover, "data/sprites/gui/slanted_tab_hover.png");
    gui_try_load(&gs->slot_tile, "data/sprites/gui/slot_tile.png");
    gui_try_load(&gs->slot_selected, "data/sprites/gui/slot_selected.png");
    gui_try_load(&gs->orb_frame, "data/sprites/gui/orb_frame.png");

    static const char* tab_sel_files[] = {
        "data/sprites/gui/tab_stone_tl_sel.png",
        "data/sprites/gui/tab_stone_tr_sel.png",
        "data/sprites/gui/tab_stone_bl_sel.png",
        "data/sprites/gui/tab_stone_br_sel.png",
        "data/sprites/gui/tab_stone_mid_sel.png",
    };
    for (int i = 0; i < 5; i++) gui_try_load(&gs->tab_stone_sel[i], tab_sel_files[i]);

    if (!ok) {
        TraceLog(LOG_WARNING, "GUI: some sprites missing from data/sprites/gui/");
    }

    /* load item sprites from data/sprites/items/{item_id}.png */
    gs->item_sprite_count = 0;
    for (int i = 0; i < NUM_ITEMS && gs->item_sprite_count < GUI_MAX_ITEM_SPRITES; i++) {
        int item_id = ITEM_DATABASE[i].item_id;
        if (item_id <= 0) continue;
        const char* path = TextFormat("data/sprites/items/%d.png", item_id);
        if (FileExists(path)) {
            int idx = gs->item_sprite_count;
            gs->item_sprite_ids[idx] = item_id;
            gs->item_sprite_tex[idx] = LoadTexture(path);
            gs->item_sprite_count++;
        }
    }

    /* consumable sprites: not in ITEM_DATABASE, load by OSRS item ID directly */
    static const int consumable_ids[] = {
        OSRS_ID_SHARK, OSRS_ID_KARAMBWAN,
        OSRS_ID_BREW_4, OSRS_ID_BREW_3, OSRS_ID_BREW_2, OSRS_ID_BREW_1,
        OSRS_ID_RESTORE_4, OSRS_ID_RESTORE_3, OSRS_ID_RESTORE_2, OSRS_ID_RESTORE_1,
        OSRS_ID_COMBAT_4, OSRS_ID_COMBAT_3, OSRS_ID_COMBAT_2, OSRS_ID_COMBAT_1,
        OSRS_ID_RANGED_4, OSRS_ID_RANGED_3, OSRS_ID_RANGED_2, OSRS_ID_RANGED_1,
        OSRS_ID_ANTIVENOM_4, OSRS_ID_ANTIVENOM_3, OSRS_ID_ANTIVENOM_2, OSRS_ID_ANTIVENOM_1,
        OSRS_ID_PRAYER_POT_4, OSRS_ID_PRAYER_POT_3, OSRS_ID_PRAYER_POT_2, OSRS_ID_PRAYER_POT_1,
        OSRS_ID_BASTION_4, OSRS_ID_BASTION_3, OSRS_ID_BASTION_2, OSRS_ID_BASTION_1,
        OSRS_ID_STAMINA_4, OSRS_ID_STAMINA_3, OSRS_ID_STAMINA_2, OSRS_ID_STAMINA_1,
    };
    for (int i = 0; i < (int)(sizeof(consumable_ids)/sizeof(consumable_ids[0])); i++) {
        if (gs->item_sprite_count >= GUI_MAX_ITEM_SPRITES) break;
        int cid = consumable_ids[i];
        const char* path = TextFormat("data/sprites/items/%d.png", cid);
        if (FileExists(path)) {
            int idx = gs->item_sprite_count;
            gs->item_sprite_ids[idx] = cid;
            gs->item_sprite_tex[idx] = LoadTexture(path);
            gs->item_sprite_count++;
        }
    }
    TraceLog(LOG_INFO, "GUI: loaded %d item sprites (incl consumables)", gs->item_sprite_count);
}

/** Look up item sprite texture by item database index. Returns NULL texture (id=0) if not found. */
static Texture2D gui_get_item_sprite(GuiState* gs, uint8_t item_idx) {
    Texture2D empty = { 0 };
    if (item_idx == ITEM_NONE || item_idx >= NUM_ITEMS) return empty;
    int item_id = ITEM_DATABASE[item_idx].item_id;
    for (int i = 0; i < gs->item_sprite_count; i++) {
        if (gs->item_sprite_ids[i] == item_id) return gs->item_sprite_tex[i];
    }
    return empty;
}

/** Look up item sprite texture by OSRS item ID directly (for consumables). */
static Texture2D gui_get_sprite_by_osrs_id(GuiState* gs, int osrs_id) {
    Texture2D empty = { 0 };
    if (osrs_id <= 0) return empty;
    for (int i = 0; i < gs->item_sprite_count; i++) {
        if (gs->item_sprite_ids[i] == osrs_id) return gs->item_sprite_tex[i];
    }
    return empty;
}

/** Unload all GUI textures. */
static void gui_unload_sprites(GuiState* gs) {
    if (!gs->sprites_loaded) return;
    for (int i = 0; i < GUI_NUM_SLOT_SPRITES; i++) UnloadTexture(gs->slot_sprites[i]);
    UnloadTexture(gs->slot_tile_bg);
    for (int i = 0; i < GUI_TAB_COUNT; i++) UnloadTexture(gs->tab_icons[i]);
    for (int i = 0; i < GUI_NUM_PRAYERS; i++) {
        UnloadTexture(gs->prayer_on[i]);
        UnloadTexture(gs->prayer_off[i]);
    }
    for (int i = 0; i < GUI_NUM_SPELLS; i++) {
        UnloadTexture(gs->spell_on[i]);
        UnloadTexture(gs->spell_off[i]);
    }
    if (gs->spec_bar_loaded) UnloadTexture(gs->spec_bar);
    if (gs->chrome_loaded) {
        UnloadTexture(gs->side_panel_bg);
        UnloadTexture(gs->tabs_row_bottom);
        UnloadTexture(gs->tabs_row_top);
        for (int i = 0; i < 5; i++) UnloadTexture(gs->tab_stone_sel[i]);
    }
    if (gs->slanted_tab.id) UnloadTexture(gs->slanted_tab);
    if (gs->slanted_tab_hover.id) UnloadTexture(gs->slanted_tab_hover);
    if (gs->slot_tile.id) UnloadTexture(gs->slot_tile);
    if (gs->slot_selected.id) UnloadTexture(gs->slot_selected);
    if (gs->orb_frame.id) UnloadTexture(gs->orb_frame);
    for (int i = 0; i < gs->item_sprite_count; i++) UnloadTexture(gs->item_sprite_tex[i]);
    gs->item_sprite_count = 0;
    gs->sprites_loaded = 0;
}


static const char* gui_item_short_name(uint8_t item_idx) {
    if (item_idx == ITEM_NONE || item_idx >= NUM_ITEMS) return "";
    const char* full = ITEM_DATABASE[item_idx].name;
    switch (item_idx) {
        case ITEM_HELM_NEITIZNOT:    return "Neit helm";
        case ITEM_GOD_CAPE:          return "God cape";
        case ITEM_GLORY:             return "Glory";
        case ITEM_BLACK_DHIDE_BODY:  return "Dhide body";
        case ITEM_MYSTIC_TOP:        return "Mystic top";
        case ITEM_RUNE_PLATELEGS:    return "Rune legs";
        case ITEM_MYSTIC_BOTTOM:     return "Mystic bot";
        case ITEM_WHIP:              return "Whip";
        case ITEM_RUNE_CROSSBOW:     return "Rune cbow";
        case ITEM_AHRIM_STAFF:       return "Ahrim stf";
        case ITEM_DRAGON_DAGGER:     return "DDS";
        case ITEM_DRAGON_DEFENDER:   return "D defender";
        case ITEM_SPIRIT_SHIELD:     return "Spirit sh";
        case ITEM_BARROWS_GLOVES:    return "B gloves";
        case ITEM_CLIMBING_BOOTS:    return "Climb boot";
        case ITEM_BERSERKER_RING:    return "B ring";
        case ITEM_DIAMOND_BOLTS_E:   return "D bolts(e)";
        case ITEM_GHRAZI_RAPIER:     return "Rapier";
        case ITEM_INQUISITORS_MACE:  return "Inq mace";
        case ITEM_STAFF_OF_DEAD:     return "SOTD";
        case ITEM_KODAI_WAND:        return "Kodai";
        case ITEM_VOLATILE_STAFF:    return "Volatile";
        case ITEM_ZURIELS_STAFF:     return "Zuriel stf";
        case ITEM_ARMADYL_CROSSBOW:  return "ACB";
        case ITEM_ZARYTE_CROSSBOW:   return "ZCB";
        case ITEM_DRAGON_CLAWS:      return "D claws";
        case ITEM_AGS:               return "AGS";
        case ITEM_ANCIENT_GS:        return "Anc GS";
        case ITEM_GRANITE_MAUL:      return "G maul";
        case ITEM_ELDER_MAUL:        return "Elder maul";
        case ITEM_DARK_BOW:          return "Dark bow";
        case ITEM_HEAVY_BALLISTA:    return "Ballista";
        case ITEM_VESTAS:            return "Vesta's";
        case ITEM_VOIDWAKER:         return "Voidwaker";
        case ITEM_STATIUS_WARHAMMER: return "SWH";
        case ITEM_MORRIGANS_JAVELIN: return "Morr jav";
        case ITEM_ANCESTRAL_HAT:     return "Anc hat";
        case ITEM_ANCESTRAL_TOP:     return "Anc top";
        case ITEM_ANCESTRAL_BOTTOM:  return "Anc bot";
        case ITEM_AHRIMS_ROBETOP:    return "Ahrim top";
        case ITEM_AHRIMS_ROBESKIRT:  return "Ahrim skrt";
        case ITEM_KARILS_TOP:        return "Karil top";
        case ITEM_BANDOS_TASSETS:    return "Tassets";
        case ITEM_BLESSED_SPIRIT_SHIELD: return "BSS";
        case ITEM_FURY:              return "Fury";
        case ITEM_OCCULT_NECKLACE:   return "Occult";
        case ITEM_INFERNAL_CAPE:     return "Infernal";
        case ITEM_ETERNAL_BOOTS:     return "Eternal";
        case ITEM_SEERS_RING_I:      return "Seers (i)";
        case ITEM_LIGHTBEARER:       return "Lightbear";
        case ITEM_MAGES_BOOK:        return "Mage book";
        case ITEM_DRAGON_ARROWS:     return "D arrows";
        case ITEM_TORAGS_PLATELEGS:  return "Torag legs";
        case ITEM_DHAROKS_PLATELEGS: return "DH legs";
        case ITEM_VERACS_PLATESKIRT: return "Verac skrt";
        case ITEM_TORAGS_HELM:       return "Torag helm";
        case ITEM_DHAROKS_HELM:      return "DH helm";
        case ITEM_VERACS_HELM:       return "Verac helm";
        case ITEM_GUTHANS_HELM:      return "Guth helm";
        case ITEM_OPAL_DRAGON_BOLTS: return "Opal bolt";
        case ITEM_IMBUED_SARA_CAPE:  return "Sara cape";
        case ITEM_EYE_OF_AYAK:       return "Eye Ayak";
        case ITEM_ELIDINIS_WARD_F:   return "Eld ward";
        case ITEM_CONFLICTION_GAUNTLETS: return "Confl gnt";
        case ITEM_AVERNIC_TREADS:    return "Avernic bt";
        case ITEM_RING_OF_SUFFERING_RI: return "Suff (ri)";
        case ITEM_TWISTED_BOW:       return "T bow";
        case ITEM_ELYSIAN_SPIRIT_SHIELD: return "Elysian";
        case ITEM_MASORI_MASK_F:     return "Masori msk";
        case ITEM_MASORI_BODY_F:     return "Masori bod";
        case ITEM_MASORI_CHAPS_F:    return "Masori chp";
        case ITEM_NECKLACE_OF_ANGUISH: return "Anguish";
        case ITEM_DIZANAS_QUIVER:    return "Dizana qvr";
        case ITEM_ZARYTE_VAMBRACES:  return "Zaryte vam";
        case ITEM_TOXIC_BLOWPIPE:    return "Blowpipe";
        case ITEM_AHRIMS_HOOD:       return "Ahrim hood";
        case ITEM_TORMENTED_BRACELET: return "Tormented";
        case ITEM_SANGUINESTI_STAFF: return "Sang staff";
        case ITEM_INFINITY_BOOTS:    return "Inf boots";
        case ITEM_GOD_BLESSING:      return "Blessing";
        case ITEM_RING_OF_RECOIL:    return "Recoil";
        case ITEM_VENATOR_RING:      return "Venator";
        case ITEM_VIRTUS_MASK:       return "Virtus msk";
        case ITEM_VIRTUS_ROBE_TOP:   return "Virtus top";
        case ITEM_VIRTUS_ROBE_BOTTOM:return "Virtus bot";
        case ITEM_CRYSTAL_HELM:      return "Crystal hm";
        case ITEM_AVAS_ASSEMBLER:    return "Assembler";
        case ITEM_CRYSTAL_BODY:      return "Crystal bd";
        case ITEM_CRYSTAL_LEGS:      return "Crystal lg";
        case ITEM_BOW_OF_FAERDHINEN: return "Fbow";
        case ITEM_BLESSED_DHIDE_BOOTS: return "Bless boot";
        case ITEM_MYSTIC_HAT:        return "Mystic hat";
        case ITEM_TRIDENT_OF_SWAMP:  return "Trident";
        case ITEM_BOOK_OF_DARKNESS:  return "Book dark";
        case ITEM_AMETHYST_ARROW:    return "Ameth arw";
        case ITEM_MYSTIC_BOOTS:      return "Myst boots";
        case ITEM_BLESSED_COIF:      return "Bless coif";
        case ITEM_BLACK_DHIDE_CHAPS: return "Dhide chap";
        case ITEM_MAGIC_SHORTBOW_I:  return "MSB (i)";
        case ITEM_AVAS_ACCUMULATOR:  return "Accumulate";
        default: return full;
    }
}


/** Draw text with OSRS-style shadow (black at +1,+1, then color). */
static void gui_text_shadow(const char* text, int x, int y, int size, Color color) {
    DrawText(text, x + 1, y + 1, size, GUI_TEXT_SHADOW);
    DrawText(text, x, y, size, color);
}

/** Draw an OSRS-style beveled slot rectangle. */
static void gui_draw_slot(int x, int y, int w, int h, Color fill) {
    DrawRectangle(x, y, w, h, fill);
    DrawRectangleLines(x, y, w, h, GUI_BORDER);
    DrawLine(x + 1, y + 1, x + w - 2, y + 1, GUI_BORDER_LT);
    DrawLine(x + 1, y + 1, x + 1, y + h - 2, GUI_BORDER_LT);
}

/** Draw texture centered within a box, scaled to fit. */
static void gui_draw_tex_centered(Texture2D tex, int bx, int by, int bw, int bh) {
    if (tex.id == 0) return;
    /* scale to fit while maintaining aspect ratio */
    float sx = (float)(bw - 4) / (float)tex.width;
    float sy = (float)(bh - 4) / (float)tex.height;
    float s = (sx < sy) ? sx : sy;
    int dw = (int)(tex.width * s);
    int dh = (int)(tex.height * s);
    int dx = bx + (bw - dw) / 2;
    int dy = by + (bh - dh) / 2;
    DrawTextureEx(tex, (Vector2){ (float)dx, (float)dy }, 0.0f, s, WHITE);
}

/** Draw an equipment slot using real OSRS slot tile sprite + item/silhouette sprite. */
static void gui_draw_equip_slot(GuiState* gs, int x, int y, int w, int h,
                                int gear_slot, uint8_t item_idx) {
    /* draw slot_tile (real OSRS 36x36 stone square) as background */
    if (gs->slot_tile.id != 0) {
        Rectangle src = { 0, 0, (float)gs->slot_tile.width, (float)gs->slot_tile.height };
        Rectangle dst = { (float)x, (float)y, (float)w, (float)h };
        DrawTexturePro(gs->slot_tile, src, dst, (Vector2){0,0}, 0.0f, WHITE);
    } else {
        gui_draw_slot(x, y, w, h, GUI_BG_SLOT);
    }

    /* draw item sprite if equipped, else slot silhouette */
    if (item_idx != ITEM_NONE && item_idx < NUM_ITEMS) {
        Texture2D item_tex = gui_get_item_sprite(gs, item_idx);
        if (item_tex.id != 0) {
            gui_draw_tex_centered(item_tex, x, y, w, h);
        } else {
            const char* name = gui_item_short_name(item_idx);
            gui_text_shadow(name, x + 2, y + h / 2 - 4, 7, GUI_TEXT_YELLOW);
        }
    } else if (gs->sprites_loaded && gear_slot >= 0 && gear_slot < GUI_NUM_SLOT_SPRITES) {
        Texture2D bg = gs->slot_sprites[gear_slot];
        if (bg.id != 0) {
            gui_draw_tex_centered(bg, x, y, w, h);
        }
    }
}


static void gui_draw_status_bar(GuiState* gs, Player* p) {
    int sx = gs->panel_x + 6;
    int sy = gs->panel_y;
    int bar_w = gs->panel_w - 12;
    int bar_h = 12;
    int gap = 2;

    /* HP bar */
    float hp_pct = (p->base_hitpoints > 0) ?
        (float)p->current_hitpoints / (float)p->base_hitpoints : 0.0f;
    DrawRectangle(sx, sy, bar_w, bar_h, GUI_HP_RED);
    DrawRectangle(sx, sy, (int)(bar_w * hp_pct), bar_h, GUI_HP_GREEN);
    DrawRectangleLines(sx, sy, bar_w, bar_h, GUI_BORDER);
    gui_text_shadow(TextFormat("HP %d/%d", p->current_hitpoints, p->base_hitpoints),
                    sx + 4, sy + 1, 8, GUI_TEXT_WHITE);
    sy += bar_h + gap;

    /* prayer bar */
    float pray_pct = (p->base_prayer > 0) ?
        (float)p->current_prayer / (float)p->base_prayer : 0.0f;
    DrawRectangle(sx, sy, bar_w, bar_h, GUI_SPEC_DARK);
    DrawRectangle(sx, sy, (int)(bar_w * pray_pct), bar_h, GUI_TEXT_CYAN);
    DrawRectangleLines(sx, sy, bar_w, bar_h, GUI_BORDER);
    gui_text_shadow(TextFormat("Pray %d/%d", p->current_prayer, p->base_prayer),
                    sx + 4, sy + 1, 8, GUI_TEXT_WHITE);
    sy += bar_h + gap;

    /* spec bar */
    float spec_pct = (float)p->special_energy / 100.0f;
    DrawRectangle(sx, sy, bar_w, bar_h, GUI_SPEC_DARK);
    DrawRectangle(sx, sy, (int)(bar_w * spec_pct), bar_h, GUI_SPEC_GREEN);
    DrawRectangleLines(sx, sy, bar_w, bar_h, GUI_BORDER);
    gui_text_shadow(TextFormat("Spec %d%%", p->special_energy),
                    sx + 4, sy + 1, 8, GUI_TEXT_WHITE);
}


static void gui_draw_tab_bar(GuiState* gs) {
    /* tabs drawn at top: right after the status bar */
    int ty = gs->panel_y + gs->status_bar_h;

    /* draw the tab row background strip (real OSRS asset) */
    if (gs->chrome_loaded && gs->tabs_row_top.id != 0) {
        Rectangle src = { 0, 0, (float)gs->tabs_row_top.width, (float)gs->tabs_row_top.height };
        Rectangle dst = { (float)gs->panel_x, (float)ty, (float)gs->panel_w, (float)gs->tab_h };
        DrawTexturePro(gs->tabs_row_top, src, dst, (Vector2){0,0}, 0.0f, WHITE);
    } else {
        DrawRectangle(gs->panel_x, ty, gs->panel_w, gs->tab_h, GUI_TAB_INACTIVE);
    }

    /* draw individual tabs using slanted tab sprites */
    int tab_w = gs->panel_w / GUI_TAB_COUNT;
    for (int i = 0; i < GUI_TAB_COUNT; i++) {
        int tx = gs->panel_x + i * tab_w;
        int is_active = (i == (int)gs->active_tab);

        /* active tab: draw selected tab stone pieces */
        if (is_active && gs->tab_stone_sel[4].id != 0) {
            /* use middle selected piece stretched to fill tab area */
            Rectangle src = { 0, 0, (float)gs->tab_stone_sel[4].width, (float)gs->tab_stone_sel[4].height };
            Rectangle dst = { (float)tx, (float)ty, (float)tab_w, (float)gs->tab_h };
            DrawTexturePro(gs->tab_stone_sel[4], src, dst, (Vector2){0,0}, 0.0f, WHITE);
        }

        /* draw tab icon sprite centered in tab */
        if (gs->sprites_loaded && gs->tab_icons[i].id != 0) {
            Color tint = is_active ? WHITE : CLITERAL(Color){ 160, 160, 160, 255 };
            Texture2D tex = gs->tab_icons[i];
            float ssx = (float)(tab_w - 8) / (float)tex.width;
            float ssy = (float)(gs->tab_h - 6) / (float)tex.height;
            float s = (ssx < ssy) ? ssx : ssy;
            int dw = (int)(tex.width * s);
            int dh = (int)(tex.height * s);
            int dx = tx + (tab_w - dw) / 2;
            int dy = ty + (gs->tab_h - dh) / 2;
            DrawTextureEx(tex, (Vector2){ (float)dx, (float)dy }, 0.0f, s, tint);
        }
    }
}

static int gui_handle_tab_click(GuiState* gs, int mouse_x, int mouse_y) {
    /* tabs are at top, right after status bar */
    int ty = gs->panel_y + gs->status_bar_h;
    if (mouse_y < ty || mouse_y > ty + gs->tab_h) return 0;
    if (mouse_x < gs->panel_x || mouse_x >= gs->panel_x + gs->panel_w) return 0;

    int tab_w = gs->panel_w / GUI_TAB_COUNT;
    int idx = (mouse_x - gs->panel_x) / tab_w;
    if (idx >= 0 && idx < GUI_TAB_COUNT) {
        gs->active_tab = (GuiTab)idx;
        return 1;
    }
    return 0;
}


static int gui_content_y(GuiState* gs) {
    return gs->panel_y + gs->status_bar_h + gs->tab_h;
}


/* inventory grid dimensions — scaled to fill the 320px panel width.
   OSRS native: 42x36 cell pitch, 36x32 sprites. we scale ~1.81x so
   the 4-column grid (304px) fills the panel with 8px padding each side. */
#define INV_COLS 4
#define INV_ROWS 7
/* OSRS native inventory cell pitch is ~42x36, scaled 1.5x to match window. */
#define INV_CELL_W 63
#define INV_CELL_H 54
#define INV_SPRITE_W 54
#define INV_SPRITE_H 48

/** Get the OSRS item ID for a consumable based on remaining doses/count. */
static int gui_consumable_osrs_id(InvSlotType type, int doses) {
    switch (type) {
        case INV_SLOT_FOOD:       return OSRS_ID_SHARK;
        case INV_SLOT_KARAMBWAN:  return OSRS_ID_KARAMBWAN;
        case INV_SLOT_BREW:
            if (doses >= 4) return OSRS_ID_BREW_4;
            if (doses == 3) return OSRS_ID_BREW_3;
            if (doses == 2) return OSRS_ID_BREW_2;
            return OSRS_ID_BREW_1;
        case INV_SLOT_RESTORE:
            if (doses >= 4) return OSRS_ID_RESTORE_4;
            if (doses == 3) return OSRS_ID_RESTORE_3;
            if (doses == 2) return OSRS_ID_RESTORE_2;
            return OSRS_ID_RESTORE_1;
        case INV_SLOT_COMBAT_POT:
            if (doses >= 4) return OSRS_ID_COMBAT_4;
            if (doses == 3) return OSRS_ID_COMBAT_3;
            if (doses == 2) return OSRS_ID_COMBAT_2;
            return OSRS_ID_COMBAT_1;
        case INV_SLOT_RANGED_POT:
            if (doses >= 4) return OSRS_ID_RANGED_4;
            if (doses == 3) return OSRS_ID_RANGED_3;
            if (doses == 2) return OSRS_ID_RANGED_2;
            return OSRS_ID_RANGED_1;
        case INV_SLOT_ANTIVENOM:
            if (doses >= 4) return OSRS_ID_ANTIVENOM_4;
            if (doses == 3) return OSRS_ID_ANTIVENOM_3;
            if (doses == 2) return OSRS_ID_ANTIVENOM_2;
            return OSRS_ID_ANTIVENOM_1;
        case INV_SLOT_PRAYER_POT:
            if (doses >= 4) return OSRS_ID_PRAYER_POT_4;
            if (doses == 3) return OSRS_ID_PRAYER_POT_3;
            if (doses == 2) return OSRS_ID_PRAYER_POT_2;
            return OSRS_ID_PRAYER_POT_1;
        case INV_SLOT_BASTION_POT:
            if (doses >= 4) return OSRS_ID_BASTION_4;
            if (doses == 3) return OSRS_ID_BASTION_3;
            if (doses == 2) return OSRS_ID_BASTION_2;
            return OSRS_ID_BASTION_1;
        case INV_SLOT_STAMINA_POT:
            if (doses >= 4) return OSRS_ID_STAMINA_4;
            if (doses == 3) return OSRS_ID_STAMINA_3;
            if (doses == 2) return OSRS_ID_STAMINA_2;
            return OSRS_ID_STAMINA_1;
        default: return 0;
    }
}

/** Find first empty slot in inventory grid (scanning left→right, top→bottom).
    Returns -1 if inventory is full. */
static int gui_inv_first_empty(GuiState* gs) {
    for (int i = 0; i < INV_GRID_SLOTS; i++) {
        if (gs->inv_grid[i].type == INV_SLOT_EMPTY) return i;
    }
    return -1;
}

/** Find the slot index of an equipment item in the inventory grid.
    Returns -1 if not found. */
static int gui_inv_find_equipment(GuiState* gs, uint8_t item_db_idx) {
    for (int i = 0; i < INV_GRID_SLOTS; i++) {
        if (gs->inv_grid[i].type == INV_SLOT_EQUIPMENT &&
            gs->inv_grid[i].item_db_idx == item_db_idx) return i;
    }
    return -1;
}

/** Remove the last occurrence of a consumable type from the inventory grid.
    In OSRS, eating removes from the slot the item is in — we remove from the
    last slot of that type (bottom-right first) since that's where the cursor
    typically is when spam-eating. Returns 1 if removed, 0 if not found. */
static int gui_inv_remove_last_consumable(GuiState* gs, InvSlotType type) {
    for (int i = INV_GRID_SLOTS - 1; i >= 0; i--) {
        if (gs->inv_grid[i].type == type) {
            gs->inv_grid[i].type = INV_SLOT_EMPTY;
            gs->inv_grid[i].item_db_idx = 0;
            gs->inv_grid[i].osrs_id = 0;
            return 1;
        }
    }
    return 0;
}

/** Place an equipment item into the inventory grid at the first empty slot.
    Returns the slot index, or -1 if full. */
static int gui_inv_place_equipment(GuiState* gs, uint8_t item_db_idx) {
    int slot = gui_inv_first_empty(gs);
    if (slot < 0) return -1;
    gs->inv_grid[slot].type = INV_SLOT_EQUIPMENT;
    gs->inv_grid[slot].item_db_idx = item_db_idx;
    gs->inv_grid[slot].osrs_id = ITEM_DATABASE[item_db_idx].item_id;
    return slot;
}

/** Copy the player-side inventory snapshot that incremental GUI updates diff against. */
static void gui_snapshot_inventory_state(GuiState* gs, const Player* p) {
    memcpy(gs->inv_prev_equipped, p->equipped, NUM_GEAR_SLOTS);
    gs->inv_prev_food_count = p->food_count;
    gs->inv_prev_karambwan_count = p->karambwan_count;
    gs->inv_prev_brew_doses = p->brew_doses;
    gs->inv_prev_restore_doses = p->restore_doses;
    gs->inv_prev_prayer_pot_doses = p->prayer_pot_doses;
    gs->inv_prev_combat_doses = p->combat_potion_doses;
    gs->inv_prev_ranged_doses = p->ranged_potion_doses;
    gs->inv_prev_bastion_doses = p->bastion_doses;
    gs->inv_prev_stamina_doses = p->stamina_doses;
    gs->inv_prev_antivenom_doses = p->antivenom_doses;
}

/** Return 1 when any inventory-tracked consumable count changed. */
static int gui_inventory_consumables_changed(const GuiState* gs, const Player* p) {
    return p->food_count != gs->inv_prev_food_count
        || p->karambwan_count != gs->inv_prev_karambwan_count
        || p->brew_doses != gs->inv_prev_brew_doses
        || p->restore_doses != gs->inv_prev_restore_doses
        || p->prayer_pot_doses != gs->inv_prev_prayer_pot_doses
        || p->combat_potion_doses != gs->inv_prev_combat_doses
        || p->ranged_potion_doses != gs->inv_prev_ranged_doses
        || p->bastion_doses != gs->inv_prev_bastion_doses
        || p->stamina_doses != gs->inv_prev_stamina_doses
        || p->antivenom_doses != gs->inv_prev_antivenom_doses;
}

/** Clear inventory-only GUI state that must not leak across resets. */
static void gui_reset_inventory_ui_state(GuiState* gs) {
    gs->inv_grid_dirty = 1;
    gs->human_clicked_inv_slot = -1;
    gs->inv_dim_slot = -1;
    gs->inv_dim_timer = 0;
    gs->inv_drag_active = 0;
    gs->inv_drag_src_slot = -1;
    gs->inv_drag_start_x = 0;
    gs->inv_drag_start_y = 0;
    gs->inv_drag_mouse_x = 0;
    gs->inv_drag_mouse_y = 0;
}

/** Full inventory grid build from player state. Called once at reset.
    Equipment items go first (unequipped gear), then consumables.
    After this, use gui_update_inventory() for incremental changes. */
static void gui_populate_inventory(GuiState* gs, Player* p) {
    memset(gs->inv_grid, 0, sizeof(gs->inv_grid));
    int n = 0;

    /* unequipped gear items from the slot inventory */
    for (int s = 0; s < NUM_GEAR_SLOTS && n < INV_GRID_SLOTS; s++) {
        for (int i = 0; i < p->num_items_in_slot[s] && n < INV_GRID_SLOTS; i++) {
            uint8_t item = p->inventory[s][i];
            if (item == ITEM_NONE) continue;
            /* skip if currently equipped */
            int is_equipped = 0;
            for (int e = 0; e < NUM_GEAR_SLOTS; e++) {
                if (p->equipped[e] == item) { is_equipped = 1; break; }
            }
            if (is_equipped) continue;
            /* skip duplicates */
            int dup = 0;
            for (int j = 0; j < n; j++) {
                if (gs->inv_grid[j].type == INV_SLOT_EQUIPMENT &&
                    gs->inv_grid[j].item_db_idx == item) { dup = 1; break; }
            }
            if (dup) continue;
            gs->inv_grid[n].type = INV_SLOT_EQUIPMENT;
            gs->inv_grid[n].item_db_idx = item;
            gs->inv_grid[n].osrs_id = ITEM_DATABASE[item].item_id;
            n++;
        }
    }

    /* consumables: food/potions are NOT stackable in OSRS.
       each shark = 1 slot. each potion vial = 1 slot (with dose-specific sprite).
       total doses are split into individual vials: e.g. 7 brew doses = 1x3-dose + 1x4-dose. */

    /* food: each unit = 1 slot */
    for (int i = 0; i < p->food_count && n < INV_GRID_SLOTS; i++) {
        gs->inv_grid[n].type = INV_SLOT_FOOD;
        gs->inv_grid[n].osrs_id = OSRS_ID_SHARK;
        n++;
    }
    for (int i = 0; i < p->karambwan_count && n < INV_GRID_SLOTS; i++) {
        gs->inv_grid[n].type = INV_SLOT_KARAMBWAN;
        gs->inv_grid[n].osrs_id = OSRS_ID_KARAMBWAN;
        n++;
    }

    /* potions: split doses into individual vials (4-dose first, remainder last) */
    #define ADD_POTION_VIALS(doses_total, slot_type) do { \
        int _rem = (doses_total); \
        while (_rem > 0 && n < INV_GRID_SLOTS) { \
            int _d = (_rem >= 4) ? 4 : _rem; \
            gs->inv_grid[n].type = (slot_type); \
            gs->inv_grid[n].osrs_id = gui_consumable_osrs_id((slot_type), _d); \
            _rem -= _d; \
            n++; \
        } \
    } while(0)

    ADD_POTION_VIALS(p->brew_doses, INV_SLOT_BREW);
    ADD_POTION_VIALS(p->restore_doses, INV_SLOT_RESTORE);
    ADD_POTION_VIALS(p->combat_potion_doses, INV_SLOT_COMBAT_POT);
    ADD_POTION_VIALS(p->ranged_potion_doses, INV_SLOT_RANGED_POT);
    ADD_POTION_VIALS(p->bastion_doses, INV_SLOT_BASTION_POT);
    ADD_POTION_VIALS(p->stamina_doses, INV_SLOT_STAMINA_POT);
    ADD_POTION_VIALS(p->antivenom_doses, INV_SLOT_ANTIVENOM);
    ADD_POTION_VIALS(p->prayer_pot_doses, INV_SLOT_PRAYER_POT);
    #undef ADD_POTION_VIALS

    /* snapshot player state for incremental change detection */
    gui_snapshot_inventory_state(gs, p);
}

/** Update potion vial doses in-place when doses change.
    E.g. drinking 1 dose from a 4-dose brew changes it to 3-dose (different sprite).
    When human_clicked_inv_slot targets a vial of this type, that specific vial loses
    the dose first (OSRS behavior: you drink from the vial you clicked). */
static void gui_inv_update_potion_doses(GuiState* gs, InvSlotType type,
                                         int total_doses) {
    /* collect existing vials of this type */
    int vial_slots[INV_GRID_SLOTS];
    int vial_count = 0;
    for (int i = 0; i < INV_GRID_SLOTS; i++) {
        if (gs->inv_grid[i].type == type) {
            vial_slots[vial_count++] = i;
        }
    }
    if (vial_count == 0) return;

    /* figure out how many doses were lost */
    int old_total = 0;
    for (int v = 0; v < vial_count; v++) {
        /* reverse-lookup current dose count from OSRS ID */
        int oid = gs->inv_grid[vial_slots[v]].osrs_id;
        int d4 = gui_consumable_osrs_id(type, 4);
        int d3 = gui_consumable_osrs_id(type, 3);
        int d2 = gui_consumable_osrs_id(type, 2);
        int d1 = gui_consumable_osrs_id(type, 1);
        if (oid == d4) old_total += 4;
        else if (oid == d3) old_total += 3;
        else if (oid == d2) old_total += 2;
        else if (oid == d1) old_total += 1;
    }
    int doses_lost = old_total - total_doses;

    /* if a human clicked a specific vial of this type, decrement that one first */
    int clicked = gs->human_clicked_inv_slot;
    if (doses_lost > 0 && clicked >= 0 && clicked < INV_GRID_SLOTS &&
        gs->inv_grid[clicked].type == type) {
        /* find current dose count of clicked vial */
        int oid = gs->inv_grid[clicked].osrs_id;
        int cur_dose = 0;
        for (int d = 4; d >= 1; d--) {
            if (oid == gui_consumable_osrs_id(type, d)) { cur_dose = d; break; }
        }
        if (cur_dose > 0) {
            int take = (doses_lost < cur_dose) ? doses_lost : cur_dose;
            cur_dose -= take;
            doses_lost -= take;
            if (cur_dose <= 0) {
                gs->inv_grid[clicked].type = INV_SLOT_EMPTY;
                gs->inv_grid[clicked].item_db_idx = 0;
                gs->inv_grid[clicked].osrs_id = 0;
            } else {
                gs->inv_grid[clicked].osrs_id = gui_consumable_osrs_id(type, cur_dose);
            }
        }
    }

    /* if doses still need removing (non-human or multiple doses lost),
       take from remaining vials in reverse order (last first) */
    for (int v = vial_count - 1; v >= 0 && doses_lost > 0; v--) {
        int slot = vial_slots[v];
        if (slot == clicked) continue; /* already handled */
        if (gs->inv_grid[slot].type != type) continue;
        int oid = gs->inv_grid[slot].osrs_id;
        int cur_dose = 0;
        for (int d = 4; d >= 1; d--) {
            if (oid == gui_consumable_osrs_id(type, d)) { cur_dose = d; break; }
        }
        if (cur_dose <= 0) continue;
        int take = (doses_lost < cur_dose) ? doses_lost : cur_dose;
        cur_dose -= take;
        doses_lost -= take;
        if (cur_dose <= 0) {
            gs->inv_grid[slot].type = INV_SLOT_EMPTY;
            gs->inv_grid[slot].item_db_idx = 0;
            gs->inv_grid[slot].osrs_id = 0;
        } else {
            gs->inv_grid[slot].osrs_id = gui_consumable_osrs_id(type, cur_dose);
        }
    }
}

/** Incremental inventory update. Detects gear switches and consumable changes
    by comparing against the previous snapshot, then modifies only affected slots.
    Items stay in their assigned positions — no compaction on eat/drink.

    OSRS gear swap rule: when you click an inventory item to equip it, the
    previously equipped item goes into that exact inventory slot (direct swap).
    Exception: equipping a 2H weapon while a shield is equipped — the shield
    goes to the first empty inventory slot since it wasn't directly clicked. */
static void gui_update_inventory(GuiState* gs, Player* p) {
    /* --- gear switches: direct slot swaps --- */
    for (int s = 0; s < NUM_GEAR_SLOTS; s++) {
        uint8_t prev = gs->inv_prev_equipped[s];
        uint8_t curr = p->equipped[s];
        if (prev == curr) continue;

        if (curr != ITEM_NONE && prev != ITEM_NONE) {
            /* swap: new item was in inventory, old item takes its exact slot */
            int src = gui_inv_find_equipment(gs, curr);
            if (src >= 0) {
                /* check if old item is still a valid swap item */
                int in_loadout = 0;
                for (int g = 0; g < NUM_GEAR_SLOTS; g++) {
                    for (int i = 0; i < p->num_items_in_slot[g]; i++) {
                        if (p->inventory[g][i] == prev) { in_loadout = 1; break; }
                    }
                    if (in_loadout) break;
                }
                if (in_loadout) {
                    /* direct swap: old item goes into the slot the new item came from */
                    gs->inv_grid[src].type = INV_SLOT_EQUIPMENT;
                    gs->inv_grid[src].item_db_idx = prev;
                    gs->inv_grid[src].osrs_id = ITEM_DATABASE[prev].item_id;
                } else {
                    /* old item not in loadout — just clear the slot */
                    gs->inv_grid[src].type = INV_SLOT_EMPTY;
                    gs->inv_grid[src].item_db_idx = 0;
                    gs->inv_grid[src].osrs_id = 0;
                }
            }
        } else if (curr != ITEM_NONE) {
            /* equipping from inventory, nothing was in this gear slot before */
            int src = gui_inv_find_equipment(gs, curr);
            if (src >= 0) {
                gs->inv_grid[src].type = INV_SLOT_EMPTY;
                gs->inv_grid[src].item_db_idx = 0;
                gs->inv_grid[src].osrs_id = 0;
            }
        } else if (prev != ITEM_NONE) {
            /* gear slot cleared (e.g. shield removed by 2H weapon equip).
               the old item goes to the first empty inventory slot. */
            int in_loadout = 0;
            for (int g = 0; g < NUM_GEAR_SLOTS; g++) {
                for (int i = 0; i < p->num_items_in_slot[g]; i++) {
                    if (p->inventory[g][i] == prev) { in_loadout = 1; break; }
                }
                if (in_loadout) break;
            }
            if (in_loadout && gui_inv_find_equipment(gs, prev) < 0) {
                gui_inv_place_equipment(gs, prev);
            }
        }
    }

    /* --- consumable changes: remove clicked slot or fall back to last --- */

    /* if a human clicked a specific consumable slot, remove that exact slot first */
    int clicked = gs->human_clicked_inv_slot;
    int clicked_used = 0;

    /* food */
    int food_diff = gs->inv_prev_food_count - p->food_count;
    for (int i = 0; i < food_diff; i++) {
        if (!clicked_used && clicked >= 0 && clicked < INV_GRID_SLOTS &&
            gs->inv_grid[clicked].type == INV_SLOT_FOOD) {
            gs->inv_grid[clicked].type = INV_SLOT_EMPTY;
            gs->inv_grid[clicked].item_db_idx = 0;
            gs->inv_grid[clicked].osrs_id = 0;
            clicked_used = 1;
        } else {
            gui_inv_remove_last_consumable(gs, INV_SLOT_FOOD);
        }
    }

    /* karambwan */
    int karam_diff = gs->inv_prev_karambwan_count - p->karambwan_count;
    for (int i = 0; i < karam_diff; i++) {
        if (!clicked_used && clicked >= 0 && clicked < INV_GRID_SLOTS &&
            gs->inv_grid[clicked].type == INV_SLOT_KARAMBWAN) {
            gs->inv_grid[clicked].type = INV_SLOT_EMPTY;
            gs->inv_grid[clicked].item_db_idx = 0;
            gs->inv_grid[clicked].osrs_id = 0;
            clicked_used = 1;
        } else {
            gui_inv_remove_last_consumable(gs, INV_SLOT_KARAMBWAN);
        }
    }

    /* potions: dose changes update existing vials in-place (sprite change),
       and remove empty vials when a full vial is consumed.
       human_clicked_inv_slot is still set here so the clicked vial loses the dose. */
    if (p->brew_doses != gs->inv_prev_brew_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_BREW, p->brew_doses);
    }
    if (p->restore_doses != gs->inv_prev_restore_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_RESTORE, p->restore_doses);
    }
    if (p->combat_potion_doses != gs->inv_prev_combat_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_COMBAT_POT, p->combat_potion_doses);
    }
    if (p->ranged_potion_doses != gs->inv_prev_ranged_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_RANGED_POT, p->ranged_potion_doses);
    }
    if (p->bastion_doses != gs->inv_prev_bastion_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_BASTION_POT, p->bastion_doses);
    }
    if (p->stamina_doses != gs->inv_prev_stamina_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_STAMINA_POT, p->stamina_doses);
    }
    if (p->antivenom_doses != gs->inv_prev_antivenom_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_ANTIVENOM, p->antivenom_doses);
    }
    if (p->prayer_pot_doses != gs->inv_prev_prayer_pot_doses) {
        gui_inv_update_potion_doses(gs, INV_SLOT_PRAYER_POT, p->prayer_pot_doses);
    }

    /* only clear human click when a consumable was actually used this frame.
       if no diff happened yet, keep it for the next tick when the sim processes the action. */
    if (clicked_used || gui_inventory_consumables_changed(gs, p)) {
        gs->human_clicked_inv_slot = -1;
    }

    /* update snapshot */
    gui_snapshot_inventory_state(gs, p);
}

/** Get the inventory grid screen position for a slot index. */
static void gui_inv_slot_pos(GuiState* gs, int slot, int* out_x, int* out_y) {
    int grid_w = INV_COLS * INV_CELL_W;
    int grid_x = gs->panel_x + (gs->panel_w - grid_w) / 2;
    int grid_y = gui_content_y(gs) + 4;
    int col = slot % INV_COLS;
    int row = slot / INV_COLS;
    *out_x = grid_x + col * INV_CELL_W;
    *out_y = grid_y + row * INV_CELL_H;
}

/** Hit test: return inventory slot index at screen position, or -1. */
static int gui_inv_slot_at(GuiState* gs, int mx, int my) {
    for (int i = 0; i < INV_GRID_SLOTS; i++) {
        int sx, sy;
        gui_inv_slot_pos(gs, i, &sx, &sy);
        if (mx >= sx && mx < sx + INV_CELL_W && my >= sy && my < sy + INV_CELL_H) {
            return i;
        }
    }
    return -1;
}

/** Handle inventory click: equip gear items, eat/drink consumables.
    hi is a HumanInput* (from osrs_pvp_human_input_types.h, included above).
    When non-NULL and enabled, food/potion clicks set pending_* fields instead of
    directly mutating player state, so the action system handles timers. */

static InvAction gui_inv_click(GuiState* gs, Player* p, int slot,
                                HumanInput* hi) {
    if (slot < 0 || slot >= INV_GRID_SLOTS) return INV_ACTION_NONE;
    InvSlot* inv = &gs->inv_grid[slot];
    if (inv->type == INV_SLOT_EMPTY) return INV_ACTION_NONE;

    /* start dim animation */
    gs->inv_dim_slot = slot;
    gs->inv_dim_timer = INV_DIM_TICKS;

    /* when human control is active, route food/potion through action system
       instead of directly mutating player state (respects timers) */
    int human_active = (hi && hi->enabled);

    switch (inv->type) {
        case INV_SLOT_EQUIPMENT: {
            int gear_slot = item_to_gear_slot(inv->item_db_idx);
            if (gear_slot >= 0) {
                if (human_active) {
                    human_input_queue_equip_inventory_item(hi, slot, inv->item_db_idx, gear_slot);
                    gs->human_clicked_inv_slot = slot;
                } else {
                    slot_equip_item(p, gear_slot, inv->item_db_idx);
                }
            }
            return INV_ACTION_EQUIP;
        }
        case INV_SLOT_FOOD:
            if (human_active) {
                hi->pending_food = 1;
                human_input_queue_eat(hi, 0);
                gs->human_clicked_inv_slot = slot;
            }
            else { eat_food(p, 0); }
            return INV_ACTION_EAT;
        case INV_SLOT_KARAMBWAN:
            if (human_active) {
                hi->pending_karambwan = 1;
                human_input_queue_eat(hi, 1);
                gs->human_clicked_inv_slot = slot;
            }
            else { eat_food(p, 1); }
            return INV_ACTION_EAT;
        case INV_SLOT_BREW:
            if (human_active) {
                hi->pending_potion = POTION_BREW;
                human_input_queue_drink(hi, POTION_BREW, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        case INV_SLOT_RESTORE:
            if (human_active) {
                hi->pending_potion = POTION_RESTORE;
                human_input_queue_drink(hi, POTION_RESTORE, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        case INV_SLOT_COMBAT_POT:
            if (human_active) {
                hi->pending_potion = POTION_COMBAT;
                human_input_queue_drink(hi, POTION_COMBAT, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        case INV_SLOT_RANGED_POT:
            if (human_active) {
                hi->pending_potion = POTION_RANGED;
                human_input_queue_drink(hi, POTION_RANGED, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        case INV_SLOT_ANTIVENOM:
            if (human_active) {
                hi->pending_potion = POTION_ANTIVENOM;
                human_input_queue_drink(hi, POTION_ANTIVENOM, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        case INV_SLOT_PRAYER_POT:
            if (human_active) {
                hi->pending_potion = POTION_PRAYER_POT;
                human_input_queue_drink(hi, POTION_PRAYER_POT, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        case INV_SLOT_BASTION_POT:
            if (human_active) {
                hi->pending_potion = POTION_BASTION;
                human_input_queue_drink(hi, POTION_BASTION, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        case INV_SLOT_STAMINA_POT:
            if (human_active) {
                hi->pending_potion = POTION_STAMINA;
                human_input_queue_drink(hi, POTION_STAMINA, slot);
                gs->human_clicked_inv_slot = slot;
            }
            return INV_ACTION_DRINK;
        default:
            return INV_ACTION_NONE;
    }
}

/** Handle inventory mouse input: clicks, drag start/move/release.
    When hi is non-NULL and enabled, food/potion clicks route through the
    action system instead of directly mutating player state. */
static void gui_inv_handle_mouse(GuiState* gs, Player* p, HumanInput* hi) {
    if (gs->active_tab != GUI_TAB_INVENTORY) return;

    int mx = GetMouseX();
    int my = GetMouseY();

    /* drag in progress */
    if (gs->inv_drag_active) {
        gs->inv_drag_mouse_x = mx;
        gs->inv_drag_mouse_y = my;

        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            /* drop: swap src and target slots */
            int target = gui_inv_slot_at(gs, mx, my);
            if (target >= 0 && target != gs->inv_drag_src_slot) {
                InvSlot tmp = gs->inv_grid[target];
                gs->inv_grid[target] = gs->inv_grid[gs->inv_drag_src_slot];
                gs->inv_grid[gs->inv_drag_src_slot] = tmp;
            }
            gs->inv_drag_active = 0;
            gs->inv_drag_src_slot = -1;
        }
        return;
    }

    /* new click */
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        int slot = gui_inv_slot_at(gs, mx, my);
        if (slot >= 0 && gs->inv_grid[slot].type != INV_SLOT_EMPTY) {
            gs->inv_drag_start_x = mx;
            gs->inv_drag_start_y = my;
            gs->inv_drag_src_slot = slot;
        }
    }

    /* check if held mouse has moved past dead zone → start drag */
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && gs->inv_drag_src_slot >= 0 && !gs->inv_drag_active) {
        int dx = mx - gs->inv_drag_start_x;
        int dy = my - gs->inv_drag_start_y;
        if (dx > INV_DRAG_DEAD_ZONE || dx < -INV_DRAG_DEAD_ZONE ||
            dy > INV_DRAG_DEAD_ZONE || dy < -INV_DRAG_DEAD_ZONE) {
            gs->inv_drag_active = 1;
            gs->inv_drag_mouse_x = mx;
            gs->inv_drag_mouse_y = my;
            /* dim the source slot during drag */
            gs->inv_dim_slot = gs->inv_drag_src_slot;
            gs->inv_dim_timer = 9999;  /* stays dim during entire drag */
        }
    }

    /* click release without drag = activate item */
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && gs->inv_drag_src_slot >= 0 && !gs->inv_drag_active) {
        gui_inv_click(gs, p, gs->inv_drag_src_slot, hi);
        gs->inv_drag_src_slot = -1;
    }
}

/** Tick the inventory dim timer (call at 50 Hz). */
static void gui_inv_tick(GuiState* gs) {
    if (gs->inv_dim_timer > 0 && !gs->inv_drag_active) {
        gs->inv_dim_timer--;
        if (gs->inv_dim_timer <= 0) {
            gs->inv_dim_slot = -1;
        }
    }
}

static void gui_draw_inventory(GuiState* gs, Player* p) {
    /* full rebuild on first frame or reset, incremental updates after */
    if (gs->inv_grid_dirty) {
        gui_populate_inventory(gs, p);
        gs->inv_grid_dirty = 0;
    } else {
        gui_update_inventory(gs, p);
    }

    /* draw 4x7 slot backgrounds (subtle dark rectangles matching OSRS inventory) */
    for (int slot = 0; slot < INV_GRID_SLOTS; slot++) {
        int cx, cy;
        gui_inv_slot_pos(gs, slot, &cx, &cy);
        /* OSRS inventory slots have a very subtle dark border/tint.
           draw a 36x32 centered slot background to delineate cells. */
        int sx = cx + (INV_CELL_W - INV_SPRITE_W) / 2;
        int sy = cy + (INV_CELL_H - INV_SPRITE_H) / 2;
        DrawRectangle(sx, sy, INV_SPRITE_W, INV_SPRITE_H,
                      CLITERAL(Color){ 0, 0, 0, 30 });
    }

    /* draw items (sprites are 36x32 native, scaled to INV_SPRITE_W x INV_SPRITE_H) */
    for (int slot = 0; slot < INV_GRID_SLOTS; slot++) {
        int cx, cy;
        gui_inv_slot_pos(gs, slot, &cx, &cy);
        InvSlot* inv = &gs->inv_grid[slot];

        if (inv->type == INV_SLOT_EMPTY) continue;

        /* determine sprite */
        Texture2D tex = { 0 };
        if (inv->type == INV_SLOT_EQUIPMENT) {
            tex = gui_get_item_sprite(gs, inv->item_db_idx);
        } else {
            tex = gui_get_sprite_by_osrs_id(gs, inv->osrs_id);
        }

        /* dim tint: 50% alpha when clicked/dragged (matches OSRS var17=128) */
        int is_dimmed = (gs->inv_dim_slot == slot && gs->inv_dim_timer > 0);
        Color tint = is_dimmed ? CLITERAL(Color){ 255, 255, 255, 128 } : WHITE;

        int dx = cx + (INV_CELL_W - INV_SPRITE_W) / 2;
        int dy = cy + (INV_CELL_H - INV_SPRITE_H) / 2;

        /* skip drawing at grid position if being dragged (drawn at cursor instead) */
        if (gs->inv_drag_active && slot == gs->inv_drag_src_slot) {
            if (tex.id != 0) {
                Rectangle src = { 0, 0, (float)tex.width, (float)tex.height };
                Rectangle dst = { (float)dx, (float)dy, (float)INV_SPRITE_W, (float)INV_SPRITE_H };
                DrawTexturePro(tex, src, dst, (Vector2){0,0}, 0.0f,
                               CLITERAL(Color){ 255, 255, 255, 80 });
            }
            continue;
        }

        if (tex.id != 0) {
            Rectangle src = { 0, 0, (float)tex.width, (float)tex.height };
            Rectangle dst = { (float)dx, (float)dy, (float)INV_SPRITE_W, (float)INV_SPRITE_H };
            DrawTexturePro(tex, src, dst, (Vector2){0,0}, 0.0f, tint);
        } else {
            const char* name = (inv->type == INV_SLOT_EQUIPMENT)
                ? gui_item_short_name(inv->item_db_idx) : "???";
            gui_text_shadow(name, cx + 2, cy + 12, 7, GUI_TEXT_YELLOW);
        }
    }

    /* draw dragged item at cursor position */
    if (gs->inv_drag_active && gs->inv_drag_src_slot >= 0) {
        InvSlot* drag = &gs->inv_grid[gs->inv_drag_src_slot];
        Texture2D tex = { 0 };
        if (drag->type == INV_SLOT_EQUIPMENT) {
            tex = gui_get_item_sprite(gs, drag->item_db_idx);
        } else {
            tex = gui_get_sprite_by_osrs_id(gs, drag->osrs_id);
        }
        if (tex.id != 0) {
            int dx = gs->inv_drag_mouse_x - INV_SPRITE_W / 2;
            int dy = gs->inv_drag_mouse_y - INV_SPRITE_H / 2;
            Rectangle src = { 0, 0, (float)tex.width, (float)tex.height };
            Rectangle dst = { (float)dx, (float)dy, (float)INV_SPRITE_W, (float)INV_SPRITE_H };
            DrawTexturePro(tex, src, dst, (Vector2){0,0}, 0.0f,
                           CLITERAL(Color){ 255, 255, 255, 200 });
        }

        /* highlight target slot under cursor */
        int target = gui_inv_slot_at(gs, gs->inv_drag_mouse_x, gs->inv_drag_mouse_y);
        if (target >= 0 && target != gs->inv_drag_src_slot) {
            int tx, ty;
            gui_inv_slot_pos(gs, target, &tx, &ty);
            DrawRectangle(tx, ty, INV_CELL_W, INV_CELL_H,
                          CLITERAL(Color){ 255, 255, 255, 40 });
        }
    }
}


static void gui_draw_equipment(GuiState* gs, Player* p) {
    int oy = gui_content_y(gs) + 8;

    gui_text_shadow("Worn Equipment", gs->panel_x + 8, oy, 12, GUI_TEXT_ORANGE);
    oy += 22;

    /* OSRS paperdoll: 5 rows, 3-column layout centered in panel.
       slot sizes scaled to fill panel width (320px - 16px padding = 304px). */
    int gap = 6;
    int sw = (gs->panel_w - 16 - gap * 2) / 3;  /* ~97px per slot */
    int sh = (int)(sw * 0.75f);                    /* maintain ~4:3 aspect ratio (~73px) */
    int cx = gs->panel_x + gs->panel_w / 2;
    int r3_w = sw * 3 + gap * 2;
    int r3_x = cx - r3_w / 2;

    /* row 0: head (centered) */
    gui_draw_equip_slot(gs, cx - sw / 2, oy, sw, sh, GEAR_SLOT_HEAD, p->equipped[GEAR_SLOT_HEAD]);
    oy += sh + gap;

    /* row 1: cape, neck, ammo */
    gui_draw_equip_slot(gs, r3_x, oy, sw, sh, GEAR_SLOT_CAPE, p->equipped[GEAR_SLOT_CAPE]);
    gui_draw_equip_slot(gs, r3_x + sw + gap, oy, sw, sh, GEAR_SLOT_NECK, p->equipped[GEAR_SLOT_NECK]);
    gui_draw_equip_slot(gs, r3_x + 2 * (sw + gap), oy, sw, sh, GEAR_SLOT_AMMO, p->equipped[GEAR_SLOT_AMMO]);
    oy += sh + gap;

    /* row 2: weapon, body, shield */
    gui_draw_equip_slot(gs, r3_x, oy, sw, sh, GEAR_SLOT_WEAPON, p->equipped[GEAR_SLOT_WEAPON]);
    gui_draw_equip_slot(gs, r3_x + sw + gap, oy, sw, sh, GEAR_SLOT_BODY, p->equipped[GEAR_SLOT_BODY]);
    gui_draw_equip_slot(gs, r3_x + 2 * (sw + gap), oy, sw, sh, GEAR_SLOT_SHIELD, p->equipped[GEAR_SLOT_SHIELD]);
    oy += sh + gap;

    /* row 3: legs (centered) */
    gui_draw_equip_slot(gs, cx - sw / 2, oy, sw, sh, GEAR_SLOT_LEGS, p->equipped[GEAR_SLOT_LEGS]);
    oy += sh + gap;

    /* row 4: hands, feet, ring */
    gui_draw_equip_slot(gs, r3_x, oy, sw, sh, GEAR_SLOT_HANDS, p->equipped[GEAR_SLOT_HANDS]);
    gui_draw_equip_slot(gs, r3_x + sw + gap, oy, sw, sh, GEAR_SLOT_FEET, p->equipped[GEAR_SLOT_FEET]);
    gui_draw_equip_slot(gs, r3_x + 2 * (sw + gap), oy, sw, sh, GEAR_SLOT_RING, p->equipped[GEAR_SLOT_RING]);
}


/* enum order is already display order — 5 cols × 6 rows, 29 prayers + 1 empty.
   grid is just the identity: position i maps to enum value i. */
#define GUI_PRAYER_GRID_COUNT GUI_NUM_PRAYERS

/** Check if a prayer grid slot is currently active based on player state. */
static int gui_prayer_is_active(GuiPrayerIdx pidx, Player* p) {
    switch (pidx) {
        case GUI_PRAY_PROTECT_MAGIC:    return p->prayer == PRAYER_PROTECT_MAGIC;
        case GUI_PRAY_PROTECT_MISSILES: return p->prayer == PRAYER_PROTECT_RANGED;
        case GUI_PRAY_PROTECT_MELEE:    return p->prayer == PRAYER_PROTECT_MELEE;
        case GUI_PRAY_REDEMPTION:       return p->prayer == PRAYER_REDEMPTION;
        case GUI_PRAY_SMITE:            return p->prayer == PRAYER_SMITE;
        case GUI_PRAY_PIETY:            return p->offensive_prayer == OFFENSIVE_PRAYER_PIETY;
        case GUI_PRAY_RIGOUR:           return p->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR;
        case GUI_PRAY_AUGURY:           return p->offensive_prayer == OFFENSIVE_PRAYER_AUGURY;
        default: return 0;
    }
}

static void gui_draw_prayer(GuiState* gs, Player* p) {
    int oy = gui_content_y(gs) + 4;

    /* prayer points bar at top */
    int bar_x = gs->panel_x + 8;
    int bar_w = gs->panel_w - 16;
    int bar_h = 18;
    float pray_pct = (p->base_prayer > 0) ?
        (float)p->current_prayer / (float)p->base_prayer : 0.0f;
    DrawRectangle(bar_x, oy, bar_w, bar_h, GUI_SPEC_DARK);
    DrawRectangle(bar_x, oy, (int)(bar_w * pray_pct), bar_h, GUI_TEXT_CYAN);
    DrawRectangleLines(bar_x, oy, bar_w, bar_h, GUI_BORDER);
    gui_text_shadow(TextFormat("%d / %d", p->current_prayer, p->base_prayer),
                    bar_x + bar_w / 2 - 20, oy + 3, 10, GUI_TEXT_WHITE);
    oy += bar_h + 6;

    /* 5-column grid of all 25 prayers.
       OSRS native: 37x37 pitch, scaled to fill 320px panel (~60px cells). */
    int cols = 5;
    int gap = 2;
    int icon_sz = (gs->panel_w - 16 - gap * (cols - 1)) / cols;  /* ~60px */
    int grid_w = cols * icon_sz + (cols - 1) * gap;
    int gx = gs->panel_x + (gs->panel_w - grid_w) / 2;

    for (int i = 0; i < GUI_PRAYER_GRID_COUNT; i++) {
        int col = i % cols;
        int row = i / cols;
        int ix = gx + col * (icon_sz + gap);
        int iy = oy + row * (icon_sz + gap);

        GuiPrayerIdx pidx = (GuiPrayerIdx)i;
        int active = gui_prayer_is_active(pidx, p);

        /* draw slot_tile background */
        if (gs->slot_tile.id != 0) {
            Rectangle src = { 0, 0, (float)gs->slot_tile.width, (float)gs->slot_tile.height };
            Rectangle dst = { (float)ix, (float)iy, (float)icon_sz, (float)icon_sz };
            DrawTexturePro(gs->slot_tile, src, dst, (Vector2){0,0}, 0.0f, WHITE);
        }

        /* active prayer: yellow highlight overlay */
        if (active) {
            DrawRectangle(ix, iy, icon_sz, icon_sz, GUI_PRAYER_ON);
        }

        /* draw prayer sprite (scaled to cell) */
        if (gs->sprites_loaded) {
            Texture2D tex = active ? gs->prayer_on[pidx] : gs->prayer_off[pidx];
            if (tex.id != 0) {
                Rectangle src = { 0, 0, (float)tex.width, (float)tex.height };
                Rectangle dst = { (float)ix, (float)iy, (float)icon_sz, (float)icon_sz };
                DrawTexturePro(tex, src, dst, (Vector2){0,0}, 0.0f, WHITE);
            }
        }
    }
}


static void gui_draw_combat(GuiState* gs, Player* p) {
    int ox = gs->panel_x + 8;
    int oy = gui_content_y(gs) + 8;

    /* weapon name + sprite (scaled to match panel) */
    const char* wpn_name = "Unarmed";
    if (p->equipped[GEAR_SLOT_WEAPON] != ITEM_NONE &&
        p->equipped[GEAR_SLOT_WEAPON] < NUM_ITEMS) {
        wpn_name = gui_item_short_name(p->equipped[GEAR_SLOT_WEAPON]);
    }

    Texture2D wpn_tex = gui_get_item_sprite(gs, p->equipped[GEAR_SLOT_WEAPON]);
    if (wpn_tex.id != 0) {
        Rectangle src = { 0, 0, (float)wpn_tex.width, (float)wpn_tex.height };
        Rectangle dst = { (float)ox, (float)oy, 60.0f, 54.0f };
        DrawTexturePro(wpn_tex, src, dst, (Vector2){0,0}, 0.0f, WHITE);
        gui_text_shadow(wpn_name, ox + 66, oy + 16, 14, GUI_TEXT_ORANGE);
        oy += 60;
    } else {
        gui_text_shadow(wpn_name, ox, oy, 14, GUI_TEXT_ORANGE);
        oy += 22;
    }

    /* 4 attack style buttons (2x2 grid) scaled to fill panel width.
       style names are weapon-specific in real OSRS (bow=Accurate/Rapid/Longrange,
       scythe=Chop/Jab/Block, godsword=Chop/Slash/Smash/Block, etc.). */
    const char* style_names[4] = { "Accurate", "Aggressive", "Controlled", "Defensive" };
    int num_styles = 4;
    switch (p->equipped[GEAR_SLOT_WEAPON]) {
        case ITEM_TOXIC_BLOWPIPE:
        case ITEM_ZARYTE_CROSSBOW:
        case ITEM_TWISTED_BOW:
            style_names[0] = "Accurate"; style_names[1] = "Rapid";
            style_names[2] = "Longrange"; num_styles = 3; break;
        case ITEM_SCYTHE_OF_VITUR:
            style_names[0] = "Chop"; style_names[1] = "Jab";
            style_names[2] = "Block"; num_styles = 3; break;
        case ITEM_SGS:
            style_names[0] = "Chop"; style_names[1] = "Slash";
            style_names[2] = "Smash"; style_names[3] = "Block"; break;
        case ITEM_DRAGON_CLAWS:
            style_names[0] = "Chop"; style_names[1] = "Slash";
            style_names[2] = "Lunge"; style_names[3] = "Block"; break;
        case ITEM_KODAI_WAND:
            style_names[0] = "Bash"; style_names[1] = "Pound";
            style_names[2] = "Focus"; style_names[3] = "Block"; break;
        default: break;
    }
    int btn_gap = 6;
    int btn_w = (gs->panel_w - 16 - btn_gap) / 2;
    int btn_h = 60;

    for (int i = 0; i < num_styles; i++) {
        int col = i % 2;
        int row = i / 2;
        int bx = ox + col * (btn_w + btn_gap);
        int by = oy + row * (btn_h + btn_gap);

        int active = ((int)p->fight_style == i);

        if (gs->slot_tile.id != 0) {
            Rectangle src = { 0, 0, (float)gs->slot_tile.width, (float)gs->slot_tile.height };
            Rectangle dst = { (float)bx, (float)by, (float)btn_w, (float)btn_h };
            DrawTexturePro(gs->slot_tile, src, dst, (Vector2){0,0}, 0.0f, WHITE);
        }
        if (active) {
            DrawRectangle(bx, by, btn_w, btn_h, GUI_PRAYER_ON);
        }
        DrawRectangleLines(bx, by, btn_w, btn_h, GUI_BORDER);

        Color txt_c = active ? GUI_TEXT_YELLOW : GUI_TEXT_WHITE;
        int txt_w = MeasureText(style_names[i], 11);
        gui_text_shadow(style_names[i], bx + btn_w / 2 - txt_w / 2, by + btn_h / 2 - 5, 11, txt_c);
    }
    oy += 2 * (btn_h + btn_gap) + 10;

    /* special attack bar — clickable. yellow border when spec is queued (OSRS-style). */
    Color spec_label_color = p->spec_armed ? GUI_TEXT_YELLOW : GUI_TEXT_WHITE;
    gui_text_shadow("Special Attack", ox, oy, 11, spec_label_color);
    oy += 16;

    int spec_w = gs->panel_w - 16;
    int spec_h = 26;
    float spec_pct = (float)p->special_energy / 100.0f;

    /* background: draw spec bar sprite if available, else dark rect */
    if (gs->spec_bar_loaded && gs->spec_bar.id != 0) {
        /* stretch spec bar sprite to fill */
        Rectangle src = { 0, 0, (float)gs->spec_bar.width, (float)gs->spec_bar.height };
        Rectangle dst = { (float)ox, (float)oy, (float)spec_w, (float)spec_h };
        DrawTexturePro(gs->spec_bar, src, dst, (Vector2){0, 0}, 0.0f, CLITERAL(Color){80, 80, 80, 255});
        /* green fill overlay */
        DrawRectangle(ox, oy, (int)(spec_w * spec_pct), spec_h,
                      CLITERAL(Color){ 0, 180, 0, 160 });
    } else {
        DrawRectangle(ox, oy, spec_w, spec_h, GUI_SPEC_DARK);
        DrawRectangle(ox, oy, (int)(spec_w * spec_pct), spec_h, GUI_SPEC_GREEN);
    }
    /* active highlight: bright yellow-green border when spec is queued */
    if (p->spec_armed) {
        DrawRectangle(ox, oy, spec_w, spec_h, CLITERAL(Color){ 200, 200, 50, 60 });
        DrawRectangleLines(ox, oy, spec_w, spec_h, GUI_TEXT_YELLOW);
    } else {
        DrawRectangleLines(ox, oy, spec_w, spec_h, GUI_BORDER);
    }
    gui_text_shadow(TextFormat("%d%%", p->special_energy),
                    ox + spec_w / 2 - 10, oy + 4, 10, GUI_TEXT_WHITE);
}


typedef struct {
    const char* name;
    GuiSpellIdx idx;
} GuiSpellEntry;

/* Ancient spellbook grid (4 cols × 4 rows = 16 combat spells, + vengeance).
   sort is by level: rows go Rush/Burst/Blitz/Barrage top-to-bottom, and
   within each row the order is Smoke / Shadow / Blood / Ice (ascending level).
   only Ice/Blood/Vengeance are castable in this env — Smoke/Shadow render
   greyed out with the "off" sprite and don't respond to clicks. */
static const GuiSpellEntry GUI_SPELL_GRID[] = {
    { "Smoke Rush",    GUI_SPELL_SMOKE_RUSH },
    { "Shadow Rush",   GUI_SPELL_SHADOW_RUSH },
    { "Blood Rush",    GUI_SPELL_BLOOD_RUSH },
    { "Ice Rush",      GUI_SPELL_ICE_RUSH },
    { "Smoke Burst",   GUI_SPELL_SMOKE_BURST },
    { "Shadow Burst",  GUI_SPELL_SHADOW_BURST },
    { "Blood Burst",   GUI_SPELL_BLOOD_BURST },
    { "Ice Burst",     GUI_SPELL_ICE_BURST },
    { "Smoke Blitz",   GUI_SPELL_SMOKE_BLITZ },
    { "Shadow Blitz",  GUI_SPELL_SHADOW_BLITZ },
    { "Blood Blitz",   GUI_SPELL_BLOOD_BLITZ },
    { "Ice Blitz",     GUI_SPELL_ICE_BLITZ },
    { "Smoke Barrage", GUI_SPELL_SMOKE_BARRAGE },
    { "Shadow Barrage",GUI_SPELL_SHADOW_BARRAGE },
    { "Blood Barrage", GUI_SPELL_BLOOD_BARRAGE },
    { "Ice Barrage",   GUI_SPELL_ICE_BARRAGE },
    { "Vengeance",     GUI_SPELL_VENGEANCE },
};
#define GUI_SPELL_GRID_COUNT 17

/* which spells are castable in this env (others render greyed out). */
static inline int gui_spell_castable(GuiSpellIdx s) {
    return (s == GUI_SPELL_VENGEANCE)
        || (s >= GUI_SPELL_ICE_RUSH && s <= GUI_SPELL_ICE_BURST)     /* ice */
        || (s >= GUI_SPELL_BLOOD_RUSH && s <= GUI_SPELL_BLOOD_BURST) /* blood */
        || (s == GUI_SPELL_ICE_BLITZ || s == GUI_SPELL_ICE_BARRAGE)
        || (s == GUI_SPELL_BLOOD_BLITZ || s == GUI_SPELL_BLOOD_BARRAGE);
}

static void gui_draw_spellbook(GuiState* gs, Player* p) {
    int oy = gui_content_y(gs) + 8;

    /* 4-column grid of all spells, scaled to fill panel width.
       OSRS ancient spellbook native is ~26x26 icons. we scale up to match
       the inventory cell size for visual consistency. */
    int cols = 4;
    int gap = 2;
    int icon_sz = (gs->panel_w - 16 - gap * (cols - 1)) / cols;  /* ~76px */
    int grid_w = cols * icon_sz + (cols - 1) * gap;
    int gx = gs->panel_x + (gs->panel_w - grid_w) / 2;

    for (int i = 0; i < GUI_SPELL_GRID_COUNT; i++) {
        int col = i % cols;
        int row = i / cols;
        int ix = gx + col * (icon_sz + gap);
        int iy = oy + row * (icon_sz + gap);

        GuiSpellIdx sidx_here = GUI_SPELL_GRID[i].idx;
        /* active highlight for vengeance */
        int active = (sidx_here == GUI_SPELL_VENGEANCE && p->veng_active);
        /* spell-targeting mode: highlight the pending spell cell */
        int targeting = (gs->pending_spell_highlight >= 0 &&
                         (int)sidx_here == gs->pending_spell_highlight);

        /* slot_tile background */
        if (gs->slot_tile.id != 0) {
            Rectangle src = { 0, 0, (float)gs->slot_tile.width, (float)gs->slot_tile.height };
            Rectangle dst = { (float)ix, (float)iy, (float)icon_sz, (float)icon_sz };
            DrawTexturePro(gs->slot_tile, src, dst, (Vector2){0,0}, 0.0f, WHITE);
        }

        if (active) {
            DrawRectangle(ix, iy, icon_sz, icon_sz, GUI_PRAYER_ON);
        }
        if (targeting) {
            /* yellow 2px border around the targeted spell */
            DrawRectangleLinesEx((Rectangle){(float)ix, (float)iy, (float)icon_sz, (float)icon_sz}, 2.0f, YELLOW);
        }

        /* draw spell sprite. non-castable spells (smoke/shadow in this env)
           use the greyed "off" sprite so the panel looks authentic. */
        GuiSpellIdx sidx = GUI_SPELL_GRID[i].idx;
        if (gs->sprites_loaded) {
            int castable = gui_spell_castable(sidx);
            Texture2D tex = castable ? gs->spell_on[sidx] : gs->spell_off[sidx];
            if (tex.id != 0) {
                Rectangle src = { 0, 0, (float)tex.width, (float)tex.height };
                Rectangle dst = { (float)ix, (float)iy, (float)icon_sz, (float)icon_sz };
                DrawTexturePro(tex, src, dst, (Vector2){0,0}, 0.0f, WHITE);
            }
        }
    }
    oy += ((GUI_SPELL_GRID_COUNT + cols - 1) / cols) * (icon_sz + gap) + 12;

    /* veng cooldown below the grid */
    int ox = gs->panel_x + 8;
    gui_text_shadow(TextFormat("Veng cooldown: %d", p->veng_cooldown),
                    ox, oy, 10, p->veng_cooldown > 0 ? GUI_TEXT_RED : GUI_TEXT_GREEN);
    (void)p;
}
/* stats panel — OSRS-authentic skills tab layout                            */
/*                                                                           */
/* matches the real OSRS fixed-mode skills interface: 3-column grid,         */
/* skill icon on left, current level center, base level right.               */
/* only combat skills are shown; non-combat rows are empty.                  */
/* below the skill grid: combat info (max hit, gear, prayer, consumables).   */
/*                                                                           */
/* OSRS skill grid order (3 columns, 8 rows):                                */
/*   col 0: Attack, Strength, Defence, Ranged, Prayer, Magic, RC, Constr    */
/*   col 1: Hitpoints, Agility, Herblore, Thieving, Crafting, Fletch, ...   */
/*   col 2: Mining, Smithing, Fishing, Cooking, Firemaking, WC, ...          */
/* we show rows 0-5 (the 7 combat skills) and leave col 2 empty.            */
/* skill icon indices (matches skill_icon_files load order) */
#define SKILL_ICON_ATTACK    0
#define SKILL_ICON_STRENGTH  1
#define SKILL_ICON_DEFENCE   2
#define SKILL_ICON_RANGED    3
#define SKILL_ICON_PRAYER    4
#define SKILL_ICON_MAGIC     5
#define SKILL_ICON_HITPOINTS 6

/* draw one skill cell: icon + current level (left) + base level (right).
   OSRS style: yellow if current == base, green if boosted, red if drained.
   cell dimensions match the real client scaled to our panel width. */
static void gui_draw_skill_cell(GuiState* gs, int cx, int cy, int cw, int ch,
                                 int icon_idx, int current, int base) {
    /* dark cell background with border (matches OSRS skill cell) */
    DrawRectangle(cx, cy, cw, ch, (Color){30, 27, 20, 255});
    DrawRectangleLines(cx, cy, cw, ch, (Color){60, 54, 42, 255});

    /* skill icon (scaled to fit cell height with padding) */
    int icon_sz = ch - 6;
    int icon_x = cx + 3;
    int icon_y = cy + 3;
    if (gs->skill_icons_loaded && icon_idx >= 0 && icon_idx < 7 &&
        gs->skill_icons[icon_idx].id != 0) {
        Texture2D tex = gs->skill_icons[icon_idx];
        Rectangle src = { 0, 0, (float)tex.width, (float)tex.height };
        Rectangle dst = { (float)icon_x, (float)icon_y, (float)icon_sz, (float)icon_sz };
        DrawTexturePro(tex, src, dst, (Vector2){0,0}, 0.0f, WHITE);
    }

    /* level color: yellow=normal, green=boosted, red=drained */
    Color lvl_color = GUI_TEXT_YELLOW;
    if (current > base) lvl_color = GUI_TEXT_GREEN;
    else if (current < base) lvl_color = (Color){255, 60, 60, 255};

    /* current level (left side, after icon) */
    int text_y = cy + ch / 2 - 5;
    gui_text_shadow(TextFormat("%d", current), icon_x + icon_sz + 4, text_y, 10, lvl_color);

    /* base level (right-aligned) */
    const char* base_str = TextFormat("%d", base);
    int bw = MeasureText(base_str, 10);
    gui_text_shadow(base_str, cx + cw - bw - 4, text_y, 10, lvl_color);
}

static const char* gui_gear_name(GearSet g) {
    switch (g) {
        case GEAR_MAGE:   return "Mage";
        case GEAR_RANGED: return "Ranged";
        case GEAR_MELEE:  return "Melee";
        case GEAR_SPEC:   return "Spec";
        case GEAR_TANK:   return "Tank";
        default:          return "???";
    }
}

static const char* gui_prayer_name(OverheadPrayer pr) {
    switch (pr) {
        case PRAYER_PROTECT_MAGIC:  return "Protect Magic";
        case PRAYER_PROTECT_RANGED: return "Protect Ranged";
        case PRAYER_PROTECT_MELEE:  return "Protect Melee";
        case PRAYER_SMITE:          return "Smite";
        case PRAYER_REDEMPTION:     return "Redemption";
        default:                    return "None";
    }
}

static void gui_draw_stats(GuiState* gs, Player* p) {
    int ox = gs->panel_x + 4;
    int oy = gui_content_y(gs) + 4;

    /* OSRS skill grid: 3 columns, 6 visible rows for combat skills.
       cell dimensions scale to panel width. OSRS original: 62x32 in 190px panel. */
    int gap = 2;
    int cols = 3;
    int cw = (gs->panel_w - 8 - gap * (cols - 1)) / cols;
    int ch = 32;

    /* OSRS grid: row x col → (icon_idx, current, base) or -1 for empty.
       col 0: attack, strength, defence, ranged, prayer, magic
       col 1: hitpoints, then empty
       col 2: empty (non-combat) */
    int grid_icon[6][3];
    int grid_cur[6][3];
    int grid_base[6][3];
    memset(grid_icon, -1, sizeof(grid_icon));
    memset(grid_cur, 0, sizeof(grid_cur));
    memset(grid_base, 0, sizeof(grid_base));

    /* col 0: combat stats in OSRS order */
    grid_icon[0][0] = SKILL_ICON_ATTACK;    grid_cur[0][0] = p->current_attack;    grid_base[0][0] = p->base_attack;
    grid_icon[1][0] = SKILL_ICON_STRENGTH;  grid_cur[1][0] = p->current_strength;  grid_base[1][0] = p->base_strength;
    grid_icon[2][0] = SKILL_ICON_DEFENCE;   grid_cur[2][0] = p->current_defence;   grid_base[2][0] = p->base_defence;
    grid_icon[3][0] = SKILL_ICON_RANGED;    grid_cur[3][0] = p->current_ranged;    grid_base[3][0] = p->base_ranged;
    grid_icon[4][0] = SKILL_ICON_PRAYER;    grid_cur[4][0] = p->current_prayer;    grid_base[4][0] = p->base_prayer;
    grid_icon[5][0] = SKILL_ICON_MAGIC;     grid_cur[5][0] = p->current_magic;     grid_base[5][0] = p->base_magic;

    /* col 1: hitpoints at row 0, rest stays -1 (empty) */
    grid_icon[0][1] = SKILL_ICON_HITPOINTS; grid_cur[0][1] = p->current_hitpoints; grid_base[0][1] = p->base_hitpoints;

    /* draw the grid */
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < cols; c++) {
            int cx = ox + c * (cw + gap);
            int cy = oy + r * (ch + gap);
            if (grid_icon[r][c] >= 0) {
                gui_draw_skill_cell(gs, cx, cy, cw, ch,
                    grid_icon[r][c], grid_cur[r][c], grid_base[r][c]);
            } else {
                /* empty cell: just dark bg */
                DrawRectangle(cx, cy, cw, ch, (Color){20, 18, 14, 255});
                DrawRectangleLines(cx, cy, cw, ch, (Color){40, 36, 28, 255});
            }
        }
    }
    oy += 6 * (ch + gap) + 6;

    /* separator */
    int bar_w = gs->panel_w - 8;
    DrawLine(ox, oy, ox + bar_w, oy, GUI_BORDER);
    oy += 6;

    /* combat info below the skill grid */
    int lh = 17;
    gui_text_shadow(TextFormat("Gear: %s", gui_gear_name(p->current_gear)),
                    ox + 2, oy, 10, GUI_TEXT_ORANGE);
    oy += lh;
    gui_text_shadow(TextFormat("Max Hit: %d   Str Bonus: %d", p->gui_max_hit, p->gui_strength_bonus),
                    ox + 2, oy, 10, GUI_TEXT_YELLOW);
    oy += lh;
    gui_text_shadow(TextFormat("Speed: %d   Range: %d", p->gui_attack_speed, p->gui_attack_range),
                    ox + 2, oy, 10, GUI_TEXT_WHITE);
    oy += lh;
    gui_text_shadow(TextFormat("Prayer: %s", gui_prayer_name(p->prayer)),
                    ox + 2, oy, 10, GUI_TEXT_CYAN);
    oy += lh + 4;

    /* separator */
    DrawLine(ox, oy, ox + bar_w, oy, GUI_BORDER);
    oy += 6;

    /* consumables */
    gui_text_shadow(TextFormat("Brews: %d  Restores: %d", p->brew_doses, p->restore_doses),
                    ox + 2, oy, 10, GUI_TEXT_WHITE);
    oy += lh;
    gui_text_shadow(TextFormat("Bastion: %d  Stamina: %d",
                    p->combat_potion_doses, p->ranged_potion_doses),
                    ox + 2, oy, 10, GUI_TEXT_WHITE);
    oy += lh;

    /* special attack energy bar */
    int spec_bar_w = bar_w;
    int spec_bar_h = 14;
    float spec_pct = (float)p->special_energy / 100.0f;
    DrawRectangle(ox, oy, spec_bar_w, spec_bar_h, GUI_SPEC_DARK);
    DrawRectangle(ox, oy, (int)(spec_bar_w * spec_pct), spec_bar_h, GUI_SPEC_GREEN);
    DrawRectangleLines(ox, oy, spec_bar_w, spec_bar_h, GUI_BORDER);
    gui_text_shadow(TextFormat("Spec: %d%%", p->special_energy),
                    ox + 4, oy + 1, 10, GUI_TEXT_WHITE);
}


static void gui_cycle_entity(GuiState* gs) {
    if (gs->gui_entity_count <= 0) return;
    gs->gui_entity_idx = (gs->gui_entity_idx + 1) % gs->gui_entity_count;
}

static void gui_draw(GuiState* gs, Player* p) {
    int px = gs->panel_x;
    int py = gs->panel_y + gs->status_bar_h + gs->tab_h;  /* content area starts after status + tabs */
    int pw = gs->panel_w;
    int ph = gs->panel_h - gs->status_bar_h - gs->tab_h;  /* remaining height for content */

    /* draw real OSRS stone panel background, stretched to fill content area (single draw, not tiled) */
    if (gs->chrome_loaded && gs->side_panel_bg.id != 0) {
        Rectangle src = { 0, 0, (float)gs->side_panel_bg.width, (float)gs->side_panel_bg.height };
        Rectangle dst = { (float)px, (float)py, (float)pw, (float)ph };
        DrawTexturePro(gs->side_panel_bg, src, dst, (Vector2){0,0}, 0.0f, WHITE);
    } else {
        DrawRectangle(px, py, pw, ph, GUI_BG_DARK);
    }

    /* entity selector header */
    if (gs->gui_entity_count > 1) {
        int hx = px + 4;
        int hy = py + 2;
        const char* etype = (p->entity_type == ENTITY_NPC) ? "NPC" : "Player";
        gui_text_shadow(TextFormat("[G] %s %d/%d", etype,
                        gs->gui_entity_idx + 1, gs->gui_entity_count),
                        hx, hy, 8, GUI_TEXT_ORANGE);
    }

    /* draw status bar (HP/prayer/spec) above the tab row */
    gui_draw_status_bar(gs, p);

    /* draw tab bar at top (after status bar) */
    gui_draw_tab_bar(gs);

    /* dispatch to active tab content */
    switch (gs->active_tab) {
        case GUI_TAB_COMBAT:    gui_draw_combat(gs, p);    break;
        case GUI_TAB_INVENTORY: gui_draw_inventory(gs, p); break;
        case GUI_TAB_EQUIPMENT: gui_draw_equipment(gs, p); break;
        case GUI_TAB_PRAYER:    gui_draw_prayer(gs, p);    break;
        case GUI_TAB_SPELLBOOK: gui_draw_spellbook(gs, p); break;
        case GUI_TAB_STATS:     gui_draw_stats(gs, p);  break;
        case GUI_TAB_QUESTS:    /* empty tab */ break;
        default: break;
    }
}

#endif /* OSRS_GUI_H */
