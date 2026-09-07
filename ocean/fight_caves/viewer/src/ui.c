#include "ui.h"
#include "fc_minimap.h"
#include "ui_reference.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OSRS_ORANGE ((Color){255, 152, 31, 255})
#define OSRS_YELLOW ((Color){255, 255, 0, 255})
#define OSRS_GREEN  ((Color){0, 255, 0, 255})
#define OSRS_RED    ((Color){255, 40, 25, 255})
#define OSRS_BLUE   ((Color){90, 170, 255, 255})
#define OSRS_PANEL  ((Color){31, 25, 18, 232})
#define OSRS_TAB_PRESS_SECONDS 0.12f
#define RUNEC_UI_SPELL_COUNT 80
#define RUNEC_UI_SPELL_COLS 8
#define RUNEC_UI_SPELL_X0 4
#define RUNEC_UI_SPELL_Y0 6
#define RUNEC_UI_SPELL_STEP_X 23
#define RUNEC_UI_SPELL_STEP_Y 24
#define RUNEC_UI_SPELL_ICON_SIZE 22

typedef struct RuneCUiLayout {
    Rectangle chat;
    Rectangle chat_messages;
    Rectangle chat_controls;
    Rectangle map;
    Rectangle minimap;
    Rectangle compass;
    Rectangle worldmap_button;
    Rectangle hp_orb;
    Rectangle prayer_orb;
    Rectangle run_orb;
    Rectangle spec_orb;
    Rectangle side;
    Rectangle side_bg;
    Rectangle side_content;
    Rectangle tab[RUNEC_UI_TAB_COUNT];
} RuneCUiLayout;

static const char *g_prayer_names[25] = {
    "Thick Skin", "Burst of Strength", "Clarity of Thought", "Sharp Eye",
    "Mystic Will", "Rock Skin", "Superhuman Strength", "Improved Reflexes",
    "Rapid Restore", "Rapid Heal", "Protect Item", "Hawk Eye",
    "Mystic Lore", "Steel Skin", "Ultimate Strength", "Incredible Reflexes",
    "Protect from Magic", "Protect from Missiles", "Protect from Melee",
    "Eagle Eye", "Mystic Might", "Retribution", "Redemption",
    "Smite", "Preserve",
};

typedef struct RuneCUiSpellSlotRef {
    const char *name;
    int standard_icon_frame;
} RuneCUiSpellSlotRef;

static const RuneCUiSpellSlotRef g_standard_spell_slots[RUNEC_UI_SPELL_COUNT] = {
    {"Lumbridge Home Teleport", 70},
    {"Wind Strike", 0},
    {"Confuse", 50},
    {"Crossbow Bolt Enchantments", 65},
    {"Water Strike", 1},
    {"Lvl-1 Enchant", 33},
    {"Earth Strike", 2},
    {"Weaken", 51},
    {"Fire Strike", 3},
    {"Bones to Bananas", 45},
    {"Wind Bolt", 4},
    {"Curse", 52},
    {"Bind", 56},
    {"Low Level Alchemy", 48},
    {"Water Bolt", 5},
    {"Varrock Teleport", 20},
    {"Lvl-2 Enchant", 34},
    {"Earth Bolt", 6},
    {"Lumbridge Teleport", 21},
    {"Telekinetic Grab", 46},
    {"Fire Bolt", 7},
    {"Falador Teleport", 22},
    {"Crumble Undead", 47},
    {"Teleport to House", 23},
    {"Wind Blast", 8},
    {"Superheat Item", 68},
    {"Camelot Teleport", 24},
    {"Water Blast", 9},
    {"Lvl-3 Enchant", 35},
    {"Iban Blast", 40},
    {"Snare", 57},
    {"Magic Dart", 66},
    {"Ardougne Teleport", 25},
    {"Earth Blast", 10},
    {"High Level Alchemy", 49},
    {"Charge Water Orb", 61},
    {"Lvl-4 Enchant", 36},
    {"Watchtower Teleport", 26},
    {"Fire Blast", 11},
    {"Charge Earth Orb", 62},
    {"Bones to Peaches", 69},
    {"Saradomin Strike", 43},
    {"Claws of Guthix", 42},
    {"Flames of Zamorak", 41},
    {"Trollheim Teleport", 27},
    {"Wind Wave", 12},
    {"Charge Fire Orb", 63},
    {"Water Wave", 13},
    {"Teleport to Ape Atoll", 28},
    {"Earth Wave", 14},
    {"Lvl-5 Enchant", 37},
    {"Kourend Castle Teleport", 29},
    {"Charge Air Orb", 64},
    {"Vulnerability", 53},
    {"Lvl-6 Enchant", 38},
    {"Teleport to Target", 67},
    {"Enfeeble", 54},
    {"Teleother Lumbridge", 30},
    {"Fire Wave", 15},
    {"Entangle", 58},
    {"Stun", 55},
    {"Charge", 44},
    {"Wind Surge", 16},
    {"Teleother Falador", 31},
    {"Water Surge", 17},
    {"Tele Block", 60},
    {"Lvl-7 Enchant", 39},
    {"Earth Surge", 18},
    {"Teleother Camelot", 32},
    {"Fire Surge", 19},
    {"Civitas illa Fortis Teleport", 72},
    {"Jewellery Enchantments", 71},
    {"Monster Inspect", 73},
    {"Summon Boat", 75},
    {"Teleport to Boat", 74},
    {"Alchemic Divergence", 77},
    {"Alchemic Convergence", 78},
    {"Minigame Teleport", 76},
    {"League Home Teleport", 79},
    {NULL, -1},
};

static const char *spell_name(int slot) {
    if (slot >= 0
            && slot < (int)(sizeof(g_standard_spell_slots)
                            / sizeof(g_standard_spell_slots[0]))
            && g_standard_spell_slots[slot].name) {
        return g_standard_spell_slots[slot].name;
    }
    return TextFormat("Spell %d", slot + 1);
}

static const char *g_equipment_names[RUNEC_UI_EQUIP_SLOT_COUNT] = {
    "Head", "Cape", "Neck", "Weapon", "Body", "Shield", "Ammo",
    "Legs", "Unused", "Hands", "Feet", "Unused", "Ring", "Quiver",
};

static const Rectangle g_equipment_offsets[RUNEC_UI_EQUIP_SLOT_COUNT] = {
    {77, 4, 36, 36},
    {36, 43, 36, 36},
    {77, 43, 36, 36},
    {21, 82, 36, 36},
    {77, 82, 36, 36},
    {133, 82, 36, 36},
    {133, 43, 36, 36},
    {77, 122, 36, 36},
    {-1000, -1000, 0, 0},
    {21, 162, 36, 36},
    {77, 162, 36, 36},
    {-1000, -1000, 0, 0},
    {133, 162, 36, 36},
    {118, 43, 36, 36},
};

static const char *g_worn_icon_names[RUNEC_UI_EQUIP_SLOT_COUNT] = {
    "wornicons_0", "wornicons_1", "wornicons_2", "wornicons_3",
    "wornicons_4", "wornicons_5", "wornicons_10", "wornicons_6",
    NULL, "wornicons_7", "wornicons_8", NULL, "wornicons_9", "wornicons_11",
};

typedef struct RuneCUiCombatStyleDef {
    int visible;
    int style_index;
    const char *label;
    const char *mode;
    const char *icon_asset;
} RuneCUiCombatStyleDef;

typedef struct RuneCUiCombatProfile {
    int osrs_category;
    const RuneCUiCombatStyleDef styles[RUNEC_UI_COMBAT_STYLE_COUNT];
} RuneCUiCombatProfile;

#define COMBAT_STYLE(style_index, label, mode, icon_asset) \
    {1, style_index, label, mode, icon_asset}
#define COMBAT_STYLE_HIDDEN \
    {0, 0, "", "", ""}

/*
 * Exact b237 combat_interface_weapon_category DB rows from the local
 * Joshua-F dump. Core owns the selected style; the viewer owns this
 * presentation mapping so weapon tabs use the same labels/icons as OSRS.
 */
static const RuneCUiCombatProfile g_combat_profiles[] = {
    {0, {
        COMBAT_STYLE(0, "Punch", "Accurate", "combaticons_14"),
        COMBAT_STYLE(1, "Kick", "Aggressive", "combaticons_15"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_16"),
        COMBAT_STYLE_HIDDEN,
    }},
    {1, {
        COMBAT_STYLE(0, "Chop", "Accurate", "combaticons_1"),
        COMBAT_STYLE(1, "Hack", "Aggressive", "combaticons_2"),
        COMBAT_STYLE(2, "Smash", "Aggressive", "combaticons_3"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_0"),
    }},
    {2, {
        COMBAT_STYLE(0, "Pound", "Accurate", "combaticons2_2"),
        COMBAT_STYLE(1, "Pummel", "Aggressive", "combaticons2_3"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons2_0"),
        COMBAT_STYLE_HIDDEN,
    }},
    {3, {
        COMBAT_STYLE(0, "Accurate", "Accurate", "combaticons2_15"),
        COMBAT_STYLE(1, "Rapid", "Rapid", "combaticons2_16"),
        COMBAT_STYLE(3, "Longrange", "Longrange", "combaticons2_17"),
        COMBAT_STYLE_HIDDEN,
    }},
    {4, {
        COMBAT_STYLE(0, "Chop", "Accurate", "combaticons3_6"),
        COMBAT_STYLE(1, "Slash", "Aggressive", "combaticons3_5"),
        COMBAT_STYLE(2, "Lunge", "Controlled", "combaticons3_4"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons3_7"),
    }},
    {5, {
        COMBAT_STYLE(0, "Accurate", "Accurate", "combaticons2_5"),
        COMBAT_STYLE(1, "Rapid", "Rapid", "combaticons2_6"),
        COMBAT_STYLE(3, "Longrange", "Longrange", "combaticons2_7"),
        COMBAT_STYLE_HIDDEN,
    }},
    {6, {
        COMBAT_STYLE(0, "Scorch", "Accurate", "combaticons3_16"),
        COMBAT_STYLE(1, "Flare", "Aggressive", "combaticons3_17"),
        COMBAT_STYLE(2, "Blaze", "Defensive", "combaticons3_18"),
        COMBAT_STYLE_HIDDEN,
    }},
    {7, {
        COMBAT_STYLE(0, "Short fuse", "Accurate", "combaticons3_15"),
        COMBAT_STYLE(1, "Medium fuse", "Rapid", "combaticons3_9"),
        COMBAT_STYLE(3, "Long fuse", "Longrange", "combaticons3_8"),
        COMBAT_STYLE_HIDDEN,
    }},
    {8, {
        COMBAT_STYLE(0, "Aim and Fire", "Accurate", "prayeron_13"),
        COMBAT_STYLE(1, "Kick", "Aggressive", "combaticons_15"),
        COMBAT_STYLE_HIDDEN,
        COMBAT_STYLE_HIDDEN,
    }},
    {9, {
        COMBAT_STYLE(0, "Chop", "Accurate", "combaticons_6"),
        COMBAT_STYLE(1, "Slash", "Aggressive", "combaticons_5"),
        COMBAT_STYLE(2, "Lunge", "Controlled", "combaticons_7"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_4"),
    }},
    {10, {
        COMBAT_STYLE(0, "Chop", "Accurate", "combaticons_6"),
        COMBAT_STYLE(1, "Slash", "Aggressive", "combaticons_5"),
        COMBAT_STYLE(2, "Smash", "Aggressive", "combaticons_5"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_4"),
    }},
    {11, {
        COMBAT_STYLE(0, "Spike", "Accurate", "combaticons3_1"),
        COMBAT_STYLE(1, "Impale", "Aggressive", "combaticons3_3"),
        COMBAT_STYLE(2, "Smash", "Aggressive", "combaticons3_2"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons3_0"),
    }},
    {12, {
        COMBAT_STYLE(0, "Jab", "Controlled", "combaticons3_11"),
        COMBAT_STYLE(1, "Swipe", "Aggressive", "combaticons3_12"),
        COMBAT_STYLE(3, "Fend", "Defensive", "combaticons3_10"),
        COMBAT_STYLE_HIDDEN,
    }},
    {13, {
        COMBAT_STYLE(0, "Bash", "Accurate", "combaticons2_13"),
        COMBAT_STYLE(1, "Pound", "Aggressive", "combaticons2_14"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_19"),
        COMBAT_STYLE_HIDDEN,
    }},
    {14, {
        COMBAT_STYLE(0, "Reap", "Accurate", "combaticons2_19"),
        COMBAT_STYLE(1, "Chop", "Aggressive", "combaticons2_9"),
        COMBAT_STYLE(2, "Jab", "Controlled", "combaticons2_18"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons2_8"),
    }},
    {15, {
        COMBAT_STYLE(0, "Lunge", "Controlled", "combaticons_8"),
        COMBAT_STYLE(1, "Swipe", "Controlled", "combaticons_18"),
        COMBAT_STYLE(2, "Pound", "Controlled", "combaticons_9"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_17"),
    }},
    {16, {
        COMBAT_STYLE(0, "Pound", "Accurate", "combaticons_13"),
        COMBAT_STYLE(1, "Pummel", "Aggressive", "combaticons_11"),
        COMBAT_STYLE(2, "Spike", "Controlled", "combaticons_12"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_10"),
    }},
    {17, {
        COMBAT_STYLE(0, "Stab", "Accurate", "combaticons_7"),
        COMBAT_STYLE(1, "Lunge", "Aggressive", "combaticons_6"),
        COMBAT_STYLE(2, "Slash", "Controlled", "combaticons_5"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_4"),
    }},
    {18, {
        COMBAT_STYLE(0, "Bash", "Accurate", "combaticons2_13"),
        COMBAT_STYLE(1, "Pound", "Aggressive", "combaticons2_14"),
        COMBAT_STYLE(3, "Focus", "Defensive", "combaticons_19"),
        COMBAT_STYLE_HIDDEN,
    }},
    {19, {
        COMBAT_STYLE(0, "Accurate", "Accurate", "combaticons2_10"),
        COMBAT_STYLE(1, "Rapid", "Rapid", "combaticons2_11"),
        COMBAT_STYLE(3, "Longrange", "Longrange", "combaticons2_12"),
        COMBAT_STYLE_HIDDEN,
    }},
    {20, {
        COMBAT_STYLE(0, "Flick", "Accurate", "combaticons3_13"),
        COMBAT_STYLE(1, "Lash", "Controlled", "combaticons3_14"),
        COMBAT_STYLE(3, "Deflect", "Defensive", "combaticons3_13"),
        COMBAT_STYLE_HIDDEN,
    }},
    {21, {
        COMBAT_STYLE(0, "Jab", "Accurate", "combaticons2_13"),
        COMBAT_STYLE(1, "Swipe", "Aggressive", "combaticons2_14"),
        COMBAT_STYLE(3, "Fend", "Defensive", "combaticons_19"),
        COMBAT_STYLE_HIDDEN,
    }},
    {22, {
        COMBAT_STYLE(0, "Jab", "Accurate", "combaticons2_13"),
        COMBAT_STYLE(1, "Swipe", "Aggressive", "combaticons2_14"),
        COMBAT_STYLE(3, "Fend", "Defensive", "combaticons_19"),
        COMBAT_STYLE_HIDDEN,
    }},
    {24, {
        COMBAT_STYLE(0, "Accurate", "Accurate", "combaticons2_10"),
        COMBAT_STYLE(1, "Accurate", "Accurate", "combaticons2_10"),
        COMBAT_STYLE(3, "Longrange", "Longrange", "combaticons2_12"),
        COMBAT_STYLE_HIDDEN,
    }},
    {25, {
        COMBAT_STYLE(0, "Lunge", "Controlled", "combaticons_8"),
        COMBAT_STYLE(1, "Swipe", "Controlled", "combaticons_18"),
        COMBAT_STYLE(2, "Pound", "Controlled", "combaticons_9"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_17"),
    }},
    {26, {
        COMBAT_STYLE(0, "Jab", "Controlled", "combaticons3_11"),
        COMBAT_STYLE(1, "Swipe", "Aggressive", "combaticons3_12"),
        COMBAT_STYLE(3, "Fend", "Defensive", "combaticons3_10"),
        COMBAT_STYLE_HIDDEN,
    }},
    {27, {
        COMBAT_STYLE(0, "Pound", "Accurate", "combaticons2_2"),
        COMBAT_STYLE(1, "Pummel", "Aggressive", "combaticons2_3"),
        COMBAT_STYLE(2, "Smash", "Aggressive", "combaticons2_0"),
        COMBAT_STYLE_HIDDEN,
    }},
    {28, {
        COMBAT_STYLE(1, "Pummel", "Aggressive", "combaticons2_1"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons2_0"),
        COMBAT_STYLE_HIDDEN,
        COMBAT_STYLE_HIDDEN,
    }},
    {29, {
        COMBAT_STYLE(0, "Accurate", "Accurate", "combaticons2_10"),
        COMBAT_STYLE(1, "Accurate", "Accurate", "combaticons2_10"),
        COMBAT_STYLE(3, "Longrange", "Longrange", "combaticons2_12"),
        COMBAT_STYLE_HIDDEN,
    }},
    {30, {
        COMBAT_STYLE(0, "Stab", "Accurate", "combaticons_7"),
        COMBAT_STYLE(1, "Lunge", "Aggressive", "combaticons_6"),
        COMBAT_STYLE(2, "Pound", "Controlled", "combaticons_5"),
        COMBAT_STYLE(3, "Block", "Defensive", "combaticons_4"),
    }},
};

#undef COMBAT_STYLE
#undef COMBAT_STYLE_HIDDEN

static void copy_text(char *dst, size_t cap, const char *src) {
    if (cap == 0)
        return;
    if (!src)
        src = "";
    snprintf(dst, cap, "%s", src);
}

static const RuneCUiCombatProfile *combat_profile_for_osrs_category(int category) {
    int count = (int)(sizeof(g_combat_profiles) / sizeof(g_combat_profiles[0]));
    for (int i = 0; i < count; i++) {
        if (g_combat_profiles[i].osrs_category == category)
            return &g_combat_profiles[i];
    }
    return &g_combat_profiles[0];
}

static int osrs_combat_category_from_core_weapon(int core_weapon_category) {
    switch (core_weapon_category) {
    case 0: return 0;   /* unarmed */
    case 1: return 10;  /* 2h sword */
    case 2: return 1;   /* axe */
    case 4: return 2;   /* blunt */
    case 5: return 27;  /* bludgeon */
    case 6: return 28;  /* bulwark */
    case 7: return 7;   /* chinchompa/grenade */
    case 8: return 4;   /* claw */
    case 9: return 5;   /* crossbow */
    case 10: return 20; /* whip */
    case 11: return 6;  /* fixed device */
    case 12: return 8;  /* gun */
    case 13: return 11; /* pickaxe */
    case 14: return 12; /* polearm */
    case 15: return 13; /* polestaff */
    case 16: return 24; /* powered staff */
    case 17: return 14; /* scythe */
    case 18: return 9;  /* slash sword */
    case 19: return 15; /* spear */
    case 20: return 16; /* spiked */
    case 21: return 17; /* stab sword */
    case 22: return 18; /* staff */
    case 23: return 19; /* thrown */
    case 24: return 10; /* two-handed sword */
    case 25: return 3;  /* bow */
    case 26: return 6;  /* salamander */
    case 27: return 6;  /* multi-style fallback */
    case 28: return 29; /* powered wand */
    case 29: return 21; /* bladed staff */
    case 30: return 30; /* partisan */
    default: return 0;
    }
}

static const RuneCUiCombatStyleOption *combat_style_option_by_index(
    const RuneCUiState *ui,
    int style_index) {
    if (!ui)
        return NULL;
    for (int i = 0; i < RUNEC_UI_COMBAT_STYLE_COUNT; i++) {
        const RuneCUiCombatStyleOption *option = &ui->combat_styles[i];
        if (option->visible && option->style_index == style_index)
            return option;
    }
    return NULL;
}

static const RuneCUiCombatStyleOption *selected_combat_style_option(
    const RuneCUiState *ui) {
    const RuneCUiCombatStyleOption *option =
        combat_style_option_by_index(ui, ui ? ui->selected_combat_style : 0);
    if (option)
        return option;
    if (ui && ui->selected_combat_style == 2)
        option = combat_style_option_by_index(ui, 3);
    if (option)
        return option;
    if (!ui)
        return NULL;
    for (int i = 0; i < RUNEC_UI_COMBAT_STYLE_COUNT; i++) {
        if (ui->combat_styles[i].visible)
            return &ui->combat_styles[i];
    }
    return NULL;
}

static int combat_style_option_selected(const RuneCUiState *ui,
                                        const RuneCUiCombatStyleOption *option) {
    if (!ui || !option || !option->visible)
        return 0;
    if (ui->selected_combat_style == option->style_index)
        return 1;
    return ui->selected_combat_style == 2
        && option->style_index == 3
        && combat_style_option_by_index(ui, 2) == NULL;
}

void runec_ui_set_combat_weapon_name(RuneCUiState *ui, const char *name) {
    if (!ui)
        return;
    copy_text(ui->combat_weapon_name, sizeof(ui->combat_weapon_name),
              name && name[0] ? name : "Unarmed");
}

void runec_ui_set_combat_style_profile(RuneCUiState *ui, int core_weapon_category) {
    if (!ui)
        return;
    ui->combat_weapon_category = core_weapon_category;
    int osrs_category = osrs_combat_category_from_core_weapon(core_weapon_category);
    const RuneCUiCombatProfile *profile =
        combat_profile_for_osrs_category(osrs_category);
    for (int i = 0; i < RUNEC_UI_COMBAT_STYLE_COUNT; i++) {
        const RuneCUiCombatStyleDef *src = &profile->styles[i];
        RuneCUiCombatStyleOption *dst = &ui->combat_styles[i];
        dst->visible = src->visible;
        dst->style_index = src->style_index;
        copy_text(dst->label, sizeof(dst->label), src->label);
        copy_text(dst->mode, sizeof(dst->mode), src->mode);
        copy_text(dst->icon_asset, sizeof(dst->icon_asset), src->icon_asset);
    }
    if (!selected_combat_style_option(ui)) {
        for (int i = 0; i < RUNEC_UI_COMBAT_STYLE_COUNT; i++) {
            if (ui->combat_styles[i].visible) {
                ui->selected_combat_style = ui->combat_styles[i].style_index;
                break;
            }
        }
    }
}

static float fmin2(float a, float b) {
    return a < b ? a : b;
}

const char *runec_ui_tab_name(RuneCUiTab tab) {
    switch (tab) {
    case RUNEC_UI_TAB_COMBAT: return "Combat";
    case RUNEC_UI_TAB_SKILLS: return "Skills";
    case RUNEC_UI_TAB_QUESTS: return "Quests";
    case RUNEC_UI_TAB_INVENTORY: return "Inventory";
    case RUNEC_UI_TAB_EQUIPMENT: return "Equipment";
    case RUNEC_UI_TAB_PRAYER: return "Prayer";
    case RUNEC_UI_TAB_SPELLBOOK: return "Spellbook";
    case RUNEC_UI_TAB_SETTINGS: return "Settings";
    case RUNEC_UI_TAB_CLAN_CHAT: return "Clan Chat";
    case RUNEC_UI_TAB_FRIENDS: return "Friends";
    default: return "Unknown";
    }
}

static void ui_layout(int screen_w, int screen_h, RuneCUiLayout *out) {
    memset(out, 0, sizeof(*out));

    out->chat = (Rectangle){0, (float)screen_h - RUNEC_OSRS_CHAT_H,
                            RUNEC_OSRS_CHAT_W, RUNEC_OSRS_CHAT_H};
    out->chat_messages = (Rectangle){7, out->chat.y + 7, 506, 126};
    out->chat_controls = (Rectangle){0, out->chat.y + 142, 519, 23};

    out->map = (Rectangle){(float)screen_w - RUNEC_OSRS_MAP_CONTAINER_W, 0,
                           RUNEC_OSRS_MAP_CONTAINER_W, RUNEC_OSRS_MAP_CONTAINER_H};
    out->minimap = (Rectangle){out->map.x + RUNEC_OSRS_MINIMAP_X,
                               out->map.y + RUNEC_OSRS_MINIMAP_Y,
                               RUNEC_OSRS_MINIMAP_W, RUNEC_OSRS_MINIMAP_H};
    out->compass = (Rectangle){out->map.x + RUNEC_OSRS_COMPASS_X,
                               out->map.y + RUNEC_OSRS_COMPASS_Y,
                               RUNEC_OSRS_COMPASS_W, RUNEC_OSRS_COMPASS_H};
    out->hp_orb = (Rectangle){out->map.x + RUNEC_OSRS_ORBS_X + RUNEC_OSRS_HP_X,
                              out->map.y + RUNEC_OSRS_ORBS_Y + RUNEC_OSRS_HP_Y, 57, 34};
    out->prayer_orb = (Rectangle){out->map.x + RUNEC_OSRS_ORBS_X + RUNEC_OSRS_PRAYER_X,
                                  out->map.y + RUNEC_OSRS_ORBS_Y + RUNEC_OSRS_PRAYER_Y, 57, 34};
    out->run_orb = (Rectangle){out->map.x + RUNEC_OSRS_ORBS_X + RUNEC_OSRS_RUN_X,
                               out->map.y + RUNEC_OSRS_ORBS_Y + RUNEC_OSRS_RUN_Y, 57, 34};
    out->spec_orb = (Rectangle){out->map.x + RUNEC_OSRS_ORBS_X + RUNEC_OSRS_SPEC_X,
                                out->map.y + RUNEC_OSRS_ORBS_Y + RUNEC_OSRS_SPEC_Y, 57, 34};
    out->worldmap_button = (Rectangle){out->map.x + RUNEC_OSRS_ORBS_X + RUNEC_OSRS_WORLDMAP_X,
                                       out->map.y + RUNEC_OSRS_ORBS_Y + RUNEC_OSRS_WORLDMAP_Y,
                                       30, 30};

    out->side = (Rectangle){(float)screen_w - RUNEC_OSRS_SIDE_MENU_W,
                            (float)screen_h - RUNEC_OSRS_SIDE_MENU_H,
                            RUNEC_OSRS_SIDE_MENU_W, RUNEC_OSRS_SIDE_MENU_H};
    out->side_bg = out->side;
    out->side_content = (Rectangle){out->side.x + RUNEC_OSRS_SIDE_CONTENT_X,
                                    out->side.y + RUNEC_OSRS_SIDE_CONTENT_Y,
                                    RUNEC_OSRS_SIDE_CONTENT_W,
                                    RUNEC_OSRS_SIDE_CONTENT_H};

    for (int i = 0; i < (int)(sizeof(RUNEC_OSRS_SIDE_STONES) / sizeof(RUNEC_OSRS_SIDE_STONES[0])); i++) {
        const RuneCUiStoneRef *ref = &RUNEC_OSRS_SIDE_STONES[i];
        if (ref->logical_tab < 0 || ref->logical_tab >= RUNEC_UI_TAB_COUNT)
            continue;
        float row_y = out->side.y + (i < 7 ? RUNEC_OSRS_SIDE_TOP_Y : RUNEC_OSRS_SIDE_BOTTOM_Y);
        out->tab[ref->logical_tab] =
            (Rectangle){out->side.x + ref->rect.x, row_y + ref->rect.y,
                        ref->rect.width, ref->rect.height};
    }
}

static int mouse_over_ui(const RuneCUiLayout *layout, Vector2 mouse) {
    if (CheckCollisionPointRec(mouse, layout->chat)
        || CheckCollisionPointRec(mouse, layout->side)
        || CheckCollisionPointRec(mouse, layout->map))
        return 1;
    for (int i = 0; i < RUNEC_UI_TAB_COUNT; i++) {
        if (CheckCollisionPointRec(mouse, layout->tab[i]))
            return 1;
    }
    return 0;
}

static void clear_intent(RuneCUiState *ui) {
    memset(&ui->last_intent, 0, sizeof(ui->last_intent));
}

static void set_context(RuneCUiState *ui, Vector2 pos, const char *title,
                        const char **actions, int action_count) {
    ui->context_open = 1;
    ui->context_pos = pos;
    copy_text(ui->context_title, sizeof(ui->context_title), title);
    ui->context_source_kind = RUNEC_UI_CONTEXT_NONE;
    ui->context_source_slot = -1;
    ui->context_source_item_id = 0;
    if (action_count > RUNEC_UI_CONTEXT_ACTIONS)
        action_count = RUNEC_UI_CONTEXT_ACTIONS;
    ui->context_action_count = action_count;
    for (int i = 0; i < action_count; i++) {
        copy_text(ui->context_actions[i], sizeof(ui->context_actions[i]), actions[i]);
    }
}

static void set_context_source(RuneCUiState *ui,
                               RuneCUiContextSourceKind source_kind,
                               int source_slot,
                               uint32_t source_item_id) {
    ui->context_source_kind = source_kind;
    ui->context_source_slot = source_slot;
    ui->context_source_item_id = source_item_id;
}

void runec_ui_clear_selected_target(RuneCUiState *ui) {
    if (!ui)
        return;
    memset(&ui->selected_target, 0, sizeof(ui->selected_target));
    ui->selected_target.source_slot = -1;
}

static void set_selected_item_target(RuneCUiState *ui, int slot) {
    if (!ui || slot < 0 || slot >= RUNEC_UI_INV_SLOT_COUNT
            || !ui->inventory[slot].enabled)
        return;
    ui->selected_target.kind = RUNEC_UI_SELECTED_ITEM;
    ui->selected_target.source_slot = slot;
    ui->selected_target.source_item_id = ui->inventory[slot].item_id;
    copy_text(ui->selected_target.label, sizeof(ui->selected_target.label),
              ui->inventory[slot].label);
    copy_text(ui->selected_target.verb, sizeof(ui->selected_target.verb), "Use");
}

static void set_selected_spell_target(RuneCUiState *ui, int slot,
                                      const char *name) {
    if (!ui || slot < 0)
        return;
    ui->selected_target.kind = RUNEC_UI_SELECTED_SPELL;
    ui->selected_target.source_slot = slot;
    ui->selected_target.source_item_id = 0;
    copy_text(ui->selected_target.label, sizeof(ui->selected_target.label), name);
    copy_text(ui->selected_target.verb, sizeof(ui->selected_target.verb), "Cast");
}

void runec_ui_init(RuneCUiState *ui) {
    memset(ui, 0, sizeof(*ui));
    ui->active_tab = RUNEC_UI_TAB_SKILLS;
    ui->selected_inventory_slot = -1;
    ui->selected_equipment_slot = -1;
    ui->context_source_slot = -1;
    ui->selected_target.source_slot = -1;
    ui->drag.source_slot = -1;
    ui->selected_combat_style = 0;
    ui->auto_retaliate = 1;
    ui->special_attack_enabled = 0;
    ui->special_attack_energy = 100;
    runec_ui_set_combat_weapon_name(ui, "Abyssal whip");
    runec_ui_set_combat_style_profile(ui, 10);
    ui->hitpoints = 99;
    ui->hitpoints_max = 99;
    ui->prayer_points = 77;
    ui->prayer_points_max = 77;
    ui->run_energy = 100;
    ui->run_enabled = 1;
    ui->combat_level = 126;
    for (int i = 0; i < RUNEC_UI_SKILL_COUNT; i++) {
        ui->skill_current[i] = 1;
        ui->skill_base[i] = 1;
    }
    ui->skill_current[8] = 10;
    ui->skill_base[8] = 10;
    ui->skill_total = 33;
    const char *start_tab = getenv("RUNEC_UI_START_TAB");
    if (start_tab && start_tab[0]) {
        for (int i = 0; i < RUNEC_UI_TAB_COUNT; i++) {
            if (strcmp(start_tab, runec_ui_tab_name((RuneCUiTab)i)) == 0) {
                ui->active_tab = (RuneCUiTab)i;
                break;
            }
        }
        if (start_tab[0] >= '0' && start_tab[0] <= '9') {
            int tab_index = atoi(start_tab);
            if (tab_index >= 0 && tab_index < RUNEC_UI_TAB_COUNT)
                ui->active_tab = (RuneCUiTab)tab_index;
        }
    }

    runec_ui_assets_load(&ui->assets);
    Image minimap = GenImageColor(152, 152, BLANK);
    ui->minimap_texture = LoadTextureFromImage(minimap);
    UnloadImage(minimap);
    if (ui->minimap_texture.id != 0) {
        SetTextureFilter(ui->minimap_texture, TEXTURE_FILTER_POINT);
        ui->minimap_texture_ready = 1;
    }

    ui->inventory[0] = (RuneCUiSlot){6570, 6570, 1, "Fire cape", 1};
    ui->inventory[1] = (RuneCUiSlot){21295, 21295, 1, "Infernal cape", 1};
    ui->inventory[2] = (RuneCUiSlot){1042, 1042, 1, "Blue partyhat", 1};
    ui->inventory[3] = (RuneCUiSlot){1044, 1044, 1, "Green partyhat", 1};
    ui->inventory[4] = (RuneCUiSlot){1046, 1046, 1, "Purple partyhat", 1};
    ui->inventory[5] = (RuneCUiSlot){1048, 1048, 1, "White partyhat", 1};
    ui->inventory[6] = (RuneCUiSlot){4151, 4151, 1, "Abyssal whip", 1};
    ui->inventory[7] = (RuneCUiSlot){11802, 11802, 1, "Armadyl godsword", 1};
    ui->inventory[8] = (RuneCUiSlot){11832, 11832, 1, "Bandos chestplate", 1};
    ui->inventory[9] = (RuneCUiSlot){11834, 11834, 1, "Bandos tassets", 1};
    ui->inventory[10] = (RuneCUiSlot){26382, 26382, 1, "Torva full helm", 1};
    ui->inventory[11] = (RuneCUiSlot){26384, 26384, 1, "Torva platebody", 1};
    ui->inventory[12] = (RuneCUiSlot){26386, 26386, 1, "Torva platelegs", 1};
    ui->inventory[13] = (RuneCUiSlot){10350, 10350, 1, "3a full helmet", 1};
    ui->inventory[14] = (RuneCUiSlot){10348, 10348, 1, "3a platebody", 1};
    ui->inventory[15] = (RuneCUiSlot){10346, 10346, 1, "3a platelegs", 1};
    ui->inventory[16] = (RuneCUiSlot){10352, 10352, 1, "3a kiteshield", 1};
    ui->inventory[17] = (RuneCUiSlot){995, 1004, 10000000, "Coins", 1};
    ui->equipment[0] = (RuneCUiSlot){11826, 11826, 1, "Helm", 1};
    ui->equipment[3] = (RuneCUiSlot){4151, 4151, 1, "Abyssal whip", 1};
    ui->equipment[4] = (RuneCUiSlot){11828, 11828, 1, "Body", 1};
    ui->equipment[7] = (RuneCUiSlot){11830, 11830, 1, "Legs", 1};

}

void runec_ui_shutdown(RuneCUiState *ui) {
    if (ui->minimap_texture_ready) {
        UnloadTexture(ui->minimap_texture);
        ui->minimap_texture_ready = 0;
    }
    for (int i = 0; i < ui->item_icon_count; i++) {
        if (ui->item_icons[i].ready && ui->item_icons[i].texture.id != 0)
            UnloadTexture(ui->item_icons[i].texture);
    }
    ui->item_icon_count = 0;
    runec_ui_assets_unload(&ui->assets);
}

void runec_ui_clear_minimap(RuneCUiState *ui) {
    ui->minimap_dot_count = 0;
}

void runec_ui_add_minimap_dot(RuneCUiState *ui, float dx, float dy,
                              RuneCUiMinimapDotKind kind) {
    if (ui->minimap_dot_count >= RUNEC_UI_MINIMAP_DOTS)
        return;
    ui->minimap_dots[ui->minimap_dot_count++] =
        (RuneCUiMinimapDot){dx, dy, kind};
}

void runec_ui_update_minimap(RuneCUiState *ui, const Color *pixels,
                             int width, int height) {
    if (!ui->minimap_texture_ready || !pixels || width != 152 || height != 152)
        return;
    UpdateTexture(ui->minimap_texture, pixels);
}

void runec_ui_set_minimap_rotation(RuneCUiState *ui, float radians) {
    if (!ui) return;
    ui->minimap_rotation = radians;
}

void runec_ui_set_item_icon(RuneCUiState *ui, uint32_t icon_item_id, Texture2D texture) {
    if (!ui || icon_item_id == 0 || texture.id == 0)
        return;
    for (int i = 0; i < ui->item_icon_count; i++) {
        if (ui->item_icons[i].item_id == icon_item_id) {
            if (ui->item_icons[i].ready && ui->item_icons[i].texture.id != 0)
                UnloadTexture(ui->item_icons[i].texture);
            ui->item_icons[i].texture = texture;
            ui->item_icons[i].ready = 1;
            return;
        }
    }
    if (ui->item_icon_count >= RUNEC_UI_ITEM_ICON_CACHE) {
        UnloadTexture(texture);
        return;
    }
    ui->item_icons[ui->item_icon_count++] =
        (RuneCUiItemIcon){icon_item_id, texture, 1};
}

static int handle_context_click(RuneCUiState *ui, Vector2 mouse) {
    if (!ui->context_open)
        return 0;

    Rectangle box = {ui->context_pos.x, ui->context_pos.y,
                     158.0f, 24.0f + ui->context_action_count * 20.0f};
    if (!CheckCollisionPointRec(mouse, box)) {
        ui->context_open = 0;
        ui->context_source_kind = RUNEC_UI_CONTEXT_NONE;
        ui->context_source_slot = -1;
        ui->context_source_item_id = 0;
        return 0;
    }

    for (int i = 0; i < ui->context_action_count; i++) {
        Rectangle item = {box.x + 4, box.y + 22 + i * 20.0f, box.width - 8, 18};
        if (CheckCollisionPointRec(mouse, item)) {
            const char *action = ui->context_actions[i];
            if (ui->context_source_kind == RUNEC_UI_CONTEXT_INVENTORY) {
                if (strcmp(action, "Use") == 0) {
                    set_selected_item_target(ui, ui->context_source_slot);
                    ui->last_intent.kind = RUNEC_UI_INTENT_SELECTED_ITEM;
                    ui->last_intent.primary = ui->context_source_slot;
                    ui->last_intent.secondary = (int)ui->context_source_item_id;
                } else {
                    ui->last_intent.kind = RUNEC_UI_INTENT_INVENTORY_ACTION;
                    ui->last_intent.primary = ui->context_source_slot;
                    ui->last_intent.secondary = i;
                }
            } else if (ui->context_source_kind == RUNEC_UI_CONTEXT_EQUIPMENT) {
                ui->last_intent.kind = RUNEC_UI_INTENT_EQUIPMENT_ACTION;
                ui->last_intent.primary = ui->context_source_slot;
                ui->last_intent.secondary = i;
            } else if (ui->context_source_kind == RUNEC_UI_CONTEXT_PRAYER) {
                if (strcmp(action, "Activate") == 0) {
                    ui->last_intent.kind = RUNEC_UI_INTENT_PRAYER_SLOT;
                    ui->last_intent.primary = ui->context_source_slot;
                    copy_text(ui->last_intent.text,
                              sizeof(ui->last_intent.text),
                              ui->context_title);
                } else if (strcmp(action, "Quick-prayer") == 0) {
                    ui->last_intent.kind = RUNEC_UI_INTENT_QUICK_PRAYER_SLOT;
                    ui->last_intent.primary = ui->context_source_slot;
                    copy_text(ui->last_intent.text,
                              sizeof(ui->last_intent.text),
                              ui->context_title);
                } else {
                    ui->last_intent.kind = RUNEC_UI_INTENT_CONTEXT_ACTION;
                    ui->last_intent.primary = i;
                    ui->last_intent.secondary = ui->context_source_slot;
                }
            } else if (ui->context_source_kind == RUNEC_UI_CONTEXT_SPELL
                    && strcmp(action, "Cast") == 0) {
                set_selected_spell_target(ui, ui->context_source_slot,
                                          ui->context_title);
                ui->last_intent.kind = RUNEC_UI_INTENT_SELECTED_SPELL;
                ui->last_intent.primary = ui->context_source_slot;
                ui->last_intent.secondary = 0;
            } else if (ui->context_source_kind == RUNEC_UI_CONTEXT_SPELL
                    && strcmp(action, "Autocast") == 0) {
                ui->last_intent.kind = RUNEC_UI_INTENT_AUTOCAST_SPELL;
                ui->last_intent.primary = ui->context_source_slot;
                ui->last_intent.secondary = 0;
                copy_text(ui->last_intent.text,
                          sizeof(ui->last_intent.text),
                          ui->context_title);
            } else {
                ui->last_intent.kind = RUNEC_UI_INTENT_CONTEXT_ACTION;
                ui->last_intent.primary = i;
                ui->last_intent.secondary = ui->context_source_slot;
            }
            ui->last_intent.position = mouse;
            if (!ui->last_intent.text[0])
                copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
                          action);
            ui->context_open = 0;
            ui->context_source_kind = RUNEC_UI_CONTEXT_NONE;
            ui->context_source_slot = -1;
            ui->context_source_item_id = 0;
            return 1;
        }
    }

    return 1;
}

static Rectangle inv_slot_rect(const RuneCUiLayout *layout, int slot) {
    int col = slot % 4;
    int row = slot / 4;
    return (Rectangle){
        layout->side_content.x + RUNEC_OSRS_INVENTORY_SLOT_X + col * RUNEC_OSRS_INVENTORY_SLOT_STEP_X,
        layout->side_content.y + RUNEC_OSRS_INVENTORY_SLOT_Y + row * RUNEC_OSRS_INVENTORY_SLOT_STEP_Y,
        RUNEC_OSRS_INVENTORY_SLOT_W,
        RUNEC_OSRS_INVENTORY_SLOT_H,
    };
}

static int inv_slot_at(const RuneCUiLayout *layout, Vector2 mouse) {
    for (int i = 0; i < RUNEC_UI_INV_SLOT_COUNT; i++) {
        if (CheckCollisionPointRec(mouse, inv_slot_rect(layout, i)))
            return i;
    }
    return -1;
}

static Rectangle equip_slot_rect(const RuneCUiLayout *layout, int slot) {
    if (slot < 0 || slot >= RUNEC_UI_EQUIP_SLOT_COUNT)
        return (Rectangle){0, 0, 0, 0};
    Rectangle off = g_equipment_offsets[slot];
    if (off.width <= 0 || off.height <= 0)
        return off;
    return (Rectangle){
        layout->side_content.x + off.x,
        layout->side_content.y + off.y,
        off.width,
        off.height,
    };
}

static int equipment_slot_at(const RuneCUiLayout *layout, Vector2 mouse) {
    for (int i = 0; i < RUNEC_UI_EQUIP_SLOT_COUNT; i++) {
        Rectangle r = equip_slot_rect(layout, i);
        if (r.width > 0 && CheckCollisionPointRec(mouse, r))
            return i;
    }
    return -1;
}

static int ui_inventory_slot_at(const RuneCUiLayout *layout,
                                Vector2 mouse) {
    return inv_slot_at(layout, mouse);
}

static int ui_equipment_slot_at(const RuneCUiLayout *layout,
                                Vector2 mouse) {
    return equipment_slot_at(layout, mouse);
}

static Rectangle skill_slot_rect(const RuneCUiLayout *layout, int slot) {
    if (slot < 0 || slot >= (int)(sizeof(RUNEC_OSRS_SKILLS) / sizeof(RUNEC_OSRS_SKILLS[0])))
        return (Rectangle){0, 0, 0, 0};
    Rectangle r = RUNEC_OSRS_SKILLS[slot].rect;
    return (Rectangle){layout->side_content.x + r.x, layout->side_content.y + r.y,
                       r.width, r.height};
}

static int skill_slot_at(const RuneCUiLayout *layout, Vector2 mouse) {
    int count = (int)(sizeof(RUNEC_OSRS_SKILLS) / sizeof(RUNEC_OSRS_SKILLS[0]));
    for (int i = 0; i < count; i++) {
        if (CheckCollisionPointRec(mouse, skill_slot_rect(layout, i)))
            return i;
    }
    Rectangle total = {layout->side_content.x + RUNEC_OSRS_STATS_TOTAL.x,
                       layout->side_content.y + RUNEC_OSRS_STATS_TOTAL.y,
                       RUNEC_OSRS_STATS_TOTAL.width, RUNEC_OSRS_STATS_TOTAL.height};
    if (CheckCollisionPointRec(mouse, total))
        return count;
    return -1;
}

static Rectangle side_ref_rect(const RuneCUiLayout *layout, Rectangle ref) {
    return (Rectangle){layout->side_content.x + ref.x, layout->side_content.y + ref.y,
                       ref.width, ref.height};
}

static const RuneCUiCombatStyleOption *combat_style_at(
    const RuneCUiState *ui,
    const RuneCUiLayout *layout,
    Vector2 mouse) {
    if (!ui)
        return NULL;
    int visible_slot = 0;
    int layout_count = (int)(sizeof(RUNEC_OSRS_COMBAT_STYLES)
        / sizeof(RUNEC_OSRS_COMBAT_STYLES[0]));
    for (int i = 0; i < RUNEC_UI_COMBAT_STYLE_COUNT && visible_slot < layout_count; i++) {
        const RuneCUiCombatStyleOption *option = &ui->combat_styles[i];
        if (!option->visible)
            continue;
        const RuneCUiCombatStyleRef *slot =
            &RUNEC_OSRS_COMBAT_STYLES[visible_slot++];
        if (CheckCollisionPointRec(mouse, side_ref_rect(layout, slot->rect)))
            return option;
    }
    return NULL;
}

static Rectangle grid_cell_rect(const RuneCUiLayout *layout, int index,
                                int cols, float x0, float y0,
                                float step_x, float step_y,
                                float w, float h) {
    int col = index % cols;
    int row = index / cols;
    return (Rectangle){
        layout->side_content.x + x0 + col * step_x,
        layout->side_content.y + y0 + row * step_y,
        w,
        h,
    };
}

static int grid_index_at(const RuneCUiLayout *layout, Vector2 mouse,
                         int count, int cols, float x0, float y0,
                         float step_x, float step_y, float w, float h) {
    for (int i = 0; i < count; i++) {
        if (CheckCollisionPointRec(mouse,
                grid_cell_rect(layout, i, cols, x0, y0, step_x, step_y, w, h)))
            return i;
    }
    return -1;
}

static void update_tab_press_timers(RuneCUiState *ui, float dt) {
    for (int i = 0; i < RUNEC_UI_TAB_COUNT; i++) {
        if (ui->tab_press_timer[i] <= 0.0f)
            continue;
        ui->tab_press_timer[i] -= dt;
        if (ui->tab_press_timer[i] < 0.0f)
            ui->tab_press_timer[i] = 0.0f;
    }
}

static int handle_selected_target_cancel(RuneCUiState *ui) {
    if (ui->selected_target.kind == RUNEC_UI_SELECTED_NONE ||
        !IsKeyPressed(KEY_ESCAPE))
        return 0;
    runec_ui_clear_selected_target(ui);
    ui->last_intent.kind = RUNEC_UI_INTENT_SELECTED_TARGET_CANCEL;
    return 1;
}

static int handle_drag_release(RuneCUiState *ui,
                               const RuneCUiLayout *layout,
                               Vector2 mouse) {
    if (!ui->drag.active || !IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
        return 0;

    RuneCUiDragState drag = ui->drag;
    ui->drag.active = 0;
    ui->drag.source_kind = RUNEC_UI_CONTEXT_NONE;
    ui->drag.source_slot = -1;

    if (drag.source_kind == RUNEC_UI_CONTEXT_INVENTORY) {
        int target = ui_inventory_slot_at(layout, mouse);
        float dx = mouse.x - drag.start.x;
        float dy = mouse.y - drag.start.y;
        int moved = fabsf(dx) > 3.0f || fabsf(dy) > 3.0f;
        if (target >= 0 && moved) {
            ui->last_intent.kind = RUNEC_UI_INTENT_INVENTORY_DRAG;
            ui->last_intent.primary = drag.source_slot;
            ui->last_intent.secondary = target;
            ui->last_intent.position = mouse;
            return 1;
        }
        int previous = ui->selected_inventory_slot;
        ui->selected_inventory_slot = drag.source_slot;
        ui->last_intent.kind = RUNEC_UI_INTENT_INVENTORY_SLOT;
        ui->last_intent.primary = drag.source_slot;
        ui->last_intent.secondary = previous;
        ui->last_intent.position = mouse;
        return 1;
    }

    if (drag.source_kind == RUNEC_UI_CONTEXT_EQUIPMENT) {
        ui->selected_equipment_slot = drag.source_slot;
        ui->last_intent.kind = RUNEC_UI_INTENT_EQUIPMENT_SLOT;
        ui->last_intent.primary = drag.source_slot;
        ui->last_intent.position = mouse;
        return 1;
    }
    return 0;
}

static int handle_tab_click(RuneCUiState *ui,
                            const RuneCUiLayout *layout,
                            Vector2 mouse) {
    for (int i = 0; i < RUNEC_UI_TAB_COUNT; i++) {
        if (!CheckCollisionPointRec(mouse, layout->tab[i]))
            continue;
        ui->active_tab = (RuneCUiTab)i;
        ui->tab_press_timer[i] = OSRS_TAB_PRESS_SECONDS;
        ui->last_intent.kind = RUNEC_UI_INTENT_TAB;
        ui->last_intent.primary = i;
        ui->last_intent.position = mouse;
        return 1;
    }
    return 0;
}

static int handle_orb_or_minimap_click(RuneCUiState *ui,
                                       const RuneCUiLayout *layout,
                                       Vector2 mouse) {
    Rectangle run_button = {
        layout->run_orb.x + 3,
        layout->run_orb.y + 5,
        50,
        26,
    };
    if (CheckCollisionPointRec(mouse, run_button)) {
        ui->last_intent.kind = RUNEC_UI_INTENT_RUN_TOGGLE;
        ui->last_intent.position = mouse;
        return 1;
    }
    if (CheckCollisionPointRec(mouse, layout->prayer_orb)) {
        ui->last_intent.kind = RUNEC_UI_INTENT_QUICK_PRAYER_TOGGLE;
        ui->last_intent.primary = 1;
        ui->last_intent.position = mouse;
        copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
                  "Quick-prayer");
        return 1;
    }
    if (CheckCollisionPointRec(mouse, layout->spec_orb)) {
        ui->special_attack_enabled = !ui->special_attack_enabled;
        ui->last_intent.kind = RUNEC_UI_INTENT_SPECIAL_ATTACK;
        ui->last_intent.primary = ui->special_attack_enabled;
        ui->last_intent.secondary = ui->special_attack_energy;
        ui->last_intent.position = mouse;
        return 1;
    }

    Vector2 center = {
        layout->minimap.x + layout->minimap.width * 0.5f,
        layout->minimap.y + layout->minimap.height * 0.5f,
    };
    float dx = mouse.x - center.x;
    float dy = mouse.y - center.y;
    if (!CheckCollisionPointRec(mouse, layout->minimap) ||
        dx * dx + dy * dy > 75.0f * 75.0f)
        return 0;
    ui->last_intent.kind = RUNEC_UI_INTENT_MINIMAP_CLICK;
    ui->last_intent.primary = (int)(mouse.x - layout->minimap.x);
    ui->last_intent.secondary = (int)(mouse.y - layout->minimap.y);
    ui->last_intent.position = mouse;
    return 1;
}

static int handle_combat_click(RuneCUiState *ui,
                               const RuneCUiLayout *layout,
                               Vector2 mouse) {
    const RuneCUiCombatStyleOption *style =
        combat_style_at(ui, layout, mouse);
    if (style) {
        ui->selected_combat_style = style->style_index;
        ui->last_intent.kind = RUNEC_UI_INTENT_COMBAT_STYLE;
        ui->last_intent.primary = style->style_index;
        ui->last_intent.position = mouse;
        copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
                  style->label);
        return 1;
    }
    if (CheckCollisionPointRec(mouse,
            side_ref_rect(layout, RUNEC_OSRS_COMBAT_RETALIATE))) {
        ui->auto_retaliate = !ui->auto_retaliate;
        ui->last_intent.kind = RUNEC_UI_INTENT_AUTO_RETALIATE;
        ui->last_intent.primary = ui->auto_retaliate;
        ui->last_intent.position = mouse;
        return 1;
    }
    if (CheckCollisionPointRec(mouse,
            side_ref_rect(layout, RUNEC_OSRS_COMBAT_SPECIAL_BAR))) {
        ui->special_attack_enabled = !ui->special_attack_enabled;
        ui->last_intent.kind = RUNEC_UI_INTENT_SPECIAL_ATTACK;
        ui->last_intent.primary = ui->special_attack_enabled;
        ui->last_intent.secondary = ui->special_attack_energy;
        ui->last_intent.position = mouse;
        return 1;
    }
    return 0;
}

static int handle_inventory_click(RuneCUiState *ui,
                                  const RuneCUiLayout *layout,
                                  Vector2 mouse) {
    int slot = ui_inventory_slot_at(layout, mouse);
    if (slot < 0)
        return 0;
    if (ui->selected_target.kind == RUNEC_UI_SELECTED_ITEM ||
        ui->selected_target.kind == RUNEC_UI_SELECTED_SPELL) {
        ui->last_intent.kind = ui->selected_target.kind == RUNEC_UI_SELECTED_ITEM
            ? RUNEC_UI_INTENT_SELECTED_ITEM_ON_ITEM
            : RUNEC_UI_INTENT_SELECTED_SPELL_ON_ITEM;
        ui->last_intent.primary = ui->selected_target.source_slot;
        ui->last_intent.secondary = slot;
        ui->last_intent.position = mouse;
        snprintf(ui->last_intent.text, sizeof(ui->last_intent.text),
                 "%s -> %s", ui->selected_target.label,
                 ui->inventory[slot].enabled
                    ? ui->inventory[slot].label : "slot");
        runec_ui_clear_selected_target(ui);
        return 1;
    }
    ui->drag.active = 1;
    ui->drag.source_kind = RUNEC_UI_CONTEXT_INVENTORY;
    ui->drag.source_slot = slot;
    ui->drag.start = mouse;
    return 1;
}

static int handle_equipment_click(RuneCUiState *ui,
                                  const RuneCUiLayout *layout,
                                  Vector2 mouse) {
    int slot = ui_equipment_slot_at(layout, mouse);
    if (slot < 0)
        return 0;
    ui->drag.active = 1;
    ui->drag.source_kind = RUNEC_UI_CONTEXT_EQUIPMENT;
    ui->drag.source_slot = slot;
    ui->drag.start = mouse;
    return 1;
}

static int handle_prayer_click(RuneCUiState *ui,
                               const RuneCUiLayout *layout,
                               Vector2 mouse) {
    int slot = grid_index_at(layout, mouse, 25, 5, 8, 8, 36, 36, 34, 34);
    if (slot < 0)
        return 0;
    ui->last_intent.kind = RUNEC_UI_INTENT_PRAYER_SLOT;
    ui->last_intent.primary = slot;
    ui->last_intent.position = mouse;
    copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
              g_prayer_names[slot]);
    return 1;
}

static int handle_spellbook_click(RuneCUiState *ui,
                                  const RuneCUiLayout *layout,
                                  Vector2 mouse) {
    int slot = grid_index_at(layout, mouse, RUNEC_UI_SPELL_COUNT,
        RUNEC_UI_SPELL_COLS, RUNEC_UI_SPELL_X0, RUNEC_UI_SPELL_Y0,
        RUNEC_UI_SPELL_STEP_X, RUNEC_UI_SPELL_STEP_Y,
        RUNEC_UI_SPELL_ICON_SIZE, RUNEC_UI_SPELL_ICON_SIZE);
    if (slot < 0)
        return 0;
    set_selected_spell_target(ui, slot, spell_name(slot));
    ui->last_intent.kind = RUNEC_UI_INTENT_SELECTED_SPELL;
    ui->last_intent.primary = slot;
    ui->last_intent.position = mouse;
    copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
              spell_name(slot));
    return 1;
}

static int handle_skills_click(RuneCUiState *ui,
                               const RuneCUiLayout *layout,
                               Vector2 mouse) {
    int slot = skill_slot_at(layout, mouse);
    if (slot < 0)
        return 0;
    ui->last_intent.kind = RUNEC_UI_INTENT_SKILL_SLOT;
    ui->last_intent.primary = slot;
    ui->last_intent.position = mouse;
    copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
              slot < (int)(sizeof(RUNEC_OSRS_SKILLS) /
                           sizeof(RUNEC_OSRS_SKILLS[0]))
                ? RUNEC_OSRS_SKILLS[slot].name : "Total level");
    return 1;
}

static int handle_active_tab_click(RuneCUiState *ui,
                                   const RuneCUiLayout *layout,
                                   Vector2 mouse) {
    switch (ui->active_tab) {
    case RUNEC_UI_TAB_COMBAT:
        return handle_combat_click(ui, layout, mouse);
    case RUNEC_UI_TAB_INVENTORY:
        return handle_inventory_click(ui, layout, mouse);
    case RUNEC_UI_TAB_EQUIPMENT:
        return handle_equipment_click(ui, layout, mouse);
    case RUNEC_UI_TAB_PRAYER:
        return handle_prayer_click(ui, layout, mouse);
    case RUNEC_UI_TAB_SPELLBOOK:
        return handle_spellbook_click(ui, layout, mouse);
    case RUNEC_UI_TAB_SKILLS:
        return handle_skills_click(ui, layout, mouse);
    default:
        return 0;
    }
}

static int handle_primary_click(RuneCUiState *ui,
                                const RuneCUiLayout *layout,
                                Vector2 mouse) {
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        return 0;
    if (handle_context_click(ui, mouse) || handle_tab_click(ui, layout, mouse) ||
        handle_orb_or_minimap_click(ui, layout, mouse) ||
        handle_active_tab_click(ui, layout, mouse))
        return 1;
    return 0;
}

static int handle_context_menu_open(RuneCUiState *ui,
                                    const RuneCUiLayout *layout,
                                    Vector2 mouse) {
    if (!IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE) ||
        !mouse_over_ui(layout, mouse))
        return 0;
    runec_ui_clear_selected_target(ui);

    if (ui->active_tab == RUNEC_UI_TAB_COMBAT) {
        const RuneCUiCombatStyleOption *style =
            combat_style_at(ui, layout, mouse);
        if (style) {
            static const char *actions[] = {"Select", "Examine"};
            set_context(ui, mouse, style->label, actions, 2);
            return 1;
        }
        if (CheckCollisionPointRec(mouse,
                side_ref_rect(layout, RUNEC_OSRS_COMBAT_RETALIATE))) {
            static const char *actions[] = {"Toggle", "Examine"};
            set_context(ui, mouse, "Auto Retaliate", actions, 2);
            return 1;
        }
        if (CheckCollisionPointRec(mouse,
                side_ref_rect(layout, RUNEC_OSRS_COMBAT_SPECIAL_BAR))) {
            static const char *actions[] = {"Use", "Examine"};
            set_context(ui, mouse, "Special Attack", actions, 2);
            return 1;
        }
    }
    if (ui->active_tab == RUNEC_UI_TAB_INVENTORY) {
        int slot = ui_inventory_slot_at(layout, mouse);
        if (slot >= 0) {
            static const char *actions[] = {"Use", "Examine", "Drop"};
            static const char *empty_actions[] = {"Cancel"};
            const char *title = ui->inventory[slot].enabled
                ? ui->inventory[slot].label : "Empty inventory slot";
            if (ui->inventory[slot].enabled) {
                set_context(ui, mouse, title, actions, 3);
                set_context_source(ui, RUNEC_UI_CONTEXT_INVENTORY, slot,
                                   ui->inventory[slot].item_id);
            } else {
                set_context(ui, mouse, title, empty_actions, 1);
            }
            return 1;
        }
    }
    if (ui->active_tab == RUNEC_UI_TAB_EQUIPMENT) {
        int slot = ui_equipment_slot_at(layout, mouse);
        if (slot >= 0) {
            static const char *actions[] = {"Remove", "Examine"};
            set_context(ui, mouse, g_equipment_names[slot], actions, 2);
            set_context_source(ui, RUNEC_UI_CONTEXT_EQUIPMENT, slot,
                               ui->equipment[slot].item_id);
            return 1;
        }
    }
    if (ui->active_tab == RUNEC_UI_TAB_PRAYER) {
        int slot = grid_index_at(layout, mouse, 25, 5,
                                 8, 8, 36, 36, 34, 34);
        if (slot >= 0) {
            static const char *actions[] = {
                "Activate", "Quick-prayer", "Examine"
            };
            set_context(ui, mouse, g_prayer_names[slot], actions, 3);
            set_context_source(ui, RUNEC_UI_CONTEXT_PRAYER, slot, 0);
            return 1;
        }
    }
    if (ui->active_tab == RUNEC_UI_TAB_SPELLBOOK) {
        int slot = grid_index_at(layout, mouse, RUNEC_UI_SPELL_COUNT,
            RUNEC_UI_SPELL_COLS, RUNEC_UI_SPELL_X0, RUNEC_UI_SPELL_Y0,
            RUNEC_UI_SPELL_STEP_X, RUNEC_UI_SPELL_STEP_Y,
            RUNEC_UI_SPELL_ICON_SIZE, RUNEC_UI_SPELL_ICON_SIZE);
        if (slot >= 0) {
            static const char *actions[] = {"Cast", "Autocast", "Examine"};
            set_context(ui, mouse, spell_name(slot), actions, 3);
            set_context_source(ui, RUNEC_UI_CONTEXT_SPELL, slot, 0);
            return 1;
        }
    }

    static const char *actions[] = {"Cancel"};
    set_context(ui, mouse, "RuneC", actions, 1);
    return 1;
}

int runec_ui_handle_input(RuneCUiState *ui, int screen_w, int screen_h) {
    RuneCUiLayout layout;
    ui_layout(screen_w, screen_h, &layout);
    Vector2 mouse = GetMousePosition();
    clear_intent(ui);
    update_tab_press_timers(ui, GetFrameTime());

    if (handle_selected_target_cancel(ui) ||
        handle_drag_release(ui, &layout, mouse) ||
        handle_primary_click(ui, &layout, mouse) ||
        handle_context_menu_open(ui, &layout, mouse))
        return 1;

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && !ui->context_open)
        return 0;
    return mouse_over_ui(&layout, mouse);
}

static void draw_text_shadow(const RuneCUiState *ui, const char *text,
                             float x, float y, float size, Color color) {
    runec_ui_draw_text_shadow(&ui->assets, text, x, y, size, color);
}

static void draw_centered_text(const RuneCUiState *ui, const char *text,
                               Rectangle rect, float size, Color color) {
    if (size < 12.0f)
        size = 12.0f;
    Font font = runec_ui_font_for_size(&ui->assets, size);
    Vector2 m = MeasureTextEx(font, text, size, 0);
    draw_text_shadow(ui, text, rect.x + (rect.width - m.x) * 0.5f,
                     rect.y + (rect.height - m.y) * 0.5f, size, color);
}

static int draw_asset_centered(const RuneCUiState *ui, const char *name,
                               Rectangle rect, float max_w, float max_h, Color tint) {
    const Texture2D *tex = runec_ui_asset(&ui->assets, name);
    if (!tex)
        return 0;
    float scale = fmin2(max_w / (float)tex->width, max_h / (float)tex->height);
    if (scale > 1.0f)
        scale = 1.0f;
    Rectangle dst = {
        rect.x + (rect.width - tex->width * scale) * 0.5f,
        rect.y + (rect.height - tex->height * scale) * 0.5f,
        tex->width * scale,
        tex->height * scale,
    };
    runec_ui_draw_asset(&ui->assets, name, dst, tint);
    return 1;
}

static void draw_asset_tiled(const RuneCUiState *ui, const char *name,
                             Rectangle dst, Color tint) {
    const Texture2D *tex = runec_ui_asset(&ui->assets, name);
    if (!tex) {
        DrawRectangleRec(dst, OSRS_PANEL);
        return;
    }
    for (float y = dst.y; y < dst.y + dst.height; y += (float)tex->height) {
        for (float x = dst.x; x < dst.x + dst.width; x += (float)tex->width) {
            float w = fmin2((float)tex->width, dst.x + dst.width - x);
            float h = fmin2((float)tex->height, dst.y + dst.height - y);
            DrawTexturePro(*tex, (Rectangle){0, 0, w, h},
                           (Rectangle){x, y, w, h}, (Vector2){0, 0}, 0, tint);
        }
    }
}

static void draw_side_chrome(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    Rectangle backing = {layout->side.x + 20, layout->side.y + 27, 200, 281};
    draw_asset_tiled(ui, "tradebacking_dark", backing, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, "tradebacking_dark"))
        DrawRectangleRec(backing, OSRS_PANEL);

    runec_ui_draw_asset(&ui->assets, "osrs_stretch_side_topbottom_0",
                        (Rectangle){layout->side.x, layout->side.y + RUNEC_OSRS_SIDE_TOP_Y,
                                    241, 37}, WHITE);
    runec_ui_draw_asset(&ui->assets, "osrs_stretch_side_topbottom_1",
                        (Rectangle){layout->side.x, layout->side.y + RUNEC_OSRS_SIDE_BOTTOM_Y,
                                    241, 37}, WHITE);
    runec_ui_draw_asset(&ui->assets, "osrs_stretch_side_columns_0",
                        (Rectangle){layout->side.x + 2, layout->side.y + 37, 26, 261}, WHITE);
    runec_ui_draw_asset(&ui->assets, "osrs_stretch_side_columns_1",
                        (Rectangle){layout->side.x + 212, layout->side.y + 37, 26, 261}, WHITE);

    for (int i = 0; i < (int)(sizeof(RUNEC_OSRS_SIDE_STONES) / sizeof(RUNEC_OSRS_SIDE_STONES[0])); i++) {
        const RuneCUiStoneRef *ref = &RUNEC_OSRS_SIDE_STONES[i];
        if (ref->logical_tab != ui->active_tab)
            continue;
        float row_y = layout->side.y + (i < 7 ? RUNEC_OSRS_SIDE_TOP_Y : RUNEC_OSRS_SIDE_BOTTOM_Y);
        float pressed = ui->tab_press_timer[ui->active_tab] > 0.0f ? 1.0f : 0.0f;
        Rectangle stone = {layout->side.x + ref->rect.x,
                           row_y + ref->rect.y + pressed,
                           ref->rect.width, ref->rect.height};
        draw_asset_tiled(ui, ref->stone_asset, stone, WHITE);
        DrawRectangleRec(stone, (Color){145, 22, 18, pressed > 0.0f ? 72 : 44});
        break;
    }

    for (int i = 0; i < (int)(sizeof(RUNEC_OSRS_SIDE_STONES) / sizeof(RUNEC_OSRS_SIDE_STONES[0])); i++) {
        const RuneCUiStoneRef *ref = &RUNEC_OSRS_SIDE_STONES[i];
        float row_y = layout->side.y + (i < 7 ? RUNEC_OSRS_SIDE_TOP_Y : RUNEC_OSRS_SIDE_BOTTOM_Y);
        float pressed = ref->logical_tab == ui->active_tab
            && ui->tab_press_timer[ui->active_tab] > 0.0f ? 1.0f : 0.0f;
        Rectangle icon = {layout->side.x + ref->icon_rect.x, row_y + ref->icon_rect.y,
                          ref->icon_rect.width, ref->icon_rect.height};
        icon.y += pressed;
        runec_ui_draw_asset(&ui->assets, ref->icon_asset, icon, WHITE);
    }
}

static void draw_orb(const RuneCUiState *ui, Rectangle rect, const char *filler,
                     const char *icon, int value, int max_value, Color color) {
    runec_ui_draw_asset(&ui->assets, "orb_frame_0", rect, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, "orb_frame_0"))
        DrawRectangleRounded((Rectangle){rect.x + 0, rect.y + 7, 34, 20}, 0.22f, 5,
                             (Color){50, 46, 37, 235});
    Rectangle fill = {rect.x + 27, rect.y + 4, 26, 26};
    runec_ui_draw_asset(&ui->assets, "orb_filler_0", fill, WHITE);
    runec_ui_draw_asset(&ui->assets, filler, fill, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, filler))
        DrawCircle((int)(fill.x + 14), (int)(fill.y + 14), 12, color);
    draw_asset_centered(ui, icon, fill, 22, 22, WHITE);

    char text[16];
    snprintf(text, sizeof(text), "%d", value);
    Color text_color = max_value > 0 && value < max_value / 3 ? OSRS_RED : OSRS_GREEN;
    draw_centered_text(ui, text, (Rectangle){rect.x + 3, rect.y + 14, 24, 13}, 12, text_color);
}

static Color orb_value_color(int value, int max_value) {
    if (max_value <= 0)
        max_value = 1;
    if (value < 0)
        value = 0;
    if (value > max_value)
        value = max_value;

    int half = max_value / 2;
    if (half <= 0)
        return value >= max_value ? (Color){0, 255, 0, 255}
                                  : (Color){255, 0, 0, 255};
    if (value > half) {
        int red = 255 - 255 * (value - half) / half;
        return (Color){(unsigned char)red, 255, 0, 255};
    }
    int green = 255 * value / half;
    return (Color){255, (unsigned char)green, 0, 255};
}

static void draw_run_orb(const RuneCUiState *ui, Rectangle rect) {
    Rectangle button = {rect.x + 3, rect.y + 5, 50, 26};
    const char *frame = CheckCollisionPointRec(GetMousePosition(), button)
        ? "orb_frame_1" : "orb_frame_0";
    runec_ui_draw_asset(&ui->assets, frame, rect, WHITE);

    Rectangle fill = {rect.x + 27, rect.y + 4, 26, 26};
    const char *filler = ui->run_enabled ? "orb_filler_6" : "orb_filler_5";
    const char *icon = ui->run_enabled ? "orb_icon_3" : "orb_icon_2";
    runec_ui_draw_asset(&ui->assets, filler, fill,
                        (Color){255, 255, 255, 230});

    int energy = ui->run_energy;
    if (energy < 0) energy = 0;
    if (energy > 100) energy = 100;
    int empty_height = 26 * (100 - energy) / 100;
    const Texture2D *empty = runec_ui_asset(&ui->assets, "orb_filler_0");
    if (empty && empty_height > 0) {
        float source_height = (float)empty->height * (float)empty_height /
                              fill.height;
        DrawTexturePro(*empty,
                       (Rectangle){0, 0, (float)empty->width, source_height},
                       (Rectangle){fill.x, fill.y, fill.width,
                                   (float)empty_height},
                       (Vector2){0, 0}, 0.0f, WHITE);
    }
    draw_asset_centered(ui, icon, fill, 26, 26, WHITE);

    char text[16];
    snprintf(text, sizeof(text), "%d", energy);
    draw_centered_text(ui, text,
                       (Rectangle){rect.x + 3, rect.y + 14, 24, 13},
                       12, orb_value_color(energy, 100));
}

static void draw_minimap(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    Vector2 center = {layout->minimap.x + layout->minimap.width * 0.5f,
                      layout->minimap.y + layout->minimap.height * 0.5f};
    if (ui->minimap_texture_ready) {
        DrawTexturePro(ui->minimap_texture,
                       (Rectangle){0, 0, 152, 152},
                       layout->minimap, (Vector2){0, 0}, 0, WHITE);
    } else {
        DrawCircle((int)center.x, (int)center.y, 72.0f, (Color){85, 124, 52, 255});
    }

    for (int i = 0; i < ui->minimap_dot_count; i++) {
        const RuneCUiMinimapDot *dot = &ui->minimap_dots[i];
        float px = center.x +
            dot->dx * FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
        float py = center.y -
            dot->dy * FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
        float dx = px - center.x;
        float dy = py - center.y;
        if (dx * dx + dy * dy > 68.0f * 68.0f)
            continue;
        Color c = OSRS_YELLOW;
        float radius = 2.0f;
        if (dot->kind == RUNEC_UI_MINIMAP_DOT_PLAYER) {
            c = WHITE;
            radius = 3.0f;
        } else if (dot->kind == RUNEC_UI_MINIMAP_DOT_DESTINATION) {
            c = OSRS_RED;
            radius = 3.0f;
        }
        DrawCircle((int)px, (int)py, radius, c);
    }

    const Texture2D *compass = runec_ui_asset(&ui->assets, "compass");
    if (compass) {
        Rectangle source = {0, 0, (float)compass->width,
                            (float)compass->height};
        Rectangle destination = {
            layout->compass.x + layout->compass.width * 0.5f,
            layout->compass.y + layout->compass.height * 0.5f,
            layout->compass.width,
            layout->compass.height,
        };
        Vector2 origin = {layout->compass.width * 0.5f,
                          layout->compass.height * 0.5f};
        DrawTexturePro(*compass, source, destination, origin,
                       ui->minimap_rotation *
                           (180.0f / 3.14159265358979323846f),
                       WHITE);
    } else {
        runec_ui_draw_asset(&ui->assets, "resize_compass_mask", layout->compass, WHITE);
        draw_text_shadow(ui, "N", layout->compass.x + 16, layout->compass.y + 12, 14, OSRS_ORANGE);
    }

    Rectangle cover = {layout->map.x + RUNEC_OSRS_MAP_SURROUND_X,
                       layout->map.y + RUNEC_OSRS_MAP_SURROUND_Y,
                       RUNEC_OSRS_MAP_SURROUND_W, RUNEC_OSRS_MAP_SURROUND_H};
    runec_ui_draw_asset(&ui->assets, "osrs_stretch_mapsurround", cover, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, "osrs_stretch_mapsurround")) {
        DrawCircleLines((int)center.x, (int)center.y, 73.0f, (Color){51, 48, 35, 255});
        DrawCircleLines((int)center.x, (int)center.y, 70.0f, (Color){180, 166, 104, 255});
    }

    draw_orb(ui, layout->hp_orb, "orb_filler_1", "orb_icon_0",
             ui->hitpoints, ui->hitpoints_max, OSRS_RED);
    draw_orb(ui, layout->prayer_orb, "orb_filler_4", "orb_icon_1",
             ui->prayer_points, ui->prayer_points_max, OSRS_BLUE);
    draw_run_orb(ui, layout->run_orb);
    draw_orb(ui, layout->spec_orb, "orb_filler_9", "orb_icon_6",
             ui->special_attack_energy, 100, OSRS_YELLOW);

    runec_ui_draw_asset(&ui->assets, "ring_30", layout->worldmap_button, WHITE);
    draw_asset_centered(ui, "worldmap_icon_0", layout->worldmap_button, 22, 22, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, "worldmap_icon_0"))
        draw_centered_text(ui, "?", layout->worldmap_button, 14, OSRS_YELLOW);
}

static void draw_chat_panel_chrome(const RuneCUiState *ui,
                                   const RuneCUiLayout *layout) {
    runec_ui_draw_asset(&ui->assets, "chatbox_bg", layout->chat, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, "chatbox_bg"))
        DrawRectangleRec(layout->chat, (Color){22, 19, 15, 235});

    DrawRectangleRec(layout->chat_messages, (Color){0, 0, 0, 104});
    runec_ui_draw_asset(&ui->assets, "main_stones_bottom",
                        layout->chat_controls, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, "main_stones_bottom"))
        DrawRectangleRec(layout->chat_controls, (Color){74, 62, 48, 235});
}

static Color item_color(uint32_t item_id) {
    switch (item_id) {
    case 4151: return (Color){176, 178, 164, 255};
    case 385: return (Color){80, 154, 178, 255};
    case 2434: return (Color){110, 190, 70, 255};
    case 995: return (Color){228, 188, 58, 255};
    default: return (Color){128, 92, 52, 255};
    }
}

static int coin_stack_visual_quantity(int quantity) {
    if (quantity <= 1) return 1;
    if (quantity == 2) return 2;
    if (quantity == 3) return 3;
    if (quantity == 4) return 4;
    if (quantity < 25) return 5;
    if (quantity < 100) return 25;
    if (quantity < 250) return 100;
    if (quantity < 1000) return 250;
    if (quantity < 10000) return 1000;
    return 10000;
}

static void item_icon_asset_name(const RuneCUiSlot *slot, char *dst, size_t cap) {
    uint32_t icon_item_id = slot->icon_item_id ? slot->icon_item_id : slot->item_id;
    if (icon_item_id != slot->item_id) {
        snprintf(dst, cap, "item_%u", icon_item_id);
        return;
    }
    if (slot->item_id == 995) {
        snprintf(dst, cap, "item_995_%d", coin_stack_visual_quantity(slot->quantity));
        return;
    }
    snprintf(dst, cap, "item_%u", slot->item_id);
}

static int label_word_is_filler(const char *word, int len) {
    return (len == 2 && strncmp(word, "of", 2) == 0) ||
           (len == 3 && strncmp(word, "the", 3) == 0) ||
           (len == 3 && strncmp(word, "and", 3) == 0);
}

static void slot_label_abbrev(const RuneCUiSlot *slot, char *dst, size_t cap) {
    if (!dst || cap == 0)
        return;
    dst[0] = '\0';
    if (!slot || !slot->label[0])
        return;

    int out = 0;
    const char *p = slot->label;
    while (*p && out < (int)cap - 1) {
        while (*p && !isalnum((unsigned char)*p))
            p++;
        if (!*p)
            break;

        char word[16];
        int len = 0;
        unsigned char first = (unsigned char)*p;
        while (*p && isalnum((unsigned char)*p)) {
            if (len < (int)sizeof(word) - 1)
                word[len++] = (char)tolower((unsigned char)*p);
            p++;
        }
        word[len] = '\0';
        if (len > 0 && !label_word_is_filler(word, len))
            dst[out++] = (char)toupper(first);
    }
    dst[out] = '\0';
}

static void draw_slot_label_fallback(const RuneCUiState *ui, const RuneCUiSlot *slot,
                                     Rectangle r) {
    char abbr[4];
    slot_label_abbrev(slot, abbr, sizeof(abbr));
    if (!abbr[0])
        return;

    float size = 10.0f;
    Font font = runec_ui_font_for_size(&ui->assets, size);
    Vector2 m = MeasureTextEx(font, abbr, size, 0);
    if (m.x > r.width - 4.0f) {
        size = 8.0f;
        font = runec_ui_font_for_size(&ui->assets, size);
        m = MeasureTextEx(font, abbr, size, 0);
    }
    draw_text_shadow(ui, abbr, r.x + (r.width - m.x) * 0.5f,
                     r.y + (r.height - m.y) * 0.5f, size,
                     (Color){238, 218, 162, 245});
}

static Color stack_text_color(int quantity) {
    if (quantity >= 10000000)
        return OSRS_GREEN;
    if (quantity >= 100000)
        return WHITE;
    return OSRS_YELLOW;
}

static void format_stack_quantity(int quantity, char *dst, size_t cap) {
    if (quantity >= 10000000) {
        snprintf(dst, cap, "%dM", quantity / 1000000);
    } else if (quantity >= 100000) {
        snprintf(dst, cap, "%dK", quantity / 1000);
    } else {
        snprintf(dst, cap, "%d", quantity);
    }
}

static const Texture2D *ui_item_icon_texture(const RuneCUiState *ui,
                                             uint32_t icon_item_id) {
    for (int i = 0; i < ui->item_icon_count; i++) {
        if (ui->item_icons[i].ready && ui->item_icons[i].item_id == icon_item_id)
            return &ui->item_icons[i].texture;
    }
    return NULL;
}

static void draw_inventory_item(const RuneCUiState *ui, const RuneCUiSlot *slot, Rectangle r) {
    uint32_t icon_item_id = slot->icon_item_id ? slot->icon_item_id : slot->item_id;
    const Texture2D *runtime_icon = ui_item_icon_texture(ui, icon_item_id);
    if (runtime_icon && runtime_icon->id != 0) {
        float sx = r.width / (float)runtime_icon->width;
        float sy = r.height / (float)runtime_icon->height;
        float scale = sx < sy ? sx : sy;
        Rectangle dst = {
            r.x + (r.width - (float)runtime_icon->width * scale) * 0.5f,
            r.y + (r.height - (float)runtime_icon->height * scale) * 0.5f,
            (float)runtime_icon->width * scale,
            (float)runtime_icon->height * scale
        };
        DrawTexturePro(*runtime_icon,
                       (Rectangle){0, 0, (float)runtime_icon->width,
                                   (float)runtime_icon->height},
                       dst, (Vector2){0, 0}, 0.0f, WHITE);
        return;
    }

    char icon_name[32];
    item_icon_asset_name(slot, icon_name, sizeof(icon_name));
    if (draw_asset_centered(ui, icon_name, r, RUNEC_OSRS_INVENTORY_SLOT_W,
                            RUNEC_OSRS_INVENTORY_SLOT_H, WHITE))
        return;

    Color c = item_color(slot->item_id);
    if (slot->item_id == 4151) {
        DrawLineEx((Vector2){r.x + 9, r.y + 25}, (Vector2){r.x + 24, r.y + 6}, 4.0f,
                   (Color){42, 40, 38, 255});
        DrawLineEx((Vector2){r.x + 11, r.y + 23}, (Vector2){r.x + 25, r.y + 7}, 2.0f, c);
        DrawCircle((int)(r.x + 9), (int)(r.y + 25), 4.0f, (Color){83, 50, 38, 255});
    } else if (slot->item_id == 385) {
        DrawEllipse((int)(r.x + 16), (int)(r.y + 17), 12.0f, 7.0f, c);
        DrawTriangle((Vector2){r.x + 5, r.y + 17}, (Vector2){r.x + 1, r.y + 11},
                     (Vector2){r.x + 1, r.y + 23}, c);
        DrawCircle((int)(r.x + 23), (int)(r.y + 15), 1.5f, BLACK);
    } else if (slot->item_id == 2434) {
        DrawRectangleRounded((Rectangle){r.x + 11, r.y + 5, 10, 22}, 0.35f, 4,
                             (Color){52, 42, 34, 255});
        DrawRectangleRounded((Rectangle){r.x + 12, r.y + 10, 8, 15}, 0.35f, 4, c);
        DrawRectangleRec((Rectangle){r.x + 11, r.y + 4, 10, 4}, (Color){196, 196, 182, 255});
    } else if (slot->item_id == 995) {
        DrawCircle((int)(r.x + 14), (int)(r.y + 14), 7.0f, c);
        DrawCircle((int)(r.x + 19), (int)(r.y + 18), 7.0f, (Color){210, 156, 40, 255});
        DrawCircle((int)(r.x + 13), (int)(r.y + 21), 6.0f, (Color){238, 204, 72, 255});
    } else {
        DrawRectangleRounded((Rectangle){r.x + 6, r.y + 6, 20, 20}, 0.22f, 4, c);
        draw_slot_label_fallback(ui, slot, r);
    }
    (void)ui;
}

static void draw_inventory(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    for (int i = 0; i < RUNEC_UI_INV_SLOT_COUNT; i++) {
        Rectangle r = inv_slot_rect(layout, i);
        if (ui->selected_inventory_slot == i)
            DrawRectangleLinesEx((Rectangle){r.x - 1, r.y - 1, r.width + 2, r.height + 2},
                                 2.0f, OSRS_YELLOW);
        if (ui->inventory[i].enabled) {
            draw_inventory_item(ui, &ui->inventory[i], r);
            if (ui->inventory[i].quantity > 1) {
                char q[16];
                format_stack_quantity(ui->inventory[i].quantity, q, sizeof(q));
                draw_text_shadow(ui, q, r.x + 1, r.y - 1, 10,
                                 stack_text_color(ui->inventory[i].quantity));
            }
        }
    }
}

static void draw_equipment(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    draw_asset_tiled(ui, "miscgraphics_2",
                     side_ref_rect(layout, (Rectangle){77, 39, 36, 124}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_2",
                     side_ref_rect(layout, (Rectangle){21, 118, 36, 45}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_2",
                     side_ref_rect(layout, (Rectangle){133, 118, 36, 45}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_3",
                     side_ref_rect(layout, (Rectangle){56, 81, 78, 36}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_3",
                     side_ref_rect(layout, (Rectangle){71, 42, 48, 36}), WHITE);

    for (int i = 0; i < RUNEC_UI_EQUIP_SLOT_COUNT; i++) {
        Rectangle r = equip_slot_rect(layout, i);
        if (r.width <= 0)
            continue;
        if (ui->equipment[i].enabled) {
            draw_inventory_item(ui, &ui->equipment[i], r);
            if (ui->equipment[i].quantity > 1) {
                char q[16];
                format_stack_quantity(ui->equipment[i].quantity, q, sizeof(q));
                draw_text_shadow(ui, q, r.x + 1, r.y - 1, 10,
                                 stack_text_color(ui->equipment[i].quantity));
            }
        } else if (g_worn_icon_names[i]) {
            draw_asset_centered(ui, g_worn_icon_names[i], r, 28, 28, (Color){190, 178, 150, 175});
        }
        if (ui->selected_equipment_slot == i) {
            DrawRectangleLinesEx((Rectangle){r.x - 1, r.y - 1, r.width + 2, r.height + 2},
                                 2.0f, OSRS_YELLOW);
        }
    }

    for (int i = 0; i < (int)(sizeof(RUNEC_OSRS_WORN_BUTTONS) / sizeof(RUNEC_OSRS_WORN_BUTTONS[0])); i++) {
        const RuneCUiWornButtonRef *ref = &RUNEC_OSRS_WORN_BUTTONS[i];
        Rectangle b = {layout->side_content.x + ref->rect.x,
                       layout->side_content.y + ref->rect.y,
                       ref->rect.width, ref->rect.height};
        Rectangle icon = {layout->side_content.x + ref->icon_rect.x,
                          layout->side_content.y + ref->icon_rect.y,
                          ref->icon_rect.width, ref->icon_rect.height};
        runec_ui_draw_asset(&ui->assets, "combatboxes_0", b, WHITE);
        if (!runec_ui_asset_ready(&ui->assets, "combatboxes_0")) {
            DrawRectangleRounded(b, 0.18f, 5, (Color){54, 46, 35, 235});
            DrawRectangleLinesEx(b, 1, (Color){119, 99, 68, 255});
        }
        draw_asset_centered(ui, ref->asset, icon, icon.width, icon.height, WHITE);
    }
}

static void draw_prayer(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    for (int i = 0; i < 25; i++) {
        Rectangle r = grid_cell_rect(layout, i, 5, 8, 8, 36, 36, 34, 34);
        DrawRectangleRec(r, (Color){16, 13, 10, 95});
        char name[32];
        snprintf(name, sizeof(name), "prayer%s_%d",
                 (ui->active_prayers & (1u << i)) ? "on" : "off", i);
        if (!draw_asset_centered(ui, name, r, 30, 30, WHITE))
            draw_centered_text(ui, TextFormat("%d", i + 1), r, 10, OSRS_ORANGE);
    }
}

static void draw_spellbook(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    for (int i = 0; i < RUNEC_UI_SPELL_COUNT; i++) {
        Rectangle r = grid_cell_rect(layout, i, RUNEC_UI_SPELL_COLS,
                                     RUNEC_UI_SPELL_X0, RUNEC_UI_SPELL_Y0,
                                     RUNEC_UI_SPELL_STEP_X,
                                     RUNEC_UI_SPELL_STEP_Y,
                                     RUNEC_UI_SPELL_ICON_SIZE,
                                     RUNEC_UI_SPELL_ICON_SIZE);
        if (!g_standard_spell_slots[i].name)
            continue;
        DrawRectangleRec(r, (Color){12, 12, 28, 105});
        char name[32];
        snprintf(name, sizeof(name), "standard_spell_on_%d",
                 g_standard_spell_slots[i].standard_icon_frame);
        int drew = draw_asset_centered(ui, name, r, RUNEC_UI_SPELL_ICON_SIZE,
                                       RUNEC_UI_SPELL_ICON_SIZE, WHITE);
        if (!drew)
            draw_centered_text(ui, TextFormat("%d", i + 1), r, 10, OSRS_ORANGE);
    }
}

static void draw_skills(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    int count = (int)(sizeof(RUNEC_OSRS_SKILLS) / sizeof(RUNEC_OSRS_SKILLS[0]));
    for (int i = 0; i < count; i++) {
        Rectangle r = skill_slot_rect(layout, i);
        DrawRectangleRec(r, (Color){72, 70, 60, 232});
        DrawLineEx((Vector2){r.x, r.y}, (Vector2){r.x + r.width - 1, r.y}, 1,
                   (Color){139, 130, 104, 255});
        DrawLineEx((Vector2){r.x, r.y}, (Vector2){r.x, r.y + r.height - 1}, 1,
                   (Color){139, 130, 104, 255});
        DrawLineEx((Vector2){r.x, r.y + r.height - 1},
                   (Vector2){r.x + r.width - 1, r.y + r.height - 1}, 1,
                   (Color){28, 25, 21, 255});
        DrawLineEx((Vector2){r.x + r.width - 1, r.y},
                   (Vector2){r.x + r.width - 1, r.y + r.height - 1}, 1,
                   (Color){28, 25, 21, 255});
        char icon[32];
        snprintf(icon, sizeof(icon), "skill_icon_%d", RUNEC_OSRS_SKILLS[i].icon_index);
        draw_asset_centered(ui, icon, (Rectangle){r.x + 3, r.y + 3, 24, 24}, 24, 24, WHITE);
        int current_level = i < RUNEC_UI_SKILL_COUNT && ui->skill_current[i] > 0
                          ? ui->skill_current[i] : 1;
        int base_level = i < RUNEC_UI_SKILL_COUNT && ui->skill_base[i] > 0
                       ? ui->skill_base[i] : current_level;
        char cur[16];
        char base[16];
        snprintf(cur, sizeof(cur), "%d", current_level);
        snprintf(base, sizeof(base), "%d", base_level);
        Color cur_color = current_level < base_level ? (Color){220, 45, 31, 255}
                        : current_level > base_level ? OSRS_GREEN : OSRS_YELLOW;
        draw_text_shadow(ui, cur, r.x + 39, r.y + 2, 10, cur_color);
        draw_text_shadow(ui, base, r.x + 39, r.y + 17, 9, OSRS_GREEN);
    }

    Rectangle total = {layout->side_content.x + RUNEC_OSRS_STATS_TOTAL.x,
                       layout->side_content.y + RUNEC_OSRS_STATS_TOTAL.y,
                       RUNEC_OSRS_STATS_TOTAL.width, RUNEC_OSRS_STATS_TOTAL.height};
    DrawRectangleRec(total, (Color){7, 7, 7, 238});
    DrawRectangleLinesEx(total, 1, (Color){99, 91, 68, 255});
    char total_text[32];
    snprintf(total_text, sizeof(total_text), "Total level: %d",
             ui->skill_total > 0 ? ui->skill_total : 0);
    draw_centered_text(ui, total_text, total, 10, OSRS_YELLOW);
}

static void draw_combat_box(const RuneCUiState *ui, Rectangle r, int selected) {
    const char *asset = selected ? "combatboxes_1" : "combatboxes_0";
    runec_ui_draw_asset(&ui->assets, asset, r, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, asset)) {
        DrawRectangleRec(r, selected ? (Color){83, 61, 43, 245} : (Color){45, 39, 31, 235});
        DrawRectangleLinesEx(r, 1, selected ? OSRS_YELLOW : (Color){103, 89, 63, 255});
    }
    if (selected)
        DrawRectangleRec(r, (Color){120, 27, 20, 54});
}

static void draw_combat(const RuneCUiState *ui, const RuneCUiLayout *layout) {
    Rectangle header = side_ref_rect(layout, RUNEC_OSRS_COMBAT_HEADER);
    const char *weapon_name = ui->combat_weapon_name[0]
        ? ui->combat_weapon_name : "Unarmed";
    draw_centered_text(ui, weapon_name, side_ref_rect(layout, RUNEC_OSRS_COMBAT_TITLE),
                       13, OSRS_ORANGE);
    char level[32];
    snprintf(level, sizeof(level), "Combat Lvl: %d", ui->combat_level);
    draw_centered_text(ui, level, side_ref_rect(layout, RUNEC_OSRS_COMBAT_LEVEL),
                       12, OSRS_ORANGE);
    DrawLineEx((Vector2){header.x + 10, header.y + 42},
               (Vector2){header.x + header.width - 10, header.y + 42}, 1,
               (Color){75, 64, 45, 180});

    int visible_slot = 0;
    int layout_count = (int)(sizeof(RUNEC_OSRS_COMBAT_STYLES)
        / sizeof(RUNEC_OSRS_COMBAT_STYLES[0]));
    for (int i = 0; i < RUNEC_UI_COMBAT_STYLE_COUNT && visible_slot < layout_count; i++) {
        const RuneCUiCombatStyleOption *option = &ui->combat_styles[i];
        if (!option->visible)
            continue;
        const RuneCUiCombatStyleRef *style =
            &RUNEC_OSRS_COMBAT_STYLES[visible_slot++];
        int selected = combat_style_option_selected(ui, option);
        Rectangle button = side_ref_rect(layout, style->rect);
        draw_combat_box(ui, button, selected);
        draw_asset_centered(ui, option->icon_asset, side_ref_rect(layout, style->icon_rect),
                            34, 24, WHITE);
        draw_centered_text(ui, option->label, side_ref_rect(layout, style->text_rect),
                           10, selected ? OSRS_YELLOW : OSRS_ORANGE);
    }

    Rectangle retaliate = side_ref_rect(layout, RUNEC_OSRS_COMBAT_RETALIATE);
    draw_combat_box(ui, retaliate, ui->auto_retaliate);
    draw_asset_centered(ui, "combat_shield", side_ref_rect(layout, RUNEC_OSRS_COMBAT_RETALIATE_ICON),
                        26, 39, WHITE);
    draw_centered_text(ui, ui->auto_retaliate ? "Auto Retaliate" : "Retaliate Off",
                       side_ref_rect(layout, RUNEC_OSRS_COMBAT_RETALIATE_TEXT), 11, OSRS_ORANGE);

    Rectangle spec = side_ref_rect(layout, RUNEC_OSRS_COMBAT_SPECIAL_BAR);
    runec_ui_draw_asset(&ui->assets, "combatboxes_special_attack", spec, WHITE);
    if (!runec_ui_asset_ready(&ui->assets, "combatboxes_special_attack"))
        DrawRectangleRec(spec, (Color){32, 28, 22, 235});
    Rectangle empty = {spec.x + 2, spec.y + 7, spec.width - 4, 12};
    DrawRectangleRec(empty, (Color){115, 6, 6, 255});
    Rectangle fill = empty;
    fill.width *= (float)ui->special_attack_energy / 100.0f;
    DrawRectangleRec(fill, ui->special_attack_enabled ? OSRS_GREEN : (Color){57, 125, 59, 255});
    DrawRectangleLinesEx((Rectangle){spec.x + 2, spec.y + 6, spec.width - 4, 14}, 1,
                         (Color){44, 42, 35, 255});
    char spec_text[32];
    snprintf(spec_text, sizeof(spec_text), "Special Attack: %d%%", ui->special_attack_energy);
    draw_centered_text(ui, spec_text, spec, 10, OSRS_YELLOW);

    const RuneCUiCombatStyleOption *selected_style =
        selected_combat_style_option(ui);
    const char *mode = selected_style ? selected_style->mode : "Accurate";
    char category[64];
    snprintf(category, sizeof(category), "Attack style: %s", mode);
    draw_centered_text(ui, category, side_ref_rect(layout, RUNEC_OSRS_COMBAT_CATEGORY),
                       12, OSRS_ORANGE);
}

static void draw_placeholder_tab(const RuneCUiState *ui, const RuneCUiLayout *layout,
                                 const char *title, const char *body) {
    draw_centered_text(ui, title, (Rectangle){layout->side_content.x, layout->side_content.y + 26, 190, 20},
                       15, OSRS_YELLOW);
    draw_centered_text(ui, body, (Rectangle){layout->side_content.x + 10, layout->side_content.y + 105, 170, 36},
                       12, OSRS_ORANGE);
}

static void draw_side(RuneCUiState *ui, const RuneCUiLayout *layout) {
    draw_side_chrome(ui, layout);

    if (ui->active_tab == RUNEC_UI_TAB_COMBAT) {
        draw_combat(ui, layout);
        return;
    }

    switch (ui->active_tab) {
    case RUNEC_UI_TAB_INVENTORY:
        draw_inventory(ui, layout);
        break;
    case RUNEC_UI_TAB_EQUIPMENT:
        draw_equipment(ui, layout);
        break;
    case RUNEC_UI_TAB_PRAYER:
        draw_prayer(ui, layout);
        break;
    case RUNEC_UI_TAB_SPELLBOOK:
        draw_spellbook(ui, layout);
        break;
    case RUNEC_UI_TAB_SKILLS:
        draw_skills(ui, layout);
        break;
    case RUNEC_UI_TAB_COMBAT:
        draw_combat(ui, layout);
        break;
    case RUNEC_UI_TAB_QUESTS:
        draw_placeholder_tab(ui, layout, "Quest List", "Quest journal surface.");
        break;
    case RUNEC_UI_TAB_SETTINGS:
        draw_placeholder_tab(ui, layout, "Settings", "Viewer options surface.");
        break;
    case RUNEC_UI_TAB_CLAN_CHAT:
        break;
    case RUNEC_UI_TAB_FRIENDS:
        break;
    default:
        break;
    }
}

static void draw_context(const RuneCUiState *ui) {
    if (!ui->context_open)
        return;
    Rectangle box = {ui->context_pos.x, ui->context_pos.y,
                     158.0f, 24.0f + ui->context_action_count * 20.0f};
    DrawRectangleRec(box, (Color){53, 44, 31, 244});
    DrawRectangleLinesEx(box, 1, (Color){170, 137, 72, 255});
    draw_text_shadow(ui, ui->context_title, box.x + 5, box.y + 4, 11, OSRS_YELLOW);
    for (int i = 0; i < ui->context_action_count; i++) {
        Rectangle item = {box.x + 4, box.y + 22 + i * 20.0f, box.width - 8, 18};
        DrawRectangleRec(item, (Color){28, 23, 17, 215});
        draw_text_shadow(ui, ui->context_actions[i], item.x + 4, item.y + 3, 11, OSRS_ORANGE);
    }
}

static void draw_selected_target(const RuneCUiState *ui) {
    if (ui->drag.active && ui->drag.source_kind == RUNEC_UI_CONTEXT_INVENTORY
            && ui->drag.source_slot >= 0
            && ui->drag.source_slot < RUNEC_UI_INV_SLOT_COUNT
            && ui->inventory[ui->drag.source_slot].enabled) {
        Vector2 mouse = GetMousePosition();
        Rectangle r = {mouse.x - 16, mouse.y - 16,
                       RUNEC_OSRS_INVENTORY_SLOT_W,
                       RUNEC_OSRS_INVENTORY_SLOT_H};
        DrawRectangleRec(r, (Color){0, 0, 0, 80});
        draw_inventory_item(ui, &ui->inventory[ui->drag.source_slot], r);
    }
    if (ui->selected_target.kind == RUNEC_UI_SELECTED_NONE)
        return;
    Vector2 mouse = GetMousePosition();
    char text[96];
    snprintf(text, sizeof(text), "%s %s ->", ui->selected_target.verb,
             ui->selected_target.label);
    Font font = runec_ui_font_for_size(&ui->assets, 12.0f);
    int width = (int)ceilf(MeasureTextEx(
        font, text, 12.0f, 0.0f).x) + 10;
    Rectangle box = {mouse.x + 12, mouse.y + 12, (float)width, 20};
    DrawRectangleRec(box, (Color){28, 23, 17, 230});
    DrawRectangleLinesEx(box, 1, (Color){170, 137, 72, 255});
    draw_text_shadow(ui, text, box.x + 5, box.y + 4, 11, OSRS_YELLOW);
}

void runec_ui_draw(RuneCUiState *ui, int screen_w, int screen_h) {
    RuneCUiLayout layout;
    ui_layout(screen_w, screen_h, &layout);

    draw_chat_panel_chrome(ui, &layout);
    draw_minimap(ui, &layout);
    draw_side(ui, &layout);
    draw_selected_target(ui);
    draw_context(ui);
}

Rectangle runec_ui_chat_panel_rect(int screen_w, int screen_h) {
    RuneCUiLayout layout;
    ui_layout(screen_w, screen_h, &layout);
    return layout.chat;
}
