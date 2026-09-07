#ifndef RUNEC_VIEWER_UI_REFERENCE_H
#define RUNEC_VIEWER_UI_REFERENCE_H

#include "raylib.h"

/*
 * OSRS UI reference constants copied from the local rsmod/model_dump interface
 * dumps. These mirror what Void/RSMod open server-side; the viewer should use
 * these as layout authority instead of hand-tuned gameframe coordinates.
 *
 * Source files:
 * - data/source/model_dump/osrs-dumps/interface/toplevel_osrs_stretch.if3
 * - data/source/model_dump/osrs-dumps/interface/orbs.if3
 * - data/source/model_dump/osrs-dumps/interface/stats.if3
 * - data/source/model_dump/osrs-dumps/interface/inventory.if3
 * - data/source/model_dump/osrs-dumps/interface/combat_interface.if3
 * - data/source/model_dump/osrs-dumps/interface/wornitems.if3
 */

#define RUNEC_IFACE_TOPLEVEL_OSRS_STRETCH 161
#define RUNEC_IFACE_ORBS 160
#define RUNEC_IFACE_INVENTORY 149
#define RUNEC_IFACE_STATS 320
#define RUNEC_IFACE_WORNITEMS 387
#define RUNEC_IFACE_PRAYERBOOK 541
#define RUNEC_IFACE_MAGIC_SPELLBOOK 218
#define RUNEC_IFACE_COMBAT_INTERFACE 593

#define RUNEC_OSRS_MAP_CONTAINER_W 211.0f
#define RUNEC_OSRS_MAP_CONTAINER_H 207.0f
#define RUNEC_OSRS_MINIMAP_X 53.0f
#define RUNEC_OSRS_MINIMAP_Y 8.0f
#define RUNEC_OSRS_MINIMAP_W 152.0f
#define RUNEC_OSRS_MINIMAP_H 152.0f
#define RUNEC_OSRS_MAP_SURROUND_X 29.0f
#define RUNEC_OSRS_MAP_SURROUND_Y 0.0f
#define RUNEC_OSRS_MAP_SURROUND_W 182.0f
#define RUNEC_OSRS_MAP_SURROUND_H 166.0f
#define RUNEC_OSRS_COMPASS_X 34.0f
#define RUNEC_OSRS_COMPASS_Y 5.0f
#define RUNEC_OSRS_COMPASS_W 35.0f
#define RUNEC_OSRS_COMPASS_H 35.0f

#define RUNEC_OSRS_ORBS_X 0.0f
#define RUNEC_OSRS_ORBS_Y 10.0f
#define RUNEC_OSRS_ORBS_W 207.0f
#define RUNEC_OSRS_ORBS_H 197.0f
#define RUNEC_OSRS_HP_X 0.0f
#define RUNEC_OSRS_HP_Y 37.0f
#define RUNEC_OSRS_PRAYER_X 0.0f
#define RUNEC_OSRS_PRAYER_Y 71.0f
#define RUNEC_OSRS_RUN_X 10.0f
#define RUNEC_OSRS_RUN_Y 103.0f
#define RUNEC_OSRS_SPEC_X 32.0f
#define RUNEC_OSRS_SPEC_Y 128.0f
#define RUNEC_OSRS_WORLDMAP_X 177.0f
#define RUNEC_OSRS_WORLDMAP_Y 137.0f

#define RUNEC_OSRS_CHAT_W 519.0f
#define RUNEC_OSRS_CHAT_H 165.0f
#define RUNEC_OSRS_SIDE_MENU_W 241.0f
#define RUNEC_OSRS_SIDE_MENU_H 335.0f
#define RUNEC_OSRS_SIDE_CONTENT_X 25.0f
#define RUNEC_OSRS_SIDE_CONTENT_Y 37.0f
#define RUNEC_OSRS_SIDE_CONTENT_W 190.0f
#define RUNEC_OSRS_SIDE_CONTENT_H 261.0f
#define RUNEC_OSRS_SIDE_TOP_Y 0.0f
#define RUNEC_OSRS_SIDE_BOTTOM_Y 298.0f

#define RUNEC_OSRS_INVENTORY_SLOT_COLS 4
#define RUNEC_OSRS_INVENTORY_SLOT_COUNT 28
#define RUNEC_OSRS_INVENTORY_SLOT_X 14.0f
#define RUNEC_OSRS_INVENTORY_SLOT_Y 8.0f
#define RUNEC_OSRS_INVENTORY_SLOT_STEP_X 42.0f
#define RUNEC_OSRS_INVENTORY_SLOT_STEP_Y 36.0f
#define RUNEC_OSRS_INVENTORY_SLOT_W 32.0f
#define RUNEC_OSRS_INVENTORY_SLOT_H 32.0f

typedef struct RuneCUiStoneRef {
    int logical_tab;
    const char *stone_asset;
    const char *icon_asset;
    Rectangle rect;
    Rectangle icon_rect;
} RuneCUiStoneRef;

static const RuneCUiStoneRef RUNEC_OSRS_SIDE_STONES[] = {
    {0, "side_stone_highlights_0", "side_icon_combat",    {0,   0, 38, 36}, {4,   0, 33, 36}},
    {1, "side_stone_highlights_4", "side_icon_stats",     {38,  0, 33, 36}, {38,  0, 33, 36}},
    {2, "side_stone_highlights_4", "side_icon_quests",    {71,  0, 38, 36}, {71,  0, 33, 36}},
    {3, "side_stone_highlights_4", "side_icon_inventory", {104, 0, 33, 36}, {104, 0, 33, 36}},
    {4, "side_stone_highlights_4", "side_icon_equipment", {137, 0, 33, 36}, {137, 0, 33, 36}},
    {5, "side_stone_highlights_4", "side_icon_prayer",    {170, 0, 33, 36}, {170, 0, 33, 36}},
    {6, "side_stone_highlights_1", "side_icon_magic",     {203, 0, 38, 36}, {204, 0, 33, 36}},
    {8, "side_stone_highlights_2", "side_icon_clan",     {0,   0, 38, 36}, {4,   0, 33, 36}},
    {9, "side_stone_highlights_4", "side_icon_friends",  {38,  0, 33, 36}, {38,  0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_grouping", {71,  0, 33, 36}, {71,  0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_logout",   {104, 0, 33, 36}, {104, 0, 33, 36}},
    {7, "side_stone_highlights_4", "side_icon_options",   {137, 0, 33, 36}, {137, 0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_emotes",   {170, 0, 33, 36}, {170, 0, 33, 36}},
    {-1, "side_stone_highlights_3", "side_icon_music",    {203, 0, 38, 36}, {204, 0, 33, 36}},
};

typedef struct RuneCUiSkillRef {
    const char *name;
    int icon_index;
    Rectangle rect;
} RuneCUiSkillRef;

static const RuneCUiSkillRef RUNEC_OSRS_SKILLS[] = {
    {"Attack",       0,  {1,   1,   62, 30}},
    {"Strength",     1,  {1,   31,  62, 30}},
    {"Defence",      2,  {1,   61,  62, 30}},
    {"Ranged",       3,  {1,   91,  62, 30}},
    {"Prayer",       4,  {1,   121, 62, 30}},
    {"Magic",        5,  {1,   151, 62, 30}},
    {"Runecraft",    18, {1,   181, 62, 30}},
    {"Construction", 22, {1,   211, 62, 32}},
    {"Hitpoints",    6,  {64,  1,   62, 30}},
    {"Agility",      7,  {64,  31,  62, 30}},
    {"Herblore",     8,  {64,  61,  62, 30}},
    {"Thieving",     9,  {64,  91,  62, 30}},
    {"Crafting",     10, {64,  121, 62, 30}},
    {"Fletching",    11, {64,  151, 62, 30}},
    {"Slayer",       19, {64,  181, 62, 30}},
    {"Hunter",       21, {64,  211, 62, 32}},
    {"Mining",       12, {127, 1,   62, 30}},
    {"Smithing",     13, {127, 31,  62, 30}},
    {"Fishing",      14, {127, 61,  62, 30}},
    {"Cooking",      15, {127, 91,  62, 30}},
    {"Firemaking",   16, {127, 121, 62, 30}},
    {"Woodcutting",  17, {127, 151, 62, 30}},
    {"Farming",      20, {127, 181, 62, 30}},
    {"Sailing",      23, {127, 211, 62, 32}},
};

static const Rectangle RUNEC_OSRS_STATS_TOTAL = {0, 241, 190, 19};

typedef struct RuneCUiCombatStyleRef {
    const char *label;
    const char *mode;
    const char *icon_asset;
    int visible;
    Rectangle rect;
    Rectangle icon_rect;
    Rectangle text_rect;
} RuneCUiCombatStyleRef;

static const RuneCUiCombatStyleRef RUNEC_OSRS_COMBAT_STYLES[] = {
    {"Flick",   "Accurate",  "sideicons_interface_0", 1, {20,  46, 68, 47}, {37,  51, 34, 24}, {20,  67, 68, 13}},
    {"Lash",    "Controlled", "sideicons_interface_1", 1, {102, 46, 68, 47}, {119, 51, 34, 24}, {102, 76, 68, 13}},
    {"Deflect", "Defensive",  "sideicons_interface_2", 1, {20,  99, 68, 47}, {37, 104, 34, 24}, {20, 129, 68, 13}},
    {"",        "",           "sideicons_interface_3", 0, {102, 99, 68, 47}, {119,104, 34, 24}, {102,129, 68, 13}},
};

static const Rectangle RUNEC_OSRS_COMBAT_HEADER = {10, 0, 170, 44};
static const Rectangle RUNEC_OSRS_COMBAT_TITLE = {10, 6, 170, 14};
static const Rectangle RUNEC_OSRS_COMBAT_LEVEL = {10, 26, 170, 12};
static const Rectangle RUNEC_OSRS_COMBAT_RETALIATE = {20, 153, 150, 47};
static const Rectangle RUNEC_OSRS_COMBAT_RETALIATE_ICON = {27, 156, 26, 39};
static const Rectangle RUNEC_OSRS_COMBAT_RETALIATE_TEXT = {58, 156, 108, 39};
static const Rectangle RUNEC_OSRS_COMBAT_SPECIAL_BAR = {20, 200, 150, 26};
static const Rectangle RUNEC_OSRS_COMBAT_CATEGORY = {0, 233, 190, 28};

typedef struct RuneCUiWornButtonRef {
    const char *asset;
    Rectangle rect;
    Rectangle icon_rect;
} RuneCUiWornButtonRef;

static const RuneCUiWornButtonRef RUNEC_OSRS_WORN_BUTTONS[] = {
    {"options_icons_16", {7,   208, 40, 40}, {10,  210, 32, 32}},
    {"options_icons_28", {52,  208, 40, 40}, {56,  212, 32, 32}},
    {"options_icons_18", {97,  208, 40, 40}, {99,  211, 34, 34}},
    {"whistle",          {142, 208, 40, 40}, {145, 211, 32, 32}},
};

#endif
