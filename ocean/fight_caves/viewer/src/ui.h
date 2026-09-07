#ifndef RUNEC_VIEWER_UI_H
#define RUNEC_VIEWER_UI_H

#include "raylib.h"
#include "ui_assets.h"
#include <stddef.h>
#include <stdint.h>

#define RUNEC_UI_INV_SLOT_COUNT 28
#define RUNEC_UI_EQUIP_SLOT_COUNT 14
#define RUNEC_UI_CHAT_INPUT_MAX 128
#define RUNEC_UI_CONTEXT_ACTIONS 5
#define RUNEC_UI_MINIMAP_DOTS 256
#define RUNEC_UI_ITEM_ICON_CACHE 512
#define RUNEC_UI_SKILL_COUNT 24
#define RUNEC_UI_COMBAT_STYLE_COUNT 4

typedef enum RuneCUiTab {
    RUNEC_UI_TAB_NONE = -1,
    RUNEC_UI_TAB_COMBAT = 0,
    RUNEC_UI_TAB_SKILLS,
    RUNEC_UI_TAB_QUESTS,
    RUNEC_UI_TAB_INVENTORY,
    RUNEC_UI_TAB_EQUIPMENT,
    RUNEC_UI_TAB_PRAYER,
    RUNEC_UI_TAB_SPELLBOOK,
    RUNEC_UI_TAB_SETTINGS,
    RUNEC_UI_TAB_CLAN_CHAT,
    RUNEC_UI_TAB_FRIENDS,
    RUNEC_UI_TAB_COUNT
} RuneCUiTab;

typedef enum RuneCUiIntentKind {
    RUNEC_UI_INTENT_NONE = 0,
    RUNEC_UI_INTENT_TAB,
    RUNEC_UI_INTENT_INVENTORY_SLOT,
    RUNEC_UI_INTENT_EQUIPMENT_SLOT,
    RUNEC_UI_INTENT_PRAYER_SLOT,
    RUNEC_UI_INTENT_QUICK_PRAYER_SLOT,
    RUNEC_UI_INTENT_QUICK_PRAYER_TOGGLE,
    RUNEC_UI_INTENT_AUTOCAST_SPELL,
    RUNEC_UI_INTENT_SKILL_SLOT,
    RUNEC_UI_INTENT_MINIMAP_CLICK,
    RUNEC_UI_INTENT_RUN_TOGGLE,
    RUNEC_UI_INTENT_COMBAT_STYLE,
    RUNEC_UI_INTENT_AUTO_RETALIATE,
    RUNEC_UI_INTENT_SPECIAL_ATTACK,
    RUNEC_UI_INTENT_CONTEXT_ACTION,
    RUNEC_UI_INTENT_INVENTORY_ACTION,
    RUNEC_UI_INTENT_EQUIPMENT_ACTION,
    RUNEC_UI_INTENT_INVENTORY_DRAG,
    RUNEC_UI_INTENT_SELECTED_ITEM,
    RUNEC_UI_INTENT_SELECTED_SPELL,
    RUNEC_UI_INTENT_SELECTED_ITEM_ON_ITEM,
    RUNEC_UI_INTENT_SELECTED_SPELL_ON_ITEM,
    RUNEC_UI_INTENT_SELECTED_TARGET_CANCEL
} RuneCUiIntentKind;

typedef struct RuneCUiIntent {
    RuneCUiIntentKind kind;
    int primary;
    int secondary;
    Vector2 position;
    char text[RUNEC_UI_CHAT_INPUT_MAX];
} RuneCUiIntent;

typedef struct RuneCUiSlot {
    uint32_t item_id;
    uint32_t icon_item_id;
    int quantity;
    char label[24];
    int enabled;
} RuneCUiSlot;

typedef enum RuneCUiMinimapDotKind {
    RUNEC_UI_MINIMAP_DOT_NPC = 0,
    RUNEC_UI_MINIMAP_DOT_PLAYER,
    RUNEC_UI_MINIMAP_DOT_DESTINATION
} RuneCUiMinimapDotKind;

typedef struct RuneCUiMinimapDot {
    float dx;
    float dy;
    RuneCUiMinimapDotKind kind;
} RuneCUiMinimapDot;

typedef struct RuneCUiItemIcon {
    uint32_t item_id;
    Texture2D texture;
    int ready;
} RuneCUiItemIcon;

typedef struct RuneCUiCombatStyleOption {
    int visible;
    int style_index;
    char label[24];
    char mode[32];
    char icon_asset[32];
} RuneCUiCombatStyleOption;

typedef enum RuneCUiContextSourceKind {
    RUNEC_UI_CONTEXT_NONE = 0,
    RUNEC_UI_CONTEXT_INVENTORY,
    RUNEC_UI_CONTEXT_EQUIPMENT,
    RUNEC_UI_CONTEXT_PRAYER,
    RUNEC_UI_CONTEXT_SPELL
} RuneCUiContextSourceKind;

typedef enum RuneCUiSelectedTargetKind {
    RUNEC_UI_SELECTED_NONE = 0,
    RUNEC_UI_SELECTED_ITEM,
    RUNEC_UI_SELECTED_SPELL
} RuneCUiSelectedTargetKind;

typedef struct RuneCUiSelectedTarget {
    RuneCUiSelectedTargetKind kind;
    int source_slot;
    uint32_t source_item_id;
    char label[48];
    char verb[24];
} RuneCUiSelectedTarget;

typedef struct RuneCUiDragState {
    int active;
    RuneCUiContextSourceKind source_kind;
    int source_slot;
    Vector2 start;
} RuneCUiDragState;

typedef struct RuneCUiState {
    RuneCUiTab active_tab;
    RuneCUiIntent last_intent;
    float tab_press_timer[RUNEC_UI_TAB_COUNT];

    RuneCUiSlot inventory[RUNEC_UI_INV_SLOT_COUNT];
    RuneCUiSlot equipment[RUNEC_UI_EQUIP_SLOT_COUNT];
    int selected_inventory_slot;
    int selected_equipment_slot;
    int selected_combat_style;
    int auto_retaliate;
    int special_attack_enabled;
    int special_attack_energy;
    int combat_weapon_category;
    char combat_weapon_name[64];
    RuneCUiCombatStyleOption combat_styles[RUNEC_UI_COMBAT_STYLE_COUNT];

    int hitpoints;
    int hitpoints_max;
    int prayer_points;
    int prayer_points_max;
    uint32_t active_prayers;
    int run_energy;
    int run_enabled;
    int combat_level;
    int skill_current[RUNEC_UI_SKILL_COUNT];
    int skill_base[RUNEC_UI_SKILL_COUNT];
    int skill_total;

    int context_open;
    Vector2 context_pos;
    char context_title[48];
    char context_actions[RUNEC_UI_CONTEXT_ACTIONS][32];
    int context_action_count;
    RuneCUiContextSourceKind context_source_kind;
    int context_source_slot;
    uint32_t context_source_item_id;

    RuneCUiSelectedTarget selected_target;
    RuneCUiDragState drag;

    RuneCUiMinimapDot minimap_dots[RUNEC_UI_MINIMAP_DOTS];
    int minimap_dot_count;
    float minimap_rotation;
    Texture2D minimap_texture;
    int minimap_texture_ready;
    RuneCUiItemIcon item_icons[RUNEC_UI_ITEM_ICON_CACHE];
    int item_icon_count;

    RuneCUiAssets assets;
} RuneCUiState;

void runec_ui_init(RuneCUiState *ui);
void runec_ui_shutdown(RuneCUiState *ui);
void runec_ui_clear_minimap(RuneCUiState *ui);
void runec_ui_add_minimap_dot(RuneCUiState *ui, float dx, float dy,
                              RuneCUiMinimapDotKind kind);
void runec_ui_update_minimap(RuneCUiState *ui, const Color *pixels,
                             int width, int height);
void runec_ui_set_minimap_rotation(RuneCUiState *ui, float radians);
void runec_ui_set_item_icon(RuneCUiState *ui, uint32_t icon_item_id, Texture2D texture);
void runec_ui_set_combat_weapon_name(RuneCUiState *ui, const char *name);
void runec_ui_set_combat_style_profile(RuneCUiState *ui, int core_weapon_category);
void runec_ui_clear_selected_target(RuneCUiState *ui);
int runec_ui_handle_input(RuneCUiState *ui, int screen_w, int screen_h);
void runec_ui_draw(RuneCUiState *ui, int screen_w, int screen_h);
Rectangle runec_ui_chat_panel_rect(int screen_w, int screen_h);
const char *runec_ui_tab_name(RuneCUiTab tab);

#endif
