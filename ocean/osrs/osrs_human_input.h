#ifndef OSRS_HUMAN_INPUT_H
#define OSRS_HUMAN_INPUT_H

#include "osrs_types.h"
#include "osrs_items.h"
#include "osrs_human_input_types.h"
#include "osrs_encounter.h"

struct RenderClient;

static void human_set_click_cross(HumanInput* hi, int screen_x, int screen_y, int is_attack) {
    hi->click_screen_x = screen_x;
    hi->click_screen_y = screen_y;
    hi->click_cross_timer = 0;
    hi->click_cross_active = 1;
    hi->click_is_attack = is_attack;
}



static int human_gui_rect_contains(Rectangle rect, int mouse_x, int mouse_y) {
    return mouse_x >= rect.x && mouse_x < rect.x + rect.width &&
           mouse_y >= rect.y && mouse_y < rect.y + rect.height;
}

static int human_gui_prayer_idx_at(GuiState* gs, int mouse_x, int mouse_y) {
    int cols = GUI_PRAYER_GRID_COLS;
    int gap, icon_sz, gx, gy;
    gui_prayer_grid_metrics(gs, &gx, &gy, &icon_sz, &gap);

    if (mouse_x < gx || mouse_y < gy) return -1;
    int col = (mouse_x - gx) / (icon_sz + gap);
    int row = (mouse_y - gy) / (icon_sz + gap);
    if (col < 0 || col >= cols) return -1;

    int idx = row * cols + col;
    if (idx < 0 || idx >= GUI_PRAYER_GRID_COUNT) return -1;

    int cell_x = gx + col * (icon_sz + gap);
    int cell_y = gy + row * (icon_sz + gap);
    if (mouse_x >= cell_x + icon_sz || mouse_y >= cell_y + icon_sz) return -1;

    return idx;
}

static const char* human_gui_prayer_name(GuiPrayerIdx pidx) {
    switch (pidx) {
        case GUI_PRAY_PROTECT_MAGIC:    return "Protect from Magic";
        case GUI_PRAY_PROTECT_MISSILES: return "Protect from Missiles";
        case GUI_PRAY_PROTECT_MELEE:    return "Protect from Melee";
        case GUI_PRAY_SMITE:            return "Smite";
        case GUI_PRAY_REDEMPTION:       return "Redemption";
        case GUI_PRAY_PIETY:            return "Piety";
        case GUI_PRAY_RIGOUR:           return "Rigour";
        case GUI_PRAY_AUGURY:           return "Augury";
        default: return NULL;
    }
}

static int human_apply_prayer_idx(HumanInput* hi, Player* p, GuiPrayerIdx pidx) {
    static const struct {
        GuiPrayerIdx idx;
        OverheadPrayer target;
        int refresh_action;
    } overhead_rows[] = {
        { GUI_PRAY_PROTECT_MAGIC, PRAYER_PROTECT_MAGIC, ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC },
        { GUI_PRAY_PROTECT_MISSILES, PRAYER_PROTECT_RANGED, ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED },
        { GUI_PRAY_PROTECT_MELEE, PRAYER_PROTECT_MELEE, ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE },
        { GUI_PRAY_SMITE, PRAYER_SMITE, ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE },
        { GUI_PRAY_REDEMPTION, PRAYER_REDEMPTION, ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION },
    };
    static const struct {
        GuiPrayerIdx idx;
        OffensivePrayer target;
        int refresh_action;
    } offensive_rows[] = {
        { GUI_PRAY_PIETY, OFFENSIVE_PRAYER_PIETY, ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY },
        { GUI_PRAY_RIGOUR, OFFENSIVE_PRAYER_RIGOUR, ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR },
        { GUI_PRAY_AUGURY, OFFENSIVE_PRAYER_AUGURY, ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY },
    };
    for (size_t i = 0; i < sizeof(overhead_rows) / sizeof(overhead_rows[0]); i++) {
        if (overhead_rows[i].idx != pidx) continue;
        hi->pending_prayer = p->prayer == overhead_rows[i].target
            ? ENCOUNTER_OVERHEAD_OFF : overhead_rows[i].refresh_action;
        human_input_queue_overhead_prayer(hi, hi->pending_prayer);
        return 1;
    }
    for (size_t i = 0; i < sizeof(offensive_rows) / sizeof(offensive_rows[0]); i++) {
        if (offensive_rows[i].idx != pidx) continue;
        hi->pending_offensive_prayer =
            p->offensive_prayer == offensive_rows[i].target
                ? ENCOUNTER_OFFENSIVE_OFF : offensive_rows[i].refresh_action;
        human_input_queue_offensive_prayer(hi, hi->pending_offensive_prayer);
        return 1;
    }
    return 0;
}

static int human_gui_spell_idx_at(GuiState* gs, int mouse_x, int mouse_y) {
    int cols = GUI_SPELL_GRID_COLS;
    int gx, gy;
    gui_spell_grid_origin(gs, &gx, &gy);

    if (mouse_x < gx || mouse_y < gy) return -1;
    int col = (mouse_x - gx) / GUI_SPELL_PITCH_X;
    int row = (mouse_y - gy) / GUI_SPELL_PITCH_Y;
    if (col < 0 || col >= cols) return -1;

    int idx = row * cols + col;
    if (idx < 0 || idx >= GUI_SPELL_GRID_COUNT) return -1;

    int cell_x = gx + col * GUI_SPELL_PITCH_X;
    int cell_y = gy + row * GUI_SPELL_PITCH_Y;
    if (mouse_x >= cell_x + GUI_SPELL_ICON_PX || mouse_y >= cell_y + GUI_SPELL_ICON_PX)
        return -1;

    return GUI_SPELL_GRID[idx].idx;
}

static const char* human_gui_spell_name(GuiSpellIdx sidx) {
    for (int i = 0; i < GUI_SPELL_GRID_COUNT; i++) {
        if (GUI_SPELL_GRID[i].idx == sidx) return GUI_SPELL_GRID[i].name;
    }
    return NULL;
}

static int human_select_spell_idx(HumanInput* hi, GuiSpellIdx sidx) {
    if (!gui_spell_castable(sidx)) return 0;

    if (gui_spell_is_ice(sidx)) {
        human_input_apply_ui_intent(
            hi, osrs_ui_intent_select_spell(PVP_ATTACK_ICE, (int)sidx));
        return 1;
    }

    if (gui_spell_is_blood(sidx)) {
        human_input_apply_ui_intent(
            hi, osrs_ui_intent_select_spell(PVP_ATTACK_BLOOD, (int)sidx));
        return 1;
    }

    return 0;
}

static int human_apply_selected_target_to_widget(HumanInput* hi, uint32_t widget_component_id) {
    if (hi->cursor_mode == CURSOR_ITEM_TARGET) {
        OsrsUiIntent intent = osrs_ui_intent_item_on_widget(
            hi->selected_item_inventory_slot,
            hi->selected_item_db_idx,
            hi->selected_item_osrs_id,
            widget_component_id);
        human_input_apply_ui_intent(hi, intent);
        return 1;
    }

    if (hi->cursor_mode == CURSOR_SPELL_TARGET) {
        human_input_apply_ui_intent(hi, osrs_ui_intent_spell_on_widget(
            hi->selected_spell,
            hi->selected_spell_gui_idx,
            widget_component_id));
        return 1;
    }

    return 0;
}

static int human_gui_combat_style_index_at(GuiState* gs, Player* p, int mouse_x, int mouse_y) {
    GuiCombatStyleOptions styles = gui_combat_style_options(p->equipped[GEAR_SLOT_WEAPON]);
    for (int i = 0; i < styles.count; i++) {
        if (human_gui_rect_contains(gui_combat_style_rect(gs, i), mouse_x, mouse_y)) {
            return i;
        }
    }
    return -1;
}

static int human_apply_combat_style(
    HumanInput* hi,
    GuiState* gs,
    Player* p,
    FightStyle fight_style
) {
    if (hi->enabled)
        human_input_queue_fight_style(hi, fight_style);
    else
        p->fight_style = fight_style;

    if (fight_style == FIGHT_STYLE_AUTOCAST ||
            fight_style == FIGHT_STYLE_DEFENSIVE_AUTOCAST) {
        int defensive = fight_style == FIGHT_STYLE_DEFENSIVE_AUTOCAST;
        if (hi->enabled) {
            human_input_queue_set_autocast(hi, gui_autocast_spell(p), defensive);
        } else {
            p->autocast_enabled = 1;
            p->autocast_defensive = defensive;
        }
    }

    (void)gs;
    return 1;
}

static int human_gui_autocast_button_hit(GuiState* gs, Player* p, int mouse_x, int mouse_y) {
    if (!item_supports_ancient_autocast(p->equipped[GEAR_SLOT_WEAPON])) return 0;
    return human_gui_rect_contains(gui_side_ref_rect(gs, gui_combat_autocast_rect()),
        mouse_x, mouse_y);
}

static int human_gui_autocast_spell_at(GuiState* gs, Player* p, int mouse_x, int mouse_y) {
    if (!item_supports_ancient_autocast(p->equipped[GEAR_SLOT_WEAPON])) return -1;
    int spells[2] = { ENCOUNTER_SPELL_BLOOD, ENCOUNTER_SPELL_ICE };
    for (int i = 0; i < 2; i++) {
        if (human_gui_rect_contains(gui_side_ref_rect(gs, gui_combat_autocast_spell_rect(i)),
                mouse_x, mouse_y)) {
            return spells[i];
        }
    }
    return -1;
}

static int human_apply_autocast_spell(
    HumanInput* hi,
    GuiState* gs,
    Player* p,
    int spell,
    int defensive
) {
    if (hi->enabled) {
        human_input_queue_set_autocast(hi, spell, defensive);
    } else {
        p->autocast_enabled = 1;
        p->autocast_defensive = defensive;
        p->autocast_spell = spell;
    }
    gs->autocast_selector_open = 0;
    return 1;
}

static int human_gui_spec_hit(GuiState* gs, int mouse_x, int mouse_y) {
    return human_gui_rect_contains(gui_side_ref_rect(gs, gui_combat_special_rect()),
        mouse_x, mouse_y);
}

static void human_apply_spec_toggle(HumanInput* hi) {
    hi->pending_spec = 1;
    human_input_queue_spec_toggle(hi);
}

static void human_handle_prayer_click(HumanInput* hi, GuiState* gs, Player* p,
                                       int mouse_x, int mouse_y) {
    int idx = human_gui_prayer_idx_at(gs, mouse_x, mouse_y);
    if (idx >= 0) {
        if (hi->cursor_mode != CURSOR_NORMAL) {
            human_apply_selected_target_to_widget(hi,
                osrs_ui_intent_widget_component_id(OSRS_UI_GROUP_PRAYERBOOK, idx));
            return;
        }
        human_apply_prayer_idx(hi, p, (GuiPrayerIdx)idx);
    }
}

static void human_handle_spell_click(HumanInput* hi, GuiState* gs,
                                      int mouse_x, int mouse_y) {
    int idx = human_gui_spell_idx_at(gs, mouse_x, mouse_y);
    if (idx >= 0) {
        if (hi->cursor_mode != CURSOR_NORMAL) {
            human_apply_selected_target_to_widget(hi,
                osrs_ui_intent_widget_component_id(OSRS_UI_GROUP_MAGIC_SPELLBOOK, idx));
            return;
        }
        human_select_spell_idx(hi, (GuiSpellIdx)idx);
    }
}

static void human_handle_combat_click(HumanInput* hi, GuiState* gs, Player* p,
                                       int mouse_x, int mouse_y) {
    GuiCombatStyleOptions styles = gui_combat_style_options(p->equipped[GEAR_SLOT_WEAPON]);

    int style_idx = human_gui_combat_style_index_at(gs, p, mouse_x, mouse_y);
    if (style_idx >= 0) {
        if (hi->cursor_mode != CURSOR_NORMAL) {
            human_apply_selected_target_to_widget(hi,
                osrs_ui_intent_widget_component_id(OSRS_UI_GROUP_COMBAT_INTERFACE, style_idx));
            return;
        }
        human_apply_combat_style(hi, gs, p, styles.values[style_idx]);
        return;
    }

    if (item_supports_ancient_autocast(p->equipped[GEAR_SLOT_WEAPON])) {
        if (human_gui_autocast_button_hit(gs, p, mouse_x, mouse_y)) {
            if (hi->cursor_mode != CURSOR_NORMAL) {
                human_apply_selected_target_to_widget(hi,
                    osrs_ui_intent_widget_component_id(OSRS_UI_GROUP_COMBAT_INTERFACE, 100));
                return;
            }
            gs->autocast_selector_open = !gs->autocast_selector_open;
            return;
        }

        if (gs->autocast_selector_open) {
            int spell = human_gui_autocast_spell_at(gs, p, mouse_x, mouse_y);
            if (spell >= 0) {
                if (hi->cursor_mode != CURSOR_NORMAL) {
                    human_apply_selected_target_to_widget(hi,
                        osrs_ui_intent_widget_component_id(
                            OSRS_UI_GROUP_COMBAT_INTERFACE, 110 + spell));
                    return;
                }
                int defensive = p->fight_style == FIGHT_STYLE_DEFENSIVE_AUTOCAST ||
                    p->autocast_defensive;
                human_apply_autocast_spell(hi, gs, p, spell, defensive);
                return;
            }
        }
    }

    if (human_gui_spec_hit(gs, mouse_x, mouse_y)) {
        if (hi->cursor_mode != CURSOR_NORMAL) {
            human_apply_selected_target_to_widget(hi,
                osrs_ui_intent_widget_component_id(OSRS_UI_GROUP_COMBAT_INTERFACE, 120));
            return;
        }
        human_apply_spec_toggle(hi);
    }
}

#define CLICK_CROSS_NUM_FRAMES 4
#define CLICK_CROSS_ANIM_TICKS 20

static void human_draw_click_cross(HumanInput* hi, Texture2D* cross_sprites) {
    if (!hi->click_cross_active) return;
    if (hi->click_cross_timer >= CLICK_CROSS_ANIM_TICKS) {
        hi->click_cross_active = 0;
        return;
    }

    int frame = hi->click_cross_timer * CLICK_CROSS_NUM_FRAMES / CLICK_CROSS_ANIM_TICKS;
    if (frame >= CLICK_CROSS_NUM_FRAMES) frame = CLICK_CROSS_NUM_FRAMES - 1;
    int sprite_idx = hi->click_is_attack ? frame + CLICK_CROSS_NUM_FRAMES : frame;

    int cx = hi->click_screen_x;
    int cy = hi->click_screen_y;
    Texture2D tex = cross_sprites[sprite_idx];
    DrawTexture(tex, cx - tex.width / 2, cy - tex.height / 2, WHITE);
}

static void human_tick_visuals(HumanInput* hi) {
    if (hi->click_cross_active) {
        hi->click_cross_timer++;
        if (hi->click_cross_timer >= CLICK_CROSS_ANIM_TICKS) {
            hi->click_cross_active = 0;
        }
    }
}

#endif
