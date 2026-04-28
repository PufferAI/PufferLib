/**
 * @file osrs_pvp_human_input.h
 * @brief Interactive human control for the visual debug viewer.
 *
 * Collects mouse/keyboard input as semantic intents between render frames,
 * then translates them to encounter-specific action arrays at tick boundary.
 * Toggle human control with H key. Works across PvP and encounter modes.
 *
 * Architecture: clicks at 60Hz → HumanInput staging buffer → per-encounter
 * translator at tick rate → int[] action array fed to step().
 */

#ifndef OSRS_HUMAN_INPUT_H
#define OSRS_HUMAN_INPUT_H

#include "osrs_types.h"
#include "osrs_human_input_types.h"
#include "osrs_encounter.h"

/* forward declare — full struct lives in osrs_pvp_render.h */
struct RenderClient;


/** Convert screen X to world tile X (inverse of render_world_to_screen_x_rc).
    tile_size = RENDER_TILE_SIZE (passed to avoid header ordering issues). */
static inline int human_screen_to_world_x(int screen_x, int arena_base_x,
                                           int tile_size) {
    return screen_x / tile_size + arena_base_x;
}

/** Convert screen Y to world tile Y (inverse of render_world_to_screen_y_rc).
    OSRS Y increases north, screen Y increases down. */
static inline int human_screen_to_world_y(int screen_y, int arena_base_y,
                                           int arena_height, int header_h,
                                           int tile_size) {
    int flipped = (screen_y - header_h) / tile_size;
    return arena_base_y + (arena_height - 1) - flipped;
}


/** Check if world tile (wx,wy) is within an NPC's bounding box.
    OSRS NPCs occupy npc_size x npc_size tiles anchored at (x,y) as southwest corner.
    Players have npc_size 0 or 1, occupying just their tile. */
static int human_tile_hits_entity(RenderEntity* ent, int wx, int wy) {
    int size = ent->npc_size > 1 ? ent->npc_size : 1;
    return wx >= ent->x && wx < ent->x + size &&
           wy >= ent->y && wy < ent->y + size;
}

typedef int (*human_can_attack_entity_fn)(
    void* ctx, const RenderEntity* entity, int entity_idx, int gui_entity_idx);

/** Set click cross at screen position (2D overlay, like real OSRS client). */
static void human_set_click_cross(HumanInput* hi, int screen_x, int screen_y, int is_attack) {
    hi->click_screen_x = screen_x;
    hi->click_screen_y = screen_y;
    hi->click_cross_timer = 0;
    hi->click_cross_active = 1;
    hi->click_is_attack = is_attack;
}

/** Process a world-tile click: attack entity if hit, otherwise move.
    screen_x/y is the raw mouse position for the click cross overlay. */
static void human_process_tile_click(HumanInput* hi,
                                      int wx, int wy,
                                      int screen_x, int screen_y,
                                      RenderEntity* entities, int entity_count,
                                      int gui_entity_idx,
                                      human_can_attack_entity_fn can_attack_entity,
                                      void* can_attack_ctx) {
    /* check if an attackable entity occupies this tile (bounding box) */
    for (int i = 0; i < entity_count; i++) {
        if (i == gui_entity_idx && entities[i].entity_type == ENTITY_PLAYER) continue;
        if (!entities[i].npc_visible && entities[i].entity_type == ENTITY_NPC) continue;
        if (human_tile_hits_entity(&entities[i], wx, wy)) {
            if (can_attack_entity &&
                !can_attack_entity(can_attack_ctx, &entities[i], i, gui_entity_idx)) {
                continue;
            }
            hi->pending_attack = 1;
            hi->pending_target_idx = entities[i].npc_slot;
            /* attack cancels movement — server stops walking to old dest
               and auto-walks toward target instead (OSRS server behavior) */
            hi->pending_move_x = -1;
            hi->pending_move_y = -1;
            if (hi->cursor_mode == CURSOR_SPELL_TARGET) {
                hi->pending_spell = hi->selected_spell;
                human_input_queue_spell_target(hi, hi->selected_spell, hi->pending_target_idx);
                hi->cursor_mode = CURSOR_NORMAL;
            } else {
                human_input_queue_attack_npc(hi, hi->pending_target_idx);
            }
            human_set_click_cross(hi, screen_x, screen_y, 1);
            return;
        }
    }

    /* no entity — movement click */
    if (hi->cursor_mode == CURSOR_SPELL_TARGET) {
        hi->cursor_mode = CURSOR_NORMAL;
        return;
    }

    hi->pending_move_x = wx;
    hi->pending_move_y = wy;
    human_input_queue_walk(hi, wx, wy);
    human_set_click_cross(hi, screen_x, screen_y, 0);
}

/** Handle ground/entity click in 2D grid mode.
    tile_size and header_h are RENDER_TILE_SIZE/RENDER_HEADER_HEIGHT. */
static void human_handle_ground_click(HumanInput* hi,
                                       int mouse_x, int mouse_y,
                                       int arena_base_x, int arena_base_y,
                                       int arena_width, int arena_height,
                                       RenderEntity* entities, int entity_count,
                                       int gui_entity_idx,
                                       human_can_attack_entity_fn can_attack_entity,
                                       void* can_attack_ctx,
                                       int tile_size, int header_h) {
    if (mouse_y < header_h) return;
    int grid_pixel_w = arena_width * tile_size;
    int grid_pixel_h = arena_height * tile_size;
    if (mouse_x < 0 || mouse_x >= grid_pixel_w) return;
    if (mouse_y >= header_h + grid_pixel_h) return;

    int wx = human_screen_to_world_x(mouse_x, arena_base_x, tile_size);
    int wy = human_screen_to_world_y(mouse_y, arena_base_y, arena_height,
                                      header_h, tile_size);
    human_process_tile_click(hi, wx, wy, mouse_x, mouse_y,
                              entities, entity_count, gui_entity_idx,
                              can_attack_entity, can_attack_ctx);
}

/** Handle prayer icon click. Hit-tests the 5-col prayer grid.
    Reuses the same layout math as gui_draw_prayer(). */
static void human_handle_prayer_click(HumanInput* hi, GuiState* gs, Player* p,
                                       int mouse_x, int mouse_y) {
    int oy = gui_content_y(gs) + 4;
    /* skip prayer points bar */
    int bar_h = 18;
    oy += bar_h + 6;

    int cols = 5;
    int gap = 2;
    int icon_sz = (gs->panel_w - 16 - gap * (cols - 1)) / cols;
    int grid_w = cols * icon_sz + (cols - 1) * gap;
    int gx = gs->panel_x + (gs->panel_w - grid_w) / 2;

    /* hit-test: which cell was clicked? */
    if (mouse_x < gx || mouse_y < oy) return;
    int col = (mouse_x - gx) / (icon_sz + gap);
    int row = (mouse_y - oy) / (icon_sz + gap);
    if (col < 0 || col >= cols) return;

    int idx = row * cols + col;
    if (idx < 0 || idx >= GUI_PRAYER_GRID_COUNT) return;

    /* check click is within the cell bounds (not in the gap) */
    int cell_x = gx + col * (icon_sz + gap);
    int cell_y = oy + row * (icon_sz + gap);
    if (mouse_x > cell_x + icon_sz || mouse_y > cell_y + icon_sz) return;

    GuiPrayerIdx pidx = (GuiPrayerIdx)idx;
    (void)p;  /* current-state check no longer needed — toggle is handled by env */

    /* emit toggle actions directly (new ENCOUNTER_OVERHEAD_* / ENCOUNTER_OFFENSIVE_* encoding).
       the env handles the target-already-active → off transition. */
    switch (pidx) {
        case GUI_PRAY_PROTECT_MAGIC:
            hi->pending_prayer = ENCOUNTER_OVERHEAD_TOGGLE_MAGIC;
            human_input_queue_overhead_prayer(hi, hi->pending_prayer);
            break;
        case GUI_PRAY_PROTECT_MISSILES:
            hi->pending_prayer = ENCOUNTER_OVERHEAD_TOGGLE_RANGED;
            human_input_queue_overhead_prayer(hi, hi->pending_prayer);
            break;
        case GUI_PRAY_PROTECT_MELEE:
            hi->pending_prayer = ENCOUNTER_OVERHEAD_TOGGLE_MELEE;
            human_input_queue_overhead_prayer(hi, hi->pending_prayer);
            break;
        case GUI_PRAY_SMITE:
            hi->pending_prayer = ENCOUNTER_OVERHEAD_TOGGLE_SMITE;
            human_input_queue_overhead_prayer(hi, hi->pending_prayer);
            break;
        case GUI_PRAY_REDEMPTION:
            hi->pending_prayer = ENCOUNTER_OVERHEAD_TOGGLE_REDEMPTION;
            human_input_queue_overhead_prayer(hi, hi->pending_prayer);
            break;
        case GUI_PRAY_PIETY:
            hi->pending_offensive_prayer = ENCOUNTER_OFFENSIVE_TOGGLE_PIETY;
            human_input_queue_offensive_prayer(hi, hi->pending_offensive_prayer);
            break;
        case GUI_PRAY_RIGOUR:
            hi->pending_offensive_prayer = ENCOUNTER_OFFENSIVE_TOGGLE_RIGOUR;
            human_input_queue_offensive_prayer(hi, hi->pending_offensive_prayer);
            break;
        case GUI_PRAY_AUGURY:
            hi->pending_offensive_prayer = ENCOUNTER_OFFENSIVE_TOGGLE_AUGURY;
            human_input_queue_offensive_prayer(hi, hi->pending_offensive_prayer);
            break;
        default:
            break;  /* non-actionable prayer */
    }
}

/** Handle spell icon click. Hit-tests the 4-col spell grid.
    Ice/blood spells enter CURSOR_SPELL_TARGET mode; vengeance is instant. */
static void human_handle_spell_click(HumanInput* hi, GuiState* gs,
                                      int mouse_x, int mouse_y) {
    int oy = gui_content_y(gs) + 8;
    int cols = 4;
    int gap = 2;
    int icon_sz = (gs->panel_w - 16 - gap * (cols - 1)) / cols;
    int grid_w = cols * icon_sz + (cols - 1) * gap;
    int gx = gs->panel_x + (gs->panel_w - grid_w) / 2;

    if (mouse_x < gx || mouse_y < oy) return;
    int col = (mouse_x - gx) / (icon_sz + gap);
    int row = (mouse_y - oy) / (icon_sz + gap);
    if (col < 0 || col >= cols) return;

    int idx = row * cols + col;
    if (idx < 0 || idx >= GUI_SPELL_GRID_COUNT) return;

    int cell_x = gx + col * (icon_sz + gap);
    int cell_y = oy + row * (icon_sz + gap);
    if (mouse_x > cell_x + icon_sz || mouse_y > cell_y + icon_sz) return;

    GuiSpellIdx sidx = GUI_SPELL_GRID[idx].idx;

    /* only castable spells respond to clicks (Smoke/Shadow are greyed out) */
    if (!gui_spell_castable(sidx)) return;

    if (sidx == GUI_SPELL_VENGEANCE) {
        /* vengeance is instant — no targeting needed */
        hi->pending_veng = 1;
    } else if (sidx == GUI_SPELL_ICE_RUSH || sidx == GUI_SPELL_ICE_BURST ||
               sidx == GUI_SPELL_ICE_BLITZ || sidx == GUI_SPELL_ICE_BARRAGE) {
        hi->cursor_mode = CURSOR_SPELL_TARGET;
        hi->selected_spell = ATTACK_ICE;
        hi->selected_spell_gui_idx = (int)sidx;
    } else if (sidx == GUI_SPELL_BLOOD_RUSH || sidx == GUI_SPELL_BLOOD_BURST ||
               sidx == GUI_SPELL_BLOOD_BLITZ || sidx == GUI_SPELL_BLOOD_BARRAGE) {
        hi->cursor_mode = CURSOR_SPELL_TARGET;
        hi->selected_spell = ATTACK_BLOOD;
        hi->selected_spell_gui_idx = (int)sidx;
    }
}

/** Handle combat panel click (fight style buttons + spec bar).
    Fight style is set directly on Player (not in the action space).
    Spec bar click sets pending_spec. */
static void human_handle_combat_click(HumanInput* hi, GuiState* gs, Player* p,
                                       int mouse_x, int mouse_y) {
    int ox = gs->panel_x + 8;
    int oy = gui_content_y(gs) + 8;

    /* skip weapon name/sprite area */
    Texture2D wpn_tex = gui_get_item_sprite(gs, p->equipped[GEAR_SLOT_WEAPON]);
    oy += (wpn_tex.id != 0) ? 60 : 22;

    /* 2x2 fight style buttons */
    int btn_gap = 6;
    int btn_w = (gs->panel_w - 16 - btn_gap) / 2;
    int btn_h = 60;

    for (int i = 0; i < 4; i++) {
        int col = i % 2;
        int row = i / 2;
        int bx = ox + col * (btn_w + btn_gap);
        int by = oy + row * (btn_h + btn_gap);
        if (mouse_x >= bx && mouse_x < bx + btn_w &&
            mouse_y >= by && mouse_y < by + btn_h) {
            if (hi->enabled)
                human_input_queue_fight_style(hi, i);
            else
                p->fight_style = (FightStyle)i;
            return;
        }
    }
    oy += 2 * (btn_h + btn_gap) + 10;

    /* skip "Special Attack" label */
    oy += 16;

    /* spec bar */
    int spec_w = gs->panel_w - 16;
    int spec_h = 26;
    if (mouse_x >= ox && mouse_x < ox + spec_w &&
        mouse_y >= oy && mouse_y < oy + spec_h) {
        hi->pending_spec = 1;
        human_input_queue_spec_toggle(hi);
    }
}


/** Translate human input to PvP 7-head action array for agent 0.
    Movement is target-relative (ADJACENT/UNDER/DIAGONAL/FARCAST_N). */
static void human_to_pvp_actions(HumanInput* hi, int* actions,
                                  Player* agent, Player* target) {
    /* zero all heads */
    for (int h = 0; h < NUM_ACTION_HEADS; h++) actions[h] = 0;

    /* HEAD_LOADOUT: keep current gear (human equips items via inventory clicks) */
    actions[HEAD_LOADOUT] = LOADOUT_KEEP;

    /* HEAD_COMBAT: attack or movement */
    if (hi->pending_attack) {
        if (hi->pending_spell == ATTACK_ICE) {
            actions[HEAD_COMBAT] = ATTACK_ICE;
        } else if (hi->pending_spell == ATTACK_BLOOD) {
            actions[HEAD_COMBAT] = ATTACK_BLOOD;
        } else {
            actions[HEAD_COMBAT] = ATTACK_ATK;
        }
    } else if (hi->pending_move_x >= 0 && hi->pending_move_y >= 0) {
        /* convert absolute tile to target-relative movement */
        int dx = hi->pending_move_x - target->x;
        int dy = hi->pending_move_y - target->y;
        int dist = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);  /* chebyshev */

        if (dist == 0) {
            actions[HEAD_COMBAT] = MOVE_UNDER;
        } else if (dist == 1) {
            /* check if cardinal (adjacent) or diagonal */
            if (dx == 0 || dy == 0) {
                actions[HEAD_COMBAT] = MOVE_ADJACENT;
            } else {
                actions[HEAD_COMBAT] = MOVE_DIAGONAL;
            }
        } else {
            /* farcast: clamp to 2-7 */
            int fc = dist;
            if (fc < 2) fc = 2;
            if (fc > 7) fc = 7;
            actions[HEAD_COMBAT] = MOVE_FARCAST_2 + (fc - 2);
        }
    }

    /* HEAD_OVERHEAD: prayer */
    if (hi->pending_prayer >= 0) {
        actions[HEAD_OVERHEAD] = hi->pending_prayer;
    }

    /* HEAD_FOOD */
    if (hi->pending_food) {
        actions[HEAD_FOOD] = FOOD_EAT;
    }

    /* HEAD_POTION */
    if (hi->pending_potion > 0) {
        actions[HEAD_POTION] = hi->pending_potion;
    }

    /* HEAD_KARAMBWAN */
    if (hi->pending_karambwan) {
        actions[HEAD_KARAMBWAN] = KARAM_EAT;
    }

    /* HEAD_VENG */
    if (hi->pending_veng) {
        actions[HEAD_VENG] = VENG_CAST;
    }

    /* spec: use LOADOUT_SPEC_MELEE/RANGE/MAGIC based on current weapon style */
    if (hi->pending_spec) {
        AttackStyle style = get_item_attack_style(agent->equipped[GEAR_SLOT_WEAPON]);
        switch (style) {
            case ATTACK_STYLE_MELEE:  actions[HEAD_LOADOUT] = LOADOUT_SPEC_MELEE; break;
            case ATTACK_STYLE_RANGED: actions[HEAD_LOADOUT] = LOADOUT_SPEC_RANGE; break;
            case ATTACK_STYLE_MAGIC:  actions[HEAD_LOADOUT] = LOADOUT_SPEC_MAGIC; break;
            default: break;
        }
    }

    (void)agent;
}

/* shared translate helpers (encounter_translate_movement/prayer/target)
   live in osrs_encounter.h so encounter headers can use them directly. */


/* click cross sprite textures: 4 yellow (move) + 4 red (attack) animation frames.
   loaded from data/sprites/gui/cross_*.png, indexed [0..3] yellow, [4..7] red. */
#define CLICK_CROSS_NUM_FRAMES 4
#define CLICK_CROSS_ANIM_TICKS 20  /* total animation duration in client ticks (50Hz) */

/** Draw click cross at screen-space position using sprite animation.
    cross_sprites must point to 8 loaded Texture2D (4 yellow + 4 red).
    Falls back to line drawing if sprites aren't loaded. */
static void human_draw_click_cross(HumanInput* hi, Texture2D* cross_sprites, int sprites_loaded) {
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

    if (sprites_loaded && cross_sprites[sprite_idx].id > 0) {
        Texture2D tex = cross_sprites[sprite_idx];
        /* center sprite on click position (OSRS draws at mouseX-8, mouseY-8 for 16px) */
        DrawTexture(tex, cx - tex.width / 2, cy - tex.height / 2, WHITE);
    } else {
        /* fallback: simple X lines */
        float progress = 1.0f - (float)hi->click_cross_timer / CLICK_CROSS_ANIM_TICKS;
        int alpha = (int)(progress * 255);
        Color c = hi->click_is_attack
            ? CLITERAL(Color){ 255, 50, 50, (unsigned char)alpha }
            : CLITERAL(Color){ 255, 255, 0, (unsigned char)alpha };
        DrawLine(cx - 6, cy - 6, cx + 6, cy + 6, c);
        DrawLine(cx + 6, cy - 6, cx - 6, cy + 6, c);
    }
}

/** Draw HUD indicators for human control mode.
    Call from the header rendering section. */
static void human_draw_hud(HumanInput* hi) {
    if (!hi->enabled) return;

    /* "HUMAN" indicator in header */
    DrawText("HUMAN", 8, 8, 16, YELLOW);

    /* spell targeting mode indicator */
    if (hi->cursor_mode == CURSOR_SPELL_TARGET) {
        const char* spell = (hi->selected_spell == ATTACK_ICE) ? "[ICE]" :
                            (hi->selected_spell == ATTACK_BLOOD) ? "[BLOOD]" : "[SPELL]";
        DrawText(spell, 80, 8, 14, CLITERAL(Color){100, 200, 255, 255});
    }
}

/** Tick the click cross animation timer. Call at 50Hz (client tick rate). */
static void human_tick_visuals(HumanInput* hi) {
    if (hi->click_cross_active) {
        hi->click_cross_timer++;
        if (hi->click_cross_timer >= CLICK_CROSS_ANIM_TICKS) {
            hi->click_cross_active = 0;
        }
    }
}

#endif /* OSRS_HUMAN_INPUT_H */
