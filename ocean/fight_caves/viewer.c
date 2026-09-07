/*
 * viewer.c — Fight Caves playable debug viewer.
 *
 * Phase 8: Human-playable Fight Caves with all backend systems connected.
 *
 * Controls:
 *   WASD        — move (N/W/S/E)            Space    — pause/resume
 *   1/2/3       — protect melee/range/magic  Right    — single-step tick
 *   F           — eat shark                  Console  — targets/wave/TPS
 *   P           — drink prayer potion        R        — reset episode
 *   Tab         — cycle attack target        A        — toggle auto/manual
 *   O or D*     — toggle debug overlay       G        — grid  C — collision
 *   4/5         — camera presets             L        — toggle camera lock
 *   Scroll      — zoom                       Right-drag — orbit camera
 *
 *   * D only toggles the overlay when not being used for east movement.
 *   Policy replay mode (`--policy-pipe`) also adds 1/2/4/0 playback presets.
 *   In replay mode, use Shift+4 / 5 for camera presets.
 */

#include "raylib.h"
#include "rlgl.h"
#include "simulation.h"
#include "assets.h"
#include "render.h"
#include "ui.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEFAULT_WINDOW_W 1244
#define DEFAULT_WINDOW_H 1064
#define MAX_TPS         60.0f
#define MIN_TPS         0.25f
#define HALF_TPS        0.50f
#define NORMAL_TPS      (5.0f / 3.0f)
#define POLICY_REPLAY_BASE_TPS NORMAL_TPS

#define FC_UI_ITEM_VIAL          229u
#define FC_UI_ITEM_SHARK         385u
#define FC_UI_ITEM_PRAYER_POT_3  139u
#define FC_UI_ITEM_PRAYER_POT_2  141u
#define FC_UI_ITEM_PRAYER_POT_1  143u
#define FC_UI_ITEM_PRAYER_POT_4  2434u

/* Colors */
#define COL_BG          CLITERAL(Color){ 80, 80, 85, 255 }
#define COL_PANEL       CLITERAL(Color){ 62, 53, 41, 255 }
#define COL_PANEL_BORDER CLITERAL(Color){ 42, 36, 28, 255 }
#define COL_TEXT_YELLOW CLITERAL(Color){ 255, 255,  0, 255 }
#define COL_TEXT_WHITE  CLITERAL(Color){ 255, 255, 255, 255 }
#define COL_TEXT_SHADOW CLITERAL(Color){   0,   0,   0, 255 }
#define COL_TEXT_DIM    CLITERAL(Color){ 130, 130, 140, 255 }
#define COL_TEXT_GREEN  CLITERAL(Color){ 100, 255, 100, 255 }
#define COL_HP_GREEN    CLITERAL(Color){  30, 255,  30, 255 }
#define COL_HP_RED      CLITERAL(Color){ 120,   0,   0, 255 }
#define COL_PRAY_BLUE   CLITERAL(Color){  50, 120, 210, 255 }
#define COL_PLAYER      CLITERAL(Color){  80, 140, 255, 255 }
#define COL_GRID        CLITERAL(Color){  30,  30,  30, 80  }
#define COL_BLOCKED     CLITERAL(Color){ 180,  30,  30, 60  }
#define COL_WALKABLE    CLITERAL(Color){  30, 120,  30, 30  }
#define COL_HIT_RED     CLITERAL(Color){ 255,  50,  50, 255 }
#define COL_HIT_BLUE    CLITERAL(Color){  50, 100, 255, 255 }

#define FC_REWARD_CONFIG_PATH_MAX 256
#define FC_WORLD_ORIGIN_X 2368
#define FC_WORLD_ORIGIN_Y 5056
typedef struct {
    AnimModelState* anim_state;
    uint16_t anim_seq;
    int anim_frame;
    float anim_timer;
} ObjectAnimRuntime;

/* NPC colors by type */
static const Color NPC_COLORS[] = {
    {128,128,128,255}, /* 0: none */
    {180,160,60,255},  /* 1: Tz-Kih (yellow) */
    {100,180,60,255},  /* 2: Tz-Kek (green) */
    {80,150,50,255},   /* 3: Tz-Kek small */
    {60,60,200,255},   /* 4: Tok-Xil (blue) */
    {200,100,60,255},  /* 5: Yt-MejKot (orange) */
    {160,40,160,255},  /* 6: Ket-Zek (purple) */
    {200,40,40,255},   /* 7: TzTok-Jad (RED) */
    {60,200,200,255},  /* 8: Yt-HurKot (cyan) */
};

/* Viewer state */
typedef struct {
    FcState state;
    FcRenderEvents render_events;
    FcActorAnimation actor_animation;
    FcCombatPresentation* combat_presentation;
    RuneCUiState ui;
    FcRenderEntity entities[FC_MAX_RENDER_ENTITIES];
    int entity_count;
    int paused, step_once;
    float tps;
    float tick_acc;
    int show_grid, show_collision;
    Camera3D camera;
    float cam_yaw, cam_pitch, cam_dist;
    int camera_locked;
    int actions[FC_NUM_ACTION_HEADS];
    uint32_t seed, last_hash;
    int episode_count;
    int attack_target;   /* NPC slot index for attack (-1 = none) */
    /* Terrain + Objects + NPC models */
    TerrainMesh* terrain;
    FcMinimapScene minimap_scene;
    ObjectMesh* objects;
    ObjectAnimSet* object_anims;
    FcAnimatedAtlas shared_model_atlas;
    NpcModelSet* object_anim_models;
    ObjectAnimRuntime* object_anim_runtimes;
    int object_anim_runtime_count;
    NpcModelSet* npc_models;
    NpcModelSet* player_model;
    /* Animation cache (shared by player + all NPCs) */
    AnimCache* anim_cache;
    /* Buffered key inputs (captured every frame, consumed on tick) */
    int pending_prayer, pending_eat, pending_drink;
    int pending_attack_npc;
    int pending_tile_x, pending_tile_y;
    FcClickFeedback click_feedback;
    int console_tab;              /* controls, player, obs, mask, reward, log */
    int console_wave_dropdown_open;
    int console_scroll[4];        /* player/obs/mask/reward vertical offsets */
    int console_content_height[4];
    /* Prayer overhead icon textures */
    Texture2D pray_melee_tex, pray_missiles_tex, pray_magic_tex;
    Texture2D click_cross_tex[FC_CLICK_CROSS_FRAME_COUNT * 2];
    int active_loadout; /* index into FC_LOADOUTS[] */
    int combat_style;   /* 0=accurate, 1=rapid, 2=long range */
    /* Prayer-tab sprites used by the active RuneC side interface. */
    Texture2D tex_pray_melee_on, tex_pray_melee_off;
    Texture2D tex_pray_range_on, tex_pray_range_off;
    Texture2D tex_pray_magic_on, tex_pray_magic_off;
    /* Debug overlay (Phase 9c) — toggled with O key */
    int dbg_flags;       /* bitmask of DBG_* flags from fc_debug_overlay.h */
    /* Debug toggles */
    int godmode;        /* 1 = player can't die */
    int policy_pipe;    /* 1 = read actions from stdin, write obs to stdout */
    int policy_episode_limit; /* 0 = unlimited auto-reset, >0 = stop after N episodes */
    int policy_episode_count; /* number of completed policy-pipe episodes */
    int start_wave;     /* 0 = wave 1 (default), >0 = skip to this wave on reset */
    int initial_sharks;
    int initial_prayer_doses;
    FcRewardParams reward_params;
    FcRewardRuntime reward_runtime;
    FcRewardBreakdown reward_breakdown;
    int reward_breakdown_tick;
    int reward_config_loaded;
    char reward_config_path[FC_REWARD_CONFIG_PATH_MAX];
    /* Obs ablation flags (matches FightCaves env). Applied AFTER fc_write_obs
     * in write_obs_to_pipe so policy replay sees the same obs distribution it
     * was trained on. See fc_apply_obs_ablation in the core fc_state.c. */
    int obs_ablate_npc_distance;
    int obs_ablate_incoming_aggregates;
    int obs_ablate_npc_valid;
} ViewerState;

/* Forward declarations */
static void draw_tex_fit(Texture2D tex, int dx, int dy, int dw, int dh,
                         Color tint);

static void viewer_trace_log_to_stderr(int log_level, const char* text,
                                       va_list args) {
    (void)log_level;
    vfprintf(stderr, text, args);
    fputc('\n', stderr);
}

static void set_ui_slot(RuneCUiSlot* slot, uint32_t item_id,
                        uint32_t icon_item_id, int quantity,
                        const char* label) {
    if (!slot) return;
    memset(slot, 0, sizeof(*slot));
    if (item_id == 0 || quantity <= 0) return;
    slot->item_id = item_id;
    slot->icon_item_id = icon_item_id ? icon_item_id : item_id;
    slot->quantity = quantity;
    snprintf(slot->label, sizeof(slot->label), "%s", label ? label : "Item");
    slot->enabled = 1;
}

static uint32_t prayer_potion_item_id_for_doses(int doses) {
    switch (doses) {
        case 4: return FC_UI_ITEM_PRAYER_POT_4;
        case 3: return FC_UI_ITEM_PRAYER_POT_3;
        case 2: return FC_UI_ITEM_PRAYER_POT_2;
        case 1: return FC_UI_ITEM_PRAYER_POT_1;
        default: return FC_UI_ITEM_VIAL;
    }
}

static const char* prayer_potion_label_for_doses(int doses) {
    switch (doses) {
        case 4: return "Prayer potion(4)";
        case 3: return "Prayer potion(3)";
        case 2: return "Prayer potion(2)";
        case 1: return "Prayer potion(1)";
        default: return "Vial";
    }
}

static uint32_t fc_ui_active_prayer_bits(int prayer) {
    switch (prayer) {
        case PRAYER_PROTECT_MAGIC: return 1u << 16;
        case PRAYER_PROTECT_RANGE: return 1u << 17;
        case PRAYER_PROTECT_MELEE: return 1u << 18;
        default: return 0;
    }
}

static int fc_ui_prayer_action_for_slot(const FcPlayer* p, int slot) {
    int prayer = PRAYER_NONE;
    int action = FC_PRAYER_OFF;
    if (slot == 16) {
        prayer = PRAYER_PROTECT_MAGIC;
        action = FC_PRAYER_MAGIC;
    } else if (slot == 17) {
        prayer = PRAYER_PROTECT_RANGE;
        action = FC_PRAYER_RANGE;
    } else if (slot == 18) {
        prayer = PRAYER_PROTECT_MELEE;
        action = FC_PRAYER_MELEE;
    } else {
        return 0;
    }
    if (!p || p->current_prayer <= 0) return 0;
    return p->prayer == prayer ? FC_PRAYER_OFF : action;
}

static void queue_viewer_prayer_button(ViewerState* v, int prayer, int action) {
    if (!v) return;
    FcPlayer* p = &v->state.player;
    if (p->current_prayer <= 0)
        return;
    v->pending_prayer = (p->prayer == prayer) ? FC_PRAYER_OFF : action;
}

static int load_ui_item_icon(RuneCUiState* ui, uint32_t item_id) {
    if (!ui || item_id == 0) return 0;
    char path[128];
    snprintf(path, sizeof(path), "data/sprites/items/item_%u.png", item_id);
    if (!fc_asset_exists(path)) return 0;
    Texture2D tex = fc_load_texture_asset(path);
    if (tex.id == 0) return 0;
    SetTextureFilter(tex, TEXTURE_FILTER_POINT);
    runec_ui_set_item_icon(ui, item_id, tex);
    return 1;
}

static int load_fc_ui_item_icons(ViewerState* v) {
    static const uint32_t ids[] = {
        FC_UI_ITEM_VIAL, FC_UI_ITEM_SHARK,
        FC_UI_ITEM_PRAYER_POT_1, FC_UI_ITEM_PRAYER_POT_2,
        FC_UI_ITEM_PRAYER_POT_3, FC_UI_ITEM_PRAYER_POT_4,
    };
    if (!v) return 0;
    int ready = 1;
    for (int i = 0; i < (int)(sizeof(ids) / sizeof(ids[0])); i++)
        ready &= load_ui_item_icon(&v->ui, ids[i]);
    for (int li = 0; li < FC_NUM_LOADOUTS; li++) {
        const FcLoadout* lo = &FC_LOADOUTS[li];
        for (int ei = 0; ei < lo->equipment_count; ei++) {
            uint32_t icon_id = lo->equipment[ei].icon_item_id
                ? lo->equipment[ei].icon_item_id
                : lo->equipment[ei].item_id;
            ready &= load_ui_item_icon(&v->ui, icon_id);
        }
    }
    return ready;
}

static void sync_fc_ui_items(ViewerState* v) {
    if (!v) return;
    FcPlayer* p = &v->state.player;
    for (int i = 0; i < RUNEC_UI_INV_SLOT_COUNT; i++)
        memset(&v->ui.inventory[i], 0, sizeof(v->ui.inventory[i]));

    int doses = p->prayer_doses_remaining;
    if (doses < 0) doses = 0;
    if (doses > FC_MAX_PRAYER_DOSES) doses = FC_MAX_PRAYER_DOSES;
    int full_pots = doses / 4;
    int partial = doses % 4;
    for (int slot = 0; slot < 8; slot++) {
        int slot_doses = 0;
        if (slot < full_pots) slot_doses = 4;
        else if (slot == full_pots && partial > 0) slot_doses = partial;
        uint32_t item_id = prayer_potion_item_id_for_doses(slot_doses);
        set_ui_slot(&v->ui.inventory[slot], item_id, item_id, 1,
                    prayer_potion_label_for_doses(slot_doses));
    }
    for (int slot = 8; slot < RUNEC_UI_INV_SLOT_COUNT; slot++) {
        if (slot - 8 < p->sharks_remaining) {
            set_ui_slot(&v->ui.inventory[slot], FC_UI_ITEM_SHARK,
                        FC_UI_ITEM_SHARK, 1, "Shark");
        }
    }

    for (int i = 0; i < RUNEC_UI_EQUIP_SLOT_COUNT; i++)
        memset(&v->ui.equipment[i], 0, sizeof(v->ui.equipment[i]));

    int loadout = v->active_loadout;
    if (loadout < 0 || loadout >= FC_NUM_LOADOUTS)
        loadout = FC_ACTIVE_LOADOUT;
    const FcLoadout* lo = &FC_LOADOUTS[loadout];
    for (int i = 0; i < lo->equipment_count; i++) {
        const FcLoadoutEquipmentItem* equip = &lo->equipment[i];
        if (equip->slot >= 0 && equip->slot < RUNEC_UI_EQUIP_SLOT_COUNT) {
            uint32_t icon_id = equip->icon_item_id ? equip->icon_item_id : equip->item_id;
            int quantity = equip->slot == FC_EQUIP_SLOT_AMMO ? p->ammo_count : 1;
            set_ui_slot(&v->ui.equipment[equip->slot], equip->item_id,
                        icon_id, quantity, equip->label);
        }
    }
}

static void sync_fc_ui_status(ViewerState* v) {
    if (!v) return;
    FcPlayer* p = &v->state.player;
    v->ui.hitpoints = p->current_hp > 0 ? (p->current_hp + 9) / 10 : 0;
    v->ui.hitpoints_max = p->max_hp > 0 ? (p->max_hp + 9) / 10 : 0;
    v->ui.prayer_points = p->current_prayer > 0 ? (p->current_prayer + 9) / 10 : 0;
    v->ui.prayer_points_max = p->max_prayer > 0 ? (p->max_prayer + 9) / 10 : 0;
    v->ui.active_prayers = fc_ui_active_prayer_bits(
        fc_actor_animation_render_prayer(&v->actor_animation, &v->state));
    v->ui.run_energy = p->run_energy / 100;
    if (v->ui.run_energy < 0) v->ui.run_energy = 0;
    if (v->ui.run_energy > 100) v->ui.run_energy = 100;
    v->ui.run_enabled = p->is_running != 0;
    v->ui.selected_combat_style = v->combat_style == 2 ? 3 : v->combat_style;
    v->ui.auto_retaliate = 1;
    v->ui.special_attack_energy = 100;
    v->ui.combat_level = 126;
    int loadout = v->active_loadout;
    if (loadout < 0 || loadout >= FC_NUM_LOADOUTS)
        loadout = FC_ACTIVE_LOADOUT;
    const FcLoadout* lo = &FC_LOADOUTS[loadout];
    runec_ui_set_combat_weapon_name(&v->ui, lo->weapon_name);
    runec_ui_set_combat_style_profile(&v->ui, lo->combat_style_profile);

    for (int i = 0; i < RUNEC_UI_SKILL_COUNT; i++) {
        v->ui.skill_current[i] = 1;
        v->ui.skill_base[i] = 1;
    }
    v->ui.skill_current[0] = v->ui.skill_base[0] = p->attack_level;
    v->ui.skill_current[1] = v->ui.skill_base[1] = p->strength_level;
    v->ui.skill_current[2] = v->ui.skill_base[2] = p->defence_level;
    v->ui.skill_current[3] = v->ui.skill_base[3] = p->ranged_level;
    v->ui.skill_current[4] = v->ui.skill_base[4] = p->prayer_level;
    v->ui.skill_current[5] = v->ui.skill_base[5] = p->magic_level;
    v->ui.skill_current[8] = v->ui.skill_base[8] = p->max_hp / 10;
    int total = 0;
    for (int i = 0; i < RUNEC_UI_SKILL_COUNT; i++)
        total += v->ui.skill_base[i];
    v->ui.skill_total = total;
}

static void sync_fc_ui_minimap(ViewerState* v) {
    if (!v) return;
    FcPlayer* p = &v->state.player;
    FcVisualPose player_pose =
        fc_visual_scene_player_pose(&v->actor_animation.scene);
    float player_x = v->actor_animation.scene.player.active
        ? player_pose.x : (float)p->x + 0.5f;
    float player_y = v->actor_animation.scene.player.active
        ? player_pose.y : (float)p->y + 0.5f;
    Color pixels[FC_MINIMAP_DISPLAY_SIZE * FC_MINIMAP_DISPLAY_SIZE];
    fc_minimap_render(&v->minimap_scene, player_x, player_y, v->cam_yaw,
                      pixels);
    runec_ui_update_minimap(&v->ui, pixels, FC_MINIMAP_DISPLAY_SIZE,
                            FC_MINIMAP_DISPLAY_SIZE);
    runec_ui_set_minimap_rotation(&v->ui, v->cam_yaw);

    runec_ui_clear_minimap(&v->ui);
    runec_ui_add_minimap_dot(&v->ui, 0.0f, 0.0f, RUNEC_UI_MINIMAP_DOT_PLAYER);
    if (v->click_feedback.destination_active) {
        int tx = v->click_feedback.destination_x;
        int ty = v->click_feedback.destination_y;
        Vector2 offset = fc_minimap_rotate_offset(
            (float)tx + 0.5f - player_x,
            (float)ty + 0.5f - player_y, v->cam_yaw);
        runec_ui_add_minimap_dot(&v->ui, offset.x, offset.y,
                                 RUNEC_UI_MINIMAP_DOT_DESTINATION);
    } else if (p->route_idx < p->route_len && p->route_len > 0) {
        int tx = p->route_x[p->route_len - 1];
        int ty = p->route_y[p->route_len - 1];
        Vector2 offset = fc_minimap_rotate_offset(
            (float)tx + 0.5f - player_x,
            (float)ty + 0.5f - player_y, v->cam_yaw);
        runec_ui_add_minimap_dot(&v->ui, offset.x, offset.y,
                                 RUNEC_UI_MINIMAP_DOT_DESTINATION);
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* n = &v->state.npcs[i];
        if (!n->active || n->is_dead) continue;
        FcVisualPose npc_pose =
            fc_visual_scene_npc_pose(&v->actor_animation.scene, i);
        float npc_x = v->actor_animation.scene.npcs[i].active
            ? npc_pose.x : (float)n->x + (float)n->size * 0.5f;
        float npc_y = v->actor_animation.scene.npcs[i].active
            ? npc_pose.y : (float)n->y + (float)n->size * 0.5f;
        Vector2 offset = fc_minimap_rotate_offset(
            npc_x - player_x, npc_y - player_y, v->cam_yaw);
        runec_ui_add_minimap_dot(&v->ui, offset.x, offset.y,
                                 RUNEC_UI_MINIMAP_DOT_NPC);
    }
}

static void sync_fc_ui(ViewerState* v) {
    sync_fc_ui_items(v);
    sync_fc_ui_status(v);
    sync_fc_ui_minimap(v);
}

static void queue_player_tile_request(ViewerState* v, int tx, int ty,
                                      float screen_x, float screen_y) {
    if (!v || tx < 0 || tx >= FC_ARENA_WIDTH || ty < 0 || ty >= FC_ARENA_HEIGHT)
        return;
    v->pending_tile_x = tx;
    v->pending_tile_y = ty;
    v->pending_attack_npc = -1;
    fc_click_feedback_select_move(&v->click_feedback, &v->state, tx, ty,
                                  screen_x, screen_y);
}

static void queue_player_attack_request(ViewerState* v, int npc_idx,
                                        float screen_x, float screen_y) {
    if (!v || npc_idx < 0 || npc_idx >= FC_MAX_NPCS) return;
    v->pending_attack_npc = npc_idx;
    v->pending_tile_x = -1;
    v->pending_tile_y = -1;
    fc_click_feedback_select_interaction(&v->click_feedback,
                                         screen_x, screen_y);
}

static void handle_runec_ui_intent(ViewerState* v) {
    if (!v) return;
    RuneCUiIntent* intent = &v->ui.last_intent;
    FcPlayer* p = &v->state.player;
    switch (intent->kind) {
        case RUNEC_UI_INTENT_INVENTORY_SLOT:
            if (intent->primary >= 0 && intent->primary < 8) {
                int full_pots = p->prayer_doses_remaining / 4;
                int partial = p->prayer_doses_remaining % 4;
                if (intent->primary < full_pots ||
                        (intent->primary == full_pots && partial > 0))
                    v->pending_drink = FC_DRINK_PRAYER_POT;
            } else if (intent->primary >= 8 && intent->primary < 28) {
                if (intent->primary - 8 < p->sharks_remaining)
                    v->pending_eat = FC_EAT_SHARK;
            }
            break;
        case RUNEC_UI_INTENT_INVENTORY_ACTION:
            if (strcmp(intent->text, "Use") == 0 || strcmp(intent->text, "Drink") == 0)
                v->pending_drink = FC_DRINK_PRAYER_POT;
            else if (strcmp(intent->text, "Eat") == 0)
                v->pending_eat = FC_EAT_SHARK;
            break;
        case RUNEC_UI_INTENT_PRAYER_SLOT: {
            int action = fc_ui_prayer_action_for_slot(p, intent->primary);
            if (action) v->pending_prayer = action;
            break;
        }
        case RUNEC_UI_INTENT_COMBAT_STYLE:
            v->combat_style = intent->primary == 3 ? 2 : intent->primary;
            if (v->combat_style < 0) v->combat_style = 0;
            if (v->combat_style > 2) v->combat_style = 2;
            break;
        case RUNEC_UI_INTENT_RUN_TOGGLE:
            if (!v->policy_pipe) {
                fc_request_set_running(&v->state, !p->is_running);
                v->ui.run_enabled = p->is_running != 0;
            }
            break;
        case RUNEC_UI_INTENT_MINIMAP_CLICK: {
            FcVisualPose player_pose =
                fc_visual_scene_player_pose(&v->actor_animation.scene);
            float player_x = v->actor_animation.scene.player.active
                ? player_pose.x : (float)p->x + 0.5f;
            float player_y = v->actor_animation.scene.player.active
                ? player_pose.y : (float)p->y + 0.5f;
            int tile_x = -1;
            int tile_y = -1;
            Vector2 mouse = GetMousePosition();
            if (fc_minimap_click_to_tile(
                    (float)intent->primary, (float)intent->secondary,
                    player_x, player_y, v->cam_yaw, &tile_x, &tile_y)) {
                queue_player_tile_request(v, tile_x, tile_y,
                                          mouse.x, mouse.y);
            }
            break;
        }
        default:
            break;
    }
}

static void text_s(const char* t, int x, int y, int sz, Color c) {
    fc_osrs_draw_text(t, x+1, y+1, sz, COL_TEXT_SHADOW);
    fc_osrs_draw_text(t, x, y, sz, c);
}

static const char* fc_terminal_name(int terminal) {
    switch (terminal) {
        case TERMINAL_PLAYER_DEATH: return "player_death";
        case TERMINAL_CAVE_COMPLETE: return "cave_complete";
        case TERMINAL_TICK_CAP: return "tick_cap";
        default: return "none";
    }
}

static int float_near(float a, float b) {
    return fabsf(a - b) < 0.0001f;
}

static char* trim_ascii(char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\0') return s;

    char* end = s + strlen(s) - 1;
    while (end >= s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
    return s;
}

static void reward_params_apply_key(FcRewardParams* params,
                                    const char* key,
                                    const char* value) {
    if (strcmp(key, "w_damage_dealt") == 0) params->w_damage_dealt = strtof(value, NULL);
    else if (strcmp(key, "w_progress") == 0) params->w_progress = strtof(value, NULL);
    else if (strcmp(key, "negative_progress_multiplier") == 0) params->negative_progress_multiplier = strtof(value, NULL);
    else if (strcmp(key, "w_damage_taken") == 0) params->w_damage_taken = strtof(value, NULL);
    else if (strcmp(key, "w_npc_kill") == 0) params->w_npc_kill = strtof(value, NULL);
    else if (strcmp(key, "w_wave_clear") == 0) params->w_wave_clear = strtof(value, NULL);
    else if (strcmp(key, "w_jad_kill") == 0) params->w_jad_kill = strtof(value, NULL);
    else if (strcmp(key, "w_cave_complete") == 0) params->w_cave_complete = strtof(value, NULL);
    else if (strcmp(key, "w_player_death") == 0) params->w_player_death = strtof(value, NULL);
    else if (strcmp(key, "scale_player_death_with_progress") == 0) params->scale_player_death_with_progress = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "player_death_min_scale") == 0) params->player_death_min_scale = strtof(value, NULL);
    else if (strcmp(key, "w_correct_jad_prayer") == 0) params->w_correct_jad_prayer = strtof(value, NULL);
    else if (strcmp(key, "w_correct_danger_prayer") == 0) params->w_correct_danger_prayer = strtof(value, NULL);
    else if (strcmp(key, "w_prayer_lost") == 0) params->w_prayer_lost = strtof(value, NULL);
    else if (strcmp(key, "w_invalid_action") == 0) params->w_invalid_action = strtof(value, NULL);
    else if (strcmp(key, "w_tick_penalty") == 0) params->w_tick_penalty = strtof(value, NULL);
    else if (strcmp(key, "shape_unnecessary_prayer_penalty") == 0) params->shape_unnecessary_prayer_penalty = strtof(value, NULL);
    else if (strcmp(key, "shape_wave_stall_base_penalty") == 0) params->shape_wave_stall_base_penalty = strtof(value, NULL);
    else if (strcmp(key, "shape_wave_stall_cap") == 0) params->shape_wave_stall_cap = strtof(value, NULL);
    else if (strcmp(key, "shape_wave_stall_start") == 0) params->shape_wave_stall_start = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "shape_wave_stall_ramp_interval") == 0) params->shape_wave_stall_ramp_interval = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "shape_jad_heal_penalty") == 0) params->shape_jad_heal_penalty = strtof(value, NULL);
    else if (strcmp(key, "shape_npc_heal_penalty") == 0) params->shape_npc_heal_penalty = strtof(value, NULL);
    else if (strcmp(key, "shape_no_progress_penalty_1") == 0) params->shape_no_progress_penalty_1 = strtof(value, NULL);
    else if (strcmp(key, "shape_no_progress_penalty_2") == 0) params->shape_no_progress_penalty_2 = strtof(value, NULL);
    else if (strcmp(key, "shape_no_progress_penalty_3") == 0) params->shape_no_progress_penalty_3 = strtof(value, NULL);
    else if (strcmp(key, "shape_no_attack_base_penalty") == 0) params->shape_no_attack_base_penalty = strtof(value, NULL);
    else if (strcmp(key, "shape_no_attack_wave_scale") == 0) params->shape_no_attack_wave_scale = strtof(value, NULL);
    else if (strcmp(key, "shape_no_progress_start_1") == 0) params->shape_no_progress_start_1 = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "shape_no_progress_start_2") == 0) params->shape_no_progress_start_2 = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "shape_no_progress_start_3") == 0) params->shape_no_progress_start_3 = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "shape_no_attack_start") == 0) params->shape_no_attack_start = (int)strtol(value, NULL, 10);
}

static void obs_ablation_apply_key(ViewerState* v,
                                   const char* key,
                                   const char* value) {
    if (strcmp(key, "obs_ablate_npc_distance") == 0)
        v->obs_ablate_npc_distance = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "obs_ablate_incoming_aggregates") == 0)
        v->obs_ablate_incoming_aggregates = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "obs_ablate_npc_valid") == 0)
        v->obs_ablate_npc_valid = (int)strtol(value, NULL, 10);
}

static void initial_supplies_apply_key(ViewerState* v,
                                       const char* key,
                                       const char* value) {
    if (strcmp(key, "initial_sharks") == 0)
        v->initial_sharks = (int)strtol(value, NULL, 10);
    else if (strcmp(key, "initial_prayer_doses") == 0)
        v->initial_prayer_doses = (int)strtol(value, NULL, 10);
}

static void apply_initial_supplies(ViewerState* v) {
    if (v->initial_sharks < 0) v->initial_sharks = 0;
    if (v->initial_sharks > FC_MAX_SHARKS) v->initial_sharks = FC_MAX_SHARKS;
    if (v->initial_prayer_doses < 0) v->initial_prayer_doses = 0;
    if (v->initial_prayer_doses > FC_MAX_PRAYER_DOSES)
        v->initial_prayer_doses = FC_MAX_PRAYER_DOSES;
    v->state.player.sharks_remaining = v->initial_sharks;
    v->state.player.prayer_doses_remaining = v->initial_prayer_doses;
}

static void load_reward_params(ViewerState* v) {
    v->reward_params = fc_reward_default_params();
    v->initial_sharks = 0;
    v->initial_prayer_doses = 0;
    v->obs_ablate_npc_distance = 0;
    v->obs_ablate_incoming_aggregates = 0;
    v->obs_ablate_npc_valid = 0;
    v->reward_config_loaded = 0;
    snprintf(v->reward_config_path, sizeof(v->reward_config_path), "%s", "defaults");

    {
        char config_path[FC_ASSET_PATH_MAX];
        FILE* f;
        if (!fc_repo_resolve_path("config/fight_caves.ini",
                                  config_path, sizeof(config_path))) {
            return;
        }
        f = fopen(config_path, "r");
        if (!f) return;

        char line[512];
        int in_env = 0;
        while (fgets(line, sizeof(line), f)) {
            char* comment = strchr(line, '#');
            if (comment) *comment = '\0';

            char* text = trim_ascii(line);
            if (*text == '\0') continue;

            if (*text == '[') {
                char* close = strchr(text, ']');
                if (!close) continue;
                *close = '\0';
                in_env = (strcmp(text + 1, "env") == 0);
                continue;
            }

            if (!in_env) continue;

            char* eq = strchr(text, '=');
            if (!eq) continue;
            *eq = '\0';

            char* key = trim_ascii(text);
            char* value = trim_ascii(eq + 1);
            if (*key == '\0' || *value == '\0') continue;

            reward_params_apply_key(&v->reward_params, key, value);
            obs_ablation_apply_key(v, key, value);
            initial_supplies_apply_key(v, key, value);
        }

        fclose(f);
        v->reward_config_loaded = 1;
        strncpy(v->reward_config_path, config_path, sizeof(v->reward_config_path) - 1);
        v->reward_config_path[sizeof(v->reward_config_path) - 1] = '\0';
        return;
    }
}

static void reset_reward_tracking(ViewerState* v) {
    fc_reward_runtime_reset(&v->reward_runtime);
    memset(&v->reward_breakdown, 0, sizeof(v->reward_breakdown));
    v->reward_breakdown_tick = -1;
}

static void update_reward_breakdown(ViewerState* v) {
    if (v->reward_breakdown_tick == v->state.tick) return;
    v->reward_breakdown = fc_reward_compute_breakdown(
        &v->state, &v->reward_params, &v->reward_runtime);
    fc_reward_sync_progress_state(&v->state, &v->reward_runtime);
    v->reward_breakdown_tick = v->state.tick;
}

static const float MANUAL_TPS_PRESETS[] = {
    0.25f, 0.50f, NORMAL_TPS, 4.0f, 10.0f, 15.0f, 30.0f, 60.0f
};
static const char* MANUAL_TPS_LABELS[] = {
    "0.25", "0.5", "5/3", "4", "10", "15", "30", "60"
};
#define NUM_MANUAL_TPS_PRESETS ((int)(sizeof(MANUAL_TPS_PRESETS) / sizeof(MANUAL_TPS_PRESETS[0])))

static float policy_replay_multiplier_to_tps(int multiplier) {
    switch (multiplier) {
        case 1: return (float)POLICY_REPLAY_BASE_TPS;
        case 2: return (float)(POLICY_REPLAY_BASE_TPS * 2);
        case 4: return (float)(POLICY_REPLAY_BASE_TPS * 4);
        case 10: return (float)(POLICY_REPLAY_BASE_TPS * 10);
        default: return (float)POLICY_REPLAY_BASE_TPS;
    }
}

static int policy_replay_tps_to_multiplier(float tps) {
    if (float_near(tps, (float)POLICY_REPLAY_BASE_TPS)) return 1;
    if (float_near(tps, (float)(POLICY_REPLAY_BASE_TPS * 2))) return 2;
    if (float_near(tps, (float)(POLICY_REPLAY_BASE_TPS * 4))) return 4;
    if (float_near(tps, (float)(POLICY_REPLAY_BASE_TPS * 10))) return 10;
    return 0;
}

static int policy_replay_normalize_multiplier(int multiplier) {
    switch (multiplier) {
        case 1:
        case 2:
        case 4:
        case 10:
            return multiplier;
        default:
            return 1;
    }
}

static void set_viewer_tps(ViewerState* v, float tps) {
    if (tps < MIN_TPS) tps = MIN_TPS;
    if (tps > MAX_TPS) tps = MAX_TPS;
    v->tps = tps;
    if (v->tick_acc >= 1.0f)
        v->tick_acc = fmodf(v->tick_acc, 1.0f);
}

static void set_policy_replay_speed(ViewerState* v, int multiplier) {
    int normalized = policy_replay_normalize_multiplier(multiplier);
    set_viewer_tps(v, policy_replay_multiplier_to_tps(normalized));
    fprintf(stderr, "[policy-pipe] Replay speed set to %dx (%.2f TPS)\n",
            normalized, v->tps);
}

static void cycle_policy_replay_speed(ViewerState* v, int direction) {
    static const int presets[] = {1, 2, 4, 10};
    int current = policy_replay_tps_to_multiplier(v->tps);
    int idx = 0;

    for (int i = 0; i < 4; i++) {
        if (presets[i] == current) {
            idx = i;
            break;
        }
    }

    idx += direction;
    if (idx < 0) idx = 0;
    if (idx > 3) idx = 3;
    set_policy_replay_speed(v, presets[idx]);
}

static void set_manual_speed(ViewerState* v, float tps) {
    float best = MANUAL_TPS_PRESETS[0];
    float best_diff = fabsf(tps - best);
    for (int i = 1; i < NUM_MANUAL_TPS_PRESETS; i++) {
        float diff = fabsf(tps - MANUAL_TPS_PRESETS[i]);
        if (diff < best_diff) {
            best = MANUAL_TPS_PRESETS[i];
            best_diff = diff;
        }
    }
    set_viewer_tps(v, best);
}

static void toggle_debug_overlay(ViewerState* v) {
    if (!v) return;
    v->dbg_flags = v->dbg_flags ? 0 : DBG_ALL;
}

static void toggle_godmode(ViewerState* v) {
    if (!v) return;
    v->godmode = !v->godmode;
    fprintf(stderr, "GODMODE: %s\n", v->godmode ? "ON" : "OFF");
}

static void print_policy_episode_summary(const ViewerState* v) {
    const FcState* s = &v->state;
    FcEpisodeSummary summary;
    fc_episode_summary_build(s, s->tick, &summary);

    fprintf(stderr,
        "[policy-pipe] episode_summary "
        "{\"episode\":%d,\"seed\":%u,\"terminal\":\"%s\","
        "\"env/episode_length\":%d,"
        "\"env/wave_reached\":%d,"
        "\"env/most_npcs_slayed\":%d,"
        "\"env/prayer_uptime_melee\":%.6f,"
        "\"env/prayer_uptime_range\":%.6f,"
        "\"env/prayer_uptime_magic\":%.6f,"
        "\"env/correct_prayer\":%d,"
        "\"env/wrong_prayer_hits\":%d,"
        "\"env/no_prayer_hits\":%d,"
        "\"env/prayer_switches\":%d,"
        "\"env/damage_blocked\":%d,"
        "\"env/dmg_taken_avg\":%d,"
        "\"env/attack_when_ready_rate\":%.6f,"
        "\"env/tokxil_melee_ticks\":%d,"
        "\"env/ketzek_melee_ticks\":%d,"
        "\"env/max_wave_ticks\":%d,"
        "\"env/max_wave_ticks_wave\":%d,"
        "\"env/reached_wave_63\":%d,"
        "\"env/jad_kill_rate\":%d,"
        "\"env/target_held_ticks\":%d,"
        "\"env/no_target_ticks\":%d,"
        "\"env/target_in_range_los_ticks\":%d,"
        "\"env/target_out_of_range_or_los_ticks\":%d,"
        "\"env/attack_cooldown_wait_ticks\":%d,"
        "\"env/ready_but_no_attack_ticks\":%d,"
        "\"env/action_move_idle_ticks\":%d,"
        "\"env/action_move_walk_ticks\":%d,"
        "\"env/action_move_run_ticks\":%d,"
        "\"env/action_attack_none_ticks\":%d,"
        "\"env/action_attack_target_ticks\":%d,"
        "\"env/action_prayer_noop_ticks\":%d,"
        "\"env/action_prayer_cmd_ticks\":%d",
        v->policy_episode_count + 1,
        v->seed,
        fc_terminal_name(s->terminal),
        summary.episode_length,
        summary.wave_reached,
        summary.npcs_slayed,
        summary.prayer_uptime_melee,
        summary.prayer_uptime_range,
        summary.prayer_uptime_magic,
        summary.correct_prayer,
        summary.wrong_prayer_hits,
        summary.no_prayer_hits,
        summary.prayer_switches,
        summary.damage_blocked,
        summary.damage_taken,
        summary.attack_when_ready_rate,
        summary.tokxil_melee_ticks,
        summary.ketzek_melee_ticks,
        summary.max_wave_ticks,
        summary.max_wave_ticks_wave,
        summary.reached_wave_63,
        summary.jad_killed,
        summary.target_held_ticks,
        summary.no_target_ticks,
        summary.target_in_range_los_ticks,
        summary.target_out_of_range_or_los_ticks,
        summary.attack_cooldown_wait_ticks,
        summary.ready_but_no_attack_ticks,
        summary.action_move_idle_ticks,
        summary.action_move_walk_ticks,
        summary.action_move_run_ticks,
        summary.action_attack_none_ticks,
        summary.action_attack_target_ticks,
        summary.action_prayer_noop_ticks,
        summary.action_prayer_cmd_ticks);

    for (int i = 1; i < NPC_TYPE_COUNT; i++) {
        const char* npc = fc_episode_npc_metric_name(i);
        fprintf(stderr,
            ",\"env/dmg_to_%s\":%d"
            ",\"env/resolved_hits_to_%s\":%d"
            ",\"env/damaging_hits_to_%s\":%d"
            ",\"env/attack_cycles_to_%s\":%d"
            ",\"env/target_ticks_%s\":%d",
            npc, summary.damage_to_npc_type[i],
            npc, summary.resolved_hits_to_npc_type[i],
            npc, summary.damaging_hits_to_npc_type[i],
            npc, summary.attack_cycles_to_npc_type[i],
            npc, summary.target_ticks_by_npc_type[i]);
    }

    fprintf(stderr, ",\"env/n\":1.0}\n");
}

static void reset_ep(ViewerState* v) {
    load_reward_params(v);
    reset_reward_tracking(v);
    v->seed = (uint32_t)GetRandomValue(1, 999999);
    fc_reset(&v->state, v->seed);
    /* Skip to start_wave if set */
    if (v->start_wave > 1 && v->start_wave <= FC_NUM_WAVES) {
        for (int i = 0; i < FC_MAX_NPCS; i++) {
            v->state.npcs[i].active = 0;
            v->state.npcs[i].is_dead = 0;
        }
        v->state.npcs_remaining = 0;
        v->state.current_wave = v->start_wave;
        fc_wave_spawn(&v->state, v->start_wave);
        v->state.player.current_hp = v->state.player.max_hp;
        v->state.player.current_prayer = v->state.player.max_prayer;
    }
    apply_initial_supplies(v);
    fc_reward_runtime_begin_episode(&v->reward_runtime, &v->state);
    fc_fill_render_entities(&v->state, v->entities, &v->entity_count);
    fc_fill_render_events(&v->state, &v->render_events);
    v->last_hash = fc_state_hash(&v->state);
    v->episode_count++;
    v->attack_target = -1;
    memset(v->actions, 0, sizeof(v->actions));
    fc_combat_presentation_reset(v->combat_presentation);
    fc_actor_animation_reset(&v->actor_animation, &v->state,
                             v->player_model, v->active_loadout);
    v->pending_prayer = 0;
    v->pending_eat = 0;
    v->pending_drink = 0;
    v->pending_attack_npc = -1;
    v->pending_tile_x = -1;
    v->pending_tile_y = -1;
    fc_click_feedback_reset(&v->click_feedback);
    dbg_log_clear();
}

static void viewer_jump_to_wave(ViewerState* v, int wave) {
    if (!v) return;
    if (wave < 1) wave = 1;
    if (wave > FC_NUM_WAVES) wave = FC_NUM_WAVES;

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        v->state.npcs[i].active = 0;
        v->state.npcs[i].is_dead = 0;
    }
    v->state.npcs_remaining = 0;
    v->state.current_wave = wave;
    fc_wave_spawn(&v->state, wave);
    v->state.player.current_hp = v->state.player.max_hp;
    v->state.player.current_prayer = v->state.player.max_prayer;
    v->state.terminal = TERMINAL_NONE;
    v->state.jad_healers_spawned = 0;
    reset_reward_tracking(v);
    fc_reward_runtime_begin_episode(&v->reward_runtime, &v->state);
    fc_fill_render_entities(&v->state, v->entities, &v->entity_count);
    fc_fill_render_events(&v->state, &v->render_events);
    fc_combat_presentation_reset(v->combat_presentation);
    fc_actor_animation_reset(&v->actor_animation, &v->state,
                             v->player_model, v->active_loadout);
    v->attack_target = -1;
    fc_click_feedback_reset(&v->click_feedback);
    dbg_log_clear();
}
/* Terrain loader — the terrain mesh is the floor heightmap.
 * The red/black lava pattern comes from the objects mesh (fightcaves.objects),
 * not the terrain. The terrain is just the ground surface.
 * We keep original cache colors (dark base) without modification. */
static TerrainMesh* load_terrain(ViewerState* v) {
    (void)v;
    TerrainMesh* tm = terrain_load("fightcaves.terrain");
    if (tm && tm->loaded) {
        terrain_offset(tm, FC_WORLD_ORIGIN_X, FC_WORLD_ORIGIN_Y);
        return tm;
    }
    return NULL;
}

/* Objects loader — no modifications */
static ObjectMesh* load_objects_with_terrain(TerrainMesh* tm) {
    (void)tm;
    ObjectMesh* om = objects_load("fightcaves.objects");
    if (om && om->loaded) {
        objects_offset(om, FC_WORLD_ORIGIN_X, FC_WORLD_ORIGIN_Y);

        /* No modifications to objects mesh — original cache data */

        return om;
    }
    return NULL;
}

/* Forward declaration */
static float ground_y(ViewerState* v, int tile_x, int tile_y);

/* ======================================================================== */
/* Human input → action heads                                                */
/* ======================================================================== */

/* Raycast from mouse position to find the tile coordinate on the ground plane */
static int raycast_to_tile(ViewerState* v, int* out_x, int* out_y) {
    Ray ray = GetScreenToWorldRay(GetMousePosition(), v->camera);
    /* Intersect with Y = ground_y plane */
    float gy = ground_y(v, 32, 32);
    if (fabsf(ray.direction.y) < 0.001f) return 0;  /* ray parallel to ground */
    float t = (gy - ray.position.y) / ray.direction.y;
    if (t < 0) return 0;  /* behind camera */
    float wx = ray.position.x + ray.direction.x * t;
    float wz = ray.position.z + ray.direction.z * t;
    /* Convert to tile coords: X = world X, tile Y = -world Z */
    int tx = (int)floorf(wx);
    int ty = (int)floorf(-wz);
    if (tx < 0 || tx >= FC_ARENA_WIDTH || ty < 0 || ty >= FC_ARENA_HEIGHT) return 0;
    *out_x = tx;
    *out_y = ty;
    return 1;
}

/* Find NPC at clicked tile — checks LIVE state, not render snapshot.
 * Returns NPC array index (0..FC_MAX_NPCS-1) or -1 if no NPC there. */
static int find_clicked_npc_idx(ViewerState* v, int tile_x, int tile_y) {
    int best = -1;
    int best_dist = 999;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* n = &v->state.npcs[i];
        if (!n->active || n->is_dead) continue;
        /* Check if tile is within the NPC's footprint (or 1 tile adjacent) */
        if (tile_x >= n->x - 1 && tile_x <= n->x + n->size &&
            tile_y >= n->y - 1 && tile_y <= n->y + n->size) {
            /* Prefer the closest NPC center */
            int cx = n->x + n->size/2;
            int cy = n->y + n->size/2;
            int d = abs(tile_x - cx) + abs(tile_y - cy);
            if (d < best_dist) { best_dist = d; best = i; }
        }
    }
    return best;
}

/* Called EVERY FRAME to capture clicks (which only fire once at 60fps).
 * Buffers authoritative actions and starts presentation-only feedback. */
static void process_human_clicks(ViewerState* v, int ui_capture) {
    FcPlayer* p = &v->state.player;

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        if (ui_capture) return;
        Vector2 mpos = GetMousePosition();
        int tx = -1;
        int ty = -1;
        int rc = raycast_to_tile(v, &tx, &ty);
        fprintf(stderr, "CLICK mouse=(%.0f,%.0f) raycast=%d tile=(%d,%d) player=(%d,%d)",
                mpos.x, mpos.y, rc, tx, ty, p->x, p->y);
        if (rc) {
            int npc_idx = find_clicked_npc_idx(v, tx, ty);
            if (npc_idx >= 0) {
                queue_player_attack_request(v, npc_idx, mpos.x, mpos.y);
                fprintf(stderr, " → ATTACK npc_idx=%d\n", npc_idx);
            } else {
                int walkable = v->state.walkable[tx][ty];
                fprintf(stderr, " walkable=%d", walkable);
                queue_player_tile_request(v, tx, ty, mpos.x, mpos.y);
                fprintf(stderr, " → MOVE%s\n", walkable ? "" : "-NEAR");
            }
        } else {
            fprintf(stderr, " → MISS (raycast failed)\n");
        }
    }
}

/* Called EVERY FRAME for key presses. Buffers actions for next tick. */
static void process_human_keys(ViewerState* v) {
    FcPlayer* p = &v->state.player;
    if (IsKeyPressed(KEY_ONE))   v->pending_prayer = (p->prayer == PRAYER_PROTECT_MELEE) ? FC_PRAYER_OFF : FC_PRAYER_MELEE;
    if (IsKeyPressed(KEY_TWO))   v->pending_prayer = (p->prayer == PRAYER_PROTECT_RANGE) ? FC_PRAYER_OFF : FC_PRAYER_RANGE;
    if (IsKeyPressed(KEY_THREE)) v->pending_prayer = (p->prayer == PRAYER_PROTECT_MAGIC) ? FC_PRAYER_OFF : FC_PRAYER_MAGIC;
    if (IsKeyPressed(KEY_F))     v->pending_eat = FC_EAT_SHARK;
    if (IsKeyPressed(KEY_P))     v->pending_drink = FC_DRINK_PRAYER_POT;
    if (IsKeyPressed(KEY_X))
        fc_request_set_running(&v->state, !p->is_running);

    /* --- Debug toggles (testing only) --- */
    /* F9: toggle godmode (player can't die) */
    if (IsKeyPressed(KEY_F9)) toggle_godmode(v);
    /* F1-F8: spawn NPC type 1-8 near the player */
    for (int fk = 0; fk < 8; fk++) {
        if (IsKeyPressed(KEY_F1 + fk)) {
            int npc_type = fk + 1;
            /* Find free NPC slot */
            for (int si = 0; si < FC_MAX_NPCS; si++) {
                if (!v->state.npcs[si].active) {
                    int sx = p->x + 5, sy = p->y;
                    const FcNpcStats* stats = fc_npc_get_stats(npc_type);
                    /* Find nearby walkable tile */
                    for (int r = 0; r < 10; r++) {
                        for (int dx = -r; dx <= r; dx++) {
                            int ty = p->y + r, tx = p->x + dx + 3;
                            if (tx >= 0 && tx < FC_ARENA_WIDTH && ty >= 0 && ty < FC_ARENA_HEIGHT &&
                                fc_footprint_walkable(tx, ty, stats->size, v->state.walkable)) {
                                sx = tx; sy = ty; r = 99; break;
                            }
                        }
                    }
                    fc_npc_spawn(&v->state.npcs[si], npc_type, sx, sy,
                                 v->state.next_spawn_index++);
                    /* Don't increment npcs_remaining — debug spawns shouldn't
                     * affect wave progression. Wave clear checks npcs_remaining. */
                    fprintf(stderr, "DEBUG SPAWN: NPC type %d at (%d,%d)\n", npc_type, sx, sy);
                    break;
                }
            }
        }
    }
}

/* Called on TICK frames: build action array from buffered inputs. */
static void build_human_actions(ViewerState* v) {
    memset(v->actions, 0, sizeof(v->actions));
    v->actions[0] = FC_MOVE_IDLE;
    v->actions[1] = FC_ATTACK_NONE;
    if (v->pending_attack_npc >= 0) {
        int visible[FC_VISIBLE_NPCS];
        int count = fc_visible_npc_indices(&v->state, visible);
        for (int slot = 0; slot < count; slot++) {
            if (visible[slot] == v->pending_attack_npc) {
                v->actions[1] = slot + 1;
                break;
            }
        }
    }
    if (v->pending_tile_x >= 0 && v->pending_tile_y >= 0) {
        v->actions[5] = v->pending_tile_x + 1;
        v->actions[6] = v->pending_tile_y + 1;
    }
    /* Buffered prayer/eat/drink */
    v->actions[2] = v->pending_prayer;
    v->actions[3] = v->pending_eat;
    v->actions[4] = v->pending_drink;
    /* Clear buffers */
    v->pending_prayer = 0;
    v->pending_eat = 0;
    v->pending_drink = 0;
    v->pending_attack_npc = -1;
    v->pending_tile_x = -1;
    v->pending_tile_y = -1;
}

/* ======================================================================== */
/* Policy pipe mode — read actions from stdin, write obs to stdout          */
/* ======================================================================== */

static int read_policy_actions(ViewerState* v) {
    for (int i = 0; i < FC_PUFFER_NUM_ATNS; i++) {
        int action;
        if (scanf("%d", &action) != 1)
            return 0;
        v->actions[i] = action;
    }
    for (int i = FC_PUFFER_NUM_ATNS; i < FC_NUM_ACTION_HEADS; i++) v->actions[i] = 0;
    return 1;
}

static void write_obs_to_pipe(ViewerState* v) {
    /* Write the same policy obs + action mask contract used by Puffer training. */
    float obs_buf[FC_OBS_SIZE];
    fc_write_obs(&v->state, obs_buf);
    /* Mirror training-time obs ablation so the policy sees the distribution
     * it was trained on (no-op when all flags are 0). */
    fc_apply_obs_ablation(obs_buf,
                          v->obs_ablate_npc_distance,
                          v->obs_ablate_incoming_aggregates,
                          v->obs_ablate_npc_valid);
    float mask_buf[FC_ACTION_MASK_SIZE];
    fc_write_mask(&v->state, mask_buf);

    /* Policy obs: first FC_POLICY_OBS_SIZE floats */
    for (int i = 0; i < FC_POLICY_OBS_SIZE; i++)
        printf("%.6f ", obs_buf[i]);
    for (int i = 0; i < FC_PUFFER_MASK_SIZE; i++)
        printf("%.6f ", mask_buf[i]);
    printf("\n");
    fflush(stdout);
}

/* Entity ground Y — slightly above terrain so entities stand on the flattened cracks */
static float ground_y(ViewerState* v, int tile_x, int tile_y) {
    if (v->terrain && v->terrain->loaded) {
        return terrain_height_at(v->terrain, tile_x, tile_y) + 0.1f;
    }
    return 0.0f;
}

static float ground_y_smooth(ViewerState* v, float tile_x, float tile_y) {
    int x0 = (int)floorf(tile_x);
    int y0 = (int)floorf(tile_y);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= FC_ARENA_WIDTH) x0 = FC_ARENA_WIDTH - 1;
    if (y0 >= FC_ARENA_HEIGHT) y0 = FC_ARENA_HEIGHT - 1;
    int x1 = x0 + 1 < FC_ARENA_WIDTH ? x0 + 1 : x0;
    int y1 = y0 + 1 < FC_ARENA_HEIGHT ? y0 + 1 : y0;
    float tx = tile_x - floorf(tile_x);
    float ty = tile_y - floorf(tile_y);
    float h00 = ground_y(v, x0, y0);
    float h10 = ground_y(v, x1, y0);
    float h01 = ground_y(v, x0, y1);
    float h11 = ground_y(v, x1, y1);
    float h0 = h00 + (h10 - h00) * tx;
    float h1 = h01 + (h11 - h01) * tx;
    return h0 + (h1 - h0) * ty;
}

typedef struct {
    float x;
    float y;
    float face_angle;
    int moving;
    FcVisualLocomotion locomotion;
} EntityRenderPose;

static EntityRenderPose entity_render_pose(const ViewerState* v,
                                           const FcRenderEntity* e) {
    EntityRenderPose pose = {0};
    if (!v || !e) return pose;
    FcVisualPose visual = e->entity_type == ENTITY_PLAYER
        ? fc_visual_scene_player_pose(&v->actor_animation.scene)
        : fc_visual_scene_npc_pose(&v->actor_animation.scene, e->npc_slot);
    pose.x = visual.x;
    pose.y = visual.y;
    pose.face_angle = visual.yaw_degrees;
    pose.moving = visual.moving;
    pose.locomotion = visual.locomotion;
    return pose;
}

static Vector2 click_route_point_to_screen(ViewerState* v,
                                           float tile_x, float tile_y) {
    Vector3 world = {
        tile_x,
        ground_y_smooth(v, tile_x, tile_y) + 0.08f,
        -tile_y,
    };
    return GetWorldToScreen(world, v->camera);
}

static void draw_click_destination_3d(ViewerState* v) {
    if (!v || !v->click_feedback.destination_active) return;
    int tx = v->click_feedback.destination_x;
    int ty = v->click_feedback.destination_y;
    if (tx < 0 || tx >= FC_ARENA_WIDTH ||
        ty < 0 || ty >= FC_ARENA_HEIGHT) {
        return;
    }

    float y = ground_y(v, tx, ty) + 0.025f;
    Vector3 center = {(float)tx + 0.5f, y, -((float)ty + 0.5f)};
    Color fill = {255, 215, 0, 70};
    Color edge = {255, 235, 70, 230};
    DrawCube(center, 0.92f, 0.025f, 0.92f, fill);
    DrawCubeWires(center, 0.92f, 0.025f, 0.92f, edge);
}

static void draw_click_route_2d(ViewerState* v) {
    if (!v) return;
    const int* route_x = NULL;
    const int* route_y = NULL;
    int start = 0;
    int len = 0;
    if (!fc_click_feedback_route(&v->click_feedback, &v->state,
                                 &route_x, &route_y, &start, &len)) {
        return;
    }

    FcVisualPose player =
        fc_visual_scene_player_pose(&v->actor_animation.scene);
    Vector2 previous = click_route_point_to_screen(v, player.x, player.y);
    Color line = {255, 220, 30, 210};
    Color point = {255, 245, 120, 240};
    for (int i = start; i < len; i++) {
        Vector2 next = click_route_point_to_screen(
            v, (float)route_x[i] + 0.5f, (float)route_y[i] + 0.5f);
        DrawLineEx(previous, next, 2.0f, line);
        DrawCircleV(next, 2.5f, point);
        previous = next;
    }
}

static void draw_click_cross(ViewerState* v) {
    if (!v) return;
    int frame = fc_click_feedback_cross_frame(&v->click_feedback);
    if (frame < 0) return;
    int base = v->click_feedback.cross_kind == FC_CLICK_CROSS_INTERACTION
        ? FC_CLICK_CROSS_FRAME_COUNT : 0;
    Texture2D texture = v->click_cross_tex[base + frame];
    if (texture.id == 0) return;
    DrawTexture(texture,
                (int)roundf(v->click_feedback.cross_screen_x) -
                    texture.width / 2,
                (int)roundf(v->click_feedback.cross_screen_y) -
                    texture.height / 2,
                WHITE);
}

static Vector3 camera_follow_target(const ViewerState* v) {
    float cx = FC_ARENA_WIDTH * 0.5f;
    float cy = -(FC_ARENA_HEIGHT * 0.5f);
    if (v->entity_count > 0) {
        EntityRenderPose pose = entity_render_pose(v, &v->entities[0]);
        cx = pose.x;
        cy = -pose.y;
    }
    return (Vector3){cx, 0.5f, cy};
}

static void draw_npc_prayer_window_indicators(ViewerState* v) {
    if (!v || !v->dbg_flags) return;

    for (int i = 0; i < v->entity_count; i++) {
        const FcRenderEntity* entity = &v->entities[i];
        int npc_idx = entity->npc_slot;
        if (entity->entity_type != ENTITY_NPC || entity->is_dead ||
            npc_idx < 0 || npc_idx >= FC_MAX_NPCS ||
            !fc_actor_animation_prayer_window_active(
                &v->actor_animation, npc_idx, v->state.tick)) {
            continue;
        }

        EntityRenderPose pose = entity_render_pose(v, entity);
        Vector3 anchor = {
            pose.x,
            ground_y_smooth(v, pose.x, pose.y) +
                2.2f + (float)entity->size * 0.8f,
            -pose.y
        };
        dbg_draw_prayer_window_indicator(anchor, v->camera);
    }
}

/* ======================================================================== */
/* Scene drawing                                                             */
/* ======================================================================== */

static int object_anim_row_visible(const ObjectAnimPlacement* row) {
    if (!row) return 0;
    return (row->flags & OANM_FLAG_DYNAMIC_REPLACEMENT) == 0;
}

static void draw_animated_objects(ViewerState* v) {
    if (!v || !v->object_anims || !v->object_anims->loaded ||
        !v->object_anim_models || !v->object_anim_runtimes)
        return;

    float dt = GetFrameTime();
    rlDisableBackfaceCulling();
    for (int i = 0; i < v->object_anims->count; i++) {
        ObjectAnimPlacement* row = &v->object_anims->rows[i];
        if (!object_anim_row_visible(row)) continue;

        NpcModelEntry* entry = fc_npc_model_find(v->object_anim_models,
                                                 row->model_id);
        if (!entry || !entry->loaded) continue;

        if (row->animation_id >= 0 && v->anim_cache) {
            ObjectAnimRuntime* rt = &v->object_anim_runtimes[i];
            fc_model_animation_update(entry, v->anim_cache, &rt->anim_state,
                                      &rt->anim_seq, &rt->anim_frame,
                                      &rt->anim_timer, row->animation_id, dt,
                                      row->phase_ticks);
        }

        DrawModelEx(entry->model,
                    (Vector3){row->pos_x, row->pos_y, row->pos_z},
                    (Vector3){0, 1, 0}, 0.0f,
                    (Vector3){1, 1, 1}, WHITE);
    }
    rlEnableBackfaceCulling();
}

static void draw_actor_footprint(ViewerState* v,
                                 const FcRenderEntity* entity) {
    if (!v || !entity || entity->size <= 0) {
        return;
    }
    if (entity->is_dead &&
        (entity->entity_type != ENTITY_NPC ||
         !fc_combat_presentation_npc_death_deferred(
             v->combat_presentation, &v->state, entity->npc_slot))) {
        return;
    }

    const int is_player = entity->entity_type == ENTITY_PLAYER;
    const Color fill = is_player
        ? CLITERAL(Color){50, 220, 100, 72}
        : CLITERAL(Color){50, 140, 255, 62};
    const Color outline = is_player
        ? CLITERAL(Color){80, 255, 130, 225}
        : CLITERAL(Color){80, 180, 255, 210};
    for (int offset_x = 0; offset_x < entity->size; offset_x++) {
        for (int offset_y = 0; offset_y < entity->size; offset_y++) {
            float tile_x = (float)(entity->x + offset_x) + 0.5f;
            float tile_y = (float)(entity->y + offset_y) + 0.5f;
            float height = ground_y_smooth(v, tile_x, tile_y) + 0.035f;
            Vector3 center = {tile_x, height, -tile_y};
            DrawCube(center, 0.94f, 0.02f, 0.94f, fill);
        }
    }

    float min_x = (float)entity->x + 0.03f;
    float max_x = (float)(entity->x + entity->size) - 0.03f;
    float min_y = (float)entity->y + 0.03f;
    float max_y = (float)(entity->y + entity->size) - 0.03f;
    Vector3 northwest = {
        min_x, ground_y_smooth(v, min_x, min_y) + 0.055f, -min_y};
    Vector3 northeast = {
        max_x, ground_y_smooth(v, max_x, min_y) + 0.055f, -min_y};
    Vector3 southeast = {
        max_x, ground_y_smooth(v, max_x, max_y) + 0.055f, -max_y};
    Vector3 southwest = {
        min_x, ground_y_smooth(v, min_x, max_y) + 0.055f, -max_y};
    DrawLine3D(northwest, northeast, outline);
    DrawLine3D(northeast, southeast, outline);
    DrawLine3D(southeast, southwest, outline);
    DrawLine3D(southwest, northwest, outline);
}

static void draw_scene(ViewerState* v) {
    if (v->camera_locked) {
        v->camera.target = camera_follow_target(v);
    }
    v->camera.position = (Vector3){
        v->camera.target.x + v->cam_dist*cosf(v->cam_pitch)*sinf(v->cam_yaw),
        v->cam_dist*sinf(v->cam_pitch),
        v->camera.target.z + v->cam_dist*cosf(v->cam_pitch)*cosf(v->cam_yaw) };
    BeginMode3D(v->camera);

    /* Terrain + objects */
    if (v->terrain && v->terrain->loaded) {
        rlDisableBackfaceCulling();
        DrawModel(v->terrain->model, (Vector3){0,0,0}, 1.0f, WHITE);
        rlEnableBackfaceCulling();
    }
    if (v->objects && v->objects->loaded) {
        rlDisableBackfaceCulling();
        DrawModel(v->objects->model, (Vector3){0,0,0}, 1.0f, WHITE);
        rlEnableBackfaceCulling();
    }
    draw_animated_objects(v);

    /* Grid overlay */
    if (v->show_grid) {
        for (int x = 0; x <= FC_ARENA_WIDTH; x++)
            DrawLine3D((Vector3){(float)x,0.01f,0}, (Vector3){(float)x,0.01f,-(float)FC_ARENA_HEIGHT}, COL_GRID);
        for (int z = 0; z <= FC_ARENA_HEIGHT; z++)
            DrawLine3D((Vector3){0,0.01f,-(float)z}, (Vector3){(float)FC_ARENA_WIDTH,0.01f,-(float)z}, COL_GRID);
    }

    /* Collision overlay */
    if (v->show_collision) {
        for (int tx = 0; tx < FC_ARENA_WIDTH; tx++) {
            for (int ty = 0; ty < FC_ARENA_HEIGHT; ty++) {
                Color c = v->state.walkable[tx][ty] ? COL_WALKABLE : COL_BLOCKED;
                DrawCube((Vector3){tx+0.5f, 0.02f, -(ty+0.5f)}, 0.9f, 0.02f, 0.9f, c);
            }
        }
    }

    /* Movement feedback is immediate, even though the selected action remains
     * buffered until the next authoritative simulation tick. */
    draw_click_destination_3d(v);

    /* Entities */
    for (int i = 0; i < v->entity_count; i++) {
        FcRenderEntity* e = &v->entities[i];

        EntityRenderPose pose = entity_render_pose(v, e);
        float ex = pose.x;
        float ey = -pose.y;

        /* Sample terrain continuously along the interpolated movement path. */
        float gy = ground_y_smooth(v, pose.x, pose.y);

        if (e->entity_type == ENTITY_PLAYER) {
            draw_actor_footprint(v, e);

            /* Player model or fallback cylinder */
            NpcModelEntry* pm = fc_actor_player_model_entry(
                v->player_model, v->active_loadout);
            if (pm && pm->loaded) {
                Vector3 pos = {ex, gy, ey};
                float face_angle = pose.face_angle;
                rlDisableBackfaceCulling();
                DrawModelEx(pm->model, pos, (Vector3){0,1,0}, face_angle, (Vector3){1,1,1}, WHITE);
                rlEnableBackfaceCulling();
            } else {
                DrawCylinder((Vector3){ex, gy, ey}, 0.4f, 0.4f, 2.0f, 8, COL_PLAYER);
                DrawCylinderWires((Vector3){ex, gy, ey}, 0.4f, 0.4f, 2.0f, 8, WHITE);
            }

            /* Prayer icon above player — rendered as 2D text after EndMode3D */
            /* (handled below in the 2D overlay section) */
        } else {
            draw_actor_footprint(v, e);

            /* NPC: try to render actual model, fallback to colored cube */
            uint32_t mid = fc_npc_type_to_model_id(e->npc_type);
            NpcModelEntry* nme = v->npc_models ? fc_npc_model_find(v->npc_models, mid) : NULL;

            if (nme) {
                /* Facing is maintained by the client-style actor runtime. */
                Vector3 pos = {ex, gy, ey};
                float face_angle = pose.face_angle;
                if (e->npc_slot >= 0 && e->npc_slot < FC_MAX_NPCS &&
                    v->actor_animation.npc_states[e->npc_slot]) {
                    /* Same-type NPCs share an asset mesh. Upload this actor's
                     * transformed vertices immediately before its draw call. */
                    fc_actor_animation_upload_npc(
                        &v->actor_animation, e->npc_slot, nme);
                }
                rlDisableBackfaceCulling();
                DrawModelEx(nme->model, pos, (Vector3){0,1,0}, face_angle, (Vector3){1,1,1}, WHITE);
                rlEnableBackfaceCulling();
            } else {
                /* Fallback: colored cube */
                float s = (float)e->size * 0.45f;
                float h = 1.0f + (float)e->size * 0.5f;
                Color col = (e->npc_type > 0 && e->npc_type < 9) ? NPC_COLORS[e->npc_type] : GRAY;
                if (e->died_this_tick &&
                    !fc_combat_presentation_npc_death_deferred(
                        v->combat_presentation, &v->state, e->npc_slot)) {
                    h *= 0.3f;
                    col.a = 100;
                }
                DrawCube((Vector3){ex, gy + h*0.5f, ey}, s*2, h, s*2, col);
                DrawCubeWires((Vector3){ex, gy + h*0.5f, ey}, s*2, h, s*2, WHITE);
            }

        }
    }

    FcCombatPresentationContext combat_context = {
        .state = &v->state,
        .events = &v->render_events,
        .scene = &v->actor_animation.scene,
        .terrain = v->terrain,
        .anim_cache = v->anim_cache,
        .player_profile = fc_player_visual_profile(v->active_loadout),
        .tps = v->tps,
    };
    fc_combat_presentation_draw_world(v->combat_presentation,
                                      &combat_context, GetFrameTime());
    /* Debug overlays — 3D collision tiles (before EndMode3D) */
    if (v->dbg_flags) debug_overlay_3d(&v->state, v->dbg_flags);

    EndMode3D();

    FcCombatPresentationDrawContext combat_draw_context = {
        .presentation = combat_context,
        .entities = v->entities,
        .entity_count = v->entity_count,
        .player_models = v->player_model,
        .npc_models = v->npc_models,
        .active_loadout = v->active_loadout,
        .ui_assets = &v->ui.assets,
        .camera = v->camera,
    };
    /* Native client actor overheads are fixed-size screen-space sprites. */
    fc_combat_presentation_draw_healthbars(v->combat_presentation,
                                            &combat_draw_context);

    /* Debug overlays — 2D screen-space (LOS, path, range — after EndMode3D) */
    if (v->dbg_flags) {
        int debug_flags = v->dbg_flags;
        if (v->click_feedback.destination_active)
            debug_flags &= ~DBG_PATH;
        debug_overlay_screen(&v->state, v->camera, debug_flags);
        draw_npc_prayer_window_indicators(v);
    }

    draw_click_route_2d(v);

    fc_combat_presentation_draw_hitsplats(v->combat_presentation,
                                          &combat_draw_context);

    /* Prayer overhead icon — 2D projected from player head position */
    int rendered_prayer = fc_actor_animation_render_prayer(
        &v->actor_animation, &v->state);
    if (v->entity_count > 0 && rendered_prayer != PRAYER_NONE) {
        EntityRenderPose pose = entity_render_pose(v, &v->entities[0]);
        float p_gy = ground_y_smooth(v, pose.x, pose.y);
        Vector3 head_pos = {pose.x, p_gy + 3.0f, -pose.y};
        Vector2 scr = GetWorldToScreen(head_pos, v->camera);
        int px = (int)scr.x, py = (int)scr.y;

        /* Draw actual prayer sprite texture */
        Texture2D tex = {0};
        if (rendered_prayer == PRAYER_PROTECT_MELEE && v->pray_melee_tex.id > 0)
            tex = v->pray_melee_tex;
        else if (rendered_prayer == PRAYER_PROTECT_RANGE && v->pray_missiles_tex.id > 0)
            tex = v->pray_missiles_tex;
        else if (rendered_prayer == PRAYER_PROTECT_MAGIC && v->pray_magic_tex.id > 0)
            tex = v->pray_magic_tex;

        if (tex.id > 0) {
            /* Scale sprite to ~28x28 pixels and center on projected position */
            float scale = 28.0f / (float)tex.width;
            int dw = (int)(tex.width * scale);
            int dh = (int)(tex.height * scale);
            DrawTextureEx(tex, (Vector2){(float)(px - dw/2), (float)(py - dh/2)},
                          0.0f, scale, WHITE);
        } else {
            /* Fallback: letter if textures not loaded */
            const char* icon_txt;
            if (rendered_prayer == PRAYER_PROTECT_MELEE) icon_txt = "M";
            else if (rendered_prayer == PRAYER_PROTECT_RANGE) icon_txt = "R";
            else icon_txt = "W";
            DrawCircle(px, py, 14, (Color){255,255,255,220});
            int itw = fc_osrs_measure_text(icon_txt, 18);
            fc_osrs_draw_text(icon_txt, px - itw/2, py - 9, 18, (Color){0,0,0,255});
        }
    }
}

/* ======================================================================== */
/* UI drawing                                                                */
/* ======================================================================== */

static Rectangle runec_side_content_rect(void) {
    int screen_w = GetScreenWidth();
    int screen_h = GetScreenHeight();

    Rectangle side = {
        (float)screen_w - RUNEC_OSRS_SIDE_MENU_W,
        (float)screen_h - RUNEC_OSRS_SIDE_MENU_H,
        RUNEC_OSRS_SIDE_MENU_W,
        RUNEC_OSRS_SIDE_MENU_H
    };
    return (Rectangle){
        side.x + RUNEC_OSRS_SIDE_CONTENT_X,
        side.y + RUNEC_OSRS_SIDE_CONTENT_Y,
        RUNEC_OSRS_SIDE_CONTENT_W,
        RUNEC_OSRS_SIDE_CONTENT_H
    };
}

#define RUNEC_CONSOLE_TAB_COUNT 6
#define RUNEC_CONSOLE_NPC_ROWS 8
#define RUNEC_CONSOLE_NPC_ROW_H 13
#define RUNEC_CONSOLE_TPS_COLS 4

static Rectangle runec_console_panel_rect(void) {
    return runec_ui_chat_panel_rect(GetScreenWidth(), GetScreenHeight());
}

static Rectangle runec_console_body_rect(Rectangle panel) {
    return (Rectangle){panel.x + 7.0f, panel.y + 7.0f,
                       panel.width - 13.0f, panel.height - 39.0f};
}

static Rectangle runec_console_tab_rect(Rectangle panel, int index) {
    float width = (panel.width - 6.0f) / (float)RUNEC_CONSOLE_TAB_COUNT;
    return (Rectangle){panel.x + 3.0f + width * (float)index,
                       panel.y + panel.height - 23.0f,
                       width, 21.0f};
}

static Rectangle runec_console_debug_button_rect(Rectangle body) {
    float right_x = body.x + 286.0f;
    float width = body.x + body.width - right_x;
    return (Rectangle){right_x, body.y + 15.0f,
                       (width - 4.0f) * 0.5f, 18.0f};
}

static Rectangle runec_console_god_button_rect(Rectangle body) {
    Rectangle debug = runec_console_debug_button_rect(body);
    return (Rectangle){debug.x + debug.width + 4.0f, debug.y,
                       debug.width, debug.height};
}

static Rectangle runec_console_wave_button_rect(Rectangle body) {
    return (Rectangle){body.x + 286.0f, body.y + 37.0f,
                       body.width - 286.0f, 20.0f};
}

static Rectangle runec_console_tps_button_rect(Rectangle body,
                                               int index) {
    float box_x = body.x + 286.0f;
    float box_w = body.width - 286.0f;
    const float gap = 3.0f;
    float button_w = (box_w - gap * (RUNEC_CONSOLE_TPS_COLS - 1)) /
                     RUNEC_CONSOLE_TPS_COLS;
    int row = index / RUNEC_CONSOLE_TPS_COLS;
    int col = index % RUNEC_CONSOLE_TPS_COLS;
    return (Rectangle){box_x + col * (button_w + gap),
                       body.y + 78.0f + row * 20.0f,
                       button_w, 17.0f};
}

static Rectangle runec_console_target_row_rect(Rectangle body,
                                               int row) {
    return (Rectangle){body.x + 1.0f,
                       body.y + 16.0f + row * RUNEC_CONSOLE_NPC_ROW_H,
                       277.0f, RUNEC_CONSOLE_NPC_ROW_H};
}

static Rectangle runec_console_wave_cell_rect(Rectangle body,
                                              int wave) {
    const int columns = 9;
    const float gap = 1.0f;
    float cell_w = (body.width - 6.0f - gap * (columns - 1)) / columns;
    int index = wave - 1;
    int row = index / columns;
    int col = index % columns;
    return (Rectangle){body.x + 3.0f + col * (cell_w + gap),
                       body.y + 18.0f + row * 15.0f,
                       cell_w, 14.0f};
}

static void draw_runec_console_button(Rectangle rect, const char* label,
                                      int selected) {
    int hovered = CheckCollisionPointRec(GetMousePosition(), rect);
    Color background = selected ? CLITERAL(Color){82, 73, 61, 245}
        : (hovered ? CLITERAL(Color){72, 63, 51, 245}
                   : CLITERAL(Color){42, 36, 28, 235});
    Color text = selected ? COL_TEXT_YELLOW : COL_TEXT_WHITE;
    DrawRectangleRec(rect, background);
    DrawRectangleLinesEx(rect, 1, COL_PANEL_BORDER);
    int font_size = 8;
    fc_osrs_draw_text(label,
             (int)(rect.x + (rect.width - fc_osrs_measure_text(label, font_size)) * 0.5f),
             (int)(rect.y + (rect.height - font_size) * 0.5f),
             font_size, text);
}

static void draw_runec_console_wave_grid(ViewerState* v, Rectangle body) {
    Vector2 mouse = GetMousePosition();
    DrawRectangleRec(body, CLITERAL(Color){18, 16, 13, 252});
    DrawRectangleLinesEx(body, 1, COL_PANEL_BORDER);
    fc_osrs_draw_text("Select Wave", (int)body.x + 4, (int)body.y + 3,
             9, COL_TEXT_YELLOW);
    fc_osrs_draw_text("click current selection to close",
             (int)(body.x + body.width) - 155, (int)body.y + 4,
             7, COL_TEXT_DIM);
    for (int wave = 1; wave <= FC_NUM_WAVES; wave++) {
        Rectangle cell = runec_console_wave_cell_rect(body, wave);
        int current = wave == v->state.current_wave;
        int hovered = CheckCollisionPointRec(mouse, cell);
        Color bg = current ? CLITERAL(Color){60, 80, 40, 250}
            : (hovered ? CLITERAL(Color){72, 63, 51, 245}
                       : CLITERAL(Color){35, 30, 24, 240});
        DrawRectangleRec(cell, bg);
        DrawRectangleLinesEx(cell, 1, COL_PANEL_BORDER);
        char label[8];
        snprintf(label, sizeof(label), "%d", wave);
        fc_osrs_draw_text(label,
                 (int)(cell.x + (cell.width - fc_osrs_measure_text(label, 8)) * 0.5f),
                 (int)cell.y + 3, 8,
                 current ? COL_TEXT_YELLOW : COL_TEXT_WHITE);
    }
}

static void draw_runec_console_controls(ViewerState* v, Rectangle body) {
    static const char* NPC_SHORT[] = {
        "?", "Tz-Kih", "Tz-Kek", "Kek-Sm", "Tok-Xil",
        "MejKot", "Ket-Zek", "Jad", "HurKot"
    };
    Vector2 mouse = GetMousePosition();
    int x = (int)body.x + 4;
    char text[128];

    fc_osrs_draw_text("NPC Targets", x, (int)body.y + 3, 9, COL_TEXT_YELLOW);
    int shown = 0;
    for (int ni = 0; ni < FC_MAX_NPCS && shown < RUNEC_CONSOLE_NPC_ROWS; ni++) {
        FcNpc* npc = &v->state.npcs[ni];
        if (!npc->active || npc->is_dead) continue;
        Rectangle row = runec_console_target_row_rect(body, shown);
        int selected = ni == v->state.player.attack_target_idx;
        int hovered = CheckCollisionPointRec(mouse, row);
        if (selected || hovered) {
            DrawRectangleRec(row, selected
                ? CLITERAL(Color){82, 73, 61, 205}
                : CLITERAL(Color){52, 45, 35, 190});
        }
        const char* name = npc->npc_type > 0 && npc->npc_type < 9
            ? NPC_SHORT[npc->npc_type] : "?";
        snprintf(text, sizeof(text), "%s%s[%d] d:%d",
                 selected ? ">" : " ", name, ni,
                 fc_distance_to_npc(v->state.player.x,
                                    v->state.player.y, npc));
        fc_osrs_draw_text(text, (int)row.x + 2, (int)row.y + 3, 8,
                 selected ? COL_TEXT_YELLOW : COL_TEXT_WHITE);

        int bar_x = (int)row.x + 116;
        int bar_w = 116;
        float hp = npc->max_hp > 0
            ? (float)npc->current_hp / (float)npc->max_hp : 0.0f;
        if (hp < 0.0f) hp = 0.0f;
        if (hp > 1.0f) hp = 1.0f;
        DrawRectangle(bar_x, (int)row.y + 3, bar_w, 7, COL_HP_RED);
        DrawRectangle(bar_x, (int)row.y + 3,
                      (int)((float)bar_w * hp), 7, COL_HP_GREEN);
        snprintf(text, sizeof(text), "%d", npc->current_hp / 10);
        fc_osrs_draw_text(text, bar_x + bar_w + 4, (int)row.y + 2,
                 8, COL_TEXT_DIM);
        shown++;
    }
    if (shown == 0)
        fc_osrs_draw_text("No NPCs alive", x + 2, (int)body.y + 20,
                 8, COL_TEXT_DIM);

    int right_x = (int)body.x + 286;
    DrawLine(right_x - 5, (int)body.y + 2,
             right_x - 5, (int)(body.y + body.height) - 2,
             COL_PANEL_BORDER);
    fc_osrs_draw_text("Viewer Controls", right_x, (int)body.y + 3,
             9, COL_TEXT_YELLOW);

    char debug_label[24];
    snprintf(debug_label, sizeof(debug_label), "Debug: %s",
             v->dbg_flags ? "ON" : "OFF");
    draw_runec_console_button(runec_console_debug_button_rect(body),
                              debug_label, v->dbg_flags != 0);
    char god_label[24];
    snprintf(god_label, sizeof(god_label), "God: %s",
             v->godmode ? "ON" : "OFF");
    draw_runec_console_button(runec_console_god_button_rect(body),
                              god_label, v->godmode);

    Rectangle wave = runec_console_wave_button_rect(body);
    snprintf(text, sizeof(text), "Jump to Wave: %d  v",
             v->state.current_wave);
    draw_runec_console_button(wave, text, 0);

    fc_osrs_draw_text(v->policy_pipe ? "Replay TPS" : "TPS Presets",
             right_x, (int)body.y + 62, 8, COL_TEXT_DIM);
    for (int i = 0; i < NUM_MANUAL_TPS_PRESETS; i++) {
        draw_runec_console_button(runec_console_tps_button_rect(body, i),
                                  MANUAL_TPS_LABELS[i],
                                  float_near(v->tps, MANUAL_TPS_PRESETS[i]));
    }

    if (v->console_wave_dropdown_open)
        draw_runec_console_wave_grid(v, body);
}

static void draw_runec_console_diagnostics(ViewerState* v, Rectangle body) {
    int debug_tab = v->console_tab - 1;
    int available_height = (int)body.height - 6;
    int scroll = 0;
    if (debug_tab >= 0 && debug_tab < 4) {
        int max_scroll = v->console_content_height[debug_tab] - available_height;
        if (max_scroll < 0) max_scroll = 0;
        if (v->console_scroll[debug_tab] < 0)
            v->console_scroll[debug_tab] = 0;
        if (v->console_scroll[debug_tab] > max_scroll)
            v->console_scroll[debug_tab] = max_scroll;
        scroll = v->console_scroll[debug_tab];
    }

    int content_y = (int)body.y + 3 - scroll;
    BeginScissorMode((int)body.x, (int)body.y,
                     (int)body.width, (int)body.height);
    int end_y = dbg_draw_panel_tabs(
        &v->state,
        &v->reward_breakdown, &v->reward_runtime,
        v->reward_config_loaded, v->reward_config_path,
        (int)body.x, (int)body.x + 4, content_y,
        (int)body.width, debug_tab, 0, available_height);
    EndScissorMode();

    if (debug_tab >= 0 && debug_tab < 4) {
        int height = end_y - content_y;
        v->console_content_height[debug_tab] = height > 0 ? height : 0;
        int max_scroll = height - available_height;
        if (max_scroll < 0) max_scroll = 0;
        if (v->console_scroll[debug_tab] > max_scroll)
            v->console_scroll[debug_tab] = max_scroll;
        if (max_scroll > 0) {
            int track_x = (int)(body.x + body.width) - 6;
            int track_y = (int)body.y + 3;
            int track_h = available_height;
            DrawRectangle(track_x, track_y, 4, track_h,
                          CLITERAL(Color){30, 26, 20, 230});
            float visible_fraction =
                (float)available_height / (float)height;
            int thumb_h = (int)((float)track_h * visible_fraction);
            if (thumb_h < 12) thumb_h = 12;
            float scroll_fraction = max_scroll > 0
                ? (float)v->console_scroll[debug_tab] / (float)max_scroll
                : 0.0f;
            int thumb_y = track_y +
                (int)((float)(track_h - thumb_h) * scroll_fraction);
            DrawRectangle(track_x, thumb_y, 4, thumb_h,
                          CLITERAL(Color){120, 110, 90, 255});
        }
    }
}

static void draw_runec_console(ViewerState* v) {
    static const char* labels[RUNEC_CONSOLE_TAB_COUNT] = {
        "Controls", "Player", "Obs", "Mask", "Reward", "Log"
    };
    Rectangle panel = runec_console_panel_rect();
    Rectangle body = runec_console_body_rect(panel);
    DrawRectangleRec(body, CLITERAL(Color){0, 0, 0, 76});
    if (v->console_tab == 0)
        draw_runec_console_controls(v, body);
    else
        draw_runec_console_diagnostics(v, body);

    for (int tab = 0; tab < RUNEC_CONSOLE_TAB_COUNT; tab++) {
        Rectangle rect = runec_console_tab_rect(panel, tab);
        runec_ui_draw_asset(&v->ui.assets, "chat_tab_button_0", rect, WHITE);
        int selected = tab == v->console_tab;
        int hovered = CheckCollisionPointRec(GetMousePosition(), rect);
        DrawRectangleRec(rect, selected
            ? CLITERAL(Color){82, 73, 61, 96}
            : (hovered ? CLITERAL(Color){90, 78, 62, 72}
                       : CLITERAL(Color){0, 0, 0, 18}));
        if (selected)
            DrawLine((int)rect.x + 2, (int)(rect.y + rect.height) - 2,
                     (int)(rect.x + rect.width) - 2,
                     (int)(rect.y + rect.height) - 2,
                     COL_TEXT_YELLOW);
        fc_osrs_draw_text(labels[tab],
                 (int)(rect.x +
                     (rect.width - fc_osrs_measure_text(labels[tab], 8)) * 0.5f),
                 (int)rect.y + 7, 8,
                 selected ? COL_TEXT_YELLOW : COL_TEXT_WHITE);
    }
}

static int process_runec_console_input(ViewerState* v) {
    if (!v) return 0;
    Rectangle panel = runec_console_panel_rect();
    Vector2 mouse = GetMousePosition();
    if (!CheckCollisionPointRec(mouse, panel))
        return 0;

    int clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
    if (clicked) {
        for (int tab = 0; tab < RUNEC_CONSOLE_TAB_COUNT; tab++) {
            if (!CheckCollisionPointRec(mouse,
                                        runec_console_tab_rect(panel, tab)))
                continue;
            v->console_tab = tab;
            v->console_wave_dropdown_open = 0;
            return 1;
        }
    }

    Rectangle body = runec_console_body_rect(panel);
    if (v->console_tab == 0 && clicked) {
        if (v->console_wave_dropdown_open) {
            for (int wave = 1; wave <= FC_NUM_WAVES; wave++) {
                if (!CheckCollisionPointRec(
                        mouse, runec_console_wave_cell_rect(body, wave)))
                    continue;
                if (wave != v->state.current_wave)
                    viewer_jump_to_wave(v, wave);
                v->console_wave_dropdown_open = 0;
                return 1;
            }
            v->console_wave_dropdown_open = 0;
            return 1;
        }

        if (CheckCollisionPointRec(mouse,
                                   runec_console_debug_button_rect(body))) {
            toggle_debug_overlay(v);
            return 1;
        }
        if (CheckCollisionPointRec(mouse,
                                   runec_console_god_button_rect(body))) {
            toggle_godmode(v);
            return 1;
        }
        if (CheckCollisionPointRec(mouse,
                                   runec_console_wave_button_rect(body))) {
            v->console_wave_dropdown_open = 1;
            return 1;
        }
        for (int i = 0; i < NUM_MANUAL_TPS_PRESETS; i++) {
            if (!CheckCollisionPointRec(
                    mouse, runec_console_tps_button_rect(body, i)))
                continue;
            set_manual_speed(v, MANUAL_TPS_PRESETS[i]);
            return 1;
        }

        int shown = 0;
        for (int ni = 0; ni < FC_MAX_NPCS &&
                 shown < RUNEC_CONSOLE_NPC_ROWS; ni++) {
            FcNpc* npc = &v->state.npcs[ni];
            if (!npc->active || npc->is_dead) continue;
            if (CheckCollisionPointRec(
                    mouse, runec_console_target_row_rect(body, shown))) {
                queue_player_attack_request(v, ni, mouse.x, mouse.y);
                fprintf(stderr, "CONSOLE CLICK -> ATTACK npc_idx=%d\n", ni);
                return 1;
            }
            shown++;
        }
    }

    if (v->console_tab >= 1 && v->console_tab <= 4 &&
        CheckCollisionPointRec(mouse, body)) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            int index = v->console_tab - 1;
            int max_scroll = v->console_content_height[index] -
                             ((int)body.height - 6);
            if (max_scroll < 0) max_scroll = 0;
            v->console_scroll[index] -= (int)wheel * 24;
            if (v->console_scroll[index] < 0)
                v->console_scroll[index] = 0;
            if (v->console_scroll[index] > max_scroll)
                v->console_scroll[index] = max_scroll;
        }
    }
    return 1;
}

static void draw_tex_fit(Texture2D tex, int dx, int dy, int dw, int dh,
                         Color tint) {
    if (tex.id == 0) return;
    Rectangle src = {0, 0, (float)tex.width, (float)tex.height};
    float sx = (float)dw / (float)tex.width;
    float sy = (float)dh / (float)tex.height;
    float scale = sx < sy ? sx : sy;
    int rw = (int)(tex.width * scale);
    int rh = (int)(tex.height * scale);
    Rectangle dst = {
        (float)(dx + (dw - rw) / 2),
        (float)(dy + (dh - rh) / 2),
        (float)rw,
        (float)rh,
    };
    DrawTexturePro(tex, src, dst, (Vector2){0, 0}, 0, tint);
}

static Rectangle runec_prayer_button_rect(Rectangle content, int index) {
    const int btn_h = 34;
    const int gap = 3;
    int x = (int)content.x + 8;
    int y = (int)content.y + 8 + 34 + index * (btn_h + gap);
    int w = (int)content.width - 16;
    if (w < 120) w = 120;
    return (Rectangle){(float)x, (float)y, (float)w, (float)btn_h};
}

static void draw_runec_prayer_tab(ViewerState* v, Rectangle content) {
    BeginScissorMode((int)content.x, (int)content.y,
                     (int)content.width, (int)content.height);

    DrawRectangleRec(content, COL_PANEL);

    FcPlayer* p = &v->state.player;
    int rendered_prayer = fc_actor_animation_render_prayer(
        &v->actor_animation, &v->state);
    int x = (int)content.x + 8;
    int by = (int)content.y + 8;
    int right = (int)(content.x + content.width) - 4;
    char b[64];

    snprintf(b, sizeof(b), "Prayer: %d / %d",
             p->current_prayer / 10, p->max_prayer / 10);
    text_s(b, x, by, 10, COL_PRAY_BLUE);
    by += 16;

    if (rendered_prayer != PRAYER_NONE) {
        int resistance = 60 + 2 * p->prayer_bonus;
        snprintf(b, sizeof(b), "Drain rate: 12 / %d resist", resistance);
        text_s(b, x, by, 8, COL_TEXT_DIM);
    } else {
        text_s("No prayer active", x, by, 8, COL_TEXT_DIM);
    }
    by += 14;

    DrawLine((int)content.x + 4, by - 2, right, by - 2, COL_PANEL_BORDER);

    static const char* pray_names[] = {
        "Prot. Melee", "Prot. Range", "Prot. Magic"
    };
    static const int pray_vals[] = {
        PRAYER_PROTECT_MELEE, PRAYER_PROTECT_RANGE, PRAYER_PROTECT_MAGIC
    };
    Color pray_colors[] = { COL_TEXT_YELLOW, COL_TEXT_GREEN, COL_PRAY_BLUE };
    Texture2D tex_on[] = {
        v->tex_pray_melee_on, v->tex_pray_range_on, v->tex_pray_magic_on
    };
    Texture2D tex_off[] = {
        v->tex_pray_melee_off, v->tex_pray_range_off, v->tex_pray_magic_off
    };

    int no_points = (p->current_prayer <= 0);
    Vector2 mouse = GetMousePosition();
    const Color slot_empty = CLITERAL(Color){30, 26, 20, 255};
    const Color pray_active = CLITERAL(Color){60, 120, 200, 200};
    const Color tab_hover = CLITERAL(Color){72, 63, 51, 255};
    const Color pray_button = CLITERAL(Color){50, 44, 36, 255};
    for (int i = 0; i < 3; i++) {
        Rectangle br = runec_prayer_button_rect(content, i);
        int is_active = (rendered_prayer == pray_vals[i]);
        int hovered = CheckCollisionPointRec(mouse, br);

        Color bg;
        if (no_points) {
            bg = slot_empty;
        } else if (is_active) {
            bg = pray_active;
        } else if (hovered) {
            bg = tab_hover;
        } else {
            bg = pray_button;
        }
        DrawRectangleRec(br, bg);
        DrawRectangleLinesEx(br, is_active ? 2 : 1,
                             is_active ? pray_colors[i] : COL_PANEL_BORDER);

        Texture2D icon = is_active ? tex_on[i] : tex_off[i];
        Color icon_tint = no_points ? CLITERAL(Color){80,80,80,255} : WHITE;
        draw_tex_fit(icon, (int)br.x + 4, (int)br.y + 2, 30, 30, icon_tint);

        Color tc = no_points ? COL_TEXT_DIM
            : (is_active ? COL_TEXT_WHITE : pray_colors[i]);
        text_s(pray_names[i], (int)br.x + 38, (int)br.y + 7, 10, tc);

        snprintf(b, sizeof(b), "[%d]", i + 1);
        text_s(b, (int)br.x + 38, (int)br.y + 21, 8, COL_TEXT_DIM);
        if (is_active) {
            text_s("ACTIVE", (int)(br.x + br.width) - 44,
                   (int)br.y + 21, 8, COL_TEXT_WHITE);
        }
    }

    by = (int)runec_prayer_button_rect(content, 2).y + 34 + 9;
    snprintf(b, sizeof(b), "Prayer bonus: +%d", p->prayer_bonus);
    text_s(b, x, by, 8, COL_TEXT_DIM);

    EndScissorMode();
}

static void draw_runec_side_overrides(ViewerState* v) {
    Rectangle content = runec_side_content_rect();
    if (v->ui.active_tab == RUNEC_UI_TAB_PRAYER)
        draw_runec_prayer_tab(v, content);
}

static int process_runec_prayer_click(ViewerState* v) {
    if (!v || v->ui.active_tab != RUNEC_UI_TAB_PRAYER)
        return 0;
    int left_click = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
    int right_click = IsMouseButtonPressed(MOUSE_BUTTON_RIGHT);
    if (!left_click && !right_click)
        return 0;

    Rectangle content = runec_side_content_rect();
    Vector2 mouse = GetMousePosition();
    if (!CheckCollisionPointRec(mouse, content))
        return 0;

    if (right_click)
        return 1;

    static const struct {
        int prayer;
        int action;
    } buttons[] = {
        {PRAYER_PROTECT_MELEE, FC_PRAYER_MELEE},
        {PRAYER_PROTECT_RANGE, FC_PRAYER_RANGE},
        {PRAYER_PROTECT_MAGIC, FC_PRAYER_MAGIC},
    };

    for (int i = 0; i < (int)(sizeof(buttons) / sizeof(buttons[0])); i++) {
        Rectangle r = runec_prayer_button_rect(content, i);
        if (!CheckCollisionPointRec(mouse, r))
            continue;
        queue_viewer_prayer_button(v, buttons[i].prayer, buttons[i].action);
        return 1;
    }

    return 1;
}
/* ======================================================================== */
/* Main                                                                      */
/* ======================================================================== */

int main(int argc, char** argv) {
    int exit_code = 0;
    int screenshot_mode = 0;
    const char* screenshot_path = NULL;
    int policy_pipe_flag = 0;
    int policy_speed_flag = 1;
    int policy_episode_limit_flag = 0;
    int start_wave_flag = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--screenshot") == 0 && i+1 < argc) {
            screenshot_mode = 1;
            screenshot_path = argv[++i];
        } else if (strcmp(argv[i], "--policy-pipe") == 0) {
            policy_pipe_flag = 1;
        } else if (strcmp(argv[i], "--speed") == 0 && i+1 < argc) {
            policy_speed_flag = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--episodes") == 0 && i+1 < argc) {
            policy_episode_limit_flag = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--start-wave") == 0 && i+1 < argc) {
            start_wave_flag = atoi(argv[++i]);
        }
    }
    fprintf(stderr,"=== Fight Caves Viewer (Phase 8 — Playable) ===\n");
    /* In policy-pipe mode, suppress Raylib's INFO logs which go to stdout
     * and would corrupt the pipe protocol. */
    if (policy_pipe_flag) {
        SetTraceLogCallback(viewer_trace_log_to_stderr);
        SetTraceLogLevel(LOG_WARNING);
    }
    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_MSAA_4X_HINT);
    InitWindow(DEFAULT_WINDOW_W, DEFAULT_WINDOW_H,
               "Fight Caves RL — Playable Viewer");
    if (!IsWindowReady()) {
        fprintf(stderr,
                "error: viewer window initialization failed; verify the "
                "graphical display and OpenGL driver\n");
        return 1;
    }
    SetTargetFPS(60);

    ViewerState v; memset(&v, 0, sizeof(v));
    fc_init(&v.state);
    fc_actor_animation_init(&v.actor_animation);
    runec_ui_init(&v.ui);
    if (!fc_osrs_text_init()) {
        fprintf(stderr,
                "error: required OSRS viewer fonts failed to load\n");
        runec_ui_shutdown(&v.ui);
        CloseWindow();
        return 1;
    }
    int item_icons_ready = load_fc_ui_item_icons(&v);
    v.paused = 1; v.tps = NORMAL_TPS;
    v.active_loadout = FC_ACTIVE_LOADOUT;
    v.attack_target = -1;
    v.cam_yaw = 0; v.cam_pitch = 0.8f; v.cam_dist = 30;
    v.camera_locked = 1;
    v.camera.up = (Vector3){0,1,0}; v.camera.fovy = 32;
    v.camera.projection = CAMERA_PERSPECTIVE;
    v.camera.target = (Vector3){FC_ARENA_WIDTH * 0.5f, 0.5f, -(FC_ARENA_HEIGHT * 0.5f)};

    v.terrain = load_terrain(&v);
    /* OSRS rasterizes the current 104x104 scene from cache terrain and
     * locations into a 512x512 minimap. This asset contains the Fight Caves
     * mapsquare centered in that same scene format; runtime only crops and
     * rotates it around the player. */
    Image minimap_image = fc_load_image_asset("fightcaves.minimap.png");
    if (minimap_image.data) {
        Color* minimap_pixels = LoadImageColors(minimap_image);
        if (!minimap_pixels || !fc_minimap_scene_load_pixels(
                &v.minimap_scene, minimap_pixels,
                minimap_image.width, minimap_image.height)) {
            fprintf(stderr, "error: Fight Caves minimap failed to load\n");
        } else {
            fprintf(stderr,
                    "minimap: loaded cache scene raster %dx%d\n",
                    minimap_image.width, minimap_image.height);
        }
        UnloadImageColors(minimap_pixels);
        UnloadImage(minimap_image);
    } else {
        fprintf(stderr, "error: missing fightcaves.minimap.png\n");
    }
    v.objects = load_objects_with_terrain(v.terrain);
    if (fc_asset_exists("fightcaves.oanim"))
        v.object_anims = object_anims_load("fightcaves.oanim");
    if (v.object_anims)
        object_anims_offset(v.object_anims, FC_WORLD_ORIGIN_X, FC_WORLD_ORIGIN_Y);
    if (!fc_animated_atlas_load(&v.shared_model_atlas, "fightcaves.atlas", 0))
        fprintf(stderr, "error: shared model atlas failed to load\n");
    if (fc_asset_exists("fightcaves.object_anim.models"))
        v.object_anim_models = fc_npc_models_load(
            "fightcaves.object_anim.models", v.shared_model_atlas.texture);
    if (v.object_anims && v.object_anims->count > 0) {
        v.object_anim_runtimes = (ObjectAnimRuntime*)calloc(
            (size_t)v.object_anims->count, sizeof(*v.object_anim_runtimes));
        if (v.object_anim_runtimes)
            v.object_anim_runtime_count = v.object_anims->count;
    }
    if (!v.terrain || !v.terrain->loaded) v.show_grid = 1;

    /* Load NPC models */
    {
        if (fc_asset_exists("fc_npcs.models"))
            v.npc_models = fc_npc_models_load("fc_npcs.models", (Texture2D){0});
        if (!v.npc_models) fprintf(stderr, "error: NPC models failed to load\n");
    }

    /* Load player model */
    {
        if (fc_asset_exists("fc_player.models"))
            v.player_model = fc_npc_models_load("fc_player.models", (Texture2D){0});
    }

    /* Load the animation cache shared by actor and combat presentation. */
    if (fc_asset_exists("fc_all.anims"))
        v.anim_cache = anim_cache_load("fc_all.anims");
    v.combat_presentation = fc_combat_presentation_create(
        v.shared_model_atlas.texture);
    if (!v.combat_presentation)
        fprintf(stderr, "error: combat presentation initialization failed\n");

    /* Load prayer overhead icon textures */
    {
        if (fc_asset_exists("data/sprites/ui/prayeron_14.png")) {
            v.pray_melee_tex = fc_load_texture_asset("data/sprites/ui/prayeron_14.png");
            v.pray_missiles_tex = fc_load_texture_asset("data/sprites/ui/prayeron_13.png");
            v.pray_magic_tex = fc_load_texture_asset("data/sprites/ui/prayeron_12.png");
            fprintf(stderr, "Prayer icons loaded from %s\n", fc_asset_root());
        } else {
            fprintf(stderr, "error: prayer icons not found under asset root %s\n",
                    fc_asset_root());
        }
    }

    /* Native b237 click crosses: frames 0-3 are movement (yellow), frames
     * 4-7 are interaction (red). RuneC advances one frame every 100 ms. */
    int click_cross_loaded = 0;
    {
        for (int i = 0; i < FC_CLICK_CROSS_FRAME_COUNT * 2; i++) {
            char path[64];
            snprintf(path, sizeof(path),
                     "data/sprites/ui/cross_%d.png", i);
            v.click_cross_tex[i] = fc_load_texture_asset(path);
            if (v.click_cross_tex[i].id > 0) {
                SetTextureFilter(v.click_cross_tex[i], TEXTURE_FILTER_POINT);
                click_cross_loaded++;
            }
        }
        fprintf(stderr, "Click cross sprites loaded: %d/8\n",
                click_cross_loaded);
        if (click_cross_loaded != FC_CLICK_CROSS_FRAME_COUNT * 2)
            fprintf(stderr, "error: required click cross sprites failed to load\n");
    }

    /* Load prayer icons used by the active RuneC prayer override. */
    {
        v.tex_pray_melee_on = fc_load_texture_asset(
            "data/sprites/ui/prayeron_14.png");
        v.tex_pray_melee_off = fc_load_texture_asset(
            "data/sprites/ui/prayeroff_14.png");
        v.tex_pray_range_on = fc_load_texture_asset(
            "data/sprites/ui/prayeron_13.png");
        v.tex_pray_range_off = fc_load_texture_asset(
            "data/sprites/ui/prayeroff_13.png");
        v.tex_pray_magic_on = fc_load_texture_asset(
            "data/sprites/ui/prayeron_12.png");
        v.tex_pray_magic_off = fc_load_texture_asset(
            "data/sprites/ui/prayeroff_12.png");
    }

    int required_resources_ready = 1;
#define REQUIRE_VIEWER_RESOURCE(condition, description) do {                 \
        if (!(condition)) {                                                  \
            fprintf(stderr, "error: required viewer resource failed: %s\n", \
                    description);                                            \
            required_resources_ready = 0;                                    \
        }                                                                    \
    } while (0)
    REQUIRE_VIEWER_RESOURCE(v.ui.assets.missing_required_count == 0,
                            "RuneC UI sprites");
    REQUIRE_VIEWER_RESOURCE(v.ui.assets.font_loaded &&
                            v.ui.assets.small_font_loaded,
                            "RuneC UI fonts");
    REQUIRE_VIEWER_RESOURCE(v.ui.minimap_texture_ready,
                            "minimap render texture");
    REQUIRE_VIEWER_RESOURCE(item_icons_ready, "Fight Caves item icons");
    REQUIRE_VIEWER_RESOURCE(v.terrain && v.terrain->loaded, "terrain mesh");
    REQUIRE_VIEWER_RESOURCE(v.minimap_scene.ready, "minimap scene raster");
    REQUIRE_VIEWER_RESOURCE(v.objects && v.objects->loaded &&
                            v.objects->atlas.texture.id > 0,
                            "terrain objects and atlas");
    REQUIRE_VIEWER_RESOURCE(v.object_anims && v.object_anims->loaded,
                            "object animation placements");
    REQUIRE_VIEWER_RESOURCE(v.shared_model_atlas.texture.id > 0,
                            "shared model atlas");
    REQUIRE_VIEWER_RESOURCE(v.object_anim_models &&
                            v.object_anim_models->loaded,
                            "animated object models");
    REQUIRE_VIEWER_RESOURCE(!v.object_anims || v.object_anims->count == 0 ||
                            v.object_anim_runtimes,
                            "object animation runtime allocation");
    REQUIRE_VIEWER_RESOURCE(v.npc_models && v.npc_models->loaded,
                            "NPC models");
    REQUIRE_VIEWER_RESOURCE(v.player_model && v.player_model->loaded,
                            "player models");
    REQUIRE_VIEWER_RESOURCE(v.anim_cache, "actor animation data");
    REQUIRE_VIEWER_RESOURCE(
        fc_combat_presentation_ready(v.combat_presentation),
        "projectile, spot-animation, healthbar, and hitsplat data");
    REQUIRE_VIEWER_RESOURCE(v.pray_melee_tex.id > 0 &&
                            v.pray_missiles_tex.id > 0 &&
                            v.pray_magic_tex.id > 0,
                            "overhead Prayer icons");
    REQUIRE_VIEWER_RESOURCE(
        click_cross_loaded == FC_CLICK_CROSS_FRAME_COUNT * 2,
        "click cross sprites");
    REQUIRE_VIEWER_RESOURCE(v.tex_pray_melee_on.id > 0 &&
                            v.tex_pray_melee_off.id > 0 &&
                            v.tex_pray_range_on.id > 0 &&
                            v.tex_pray_range_off.id > 0 &&
                            v.tex_pray_magic_on.id > 0 &&
                            v.tex_pray_magic_off.id > 0,
                            "Prayer interface icons");
#undef REQUIRE_VIEWER_RESOURCE
    if (!required_resources_ready) {
        fprintf(stderr,
                "error: viewer startup aborted instead of using reduced "
                "graphics; reinstall and verify assets with: python3 "
                "ocean/fight_caves/tools.py setup --all --force\n");
        exit_code = 1;
        goto cleanup;
    }

    v.combat_style = 1;  /* Rapid default */
    v.policy_pipe = policy_pipe_flag;
    v.policy_episode_limit = policy_episode_limit_flag;
    v.start_wave = start_wave_flag;
    if (v.policy_pipe)
        set_policy_replay_speed(&v, policy_speed_flag);

    reset_ep(&v);

    /* Policy pipe: write initial obs so Python can send first action */
    if (v.policy_pipe) {
        v.paused = 0;
        fprintf(stderr, "[policy-pipe] Mode active. Reading actions from stdin.\n");
        write_obs_to_pipe(&v);
    }

    int frame_count = 0;

    while (!WindowShouldClose()) {
        int quit_after_tick = 0;
        int ui_capture = 0;
        /* Screenshot mode */
        if (screenshot_mode && frame_count == 5) {
            TakeScreenshot(screenshot_path);
            fprintf(stderr, "Screenshot saved to %s\n", screenshot_path);
            break;
        }
        frame_count++;

        /* Global keys (always active) */
        if (IsKeyPressed(KEY_Q)) break;
        if (IsKeyPressed(KEY_SPACE)) v.paused = !v.paused;
        if (IsKeyPressed(KEY_RIGHT)) v.step_once = 1;
        if (v.policy_pipe) {
            if (IsKeyPressed(KEY_ONE)) set_policy_replay_speed(&v, 1);
            if (IsKeyPressed(KEY_TWO)) set_policy_replay_speed(&v, 2);
            if (!IsKeyDown(KEY_LEFT_SHIFT) && !IsKeyDown(KEY_RIGHT_SHIFT) &&
                IsKeyPressed(KEY_FOUR)) set_policy_replay_speed(&v, 4);
            if (IsKeyPressed(KEY_ZERO)) set_policy_replay_speed(&v, 10);
            if (IsKeyPressed(KEY_UP)) cycle_policy_replay_speed(&v, +1);
            if (IsKeyPressed(KEY_DOWN)) cycle_policy_replay_speed(&v, -1);
        }
        if (IsKeyPressed(KEY_R)) reset_ep(&v);
        if (IsKeyPressed(KEY_L)) {
            if (v.camera_locked) {
                v.camera.target = camera_follow_target(&v);
            }
            v.camera_locked = !v.camera_locked;
        }

        /* Toggle keys */
        if (IsKeyPressed(KEY_G)) v.show_grid = !v.show_grid;
        if (IsKeyPressed(KEY_C)) v.show_collision = !v.show_collision;
        /* O: cycle debug overlay modes. O=all on/off, Shift+O=cycle sub-modes */
        if (IsKeyPressed(KEY_O)) {
            if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
                /* Cycle through individual modes */
                if (v.dbg_flags == 0) v.dbg_flags = DBG_COLLISION;
                else if (v.dbg_flags == DBG_COLLISION) v.dbg_flags = DBG_LOS;
                else if (v.dbg_flags == DBG_LOS) v.dbg_flags = DBG_PATH | DBG_RANGE;
                else v.dbg_flags = 0;
            } else {
                /* Toggle all on/off */
                toggle_debug_overlay(&v);
            }
        }
        /* D: match the on-screen controls without interfering with east movement */
        if (IsKeyPressed(KEY_D) && !IsKeyDown(KEY_W) && !IsKeyDown(KEY_A) && !IsKeyDown(KEY_S)) {
            toggle_debug_overlay(&v);
        }
        /* Camera presets */
        if ((!v.policy_pipe && IsKeyPressed(KEY_FOUR)) ||
            (v.policy_pipe &&
             (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) &&
             IsKeyPressed(KEY_FOUR))) {
            v.cam_yaw=0; v.cam_pitch=1.35f; v.cam_dist=120;
        }
        if ((!v.policy_pipe && IsKeyPressed(KEY_FIVE)) ||
            (v.policy_pipe &&
             (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) &&
             IsKeyPressed(KEY_FIVE))) {
            v.cam_yaw=0; v.cam_pitch=0.6f; v.cam_dist=50;
        }

        sync_fc_ui(&v);
        ui_capture = process_runec_prayer_click(&v);
        if (!ui_capture)
            ui_capture = process_runec_console_input(&v);
        if (!ui_capture) {
            ui_capture = runec_ui_handle_input(&v.ui, GetScreenWidth(), GetScreenHeight());
            handle_runec_ui_intent(&v);
        } else {
            v.ui.last_intent.kind = RUNEC_UI_INTENT_NONE;
        }

        /* Camera orbit + zoom */
        if (!ui_capture && IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 d = GetMouseDelta();
            v.cam_yaw += d.x*0.005f; v.cam_pitch -= d.y*0.005f;
            if (v.cam_pitch < 0.1f) v.cam_pitch = 0.1f;
            if (v.cam_pitch > 1.4f) v.cam_pitch = 1.4f;
        }
        float wh = GetMouseWheelMove();
        if (!ui_capture && wh != 0) {
            v.cam_dist *= (wh > 0) ? (1.0f/1.15f) : 1.15f;
            if (v.cam_dist < 5) v.cam_dist = 5;
            if (v.cam_dist > 300) v.cam_dist = 300;
        }

        /* Tick processing */
        int tick = 0;
        if (!v.paused) {
            v.tick_acc += GetFrameTime() * (float)v.tps;
            if (v.tick_acc >= 1.0f) {
                v.tick_acc = fmodf(v.tick_acc, 1.0f);
                tick = 1;
            }
        }
        if (v.step_once) { tick = 1; v.step_once = 0; }

        /* Capture clicks and key presses EVERY frame (60fps).
         * These set routes/targets/buffers on the player struct.
         * The tick loop reads them when the next tick fires. */
        if (!v.policy_pipe && v.state.terminal == TERMINAL_NONE) {
            process_human_clicks(&v, ui_capture);
            process_human_keys(&v);
        }

        if (tick && v.state.terminal == TERMINAL_NONE) {
            int used_human_actions = 0;
            /* Build action array for this tick */
            if (v.policy_pipe) {
                if (!read_policy_actions(&v)) {
                    fprintf(stderr, "[policy-pipe] EOF on stdin, stopping.\n");
                    break;
                }
            } else {
                build_human_actions(&v);
                used_human_actions = 1;
            }

            fc_actor_animation_capture_tick_start(&v.actor_animation,
                                                  &v.state);

            /* Step simulation */
            fc_step(&v.state, v.actions);
            if (used_human_actions && v.actions[5] > 0 && v.actions[6] > 0)
                fc_click_feedback_accept_move_tick(&v.click_feedback,
                                                   &v.state);
            fc_click_feedback_sync(&v.click_feedback, &v.state);

            /* Playable-viewer test aid only. The simulator has already
             * resolved the hit; keep the local session alive at one HP. */
            if (v.godmode &&
                v.state.terminal == TERMINAL_PLAYER_DEATH) {
                v.state.player.current_hp = 10;
                v.state.terminal = TERMINAL_NONE;
            }
            fc_fill_render_events(&v.state, &v.render_events);
            fc_actor_animation_ingest_tick(&v.actor_animation, &v.state,
                                           &v.render_events);
            update_reward_breakdown(&v);
            fc_actor_animation_ingest_events(
                &v.actor_animation, &v.render_events, v.anim_cache,
                v.active_loadout, v.tps);

            /* Debug event log — record events from this tick */
            dbg_log_tick(&v.state);

            /* Snap prev positions for newly spawned NPCs so they don't fly.
             * An NPC that wasn't active last tick but is now = new spawn. */
            for (int ni = 0; ni < FC_MAX_NPCS; ni++) {
                if (v.state.npcs[ni].active &&
                    !fc_actor_animation_previous_npc_active(
                        &v.actor_animation, ni)) {
                    fc_combat_presentation_clear_npc_healthbar(
                        v.combat_presentation, ni);
                }
            }

            fc_fill_render_entities(&v.state, v.entities, &v.entity_count);
            v.last_hash = fc_state_hash(&v.state);

            FcCombatPresentationContext combat_context = {
                .state = &v.state,
                .events = &v.render_events,
                .scene = &v.actor_animation.scene,
                .terrain = v.terrain,
                .anim_cache = v.anim_cache,
                .player_profile = fc_player_visual_profile(v.active_loadout),
                .tps = v.tps,
            };
            fc_combat_presentation_ingest_tick(v.combat_presentation,
                                                &combat_context);
            /* Sync viewer attack_target with player's backend target */
            v.attack_target = v.state.player.attack_target_idx;
            /* Auto-clear if target NPC died */
            if (v.state.player.attack_target_idx >= 0) {
                FcNpc* tn = &v.state.npcs[v.state.player.attack_target_idx];
                if (!tn->active || tn->is_dead) {
                    v.attack_target = -1;
                }
            }

            if (v.state.terminal != TERMINAL_NONE) {
                if (v.policy_pipe) {
                    print_policy_episode_summary(&v);
                    v.policy_episode_count++;
                    /* Write terminal obs, then auto-reset unless a fixed episode limit was requested. */
                    write_obs_to_pipe(&v);
                    if (v.policy_episode_limit > 0 &&
                        v.policy_episode_count >= v.policy_episode_limit) {
                        quit_after_tick = 1;
                    } else {
                        reset_ep(&v);
                    }
                } else {
                    v.paused = 1;
                }
            } else if (v.policy_pipe) {
                write_obs_to_pipe(&v);
            }
        }

        if (quit_after_tick) {
            fprintf(stderr, "[policy-pipe] Episode limit reached, exiting viewer.\n");
            break;
        }

        float frame_dt = GetFrameTime();
        fc_click_feedback_update(&v.click_feedback, frame_dt);
        FcCombatPresentationContext combat_context = {
            .state = &v.state,
            .events = &v.render_events,
            .scene = &v.actor_animation.scene,
            .terrain = v.terrain,
            .anim_cache = v.anim_cache,
            .player_profile = fc_player_visual_profile(v.active_loadout),
            .tps = v.tps,
        };
        unsigned char deferred_deaths[FC_MAX_NPCS];
        fc_combat_presentation_deferred_deaths(
            v.combat_presentation, &v.state, deferred_deaths);
        fc_actor_animation_update_scene(
            &v.actor_animation, &v.state, v.anim_cache, v.tps, frame_dt,
            !v.paused || v.policy_pipe, deferred_deaths);
        if (v.objects)
            fc_animated_atlas_update(&v.objects->atlas, frame_dt);
        fc_combat_presentation_update(v.combat_presentation,
                                      &combat_context, frame_dt);
        fc_combat_presentation_deferred_deaths(
            v.combat_presentation, &v.state, deferred_deaths);
        for (int i = 0; i < FC_MAX_NPCS; i++) {
            if (!v.state.npcs[i].active && !v.state.npcs[i].died_this_tick)
                fc_combat_presentation_clear_npc_healthbar(
                    v.combat_presentation, i);
        }
        fc_actor_animation_update_models(
            &v.actor_animation, &v.state, v.player_model, v.npc_models,
            v.anim_cache, v.active_loadout, v.tps, frame_dt, deferred_deaths);
        /* Draw */
        BeginDrawing();
        ClearBackground(COL_BG);
        draw_scene(&v);
        sync_fc_ui(&v);
        runec_ui_draw(&v.ui, GetScreenWidth(), GetScreenHeight());
        draw_runec_side_overrides(&v);
        draw_runec_console(&v);
        draw_click_cross(&v);

        EndDrawing();
    }

cleanup:
    if (v.pray_melee_tex.id > 0) UnloadTexture(v.pray_melee_tex);
    if (v.pray_missiles_tex.id > 0) UnloadTexture(v.pray_missiles_tex);
    if (v.pray_magic_tex.id > 0) UnloadTexture(v.pray_magic_tex);
    for (int i = 0; i < FC_CLICK_CROSS_FRAME_COUNT * 2; i++) {
        if (v.click_cross_tex[i].id > 0)
            UnloadTexture(v.click_cross_tex[i]);
    }
    if (v.tex_pray_melee_on.id > 0) UnloadTexture(v.tex_pray_melee_on);
    if (v.tex_pray_melee_off.id > 0) UnloadTexture(v.tex_pray_melee_off);
    if (v.tex_pray_range_on.id > 0) UnloadTexture(v.tex_pray_range_on);
    if (v.tex_pray_range_off.id > 0) UnloadTexture(v.tex_pray_range_off);
    if (v.tex_pray_magic_on.id > 0) UnloadTexture(v.tex_pray_magic_on);
    if (v.tex_pray_magic_off.id > 0) UnloadTexture(v.tex_pray_magic_off);
    fc_combat_presentation_destroy(v.combat_presentation);
    fc_actor_animation_shutdown(&v.actor_animation);
    if (v.object_anim_runtimes) {
        for (int i = 0; i < v.object_anim_runtime_count; i++) {
            if (v.object_anim_runtimes[i].anim_state)
                anim_model_state_free(v.object_anim_runtimes[i].anim_state);
        }
        free(v.object_anim_runtimes);
    }
    if (v.anim_cache) anim_cache_free(v.anim_cache);
    if (v.player_model) fc_npc_models_unload(v.player_model);
    if (v.npc_models) fc_npc_models_unload(v.npc_models);
    if (v.object_anim_models) fc_npc_models_unload(v.object_anim_models);
    fc_animated_atlas_unload(&v.shared_model_atlas);
    if (v.object_anims) object_anims_free(v.object_anims);
    objects_free(v.objects);
    fc_minimap_scene_free(&v.minimap_scene);
    terrain_free(v.terrain);
    fc_osrs_text_shutdown();
    runec_ui_shutdown(&v.ui);
    CloseWindow();
    return exit_code;
}
