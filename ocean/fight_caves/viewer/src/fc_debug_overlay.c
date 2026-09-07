/*
 * fc_debug_overlay.c — viewer debug tooling and overlays.
 *
 * Separate from viewer.c to isolate debug visualization from core viewer code.
 * All functions are read-only — they never modify FcState or ViewerState.
 *
 * Two entry points called from viewer.c:
 *   debug_overlay_3d()  — called inside BeginMode3D/EndMode3D (tiles, rays, ranges)
 *   debug_overlay_screen() — called after EndMode3D (screen-space overlays)
 *
 * Toggle with 'O' key (debug overlay master toggle).
 * Sub-toggles cycle with Shift+O or number keys in debug mode.
 */

#include "fc_debug_overlay.h"
#include "raylib.h"
#include "rlgl.h"
#include "fc_types.h"
#include "fc_contracts.h"
#include "fc_api.h"
#include "fc_pathfinding.h"
#include "fc_reward.h"
#include "fc_combat.h"
#include "fc_npc.h"
#include "fc_osrs_text.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdarg.h>

/* ======================================================================== */
/* Event log ring buffer                                                    */
/* ======================================================================== */

#define DBG_LOG_MAX_ENTRIES  128
#define DBG_LOG_MAX_MSG      80

typedef struct {
    char entries[DBG_LOG_MAX_ENTRIES][DBG_LOG_MAX_MSG];
    int tick[DBG_LOG_MAX_ENTRIES];      /* game tick when event occurred */
    Color color[DBG_LOG_MAX_ENTRIES];   /* display color per entry */
    int head;                           /* next write position */
    int count;                          /* total entries (capped at MAX) */
    int scroll_offset;                  /* for scrollable display */
} DbgEventLog;

static DbgEventLog g_dbg_log = {0};

void dbg_log_clear(void) {
    g_dbg_log.head = 0;
    g_dbg_log.count = 0;
    g_dbg_log.scroll_offset = 0;
}

static void dbg_log_event(int game_tick, Color c, const char* fmt, ...) {
    int idx = g_dbg_log.head;
    g_dbg_log.tick[idx] = game_tick;
    g_dbg_log.color[idx] = c;

    va_list args;
    va_start(args, fmt);
    vsnprintf(g_dbg_log.entries[idx], DBG_LOG_MAX_MSG, fmt, args);
    va_end(args);

    g_dbg_log.head = (g_dbg_log.head + 1) % DBG_LOG_MAX_ENTRIES;
    if (g_dbg_log.count < DBG_LOG_MAX_ENTRIES) g_dbg_log.count++;
}

/* Call once per tick to auto-generate events from state changes.
 * Reads current state and emits relevant log entries. */
void dbg_log_tick(const FcState* state) {
    int t = state->tick;
    const FcPlayer* p = &state->player;
    static const char* npc_names[] = {"?","Tz-Kih","Tz-Kek","Kek-Sm","Tok-Xil",
        "MejKot","Ket-Zek","Jad","HurKot"};
    float rwd[FC_REWARD_FEATURES];

    fc_write_reward_features(state, rwd);

    /* Player took damage */
    if (p->damage_taken_this_tick > 0) {
        dbg_log_event(t, CLITERAL(Color){255,120,120,255},
                      "Player took %d damage", p->damage_taken_this_tick / 10);
    }
    if (state->tz_kih_prayer_drain_this_tick > 0) {
        dbg_log_event(t, CLITERAL(Color){100,190,255,255},
                      "Tz-Kih drained %d Prayer",
                      (state->tz_kih_prayer_drain_this_tick + 9) / 10);
    }

    /* Player ate food */
    if (p->food_eaten_this_tick) {
        dbg_log_event(t, CLITERAL(Color){180,255,180,255},
                      "Ate shark (+20 HP) [%d left]", p->sharks_remaining);
    }

    /* Player drank potion */
    if (p->potion_used_this_tick) {
        dbg_log_event(t, CLITERAL(Color){120,180,255,255},
                      "Drank ppot (+17 pray) [%d doses left]", p->prayer_doses_remaining);
    }

    /* Player switched prayer */
    if (p->prayer_changed_this_tick) {
        static const char* pray_names[] = {"OFF","Prot Melee","Prot Range","Prot Magic"};
        dbg_log_event(t, CLITERAL(Color){255,255,100,255},
                      "Prayer -> %s", pray_names[p->prayer]);
    }

    /* Player attack attempt */
    if (rwd[FC_RWD_ATTACK_ATTEMPT] > 0.0f && p->attack_target_idx >= 0) {
        const FcNpc* tgt = &state->npcs[p->attack_target_idx];
        const char* nm = (tgt->npc_type > 0 && tgt->npc_type < 9) ? npc_names[tgt->npc_type] : "?";
        dbg_log_event(t, CLITERAL(Color){200,200,200,255},
                      "Attacked %s [%d]", nm, p->attack_target_idx);
    }

    /* NPC events */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];

        /* NPC damage taken */
        if (n->damage_taken_this_tick > 0) {
            const char* nm = (n->npc_type > 0 && n->npc_type < 9) ? npc_names[n->npc_type] : "?";
            dbg_log_event(t, CLITERAL(Color){255,200,100,255},
                          "%s[%d] took %d dmg (%d HP left)",
                          nm, i, n->damage_taken_this_tick / 10, n->current_hp / 10);
        }
        if (n->healing_received_this_tick > 0) {
            const char* nm = (n->npc_type > 0 && n->npc_type < 9) ? npc_names[n->npc_type] : "?";
            dbg_log_event(t, CLITERAL(Color){120,255,140,255},
                          "%s[%d] healed %d HP",
                          nm, i, (n->healing_received_this_tick + 9) / 10);
        }

        /* NPC died */
        if (n->died_this_tick) {
            const char* nm = (n->npc_type > 0 && n->npc_type < 9) ? npc_names[n->npc_type] : "?";
            dbg_log_event(t, CLITERAL(Color){255,80,80,255},
                          "%s[%d] KILLED", nm, i);
        }

    }

    /* Movement — log new route destination (walk-to-tile or click-to-move) */
    {
        static int prev_route_dest_x = -1, prev_route_dest_y = -1;
        if (p->route_len > 0 && p->route_idx < p->route_len) {
            int dx = p->route_x[p->route_len - 1];
            int dy = p->route_y[p->route_len - 1];
            if (dx != prev_route_dest_x || dy != prev_route_dest_y) {
                dbg_log_event(t, CLITERAL(Color){180,220,255,255},
                              "Move to tile (%d,%d) [%d steps]", dx, dy, p->route_len - p->route_idx);
                prev_route_dest_x = dx;
                prev_route_dest_y = dy;
            }
        } else {
            prev_route_dest_x = -1;
            prev_route_dest_y = -1;
        }
    }

    /* Wave events */
    if (state->wave_just_cleared) {
        dbg_log_event(t, CLITERAL(Color){100,255,255,255},
                      "Wave %d CLEARED — advancing", state->current_wave - 1);
    }

    /* Jad killed */
    if (state->jad_killed) {
        dbg_log_event(t, CLITERAL(Color){255,215,0,255}, "*** JAD DEFEATED ***");
    }

    /* Player death */
    if (state->terminal == TERMINAL_PLAYER_DEATH && p->current_hp <= 0) {
        dbg_log_event(t, CLITERAL(Color){255,0,0,255}, "*** PLAYER DIED ***");
    }

    /* Cave complete */
    if (state->terminal == TERMINAL_CAVE_COMPLETE) {
        dbg_log_event(t, CLITERAL(Color){0,255,0,255}, "*** CAVE COMPLETE — FIRE CAPE ***");
    }
}

/* Colors */
#define DBG_COL_WALK     CLITERAL(Color){  30, 180,  30, 40 }
#define DBG_COL_BLOCK    CLITERAL(Color){ 200,  30,  30, 60 }
#define DBG_COL_LOS_OK   CLITERAL(Color){  30, 255,  30, 200 }
#define DBG_COL_LOS_FAIL CLITERAL(Color){ 255,  30,  30, 200 }
#define DBG_COL_PATH     CLITERAL(Color){ 255, 255,   0, 160 }
#define DBG_COL_RANGE    CLITERAL(Color){ 100, 200, 255, 120 }
#define DBG_COL_LABEL    CLITERAL(Color){ 200, 200, 200, 255 }
#define DBG_COL_VALUE    CLITERAL(Color){ 255, 255, 100, 255 }
#define DBG_COL_GOOD     CLITERAL(Color){ 100, 255, 100, 255 }
#define DBG_COL_BAD      CLITERAL(Color){ 255, 100, 100, 255 }
#define DBG_COL_DIM      CLITERAL(Color){ 120, 120, 120, 255 }

static const char* dbg_basename(const char* path) {
    const char* slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static Color dbg_reward_color(float value) {
    if (value > 0.0001f) return DBG_COL_GOOD;
    if (value < -0.0001f) return DBG_COL_BAD;
    return DBG_COL_DIM;
}

/* ======================================================================== */
/* A. Collision / LOS / Path / Range overlays (3D)                          */
/* ======================================================================== */

/* Draw walkable/blocked tile overlay */
static void dbg_draw_collision(const FcState* state) {
    for (int tx = 0; tx < FC_ARENA_WIDTH; tx++) {
        for (int ty = 0; ty < FC_ARENA_HEIGHT; ty++) {
            Color c = state->walkable[tx][ty] ? DBG_COL_WALK : DBG_COL_BLOCK;
            DrawCube((Vector3){tx + 0.5f, 0.02f, -(ty + 0.5f)}, 0.9f, 0.02f, 0.9f, c);
        }
    }
}

/* Master 3D overlay — called inside BeginMode3D/EndMode3D.
 * Only collision tiles use 3D (they work with depth test disabled). */
void debug_overlay_3d(const FcState* state, int dbg_flags) {
    rlDisableDepthTest();
    if (dbg_flags & DBG_COLLISION) dbg_draw_collision(state);
    rlEnableDepthTest();
}

/* ======================================================================== */
/* 2D screen-space overlays (LOS, path, range) — drawn after EndMode3D     */
/* Uses GetWorldToScreen() projection so nothing can occlude them.          */
/* ======================================================================== */

static Vector2 dbg_tile_to_screen(int tx, int ty, Camera3D cam) {
    Vector3 w = { tx + 0.5f, 0.5f, -(ty + 0.5f) };
    return GetWorldToScreen(w, cam);
}

/* LOS rays — 2D projected lines from player to NPCs */
static void dbg_draw_los_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    Vector2 ps = dbg_tile_to_screen(p->x, p->y, cam);

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];
        if (!n->active || n->is_dead) continue;

        int ncx = n->x + n->size / 2;
        int ncy = n->y + n->size / 2;
        Vector2 ns = dbg_tile_to_screen(ncx, ncy, cam);

        int has_los = fc_has_los_between_areas(
            p->x, p->y, 1, n->x, n->y, n->size, state->los_flags);
        Color c = has_los ? DBG_COL_LOS_OK : DBG_COL_LOS_FAIL;
        DrawLineEx(ps, ns, 3.0f, c);
        DrawCircleV(ns, 6.0f, c);
    }
}

/* Path visualization — 2D projected dots and lines along route */
static void dbg_draw_path_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    if (p->route_idx >= p->route_len) return;

    Vector2 prev = {0};
    for (int i = p->route_idx; i < p->route_len; i++) {
        Vector2 s = dbg_tile_to_screen(p->route_x[i], p->route_y[i], cam);
        DrawCircleV(s, 4.0f, DBG_COL_PATH);
        if (i > p->route_idx) {
            DrawLineEx(prev, s, 2.0f, DBG_COL_PATH);
        }
        prev = s;
    }

    /* Destination marker — larger circle */
    if (p->route_len > 0) {
        Vector2 dest = dbg_tile_to_screen(
            p->route_x[p->route_len - 1], p->route_y[p->route_len - 1], cam);
        DrawCircleV(dest, 8.0f, DBG_COL_PATH);
        DrawCircleLinesV(dest, 12.0f, DBG_COL_PATH);
    }
}

/* Attack range ring — 2D projected Chebyshev boundary */
static void dbg_draw_range_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    int range = 7;

    for (int dx = -range; dx <= range; dx++) {
        for (int dy = -range; dy <= range; dy++) {
            int dist = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);
            if (dist == range) {
                int tx = p->x + dx;
                int ty = p->y + dy;
                if (tx >= 0 && tx < FC_ARENA_WIDTH && ty >= 0 && ty < FC_ARENA_HEIGHT) {
                    Vector2 s = dbg_tile_to_screen(tx, ty, cam);
                    DrawCircleV(s, 3.0f, DBG_COL_RANGE);
                }
            }
        }
    }
}

static void dbg_draw_run_energy(const FcState* state) {
    int energy = state->player.run_energy;
    if (energy < 0) energy = 0;
    if (energy > FC_RUN_ENERGY_MAX) energy = FC_RUN_ENERGY_MAX;

    char text[32];
    snprintf(text, sizeof(text), "Run energy: %d.%02d%%",
             energy / 100, energy % 100);
    int width = fc_osrs_measure_text(text, 9);
    DrawRectangle(6, 6, width + 10, 18,
                  CLITERAL(Color){20, 18, 14, 210});
    DrawRectangleLines(6, 6, width + 10, 18,
                       CLITERAL(Color){120, 110, 90, 255});
    fc_osrs_draw_text(text, 11, 11, 9, DBG_COL_VALUE);
}

/* Master 2D overlays for LOS/path/range — called after EndMode3D */
void debug_overlay_screen(const FcState* state, Camera3D cam, int dbg_flags) {
    if (dbg_flags & DBG_LOS)   dbg_draw_los_2d(state, cam);
    if (dbg_flags & DBG_PATH)  dbg_draw_path_2d(state, cam);
    if (dbg_flags & DBG_RANGE) dbg_draw_range_2d(state, cam);
    dbg_draw_run_energy(state);
}

void dbg_draw_prayer_window_indicator(Vector3 world_anchor, Camera3D cam) {
    Vector2 screen = GetWorldToScreen(world_anchor, cam);
    /* Keep the marker visibly separate from the centered overhead HP bar. */
    screen.x += 24.0f;
    screen.y -= 12.0f;
    int screen_width = GetScreenWidth();
    int screen_height = GetScreenHeight();
    if (screen.x < -20.0f || screen.x > (float)screen_width + 20.0f ||
        screen.y < -20.0f || screen.y > (float)screen_height + 20.0f) {
        return;
    }

    DrawCircleV(screen, 15.0f, CLITERAL(Color){70, 255, 100, 70});
    DrawCircleV(screen, 9.0f, CLITERAL(Color){70, 255, 100, 250});
    DrawCircleLines((int)screen.x, (int)screen.y, 10.0f, WHITE);
}

/* ======================================================================== */
/* Compact debug tabs (usable in any viewer panel)                          */
/* ======================================================================== */

/* Draw debug info as a tabbed panel section.
 * px = panel X, x = content X, by = Y start position, pw = panel width.
 * dbg_tab: 0=player, 1=obs, 2=mask, 3=reward, 4=log.
 * draw_tabs controls whether this helper renders its own tab selector.
 * content_height limits the log viewport; zero keeps the legacy 20 rows.
 * Returns end Y position. */
int dbg_draw_panel_tabs(const FcState* state,
                        const FcRewardBreakdown* reward_breakdown,
                        const FcRewardRuntime* reward_runtime,
                        int reward_config_loaded,
                        const char* reward_config_path,
                        int px, int x, int by, int pw, int dbg_tab,
                        int draw_tabs, int content_height) {
    char buf[256];
    int lh = 14;

    if (draw_tabs) {
        DrawLine(px+4, by, px+pw-4, by,
                 CLITERAL(Color){42,36,28,255});
        by += 2;
        static const char* dtab_labels[] = {
            "Player", "Obs", "Mask", "Reward", "Log"
        };
        int num_dtabs = 5;
        int dtab_w = (pw - 12) / num_dtabs;
        int dtab_h = 16;
        for (int t = 0; t < num_dtabs; t++) {
            int tx = px + 4 + t * dtab_w;
            int selected = (t == dbg_tab);
            Color bg = selected ? CLITERAL(Color){82,73,61,255}
                                : CLITERAL(Color){42,36,28,255};
            DrawRectangle(tx, by, dtab_w, dtab_h, bg);
            Color tc = selected ? DBG_COL_VALUE : DBG_COL_DIM;
            int tw = fc_osrs_measure_text(dtab_labels[t], 8);
            fc_osrs_draw_text(dtab_labels[t], tx + (dtab_w - tw) / 2,
                     by + 4, 8, tc);
            if (selected)
                DrawLine(tx+2, by+dtab_h-1, tx+dtab_w-2,
                         by+dtab_h-1, DBG_COL_VALUE);
        }
        by += dtab_h + 3;
    }

    /* Tab content */
    if (dbg_tab == 0) {
        /* Player state */
        const FcPlayer* p = &state->player;
        static const char* pray_str[] = { "OFF", "Melee", "Range", "Magic" };

        snprintf(buf, sizeof(buf), "Pos:(%d,%d) Face:%.0f", p->x, p->y, p->facing_angle);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "HP:%d/%d Pray:%d/%d",
                 p->current_hp/10, p->max_hp/10, p->current_prayer/10, p->max_prayer/10);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Tmr atk:%d fd:%d pot:%d",
                 p->attack_timer, p->food_timer, p->potion_timer);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Prayer:%s Drain:%d Bonus:+%d",
                 pray_str[p->prayer], p->prayer_drain_counter, p->prayer_bonus);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Shark:%d Dose:%d Ammo:%d",
                 p->sharks_remaining, p->prayer_doses_remaining, p->ammo_count);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Tgt:%d Appr:%d Route:%d/%d",
                 p->attack_target_idx, p->approach_target, p->route_idx, p->route_len);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Hits:%d Run:%d Regen:%d",
                 p->num_pending_hits, p->is_running, p->hp_regen_counter);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

        /* Active NPCs compact list */
        by += 2;
        static const char* npc_names[] = {"?","Kih","Kek","KSm","Xil","Mej","Zek","Jad","Hur"};
        for (int i = 0; i < FC_MAX_NPCS; i++) {
            const FcNpc* n = &state->npcs[i];
            if (!n->active) continue;
            const char* nm = (n->npc_type>0 && n->npc_type<9) ? npc_names[n->npc_type] : "?";
            snprintf(buf, sizeof(buf), "[%d]%s hp:%d atk:%d/%d",
                     i, nm, n->current_hp/10, n->attack_timer, n->attack_speed);
            Color c = n->is_dead ? DBG_COL_BAD : DBG_COL_DIM;
            fc_osrs_draw_text(buf, x, by, 7, c); by += lh - 2;
        }

    } else if (dbg_tab == 1) {
        /* Observation values */
        float obs[FC_OBS_SIZE];
        fc_write_obs(state, obs);
        int mbase = FC_OBS_META_START;

        snprintf(buf, sizeof(buf), "HP:%.2f Pray:%.2f X:%.2f Y:%.2f",
                 obs[FC_OBS_PLAYER_HP], obs[FC_OBS_PLAYER_PRAYER],
                 obs[FC_OBS_PLAYER_X], obs[FC_OBS_PLAYER_Y]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Atk:%.2f Run:%.2f Sharks:%.2f Dose:%.2f",
                 obs[FC_OBS_PLAYER_ATK_TIMER],
                 obs[FC_OBS_PLAYER_RUN_ENERGY],
                 obs[FC_OBS_PLAYER_SHARKS],
                 obs[FC_OBS_PLAYER_DOSES]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Pray M%.0f R%.0f G%.0f T%.2f",
                 obs[FC_OBS_PLAYER_PRAY_MEL],
                 obs[FC_OBS_PLAYER_PRAY_RNG],
                 obs[FC_OBS_PLAYER_PRAY_MAG],
                 obs[FC_OBS_PLAYER_TARGET]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "PrayDDL M%.2f R%.2f G%.2f",
                 obs[FC_OBS_PLAYER_PRAY_DDL_MEL],
                 obs[FC_OBS_PLAYER_PRAY_DDL_RNG],
                 obs[FC_OBS_PLAYER_PRAY_DDL_MAG]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "In1 M%.2f R%.2f G%.2f",
                 obs[FC_OBS_PLAYER_IN_MEL_1T],
                 obs[FC_OBS_PLAYER_IN_RNG_1T],
                 obs[FC_OBS_PLAYER_IN_MAG_1T]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "In2 M%.2f R%.2f G%.2f",
                 obs[FC_OBS_PLAYER_IN_MEL_2T],
                 obs[FC_OBS_PLAYER_IN_RNG_2T],
                 obs[FC_OBS_PLAYER_IN_MAG_2T]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Meta W%.2f Rot%.2f Rem%.2f",
                 obs[mbase + FC_OBS_META_WAVE],
                 obs[mbase + FC_OBS_META_ROTATION],
                 obs[mbase + FC_OBS_META_REMAINING]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Drain:%.2f Dmg:%.2f Clear:%.0f",
                 obs[mbase + FC_OBS_META_PRAY_DRAIN],
                 obs[mbase + FC_OBS_META_DMG_T_TICK],
                 obs[mbase + FC_OBS_META_WAVE_CLR]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "In3 M%.2f R%.2f G%.2f",
                 obs[mbase + FC_OBS_META_IN_MEL_3T],
                 obs[mbase + FC_OBS_META_IN_RNG_3T],
                 obs[mbase + FC_OBS_META_IN_MAG_3T]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;
        snprintf(buf, sizeof(buf), "Prog C%.2f W%.2f Rem%.2f NoP%.2f",
                 obs[mbase + FC_OBS_META_CAVE_PROG],
                 obs[mbase + FC_OBS_META_WAVE_PROG],
                 obs[mbase + FC_OBS_META_WORK_REM],
                 obs[mbase + FC_OBS_META_NO_PROG]);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

        by += 2;
        fc_osrs_draw_text("NPC slots:", x, by, 7, DBG_COL_DIM); by += lh - 1;
        for (int s = 0; s < FC_OBS_NPC_SLOTS; s++) {
            int base = FC_OBS_NPC_START + s * FC_OBS_NPC_STRIDE;
            if (obs[base + FC_NPC_VALID] < 0.5f) continue;
            char tele = '-';
            if (obs[base + FC_NPC_TELE_MELEE]  > 0.5f) tele = 'M';
            else if (obs[base + FC_NPC_TELE_RANGED] > 0.5f) tele = 'R';
            else if (obs[base + FC_NPC_TELE_MAGIC]  > 0.5f) tele = 'A';
            snprintf(buf, sizeof(buf), "[%d] hp:%.2f d:%.2f tele:%c los:%.0f pd:%.2f ddl:%.0f/%.2f",
                     s,
                     obs[base + FC_NPC_HP],
                     obs[base + FC_NPC_DISTANCE],
                     tele,
                     obs[base + FC_NPC_LOS],
                     obs[base + FC_NPC_PENDING_STYLE],
                     obs[base + FC_NPC_PENDING_PRAYER_WINDOW],
                     obs[base + FC_NPC_PENDING_PRAYER_DEADLINE]);
            fc_osrs_draw_text(buf, x, by, 7, DBG_COL_LABEL); by += lh - 2;
        }

    } else if (dbg_tab == 2) {
        /* Action mask */
        float mask[FC_ACTION_MASK_SIZE];
        fc_write_mask(state, mask);

        fc_osrs_draw_text("MOVE:", x, by, 8, DBG_COL_DIM);
        snprintf(buf, sizeof(buf), " ");
        int len = 1;
        for (int m = 0; m < FC_MOVE_DIM && len < 120; m++)
            buf[len++] = mask[FC_MASK_MOVE_START + m] > 0.5f ? '1' : '0';
        buf[len] = '\0';
        fc_osrs_draw_text(buf, x + 30, by, 8, DBG_COL_LABEL); by += lh;

        fc_osrs_draw_text("ATK:", x, by, 8, DBG_COL_DIM);
        len = 1; buf[0] = ' ';
        for (int m = 0; m < FC_ATTACK_DIM && len < 120; m++)
            buf[len++] = mask[FC_MASK_ATTACK_START + m] > 0.5f ? '1' : '0';
        buf[len] = '\0';
        fc_osrs_draw_text(buf, x + 30, by, 8, DBG_COL_LABEL); by += lh;

        fc_osrs_draw_text("PRAY:", x, by, 8, DBG_COL_DIM);
        len = 1; buf[0] = ' ';
        for (int m = 0; m < FC_PRAYER_DIM && len < 120; m++)
            buf[len++] = mask[FC_MASK_PRAYER_START + m] > 0.5f ? '1' : '0';
        buf[len] = '\0';
        fc_osrs_draw_text(buf, x + 30, by, 8, DBG_COL_LABEL); by += lh;

        snprintf(buf, sizeof(buf), "EAT:%c%c%c  DRINK:%c%c",
                 mask[FC_MASK_EAT_START+0]>0.5f?'1':'0',
                 mask[FC_MASK_EAT_START+1]>0.5f?'1':'0',
                 mask[FC_MASK_EAT_START+2]>0.5f?'1':'0',
                 mask[FC_MASK_DRINK_START+0]>0.5f?'1':'0',
                 mask[FC_MASK_DRINK_START+1]>0.5f?'1':'0');
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

        int vx = 0, vy = 0;
        for (int m = 0; m < FC_MOVE_TARGET_X_DIM; m++)
            if (mask[FC_MASK_TARGET_X_START + m] > 0.5f) vx++;
        for (int m = 0; m < FC_MOVE_TARGET_Y_DIM; m++)
            if (mask[FC_MASK_TARGET_Y_START + m] > 0.5f) vy++;
        snprintf(buf, sizeof(buf), "TGT_X:%d/65 TGT_Y:%d/65", vx, vy);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += lh;

    } else if (dbg_tab == 3) {
        const FcRewardBreakdown* b = reward_breakdown;
        int sh = 14;
        const char* cfg_name = dbg_basename(reward_config_path);

        fc_osrs_draw_text("Training reward parity", x, by, 8, DBG_COL_VALUE); by += sh + 2;
        snprintf(buf, sizeof(buf), "cfg:%s%s",
                 cfg_name,
                 reward_config_loaded ? "" : " (defaults)");
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += sh;
        snprintf(buf, sizeof(buf), "total:%+.4f noatk_t:%d",
                 b->total,
                 reward_runtime->ticks_since_attack);
        fc_osrs_draw_text(buf, x, by, 8, dbg_reward_color(b->total)); by += sh;
        snprintf(buf, sizeof(buf), "threat any:%d melee:%d",
                 b->threat_ctx.any_threat,
                 b->threat_ctx.melee_pressure_npcs);
        fc_osrs_draw_text(buf, x, by, 8, DBG_COL_LABEL); by += sh + 2;

        {
            struct {
                const char* name;
                float value;
            } terms[] = {
                {"dmg_dealt", b->damage_dealt},
                {"progress", b->progress},
                {"dmg_taken", b->damage_taken},
                {"npc_kill", b->npc_kill},
                {"wave_clear", b->wave_clear},
                {"jad_kill", b->jad_kill},
                {"complete", b->cave_complete},
                {"death", b->player_death},
                {"jad_ok", b->correct_jad_prayer},
                {"danger_ok", b->correct_danger_prayer},
                {"pray_lost", b->prayer_lost},
                {"pray_waste", b->unnecessary_prayer},
                {"wave_stall", b->wave_stall},
                {"no_progress", b->no_progress},
                {"no_attack", b->no_attack},
                {"jad_heal", b->jad_heal},
                {"npc_heal", b->npc_heal},
                {"invalid", b->invalid_action},
                {"tick_pen", b->tick_penalty},
            };
            int term_count = (int)(sizeof(terms) / sizeof(terms[0]));

            for (int i = 0; i < term_count; i++) {
                snprintf(buf, sizeof(buf), "%-12s %+.4f",
                         terms[i].name, terms[i].value);
                fc_osrs_draw_text(buf, x, by, 7, dbg_reward_color(terms[i].value));
                by += sh;
            }
        }
    } else if (dbg_tab == 4) {
        /* Event log — scrollable with scrollbar, most recent at top */
        int entry_h = lh - 2;
        int max_visible = content_height > 0
            ? content_height / entry_h : 20;
        if (max_visible < 1) max_visible = 1;
        int total = g_dbg_log.count;
        int log_h = max_visible * entry_h;
        int scrollbar_w = 10;
        if (total == 0) {
            fc_osrs_draw_text("No events yet", x, by, 8, DBG_COL_DIM);
            by += lh;
        } else {
            int max_scroll = total - max_visible;
            if (max_scroll < 0) max_scroll = 0;

            /* Clamp scroll */
            if (g_dbg_log.scroll_offset > max_scroll)
                g_dbg_log.scroll_offset = max_scroll;
            if (g_dbg_log.scroll_offset < 0)
                g_dbg_log.scroll_offset = 0;

            /* Mouse wheel scroll */
            float wheel = GetMouseWheelMove();
            if (wheel != 0.0f) {
                Vector2 mp = GetMousePosition();
                if (mp.x >= px && mp.x < px + pw &&
                    mp.y >= by && mp.y < by + log_h) {
                    g_dbg_log.scroll_offset -= (int)wheel * 3;
                    if (g_dbg_log.scroll_offset < 0) g_dbg_log.scroll_offset = 0;
                    if (g_dbg_log.scroll_offset > max_scroll) g_dbg_log.scroll_offset = max_scroll;
                }
            }

            /* Scrollbar track */
            int sb_x = px + pw - scrollbar_w - 4;
            int sb_y = by;
            DrawRectangle(sb_x, sb_y, scrollbar_w, log_h, CLITERAL(Color){30,26,20,255});

            /* Scrollbar thumb */
            if (total > max_visible) {
                float thumb_frac = (float)max_visible / (float)total;
                int thumb_h = (int)(log_h * thumb_frac);
                if (thumb_h < 12) thumb_h = 12;
                float scroll_frac = (max_scroll > 0) ? (float)g_dbg_log.scroll_offset / (float)max_scroll : 0;
                int thumb_y = sb_y + (int)((log_h - thumb_h) * scroll_frac);
                DrawRectangle(sb_x, thumb_y, scrollbar_w, thumb_h, CLITERAL(Color){120,110,90,255});

                /* Drag scrollbar with mouse */
                if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                    Vector2 mp = GetMousePosition();
                    if (mp.x >= sb_x && mp.x < sb_x + scrollbar_w &&
                        mp.y >= sb_y && mp.y < sb_y + log_h) {
                        float click_frac = (mp.y - sb_y) / (float)log_h;
                        g_dbg_log.scroll_offset = (int)(click_frac * (float)(max_scroll + 1));
                        if (g_dbg_log.scroll_offset > max_scroll) g_dbg_log.scroll_offset = max_scroll;
                    }
                }
            }

            /* Draw entries */
            int drawn = 0;
            for (int i = g_dbg_log.scroll_offset; i < total && drawn < max_visible; i++) {
                int idx = (g_dbg_log.head - 1 - i + DBG_LOG_MAX_ENTRIES * 2) % DBG_LOG_MAX_ENTRIES;
                snprintf(buf, sizeof(buf), "t%d %s", g_dbg_log.tick[idx], g_dbg_log.entries[idx]);
                /* Truncate to fit content width */
                fc_osrs_draw_text(buf, x, by, 7, g_dbg_log.color[idx]);
                by += entry_h;
                drawn++;
            }

            /* Pad remaining space if fewer entries than max_visible */
            by += (max_visible - drawn) * entry_h;
        }
    }

    return by;
}
