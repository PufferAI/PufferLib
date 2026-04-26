/**
 * @fileoverview Raylib debug viewer for OSRS PvP simulation.
 *
 * Top-down 2D tile grid with full debug overlay: player state, HP bars,
 * prayer icons, gear labels, hit splats, collision map visualization.
 * Included conditionally via OSRS_VISUAL define.
 *
 * Follows PufferLib's Client + make_client + c_render pattern.
 */

#ifndef OSRS_RENDER_H
#define OSRS_RENDER_H

#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"
#include "osrs_models.h"
#include "osrs_anim.h"
#include "osrs_combat.h"
#include "osrs_pvp_combat.h"
#include "osrs_pvp_effects.h"
#include "data/player_models.h"
#include "data/npc_models.h"
#include "osrs_terrain.h"
#include "osrs_objects.h"
#include "osrs_gui.h"
#include "osrs_human_input.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ======================================================================== */
/* constants                                                                 */
/* ======================================================================== */

#define RENDER_TILE_SIZE       20
/* window sized at 1.5x the OSRS fixed-client layout (765x503 → 1148x755)
   so it's comfortable to look at on modern monitors while keeping the
   same aspect ratio as the real game. */
#define RENDER_WINDOW_W        1148
#define RENDER_WINDOW_H        755
#define RENDER_PANEL_WIDTH     285   /* 190 * 1.5 */
#define RENDER_HEADER_HEIGHT   0     /* OSRS client has no top header strip */
#define RENDER_SPLATS_PER_PLAYER 4   /* OSRS max: 4 simultaneous splats per entity */
#define RENDER_HISTORY_SIZE    2000  /* max ticks of rewind history */
#define MAX_RENDER_ENTITIES    64    /* max entities rendered (players + NPCs/bosses/adds) */

#define RENDER_GRID_W (RENDER_WINDOW_W - RENDER_PANEL_WIDTH)  /* = 575, OSRS ≈ 512 */
#define RENDER_GRID_H (RENDER_WINDOW_H - RENDER_HEADER_HEIGHT)  /* = 503, OSRS ≈ 334 */

/* colors */
#define COLOR_BG          CLITERAL(Color){ 20, 20, 25, 255 }
#define COLOR_GRID        CLITERAL(Color){ 45, 45, 55, 255 }
#define COLOR_HEADER_BG   CLITERAL(Color){ 30, 30, 40, 255 }
#define COLOR_PANEL_BG    CLITERAL(Color){ 25, 25, 35, 255 }
#define COLOR_P0          CLITERAL(Color){ 80, 140, 255, 255 }
#define COLOR_P1          CLITERAL(Color){ 255, 90, 90, 255 }
#define COLOR_P0_LIGHT    CLITERAL(Color){ 80, 140, 255, 60 }
#define COLOR_P1_LIGHT    CLITERAL(Color){ 255, 90, 90, 60 }
#define COLOR_FREEZE      CLITERAL(Color){ 100, 170, 255, 90 }
#define COLOR_VENG        CLITERAL(Color){ 255, 220, 50, 255 }
#define COLOR_BLOCKED     CLITERAL(Color){ 200, 50, 50, 50 }
#define COLOR_WALL        CLITERAL(Color){ 220, 150, 40, 50 }
#define COLOR_BRIDGE      CLITERAL(Color){ 50, 120, 220, 50 }
#define COLOR_WALL_LINE   CLITERAL(Color){ 255, 180, 50, 180 }
#define COLOR_HP_GREEN    CLITERAL(Color){ 50, 200, 50, 255 }
#define COLOR_HP_RED      CLITERAL(Color){ 200, 50, 50, 255 }
#define COLOR_HP_BG       CLITERAL(Color){ 40, 40, 40, 200 }
#define COLOR_SPEC_BAR    CLITERAL(Color){ 230, 170, 30, 255 }
#define COLOR_TEXT         CLITERAL(Color){ 200, 200, 210, 255 }
#define COLOR_TEXT_DIM     CLITERAL(Color){ 130, 130, 140, 255 }
#define COLOR_LABEL        CLITERAL(Color){ 170, 170, 180, 255 }

/* ======================================================================== */
/* active projectile flights (sub-tick interpolation at 50 Hz)               */
/* ======================================================================== */

/* OSRS projectile flight parameters (from deob client Projectile.java):
 *   x/y: linear interpolation from source to target
 *   height: parabolic arc — initial slope from 'curve' param,
 *           quadratic correction to hit end_height exactly.
 *   Zulrah attacks: delay=1, duration=35 client ticks, startH=85, endH=40,
 *                   curve=16 (~22.5 degree launch angle).
 *   1 client tick = 20ms, 1 server tick = 600ms = 30 client ticks.
 */

/* inferno can keep multiple Zuk healer spark volleys and other flights alive
   at once. size this from the real encounter envelope instead of a tiny demo
   value so visuals never silently disappear. */
#define MAX_FLIGHT_PROJECTILES 64
#define PROJ_OSRS_SLOPE_TO_RAD 0.02454369f  /* pi/128, converts OSRS slope units to radians */

typedef struct {
    int active;
    float src_x, src_y;         /* source tile position */
    float dst_x, dst_y;         /* target tile position (updated each tick if tracking) */
    float x, y;                 /* current interpolated position */
    float progress;             /* 0.0 (spawned) → 1.0 (arrived) */
    float speed;                /* progress per client tick (1.0/duration) */
    float start_height;         /* height at source (tiles above ground) */
    float end_height;           /* height at target (tiles above ground) */
    float curve;                /* OSRS slope param (16 = ~22.5 degrees) */
    int style;                  /* 0=ranged, 1=magic, 2=melee, 3=cloud */
    int damage;                 /* hit splat value at arrival */

    /* OSRS tracking: projectiles re-aim toward target each sub-tick */
    float vel_x, vel_y;         /* current horizontal velocity (tiles per progress unit) */
    float height_vel;           /* current vertical velocity */
    float height_accel;         /* quadratic height correction */
    float yaw;                  /* current facing direction (radians) */
    float pitch;                /* current vertical tilt (radians) */
    float arc_height;           /* sinusoidal arc peak in tiles (0 = use quadratic) */
    int tracks_target;          /* 1 = re-aim toward target each tick */
    int start_delay;            /* client ticks before projectile becomes visible/moves */
    int motion_mode;            /* EncounterProjectileMotionMode */
    float offset_x, offset_y, offset_z;
    uint32_t model_id;          /* GFX model from cache (0 = style-based fallback) */
    int anim_id;                /* spotanim animation sequence (-1 = static model) */
    int anim_frame;
    int anim_tick_counter;
    AnimModelState* anim_state;
    int impact_gfx_id;          /* landing spotanim to spawn on arrival */
} FlightProjectile;

/* ======================================================================== */
/* per-player composite model                                                */
/* ======================================================================== */

/* OSRS composites all body parts + equipment into a single merged model
 * before animating. this ensures vertex skin label groups span the full
 * body, so origin/pivot transforms compute correct centroids.
 * we replicate that here: one composite mesh per player. */

#define COMPOSITE_MAX_BASE_VERTS 12000  /* ~16 models * ~750 base verts each */
#define COMPOSITE_MAX_FACES      8000   /* ~16 models * ~500 faces each */
#define COMPOSITE_MAX_EXP_VERTS  (COMPOSITE_MAX_FACES * 3)

typedef struct {
    /* merged base geometry for animation */
    int16_t   base_vertices[COMPOSITE_MAX_BASE_VERTS * 3];
    uint8_t   vertex_skins[COMPOSITE_MAX_BASE_VERTS];
    uint16_t  face_indices[COMPOSITE_MAX_FACES * 3];
    uint8_t   face_pri_delta[COMPOSITE_MAX_FACES]; /* per-face priority delta relative to source model min */
    int       base_vert_count;
    int       face_count;

    /* raylib mesh (pre-allocated at max capacity, updated per frame) */
    Mesh  mesh;
    Model model;
    int   gpu_ready;

    /* animation working state (rebuilt on equipment change) */
    AnimModelState* anim_state;

    /* change detection: last-seen equipment (players) or NPC def ID (NPCs) */
    uint8_t last_equipped[NUM_GEAR_SLOTS];
    int     last_npc_def_id;
    int     needs_rebuild;
} PlayerComposite;

/* ======================================================================== */
/* convex hull click detection (ported from RuneLite Jarvis.java)            */
/* ======================================================================== */

#define HULL_MAX_POINTS 256  /* max hull vertices (models rarely exceed 100) */

typedef struct {
    int xs[HULL_MAX_POINTS];
    int ys[HULL_MAX_POINTS];
    int count;
} ConvexHull2D;

/** Jarvis march: compute 2D convex hull from screen-space points.
    xs/ys are input arrays of length n. out is populated with the hull.
    ported from RuneLite Jarvis.java. */
static void hull_compute(const int* xs, const int* ys, int n, ConvexHull2D* out) {
    out->count = 0;
    if (n < 3) return;

    /* find leftmost point */
    int left = 0;
    for (int i = 1; i < n; i++) {
        if (xs[i] < xs[left] || (xs[i] == xs[left] && ys[i] < ys[left]))
            left = i;
    }

    int current = left;
    do {
        int cx = xs[current], cy = ys[current];
        if (out->count >= HULL_MAX_POINTS) return;
        out->xs[out->count] = cx;
        out->ys[out->count] = cy;
        out->count++;

        /* safety: hull can't have more points than input */
        if (out->count > n) { out->count = 0; return; }

        int next = 0;
        int nx = xs[0], ny = ys[0];
        for (int i = 1; i < n; i++) {
            /* cross product: positive means i is to the left of current→next */
            long long cp = (long long)(ys[i] - cy) * (nx - xs[i])
                         - (long long)(xs[i] - cx) * (ny - ys[i]);
            if (cp > 0) {
                next = i; nx = xs[i]; ny = ys[i];
            } else if (cp == 0) {
                /* collinear: pick the farther point */
                long long d_i = (long long)(cx - xs[i]) * (cx - xs[i])
                              + (long long)(cy - ys[i]) * (cy - ys[i]);
                long long d_n = (long long)(cx - nx) * (cx - nx)
                              + (long long)(cy - ny) * (cy - ny);
                if (d_i > d_n) { next = i; nx = xs[i]; ny = ys[i]; }
            }
        }
        current = next;
    } while (current != left);
}

/** Point-in-polygon test (ray casting method).
    returns 1 if (px, py) is inside the convex hull. */
static int hull_contains(const ConvexHull2D* hull, int px, int py) {
    if (hull->count < 3) return 0;
    int inside = 0;
    for (int i = 0, j = hull->count - 1; i < hull->count; j = i++) {
        int xi = hull->xs[i], yi = hull->ys[i];
        int xj = hull->xs[j], yj = hull->ys[j];
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
            inside = !inside;
    }
    return inside;
}

/* ======================================================================== */
/* render client                                                             */
/* ======================================================================== */

/* per-entity hitsplat slot matching OSRS Entity.java exactly:
   - hitmarkMove starts at +5.0, decreases by 0.25/client-tick, clamps at -5.0
   - hitmarkTrans (opacity) starts at 230, stays there (mode 2 never fades)
   - hitsLoopCycle: expires after 70 client ticks
   - slot layout from Client.java:6052: slot 0=center, 1=up20, 2=left15+up10, 3=right15+up10 */
typedef struct {
    int active;
    int damage;
    double hitmark_move;   /* OSRS hitmarkMove: starts +5, decrements to -5 */
    int hitmark_trans;     /* OSRS hitmarkTrans: opacity 0-230, starts 230 */
    int ticks_remaining;   /* counts down from 70 client ticks */
} HitSplat;

/* ======================================================================== */
/* right-click context menu (OSRS-style)                                    */
/* ======================================================================== */

#define CONTEXT_MENU_MAX_ITEMS 8
#define CONTEXT_MENU_ROW_H     15
#define CONTEXT_MENU_PADDING    4
#define CONTEXT_MENU_MIN_W    150

typedef enum {
    CMENU_ACTION_NONE = 0,
    CMENU_ACTION_WALK_HERE,
    CMENU_ACTION_ATTACK,
    CMENU_ACTION_CANCEL,
} ContextMenuAction;

typedef struct {
    ContextMenuAction action;
    int entity_idx;         /* render entity index for ATTACK, -1 for walk/cancel */
    char label[64];         /* display text, e.g. "Attack Jal-Zek" */
} ContextMenuItem;

typedef struct {
    int visible;
    int screen_x, screen_y; /* top-left of the menu popup */
    int width;               /* computed from widest label */
    int item_count;
    ContextMenuItem items[CONTEXT_MENU_MAX_ITEMS];
    int walk_tile_x, walk_tile_y;  /* world tile for "Walk here" */
    int hover_idx;           /* item currently hovered, -1 = none */
} ContextMenu;

typedef struct {
    /* viewer state */
    int is_paused;
    float ticks_per_second;
    int step_once;
    int step_back;

    /* overlay toggles */
    int show_collision;
    int show_pathfinding;
    int show_models;
    int show_safe_spots;
    int show_debug;       /* toggle raycast debug, hulls, hitboxes, projectile trails */

    /* 3D model rendering */
    ModelCache* model_cache;
    AnimCache* anim_cache;
    ModelCache* npc_model_cache;  /* secondary cache for encounter-specific NPC models */
    AnimCache* npc_anim_cache;    /* secondary cache for encounter-specific NPC anims */
    float model_scale;

    /* overhead prayer icon textures (from headicons_prayer sprites) */
    Texture2D prayer_icons[6];  /* indexed by headIcon: 0=melee,1=ranged,2=magic,3=retri,4=smite,5=redemp */
    int prayer_icons_loaded;

    /* hitsplat sprite textures (from hitmarks sprites, 317 mode 0).
       0=blue(miss), 1=red(regular), 2=green(poison), 3=dark(venom), 4=yellow(shield) */
    Texture2D hitmark_sprites[5];
    int hitmark_sprites_loaded;

    /* click cross sprites: 4 yellow (move) + 4 red (attack) animation frames */
    Texture2D click_cross_sprites[8];
    int click_cross_loaded;

    /* debug: last raycast-selected tile (-1 = none) */
    int debug_hit_wx, debug_hit_wy;
    float debug_ray_hit_x, debug_ray_hit_y, debug_ray_hit_z;
    /* ray-plane comparison */
    int debug_plane_wx, debug_plane_wy;
    /* ray info */
    Vector3 debug_ray_origin, debug_ray_dir;

    /* render entities: populated per-frame from env->players or encounter vtable.
       index 0 = agent, 1+ = opponents/NPCs/bosses.
       stored by value (not pointer) via fill_render_entities. */
    RenderEntity entities[MAX_RENDER_ENTITIES];
    int entity_count;

    /* per-entity composite model (merged body + equipment, animated as one) */
    PlayerComposite composites[MAX_RENDER_ENTITIES];

    /* per-entity 2D convex hull for click detection (projected model vertices).
       recomputed every frame after 3D rendering, used by click handler. */
    ConvexHull2D entity_hulls[MAX_RENDER_ENTITIES];
    float entity_visual_top_y[MAX_RENDER_ENTITIES];  /* world-space top of animated mesh */
    float entity_visual_mid_y[MAX_RENDER_ENTITIES];  /* world-space middle of animated mesh */

    /* per-entity two-track animation (matches OSRS primary + secondary system) */
    struct {
        /* primary track: action animation (attack, cast, eat, block, death) */
        int primary_seq_id;       /* -1 = inactive */
        int primary_frame_idx;
        int primary_ticks;
        int primary_loops;        /* how many times the anim has looped */

        /* secondary track: pose animation (idle, walk, run — always active) */
        int secondary_seq_id;
        int secondary_frame_idx;
        int secondary_ticks;
    } anim[MAX_RENDER_ENTITIES];

    /* entity identity tracking — detect slot compaction shifts to reset stale anim/composite */
    int prev_npc_slot[MAX_RENDER_ENTITIES];
    int prev_entity_count;

    /* terrain */
    TerrainMesh* terrain;

    /* placed objects (walls, buildings, trees) */
    ObjectMesh* objects;
    ObjectMesh* objects_zuk;  /* post-Zuk variant (prison walls removed) */
    int zuk_active;           /* set when Zuk NPC (7706) is present */

    /* NPC models at spawn positions */
    ObjectMesh* npcs;

    /* 3D camera mode (T to toggle) */
    int mode_3d;
    float cam_yaw;      /* radians, 0 = looking north */
    float cam_pitch;     /* radians, clamped */
    float cam_dist;      /* distance from target */
    float cam_target_x;  /* world X (tile coords) */
    float cam_target_z;  /* world Z (tile coords) */

    /* camera zoom (scroll wheel zooms entire view) */
    float zoom;

    /* per-entity hit splats (4 slots each, OSRS style) */
    HitSplat splats[MAX_RENDER_ENTITIES][RENDER_SPLATS_PER_PLAYER];

    /* per-entity sub-tile position and facing (OSRS: 128 units per tile) */
    int sub_x[MAX_RENDER_ENTITIES], sub_y[MAX_RENDER_ENTITIES];
    int dest_x[MAX_RENDER_ENTITIES], dest_y[MAX_RENDER_ENTITIES];
    int visual_moving[MAX_RENDER_ENTITIES];
    int visual_running[MAX_RENDER_ENTITIES];
    int step_tracker[MAX_RENDER_ENTITIES];
    float yaw[MAX_RENDER_ENTITIES];
    float target_yaw[MAX_RENDER_ENTITIES];
    int facing_opponent[MAX_RENDER_ENTITIES];

    /* HP bar visibility timer: only shown after taking damage.
       matches OSRS Entity.cycleStatus (300 client ticks = 6s).
       in game ticks: set to env->tick + 10, visible while tick < this. */
    int hp_bar_visible_until[MAX_RENDER_ENTITIES];

    /* visual effects: spell impacts, projectiles */
    ActiveEffect effects[MAX_ACTIVE_EFFECTS];
    int effect_client_tick_counter;  /* monotonic 50 Hz counter for effect timing */

    /* client-tick accumulator: OSRS runs both movement AND animation at 50 Hz
       (20ms per client tick). we accumulate real time and process the correct
       number of steps per render frame, matching the real client exactly. */
    double client_tick_accumulator;

    /* arena bounds (overridden by encounter, defaults to FIGHT_AREA_*) */
    int arena_base_x, arena_base_y;
    int arena_width, arena_height;

    /* encounter visual overlay (populated by encounter's render_post_tick) */
    EncounterOverlay encounter_overlay;

    /* pre-built static models for overlay rendering (clouds, projectiles, snakelings).
       built once at init from model cache, drawn at overlay positions each frame. */
    Model cloud_model;       int cloud_model_ready;
    Model snakeling_model;   int snakeling_model_ready;
    Model ranged_proj_model; int ranged_proj_model_ready;
    Model magic_proj_model;  int magic_proj_model_ready;
    Model cloud_proj_model;  int cloud_proj_model_ready;
    Model pillar_models[4];  int pillar_models_ready;  /* 0=100%, 1=75%, 2=50%, 3=25% HP */

    /* active projectile flights: interpolated at 50Hz between game ticks.
       spawned from encounter overlay events, auto-expired on arrival. */
    FlightProjectile flights[MAX_FLIGHT_PROJECTILES];

    /* dynamic projectile model cache: lazily loads per-NPC-type projectile models */
#define MAX_PROJ_MODELS 24
    struct { uint32_t id; Model model; int ready; } proj_models[MAX_PROJ_MODELS];
    int proj_model_count;

    /* collision map: pointer to env's CollisionMap (shared, not owned).
       world offset translates arena coords to collision map world coords. */
    const CollisionMap* collision_map;
    int collision_world_offset_x;
    int collision_world_offset_y;

    /* tick pacing */
    double last_tick_time;

    /* rewind history: ring buffer of env snapshots */
    OsrsEnv* history;     /* heap-allocated array of RENDER_HISTORY_SIZE snapshots */
    int history_count;    /* how many snapshots stored (up to RENDER_HISTORY_SIZE) */
    int history_cursor;   /* current position when rewinding (-1 = live) */

    /* OSRS GUI panel system (inventory, equipment, prayer, combat, spellbook) */
    GuiState gui;

    /* interactive human control (H key toggle) */
    HumanInput human_input;

    /* cursor hover tile: tile under mouse cursor, updated every frame.
       -1 = no valid tile under cursor (off-arena or off-screen). */
    int hover_tile_x, hover_tile_y;

    /* right-click context menu (OSRS-style popup) */
    ContextMenu context_menu;
} RenderClient;

/* forward declarations */
static Camera3D render_build_3d_camera(RenderClient* rc);

/** Get the raw Player* for a given entity index (for GUI functions that need full Player state).
    Returns the Player* from get_entity for encounters that use Player structs (PvP, Zulrah).
    Returns NULL if no encounter or index is out of range. GUI code must NULL-check. */
static Player* render_get_player_ptr(OsrsEnv* env, int index) {
    if (env->encounter_def && env->encounter_state) {
        const EncounterDef* def = (const EncounterDef*)env->encounter_def;
        return (Player*)def->get_entity(env->encounter_state, index);
    }
    if (index >= 0 && index < NUM_AGENTS)
        return &env->players[index];
    return NULL;
}

/** Look up an animation sequence, checking secondary NPC cache as fallback. */
static AnimSequence* render_get_anim_sequence(RenderClient* rc, uint16_t seq_id) {
    AnimSequence* seq = NULL;
    if (rc->anim_cache) seq = anim_get_sequence(rc->anim_cache, seq_id);
    if (!seq && rc->npc_anim_cache) seq = anim_get_sequence(rc->npc_anim_cache, seq_id);
    return seq;
}

/** Look up an animation framebase, checking secondary NPC cache as fallback. */
static AnimFrameBase* render_get_framebase(RenderClient* rc, uint16_t base_id) {
    AnimFrameBase* fb = NULL;
    if (rc->anim_cache) fb = anim_get_framebase(rc->anim_cache, base_id);
    if (!fb && rc->npc_anim_cache) fb = anim_get_framebase(rc->npc_anim_cache, base_id);
    return fb;
}

/* ======================================================================== */
/* coordinate helpers                                                        */
/* ======================================================================== */

static inline int render_world_to_screen_x_rc(RenderClient* rc, int world_x) {
    return (world_x - rc->arena_base_x) * RENDER_TILE_SIZE;
}

static inline int render_world_to_screen_y_rc(RenderClient* rc, int world_y) {
    /* flip Y: OSRS Y increases north, screen Y increases down */
    int local_y = world_y - rc->arena_base_y;
    int flipped = (rc->arena_height - 1) - local_y;
    return RENDER_HEADER_HEIGHT + flipped * RENDER_TILE_SIZE;
}

/* legacy wrappers using default FIGHT_AREA bounds */
static inline int render_world_to_screen_x(int world_x) {
    return (world_x - FIGHT_AREA_BASE_X) * RENDER_TILE_SIZE;
}

static inline int render_world_to_screen_y(int world_y) {
    int local_y = world_y - FIGHT_AREA_BASE_Y;
    int flipped = (FIGHT_AREA_HEIGHT - 1) - local_y;
    return RENDER_HEADER_HEIGHT + flipped * RENDER_TILE_SIZE;
}

/* forward declarations for composite model system (defined after lifecycle) */
static void composite_free(PlayerComposite* comp);
static int render_select_secondary(RenderClient* rc, int player_idx);

/* forward declaration: inferno_npc_name is defined later in drawing section */
static const char* inferno_npc_name(int npc_def_id);

/* ======================================================================== */
/* right-click context menu helpers                                          */
/* ======================================================================== */

/** Resolve display name for a render entity (NPC or player).
    Uses the same lookup chain as render_draw_panel_npc: zulrah forms,
    inferno NPCs, then fallback to "NPC <def_id>". */
static const char* render_entity_display_name(RenderEntity* ent) {
    if (ent->entity_type == ENTITY_PLAYER) return "Player";

    /* zulrah forms */
    if (ent->npc_def_id == 2042) return "Zulrah";
    if (ent->npc_def_id == 2043) return "Zulrah";
    if (ent->npc_def_id == 2044) return "Zulrah";

    /* inferno NPCs */
    const char* inf = inferno_npc_name(ent->npc_def_id);
    if (inf) return inf;

    return TextFormat("NPC %d", ent->npc_def_id);
}

typedef struct {
    RenderClient* rc;
    OsrsEnv* env;
} RenderHumanAttackCtx;

static int render_can_human_attack_entity(
    void* ctx, const RenderEntity* entity, int entity_idx, int gui_entity_idx
) {
    RenderHumanAttackCtx* attack_ctx = (RenderHumanAttackCtx*)ctx;
    if (entity_idx == gui_entity_idx && entity->entity_type == ENTITY_PLAYER) {
        return 0;
    }

    if (entity->entity_type == ENTITY_NPC && !entity->npc_visible) {
        return 0;
    }

    if (attack_ctx->env->encounter_def && attack_ctx->env->encounter_state) {
        const EncounterDef* def = (const EncounterDef*)attack_ctx->env->encounter_def;
        if (entity->entity_type == ENTITY_NPC && def->is_human_targetable_npc_slot) {
            return def->is_human_targetable_npc_slot(
                attack_ctx->env->encounter_state, entity->npc_slot);
        }
    }

    return 1;
}

/** Clear/hide the context menu. */
static void context_menu_dismiss(ContextMenu* cm) {
    cm->visible = 0;
    cm->item_count = 0;
    cm->hover_idx = -1;
}

/** Add an item to the context menu. */
static void context_menu_add(ContextMenu* cm, ContextMenuAction action,
                              int entity_idx, const char* label) {
    if (cm->item_count >= CONTEXT_MENU_MAX_ITEMS) return;
    ContextMenuItem* item = &cm->items[cm->item_count++];
    item->action = action;
    item->entity_idx = entity_idx;
    snprintf(item->label, sizeof(item->label), "%s", label);
}

/** Build context menu from a right-click at screen position (mx, my).
    Performs entity hull hit-testing (3D) or tile hit-testing (2D),
    then builds the appropriate menu items. */
static void context_menu_build(RenderClient* rc, OsrsEnv* env, int mx, int my) {
    ContextMenu* cm = &rc->context_menu;
    RenderHumanAttackCtx attack_ctx = { .rc = rc, .env = env };
    cm->item_count = 0;
    cm->hover_idx = -1;
    cm->walk_tile_x = -1;
    cm->walk_tile_y = -1;

    /* collect all NPCs/entities under cursor (3D hull test or 2D tile test) */
    int hit_entities[MAX_RENDER_ENTITIES];
    int hit_count = 0;

    if (rc->mode_3d) {
        /* 3D: test against convex hulls */
        for (int ei = 0; ei < rc->entity_count; ei++) {
            RenderEntity* ent = &rc->entities[ei];
            if (!render_can_human_attack_entity(
                    &attack_ctx, ent, ei, rc->gui.gui_entity_idx)) {
                continue;
            }
            if (hull_contains(&rc->entity_hulls[ei], mx, my)) {
                if (hit_count < MAX_RENDER_ENTITIES)
                    hit_entities[hit_count++] = ei;
            }
        }

        /* resolve ground tile under cursor via ray-box intersection */
        Camera3D cam = render_build_3d_camera(rc);
        Ray ray = GetScreenToWorldRay((Vector2){ (float)mx, (float)my }, cam);
        float best_dist = 1e30f;
        for (int dy = 0; dy < rc->arena_height; dy++) {
            for (int dx = 0; dx < rc->arena_width; dx++) {
                int wx = rc->arena_base_x + dx;
                int wy = rc->arena_base_y + dy;
                float tx = (float)wx;
                float tz = -(float)(wy + 1);
                float ground_y = rc->terrain
                    ? terrain_height_avg(rc->terrain, wx, wy) : 2.0f;
                BoundingBox box = {
                    .min = { tx, ground_y - 0.1f, tz },
                    .max = { tx + 1.0f, ground_y, tz + 1.0f },
                };
                RayCollision col = GetRayCollisionBox(ray, box);
                if (col.hit && col.distance < best_dist) {
                    best_dist = col.distance;
                    cm->walk_tile_x = wx;
                    cm->walk_tile_y = wy;
                }
            }
        }
    } else {
        /* 2D: check tile under cursor, then check entities on that tile */
        if (my >= RENDER_HEADER_HEIGHT) {
            int grid_pixel_w = rc->arena_width * RENDER_TILE_SIZE;
            int grid_pixel_h = rc->arena_height * RENDER_TILE_SIZE;
            if (mx >= 0 && mx < grid_pixel_w &&
                my < RENDER_HEADER_HEIGHT + grid_pixel_h) {
                int wx = human_screen_to_world_x(mx, rc->arena_base_x, RENDER_TILE_SIZE);
                int wy = human_screen_to_world_y(my, rc->arena_base_y, rc->arena_height,
                                                  RENDER_HEADER_HEIGHT, RENDER_TILE_SIZE);
                cm->walk_tile_x = wx;
                cm->walk_tile_y = wy;

                for (int ei = 0; ei < rc->entity_count; ei++) {
                    RenderEntity* ent = &rc->entities[ei];
                    if (!render_can_human_attack_entity(
                            &attack_ctx, ent, ei, rc->gui.gui_entity_idx)) {
                        continue;
                    }
                    if (human_tile_hits_entity(ent, wx, wy)) {
                        if (hit_count < MAX_RENDER_ENTITIES)
                            hit_entities[hit_count++] = ei;
                    }
                }
            }
        }
    }

    /* build menu items: "Attack <NPC>" for each hit entity, then "Walk here" */
    for (int i = 0; i < hit_count; i++) {
        int ei = hit_entities[i];
        RenderEntity* ent = &rc->entities[ei];
        const char* name = render_entity_display_name(ent);
        char label[64];
        snprintf(label, sizeof(label), "Attack %s", name);
        context_menu_add(cm, CMENU_ACTION_ATTACK, ei, label);
    }

    if (cm->walk_tile_x >= 0)
        context_menu_add(cm, CMENU_ACTION_WALK_HERE, -1, "Walk here");

    context_menu_add(cm, CMENU_ACTION_CANCEL, -1, "Cancel");

    /* compute menu width from widest label */
    int max_w = CONTEXT_MENU_MIN_W;
    for (int i = 0; i < cm->item_count; i++) {
        int w = MeasureText(cm->items[i].label, 10) + CONTEXT_MENU_PADDING * 2 + 4;
        if (w > max_w) max_w = w;
    }
    cm->width = max_w;

    /* position: at cursor, clamped to screen bounds */
    int menu_h = cm->item_count * CONTEXT_MENU_ROW_H + CONTEXT_MENU_PADDING * 2;
    cm->screen_x = mx;
    cm->screen_y = my;
    if (cm->screen_x + cm->width > RENDER_WINDOW_W)
        cm->screen_x = RENDER_WINDOW_W - cm->width;
    if (cm->screen_y + menu_h > RENDER_WINDOW_H)
        cm->screen_y = RENDER_WINDOW_H - menu_h;
    if (cm->screen_x < 0) cm->screen_x = 0;
    if (cm->screen_y < 0) cm->screen_y = 0;

    cm->visible = (cm->item_count > 0);
}

/** Execute a context menu item action on the HumanInput staging buffer. */
static void context_menu_execute(RenderClient* rc, int item_idx) {
    ContextMenu* cm = &rc->context_menu;
    if (item_idx < 0 || item_idx >= cm->item_count) return;
    ContextMenuItem* item = &cm->items[item_idx];

    switch (item->action) {
        case CMENU_ACTION_WALK_HERE:
            rc->human_input.pending_move_x = cm->walk_tile_x;
            rc->human_input.pending_move_y = cm->walk_tile_y;
            rc->human_input.pending_attack = 0;
            human_input_queue_walk(&rc->human_input, cm->walk_tile_x, cm->walk_tile_y);
            human_set_click_cross(&rc->human_input, cm->screen_x, cm->screen_y, 0);
            break;

        case CMENU_ACTION_ATTACK: {
            int ei = item->entity_idx;
            if (ei >= 0 && ei < rc->entity_count) {
                rc->human_input.pending_attack = 1;
                rc->human_input.pending_target_idx = rc->entities[ei].npc_slot;
                rc->human_input.pending_move_x = -1;
                rc->human_input.pending_move_y = -1;
                if (rc->human_input.cursor_mode == CURSOR_SPELL_TARGET) {
                    rc->human_input.pending_spell = rc->human_input.selected_spell;
                    human_input_queue_spell_target(
                        &rc->human_input,
                        rc->human_input.selected_spell,
                        rc->human_input.pending_target_idx);
                    rc->human_input.cursor_mode = CURSOR_NORMAL;
                } else {
                    human_input_queue_attack_npc(
                        &rc->human_input,
                        rc->human_input.pending_target_idx);
                }
                human_set_click_cross(&rc->human_input, cm->screen_x, cm->screen_y, 1);
            }
            break;
        }

        case CMENU_ACTION_CANCEL:
        case CMENU_ACTION_NONE:
            break;
    }

    context_menu_dismiss(cm);
}

/** Draw the context menu as a 2D overlay. Call just before EndDrawing().
    OSRS style: dark brown/black rectangle, white text, yellow highlight on hover. */
static void context_menu_draw(RenderClient* rc) {
    ContextMenu* cm = &rc->context_menu;
    if (!cm->visible || cm->item_count == 0) return;

    int mx = GetMouseX();
    int my = GetMouseY();
    int menu_h = cm->item_count * CONTEXT_MENU_ROW_H + CONTEXT_MENU_PADDING * 2;

    /* update hover state */
    cm->hover_idx = -1;
    if (mx >= cm->screen_x && mx < cm->screen_x + cm->width &&
        my >= cm->screen_y && my < cm->screen_y + menu_h) {
        int row = (my - cm->screen_y - CONTEXT_MENU_PADDING) / CONTEXT_MENU_ROW_H;
        if (row >= 0 && row < cm->item_count)
            cm->hover_idx = row;
    }

    /* OSRS menu colors */
    Color bg     = (Color){ 57, 49, 35, 240 };
    Color border = (Color){ 90, 80, 60, 255 };
    Color text_normal = (Color){ 255, 255, 255, 255 };
    Color text_hover  = (Color){ 255, 255, 0, 255 };
    Color hover_bg    = (Color){ 80, 70, 50, 200 };

    /* draw background with border */
    DrawRectangle(cm->screen_x - 1, cm->screen_y - 1,
                  cm->width + 2, menu_h + 2, border);
    DrawRectangle(cm->screen_x, cm->screen_y, cm->width, menu_h, bg);

    /* draw items */
    for (int i = 0; i < cm->item_count; i++) {
        int iy = cm->screen_y + CONTEXT_MENU_PADDING + i * CONTEXT_MENU_ROW_H;

        if (i == cm->hover_idx) {
            DrawRectangle(cm->screen_x + 1, iy, cm->width - 2, CONTEXT_MENU_ROW_H, hover_bg);
        }

        Color tc = (i == cm->hover_idx) ? text_hover : text_normal;
        DrawText(cm->items[i].label,
                 cm->screen_x + CONTEXT_MENU_PADDING + 2,
                 iy + 2, 10, tc);
    }
}

/* ======================================================================== */
/* lifecycle                                                                 */
/* ======================================================================== */

static RenderClient* render_make_client(void) {
    RenderClient* rc = (RenderClient*)calloc(1, sizeof(RenderClient));
    rc->ticks_per_second = 1.667f;  /* OSRS game tick = 600ms */
    rc->last_tick_time = 0.0;
    rc->model_scale = 0.15f;  /* ~20px tile / ~150 model units */
    rc->zoom = 1.0f;
    rc->arena_base_x = FIGHT_AREA_BASE_X;
    rc->arena_base_y = FIGHT_AREA_BASE_Y;
    rc->arena_width = FIGHT_AREA_WIDTH;
    rc->arena_height = FIGHT_AREA_HEIGHT;
    rc->mode_3d = 1;
    rc->show_safe_spots = 0;
    rc->show_debug = 0;
    rc->cam_yaw = 0.0f;
    rc->cam_pitch = 0.6f;    /* ~34 degrees, similar to OSRS default */
    rc->cam_dist = 40.0f;
    /* fight area center (Z negated: OSRS +Y = north maps to -Z) */
    rc->cam_target_x = (float)rc->arena_base_x + (float)rc->arena_width / 2.0f;
    rc->cam_target_z = -((float)rc->arena_base_y + (float)rc->arena_height / 2.0f);
    rc->history = (OsrsEnv*)calloc(RENDER_HISTORY_SIZE, sizeof(OsrsEnv));
    rc->history_count = 0;
    rc->history_cursor = -1;  /* -1 = live (not rewinding) */
    rc->entity_count = 0;  /* populated by render_populate_entities */
    rc->prev_entity_count = 0;
    rc->hover_tile_x = -1;
    rc->hover_tile_y = -1;
    for (int i = 0; i < MAX_RENDER_ENTITIES; i++) {
        rc->anim[i].primary_seq_id = -1;
        rc->anim[i].secondary_seq_id = 808; /* ANIM_SEQ_IDLE */
        rc->prev_npc_slot[i] = -1;
    }

    InitWindow(RENDER_WINDOW_W, RENDER_WINDOW_H, "OSRS PvP Debug Viewer");
    SetTargetFPS(60);

    /* load overhead prayer icon textures from exported sprites.
       OSRS headIcon index: 0=melee, 1=ranged, 2=magic, 3=retribution, 4=smite, 5=redemption */
    {
        const char* paths[] = {
            "data/sprites/gui/headicons_prayer_0.png",
            "data/sprites/gui/headicons_prayer_1.png",
            "data/sprites/gui/headicons_prayer_2.png",
            "data/sprites/gui/headicons_prayer_3.png",
            "data/sprites/gui/headicons_prayer_4.png",
            "data/sprites/gui/headicons_prayer_5.png",
        };
        rc->prayer_icons_loaded = 1;
        for (int i = 0; i < 6; i++) {
            if (FileExists(paths[i])) {
                rc->prayer_icons[i] = LoadTexture(paths[i]);
            } else {
                rc->prayer_icons_loaded = 0;
            }
        }
    }

    /* load hitsplat sprite textures (317 classic: hitmarks_0..4.png) */
    {
        rc->hitmark_sprites_loaded = 1;
        for (int i = 0; i < 5; i++) {
            const char* path = TextFormat("data/sprites/gui/hitmarks_%d.png", i);
            if (FileExists(path)) {
                rc->hitmark_sprites[i] = LoadTexture(path);
            } else {
                rc->hitmark_sprites_loaded = 0;
            }
        }
    }

    /* load click cross sprite textures (4 yellow + 4 red animation frames) */
    {
        static const char* cross_names[8] = {
            "cross_yellow_1", "cross_yellow_2", "cross_yellow_3", "cross_yellow_4",
            "cross_red_1", "cross_red_2", "cross_red_3", "cross_red_4",
        };
        rc->click_cross_loaded = 1;
        for (int i = 0; i < 8; i++) {
            const char* path = TextFormat("data/sprites/gui/%s.png", cross_names[i]);
            if (FileExists(path)) {
                rc->click_cross_sprites[i] = LoadTexture(path);
            } else {
                rc->click_cross_loaded = 0;
            }
        }
    }

    rc->debug_hit_wx = -1;
    rc->debug_hit_wy = -1;

    /* initialize GUI panel system */
    rc->gui.active_tab = GUI_TAB_INVENTORY;
    rc->gui.panel_x = RENDER_GRID_W;
    rc->gui.panel_y = RENDER_HEADER_HEIGHT;
    rc->gui.panel_w = RENDER_PANEL_WIDTH;
    rc->gui.panel_h = RENDER_GRID_H;  /* full height — boss info moved to top-left overlay */
    rc->gui.tab_h = 28;
    rc->gui.status_bar_h = 42;  /* 3 bars x 12px + 2px gaps */
    rc->gui.gui_entity_idx = 0;
    rc->gui.gui_entity_count = 0;

    /* inventory interaction state */
    gui_reset_inventory_ui_state(&rc->gui);

    /* human input control */
    human_input_init(&rc->human_input);

    /* context menu (calloc zeroes everything, just set hover_idx sentinel) */
    rc->context_menu.hover_idx = -1;

    /* load GUI sprites from exported cache data */
    gui_load_sprites(&rc->gui);

    return rc;
}

/**
 * Build a static raylib Model from a cached OsrsModel.
 * Copies expanded vertex + color data into a new Mesh and uploads to GPU.
 * Returns 1 on success, 0 if model not found.
 */
static int render_build_static_model(ModelCache* cache, uint32_t model_id, Model* out) {
    OsrsModel* om = model_cache_get(cache, model_id);
    if (!om || om->mesh.vertexCount == 0) return 0;

    Mesh mesh = { 0 };
    mesh.vertexCount = om->mesh.vertexCount;
    mesh.triangleCount = om->mesh.triangleCount;
    mesh.vertices = (float*)RL_MALLOC(mesh.vertexCount * 3 * sizeof(float));
    mesh.colors = (unsigned char*)RL_MALLOC(mesh.vertexCount * 4);
    memcpy(mesh.vertices, om->mesh.vertices, mesh.vertexCount * 3 * sizeof(float));
    memcpy(mesh.colors, om->mesh.colors, mesh.vertexCount * 4);

    UploadMesh(&mesh, false);
    *out = LoadModelFromMesh(mesh);
    return 1;
}

/** Lazily load and cache an explicit projectile model by GFX model ID.
 *  Searches both model_cache and npc_model_cache. model_id 0 means the
 *  caller intentionally wants style-based fallback; missing explicit models
 *  abort so backend/render drift fails loudly. */
static Model* render_get_proj_model(RenderClient* rc, uint32_t model_id) {
    if (model_id == 0) return NULL;
    for (int i = 0; i < rc->proj_model_count; i++) {
        if (rc->proj_models[i].id == model_id)
            return rc->proj_models[i].ready ? &rc->proj_models[i].model : NULL;
    }
    if (rc->proj_model_count >= MAX_PROJ_MODELS) {
        fprintf(stderr, "render: projectile model cache exhausted while loading model %u\n",
                model_id);
        abort();
    }
    int idx = rc->proj_model_count++;
    rc->proj_models[idx].id = model_id;
    rc->proj_models[idx].ready = render_build_static_model(
        rc->model_cache, model_id, &rc->proj_models[idx].model);
    if (!rc->proj_models[idx].ready && rc->npc_model_cache) {
        rc->proj_models[idx].ready = render_build_static_model(
            rc->npc_model_cache, model_id, &rc->proj_models[idx].model);
    }
    if (!rc->proj_models[idx].ready) {
        fprintf(stderr, "render: explicit projectile model %u is missing from loaded caches\n",
                model_id);
        abort();
    }
    return &rc->proj_models[idx].model;
}

static OsrsModel* render_get_projectile_osrs_model(RenderClient* rc, uint32_t model_id) {
    if (model_id == 0) return NULL;
    OsrsModel* om = model_cache_get(rc->model_cache, model_id);
    if (!om && rc->npc_model_cache)
        om = model_cache_get(rc->npc_model_cache, model_id);
    if (!om) {
        fprintf(stderr, "render: explicit projectile model %u is missing from loaded caches\n",
                model_id);
        abort();
    }
    return om;
}

static AnimModelState* render_create_projectile_anim_state(
    RenderClient* rc, uint32_t model_id, int anim_id
) {
    if (anim_id < 0 || model_id == 0) return NULL;
    OsrsModel* om = render_get_projectile_osrs_model(rc, model_id);
    if (!om->vertex_skins || om->base_vert_count == 0) {
        fprintf(stderr, "render: projectile model %u cannot play animation %d\n",
                model_id, anim_id);
        abort();
    }
    return anim_model_state_create(om->vertex_skins, om->base_vert_count);
}

/**
 * Build all overlay models (clouds, projectiles, snakelings) from the model cache.
 * Call after model_cache is loaded.
 */
static void render_init_overlay_models(RenderClient* rc) {
    if (!rc->model_cache) return;

    rc->cloud_model_ready = render_build_static_model(
        rc->model_cache, GFX_TOXIC_CLOUD_MODEL, &rc->cloud_model);
    rc->snakeling_model_ready = render_build_static_model(
        rc->model_cache, SNAKELING_MODEL_ID, &rc->snakeling_model);
    rc->ranged_proj_model_ready = render_build_static_model(
        rc->model_cache, GFX_RANGED_PROJ_MODEL, &rc->ranged_proj_model);
    rc->magic_proj_model_ready = render_build_static_model(
        rc->model_cache, GFX_MAGIC_PROJ_MODEL, &rc->magic_proj_model);

    rc->cloud_proj_model_ready = render_build_static_model(
        rc->model_cache, GFX_CLOUD_PROJ_MODEL, &rc->cloud_proj_model);
    {
        uint32_t pillar_ids[4] = { INF_PILLAR_MODEL_100, INF_PILLAR_MODEL_75,
                                    INF_PILLAR_MODEL_50, INF_PILLAR_MODEL_25 };
        rc->pillar_models_ready = 1;
        for (int i = 0; i < 4; i++) {
            if (!render_build_static_model(rc->model_cache, pillar_ids[i], &rc->pillar_models[i]))
                rc->pillar_models_ready = 0;
        }
    }

    if (rc->cloud_model_ready) printf("overlay: cloud model loaded\n");
    if (rc->pillar_models_ready) printf("overlay: pillar models loaded (4 HP levels)\n");
    if (rc->snakeling_model_ready) printf("overlay: snakeling model loaded\n");
    if (rc->ranged_proj_model_ready) printf("overlay: ranged projectile model loaded\n");
    if (rc->magic_proj_model_ready) printf("overlay: magic projectile model loaded\n");
    if (rc->cloud_proj_model_ready) printf("overlay: cloud projectile model loaded\n");
}

/* ======================================================================== */
/* projectile flight system                                                   */
/* ======================================================================== */

/**
 * Spawn a flight projectile with OSRS-accurate parabolic arc and target tracking.
 *
 * Matches Projectile.java setDestination():
 *   - position re-computed each sub-tick toward current target
 *   - yaw/pitch updated from velocity vector each tick
 *   - height follows parabolic arc with quadratic correction
 */
static void flight_deactivate(FlightProjectile* fp) {
    if (fp->anim_state) {
        anim_model_state_free(fp->anim_state);
        fp->anim_state = NULL;
    }
    fp->active = 0;
}

static void flight_clear_all(RenderClient* rc) {
    for (int i = 0; i < MAX_FLIGHT_PROJECTILES; i++) {
        flight_deactivate(&rc->flights[i]);
    }
}

static void flight_finish(RenderClient* rc, FlightProjectile* fp) {
    if (fp->impact_gfx_id > 0) {
        effect_spawn_spotanim_subtile(
            rc->effects, fp->impact_gfx_id,
            fp->dst_x * 128.0f, fp->dst_y * 128.0f,
            rc->effect_client_tick_counter + 1,
            rc->anim_cache, rc->model_cache, rc->npc_model_cache);
    }
    flight_deactivate(fp);
}

static void flight_spawn(RenderClient* rc,
                         float src_x, float src_y, float dst_x, float dst_y,
                         int style, int damage,
                         int duration_ticks, int start_h, int end_h, int curve,
                         float arc_height, int tracks_target, uint32_t model_id,
                         int anim_id, int impact_gfx_id,
                         int start_delay, int motion_mode,
                         float offset_x, float offset_y, float offset_z) {
    int slot = -1;
    for (int i = 0; i < MAX_FLIGHT_PROJECTILES; i++) {
        if (!rc->flights[i].active) { slot = i; break; }
    }
    if (slot < 0) return;

    FlightProjectile* fp = &rc->flights[slot];
    memset(fp, 0, sizeof(FlightProjectile));
    fp->active = 1;
    fp->src_x = src_x;
    fp->src_y = src_y;
    fp->dst_x = dst_x;
    fp->dst_y = dst_y;
    fp->x = src_x;
    fp->y = src_y;
    fp->progress = 0.0f;
    fp->speed = 1.0f / (float)duration_ticks;
    fp->start_height = (float)start_h / 128.0f;
    fp->end_height = (float)end_h / 128.0f;
    fp->curve = (float)curve;
    fp->style = style;
    fp->damage = damage;
    fp->arc_height = arc_height;
    fp->tracks_target = tracks_target;
    fp->model_id = model_id;
    fp->anim_id = anim_id;
    fp->anim_frame = 0;
    fp->anim_tick_counter = 0;
    fp->anim_state = render_create_projectile_anim_state(rc, model_id, anim_id);
    fp->impact_gfx_id = impact_gfx_id;
    fp->start_delay = start_delay;
    fp->motion_mode = motion_mode;
    fp->offset_x = offset_x;
    fp->offset_y = offset_y;
    fp->offset_z = offset_z;

    /* height arc: OSRS SceneProjectile.calculateIncrements
       skip quadratic computation when using sinusoidal arc */
    float dx = dst_x - src_x, dy = dst_y - src_y;
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist < 0.01f) dist = 1.0f;
    if (arc_height > 0.0f) {
        fp->height_vel = 0.0f;
        fp->height_accel = 0.0f;
    } else {
        fp->height_vel = -dist * tanf(curve * PROJ_OSRS_SLOPE_TO_RAD);
        fp->height_accel = 2.0f * (fp->end_height - fp->start_height - fp->height_vel);
    }

    /* initial facing */
    fp->yaw = atan2f(dx, dy);
    fp->pitch = (arc_height > 0.0f) ? 0.0f : atan2f(fp->height_vel, dist);
}

static void flight_advance_animation(RenderClient* rc, FlightProjectile* fp) {
    if (fp->anim_id < 0) return;
    AnimSequence* seq = render_get_anim_sequence(rc, (uint16_t)fp->anim_id);
    if (!seq || seq->frame_count <= 0) {
        fprintf(stderr, "render: projectile animation %d is missing\n", fp->anim_id);
        abort();
    }
    fp->anim_tick_counter++;
    while (fp->anim_tick_counter >= seq->frames[fp->anim_frame].delay) {
        fp->anim_tick_counter -= seq->frames[fp->anim_frame].delay;
        fp->anim_frame++;
        if (fp->anim_frame >= seq->frame_count) {
            fp->anim_frame = 0;
        }
    }
}

static inline Matrix render_projectile_transform(
    float scale_x, float scale_y, float scale_z,
    float yaw, float pitch, Vector3 position
) {
    Matrix transform = MatrixScale(-scale_x, scale_y, scale_z);
    transform = MatrixMultiply(
        transform,
        MatrixMultiply(MatrixRotateY(yaw + 1.5707963f), MatrixRotateX(pitch)));
    transform = MatrixMultiply(transform, MatrixTranslate(position.x, position.y, position.z));
    return transform;
}

static inline Matrix render_projectile_transform_offset(
    float scale_x, float scale_y, float scale_z,
    float yaw, float pitch, Vector3 position,
    float offset_x, float offset_y, float offset_z
) {
    Matrix transform = MatrixScale(-scale_x, scale_y, scale_z);
    transform = MatrixMultiply(
        transform, MatrixTranslate(offset_x, offset_z, offset_y));
    transform = MatrixMultiply(
        transform,
        MatrixMultiply(MatrixRotateY(yaw + 1.5707963f), MatrixRotateX(pitch)));
    transform = MatrixMultiply(transform, MatrixTranslate(position.x, position.y, position.z));
    return transform;
}

static inline int render_pvp_ranged_spec_weapon_for_item(uint8_t weapon_db_idx) {
    switch (weapon_db_idx) {
        case ITEM_DARK_BOW:           return RANGED_SPEC_DARK_BOW;
        case ITEM_HEAVY_BALLISTA:     return RANGED_SPEC_BALLISTA;
        case ITEM_ARMADYL_CROSSBOW:   return RANGED_SPEC_ACB;
        case ITEM_ZARYTE_CROSSBOW:    return RANGED_SPEC_ZCB;
        case ITEM_MAGIC_SHORTBOW_I:   return RANGED_SPEC_MSB;
        case ITEM_MORRIGANS_JAVELIN:  return RANGED_SPEC_MORRIGANS;
        default:                      return RANGED_SPEC_NONE;
    }
}

static inline int render_pvp_distance_to_target(
    const RenderEntity* attacker, const RenderEntity* target
) {
    int target_size = target->entity_type == ENTITY_NPC && target->npc_size > 1
        ? target->npc_size : 1;
    return encounter_dist_to_npc(
        attacker->x, attacker->y, target->x, target->y, target_size);
}

/**
 * Advance all active flights by one client tick (20ms).
 *
 * Matches OSRS Projectile.setDestination() tracking:
 *   remaining = (cycleEnd - currentCycle)
 *   vel = (target - current) / remaining
 *   orientation = atan2(vel_x, vel_y)
 *   pitch = atan2(height_vel, horiz_speed)
 */
static void flight_client_tick(RenderClient* rc) {
    for (int i = 0; i < MAX_FLIGHT_PROJECTILES; i++) {
        FlightProjectile* fp = &rc->flights[i];
        if (!fp->active) continue;

        /* start delay: count down before projectile becomes visible/moves */
        if (fp->start_delay > 0) {
            fp->start_delay--;
            continue;
        }

        if (fp->motion_mode == ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED) {
            fp->x = fp->dst_x;
            fp->y = fp->dst_y;
            fp->pitch = 0.0f;
            flight_advance_animation(rc, fp);
            fp->progress += fp->speed;
            if (fp->progress >= 1.0f) {
                flight_finish(rc, fp);
            }
            continue;
        }

        /* remaining sub-ticks (avoid div by zero) */
        float remaining = (1.0f - fp->progress) / fp->speed;
        if (remaining < 0.5f) remaining = 0.5f;

        /* re-aim velocity toward current target (OSRS tracking) */
        float vx = (fp->dst_x - fp->x) / remaining;
        float vy = (fp->dst_y - fp->y) / remaining;
        float horiz_speed = sqrtf(vx * vx + vy * vy);

        fp->x += vx;
        fp->y += vy;

        /* update facing from velocity vector */
        if (horiz_speed > 0.001f) {
            fp->yaw = atan2f(vx, vy);
            float h_vel = fp->height_vel + fp->height_accel * fp->progress;
            fp->pitch = atan2f(h_vel, horiz_speed);
        }

        flight_advance_animation(rc, fp);
        fp->progress += fp->speed;
        if (fp->progress >= 1.0f) {
            flight_finish(rc, fp);
        }
    }
}

/**
 * Get the interpolated world position of a flight projectile.
 */
static Vector3 flight_get_position(const FlightProjectile* fp, float src_ground, float dst_ground) {
    if (fp->motion_mode == ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED) {
        return (Vector3){ fp->x + 0.5f, dst_ground + fp->end_height,
            -(fp->y + 1.0f) + 0.5f };
    }

    float t = fp->progress;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    float ground = src_ground + (dst_ground - src_ground) * t;
    float h;
    if (fp->arc_height > 0.0f) {
        /* sinusoidal arc (from InfernoTrainer ArcProjectileMotionInterpolator) */
        h = sinf(t * 3.14159265f) * fp->arc_height
            + fp->start_height + (fp->end_height - fp->start_height) * t;
    } else {
        /* quadratic arc (OSRS SceneProjectile) */
        h = fp->start_height + fp->height_vel * t + 0.5f * fp->height_accel * t * t;
    }

    return (Vector3){ fp->x + 0.5f, ground + h, -(fp->y + 1.0f) + 0.5f };
}

static void render_destroy_client(RenderClient* rc) {
    flight_clear_all(rc);
    /* free GUI panel sprites */
    gui_unload_sprites(&rc->gui);
    /* free prayer icon textures */
    if (rc->prayer_icons_loaded) {
        for (int i = 0; i < 6; i++) {
            UnloadTexture(rc->prayer_icons[i]);
        }
    }
    /* free hitsplat sprite textures */
    if (rc->hitmark_sprites_loaded) {
        for (int i = 0; i < 5; i++) {
            UnloadTexture(rc->hitmark_sprites[i]);
        }
    }
    /* free click cross sprite textures */
    if (rc->click_cross_loaded) {
        for (int i = 0; i < 8; i++) {
            UnloadTexture(rc->click_cross_sprites[i]);
        }
    }
    /* free overlay models */
    if (rc->cloud_model_ready) UnloadModel(rc->cloud_model);
    if (rc->snakeling_model_ready) UnloadModel(rc->snakeling_model);
    if (rc->ranged_proj_model_ready) UnloadModel(rc->ranged_proj_model);
    if (rc->magic_proj_model_ready) UnloadModel(rc->magic_proj_model);
    if (rc->cloud_proj_model_ready) UnloadModel(rc->cloud_proj_model);
    if (rc->pillar_models_ready) {
        for (int i = 0; i < 4; i++) UnloadModel(rc->pillar_models[i]);
    }
    /* free dynamic projectile model cache */
    for (int i = 0; i < rc->proj_model_count; i++) {
        if (rc->proj_models[i].ready) UnloadModel(rc->proj_models[i].model);
    }
    /* free per-entity composite models */
    for (int p = 0; p < MAX_RENDER_ENTITIES; p++) {
        composite_free(&rc->composites[p]);
    }
    if (rc->model_cache) {
        model_cache_free(rc->model_cache);
        rc->model_cache = NULL;
    }
    if (rc->anim_cache) {
        anim_cache_free(rc->anim_cache);
        rc->anim_cache = NULL;
    }
    if (rc->terrain) {
        terrain_free(rc->terrain);
        rc->terrain = NULL;
    }
    if (rc->objects) {
        objects_free(rc->objects);
        rc->objects = NULL;
    }
    if (rc->objects_zuk) {
        objects_free(rc->objects_zuk);
        rc->objects_zuk = NULL;
    }
    if (rc->npcs) {
        objects_free(rc->npcs);
        rc->npcs = NULL;
    }
    human_input_destroy(&rc->human_input);
    CloseWindow();
    free(rc->history);
    free(rc);
}

/* ======================================================================== */
/* input                                                                     */
/* ======================================================================== */

static void render_handle_input(RenderClient* rc, OsrsEnv* env) {
    RenderHumanAttackCtx attack_ctx = { .rc = rc, .env = env };
    if (IsKeyPressed(KEY_SPACE))  rc->is_paused = !rc->is_paused;

    if (IsKeyPressed(KEY_RIGHT) && rc->is_paused) {
        if (rc->history_cursor >= 0) {
            /* in rewind mode: advance through history */
            if (rc->history_cursor < rc->history_count - 1) {
                rc->history_cursor++;
                rc->step_back = 1;  /* triggers restore in main loop */
            } else {
                /* restore latest snapshot then return to live */
                rc->history_cursor = rc->history_count - 1;
                rc->step_back = 1;
            }
        } else {
            rc->step_once = 1;  /* live mode: step sim forward */
        }
    }

    if (IsKeyPressed(KEY_LEFT) && rc->is_paused) {
        if (rc->history_cursor == -1 && rc->history_count > 1) {
            /* enter rewind from live: go to second-to-last snapshot */
            rc->history_cursor = rc->history_count - 2;
        } else if (rc->history_cursor > 0) {
            rc->history_cursor--;
        }
        rc->step_back = 1;
    }

    if (IsKeyPressed(KEY_TAB))    ToggleFullscreen();
    if (IsKeyPressed(KEY_C))      rc->show_collision = !rc->show_collision;
    if (IsKeyPressed(KEY_P))      rc->show_pathfinding = !rc->show_pathfinding;
    if (IsKeyPressed(KEY_M))      rc->show_models = !rc->show_models;
    if (IsKeyPressed(KEY_S))      rc->show_safe_spots = !rc->show_safe_spots;
    if (IsKeyPressed(KEY_D))      rc->show_debug = !rc->show_debug;
    if (IsKeyPressed(KEY_T))      rc->mode_3d = !rc->mode_3d;

    float wheel = GetMouseWheelMove();

    if (rc->mode_3d) {
        /* 3D camera controls: right-drag to orbit (disabled in human mode where
           right-click opens context menu — use middle-drag for camera instead),
           scroll to zoom */
        if (!rc->human_input.enabled && IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 delta = GetMouseDelta();
            rc->cam_yaw -= delta.x * 0.005f;
            rc->cam_pitch += delta.y * 0.005f;
            if (rc->cam_pitch < 0.1f) rc->cam_pitch = 0.1f;
            if (rc->cam_pitch > 1.4f) rc->cam_pitch = 1.4f;
        }
        /* middle-drag: orbit in human mode (since right-click opens menu),
           pan in non-human mode */
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            Vector2 delta = GetMouseDelta();
            if (rc->human_input.enabled) {
                /* orbit */
                rc->cam_yaw -= delta.x * 0.005f;
                rc->cam_pitch += delta.y * 0.005f;
                if (rc->cam_pitch < 0.1f) rc->cam_pitch = 0.1f;
                if (rc->cam_pitch > 1.4f) rc->cam_pitch = 1.4f;
            } else {
                /* pan: world drags with the mouse (grab-and-drag convention) */
                float cs = cosf(rc->cam_yaw), sn = sinf(rc->cam_yaw);
                rc->cam_target_x += (delta.x * cs + delta.y * sn) * 0.05f;
                rc->cam_target_z += (-delta.x * sn + delta.y * cs) * 0.05f;
            }
        }
        if (wheel != 0.0f) {
            rc->cam_dist *= (wheel > 0) ? (1.0f / 1.15f) : 1.15f;
            if (rc->cam_dist < 5.0f) rc->cam_dist = 5.0f;
            if (rc->cam_dist > 200.0f) rc->cam_dist = 200.0f;
        }
        /* camera follow: lock onto controlled entity's sub-tile position */
        if (rc->human_input.enabled && rc->entity_count > 0) {
            int eidx = rc->gui.gui_entity_idx;
            if (eidx < rc->entity_count) {
                float tx = (float)rc->sub_x[eidx] / 128.0f;
                float tz = -(float)rc->sub_y[eidx] / 128.0f;
                /* smooth follow: time-normalized exponential decay so camera
                   feel is consistent regardless of frame rate. decay rate
                   tuned for 0.15 per frame at 60fps baseline. */
                float dt = GetFrameTime();
                float lerp = 1.0f - powf(0.85f, dt * 60.0f);
                rc->cam_target_x += (tx - rc->cam_target_x) * lerp;
                rc->cam_target_z += (tz - rc->cam_target_z) * lerp;
            }
        }
    } else {
        /* 2D zoom */
        if (wheel != 0.0f) {
            rc->zoom *= (wheel > 0) ? 1.15f : (1.0f / 1.15f);
            if (rc->zoom < 0.3f) rc->zoom = 0.3f;
            if (rc->zoom > 8.0f) rc->zoom = 8.0f;
        }
    }

    /* number keys 1-5: GUI tab switching */
    if (IsKeyPressed(KEY_ONE))    rc->gui.active_tab = GUI_TAB_INVENTORY;
    if (IsKeyPressed(KEY_TWO))    rc->gui.active_tab = GUI_TAB_COMBAT;
    if (IsKeyPressed(KEY_THREE))  rc->gui.active_tab = GUI_TAB_PRAYER;
    if (IsKeyPressed(KEY_FOUR))   rc->gui.active_tab = GUI_TAB_SPELLBOOK;
    if (IsKeyPressed(KEY_FIVE))   rc->gui.active_tab = GUI_TAB_EQUIPMENT;

    /* 9/0: replay speed control (discrete steps) */
    {
        static const float speed_steps[] = { 0.5f, 1.0f, 1.667f, 5.0f, 15.0f, 50.0f, 0.0f };
        static const int num_steps = sizeof(speed_steps) / sizeof(speed_steps[0]);
        if (IsKeyPressed(KEY_NINE) || IsKeyPressed(KEY_ZERO)) {
            /* find current step index */
            int cur = -1;
            for (int i = 0; i < num_steps; i++) {
                if (speed_steps[i] == rc->ticks_per_second) { cur = i; break; }
            }
            if (cur < 0) cur = 2; /* default to OSRS speed if not on a step */
            if (IsKeyPressed(KEY_NINE) && cur > 0)             rc->ticks_per_second = speed_steps[cur - 1];
            if (IsKeyPressed(KEY_ZERO) && cur < num_steps - 1) rc->ticks_per_second = speed_steps[cur + 1];
        }
    }

    /* H key: toggle human control */
    if (IsKeyPressed(KEY_H)) {
        rc->human_input.enabled = !rc->human_input.enabled;
        if (!rc->human_input.enabled) {
            human_input_clear_pending(&rc->human_input);
            human_input_clear_move(&rc->human_input);
            rc->human_input.cursor_mode = CURSOR_NORMAL;
            context_menu_dismiss(&rc->context_menu);
        }
        fprintf(stderr, "human control: %s\n", rc->human_input.enabled ? "ON" : "OFF");
    }

    /* ESC: dismiss context menu first, then cancel spell targeting */
    if (IsKeyPressed(KEY_ESCAPE)) {
        if (rc->context_menu.visible) {
            context_menu_dismiss(&rc->context_menu);
        } else if (rc->human_input.cursor_mode == CURSOR_SPELL_TARGET) {
            rc->human_input.cursor_mode = CURSOR_NORMAL;
        }
    }

    /* GUI: G cycles viewed entity, tab clicks switch panels */
    if (IsKeyPressed(KEY_G))      gui_cycle_entity(&rc->gui);
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        int mx = GetMouseX();
        int my = GetMouseY();
        int handled = 0;

        /* 0. context menu: if visible, intercept click for item selection or dismissal */
        if (rc->context_menu.visible) {
            ContextMenu* cm = &rc->context_menu;
            int menu_h = cm->item_count * CONTEXT_MENU_ROW_H + CONTEXT_MENU_PADDING * 2;
            if (mx >= cm->screen_x && mx < cm->screen_x + cm->width &&
                my >= cm->screen_y && my < cm->screen_y + menu_h) {
                int row = (my - cm->screen_y - CONTEXT_MENU_PADDING) / CONTEXT_MENU_ROW_H;
                if (row >= 0 && row < cm->item_count)
                    context_menu_execute(rc, row);
                else
                    context_menu_dismiss(cm);
            } else {
                context_menu_dismiss(cm);
            }
            handled = 1;
        }

        /* 1. tab bar click */
        if (!handled)
            handled = gui_handle_tab_click(&rc->gui, mx, my);

        /* 2. panel content area (when human control is on) */
        if (!handled && rc->human_input.enabled &&
            mx >= rc->gui.panel_x && mx < rc->gui.panel_x + rc->gui.panel_w &&
            my >= rc->gui.panel_y && my < rc->gui.panel_y + rc->gui.panel_h) {

            Player* viewed = (rc->entity_count > 0 && rc->gui.gui_entity_idx < rc->entity_count)
                ? render_get_player_ptr(env, rc->gui.gui_entity_idx) : NULL;

            if (viewed) {
                switch (rc->gui.active_tab) {
                    case GUI_TAB_PRAYER:
                        human_handle_prayer_click(&rc->human_input, &rc->gui, viewed, mx, my);
                        handled = 1;
                        break;
                    case GUI_TAB_SPELLBOOK:
                        human_handle_spell_click(&rc->human_input, &rc->gui, mx, my);
                        handled = 1;
                        break;
                    case GUI_TAB_COMBAT:
                        human_handle_combat_click(&rc->human_input, &rc->gui, viewed, mx, my);
                        handled = 1;
                        break;
                    default:
                        break;  /* inventory handled separately by gui_inv_handle_mouse */
                }
            }
        }

        /* 3. ground/entity click (game grid area, left of panel) */
        if (!handled && rc->human_input.enabled && mx < rc->gui.panel_x) {
            if (rc->mode_3d) {
                /* 3D entity click: test mouse against convex hull of each entity's
                   projected model (ported from RuneLite RSNPCMixin.getConvexHull).
                   check entities FIRST before ground tiles. */
                int entity_hit = 0;
                for (int ei = 0; ei < rc->entity_count; ei++) {
                    RenderEntity* ent = &rc->entities[ei];
                    if (!render_can_human_attack_entity(
                            &attack_ctx, ent, ei, rc->gui.gui_entity_idx)) {
                        continue;
                    }
                    if (hull_contains(&rc->entity_hulls[ei], mx, my)) {
                        rc->human_input.pending_attack = 1;
                        rc->human_input.pending_target_idx = rc->entities[ei].npc_slot;
                        /* attack cancels movement — server stops walking to old dest */
                        rc->human_input.pending_move_x = -1;
                        rc->human_input.pending_move_y = -1;
                        if (rc->human_input.cursor_mode == CURSOR_SPELL_TARGET) {
                            rc->human_input.pending_spell = rc->human_input.selected_spell;
                            human_input_queue_spell_target(
                                &rc->human_input,
                                rc->human_input.selected_spell,
                                rc->human_input.pending_target_idx);
                            rc->human_input.cursor_mode = CURSOR_NORMAL;
                        } else {
                            human_input_queue_attack_npc(
                                &rc->human_input,
                                rc->human_input.pending_target_idx);
                        }
                        human_set_click_cross(&rc->human_input, mx, my, 1);
                        entity_hit = 1;
                        break;
                    }
                }

                if (entity_hit) { /* already handled */ }
                else {
                /* 3D ground click: ray-box intersection against actual tile cube geometry */
                Camera3D cam = render_build_3d_camera(rc);
                Ray ray = GetScreenToWorldRay((Vector2){ (float)mx, (float)my }, cam);
                rc->debug_ray_origin = ray.position;
                rc->debug_ray_dir = ray.direction;

                /* ray-box intersection against tile cubes at actual ground height.
                   when terrain is loaded, each tile sits at terrain_height_avg().
                   when no terrain (flat encounters), tiles sit at plat_y=2.0. */
                float best_dist = 1e30f;
                int best_wx = -1, best_wy = -1;
                for (int dy = 0; dy < rc->arena_height; dy++) {
                    for (int dx = 0; dx < rc->arena_width; dx++) {
                        int wx = rc->arena_base_x + dx;
                        int wy = rc->arena_base_y + dy;
                        float tx = (float)wx;
                        float tz = -(float)(wy + 1);
                        float ground_y = rc->terrain
                            ? terrain_height_avg(rc->terrain, wx, wy)
                            : 2.0f;
                        BoundingBox box = {
                            .min = { tx, ground_y - 0.1f, tz },
                            .max = { tx + 1.0f, ground_y, tz + 1.0f },
                        };
                        RayCollision col = GetRayCollisionBox(ray, box);
                        if (col.hit && col.distance < best_dist) {
                            best_dist = col.distance;
                            best_wx = wx;
                            best_wy = wy;
                            rc->debug_ray_hit_x = col.point.x;
                            rc->debug_ray_hit_y = col.point.y;
                            rc->debug_ray_hit_z = col.point.z;
                        }
                    }
                }
                rc->debug_hit_wx = best_wx;
                rc->debug_hit_wy = best_wy;
                rc->debug_plane_wx = -1;
                rc->debug_plane_wy = -1;
                if (best_wx >= 0) {
                    /* ground click while spell-targeting: walk AND cancel
                       (matches OSRS — clicking ground doesn't just cancel) */
                    if (rc->human_input.cursor_mode == CURSOR_SPELL_TARGET) {
                        rc->human_input.cursor_mode = CURSOR_NORMAL;
                    }
                    rc->human_input.pending_move_x = best_wx;
                    rc->human_input.pending_move_y = best_wy;
                    human_input_queue_walk(&rc->human_input, best_wx, best_wy);
                    human_set_click_cross(&rc->human_input, mx, my, 0);
                }
                } /* end else (ground click) */
            } else {
                human_handle_ground_click(&rc->human_input, mx, my,
                                           rc->arena_base_x, rc->arena_base_y,
                                           rc->arena_width, rc->arena_height,
                                           rc->entities, rc->entity_count,
                                           rc->gui.gui_entity_idx,
                                           render_can_human_attack_entity,
                                           &attack_ctx,
                                           RENDER_TILE_SIZE, RENDER_HEADER_HEIGHT);
            }
        }
    }

    /* right-click: open context menu (human mode) or cancel spell targeting */
    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
        if (rc->human_input.enabled) {
            int rmx = GetMouseX();
            int rmy = GetMouseY();
            /* only open menu in game area (left of GUI panel) */
            if (rmx < rc->gui.panel_x) {
                /* cancel spell targeting on right-click (OSRS behavior) */
                if (rc->human_input.cursor_mode == CURSOR_SPELL_TARGET)
                    rc->human_input.cursor_mode = CURSOR_NORMAL;
                context_menu_build(rc, env, rmx, rmy);
            } else {
                context_menu_dismiss(&rc->context_menu);
            }
        } else if (rc->human_input.cursor_mode == CURSOR_SPELL_TARGET) {
            rc->human_input.cursor_mode = CURSOR_NORMAL;
        }
    }

    /* hover tile: raycast from cursor to ground every frame (3D mode only).
       uses the same ray-box intersection as the click handler so the highlighted
       tile always matches what a click would select. */
    rc->hover_tile_x = -1;
    rc->hover_tile_y = -1;
    if (rc->mode_3d) {
        int hmx = GetMouseX();
        int hmy = GetMouseY();
        /* only raycast inside the game area (left of GUI panel) */
        if (hmx >= 0 && hmx < rc->gui.panel_x &&
            hmy >= 0 && hmy < RENDER_WINDOW_H) {
            Camera3D hcam = render_build_3d_camera(rc);
            Ray hray = GetScreenToWorldRay((Vector2){ (float)hmx, (float)hmy }, hcam);
            float best_dist = 1e30f;
            for (int dy = 0; dy < rc->arena_height; dy++) {
                for (int dx = 0; dx < rc->arena_width; dx++) {
                    int wx = rc->arena_base_x + dx;
                    int wy = rc->arena_base_y + dy;
                    float tx = (float)wx;
                    float tz = -(float)(wy + 1);
                    float ground_y = rc->terrain
                        ? terrain_height_avg(rc->terrain, wx, wy)
                        : 2.0f;
                    BoundingBox box = {
                        .min = { tx, ground_y - 0.1f, tz },
                        .max = { tx + 1.0f, ground_y, tz + 1.0f },
                    };
                    RayCollision col = GetRayCollisionBox(hray, box);
                    if (col.hit && col.distance < best_dist) {
                        best_dist = col.distance;
                        rc->hover_tile_x = wx;
                        rc->hover_tile_y = wy;
                    }
                }
            }
        }
    }
}

/* ======================================================================== */
/* rewind history                                                            */
/* ======================================================================== */

/* save current env state to history ring buffer (call after each pvp_step) */
static void render_save_snapshot(RenderClient* rc, OsrsEnv* env) {
    if (rc->history_count < RENDER_HISTORY_SIZE) {
        rc->history[rc->history_count] = *env;
        rc->history_count++;
    }
    /* if buffer full, stop recording (2000 ticks is plenty for one episode) */
}

/* restore env state from history snapshot, preserving render-side pointers */
static void render_restore_snapshot(RenderClient* rc, OsrsEnv* env) {
    if (rc->history_cursor < 0 || rc->history_cursor >= rc->history_count) return;

    void* saved_client = env->client;
    void* saved_cmap = env->collision_map;
    float* saved_ocean_obs = env->ocean_io.agent_obs;
    int* saved_ocean_acts = env->ocean_io.agent_actions;
    float* saved_ocean_rew = env->ocean_io.agent_rewards;
    unsigned char* saved_ocean_term = env->ocean_io.agent_terminals;

    *env = rc->history[rc->history_cursor];

    env->client = saved_client;
    env->collision_map = saved_cmap;
    env->ocean_io.agent_obs = saved_ocean_obs;
    env->ocean_io.agent_actions = saved_ocean_acts;
    env->ocean_io.agent_rewards = saved_ocean_rew;
    env->ocean_io.agent_terminals = saved_ocean_term;
}

/* reset history (call on episode reset) */
static void render_clear_history(RenderClient* rc) {
    rc->history_count = 0;
    rc->history_cursor = -1;
}

/* forward declaration: render_push_splat used by render_post_tick, defined later */
static void render_push_splat(RenderClient* rc, int damage, int pidx);

/* ======================================================================== */
/* entity population                                                         */
/* ======================================================================== */

/* populate rc->entities from env->players (legacy) or encounter vtable.
   call before render_post_tick and pvp_render so all draw code uses rc->entities.
   uses fill_render_entities when available, falls back to get_entity + cast. */
static void render_populate_entities(RenderClient* rc, OsrsEnv* env) {
    if (env->encounter_def && env->encounter_state) {
        const EncounterDef* def = (const EncounterDef*)env->encounter_def;
        if (def->fill_render_entities) {
            int count = 0;
            def->fill_render_entities(env->encounter_state, rc->entities, MAX_RENDER_ENTITIES, &count);
            rc->entity_count = count;
            /* detect Zuk presence for object variant swap */
            rc->zuk_active = 0;
            for (int zi = 0; zi < count; zi++) {
                if (rc->entities[zi].npc_def_id == 7706) { rc->zuk_active = 1; break; }
            }
            /* debug: print entity info on first populate */
            static int debug_once = 1;
            if (debug_once && count > 0) {
                debug_once = 0;
                fprintf(stderr, "render_populate: %d entities\n", count);
                for (int di = 0; di < count && di < 5; di++) {
                    fprintf(stderr, "  [%d] type=%d npc_id=%d visible=%d size=%d pos=(%d,%d) hp=%d/%d\n",
                            di, rc->entities[di].entity_type, rc->entities[di].npc_def_id,
                            rc->entities[di].npc_visible, rc->entities[di].npc_size,
                            rc->entities[di].x, rc->entities[di].y,
                            rc->entities[di].current_hitpoints, rc->entities[di].base_hitpoints);
                }
            }
        } else {
            /* legacy fallback: cast get_entity to Player* */
            int count = def->get_entity_count(env->encounter_state);
            if (count > MAX_RENDER_ENTITIES) count = MAX_RENDER_ENTITIES;
            rc->entity_count = count;
            for (int i = 0; i < count; i++) {
                Player* p = (Player*)def->get_entity(env->encounter_state, i);
                if (p) render_entity_from_player(p, &rc->entities[i]);
            }
        }
        /* override arena bounds from encounter if set */
        if (def->arena_width > 0 && def->arena_height > 0) {
            rc->arena_base_x = def->arena_base_x;
            rc->arena_base_y = def->arena_base_y;
            rc->arena_width = def->arena_width;
            rc->arena_height = def->arena_height;
        }
    } else {
        rc->entity_count = NUM_AGENTS;
        for (int i = 0; i < NUM_AGENTS; i++) {
            render_entity_from_player(&env->players[i], &rc->entities[i]);
        }
    }
}

/* ======================================================================== */
/* tick notification: position tracking, facing, effects                      */
/* ======================================================================== */

/**
 * Call BEFORE pvp_step to record pre-tick positions for movement direction.
 */
static void render_pre_tick(RenderClient* rc, OsrsEnv* env) {
    (void)rc; (void)env;
    /* destination is updated in post_tick after positions change */
}

/**
 * Call AFTER pvp_step to update movement destination and facing direction.
 *
 * Movement model matches OSRS client (Entity.java nextStep):
 * - positions stored as sub-tile coords (128 units per tile)
 * - each client frame, visual position moves toward destination at fixed speed
 * - walk = 4 sub-units/frame, run = 8 sub-units/frame (at 50 FPS client ticks)
 * - if distance > 256 sub-units (2 tiles), snap instantly (teleport)
 * - animation stalls (walkFlag=0) pause movement, then catch up at double speed
 */
static void render_post_tick(RenderClient* rc, OsrsEnv* env) {
    render_populate_entities(rc, env);

    /* detect entity identity changes from slot compaction (NPC deaths cause
       remaining NPCs to shift to lower indices). reset stale animation and
       composite state when a slot's NPC identity changes. */
    for (int i = 0; i < rc->entity_count; i++) {
        if (rc->entities[i].npc_slot != rc->prev_npc_slot[i]) {
            rc->anim[i].primary_seq_id = -1;
            rc->anim[i].primary_frame_idx = 0;
            rc->anim[i].primary_ticks = 0;
            rc->anim[i].primary_loops = 0;
            rc->anim[i].secondary_seq_id = -1;
            rc->anim[i].secondary_frame_idx = 0;
            rc->anim[i].secondary_ticks = 0;
            rc->composites[i].needs_rebuild = 1;
            /* flush hitsplat state so dying NPC's splats don't transfer
               to the NPC that shifts into this slot after death compaction */
            for (int s = 0; s < RENDER_SPLATS_PER_PLAYER; s++)
                rc->splats[i][s].active = 0;
            rc->hp_bar_visible_until[i] = 0;
            /* reset interpolation state — zeroed sub triggers the teleport-snap
               guard below, which cleanly snaps to the new entity's actual tile */
            rc->sub_x[i] = 0;
            rc->sub_y[i] = 0;
            rc->dest_x[i] = 0;
            rc->dest_y[i] = 0;
            rc->step_tracker[i] = 0;
        }
        rc->prev_npc_slot[i] = rc->entities[i].npc_slot;
    }
    for (int i = rc->entity_count; i < rc->prev_entity_count; i++)
        rc->prev_npc_slot[i] = -1;
    rc->prev_entity_count = rc->entity_count;

    for (int i = 0; i < rc->entity_count; i++) {
        RenderEntity* p = &rc->entities[i];

        /* convert game tile to sub-tile destination (128 units/tile, centered).
           the entity's (x,y) is the SW anchor tile. for size-1 entities,
           center on that tile (+ 64 sub-units). for NxN NPCs, center on
           the NxN footprint (offset by size/2 tiles from SW corner). */
        int size = p->npc_size > 1 ? p->npc_size : 1;
        int new_dest_x = p->x * 128 + size * 64;
        int new_dest_y = p->y * 128 + size * 64;

        /* NPC teleport: snap position when entity appears far from tracked position.
           this handles Zulrah dive→surface, new NPC spawns, and entity slot reuse.
           snap if distance > 2 tiles (matching deob client Canvas.method334 which
           snaps at >256 sub-units = 2 tiles). the 1-tile threshold was too aggressive
           and caused snapping during normal attack-anim stall catch-up. */
        if (p->entity_type == ENTITY_NPC && p->npc_visible) {
            /* distance to destination in tiles — uses dest center, not SW anchor,
               so large NPCs (size 5 shield) don't false-trigger the snap. */
            int tile_dx = abs(rc->sub_x[i] - new_dest_x) / 128;
            int tile_dy = abs(rc->sub_y[i] - new_dest_y) / 128;
            if (tile_dx > 2 || tile_dy > 2 || (rc->sub_x[i] == 0 && rc->sub_y[i] == 0)) {
                rc->sub_x[i] = new_dest_x;
                rc->sub_y[i] = new_dest_y;
                rc->dest_x[i] = new_dest_x;
                rc->dest_y[i] = new_dest_y;
            }
        }

        /* detect if player moved this tick (destination changed) */
        int moved = (new_dest_x != rc->dest_x[i] || new_dest_y != rc->dest_y[i]);

        /* update destination — NO snap-to-previous-dest. the real OSRS client
           (Canvas.java:165-188) simply advances actor.x toward dest by speed
           each client tick with no snap. sub smoothly interpolates from wherever
           it currently is toward the new dest. the dynamic walk speed in
           render_client_tick ensures arrival within one game tick. */
        rc->dest_x[i] = new_dest_x;
        rc->dest_y[i] = new_dest_y;

        /* latch walk/run state from the game — this drives the secondary
           animation selection in the client-tick loop. the game's is_running
           flag tells us whether the player was running this tick. */
        rc->visual_running[i] = p->is_running;

        /* update target facing direction (gradual turn applied in client tick).
           matches OSRS appendFocusDestination/nextStep priority:
           attacking/dead → face opponent (recalculated every client tick)
           moving → face movement direction (set once per step)
           idle → face opponent (recalculated every client tick) */
        if (p->attack_style_this_tick != ATTACK_STYLE_NONE ||
            p->current_hitpoints <= 0) {
            if (p->attack_target_entity_idx >= 0 || p->current_hitpoints <= 0) {
                rc->facing_opponent[i] = 1;
            } else {
                /* attacking non-entity target (nibbler → pillar): face dest tile */
                int face_x = p->dest_x * 128 + 64;
                int face_y = p->dest_y * 128 + 64;
                float dx = (float)(face_x - rc->sub_x[i]);
                float dy = (float)(face_y - rc->sub_y[i]);
                if (dx != 0.0f || dy != 0.0f)
                    rc->target_yaw[i] = atan2f(-dx, dy);
                rc->facing_opponent[i] = 0;
            }
        } else if (moved) {
            float dx = (float)(new_dest_x - rc->sub_x[i]);
            float dy = (float)(new_dest_y - rc->sub_y[i]);
            if (dx != 0.0f || dy != 0.0f) {
                rc->target_yaw[i] = atan2f(-dx, dy);
            }
            rc->facing_opponent[i] = 0;
        } else {
            if (p->attack_target_entity_idx >= 0) {
                rc->facing_opponent[i] = 1;
            } else {
                /* idle, no entity target: face dest tile (nibblers near pillars) */
                int face_x = p->dest_x * 128 + 64;
                int face_y = p->dest_y * 128 + 64;
                float dx = (float)(face_x - rc->sub_x[i]);
                float dy = (float)(face_y - rc->sub_y[i]);
                if (dx != 0.0f || dy != 0.0f)
                    rc->target_yaw[i] = atan2f(-dx, dy);
                rc->facing_opponent[i] = 0;
            }
        }

        /* shield always faces south (yaw = PI) */
        if (p->npc_def_id == 7707) {
            rc->target_yaw[i] = 3.14159265f;
            rc->facing_opponent[i] = 0;
        }

        /* HP bar + hitsplat: triggered once per game tick when a hit lands.
           HP bar: OSRS cycleStatus = clientTick + 300 (6s = 10 game ticks).
           hitsplat: one splat per hit, fills the next available slot (0-3). */
        if (p->hit_landed_this_tick) {
            rc->hp_bar_visible_until[i] = env->tick + 10;
            render_push_splat(rc, p->hit_damage, i);
        }
    }

    /* spawn visual effects (projectiles, spell impacts) based on this tick's events.
       works for any entity count — uses attack_target_entity_idx for multi-entity encounters. */
    int ct = rc->effect_client_tick_counter;
    for (int i = 0; i < rc->entity_count; i++) {
        RenderEntity* p = &rc->entities[i];
        /* resolve target: use attack_target_entity_idx if set, otherwise PvP fallback */
        int target_i;
        if (p->attack_target_entity_idx >= 0) {
            target_i = p->attack_target_entity_idx;
        } else if (rc->entity_count == 2) {
            target_i = 1 - i;
        } else {
            target_i = (i == 0) ? 1 : 0;
        }
        if (target_i < 0 || target_i >= rc->entity_count) continue;
        RenderEntity* t = &rc->entities[target_i];

        /* attacker projectile effects: only for PvP (no encounter overlay).
           encounters with render_post_tick handle their own projectiles via
           encounter_emit_projectile -> flight system. */
        int has_encounter_overlay = (env->encounter_def &&
            ((const EncounterDef*)env->encounter_def)->render_post_tick);

        if (!has_encounter_overlay) {
            /* attacker cast a spell this tick — spawn projectile */
            if (p->attack_style_this_tick == ATTACK_STYLE_MAGIC) {
                uint8_t wpn = p->equipped[GEAR_SLOT_WEAPON];
                int dist = render_pvp_distance_to_target(p, t);
                int duration_ticks = pvp_magic_hit_delay(dist) * 30;
                if (wpn == ITEM_TRIDENT_OF_SWAMP || wpn == ITEM_SANGUINESTI_STAFF ||
                    wpn == ITEM_EYE_OF_AYAK) {
                    /* trident/sang/ayak: powered staff projectile */
                    effect_spawn_projectile(rc->effects, GFX_TRIDENT_PROJ,
                        p->x, p->y, t->x, t->y,
                        0, duration_ticks, 40 * 4, 30 * 4, 16, ct,
                        rc->model_cache, rc->npc_model_cache);
                } else if (p->magic_type_this_tick == 1) {
                    /* ice barrage: projectile orb rises from target tile
                       heights *4 per reference (stream.readUnsignedByte() * 4) */
                    effect_spawn_projectile(rc->effects, GFX_ICE_BARRAGE_PROJ,
                        t->x, t->y, t->x, t->y,  /* src=dst (rises in place) */
                        0, 56, 43 * 4, 0, 16, ct,
                        rc->model_cache, rc->npc_model_cache);
                }
                /* blood barrage: no projectile, impact spawns on hit */
            }

            /* attacker fired a ranged attack this tick */
            if (p->attack_style_this_tick == ATTACK_STYLE_RANGED) {
                uint8_t wpn = p->equipped[GEAR_SLOT_WEAPON];
                int dist = render_pvp_distance_to_target(p, t);
                int gfx;
                if (wpn == ITEM_TOXIC_BLOWPIPE) {
                    gfx = GFX_DRAGON_DART;
                } else if (wpn == ITEM_MAGIC_SHORTBOW_I || wpn == ITEM_DARK_BOW ||
                           wpn == ITEM_BOW_OF_FAERDHINEN) {
                    gfx = GFX_RUNE_ARROW;
                } else if (wpn == ITEM_TWISTED_BOW) {
                    gfx = GFX_DRAGON_ARROW;
                } else {
                    gfx = GFX_BOLT;  /* crossbows, default */
                }
                int duration_ticks = p->used_special_this_tick
                    ? pvp_ranged_hit_delay_for_weapon(
                        dist, 1, render_pvp_ranged_spec_weapon_for_item(wpn)) * 30
                    : pvp_ranged_hit_delay(dist) * 30;
                /* heights *4 per reference: 43*4=172 start, 31*4=124 end */
                effect_spawn_projectile(rc->effects, gfx,
                    p->x, p->y, t->x, t->y,
                    0, duration_ticks, 43 * 4, 31 * 4, 16, ct,
                    rc->model_cache, rc->npc_model_cache);
            }
        }

        /* defender: check what landed on entity p this tick.
           for NPC defenders, the attacker is entity 0 (the player).
           for player (entity 0), attacker is the current target entity. */
        if (p->hit_landed_this_tick) {
            RenderEntity* att;
            if (i == 0) {
                att = t;  /* player was hit — attacker is target entity */
            } else {
                att = &rc->entities[0];  /* NPC was hit — attacker is player */
            }

            /* check if attacker used a powered staff (trident/sang/ayak) */
            uint8_t att_wpn = att->equipped[GEAR_SLOT_WEAPON];
            int att_is_powered_staff = (att_wpn == ITEM_TRIDENT_OF_SWAMP ||
                att_wpn == ITEM_SANGUINESTI_STAFF || att_wpn == ITEM_EYE_OF_AYAK);

            if (att_is_powered_staff && att->attack_style_this_tick == ATTACK_STYLE_MAGIC) {
                /* powered staff hit: trident impact splash */
                if (p->hit_was_successful) {
                    effect_spawn_spotanim(rc->effects, GFX_TRIDENT_IMPACT,
                        p->x, p->y, ct, rc->anim_cache,
                        rc->model_cache, rc->npc_model_cache);
                } else {
                    effect_spawn_spotanim(rc->effects, GFX_SPLASH,
                        p->x, p->y, ct, rc->anim_cache,
                        rc->model_cache, rc->npc_model_cache);
                }
            } else {
                /* barrage impact: use hit_spell_type (set when pending hit resolves)
                   instead of magic_type_this_tick (stale by deferred hit landing).
                   ENCOUNTER_SPELL_ICE=1 -> ice barrage, ENCOUNTER_SPELL_BLOOD=2 -> blood. */
                /* use hit_spell_type from pending hit resolution only. the magic_type_this_tick
                   fallback caused blood/ice effects on tbow hits when barrage fired same tick. */
                int spell = p->hit_spell_type;
                if (spell > 0) {
                    /* center effect on NPC footprint center using sub-tile precision.
                       for size 2: center at (x*128 + 128, y*128 + 128) = between 4 tiles.
                       for size 3: center at (x*128 + 192, y*128 + 192) = middle tile center. */
                    float fx = (float)p->x * 128.0f + (float)p->npc_size * 64.0f;
                    float fy = (float)p->y * 128.0f + (float)p->npc_size * 64.0f;
                    if (p->hit_was_successful) {
                        int gfx = (spell == 1)  /* ENCOUNTER_SPELL_ICE */
                            ? GFX_ICE_BARRAGE_HIT : GFX_BLOOD_BARRAGE_HIT;
                        effect_spawn_spotanim_subtile(rc->effects, gfx,
                            fx, fy, ct, rc->anim_cache,
                            rc->model_cache, rc->npc_model_cache);
                    } else {
                        effect_spawn_spotanim_subtile(rc->effects, GFX_SPLASH,
                            fx, fy, ct, rc->anim_cache,
                            rc->model_cache, rc->npc_model_cache);
                    }
                }
            }
        }
    }

    /* update encounter overlay (clouds, boss state) */
    if (env->encounter_def && env->encounter_state) {
        const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
        if (edef->render_post_tick) {
            edef->render_post_tick(env->encounter_state, &rc->encounter_overlay);

            /* spawn flight projectiles from overlay events.
               per-projectile params with backward-compat defaults. */
            EncounterOverlay* ov = &rc->encounter_overlay;
            for (int i = 0; i < ov->projectile_count; i++) {
                if (!ov->projectiles[i].active) continue;
                int src_sz = ov->projectiles[i].src_size > 0 ? ov->projectiles[i].src_size : ov->boss_size;
                int dst_sz = ov->projectiles[i].dst_size > 0 ? ov->projectiles[i].dst_size : 1;
                float sx = (float)ov->projectiles[i].src_x + (float)(src_sz - 1) / 2.0f + 0.5f;
                float sy = (float)ov->projectiles[i].src_y + (float)(src_sz - 1) / 2.0f + 0.5f;
                float dx = (float)ov->projectiles[i].dst_x + (float)(dst_sz - 1) / 2.0f + 0.5f;
                float dy = (float)ov->projectiles[i].dst_y + (float)(dst_sz - 1) / 2.0f + 0.5f;

                /* use per-projectile params, with defaults for backward compat */
                int dur = ov->projectiles[i].duration_ticks > 0 ? ov->projectiles[i].duration_ticks : 35;
                int sh  = ov->projectiles[i].start_h > 0 ? ov->projectiles[i].start_h : 85;
                int eh  = ov->projectiles[i].end_h > 0 ? ov->projectiles[i].end_h : 40;
                int cv  = ov->projectiles[i].curve > 0 ? ov->projectiles[i].curve : 16;
                float arc = ov->projectiles[i].arc_height;
                int trk = ov->projectiles[i].tracks_target;

                /* cloud/orb styles: offset dst to tile center */
                if (ov->projectiles[i].style == 3 || ov->projectiles[i].style == 4) {
                    dx += 0.5f;
                    dy += 0.5f;
                }

                flight_spawn(rc, sx, sy, dx, dy,
                    ov->projectiles[i].style, ov->projectiles[i].damage,
                    dur, sh, eh, cv, arc, trk,
                    ov->projectiles[i].model_id,
                    ov->projectiles[i].anim_id,
                    ov->projectiles[i].impact_gfx_id,
                    ov->projectiles[i].start_delay,
                    ov->projectiles[i].motion_mode,
                    ov->projectiles[i].offset_x,
                    ov->projectiles[i].offset_y,
                    ov->projectiles[i].offset_z);
            }

            /* update tracking projectile targets to player's current position */
            if (rc->entity_count > 0) {
                float px = (float)rc->entities[0].x;
                float py = (float)rc->entities[0].y;
                for (int fi = 0; fi < MAX_FLIGHT_PROJECTILES; fi++) {
                    if (rc->flights[fi].active && rc->flights[fi].tracks_target) {
                        rc->flights[fi].dst_x = px;
                        rc->flights[fi].dst_y = py;
                    }
                }
            }
        }
    }
}

/**
 * One client-tick step: movement + animation advancement.
 *
 * Matches OSRS client processMovement() (Client.java:12996) which calls
 * nextStep() then updateAnimation() once per 20ms client tick. By running
 * both movement and animation at the same rate, they stay perfectly in sync.
 *
 * Movement: faithful to Entity.nextStep() (Client.java:13074)
 * Animation: faithful to updateAnimation() (Client.java:13272)
 */
static void render_client_tick(RenderClient* rc, int player_idx) {
    /* --- nextStep: advance sub-tile position toward destination ---
       faithful to Entity.nextStep() (Client.java:13074-13213).

       when a non-melee animation is playing (walkFlag==0), sub-tile
       movement stalls. stepTracker accumulates stalled frames, then
       drives 2x catch-up speed once the animation ends. */
    int dx = rc->dest_x[player_idx] - rc->sub_x[player_idx];
    int dy = rc->dest_y[player_idx] - rc->sub_y[player_idx];

    if (dx == 0 && dy == 0) {
        /* not moving */
        rc->visual_moving[player_idx] = 0;
        rc->step_tracker[player_idx] = 0;
    } else {
        /* check movement stall: animations without interleave_order (cast,
           ranged, death) stall sub-tile movement. animations WITH interleave
           (melee, eat, block) allow walking — the interleave blends upper body
           attack with lower body walk. matches the real client behavior where
           the stall correlates with having no interleave_order.
           the 317 cache doesn't set walkFlag=0 for these, but modern OSRS
           clearly stalls movement during cast/ranged (confirmed from footage). */
        int stall = 0;
        if (rc->anim[player_idx].primary_seq_id >= 0 &&
            rc->anim[player_idx].primary_loops == 0 && rc->anim_cache) {
            AnimSequence* seq = render_get_anim_sequence(
                rc, (uint16_t)rc->anim[player_idx].primary_seq_id);
            if (seq && seq->interleave_count == 0) {
                stall = 1;
            }
        }

        if (stall) {
            rc->step_tracker[player_idx]++;
            rc->visual_moving[player_idx] = 0;
        } else {
            rc->visual_moving[player_idx] = 1;

            /* base speed: floor division so entities NEVER arrive early.
               real OSRS uses constant speed 4 (walk) / 8 (run) su/ct at 30 ct/gt.
               floor(128/30)=4 means 32 ticks to traverse — entity is always in motion
               and never stalls at tile center waiting for next game tick. */
            float tps = rc->ticks_per_second > 0.0f ? rc->ticks_per_second : 50.0f;
            int ct_per_gt = (int)(50.0f / tps + 0.5f);  /* round, not truncate */
            if (ct_per_gt < 1) ct_per_gt = 1;
            int base_walk = 128 / ct_per_gt;  /* floor, not ceil */
            if (base_walk < 1) base_walk = 1;
            int speed = rc->visual_running[player_idx] ? base_walk * 2 : base_walk;

            /* catch-up: double speed while step_tracker > 0 (one stalled
               frame recovered per catch-up frame). */
            if (rc->step_tracker[player_idx] > 0) {
                speed *= 2;
                rc->step_tracker[player_idx]--;
            }

            if (dx > 0) rc->sub_x[player_idx] += (dx > speed) ? speed : dx;
            else if (dx < 0) rc->sub_x[player_idx] += (dx < -speed) ? -speed : dx;

            if (dy > 0) rc->sub_y[player_idx] += (dy > speed) ? speed : dy;
            else if (dy < 0) rc->sub_y[player_idx] += (dy < -speed) ? -speed : dy;

            /* when walking (not facing opponent), update target_yaw to movement
               direction each client tick, matching nextStep's turnDirection
               assignment from step delta. */
            if (!rc->facing_opponent[player_idx] && rc->entities[player_idx].npc_def_id != 7707) {
                float fdx = (float)dx;
                float fdy = (float)dy;
                if (fdx != 0.0f || fdy != 0.0f) {
                    rc->target_yaw[player_idx] = atan2f(-fdx, fdy);
                }
            }
        }
    }

    /* --- appendFocusDestination: gradual turn toward target yaw ---
       matches Entity.appendFocusDestination (Client.java:13215).
       turn rate = 32 / 2048 of a full circle per client tick.
       when facing opponent, recompute target_yaw from visual positions
       every client tick (reference: interactingEntity != -1 path). */
    {
        if (rc->facing_opponent[player_idx]) {
            /* recompute target yaw from current visual positions each client tick,
               matching how appendFocusDestination recalculates from live coords */
            int opp;
            if (rc->entities[player_idx].attack_target_entity_idx >= 0) {
                opp = rc->entities[player_idx].attack_target_entity_idx;
            } else {
                opp = (rc->entity_count == 2) ? (1 - player_idx) : (player_idx == 0 ? 1 : 0);
            }
            float dx = (float)(rc->sub_x[opp] - rc->sub_x[player_idx]);
            float dy = (float)(rc->sub_y[opp] - rc->sub_y[player_idx]);
            if (dx != 0.0f || dy != 0.0f) {
                rc->target_yaw[player_idx] = atan2f(-dx, dy);
            }
        }

        /* step current yaw toward target by turn_speed per client tick.
           32 / 2048 * 2π ≈ 0.0982 radians. snap if within turn_speed. */
        float turn_speed = 32.0f / 2048.0f * 2.0f * 3.14159265f;
        float diff = rc->target_yaw[player_idx] - rc->yaw[player_idx];

        /* normalize to [-π, π] for shortest-path turning */
        while (diff > 3.14159265f) diff -= 2.0f * 3.14159265f;
        while (diff < -3.14159265f) diff += 2.0f * 3.14159265f;

        if (fabsf(diff) <= turn_speed) {
            rc->yaw[player_idx] = rc->target_yaw[player_idx];
        } else if (diff > 0.0f) {
            rc->yaw[player_idx] += turn_speed;
        } else {
            rc->yaw[player_idx] -= turn_speed;
        }

        /* normalize yaw to [-π, π] */
        while (rc->yaw[player_idx] > 3.14159265f) rc->yaw[player_idx] -= 2.0f * 3.14159265f;
        while (rc->yaw[player_idx] < -3.14159265f) rc->yaw[player_idx] += 2.0f * 3.14159265f;
    }

    /* --- updateAnimation: advance both animation tracks --- */

    /* secondary (pose): select based on visual movement state.
       NPCs switch between idle and walk animations on the secondary track
       (matching real OSRS client — walk/idle are secondary, attacks are primary).
       this prevents the stall mechanism from freezing movement during walk. */
    int new_secondary;
    if (rc->entities[player_idx].entity_type == ENTITY_NPC) {
        const NpcModelMapping* nm = npc_model_lookup(
            (uint16_t)rc->entities[player_idx].npc_def_id);
        if (nm) {
            new_secondary = rc->visual_moving[player_idx]
                ? (nm->walk_anim != 65535 ? (int)nm->walk_anim : (int)nm->idle_anim)
                : (int)nm->idle_anim;
        } else {
            new_secondary = -1;
        }
    } else {
        new_secondary = render_select_secondary(rc, player_idx);
    }
    if (rc->anim[player_idx].secondary_seq_id != new_secondary) {
        rc->anim[player_idx].secondary_seq_id = new_secondary;
        rc->anim[player_idx].secondary_frame_idx = 0;
        rc->anim[player_idx].secondary_ticks = 0;
    }

    /* advance secondary frame timing */
    if (rc->anim_cache && rc->anim[player_idx].secondary_seq_id >= 0) {
        AnimSequence* seq = render_get_anim_sequence(
            rc, (uint16_t)rc->anim[player_idx].secondary_seq_id);
        if (seq && seq->frame_count > 0) {
            int fidx = rc->anim[player_idx].secondary_frame_idx % seq->frame_count;
            int delay = seq->frames[fidx].delay > 0 ? seq->frames[fidx].delay : 1;
            rc->anim[player_idx].secondary_ticks++;
            if (rc->anim[player_idx].secondary_ticks >= delay) {
                rc->anim[player_idx].secondary_ticks = 0;
                rc->anim[player_idx].secondary_frame_idx =
                    (fidx + 1) % seq->frame_count;
            }
        }
    }

    /* advance primary frame timing (if active) */
    if (rc->anim_cache && rc->anim[player_idx].primary_seq_id >= 0) {
        AnimSequence* seq = render_get_anim_sequence(
            rc, (uint16_t)rc->anim[player_idx].primary_seq_id);
        if (seq && seq->frame_count > 0) {
            int fidx = rc->anim[player_idx].primary_frame_idx % seq->frame_count;
            int delay = seq->frames[fidx].delay > 0 ? seq->frames[fidx].delay : 1;
            rc->anim[player_idx].primary_ticks++;
            if (rc->anim[player_idx].primary_ticks >= delay) {
                rc->anim[player_idx].primary_ticks = 0;
                int next = (fidx + 1) % seq->frame_count;
                rc->anim[player_idx].primary_frame_idx = next;
                /* detect loop completion (wrapped back to 0) */
                if (next == 0) {
                    rc->anim[player_idx].primary_loops++;
                }
            }
        }
    }
}

/**
 * Get world position from sub-tile coordinates (128 units = 1 tile).
 */
static void render_get_visual_pos(
    RenderClient* rc, int player_idx,
    float* out_x, float* out_z, float* out_ground
) {
    /* convert sub-tile to world (128 units per tile) */
    float tile_x = (float)rc->sub_x[player_idx] / 128.0f;
    float tile_y = (float)rc->sub_y[player_idx] / 128.0f;

    *out_x = tile_x;
    *out_z = -tile_y;

    if (rc->terrain) {
        *out_ground = terrain_height_avg(rc->terrain,
            (int)tile_x, (int)tile_y);
    } else {
        *out_ground = 2.0f;
    }
}

/* ======================================================================== */
/* hit splats                                                                */
/* ======================================================================== */

/* advance splat animation by one client tick (20ms).
   exact OSRS logic from Client.java:6107-6143 (mode 2 animated):
   - hitmarkMove starts at +5.0, decrements by 0.25 until -5.0 (40 ticks to settle)
   - hitmarkTrans starts at 230, stays there (mode 2 clamp means fade at -26 never fires)
   - hitsLoopCycle expires after 70 client ticks → splat just disappears */
static void render_update_splats_client_tick(RenderClient* rc) {
    for (int p = 0; p < rc->entity_count; p++) {
        for (int i = 0; i < RENDER_SPLATS_PER_PLAYER; i++) {
            HitSplat* s = &rc->splats[p][i];
            if (!s->active) continue;
            /* OSRS splats stay in place — no vertical drift */
            s->ticks_remaining--;
            if (s->ticks_remaining <= 0) {
                s->active = 0;
            }
        }
    }
}

/* OSRS Entity.damage(): find first expired slot, init with standard values */
static void render_push_splat(RenderClient* rc, int damage, int pidx) {
    for (int i = 0; i < RENDER_SPLATS_PER_PLAYER; i++) {
        if (!rc->splats[pidx][i].active) {
            rc->splats[pidx][i] = (HitSplat){
                .active = 1,
                .damage = damage,
                .hitmark_move = 5.0,
                .hitmark_trans = 230,
                .ticks_remaining = 70,
            };
            return;
        }
    }
    /* all 4 slots full: overwrite the one closest to expiry */
    int oldest = 0;
    for (int i = 1; i < RENDER_SPLATS_PER_PLAYER; i++) {
        if (rc->splats[pidx][i].ticks_remaining < rc->splats[pidx][oldest].ticks_remaining)
            oldest = i;
    }
    rc->splats[pidx][oldest] = (HitSplat){
        .active = 1,
        .damage = damage,
        .hitmark_move = 5.0,
        .hitmark_trans = 230,
        .ticks_remaining = 70,
    };
}


/* ======================================================================== */
/* drawing: grid                                                             */
/* ======================================================================== */

static void render_draw_grid(RenderClient* rc, OsrsEnv* env) {
    const CollisionMap* cmap = rc->collision_map;
    int ts = RENDER_TILE_SIZE;

    for (int dx = 0; dx < rc->arena_width; dx++) {
        for (int dy = 0; dy < rc->arena_height; dy++) {
            int wx = rc->arena_base_x + dx;
            int wy = rc->arena_base_y + dy;
            int sx = render_world_to_screen_x_rc(rc, wx);
            int sy = render_world_to_screen_y_rc(rc, wy);

            /* collision overlay */
            if (rc->show_collision && cmap != NULL) {
                int flags = collision_get_flags(cmap, 0,
                    wx + rc->collision_world_offset_x,
                    wy + rc->collision_world_offset_y);
                if (flags & COLLISION_BLOCKED) {
                    DrawRectangle(sx, sy, ts, ts, COLOR_BLOCKED);
                } else if (flags & COLLISION_BRIDGE) {
                    DrawRectangle(sx, sy, ts, ts, COLOR_BRIDGE);
                } else if (flags & 0x0FF) {
                    /* any wall flag in lower byte */
                    DrawRectangle(sx, sy, ts, ts, COLOR_WALL);
                }

                /* wall segment lines (2px on tile edges) */
                if (flags & (COLLISION_WALL_NORTH | COLLISION_IMPENETRABLE_WALL_NORTH))
                    DrawRectangle(sx, sy, ts, 2, COLOR_WALL_LINE);
                if (flags & (COLLISION_WALL_SOUTH | COLLISION_IMPENETRABLE_WALL_SOUTH))
                    DrawRectangle(sx, sy + ts - 2, ts, 2, COLOR_WALL_LINE);
                if (flags & (COLLISION_WALL_WEST | COLLISION_IMPENETRABLE_WALL_WEST))
                    DrawRectangle(sx, sy, 2, ts, COLOR_WALL_LINE);
                if (flags & (COLLISION_WALL_EAST | COLLISION_IMPENETRABLE_WALL_EAST))
                    DrawRectangle(sx + ts - 2, sy, 2, ts, COLOR_WALL_LINE);
            }

            /* encounter arena: color tiles based on encounter type */
            if (env->encounter_def && cmap == NULL) {
                if (rc->npc_model_cache) {
                    /* inferno: dark cave floor */
                    int shade = 18 + ((dx * 7 + dy * 13) % 10);
                    DrawRectangle(sx, sy, ts, ts, CLITERAL(Color){
                        (unsigned char)(shade + 3), (unsigned char)(shade - 1),
                        (unsigned char)(shade - 3), 255 });
                } else {
                    /* zulrah: platform vs water */
                    int on_plat = (dx >= ZUL_PLATFORM_MIN && dx <= ZUL_PLATFORM_MAX &&
                                   dy >= ZUL_PLATFORM_MIN && dy <= ZUL_PLATFORM_MAX);
                    if (on_plat)
                        DrawRectangle(sx, sy, ts, ts, CLITERAL(Color){ 30, 60, 30, 255 });
                    else
                        DrawRectangle(sx, sy, ts, ts, CLITERAL(Color){ 20, 30, 50, 255 });
                }
            }

            /* grid lines */
            DrawRectangleLines(sx, sy, ts, ts, COLOR_GRID);
        }
    }

    /* encounter overlay: area hazards and encounter adds in 2D */
    if (env->encounter_def) {
        EncounterOverlay* ov = &rc->encounter_overlay;

        /* current hazard renderer: 3x3 poison cloud footprint from the SW tile. */
        for (int i = 0; i < ov->hazard_count; i++) {
            if (!ov->hazards[i].active) continue;
            for (int cdx = 0; cdx < 3; cdx++) {
                for (int cdy = 0; cdy < 3; cdy++) {
                    int cx = ov->hazards[i].x + cdx;
                    int cy = ov->hazards[i].y + cdy;
                    int csx = render_world_to_screen_x_rc(rc, cx);
                    int csy = render_world_to_screen_y_rc(rc, cy);
                    DrawRectangle(csx, csy, ts, ts, CLITERAL(Color){ 60, 140, 40, 120 });
                    /* darker border for tile definition */
                    DrawRectangleLines(csx, csy, ts, ts, CLITERAL(Color){ 40, 100, 20, 150 });
                }
            }
        }

        /* boss hitbox: NxN form-colored tiles */
        if (rc->show_debug && ov->boss_visible && ov->boss_size > 0) {
            Color form_col;
            switch (ov->boss_form) {
                case 0: form_col = CLITERAL(Color){ 50, 200, 50, 60 }; break;
                case 1: form_col = CLITERAL(Color){ 200, 50, 50, 60 }; break;
                case 2: form_col = CLITERAL(Color){ 50, 100, 255, 60 }; break;
                default: form_col = CLITERAL(Color){ 200, 200, 200, 60 }; break;
            }
            for (int bx = 0; bx < ov->boss_size; bx++) {
                for (int by = 0; by < ov->boss_size; by++) {
                    int bsx = render_world_to_screen_x_rc(rc, ov->boss_x + bx);
                    int bsy = render_world_to_screen_y_rc(rc, ov->boss_y + by);
                    DrawRectangle(bsx, bsy, ts, ts, form_col);
                }
            }
            /* border */
            Color border = form_col; border.a = 200;
            int bsx0 = render_world_to_screen_x_rc(rc, ov->boss_x);
            int bsy0 = render_world_to_screen_y_rc(rc, ov->boss_y);
            DrawRectangleLines(bsx0, bsy0, ts * ov->boss_size, ts * ov->boss_size, border);
        }

        /* melee target tile */
        if (ov->melee_target_active) {
            int msx = render_world_to_screen_x_rc(rc, ov->melee_target_x);
            int msy = render_world_to_screen_y_rc(rc, ov->melee_target_y);
            DrawRectangle(msx, msy, ts, ts, CLITERAL(Color){ 255, 50, 50, 120 });
            DrawRectangleLines(msx, msy, ts, ts, CLITERAL(Color){ 255, 50, 50, 220 });
        }

        /* encounter adds: current renderer uses Zulrah snakeling colors. */
        for (int i = 0; i < ov->add_count; i++) {
            if (!ov->adds[i].active) continue;
            int ssx = render_world_to_screen_x_rc(rc, ov->adds[i].x);
            int ssy = render_world_to_screen_y_rc(rc, ov->adds[i].y);
            Color sc = ov->adds[i].variant
                ? CLITERAL(Color){ 100, 150, 255, 200 }
                : CLITERAL(Color){ 255, 150, 50, 200 };
            DrawRectangle(ssx + 3, ssy + 3, ts - 6, ts - 6, sc);
        }

        /* in-flight projectiles (interpolated at 50 Hz) */
        for (int i = 0; i < MAX_FLIGHT_PROJECTILES; i++) {
            FlightProjectile* fp = &rc->flights[i];
            if (!fp->active || fp->start_delay > 0) continue;
            float t = fp->progress;
            float cur_x = fp->motion_mode == ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED
                ? fp->dst_x : fp->src_x + (fp->dst_x - fp->src_x) * t;
            float cur_y = fp->motion_mode == ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED
                ? fp->dst_y : fp->src_y + (fp->dst_y - fp->src_y) * t;
            int psx = render_world_to_screen_x_rc(rc, (int)fp->src_x) + ts / 2;
            int psy = render_world_to_screen_y_rc(rc, (int)fp->src_y) + ts / 2;
            int pcx = render_world_to_screen_x_rc(rc, (int)cur_x) + ts / 2;
            int pcy = render_world_to_screen_y_rc(rc, (int)cur_y) + ts / 2;
            Color pc;
            switch (fp->style) {
                case 0: pc = CLITERAL(Color){ 80, 220, 80, 255 }; break;
                case 1: pc = CLITERAL(Color){ 80, 130, 255, 255 }; break;
                case 2: pc = CLITERAL(Color){ 255, 80, 80, 255 }; break;
                default: pc = WHITE; break;
            }
            if (fp->motion_mode != ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED)
                DrawLine(psx, psy, pcx, pcy, pc);
            DrawCircle(pcx, pcy, 4.0f, pc);
        }
    }
}

/* ======================================================================== */
/* drawing: players                                                          */
/* ======================================================================== */

static const char* render_prayer_label(OverheadPrayer p) {
    switch (p) {
        case PRAYER_PROTECT_MAGIC:  return "Ma";
        case PRAYER_PROTECT_RANGED: return "Ra";
        case PRAYER_PROTECT_MELEE:  return "Me";
        case PRAYER_SMITE:          return "Sm";
        case PRAYER_REDEMPTION:     return "Re";
        default: return NULL;
    }
}

static const char* render_gear_label(GearSet g) {
    switch (g) {
        case GEAR_MELEE:  return "M";
        case GEAR_RANGED: return "R";
        case GEAR_MAGE:   return "Ma";
        case GEAR_SPEC:   return "S";
        case GEAR_TANK:   return "T";
        default: return "?";
    }
}

/* draw zulrah safe spot markers on the 2D grid.
   S key toggles. shows all 15 stand locations as colored diamonds,
   with the current phase's active stand/stall highlighted. */
static void render_draw_safe_spots(RenderClient* rc, OsrsEnv* env) {
    if (!env->encounter_def || !env->encounter_state) return;
    const EncounterDef* edef = (const EncounterDef*)env->encounter_def;
    if (strcmp(edef->name, "zulrah") != 0) return;

    int ts = RENDER_TILE_SIZE;
    ZulrahState* zs = (ZulrahState*)env->encounter_state;

    /* current phase's active stand + stall */
    const ZulRotationPhase* phase = zul_current_phase(zs);
    int active_stand = phase->stand;
    int active_stall = phase->stall;

    for (int i = 0; i < ZUL_NUM_STAND_LOCATIONS; i++) {
        int lx = ZUL_STAND_COORDS[i][0];
        int ly = ZUL_STAND_COORDS[i][1];
        int sx = render_world_to_screen_x_rc(rc, rc->arena_base_x + lx);
        int sy = render_world_to_screen_y_rc(rc, rc->arena_base_y + ly);

        /* color: green for active stand, yellow for active stall,
           dim cyan for inactive spots */
        Color col;
        if (i == active_stand)
            col = (Color){0, 255, 0, 180};
        else if (i == active_stall)
            col = (Color){255, 255, 0, 180};
        else
            col = (Color){0, 180, 180, 80};

        /* draw diamond shape */
        int cx = sx + ts / 2, cy = sy + ts / 2;
        int r = ts / 2 - 1;
        DrawTriangle(
            (Vector2){(float)cx, (float)(cy - r)},
            (Vector2){(float)(cx - r), (float)cy},
            (Vector2){(float)(cx + r), (float)cy}, col);
        DrawTriangle(
            (Vector2){(float)(cx - r), (float)cy},
            (Vector2){(float)cx, (float)(cy + r)},
            (Vector2){(float)(cx + r), (float)cy}, col);

        /* label with index */
        DrawText(TextFormat("%d", i), sx + 2, sy + 2, 8, WHITE);
    }
}

static void render_draw_players(RenderClient* rc) {
    int ts = RENDER_TILE_SIZE;

    for (int i = 0; i < rc->entity_count; i++) {
        RenderEntity* p = &rc->entities[i];
        Color color = (i == 0) ? COLOR_P0 : COLOR_P1;
        int sx = render_world_to_screen_x_rc(rc, p->x);
        int sy = render_world_to_screen_y_rc(rc, p->y);
        int inset = 3;

        /* NPC coloring: form-specific for Zulrah, red for snakelings */
        if (p->entity_type == ENTITY_NPC) {
            if (p->npc_def_id == 2042)      color = GREEN;
            else if (p->npc_def_id == 2043) color = RED;
            else if (p->npc_def_id == 2044) color = CLITERAL(Color){ 80, 140, 255, 255 };
            else                            color = CLITERAL(Color){ 200, 50, 50, 200 };

            /* skip invisible NPCs (e.g. diving Zulrah) */
            if (!p->npc_visible) continue;
        }

        /* player/entity body */
        DrawRectangle(sx + inset, sy + inset, ts - inset * 2, ts - inset * 2, color);

        /* freeze overlay */
        if (p->frozen_ticks > 0) {
            DrawRectangle(sx, sy, ts, ts, COLOR_FREEZE);
        }

        /* HP bar (above tile) */
        int bar_w = ts - 2;
        int bar_h = 3;
        int bar_x = sx + 1;
        int bar_y = sy - bar_h - 1;
        float hp_frac = (float)p->current_hitpoints / (float)p->base_hitpoints;
        if (hp_frac < 0.0f) hp_frac = 0.0f;
        if (hp_frac > 1.0f) hp_frac = 1.0f;

        Color hp_color = (hp_frac > 0.5f) ? COLOR_HP_GREEN : COLOR_HP_RED;
        DrawRectangle(bar_x, bar_y, bar_w, bar_h, COLOR_HP_BG);
        DrawRectangle(bar_x, bar_y, (int)(bar_w * hp_frac), bar_h, hp_color);

        /* spec energy bar (thin, below HP bar) */
        float spec_frac = p->special_energy / 100.0f;
        DrawRectangle(bar_x, bar_y - 2, (int)(bar_w * spec_frac), 1, COLOR_SPEC_BAR);

        /* overhead prayer label (above HP bar) */
        const char* pray_lbl = render_prayer_label(p->prayer);
        if (pray_lbl) {
            DrawText(pray_lbl, sx + 2, bar_y - 10, 8, WHITE);
        }

        /* gear label (inside tile, bottom) */
        const char* gear_lbl = render_gear_label(p->visible_gear);
        DrawText(gear_lbl, sx + 2, sy + ts - 10, 8, WHITE);

        /* veng indicator */
        if (p->veng_active) {
            DrawText("V", sx + ts - 8, sy + 1, 8, COLOR_VENG);
        }
    }
}

/* ======================================================================== */
/* drawing: destination markers                                              */
/* ======================================================================== */

static void render_draw_dest_markers(RenderClient* rc) {
    int ts = RENDER_TILE_SIZE;

    for (int i = 0; i < rc->entity_count; i++) {
        RenderEntity* p = &rc->entities[i];
        Color dest_color = (i == 0) ? COLOR_P0_LIGHT : COLOR_P1_LIGHT;
        if (p->dest_x != p->x || p->dest_y != p->y) {
            int sx = render_world_to_screen_x_rc(rc, p->dest_x);
            int sy = render_world_to_screen_y_rc(rc, p->dest_y);
            DrawRectangleLines(sx + 1, sy + 1, ts - 2, ts - 2, dest_color);
        }
    }
}

/* ======================================================================== */
/* drawing: splats                                                           */
/* ======================================================================== */

/* draw a hitsplat using the actual cache sprites (317 mode 0).
   Client.java:6052-6073: hitMarks[type].drawSprite(spriteDrawX - 12, spriteDrawY - 12)
   then smallFont.drawText centered on the sprite.
   sprite index: 0=blue(miss), 1=red(regular hit). sprites are 24x23px. */
static void render_draw_hitmark(RenderClient* rc, int cx, int cy, int damage, int opacity) {
    unsigned char a = (unsigned char)(opacity > 255 ? 255 : (opacity < 0 ? 0 : opacity));
    int sprite_idx = (damage > 0) ? 1 : 0;  /* red for hits, blue for misses */

    if (rc->hitmark_sprites_loaded) {
        /* draw the actual cache sprite, centered at (cx, cy).
           OSRS draws at spriteDrawX-12, spriteDrawY-12 (centering a 24x23 sprite) */
        Texture2D tex = rc->hitmark_sprites[sprite_idx];
        float draw_x = (float)cx - (float)tex.width / 2.0f;
        float draw_y = (float)cy - (float)tex.height / 2.0f;
        DrawTexture(tex, (int)draw_x, (int)draw_y, (Color){ 255, 255, 255, a });
    } else {
        /* fallback: colored circle if sprites missing */
        Color bg = (damage > 0) ? (Color){ 175, 25, 25, a } : (Color){ 65, 105, 225, a };
        DrawCircle(cx, cy, 12.0f, bg);
    }

    /* damage number: white text with black shadow, centered on the sprite.
       OSRS Client.java:6070-6071: smallFont.drawText at spriteDrawY+5, spriteDrawX */
    const char* txt = TextFormat("%d", damage);
    int tw = MeasureText(txt, 10);
    DrawText(txt, cx - tw / 2 + 1, cy - 4, 10, (Color){ 0, 0, 0, a });
    DrawText(txt, cx - tw / 2, cy - 5, 10, (Color){ 255, 255, 255, a });
}

/* slot offset layout from Client.java:6052-6072 (mode 0, used across modes):
   slot 0: center
   slot 1: up 20px
   slot 2: left 15px, up 10px
   slot 3: right 15px, up 10px */
static void render_splat_slot_offset(int slot, int* dx, int* dy) {
    switch (slot) {
        case 0: *dx = 0;   *dy = 0;   break;
        case 1: *dx = 0;   *dy = -20; break;
        case 2: *dx = -15; *dy = -10; break;
        case 3: *dx = 15;  *dy = -10; break;
        default: *dx = 0;  *dy = 0;   break;
    }
}

/* 2D mode: draw splats at entity tile positions */
static void render_draw_splats_2d(RenderClient* rc) {
    for (int p = 0; p < rc->entity_count; p++) {
        RenderEntity* pl = &rc->entities[p];
        int base_x = render_world_to_screen_x_rc(rc, pl->x) + RENDER_TILE_SIZE / 2;
        int base_y = render_world_to_screen_y_rc(rc, pl->y) + RENDER_TILE_SIZE / 2;

        for (int i = 0; i < RENDER_SPLATS_PER_PLAYER; i++) {
            HitSplat* s = &rc->splats[p][i];
            if (!s->active) continue;
            int slot_dx, slot_dy;
            render_splat_slot_offset(i, &slot_dx, &slot_dy);
            int sx = base_x + slot_dx;
            int sy = base_y + slot_dy + (int)s->hitmark_move;
            render_draw_hitmark(rc, sx, sy, s->damage, s->hitmark_trans);
        }
    }
}

/* ======================================================================== */
/* drawing: header                                                           */
/* ======================================================================== */

static void render_draw_header(RenderClient* rc, OsrsEnv* env) {
    DrawRectangle(0, 0, RENDER_WINDOW_W, RENDER_HEADER_HEIGHT, COLOR_HEADER_BG);

    /* left: tick + speed + pause/rewind */
    const char* speed_txt;
    if (rc->ticks_per_second <= 0.0f)         speed_txt = "max";
    else if (rc->ticks_per_second < 1.0f)     speed_txt = TextFormat("%.1f t/s", rc->ticks_per_second);
    else                                      speed_txt = TextFormat("%.0f t/s", rc->ticks_per_second);
    const char* mode_txt = "";
    if (rc->history_cursor >= 0) {
        mode_txt = TextFormat("  [REWIND %d/%d]", rc->history_cursor + 1, rc->history_count);
    } else if (rc->is_paused) {
        mode_txt = "  [PAUSED]";
    }
    int hdr_tick = env->tick;
    if (env->encounter_def && env->encounter_state)
        hdr_tick = ((const EncounterDef*)env->encounter_def)->get_tick(env->encounter_state);
    DrawText(TextFormat("Tick: %d  |  Speed: %s%s", hdr_tick, speed_txt, mode_txt),
             10, 12, 16, rc->history_cursor >= 0 ? COLOR_VENG : COLOR_TEXT);

    /* human control indicator */
    human_draw_hud(&rc->human_input);

    /* right: HP summary (show first 2 entities) */
    if (rc->entity_count >= 2) {
        RenderEntity* p0 = &rc->entities[0];
        RenderEntity* p1 = &rc->entities[1];
        const char* hp_txt = TextFormat("P0: %d/%d   P1: %d/%d",
            p0->current_hitpoints, p0->base_hitpoints,
            p1->current_hitpoints, p1->base_hitpoints);
        int hp_w = MeasureText(hp_txt, 16);
        DrawText(hp_txt, RENDER_GRID_W - hp_w - 10, 12, 16, COLOR_TEXT);
    } else if (rc->entity_count == 1) {
        RenderEntity* p0 = &rc->entities[0];
        const char* hp_txt = TextFormat("P0: %d/%d",
            p0->current_hitpoints, p0->base_hitpoints);
        int hp_w = MeasureText(hp_txt, 16);
        DrawText(hp_txt, RENDER_GRID_W - hp_w - 10, 12, 16, COLOR_TEXT);
    }
}

/* ======================================================================== */
/* drawing: NPC/boss info panel (below GUI tabs)                             */
/* ======================================================================== */

/** Look up inferno NPC name from npc_def_id. returns NULL if not an inferno NPC. */
static const char* inferno_npc_name(int npc_def_id) {
    switch (npc_def_id) {
        case 7691: return "Jal-Nib";
        case 7692: return "Jal-MejRah";
        case 7693: return "Jal-Ak";
        case 7694: return "Jal-AkRek-Ket";
        case 7695: return "Jal-AkRek-Xil";
        case 7696: return "Jal-AkRek-Mej";
        case 7697: return "Jal-ImKot";
        case 7698: return "Jal-Xil";
        case 7699: return "Jal-Zek";
        case 7700: return "JalTok-Jad";
        case 7701: return "Yt-HurKot";
        case 7706: return "TzKal-Zuk";
        case 7707: return "Ancestral Glyph";
        case 7708: return "Jal-MejJak";
        default:   return NULL;
    }
}

static void render_draw_panel_npc(int x, int y, RenderEntity* p, OsrsEnv* env) {
    int line_h = 14;

    /* determine NPC display name and color from npc_def_id */
    const char* npc_name = NULL;
    Color name_color = COLOR_TEXT;

    /* zulrah forms */
    if (p->npc_def_id == 2042)      { npc_name = "Zulrah [GREEN]"; name_color = GREEN; }
    else if (p->npc_def_id == 2043) { npc_name = "Zulrah [RED]"; name_color = RED; }
    else if (p->npc_def_id == 2044) { npc_name = "Zulrah [BLUE]"; name_color = CLITERAL(Color){ 80, 140, 255, 255 }; }

    /* inferno NPCs */
    if (!npc_name) {
        const char* inf_name = inferno_npc_name(p->npc_def_id);
        if (inf_name) {
            npc_name = inf_name;
            name_color = CLITERAL(Color){ 255, 120, 50, 255 };  /* inferno orange */
        }
    }

    if (!npc_name) npc_name = TextFormat("NPC %d", p->npc_def_id);

    DrawText(npc_name, x, y, 14, name_color);
    y += line_h + 4;

    DrawText(TextFormat("HP:     %d / %d", p->current_hitpoints, p->base_hitpoints), x, y, 10, COLOR_TEXT);
    y += line_h;
    DrawText(TextFormat("Pos:    (%d, %d)", p->x, p->y), x, y, 10, COLOR_TEXT_DIM);
    y += line_h;

    /* encounter-specific state overlay */
    if (env->encounter_def && env->encounter_state) {
        const EncounterDef* edef = (const EncounterDef*)env->encounter_def;

        if (strcmp(edef->name, "zulrah") == 0) {
            /* zulrah-specific state */
            ZulrahState* zs = (ZulrahState*)env->encounter_state;
            DrawText(TextFormat("Visible: %s", zs->zulrah_visible ? "yes" : "no"), x, y, 10, COLOR_TEXT_DIM);
            y += line_h;
            DrawText(TextFormat("Phase: %d  Surface: %d  %s", zs->phase_timer, zs->surface_timer,
                zs->is_diving ? "DIVING" : ""), x, y, 10, zs->is_diving ? COLOR_FREEZE : COLOR_TEXT_DIM);
            y += line_h;
            const char* rot_names[] = { "Magma A", "Magma B", "Serp", "Tanz" };
            const char* rot_name = (zs->rotation_index >= 0 && zs->rotation_index < 4)
                ? rot_names[zs->rotation_index] : "???";
            DrawText(TextFormat("Rotation: %s (phase %d/%d)", rot_name,
                zs->phase_index + 1,
                (zs->rotation_index >= 0 && zs->rotation_index < 4)
                    ? ZUL_ROT_LENGTHS[zs->rotation_index] : 0),
                x, y, 10, COLOR_TEXT);
            y += line_h;
            DrawText(TextFormat("Action:  %d/%d (timer %d)", zs->action_index,
                zs->action_progress, zs->action_timer), x, y, 10, COLOR_TEXT_DIM);
            y += line_h;

            int snakes = 0, clouds = 0;
            for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++)
                if (zs->snakelings[i].active) snakes++;
            for (int i = 0; i < ZUL_MAX_CLOUDS; i++)
                if (zs->clouds[i].active) clouds++;
            DrawText(TextFormat("Snakelings: %d  Clouds: %d", snakes, clouds), x, y, 10, COLOR_TEXT_DIM);

        } else if (strcmp(edef->name, "inferno") == 0) {
            /* inferno-specific state */
            InfernoState* is = (InfernoState*)env->encounter_state;
            DrawText(TextFormat("Wave:   %d / %d", is->wave + 1, INF_NUM_WAVES), x, y, 10, COLOR_TEXT);
            y += line_h;

            int active_npcs = 0;
            for (int i = 0; i < INF_MAX_NPCS; i++)
                if (is->npcs[i].active) active_npcs++;
            DrawText(TextFormat("NPCs:   %d active", active_npcs), x, y, 10, COLOR_TEXT_DIM);
            y += line_h;

            int pillars_alive = 0;
            for (int i = 0; i < INF_NUM_PILLARS; i++)
                if (is->pillars[i].active) pillars_alive++;
            DrawText(TextFormat("Pillars: %d / %d", pillars_alive, INF_NUM_PILLARS), x, y, 10, COLOR_TEXT_DIM);
        }
    }
    (void)y;
}

/* render_draw_panel removed — replaced by gui_draw() in osrs_pvp_gui.h */

/* ======================================================================== */
/* drawing: 3D world mode                                                    */
/* ======================================================================== */

static Camera3D render_build_3d_camera(RenderClient* rc) {
    Camera3D cam = { 0 };
    float cx = rc->cam_target_x;
    float cz = rc->cam_target_z;
    /* sample terrain height at camera target (heightmap uses OSRS coords, negate Z back) */
    float cy = (rc->terrain) ? terrain_height_at(rc->terrain, (int)cx, (int)(-cz)) : 2.0f;

    float d = rc->cam_dist;
    float px = cx + d * cosf(rc->cam_pitch) * sinf(rc->cam_yaw);
    float py = cy + d * sinf(rc->cam_pitch);
    float pz = cz + d * cosf(rc->cam_pitch) * cosf(rc->cam_yaw);

    cam.position = (Vector3){ px, py, pz };
    cam.target = (Vector3){ cx, cy, cz };
    cam.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    cam.fovy = 60.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    return cam;
}

/* ======================================================================== */
/* animation selection                                                        */
/* ======================================================================== */

/* animation sequence IDs (from OSRS 317 cache via export_animations.py) */
#define ANIM_SEQ_IDLE           808
#define ANIM_SEQ_WALK           819
#define ANIM_SEQ_RUN            824
#define ANIM_SEQ_EAT            829
#define ANIM_SEQ_DEATH          836
#define ANIM_SEQ_CAST_STANDARD  1162
#define ANIM_SEQ_CAST_BARRAGE   1979
#define ANIM_SEQ_CAST_VENG      4410
#define ANIM_SEQ_BLOCK_SHIELD   1156
#define ANIM_SEQ_BLOCK_MELEE    424

/**
 * Get the attack animation ID for a weapon item (database index).
 * Returns normal attack anim, or special attack anim if is_special.
 */
static int render_get_attack_anim(uint8_t weapon_db_idx, int is_special) {
    if (weapon_db_idx >= NUM_ITEMS) return 422; /* generic punch */

    switch (weapon_db_idx) {
    case ITEM_WHIP:            return 1658;
    case ITEM_GHRAZI_RAPIER:   return 8145;
    case ITEM_INQUISITORS_MACE: return is_special ? 1060 : 400;
    case ITEM_STAFF_OF_DEAD:   return 414;
    case ITEM_KODAI_WAND:      return 414;
    case ITEM_VOLATILE_STAFF:  return is_special ? 8532 : 414;
    case ITEM_AHRIM_STAFF:     return 393;
    case ITEM_ZURIELS_STAFF:   return 393;
    case ITEM_DRAGON_DAGGER:   return is_special ? 1062 : 376;
    case ITEM_DRAGON_CLAWS:    return is_special ? 7514 : 393;
    case ITEM_AGS:             return is_special ? 7644 : 7045;
    case ITEM_ANCIENT_GS:      return is_special ? 7644 : 7045;
    case ITEM_GRANITE_MAUL:    return is_special ? 1667 : 1665;
    case ITEM_ELDER_MAUL:      return 7516;
    case ITEM_STATIUS_WARHAMMER: return is_special ? 1378 : 401;
    case ITEM_VOIDWAKER:       return is_special ? 1378 : 401;
    case ITEM_VESTAS:          return is_special ? 7515 : 390;
    case ITEM_RUNE_CROSSBOW:
    case ITEM_ARMADYL_CROSSBOW:
    case ITEM_ZARYTE_CROSSBOW: return 4230;
    case ITEM_DARK_BOW:        return 426;
    case ITEM_HEAVY_BALLISTA:  return 7218;
    case ITEM_MORRIGANS_JAVELIN: return 806;
    /* zulrah encounter weapons */
    case ITEM_TRIDENT_OF_SWAMP:   return 1167;  /* HUMAN_CASTWAVE_STAFF */
    case ITEM_SANGUINESTI_STAFF:  return 1167;
    case ITEM_EYE_OF_AYAK:        return 1167;
    case ITEM_MAGIC_SHORTBOW_I:   return is_special ? 1074 : 426;  /* snapshot / bow */
    case ITEM_BOW_OF_FAERDHINEN:  return 426;   /* shortbow */
    case ITEM_TWISTED_BOW:        return 426;   /* shortbow */
    case ITEM_TOXIC_BLOWPIPE:     return is_special ? 5061 : 5061; /* blowpipe */
    default:                   return 422;
    }
}

/**
 * Determine the primary (action) animation for this tick.
 * Returns -1 if no action animation should play.
 * Primary animations are server-driven in the real client: attacks, casts, etc.
 * They play once then auto-expire (loopCount=1 effectively).
 */
static int render_select_primary(RenderEntity* p) {
    if (p->current_hitpoints <= 0) return ANIM_SEQ_DEATH;

    if (p->attack_style_this_tick != ATTACK_STYLE_NONE) {
        if (p->attack_style_this_tick == ATTACK_STYLE_MAGIC) {
            /* powered staves (trident/sang/ayak) use their own cast anim */
            uint8_t wpn = p->equipped[GEAR_SLOT_WEAPON];
            if (wpn == ITEM_TRIDENT_OF_SWAMP || wpn == ITEM_SANGUINESTI_STAFF ||
                wpn == ITEM_EYE_OF_AYAK)
                return 1167;  /* HUMAN_CASTWAVE_STAFF */
            return ANIM_SEQ_CAST_BARRAGE;
        }
        return render_get_attack_anim(
            p->equipped[GEAR_SLOT_WEAPON], p->used_special_this_tick);
    }

    if (p->ate_food_this_tick || p->ate_karambwan_this_tick) {
        return ANIM_SEQ_EAT;
    }

    if (p->cast_veng_this_tick) {
        return ANIM_SEQ_CAST_VENG;
    }

    if (p->hit_landed_this_tick && p->equipped[GEAR_SLOT_SHIELD] < NUM_ITEMS) {
        return ANIM_SEQ_BLOCK_SHIELD;
    }

    return -1; /* no action this tick */
}

/**
 * Determine the secondary (pose) animation based on VISUAL movement state.
 *
 * In the real client (nextStep), this is set based on the entity's sub-tile
 * movement: idle when not moving, walk or run based on moveSpeed. We use the
 * visual_moving/visual_running flags set by the client-tick loop.
 */
static int render_select_secondary(RenderClient* rc, int player_idx) {
    if (!rc->visual_moving[player_idx]) return ANIM_SEQ_IDLE;
    if (rc->visual_running[player_idx]) return ANIM_SEQ_RUN;
    return ANIM_SEQ_WALK;
}

/* ======================================================================== */
/* composite model building                                                   */
/* ======================================================================== */

/**
 * Append a single OsrsModel's geometry into the player composite.
 * Offsets face indices by the current base vertex count so the merged
 * index buffer references the correct vertices.
 */
static void composite_add_model(PlayerComposite* comp, OsrsModel* om) {
    if (!om->base_vertices || om->base_vert_count == 0) return;

    int bv_off = comp->base_vert_count;
    int fc_off = comp->face_count;

    /* bounds check */
    if (bv_off + om->base_vert_count > COMPOSITE_MAX_BASE_VERTS) return;
    if (fc_off + om->mesh.triangleCount > COMPOSITE_MAX_FACES) return;

    /* append base vertices */
    memcpy(comp->base_vertices + bv_off * 3,
           om->base_vertices, om->base_vert_count * 3 * sizeof(int16_t));

    /* append vertex skins */
    memcpy(comp->vertex_skins + bv_off,
           om->vertex_skins, om->base_vert_count);

    /* append face indices (offset by base vertex count) */
    int nfi = om->mesh.triangleCount * 3;
    for (int f = 0; f < nfi; f++) {
        comp->face_indices[fc_off * 3 + f] = om->face_indices[f] + (uint16_t)bv_off;
    }

    /* append face priority deltas (relative to this model's min, not composite global).
     * prevents adjacent faces within a uniform-priority model from getting offset. */
    if (om->face_priorities) {
        for (int f = 0; f < om->mesh.triangleCount; f++) {
            int delta = (int)om->face_priorities[f] - (int)om->min_priority;
            comp->face_pri_delta[fc_off + f] = (uint8_t)(delta > 0 ? delta : 0);
        }
    } else {
        memset(comp->face_pri_delta + fc_off, 0, om->mesh.triangleCount);
    }

    /* append expanded colors into the composite mesh color buffer */
    int exp_off = fc_off * 3;
    memcpy(comp->mesh.colors + exp_off * 4,
           om->mesh.colors, om->mesh.triangleCount * 3 * 4);

    comp->base_vert_count += om->base_vert_count;
    comp->face_count += om->mesh.triangleCount;
}

/**
 * Rebuild a player's composite model from their visible body parts + equipment.
 * Called when equipment changes or on first frame.
 */
static void composite_rebuild(
    PlayerComposite* comp, ModelCache* cache, RenderEntity* p
) {
    comp->base_vert_count = 0;
    comp->face_count = 0;

    /* add visible body parts (hide parts covered by equipment) */
    for (int bp = 0; bp < BODY_PART_COUNT; bp++) {
        int hide = 0;
        if (bp == BODY_PART_HEAD) {
            hide = (p->equipped[GEAR_SLOT_HEAD] < NUM_ITEMS);
        } else if (bp == BODY_PART_TORSO) {
            hide = (p->equipped[GEAR_SLOT_BODY] < NUM_ITEMS);
        } else if (bp == BODY_PART_ARMS) {
            uint8_t body_idx = p->equipped[GEAR_SLOT_BODY];
            if (body_idx < NUM_ITEMS) {
                hide = item_has_sleeves(ITEM_DATABASE[body_idx].item_id);
            }
        } else if (bp == BODY_PART_LEGS) {
            hide = (p->equipped[GEAR_SLOT_LEGS] < NUM_ITEMS);
        } else if (bp == BODY_PART_HANDS) {
            hide = (p->equipped[GEAR_SLOT_HANDS] < NUM_ITEMS);
        } else if (bp == BODY_PART_FEET) {
            hide = (p->equipped[GEAR_SLOT_FEET] < NUM_ITEMS);
        }
        if (hide) continue;

        OsrsModel* om = model_cache_get(cache, DEFAULT_BODY_MODELS[bp]);
        if (om) composite_add_model(comp, om);
    }

    /* add equipped item wield models */
    static const int VISIBLE_SLOTS[] = {
        GEAR_SLOT_HEAD, GEAR_SLOT_CAPE, GEAR_SLOT_NECK,
        GEAR_SLOT_WEAPON, GEAR_SLOT_SHIELD, GEAR_SLOT_BODY,
        GEAR_SLOT_LEGS, GEAR_SLOT_HANDS, GEAR_SLOT_FEET,
    };
    for (int s = 0; s < 9; s++) {
        int slot = VISIBLE_SLOTS[s];
        uint8_t db_idx = p->equipped[slot];
        if (db_idx >= NUM_ITEMS) continue;

        uint16_t item_id = ITEM_DATABASE[db_idx].item_id;
        uint32_t model_id = item_to_wield_model(item_id);
        if (model_id == 0xFFFFFFFF) continue;

        OsrsModel* om = model_cache_get(cache, model_id);
        if (om) composite_add_model(comp, om);
    }

    /* rebuild animation state for the new composite geometry */
    if (comp->anim_state) {
        anim_model_state_free(comp->anim_state);
        comp->anim_state = NULL;
    }
    if (comp->base_vert_count > 0) {
        comp->anim_state = anim_model_state_create(
            comp->vertex_skins, comp->base_vert_count);
    }

    /* save equipment state for change detection */
    memcpy(comp->last_equipped, p->equipped, NUM_GEAR_SLOTS);
    comp->needs_rebuild = 0;
}

/**
 * Rebuild an NPC's composite from a single cache model (no equipment composition).
 * Used for Zulrah forms, snakelings, and other encounter NPCs.
 */
static void composite_rebuild_npc(
    PlayerComposite* comp, ModelCache* cache, ModelCache* npc_cache, int npc_def_id
) {
    comp->base_vert_count = 0;
    comp->face_count = 0;

    /* zero mesh buffers to prevent stale GPU data from showing as garbled geometry
       if the model fails to load or exceeds composite limits */
    if (comp->mesh.vertices)
        memset(comp->mesh.vertices, 0, COMPOSITE_MAX_EXP_VERTS * 3 * sizeof(float));
    if (comp->mesh.colors)
        memset(comp->mesh.colors, 0, COMPOSITE_MAX_EXP_VERTS * 4);

    /* look up model ID from NPC definition */
    uint32_t model_id = 0;
    const NpcModelMapping* mapping = npc_model_lookup((uint16_t)npc_def_id);
    if (!mapping) {
        fprintf(stderr, "render: missing NPC model mapping for npc_def_id=%d\n", npc_def_id);
        abort();
    }
    model_id = mapping->model_id;

    OsrsModel* om = model_cache_get(cache, model_id);
    /* fallback: check secondary NPC model cache (inferno etc.) */
    if (!om && npc_cache)
        om = model_cache_get(npc_cache, model_id);
    if (!om) {
        fprintf(stderr, "render: npc_def_id=%d mapped model %u is missing from loaded caches\n",
                npc_def_id, model_id);
        abort();
    }
    composite_add_model(comp, om);

    /* rebuild animation state */
    if (comp->anim_state) {
        anim_model_state_free(comp->anim_state);
        comp->anim_state = NULL;
    }
    if (comp->base_vert_count > 0) {
        comp->anim_state = anim_model_state_create(
            comp->vertex_skins, comp->base_vert_count);
    }

    comp->last_npc_def_id = npc_def_id;
    comp->needs_rebuild = 0;
}

/**
 * Initialize the composite's GPU resources (once, at max capacity).
 * Uses dynamic=true since we update vertices every frame.
 */
static void composite_init_gpu(PlayerComposite* comp) {
    if (comp->gpu_ready) return;

    comp->mesh.vertexCount = COMPOSITE_MAX_EXP_VERTS;
    comp->mesh.triangleCount = COMPOSITE_MAX_FACES;
    comp->mesh.vertices = (float*)RL_CALLOC(COMPOSITE_MAX_EXP_VERTS * 3, sizeof(float));
    comp->mesh.colors = (unsigned char*)RL_CALLOC(COMPOSITE_MAX_EXP_VERTS, 4);

    UploadMesh(&comp->mesh, true);  /* dynamic VBO for per-frame updates */
    comp->model = LoadModelFromMesh(comp->mesh);
    comp->gpu_ready = 1;
}

/**
 * Apply per-face render priority offset to prevent z-fighting on coplanar faces.
 * Pushes higher-priority faces slightly along their face normal (toward camera).
 * Uses pre-computed per-face deltas (relative to each source model's min priority)
 * so uniform-priority models get zero offset.
 */
static void composite_apply_priority_offset(
    float* mesh_verts, const uint8_t* pri_deltas, int face_count
) {
    for (int fi = 0; fi < face_count; fi++) {
        int delta = (int)pri_deltas[fi];
        if (delta <= 0) continue;

        int vi = fi * 9;
        float ax = mesh_verts[vi], ay = mesh_verts[vi+1], az = mesh_verts[vi+2];
        float bx = mesh_verts[vi+3], by = mesh_verts[vi+4], bz = mesh_verts[vi+5];
        float cx = mesh_verts[vi+6], cy = mesh_verts[vi+7], cz = mesh_verts[vi+8];

        /* face normal via cross product */
        float e1x = bx-ax, e1y = by-ay, e1z = bz-az;
        float e2x = cx-ax, e2y = cy-ay, e2z = cz-az;
        float nx = e1y*e2z - e1z*e2y;
        float ny = e1z*e2x - e1x*e2z;
        float nz = e1x*e2y - e1y*e2x;
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len < 0.001f) continue;

        /* 0.15 OSRS units per priority level (matches Python exporter) */
        float bias = (float)delta * 0.15f / len;
        nx *= bias; ny *= bias; nz *= bias;

        /* push along -normal (matches exporter convention for OSRS CW winding) */
        mesh_verts[vi]   -= nx; mesh_verts[vi+1] -= ny; mesh_verts[vi+2] -= nz;
        mesh_verts[vi+3] -= nx; mesh_verts[vi+4] -= ny; mesh_verts[vi+5] -= nz;
        mesh_verts[vi+6] -= nx; mesh_verts[vi+7] -= ny; mesh_verts[vi+8] -= nz;
    }
}

/**
 * Animate composite model, upload to GPU, and draw.
 */
/**
 * Apply animation(s), re-expand vertices, upload to GPU, and draw.
 *
 * When both primary and secondary are provided with an interleave_order,
 * uses interleaved application (upper body from primary, legs from secondary).
 * Otherwise falls back to single-frame application.
 */
static void composite_animate_and_draw(
    PlayerComposite* comp,
    const AnimFrameData* secondary_frame, const AnimFrameBase* secondary_fb,
    const AnimFrameData* primary_frame, const AnimFrameBase* primary_fb,
    const uint8_t* interleave_order, int interleave_count,
    Matrix transform
) {
    if (!comp->anim_state || comp->face_count == 0) return;

    /* apply animation transforms to base vertices */
    if (primary_frame && secondary_frame && interleave_order && interleave_count > 0) {
        /* two-track: primary owns upper body, secondary owns legs */
        anim_apply_frame_interleaved(
            comp->anim_state, comp->base_vertices,
            secondary_frame, secondary_fb,
            primary_frame, primary_fb,
            interleave_order, interleave_count);
    } else if (primary_frame) {
        /* primary only (death, or anims without interleave_order) */
        anim_apply_frame(comp->anim_state, comp->base_vertices,
                         primary_frame, primary_fb);
    } else if (secondary_frame) {
        /* secondary only (walk/idle, no action) */
        anim_apply_frame(comp->anim_state, comp->base_vertices,
                         secondary_frame, secondary_fb);
    }

    /* re-expand animated base verts into mesh vertex buffer */
    anim_update_mesh(comp->mesh.vertices, comp->anim_state,
                     comp->face_indices, comp->face_count);

    /* apply face priority offset to prevent z-fighting on coplanar faces */
    composite_apply_priority_offset(
        comp->mesh.vertices, comp->face_pri_delta, comp->face_count);

    /* sanity clamp: catch degenerate animation frames that produce extreme
       vertex positions (int16_t overflow in animation math). without this,
       a single bad frame can create a screen-filling triangle. OSRS model
       coords are typically ±2000; 10000 is already way beyond any real model. */
    {
        int nv = comp->face_count * 3 * 3;
        for (int i = 0; i < nv; i++) {
            if (comp->mesh.vertices[i] > 10000.0f) comp->mesh.vertices[i] = 10000.0f;
            else if (comp->mesh.vertices[i] < -10000.0f) comp->mesh.vertices[i] = -10000.0f;
        }
    }

    /* upload updated vertices and colors to GPU */
    int exp_verts = comp->face_count * 3;
    UpdateMeshBuffer(comp->mesh, 0, comp->mesh.vertices,
                     exp_verts * 3 * sizeof(float), 0);
    UpdateMeshBuffer(comp->mesh, 3, comp->mesh.colors,
                     exp_verts * 4, 0);

    /* draw with the current face count. CRITICAL: must set vertexCount on
       model.meshes[0], NOT comp->mesh — LoadModelFromMesh copies the mesh
       struct by value, so comp->mesh and model.meshes[0] are independent.
       DrawModel reads model.meshes[0].vertexCount for glDrawArrays count. */
    comp->model.meshes[0].vertexCount = exp_verts;
    comp->model.meshes[0].triangleCount = comp->face_count;
    comp->model.transform = transform;
    DrawModel(comp->model, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);

    /* restore max counts so the VBO stays valid for next UpdateMeshBuffer */
    comp->model.meshes[0].vertexCount = COMPOSITE_MAX_EXP_VERTS;
    comp->model.meshes[0].triangleCount = COMPOSITE_MAX_FACES;
}

static void composite_free(PlayerComposite* comp) {
    if (comp->gpu_ready) {
        UnloadModel(comp->model);
        comp->gpu_ready = 0;
    }
    anim_model_state_free(comp->anim_state);
    comp->anim_state = NULL;
}

/* ======================================================================== */
/* per-player animation + composite orchestration                             */
/* ======================================================================== */

/**
 * Rebuild composite if equipment changed, run two-track animation, draw.
 *
 * Two-track animation system (matches OSRS client):
 *   - secondary: always running (idle/walk/run), loops forever
 *   - primary: triggered by actions (attack/cast/eat/block/death), plays
 *     once then expires. when active with interleave_order, overrides
 *     secondary for upper body groups.
 */
static void render_player_composite(
    RenderClient* rc, int player_idx, Matrix transform
) {
    if (!rc->model_cache) return;

    PlayerComposite* comp = &rc->composites[player_idx];
    RenderEntity* p = &rc->entities[player_idx];

    composite_init_gpu(comp);

    /* branch on entity type: NPCs use single-model composites */
    if (p->entity_type == ENTITY_NPC) {
        if (comp->needs_rebuild || comp->last_npc_def_id != p->npc_def_id) {
            composite_rebuild_npc(comp, rc->model_cache, rc->npc_model_cache, p->npc_def_id);
        }
    } else {
        if (comp->needs_rebuild ||
            memcmp(comp->last_equipped, p->equipped, NUM_GEAR_SLOTS) != 0) {
            composite_rebuild(comp, rc->model_cache, p);
        }
    }

    if (!rc->anim_cache || !comp->anim_state) {
        /* no animation: draw static */
        if (comp->face_count > 0) {
            int exp_verts = comp->face_count * 3;
            comp->model.meshes[0].vertexCount = exp_verts;
            comp->model.meshes[0].triangleCount = comp->face_count;
            comp->model.transform = transform;
            DrawModel(comp->model, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
            comp->model.meshes[0].vertexCount = COMPOSITE_MAX_EXP_VERTS;
            comp->model.meshes[0].triangleCount = COMPOSITE_MAX_FACES;
        }
        return;
    }

    /* --- primary track: trigger new actions and expire finished ones ---
       primary is triggered per game tick (render_post_tick sets flags),
       but frame advancement happens in render_client_tick at 50 Hz.

       bug fix: when the same anim fires again after expiry (e.g. two
       consecutive whip attacks), we must restart it. check both seq_id
       change AND whether the current one has already finished (loops > 0). */
    int new_primary;
    if (p->entity_type == ENTITY_NPC) {
        /* NPCs set their animation via npc_anim_id from the encounter.
           idle is secondary (looping), attack/dive/surface are primary (play-once). */
        const NpcModelMapping* nm = npc_model_lookup((uint16_t)p->npc_def_id);
        int idle = nm ? (int)nm->idle_anim : -1;
        new_primary = (p->npc_anim_id >= 0 && p->npc_anim_id != idle)
            ? p->npc_anim_id : -1;
    } else {
        new_primary = render_select_primary(p);
    }
    if (new_primary >= 0) {
        int need_restart = (rc->anim[player_idx].primary_seq_id != new_primary) ||
                           (rc->anim[player_idx].primary_loops > 0);
        if (need_restart) {
            rc->anim[player_idx].primary_seq_id = new_primary;
            rc->anim[player_idx].primary_frame_idx = 0;
            rc->anim[player_idx].primary_ticks = 0;
            rc->anim[player_idx].primary_loops = 0;
        }
    }

    /* expire primary after one loop (death never expires) */
    if (rc->anim[player_idx].primary_seq_id >= 0 &&
        rc->anim[player_idx].primary_loops > 0 &&
        rc->anim[player_idx].primary_seq_id != ANIM_SEQ_DEATH) {
        rc->anim[player_idx].primary_seq_id = -1;
    }

    /* --- read current frame data (set by render_client_tick at 50 Hz) --- */
    AnimSequenceFrame *sec_sf = NULL, *pri_sf = NULL;
    AnimFrameBase *sec_fb = NULL, *pri_fb = NULL;

    /* secondary frame */
    if (rc->anim[player_idx].secondary_seq_id >= 0) {
        AnimSequence* seq = render_get_anim_sequence(
            rc, (uint16_t)rc->anim[player_idx].secondary_seq_id);
        if (seq && seq->frame_count > 0) {
            int fidx = rc->anim[player_idx].secondary_frame_idx % seq->frame_count;
            AnimSequenceFrame* sf = &seq->frames[fidx];
            if (sf->frame.framebase_id != 0xFFFF) {
                AnimFrameBase* fb = render_get_framebase(rc, sf->frame.framebase_id);
                if (fb) { sec_sf = sf; sec_fb = fb; }
            }
        }
    }

    /* primary frame */
    if (rc->anim[player_idx].primary_seq_id >= 0) {
        AnimSequence* seq = render_get_anim_sequence(
            rc, (uint16_t)rc->anim[player_idx].primary_seq_id);
        if (seq && seq->frame_count > 0) {
            int fidx = rc->anim[player_idx].primary_frame_idx % seq->frame_count;
            AnimSequenceFrame* sf = &seq->frames[fidx];
            if (sf->frame.framebase_id != 0xFFFF) {
                AnimFrameBase* fb = render_get_framebase(rc, sf->frame.framebase_id);
                if (fb) { pri_sf = sf; pri_fb = fb; }
            }
        }
    }

    /* --- resolve interleave_order from the primary sequence --- */
    const uint8_t* interleave = NULL;
    int interleave_count = 0;
    if (pri_sf) {
        AnimSequence* prim_seq = render_get_anim_sequence(
            rc, (uint16_t)rc->anim[player_idx].primary_seq_id);
        if (prim_seq && prim_seq->interleave_order) {
            interleave = prim_seq->interleave_order;
            interleave_count = prim_seq->interleave_count;
        }
    }

    /* --- animate and draw --- */
    composite_animate_and_draw(
        comp,
        sec_sf ? &sec_sf->frame : NULL, sec_fb,
        pri_sf ? &pri_sf->frame : NULL, pri_fb,
        interleave, interleave_count,
        transform);
}

static void render_draw_3d_world(RenderClient* rc) {
    /* tighten near/far clip planes for depth buffer precision.
       default 0.01/1000 = 100,000:1 ratio wastes precision and causes
       z-fighting across the entire scene. 0.5/500 = 1000:1 is sufficient
       for our tile-scale world (camera is never closer than ~1 tile). */
    rlSetClipPlanes(0.5, 500.0);

    Camera3D cam = render_build_3d_camera(rc);
    BeginMode3D(cam);

    /* terrain mesh (PvP wilderness) or flat ground plane (encounters) */
    if (rc->terrain && rc->terrain->loaded) {
        DrawModel(rc->terrain->model, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);

        /* 3D collision overlay on terrain: semi-transparent quads at tile height */
        if (rc->show_collision && rc->collision_map) {
            for (int dx = 0; dx < rc->arena_width; dx++) {
                for (int dy = 0; dy < rc->arena_height; dy++) {
                    int wx = rc->arena_base_x + dx + rc->collision_world_offset_x;
                    int wy = rc->arena_base_y + dy + rc->collision_world_offset_y;
                    int flags = collision_get_flags(rc->collision_map, 0, wx, wy);

                    Color col = { 0, 0, 0, 0 };
                    if (flags & COLLISION_BLOCKED) {
                        col = CLITERAL(Color){ 200, 50, 50, 80 };
                    } else if (flags & COLLISION_BRIDGE) {
                        col = CLITERAL(Color){ 50, 120, 220, 80 };
                    } else if (flags & 0x0FF) {
                        col = CLITERAL(Color){ 220, 150, 40, 60 };
                    } else {
                        col = CLITERAL(Color){ 50, 200, 50, 40 };
                    }

                    float tx = (float)(rc->arena_base_x + dx);
                    float tz = -(float)(rc->arena_base_y + dy + 1);
                    /* sample terrain height at tile */
                    float ground = terrain_height_avg(rc->terrain,
                        rc->arena_base_x + dx, rc->arena_base_y + dy);
                    DrawCube((Vector3){ tx + 0.5f, ground + 0.05f, tz + 0.5f },
                             1.0f, 0.02f, 1.0f, col);
                }
            }
        }
    } else if (rc->npc_model_cache) {
        /* inferno: dark cave floor. all tiles are walkable ground. */
        float plat_y = 2.0f;
        for (int dx = 0; dx < rc->arena_width; dx++) {
            for (int dy = 0; dy < rc->arena_height; dy++) {
                float tx = (float)(rc->arena_base_x + dx);
                float tz = -(float)(rc->arena_base_y + dy + 1);

                /* volcanic rock with subtle variation — bright enough to distinguish from background */
                int shade = 45 + ((dx * 7 + dy * 13) % 15);
                int r = shade + ((dx * 3 + dy * 11) % 10);  /* slight reddish tint */
                Color c = { (unsigned char)r, (unsigned char)(shade - 3), (unsigned char)(shade - 6), 255 };
                DrawCube((Vector3){ tx + 0.5f, plat_y - 0.05f, tz + 0.5f },
                         1.0f, 0.1f, 1.0f, c);
            }
        }
    } else {
        /* zulrah / generic encounter: raised green platform over blue water.
           the real arena is instanced so it can't be exported from the cache. */
        float water_y = 1.5f;
        float plat_y = 2.0f;

        for (int dx = 0; dx < rc->arena_width; dx++) {
            for (int dy = 0; dy < rc->arena_height; dy++) {
                float tx = (float)(rc->arena_base_x + dx);
                float tz = -(float)(rc->arena_base_y + dy + 1);

                /* determine platform vs water: use collision map if available,
                   otherwise fall back to hardcoded platform bounds */
                int on_plat;
                if (rc->collision_map) {
                    int wx = rc->arena_base_x + dx + rc->collision_world_offset_x;
                    int wy = rc->arena_base_y + dy + rc->collision_world_offset_y;
                    on_plat = collision_tile_walkable(rc->collision_map, 0, wx, wy);
                } else {
                    on_plat = (dx >= ZUL_PLATFORM_MIN && dx <= ZUL_PLATFORM_MAX &&
                               dy >= ZUL_PLATFORM_MIN && dy <= ZUL_PLATFORM_MAX);
                }

                if (on_plat) {
                    int shade = 35 + ((dx * 7 + dy * 13) % 15);
                    Color c = { (unsigned char)shade, (unsigned char)(shade * 2), (unsigned char)shade, 255 };
                    DrawCube((Vector3){ tx + 0.5f, plat_y - 0.05f, tz + 0.5f },
                             1.0f, 0.1f, 1.0f, c);
                } else {
                    int shade = 15 + ((dx * 3 + dy * 5) % 10);
                    Color c = { (unsigned char)(shade / 2), (unsigned char)shade, (unsigned char)(shade * 3), 255 };
                    DrawCube((Vector3){ tx + 0.5f, water_y - 0.05f, tz + 0.5f },
                             1.0f, 0.1f, 1.0f, c);
                }
            }
        }
    }

    /* inferno pillars: "Rocky support" objects with 4 HP-level models.
       dynamically spawned (not in static objects file). */
    if (rc->npc_model_cache && rc->gui.encounter_state) {
        InfernoState* is = (InfernoState*)rc->gui.encounter_state;
        float plat_y = 2.0f;
        float ms = 1.0f / 128.0f;
        for (int p = 0; p < INF_NUM_PILLARS; p++) {
            if (!is->pillars[p].active) continue;
            float hp_frac = (float)is->pillars[p].hp / (float)INF_PILLAR_HP;

            float cx = (float)is->pillars[p].x + INF_PILLAR_SIZE / 2.0f;
            float cz = -(float)(is->pillars[p].y + INF_PILLAR_SIZE / 2) - 0.5f;

            if (rc->pillar_models_ready) {
                /* select model by HP: 100%, 75%, 50%, 25% */
                int mi = 0;
                if (hp_frac <= 0.25f) mi = 3;
                else if (hp_frac <= 0.50f) mi = 2;
                else if (hp_frac <= 0.75f) mi = 1;

                rlDisableBackfaceCulling();
                rc->pillar_models[mi].transform = MatrixMultiply(
                    MatrixScale(-ms, ms, ms),
                    MatrixTranslate(cx, plat_y, cz));
                DrawModel(rc->pillar_models[mi], (Vector3){0,0,0}, 1.0f, WHITE);
                rlEnableBackfaceCulling();
            } else {
                /* fallback: colored DrawCube blocks */
                int base_r = (int)(140 * hp_frac + 180 * (1.0f - hp_frac));
                int base_g = (int)(130 * hp_frac + 40 * (1.0f - hp_frac));
                int base_b = (int)(100 * hp_frac + 20 * (1.0f - hp_frac));
                Color pillar_col = { (unsigned char)base_r, (unsigned char)base_g, (unsigned char)base_b, 240 };
                for (int dx = 0; dx < INF_PILLAR_SIZE; dx++) {
                    for (int dy = 0; dy < INF_PILLAR_SIZE; dy++) {
                        float tx = (float)(is->pillars[p].x + dx);
                        float tz2 = -(float)(is->pillars[p].y + dy + 1);
                        for (int h = 0; h < 3; h++) {
                            DrawCube((Vector3){ tx + 0.5f, plat_y + 0.5f + (float)h, tz2 + 0.5f },
                                     0.95f, 0.95f, 0.95f, pillar_col);
                        }
                    }
                }
            }
        }
    }

    /* debug: highlight the last raycast-selected tile */
    if (rc->show_debug && rc->debug_hit_wx >= 0) {
        float dtx = (float)rc->debug_hit_wx;
        float dtz = -(float)(rc->debug_hit_wy + 1);
        float dgy = rc->terrain
            ? terrain_height_avg(rc->terrain, rc->debug_hit_wx, rc->debug_hit_wy)
            : 2.0f;
        DrawCube((Vector3){ dtx + 0.5f, dgy + 0.02f, dtz + 0.5f },
                 1.0f, 0.02f, 1.0f, (Color){ 255, 0, 255, 180 });
        DrawSphere((Vector3){ rc->debug_ray_hit_x, rc->debug_ray_hit_y, rc->debug_ray_hit_z },
                   0.1f, RED);
    }
    /* draw ray as line from origin forward */
    if (rc->show_debug && rc->debug_hit_wx >= 0) {
        Vector3 a = rc->debug_ray_origin;
        Vector3 b = { a.x + rc->debug_ray_dir.x * 50.0f,
                      a.y + rc->debug_ray_dir.y * 50.0f,
                      a.z + rc->debug_ray_dir.z * 50.0f };
        DrawLine3D(a, b, YELLOW);
    }

    /* debug: draw game-logic tile positions for all entities.
       green = player, cyan = NPCs. shows where the game thinks entities are
       vs where the 3D model renders (which uses sub_x/sub_y interpolation). */
    if (rc->show_debug) {
        for (int i = 0; i < rc->entity_count; i++) {
            RenderEntity* ep = &rc->entities[i];
            if (ep->entity_type == ENTITY_NPC && !ep->npc_visible) continue;
            float tx = (float)ep->x;
            float ty = (float)ep->y;
            float tz = -(ty + 1.0f);
            float ground = rc->terrain
                ? terrain_height_avg(rc->terrain, ep->x, ep->y) : 2.0f;
            int sz = ep->npc_size > 1 ? ep->npc_size : 1;
            Color col = (ep->entity_type == ENTITY_PLAYER)
                ? CLITERAL(Color){ 0, 255, 0, 100 }
                : CLITERAL(Color){ 0, 200, 255, 80 };
            for (int dx = 0; dx < sz; dx++) {
                for (int dy = 0; dy < sz; dy++) {
                    float mx = tx + (float)dx;
                    float mz = tz - (float)dy;
                    DrawCube((Vector3){ mx + 0.5f, ground + 0.08f, mz + 0.5f },
                             0.9f, 0.04f, 0.9f, col);
                }
            }
        }
    }

    /* entity click hitboxes are now drawn as 2D convex hulls after EndMode3D */

    /* encounter overlay: drawn on top of terrain or procedural arena */
    {
        EncounterOverlay* ov = &rc->encounter_overlay;
        int has_terrain = rc->terrain && rc->terrain->loaded;

        /* helper: get ground height at a tile position */
        #define OV_GROUND(tile_x, tile_y) \
            (has_terrain ? terrain_height_avg(rc->terrain, (tile_x), (tile_y)) : 2.0f)

        /* current hazard renderer: object 11700 centered on a 3x3 damage area */
        float ms = 1.0f / 128.0f;
        for (int i = 0; i < ov->hazard_count; i++) {
            if (!ov->hazards[i].active) continue;
            float ground = OV_GROUND(ov->hazards[i].x + 1, ov->hazards[i].y + 1);

            if (rc->cloud_model_ready) {
                float cx = (float)ov->hazards[i].x + 1.5f;
                float cz = -(float)(ov->hazards[i].y + 2) + 0.5f;
                rlDisableBackfaceCulling();
                rc->cloud_model.transform = MatrixMultiply(
                    MatrixScale(ms, ms, ms),
                    MatrixTranslate(cx, ground + 0.1f, cz));
                DrawModel(rc->cloud_model, (Vector3){0,0,0}, 1.0f, WHITE);
                rlEnableBackfaceCulling();
            } else {
                /* fallback: semi-transparent tiles if model not loaded */
                for (int cdx = 0; cdx < 3; cdx++) {
                    for (int cdy = 0; cdy < 3; cdy++) {
                        float fx = (float)(ov->hazards[i].x + cdx);
                        float fz = -(float)(ov->hazards[i].y + cdy + 1);
                        float tg = OV_GROUND(ov->hazards[i].x + cdx, ov->hazards[i].y + cdy);
                        DrawCube((Vector3){ fx + 0.5f, tg + 0.08f, fz + 0.5f },
                                 0.95f, 0.06f, 0.95f,
                                 CLITERAL(Color){ 80, 180, 50, 100 });
                    }
                }
            }
        }

        /* boss hitbox: NxN form-colored tiles on the ground */
        if (rc->show_debug && ov->boss_visible && ov->boss_size > 0) {
            Color form_col;
            switch (ov->boss_form) {
                case 0: form_col = CLITERAL(Color){ 50, 200, 50, 80 }; break;  /* green */
                case 1: form_col = CLITERAL(Color){ 200, 50, 50, 80 }; break;  /* red */
                case 2: form_col = CLITERAL(Color){ 50, 100, 255, 80 }; break; /* blue */
                default: form_col = CLITERAL(Color){ 200, 200, 200, 80 }; break;
            }
            Color border_col = form_col;
            border_col.a = 200;
            int sz = ov->boss_size;
            for (int bx = 0; bx < sz; bx++) {
                for (int by = 0; by < sz; by++) {
                    int tx = ov->boss_x + bx;
                    int ty = ov->boss_y + by;
                    float ground = OV_GROUND(tx, ty);
                    float fx = (float)tx;
                    float fz = -(float)(ty + 1);
                    DrawCube((Vector3){ fx + 0.5f, ground + 0.04f, fz + 0.5f },
                             1.0f, 0.02f, 1.0f, form_col);
                }
            }
            /* border outline */
            float x0 = (float)ov->boss_x;
            float x1 = (float)(ov->boss_x + sz);
            float z0 = -(float)(ov->boss_y + sz);
            float z1 = -(float)ov->boss_y;
            float border_y = OV_GROUND(ov->boss_x + sz/2, ov->boss_y + sz/2) + 0.06f;
            DrawLine3D((Vector3){x0, border_y, z0}, (Vector3){x1, border_y, z0}, border_col);
            DrawLine3D((Vector3){x1, border_y, z0}, (Vector3){x1, border_y, z1}, border_col);
            DrawLine3D((Vector3){x1, border_y, z1}, (Vector3){x0, border_y, z1}, border_col);
            DrawLine3D((Vector3){x0, border_y, z1}, (Vector3){x0, border_y, z0}, border_col);
        }

        /* melee targeting indicator: red tile where boss is aiming */
        if (ov->melee_target_active) {
            float ground = OV_GROUND(ov->melee_target_x, ov->melee_target_y);
            float mx = (float)ov->melee_target_x;
            float mz = -(float)(ov->melee_target_y + 1);
            DrawCube((Vector3){ mx + 0.5f, ground + 0.06f, mz + 0.5f },
                     1.0f, 0.04f, 1.0f,
                     CLITERAL(Color){ 255, 50, 50, 150 });
        }

        /* encounter adds: current renderer uses the snakeling model or cubes. */
        for (int i = 0; i < ov->add_count; i++) {
            if (!ov->adds[i].active) continue;
            float ground = OV_GROUND(ov->adds[i].x, ov->adds[i].y);
            float sx = (float)ov->adds[i].x + 0.5f;
            float sz = -(float)(ov->adds[i].y + 1) + 0.5f;
            if (rc->snakeling_model_ready) {
                rlDisableBackfaceCulling();
                rc->snakeling_model.transform = MatrixMultiply(
                    MatrixScale(ms, ms, ms),
                    MatrixTranslate(sx, ground, sz));
                DrawModel(rc->snakeling_model, (Vector3){0,0,0}, 1.0f, WHITE);
                rlEnableBackfaceCulling();
            } else {
                Color sc = ov->adds[i].variant
                    ? CLITERAL(Color){ 100, 150, 255, 200 }
                    : CLITERAL(Color){ 255, 150, 50, 200 };
                DrawCube((Vector3){ sx, ground + 0.2f, sz },
                         0.6f, 0.3f, 0.6f, sc);
            }
        }

        /* projectiles: render in-flight projectiles with interpolated positions.
           flight_spawn() creates flights from overlay events (in render_post_tick),
           flight_client_tick() advances progress at 50Hz, we just draw here. */
        for (int i = 0; i < MAX_FLIGHT_PROJECTILES; i++) {
            FlightProjectile* fp = &rc->flights[i];
            if (!fp->active || fp->start_delay > 0) continue;

            float src_ground = OV_GROUND((int)fp->src_x, (int)fp->src_y);
            float dst_ground = OV_GROUND((int)fp->dst_x, (int)fp->dst_y);
            Vector3 pos = flight_get_position(fp, src_ground, dst_ground);

            Model* proj_model = NULL;
            if (fp->anim_id >= 0 && fp->model_id > 0) {
                OsrsModel* om = render_get_projectile_osrs_model(rc, fp->model_id);
                AnimSequence* seq = render_get_anim_sequence(rc, (uint16_t)fp->anim_id);
                if (!fp->anim_state || !seq || seq->frame_count <= 0 || !om->face_indices) {
                    fprintf(stderr, "render: projectile model %u cannot render animation %d\n",
                            fp->model_id, fp->anim_id);
                    abort();
                }
                if (fp->anim_frame >= seq->frame_count) fp->anim_frame = 0;
                AnimSequenceFrame* sf = &seq->frames[fp->anim_frame];
                AnimFrameBase* fb = render_get_framebase(rc, sf->frame.framebase_id);
                if (!fb) {
                    fprintf(stderr, "render: projectile animation framebase %u is missing\n",
                            sf->frame.framebase_id);
                    abort();
                }
                anim_apply_frame(fp->anim_state, om->base_vertices, &sf->frame, fb);
                anim_update_mesh(om->mesh.vertices, fp->anim_state,
                    om->face_indices, om->mesh.triangleCount);
                UpdateMeshBuffer(om->mesh, 0, om->mesh.vertices,
                    om->mesh.triangleCount * 9 * sizeof(float), 0);
                proj_model = &om->model;
            } else if (fp->model_id > 0) {
                proj_model = render_get_proj_model(rc, fp->model_id);
            }
            if (!proj_model) {
                /* model_id 0 intentionally falls back to the generic style mesh */
                if (fp->style == 0 && rc->ranged_proj_model_ready)
                    proj_model = &rc->ranged_proj_model;
                else if (fp->style == 1 && rc->magic_proj_model_ready)
                    proj_model = &rc->magic_proj_model;
                else if (fp->style == 3 && rc->cloud_proj_model_ready)
                    proj_model = &rc->cloud_proj_model;
                else if (fp->style == 4 && rc->ranged_proj_model_ready)
                    proj_model = &rc->ranged_proj_model;  /* spawn orb reuses ranged mesh */
            }

            if (proj_model) {
                rlDisableBackfaceCulling();
                float pms = 1.0f / 128.0f;
                proj_model->transform = render_projectile_transform_offset(
                    pms, pms, pms, fp->yaw, fp->pitch, pos,
                    fp->offset_x, fp->offset_y, fp->offset_z);
                DrawModel(*proj_model, (Vector3){0,0,0}, 1.0f, WHITE);
                rlEnableBackfaceCulling();
            }

            /* trail line from source to current position */
            if (rc->show_debug &&
                fp->motion_mode != ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED) {
                Color pc;
                switch (fp->style) {
                    case 0: pc = CLITERAL(Color){ 80, 220, 80, 150 }; break;
                    case 1: pc = CLITERAL(Color){ 80, 130, 255, 150 }; break;
                    case 2: pc = CLITERAL(Color){ 255, 80, 80, 150 }; break;
                    case 3: pc = CLITERAL(Color){ 50, 180, 50, 150 }; break;
                    case 4: pc = CLITERAL(Color){ 230, 230, 230, 150 }; break;
                    default: pc = WHITE; break;
                }
                Vector3 src_pos = { fp->src_x + 0.5f, src_ground + fp->start_height,
                                   -(fp->src_y + 1.0f) + 0.5f };
                DrawLine3D(src_pos, pos, pc);
                DrawSphere(pos, 0.12f, pc);
            }
        }

        /* safe spot markers: colored quads on ground at each stand location */
        if (rc->show_safe_spots && rc->gui.encounter_state) {
            const EncounterDef* edef_ss = (const EncounterDef*)rc->gui.encounter_def;
            if (edef_ss && strcmp(edef_ss->name, "zulrah") == 0) {
                ZulrahState* zs_ss = (ZulrahState*)rc->gui.encounter_state;
                const ZulRotationPhase* phase_ss = zul_current_phase(zs_ss);
                int act_stand = phase_ss->stand;
                int act_stall = phase_ss->stall;

                for (int si = 0; si < ZUL_NUM_STAND_LOCATIONS; si++) {
                    int lx = ZUL_STAND_COORDS[si][0];
                    int ly = ZUL_STAND_COORDS[si][1];
                    float sx = (float)(rc->arena_base_x + lx) + 0.5f;
                    float sz = -(float)(rc->arena_base_y + ly + 1) + 0.5f;
                    float gy = OV_GROUND(rc->arena_base_x + lx, rc->arena_base_y + ly);

                    Color col;
                    if (si == act_stand)
                        col = (Color){0, 255, 0, 160};
                    else if (si == act_stall)
                        col = (Color){255, 255, 0, 160};
                    else
                        col = (Color){0, 180, 180, 80};

                    DrawCube((Vector3){ sx, gy + 0.08f, sz },
                             0.7f, 0.04f, 0.7f, col);
                }
            }
        }

        #undef OV_GROUND
    }

    /* placed objects — disable backface culling since OSRS uses flat
       billboard-style quads for trees/plants (two crossing planes) */
    {
        /* use post-Zuk objects (prison walls removed) when Zuk is present */
        ObjectMesh* obj = (rc->objects_zuk && rc->objects_zuk->loaded && rc->zuk_active)
            ? rc->objects_zuk : rc->objects;
        if (obj && obj->loaded) {
            rlDisableBackfaceCulling();
            DrawModel(obj->model, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
            rlEnableBackfaceCulling();
        }
    }

    /* NPC models at spawn positions */
    if (rc->npcs && rc->npcs->loaded) {
        rlDisableBackfaceCulling();
        DrawModel(rc->npcs->model, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
        rlEnableBackfaceCulling();
    }

    /* entity 3D models: composite body + equipment, animated as one unit */
    if (rc->model_cache) {
        float ms = 1.0f / 128.0f;

        rlDisableBackfaceCulling();
        for (int i = 0; i < rc->entity_count; i++) {
            RenderEntity* ep = &rc->entities[i];

            /* skip invisible NPCs (diving, dead, etc.) */
            if (ep->entity_type == ENTITY_NPC && !ep->npc_visible) continue;

            float px, pz, ground;
            render_get_visual_pos(rc, i, &px, &pz, &ground);

            /* negate X scale to fix model mirroring: OSRS models are authored
               in a left-handed coordinate system but we render in right-handed
               (raylib/OpenGL). negating X flips the handedness so weapons
               appear in the correct (right) hand. */
            Matrix base = MatrixScale(-ms, ms, ms);
            base = MatrixMultiply(base, MatrixRotateY(rc->yaw[i]));
            base = MatrixMultiply(base, MatrixTranslate(px, ground, pz));

            /* rebuild composite if equipment changed, animate, upload, draw */
            render_player_composite(rc, i, base);


            /* project animated mesh vertices to 2D screen for convex hull click detection.
               ported from RuneLite RSModelMixin.getConvexHull → Perspective.modelToCanvas.
               we sample every Nth vertex for performance (full hull is overkill). */
            PlayerComposite* comp = &rc->composites[i];
            Camera3D hull_cam = render_build_3d_camera(rc);
            int nv = comp->face_count * 3;  /* actual used verts, not pre-allocated capacity */
            int stride = (nv > 200) ? (nv / 100) : 1;  /* sample ~100 verts max */
            int hull_n = 0;
            float min_model_y = 1000000.0f;
            float max_model_y = -1000000.0f;
            /* stack arrays for projection — max 200 sampled points */
            int hull_xs[256], hull_ys[256];
            for (int vi = 0; vi < nv; vi++) {
                float vy = comp->mesh.vertices[vi * 3 + 1];
                if (vy < min_model_y) min_model_y = vy;
                if (vy > max_model_y) max_model_y = vy;
            }
            if (nv > 0 && min_model_y <= max_model_y) {
                rc->entity_visual_mid_y[i] = ground + min_model_y * ms
                    + (max_model_y - min_model_y) * ms * 0.5f;
                rc->entity_visual_top_y[i] = ground + max_model_y * ms;
            } else {
                int ent_size = (ep->entity_type == ENTITY_NPC && ep->npc_size > 1) ? ep->npc_size : 1;
                rc->entity_visual_mid_y[i] = ground + 0.75f + 0.25f * (float)ent_size;
                rc->entity_visual_top_y[i] = ground + 1.5f + 0.5f * (float)ent_size;
            }
            for (int vi = 0; vi < nv && hull_n < 256; vi += stride) {
                float vx = comp->mesh.vertices[vi * 3 + 0];
                float vy = comp->mesh.vertices[vi * 3 + 1];
                float vz = comp->mesh.vertices[vi * 3 + 2];
                /* transform model-space vertex by the same matrix used for drawing */
                Vector3 wv = Vector3Transform((Vector3){ vx, vy, vz }, base);
                Vector2 sv = GetWorldToScreen(wv, hull_cam);
                /* skip off-screen / behind camera */
                if (sv.x < -1000 || sv.x > 5000 || sv.y < -1000 || sv.y > 5000) continue;
                hull_xs[hull_n] = (int)sv.x;
                hull_ys[hull_n] = (int)sv.y;
                hull_n++;
            }
            hull_compute(hull_xs, hull_ys, hull_n, &rc->entity_hulls[i]);
        }
        rlEnableBackfaceCulling();
    }

    /* visual effects: spell impacts, projectiles */
    if (rc->model_cache) {
        rlDisableBackfaceCulling();
        float eff_scale = 1.0f / 128.0f;
        int eff_ct = rc->effect_client_tick_counter;

        for (int i = 0; i < MAX_ACTIVE_EFFECTS; i++) {
            ActiveEffect* e = &rc->effects[i];
            if (e->type == EFFECT_NONE) continue;
            if (!e->meta) continue;

            /* look up model */
            OsrsModel* om = model_cache_get(rc->model_cache, e->meta->model_id);
            if (!om && rc->npc_model_cache)
                om = model_cache_get(rc->npc_model_cache, e->meta->model_id);
            if (!om) continue;

            /* position: sub-tile coords -> tile coords -> raylib world */
            float ex = (float)(e->cur_x / 128.0);
            float ez = -(float)(e->cur_y / 128.0);
            float ground = rc->terrain
                ? terrain_height_avg(rc->terrain, (int)ex, (int)(e->cur_y / 128.0))
                : 2.0f;
            float ey = ground + (float)(e->height / 128.0);

            /* apply scale from spotanim def */
            float sx = eff_scale * (float)e->meta->resize_xy / 128.0f;
            float sz = eff_scale * (float)e->meta->resize_z / 128.0f;

            /* animate: apply current frame to per-effect anim state,
               then write transformed vertices into the shared mesh.
               note: this temporarily modifies the shared OsrsModel mesh,
               which is fine since effects render sequentially. */
            if (e->anim_state && e->meta->anim_seq_id >= 0 && rc->anim_cache
                && om->face_indices) {
                AnimSequence* seq = render_get_anim_sequence(rc, e->meta->anim_seq_id);
                if (seq && e->anim_frame < seq->frame_count) {
                    AnimSequenceFrame* sf = &seq->frames[e->anim_frame];
                    AnimFrameBase* fb = render_get_framebase(rc,
                        sf->frame.framebase_id);
                    if (fb) {
                        anim_apply_frame(e->anim_state, om->base_vertices,
                            &sf->frame, fb);
                        anim_update_mesh(om->mesh.vertices, e->anim_state,
                            om->face_indices, om->mesh.triangleCount);
                        UpdateMeshBuffer(om->mesh, 0, om->mesh.vertices,
                            om->mesh.triangleCount * 9 * sizeof(float), 0);
                    }
                }
            }

            /* build transform */
            Matrix t;

            /* projectile orientation: yaw + pitch from trajectory direction.
               uses atan2 on the velocity vector (same approach as the flight
               system) to orient the model from source toward target. */
            if (e->type == EFFECT_PROJECTILE && e->started) {
                /* direction in raylib coords: x_increment maps to world X,
                   y_increment maps to world -Z (OSRS Y → raylib -Z) */
                float dx = (float)e->x_increment;
                float dz = -(float)e->y_increment;
                float horiz = sqrtf(dx * dx + dz * dz);
                float yaw = atan2f(dx, dz);
                float pitch = atan2f((float)e->height_increment, horiz > 0.001f ? horiz : 0.001f);
                t = render_projectile_transform(sx, sx, sz, yaw, pitch,
                    (Vector3){ ex, ey, ez });
            } else {
                t = MatrixMultiply(MatrixScale(-sx, sx, sz), MatrixTranslate(ex, ey, ez));
            }
            om->model.transform = t;

            /* spotanim fade: 20% fade in, 60% full, 20% fade out */
            Color tint = WHITE;
            if (e->type == EFFECT_SPOTANIM && e->stop_tick > e->start_tick) {
                int total = e->stop_tick - e->start_tick;
                int elapsed = eff_ct - e->start_tick;
                float progress = (float)elapsed / (float)total;
                float alpha = 1.0f;
                if (progress < 0.2f) alpha = progress / 0.2f;
                else if (progress > 0.8f) alpha = (1.0f - progress) / 0.2f;
                if (alpha < 0.0f) alpha = 0.0f;
                if (alpha > 1.0f) alpha = 1.0f;
                tint = (Color){ 255, 255, 255, (unsigned char)(alpha * 255) };
            }
            DrawModel(om->model, (Vector3){ 0, 0, 0 }, 1.0f, tint);
        }
        rlEnableBackfaceCulling();
    }

    /* fight area boundary wireframe (Z negated) */
    float fa_x = (float)rc->arena_base_x;
    float fa_z = -(float)rc->arena_base_y;
    float fa_w = (float)rc->arena_width;
    float fa_h = -(float)rc->arena_height;  /* negative because Z is negated */
    float bh = rc->terrain ? terrain_height_at(rc->terrain, rc->arena_base_x, rc->arena_base_y) : 2.0f;
    DrawLine3D(
        (Vector3){ fa_x, bh, fa_z },
        (Vector3){ fa_x + fa_w, bh, fa_z }, YELLOW);
    DrawLine3D(
        (Vector3){ fa_x + fa_w, bh, fa_z },
        (Vector3){ fa_x + fa_w, bh, fa_z + fa_h }, YELLOW);
    DrawLine3D(
        (Vector3){ fa_x + fa_w, bh, fa_z + fa_h },
        (Vector3){ fa_x, bh, fa_z + fa_h }, YELLOW);
    DrawLine3D(
        (Vector3){ fa_x, bh, fa_z + fa_h },
        (Vector3){ fa_x, bh, fa_z }, YELLOW);

    /* click cross is now drawn as 2D overlay in pvp_render, not in 3D world */

    /* debug: player→NPC LOS lines (green=can attack, red=blocked/out of range) */
    if (rc->show_debug && rc->gui.encounter_state) {
        InfernoState* is = (InfernoState*)rc->gui.encounter_state;
        float plat_y = 2.0f;
        float ph = plat_y + 1.0f;  /* player line height */
        float player_wx = (float)is->player.x + 0.5f;
        float player_wz = -(float)is->player.y - 0.5f;
        const EncounterLoadoutStats* ls = &is->loadout_stats[is->weapon_set];
        for (int ni = 0; ni < INF_MAX_NPCS; ni++) {
            InfNPC* npc = &is->npcs[ni];
            if (!npc->active || npc->death_ticks > 0) continue;
            float half = (float)(npc->size - 1) / 2.0f;
            float npc_wx = (float)npc->x + half + 0.5f;
            float npc_wz = -(float)npc->y - half - 0.5f;
            int can_atk = encounter_player_can_attack(
                is->player.x, is->player.y,
                npc->x, npc->y, npc->size,
                ls->attack_range, is->los_blockers, is->los_blocker_count);
            Color lc = can_atk ? GREEN : RED;
            DrawLine3D(
                (Vector3){ player_wx, ph, player_wz },
                (Vector3){ npc_wx, ph, npc_wz },
                lc);
        }
    }

    /* hover tile outline: semi-transparent cyan border on the tile under cursor.
       similar to RuneLite's "Tile Indicators" plugin. drawn as 4 lines slightly
       above ground to avoid z-fighting with terrain/floor. */
    if (rc->hover_tile_x >= 0) {
        float htx = (float)rc->hover_tile_x;
        float htz = -(float)(rc->hover_tile_y + 1);
        float hgy = rc->terrain
            ? terrain_height_avg(rc->terrain, rc->hover_tile_x, rc->hover_tile_y)
            : 2.0f;
        float hy = hgy + 0.03f;  /* slight offset above ground */
        Color hcol = CLITERAL(Color){ 0, 220, 220, 180 };
        DrawLine3D((Vector3){ htx,       hy, htz },       (Vector3){ htx + 1.0f, hy, htz },       hcol);
        DrawLine3D((Vector3){ htx + 1.0f, hy, htz },       (Vector3){ htx + 1.0f, hy, htz + 1.0f }, hcol);
        DrawLine3D((Vector3){ htx + 1.0f, hy, htz + 1.0f }, (Vector3){ htx,       hy, htz + 1.0f }, hcol);
        DrawLine3D((Vector3){ htx,       hy, htz + 1.0f }, (Vector3){ htx,       hy, htz },       hcol);
    }

    EndMode3D();
}

/* ======================================================================== */
/* drawing: 2D overlay models (for 2D mode)                                  */
/* ======================================================================== */

static void render_draw_models_2d_overlay(RenderClient* rc) {
    if (!rc->model_cache) return;

    float hw = (float)RENDER_WINDOW_W / 2.0f;
    float hh = (float)RENDER_WINDOW_H / 2.0f;

    Camera3D cam = { 0 };
    cam.position = (Vector3){ hw, hh, 1000.0f };
    cam.target = (Vector3){ hw, hh, 0.0f };
    cam.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    cam.fovy = (float)RENDER_WINDOW_H;
    cam.projection = CAMERA_ORTHOGRAPHIC;

    float zoom_cx = (float)RENDER_GRID_W / 2.0f;
    float zoom_cy = (float)(RENDER_HEADER_HEIGHT + RENDER_GRID_H / 2.0f);

    BeginMode3D(cam);

    for (int i = 0; i < rc->entity_count; i++) {
        RenderEntity* p = &rc->entities[i];

        uint8_t slot_idx = p->equipped[GEAR_SLOT_WEAPON];
        uint8_t db_idx = get_item_for_slot(GEAR_SLOT_WEAPON, slot_idx);
        if (db_idx == ITEM_NONE || db_idx >= NUM_ITEMS) continue;

        uint16_t item_id = ITEM_DATABASE[db_idx].item_id;
        uint32_t model_id = item_to_inv_model(item_id);
        if (model_id == 0xFFFFFFFF) continue;

        OsrsModel* om = model_cache_get(rc->model_cache, model_id);
        if (!om) continue;

        float sx = (float)render_world_to_screen_x_rc(rc, p->x) + (float)RENDER_TILE_SIZE / 2.0f;
        float sy = (float)render_world_to_screen_y_rc(rc, p->y) + (float)RENDER_TILE_SIZE / 2.0f;

        float zsx = zoom_cx + (sx - zoom_cx) * rc->zoom;
        float zsy = zoom_cy + (sy - zoom_cy) * rc->zoom;

        float wx = zsx;
        float wy = (float)RENDER_WINDOW_H - zsy;

        float scale = rc->model_scale * rc->zoom;

        Matrix transform = MatrixScale(scale, scale, scale);
        transform = MatrixMultiply(transform, MatrixTranslate(wx, wy, 0.0f));

        om->model.transform = transform;
        DrawModel(om->model, (Vector3){ 0, 0, 0 }, 1.0f, WHITE);
    }

    EndMode3D();
}

/* ======================================================================== */
/* overhead status: prayer icons + HP bar (2D overlay on 3D scene)            */
/* ======================================================================== */

/**
 * Draw overhead prayer icons and HP bars above players in 3D mode.
 *
 * Layout matches OSRS client (Client.java:6011-6049):
 * - HP bar: 30px wide, 5px tall, green fill + red remainder, drawn at
 *   entity height + 15 via npcScreenPos. visible for 6s after taking damage.
 * - Prayer icon: drawn above the HP bar.
 *
 * Both are 2D sprites/rects drawn at screen-projected head position.
 */
static void render_draw_overhead_status(RenderClient* rc, OsrsEnv* env) {
    Camera3D cam = render_build_3d_camera(rc);

    /* map our OverheadPrayer enum → OSRS headIcon sprite index */
    static const int prayer_to_headicon[] = {
        -1, /* PRAYER_NONE */
         2, /* PRAYER_PROTECT_MAGIC  → headIcon 2 (magic) */
         1, /* PRAYER_PROTECT_RANGED → headIcon 1 (ranged) */
         0, /* PRAYER_PROTECT_MELEE  → headIcon 0 (melee) */
         4, /* PRAYER_SMITE          → headIcon 4 (smite) */
         5, /* PRAYER_REDEMPTION     → headIcon 5 (redemption) */
    };

    for (int i = 0; i < rc->entity_count; i++) {
        RenderEntity* p = &rc->entities[i];

        /* skip invisible NPCs */
        if (p->entity_type == ENTITY_NPC && !p->npc_visible) continue;

        /* project entity positions to screen coordinates.
           OSRS draws splats at entity.height/2 (abdomen), HP bar + prayer at top.
           head height scales with NPC size — larger models need higher overhead bars.
           approximate: model height in tiles ~ 1.5 + 0.5*size (player=2.0, zuk=5.0). */
        float px, pz, ground;
        render_get_visual_pos(rc, i, &px, &pz, &ground);
        int ent_size = (p->entity_type == ENTITY_NPC && p->npc_size > 1) ? p->npc_size : 1;
        float head_y = rc->entity_visual_top_y[i];
        float abdomen_y = rc->entity_visual_mid_y[i];
        if (head_y <= abdomen_y) {
            head_y = ground + 1.5f + 0.5f * (float)ent_size;
            abdomen_y = ground + 0.75f + 0.25f * (float)ent_size;
        }
        Vector2 screen_head = GetWorldToScreen((Vector3){ px, head_y, pz }, cam);
        Vector2 screen_abdomen = GetWorldToScreen((Vector3){ px, abdomen_y, pz }, cam);

        /* skip if off screen */
        if (screen_head.x < -50 || screen_head.x > RENDER_WINDOW_W + 50 ||
            screen_head.y < -50 || screen_head.y > RENDER_WINDOW_H + 50) continue;

        /* hitsplats: drawn at entity.height/2 (abdomen) with slot-based layout.
           OSRS Client.java:6107 — npcScreenPos(entity, entity.height / 2).
           splats stay in place (no vertical drift in OSRS). */
        for (int si = 0; si < RENDER_SPLATS_PER_PLAYER; si++) {
            HitSplat* s = &rc->splats[i][si];
            if (!s->active) continue;
            int slot_dx, slot_dy;
            render_splat_slot_offset(si, &slot_dx, &slot_dy);
            int sx = (int)screen_abdomen.x + slot_dx;
            int sy = (int)screen_abdomen.y + slot_dy;
            render_draw_hitmark(rc, sx, sy, s->damage, s->hitmark_trans);
        }

        /* track vertical offset for stacking elements above the player.
           screen Y increases downward, so we go negative to go up. */
        float cursor_y = screen_head.y;

        /* HP bar: width scales with NPC size, matching OSRS HealthBarDefinition
           widths (30 for size 1, up to ~160 for size 7). plain colored rectangle
           matches the no-sprite fallback path in the engine. */
        if (env->tick < rc->hp_bar_visible_until[i]) {
            /* OSRS bar widths by common NPC sizes (from cache HealthBarDefinitions):
               size 1→30, 2→40, 3→50, 4→60, 5→80, 7→120 */
            static const int BAR_WIDTH_BY_SIZE[] = {
                30, 30, 40, 50, 60, 80, 100, 120
            };
            int bw_idx = ent_size;
            if (bw_idx > 7) bw_idx = 7;
            int bar_w = BAR_WIDTH_BY_SIZE[bw_idx];
            int bar_h = 5;
            float hp_frac = (float)p->current_hitpoints / (float)p->base_hitpoints;
            if (hp_frac < 0.0f) hp_frac = 0.0f;
            if (hp_frac > 1.0f) hp_frac = 1.0f;
            int green_w = (int)(hp_frac * bar_w);

            int bar_x = (int)screen_head.x - bar_w / 2;
            int bar_y = (int)cursor_y - bar_h / 2;
            DrawRectangle(bar_x, bar_y, green_w, bar_h, COLOR_HP_GREEN);
            DrawRectangle(bar_x + green_w, bar_y, bar_w - green_w, bar_h, COLOR_HP_RED);
            cursor_y -= (float)(bar_h + 2);
        }

        /* prayer icon: drawn above the HP bar */
        if (rc->prayer_icons_loaded &&
            p->prayer > PRAYER_NONE && p->prayer <= PRAYER_REDEMPTION) {
            int icon_idx = prayer_to_headicon[p->prayer];
            if (icon_idx >= 0 && icon_idx < 6) {
                Texture2D tex = rc->prayer_icons[icon_idx];
                float scale = 1.0f;
                float draw_x = screen_head.x - (float)tex.width * scale / 2.0f;
                float draw_y = cursor_y - (float)tex.height * scale;
                DrawTextureEx(tex, (Vector2){ draw_x, draw_y }, 0.0f, scale, WHITE);
            }
        }

        /* debug: per-NPC combat state below the entity (only for NPCs) */
        if (rc->show_debug && p->entity_type == ENTITY_NPC && rc->gui.encounter_state) {
            InfernoState* is = (InfernoState*)rc->gui.encounter_state;
            int slot = p->npc_slot;
            if (slot >= 0 && slot < INF_MAX_NPCS && is->npcs[slot].active) {
                InfNPC* npc = &is->npcs[slot];
                int dy = (int)screen_head.y + 10;
                int dx = (int)screen_head.x;
                int fs = 10;

                /* attack timer + style */
                const char* style_str = "???";
                Color style_col = WHITE;
                int style = (npc->type == INF_NPC_JAD) ? npc->jad_attack_style : npc->attack_style;
                if (style == ATTACK_STYLE_MAGIC)  { style_str = "MAG"; style_col = BLUE; }
                if (style == ATTACK_STYLE_RANGED) { style_str = "RNG"; style_col = GREEN; }
                if (style == ATTACK_STYLE_MELEE)  { style_str = "MEL"; style_col = RED; }

                const char* atk_txt = TextFormat("ATK:%d %s", npc->attack_timer, style_str);
                int tw = MeasureText(atk_txt, fs);
                DrawText(atk_txt, dx - tw/2, dy, fs, style_col);
                dy += fs + 1;

                /* frozen ticks */
                if (npc->frozen_ticks > 0) {
                    const char* frz_txt = TextFormat("FRZ:%d", npc->frozen_ticks);
                    int fw = MeasureText(frz_txt, fs);
                    DrawText(frz_txt, dx - fw/2, dy, fs, (Color){100, 200, 255, 255});
                    dy += fs + 1;
                }

                /* NPC→player LOS (skip nibblers — they target pillars, not player) */
                if (npc->type != INF_NPC_NIBBLER) {
                    int npc_los = inf_npc_has_los(is, slot);
                    const char* los_txt = npc_los ? "NPC>P" : "NPC>P X";
                    Color los_col = npc_los ? GREEN : RED;
                    int lw = MeasureText(los_txt, fs);
                    DrawText(los_txt, dx - lw/2, dy, fs, los_col);
                    dy += fs + 1;
                }

                /* player→NPC LOS + range */
                {
                    const EncounterLoadoutStats* ls = &is->loadout_stats[is->weapon_set];
                    int can_atk = encounter_player_can_attack(
                        is->player.x, is->player.y,
                        npc->x, npc->y, npc->size,
                        ls->attack_range, is->los_blockers, is->los_blocker_count);
                    const char* patk_txt = can_atk ? "P>NPC" : "P>NPC X";
                    Color patk_col = can_atk ? GREEN : RED;
                    int pw = MeasureText(patk_txt, fs);
                    DrawText(patk_txt, dx - pw/2, dy, fs, patk_col);
                    dy += fs + 1;
                }

                /* blob scan state */
                if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) {
                    const char* scan = "SCAN:???";
                    if (npc->blob_scanned_prayer == PRAYER_PROTECT_MAGIC) scan = "SCAN>RNG";
                    else if (npc->blob_scanned_prayer == PRAYER_PROTECT_RANGED) scan = "SCAN>MAG";
                    int sw = MeasureText(scan, fs);
                    DrawText(scan, dx - sw/2, dy, fs, YELLOW);
                }
            }
        }
    }
}

/* ======================================================================== */
/* main render entry point                                                   */
/* ======================================================================== */

void pvp_render(OsrsEnv* env) {
    RenderClient* rc = (RenderClient*)env->client;
    if (rc == NULL) {
        rc = render_make_client();
        env->client = rc;
    }

    /* ensure entity pointers are current (may be called without render_post_tick
       during pause, rewind, or initial frame) */
    render_populate_entities(rc, env);

    render_handle_input(rc, env);

    /* inventory mouse interaction (clicks, drags) — runs every frame.
       gui functions need the full Player* (inventory, stats, etc.) */
    if (rc->entity_count > 0 && rc->gui.gui_entity_idx < rc->entity_count) {
        Player* gui_p = render_get_player_ptr(env, rc->gui.gui_entity_idx);
        if (gui_p) gui_inv_handle_mouse(&rc->gui, gui_p, &rc->human_input);
    }

    /* run client ticks at 50 Hz (20ms each), matching the real OSRS client's
       processMovement() rate. both movement AND animation advance together
       in each client tick, keeping them perfectly in sync. */
    {
        double dt = GetFrameTime();
        rc->client_tick_accumulator += dt;
        double client_tick = 0.020;  /* 20ms = OSRS client tick */
        int steps = (int)(rc->client_tick_accumulator / client_tick);
        if (steps > 0) {
            rc->client_tick_accumulator -= steps * client_tick;
            /* cap to avoid spiral if frame rate drops badly */
            if (steps > 60) steps = 60;
            for (int s = 0; s < steps; s++) {
                for (int i = 0; i < rc->entity_count; i++) {
                    render_client_tick(rc, i);
                }
                /* advance visual effects, hitsplats, and projectile flights at 50 Hz */
                render_update_splats_client_tick(rc);
                flight_client_tick(rc);
                rc->effect_client_tick_counter++;
                effect_client_tick(rc->effects, rc->effect_client_tick_counter,
                    rc->anim_cache);
                gui_inv_tick(&rc->gui);
                human_tick_visuals(&rc->human_input);
            }
        }
    }

    BeginDrawing();
    ClearBackground(COLOR_BG);

    if (rc->mode_3d) {
        /* full 3D world view */
        render_draw_3d_world(rc);

        /* overhead prayer icons (2D overlay after 3D scene) */
        render_draw_overhead_status(rc, env);

        /* debug: player/global state panel (bottom-left) */
        if (rc->show_debug && env->encounter_state) {
            InfernoState* is = (InfernoState*)env->encounter_state;
            int dy = RENDER_WINDOW_H - 160;
            int dx = 10;
            int fs = 12;
            Color dc = (Color){220, 220, 220, 255};

            /* target */
            int tgt = is->interaction.target_slot;
            if (tgt >= 0 && tgt < INF_MAX_NPCS) {
                InfNPC* tn = &is->npcs[tgt];
                const char* tname = inferno_npc_name(INF_NPC_DEF_IDS[tn->type]);
                DrawText(TextFormat("TARGET: %s [#%d]", tname, tgt), dx, dy, fs, dc);
            } else {
                DrawText("TARGET: none", dx, dy, fs, (Color){180, 80, 80, 255});
            }
            dy += fs + 2;

            /* weapon + attack timer */
            const char* gear = is->weapon_set == INF_GEAR_MAGE ? "mage" :
                               is->weapon_set == INF_GEAR_TBOW ? "tbow" : "bp";
            const char* spell = is->spell_choice == ENCOUNTER_SPELL_ICE ? "ice" :
                                is->spell_choice == ENCOUNTER_SPELL_BLOOD ? "blood" : "none";
            DrawText(TextFormat("GEAR: %s  ATK: %d/%d  SPELL: %s", gear,
                is->player.attack_timer, is->loadout_stats[is->weapon_set].attack_speed, spell),
                dx, dy, fs, dc);
            dy += fs + 2;

            /* stats */
            DrawText(TextFormat("RNG:%d MAG:%d DEF:%d",
                is->player.current_ranged, is->player.current_magic, is->player.current_defence),
                dx, dy, fs, dc);
            dy += fs + 2;

            /* consumables */
            DrawText(TextFormat("BREW:%d REST:%d BAST:%d STAM:%d",
                is->player.brew_doses, is->player.restore_doses,
                is->player.bastion_doses, is->player.stamina_doses),
                dx, dy, fs, dc);
            dy += fs + 2;

            /* pending hits */
            int mag_hits = 0, rng_hits = 0;
            for (int h = 0; h < is->player_pending_hit_count; h++) {
                if (is->player_pending_hits[h].attack_style == ATTACK_STYLE_MAGIC) mag_hits++;
                else rng_hits++;
            }
            if (is->player_pending_hit_count > 0) {
                DrawText(TextFormat("INCOMING: %d (%dM %dR)",
                    is->player_pending_hit_count, mag_hits, rng_hits),
                    dx, dy, fs, (Color){255, 150, 150, 255});
            }
        }

        /* debug: draw entity convex hulls as 2D outlines */
        if (rc->show_debug) {
            for (int ei = 0; ei < rc->entity_count; ei++) {
                ConvexHull2D* h = &rc->entity_hulls[ei];
                if (h->count < 3) continue;
                Color col = (ei == rc->gui.gui_entity_idx)
                    ? (Color){ 0, 255, 0, 180 } : (Color){ 255, 0, 0, 180 };
                for (int hi = 0; hi < h->count; hi++) {
                    int ni = (hi + 1) % h->count;
                    DrawLine(h->xs[hi], h->ys[hi], h->xs[ni], h->ys[ni], col);
                }
            }
        }

        /* 3D mode HUD */
        int display_tick = env->tick;
        if (env->encounter_def && env->encounter_state)
            display_tick = ((const EncounterDef*)env->encounter_def)->get_tick(env->encounter_state);
        DrawText(TextFormat("Tick: %d  [3D MODE - T to toggle]", display_tick),
                 10, 12, 16, COLOR_TEXT);

        /* entity HP summary top-right */
        if (rc->entity_count >= 2) {
            RenderEntity* p0 = &rc->entities[0];
            RenderEntity* p1 = &rc->entities[1];
            const char* hp_txt = TextFormat("P0: %d/%d   P1: %d/%d",
                p0->current_hitpoints, p0->base_hitpoints,
                p1->current_hitpoints, p1->base_hitpoints);
            int hp_w = MeasureText(hp_txt, 16);
            DrawText(hp_txt, RENDER_WINDOW_W - hp_w - 10, 12, 16, COLOR_TEXT);
        }

        /* controls reminder */
        DrawText("Right-drag: orbit  Mid-drag: pan  Scroll: zoom  SPACE: pause  S: safe spots  D: debug  G: cycle entity  H: human",
                 10, RENDER_WINDOW_H - 20, 10, COLOR_TEXT_DIM);
    } else {
        /* 2D mode: existing grid view */
        Camera2D cam2d = { 0 };
        cam2d.offset = (Vector2){ (float)RENDER_GRID_W / 2.0f,
                                  (float)(RENDER_HEADER_HEIGHT + RENDER_GRID_H / 2.0f) };
        cam2d.target = cam2d.offset;
        /* camera follow in 2D: center on controlled entity */
        if (rc->human_input.enabled && rc->entity_count > 0) {
            int eidx = rc->gui.gui_entity_idx;
            if (eidx < rc->entity_count) {
                float px = ((float)rc->sub_x[eidx] / 128.0f - (float)rc->arena_base_x) *
                           RENDER_TILE_SIZE + RENDER_TILE_SIZE / 2.0f;
                float py = ((float)(rc->arena_height - 1) -
                           ((float)rc->sub_y[eidx] / 128.0f - (float)rc->arena_base_y)) *
                           RENDER_TILE_SIZE + RENDER_HEADER_HEIGHT + RENDER_TILE_SIZE / 2.0f;
                cam2d.target = (Vector2){ px, py };
            }
        }
        cam2d.zoom = rc->zoom;

        BeginMode2D(cam2d);
        render_draw_grid(rc, env);
        if (rc->show_safe_spots) render_draw_safe_spots(rc, env);
        render_draw_dest_markers(rc);
        render_draw_players(rc);
        render_draw_splats_2d(rc);
        EndMode2D();

        if (rc->show_models) {
            render_draw_models_2d_overlay(rc);
        }

        render_draw_header(rc, env);
        DrawText("SPACE: pause  C: collision  S: safe spots  G: cycle entity  H: human control",
                 10, RENDER_WINDOW_H - 20, 10, COLOR_TEXT_DIM);
    }

    /* OSRS GUI panel system: shows selected entity's state.
       Renders in both 2D and 3D mode as a side panel overlay.
       G key cycles through entities (player 0, player 1, NPCs, etc). */
    rc->gui.gui_entity_count = rc->entity_count;
    rc->gui.encounter_state = env->encounter_state;
    rc->gui.encounter_def = env->encounter_def;
    if (rc->gui.gui_entity_idx >= rc->entity_count)
        rc->gui.gui_entity_idx = 0;
    /* draw click cross at screen-space position (2D overlay, like real OSRS) */
    human_draw_click_cross(&rc->human_input,
                            rc->click_cross_sprites,
                            rc->click_cross_loaded);

    /* debug: show raycast tile selection info */
    if (rc->show_debug) {
        char dbg[256];
        snprintf(dbg, sizeof(dbg), "box: (%d,%d) plane: (%d,%d) hit3d: (%.1f,%.1f,%.1f)",
                 rc->debug_hit_wx, rc->debug_hit_wy,
                 rc->debug_plane_wx, rc->debug_plane_wy,
                 rc->debug_ray_hit_x, rc->debug_ray_hit_y, rc->debug_ray_hit_z);
        DrawText(dbg, 10, 30, 16, MAGENTA);
        snprintf(dbg, sizeof(dbg), "ray org: (%.1f,%.1f,%.1f) dir: (%.3f,%.3f,%.3f)",
                 rc->debug_ray_origin.x, rc->debug_ray_origin.y, rc->debug_ray_origin.z,
                 rc->debug_ray_dir.x, rc->debug_ray_dir.y, rc->debug_ray_dir.z);
        DrawText(dbg, 10, 48, 16, MAGENTA);
    }

    if (rc->entity_count > 0) {
        /* gui_draw needs full Player* for inventory/stats/prayers.
           render_get_player_ptr fetches from encounter vtable. */
        Player* gui_player = render_get_player_ptr(env, rc->gui.gui_entity_idx);
        /* plumb spell-targeting state into GUI so the selected spell shows a
           highlight border while awaiting enemy click */
        rc->gui.pending_spell_highlight = -1;
        if (rc->human_input.cursor_mode == CURSOR_SPELL_TARGET) {
            rc->gui.pending_spell_highlight = rc->human_input.selected_spell_gui_idx;
        }
        if (gui_player) gui_draw(&rc->gui, gui_player);

        /* boss/NPC info: top-left overlay (instead of below panel) */
        RenderEntity* gui_re = &rc->entities[rc->gui.gui_entity_idx];
        if (gui_re->entity_type != ENTITY_NPC && rc->entity_count > 1) {
            for (int ei = 0; ei < rc->entity_count; ei++) {
                if (rc->entities[ei].entity_type == ENTITY_NPC) {
                    render_draw_panel_npc(10, RENDER_HEADER_HEIGHT + 8,
                                         &rc->entities[ei], env);
                    break;
                }
            }
        }
    }

    /* right-click context menu: drawn last so it renders on top of everything */
    context_menu_draw(rc);

    EndDrawing();
}

#endif /* OSRS_RENDER_H */
