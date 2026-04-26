/**
 * @file encounter_inferno.h
 * @brief The Inferno — 69-wave PvM challenge with prayer switching and pillar safespotting.
 *
 * core mechanic: 3 destructible pillars block NPC projectiles. the player must
 * position behind pillars to limit incoming attacks to one prayer style at a time.
 * nibblers eat pillars, meleer can dig through them. losing all pillars = death spiral.
 *
 * monster types: nibbler (pillar eater), bat (short-range ranger), blob (prayer reader,
 * splits into 3 on death), meleer (burrows to player), ranger, mager (resurrects dead mobs),
 * jad (random 50/50 range/mage), zuk (final boss with shield mechanic).
 *
 * reference: InfernoTrainer TypeScript, runelite inferno plugin
 */

#ifndef ENCOUNTER_INFERNO_H
#define ENCOUNTER_INFERNO_H

#include "../osrs_types.h"
#include "../osrs_items.h"
#include "../osrs_monsters_generated.h"
#include "../osrs_collision.h"
#include "../osrs_combat.h"
#include "../osrs_special_attacks.h"
#include "../osrs_pvp_gear.h"
#include "../osrs_encounter.h"
#include "../osrs_interaction.h"
#include "../data/npc_models.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ======================================================================== */
/* arena constants                                                           */
/* ======================================================================== */

#define INF_ARENA_MIN_X    11
#define INF_ARENA_MAX_X    39
#define INF_ARENA_MIN_Y    14
#define INF_ARENA_MAX_Y    43
#define INF_ARENA_WIDTH    (INF_ARENA_MAX_X - INF_ARENA_MIN_X + 1)  /* 29 */
#define INF_ARENA_HEIGHT   (INF_ARENA_MAX_Y - INF_ARENA_MIN_Y + 1)  /* 30 */

#define INF_PLAYER_START_X 25
#define INF_PLAYER_START_Y 16
#define INF_ZUK_PLAYER_START_X 25
#define INF_ZUK_PLAYER_START_Y 42

#define INF_NUM_PILLARS   3
#define INF_PILLAR_SIZE   3
#define INF_PILLAR_HP     255

static const int INF_PILLAR_POS[INF_NUM_PILLARS][2] = {
    { 21, 20 },  /* south pillar */
    { 11, 34 },  /* west pillar */
    { 28, 36 },  /* north pillar */
};

/* 9 mob spawn positions (shuffled per wave) */
#define INF_NUM_SPAWN_POS 9
static const int INF_SPAWN_POS[INF_NUM_SPAWN_POS][2] = {
    {12, 38}, {33, 38}, {14, 32}, {34, 31}, {27, 26},
    {16, 20}, {34, 18}, {12, 15}, {26, 15},
};

/* nibbler spawn position (near pillars) */
#define INF_NIBBLER_SPAWN_X 20
#define INF_NIBBLER_SPAWN_Y 32

#define INF_MAX_TICKS     18000  /* 3 hours at 0.6s/tick */
#define INF_NUM_WAVES     69
#define INF_NUM_ACTION_HEADS 9

/* ======================================================================== */
/* NPC types                                                                 */
/* ======================================================================== */

typedef enum {
    INF_NPC_NIBBLER = 0,      /* Jal-Nib: melee, eats pillars */
    INF_NPC_BAT,              /* Jal-MejRah: short-range ranged, drains run */
    INF_NPC_BLOB,             /* Jal-Ak: prayer reader, splits into 3 on death */
    INF_NPC_BLOB_MELEE,       /* Jal-Ak-Rek-Ket: melee split from blob */
    INF_NPC_BLOB_RANGE,       /* Jal-Ak-Rek-Xil: range split from blob */
    INF_NPC_BLOB_MAGE,        /* Jal-Ak-Rek-Mej: mage split from blob */
    INF_NPC_MELEER,           /* Jal-ImKot: melee, can dig */
    INF_NPC_RANGER,           /* Jal-Xil: ranged, can melee if close */
    INF_NPC_MAGER,            /* Jal-Zek: magic, resurrects dead mobs, can melee if close */
    INF_NPC_JAD,              /* JalTok-Jad: random 50/50 range/mage */
    INF_NPC_ZUK,              /* TzKal-Zuk: final boss */
    INF_NPC_HEALER_JAD,       /* Yt-HurKot: jad healer */
    INF_NPC_HEALER_ZUK,       /* Jal-MejJak: zuk healer */
    INF_NPC_ZUK_SHIELD,       /* shield NPC during Zuk */
    INF_NUM_NPC_TYPES
} InfNPCType;

/* OSRS NPC definition IDs — maps InfNPCType enum to actual cache NPC IDs
 * used by the renderer to look up models/animations in npc_models.h */
static const int INF_NPC_DEF_IDS[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER]    = 7691,  /* Jal-Nib */
    [INF_NPC_BAT]        = 7692,  /* Jal-MejRah */
    [INF_NPC_BLOB]       = 7693,  /* Jal-Ak */
    [INF_NPC_BLOB_MELEE] = 7696,  /* Jal-AkRek-Ket (melee split) */
    [INF_NPC_BLOB_RANGE] = 7695,  /* Jal-AkRek-Xil (range split) */
    [INF_NPC_BLOB_MAGE]  = 7694,  /* Jal-AkRek-Mej (mage split) */
    [INF_NPC_MELEER]     = 7697,  /* Jal-ImKot */
    [INF_NPC_RANGER]     = 7698,  /* Jal-Xil */
    [INF_NPC_MAGER]      = 7699,  /* Jal-Zek */
    [INF_NPC_JAD]        = 7700,  /* JalTok-Jad */
    [INF_NPC_ZUK]        = 7706,  /* TzKal-Zuk */
    [INF_NPC_HEALER_JAD] = 7701,  /* Yt-HurKot */
    [INF_NPC_HEALER_ZUK] = 7708,  /* Jal-MejJak */
    [INF_NPC_ZUK_SHIELD] = 7707,  /* Ancestral Glyph */
};

typedef struct {
    int hp;
    int attack_speed;
    int attack_range;
    int size;
    int default_style;   /* ATTACK_STYLE_* */
    int melee_style;     /* MELEE_STYLE_STAB/SLASH/CRUSH for incoming defence bonus selection */
    int can_melee;       /* 1 if can switch to melee when close */

    /* combat levels (used for attack rolls and max hit computation) */
    int att_level, str_level, def_level, range_level, magic_level;

    /* attack bonuses (for NPC attack roll: (level + 9) * (bonus + 64)) */
    int melee_att_bonus;   /* best of stab/slash/crush */
    int range_att_bonus;
    int magic_att_bonus;

    /* strength bonuses (for max hit formulas) */
    int melee_str_bonus;   /* bonuses.other.meleeStrength */
    int ranged_str_bonus;  /* bonuses.other.rangedStrength */
    int magic_base_dmg;    /* base spell damage (magicMaxHit() in InfernoTrainer) */
    int magic_dmg_pct;     /* magic damage multiplier as % (100 = 1.0x) */

    /* defence bonuses (for player hit chance against this NPC) */
    int stab_def, slash_def, crush_def;
    int magic_def_bonus;
    int ranged_def_bonus;

    /* wiki max hit cap: 0 = no cap (use formula), >0 = clamp to this value.
       needed for Jad/Zuk where InfernoTrainer multipliers overshoot wiki values. */
    int max_hit_cap;

    int stun_on_spawn;   /* ticks of stun when first spawned */
    int can_move;        /* 0 = cannot move (zuk, zuk healers) */
} InfNPCStats;

/* encounter-specific fields not in the generated monster database */
typedef struct {
    int attack_range;
    int default_style;   /* ATTACK_STYLE_* */
    int melee_style;     /* MELEE_STYLE_STAB/SLASH/CRUSH for incoming defence bonus selection */
    int can_melee;       /* 1 if can switch to melee when close */
    int magic_base_dmg;  /* base spell damage (magicMaxHit() in InfernoTrainer) */
    int magic_dmg_pct;   /* magic damage multiplier as % (100 = 1.0x) */
    int max_hit_cap;     /* 0 = no cap, >0 = clamp formula result to this */
    int stun_on_spawn;   /* ticks of stun when first spawned */
    int can_move;        /* 0 = cannot move (zuk, zuk healers) */
} InfNPCOverlay;

/* maps InfNPCType -> MonsterIndex for MONSTER_DATABASE lookup */
static const MonsterIndex INF_NPC_TO_MON[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER]    = MON_JAL_NIB,
    [INF_NPC_BAT]        = MON_JAL_MEJRAH,
    [INF_NPC_BLOB]       = MON_JAL_AK,
    [INF_NPC_BLOB_MELEE] = MON_JAL_AKREK_KET,
    [INF_NPC_BLOB_RANGE] = MON_JAL_AKREK_XIL,
    [INF_NPC_BLOB_MAGE]  = MON_JAL_AKREK_MEJ,
    [INF_NPC_MELEER]     = MON_JAL_IMKOT,
    [INF_NPC_RANGER]     = MON_JAL_XIL,
    [INF_NPC_MAGER]      = MON_JAL_ZEK,
    [INF_NPC_JAD]        = MON_JALTOK_JAD,
    [INF_NPC_ZUK]        = MON_TZKAL_ZUK,
    [INF_NPC_HEALER_JAD] = MON_YT_HURKOT,
    [INF_NPC_HEALER_ZUK] = MON_JAL_MEJJAK,
    [INF_NPC_ZUK_SHIELD] = MON_ZUK_SHIELD,
};

/* encounter-specific overlay: fields the generated DB doesn't cover */
static const InfNPCOverlay INF_NPC_OVERLAY[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER]    = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_CRUSH, 0,   0,   0, 0, 1, 1 },
    [INF_NPC_BAT]        = { 4,  ATTACK_STYLE_RANGED, MELEE_STYLE_STAB,  0,   0,   0, 0, 0, 1 },
    [INF_NPC_BLOB]       = { 15, ATTACK_STYLE_MAGIC,  MELEE_STYLE_CRUSH, 1,  29, 100, 0, 0, 1 },
    [INF_NPC_BLOB_MELEE] = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_CRUSH, 0,   0,   0, 0, 0, 1 },
    [INF_NPC_BLOB_RANGE] = { 15, ATTACK_STYLE_RANGED, MELEE_STYLE_STAB,  0,   0,   0, 0, 0, 1 },
    [INF_NPC_BLOB_MAGE]  = { 15, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  0,  18, 100, 0, 0, 1 },
    [INF_NPC_MELEER]     = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_SLASH, 0,   0,   0, 0, 0, 1 },
    [INF_NPC_RANGER]     = { 15, ATTACK_STYLE_RANGED, MELEE_STYLE_CRUSH, 1,   0,   0, 0, 0, 1 },
    [INF_NPC_MAGER]      = { 15, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  1,  70, 100, 0, 0, 1 },
    [INF_NPC_JAD]        = { 50, ATTACK_STYLE_RANGED, MELEE_STYLE_STAB,  1, 113, 100, 113, 0, 1 },
    [INF_NPC_ZUK]        = { 99, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  0, 148, 100, 0, 8, 0 },
    [INF_NPC_HEALER_JAD] = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_CRUSH, 0,   0,   0, 0, 1, 1 },  /* stun_on_spawn=1 per YtHurKot.ts:50 */
    [INF_NPC_HEALER_ZUK] = { 99, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  0,  10, 100, 0, 1, 0 },  /* stun_on_spawn=1 per InfernoTrainer JalMejJak.ts SPAWN_DELAY */
    [INF_NPC_ZUK_SHIELD] = { 0,  ATTACK_STYLE_NONE,   MELEE_STYLE_STAB,  0,   0,   0, 0, 1, 0 },
};

/* populated at startup by inf_build_npc_stats() */
static InfNPCStats INF_NPC_STATS[INF_NUM_NPC_TYPES];

/* merge MONSTER_DATABASE + INF_NPC_OVERLAY into INF_NPC_STATS */
static void inf_build_npc_stats(void) {
    for (int i = 0; i < INF_NUM_NPC_TYPES; i++) {
        const MonsterStats* m = &MONSTER_DATABASE[INF_NPC_TO_MON[i]];
        const InfNPCOverlay* o = &INF_NPC_OVERLAY[i];
        InfNPCStats* s = &INF_NPC_STATS[i];

        /* from generated monster DB */
        s->hp              = m->hp;
        s->attack_speed    = m->attack_speed;
        s->size            = m->size;
        s->att_level       = m->att_level;
        s->str_level       = m->str_level;
        s->def_level       = m->def_level;
        s->range_level     = m->range_level;
        s->magic_level     = m->magic_level;
        s->melee_att_bonus = m->melee_att_bonus;
        s->range_att_bonus = m->range_att_bonus;
        s->magic_att_bonus = m->magic_att_bonus;
        s->melee_str_bonus = m->melee_str_bonus;
        s->ranged_str_bonus = m->ranged_str_bonus;
        s->stab_def        = m->stab_def;
        s->slash_def       = m->slash_def;
        s->crush_def       = m->crush_def;
        s->magic_def_bonus = m->magic_def;    /* name mapping */
        s->ranged_def_bonus = m->ranged_def;  /* name mapping */

        /* from encounter overlay */
        s->attack_range    = o->attack_range;
        s->default_style   = o->default_style;
        s->melee_style     = o->melee_style;
        s->can_melee       = o->can_melee;
        s->magic_base_dmg  = o->magic_base_dmg;
        s->magic_dmg_pct   = o->magic_dmg_pct;
        s->max_hit_cap     = o->max_hit_cap;
        s->stun_on_spawn   = o->stun_on_spawn;
        s->can_move        = o->can_move;
    }
}

/* ======================================================================== */
/* wave compositions                                                         */
/* ======================================================================== */

#define INF_MAX_NPCS_PER_WAVE 9  /* wave 62: NNN BB BL M R MA = 9 */

typedef struct {
    uint8_t types[INF_MAX_NPCS_PER_WAVE];
    int count;
} InfWaveDef;

static const InfWaveDef INF_WAVES[INF_NUM_WAVES] = {
    #define N INF_NPC_NIBBLER
    #define B INF_NPC_BAT
    #define BL INF_NPC_BLOB
    #define M INF_NPC_MELEER
    #define R INF_NPC_RANGER
    #define MA INF_NPC_MAGER
    #define J INF_NPC_JAD
    #define Z INF_NPC_ZUK
    #define W(...) { .types = { __VA_ARGS__ }, .count = sizeof((uint8_t[]){__VA_ARGS__}) }

    /* waves 1-8: bats + nibblers introduction */
    [0]  = W(N,N,N, B),
    [1]  = W(N,N,N, B,B),
    [2]  = W(N,N,N, N,N,N),
    [3]  = W(N,N,N, BL),
    [4]  = W(N,N,N, B,BL),
    [5]  = W(N,N,N, B,B,BL),
    [6]  = W(N,N,N, BL,BL),
    [7]  = W(N,N,N, N,N,N),

    /* waves 9-17: meleer introduction */
    [8]  = W(N,N,N, M),
    [9]  = W(N,N,N, B,M),
    [10] = W(N,N,N, B,B,M),
    [11] = W(N,N,N, BL,M),
    [12] = W(N,N,N, B,BL,M),
    [13] = W(N,N,N, B,B,BL,M),
    [14] = W(N,N,N, BL,BL,M),
    [15] = W(N,N,N, M,M),
    [16] = W(N,N,N, N,N,N),

    /* waves 18-34: ranger introduction */
    [17] = W(N,N,N, R),
    [18] = W(N,N,N, B,R),
    [19] = W(N,N,N, B,B,R),
    [20] = W(N,N,N, BL,R),
    [21] = W(N,N,N, B,BL,R),
    [22] = W(N,N,N, B,B,BL,R),
    [23] = W(N,N,N, BL,BL,R),
    [24] = W(N,N,N, M,R),
    [25] = W(N,N,N, B,M,R),
    [26] = W(N,N,N, B,B,M,R),
    [27] = W(N,N,N, BL,M,R),
    [28] = W(N,N,N, B,BL,M,R),
    [29] = W(N,N,N, B,B,BL,M,R),
    [30] = W(N,N,N, BL,BL,M,R),
    [31] = W(N,N,N, M,M,R),
    [32] = W(N,N,N, R,R),
    [33] = W(N,N,N, N,N,N),

    /* waves 35-66: mager introduction (all combinations) */
    [34] = W(N,N,N, MA),
    [35] = W(N,N,N, B,MA),
    [36] = W(N,N,N, B,B,MA),
    [37] = W(N,N,N, BL,MA),
    [38] = W(N,N,N, B,BL,MA),
    [39] = W(N,N,N, B,B,BL,MA),
    [40] = W(N,N,N, BL,BL,MA),
    [41] = W(N,N,N, M,MA),
    [42] = W(N,N,N, B,M,MA),
    [43] = W(N,N,N, B,B,M,MA),
    [44] = W(N,N,N, BL,M,MA),
    [45] = W(N,N,N, B,BL,M,MA),
    [46] = W(N,N,N, B,B,BL,M,MA),
    [47] = W(N,N,N, BL,BL,M,MA),
    [48] = W(N,N,N, M,M,MA),
    [49] = W(N,N,N, R,MA),
    [50] = W(N,N,N, B,R,MA),
    [51] = W(N,N,N, B,B,R,MA),
    [52] = W(N,N,N, BL,R,MA),
    [53] = W(N,N,N, B,BL,R,MA),
    [54] = W(N,N,N, B,B,BL,R,MA),
    [55] = W(N,N,N, BL,BL,R,MA),
    [56] = W(N,N,N, M,R,MA),
    [57] = W(N,N,N, B,M,R,MA),
    [58] = W(N,N,N, B,B,M,R,MA),
    [59] = W(N,N,N, BL,M,R,MA),
    [60] = W(N,N,N, B,BL,M,R,MA),
    [61] = W(N,N,N, B,B,BL,M,R,MA),
    [62] = W(N,N,N, BL,BL,M,R,MA),
    [63] = W(N,N,N, M,M,R,MA),
    [64] = W(N,N,N, R,R,MA),
    [65] = W(N,N,N, MA,MA),

    /* waves 67-69: jads + zuk */
    [66] = W(J),
    [67] = W(J,J,J),
    [68] = W(Z),

    #undef N
    #undef B
    #undef BL
    #undef M
    #undef R
    #undef MA
    #undef J
    #undef Z
    #undef W
};

/* ======================================================================== */
/* NPC state                                                                 */
/* ======================================================================== */

/* max active NPCs: wave 62 has 9 + blob splits (3 per blob, up to 2 blobs = 6) + healers */
#define INF_MAX_NPCS      32
#define INF_OBS_NPCS      37

/* dead mob store for mager resurrection */
#define INF_MAX_DEAD_MOBS 16

typedef struct {
    InfNPCType type;
    int x, y;
    int hp, max_hp;
} InfDeadMob;

/* 4 zuk healers × 3 sparks per volley × 2 overlapping volleys = 24.
   cap is 32 for headroom so volleys don't silently drop sparks. */
#define INF_MAX_PENDING_SPARKS 32

typedef struct {
    int active;
    int src_x, src_y;
    int x, y;
    int damage;
    int ticks_remaining;
    int visual_emitted;
} InfPendingSpark;

typedef struct {
    InfNPCType type;
    int x, y;
    int hp, max_hp;
    int size;
    int attack_timer;      /* ticks until next attack */
    int attack_style;      /* current attack style (may differ from default for blobs) */
    int active;
    int target_x, target_y; /* movement destination */
    int stun_timer;        /* ticks of stun remaining (cannot act) */

    /* type-specific state */
    int no_los_ticks;      /* meleer: consecutive ticks without LOS to player */
    int dig_freeze_timer;  /* meleer: ticks remaining in dig animation */
    int dig_attack_delay;  /* meleer: ticks after emerging before first attack */

    /* blob prayer-reading state */
    int blob_scan_timer;   /* blob: ticks remaining in scan phase (reads prayer) */
    int blob_scanned_prayer; /* blob: prayer read during scan (OverheadPrayer value) */
    int had_los_last_tick; /* blob: previous-tick LOS latch for immediate scans on LOS gain */

    /* jad state */
    int jad_attack_style;  /* jad: committed next ranged/magic style preview, or NONE if unknown */
    int jad_healer_spawned; /* jad: 1 if healers have been spawned */
    int jad_owner_idx;     /* healer: which jad this healer belongs to (-1 = none) */

    /* mager resurrection state */
    int resurrect_cooldown; /* mager: ticks until next resurrection attempt */
    int resurrection_count; /* number of times this original mob has been resurrected */

    /* freeze state (ice barrage) */
    int frozen_ticks;       /* ticks remaining in ice barrage freeze */

    /* heal state */
    int heal_target;       /* healer: NPC index being healed (-1 = none) */
    int heal_timer;        /* healer: ticks until next heal tick */

    /* pending hit from player attack (projectile in flight) */
    EncounterPendingHit pending_hit;

    /* death linger: NPC stays visible for death animation + final hitsplat.
       >0 means dying (decremented each tick), 0 = alive or fully removed. */
    int death_ticks;

    /* aggro target: -1 = player (default), >= 0 = NPC index.
       used for set→shield targeting and healer→zuk healing. */
    int aggro_target;

    /* per-tick render flags (cleared at start of each tick) */
    int attacked_this_tick;     /* 1 when NPC attacks this tick */
    int attack_style_this_tick; /* actual style that fired this tick */
    int attack_visual_target;   /* NPC index this attack visually targets (-1 = player) */
    int moved_this_tick;        /* 1 when NPC moves this tick */
    int hit_landed_this_tick; /* 1 when this NPC was hit by player */
    int hit_damage;          /* damage dealt to this NPC this tick */
    int hit_spell_type;      /* ENCOUNTER_SPELL_* from the pending hit that just landed */
} InfNPC;

/* ======================================================================== */
/* pillar state                                                              */
/* ======================================================================== */

typedef struct {
    int x, y;
    int hp;
    int active;
} InfPillar;

/* ======================================================================== */
/* zuk state                                                                 */
/* ======================================================================== */

typedef struct {
    /* shield */
    int shield_idx;        /* NPC index of shield (-1 if dead) */
    int shield_dir;        /* +1 or -1 */
    int shield_freeze;     /* ticks of freeze at boundary */

    /* spawn timers */
    int initial_delay;     /* 14 ticks before first attack */
    int set_timer;         /* ticks until next set spawn (starts at 72) */
    int set_interval;      /* 350 ticks between set spawns */
    int enraged;           /* 1 when HP < 240 */

    int healer_spawned;    /* 1 when healers have been spawned */
    int jad_spawned;       /* 1 when jad has been spawned during shield phase */

    /* set timer pause: pauses between HP 600→480, resumes with +175 when jad spawns */
    int timer_paused;
    int has_paused;
} InfZukState;

/* ======================================================================== */
/* weapon sets and pre-computed stats                                         */
/* ======================================================================== */

typedef enum {
    INF_GEAR_MAGE = 0,
    INF_GEAR_TBOW,
    INF_GEAR_BP,
    INF_NUM_WEAPON_SETS
} InfWeaponSet;

/* gear loadout arrays per weapon set */
static const uint8_t INF_MAGE_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_MASORI_MASK_F,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_OCCULT_NECKLACE,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_ARROWS,
    [GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND,
    [GEAR_SLOT_SHIELD] = ITEM_ELYSIAN_SPIRIT_SHIELD,
    [GEAR_SLOT_BODY]   = ITEM_VIRTUS_ROBE_TOP,
    [GEAR_SLOT_LEGS]   = ITEM_VIRTUS_ROBE_BOTTOM,
    [GEAR_SLOT_HANDS]  = ITEM_CONFLICTION_GAUNTLETS,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t INF_RANGE_TBOW_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_MASORI_MASK_F,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_NECKLACE_OF_ANGUISH,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_ARROWS,
    [GEAR_SLOT_WEAPON] = ITEM_TWISTED_BOW,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,  /* tbow is 2h */
    [GEAR_SLOT_BODY]   = ITEM_MASORI_BODY_F,
    [GEAR_SLOT_LEGS]   = ITEM_MASORI_CHAPS_F,
    [GEAR_SLOT_HANDS]  = ITEM_ZARYTE_VAMBRACES,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t INF_RANGE_BP_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_MASORI_MASK_F,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_NECKLACE_OF_ANGUISH,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_DART,
    [GEAR_SLOT_WEAPON] = ITEM_TOXIC_BLOWPIPE,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,  /* bp is 2h */
    [GEAR_SLOT_BODY]   = ITEM_MASORI_BODY_F,
    [GEAR_SLOT_LEGS]   = ITEM_MASORI_CHAPS_F,
    [GEAR_SLOT_HANDS]  = ITEM_ZARYTE_VAMBRACES,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

/* pointer array for loadout switching */
static const uint8_t* const INF_LOADOUTS[INF_NUM_WEAPON_SETS] = {
    INF_MAGE_LOADOUT,
    INF_RANGE_TBOW_LOADOUT,
    INF_RANGE_BP_LOADOUT,
};

/* tank overlay items (justiciar) */
/* ======================================================================== */
/* encounter state                                                           */
/* ======================================================================== */

typedef struct {
    Player player;

    InfNPC npcs[INF_MAX_NPCS];
    int current_obs_slots[INF_OBS_NPCS];
    InfPillar pillars[INF_NUM_PILLARS];
    InfZukState zuk;

    /* dead mob store for mager resurrection */
    InfDeadMob dead_mobs[INF_MAX_DEAD_MOBS];
    int dead_mob_count;

    /* LOS blockers (rebuilt when pillars change) */
    LOSBlocker los_blockers[INF_NUM_PILLARS];
    int los_blocker_count;

    /* wave tracking */
    int wave;              /* current wave (0-indexed, 0-68) */
    int wave_spawn_target; /* wave index queued to spawn when the inter-wave delay ends */
    int tick;
    int wave_spawn_delay;  /* ticks until first wave spawns (0 = spawn immediately) */
    int episode_over;
    int winner;            /* 0 = player won (zuk dead), 1 = player died */

    /* reward tracking */
    float reward;
    float episode_return;  /* accumulated reward over entire episode */
    float min_zuk_hp_seen; /* final-wave low watermark for irreversible Zuk progress */
    float damage_dealt_this_tick;
    float damage_zuk_healers_this_tick;
    float shield_damage_this_tick;
    int healer_tags_this_tick;
    float damage_received_this_tick;
    float hp_restored_this_tick;
    int prayer_correct_this_tick;  /* count of NPC attacks blocked by prayer this tick */
    int wave_completed_this_tick;
    int pillar_lost_this_tick;     /* -1 = none, 0-2 = which pillar was destroyed */

    /* cumulative stats for diagnostics */
    float total_damage_dealt;
    float total_zuk_healer_damage;
    float total_damage_received;
    float total_hp_restored;   /* cumulative HP restored to enemies this episode */
    int total_waves_cleared;
    int ticks_without_action;  /* consecutive ticks with no attack or movement */
    int total_prayer_correct;  /* times prayer blocked an NPC attack */
    int total_npc_attacks;     /* total NPC attacks on player (for prayer_correct_rate) */
    int total_unavoidable_off; /* off-prayer hits where a different style was correctly prayed */
    int off_prayer_hits_this_tick;
    /* per-tick tracking for multi-style analysis */
    int tick_styles_fired;     /* bitmask of styles that fired this tick (bit0=mel,1=rng,2=mag) */
    int tick_attacks_fired;    /* count of NPC attacks that fired this tick */
    /* per-NPC-type prayer and damage tracking (for wandb, not dashboard) */
    int prayer_correct_by_type[INF_NUM_NPC_TYPES];
    int attacks_by_type[INF_NUM_NPC_TYPES];
    float dmg_from_type[INF_NUM_NPC_TYPES];
    int last_hit_by_type;      /* NPC type that last dealt damage to player (-1=none) */
    int killed_by_type[INF_NUM_NPC_TYPES];  /* count of deaths caused by each NPC type */
    int total_idle_ticks;      /* cumulative ticks of ticks_without_action > 0 */
    int total_brews_used;      /* brew doses consumed this episode */
    int total_blood_healed;    /* HP healed via blood barrage this episode */
    int total_npc_kills;       /* NPCs killed this episode */
    int total_gear_switches;   /* gear switch actions this episode */

    /* Zuk-specific diagnostics */
    int behind_shield_ticks;       /* ticks spent behind shield during Zuk wave */
    int behind_shield_this_tick;   /* 1 if behind shield this tick (for reward) */
    int total_zuk_ticks;           /* total ticks during Zuk wave (for behind_shield_pct) */

    /* action distribution: count of action-0 (noop) per head.
       high noop_rate = policy collapsed to doing nothing on that head. */
    int action_noop_count[INF_NUM_ACTION_HEADS];
    int action_total_count;    /* total ticks (denominator for noop rates) */

    /* per-tick reward event flags (cleared each tick) */
    int brewed_this_tick;      /* 1 if player drank a brew this tick */
    int blood_heal_this_tick;  /* HP healed from blood barrage this tick */

    /* player combat state */
    OsrsInteraction interaction;  /* shared interaction state */
    int player_last_interaction_target_slot;
    int player_last_interaction_age;

    /* gear state */
    InfWeaponSet weapon_set;
    EncounterLoadoutStats loadout_stats[INF_NUM_WEAPON_SETS];
    int human_command_mode;
    EncounterLoadoutStats human_loadout_stats;
    const HumanCommand* human_commands;
    int human_command_count;
    int armor_tank;            /* reserved loadout slot; justiciar overlay removed */
    int stamina_active_ticks;  /* countdown for stamina effect */
    int spell_choice;          /* 0 = blood barrage, 1 = ice barrage */

    /* per-tick player attack event for renderer projectiles */
    int player_attacked_this_tick;  /* 1 if player fired an attack this tick */
    int player_attack_npc_idx;      /* NPC index targeted by player attack */
    int player_attack_dmg;          /* total damage dealt */
    int player_attack_style_id;     /* ATTACK_STYLE_* of the player attack */

    /* pending hits on player from NPC attacks (projectiles in flight) */
    EncounterPendingHit player_pending_hits[ENCOUNTER_MAX_PENDING_HITS];
    int player_pending_hit_count;
    InfPendingSpark pending_sparks[INF_MAX_PENDING_SPARKS];

    /* nibbler pillar target: random pillar chosen per wave, all nibblers attack it */
    int nibbler_target_pillar;

    /* spawn position shuffle buffer */
    int spawn_order[INF_NUM_SPAWN_POS];

    /* collision map (loaded from cache, passed via put_ptr) */
    const CollisionMap* collision_map;
    int world_offset_x, world_offset_y;

    /* human click-to-move destination (-1 = no dest) */
    int player_dest_x, player_dest_y;

    /* per-tick LOS cache: lazy — computed on first access, reused for rest of tick.
       -1 = not yet computed, 0 = no LOS, 1 = has LOS. invalidated at tick start
       and on pillar collapse. */
    int8_t npc_los_cache[INF_MAX_NPCS];

    /* OSRS entity collision flags. pathfinding ignores these, movement application checks them. */
    uint8_t npc_collision_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT];
    uint8_t player_collision_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT];

    /* config */
    int start_wave;        /* for curriculum: start from a later wave */
    uint32_t rng_state;
    float damage_reward_coeff;
    float shield_penalty_coeff;
    float tag_reward_coeff;

    Log log;
} InfernoState;

/* prayer check and RNG: use shared encounter_prayer_correct_for_style(),
   encounter_rand_int(), encounter_rand_float() from osrs_combat.h */

static void inf_shuffle_spawns(InfernoState* s) {
    for (int i = 0; i < INF_NUM_SPAWN_POS; i++)
        s->spawn_order[i] = i;
    encounter_shuffle(s->spawn_order, INF_NUM_SPAWN_POS, &s->rng_state);
}

/* ======================================================================== */
/* LOS helper: rebuild blocker array from active pillars                     */
/* ======================================================================== */

static void inf_rebuild_los(InfernoState* s) {
    s->los_blocker_count = 0;
    for (int i = 0; i < INF_NUM_PILLARS; i++) {
        if (s->pillars[i].active) {
            LOSBlocker* b = &s->los_blockers[s->los_blocker_count++];
            b->x = s->pillars[i].x;
            b->y = s->pillars[i].y;
            b->size = INF_PILLAR_SIZE;
            b->los_mask = LOS_FULL_MASK;
        }
    }
}

/* check if NPC at index i has LOS to player (uncached — direct ray-trace) */
static int inf_npc_has_los_direct(InfernoState* s, int i) {
    InfNPC* npc = &s->npcs[i];
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    return npc_has_line_of_sight(s->los_blockers, s->los_blocker_count,
                                 npc->x, npc->y, npc->size,
                                 s->player.x, s->player.y,
                                 stats->attack_range);
}

/* cached LOS check — lazy: computes on first access per tick, caches for reuse.
   cache entries: -1 = not yet computed, 0 = no LOS, 1 = has LOS. */
static int inf_npc_has_los(InfernoState* s, int i) {
    if (s->npc_los_cache[i] >= 0)
        return s->npc_los_cache[i];
    int result = inf_npc_has_los_direct(s, i);
    s->npc_los_cache[i] = (int8_t)result;
    return result;
}

/* invalidate entire LOS cache (call at start of tick and on pillar collapse) */
static inline void inf_invalidate_los_cache(InfernoState* s) {
    memset(s->npc_los_cache, -1, sizeof(s->npc_los_cache));
}

static inline int inf_is_final_wave(const InfernoState* s) {
    return s->wave == INF_NUM_WAVES - 1;
}

static int inf_find_live_zuk_idx(const InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].active && s->npcs[i].type == INF_NPC_ZUK) return i;
    }
    return -1;
}

enum {
    INF_STYLE_MASK_MELEE = 1 << 0,
    INF_STYLE_MASK_RANGED = 1 << 1,
    INF_STYLE_MASK_MAGIC = 1 << 2,
};

static inline int inf_attack_style_mask_bit(int style) {
    if (style == ATTACK_STYLE_MELEE) return INF_STYLE_MASK_MELEE;
    if (style == ATTACK_STYLE_RANGED) return INF_STYLE_MASK_RANGED;
    if (style == ATTACK_STYLE_MAGIC) return INF_STYLE_MASK_MAGIC;
    return 0;
}

static inline int inf_cardinal_contact_with_npc(int px, int py, int nx, int ny, int npc_size) {
    int cx = px < nx ? nx : (px > nx + npc_size - 1 ? nx + npc_size - 1 : px);
    int cy = py < ny ? ny : (py > ny + npc_size - 1 ? ny + npc_size - 1 : py);
    int dx = px - cx; if (dx < 0) dx = -dx;
    int dy = py - cy; if (dy < 0) dy = -dy;
    return dx + dy == 1;
}

static inline int inf_melee_fallback_possible(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    if (!stats->can_melee || planned_style == ATTACK_STYLE_MELEE || dist != 1)
        return 0;

    switch (npc->type) {
        case INF_NPC_RANGER:
        case INF_NPC_MAGER:
            return 1;
        case INF_NPC_BLOB:
        case INF_NPC_JAD:
            return inf_cardinal_contact_with_npc(
                s->player.x, s->player.y, npc->x, npc->y, npc->size);
        default:
            return 0;
    }
}

static inline int inf_attack_style_options_mask(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    int mask = inf_attack_style_mask_bit(planned_style);
    if (inf_melee_fallback_possible(s, npc, stats, planned_style, dist))
        mask |= INF_STYLE_MASK_MELEE;
    return mask;
}

static inline int inf_attack_style_from_mask(int style_mask) {
    if (style_mask == INF_STYLE_MASK_MELEE) return ATTACK_STYLE_MELEE;
    if (style_mask == INF_STYLE_MASK_RANGED) return ATTACK_STYLE_RANGED;
    if (style_mask == INF_STYLE_MASK_MAGIC) return ATTACK_STYLE_MAGIC;
    return ATTACK_STYLE_NONE;
}

static inline int inf_attack_style_obs_preview(int style_mask) {
    int primary_mask = style_mask & (INF_STYLE_MASK_RANGED | INF_STYLE_MASK_MAGIC);
    int primary_style = inf_attack_style_from_mask(primary_mask);
    if (primary_style != ATTACK_STYLE_NONE)
        return primary_style;
    return inf_attack_style_from_mask(style_mask);
}

static inline int inf_attack_style_telegraph_mask(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    /* Jad only telegraphs its ranged/magic branch. Melee is an immediate
       fallback choice at fire time, not a prayer-switch cue. */
    if (npc->type == INF_NPC_JAD)
        return inf_attack_style_mask_bit(planned_style);
    return inf_attack_style_options_mask(s, npc, stats, planned_style, dist);
}

static inline int inf_pending_hit_obs_timer(const EncounterPendingHit* ph) {
    if (ph->check_prayer && ph->prayer_check_delay > 0)
        return ph->prayer_check_delay;
    return ph->ticks_remaining;
}

#define INF_JAD_PROJECTILE_DELAY 3
#define INF_ANIM_JALTOK_JAD_MAGIC_ATTACK INF_GEN_ANIM_JALTOK_JAD_ATTACK_MAGIC
#define INF_ANIM_JALTOK_JAD_RANGED_ATTACK INF_GEN_ANIM_JALTOK_JAD_ATTACK_RANGED
#define INF_ANIM_JALTOK_JAD_MELEE_ATTACK INF_GEN_ANIM_JALTOK_JAD_ATTACK_MELEE

static inline int inf_jad_post_prayer_flight_ticks(int hit_delay) {
    int ticks = hit_delay - INF_JAD_PROJECTILE_DELAY;
    return ticks > 1 ? ticks : 1;
}

static inline int inf_jad_land_delay_from_fire(int hit_delay) {
    return INF_JAD_PROJECTILE_DELAY + inf_jad_post_prayer_flight_ticks(hit_delay);
}

static inline int inf_jad_visible_duration_ticks(int hit_delay) {
    return (inf_jad_post_prayer_flight_ticks(hit_delay) + 1) * 30;
}

static inline int inf_jad_roll_primary_style(uint32_t* rng_state) {
    return (encounter_rand_int(rng_state, 2) == 0)
        ? ATTACK_STYLE_RANGED
        : ATTACK_STYLE_MAGIC;
}

static inline int inf_npc_attack_anim_id(const InfNPC* npc, const NpcModelMapping* nm) {
    if (npc->type == INF_NPC_JAD) {
        switch (npc->attack_style_this_tick) {
            case ATTACK_STYLE_MAGIC:
                return INF_ANIM_JALTOK_JAD_MAGIC_ATTACK;
            case ATTACK_STYLE_RANGED:
                return INF_ANIM_JALTOK_JAD_RANGED_ATTACK;
            case ATTACK_STYLE_MELEE:
                return INF_ANIM_JALTOK_JAD_MELEE_ATTACK;
            default:
                return -1;
        }
    }

    if (!nm || nm->attack_anim == 65535)
        return -1;
    return (int)nm->attack_anim;
}

static inline int inf_choose_attack_style_for_tick(
    uint32_t* rng_state, int style_mask
) {
    int primary_style = inf_attack_style_from_mask(style_mask & ~INF_STYLE_MASK_MELEE);
    if ((style_mask & INF_STYLE_MASK_MELEE) && primary_style != ATTACK_STYLE_NONE)
        return (encounter_rand_int(rng_state, 2) == 0) ? ATTACK_STYLE_MELEE : primary_style;
    return inf_attack_style_from_mask(style_mask);
}

/* ======================================================================== */
/* dead mob store for mager resurrection                                     */
/* ======================================================================== */

static inline int inf_dead_mob_is_resurrectable(InfNPCType type) {
    switch (type) {
        case INF_NPC_BAT:
        case INF_NPC_BLOB:
        case INF_NPC_MELEER:
        case INF_NPC_RANGER:
        case INF_NPC_MAGER:
            return 1;
        default:
            return 0;
    }
}

static void inf_store_dead_mob(InfernoState* s, InfNPC* npc) {
    if (s->dead_mob_count >= INF_MAX_DEAD_MOBS) return;
    /* only store the exact types that register with InfernoMobDeathStore in
       the reference: bat, blob parent, meleer, ranger, and mager. */
    if (!inf_dead_mob_is_resurrectable(npc->type)) return;
    if (npc->resurrection_count != 0) return;

    InfDeadMob* dm = &s->dead_mobs[s->dead_mob_count++];
    dm->type = npc->type;
    dm->x = npc->x;
    dm->y = npc->y;
    dm->hp = npc->max_hp / 2;  /* resurrect at 50% HP */
    dm->max_hp = npc->max_hp;
}

/* ======================================================================== */
/* forward declarations                                                      */
/* ======================================================================== */

static float inf_compute_reward(InfernoState* s);
static void inf_spawn_wave(InfernoState* s);
static void inf_tick_npcs(InfernoState* s);
static void inf_tick_player(InfernoState* s, const int* actions);
static void inf_apply_npc_death(InfernoState* s, int npc_idx);
static int inf_mager_resurrect(InfernoState* s, int idx);
static void inf_queue_zuk_healer_sparks(InfernoState* s, const InfNPC* npc);
static void inf_resolve_pending_sparks(InfernoState* s);
static void inf_rebuild_player_collision_flags(InfernoState* s);

/* ======================================================================== */
/* lifecycle                                                                 */
/* ======================================================================== */

static EncounterState* inf_create(void) {
    InfernoState* s = (InfernoState*)calloc(1, sizeof(InfernoState));
    s->rng_state = 12345;
    return (EncounterState*)s;
}

static void inf_destroy(EncounterState* state) {
    free(state);
}

static void inf_reset(EncounterState* state, uint32_t seed) {
    inf_build_npc_stats();
    InfernoState* s = (InfernoState*)state;
    Log saved_log = s->log;
    int saved_start = s->start_wave;
    uint32_t saved_rng = s->rng_state;
    const CollisionMap* saved_cmap = s->collision_map;
    int saved_wox = s->world_offset_x;
    int saved_woy = s->world_offset_y;
    float saved_damage_reward_coeff = s->damage_reward_coeff;
    float saved_shield_penalty_coeff = s->shield_penalty_coeff;
    float saved_tag_reward_coeff = s->tag_reward_coeff;
    memset(s, 0, sizeof(InfernoState));
    s->log = saved_log;
    s->start_wave = saved_start;
    s->collision_map = saved_cmap;
    s->world_offset_x = saved_wox;
    s->world_offset_y = saved_woy;
    s->rng_state = encounter_resolve_seed(saved_rng, seed);
    s->damage_reward_coeff = saved_damage_reward_coeff;
    s->shield_penalty_coeff = saved_shield_penalty_coeff;
    s->tag_reward_coeff = saved_tag_reward_coeff;

    /* human click-to-move: no destination after reset */
    s->player_dest_x = -1;
    s->player_dest_y = -1;
    s->player_last_interaction_target_slot = -1;
    s->player_last_interaction_age = 1;

    /* player */
    s->player.entity_type = ENTITY_PLAYER;
    s->player.base_hitpoints = 99;
    s->player.current_hitpoints = 99;
    s->player.base_prayer = 99;
    s->player.current_prayer = 99;
    s->player.base_attack = 99;
    s->player.base_strength = 99;
    s->player.base_defence = 99;
    s->player.base_ranged = 99;
    s->player.base_magic = 99;
    s->player.current_ranged = 99;
    s->player.current_magic = 99;
    s->player.current_attack = 99;
    s->player.current_strength = 99;
    s->player.current_defence = 99;
    osrs_item_effect_state_init(&s->player.item_effect_state);
    /* start in mage gear */
    s->weapon_set = INF_GEAR_MAGE;
    s->armor_tank = 0;
    encounter_apply_loadout(&s->player, INF_MAGE_LOADOUT, GEAR_MAGE);
    {
        encounter_populate_inventory(&s->player, INF_LOADOUTS, INF_NUM_WEAPON_SETS, NULL);

        /* Ammo slot items (dragon darts, dragon arrows) should NOT appear as
           swappable inventory items — in real OSRS darts live inside the
           blowpipe and arrows inside dizana's quiver. Clear them here; the
           equipment panel still shows the correct ammo for the active weapon. */
        for (int i = 0; i < MAX_ITEMS_PER_SLOT; i++) {
            s->player.inventory[GEAR_SLOT_AMMO][i] = ITEM_NONE;
        }
        s->player.num_items_in_slot[GEAR_SLOT_AMMO] = 0;
    }
    s->player.brew_doses = 24;     /* 6 pots x 4 doses */
    s->player.restore_doses = 40;  /* 10 pots x 4 doses */
    s->player.bastion_doses = 8;   /* 2 pots x 4 doses */
    s->player.stamina_doses = 4;   /* 1 pot x 4 doses */
    s->stamina_active_ticks = 0;
    s->player.prayer = PRAYER_NONE;
    osrs_interaction_init(&s->interaction);
    s->player.spec_armed = 0;
    s->player.special_energy = 100;
    s->player.run_energy = 10000;  /* full run energy (OSRS stores as 0-10000) */
    s->last_hit_by_type = -1;

    /* compute loadout stats from item database (replaces old hardcoded INF_WEAPON_STATS).
       mage is kodai + barrage — pure autocast, no invisible bonus.
       ranged loadouts run in rapid stance: -1 to attack_speed (BP 3→2, tbow 6→5). */
    /* offensive prayer is now agent-controlled runtime state (Player.offensive_prayer),
       not baked into the loadout. pass NONE at reset; inf_player_pretick() calls
       encounter_update_loadout_level() whenever the agent toggles a prayer. */
    encounter_compute_loadout_stats(INF_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30, &s->loadout_stats[INF_GEAR_MAGE]);
    encounter_compute_loadout_stats(INF_RANGE_TBOW_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0, &s->loadout_stats[INF_GEAR_TBOW]);
    encounter_compute_loadout_stats(INF_RANGE_BP_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0, &s->loadout_stats[INF_GEAR_BP]);

    /* spawn position depends on wave */
    int is_zuk_wave = (saved_start >= INF_NUM_WAVES - 1);
    s->player.x = is_zuk_wave ? INF_ZUK_PLAYER_START_X : INF_PLAYER_START_X;
    s->player.y = is_zuk_wave ? INF_ZUK_PLAYER_START_Y : INF_PLAYER_START_Y;
    inf_rebuild_player_collision_flags(s);

    /* pillars: all destroyed at end of wave 66 (index 65), so waves 66+ have none */
    for (int i = 0; i < INF_NUM_PILLARS; i++) {
        s->pillars[i].x = INF_PILLAR_POS[i][0];
        s->pillars[i].y = INF_PILLAR_POS[i][1];
        if (saved_start >= 66) {
            s->pillars[i].hp = 0;
            s->pillars[i].active = 0;
        } else {
            s->pillars[i].hp = INF_PILLAR_HP;
            s->pillars[i].active = 1;
        }
    }
    inf_rebuild_los(s);

    /* dead mob store */
    s->dead_mob_count = 0;

    /* start at configured wave (for curriculum).
       delay first wave spawn by 10 ticks — in-game there's a delay
       after entering the inferno before wave 1 begins. */
    s->wave = s->start_wave;
    s->wave_spawn_target = s->start_wave;
    s->wave_spawn_delay = 10;
}

/* ======================================================================== */
/* spawn: place NPCs for current wave                                        */
/* ======================================================================== */

/* find a free NPC slot, return index or -1 */
static int inf_find_free_npc(InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!s->npcs[i].active) return i;
    }
    return -1;
}

static int inf_grid_index(int x, int y, int* gx, int* gy) {
    *gx = x - INF_ARENA_MIN_X;
    *gy = y - INF_ARENA_MIN_Y;
    return *gx >= 0 && *gx < INF_ARENA_WIDTH && *gy >= 0 && *gy < INF_ARENA_HEIGHT;
}

static int inf_npc_sets_collision_flag(InfNPCType type) {
    return type != INF_NPC_NIBBLER;
}

static int inf_npc_effective_size(const InfNPC* npc) {
    return npc->size > 0 ? npc->size : INF_NPC_STATS[npc->type].size;
}

static void inf_clear_npc_collision_footprint(InfernoState* s, int x, int y, int size) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy))
                s->npc_collision_flags[gx][gy] = 0;
        }
    }
}

static void inf_stamp_npc_collision_footprint(InfernoState* s, int x, int y, int size) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy))
                s->npc_collision_flags[gx][gy] = 1;
        }
    }
}

static void inf_clear_player_collision_flags(InfernoState* s) {
    memset(s->player_collision_flags, 0, sizeof(s->player_collision_flags));
}

static void inf_stamp_player_collision_flags(InfernoState* s) {
    int gx, gy;
    if (inf_grid_index(s->player.x, s->player.y, &gx, &gy))
        s->player_collision_flags[gx][gy] = 1;
}

static void inf_rebuild_player_collision_flags(InfernoState* s) {
    inf_clear_player_collision_flags(s);
    inf_stamp_player_collision_flags(s);
}

static void inf_rebuild_entity_collision_flags(InfernoState* s) {
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    inf_rebuild_player_collision_flags(s);
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        InfNPC* npc = &s->npcs[i];
        if (!npc->active) continue;
        if (!inf_npc_sets_collision_flag(npc->type)) continue;
        inf_stamp_npc_collision_footprint(s, npc->x, npc->y, inf_npc_effective_size(npc));
    }
}

static void inf_update_npc_collision_flags(
    InfernoState* s, int idx, int ox, int oy, int nx, int ny, int sz
) {
    if (idx >= 0 && idx < INF_MAX_NPCS &&
        !inf_npc_sets_collision_flag(s->npcs[idx].type))
        return;
    inf_clear_npc_collision_footprint(s, ox, oy, sz);
    inf_stamp_npc_collision_footprint(s, nx, ny, sz);
}

static void inf_deactivate_npc(InfernoState* s, int idx) {
    if (idx < 0 || idx >= INF_MAX_NPCS) return;
    InfNPC* npc = &s->npcs[idx];
    if (npc->active && inf_npc_sets_collision_flag(npc->type))
        inf_clear_npc_collision_footprint(s, npc->x, npc->y, inf_npc_effective_size(npc));
    npc->active = 0;
}

/* initialize an NPC at a given slot */
static void inf_init_npc(InfernoState* s, int idx, InfNPCType type, int x, int y) {
    InfNPC* npc = &s->npcs[idx];
    const InfNPCStats* stats = &INF_NPC_STATS[type];
    memset(npc, 0, sizeof(InfNPC));

    npc->type = type;
    npc->hp = stats->hp;
    npc->max_hp = stats->hp;
    npc->size = stats->size;
    npc->attack_timer = stats->attack_speed;
    npc->attack_style = stats->default_style;
    npc->jad_attack_style = ATTACK_STYLE_NONE;
    npc->attack_style_this_tick = ATTACK_STYLE_NONE;
    npc->active = 1;
    npc->x = x;
    npc->y = y;
    npc->target_x = x;
    npc->target_y = y;
    npc->attack_visual_target = -1;
    npc->heal_target = -1;
    npc->jad_owner_idx = -1;
    npc->aggro_target = -1;
    npc->blob_scanned_prayer = -1;
    npc->had_los_last_tick = 0;
    npc->stun_timer = stats->stun_on_spawn;

    if (inf_npc_sets_collision_flag(type))
        inf_stamp_npc_collision_footprint(s, x, y, stats->size);
}

static void inf_spawn_wave(InfernoState* s) {
    if (s->wave >= INF_NUM_WAVES) return;

    const InfWaveDef* w = &INF_WAVES[s->wave];

    /* clear all NPCs and pending hits */
    for (int i = 0; i < INF_MAX_NPCS; i++) s->npcs[i].active = 0;
    s->player_pending_hit_count = 0;
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    inf_rebuild_player_collision_flags(s);

    /* clear dead mob store each wave */
    s->dead_mob_count = 0;

    /* shuffle spawn positions */
    inf_shuffle_spawns(s);

    /* pick random active pillar for nibblers this wave */
    {
        int active_pillars[INF_NUM_PILLARS];
        int num_active = 0;
        for (int p = 0; p < INF_NUM_PILLARS; p++) {
            if (s->pillars[p].active) active_pillars[num_active++] = p;
        }
        s->nibbler_target_pillar = (num_active > 0)
            ? active_pillars[encounter_rand_int(&s->rng_state, num_active)] : -1;
    }

    /* waves 67-69 have no pillars — ref InfernoRegion.ts:359
       (`this.wave < 67 || this.wave >= 70` means pillars exist outside this range).
       clear on spawn so mid-episode transitions also collapse any survivors. */
    if (s->wave >= 66) {
        int pillars_changed = 0;
        for (int p = 0; p < INF_NUM_PILLARS; p++) {
            if (s->pillars[p].active) {
                s->pillars[p].active = 0;
                s->pillars[p].hp = 0;
                pillars_changed = 1;
            }
        }
        if (pillars_changed) inf_rebuild_los(s);
    }

    /* wave 67 (index 66): single jad at fixed position. ref InfernoRegion.ts:441-451.
       our coord system is Y-flipped vs reference (pillars confirm ours_y = 57 - ref_y).
       ref: player (18, 25), jad (23, 27), stun=1, attackSpeed=8, healers=5. */
    if (s->wave == 66) {
        s->player.x = 18;
        s->player.y = 32;  /* 57 - 25 */
        inf_rebuild_player_collision_flags(s);
        int slot = inf_find_free_npc(s);
        if (slot >= 0) {
            inf_init_npc(s, slot, INF_NPC_JAD, 23, 30);  /* 57 - 27 = 30 */
            s->npcs[slot].stun_timer = 1;
            s->npcs[slot].attack_timer = 8;
        }
        return;
    }

    /* wave 68 (index 67): three jads at fixed positions with staggered first attacks.
       ref InfernoRegion.ts:452-479. stunTimers [1,4,7] shuffled across the 3 jads so
       their first attacks stagger instead of landing on the same tick.
       ref: player (25, 27), jads (18, 24), (28, 24), (23, 35). attackSpeed=9, healers=3. */
    if (s->wave == 67) {
        s->player.x = 25;
        s->player.y = 30;  /* 57 - 27 */
        inf_rebuild_player_collision_flags(s);
        /* shuffle [1, 4, 7] via Fisher-Yates */
        int stuns[3] = { 1, 4, 7 };
        for (int i = 2; i > 0; i--) {
            int j = encounter_rand_int(&s->rng_state, i + 1);
            int tmp = stuns[i]; stuns[i] = stuns[j]; stuns[j] = tmp;
        }
        static const int JAD_POS[3][2] = {
            {18, 33},  /* 57 - 24 */
            {28, 33},  /* 57 - 24 */
            {23, 22},  /* 57 - 35 */
        };
        for (int i = 0; i < 3; i++) {
            int slot = inf_find_free_npc(s);
            if (slot < 0) break;
            inf_init_npc(s, slot, INF_NPC_JAD, JAD_POS[i][0], JAD_POS[i][1]);
            s->npcs[slot].stun_timer = stuns[i];
            s->npcs[slot].attack_timer = 9;
        }
        return;
    }

    /* zuk wave (wave 69, index 68) is special */
    if (s->wave == 68) {
        /* spawn Zuk — fixed position, cannot move */
        int zuk_idx = inf_find_free_npc(s);
        if (zuk_idx >= 0) {
            inf_init_npc(s, zuk_idx, INF_NPC_ZUK, 22, 50);
            s->min_zuk_hp_seen = (float)s->npcs[zuk_idx].hp;
            /* InfernoTrainer: stunned=8, attackDelay=14. stun counts down first,
               then attackDelay ticks down to 0 before first attack fires. */
            s->npcs[zuk_idx].stun_timer = 8;
            s->npcs[zuk_idx].attack_timer = 14;
        }

        /* spawn shield */
        int shield_idx = inf_find_free_npc(s);
        if (shield_idx >= 0) {
            inf_init_npc(s, shield_idx, INF_NPC_ZUK_SHIELD, 23, 44);
            s->zuk.shield_idx = shield_idx;
            s->zuk.shield_dir = (encounter_rand_int(&s->rng_state, 2) == 0) ? 1 : -1;
            s->zuk.shield_freeze = 1;  /* 1-tick freeze on spawn */
        }

        /* zuk state */
        s->zuk.initial_delay = 14;
        s->zuk.set_timer = 72;
        s->zuk.set_interval = 350;
        s->zuk.enraged = 0;
        s->zuk.healer_spawned = 0;
        s->zuk.jad_spawned = 0;
        s->zuk.timer_paused = 0;
        s->zuk.has_paused = 0;

        /* player starts at zuk position */
        s->player.x = INF_ZUK_PLAYER_START_X;
        s->player.y = INF_ZUK_PLAYER_START_Y;
        inf_rebuild_player_collision_flags(s);
        return;
    }

    /* regular waves: spawn NPCs at shuffled positions */
    int spawn_idx = 0;
    for (int i = 0; i < w->count && i < INF_MAX_NPCS; i++) {
        InfNPCType type = (InfNPCType)w->types[i];
        int slot = inf_find_free_npc(s);
        if (slot < 0) break;

        int sx, sy;
        if (type == INF_NPC_NIBBLER) {
            /* nibblers spawn near pillars with small random offset */
            sx = INF_NIBBLER_SPAWN_X + encounter_rand_int(&s->rng_state, 3) - 1;
            sy = INF_NIBBLER_SPAWN_Y + encounter_rand_int(&s->rng_state, 3) - 1;
        } else {
            int pi = s->spawn_order[spawn_idx % INF_NUM_SPAWN_POS];
            sx = INF_SPAWN_POS[pi][0];
            sy = INF_SPAWN_POS[pi][1];
            spawn_idx++;
        }

        inf_init_npc(s, slot, type, sx, sy);
    }
}

/* ======================================================================== */
/* NPC AI: movement                                                          */
/* ======================================================================== */

static int inf_in_arena(int x, int y) {
    return x >= INF_ARENA_MIN_X && x <= INF_ARENA_MAX_X &&
           y >= INF_ARENA_MIN_Y && y <= INF_ARENA_MAX_Y;
}

static int inf_blocked_by_pillar(InfernoState* s, int x, int y, int size) {
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        if (!s->pillars[p].active) continue;
        if (los_aabb_overlap(x, y, size,
                             s->pillars[p].x, s->pillars[p].y, INF_PILLAR_SIZE))
            return 1;
    }
    return 0;
}

/* BFS dynamic obstacle callback — pillars block pathfinding.
   receives absolute world coords, converts to local for pillar check. */
static int inf_pathfind_blocked(void* ctx, int abs_x, int abs_y) {
    InfernoState* s = (InfernoState*)ctx;
    int lx = abs_x - s->world_offset_x;
    int ly = abs_y - s->world_offset_y;
    return inf_blocked_by_pillar(s, lx, ly, 1);
}

static int inf_npc_collision_flag_blocked(InfernoState* s, int x, int y, int size) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy) &&
                s->npc_collision_flags[gx][gy])
                return 1;
        }
    }
    return 0;
}

static int inf_player_collision_flag_blocked(InfernoState* s, int x, int y, int size) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy) &&
                s->player_collision_flags[gx][gy])
                return 1;
        }
    }
    return 0;
}

/* NPC movement blocked callback for encounter_npc_step_toward. */
typedef struct { InfernoState* s; int self_idx; } InfMoveCtx;

static int inf_npc_blocked(void* ctx, int x, int y, int size) {
    InfMoveCtx* mc = (InfMoveCtx*)ctx;
    InfernoState* s = mc->s;
    (void)mc->self_idx;
    if (!inf_in_arena(x, y)) return 1;
    if (inf_blocked_by_pillar(s, x, y, size)) return 1;
    if (s->collision_map &&
        !collision_tile_walkable(s->collision_map, 0,
            x + s->world_offset_x, y + s->world_offset_y))
        return 1;
    if (inf_player_collision_flag_blocked(s, x, y, size)) return 1;
    return inf_npc_collision_flag_blocked(s, x, y, size);
}

static int inf_npc_overlap_hold(void* ctx) {
    const InfMoveCtx* mc = (const InfMoveCtx*)ctx;
    const InfernoState* s = mc->s;
    return s->player_last_interaction_age == 0 &&
           s->player_last_interaction_target_slot == mc->self_idx;
}

/* forward declaration — defined after potions/food section */
static int inf_tile_walkable(void* ctx, int x, int y);

static int inf_npc_terrain_blocked(InfernoState* s, int x, int y, int size) {
    if (!inf_in_arena(x, y)) return 1;
    if (inf_blocked_by_pillar(s, x, y, size)) return 1;
    if (!s->collision_map) return 0;
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            if (!collision_tile_walkable(
                    s->collision_map, 0,
                    x + dx + s->world_offset_x,
                    y + dy + s->world_offset_y)) {
                return 1;
            }
        }
    }
    return 0;
}

static void inf_npc_move(InfernoState* s, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (!npc->active) return;
    if (npc->stun_timer > 0) return;
    if (npc->dig_freeze_timer > 0) return;
    if (npc->frozen_ticks > 0) return;

    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (!stats->can_move) return;
    int uses_collision_flag = inf_npc_sets_collision_flag(npc->type);
    if (uses_collision_flag)
        inf_clear_npc_collision_footprint(s, npc->x, npc->y, npc->size);

    /* OSRS: NPC shuffles off player tile when overlapping (Mob.ts:109-153).
       if the NPC steps out, skip further movement this tick. */
    if (npc->type != INF_NPC_NIBBLER) {
        InfMoveCtx mc = { s, idx };
        int stepped = encounter_npc_step_out_from_under(
            &npc->x, &npc->y, npc->size,
            s->player.x, s->player.y,
            inf_npc_blocked, &mc, inf_npc_overlap_hold, &s->rng_state);
        if (stepped == ENCOUNTER_NPC_UNDER_PLAYER_MOVED) {
            npc->moved_this_tick = 1;
            if (uses_collision_flag)
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
            return;
        }
        if (stepped == ENCOUNTER_NPC_UNDER_PLAYER_HELD) {
            if (uses_collision_flag)
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
            return;
        }
    }

    /* target selection: pillar (nibbler), aggroed NPC (shield/jad/zuk), or player */
    int tx, ty;
    int target_size = 1;
    if (npc->type == INF_NPC_NIBBLER) {
        int p = s->nibbler_target_pillar;
        if (p >= 0 && p < INF_NUM_PILLARS && s->pillars[p].active) {
            tx = s->pillars[p].x;
            ty = s->pillars[p].y;
        } else {
            int found = 0;
            for (int pp = 0; pp < INF_NUM_PILLARS; pp++) {
                if (s->pillars[pp].active) {
                    tx = s->pillars[pp].x;
                    ty = s->pillars[pp].y;
                    found = 1;
                    break;
                }
            }
            if (!found) { tx = s->player.x; ty = s->player.y; }
        }
        target_size = INF_PILLAR_SIZE;
    } else if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
               s->npcs[npc->aggro_target].active) {
        /* targeting another NPC (set→shield, jad→shield) */
        tx = s->npcs[npc->aggro_target].x;
        ty = s->npcs[npc->aggro_target].y;
        target_size = s->npcs[npc->aggro_target].size;
    } else {
        /* default: target player. clear stale aggro if target died. */
        if (npc->aggro_target >= 0) npc->aggro_target = -1;
        tx = s->player.x;
        ty = s->player.y;
    }
    npc->target_x = tx;
    npc->target_y = ty;

    /* NPCs stop moving once they can attack their current target.
       reference: InfernoTrainer Unit.ts:383 canMove = !hasLOS (where
       hasLOS is relative to the NPC's current aggro target). */
    if (npc->type != INF_NPC_NIBBLER) {
        if (entity_has_line_of_sight(
                s->los_blockers, s->los_blocker_count,
                npc->x, npc->y, npc->size,
                tx, ty, target_size,
                stats->attack_range)) {
            if (uses_collision_flag)
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
            return;
        }
    }

    int ox = npc->x, oy = npc->y;
    InfMoveCtx mc = { s, idx };
    encounter_npc_step_toward(&npc->x, &npc->y, tx, ty, npc->size,
                              target_size, stats->attack_range == 1,
                              inf_npc_blocked, &mc);
    if (npc->x != ox || npc->y != oy) {
        npc->moved_this_tick = 1;
    }
    if (uses_collision_flag)
        inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
}

/* ======================================================================== */
/* NPC AI: meleer dig mechanic                                               */
/* ======================================================================== */

/* meleer digs when no LOS for 38+ ticks, 10% per tick, forced at 50 */
static void inf_meleer_dig_check(InfernoState* s, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (npc->type != INF_NPC_MELEER || !npc->active) return;
    if (npc->dig_freeze_timer > 0) {
        npc->dig_freeze_timer--;
        if (npc->dig_freeze_timer == 0 && npc->dig_attack_delay == 0) {
            /* emerge: use the reference ordered landing candidates around the
               player, then fall back to the default NW corner if all preferred
               tiles are blocked by arena terrain/entities. */
            int ox = npc->x, oy = npc->y;
            if (inf_npc_sets_collision_flag(npc->type))
                inf_clear_npc_collision_footprint(s, ox, oy, npc->size);
            int candidates[5][2] = {
                { s->player.x - npc->size + 1, s->player.y - npc->size + 1 },
                { s->player.x,                 s->player.y                 },
                { s->player.x - npc->size + 1, s->player.y                 },
                { s->player.x,                 s->player.y - npc->size + 1 },
                { s->player.x - 1,             s->player.y - 1             },
            };
            int landing_x = candidates[4][0];
            int landing_y = candidates[4][1];
            for (int i = 0; i < 4; i++) {
                if (inf_npc_terrain_blocked(s, candidates[i][0], candidates[i][1], npc->size))
                    continue;
                landing_x = candidates[i][0];
                landing_y = candidates[i][1];
                break;
            }
            npc->x = landing_x;
            npc->y = landing_y;
            if (inf_npc_sets_collision_flag(npc->type))
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
            npc->stun_timer = 2;  /* 2-tick freeze after emerging */
            npc->dig_attack_delay = 6;  /* 6-tick delay before attacking */
            npc->no_los_ticks = 0;
        }
        return;
    }
    if (npc->dig_attack_delay > 0) {
        npc->dig_attack_delay--;
        return;
    }

    /* track LOS absence */
    if (!inf_npc_has_los(s, idx)) {
        npc->no_los_ticks++;
    } else {
        npc->no_los_ticks = 0;
        return;
    }

    /* check dig trigger */
    if (npc->no_los_ticks >= 50) {
        /* forced dig */
        npc->dig_freeze_timer = 6;
    } else if (npc->no_los_ticks >= 38) {
        /* 10% chance per tick */
        if (encounter_rand_int(&s->rng_state, 10) == 0) {
            npc->dig_freeze_timer = 6;
        }
    }
}

static AttackStyle inf_player_equipped_attack_style(const InfernoState* s) {
    uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    AttackStyle style = (AttackStyle)get_item_attack_style(weapon);
    if (style == ATTACK_STYLE_MAGIC ||
        style == ATTACK_STYLE_RANGED ||
        style == ATTACK_STYLE_MELEE) {
        return style;
    }
    return ATTACK_STYLE_RANGED;
}

static void inf_refresh_human_loadout_stats(InfernoState* s) {
    AttackStyle style = inf_player_equipped_attack_style(s);
    int spell_base_damage = (style == ATTACK_STYLE_MAGIC) ? 30 : 0;
    encounter_compute_player_equipped_stats(
        &s->player, style, s->player.fight_style, spell_base_damage,
        &s->human_loadout_stats);
}

static const EncounterLoadoutStats* inf_current_loadout_stats(InfernoState* s) {
    if (s->human_command_mode) {
        inf_refresh_human_loadout_stats(s);
        return &s->human_loadout_stats;
    }
    return &s->loadout_stats[s->weapon_set];
}

static int inf_player_weapon_is(const InfernoState* s, uint8_t item) {
    return s->player.equipped[GEAR_SLOT_WEAPON] == item;
}

/* ======================================================================== */
/* NPC AI: attacks                                                           */
/* ======================================================================== */

static void inf_npc_attack(InfernoState* s, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (!npc->active) return;
    if (npc->stun_timer > 0) { npc->stun_timer--; return; }
    if (npc->dig_freeze_timer > 0) return;
    if (npc->dig_attack_delay > 0) return;
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];

    int has_los_now = 0;
    if (stats->attack_range > 1) {
        has_los_now = inf_npc_has_los(s, idx);
    }

    if (npc->type == INF_NPC_BLOB &&
        npc->blob_scanned_prayer < 0 &&
        has_los_now &&
        !npc->had_los_last_tick) {
        npc->blob_scanned_prayer = (int)s->player.prayer;
        /* commit to opposite of scanned prayer; random if scanned prayer is neither */
        if (s->player.prayer == PRAYER_PROTECT_MAGIC) npc->attack_style = ATTACK_STYLE_RANGED;
        else if (s->player.prayer == PRAYER_PROTECT_RANGED) npc->attack_style = ATTACK_STYLE_MAGIC;
        else npc->attack_style = (encounter_rand_int(&s->rng_state, 2) == 0) ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED;
        npc->had_los_last_tick = has_los_now;
        npc->attacked_this_tick = 1;
        npc->attack_timer = stats->attack_speed;
        return;
    }

    npc->had_los_last_tick = has_los_now;

    /* decrement first, then check — matches SDK (Unit.ts:237 attackDelay-- then
       Mob.ts:326 attackDelay <= 0). without this, NPCs attack 1 tick slower. */
    if (npc->attack_timer > 0) npc->attack_timer--;
    if (npc->attack_timer > 0) return;

    /* shield doesn't attack */
    if (npc->type == INF_NPC_ZUK_SHIELD) return;

    /* NPC targeting another NPC (set/jad → shield): always hit, random damage.
       both healer types excluded — their heal handlers below RESTORE HP
       to their target instead of damaging it. */
    if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS
        && npc->type != INF_NPC_HEALER_ZUK
        && npc->type != INF_NPC_HEALER_JAD) {
        InfNPC* target = &s->npcs[npc->aggro_target];
        if (target->active) {
            int max_hit = osrs_npc_magic_max_hit(stats->magic_base_dmg, stats->magic_dmg_pct);
            if (stats->can_melee) {
                max_hit = osrs_npc_melee_max_hit(stats->str_level, stats->melee_str_bonus);
            }
            int dmg = encounter_rand_int(&s->rng_state, max_hit + 1);
            int target_hp_before = target->hp;
            encounter_damage_npc(&target->hp, &target->hit_landed_this_tick,
                                 &target->hit_damage, dmg);
            if (target->type == INF_NPC_ZUK_SHIELD) {
                s->shield_damage_this_tick += (float)(target_hp_before - target->hp);
            }
            /* shield death: redirect all NPCs targeting it to the player */
            if (target->hp <= 0 && target->type == INF_NPC_ZUK_SHIELD) {
                inf_deactivate_npc(s, npc->aggro_target);
                s->zuk.shield_idx = -1;
                for (int i = 0; i < INF_MAX_NPCS; i++) {
                    if (s->npcs[i].aggro_target == npc->aggro_target)
                        s->npcs[i].aggro_target = -1;
                }
            }
            npc->attacked_this_tick = 1;
            npc->attack_style_this_tick = stats->can_melee
                ? ATTACK_STYLE_MELEE : stats->default_style;
            npc->attack_visual_target = npc->aggro_target;
            npc->attack_timer = stats->attack_speed;
            return;
        } else {
            npc->aggro_target = -1;  /* target died, fall through to player attack */
        }
    }

    /* nibbler attacks pillars, not player */
    if (npc->type == INF_NPC_NIBBLER) {
        for (int p = 0; p < INF_NUM_PILLARS; p++) {
            if (!s->pillars[p].active) continue;
            int ddx = npc->x - s->pillars[p].x;
            int ddy = npc->y - s->pillars[p].y;
            if (ddx >= -1 && ddx <= INF_PILLAR_SIZE && ddy >= -1 && ddy <= INF_PILLAR_SIZE) {
                /* nibblers deal 0-4 damage per hit (ref: InfernoTrainer JalNib.ts).
                   bypasses combat formula — custom weapon directly rolls rand(0..4). */
                int dmg = encounter_rand_int(&s->rng_state, 5);
                s->pillars[p].hp -= dmg;
                if (s->pillars[p].hp <= 0) {
                    s->pillars[p].active = 0;
                    s->pillar_lost_this_tick = p;
                    inf_rebuild_los(s);
                    inf_invalidate_los_cache(s);
                    /* pillar death AOE: deals damage to all mobs + player within 1 tile */
                    for (int n = 0; n < INF_MAX_NPCS; n++) {
                        if (!s->npcs[n].active) continue;
                        int ndx = s->npcs[n].x - s->pillars[p].x;
                        int ndy = s->npcs[n].y - s->pillars[p].y;
                        if (ndx >= -1 && ndx <= INF_PILLAR_SIZE && ndy >= -1 && ndy <= INF_PILLAR_SIZE) {
                            encounter_damage_npc(&s->npcs[n].hp, &s->npcs[n].hit_landed_this_tick, &s->npcs[n].hit_damage, 12);
                            inf_apply_npc_death(s, n);
                        }
                    }
                    /* also damages the player if standing next to the pillar */
                    {
                        int pdx = s->player.x - s->pillars[p].x;
                        int pdy = s->player.y - s->pillars[p].y;
                        if (pdx >= -1 && pdx <= INF_PILLAR_SIZE && pdy >= -1 && pdy <= INF_PILLAR_SIZE) {
                            /* pillar collapse: 49 damage (observed in-game, server-side formula unknown) */
                            encounter_damage_player(&s->player, 49, &s->damage_received_this_tick);
                        }
                    }
                }
                npc->attacked_this_tick = 1;
                npc->attack_timer = stats->attack_speed;
                return;
            }
        }
        return;
    }

    /* zuk healer: heal Zuk while it is untapped, switch to player-facing
       sparks after tag. */
    if (npc->type == INF_NPC_HEALER_ZUK) {
        if (npc->aggro_target >= 0) {
            /* heal Zuk with a 3-tick projectile */
            int ti = npc->aggro_target;
            if (ti >= 0 && ti < INF_MAX_NPCS && s->npcs[ti].active && s->npcs[ti].type == INF_NPC_ZUK) {
                npc->heal_target = ti;
                npc->heal_timer = 3;
                npc->attack_visual_target = ti;
            }
        } else {
            /* tagged healer: stop healing and fire the 3-spark ground pattern. */
            npc->heal_target = -1;
            npc->heal_timer = 0;
            npc->attack_visual_target = -1;
            inf_queue_zuk_healer_sparks(s, npc);
        }
        npc->attacked_this_tick = 1;
        npc->attack_style_this_tick = ATTACK_STYLE_MAGIC;
        npc->attack_timer = stats->attack_speed;
        return;
    }

    /* jad healer: heals its Jad while aggro_target points at the boss, then
       switches to crush melee after a player tag. */
    if (npc->type == INF_NPC_HEALER_JAD) {
        int jad_idx = npc->jad_owner_idx;
        if (npc->aggro_target >= 0) {
            if (jad_idx >= 0 && s->npcs[jad_idx].active &&
                s->npcs[jad_idx].type == INF_NPC_JAD) {
                /* heal Jad with the reference 3-tick delay. */
                npc->heal_target = jad_idx;
                npc->heal_timer = 3;
                npc->attack_visual_target = jad_idx;
            }
            npc->attacked_this_tick = 1;
            npc->attack_timer = stats->attack_speed;
            return;
        }
        /* if player has tagged this healer, it only attacks on cardinal melee
           contact. diagonal corners do not count. */
        if (entity_has_line_of_sight(
                s->los_blockers, s->los_blocker_count,
                npc->x, npc->y, 1,
                s->player.x, s->player.y, 1,
                1)) {
            int max_hit = osrs_npc_melee_max_hit(stats->str_level, stats->melee_str_bonus);
            int dmg = encounter_rand_int(&s->rng_state, max_hit + 1);
            /* accuracy roll */
            int att_roll = osrs_npc_attack_roll(stats->att_level, stats->melee_att_bonus);
            const EncounterLoadoutStats* ls = inf_current_loadout_stats(s);
            int def_bonus = ls->def_crush;
            int def_roll = osrs_player_def_roll_vs_npc(s->player.current_defence, s->player.current_magic, def_bonus, ATTACK_STYLE_MELEE);
            if (encounter_rand_float(&s->rng_state) >= osrs_hit_chance(att_roll, def_roll)) dmg = 0;
            int prayer_matches = (s->player.prayer == PRAYER_PROTECT_MELEE);
              if (prayer_matches) { dmg = 0; s->prayer_correct_this_tick = 1; }
              else if (dmg > 0) { s->off_prayer_hits_this_tick++; }
            encounter_damage_player(&s->player, dmg, &s->damage_received_this_tick);
            npc->attack_style_this_tick = ATTACK_STYLE_MELEE;
            npc->attacked_this_tick = 1;
            npc->attack_visual_target = -1;
            npc->attack_timer = stats->attack_speed;
        }
        return;
    }

    /* ranged/magic NPCs need LOS, except blobs once they have already stored
       a prayer scan. ref: InfernoTrainer JalAk.ts attackIfPossible(). */
    if (stats->attack_range > 1 &&
        !(npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) &&
        !has_los_now) return;

    /* compute distance to player */
    int dist = encounter_dist_to_npc(s->player.x, s->player.y,
                                      npc->x, npc->y, npc->size);
    if (dist == 0 || dist > stats->attack_range) return;

    /* blob prayer reading: 2-phase attack with attack_speed = 3.
       scan tick: read player prayer, set timer, return (shows scan animation).
       fire tick: determine style from scanned prayer, fall through to common attack.
       total cycle = 6 ticks (3 scan + 3 cooldown).
       ref: InfernoTrainer JalAk.ts attackIfPossible() */
    if (npc->type == INF_NPC_BLOB) {
        if (npc->blob_scanned_prayer < 0) {
            /* no pending scan → start scan phase: read prayer, commit to opposite style */
            npc->blob_scanned_prayer = (int)s->player.prayer;
            if (s->player.prayer == PRAYER_PROTECT_MAGIC) npc->attack_style = ATTACK_STYLE_RANGED;
            else if (s->player.prayer == PRAYER_PROTECT_RANGED) npc->attack_style = ATTACK_STYLE_MAGIC;
            else npc->attack_style = (encounter_rand_int(&s->rng_state, 2) == 0) ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED;
            npc->attacked_this_tick = 1;  /* triggers scan animation */
            npc->attack_timer = stats->attack_speed;  /* 3 */
            return;
        }
        /* has pending scan → determine style and fall through to fire */
        npc->blob_scanned_prayer = -1;
        /* fall through to common attack code */
    }

    /* determine actual attack style */
    int actual_style = npc->attack_style;

    /* jad: use the committed preview style if present. if the preview was not
       seeded, fall back to the 50/50 primary-style roll. */
    if (npc->type == INF_NPC_JAD) {
        actual_style = npc->jad_attack_style;
        if (actual_style == ATTACK_STYLE_NONE)
            actual_style = inf_jad_roll_primary_style(&s->rng_state);
    }

    /* zuk: typeless attack (not blockable by prayer).
       InfernoTrainer TzKalZuk.ts: Zuk always fires at the shield if the player is
       behind it (shield absorbs, 0 damage). otherwise fires at the player.
       hit delay = 4 ticks (ZukWeapon: setDelay=4). */
    if (npc->type == INF_NPC_ZUK) {
        int si = s->zuk.shield_idx;
        int player_behind_shield = 0;
        if (si >= 0 && s->npcs[si].active) {
            InfNPC* shield = &s->npcs[si];
            player_behind_shield = (s->player.x >= shield->x &&
                                    s->player.x < shield->x + shield->size &&
                                    s->player.y >= 41);
        }

        if (player_behind_shield) {
            /* shield absorbs — Zuk fires at shield, 0 damage */
            npc->attacked_this_tick = 1;
            npc->attack_style_this_tick = ATTACK_STYLE_MAGIC;
            npc->attack_visual_target = si;
        } else {
            /* typeless hit on player — not blockable by prayer, no accuracy roll.
               rolled 0..max_hit, queued at T and lands at T+4 unchanged. */
            int max_hit = osrs_npc_magic_max_hit(stats->magic_base_dmg, stats->magic_dmg_pct);
            int dmg = encounter_rand_int(&s->rng_state, max_hit + 1);
            if (s->player_pending_hit_count < ENCOUNTER_MAX_PENDING_HITS) {
                EncounterPendingHit* ph = &s->player_pending_hits[s->player_pending_hit_count++];
                ph->active = 1;
                ph->damage = dmg;
                ph->ticks_remaining = 4;
                ph->attack_style = ATTACK_STYLE_NONE;  /* typeless — not blockable */
                ph->check_prayer = 0;
                ph->prayer_check_delay = 0;
                ph->source_npc_type = npc->type;
            }
            s->last_hit_by_type = INF_NPC_ZUK;
            npc->attacked_this_tick = 1;
            npc->attack_style_this_tick = ATTACK_STYLE_MAGIC;
            /* attack_visual_target = -1 (player), already default */
        }
        npc->attack_timer = s->zuk.enraged ? 7 : stats->attack_speed;
        return;
    }

    /* inferno has NPC-specific melee fallback rules when close enough to hit. */
    {
        int style_mask = inf_attack_style_options_mask(
            s, npc, stats, actual_style, dist);
        actual_style = inf_choose_attack_style_for_tick(
            &s->rng_state, style_mask);
    }

    if (npc->type == INF_NPC_MAGER &&
        actual_style == ATTACK_STYLE_MAGIC &&
        inf_mager_resurrect(s, idx)) {
        npc->attacked_this_tick = 1;
        npc->attack_style_this_tick = ATTACK_STYLE_MAGIC;
        npc->attack_timer = stats->attack_speed;
        return;
    }

    /* max hit from stats + bonuses via OSRS combat formulas */
    int max_hit = osrs_npc_max_hit(actual_style,
        stats->str_level, stats->range_level,
        stats->melee_str_bonus, stats->ranged_str_bonus,
        stats->magic_base_dmg, stats->magic_dmg_pct);
    if (stats->max_hit_cap > 0 && max_hit > stats->max_hit_cap)
        max_hit = stats->max_hit_cap;
    int is_delayed_jad = (npc->type == INF_NPC_JAD &&
                          actual_style != ATTACK_STYLE_MELEE);
    int dmg = is_delayed_jad ? max_hit : encounter_rand_int(&s->rng_state, max_hit + 1);

    /* accuracy roll: NPC attack roll vs player defence roll */
    if (!is_delayed_jad) {
        int att_lvl, att_bonus;
        if (actual_style == ATTACK_STYLE_MELEE) {
            att_lvl = stats->att_level; att_bonus = stats->melee_att_bonus;
        } else if (actual_style == ATTACK_STYLE_RANGED) {
            att_lvl = stats->range_level; att_bonus = stats->range_att_bonus;
        } else {
            att_lvl = stats->magic_level; att_bonus = stats->magic_att_bonus;
        }
        int att_roll = osrs_npc_attack_roll(att_lvl, att_bonus);
        const EncounterLoadoutStats* ls = inf_current_loadout_stats(s);
        int def_bonus = encounter_player_def_bonus(
            ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
            actual_style, stats->melee_style);
        int def_roll = osrs_player_def_roll_vs_npc(s->player.current_defence, s->player.current_magic, def_bonus, actual_style);
        if (encounter_rand_float(&s->rng_state) >= osrs_hit_chance(att_roll, def_roll))
            dmg = 0;  /* missed */
    }

    /* compute hit delay based on attack style */
    int hit_delay = 0;
    if (actual_style == ATTACK_STYLE_MAGIC)
        hit_delay = encounter_magic_hit_delay(dist, 0);
    else if (actual_style == ATTACK_STYLE_RANGED)
        hit_delay = encounter_ranged_hit_delay(dist, 0);
    /* melee: delay = 0 */

    /* track which styles fired this tick for multi-style analysis */
    if (actual_style == ATTACK_STYLE_MELEE) s->tick_styles_fired |= 1;
    else if (actual_style == ATTACK_STYLE_RANGED) s->tick_styles_fired |= 2;
    else if (actual_style == ATTACK_STYLE_MAGIC) s->tick_styles_fired |= 4;
    s->tick_attacks_fired++;
    s->attacks_by_type[npc->type]++;

    /* bat (JalMejRah): drain 3 run energy (300 internal) on every attack.
       ref: OSRS wiki Jal-MejRah, InfernoTrainer JalMejRah.ts */
    if (npc->type == INF_NPC_BAT) {
        s->player.run_energy -= 300;
        if (s->player.run_energy < 0) s->player.run_energy = 0;
    }

    if (hit_delay == 0) {
        /* melee: instant damage, check prayer now */
        int prayer_matches = encounter_prayer_correct_for_style(s->player.prayer, actual_style);
          if (prayer_matches) { dmg = 0; s->prayer_correct_this_tick++; s->prayer_correct_by_type[npc->type]++; }
          else if (dmg > 0) { s->off_prayer_hits_this_tick++; }
        s->dmg_from_type[npc->type] += (float)dmg;
        if (dmg > 0) s->last_hit_by_type = npc->type;
        encounter_damage_player(&s->player, dmg, &s->damage_received_this_tick);
    } else {
            /* ranged/magic: queue pending hit on player */
        if (s->player_pending_hit_count < ENCOUNTER_MAX_PENDING_HITS) {
            int is_jad = (npc->type == INF_NPC_JAD);
            if (!is_jad) {
                int prayer_matches = encounter_prayer_correct_for_style(s->player.prayer, actual_style);
          if (prayer_matches) { dmg = 0; s->prayer_correct_this_tick++; s->prayer_correct_by_type[npc->type]++; }
          else if (dmg > 0) { s->off_prayer_hits_this_tick++; }
            }
            if (!is_delayed_jad) {
                s->dmg_from_type[npc->type] += (float)dmg;
                if (dmg > 0) s->last_hit_by_type = npc->type;
            }
            /* bat stat drain: 50% chance on successful hit when not praying protect
               from missiles, drain all combat stats by 1. ref: OSRS wiki Jal-MejRah */
            if (npc->type == INF_NPC_BAT && dmg > 0 &&
                encounter_rand_int(&s->rng_state, 2) == 0) {
                if (s->player.current_attack > 0) s->player.current_attack--;
                if (s->player.current_strength > 0) s->player.current_strength--;
                if (s->player.current_defence > 0) s->player.current_defence--;
                if (s->player.current_ranged > 0) s->player.current_ranged--;
                if (s->player.current_magic > 0) s->player.current_magic--;
            }
            EncounterPendingHit* ph = &s->player_pending_hits[s->player_pending_hit_count++];
            ph->active = 1;
            ph->damage = dmg;
            ph->ticks_remaining = is_jad
                ? inf_jad_land_delay_from_fire(hit_delay) + 1
                : hit_delay;
            ph->attack_style = actual_style;
            ph->check_prayer = is_jad ? 1 : 0;
            ph->prayer_check_delay = is_jad ? INF_JAD_PROJECTILE_DELAY + 1 : 0;
            ph->source_npc_type = npc->type;
        }
    }

    npc->attacked_this_tick = 1;
    npc->attack_style_this_tick = actual_style;
    if (npc->type == INF_NPC_JAD)
        npc->jad_attack_style = ATTACK_STYLE_NONE;
    npc->attack_timer = stats->attack_speed;

    /* jad attack speed varies by wave */
    if (npc->type == INF_NPC_JAD) {
        if (s->wave == 66)      npc->attack_timer = 8;  /* wave 67 */
        else if (s->wave == 67) npc->attack_timer = 9;  /* wave 68 */
        else                    npc->attack_timer = 8;  /* zuk wave */
    }
}

/* ======================================================================== */
/* NPC AI: mager resurrection                                                */
/* ======================================================================== */

static int inf_find_mager_respawn_tile(
    InfernoState* s, int size, int* out_x, int* out_y
) {
    InfMoveCtx mc = { s, -1 };
    for (int x = 26; x < 33; x++) {
        for (int y = 24; y < 37; y++) {
            if (inf_npc_blocked(&mc, x, y, size)) continue;
            if (encounter_entity_footprints_overlap(x, y, size,
                                                    s->player.x, s->player.y, 1))
                continue;
            *out_x = x;
            *out_y = y;
            return 1;
        }
    }

    *out_x = 21;
    *out_y = 22;
    return !inf_npc_blocked(&mc, *out_x, *out_y, size);
}

/* mager resurrection is only evaluated on a real magic attack opportunity,
   not every NPC tick. */
static int inf_mager_resurrect(InfernoState* s, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (npc->type != INF_NPC_MAGER || !npc->active) return 0;
    if (s->wave >= 68) return 0;  /* no resurrection during Zuk wave */
    if (npc->resurrect_cooldown > 0) return 0;
    if (s->dead_mob_count == 0) return 0;

    /* 10% chance when the mager converts a ready magic attack into a respawn. */
    if (encounter_rand_int(&s->rng_state, 10) != 0) return 0;

    /* pick a random dead mob */
    int di = encounter_rand_int(&s->rng_state, s->dead_mob_count);
    InfDeadMob* dm = &s->dead_mobs[di];

    int slot = inf_find_free_npc(s);
    if (slot < 0) return 0;

    int rx = 21;
    int ry = 22;
    if (!inf_find_mager_respawn_tile(s, INF_NPC_STATS[dm->type].size, &rx, &ry))
        return 0;

    inf_init_npc(s, slot, dm->type, rx, ry);
    s->npcs[slot].hp = dm->hp;      /* 50% of max HP */
    s->npcs[slot].max_hp = dm->max_hp;
    s->npcs[slot].resurrection_count = 1;

    /* remove from dead store (swap with last) */
    s->dead_mobs[di] = s->dead_mobs[s->dead_mob_count - 1];
    s->dead_mob_count--;

    /* 8-tick cooldown */
    npc->resurrect_cooldown = 8;
    return 1;
}

/* ======================================================================== */
/* NPC AI: jad healer spawning                                               */
/* ======================================================================== */

#define INF_JAD_HEALER_MAX_SPAWN_CANDIDATES 165

static void inf_sample_jad_healer_spawn(InfernoState* s, const InfNPC* jad, int* out_x, int* out_y) {
    int min_dx = -5;
    int max_dx = 5;
    int min_dy = -4;
    int max_dy = 10;
    if (s->wave == 68) {
        min_dx = 0;
        max_dx = 5;
        min_dy = 5;
        max_dy = 8;
    }

    int offsets[INF_JAD_HEALER_MAX_SPAWN_CANDIDATES][2];
    int order[INF_JAD_HEALER_MAX_SPAWN_CANDIDATES];
    int count = 0;
    for (int dx = min_dx; dx <= max_dx; dx++) {
        for (int dy = min_dy; dy <= max_dy; dy++) {
            offsets[count][0] = dx;
            offsets[count][1] = dy;
            order[count] = count;
            count++;
        }
    }
    encounter_shuffle(order, count, &s->rng_state);

    for (int i = 0; i < count; i++) {
        int hx = jad->x + offsets[order[i]][0];
        int hy = jad->y + offsets[order[i]][1];
        if (encounter_entity_footprints_overlap(hx, hy, 1, jad->x, jad->y, jad->size))
            continue;
        if (inf_npc_terrain_blocked(s, hx, hy, 1))
            continue;
        *out_x = hx;
        *out_y = hy;
        return;
    }

    fprintf(stderr, "FATAL: no valid Jad healer spawn tile for Jad at (%d,%d) on wave %d\n",
            jad->x, jad->y, s->wave + 1);
    abort();
}

static void inf_jad_check_healers(InfernoState* s, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (npc->type != INF_NPC_JAD || !npc->active) return;
    if (npc->jad_healer_spawned) return;

    /* spawn healers when below 50% HP */
    if (npc->hp > npc->max_hp / 2) return;
    npc->jad_healer_spawned = 1;

    int num_healers;
    if (s->wave == 66)      num_healers = 5;  /* wave 67: 5 healers */
    else if (s->wave == 67) num_healers = 3;  /* wave 68: 3 per jad */
    else                    num_healers = 3;  /* zuk wave: 3 */

    for (int h = 0; h < num_healers; h++) {
        int slot = inf_find_free_npc(s);
        if (slot < 0) break;
        int hx = npc->x;
        int hy = npc->y;
        inf_sample_jad_healer_spawn(s, npc, &hx, &hy);
        inf_init_npc(s, slot, INF_NPC_HEALER_JAD, hx, hy);
        s->npcs[slot].jad_owner_idx = idx;
        s->npcs[slot].aggro_target = idx;
    }
}

/* ======================================================================== */
/* NPC AI: zuk phases                                                        */
/* ======================================================================== */

static void inf_zuk_tick(InfernoState* s) {
    if (!inf_is_final_wave(s)) return;

    /* find zuk NPC */
    int zuk_idx = inf_find_live_zuk_idx(s);
    if (zuk_idx < 0) return;
    InfNPC* zuk = &s->npcs[zuk_idx];

    /* shield oscillation */
    int si = s->zuk.shield_idx;
    if (si >= 0 && s->npcs[si].active) {
        InfNPC* shield = &s->npcs[si];
        if (s->zuk.shield_freeze > 0) {
            s->zuk.shield_freeze--;
        } else {
            int ox = shield->x;
            int oy = shield->y;
            inf_clear_npc_collision_footprint(s, ox, oy, shield->size);
            shield->x += s->zuk.shield_dir;
            /* boundary check: 5-tick freeze at edges */
            if (shield->x < 11) {
                shield->x = 11;
                s->zuk.shield_freeze = 5;
                s->zuk.shield_dir = 1;
            } else if (shield->x > 35) {
                shield->x = 35;
                s->zuk.shield_freeze = 5;
                s->zuk.shield_dir = -1;
            }
            inf_stamp_npc_collision_footprint(s, shield->x, shield->y, shield->size);
        }
    }

    /* set timer pause: freeze at HP < 600, resume at HP < 480 with +175 ticks */
    if (!s->zuk.has_paused && zuk->hp < 600) {
        s->zuk.timer_paused = 1;
        s->zuk.has_paused = 1;
    }

    /* jad spawn at HP < 480 (with shield alive, timer paused) */
    if (!s->zuk.jad_spawned && s->zuk.timer_paused && zuk->hp < 480 &&
        si >= 0 && s->npcs[si].active) {
        s->zuk.jad_spawned = 1;
        s->zuk.timer_paused = 0;
        s->zuk.set_timer += 175;
        int j_slot = inf_find_free_npc(s);
        if (j_slot >= 0) {
            inf_init_npc(s, j_slot, INF_NPC_JAD, 24, 32);
            s->npcs[j_slot].aggro_target = si;  /* target shield */
            s->npcs[j_slot].stun_timer = 7;     /* spawn delay */
        }
    }

    /* set timer: spawns JalZek + JalXil targeting the shield */
    if (!s->zuk.timer_paused) {
        if (s->zuk.set_timer > 0) {
            s->zuk.set_timer--;
        } else {
            int m_slot = inf_find_free_npc(s);
            if (m_slot >= 0) {
                inf_init_npc(s, m_slot, INF_NPC_MAGER, 20, 36);
                if (si >= 0) s->npcs[m_slot].aggro_target = si;
                s->npcs[m_slot].stun_timer = 7;  /* spawn delay */
            }
            int r_slot = inf_find_free_npc(s);
            if (r_slot >= 0) {
                inf_init_npc(s, r_slot, INF_NPC_RANGER, 29, 36);
                if (si >= 0) s->npcs[r_slot].aggro_target = si;
                s->npcs[r_slot].stun_timer = 9;  /* spawn delay */
            }
            s->zuk.set_timer = s->zuk.set_interval;
        }
    }

    /* healer spawn at HP < 240: 4 JalMejJak targeting Zuk, sets enraged */
    if (!s->zuk.healer_spawned && zuk->hp < 240) {
        s->zuk.healer_spawned = 1;
        s->zuk.enraged = 1;
        static const int healer_pos[4][2] = {
            {16, 48}, {20, 48}, {30, 48}, {34, 48}
        };
        for (int h = 0; h < 4; h++) {
            int slot = inf_find_free_npc(s);
            if (slot >= 0) {
                inf_init_npc(s, slot, INF_NPC_HEALER_ZUK,
                             healer_pos[h][0], healer_pos[h][1]);
                s->npcs[slot].aggro_target = zuk_idx;  /* heal Zuk until tagged */
            }
        }
    }

    /* on zuk death: all other mobs die */
    if (zuk->hp <= 0) {
        for (int i = 0; i < INF_MAX_NPCS; i++) {
            inf_deactivate_npc(s, i);
        }
    }
}

static void inf_healer_apply_landed_heal(InfernoState* s, int idx) {
    InfNPC* npc = &s->npcs[idx];
    int target_idx = npc->heal_target;
    int heal_cap = (npc->type == INF_NPC_HEALER_ZUK) ? 25 : 20;

    npc->heal_target = -1;
    if (target_idx < 0 || target_idx >= INF_MAX_NPCS) return;
    if (!s->npcs[target_idx].active) return;

    int heal = encounter_rand_int(&s->rng_state, heal_cap);
    int before = s->npcs[target_idx].hp;
    s->npcs[target_idx].hp += heal;
    if (s->npcs[target_idx].hp > s->npcs[target_idx].max_hp)
        s->npcs[target_idx].hp = s->npcs[target_idx].max_hp;
    /* effective heal (clamped at max_hp) — feeds hp_restored_this_tick so the
       agent loses reward on ticks where its damage is being undone. */
    s->hp_restored_this_tick += (float)(s->npcs[target_idx].hp - before);
}

static void inf_queue_pending_spark(
    InfernoState* s, int src_x, int src_y, int x, int y, int damage, int ticks_remaining
) {
    for (int i = 0; i < INF_MAX_PENDING_SPARKS; i++) {
        if (s->pending_sparks[i].active) continue;
        s->pending_sparks[i].active = 1;
        s->pending_sparks[i].src_x = src_x;
        s->pending_sparks[i].src_y = src_y;
        s->pending_sparks[i].x = x;
        s->pending_sparks[i].y = y;
        s->pending_sparks[i].damage = damage;
        s->pending_sparks[i].ticks_remaining = ticks_remaining;
        s->pending_sparks[i].visual_emitted = 0;
        return;
    }
}

static void inf_queue_zuk_healer_sparks(InfernoState* s, const InfNPC* npc) {
    int clamped_x = s->player.x;
    if (clamped_x < npc->x - 5) clamped_x = npc->x - 5;
    if (clamped_x > npc->x + 4) clamped_x = npc->x + 4;

    /* InfernoTrainer JalMejJak AoeWeapon: 2 random sparks target y=14+rand(0..3)
       in reference coords. our coord system is Y-flipped (ours_y = 57 - ref_y),
       so ref y=14..17 maps to ours y=40..43 (near player/zuk band). */
    inf_queue_pending_spark(s, npc->x, npc->y, clamped_x, s->player.y,
                            5 + encounter_rand_int(&s->rng_state, 6), 4);
    inf_queue_pending_spark(s,
                            npc->x, npc->y,
                            npc->x + encounter_rand_int(&s->rng_state, 11) - 5,
                            40 + encounter_rand_int(&s->rng_state, 4),
                            5 + encounter_rand_int(&s->rng_state, 6), 4);
    inf_queue_pending_spark(s,
                            npc->x, npc->y,
                            npc->x + encounter_rand_int(&s->rng_state, 11) - 5,
                            40 + encounter_rand_int(&s->rng_state, 4),
                            5 + encounter_rand_int(&s->rng_state, 6), 4);
}

static void inf_resolve_pending_sparks(InfernoState* s) {
    for (int i = 0; i < INF_MAX_PENDING_SPARKS; i++) {
        if (!s->pending_sparks[i].active) continue;
        s->pending_sparks[i].ticks_remaining--;
        if (s->pending_sparks[i].ticks_remaining > 0) continue;

        if (encounter_entity_footprints_overlap(
                s->pending_sparks[i].x - 1, s->pending_sparks[i].y - 1, 3,
                s->player.x, s->player.y, 1)) {
            encounter_damage_player(&s->player, s->pending_sparks[i].damage,
                                    &s->damage_received_this_tick);
            s->last_hit_by_type = INF_NPC_HEALER_ZUK;
        }
        s->pending_sparks[i].active = 0;
    }
}

/* ======================================================================== */
/* NPC AI: tick all NPCs                                                     */
/* ======================================================================== */

static void inf_tick_npcs(InfernoState* s) {
    /* NPC per-tick flags are cleared in inf_step BEFORE inf_tick_player,
       so player hit flags survive through both tick functions into render_post_tick. */

    /* zuk-specific phases first */
    inf_zuk_tick(s);

    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!s->npcs[i].active) continue;

        if ((s->npcs[i].type == INF_NPC_HEALER_JAD || s->npcs[i].type == INF_NPC_HEALER_ZUK) &&
            s->npcs[i].death_ticks == 0 && s->npcs[i].heal_timer > 0) {
            s->npcs[i].heal_timer--;
            if (s->npcs[i].heal_timer == 0)
                inf_healer_apply_landed_heal(s, i);
        }

        /* death linger: decrement and deactivate when done */
        if (s->npcs[i].death_ticks > 0) {
            s->npcs[i].death_ticks--;
            if (s->npcs[i].death_ticks == 0) inf_deactivate_npc(s, i);
            continue;  /* dying NPCs don't move or attack */
        }

        /* decrement ice barrage freeze timer */
        if (s->npcs[i].frozen_ticks > 0) s->npcs[i].frozen_ticks--;

        if (s->npcs[i].type == INF_NPC_MAGER && s->npcs[i].resurrect_cooldown > 0)
            s->npcs[i].resurrect_cooldown--;

        /* meleer dig check */
        if (s->npcs[i].type == INF_NPC_MELEER)
            inf_meleer_dig_check(s, i);

        inf_npc_move(s, i);
        inf_npc_attack(s, i);

        /* jad healer spawning */
        if (s->npcs[i].type == INF_NPC_JAD)
            inf_jad_check_healers(s, i);
    }
}

/* ======================================================================== */
/* player actions                                                            */
/* ======================================================================== */

#define INF_HEAD_MOVE      0   /* 25: idle + 8 walk + 16 run */
#define INF_HEAD_PRAYER    1   /* 4: no_change, toggle_melee, toggle_ranged, toggle_magic (ENCOUNTER_OVERHEAD_DIM_PVE) */
#define INF_HEAD_TARGET    2   /* INF_OBS_NPCS+1: none or observation slot */
#define INF_HEAD_GEAR      3   /* 5: no_switch, mage, tbow, bp, tank */
#define INF_HEAD_EAT       4   /* 2: none, brew */
#define INF_HEAD_POTION    5   /* 4: none, restore, bastion, stamina */
#define INF_HEAD_SPELL     6   /* 3: no_change, blood_barrage, ice_barrage */
#define INF_HEAD_SPEC      7   /* 2: no_change, toggle (arm/disarm blowpipe spec) */
#define INF_HEAD_OFFENSIVE 8   /* 4: no_change, toggle_piety, toggle_rigour, toggle_augury (ENCOUNTER_OFFENSIVE_DIM) */
static const int INF_ACTION_DIMS[INF_NUM_ACTION_HEADS] = {
    ENCOUNTER_MOVE_ACTIONS, ENCOUNTER_OVERHEAD_DIM_PVE, INF_OBS_NPCS+1, 5, 2, 4, 3, 2, ENCOUNTER_OFFENSIVE_DIM
};
#define INF_ACTION_MASK_SIZE (ENCOUNTER_MOVE_ACTIONS + ENCOUNTER_OVERHEAD_DIM_PVE + INF_OBS_NPCS+1 + 5 + 2 + 4 + 3 + 2 + ENCOUNTER_OFFENSIVE_DIM)

/* movement uses shared encounter_move_to_target from osrs_encounter.h */

/* walkability callback for encounter_move_to_target */
static int inf_tile_walkable(void* ctx, int x, int y) {
    InfernoState* s = (InfernoState*)ctx;
    if (!inf_in_arena(x, y)) return 0;
    if (inf_blocked_by_pillar(s, x, y, 1)) return 0;
    if (s->collision_map)
        if (!collision_tile_walkable(s->collision_map, 0,
                x + s->world_offset_x, y + s->world_offset_y))
            return 0;
    return !inf_npc_collision_flag_blocked(s, x, y, 1);
}

/* sara brew heal at base HP 99: floor(99*0.15)+2 = 16. ref: osrs_consumables.h osrs_brew_effect */
#define INF_BREW_HEAL       (99 * 15 / 100 + 2)
/* super restore at base prayer 99: 8 + floor(99/4) = 32. ref: osrs_consumables.h osrs_drink_potion */
#define INF_RESTORE_AMOUNT  (8 + 99 / 4)

/* apply NPC death: blob split, mager resurrection store, jad healer cleanup.
   call after reducing npc->hp. checks if hp <= 0 and handles death effects. */
static void inf_apply_npc_death(InfernoState* s, int npc_idx) {
    InfNPC* npc = &s->npcs[npc_idx];
    if (npc->hp > 0 || !npc->active || npc->death_ticks > 0) return;
    /* keep active=1 for death_ticks so renderer shows final hitsplat + death anim.
       inf_tick_npcs decrements death_ticks and sets active=0 when it reaches 0. */
    npc->death_ticks = 2;
    s->total_npc_kills++;

    if (npc->type == INF_NPC_BLOB) {
        InfNPCType split_types[3] = {
            INF_NPC_BLOB_MELEE, INF_NPC_BLOB_RANGE, INF_NPC_BLOB_MAGE
        };
        for (int sp = 0; sp < 3; sp++) {
            int slot = inf_find_free_npc(s);
            if (slot < 0) break;
            inf_init_npc(s, slot, split_types[sp], npc->x + (sp - 1), npc->y);
        }
    } else {
        inf_store_dead_mob(s, npc);
    }

    if (npc->type == INF_NPC_JAD) {
        for (int j = 0; j < INF_MAX_NPCS; j++) {
            if (s->npcs[j].active &&
                s->npcs[j].type == INF_NPC_HEALER_JAD &&
                s->npcs[j].jad_owner_idx == npc_idx) {
                inf_deactivate_npc(s, j);
            }
        }
    }
}

static void inf_player_pretick(InfernoState* s, const int* actions) {
    /* apply prayer actions. each helper returns 1 on OFF→ON transition so we
       can skip that slot's drain this tick (wiki: no drain on activation tick). */
    OffensivePrayer prev_offensive = s->player.offensive_prayer;
    if (encounter_apply_overhead_action(&s->player.prayer, actions[INF_HEAD_PRAYER])) {
        s->player.prayer_just_activated = 1;
    }
    if (encounter_apply_offensive_action(&s->player.offensive_prayer, actions[INF_HEAD_OFFENSIVE])) {
        s->player.offensive_prayer_just_activated = 1;
    }
    /* inferno loadouts have ~0 prayer bonus (armadyl/ancestral/torva); pass 0.
       drain can also clear offensive_prayer (pp<=0 auto-clear), so recompute
       AFTER drain to catch both the apply-side change and the drain-side clear. */
    encounter_drain_all_prayers(&s->player, 0);
    /* offensive prayer is baked into eff_level/max_hit via the loadout cache.
       recompute all loadouts on any change so combat math reflects current state. */
    if (s->player.offensive_prayer != prev_offensive) {
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
        if (s->human_command_mode)
            inf_refresh_human_loadout_stats(s);
    }
}

static FightStyle inf_default_fight_style_for_style(AttackStyle style) {
    if (style == ATTACK_STYLE_MAGIC) return FIGHT_STYLE_AUTOCAST;
    if (style == ATTACK_STYLE_RANGED) return FIGHT_STYLE_RAPID;
    return FIGHT_STYLE_ACCURATE;
}

static void inf_note_human_weapon_set(InfernoState* s) {
    uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    for (int g = 0; g < INF_NUM_WEAPON_SETS; g++) {
        if (INF_LOADOUTS[g][GEAR_SLOT_WEAPON] == weapon) {
            s->weapon_set = (InfWeaponSet)g;
            return;
        }
    }
}

static void inf_apply_human_player_commands(InfernoState* s) {
    int did_change_equipment = 0;
    for (int i = 0; i < s->human_command_count; i++) {
        const HumanCommand* cmd = &s->human_commands[i];
        if (cmd->kind == HUMAN_COMMAND_EQUIP_INVENTORY_ITEM) {
            if (cmd->gear_slot >= 0 && cmd->gear_slot < NUM_GEAR_SLOTS &&
                cmd->item_db_idx >= 0 && cmd->item_db_idx < NUM_ITEMS) {
                int changed = slot_equip_item(&s->player, cmd->gear_slot, (uint8_t)cmd->item_db_idx);
                if (changed) {
                    s->total_gear_switches++;
                    did_change_equipment = 1;
                    if (cmd->gear_slot == GEAR_SLOT_WEAPON) {
                        AttackStyle style = inf_player_equipped_attack_style(s);
                        s->player.fight_style = inf_default_fight_style_for_style(style);
                        inf_note_human_weapon_set(s);
                    }
                }
            }
        } else if (cmd->kind == HUMAN_COMMAND_FIGHT_STYLE) {
            if (cmd->fight_style >= FIGHT_STYLE_ACCURATE &&
                cmd->fight_style <= FIGHT_STYLE_DEFENSIVE_AUTOCAST) {
                s->player.fight_style = (FightStyle)cmd->fight_style;
                did_change_equipment = 1;
            }
        }
    }
    if (did_change_equipment)
        inf_refresh_human_loadout_stats(s);
}

static void inf_tick_player(InfernoState* s, const int* actions) {
    if (s->player_last_interaction_age == 0)
        s->player_last_interaction_age = 1;

    if (s->human_command_mode) {
        inf_apply_human_player_commands(s);
    } else {
        int gear_act = actions[INF_HEAD_GEAR];
        if (gear_act >= 1) s->total_gear_switches++;
        if (gear_act >= 1 && gear_act <= 3) {
            InfWeaponSet new_set = (InfWeaponSet)(gear_act - 1);
            s->weapon_set = new_set;
            s->armor_tank = 0;
            GearSet gs = (new_set == INF_GEAR_MAGE) ? GEAR_MAGE : GEAR_RANGED;
            encounter_apply_loadout(&s->player, INF_LOADOUTS[new_set], gs);
        } else if (gear_act == 4) {
            s->armor_tank = 0;
        }
        {
            uint8_t current_weapon = s->player.equipped[GEAR_SLOT_WEAPON];
            if (current_weapon != INF_LOADOUTS[s->weapon_set][GEAR_SLOT_WEAPON]) {
                for (int g = 0; g < INF_NUM_WEAPON_SETS; g++) {
                    if (INF_LOADOUTS[g][GEAR_SLOT_WEAPON] == current_weapon) {
                        s->weapon_set = (InfWeaponSet)g;
                        GearSet gs = (g == INF_GEAR_MAGE) ? GEAR_MAGE : GEAR_RANGED;
                        encounter_apply_loadout(&s->player, INF_LOADOUTS[g], gs);
                        break;
                    }
                }
            }
        }
    }

    /* spell choice for mage attacks — normalize to ENCOUNTER_SPELL_*.
       human sends ATTACK_ICE=2 / ATTACK_BLOOD=3, RL sends 0=blood / 1=ice. */
    /* spell: 0=no change, 1=blood barrage, 2=ice barrage */
    int spell_act = actions[INF_HEAD_SPELL];
    if (spell_act == 2)
        s->spell_choice = ENCOUNTER_SPELL_ICE;
    else if (spell_act == 1)
        s->spell_choice = ENCOUNTER_SPELL_BLOOD;

    /* special energy regen: 10 energy every 50 ticks (30 seconds) */
    encounter_tick_spec_regen(&s->player);

    /* spec toggle: arm/disarm (does NOT interrupt interaction) */
    if (actions[INF_HEAD_SPEC] == 1)
        osrs_spec_toggle(&s->player.spec_armed);

    /* stat decay: every 60 ticks, boosted/drained stats move 1 toward base */
    if (s->tick > 0 && s->tick % 60 == 0) {
        int* stats[] = { &s->player.current_ranged, &s->player.current_magic,
                         &s->player.current_attack, &s->player.current_strength,
                         &s->player.current_defence };
        for (int si = 0; si < 5; si++) {
            if (*stats[si] > 99) (*stats[si])--;
            else if (*stats[si] < 99) (*stats[si])++;
        }
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
        if (s->human_command_mode)
            inf_refresh_human_loadout_stats(s);
    }

    /* consumables — shared 3-tick potion timer */
    if (s->player.potion_timer > 0) s->player.potion_timer--;
    if (s->stamina_active_ticks > 0) s->stamina_active_ticks--;

    /* brew (INF_HEAD_EAT): heals 16 HP, can overcap to base+16 */
    int eat_act = actions[INF_HEAD_EAT];
    if (eat_act == 1 && s->player.brew_doses > 0 && s->player.potion_timer == 0
        && s->player.current_hitpoints < s->player.base_hitpoints) {
        s->player.current_hitpoints += INF_BREW_HEAL;
        if (s->player.current_hitpoints > s->player.base_hitpoints + INF_BREW_HEAL)
            s->player.current_hitpoints = s->player.base_hitpoints + INF_BREW_HEAL;
        s->player.brew_doses--;
        s->player.potion_timer = 3;
        s->player.ate_food_this_tick = 1;
        s->brewed_this_tick = 1;
        encounter_brew_drain_stats(&s->player);
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
        if (s->human_command_mode)
            inf_refresh_human_loadout_stats(s);
    }

    /* potions (INF_HEAD_POTION): 1=restore, 2=bastion, 3=stamina */
    int pot_act = actions[INF_HEAD_POTION];
    if (pot_act == 1 && s->player.restore_doses > 0 && s->player.potion_timer == 0) {
        /* super restore: restores prayer + all combat stats */
        s->player.current_prayer += INF_RESTORE_AMOUNT;
        if (s->player.current_prayer > s->player.base_prayer)
            s->player.current_prayer = s->player.base_prayer;
        encounter_restore_stats(&s->player);
        s->player.restore_doses--;
        s->player.potion_timer = 3;
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
        if (s->human_command_mode)
            inf_refresh_human_loadout_stats(s);
    } else if (pot_act == 2 && s->player.bastion_doses > 0 && s->player.potion_timer == 0) {
        encounter_bastion_boost(&s->player);
        s->player.bastion_doses--;
        s->player.potion_timer = 3;
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
        if (s->human_command_mode)
            inf_refresh_human_loadout_stats(s);
    } else if (pot_act == 3 && s->player.stamina_doses > 0 && s->player.potion_timer == 0) {
        s->stamina_active_ticks = 200;
        s->player.stamina_doses--;
        s->player.potion_timer = 3;
    }

    /* inventory actions interrupt interaction (per OSRS entity interaction rules) */
    if (eat_act > 0)
        osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_EAT);
    if (pot_act > 0)
        osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_DRINK);

    /* attack target selection: persistent until NPC dies or player clicks ground.
       target=0 means "no new target this tick" (preserves existing target). */
    int target = actions[INF_HEAD_TARGET];
    int has_new_target = 0;
    if (target > 0 && target <= INF_OBS_NPCS) {
        int obs_idx = target - 1;
        int npc_idx = s->current_obs_slots[obs_idx];
        if (npc_idx >= 0 && npc_idx < INF_MAX_NPCS && 
            s->npcs[npc_idx].active && s->npcs[npc_idx].death_ticks == 0 &&
            s->npcs[npc_idx].type != INF_NPC_ZUK_SHIELD) {
            osrs_interaction_set(&s->interaction, npc_idx);
            s->player_last_interaction_target_slot = npc_idx;
            s->player_last_interaction_age = 0;
            has_new_target = 1;
        }
    }
    /* explicit movement (ground click or RL move) cancels attack target,
       but only if no new target was set this tick. auto-chase movement
       does NOT cancel — only explicit user actions do. */
    int has_explicit_move = (actions[INF_HEAD_MOVE] > 0 || s->player_dest_x >= 0);
    if (!has_new_target && has_explicit_move)
        osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_MOVE);
    /* clear target if NPC died or is dying */
    if (osrs_interaction_active(&s->interaction) &&
        (!s->npcs[s->interaction.target_slot].active ||
         s->npcs[s->interaction.target_slot].death_ticks > 0)) {
        osrs_interaction_clear(&s->interaction);
    }

    /* movement: explicit move, auto-chase toward target, or idle.
       OSRS order: target selection → movement → attack check. */
    if (has_explicit_move && !osrs_interaction_active(&s->interaction)) {
        /* explicit movement (ground click or RL agent) — no attack target */
        if (s->player_dest_x >= 0) {
            encounter_move_toward_dest(&s->player, &s->player_dest_x, &s->player_dest_y,
                s->collision_map, s->world_offset_x, s->world_offset_y,
                inf_tile_walkable, s, inf_pathfind_blocked, s,
                INF_ARENA_MIN_X, INF_ARENA_MIN_Y, INF_ARENA_WIDTH, INF_ARENA_HEIGHT);
        } else {
            int move_act = actions[INF_HEAD_MOVE];
            s->player.is_running = 0;
            if (move_act > 0 && move_act < ENCOUNTER_MOVE_ACTIONS) {
                encounter_move_to_target(&s->player,
                    ENCOUNTER_MOVE_TARGET_DX[move_act], ENCOUNTER_MOVE_TARGET_DY[move_act],
                    inf_tile_walkable, s);
            }
        }
    } else if (osrs_interaction_active(&s->interaction)) {
        /* auto-chase: pathfind toward attack target when out of range */
        InfNPC* chase_npc = &s->npcs[s->interaction.target_slot];
        const EncounterLoadoutStats* ls = inf_current_loadout_stats(s);
        encounter_chase_attack_target(&s->player,
            chase_npc->x, chase_npc->y, INF_NPC_STATS[chase_npc->type].size,
            ls->attack_range,
            s->collision_map, s->world_offset_x, s->world_offset_y,
            inf_tile_walkable, s, inf_pathfind_blocked, s,
            s->los_blockers, s->los_blocker_count,
            INF_ARENA_MIN_X, INF_ARENA_MIN_Y, INF_ARENA_WIDTH, INF_ARENA_HEIGHT);
    }
    inf_rebuild_player_collision_flags(s);

    /* player attacks targeted NPC */
    if (s->player.attack_timer > 0) s->player.attack_timer--;
    if (osrs_interaction_active(&s->interaction) && s->player.attack_timer == 0) {
        InfNPC* target_npc = &s->npcs[s->interaction.target_slot];
        if (target_npc->active) {
            const EncounterLoadoutStats* ls = inf_current_loadout_stats(s);

            /* range + LOS check: must have line of sight through pillars */
            int target_dist = encounter_dist_to_npc(s->player.x, s->player.y,
                target_npc->x, target_npc->y, target_npc->size);

            /* magic level gate: can't cast a barrage below its required level.
               real OSRS greys out the spell button; here we skip the entire attack
               so no damage lands, attack_timer does not reset, and the agent retries
               next tick (or picks a different spell). */
            int is_magic_attack = (ls->style == ATTACK_STYLE_MAGIC);
            int weapon_is_blowpipe = inf_player_weapon_is(s, ITEM_TOXIC_BLOWPIPE);
            int weapon_is_tbow = inf_player_weapon_is(s, ITEM_TWISTED_BOW);
            int mage_blocked = is_magic_attack &&
                (s->player.current_magic < ((s->spell_choice == ENCOUNTER_SPELL_ICE)
                    ? ICE_BARRAGE_LEVEL : BLOOD_BARRAGE_LEVEL));

            if (!mage_blocked && encounter_player_can_attack(s->player.x, s->player.y,
                    target_npc->x, target_npc->y, target_npc->size,
                    ls->attack_range, s->los_blockers, s->los_blocker_count)) {
                /* compute hit delay for projectile flight */
                int hit_delay;
                if (ls->style == ATTACK_STYLE_MAGIC)
                    hit_delay = encounter_magic_hit_delay(target_dist, 1);
                else if (weapon_is_blowpipe)
                    hit_delay = encounter_blowpipe_hit_delay(target_dist, 1);
                else
                    hit_delay = encounter_ranged_hit_delay(target_dist, 1);

                int total_dmg = 0;

                if (is_magic_attack) {
                    /* barrage spells: 3x3 AoE via shared osrs_barrage_resolve.
                       ice barrage: freeze on hit (including 0 dmg), not on splash.
                       blood barrage: heal 25% of total AoE damage (applied when hits land). */
                    OsrsTargetRef target_ref = {
                        .kind = OSRS_TARGET_NPC,
                        .id = s->interaction.target_slot,
                    };
                    OsrsMagicAttackKind magic_kind = (s->spell_choice == ENCOUNTER_SPELL_ICE)
                        ? OSRS_MAGIC_ATTACK_ANCIENT_ICE
                        : OSRS_MAGIC_ATTACK_ANCIENT_BLOOD;
                    OsrsPreparedAttackEffects attack_effects = osrs_prepare_attack_effects(
                        &s->player.equipment_effect_profile,
                        &s->player.item_effect_state,
                        s->player.equipped[GEAR_SLOT_WEAPON],
                        ATTACK_STYLE_MAGIC,
                        magic_kind,
                        target_ref,
                        1,
                        ls->eff_level * (ls->attack_bonus + 64),
                        ls->max_hit,
                        0,
                        0,
                        s->player.current_hitpoints,
                        s->player.base_hitpoints
                    );

                    /* build target array: primary target first, then all other active NPCs */
                    BarrageTarget btargets[INF_MAX_NPCS + 1];
                    int bt_count = 0;
                    {
                        const InfNPCStats* ns = &INF_NPC_STATS[target_npc->type];
                        btargets[bt_count++] = (BarrageTarget){
                            .active = 1, .x = target_npc->x, .y = target_npc->y,
                            .def_level = ns->def_level, .magic_def_bonus = ns->magic_def_bonus,
                            .npc_idx = s->interaction.target_slot,
                            .frozen_ticks = &s->npcs[s->interaction.target_slot].frozen_ticks,
                            .hit = 0, .damage = 0
                        };
                    }
                    for (int i = 0; i < INF_MAX_NPCS; i++) {
                        if (i == s->interaction.target_slot || !s->npcs[i].active) continue;
                        /* shield is invulnerable — no damage, no freeze from AoE */
                        if (s->npcs[i].type == INF_NPC_ZUK_SHIELD) continue;
                        /* skip dying NPCs (2-tick death anim window) — already logged as killed */
                        if (s->npcs[i].death_ticks > 0) continue;
                        const InfNPCStats* ns2 = &INF_NPC_STATS[s->npcs[i].type];
                        btargets[bt_count++] = (BarrageTarget){
                            .active = 1, .x = s->npcs[i].x, .y = s->npcs[i].y,
                            .def_level = ns2->def_level, .magic_def_bonus = ns2->magic_def_bonus,
                            .npc_idx = i,
                            .frozen_ticks = &s->npcs[i].frozen_ticks,
                            .hit = 0, .damage = 0
                        };
                    }

                    /* resolve barrage: accuracy/damage rolls + instant freeze for ice.
                       freeze is applied by the shared function at cast time. */
                    BarrageResult br = osrs_barrage_resolve(
                        btargets, bt_count, attack_effects.attack_roll, attack_effects.max_hit,
                        &s->rng_state, s->spell_choice, attack_effects.use_double_accuracy);
                    total_dmg = br.total_damage;
                    osrs_finalize_attack_effects(
                        &s->player.equipment_effect_profile,
                        &s->player.item_effect_state,
                        s->player.equipped[GEAR_SLOT_WEAPON],
                        ATTACK_STYLE_MAGIC,
                        magic_kind,
                        target_ref,
                        1,
                        attack_effects.use_double_accuracy,
                        btargets[0].hit,
                        btargets[0].damage,
                        &s->rng_state
                    );

                    /* queue pending hits for delayed damage */
                    for (int i = 0; i < bt_count; i++) {
                        if (!btargets[i].active || !btargets[i].hit) continue;
                        int nidx = btargets[i].npc_idx;
                        EncounterPendingHit* ph = &s->npcs[nidx].pending_hit;
                        ph->active = 1;
                        ph->damage = btargets[i].damage;
                        ph->ticks_remaining = hit_delay;
                        ph->attack_style = ATTACK_STYLE_MAGIC;
                        ph->check_prayer = 0;
                        ph->spell_type = s->spell_choice;
                    }

                } else if (weapon_is_tbow) {
                    const InfNPCStats* ns = &INF_NPC_STATS[target_npc->type];
                    OsrsPreparedAttackEffects attack_effects = osrs_prepare_attack_effects(
                        &s->player.equipment_effect_profile,
                        &s->player.item_effect_state,
                        s->player.equipped[GEAR_SLOT_WEAPON],
                        ATTACK_STYLE_RANGED,
                        OSRS_MAGIC_ATTACK_NONE,
                        (OsrsTargetRef){ .kind = OSRS_TARGET_NPC, .id = s->interaction.target_slot },
                        1,
                        ls->eff_level * (ls->attack_bonus + 64),
                        ls->max_hit,
                        ns->magic_level,
                        ns->magic_att_bonus,
                        s->player.current_hitpoints,
                        s->player.base_hitpoints
                    );
                    int def_roll = (ns->def_level + 8) * (ns->ranged_def_bonus + 64);
                    if (encounter_rand_float(&s->rng_state) < osrs_hit_chance(attack_effects.attack_roll, def_roll)) {
                        total_dmg = encounter_rand_int(&s->rng_state, attack_effects.max_hit + 1);
                    }
                    EncounterPendingHit* ph = &target_npc->pending_hit;
                    ph->active = 1;
                    ph->damage = total_dmg;
                    ph->ticks_remaining = hit_delay;
                    ph->attack_style = ATTACK_STYLE_RANGED;
                    ph->check_prayer = 0;
                    ph->spell_type = 0;

                } else if (weapon_is_blowpipe && s->player.spec_armed &&
                           encounter_use_spec(&s->player, BLOWPIPE_SPEC_COST)) {
                    /* blowpipe spec: 2x accuracy, 1.5x max hit, heal 50% of damage */
                    osrs_spec_disarm(&s->player.spec_armed);
                    const InfNPCStats* ns = &INF_NPC_STATS[target_npc->type];
                    int base_att_roll = ls->eff_level * (ls->attack_bonus + 64);
                    total_dmg = osrs_blowpipe_spec_resolve(
                        base_att_roll, ls->max_hit,
                        ns->def_level, ns->ranged_def_bonus, &s->rng_state);
                    int heal = total_dmg * BLOWPIPE_SPEC_HEAL_PCT / 100;
                    s->player.current_hitpoints += heal;
                    if (s->player.current_hitpoints > s->player.base_hitpoints)
                        s->player.current_hitpoints = s->player.base_hitpoints;
                    EncounterPendingHit* ph = &target_npc->pending_hit;
                    ph->active = 1;
                    ph->damage = total_dmg;
                    ph->ticks_remaining = hit_delay;
                    ph->attack_style = ATTACK_STYLE_RANGED;
                    ph->check_prayer = 0;
                    ph->spell_type = 0;

                } else {
                    /* blowpipe: single target, normal attack */
                    const InfNPCStats* ns = &INF_NPC_STATS[target_npc->type];
                    int att_roll = ls->eff_level * (ls->attack_bonus + 64);
                    int def_roll = (ns->def_level + 8) * (ns->ranged_def_bonus + 64);
                    if (encounter_rand_float(&s->rng_state) < osrs_hit_chance(att_roll, def_roll)) {
                        total_dmg = encounter_rand_int(&s->rng_state, ls->max_hit + 1);
                    }
                    EncounterPendingHit* ph = &target_npc->pending_hit;
                    ph->active = 1;
                    ph->damage = total_dmg;
                    ph->ticks_remaining = hit_delay;
                    ph->attack_style = ATTACK_STYLE_RANGED;
                    ph->check_prayer = 0;
                    ph->spell_type = 0;
                }

                s->player.attack_timer = ls->attack_speed;

                /* player projectile event for renderer */
                s->player_attacked_this_tick = 1;
                s->player_attack_npc_idx = s->interaction.target_slot;
                s->player_attack_dmg = total_dmg;
                s->player_attack_style_id = ls->style;

                /* player attack animation + spell type for renderer effect system */
                s->player.attack_style_this_tick = ls->style;
                if (ls->style == ATTACK_STYLE_MAGIC) {
                    /* 0=none, 1=ice, 2=blood */
                    s->player.magic_type_this_tick = (s->spell_choice == ENCOUNTER_SPELL_ICE) ? 1 : 2;
                }
            }
        }
    }
}

static int inf_roll_delayed_jad_damage(InfernoState* s, int attack_style) {
    const InfNPCStats* stats = &INF_NPC_STATS[INF_NPC_JAD];
    int max_hit = osrs_npc_max_hit(attack_style,
        stats->str_level, stats->range_level,
        stats->melee_str_bonus, stats->ranged_str_bonus,
        stats->magic_base_dmg, stats->magic_dmg_pct);
    if (stats->max_hit_cap > 0 && max_hit > stats->max_hit_cap)
        max_hit = stats->max_hit_cap;

    int dmg = encounter_rand_int(&s->rng_state, max_hit + 1);
    int att_lvl, att_bonus;
    if (attack_style == ATTACK_STYLE_RANGED) {
        att_lvl = stats->range_level;
        att_bonus = stats->range_att_bonus;
    } else {
        att_lvl = stats->magic_level;
        att_bonus = stats->magic_att_bonus;
    }

    int att_roll = osrs_npc_attack_roll(att_lvl, att_bonus);
    const EncounterLoadoutStats* ls = inf_current_loadout_stats(s);
    int def_bonus = encounter_player_def_bonus(
        ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
        attack_style, stats->melee_style);
    int def_roll = osrs_player_def_roll_vs_npc(
        s->player.current_defence, s->player.current_magic, def_bonus, attack_style);
    if (encounter_rand_float(&s->rng_state) >= osrs_hit_chance(att_roll, def_roll))
        dmg = 0;
    return dmg;
}

static void inf_apply_delayed_prayer_check(InfernoState* s, EncounterPendingHit* hit) {
    if (encounter_prayer_correct_for_style(s->player.prayer, hit->attack_style)) {
        hit->damage = 0;
        s->prayer_correct_this_tick++;
        if (hit->source_npc_type >= 0 && hit->source_npc_type < INF_NUM_NPC_TYPES)
            s->prayer_correct_by_type[hit->source_npc_type]++;
    } else if (hit->source_npc_type == INF_NPC_JAD) {
        hit->damage = inf_roll_delayed_jad_damage(s, hit->attack_style);
        s->dmg_from_type[INF_NPC_JAD] += (float)hit->damage;
        if (hit->damage > 0) {
            s->last_hit_by_type = INF_NPC_JAD;
            s->off_prayer_hits_this_tick++;
        }
    } else if (hit->damage > 0 && hit->attack_style != ATTACK_STYLE_NONE) {
        s->off_prayer_hits_this_tick++;
    }
    hit->check_prayer = 0;
}

static void inf_resolve_player_pending_hits(InfernoState* s) {
    for (int i = 0; i < s->player_pending_hit_count; i++) {
        EncounterPendingHit* hit = &s->player_pending_hits[i];

        if (hit->check_prayer && hit->prayer_check_delay > 0 &&
            hit->source_npc_type != INF_NPC_JAD) {
            hit->prayer_check_delay--;
            if (hit->prayer_check_delay == 0) {
                inf_apply_delayed_prayer_check(s, hit);
            }
        }

        hit->ticks_remaining--;
        if (hit->ticks_remaining <= 0) {
            int dmg = hit->damage;
            if (hit->check_prayer) {
                if (encounter_prayer_correct_for_style(s->player.prayer, hit->attack_style)) {
                    dmg = 0;
                    s->prayer_correct_this_tick++;
                    if (hit->source_npc_type >= 0 && hit->source_npc_type < INF_NUM_NPC_TYPES)
                        s->prayer_correct_by_type[hit->source_npc_type]++;
                } else if (dmg > 0 && hit->attack_style != ATTACK_STYLE_NONE) {
                    s->off_prayer_hits_this_tick++;
                }
            } else if (dmg > 0 && hit->attack_style != ATTACK_STYLE_NONE &&
                       hit->source_npc_type != INF_NPC_JAD) {
                s->off_prayer_hits_this_tick++;
            }

            encounter_damage_player(
                &s->player, dmg, &s->damage_received_this_tick);
            s->player_pending_hits[i] =
                s->player_pending_hits[--s->player_pending_hit_count];
            i--;
        }
    }
}

static void inf_resolve_jad_prayer_checks_after_player(InfernoState* s) {
    for (int i = 0; i < s->player_pending_hit_count; i++) {
        EncounterPendingHit* hit = &s->player_pending_hits[i];
        if (!hit->check_prayer ||
            hit->source_npc_type != INF_NPC_JAD ||
            hit->prayer_check_delay <= 0) {
            continue;
        }
        hit->prayer_check_delay--;
        if (hit->prayer_check_delay == 0)
            inf_apply_delayed_prayer_check(s, hit);
    }
}

/* ======================================================================== */
/* reward                                                                    */
/* ======================================================================== */

static int inf_healer_is_actively_healing(const InfernoState* s, const InfNPC* npc) {
    if (!npc->active || npc->death_ticks > 0) return 0;
    if (npc->type != INF_NPC_HEALER_JAD && npc->type != INF_NPC_HEALER_ZUK) return 0;
    if (npc->aggro_target < 0 || npc->aggro_target >= INF_MAX_NPCS) return 0;
    return s->npcs[npc->aggro_target].active;
}

static float inf_compute_reward(InfernoState* s) {
    s->total_damage_dealt += s->damage_dealt_this_tick;
    s->total_zuk_healer_damage += s->damage_zuk_healers_this_tick;
    s->total_damage_received += s->damage_received_this_tick;
    s->total_hp_restored += s->hp_restored_this_tick;

    if (s->episode_over)
        return (s->winner == 0) ? 1.0f : 0.0f;

    int healer_is_actively_healing = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (inf_healer_is_actively_healing(s, &s->npcs[i])) {
            healer_is_actively_healing = 1;
            break;
        }
    }

    float reward = 0.0f;
    if (healer_is_actively_healing) {
        reward = s->tag_reward_coeff * (float)s->healer_tags_this_tick;
    } else if (inf_is_final_wave(s)) {
        int zuk_idx = inf_find_live_zuk_idx(s);
        if (zuk_idx >= 0) {
            float zuk_hp = (float)s->npcs[zuk_idx].hp;
            if (zuk_hp < s->min_zuk_hp_seen) {
                reward = s->damage_reward_coeff * (s->min_zuk_hp_seen - zuk_hp);
                s->min_zuk_hp_seen = zuk_hp;
            }
        }
    } else {
        reward = s->damage_reward_coeff *
            fmaxf(0.0f, s->damage_dealt_this_tick - s->hp_restored_this_tick);
    }
    reward -= s->shield_penalty_coeff * s->shield_damage_this_tick;
    return reward;
}

/* ======================================================================== */
/* step                                                                      */
/* ======================================================================== */

static void inf_step(EncounterState* state, const int* actions) {
    InfernoState* s = (InfernoState*)state;
    if (s->episode_over) return;

    /* clear per-tick state */
    s->reward = 0.0f;
    s->damage_dealt_this_tick = 0.0f;
    s->damage_zuk_healers_this_tick = 0.0f;
    s->shield_damage_this_tick = 0.0f;
    s->healer_tags_this_tick = 0;
    s->damage_received_this_tick = 0.0f;
    s->hp_restored_this_tick = 0.0f;
    s->prayer_correct_this_tick = 0;
    s->off_prayer_hits_this_tick = 0;
    s->tick_styles_fired = 0;
    s->tick_attacks_fired = 0;
    s->wave_completed_this_tick = 0;
    s->pillar_lost_this_tick = -1;
    s->player_attacked_this_tick = 0;
    s->brewed_this_tick = 0;
    s->blood_heal_this_tick = 0;
    s->behind_shield_this_tick = 0;
    encounter_clear_tick_flags(&s->player);
    /* clear NPC per-tick flags BEFORE player actions, so hit flags set by
       inf_tick_player survive through inf_tick_npcs into render_post_tick */
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        s->npcs[i].attacked_this_tick = 0;
        s->npcs[i].attack_style_this_tick = ATTACK_STYLE_NONE;
        s->npcs[i].attack_visual_target = -1;
        s->npcs[i].moved_this_tick = 0;
        s->npcs[i].hit_landed_this_tick = 0;
        s->npcs[i].hit_damage = 0;
        s->npcs[i].hit_spell_type = 0;
    }
    s->tick++;
    inf_player_pretick(s, actions);

    /* inter-wave countdowns resolve before the player phase, but the queued wave
       does not spawn until the END of the tick. */
    int spawn_wave_now = 0;
    if (s->wave_spawn_delay > 0) {
        s->wave_spawn_delay--;
        if (s->wave_spawn_delay == 0)
            spawn_wave_now = 1;
    }
    int in_wave_gap = (s->wave_spawn_delay > 0 || spawn_wave_now);

    /* ------------------------------------------------------------------ */
    /* OSRS-style tick order: NPCs move/attack first, then projectile landings,
       then the player's movement/attack phase. */
    /* ------------------------------------------------------------------ */
    if (!in_wave_gap) {
        inf_rebuild_player_collision_flags(s);
        inf_invalidate_los_cache(s);
        inf_tick_npcs(s);
    }

    {
        int blood_heal_acc = 0;
        for (int i = 0; i < INF_MAX_NPCS; i++) {
            if (!s->npcs[i].active || s->npcs[i].death_ticks > 0) continue;
            int spell = s->npcs[i].pending_hit.spell_type;
            float dmg_before = s->damage_dealt_this_tick;
            int landed = encounter_resolve_npc_pending_hit(
                &s->npcs[i].pending_hit,
                &s->npcs[i].hp, &s->npcs[i].hit_landed_this_tick, &s->npcs[i].hit_damage,
                &s->npcs[i].frozen_ticks, &blood_heal_acc, &s->damage_dealt_this_tick);
            if (landed) {
                float landed_damage = s->damage_dealt_this_tick - dmg_before;
                if (s->npcs[i].type == INF_NPC_HEALER_ZUK) {
                    s->damage_zuk_healers_this_tick += landed_damage;
                }
                s->npcs[i].hit_spell_type = spell;
                /* tagging: aggro switches on projectile LAND, not FIRE. matches
                   osrs-sdk Unit.ts:645 shouldChangeAggro, called post-damage.
                   without this, healers/set-spawns clear aggro on the fire tick
                   and never land a heal/shield-hit before switching to player. */
                if (s->npcs[i].aggro_target != -1) {
                    if (s->npcs[i].type == INF_NPC_HEALER_ZUK
                        || s->npcs[i].type == INF_NPC_HEALER_JAD) {
                        s->healer_tags_this_tick++;
                    }
                    s->npcs[i].aggro_target = -1;
                    s->npcs[i].stun_timer = 2;  /* flinch: 2-tick delay on aggro switch */
                }
                inf_apply_npc_death(s, i);
            }
        }
        if (blood_heal_acc > 0) {
            int healed = blood_heal_acc / 4;
            s->player.current_hitpoints += healed;
            if (s->player.current_hitpoints > s->player.base_hitpoints)
                s->player.current_hitpoints = s->player.base_hitpoints;
            s->blood_heal_this_tick = healed;
        }
    }

    inf_resolve_player_pending_hits(s);
    inf_resolve_pending_sparks(s);

    /* if npc damage killed the player, stop the tick here — a corpse can't
       eat, attack, or move. without this check the hp-clamp-at-0 in
       encounter_damage_player combined with a subsequent brew in
       inf_tick_player would resurrect lethal hits (observed: zuk 140 lands
       on 115 hp → clamp to 0 → brew +16 → alive at 16). the episode_return
       stays valid because inf_compute_reward's episode_over branch returns
       the end-of-episode shape and no further damage is accrued. */
    if (s->player.current_hitpoints <= 0) {
        if (s->last_hit_by_type >= 0 && s->last_hit_by_type < INF_NUM_NPC_TYPES)
            s->killed_by_type[s->last_hit_by_type]++;
        s->episode_over = 1;
        s->winner = 1;
        s->reward = 0.0f;
        return;
    }

    /* player actions */
    inf_tick_player(s, actions);
    inf_resolve_jad_prayer_checks_after_player(s);

    /* idle penalty counter: consecutive ticks where player could attack but didn't */
    {
        int has_alive_npc = 0;
        if (!in_wave_gap) {
            for (int i = 0; i < INF_MAX_NPCS; i++) {
                if (s->npcs[i].active && s->npcs[i].death_ticks == 0) {
                    has_alive_npc = 1; break;
                }
            }
        }
        if (has_alive_npc && s->player.attack_timer == 0 && !s->player_attacked_this_tick)
            s->ticks_without_action++;
        else
            s->ticks_without_action = 0;
    }

    /* accumulate diagnostic counters.
       prayer_correct_this_tick is a count (multiple NPCs can attack same tick).
       total_npc_attacks counts attacks directed at the player (not nibbler→pillar). */
    s->total_prayer_correct += s->prayer_correct_this_tick;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].attacked_this_tick && s->npcs[i].aggro_target < 0 && s->npcs[i].type != INF_NPC_NIBBLER && !(s->npcs[i].type == INF_NPC_BLOB && s->npcs[i].blob_scanned_prayer >= 0))
            s->total_npc_attacks++;
    }
    /* multi-style analysis: count off-prayer hits that were unavoidable because
       a different style was correctly prayed on the same tick. popcount of
       tick_styles_fired tells us how many distinct styles fired. if 2+, the
       off-prayer hits from non-prayed styles are "unavoidable" (can only pray one). */
    if (s->tick_attacks_fired > 0) {
        int styles = s->tick_styles_fired;
        int n_styles = ((styles >> 0) & 1) + ((styles >> 1) & 1) + ((styles >> 2) & 1);
        if (n_styles >= 2 && s->prayer_correct_this_tick > 0) {
            /* we prayed correctly against at least one style, but other styles
               also fired — those off-prayer hits were unavoidable */
            int off_prayer = s->tick_attacks_fired - s->prayer_correct_this_tick;
            s->total_unavoidable_off += off_prayer;
        }
    }
    if (s->ticks_without_action > 0) s->total_idle_ticks++;
    s->total_brews_used += s->brewed_this_tick;
    s->total_blood_healed += s->blood_heal_this_tick;

    /* Zuk shield tracking: are we behind the shield this tick? */
    if (!in_wave_gap && s->wave == 68) {
        s->total_zuk_ticks++;
        int si = s->zuk.shield_idx;
        if (si >= 0 && s->npcs[si].active) {
            int sx = s->npcs[si].x;
            int sz = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size;
            if (s->player.x >= sx && s->player.x < sx + sz && s->player.y >= 41) {
                s->behind_shield_ticks++;
                s->behind_shield_this_tick = 1;
            }
        }
    }

    /* action noop tracking */
    s->action_total_count++;
    for (int h = 0; h < INF_NUM_ACTION_HEADS; h++) {
        if (actions[h] == 0) s->action_noop_count[h]++;
    }

    /* bank the tick's irreversible HP progress before computing reward. all
       damage landings and healer applies have resolved by this point. */
    s->reward = inf_compute_reward(s);
    s->episode_return += s->reward;

    /* check player death */
    if (s->player.current_hitpoints <= 0) {
        if (s->last_hit_by_type >= 0 && s->last_hit_by_type < INF_NUM_NPC_TYPES)
            s->killed_by_type[s->last_hit_by_type]++;
        s->episode_over = 1;
        s->winner = 1;
        /* terminal reward: override the per-tick reward already computed above.
           do NOT call inf_compute_reward again (would double-count damage stats). */
        s->reward = 0.0f;  /* player died. Do not negative reward to avoid stalling.*/
        return;
    }

    if (spawn_wave_now) {
        s->wave = s->wave_spawn_target;
        inf_spawn_wave(s);
        inf_invalidate_los_cache(s);
        return;
    }
    if (s->wave_spawn_delay > 0) return;

    /* check wave completion */
    int all_dead = 1;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].active) { all_dead = 0; break; }
    }
    if (all_dead) {
        s->wave_completed_this_tick = 1;
        s->total_waves_cleared++;
        s->reward = 1.0f;
        s->episode_return += 1.0f;
        if (s->wave + 1 >= INF_NUM_WAVES) {
            s->episode_over = 1;
            s->winner = 0;
        } else {
            s->wave_spawn_target = s->wave + 1;
            s->wave_spawn_delay = 9;
        }
    }

    /* timeout — zero reward so agent isn't rewarded/penalized for running out of time */
    if (s->tick >= INF_MAX_TICKS) {
        s->episode_over = 1;
        s->winner = 1;
        s->reward = 0.0f;
    }
}

/* ======================================================================== */
/* observations                                                              */
/* ======================================================================== */

/* obs layout: 49 player + 12 pillar + 33*32 NPC + 5*8 pending hits = 1157 */
#define INF_PLAYER_OBS_SIZE 52   /* +3 for offensive prayer one-hot (piety/rigour/augury) */
#define INF_TOTAL_NPC_OBS_SIZE 282
#define INF_FEATURES_PER_HIT 5
#define INF_NUM_OBS (INF_PLAYER_OBS_SIZE + 12 + INF_TOTAL_NPC_OBS_SIZE + INF_FEATURES_PER_HIT * ENCOUNTER_MAX_PENDING_HITS)

/* max hit per NPC type, normalized by mager max (70). for prayer priority obs. */
static const float INF_NPC_MAX_HIT_NORM[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER]    = 0.0f,
    [INF_NPC_BAT]        = 19.0f / 70.0f,
    [INF_NPC_BLOB]       = 29.0f / 70.0f,
    [INF_NPC_BLOB_MELEE] = 18.0f / 70.0f,
    [INF_NPC_BLOB_RANGE] = 18.0f / 70.0f,
    [INF_NPC_BLOB_MAGE]  = 25.0f / 70.0f,
    [INF_NPC_MELEER]     = 49.0f / 70.0f,
    [INF_NPC_RANGER]     = 46.0f / 70.0f,
    [INF_NPC_MAGER]      = 70.0f / 70.0f,
    [INF_NPC_JAD]        = 113.0f / 70.0f,
    [INF_NPC_ZUK]        = 148.0f / 70.0f,
    [INF_NPC_HEALER_JAD] = 13.0f / 70.0f,
    [INF_NPC_HEALER_ZUK] = 24.0f / 70.0f,
    [INF_NPC_ZUK_SHIELD] = 0.0f,
};

static void inf_write_obs(EncounterState* state, float* obs) {
    InfernoState* s = (InfernoState*)state;
    memset(obs, 0, INF_NUM_OBS * sizeof(float));
    int i = 0;
    int px = s->player.x, py = s->player.y;
    const EncounterLoadoutStats* ls = inf_current_loadout_stats(s);

    /* player state (26 features) */
    obs[i++] = (float)s->player.current_hitpoints / 99.0f;
    obs[i++] = (float)(px - INF_ARENA_MIN_X) / (float)INF_ARENA_WIDTH;   /* dist to west wall */
    obs[i++] = (float)(INF_ARENA_MAX_X - px) / (float)INF_ARENA_WIDTH;  /* dist to east wall */
    obs[i++] = (float)(py - INF_ARENA_MIN_Y) / (float)INF_ARENA_HEIGHT; /* dist to south wall */
    obs[i++] = (float)(INF_ARENA_MAX_Y - py) / (float)INF_ARENA_HEIGHT; /* dist to north wall */
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    /* offensive prayer one-hot (none implied by all-three-zero). */
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_PIETY) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;
    obs[i++] = (float)s->player.brew_doses / 24.0f;
    obs[i++] = (float)s->player.restore_doses / 40.0f;
    obs[i++] = (float)s->player.current_prayer / 99.0f;
    obs[i++] = (float)s->wave / (float)INF_NUM_WAVES;
    /* tick normalization: Zuk-only (~300 ticks) vs full runs (~18000 ticks) */
    obs[i++] = 0.0f;
    //    (s->start_wave >= 68) ? (float)s->tick / 500.0f
    //                                 : (float)s->tick / (float)INF_MAX_TICKS;
    obs[i++] = (s->weapon_set == INF_GEAR_MAGE) ? 1.0f : 0.0f;
    obs[i++] = (s->weapon_set == INF_GEAR_TBOW) ? 1.0f : 0.0f;
    obs[i++] = (s->weapon_set == INF_GEAR_BP) ? 1.0f : 0.0f;
    obs[i++] = s->armor_tank ? 1.0f : 0.0f;
    obs[i++] = (float)s->player.bastion_doses / 8.0f;
    obs[i++] = (float)s->player.stamina_doses / 4.0f;
    obs[i++] = (s->stamina_active_ticks > 0) ? 1.0f : 0.0f;
    obs[i++] = (float)s->player.potion_timer / 3.0f;
    obs[i++] = (float)s->player.attack_timer / 8.0f;
    /* new: combat stats, target, weapon range, dead mob pool */
    obs[i++] = (float)s->player.current_defence / 99.0f;
    obs[i++] = (float)s->player.current_ranged / 99.0f;
    obs[i++] = (float)s->player.current_magic / 99.0f;
    obs[i++] = osrs_interaction_active(&s->interaction) ? 1.0f : 0.0f;
    obs[i++] = (float)ls->attack_range / 15.0f;
    obs[i++] = (float)s->dead_mob_count / (float)INF_MAX_DEAD_MOBS;
    /* gear stats: current loadout combat performance */
    obs[i++] = (float)ls->max_hit / 80.0f;
    obs[i++] = (float)ls->attack_speed / 6.0f;
    obs[i++] = (float)ls->def_stab / 300.0f;
    obs[i++] = (float)ls->def_magic / 300.0f;
    obs[i++] = (float)ls->def_ranged / 300.0f;
    obs[i++] = (float)s->player.special_energy / 100.0f;

    /* prayer-critical: distilled from NPC array and pending hits */
    {
        int min_timer = 999;
        int min_style = 0;
        int has_melee_2 = 0, has_ranged_2 = 0, has_magic_2 = 0;
        
        /* 1. Pending hits (handles Jad, which checks prayer on impact) */
        for (int h = 0; h < s->player_pending_hit_count; h++) {
            EncounterPendingHit* ph = &s->player_pending_hits[h];
            if (ph->check_prayer) {
                int t = inf_pending_hit_obs_timer(ph);
                if (t < min_timer) {
                    min_timer = t;
                    min_style = ph->attack_style;
                }
                if (t <= 2) {
                    if (ph->attack_style == ATTACK_STYLE_MELEE) has_melee_2 = 1;
                    if (ph->attack_style == ATTACK_STYLE_RANGED) has_ranged_2 = 1;
                    if (ph->attack_style == ATTACK_STYLE_MAGIC) has_magic_2 = 1;
                }
                if (t == 1) {
                                    }
            }
        }

        /* 2. NPCs firing or telegraphing (non-Jad checks prayer on launch;
           Jad enters here only once its committed preview style is visible). */
        for (int n = 0; n < INF_MAX_NPCS; n++) {
            InfNPC* npc = &s->npcs[n];
            if (!npc->active || npc->death_ticks > 0) continue;
            if (npc->type == INF_NPC_ZUK ||
                npc->type == INF_NPC_ZUK_SHIELD || npc->type == INF_NPC_NIBBLER || 
                npc->type == INF_NPC_HEALER_ZUK) continue;
                
            const InfNPCStats* st = &INF_NPC_STATS[npc->type];
            if (npc->frozen_ticks > 0 || npc->stun_timer > 0) continue;
            
            /* Relaxed check: if they can reach us or shoot us soon, track them. */
            int dist = encounter_dist_to_npc(s->player.x, s->player.y, npc->x, npc->y, npc->size);
            if (dist == 0) continue; /* Under us, usually can't attack or moves out */
            
            /* If they are completely out of range (more than 1 tile away from being able to attack), 
               we might ignore them, but to be safe we'll just track all aggroed NPCs that are off cooldown. 
               We'll just leave dist and LOS out of the strict filter. */

            int style = npc->attack_style;
            if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) {
                OverheadPrayer scanned = (OverheadPrayer)npc->blob_scanned_prayer;
                if (scanned == PRAYER_PROTECT_MAGIC) style = ATTACK_STYLE_RANGED;
                else if (scanned == PRAYER_PROTECT_RANGED) style = ATTACK_STYLE_MAGIC;
            } else if (npc->type == INF_NPC_JAD) {
                style = npc->jad_attack_style;
                if (style == ATTACK_STYLE_NONE) continue;
            }
            int style_mask = inf_attack_style_telegraph_mask(
                s, npc, st, style, dist);
            int preview_style = inf_attack_style_obs_preview(style_mask);

            int t = npc->attack_timer;
            if (t == 0) t = 1; /* Safety fallback if timer hit 0 */

            if (t < min_timer) {
                min_timer = t;
                min_style = preview_style;
            }
            if (t <= 2) {
                if (style_mask & INF_STYLE_MASK_MELEE) has_melee_2 = 1;
                if (style_mask & INF_STYLE_MASK_RANGED) has_ranged_2 = 1;
                if (style_mask & INF_STYLE_MASK_MAGIC) has_magic_2 = 1;
            }
                    }

        int conflict_count = has_melee_2 + has_ranged_2 + has_magic_2;
        obs[i++] = (min_timer < 999) ? (float)min_timer / 10.0f : 1.0f;
        obs[i++] = (min_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
        obs[i++] = (min_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
        obs[i++] = (min_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
        obs[i++] = (float)conflict_count / 3.0f;
    }

    /* Zuk-phase features (10 features: 1 flag + 9 Zuk-specific) */
    {
        int is_zuk = (s->wave == 68);
        obs[i++] = is_zuk ? 1.0f : 0.0f;

        if (is_zuk) {
            int si = s->zuk.shield_idx;
            int shield_active = (si >= 0 && s->npcs[si].active);

            /* shield direction/freeze are only meaningful while the shield exists.
               once the shield dies, zero these instead of leaking stale state. */
            obs[i++] = shield_active
                ? ((s->zuk.shield_freeze > 0) ? 0.0f : (float)s->zuk.shield_dir)
                : 0.0f;
            obs[i++] = shield_active ? (float)s->zuk.shield_freeze / 5.0f : 0.0f;
            /* am I behind the shield right now? + signed distance to shield center.
               the binary tells the agent if it's safe. the signed distance gives a
               gradient: negative = move east, positive = move west, 0 = centered. */
            int behind = 0;
            float shield_offset = 0.0f;
            if (shield_active) {
                int sx = s->npcs[si].x;
                int sz = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size;
                int shield_center = sx + sz / 2;
                shield_offset = (float)(px - shield_center) / 15.0f;  /* normalized, ~[-1,1] range */
                behind = (px >= sx && px < sx + sz && py >= 41);
            }
            obs[i++] = behind ? 1.0f : 0.0f;
            obs[i++] = shield_offset;
            /* Zuk enraged (attack speed 7 instead of 8) */
            obs[i++] = s->zuk.enraged ? 1.0f : 0.0f;
            /* set spawn timer / 350 */
            obs[i++] = (float)s->zuk.set_timer / 350.0f;
            /* set timer paused (Jad spawn phase) */
            obs[i++] = s->zuk.timer_paused ? 1.0f : 0.0f;
            /* Jad has spawned during Zuk fight */
            obs[i++] = s->zuk.jad_spawned ? 1.0f : 0.0f;
            /* Zuk healers have spawned */
            obs[i++] = s->zuk.healer_spawned ? 1.0f : 0.0f;
        } else {
            for (int z = 0; z < 9; z++) obs[i++] = 0.0f;
        }
    }

    /* pillars (12 features: active, hp, relative dx, relative dy per pillar) */
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        obs[i++] = s->pillars[p].active ? 1.0f : 0.0f;
        obs[i++] = (float)s->pillars[p].hp / (float)INF_PILLAR_HP;
        obs[i++] = (float)(s->pillars[p].x - px) / (float)INF_ARENA_WIDTH;
        obs[i++] = (float)(s->pillars[p].y - py) / (float)INF_ARENA_HEIGHT;
    }

    int obs_slots[INF_OBS_NPCS];
    for (int j = 0; j < INF_OBS_NPCS; j++) obs_slots[j] = -1;
    
    int slot_counts[INF_NUM_NPC_TYPES] = {0};
    int slot_offsets[INF_NUM_NPC_TYPES];
    int slot_max[INF_NUM_NPC_TYPES];
    
    slot_offsets[INF_NPC_MAGER] = 0; slot_max[INF_NPC_MAGER] = 2;
    slot_offsets[INF_NPC_RANGER] = 2; slot_max[INF_NPC_RANGER] = 2;
    slot_offsets[INF_NPC_MELEER] = 4; slot_max[INF_NPC_MELEER] = 2;
    slot_offsets[INF_NPC_BLOB] = 6; slot_max[INF_NPC_BLOB] = 2;
    slot_offsets[INF_NPC_BAT] = 8; slot_max[INF_NPC_BAT] = 2;
    slot_offsets[INF_NPC_BLOB_MAGE] = 10; slot_max[INF_NPC_BLOB_MAGE] = 2;
    slot_offsets[INF_NPC_BLOB_RANGE] = 12; slot_max[INF_NPC_BLOB_RANGE] = 2;
    slot_offsets[INF_NPC_BLOB_MELEE] = 14; slot_max[INF_NPC_BLOB_MELEE] = 2;
    slot_offsets[INF_NPC_NIBBLER] = 16; slot_max[INF_NPC_NIBBLER] = 6;
    slot_offsets[INF_NPC_JAD] = 22; slot_max[INF_NPC_JAD] = 3;
    slot_offsets[INF_NPC_ZUK] = 25; slot_max[INF_NPC_ZUK] = 1;
    slot_offsets[INF_NPC_ZUK_SHIELD] = 26; slot_max[INF_NPC_ZUK_SHIELD] = 1;
    slot_offsets[INF_NPC_HEALER_JAD] = 27; slot_max[INF_NPC_HEALER_JAD] = 6;
    slot_offsets[INF_NPC_HEALER_ZUK] = 33; slot_max[INF_NPC_HEALER_ZUK] = 4;

    for (int n = 0; n < INF_MAX_NPCS; n++) {
        InfNPC* npc = &s->npcs[n];
        if (npc->active && npc->death_ticks == 0) {
            int t = npc->type;
            if (slot_counts[t] < slot_max[t]) {
                obs_slots[slot_offsets[t] + slot_counts[t]] = n;
                slot_counts[t]++;
            }
        }
    }
    for (int j = 0; j < INF_OBS_NPCS; j++) {
        s->current_obs_slots[j] = obs_slots[j];
    }

    /* NPCs: variable features per slot, fixed order */
    int slot_types[INF_OBS_NPCS];
    int st_idx = 0;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_MAGER;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_RANGER;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_MELEER;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_BLOB;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_BAT;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_BLOB_MAGE;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_BLOB_RANGE;
    for (int j = 0; j < 2; j++) slot_types[st_idx++] = INF_NPC_BLOB_MELEE;
    for (int j = 0; j < 6; j++) slot_types[st_idx++] = INF_NPC_NIBBLER;
    for (int j = 0; j < 3; j++) slot_types[st_idx++] = INF_NPC_JAD;
    for (int j = 0; j < 1; j++) slot_types[st_idx++] = INF_NPC_ZUK;
    for (int j = 0; j < 1; j++) slot_types[st_idx++] = INF_NPC_ZUK_SHIELD;
    for (int j = 0; j < 6; j++) slot_types[st_idx++] = INF_NPC_HEALER_JAD;
    for (int j = 0; j < 4; j++) slot_types[st_idx++] = INF_NPC_HEALER_ZUK;

    for (int k = 0; k < INF_OBS_NPCS; k++) {
        int n = obs_slots[k];
        int type = slot_types[k];
        
        int has_style = (type == INF_NPC_BLOB || type == INF_NPC_JAD);
        int has_scan = (type == INF_NPC_BLOB);
        int has_los = (type != INF_NPC_NIBBLER && type != INF_NPC_MELEER && type != INF_NPC_HEALER_JAD && type != INF_NPC_ZUK_SHIELD);
        int has_aggro = (type != INF_NPC_NIBBLER && type != INF_NPC_ZUK_SHIELD);
        int has_timer = (type != INF_NPC_NIBBLER && type != INF_NPC_HEALER_JAD && type != INF_NPC_ZUK_SHIELD);
        int has_targeted = 1;
        
        int num_features = 4; // HP, RelX, RelY, AoE
        if (has_timer) num_features += 1;
        if (has_style) num_features += 3;
        if (has_los) num_features += 1;
        if (has_scan) num_features += 3;
        if (has_aggro) num_features += 1;
        if (has_targeted) num_features += 1;

        if (n >= 0) {
            InfNPC* npc = &s->npcs[n];
            obs[i++] = (float)npc->hp / (float)npc->max_hp;
            obs[i++] = (float)(npc->x - px) / (float)INF_ARENA_WIDTH;
            obs[i++] = (float)(npc->y - py) / (float)INF_ARENA_HEIGHT;
            if (has_timer) obs[i++] = (float)npc->attack_timer / 10.0f;
            
            if (has_style) {
                int style = (npc->type == INF_NPC_JAD) ? npc->jad_attack_style : npc->attack_style;
                obs[i++] = (style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
                obs[i++] = (style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
                obs[i++] = (style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
            }
            
            if (has_los) obs[i++] = inf_npc_has_los(s, n) ? 1.0f : 0.0f;
            
            if (has_scan) {
                if (npc->blob_scanned_prayer >= 0) {
                    OverheadPrayer scanned = (OverheadPrayer)npc->blob_scanned_prayer;
                    obs[i++] = (scanned == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
                    obs[i++] = (scanned == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
                    obs[i++] = (scanned != PRAYER_PROTECT_MAGIC && scanned != PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
                } else {
                    obs[i++] = 0.0f; obs[i++] = 0.0f; obs[i++] = 0.0f;
                }
            }
            
            /* barrage AoE count: unique blocking NPCs in the 3x3 area */
            {
                int aoe_count = 0;
                for (int oidx = 0; oidx < INF_MAX_NPCS; oidx++) {
                    if (oidx == n) continue;
                    InfNPC* other = &s->npcs[oidx];
                    if (!other->active) continue;
                    if (!inf_npc_sets_collision_flag(other->type)) continue;
                    if (encounter_entity_footprints_overlap(
                            other->x, other->y, inf_npc_effective_size(other),
                            npc->x - 1, npc->y - 1, 3)) {
                        aoe_count++;
                    }
                }
                obs[i++] = (float)aoe_count / 8.0f;
            }
            
            if (has_aggro) obs[i++] = (npc->aggro_target < 0) ? 1.0f : 0.0f;
            if (has_targeted) obs[i++] = (osrs_interaction_active(&s->interaction) && s->interaction.target_slot == n) ? 1.0f : 0.0f;
        } else {
            for (int j = 0; j < num_features; j++) obs[i++] = 0.0f;
        }
    }

    /* assert NPC section wrote exactly the right number of features.
       if this fires, INF_FEATURES_PER_NPC doesn't match the actual feature count. */
    {
        int expected_npc_end = INF_PLAYER_OBS_SIZE + 12 + INF_TOTAL_NPC_OBS_SIZE;
        if (i != expected_npc_end) {
            fprintf(stderr, "FATAL: obs misaligned after NPC section: i=%d expected=%d\n",
                    i, expected_npc_end);
            abort();
        }
    }

    /* pending hits on player (INF_FEATURES_PER_HIT * ENCOUNTER_MAX_PENDING_HITS) */
    for (int h = 0; h < ENCOUNTER_MAX_PENDING_HITS; h++) {
        if (h < s->player_pending_hit_count) {
            EncounterPendingHit* ph = &s->player_pending_hits[h];
            obs[i++] = 1.0f;  /* active */
            obs[i++] = (ph->attack_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
            obs[i++] = (ph->attack_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
            obs[i++] = (float)inf_pending_hit_obs_timer(ph) / 10.0f;
            obs[i++] = (float)ph->damage / 150.0f;  /* normalized damage magnitude (Zuk max ~148) */
        } else {
            for (int j = 0; j < INF_FEATURES_PER_HIT; j++) obs[i++] = 0.0f;
        }
    }

    /* sanity: verify we wrote exactly INF_NUM_OBS features */
    if (i != INF_NUM_OBS) {
        fprintf(stderr, "BUG: inf_write_obs wrote %d features, expected %d\n", i, INF_NUM_OBS);
        abort();
    }
}

static int inf_find_target_obs_slot(const InfernoState* s, int npc_slot) {
    if (npc_slot < 0 || npc_slot >= INF_MAX_NPCS) return -1;
    for (int j = 0; j < INF_OBS_NPCS; j++) {
        if (s->current_obs_slots[j] == npc_slot) return j;
    }
    return -1;
}

static int inf_obs_slot_is_targetable(const InfernoState* s, int obs_slot) {
    if (obs_slot < 0 || obs_slot >= INF_OBS_NPCS) return 0;
    int npc_slot = s->current_obs_slots[obs_slot];
    if (npc_slot < 0 || npc_slot >= INF_MAX_NPCS) return 0;
    return s->npcs[npc_slot].active &&
        s->npcs[npc_slot].death_ticks == 0 &&
        s->npcs[npc_slot].type != INF_NPC_ZUK_SHIELD;
}

static int inf_is_human_targetable_npc_slot(EncounterState* state, int npc_slot) {
    const InfernoState* s = (const InfernoState*)state;
    return inf_obs_slot_is_targetable(s, inf_find_target_obs_slot(s, npc_slot));
}

static void inf_write_mask(EncounterState* state, float* mask) {
    InfernoState* s = (InfernoState*)state;
    int offset = 0;

    /* HEAD_MOVE (25): idle always valid, walk/run valid if target tile reachable */
    mask[offset++] = 1.0f;  /* idle always valid */
    for (int d = 1; d < ENCOUNTER_MOVE_ACTIONS; d++) {
        int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[d];
        int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[d];
        mask[offset++] = (inf_in_arena(nx, ny) && !inf_blocked_by_pillar(s, nx, ny, 1))
                         ? 1.0f : 0.0f;
    }

    /* HEAD_PRAYER (4): 0=no_change always valid, 1-3=toggle_melee/ranged/magic.
       toggles are always valid (the env decides on/off based on current state)
       but require pp > 0 to activate; mask activation paths when pp=0.
       since the agent can't know if the toggle will activate vs deactivate without
       additional state, we keep toggles available when pp>0 and when the matching
       prayer is already active (toggle-off is free). */
    mask[offset++] = 1.0f;  /* no_change always valid */
    mask[offset++] = (s->player.current_prayer > 0 || s->player.prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    mask[offset++] = (s->player.current_prayer > 0 || s->player.prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    mask[offset++] = (s->player.current_prayer > 0 || s->player.prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;

    /* HEAD_TARGET (INF_OBS_NPCS+1): none always valid, slot valid only if its
       mapped NPC is alive (not dying) and not the Zuk shield (invulnerable). */
    mask[offset++] = 1.0f;  /* no target */
    for (int n = 0; n < INF_OBS_NPCS; n++) {
        mask[offset++] = inf_obs_slot_is_targetable(s, n) ? 1.0f : 0.0f;
    }

    /* HEAD_GEAR (5): no_switch, mage, tbow, bp, tank */
    mask[offset++] = 1.0f;  /* no_switch always valid */
    mask[offset++] = (s->weapon_set != INF_GEAR_MAGE) ? 1.0f : 0.0f;
    mask[offset++] = (s->weapon_set != INF_GEAR_TBOW) ? 1.0f : 0.0f;
    mask[offset++] = (s->weapon_set != INF_GEAR_BP) ? 1.0f : 0.0f;
    mask[offset++] = 0.0f;  /* tank overlay removed */

    /* HEAD_EAT (2): none, brew */
    mask[offset++] = 1.0f;  /* none always valid */
    mask[offset++] = (s->player.brew_doses > 0 &&
                      s->player.potion_timer == 0 &&
                      s->player.current_hitpoints < s->player.base_hitpoints)
                     ? 1.0f : 0.0f;

    /* HEAD_POTION (4): none, restore, bastion, stamina */
    mask[offset++] = 1.0f;  /* none always valid */
    /* restore: unmask if any stat is drained or prayer is low enough to not waste.
       "stats drained" = any combat stat below base 99. */
    {
        int pray_missing = s->player.base_prayer - s->player.current_prayer;
        int stats_drained = s->player.current_attack < 99 || s->player.current_strength < 99 ||
                            s->player.current_defence < 99 || s->player.current_ranged < 99 ||
                            s->player.current_magic < 99;
        int pray_worth = pray_missing >= (INF_RESTORE_AMOUNT + 1) / 2;
        mask[offset++] = (s->player.restore_doses > 0 &&
                          s->player.potion_timer == 0 &&
                          (stats_drained || pray_worth))
                         ? 1.0f : 0.0f;
    }
    /* bastion: only worth drinking at 99-105 ranged (drained = restore first, >105 = still boosted) */
    mask[offset++] = (s->player.bastion_doses > 0 && s->player.potion_timer == 0 &&
                      s->player.current_ranged >= 99 && s->player.current_ranged <= 105)
                     ? 1.0f : 0.0f;
    /* stamina: mask if no doses, timer active, or already active */
    mask[offset++] = (s->player.stamina_doses > 0 &&
                      s->player.potion_timer == 0 &&
                      s->stamina_active_ticks == 0)
                     ? 1.0f : 0.0f;

    /* HEAD_SPELL (3): no_change, blood_barrage, ice_barrage.
       noop always valid. blood masked at full HP or if magic level is too low
       to cast blood barrage (req 92). ice masked if magic level too low (req 94).
       both spells masked when not in mage gear. */
    mask[offset++] = 1.0f;  /* no_change always valid */
    mask[offset++] = (s->weapon_set == INF_GEAR_MAGE &&
                      s->player.current_magic >= BLOOD_BARRAGE_LEVEL &&
                      s->player.current_hitpoints < s->player.base_hitpoints) ? 1.0f : 0.0f;
    mask[offset++] = (s->weapon_set == INF_GEAR_MAGE &&
                      s->player.current_magic >= ICE_BARRAGE_LEVEL) ? 1.0f : 0.0f;

    /* HEAD_SPEC (2): no_change, toggle. allow when blowpipe equipped + enough energy. */
    mask[offset++] = 1.0f;  /* no_change always valid */
    mask[offset++] = (s->weapon_set == INF_GEAR_BP &&
                      s->player.special_energy >= BLOWPIPE_SPEC_COST)
                     ? 1.0f : 0.0f;

    /* HEAD_OFFENSIVE (4): 0=no_change always valid, 1-3=toggle_piety/rigour/augury.
       same rule as overhead — toggles need pp>0 to activate, free to deactivate. */
    mask[offset++] = 1.0f;  /* no_change always valid */
    mask[offset++] = (s->player.current_prayer > 0 || s->player.offensive_prayer == OFFENSIVE_PRAYER_PIETY)  ? 1.0f : 0.0f;
    mask[offset++] = (s->player.current_prayer > 0 || s->player.offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    mask[offset++] = (s->player.current_prayer > 0 || s->player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;
}

/* ======================================================================== */
/* query functions                                                           */
/* ======================================================================== */

static float inf_get_reward(EncounterState* state) {
    return ((InfernoState*)state)->reward;
}

static int inf_is_terminal(EncounterState* state) {
    return ((InfernoState*)state)->episode_over;
}

static int inf_get_entity_count(EncounterState* state) {
    InfernoState* s = (InfernoState*)state;
    int count = 1;
    for (int i = 0; i < INF_MAX_NPCS; i++)
        if (s->npcs[i].active) count++;
    return count;
}

static void* inf_get_entity(EncounterState* state, int index) {
    InfernoState* s = (InfernoState*)state;
    /* only index 0 (player) returns a valid Player*.
     * NPC indices can't return Player* since InfNPC is a different struct.
     * GUI/human input code must NULL-check. */
    if (index == 0) return &s->player;
    return NULL;
}

/* render entity population */
static void inf_fill_render_entities(EncounterState* state, RenderEntity* out, int max_entities, int* count) {
    InfernoState* s = (InfernoState*)state;
    int n = 0;

    {
        const EncounterLoadoutStats* ls = inf_current_loadout_stats(s);
        s->player.gui_max_hit = ls->max_hit;
        s->player.gui_attack_speed = ls->attack_speed;
        s->player.gui_attack_range = ls->attack_range;
        s->player.gui_strength_bonus = ls->strength_bonus;
    }

    /* index 0: the player */
    if (n < max_entities) {
        render_entity_from_player(&s->player, &out[n++]);
    }

    /* active NPCs: manually fill since InfNPC is not a Player */
    for (int i = 0; i < INF_MAX_NPCS && n < max_entities; i++) {
        InfNPC* npc = &s->npcs[i];
        if (!npc->active) continue;

        RenderEntity* re = &out[n++];
        memset(re, 0, sizeof(RenderEntity));
        memset(re->equipped, ITEM_NONE, NUM_GEAR_SLOTS);
        re->entity_type = ENTITY_NPC;
        re->npc_def_id = INF_NPC_DEF_IDS[npc->type];
        re->npc_slot = i;
        /* facing: nibblers → pillar (dest-based), NPCs attacking shield → shield
           (dest-based), all others → player (entity 0). */
        if (npc->type == INF_NPC_NIBBLER) {
            re->attack_target_entity_idx = -1;
        } else if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
                   s->npcs[npc->aggro_target].active) {
            /* attacking another NPC (e.g. shield) — use dest-based facing */
            re->attack_target_entity_idx = -1;
        } else {
            re->attack_target_entity_idx = 0;  /* player */
        }
        re->npc_visible = npc->active;
        re->npc_size = npc->size;
        {
            const NpcModelMapping* nm = npc_model_lookup(INF_NPC_DEF_IDS[npc->type]);
            if (npc->death_ticks > 0) {
                /* dying: hold idle pose while hitsplat + health bar display */
                re->npc_anim_id = nm ? (int)nm->idle_anim : -1;
            } else if (npc->type == INF_NPC_MELEER &&
                       npc->dig_freeze_timer == 6) {
                re->npc_anim_id = INF_GEN_ANIM_MELEER_DIG_DOWN;
            } else if (npc->type == INF_NPC_MELEER &&
                       npc->dig_attack_delay == 6) {
                re->npc_anim_id = INF_GEN_ANIM_MELEER_DIG_UP;
            } else if (npc->attacked_this_tick) {
                re->npc_anim_id = inf_npc_attack_anim_id(npc, nm);
            } else {
                /* walk/idle handled by secondary track in render_client_tick.
                   setting walk as primary causes stall (interleave_count==0)
                   which freezes movement and creates tile-to-tile teleporting. */
                re->npc_anim_id = -1;
            }
        }
        re->x = npc->x;
        re->y = npc->y;
        /* nibblers: set dest to pillar center so renderer faces them toward
           the pillar instead of the player when idle/attacking */
        if (npc->type == INF_NPC_NIBBLER) {
            int tp = s->nibbler_target_pillar;
            if (tp >= 0 && tp < INF_NUM_PILLARS && s->pillars[tp].active) {
                re->dest_x = s->pillars[tp].x + INF_PILLAR_SIZE / 2;
                re->dest_y = s->pillars[tp].y + INF_PILLAR_SIZE / 2;
            } else {
                re->dest_x = npc->x;
                re->dest_y = npc->y;
            }
        } else if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
                   s->npcs[npc->aggro_target].active) {
            /* attacking shield/other NPC — face toward target NPC center */
            InfNPC* at = &s->npcs[npc->aggro_target];
            re->dest_x = at->x + at->size / 2;
            re->dest_y = at->y + at->size / 2;
        } else {
            re->dest_x = npc->target_x;
            re->dest_y = npc->target_y;
        }
        re->current_hitpoints = npc->hp;
        re->base_hitpoints = npc->max_hp;
        re->attack_style_this_tick = npc->attacked_this_tick
            ? (AttackStyle)npc->attack_style_this_tick : ATTACK_STYLE_NONE;
        re->hit_landed_this_tick = npc->hit_landed_this_tick;
        re->hit_damage = npc->hit_damage;
        /* barrage hits that pass accuracy are queued; splashes never enter the queue.
           so any NPC with hit_landed_this_tick from a pending hit was a successful hit. */
        re->hit_was_successful = npc->hit_landed_this_tick;
        re->hit_spell_type = npc->hit_spell_type;
    }

    encounter_resolve_attack_target(out, n, s->interaction.target_slot);
    *count = n;
}

static void inf_put_int(EncounterState* state, const char* key, int value) {
    InfernoState* s = (InfernoState*)state;
    /* wave is 1-indexed externally (wave 1 = first, wave 69 = Zuk), 0-indexed internally */
    if (strcmp(key, "start_wave") == 0) s->start_wave = (value > 0) ? value - 1 : 0;
    else if (strcmp(key, "seed") == 0) s->rng_state = (uint32_t)value;
    else if (strcmp(key, "world_offset_x") == 0) s->world_offset_x = value;
    else if (strcmp(key, "world_offset_y") == 0) s->world_offset_y = value;
    else if (strcmp(key, "player_dest_x") == 0) s->player_dest_x = value;
    else if (strcmp(key, "player_dest_y") == 0) s->player_dest_y = value;
    else if (strcmp(key, "human_command_mode") == 0) s->human_command_mode = value;
}

static void inf_put_float(EncounterState* state, const char* key, float value) {
    InfernoState* s = (InfernoState*)state;
    if (strcmp(key, "damage_reward_coeff") == 0) s->damage_reward_coeff = value;
    else if (strcmp(key, "shield_penalty_coeff") == 0) s->shield_penalty_coeff = value;
    else if (strcmp(key, "tag_reward_coeff") == 0) s->tag_reward_coeff = value;
    else assert(0 && "unknown inferno float config");
}

static void inf_put_ptr(EncounterState* state, const char* key, void* value) {
    InfernoState* s = (InfernoState*)state;
    if (strcmp(key, "collision_map") == 0) s->collision_map = (const CollisionMap*)value;
}

static int inf_get_tick(EncounterState* state) {
    return ((InfernoState*)state)->tick;
}

static int inf_get_winner(EncounterState* state) {
    return ((InfernoState*)state)->winner;
}

static void* inf_get_log(EncounterState* state) {
    InfernoState* s = (InfernoState*)state;
    if (s->episode_over) {
        s->log.episode_return += s->episode_return;
        s->log.episode_length += (float)s->tick;
        s->log.wins += (s->winner == 0) ? 1.0f : 0.0f;
        s->log.damage_dealt += s->total_damage_dealt;
        s->log.zuk_healer_damage += s->total_zuk_healer_damage;
        s->log.damage_received += s->total_damage_received;
        s->log.wave += (float)s->wave;
        s->log.prayer_correct += (float)s->total_prayer_correct;
        s->log.prayer_total += (float)s->total_npc_attacks;
        s->log.idle_ticks += (float)s->total_idle_ticks;
        s->log.brews_used += (float)s->total_brews_used;
        s->log.blood_healed += (float)s->total_blood_healed;
        s->log.n += 1.0f;
        s->log.npc_kills += (float)s->total_npc_kills;
        s->log.gear_switches += (float)s->total_gear_switches;
        s->log.current_ranged += (float)s->player.current_ranged;
        s->log.current_magic += (float)s->player.current_magic;
        s->log.min_zuk_hp_seen += (s->winner == 0)
            ? 0.0f
            : (s->min_zuk_hp_seen > 0.0f ? s->min_zuk_hp_seen : 1200.0f);
    }
    return &s->log;
}

/* ======================================================================== */
/* render post-tick: populate overlay projectiles for renderer               */
/* ======================================================================== */


static void inf_render_post_tick(EncounterState* state, EncounterOverlay* ov) {
    InfernoState* s = (InfernoState*)state;
    ov->projectile_count = 0;

    /* NPC attack projectiles — per-NPC-type flight parameters */
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        InfNPC* npc = &s->npcs[i];
        if (!npc->active || !npc->attacked_this_tick) continue;
        if (ov->projectile_count >= ENCOUNTER_MAX_OVERLAY_PROJECTILES) break;

        /* nibblers attack pillars, not worth showing as projectile */
        if (npc->type == INF_NPC_NIBBLER) continue;

        /* blob scan animation (no projectile) — only emit on the actual fire tick.
           blob_scanned_prayer >= 0 means scan just happened, -1 means fire. */
        if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) continue;

        const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
        int actual_style = npc->attack_style_this_tick;

        /* Zuk is typeless — show as magic for visual purposes. */
        if (actual_style == ATTACK_STYLE_NONE && npc->type == INF_NPC_ZUK)
            actual_style = ATTACK_STYLE_MAGIC;

        /* tagged Zuk healers spawn their own 3-spark visuals from pending_sparks. */
        if (npc->type == INF_NPC_HEALER_ZUK && npc->attack_visual_target < 0)
            continue;

        if (actual_style == ATTACK_STYLE_NONE) continue;

        /* melee attacks are instant — no in-flight projectile */
        if (actual_style == ATTACK_STYLE_MELEE) continue;

        int proj_style = encounter_attack_style_to_proj_style(actual_style);
        int npc_size = stats->size;
        int start_h = (int)(npc_size * 0.75f * 128);
        int end_h = 64;  /* default: player size 1 * 0.5 * 128 */
        int curve = 16;
        float arc = 0.0f;
        int tracks = 1;

        /* projectile target: shield or player */
        int target_x = s->player.x, target_y = s->player.y;
        if (npc->attack_visual_target >= 0 && npc->attack_visual_target < INF_MAX_NPCS) {
            InfNPC* vt = &s->npcs[npc->attack_visual_target];
            target_x = vt->x + vt->size / 2;
            target_y = vt->y + vt->size / 2;
            end_h = (int)(vt->size * 0.5f * 128);
            tracks = 0;  /* don't track player — projectile targets shield/NPC */
        }
        int dist = encounter_dist_to_npc(target_x, target_y,
            npc->x, npc->y, npc_size);
        int hit_delay;
        if (actual_style == ATTACK_STYLE_MAGIC)
            hit_delay = encounter_magic_hit_delay(dist, 0);
        else
            hit_delay = encounter_ranged_hit_delay(dist, 0);
        int duration = hit_delay * 30;

        /* per-NPC-type projectile GFX model ID */
        uint32_t proj_model_id = 0;
        switch (npc->type) {
            case INF_NPC_BAT:        proj_model_id = INF_GFX_1374_MODEL; break;
            case INF_NPC_BLOB:       proj_model_id = (actual_style == ATTACK_STYLE_RANGED) ? INF_GFX_1378_MODEL : INF_GFX_1380_MODEL; break;
            case INF_NPC_BLOB_RANGE: proj_model_id = INF_GFX_1379_MODEL; break;
            case INF_NPC_BLOB_MAGE:  proj_model_id = INF_GFX_1381_MODEL; break;
            case INF_NPC_BLOB_MELEE: proj_model_id = INF_GFX_1382_MODEL; break;
            case INF_NPC_RANGER:     proj_model_id = INF_GFX_1377_MODEL; break;
            case INF_NPC_MAGER:      proj_model_id = INF_GFX_1376_MODEL; break;
            case INF_NPC_JAD:
                proj_model_id = (actual_style == ATTACK_STYLE_MAGIC)
                    ? INF_GFX_448_MODEL
                    : INF_GFX_451_MODEL;
                break;
            case INF_NPC_ZUK:        proj_model_id = INF_GFX_1375_MODEL; break;
            case INF_NPC_HEALER_ZUK: proj_model_id = INF_GFX_660_MODEL; break;
            default: break;
        }

        /* NPC-specific flight overrides */
        switch (npc->type) {
            case INF_NPC_MAGER:
                /* InfernoTrainer JalZek MagicWeapon: visualDelayTicks=2,
                   visualHitEarlyTicks=-1. projectile invisible for 2 ticks,
                   then visual flies in and arrives 1 tick AFTER the hit lands.
                   visible duration = (hit_delay + 1) - 2 = hit_delay - 1 ticks.
                   start_delay=2 ticks is set after the emit below. */
                duration = (hit_delay - 1) * 30;
                if (duration < 30) duration = 30;
                break;
            case INF_NPC_RANGER:
                /* InfernoTrainer JalXil RangedWeapon: reduceDelay=-2 (hit
                   delay +2 ticks), visualDelayTicks=3. visible duration =
                   (hit_delay + 2) - 3 = hit_delay - 1 ticks. start_delay=3
                   ticks is set after the emit below. NOTE: sim-side hit delay
                   is currently NOT adjusted by +2 — damage lands 2 ticks
                   earlier than reference. */
                duration = (hit_delay - 1) * 30;
                if (duration < 30) duration = 30;
                break;
            case INF_NPC_JAD:
                if (actual_style == ATTACK_STYLE_MAGIC) {
                    arc = 1.0f;  /* arcing magic projectile */
                } else {
                    start_h = end_h;
                }
                duration = inf_jad_visible_duration_ticks(hit_delay);
                break;
            case INF_NPC_HEALER_ZUK:
                arc = 3.0f;      /* high arcing spark */
                duration = (npc->attack_visual_target >= 0) ? 3 * 30 : 4 * 30;
                break;
            case INF_NPC_ZUK:
                /* InfernoTrainer: setDelay=4, visualDelayTicks=2.
                   projectile invisible for 2 ticks, visible for 2 ticks. */
                duration = 2 * 30;  /* 2-tick visible flight */
                break;
            default: break;
        }

        if (npc->type == INF_NPC_JAD && actual_style == ATTACK_STYLE_MAGIC) {
            if (ov->projectile_count + 3 > ENCOUNTER_MAX_OVERLAY_PROJECTILES) break;
            uint32_t model_ids[3] = {
                INF_GFX_448_MODEL, INF_GFX_449_MODEL, INF_GFX_450_MODEL
            };
            int anim_ids[3] = {
                INF_GFX_448_ANIM, INF_GFX_449_ANIM, INF_GFX_450_ANIM
            };
            float offsets[3] = {1.0f, 0.5f, 0.0f};
            for (int j = 0; j < 3; j++) {
                int pi = encounter_emit_projectile(ov,
                    npc->x, npc->y, target_x, target_y,
                    proj_style, (int)s->damage_received_this_tick,
                    duration, start_h, end_h, curve, arc, tracks, npc_size, 1,
                    model_ids[j], 0);
                if (pi >= 0) {
                    ov->projectiles[pi].start_delay = INF_JAD_PROJECTILE_DELAY * 30;
                    encounter_set_projectile_animation(ov, pi, anim_ids[j]);
                    encounter_set_projectile_offset(ov, pi, 0.0f, offsets[j], 0.0f);
                }
            }
            continue;
        }

        int impact_gfx_id = (npc->type == INF_NPC_HEALER_ZUK) ? INF_GFX_659_ID : 0;
        int pi = encounter_emit_projectile(ov,
            npc->x, npc->y, target_x, target_y,
            proj_style, (int)s->damage_received_this_tick,
            duration, start_h, end_h, curve, arc, tracks, npc_size, 1,
            proj_model_id, impact_gfx_id);

        /* Zuk: 2-tick visual delay (projectile invisible until tick N+2) */
        if (pi >= 0 && npc->type == INF_NPC_ZUK)
            ov->projectiles[pi].start_delay = 2 * 30;

        /* Jad: 3-tick visual delay (InfernoTrainer JAD_PROJECTILE_DELAY=3) */
        if (pi >= 0 && npc->type == INF_NPC_JAD)
            ov->projectiles[pi].start_delay = INF_JAD_PROJECTILE_DELAY * 30;

        if (pi >= 0 && npc->type == INF_NPC_JAD &&
            actual_style == ATTACK_STYLE_RANGED)
            encounter_set_projectile_motion_mode(
                ov, pi, ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED);

        if (pi >= 0 && npc->type == INF_NPC_JAD &&
            actual_style == ATTACK_STYLE_RANGED)
            encounter_set_projectile_animation(ov, pi, INF_GFX_451_ANIM);

        /* Mager: 2-tick visualDelayTicks (InfernoTrainer JalZek MagicWeapon) */
        if (pi >= 0 && npc->type == INF_NPC_MAGER)
            ov->projectiles[pi].start_delay = 2 * 30;

        /* Ranger: 3-tick visualDelayTicks (InfernoTrainer JalXil RangedWeapon) */
        if (pi >= 0 && npc->type == INF_NPC_RANGER)
            ov->projectiles[pi].start_delay = 3 * 30;
    }

    for (int i = 0; i < INF_MAX_PENDING_SPARKS; i++) {
        InfPendingSpark* spark = &s->pending_sparks[i];
        if (!spark->active || spark->visual_emitted) continue;
        if (ov->projectile_count >= ENCOUNTER_MAX_OVERLAY_PROJECTILES) break;

        encounter_emit_projectile(
            ov,
            spark->src_x, spark->src_y, spark->x, spark->y,
            encounter_attack_style_to_proj_style(ATTACK_STYLE_MAGIC),
            spark->damage,
            4 * 30, 96, 64, 16, 3.0f, 0, 1, 1,
            INF_GFX_660_MODEL, INF_GFX_659_ID);
        spark->visual_emitted = 1;
    }

    /* player attack projectile (ranged/magic only — melee has no projectile) */
    if (s->player_attacked_this_tick &&
        s->player_attack_style_id != ATTACK_STYLE_MELEE &&
        ov->projectile_count < ENCOUNTER_MAX_OVERLAY_PROJECTILES) {
        int target_idx = s->player_attack_npc_idx;
        if (target_idx >= 0 && target_idx < INF_MAX_NPCS) {
            InfNPC* target = &s->npcs[target_idx];
            int target_size = INF_NPC_STATS[target->type].size;
            int p_start_h = 64;   /* player: size 1 * 0.5 * 128 */
            int p_end_h = (int)(target_size * 0.5f * 128);
            int p_dist = encounter_dist_to_npc(s->player.x, s->player.y,
                target->x, target->y, target_size);
            int p_style = encounter_attack_style_to_proj_style(s->player_attack_style_id);
            float p_arc = 0.0f;
            int p_tracks = 0;  /* don't track — tracking loop targets entity 0 (player) */
            int p_duration;
            uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];

            uint32_t player_proj_model = 0;
            if (s->player_attack_style_id == ATTACK_STYLE_MAGIC) {
                p_duration = encounter_magic_hit_delay(p_dist, 1) * 30;
                p_arc = 0.0f;
                /* barrage: no projectile model (effect system handles it) */
            } else if (weapon == ITEM_TWISTED_BOW) {
                p_duration = encounter_ranged_hit_delay(p_dist, 1) * 30;
                p_arc = 1.0f;
                player_proj_model = INF_GFX_1120_MODEL;
            } else {
                /* blowpipe */
                p_duration = encounter_blowpipe_hit_delay(p_dist, 1) * 30;
                p_arc = 0.5f;
                player_proj_model = 26379;  /* dragon dart */
            }

            /* barrage has no in-flight projectile in OSRS (only hit splash) */
            if (player_proj_model > 0) {
                encounter_emit_projectile(ov,
                    s->player.x, s->player.y, target->x, target->y,
                    p_style, s->player_attack_dmg,
                    p_duration, p_start_h, p_end_h, 16, p_arc, p_tracks,
                    1, target_size, player_proj_model, 0);
            }
        }
    }
}

/* ======================================================================== */
/* human input translator                                                    */
/* ======================================================================== */

static void* inf_get_player_for_input(void* state, int idx) {
    InfernoState* s = (InfernoState*)state;
    return (idx == 0) ? (void*)&s->player : NULL;
}

static void inf_translate_human_input(HumanInput* hi, int* actions, EncounterState* state) {
    for (int h = 0; h < INF_NUM_ACTION_HEADS; h++) actions[h] = 0;

    encounter_translate_movement(hi, actions, INF_HEAD_MOVE, inf_get_player_for_input, state);
    encounter_translate_prayer(hi, actions, INF_HEAD_PRAYER);
    encounter_translate_offensive_prayer(hi, actions, INF_HEAD_OFFENSIVE);
    InfernoState* s = (InfernoState*)state;
    /* map raw pending_target_idx to the observation slot the agent sees */
    if (hi->pending_target_idx >= 0) {
        int found_slot = inf_find_target_obs_slot(s, hi->pending_target_idx);
        if (inf_obs_slot_is_targetable(s, found_slot)) {
            actions[INF_HEAD_TARGET] = found_slot + 1;
        } else {
            actions[INF_HEAD_TARGET] = 0;
        }
    } else {
        actions[INF_HEAD_TARGET] = 0;
    }

    /* gear switch */
    if (hi->pending_gear > 0) actions[INF_HEAD_GEAR] = hi->pending_gear;

    /* eat: brew */
    if (hi->pending_food || hi->pending_potion == POTION_BREW)
        actions[INF_HEAD_EAT] = 1;

    /* potions: restore=1, bastion=2, stamina=3 */
    if (hi->pending_potion == POTION_RESTORE) actions[INF_HEAD_POTION] = 1;
    else if (hi->pending_potion == POTION_BASTION) actions[INF_HEAD_POTION] = 2;
    else if (hi->pending_potion == POTION_STAMINA) actions[INF_HEAD_POTION] = 3;

    /* spell: 0=no change, 1=blood, 2=ice */
    if (hi->pending_spell == ATTACK_BLOOD) actions[INF_HEAD_SPELL] = 1;
    else if (hi->pending_spell == ATTACK_ICE) actions[INF_HEAD_SPELL] = 2;

    /* spec */
    if (hi->pending_spec) actions[INF_HEAD_SPEC] = 1;
}

static void inf_translate_human_commands(HumanInput* hi, int* actions, InfernoState* s) {
    for (int h = 0; h < INF_NUM_ACTION_HEADS; h++) actions[h] = 0;

    for (int i = 0; i < hi->commands.count; i++) {
        const HumanCommand* cmd = &hi->commands.items[i];
        switch (cmd->kind) {
            case HUMAN_COMMAND_WALK:
                s->player_dest_x = cmd->world_x;
                s->player_dest_y = cmd->world_y;
                actions[INF_HEAD_TARGET] = 0;
                actions[INF_HEAD_SPELL] = 0;
                break;
            case HUMAN_COMMAND_ATTACK_NPC: {
                int found_slot = inf_find_target_obs_slot(s, cmd->npc_slot);
                actions[INF_HEAD_TARGET] = inf_obs_slot_is_targetable(s, found_slot)
                    ? found_slot + 1 : 0;
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                break;
            }
            case HUMAN_COMMAND_SPELL_TARGET: {
                int found_slot = inf_find_target_obs_slot(s, cmd->npc_slot);
                actions[INF_HEAD_TARGET] = inf_obs_slot_is_targetable(s, found_slot)
                    ? found_slot + 1 : 0;
                if (cmd->spell == ATTACK_BLOOD) actions[INF_HEAD_SPELL] = 1;
                else if (cmd->spell == ATTACK_ICE) actions[INF_HEAD_SPELL] = 2;
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                break;
            }
            case HUMAN_COMMAND_OVERHEAD_PRAYER:
                actions[INF_HEAD_PRAYER] = cmd->overhead_prayer;
                break;
            case HUMAN_COMMAND_OFFENSIVE_PRAYER:
                actions[INF_HEAD_OFFENSIVE] = cmd->offensive_prayer;
                break;
            case HUMAN_COMMAND_EAT:
                actions[INF_HEAD_EAT] = 1;
                break;
            case HUMAN_COMMAND_DRINK:
                if (cmd->potion == POTION_BREW) actions[INF_HEAD_EAT] = 1;
                else if (cmd->potion == POTION_RESTORE) actions[INF_HEAD_POTION] = 1;
                else if (cmd->potion == POTION_BASTION) actions[INF_HEAD_POTION] = 2;
                else if (cmd->potion == POTION_STAMINA) actions[INF_HEAD_POTION] = 3;
                break;
            case HUMAN_COMMAND_SPEC_TOGGLE:
                actions[INF_HEAD_SPEC] = 1;
                break;
            case HUMAN_COMMAND_EQUIP_INVENTORY_ITEM:
            case HUMAN_COMMAND_FIGHT_STYLE:
            case HUMAN_COMMAND_NONE:
                break;
        }
    }
}

static void inf_step_human_commands(EncounterState* state, HumanInput* hi) {
    InfernoState* s = (InfernoState*)state;
    int actions[INF_NUM_ACTION_HEADS];
    s->human_command_mode = 1;
    s->human_commands = hi->commands.items;
    s->human_command_count = hi->commands.count;
    inf_refresh_human_loadout_stats(s);
    inf_translate_human_commands(hi, actions, s);
    inf_step(state, actions);
    s->human_commands = NULL;
    s->human_command_count = 0;
    human_input_clear_pending(hi);
}

/* ======================================================================== */
/* encounter definition                                                      */
/* ======================================================================== */

static const EncounterDef ENCOUNTER_INFERNO = {
    .name = "inferno",
    .obs_size = INF_NUM_OBS,
    .num_action_heads = INF_NUM_ACTION_HEADS,
    .action_head_dims = INF_ACTION_DIMS,
    .mask_size = INF_ACTION_MASK_SIZE,

    .create = inf_create,
    .destroy = inf_destroy,
    .reset = inf_reset,
    .step = inf_step,
    .step_human_commands = inf_step_human_commands,

    .write_obs = inf_write_obs,
    .write_mask = inf_write_mask,
    .get_reward = inf_get_reward,
    .is_terminal = inf_is_terminal,

    .get_entity_count = inf_get_entity_count,
    .get_entity = inf_get_entity,
    .fill_render_entities = inf_fill_render_entities,

    .put_int = inf_put_int,
    .put_float = inf_put_float,
    .put_ptr = inf_put_ptr,

    .arena_base_x = INF_ARENA_MIN_X,
    .arena_base_y = INF_ARENA_MIN_Y,
    .arena_width = INF_ARENA_WIDTH,
    .arena_height = INF_ARENA_HEIGHT,

    .render_post_tick = inf_render_post_tick,
    .get_log = inf_get_log,
    .get_tick = inf_get_tick,
    .get_winner = inf_get_winner,

    .translate_human_input = inf_translate_human_input,
    .is_human_targetable_npc_slot = inf_is_human_targetable_npc_slot,
    .head_move = INF_HEAD_MOVE,
    .head_prayer = INF_HEAD_PRAYER,
    .head_target = INF_HEAD_TARGET,
};

__attribute__((constructor))
static void inf_register(void) {
    encounter_register(&ENCOUNTER_INFERNO);
}

#endif /* ENCOUNTER_INFERNO_H */
