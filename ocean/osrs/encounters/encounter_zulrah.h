/**
 * @file encounter_zulrah.h
 * @brief Zulrah boss encounter — real OSRS mechanics with 4 fixed rotations.
 *
 * Implements the actual OSRS Zulrah fight from the wiki:
 *   - 4 predetermined rotations (11-13 phases each)
 *   - 4 positions: middle, south, east, west
 *   - 3 forms: green/serpentine (ranged), red/magma (melee), blue/tanzanite (magic+ranged)
 *   - Jad phase: serpentine alternating ranged/magic or magic/ranged
 *   - Fixed action sequences per phase (exact # attacks, clouds, snakelings)
 *   - Dive transitions between phases (~5 ticks)
 *
 * Form mechanics (from OSRS wiki):
 *   Green (2042): ranged attacks with accuracy roll, max hit 41. def_magic -45, def_ranged +50.
 *   Red (2043): melee — stares at player tile, whips tail after 3-tick delay.
 *               accuracy roll + max hit 41 + stun if hit. def_magic 0, def_ranged +300.
 *   Blue (2044): random magic+ranged attacks (75% magic, 25% ranged).
 *               Magic always accurate. def_magic +300, def_ranged 0.
 *
 * Damage cap: hits over 50 → random 45-50 (Mod Ash confirmed).
 * Clouds: 3x3 area, 1-5 damage per tick.
 * Venom: 25% chance per ranged/magic attack (even through prayer).
 * Snakelings: 1 HP, melee max 15 / magic max 13, random type at spawn.
 * NPC size: 5x5. Attack speed: 3 ticks. Melee interval: 6 ticks total.
 * Gear tiers: 3 tiers (budget/mid/BIS) with precomputed stats.
 * Blowpipe spec: 50% energy, heals 50% of damage dealt.
 * Antivenom: extended anti-venom+ grants 300 ticks immunity.
 *
 * Entity layout: player (0), zulrah (1), up to 4 snakelings (2-5).
 */

#ifndef ENCOUNTER_ZULRAH_H
#define ENCOUNTER_ZULRAH_H

#include "../osrs_encounter.h"
#include "../osrs_interaction.h"
#include "../osrs_types.h"
#include "../osrs_items.h"
#include "../osrs_combat.h"
#include "../osrs_special_attacks.h"
#include "../osrs_pvp_gear.h"
#include "../osrs_consumables.h"
#include "../osrs_damage.h"
#include "../osrs_collision.h"
#include "../osrs_monsters_generated.h"
#include "../data/npc_models.h"
#include <stdlib.h>
#include <string.h>

/* ======================================================================== */
/* constants                                                                 */
/* ======================================================================== */

#define ZUL_ARENA_SIZE    28
#define ZUL_NPC_SIZE      5

/* player platform bounds (walkable tiles) — centered on the shrine area */
#define ZUL_PLATFORM_MIN  5
#define ZUL_PLATFORM_MAX  22

/* 4 zulrah positions (relative coords on 28x28 grid).
   mapped from OSRS world coords via RuneLite plugin ZulrahLocation.java,
   anchored to NORTH=(2266,3073). base offset: (2254, 3060). */
#define ZUL_POS_NORTH   0   /* RuneLite: NORTH / "middle" */
#define ZUL_POS_SOUTH   1
#define ZUL_POS_EAST    2
#define ZUL_POS_WEST    3
#define ZUL_NUM_POSITIONS 4

static const int ZUL_POSITIONS[ZUL_NUM_POSITIONS][2] = {
    { 10, 12 },  /* NORTH: center/north */
    { 10,  1 },  /* SOUTH: bottom edge */
    { 20, 10 },  /* EAST:  right edge */
    {  0, 10 },  /* WEST:  left edge */
};

/* player starting position — shrine entry (2267,3068) per RuneLite StandLocation */
#define ZUL_PLAYER_START_X  11
#define ZUL_PLAYER_START_Y  7

/* zulrah combat stats sourced from MONSTER_DATABASE (osrs_monsters_generated.h).
   all three forms share: hp=500, def_level=300, attack_speed=3.
   per-form defence: green magic_def=-45 ranged_def=50, red 0/300, blue 300/0. */
#define ZUL_BASE_HP  MONSTER_DATABASE[MON_ZULRAH_GREEN].hp  /* convenience alias for binding.c */

/* melee form: stares then whips. accuracy roll + max hit 41. stun if hit.
   wiki: melee attack speed 6. RuneLite plugin sets attackTicks=8 on melee anims
   (5806/5807) but that counter likely includes animation delay + display offset.
   stare 3 ticks, then whip (interval 3+3=6). */
#define ZUL_MELEE_STARE_TICKS 3   /* ticks before tail whip */
#define ZUL_MELEE_INTERVAL    6   /* total ticks between melee attack starts (wiki: attack speed 6) */
#define ZUL_MELEE_STUN_TICKS  5   /* stun duration */

/* damage cap: hits over 50 → random 45-50 */
#define ZUL_DAMAGE_CAP  50
#define ZUL_DAMAGE_CAP_MIN 45

/* phase transition timing (from video analysis):
   surface anim plays at start of phase, dive anim plays at end.
   phaseTicks covers the entire duration including both animations.
   no gap between phases — dive ends, next phase surfaces immediately. */
#define ZUL_SURFACE_TICKS_INITIAL 3  /* first phase: initial rise (anim 5071) */
#define ZUL_SURFACE_TICKS         2  /* subsequent phases: rise (anim 5073) */
#define ZUL_DIVE_ANIM_TICKS       3  /* dig animation at end of phase */

/* hazards */
#define ZUL_MAX_CLOUDS     7   /* observed server-side cap: 4 spits * 2 = 8 but only 7 persist */
#define ZUL_MAX_SNAKELINGS 4
#define ZUL_CLOUD_SIZE     3   /* 3x3 area (wiki confirmed) */
#define ZUL_CLOUD_DURATION 30  /* ticks before cloud fades (from RuneLite Zulrah plugin: toxicCloudsMap.put(obj, 30)) */
#define ZUL_CLOUD_DAMAGE_MIN 1
#define ZUL_CLOUD_DAMAGE_MAX 5    /* wiki: 1-5 per tick */

/* snakelings: stats from MONSTER_DATABASE (MON_ZULRAH_SNAKELING_MELEE/MAGIC) */
#define ZUL_SNAKELING_HP       1
#define ZUL_SNAKELING_SPEED    3
#define ZUL_SNAKELING_LIFESPAN 67  /* ~40 seconds = 40/0.6 ticks */

/* venom: escalating 6->8->10->...->20 every 30 ticks (~18 seconds) */
#define ZUL_VENOM_INTERVAL  30
#define ZUL_VENOM_START     6
#define ZUL_VENOM_MAX       20

/* NPC attack rolls: computed from MONSTER_DATABASE via osrs_npc_attack_roll() */

/* spawn timing for clouds/snakelings during phase actions */
#define ZUL_SPAWN_INTERVAL  3  /* ticks between each cloud/snakeling spit (same as attack speed) */
#define ZUL_CLOUD_FLIGHT_1  3  /* ticks for first cloud projectile to land */
#define ZUL_CLOUD_FLIGHT_2  4  /* ticks for second cloud projectile to land */

/* antivenom */
#define ZUL_ANTIVENOM_DURATION   300    /* extended anti-venom+: 3 minutes = 300 ticks */
#define ZUL_ANTIVENOM_DOSES      4

/* blowpipe spec — cost looked up via osrs_spec_cost(weapon) at runtime */

/* thrall: greater ghost (arceuus spellbook, always hits, ignores armour).
 * max hit 3, attack speed 4 ticks. duration = 0.6 * magic_level seconds
 * = magic_level ticks (at 99 magic = 99 ticks ≈ 59.4s, then resummon). */
#define ZUL_THRALL_MAX_HIT       3
#define ZUL_THRALL_SPEED         4      /* attacks every 4 ticks */
#define ZUL_THRALL_DURATION      99     /* ticks (0.6 * 99 magic = 59.4s) */
#define ZUL_THRALL_COOLDOWN      17     /* 10 second resummon cooldown */

/* player starting stats */
#define ZUL_PLAYER_HP         99
#define ZUL_PLAYER_PRAYER     77
#define ZUL_PLAYER_FOOD       10     /* sharks */
#define ZUL_PLAYER_KARAMBWAN  4
#define ZUL_PLAYER_RESTORE_DOSES 8   /* prayer potion doses (4 per pot = 2 pots) */
#define ZUL_MAX_TICKS         600

/* ======================================================================== */
/* observation and action space                                              */
/* ======================================================================== */

#define ZUL_NUM_OBS           84  /* +3 for offensive prayer one-hot (piety/rigour/augury) */
#define ZUL_NUM_ACTION_HEADS  7

#define ZUL_MOVE_DIM      ENCOUNTER_MOVE_ACTIONS
#define ZUL_ATTACK_DIM    3
#define ZUL_PRAYER_DIM    ENCOUNTER_OVERHEAD_DIM_PVE   /* 4: no_change, toggle_melee/ranged/magic */
#define ZUL_OFFENSIVE_DIM ENCOUNTER_OFFENSIVE_DIM      /* 4: no_change, toggle_piety/rigour/augury */
#define ZUL_FOOD_DIM      3   /* none, shark, karambwan */
#define ZUL_POTION_DIM    3   /* none, restore, antivenom */
#define ZUL_SPEC_DIM      2

#define ZUL_ACTION_MASK_SIZE (ZUL_MOVE_DIM + ZUL_ATTACK_DIM + ZUL_PRAYER_DIM + \
    ZUL_FOOD_DIM + ZUL_POTION_DIM + ZUL_SPEC_DIM + ZUL_OFFENSIVE_DIM)

#define ZUL_HEAD_MOVE       0
#define ZUL_HEAD_ATTACK     1
#define ZUL_HEAD_PRAYER     2
#define ZUL_HEAD_FOOD       3
#define ZUL_HEAD_POTION     4
#define ZUL_HEAD_SPEC       5
#define ZUL_HEAD_OFFENSIVE  6

#define ZUL_MOVE_STAY 0
#define ZUL_ATK_NONE  0
#define ZUL_ATK_MAGE  1
#define ZUL_ATK_RANGE 2

/* ======================================================================== */
/* enums                                                                     */
/* ======================================================================== */

typedef enum {
    ZUL_FORM_GREEN = 0,  /* 2042: serpentine, ranged */
    ZUL_FORM_RED,        /* 2043: magma, melee */
    ZUL_FORM_BLUE,       /* 2044: tanzanite, magic+ranged */
} ZulrahForm;

/* map zulrah form to MONSTER_DATABASE index */
static const int ZUL_FORM_MONSTER_IDX[] = {
    [ZUL_FORM_GREEN] = MON_ZULRAH_GREEN,
    [ZUL_FORM_RED]   = MON_ZULRAH_RED,
    [ZUL_FORM_BLUE]  = MON_ZULRAH_BLUE,
};

typedef enum {
    ZUL_GEAR_MAGE = 0,
    ZUL_GEAR_RANGE,
} ZulrahGearStyle;

/* ======================================================================== */
/* rotation data: action types for phase sequences                           */
/* ======================================================================== */

typedef enum {
    ZA_END = 0,           /* sentinel — end of action list */
    ZA_RANGED,            /* green form ranged attacks */
    ZA_MAGIC_RANGED,      /* blue form random magic/ranged (magic more frequent) */
    ZA_MELEE,             /* red form melee (stare + tail whip) */
    ZA_JAD_RM,            /* jad: alternating, starting with ranged */
    ZA_JAD_MR,            /* jad: alternating, starting with magic */
    ZA_CLOUDS,            /* venom cloud barrages */
    ZA_SNAKELINGS,        /* snakeling orbs */
    ZA_SNAKECLOUD_ALT,    /* alternating: snakeling, cloud, snakeling, cloud... */
    ZA_CLOUDSNAKE_ALT,    /* alternating: cloud, snakeling, cloud, snakeling... */
} ZulActionType;

typedef struct {
    uint8_t type;   /* ZulActionType */
    uint8_t count;
} ZulAction;

#define ZUL_MAX_PHASE_ACTIONS 6

typedef struct {
    uint8_t position;  /* ZUL_POS_NORTH etc. */
    uint8_t form;      /* ZUL_FORM_GREEN etc. */
    uint8_t stand;     /* ZUL_STAND_* — safe tile for this phase */
    uint8_t stall;     /* ZUL_STAND_* — stall tile (or ZUL_STAND_NONE) */
    uint8_t phase_ticks;  /* total ticks at this position (from RuneLite plugin phaseTicks) */
    ZulAction actions[ZUL_MAX_PHASE_ACTIONS];
} ZulRotationPhase;

#define ZUL_MAX_ROT_PHASES 13
#define ZUL_NUM_ROTATIONS  4

/* stand locations converted from RuneLite plugin StandLocation.java.
   plugin uses OSRS local coords (128 units/tile). conversion:
   grid_x = local_x/128 - 38, grid_y = local_y/128 - 44
   (derived from NORTH zulrah center (6720,7616) → grid (12,13)) */
/* safe tile positions adapted from zulrah helper plugin — approximate. */
#define ZUL_STAND_SOUTHWEST       0   /* (10,  9) */
#define ZUL_STAND_WEST            1   /* ( 8, 15) */
#define ZUL_STAND_CENTER          2   /* (15, 10) */
#define ZUL_STAND_NORTHEAST_TOP   3   /* (20, 17) */
#define ZUL_STAND_NORTHEAST_BOT   4   /* (19, 17) */
#define ZUL_STAND_NORTHWEST_TOP   5   /* ( 8, 16) */
#define ZUL_STAND_NORTHWEST_BOT   6   /* (10, 17) */
#define ZUL_STAND_EAST_PILLAR_S   7   /* (18, 10) */
#define ZUL_STAND_EAST_PILLAR     8   /* (18, 11) */
#define ZUL_STAND_EAST_PILLAR_N   9   /* (18, 13) */
#define ZUL_STAND_EAST_PILLAR_N2 10   /* (18, 14) */
#define ZUL_STAND_WEST_PILLAR_S  11   /* (10, 10) */
#define ZUL_STAND_WEST_PILLAR    12   /* (10, 11) */
#define ZUL_STAND_WEST_PILLAR_N  13   /* (10, 13) */
#define ZUL_STAND_WEST_PILLAR_N2 14   /* (10, 14) */
#define ZUL_NUM_STAND_LOCATIONS  15
#define ZUL_STAND_NONE           255  /* no stall location */

static const int ZUL_STAND_COORDS[ZUL_NUM_STAND_LOCATIONS][2] = {
    {  8,  8 },  /* SOUTHWEST */
    {  6, 14 },  /* WEST */
    { 13,  9 },  /* CENTER */
    { 18, 16 },  /* NORTHEAST_TOP */
    { 17, 16 },  /* NORTHEAST_BOTTOM */
    {  6, 15 },  /* NORTHWEST_TOP */
    {  8, 16 },  /* NORTHWEST_BOTTOM */
    { 16,  9 },  /* EAST_PILLAR_S */
    { 16, 10 },  /* EAST_PILLAR */
    { 16, 12 },  /* EAST_PILLAR_N */
    { 16, 13 },  /* EAST_PILLAR_N2 */
    {  8,  9 },  /* WEST_PILLAR_S */
    {  8, 10 },  /* WEST_PILLAR */
    {  8, 12 },  /* WEST_PILLAR_N */
    {  8, 13 },  /* WEST_PILLAR_N2 */
};

#define ZA(t,c) { (uint8_t)(t), (uint8_t)(c) }
#define ZE { 0, 0 }  /* ZA_END sentinel */
#define _N ZUL_STAND_NONE  /* no stall location shorthand */

/* rotation 1: "Magma A" — 11 phases
   stand/stall/phaseTicks from RuneLite plugin RotationType.java ROT_A */
static const ZulRotationPhase ZUL_ROT1[11] = {
    /* 1  */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
    /* 2  */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
    /* 3  */ { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 18, { ZA(ZA_MAGIC_RANGED,4), ZE } },
    /* 4  */ { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, ZUL_STAND_WEST_PILLAR_N2, 39, { ZA(ZA_RANGED,5), ZA(ZA_SNAKELINGS,2), ZA(ZA_CLOUDS,2), ZA(ZA_SNAKELINGS,2), ZE } },
    /* 5  */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_WEST_PILLAR_N, _N, 22, { ZA(ZA_MELEE,2), ZE } },
    /* 6  */ { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_EAST_PILLAR_S, 20, { ZA(ZA_MAGIC_RANGED,5), ZE } },
    /* 7  */ { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR, _N, 28, { ZA(ZA_CLOUDS,3), ZA(ZA_SNAKELINGS,4), ZE } },
    /* 8  */ { ZUL_POS_SOUTH,  ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR, ZUL_STAND_EAST_PILLAR_N2, 36, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKECLOUD_ALT,5), ZE } },
    /* 9  */ { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_EAST_PILLAR_S, 48, { ZA(ZA_JAD_RM,10), ZA(ZA_CLOUDS,4), ZE } },
    /* 10 */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
    /* 11 */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

/* rotation 2: "Magma B" — 11 phases
   stand/stall/phaseTicks from RuneLite plugin RotationType.java ROT_B */
static const ZulRotationPhase ZUL_ROT2[11] = {
    /* 1  */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
    /* 2  */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
    /* 3  */ { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 18, { ZA(ZA_MAGIC_RANGED,4), ZE } },
    /* 4  */ { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_S, _N, 28, { ZA(ZA_CLOUDS,3), ZA(ZA_SNAKELINGS,4), ZE } },
    /* 5  */ { ZUL_POS_SOUTH,  ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_N, ZUL_STAND_WEST_PILLAR_N2, 39, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKELINGS,2), ZA(ZA_CLOUDS,2), ZA(ZA_SNAKELINGS,2), ZE } },
    /* 6  */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_WEST_PILLAR_N, _N, 21, { ZA(ZA_MELEE,2), ZE } },
    /* 7  */ { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_CENTER, ZUL_STAND_WEST_PILLAR_S, 20, { ZA(ZA_RANGED,5), ZE } },
    /* 8  */ { ZUL_POS_SOUTH,  ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_WEST_PILLAR_N2, 36, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKECLOUD_ALT,5), ZE } },
    /* 9  */ { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_EAST_PILLAR_S, 48, { ZA(ZA_JAD_RM,10), ZA(ZA_CLOUDS,4), ZE } },
    /* 10 */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
    /* 11 */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

/* rotation 3: "Serp" — 12 phases
   stand/stall/phaseTicks from RuneLite plugin RotationType.java ROT_C */
static const ZulRotationPhase ZUL_ROT3[12] = {
    /* 1  */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
    /* 2  */ { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 30, { ZA(ZA_RANGED,5), ZA(ZA_SNAKELINGS,3), ZE } },
    /* 3  */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_WEST, _N, 40, { ZA(ZA_CLOUDSNAKE_ALT,6), ZA(ZA_MELEE,2), ZE } },
    /* 4  */ { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST, ZUL_STAND_EAST_PILLAR_S, 20, { ZA(ZA_MAGIC_RANGED,5), ZE } },
    /* 5  */ { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_S, ZUL_STAND_EAST_PILLAR_N2, 20, { ZA(ZA_RANGED,5), ZE } },
    /* 6  */ { ZUL_POS_EAST,   ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_S, ZUL_STAND_WEST_PILLAR_S, 20, { ZA(ZA_MAGIC_RANGED,5), ZE } },
    /* 7  */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, _N, 25, { ZA(ZA_CLOUDS,3), ZA(ZA_SNAKELINGS,3), ZE } },
    /* 8  */ { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, _N, 20, { ZA(ZA_RANGED,5), ZE } },
    /* 9  */ { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 36, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_CLOUDS,2), ZA(ZA_SNAKELINGS,3), ZE } },
    /* 10 */ { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_N, _N, 35, { ZA(ZA_JAD_MR,10), ZE } },
    /* 11 */ { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_NORTHEAST_TOP, _N, 18, { ZA(ZA_SNAKELINGS,4), ZE } },
    /* 12 */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

/* rotation 4: "Tanz" — 13 phases
   stand/stall/phaseTicks from RuneLite plugin RotationType.java ROT_D */
static const ZulRotationPhase ZUL_ROT4[13] = {
    /* 1  */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
    /* 2  */ { ZUL_POS_EAST,   ZUL_FORM_BLUE,  ZUL_STAND_NORTHEAST_TOP, _N, 36, { ZA(ZA_SNAKELINGS,4), ZA(ZA_MAGIC_RANGED,6), ZE } },
    /* 3  */ { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, ZUL_STAND_WEST_PILLAR_N2, 24, { ZA(ZA_RANGED,4), ZA(ZA_CLOUDS,2), ZE } },
    /* 4  */ { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_N, _N, 30, { ZA(ZA_SNAKELINGS,4), ZA(ZA_MAGIC_RANGED,4), ZE } },
    /* 5  */ { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_EAST_PILLAR_N, _N, 28, { ZA(ZA_MELEE,2), ZA(ZA_CLOUDS,2), ZE } },
    /* 6  */ { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR, _N, 17, { ZA(ZA_RANGED,4), ZE } },
    /* 7  */ { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR, _N, 34, { ZA(ZA_SNAKELINGS,6), ZA(ZA_CLOUDS,3), ZE } },
    /* 8  */ { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_S, _N, 33, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKELINGS,4), ZE } },
    /* 9  */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 20, { ZA(ZA_RANGED,4), ZE } },
    /* 10 */ { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 27, { ZA(ZA_MAGIC_RANGED,4), ZA(ZA_CLOUDS,3), ZE } },
    /* 11 */ { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_N, _N, 29, { ZA(ZA_JAD_MR,8), ZE } },
    /* 12 */ { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_NORTHEAST_TOP, _N, 18, { ZA(ZA_SNAKELINGS,4), ZE } },
    /* 13 */ { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

#undef _N

/* rotation table: pointers + lengths */
static const ZulRotationPhase* const ZUL_ROTATIONS[ZUL_NUM_ROTATIONS] = {
    ZUL_ROT1, ZUL_ROT2, ZUL_ROT3, ZUL_ROT4,
};
static const int ZUL_ROT_LENGTHS[ZUL_NUM_ROTATIONS] = { 11, 11, 12, 13 };

#undef ZA
#undef ZE

/* ======================================================================== */
/* static arrays                                                             */
/* ======================================================================== */

static const int ZUL_ACTION_HEAD_DIMS[ZUL_NUM_ACTION_HEADS] = {
    ZUL_MOVE_DIM, ZUL_ATTACK_DIM, ZUL_PRAYER_DIM,
    ZUL_FOOD_DIM, ZUL_POTION_DIM, ZUL_SPEC_DIM, ZUL_OFFENSIVE_DIM,
};

/* movement uses shared encounter_move_to_target + ENCOUNTER_MOVE_TARGET_DX/DY from osrs_encounter.h */

/* number of gear tiers for validation (used in put_int) */
#define ZUL_NUM_GEAR_TIERS 3

/* per-tier equipped item loadouts: [tier][slot] = ItemIndex.
   mage_loadout is worn while casting mage. range_loadout while ranging.
   slots: HEAD CAPE NECK AMMO WEAPON SHIELD BODY LEGS HANDS FEET RING */
static const uint8_t ZUL_MAGE_LOADOUT[ZUL_NUM_GEAR_TIERS][NUM_GEAR_SLOTS] = {
    /* tier 0: mystic + trident + book of darkness */
    { ITEM_MYSTIC_HAT, ITEM_GOD_CAPE, ITEM_GLORY, ITEM_AMETHYST_ARROW,
      ITEM_TRIDENT_OF_SWAMP, ITEM_BOOK_OF_DARKNESS, ITEM_MYSTIC_TOP, ITEM_MYSTIC_BOTTOM,
      ITEM_BARROWS_GLOVES, ITEM_MYSTIC_BOOTS, ITEM_RING_OF_RECOIL },
    /* tier 1: ahrim's + sang staff + mage's book */
    { ITEM_AHRIMS_HOOD, ITEM_GOD_CAPE, ITEM_OCCULT_NECKLACE, ITEM_GOD_BLESSING,
      ITEM_SANGUINESTI_STAFF, ITEM_MAGES_BOOK, ITEM_AHRIMS_ROBETOP, ITEM_AHRIMS_ROBESKIRT,
      ITEM_TORMENTED_BRACELET, ITEM_INFINITY_BOOTS, ITEM_RING_OF_RECOIL },
    /* tier 2: ancestral + eye of ayak + elidinis' ward */
    { ITEM_ANCESTRAL_HAT, ITEM_IMBUED_SARA_CAPE, ITEM_OCCULT_NECKLACE, ITEM_DRAGON_ARROWS,
      ITEM_EYE_OF_AYAK, ITEM_ELIDINIS_WARD_F, ITEM_ANCESTRAL_TOP, ITEM_ANCESTRAL_BOTTOM,
      ITEM_CONFLICTION_GAUNTLETS, ITEM_AVERNIC_TREADS, ITEM_RING_OF_SUFFERING_RI },
};

static const uint8_t ZUL_RANGE_LOADOUT[ZUL_NUM_GEAR_TIERS][NUM_GEAR_SLOTS] = {
    /* tier 0: black d'hide + magic shortbow (i) */
    { ITEM_BLESSED_COIF, ITEM_AVAS_ACCUMULATOR, ITEM_GLORY, ITEM_AMETHYST_ARROW,
      ITEM_MAGIC_SHORTBOW_I, ITEM_NONE, ITEM_BLACK_DHIDE_BODY, ITEM_BLACK_DHIDE_CHAPS,
      ITEM_BARROWS_GLOVES, ITEM_MYSTIC_BOOTS, ITEM_RING_OF_RECOIL },
    /* tier 1: crystal + bow of faerdhinen */
    { ITEM_CRYSTAL_HELM, ITEM_AVAS_ASSEMBLER, ITEM_NECKLACE_OF_ANGUISH, ITEM_GOD_BLESSING,
      ITEM_BOW_OF_FAERDHINEN, ITEM_NONE, ITEM_CRYSTAL_BODY, ITEM_CRYSTAL_LEGS,
      ITEM_BARROWS_GLOVES, ITEM_BLESSED_DHIDE_BOOTS, ITEM_RING_OF_RECOIL },
    /* tier 2: masori + twisted bow */
    { ITEM_MASORI_MASK_F, ITEM_DIZANAS_QUIVER, ITEM_NECKLACE_OF_ANGUISH, ITEM_DRAGON_ARROWS,
      ITEM_TWISTED_BOW, ITEM_NONE, ITEM_MASORI_BODY_F, ITEM_MASORI_CHAPS_F,
      ITEM_ZARYTE_VAMBRACES, ITEM_AVERNIC_TREADS, ITEM_RING_OF_SUFFERING_RI },
};

/* gear switching uses shared helpers from osrs_encounter.h:
   encounter_apply_loadout() and encounter_populate_inventory(). */
static void zul_populate_player_inventory(Player* p, int gear_tier) {
    const uint8_t* loadouts[] = {
        ZUL_MAGE_LOADOUT[gear_tier],
        ZUL_RANGE_LOADOUT[gear_tier],
    };
    encounter_populate_inventory(p, loadouts, 2, NULL);
}

/* snakeling spawn positions (shifted +6x for new base offset 2254,3060) */
#define ZUL_NUM_SNAKELING_POSITIONS 5
static const int ZUL_SNAKELING_POSITIONS[ZUL_NUM_SNAKELING_POSITIONS][2] = {
    { 7, 14 }, { 7, 10 }, { 12, 8 }, { 17, 10 }, { 17, 16 },
};

/* cloud spawn: pick random platform tile that isn't a safe tile for this phase */
static int zul_tile_is_safe(int x, int y, int stand_id, int stall_id) {
    /* safe tiles: the stand location and stall location for this phase */
    if (stand_id < ZUL_NUM_STAND_LOCATIONS) {
        int sx = ZUL_STAND_COORDS[stand_id][0];
        int sy = ZUL_STAND_COORDS[stand_id][1];
        /* 2-tile radius around stand spot is "safe enough" for the agent */
        if (abs(x - sx) <= 1 && abs(y - sy) <= 1) return 1;
    }
    if (stall_id < ZUL_NUM_STAND_LOCATIONS) {
        int sx = ZUL_STAND_COORDS[stall_id][0];
        int sy = ZUL_STAND_COORDS[stall_id][1];
        if (abs(x - sx) <= 1 && abs(y - sy) <= 1) return 1;
    }
    return 0;
}

/* ======================================================================== */
/* structs                                                                   */
/* ======================================================================== */

typedef struct {
    int x, y;
    int active;
    int ticks_remaining;
} ZulrahCloud;

/* cloud projectile in flight — becomes a ZulrahCloud when delay reaches 0 */
#define ZUL_MAX_PENDING_CLOUDS 16
typedef struct {
    int x, y;
    int delay;  /* ticks until cloud spawns (0 = inactive) */
} ZulrahPendingCloud;

typedef struct {
    Player entity;
    int active;
    int attack_timer;
    int is_magic;       /* 1=magic attacks, 0=melee attacks (random at spawn) */
    int lifespan;       /* ticks until auto-death */
} ZulrahSnakeling;

typedef struct {
    /* entities */
    Player player;
    Player zulrah;

    /* rotation tracking */
    int rotation_index;       /* which of 4 rotations (0-3) */
    int phase_index;          /* current phase within rotation (0-based) */

    /* phase action execution */
    int action_index;         /* which action in current phase's action list */
    int action_progress;      /* how many of the current action's count completed */
    int action_timer;         /* ticks until next action fires */
    int jad_is_magic_next;    /* for jad phase: 1 if next attack is magic */

    /* zulrah state */
    ZulrahForm current_form;
    int zulrah_visible;
    int zulrah_attacking;     /* currently in an attacking phase (not spawning/diving) */

    /* melee state: stare at tile, then whip */
    int melee_target_x, melee_target_y;
    int melee_pending;
    int melee_stare_timer;

    /* phase timing: phaseTicks covers surface + actions + dive.
       phase_timer counts down each tick. surface_timer delays actions at start. */
    int phase_timer;
    int surface_timer;    /* ticks of surface animation before actions start */
    int is_diving;        /* set when phase_timer <= ZUL_DIVE_ANIM_TICKS */

    /* player stun (from melee hit) */
    int player_stunned_ticks;

    /* hazards */
    ZulrahCloud clouds[ZUL_MAX_CLOUDS];
    ZulrahPendingCloud pending_clouds[ZUL_MAX_PENDING_CLOUDS];
    ZulrahSnakeling snakelings[ZUL_MAX_SNAKELINGS];

    /* player combat */
    ZulrahGearStyle player_gear;
    OsrsInteraction interaction;  /* shared interaction state (persistent entity targeting) */
    int player_dest_x, player_dest_y;   /* click destination for 2-tile run clamping */
    int player_dest_explicit;            /* 1 = dest set via put_int (human click), skip direction-based override */

    /* venom + antivenom */
    int venom_counter;
    int venom_timer;
    int antivenom_timer;          /* ticks remaining of anti-venom immunity */

    /* gear tier */
    int gear_tier;                /* 0=budget, 1=mid, 2=BIS */

    /* derived combat stats (computed from ITEM_DATABASE + loadout in zul_reset) */
    EncounterLoadoutStats mage_stats;
    EncounterLoadoutStats range_stats;
    int human_command_mode;
    EncounterLoadoutStats human_loadout_stats;
    const HumanCommand* human_commands;
    int human_command_count;

    /* eye of ayak soul rend: cumulative magic defence drain on zulrah.
     * carries over between forms (magic defence is a stat, not a level). */
    int magic_def_drain;

    /* thrall (arceuus greater ghost): auto-attacks zulrah every 4 ticks,
     * always hits 0-3, ignores armour. auto-resummons after expiry + cooldown. */
    int thrall_active;
    int thrall_attack_timer;
    int thrall_duration_remaining;
    int thrall_cooldown;

    /* collision */
    void* collision_map;      /* CollisionMap* for walkability checks */
    int world_offset_x;       /* local (0,0) = world (offset_x, offset_y) */
    int world_offset_y;

    /* episode */
    int tick;
    int episode_over;
    int winner;
    uint32_t rng_state;

    /* reward tracking */
    float reward;
    float episode_return;  /* running sum of reward across all ticks */
    float damage_dealt_this_tick;
    float damage_received_this_tick;
    int prayer_blocked_this_tick;
    float total_damage_dealt;
    float total_damage_received;
    int total_prayer_correct;
    int total_prayer_total;
    int total_gear_switches;
    int total_food_eaten;
    int total_potions_used;
    int total_venom_ticks;        /* ticks spent venomed */
    int total_phases_completed;

    /* visual: attack events this tick for projectile rendering */
    struct {
        int src_x, src_y, dst_x, dst_y;
        int style;   /* 0=ranged, 1=magic, 2=melee */
        int damage;
    } attack_events[8];
    int attack_event_count;

    /* visual: cloud projectile events this tick (style=3, fly from zulrah to landing) */
    struct {
        int src_x, src_y, dst_x, dst_y;
        int flight_ticks;  /* how many game ticks the projectile flies */
    } cloud_events[4];
    int cloud_event_count;

    Log log;
} ZulrahState;

/* RNG: use shared encounter_rand_int(), encounter_rand_float() from osrs_combat.h */
static const EncounterLoadoutStats* zul_current_loadout_stats(ZulrahState* s, int is_mage);

/* ======================================================================== */
/* helpers                                                                   */
/* ======================================================================== */

static inline int zul_on_platform_bounds(int x, int y) {
    return x >= ZUL_PLATFORM_MIN && x <= ZUL_PLATFORM_MAX &&
           y >= ZUL_PLATFORM_MIN && y <= ZUL_PLATFORM_MAX;
}

/* check if local tile (x,y) is walkable via collision map, fallback to bbox */
static inline int zul_on_platform(ZulrahState* s, int x, int y) {
    if (!s->collision_map) return zul_on_platform_bounds(x, y);
    int wx = x + s->world_offset_x;
    int wy = y + s->world_offset_y;
    return collision_tile_walkable((const CollisionMap*)s->collision_map, 0, wx, wy);
}

/* BFS pathfinding uses shared encounter_pathfind from osrs_encounter.h */
#define zul_pathfind(s, sx, sy, dx, dy) \
    encounter_pathfind((const CollisionMap*)(s)->collision_map, \
        (s)->world_offset_x, (s)->world_offset_y, (sx), (sy), (dx), (dy), NULL, NULL)

/* walkability callback for encounter_move_toward_dest */
static int zul_tile_walkable(void* ctx, int x, int y) {
    return zul_on_platform((ZulrahState*)ctx, x, y);
}

/* cloud overlap: player (1x1) inside cloud (3x3) */
static inline int zul_player_in_cloud(int cx, int cy, int px, int py) {
    return px >= cx && px < cx + ZUL_CLOUD_SIZE &&
           py >= cy && py < cy + ZUL_CLOUD_SIZE;
}


static int zul_form_npc_id(ZulrahForm f) {
    return MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[f]].npc_id;
}

/* apply damage cap: hits over 50 → random 45-50 */
static inline int zul_cap_damage(ZulrahState* s, int damage) {
    if (damage > ZUL_DAMAGE_CAP) {
        return ZUL_DAMAGE_CAP_MIN + encounter_rand_int(&s->rng_state, ZUL_DAMAGE_CAP - ZUL_DAMAGE_CAP_MIN + 1);
    }
    return damage;
}

/* ======================================================================== */
/* damage application                                                        */
/* ======================================================================== */

/** Apply damage to the player. If attacker is non-NULL and player has a recoil
    ring equipped, reflects floor(damage * 0.1) + 1 back to the attacker.
    Pass NULL for environmental damage (clouds, venom) where recoil doesn't apply. */
static void zul_apply_player_damage(ZulrahState* s, int damage, AttackStyle style,
                                    Player* attacker) {
    if (damage <= 0) return;
    encounter_damage_player(&s->player, damage, &s->damage_received_this_tick);
    s->total_damage_received += damage;
    s->player.hit_style = style;

    if (attacker) {
        osrs_ensure_player_equipment(&s->player);
        DamageResult damage_result = osrs_apply_passive_damage_pipeline(
            damage,
            style,
            s->player.prayer,
            /* is_pvp */ 0,
            /* target_veng_active */ 0,
            /* attacker_smite_active */ 0,
            &s->player.equipment_effect_profile,
            &s->player.item_effect_state,
            &s->rng_state
        );
        if (damage_result.recoil_damage > 0) {
            int recoil = damage_result.recoil_damage;
            if (s->player.equipment_effect_profile.recoil_source == OSRS_RECOIL_SOURCE_RING_OF_RECOIL &&
                recoil > s->player.item_effect_state.recoil_charges) {
                recoil = s->player.item_effect_state.recoil_charges;
            }
            encounter_damage_player(attacker, recoil, NULL);
            osrs_consume_recoil_charges(&s->player, recoil);
        }
    }
}

/* venom from ranged/magic attacks — 25% chance per hit (wiki/deob confirmed).
   applied even through prayer (unless miss). starts venom counter if not already venomed. */
static void zul_try_envenom(ZulrahState* s) {
    if (s->venom_counter > 0) return;                /* already venomed */
    if (s->antivenom_timer > 0) return;               /* anti-venom active */
    if (encounter_rand_int(&s->rng_state, 4) != 0) return;              /* 25% chance */
    s->venom_counter = 1;
    s->venom_timer = ZUL_VENOM_INTERVAL;
}

/* ======================================================================== */
/* NPC accuracy roll                                                         */
/* ======================================================================== */

/* OSRS accuracy formula: if att > def: 1 - (def+2)/(2*(att+1)), else att/(2*(def+1)) */
/* hit chance: use shared OSRS accuracy formula from osrs_combat.h */

/* compute player's defence roll against a specific NPC attack style.
   uses current gear loadout stats (derived from ITEM_DATABASE).
   magic defence uses 70% magic level + 30% defence level per OSRS formula. */
static int zul_player_def_roll(ZulrahState* s, int attack_style) {
    const EncounterLoadoutStats* ls = zul_current_loadout_stats(
        s, s->player_gear == ZUL_GEAR_MAGE);
    /* melee_style=2 (crush) for zulrah tail whip */
    int def_bonus = encounter_player_def_bonus(
        ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
        attack_style, 2);
    int roll = osrs_player_def_roll_vs_npc(99, 99, def_bonus, attack_style);
    return roll > 0 ? roll : 0;
}

/* ======================================================================== */
/* zulrah attack dispatch                                                    */
/* ======================================================================== */

/* record a visual attack event for projectile rendering */
static void zul_record_attack(ZulrahState* s, int src_x, int src_y,
                               int dst_x, int dst_y, int style, int damage) {
    s->zulrah.npc_anim_id = ZULRAH_ANIM_ATTACK;
    if (s->attack_event_count >= 8) return;
    int i = s->attack_event_count++;
    s->attack_events[i].src_x = src_x;
    s->attack_events[i].src_y = src_y;
    s->attack_events[i].dst_x = dst_x;
    s->attack_events[i].dst_y = dst_y;
    s->attack_events[i].style = style;
    s->attack_events[i].damage = damage;
}

/* ranged attack (green form, or blue form ranged variant).
   unlike magic, ranged CAN miss (accuracy roll required).
   wiki: "ranged and magic attacks will envenom the player unless they miss,
   even if blocked by a protection prayer." so venom only on hit. */
/* known sim gap: ranged/magic attacks ignore LOS (no pillar blocking for projectiles). */
static void zul_attack_ranged(ZulrahState* s) {
    const MonsterStats* m = &MONSTER_DATABASE[MON_ZULRAH_GREEN];
    int npc_att_roll = osrs_npc_attack_roll(m->range_level, m->range_att_bonus);
    int dmg = 0;
    int did_hit = 0;
    s->total_prayer_total++;
    if (encounter_prayer_correct_for_style(s->player.prayer, ATTACK_STYLE_RANGED)) {
        s->total_prayer_correct++;
        /* prayer blocks damage but venom still applies (unless miss) */
        int def_roll = zul_player_def_roll(s, ATTACK_STYLE_RANGED);
        float chance = osrs_hit_chance(npc_att_roll, def_roll);
        did_hit = (encounter_rand_float(&s->rng_state) < chance);
        if (did_hit) {
            s->prayer_blocked_this_tick = 1;
        }
    } else {
        int def_roll = zul_player_def_roll(s, ATTACK_STYLE_RANGED);
        float chance = osrs_hit_chance(npc_att_roll, def_roll);
        if (encounter_rand_float(&s->rng_state) < chance) {
            did_hit = 1;
            dmg = encounter_rand_int(&s->rng_state, m->max_hit + 1);
            zul_apply_player_damage(s, dmg, ATTACK_STYLE_RANGED, &s->zulrah);
        }
    }
    if (did_hit) zul_try_envenom(s);
    zul_record_attack(s, s->zulrah.x, s->zulrah.y,
                      s->player.x, s->player.y, 0, dmg);
}

/* magic attack (blue form, always accurate per wiki).
   wiki: "ranged and magic attacks will envenom the player unless they miss,
   even if blocked by a protection prayer." magic never misses → always envenoms. */
static void zul_attack_magic(ZulrahState* s) {
    int dmg = 0;
    s->total_prayer_total++;
    if (encounter_prayer_correct_for_style(s->player.prayer, ATTACK_STYLE_MAGIC)) {
        s->total_prayer_correct++;
        s->prayer_blocked_this_tick = 1;
    } else {
        dmg = encounter_rand_int(&s->rng_state, MONSTER_DATABASE[MON_ZULRAH_BLUE].max_hit + 1);
        zul_apply_player_damage(s, dmg, ATTACK_STYLE_MAGIC, &s->zulrah);
    }
    /* magic always hits → always try envenom (even if prayer blocked damage) */
    zul_try_envenom(s);
    zul_record_attack(s, s->zulrah.x, s->zulrah.y,
                      s->player.x, s->player.y, 1, dmg);
}

/* blue/tanzanite form: random magic or ranged. wiki says magic more frequent. */
static void zul_attack_magic_ranged(ZulrahState* s) {
    if (encounter_rand_int(&s->rng_state, 4) < 3) {  /* 75% magic, 25% ranged */
        zul_attack_magic(s);
    } else {
        zul_attack_ranged(s);
    }
}

/* pillar safespot: the east and west pillars block Zulrah's melee tail whip.
   pillars at (15,10) east and (9,10) west.
   safe tile is 2 tiles east/west + 1 tile north of each pillar:
     east safespot: (17, 11)
     west safespot: ( 7, 11)  */
static int zul_on_pillar_safespot(int px, int py) {
    if (py != 11) return 0;
    return (px == 17 || px == 7);
}

/* red/magma melee: initiate stare at player's tile */
static void zul_melee_start(ZulrahState* s) {
    s->melee_target_x = s->player.x;
    s->melee_target_y = s->player.y;
    s->melee_pending = 1;
    s->melee_stare_timer = ZUL_MELEE_STARE_TICKS;
}

/* melee hit lands after stare completes.
   wiki: "If the player does not move away from the targeted area in time,
   they will be dealt 20-30 damage and be stunned for several seconds."
   no accuracy roll — guaranteed hit if player is on the targeted tile. */
static void zul_melee_hit(ZulrahState* s) {
    s->melee_pending = 0;
    int dmg = 0;
    if (s->player.x == s->melee_target_x && s->player.y == s->melee_target_y
        && !zul_on_pillar_safespot(s->player.x, s->player.y)) {
        s->total_prayer_total++;
        if (encounter_prayer_correct_for_style(s->player.prayer, ATTACK_STYLE_MELEE)) {
            s->total_prayer_correct++;
            s->prayer_blocked_this_tick = 1;
        } else {
            dmg = 20 + encounter_rand_int(&s->rng_state, 11);  /* 20-30 per wiki */
            zul_apply_player_damage(s, dmg, ATTACK_STYLE_MELEE, &s->zulrah);
            s->player_stunned_ticks = ZUL_MELEE_STUN_TICKS;
        }
    }
    zul_record_attack(s, s->zulrah.x, s->zulrah.y,
                      s->melee_target_x, s->melee_target_y, 2, dmg);
}

/* jad phase: alternating ranged/magic */
static void zul_attack_jad(ZulrahState* s) {
    if (s->jad_is_magic_next) {
        zul_attack_magic(s);
    } else {
        zul_attack_ranged(s);
    }
    s->jad_is_magic_next = !s->jad_is_magic_next;
}

/* ======================================================================== */
/* player attacks zulrah                                                     */
/* ======================================================================== */

/* per-form defence bonuses from MONSTER_DATABASE */
static inline void zul_form_def_bonuses(ZulrahForm form, int* def_magic, int* def_ranged) {
    const MonsterStats* m = &MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[form]];
    *def_magic = m->magic_def;
    *def_ranged = m->ranged_def;
}

static AttackStyle zul_player_equipped_attack_style(const ZulrahState* s) {
    AttackStyle style = (AttackStyle)get_item_attack_style(s->player.equipped[GEAR_SLOT_WEAPON]);
    if (style == ATTACK_STYLE_MAGIC ||
        style == ATTACK_STYLE_RANGED ||
        style == ATTACK_STYLE_MELEE) {
        return style;
    }
    return ATTACK_STYLE_RANGED;
}

static void zul_refresh_human_loadout_stats(ZulrahState* s) {
    AttackStyle style = zul_player_equipped_attack_style(s);
    FightStyle fight_style = s->player.fight_style;
    int spell_base_damage = (style == ATTACK_STYLE_MAGIC) ? 30 : 0;
    encounter_compute_player_equipped_stats(
        &s->player, style, fight_style, spell_base_damage,
        &s->human_loadout_stats);
}

static const EncounterLoadoutStats* zul_current_loadout_stats(ZulrahState* s, int is_mage) {
    if (s->human_command_mode) {
        zul_refresh_human_loadout_stats(s);
        return &s->human_loadout_stats;
    }
    return is_mage ? &s->mage_stats : &s->range_stats;
}

static int zul_player_attack_hits(
    ZulrahState* s, int is_mage, const OsrsPreparedAttackEffects* attack_effects
) {
    int att_roll = attack_effects->attack_roll;
    /* crystal armor set bonus: +30% ranged accuracy with bowfa (tier 1 only) */
    if (!is_mage && s->gear_tier == 1)
        att_roll = att_roll * 130 / 100;

    int def_magic = 0, def_ranged = 0;
    zul_form_def_bonuses(s->current_form, &def_magic, &def_ranged);
    /* apply eye of ayak magic defence drain (carries across forms) */
    if (is_mage) {
        def_magic -= s->magic_def_drain;
        if (def_magic < -64) def_magic = -64;  /* can't go below -64 (makes def_roll 0) */
    }
    int def_bonus = is_mage ? def_magic : def_ranged;
    int def_roll = (MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[s->current_form]].def_level + 8) * (def_bonus + 64);
    if (def_roll < 0) def_roll = 0;

    if (attack_effects->use_double_accuracy) {
        return encounter_rand_float(&s->rng_state) < osrs_hit_chance_double(att_roll, def_roll);
    }

    return encounter_rand_float(&s->rng_state) < osrs_hit_chance(att_roll, def_roll);
}

static void zul_player_attack(ZulrahState* s, int is_mage) {
    if (!s->zulrah_visible || s->is_diving) return;
    if (s->player.attack_timer > 0) return;
    if (s->player_stunned_ticks > 0) return;

    const EncounterLoadoutStats* ls = zul_current_loadout_stats(s, is_mage);
    int gear_ok = s->human_command_mode
        ? ((is_mage && ls->style == ATTACK_STYLE_MAGIC) ||
           (!is_mage && ls->style == ATTACK_STYLE_RANGED))
        : ((is_mage && s->player_gear == ZUL_GEAR_MAGE) ||
           (!is_mage && s->player_gear == ZUL_GEAR_RANGE));
    const MonsterStats* monster = &MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[s->current_form]];
    OsrsMagicAttackKind magic_kind = is_mage ? OSRS_MAGIC_ATTACK_POWERED_STAFF : OSRS_MAGIC_ATTACK_NONE;
    OsrsPreparedAttackEffects attack_effects = osrs_prepare_attack_effects(
        &s->player.equipment_effect_profile,
        &s->player.item_effect_state,
        s->player.equipped[GEAR_SLOT_WEAPON],
        is_mage ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED,
        magic_kind,
        (OsrsTargetRef){ .kind = OSRS_TARGET_NPC, .id = 0 },
        1,
        osrs_player_att_roll(ls->eff_level, ls->attack_bonus),
        ls->max_hit,
        monster->magic_level,
        monster->magic_att_bonus,
        s->player.current_hitpoints,
        s->player.base_hitpoints
    );
    s->player.attack_timer = is_mage ? 4 : ls->attack_speed;
    if (!gear_ok) return;

    int dmg = 0;
    int hit = zul_player_attack_hits(s, is_mage, &attack_effects);
    if (hit) {
        dmg = encounter_rand_int(&s->rng_state, attack_effects.max_hit + 1);
        dmg = zul_cap_damage(s, dmg);
        encounter_damage_player(&s->zulrah, dmg, &s->damage_dealt_this_tick);
        s->total_damage_dealt += dmg;
    }
    {
        OsrsPostAttackEffects post_effects = osrs_finalize_attack_effects(
            &s->player.equipment_effect_profile,
            &s->player.item_effect_state,
            s->player.equipped[GEAR_SLOT_WEAPON],
            is_mage ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED,
            magic_kind,
            (OsrsTargetRef){ .kind = OSRS_TARGET_NPC, .id = 0 },
            1,
            attack_effects.use_double_accuracy,
            hit,
            dmg,
            &s->rng_state
        );
        if (post_effects.heal_amount > 0) {
            s->player.current_hitpoints += post_effects.heal_amount;
            if (s->player.current_hitpoints > s->player.base_hitpoints)
                s->player.current_hitpoints = s->player.base_hitpoints;
        }
    }
    s->player.just_attacked = 1;
    s->player.last_attack_style = is_mage ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED;
    s->player.attack_style_this_tick = is_mage ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED;

    /* visual: hit splat + HP bar on Zulrah */
    s->zulrah.hit_landed_this_tick = 1;
    s->zulrah.hit_damage = dmg;
    s->zulrah.hit_was_successful = (dmg > 0);
}

/* special attack: dispatched via osrs_resolve_spec based on equipped weapon.
   tier 0: MSB(i) from ranged gear. tier 1: bowfa has no spec (osrs_spec_cost=0).
   tier 2: eye of ayak from mage gear. */
static void zul_player_spec(ZulrahState* s) {
    if (!s->zulrah_visible || s->is_diving) return;
    if (s->player.attack_timer > 0) return;
    if (s->player_stunned_ticks > 0) return;

    int is_mage = (s->player_gear == ZUL_GEAR_MAGE);
    const EncounterLoadoutStats* ls = zul_current_loadout_stats(s, is_mage);
    int weapon = s->human_command_mode
        ? s->player.equipped[GEAR_SLOT_WEAPON]
        : (is_mage
            ? ZUL_MAGE_LOADOUT[s->gear_tier][GEAR_SLOT_WEAPON]
            : ZUL_RANGE_LOADOUT[s->gear_tier][GEAR_SLOT_WEAPON]);

    int cost = osrs_spec_cost(weapon);
    if (cost == 0) return;  /* weapon has no spec (e.g. bowfa) */
    if (s->player.special_energy < cost) return;

    /* compute defence roll for current form */
    const MonsterStats* m = &MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[s->current_form]];
    int def_bonus = is_mage ? (m->magic_def - s->magic_def_drain) : m->ranged_def;
    if (is_mage && def_bonus < -64) def_bonus = -64;
    int def_roll = (m->def_level + 8) * (def_bonus + 64);
    if (def_roll < 0) def_roll = 0;

    int att_roll = osrs_player_att_roll(ls->eff_level, ls->attack_bonus);
    SpecResult sr = osrs_resolve_spec(weapon, att_roll, ls->max_hit,
                                       def_roll, m->def_level, &s->rng_state);

    s->player.special_energy -= sr.spec_cost;
    s->player.special_energy = s->player.special_energy;
    s->player.just_attacked = 1;
    s->player.used_special_this_tick = 1;
    s->player.last_attack_style = is_mage ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED;
    s->player.attack_style_this_tick = s->player.last_attack_style;
    s->player.attack_timer = sr.attack_speed_override ? sr.attack_speed_override : ls->attack_speed;

    /* apply damage with per-hit capping */
    int total_dmg = 0;
    for (int i = 0; i < sr.num_hits; i++) {
        int dmg = zul_cap_damage(s, sr.damage[i]);
        encounter_damage_player(&s->zulrah, dmg, NULL);
        total_dmg += dmg;
    }

    /* apply heal (blowpipe, SGS) */
    if (sr.heal > 0) {
        s->player.current_hitpoints += sr.heal;
        if (s->player.current_hitpoints > s->player.base_hitpoints)
            s->player.current_hitpoints = s->player.base_hitpoints;
    }

    /* apply magic def drain (eye of ayak) */
    s->magic_def_drain += sr.magic_def_drain;

    s->damage_dealt_this_tick += total_dmg;
    s->total_damage_dealt += total_dmg;
    s->zulrah.hit_landed_this_tick = 1;
    s->zulrah.hit_damage = total_dmg;
    s->zulrah.hit_was_successful = (total_dmg > 0);
}

/* ======================================================================== */
/* snakelings                                                                */
/* ======================================================================== */

/* pick a walkable spawn position for a snakeling, falling back to player's tile */
static void zul_pick_snakeling_pos(ZulrahState* s, int* ox, int* oy) {
    /* try predefined positions in random order */
    int order[ZUL_NUM_SNAKELING_POSITIONS];
    for (int i = 0; i < ZUL_NUM_SNAKELING_POSITIONS; i++) order[i] = i;
    encounter_shuffle(order, ZUL_NUM_SNAKELING_POSITIONS, &s->rng_state);
    for (int i = 0; i < ZUL_NUM_SNAKELING_POSITIONS; i++) {
        int px = ZUL_SNAKELING_POSITIONS[order[i]][0];
        int py = ZUL_SNAKELING_POSITIONS[order[i]][1];
        if (zul_on_platform(s, px, py) &&
            !(px == s->player.x && py == s->player.y)) {
            *ox = px; *oy = py; return;
        }
    }
    /* fallback: spawn near player */
    *ox = s->player.x;
    *oy = s->player.y;
}

static void zul_spawn_snakeling(ZulrahState* s) {
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++) {
        if (s->snakelings[i].active) continue;
        ZulrahSnakeling* sn = &s->snakelings[i];
        memset(sn, 0, sizeof(ZulrahSnakeling));
        sn->active = 1;
        sn->entity.entity_type = ENTITY_NPC;
        sn->entity.npc_size = 1;
        sn->entity.npc_visible = 1;
        sn->is_magic = encounter_rand_int(&s->rng_state, 2);
        sn->entity.npc_def_id = sn->is_magic ? 2046 : 2045;
        sn->entity.npc_anim_id = SNAKELING_ANIM_IDLE;
        zul_pick_snakeling_pos(s, &sn->entity.x, &sn->entity.y);
        sn->entity.current_hitpoints = ZUL_SNAKELING_HP;
        sn->entity.base_hitpoints = ZUL_SNAKELING_HP;
        sn->attack_timer = ZUL_SNAKELING_SPEED;
        sn->lifespan = ZUL_SNAKELING_LIFESPAN;

        /* emit spawn orb projectile event (style=4) */
        if (s->attack_event_count < 8) {
            int ei = s->attack_event_count++;
            s->attack_events[ei].src_x = s->zulrah.x;
            s->attack_events[ei].src_y = s->zulrah.y;
            s->attack_events[ei].dst_x = sn->entity.x;
            s->attack_events[ei].dst_y = sn->entity.y;
            s->attack_events[ei].style = 4;  /* snakeling spawn orb */
            s->attack_events[ei].damage = 0;
        }
        return;
    }
}

static void zul_snakeling_tick(ZulrahState* s) {
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++) {
        ZulrahSnakeling* sn = &s->snakelings[i];
        if (!sn->active) continue;

        /* lifespan: die after ~40 seconds */
        sn->lifespan--;
        if (sn->lifespan <= 0) { sn->active = 0; continue; }

        /* move toward player — stop when within attack range (Chebyshev ≤ 1).
           NPCs use greedy single-step movement, not full BFS. */
        int adx = abs_int(sn->entity.x - s->player.x);
        int ady = abs_int(sn->entity.y - s->player.y);
        int in_range = (adx <= 1 && ady <= 1);
        int moved = 0;
        if (!in_range) {
            PathResult pr = zul_pathfind(s, sn->entity.x, sn->entity.y,
                                          s->player.x, s->player.y);
            if (pr.found && (pr.next_dx != 0 || pr.next_dy != 0)) {
                int nx = sn->entity.x + pr.next_dx;
                int ny = sn->entity.y + pr.next_dy;
                if (zul_on_platform(s, nx, ny)) {
                    sn->entity.x = nx; sn->entity.y = ny; moved = 1;
                }
            }
        }
        sn->entity.npc_anim_id = moved ? SNAKELING_ANIM_WALK : SNAKELING_ANIM_IDLE;

        /* attack — recheck range after movement */
        if (sn->attack_timer > 0) { sn->attack_timer--; continue; }
        adx = abs_int(sn->entity.x - s->player.x);
        ady = abs_int(sn->entity.y - s->player.y);
        if (adx > 1 || ady > 1) continue;

        sn->attack_timer = ZUL_SNAKELING_SPEED;
        sn->entity.npc_anim_id = sn->is_magic ? SNAKELING_ANIM_MAGIC : SNAKELING_ANIM_MELEE;
        AttackStyle sn_style = sn->is_magic ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_MELEE;
        if (encounter_prayer_correct_for_style(s->player.prayer, sn_style)) {
            s->prayer_blocked_this_tick = 1; continue;
        }
        int sn_max = sn->is_magic ? MONSTER_DATABASE[MON_ZULRAH_SNAKELING_MAGIC].max_hit
                                   : MONSTER_DATABASE[MON_ZULRAH_SNAKELING_MELEE].max_hit;
        int dmg = encounter_rand_int(&s->rng_state, sn_max + 1);
        AttackStyle st = sn->is_magic ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_MELEE;
        zul_apply_player_damage(s, dmg, st, &sn->entity);

        /* recoil may have killed the snakeling — check and deactivate */
        if (sn->entity.current_hitpoints <= 0) {
            sn->entity.current_hitpoints = 0;
            sn->active = 0;
        }
    }
}

/* ======================================================================== */
/* clouds                                                                    */
/* ======================================================================== */

/* forward: needed by zul_spawn_cloud to get current phase's safe tiles */
static const ZulRotationPhase* zul_current_phase(ZulrahState* s) {
    return &ZUL_ROTATIONS[s->rotation_index][s->phase_index];
}

/* check if a 3x3 cloud area starting at (x,y) is fully within walkable arena.
   all 9 tiles must be walkable so the cloud doesn't extend outside the platform. */
static int zul_cloud_fits(ZulrahState* s, int x, int y) {
    for (int dx = 0; dx < ZUL_CLOUD_SIZE; dx++) {
        for (int dy = 0; dy < ZUL_CLOUD_SIZE; dy++) {
            if (!zul_on_platform(s, x + dx, y + dy)) return 0;
        }
    }
    return 1;
}

/* pick a valid cloud position: walkable 3x3, not safe, not overlapping.
 * known sim gap: positions are random; real game uses phase-specific targeting. */
static int zul_pick_cloud_pos(ZulrahState* s, int stand, int stall, int* ox, int* oy) {
    int attempts = 0;
    while (attempts++ < 100) {
        int x = ZUL_PLATFORM_MIN + encounter_rand_int(&s->rng_state, ZUL_PLATFORM_MAX - ZUL_PLATFORM_MIN + 1);
        int y = ZUL_PLATFORM_MIN + encounter_rand_int(&s->rng_state, ZUL_PLATFORM_MAX - ZUL_PLATFORM_MIN + 1);

        if (!zul_cloud_fits(s, x, y)) continue;
        if (zul_tile_is_safe(x, y, stand, stall)) continue;

        /* check 3x3 bounding box overlap with active and pending clouds.
           two 3x3 clouds overlap if their anchor tiles are within 2 in each axis. */
        int overlap = 0;
        for (int j = 0; j < ZUL_MAX_CLOUDS && !overlap; j++) {
            if (s->clouds[j].active &&
                abs(s->clouds[j].x - x) < ZUL_CLOUD_SIZE &&
                abs(s->clouds[j].y - y) < ZUL_CLOUD_SIZE)
                overlap = 1;
        }
        for (int j = 0; j < ZUL_MAX_PENDING_CLOUDS && !overlap; j++) {
            if (s->pending_clouds[j].delay > 0 &&
                abs(s->pending_clouds[j].x - x) < ZUL_CLOUD_SIZE &&
                abs(s->pending_clouds[j].y - y) < ZUL_CLOUD_SIZE)
                overlap = 1;
        }
        if (overlap) continue;

        *ox = x; *oy = y;
        return 1;
    }
    return 0;
}

/* queue a pending cloud with a flight delay */
static void zul_queue_pending_cloud(ZulrahState* s, int x, int y, int delay) {
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++) {
        if (s->pending_clouds[i].delay <= 0) {
            s->pending_clouds[i].x = x;
            s->pending_clouds[i].y = y;
            s->pending_clouds[i].delay = delay;
            return;
        }
    }
}

/* activate a pending cloud into the first free cloud slot */
static void zul_activate_cloud(ZulrahState* s, int x, int y) {
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++) {
        if (!s->clouds[i].active) {
            s->clouds[i].x = x;
            s->clouds[i].y = y;
            s->clouds[i].active = 1;
            s->clouds[i].ticks_remaining = ZUL_CLOUD_DURATION;
            return;
        }
    }
    /* all slots full — cloud doesn't spawn (observed 7-cloud cap) */
}

/* emit a cloud projectile event for the renderer */
static void zul_emit_cloud_event(ZulrahState* s, int dst_x, int dst_y, int flight_ticks) {
    if (s->cloud_event_count >= 4) return;
    int i = s->cloud_event_count++;
    s->cloud_events[i].src_x = s->zulrah.x;
    s->cloud_events[i].src_y = s->zulrah.y;
    s->cloud_events[i].dst_x = dst_x;
    s->cloud_events[i].dst_y = dst_y;
    s->cloud_events[i].flight_ticks = flight_ticks;
}

/* spit: pick 2 positions now, queue them with staggered flight times */
static void zul_spawn_cloud(ZulrahState* s) {
    const ZulRotationPhase* phase = zul_current_phase(s);
    int stand = phase->stand;
    int stall = phase->stall;
    int x, y;
    if (zul_pick_cloud_pos(s, stand, stall, &x, &y)) {
        zul_queue_pending_cloud(s, x, y, ZUL_CLOUD_FLIGHT_1);
        zul_emit_cloud_event(s, x, y, ZUL_CLOUD_FLIGHT_1);
    }
    if (zul_pick_cloud_pos(s, stand, stall, &x, &y)) {
        zul_queue_pending_cloud(s, x, y, ZUL_CLOUD_FLIGHT_2);
        zul_emit_cloud_event(s, x, y, ZUL_CLOUD_FLIGHT_2);
    }
}

/* tick pending clouds: decrement delay, activate when ready */
static void zul_pending_cloud_tick(ZulrahState* s) {
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++) {
        if (s->pending_clouds[i].delay <= 0) continue;
        s->pending_clouds[i].delay--;
        if (s->pending_clouds[i].delay <= 0) {
            zul_activate_cloud(s, s->pending_clouds[i].x, s->pending_clouds[i].y);
        }
    }
}

static void zul_cloud_tick(ZulrahState* s) {
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++) {
        if (!s->clouds[i].active) continue;
        s->clouds[i].ticks_remaining--;
        if (s->clouds[i].ticks_remaining <= 0) { s->clouds[i].active = 0; continue; }

        /* wiki: "varying damage per tick" if player in 3x3 area */
        if (zul_player_in_cloud(s->clouds[i].x, s->clouds[i].y,
                                s->player.x, s->player.y)) {
            int dmg = ZUL_CLOUD_DAMAGE_MIN +
                      encounter_rand_int(&s->rng_state, ZUL_CLOUD_DAMAGE_MAX - ZUL_CLOUD_DAMAGE_MIN + 1);
            zul_apply_player_damage(s, dmg, ATTACK_STYLE_MAGIC, NULL);
        }
    }
}

/* ======================================================================== */
/* venom                                                                     */
/* ======================================================================== */

static void zul_venom_tick(ZulrahState* s) {
    /* antivenom timer ticks down */
    if (s->antivenom_timer > 0) s->antivenom_timer--;

    if (s->venom_counter == 0) return;
    if (s->antivenom_timer > 0) return;  /* immune while antivenom active */
    s->total_venom_ticks++;
    if (s->venom_timer > 0) { s->venom_timer--; return; }
    int dmg = ZUL_VENOM_START + 2 * (s->venom_counter - 1);
    if (dmg > ZUL_VENOM_MAX) dmg = ZUL_VENOM_MAX;
    zul_apply_player_damage(s, dmg, ATTACK_STYLE_MAGIC, NULL);
    s->venom_counter++;
    s->venom_timer = ZUL_VENOM_INTERVAL;
}

/* ======================================================================== */
/* thrall: arceuus greater ghost                                             */
/* ======================================================================== */

static void zul_thrall_tick(ZulrahState* s) {
    if (!s->thrall_active) {
        /* resummon after cooldown */
        if (s->thrall_cooldown > 0) { s->thrall_cooldown--; return; }
        s->thrall_active = 1;
        s->thrall_duration_remaining = ZUL_THRALL_DURATION;
        s->thrall_attack_timer = 1;  /* attacks on next tick */
        return;
    }

    s->thrall_duration_remaining--;
    if (s->thrall_duration_remaining <= 0) {
        /* despawn + cooldown before resummon */
        s->thrall_active = 0;
        s->thrall_cooldown = ZUL_THRALL_COOLDOWN;
        return;
    }

    /* attack: always hits, ignores armour, only when zulrah is targetable */
    if (s->thrall_attack_timer > 0) { s->thrall_attack_timer--; return; }
    s->thrall_attack_timer = ZUL_THRALL_SPEED;

    if (!s->zulrah_visible || s->is_diving) return;

    int dmg = encounter_rand_int(&s->rng_state, ZUL_THRALL_MAX_HIT + 1);
    dmg = zul_cap_damage(s, dmg);
    encounter_damage_player(&s->zulrah, dmg, &s->damage_dealt_this_tick);
    s->total_damage_dealt += dmg;
}

/* ======================================================================== */
/* phase machine: execute current action in rotation table                   */
/* ======================================================================== */

/* fire one instance of the current action (attack/cloud/snakeling) */
static void zul_fire_action(ZulrahState* s, ZulActionType type) {
    switch (type) {
        case ZA_RANGED:        zul_attack_ranged(s); break;
        case ZA_MAGIC_RANGED:  zul_attack_magic_ranged(s); break;
        case ZA_MELEE:         zul_melee_start(s); break;
        case ZA_JAD_RM:
        case ZA_JAD_MR:        zul_attack_jad(s); break;
        case ZA_CLOUDS:        zul_spawn_cloud(s); break;
        case ZA_SNAKELINGS:    zul_spawn_snakeling(s); break;
        case ZA_SNAKECLOUD_ALT:
            if (s->action_progress % 2 == 0) zul_spawn_snakeling(s);
            else zul_spawn_cloud(s);
            break;
        case ZA_CLOUDSNAKE_ALT:
            if (s->action_progress % 2 == 0) zul_spawn_cloud(s);
            else zul_spawn_snakeling(s);
            break;
        case ZA_END: break;
    }
}

/* get ticks between fires for an action type */
static int zul_action_interval(ZulActionType type) {
    switch (type) {
        case ZA_RANGED:
        case ZA_MAGIC_RANGED:
        case ZA_JAD_RM:
        case ZA_JAD_MR:        return MONSTER_DATABASE[MON_ZULRAH_GREEN].attack_speed;
        case ZA_MELEE:         return ZUL_MELEE_INTERVAL;
        case ZA_CLOUDS:
        case ZA_SNAKELINGS:
        case ZA_SNAKECLOUD_ALT:
        case ZA_CLOUDSNAKE_ALT: return ZUL_SPAWN_INTERVAL;
        default:               return 1;
    }
}

/* is this action type an attack (vs spawn)? */
static int zul_action_is_attack(ZulActionType type) {
    return type == ZA_RANGED || type == ZA_MAGIC_RANGED || type == ZA_MELEE ||
           type == ZA_JAD_RM || type == ZA_JAD_MR;
}

/* compute total ticks needed for all actions in a phase */
static int zul_phase_action_ticks(const ZulRotationPhase* phase) {
    int total = 0;
    for (int i = 0; i < ZUL_MAX_PHASE_ACTIONS; i++) {
        if (phase->actions[i].type == ZA_END) break;
        total += phase->actions[i].count * zul_action_interval((ZulActionType)phase->actions[i].type);
    }
    return total;
}

/* start a new phase: set form, position, reset action tracking.
   surface animation plays for first N ticks before actions begin.
   initial action delay is computed so actions + dive fill the remaining window. */
static void zul_enter_phase(ZulrahState* s) {
    const ZulRotationPhase* phase = zul_current_phase(s);
    s->current_form = (ZulrahForm)phase->form;
    s->zulrah.npc_def_id = zul_form_npc_id(s->current_form);
    s->zulrah.x = ZUL_POSITIONS[phase->position][0];
    s->zulrah.y = ZUL_POSITIONS[phase->position][1];
    s->zulrah_visible = 1;
    s->zulrah.npc_visible = 1;
    s->is_diving = 0;

    /* surface animation: initial rise is longer (3 ticks) than subsequent (2 ticks) */
    int is_initial = (s->phase_index == 0 && s->tick <= 1);
    s->zulrah.npc_anim_id = is_initial ? ZULRAH_ANIM_SURFACE : ZULRAH_ANIM_RISE;
    int surface_ticks = is_initial ? ZUL_SURFACE_TICKS_INITIAL : ZUL_SURFACE_TICKS;
    s->surface_timer = surface_ticks;

    s->phase_timer = phase->phase_ticks; /* total phase duration incl. surface + dive */

    /* compute initial action delay: fill the idle window between surface and first action.
       available = phase_ticks - surface - dive - action_ticks. first action fires after delay. */
    int action_ticks = zul_phase_action_ticks(phase);
    int available = phase->phase_ticks - surface_ticks - ZUL_DIVE_ANIM_TICKS - action_ticks;
    int initial_delay = (available > 1) ? available : 1;

    s->action_index = 0;
    s->action_progress = 0;
    s->action_timer = initial_delay;

    /* jad init */
    ZulActionType first_type = (ZulActionType)phase->actions[0].type;
    if (first_type == ZA_JAD_RM) s->jad_is_magic_next = 0;
    else if (first_type == ZA_JAD_MR) s->jad_is_magic_next = 1;

    /* not attacking during surface animation */
    s->zulrah_attacking = 0;
}

/* enter dive visual state — called when phase_timer reaches ZUL_DIVE_ANIM_TICKS.
   Zulrah stays visible playing dig anim for these last ticks of the phase. */
static void zul_enter_dive(ZulrahState* s) {
    s->is_diving = 1;
    s->zulrah_attacking = 0;
    s->zulrah.npc_anim_id = ZULRAH_ANIM_DIVE;
}

/* advance to next phase after dive completes */
static void zul_next_phase(ZulrahState* s) {
    int rot_len = ZUL_ROT_LENGTHS[s->rotation_index];
    s->phase_index++;
    s->total_phases_completed++;

    if (s->phase_index >= rot_len) {
        /* rotation complete — pick new random rotation, start from phase 1.
           the last phase already did ranged+clouds which counts as phase 1
           of the next rotation, so we skip to phase 1 (index 1). */
        s->rotation_index = encounter_rand_int(&s->rng_state, ZUL_NUM_ROTATIONS);
        s->phase_index = 1; /* skip cloud-only phase 1 since last phase covered it */
    }

    zul_enter_phase(s);
}

/* tick the phase machine.
   phase_timer is the single source of truth — covers surface + actions + dive.
   timeline: [surface_timer ticks] [actions] [idle] [ZUL_DIVE_ANIM_TICKS dive] → next phase */
static void zul_phase_tick(ZulrahState* s) {
    if (!s->zulrah_visible) return;

    /* decrement phase timer every tick */
    if (s->phase_timer > 0) s->phase_timer--;

    /* phase complete — immediately enter next phase */
    if (s->phase_timer <= 0) {
        s->zulrah_visible = 0;
        s->zulrah.npc_visible = 0;
        zul_next_phase(s);
        return;
    }

    /* dive animation: last N ticks of the phase */
    if (s->phase_timer <= ZUL_DIVE_ANIM_TICKS && !s->is_diving) {
        zul_enter_dive(s);
    }
    if (s->is_diving) return;

    /* surface animation: first N ticks of the phase, no actions fire.
       player CAN attack during this window (free hits). */
    if (s->surface_timer > 0) {
        s->surface_timer--;
        return;
    }

    /* active period: process actions */
    const ZulRotationPhase* phase = zul_current_phase(s);
    const ZulAction* act = &phase->actions[s->action_index];

    /* end sentinel — all actions done, idle until dive kicks in */
    if (act->type == ZA_END) {
        s->zulrah_attacking = 0;
        return;
    }

    /* wait for action timer */
    s->action_timer--;
    if (s->action_timer > 0) return;

    /* fire the action */
    zul_fire_action(s, (ZulActionType)act->type);
    s->action_progress++;

    /* check if this action segment is complete */
    if (s->action_progress >= act->count) {
        s->action_index++;
        s->action_progress = 0;

        /* check if next action exists */
        const ZulAction* next = &phase->actions[s->action_index];
        if (next->type == ZA_END) {
            s->zulrah_attacking = 0;
            return;
        }

        /* set attacking flag */
        s->zulrah_attacking = zul_action_is_attack((ZulActionType)next->type);

        /* jad init for new action */
        if (next->type == ZA_JAD_RM) s->jad_is_magic_next = 0;
        else if (next->type == ZA_JAD_MR) s->jad_is_magic_next = 1;

        s->action_timer = zul_action_interval((ZulActionType)next->type);
    } else {
        s->action_timer = zul_action_interval((ZulActionType)act->type);
    }
}

/* ======================================================================== */
/* player action processing                                                  */
/* ======================================================================== */

static void zul_process_movement(ZulrahState* s) {
    if (s->player_dest_x < 0 || s->player_dest_y < 0) return;
    if (s->player_stunned_ticks > 0) return;

    /* shared BFS click-to-move: runs (2 steps) when dest > 1 tile away */
    encounter_move_toward_dest(&s->player, &s->player_dest_x, &s->player_dest_y,
        (const CollisionMap*)s->collision_map, s->world_offset_x, s->world_offset_y,
        zul_tile_walkable, s, NULL, NULL,
        0, 0, 0, 0);
}

static void zul_process_prayer(ZulrahState* s, int overhead_action, int offensive_action) {
    if (encounter_apply_overhead_action(&s->player.prayer, overhead_action)) {
        s->player.prayer_just_activated = 1;
    }
    OffensivePrayer prev_offensive = s->player.offensive_prayer;
    if (encounter_apply_offensive_action(&s->player.offensive_prayer, offensive_action)) {
        s->player.offensive_prayer_just_activated = 1;
    }
    /* zulrah caches eff_level + max_hit in mage_stats / range_stats — recompute
       whenever offensive prayer changes so subsequent attack rolls use current state. */
    if (s->player.offensive_prayer != prev_offensive) {
        if (s->mage_stats.style == ATTACK_STYLE_MAGIC) {
            encounter_update_loadout_level(&s->mage_stats, s->player.offensive_prayer,
                s->player.current_magic, s->player.current_magic);
        }
        if (s->range_stats.style == ATTACK_STYLE_RANGED) {
            encounter_update_loadout_level(&s->range_stats, s->player.offensive_prayer,
                s->player.current_ranged, s->player.current_ranged);
        }
        if (s->human_command_mode)
            zul_refresh_human_loadout_stats(s);
    }
}

static void zul_process_food(ZulrahState* s, int a) {
    if (a == 0) return;
    FoodType type = (a == 1) ? FOOD_SHARK : FOOD_KARAMBWAN;
    int* count = (a == 1) ? &s->player.food_count : &s->player.karambwan_count;
    if (*count <= 0) return;
    EatResult r = osrs_eat_food(type, s->player.current_hitpoints,
                                 s->player.base_hitpoints, s->player.food_timer);
    if (!r.consumed) return;
    (*count)--;
    s->player.food_timer = 3;
    s->player.current_hitpoints += r.hp_healed;
    if (s->player.current_hitpoints > s->player.base_hitpoints)
        s->player.current_hitpoints = s->player.base_hitpoints;
    s->total_food_eaten++;
}

static void zul_process_potion(ZulrahState* s, int a) {
    if (a == 0) return;
    if (a == 1) {
        /* prayer potion */
        if (s->player.prayer_pot_doses <= 0) return;
        DrinkResult r = osrs_drink_potion(POTION_PRAYER_RESTORE, s->player.current_prayer,
                                           s->player.base_prayer, s->player.potion_timer);
        if (!r.consumed) return;
        s->player.prayer_pot_doses--;
        s->player.potion_timer = 3;
        s->player.current_prayer += r.prayer_restored;
        if (s->player.current_prayer > s->player.base_prayer)
            s->player.current_prayer = s->player.base_prayer;
        s->total_potions_used++;
    } else if (a == 2) {
        /* antivenom: cures venom + grants immunity */
        if (s->player.antivenom_doses <= 0) return;
        DrinkResult r = osrs_drink_potion(POTION_ANTIVENOM_PLUS, 0, 0, s->player.potion_timer);
        if (!r.consumed) return;
        s->player.antivenom_doses--;
        s->player.potion_timer = 3;
        s->venom_counter = 0;
        s->venom_timer = 0;
        s->antivenom_timer = r.antivenom_ticks;
        s->total_potions_used++;
    }
}

static void zul_process_gear(ZulrahState* s, int atk) {
    if (atk == ZUL_ATK_MAGE && s->player_gear != ZUL_GEAR_MAGE) {
        s->player_gear = ZUL_GEAR_MAGE;
        encounter_apply_loadout(&s->player, ZUL_MAGE_LOADOUT[s->gear_tier], GEAR_MAGE);
        s->total_gear_switches++;
    } else if (atk == ZUL_ATK_RANGE && s->player_gear != ZUL_GEAR_RANGE) {
        s->player_gear = ZUL_GEAR_RANGE;
        encounter_apply_loadout(&s->player, ZUL_RANGE_LOADOUT[s->gear_tier], GEAR_RANGED);
        s->total_gear_switches++;
    }
}

static FightStyle zul_default_fight_style_for_style(AttackStyle style) {
    if (style == ATTACK_STYLE_MAGIC) return FIGHT_STYLE_ACCURATE;
    if (style == ATTACK_STYLE_RANGED) return FIGHT_STYLE_RAPID;
    return FIGHT_STYLE_ACCURATE;
}

static void zul_sync_human_gear_style(ZulrahState* s) {
    AttackStyle style = zul_player_equipped_attack_style(s);
    s->player_gear = (style == ATTACK_STYLE_MAGIC) ? ZUL_GEAR_MAGE : ZUL_GEAR_RANGE;
}

static void zul_apply_human_player_commands(ZulrahState* s) {
    int did_change_stats = 0;
    for (int i = 0; i < s->human_command_count; i++) {
        const HumanCommand* cmd = &s->human_commands[i];
        if (cmd->kind == HUMAN_COMMAND_EQUIP_INVENTORY_ITEM) {
            if (cmd->gear_slot >= 0 && cmd->gear_slot < NUM_GEAR_SLOTS &&
                cmd->item_db_idx >= 0 && cmd->item_db_idx < NUM_ITEMS) {
                int changed = slot_equip_item(&s->player, cmd->gear_slot, (uint8_t)cmd->item_db_idx);
                if (changed) {
                    s->total_gear_switches++;
                    did_change_stats = 1;
                    if (cmd->gear_slot == GEAR_SLOT_WEAPON) {
                        AttackStyle style = zul_player_equipped_attack_style(s);
                        s->player.fight_style = zul_default_fight_style_for_style(style);
                        zul_sync_human_gear_style(s);
                    }
                }
            }
        } else if (cmd->kind == HUMAN_COMMAND_FIGHT_STYLE) {
            if (cmd->fight_style >= FIGHT_STYLE_ACCURATE &&
                cmd->fight_style <= FIGHT_STYLE_DEFENSIVE_AUTOCAST) {
                s->player.fight_style = (FightStyle)cmd->fight_style;
                did_change_stats = 1;
            }
        }
    }
    if (did_change_stats)
        zul_refresh_human_loadout_stats(s);
}


/* ======================================================================== */
/* observations                                                              */
/* ======================================================================== */

static void zul_write_obs(EncounterState* state, float* obs) {
    ZulrahState* s = (ZulrahState*)state;
    memset(obs, 0, ZUL_NUM_OBS * sizeof(float));
    int i = 0;

    /* player (0-15) */
    obs[i++] = (float)s->player.current_hitpoints / s->player.base_hitpoints;
    obs[i++] = (float)s->player.current_prayer / s->player.base_prayer;
    /* egocentric: distance to arena edges, normalized to [0,1] */
    obs[i++] = (float)s->player.x / ZUL_ARENA_SIZE;             /* dist to west edge */
    obs[i++] = (float)s->player.y / ZUL_ARENA_SIZE;             /* dist to south edge */
    obs[i++] = (float)s->player.attack_timer / 5.0f;
    obs[i++] = (float)s->player.food_count / ZUL_PLAYER_FOOD;
    obs[i++] = (float)s->player.karambwan_count / ZUL_PLAYER_KARAMBWAN;
    obs[i++] = (float)s->player.prayer_pot_doses / ZUL_PLAYER_RESTORE_DOSES;
    obs[i++] = (float)s->player.food_timer / 3.0f;
    obs[i++] = (float)s->player.potion_timer / 3.0f;
    obs[i++] = (s->player_gear == ZUL_GEAR_MAGE) ? 1.0f : 0.0f;
    obs[i++] = (s->player_gear == ZUL_GEAR_RANGE) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    /* offensive prayer one-hot: piety, rigour, augury, (none implied by all three = 0) */
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_PIETY) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;
    obs[i++] = (float)s->player_stunned_ticks / ZUL_MELEE_STUN_TICKS;

    /* zulrah (16-29) */
    obs[i++] = (float)s->zulrah.current_hitpoints / MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;
    /* egocentric: zulrah position relative to player */
    obs[i++] = (float)(s->zulrah.x - s->player.x) / ZUL_ARENA_SIZE;
    obs[i++] = (float)(s->zulrah.y - s->player.y) / ZUL_ARENA_SIZE;
    obs[i++] = (s->current_form == ZUL_FORM_GREEN) ? 1.0f : 0.0f;
    obs[i++] = (s->current_form == ZUL_FORM_RED) ? 1.0f : 0.0f;
    obs[i++] = (s->current_form == ZUL_FORM_BLUE) ? 1.0f : 0.0f;
    obs[i++] = s->zulrah_visible ? 1.0f : 0.0f;
    obs[i++] = s->is_diving ? 1.0f : 0.0f;
    obs[i++] = s->zulrah_attacking ? 1.0f : 0.0f;
    obs[i++] = (float)s->action_timer / MONSTER_DATABASE[MON_ZULRAH_GREEN].attack_speed;
    obs[i++] = (float)encounter_dist_to_npc(s->player.x, s->player.y, s->zulrah.x, s->zulrah.y, ZUL_NPC_SIZE) / ZUL_ARENA_SIZE;
    obs[i++] = (float)s->rotation_index / (ZUL_NUM_ROTATIONS - 1);
    obs[i++] = (float)s->phase_index / 12.0f;
    obs[i++] = (s->melee_pending) ? 1.0f : 0.0f;

    /* venom (30-31) */
    obs[i++] = (s->venom_counter > 0) ? 1.0f : 0.0f;
    obs[i++] = (float)s->venom_timer / ZUL_VENOM_INTERVAL;

    /* clouds (32-52): 7 clouds * 3 */
    for (int c = 0; c < ZUL_MAX_CLOUDS; c++) {
        obs[i++] = s->clouds[c].active ? (float)(s->clouds[c].x - s->player.x) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = s->clouds[c].active ? (float)(s->clouds[c].y - s->player.y) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = s->clouds[c].active ? 1.0f : 0.0f;
    }

    /* snakelings (44-59): 4 * 4 */
    for (int n = 0; n < ZUL_MAX_SNAKELINGS; n++) {
        ZulrahSnakeling* sn = &s->snakelings[n];
        obs[i++] = sn->active ? (float)(sn->entity.x - s->player.x) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = sn->active ? (float)(sn->entity.y - s->player.y) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = sn->active ? 1.0f : 0.0f;
        obs[i++] = sn->active ? (float)chebyshev_distance(
            s->player.x, s->player.y, sn->entity.x, sn->entity.y) / ZUL_ARENA_SIZE : 0.0f;
    }

    /* meta (60-63) */
    obs[i++] = (float)s->tick / ZUL_MAX_TICKS;
    obs[i++] = s->damage_dealt_this_tick / 50.0f;
    obs[i++] = s->damage_received_this_tick / 50.0f;
    obs[i++] = s->total_damage_dealt / MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;

    /* new features (64-67) */
    obs[i++] = (float)s->player.special_energy / 100.0f;
    obs[i++] = (s->antivenom_timer > 0) ? 1.0f : 0.0f;
    obs[i++] = (float)s->antivenom_timer / ZUL_ANTIVENOM_DURATION;
    obs[i++] = (float)s->gear_tier / (ZUL_NUM_GEAR_TIERS - 1);

    /* safe tile positions for this phase (68-71) */
    const ZulRotationPhase* phase = zul_current_phase(s);
    if (phase->stand < ZUL_NUM_STAND_LOCATIONS) {
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stand][0] - s->player.x) / ZUL_ARENA_SIZE;
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stand][1] - s->player.y) / ZUL_ARENA_SIZE;
    } else {
        obs[i++] = 0.0f; obs[i++] = 0.0f;
    }
    if (phase->stall < ZUL_NUM_STAND_LOCATIONS) {
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stall][0] - s->player.x) / ZUL_ARENA_SIZE;
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stall][1] - s->player.y) / ZUL_ARENA_SIZE;
    } else {
        obs[i++] = 0.0f; obs[i++] = 0.0f;
    }

    while (i < ZUL_NUM_OBS) obs[i++] = 0.0f;
}

/* ======================================================================== */
/* action masks                                                              */
/* ======================================================================== */

static void zul_write_mask(EncounterState* state, float* mask) {
    ZulrahState* s = (ZulrahState*)state;
    for (int i = 0; i < ZUL_ACTION_MASK_SIZE; i++) mask[i] = 1.0f;
    int off = 0;

    /* movement: 25-action system (idle + 8 walk + 16 run) */
    for (int m = 0; m < ZUL_MOVE_DIM; m++) {
        if (m > 0) {
            if (s->player_stunned_ticks > 0) { mask[off] = 0.0f; }
            else {
                int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[m];
                int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[m];
                if (!zul_on_platform(s, nx, ny)) mask[off] = 0.0f;
            }
        }
        off++;
    }
    /* attack — can't attack while Zulrah is hidden or diving */
    for (int a = 0; a < ZUL_ATTACK_DIM; a++) {
        if (a > 0 && (!s->zulrah_visible || s->is_diving || s->player.attack_timer > 0 || s->player_stunned_ticks > 0))
            mask[off] = 0.0f;
        off++;
    }
    /* overhead prayer: 0=no_change always valid, 1-3=toggle_melee/ranged/magic
       require prayer points (deactivation also needs pp>0 because activations
       are what cost — if pp=0 all prayers auto-clear anyway). */
    for (int p = 0; p < ZUL_PRAYER_DIM; p++) {
        if (p >= ENCOUNTER_OVERHEAD_TOGGLE_MELEE && s->player.current_prayer <= 0)
            mask[off] = 0.0f;
        off++;
    }
    /* food (none=0, shark=1, karambwan=2) */
    off++;  /* none always valid */
    /* shark: masked if no food, food timer active, or would overheal (HP > 79) */
    if (s->player.food_count <= 0 || s->player.food_timer > 0 ||
        s->player.current_hitpoints > s->player.base_hitpoints - osrs_food_heal_amount(FOOD_SHARK))
        mask[off] = 0.0f;
    off++;
    /* karambwan: masked if no karambwan, food timer active, or would overheal (HP > 81) */
    if (s->player.karambwan_count <= 0 || s->player.food_timer > 0 ||
        s->player.current_hitpoints > s->player.base_hitpoints - osrs_food_heal_amount(FOOD_KARAMBWAN))
        mask[off] = 0.0f;
    off++;
    /* potion (none=0, prayer_pot=1, antivenom=2) */
    off++;  /* none always valid */
    /* prayer pot: masked if no doses, potion timer active, or prayer already full */
    if (s->player.prayer_pot_doses <= 0 || s->player.potion_timer > 0 ||
        s->player.current_prayer >= s->player.base_prayer)
        mask[off] = 0.0f;
    off++;
    /* antivenom: masked if no doses, potion timer active, or already immune */
    if (s->player.antivenom_doses <= 0 || s->player.potion_timer > 0 ||
        s->antivenom_timer > 0)
        mask[off] = 0.0f;
    off++;
    /* spec toggle: allow when player has a spec weapon with cost > 0 */
    off++;  /* none always valid */
    {
        int weapon = s->player.equipped[GEAR_SLOT_WEAPON];
        int spec_cost = osrs_spec_cost(weapon);
        if (spec_cost <= 0)
            mask[off] = 0.0f;  /* weapon has no spec */
    }
    off++;
    /* offensive prayer: 0=no_change always valid, 1-3=toggle_piety/rigour/augury
       require prayer points. */
    for (int o = 0; o < ZUL_OFFENSIVE_DIM; o++) {
        if (o >= ENCOUNTER_OFFENSIVE_TOGGLE_PIETY && s->player.current_prayer <= 0)
            mask[off] = 0.0f;
        off++;
    }
}

/* ======================================================================== */
/* reward                                                                    */
/* ======================================================================== */

static float zul_compute_reward(ZulrahState* s) {
    /* terminal: +1 kill, 0 death (forfeiting future rewards is the penalty) */
    if (s->episode_over)
        return (s->winner == 0) ? 1.0f : 0.0f;

    /* per-tick shaping (small signals to bootstrap learning).
     * rewards are clamped to [-1, 1] by the training backend,
     * so keep individual components well under that. */
    float r = 0.0f;

    /* damage dealt + correct attack style bonus (green/red -> mage, blue -> range) */
    if (s->damage_dealt_this_tick > 0.0f) {
        float norm_dmg = s->damage_dealt_this_tick / 50.0f;
        r += 0.02f * norm_dmg;
        int correct = (s->current_form == ZUL_FORM_BLUE &&
                       s->player.attack_style_this_tick == ATTACK_STYLE_RANGED) ||
                      ((s->current_form == ZUL_FORM_GREEN ||
                        s->current_form == ZUL_FORM_RED) &&
                       s->player.attack_style_this_tick == ATTACK_STYLE_MAGIC);
        if (correct) r += 0.05f * norm_dmg;
    }

    /* damage taken penalty */
    if (s->damage_received_this_tick > 0.0f)
        r -= 0.01f * (s->damage_received_this_tick / 50.0f);

    return r;
}

/* ======================================================================== */
/* lifecycle                                                                 */
/* ======================================================================== */

static EncounterState* zul_create(void) {
    return (EncounterState*)calloc(1, sizeof(ZulrahState));
}

static void zul_destroy(EncounterState* state) { free(state); }

static void zul_reset(EncounterState* state, uint32_t seed) {
    ZulrahState* s = (ZulrahState*)state;
    Log saved_log = s->log;
    void* saved_cmap = s->collision_map;
    int saved_wx = s->world_offset_x;
    int saved_wy = s->world_offset_y;
    int saved_tier = s->gear_tier;
    uint32_t saved_rng = s->rng_state;
    memset(s, 0, sizeof(ZulrahState));
    s->log = saved_log;
    s->collision_map = saved_cmap;
    s->world_offset_x = saved_wx;
    s->world_offset_y = saved_wy;
    s->gear_tier = saved_tier;
    s->rng_state = encounter_resolve_seed(saved_rng, seed);

    /* player */
    s->player.entity_type = ENTITY_PLAYER;
    memset(s->player.equipped, ITEM_NONE, NUM_GEAR_SLOTS);
    s->player.base_hitpoints = ZUL_PLAYER_HP;
    s->player.current_hitpoints = ZUL_PLAYER_HP;
    s->player.base_prayer = ZUL_PLAYER_PRAYER;
    s->player.current_prayer = ZUL_PLAYER_PRAYER;
    s->player.x = ZUL_PLAYER_START_X;
    s->player.y = ZUL_PLAYER_START_Y;
    s->player.food_count = ZUL_PLAYER_FOOD;
    s->player.karambwan_count = ZUL_PLAYER_KARAMBWAN;
    s->player.prayer_pot_doses = ZUL_PLAYER_RESTORE_DOSES;
    s->player.special_energy = 100;
    s->player.antivenom_doses = ZUL_ANTIVENOM_DOSES;
    /* thrall: tier 1+ only (budget gear doesn't have arceuus access) */
    if (s->gear_tier >= 1) {
        s->thrall_active = 1;
        s->thrall_duration_remaining = ZUL_THRALL_DURATION;
        s->thrall_attack_timer = ZUL_THRALL_SPEED;
    }
    osrs_interaction_init(&s->interaction);
    s->player.spec_armed = 0;
    s->player_gear = ZUL_GEAR_MAGE;
    encounter_apply_loadout(&s->player, ZUL_MAGE_LOADOUT[s->gear_tier], GEAR_MAGE);
    zul_populate_player_inventory(&s->player, s->gear_tier);
    /* derive combat stats from ITEM_DATABASE */
    OffensivePrayer mage_prayer = (s->gear_tier >= 1) ? OFFENSIVE_PRAYER_AUGURY : OFFENSIVE_PRAYER_NONE;
    OffensivePrayer range_prayer = (s->gear_tier >= 1) ? OFFENSIVE_PRAYER_RIGOUR : OFFENSIVE_PRAYER_NONE;
    /* mage loadout is trident / Eye of Ayak — powered staves, accurate stance gets +3 magic eff.
       ranged runs in rapid stance for blowpipe/tbow (-1 attack_speed). */
    encounter_compute_loadout_stats(ZUL_MAGE_LOADOUT[s->gear_tier], ATTACK_STYLE_MAGIC,
        mage_prayer, 99, FIGHT_STYLE_ACCURATE, 30, &s->mage_stats);
    encounter_compute_loadout_stats(ZUL_RANGE_LOADOUT[s->gear_tier], ATTACK_STYLE_RANGED,
        range_prayer, 99, FIGHT_STYLE_RAPID, 0, &s->range_stats);
    /* zulrah */
    s->zulrah.entity_type = ENTITY_NPC;
    s->zulrah.npc_def_id = 2042;
    s->zulrah.npc_size = ZUL_NPC_SIZE;
    s->zulrah.npc_anim_id = ZULRAH_ANIM_IDLE;
    s->zulrah.base_hitpoints = MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;
    s->zulrah.current_hitpoints = MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;

    /* pick random rotation, start at phase 0 (cloud-only intro) */
    s->rotation_index = encounter_rand_int(&s->rng_state, ZUL_NUM_ROTATIONS);
    s->phase_index = 0;

    zul_enter_phase(s);
}

static void zul_step(EncounterState* state, const int* actions) {
    ZulrahState* s = (ZulrahState*)state;
    if (s->episode_over) return;

    s->reward = 0.0f;
    s->damage_dealt_this_tick = 0.0f;
    s->damage_received_this_tick = 0.0f;
    s->prayer_blocked_this_tick = 0;
    s->player.just_attacked = 0;
    s->player.hit_landed_this_tick = 0;
    s->player.attack_style_this_tick = ATTACK_STYLE_NONE;
    s->player.used_special_this_tick = 0;
    s->zulrah.hit_landed_this_tick = 0;
    s->attack_event_count = 0;
    s->cloud_event_count = 0;
    /* default to idle anim — but don't overwrite dive/surface animations */
    if (s->zulrah_visible && !s->is_diving && s->surface_timer <= 0)
        s->zulrah.npc_anim_id = ZULRAH_ANIM_IDLE;
    s->tick++;

    /* timers */
    if (s->player.attack_timer > 0) s->player.attack_timer--;
    if (s->player.food_timer > 0) s->player.food_timer--;
    if (s->player.potion_timer > 0) s->player.potion_timer--;
    if (s->player_stunned_ticks > 0) s->player_stunned_ticks--;

    /* pending melee hit */
    if (s->melee_pending) {
        s->melee_stare_timer--;
        if (s->melee_stare_timer <= 0) zul_melee_hit(s);
    }

    /* prayer doesn't interrupt interactions */
    zul_process_prayer(s, actions[ZUL_HEAD_PRAYER], actions[ZUL_HEAD_OFFENSIVE]);

    if (s->human_command_mode)
        zul_apply_human_player_commands(s);

    /* spec toggle: arm/disarm (does NOT interrupt interaction) */
    if (actions[ZUL_HEAD_SPEC] == 1) {
        osrs_spec_toggle(&s->player.spec_armed);
    }

    /* inventory actions interrupt interaction */
    if (actions[ZUL_HEAD_FOOD] > 0)
        osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_EAT);
    if (actions[ZUL_HEAD_POTION] > 0)
        osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_DRINK);
    zul_process_food(s, actions[ZUL_HEAD_FOOD]);
    zul_process_potion(s, actions[ZUL_HEAD_POTION]);

    /* gear switch from attack action — interrupts if actually switching */
    int atk_action = actions[ZUL_HEAD_ATTACK];
    if (!s->human_command_mode) {
        if ((atk_action == ZUL_ATK_MAGE && s->player_gear != ZUL_GEAR_MAGE) ||
            (atk_action == ZUL_ATK_RANGE && s->player_gear != ZUL_GEAR_RANGE)) {
            osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_EQUIP);
        }
        zul_process_gear(s, atk_action);
    }

    /* attack action sets interaction target (zulrah is always entity slot 0) */
    if (atk_action == ZUL_ATK_MAGE || atk_action == ZUL_ATK_RANGE) {
        osrs_interaction_set(&s->interaction, 0);
    }

    /* explicit movement interrupts interaction */
    int has_explicit_move = 0;
    if (s->player_dest_explicit) {
        s->player_dest_explicit = 0;
        has_explicit_move = 1;
    } else {
        int m = actions[ZUL_HEAD_MOVE];
        if (m > 0 && m < ZUL_MOVE_DIM) {
            s->player_dest_x = s->player.x + ENCOUNTER_MOVE_TARGET_DX[m];
            s->player_dest_y = s->player.y + ENCOUNTER_MOVE_TARGET_DY[m];
            has_explicit_move = 1;
        } else {
            s->player_dest_x = -1;
            s->player_dest_y = -1;
        }
    }
    if (has_explicit_move && !osrs_interaction_active(&s->interaction)) {
        /* pure movement, no target */
    }
    /* note: movement with active interaction = auto-chase */
    zul_process_movement(s);

    /* clear interaction if target not available */
    if (osrs_interaction_active(&s->interaction) &&
        (!s->zulrah_visible || s->is_diving)) {
        osrs_interaction_clear(&s->interaction);
    }

    /* auto-attack: fires when interaction is active + timer ready + target visible */
    if (osrs_interaction_active(&s->interaction) &&
        s->player.attack_timer == 0 && s->zulrah_visible && !s->is_diving &&
        s->player_stunned_ticks == 0) {

        if (s->player.spec_armed && s->player.special_energy >= osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON])) {
            zul_player_spec(s);
            osrs_spec_disarm(&s->player.spec_armed);
        } else {
            if (s->player_gear == ZUL_GEAR_MAGE) zul_player_attack(s, 1);
            else zul_player_attack(s, 0);
        }
    }

    if (s->zulrah.current_hitpoints <= 0) {
        s->episode_over = 1; s->winner = 0;
        s->reward = zul_compute_reward(s); s->episode_return += s->reward; return;
    }

    /* resolve pending cloud projectiles, then tick active clouds */
    zul_pending_cloud_tick(s);
    zul_cloud_tick(s);
    if (s->player.current_hitpoints <= 0) {
        s->episode_over = 1; s->winner = 1;
        s->reward = zul_compute_reward(s); s->episode_return += s->reward; return;
    }

    /* phase machine */
    zul_phase_tick(s);

    /* snakelings */
    zul_snakeling_tick(s);

    /* thrall (arceuus greater ghost) */
    zul_thrall_tick(s);

    /* venom */
    zul_venom_tick(s);

    /* prayer drain — both overhead and offensive, with activation-tick skip.
       drain can clear offensive_prayer at pp<=0; refresh mage/range caches
       afterwards so subsequent attacks don't use stale prayer-boosted stats. */
    OffensivePrayer prev_off_drain = s->player.offensive_prayer;
    encounter_drain_all_prayers(&s->player, 0);
    if (s->player.offensive_prayer != prev_off_drain) {
        if (s->mage_stats.style == ATTACK_STYLE_MAGIC) {
            encounter_update_loadout_level(&s->mage_stats, s->player.offensive_prayer,
                s->player.current_magic, s->player.current_magic);
        }
        if (s->range_stats.style == ATTACK_STYLE_RANGED) {
            encounter_update_loadout_level(&s->range_stats, s->player.offensive_prayer,
                s->player.current_ranged, s->player.current_ranged);
        }
        if (s->human_command_mode)
            zul_refresh_human_loadout_stats(s);
    }

    if (s->player.current_hitpoints <= 0) {
        s->episode_over = 1; s->winner = 1;
        s->reward = zul_compute_reward(s); s->episode_return += s->reward; return;
    }
    if (s->tick >= ZUL_MAX_TICKS) {
        s->episode_over = 1; s->winner = 1;
        s->reward = zul_compute_reward(s); s->episode_return += s->reward; return;
    }
    s->reward = zul_compute_reward(s);
    s->episode_return += s->reward;
}

/* ======================================================================== */
/* heuristic policy (for visual debug + sanity checks)                       */
/* ======================================================================== */

static void zul_heuristic_actions(ZulrahState* s, int* actions) {
    /* zero all heads */
    for (int i = 0; i < ZUL_NUM_ACTION_HEADS; i++) actions[i] = 0;

    int hp = s->player.current_hitpoints;

    /* prayer: match form. GREEN=ranged, BLUE=magic, RED=melee.
       heuristic picks target-prayer; toggle semantic means if target is already
       on this call is a no-op (since actual state == target), and if wrong one
       was on it replaces. only emit if player isn't already on target. */
    if (s->zulrah_visible && !s->is_diving) {
        switch (s->current_form) {
            case ZUL_FORM_GREEN:
                if (s->player.prayer != PRAYER_PROTECT_RANGED)
                    actions[ZUL_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_TOGGLE_RANGED;
                break;
            case ZUL_FORM_BLUE:
                if (s->player.prayer != PRAYER_PROTECT_MAGIC)
                    actions[ZUL_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_TOGGLE_MAGIC;
                break;
            case ZUL_FORM_RED:
                if (s->player.prayer != PRAYER_PROTECT_MELEE)
                    actions[ZUL_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_TOGGLE_MELEE;
                break;
        }
        /* offensive prayer: match attack style. green/blue use mage/ranged → piety is wrong.
           zulrah heuristic normally alternates mage/ranged so use augury when mage, rigour when ranged. */
        OffensivePrayer target_off = OFFENSIVE_PRAYER_NONE;
        if (s->current_form == ZUL_FORM_BLUE) target_off = OFFENSIVE_PRAYER_AUGURY;
        else if (s->current_form == ZUL_FORM_GREEN) target_off = OFFENSIVE_PRAYER_RIGOUR;
        else if (s->current_form == ZUL_FORM_RED) target_off = OFFENSIVE_PRAYER_PIETY;
        if (target_off != OFFENSIVE_PRAYER_NONE && s->player.offensive_prayer != target_off) {
            if (target_off == OFFENSIVE_PRAYER_AUGURY)
                actions[ZUL_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_TOGGLE_AUGURY;
            else if (target_off == OFFENSIVE_PRAYER_RIGOUR)
                actions[ZUL_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_TOGGLE_RIGOUR;
            else if (target_off == OFFENSIVE_PRAYER_PIETY)
                actions[ZUL_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_TOGGLE_PIETY;
        }
    }

    /* antivenom on first tick or when timer about to expire */
    if (s->player.potion_timer <= 0 && s->player.antivenom_doses > 0 &&
        s->antivenom_timer <= 5 && (s->tick <= 1 || s->antivenom_timer <= 5)) {
        actions[ZUL_HEAD_POTION] = 2;  /* antivenom */
        return;  /* potion consumes the tick */
    }

    /* eat shark at <60 HP (only if won't overheal) */
    if (hp < 60 && s->player.food_timer <= 0 && s->player.food_count > 0 &&
        hp <= s->player.base_hitpoints - osrs_food_heal_amount(FOOD_SHARK)) {
        actions[ZUL_HEAD_FOOD] = 1;  /* shark */
    }
    /* karambwan combo eat at <40 HP (emergency) */
    else if (hp < 40 && s->player.food_timer <= 0 && s->player.karambwan_count > 0 &&
             hp <= s->player.base_hitpoints - osrs_food_heal_amount(FOOD_KARAMBWAN)) {
        actions[ZUL_HEAD_FOOD] = 2;  /* karambwan */
    }

    /* restore prayer if getting low (and not already full) */
    if (s->player.current_prayer < 30 && s->player.potion_timer <= 0 &&
        s->player.prayer_pot_doses > 0 && s->player.current_prayer < s->player.base_prayer) {
        actions[ZUL_HEAD_POTION] = 1;  /* prayer pot */
    }

    /* movement: set dest to current phase's safe spot.
       zul_process_movement BFS-paths there, running when > 1 tile away. */
    {
        const ZulRotationPhase* phase = zul_current_phase(s);
        int stand = phase->stand;
        if (stand < ZUL_NUM_STAND_LOCATIONS) {
            int tx = ZUL_STAND_COORDS[stand][0];
            int ty = ZUL_STAND_COORDS[stand][1];
            if (tx != s->player.x || ty != s->player.y) {
                s->player_dest_x = tx;
                s->player_dest_y = ty;
                s->player_dest_explicit = 1;
            }
        }
    }

    /* attack: mage vs green/red (weak to magic), range vs blue (weak to range) */
    if (s->zulrah_visible && !s->is_diving) {
        if (s->current_form == ZUL_FORM_BLUE) {
            actions[ZUL_HEAD_ATTACK] = ZUL_ATK_RANGE;
            /* arm spec when energy available and not already armed */
            int spec_cost = osrs_spec_cost(ZUL_RANGE_LOADOUT[s->gear_tier][GEAR_SLOT_WEAPON]);
            if (spec_cost > 0 && s->player.special_energy >= spec_cost && !s->player.spec_armed) {
                actions[ZUL_HEAD_SPEC] = 1;  /* toggle arm */
            }
        } else {
            actions[ZUL_HEAD_ATTACK] = ZUL_ATK_MAGE;
        }
    }
}

/* ======================================================================== */
/* RL interface                                                              */
/* ======================================================================== */

static float zul_get_reward(EncounterState* state) {
    return ((ZulrahState*)state)->reward;
}
static int zul_is_terminal(EncounterState* state) {
    return ((ZulrahState*)state)->episode_over;
}

/* entity access */
static int zul_get_entity_count(EncounterState* state) {
    ZulrahState* s = (ZulrahState*)state;
    int n = 2;
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++)
        if (s->snakelings[i].active) n++;
    return n;
}
static void* zul_get_entity(EncounterState* state, int index) {
    ZulrahState* s = (ZulrahState*)state;
    if (index == 0) return &s->player;
    if (index == 1) return &s->zulrah;
    int si = 0;
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++) {
        if (s->snakelings[i].active) {
            if (si + 2 == index) return &s->snakelings[i].entity;
            si++;
        }
    }
    return &s->player;
}

/* render entity population */
static void zul_fill_render_entities(EncounterState* state, RenderEntity* out, int max_entities, int* count) {
    ZulrahState* s = (ZulrahState*)state;
    int n = 0;
    if (n < max_entities) render_entity_from_player(&s->player, &out[n++]);
    if (n < max_entities) render_entity_from_player(&s->zulrah, &out[n++]);
    for (int i = 0; i < ZUL_MAX_SNAKELINGS && n < max_entities; i++) {
        if (s->snakelings[i].active) {
            render_entity_from_player(&s->snakelings[i].entity, &out[n++]);
            /* snakelings face player when in attack range */
            int adx = abs(s->snakelings[i].entity.x - s->player.x);
            int ady = abs(s->snakelings[i].entity.y - s->player.y);
            if (adx <= 1 && ady <= 1)
                out[n - 1].attack_target_entity_idx = 0;
        }
    }
    /* player faces zulrah when interaction is active (persistent until interrupted) */
    if (osrs_interaction_active(&s->interaction))
        out[0].attack_target_entity_idx = 1;
    /* zulrah faces player during attack phases */
    if (s->zulrah_attacking && s->zulrah_visible && !s->is_diving)
        out[1].attack_target_entity_idx = 0;
    *count = n;
}

/* config */
static void zul_put_int(EncounterState* state, const char* key, int value) {
    ZulrahState* s = (ZulrahState*)state;
    if (strcmp(key, "seed") == 0) s->rng_state = (uint32_t)value;
    else if (strcmp(key, "world_offset_x") == 0) s->world_offset_x = value;
    else if (strcmp(key, "world_offset_y") == 0) s->world_offset_y = value;
    else if (strcmp(key, "gear_tier") == 0) {
        if (value >= 0 && value < ZUL_NUM_GEAR_TIERS) s->gear_tier = value;
    }
    else if (strcmp(key, "player_dest_x") == 0) {
        s->player_dest_x = value;
        if (value >= 0) s->player_dest_explicit = 1;
    }
    else if (strcmp(key, "player_dest_y") == 0) {
        s->player_dest_y = value;
        if (value >= 0) s->player_dest_explicit = 1;
    }
    else if (strcmp(key, "human_command_mode") == 0) s->human_command_mode = value;
}
static void zul_put_float(EncounterState* st, const char* k, float v) { (void)st;(void)k;(void)v; }
static void zul_put_ptr(EncounterState* st, const char* k, void* v) {
    ZulrahState* s = (ZulrahState*)st;
    if (strcmp(k, "collision_map") == 0) s->collision_map = v;
}

/* logging */
static void* zul_get_log(EncounterState* state) {
    ZulrahState* s = (ZulrahState*)state;
    if (s->episode_over) {
        s->log.episode_return += s->episode_return;
        s->log.episode_length += (float)s->tick;
        s->log.wins += (s->winner == 0) ? 1.0f : 0.0f;
        s->log.damage_dealt += s->total_damage_dealt;
        s->log.damage_received += s->total_damage_received;
        s->log.n += 1.0f;
    }
    return &s->log;
}
static int zul_get_tick(EncounterState* state) { return ((ZulrahState*)state)->tick; }

/* render overlay: expose clouds and Zulrah state to the renderer */
static void zul_render_post_tick(EncounterState* state, EncounterOverlay* ov) {
    ZulrahState* s = (ZulrahState*)state;

    /* hazards */
    ov->hazard_count = 0;
    for (int i = 0; i < ZUL_MAX_CLOUDS && ov->hazard_count < ENCOUNTER_MAX_OVERLAY_TILES; i++) {
        if (!s->clouds[i].active) continue;
        ov->hazards[ov->hazard_count].x = s->clouds[i].x;
        ov->hazards[ov->hazard_count].y = s->clouds[i].y;
        ov->hazards[ov->hazard_count].active = 1;
        ov->hazard_count++;
    }

    /* boss state: zulrah.x/y is the SW anchor tile of the NxN footprint.
       the 3D model renders centered on the footprint (x + size/2).
       hitbox spans [x, x+size) in both axes. */
    ov->boss_x = s->zulrah.x;
    ov->boss_y = s->zulrah.y;
    ov->boss_visible = s->zulrah_visible;
    ov->boss_form = (int)s->current_form;
    ov->boss_size = ZUL_NPC_SIZE;

    /* adds */
    ov->add_count = 0;
    for (int i = 0; i < ZUL_MAX_SNAKELINGS && ov->add_count < ENCOUNTER_MAX_OVERLAY_ADDS; i++) {
        if (!s->snakelings[i].active) continue;
        int si = ov->add_count++;
        ov->adds[si].x = s->snakelings[i].entity.x;
        ov->adds[si].y = s->snakelings[i].entity.y;
        ov->adds[si].active = 1;
        ov->adds[si].variant = s->snakelings[i].is_magic;
    }

    /* melee targeting indicator */
    ov->melee_target_active = s->melee_pending;
    ov->melee_target_x = s->melee_target_x;
    ov->melee_target_y = s->melee_target_y;

    /* projectile events this tick (attacks + cloud spits).
       zulrah is size 5: start_h = 5*0.75*128 = 480, end_h = 64 (player size 1) */
    ov->projectile_count = 0;
    for (int i = 0; i < s->attack_event_count && ov->projectile_count < ENCOUNTER_MAX_OVERLAY_PROJECTILES; i++) {
        if (s->attack_events[i].style == 4) {
            /* snakeling spawn orb: flies to spawn point, no tracking */
            encounter_emit_projectile(ov,
                s->attack_events[i].src_x, s->attack_events[i].src_y,
                s->attack_events[i].dst_x, s->attack_events[i].dst_y,
                4, 0,
                40, 100, 0, 12, 0.0f, 0, ZUL_NPC_SIZE, 1, 0, 0);
        } else {
            /* ranged/magic attack: tracks player, zulrah height → player height */
            uint32_t zul_proj_model = (s->attack_events[i].style == 0)
                ? GFX_RANGED_PROJ_MODEL : GFX_MAGIC_PROJ_MODEL;
            encounter_emit_projectile(ov,
                s->attack_events[i].src_x, s->attack_events[i].src_y,
                s->attack_events[i].dst_x, s->attack_events[i].dst_y,
                s->attack_events[i].style, s->attack_events[i].damage,
                35, 480, 64, 16, 0.0f, 1, ZUL_NPC_SIZE, 1, zul_proj_model, 0);
        }
    }
    for (int i = 0; i < s->cloud_event_count && ov->projectile_count < ENCOUNTER_MAX_OVERLAY_PROJECTILES; i++) {
        encounter_emit_projectile(ov,
            s->cloud_events[i].src_x, s->cloud_events[i].src_y,
            s->cloud_events[i].dst_x, s->cloud_events[i].dst_y,
            3, 0,  /* style=cloud, damage=0 */
            /* duration from flight_ticks * 30, high arc start, ground end,
               curve=10, arc_height=3.0 (high sinusoidal), no tracking, src_size=5 */
            s->cloud_events[i].flight_ticks * 30, 200, 0, 10, 3.0f, 0, ZUL_NPC_SIZE, 1,
            GFX_CLOUD_PROJ_MODEL, 0);
    }
}
static int zul_get_winner(EncounterState* state) { return ((ZulrahState*)state)->winner; }

/* ======================================================================== */
/* human input translator                                                    */
/* ======================================================================== */

static void zul_translate_human_input(HumanInput* hi, int* actions, EncounterState* state) {
    for (int h = 0; h < ZUL_NUM_ACTION_HEADS; h++) actions[h] = 0;

    encounter_translate_movement(hi, actions, ZUL_HEAD_MOVE,
                                  (void*(*)(void*,int))zul_get_entity, state);
    encounter_translate_prayer(hi, actions, ZUL_HEAD_PRAYER);
    encounter_translate_offensive_prayer(hi, actions, ZUL_HEAD_OFFENSIVE);

    /* attack style: mage or range */
    if (hi->pending_attack) {
        if (hi->pending_spell == ATTACK_ICE || hi->pending_spell == ATTACK_BLOOD)
            actions[ZUL_HEAD_ATTACK] = 1;  /* mage */
        else
            actions[ZUL_HEAD_ATTACK] = 2;  /* range */
    }

    /* food: shark on food head */
    if (hi->pending_food) actions[ZUL_HEAD_FOOD] = 1;

    /* potions: brew→food head, restore→potion 1, antivenom→potion 2 */
    if (hi->pending_potion == POTION_BREW) actions[ZUL_HEAD_FOOD] = 1;
    else if (hi->pending_potion == POTION_RESTORE) actions[ZUL_HEAD_POTION] = 1;
    else if (hi->pending_potion == POTION_ANTIVENOM) actions[ZUL_HEAD_POTION] = 2;

    /* spec */
    if (hi->pending_spec) actions[ZUL_HEAD_SPEC] = 1;

    (void)state;
}

static int zul_attack_action_for_weapon(uint8_t weapon) {
    AttackStyle style = (AttackStyle)get_item_attack_style(weapon);
    return (style == ATTACK_STYLE_MAGIC) ? ZUL_ATK_MAGE : ZUL_ATK_RANGE;
}

static void zul_translate_human_commands(HumanInput* hi, int* actions, ZulrahState* s) {
    for (int h = 0; h < ZUL_NUM_ACTION_HEADS; h++) actions[h] = 0;

    uint8_t queued_weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    for (int i = 0; i < hi->commands.count; i++) {
        const HumanCommand* cmd = &hi->commands.items[i];
        if (cmd->kind == HUMAN_COMMAND_EQUIP_INVENTORY_ITEM &&
            cmd->gear_slot == GEAR_SLOT_WEAPON &&
            cmd->item_db_idx >= 0 && cmd->item_db_idx < NUM_ITEMS) {
            queued_weapon = (uint8_t)cmd->item_db_idx;
        }

        switch (cmd->kind) {
            case HUMAN_COMMAND_WALK:
                s->player_dest_x = cmd->world_x;
                s->player_dest_y = cmd->world_y;
                s->player_dest_explicit = 1;
                actions[ZUL_HEAD_ATTACK] = ZUL_ATK_NONE;
                osrs_interaction_clear(&s->interaction);
                break;
            case HUMAN_COMMAND_ATTACK_NPC:
                actions[ZUL_HEAD_ATTACK] = zul_attack_action_for_weapon(queued_weapon);
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                s->player_dest_explicit = 0;
                break;
            case HUMAN_COMMAND_SPELL_TARGET:
                actions[ZUL_HEAD_ATTACK] = ZUL_ATK_MAGE;
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                s->player_dest_explicit = 0;
                break;
            case HUMAN_COMMAND_OVERHEAD_PRAYER:
                actions[ZUL_HEAD_PRAYER] = cmd->overhead_prayer;
                break;
            case HUMAN_COMMAND_OFFENSIVE_PRAYER:
                actions[ZUL_HEAD_OFFENSIVE] = cmd->offensive_prayer;
                break;
            case HUMAN_COMMAND_EAT:
                actions[ZUL_HEAD_FOOD] = cmd->food == 1 ? 2 : 1;
                break;
            case HUMAN_COMMAND_DRINK:
                if (cmd->potion == POTION_BREW) actions[ZUL_HEAD_FOOD] = 1;
                else if (cmd->potion == POTION_RESTORE ||
                         cmd->potion == POTION_PRAYER_POT) actions[ZUL_HEAD_POTION] = 1;
                else if (cmd->potion == POTION_ANTIVENOM) actions[ZUL_HEAD_POTION] = 2;
                break;
            case HUMAN_COMMAND_SPEC_TOGGLE:
                actions[ZUL_HEAD_SPEC] = 1;
                break;
            case HUMAN_COMMAND_EQUIP_INVENTORY_ITEM:
            case HUMAN_COMMAND_FIGHT_STYLE:
            case HUMAN_COMMAND_NONE:
                break;
        }
    }
}

static void zul_step_human_commands(EncounterState* state, HumanInput* hi) {
    ZulrahState* s = (ZulrahState*)state;
    int actions[ZUL_NUM_ACTION_HEADS];
    s->human_command_mode = 1;
    s->human_commands = hi->commands.items;
    s->human_command_count = hi->commands.count;
    zul_sync_human_gear_style(s);
    zul_refresh_human_loadout_stats(s);
    zul_translate_human_commands(hi, actions, s);
    zul_step(state, actions);
    s->human_commands = NULL;
    s->human_command_count = 0;
    human_input_clear_pending(hi);
}

/* ======================================================================== */
/* encounter definition                                                      */
/* ======================================================================== */

static const EncounterDef ENCOUNTER_ZULRAH = {
    .name = "zulrah",
    .obs_size = ZUL_NUM_OBS,
    .num_action_heads = ZUL_NUM_ACTION_HEADS,
    .action_head_dims = ZUL_ACTION_HEAD_DIMS,
    .mask_size = ZUL_ACTION_MASK_SIZE,
    .create = zul_create,
    .destroy = zul_destroy,
    .reset = zul_reset,
    .step = zul_step,
    .step_human_commands = zul_step_human_commands,
    .write_obs = zul_write_obs,
    .write_mask = zul_write_mask,
    .get_reward = zul_get_reward,
    .is_terminal = zul_is_terminal,
    .get_entity_count = zul_get_entity_count,
    .get_entity = zul_get_entity,
    .fill_render_entities = zul_fill_render_entities,
    .put_int = zul_put_int,
    .put_float = zul_put_float,
    .put_ptr = zul_put_ptr,
    .arena_base_x = 0,
    .arena_base_y = 0,
    .arena_width = ZUL_ARENA_SIZE,
    .arena_height = ZUL_ARENA_SIZE,
    .render_post_tick = zul_render_post_tick,
    .get_log = zul_get_log,
    .get_tick = zul_get_tick,
    .get_winner = zul_get_winner,

    .translate_human_input = zul_translate_human_input,
    .head_move = ZUL_HEAD_MOVE,
    .head_prayer = ZUL_HEAD_PRAYER,
    .head_target = -1,
};

__attribute__((constructor))
static void zul_register(void) {
    encounter_register(&ENCOUNTER_ZULRAH);
}

#endif /* ENCOUNTER_ZULRAH_H */
