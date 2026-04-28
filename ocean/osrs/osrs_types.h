/**
 * @fileoverview osrs_types.h — shared core types for the current ocean OSRS envs.
 *
 * Contains the enums, structs, and constants used by the current simulation
 * layer. Not every field is meaningful for every encounter; treat this as a
 * shared ABI, not universal game truth.
 */

/* ============================================================================
 * CRITICAL: OSRS TICK-BASED TIMING MODEL
 * ============================================================================
 *
 * READ THIS BEFORE MODIFYING ANY COMBAT OR MOVEMENT CODE.
 *
 * The current simulation models a 600ms tick cycle. actions are queued and
 * execute on the NEXT tick, not immediately. encounter-specific ordering can
 * still layer on top of that shared queue.
 *
 * ---------------------------------------------------------------------------
 * TICK TIMING OVERVIEW
 * ---------------------------------------------------------------------------
 *
 *   TICK N (current state):
 *     - Player SEES: positions, HP, gear, prayers, everything visible
 *     - Player QUEUES: their reaction to what they see (actions for next tick)
 *     - Actions execute: NOTHING YET - actions are just queued
 *
 *   TICK N+1 (next tick):
 *     - Queued actions from tick N EXECUTE (movement first, then attacks)
 *     - New state becomes visible
 *     - Player queues new reaction
 *
 * ---------------------------------------------------------------------------
 * MOVEMENT + ATTACK IN SAME TICK
 * ---------------------------------------------------------------------------
 *
 * When you queue an attack, the game automatically handles movement:
 *
 *   Example: dist=3, melee weapon (range=1), queue "attack"
 *     - Tick N+1: move 2 tiles (running), now dist=1, attack fires
 *     - Both movement and attack happen in the SAME tick
 *
 *   Example: dist=0 (under target), queue "attack"
 *     - Tick N+1: auto-step to adjacent tile (dist=1), attack fires
 *     - The step-out is IMPLICIT - part of the attack action
 *
 * ---------------------------------------------------------------------------
 * CONFLICTING ACTIONS (IMPORTANT!)
 * ---------------------------------------------------------------------------
 *
 * When EXPLICIT movement conflicts with IMPLICIT attack movement:
 *
 *   Example: dist=0, queue BOTH "attack" AND "move under"
 *     - Attack needs: step out to dist=1
 *     - Movement wants: stay at dist=0
 *     - RESULT: Explicit movement wins, attack is CANCELLED
 *
 * This is because the end state cannot be BOTH dist=1 (for attack) AND
 * dist=0 (from explicit movement). Explicit actions override implicit ones.
 *
 * ---------------------------------------------------------------------------
 * STEP UNDER STRATEGY (current NH/PvP model)
 * ---------------------------------------------------------------------------
 *
 * Common tactic when opponent is frozen:
 *
 *   Tick 9: You're under frozen opponent (dist=0), queue "attack" ONLY
 *   Tick 10: Step out to dist=1, attack fires, queue "move under" ONLY
 *   Tick 11: Move back under (dist=0), opponent couldn't hit you
 *
 * The frozen opponent can only hit you if they ALSO queue attack on tick 9.
 * Both attacks would fire on tick 10 when you're both effectively at dist=1.
 *
 * ---------------------------------------------------------------------------
 * RECORDING FORMAT
 * ---------------------------------------------------------------------------
 *
 * Fight recordings show for each tick:
 *   - STATE: What the player sees RIGHT NOW (positions, HP, gear, etc.)
 *   - ACTIONS: What the player QUEUED as reaction (executes NEXT tick)
 *
 * Example (valid sequence):
 *   Tick 9 state: dist=3, actions=["RNG"]
 *   Tick 10 state: dist=1, attack fires (moved 2 tiles + attacked)
 *
 * Counter-example (conflicting sequence - attack cancelled):
 *   Tick 9 state: dist=0, actions=["RNG", "under"]
 *   Tick 10 state: dist=0, NO attack (explicit "under" cancelled the attack)
 *   The attack needed dist=1, but "under" forced dist=0. Conflict resolved
 *   by cancelling the attack - explicit movement wins over implicit step-out.
 *
 * ========================================================================= */

#ifndef OSRS_TYPES_H
#define OSRS_TYPES_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include "osrs_interaction.h"

#define NUM_AGENTS 2
#define MAX_PENDING_HITS 8
#define HISTORY_SIZE 5

#define TICK_DURATION_MS 600
#define MAX_EPISODE_TICKS 300

#define WILD_MIN_X 2940
#define WILD_MAX_X 3392
#define WILD_MIN_Y 3525
#define WILD_MAX_Y 3968
#define FIGHT_AREA_BASE_X 3041
#define FIGHT_AREA_BASE_Y 3530
#define FIGHT_AREA_WIDTH 61
#define FIGHT_AREA_HEIGHT 28
#define FIGHT_NEARBY_RADIUS 5

#define ONLY_SWITCH_PRAYER_WHEN_ABOUT_TO_ATTACK 1
#define ONLY_SWITCH_GEAR_WHEN_ATTACK_SOON 1
#define ALLOW_SMITE 1
#define ALLOW_REDEMPTION 1
#define ALLOW_MOVING_IF_CAN_ATTACK 0

#define ICE_RUSH_LEVEL 58
#define ICE_BURST_LEVEL 70
#define ICE_BLITZ_LEVEL 82
#define ICE_BARRAGE_LEVEL 94

#define BLOOD_RUSH_LEVEL 56
#define BLOOD_BURST_LEVEL 68
#define BLOOD_BLITZ_LEVEL 80
#define BLOOD_BARRAGE_LEVEL 92

#define ICE_RUSH_MAX_HIT 18
#define ICE_BURST_MAX_HIT 22
#define ICE_BLITZ_MAX_HIT 26
#define ICE_BARRAGE_MAX_HIT 30

#define BLOOD_RUSH_MAX_HIT 15
#define BLOOD_BURST_MAX_HIT 21
#define BLOOD_BLITZ_MAX_HIT 25
#define BLOOD_BARRAGE_MAX_HIT 29

#define ATTACK_TIMER_INACTIVE -1000000

// Number of equipment slots (HEAD, CAPE, NECK, AMMO, WEAPON, SHIELD, BODY, LEGS, HANDS, FEET, RING)
#define NUM_GEAR_SLOTS 11
// 8 action heads: one decision per head per tick. no click encoding.
// current ocean envs use a loadout preset plus separate combat/prayer/etc. heads.

#define NUM_ACTION_HEADS 8

// Action head indices
#define HEAD_LOADOUT    0
#define HEAD_COMBAT     1   // current combined attack + movement head for loadout-backed envs
#define HEAD_OVERHEAD   2
#define HEAD_FOOD       3
#define HEAD_POTION     4
#define HEAD_KARAMBWAN  5
#define HEAD_VENG       6
#define HEAD_OFFENSIVE  7   // toggle piety/rigour/augury — agent-controlled, no longer auto-assigned

// Per-head action dimensions
#define LOADOUT_DIM     9   // KEEP, MELEE, RANGE, MAGE, TANK, SPEC_MELEE, SPEC_RANGE, SPEC_MAGIC, GMAUL
#define COMBAT_DIM     13   // NONE, ATK, ICE, BLOOD, ADJACENT, UNDER, DIAGONAL, FARCAST_2..7
#define OVERHEAD_DIM    6   // ENCOUNTER_OVERHEAD_DIM_PVP: no_change, toggle_{melee,ranged,magic,smite,redemption}
#define FOOD_DIM        2   // NONE, EAT
#define POTION_DIM      5   // PvP head only: NONE, BREW, RESTORE, COMBAT, RANGED
#define KARAMBWAN_DIM   2   // NONE, EAT
#define VENG_DIM        2   // NONE, CAST
#define OFFENSIVE_DIM   4   // ENCOUNTER_OFFENSIVE_DIM: no_change, toggle_{piety,rigour,augury}

// Total action mask size: sum of all head dims
#define ACTION_MASK_SIZE (LOADOUT_DIM + COMBAT_DIM + OVERHEAD_DIM + \
    FOOD_DIM + POTION_DIM + KARAMBWAN_DIM + VENG_DIM + OFFENSIVE_DIM)

// Per-head action dims array
static const int ACTION_HEAD_DIMS[NUM_ACTION_HEADS] = {
    LOADOUT_DIM,
    COMBAT_DIM,
    OVERHEAD_DIM,
    FOOD_DIM,
    POTION_DIM,
    KARAMBWAN_DIM,
    VENG_DIM,
    OFFENSIVE_DIM,
};

// Number of item stats per item (for observations)
#define NUM_ITEM_STATS 18

// Maximum items per slot for observation padding
#define MAX_ITEMS_PER_SLOT 10

// Dynamic gear slots that change during combat
// 8 slots: weapon, shield, body, legs, head, cape, neck, ring
#define NUM_DYNAMIC_GEAR_SLOTS 8

//* Observation size: 182 base + 1 voidwaker flag + 7 reward signals = 190 */
#define SLOT_NUM_OBSERVATIONS 190
// PLAYER BASE STATS (NH maxed accounts - 99 all combat)

#define MAXED_BASE_ATTACK 99
#define MAXED_BASE_STRENGTH 99
#define MAXED_BASE_DEFENCE 99
#define LMS_BASE_DEFENCE 75
#define MAXED_BASE_RANGED 99
#define MAXED_BASE_MAGIC 99
#define MAXED_BASE_PRAYER 77
#define MAXED_BASE_HITPOINTS 99

// LMS supply counts (1 brew, 2 restores, 1 combat pot, 1 ranged pot, 2 karams, 11 sharks)
#define MAXED_FOOD_COUNT 11
#define MAXED_KARAMBWAN_COUNT 2
#define MAXED_BREW_DOSES 4       // 1 saradomin brew (4 doses)
#define MAXED_RESTORE_DOSES 8    // 2 super restores (4 doses each)
#define MAXED_COMBAT_POTION_DOSES 4
#define MAXED_RANGED_POTION_DOSES 4

#define MAXED_MELEE_ATTACK_SPEED_OBS 4
#define MAXED_RANGED_ATTACK_SPEED_OBS 5
#define RUN_ENERGY_RECOVER_TICKS 3

typedef enum {
    ATTACK_STYLE_NONE = 0,
    ATTACK_STYLE_MELEE,
    ATTACK_STYLE_RANGED,
    ATTACK_STYLE_MAGIC
} AttackStyle;

typedef enum {
    MELEE_STYLE_STAB = 0,
    MELEE_STYLE_SLASH,
    MELEE_STYLE_CRUSH,
} MeleeStyle;

typedef enum {
    PRAYER_NONE = 0,
    PRAYER_PROTECT_MAGIC,
    PRAYER_PROTECT_RANGED,
    PRAYER_PROTECT_MELEE,
    PRAYER_SMITE,
    PRAYER_REDEMPTION
} OverheadPrayer;

typedef enum {
    GEAR_MAGE = 0,
    GEAR_RANGED,
    GEAR_MELEE,
    GEAR_SPEC,
    GEAR_TANK
} GearSet;

typedef enum {
    OFFENSIVE_PRAYER_NONE = 0,
    OFFENSIVE_PRAYER_MELEE_LOW,
    OFFENSIVE_PRAYER_RANGED_LOW,
    OFFENSIVE_PRAYER_MAGIC_LOW,
    OFFENSIVE_PRAYER_PIETY,
    OFFENSIVE_PRAYER_RIGOUR,
    OFFENSIVE_PRAYER_AUGURY
} OffensivePrayer;

/* combat stance — the axis the player picks per-weapon in the OSRS combat tab.
   orthogonal to AttackStyle (damage category). the stance drives:
     - invisible level bonuses (att / str / def) per osrs wiki "Combat Options"
     - attack speed modifier (rapid = base - 1)
     - attack range modifier (longrange = base + 2)
   the 4 melee stances keep the existing enum order where 0 means accurate. */
typedef enum {
    FIGHT_STYLE_ACCURATE = 0,             /* melee: +3 att. ranged: +3 att. powered staff: +3 magic. */
    FIGHT_STYLE_AGGRESSIVE,                /* melee: +3 str. */
    FIGHT_STYLE_CONTROLLED,                /* melee: +1 att/str/def. */
    FIGHT_STYLE_DEFENSIVE,                 /* melee: +3 def. */
    FIGHT_STYLE_RAPID,                     /* ranged: speed - 1, no bonus. */
    FIGHT_STYLE_LONGRANGE,                 /* ranged / powered staff: +3 def, +2 range. powered staff also +1 magic. */
    FIGHT_STYLE_AUTOCAST,                  /* magic (non-powered staff): no invisible bonus. */
    FIGHT_STYLE_DEFENSIVE_AUTOCAST,        /* magic: no invisible bonus (defence XP split only). */
} FightStyle;

typedef enum {
    MELEE_BONUS_STAB = 0,
    MELEE_BONUS_SLASH,
    MELEE_BONUS_CRUSH
} MeleeBonusType;

typedef enum {
    MELEE_SPEC_NONE = 0,
    MELEE_SPEC_AGS,
    MELEE_SPEC_DRAGON_CLAWS,
    MELEE_SPEC_GRANITE_MAUL,
    MELEE_SPEC_DRAGON_DAGGER,
    MELEE_SPEC_VOIDWAKER,
    MELEE_SPEC_DWH,
    MELEE_SPEC_BGS,
    MELEE_SPEC_ZGS,
    MELEE_SPEC_SGS,
    MELEE_SPEC_ANCIENT_GS,
    MELEE_SPEC_VESTAS,
    MELEE_SPEC_ABYSSAL_DAGGER,
    MELEE_SPEC_DRAGON_LONGSWORD,
    MELEE_SPEC_DRAGON_MACE,
    MELEE_SPEC_ABYSSAL_BLUDGEON
} MeleeSpecWeapon;

typedef enum {
    RANGED_SPEC_NONE = 0,
    RANGED_SPEC_DARK_BOW,
    RANGED_SPEC_BALLISTA,
    RANGED_SPEC_ACB,
    RANGED_SPEC_ZCB,
    RANGED_SPEC_DRAGON_KNIFE,
    RANGED_SPEC_MSB,
    RANGED_SPEC_MORRIGANS
} RangedSpecWeapon;

typedef enum {
    MAGIC_SPEC_NONE = 0,
    MAGIC_SPEC_VOLATILE_STAFF
} MagicSpecWeapon;

/** Equipment slot indices. */
typedef enum {
    GEAR_SLOT_HEAD = 0,
    GEAR_SLOT_CAPE,
    GEAR_SLOT_NECK,
    GEAR_SLOT_AMMO,
    GEAR_SLOT_WEAPON,
    GEAR_SLOT_SHIELD,
    GEAR_SLOT_BODY,
    GEAR_SLOT_LEGS,
    GEAR_SLOT_HANDS,
    GEAR_SLOT_FEET,
    GEAR_SLOT_RING,
} GearSlotIndex;

/** Dynamic gear slots that change during combat. */
static const int DYNAMIC_GEAR_SLOTS[NUM_DYNAMIC_GEAR_SLOTS] = {
    GEAR_SLOT_WEAPON, GEAR_SLOT_SHIELD, GEAR_SLOT_BODY, GEAR_SLOT_LEGS,
    GEAR_SLOT_HEAD, GEAR_SLOT_CAPE, GEAR_SLOT_NECK, GEAR_SLOT_RING
};

/** Loadout action head options. */
typedef enum {
    LOADOUT_KEEP = 0,
    LOADOUT_MELEE,
    LOADOUT_RANGE,
    LOADOUT_MAGE,
    LOADOUT_TANK,
    LOADOUT_SPEC_MELEE,
    LOADOUT_SPEC_RANGE,
    LOADOUT_SPEC_MAGIC,
    LOADOUT_GMAUL,
} LoadoutAction;

/**
 * Combat action head values (merged attack + movement, dim=13).
 * Attacks and movement are mutually exclusive per tick.
 * OSRS melee requires cardinal adjacency; auto-walk handles positioning.
 */
#define ATTACK_NONE      0
#define ATTACK_ATK       1
#define ATTACK_ICE       2
#define ATTACK_BLOOD     3
#define MOVE_ADJACENT    4
#define MOVE_UNDER       5
#define MOVE_DIAGONAL    6
#define MOVE_FARCAST_2   7
#define MOVE_FARCAST_3   8
#define MOVE_FARCAST_4   9
#define MOVE_FARCAST_5  10
#define MOVE_FARCAST_6  11
#define MOVE_FARCAST_7  12
#define MOVE_NONE ATTACK_NONE

static inline int is_attack_action(int v) { return v >= ATTACK_ATK && v <= ATTACK_BLOOD; }
static inline int is_move_action(int v) { return v >= MOVE_ADJACENT && v <= MOVE_FARCAST_7; }

/** Overhead prayer action head options. */
typedef enum {
    OVERHEAD_NONE = 0,
    OVERHEAD_MAGE,
    OVERHEAD_RANGED,
    OVERHEAD_MELEE,
    OVERHEAD_SMITE,
    OVERHEAD_REDEMPTION,
} OverheadAction;

/** Food action head options. */
typedef enum {
    FOOD_NONE = 0,
    FOOD_EAT,
} FoodAction;

/** Potion intent values.
    The PvP potion action head still only exposes POTION_DIM = 5.
    Extra values below are human-mode/debug intents for encounters that use
    different consumable layouts. */
typedef enum {
    POTION_NONE = 0,
    POTION_BREW,
    POTION_RESTORE,
    POTION_COMBAT,
    POTION_RANGED,
    POTION_ANTIVENOM,   /* zulrah only — outside PvP action space (POTION_DIM=5) */
    POTION_BASTION,     /* inferno human mode only */
    POTION_STAMINA,     /* inferno human mode only */
    POTION_PRAYER_POT,  /* distinct from super restore; not auto-aliased */
} PotionAction;

/** Karambwan action head options. */
typedef enum {
    KARAM_NONE = 0,
    KARAM_EAT,
} KaramAction;

/** Vengeance action head options. */
typedef enum {
    VENG_NONE = 0,
    VENG_CAST,
} VengAction;

/* Slot-based gear bonus struct used by the current ocean envs. same data as
   EquipmentBonuses (osrs_combat.h) but with a different naming convention
   (stab_attack vs attack_stab). the adapter compute_slot_gear_bonuses()
   in osrs_pvp_gear.h bridges them. */
typedef struct {
    int stab_attack;
    int slash_attack;
    int crush_attack;
    int magic_attack;
    int ranged_attack;
    int stab_defence;
    int slash_defence;
    int crush_defence;
    int magic_defence;
    int ranged_defence;
    int melee_strength;
    int ranged_strength;
    int magic_strength;
    int attack_speed;
    int attack_range;
} GearBonuses;

typedef struct {
    int magic_attack;
    int magic_strength;
    int ranged_attack;
    int ranged_strength;
    int melee_attack;
    int melee_strength;
    int magic_defence;
    int ranged_defence;
    int melee_defence;
} VisibleGearBonuses;

typedef struct {
    int damage;
    int ticks_until_hit;
    AttackStyle attack_type;
    int is_special;
    int hit_success;
    int freeze_ticks;
    int heal_percent;
    int drain_type;
    int drain_percent;
    int flat_heal;  // fixed HP heal for attacker (e.g. ancient GS blood sacrifice)
    int is_morr_bleed;  // when this hit lands, set morr_dot_remaining to damage dealt
    OverheadPrayer defender_prayer_at_attack;
} PendingHit;
// ENTITY TYPE (player vs NPC — used by renderer and encounter system)

typedef enum {
    ENTITY_PLAYER = 0,
    ENTITY_NPC = 1,
} EntityType;

typedef enum {
    OSRS_MAGIC_ATTACK_NONE = 0,
    OSRS_MAGIC_ATTACK_ANCIENT_ICE,
    OSRS_MAGIC_ATTACK_ANCIENT_BLOOD,
    OSRS_MAGIC_ATTACK_STANDARD_SPELL,
    OSRS_MAGIC_ATTACK_POWERED_STAFF,
} OsrsMagicAttackKind;

typedef enum {
    OSRS_TARGET_NONE = 0,
    OSRS_TARGET_PLAYER,
    OSRS_TARGET_NPC,
} OsrsTargetKind;

typedef struct {
    OsrsTargetKind kind;
    int id;
} OsrsTargetRef;

typedef enum {
    OSRS_RECOIL_SOURCE_NONE = 0,
    OSRS_RECOIL_SOURCE_RING_OF_RECOIL,
    OSRS_RECOIL_SOURCE_RING_OF_SUFFERING_RI,
} OsrsRecoilSource;

typedef enum {
    OSRS_SPEC_REGEN_MODE_NORMAL = 0,
    OSRS_SPEC_REGEN_MODE_LIGHTBEARER,
} OsrsSpecRegenMode;

typedef struct {
    uint32_t effect_mask;
    uint8_t weapon_item;
    uint8_t ring_item;
    uint8_t shield_item;
    uint8_t virtus_piece_count;
    uint8_t dharok_piece_count;
    OsrsRecoilSource recoil_source;
    OsrsSpecRegenMode spec_regen_mode;
} OsrsEquipmentEffectProfile;

typedef struct {
    int special_regen_ticks;
    int recoil_charges;
    uint8_t confliction_is_primed;
    uint8_t confliction_weapon_item;
    OsrsMagicAttackKind confliction_magic_kind;
    OsrsTargetRef confliction_target;
} OsrsItemEffectState;

typedef struct {
    EntityType entity_type;  /* ENTITY_PLAYER or ENTITY_NPC */
    int npc_def_id;          /* NPC definition ID (unused for players) */
    int npc_visible;         /* render visibility flag (NPCs only, e.g. Zulrah dive) */
    int npc_size;            /* NPC hitbox size in tiles (1 for players, 5 for Zulrah, etc.) */
    int npc_anim_id;         /* current animation seq ID (-1 = use idle from NpcModelMapping) */

    // Game mode flags (set per-player during c_reset)
    int is_lms;

    // Base stats
    int base_attack;
    int base_strength;
    int base_defence;
    int base_ranged;
    int base_magic;
    int base_prayer;
    int base_hitpoints;

    // Current stats
    int current_attack;
    int current_strength;
    int current_defence;
    int current_ranged;
    int current_magic;
    int current_prayer;
    int current_hitpoints;

    // Special attack state
    int special_energy;
    int spec_regen_active;
    int spec_armed;            /* 1 = next attack uses special (shared across encounters) */
    OsrsInteraction interaction;  /* shared entity interaction state */
    OsrsItemEffectState item_effect_state;

    // Gear
    GearSet current_gear;       // tracks active combat style for visible_gear and style checks
    GearSet visible_gear;       // external: actual weapon damage type (MELEE/RANGED/MAGE only, no GEAR_SPEC)

    // Consumables
    int food_count;
    int karambwan_count;
    int brew_doses;
    int restore_doses;
    int prayer_pot_doses;
    int combat_potion_doses;
    int ranged_potion_doses;
    int bastion_doses;
    int stamina_doses;
    int antivenom_doses;

    // Timers
    int attack_timer;
    int attack_timer_uncapped;
    int has_attack_timer;
    int food_timer;
    int potion_timer;
    int karambwan_timer;

    // Consumable tracking (used for timing/metrics)
    // Set to 1 during execute_slot_switches if any consumable succeeded this tick
    uint8_t consumable_used_this_tick;
    int last_food_heal;
    int last_food_waste;
    int last_karambwan_heal;
    int last_karambwan_waste;
    int last_brew_heal;
    int last_brew_waste;
    int last_potion_type;
    int last_potion_was_waste;

    // Freeze state
    int frozen_ticks;
    int freeze_immunity_ticks;

    // Vengeance
    int veng_active;
    int veng_cooldown;

    // Prayer and style
    OverheadPrayer prayer;
    OffensivePrayer offensive_prayer;
    FightStyle fight_style;
    int prayer_drain_counter;  // Accumulates drain, triggers at drain_resistance
    /* activation-tick bookkeeping: OSRS does not drain a prayer on the tick it
       was activated (wiki: "the game does not drain prayer for prayers on the
       tick they are activated"). these flags are set to 1 on OFF→ON transitions
       in encounter_apply_{overhead,offensive}_action() and cleared by
       encounter_drain_all_prayers(). required for 1-tick prayer flicking. */
    uint8_t prayer_just_activated;
    uint8_t offensive_prayer_just_activated;

    // Position
    int x, y;
    int dest_x, dest_y;
    int is_moving;
    int is_running;
    int run_energy;
    int run_recovery_ticks;
    int last_obs_target_x;
    int last_obs_target_y;

    // Attack tracking
    int just_attacked;
    AttackStyle last_attack_style;
    int last_queued_hit_damage;  // Damage of most recent attack (XP drop equivalent)
    int attack_was_on_prayer;    // 1 if defender had correct prayer when attack processed
    int attack_click_canceled;
    int attack_click_ready;
    int last_attack_dx;
    int last_attack_dy;
    int last_attack_dist;

    // Pending hits queue
    PendingHit pending_hits[MAX_PENDING_HITS];
    int num_pending_hits;
    int damage_applied_this_tick;
    int did_attack_auto_move;  // set in attack movement phase, read in attack combat phase

    // Hit event tracking for event log
    int hit_landed_this_tick;
    int hit_was_successful;
    int hit_damage;
    AttackStyle hit_style;
    OverheadPrayer hit_defender_prayer;
    int hit_was_on_prayer;
    int hit_attacker_idx;
    int freeze_applied_this_tick;

    // Morrigan's javelin DoT (Phantom Strike): 5 HP every 3 ticks from calc tick
    int morr_dot_remaining;      // remaining bleed damage to deliver
    int morr_dot_tick_counter;   // ticks until next bleed (counts down from 3)

    // Damage tracking
    float last_target_health_percent;
    float tick_damage_scale;
    float damage_dealt_scale;
    float damage_received_scale;

    // Hit statistics
    int total_target_hit_count;
    int target_hit_melee_count;
    int target_hit_ranged_count;
    int target_hit_magic_count;
    int target_hit_off_prayer_count;
    int target_hit_correct_count;

    int total_target_pray_count;
    int target_pray_melee_count;
    int target_pray_ranged_count;
    int target_pray_magic_count;
    int target_pray_correct_count;

    int player_hit_melee_count;
    int player_hit_ranged_count;
    int player_hit_magic_count;

    int player_pray_melee_count;
    int player_pray_ranged_count;
    int player_pray_magic_count;

    // History buffers
    AttackStyle recent_target_attack_styles[HISTORY_SIZE];
    AttackStyle recent_player_attack_styles[HISTORY_SIZE];
    AttackStyle recent_target_prayer_styles[HISTORY_SIZE];
    AttackStyle recent_player_prayer_styles[HISTORY_SIZE];
    int recent_target_prayer_correct[HISTORY_SIZE];
    int recent_target_hit_correct[HISTORY_SIZE];
    int recent_target_attack_index;
    int recent_player_attack_index;
    int recent_target_prayer_index;
    int recent_player_prayer_index;
    int recent_target_prayer_correct_index;
    int recent_target_hit_correct_index;

    // Observed target stats
    int target_magic_accuracy;
    int target_magic_strength;
    int target_ranged_accuracy;
    int target_ranged_strength;
    int target_melee_accuracy;
    int target_melee_strength;
    int target_magic_gear_magic_defence;
    int target_magic_gear_ranged_defence;
    int target_magic_gear_melee_defence;
    int target_ranged_gear_magic_defence;
    int target_ranged_gear_ranged_defence;
    int target_ranged_gear_melee_defence;
    int target_melee_gear_magic_defence;
    int target_melee_gear_ranged_defence;
    int target_melee_gear_melee_defence;

    // Prayer correctness flags
    int player_prayed_correct;
    int target_prayed_correct;

    // Total damage
    float total_damage_dealt;
    float total_damage_received;

    // Equipment flags
    int is_lunar_spellbook;
    int observed_target_lunar_spellbook;
    int has_blood_fury;

    // Spec weapons
    MeleeSpecWeapon melee_spec_weapon;
    RangedSpecWeapon ranged_spec_weapon;
    MagicSpecWeapon magic_spec_weapon;

    // Bolt procs
    float bolt_proc_damage;
    int bolt_ignores_defense;

    // Slot-based mode equipment (per-slot item indices, 255 = empty)
    // equipped[GEAR_SLOT_*] = item index from ITEMS_BY_SLOT table, or 255 if empty
    uint8_t equipped[NUM_GEAR_SLOTS];

    // Available items per slot (for action masking and observations)
    // inventory[slot][item_idx] = item database index, 255 = no item
    uint8_t inventory[NUM_GEAR_SLOTS][MAX_ITEMS_PER_SLOT];

    // Number of items available per slot
    uint8_t num_items_in_slot[NUM_GEAR_SLOTS];

    // Cached bonuses for slot-based mode
    GearBonuses slot_cached_bonuses;
    OsrsEquipmentEffectProfile equipment_effect_profile;
    int slot_gear_dirty;

    // Per-tick action tracking for reward shaping
    // These are set when actions actually execute (not when queued)
    AttackStyle attack_style_this_tick;  // Actual attack style used (NONE if no attack)
    int magic_type_this_tick;            // 0=none, 1=ice, 2=blood (for visual effects)
    int used_special_this_tick;          // 1 if special attack was used
    int ate_food_this_tick;              // 1 if regular food was consumed
    int ate_karambwan_this_tick;         // 1 if karambwan was consumed
    int ate_brew_this_tick;             // 1 if saradomin brew was consumed
    int cast_veng_this_tick;            // 1 if vengeance was cast (for animation)
    int clicks_this_tick;               // accumulated click count for progressive penalty

    // Previous tick HP percent for reward shaping (premature/wasted eat checks)
    float prev_hp_percent;

    // GUI stats panel fields (synced from encounter state each render tick)
    int gui_max_hit;
    int gui_attack_speed;
    int gui_attack_range;
    int gui_strength_bonus;
} Player;

typedef struct {
    float episode_return;
    float episode_length;
    float wins;
    float damage_dealt;
    float damage_received;
    float wave;
    float prayer_correct;
    float prayer_total;
    float idle_ticks;
    float brews_used;
    float blood_healed;
    /* behavioral metrics */
    float npc_kills;
    float gear_switches;
    float current_ranged;
    float current_magic;
    float unavoidable_off_prayer;  /* off-prayer hits where correct prayer was on a different style */
    float brews_remaining;         /* brew doses left at end of episode */
    float restores_remaining;      /* restore doses left at end of episode */
    float prayer_at_death;         /* prayer points at end of episode */
    /* Zuk diagnostics */
    float behind_shield_pct;   /* fraction of Zuk ticks behind shield */
    float zuk_hp_remaining;    /* Zuk HP at episode end (0 if killed) */
    float min_zuk_hp_seen;     /* lowest Zuk HP reached during the episode */
    float hp_restored;         /* HP restored to enemies (healers + mager) this episode */
    float zuk_healer_damage;   /* total damage dealt to Zuk healers this episode */
    /* Inferno action noop rates by named head. */
    float noop_move;
    float noop_prayer;
    float noop_target;
    float noop_gear;
    float noop_eat;
    float noop_potion;
    float noop_spell;
    float noop_spec;
    /* per-NPC-type stats (14 types each, for wandb only — not shown on dashboard) */
    float prayer_correct_by_type[14];
    float attacks_by_type[14];
    float dmg_from_type[14];
    float killed_by_type[14];
    float start_wave;   /* config start_wave (for score formula branching) */
    float n;
} Log;

typedef struct {
    // Per-tick shaping coefficients
    float damage_dealt_coef;         // per-HP dealt
    float damage_received_coef;      // per-HP received (negative)
    float correct_prayer_bonus;      // blocked attack with correct prayer
    float wrong_prayer_penalty;      // got hit off-prayer
    float prayer_switch_no_attack_penalty;  // switched protection prayer but opponent didn't attack
    float off_prayer_hit_bonus;      // hit opponent off-prayer
    float melee_frozen_penalty;      // melee while frozen and out of range
    float wasted_eat_penalty;        // per wasted HP of healing overflow
    float premature_eat_penalty;     // eating above premature threshold
    float magic_no_staff_penalty;    // casting magic without staff
    float gear_mismatch_penalty;     // attacking with negative bonus for the attack style
    float spec_off_prayer_bonus;     // spec when target not praying melee
    float spec_low_defence_bonus;    // spec when target in mage gear
    float spec_low_hp_bonus;         // spec when target below 50% HP
    float smart_triple_eat_bonus;    // triple eat at low HP
    float wasted_triple_eat_penalty; // per wasted karam HP at high HP
    float damage_burst_bonus;        // per HP above burst threshold
    int   damage_burst_threshold;    // minimum damage for burst bonus
    float premature_eat_threshold;   // HP percent above which eating is premature (70/99)
    // Terminal shaping
    float ko_bonus;                  // bonus for KO (opponent had food left)
    float wasted_resources_penalty;  // dying with food/brews left
    // Scale (annealed from Python during training)
    float shaping_scale;             // 1.0 → floor over training
    int   enabled;                   // 0 = sparse only, 1 = full shaping
    // Always-on behavioral penalties (independent of `enabled`)
    int   prayer_penalty_enabled;    // wasteful prayer switch penalty
    int   click_penalty_enabled;     // progressive excess-click penalty
    int   click_penalty_threshold;   // free clicks before penalty kicks in
    float click_penalty_coef;        // penalty per excess click (negative)
} RewardShapingConfig;
// OPPONENT TYPES (used by osrs_pvp_opponents.h functions)

typedef enum {
    OPP_NONE = 0,
    OPP_TRUE_RANDOM,
    OPP_PANICKING,
    OPP_WEAK_RANDOM,
    OPP_SEMI_RANDOM,
    OPP_STICKY_PRAYER,
    OPP_RANDOM_EATER,
    OPP_PRAYER_ROOKIE,
    OPP_IMPROVED,
    OPP_MIXED_EASY,
    OPP_MIXED_MEDIUM,
    OPP_ONETICK,
    OPP_UNPREDICTABLE_IMPROVED,
    OPP_UNPREDICTABLE_ONETICK,
    OPP_MIXED_HARD,
    OPP_MIXED_HARD_BALANCED,
    OPP_PFSP,
    OPP_NOVICE_NH,
    OPP_APPRENTICE_NH,
    OPP_COMPETENT_NH,
    OPP_INTERMEDIATE_NH,
    OPP_ADVANCED_NH,
    OPP_PROFICIENT_NH,
    OPP_EXPERT_NH,
    OPP_MASTER_NH,
    OPP_SAVANT_NH,
    OPP_NIGHTMARE_NH,
    OPP_VENG_FIGHTER,
    OPP_BLOOD_HEALER,
    OPP_GMAUL_COMBO,
    OPP_RANGE_KITER,
    OPP_SELFPLAY,
} OpponentType;

#define MAX_OPPONENT_POOL 32

typedef struct {
    OpponentType pool[MAX_OPPONENT_POOL];
    int cum_weights[MAX_OPPONENT_POOL];  /* cumulative weights * 1000 */
    int pool_size;
    int active_pool_idx;                 /* which pool entry is active this episode */
    float wins[MAX_OPPONENT_POOL];       /* per-opponent wins (read+reset by Python) */
    float episodes[MAX_OPPONENT_POOL];   /* per-opponent episode count */
} PFSPState;

typedef struct {
    OpponentType type;
    OpponentType active_sub_policy;
    int chosen_prayer;
    int chosen_style;
    int current_prayer;
    int current_prayer_set;
    int food_cooldown;
    int potion_cooldown;
    int karambwan_cooldown;

    /* Phase 2: onetick + realistic policy state */
    int fake_switch_pending;         /* 0/1 */
    int fake_switch_style;           /* OPP_STYLE_* or -1 */
    int opponent_prayer_at_fake;     /* OPP_STYLE_* or -1 (style they were praying) */
    int fake_switch_failed;          /* 0/1 (unpredictable_onetick only) */
    int pending_prayer_value;        /* OVERHEAD_* value, 0 = none */
    int pending_prayer_delay;        /* ticks remaining before applying */
    int last_target_gear_style;      /* OPP_STYLE_* or -1, tracks previous tick */

    /* Per-episode eating thresholds (randomized with noise) */
    float eat_triple_threshold;       /* base 0.30, range [0.25, 0.35] */
    float eat_double_threshold;       /* base 0.50, range [0.45, 0.55] */
    float eat_brew_threshold;         /* base 0.70, range [0.65, 0.75] */

    /* Per-episode randomized decision parameters */
    float prayer_accuracy;            /* chance of correct defensive prayer [0,1] */
    float off_prayer_rate;            /* chance of attacking off-prayer [0,1] */
    float offensive_prayer_rate;      /* chance of using offensive prayer [0,1] */
    float action_delay_chance;        /* per-tick chance to skip prayer+attack [0,0.3] */
    float mistake_rate;               /* per-tick chance to pick random prayer [0,0.15] */

    /* Boss opponent reading ability (master_nh, savant_nh) */
    float read_chance;               /* 0.0-1.0, chance to "read" agent action each tick */
    int has_read_this_tick;          /* 1 if read succeeded this tick */
    AttackStyle read_agent_style;    /* agent's pending attack style (if read) */
    OverheadPrayer read_agent_prayer;/* agent's pending overhead prayer (if read) */
    int read_agent_moving;           /* boss read: 1 if agent is moving (not attacking) */

    /* Anti-kite flee tracking */
    int prev_dist_to_target;         /* previous tick distance for flee tracking */
    int target_fleeing_ticks;        /* consecutive ticks distance has been increasing */

    /* gmaul_combo state */
    int combo_state;                 /* 0=idle, 1=spec_fired (follow with gmaul next tick) */
    float ko_threshold;              /* target HP fraction to trigger KO sequence */

    /* Offensive prayer miss: chance to attack without switching loadout (skipping auto-prayer) */
    float offensive_prayer_miss;

    /* Per-episode style bias: weighted preference for melee/ranged/mage */
    float style_bias[3];
} OpponentState;

typedef struct {
    int is_pvp_arena;
    int use_c_opponent;  /* 1 = generate opponent actions in C */
    int use_c_opponent_p0;
    int use_external_opponent_actions;  /* 1 = use external actions for player 1 */
    int external_opponent_actions[NUM_ACTION_HEADS];
    OpponentState opponent;
    OpponentState opponent_p0;
    PFSPState pfsp;
    float gear_tier_weights[4];  /* 4 tiers, sum to 1.0 */
} OsrsPvpRuntime;

typedef struct {
    float* agent_obs;               /* OCEAN_OBS_SIZE floats (normalized obs + mask) */
    float* agent_obs_p1;            /* player 1 obs when self-play is enabled */
    unsigned char* selfplay_mask;   /* 1 byte: 1 when this env is in self-play */
    int* agent_actions;             /* NUM_ACTION_HEADS ints (agent 0 actions) */
    float* agent_rewards;           /* 1 float (agent 0 reward) */
    unsigned char* agent_terminals; /* 1 byte (agent 0 terminal) */
} OsrsOceanBuffers;

// Combined observation size: raw obs + action masks (for ocean mode)
#define OCEAN_OBS_SIZE (SLOT_NUM_OBSERVATIONS + ACTION_MASK_SIZE)

typedef struct {
    Log log;

    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* action_masks;
    unsigned char action_masks_agents;
    int num_agents;

    Player players[NUM_AGENTS];

    int tick;
    int episode_over;
    int winner;
    int auto_reset;
    int pid_holder;
    int pid_shuffle_countdown;  // ticks until next PID swap (100-150)

    int is_lms;

    uint32_t rng_state;
    uint32_t rng_seed;
    int has_rng_seed;

    // Async action processing (OSRS-accurate timing)
    // Actions submitted on tick N take effect at START of tick N+1
    int pending_actions[NUM_AGENTS * NUM_ACTION_HEADS];
    int last_executed_actions[NUM_AGENTS * NUM_ACTION_HEADS];

    // Reward shaping configuration (coefficients + annealing scale)
    RewardShapingConfig shaping;

    // PvP-only runtime state. encounters that bypass the PvP stack can ignore this.
    OsrsPvpRuntime pvp_runtime;

    // Encounter dispatch. NULL uses the default PvP step/reset path.
    const void* encounter_def;     /* EncounterDef* — void* to avoid include dependency */
    void* encounter_state;          /* EncounterState* — owned by this env */

    // Collision map (shared across envs, read-only after init). NULL = flat arena.
    void* collision_map;  /* CollisionMap* — void* to avoid forward-decl dependency */

    // Raylib render client (NULL = headless). Initialized on first render call.
    void* client;

    // PufferLib / ocean-side agent buffers.
    OsrsOceanBuffers ocean_io;
    float _episode_return;         // Running episode return accumulator

    // Internal 2-agent buffers for game logic
    float _obs_buf[NUM_AGENTS * SLOT_NUM_OBSERVATIONS];
    int _acts_buf[NUM_AGENTS * NUM_ACTION_HEADS];
    float _rews_buf[NUM_AGENTS];
    unsigned char _terms_buf[NUM_AGENTS];
    unsigned char _masks_buf[NUM_AGENTS * ACTION_MASK_SIZE];

} OsrsEnv;

static inline int abs_int(int val) {
    return val < 0 ? -val : val;
}

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}

static inline int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

static inline float clampf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

/** Chebyshev distance - OSRS uses this for range checks. */
static inline int chebyshev_distance(int x1, int y1, int x2, int y2) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    return (dx > dy) ? dx : dy;
}

/**
 * OSRS melee range check: CARDINAL ADJACENCY ONLY (N/S/E/W).
 *
 * Standard melee weapons (range=1) can ONLY hit from cardinal-adjacent tiles.
 * Diagonal tiles (Chebyshev dist=1) are NOT valid melee positions.
 * Auto-walk for melee attacks also paths to cardinal adjacency, not diagonal.
 *
 * This applies to all current weapons. Halberds (range=2, e.g. noxious halberd)
 * will need a separate range model when added — they can hit from 2 tiles away
 * including some diagonal positions.
 */
static inline int is_in_melee_range(Player* p, Player* t) {
    int dx = abs_int(p->x - t->x);
    int dy = abs_int(p->y - t->y);
    return (dx == 1 && dy == 0) || (dx == 0 && dy == 1);
}

static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline int rand_int(OsrsEnv* env, int max) {
    if (max <= 0) return 0;
    return xorshift32(&env->rng_state) % max;
}

static inline float rand_float(OsrsEnv* env) {
    return (float)xorshift32(&env->rng_state) / (float)UINT32_MAX;
}

static inline int is_in_wilderness(int x, int y) {
    return x >= WILD_MIN_X && x <= WILD_MAX_X && y >= WILD_MIN_Y && y <= WILD_MAX_Y;
}

static inline int tile_hash(int x, int y) {
    return (x << 15) | y;
}

static inline int remaining_ticks(int ticks) {
    return ticks > 0 ? ticks : 0;
}

static inline int get_attack_timer_uncapped(Player* p) {
    return p->has_attack_timer ? p->attack_timer_uncapped : -100;
}

static inline int can_attack_now(Player* p) {
    if (!p->has_attack_timer) return 1;
    return p->attack_timer < 0;
}

static inline int can_move(Player* p) {
    return p->frozen_ticks <= 0;
}

/** Safe ratio calculation (returns 0 if denominator is 0). */
static inline float ratio_or_zero(int numerator, int denominator) {
    if (denominator == 0) {
        return 0.0f;
    }
    return (float)numerator / (float)denominator;
}

/** Scale confidence based on sample count (saturates at 10). */
static inline float confidence_scale(int count) {
    if (count >= 10) {
        return 1.0f;
    }
    return (float)count / 10.0f;
}

/** Check if lightbearer spec regeneration is active from equipped gear. */
static inline int is_lightbearer_equipped(Player* p) {
    return p->equipment_effect_profile.spec_regen_mode == OSRS_SPEC_REGEN_MODE_LIGHTBEARER;
}

#define RECOIL_MAX_CHARGES 40

#endif // OSRS_TYPES_H
