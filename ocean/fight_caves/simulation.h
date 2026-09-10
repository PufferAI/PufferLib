#ifndef FIGHT_CAVES_SIMULATION_H
#define FIGHT_CAVES_SIMULATION_H

/* Player Init */

#include <stdint.h>

/*
 * Player skill, equipment, and consumable configuration shared by the core,
 * training adapter, viewer, and asset tooling. The immutable table itself is
 * defined once in fc_loadouts.c.
 */

typedef enum {
    FC_LOADOUT_BLACK_DHIDE_RCB = 0,
    FC_LOADOUT_SOTA_TBOW = 1,
    FC_LOADOUT_LOW_DEF_RCB = 2,
    FC_LOADOUT_RCB_PURE = 3,
    FC_LOADOUT_MSBI_PURE = 4,
    FC_LOADOUT_BLOWPIPE_PURE = 5,
    FC_LOADOUT_ACB_ARMADYL = 6,
    FC_LOADOUT_BOWFA_CRYSTAL = 7,
    FC_LOADOUT_TBOW_MASORI = 8,
    FC_LOADOUT_COUNT
} FcLoadoutId;

#ifndef FC_ACTIVE_LOADOUT
#define FC_ACTIVE_LOADOUT FC_LOADOUT_SOTA_TBOW
#endif

#define FC_LOADOUT_EQUIP_MAX 12
#define FC_LOADOUT_MODEL_ITEM_MAX 12
#define FC_PLAYER_MODEL_BASE 0xFC000000u

typedef enum {
    FC_EQUIP_SLOT_HEAD = 0,
    FC_EQUIP_SLOT_CAPE = 1,
    FC_EQUIP_SLOT_NECK = 2,
    FC_EQUIP_SLOT_WEAPON = 3,
    FC_EQUIP_SLOT_BODY = 4,
    FC_EQUIP_SLOT_SHIELD = 5,
    FC_EQUIP_SLOT_AMMO = 13,
    FC_EQUIP_SLOT_LEGS = 7,
    FC_EQUIP_SLOT_HANDS = 9,
    FC_EQUIP_SLOT_FEET = 10,
    FC_EQUIP_SLOT_RING = 12,
} FcEquipmentSlot;

typedef struct {
    int slot;
    uint32_t item_id;
    uint32_t icon_item_id;
    const char* label;
} FcLoadoutEquipmentItem;

typedef enum {
    FC_CRYSTAL_PIECE_NONE = 0,
    FC_CRYSTAL_PIECE_HELM = 1 << 0,
    FC_CRYSTAL_PIECE_BODY = 1 << 1,
    FC_CRYSTAL_PIECE_LEGS = 1 << 2,
    FC_CRYSTAL_PIECE_ALL = FC_CRYSTAL_PIECE_HELM |
                           FC_CRYSTAL_PIECE_BODY |
                           FC_CRYSTAL_PIECE_LEGS
} FcCrystalPieceMask;

/* Exact per-piece modifiers. One percentage point is 100 basis points. */
#define FC_CRYSTAL_HELM_ACCURACY_BP  500
#define FC_CRYSTAL_HELM_DAMAGE_BP    250
#define FC_CRYSTAL_BODY_ACCURACY_BP 1500
#define FC_CRYSTAL_BODY_DAMAGE_BP    750
#define FC_CRYSTAL_LEGS_ACCURACY_BP 1000
#define FC_CRYSTAL_LEGS_DAMAGE_BP    500

typedef struct {
    const char* name;
    const char* weapon_name;
    uint32_t player_model_id;
    int combat_style_profile;
    int max_hp, max_prayer;
    int attack_lvl, strength_lvl, defence_lvl;
    int ranged_lvl, prayer_lvl, magic_lvl;
    int weapon_kind;
    int weapon_uses_ammo;
    int crystal_piece_mask;
    int weapon_speed;
    int weapon_range;
    int ranged_atk, ranged_str;
    int def_stab, def_slash, def_crush, def_magic, def_ranged;
    int prayer_bonus;
    int ammo;
    int equipment_count;
    FcLoadoutEquipmentItem equipment[FC_LOADOUT_EQUIP_MAX];
    int model_item_count;
    int model_item_ids[FC_LOADOUT_MODEL_ITEM_MAX];
} FcLoadout;

typedef enum {
    FC_WEAPON_GENERIC_RANGED = 0,
    FC_WEAPON_TWISTED_BOW = 1,
    FC_WEAPON_BOW_OF_FAERDHINEN = 2,
    FC_WEAPON_UNARMED = 3
} FcWeaponKind;

#define FC_NUM_LOADOUTS FC_LOADOUT_COUNT

extern const FcLoadout FC_LOADOUTS[FC_NUM_LOADOUTS];

#define FC_PLAYER_MAX_HP        (FC_LOADOUTS[FC_ACTIVE_LOADOUT].max_hp)
#define FC_PLAYER_MAX_PRAYER    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].max_prayer)
#define FC_PLAYER_DEFENCE_LVL   (FC_LOADOUTS[FC_ACTIVE_LOADOUT].defence_lvl)
#define FC_PLAYER_RANGED_LVL    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].ranged_lvl)
#define FC_PLAYER_PRAYER_LVL    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].prayer_lvl)
#define FC_PLAYER_MAGIC_LVL     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].magic_lvl)
#define FC_PLAYER_WEAPON_USES_AMMO \
    (FC_LOADOUTS[FC_ACTIVE_LOADOUT].weapon_uses_ammo)
#define FC_PLAYER_WEAPON_SPEED  (FC_LOADOUTS[FC_ACTIVE_LOADOUT].weapon_speed)
#define FC_PLAYER_WEAPON_RANGE  (FC_LOADOUTS[FC_ACTIVE_LOADOUT].weapon_range)
#define FC_EQUIP_RANGED_ATK     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].ranged_atk)
#define FC_EQUIP_RANGED_STR     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].ranged_str)
#define FC_EQUIP_DEF_CRUSH      (FC_LOADOUTS[FC_ACTIVE_LOADOUT].def_crush)
#define FC_EQUIP_DEF_MAGIC      (FC_LOADOUTS[FC_ACTIVE_LOADOUT].def_magic)
#define FC_EQUIP_DEF_RANGED     (FC_LOADOUTS[FC_ACTIVE_LOADOUT].def_ranged)


/* Types */

#include <stdint.h>

/*
 * fc_types.h — Core data structures for Fight Caves simulation.
 *
 * Design rules (from PufferLib OSRS PvP reference):
 *   - All state is flat fields on structs. No nested pointers, no heap alloc per tick.
 *   - Enables cache locality, fast memset reset, linear observation generation.
 *   - All structs are zeroed on reset via memset; zero must be a safe default for every field.
 *
 * PR 2 combat note — PvM prayer semantics:
 *   Protection prayers in Fight Caves BLOCK 100% of the matching NPC attack style.
 *   This is standard OSRS PvM behavior, NOT the PvP 60% reduction from osrs_pvp.
 *   Exceptions (e.g. TzTok-Jad hits through wrong prayer) must be explicit per NPC/attack.
 *   See fc_combat.c (PR 2) for implementation.
 */

/* ======================================================================== */
/* Enums                                                                     */
/* ======================================================================== */

typedef enum {
    ENTITY_PLAYER = 0,
    ENTITY_NPC    = 1
} FcEntityType;

/* NPC type codes — map to OSRS Fight Caves monsters */
typedef enum {
    NPC_NONE       = 0,
    NPC_TZ_KIH     = 1,  /* Lv 22, melee, drains prayer on hit */
    NPC_TZ_KEK     = 2,  /* Lv 45, melee, splits into 2 small on death */
    NPC_TZ_KEK_SM  = 3,  /* Lv 22, melee, small Tz-Kek spawn (from split) */
    NPC_TOK_XIL    = 4,  /* Lv 90, ranged */
    NPC_YT_MEJKOT  = 5,  /* Lv 180, melee + heals nearby NPCs */
    NPC_KET_ZEK    = 6,  /* Lv 360, magic (primary) + melee */
    NPC_TZTOK_JAD  = 7,  /* Lv 702, magic + ranged, prayer switching */
    NPC_YT_HURKOT  = 8,  /* Jad healer, permanently targets player once tagged */
    NPC_TYPE_COUNT  = 9
} FcNpcType;

/* Attack styles */
typedef enum {
    ATTACK_NONE   = 0,
    ATTACK_MELEE  = 1,
    ATTACK_RANGED = 2,
    ATTACK_MAGIC  = 3
} FcAttackStyle;

/* Exact incoming attack type used for equipment-defence selection. This is
 * intentionally separate from FcAttackStyle, which remains the broad style
 * used for prayer matching, observations, and animations. */
typedef enum {
    FC_ATTACK_TYPE_NONE   = 0,
    FC_ATTACK_TYPE_STAB   = 1,
    FC_ATTACK_TYPE_SLASH  = 2,
    FC_ATTACK_TYPE_CRUSH  = 3,
    FC_ATTACK_TYPE_RANGED = 4,
    FC_ATTACK_TYPE_MAGIC  = 5
} FcAttackType;

/* Protection prayers */
typedef enum {
    PRAYER_NONE          = 0,
    PRAYER_PROTECT_MELEE = 1,
    PRAYER_PROTECT_RANGE = 2,
    PRAYER_PROTECT_MAGIC = 3
} FcPrayer;

/* Terminal state codes */
typedef enum {
    TERMINAL_NONE          = 0,
    TERMINAL_PLAYER_DEATH  = 1,
    TERMINAL_CAVE_COMPLETE = 2,
    TERMINAL_TICK_CAP      = 3
} FcTerminalCode;

/* Invalid-action diagnostic classes for Puffer-facing heads 0-2. */
typedef enum {
    FC_INVALID_ACTION_MOVE   = 0,
    FC_INVALID_ACTION_ATTACK = 1,
    FC_INVALID_ACTION_PRAYER = 2,
    FC_INVALID_ACTION_CLASS_COUNT = 3
} FcInvalidActionClass;

/* NPC spawn direction for wave rotations */
typedef enum {
    SPAWN_SOUTH      = 0,
    SPAWN_SOUTH_WEST = 1,
    SPAWN_NORTH_WEST = 2,
    SPAWN_SOUTH_EAST = 3,
    SPAWN_CENTER     = 4
} FcSpawnDir;

/* ======================================================================== */
/* Constants                                                                 */
/* ======================================================================== */

/* Arena */
#define FC_ARENA_WIDTH   64
#define FC_ARENA_HEIGHT  64

/* OSRS movement-wall flags. These retain the cache's low-byte directional
 * layout so a wall blocks only the boundary it occupies, not the whole tile. */
#define FC_MOVE_WALL_NORTH_WEST (1u << 0)
#define FC_MOVE_WALL_NORTH      (1u << 1)
#define FC_MOVE_WALL_NORTH_EAST (1u << 2)
#define FC_MOVE_WALL_EAST       (1u << 3)
#define FC_MOVE_WALL_SOUTH_EAST (1u << 4)
#define FC_MOVE_WALL_SOUTH      (1u << 5)
#define FC_MOVE_WALL_SOUTH_WEST (1u << 6)
#define FC_MOVE_WALL_WEST       (1u << 7)

/* Directional projectile/line-of-sight collision flags. These are separate
 * from walkability: an obstacle may block movement without blocking attacks. */
#define FC_LOS_NORTH (1u << 0)
#define FC_LOS_EAST  (1u << 1)
#define FC_LOS_SOUTH (1u << 2)
#define FC_LOS_WEST  (1u << 3)
#define FC_LOS_FULL  (1u << 4)

/* Entity limits */
#define FC_MAX_NPCS      16   /* max simultaneous NPCs in the arena */
#define FC_VISIBLE_NPCS   8   /* max NPCs in observation (see fc_contracts.h) */
#define FC_MAX_PENDING_HITS 8 /* per entity pending hit queue */

/* Waves */
#define FC_NUM_WAVES     63
#define FC_NUM_ROTATIONS 15

/* Consumables (standard Fight Caves loadout) */
#define FC_MAX_SHARKS            20
#define FC_MAX_PRAYER_DOSES      32  /* 8 potions × 4 doses */

/* Tick timing */
#define FC_FOOD_COOLDOWN_TICKS    3  /* food_delay: 3 ticks */
#define FC_POTION_COOLDOWN_TICKS  2  /* drink_delay: 2 ticks (NOT 3 — separate clock from food) */
#define FC_COMBO_EAT_TICKS        1  /* karambwan combo delay after food */
#define FC_MAX_EPISODE_TICKS   200000 /* ~33 hours at 0.6s/tick — force prayer drain */
#define FC_HP_REGEN_INTERVAL     100 /* HP regen: 1 HP every 100 ticks (60 seconds) */

/* OSRS run energy uses 100 internal units per displayed percent. The canonical
 * no-supplies Masori/Twisted-bow loadout floors to 30 kg. Applying the current
 * integer formulas at level 99 Agility loses 60 units per two-step running
 * tick and restores 24 units per other tick. */
#define FC_RUN_ENERGY_MAX       10000
#define FC_RUN_ENERGY_MIN_START   100
#define FC_RUN_AGILITY_LEVEL        99
#define FC_RUN_WEIGHT_KG            30
#define FC_RUN_ENERGY_DRAIN \
    ((60 + (67 * FC_RUN_WEIGHT_KG) / 64) * \
     (300 - FC_RUN_AGILITY_LEVEL) / 300)
#define FC_RUN_ENERGY_RESTORE \
    (15 + FC_RUN_AGILITY_LEVEL / 10)

/* Player base stats — defined in fc_player_init.h */
/* ======================================================================== */
/* Pending Hit (projectile in flight or delayed melee)                       */
/* ======================================================================== */

/*
 * Attacks queue a PendingHit with a tick delay before damage applies.
 * Prayer blocking is locked into the pending hit before impact:
 *   - normal Fight Caves NPCs snapshot prayer on the attack tick
 *   - Jad special-cases ranged/magic to snapshot shortly after the tell
 * The projectile/hitsplat can still land later, but prayer no longer re-checks
 * the live player prayer on impact.
 *
 * This models OSRS projectile flight:
 *   Melee:  0 tick delay (instant)
 *   Ranged: 1 + floor((3 + distance) / 6) ticks
 *   Magic:  1 + floor((1 + distance) / 3) ticks
 *
 * PR 2 note: Protection prayer blocks 100% damage if correct style match.
 * Unlike PvP (60% reduction), PvM prayer fully blocks the hit.
 * Exception: Jad attacks always deal damage if WRONG prayer is active.
 */
typedef struct {
    int active;           /* 1 if this slot is in use */
    int damage;           /* pre-prayer damage roll (0 = miss) */
    int ticks_remaining;  /* ticks until hit resolves */
    int attack_style;     /* FcAttackStyle of the incoming hit */
    int source_npc_idx;   /* index into FcState.npcs[] of the attacker */
    int prayer_drain;     /* base prayer drain in tenths (Tz-Kih adds final damage) */
    int prayer_snapshot;  /* prayer locked for this hit; -1 = snapshot pending */
    int prayer_lock_tick; /* first tick on which prayer_snapshot should be filled */
} FcPendingHit;

/* ======================================================================== */
/* Player                                                                    */
/* ======================================================================== */

#define FC_INVENTORY_SLOTS 28
#define FC_EQUIPMENT_SLOTS 14

typedef struct {
    int item_id;                 /* 0 denotes an empty slot */
    int quantity;
    int charges;                 /* loaded darts remain with the blowpipe */
} FcItemStack;

typedef struct {
    /* Position */
    int x, y;

    /* Vitals (in tenths for precision: 700 = 70.0 HP) */
    int current_hp, max_hp;
    int current_prayer, max_prayer;

    /* Active prayer */
    int prayer;                /* FcPrayer enum: final/live overhead */
    int prayer_at_tick_start;  /* immutable broad-style protection snapshot */

    /* Prayer drain counter (OSRS counter-based system from PrayerDrain.kt):
     * Each tick: counter += active prayer drain rate (12 for protect prayers).
     * When counter > resistance (60 + 2*prayer_bonus): drain 1 point, counter -= resistance.
     * This field is an integer accumulator, NOT in tenths. */
    int prayer_drain_counter;

    /* Consumables */
    int sharks_remaining;
    int prayer_doses_remaining;

    /* Timers (tick countdown, 0 = ready) */
    int attack_timer;
    int food_timer;
    int potion_timer;
    int combo_timer;    /* karambwan combo eat delay */

    /* Run */
    int run_energy;     /* 0-FC_RUN_ENERGY_MAX (100.00%) */
    int is_running;     /* 1 if run mode active */

    /* Combat stats (from FightCaveEpisodeInitializer.kt) */
    int attack_level;
    int strength_level;
    int defence_level;
    int ranged_level;
    int prayer_level;
    int magic_level;
    int weapon_kind;
    int weapon_uses_ammo;
    int crystal_piece_mask;
    int weapon_speed;
    int weapon_range;

    /* Equipment bonuses (exact values from Void 634 cache item definitions) */
    int ranged_attack_bonus;
    int ranged_strength_bonus;
    int defence_stab, defence_slash, defence_crush;
    int defence_magic, defence_ranged;
    int prayer_bonus;

    /* Ammo */
    int ammo_count;

    /* HP regen counter (ticks since last regen) */
    int hp_regen_counter;

    /* Click-to-move route (like RSMod RouteDestination / Void walkTo).
     * Set once on click, consumed one step per tick until empty.
     * When route_len == route_idx, player stands still. */
    #define FC_MAX_ROUTE 64
    int route_x[FC_MAX_ROUTE];
    int route_y[FC_MAX_ROUTE];
    int route_len;      /* total steps in current route */
    int route_idx;      /* next step to consume (0..route_len-1) */

    /* Facing direction — angle in degrees, set each movement step */
    float facing_angle;

    /* Attack target — NPC array index, or -1 for none. */
    int attack_target_idx;
    /* 1 = player explicitly clicked this NPC (approach + attack).
     * 0 = auto-retaliate set target (attack in place only, no approach). */
    int approach_target;
    int approach_target_x;
    int approach_target_y;
    int approach_target_size;

    /* Pending hits (from NPC attacks in flight) */
    FcPendingHit pending_hits[FC_MAX_PENDING_HITS];
    int num_pending_hits;

    /* Per-tick event flags (cleared each tick, used for obs/reward/hitsplats) */
    int damage_taken_this_tick;
    int hit_style_this_tick;    /* FcAttackStyle of the last hit that resolved this tick */
    int hit_source_npc_type;    /* FcNpcType of the NPC that landed the last hit this tick */
    int hit_locked_prayer_this_tick; /* FcPrayer snapshot used for the last resolved hit */
    int hit_blocked_this_tick;  /* 1 if the last resolved hit this tick was prayer-blocked */
    int hit_landed_this_tick;
    int food_eaten_this_tick;
    int potion_used_this_tick;
    int prayer_changed_this_tick;

    /* Cumulative stats (for reward/logging) */
    int total_damage_taken;
    int total_food_eaten;
    int total_potions_used;
    FcItemStack inventory[FC_INVENTORY_SLOTS];
    FcItemStack equipment[FC_EQUIPMENT_SLOTS];
    int melee_attack_bonus, melee_strength_bonus;
    int selected_food_slot, selected_potion_slot;
} FcPlayer;

/* ======================================================================== */
/* NPC                                                                       */
/* ======================================================================== */

typedef struct {
    /* Identity */
    int active;       /* 1 if alive and in the arena */
    int npc_type;     /* FcNpcType */
    int spawn_index;  /* unique index across the episode, stable for NPC slot ordering */

    /* Position */
    int x, y;
    int size;         /* tile size: 1 for most, 2+ for larger NPCs */

    /* Vitals */
    int current_hp, max_hp;
    int is_dead;
    int death_timer;      /* ticks remaining before despawn (0 = despawn immediately) */

    /* Combat */
    int attack_style;       /* FcAttackStyle: what this NPC attacks with */
    int attack_timer;       /* tick countdown to next attack */
    int attack_speed;       /* ticks between attacks */
    int attack_range;       /* tile distance for ranged/magic, 1 for melee */

    /* AI */
    int movement_speed;     /* 1 = walk, 2 = run */

    /* NPC healing */
    int heal_timer;         /* ticks until next independent heal attempt (Yt-HurKot) */
    int heal_amount;        /* HP healed per proc */

    /* Yt-HurKot (Jad healer) */
    int healer_distracted;  /* permanent player aggro after the healer is tagged */
    int heal_target_idx;    /* NPC index of the entity being healed (Jad) */
    int is_respawned_jad_healer; /* 1 for generations spawned after the first */

    /* Per-tick event flags */
    int damage_taken_this_tick;
    int prayer_drain_dealt_this_tick;
    int healing_received_this_tick;
    int healing_given_this_tick;
    int healed_by_mejkot_this_tick;
    int healed_by_hurkot_this_tick;
    int healed_self_this_tick;
    int died_this_tick;

    /* Pending hits (player attacks in flight toward this NPC) */
    FcPendingHit pending_hits[FC_MAX_PENDING_HITS];
    int num_pending_hits;
} FcNpc;

/* ======================================================================== */
/* Wave entry (spawn table row)                                              */
/* ======================================================================== */

#define FC_MAX_SPAWNS_PER_WAVE 6

typedef struct {
    int npc_types[FC_MAX_SPAWNS_PER_WAVE];  /* FcNpcType per spawn */
    int num_spawns;
} FcWaveEntry;

/* ======================================================================== */
/* Render entity (value type for viewer — filled by fc_fill_render_entities) */
/* ======================================================================== */

/*
 * The viewer never reads FcPlayer/FcNpc directly. It receives an array of
 * these render entities via the fc_fill_render_entities callback.
 * This decouples rendering from simulation internals.
 */
typedef struct {
    int entity_type;    /* FcEntityType */
    int npc_type;       /* FcNpcType (0 for player) */
    int x, y;
    int size;           /* tile size */
    int current_hp, max_hp;
    int attack_style;   /* FcAttackStyle: current/last attack style */
    int prayer;         /* FcPrayer: active prayer (player only) */
    int is_dead;

    /* Per-tick events for hitsplat/animation rendering */
    int damage_taken_this_tick;
    int healing_received_this_tick;
    int hit_landed_this_tick;
    int died_this_tick;

    int npc_slot;       /* NPC array index (for stable interpolation lookup) */
} FcRenderEntity;

#define FC_MAX_RENDER_ENTITIES (1 + FC_MAX_NPCS)  /* player + NPCs */

/* ======================================================================== */
/* Per-tick rendering events                                                 */
/* ======================================================================== */

#define FC_MAX_RENDER_MOVE_WAYPOINTS 2
#define FC_MAX_RENDER_NPC_ATTACKS FC_MAX_NPCS
#define FC_MAX_RENDER_HITS 32

typedef struct {
    int npc_slot;
    int npc_type;
    int attack_style;
    int source_x;
    int source_y;
    int source_size;
    int target_x;
    int target_y;
    int hit_delay_ticks;
    int prayer_lock_tick;
    int hit_queued;
} FcRenderNpcAttack;

typedef struct {
    int target_entity_type;  /* FcEntityType */
    int target_npc_slot;     /* -1 when the player is the target */
    int source_npc_slot;     /* -1 when the player is the source */
    int attack_style;
    int damage;
    int blocked;
} FcRenderHit;

/*
 * Authoritative, read-only facts captured while a simulation tick executes.
 * These events let renderers reproduce transitions whose intermediate state
 * is no longer present in the final FcState snapshot. They do not participate
 * in combat, observations, rewards, or action validation.
 */
typedef struct {
    /* Prayer transition. A successful flick can begin and end on the same
     * prayer, so final player.prayer alone cannot represent the off edge. */
    int prayer_prior;
    int prayer_final;
    int prayer_off_performed;
    int prayer_on_succeeded;
    int prayer_flick_performed;

    /* Player ranged attack, captured at its authoritative stationary launch
     * tile before later phases can change target state or facing direction. */
    int player_attack_fired;
    int player_attack_source_x;
    int player_attack_source_y;
    int player_attack_target_npc_slot;
    int player_attack_target_x;
    int player_attack_target_y;
    int player_attack_target_size;
    int player_attack_hit_delay_ticks;

    /* NPC attacks captured when the NPC AI commits the attack. There can be
     * at most one launch per NPC during a simulation tick. */
    int npc_attack_count;
    FcRenderNpcAttack npc_attacks[FC_MAX_RENDER_NPC_ATTACKS];

    /* Hits captured when pending-hit resolution consumes them. This includes
     * misses and prayer-blocked hits whose final damage is zero. */
    int hit_count;
    FcRenderHit hits[FC_MAX_RENDER_HITS];

    /* Exact player movement path consumed during this tick. Waypoints contain
     * each successfully reached tile, including the final tile. */
    int player_move_start_x;
    int player_move_start_y;
    int player_move_waypoint_count;
    int player_move_waypoint_x[FC_MAX_RENDER_MOVE_WAYPOINTS];
    int player_move_waypoint_y[FC_MAX_RENDER_MOVE_WAYPOINTS];
} FcRenderEvents;

/* ======================================================================== */
/* Top-level simulation state                                                */
/* ======================================================================== */

typedef struct {
    FcPlayer player;
    FcNpc npcs[FC_MAX_NPCS];

    /* Compile-selected loadout copied into state for diagnostics/contracts. */
    int active_loadout;

    /* Viewer-facing transition facts for the most recently completed tick. */
    FcRenderEvents render_events;

    /* Wave progression */
    int current_wave;       /* 1-indexed: 1..63. 0 = not started */
    int rotation_id;        /* 0..14, selected at episode start */
    int npcs_remaining;     /* count of active (alive) NPCs in current wave */
    int total_npcs_killed;
    int next_spawn_index;   /* monotonic counter for NPC spawn ordering */

    /* Tick */
    int tick;

    /* Terminal */
    int terminal;           /* FcTerminalCode */

    /* RNG — XORshift32, single state, seeded at reset */
    uint32_t rng_state;
    uint32_t rng_seed;      /* saved for replay */

    /* Whole-tile movement collision (1 = standable, 0 = blocked). */
    uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];

    /* Directional movement walls, independent from whole-tile blocking. */
    uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];

    /* Directional projectile collision, independent from movement. */
    uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];

    /* Jad healer state */
    int jad_healers_spawned;  /* 1 until Jad has been healed back to full HP */
    int jad_healer_spawn_generations; /* successful healer generations spawned this wave */

    /* Per-tick aggregated event flags (for reward features) */
    int damage_dealt_this_tick;
    int hits_landed_this_tick;   /* count of player pending-hits that dealt damage this tick */
    int damage_taken_this_tick;
    int prayer_lost_this_tick;   /* prayer points lost this tick, in tenths */
    int overhead_prayer_lost_this_tick; /* passive overhead drain, in tenths */
    int tz_kih_prayer_drain_this_tick; /* Tz-Kih share of prayer loss, in tenths */
    int npcs_killed_this_tick;
    int respawned_jad_healers_killed_this_tick;
    int wave_just_cleared;
    int jad_damage_this_tick;
    int jad_killed;
    int correct_jad_prayer;
    int wrong_jad_prayer;
    int correct_danger_prayer;
    int wrong_danger_prayer;
    int attack_attempt_this_tick;
    int invalid_action_this_tick;
    int invalid_action_class_this_tick[FC_INVALID_ACTION_CLASS_COUNT];
    int movement_this_tick;
    int idle_this_tick;
    int food_used_this_tick;
    int prayer_potion_used_this_tick;
    int jad_heal_procs_this_tick;   /* number of Yt-HurKot heal procs that restored Jad HP */
    int npc_heal_procs_this_tick;   /* number of NPC heal procs that restored any NPC HP */
    int npc_heal_amount_this_tick;  /* total NPC HP restored this tick */
    int mejkot_heal_amount_this_tick; /* total HP restored by Yt-MejKot this tick */
    int jad_heal_amount_this_tick;    /* total HP restored to Jad this tick */

    /* Derived progression state, maintained by reward/runtime code for obs. */
    float progress_required_work_start;
    float progress_required_work_remaining;
    float progress_current_wave_progress;
    float progress_cave_progress;
    int progress_ticks_since_positive;

    /* Episode-level analytics (cumulative, zeroed on fc_reset via memset) */
    int ep_ticks_pray_melee;    /* ticks with protect melee active */
    int ep_ticks_pray_range;    /* ticks with protect range active */
    int ep_ticks_pray_magic;    /* ticks with protect magic active */
    int ep_correct_blocks;      /* hits correctly blocked by matching prayer */
    int ep_wrong_prayer_hits;   /* hits where prayer active but wrong type */
    int ep_no_prayer_hits;      /* hits where no prayer was active */
    int ep_damage_blocked;      /* total damage prevented by correct prayer */
    int ep_prayer_switches;     /* number of prayer changes */
    int ep_pots_used;           /* prayer pot doses consumed */
    int ep_pots_wasted;         /* doses consumed when prayer was above 20% */
    int ep_pot_pre_prayer_sum;  /* prayer points before each potion use */
    int ep_food_eaten;          /* sharks consumed */
    int ep_food_pre_hp_sum;     /* HP before each food use */
    int ep_food_overhealed;     /* sharks that overhealed (wasted HP) */
    int ep_pots_overrestored;   /* doses that over-restored (wasted prayer) */
    int ep_tokxil_melee_ticks;  /* ticks with any Tok-Xil at melee distance */
    int ep_ketzek_melee_ticks;  /* ticks with any Ket-Zek at melee distance */
    int ep_attack_ready_ticks;  /* ticks where attack cooldown was ready */
    int ep_attack_attempt_ticks;/* ready ticks where a real attack fired */
    int ep_invalid_action_classes[FC_INVALID_ACTION_CLASS_COUNT];
    int ep_damage_to_npc_type[NPC_TYPE_COUNT];       /* player damage by NPC type */
    int ep_resolved_hits_to_npc_type[NPC_TYPE_COUNT];/* all resolved player hitsplats, including 0s */
    int ep_damaging_hits_to_npc_type[NPC_TYPE_COUNT];/* resolved player hitsplats with damage > 0 */
    int ep_attack_cycles_to_npc_type[NPC_TYPE_COUNT];/* actual attack cycles fired by target type */
    int ep_target_ticks_by_npc_type[NPC_TYPE_COUNT]; /* ticks with active attack target by type */
    int ep_target_held_ticks;        /* ticks with any active attack target */
    int ep_no_target_ticks;          /* ticks with NPCs alive and no active attack target */
    int ep_target_in_range_los_ticks;/* target held, in range, and line of sight available */
    int ep_target_out_of_range_or_los_ticks; /* target held but cannot currently fire */
    int ep_attack_cooldown_wait_ticks;       /* target held and fireable, but weapon cooling down */
    int ep_ready_but_no_attack_ticks;        /* target held/fireable/ready but no attack cycle launched */
    int ep_action_move_idle_ticks;
    int ep_action_move_walk_ticks;
    int ep_action_move_run_ticks;
    int ep_action_attack_none_ticks;
    int ep_action_attack_target_ticks;
    int ep_action_prayer_noop_ticks;
    int ep_action_prayer_cmd_ticks;
    int ep_reached_wave_63;     /* 1 if episode reached Jad wave */
    int ep_jad_killed;          /* 1 if Jad died at any point this episode */
    int wave_start_tick;        /* tick when current wave was spawned */
    int ep_max_wave_ticks;      /* longest single wave duration in ticks */
    int ep_max_wave_ticks_wave; /* which wave number that was */
} FcState;


/* Contracts */

/* Machine-readable training-contract metadata. These identifiers describe
 * compiled semantics, not tunable reward weights. A selected runnable config
 * may override FC_REWARD_VERSION at build time when it preserves its own
 * reward-family name under the same Prayer parity semantics. */
#define FC_CONTRACT_DUMP_SCHEMA_VERSION 1
#ifndef FC_OBSERVATION_VERSION
#define FC_OBSERVATION_VERSION "fight_caves_puffer_policy_obs_v9_run_energy_prayer_timing_mask8_no_supplies"
#endif
#ifndef FC_ACTION_VERSION
#define FC_ACTION_VERSION "fight_caves_multidiscrete_3_head_no_supplies_v4_run_energy_prayer8_stationary_attack_tick"
#endif
#ifndef FC_REWARD_VERSION
#define FC_REWARD_VERSION "fight_caves_v4_progress_npc_heal_penalty_m0005_prayer_snapshot_flick_drain"
#endif
#ifndef FC_PRAYER_TIMING_VERSION
#define FC_PRAYER_TIMING_VERSION "fight_caves_prayer_timing_v1_tick_start_snapshot_flick_drain_jad_lock"
#endif

/*
 * fc_contracts.h — Observation, action, reward, and mask contracts.
 *
 * SINGLE SOURCE OF TRUTH for all buffer layouts. Python reads these constants
 * (via codegen or manual sync) — it never redefines them.
 *
 * All observations are float32, normalized to [0,1]. All actions are int32 per
 * head. The canonical core buffer contains policy observations followed by raw
 * reward features. The Puffer adapter exposes its own checkpoint-compatible
 * model input, documented below; reward features are not model inputs.
 */

/* ======================================================================== */
/* Observation layout                                                        */
/* ======================================================================== */

/*
 * The optional complete core buffer is split into three contiguous regions:
 *
 *   [0 .. FC_POLICY_OBS_SIZE-1]                          policy observations
 *   [FC_POLICY_OBS_SIZE .. FC_POLICY_OBS_SIZE+FC_REWARD_FEATURES-1]  reward features
 *   [FC_TOTAL_OBS .. FC_TOTAL_OBS+FC_ACTION_MASK_SIZE-1] action mask
 *
 * Core callers may use policy observations plus raw reward features
 * (FC_TOTAL_OBS), then append the seven-head core mask for a 475-float
 * diagnostic buffer (FC_OBS_SIZE).
 *
 * Puffer does not receive that diagnostic layout. Its model input is 320
 * floats: FC_POLICY_OBS_SIZE (286) followed by FC_PUFFER_MASK_SIZE (34) mask
 * bits for the no-supplies policy heads. Those same 34 bits are also supplied
 * through PufferLib's native action-mask channel. The 20 reward features and
 * the masks for core-only action heads are not exposed to the model.
 */

/* --- Player features (23 floats) --- */
#define FC_OBS_PLAYER_START     0
#define FC_OBS_PLAYER_HP        0   /* current_hp / max_hp */
#define FC_OBS_PLAYER_PRAYER    1   /* current_prayer / max_prayer */
#define FC_OBS_PLAYER_X         2   /* x / ARENA_WIDTH */
#define FC_OBS_PLAYER_Y         3   /* y / ARENA_HEIGHT */
#define FC_OBS_PLAYER_ATK_TIMER 4   /* attack_timer / max_attack_timer */
#define FC_OBS_PLAYER_PRAY_MEL  5   /* prayer == PROTECT_MELEE (0 or 1) */
#define FC_OBS_PLAYER_PRAY_RNG  6   /* prayer == PROTECT_RANGE (0 or 1) */
#define FC_OBS_PLAYER_PRAY_MAG  7   /* prayer == PROTECT_MAGIC (0 or 1) */
#define FC_OBS_PLAYER_SHARKS    8   /* sharks_remaining / MAX_SHARKS */
#define FC_OBS_PLAYER_DOSES     9   /* prayer_doses / MAX_DOSES */
#define FC_OBS_PLAYER_IN_MEL_1T 10  /* normalized count of melee hits landing in 1 tick */
#define FC_OBS_PLAYER_IN_RNG_1T 11  /* normalized count of ranged hits landing in 1 tick */
#define FC_OBS_PLAYER_IN_MAG_1T 12  /* normalized count of magic hits landing in 1 tick */
#define FC_OBS_PLAYER_IN_MEL_2T 13  /* normalized count of melee hits landing in 2 ticks */
#define FC_OBS_PLAYER_IN_RNG_2T 14  /* normalized count of ranged hits landing in 2 ticks */
#define FC_OBS_PLAYER_IN_MAG_2T 15  /* normalized count of magic hits landing in 2 ticks */
#define FC_OBS_PLAYER_TARGET    16  /* attack_target NPC slot index / 8 (0=no target, 0.125-1.0=slot 0-7) */
#define FC_OBS_PLAYER_PRAY_DDL_MEL 17  /* style-specific prayer deadline urgency for actionable melee hits */
#define FC_OBS_PLAYER_PRAY_DDL_RNG 18  /* style-specific prayer deadline urgency for actionable ranged hits */
#define FC_OBS_PLAYER_PRAY_DDL_MAG 19  /* style-specific prayer deadline urgency for actionable magic hits */
#define FC_OBS_PLAYER_PRAYER_LOST 20   /* total prayer lost this tick / max_prayer */
#define FC_OBS_PLAYER_OVERHEAD_PRAYER_LOST 21 /* 1 if passive overhead drain removed prayer this tick */
#define FC_OBS_PLAYER_RUN_ENERGY 22 /* current run energy / FC_RUN_ENERGY_MAX */
#define FC_OBS_PLAYER_SIZE      23

/* --- Per-NPC features (31 floats x 8 visible NPCs = 248 floats) --- */
/*
 * NPC slot ordering — deterministic rules for the 8 visible NPC slots:
 *
 *   1. Only active (alive) NPCs are eligible for slots.
 *   2. Sort eligible NPCs by:
 *      a. Chebyshev distance to the nearest tile of the NPC footprint,
 *         ascending (closest first).
 *      b. On distance tie: spawn_index ascending (earlier spawns first).
 *   3. Take the first 8 from the sorted list.
 *   4. If fewer than 8 active NPCs, remaining slots are zeroed (valid=0).
 *
 * Overflow behavior: If more than 8 NPCs are alive, the 8 closest are visible.
 * NPCs beyond slot 8 are still simulated (they attack, move, take damage) but
 * are not included in the observation. The agent cannot directly target overflow
 * NPCs via the ATTACK action head, but area/splash effects may still hit them.
 *
 * The spawn_index tiebreaker ensures deterministic ordering when distances are
 * equal, which is critical for replay consistency and debug reproducibility.
 */
#define FC_OBS_NPC_START        FC_OBS_PLAYER_SIZE  /* 23 */
#define FC_OBS_NPC_STRIDE       31
#define FC_OBS_NPC_SLOTS        8   /* FC_VISIBLE_NPCS */

/* Per-NPC feature offsets within stride.
 *
 * Telegraph bits (TELE_MELEE/RANGED/MAGIC) are one-hot: the style this NPC
 * would use RIGHT NOW based on distance (no LOS check). Stays on even when
 * LOS=0 so the agent can prepare prayer for when LOS resumes. Untagged
 * Yt-HurKot does not telegraph; a tagged healer telegraphs melee. Jad
 * telegraphs only after it commits a pending_hit (style is stochastic).
 * All zero when NPC slot is empty/dead.
 */
#define FC_NPC_VALID            0   /* 1 if slot occupied, 0 if empty */
#define FC_NPC_X                1   /* x / ARENA_WIDTH */
#define FC_NPC_Y                2   /* y / ARENA_HEIGHT */
#define FC_NPC_HP               3   /* current_hp / max_hp */
#define FC_NPC_DISTANCE         4   /* chebyshev distance to NPC footprint / ARENA_WIDTH */
#define FC_NPC_TELE_MELEE       5   /* one-hot: NPC would melee at current distance */
#define FC_NPC_TELE_RANGED      6   /* one-hot: NPC would range at current distance */
#define FC_NPC_TELE_MAGIC       7   /* one-hot: NPC would magic at current distance */
#define FC_NPC_ATK_TIMER        8   /* attack_timer / attack_speed */
#define FC_NPC_LOS              9   /* 1 if player has line of sight, 0 if blocked */
#define FC_NPC_PENDING_STYLE    10  /* incoming attack style (0=none, 0.33/0.67/1.0) */
#define FC_NPC_PENDING_TICKS    11  /* ticks until incoming attack resolves (normalized) */
#define FC_NPC_TYPE_TZ_KIH      12  /* one-hot NPC identity: Tz-Kih */
#define FC_NPC_TYPE_TZ_KEK      13  /* one-hot NPC identity: Tz-Kek */
#define FC_NPC_TYPE_TZ_KEK_SM   14  /* one-hot NPC identity: small Tz-Kek */
#define FC_NPC_TYPE_TOK_XIL     15  /* one-hot NPC identity: Tok-Xil */
#define FC_NPC_TYPE_YT_MEJKOT   16  /* one-hot NPC identity: Yt-MejKot */
#define FC_NPC_TYPE_KET_ZEK     17  /* one-hot NPC identity: Ket-Zek */
#define FC_NPC_TYPE_TZTOK_JAD   18  /* one-hot NPC identity: TzTok-Jad */
#define FC_NPC_TYPE_YT_HURKOT   19  /* one-hot NPC identity: Yt-HurKot */
#define FC_NPC_PENDING_PRAYER_WINDOW   20  /* 1 if current prayer action can still affect this pending hit */
#define FC_NPC_PENDING_PRAYER_DEADLINE 21  /* urgency until prayer locks: 1=act now, 0=no actionable hit */
#define FC_NPC_PRAYER_DRAIN_DEALT      22  /* actual prayer drained this tick / source maximum */
#define FC_NPC_HEAL_RECEIVED           23  /* HP restored to this NPC this tick / max_hp */
#define FC_NPC_HEAL_GIVEN              24  /* HP this NPC restored this tick / configured heal_amount */
#define FC_NPC_HEALED_BY_MEJKOT        25  /* 1 if Yt-MejKot restored this NPC's HP this tick */
#define FC_NPC_HEALED_BY_HURKOT        26  /* 1 if Yt-HurKot restored this NPC's HP this tick */
#define FC_NPC_HEALED_SELF             27  /* 1 if this NPC restored its own HP this tick */
#define FC_NPC_TARGETS_PLAYER          28  /* 1 if the NPC's current movement/combat target is the player */
#define FC_NPC_HEAL_COOLDOWN           29  /* normalized time until the next possible healing cycle */
#define FC_NPC_KILL_REWARD_ELIGIBLE    30  /* 1 if this NPC's death would pay FC_RWD_NPC_KILL */

#define FC_OBS_NPC_TOTAL        (FC_OBS_NPC_STRIDE * FC_OBS_NPC_SLOTS)  /* 248 */

/* --- Wave/meta features (15 floats) --- */
#define FC_OBS_META_START       (FC_OBS_NPC_START + FC_OBS_NPC_TOTAL)  /* 271 */
#define FC_OBS_META_WAVE        0   /* current_wave / NUM_WAVES */
#define FC_OBS_META_ROTATION    1   /* rotation_id / NUM_ROTATIONS */
#define FC_OBS_META_REMAINING   2   /* npcs_remaining / MAX_NPCS */
#define FC_OBS_META_PRAY_DRAIN  3   /* prayer_drain_counter / drain_resistance */
#define FC_OBS_META_IN_MEL_3T   4   /* normalized count of melee hits landing in 3 ticks */
#define FC_OBS_META_IN_RNG_3T   5   /* normalized count of ranged hits landing in 3 ticks */
#define FC_OBS_META_IN_MAG_3T   6   /* normalized count of magic hits landing in 3 ticks */
#define FC_OBS_META_DMG_T_TICK  7   /* damage_taken_this_tick / max_hp */
#define FC_OBS_META_WAVE_CLR    8   /* wave_just_cleared (0 or 1) */
#define FC_OBS_META_CAVE_PROG   9   /* derived cave progress, [0,1] */
#define FC_OBS_META_WAVE_PROG   10  /* derived current-wave progress, [0,1] */
#define FC_OBS_META_WORK_REM    11  /* required_work_remaining / required_work_start */
#define FC_OBS_META_NO_PROG     12  /* ticks_since_positive_progress / 2400 */
#define FC_OBS_META_NPC_HEALING 13  /* HP restored this tick / required work at wave start */
#define FC_OBS_META_REWARDABLE_NPC_KILL 14  /* 1 if at least one eligible NPC died this tick */
#define FC_OBS_META_SIZE        15

/* --- Policy observation total --- */
#define FC_POLICY_OBS_SIZE      (FC_OBS_PLAYER_SIZE + FC_OBS_NPC_TOTAL + FC_OBS_META_SIZE)  /* 286 */

/* --- Reward features (20 floats) --- */
/*
 * These are packed AFTER policy observations in the same buffer.
 * The trainer reads them for reward shaping and logging.
 * The policy DOES NOT consume these by default.
 * Python applies configurable shaping weights to produce the scalar reward.
 */
#define FC_REWARD_START         FC_POLICY_OBS_SIZE  /* 286 */
#define FC_RWD_DAMAGE_DEALT     0   /* NPC HP reduced this tick (normalized) */
#define FC_RWD_DAMAGE_TAKEN     1   /* player HP reduced this tick */
#define FC_RWD_NPC_KILL         2   /* rewardable deaths; excludes respawned Jad healers */
#define FC_RWD_WAVE_CLEAR       3   /* all wave NPCs dead */
#define FC_RWD_JAD_DAMAGE       4   /* Jad HP reduced this tick */
#define FC_RWD_JAD_KILL         5   /* Jad defeated */
#define FC_RWD_PLAYER_DEATH     6   /* player HP <= 0 */
#define FC_RWD_CAVE_COMPLETE    7   /* all 63 waves cleared */
#define FC_RWD_FOOD_USED        8   /* shark consumed this tick */
#define FC_RWD_PRAYER_POT_USED  9   /* potion consumed this tick */
#define FC_RWD_CORRECT_JAD_PRAY 10  /* Jad-specific correct-block diagnostic */
#define FC_RWD_WRONG_JAD_PRAY   11  /* Jad-specific wrong-block diagnostic */
#define FC_RWD_INVALID_ACTION   12  /* rejected/masked action attempted */
#define FC_RWD_MOVEMENT         13  /* walk/run action executed */
#define FC_RWD_IDLE             14  /* wait/idle action */
#define FC_RWD_TICK_PENALTY     15  /* fires every tick (time discount) */
#define FC_RWD_CORRECT_DANGER_PRAY 16  /* prayer matched any resolved NPC style, including Jad */
#define FC_RWD_WRONG_DANGER_PRAY   17  /* prayer missed resolved non-Jad NPC style */
#define FC_RWD_ATTACK_ATTEMPT   18  /* valid attack cycle launched this tick */
#define FC_RWD_PRAYER_LOST      19  /* prayer points lost this tick */
#define FC_REWARD_FEATURES      20

/* --- Total observation (policy obs + reward features) --- */
#define FC_TOTAL_OBS            (FC_POLICY_OBS_SIZE + FC_REWARD_FEATURES)  /* 306 */

/* ======================================================================== */
/* Action space — 7 canonical core heads                                     */
/* ======================================================================== */

/*
 * Canonical action interface shared by:
 *   - Headless RL training
 *   - Human playable viewer (click/keyboard → action buffer)
 *   - Replay playback (recorded action buffer per tick)
 *   - Policy playback (policy output → action buffer)
 *
 * Human input (click-to-move, click-to-attack) must translate into these
 * head values. The viewer enqueues canonical MOVE steps per tick via
 * pathfinding; it NEVER bypasses the action interface to mutate state.
 */

#define FC_NUM_ACTION_HEADS     7

/* Head 0: MOVE — directional tile movement (low-level) */
/*
 *   0     = idle (no movement)
 *   1-8   = walk 1 tile (N, NE, E, SE, S, SW, W, NW)
 *   9-16  = run 2 tiles (N, NE, E, SE, S, SW, W, NW)
 *
 * For fine-grained per-tick control. Ignored when a BFS route is active
 * (set via heads 5+6 or viewer click-to-move).
 */
#define FC_MOVE_DIM             17
#define FC_MOVE_IDLE             0
#define FC_MOVE_WALK_N           1
#define FC_MOVE_WALK_NE          2
#define FC_MOVE_WALK_E           3
#define FC_MOVE_WALK_SE          4
#define FC_MOVE_WALK_S           5
#define FC_MOVE_WALK_SW          6
#define FC_MOVE_WALK_W           7
#define FC_MOVE_WALK_NW          8
#define FC_MOVE_RUN_N            9
#define FC_MOVE_RUN_NE          10
#define FC_MOVE_RUN_E           11
#define FC_MOVE_RUN_SE          12
#define FC_MOVE_RUN_S           13
#define FC_MOVE_RUN_SW          14
#define FC_MOVE_RUN_W           15
#define FC_MOVE_RUN_NW          16

/* Direction offset tables (dx, dy) for walk actions 1-8 */
static const int FC_MOVE_DX[17] = {
    0,                             /* idle */
    0, 1, 1, 1, 0, -1, -1, -1,    /* walk: N, NE, E, SE, S, SW, W, NW */
    0, 2, 2, 2, 0, -2, -2, -2     /* run:  N, NE, E, SE, S, SW, W, NW (2-tile target) */
};
static const int FC_MOVE_DY[17] = {
    0,
    1, 1, 0, -1, -1, -1, 0, 1,    /* walk */
    2, 2, 0, -2, -2, -2, 0, 2     /* run */
};

/* Head 1: ATTACK — target a visible NPC by slot index */
#define FC_ATTACK_DIM            9  /* 0=none, 1-8=NPC slot 0-7 */
#define FC_ATTACK_NONE           0

/* Head 2: PRAYER — toggle protection prayer */
/*
 *   0 = no change
 *   1 = prayer off
 *   2 = protect from magic
 *   3 = protect from missiles (ranged)
 *   4 = protect from melee
 *   5 = explicit OFF edge, then protect from magic
 *   6 = explicit OFF edge, then protect from missiles (ranged)
 *   7 = explicit OFF edge, then protect from melee
 *
 * PR 2 note — PvM prayer semantics:
 *   Correct protection prayer BLOCKS 100% of the matching NPC attack style.
 *   This is NOT the PvP 60% reduction. Full block is standard OSRS PvM behavior.
 *   Only exception: attacking while wrong prayer is active against Jad still takes
 *   full damage. Prayer must be switched before the hit's snapshot/lock tick.
 */
#define FC_PRAYER_DIM            8
#define FC_PRAYER_NO_CHANGE      0
#define FC_PRAYER_OFF            1
#define FC_PRAYER_MAGIC          2
#define FC_PRAYER_RANGE          3
#define FC_PRAYER_MELEE          4
#define FC_PRAYER_FLICK_MAGIC    5
#define FC_PRAYER_FLICK_RANGE    6
#define FC_PRAYER_FLICK_MELEE    7

/* Head 3: EAT */
#define FC_EAT_DIM               3
#define FC_EAT_NONE              0
#define FC_EAT_SHARK             1
#define FC_EAT_COMBO             2  /* karambwan combo eat (if available) */

/* Head 4: DRINK */
#define FC_DRINK_DIM             2
#define FC_DRINK_NONE            0
#define FC_DRINK_PRAYER_POT      1

/* Head 5: MOVE_TARGET_X — BFS pathfinding target X coordinate (high-level) */
/*
 *   0     = no pathfind target (use directional head 0 instead)
 *   1-64  = tile X coordinate 0-63
 *
 * When BOTH head 5 and head 6 are non-zero, the backend calls BFS pathfind
 * to tile (target_x-1, target_y-1) and sets the player route. This is
 * identical to a human clicking a tile in the viewer.
 *
 * The route is consumed one step per tick (or two if running). While a route
 * is active, directional actions (head 0) are ignored.
 *
 * If the exact target is unwalkable or unreachable, native move-near chooses
 * the best reachable endpoint within ten tiles. If none exists, it is a no-op.
 */
#define FC_MOVE_TARGET_X_DIM    65  /* 0=no-op, 1-64=tile x 0-63 */
#define FC_MOVE_TARGET_X_NONE    0

/* Head 6: MOVE_TARGET_Y — BFS pathfinding target Y coordinate */
#define FC_MOVE_TARGET_Y_DIM    65  /* 0=no-op, 1-64=tile y 0-63 */
#define FC_MOVE_TARGET_Y_NONE    0

/* Head dimension array (for binding.c) */
#define FC_ACT_SIZES { FC_MOVE_DIM, FC_ATTACK_DIM, FC_PRAYER_DIM, FC_EAT_DIM, FC_DRINK_DIM, FC_MOVE_TARGET_X_DIM, FC_MOVE_TARGET_Y_DIM }

/* Action head dimensions as a static array (for iteration in viewer/binding) */
static const int FC_ACTION_DIMS[FC_NUM_ACTION_HEADS] = FC_ACT_SIZES;

/* Puffer-facing no-supplies policy contract.
 * This is the single source of truth for the action heads and masks consumed
 * by Puffer training and policy replay. Core still supports all canonical
 * action heads; the Puffer policy emits the prefix below. */
#define FC_PUFFER_NUM_ATNS      3
#define FC_PUFFER_ACT_SIZES     { FC_MOVE_DIM, FC_ATTACK_DIM, FC_PRAYER_DIM }
#define FC_PUFFER_MASK_SIZE     (FC_MOVE_DIM + FC_ATTACK_DIM + FC_PRAYER_DIM)
#define FC_PUFFER_OBS_SIZE      (FC_POLICY_OBS_SIZE + FC_PUFFER_MASK_SIZE)

static const int FC_PUFFER_ACTION_DIMS[FC_PUFFER_NUM_ATNS] = FC_PUFFER_ACT_SIZES;

/* ======================================================================== */
/* Action mask                                                               */
/* ======================================================================== */

/*
 * Per-tick binary mask: 1.0 = valid, 0.0 = invalid.
 * One float per action value per head. Appended after FC_TOTAL_OBS in the buffer.
 *
 * Layout: [MOVE(17)] [ATTACK(9)] [PRAYER(8)] [EAT(3)] [DRINK(2)] [TARGET_X(65)] [TARGET_Y(65)]
 */
#define FC_ACTION_MASK_SIZE     (FC_MOVE_DIM + FC_ATTACK_DIM + FC_PRAYER_DIM + FC_EAT_DIM + FC_DRINK_DIM + FC_MOVE_TARGET_X_DIM + FC_MOVE_TARGET_Y_DIM)  /* 169 */

/* Mask region offsets within the mask buffer */
#define FC_MASK_MOVE_START       0
#define FC_MASK_ATTACK_START     FC_MOVE_DIM                                    /* 17 */
#define FC_MASK_PRAYER_START     (FC_MASK_ATTACK_START + FC_ATTACK_DIM)         /* 26 */
#define FC_MASK_EAT_START        (FC_MASK_PRAYER_START + FC_PRAYER_DIM)         /* 34 */
#define FC_MASK_DRINK_START      (FC_MASK_EAT_START + FC_EAT_DIM)              /* 37 */
#define FC_MASK_TARGET_X_START   (FC_MASK_DRINK_START + FC_DRINK_DIM)           /* 39 */
#define FC_MASK_TARGET_Y_START   (FC_MASK_TARGET_X_START + FC_MOVE_TARGET_X_DIM) /* 104 */

/* ======================================================================== */
/* Optional complete core diagnostic buffer size                             */
/* ======================================================================== */

/*
 * Total floats in the full FC backend buffer:
 *   FC_POLICY_OBS_SIZE (286) + FC_REWARD_FEATURES (20) + FC_ACTION_MASK_SIZE (169) = 475
 *
 * The PufferLib adapter does not allocate or expose this layout. It allocates
 * FC_PUFFER_OBS_SIZE (320), copies the no-supplies mask into the model input at
 * FC_POLICY_OBS_SIZE, and publishes the same flags through the native mask
 * pointer. Reward computation reads authoritative state through fc_reward.
 */
#define FC_OBS_SIZE             (FC_TOTAL_OBS + FC_ACTION_MASK_SIZE)  /* 475 */

/* ======================================================================== */
/* Normalization divisors                                                     */
/* ======================================================================== */

/*
 * Each observation feature is divided by its divisor to normalize to [0,1].
 * Divisors are defined in fc_obs.c and indexed by feature offset.
 * Policy sees normalized values only.
 */


/* Episode Summary */

/* Read-only, consumer-neutral episode metrics derived from FcState. Training
 * may aggregate these values and evaluators may serialize them, but neither
 * consumer should independently reproduce their formulas. */
typedef struct {
    int episode_length;
    int wave_reached;
    int npcs_slayed;
    float prayer_uptime_melee;
    float prayer_uptime_range;
    float prayer_uptime_magic;
    int correct_prayer;
    int wrong_prayer_hits;
    int no_prayer_hits;
    int prayer_switches;
    int damage_blocked;
    int damage_taken;
    float attack_when_ready_rate;
    int invalid_move;
    int invalid_attack;
    int invalid_prayer;
    int tokxil_melee_ticks;
    int ketzek_melee_ticks;
    int max_wave_ticks;
    int max_wave_ticks_wave;
    int reached_wave_63;
    int jad_killed;
    int player_died;
    int damage_to_npc_type[NPC_TYPE_COUNT];
    int resolved_hits_to_npc_type[NPC_TYPE_COUNT];
    int damaging_hits_to_npc_type[NPC_TYPE_COUNT];
    int attack_cycles_to_npc_type[NPC_TYPE_COUNT];
    int target_ticks_by_npc_type[NPC_TYPE_COUNT];
    int target_held_ticks;
    int no_target_ticks;
    int target_in_range_los_ticks;
    int target_out_of_range_or_los_ticks;
    int attack_cooldown_wait_ticks;
    int ready_but_no_attack_ticks;
    int action_move_idle_ticks;
    int action_move_walk_ticks;
    int action_move_run_ticks;
    int action_attack_none_ticks;
    int action_attack_target_ticks;
    int action_prayer_noop_ticks;
    int action_prayer_cmd_ticks;
} FcEpisodeSummary;

/* episode_length is supplied by the consumer because standalone simulation
 * ticks and adapter step counts can intentionally differ in tests/tools. */
void fc_episode_summary_build(const FcState* state, int episode_length,
                              FcEpisodeSummary* summary);

/* Stable lowercase suffix shared by training and evaluator metric keys. */
const char* fc_episode_npc_metric_name(int npc_type);


/* Combat */

/* OSRS accuracy formula: returns hit probability in [0,1] */
float fc_hit_chance(int att_roll, int def_roll);

/* NPC combat */
int fc_npc_attack_roll(int att_level, int att_bonus);

/* Player combat */
int fc_player_def_roll(const FcPlayer* p, FcAttackType attack_type);
int fc_player_ranged_base_attack_roll(const FcPlayer* p);
int fc_player_ranged_attack_roll(const FcPlayer* p, const FcNpc* target);
int fc_player_ranged_base_max_hit_hp(const FcPlayer* p);
int fc_player_ranged_final_max_hit_hp(const FcPlayer* p, const FcNpc* target);

/* Damage helpers accept a whole-HP maximum and return tenths storage units. */
int fc_roll_player_damage_tenths(FcState* state, int final_max_hit_hp);
int fc_roll_npc_damage_tenths(FcState* state, int final_max_hit_hp);

/* Twisted-bow staged multiplier boundaries. */
int fc_tbow_accuracy_multiplier_pct(int target_magic_level);
int fc_tbow_damage_multiplier_pct(int target_magic_level);

/* PvM prayer: returns 1 if prayer blocks the attack style (100% block) */
int fc_prayer_blocks_style(int prayer, int attack_style);

/* Distance to multi-tile NPC (Chebyshev) */
int fc_distance_to_npc(int px, int py, const FcNpc* npc);

/* Hit delay */
int fc_ranged_hit_delay(int distance);  /* player ranged projectile */
int fc_npc_hit_delay(int npc_type, int attack_style, int distance);  /* per-NPC exact timing */

/* NPC defence roll (for player accuracy against NPC) */
int fc_npc_def_roll(int def_level, int def_bonus);

/* Pending hit queue */
int fc_queue_pending_hit(FcPendingHit hits[], int* num_hits, int max_hits,
                         int damage, int ticks, int style, int source_idx,
                         int prayer_drain);

/* Resolve pending hits (call each tick) */
void fc_resolve_player_pending_hits(FcState* state);
void fc_resolve_npc_pending_hits(FcState* state, int npc_idx);


/* Npc */

/* NPC stat table entry (one per NPC_TYPE) */
typedef struct {
    int max_hp;
    int attack_style;       /* FcAttackStyle: primary style (ranged/magic for dual-mode) */
    int attack_speed;       /* ticks between attacks */
    int attack_range;       /* primary attack range (1 for melee-only, 14 for ranged/magic) */
    int melee_max_hit_tenths;
    int ranged_max_hit_tenths;
    int magic_max_hit_tenths;
    int att_level;          /* melee Attack */
    int ranged_level;
    int magic_level;        /* also the Twisted-bow target input */
    int melee_attack_bonus;
    int ranged_attack_bonus;
    int magic_attack_bonus;
    int def_level;          /* NPC defence level (for player attack accuracy) */
    int ranged_def_bonus;   /* NPC equipment defence vs Ranged */
    int melee_attack_type;  /* FcAttackType */
    int size;               /* tile footprint */
    int movement_speed;     /* 1=walk, 2=run */
    int prayer_drain;       /* base prayer drain in tenths (Tz-Kih specific) */
    int heal_amount;        /* HP healed per proc */
    int heal_interval;      /* ticks between independent Yt-HurKot heals */
} FcNpcStats;

/* Get stats for a given NPC type */
const FcNpcStats* fc_npc_get_stats(int npc_type);

/* Unit-explicit style maximum boundaries. Unsupported/invalid styles return
 * zero. The HP accessor rejects non-integral tenths values by returning zero. */
int fc_npc_max_hit_tenths_for_style(const FcNpcStats* stats, int attack_style);
int fc_npc_max_hit_hp_for_style(const FcNpcStats* stats, int attack_style);

/* Returns nonzero when every populated style maximum is valid tenths data. */
int fc_npc_stats_valid(const FcNpcStats* stats);

/* Initialize an NPC slot from type and spawn position */
void fc_npc_spawn(FcNpc* npc, int npc_type, int x, int y, int spawn_index);

/* True when the NPC could attack the player if its top-left footprint stood
 * at candidate_x,candidate_y. This checks attack style, range, melee contact,
 * and static LOS. */
int fc_npc_position_can_attack_player(const FcState* state, const FcNpc* npc,
                                      int candidate_x, int candidate_y);

/* Run NPC AI for one tick: movement + attack decision */
void fc_npc_tick(FcState* state, int npc_idx);

/* Tz-Kek split-on-death: spawn 2 NPC_TZ_KEK_SM at death position */
void fc_npc_tz_kek_split(FcState* state, int dead_x, int dead_y);


/* Pathfinding */

/* Attack-route range 0 means cardinal melee contact, not ranged LOS. */
#define FC_ROUTE_MELEE_RANGE 0

/* ======================================================================== */
/* Tile queries                                                              */
/* ======================================================================== */

/* Check if a single tile is walkable (bounds + collision). */
int fc_tile_walkable(int x, int y,
                     const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* Check if an entity of given size can stand at (x,y).
 * All tiles in the [x..x+size-1, y..y+size-1] footprint must be walkable. */
int fc_footprint_walkable(int x, int y, int size,
                          const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* Check one sized step using the native client/RSMod leading-edge collision
 * masks, including the distinct size-1, size-2, and large-actor rules. */
int fc_footprint_step_walkable(
    int x, int y, int dx, int dy, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* ======================================================================== */
/* Dynamic occupancy                                                         */
/* ======================================================================== */

/* Clear an occupancy grid to all-free. */
void fc_clear_occupancy(uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* Mark all in-bounds tiles in a footprint occupied. Out-of-bounds tiles are
 * ignored here; availability checks still reject out-of-bounds footprints via
 * the static walkability check. */
void fc_mark_footprint_occupied(uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                                int x, int y, int size);

/* Build an occupancy grid from the current live entities. Pass ignore_npc_idx
 * to omit the moving NPC's own current footprint. Set ignore_player when
 * validating player movement or intentionally ignoring the player. */
void fc_build_occupancy(const FcState* state,
                        uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                        int ignore_npc_idx,
                        int ignore_player);

/* Check static terrain plus a caller-provided dynamic occupancy grid. */
int fc_footprint_available_dynamic(
    int x, int y, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* Dynamic equivalent of the native sized-step rule. Occupied tiles behave as
 * whole-tile blockers on the exact leading-edge cells checked by that rule. */
int fc_footprint_step_available_dynamic(
    int x, int y, int dx, int dy, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* ======================================================================== */
/* Movement                                                                  */
/* ======================================================================== */

/* Move a size-1 entity from (x,y) toward offset (dx,dy) for up to max_steps.
 * Diagonal-first fallback. Returns number of tiles moved. Updates *x,*y. */
int fc_move_toward(int* x, int* y, int dx, int dy, int max_steps,
                   const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                   const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* Same movement operation, additionally returning each successfully consumed
 * tile in step_x/step_y up to step_capacity entries. */
int fc_move_toward_traced(
    int* x, int* y, int dx, int dy, int max_steps,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int* step_x, int* step_y, int step_capacity);

/* Dynamic-aware sized step. Diagonal movement checks the final footprint and
 * both cardinal side footprints to prevent static or dynamic corner clipping. */
int fc_npc_step_toward_sized_dynamic(
    int* x, int* y, int target_x, int target_y, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* ======================================================================== */
/* Line of sight                                                             */
/* ======================================================================== */

/* Chebyshev distance between the nearest tiles of two rectangular areas. */
int fc_distance_between_areas(int src_x, int src_y, int src_size,
                              int dst_x, int dst_y, int dst_size);

/* Footprint-aware LOS using the closest coordinate from each rectangle, as in
 * the native line validator. */
int fc_has_los_between_areas(
    int src_x, int src_y, int src_size,
    int dst_x, int dst_y, int dst_size,
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* Rectangular-exclusive melee reach: only an open shared cardinal edge is
 * valid. Diagonal contact and overlapping footprints are rejected. */
int fc_npc_can_melee_player(int player_x, int player_y,
                            int npc_x, int npc_y, int npc_size,
                            const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                            const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);

/* ======================================================================== */
/* BFS pathfinding (for click-to-move)                                       */
/* ======================================================================== */

/* Native move-near fallback for human click-to-move. If the exact destination
 * is unreachable, chooses the reachable tile within ten tiles with the lowest
 * squared destination distance, breaking ties by route length. */
int fc_pathfind_bfs_move_near(
    int sx, int sy, int dx, int dy,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps);

/* Route to the first shortest-path tile outside a target rectangle that has
 * both the requested attack range and authoritative projectile LOS. */
int fc_pathfind_attack_position(
    int sx, int sy, int target_x, int target_y, int target_size,
    int attack_range,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps);


/* Prayer */

typedef struct {
    int prior_prayer;
    int requested_final_prayer;
    int actual_final_prayer;
    int off_requested;
    int off_performed;
    int on_requested;
    int on_succeeded;
    int explicit_off_then_on;
    int final_state_changed;
} FcPrayerTransition;

/* Drain prayer points based on active prayer and bonus.
 * prayer_active_at_tick_start should reflect the prayer state before the
 * current tick's input actions were applied, so 1-tick flicks do not drain. */
int fc_prayer_drain_tick(FcPlayer* p, int prayer_at_tick_start,
                         const FcPrayerTransition* transition);

/* Apply a prayer action (from FC_PRAYER_* constants in fc_contracts.h) */
FcPrayerTransition fc_prayer_apply_action(FcPlayer* p, int prayer_action);

/* Apply capped Prayer loss in tenths. Every Prayer-loss source uses this
 * boundary so depletion always deactivates the overhead and clears fraction. */
int fc_prayer_apply_loss_tenths(FcPlayer* p, int requested_loss_tenths);

/* Prayer potion restore amount in tenths (level-dependent) */
int fc_prayer_potion_restore(int prayer_level);


/* Reward */

typedef struct {
    float w_damage_dealt;      /* legacy per-hit shaping; active config uses 0 */
    float w_progress;          /* reward per required-work unit removed */
    float negative_progress_multiplier;
    float w_damage_taken;
    float w_npc_kill;
    float w_wave_clear;
    float w_jad_kill;
    float w_cave_complete;
    float w_player_death;
    int scale_player_death_with_progress;
    float player_death_min_scale;
    float w_correct_jad_prayer;
    float w_correct_danger_prayer;
    float w_prayer_lost;
    float w_invalid_action;
    float w_tick_penalty;

    float shape_unnecessary_prayer_penalty;
    float shape_wave_stall_base_penalty;
    float shape_wave_stall_cap;
    float shape_jad_heal_penalty;
    float shape_npc_heal_penalty;
    float shape_no_progress_penalty_1;
    float shape_no_progress_penalty_2;
    float shape_no_progress_penalty_3;
    float shape_no_attack_base_penalty;
    float shape_no_attack_wave_scale;

    int shape_wave_stall_start;
    int shape_wave_stall_ramp_interval;
    int shape_no_progress_start_1;
    int shape_no_progress_start_2;
    int shape_no_progress_start_3;
    int shape_no_attack_start;
} FcRewardParams;

typedef struct {
    int ticks_since_attack;
    int ticks_in_wave;
    float required_work_at_wave_start;
    float cave_progress_prev;
    float last_required_work_remaining;
    float last_current_wave_progress;
    float last_cave_progress;
    float last_progress_delta;
    float last_progress_reward;
    float last_net_required_work_removed;
    int ticks_since_positive_progress;
    int positive_progress_ticks;
    int zero_progress_ticks;
    int negative_progress_ticks;
} FcRewardRuntime;

typedef struct {
    int melee_pressure_npcs;
    int any_threat;
    int tokxil_melee;
    int ketzek_melee;
} FcRewardThreatContext;

typedef struct {
    float raw[FC_REWARD_FEATURES];

    float damage_dealt;
    float progress;
    float damage_taken;
    float npc_kill;
    float wave_clear;
    float jad_kill;
    float cave_complete;
    float player_death;

    float correct_jad_prayer;
    float correct_danger_prayer;
    float prayer_lost;
    float unnecessary_prayer;
    float wave_stall;
    float no_progress;
    float no_attack;
    float jad_heal;
    float npc_heal;

    float invalid_action;
    float tick_penalty;

    float total;
    FcRewardThreatContext threat_ctx;
} FcRewardBreakdown;

/* One slot per named breakdown field, excluding raw inputs and the total. */
typedef enum {
    FC_CH_DAMAGE_DEALT = 0,
    FC_CH_PROGRESS,
    FC_CH_DAMAGE_TAKEN,
    FC_CH_NPC_KILL,
    FC_CH_WAVE_CLEAR,
    FC_CH_JAD_KILL,
    FC_CH_CAVE_COMPLETE,
    FC_CH_PLAYER_DEATH,
    FC_CH_CORRECT_JAD_PRAYER,
    FC_CH_CORRECT_DANGER_PRAYER,
    FC_CH_PRAYER_LOST,
    FC_CH_UNNECESSARY_PRAYER,
    FC_CH_WAVE_STALL,
    FC_CH_NO_PROGRESS,
    FC_CH_NO_ATTACK,
    FC_CH_JAD_HEAL,
    FC_CH_NPC_HEAL,
    FC_CH_INVALID_ACTION,
    FC_CH_TICK_PENALTY,
    FC_CH_COUNT
} FcRwdChannel;

extern const char* const FC_CH_NAMES[FC_CH_COUNT];

void fc_reward_breakdown_channels(const FcRewardBreakdown* breakdown,
                                  float out[FC_CH_COUNT]);
FcRewardParams fc_reward_default_params(void);
void fc_reward_runtime_reset(FcRewardRuntime* runtime);
float fc_reward_player_death_scale(const FcRewardParams* params,
                                   float cave_progress);
float fc_reward_required_work_remaining(const FcState* state);
void fc_reward_sync_progress_state(FcState* state,
                                   const FcRewardRuntime* runtime);
void fc_reward_runtime_begin_episode(FcRewardRuntime* runtime, FcState* state);
FcRewardBreakdown fc_reward_compute_breakdown(
    const FcState* state, const FcRewardParams* params, FcRewardRuntime* runtime);


/* Wave */

/*
 * fc_wave.h — Wave system for Fight Caves.
 *
 * 63 waves, 15 spawn rotations per wave.
 * Wave table sourced from OSRS wiki + Kotlin archive TOML data.
 *
 * Spawn directions map to arena coordinates:
 *   SPAWN_SOUTH      → (32, 5)   bottom center
 *   SPAWN_SOUTH_WEST → (8, 8)    bottom left
 *   SPAWN_NORTH_WEST → (8, 55)   top left
 *   SPAWN_SOUTH_EAST → (55, 8)   bottom right
 *   SPAWN_CENTER     → (32, 32)  arena center
 */

/* Spawn direction → arena coordinate mapping */
void fc_spawn_position(int spawn_dir, int* x, int* y);

/* Spawn all NPCs for the given wave using the state's rotation_id */
void fc_wave_spawn(FcState* state, int wave_num);

/* Check if wave is cleared and advance to next wave.
 * Called from check_terminal in fc_tick.c.
 * Returns 1 if wave was advanced, 0 otherwise. */
int fc_wave_check_advance(FcState* state);

/* Jad healer spawn threshold: spawn 4 Yt-HurKot when Jad HP drops below this.
 * Respawns are re-armed only if the previous healers restored Jad to full HP. */
#define FC_JAD_HEALER_THRESHOLD_HP_TENTHS  1500  /* 150 HP */
#define FC_JAD_NUM_HEALERS            4


/* Api */

/*
 * fc_api.h — Public API for the Fight Caves simulation.
 *
 * Follows the encounter vtable pattern from PufferLib OSRS PvP/Zulrah:
 *   init → reset(seed) → [step(actions) → write_obs → ...]* → destroy
 *
 * All functions operate on FcState. No global/static state.
 */

/* ======================================================================== */
/* Lifecycle                                                                 */
/* ======================================================================== */

/* Initialize state to a clean starting configuration. Zeroes all fields.
 * Must be called once before the first reset. */
void fc_init(FcState* state);

/* Reset the episode with the given seed. Seeds the RNG, resets player/NPCs,
 * selects a random wave rotation, sets up the arena.
 * After reset, state is at tick 0 with wave 1 ready. */
void fc_reset(FcState* state, uint32_t seed);

/* Advance the simulation by one tick with the given actions.
 * actions[0..FC_NUM_ACTION_HEADS-1] are the canonical core action head values.
 * After step, per-tick event flags and terminal status are set. */
void fc_step(FcState* state, const int actions[FC_NUM_ACTION_HEADS]);

/* Canonical client preference request used by playable frontends. Gameplay
 * movement still occurs only through fc_step(). */
void fc_request_set_running(FcState* state, int enabled);

/* Internal tick loop (called by fc_step). Exposed for testing. */
void fc_tick(FcState* state, const int actions[FC_NUM_ACTION_HEADS]);

/* Free any resources (currently none — FcState is stack/caller-allocated).
 * Called for API symmetry and future-proofing. */
void fc_destroy(FcState* state);

/* ======================================================================== */
/* Observation / Mask / Reward                                               */
/* ======================================================================== */

/* Write the observation buffer (policy obs + reward features).
 * out must have room for FC_TOTAL_OBS floats.
 * Values are normalized to [0,1] via divisor tables. */
void fc_write_obs(const FcState* state, float* out);

/* Zero specific observation slots in-place (for obs-ablation experiments).
 * Apply AFTER fc_write_obs. Each flag, when non-zero, zeroes the
 * corresponding region of the policy obs (does not touch reward features
 * or the action mask):
 *   ablate_npc_distance         — zeroes FC_NPC_DISTANCE for all 8 NPC slots
 *   ablate_incoming_aggregates  — zeroes the 6 player-block IN_*_1T/2T fields
 *                                  + the 3 meta-block IN_*_3T fields
 *   ablate_npc_valid            — zeroes FC_NPC_VALID for all 8 NPC slots
 *
 * out must point to the same buffer fc_write_obs filled (length FC_TOTAL_OBS).
 * No-op when all three flags are 0. */
void fc_apply_obs_ablation(float* out,
                           int ablate_npc_distance,
                           int ablate_incoming_aggregates,
                           int ablate_npc_valid);

/* Fill out_indices with the active NPC array indices currently visible in
 * policy NPC slots, using the same ordering as fc_write_obs and fc_write_mask.
 * Returns the number of filled slots, capped at FC_VISIBLE_NPCS. */
int fc_visible_npc_indices(const FcState* state, int out_indices[FC_VISIBLE_NPCS]);

/* Write the action mask buffer.
 * out must have room for FC_ACTION_MASK_SIZE floats.
 * 1.0 = valid action, 0.0 = invalid. */
void fc_write_mask(const FcState* state, float* out);

/* Fill out_classes with 0/1 invalid-action diagnostics for Puffer-facing heads
 * 0-2 only: move, attack, prayer. Core consumable/path-target heads stay
 * excluded because the no-supplies policy does not emit them. */
void fc_action_invalid_classes(const FcState* state,
                               const int actions[FC_NUM_ACTION_HEADS],
                               int out_classes[FC_INVALID_ACTION_CLASS_COUNT]);

/* Compute and write reward features for the current tick.
 * out must have room for FC_REWARD_FEATURES floats.
 * These are raw feature values (not weighted). Python applies shaping weights. */
void fc_write_reward_features(const FcState* state, float* out);

/* Returns 1 if the episode has terminated (player death, cave complete, tick cap). */
int fc_is_terminal(const FcState* state);

/* ======================================================================== */
/* Determinism                                                               */
/* ======================================================================== */

/* Version 5 includes inventory, equipment, selected consumable slots and
 * unarmed bonuses. Policy observations and action dimensions are unchanged. */
#define FC_STATE_HASH_VERSION 5u

/*
 * Compute a deterministic hash of the game state.
 *
 * Implementation: FNV-1a hash over explicit field values written to a
 * canonical byte sequence. Does NOT hash raw struct padding bytes.
 * This guarantees that two states with identical logical content produce
 * identical hashes regardless of compiler padding or uninitialized bytes.
 *
 * Usage: Call after each fc_step to verify determinism. Two runs with
 * the same (seed, action_sequence) must produce identical hashes at every tick.
 */
uint32_t fc_state_hash(const FcState* state);

/* ======================================================================== */
/* Rendering                                                                 */
/* ======================================================================== */

/*
 * Fill an array of render entities for the viewer.
 * The viewer calls this each tick to get a snapshot of all visible entities.
 * entities[] must have room for FC_MAX_RENDER_ENTITIES entries.
 * *count is set to the number of filled entries.
 *
 * Entity 0 is always the player. Entities 1..count-1 are active NPCs.
 */
void fc_fill_render_entities(const FcState* state, FcRenderEntity* entities, int* count);

/* Copy the authoritative transition facts captured during the last tick. */
void fc_fill_render_events(const FcState* state, FcRenderEvents* events);

/* ======================================================================== */
/* RNG (exposed for testing; normal callers use fc_reset to seed)             */
/* ======================================================================== */

/* Seed the RNG state. Called internally by fc_reset. */
void fc_rng_seed(FcState* state, uint32_t seed);

/* Generate a random uint32. Advances the RNG state. */
uint32_t fc_rng_next(FcState* state);

/* Generate a random int in [0, max) */
int fc_rng_int(FcState* state, int max);

/* Generate a random float in [0.0, 1.0) */
float fc_rng_float(FcState* state);


/* Items */

typedef struct {
    int id;
    const char *name;
    int slot;                    /* -1 for inventory-only items */
    int stackable, two_handed;
    int ranged_level, defence_level, hitpoints_level;
    int ranged_attack, ranged_strength;
    int defence[5];               /* stab, slash, crush, magic, ranged */
    int prayer, melee_attack, melee_strength;
    int weapon_kind, speed, range, ammo_kind;
    int ammo_tier;                /* supported ammunition: 0 standard, 1 dragon */
    int crystal_piece;
    int visual_profile;
} FcItemDef;

typedef enum {
    FC_ITEM_OK, FC_ITEM_INVALID, FC_ITEM_NO_SPACE, FC_ITEM_REQUIREMENTS,
    FC_ITEM_BUSY
} FcItemResult;

const FcItemDef *fc_item_definition(int item_id);
const char *fc_item_result_message(FcItemResult result);
/* Immediate inventory transactions between ticks. They do not advance time,
 * reset cooldowns, roll RNG, or alter already-launched attacks. */
FcItemResult fc_equip_item(FcState *state, int inventory_slot);
FcItemResult fc_unequip_item(FcState *state, int equipment_slot);
FcItemResult fc_inventory_swap(FcState *state, int first, int second);
/* Select the actual slot consumed by the next canonical food/potion action. */
FcItemResult fc_select_consumable(FcState *state, int inventory_slot);
void fc_set_initial_supplies(FcState *state, int sharks, int prayer_doses);


/* Action Internal */

/* An already-active run may consume its remaining energy below 1%. Starting
 * run mode follows the OSRS client/server minimum of one displayed percent. */
static inline int fc_player_can_run(const FcPlayer* player) {
    return player->run_energy > 0 &&
           (player->is_running ||
            player->run_energy >= FC_RUN_ENERGY_MIN_START);
}

int fc_eat_action_valid(const FcState* state, int action);
int fc_drink_action_valid(const FcState* state, int action);


/* Items Internal */
void fc_items_init(FcPlayer *player, const FcLoadout *loadout);
void fc_items_consume(FcPlayer *player, int potion);
void fc_items_spend_ammo(FcPlayer *player);

/* Spawn Internal */

int fc_spawn_find_available_footprint(const FcState* state,
                                      int preferred_x, int preferred_y,
                                      int size, int max_radius,
                                      int* out_x, int* out_y);

int fc_spawn_npc_first_free(FcState* state, int npc_type, int x, int y);


/* Wave Internal */

void fc_wave_record_current_duration(FcState* state);


/* Combat */
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

/*
 * fc_combat.c — OSRS combat math and pending hit resolution.
 *
 * Formulas adapted from osrs_combat_shared.h (PufferLib).
 *
 * PvM prayer semantics:
 *   Correct protection prayer BLOCKS 100% of the matching NPC attack style.
 *   This is standard OSRS PvM — NOT the PvP 60% reduction.
 *   Exceptions must be explicit per NPC/attack (e.g. Jad wrong-prayer still takes damage).
 */

/* ======================================================================== */
/* OSRS accuracy formula                                                     */
/* ======================================================================== */

float fc_hit_chance(int att_roll, int def_roll) {
    if (att_roll > def_roll)
        return 1.0f - (float)(def_roll + 2) / (2.0f * (float)(att_roll + 1));
    else
        return (float)att_roll / (2.0f * (float)(def_roll + 1));
}

/* ======================================================================== */
/* NPC attack/max-hit formulas                                               */
/* ======================================================================== */

int fc_npc_attack_roll(int att_level, int att_bonus) {
    /* NPCs use level + invisible_boost(9) × (bonus + 64) */
    return (att_level + 9) * (att_bonus + 64);
}

/* ======================================================================== */
/* Player defence roll                                                       */
/* ======================================================================== */

int fc_player_def_roll(const FcPlayer* p, FcAttackType attack_type) {
    int def_bonus;
    switch (attack_type) {
        case FC_ATTACK_TYPE_STAB:   def_bonus = p->defence_stab; break;
        case FC_ATTACK_TYPE_SLASH:  def_bonus = p->defence_slash; break;
        case FC_ATTACK_TYPE_CRUSH:  def_bonus = p->defence_crush; break;
        case FC_ATTACK_TYPE_RANGED: def_bonus = p->defence_ranged; break;
        case FC_ATTACK_TYPE_MAGIC:  def_bonus = p->defence_magic; break;
        default:                    def_bonus = 0; break;
    }

    int eff_def;
    if (attack_type == FC_ATTACK_TYPE_MAGIC) {
        /* OSRS truncates the Defence and Magic contributions separately. */
        eff_def = 3 * p->defence_level / 10 +
                  7 * p->magic_level / 10 + 8;
    } else {
        eff_def = p->defence_level + 8;
    }
    return eff_def * (def_bonus + 64);
}

/* ======================================================================== */
/* Player ranged attack / max-hit                                            */
/* ======================================================================== */

static int fc_player_effective_ranged_level(const FcPlayer* p) {
    /* Rapid is the active DPS style for both RCB and TBow in this sim. */
    return p->ranged_level + 8;
}

int fc_player_ranged_base_attack_roll(const FcPlayer* p) {
    int eff_ranged = fc_player_effective_ranged_level(p);
    return eff_ranged * (p->ranged_attack_bonus + 64);
}

static int fc_tbow_target_magic_level(const FcNpc* target) {
    const FcNpcStats* stats = fc_npc_get_stats(target->npc_type);
    int magic_level = stats->magic_level;

    if (magic_level < 0) magic_level = 0;
    if (magic_level > 250) magic_level = 250;  /* non-CoX cap */
    return magic_level;
}

int fc_tbow_accuracy_multiplier_pct(int target_magic_level) {
    int64_t magic = target_magic_level;
    if (magic < 0) magic = 0;
    if (magic > 250) magic = 250;

    int64_t inner = 3 * magic / 10;
    int64_t delta = inner - 100;
    int64_t pct = 140 + (3 * magic - 10) / 100 -
                  delta * delta / 100;
    if (pct < 0) pct = 0;
    if (pct > 140) pct = 140;
    return (int)pct;
}

int fc_tbow_damage_multiplier_pct(int target_magic_level) {
    int64_t magic = target_magic_level;
    if (magic < 0) magic = 0;
    if (magic > 250) magic = 250;

    int64_t inner = 3 * magic / 10;
    int64_t delta = inner - 140;
    int64_t pct = 250 + (3 * magic - 14) / 100 -
                  delta * delta / 100;
    if (pct < 0) pct = 0;
    if (pct > 250) pct = 250;
    return (int)pct;
}

static void fc_crystal_modifiers_bp(int crystal_piece_mask,
                                    int* accuracy_bp, int* damage_bp) {
    int mask = crystal_piece_mask & FC_CRYSTAL_PIECE_ALL;
    *accuracy_bp = 0;
    *damage_bp = 0;

    if (mask & FC_CRYSTAL_PIECE_HELM) {
        *accuracy_bp += FC_CRYSTAL_HELM_ACCURACY_BP;
        *damage_bp += FC_CRYSTAL_HELM_DAMAGE_BP;
    }
    if (mask & FC_CRYSTAL_PIECE_BODY) {
        *accuracy_bp += FC_CRYSTAL_BODY_ACCURACY_BP;
        *damage_bp += FC_CRYSTAL_BODY_DAMAGE_BP;
    }
    if (mask & FC_CRYSTAL_PIECE_LEGS) {
        *accuracy_bp += FC_CRYSTAL_LEGS_ACCURACY_BP;
        *damage_bp += FC_CRYSTAL_LEGS_DAMAGE_BP;
    }
}

static int fc_apply_basis_points(int value, int bonus_bp) {
    return (int)((int64_t)value * (10000 + bonus_bp) / 10000);
}

int fc_player_ranged_attack_roll(const FcPlayer* p, const FcNpc* target) {
    int attack_roll = fc_player_ranged_base_attack_roll(p);

    if (p->weapon_kind == FC_WEAPON_TWISTED_BOW && target) {
        attack_roll = (int)((int64_t)attack_roll *
            fc_tbow_accuracy_multiplier_pct(fc_tbow_target_magic_level(target)) /
            100);
    } else if (p->weapon_kind == FC_WEAPON_BOW_OF_FAERDHINEN) {
        int accuracy_bp;
        int damage_bp;
        fc_crystal_modifiers_bp(p->crystal_piece_mask,
                                &accuracy_bp, &damage_bp);
        (void)damage_bp;
        attack_roll = fc_apply_basis_points(attack_roll, accuracy_bp);
    }

    return attack_roll;
}

int fc_player_ranged_base_max_hit_hp(const FcPlayer* p) {
    int eff_str = fc_player_effective_ranged_level(p);
    return (int)(((int64_t)eff_str * (p->ranged_strength_bonus + 64) + 320) /
                 640);
}

int fc_player_ranged_final_max_hit_hp(const FcPlayer* p, const FcNpc* target) {
    int base_hp = fc_player_ranged_base_max_hit_hp(p);

    if (p->weapon_kind == FC_WEAPON_TWISTED_BOW && target) {
        base_hp = (int)((int64_t)base_hp *
            fc_tbow_damage_multiplier_pct(fc_tbow_target_magic_level(target)) /
            100);
    } else if (p->weapon_kind == FC_WEAPON_BOW_OF_FAERDHINEN) {
        int accuracy_bp;
        int damage_bp;
        fc_crystal_modifiers_bp(p->crystal_piece_mask,
                                &accuracy_bp, &damage_bp);
        (void)accuracy_bp;
        base_hp = fc_apply_basis_points(base_hp, damage_bp);
    }

    return base_hp;
}

static int fc_damage_max_valid(const FcState* state, int final_max_hit_hp) {
    return state != NULL && final_max_hit_hp >= 0 &&
           final_max_hit_hp <= INT_MAX / 10;
}

int fc_roll_player_damage_tenths(FcState* state, int final_max_hit_hp) {
    if (!fc_damage_max_valid(state, final_max_hit_hp) ||
        final_max_hit_hp == 0) {
        return 0;
    }
    int rolled_hp = fc_rng_int(state, final_max_hit_hp + 1);
    if (rolled_hp == 0) rolled_hp = 1;
    return rolled_hp * 10;
}

int fc_roll_npc_damage_tenths(FcState* state, int final_max_hit_hp) {
    if (!fc_damage_max_valid(state, final_max_hit_hp) ||
        final_max_hit_hp == 0) {
        return 0;
    }
    return fc_rng_int(state, final_max_hit_hp + 1) * 10;
}

/* ======================================================================== */
/* Prayer check                                                              */
/* ======================================================================== */

int fc_prayer_blocks_style(int prayer, int attack_style) {
    /*
     * PvM: correct protection prayer blocks 100% of matching style.
     * Our enum mapping:
     *   PRAYER_PROTECT_MELEE(1) blocks ATTACK_MELEE(1)
     *   PRAYER_PROTECT_RANGE(2) blocks ATTACK_RANGED(2)
     *   PRAYER_PROTECT_MAGIC(3) blocks ATTACK_MAGIC(3)
     */
    if (prayer == PRAYER_NONE || attack_style == ATTACK_NONE) return 0;
    return (prayer == attack_style) ? 1 : 0;
}

/* ======================================================================== */
/* Chebyshev distance to multi-tile NPC                                      */
/* ======================================================================== */

int fc_distance_to_npc(int px, int py, const FcNpc* npc) {
    return fc_distance_between_areas(px, py, 1,
                                     npc->x, npc->y, npc->size);
}

/* ======================================================================== */
/* Hit delay formulas                                                        */
/* ======================================================================== */

/*
 * OSRS projectile hit delay:
 *   travel_time = time_offset + (distance * multiplier)  [in client ticks, 20ms each]
 *   game_ticks = travel_time / 30 + 1                    [CLIENT_TICKS.toTicks() = n/30]
 *
 * Per-NPC projectile definitions from tzhaar_fight_cave.gfx.toml:
 *   tok_xil_shoot:   delay=32, height=256, curve=16, no offset/mult → default mult=5
 *   ket_zek_travel:  delay=28, height=128, curve=16, offset=8, mult=8
 *   tztok_jad_travel: delay=86, height=50, curve=16, mult=8, no offset
 *   Jad ranged: no projectile, fixed client delay=120
 *
 * Melee: delay 1 (resolves same tick in our system — queued then resolved in same tick loop)
 */

/* Player ranged projectile timing */
int fc_ranged_hit_delay(int distance) {
    /* Keep the existing lightweight projectile timing for player ranged attacks. */
    int travel = 5 * distance;  /* default multiplier for player ranged */
    return travel / 30 + 1;
}

/*
 * Per-NPC-type hit delay — uses exact projectile timing from Void 634 gfx.toml.
 * Called from NPC attack code for precise parity with RSPS.
 */
int fc_npc_hit_delay(int npc_type, int attack_style, int distance) {
    if (attack_style == ATTACK_MELEE) return 1;

    switch (npc_type) {
        case NPC_TOK_XIL:
            /* tok_xil_shoot: default mult=5, no offset */
            return (5 * distance) / 30 + 1;

        case NPC_KET_ZEK:
            /* ket_zek_travel: offset=8, mult=8 */
            return (8 + 8 * distance) / 30 + 1;

        case NPC_TZTOK_JAD:
            if (attack_style == ATTACK_MAGIC) {
                /* Keep at least one full policy decision tick between Jad's
                 * tell and impact, including when Magic is selected in melee. */
                int delay = (8 * distance) / 30 + 1;
                return delay < 2 ? 2 : delay;
            } else {
                /* Jad ranged: no projectile, fixed client delay=120 */
                return 120 / 30 + 1;  /* = 5 game ticks */
            }

        default:
            /* Fallback for any other ranged/magic NPC */
            if (attack_style == ATTACK_RANGED) return (5 * distance) / 30 + 1;
            return (8 + 8 * distance) / 30 + 1;
    }
}

/* ======================================================================== */
/* NPC defence roll (for player attack accuracy against NPC)                 */
/* ======================================================================== */

int fc_npc_def_roll(int def_level, int def_bonus) {
    /* NPC defence: (def_level + 9) × (def_bonus + 64) */
    return (def_level + 9) * (def_bonus + 64);
}

/* ======================================================================== */
/* Queue a pending hit                                                       */
/* ======================================================================== */

int fc_queue_pending_hit(FcPendingHit hits[], int* num_hits, int max_hits,
                         int damage, int ticks, int style, int source_idx,
                         int prayer_drain) {
    if (*num_hits >= max_hits) return 0;
    FcPendingHit* h = &hits[*num_hits];
    h->active = 1;
    h->damage = damage;
    h->ticks_remaining = ticks;
    h->attack_style = style;
    h->source_npc_idx = source_idx;
    h->prayer_drain = prayer_drain;
    h->prayer_snapshot = PRAYER_NONE;
    h->prayer_lock_tick = -1;
    (*num_hits)++;
    return 1;
}

/* ======================================================================== */
/* Resolve pending hits (called each tick)                                   */
/* ======================================================================== */

static void record_render_hit(FcState* state, int target_entity_type,
                              int target_npc_slot, int source_npc_slot,
                              int attack_style, int damage, int blocked) {
    FcRenderEvents* events = &state->render_events;
    if (events->hit_count >= FC_MAX_RENDER_HITS) return;

    FcRenderHit* hit = &events->hits[events->hit_count++];
    hit->target_entity_type = target_entity_type;
    hit->target_npc_slot = target_npc_slot;
    hit->source_npc_slot = source_npc_slot;
    hit->attack_style = attack_style;
    hit->damage = damage;
    hit->blocked = blocked;
}

void fc_resolve_player_pending_hits(FcState* state) {
    FcPlayer* p = &state->player;
    int write = 0;

    for (int i = 0; i < p->num_pending_hits; i++) {
        FcPendingHit* h = &p->pending_hits[i];
        if (!h->active) continue;

        h->ticks_remaining--;
        if (h->ticks_remaining <= 0) {
            /* Hit resolves now — use the prayer locked into this hit. */
            int locked_prayer = h->prayer_snapshot >= 0
                ? h->prayer_snapshot : PRAYER_NONE;
            int blocked = fc_prayer_blocks_style(locked_prayer, h->attack_style);
            int final_damage = blocked ? 0 : h->damage;

            /* Apply damage */
            p->current_hp -= final_damage;
            if (p->current_hp < 0) p->current_hp = 0;

            p->damage_taken_this_tick += final_damage;
            p->hit_style_this_tick = h->attack_style;
            p->hit_source_npc_type = state->npcs[h->source_npc_idx].npc_type;
            p->hit_locked_prayer_this_tick = locked_prayer;
            p->hit_blocked_this_tick = blocked;
            state->damage_taken_this_tick += final_damage;
            p->total_damage_taken += final_damage;
            p->hit_landed_this_tick = 1;
            record_render_hit(state, ENTITY_PLAYER, -1,
                              h->source_npc_idx, h->attack_style,
                              final_damage, blocked);

            /* Auto-retaliate: if player has no target, target the attacker.
             * approach_target stays 0 — player attacks in place, doesn't chase. */
            if (p->attack_target_idx < 0 && h->source_npc_idx >= 0) {
                FcNpc* attacker = &state->npcs[h->source_npc_idx];
                if (attacker->active && !attacker->is_dead) {
                    p->attack_target_idx = h->source_npc_idx;
                    p->approach_target = 0;  /* don't chase, attack from here */
                    p->approach_target_x = -1;
                    p->approach_target_y = -1;
                    p->approach_target_size = 0;
                }
            }

            /* Tz-Kih drains damage dealt + 1 Prayer point, with the base point
             * still applying to misses and prayer-blocked attacks. HP and Prayer
             * both use tenths internally, so final_damage can be added directly. */
            if (h->prayer_drain > 0) {
                int drain = h->prayer_drain;
                if (h->source_npc_idx >= 0 && h->source_npc_idx < FC_MAX_NPCS &&
                    state->npcs[h->source_npc_idx].npc_type == NPC_TZ_KIH) {
                    drain += final_damage;
                }
                int actual_drain = fc_prayer_apply_loss_tenths(p, drain);
                state->prayer_lost_this_tick += actual_drain;
                state->tz_kih_prayer_drain_this_tick += actual_drain;
                if (h->source_npc_idx >= 0 && h->source_npc_idx < FC_MAX_NPCS) {
                    state->npcs[h->source_npc_idx].prayer_drain_dealt_this_tick +=
                        actual_drain;
                }
            }

            /* Track prayer correctness. Correctly blocked Jad hits also use
             * the shared correct-prayer reward applied to every NPC. */
            if (state->npcs[h->source_npc_idx].npc_type == NPC_TZTOK_JAD) {
                if (blocked) {
                    state->correct_jad_prayer = 1;
                    state->correct_danger_prayer = 1;
                } else {
                    state->wrong_jad_prayer = 1;
                }
            } else if (h->attack_style == ATTACK_RANGED ||
                       h->attack_style == ATTACK_MAGIC ||
                       h->attack_style == ATTACK_MELEE) {
                if (blocked) state->correct_danger_prayer = 1;
                else state->wrong_danger_prayer = 1;
            }

            /* Episode-level hit analytics */
            if (locked_prayer != PRAYER_NONE) {
                if (blocked) {
                    state->ep_correct_blocks++;
                    state->ep_damage_blocked += h->damage;
                } else {
                    state->ep_wrong_prayer_hits++;
                }
            } else {
                state->ep_no_prayer_hits++;
            }

            h->active = 0;  /* consumed */
        } else {
            /* Still in flight — keep */
            if (write != i) p->pending_hits[write] = *h;
            write++;
        }
    }
    p->num_pending_hits = write;
}

static void complete_fight_caves(FcState* state) {
    state->jad_killed = 1;
    state->ep_jad_killed = 1;

    /* Jad death completes the cave and immediately despawns surviving
     * healers, matching the encounter lifecycle. */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* other = &state->npcs[i];
        if (other->active && !other->is_dead &&
            other->npc_type == NPC_YT_HURKOT) {
            other->is_dead = 1;
            other->died_this_tick = 1;
            other->death_timer = 0;
            state->npcs_remaining--;
        }
    }
    state->wave_just_cleared = 1;
    state->terminal = TERMINAL_CAVE_COMPLETE;
    fc_wave_record_current_duration(state);
}

static void resolve_npc_death(FcState* state, FcNpc* npc) {
    npc->is_dead = 1;
    npc->died_this_tick = 1;
    npc->death_timer = 3;

    /* Each entity is one kill. A Tz-Kek parent is counted here before its two
     * children are counted through this same path later. */
    state->npcs_killed_this_tick++;
    if (npc->npc_type == NPC_YT_HURKOT &&
        npc->is_respawned_jad_healer) {
        state->respawned_jad_healers_killed_this_tick++;
    }
    state->total_npcs_killed++;

    if (npc->npc_type == NPC_TZTOK_JAD) {
        complete_fight_caves(state);
    }

    /* The Tz-Kek parent was pre-counted as two at spawn. Only its split
     * children decrement the wave's remaining-work count when they die. */
    if (npc->npc_type == NPC_TZ_KEK) {
        fc_npc_tz_kek_split(state, npc->x, npc->y);
    } else {
        state->npcs_remaining--;
    }
}

void fc_resolve_npc_pending_hits(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    int write = 0;

    for (int i = 0; i < npc->num_pending_hits; i++) {
        FcPendingHit* h = &npc->pending_hits[i];
        if (!h->active) continue;

        h->ticks_remaining--;
        if (h->ticks_remaining <= 0) {
            /* Player's hit lands on NPC */
            npc->current_hp -= h->damage;
            if (npc->current_hp < 0) npc->current_hp = 0;

            npc->damage_taken_this_tick += h->damage;
            state->damage_dealt_this_tick += h->damage;
            if (npc->npc_type > NPC_NONE && npc->npc_type < NPC_TYPE_COUNT) {
                state->ep_resolved_hits_to_npc_type[npc->npc_type]++;
                state->ep_damage_to_npc_type[npc->npc_type] += h->damage;
                if (h->damage > 0) {
                    state->ep_damaging_hits_to_npc_type[npc->npc_type]++;
                }
            }
            if (h->damage > 0) {
                state->hits_landed_this_tick++;
            }
            record_render_hit(state, ENTITY_NPC, npc_idx, -1,
                              h->attack_style, h->damage, 0);

            /* Track Jad-specific damage */
            if (npc->npc_type == NPC_TZTOK_JAD) {
                state->jad_damage_this_tick += h->damage;
            }

            /* Yt-HurKot: any landed player attack distracts healer, including 0s. */
            if (npc->npc_type == NPC_YT_HURKOT) {
                npc->healer_distracted = 1;
                npc->heal_target_idx = -1;
            }

            /* NPC death — keep active for a few ticks so viewer can
             * show the killing hitsplat and death animation. */
            if (npc->current_hp <= 0 && !npc->is_dead) {
                resolve_npc_death(state, npc);
            }

            h->active = 0;
        } else {
            if (write != i) npc->pending_hits[write] = *h;
            write++;
        }
    }
    npc->num_pending_hits = write;
}


/* Episode Summary */
#include <string.h>

void fc_episode_summary_build(const FcState* state, int episode_length,
                              FcEpisodeSummary* summary) {
    if (!summary) return;
    memset(summary, 0, sizeof(*summary));
    if (!state) return;

    summary->episode_length = episode_length;
    summary->wave_reached = state->current_wave;
    summary->npcs_slayed = state->total_npcs_killed;
    if (episode_length > 0) {
        summary->prayer_uptime_melee =
            (float)state->ep_ticks_pray_melee / (float)episode_length;
        summary->prayer_uptime_range =
            (float)state->ep_ticks_pray_range / (float)episode_length;
        summary->prayer_uptime_magic =
            (float)state->ep_ticks_pray_magic / (float)episode_length;
    }
    summary->correct_prayer = state->ep_correct_blocks;
    summary->wrong_prayer_hits = state->ep_wrong_prayer_hits;
    summary->no_prayer_hits = state->ep_no_prayer_hits;
    summary->prayer_switches = state->ep_prayer_switches;
    summary->damage_blocked = state->ep_damage_blocked;
    summary->damage_taken = state->player.total_damage_taken;
    if (state->ep_attack_ready_ticks > 0) {
        summary->attack_when_ready_rate =
            (float)state->ep_attack_attempt_ticks /
            (float)state->ep_attack_ready_ticks;
    }
    summary->invalid_move =
        state->ep_invalid_action_classes[FC_INVALID_ACTION_MOVE];
    summary->invalid_attack =
        state->ep_invalid_action_classes[FC_INVALID_ACTION_ATTACK];
    summary->invalid_prayer =
        state->ep_invalid_action_classes[FC_INVALID_ACTION_PRAYER];
    summary->tokxil_melee_ticks = state->ep_tokxil_melee_ticks;
    summary->ketzek_melee_ticks = state->ep_ketzek_melee_ticks;
    summary->max_wave_ticks = state->ep_max_wave_ticks;
    summary->max_wave_ticks_wave = state->ep_max_wave_ticks_wave;
    summary->reached_wave_63 = state->ep_reached_wave_63;
    summary->jad_killed = state->ep_jad_killed;
    summary->player_died = state->terminal == TERMINAL_PLAYER_DEATH;

    memcpy(summary->damage_to_npc_type, state->ep_damage_to_npc_type,
           sizeof(summary->damage_to_npc_type));
    memcpy(summary->resolved_hits_to_npc_type,
           state->ep_resolved_hits_to_npc_type,
           sizeof(summary->resolved_hits_to_npc_type));
    memcpy(summary->damaging_hits_to_npc_type,
           state->ep_damaging_hits_to_npc_type,
           sizeof(summary->damaging_hits_to_npc_type));
    memcpy(summary->attack_cycles_to_npc_type,
           state->ep_attack_cycles_to_npc_type,
           sizeof(summary->attack_cycles_to_npc_type));
    memcpy(summary->target_ticks_by_npc_type,
           state->ep_target_ticks_by_npc_type,
           sizeof(summary->target_ticks_by_npc_type));

    summary->target_held_ticks = state->ep_target_held_ticks;
    summary->no_target_ticks = state->ep_no_target_ticks;
    summary->target_in_range_los_ticks =
        state->ep_target_in_range_los_ticks;
    summary->target_out_of_range_or_los_ticks =
        state->ep_target_out_of_range_or_los_ticks;
    summary->attack_cooldown_wait_ticks =
        state->ep_attack_cooldown_wait_ticks;
    summary->ready_but_no_attack_ticks =
        state->ep_ready_but_no_attack_ticks;
    summary->action_move_idle_ticks = state->ep_action_move_idle_ticks;
    summary->action_move_walk_ticks = state->ep_action_move_walk_ticks;
    summary->action_move_run_ticks = state->ep_action_move_run_ticks;
    summary->action_attack_none_ticks = state->ep_action_attack_none_ticks;
    summary->action_attack_target_ticks =
        state->ep_action_attack_target_ticks;
    summary->action_prayer_noop_ticks = state->ep_action_prayer_noop_ticks;
    summary->action_prayer_cmd_ticks = state->ep_action_prayer_cmd_ticks;
}

const char* fc_episode_npc_metric_name(int npc_type) {
    switch (npc_type) {
        case NPC_NONE:      return "none";
        case NPC_TZ_KIH:    return "tz_kih";
        case NPC_TZ_KEK:    return "tz_kek";
        case NPC_TZ_KEK_SM: return "tz_kek_sm";
        case NPC_TOK_XIL:   return "tok_xil";
        case NPC_YT_MEJKOT: return "yt_mejkot";
        case NPC_KET_ZEK:   return "ket_zek";
        case NPC_TZTOK_JAD: return "tztok_jad";
        case NPC_YT_HURKOT: return "yt_hurkot";
        default:            return "unknown";
    }
}


/* Hash */
#include <stdint.h>
#include <string.h>

/*
 * Version 2 serializes every FcState field explicitly in the documented order
 * below, including whole-tile, directional movement, and projectile collision
 * maps. Signed integers and floats are represented by 32 bits, then fed
 * least-significant byte first. Arena bytes are fed directly. This order is
 * the canonical contract: never hash struct storage, padding, or pointers.
 */

#define FC_HASH_FNV_OFFSET UINT32_C(0x811c9dc5)
#define FC_HASH_FNV_PRIME UINT32_C(0x01000193)

_Static_assert(sizeof(int) == sizeof(int32_t),
               "canonical state hash requires 32-bit int");
_Static_assert(sizeof(float) == sizeof(uint32_t),
               "canonical state hash requires 32-bit float");

static uint32_t fc_hash_u8(uint32_t hash, uint8_t value) {
    return (hash ^ value) * FC_HASH_FNV_PRIME;
}

static uint32_t fc_hash_u32(uint32_t hash, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        hash = fc_hash_u8(hash, (uint8_t)(value >> shift));
    }
    return hash;
}

static uint32_t fc_hash_i32(uint32_t hash, int value) {
    return fc_hash_u32(hash, (uint32_t)(int32_t)value);
}

static uint32_t fc_hash_f32(uint32_t hash, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return fc_hash_u32(hash, bits);
}

#define FC_HASH_I32(value) hash = fc_hash_i32(hash, (value))
#define FC_HASH_U32(value) hash = fc_hash_u32(hash, (value))
#define FC_HASH_F32(value) hash = fc_hash_f32(hash, (value))

static uint32_t fc_hash_pending_hit(uint32_t hash, const FcPendingHit* hit) {
    FC_HASH_I32(hit->active);
    FC_HASH_I32(hit->damage);
    FC_HASH_I32(hit->ticks_remaining);
    FC_HASH_I32(hit->attack_style);
    FC_HASH_I32(hit->source_npc_idx);
    FC_HASH_I32(hit->prayer_drain);
    FC_HASH_I32(hit->prayer_snapshot);
    FC_HASH_I32(hit->prayer_lock_tick);
    return hash;
}

static uint32_t fc_hash_player(uint32_t hash, const FcPlayer* player) {
    FC_HASH_I32(player->x);
    FC_HASH_I32(player->y);
    FC_HASH_I32(player->current_hp);
    FC_HASH_I32(player->max_hp);
    FC_HASH_I32(player->current_prayer);
    FC_HASH_I32(player->max_prayer);
    FC_HASH_I32(player->prayer);
    FC_HASH_I32(player->prayer_at_tick_start);
    FC_HASH_I32(player->prayer_drain_counter);
    FC_HASH_I32(player->sharks_remaining);
    FC_HASH_I32(player->prayer_doses_remaining);
    FC_HASH_I32(player->attack_timer);
    FC_HASH_I32(player->food_timer);
    FC_HASH_I32(player->potion_timer);
    FC_HASH_I32(player->combo_timer);
    FC_HASH_I32(player->run_energy);
    FC_HASH_I32(player->is_running);
    FC_HASH_I32(player->attack_level);
    FC_HASH_I32(player->strength_level);
    FC_HASH_I32(player->defence_level);
    FC_HASH_I32(player->ranged_level);
    FC_HASH_I32(player->prayer_level);
    FC_HASH_I32(player->magic_level);
    FC_HASH_I32(player->weapon_kind);
    FC_HASH_I32(player->weapon_uses_ammo);
    FC_HASH_I32(player->crystal_piece_mask);
    FC_HASH_I32(player->weapon_speed);
    FC_HASH_I32(player->weapon_range);
    FC_HASH_I32(player->ranged_attack_bonus);
    FC_HASH_I32(player->ranged_strength_bonus);
    FC_HASH_I32(player->defence_stab);
    FC_HASH_I32(player->defence_slash);
    FC_HASH_I32(player->defence_crush);
    FC_HASH_I32(player->defence_magic);
    FC_HASH_I32(player->defence_ranged);
    FC_HASH_I32(player->prayer_bonus);
    FC_HASH_I32(player->ammo_count);
    FC_HASH_I32(player->hp_regen_counter);
    for (int i = 0; i < FC_MAX_ROUTE; ++i) {
        FC_HASH_I32(player->route_x[i]);
        FC_HASH_I32(player->route_y[i]);
    }
    FC_HASH_I32(player->route_len);
    FC_HASH_I32(player->route_idx);
    FC_HASH_F32(player->facing_angle);
    FC_HASH_I32(player->attack_target_idx);
    FC_HASH_I32(player->approach_target);
    FC_HASH_I32(player->approach_target_x);
    FC_HASH_I32(player->approach_target_y);
    FC_HASH_I32(player->approach_target_size);
    for (int i = 0; i < FC_MAX_PENDING_HITS; ++i) {
        hash = fc_hash_pending_hit(hash, &player->pending_hits[i]);
    }
    FC_HASH_I32(player->num_pending_hits);
    FC_HASH_I32(player->damage_taken_this_tick);
    FC_HASH_I32(player->hit_style_this_tick);
    FC_HASH_I32(player->hit_source_npc_type);
    FC_HASH_I32(player->hit_locked_prayer_this_tick);
    FC_HASH_I32(player->hit_blocked_this_tick);
    FC_HASH_I32(player->hit_landed_this_tick);
    FC_HASH_I32(player->food_eaten_this_tick);
    FC_HASH_I32(player->potion_used_this_tick);
    FC_HASH_I32(player->prayer_changed_this_tick);
    FC_HASH_I32(player->total_damage_taken);
    FC_HASH_I32(player->total_food_eaten);
    FC_HASH_I32(player->total_potions_used);
    for (int i = 0; i < FC_INVENTORY_SLOTS; i++) {
        FC_HASH_I32(player->inventory[i].item_id);
        FC_HASH_I32(player->inventory[i].quantity);
        FC_HASH_I32(player->inventory[i].charges);
    }
    for (int i = 0; i < FC_EQUIPMENT_SLOTS; i++) {
        FC_HASH_I32(player->equipment[i].item_id);
        FC_HASH_I32(player->equipment[i].quantity);
        FC_HASH_I32(player->equipment[i].charges);
    }
    FC_HASH_I32(player->melee_attack_bonus);
    FC_HASH_I32(player->melee_strength_bonus);
    FC_HASH_I32(player->selected_food_slot);
    FC_HASH_I32(player->selected_potion_slot);
    return hash;
}

static uint32_t fc_hash_npc(uint32_t hash, const FcNpc* npc) {
    FC_HASH_I32(npc->active);
    FC_HASH_I32(npc->npc_type);
    FC_HASH_I32(npc->spawn_index);
    FC_HASH_I32(npc->x);
    FC_HASH_I32(npc->y);
    FC_HASH_I32(npc->size);
    FC_HASH_I32(npc->current_hp);
    FC_HASH_I32(npc->max_hp);
    FC_HASH_I32(npc->is_dead);
    FC_HASH_I32(npc->death_timer);
    FC_HASH_I32(npc->attack_style);
    FC_HASH_I32(npc->attack_timer);
    FC_HASH_I32(npc->attack_speed);
    FC_HASH_I32(npc->attack_range);
    FC_HASH_I32(npc->movement_speed);
    FC_HASH_I32(npc->heal_timer);
    FC_HASH_I32(npc->heal_amount);
    FC_HASH_I32(npc->healer_distracted);
    FC_HASH_I32(npc->heal_target_idx);
    FC_HASH_I32(npc->is_respawned_jad_healer);
    FC_HASH_I32(npc->damage_taken_this_tick);
    FC_HASH_I32(npc->prayer_drain_dealt_this_tick);
    FC_HASH_I32(npc->healing_received_this_tick);
    FC_HASH_I32(npc->healing_given_this_tick);
    FC_HASH_I32(npc->healed_by_mejkot_this_tick);
    FC_HASH_I32(npc->healed_by_hurkot_this_tick);
    FC_HASH_I32(npc->healed_self_this_tick);
    FC_HASH_I32(npc->died_this_tick);
    for (int i = 0; i < FC_MAX_PENDING_HITS; ++i) {
        hash = fc_hash_pending_hit(hash, &npc->pending_hits[i]);
    }
    FC_HASH_I32(npc->num_pending_hits);
    return hash;
}

uint32_t fc_state_hash(const FcState* state) {
    uint32_t hash = FC_HASH_FNV_OFFSET;

    hash = fc_hash_player(hash, &state->player);
    for (int i = 0; i < FC_MAX_NPCS; ++i) {
        hash = fc_hash_npc(hash, &state->npcs[i]);
    }

    FC_HASH_I32(state->active_loadout);
    FC_HASH_I32(state->current_wave);
    FC_HASH_I32(state->rotation_id);
    FC_HASH_I32(state->npcs_remaining);
    FC_HASH_I32(state->total_npcs_killed);
    FC_HASH_I32(state->next_spawn_index);
    FC_HASH_I32(state->tick);
    FC_HASH_I32(state->terminal);
    FC_HASH_U32(state->rng_state);
    FC_HASH_U32(state->rng_seed);
    for (int x = 0; x < FC_ARENA_WIDTH; ++x) {
        for (int y = 0; y < FC_ARENA_HEIGHT; ++y) {
            hash = fc_hash_u8(hash, state->walkable[x][y]);
        }
    }
    for (int x = 0; x < FC_ARENA_WIDTH; ++x) {
        for (int y = 0; y < FC_ARENA_HEIGHT; ++y) {
            hash = fc_hash_u8(hash, state->movement_flags[x][y]);
        }
    }
    for (int x = 0; x < FC_ARENA_WIDTH; ++x) {
        for (int y = 0; y < FC_ARENA_HEIGHT; ++y) {
            hash = fc_hash_u8(hash, state->los_flags[x][y]);
        }
    }

    FC_HASH_I32(state->jad_healers_spawned);
    FC_HASH_I32(state->jad_healer_spawn_generations);

    FC_HASH_I32(state->damage_dealt_this_tick);
    FC_HASH_I32(state->hits_landed_this_tick);
    FC_HASH_I32(state->damage_taken_this_tick);
    FC_HASH_I32(state->prayer_lost_this_tick);
    FC_HASH_I32(state->overhead_prayer_lost_this_tick);
    FC_HASH_I32(state->tz_kih_prayer_drain_this_tick);
    FC_HASH_I32(state->npcs_killed_this_tick);
    FC_HASH_I32(state->respawned_jad_healers_killed_this_tick);
    FC_HASH_I32(state->wave_just_cleared);
    FC_HASH_I32(state->jad_damage_this_tick);
    FC_HASH_I32(state->jad_killed);
    FC_HASH_I32(state->correct_jad_prayer);
    FC_HASH_I32(state->wrong_jad_prayer);
    FC_HASH_I32(state->correct_danger_prayer);
    FC_HASH_I32(state->wrong_danger_prayer);
    FC_HASH_I32(state->attack_attempt_this_tick);
    FC_HASH_I32(state->invalid_action_this_tick);
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; ++i) {
        FC_HASH_I32(state->invalid_action_class_this_tick[i]);
    }
    FC_HASH_I32(state->movement_this_tick);
    FC_HASH_I32(state->idle_this_tick);
    FC_HASH_I32(state->food_used_this_tick);
    FC_HASH_I32(state->prayer_potion_used_this_tick);
    FC_HASH_I32(state->jad_heal_procs_this_tick);
    FC_HASH_I32(state->npc_heal_procs_this_tick);
    FC_HASH_I32(state->npc_heal_amount_this_tick);
    FC_HASH_I32(state->mejkot_heal_amount_this_tick);
    FC_HASH_I32(state->jad_heal_amount_this_tick);

    FC_HASH_F32(state->progress_required_work_start);
    FC_HASH_F32(state->progress_required_work_remaining);
    FC_HASH_F32(state->progress_current_wave_progress);
    FC_HASH_F32(state->progress_cave_progress);
    FC_HASH_I32(state->progress_ticks_since_positive);

    FC_HASH_I32(state->ep_ticks_pray_melee);
    FC_HASH_I32(state->ep_ticks_pray_range);
    FC_HASH_I32(state->ep_ticks_pray_magic);
    FC_HASH_I32(state->ep_correct_blocks);
    FC_HASH_I32(state->ep_wrong_prayer_hits);
    FC_HASH_I32(state->ep_no_prayer_hits);
    FC_HASH_I32(state->ep_damage_blocked);
    FC_HASH_I32(state->ep_prayer_switches);
    FC_HASH_I32(state->ep_pots_used);
    FC_HASH_I32(state->ep_pots_wasted);
    FC_HASH_I32(state->ep_pot_pre_prayer_sum);
    FC_HASH_I32(state->ep_food_eaten);
    FC_HASH_I32(state->ep_food_pre_hp_sum);
    FC_HASH_I32(state->ep_food_overhealed);
    FC_HASH_I32(state->ep_pots_overrestored);
    FC_HASH_I32(state->ep_tokxil_melee_ticks);
    FC_HASH_I32(state->ep_ketzek_melee_ticks);
    FC_HASH_I32(state->ep_attack_ready_ticks);
    FC_HASH_I32(state->ep_attack_attempt_ticks);
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; ++i) {
        FC_HASH_I32(state->ep_invalid_action_classes[i]);
    }
    for (int i = 0; i < NPC_TYPE_COUNT; ++i) {
        FC_HASH_I32(state->ep_damage_to_npc_type[i]);
        FC_HASH_I32(state->ep_resolved_hits_to_npc_type[i]);
        FC_HASH_I32(state->ep_damaging_hits_to_npc_type[i]);
        FC_HASH_I32(state->ep_attack_cycles_to_npc_type[i]);
        FC_HASH_I32(state->ep_target_ticks_by_npc_type[i]);
    }
    FC_HASH_I32(state->ep_target_held_ticks);
    FC_HASH_I32(state->ep_no_target_ticks);
    FC_HASH_I32(state->ep_target_in_range_los_ticks);
    FC_HASH_I32(state->ep_target_out_of_range_or_los_ticks);
    FC_HASH_I32(state->ep_attack_cooldown_wait_ticks);
    FC_HASH_I32(state->ep_ready_but_no_attack_ticks);
    FC_HASH_I32(state->ep_action_move_idle_ticks);
    FC_HASH_I32(state->ep_action_move_walk_ticks);
    FC_HASH_I32(state->ep_action_move_run_ticks);
    FC_HASH_I32(state->ep_action_attack_none_ticks);
    FC_HASH_I32(state->ep_action_attack_target_ticks);
    FC_HASH_I32(state->ep_action_prayer_noop_ticks);
    FC_HASH_I32(state->ep_action_prayer_cmd_ticks);
    FC_HASH_I32(state->ep_reached_wave_63);
    FC_HASH_I32(state->ep_jad_killed);
    FC_HASH_I32(state->wave_start_tick);
    FC_HASH_I32(state->ep_max_wave_ticks);
    FC_HASH_I32(state->ep_max_wave_ticks_wave);
    return hash;
}

#undef FC_HASH_I32
#undef FC_HASH_U32
#undef FC_HASH_F32

#undef FC_HASH_FNV_OFFSET
#undef FC_HASH_FNV_PRIME
#undef FC_HASH_I32
#undef FC_HASH_U32
#undef FC_HASH_F32

/* Items */
#include <limits.h>
#include <string.h>

/* Pinned to the existing FcLoadout balance, including legacy d'hide defence
 * and Pegasian strength. Equipment switching must not rebalance training.
 * Only the items supplied by our presets and their consumables are supported. */
enum { AMMO_NONE, AMMO_ARROW, AMMO_BOLT, AMMO_LOADED_DART };
static const FcItemDef ITEMS[] = {
    {.id=1169, .name="Coif", .slot=FC_EQUIP_SLOT_HEAD,
     .ranged_attack=2, .ranged_strength=0, .defence={4,6,8,4,4}, .prayer=0,
     .ranged_level=20, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=22109, .name="Ava's assembler", .slot=FC_EQUIP_SLOT_CAPE,
     .ranged_attack=8, .ranged_strength=2, .defence={1,1,1,8,2}, .prayer=0,
     .ranged_level=70, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=19547, .name="Necklace of anguish", .slot=FC_EQUIP_SLOT_NECK,
     .ranged_attack=15, .ranged_strength=5, .defence={0,0,0,0,0}, .prayer=2,
     .ranged_level=0, .defence_level=0, .hitpoints_level=75, .melee_attack=0, .melee_strength=0},
    {.id=27235, .name="Masori mask (f)", .slot=FC_EQUIP_SLOT_HEAD,
     .ranged_attack=12, .ranged_strength=2, .defence={8,10,12,12,9}, .prayer=1,
     .ranged_level=80, .defence_level=80, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=27238, .name="Masori body (f)", .slot=FC_EQUIP_SLOT_BODY,
     .ranged_attack=43, .ranged_strength=4, .defence={59,52,64,74,60}, .prayer=1,
     .ranged_level=80, .defence_level=80, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=27241, .name="Masori chaps (f)", .slot=FC_EQUIP_SLOT_LEGS,
     .ranged_attack=27, .ranged_strength=2, .defence={35,30,39,46,37}, .prayer=1,
     .ranged_level=80, .defence_level=80, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=26235, .name="Zaryte vambraces", .slot=FC_EQUIP_SLOT_HANDS,
     .ranged_attack=18, .ranged_strength=2, .defence={8,8,8,5,8}, .prayer=1,
     .ranged_level=80, .defence_level=45, .hitpoints_level=0, .melee_attack=-8, .melee_strength=0},
    {.id=13237, .name="Pegasian boots", .slot=FC_EQUIP_SLOT_FEET,
     .ranged_attack=12, .ranged_strength=0, .defence={5,5,5,5,5}, .prayer=0,
     .ranged_level=75, .defence_level=75, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=28310, .name="Venator ring", .slot=FC_EQUIP_SLOT_RING,
     .ranged_attack=10, .ranged_strength=2, .defence={0,0,0,0,0}, .prayer=0,
     .ranged_level=0, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=2503, .name="Black d'hide body", .slot=FC_EQUIP_SLOT_BODY,
     .ranged_attack=30, .ranged_strength=0, .defence={55,47,60,50,55}, .prayer=0,
     .ranged_level=70, .defence_level=40, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=2497, .name="Black d'hide chaps", .slot=FC_EQUIP_SLOT_LEGS,
     .ranged_attack=17, .ranged_strength=0, .defence={31,25,33,28,31}, .prayer=0,
     .ranged_level=70, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=2491, .name="Black d'hide vambraces", .slot=FC_EQUIP_SLOT_HANDS,
     .ranged_attack=11, .ranged_strength=0, .defence={6,5,7,8,0}, .prayer=0,
     .ranged_level=70, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=6328, .name="Snakeskin boots", .slot=FC_EQUIP_SLOT_FEET,
     .ranged_attack=3, .ranged_strength=0, .defence={1,1,2,1,0}, .prayer=0,
     .ranged_level=30, .defence_level=30, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=2581, .name="Robin hood hat", .slot=FC_EQUIP_SLOT_HEAD,
     .ranged_attack=8, .ranged_strength=0, .defence={4,6,8,4,4}, .prayer=0,
     .ranged_level=40, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=10499, .name="Ava's accumulator", .slot=FC_EQUIP_SLOT_CAPE,
     .ranged_attack=4, .ranged_strength=0, .defence={0,1,0,4,0}, .prayer=0,
     .ranged_level=50, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=1704, .name="Amulet of glory", .slot=FC_EQUIP_SLOT_NECK,
     .ranged_attack=10, .ranged_strength=0, .defence={3,3,3,3,3}, .prayer=3,
     .ranged_level=0, .defence_level=0, .hitpoints_level=0, .melee_attack=10, .melee_strength=6},
    {.id=12596, .name="Rangers' tunic", .slot=FC_EQUIP_SLOT_BODY,
     .ranged_attack=15, .ranged_strength=0, .defence={6,9,12,6,6}, .prayer=0,
     .ranged_level=40, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=12610, .name="Book of law", .slot=FC_EQUIP_SLOT_SHIELD,
     .ranged_attack=10, .ranged_strength=0, .defence={0,0,0,0,0}, .prayer=5,
     .ranged_level=0, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=2495, .name="Red d'hide chaps", .slot=FC_EQUIP_SLOT_LEGS,
     .ranged_attack=14, .ranged_strength=0, .defence={28,22,30,20,28}, .prayer=0,
     .ranged_level=60, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=11126, .name="Combat bracelet", .slot=FC_EQUIP_SLOT_HANDS,
     .ranged_attack=7, .ranged_strength=0, .defence={5,5,5,3,5}, .prayer=0,
     .ranged_level=0, .defence_level=0, .hitpoints_level=0, .melee_attack=7, .melee_strength=6},
    {.id=2577, .name="Ranger boots", .slot=FC_EQUIP_SLOT_FEET,
     .ranged_attack=8, .ranged_strength=0, .defence={2,3,4,2,0}, .prayer=0,
     .ranged_level=40, .defence_level=0, .hitpoints_level=0, .melee_attack=0, .melee_strength=0},
    {.id=11826, .name="Armadyl helmet", .slot=FC_EQUIP_SLOT_HEAD,
     .ranged_attack=10, .ranged_strength=0, .defence={6,8,10,10,8}, .prayer=1,
     .ranged_level=70, .defence_level=70, .hitpoints_level=0, .melee_attack=-5, .melee_strength=0},
    {.id=11828, .name="Armadyl chestplate", .slot=FC_EQUIP_SLOT_BODY,
     .ranged_attack=33, .ranged_strength=0, .defence={56,48,61,70,57}, .prayer=1,
     .ranged_level=70, .defence_level=70, .hitpoints_level=0, .melee_attack=-7, .melee_strength=0},
    {.id=11830, .name="Armadyl chainskirt", .slot=FC_EQUIP_SLOT_LEGS,
     .ranged_attack=20, .ranged_strength=0, .defence={32,26,34,40,33}, .prayer=1,
     .ranged_level=70, .defence_level=70, .hitpoints_level=0, .melee_attack=-6, .melee_strength=0},
    {.id=7462, .name="Barrows gloves", .slot=FC_EQUIP_SLOT_HANDS,
     .ranged_attack=12, .ranged_strength=0, .defence={12,12,12,6,12}, .prayer=0,
     .ranged_level=0, .defence_level=0, .hitpoints_level=0, .melee_attack=12, .melee_strength=12},
    {.id=23971, .name="Crystal helm", .slot=FC_EQUIP_SLOT_HEAD,
     .ranged_attack=9, .ranged_strength=0, .defence={12,8,14,10,18}, .prayer=2,
     .ranged_level=70, .defence_level=70, .hitpoints_level=0, .melee_attack=0, .melee_strength=0, .crystal_piece=FC_CRYSTAL_PIECE_HELM},
    {.id=23975, .name="Crystal body", .slot=FC_EQUIP_SLOT_BODY,
     .ranged_attack=31, .ranged_strength=0, .defence={46,38,48,44,68}, .prayer=3,
     .ranged_level=70, .defence_level=70, .hitpoints_level=0, .melee_attack=0, .melee_strength=0, .crystal_piece=FC_CRYSTAL_PIECE_BODY},
    {.id=23979, .name="Crystal legs", .slot=FC_EQUIP_SLOT_LEGS,
     .ranged_attack=18, .ranged_strength=0, .defence={26,21,30,34,38}, .prayer=2,
     .ranged_level=70, .defence_level=70, .hitpoints_level=0, .melee_attack=0, .melee_strength=0, .crystal_piece=FC_CRYSTAL_PIECE_LEGS},
    {.id=9185, .name="Rune crossbow", .slot=FC_EQUIP_SLOT_WEAPON,
     .ranged_attack=90, .ranged_strength=0, .two_handed=0, .ranged_level=61,
     .weapon_kind=0, .speed=5, .range=7, .ammo_kind=AMMO_BOLT, .visual_profile=0},
    {.id=20997, .name="Twisted bow", .slot=FC_EQUIP_SLOT_WEAPON,
     .ranged_attack=70, .ranged_strength=20, .two_handed=1, .ranged_level=85,
     .weapon_kind=1, .speed=5, .range=10, .ammo_kind=AMMO_ARROW, .ammo_tier=1, .visual_profile=1},
    {.id=12788, .name="Magic shortbow (i)", .slot=FC_EQUIP_SLOT_WEAPON,
     .ranged_attack=75, .ranged_strength=0, .two_handed=1, .ranged_level=50,
     .weapon_kind=0, .speed=3, .range=7, .ammo_kind=AMMO_ARROW, .visual_profile=4},
    {.id=12926, .name="Toxic blowpipe", .slot=FC_EQUIP_SLOT_WEAPON,
     .ranged_attack=30, .ranged_strength=20, .two_handed=1, .ranged_level=75,
     .weapon_kind=0, .speed=2, .range=5, .ammo_kind=AMMO_LOADED_DART, .visual_profile=5},
    {.id=11785, .name="Armadyl crossbow", .slot=FC_EQUIP_SLOT_WEAPON,
     .ranged_attack=100, .ranged_strength=0, .two_handed=0, .ranged_level=70,
     .weapon_kind=0, .speed=5, .range=8, .ammo_kind=AMMO_BOLT, .ammo_tier=1, .visual_profile=6, .prayer=1},
    {.id=25867, .name="Bow of faerdhinen (c)", .slot=FC_EQUIP_SLOT_WEAPON,
     .ranged_attack=128, .ranged_strength=106, .two_handed=1, .ranged_level=80,
     .weapon_kind=2, .speed=4, .range=10, .ammo_kind=AMMO_NONE, .visual_profile=7},
    {.id=9143, .name="Adamant bolts", .slot=FC_EQUIP_SLOT_AMMO,
     .stackable=1, .ranged_strength=100, .ammo_kind=AMMO_BOLT},
    {.id=11212, .name="Dragon arrow", .slot=FC_EQUIP_SLOT_AMMO,
     .stackable=1, .ranged_strength=60, .ammo_kind=AMMO_ARROW, .ammo_tier=1},
    {.id=892, .name="Rune arrow", .slot=FC_EQUIP_SLOT_AMMO,
     .stackable=1, .ranged_strength=49, .ammo_kind=AMMO_ARROW},
    {.id=21946, .name="Diamond dragon bolts (e)", .slot=FC_EQUIP_SLOT_AMMO,
     .stackable=1, .ranged_strength=122, .ammo_kind=AMMO_BOLT, .ammo_tier=1},
    {.id=385, .name="Shark", .slot=-1},
    {.id=2434, .name="Prayer potion(4)", .slot=-1},
    {.id=139, .name="Prayer potion(3)", .slot=-1},
    {.id=141, .name="Prayer potion(2)", .slot=-1},
    {.id=143, .name="Prayer potion(1)", .slot=-1},
    {.id=229, .name="Vial", .slot=-1},
};

const FcItemDef *fc_item_definition(int item_id) {
    for (unsigned i = 0; i < sizeof(ITEMS) / sizeof(ITEMS[0]); i++)
        if (ITEMS[i].id == item_id) return &ITEMS[i];
    return NULL;
}

static void recalculate_equipment(FcPlayer *p) {
    p->ranged_attack_bonus = p->ranged_strength_bonus = 0;
    p->defence_stab = p->defence_slash = p->defence_crush = 0;
    p->defence_magic = p->defence_ranged = p->prayer_bonus = 0;
    p->melee_attack_bonus = p->melee_strength_bonus = p->crystal_piece_mask = 0;
    const FcItemDef *weapon = fc_item_definition(p->equipment[FC_EQUIP_SLOT_WEAPON].item_id);
    const FcItemDef *ammo = fc_item_definition(p->equipment[FC_EQUIP_SLOT_AMMO].item_id);
    /* Quiver items may be worn with any weapon, but firing requires both the
     * correct category and a supported tier (RSMod validateArrows/Bolts). */
    int usable_ammo = weapon && ammo && weapon->ammo_kind == ammo->ammo_kind &&
        ammo->ammo_tier <= weapon->ammo_tier;
    for (int i = 0; i < FC_EQUIPMENT_SLOTS; i++) {
        const FcItemDef *item = fc_item_definition(p->equipment[i].item_id);
        if (!item) continue;
        if (i == FC_EQUIP_SLOT_AMMO && !usable_ammo)
            continue;
        p->ranged_attack_bonus += item->ranged_attack;
        p->ranged_strength_bonus += item->ranged_strength;
        p->defence_stab += item->defence[0];
        p->defence_slash += item->defence[1];
        p->defence_crush += item->defence[2];
        p->defence_magic += item->defence[3];
        p->defence_ranged += item->defence[4];
        p->prayer_bonus += item->prayer;
        p->melee_attack_bonus += item->melee_attack;
        p->melee_strength_bonus += item->melee_strength;
        p->crystal_piece_mask |= item->crystal_piece;
    }
    p->weapon_kind = weapon ? weapon->weapon_kind : FC_WEAPON_UNARMED;
    p->weapon_speed = weapon ? weapon->speed : 4;
    p->weapon_range = weapon ? weapon->range : 1;
    p->weapon_uses_ammo = weapon && weapon->ammo_kind != AMMO_NONE;
    p->ammo_count = 0;
    if (weapon && weapon->ammo_kind == AMMO_LOADED_DART) {
        p->ammo_count = p->equipment[FC_EQUIP_SLOT_WEAPON].charges;
        p->ranged_strength_bonus += 17; /* preset's loaded adamant darts */
    } else if (usable_ammo) {
        p->ammo_count = p->equipment[FC_EQUIP_SLOT_AMMO].quantity;
    }
}

static int potion_doses(int id) {
    switch (id) {
        case 2434: return 4;
        case 139: return 3;
        case 141: return 2;
        case 143: return 1;
        default: return 0;
    }
}

static void reset_supplies(FcPlayer *p, int sharks, int doses) {
    if (sharks < 0) sharks = 0;
    if (sharks > FC_MAX_SHARKS) sharks = FC_MAX_SHARKS;
    if (doses < 0) doses = 0;
    if (doses > FC_MAX_PRAYER_DOSES) doses = FC_MAX_PRAYER_DOSES;
    memset(p->inventory, 0, sizeof(p->inventory));
    p->sharks_remaining = sharks;
    p->prayer_doses_remaining = doses;
    const int pots[] = {0, 143, 141, 139, 2434};
    int slot = 0;
    while (doses > 0) {
        int count = doses > 4 ? 4 : doses;
        p->inventory[slot++] = (FcItemStack){pots[count], 1, 0};
        doses -= count;
    }
    for (int i = 0; i < sharks; i++)
        p->inventory[slot++] = (FcItemStack){385, 1, 0};
    p->selected_food_slot = p->selected_potion_slot = -1;
}

void fc_set_initial_supplies(FcState *state, int sharks, int prayer_doses) {
    /* Reset-time configuration only, not a gameplay inventory refill API. */
    if (state && state->tick == 0)
        reset_supplies(&state->player, sharks, prayer_doses);
}

void fc_items_init(FcPlayer *p, const FcLoadout *loadout) {
    memset(p->equipment, 0, sizeof(p->equipment));
    for (int i = 0; i < loadout->equipment_count; i++) {
        const FcLoadoutEquipmentItem *item = &loadout->equipment[i];
        if (item->item_id == 810) continue; /* darts are loaded in the blowpipe */
        p->equipment[item->slot] = (FcItemStack){
            (int)item->item_id, item->slot == FC_EQUIP_SLOT_AMMO ? loadout->ammo : 1,
            item->item_id == 12926 ? loadout->ammo : 0
        };
    }
    recalculate_equipment(p);
    reset_supplies(p, FC_MAX_SHARKS, FC_MAX_PRAYER_DOSES);
}

const char *fc_item_result_message(FcItemResult result) {
    switch (result) {
        case FC_ITEM_OK: return "";
        case FC_ITEM_NO_SPACE: return "You don't have enough inventory space.";
        case FC_ITEM_REQUIREMENTS: return "Your levels are too low to wear this item.";
        case FC_ITEM_BUSY: return "You can't change equipment right now.";
        default: return "You can't use that item here.";
    }
}

static int free_slot(const FcItemStack inventory[FC_INVENTORY_SLOTS]) {
    for (int i = 0; i < FC_INVENTORY_SLOTS; i++)
        if (!inventory[i].item_id) return i;
    return -1;
}

static int add_to_inventory(FcItemStack inventory[FC_INVENTORY_SLOTS], FcItemStack item) {
    if (!item.item_id) return 1;
    const FcItemDef *def = fc_item_definition(item.item_id);
    if (!def || item.quantity <= 0) return 0;
    if (def->stackable) {
        for (int i = 0; i < FC_INVENTORY_SLOTS; i++) {
            if (inventory[i].item_id != item.item_id) continue;
            if (item.quantity > INT_MAX - inventory[i].quantity) return 0;
            inventory[i].quantity += item.quantity;
            return 1;
        }
    }
    int index = free_slot(inventory);
    if (index < 0) return 0;
    inventory[index] = item;
    return 1;
}

static int can_change_items(const FcState *state) {
    return state && !state->terminal && state->player.current_hp > 0;
}

FcItemResult fc_equip_item(FcState *state, int index) {
    if (!can_change_items(state)) return FC_ITEM_BUSY;
    FcPlayer *p = &state->player;
    if (index < 0 || index >= FC_INVENTORY_SLOTS) return FC_ITEM_INVALID;
    FcItemStack incoming = p->inventory[index];
    const FcItemDef *item = fc_item_definition(incoming.item_id);
    if (!item || item->slot < 0 || incoming.quantity <= 0 ||
        (!item->stackable && incoming.quantity != 1)) return FC_ITEM_INVALID;
    if (p->ranged_level < item->ranged_level || p->defence_level < item->defence_level ||
        p->max_hp / 10 < item->hitpoints_level) return FC_ITEM_REQUIREMENTS;
    /* Plan on copies, including both hands. A failed secondary transfer cannot
     * partially equip an item, remove a shield, or interrupt combat. */
    FcItemStack inventory[FC_INVENTORY_SLOTS], equipment[FC_EQUIPMENT_SLOTS];
    memcpy(inventory, p->inventory, sizeof(inventory));
    memcpy(equipment, p->equipment, sizeof(equipment));
    FcItemStack *worn = &equipment[item->slot];
    if (item->stackable && worn->item_id == incoming.item_id) {
        int amount = INT_MAX - worn->quantity;
        if (amount > incoming.quantity) amount = incoming.quantity;
        if (!amount) return FC_ITEM_NO_SPACE;
        worn->quantity += amount;
        inventory[index].quantity -= amount;
        if (!inventory[index].quantity) inventory[index] = (FcItemStack){0};
    } else {
        inventory[index] = *worn;
        *worn = incoming;
    }
    const FcItemDef *weapon = fc_item_definition(equipment[FC_EQUIP_SLOT_WEAPON].item_id);
    int displaced = item->two_handed ? FC_EQUIP_SLOT_SHIELD :
        item->slot == FC_EQUIP_SLOT_SHIELD && weapon && weapon->two_handed ?
        FC_EQUIP_SLOT_WEAPON : -1;
    if (displaced >= 0 && equipment[displaced].item_id) {
        /* RSMod returns the conflict to the source slot when it wasn't needed
         * for a primary swap, otherwise to the first free inventory slot. */
        if (!inventory[index].item_id) inventory[index] = equipment[displaced];
        else if (!add_to_inventory(inventory, equipment[displaced])) return FC_ITEM_NO_SPACE;
        equipment[displaced] = (FcItemStack){0};
    }
    memcpy(p->inventory, inventory, sizeof(inventory));
    memcpy(p->equipment, equipment, sizeof(equipment));
    recalculate_equipment(p);
    p->attack_target_idx = -1; /* held-item action interrupts interaction */
    p->approach_target = 0;
    p->approach_target_x = p->approach_target_y = -1;
    p->approach_target_size = 0;
    return FC_ITEM_OK;
}

FcItemResult fc_unequip_item(FcState *state, int slot) {
    if (!can_change_items(state)) return FC_ITEM_BUSY;
    if (slot < 0 || slot >= FC_EQUIPMENT_SLOTS ||
        !state->player.equipment[slot].item_id) return FC_ITEM_INVALID;
    FcPlayer *p = &state->player;
    FcItemStack inventory[FC_INVENTORY_SLOTS];
    memcpy(inventory, p->inventory, sizeof(inventory));
    if (!add_to_inventory(inventory, p->equipment[slot])) return FC_ITEM_NO_SPACE;
    memcpy(p->inventory, inventory, sizeof(inventory));
    p->equipment[slot] = (FcItemStack){0};
    recalculate_equipment(p);
    /* Worn-item Remove does not cancel the existing interaction. */
    return FC_ITEM_OK;
}

FcItemResult fc_inventory_swap(FcState *state, int first, int second) {
    if (!can_change_items(state)) return FC_ITEM_BUSY;
    if (first < 0 || first >= FC_INVENTORY_SLOTS || second < 0 ||
        second >= FC_INVENTORY_SLOTS) return FC_ITEM_INVALID;
    FcItemStack temp = state->player.inventory[first];
    state->player.inventory[first] = state->player.inventory[second];
    state->player.inventory[second] = temp;
    state->player.selected_food_slot = state->player.selected_potion_slot = -1;
    return FC_ITEM_OK;
}

FcItemResult fc_select_consumable(FcState *state, int slot) {
    if (!can_change_items(state)) return FC_ITEM_BUSY;
    if (slot < 0 || slot >= FC_INVENTORY_SLOTS) return FC_ITEM_INVALID;
    int id = state->player.inventory[slot].item_id;
    if (id == 385) state->player.selected_food_slot = slot;
    else if (potion_doses(id)) state->player.selected_potion_slot = slot;
    else return FC_ITEM_INVALID;
    return FC_ITEM_OK;
}

void fc_items_consume(FcPlayer *p, int potion) {
    int selected = potion ? p->selected_potion_slot : p->selected_food_slot;
    for (int n = -1; n < FC_INVENTORY_SLOTS; n++) {
        int slot = n < 0 ? selected : n;
        if (slot < 0 || slot >= FC_INVENTORY_SLOTS) continue;
        FcItemStack *item = &p->inventory[slot];
        int doses = potion_doses(item->item_id);
        if (potion && doses) {
            const int replacement[] = {229, 143, 141, 139};
            item->item_id = replacement[doses - 1];
            break;
        }
        if (!potion && item->item_id == 385) {
            *item = (FcItemStack){0};
            break;
        }
    }
    if (potion) p->prayer_doses_remaining--;
    else p->sharks_remaining--;
}

void fc_items_spend_ammo(FcPlayer *p) {
    p->ammo_count--;
    FcItemStack *weapon = &p->equipment[FC_EQUIP_SLOT_WEAPON];
    if (weapon->item_id == 12926) weapon->charges--;
    else {
        FcItemStack *ammo = &p->equipment[FC_EQUIP_SLOT_AMMO];
        if (ammo->quantity > 0 && --ammo->quantity == 0) {
            *ammo = (FcItemStack){0};
            recalculate_equipment(p);
        }
    }
}


/* Loadouts */
/*
 * LOADOUT A: Mid-level — Black D'hide + Rune Crossbow
 *
 *   Slot        Item                    Rng Atk  Rng Str  Stab  Slash Crush Magic Ranged Prayer
 *   ----        ----                    -------  -------  ----  ----- ----- ----- ------ ------
 *   Head        Coif                      2        0        4     6     8     4     4      0
 *   Weapon      Rune Crossbow            90        0        0     0     0     0     0      0
 *   Body        Black D'hide Body        30        0       55    47    60    50    55      0
 *   Legs        Black D'hide Chaps       17        0       31    25    33    28    31      0
 *   Hands       Black D'hide Vambraces   11        0        6     5     7     8     0      0
 *   Feet        Snakeskin Boots           3        0        1     1     2     1     0      0
 *   Ammo        Adamant Bolts             0      100        0     0     0     0     0      0
 *                                      ----     ----     ----  ----  ----  ----  ----   ----
 *   TOTAL                               153      100       97    84   110    91    90      0
 */

/*
 * LOADOUT B: End-game — Masori (f) + Twisted Bow
 *
 *   Slot        Item                    Rng Atk  Rng Str  Stab  Slash Crush Magic Ranged Prayer
 *   ----        ----                    -------  -------  ----  ----- ----- ----- ------ ------
 *   Head        Masori mask (f)          12        2        8    10    12    12     9      1
 *   Cape        Ava's assembler           8        2        0     0     0     0     0      0
 *   Neck        Necklace of anguish      15        5        0     0     0     0     0      2
 *   Weapon      Twisted bow              70       20        0     0     0     0     0      0
 *   Body        Masori body (f)          43        4       59    52    64    74    60      1
 *   Legs        Masori chaps (f)         27        2       35    30    39    46    37      1
 *   Hands       Zaryte vambraces         18        2        8     8     8     5     8      1
 *   Feet        Pegasian boots           12        0        5     5     5     5     5      0
 *   Ring        Venator ring             10        2        0     0     0     0     0      0
 *   Ammo        Dragon arrows             0       60        0     0     0     0     0      0
 *   Shield      (none)                    0        0        0     0     0     0     0      0
 *                                      ----     ----     ----  ----  ----  ----  ----   ----
 *   TOTAL                               215       99      116   106   129   150   121      6
 */

const FcLoadout FC_LOADOUTS[FC_NUM_LOADOUTS] = {
    /* [FC_LOADOUT_BLACK_DHIDE_RCB] Mid-level — Black D'hide + Rune Crossbow */
    {
        .name         = "A: Black D'hide + RCB",
        .weapon_name  = "Rune crossbow",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_BLACK_DHIDE_RCB,
        .combat_style_profile = 9,
        .max_hp       = 700,   /* 70 HP */
        .max_prayer   = 430,   /* 43 prayer */
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 70,
        .ranged_lvl   = 70,
        .prayer_lvl   = 43,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_GENERIC_RANGED,
        .weapon_uses_ammo = 1,
        .weapon_speed = 5,
        .weapon_range = 7,
        .ranged_atk   = 153,   /* 2+90+30+17+11+3 */
        .ranged_str   = 100,   /* adamant bolts */
        .def_stab     = 97,    /* 4+0+55+31+6+1 */
        .def_slash    = 84,    /* 6+0+47+25+5+1 */
        .def_crush    = 110,   /* 8+0+60+33+7+2 */
        .def_magic    = 91,    /* 4+0+50+28+8+1 */
        .def_ranged   = 90,    /* 4+0+55+31+0+0 */
        .prayer_bonus = 0,
        .ammo         = 50000,
        .equipment_count = 7,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   1169, 0, "Coif"},
            {FC_EQUIP_SLOT_WEAPON, 9185, 0, "Rune crossbow"},
            {FC_EQUIP_SLOT_BODY,   2503, 0, "Black d'hide body"},
            {FC_EQUIP_SLOT_AMMO,   9143, 0, "Adamant bolts"},
            {FC_EQUIP_SLOT_LEGS,   2497, 0, "Black d'hide chaps"},
            {FC_EQUIP_SLOT_HANDS,  2491, 0, "Black d'hide vambraces"},
            {FC_EQUIP_SLOT_FEET,   6328, 0, "Snakeskin boots"},
        },
        .model_item_count = 6,
        .model_item_ids = {1169, 9185, 2503, 2497, 2491, 6328},
    },
    /* [FC_LOADOUT_SOTA_TBOW] End-game — Masori (f) + Twisted Bow */
    {
        .name         = "B: Masori (f) + TBow",
        .weapon_name  = "Twisted bow",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_SOTA_TBOW,
        .combat_style_profile = 25,
        .max_hp       = 990,   /* 99 HP */
        .max_prayer   = 990,   /* 99 prayer */
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 99,
        .ranged_lvl   = 99,
        .prayer_lvl   = 99,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_TWISTED_BOW,
        .weapon_uses_ammo = 1,
        .weapon_speed = 5,     /* rapid */
        .weapon_range = 10,
        .ranged_atk   = 215,   /* 12+8+15+70+43+27+18+12+10 */
        .ranged_str   = 99,    /* 2+2+5+20+4+2+2+0+2+60(dragon arrows) */
        .def_stab     = 116,   /* 8+1+0+0+59+35+8+5+0+0 */
        .def_slash    = 106,   /* 10+1+0+0+52+30+8+5+0+0 */
        .def_crush    = 129,   /* 12+1+0+0+64+39+8+5+0+0 */
        .def_magic    = 150,   /* 12+8+0+0+74+46+5+5+0+0 */
        .def_ranged   = 121,   /* 9+2+0+0+60+37+8+5+0+0 */
        .prayer_bonus = 6,     /* 1+0+2+0+1+1+1+0+0+0 */
        .ammo         = 50000,
        .equipment_count = 10,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   27235, 0, "Masori mask (f)"},
            {FC_EQUIP_SLOT_CAPE,   22109, 0, "Ava's assembler"},
            {FC_EQUIP_SLOT_NECK,   19547, 0, "Necklace of anguish"},
            {FC_EQUIP_SLOT_WEAPON, 20997, 0, "Twisted bow"},
            {FC_EQUIP_SLOT_BODY,   27238, 0, "Masori body (f)"},
            {FC_EQUIP_SLOT_AMMO,   11212, 0, "Dragon arrows"},
            {FC_EQUIP_SLOT_LEGS,   27241, 0, "Masori chaps (f)"},
            {FC_EQUIP_SLOT_HANDS,  26235, 0, "Zaryte vambraces"},
            {FC_EQUIP_SLOT_FEET,   13237, 0, "Pegasian boots"},
            {FC_EQUIP_SLOT_RING,   28310, 0, "Venator ring"},
        },
        .model_item_count = 8,
        .model_item_ids = {27235, 22109, 19547, 20997, 27238, 27241, 26235, 13237},
    },
    /* [FC_LOADOUT_LOW_DEF_RCB] Low-defence — Robin Hood + Red D'hide + RCB */
    {
        .name         = "C: 1 Def Robin + RCB",
        .weapon_name  = "Rune crossbow",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_LOW_DEF_RCB,
        .combat_style_profile = 9,
        .max_hp       = 550,   /* 55 HP */
        .max_prayer   = 430,   /* 43 prayer */
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 1,
        .ranged_lvl   = 61,
        .prayer_lvl   = 43,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_GENERIC_RANGED,
        .weapon_uses_ammo = 1,
        .weapon_speed = 5,     /* rapid rune crossbow */
        .weapon_range = 7,
        .ranged_atk   = 166,   /* 8+4+10+90+15+14+7+8+10 */
        .ranged_str   = 100,   /* adamant bolts */
        .def_stab     = 48,    /* 4+0+3+0+6+28+5+2+0 */
        .def_slash    = 49,    /* 6+1+3+0+9+22+5+3+0 */
        .def_crush    = 62,    /* 8+0+3+0+12+30+5+4+0 */
        .def_magic    = 42,    /* 4+4+3+0+6+20+3+2+0 */
        .def_ranged   = 46,    /* 4+0+3+0+6+28+5+0+0 */
        .prayer_bonus = 8,     /* glory + book of law */
        .ammo         = 50000,
        .equipment_count = 10,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   2581,  0, "Robin hood hat"},
            {FC_EQUIP_SLOT_CAPE,   10499, 0, "Ava's accumulator"},
            {FC_EQUIP_SLOT_NECK,   1704,  0, "Amulet of glory"},
            {FC_EQUIP_SLOT_WEAPON, 9185,  0, "Rune crossbow"},
            {FC_EQUIP_SLOT_BODY,   12596, 0, "Rangers' tunic"},
            {FC_EQUIP_SLOT_SHIELD, 12610, 0, "Book of law"},
            {FC_EQUIP_SLOT_AMMO,   9143,  0, "Adamant bolts"},
            {FC_EQUIP_SLOT_LEGS,   2495,  0, "Red d'hide chaps"},
            {FC_EQUIP_SLOT_HANDS,  11126, 0, "Combat bracelet"},
            {FC_EQUIP_SLOT_FEET,   2577,  0, "Ranger boots"},
        },
        .model_item_count = 9,
        .model_item_ids = {2581, 10499, 1704, 9185, 12596, 12610, 2495, 11126, 2577},
    },
    /* [FC_LOADOUT_RCB_PURE] Low-level 1-def Fight Caves rune crossbow pure */
    {
        .name         = "RCB Pure",
        .weapon_name  = "Rune crossbow",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_RCB_PURE,
        .combat_style_profile = 9,
        .max_hp       = 550,
        .max_prayer   = 430,
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 1,
        .ranged_lvl   = 61,
        .prayer_lvl   = 43,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_GENERIC_RANGED,
        .weapon_uses_ammo = 1,
        .weapon_speed = 5,
        .weapon_range = 7,
        .ranged_atk   = 166,
        .ranged_str   = 100,
        .def_stab     = 48,
        .def_slash    = 49,
        .def_crush    = 62,
        .def_magic    = 42,
        .def_ranged   = 46,
        .prayer_bonus = 8,
        .ammo         = 50000,
        .equipment_count = 10,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   2581,  0, "Robin hood hat"},
            {FC_EQUIP_SLOT_CAPE,   10499, 0, "Ava's accumulator"},
            {FC_EQUIP_SLOT_NECK,   1704,  0, "Amulet of glory"},
            {FC_EQUIP_SLOT_WEAPON, 9185,  0, "Rune crossbow"},
            {FC_EQUIP_SLOT_BODY,   12596, 0, "Rangers' tunic"},
            {FC_EQUIP_SLOT_SHIELD, 12610, 0, "Book of law"},
            {FC_EQUIP_SLOT_AMMO,   9143,  0, "Adamant bolts"},
            {FC_EQUIP_SLOT_LEGS,   2495,  0, "Red d'hide chaps"},
            {FC_EQUIP_SLOT_HANDS,  11126, 0, "Combat bracelet"},
            {FC_EQUIP_SLOT_FEET,   2577,  0, "Ranger boots"},
        },
        .model_item_count = 9,
        .model_item_ids = {2581, 10499, 1704, 9185, 12596, 12610, 2495, 11126, 2577},
    },
    /* [FC_LOADOUT_MSBI_PURE] Faster but weaker 1-def magic shortbow pure */
    {
        .name         = "MSB(i) Pure",
        .weapon_name  = "Magic shortbow (i)",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_MSBI_PURE,
        .combat_style_profile = 25,
        .max_hp       = 600,
        .max_prayer   = 430,
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 1,
        .ranged_lvl   = 70,
        .prayer_lvl   = 43,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_GENERIC_RANGED,
        .weapon_uses_ammo = 1,
        .weapon_speed = 3,
        .weapon_range = 7,
        .ranged_atk   = 141,
        .ranged_str   = 49,
        .def_stab     = 48,
        .def_slash    = 49,
        .def_crush    = 62,
        .def_magic    = 42,
        .def_ranged   = 46,
        .prayer_bonus = 3,
        .ammo         = 50000,
        .equipment_count = 9,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   2581,  0, "Robin hood hat"},
            {FC_EQUIP_SLOT_CAPE,   10499, 0, "Ava's accumulator"},
            {FC_EQUIP_SLOT_NECK,   1704,  0, "Amulet of glory"},
            {FC_EQUIP_SLOT_WEAPON, 12788, 0, "Magic shortbow (i)"},
            {FC_EQUIP_SLOT_AMMO,   892,   0, "Rune arrow"},
            {FC_EQUIP_SLOT_BODY,   12596, 0, "Rangers' tunic"},
            {FC_EQUIP_SLOT_LEGS,   2495,  0, "Red d'hide chaps"},
            {FC_EQUIP_SLOT_HANDS,  11126, 0, "Combat bracelet"},
            {FC_EQUIP_SLOT_FEET,   2577,  0, "Ranger boots"},
        },
        .model_item_count = 8,
        .model_item_ids = {2581, 10499, 1704, 12788, 12596, 2495, 11126, 2577},
    },
    /* [FC_LOADOUT_BLOWPIPE_PURE] Fast 1-def toxic blowpipe pure with loaded adamant darts */
    {
        .name         = "Blowpipe Pure",
        .weapon_name  = "Toxic blowpipe",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_BLOWPIPE_PURE,
        .combat_style_profile = 23,
        .max_hp       = 750,
        .max_prayer   = 430,
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 1,
        .ranged_lvl   = 75,
        .prayer_lvl   = 43,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_GENERIC_RANGED,
        .weapon_uses_ammo = 1,
        .weapon_speed = 2,
        .weapon_range = 5,
        .ranged_atk   = 101,
        .ranged_str   = 42,
        .def_stab     = 45,
        .def_slash    = 46,
        .def_crush    = 59,
        .def_magic    = 39,
        .def_ranged   = 43,
        .prayer_bonus = 2,
        .ammo         = 50000,
        .equipment_count = 9,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   2581,  0, "Robin hood hat"},
            {FC_EQUIP_SLOT_CAPE,   10499, 0, "Ava's accumulator"},
            {FC_EQUIP_SLOT_NECK,   19547, 0, "Necklace of anguish"},
            {FC_EQUIP_SLOT_WEAPON, 12926, 0, "Toxic blowpipe"},
            {FC_EQUIP_SLOT_AMMO,   810,   0, "Adamant dart"},
            {FC_EQUIP_SLOT_BODY,   12596, 0, "Rangers' tunic"},
            {FC_EQUIP_SLOT_LEGS,   2495,  0, "Red d'hide chaps"},
            {FC_EQUIP_SLOT_HANDS,  11126, 0, "Combat bracelet"},
            {FC_EQUIP_SLOT_FEET,   2577,  0, "Ranger boots"},
        },
        .model_item_count = 8,
        .model_item_ids = {2581, 10499, 19547, 12926, 12596, 2495, 11126, 2577},
    },
    /* [FC_LOADOUT_ACB_ARMADYL] Tankier high-level Armadyl crossbow + Armadyl armour */
    {
        .name         = "ACB Armadyl",
        .weapon_name  = "Armadyl crossbow",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_ACB_ARMADYL,
        .combat_style_profile = 9,
        .max_hp       = 800,
        .max_prayer   = 700,
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 75,
        .ranged_lvl   = 80,
        .prayer_lvl   = 70,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_GENERIC_RANGED,
        .weapon_uses_ammo = 1,
        .weapon_speed = 5,
        .weapon_range = 8,
        .ranged_atk   = 220,
        .ranged_str   = 129,
        .def_stab     = 112,
        .def_slash    = 100,
        .def_crush    = 123,
        .def_magic    = 139,
        .def_ranged   = 117,
        .prayer_bonus = 11,
        .ammo         = 50000,
        .equipment_count = 10,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   11826, 0, "Armadyl helmet"},
            {FC_EQUIP_SLOT_CAPE,   22109, 0, "Ava's assembler"},
            {FC_EQUIP_SLOT_NECK,   19547, 0, "Necklace of anguish"},
            {FC_EQUIP_SLOT_WEAPON, 11785, 0, "Armadyl crossbow"},
            {FC_EQUIP_SLOT_BODY,   11828, 0, "Armadyl chestplate"},
            {FC_EQUIP_SLOT_SHIELD, 12610, 0, "Book of law"},
            {FC_EQUIP_SLOT_AMMO,   21946, 0, "Diamond dragon bolts (e)"},
            {FC_EQUIP_SLOT_LEGS,   11830, 0, "Armadyl chainskirt"},
            {FC_EQUIP_SLOT_HANDS,  7462,  0, "Barrows gloves"},
            {FC_EQUIP_SLOT_FEET,   13237, 0, "Pegasian boots"},
        },
        .model_item_count = 9,
        .model_item_ids = {11826, 22109, 19547, 11785, 11828, 12610, 11830, 7462, 13237},
    },
    /* [FC_LOADOUT_BOWFA_CRYSTAL] Bowfa + crystal armour. */
    {
        .name         = "Bowfa Crystal",
        .weapon_name  = "Bow of faerdhinen (c)",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_BOWFA_CRYSTAL,
        .combat_style_profile = 25,
        .max_hp       = 850,
        .max_prayer   = 700,
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 75,
        .ranged_lvl   = 85,
        .prayer_lvl   = 70,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_BOW_OF_FAERDHINEN,
        .weapon_uses_ammo = 0,
        .crystal_piece_mask = FC_CRYSTAL_PIECE_ALL,
        .weapon_speed = 4,
        .weapon_range = 10,
        .ranged_atk   = 233,
        .ranged_str   = 113,
        .def_stab     = 102,
        .def_slash    = 85,
        .def_crush    = 110,
        .def_magic    = 107,
        .def_ranged   = 143,
        .prayer_bonus = 9,
        .ammo         = 0,
        .equipment_count = 8,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   23971, 0, "Crystal helm"},
            {FC_EQUIP_SLOT_CAPE,   22109, 0, "Ava's assembler"},
            {FC_EQUIP_SLOT_NECK,   19547, 0, "Necklace of anguish"},
            {FC_EQUIP_SLOT_WEAPON, 25867, 0, "Bow of faerdhinen (c)"},
            {FC_EQUIP_SLOT_BODY,   23975, 0, "Crystal body"},
            {FC_EQUIP_SLOT_LEGS,   23979, 0, "Crystal legs"},
            {FC_EQUIP_SLOT_HANDS,  7462,  0, "Barrows gloves"},
            {FC_EQUIP_SLOT_FEET,   13237, 0, "Pegasian boots"},
        },
        .model_item_count = 8,
        .model_item_ids = {23971, 22109, 19547, 25867, 23975, 23979, 7462, 13237},
    },
    /* [FC_LOADOUT_TBOW_MASORI] Max-ish Twisted bow + fortified Masori loadout */
    {
        .name         = "Tbow Masori",
        .weapon_name  = "Twisted bow",
        .player_model_id = FC_PLAYER_MODEL_BASE + FC_LOADOUT_TBOW_MASORI,
        .combat_style_profile = 25,
        .max_hp       = 990,
        .max_prayer   = 770,
        .attack_lvl   = 1,
        .strength_lvl = 1,
        .defence_lvl  = 80,
        .ranged_lvl   = 99,
        .prayer_lvl   = 77,
        .magic_lvl    = 1,
        .weapon_kind  = FC_WEAPON_TWISTED_BOW,
        .weapon_uses_ammo = 1,
        .weapon_speed = 5,
        .weapon_range = 10,
        .ranged_atk   = 205,
        .ranged_str   = 97,
        .def_stab     = 116,
        .def_slash    = 106,
        .def_crush    = 129,
        .def_magic    = 150,
        .def_ranged   = 121,
        .prayer_bonus = 6,
        .ammo         = 50000,
        .equipment_count = 9,
        .equipment = {
            {FC_EQUIP_SLOT_HEAD,   27235, 0, "Masori mask (f)"},
            {FC_EQUIP_SLOT_CAPE,   22109, 0, "Ava's assembler"},
            {FC_EQUIP_SLOT_NECK,   19547, 0, "Necklace of anguish"},
            {FC_EQUIP_SLOT_WEAPON, 20997, 0, "Twisted bow"},
            {FC_EQUIP_SLOT_BODY,   27238, 0, "Masori body (f)"},
            {FC_EQUIP_SLOT_AMMO,   11212, 0, "Dragon arrow"},
            {FC_EQUIP_SLOT_LEGS,   27241, 0, "Masori chaps (f)"},
            {FC_EQUIP_SLOT_HANDS,  26235, 0, "Zaryte vambraces"},
            {FC_EQUIP_SLOT_FEET,   13237, 0, "Pegasian boots"},
        },
        .model_item_count = 8,
        .model_item_ids = {27235, 22109, 19547, 20997, 27238, 27241, 26235, 13237},
    },
};


/* Npc */
#include <stddef.h>

/*
 * fc_npc.c — NPC framework with stat table and type-specific AI dispatch.
 *
 * PR 5: All 8 NPC types have full AI.
 *
 * NPC AI per tick (generic):
 *   1. If dead or inactive, skip.
 *   2. Decrement attack timer.
 *   3. Type-specific behavior (Jad style selection, Yt-MejKot heal, Yt-HurKot heal).
 *   4. If not in attack range, move toward player (greedy step).
 *   5. If in range and attack timer ready, roll attack and queue pending hit.
 *
 * Type-specific:
 *   Tz-Kih:     Melee + prayer drain on hit.
 *   Tz-Kek:     Melee. Splits into 2 small Tz-Kek on death.
 *   Tz-Kek-Sm:  Melee. (no special)
 *   Tok-Xil:    Ranged with projectile delay.
 *   Yt-MejKot:  Chooses one melee-cycle action: attack or heal a weak NPC.
 *   Ket-Zek:    Magic with projectile delay.
 *   TzTok-Jad:  Magic/ranged at distance; melee/magic/ranged at range 1.
 *   Yt-HurKot:  Heals Jad in range; attacks the player after being tagged.
 */

/* ======================================================================== */
/* NPC stat table                                                            */
/* ======================================================================== */

/*
 * Fight Caves stats use the reviewed OSRS parity table. Sizes and non-combat
 * behavior fields retain their existing cache/config-derived values.
 */
static const FcNpcStats NPC_STATS[NPC_TYPE_COUNT] = {
    [NPC_NONE] = {0},

    /* NPC_TZ_KIH: Lv 22 melee bat. Drains damage + 1 Prayer point.
     * Void 634: HP 100, Att 20, Str 30, Def 15, size 1, stab max 40 */
    [NPC_TZ_KIH] = {
        .max_hp = 100, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 40,
        .att_level = 20, .ranged_level = 30, .magic_level = 15,
        .def_level = 15, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_STAB,
        .size = 1, .movement_speed = 1, .prayer_drain = 10,
    },

    /* NPC_TZ_KEK: Lv 45 melee blob. Splits into 2 small on death.
     * Void 634: HP 200, Att 40, Str 60, Def 30, size 2, crush max 70 */
    [NPC_TZ_KEK] = {
        .max_hp = 200, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 70,
        .att_level = 40, .ranged_level = 60, .magic_level = 30,
        .def_level = 30, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 2, .movement_speed = 1,
    },

    /* NPC_TZ_KEK_SM: Lv 22 small blob (from split).
     * Void 634: HP 100, Att 20, Str 30, Def 15, size 1, crush max 40 */
    [NPC_TZ_KEK_SM] = {
        .max_hp = 100, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 40,
        .att_level = 20, .ranged_level = 30, .magic_level = 15,
        .def_level = 15, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 1, .movement_speed = 1,
    },

    /* NPC_TOK_XIL: Lv 90 ranged + melee (DUAL MODE).
     * Void 634: HP 400, Att 80, Str 120, Def 60, Rng 120, size 3
     * Current Fight Caves maxima are 130 for both melee and Ranged. */
    [NPC_TOK_XIL] = {
        .max_hp = 400, .attack_style = ATTACK_RANGED,
        .attack_speed = 4, .attack_range = 14,
        .melee_max_hit_tenths = 130, .ranged_max_hit_tenths = 130,
        .att_level = 80, .ranged_level = 120, .magic_level = 60,
        .def_level = 60, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 3, .movement_speed = 1,
    },

    /* NPC_YT_MEJKOT: Lv 180 melee + heals self/nearby NPCs with HP < 50% max.
     * Void 634: HP 800, Att 160, Str 240, Def 120, size 4
     * combat.toml: crush max 250. Heals 100 tenths (10 HP) as its attack. */
    [NPC_YT_MEJKOT] = {
        .max_hp = 800, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 250,
        .att_level = 160, .ranged_level = 240, .magic_level = 120,
        .def_level = 120, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 4, .movement_speed = 1, .heal_amount = 100,
    },

    /* NPC_KET_ZEK: Lv 360 magic + melee (DUAL MODE).
     * Void 634: HP 1600, Att 320, Str 480, Def 240, Mag 240, size 5
     * Current Fight Caves maxima are 550 melee and 520 Magic. The Magic
     * attack has +60 accuracy; the melee attack has no equipment bonus. */
    [NPC_KET_ZEK] = {
        .max_hp = 1600, .attack_style = ATTACK_MAGIC,
        .attack_speed = 4, .attack_range = 14,
        .melee_max_hit_tenths = 550, .magic_max_hit_tenths = 520,
        .att_level = 320, .ranged_level = 480, .magic_level = 240,
        .magic_attack_bonus = 60,
        .def_level = 240, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_STAB,
        .size = 5, .movement_speed = 1,
    },

    /* NPC_TZTOK_JAD: Lv 702 magic + ranged + melee.
     * Void 634: HP 2500, Att 640, Str 960, Def 480, Mag 480, Rng 960, size 5
     * combat.toml: melee stab max 970 (range 1), magic max 950 (range 14), ranged max 970
     * attack speed 8 (double normal), range 14. The Magic attack has +60
     * accuracy; the melee and Ranged attacks have no equipment bonus. */
    [NPC_TZTOK_JAD] = {
        .max_hp = 2500, .attack_style = ATTACK_MAGIC,
        .attack_speed = 8, .attack_range = 14,
        .melee_max_hit_tenths = 970, .ranged_max_hit_tenths = 970,
        .magic_max_hit_tenths = 950,
        .att_level = 640, .ranged_level = 960, .magic_level = 480,
        .magic_attack_bonus = 60,
        .def_level = 480, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_STAB,
        .size = 5, .movement_speed = 1,
    },

    /* NPC_YT_HURKOT: Lv 108 Jad healer. Heals Jad 50 tenths (5 HP) every 4 ticks within 5 tiles.
     * Void 634: HP 600, Att 140, Str 100, Def 60, size 1
     * combat.toml: crush max 140 */
    [NPC_YT_HURKOT] = {
        .max_hp = 600, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .melee_max_hit_tenths = 140,
        .att_level = 140, .ranged_level = 120, .magic_level = 120,
        .def_level = 60, .ranged_def_bonus = 100,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 1, .movement_speed = 1,
        .heal_amount = 50, .heal_interval = 4,
    },
};

int fc_npc_max_hit_tenths_for_style(const FcNpcStats* stats, int attack_style) {
    if (stats == NULL) return 0;
    switch (attack_style) {
        case ATTACK_MELEE: return stats->melee_max_hit_tenths;
        case ATTACK_RANGED: return stats->ranged_max_hit_tenths;
        case ATTACK_MAGIC: return stats->magic_max_hit_tenths;
        default: return 0;
    }
}

int fc_npc_max_hit_hp_for_style(const FcNpcStats* stats, int attack_style) {
    int max_hit_tenths = fc_npc_max_hit_tenths_for_style(stats, attack_style);
    if (max_hit_tenths < 0 || max_hit_tenths % 10 != 0) return 0;
    return max_hit_tenths / 10;
}

int fc_npc_stats_valid(const FcNpcStats* stats) {
    if (stats == NULL) return 0;
    const int maxima[] = {
        stats->melee_max_hit_tenths,
        stats->ranged_max_hit_tenths,
        stats->magic_max_hit_tenths,
    };
    for (int i = 0; i < 3; i++) {
        if (maxima[i] < 0 || maxima[i] % 10 != 0) return 0;
    }
    return 1;
}

const FcNpcStats* fc_npc_get_stats(int npc_type) {
    if (npc_type < 0 || npc_type >= NPC_TYPE_COUNT) return &NPC_STATS[0];
    return &NPC_STATS[npc_type];
}

static int npc_attack_level_for_style(const FcNpcStats* stats,
                                      int attack_style) {
    switch (attack_style) {
        case ATTACK_MELEE: return stats->att_level;
        case ATTACK_RANGED: return stats->ranged_level;
        case ATTACK_MAGIC: return stats->magic_level;
        default: return 0;
    }
}

static int npc_attack_bonus_for_style(const FcNpcStats* stats,
                                      int attack_style) {
    switch (attack_style) {
        case ATTACK_MELEE: return stats->melee_attack_bonus;
        case ATTACK_RANGED: return stats->ranged_attack_bonus;
        case ATTACK_MAGIC: return stats->magic_attack_bonus;
        default: return 0;
    }
}

static FcAttackType npc_attack_type_for_style(const FcNpcStats* stats,
                                              int attack_style) {
    switch (attack_style) {
        case ATTACK_MELEE: return (FcAttackType)stats->melee_attack_type;
        case ATTACK_RANGED: return FC_ATTACK_TYPE_RANGED;
        case ATTACK_MAGIC: return FC_ATTACK_TYPE_MAGIC;
        default: return FC_ATTACK_TYPE_NONE;
    }
}

/* ======================================================================== */
/* Spawn                                                                     */
/* ======================================================================== */

void fc_npc_spawn(FcNpc* npc, int npc_type, int x, int y, int spawn_index) {
    const FcNpcStats* stats = fc_npc_get_stats(npc_type);

    npc->active = 1;
    npc->npc_type = npc_type;
    npc->spawn_index = spawn_index;
    npc->x = x;
    npc->y = y;
    npc->size = stats->size;
    npc->current_hp = stats->max_hp;
    npc->max_hp = stats->max_hp;
    npc->is_dead = 0;
    npc->attack_style = stats->attack_style;
    npc->attack_timer = stats->attack_speed;  /* first attack after full cooldown */
    npc->attack_speed = stats->attack_speed;
    npc->attack_range = stats->attack_range;
    npc->movement_speed = stats->movement_speed;
    npc->heal_timer = stats->heal_interval;  /* start at full cooldown */
    npc->heal_amount = stats->heal_amount;
    npc->healer_distracted = 0;
    npc->heal_target_idx = -1;
    npc->is_respawned_jad_healer = 0;
    npc->damage_taken_this_tick = 0;
    npc->prayer_drain_dealt_this_tick = 0;
    npc->healing_received_this_tick = 0;
    npc->healing_given_this_tick = 0;
    npc->healed_by_mejkot_this_tick = 0;
    npc->healed_by_hurkot_this_tick = 0;
    npc->healed_self_this_tick = 0;
    npc->died_this_tick = 0;
    npc->num_pending_hits = 0;
}

static void build_npc_movement_occupancy(
    const FcState* state,
    int npc_idx,
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    fc_build_occupancy(state, occupied, npc_idx, 0);
}

static int npc_dynamic_step_toward(FcState* state, int npc_idx,
                                   int target_x, int target_y) {
    FcNpc* npc = &state->npcs[npc_idx];
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    build_npc_movement_occupancy(state, npc_idx, occupied);
    return fc_npc_step_toward_sized_dynamic(&npc->x, &npc->y,
                                            target_x, target_y, npc->size,
                                            state->walkable,
                                            state->movement_flags,
                                            occupied);
}

static int min_i(int a, int b) {
    return a < b ? a : b;
}

static int max_i(int a, int b) {
    return a > b ? a : b;
}

static void npc_naive_player_chase_destination(const FcNpc* npc,
                                               const FcPlayer* player,
                                               int* out_x, int* out_y) {
    int source_width = npc->size;
    int source_length = npc->size;
    int target_width = 1;
    int target_length = 1;
    int diagonal = (npc->x - player->x) + (npc->y - player->y);
    int anti = (npc->x - player->x) - (npc->y - player->y);
    int south_west_clockwise = anti < 0;
    int north_west_clockwise =
        diagonal >= (target_length - 1) - (source_width - 1);
    int north_east_clockwise = anti > source_width - source_length;
    int south_east_clockwise =
        diagonal <= (target_width - 1) - (source_length - 1);

    if (south_west_clockwise && !north_west_clockwise) {
        int off_y;
        if (diagonal >= -source_width) {
            off_y = min_i(diagonal + source_width, target_length - 1);
        } else if (anti > -source_width) {
            off_y = -(source_width + anti);
        } else {
            off_y = 0;
        }
        *out_x = player->x - source_width;
        *out_y = player->y + off_y;
    } else if (north_west_clockwise && !north_east_clockwise) {
        int off_x;
        if (anti >= -target_length) {
            off_x = min_i(anti + target_length, target_width - 1);
        } else if (diagonal < target_length) {
            off_x = max_i(diagonal - target_length, -(source_width - 1));
        } else {
            off_x = 0;
        }
        *out_x = player->x + off_x;
        *out_y = player->y + target_length;
    } else if (north_east_clockwise && !south_east_clockwise) {
        int off_y;
        if (anti <= target_width) {
            off_y = target_length - anti;
        } else if (diagonal < target_width) {
            off_y = max_i(diagonal - target_width, -(source_length - 1));
        } else {
            off_y = 0;
        }
        *out_x = player->x + target_width;
        *out_y = player->y + off_y;
    } else {
        int off_x;
        if (diagonal > -source_length) {
            off_x = min_i(diagonal + source_length, target_width - 1);
        } else if (anti < source_length) {
            off_x = max_i(anti - source_length, -(source_length - 1));
        } else {
            off_x = 0;
        }
        *out_x = player->x + off_x;
        *out_y = player->y - source_length;
    }
}

int fc_npc_position_can_attack_player(const FcState* state, const FcNpc* npc,
                                      int candidate_x, int candidate_y) {
    if (!state || !npc || !npc->active || npc->is_dead) return 0;

    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    const FcPlayer* p = &state->player;

    int can_melee = fc_npc_can_melee_player(p->x, p->y,
                                            candidate_x, candidate_y,
                                            npc->size, state->walkable,
                                            state->movement_flags);
    if (can_melee &&
        (stats->melee_max_hit_tenths > 0 ||
         npc->attack_style == ATTACK_MELEE)) {
        return 1;
    }

    int distance = fc_distance_between_areas(
        p->x, p->y, 1, candidate_x, candidate_y, npc->size);
    if (npc->attack_style != ATTACK_MELEE && distance > 0 &&
        distance <= npc->attack_range) {
        return fc_has_los_between_areas(
            candidate_x, candidate_y, npc->size,
            p->x, p->y, 1, state->los_flags);
    }

    return 0;
}

static int npc_dynamic_step_toward_player_bounds(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    int target_x;
    int target_y;

    npc_naive_player_chase_destination(npc, &state->player,
                                       &target_x, &target_y);
    if (target_x == npc->x && target_y == npc->y) return 0;

    return npc_dynamic_step_toward(state, npc_idx, target_x, target_y);
}

/* ======================================================================== */
/* Tz-Kek: split on death — spawn 2 small Tz-Kek                            */
/* ======================================================================== */

void fc_npc_tz_kek_split(FcState* state, int dead_x, int dead_y) {
    /* Spawn 2 NPC_TZ_KEK_SM at/near the death position.
     * Do NOT increment npcs_remaining — the parent Tz-Kek was pre-counted as 2
     * at wave spawn time. These children inherit that count and decrement normally
     * when they die. (Matches RSPS: parent not in npcDespawn list, children are.) */
    const FcNpcStats* child_stats = fc_npc_get_stats(NPC_TZ_KEK_SM);
    for (int spawned = 0; spawned < 2; spawned++) {
        int sx = dead_x + (spawned == 0 ? 0 : 1);
        int sy = dead_y;
        /* Clamp to arena */
        if (sx >= FC_ARENA_WIDTH - 1) sx = dead_x - 1;
        if (sx < 1) sx = 1;

        if (!fc_spawn_find_available_footprint(
                state, sx, sy, child_stats->size, FC_ARENA_WIDTH - 1,
                &sx, &sy)) {
            break;
        }
        if (fc_spawn_npc_first_free(state, NPC_TZ_KEK_SM, sx, sy) < 0) {
            break;
        }
        /* No npcs_remaining++ — already pre-counted */
    }
}

/* ======================================================================== */
/* Jad direct attack selection                                               */
/* ======================================================================== */

static void record_npc_attack(FcState* state, const FcNpc* npc, int npc_idx,
                              int attack_style, int hit_delay_ticks,
                              int prayer_lock_tick, int hit_queued) {
    FcRenderEvents* events = &state->render_events;
    if (events->npc_attack_count >= FC_MAX_RENDER_NPC_ATTACKS) return;

    FcRenderNpcAttack* attack =
        &events->npc_attacks[events->npc_attack_count++];
    attack->npc_slot = npc_idx;
    attack->npc_type = npc->npc_type;
    attack->attack_style = attack_style;
    attack->source_x = npc->x;
    attack->source_y = npc->y;
    attack->source_size = npc->size;
    attack->target_x = state->player.x;
    attack->target_y = state->player.y;
    attack->hit_delay_ticks = hit_delay_ticks;
    attack->prayer_lock_tick = prayer_lock_tick;
    attack->hit_queued = hit_queued;
}

static void launch_npc_attack(FcState* state, FcNpc* npc, int npc_idx,
                              int attack_style, int hit_delay_ticks,
                              int prayer_drain, int prayer_lock_delay_ticks) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    FcPlayer* player = &state->player;
    int max_hit_hp = fc_npc_max_hit_hp_for_style(stats, attack_style);
    int attack_level = npc_attack_level_for_style(stats, attack_style);
    int attack_bonus = npc_attack_bonus_for_style(stats, attack_style);
    int attack_roll = fc_npc_attack_roll(attack_level, attack_bonus);
    FcAttackType attack_type =
        npc_attack_type_for_style(stats, attack_style);
    int defence_roll = fc_player_def_roll(player, attack_type);
    float hit_chance = fc_hit_chance(attack_roll, defence_roll);
    int hit = fc_rng_float(state) < hit_chance;
    int damage = hit ? fc_roll_npc_damage_tenths(state, max_hit_hp) : 0;

    int hit_queued = fc_queue_pending_hit(
        player->pending_hits, &player->num_pending_hits, FC_MAX_PENDING_HITS,
        damage, hit_delay_ticks, attack_style, npc_idx, prayer_drain);
    int prayer_lock_tick = -1;
    if (hit_queued) {
        FcPendingHit* queued =
            &player->pending_hits[player->num_pending_hits - 1];
        if (prayer_lock_delay_ticks > 0) {
            queued->prayer_snapshot = -1;
            queued->prayer_lock_tick =
                state->tick + prayer_lock_delay_ticks;
            prayer_lock_tick = queued->prayer_lock_tick;
        } else {
            queued->prayer_snapshot = player->prayer_at_tick_start;
        }
    }

    record_npc_attack(state, npc, npc_idx, attack_style, hit_delay_ticks,
                      prayer_lock_tick, hit_queued);
    npc->attack_timer = npc->attack_speed;
}

static void jad_attack(FcState* state, FcNpc* npc, int npc_idx) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    FcPlayer* p = &state->player;
    int dist = fc_distance_to_npc(p->x, p->y, npc);
    int can_melee = fc_npc_can_melee_player(p->x, p->y, npc->x, npc->y,
                                            npc->size, state->walkable,
                                            state->movement_flags);

    if (npc->attack_timer > 0) return;

    int use_style = ATTACK_NONE;
    int in_range = 0;

    int can_use_distance_styles =
        dist > 0 && dist <= npc->attack_range &&
        fc_has_los_between_areas(
            npc->x, npc->y, npc->size,
            p->x, p->y, 1, state->los_flags);

    if (can_melee && stats->melee_max_hit_tenths > 0) {
        /* In melee range Jad can still choose Magic or Ranged. All three
         * configured attacks have equal selection weight. */
        int choice = can_use_distance_styles ? fc_rng_int(state, 3) : 0;
        if (choice == 0) {
            use_style = ATTACK_MELEE;
        } else if (choice == 1) {
            use_style = ATTACK_MAGIC;
        } else {
            use_style = ATTACK_RANGED;
        }
        in_range = 1;
    } else if (can_use_distance_styles) {
        use_style = (fc_rng_int(state, 2) == 0) ? ATTACK_MAGIC : ATTACK_RANGED;
        in_range = 1;
    }

    if (!in_range) return;

    int delay = fc_npc_hit_delay(npc->npc_type, use_style, dist);
    if (use_style != ATTACK_MELEE && delay < 3) delay = 3;
    int prayer_lock_delay = use_style == ATTACK_MELEE ? 0 : 2;
    launch_npc_attack(state, npc, npc_idx, use_style, delay, 0,
                      prayer_lock_delay);
}

/* ======================================================================== */
/* Yt-MejKot: heal nearby NPCs                                              */
/* ======================================================================== */

static int apply_npc_heal(FcState* state, FcNpc* source, FcNpc* target,
                          int amount) {
    int before = target->current_hp;
    target->current_hp += amount;
    if (target->current_hp > target->max_hp) {
        target->current_hp = target->max_hp;
    }
    amount = target->current_hp - before;
    if (amount <= 0) return 0;
    source->healing_given_this_tick += amount;
    target->healing_received_this_tick += amount;
    if (source->npc_type == NPC_YT_MEJKOT) {
        target->healed_by_mejkot_this_tick = 1;
    } else if (source->npc_type == NPC_YT_HURKOT) {
        target->healed_by_hurkot_this_tick = 1;
    }
    if (source == target) target->healed_self_this_tick = 1;

    state->npc_heal_procs_this_tick++;
    state->npc_heal_amount_this_tick += amount;
    if (source->npc_type == NPC_YT_MEJKOT) {
        state->mejkot_heal_amount_this_tick += amount;
    }
    if (target->npc_type == NPC_TZTOK_JAD) {
        state->jad_heal_amount_this_tick += amount;
    }
    return amount;
}

static int npc_anchor_distance(const FcNpc* a, const FcNpc* b) {
    return fc_distance_between_areas(a->x, a->y, 1, b->x, b->y, 1);
}

static FcNpc* yt_mejkot_heal_target(FcState* state, FcNpc* npc) {
    if (npc->current_hp < npc->max_hp / 2) return npc;

    FcNpc* best = NULL;
    int best_distance = FC_ARENA_WIDTH;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* target = &state->npcs[i];
        if (target == npc || !target->active || target->is_dead) continue;
        if (target->current_hp >= target->max_hp / 2) continue;

        int distance = npc_anchor_distance(npc, target);
        if (distance > 8) continue;
        if (!best || distance < best_distance ||
            (distance == best_distance && target->spawn_index < best->spawn_index)) {
            best = target;
            best_distance = distance;
        }
    }
    return best;
}

static int yt_mejkot_try_heal(FcState* state, FcNpc* npc) {
    if (npc->attack_timer > 0) return 0;
    if (!fc_npc_can_melee_player(state->player.x, state->player.y,
                                 npc->x, npc->y, npc->size,
                                 state->walkable, state->movement_flags)) {
        return 0;
    }

    FcNpc* target = yt_mejkot_heal_target(state, npc);
    if (!target) return 0;

    apply_npc_heal(state, npc, target, npc->heal_amount);
    npc->heal_target_idx = (int)(target - state->npcs);
    npc->attack_timer = npc->attack_speed;
    return 1;
}

/* ======================================================================== */
/* Yt-HurKot: heal Jad until permanently tagged onto the player               */
/* ======================================================================== */

#define FC_HURKOT_HEAL_RANGE 5
static void npc_generic_attack(FcState* state, FcNpc* npc, int npc_idx);

static int find_active_jad(const FcState* state) {
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* npc = &state->npcs[i];
        if (npc->active && !npc->is_dead && npc->npc_type == NPC_TZTOK_JAD) {
            return i;
        }
    }
    return -1;
}

static void yt_hurkot_heal_cycle(FcState* state, FcNpc* npc, FcNpc* jad) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    if (npc->heal_timer > 1) {
        npc->heal_timer--;
        return;
    }

    npc->heal_timer = stats->heal_interval;
    if (npc_anchor_distance(npc, jad) > FC_HURKOT_HEAL_RANGE ||
        jad->current_hp >= jad->max_hp) {
        return;
    }

    if (apply_npc_heal(state, npc, jad, npc->heal_amount) > 0) {
        state->jad_heal_procs_this_tick++;
    }
}

static void yt_hurkot_tick(FcState* state, FcNpc* npc, int npc_idx) {
    /* Once tagged, a healer permanently pursues the player using the same
     * local, non-routing movement as every other NPC. It no longer heals Jad. */
    if (npc->healer_distracted) {
        npc->heal_target_idx = -1;
        if (!fc_npc_position_can_attack_player(state, npc, npc->x, npc->y)) {
            for (int step = 0; step < npc->movement_speed; step++) {
                if (!npc_dynamic_step_toward_player_bounds(state, npc_idx)) break;
            }
        }
        npc_generic_attack(state, npc, npc_idx);
        return;
    }

    int jad_idx = find_active_jad(state);
    FcNpc* jad = jad_idx >= 0 ? &state->npcs[jad_idx] : NULL;
    npc->heal_target_idx = jad_idx;
    if (jad) yt_hurkot_heal_cycle(state, npc, jad);

    if (jad && npc_anchor_distance(npc, jad) > FC_HURKOT_HEAL_RANGE) {
        npc_dynamic_step_toward(state, npc_idx, jad->x, jad->y);
    }
}

/* ======================================================================== */
/* Generic NPC attack (melee/ranged/magic, non-Jad)                          */
/* ======================================================================== */

/*
 * Tok-Xil switches to its weaker melee attack at contact. Ket-Zek keeps both
 * its Magic and Melee attacks valid at contact and samples between them.
 */
static void npc_generic_attack(FcState* state, FcNpc* npc, int npc_idx) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    FcPlayer* p = &state->player;
    int dist = fc_distance_to_npc(p->x, p->y, npc);
    int can_melee = fc_npc_can_melee_player(p->x, p->y, npc->x, npc->y,
                                            npc->size, state->walkable,
                                            state->movement_flags);

    if (npc->attack_timer > 0) return;

    /* Determine attack style and max hit based on distance */
    int use_style = npc->attack_style;  /* primary style */
    int in_range = 0;
    int primary_in_range =
        npc->attack_style != ATTACK_MELEE &&
        dist > 0 && dist <= npc->attack_range &&
        fc_has_los_between_areas(
            npc->x, npc->y, npc->size,
            p->x, p->y, 1, state->los_flags);

    if (can_melee && stats->melee_max_hit_tenths > 0) {
        if (npc->npc_type == NPC_KET_ZEK && primary_in_range &&
            fc_rng_int(state, 2) == 0) {
            use_style = npc->attack_style;
        } else {
            use_style = ATTACK_MELEE;
        }
        in_range = 1;
    } else if (can_melee && npc->attack_style == ATTACK_MELEE) {
        /* Pure melee NPC, in range */
        in_range = 1;
    } else if (primary_in_range) {
        in_range = 1;
    }

    if (!in_range) return;

    int delay = fc_npc_hit_delay(npc->npc_type, use_style, dist);
    launch_npc_attack(state, npc, npc_idx, use_style, delay,
                      stats->prayer_drain, 0);
}

static int npc_has_attack_position(FcState* state, FcNpc* npc) {
    return fc_npc_position_can_attack_player(state, npc, npc->x, npc->y);
}

/* ======================================================================== */
/* NPC AI tick — type dispatch                                               */
/* ======================================================================== */

void fc_npc_tick(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    if (!npc->active || npc->is_dead) return;

    if (npc->npc_type != NPC_YT_HURKOT) npc->heal_target_idx = -1;

    /* Decrement attack timer */
    if (npc->attack_timer > 0) npc->attack_timer--;

    /* --- Type-specific pre-attack behavior --- */

    /* Yt-HurKot either heals/follows Jad or permanently targets the player. */
    if (npc->npc_type == NPC_YT_HURKOT) {
        yt_hurkot_tick(state, npc, npc_idx);
        return;
    }

    /* Jad: move into range, then choose its attack style when the hit is queued */
    if (npc->npc_type == NPC_TZTOK_JAD) {
        if (!npc_has_attack_position(state, npc)) {
            for (int step = 0; step < npc->movement_speed; step++) {
                if (!npc_dynamic_step_toward_player_bounds(state, npc_idx)) break;
            }
        }
        jad_attack(state, npc, npc_idx);
        return;
    }

    /* --- Generic movement + attack for all other types --- */

    /* Movement: keep walking until this tile can actually attack. */
    if (!npc_has_attack_position(state, npc)) {
        for (int step = 0; step < npc->movement_speed; step++) {
            if (!npc_dynamic_step_toward_player_bounds(state, npc_idx)) break;
        }
    }

    /* A MejKot heal is an attack-cycle choice, not a parallel action. */
    if (npc->npc_type == NPC_YT_MEJKOT && yt_mejkot_try_heal(state, npc)) {
        return;
    }

    npc_generic_attack(state, npc, npc_idx);
}

#undef FC_HURKOT_HEAL_RANGE

/* Pathfinding */
#include <stddef.h>
#include <string.h>

/*
 * fc_pathfinding.c — Grid movement, footprint checks, and LOS for Fight Caves.
 *
 * Key design:
 *   - NPCs have sizes 1-5 (Jad and Ket-Zek are 5x5!). Movement must check
 *     the entire footprint at the destination tile.
 *   - Projectile LOS uses its own directional collision flags.
 *   - Movement uses whole-tile blocking plus directional wall flags from the
 *     authoritative cache data, never the visual mesh.
 */

/* ======================================================================== */
/* Tile queries                                                              */
/* ======================================================================== */

int fc_tile_walkable(int x, int y,
                     const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (x < 0 || x >= FC_ARENA_WIDTH || y < 0 || y >= FC_ARENA_HEIGHT) return 0;
    return walkable[x][y];
}

int fc_footprint_walkable(int x, int y, int size,
                          const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    /* Check all tiles in the NPC's [x..x+size-1, y..y+size-1] footprint */
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            if (!fc_tile_walkable(x + dx, y + dy, walkable)) return 0;
        }
    }
    return 1;
}

/* ======================================================================== */
/* Dynamic occupancy                                                         */
/* ======================================================================== */

void fc_clear_occupancy(uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    for (int x = 0; x < FC_ARENA_WIDTH; x++) {
        for (int y = 0; y < FC_ARENA_HEIGHT; y++) {
            occupied[x][y] = 0;
        }
    }
}

void fc_mark_footprint_occupied(uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                                int x, int y, int size) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int tx = x + dx;
            int ty = y + dy;
            if (tx >= 0 && tx < FC_ARENA_WIDTH &&
                ty >= 0 && ty < FC_ARENA_HEIGHT) {
                occupied[tx][ty] = 1;
            }
        }
    }
}

void fc_build_occupancy(const FcState* state,
                        uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                        int ignore_npc_idx,
                        int ignore_player) {
    fc_clear_occupancy(occupied);

    if (!ignore_player) {
        fc_mark_footprint_occupied(occupied, state->player.x, state->player.y, 1);
    }

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* npc = &state->npcs[i];
        if (i == ignore_npc_idx) continue;
        if (!npc->active || npc->is_dead) continue;
        fc_mark_footprint_occupied(occupied, npc->x, npc->y, npc->size);
    }
}

int fc_footprint_available_dynamic(
    int x, int y, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (size <= 0) return 0;
    if (!fc_footprint_walkable(x, y, size, walkable)) return 0;

    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            if (occupied[x + dx][y + dy]) return 0;
        }
    }
    return 1;
}

/* Low-byte equivalents of the composite clipping masks used by the native
 * client route finder. A non-walkable or occupied tile is the local equivalent
 * of its whole-tile LOC blocker. */
#define FC_BLOCK_WEST FC_MOVE_WALL_EAST
#define FC_BLOCK_EAST FC_MOVE_WALL_WEST
#define FC_BLOCK_SOUTH FC_MOVE_WALL_NORTH
#define FC_BLOCK_NORTH FC_MOVE_WALL_SOUTH
#define FC_BLOCK_SOUTH_WEST \
    (FC_MOVE_WALL_NORTH | FC_MOVE_WALL_NORTH_EAST | FC_MOVE_WALL_EAST)
#define FC_BLOCK_SOUTH_EAST \
    (FC_MOVE_WALL_NORTH_WEST | FC_MOVE_WALL_NORTH | FC_MOVE_WALL_WEST)
#define FC_BLOCK_NORTH_WEST \
    (FC_MOVE_WALL_EAST | FC_MOVE_WALL_SOUTH_EAST | FC_MOVE_WALL_SOUTH)
#define FC_BLOCK_NORTH_EAST \
    (FC_MOVE_WALL_SOUTH | FC_MOVE_WALL_SOUTH_WEST | FC_MOVE_WALL_WEST)
#define FC_BLOCK_NORTH_AND_SOUTH_EAST \
    (FC_MOVE_WALL_NORTH | FC_MOVE_WALL_NORTH_EAST | FC_MOVE_WALL_EAST | \
     FC_MOVE_WALL_SOUTH_EAST | FC_MOVE_WALL_SOUTH)
#define FC_BLOCK_NORTH_AND_SOUTH_WEST \
    (FC_MOVE_WALL_NORTH_WEST | FC_MOVE_WALL_NORTH | FC_MOVE_WALL_SOUTH | \
     FC_MOVE_WALL_SOUTH_WEST | FC_MOVE_WALL_WEST)
#define FC_BLOCK_NORTH_EAST_AND_WEST \
    (FC_MOVE_WALL_NORTH_WEST | FC_MOVE_WALL_NORTH | \
     FC_MOVE_WALL_NORTH_EAST | FC_MOVE_WALL_EAST | FC_MOVE_WALL_WEST)
#define FC_BLOCK_SOUTH_EAST_AND_WEST \
    (FC_MOVE_WALL_EAST | FC_MOVE_WALL_SOUTH_EAST | FC_MOVE_WALL_SOUTH | \
     FC_MOVE_WALL_SOUTH_WEST | FC_MOVE_WALL_WEST)

static int fc_step_tile_blocked(
    int x, int y, uint8_t mask,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (x < 0 || x >= FC_ARENA_WIDTH || y < 0 || y >= FC_ARENA_HEIGHT)
        return 1;
    return !walkable[x][y] || (movement_flags[x][y] & mask) != 0 ||
           (occupied && occupied[x][y]);
}

static int fc_footprint_step_valid(
    int x, int y, int dx, int dy, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (size <= 0 || dx < -1 || dx > 1 || dy < -1 || dy > 1 ||
        (dx == 0 && dy == 0)) return 0;

#define BLOCKED(tx, ty, mask) \
    fc_step_tile_blocked((tx), (ty), (uint8_t)(mask), walkable, \
                         movement_flags, occupied)

    if (dx == 0 && dy == -1) {
        if (size == 1) return !BLOCKED(x, y - 1, FC_BLOCK_SOUTH);
        if (BLOCKED(x, y - 1, FC_BLOCK_SOUTH_WEST) ||
            BLOCKED(x + size - 1, y - 1, FC_BLOCK_SOUTH_EAST)) return 0;
        for (int i = 1; i < size - 1; i++)
            if (BLOCKED(x + i, y - 1, FC_BLOCK_NORTH_EAST_AND_WEST)) return 0;
        return 1;
    }
    if (dx == 0 && dy == 1) {
        if (size == 1) return !BLOCKED(x, y + 1, FC_BLOCK_NORTH);
        if (BLOCKED(x, y + size, FC_BLOCK_NORTH_WEST) ||
            BLOCKED(x + size - 1, y + size, FC_BLOCK_NORTH_EAST)) return 0;
        for (int i = 1; i < size - 1; i++)
            if (BLOCKED(x + i, y + size, FC_BLOCK_SOUTH_EAST_AND_WEST)) return 0;
        return 1;
    }
    if (dx == -1 && dy == 0) {
        if (size == 1) return !BLOCKED(x - 1, y, FC_BLOCK_WEST);
        if (BLOCKED(x - 1, y, FC_BLOCK_SOUTH_WEST) ||
            BLOCKED(x - 1, y + size - 1, FC_BLOCK_NORTH_WEST)) return 0;
        for (int i = 1; i < size - 1; i++)
            if (BLOCKED(x - 1, y + i, FC_BLOCK_NORTH_AND_SOUTH_EAST)) return 0;
        return 1;
    }
    if (dx == 1 && dy == 0) {
        if (size == 1) return !BLOCKED(x + 1, y, FC_BLOCK_EAST);
        if (BLOCKED(x + size, y, FC_BLOCK_SOUTH_EAST) ||
            BLOCKED(x + size, y + size - 1, FC_BLOCK_NORTH_EAST)) return 0;
        for (int i = 1; i < size - 1; i++)
            if (BLOCKED(x + size, y + i, FC_BLOCK_NORTH_AND_SOUTH_WEST)) return 0;
        return 1;
    }
    if (dx == -1 && dy == -1) {
        if (size == 1)
            return !BLOCKED(x - 1, y - 1, FC_BLOCK_SOUTH_WEST) &&
                   !BLOCKED(x - 1, y, FC_BLOCK_WEST) &&
                   !BLOCKED(x, y - 1, FC_BLOCK_SOUTH);
        if (BLOCKED(x - 1, y - 1, FC_BLOCK_SOUTH_WEST)) return 0;
        for (int i = 1; i < size; i++) {
            if (BLOCKED(x - 1, y + i - 1, FC_BLOCK_NORTH_AND_SOUTH_EAST) ||
                BLOCKED(x + i - 1, y - 1, FC_BLOCK_NORTH_EAST_AND_WEST)) return 0;
        }
        return 1;
    }
    if (dx == -1 && dy == 1) {
        if (size == 1)
            return !BLOCKED(x - 1, y + 1, FC_BLOCK_NORTH_WEST) &&
                   !BLOCKED(x - 1, y, FC_BLOCK_WEST) &&
                   !BLOCKED(x, y + 1, FC_BLOCK_NORTH);
        if (BLOCKED(x - 1, y + size, FC_BLOCK_NORTH_WEST)) return 0;
        for (int i = 1; i < size; i++) {
            if (BLOCKED(x - 1, y + i, FC_BLOCK_NORTH_AND_SOUTH_EAST) ||
                BLOCKED(x + i - 1, y + size, FC_BLOCK_SOUTH_EAST_AND_WEST)) return 0;
        }
        return 1;
    }
    if (dx == 1 && dy == -1) {
        if (size == 1)
            return !BLOCKED(x + 1, y - 1, FC_BLOCK_SOUTH_EAST) &&
                   !BLOCKED(x + 1, y, FC_BLOCK_EAST) &&
                   !BLOCKED(x, y - 1, FC_BLOCK_SOUTH);
        if (BLOCKED(x + size, y - 1, FC_BLOCK_SOUTH_EAST)) return 0;
        for (int i = 1; i < size; i++) {
            if (BLOCKED(x + size, y + i - 1, FC_BLOCK_NORTH_AND_SOUTH_WEST) ||
                BLOCKED(x + i, y - 1, FC_BLOCK_NORTH_EAST_AND_WEST)) return 0;
        }
        return 1;
    }
    if (dx == 1 && dy == 1) {
        if (size == 1)
            return !BLOCKED(x + 1, y + 1, FC_BLOCK_NORTH_EAST) &&
                   !BLOCKED(x + 1, y, FC_BLOCK_EAST) &&
                   !BLOCKED(x, y + 1, FC_BLOCK_NORTH);
        if (BLOCKED(x + size, y + size, FC_BLOCK_NORTH_EAST)) return 0;
        for (int i = 1; i < size; i++) {
            if (BLOCKED(x + i, y + size, FC_BLOCK_SOUTH_EAST_AND_WEST) ||
                BLOCKED(x + size, y + i, FC_BLOCK_NORTH_AND_SOUTH_WEST)) return 0;
        }
        return 1;
    }
#undef BLOCKED
    return 0;
}

int fc_footprint_step_walkable(
    int x, int y, int dx, int dy, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    return fc_footprint_step_valid(x, y, dx, dy, size, walkable,
                                   movement_flags, NULL);
}

int fc_footprint_step_available_dynamic(
    int x, int y, int dx, int dy, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    return fc_footprint_step_valid(x, y, dx, dy, size, walkable,
                                   movement_flags, occupied);
}

/* ======================================================================== */
/* Size-1 movement (player, small NPCs)                                      */
/* ======================================================================== */

int fc_move_toward_traced(
    int* x, int* y, int dx, int dy, int max_steps,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int* step_x, int* step_y, int step_capacity) {
    int tx = *x + dx;
    int ty = *y + dy;
    int steps = 0;

    for (int step = 0; step < max_steps; step++) {
        if (*x == tx && *y == ty) break;

        int sx = 0, sy = 0;
        if (tx > *x) sx = 1; else if (tx < *x) sx = -1;
        if (ty > *y) sy = 1; else if (ty < *y) sy = -1;

        /* Try diagonal first, then x-only, then y-only */
        int moved = 0;
        if (sx != 0 && sy != 0 &&
            fc_footprint_step_walkable(
                *x, *y, sx, sy, 1, walkable, movement_flags)) {
            *x += sx; *y += sy; moved = 1;
        } else if (sx != 0 && fc_footprint_step_walkable(
                       *x, *y, sx, 0, 1, walkable, movement_flags)) {
            *x += sx; moved = 1;
        } else if (sy != 0 && fc_footprint_step_walkable(
                       *x, *y, 0, sy, 1, walkable, movement_flags)) {
            *y += sy; moved = 1;
        } else {
            break;
        }
        if (moved) {
            if (steps < step_capacity && step_x && step_y) {
                step_x[steps] = *x;
                step_y[steps] = *y;
            }
            steps++;
        }
    }
    return steps;
}

int fc_move_toward(int* x, int* y, int dx, int dy, int max_steps,
                   const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                   const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    return fc_move_toward_traced(x, y, dx, dy, max_steps, walkable,
                                 movement_flags, NULL, NULL, 0);
}

/* ======================================================================== */
/* Size-aware NPC movement                                                   */
/* ======================================================================== */

int fc_npc_step_toward_sized_dynamic(
    int* x, int* y, int target_x, int target_y, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    int dx = 0, dy = 0;
    if (target_x > *x) dx = 1; else if (target_x < *x) dx = -1;
    if (target_y > *y) dy = 1; else if (target_y < *y) dy = -1;

    if (dx == 0 && dy == 0) return 0;

    if (dx != 0 && dy != 0 &&
        fc_footprint_step_available_dynamic(
            *x, *y, dx, dy, size,
            walkable, movement_flags, occupied)) {
        *x += dx; *y += dy; return 1;
    }
    if (dx != 0 &&
        fc_footprint_step_available_dynamic(
            *x, *y, dx, 0, size,
            walkable, movement_flags, occupied)) {
        *x += dx; return 1;
    }
    if (dy != 0 &&
        fc_footprint_step_available_dynamic(
            *x, *y, 0, dy, size,
            walkable, movement_flags, occupied)) {
        *y += dy; return 1;
    }
    return 0;
}

/* ======================================================================== */
/* Line of sight — directional projectile collision                         */
/* ======================================================================== */

int fc_distance_between_areas(int src_x, int src_y, int src_size,
                              int dst_x, int dst_y, int dst_size) {
    if (src_size <= 0 || dst_size <= 0) return 0;
    int src_max_x = src_x + src_size - 1;
    int src_max_y = src_y + src_size - 1;
    int dst_max_x = dst_x + dst_size - 1;
    int dst_max_y = dst_y + dst_size - 1;
    int dx = src_max_x < dst_x ? dst_x - src_max_x :
             dst_max_x < src_x ? src_x - dst_max_x : 0;
    int dy = src_max_y < dst_y ? dst_y - src_max_y :
             dst_max_y < src_y ? src_y - dst_max_y : 0;
    return dx > dy ? dx : dy;
}

static int fc_has_line_of_sight(
    int x0, int y0, int x1, int y1,
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (x0 < 0 || y0 < 0 || x0 >= FC_ARENA_WIDTH || y0 >= FC_ARENA_HEIGHT ||
        x1 < 0 || y1 < 0 || x1 >= FC_ARENA_WIDTH || y1 >= FC_ARENA_HEIGHT) {
        return 0;
    }
    if (x0 == x1 && y0 == y1) return 1;
    if (los_flags[x0][y0] & FC_LOS_FULL) return 0;

    int dx = x1 - x0;
    int dy = y1 - y0;
    int dx_abs = dx < 0 ? -dx : dx;
    int dy_abs = dy < 0 ? -dy : dy;
    uint8_t x_flags = FC_LOS_FULL | (dx < 0 ? FC_LOS_EAST : FC_LOS_WEST);
    uint8_t y_flags = FC_LOS_FULL | (dy < 0 ? FC_LOS_NORTH : FC_LOS_SOUTH);

    /* Fixed-point major-axis traversal matches directional OSRS tile LOS.
     * Each crossed destination tile supplies the boundary flag to test. */
    if (dx_abs > dy_abs) {
        int x = x0;
        int y_big = (y0 << 16) + 0x8000;
        int slope = (dy * 65536) / dx_abs;
        if (dy < 0) y_big--;
        int direction = dx < 0 ? -1 : 1;

        while (x != x1) {
            x += direction;
            int y = y_big >> 16;
            uint8_t step_x_flags = x_flags;
            if (x == x1 && y == y1) step_x_flags &= (uint8_t)~FC_LOS_FULL;
            if (los_flags[x][y] & step_x_flags) return 0;
            y_big += slope;
            int next_y = y_big >> 16;
            uint8_t step_y_flags = y_flags;
            if (x == x1 && next_y == y1) step_y_flags &= (uint8_t)~FC_LOS_FULL;
            if (next_y != y && (los_flags[x][next_y] & step_y_flags)) return 0;
        }
    } else {
        int y = y0;
        int x_big = (x0 << 16) + 0x8000;
        int slope = (dx * 65536) / dy_abs;
        if (dx < 0) x_big--;
        int direction = dy < 0 ? -1 : 1;

        while (y != y1) {
            y += direction;
            int x = x_big >> 16;
            uint8_t step_y_flags = y_flags;
            if (x == x1 && y == y1) step_y_flags &= (uint8_t)~FC_LOS_FULL;
            if (los_flags[x][y] & step_y_flags) return 0;
            x_big += slope;
            int next_x = x_big >> 16;
            uint8_t step_x_flags = x_flags;
            if (next_x == x1 && y == y1) step_x_flags &= (uint8_t)~FC_LOS_FULL;
            if (next_x != x && (los_flags[next_x][y] & step_x_flags)) return 0;
        }
    }

    return 1;
}

static int fc_closest_area_coordinate(int anchor, int other, int size) {
    if (anchor >= other) return anchor;
    if (anchor + size - 1 <= other) return anchor + size - 1;
    return other;
}

int fc_has_los_between_areas(
    int src_x, int src_y, int src_size,
    int dst_x, int dst_y, int dst_size,
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (src_size <= 0 || dst_size <= 0) return 0;
    if (src_x < 0 || src_y < 0 || src_x + src_size > FC_ARENA_WIDTH ||
        src_y + src_size > FC_ARENA_HEIGHT ||
        dst_x < 0 || dst_y < 0 || dst_x + dst_size > FC_ARENA_WIDTH ||
        dst_y + dst_size > FC_ARENA_HEIGHT) {
        return 0;
    }

    int ray_src_x = fc_closest_area_coordinate(src_x, dst_x, src_size);
    int ray_src_y = fc_closest_area_coordinate(src_y, dst_y, src_size);
    int ray_dst_x = fc_closest_area_coordinate(dst_x, src_x, dst_size);
    int ray_dst_y = fc_closest_area_coordinate(dst_y, src_y, dst_size);
    return fc_has_line_of_sight(ray_src_x, ray_src_y,
                                ray_dst_x, ray_dst_y, los_flags);
}

int fc_npc_can_melee_player(int player_x, int player_y,
                            int npc_x, int npc_y, int npc_size,
                            const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                            const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (npc_size <= 0) return 0;
    int npc_max_x = npc_x + npc_size - 1;
    int npc_max_y = npc_y + npc_size - 1;

    /* Rectangular-exclusive reach: shared cardinal edges are valid; diagonal
     * corner contact and overlapping footprints are not. */
    (void)walkable;
    uint8_t source_wall;
    if (player_x == npc_x - 1 && player_y >= npc_y && player_y <= npc_max_y)
        source_wall = FC_MOVE_WALL_EAST;
    else if (player_x == npc_max_x + 1 &&
             player_y >= npc_y && player_y <= npc_max_y)
        source_wall = FC_MOVE_WALL_WEST;
    else if (player_y == npc_y - 1 &&
             player_x >= npc_x && player_x <= npc_max_x)
        source_wall = FC_MOVE_WALL_NORTH;
    else if (player_y == npc_max_y + 1 &&
             player_x >= npc_x && player_x <= npc_max_x)
        source_wall = FC_MOVE_WALL_SOUTH;
    else
        return 0;

    return player_x >= 0 && player_x < FC_ARENA_WIDTH &&
           player_y >= 0 && player_y < FC_ARENA_HEIGHT &&
           (movement_flags[player_x][player_y] & source_wall) == 0;
}

/* ======================================================================== */
/* BFS pathfinding                                                           */
/* ======================================================================== */

static const int FC_ROUTE_DIRECTIONS[8][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1},
    {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
};

static int fc_area_distance(int x, int y, int dst_x, int dst_y, int dst_size) {
    return fc_distance_between_areas(x, y, 1, dst_x, dst_y, dst_size);
}

typedef enum {
    FC_ROUTE_EXACT,
    FC_ROUTE_ATTACK,
} FcRouteGoalKind;

static int fc_route_goal_reached(
    FcRouteGoalKind kind, int x, int y,
    int dst_x, int dst_y, int dst_size, int attack_range,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (kind == FC_ROUTE_EXACT) return x == dst_x && y == dst_y;
    if (attack_range == FC_ROUTE_MELEE_RANGE)
        return fc_npc_can_melee_player(x, y, dst_x, dst_y, dst_size,
                                       walkable, movement_flags);
    int distance = fc_area_distance(x, y, dst_x, dst_y, dst_size);
    return distance > 0 && distance <= attack_range &&
           fc_has_los_between_areas(x, y, 1, dst_x, dst_y, dst_size,
                                    los_flags);
}

static int fc_reconstruct_route(
    int sx, int sy, int end_x, int end_y,
    const int8_t pdx[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const int8_t pdy[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps) {
    int px[FC_ARENA_WIDTH * FC_ARENA_HEIGHT];
    int py[FC_ARENA_WIDTH * FC_ARENA_HEIGHT];
    int plen = 0;
    int x = end_x;
    int y = end_y;
    while ((x != sx || y != sy) && plen < FC_ARENA_WIDTH * FC_ARENA_HEIGHT) {
        px[plen] = x;
        py[plen] = y;
        plen++;
        int back_x = pdx[x][y];
        int back_y = pdy[x][y];
        x += back_x;
        y += back_y;
    }
    int steps = plen < max_steps ? plen : max_steps;
    for (int i = 0; i < steps; i++) {
        out_x[i] = px[plen - 1 - i];
        out_y[i] = py[plen - 1 - i];
    }
    return steps;
}

static int fc_bfs_route(
    int sx, int sy, int dst_x, int dst_y, int dst_size,
    int move_near, FcRouteGoalKind goal_kind, int attack_range,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps) {
    int8_t pdx[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    int8_t pdy[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    uint8_t vis[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    uint16_t distance[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    int qx[FC_ARENA_WIDTH * FC_ARENA_HEIGHT];
    int qy[FC_ARENA_WIDTH * FC_ARENA_HEIGHT];
    int qh = 0, qt = 0;

    if (sx < 0 || sx >= FC_ARENA_WIDTH || sy < 0 || sy >= FC_ARENA_HEIGHT ||
        dst_size <= 0 || max_steps <= 0) return 0;
    memset(vis, 0, sizeof(vis));
    memset(distance, 0, sizeof(distance));

    vis[sx][sy] = 1;
    qx[qt] = sx; qy[qt] = sy; qt++;
    int found_x = -1;
    int found_y = -1;
    while (qh < qt) {
        int cx = qx[qh], cy = qy[qh]; qh++;
        if (fc_route_goal_reached(goal_kind, cx, cy, dst_x, dst_y, dst_size,
                                  attack_range, walkable, movement_flags, los_flags)) {
            found_x = cx;
            found_y = cy;
            break;
        }
        for (int d = 0; d < 8; d++) {
            int step_x = FC_ROUTE_DIRECTIONS[d][0];
            int step_y = FC_ROUTE_DIRECTIONS[d][1];
            int nx = cx + step_x, ny = cy + step_y;
            if (nx < 0 || nx >= FC_ARENA_WIDTH || ny < 0 || ny >= FC_ARENA_HEIGHT) continue;
            if (vis[nx][ny] || !fc_footprint_step_walkable(
                    cx, cy, step_x, step_y, 1,
                    walkable, movement_flags)) continue;
            vis[nx][ny] = 1;
            pdx[nx][ny] = (int8_t)-step_x;
            pdy[nx][ny] = (int8_t)-step_y;
            distance[nx][ny] = (uint16_t)(distance[cx][cy] + 1u);
            qx[qt] = nx; qy[qt] = ny; qt++;
        }
    }

    if (found_x < 0 && move_near && goal_kind == FC_ROUTE_EXACT) {
        int best_cost = 1000;
        int best_distance = 100;
        for (int x = dst_x - 10; x <= dst_x + 10; x++) {
            for (int y = dst_y - 10; y <= dst_y + 10; y++) {
                if (x < 0 || x >= FC_ARENA_WIDTH || y < 0 || y >= FC_ARENA_HEIGHT ||
                    !vis[x][y] || distance[x][y] >= 100) continue;
                int off_x = x - dst_x;
                int off_y = y - dst_y;
                int cost = off_x * off_x + off_y * off_y;
                if (cost < best_cost ||
                    (cost == best_cost && distance[x][y] < best_distance)) {
                    best_cost = cost;
                    best_distance = distance[x][y];
                    found_x = x;
                    found_y = y;
                }
            }
        }
    }

    if (found_x < 0 || (found_x == sx && found_y == sy)) return 0;
    return fc_reconstruct_route(sx, sy, found_x, found_y, pdx, pdy,
                                out_x, out_y, max_steps);
}

int fc_pathfind_bfs_move_near(
    int sx, int sy, int dx, int dy,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps) {
    return fc_bfs_route(sx, sy, dx, dy, 1, 1, FC_ROUTE_EXACT, 0,
                        walkable, movement_flags, NULL,
                        out_x, out_y, max_steps);
}

int fc_pathfind_attack_position(
    int sx, int sy, int target_x, int target_y, int target_size,
    int attack_range,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps) {
    return fc_bfs_route(sx, sy, target_x, target_y, target_size, 0,
                        FC_ROUTE_ATTACK, attack_range, walkable,
                        movement_flags, los_flags, out_x, out_y, max_steps);
}

#undef FC_BLOCK_WEST
#undef FC_BLOCK_EAST
#undef FC_BLOCK_SOUTH
#undef FC_BLOCK_NORTH
#undef FC_BLOCK_SOUTH_WEST
#undef FC_BLOCK_SOUTH_EAST
#undef FC_BLOCK_NORTH_WEST
#undef FC_BLOCK_NORTH_EAST
#undef FC_BLOCK_NORTH_AND_SOUTH_EAST
#undef FC_BLOCK_NORTH_AND_SOUTH_WEST
#undef FC_BLOCK_NORTH_EAST_AND_WEST
#undef FC_BLOCK_SOUTH_EAST_AND_WEST
#undef BLOCKED

/* Prayer */
#include <stddef.h>

/*
 * fc_prayer.c — Prayer activation, drain, and potion restore.
 *
 * OSRS prayer drain (from PrayerDrain.kt):
 *   Each tick: prayerDrainCounter += totalDrainEffect (sum of active prayer drains)
 *   prayerDrainResistance = 60 + (prayerBonus * 2)
 *   While counter > resistance: drain 1 prayer point, counter -= resistance
 *
 *   Protection prayers all cost drain=12 per tick (from prayers.toml).
 *   Prayer points are in tenths (430 = 43 prayer points).
 *   Each "drain 1 point" = subtract 10 tenths.
 *
 * Prayer potion restore:
 *   floor(prayer_level * 0.25) + 7 points per dose.
 *   For level 43: floor(10.75) + 7 = 17 points → 170 in tenths.
 */

#define PRAYER_OVERHEAD_DRAIN_RATE 12

static void enforce_post_loss_invariant(FcPlayer* p) {
    if (p->current_prayer > 0) return;
    p->current_prayer = 0;
    p->prayer = PRAYER_NONE;
    p->prayer_drain_counter = 0;
}

int fc_prayer_drain_tick(FcPlayer* p, int prayer_at_tick_start,
                         const FcPrayerTransition* transition) {
    int prayer_before = p->current_prayer;

    if (p->current_prayer <= 0) {
        enforce_post_loss_invariant(p);
        return 0;
    }

    if (p->prayer == PRAYER_NONE) return 0;
    if (prayer_at_tick_start == PRAYER_NONE) return 0;

    int performed_flick = transition != NULL &&
        transition->explicit_off_then_on &&
        transition->off_performed &&
        transition->on_succeeded;
    if (performed_flick) return 0;

    /* Counter-based drain matching OSRS PrayerDrain.kt exactly:
     * Accumulate drain rate each tick, drain 1 point when counter exceeds resistance. */
    int drain_rate = PRAYER_OVERHEAD_DRAIN_RATE;  /* all 3 protect prayers = 12 */
    int resistance = 60 + 2 * p->prayer_bonus;

    p->prayer_drain_counter += drain_rate;

    int requested_loss = 0;
    while (p->prayer_drain_counter > resistance) {
        p->prayer_drain_counter -= resistance;
        requested_loss += 10;
        if (requested_loss >= p->current_prayer) break;
    }

    if (requested_loss > 0) {
        fc_prayer_apply_loss_tenths(p, requested_loss);
    }
    return prayer_before - p->current_prayer;
}

FcPrayerTransition fc_prayer_apply_action(FcPlayer* p, int prayer_action) {
    FcPrayerTransition result = {0};
    result.prior_prayer = p->prayer;
    result.requested_final_prayer = p->prayer;

    switch (prayer_action) {
        case FC_PRAYER_NO_CHANGE:
            break;
        case FC_PRAYER_OFF:
            result.requested_final_prayer = PRAYER_NONE;
            result.off_requested = (p->prayer != PRAYER_NONE);
            p->prayer = PRAYER_NONE;
            break;
        case FC_PRAYER_MAGIC:
            result.requested_final_prayer = PRAYER_PROTECT_MAGIC;
            result.on_requested = (p->prayer != PRAYER_PROTECT_MAGIC);
            p->prayer = (p->current_prayer > 0) ? PRAYER_PROTECT_MAGIC : PRAYER_NONE;
            break;
        case FC_PRAYER_RANGE:
            result.requested_final_prayer = PRAYER_PROTECT_RANGE;
            result.on_requested = (p->prayer != PRAYER_PROTECT_RANGE);
            p->prayer = (p->current_prayer > 0) ? PRAYER_PROTECT_RANGE : PRAYER_NONE;
            break;
        case FC_PRAYER_MELEE:
            result.requested_final_prayer = PRAYER_PROTECT_MELEE;
            result.on_requested = (p->prayer != PRAYER_PROTECT_MELEE);
            p->prayer = (p->current_prayer > 0) ? PRAYER_PROTECT_MELEE : PRAYER_NONE;
            break;
        case FC_PRAYER_FLICK_MAGIC:
        case FC_PRAYER_FLICK_RANGE:
        case FC_PRAYER_FLICK_MELEE: {
            int requested_prayer = prayer_action == FC_PRAYER_FLICK_MAGIC
                ? PRAYER_PROTECT_MAGIC
                : (prayer_action == FC_PRAYER_FLICK_RANGE
                    ? PRAYER_PROTECT_RANGE : PRAYER_PROTECT_MELEE);
            result.requested_final_prayer = requested_prayer;
            result.off_requested = 1;
            result.off_performed = p->prayer != PRAYER_NONE;
            result.on_requested = 1;
            result.explicit_off_then_on = 1;
            p->prayer = PRAYER_NONE;
            if (p->current_prayer > 0) p->prayer = requested_prayer;
            break;
        }
        default:
            break;
    }

    result.actual_final_prayer = p->prayer;
    if (!result.explicit_off_then_on) {
        result.off_performed = result.off_requested;
    }
    result.on_succeeded = result.on_requested &&
        result.actual_final_prayer == result.requested_final_prayer;
    result.final_state_changed =
        result.actual_final_prayer != result.prior_prayer;
    return result;
}

int fc_prayer_apply_loss_tenths(FcPlayer* p, int requested_loss_tenths) {
    if (p == NULL) return 0;
    int prayer_before = p->current_prayer;
    if (requested_loss_tenths > 0 && p->current_prayer > 0) {
        int loss = requested_loss_tenths;
        if (loss > p->current_prayer) loss = p->current_prayer;
        p->current_prayer -= loss;
    }
    enforce_post_loss_invariant(p);
    return prayer_before - p->current_prayer;
}

int fc_prayer_potion_restore(int prayer_level) {
    /* floor(level * 0.25) + 7 points → in tenths */
    return (prayer_level / 4 + 7) * 10;
}

#undef PRAYER_OVERHEAD_DRAIN_RATE

/* Reward */
#include <string.h>

const char* const FC_CH_NAMES[FC_CH_COUNT] = {
    "damage_dealt", "progress", "damage_taken", "npc_kill", "wave_clear",
    "jad_kill", "cave_complete", "player_death", "correct_jad_prayer",
    "correct_danger_prayer", "prayer_lost", "unnecessary_prayer", "wave_stall",
    "no_progress", "no_attack", "jad_heal", "npc_heal", "invalid_action",
    "tick_penalty"
};

/* Populate a contiguous array view of the breakdown channels for iteration.
 * Order matches FcRwdChannel enum above. */
void fc_reward_breakdown_channels(const FcRewardBreakdown* b,
                                  float out[FC_CH_COUNT]) {
    out[FC_CH_DAMAGE_DEALT]             = b->damage_dealt;
    out[FC_CH_PROGRESS]                 = b->progress;
    out[FC_CH_DAMAGE_TAKEN]             = b->damage_taken;
    out[FC_CH_NPC_KILL]                 = b->npc_kill;
    out[FC_CH_WAVE_CLEAR]               = b->wave_clear;
    out[FC_CH_JAD_KILL]                 = b->jad_kill;
    out[FC_CH_CAVE_COMPLETE]            = b->cave_complete;
    out[FC_CH_PLAYER_DEATH]             = b->player_death;
    out[FC_CH_CORRECT_JAD_PRAYER]       = b->correct_jad_prayer;
    out[FC_CH_CORRECT_DANGER_PRAYER]    = b->correct_danger_prayer;
    out[FC_CH_PRAYER_LOST]              = b->prayer_lost;
    out[FC_CH_UNNECESSARY_PRAYER]       = b->unnecessary_prayer;
    out[FC_CH_WAVE_STALL]               = b->wave_stall;
    out[FC_CH_NO_PROGRESS]              = b->no_progress;
    out[FC_CH_NO_ATTACK]                = b->no_attack;
    out[FC_CH_JAD_HEAL]                 = b->jad_heal;
    out[FC_CH_NPC_HEAL]                 = b->npc_heal;
    out[FC_CH_INVALID_ACTION]           = b->invalid_action;
    out[FC_CH_TICK_PENALTY]             = b->tick_penalty;
}

FcRewardParams fc_reward_default_params(void) {
    FcRewardParams params;
    memset(&params, 0, sizeof(params));

    params.w_damage_dealt = 0.0f;
    params.w_progress = 0.001f;
    params.negative_progress_multiplier = 1.0f;
    params.w_damage_taken = -0.25f;
    params.w_npc_kill = 0.0f;
    params.w_wave_clear = 0.0f;
    params.w_jad_kill = 0.0f;
    params.w_cave_complete = 1.0f;
    params.w_player_death = -1.0f;
    params.scale_player_death_with_progress = 0;
    params.player_death_min_scale = 0.1f;
    params.w_correct_jad_prayer = 0.0f;
    params.w_correct_danger_prayer = 0.005f;
    params.w_prayer_lost = -0.02f;
    params.w_invalid_action = -0.1f;
    params.w_tick_penalty = -0.0001f;

    params.shape_unnecessary_prayer_penalty = 0.0f;
    params.shape_wave_stall_base_penalty = 0.0f;
    params.shape_wave_stall_cap = 0.0f;
    params.shape_jad_heal_penalty = 0.0f;
    params.shape_npc_heal_penalty = 0.0f;
    params.shape_no_progress_penalty_1 = -0.001f;
    params.shape_no_progress_penalty_2 = -0.005f;
    params.shape_no_progress_penalty_3 = -0.02f;
    params.shape_no_attack_base_penalty = -0.005f;
    params.shape_no_attack_wave_scale = 0.05f;

    params.shape_wave_stall_start = 0;
    params.shape_wave_stall_ramp_interval = 0;
    params.shape_no_progress_start_1 = 800;
    params.shape_no_progress_start_2 = 1600;
    params.shape_no_progress_start_3 = 2400;
    params.shape_no_attack_start = 50;

    return params;
}

void fc_reward_runtime_reset(FcRewardRuntime* runtime) {
    memset(runtime, 0, sizeof(*runtime));
}

static float reward_clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

float fc_reward_player_death_scale(
        const FcRewardParams* params, float cave_progress) {
    if (!params->scale_player_death_with_progress) return 1.0f;

    float floor = reward_clamp01(params->player_death_min_scale);
    float progress = reward_clamp01(cave_progress);
    return floor + (1.0f - floor) * progress;
}

float fc_reward_required_work_remaining(const FcState* state) {
    if (state->terminal == TERMINAL_CAVE_COMPLETE) {
        return 0.0f;
    }

    float work = 0.0f;
    const FcNpcStats* small_kek_stats = fc_npc_get_stats(NPC_TZ_KEK_SM);

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* npc = &state->npcs[i];
        if (!npc->active || npc->is_dead) continue;

        if (state->current_wave == FC_NUM_WAVES) {
            if (npc->npc_type == NPC_TZTOK_JAD) {
                work += (float)npc->current_hp;
            }
            continue;
        }

        if (npc->npc_type == NPC_TZ_KEK) {
            work += (float)npc->current_hp +
                    2.0f * (float)small_kek_stats->max_hp;
        } else {
            work += (float)npc->current_hp;
        }
    }

    return (work > 0.0f) ? work : 0.0f;
}

static float reward_current_wave_progress(
        const FcState* state, const FcRewardRuntime* runtime,
        float required_work_remaining) {
    if (state->terminal == TERMINAL_CAVE_COMPLETE) {
        return 1.0f;
    }
    if (state->wave_just_cleared && state->terminal == TERMINAL_NONE) {
        return 0.0f;
    }
    if (runtime->required_work_at_wave_start <= 0.0f) {
        return (required_work_remaining <= 0.0f) ? 1.0f : 0.0f;
    }

    return reward_clamp01(
        1.0f - required_work_remaining / runtime->required_work_at_wave_start);
}

static float reward_cave_progress(
        const FcState* state, float current_wave_progress) {
    if (state->terminal == TERMINAL_CAVE_COMPLETE) {
        return 1.0f;
    }

    int waves_cleared = state->current_wave - 1;
    if (waves_cleared < 0) waves_cleared = 0;
    if (waves_cleared > FC_NUM_WAVES) waves_cleared = FC_NUM_WAVES;
    return reward_clamp01(
        ((float)waves_cleared + current_wave_progress) / (float)FC_NUM_WAVES);
}

void fc_reward_sync_progress_state(
        FcState* state, const FcRewardRuntime* runtime) {
    state->progress_required_work_start = runtime->required_work_at_wave_start;
    state->progress_required_work_remaining = runtime->last_required_work_remaining;
    state->progress_current_wave_progress = runtime->last_current_wave_progress;
    state->progress_cave_progress = runtime->last_cave_progress;
    state->progress_ticks_since_positive = runtime->ticks_since_positive_progress;
}

void fc_reward_runtime_begin_episode(
        FcRewardRuntime* runtime, FcState* state) {
    fc_reward_runtime_reset(runtime);
    runtime->required_work_at_wave_start = fc_reward_required_work_remaining(state);
    runtime->last_required_work_remaining = runtime->required_work_at_wave_start;
    runtime->last_current_wave_progress = reward_current_wave_progress(
        state, runtime, runtime->last_required_work_remaining);
    runtime->last_cave_progress = reward_cave_progress(
        state, runtime->last_current_wave_progress);
    runtime->cave_progress_prev = runtime->last_cave_progress;
    fc_reward_sync_progress_state(state, runtime);
}

static FcRewardThreatContext reward_collect_threat_context(
        const FcState* state) {
    FcRewardThreatContext ctx;
    const FcPlayer* p = &state->player;

    memset(&ctx, 0, sizeof(ctx));

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];
        if (!n->active || n->is_dead) continue;

        int dist = fc_distance_to_npc(p->x, p->y, n);
        if (dist <= 1) {
            ctx.melee_pressure_npcs++;
            if (n->npc_type == NPC_TOK_XIL) ctx.tokxil_melee = 1;
            if (n->npc_type == NPC_KET_ZEK) ctx.ketzek_melee = 1;
        }

        if (dist <= n->attack_range) {
            ctx.any_threat = 1;
        }
    }

    for (int i = 0; i < p->num_pending_hits; i++) {
        const FcPendingHit* ph = &p->pending_hits[i];
        if (!ph->active) continue;
        ctx.any_threat = 1;
    }

    return ctx;
}

FcRewardBreakdown fc_reward_compute_breakdown(
        const FcState* state, const FcRewardParams* params, FcRewardRuntime* runtime) {
    FcRewardBreakdown out;
    const FcPlayer* p = &state->player;
    int prayer_reward_idle;

    memset(&out, 0, sizeof(out));
    fc_write_reward_features(state, out.raw);
    out.threat_ctx = reward_collect_threat_context(state);
    prayer_reward_idle =
        (runtime->ticks_since_attack >= 1 && out.raw[FC_RWD_ATTACK_ATTEMPT] <= 0.0f);

    {
        float work_remaining = fc_reward_required_work_remaining(state);
        float wave_progress = reward_current_wave_progress(
            state, runtime, work_remaining);
        float cave_progress = reward_cave_progress(state, wave_progress);
        float start_work = runtime->required_work_at_wave_start;
        float progress_delta = cave_progress - runtime->cave_progress_prev;
        float net_work_removed = progress_delta * (float)FC_NUM_WAVES *
            ((start_work > 0.0f) ? start_work : 0.0f);

        /* Scalar reward uses raw net required-work removed. The cave-progress
         * delta stays normalized for observations/logs, while this channel pays
         * for actual HP/work removed and goes negative when healing restores
         * work. The optional negative multiplier makes restored work more costly
         * without changing positive progress. Multiplying by the wave's start
         * work preserves wave-clear handling and avoids treating the next wave
         * spawn as negative progress. */
        float progress_weight = params->w_progress;
        if (net_work_removed < 0.0f) {
            progress_weight *= params->negative_progress_multiplier;
        }
        out.progress = net_work_removed * progress_weight;

        runtime->last_required_work_remaining = work_remaining;
        runtime->last_current_wave_progress = wave_progress;
        runtime->last_cave_progress = cave_progress;
        runtime->last_progress_delta = progress_delta;
        runtime->last_progress_reward = out.progress;
        runtime->last_net_required_work_removed = net_work_removed;

        if (net_work_removed > 0.0001f) {
            runtime->ticks_since_positive_progress = 0;
            runtime->positive_progress_ticks++;
        } else {
            runtime->ticks_since_positive_progress++;
            if (net_work_removed < -0.0001f) {
                runtime->negative_progress_ticks++;
            } else {
                runtime->zero_progress_ticks++;
            }
        }

        if (params->shape_no_progress_start_1 > 0 &&
            runtime->ticks_since_positive_progress > params->shape_no_progress_start_1) {
            out.no_progress += params->shape_no_progress_penalty_1;
        }
        if (params->shape_no_progress_start_2 > 0 &&
            runtime->ticks_since_positive_progress > params->shape_no_progress_start_2) {
            out.no_progress += params->shape_no_progress_penalty_2;
        }
        if (params->shape_no_progress_start_3 > 0 &&
            runtime->ticks_since_positive_progress > params->shape_no_progress_start_3) {
            out.no_progress += params->shape_no_progress_penalty_3;
        }
    }

    /* damage_dealt fires per damaging hit: (damage + damaging_hits) * w.
     * Base reward per hit only applies when actual damage is dealt; zero
     * damage impacts still resolve mechanically but do not pay damage reward. */
    out.damage_dealt = (out.raw[FC_RWD_DAMAGE_DEALT] +
                       (float)state->hits_landed_this_tick) * params->w_damage_dealt;

    {
        float dmg_frac = out.raw[FC_RWD_DAMAGE_TAKEN];
        out.damage_taken = dmg_frac * params->w_damage_taken;
    }

    out.npc_kill = out.raw[FC_RWD_NPC_KILL] * params->w_npc_kill;

    if (out.raw[FC_RWD_WAVE_CLEAR] > 0.0f) {
        int cleared_wave = state->current_wave - 1;
        if (cleared_wave < 1) cleared_wave = 1;
        out.wave_clear = params->w_wave_clear * (float)cleared_wave;
    }

    out.jad_kill = out.raw[FC_RWD_JAD_KILL] * params->w_jad_kill;
    out.cave_complete = out.raw[FC_RWD_CAVE_COMPLETE] * params->w_cave_complete;
    out.player_death = out.raw[FC_RWD_PLAYER_DEATH] * params->w_player_death *
        fc_reward_player_death_scale(params, runtime->last_cave_progress);

    /* Jad now participates in the same correct-block reward as every other
     * NPC. Preserve the existing attack-idle suppression unchanged. Jad's
     * separate channel remains an optional additional bonus. */
    if (!prayer_reward_idle) {
        out.correct_jad_prayer =
            out.raw[FC_RWD_CORRECT_JAD_PRAY] * params->w_correct_jad_prayer;
        out.correct_danger_prayer =
            out.raw[FC_RWD_CORRECT_DANGER_PRAY] * params->w_correct_danger_prayer;
    }
    out.prayer_lost = out.raw[FC_RWD_PRAYER_LOST] * params->w_prayer_lost;
    if (p->prayer != PRAYER_NONE && !out.threat_ctx.any_threat) {
        out.unnecessary_prayer = params->shape_unnecessary_prayer_penalty;
    }

    out.invalid_action = out.raw[FC_RWD_INVALID_ACTION] * params->w_invalid_action;
    out.tick_penalty = out.raw[FC_RWD_TICK_PENALTY] * params->w_tick_penalty;

    if (out.raw[FC_RWD_ATTACK_ATTEMPT] > 0.0f) {
        runtime->ticks_since_attack = 0;
    } else if (state->npcs_remaining > 0 && p->attack_timer <= 0) {
        runtime->ticks_since_attack++;
    } else if (state->npcs_remaining <= 0) {
        runtime->ticks_since_attack = 0;
    }

    if (params->shape_no_attack_start > 0 &&
        params->shape_no_attack_base_penalty != 0.0f &&
        state->npcs_remaining > 0 &&
        runtime->ticks_since_attack > params->shape_no_attack_start) {
        float wave = (float)state->current_wave;
        if (wave < 1.0f) wave = 1.0f;
        float multiplier = 1.0f +
            params->shape_no_attack_wave_scale * (wave - 1.0f);
        if (multiplier < 0.0f) multiplier = 0.0f;
        out.no_attack = params->shape_no_attack_base_penalty * multiplier;
    }

    /* Wave-stall penalty — timer-based, fires every tick past the threshold
     * while the wave still has NPCs. Ramps linearly and clamps at cap.
     * Runtime timer resets when a wave_clear fires this tick. */
    if (state->npcs_remaining > 0) {
        runtime->ticks_in_wave++;
        if (params->shape_wave_stall_base_penalty != 0.0f &&
            runtime->ticks_in_wave > params->shape_wave_stall_start) {
            int over = runtime->ticks_in_wave - params->shape_wave_stall_start;
            int ramps = (params->shape_wave_stall_ramp_interval > 0)
                ? over / params->shape_wave_stall_ramp_interval : 0;
            float p = params->shape_wave_stall_base_penalty * (1.0f + (float)ramps);
            float cap = params->shape_wave_stall_cap;
            if (cap != 0.0f && p < cap) p = cap;
            out.wave_stall = p;
        }
    }
    if (out.raw[FC_RWD_WAVE_CLEAR] > 0.0f) {
        runtime->ticks_in_wave = 0;
    }

    /* Jad heal penalty: fires per Yt-HurKot heal proc that landed on Jad
     * this tick. Encourages the agent to break healer link or kill healers
     * before they restore Jad's HP. */
    if (state->jad_heal_procs_this_tick > 0 &&
        params->shape_jad_heal_penalty != 0.0f) {
        out.jad_heal = params->shape_jad_heal_penalty *
                       (float)state->jad_heal_procs_this_tick;
    }
    if (state->npc_heal_procs_this_tick > 0 &&
        params->shape_npc_heal_penalty != 0.0f) {
        out.npc_heal = params->shape_npc_heal_penalty *
                       (float)state->npc_heal_procs_this_tick;
    }

    if (state->wave_just_cleared && state->terminal == TERMINAL_NONE) {
        runtime->required_work_at_wave_start =
            fc_reward_required_work_remaining(state);
        runtime->last_required_work_remaining =
            runtime->required_work_at_wave_start;
        runtime->last_current_wave_progress = 0.0f;
        runtime->last_cave_progress =
            reward_cave_progress(state, runtime->last_current_wave_progress);
    }
    runtime->cave_progress_prev = runtime->last_cave_progress;

    out.total =
        out.damage_dealt +
        out.progress +
        out.damage_taken +
        out.npc_kill +
        out.wave_clear +
        out.jad_kill +
        out.cave_complete +
        out.player_death +
        out.correct_jad_prayer +
        out.correct_danger_prayer +
        out.prayer_lost +
        out.unnecessary_prayer +
        out.wave_stall +
        out.no_progress +
        out.no_attack +
        out.jad_heal +
        out.npc_heal +
        out.invalid_action +
        out.tick_penalty;

    return out;
}


/* Rng */
/*
 * XORshift32 RNG — single state, deterministic, seeded at reset.
 * All randomness in the simulation flows through this RNG.
 * Adopted from PufferLib OSRS PvP (osrs_pvp_types.h).
 */

void fc_rng_seed(FcState* state, uint32_t seed) {
    /* XORshift32 cannot have state 0 — if seed is 0, use a fixed nonzero value */
    state->rng_state = (seed != 0) ? seed : 0x12345678u;
    state->rng_seed = seed;
}

uint32_t fc_rng_next(FcState* state) {
    uint32_t x = state->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state->rng_state = x;
    return x;
}

int fc_rng_int(FcState* state, int max) {
    if (max <= 0) return 0;
    return (int)(fc_rng_next(state) % (uint32_t)max);
}

float fc_rng_float(FcState* state) {
    return (float)(fc_rng_next(state) & 0x00FFFFFFu) / (float)0x01000000u;
}


/* Spawn */
int fc_spawn_find_available_footprint(const FcState* state,
                                      int preferred_x, int preferred_y,
                                      int size, int max_radius,
                                      int* out_x, int* out_y) {
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    fc_build_occupancy(state, occupied, -1, 0);

    *out_x = preferred_x;
    *out_y = preferred_y;
    if (fc_footprint_available_dynamic(preferred_x, preferred_y, size,
                                       state->walkable, occupied)) {
        return 1;
    }

    for (int radius = 1; radius <= max_radius; radius++) {
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                if (dx != -radius && dx != radius &&
                    dy != -radius && dy != radius) {
                    continue;
                }
                int x = preferred_x + dx;
                int y = preferred_y + dy;
                if (fc_footprint_available_dynamic(x, y, size,
                                                   state->walkable, occupied)) {
                    *out_x = x;
                    *out_y = y;
                    return 1;
                }
            }
        }
    }

    return 0;
}

int fc_spawn_npc_first_free(FcState* state, int npc_type, int x, int y) {
    for (int slot = 0; slot < FC_MAX_NPCS; slot++) {
        if (state->npcs[slot].active) continue;
        fc_npc_spawn(&state->npcs[slot], npc_type, x, y,
                     state->next_spawn_index++);
        return slot;
    }
    return -1;
}


/* State */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * fc_state.c — State allocation, initialization, reset, rendering.
 *
 * FcState is caller-allocated (stack or heap). These functions
 * initialize and reset it. memset to zero is the canonical reset
 * mechanism — all fields must have safe zero defaults.
 */

/* ======================================================================== */
/* Arena collision map (from Void 634 cache, region 37,79, level 0)          */
/* ======================================================================== */

/*
 * Binary collision extracted via DumpFcCollision.kt from the Void 634 cache.
 * fightcaves.collision stores whole-tile blocking as 64*64 row-major bytes.
 * fightcaves.movement stores the corresponding directional wall bits.
 *
 * Loaded from resources/fight_caves/runtime in a packaged PufferLib checkout.
 * All three arena maps are required; running without one would change the
 * simulation's movement or line-of-sight rules.
 */
/* Cached arena data — loaded once and shared by all envs to avoid per-reset
 * file I/O. Each map retains separate storage and initialization state. */
typedef struct {
    uint8_t cells[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    int loaded;
} FcArenaMapCache;

static FcArenaMapCache g_collision_cache;
static FcArenaMapCache g_movement_cache;
static FcArenaMapCache g_los_cache;

static const char* const g_arena_asset_path_formats[] = {
    "resources/fight_caves/runtime/%s",
    NULL
};

static void fail_required_arena_asset(const char* filename,
                                      const char* override_name) {
    fprintf(stderr,
            "fatal: required Fight Caves arena asset '%s' is missing, "
            "unreadable, or not exactly %d bytes.\n"
            "Set %s to the asset's absolute path or install the pinned "
            "runtime bundle with 'python3 ocean/fight_caves/tools.py "
            "setup --core'.\n",
            filename, FC_ARENA_WIDTH * FC_ARENA_HEIGHT,
            override_name);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

static void load_required_arena_map(FcArenaMapCache* cache,
                                    const char* filename,
                                    const char* override_name) {
    FILE* file = NULL;
    const char* override_path;
    char path[256];
    uint8_t bytes[FC_ARENA_WIDTH * FC_ARENA_HEIGHT];
    size_t count;

    if (cache->loaded) return;
    override_path = getenv(override_name);
    if (override_path && override_path[0]) {
        file = fopen(override_path, "rb");
        if (!file) {
            fprintf(stderr,
                    "fatal: %s points to an unreadable Fight Caves arena "
                    "asset: %s\n",
                    override_name, override_path);
            fail_required_arena_asset(filename, override_name);
        }
    } else {
        for (int i = 0; !file && g_arena_asset_path_formats[i]; i++) {
            int length = snprintf(path, sizeof(path),
                                  g_arena_asset_path_formats[i], filename);
            if (length > 0 && (size_t)length < sizeof(path)) {
                file = fopen(path, "rb");
            }
        }
    }
    if (!file) fail_required_arena_asset(filename, override_name);

    count = fread(bytes, 1, sizeof(bytes), file);
    fclose(file);
    if (count != sizeof(bytes)) {
        fail_required_arena_asset(filename, override_name);
    }

    /* Binary files are row-major [y][x]; FcState maps are [x][y]. */
    for (int y = 0; y < FC_ARENA_HEIGHT; y++) {
        for (int x = 0; x < FC_ARENA_WIDTH; x++) {
            cache->cells[x][y] = bytes[y * FC_ARENA_WIDTH + x];
        }
    }
    cache->loaded = 1;
}

static void setup_arena(FcState* state) {
    load_required_arena_map(&g_collision_cache, "fightcaves.collision",
                            "FC_COLLISION_PATH");
    load_required_arena_map(&g_movement_cache, "fightcaves.movement",
                            "FC_MOVEMENT_PATH");
    load_required_arena_map(&g_los_cache, "fightcaves.los", "FC_LOS_PATH");
    memcpy(state->walkable, g_collision_cache.cells, sizeof(state->walkable));
    memcpy(state->movement_flags, g_movement_cache.cells,
           sizeof(state->movement_flags));
    memcpy(state->los_flags, g_los_cache.cells, sizeof(state->los_flags));
}

/* Player initialization — the loadout table is the combat-state authority. */

static void apply_loadout_combat_fields(FcPlayer* p,
                                        const FcLoadout* loadout) {
    p->max_hp = loadout->max_hp;
    p->max_prayer = loadout->max_prayer;
    p->attack_level = loadout->attack_lvl;
    p->strength_level = loadout->strength_lvl;
    p->defence_level = loadout->defence_lvl;
    p->ranged_level = loadout->ranged_lvl;
    p->prayer_level = loadout->prayer_lvl;
    p->magic_level = loadout->magic_lvl;
    fc_items_init(p, loadout);
}

static void init_player(FcPlayer* p) {
    const FcLoadout* loadout = &FC_LOADOUTS[FC_ACTIVE_LOADOUT];
    apply_loadout_combat_fields(p, loadout);
    p->x = FC_ARENA_WIDTH / 2;
    p->y = FC_ARENA_HEIGHT / 2;
    p->current_hp = p->max_hp;
    p->current_prayer = p->max_prayer;
    p->prayer = PRAYER_NONE;
    p->prayer_at_tick_start = PRAYER_NONE;
    p->attack_timer = 0;
    p->food_timer = 0;
    p->potion_timer = 0;
    p->combo_timer = 0;
    p->run_energy = FC_RUN_ENERGY_MAX;
    p->is_running = 1;
    p->hp_regen_counter = 0;
    p->route_len = 0;
    p->route_idx = 0;
    p->attack_target_idx = -1;
    p->approach_target = 0;
    p->approach_target_x = -1;
    p->approach_target_y = -1;
    p->approach_target_size = 0;
}

/* ======================================================================== */
/* Lifecycle                                                                 */
/* ======================================================================== */

static void validate_npc_table_or_abort(void) {
    for (int npc_type = NPC_TZ_KIH; npc_type < NPC_TYPE_COUNT; npc_type++) {
        const FcNpcStats* stats = fc_npc_get_stats(npc_type);
        if (fc_npc_stats_valid(stats)) continue;

        fprintf(stderr,
                "fc_init: invalid NPC maxima for type %d: melee=%d ranged=%d magic=%d tenths\n",
                npc_type, stats->melee_max_hit_tenths,
                stats->ranged_max_hit_tenths,
                stats->magic_max_hit_tenths);
        abort();
    }
}

static void validate_loadout_table_or_abort(void) {
    if (FC_ACTIVE_LOADOUT < 0 || FC_ACTIVE_LOADOUT >= FC_NUM_LOADOUTS) {
        fprintf(stderr, "fc_init: active loadout %d is outside [0,%d)\n",
                FC_ACTIVE_LOADOUT, FC_NUM_LOADOUTS);
        abort();
    }

    for (int loadout_id = 0; loadout_id < FC_NUM_LOADOUTS; loadout_id++) {
        const FcLoadout* loadout = &FC_LOADOUTS[loadout_id];
        int valid = loadout->max_hp > 0 && loadout->max_prayer > 0 &&
            loadout->attack_lvl >= 1 && loadout->strength_lvl >= 1 &&
            loadout->defence_lvl >= 1 && loadout->ranged_lvl >= 1 &&
            loadout->prayer_lvl >= 1 && loadout->magic_lvl >= 1 &&
            loadout->weapon_kind >= FC_WEAPON_GENERIC_RANGED &&
            loadout->weapon_kind <= FC_WEAPON_BOW_OF_FAERDHINEN &&
            (loadout->weapon_uses_ammo == 0 ||
             loadout->weapon_uses_ammo == 1) &&
            loadout->ammo >= 0 &&
            (loadout->crystal_piece_mask & ~FC_CRYSTAL_PIECE_ALL) == 0 &&
            loadout->equipment_count >= 0 &&
            loadout->equipment_count <= FC_LOADOUT_EQUIP_MAX &&
            loadout->model_item_count >= 0 &&
            loadout->model_item_count <= FC_LOADOUT_MODEL_ITEM_MAX;

        FcPlayer player = {0};
        apply_loadout_combat_fields(&player, loadout);
        if (fc_player_ranged_base_max_hit_hp(&player) <= 0) valid = 0;
        for (int npc_type = NPC_TZ_KIH;
             valid && npc_type < NPC_TYPE_COUNT; npc_type++) {
            FcNpc target = {0};
            target.npc_type = npc_type;
            if (fc_player_ranged_final_max_hit_hp(&player, &target) <= 0)
                valid = 0;
        }

        if (valid) continue;
        fprintf(stderr,
                "fc_init: invalid loadout %d (skills=%d/%d/%d/%d/%d/%d weapon=%d ammo=%d/%d crystal=%d)\n",
                loadout_id, loadout->attack_lvl, loadout->strength_lvl,
                loadout->defence_lvl, loadout->ranged_lvl,
                loadout->prayer_lvl, loadout->magic_lvl,
                loadout->weapon_kind, loadout->weapon_uses_ammo,
                loadout->ammo, loadout->crystal_piece_mask);
        abort();
    }
}

void fc_init(FcState* state) {
    validate_npc_table_or_abort();
    validate_loadout_table_or_abort();
    memset(state, 0, sizeof(FcState));
    state->active_loadout = FC_ACTIVE_LOADOUT;
}

void fc_reset(FcState* state, uint32_t seed) {
    /* Zero everything first — ensures no stale state, padding is clean */
    memset(state, 0, sizeof(FcState));

    /* Seed RNG */
    fc_rng_seed(state, seed);

    /* Select random rotation */
    state->rotation_id = fc_rng_int(state, FC_NUM_ROTATIONS);

    /* Setup arena */
    setup_arena(state);

    /* Initialize player */
    init_player(&state->player);
    state->active_loadout = FC_ACTIVE_LOADOUT;
    state->render_events.player_attack_target_npc_slot = -1;
    state->render_events.player_move_start_x = state->player.x;
    state->render_events.player_move_start_y = state->player.y;

    /* Spawn wave 1 NPCs */
    state->current_wave = 1;
    state->next_spawn_index = 0;
    state->wave_start_tick = 0;
    fc_wave_spawn(state, 1);
}

void fc_step(FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
    if (state->terminal != TERMINAL_NONE) return;  /* episode over */
    fc_tick(state, actions);
}

void fc_request_set_running(FcState* state, int enabled) {
    if (!state || state->terminal != TERMINAL_NONE) return;
    state->player.is_running = enabled &&
        state->player.run_energy >= FC_RUN_ENERGY_MIN_START;
}

void fc_destroy(FcState* state) {
    /* Currently no heap allocations. Zero for safety. */
    memset(state, 0, sizeof(FcState));
}

/* ======================================================================== */
/* Observation / Mask / Reward                                               */
/* ======================================================================== */

/* Sort helper: indices of active NPCs, sorted by (distance, spawn_index) */
static int compare_npc_slots(const void* a, const void* b, const FcState* state) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    int da = fc_distance_to_npc(state->player.x, state->player.y, &state->npcs[ia]);
    int db = fc_distance_to_npc(state->player.x, state->player.y, &state->npcs[ib]);
    if (da != db) return da - db;
    return state->npcs[ia].spawn_index - state->npcs[ib].spawn_index;
}

/* Simple insertion sort for small arrays (max 16 elements) */
static void sort_npc_indices(int* indices, int count, const FcState* state) {
    for (int i = 1; i < count; i++) {
        int key = indices[i];
        int j = i - 1;
        while (j >= 0 && compare_npc_slots(&key, &indices[j], state) < 0) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

int fc_visible_npc_indices(const FcState* state, int out_indices[FC_VISIBLE_NPCS]) {
    int active_indices[FC_MAX_NPCS];
    int active_count = 0;

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (state->npcs[i].active && !state->npcs[i].is_dead) {
            active_indices[active_count++] = i;
        }
    }

    sort_npc_indices(active_indices, active_count, state);

    int visible = (active_count < FC_VISIBLE_NPCS) ? active_count : FC_VISIBLE_NPCS;
    for (int slot = 0; slot < visible; slot++) {
        out_indices[slot] = active_indices[slot];
    }
    return visible;
}

static int move_action_valid(const FcState* state, int action) {
    const FcPlayer* p = &state->player;

    if (action < 0 || action >= FC_MOVE_DIM) return 0;
    if (action == FC_MOVE_IDLE) return 1;
    if (action >= FC_MOVE_RUN_N && !fc_player_can_run(p)) {
        return 0;
    }

    int tx = p->x;
    int ty = p->y;
    int max_steps = (action >= FC_MOVE_RUN_N) ? 2 : 1;
    return fc_move_toward(&tx, &ty, FC_MOVE_DX[action], FC_MOVE_DY[action],
                          max_steps, state->walkable,
                          state->movement_flags) > 0;
}

static int attack_action_valid(const FcState* state, int action) {
    int visible_indices[FC_VISIBLE_NPCS];
    int visible;
    int slot;

    if (action < 0 || action >= FC_ATTACK_DIM) return 0;
    if (action == FC_ATTACK_NONE) return 1;

    visible = fc_visible_npc_indices(state, visible_indices);
    slot = action - 1;
    return slot < visible;
}

static int prayer_action_valid(int action) {
    return action >= 0 && action < FC_PRAYER_DIM;
}

int fc_eat_action_valid(const FcState* state, int action) {
    const FcPlayer* p = &state->player;

    if (action < 0 || action >= FC_EAT_DIM) return 0;
    if (action == FC_EAT_NONE) return 1;
    if (action == FC_EAT_SHARK) {
        return p->sharks_remaining > 0 &&
               p->food_timer <= 0 &&
               p->current_hp < p->max_hp;
    }
    if (action == FC_EAT_COMBO) {
        return p->sharks_remaining > 0 &&
               p->combo_timer <= 0 &&
               p->current_hp < p->max_hp;
    }
    return 0;
}

int fc_drink_action_valid(const FcState* state, int action) {
    const FcPlayer* p = &state->player;

    if (action < 0 || action >= FC_DRINK_DIM) return 0;
    if (action == FC_DRINK_NONE) return 1;
    if (action == FC_DRINK_PRAYER_POT) {
        return p->prayer_doses_remaining > 0 &&
               p->potion_timer <= 0 &&
               p->current_prayer < p->max_prayer;
    }
    return 0;
}

static int attack_style_summary_idx(int style) {
    switch (style) {
        case ATTACK_MELEE:  return 0;
        case ATTACK_RANGED: return 1;
        case ATTACK_MAGIC:  return 2;
        default:            return -1;
    }
}

static float normalize_incoming_count(int count) {
    if (count <= 0) return 0.0f;
    if (count >= 4) return 1.0f;
    return (float)count / 4.0f;
}

static float clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static float normalize_prayer_drain_counter(const FcPlayer* p) {
    int resistance = 60 + 2 * p->prayer_bonus;
    if (resistance <= 0) return 0.0f;
    float normalized = (float)p->prayer_drain_counter / (float)resistance;
    if (normalized < 0.0f) return 0.0f;
    if (normalized > 1.0f) return 1.0f;
    return normalized;
}

static float normalize_npc_prayer_drain(const FcNpc* npc) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    int maximum = stats->prayer_drain;
    if (npc->npc_type == NPC_TZ_KIH) {
        maximum += fc_npc_max_hit_tenths_for_style(stats, ATTACK_MELEE);
    }
    if (maximum <= 0) return 0.0f;
    return clamp01((float)npc->prayer_drain_dealt_this_tick / (float)maximum);
}

static float normalize_npc_heal_cooldown(const FcNpc* npc) {
    if (npc->npc_type == NPC_YT_MEJKOT) {
        if (npc->attack_speed <= 0) return 0.0f;
        return clamp01((float)npc->attack_timer / (float)npc->attack_speed);
    }

    if (npc->npc_type == NPC_YT_HURKOT && !npc->healer_distracted) {
        const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
        if (stats->heal_interval <= 0) return 0.0f;
        return clamp01((float)npc->heal_timer / (float)stats->heal_interval);
    }

    return 0.0f;
}

static int pending_hit_prayer_actionable(const FcState* state,
                                         const FcPendingHit* ph) {
    if (!ph->active) return 0;
    if (ph->prayer_snapshot >= 0) return 0;
    if (ph->prayer_lock_tick < 0) return 0;
    if (state->tick >= ph->prayer_lock_tick) return 0;
    return attack_style_summary_idx(ph->attack_style) >= 0;
}

static float pending_hit_prayer_deadline_urgency(const FcState* state,
                                                 const FcPendingHit* ph) {
    if (!pending_hit_prayer_actionable(state, ph)) return 0.0f;

    int ticks_until_lock = ph->prayer_lock_tick - state->tick;
    if (ticks_until_lock > 4) ticks_until_lock = 4;

    return (float)(5 - ticks_until_lock) / 4.0f;
}

/* Distance-only attack-style telegraph: what style would this NPC throw if it
 * attacked right now from its current position? Does NOT check LOS; the LOS
 * bit is a separate obs feature so the agent can distinguish "melee threat,
 * safespotted" (style=MELEE, LOS=0) from "melee threat, can hit me" (style=MELEE,
 * LOS=1). Returns ATTACK_NONE for empty slots, an untagged Yt-HurKot, and
 * stochastic style choices before the NPC has committed an attack. */
static int npc_telegraph_style(const FcState* state, const FcNpc* npc) {
    if (!npc->active || npc->is_dead) return ATTACK_NONE;
    if (npc->npc_type == NPC_YT_HURKOT) {
        return npc->healer_distracted ? ATTACK_MELEE : ATTACK_NONE;
    }

    if (npc->npc_type == NPC_TZTOK_JAD) {
        /* Jad alternates magic/ranged randomly at commit; read the committed
         * style from the queued pending_hit. No prediction before commit. */
        const FcPlayer* p = &state->player;
        for (int i = 0; i < p->num_pending_hits; i++) {
            const FcPendingHit* ph = &p->pending_hits[i];
            if (ph->active && ph->source_npc_idx >= 0 &&
                state->npcs[ph->source_npc_idx].npc_type == NPC_TZTOK_JAD) {
                return ph->attack_style;
            }
        }
        return ATTACK_NONE;
    }

    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    int can_melee = fc_npc_can_melee_player(state->player.x, state->player.y,
                                            npc->x, npc->y, npc->size,
                                            state->walkable,
                                            state->movement_flags);
    if (npc->npc_type == NPC_KET_ZEK && can_melee &&
        fc_has_los_between_areas(
            npc->x, npc->y, npc->size,
            state->player.x, state->player.y, 1, state->los_flags)) {
        return ATTACK_NONE;
    }
    /* Tok-Xil telegraphs melee when adjacent. Ket-Zek returns NONE above while
     * both adjacent styles remain possible. Pure melee NPCs (Yt-MejKot,
     * Tz-Kih, Tz-Kek) telegraph MELEE even when far
     * — they'll close the gap and that's what they'll hit with. */
    if (can_melee && (stats->melee_max_hit_tenths > 0 ||
                      npc->attack_style == ATTACK_MELEE)) {
        return ATTACK_MELEE;
    }
    return npc->attack_style;
}

static int npc_type_obs_offset(int npc_type) {
    switch (npc_type) {
        case NPC_TZ_KIH: return FC_NPC_TYPE_TZ_KIH;
        case NPC_TZ_KEK: return FC_NPC_TYPE_TZ_KEK;
        case NPC_TZ_KEK_SM: return FC_NPC_TYPE_TZ_KEK_SM;
        case NPC_TOK_XIL: return FC_NPC_TYPE_TOK_XIL;
        case NPC_YT_MEJKOT: return FC_NPC_TYPE_YT_MEJKOT;
        case NPC_KET_ZEK: return FC_NPC_TYPE_KET_ZEK;
        case NPC_TZTOK_JAD: return FC_NPC_TYPE_TZTOK_JAD;
        case NPC_YT_HURKOT: return FC_NPC_TYPE_YT_HURKOT;
        default: return -1;
    }
}

static int npc_kill_reward_eligible(const FcNpc* npc) {
    return npc->npc_type != NPC_YT_HURKOT ||
           !npc->is_respawned_jad_healer;
}

static int rewardable_npc_kills_this_tick(const FcState* state) {
    int count = state->npcs_killed_this_tick -
                state->respawned_jad_healers_killed_this_tick;
    return count > 0 ? count : 0;
}

void fc_write_obs(const FcState* state, float* out) {
    memset(out, 0, sizeof(float) * FC_TOTAL_OBS);

    const FcPlayer* p = &state->player;
    int incoming_counts[3][3] = {{0}};
    float prayer_deadline_urgency[3] = {0.0f, 0.0f, 0.0f};

    /* Compact incoming-hit timeline summary.
     * Counts by style for hits landing in 1, 2, and 3 ticks. This gives the
     * policy a relative timing signal without leaking absolute episode clocks.
     * Prayer deadline urgency is separate: it marks pending hits whose prayer
     * snapshot has not locked yet, which is the actual decision window. */
    for (int hi = 0; hi < p->num_pending_hits; hi++) {
        const FcPendingHit* ph = &p->pending_hits[hi];
        if (!ph->active) continue;
        int style_idx = attack_style_summary_idx(ph->attack_style);
        if (style_idx >= 0 && pending_hit_prayer_actionable(state, ph)) {
            float urgency = pending_hit_prayer_deadline_urgency(state, ph);
            if (urgency > prayer_deadline_urgency[style_idx]) {
                prayer_deadline_urgency[style_idx] = urgency;
            }
        }
        if (ph->ticks_remaining < 1 || ph->ticks_remaining > 3) continue;
        if (style_idx < 0) continue;
        int bucket = ph->ticks_remaining - 1;
        if (incoming_counts[bucket][style_idx] < 4) {
            incoming_counts[bucket][style_idx]++;
        }
    }

    /* Player features */
    float* player = out + FC_OBS_PLAYER_START;
    player[FC_OBS_PLAYER_HP]        = (p->max_hp > 0) ? (float)p->current_hp / (float)p->max_hp : 0.0f;
    player[FC_OBS_PLAYER_PRAYER]    = (p->max_prayer > 0) ? (float)p->current_prayer / (float)p->max_prayer : 0.0f;
    player[FC_OBS_PLAYER_X]         = (float)p->x / (float)FC_ARENA_WIDTH;
    player[FC_OBS_PLAYER_Y]         = (float)p->y / (float)FC_ARENA_HEIGHT;
    player[FC_OBS_PLAYER_ATK_TIMER] = (p->weapon_speed > 0)
        ? (float)p->attack_timer / (float)p->weapon_speed : 0.0f;
    player[FC_OBS_PLAYER_PRAY_MEL]  = (p->prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_PRAY_RNG]  = (p->prayer == PRAYER_PROTECT_RANGE) ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_PRAY_MAG]  = (p->prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_SHARKS]    = (float)p->sharks_remaining / (float)FC_MAX_SHARKS;
    player[FC_OBS_PLAYER_DOSES]     = (float)p->prayer_doses_remaining / (float)FC_MAX_PRAYER_DOSES;
    player[FC_OBS_PLAYER_IN_MEL_1T] = normalize_incoming_count(incoming_counts[0][0]);
    player[FC_OBS_PLAYER_IN_RNG_1T] = normalize_incoming_count(incoming_counts[0][1]);
    player[FC_OBS_PLAYER_IN_MAG_1T] = normalize_incoming_count(incoming_counts[0][2]);
    player[FC_OBS_PLAYER_IN_MEL_2T] = normalize_incoming_count(incoming_counts[1][0]);
    player[FC_OBS_PLAYER_IN_RNG_2T] = normalize_incoming_count(incoming_counts[1][1]);
    player[FC_OBS_PLAYER_IN_MAG_2T] = normalize_incoming_count(incoming_counts[1][2]);
    player[FC_OBS_PLAYER_TARGET]    = 0.0f;  /* filled after NPC slot computation below */
    player[FC_OBS_PLAYER_PRAY_DDL_MEL] = prayer_deadline_urgency[0];
    player[FC_OBS_PLAYER_PRAY_DDL_RNG] = prayer_deadline_urgency[1];
    player[FC_OBS_PLAYER_PRAY_DDL_MAG] = prayer_deadline_urgency[2];
    player[FC_OBS_PLAYER_PRAYER_LOST] = (p->max_prayer > 0)
        ? clamp01((float)state->prayer_lost_this_tick / (float)p->max_prayer)
        : 0.0f;
    player[FC_OBS_PLAYER_OVERHEAD_PRAYER_LOST] =
        state->overhead_prayer_lost_this_tick > 0 ? 1.0f : 0.0f;
    player[FC_OBS_PLAYER_RUN_ENERGY] =
        clamp01((float)p->run_energy / (float)FC_RUN_ENERGY_MAX);

    /* NPC slot selection: gather active NPCs, sort, take first 8 */
    int active_indices[FC_VISIBLE_NPCS];
    int visible = fc_visible_npc_indices(state, active_indices);
    for (int slot = 0; slot < visible; slot++) {
        const FcNpc* n = &state->npcs[active_indices[slot]];
        float* npc_out = out + FC_OBS_NPC_START + slot * FC_OBS_NPC_STRIDE;

        npc_out[FC_NPC_VALID]         = 1.0f;
        npc_out[FC_NPC_X]             = (float)n->x / (float)FC_ARENA_WIDTH;
        npc_out[FC_NPC_Y]             = (float)n->y / (float)FC_ARENA_HEIGHT;
        npc_out[FC_NPC_HP]            = (n->max_hp > 0) ? (float)n->current_hp / (float)n->max_hp : 0.0f;
        npc_out[FC_NPC_DISTANCE]      =
            (float)fc_distance_to_npc(p->x, p->y, n) / (float)FC_ARENA_WIDTH;
        float has_los = (float)fc_has_los_between_areas(
            p->x, p->y, 1, n->x, n->y, n->size, state->los_flags);
        int tele = npc_telegraph_style(state, n);
        npc_out[FC_NPC_TELE_MELEE]    = (tele == ATTACK_MELEE)  ? 1.0f : 0.0f;
        npc_out[FC_NPC_TELE_RANGED]   = (tele == ATTACK_RANGED) ? 1.0f : 0.0f;
        npc_out[FC_NPC_TELE_MAGIC]    = (tele == ATTACK_MAGIC)  ? 1.0f : 0.0f;
        npc_out[FC_NPC_ATK_TIMER]     = (n->attack_speed > 0) ? (float)n->attack_timer / (float)n->attack_speed : 0.0f;
        npc_out[FC_NPC_LOS]           = has_los;
        npc_out[FC_NPC_PRAYER_DRAIN_DEALT] = normalize_npc_prayer_drain(n);
        npc_out[FC_NPC_HEAL_RECEIVED] = (n->max_hp > 0)
            ? clamp01((float)n->healing_received_this_tick / (float)n->max_hp)
            : 0.0f;
        npc_out[FC_NPC_HEAL_GIVEN] = (n->heal_amount > 0)
            ? clamp01((float)n->healing_given_this_tick / (float)n->heal_amount)
            : 0.0f;
        npc_out[FC_NPC_HEALED_BY_MEJKOT] =
            n->healed_by_mejkot_this_tick ? 1.0f : 0.0f;
        npc_out[FC_NPC_HEALED_BY_HURKOT] =
            n->healed_by_hurkot_this_tick ? 1.0f : 0.0f;
        npc_out[FC_NPC_HEALED_SELF] = n->healed_self_this_tick ? 1.0f : 0.0f;
        npc_out[FC_NPC_TARGETS_PLAYER] =
            (n->npc_type != NPC_YT_HURKOT || n->healer_distracted) ? 1.0f : 0.0f;
        npc_out[FC_NPC_HEAL_COOLDOWN] = normalize_npc_heal_cooldown(n);
        npc_out[FC_NPC_KILL_REWARD_ELIGIBLE] =
            npc_kill_reward_eligible(n) ? 1.0f : 0.0f;
        int type_offset = npc_type_obs_offset(n->npc_type);
        if (type_offset >= 0) {
            npc_out[type_offset] = 1.0f;
        }

        /* Pending attack from this NPC — scan player's pending hits */
        npc_out[FC_NPC_PENDING_STYLE] = 0.0f;
        npc_out[FC_NPC_PENDING_TICKS] = 0.0f;
        npc_out[FC_NPC_PENDING_PRAYER_WINDOW] = 0.0f;
        npc_out[FC_NPC_PENDING_PRAYER_DEADLINE] = 0.0f;
        for (int hi = 0; hi < p->num_pending_hits; hi++) {
            const FcPendingHit* ph = &p->pending_hits[hi];
            if (ph->active && ph->source_npc_idx == active_indices[slot]) {
                npc_out[FC_NPC_PENDING_STYLE] = (float)ph->attack_style / 3.0f;
                npc_out[FC_NPC_PENDING_TICKS] = (float)ph->ticks_remaining / 10.0f;
                if (pending_hit_prayer_actionable(state, ph)) {
                    npc_out[FC_NPC_PENDING_PRAYER_WINDOW] = 1.0f;
                    npc_out[FC_NPC_PENDING_PRAYER_DEADLINE] =
                        pending_hit_prayer_deadline_urgency(state, ph);
                }
                break;  /* report first pending hit from this NPC */
            }
        }
    }
    /* Remaining NPC slots already zeroed by memset */

    /* Player target: which visible NPC slot is the current attack target */
    if (p->attack_target_idx >= 0) {
        for (int s = 0; s < visible; s++) {
            if (active_indices[s] == p->attack_target_idx) {
                player[FC_OBS_PLAYER_TARGET] = (float)(s + 1) / 8.0f;
                break;
            }
        }
    }

    /* Wave/meta features */
    float* meta = out + FC_OBS_META_START;
    meta[FC_OBS_META_WAVE]       = (float)state->current_wave / (float)FC_NUM_WAVES;
    meta[FC_OBS_META_ROTATION]   = (float)state->rotation_id / (float)FC_NUM_ROTATIONS;
    meta[FC_OBS_META_REMAINING]  = (float)state->npcs_remaining / (float)FC_MAX_NPCS;
    meta[FC_OBS_META_PRAY_DRAIN] = normalize_prayer_drain_counter(p);
    meta[FC_OBS_META_IN_MEL_3T]  = normalize_incoming_count(incoming_counts[2][0]);
    meta[FC_OBS_META_IN_RNG_3T]  = normalize_incoming_count(incoming_counts[2][1]);
    meta[FC_OBS_META_IN_MAG_3T]  = normalize_incoming_count(incoming_counts[2][2]);
    meta[FC_OBS_META_DMG_T_TICK] = (p->max_hp > 0) ? (float)state->damage_taken_this_tick / (float)p->max_hp : 0.0f;
    meta[FC_OBS_META_WAVE_CLR]   = (float)state->wave_just_cleared;
    meta[FC_OBS_META_CAVE_PROG]  = clamp01(state->progress_cave_progress);
    meta[FC_OBS_META_WAVE_PROG]  = clamp01(state->progress_current_wave_progress);
    meta[FC_OBS_META_WORK_REM]   = (state->progress_required_work_start > 0.0f)
        ? clamp01(state->progress_required_work_remaining /
                  state->progress_required_work_start)
        : 0.0f;
    meta[FC_OBS_META_NO_PROG]    = clamp01((float)state->progress_ticks_since_positive / 2400.0f);
    meta[FC_OBS_META_NPC_HEALING] = (state->progress_required_work_start > 0.0f)
        ? clamp01((float)state->npc_heal_amount_this_tick /
                  state->progress_required_work_start)
        : 0.0f;
    meta[FC_OBS_META_REWARDABLE_NPC_KILL] =
        rewardable_npc_kills_this_tick(state) > 0 ? 1.0f : 0.0f;

    /* Reward features (at offset FC_REWARD_START) — written by fc_write_reward_features */
    fc_write_reward_features(state, out + FC_REWARD_START);
}

void fc_apply_obs_ablation(float* out,
                           int ablate_npc_distance,
                           int ablate_incoming_aggregates,
                           int ablate_npc_valid) {
    if (ablate_npc_distance) {
        for (int s = 0; s < FC_OBS_NPC_SLOTS; s++) {
            out[FC_OBS_NPC_START + s * FC_OBS_NPC_STRIDE + FC_NPC_DISTANCE] = 0.0f;
        }
    }
    if (ablate_incoming_aggregates) {
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MEL_1T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_RNG_1T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MAG_1T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MEL_2T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_RNG_2T] = 0.0f;
        out[FC_OBS_PLAYER_START + FC_OBS_PLAYER_IN_MAG_2T] = 0.0f;
        out[FC_OBS_META_START + FC_OBS_META_IN_MEL_3T] = 0.0f;
        out[FC_OBS_META_START + FC_OBS_META_IN_RNG_3T] = 0.0f;
        out[FC_OBS_META_START + FC_OBS_META_IN_MAG_3T] = 0.0f;
    }
    if (ablate_npc_valid) {
        for (int s = 0; s < FC_OBS_NPC_SLOTS; s++) {
            out[FC_OBS_NPC_START + s * FC_OBS_NPC_STRIDE + FC_NPC_VALID] = 0.0f;
        }
    }
}

void fc_write_reward_features(const FcState* state, float* out) {
    memset(out, 0, sizeof(float) * FC_REWARD_FEATURES);

    out[FC_RWD_DAMAGE_DEALT]     = (float)state->damage_dealt_this_tick / 1000.0f;
    out[FC_RWD_DAMAGE_TAKEN]     = (state->player.max_hp > 0) ?
                                   (float)state->damage_taken_this_tick / (float)state->player.max_hp : 0.0f;
    out[FC_RWD_NPC_KILL]         = (float)rewardable_npc_kills_this_tick(state);
    out[FC_RWD_WAVE_CLEAR]       = (float)state->wave_just_cleared;
    out[FC_RWD_JAD_DAMAGE]       = (float)state->jad_damage_this_tick / 1000.0f;
    out[FC_RWD_JAD_KILL]         = (float)state->jad_killed;
    out[FC_RWD_PLAYER_DEATH]     = (state->terminal == TERMINAL_PLAYER_DEATH) ? 1.0f : 0.0f;
    out[FC_RWD_CAVE_COMPLETE]    = (state->terminal == TERMINAL_CAVE_COMPLETE) ? 1.0f : 0.0f;
    out[FC_RWD_FOOD_USED]        = (float)state->food_used_this_tick;
    out[FC_RWD_PRAYER_POT_USED]  = (float)state->prayer_potion_used_this_tick;
    out[FC_RWD_CORRECT_JAD_PRAY] = (float)state->correct_jad_prayer;
    out[FC_RWD_WRONG_JAD_PRAY]   = (float)state->wrong_jad_prayer;
    out[FC_RWD_INVALID_ACTION]   = (float)state->invalid_action_this_tick;
    out[FC_RWD_MOVEMENT]         = (float)state->movement_this_tick;
    out[FC_RWD_IDLE]             = (float)state->idle_this_tick;
    out[FC_RWD_TICK_PENALTY]     = 1.0f;  /* always fires */
    out[FC_RWD_CORRECT_DANGER_PRAY] = (float)state->correct_danger_prayer;
    out[FC_RWD_WRONG_DANGER_PRAY]   = (float)state->wrong_danger_prayer;
    out[FC_RWD_ATTACK_ATTEMPT]      = (float)state->attack_attempt_this_tick;
    out[FC_RWD_PRAYER_LOST]         = (float)state->prayer_lost_this_tick / 10.0f;
}

void fc_action_invalid_classes(const FcState* state,
                               const int actions[FC_NUM_ACTION_HEADS],
                               int out_classes[FC_INVALID_ACTION_CLASS_COUNT]) {
    /* Keep this aligned with the Puffer-facing policy mask surface:
     * move, attack, and prayer only. Consumable and path-target heads remain
     * canonical core actions, but are not emitted by the no-supplies policy. */
    out_classes[FC_INVALID_ACTION_MOVE] = !move_action_valid(state, actions[0]);
    out_classes[FC_INVALID_ACTION_ATTACK] = !attack_action_valid(state, actions[1]);
    out_classes[FC_INVALID_ACTION_PRAYER] = !prayer_action_valid(actions[2]);
}

void fc_write_mask(const FcState* state, float* out) {
    /* Set all to valid, then mask invalid */
    for (int i = 0; i < FC_ACTION_MASK_SIZE; i++) {
        out[i] = 1.0f;
    }

    /* MOVE: idle always valid. Walk/run directions masked if destination not walkable */
    for (int m = 1; m < FC_MOVE_DIM; m++) {
        if (!move_action_valid(state, m)) {
            out[FC_MASK_MOVE_START + m] = 0.0f;
        }
    }

    /* ATTACK: slot 0 (none) always valid. Slots 1-8 masked if no NPC in that slot */
    for (int attack = FC_ATTACK_NONE + 1; attack < FC_ATTACK_DIM; attack++) {
        if (!attack_action_valid(state, attack)) {
            out[FC_MASK_ATTACK_START + attack] = 0.0f;
        }
    }

    /* PRAYER: leave fully unmasked.
     * Prayer toggles may be legal even when they are redundant or no-op, and
     * the policy should learn those costs from the environment rather than
     * having them hidden by the mask. */

    /* EAT */
    if (!fc_eat_action_valid(state, FC_EAT_SHARK)) {
        out[FC_MASK_EAT_START + FC_EAT_SHARK] = 0.0f;
    }
    if (!fc_eat_action_valid(state, FC_EAT_COMBO)) {
        out[FC_MASK_EAT_START + FC_EAT_COMBO] = 0.0f;
    }

    /* DRINK */
    if (!fc_drink_action_valid(state, FC_DRINK_PRAYER_POT)) {
        out[FC_MASK_DRINK_START + FC_DRINK_PRAYER_POT] = 0.0f;
    }
}

int fc_is_terminal(const FcState* state) {
    return state->terminal != TERMINAL_NONE;
}

/* ======================================================================== */
/* Render entities                                                           */
/* ======================================================================== */

void fc_fill_render_entities(const FcState* state, FcRenderEntity* entities, int* count) {
    int idx = 0;

    /* Entity 0: player */
    FcRenderEntity* pe = &entities[idx++];
    memset(pe, 0, sizeof(FcRenderEntity));
    pe->entity_type = ENTITY_PLAYER;
    pe->x = state->player.x;
    pe->y = state->player.y;
    pe->size = 1;
    pe->current_hp = state->player.current_hp;
    pe->max_hp = state->player.max_hp;
    pe->prayer = state->player.prayer;
    pe->damage_taken_this_tick = state->player.damage_taken_this_tick;
    pe->hit_landed_this_tick = state->player.hit_landed_this_tick;

    /* Active NPCs + NPCs that just died this tick (for death hitsplat visibility) */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];
        if (!n->active && !n->died_this_tick) continue;

        FcRenderEntity* ne = &entities[idx++];
        memset(ne, 0, sizeof(FcRenderEntity));
        ne->entity_type = ENTITY_NPC;
        ne->npc_type = n->npc_type;
        ne->x = n->x;
        ne->y = n->y;
        ne->size = n->size;
        ne->current_hp = n->current_hp;
        ne->max_hp = n->max_hp;
        ne->attack_style = n->attack_style;
        ne->is_dead = n->is_dead;
        ne->damage_taken_this_tick = n->damage_taken_this_tick;
        ne->healing_received_this_tick = n->healing_received_this_tick;
        ne->died_this_tick = n->died_this_tick;
        ne->npc_slot = i;
    }

    *count = idx;
}

void fc_fill_render_events(const FcState* state, FcRenderEvents* events) {
    if (!state || !events) return;
    *events = state->render_events;
}


/* Tick */
#include <math.h>
/*
 * fc_tick.c — Main tick loop for Fight Caves simulation.
 *
 * Processing order (adapted from PufferLib PvP two-phase execution):
 *
 *   1. Clear per-tick event flags
 *   2. Process player actions:
 *      a. Prayer toggle (instant)
 *      b. Eat food / drink potion (if timer ready)
 *      c. Attack initiation from the pre-movement tile
 *      d. Movement (route or directional head), unless an attack fired
 *      e. Run-energy drain or restoration from actual movement
 *   3. Decrement player timers (attack, food, potion, combo)
 *   4. Prayer drain (only if prayer stayed active across the tick boundary)
 *   5. NPC AI tick (movement + attack) for all active NPCs
 *   6. Resolve pending hits (NPC → player, player → NPC)
 *   7. Check terminal conditions
 *   8. Increment tick and lock prayer snapshots due at the new boundary
 */

/* ======================================================================== */
/* Clear per-tick flags                                                      */
/* ======================================================================== */

static void clear_per_tick_flags(FcState* state) {
    state->render_events = (FcRenderEvents){0};
    state->render_events.player_attack_target_npc_slot = -1;
    state->render_events.player_move_start_x = state->player.x;
    state->render_events.player_move_start_y = state->player.y;
    state->damage_dealt_this_tick = 0;
    state->hits_landed_this_tick = 0;
    state->damage_taken_this_tick = 0;
    state->prayer_lost_this_tick = 0;
    state->overhead_prayer_lost_this_tick = 0;
    state->tz_kih_prayer_drain_this_tick = 0;
    state->npcs_killed_this_tick = 0;
    state->respawned_jad_healers_killed_this_tick = 0;
    state->wave_just_cleared = 0;
    state->jad_damage_this_tick = 0;
    state->jad_killed = 0;
    state->correct_jad_prayer = 0;
    state->wrong_jad_prayer = 0;
    state->correct_danger_prayer = 0;
    state->wrong_danger_prayer = 0;
    state->attack_attempt_this_tick = 0;
    state->invalid_action_this_tick = 0;
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; i++) {
        state->invalid_action_class_this_tick[i] = 0;
    }
    state->movement_this_tick = 0;
    state->idle_this_tick = 0;
    state->food_used_this_tick = 0;
    state->prayer_potion_used_this_tick = 0;
    state->jad_heal_procs_this_tick = 0;
    state->npc_heal_procs_this_tick = 0;
    state->npc_heal_amount_this_tick = 0;
    state->mejkot_heal_amount_this_tick = 0;
    state->jad_heal_amount_this_tick = 0;

    FcPlayer* p = &state->player;
    p->damage_taken_this_tick = 0;
    p->hit_style_this_tick = 0;
    p->hit_source_npc_type = 0;
    p->hit_locked_prayer_this_tick = 0;
    p->hit_blocked_this_tick = 0;
    p->hit_landed_this_tick = 0;
    p->food_eaten_this_tick = 0;
    p->potion_used_this_tick = 0;
    p->prayer_changed_this_tick = 0;

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        state->npcs[i].damage_taken_this_tick = 0;
        state->npcs[i].prayer_drain_dealt_this_tick = 0;
        state->npcs[i].healing_received_this_tick = 0;
        state->npcs[i].healing_given_this_tick = 0;
        state->npcs[i].healed_by_mejkot_this_tick = 0;
        state->npcs[i].healed_by_hurkot_this_tick = 0;
        state->npcs[i].healed_self_this_tick = 0;
        state->npcs[i].died_this_tick = 0;
    }
}

static void record_player_move_waypoint(FcState* state) {
    FcRenderEvents* events = &state->render_events;
    int index = events->player_move_waypoint_count;
    if (index >= FC_MAX_RENDER_MOVE_WAYPOINTS) return;
    events->player_move_waypoint_x[index] = state->player.x;
    events->player_move_waypoint_y[index] = state->player.y;
    events->player_move_waypoint_count++;
}

static void set_player_facing_from_delta(FcPlayer* player, float dx, float dy) {
    if (dx == 0.0f && dy == 0.0f) return;
    player->facing_angle = atan2f(dx, -dy) * (180.0f / 3.14159f);
}

/* ======================================================================== */
/* Resolve NPC visible-slot index to NPC array index                         */
/* ======================================================================== */

/* Same ordering as observation writer — must be identical for consistency */
static int npc_slot_to_index(const FcState* state, int slot) {
    int visible_indices[FC_VISIBLE_NPCS];
    int visible = fc_visible_npc_indices(state, visible_indices);
    if (slot < 0 || slot >= visible) return -1;
    return visible_indices[slot];
}

/* ======================================================================== */
/* Process player actions                                                    */
/* ======================================================================== */

static void record_player_action_selection(
    FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
    int act_move = actions[0];
    int act_attack = actions[1];
    int act_prayer = actions[2];
    int invalid_classes[FC_INVALID_ACTION_CLASS_COUNT];

    if (state->npcs_remaining > 0) {
        if (act_move == FC_MOVE_IDLE) {
            state->ep_action_move_idle_ticks++;
        } else if (act_move >= FC_MOVE_WALK_N && act_move < FC_MOVE_RUN_N) {
            state->ep_action_move_walk_ticks++;
        } else if (act_move >= FC_MOVE_RUN_N && act_move < FC_MOVE_DIM) {
            state->ep_action_move_run_ticks++;
        }

        if (act_attack == FC_ATTACK_NONE) {
            state->ep_action_attack_none_ticks++;
        } else {
            state->ep_action_attack_target_ticks++;
        }

        if (act_prayer == 0) {
            state->ep_action_prayer_noop_ticks++;
        } else {
            state->ep_action_prayer_cmd_ticks++;
        }
    }

    fc_action_invalid_classes(state, actions, invalid_classes);
    for (int i = 0; i < FC_INVALID_ACTION_CLASS_COUNT; i++) {
        state->invalid_action_class_this_tick[i] = invalid_classes[i];
        if (invalid_classes[i]) {
            state->invalid_action_this_tick = 1;
            state->ep_invalid_action_classes[i]++;
        }
    }
}

static void apply_player_prayer_action(
    FcState* state, int action, FcPrayerTransition* transition) {
    FcPlayer* player = &state->player;

    *transition = fc_prayer_apply_action(player, action);
    state->render_events.prayer_prior = transition->prior_prayer;
    state->render_events.prayer_final = transition->actual_final_prayer;
    state->render_events.prayer_off_performed = transition->off_performed;
    state->render_events.prayer_on_succeeded = transition->on_succeeded;
    state->render_events.prayer_flick_performed =
        transition->explicit_off_then_on &&
        transition->off_performed &&
        transition->on_succeeded;
    if (transition->final_state_changed ||
        (transition->explicit_off_then_on &&
         transition->off_performed && transition->on_succeeded)) {
        player->prayer_changed_this_tick = 1;
    }
}

static void apply_player_supplies(FcState* state, int eat_action,
                                  int drink_action) {
    FcPlayer* player = &state->player;

    if (eat_action != FC_EAT_NONE &&
        fc_eat_action_valid(state, eat_action)) {
        int heal = eat_action == FC_EAT_SHARK ? 200 : 180;
        int* cooldown_timer = eat_action == FC_EAT_SHARK
            ? &player->food_timer : &player->combo_timer;
        int cooldown = eat_action == FC_EAT_SHARK
            ? FC_FOOD_COOLDOWN_TICKS : FC_COMBO_EAT_TICKS;
        int pre_eat_hp = player->current_hp;
        state->ep_food_pre_hp_sum += pre_eat_hp;
        int hp_missing = player->max_hp - player->current_hp;
        if (heal > hp_missing) state->ep_food_overhealed++;
        state->ep_food_eaten++;
        player->total_food_eaten++;
        player->current_hp += heal;
        if (player->current_hp > player->max_hp) {
            player->current_hp = player->max_hp;
        }
        fc_items_consume(player, 0);
        *cooldown_timer = cooldown;
        player->food_eaten_this_tick = 1;
        state->food_used_this_tick = 1;
    }

    if (drink_action == FC_DRINK_PRAYER_POT &&
        fc_drink_action_valid(state, drink_action)) {
        int pre_drink_prayer = player->current_prayer;
        state->ep_pot_pre_prayer_sum += pre_drink_prayer;
        int prayer_missing = player->max_prayer - player->current_prayer;
        state->ep_pots_used++;
        if (player->current_prayer > player->max_prayer / 5) {
            state->ep_pots_wasted++;
        }
        player->total_potions_used++;
        int restore = fc_prayer_potion_restore(FC_PLAYER_PRAYER_LVL);
        if (restore > prayer_missing) state->ep_pots_overrestored++;
        player->current_prayer += restore;
        if (player->current_prayer > player->max_prayer) {
            player->current_prayer = player->max_prayer;
        }
        fc_items_consume(player, 1);
        player->potion_timer = FC_POTION_COOLDOWN_TICKS;
        player->potion_used_this_tick = 1;
        state->prayer_potion_used_this_tick = 1;
    }
    player->selected_food_slot = player->selected_potion_slot = -1;
}

static void prepare_player_interaction(FcState* state, int explicit_move,
                                       int explicit_attack,
                                       int requested_attack_idx) {
    FcPlayer* player = &state->player;

    /* Explicit movement starts a fresh movement intent before auto-attack can
     * consume a stale route or combat approach. */
    if (explicit_move) {
        player->route_len = 0;
        player->route_idx = 0;
        player->approach_target = 0;
        player->approach_target_x = -1;
        player->approach_target_y = -1;
        player->approach_target_size = 0;
        if (!explicit_attack) {
            player->attack_target_idx = -1;
        }
    }

    /* Target selection precedes movement so movement cannot rebind a slot or
     * make the selected attack valid retroactively. */
    if (explicit_attack && requested_attack_idx >= 0 &&
        state->npcs[requested_attack_idx].active &&
        !state->npcs[requested_attack_idx].is_dead) {
        if (player->attack_target_idx != requested_attack_idx) {
            player->approach_target_x = -1;
            player->approach_target_y = -1;
            player->approach_target_size = 0;
        }
        player->attack_target_idx = requested_attack_idx;
        player->approach_target = explicit_move ? 0 : 1;
    }
}

static void launch_player_attack(FcState* state, FcNpc* target, int distance) {
    FcPlayer* player = &state->player;
    int melee = player->weapon_kind == FC_WEAPON_UNARMED;
    /* Unarmed Punch is accurate/crush (+3 Attack). Fight Caves NPCs all
     * have zero crush defence bonus, including the ranged-resistant healers. */
    int att_roll = melee ? (player->attack_level + 11) *
        (player->melee_attack_bonus + 64) : fc_player_ranged_attack_roll(player, target);
    const FcNpcStats* target_stats = fc_npc_get_stats(target->npc_type);
    int def_roll = fc_npc_def_roll(target_stats->def_level,
                                   melee ? 0 : target_stats->ranged_def_bonus);
    float chance = fc_hit_chance(att_roll, def_roll);
    int hit = fc_rng_float(state) < chance ? 1 : 0;
    int final_max_hit_hp = melee ? (320 + (player->strength_level + 8) *
        (player->melee_strength_bonus + 64)) / 640 :
        fc_player_ranged_final_max_hit_hp(player, target);
    int damage = hit
        ? fc_roll_player_damage_tenths(state, final_max_hit_hp) : 0;
    int delay = melee ? 1 : fc_ranged_hit_delay(distance);

    fc_queue_pending_hit(target->pending_hits, &target->num_pending_hits,
                         FC_MAX_PENDING_HITS, damage, delay,
                         melee ? ATTACK_MELEE : ATTACK_RANGED, -1, 0);
    state->attack_attempt_this_tick = 1;
    state->render_events.player_attack_fired = 1;
    state->render_events.player_attack_source_x = player->x;
    state->render_events.player_attack_source_y = player->y;
    state->render_events.player_attack_target_npc_slot =
        player->attack_target_idx;
    state->render_events.player_attack_target_x = target->x;
    state->render_events.player_attack_target_y = target->y;
    state->render_events.player_attack_target_size = target->size;
    state->render_events.player_attack_hit_delay_ticks = delay;
    if (target->npc_type > NPC_NONE && target->npc_type < NPC_TYPE_COUNT) {
        state->ep_attack_cycles_to_npc_type[target->npc_type]++;
    }
    player->attack_timer = player->weapon_speed;
    if (player->weapon_uses_ammo && player->ammo_count > 0) {
        fc_items_spend_ammo(player);
    }
    player->hit_landed_this_tick = 1;
}

static void record_player_target_held(FcState* state, const FcNpc* target) {
    if (target->npc_type > NPC_NONE && target->npc_type < NPC_TYPE_COUNT) {
        state->ep_target_ticks_by_npc_type[target->npc_type]++;
    }
    state->ep_target_held_ticks++;
}

static int process_player_target(FcState* state,
                                 int explicit_directional_move,
                                 int explicit_tile_move) {
    FcPlayer* player = &state->player;
    int metrics_recorded = 0;

    /* Like Void CombatMovement: approach until the current target is in range,
     * then attack on cooldown and remain stationary for this tick. */
    if (player->attack_target_idx < 0 ||
        (player->weapon_uses_ammo && player->ammo_count <= 0)) {
        return metrics_recorded;
    }

    FcNpc* target = &state->npcs[player->attack_target_idx];
    if (!target->active || target->is_dead) {
        player->attack_target_idx = -1;
        player->approach_target = 0;
        player->approach_target_x = -1;
        player->approach_target_y = -1;
        player->approach_target_size = 0;
        return metrics_recorded;
    }

    int dist = fc_distance_to_npc(player->x, player->y, target);
    int weapon_range = player->weapon_range;
    int has_los = fc_has_los_between_areas(
        player->x, player->y, 1,
        target->x, target->y, target->size, state->los_flags);
    int target_can_fire = dist > 0 && dist <= weapon_range && has_los;
    if (player->weapon_kind == FC_WEAPON_UNARMED) {
        weapon_range = FC_ROUTE_MELEE_RANGE;
        target_can_fire = fc_npc_can_melee_player(player->x, player->y,
            target->x, target->y, target->size, state->walkable, state->movement_flags);
    }
    int target_ready = player->attack_timer <= 0;

    record_player_target_held(state, target);
    metrics_recorded = 1;
    if (target_can_fire) {
        state->ep_target_in_range_los_ticks++;
        if (!target_ready) {
            state->ep_attack_cooldown_wait_ticks++;
        }
    } else {
        state->ep_target_out_of_range_or_los_ticks++;
    }

    int route_endpoint_can_fire = 0;
    int target_moved =
        player->approach_target_x != target->x ||
        player->approach_target_y != target->y ||
        player->approach_target_size != target->size;
    if (player->route_idx < player->route_len) {
        int endpoint = player->route_len - 1;
        int rx = player->route_x[endpoint];
        int ry = player->route_y[endpoint];
        int route_dist = fc_distance_between_areas(
            rx, ry, 1, target->x, target->y, target->size);
        route_endpoint_can_fire = route_dist > 0 &&
            route_dist <= weapon_range &&
            fc_has_los_between_areas(
                rx, ry, 1, target->x, target->y, target->size,
                state->los_flags);
        if (player->weapon_kind == FC_WEAPON_UNARMED)
            route_endpoint_can_fire = fc_npc_can_melee_player(rx, ry,
                target->x, target->y, target->size, state->walkable, state->movement_flags);
    }

    if (!target_can_fire && player->approach_target &&
        (target_moved || player->route_idx >= player->route_len ||
         !route_endpoint_can_fire) &&
        !explicit_directional_move && !explicit_tile_move) {
        /* Rebuild against the target's current rectangle whenever the queued
         * endpoint is no longer a valid firing tile. */
        player->route_len = fc_pathfind_attack_position(
            player->x, player->y, target->x, target->y, target->size,
            weapon_range, state->walkable, state->movement_flags,
            state->los_flags, player->route_x, player->route_y, FC_MAX_ROUTE);
        player->route_idx = 0;
        player->approach_target_x = target->x;
        player->approach_target_y = target->y;
        player->approach_target_size = target->size;
    }

    float target_x = (float)target->x + (float)target->size * 0.5f;
    float target_y = (float)target->y + (float)target->size * 0.5f;
    set_player_facing_from_delta(
        player, target_x - ((float)player->x + 0.5f),
        target_y - ((float)player->y + 0.5f));

    if (target_can_fire && target_ready) {
        launch_player_attack(state, target, dist);
    }

    if (target_can_fire && target_ready &&
        !state->attack_attempt_this_tick) {
        state->ep_ready_but_no_attack_ticks++;
    }
    return metrics_recorded;
}

static int process_player_movement(FcState* state, int move_action,
                                   int target_x_action, int target_y_action,
                                   int explicit_move, int explicit_attack) {
    FcPlayer* player = &state->player;
    int moved_steps = 0;

    /* If no attack fired, explicit movement replaces the combat interaction.
     * A fired attack wins the conflict and keeps its target for this tick. */
    if (explicit_move && !state->attack_attempt_this_tick) {
        player->approach_target = 0;
        if (explicit_attack) {
            player->attack_target_idx = -1;
        }
    }

    if (!state->attack_attempt_this_tick &&
        target_x_action > 0 && target_y_action > 0) {
        int target_x = target_x_action - 1;
        int target_y = target_y_action - 1;
        if (target_x < FC_ARENA_WIDTH && target_y < FC_ARENA_HEIGHT) {
            player->route_len = fc_pathfind_bfs_move_near(
                player->x, player->y, target_x, target_y,
                state->walkable, state->movement_flags,
                player->route_x, player->route_y, FC_MAX_ROUTE);
            player->route_idx = 0;
            player->attack_target_idx = -1;
            player->approach_target = 0;
            player->approach_target_x = -1;
            player->approach_target_y = -1;
            player->approach_target_size = 0;
        }
    }

    /* Routes take priority over the directional action. Attacking suppresses
     * both forms of movement for this tick only. */
    if (state->attack_attempt_this_tick) {
        player->route_len = 0;
        player->route_idx = 0;
    } else if (player->route_idx < player->route_len) {
        int steps = player->is_running && player->run_energy > 0 ? 2 : 1;
        for (int i = 0;
             i < steps && player->route_idx < player->route_len; i++) {
            int next_x = player->route_x[player->route_idx];
            int next_y = player->route_y[player->route_idx];
            int dx = next_x - player->x;
            int dy = next_y - player->y;
            if (fc_footprint_step_walkable(
                    player->x, player->y, dx, dy, 1,
                    state->walkable, state->movement_flags)) {
                set_player_facing_from_delta(player, (float)dx, (float)dy);
                player->x = next_x;
                player->y = next_y;
                state->movement_this_tick = 1;
                record_player_move_waypoint(state);
                moved_steps++;
            } else {
                player->route_len = player->route_idx;
                break;
            }
            player->route_idx++;
        }
    } else if (move_action == FC_MOVE_IDLE) {
        state->idle_this_tick = 1;
    } else if (move_action >= FC_MOVE_WALK_N &&
               move_action <= FC_MOVE_RUN_NW) {
        int dx = FC_MOVE_DX[move_action];
        int dy = FC_MOVE_DY[move_action];
        int wants_run = move_action >= FC_MOVE_RUN_N;
        int max_steps = wants_run
            ? (fc_player_can_run(player) ? 2 : 0)
            : 1;
        int old_x = player->x;
        int old_y = player->y;
        int step_x[FC_MAX_RENDER_MOVE_WAYPOINTS];
        int step_y[FC_MAX_RENDER_MOVE_WAYPOINTS];
        int moved = fc_move_toward_traced(
            &player->x, &player->y, dx, dy, max_steps,
            state->walkable, state->movement_flags,
            step_x, step_y, FC_MAX_RENDER_MOVE_WAYPOINTS);
        if (moved > 0) {
            int recorded = moved;
            if (recorded > FC_MAX_RENDER_MOVE_WAYPOINTS) {
                recorded = FC_MAX_RENDER_MOVE_WAYPOINTS;
            }
            state->render_events.player_move_waypoint_count = recorded;
            for (int i = 0; i < recorded; i++) {
                state->render_events.player_move_waypoint_x[i] = step_x[i];
                state->render_events.player_move_waypoint_y[i] = step_y[i];
            }
            set_player_facing_from_delta(
                player, (float)(player->x - old_x),
                (float)(player->y - old_y));
            state->movement_this_tick = 1;
            player->is_running = moved >= 2 ? 1 : 0;
            moved_steps = moved;
        }
    }
    return moved_steps;
}

static void update_player_run_energy(FcPlayer* player, int moved_steps) {
    if (moved_steps >= 2) {
        player->run_energy -= FC_RUN_ENERGY_DRAIN;
        if (player->run_energy <= 0) {
            player->run_energy = 0;
            player->is_running = 0;
        }
        return;
    }

    player->run_energy += FC_RUN_ENERGY_RESTORE;
    if (player->run_energy > FC_RUN_ENERGY_MAX) {
        player->run_energy = FC_RUN_ENERGY_MAX;
    }
}

static void record_player_action_outcome(FcState* state, int was_attack_ready,
                                         int target_metrics_recorded) {
    FcPlayer* player = &state->player;

    if (state->npcs_remaining > 0 && !target_metrics_recorded) {
        if (player->attack_target_idx >= 0) {
            FcNpc* target = &state->npcs[player->attack_target_idx];
            if (target->active && !target->is_dead) {
                record_player_target_held(state, target);
            } else {
                state->ep_no_target_ticks++;
            }
        } else {
            state->ep_no_target_ticks++;
        }
    }

    if (was_attack_ready) {
        state->ep_attack_ready_ticks++;
        if (state->attack_attempt_this_tick) {
            state->ep_attack_attempt_ticks++;
        }
    }
}

static void process_player_actions(FcState* state,
                                   const int actions[FC_NUM_ACTION_HEADS],
                                   FcPrayerTransition* prayer_transition) {
    FcPlayer* p = &state->player;
    int was_attack_ready = (p->attack_timer <= 0 && state->npcs_remaining > 0);

    int act_move     = actions[0];
    int act_attack   = actions[1];
    int act_prayer   = actions[2];
    int act_eat      = actions[3];
    int act_drink    = actions[4];
    int act_target_x = actions[5];
    int act_target_y = actions[6];
    int explicit_directional_move = (act_move != FC_MOVE_IDLE);
    int explicit_tile_move = (act_target_x > 0 && act_target_y > 0);
    int explicit_move = explicit_directional_move || explicit_tile_move;
    int explicit_attack = (act_attack > FC_ATTACK_NONE);
    int requested_attack_idx = -1;
    int target_metrics_recorded = 0;
    record_player_action_selection(state, actions);

    /* Resolve attack slots against the pre-action NPC slot ordering. The action
     * was chosen from the previous observation, so movement later in this tick
     * must not rebind slot N to a different NPC identity. */
    if (explicit_attack) {
        requested_attack_idx = npc_slot_to_index(state, act_attack - 1);
    }

    /* Prayer remains instant and precedes supplies. */
    apply_player_prayer_action(state, act_prayer, prayer_transition);
    apply_player_supplies(state, act_eat, act_drink);

    prepare_player_interaction(state, explicit_move, explicit_attack,
                               requested_attack_idx);

    target_metrics_recorded = process_player_target(
        state, explicit_directional_move, explicit_tile_move);

    int moved_steps = process_player_movement(
        state, act_move, act_target_x, act_target_y,
        explicit_move, explicit_attack);
    update_player_run_energy(p, moved_steps);
    record_player_action_outcome(state, was_attack_ready,
                                 target_metrics_recorded);
}

/* ======================================================================== */
/* Decrement player timers                                                   */
/* ======================================================================== */

static void decrement_player_timers(FcPlayer* p) {
    if (p->attack_timer > 0) p->attack_timer--;
    if (p->food_timer > 0) p->food_timer--;
    if (p->potion_timer > 0) p->potion_timer--;
    if (p->combo_timer > 0) p->combo_timer--;
}

/* ======================================================================== */
/* Check terminal conditions                                                 */
/* ======================================================================== */

/* ======================================================================== */
/* Jad healer auto-spawn                                                     */
/* ======================================================================== */

/*
 * Jad healer spawn (from TzhaarFightCave.kt npcLevelChanged handler):
 *   Trigger: Jad HP drops below 150 HP.
 *   Spawns up to 4 Yt-HurKot in the five Fight Cave spawn regions other than
 *   north-east (fills missing slots: 4 - currently_alive).
 *   Respawn: Only after healers restore Jad to full HP and he crosses the
 *   threshold again. Crossing back above the threshold is not enough to re-arm.
 */
static void check_jad_healers(FcState* state) {
    if (state->current_wave != FC_NUM_WAVES) return;  /* only on wave 63 */

    /* Find Jad */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* jad = &state->npcs[i];
        if (jad->npc_type != NPC_TZTOK_JAD || !jad->active || jad->is_dead) continue;

        /* Re-arm respawns only after Jad has been healed all the way to full. */
        if (jad->current_hp >= jad->max_hp) {
            state->jad_healers_spawned = 0;
            return;
        }

        if (jad->current_hp >= FC_JAD_HEALER_THRESHOLD_HP_TENTHS) return;

        /* Below threshold — spawn healers if not already spawned this cycle */
        if (state->jad_healers_spawned) return;

        /* Count currently alive healers */
        int alive_healers = 0;
        for (int h = 0; h < FC_MAX_NPCS; h++) {
            if (state->npcs[h].active && !state->npcs[h].is_dead &&
                state->npcs[h].npc_type == NPC_YT_HURKOT) {
                alive_healers++;
            }
        }

        /* Spawn up to 4 total (fill missing slots) */
        int to_spawn = FC_JAD_NUM_HEALERS - alive_healers;
        int spawn_dirs[5] = {
            SPAWN_NORTH_WEST,
            SPAWN_SOUTH_WEST,
            SPAWN_SOUTH,
            SPAWN_SOUTH_EAST,
            SPAWN_CENTER,
        };
        for (int i = 4; i > 0; i--) {
            int j = fc_rng_int(state, i + 1);
            int tmp = spawn_dirs[i];
            spawn_dirs[i] = spawn_dirs[j];
            spawn_dirs[j] = tmp;
        }
        const FcNpcStats* healer_stats = fc_npc_get_stats(NPC_YT_HURKOT);
        int is_respawn_generation = state->jad_healer_spawn_generations > 0;
        int spawned = 0;
        for (int h = 0; h < to_spawn; h++) {
            int hx, hy;
            fc_spawn_position(spawn_dirs[h], &hx, &hy);

            if (!fc_spawn_find_available_footprint(
                    state, hx, hy, healer_stats->size, FC_ARENA_WIDTH - 1,
                    &hx, &hy)) {
                continue;
            }

            int slot = fc_spawn_npc_first_free(state, NPC_YT_HURKOT, hx, hy);
            if (slot < 0) break;
            state->npcs[slot].is_respawned_jad_healer =
                is_respawn_generation;
            state->npcs_remaining++;
            spawned++;
        }
        if (spawned > 0) {
            state->jad_healers_spawned = 1;
            state->jad_healer_spawn_generations++;
        }
        return;
    }
}

/* ======================================================================== */
/* Check terminal conditions                                                 */
/* ======================================================================== */

static void check_terminal(FcState* state) {
    if (state->terminal != TERMINAL_NONE) return;

    /* Player death */
    if (state->player.current_hp <= 0) {
        state->terminal = TERMINAL_PLAYER_DEATH;
        return;
    }

    /* Wave advancement (handles wave-clear and cave-complete) */
    fc_wave_check_advance(state);

    /* Jad healer spawn check */
    check_jad_healers(state);

    /* Tick cap */
    if (state->tick >= FC_MAX_EPISODE_TICKS) {
        state->terminal = TERMINAL_TICK_CAP;
    }
}

static void lock_pending_prayers_at_boundary(FcState* state) {
    FcPlayer* p = &state->player;
    for (int i = 0; i < p->num_pending_hits; i++) {
        FcPendingHit* hit = &p->pending_hits[i];
        if (!hit->active || hit->prayer_snapshot >= 0 ||
            hit->prayer_lock_tick < 0 ||
            state->tick < hit->prayer_lock_tick) {
            continue;
        }
        hit->prayer_snapshot = p->prayer;
    }
}

/* ======================================================================== */
/* Main tick entry point                                                     */
/* ======================================================================== */

void fc_tick(FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
    state->player.prayer_at_tick_start = state->player.prayer;
    FcPrayerTransition prayer_transition = {0};

    /* 1. Clear per-tick flags */
    clear_per_tick_flags(state);
    /* 2. Process player actions */
    process_player_actions(state, actions, &prayer_transition);

    /* 3. Decrement player timers */
    decrement_player_timers(&state->player);

    /* 4. Prayer drain */
    state->overhead_prayer_lost_this_tick =
        fc_prayer_drain_tick(&state->player,
                             state->player.prayer_at_tick_start,
                             &prayer_transition);
    state->prayer_lost_this_tick += state->overhead_prayer_lost_this_tick;

    /* 4b. HP regen (1 HP = 10 tenths every FC_HP_REGEN_INTERVAL ticks) */
    if (state->player.current_hp > 0 && state->player.current_hp < state->player.max_hp) {
        state->player.hp_regen_counter++;
        if (state->player.hp_regen_counter >= FC_HP_REGEN_INTERVAL) {
            state->player.hp_regen_counter = 0;
            state->player.current_hp += 10;  /* 1 HP in tenths */
            if (state->player.current_hp > state->player.max_hp) {
                state->player.current_hp = state->player.max_hp;
            }
        }
    }

    /* 5. NPC AI tick */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        fc_npc_tick(state, i);
    }

    /* 6. Resolve pending hits */
    fc_resolve_player_pending_hits(state);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (state->npcs[i].active) {
            fc_resolve_npc_pending_hits(state, i);
        }
    }

    /* 6b. Process death timers — dead NPCs stay visible briefly */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* n = &state->npcs[i];
        if (n->is_dead && n->active) {
            if (n->death_timer > 0) {
                n->death_timer--;
            } else {
                n->active = 0;  /* fully despawn */
            }
        }
    }

    /* 7. Check terminal */
    check_terminal(state);

    /* 8. Episode analytics */
    if (state->player.prayer_at_tick_start == PRAYER_PROTECT_MELEE)
        state->ep_ticks_pray_melee++;
    if (state->player.prayer_at_tick_start == PRAYER_PROTECT_RANGE)
        state->ep_ticks_pray_range++;
    if (state->player.prayer_at_tick_start == PRAYER_PROTECT_MAGIC)
        state->ep_ticks_pray_magic++;
    if (state->player.prayer_changed_this_tick)
        state->ep_prayer_switches++;
    if (state->current_wave >= 63)
        state->ep_reached_wave_63 = 1;

    /* 9. Increment tick */
    state->tick++;

    /* This is the pre-action boundary for the next policy decision. Jad's
     * T+2 lock must be visible before that observation and cannot be changed
     * by the action selected from it. */
    lock_pending_prayers_at_boundary(state);
}


/* Wave */
/*
 * fc_wave.c — 63-wave Fight Caves spawn table with 15 rotations.
 *
 * Wave data sourced from OSRS wiki + Kotlin archive TOML:
 *   archive/kotlin-final:runescape-rl/src/headless-env/data/minigame/
 *     tzhaar_fight_cave/tzhaar_fight_cave_waves.toml
 *
 * Pattern: each new NPC tier is introduced on waves 1,3,7,15,31,63.
 * The binary pattern gives 2^tier - 1 waves before each new tier.
 *
 * Spawn directions map to arena coordinates:
 *   SPAWN_SOUTH      → (32, 5)
 *   SPAWN_SOUTH_WEST → (8, 8)
 *   SPAWN_NORTH_WEST → (8, 55)
 *   SPAWN_SOUTH_EAST → (55, 8)
 *   SPAWN_CENTER     → (32, 32)
 */

/* ======================================================================== */
/* Wave table (63 waves × up to 6 NPCs)                                     */
/* ======================================================================== */

static const FcWaveEntry WAVE_TABLE[FC_NUM_WAVES] = {
    /* Wave  1 */ { {NPC_TZ_KIH, 0, 0, 0, 0, 0}, 1 },
    /* Wave  2 */ { {NPC_TZ_KIH, NPC_TZ_KIH, 0, 0, 0, 0}, 2 },
    /* Wave  3 */ { {NPC_TZ_KEK, 0, 0, 0, 0, 0}, 1 },
    /* Wave  4 */ { {NPC_TZ_KIH, NPC_TZ_KEK, 0, 0, 0, 0}, 2 },
    /* Wave  5 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, 0, 0, 0}, 3 },
    /* Wave  6 */ { {NPC_TZ_KEK, NPC_TZ_KEK, 0, 0, 0, 0}, 2 },
    /* Wave  7 */ { {NPC_TOK_XIL, 0, 0, 0, 0, 0}, 1 },
    /* Wave  8 */ { {NPC_TZ_KIH, NPC_TOK_XIL, 0, 0, 0, 0}, 2 },
    /* Wave  9 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TOK_XIL, 0, 0, 0}, 3 },
    /* Wave 10 */ { {NPC_TZ_KEK, NPC_TOK_XIL, 0, 0, 0, 0}, 2 },
    /* Wave 11 */ { {NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, 0, 0, 0}, 3 },
    /* Wave 12 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, 0, 0}, 4 },
    /* Wave 13 */ { {NPC_TZ_KEK, NPC_TZ_KEK, NPC_TOK_XIL, 0, 0, 0}, 3 },
    /* Wave 14 */ { {NPC_TOK_XIL, NPC_TOK_XIL, 0, 0, 0, 0}, 2 },
    /* Wave 15 */ { {NPC_YT_MEJKOT, 0, 0, 0, 0, 0}, 1 },
    /* Wave 16 */ { {NPC_TZ_KIH, NPC_YT_MEJKOT, 0, 0, 0, 0}, 2 },
    /* Wave 17 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_YT_MEJKOT, 0, 0, 0}, 3 },
    /* Wave 18 */ { {NPC_TZ_KEK, NPC_YT_MEJKOT, 0, 0, 0, 0}, 2 },
    /* Wave 19 */ { {NPC_TZ_KIH, NPC_TZ_KEK, NPC_YT_MEJKOT, 0, 0, 0}, 3 },
    /* Wave 20 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, NPC_YT_MEJKOT, 0, 0}, 4 },
    /* Wave 21 */ { {NPC_TZ_KEK, NPC_TZ_KEK, NPC_YT_MEJKOT, 0, 0, 0}, 3 },
    /* Wave 22 */ { {NPC_TOK_XIL, NPC_YT_MEJKOT, 0, 0, 0, 0}, 2 },
    /* Wave 23 */ { {NPC_TZ_KIH, NPC_TOK_XIL, NPC_YT_MEJKOT, 0, 0, 0}, 3 },
    /* Wave 24 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TOK_XIL, NPC_YT_MEJKOT, 0, 0}, 4 },
    /* Wave 25 */ { {NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, 0, 0, 0}, 3 },
    /* Wave 26 */ { {NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, 0, 0}, 4 },
    /* Wave 27 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, 0}, 5 },
    /* Wave 28 */ { {NPC_TZ_KEK, NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, 0, 0}, 4 },
    /* Wave 29 */ { {NPC_TOK_XIL, NPC_TOK_XIL, NPC_YT_MEJKOT, 0, 0, 0}, 3 },
    /* Wave 30 */ { {NPC_YT_MEJKOT, NPC_YT_MEJKOT, 0, 0, 0, 0}, 2 },
    /* Wave 31 */ { {NPC_KET_ZEK, 0, 0, 0, 0, 0}, 1 },
    /* Wave 32 */ { {NPC_TZ_KIH, NPC_KET_ZEK, 0, 0, 0, 0}, 2 },
    /* Wave 33 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 34 */ { {NPC_TZ_KEK, NPC_KET_ZEK, 0, 0, 0, 0}, 2 },
    /* Wave 35 */ { {NPC_TZ_KIH, NPC_TZ_KEK, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 36 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 37 */ { {NPC_TZ_KEK, NPC_TZ_KEK, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 38 */ { {NPC_TOK_XIL, NPC_KET_ZEK, 0, 0, 0, 0}, 2 },
    /* Wave 39 */ { {NPC_TZ_KIH, NPC_TOK_XIL, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 40 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TOK_XIL, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 41 */ { {NPC_TZ_KEK, NPC_TOK_XIL, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 42 */ { {NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 43 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, NPC_KET_ZEK, 0}, 5 },
    /* Wave 44 */ { {NPC_TZ_KEK, NPC_TZ_KEK, NPC_TOK_XIL, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 45 */ { {NPC_TOK_XIL, NPC_TOK_XIL, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 46 */ { {NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0, 0, 0}, 2 },
    /* Wave 47 */ { {NPC_TZ_KIH, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 48 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 49 */ { {NPC_TZ_KEK, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 50 */ { {NPC_TZ_KIH, NPC_TZ_KEK, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 51 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, NPC_YT_MEJKOT, NPC_KET_ZEK, 0}, 5 },
    /* Wave 52 */ { {NPC_TZ_KEK, NPC_TZ_KEK, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 53 */ { {NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 54 */ { {NPC_TZ_KIH, NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 55 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK, 0}, 5 },
    /* Wave 56 */ { {NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 57 */ { {NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK, 0}, 5 },
    /* Wave 58 */ { {NPC_TZ_KIH, NPC_TZ_KIH, NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK}, 6 },
    /* Wave 59 */ { {NPC_TZ_KEK, NPC_TZ_KEK, NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK, 0}, 5 },
    /* Wave 60 */ { {NPC_TOK_XIL, NPC_TOK_XIL, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0}, 4 },
    /* Wave 61 */ { {NPC_YT_MEJKOT, NPC_YT_MEJKOT, NPC_KET_ZEK, 0, 0, 0}, 3 },
    /* Wave 62 */ { {NPC_KET_ZEK, NPC_KET_ZEK, 0, 0, 0, 0}, 2 },
    /* Wave 63 */ { {NPC_TZTOK_JAD, 0, 0, 0, 0, 0}, 1 },
};

static const int WAVE_ROTATIONS[FC_NUM_WAVES][FC_NUM_ROTATIONS][FC_MAX_SPAWNS_PER_WAVE] = {
    { /* Wave 1 */
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
    },
    { /* Wave 2 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
    },
    { /* Wave 3 */
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
    },
    { /* Wave 4 */
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 5 */
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
    },
    { /* Wave 6 */
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 7 */
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
    },
    { /* Wave 8 */
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
    },
    { /* Wave 9 */
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
    },
    { /* Wave 10 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
    },
    { /* Wave 11 */
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
    },
    { /* Wave 12 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
    },
    { /* Wave 13 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
    },
    { /* Wave 14 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 15 */
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
    },
    { /* Wave 16 */
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 17 */
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
    },
    { /* Wave 18 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
    },
    { /* Wave 19 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
    },
    { /* Wave 20 */
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
    },
    { /* Wave 21 */
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
    },
    { /* Wave 22 */
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 23 */
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
    },
    { /* Wave 24 */
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
    },
    { /* Wave 25 */
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
    },
    { /* Wave 26 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
    },
    { /* Wave 27 */
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
    },
    { /* Wave 28 */
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
    },
    { /* Wave 29 */
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
    },
    { /* Wave 30 */
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 31 */
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
    },
    { /* Wave 32 */
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
    },
    { /* Wave 33 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
    },
    { /* Wave 34 */
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 35 */
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
    },
    { /* Wave 36 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
    },
    { /* Wave 37 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
    },
    { /* Wave 38 */
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
    },
    { /* Wave 39 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
    },
    { /* Wave 40 */
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
    },
    { /* Wave 41 */
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
    },
    { /* Wave 42 */
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
    },
    { /* Wave 43 */
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0},
    },
    { /* Wave 44 */
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
    },
    { /* Wave 45 */
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
    },
    { /* Wave 46 */
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
    },
    { /* Wave 47 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
    },
    { /* Wave 48 */
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
    },
    { /* Wave 49 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
    },
    { /* Wave 50 */
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
    },
    { /* Wave 51 */
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
    },
    { /* Wave 52 */
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
    },
    { /* Wave 53 */
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0, 0},
    },
    { /* Wave 54 */
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
    },
    { /* Wave 55 */
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0},
    },
    { /* Wave 56 */
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
    },
    { /* Wave 57 */
        {SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0},
        {SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
    },
    { /* Wave 58 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST},
    },
    { /* Wave 59 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0},
    },
    { /* Wave 60 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, 0, 0},
    },
    { /* Wave 61 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, 0, 0, 0},
    },
    { /* Wave 62 */
        {SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_CENTER, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, SPAWN_SOUTH, 0, 0, 0, 0},
        {SPAWN_CENTER, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH, SPAWN_NORTH_WEST, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, SPAWN_SOUTH, 0, 0, 0, 0},
    },
    { /* Wave 63 */
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_EAST, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_CENTER, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_NORTH_WEST, 0, 0, 0, 0, 0},
        {SPAWN_SOUTH, 0, 0, 0, 0, 0},
    },
};

/* ======================================================================== */
/* Spawn position from direction                                             */
/* ======================================================================== */

/*
 * Spawn positions derived from RSPS tzhaar_fight_cave.areas.toml.
 * World coords converted to arena-local (world - 2368, world - 5056).
 * Centers of each spawn area, verified walkable for all NPC sizes (1-5).
 *
 *   NORTH_WEST: world [2378,2385]x[5102,5109] → local [10,17]x[46,53] center (13,49)
 *   SOUTH_WEST: world [2379,2385]x[5070,5076] → local [11,17]x[14,20] center (14,17)
 *   SOUTH:      world [2402,2408]x[5070,5076] → local [34,40]x[14,20] center (37,17)
 *   SOUTH_EAST: world [2416,2422]x[5080,5086] → local [48,54]x[24,30] center (51,27)
 *   CENTER:     world [2397,2403]x[5085,5091] → local [29,35]x[29,35] center (32,32)
 */
void fc_spawn_position(int spawn_dir, int* x, int* y) {
    switch (spawn_dir) {
        case SPAWN_SOUTH:      *x = 37; *y = 17; break;
        case SPAWN_SOUTH_WEST: *x = 14; *y = 17; break;
        case SPAWN_NORTH_WEST: *x = 13; *y = 49; break;
        case SPAWN_SOUTH_EAST: *x = 51; *y = 27; break;
        case SPAWN_CENTER:     *x = 32; *y = 32; break;
        default:               *x = 32; *y = 32; break;
    }
}

/* ======================================================================== */
/* Wave table accessors                                                      */
/* ======================================================================== */

static const FcWaveEntry* fc_wave_get(int wave_num) {
    if (wave_num < 1 || wave_num > FC_NUM_WAVES) return &WAVE_TABLE[0];
    return &WAVE_TABLE[wave_num - 1];
}

static int fc_wave_spawn_dir(int wave_num, int rotation, int npc_index) {
    if (wave_num < 1 || wave_num > FC_NUM_WAVES) return SPAWN_CENTER;
    if (rotation < 0 || rotation >= FC_NUM_ROTATIONS) return SPAWN_CENTER;
    if (npc_index < 0 || npc_index >= FC_MAX_SPAWNS_PER_WAVE) return SPAWN_CENTER;
    return WAVE_ROTATIONS[wave_num - 1][rotation][npc_index];
}

/* ======================================================================== */
/* Spawn wave NPCs into the arena                                            */
/* ======================================================================== */

void fc_wave_spawn(FcState* state, int wave_num) {
    const FcWaveEntry* wave = fc_wave_get(wave_num);
    int rotation = state->rotation_id;

    for (int i = 0; i < wave->num_spawns; i++) {
        int npc_type = wave->npc_types[i];
        int dir = fc_wave_spawn_dir(wave_num, rotation, i);
        int sx, sy;
        fc_spawn_position(dir, &sx, &sy);

        const FcNpcStats* stats = fc_npc_get_stats(npc_type);
        /* Wave spawning retains its radius-five fallback policy: if no valid
         * footprint is found, the original regional tile is still used. */
        (void)fc_spawn_find_available_footprint(
            state, sx, sy, stats->size, 5, &sx, &sy);

        if (fc_spawn_npc_first_free(state, npc_type, sx, sy) < 0) continue;
        /* Tz-Kek counts as 2 in wave remaining (pre-counts the split).
         * RSPS: ids.sumOf { if (it == "tz_kek") 2 else 1 } */
        state->npcs_remaining += (npc_type == NPC_TZ_KEK) ? 2 : 1;
    }
}

/* ======================================================================== */
/* Wave advancement                                                          */
/* ======================================================================== */

void fc_wave_record_current_duration(FcState* state) {
    int wave_ticks = state->tick - state->wave_start_tick;
    if (wave_ticks > state->ep_max_wave_ticks) {
        state->ep_max_wave_ticks = wave_ticks;
        state->ep_max_wave_ticks_wave = state->current_wave;
    }
}

int fc_wave_check_advance(FcState* state) {
    /* Don't advance if wave hasn't started or NPCs still alive */
    if (state->current_wave <= 0) return 0;
    if (state->npcs_remaining > 0) return 0;
    if (state->terminal != TERMINAL_NONE) return 0;

    state->wave_just_cleared = 1;

    fc_wave_record_current_duration(state);

    /* Check if all waves complete */
    if (state->current_wave >= FC_NUM_WAVES) {
        state->terminal = TERMINAL_CAVE_COMPLETE;
        return 0;
    }

    /* Advance to next wave */
    state->current_wave++;
    state->jad_healers_spawned = 0;
    state->jad_healer_spawn_generations = 0;
    state->wave_start_tick = state->tick;
    fc_wave_spawn(state, state->current_wave);

    return 1;
}


#endif
