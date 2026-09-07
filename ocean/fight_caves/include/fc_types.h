#ifndef FC_TYPES_H
#define FC_TYPES_H

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
#include "fc_player_init.h"

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

#endif /* FC_TYPES_H */
