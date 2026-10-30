#ifndef FIGHT_CAVES_CORE_H
#define FIGHT_CAVES_CORE_H

/* Define FC_CORE_IMPLEMENTATION in the owning C translation unit.
 * Other consumers (including the C++ trainer) receive declarations only. */
#if defined(FC_CORE_IMPLEMENTATION) && defined(__cplusplus)
#error "Compile the Fight Caves core implementation as C, not C++."
#endif

#include <stdint.h>

/* Player Init */

/*
 * Player skill, equipment, and consumable configuration shared by the core,
 * training adapter and viewer. Combat bonuses are derived from item definitions.
 */

/* The one player loadout, Masori (f) with a Twisted bow. Its id is stored in
 * the hashed FcState.active_loadout and keeps the value it had when this was
 * one of nine presets. */
typedef enum {
    FC_LOADOUT_SOTA_TBOW = 1
} FcLoadoutId;

#define FC_LOADOUT_EQUIP_MAX 12

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
    int combat_style_profile;   /* the viewer's combat-tab weapon category */
    int max_hp, max_prayer;     /* tenths */
    int attack_lvl, strength_lvl, defence_lvl;
    int ranged_lvl, prayer_lvl, magic_lvl;
    int ammo;
    int equipment_count;
    int equipment[FC_LOADOUT_EQUIP_MAX];
} FcLoadout;

/* Stored in the hashed FcPlayer.weapon_kind, so the values stay as they were. */
typedef enum {
    FC_WEAPON_TWISTED_BOW = 1,
    FC_WEAPON_UNARMED = 3
} FcWeaponKind;

/* Types */

/*
 * All state is flat fields on structs: no nested pointers and no heap
 * allocation per tick, which keeps copies, hashing and observation writing
 * linear. fc_reset zeroes FcState with memset, so zero must be a safe default
 * for every field it does not set.
 */

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
    ATTACK_MAGIC  = 3,
    ATTACK_STYLE_COUNT = 4
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


/* NPC spawn direction for wave rotations */
typedef enum {
    SPAWN_SOUTH      = 0,
    SPAWN_SOUTH_WEST = 1,
    SPAWN_NORTH_WEST = 2,
    SPAWN_SOUTH_EAST = 3,
    SPAWN_CENTER     = 4
} FcSpawnDir;

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
#define FC_VISIBLE_NPCS   8   /* max NPCs in observation (FC_OBS_NPC_SLOTS) */
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

/*
 * Pending hit: a projectile in flight or a delayed melee hit. Attacks queue
 * one with a delay in ticks; pending hits resolve after the NPCs act, so a
 * delay of 1 lands on the attack tick. Melee uses 1; ranged and magic use the
 * projectile travel time from fc_ranged_hit_delay (player) or
 * fc_npc_hit_delay (NPCs).
 *
 * Prayer blocking is locked into the pending hit before impact:
 *   - normal Fight Caves NPCs snapshot prayer on the attack tick
 *   - Jad's ranged and magic snapshot two ticks after the attack
 * The projectile/hitsplat can still land later, but prayer no longer re-checks
 * the live player prayer on impact.
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

#define FC_INVENTORY_SLOTS 28
#define FC_EQUIPMENT_SLOTS 14

typedef struct {
    int item_id;                 /* 0 denotes an empty slot */
    int quantity;
    int charges;                 /* always 0: no item holds charges */
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
    int crystal_piece_mask;     /* always 0: no crystal armour */
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
     * Set once on click, consumed one step per tick (two when running).
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
    int hit_landed_this_tick;
    int food_eaten_this_tick;
    int potion_used_this_tick;
    int prayer_changed_this_tick;

    FcItemStack inventory[FC_INVENTORY_SLOTS];
    FcItemStack equipment[FC_EQUIPMENT_SLOTS];
    int melee_attack_bonus, melee_strength_bonus;
    int selected_food_slot, selected_potion_slot;
} FcPlayer;

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

/* Wave spawn table row */
#define FC_MAX_SPAWNS_PER_WAVE 6

typedef struct {
    int npc_types[FC_MAX_SPAWNS_PER_WAVE];  /* FcNpcType per spawn */
    int num_spawns;
} FcWaveEntry;

/* Viewer-facing snapshot of the player or an NPC, filled each tick by
 * fc_fill_render_entities. */
typedef struct {
    int entity_type;    /* FcEntityType */
    int npc_type;       /* FcNpcType (0 for player) */
    int x, y;
    int size;           /* tile size */
    int current_hp, max_hp;
    int is_dead;
    int died_this_tick;

    int npc_slot;       /* NPC array index (for stable interpolation lookup) */
} FcRenderEntity;

#define FC_MAX_RENDER_ENTITIES (1 + FC_MAX_NPCS)  /* player + NPCs */

/* Per-tick rendering events */
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
    int prayer_lock_tick;
    int hit_queued;
} FcRenderNpcAttack;

typedef struct {
    int target_entity_type;  /* FcEntityType */
    int target_npc_slot;     /* -1 when the player is the target */
    int source_npc_slot;     /* -1 when the player is the source */
    int attack_style;
    int damage;
} FcRenderHit;

/*
 * Authoritative, read-only facts captured while a simulation tick executes.
 * These events let renderers reproduce transitions whose intermediate state
 * is no longer present in the final FcState snapshot. They do not participate
 * in combat, observations, rewards, or action validation.
 */
typedef struct {
    /* A successful flick can begin and end on the same prayer, so the final
     * player.prayer alone cannot show its off edge. */
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

/* Top-level simulation state */
typedef struct {
    FcPlayer player;
    FcNpc npcs[FC_MAX_NPCS];

    /* Always FC_LOADOUT_SOTA_TBOW, the only loadout. */
    int active_loadout;

    /* Viewer-facing transition facts for the most recently completed tick. */
    FcRenderEvents render_events;

    /* Wave progression */
    int current_wave;       /* 1-indexed: 1..63. 0 = not started */
    int rotation_id;        /* 0..14, selected at episode start */
    int npcs_remaining;     /* count of active (alive) NPCs in current wave */
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

    /* Per-tick aggregated event flags (for rewards and observations) */
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
    int movement_this_tick;
    int idle_this_tick;
    int food_used_this_tick;
    int prayer_potion_used_this_tick;
    int jad_heal_procs_this_tick;   /* number of Yt-HurKot heal procs that restored Jad HP */
    int npc_heal_procs_this_tick;   /* number of NPC heal procs that restored any NPC HP */
    int npc_heal_amount_this_tick;  /* total NPC HP restored this tick */
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
    int ep_wrong_prayer_hits;   /* hits where prayer active but wrong type */
    int ep_reached_wave_63;     /* 1 if episode reached Jad wave */
    int ep_jad_killed;          /* 1 if Jad died at any point this episode */
} FcState;


/* Contracts */

/*
 * Observation, action and mask contracts.
 *
 * SINGLE SOURCE OF TRUTH for all buffer layouts.
 *
 * All observations are float32, normalized to [0,1]. All actions are int32 per
 * head.
 */

/*
 * Observation layout. fc_write_obs writes the FC_POLICY_OBS_SIZE (286) policy
 * observations. Puffer's model input is 320 floats: those observations
 * followed by FC_PUFFER_MASK_SIZE (34) mask bits for the no-supplies policy
 * heads. Those same 34 bits are also supplied through PufferLib's native
 * action-mask channel; the masks for core-only action heads are not exposed
 * to the model.
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
/* 10-15 are always 0. They held incoming-hit counts, which training zeroed;
 * the slots stay so the policy's 286-float layout does not change. */
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
 * are not included in the observation. The agent cannot select overflow NPCs
 * with the ATTACK head, although auto-retaliate can still target them.
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
#define FC_NPC_KILL_REWARD_ELIGIBLE    30  /* 1 unless this NPC is a respawned Jad healer */

#define FC_OBS_NPC_TOTAL        (FC_OBS_NPC_STRIDE * FC_OBS_NPC_SLOTS)  /* 248 */

/* --- Wave/meta features (15 floats) --- */
#define FC_OBS_META_START       (FC_OBS_NPC_START + FC_OBS_NPC_TOTAL)  /* 271 */
#define FC_OBS_META_WAVE        0   /* current_wave / NUM_WAVES */
#define FC_OBS_META_ROTATION    1   /* rotation_id / NUM_ROTATIONS */
#define FC_OBS_META_REMAINING   2   /* npcs_remaining / MAX_NPCS */
#define FC_OBS_META_PRAY_DRAIN  3   /* prayer_drain_counter / drain_resistance */
/* 4-6 are always 0, like the player block's 10-15. */
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

/*
 * Action space: 7 canonical core heads, shared by headless training, policy
 * replay and the playable viewer. The viewer translates human input into
 * these heads: a clicked tile becomes heads 5 and 6, a clicked NPC its head 1
 * slot. Equipment and inventory changes between ticks use the item
 * transactions (fc_equip_item and friends) instead.
 */

#define FC_NUM_ACTION_HEADS     7

/* Head 0: MOVE — directional tile movement (low-level) */
/*
 *   0     = idle (no movement)
 *   1-8   = walk 1 tile (N, NE, E, SE, S, SW, W, NW)
 *   9-16  = run 2 tiles (N, NE, E, SE, S, SW, W, NW)
 *
 * For fine-grained per-tick control. A directional step cancels a route in
 * progress; it is ignored when heads 5 and 6 set a route in the same tick.
 */
#define FC_MOVE_DIM             17
#define FC_MOVE_IDLE             0
#define FC_MOVE_WALK_N           1
#define FC_MOVE_RUN_N            9
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
 * A hit is checked against the prayer locked into it (see FcPendingHit), so
 * the switch must happen before that lock.
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
 * The route is consumed one step per tick (or two if running). A directional
 * action (head 0) on a later tick cancels it.
 *
 * If the exact target is unwalkable or unreachable, native move-near chooses
 * the best reachable endpoint within ten tiles. If none exists, it is a no-op.
 */
#define FC_MOVE_TARGET_X_DIM    65  /* 0=no-op, 1-64=tile x 0-63 */

/* Head 6: MOVE_TARGET_Y — BFS pathfinding target Y coordinate */
#define FC_MOVE_TARGET_Y_DIM    65  /* 0=no-op, 1-64=tile y 0-63 */

/* Head dimensions */
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

/*
 * Action mask. Per-tick binary mask: 1.0 = valid, 0.0 = invalid.
 * One float per action value per head.
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


/* Npc */

/* NPC stat table entry (one per NPC_TYPE). The per-style arrays are indexed
 * by FcAttackStyle; index ATTACK_NONE is unused. */
typedef struct {
    int max_hp;
    int attack_style;       /* FcAttackStyle: primary style (ranged/magic for dual-mode) */
    int attack_speed;       /* ticks between attacks */
    int attack_range;       /* primary attack range (1 for melee-only, 14 for ranged/magic) */
    int max_hit[ATTACK_STYLE_COUNT];      /* whole HP; 0 = no attack of that style */
    int attack_level[ATTACK_STYLE_COUNT]; /* Attack, Ranged and Magic levels; Magic
                                           * is also the Twisted-bow target input */
    int attack_bonus[ATTACK_STYLE_COUNT];
    int def_level;          /* NPC defence level (for player attack accuracy) */
    int ranged_def_bonus;   /* NPC equipment defence vs Ranged */
    int melee_attack_type;  /* FcAttackType */
    int size;               /* tile footprint */
    int movement_speed;     /* 1=walk, 2=run */
    int prayer_drain;       /* base prayer drain in tenths (Tz-Kih specific) */
    int heal_amount;        /* HP healed per proc */
    int heal_interval;      /* ticks between independent Yt-HurKot heals */
} FcNpcStats;


/* Pathfinding */

/* Attack-route range 0 means cardinal melee contact, not ranged LOS. */
#define FC_ROUTE_MELEE_RANGE 0


/* Prayer */

typedef struct {
    int changed;   /* the overhead prayer differs from before the action */
    int flicked;   /* a flick action turned an active prayer off and back on */
} FcPrayerTransition;


/* Reward */

typedef struct {
    float w_progress;          /* reward per required-work unit removed */
    float w_damage_taken;
    float w_cave_complete;
    float w_player_death;
    float w_prayer_lost;
    float w_invalid_action;
    float w_tick_penalty;

    float shape_npc_heal_penalty;
    float shape_no_progress_penalty_1;
    float shape_no_progress_penalty_2;
    float shape_no_progress_penalty_3;
    float shape_no_attack_base_penalty;
    float shape_no_attack_wave_scale;

    int shape_no_progress_start_1;
    int shape_no_progress_start_2;
    int shape_no_progress_start_3;
    int shape_no_attack_start;
} FcRewardParams;

typedef struct {
    int ticks_since_attack;
    float required_work_at_wave_start;
    float cave_progress_prev;
    float last_required_work_remaining;
    float last_current_wave_progress;
    float last_cave_progress;
    int ticks_since_positive_progress;
    int zero_progress_ticks;
    float npc_healing_total;
    float jad_healing_total;
} FcRewardRuntime;

/* Episode analytics shared by training and checkpoint evaluation. */
typedef struct {
    int zero_progress_ticks;
    int wave_reached;
    int wrong_prayer_hits;
    int reached_wave_63;
    int jad_kill_rate;
    float prayer_uptime_range;
    float prayer_uptime_melee;
    float prayer_uptime_magic;
    float npc_healing_total;
    float jad_healing_total;
    int episode_length;
} FcEpisodeSummary;


typedef struct {
    float progress;
    float damage_taken;
    float cave_complete;
    float player_death;

    float prayer_lost;
    float no_progress;
    float no_attack;
    float npc_heal;

    float invalid_action;
    float tick_penalty;

    float total;
} FcRewardBreakdown;

FcRewardParams fc_reward_default_params(void);


/* Wave */

/* Jad healer spawn threshold: spawn 4 Yt-HurKot when Jad HP drops below this.
 * Respawns are re-armed only if the previous healers restored Jad to full HP. */
#define FC_JAD_HEALER_THRESHOLD_HP_TENTHS  1500  /* 150 HP */
#define FC_JAD_NUM_HEALERS            4


/* Api */

/*
 * The functions with external linkage, for fight_caves.c and the C/C++
 * linkage test; everything else is static to binding.c's translation unit.
 * An episode is fc_init, then fc_reset(seed) and one fc_step(actions) per
 * tick, each followed by fc_write_obs and fc_write_mask.
 */

/* Initialize state to a clean starting configuration. Zeroes all fields.
 * Must be called once before the first reset. */
void fc_init(FcState* state);

/* Write the policy observation.
 * out must have room for FC_POLICY_OBS_SIZE floats.
 * Values are normalized to [0,1]. */
void fc_write_obs(const FcState* state, float* out);

/* Write the action mask buffer.
 * out must have room for FC_ACTION_MASK_SIZE floats.
 * 1.0 = valid action, 0.0 = invalid. */
void fc_write_mask(const FcState* state, float* out);

/* Bumped whenever the hashed fields or their order change. */
#define FC_STATE_HASH_VERSION 6u

/*
 * Compute a deterministic hash of the game state.
 *
 * Implementation: FNV-1a over the state's fields (all but render_events) in
 * declaration order, least-significant byte first. The hashed structs are
 * asserted to have no padding, so two states with identical content produce
 * identical hashes.
 *
 * Usage: Call after each fc_step to verify determinism. Two runs with
 * the same (seed, action_sequence) must produce identical hashes at every tick.
 */
uint32_t fc_state_hash(const FcState* state);


/* Items */

typedef struct {
    int id;
    const char *name;
    int slot;                    /* -1 for inventory-only items */
    int stackable;
    int ranged_level, defence_level, hitpoints_level;
    int ranged_attack, ranged_strength;
    int defence[5];               /* stab, slash, crush, magic, ranged */
    int prayer, melee_attack, melee_strength;
    int weapon_kind, speed, range;
} FcItemDef;

typedef enum {
    FC_ITEM_OK, FC_ITEM_INVALID, FC_ITEM_NO_SPACE, FC_ITEM_REQUIREMENTS,
    FC_ITEM_BUSY
} FcItemResult;

/* Item ids the rules and the viewer refer to by name. */
enum {
    FC_ITEM_ID_SHARK = 385,
    FC_ITEM_ID_VIAL = 229,
    FC_ITEM_ID_PRAYER_POTION_1 = 143,
    FC_ITEM_ID_PRAYER_POTION_2 = 141,
    FC_ITEM_ID_PRAYER_POTION_3 = 139,
    FC_ITEM_ID_PRAYER_POTION_4 = 2434,
};


#ifdef FC_CORE_IMPLEMENTATION

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Used before their definitions below. */
static const FcNpcStats* fc_npc_get_stats(int npc_type);
static int fc_rng_int(FcState* state, int max);
static float fc_rng_float(FcState* state);
static int fc_distance_between_areas(int src_x, int src_y, int src_size,
                                     int dst_x, int dst_y, int dst_size);
static int fc_has_los_between_areas(
    int src_x, int src_y, int src_size,
    int dst_x, int dst_y, int dst_size,
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);
static int fc_npc_can_melee_player(int player_x, int player_y,
                                   int npc_x, int npc_y, int npc_size,
                                   const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);
static void fc_build_occupancy(const FcState* state,
                               uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                               int ignore_npc_idx);
static int fc_step_toward(
    int* x, int* y, int target_x, int target_y, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]);
static int fc_prayer_apply_loss_tenths(FcPlayer* p, int requested_loss_tenths);
static void fc_npc_tz_kek_split(FcState* state, int dead_x, int dead_y);
static int fc_spawn_find_available_footprint(const FcState* state,
                                             int preferred_x, int preferred_y,
                                             int size, int max_radius,
                                             int* out_x, int* out_y);
static int fc_spawn_npc_first_free(FcState* state, int npc_type, int x, int y);
static void fc_wave_spawn(FcState* state, int wave_num);
static void fc_tick(FcState* state, const int actions[FC_NUM_ACTION_HEADS]);
static void fc_spawn_position(int spawn_dir, int* x, int* y);
static void fc_wave_check_advance(FcState* state);

/* Combat */

/* OSRS accuracy: the chance that an attack roll beats a defence roll. */
static float fc_hit_chance(int att_roll, int def_roll) {
    if (att_roll > def_roll)
        return 1.0f - (float)(def_roll + 2) / (2.0f * (float)(att_roll + 1));
    else
        return (float)att_roll / (2.0f * (float)(def_roll + 1));
}

/* NPC attack and defence rolls: level plus the invisible boost of 9, times
 * bonus + 64. */
static int fc_npc_roll(int level, int bonus) {
    return (level + 9) * (bonus + 64);
}

/* Player defence roll against one incoming attack type. */
static int fc_player_def_roll(const FcPlayer* p, FcAttackType attack_type) {
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

/* Twisted-bow multiplier in percent against the target's Magic level m
 * (capped at 250 outside the Chambers of Xeric):
 *   cap + (3m - center/10) / 100 - (3m/10 - center)^2 / 100, within [0, cap].
 * Accuracy uses cap 140 and center 100; damage uses cap 250 and center 140. */
static int fc_tbow_multiplier_pct(int target_magic_level, int cap, int center) {
    int64_t magic = target_magic_level;
    if (magic < 0) magic = 0;
    if (magic > 250) magic = 250;

    int64_t delta = 3 * magic / 10 - center;
    int64_t pct = cap + (3 * magic - center / 10) / 100 -
                  delta * delta / 100;
    if (pct < 0) pct = 0;
    if (pct > cap) pct = cap;
    return (int)pct;
}

/* Player ranged attack roll and max hit (whole HP) against a target. Rapid is
 * the active style, so the effective level is Ranged + 8. The Twisted bow
 * scales both by the target's Magic level. */
static void fc_player_ranged_rolls(const FcPlayer* p, const FcNpc* target,
                                   int* attack_roll, int* max_hit) {
    int level = p->ranged_level + 8;
    *attack_roll = level * (p->ranged_attack_bonus + 64);
    *max_hit = (int)(((int64_t)level * (p->ranged_strength_bonus + 64) + 320) /
                     640);
    if (p->weapon_kind == FC_WEAPON_TWISTED_BOW) {
        const FcNpcStats* stats = fc_npc_get_stats(target->npc_type);
        int magic = stats->attack_level[ATTACK_MAGIC];
        *attack_roll = (int)((int64_t)*attack_roll *
                             fc_tbow_multiplier_pct(magic, 140, 100) / 100);
        *max_hit = (int)((int64_t)*max_hit *
                         fc_tbow_multiplier_pct(magic, 250, 140) / 100);
    }
}

/* Damage of a successful hit in tenths: a uniform roll of 0..max_hit whole HP,
 * raised to min_hit (the player's hits deal at least 1). */
static int fc_roll_damage_tenths(FcState* state, int max_hit, int min_hit) {
    if (max_hit <= 0) return 0;
    int rolled = fc_rng_int(state, max_hit + 1);
    return (rolled < min_hit ? min_hit : rolled) * 10;
}

/* PvM protection: the matching protection prayer blocks all of a hit's
 * damage (not PvP's 60% reduction) and any other prayer blocks none.
 * PRAYER_PROTECT_MELEE/RANGE/MAGIC equal ATTACK_MELEE/RANGED/MAGIC. */
static int fc_prayer_blocks_style(int prayer, int attack_style) {
    if (prayer == PRAYER_NONE || attack_style == ATTACK_NONE) return 0;
    return (prayer == attack_style) ? 1 : 0;
}

/* Chebyshev distance from a tile to an NPC's footprint. */
static int fc_distance_to_npc(int px, int py, const FcNpc* npc) {
    return fc_distance_between_areas(px, py, 1,
                                     npc->x, npc->y, npc->size);
}

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
static int fc_ranged_hit_delay(int distance) {
    /* Keep the existing lightweight projectile timing for player ranged attacks. */
    int travel = 5 * distance;  /* default multiplier for player ranged */
    return travel / 30 + 1;
}

/* Per-NPC-type hit delay from the Void 634 projectile timing above. Jad's
 * magic lands no sooner than 3 ticks after the attack, one tick after the
 * player's prayer locks for it. */
static int fc_npc_hit_delay(int npc_type, int attack_style, int distance) {
    if (attack_style == ATTACK_MELEE) return 1;
    if (npc_type == NPC_KET_ZEK) return (8 + 8 * distance) / 30 + 1;
    if (npc_type == NPC_TZTOK_JAD && attack_style == ATTACK_RANGED)
        return 120 / 30 + 1;
    if (npc_type == NPC_TZTOK_JAD) {
        int delay = (8 * distance) / 30 + 1;
        return delay < 3 ? 3 : delay;
    }
    return (5 * distance) / 30 + 1;  /* Tok-Xil */
}

/* Queue a pending hit; returns 0 when the queue is full. */
static int fc_queue_pending_hit(FcPendingHit hits[], int* num_hits, int max_hits,
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

/* Set the player's attack target (-1 clears it) and forget any approach
 * toward the previous one. */
static void set_player_target(FcPlayer* p, int target_idx) {
    p->attack_target_idx = target_idx;
    p->approach_target = 0;
    p->approach_target_x = -1;
    p->approach_target_y = -1;
    p->approach_target_size = 0;
}

/* Record a resolved hit for the viewer. */
static void record_render_hit(FcState* state, int target_entity_type,
                              int target_npc_slot, int source_npc_slot,
                              int attack_style, int damage) {
    FcRenderEvents* events = &state->render_events;
    if (events->hit_count >= FC_MAX_RENDER_HITS) return;

    FcRenderHit* hit = &events->hits[events->hit_count++];
    hit->target_entity_type = target_entity_type;
    hit->target_npc_slot = target_npc_slot;
    hit->source_npc_slot = source_npc_slot;
    hit->attack_style = attack_style;
    hit->damage = damage;
}

/* Resolve the hits on the player whose delay ran out; run each tick after
 * the NPCs act. */
static void fc_resolve_player_pending_hits(FcState* state) {
    FcPlayer* p = &state->player;
    int write = 0;

    for (int i = 0; i < p->num_pending_hits; i++) {
        FcPendingHit* h = &p->pending_hits[i];
        h->ticks_remaining--;
        if (h->ticks_remaining <= 0) {
            /* Hit resolves now — use the prayer locked into this hit. */
            FcNpc* source = &state->npcs[h->source_npc_idx];
            int locked_prayer = h->prayer_snapshot >= 0
                ? h->prayer_snapshot : PRAYER_NONE;
            int blocked = fc_prayer_blocks_style(locked_prayer, h->attack_style);
            int final_damage = blocked ? 0 : h->damage;

            /* Apply damage */
            p->current_hp -= final_damage;
            if (p->current_hp < 0) p->current_hp = 0;

            p->damage_taken_this_tick += final_damage;
            state->damage_taken_this_tick += final_damage;
            p->hit_landed_this_tick = 1;
            record_render_hit(state, ENTITY_PLAYER, -1,
                              h->source_npc_idx, h->attack_style,
                              final_damage);

            /* Auto-retaliate: if player has no target, target the attacker.
             * approach_target stays 0 — player attacks in place, doesn't chase. */
            if (p->attack_target_idx < 0 && source->active && !source->is_dead) {
                set_player_target(p, h->source_npc_idx);
            }

            /* Tz-Kih drains damage dealt + 1 Prayer point, with the base point
             * still applying to misses and prayer-blocked attacks. HP and Prayer
             * both use tenths internally, so final_damage can be added directly. */
            if (h->prayer_drain > 0) {
                int drain = h->prayer_drain;
                if (source->npc_type == NPC_TZ_KIH) drain += final_damage;
                int actual_drain = fc_prayer_apply_loss_tenths(p, drain);
                state->prayer_lost_this_tick += actual_drain;
                state->tz_kih_prayer_drain_this_tick += actual_drain;
                source->prayer_drain_dealt_this_tick += actual_drain;
            }

            /* Track prayer correctness. Correctly blocked Jad hits also use
             * the shared correct-prayer reward applied to every NPC. */
            if (source->npc_type == NPC_TZTOK_JAD) {
                if (blocked) {
                    state->correct_jad_prayer = 1;
                    state->correct_danger_prayer = 1;
                } else {
                    state->wrong_jad_prayer = 1;
                }
            } else if (blocked) {
                state->correct_danger_prayer = 1;
            } else {
                state->wrong_danger_prayer = 1;
            }

            if (locked_prayer != PRAYER_NONE && !blocked)
                state->ep_wrong_prayer_hits++;

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

static void fc_resolve_npc_pending_hits(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    int write = 0;

    for (int i = 0; i < npc->num_pending_hits; i++) {
        FcPendingHit* h = &npc->pending_hits[i];
        h->ticks_remaining--;
        if (h->ticks_remaining <= 0) {
            /* Player's hit lands on NPC */
            npc->current_hp -= h->damage;
            if (npc->current_hp < 0) npc->current_hp = 0;

            npc->damage_taken_this_tick += h->damage;
            state->damage_dealt_this_tick += h->damage;

            if (h->damage > 0) {
                state->hits_landed_this_tick++;
            }
            record_render_hit(state, ENTITY_NPC, npc_idx, -1,
                              h->attack_style, h->damage);

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

static void fc_episode_summary_build(const FcState* state, const FcRewardRuntime* runtime,
                                     int episode_length, FcEpisodeSummary* summary) {
    memset(summary, 0, sizeof(*summary));
    summary->episode_length = episode_length;
    summary->wave_reached = state->current_wave;
    summary->wrong_prayer_hits = state->ep_wrong_prayer_hits;
    summary->reached_wave_63 = state->ep_reached_wave_63;
    summary->jad_kill_rate = state->ep_jad_killed;
    summary->zero_progress_ticks = runtime->zero_progress_ticks;
    summary->npc_healing_total = runtime->npc_healing_total;
    summary->jad_healing_total = runtime->jad_healing_total;
    if (episode_length > 0) {
        summary->prayer_uptime_melee =
            (float)state->ep_ticks_pray_melee / (float)episode_length;
        summary->prayer_uptime_range =
            (float)state->ep_ticks_pray_range / (float)episode_length;
        summary->prayer_uptime_magic =
            (float)state->ep_ticks_pray_magic / (float)episode_length;
    }
}


/* Hash */

/*
 * The state hash is FNV-1a over FcState's bytes in declaration order, except
 * that render_events is skipped and route_x/route_y are fed as interleaved
 * (x, y) pairs. Every hashed field is a 4-byte int, uint32_t or float, or an
 * arena map byte, so on a little-endian target this feeds each value
 * least-significant byte first. The asserts below pin that layout: a field
 * added to a hashed struct (or any padding) changes these sizes, which must
 * then be updated together with FC_STATE_HASH_VERSION and the pinned
 * contract hash.
 */
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "fc_state_hash hashes state bytes and needs a little-endian target"
#endif
_Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(uint32_t) == 4,
               "fc_state_hash needs 4-byte int, float and uint32_t");
_Static_assert(sizeof(FcPendingHit) == 8 * 4, "FcPendingHit: 8 ints");
_Static_assert(sizeof(FcItemStack) == 3 * 4, "FcItemStack: 3 ints");
_Static_assert(sizeof(FcNpc) == (28 + 8 * FC_MAX_PENDING_HITS + 1) * 4,
               "FcNpc: 28 ints, the pending hits, num_pending_hits");
_Static_assert(offsetof(FcPlayer, route_y) ==
                   offsetof(FcPlayer, route_x) + FC_MAX_ROUTE * 4 &&
               offsetof(FcPlayer, route_len) ==
                   offsetof(FcPlayer, route_y) + FC_MAX_ROUTE * 4,
               "FcPlayer: route_x, route_y and route_len are adjacent");
/* FcPlayer: 38 ints, the route, route_len to approach_target_size (8), the
 * pending hits, num_pending_hits and the 5 per-tick flags, the item stacks
 * and 4 trailing ints. */
_Static_assert(sizeof(FcPlayer) ==
                   (38 + 2 * FC_MAX_ROUTE + 8 + 8 * FC_MAX_PENDING_HITS + 6 +
                    3 * (FC_INVENTORY_SLOTS + FC_EQUIPMENT_SLOTS) + 4) * 4,
               "FcPlayer layout changed");
_Static_assert(offsetof(FcState, npcs) == sizeof(FcPlayer) &&
               offsetof(FcState, active_loadout) ==
                   offsetof(FcState, npcs) + FC_MAX_NPCS * sizeof(FcNpc) &&
               offsetof(FcState, render_events) ==
                   offsetof(FcState, active_loadout) + 4 &&
               offsetof(FcState, current_wave) ==
                   offsetof(FcState, render_events) + sizeof(FcRenderEvents),
               "FcState: player, npcs, active_loadout, render_events, current_wave");
_Static_assert(sizeof(FcState) - offsetof(FcState, current_wave) ==
                   46 * 4 + 3 * FC_ARENA_WIDTH * FC_ARENA_HEIGHT,
               "FcState from current_wave: 46 4-byte fields and 3 arena maps");

static uint32_t fc_hash_bytes(uint32_t hash, const void* data, size_t size) {
    const uint8_t* bytes = data;
    for (size_t i = 0; i < size; i++) hash = (hash ^ bytes[i]) * 0x01000193u;
    return hash;
}

uint32_t fc_state_hash(const FcState* state) {
    const FcPlayer* p = &state->player;
    uint32_t hash = 0x811c9dc5u;
    hash = fc_hash_bytes(hash, p, offsetof(FcPlayer, route_x));
    for (int i = 0; i < FC_MAX_ROUTE; i++) {
        hash = fc_hash_bytes(hash, &p->route_x[i], sizeof(int));
        hash = fc_hash_bytes(hash, &p->route_y[i], sizeof(int));
    }
    hash = fc_hash_bytes(hash, &p->route_len,
                         sizeof(FcPlayer) - offsetof(FcPlayer, route_len));
    hash = fc_hash_bytes(hash, state->npcs, sizeof(state->npcs));
    hash = fc_hash_bytes(hash, &state->active_loadout, sizeof(int));
    return fc_hash_bytes(hash, &state->current_wave,
                         sizeof(FcState) - offsetof(FcState, current_wave));
}

/* Items */

/* The loadout's equipment and the supplies, pinned to the loadout's
 * existing balance: equipment switching must not rebalance training. */
static const FcItemDef ITEMS[] = {
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
    {.id=20997, .name="Twisted bow", .slot=FC_EQUIP_SLOT_WEAPON,
     .ranged_attack=70, .ranged_strength=20, .ranged_level=85,
     .weapon_kind=FC_WEAPON_TWISTED_BOW, .speed=5, .range=10},
    {.id=11212, .name="Dragon arrow", .slot=FC_EQUIP_SLOT_AMMO,
     .stackable=1, .ranged_strength=60},
    {.id=FC_ITEM_ID_SHARK, .name="Shark", .slot=-1},
    {.id=FC_ITEM_ID_PRAYER_POTION_4, .name="Prayer potion(4)", .slot=-1},
    {.id=FC_ITEM_ID_PRAYER_POTION_3, .name="Prayer potion(3)", .slot=-1},
    {.id=FC_ITEM_ID_PRAYER_POTION_2, .name="Prayer potion(2)", .slot=-1},
    {.id=FC_ITEM_ID_PRAYER_POTION_1, .name="Prayer potion(1)", .slot=-1},
    {.id=FC_ITEM_ID_VIAL, .name="Vial", .slot=-1},
};

static const FcItemDef *fc_item_definition(int item_id) {
    for (unsigned i = 0; i < sizeof(ITEMS) / sizeof(ITEMS[0]); i++)
        if (ITEMS[i].id == item_id) return &ITEMS[i];
    return NULL;
}

static void recalculate_equipment(FcPlayer *p) {
    p->ranged_attack_bonus = p->ranged_strength_bonus = 0;
    p->defence_stab = p->defence_slash = p->defence_crush = 0;
    p->defence_magic = p->defence_ranged = p->prayer_bonus = 0;
    p->melee_attack_bonus = p->melee_strength_bonus = 0;
    const FcItemDef *weapon = fc_item_definition(p->equipment[FC_EQUIP_SLOT_WEAPON].item_id);
    const FcItemDef *ammo = fc_item_definition(p->equipment[FC_EQUIP_SLOT_AMMO].item_id);
    /* The Twisted bow is the only weapon and dragon arrows the only ammunition,
     * so worn arrows count only while the bow is wielded. */
    int usable_ammo = weapon && ammo;
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
    }
    p->weapon_kind = weapon ? weapon->weapon_kind : FC_WEAPON_UNARMED;
    p->weapon_speed = weapon ? weapon->speed : 4;
    p->weapon_range = weapon ? weapon->range : 1;
    p->weapon_uses_ammo = weapon != NULL;
    p->ammo_count = usable_ammo ? p->equipment[FC_EQUIP_SLOT_AMMO].quantity : 0;
}

static int potion_doses(int id) {
    switch (id) {
        case FC_ITEM_ID_PRAYER_POTION_4: return 4;
        case FC_ITEM_ID_PRAYER_POTION_3: return 3;
        case FC_ITEM_ID_PRAYER_POTION_2: return 2;
        case FC_ITEM_ID_PRAYER_POTION_1: return 1;
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
    const int pots[] = {0, FC_ITEM_ID_PRAYER_POTION_1, FC_ITEM_ID_PRAYER_POTION_2,
                        FC_ITEM_ID_PRAYER_POTION_3, FC_ITEM_ID_PRAYER_POTION_4};
    int slot = 0;
    while (doses > 0) {
        int count = doses > 4 ? 4 : doses;
        p->inventory[slot++] = (FcItemStack){pots[count], 1, 0};
        doses -= count;
    }
    for (int i = 0; i < sharks; i++)
        p->inventory[slot++] = (FcItemStack){FC_ITEM_ID_SHARK, 1, 0};
    p->selected_food_slot = p->selected_potion_slot = -1;
}

static void fc_set_initial_supplies(FcState *state, int sharks, int prayer_doses) {
    /* Reset-time configuration only, not a gameplay inventory refill API. */
    if (state && state->tick == 0)
        reset_supplies(&state->player, sharks, prayer_doses);
}

static void fc_items_init(FcPlayer *p, const FcLoadout *loadout) {
    memset(p->equipment, 0, sizeof(p->equipment));
    for (int i = 0; i < loadout->equipment_count; i++) {
        const FcItemDef *item = fc_item_definition(loadout->equipment[i]);
        p->equipment[item->slot] = (FcItemStack){
            item->id, item->slot == FC_EQUIP_SLOT_AMMO ? loadout->ammo : 1, 0
        };
    }
    recalculate_equipment(p);
    reset_supplies(p, FC_MAX_SHARKS, FC_MAX_PRAYER_DOSES);
}

static const char *fc_item_result_message(FcItemResult result) {
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

/* Immediate inventory transactions between ticks. They do not advance time,
 * reset cooldowns, roll RNG, or alter already-launched attacks. */
static FcItemResult fc_equip_item(FcState *state, int index) {
    if (!can_change_items(state)) return FC_ITEM_BUSY;
    FcPlayer *p = &state->player;
    if (index < 0 || index >= FC_INVENTORY_SLOTS) return FC_ITEM_INVALID;
    FcItemStack incoming = p->inventory[index];
    const FcItemDef *item = fc_item_definition(incoming.item_id);
    if (!item || item->slot < 0 || incoming.quantity <= 0 ||
        (!item->stackable && incoming.quantity != 1)) return FC_ITEM_INVALID;
    if (p->ranged_level < item->ranged_level || p->defence_level < item->defence_level ||
        p->max_hp / 10 < item->hitpoints_level) return FC_ITEM_REQUIREMENTS;
    FcItemStack *worn = &p->equipment[item->slot];
    if (item->stackable && worn->item_id == incoming.item_id) {
        int amount = INT_MAX - worn->quantity;
        if (amount > incoming.quantity) amount = incoming.quantity;
        if (!amount) return FC_ITEM_NO_SPACE;
        worn->quantity += amount;
        p->inventory[index].quantity -= amount;
        if (!p->inventory[index].quantity) p->inventory[index] = (FcItemStack){0};
    } else {
        p->inventory[index] = *worn;
        *worn = incoming;
    }
    recalculate_equipment(p);
    set_player_target(p, -1); /* held-item action interrupts interaction */
    return FC_ITEM_OK;
}

static FcItemResult fc_unequip_item(FcState *state, int slot) {
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

static FcItemResult fc_inventory_swap(FcState *state, int first, int second) {
    if (!can_change_items(state)) return FC_ITEM_BUSY;
    if (first < 0 || first >= FC_INVENTORY_SLOTS || second < 0 ||
        second >= FC_INVENTORY_SLOTS) return FC_ITEM_INVALID;
    FcItemStack temp = state->player.inventory[first];
    state->player.inventory[first] = state->player.inventory[second];
    state->player.inventory[second] = temp;
    state->player.selected_food_slot = state->player.selected_potion_slot = -1;
    return FC_ITEM_OK;
}

/* Select the actual slot consumed by the next canonical food/potion action. */
static FcItemResult fc_select_consumable(FcState *state, int slot) {
    if (!can_change_items(state)) return FC_ITEM_BUSY;
    if (slot < 0 || slot >= FC_INVENTORY_SLOTS) return FC_ITEM_INVALID;
    int id = state->player.inventory[slot].item_id;
    if (id == FC_ITEM_ID_SHARK) state->player.selected_food_slot = slot;
    else if (potion_doses(id)) state->player.selected_potion_slot = slot;
    else return FC_ITEM_INVALID;
    return FC_ITEM_OK;
}

static void fc_items_consume(FcPlayer *p, int potion) {
    int selected = potion ? p->selected_potion_slot : p->selected_food_slot;
    for (int n = -1; n < FC_INVENTORY_SLOTS; n++) {
        int slot = n < 0 ? selected : n;
        if (slot < 0 || slot >= FC_INVENTORY_SLOTS) continue;
        FcItemStack *item = &p->inventory[slot];
        int doses = potion_doses(item->item_id);
        if (potion && doses) {
            const int replacement[] = {
                FC_ITEM_ID_VIAL, FC_ITEM_ID_PRAYER_POTION_1,
                FC_ITEM_ID_PRAYER_POTION_2, FC_ITEM_ID_PRAYER_POTION_3};
            item->item_id = replacement[doses - 1];
            break;
        }
        if (!potion && item->item_id == FC_ITEM_ID_SHARK) {
            *item = (FcItemStack){0};
            break;
        }
    }
    if (potion) p->prayer_doses_remaining--;
    else p->sharks_remaining--;
}

static void fc_items_spend_ammo(FcPlayer *p) {
    p->ammo_count--;
    FcItemStack *ammo = &p->equipment[FC_EQUIP_SLOT_AMMO];
    if (ammo->quantity > 0 && --ammo->quantity == 0) {
        *ammo = (FcItemStack){0};
        recalculate_equipment(p);
    }
}


/* Loadout */

static const FcLoadout FC_LOADOUT = {
    .combat_style_profile = 25,   /* bow */
    .max_hp = 990, .max_prayer = 990,
    .attack_lvl = 1, .strength_lvl = 1, .defence_lvl = 99,
    .ranged_lvl = 99, .prayer_lvl = 99, .magic_lvl = 1,
    .ammo = 50000,                /* dragon arrows */
    .equipment_count = 10,
    .equipment = {
        27235,  /* Masori mask (f) */
        22109,  /* Ava's assembler */
        19547,  /* Necklace of anguish */
        20997,  /* Twisted bow */
        27238,  /* Masori body (f) */
        11212,  /* Dragon arrow */
        27241,  /* Masori chaps (f) */
        26235,  /* Zaryte vambraces */
        13237,  /* Pegasian boots */
        28310,  /* Venator ring */
    },
};


/* Npc */

/*
 * NPC stats and AI. Each tick an active NPC (fc_npc_tick):
 *   1. Decrements its attack timer.
 *   2. Walks toward the player (greedy step) unless it can already attack
 *      from its tile.
 *   3. Attacks once its timer is ready and the player is in reach, queueing
 *      a pending hit.
 *
 * Type-specific:
 *   Tz-Kih:     Melee + prayer drain on hit.
 *   Tz-Kek:     Melee. Splits into 2 small Tz-Kek on death.
 *   Tz-Kek-Sm:  Melee. (no special)
 *   Tok-Xil:    Ranged with projectile delay; melee at contact.
 *   Yt-MejKot:  Chooses one melee-cycle action: attack or heal a weak NPC.
 *   Ket-Zek:    Magic with projectile delay; magic or melee at contact.
 *   TzTok-Jad:  Magic/ranged at distance; melee/magic/ranged at contact.
 *   Yt-HurKot:  Heals Jad in range; attacks the player after being tagged.
 */

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
        .max_hit = {[ATTACK_MELEE] = 4},
        .attack_level = {[ATTACK_MELEE] = 20, [ATTACK_RANGED] = 30,
                         [ATTACK_MAGIC] = 15},
        .def_level = 15, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_STAB,
        .size = 1, .movement_speed = 1, .prayer_drain = 10,
    },

    /* NPC_TZ_KEK: Lv 45 melee blob. Splits into 2 small on death.
     * Void 634: HP 200, Att 40, Str 60, Def 30, size 2, crush max 70 */
    [NPC_TZ_KEK] = {
        .max_hp = 200, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .max_hit = {[ATTACK_MELEE] = 7},
        .attack_level = {[ATTACK_MELEE] = 40, [ATTACK_RANGED] = 60,
                         [ATTACK_MAGIC] = 30},
        .def_level = 30, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 2, .movement_speed = 1,
    },

    /* NPC_TZ_KEK_SM: Lv 22 small blob (from split).
     * Void 634: HP 100, Att 20, Str 30, Def 15, size 1, crush max 40 */
    [NPC_TZ_KEK_SM] = {
        .max_hp = 100, .attack_style = ATTACK_MELEE,
        .attack_speed = 4, .attack_range = 1,
        .max_hit = {[ATTACK_MELEE] = 4},
        .attack_level = {[ATTACK_MELEE] = 20, [ATTACK_RANGED] = 30,
                         [ATTACK_MAGIC] = 15},
        .def_level = 15, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 1, .movement_speed = 1,
    },

    /* NPC_TOK_XIL: Lv 90 ranged + melee (DUAL MODE).
     * Void 634: HP 400, Att 80, Str 120, Def 60, Rng 120, size 3
     * Current Fight Caves maxima are 13 HP for both melee and Ranged. */
    [NPC_TOK_XIL] = {
        .max_hp = 400, .attack_style = ATTACK_RANGED,
        .attack_speed = 4, .attack_range = 14,
        .max_hit = {[ATTACK_MELEE] = 13, [ATTACK_RANGED] = 13},
        .attack_level = {[ATTACK_MELEE] = 80, [ATTACK_RANGED] = 120,
                         [ATTACK_MAGIC] = 60},
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
        .max_hit = {[ATTACK_MELEE] = 25},
        .attack_level = {[ATTACK_MELEE] = 160, [ATTACK_RANGED] = 240,
                         [ATTACK_MAGIC] = 120},
        .def_level = 120, .ranged_def_bonus = 0,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 4, .movement_speed = 1, .heal_amount = 100,
    },

    /* NPC_KET_ZEK: Lv 360 magic + melee (DUAL MODE).
     * Void 634: HP 1600, Att 320, Str 480, Def 240, Mag 240, size 5
     * Current Fight Caves maxima are 55 HP melee and 52 HP Magic. The Magic
     * attack has +60 accuracy; the melee attack has no equipment bonus. */
    [NPC_KET_ZEK] = {
        .max_hp = 1600, .attack_style = ATTACK_MAGIC,
        .attack_speed = 4, .attack_range = 14,
        .max_hit = {[ATTACK_MELEE] = 55, [ATTACK_MAGIC] = 52},
        .attack_level = {[ATTACK_MELEE] = 320, [ATTACK_RANGED] = 480,
                         [ATTACK_MAGIC] = 240},
        .attack_bonus = {[ATTACK_MAGIC] = 60},
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
        .max_hit = {[ATTACK_MELEE] = 97, [ATTACK_RANGED] = 97,
                    [ATTACK_MAGIC] = 95},
        .attack_level = {[ATTACK_MELEE] = 640, [ATTACK_RANGED] = 960,
                         [ATTACK_MAGIC] = 480},
        .attack_bonus = {[ATTACK_MAGIC] = 60},
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
        .max_hit = {[ATTACK_MELEE] = 14},
        .attack_level = {[ATTACK_MELEE] = 140, [ATTACK_RANGED] = 120,
                         [ATTACK_MAGIC] = 120},
        .def_level = 60, .ranged_def_bonus = 100,
        .melee_attack_type = FC_ATTACK_TYPE_CRUSH,
        .size = 1, .movement_speed = 1,
        .heal_amount = 50, .heal_interval = 4,
    },
};

static const FcNpcStats* fc_npc_get_stats(int npc_type) {
    if (npc_type < 0 || npc_type >= NPC_TYPE_COUNT) return &NPC_STATS[0];
    return &NPC_STATS[npc_type];
}

/* Initialize an NPC slot from type and spawn position. */
static void fc_npc_spawn(FcNpc* npc, int npc_type, int x, int y, int spawn_index) {
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

/* One step toward the target around walls, the player and the other NPCs. */
static int npc_step_toward(FcState* state, int npc_idx,
                           int target_x, int target_y) {
    FcNpc* npc = &state->npcs[npc_idx];
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    fc_build_occupancy(state, occupied, npc_idx);
    return fc_step_toward(&npc->x, &npc->y, target_x, target_y, npc->size,
                          state->walkable, state->movement_flags, occupied);
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

/* Which of its attacks an NPC could make from where it stands: melee needs
 * open cardinal contact, and its ranged or magic primary style needs the
 * player within attack range and in projectile line of sight. */
static void npc_reach(const FcState* state, const FcNpc* npc,
                      int* melee, int* distant) {
    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    const FcPlayer* p = &state->player;
    int dist = fc_distance_to_npc(p->x, p->y, npc);
    *melee = stats->max_hit[ATTACK_MELEE] > 0 &&
        fc_npc_can_melee_player(p->x, p->y, npc->x, npc->y, npc->size,
                                state->movement_flags);
    *distant = npc->attack_style != ATTACK_MELEE &&
        dist > 0 && dist <= npc->attack_range &&
        fc_has_los_between_areas(npc->x, npc->y, npc->size,
                                 p->x, p->y, 1, state->los_flags);
}

/* Walk toward the player, at most movement_speed steps, unless the NPC can
 * already attack from its tile. */
static void npc_chase_player(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    int melee, distant;
    npc_reach(state, npc, &melee, &distant);
    if (melee || distant) return;
    for (int step = 0; step < npc->movement_speed; step++) {
        int target_x, target_y;
        npc_naive_player_chase_destination(npc, &state->player,
                                           &target_x, &target_y);
        if (target_x == npc->x && target_y == npc->y) break;
        if (!npc_step_toward(state, npc_idx, target_x, target_y)) break;
    }
}

/* Tz-Kek splits on death into two small Tz-Kek. */
static void fc_npc_tz_kek_split(FcState* state, int dead_x, int dead_y) {
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

/* Record an NPC attack launched this tick for the viewer. */
static void record_npc_attack(FcState* state, const FcNpc* npc, int npc_idx,
                              int attack_style, int prayer_lock_tick,
                              int hit_queued) {
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
    attack->prayer_lock_tick = prayer_lock_tick;
    attack->hit_queued = hit_queued;
}

/* Attack once the timer is ready and the player is in reach. At melee
 * contact an NPC uses melee, except that Ket-Zek may also pick its magic, and
 * Jad its magic or ranged, with equal weight when those could hit too. At a
 * distance it uses its primary style, and Jad picks magic or ranged. */
static void npc_attack(FcState* state, FcNpc* npc, int npc_idx) {
    if (npc->attack_timer > 0) return;
    int melee, distant;
    npc_reach(state, npc, &melee, &distant);
    int jad = npc->npc_type == NPC_TZTOK_JAD;
    int style;
    if (melee && distant && jad) {
        int choice = fc_rng_int(state, 3);
        style = choice == 0 ? ATTACK_MELEE :
                choice == 1 ? ATTACK_MAGIC : ATTACK_RANGED;
    } else if (melee && distant && npc->npc_type == NPC_KET_ZEK) {
        style = fc_rng_int(state, 2) == 0 ? npc->attack_style : ATTACK_MELEE;
    } else if (melee) {
        style = ATTACK_MELEE;
    } else if (distant && jad) {
        style = fc_rng_int(state, 2) == 0 ? ATTACK_MAGIC : ATTACK_RANGED;
    } else if (distant) {
        style = npc->attack_style;
    } else {
        return;
    }

    const FcNpcStats* stats = fc_npc_get_stats(npc->npc_type);
    FcPlayer* player = &state->player;
    FcAttackType attack_type =
        style == ATTACK_MELEE ? (FcAttackType)stats->melee_attack_type :
        style == ATTACK_RANGED ? FC_ATTACK_TYPE_RANGED : FC_ATTACK_TYPE_MAGIC;
    int attack_roll = fc_npc_roll(stats->attack_level[style],
                                  stats->attack_bonus[style]);
    float hit_chance = fc_hit_chance(attack_roll,
                                     fc_player_def_roll(player, attack_type));
    int hit = fc_rng_float(state) < hit_chance;
    int damage = hit ? fc_roll_damage_tenths(state, stats->max_hit[style], 0) : 0;
    int delay = fc_npc_hit_delay(npc->npc_type, style,
                                 fc_distance_to_npc(player->x, player->y, npc));

    /* Jad's ranged and magic lock the player's prayer two ticks after the
     * attack; every other attack locks the prayer the tick started with. */
    int prayer_lock_tick = -1;
    int hit_queued = fc_queue_pending_hit(
        player->pending_hits, &player->num_pending_hits, FC_MAX_PENDING_HITS,
        damage, delay, style, npc_idx, stats->prayer_drain);
    if (hit_queued) {
        FcPendingHit* queued =
            &player->pending_hits[player->num_pending_hits - 1];
        if (jad && style != ATTACK_MELEE) {
            queued->prayer_snapshot = -1;
            queued->prayer_lock_tick = state->tick + 2;
            prayer_lock_tick = queued->prayer_lock_tick;
        } else {
            queued->prayer_snapshot = player->prayer_at_tick_start;
        }
    }

    record_npc_attack(state, npc, npc_idx, style, prayer_lock_tick, hit_queued);
    npc->attack_timer = npc->attack_speed;
}

/* Heal target by up to amount (capped at its max HP) and record the heal for
 * observations and rewards. Returns the HP restored. */
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

    if (target->npc_type == NPC_TZTOK_JAD) {
        state->jad_heal_amount_this_tick += amount;
    }
    return amount;
}

static FcNpc* yt_mejkot_heal_target(FcState* state, FcNpc* npc) {
    if (npc->current_hp < npc->max_hp / 2) return npc;

    FcNpc* best = NULL;
    int best_distance = FC_ARENA_WIDTH;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* target = &state->npcs[i];
        if (target == npc || !target->active || target->is_dead) continue;
        if (target->current_hp >= target->max_hp / 2) continue;

        int distance = fc_distance_between_areas(npc->x, npc->y, 1,
                                                 target->x, target->y, 1);
        if (distance > 8) continue;
        if (!best || distance < best_distance ||
            (distance == best_distance && target->spawn_index < best->spawn_index)) {
            best = target;
            best_distance = distance;
        }
    }
    return best;
}

/* At melee contact a Yt-MejKot may spend its attack healing itself or a
 * nearby NPC below half HP. */
static int yt_mejkot_try_heal(FcState* state, FcNpc* npc) {
    if (npc->attack_timer > 0) return 0;
    if (!fc_npc_can_melee_player(state->player.x, state->player.y,
                                 npc->x, npc->y, npc->size,
                                 state->movement_flags)) {
        return 0;
    }

    FcNpc* target = yt_mejkot_heal_target(state, npc);
    if (!target) return 0;

    apply_npc_heal(state, npc, target, npc->heal_amount);
    npc->heal_target_idx = (int)(target - state->npcs);
    npc->attack_timer = npc->attack_speed;
    return 1;
}

/* Yt-HurKot heals Jad until a player hit tags it onto the player for good.
 * Heal ranges are measured between the NPCs' south-west anchor tiles. */
#define FC_HURKOT_HEAL_RANGE 5

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
    if (fc_distance_between_areas(npc->x, npc->y, 1, jad->x, jad->y, 1) >
            FC_HURKOT_HEAL_RANGE ||
        jad->current_hp >= jad->max_hp) {
        return;
    }

    if (apply_npc_heal(state, npc, jad, npc->heal_amount) > 0) {
        state->jad_heal_procs_this_tick++;
    }
}

static void yt_hurkot_follow_jad(FcState* state, FcNpc* npc, int npc_idx) {
    int jad_idx = find_active_jad(state);
    npc->heal_target_idx = jad_idx;
    if (jad_idx < 0) return;

    FcNpc* jad = &state->npcs[jad_idx];
    yt_hurkot_heal_cycle(state, npc, jad);
    if (fc_distance_between_areas(npc->x, npc->y, 1, jad->x, jad->y, 1) >
            FC_HURKOT_HEAL_RANGE) {
        npc_step_toward(state, npc_idx, jad->x, jad->y);
    }
}

/* One tick of NPC AI: movement, then an attack or a Yt-MejKot heal. */
static void fc_npc_tick(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    if (!npc->active || npc->is_dead) return;
    if (npc->attack_timer > 0) npc->attack_timer--;

    /* Until a player hit tags it, a Yt-HurKot follows and heals Jad. Once
     * tagged it chases and attacks the player like every other NPC. */
    if (npc->npc_type == NPC_YT_HURKOT && !npc->healer_distracted) {
        yt_hurkot_follow_jad(state, npc, npc_idx);
        return;
    }
    npc->heal_target_idx = -1;
    npc_chase_player(state, npc_idx);

    /* A Yt-MejKot heal is an attack-cycle choice, not a parallel action. */
    if (npc->npc_type == NPC_YT_MEJKOT && yt_mejkot_try_heal(state, npc)) {
        return;
    }
    npc_attack(state, npc, npc_idx);
}

#undef FC_HURKOT_HEAL_RANGE

/* Pathfinding */

/*
 * Grid movement, footprint checks, and LOS:
 *   - NPCs have sizes 1-5 (Jad and Ket-Zek are 5x5). Movement must check
 *     the entire footprint at the destination tile.
 *   - Projectile LOS uses its own directional collision flags.
 *   - Movement uses whole-tile blocking plus directional wall flags from the
 *     authoritative cache data, never the visual mesh.
 */

/* An entity of this size can stand at (x,y) when every tile of its
 * [x..x+size-1, y..y+size-1] footprint is in the arena and walkable. */
static int fc_footprint_walkable(int x, int y, int size,
                                 const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int tx = x + dx;
            int ty = y + dy;
            if (tx < 0 || tx >= FC_ARENA_WIDTH || ty < 0 ||
                ty >= FC_ARENA_HEIGHT || !walkable[tx][ty]) return 0;
        }
    }
    return 1;
}

/* Mark all in-bounds tiles in a footprint occupied. Out-of-bounds tiles are
 * ignored here; availability checks still reject out-of-bounds footprints via
 * the static walkability check. */
static void fc_mark_footprint_occupied(uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
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

/* Build an occupancy grid from the player and the live NPCs. Pass
 * ignore_npc_idx to omit the moving NPC's own current footprint. */
static void fc_build_occupancy(const FcState* state,
                               uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
                               int ignore_npc_idx) {
    memset(occupied, 0, FC_ARENA_WIDTH * FC_ARENA_HEIGHT);
    fc_mark_footprint_occupied(occupied, state->player.x, state->player.y, 1);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* npc = &state->npcs[i];
        if (i == ignore_npc_idx) continue;
        if (!npc->active || npc->is_dead) continue;
        fc_mark_footprint_occupied(occupied, npc->x, npc->y, npc->size);
    }
}

/* Check static terrain plus a caller-provided dynamic occupancy grid. */
static int fc_footprint_available_dynamic(
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

/* Check one sized step using the native client/RSMod leading-edge collision
 * masks, including the distinct size-1, size-2, and large-actor rules. With an
 * occupancy grid, occupied tiles behave as whole-tile blockers on the exact
 * leading-edge cells checked by that rule. */
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

/* One greedy step of a footprint toward (target_x, target_y): the diagonal
 * first, then x, then y, each checked by the sized-step rule against walls
 * and, when occupied is not NULL, occupied tiles. Returns 1 if it moved. */
static int fc_step_toward(
    int* x, int* y, int target_x, int target_y, int size,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    int dx = 0, dy = 0;
    if (target_x > *x) dx = 1; else if (target_x < *x) dx = -1;
    if (target_y > *y) dy = 1; else if (target_y < *y) dy = -1;

    if (dx != 0 && dy != 0 &&
        fc_footprint_step_valid(*x, *y, dx, dy, size,
                                walkable, movement_flags, occupied)) {
        *x += dx; *y += dy; return 1;
    }
    if (dx != 0 &&
        fc_footprint_step_valid(*x, *y, dx, 0, size,
                                walkable, movement_flags, occupied)) {
        *x += dx; return 1;
    }
    if (dy != 0 &&
        fc_footprint_step_valid(*x, *y, 0, dy, size,
                                walkable, movement_flags, occupied)) {
        *y += dy; return 1;
    }
    return 0;
}

/* Chebyshev distance between the nearest tiles of two rectangular areas. */
static int fc_distance_between_areas(int src_x, int src_y, int src_size,
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

/* Footprint-aware LOS using the closest coordinate from each rectangle, as in
 * the native line validator. */
static int fc_has_los_between_areas(
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

static int fc_npc_can_melee_player(int player_x, int player_y,
                                   int npc_x, int npc_y, int npc_size,
                                   const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (npc_size <= 0) return 0;
    int npc_max_x = npc_x + npc_size - 1;
    int npc_max_y = npc_y + npc_size - 1;

    /* Rectangular-exclusive reach: shared cardinal edges are valid; diagonal
     * corner contact and overlapping footprints are not. */
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

/* An attacker on tile (x, y) can hit the target rectangle: through open
 * cardinal contact when range is FC_ROUTE_MELEE_RANGE, otherwise within range
 * and in projectile line of sight. */
static int fc_can_attack_from(
    int x, int y, int dst_x, int dst_y, int dst_size, int range,
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (range == FC_ROUTE_MELEE_RANGE)
        return fc_npc_can_melee_player(x, y, dst_x, dst_y, dst_size,
                                       movement_flags);
    int distance = fc_distance_between_areas(x, y, 1, dst_x, dst_y, dst_size);
    return distance > 0 && distance <= range &&
           fc_has_los_between_areas(x, y, 1, dst_x, dst_y, dst_size,
                                    los_flags);
}

/* BFS routes for click-to-move and attack approach. */
static const int FC_ROUTE_DIRECTIONS[8][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1},
    {-1, -1}, {1, -1}, {-1, 1}, {1, 1},
};

typedef enum {
    FC_ROUTE_EXACT,
    FC_ROUTE_ATTACK,
} FcRouteGoalKind;

static int fc_route_goal_reached(
    FcRouteGoalKind kind, int x, int y,
    int dst_x, int dst_y, int dst_size, int attack_range,
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (kind == FC_ROUTE_EXACT) return x == dst_x && y == dst_y;
    return fc_can_attack_from(x, y, dst_x, dst_y, dst_size, attack_range,
                              movement_flags, los_flags);
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
    FcRouteGoalKind goal_kind, int attack_range,
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
                                  attack_range, movement_flags, los_flags)) {
            found_x = cx;
            found_y = cy;
            break;
        }
        for (int d = 0; d < 8; d++) {
            int step_x = FC_ROUTE_DIRECTIONS[d][0];
            int step_y = FC_ROUTE_DIRECTIONS[d][1];
            int nx = cx + step_x, ny = cy + step_y;
            if (nx < 0 || nx >= FC_ARENA_WIDTH || ny < 0 || ny >= FC_ARENA_HEIGHT) continue;
            if (vis[nx][ny] || !fc_footprint_step_valid(
                    cx, cy, step_x, step_y, 1,
                    walkable, movement_flags, NULL)) continue;
            vis[nx][ny] = 1;
            pdx[nx][ny] = (int8_t)-step_x;
            pdy[nx][ny] = (int8_t)-step_y;
            distance[nx][ny] = (uint16_t)(distance[cx][cy] + 1u);
            qx[qt] = nx; qy[qt] = ny; qt++;
        }
    }

    if (found_x < 0 && goal_kind == FC_ROUTE_EXACT) {
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

/* Native move-near fallback for human click-to-move. If the exact destination
 * is unreachable, chooses the reachable tile within ten tiles with the lowest
 * squared destination distance, breaking ties by route length. */
static int fc_pathfind_bfs_move_near(
    int sx, int sy, int dx, int dy,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps) {
    return fc_bfs_route(sx, sy, dx, dy, 1, FC_ROUTE_EXACT, 0,
                        walkable, movement_flags, NULL,
                        out_x, out_y, max_steps);
}

/* Route to the first shortest-path tile outside a target rectangle that has
 * both the requested attack range and authoritative projectile LOS. */
static int fc_pathfind_attack_position(
    int sx, int sy, int target_x, int target_y, int target_size,
    int attack_range,
    const uint8_t walkable[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t movement_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT],
    int out_x[], int out_y[], int max_steps) {
    return fc_bfs_route(sx, sy, target_x, target_y, target_size,
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

/* Prayer */

/*
 * Prayer activation, drain, and potion restore.
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

/* Drain prayer points based on active prayer and bonus. prayer_at_tick_start
 * is the prayer before this tick's actions, so 1-tick flicks do not drain. */
static int fc_prayer_drain_tick(FcPlayer* p, int prayer_at_tick_start,
                                int flicked) {
    int prayer_before = p->current_prayer;

    if (p->current_prayer <= 0) {
        enforce_post_loss_invariant(p);
        return 0;
    }

    if (p->prayer == PRAYER_NONE) return 0;
    if (prayer_at_tick_start == PRAYER_NONE) return 0;
    if (flicked) return 0;

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

/* Apply an FC_PRAYER_* action. Turning a protection prayer on needs prayer
 * points. The flick actions turn the current prayer off first; they count as a
 * flick when an active prayer went off and the new one came on. */
static FcPrayerTransition fc_prayer_apply_action(FcPlayer* p, int prayer_action) {
    static const int protection[FC_PRAYER_DIM] = {
        [FC_PRAYER_MAGIC] = PRAYER_PROTECT_MAGIC,
        [FC_PRAYER_RANGE] = PRAYER_PROTECT_RANGE,
        [FC_PRAYER_MELEE] = PRAYER_PROTECT_MELEE,
        [FC_PRAYER_FLICK_MAGIC] = PRAYER_PROTECT_MAGIC,
        [FC_PRAYER_FLICK_RANGE] = PRAYER_PROTECT_RANGE,
        [FC_PRAYER_FLICK_MELEE] = PRAYER_PROTECT_MELEE,
    };
    FcPrayerTransition result = {0};
    int prior = p->prayer;
    if (prayer_action == FC_PRAYER_OFF) {
        p->prayer = PRAYER_NONE;
    } else if (prayer_action > FC_PRAYER_OFF && prayer_action < FC_PRAYER_DIM) {
        p->prayer = p->current_prayer > 0 ? protection[prayer_action] : PRAYER_NONE;
        result.flicked = prayer_action >= FC_PRAYER_FLICK_MAGIC &&
                         prior != PRAYER_NONE && p->prayer != PRAYER_NONE;
    }
    result.changed = p->prayer != prior;
    return result;
}

/* Apply capped Prayer loss in tenths. Every Prayer-loss source uses this
 * boundary so depletion always deactivates the overhead and clears the drain
 * counter. */
static int fc_prayer_apply_loss_tenths(FcPlayer* p, int requested_loss_tenths) {
    int prayer_before = p->current_prayer;
    if (requested_loss_tenths > 0 && p->current_prayer > 0) {
        int loss = requested_loss_tenths;
        if (loss > p->current_prayer) loss = p->current_prayer;
        p->current_prayer -= loss;
    }
    enforce_post_loss_invariant(p);
    return prayer_before - p->current_prayer;
}

static int fc_prayer_potion_restore(int prayer_level) {
    /* floor(level * 0.25) + 7 points → in tenths */
    return (prayer_level / 4 + 7) * 10;
}

#undef PRAYER_OVERHEAD_DRAIN_RATE

/* Reward */

/* Every FcRewardParams field: its [env] key in config/fight_caves.ini, whether
 * it is an int, and its default. The adapter and the viewer read the keys. The
 * defaults equal the shipped values, so running without the config (the
 * viewer's fallback) computes the training reward. */
typedef struct {
    const char* key;
    size_t offset;          /* offsetof(FcRewardParams, field) */
    int is_int;
    float default_value;
} FcRewardSetting;

static const FcRewardSetting FC_REWARD_SETTINGS[] = {
    {"w_progress",
     offsetof(FcRewardParams, w_progress), 0, 0.001f},
    {"w_damage_taken",
     offsetof(FcRewardParams, w_damage_taken), 0, -0.25f},
    {"w_cave_complete",
     offsetof(FcRewardParams, w_cave_complete), 0, 1.0f},
    {"w_player_death",
     offsetof(FcRewardParams, w_player_death), 0, -1.0f},
    {"w_prayer_lost",
     offsetof(FcRewardParams, w_prayer_lost), 0, -0.02f},
    {"w_invalid_action",
     offsetof(FcRewardParams, w_invalid_action), 0, -0.1f},
    {"w_tick_penalty",
     offsetof(FcRewardParams, w_tick_penalty), 0, -0.0001f},

    {"shape_npc_heal_penalty",
     offsetof(FcRewardParams, shape_npc_heal_penalty), 0, -0.005f},
    {"shape_no_progress_penalty_1",
     offsetof(FcRewardParams, shape_no_progress_penalty_1), 0, -0.001f},
    {"shape_no_progress_penalty_2",
     offsetof(FcRewardParams, shape_no_progress_penalty_2), 0, -0.005f},
    {"shape_no_progress_penalty_3",
     offsetof(FcRewardParams, shape_no_progress_penalty_3), 0, -0.02f},
    {"shape_no_attack_base_penalty",
     offsetof(FcRewardParams, shape_no_attack_base_penalty), 0, -0.005f},
    {"shape_no_attack_wave_scale",
     offsetof(FcRewardParams, shape_no_attack_wave_scale), 0, 0.05f},

    {"shape_no_progress_start_1",
     offsetof(FcRewardParams, shape_no_progress_start_1), 1, 800},
    {"shape_no_progress_start_2",
     offsetof(FcRewardParams, shape_no_progress_start_2), 1, 1600},
    {"shape_no_progress_start_3",
     offsetof(FcRewardParams, shape_no_progress_start_3), 1, 2400},
    {"shape_no_attack_start",
     offsetof(FcRewardParams, shape_no_attack_start), 1, 50},
};
#define FC_REWARD_SETTING_COUNT \
    ((int)(sizeof(FC_REWARD_SETTINGS) / sizeof(FC_REWARD_SETTINGS[0])))
_Static_assert(sizeof(FC_REWARD_SETTINGS) / sizeof(FC_REWARD_SETTINGS[0]) ==
               sizeof(FcRewardParams) / sizeof(float),
               "FC_REWARD_SETTINGS needs one row per FcRewardParams field");

/* Store a value in the setting's field: ints truncate, floats round to float. */
static void fc_reward_setting_store(FcRewardParams* params,
                                    const FcRewardSetting* setting,
                                    double value) {
    char* field = (char*)params + setting->offset;
    if (setting->is_int) *(int*)field = (int)value;
    else *(float*)field = (float)value;
}

FcRewardParams fc_reward_default_params(void) {
    FcRewardParams params;
    memset(&params, 0, sizeof(params));
    for (int i = 0; i < FC_REWARD_SETTING_COUNT; i++)
        fc_reward_setting_store(&params, &FC_REWARD_SETTINGS[i],
                                FC_REWARD_SETTINGS[i].default_value);
    return params;
}

static void fc_reward_runtime_reset(FcRewardRuntime* runtime) {
    memset(runtime, 0, sizeof(*runtime));
}

static float clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static float fc_reward_required_work_remaining(const FcState* state) {
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

    return clamp01(
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
    return clamp01(
        ((float)waves_cleared + current_wave_progress) / (float)FC_NUM_WAVES);
}

static void fc_reward_sync_progress_state(
        FcState* state, const FcRewardRuntime* runtime) {
    state->progress_required_work_start = runtime->required_work_at_wave_start;
    state->progress_required_work_remaining = runtime->last_required_work_remaining;
    state->progress_current_wave_progress = runtime->last_current_wave_progress;
    state->progress_cave_progress = runtime->last_cave_progress;
    state->progress_ticks_since_positive = runtime->ticks_since_positive_progress;
}

static void fc_reward_runtime_begin_episode(
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

static FcRewardBreakdown fc_reward_compute_breakdown(
        const FcState* state, const FcRewardParams* params, FcRewardRuntime* runtime) {
    FcRewardBreakdown out;
    const FcPlayer* p = &state->player;

    memset(&out, 0, sizeof(out));

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
         * delta stays normalized for observations, while this channel pays
         * for actual HP/work removed and goes negative when healing restores
         * work. Multiplying by the wave's start work preserves wave-clear
         * handling and avoids treating the next wave spawn as negative
         * progress. */
        out.progress = net_work_removed * params->w_progress;

        runtime->last_required_work_remaining = work_remaining;
        runtime->last_current_wave_progress = wave_progress;
        runtime->last_cave_progress = cave_progress;

        if (net_work_removed > 0.0001f) {
            runtime->ticks_since_positive_progress = 0;
        } else {
            runtime->ticks_since_positive_progress++;
            if (net_work_removed >= -0.0001f) runtime->zero_progress_ticks++;
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

    float damage_taken = (p->max_hp > 0)
        ? (float)state->damage_taken_this_tick / (float)p->max_hp : 0.0f;
    out.damage_taken = damage_taken * params->w_damage_taken;
    /* Flags enter as (float)flag * weight: an unset flag gives a zero with
     * the weight's sign. */
    out.cave_complete = (float)(state->terminal == TERMINAL_CAVE_COMPLETE) *
                        params->w_cave_complete;
    out.player_death = (float)(state->terminal == TERMINAL_PLAYER_DEATH) *
                       params->w_player_death;
    out.prayer_lost = ((float)state->prayer_lost_this_tick / 10.0f) *
                      params->w_prayer_lost;
    out.invalid_action = (float)state->invalid_action_this_tick *
                         params->w_invalid_action;
    out.tick_penalty = params->w_tick_penalty;

    if (state->attack_attempt_this_tick > 0) {
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

    /* Penalty per heal that restored NPC HP this tick. */
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
        out.progress +
        out.damage_taken +
        out.cave_complete +
        out.player_death +
        out.prayer_lost +
        out.no_progress +
        out.no_attack +
        out.npc_heal +
        out.invalid_action +
        out.tick_penalty;

    runtime->npc_healing_total += (float)state->npc_heal_amount_this_tick;
    runtime->jad_healing_total += (float)state->jad_heal_amount_this_tick;
    return out;
}


/* Rng */
/*
 * XORshift32 RNG — single state, deterministic, seeded at reset.
 * All randomness in the simulation flows through this RNG.
 * Adopted from PufferLib's OSRS PvP environment.
 */

static void fc_rng_seed(FcState* state, uint32_t seed) {
    /* XORshift32 cannot have state 0 — if seed is 0, use a fixed nonzero value */
    state->rng_state = (seed != 0) ? seed : 0x12345678u;
    state->rng_seed = seed;
}

static uint32_t fc_rng_next(FcState* state) {
    uint32_t x = state->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state->rng_state = x;
    return x;
}

/* Uniform int in [0, max); 0 when max <= 0. */
static int fc_rng_int(FcState* state, int max) {
    if (max <= 0) return 0;
    return (int)(fc_rng_next(state) % (uint32_t)max);
}

/* Uniform float in [0, 1) from the low 24 bits. */
static float fc_rng_float(FcState* state) {
    return (float)(fc_rng_next(state) & 0x00FFFFFFu) / (float)0x01000000u;
}


/* Spawn */
static int fc_spawn_find_available_footprint(const FcState* state,
                                             int preferred_x, int preferred_y,
                                             int size, int max_radius,
                                             int* out_x, int* out_y) {
    uint8_t occupied[FC_ARENA_WIDTH][FC_ARENA_HEIGHT];
    fc_build_occupancy(state, occupied, -1);

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

static int fc_spawn_npc_first_free(FcState* state, int npc_type, int x, int y) {
    for (int slot = 0; slot < FC_MAX_NPCS; slot++) {
        if (state->npcs[slot].active) continue;
        fc_npc_spawn(&state->npcs[slot], npc_type, x, y,
                     state->next_spawn_index++);
        return slot;
    }
    return -1;
}


/* State */

/*
 * Arena maps, episode lifecycle, observations, masks and render snapshots.
 * FcState is caller-allocated; fc_reset zeroes it with memset before setting
 * up an episode.
 */

/*
 * Arena maps from the Void 634 cache (region 37,79, level 0), extracted via
 * DumpFcCollision.kt. fightcaves.collision stores whole-tile blocking as 64*64
 * row-major bytes, fightcaves.movement the corresponding directional wall bits
 * and fightcaves.los the projectile collision flags.
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

static void fail_required_arena_asset(const char* filename,
                                      const char* override_name) {
    fprintf(stderr,
            "fatal: required Fight Caves arena asset '%s' is missing, "
            "unreadable, or not exactly %d bytes.\n"
            "Set %s to the asset's absolute path or reinstall the pinned "
            "bundles with 'python3 ocean/fight_caves/tools.py "
            "setup --force'.\n",
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
        snprintf(path, sizeof(path), "resources/fight_caves/runtime/%s", filename);
        file = fopen(path, "rb");
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

/* Player initialization — the loadout is the combat-state authority. The
 * simulation test checks the loadout itself. */
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

/* fc_reset has zeroed the player; this sets the non-zero starting values. */
static void init_player(FcPlayer* p) {
    apply_loadout_combat_fields(p, &FC_LOADOUT);
    p->x = FC_ARENA_WIDTH / 2;
    p->y = FC_ARENA_HEIGHT / 2;
    p->current_hp = p->max_hp;
    p->current_prayer = p->max_prayer;
    p->run_energy = FC_RUN_ENERGY_MAX;
    p->is_running = 1;
    set_player_target(p, -1);
}

void fc_init(FcState* state) {
    memset(state, 0, sizeof(FcState));
    state->active_loadout = FC_LOADOUT_SOTA_TBOW;
}

/* Start an episode from the seed: seeds the RNG, selects a random wave
 * rotation, sets up the arena and player, and spawns wave 1 at tick 0. */
static void fc_reset(FcState* state, uint32_t seed) {
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
    state->active_loadout = FC_LOADOUT_SOTA_TBOW;
    state->render_events.player_attack_target_npc_slot = -1;
    state->render_events.player_move_start_x = state->player.x;
    state->render_events.player_move_start_y = state->player.y;

    /* Spawn wave 1 NPCs */
    state->current_wave = 1;
    fc_wave_spawn(state, 1);
}

/* Advance one tick with the canonical core action heads; a terminal episode
 * stays where it is. */
static void fc_step(FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
    if (state->terminal != TERMINAL_NONE) return;  /* episode over */
    fc_tick(state, actions);
}

/* Canonical client preference request used by playable frontends. Gameplay
 * movement still occurs only through fc_step(). */
static void fc_request_set_running(FcState* state, int enabled) {
    if (!state || state->terminal != TERMINAL_NONE) return;
    state->player.is_running = enabled &&
        state->player.run_energy >= FC_RUN_ENERGY_MIN_START;
}

static void fc_destroy(FcState* state) {
    /* Currently no heap allocations. Zero for safety. */
    memset(state, 0, sizeof(FcState));
}

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

/* Fill out_indices with the active NPC array indices currently visible in
 * policy NPC slots, using the same ordering as fc_write_obs and fc_write_mask.
 * Returns the number of filled slots, capped at FC_VISIBLE_NPCS. */
static int fc_visible_npc_indices(const FcState* state, int out_indices[FC_VISIBLE_NPCS]) {
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

/* An already-active run may consume its remaining energy below 1%. Starting
 * run mode follows the OSRS client/server minimum of one displayed percent. */
static int fc_player_can_run(const FcPlayer* player) {
    return player->run_energy > 0 &&
           (player->is_running ||
            player->run_energy >= FC_RUN_ENERGY_MIN_START);
}

static int move_action_valid(const FcState* state, int action) {
    const FcPlayer* p = &state->player;

    if (action < 0 || action >= FC_MOVE_DIM) return 0;
    if (action == FC_MOVE_IDLE) return 1;
    if (action >= FC_MOVE_RUN_N && !fc_player_can_run(p)) {
        return 0;
    }

    /* Legal when the first step toward the direction's target is open. */
    int x = p->x;
    int y = p->y;
    return fc_step_toward(&x, &y, p->x + FC_MOVE_DX[action],
                          p->y + FC_MOVE_DY[action], 1, state->walkable,
                          state->movement_flags, NULL);
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

static int fc_eat_action_valid(const FcState* state, int action) {
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

static int fc_drink_action_valid(const FcState* state, int action) {
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
        maximum += 10 * stats->max_hit[ATTACK_MELEE];
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
    if (can_melee && stats->max_hit[ATTACK_MELEE] > 0) return ATTACK_MELEE;
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
    memset(out, 0, sizeof(float) * FC_POLICY_OBS_SIZE);

    const FcPlayer* p = &state->player;
    float prayer_deadline_urgency[3] = {0.0f, 0.0f, 0.0f};

    /* Prayer deadline urgency per style: the most urgent pending hit whose
     * prayer snapshot has not locked yet, which is the decision window. */
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
    int visible_indices[FC_VISIBLE_NPCS];
    int visible = fc_visible_npc_indices(state, visible_indices);
    for (int attack = visible + 1; attack < FC_ATTACK_DIM; attack++) {
        out[FC_MASK_ATTACK_START + attack] = 0.0f;
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

static int fc_is_terminal(const FcState* state) {
    return state->terminal != TERMINAL_NONE;
}

/* Snapshot of the visible entities for the viewer; entities[] must have room
 * for FC_MAX_RENDER_ENTITIES. Entity 0 is always the player, followed by every
 * active NPC and every NPC that died this tick. */
static void fc_fill_render_entities(const FcState* state, FcRenderEntity* entities, int* count) {
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
        ne->is_dead = n->is_dead;
        ne->died_this_tick = n->died_this_tick;
        ne->npc_slot = i;
    }

    *count = idx;
}

/* Copy the authoritative transition facts captured during the last tick. */
static void fc_fill_render_events(const FcState* state, FcRenderEvents* events) {
    if (!state || !events) return;
    *events = state->render_events;
}


/* Tick */

/*
 * Processing order of one tick (fc_tick):
 *
 *   1. Clear per-tick event flags
 *   2. Process player actions:
 *      a. Prayer toggle (instant)
 *      b. Eat food / drink potion (if timer ready)
 *      c. Attack initiation from the pre-movement tile
 *      d. Movement (route or directional head), unless an attack fired
 *      e. Run-energy drain or restoration from actual movement
 *   3. Decrement player timers (attack, food, potion, combo)
 *   4. Prayer drain (only if prayer stayed active across the tick boundary),
 *      then HP regeneration
 *   5. NPC AI tick (movement + attack) for all active NPCs
 *   6. Resolve pending hits (NPC → player, player → NPC), then death timers
 *   7. Check terminal conditions, wave advancement and Jad healer spawns
 *   8. Episode analytics
 *   9. Increment tick and lock prayer snapshots due at the new boundary
 */

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
    state->movement_this_tick = 0;
    state->idle_this_tick = 0;
    state->food_used_this_tick = 0;
    state->prayer_potion_used_this_tick = 0;
    state->jad_heal_procs_this_tick = 0;
    state->npc_heal_procs_this_tick = 0;
    state->npc_heal_amount_this_tick = 0;
    state->jad_heal_amount_this_tick = 0;

    FcPlayer* p = &state->player;
    p->damage_taken_this_tick = 0;
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

/* Same ordering as observation writer — must be identical for consistency */
static int npc_slot_to_index(const FcState* state, int slot) {
    int visible_indices[FC_VISIBLE_NPCS];
    int visible = fc_visible_npc_indices(state, visible_indices);
    if (slot < 0 || slot >= visible) return -1;
    return visible_indices[slot];
}

static void apply_player_prayer_action(
    FcState* state, int action, FcPrayerTransition* transition) {
    FcPlayer* player = &state->player;

    *transition = fc_prayer_apply_action(player, action);
    state->render_events.prayer_flick_performed = transition->flicked;
    if (transition->changed || transition->flicked) {
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
        int restore = fc_prayer_potion_restore(FC_LOADOUT.prayer_lvl);
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
        set_player_target(player,
                          explicit_attack ? player->attack_target_idx : -1);
    }

    /* Target selection precedes movement so movement cannot rebind a slot or
     * make the selected attack valid retroactively. */
    if (explicit_attack && requested_attack_idx >= 0 &&
        state->npcs[requested_attack_idx].active &&
        !state->npcs[requested_attack_idx].is_dead) {
        if (player->attack_target_idx != requested_attack_idx) {
            set_player_target(player, requested_attack_idx);
        }
        player->approach_target = explicit_move ? 0 : 1;
    }
}

static void launch_player_attack(FcState* state, FcNpc* target, int distance) {
    FcPlayer* player = &state->player;
    int melee = player->weapon_kind == FC_WEAPON_UNARMED;
    int att_roll, max_hit;
    if (melee) {
        /* Unarmed Punch is accurate/crush (+3 Attack). Fight Caves NPCs all
         * have zero crush defence bonus, including the ranged-resistant
         * healers. */
        att_roll = (player->attack_level + 11) * (player->melee_attack_bonus + 64);
        max_hit = (320 + (player->strength_level + 8) *
                   (player->melee_strength_bonus + 64)) / 640;
    } else {
        fc_player_ranged_rolls(player, target, &att_roll, &max_hit);
    }
    const FcNpcStats* target_stats = fc_npc_get_stats(target->npc_type);
    int def_roll = fc_npc_roll(target_stats->def_level,
                               melee ? 0 : target_stats->ranged_def_bonus);
    float chance = fc_hit_chance(att_roll, def_roll);
    int hit = fc_rng_float(state) < chance ? 1 : 0;
    int damage = hit ? fc_roll_damage_tenths(state, max_hit, 1) : 0;
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

    player->attack_timer = player->weapon_speed;
    if (player->weapon_uses_ammo && player->ammo_count > 0) {
        fc_items_spend_ammo(player);
    }
    player->hit_landed_this_tick = 1;
}

static void process_player_target(FcState* state,
                                  int explicit_directional_move,
                                  int explicit_tile_move) {
    FcPlayer* player = &state->player;

    /* Like Void CombatMovement: approach until the current target is in range,
     * then attack on cooldown and remain stationary for this tick. */
    if (player->attack_target_idx < 0 ||
        (player->weapon_uses_ammo && player->ammo_count <= 0)) {
        return;
    }

    FcNpc* target = &state->npcs[player->attack_target_idx];
    if (!target->active || target->is_dead) {
        set_player_target(player, -1);
        return;
    }

    int dist = fc_distance_to_npc(player->x, player->y, target);
    int weapon_range = player->weapon_kind == FC_WEAPON_UNARMED
        ? FC_ROUTE_MELEE_RANGE : player->weapon_range;
    int target_can_fire = fc_can_attack_from(
        player->x, player->y, target->x, target->y, target->size,
        weapon_range, state->movement_flags, state->los_flags);
    int target_ready = player->attack_timer <= 0;

    int route_endpoint_can_fire = 0;
    int target_moved =
        player->approach_target_x != target->x ||
        player->approach_target_y != target->y ||
        player->approach_target_size != target->size;
    if (player->route_idx < player->route_len) {
        int endpoint = player->route_len - 1;
        route_endpoint_can_fire = fc_can_attack_from(
            player->route_x[endpoint], player->route_y[endpoint],
            target->x, target->y, target->size, weapon_range,
            state->movement_flags, state->los_flags);
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
            set_player_target(player, -1);
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
            if (fc_footprint_step_valid(
                    player->x, player->y, dx, dy, 1,
                    state->walkable, state->movement_flags, NULL)) {
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
        int wants_run = move_action >= FC_MOVE_RUN_N;
        int max_steps = wants_run
            ? (fc_player_can_run(player) ? 2 : 0)
            : 1;
        int old_x = player->x;
        int old_y = player->y;
        int target_x = old_x + FC_MOVE_DX[move_action];
        int target_y = old_y + FC_MOVE_DY[move_action];
        while (moved_steps < max_steps &&
               fc_step_toward(&player->x, &player->y, target_x, target_y, 1,
                              state->walkable, state->movement_flags, NULL)) {
            record_player_move_waypoint(state);
            moved_steps++;
        }
        if (moved_steps > 0) {
            set_player_facing_from_delta(
                player, (float)(player->x - old_x),
                (float)(player->y - old_y));
            state->movement_this_tick = 1;
            player->is_running = moved_steps >= 2;
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

static void process_player_actions(FcState* state,
                                   const int actions[FC_NUM_ACTION_HEADS],
                                   FcPrayerTransition* prayer_transition) {
    FcPlayer* p = &state->player;

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
    /* Invalid-action reward is binary across the three policy heads. */
    state->invalid_action_this_tick =
        !move_action_valid(state, act_move) ||
        !attack_action_valid(state, act_attack) || !prayer_action_valid(act_prayer);

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

    process_player_target(
        state, explicit_directional_move, explicit_tile_move);

    int moved_steps = process_player_movement(
        state, act_move, act_target_x, act_target_y,
        explicit_move, explicit_attack);
    update_player_run_energy(p, moved_steps);
}

static void decrement_player_timers(FcPlayer* p) {
    if (p->attack_timer > 0) p->attack_timer--;
    if (p->food_timer > 0) p->food_timer--;
    if (p->potion_timer > 0) p->potion_timer--;
    if (p->combo_timer > 0) p->combo_timer--;
}

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
    int jad_idx = find_active_jad(state);
    if (jad_idx < 0) return;
    FcNpc* jad = &state->npcs[jad_idx];

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
}

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

static void fc_tick(FcState* state, const int actions[FC_NUM_ACTION_HEADS]) {
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
                             prayer_transition.flicked);
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
 * The 63-wave spawn table with 15 rotations, sourced from the OSRS wiki and
 * the Kotlin archive's tzhaar_fight_cave_waves.toml.
 *
 * Pattern: each new NPC tier is introduced on waves 1,3,7,15,31,63.
 * The binary pattern gives 2^tier - 1 waves before each new tier.
 * fc_spawn_position maps spawn directions to arena tiles.
 */

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

/* Each pattern gives one NPC slot's spawn direction across all 15 rotations.
 * Wave slots reuse these 15 patterns. Pattern IDs index the rows below;
 * they are independent of rotation IDs and preserve the original slot order. */
static const uint8_t SPAWN_PATTERNS[15][FC_NUM_ROTATIONS] = {
    { /* Pattern 0 */
        SPAWN_CENTER, SPAWN_CENTER, SPAWN_CENTER, SPAWN_NORTH_WEST,
        SPAWN_NORTH_WEST, SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH,
        SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST,
        SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST,
    },
    { /* Pattern 1 */
        SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_CENTER,
        SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_NORTH_WEST,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST,
        SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST,
    },
    { /* Pattern 2 */
        SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_CENTER,
        SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH,
        SPAWN_NORTH_WEST, SPAWN_NORTH_WEST, SPAWN_SOUTH,
    },
    { /* Pattern 3 */
        SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST,
        SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST,
        SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_CENTER,
    },
    { /* Pattern 4 */
        SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST,
        SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH, SPAWN_SOUTH_EAST,
        SPAWN_CENTER, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_NORTH_WEST,
        SPAWN_SOUTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST,
    },
    { /* Pattern 5 */
        SPAWN_NORTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_CENTER,
        SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH,
        SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH,
        SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST,
    },
    { /* Pattern 6 */
        SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_NORTH_WEST,
        SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH, SPAWN_SOUTH_EAST,
        SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST,
    },
    { /* Pattern 7 */
        SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH,
        SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_CENTER, SPAWN_CENTER,
        SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST, SPAWN_NORTH_WEST,
    },
    { /* Pattern 8 */
        SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST,
        SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER,
        SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST,
        SPAWN_CENTER, SPAWN_CENTER, SPAWN_SOUTH,
    },
    { /* Pattern 9 */
        SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_SOUTH,
        SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST,
        SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST,
        SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST, SPAWN_CENTER,
    },
    { /* Pattern 10 */
        SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_NORTH_WEST,
        SPAWN_CENTER, SPAWN_CENTER, SPAWN_SOUTH, SPAWN_SOUTH_WEST,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST,
        SPAWN_SOUTH, SPAWN_SOUTH_EAST, SPAWN_SOUTH,
    },
    { /* Pattern 11 */
        SPAWN_SOUTH, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST,
        SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_CENTER, SPAWN_NORTH_WEST,
        SPAWN_SOUTH, SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH,
        SPAWN_CENTER, SPAWN_NORTH_WEST, SPAWN_SOUTH_WEST,
    },
    { /* Pattern 12 */
        SPAWN_CENTER, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_CENTER,
        SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_NORTH_WEST, SPAWN_CENTER,
        SPAWN_SOUTH_WEST, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST,
        SPAWN_NORTH_WEST, SPAWN_SOUTH, SPAWN_SOUTH_EAST,
    },
    { /* Pattern 13 */
        SPAWN_SOUTH_EAST, SPAWN_SOUTH, SPAWN_SOUTH, SPAWN_SOUTH_WEST,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST, SPAWN_SOUTH_WEST,
        SPAWN_NORTH_WEST, SPAWN_NORTH_WEST, SPAWN_CENTER, SPAWN_NORTH_WEST,
        SPAWN_SOUTH, SPAWN_CENTER, SPAWN_CENTER,
    },
    { /* Pattern 14 */
        SPAWN_SOUTH_WEST, SPAWN_NORTH_WEST, SPAWN_NORTH_WEST, SPAWN_SOUTH,
        SPAWN_SOUTH, SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH_EAST,
        SPAWN_CENTER, SPAWN_SOUTH_WEST, SPAWN_SOUTH, SPAWN_CENTER,
        SPAWN_SOUTH_EAST, SPAWN_SOUTH_EAST, SPAWN_NORTH_WEST,
    },
};

/* One pattern ID per active NPC slot; unused slots are zero-initialized. */
static const uint8_t WAVE_SPAWN_PATTERNS[FC_NUM_WAVES][FC_MAX_SPAWNS_PER_WAVE] = {
    /* Wave  1 */ {0},
    /* Wave  2 */ {1, 2},
    /* Wave  3 */ {2},
    /* Wave  4 */ {3, 4},
    /* Wave  5 */ {5, 6, 3},
    /* Wave  6 */ {5, 6},
    /* Wave  7 */ {6},
    /* Wave  8 */ {7, 8},
    /* Wave  9 */ {9, 10, 7},
    /* Wave 10 */ {10, 9},
    /* Wave 11 */ {11, 12, 10},
    /* Wave 12 */ {13, 14, 11, 12},
    /* Wave 13 */ {13, 14, 11},
    /* Wave 14 */ {13, 14},
    /* Wave 15 */ {14},
    /* Wave 16 */ {1, 0},
    /* Wave 17 */ {2, 4, 1},
    /* Wave 18 */ {4, 2},
    /* Wave 19 */ {5, 3, 4},
    /* Wave 20 */ {6, 8, 5, 3},
    /* Wave 21 */ {6, 8, 5},
    /* Wave 22 */ {8, 6},
    /* Wave 23 */ {9, 7, 8},
    /* Wave 24 */ {10, 12, 9, 7},
    /* Wave 25 */ {12, 10, 9},
    /* Wave 26 */ {13, 11, 12, 10},
    /* Wave 27 */ {14, 0, 13, 11, 12},
    /* Wave 28 */ {14, 0, 13, 11},
    /* Wave 29 */ {14, 0, 13},
    /* Wave 30 */ {14, 0},
    /* Wave 31 */ {0},
    /* Wave 32 */ {2, 1},
    /* Wave 33 */ {4, 3, 2},
    /* Wave 34 */ {3, 4},
    /* Wave 35 */ {6, 5, 3},
    /* Wave 36 */ {8, 7, 6, 5},
    /* Wave 37 */ {8, 7, 6},
    /* Wave 38 */ {7, 8},
    /* Wave 39 */ {10, 9, 7},
    /* Wave 40 */ {12, 11, 10, 9},
    /* Wave 41 */ {11, 12, 10},
    /* Wave 42 */ {14, 13, 11, 12},
    /* Wave 43 */ {0, 1, 14, 13, 11},
    /* Wave 44 */ {0, 1, 14, 13},
    /* Wave 45 */ {0, 1, 14},
    /* Wave 46 */ {1, 0},
    /* Wave 47 */ {4, 2, 1},
    /* Wave 48 */ {3, 5, 4, 2},
    /* Wave 49 */ {5, 3, 4},
    /* Wave 50 */ {8, 6, 5, 3},
    /* Wave 51 */ {7, 9, 8, 6, 5},
    /* Wave 52 */ {7, 9, 8, 6},
    /* Wave 53 */ {9, 7, 8},
    /* Wave 54 */ {12, 10, 9, 7},
    /* Wave 55 */ {11, 13, 12, 10, 9},
    /* Wave 56 */ {13, 11, 12, 10},
    /* Wave 57 */ {0, 14, 13, 11, 12},
    /* Wave 58 */ {1, 2, 0, 14, 13, 11},
    /* Wave 59 */ {1, 2, 0, 14, 13},
    /* Wave 60 */ {1, 2, 0, 14},
    /* Wave 61 */ {1, 2, 0},
    /* Wave 62 */ {1, 2},
    /* Wave 63 */ {2},
};

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
static void fc_spawn_position(int spawn_dir, int* x, int* y) {
    switch (spawn_dir) {
        case SPAWN_SOUTH:      *x = 37; *y = 17; break;
        case SPAWN_SOUTH_WEST: *x = 14; *y = 17; break;
        case SPAWN_NORTH_WEST: *x = 13; *y = 49; break;
        case SPAWN_SOUTH_EAST: *x = 51; *y = 27; break;
        case SPAWN_CENTER:     *x = 32; *y = 32; break;
        default:               *x = 32; *y = 32; break;
    }
}

static const FcWaveEntry* fc_wave_get(int wave_num) {
    if (wave_num < 1 || wave_num > FC_NUM_WAVES) return &WAVE_TABLE[0];
    return &WAVE_TABLE[wave_num - 1];
}

static int fc_wave_spawn_dir(int wave_num, int rotation, int npc_index) {
    if (wave_num < 1 || wave_num > FC_NUM_WAVES) return SPAWN_CENTER;
    if (rotation < 0 || rotation >= FC_NUM_ROTATIONS) return SPAWN_CENTER;
    if (npc_index < 0 || npc_index >= FC_MAX_SPAWNS_PER_WAVE) return SPAWN_CENTER;
    /* The expanded table returned zero (SPAWN_SOUTH) for unused slots. */
    if (npc_index >= WAVE_TABLE[wave_num - 1].num_spawns) return SPAWN_SOUTH;
    int pattern = WAVE_SPAWN_PATTERNS[wave_num - 1][npc_index];
    return SPAWN_PATTERNS[pattern][rotation];
}

/* Spawn all NPCs for the given wave using the state's rotation_id. */
static void fc_wave_spawn(FcState* state, int wave_num) {
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

/* Once the wave is cleared, flag it and spawn the next wave, or complete the
 * cave after the last. */
static void fc_wave_check_advance(FcState* state) {
    /* Don't advance if wave hasn't started or NPCs still alive */
    if (state->current_wave <= 0) return;
    if (state->npcs_remaining > 0) return;
    if (state->terminal != TERMINAL_NONE) return;

    state->wave_just_cleared = 1;

    /* Check if all waves complete */
    if (state->current_wave >= FC_NUM_WAVES) {
        state->terminal = TERMINAL_CAVE_COMPLETE;
        return;
    }

    /* Advance to next wave */
    state->current_wave++;
    state->jad_healers_spawned = 0;
    state->jad_healer_spawn_generations = 0;
    fc_wave_spawn(state, state->current_wave);
}


#endif /* FC_CORE_IMPLEMENTATION */
#endif /* FIGHT_CAVES_CORE_H */
