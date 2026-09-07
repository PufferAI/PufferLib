#ifndef FC_CONTRACTS_H
#define FC_CONTRACTS_H

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

#endif /* FC_CONTRACTS_H */
