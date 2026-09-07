#ifndef FC_API_H
#define FC_API_H

#include "fc_types.h"
#include "fc_contracts.h"
#include "fc_episode_summary.h"

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

/* Version 4 removes redundant compatibility/temporary fields from the
 * complete core-owned fixed-width FcState serialization. */
#define FC_STATE_HASH_VERSION 4u

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

#endif /* FC_API_H */
