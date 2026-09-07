#ifndef FC_PATHFINDING_H
#define FC_PATHFINDING_H

#include "fc_types.h"

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

#endif /* FC_PATHFINDING_H */
