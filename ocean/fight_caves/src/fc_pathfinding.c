#include "fc_pathfinding.h"
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
    const uint8_t los_flags[FC_ARENA_WIDTH][FC_ARENA_HEIGHT]) {
    if (kind == FC_ROUTE_EXACT) return x == dst_x && y == dst_y;
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
                                  attack_range, los_flags)) {
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
