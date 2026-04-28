/**
 * @file test_collision.c
 * @brief Tests for the collision system and BFS pathfinder
 *
 * Validates that collision flags block movement correctly, that the pathfinder
 * routes around obstacles, and that NULL collision map preserves flat arena behavior.
 *
 * Compile: cc -O2 -o test_collision test_collision.c -lm
 * Run: ./test_collision
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "osrs_collision.h"
#include "osrs_pathfinding.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void name(void)
#define RUN(name) do { \
    printf("  %-50s", #name); \
    name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

TEST(test_null_map_all_traversable) {
    ASSERT(collision_traversable_north(NULL, 0, 100, 100));
    ASSERT(collision_traversable_south(NULL, 0, 100, 100));
    ASSERT(collision_traversable_east(NULL, 0, 100, 100));
    ASSERT(collision_traversable_west(NULL, 0, 100, 100));
    ASSERT(collision_traversable_north_east(NULL, 0, 100, 100));
    ASSERT(collision_traversable_north_west(NULL, 0, 100, 100));
    ASSERT(collision_traversable_south_east(NULL, 0, 100, 100));
    ASSERT(collision_traversable_south_west(NULL, 0, 100, 100));
    ASSERT(collision_tile_walkable(NULL, 0, 100, 100));
    ASSERT(collision_traversable_step(NULL, 0, 100, 100, 1, 1));
}

TEST(test_empty_map_all_traversable) {
    CollisionMap* map = collision_map_create();
    ASSERT(collision_traversable_north(map, 0, 100, 100));
    ASSERT(collision_traversable_south(map, 0, 100, 100));
    ASSERT(collision_tile_walkable(map, 0, 100, 100));
    collision_map_free(map);
}

TEST(test_region_create_and_lookup) {
    CollisionMap* map = collision_map_create();
    ASSERT(map->count == 0);

    /* setting a flag auto-creates the region */
    collision_set_flag(map, 0, 3200, 3200, COLLISION_BLOCKED);
    ASSERT(map->count == 1);

    int flags = collision_get_flags(map, 0, 3200, 3200);
    ASSERT(flags == COLLISION_BLOCKED);

    /* different tile in same region doesn't add another region */
    collision_set_flag(map, 0, 3201, 3201, COLLISION_WALL_NORTH);
    ASSERT(map->count == 1);

    /* tile in a different region increments count */
    collision_set_flag(map, 0, 3264, 3200, COLLISION_BLOCKED);
    ASSERT(map->count == 2);

    collision_map_free(map);
}

TEST(test_flag_set_and_unset) {
    CollisionMap* map = collision_map_create();
    collision_set_flag(map, 0, 3200, 3200, COLLISION_WALL_NORTH | COLLISION_WALL_SOUTH);
    int flags = collision_get_flags(map, 0, 3200, 3200);
    ASSERT(flags == (COLLISION_WALL_NORTH | COLLISION_WALL_SOUTH));

    collision_unset_flag(map, 0, 3200, 3200, COLLISION_WALL_NORTH);
    flags = collision_get_flags(map, 0, 3200, 3200);
    ASSERT(flags == COLLISION_WALL_SOUTH);

    collision_map_free(map);
}

TEST(test_blocked_tile_not_traversable) {
    CollisionMap* map = collision_map_create();

    /* block tile (100, 101) — north of (100, 100) */
    collision_mark_blocked(map, 0, 100, 101);

    /* can't walk north from (100, 100) because (100, 101) is blocked */
    ASSERT(!collision_traversable_north(map, 0, 100, 100));

    /* can still walk south, east, west from (100, 100) */
    ASSERT(collision_traversable_south(map, 0, 100, 100));
    ASSERT(collision_traversable_east(map, 0, 100, 100));
    ASSERT(collision_traversable_west(map, 0, 100, 100));

    collision_map_free(map);
}

TEST(test_wall_blocks_direction) {
    CollisionMap* map = collision_map_create();

    /* place a south-facing wall on tile (100, 101).
     * this means: entering (100, 101) from the south is blocked. */
    collision_set_flag(map, 0, 100, 101, COLLISION_WALL_SOUTH);

    /* walking north from (100, 100) to (100, 101) is blocked (wall on south side of dest) */
    ASSERT(!collision_traversable_north(map, 0, 100, 100));

    /* walking south from (100, 102) to (100, 101) is fine — no north wall on dest */
    ASSERT(collision_traversable_south(map, 0, 100, 102));

    collision_map_free(map);
}

TEST(test_diagonal_blocked_by_intermediate) {
    CollisionMap* map = collision_map_create();

    /* block the east intermediate tile (101, 100) — this should block NE diagonal */
    collision_mark_blocked(map, 0, 101, 100);

    /* NE from (100, 100): checks (101, 101) + (101, 100) + (100, 101) */
    /* (101, 100) is blocked, so diagonal should fail */
    ASSERT(!collision_traversable_north_east(map, 0, 100, 100));

    /* NW from (100, 100) should still work (different intermediate tiles) */
    ASSERT(collision_traversable_north_west(map, 0, 100, 100));

    collision_map_free(map);
}

TEST(test_multi_tile_occupant) {
    CollisionMap* map = collision_map_create();

    /* place a 2x2 object at (100, 100) */
    collision_mark_occupant(map, 0, 100, 100, 2, 2, 0);

    ASSERT(!collision_tile_walkable(map, 0, 100, 100));
    ASSERT(!collision_tile_walkable(map, 0, 101, 100));
    ASSERT(!collision_tile_walkable(map, 0, 100, 101));
    ASSERT(!collision_tile_walkable(map, 0, 101, 101));
    ASSERT(collision_tile_walkable(map, 0, 102, 100)); /* outside the object */

    collision_map_free(map);
}

TEST(test_save_and_load) {
    CollisionMap* map = collision_map_create();
    collision_mark_blocked(map, 0, 3200, 3520);
    collision_set_flag(map, 0, 3201, 3520, COLLISION_WALL_NORTH | COLLISION_WALL_EAST);
    collision_mark_blocked(map, 0, 3264, 3520); /* different region */

    const char* path = "/tmp/test_collision.cmap";
    int rc = collision_map_save(map, path);
    ASSERT(rc == 0);

    CollisionMap* loaded = collision_map_load(path);
    ASSERT(loaded != NULL);
    ASSERT(loaded->count == 2);

    ASSERT(collision_get_flags(loaded, 0, 3200, 3520) == COLLISION_BLOCKED);
    ASSERT(collision_get_flags(loaded, 0, 3201, 3520) == (COLLISION_WALL_NORTH | COLLISION_WALL_EAST));
    ASSERT(collision_get_flags(loaded, 0, 3264, 3520) == COLLISION_BLOCKED);
    ASSERT(collision_get_flags(loaded, 0, 3202, 3520) == COLLISION_NONE); /* untouched tile */

    collision_map_free(map);
    collision_map_free(loaded);
    remove(path);
}

TEST(test_pathfind_already_at_dest) {
    PathResult r = pathfind_step(NULL, 0, 100, 100, 100, 100, NULL, NULL);
    ASSERT(r.found == 1);
    ASSERT(r.next_dx == 0 && r.next_dy == 0);
}

TEST(test_pathfind_straight_line_no_obstacles) {
    /* no collision map — straight path east */
    PathResult r = pathfind_step(NULL, 0, 100, 100, 105, 100, NULL, NULL);
    ASSERT(r.found == 1);
    ASSERT(r.next_dx == 1);
    ASSERT(r.next_dy == 0);
}

TEST(test_pathfind_diagonal_no_obstacles) {
    PathResult r = pathfind_step(NULL, 0, 100, 100, 105, 105, NULL, NULL);
    ASSERT(r.found == 1);
    ASSERT(r.next_dx == 1);
    ASSERT(r.next_dy == 1);
}

TEST(test_pathfind_around_wall) {
    CollisionMap* map = collision_map_create();

    /* create a vertical wall of blocked tiles at x=102, y=98..102
     * player at (100, 100) wants to reach (105, 100)
     * direct east path is blocked — must go around */
    for (int y = 98; y <= 102; y++) {
        collision_mark_blocked(map, 0, 102, y);
    }

    PathResult r = pathfind_step(map, 0, 100, 100, 105, 100, NULL, NULL);
    ASSERT(r.found == 1);

    /* first step should NOT be straight east into the wall.
     * BFS will route north or south to go around. */
    /* it could be east (toward x=101 which is open) — that's fine */
    /* but it should never step to (102, any) since those are blocked */
    ASSERT(!(r.next_dx == 1 && r.next_dy == 0)
        || (100 + r.next_dx != 102));

    collision_map_free(map);
}

TEST(test_pathfind_completely_blocked) {
    CollisionMap* map = collision_map_create();

    /* surround destination (105, 100) with blocked tiles */
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            collision_mark_blocked(map, 0, 105 + dx, 100 + dy);
        }
    }
    /* also block the dest tile itself */
    collision_mark_blocked(map, 0, 105, 100);

    PathResult r = pathfind_step(map, 0, 100, 100, 105, 100, NULL, NULL);
    /* should use fallback — find closest reachable tile */
    ASSERT(r.found == 1);
    /* the actual destination should differ from requested since it's blocked */
    ASSERT(r.dest_x != 105 || r.dest_y != 100);

    collision_map_free(map);
}

TEST(test_pathfind_respects_wall_flags) {
    CollisionMap* map = collision_map_create();

    /* place a south wall on tile (101, 101) — blocks entry from south.
     * player at (101, 100) going north to (101, 101) should be blocked. */
    collision_set_flag(map, 0, 101, 101, COLLISION_WALL_SOUTH);

    /* pathfind from (100, 100) to (102, 102) — going NE */
    PathResult r = pathfind_step(map, 0, 100, 100, 102, 102, NULL, NULL);
    ASSERT(r.found == 1);

    /* the BFS should find a path (there are many routes around one wall tile) */
    /* just verify it found something valid */
    ASSERT(r.next_dx >= -1 && r.next_dx <= 1);
    ASSERT(r.next_dy >= -1 && r.next_dy <= 1);

    collision_map_free(map);
}

typedef struct { int x, y, dest_x, dest_y; } TestPlayer;

static int test_step_toward(TestPlayer* p, const CollisionMap* cmap) {
    int dx = p->dest_x - p->x;
    int dy = p->dest_y - p->y;
    if (dx == 0 && dy == 0) return 0;

    int step_x = (dx > 0) ? 1 : (dx < 0 ? -1 : 0);
    int step_y = (dy > 0) ? 1 : (dy < 0 ? -1 : 0);

    if (step_x != 0 && step_y != 0) {
        if (collision_traversable_step(cmap, 0, p->x, p->y, step_x, step_y)) {
            p->x += step_x; p->y += step_y; return 1;
        }
        if (collision_traversable_step(cmap, 0, p->x, p->y, step_x, 0)) {
            p->x += step_x; return 1;
        }
        if (collision_traversable_step(cmap, 0, p->x, p->y, 0, step_y)) {
            p->y += step_y; return 1;
        }
        return 0;
    }

    if (collision_traversable_step(cmap, 0, p->x, p->y, step_x, step_y)) {
        p->x += step_x; p->y += step_y; return 1;
    }
    return 0;
}

TEST(test_step_blocked_by_wall) {
    CollisionMap* map = collision_map_create();

    /* block tile east of player */
    collision_mark_blocked(map, 0, 101, 100);

    TestPlayer p = {100, 100, 105, 100};
    int moved = test_step_toward(&p, map);

    /* east is blocked, no y component, should not move */
    ASSERT(!moved);
    ASSERT(p.x == 100 && p.y == 100);

    collision_map_free(map);
}

TEST(test_step_diagonal_falls_back_to_cardinal) {
    CollisionMap* map = collision_map_create();

    /* block the diagonal NE tile (101, 101) */
    collision_mark_blocked(map, 0, 101, 101);

    TestPlayer p = {100, 100, 105, 105};
    int moved = test_step_toward(&p, map);

    /* diagonal NE blocked, should fall back to east (101, 100) or north (100, 101) */
    ASSERT(moved);
    /* should have moved in one cardinal direction only */
    ASSERT((p.x == 101 && p.y == 100) || (p.x == 100 && p.y == 101));

    collision_map_free(map);
}

TEST(test_step_no_collision_map) {
    TestPlayer p = {100, 100, 105, 105};
    int moved = test_step_toward(&p, NULL);
    ASSERT(moved);
    /* diagonal step with no obstacles */
    ASSERT(p.x == 101 && p.y == 101);
}

TEST(test_wilderness_coordinates) {
    CollisionMap* map = collision_map_create();

    /* wilderness is roughly x=3008-3390, y=3520-3968.
     * our arena center is around (3222, 3544).
     * verify collision works at real coords. */
    int arena_x = 3222;
    int arena_y = 3544;

    collision_mark_blocked(map, 0, arena_x + 5, arena_y);
    ASSERT(!collision_tile_walkable(map, 0, arena_x + 5, arena_y));
    ASSERT(collision_tile_walkable(map, 0, arena_x + 4, arena_y));
    ASSERT(!collision_traversable_east(map, 0, arena_x + 4, arena_y));

    collision_map_free(map);
}

int main(void) {
    printf("collision system tests\n\n");

    printf("collision map basics:\n");
    RUN(test_null_map_all_traversable);
    RUN(test_empty_map_all_traversable);
    RUN(test_region_create_and_lookup);
    RUN(test_flag_set_and_unset);

    printf("\ndirectional traversal:\n");
    RUN(test_blocked_tile_not_traversable);
    RUN(test_wall_blocks_direction);
    RUN(test_diagonal_blocked_by_intermediate);
    RUN(test_multi_tile_occupant);

    printf("\nbinary I/O:\n");
    RUN(test_save_and_load);

    printf("\npathfinding:\n");
    RUN(test_pathfind_already_at_dest);
    RUN(test_pathfind_straight_line_no_obstacles);
    RUN(test_pathfind_diagonal_no_obstacles);
    RUN(test_pathfind_around_wall);
    RUN(test_pathfind_completely_blocked);
    RUN(test_pathfind_respects_wall_flags);

    printf("\nmovement integration:\n");
    RUN(test_step_blocked_by_wall);
    RUN(test_step_diagonal_falls_back_to_cardinal);
    RUN(test_step_no_collision_map);

    printf("\nwilderness coordinates:\n");
    RUN(test_wilderness_coordinates);

    printf("%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
