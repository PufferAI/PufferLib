/**
 * @file test_npc_movement.c
 * @brief Tests for encounter_npc_step_toward (shared OSRS NPC chase step).
 *
 * Regression coverage for: ranged NPCs without LOS must keep walking, while
 * melee NPCs using stop_at_melee_distance must not step onto their target.
 *
 * Compile: cc -std=c11 -O0 -g -I. -Iocean/osrs -o test_npc_movement
 *                 ocean/osrs/tests/test_npc_movement.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "osrs_encounter.h"
#include "osrs_collision.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_EQ(label, actual, expected) do { \
    tests_run++; \
    if ((actual) == (expected)) { tests_passed++; } \
    else { printf("  FAIL %s: expected %d, got %d\n", label, (int)(expected), (int)(actual)); } \
} while (0)

#define ASSERT(label, cond) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { printf("  FAIL %s\n", label); } \
} while (0)

/* open-tile callback: nothing blocks */
static int blocked_never(void* ctx, int x, int y, int size) {
    (void)ctx; (void)x; (void)y; (void)size;
    return 0;
}

/* context: a single blocked rect (simulates a pillar) */
typedef struct { int bx, by, bsize; } BlockRect;
static int blocked_rect(void* ctx, int x, int y, int size) {
    BlockRect* r = (BlockRect*)ctx;
    (void)size;
    return (x >= r->bx && x < r->bx + r->bsize &&
            y >= r->by && y < r->by + r->bsize);
}

/* --- regression: in-range NPC still moves (no range-stop in helper) --- */
static void test_in_range_still_steps(void) {
    printf("--- in-range NPC still steps (no helper-level range stop) ---\n");
    int x = 0, y = 0;
    /* target at (5, 0), attack_range=10 (ranger scenario), no blockers.
       pre-fix: dist=5 <= 10, early-return, no move.
       post-fix: step toward, moves to (1, 0). */
    int moved = encounter_npc_step_toward(&x, &y, 5, 0, 1, 1, 0, blocked_never, NULL);
    ASSERT_EQ("helper returns moved=1", moved, 1);
    ASSERT_EQ("x advanced by 1", x, 1);
    ASSERT_EQ("y unchanged", y, 0);
}

/* --- pillar between NPC and target: NPC gets stuck (matches OSRS) --- */
static void test_pillar_stuck(void) {
    printf("--- pillar directly in path: NPC stuck (matches OSRS greedy) ---\n");
    /* NPC at (0,0), target at (5,0), pillar covers tile (1,0).
       dx=1, dy=0 → helper only tries cardinal X, which is blocked → no move.
       greedy doesn't try y-detour because dy=0. matches OSRS: mobs get
       stuck on obstacles in their direct path and stand there. */
    BlockRect pillar = { 1, 0, 1 };
    int x = 0, y = 0;
    int moved = encounter_npc_step_toward(&x, &y, 5, 0, 1, 1, 0, blocked_rect, &pillar);
    ASSERT_EQ("moved=0 (stuck on pillar)", moved, 0);
    ASSERT_EQ("x unchanged", x, 0);
    ASSERT_EQ("y unchanged", y, 0);
}

/* --- pillar to the side, diagonal path clear: NPC moves diagonal --- */
static void test_pillar_diagonal_path(void) {
    printf("--- pillar partially blocking: diagonal path still works ---\n");
    /* NPC at (0,0), target at (5,5) (both dx and dy nonzero), pillar at (1,0).
       greedy tries diagonal (1,1) first — clear → move there. */
    BlockRect pillar = { 1, 0, 1 };
    int x = 0, y = 0;
    int moved = encounter_npc_step_toward(&x, &y, 5, 5, 1, 1, 0, blocked_rect, &pillar);
    ASSERT_EQ("moved=1 (took diagonal)", moved, 1);
    ASSERT_EQ("x=1", x, 1);
    ASSERT_EQ("y=1", y, 1);
}

/* --- melee adjacent: step helper tries, fails naturally (player tile blocked) --- */
static void test_melee_cardinal_contact_stops_without_target_tile_block(void) {
    printf("--- melee cardinal contact stops without target tile block ---\n");
    int x = 4, y = 5;
    int moved = encounter_npc_step_toward(&x, &y, 5, 5, 1, 1, 1, blocked_never, NULL);
    ASSERT_EQ("melee contact does not move", moved, 0);
    ASSERT_EQ("x unchanged", x, 4);
    ASSERT_EQ("y unchanged", y, 5);
}

static void test_melee_diagonal_contact_tries_x_only(void) {
    printf("--- melee diagonal contact tries x-only ---\n");
    int x = 4, y = 4;
    int moved = encounter_npc_step_toward(&x, &y, 5, 5, 1, 1, 1, blocked_never, NULL);
    ASSERT_EQ("diagonal contact moves", moved, 1);
    ASSERT_EQ("x advanced", x, 5);
    ASSERT_EQ("y stayed", y, 4);
}

static void test_melee_diagonal_contact_does_not_fall_back_to_y(void) {
    printf("--- melee diagonal contact does not fall back to y-only ---\n");
    BlockRect x_block = { 5, 4, 1 };
    int x = 4, y = 4;
    int moved = encounter_npc_step_toward(&x, &y, 5, 5, 1, 1, 1, blocked_rect, &x_block);
    ASSERT_EQ("x-only blocked means no move", moved, 0);
    ASSERT_EQ("x unchanged", x, 4);
    ASSERT_EQ("y unchanged", y, 4);
}

/* --- far NPC walks normally toward target --- */
static void test_far_npc_walks(void) {
    printf("--- far NPC walks greedy toward target ---\n");
    int x = 0, y = 0;
    int moved = encounter_npc_step_toward(&x, &y, 10, 7, 1, 1, 0, blocked_never, NULL);
    ASSERT_EQ("moved", moved, 1);
    ASSERT_EQ("diagonal step x+1", x, 1);
    ASSERT_EQ("diagonal step y+1", y, 1);
}

/* ======================================================================== */
/* entity_has_line_of_sight: regression for current-target LOS. inferno      */
/* movement needs size-aware mob->mob LOS (e.g. mager -> Zuk shield), and   */
/* melee helpers must stay cardinal-only.                                    */
/* ======================================================================== */

static void test_entity_los_to_multi_tile_target_in_range_clear(void) {
    printf("--- LOS to multi-tile target: in range, clear ray ---\n");
    int has = entity_has_line_of_sight(NULL, 0, 20, 36, 4, 23, 44, 5, 15);
    ASSERT_EQ("has_los to shield = 1", has, 1);
}

static void test_entity_los_to_multi_tile_target_blocked_by_pillar(void) {
    printf("--- LOS to multi-tile target: in range, pillar blocks ---\n");
    LOSBlocker pillar = { 15, 40, 3, LOS_FULL_MASK };
    int has = entity_has_line_of_sight(&pillar, 1, 10, 40, 4, 23, 40, 5, 15);
    ASSERT_EQ("has_los through pillar = 0", has, 0);
}

static void test_entity_los_to_multi_tile_target_out_of_range(void) {
    printf("--- LOS to multi-tile target: out of range ---\n");
    int has = entity_has_line_of_sight(NULL, 0, 0, 0, 4, 25, 25, 5, 15);
    ASSERT_EQ("out of range = 0", has, 0);
}

static void test_entity_melee_cardinal_only(void) {
    printf("--- melee LOS is cardinal only ---\n");
    ASSERT_EQ("cardinal contact allowed",
              entity_has_line_of_sight(NULL, 0, 6, 5, 1, 5, 5, 1, 1), 1);
    ASSERT_EQ("diagonal contact blocked",
              entity_has_line_of_sight(NULL, 0, 6, 6, 1, 5, 5, 1, 1), 0);
}

int main(void) {
    test_in_range_still_steps();
    test_pillar_stuck();
    test_pillar_diagonal_path();
    test_melee_cardinal_contact_stops_without_target_tile_block();
    test_melee_diagonal_contact_tries_x_only();
    test_melee_diagonal_contact_does_not_fall_back_to_y();
    test_far_npc_walks();

    test_entity_los_to_multi_tile_target_in_range_clear();
    test_entity_los_to_multi_tile_target_blocked_by_pillar();
    test_entity_los_to_multi_tile_target_out_of_range();
    test_entity_melee_cardinal_only();

    printf("\n=== results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
