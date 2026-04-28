/**
 * @file test_bolt_procs.c
 * @brief tests for enchanted crossbow bolt proc system (diamond, opal, ruby).
 *
 * validates proc chances, effect formulas, ZCB enhanced procs, miss behavior,
 * and edge cases against .refs/osrs-dps-calc/src/lib/dists/bolts.ts.
 *
 * BUILD:
 *   cd pufferlib-metal
 *   cc -std=c11 -O0 -g -I. -o test_bolt_procs \
 *       ocean/osrs/tests/test_bolt_procs.c -lm
 *   ./test_bolt_procs
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ocean/osrs/osrs_bolt_procs.h"


static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_INT_EQ(label, actual, expected) do { \
    tests_run++; \
    if ((actual) == (expected)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s — got %d, expected %d\n", (label), (actual), (expected)); \
    } \
} while (0)

#define ASSERT_FLOAT_RANGE(label, actual, lo, hi) do { \
    tests_run++; \
    float _a = (actual); \
    if (_a >= (lo) && _a <= (hi)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s — got %.4f, expected [%.4f, %.4f]\n", \
               (label), _a, (float)(lo), (float)(hi)); \
    } \
} while (0)


static void test_diamond_proc_chance(void) {
    printf("test_diamond_proc_chance\n");
    uint32_t rng = 12345;
    int procs = 0;
    int trials = 10000;
    for (int i = 0; i < trials; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_DIAMOND_BOLTS_E, 30, 1, 50, 99, 200, 0, &rng);
        if (r.proc_triggered) procs++;
    }
    float rate = (float)procs / (float)trials;
    ASSERT_FLOAT_RANGE("diamond proc rate ~11%", rate, 0.08f, 0.14f);
}


static void test_diamond_effect_max(void) {
    printf("test_diamond_effect_max\n");
    /* max_hit=50: normal effectMax = 50*115/100 = 57
       ZCB effectMax = 50*126/100 = 63 */
    int max_hit = 50;
    int normal_max = max_hit * 115 / 100;  /* 57 */
    int zcb_max = max_hit * 126 / 100;     /* 63 */
    ASSERT_INT_EQ("diamond normal effectMax", normal_max, 57);
    ASSERT_INT_EQ("diamond ZCB effectMax", zcb_max, 63);

    /* verify no proc damage exceeds effectMax over many trials */
    uint32_t rng = 99999;
    int max_seen_normal = 0;
    int max_seen_zcb = 0;
    for (int i = 0; i < 50000; i++) {
        uint32_t rng_copy = rng;
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_DIAMOND_BOLTS_E, 30, 1, max_hit, 99, 200, 0, &rng);
        if (r.proc_triggered && r.modified_damage > max_seen_normal)
            max_seen_normal = r.modified_damage;

        BoltProcResult rz = osrs_resolve_bolt_proc(
            ITEM_DIAMOND_BOLTS_E, 30, 1, max_hit, 99, 200, 1, &rng_copy);
        if (rz.proc_triggered && rz.modified_damage > max_seen_zcb)
            max_seen_zcb = rz.modified_damage;
    }
    /* max seen should be <= effectMax */
    ASSERT_INT_EQ("diamond normal max_seen <= 57", max_seen_normal <= 57, 1);
    ASSERT_INT_EQ("diamond ZCB max_seen <= 63", max_seen_zcb <= 63, 1);
}


static void test_diamond_zcb_spec(void) {
    printf("test_diamond_zcb_spec\n");
    uint32_t rng = 42;
    int procs = 0;
    int trials = 100;
    for (int i = 0; i < trials; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_DIAMOND_BOLTS_E, 30, 1, 50, 99, 200, 1, &rng);
        if (r.proc_triggered) procs++;
    }
    ASSERT_INT_EQ("diamond ZCB spec guaranteed proc", procs, trials);
}


static void test_diamond_miss(void) {
    printf("test_diamond_miss\n");
    uint32_t rng = 777;
    int procs = 0;
    for (int i = 0; i < 1000; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_DIAMOND_BOLTS_E, 0, 0, 50, 99, 200, 0, &rng);
        if (r.proc_triggered) procs++;
    }
    ASSERT_INT_EQ("diamond miss = no proc", procs, 0);

    /* ZCB spec on miss should still proc */
    rng = 888;
    int zcb_procs = 0;
    for (int i = 0; i < 100; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_DIAMOND_BOLTS_E, 0, 0, 50, 99, 200, 1, &rng);
        if (r.proc_triggered) zcb_procs++;
    }
    ASSERT_INT_EQ("diamond ZCB spec procs on miss", zcb_procs, 100);
}


static void test_opal_bonus_damage(void) {
    printf("test_opal_bonus_damage\n");
    /* force proc with ZCB spec to get deterministic bonus */
    uint32_t rng = 111;
    int base = 25;
    BoltProcResult r_normal = osrs_resolve_bolt_proc(
        ITEM_OPAL_DRAGON_BOLTS, base, 1, 50, 99, 200, 1, &rng);
    /* ZCB spec uses divisor 9: floor(99/9) = 11 */
    ASSERT_INT_EQ("opal ZCB bonus", r_normal.modified_damage, base + 11);

    /* to test normal divisor: run many trials, check proc damage */
    rng = 222;
    int found_normal_bonus = 0;
    for (int i = 0; i < 50000; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_OPAL_DRAGON_BOLTS, base, 1, 50, 99, 200, 0, &rng);
        if (r.proc_triggered) {
            /* normal bonus = floor(99/10) = 9 */
            ASSERT_INT_EQ("opal normal bonus", r.modified_damage, base + 9);
            found_normal_bonus = 1;
            break;
        }
    }
    ASSERT_INT_EQ("opal normal proc found", found_normal_bonus, 1);
}


static void test_opal_works_on_miss(void) {
    printf("test_opal_works_on_miss\n");
    uint32_t rng = 333;
    int procs = 0;
    for (int i = 0; i < 20000; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_OPAL_DRAGON_BOLTS, 0, 0, 50, 99, 200, 0, &rng);
        if (r.proc_triggered) procs++;
    }
    /* should see some procs even on misses */
    ASSERT_INT_EQ("opal procs on miss", procs > 0, 1);
}


static void test_opal_proc_chance(void) {
    printf("test_opal_proc_chance\n");
    uint32_t rng = 444;
    int procs = 0;
    int trials = 20000;
    for (int i = 0; i < trials; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_OPAL_DRAGON_BOLTS, 25, 1, 50, 99, 200, 0, &rng);
        if (r.proc_triggered) procs++;
    }
    float rate = (float)procs / (float)trials;
    ASSERT_FLOAT_RANGE("opal proc rate ~5.5%", rate, 0.04f, 0.07f);
}


static void test_ruby_hp_based_damage(void) {
    printf("test_ruby_hp_based_damage\n");
    uint32_t rng = 555;
    /* force proc with ZCB spec */
    BoltProcResult r_zcb = osrs_resolve_bolt_proc(
        ITEM_RUBY_DRAGON_BOLTS_E, 30, 1, 50, 99, 500, 1, &rng);
    /* floor(500 * 22/100) = 110, cap 110 → 110 */
    ASSERT_INT_EQ("ruby ZCB 500hp", r_zcb.modified_damage, 110);

    /* normal: floor(500 * 20/100) = 100, cap 100 → 100 */
    rng = 666;
    int found = 0;
    for (int i = 0; i < 50000; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_RUBY_DRAGON_BOLTS_E, 30, 1, 50, 99, 500, 0, &rng);
        if (r.proc_triggered) {
            ASSERT_INT_EQ("ruby normal 500hp", r.modified_damage, 100);
            found = 1;
            break;
        }
    }
    ASSERT_INT_EQ("ruby normal proc found", found, 1);
}


static void test_ruby_cap(void) {
    printf("test_ruby_cap\n");
    uint32_t rng = 777;
    /* ZCB: floor(1000*22/100) = 220, capped to 110 */
    BoltProcResult r_zcb = osrs_resolve_bolt_proc(
        ITEM_RUBY_DRAGON_BOLTS_E, 30, 1, 50, 99, 1000, 1, &rng);
    ASSERT_INT_EQ("ruby ZCB 1000hp capped", r_zcb.modified_damage, 110);

    /* normal: floor(1000*20/100) = 200, capped to 100 */
    rng = 888;
    int found = 0;
    for (int i = 0; i < 50000; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_RUBY_DRAGON_BOLTS_E, 30, 1, 50, 99, 1000, 0, &rng);
        if (r.proc_triggered) {
            ASSERT_INT_EQ("ruby normal 1000hp capped", r.modified_damage, 100);
            found = 1;
            break;
        }
    }
    ASSERT_INT_EQ("ruby normal cap proc found", found, 1);
}


static void test_ruby_miss(void) {
    printf("test_ruby_miss\n");
    uint32_t rng = 999;
    int procs = 0;
    for (int i = 0; i < 5000; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_RUBY_DRAGON_BOLTS_E, 0, 0, 50, 99, 500, 0, &rng);
        if (r.proc_triggered) procs++;
    }
    ASSERT_INT_EQ("ruby miss = no proc", procs, 0);
}


static void test_ruby_zcb_spec(void) {
    printf("test_ruby_zcb_spec\n");
    uint32_t rng = 1111;
    int procs = 0;
    int trials = 100;
    for (int i = 0; i < trials; i++) {
        BoltProcResult r = osrs_resolve_bolt_proc(
            ITEM_RUBY_DRAGON_BOLTS_E, 30, 1, 50, 99, 300, 1, &rng);
        if (r.proc_triggered) procs++;
        /* floor(300*22/100) = 66, under cap 110 */
        ASSERT_INT_EQ("ruby ZCB 300hp dmg", r.modified_damage, 66);
    }
    ASSERT_INT_EQ("ruby ZCB spec guaranteed", procs, trials);
}


static void test_unknown_bolt(void) {
    printf("test_unknown_bolt\n");
    uint32_t rng = 2222;
    BoltProcResult r = osrs_resolve_bolt_proc(
        ITEM_DRAGON_ARROWS, 30, 1, 50, 99, 200, 0, &rng);
    ASSERT_INT_EQ("unknown bolt no proc", r.proc_triggered, 0);
    ASSERT_INT_EQ("unknown bolt damage unchanged", r.modified_damage, 30);

    /* even with ZCB spec flag, unknown bolt should not proc */
    BoltProcResult r2 = osrs_resolve_bolt_proc(
        ITEM_DRAGON_ARROWS, 30, 1, 50, 99, 200, 1, &rng);
    ASSERT_INT_EQ("unknown bolt + ZCB no proc", r2.proc_triggered, 0);
}


static void test_edge_cases(void) {
    printf("test_edge_cases\n");
    uint32_t rng = 3333;

    /* diamond with max_hit=0: effectMax = 0*115/100 = 0, re-roll from [0,0] = 0 */
    BoltProcResult rd = osrs_resolve_bolt_proc(
        ITEM_DIAMOND_BOLTS_E, 0, 1, 0, 1, 1, 1, &rng);
    ASSERT_INT_EQ("diamond max_hit=0 proc", rd.proc_triggered, 1);
    ASSERT_INT_EQ("diamond max_hit=0 dmg", rd.modified_damage, 0);

    /* opal with ranged_level=1: bonus = floor(1/9) = 0 (ZCB) */
    BoltProcResult ro = osrs_resolve_bolt_proc(
        ITEM_OPAL_DRAGON_BOLTS, 5, 1, 0, 1, 1, 1, &rng);
    ASSERT_INT_EQ("opal rlvl=1 proc", ro.proc_triggered, 1);
    ASSERT_INT_EQ("opal rlvl=1 dmg", ro.modified_damage, 5);  /* 5 + 0 */

    /* ruby with target_hp=1: floor(1*22/100) = 0 */
    BoltProcResult rr = osrs_resolve_bolt_proc(
        ITEM_RUBY_DRAGON_BOLTS_E, 30, 1, 50, 99, 1, 1, &rng);
    ASSERT_INT_EQ("ruby hp=1 proc", rr.proc_triggered, 1);
    ASSERT_INT_EQ("ruby hp=1 dmg", rr.modified_damage, 0);

    /* diamond dragon bolts (e) should also work (same case as diamond bolts e) */
    BoltProcResult rdd = osrs_resolve_bolt_proc(
        ITEM_DIAMOND_DRAGON_BOLTS_E, 30, 1, 50, 99, 200, 1, &rng);
    ASSERT_INT_EQ("diamond dragon bolts proc", rdd.proc_triggered, 1);
}


int main(void) {
    printf("=== bolt proc tests ===\n\n");

    test_diamond_proc_chance();
    test_diamond_effect_max();
    test_diamond_zcb_spec();
    test_diamond_miss();
    test_opal_bonus_damage();
    test_opal_works_on_miss();
    test_opal_proc_chance();
    test_ruby_hp_based_damage();
    test_ruby_cap();
    test_ruby_miss();
    test_ruby_zcb_spec();
    test_unknown_bolt();
    test_edge_cases();

    printf("\n%d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
        printf(", %d FAILED", tests_failed);
    printf("\n");

    return tests_failed > 0 ? 1 : 0;
}
