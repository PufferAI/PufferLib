/**
 * @file test_damage.c
 * @brief tests for osrs_damage.h: pending hit queue and damage pipeline
 * (prayer reduction, vengeance, recoil, smite).
 *
 * BUILD:
 *   cd pufferlib-metal
 *   cc -std=c11 -O0 -g -I. -o test_damage \
 *       ocean/osrs/tests/test_damage.c -lm
 *   ./test_damage
 *
 * REFERENCE FILES:
 *   ocean/osrs/osrs_damage.h — shared damage pipeline
 *   ocean/osrs/osrs_combat.h — osrs_prayer_reduce_damage
 *   ocean/osrs/osrs_pvp_combat.h:570-691 — original PvP apply_damage
 *   ocean/osrs/encounters/encounter_zulrah.h:651-675 — original zulrah recoil
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ocean/osrs/osrs_damage.h"


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


static void test_prayer_reduction(void) {
    printf("--- prayer reduction ---\n");

    /* PvE: correct prayer blocks 100% */
    DamageResult r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_MELEE, PRAYER_PROTECT_MELEE,
        /*is_pvp=*/0, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("pve melee prayer block", r.final_damage, 0);
    ASSERT_INT_EQ("pve melee prayer_blocked flag", r.prayer_blocked, 1);

    /* PvP: correct prayer reduces by 40% → floor(30 * 0.6) = 18 */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_MELEE, PRAYER_PROTECT_MELEE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("pvp melee prayer 40% reduction", r.final_damage, 18);
    ASSERT_INT_EQ("pvp melee prayer_blocked flag", r.prayer_blocked, 1);

    /* PvE: wrong prayer passthrough */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_MELEE, PRAYER_PROTECT_MAGIC,
        /*is_pvp=*/0, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("pve wrong prayer passthrough", r.final_damage, 30);
    ASSERT_INT_EQ("pve wrong prayer_blocked flag", r.prayer_blocked, 0);

    /* PvP: wrong prayer passthrough */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_RANGED, PRAYER_PROTECT_MELEE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("pvp wrong prayer passthrough", r.final_damage, 30);

    /* PvE: no prayer */
    r = osrs_apply_damage_pipeline(
        25, ATTACK_STYLE_MAGIC, PRAYER_NONE,
        /*is_pvp=*/0, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("pve no prayer passthrough", r.final_damage, 25);
    ASSERT_INT_EQ("pve no prayer_blocked flag", r.prayer_blocked, 0);

    /* PvP: ranged prayer block (40% reduction) → floor(50 * 0.6) = 30 */
    r = osrs_apply_damage_pipeline(
        50, ATTACK_STYLE_RANGED, PRAYER_PROTECT_RANGED,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("pvp ranged prayer reduction", r.final_damage, 30);

    /* PvE: magic prayer blocks 100% */
    r = osrs_apply_damage_pipeline(
        40, ATTACK_STYLE_MAGIC, PRAYER_PROTECT_MAGIC,
        /*is_pvp=*/0, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("pve magic prayer block", r.final_damage, 0);
}


static void test_vengeance(void) {
    printf("--- vengeance ---\n");

    /* veng active: floor(20 * 0.75) = 15 */
    DamageResult r = osrs_apply_damage_pipeline(
        20, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("veng reflect 20 -> 15", r.veng_damage, 15);
    ASSERT_INT_EQ("veng final_damage unchanged", r.final_damage, 20);

    /* veng on 0 damage: no reflect */
    r = osrs_apply_damage_pipeline(
        0, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("veng 0 damage no reflect", r.veng_damage, 0);

    /* veng inactive: no reflect */
    r = osrs_apply_damage_pipeline(
        20, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("veng inactive no reflect", r.veng_damage, 0);

    /* veng with prayer reduction: floor(18 * 0.75) = 13
       (30 raw, 40% reduction = 18, then veng on 18) */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_MELEE, PRAYER_PROTECT_MELEE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("veng after prayer reduction", r.veng_damage, 13);
    ASSERT_INT_EQ("veng after prayer final_damage", r.final_damage, 18);

    /* veng on 1 damage: floor(1 * 0.75) = 0 */
    r = osrs_apply_damage_pipeline(
        1, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("veng 1 damage reflect 0", r.veng_damage, 0);

    /* veng on PvE full prayer block: 0 damage after prayer, no reflect */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_MELEE, PRAYER_PROTECT_MELEE,
        /*is_pvp=*/0, /*veng=*/1, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("veng pve prayer block no reflect", r.veng_damage, 0);
}


static void test_recoil(void) {
    printf("--- recoil ---\n");

    /* recoil: floor(20 / 10) + 1 = 3 */
    DamageResult r = osrs_apply_damage_pipeline(
        20, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*recoil ring=*/0, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("recoil 20 -> 3", r.recoil_damage, 3);

    /* recoil: floor(1 / 10) + 1 = 1 */
    r = osrs_apply_damage_pipeline(
        1, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("recoil 1 -> 1", r.recoil_damage, 1);

    /* recoil: floor(10 / 10) + 1 = 2 */
    r = osrs_apply_damage_pipeline(
        10, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/0, /*veng=*/0, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("recoil 10 -> 2", r.recoil_damage, 2);

    /* no recoil ring: 0 */
    r = osrs_apply_damage_pipeline(
        20, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("no recoil ring 0", r.recoil_damage, 0);

    /* recoil on 0 damage: 0 */
    r = osrs_apply_damage_pipeline(
        0, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("recoil 0 damage -> 0", r.recoil_damage, 0);

    /* recoil after PvE full prayer block: 0 */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_RANGED, PRAYER_PROTECT_RANGED,
        /*is_pvp=*/0, /*veng=*/0, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("recoil pve prayer block -> 0", r.recoil_damage, 0);

    /* recoil: floor(99 / 10) + 1 = 10 */
    r = osrs_apply_damage_pipeline(
        99, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/0, /*veng=*/0, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("recoil 99 -> 10", r.recoil_damage, 10);
}


static void test_smite(void) {
    printf("--- smite ---\n");

    /* smite: floor(20 / 4) = 5 */
    DamageResult r = osrs_apply_damage_pipeline(
        20, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/1);
    ASSERT_INT_EQ("smite 20 -> 5", r.smite_drain, 5);

    /* smite: floor(1 / 4) = 0 */
    r = osrs_apply_damage_pipeline(
        1, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/1);
    ASSERT_INT_EQ("smite 1 -> 0", r.smite_drain, 0);

    /* smite: floor(7 / 4) = 1 */
    r = osrs_apply_damage_pipeline(
        7, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/1);
    ASSERT_INT_EQ("smite 7 -> 1", r.smite_drain, 1);

    /* smite inactive: 0 */
    r = osrs_apply_damage_pipeline(
        20, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/0);
    ASSERT_INT_EQ("smite inactive -> 0", r.smite_drain, 0);

    /* smite on 0 damage: 0 */
    r = osrs_apply_damage_pipeline(
        0, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/1);
    ASSERT_INT_EQ("smite 0 damage -> 0", r.smite_drain, 0);

    /* smite after PvP prayer reduction: floor(18 / 4) = 4
       (30 raw, 40% reduction = 18) */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_MELEE, PRAYER_PROTECT_MELEE,
        /*is_pvp=*/1, /*veng=*/0, /*recoil=*/0, /*smite=*/1);
    ASSERT_INT_EQ("smite after pvp prayer", r.smite_drain, 4);
}


static void test_full_pipeline_pvp(void) {
    printf("--- full pipeline PvP ---\n");

    /* 40 raw, melee, protect melee (PvP 40% reduction)
       final_damage = floor(40 * 0.6) = 24
       veng = floor(24 * 0.75) = 18
       recoil = floor(24 / 10) + 1 = 3
       smite = floor(24 / 4) = 6 */
    DamageResult r = osrs_apply_damage_pipeline(
        40, ATTACK_STYLE_MELEE, PRAYER_PROTECT_MELEE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/1, /*smite=*/1);
    ASSERT_INT_EQ("full pvp final_damage", r.final_damage, 24);
    ASSERT_INT_EQ("full pvp veng", r.veng_damage, 18);
    ASSERT_INT_EQ("full pvp recoil", r.recoil_damage, 3);
    ASSERT_INT_EQ("full pvp smite", r.smite_drain, 6);
    ASSERT_INT_EQ("full pvp prayer_blocked", r.prayer_blocked, 1);

    /* no prayer, 50 raw melee hit
       final_damage = 50
       veng = floor(50 * 0.75) = 37
       recoil = floor(50 / 10) + 1 = 6
       smite = floor(50 / 4) = 12 */
    r = osrs_apply_damage_pipeline(
        50, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/1, /*smite=*/1);
    ASSERT_INT_EQ("full pvp no prayer final", r.final_damage, 50);
    ASSERT_INT_EQ("full pvp no prayer veng", r.veng_damage, 37);
    ASSERT_INT_EQ("full pvp no prayer recoil", r.recoil_damage, 6);
    ASSERT_INT_EQ("full pvp no prayer smite", r.smite_drain, 12);
    ASSERT_INT_EQ("full pvp no prayer_blocked", r.prayer_blocked, 0);
}


static void test_full_pipeline_pve(void) {
    printf("--- full pipeline PvE ---\n");

    /* 30 raw, ranged, protect ranged (PvE = 100% block)
       final_damage = 0 → all secondary effects = 0 */
    DamageResult r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_RANGED, PRAYER_PROTECT_RANGED,
        /*is_pvp=*/0, /*veng=*/1, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("pve full block final", r.final_damage, 0);
    ASSERT_INT_EQ("pve full block veng", r.veng_damage, 0);
    ASSERT_INT_EQ("pve full block recoil", r.recoil_damage, 0);
    ASSERT_INT_EQ("pve full block smite", r.smite_drain, 0);
    ASSERT_INT_EQ("pve full block prayer_blocked", r.prayer_blocked, 1);

    /* 30 raw, magic, no prayer (PvE, veng + recoil active)
       final_damage = 30
       veng = floor(30 * 0.75) = 22
       recoil = floor(30 / 10) + 1 = 4 */
    r = osrs_apply_damage_pipeline(
        30, ATTACK_STYLE_MAGIC, PRAYER_NONE,
        /*is_pvp=*/0, /*veng=*/1, /*recoil=*/1, /*smite=*/0);
    ASSERT_INT_EQ("pve no prayer final", r.final_damage, 30);
    ASSERT_INT_EQ("pve no prayer veng", r.veng_damage, 22);
    ASSERT_INT_EQ("pve no prayer recoil", r.recoil_damage, 4);
    ASSERT_INT_EQ("pve no prayer smite", r.smite_drain, 0);
}


static void test_edge_cases(void) {
    printf("--- edge cases ---\n");

    /* 0 damage: all effects should be 0 */
    DamageResult r = osrs_apply_damage_pipeline(
        0, ATTACK_STYLE_MELEE, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/1, /*smite=*/1);
    ASSERT_INT_EQ("zero damage final", r.final_damage, 0);
    ASSERT_INT_EQ("zero damage veng", r.veng_damage, 0);
    ASSERT_INT_EQ("zero damage recoil", r.recoil_damage, 0);
    ASSERT_INT_EQ("zero damage smite", r.smite_drain, 0);

    /* 1 damage: recoil = floor(1/10)+1 = 1, veng = floor(0.75) = 0, smite = 0 */
    r = osrs_apply_damage_pipeline(
        1, ATTACK_STYLE_RANGED, PRAYER_NONE,
        /*is_pvp=*/0, /*veng=*/1, /*recoil=*/1, /*smite=*/1);
    ASSERT_INT_EQ("1 damage final", r.final_damage, 1);
    ASSERT_INT_EQ("1 damage veng", r.veng_damage, 0);
    ASSERT_INT_EQ("1 damage recoil", r.recoil_damage, 1);
    ASSERT_INT_EQ("1 damage smite", r.smite_drain, 0);

    /* large hit: 99 damage, all active
       veng = floor(99 * 0.75) = 74
       recoil = floor(99 / 10) + 1 = 10
       smite = floor(99 / 4) = 24 */
    r = osrs_apply_damage_pipeline(
        99, ATTACK_STYLE_MAGIC, PRAYER_NONE,
        /*is_pvp=*/1, /*veng=*/1, /*recoil=*/1, /*smite=*/1);
    ASSERT_INT_EQ("99 damage veng", r.veng_damage, 74);
    ASSERT_INT_EQ("99 damage recoil", r.recoil_damage, 10);
    ASSERT_INT_EQ("99 damage smite", r.smite_drain, 24);
}


static void test_has_recoil_ring(void) {
    printf("--- has_recoil_ring ---\n");

    uint8_t equipped[16];
    memset(equipped, ITEM_NONE, sizeof(equipped));

    /* no ring */
    ASSERT_INT_EQ("no ring", osrs_has_recoil_ring(equipped), 0);

    /* ring of recoil */
    equipped[GEAR_SLOT_RING] = ITEM_RING_OF_RECOIL;
    ASSERT_INT_EQ("ring of recoil", osrs_has_recoil_ring(equipped), 1);

    /* ring of suffering (i) */
    equipped[GEAR_SLOT_RING] = ITEM_RING_OF_SUFFERING_RI;
    ASSERT_INT_EQ("ring of suffering (i)", osrs_has_recoil_ring(equipped), 1);

    /* some other ring */
    equipped[GEAR_SLOT_RING] = 42;
    ASSERT_INT_EQ("other ring", osrs_has_recoil_ring(equipped), 0);
}


int main(void) {
    printf("=== osrs_damage.h test suite ===\n\n");

    test_prayer_reduction();
    test_vengeance();
    test_recoil();
    test_smite();
    test_full_pipeline_pvp();
    test_full_pipeline_pve();
    test_edge_cases();
    test_has_recoil_ring();

    printf("\n=== results: %d passed, %d failed, %d total ===\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}
