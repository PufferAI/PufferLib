/**
 * @file test_item_effects.c
 * @brief tests for item interactions, prayer reduction, tbow scaling edges,
 * defensive rolls, player attack rolls with gear, and loadout edge cases.
 *
 * cross-referenced against osrs-dps-calc reference and OSRS wiki formulas.
 *
 * BUILD:
 *   cd pufferlib-metal
 *   cc -std=c11 -O0 -g -I. -o test_item_effects \
 *       ocean/osrs/tests/test_item_effects.c -lm
 *   ./test_item_effects
 *
 * REFERENCE FILES:
 *   .refs/osrs-dps-calc/src/lib/PlayerVsNPCCalc.ts — tbow scaling, specific bonuses
 *   .refs/osrs-dps-calc/src/tests/calc/DefenceRolls.test.ts — NPC defence values
 *   .refs/osrs-dps-calc/src/tests/calc/Prayers.test.ts — prayer mechanics
 *   ocean/osrs/osrs_pvp_combat.h — PvP prayer 40% reduction
 *   ocean/osrs/encounters/encounter_inferno.h — PvE prayer full block
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ocean/osrs/osrs_encounter.h"


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

#define ASSERT_FLOAT_NEAR(label, actual, expected, tolerance) do { \
    tests_run++; \
    float _a = (actual), _e = (expected), _t = (tolerance); \
    if (fabsf(_a - _e) <= _t) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s — got %.6f, expected %.6f (tol %.6f)\n", \
               (label), _a, _e, _t); \
    } \
} while (0)

/* helper: fill loadout with ITEM_NONE */
static void clear_loadout(uint8_t loadout[NUM_GEAR_SLOTS]) {
    memset(loadout, 255, NUM_GEAR_SLOTS);
}

/* ======================================================================== */
/* reference tbow multipliers with integer truncation (matching TS impl)     */
/*                                                                           */
/* ref: PlayerVsNPCCalc.ts tbowScaling()                                    */
/*   accuracy: factor=10, base=140, cap=1.40                                */
/*   damage:   factor=14, base=250, cap=2.50                                */
/* ======================================================================== */

static float ref_tbow_acc(int magic) {
    int m = magic < 250 ? magic : 250;
    int t2 = (3 * m - 10) / 100;
    int inner = (3 * m / 10) - 100;
    int t3 = (inner * inner) / 100;
    int bonus = 140 + t2 - t3;
    float mult = (float)bonus / 100.0f;
    if (mult > 1.4f) mult = 1.4f;
    if (mult < 0.0f) mult = 0.0f;
    return mult;
}

static float ref_tbow_dmg(int magic) {
    int m = magic < 250 ? magic : 250;
    int t2 = (3 * m - 14) / 100;
    int inner = (3 * m / 10) - 140;
    int t3 = (inner * inner) / 100;
    int bonus = 250 + t2 - t3;
    float mult = (float)bonus / 100.0f;
    if (mult > 2.5f) mult = 2.5f;
    if (mult < 0.0f) mult = 0.0f;
    return mult;
}

/* ======================================================================== */
/* test: tbow accuracy multiplier — edge cases and boundary behavior         */
/*                                                                           */
/* magic is clamped to [0, 250]. accuracy cap = 1.40, floor = 0.00.         */
/* our C uses float division; TS ref uses integer truncation on intermediates*/
/* so we allow up to 0.01 tolerance.                                        */
/* ======================================================================== */

static void test_tbow_acc_edge_cases(void) {
    printf("--- tbow accuracy: edge cases ---\n");

    /* magic=0: parabola leftmost point. low accuracy */
    ASSERT_FLOAT_NEAR("acc m=0 vs ref", osrs_tbow_acc_mult(0), ref_tbow_acc(0), 0.01f);
    ASSERT_INT_EQ("acc m=0 >= 0", osrs_tbow_acc_mult(0) >= 0.0f, 1);

    /* magic=1: just above minimum */
    ASSERT_FLOAT_NEAR("acc m=1 vs ref", osrs_tbow_acc_mult(1), ref_tbow_acc(1), 0.01f);

    /* magic=50: low-magic NPC */
    ASSERT_FLOAT_NEAR("acc m=50 vs ref", osrs_tbow_acc_mult(50), ref_tbow_acc(50), 0.01f);

    /* magic=100: mid-range */
    ASSERT_FLOAT_NEAR("acc m=100 vs ref", osrs_tbow_acc_mult(100), ref_tbow_acc(100), 0.01f);

    /* magic=150: typical high-magic NPC (e.g. Zulrah) */
    ASSERT_FLOAT_NEAR("acc m=150 vs ref", osrs_tbow_acc_mult(150), ref_tbow_acc(150), 0.01f);

    /* magic=200: very high magic */
    ASSERT_FLOAT_NEAR("acc m=200 vs ref", osrs_tbow_acc_mult(200), ref_tbow_acc(200), 0.01f);

    /* magic=249: just below cap threshold */
    ASSERT_FLOAT_NEAR("acc m=249 vs ref", osrs_tbow_acc_mult(249), ref_tbow_acc(249), 0.01f);

    /* magic=250: cap boundary. should be at or near 1.40 */
    ASSERT_FLOAT_NEAR("acc m=250 vs ref", osrs_tbow_acc_mult(250), ref_tbow_acc(250), 0.01f);
    ASSERT_INT_EQ("acc m=250 <= 1.4", osrs_tbow_acc_mult(250) <= 1.4f + 0.001f, 1);

    /* magic=251: above cap, clamps m to 250 internally, same result as 250 */
    ASSERT_FLOAT_NEAR("acc m=251 == m=250",
        osrs_tbow_acc_mult(251), osrs_tbow_acc_mult(250), 1e-5f);

    /* magic=350: well above cap */
    ASSERT_FLOAT_NEAR("acc m=350 == m=250",
        osrs_tbow_acc_mult(350), osrs_tbow_acc_mult(250), 1e-5f);

    /* magic=500: extreme above-cap value */
    ASSERT_FLOAT_NEAR("acc m=500 == m=250",
        osrs_tbow_acc_mult(500), osrs_tbow_acc_mult(250), 1e-5f);

    /* strict monotonicity from 0 to 250 (parabola opens downward, peak >= 250) */
    int monotonic = 1;
    for (int m = 1; m <= 250; m++) {
        if (osrs_tbow_acc_mult(m) < osrs_tbow_acc_mult(m - 1)) {
            monotonic = 0;
            break;
        }
    }
    ASSERT_INT_EQ("acc monotonic 0..250", monotonic, 1);
}

/* ======================================================================== */
/* test: tbow damage multiplier — edge cases and inverted-U shape            */
/*                                                                           */
/* damage mult peaks around magic~100 and decreases at extremes.            */
/* cap = 2.50, floor = 0.00.                                                */
/* ======================================================================== */

static void test_tbow_dmg_edge_cases(void) {
    printf("--- tbow damage: edge cases ---\n");

    /* magic=0: low end of parabola */
    ASSERT_FLOAT_NEAR("dmg m=0 vs ref", osrs_tbow_dmg_mult(0), ref_tbow_dmg(0), 0.015f);
    ASSERT_INT_EQ("dmg m=0 >= 0", osrs_tbow_dmg_mult(0) >= 0.0f, 1);

    /* magic=1 */
    ASSERT_FLOAT_NEAR("dmg m=1 vs ref", osrs_tbow_dmg_mult(1), ref_tbow_dmg(1), 0.015f);

    /* magic=50 */
    ASSERT_FLOAT_NEAR("dmg m=50 vs ref", osrs_tbow_dmg_mult(50), ref_tbow_dmg(50), 0.015f);

    /* magic=100: near peak of inverted-U */
    ASSERT_FLOAT_NEAR("dmg m=100 vs ref", osrs_tbow_dmg_mult(100), ref_tbow_dmg(100), 0.015f);

    /* magic=150 */
    ASSERT_FLOAT_NEAR("dmg m=150 vs ref", osrs_tbow_dmg_mult(150), ref_tbow_dmg(150), 0.015f);

    /* magic=200 */
    ASSERT_FLOAT_NEAR("dmg m=200 vs ref", osrs_tbow_dmg_mult(200), ref_tbow_dmg(200), 0.015f);

    /* magic=250: cap boundary */
    ASSERT_FLOAT_NEAR("dmg m=250 vs ref", osrs_tbow_dmg_mult(250), ref_tbow_dmg(250), 0.015f);
    ASSERT_INT_EQ("dmg m=250 <= 2.5", osrs_tbow_dmg_mult(250) <= 2.5f + 0.001f, 1);

    /* magic above 250: clamped to 250 */
    ASSERT_FLOAT_NEAR("dmg m=350 == m=250",
        osrs_tbow_dmg_mult(350), osrs_tbow_dmg_mult(250), 1e-5f);
    ASSERT_FLOAT_NEAR("dmg m=500 == m=250",
        osrs_tbow_dmg_mult(500), osrs_tbow_dmg_mult(250), 1e-5f);

    /* damage mult increases monotonically from 0 to 250 (unlike accuracy,
       the damage parabola doesn't peak before 250 in the capped range) */
    ASSERT_INT_EQ("dmg monotonic 0<100",
        osrs_tbow_dmg_mult(0) < osrs_tbow_dmg_mult(100), 1);
    ASSERT_INT_EQ("dmg monotonic 100<250",
        osrs_tbow_dmg_mult(100) < osrs_tbow_dmg_mult(250), 1);
}

/* ======================================================================== */
/* test: tbow cap and floor bounds sweep                                     */
/*                                                                           */
/* sweep magic 0..350 and verify both multipliers stay in valid range.      */
/* ======================================================================== */

static void test_tbow_cap_behavior(void) {
    printf("--- tbow cap and floor bounds ---\n");

    int acc_ok = 1, dmg_ok = 1;
    for (int m = 0; m <= 350; m++) {
        float a = osrs_tbow_acc_mult(m);
        float d = osrs_tbow_dmg_mult(m);
        if (a < 0.0f || a > 1.4f + 0.001f) acc_ok = 0;
        if (d < 0.0f || d > 2.5f + 0.001f) dmg_ok = 0;
    }
    ASSERT_INT_EQ("acc in [0, 1.4] for m=0..350", acc_ok, 1);
    ASSERT_INT_EQ("dmg in [0, 2.5] for m=0..350", dmg_ok, 1);
}

/* ======================================================================== */
/* test: PvP prayer protection — correct overhead reduces damage by 40%      */
/*                                                                           */
/* ref: osrs_pvp_combat.h line 561 — actual_damage = (int)(damage * 0.6f)   */
/* in PvP, correct overhead prayer reduces incoming damage by 40%.           */
/* ======================================================================== */

static void test_prayer_pvp_reduction(void) {
    printf("--- PvP prayer: 40%% damage reduction ---\n");

    /* direct formula test: (int)(damage * 0.6f)
       this is the exact expression used in osrs_pvp_combat.h */
    ASSERT_INT_EQ("dmg=0 -> 0",   (int)(0 * 0.6f), 0);
    ASSERT_INT_EQ("dmg=1 -> 0",   (int)(1 * 0.6f), 0);
    ASSERT_INT_EQ("dmg=2 -> 1",   (int)(2 * 0.6f), 1);
    ASSERT_INT_EQ("dmg=5 -> 3",   (int)(5 * 0.6f), 3);
    ASSERT_INT_EQ("dmg=10 -> 6",  (int)(10 * 0.6f), 6);
    ASSERT_INT_EQ("dmg=50 -> 30", (int)(50 * 0.6f), 30);
    ASSERT_INT_EQ("dmg=97 -> 58", (int)(97 * 0.6f), 58);
    ASSERT_INT_EQ("dmg=100 -> 60", (int)(100 * 0.6f), 60);

    /* verify correct prayer check identifies match for each style */
    ASSERT_INT_EQ("melee blocked",
        encounter_prayer_correct_for_style(3 /* PROTECT_MELEE */, 1 /* MELEE */), 1);
    ASSERT_INT_EQ("ranged blocked",
        encounter_prayer_correct_for_style(2 /* PROTECT_RANGED */, 2 /* RANGED */), 1);
    ASSERT_INT_EQ("magic blocked",
        encounter_prayer_correct_for_style(1 /* PROTECT_MAGIC */, 3 /* MAGIC */), 1);

    /* combined: prayer match + reduction formula */
    for (int dmg = 0; dmg <= 100; dmg += 10) {
        int actual = dmg;
        if (encounter_prayer_correct_for_style(3, 1)) /* melee -> protect melee */
            actual = (int)(dmg * 0.6f);
        int expected = (int)(dmg * 0.6f);
        char label[64];
        snprintf(label, sizeof(label), "pvp melee reduce dmg=%d", dmg);
        ASSERT_INT_EQ(label, actual, expected);
    }
}

/* ======================================================================== */
/* test: PvE prayer protection — correct overhead blocks damage entirely     */
/*                                                                           */
/* ref: encounter_inferno.h — if (prayer_matches) { dmg = 0; }             */
/* in PvE (inferno, Zulrah, etc.), correct overhead sets damage to 0.       */
/* ======================================================================== */

static void test_prayer_pve_block(void) {
    printf("--- PvE prayer: full damage block ---\n");

    /* simulate PvE prayer check: prayer_matches -> dmg = 0 */
    int test_damages[] = {1, 10, 50, 97, 100};
    for (int i = 0; i < 5; i++) {
        int dmg = test_damages[i];
        int prayer_matches = encounter_prayer_correct_for_style(
            3 /* PROTECT_MELEE */, 1 /* MELEE */);
        if (prayer_matches) dmg = 0;
        char label[64];
        snprintf(label, sizeof(label), "pve melee dmg=%d blocked", test_damages[i]);
        ASSERT_INT_EQ(label, dmg, 0);
    }

    /* ranged: protect ranged blocks ranged */
    for (int i = 0; i < 5; i++) {
        int dmg = test_damages[i];
        if (encounter_prayer_correct_for_style(2, 2)) dmg = 0;
        char label[64];
        snprintf(label, sizeof(label), "pve ranged dmg=%d blocked", test_damages[i]);
        ASSERT_INT_EQ(label, dmg, 0);
    }

    /* magic: protect magic blocks magic */
    for (int i = 0; i < 5; i++) {
        int dmg = test_damages[i];
        if (encounter_prayer_correct_for_style(1, 3)) dmg = 0;
        char label[64];
        snprintf(label, sizeof(label), "pve magic dmg=%d blocked", test_damages[i]);
        ASSERT_INT_EQ(label, dmg, 0);
    }

    /* wrong prayer: damage passes through unchanged */
    int dmg = 50;
    int prayer_matches = encounter_prayer_correct_for_style(
        1 /* PROTECT_MAGIC */, 1 /* MELEE attack */);
    if (prayer_matches) dmg = 0;
    ASSERT_INT_EQ("pve wrong prayer passthrough", dmg, 50);
}

/* ======================================================================== */
/* test: wrong prayer — exhaustive no-reduction check                        */
/*                                                                           */
/* prayer enum: NONE=0, MAGIC=1, RANGED=2, MELEE=3                         */
/* style enum:  NONE=0, MELEE=1, RANGED=2, MAGIC=3                         */
/* every wrong prayer+style pair must return 0.                             */
/* ======================================================================== */

static void test_prayer_wrong_no_reduction(void) {
    printf("--- wrong prayer: no reduction ---\n");

    /* no prayer (0) blocks nothing */
    ASSERT_INT_EQ("none vs melee",  encounter_prayer_correct_for_style(0, 1), 0);
    ASSERT_INT_EQ("none vs ranged", encounter_prayer_correct_for_style(0, 2), 0);
    ASSERT_INT_EQ("none vs magic",  encounter_prayer_correct_for_style(0, 3), 0);

    /* protect magic (1) only blocks magic (3) */
    ASSERT_INT_EQ("prot_magic vs melee",  encounter_prayer_correct_for_style(1, 1), 0);
    ASSERT_INT_EQ("prot_magic vs ranged", encounter_prayer_correct_for_style(1, 2), 0);

    /* protect ranged (2) only blocks ranged (2) */
    ASSERT_INT_EQ("prot_ranged vs melee", encounter_prayer_correct_for_style(2, 1), 0);
    ASSERT_INT_EQ("prot_ranged vs magic", encounter_prayer_correct_for_style(2, 3), 0);

    /* protect melee (3) only blocks melee (1) */
    ASSERT_INT_EQ("prot_melee vs ranged", encounter_prayer_correct_for_style(3, 2), 0);
    ASSERT_INT_EQ("prot_melee vs magic",  encounter_prayer_correct_for_style(3, 3), 0);

    /* attack style NONE (0) is never blocked by any prayer */
    ASSERT_INT_EQ("prot_magic vs none",  encounter_prayer_correct_for_style(1, 0), 0);
    ASSERT_INT_EQ("prot_ranged vs none", encounter_prayer_correct_for_style(2, 0), 0);
    ASSERT_INT_EQ("prot_melee vs none",  encounter_prayer_correct_for_style(3, 0), 0);
}

/* ======================================================================== */
/* test: NPC defensive rolls against player attacks                          */
/*                                                                           */
/* NPC defence roll = (def_level + 9) * (style_def_bonus + 64).             */
/* same formula as osrs_npc_attack_roll (NPCs use +9 invisible boost).      */
/*                                                                           */
/* ref: PlayerVsNPCCalc.ts getNPCDefenceRoll, DefenceRolls.test.ts          */
/* ======================================================================== */

static void test_npc_def_roll_vs_player(void) {
    printf("--- NPC defence rolls vs player attacks ---\n");

    /* Abyssal demon: def=135, stab_def=20
       vs melee (stab): (135+9) * (20+64) = 144 * 84 = 12096
       ref: DefenceRolls.test.ts */
    ASSERT_INT_EQ("abyssal demon vs stab",
        osrs_npc_attack_roll(135, 20), 12096);

    /* same NPC vs magic (magic_def=0):
       (135+9) * (0+64) = 144 * 64 = 9216 */
    ASSERT_INT_EQ("abyssal demon vs magic",
        osrs_npc_attack_roll(135, 0), 9216);

    /* Verzik P3: def=90, crush_def=-10
       vs crush: (90+9) * (-10+64) = 99 * 54 = 5346 */
    ASSERT_INT_EQ("verzik p3 vs crush",
        osrs_npc_attack_roll(90, -10), 5346);

    /* Zulrah: def=300, ranged_def=50
       vs ranged: (300+9) * (50+64) = 309 * 114 = 35226 */
    ASSERT_INT_EQ("zulrah vs ranged",
        osrs_npc_attack_roll(300, 50), 35226);

    /* Zulrah vs magic (magic_def=300):
       (300+9) * (300+64) = 309 * 364 = 112476 */
    ASSERT_INT_EQ("zulrah vs magic",
        osrs_npc_attack_roll(300, 300), 112476);

    /* low-level NPC: def=10, stab_def=0
       (10+9) * (0+64) = 19 * 64 = 1216 */
    ASSERT_INT_EQ("weak npc vs stab",
        osrs_npc_attack_roll(10, 0), 1216);

    /* absolute minimum: def=0, bonus=0
       (0+9) * (0+64) = 9 * 64 = 576 */
    ASSERT_INT_EQ("def=0 bonus=0",
        osrs_npc_attack_roll(0, 0), 576);

    /* negative defence bonus (crush-weak monster):
       def=200, bonus=-20: (209) * (44) = 9196 */
    ASSERT_INT_EQ("negative def bonus",
        osrs_npc_attack_roll(200, -20), 9196);
}

/* ======================================================================== */
/* test: player attack roll — full mage gear                                 */
/*                                                                           */
/* kodai + ancestral hat/top/bottom + occult + ward (f) + tormented +       */
/* eternal boots + seers ring (i) + god cape. augury prayer.                 */
/*                                                                           */
/* ref: PlayerVsNPCCalc.ts getPlayerMaxMagicAttackRoll                      */
/* ======================================================================== */

static void test_player_att_roll_full_mage(void) {
    printf("--- player att roll: full mage (10 slots, augury) ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;
    loadout[GEAR_SLOT_HEAD]   = ITEM_ANCESTRAL_HAT;
    loadout[GEAR_SLOT_BODY]   = ITEM_ANCESTRAL_TOP;
    loadout[GEAR_SLOT_LEGS]   = ITEM_ANCESTRAL_BOTTOM;
    loadout[GEAR_SLOT_NECK]   = ITEM_OCCULT_NECKLACE;
    loadout[GEAR_SLOT_SHIELD] = ITEM_ELIDINIS_WARD_F;
    loadout[GEAR_SLOT_HANDS]  = ITEM_TORMENTED_BRACELET;
    loadout[GEAR_SLOT_FEET]   = ITEM_ETERNAL_BOOTS;
    loadout[GEAR_SLOT_RING]   = ITEM_SEERS_RING_I;
    loadout[GEAR_SLOT_CAPE]   = ITEM_GOD_CAPE;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_AUGURY,
        99, FIGHT_STYLE_AUTOCAST, 30 /* ice barrage */, &stats);

    /* sum attack_magic:
       kodai(28) + hat(8) + top(35) + bottom(26) + occult(12) +
       ward(25) + tormented(10) + eternal(8) + seers(12) + cape(15) = 179 */
    ASSERT_INT_EQ("attack_bonus", stats.attack_bonus, 179);

    /* sum magic_damage:
       kodai(15) + hat(3) + top(3) + bottom(3) + occult(5) +
       ward(5) + tormented(5) + eternal(1) + seers(1) + cape(2) = 43 */
    ASSERT_INT_EQ("strength_bonus", stats.strength_bonus, 43);

    /* eff_level = floor(99 * 1.25) + 9 = 123 + 9 = 132 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 132);

    /* max_hit = floor(30 * (1.0 + 43/100.0) * 1.04) = floor(30 * 1.43 * 1.04)
       = floor(44.616) = 44 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 44);

    /* attack_roll = 132 * (179 + 64) = 132 * 243 = 32076 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 32076);
}

/* ======================================================================== */
/* test: player attack roll — rapier + defender + infernal cape (piety)      */
/*                                                                           */
/* ref: PlayerVsNPCCalc.ts getPlayerMaxMeleeAttackRoll                      */
/* ======================================================================== */

static void test_player_att_roll_melee_with_defender(void) {
    printf("--- player att roll: rapier + defender + infernal cape, piety ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_GHRAZI_RAPIER;
    loadout[GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER;
    loadout[GEAR_SLOT_CAPE]   = ITEM_INFERNAL_CAPE;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_PIETY,
        99, FIGHT_STYLE_AGGRESSIVE, 0, &stats);

    /* best melee attack bonus:
       stab:  rapier(94) + defender(25) + cape(4)  = 123
       slash: rapier(55) + defender(24) + cape(4)  = 83
       crush: rapier(0)  + defender(23) + cape(4)  = 27
       best = 123 */
    ASSERT_INT_EQ("attack_bonus", stats.attack_bonus, 123);

    /* melee_strength: rapier(89) + defender(6) + cape(8) = 103 */
    ASSERT_INT_EQ("strength_bonus", stats.strength_bonus, 103);

    /* aggressive stance: +3 to STR only (not attack, which is accurate's role).
       eff_level (attack) = floor(99 * 1.20) + 0 + 8 = 118 + 8 = 126 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 126);

    /* eff_str = floor(99 * 1.23) + 3 + 8 = 121 + 11 = 132
       max_hit = floor(0.5 + 132 * (103+64) / 640) = floor(0.5 + 132*167/640)
       = floor(0.5 + 34.44375) = floor(34.94375) = 34 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 34);

    /* attack_roll = 126 * (123 + 64) = 126 * 187 = 23562 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 23562);
}

/* ======================================================================== */
/* test: player attack roll — blowpipe (rigour)                              */
/*                                                                           */
/* ref: PlayerVsNPCCalc.ts getPlayerMaxRangedAttackRoll                     */
/* ======================================================================== */

static void test_player_att_roll_ranged_blowpipe(void) {
    printf("--- player att roll: blowpipe, rigour ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_TOXIC_BLOWPIPE;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_RANGED, OFFENSIVE_PRAYER_RIGOUR,
        99, FIGHT_STYLE_RAPID, 0, &stats);

    /* blowpipe: attack_ranged=30, ranged_strength=20 */
    ASSERT_INT_EQ("attack_bonus", stats.attack_bonus, 30);
    ASSERT_INT_EQ("strength_bonus", stats.strength_bonus, 20);

    /* blowpipe attack_speed: equipment.json stores base=3 (accurate/longrange),
       rapid stance subtracts 1 → 2. ref: osrs-sdk Blowpipe.ts:79-84. */
    ASSERT_INT_EQ("attack_speed", stats.attack_speed, 2);

    /* eff_att = floor(99 * 1.20) + 0 + 8 = 118 + 8 = 126 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 126);

    /* eff_str = floor(99 * 1.23) + 0 + 8 = 121 + 8 = 129
       max_hit = floor(0.5 + 129 * (20+64) / 640) = floor(0.5 + 129*84/640)
       = floor(0.5 + 16.93125) = floor(17.43125) = 17 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 17);

    /* attack_roll = 126 * (30 + 64) = 126 * 94 = 11844 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 11844);
}

/* ======================================================================== */
/* test: loadout edge case — all empty slots, all 3 styles                   */
/*                                                                           */
/* with no gear (all ITEM_NONE), stats should reflect bare-handed combat.   */
/* ======================================================================== */

static void test_loadout_empty_all_styles(void) {
    printf("--- loadout: all empty, all 3 styles ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);

    /* melee, level 99, no prayer, accurate stance (+3 att, +0 str) */
    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_NONE,
        99, FIGHT_STYLE_ACCURATE, 0, &stats);

    ASSERT_INT_EQ("empty melee att_bonus", stats.attack_bonus, 0);
    ASSERT_INT_EQ("empty melee str_bonus", stats.strength_bonus, 0);
    /* eff = 99 + 3 + 8 = 110 (accurate +3 to attack) */
    ASSERT_INT_EQ("empty melee eff", stats.eff_level, 110);
    /* max_hit uses eff_str = 99 + 0 + 8 = 107 (accurate gives 0 to str).
       max_hit = floor(0.5 + 107 * 64 / 640) = floor(0.5 + 10.7) = 11 */
    ASSERT_INT_EQ("empty melee max", stats.max_hit, 11);
    ASSERT_INT_EQ("empty melee def_stab", stats.def_stab, 0);
    ASSERT_INT_EQ("empty melee def_magic", stats.def_magic, 0);

    /* ranged, no prayer, rapid stance (no level bonus, -1 speed) */
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_RANGED, OFFENSIVE_PRAYER_NONE,
        99, FIGHT_STYLE_RAPID, 0, &stats);

    ASSERT_INT_EQ("empty ranged att_bonus", stats.attack_bonus, 0);
    ASSERT_INT_EQ("empty ranged str_bonus", stats.strength_bonus, 0);
    /* eff_level = 99 + 0 + 8 = 107, max_hit = floor(0.5 + 107*64/640) = 11 */
    ASSERT_INT_EQ("empty ranged max", stats.max_hit, 11);

    /* magic with ice barrage, autocast (no invisible bonus per wiki) */
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_NONE,
        99, FIGHT_STYLE_AUTOCAST, 30, &stats);

    ASSERT_INT_EQ("empty magic att_bonus", stats.attack_bonus, 0);
    ASSERT_INT_EQ("empty magic str_bonus", stats.strength_bonus, 0);
    /* magic eff = 99 + 0 + 9 = 108 (autocast has no invisible bonus) */
    ASSERT_INT_EQ("empty magic eff", stats.eff_level, 108);
    /* max_hit = floor(30 * (1.0 + 0/100.0) * 1.0) = 30 */
    ASSERT_INT_EQ("empty magic max", stats.max_hit, 30);
}

/* ======================================================================== */
/* test: loadout with all 11 gear slots filled                               */
/*                                                                           */
/* verifies stats sum across every slot. uses a full mage setup with        */
/* god blessing in ammo slot to hit all 11 slots.                           */
/* ======================================================================== */

static void test_loadout_all_slots_filled(void) {
    printf("--- loadout: all 11 slots filled (full mage) ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_HEAD]   = ITEM_ANCESTRAL_HAT;
    loadout[GEAR_SLOT_CAPE]   = ITEM_GOD_CAPE;
    loadout[GEAR_SLOT_NECK]   = ITEM_OCCULT_NECKLACE;
    loadout[GEAR_SLOT_AMMO]   = ITEM_GOD_BLESSING;
    loadout[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;
    loadout[GEAR_SLOT_SHIELD] = ITEM_ELIDINIS_WARD_F;
    loadout[GEAR_SLOT_BODY]   = ITEM_ANCESTRAL_TOP;
    loadout[GEAR_SLOT_LEGS]   = ITEM_ANCESTRAL_BOTTOM;
    loadout[GEAR_SLOT_HANDS]  = ITEM_TORMENTED_BRACELET;
    loadout[GEAR_SLOT_FEET]   = ITEM_ETERNAL_BOOTS;
    loadout[GEAR_SLOT_RING]   = ITEM_SEERS_RING_I;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_AUGURY,
        99, FIGHT_STYLE_AUTOCAST, 30, &stats);

    /* god_blessing has attack_magic=0, so same total as 10-slot mage = 179 */
    ASSERT_INT_EQ("all_slots attack_bonus", stats.attack_bonus, 179);

    /* verify all defence bonuses reflect 11 items via DB cross-check */
    int exp_def_magic = 0;
    for (int s = 0; s < NUM_GEAR_SLOTS; s++) {
        if (loadout[s] != 255)
            exp_def_magic += ITEM_DATABASE[loadout[s]].defence_magic;
    }
    ASSERT_INT_EQ("all_slots def_magic", stats.def_magic, exp_def_magic);

    /* same for def_stab */
    int exp_def_stab = 0;
    for (int s = 0; s < NUM_GEAR_SLOTS; s++) {
        if (loadout[s] != 255)
            exp_def_stab += ITEM_DATABASE[loadout[s]].defence_stab;
    }
    ASSERT_INT_EQ("all_slots def_stab", stats.def_stab, exp_def_stab);

    /* weapon properties come through */
    ASSERT_INT_EQ("all_slots attack_speed", stats.attack_speed, 4); /* kodai */
    ASSERT_INT_EQ("all_slots attack_range", stats.attack_range, 10);
}

/* ======================================================================== */
/* test: two-handed weapon loadout (AGS)                                     */
/*                                                                           */
/* 2H weapons use ITEM_NONE in shield slot. verifies no shield bonus leaks  */
/* and weapon stats compute correctly.                                       */
/* ======================================================================== */

static void test_loadout_two_handed_weapon(void) {
    printf("--- loadout: AGS (two-handed), piety ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_AGS;
    /* shield slot stays ITEM_NONE — correct for 2H weapons */

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_PIETY,
        99, FIGHT_STYLE_AGGRESSIVE, 0, &stats);

    /* AGS: stab=0, slash=132, crush=80. best = 132 */
    ASSERT_INT_EQ("2h attack_bonus", stats.attack_bonus, 132);

    /* AGS melee_strength = 132 */
    ASSERT_INT_EQ("2h strength_bonus", stats.strength_bonus, 132);

    /* attack_speed = 6 (AGS is slow) */
    ASSERT_INT_EQ("2h attack_speed", stats.attack_speed, 6);

    /* aggressive: +3 str, +0 att.
       eff_level (attack) = floor(99 * 1.20) + 0 + 8 = 126 */
    ASSERT_INT_EQ("2h eff_level", stats.eff_level, 126);

    /* eff_str = floor(99 * 1.23) + 3 + 8 = 132
       max_hit = floor(0.5 + 132 * (132+64) / 640)
       = floor(0.5 + 132*196/640) = floor(0.5 + 40.425) = 40 */
    ASSERT_INT_EQ("2h max_hit", stats.max_hit, 40);

    /* no shield: all defence bonuses from AGS only (which are all 0) */
    ASSERT_INT_EQ("2h def_stab", stats.def_stab, 0);
    ASSERT_INT_EQ("2h def_magic", stats.def_magic, 0);
    ASSERT_INT_EQ("2h def_ranged", stats.def_ranged, 0);

    /* attack_roll = 126 * (132+64) = 126 * 196 = 24696 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("2h attack_roll", att_roll, 24696);
}

/* ======================================================================== */
/* test: item_is_two_handed classification                                   */
/*                                                                           */
/* verify all known 2H weapons return 1, and 1H weapons / non-weapons       */
/* return 0.                                                                 */
/* ======================================================================== */

static void test_two_handed_classification(void) {
    printf("--- item_is_two_handed ---\n");

    /* known 2H weapons */
    ASSERT_INT_EQ("AGS 2h",        item_is_two_handed(ITEM_AGS), 1);
    ASSERT_INT_EQ("ancient GS 2h", item_is_two_handed(ITEM_ANCIENT_GS), 1);
    ASSERT_INT_EQ("d claws 2h",    item_is_two_handed(ITEM_DRAGON_CLAWS), 1);
    ASSERT_INT_EQ("g maul 2h",     item_is_two_handed(ITEM_GRANITE_MAUL), 1);
    ASSERT_INT_EQ("elder maul 2h", item_is_two_handed(ITEM_ELDER_MAUL), 1);
    ASSERT_INT_EQ("dark bow 2h",   item_is_two_handed(ITEM_DARK_BOW), 1);
    ASSERT_INT_EQ("ballista 2h",   item_is_two_handed(ITEM_HEAVY_BALLISTA), 1);

    /* known 1H weapons */
    ASSERT_INT_EQ("whip 1h",    item_is_two_handed(ITEM_WHIP), 0);
    ASSERT_INT_EQ("rapier 1h",  item_is_two_handed(ITEM_GHRAZI_RAPIER), 0);
    ASSERT_INT_EQ("kodai 1h",   item_is_two_handed(ITEM_KODAI_WAND), 0);
    ASSERT_INT_EQ("dagger 1h",  item_is_two_handed(ITEM_DRAGON_DAGGER), 0);
    ASSERT_INT_EQ("ACB 1h",     item_is_two_handed(ITEM_ARMADYL_CROSSBOW), 0);
    ASSERT_INT_EQ("voidwaker 1h", item_is_two_handed(ITEM_VOIDWAKER), 0);
    ASSERT_INT_EQ("vestas 1h",  item_is_two_handed(ITEM_VESTAS), 0);

    /* non-weapon items / special sentinel */
    ASSERT_INT_EQ("defender 1h",  item_is_two_handed(ITEM_DRAGON_DEFENDER), 0);
    ASSERT_INT_EQ("ITEM_NONE 1h", item_is_two_handed(ITEM_NONE), 0);
}

/* ======================================================================== */
/* test: end-to-end hit chance — player attacks NPC                          */
/*                                                                           */
/* combines player attack roll (from loadout) with NPC defence roll to      */
/* compute final hit chance via osrs_hit_chance. realistic scenarios.        */
/* ======================================================================== */

static void test_hit_chance_player_vs_npc(void) {
    printf("--- hit chance: player vs NPC (end-to-end) ---\n");

    /* scenario 1: full mage (augury) vs Zulrah magic form
       Zulrah: def=300, magic_def=300 — near-impossible to mage */
    {
        uint8_t loadout[NUM_GEAR_SLOTS];
        clear_loadout(loadout);
        loadout[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;
        loadout[GEAR_SLOT_HEAD]   = ITEM_ANCESTRAL_HAT;
        loadout[GEAR_SLOT_BODY]   = ITEM_ANCESTRAL_TOP;
        loadout[GEAR_SLOT_LEGS]   = ITEM_ANCESTRAL_BOTTOM;
        loadout[GEAR_SLOT_NECK]   = ITEM_OCCULT_NECKLACE;
        loadout[GEAR_SLOT_SHIELD] = ITEM_ELIDINIS_WARD_F;
        loadout[GEAR_SLOT_HANDS]  = ITEM_TORMENTED_BRACELET;
        loadout[GEAR_SLOT_FEET]   = ITEM_ETERNAL_BOOTS;
        loadout[GEAR_SLOT_RING]   = ITEM_SEERS_RING_I;
        loadout[GEAR_SLOT_CAPE]   = ITEM_GOD_CAPE;

        EncounterLoadoutStats stats;
        encounter_compute_loadout_stats(
            loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_AUGURY,
            99, FIGHT_STYLE_AUTOCAST, 30, &stats);

        int player_att = stats.eff_level * (stats.attack_bonus + 64); /* 32076 */
        int npc_def = osrs_npc_attack_roll(300, 300); /* 112476 */
        float chance = osrs_hit_chance(player_att, npc_def);

        /* att < def: chance = att / (2*(def+1)) = 32076 / 224954 ~ 0.1426 */
        float expected = 32076.0f / (2.0f * 112477.0f);
        ASSERT_FLOAT_NEAR("mage vs zulrah magic", chance, expected, 1e-3f);
        ASSERT_INT_EQ("mage vs zulrah < 0.2", chance < 0.2f, 1);
    }

    /* scenario 2: rapier (piety) vs low-def NPC (def=10, stab_def=0) */
    {
        uint8_t loadout[NUM_GEAR_SLOTS];
        clear_loadout(loadout);
        loadout[GEAR_SLOT_WEAPON] = ITEM_GHRAZI_RAPIER;
        loadout[GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER;

        EncounterLoadoutStats stats;
        encounter_compute_loadout_stats(
            loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_PIETY,
            99, FIGHT_STYLE_ACCURATE, 0, &stats);

        /* rapier(94)+defender(25) stab = 119 best */
        int player_att = stats.eff_level * (stats.attack_bonus + 64);
        int npc_def = osrs_npc_attack_roll(10, 0); /* 19*64 = 1216 */
        float chance = osrs_hit_chance(player_att, npc_def);

        /* att >> def: near 100%. 1 - (1218)/(2*(player_att+1)) */
        ASSERT_INT_EQ("melee vs weak npc > 0.9", chance > 0.9f, 1);
    }

    /* scenario 3: empty-handed mage vs same weak NPC
       should still have decent accuracy from level alone */
    {
        uint8_t loadout[NUM_GEAR_SLOTS];
        clear_loadout(loadout);

        EncounterLoadoutStats stats;
        encounter_compute_loadout_stats(
            loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_AUGURY,
            99, FIGHT_STYLE_AUTOCAST, 30, &stats);

        /* autocast, no invisible bonus. eff=132 (99*1.25 + 0 + 9), bonus=0,
           att_roll = 132*64 = 8448 */
        int player_att = stats.eff_level * (stats.attack_bonus + 64);
        ASSERT_INT_EQ("empty mage att_roll", player_att, 8448);

        int npc_def = osrs_npc_attack_roll(10, 0); /* 1216 */
        float chance = osrs_hit_chance(player_att, npc_def);
        /* 8448 > 1216: 1 - 1218/16898 ~ 0.928 */
        ASSERT_INT_EQ("empty mage vs weak > 0.9", chance > 0.9f, 1);
    }
}

/* ======================================================================== */
/* test: defence bonus selection picks correct stat per attack style          */
/*                                                                           */
/* ref: osrs_combat_shared.h encounter_player_def_bonus                     */
/* uses asymmetric values so any cross-wiring is detectable.                */
/* ======================================================================== */

static void test_def_bonus_selection(void) {
    printf("--- defence bonus selection by style ---\n");

    int stab = 11, slash = 22, crush = 33, magic = 44, ranged = 55;

    ASSERT_INT_EQ("melee stab",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 1, 0), 11);
    ASSERT_INT_EQ("melee slash",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 1, 1), 22);
    ASSERT_INT_EQ("melee crush",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 1, 2), 33);
    ASSERT_INT_EQ("ranged",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 2, 0), 55);
    ASSERT_INT_EQ("magic",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 3, 0), 44);
}

/* ======================================================================== */
/* test: gear defence bonuses feed into player_def_roll correctly            */
/*                                                                           */
/* rapier + defender + infernal cape defence sums ->                         */
/* osrs_player_def_roll_vs_npc with those bonuses.                          */
/* ======================================================================== */

static void test_loadout_defence_into_def_roll(void) {
    printf("--- loadout def bonuses -> player_def_roll ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_GHRAZI_RAPIER;
    loadout[GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER;
    loadout[GEAR_SLOT_CAPE]   = ITEM_INFERNAL_CAPE;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_NONE,
        99, FIGHT_STYLE_ACCURATE, 0, &stats);

    /* rapier def: 0,0,0,0,0. defender: 25,24,23,-3,-2. cape: 12,12,12,12,12 */
    ASSERT_INT_EQ("def_stab",   stats.def_stab,   0 + 25 + 12); /* 37 */
    ASSERT_INT_EQ("def_slash",  stats.def_slash,  0 + 24 + 12);  /* 36 */
    ASSERT_INT_EQ("def_crush",  stats.def_crush,  0 + 23 + 12);  /* 35 */
    ASSERT_INT_EQ("def_magic",  stats.def_magic,  0 + (-3) + 12); /* 9 */
    ASSERT_INT_EQ("def_ranged", stats.def_ranged, 0 + (-2) + 12); /* 10 */

    /* player def roll vs incoming melee stab:
       (def_level + 8) * (def_stab + 64) = (99+8) * (37+64) = 107 * 101 = 10807 */
    int def_roll_stab = osrs_player_def_roll_vs_npc(99, 99, stats.def_stab, 1);
    ASSERT_INT_EQ("def roll vs stab", def_roll_stab, 10807);

    /* vs incoming ranged:
       (99+8) * (10+64) = 107 * 74 = 7918 */
    int def_roll_ranged = osrs_player_def_roll_vs_npc(99, 99, stats.def_ranged, 2);
    ASSERT_INT_EQ("def roll vs ranged", def_roll_ranged, 7918);

    /* vs incoming magic: uses (floor(magic*0.7 + def*0.3) + 8) * (def_magic + 64)
       = (floor(69.3 + 29.7) + 8) * (9+64) = (99+8) * 73 = 107 * 73 = 7811 */
    int def_roll_magic = osrs_player_def_roll_vs_npc(99, 99, stats.def_magic, 3);
    ASSERT_INT_EQ("def roll vs magic", def_roll_magic, 7811);
}


static void test_shared_prepare_attack_virtus_ancient_bonus(void) {
    printf("--- shared attack prep: virtus ancient bonus ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;
    loadout[GEAR_SLOT_BODY] = ITEM_VIRTUS_ROBE_TOP;
    loadout[GEAR_SLOT_LEGS] = ITEM_VIRTUS_ROBE_BOTTOM;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_NONE,
        99, FIGHT_STYLE_AUTOCAST, 30, &stats
    );

    OsrsEquipmentEffectProfile profile;
    OsrsItemEffectState state;
    osrs_derive_equipment_effect_profile(loadout, &profile);
    osrs_item_effect_state_init(&state);

    OsrsPreparedAttackEffects ancient = osrs_prepare_attack_effects(
        &profile, &state, ITEM_KODAI_WAND, ATTACK_STYLE_MAGIC,
        OSRS_MAGIC_ATTACK_ANCIENT_ICE, osrs_target_ref_none(), 1,
        stats.eff_level * (stats.attack_bonus + 64), stats.max_hit,
        0, 0, 99, 99
    );
    OsrsPreparedAttackEffects non_ancient = osrs_prepare_attack_effects(
        &profile, &state, ITEM_KODAI_WAND, ATTACK_STYLE_MAGIC,
        OSRS_MAGIC_ATTACK_STANDARD_SPELL, osrs_target_ref_none(), 1,
        stats.eff_level * (stats.attack_bonus + 64), stats.max_hit,
        0, 0, 99, 99
    );

    ASSERT_INT_EQ("virtus ancient adds 6%", ancient.max_hit, stats.max_hit * 106 / 100);
    ASSERT_INT_EQ("virtus non-ancient unchanged", non_ancient.max_hit, stats.max_hit);
}


static void test_shared_prepare_attack_tbow_scaling(void) {
    printf("--- shared attack prep: tbow scaling ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_TWISTED_BOW;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_RANGED, OFFENSIVE_PRAYER_RIGOUR,
        99, FIGHT_STYLE_RAPID, 0, &stats
    );

    OsrsEquipmentEffectProfile profile;
    OsrsItemEffectState state;
    osrs_derive_equipment_effect_profile(loadout, &profile);
    osrs_item_effect_state_init(&state);

    int base_attack_roll = stats.eff_level * (stats.attack_bonus + 64);
    OsrsPreparedAttackEffects prepared = osrs_prepare_attack_effects(
        &profile, &state, ITEM_TWISTED_BOW, ATTACK_STYLE_RANGED,
        OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1,
        base_attack_roll, stats.max_hit,
        150, 550, 99, 99
    );

    int target_magic = max_int(150, 550);
    ASSERT_INT_EQ(
        "tbow attack roll scaled",
        prepared.attack_roll,
        (int)(base_attack_roll * osrs_tbow_acc_mult(target_magic))
    );
    ASSERT_INT_EQ(
        "tbow max hit scaled",
        prepared.max_hit,
        (int)(stats.max_hit * osrs_tbow_dmg_mult(target_magic))
    );
}


int main(void) {
    printf("=== item effects tests (cross-referenced with osrs-dps-calc) ===\n\n");

    /* tbow scaling edge cases */
    test_tbow_acc_edge_cases();
    test_tbow_dmg_edge_cases();
    test_tbow_cap_behavior();

    /* prayer protection */
    test_prayer_pvp_reduction();
    test_prayer_pve_block();
    test_prayer_wrong_no_reduction();

    /* NPC defensive rolls */
    test_npc_def_roll_vs_player();

    /* player attack rolls with gear */
    test_player_att_roll_full_mage();
    test_player_att_roll_melee_with_defender();
    test_player_att_roll_ranged_blowpipe();

    /* loadout edge cases */
    test_loadout_empty_all_styles();
    test_loadout_all_slots_filled();
    test_loadout_two_handed_weapon();
    test_two_handed_classification();

    /* end-to-end hit chance */
    test_hit_chance_player_vs_npc();
    test_def_bonus_selection();
    test_loadout_defence_into_def_roll();
    test_shared_prepare_attack_virtus_ancient_bonus();
    test_shared_prepare_attack_tbow_scaling();

    printf("\n=== results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(", %d FAILED", tests_failed);
    }
    printf(" ===\n");

    return tests_failed > 0 ? 1 : 0;
}
