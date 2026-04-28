/**
 * @file test_combat_math.c
 * @brief combat math tests cross-referenced against osrs-dps-calc reference.
 *
 * tests the pure combat math functions in osrs_combat_shared.h and the loadout
 * stat computation in osrs_encounter.h against hand-computed expected values
 * derived from the TypeScript reference (.refs/osrs-dps-calc/).
 *
 * BUILD:
 *   cd pufferlib-metal
 *   cc -std=c11 -O0 -g -I. -o test_combat_math \
 *       ocean/osrs/tests/test_combat_math.c -lm
 *   ./test_combat_math
 *
 * REFERENCE FILES:
 *   .refs/osrs-dps-calc/src/lib/BaseCalc.ts      — getNormalAccuracyRoll
 *   .refs/osrs-dps-calc/src/lib/PlayerVsNPCCalc.ts — player formulas, tbow scaling
 *   .refs/osrs-dps-calc/src/tests/calc/BasicRolls.test.ts — reference test values
 *   .refs/osrs-dps-calc/src/tests/calc/DefenceRolls.test.ts
 *   .refs/osrs-dps-calc/src/tests/calc/Prayers.test.ts
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ocean/osrs/osrs_encounter.h"
#include "ocean/osrs/osrs_special_attacks.h"


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

/* ======================================================================== */
/* test: osrs_hit_chance                                                     */
/*                                                                           */
/* ref: BaseCalc.ts getNormalAccuracyRoll                                    */
/*   att > def: 1 - (def + 2) / (2 * (att + 1))                            */
/*   att <= def: att / (2 * (def + 1))                                      */
/* ======================================================================== */

static void test_hit_chance(void) {
    printf("--- osrs_hit_chance ---\n");

    /* att == def == 0: 0 / (2 * 1) = 0.0 */
    ASSERT_FLOAT_NEAR("att=0 def=0", osrs_hit_chance(0, 0), 0.0f, 1e-5f);

    /* att > def: 1 - (def+2) / (2*(att+1)) */
    /* att=10000, def=5000: 1 - 5002 / 20002 = 0.749975... */
    ASSERT_FLOAT_NEAR("att=10000 def=5000",
        osrs_hit_chance(10000, 5000),
        1.0f - 5002.0f / 20002.0f, 1e-4f);

    /* att < def: att / (2*(def+1)) */
    /* att=5000, def=10000: 5000 / 20002 = 0.249975... */
    ASSERT_FLOAT_NEAR("att=5000 def=10000",
        osrs_hit_chance(5000, 10000),
        5000.0f / 20002.0f, 1e-4f);

    /* att == def (non-zero): att / (2*(def+1)) */
    /* att=1000, def=1000: 1000 / 2002 = 0.49950... */
    ASSERT_FLOAT_NEAR("att=1000 def=1000",
        osrs_hit_chance(1000, 1000),
        1000.0f / 2002.0f, 1e-4f);

    /* very high att, def=0: near 100% */
    /* 1 - 2 / (2 * 100001) = 1 - 0.00001 = 0.99999 */
    ASSERT_FLOAT_NEAR("att=100000 def=0",
        osrs_hit_chance(100000, 0),
        1.0f - 2.0f / 200002.0f, 1e-4f);

    /* att=0, def=100: 0 / 202 = 0.0 */
    ASSERT_FLOAT_NEAR("att=0 def=100", osrs_hit_chance(0, 100), 0.0f, 1e-5f);

    /* exact boundary: att == def+1 (att just barely > def) */
    /* att=101, def=100: 1 - 102 / 204 = 1 - 0.5 = 0.5 */
    ASSERT_FLOAT_NEAR("att=101 def=100",
        osrs_hit_chance(101, 100),
        1.0f - 102.0f / 204.0f, 1e-4f);

    /* realistic combat scenario: player att ~20000, NPC def ~12000 */
    /* 1 - 12002 / 40002 = 0.69995... */
    ASSERT_FLOAT_NEAR("att=20000 def=12000",
        osrs_hit_chance(20000, 12000),
        1.0f - 12002.0f / 40002.0f, 1e-4f);
}

/* ======================================================================== */
/* test: NPC melee max hit                                                   */
/*                                                                           */
/* formula: floor((str + 8) * (bonus + 64) + 320) / 640)                    */
/* this is a simplified NPC formula (not player). ref: OSRS wiki.            */
/* ======================================================================== */

static void test_npc_melee_max_hit(void) {
    printf("--- osrs_npc_melee_max_hit ---\n");

    /* str=1, bonus=0: ((1+8)*(0+64)+320)/640 = (576+320)/640 = 896/640 = 1 */
    ASSERT_INT_EQ("str=1 bonus=0", osrs_npc_melee_max_hit(1, 0), 1);

    /* str=250, bonus=0: ((258)*(64)+320)/640 = (16512+320)/640 = 16832/640 = 26 */
    ASSERT_INT_EQ("str=250 bonus=0", osrs_npc_melee_max_hit(250, 0), 26);

    /* str=200, bonus=50: ((208)*(114)+320)/640 = (23712+320)/640 = 24032/640 = 37 */
    ASSERT_INT_EQ("str=200 bonus=50", osrs_npc_melee_max_hit(200, 50), 37);

    /* str=120, bonus=80: ((128)*(144)+320)/640 = (18432+320)/640 = 18752/640 = 29 */
    ASSERT_INT_EQ("str=120 bonus=80", osrs_npc_melee_max_hit(120, 80), 29);

    /* str=0, bonus=0: ((8)*(64)+320)/640 = (512+320)/640 = 832/640 = 1 */
    ASSERT_INT_EQ("str=0 bonus=0", osrs_npc_melee_max_hit(0, 0), 1);
}

/* ======================================================================== */
/* test: NPC ranged max hit                                                  */
/*                                                                           */
/* formula: floor(0.5 + (range + 8) * (bonus + 64) / 640)                  */
/* ======================================================================== */

static void test_npc_ranged_max_hit(void) {
    printf("--- osrs_npc_ranged_max_hit ---\n");

    /* range=1, bonus=0: 0.5 + 9*64/640 = 0.5 + 0.9 = 1.4 -> 1 */
    ASSERT_INT_EQ("range=1 bonus=0", osrs_npc_ranged_max_hit(1, 0), 1);

    /* range=250, bonus=0: 0.5 + 258*64/640 = 0.5 + 25.8 = 26.3 -> 26 */
    ASSERT_INT_EQ("range=250 bonus=0", osrs_npc_ranged_max_hit(250, 0), 26);

    /* range=120, bonus=80: 0.5 + 128*144/640 = 0.5 + 28.8 = 29.3 -> 29 */
    ASSERT_INT_EQ("range=120 bonus=80", osrs_npc_ranged_max_hit(120, 80), 29);

    /* range=0, bonus=0: 0.5 + 8*64/640 = 0.5 + 0.8 = 1.3 -> 1 */
    ASSERT_INT_EQ("range=0 bonus=0", osrs_npc_ranged_max_hit(0, 0), 1);
}

/* ======================================================================== */
/* test: NPC magic max hit                                                   */
/*                                                                           */
/* formula: base_spell_dmg * magic_dmg_pct / 100 (integer division)         */
/* ======================================================================== */

static void test_npc_magic_max_hit(void) {
    printf("--- osrs_npc_magic_max_hit ---\n");

    ASSERT_INT_EQ("base=30 pct=100", osrs_npc_magic_max_hit(30, 100), 30);
    ASSERT_INT_EQ("base=30 pct=175", osrs_npc_magic_max_hit(30, 175), 52);
    ASSERT_INT_EQ("base=46 pct=100", osrs_npc_magic_max_hit(46, 100), 46);
    ASSERT_INT_EQ("base=0 pct=200", osrs_npc_magic_max_hit(0, 200), 0);
    ASSERT_INT_EQ("base=50 pct=150", osrs_npc_magic_max_hit(50, 150), 75);
}

/* ======================================================================== */
/* test: NPC attack roll                                                     */
/*                                                                           */
/* formula: (att_level + 9) * (att_bonus + 64)                              */
/* NPCs have +9 invisible boost (not +8 like players with no stance).       */
/* ref: OSRS wiki, PlayerVsNPCCalc.ts getNPCDefenceRoll (uses +9 for NPCs) */
/* ======================================================================== */

static void test_npc_attack_roll(void) {
    printf("--- osrs_npc_attack_roll ---\n");

    /* att=250, bonus=0: (259)*(64) = 16576 */
    ASSERT_INT_EQ("att=250 bonus=0", osrs_npc_attack_roll(250, 0), 16576);

    /* att=1, bonus=0: (10)*(64) = 640 */
    ASSERT_INT_EQ("att=1 bonus=0", osrs_npc_attack_roll(1, 0), 640);

    /* att=0, bonus=0: (9)*(64) = 576 */
    ASSERT_INT_EQ("att=0 bonus=0", osrs_npc_attack_roll(0, 0), 576);

    /* att=180, bonus=176: (189)*(240) = 45360 */
    ASSERT_INT_EQ("att=180 bonus=176", osrs_npc_attack_roll(180, 176), 45360);

    /* Abyssal demon defence roll check:
       ref: DefenceRolls.test.ts — Abyssal demon def_level=135, def_bonus=20 (stab)
       NPC def roll = (135+9) * (20+64) = 144 * 84 = 12096 */
    ASSERT_INT_EQ("abyssal demon (135,20)", osrs_npc_attack_roll(135, 20), 12096);
}

/* ======================================================================== */
/* test: osrs_npc_max_hit dispatch                                           */
/*                                                                           */
/* verifies the style-based dispatcher returns correct values.               */
/* ======================================================================== */

static void test_npc_max_hit_dispatch(void) {
    printf("--- osrs_npc_max_hit (dispatch) ---\n");

    /* melee: str=200, melee_str=50 -> same as osrs_npc_melee_max_hit(200, 50) */
    ASSERT_INT_EQ("melee dispatch",
        osrs_npc_max_hit(1 /*melee*/, 200, 0, 50, 0, 0, 100),
        osrs_npc_melee_max_hit(200, 50));

    /* ranged: range=120, ranged_str=80 -> same as osrs_npc_ranged_max_hit(120, 80) */
    ASSERT_INT_EQ("ranged dispatch",
        osrs_npc_max_hit(2 /*ranged*/, 0, 120, 0, 80, 0, 100),
        osrs_npc_ranged_max_hit(120, 80));

    /* magic: base=30, pct=175 -> same as osrs_npc_magic_max_hit(30, 175) */
    ASSERT_INT_EQ("magic dispatch",
        osrs_npc_max_hit(3 /*magic*/, 0, 0, 0, 0, 30, 175),
        osrs_npc_magic_max_hit(30, 175));

    /* style=0 (none): should return 0 */
    ASSERT_INT_EQ("none dispatch", osrs_npc_max_hit(0, 200, 200, 50, 50, 30, 175), 0);
}

/* ======================================================================== */
/* test: player defence roll vs NPC                                          */
/*                                                                           */
/* ref: BaseCalc.ts / OSRS wiki                                             */
/*   vs melee/ranged: (def_level + 8) * (def_bonus + 64)                   */
/*   vs magic: (floor(magic*0.7 + def*0.3) + 8) * (def_bonus + 64)        */
/* ======================================================================== */

static void test_player_def_roll(void) {
    printf("--- osrs_player_def_roll_vs_npc ---\n");

    /* vs melee, def=99, magic=99, def_bonus=200 */
    /* (99 + 8) * (200 + 64) = 107 * 264 = 28248 */
    ASSERT_INT_EQ("vs melee def=99 bonus=200",
        osrs_player_def_roll_vs_npc(99, 99, 200, 1 /* melee */), 28248);

    /* vs ranged, def=70, magic=99, def_bonus=100 */
    /* (70 + 8) * (100 + 64) = 78 * 164 = 12792 */
    ASSERT_INT_EQ("vs ranged def=70 bonus=100",
        osrs_player_def_roll_vs_npc(70, 99, 100, 2 /* ranged */), 12792);

    /* vs magic, def=70, magic=99, def_bonus=100 */
    /* floor(99*0.7 + 70*0.3) = floor(69.3 + 21.0) = floor(90.3) = 90 */
    /* (90 + 8) * (100 + 64) = 98 * 164 = 16072 */
    ASSERT_INT_EQ("vs magic def=70 magic=99 bonus=100",
        osrs_player_def_roll_vs_npc(70, 99, 100, 3 /* magic */), 16072);

    /* vs magic with equal levels, should give same eff as melee */
    /* def=99, magic=99: floor(99*0.7+99*0.3) = floor(99) = 99 */
    ASSERT_INT_EQ("vs magic equal levels",
        osrs_player_def_roll_vs_npc(99, 99, 200, 3 /* magic */), 28248);

    /* vs magic low magic, high def: floor(1*0.7+99*0.3) = floor(0.7+29.7) = floor(30.4) = 30 */
    /* (30+8)*264 = 38*264 = 10032 */
    ASSERT_INT_EQ("vs magic low_magic high_def",
        osrs_player_def_roll_vs_npc(99, 1, 200, 3 /* magic */), 10032);
}

/* ======================================================================== */
/* test: encounter_player_def_bonus                                          */
/*                                                                           */
/* selects the correct defence bonus for incoming NPC attack.               */
/* ======================================================================== */

static void test_player_def_bonus(void) {
    printf("--- encounter_player_def_bonus ---\n");

    int stab=100, slash=110, crush=120, magic=130, ranged=140;

    ASSERT_INT_EQ("ranged attack",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 2, 0), 140);
    ASSERT_INT_EQ("magic attack",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 3, 0), 130);
    ASSERT_INT_EQ("melee stab",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 1, 0), 100);
    ASSERT_INT_EQ("melee slash",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 1, 1), 110);
    ASSERT_INT_EQ("melee crush",
        encounter_player_def_bonus(stab, slash, crush, magic, ranged, 1, 2), 120);
}

/* ======================================================================== */
/* test: overhead prayer style check                                         */
/*                                                                           */
/* ref: OSRS wiki prayer mechanics                                          */
/*   melee attack (1) blocked by protect melee (3)                          */
/*   ranged attack (2) blocked by protect ranged (2)                        */
/*   magic attack (3) blocked by protect magic (1)                          */
/* ======================================================================== */

static void test_prayer_correct(void) {
    printf("--- encounter_prayer_correct_for_style ---\n");

    /* correct prayers */
    ASSERT_INT_EQ("melee->protect melee",
        encounter_prayer_correct_for_style(3 /* PRAYER_PROTECT_MELEE */, 1 /* MELEE */), 1);
    ASSERT_INT_EQ("ranged->protect ranged",
        encounter_prayer_correct_for_style(2 /* PRAYER_PROTECT_RANGED */, 2 /* RANGED */), 1);
    ASSERT_INT_EQ("magic->protect magic",
        encounter_prayer_correct_for_style(1 /* PRAYER_PROTECT_MAGIC */, 3 /* MAGIC */), 1);

    /* wrong prayers */
    ASSERT_INT_EQ("melee->protect magic",
        encounter_prayer_correct_for_style(1, 1), 0);
    ASSERT_INT_EQ("melee->protect ranged",
        encounter_prayer_correct_for_style(2, 1), 0);
    ASSERT_INT_EQ("ranged->protect melee",
        encounter_prayer_correct_for_style(3, 2), 0);
    ASSERT_INT_EQ("magic->protect ranged",
        encounter_prayer_correct_for_style(2, 3), 0);

    /* no prayer blocks nothing */
    ASSERT_INT_EQ("none->melee",
        encounter_prayer_correct_for_style(0, 1), 0);
    ASSERT_INT_EQ("none->ranged",
        encounter_prayer_correct_for_style(0, 2), 0);
    ASSERT_INT_EQ("none->magic",
        encounter_prayer_correct_for_style(0, 3), 0);
}

/* ======================================================================== */
/* test: hit delay formulas                                                  */
/*                                                                           */
/* ref: osrs-sdk, InfernoTrainer blowpipe.ts, MagicWeapon.ts               */
/*   magic:    floor((1 + distance) / 3) + 1 [+1 if player]                */
/*   ranged:   floor((3 + distance) / 6) + 1 [+1 if player]                */
/*   blowpipe: floor(distance / 6) + 1 [+1 if player]                      */
/* ======================================================================== */

static void test_hit_delays(void) {
    printf("--- hit delay formulas ---\n");

    /* magic: floor((1+d)/3) + 1 + is_player */
    ASSERT_INT_EQ("magic d=1 npc",  encounter_magic_hit_delay(1, 0), 1); /* (2/3)=0, +1 = 1 */
    ASSERT_INT_EQ("magic d=1 plr",  encounter_magic_hit_delay(1, 1), 2);
    ASSERT_INT_EQ("magic d=5 npc",  encounter_magic_hit_delay(5, 0), 3); /* (6/3)=2, +1 = 3 */
    ASSERT_INT_EQ("magic d=5 plr",  encounter_magic_hit_delay(5, 1), 4);
    ASSERT_INT_EQ("magic d=10 npc", encounter_magic_hit_delay(10, 0), 4); /* (11/3)=3, +1 = 4 */

    /* ranged: floor((3+d)/6) + 1 + is_player */
    ASSERT_INT_EQ("ranged d=1 npc",  encounter_ranged_hit_delay(1, 0), 1); /* (4/6)=0, +1 = 1 */
    ASSERT_INT_EQ("ranged d=1 plr",  encounter_ranged_hit_delay(1, 1), 2);
    ASSERT_INT_EQ("ranged d=6 npc",  encounter_ranged_hit_delay(6, 0), 2); /* (9/6)=1, +1 = 2 */
    ASSERT_INT_EQ("ranged d=10 npc", encounter_ranged_hit_delay(10, 0), 3); /* (13/6)=2, +1 = 3 */

    /* blowpipe: floor(d/6) + 1 + is_player */
    ASSERT_INT_EQ("bp d=1 npc",  encounter_blowpipe_hit_delay(1, 0), 1); /* 0+1=1 */
    ASSERT_INT_EQ("bp d=1 plr",  encounter_blowpipe_hit_delay(1, 1), 2);
    ASSERT_INT_EQ("bp d=6 npc",  encounter_blowpipe_hit_delay(6, 0), 2); /* 1+1=2 */
    ASSERT_INT_EQ("bp d=6 plr",  encounter_blowpipe_hit_delay(6, 1), 3);
    ASSERT_INT_EQ("bp d=12 npc", encounter_blowpipe_hit_delay(12, 0), 3); /* 2+1=3 */
}

/* ======================================================================== */
/* test: chebyshev distance to multi-tile NPC                                */
/*                                                                           */
/* encounter_dist_to_npc(px, py, nx, ny, npc_size)                          */
/* returns chebyshev distance from (px,py) to nearest tile of NPC at (nx,ny)*/
/* ======================================================================== */

static void test_dist_to_npc(void) {
    printf("--- encounter_dist_to_npc ---\n");

    /* 1x1 NPC at (5,5), player at (8,5): dx=3, dy=0 -> 3 */
    ASSERT_INT_EQ("1x1 same row", encounter_dist_to_npc(8, 5, 5, 5, 1), 3);

    /* 1x1 NPC at (5,5), player ON NPC tile: 0 */
    ASSERT_INT_EQ("1x1 on top", encounter_dist_to_npc(5, 5, 5, 5, 1), 0);

    /* 3x3 NPC at (5,5) occupies (5,5)-(7,7). player at (5,5): inside -> 0 */
    ASSERT_INT_EQ("3x3 inside", encounter_dist_to_npc(5, 5, 5, 5, 3), 0);

    /* 3x3 NPC at (5,5). player at (9,6): nearest tile (7,6), dx=2, dy=0 -> 2 */
    ASSERT_INT_EQ("3x3 right", encounter_dist_to_npc(9, 6, 5, 5, 3), 2);

    /* 5x5 NPC at (10,10) occupies (10-14, 10-14). player at (10,8): nearest (10,10), dist=2 */
    ASSERT_INT_EQ("5x5 below", encounter_dist_to_npc(10, 8, 10, 10, 5), 2);

    /* 5x5 NPC at (10,10). player at (17,17): nearest (14,14), dx=3,dy=3 -> 3 */
    ASSERT_INT_EQ("5x5 diagonal", encounter_dist_to_npc(17, 17, 10, 10, 5), 3);

    /* player adjacent to 3x3: (5,5)-(7,7), player at (4,4): nearest(5,5), dx=1,dy=1 -> 1 */
    ASSERT_INT_EQ("3x3 diagonal adj", encounter_dist_to_npc(4, 4, 5, 5, 3), 1);
}

/* ======================================================================== */
/* test: twisted bow multipliers                                             */
/*                                                                           */
/* our C code returns a float multiplier. the reference uses integer         */
/* truncation on intermediates which causes small divergences.               */
/*                                                                           */
/* ref: PlayerVsNPCCalc.ts tbowScaling()                                    */
/*   accuracy: factor=10, base=140                                          */
/*     t2 = trunc((3*m - 10) / 100)                                        */
/*     t3 = trunc((trunc(3*m/10) - 100)^2 / 100)                           */
/*     bonus = 140 + t2 - t3                                                */
/*   damage: factor=14, base=250                                            */
/*     t2 = trunc((3*m - 14) / 100)                                        */
/*     t3 = trunc((trunc(3*m/10) - 140)^2 / 100)                           */
/*     bonus = 250 + t2 - t3                                                */
/*                                                                           */
/* our C uses float division (no integer truncation of intermediates).       */
/* expected divergence: up to ~0.01 in the multiplier.                      */
/* ======================================================================== */

/* reference tbow accuracy multiplier using integer truncation (matching TS) */
static float ref_tbow_acc_mult(int magic) {
    int m = magic < 250 ? magic : 250;
    int t2 = (3 * m - 10) / 100;  /* C integer division truncates toward zero */
    int inner = (3 * m / 10) - 100;
    int t3 = (inner * inner) / 100;
    int bonus = 140 + t2 - t3;
    float mult = (float)bonus / 100.0f;
    if (mult > 1.4f) mult = 1.4f;
    if (mult < 0.0f) mult = 0.0f;
    return mult;
}

/* reference tbow damage multiplier using integer truncation (matching TS) */
static float ref_tbow_dmg_mult(int magic) {
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

static void test_tbow_multipliers(void) {
    printf("--- osrs_tbow_acc_mult / osrs_tbow_dmg_mult ---\n");

    /* test accuracy multiplier shape:
       low magic -> low mult, high magic -> high mult (up to 1.4 cap) */

    /* magic=0: reference bonus = 140 + 0 - 100 = 40, mult = 0.40 */
    ASSERT_FLOAT_NEAR("acc m=0 vs ref", osrs_tbow_acc_mult(0), ref_tbow_acc_mult(0), 0.01f);

    /* magic=100: reference bonus = 140 + 2 - 49 = 93, mult = 0.93 */
    ASSERT_FLOAT_NEAR("acc m=100 vs ref", osrs_tbow_acc_mult(100), ref_tbow_acc_mult(100), 0.01f);

    /* magic=200: reference bonus = 140 + 5 - 16 = 129, mult = 1.29 */
    ASSERT_FLOAT_NEAR("acc m=200 vs ref", osrs_tbow_acc_mult(200), ref_tbow_acc_mult(200), 0.01f);

    /* magic=250: should be at or near cap (1.40) */
    ASSERT_FLOAT_NEAR("acc m=250 vs ref", osrs_tbow_acc_mult(250), ref_tbow_acc_mult(250), 0.01f);
    float acc250 = osrs_tbow_acc_mult(250);
    ASSERT_INT_EQ("acc m=250 capped at 1.4", acc250 <= 1.4f + 0.001f, 1);

    /* magic=350: should be capped same as 250 */
    ASSERT_FLOAT_NEAR("acc m=350 == m=250", osrs_tbow_acc_mult(350), osrs_tbow_acc_mult(250), 1e-5f);

    /* damage multiplier */
    /* magic=0: reference bonus = 250 + 0 - 196 = 54, mult = 0.54 */
    ASSERT_FLOAT_NEAR("dmg m=0 vs ref", osrs_tbow_dmg_mult(0), ref_tbow_dmg_mult(0), 0.015f);

    /* magic=100: reference bonus = 250 + 2 - 16 = 236, mult = 2.36 */
    ASSERT_FLOAT_NEAR("dmg m=100 vs ref", osrs_tbow_dmg_mult(100), ref_tbow_dmg_mult(100), 0.01f);

    /* magic=250: reference bonus = 250 + 7 - 42 = 215, mult = 2.15 */
    ASSERT_FLOAT_NEAR("dmg m=250 vs ref", osrs_tbow_dmg_mult(250), ref_tbow_dmg_mult(250), 0.015f);

    /* verify monotonicity: higher magic = higher multipliers */
    ASSERT_INT_EQ("acc monotonic 0<100", osrs_tbow_acc_mult(0) < osrs_tbow_acc_mult(100), 1);
    ASSERT_INT_EQ("acc monotonic 100<200", osrs_tbow_acc_mult(100) < osrs_tbow_acc_mult(200), 1);
    ASSERT_INT_EQ("dmg monotonic 0<100", osrs_tbow_dmg_mult(0) < osrs_tbow_dmg_mult(100), 1);
}

/* ======================================================================== */
/* test: blowpipe special attack                                             */
/*                                                                           */
/* ref: Blowpipe.ts — 2x accuracy, 1.5x max hit                            */
/*   spec_att_roll = base_att_roll * 2                                      */
/*   spec_max = base_max_hit * 3 / 2                                        */
/*   def_roll = (target_def + 8) * (target_ranged_def + 64)                */
/* ======================================================================== */

static void test_blowpipe_spec(void) {
    printf("--- osrs_blowpipe_spec_resolve ---\n");

    /* test the spec constants */
    ASSERT_INT_EQ("spec acc mult", BLOWPIPE_SPEC_ACC_MULT, 2);
    ASSERT_INT_EQ("spec dmg num", BLOWPIPE_SPEC_DMG_NUM, 3);
    ASSERT_INT_EQ("spec dmg den", BLOWPIPE_SPEC_DMG_DEN, 2);
    ASSERT_INT_EQ("spec heal pct", BLOWPIPE_SPEC_HEAL_PCT, 50);
    ASSERT_INT_EQ("spec cost", BLOWPIPE_SPEC_COST, 50);

    /* verify the formula produces values in range with fixed RNG */
    uint32_t rng = 12345;
    int base_att = 20000;
    int base_max = 30;
    int target_def = 100;
    int target_ranged_def = 50;

    int dmg = osrs_blowpipe_spec_resolve(
        base_att, base_max, target_def, target_ranged_def, &rng);

    /* spec_max = 30 * 3 / 2 = 45 */
    ASSERT_INT_EQ("spec dmg in range", dmg >= 0 && dmg <= 45, 1);

    /* run many trials to check range */
    int max_seen = 0, num_zeros = 0;
    rng = 42;
    for (int i = 0; i < 10000; i++) {
        int d = osrs_blowpipe_spec_resolve(
            base_att, base_max, target_def, target_ranged_def, &rng);
        if (d > max_seen) max_seen = d;
        if (d == 0) num_zeros++;
    }
    /* max should be close to spec_max=45 */
    ASSERT_INT_EQ("spec max close to 45", max_seen >= 40 && max_seen <= 45, 1);
    /* should have some misses */
    ASSERT_INT_EQ("spec has misses", num_zeros > 0, 1);
}

/* ======================================================================== */
/* test: encounter_compute_loadout_stats (player loadout)                    */
/*                                                                           */
/* ref: PlayerVsNPCCalc.ts getPlayerMaxMeleeHit, getPlayerMaxMeleeAttackRoll*/
/*                                                                           */
/* tests loadout stat computation using items from ITEM_DATABASE.            */
/* base_level=99 to match typical inferno/pvm scenarios.                     */
/* ======================================================================== */

/* helper: fill loadout with ITEM_NONE */
static void clear_loadout(uint8_t loadout[NUM_GEAR_SLOTS]) {
    memset(loadout, 255, NUM_GEAR_SLOTS);
}

static void test_loadout_melee_no_prayer(void) {
    printf("--- loadout: ghrazi rapier, no prayer, aggressive ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_GHRAZI_RAPIER;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout,
        ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_NONE,
        99,
        FIGHT_STYLE_AGGRESSIVE,
        0,    /* spell_base_damage */
        &stats
    );

    /* rapier: attack_stab=94, attack_slash=55, attack_crush=0 -> best = 94 */
    ASSERT_INT_EQ("attack_bonus", stats.attack_bonus, 94);

    /* rapier: melee_strength = 89 */
    ASSERT_INT_EQ("strength_bonus", stats.strength_bonus, 89);

    /* aggressive: +3 str only, no att bonus.
       eff_level (attack) = floor(99 * 1.0) + 0 + 8 = 107 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 107);

    /* eff_str = floor(99 * 1.0) + 3 + 8 = 110 */
    /* max_hit = floor(0.5 + 110 * (89+64) / 640) = floor(0.5 + 110*153/640) */
    /* = floor(0.5 + 26.296875) = floor(26.796875) = 26 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 26);

    /* attack_roll = eff_level * (attack_bonus + 64) = 107 * 158 = 16906 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 16906);

    ASSERT_INT_EQ("attack_speed", stats.attack_speed, 4);
}

static void test_loadout_melee_piety(void) {
    printf("--- loadout: ghrazi rapier, piety, aggressive ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_GHRAZI_RAPIER;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout,
        ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_PIETY,
        99,
        FIGHT_STYLE_AGGRESSIVE,
        0,    /* spell_base_damage */
        &stats
    );

    /* piety: att_mult=1.20, str_mult=1.23.
       aggressive: +3 str, no att bonus. */
    /* ref: PlayerVsNPCCalc.ts — Piety factorAccuracy=[120,100], factorStrength=[123,100] */

    /* eff_att_level = floor(99 * 1.20) + 0 + 8 = 126 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 126);

    /* eff_str = floor(99 * 1.23) + 3 + 8 = 121 + 11 = 132 */
    /* max_hit = floor(0.5 + 132 * 153 / 640) = floor(0.5 + 31.55625) = 32 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 32);

    /* attack_roll = 126 * (94 + 64) = 126 * 158 = 19908 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 19908);
}

static void test_loadout_ranged_rigour(void) {
    printf("--- loadout: ACB + diamond bolts (e), rigour, rapid ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_ARMADYL_CROSSBOW;
    loadout[GEAR_SLOT_AMMO] = ITEM_DIAMOND_BOLTS_E;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout,
        ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_RIGOUR,
        99,
        FIGHT_STYLE_RAPID,
        0,    /* spell_base_damage */
        &stats
    );

    /* rigour: att_mult=1.20, str_mult=1.23. rapid stance: 0 level bonus, -1 attack_speed. */
    /* ref: Prayer.ts — Rigour factorAccuracy=[120,100], factorStrength=[123,100] */

    /* attack_bonus = ACB.attack_ranged(100) + bolts.attack_ranged(0) = 100 */
    ASSERT_INT_EQ("attack_bonus", stats.attack_bonus, 100);

    /* strength_bonus = ACB.ranged_strength(0) + bolts.ranged_strength(105) = 105 */
    ASSERT_INT_EQ("strength_bonus", stats.strength_bonus, 105);

    /* eff_att = floor(99 * 1.20) + 0 + 8 = 118 + 8 = 126 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 126);

    /* eff_str = floor(99 * 1.23) + 0 + 8 = 121 + 8 = 129 */
    /* max_hit = floor(0.5 + 129 * (105 + 64) / 640) = floor(0.5 + 129*169/640) */
    /* = floor(0.5 + 34.064...) = floor(34.564) = 34 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 34);

    /* attack_roll = 126 * (100 + 64) = 126 * 164 = 20664 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 20664);
}

static void test_loadout_magic_augury(void) {
    printf("--- loadout: kodai wand, augury, autocast barrage ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout,
        ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_AUGURY,
        99,
        FIGHT_STYLE_AUTOCAST,
        30,   /* spell_base_damage (ice barrage = 30) */
        &stats
    );

    /* augury: att_mult=1.25, magic_dmg_mult=1.04 */
    /* ref: Prayer.ts — Augury factorAccuracy=[125,100], magicDamageBonus=40 (4%) */

    /* attack_bonus = kodai.attack_magic(28) = 28 */
    ASSERT_INT_EQ("attack_bonus", stats.attack_bonus, 28);

    /* strength_bonus = kodai.magic_damage(15) = 15 */
    ASSERT_INT_EQ("strength_bonus", stats.strength_bonus, 15);

    /* eff_level = floor(99 * 1.25) + 9 = 123 + 9 = 132 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 132);

    /* max_hit = floor(30 * (1.0 + 15/100.0) * 1.04) = floor(30 * 1.15 * 1.04) */
    /* = floor(35.88) = 35 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 35);

    /* attack_roll = 132 * (28 + 64) = 132 * 92 = 12144 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 12144);
}

static void test_loadout_magic_no_prayer(void) {
    printf("--- loadout: kodai wand, no prayer, autocast barrage ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout,
        ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE,
        99,
        FIGHT_STYLE_AUTOCAST,
        30,   /* ice barrage */
        &stats
    );

    /* eff_level = floor(99*1.0) + 9 = 108 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 108);

    /* max_hit = floor(30 * (1.0 + 15/100.0) * 1.0) = floor(30 * 1.15) = floor(34.5) = 34 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 34);
}

/* ======================================================================== */
/* test: loadout with full gear (multi-slot)                                 */
/*                                                                           */
/* verifies that stats from multiple gear slots sum correctly.               */
/* ======================================================================== */

static void test_loadout_full_ranged(void) {
    printf("--- loadout: tbow + masori + anguish + vambs, rigour ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_TWISTED_BOW;
    loadout[GEAR_SLOT_HEAD] = ITEM_MASORI_MASK_F;
    loadout[GEAR_SLOT_BODY] = ITEM_MASORI_BODY_F;
    loadout[GEAR_SLOT_LEGS] = ITEM_MASORI_CHAPS_F;
    loadout[GEAR_SLOT_NECK] = ITEM_NECKLACE_OF_ANGUISH;
    loadout[GEAR_SLOT_CAPE] = ITEM_DIZANAS_QUIVER;
    loadout[GEAR_SLOT_HANDS] = ITEM_ZARYTE_VAMBRACES;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout,
        ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_RIGOUR,
        99,
        FIGHT_STYLE_RAPID,
        0,
        &stats
    );

    /* sum attack_ranged: tbow(70) + mask(12) + body(43) + chaps(27) + anguish(15) + quiver(18) + vambs(18) = 203 */
    ASSERT_INT_EQ("attack_bonus", stats.attack_bonus, 203);

    /* sum ranged_strength: tbow(20) + mask(2) + body(4) + chaps(2) + anguish(5) + quiver(3) + vambs(2) = 38 */
    ASSERT_INT_EQ("strength_bonus", stats.strength_bonus, 38);

    /* eff_att = floor(99 * 1.20) + 0 + 8 = 126 */
    ASSERT_INT_EQ("eff_level", stats.eff_level, 126);

    /* eff_str = floor(99 * 1.23) + 0 + 8 = 129 */
    /* max_hit = floor(0.5 + 129 * (38 + 64) / 640) = floor(0.5 + 129*102/640) */
    /* = floor(0.5 + 13158/640) = floor(0.5 + 20.559375) = floor(21.059375) = 21 */
    ASSERT_INT_EQ("max_hit", stats.max_hit, 21);

    /* attack_roll = 126 * (203 + 64) = 126 * 267 = 33642 */
    int att_roll = stats.eff_level * (stats.attack_bonus + 64);
    ASSERT_INT_EQ("attack_roll", att_roll, 33642);
}

/* ======================================================================== */
/* test: encounter_update_loadout_level (brew drain / boost recomputation)   */
/*                                                                           */
/* verifies that eff_level and max_hit update correctly after stat changes.  */
/* ======================================================================== */

static void test_update_loadout_level(void) {
    printf("--- encounter_update_loadout_level ---\n");

    /* set up a melee loadout with piety */
    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_GHRAZI_RAPIER;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_PIETY,
        99, FIGHT_STYLE_AGGRESSIVE, 0, &stats);

    /* base (aggressive: +3 str, +0 att): eff_att=126, max=32 */
    ASSERT_INT_EQ("base eff", stats.eff_level, 126);
    ASSERT_INT_EQ("base max", stats.max_hit, 32);

    /* simulate brew drain: att drops to 90, str drops to 90 */
    encounter_update_loadout_level(&stats, OFFENSIVE_PRAYER_PIETY, 90, 90);

    /* eff_att = floor(90 * 1.20) + 0 + 8 = 108 + 8 = 116 */
    ASSERT_INT_EQ("drained eff", stats.eff_level, 116);

    /* eff_str = floor(90 * 1.23) + 3 + 8 = 110 + 11 = 121 */
    /* max_hit = floor(0.5 + 121 * 153 / 640) = floor(0.5 + 18513/640) = floor(0.5 + 28.926...) */
    /* = floor(29.426) = 29 */
    ASSERT_INT_EQ("drained max", stats.max_hit, 29);

    /* restore back to 99 */
    encounter_update_loadout_level(&stats, OFFENSIVE_PRAYER_PIETY, 99, 99);
    ASSERT_INT_EQ("restored eff", stats.eff_level, 126);
    ASSERT_INT_EQ("restored max", stats.max_hit, 32);

    /* magic loadout: max_hit doesn't depend on level (spell-based) */
    clear_loadout(loadout);
    loadout[GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_AUGURY,
        99, FIGHT_STYLE_AUTOCAST, 30, &stats);

    ASSERT_INT_EQ("magic base eff", stats.eff_level, 132);
    ASSERT_INT_EQ("magic base max", stats.max_hit, 35);

    /* drain magic to 80: eff changes, max_hit stays (spell-based) */
    encounter_update_loadout_level(&stats, OFFENSIVE_PRAYER_AUGURY, 80, 80);
    /* eff = floor(80 * 1.25) + 0 + 9 = 100 + 9 = 109 */
    ASSERT_INT_EQ("magic drained eff", stats.eff_level, 109);
    /* max_hit still = floor(30 * 1.15 * 1.04) = 35 */
    ASSERT_INT_EQ("magic drained max", stats.max_hit, 35);
}

/* ======================================================================== */
/* test: full combat scenario (NPC attacks player)                           */
/*                                                                           */
/* combines NPC attack roll + player defence roll + hit chance into one      */
/* end-to-end check. uses realistic inferno-style stats.                     */
/* ======================================================================== */

static void test_full_npc_attack_scenario(void) {
    printf("--- full NPC attack scenario ---\n");

    /* scenario: Jad (melee), att_level=480, att_bonus=0 */
    /* NPC attack roll = (480 + 9) * (0 + 64) = 489 * 64 = 31296 */
    int npc_att_roll = osrs_npc_attack_roll(480, 0);
    ASSERT_INT_EQ("jad att roll", npc_att_roll, 31296);

    /* player in justiciar: def=99, bonus=stab=200 (approx), vs melee */
    int player_def_roll = osrs_player_def_roll_vs_npc(99, 99, 200, 1 /* melee */);
    /* (99+8)*(200+64) = 107*264 = 28248 */
    ASSERT_INT_EQ("player def roll", player_def_roll, 28248);

    /* hit chance: att=31296 > def=28248 */
    /* 1 - (28248+2)/(2*(31296+1)) = 1 - 28250/62594 = 1 - 0.45131... = 0.54869... */
    float chance = osrs_hit_chance(npc_att_roll, player_def_roll);
    float expected = 1.0f - 28250.0f / 62594.0f;
    ASSERT_FLOAT_NEAR("jad hit chance", chance, expected, 1e-3f);

    /* Jad melee max hit: str=480, bonus=0 */
    /* ((480+8)*(0+64)+320)/640 = (488*64+320)/640 = (31232+320)/640 = 31552/640 = 49 */
    int jad_max = osrs_npc_melee_max_hit(480, 0);
    ASSERT_INT_EQ("jad melee max", jad_max, 49);
}

/* ======================================================================== */
/* test: RNG sanity checks                                                   */
/*                                                                           */
/* verify encounter_xorshift, encounter_rand_int, encounter_rand_float      */
/* produce values in expected ranges and aren't degenerate.                  */
/* ======================================================================== */

static void test_rng(void) {
    printf("--- RNG sanity ---\n");

    uint32_t state = 1;

    /* xorshift should produce non-zero non-degenerate values */
    uint32_t v1 = encounter_xorshift(&state);
    uint32_t v2 = encounter_xorshift(&state);
    ASSERT_INT_EQ("xor not equal", v1 != v2, 1);
    ASSERT_INT_EQ("xor nonzero 1", v1 != 0, 1);
    ASSERT_INT_EQ("xor nonzero 2", v2 != 0, 1);

    /* rand_int should be in [0, max) */
    state = 42;
    int in_range = 1;
    for (int i = 0; i < 10000; i++) {
        int r = encounter_rand_int(&state, 10);
        if (r < 0 || r >= 10) { in_range = 0; break; }
    }
    ASSERT_INT_EQ("rand_int [0,10)", in_range, 1);

    /* rand_int with max=1 should always return 0 */
    state = 123;
    int all_zero = 1;
    for (int i = 0; i < 100; i++) {
        if (encounter_rand_int(&state, 1) != 0) { all_zero = 0; break; }
    }
    ASSERT_INT_EQ("rand_int max=1 always 0", all_zero, 1);

    /* rand_int with max=0 should return 0 */
    ASSERT_INT_EQ("rand_int max=0", encounter_rand_int(&state, 0), 0);

    /* rand_float should be in [0, 1) */
    state = 999;
    int float_ok = 1;
    for (int i = 0; i < 10000; i++) {
        float f = encounter_rand_float(&state);
        if (f < 0.0f || f >= 1.0f) { float_ok = 0; break; }
    }
    ASSERT_INT_EQ("rand_float [0,1)", float_ok, 1);
}

/* ======================================================================== */
/* test: barrage AoE resolve                                                 */
/*                                                                           */
/* verify primary target always rolled, AoE only within 1 tile,             */
/* damage bounds, and freeze application.                                    */
/* ======================================================================== */

static void test_barrage_resolve(void) {
    printf("--- osrs_barrage_resolve ---\n");

    uint32_t rng = 77;
    int att_roll = 30000;
    int max_hit = 30;

    /* single target */
    BarrageTarget targets[3];
    memset(targets, 0, sizeof(targets));
    targets[0].active = 1;
    targets[0].x = 5; targets[0].y = 5;
    targets[0].def_level = 100;
    targets[0].magic_def_bonus = 50;

    BarrageResult res = osrs_barrage_resolve(targets, 1, att_roll, max_hit, &rng, 0, 0);
    ASSERT_INT_EQ("single num_hits", res.num_hits, 1);
    ASSERT_INT_EQ("single dmg range", res.total_damage >= 0 && res.total_damage <= 30, 1);

    /* two targets: one in range, one out of range */
    rng = 88;
    memset(targets, 0, sizeof(targets));
    targets[0].active = 1;
    targets[0].x = 5; targets[0].y = 5;
    targets[0].def_level = 50; targets[0].magic_def_bonus = 20;

    targets[1].active = 1;
    targets[1].x = 6; targets[1].y = 6;  /* within 1 tile of primary */
    targets[1].def_level = 50; targets[1].magic_def_bonus = 20;

    targets[2].active = 1;
    targets[2].x = 10; targets[2].y = 10;  /* far away, NOT in AoE */
    targets[2].def_level = 50; targets[2].magic_def_bonus = 20;

    res = osrs_barrage_resolve(targets, 3, att_roll, max_hit, &rng, 0, 0);
    /* should roll primary + 1 in-range, skip the far one */
    ASSERT_INT_EQ("aoe num_hits", res.num_hits, 2);
    /* far target should not have been hit */
    ASSERT_INT_EQ("far target untouched", targets[2].hit, 0);

    /* freeze test: ice barrage (spell_type=1) should set frozen_ticks */
    rng = 99;
    int frozen = 0;
    memset(targets, 0, sizeof(targets));
    targets[0].active = 1;
    targets[0].x = 5; targets[0].y = 5;
    targets[0].def_level = 1;  /* very low def = near-guaranteed hit */
    targets[0].magic_def_bonus = 0;
    targets[0].frozen_ticks = &frozen;

    /* run a few times to get at least one hit */
    int freeze_applied = 0;
    for (int i = 0; i < 100 && !freeze_applied; i++) {
        frozen = 0;
        osrs_barrage_resolve(targets, 1, 50000, max_hit, &rng, 1 /* ICE */, 0);
        if (frozen == BARRAGE_FREEZE_TICKS) freeze_applied = 1;
    }
    ASSERT_INT_EQ("ice freeze applied", freeze_applied, 1);
    ASSERT_INT_EQ("freeze duration", BARRAGE_FREEZE_TICKS, 32);
}

/* ======================================================================== */
/* test: defence bonus sum verification                                      */
/*                                                                           */
/* verifies that encounter_compute_loadout_stats correctly sums defence      */
/* bonuses across all gear slots (needed for player_def_roll calculations).  */
/* ======================================================================== */

static void test_loadout_def_bonuses(void) {
    printf("--- loadout defence bonus sums ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);

    /* justiciar set: faceguard + chestguard + legguards */
    loadout[GEAR_SLOT_HEAD] = ITEM_JUSTICIAR_FACEGUARD;
    loadout[GEAR_SLOT_BODY] = ITEM_JUSTICIAR_CHESTGUARD;
    loadout[GEAR_SLOT_LEGS] = ITEM_JUSTICIAR_LEGGUARDS;

    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_NONE,
        99, FIGHT_STYLE_ACCURATE, 0, &stats);

    /* verify defence bonuses sum correctly from ITEM_DATABASE.
       exact values depend on item stats — verify against DB directly */
    int exp_stab = ITEM_DATABASE[ITEM_JUSTICIAR_FACEGUARD].defence_stab
                 + ITEM_DATABASE[ITEM_JUSTICIAR_CHESTGUARD].defence_stab
                 + ITEM_DATABASE[ITEM_JUSTICIAR_LEGGUARDS].defence_stab;
    int exp_slash = ITEM_DATABASE[ITEM_JUSTICIAR_FACEGUARD].defence_slash
                  + ITEM_DATABASE[ITEM_JUSTICIAR_CHESTGUARD].defence_slash
                  + ITEM_DATABASE[ITEM_JUSTICIAR_LEGGUARDS].defence_slash;
    int exp_crush = ITEM_DATABASE[ITEM_JUSTICIAR_FACEGUARD].defence_crush
                  + ITEM_DATABASE[ITEM_JUSTICIAR_CHESTGUARD].defence_crush
                  + ITEM_DATABASE[ITEM_JUSTICIAR_LEGGUARDS].defence_crush;
    int exp_magic = ITEM_DATABASE[ITEM_JUSTICIAR_FACEGUARD].defence_magic
                  + ITEM_DATABASE[ITEM_JUSTICIAR_CHESTGUARD].defence_magic
                  + ITEM_DATABASE[ITEM_JUSTICIAR_LEGGUARDS].defence_magic;
    int exp_ranged = ITEM_DATABASE[ITEM_JUSTICIAR_FACEGUARD].defence_ranged
                   + ITEM_DATABASE[ITEM_JUSTICIAR_CHESTGUARD].defence_ranged
                   + ITEM_DATABASE[ITEM_JUSTICIAR_LEGGUARDS].defence_ranged;

    ASSERT_INT_EQ("def_stab", stats.def_stab, exp_stab);
    ASSERT_INT_EQ("def_slash", stats.def_slash, exp_slash);
    ASSERT_INT_EQ("def_crush", stats.def_crush, exp_crush);
    ASSERT_INT_EQ("def_magic", stats.def_magic, exp_magic);
    ASSERT_INT_EQ("def_ranged", stats.def_ranged, exp_ranged);
}

/* ======================================================================== */
/* test: edge cases for level 1 and no gear                                  */
/*                                                                           */
/* verifies formulas don't break at minimum values.                         */
/* ======================================================================== */

static void test_edge_cases(void) {
    printf("--- edge cases: level 1, no gear ---\n");

    uint8_t loadout[NUM_GEAR_SLOTS];
    clear_loadout(loadout);

    /* melee, level 1, no gear, no prayer, accurate stance (+3 att) */
    EncounterLoadoutStats stats;
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MELEE, OFFENSIVE_PRAYER_NONE,
        1, FIGHT_STYLE_ACCURATE, 0, &stats);

    /* eff = floor(1*1.0) + 3 + 8 = 12 (accurate +3 att) */
    ASSERT_INT_EQ("lv1 melee eff", stats.eff_level, 12);

    /* eff_str = 1 + 0 + 8 = 9 (accurate gives 0 to str), str_bonus = 0 */
    /* max_hit = floor(0.5 + 9 * (0+64) / 640) = floor(0.5 + 576/640) = floor(0.5 + 0.9) = 1 */
    ASSERT_INT_EQ("lv1 melee max", stats.max_hit, 1);

    /* attack_bonus = 0 (no weapon) */
    ASSERT_INT_EQ("lv1 melee att_bonus", stats.attack_bonus, 0);

    /* magic, level 1, no gear, barrage, autocast (no invisible bonus) */
    encounter_compute_loadout_stats(
        loadout, ATTACK_STYLE_MAGIC, OFFENSIVE_PRAYER_NONE,
        1, FIGHT_STYLE_AUTOCAST, 30, &stats);

    /* eff = floor(1*1.0) + 0 + 9 = 10 */
    ASSERT_INT_EQ("lv1 magic eff", stats.eff_level, 10);

    /* max_hit = floor(30 * (1.0 + 0/100.0) * 1.0) = 30 */
    ASSERT_INT_EQ("lv1 magic max", stats.max_hit, 30);
}


int main(void) {
    printf("=== combat math tests (cross-referenced with osrs-dps-calc) ===\n\n");

    /* pure math (osrs_combat_shared.h) */
    test_hit_chance();
    test_npc_melee_max_hit();
    test_npc_ranged_max_hit();
    test_npc_magic_max_hit();
    test_npc_attack_roll();
    test_npc_max_hit_dispatch();
    test_player_def_roll();
    test_player_def_bonus();
    test_prayer_correct();
    test_hit_delays();
    test_dist_to_npc();
    test_tbow_multipliers();
    test_blowpipe_spec();
    test_rng();
    test_barrage_resolve();

    /* loadout stat computation (osrs_encounter.h) */
    test_loadout_melee_no_prayer();
    test_loadout_melee_piety();
    test_loadout_ranged_rigour();
    test_loadout_magic_augury();
    test_loadout_magic_no_prayer();
    test_loadout_full_ranged();
    test_update_loadout_level();
    test_loadout_def_bonuses();
    test_edge_cases();

    /* end-to-end */
    test_full_npc_attack_scenario();

    printf("\n=== results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(", %d FAILED", tests_failed);
    }
    printf(" ===\n");

    return tests_failed > 0 ? 1 : 0;
}
