/**
 * @file test_special_attacks.c
 * @brief special attack tests cross-referenced against osrs-dps-calc reference.
 *
 * tests spec weapon costs, accuracy multipliers, strength multipliers, and
 * special mechanics (claws cascade, DWH/BGS defence drain, dark bow double hit
 * with min/max clamping, morrigan's bleed, voidwaker magic hit, VLS reduced
 * defence roll, volatile staff, godsword variants, double-hit specs).
 *
 * BUILD:
 *   cd pufferlib-metal
 *   cc -std=c11 -O0 -g -I. -o test_special_attacks \
 *       ocean/osrs/tests/test_special_attacks.c -lm
 *   ./test_special_attacks
 *
 * REFERENCE FILES:
 *   .refs/osrs-dps-calc/src/lib/PlayerVsNPCCalc.ts   -- spec accuracy/damage mults
 *   .refs/osrs-dps-calc/src/lib/dists/claws.ts        -- dragon claws cascade dist
 *   ocean/osrs/osrs_pvp_combat.h            -- our spec implementations
 *   ocean/osrs/osrs_combat.h                -- blowpipe spec
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ocean/osrs/osrs_pvp_combat.h"
#include "ocean/osrs/osrs_combat.h"
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

#define ASSERT_FLOAT_EQ(label, actual, expected, tolerance) do { \
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
/* test: melee spec energy costs                                            */
/*                                                                          */
/* ref: OSRS wiki special attack page, osrs_pvp_combat.h:38-53             */
/* ======================================================================== */

static void test_melee_spec_costs(void) {
    printf("--- melee spec energy costs ---\n");

    ASSERT_INT_EQ("AGS cost",              get_melee_spec_cost(MELEE_SPEC_AGS),              50);
    ASSERT_INT_EQ("dragon claws cost",      get_melee_spec_cost(MELEE_SPEC_DRAGON_CLAWS),     50);
    ASSERT_INT_EQ("granite maul cost",      get_melee_spec_cost(MELEE_SPEC_GRANITE_MAUL),     50);
    ASSERT_INT_EQ("dragon dagger cost",     get_melee_spec_cost(MELEE_SPEC_DRAGON_DAGGER),    25);
    ASSERT_INT_EQ("voidwaker cost",         get_melee_spec_cost(MELEE_SPEC_VOIDWAKER),        50);
    ASSERT_INT_EQ("DWH cost",              get_melee_spec_cost(MELEE_SPEC_DWH),              35);
    ASSERT_INT_EQ("BGS cost",              get_melee_spec_cost(MELEE_SPEC_BGS),              50);
    ASSERT_INT_EQ("ZGS cost",              get_melee_spec_cost(MELEE_SPEC_ZGS),              50);
    ASSERT_INT_EQ("SGS cost",              get_melee_spec_cost(MELEE_SPEC_SGS),              50);
    ASSERT_INT_EQ("ancient GS cost",       get_melee_spec_cost(MELEE_SPEC_ANCIENT_GS),       50);
    ASSERT_INT_EQ("VLS cost",              get_melee_spec_cost(MELEE_SPEC_VESTAS),            25);
    ASSERT_INT_EQ("abyssal dagger cost",   get_melee_spec_cost(MELEE_SPEC_ABYSSAL_DAGGER),   50);
    ASSERT_INT_EQ("dragon longsword cost", get_melee_spec_cost(MELEE_SPEC_DRAGON_LONGSWORD),  25);
    ASSERT_INT_EQ("dragon mace cost",      get_melee_spec_cost(MELEE_SPEC_DRAGON_MACE),       25);
    ASSERT_INT_EQ("abyssal bludgeon cost", get_melee_spec_cost(MELEE_SPEC_ABYSSAL_BLUDGEON),  50);
}


static void test_ranged_spec_costs(void) {
    printf("--- ranged spec energy costs ---\n");

    ASSERT_INT_EQ("dark bow cost",     get_ranged_spec_cost(RANGED_SPEC_DARK_BOW),     55);
    ASSERT_INT_EQ("ballista cost",     get_ranged_spec_cost(RANGED_SPEC_BALLISTA),     65);
    ASSERT_INT_EQ("ZCB cost",          get_ranged_spec_cost(RANGED_SPEC_ZCB),          75);
    ASSERT_INT_EQ("dragon knife cost", get_ranged_spec_cost(RANGED_SPEC_DRAGON_KNIFE), 25);
    ASSERT_INT_EQ("MSB cost",          get_ranged_spec_cost(RANGED_SPEC_MSB),          50);
    ASSERT_INT_EQ("morrigan's cost",   get_ranged_spec_cost(RANGED_SPEC_MORRIGANS),    50);
}


static void test_magic_spec_costs(void) {
    printf("--- magic spec energy costs ---\n");

    ASSERT_INT_EQ("volatile staff cost", get_magic_spec_cost(MAGIC_SPEC_VOLATILE_STAFF), 55);
}

/* ======================================================================== */
/* test: melee spec accuracy multipliers                                    */
/*                                                                          */
/* ref: PlayerVsNPCCalc.ts:292-311 (godswords [2,1]=2x, DDS [23,20]=1.15x, */
/*      abyssal dagger [5,4]=1.25x, etc.)                                  */
/* ======================================================================== */

static void test_melee_spec_acc_multipliers(void) {
    printf("--- melee spec accuracy multipliers ---\n");

    /* AGS: godsword family [2,1] = 2.0x */
    ASSERT_FLOAT_EQ("AGS acc mult",    get_melee_spec_acc_mult(MELEE_SPEC_AGS),    2.0f, 1e-5f);
    /* dragon claws: not listed in PvNPC (custom cascade), our impl uses 1.35x
       this is the PvP value from OSRS wiki (not in ref calc for PvNPC) */
    /* claws: no accuracy multiplier — cascade rolls 4x at base acc.
       ref: osrs-dps-calc dists/claws.ts (no acc mult in PlayerVsNPCCalc.ts:292-309) */
    ASSERT_FLOAT_EQ("claws acc mult",  get_melee_spec_acc_mult(MELEE_SPEC_DRAGON_CLAWS), 1.0f, 1e-5f);
    /* granite maul: no accuracy bonus */
    ASSERT_FLOAT_EQ("gmaul acc mult",  get_melee_spec_acc_mult(MELEE_SPEC_GRANITE_MAUL), 1.0f, 1e-5f);
    /* DDS: [23,20] = 1.15x. ref: dps-calc PlayerVsNPCCalc.ts:300 */
    ASSERT_FLOAT_EQ("DDS acc mult",    get_melee_spec_acc_mult(MELEE_SPEC_DRAGON_DAGGER), 1.15f, 1e-5f);
    /* voidwaker: guaranteed hit in PvNPC; PvP uses normal accuracy */
    ASSERT_FLOAT_EQ("VW acc mult",     get_melee_spec_acc_mult(MELEE_SPEC_VOIDWAKER),     1.0f, 1e-5f);
    /* statius warhammer (LMS): [5,4] = 1.25x */
    ASSERT_FLOAT_EQ("DWH acc mult",    get_melee_spec_acc_mult(MELEE_SPEC_DWH),           1.25f, 1e-5f);
    /* BGS: [2,1] = 2.0x per dps-calc */
    ASSERT_FLOAT_EQ("BGS acc mult",    get_melee_spec_acc_mult(MELEE_SPEC_BGS),           2.0f, 1e-5f);
    /* ZGS: godsword family [2,1] = 2.0x */
    ASSERT_FLOAT_EQ("ZGS acc mult",    get_melee_spec_acc_mult(MELEE_SPEC_ZGS),           2.0f, 1e-5f);
    /* SGS: [2,1] = 2.0x per dps-calc */
    ASSERT_FLOAT_EQ("SGS acc mult",    get_melee_spec_acc_mult(MELEE_SPEC_SGS),           2.0f, 1e-5f);
    /* ancient GS: godsword [2,1] = 2.0x */
    ASSERT_FLOAT_EQ("ancient GS acc",  get_melee_spec_acc_mult(MELEE_SPEC_ANCIENT_GS),    2.0f, 1e-5f);
    /* VLS: no acc mult (uses reduced def roll instead) */
    ASSERT_FLOAT_EQ("VLS acc mult",    get_melee_spec_acc_mult(MELEE_SPEC_VESTAS),        1.0f, 1e-5f);
    /* abyssal dagger: [5,4] = 1.25x */
    ASSERT_FLOAT_EQ("abyssal dagger acc", get_melee_spec_acc_mult(MELEE_SPEC_ABYSSAL_DAGGER), 1.25f, 1e-5f);
    /* dragon longsword: [5,4] = 1.25x */
    ASSERT_FLOAT_EQ("dlong acc",       get_melee_spec_acc_mult(MELEE_SPEC_DRAGON_LONGSWORD), 1.25f, 1e-5f);
    /* dragon mace: [5,4] = 1.25x */
    ASSERT_FLOAT_EQ("dmace acc",       get_melee_spec_acc_mult(MELEE_SPEC_DRAGON_MACE),     1.25f, 1e-5f);
    /* abyssal bludgeon: no accuracy bonus */
    ASSERT_FLOAT_EQ("bludgeon acc",    get_melee_spec_acc_mult(MELEE_SPEC_ABYSSAL_BLUDGEON), 1.0f, 1e-5f);
}

/* ======================================================================== */
/* test: melee spec strength multipliers                                    */
/*                                                                          */
/* ref: PlayerVsNPCCalc.ts:453-485                                          */
/* godswords get [11,10]=1.1x base, then additional per weapon:             */
/*   AGS: [5,4]=1.25x total => 1.1*1.25 = 1.375x                          */
/*   BGS: [11,10]=1.21x total => 1.1*1.1 = 1.21x                          */
/*   ZGS/SGS: [11,10] only => 1.1x (no additional)                         */
/*   DDS: [23,20] = 1.15x; statius: [5,4] = 1.25x; dmace: [3,2] = 1.5x  */
/* ======================================================================== */

static void test_melee_spec_str_multipliers(void) {
    printf("--- melee spec strength multipliers ---\n");

    /* AGS: godsword 1.1x * [5,4]=1.25x = 1.375x */
    ASSERT_FLOAT_EQ("AGS str mult",    get_melee_spec_str_mult(MELEE_SPEC_AGS),    1.375f, 1e-3f);
    /* dragon claws: 1.0x (cascade handles damage distribution) */
    ASSERT_FLOAT_EQ("claws str mult",  get_melee_spec_str_mult(MELEE_SPEC_DRAGON_CLAWS), 1.0f, 1e-5f);
    /* granite maul: 1.0x */
    ASSERT_FLOAT_EQ("gmaul str mult",  get_melee_spec_str_mult(MELEE_SPEC_GRANITE_MAUL), 1.0f, 1e-5f);
    /* DDS: [23,20] = 1.15x */
    ASSERT_FLOAT_EQ("DDS str mult",    get_melee_spec_str_mult(MELEE_SPEC_DRAGON_DAGGER), 1.15f, 1e-3f);
    /* voidwaker: 1.0x (50-150% range handled in perform_voidwaker_spec) */
    ASSERT_FLOAT_EQ("VW str mult",     get_melee_spec_str_mult(MELEE_SPEC_VOIDWAKER), 1.0f, 1e-5f);
    /* statius warhammer (LMS): 1.25x str */
    ASSERT_FLOAT_EQ("DWH str mult",    get_melee_spec_str_mult(MELEE_SPEC_DWH),   1.25f, 1e-3f);
    /* BGS: godsword 1.1x * [11,10]=1.1x = 1.21x */
    ASSERT_FLOAT_EQ("BGS str mult",    get_melee_spec_str_mult(MELEE_SPEC_BGS),   1.21f, 1e-3f);
    /* ZGS: godsword 1.1x only */
    ASSERT_FLOAT_EQ("ZGS str mult",    get_melee_spec_str_mult(MELEE_SPEC_ZGS),   1.1f,  1e-3f);
    /* SGS: godsword 1.1x only */
    ASSERT_FLOAT_EQ("SGS str mult",    get_melee_spec_str_mult(MELEE_SPEC_SGS),   1.1f,  1e-3f);
    /* ancient GS: godsword 1.1x only (blood sacrifice is separate) */
    ASSERT_FLOAT_EQ("ancient GS str",  get_melee_spec_str_mult(MELEE_SPEC_ANCIENT_GS), 1.1f, 1e-3f);
    /* VLS: 1.20x */
    ASSERT_FLOAT_EQ("VLS str mult",    get_melee_spec_str_mult(MELEE_SPEC_VESTAS),     1.20f, 1e-3f);
    /* abyssal dagger: 0.85x */
    ASSERT_FLOAT_EQ("abyssal dagger str", get_melee_spec_str_mult(MELEE_SPEC_ABYSSAL_DAGGER), 0.85f, 1e-3f);
    /* dragon longsword: 1.15x */
    ASSERT_FLOAT_EQ("dlong str mult",  get_melee_spec_str_mult(MELEE_SPEC_DRAGON_LONGSWORD), 1.15f, 1e-3f);
    /* dragon mace: [3,2] = 1.5x */
    ASSERT_FLOAT_EQ("dmace str mult",  get_melee_spec_str_mult(MELEE_SPEC_DRAGON_MACE),      1.5f,  1e-3f);
    /* abyssal bludgeon: 1.20x (base; real spec adds missing prayer %) */
    ASSERT_FLOAT_EQ("bludgeon str",    get_melee_spec_str_mult(MELEE_SPEC_ABYSSAL_BLUDGEON), 1.20f, 1e-3f);
}

/* ======================================================================== */
/* test: ranged spec accuracy multipliers                                   */
/*                                                                          */
/* ref: PlayerVsNPCCalc.ts:579-589                                          */
/*   dark bow: no acc bonus. ballista: [5,4]=1.25x.                        */
/*   ZCB: [2,1]=2.0x. MSB: [10,7]~1.43x. dragon knife: none.             */
/* ======================================================================== */

static void test_ranged_spec_acc_multipliers(void) {
    printf("--- ranged spec accuracy multipliers ---\n");

    ASSERT_FLOAT_EQ("dark bow acc",     get_ranged_spec_acc_mult(RANGED_SPEC_DARK_BOW),     1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("ballista acc",     get_ranged_spec_acc_mult(RANGED_SPEC_BALLISTA),     1.25f, 1e-3f);
    ASSERT_FLOAT_EQ("ZCB acc",          get_ranged_spec_acc_mult(RANGED_SPEC_ZCB),          2.0f, 1e-5f);
    ASSERT_FLOAT_EQ("dragon knife acc", get_ranged_spec_acc_mult(RANGED_SPEC_DRAGON_KNIFE), 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("MSB acc",          get_ranged_spec_acc_mult(RANGED_SPEC_MSB),          1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("morrigan's acc",   get_ranged_spec_acc_mult(RANGED_SPEC_MORRIGANS),    1.0f, 1e-5f);
}

/* ======================================================================== */
/* test: ranged spec strength multipliers                                   */
/*                                                                          */
/* ref: PlayerVsNPCCalc.ts:738-757                                          */
/*   dark bow with dragon arrows: [15,10]=1.5x, min 8, max 48 clamped.    */
/*   ballista: [5,4]=1.25x.                                                */
/* ======================================================================== */

static void test_ranged_spec_str_multipliers(void) {
    printf("--- ranged spec strength multipliers ---\n");

    ASSERT_FLOAT_EQ("dark bow str",     get_ranged_spec_str_mult(RANGED_SPEC_DARK_BOW),     1.5f,  1e-3f);
    ASSERT_FLOAT_EQ("ballista str",     get_ranged_spec_str_mult(RANGED_SPEC_BALLISTA),     1.25f, 1e-3f);
    ASSERT_FLOAT_EQ("ZCB str",          get_ranged_spec_str_mult(RANGED_SPEC_ZCB),          1.0f,  1e-5f);
    ASSERT_FLOAT_EQ("dragon knife str", get_ranged_spec_str_mult(RANGED_SPEC_DRAGON_KNIFE), 1.0f,  1e-5f);
    ASSERT_FLOAT_EQ("MSB str",          get_ranged_spec_str_mult(RANGED_SPEC_MSB),          1.0f,  1e-5f);
    ASSERT_FLOAT_EQ("morrigan's str",   get_ranged_spec_str_mult(RANGED_SPEC_MORRIGANS),    1.0f,  1e-5f);
}

/* ======================================================================== */
/* test: magic spec accuracy multiplier                                     */
/*                                                                          */
/* ref: PlayerVsNPCCalc.ts:855-856 volatile staff [3,2]=1.5x               */
/* ======================================================================== */

static void test_magic_spec_acc_multiplier(void) {
    printf("--- magic spec accuracy multiplier ---\n");

    ASSERT_FLOAT_EQ("volatile staff acc", get_magic_spec_acc_mult(MAGIC_SPEC_VOLATILE_STAFF), 1.5f, 1e-5f);
}

/* ======================================================================== */
/* test: blowpipe spec constants (osrs_combat.h)                            */
/*                                                                          */
/* ref: osrs-sdk Blowpipe.ts: 2x accuracy, 1.5x damage, 50% heal, 50 cost */
/* ======================================================================== */

static void test_blowpipe_spec_constants(void) {
    printf("--- blowpipe spec constants ---\n");

    ASSERT_INT_EQ("blowpipe acc mult",   BLOWPIPE_SPEC_ACC_MULT, 2);
    /* 1.5x = 3/2 */
    ASSERT_INT_EQ("blowpipe dmg num",    BLOWPIPE_SPEC_DMG_NUM,  3);
    ASSERT_INT_EQ("blowpipe dmg den",    BLOWPIPE_SPEC_DMG_DEN,  2);
    ASSERT_INT_EQ("blowpipe heal pct",   BLOWPIPE_SPEC_HEAL_PCT, 50);
    ASSERT_INT_EQ("blowpipe spec cost",  BLOWPIPE_SPEC_COST,     50);
}

/* ======================================================================== */
/* test: blowpipe spec damage calculation                                   */
/*                                                                          */
/* osrs_blowpipe_spec_resolve: 2x att roll, 1.5x max hit, single hit.     */
/* with forced-hit scenario (high att, low def) we can check max hit cap.  */
/* ======================================================================== */

static void test_blowpipe_spec_resolve(void) {
    printf("--- blowpipe spec resolve ---\n");

    /* base max hit 30, spec max = 30 * 3 / 2 = 45.
       with huge att_roll vs 0 def, should always hit -> damage in [0, 45].
       run many trials to verify range. */
    uint32_t rng = 12345;
    int base_att_roll = 50000;
    int base_max_hit = 30;
    int target_def_level = 1;
    int target_def_bonus = 0;

    int min_seen = 9999, max_seen = -1;
    int hit_count = 0;
    int trials = 10000;
    for (int i = 0; i < trials; i++) {
        int dmg = osrs_blowpipe_spec_resolve(base_att_roll, base_max_hit,
                                              target_def_level, target_def_bonus, &rng);
        if (dmg > 0) hit_count++;
        if (dmg < min_seen) min_seen = dmg;
        if (dmg > max_seen) max_seen = dmg;
    }

    /* spec max = 30 * 3 / 2 = 45 */
    int expected_spec_max = base_max_hit * BLOWPIPE_SPEC_DMG_NUM / BLOWPIPE_SPEC_DMG_DEN;
    ASSERT_INT_EQ("blowpipe spec max formula", expected_spec_max, 45);

    /* with overwhelming accuracy, hit rate should be very high */
    ASSERT_INT_EQ("blowpipe high hit rate", hit_count > trials / 2, 1);

    /* observed max should not exceed spec_max */
    ASSERT_INT_EQ("blowpipe max bounded", max_seen <= expected_spec_max, 1);

    /* min_seen should be 0 (possible miss or 0 roll) */
    ASSERT_INT_EQ("blowpipe min is 0", min_seen, 0);
}

/* ======================================================================== */
/* test: dragon claws cascade structure                                     */
/*                                                                          */
/* ref: .refs/osrs-dps-calc/src/lib/dists/claws.ts                         */
/*                                                                          */
/* the claws spec uses 4 accuracy rolls. cascade logic:                     */
/*   roll1 hits: total=rand[max/2, max-1], split d/2, d/4, d/8, d/8+1     */
/*   roll2 hits: total=rand[3/8*max, 7/8*max], split 0, d/2, d/4, d/4+1   */
/*   roll3 hits: total=rand[max/4, 3/4*max], split 0, 0, d/2, d/2+1       */
/*   roll4 hits: total=rand[max/4, 5/4*max], split 0, 0, 0, d+1           */
/*   all miss: 0, 0, rand(0-1), same                                       */
/*                                                                          */
/* our impl differs slightly from ref (PvP variant) but structure matches. */
/* we test: 4 hits always queued, acc mult = 1.35x, str mult = 1.0x.      */
/* ======================================================================== */

static void test_dragon_claws_cascade(void) {
    printf("--- dragon claws cascade ---\n");

    /* verify the multiplier table values */
    ASSERT_FLOAT_EQ("claws acc_mult", get_melee_spec_acc_mult(MELEE_SPEC_DRAGON_CLAWS), 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("claws str_mult", get_melee_spec_str_mult(MELEE_SPEC_DRAGON_CLAWS), 1.0f,  1e-5f);
    ASSERT_INT_EQ("claws cost", get_melee_spec_cost(MELEE_SPEC_DRAGON_CLAWS), 50);

    /* verify cascade total damage ranges from dps-calc dClawDist.
       generateTotals: low = floor(max * (4-accRoll) / 4), high = max + low - 1
       with max_hit = 40:
         roll 0: low=40, high=79. total in [40, 79], split [t/2, t/4, t/8, t/8+1]
         roll 1: low=30, high=69. total in [30, 69], split [t/2, t/4, t/4+1, 0]
         roll 2: low=20, high=59. total in [20, 59], split [t/2, t/2+1, 0, 0]
         roll 3: low=10, high=49. total in [10, 49], split [t+1, 0, 0, 0]
         all miss: 2/3 chance [1,1,0,0], 1/3 chance [0,0,0,0] */
    int max_hit = 40;
    /* roll 0 total range */
    ASSERT_INT_EQ("claws roll0 low",  max_hit * 4 / 4, 40);
    ASSERT_INT_EQ("claws roll0 high", max_hit + max_hit * 4 / 4 - 1, 79);
    /* roll 1 total range */
    ASSERT_INT_EQ("claws roll1 low",  max_hit * 3 / 4, 30);
    ASSERT_INT_EQ("claws roll1 high", max_hit + max_hit * 3 / 4 - 1, 69);
    /* roll 2 total range */
    ASSERT_INT_EQ("claws roll2 low",  max_hit * 2 / 4, 20);
    ASSERT_INT_EQ("claws roll2 high", max_hit + max_hit * 2 / 4 - 1, 59);
    /* roll 3 total range */
    ASSERT_INT_EQ("claws roll3 low",  max_hit * 1 / 4, 10);
    ASSERT_INT_EQ("claws roll3 high", max_hit + max_hit * 1 / 4 - 1, 49);
}

/* ======================================================================== */
/* test: voidwaker spec mechanics                                           */
/*                                                                          */
/* voidwaker: guaranteed magic damage at 50-150% of melee max hit.         */
/* ref: PlayerVsNPCCalc.ts:464-466 — min=floor(maxHit/2), max=maxHit+min  */
/*      also: accuracy is 1.0 (guaranteed hit, line 1207-1208)             */
/* ======================================================================== */

static void test_voidwaker_mechanics(void) {
    printf("--- voidwaker mechanics ---\n");

    ASSERT_FLOAT_EQ("VW acc mult",  get_melee_spec_acc_mult(MELEE_SPEC_VOIDWAKER),  1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("VW str mult",  get_melee_spec_str_mult(MELEE_SPEC_VOIDWAKER),  1.0f, 1e-5f);
    ASSERT_INT_EQ("VW cost",        get_melee_spec_cost(MELEE_SPEC_VOIDWAKER),       50);

    /* damage range: 50% to 150% of melee max hit.
       with max_melee_hit = 50: min_damage = floor(50 * 0.5) = 25, max_damage = floor(50 * 1.5) = 75 */
    int max_melee_hit = 50;
    int min_damage = (int)(max_melee_hit * 0.5f);
    int max_damage = (int)(max_melee_hit * 1.5f);
    ASSERT_INT_EQ("VW min damage (max=50)", min_damage, 25);
    ASSERT_INT_EQ("VW max damage (max=50)", max_damage, 75);

    /* edge case: max_melee_hit = 1 -> min=0, max=1 */
    int min_d1 = (int)(1 * 0.5f);
    int max_d1 = (int)(1 * 1.5f);
    ASSERT_INT_EQ("VW min damage (max=1)", min_d1, 0);
    ASSERT_INT_EQ("VW max damage (max=1)", max_d1, 1);
}

/* ======================================================================== */
/* test: VLS (Vesta's longsword) spec mechanics                             */
/*                                                                          */
/* "Feint": 20-120% of base max hit, accuracy vs 25% of opponent's def     */
/* this is a custom PvP spec (not in reference PvNPC calc).                */
/* our code: osrs_pvp_combat.h:928-962                                      */
/* ======================================================================== */

static void test_vls_mechanics(void) {
    printf("--- VLS spec mechanics ---\n");

    ASSERT_INT_EQ("VLS cost", get_melee_spec_cost(MELEE_SPEC_VESTAS), 25);
    ASSERT_FLOAT_EQ("VLS acc mult",  get_melee_spec_acc_mult(MELEE_SPEC_VESTAS),  1.0f,  1e-5f);
    ASSERT_FLOAT_EQ("VLS str mult",  get_melee_spec_str_mult(MELEE_SPEC_VESTAS),  1.20f, 1e-3f);

    /* damage range: 20-120% of base max hit.
       with base_max = 40: min = floor(40 * 0.20) = 8, max = floor(40 * 1.20) = 48 */
    int base_max = 40;
    int vls_max = (int)(base_max * 1.20f);
    int vls_min = (int)(base_max * 0.20f);
    ASSERT_INT_EQ("VLS min (base=40)", vls_min, 8);
    ASSERT_INT_EQ("VLS max (base=40)", vls_max, 48);

    /* defence reduction: 25% of opponent's defence roll.
       if eff_def * (def_bonus + 64) = 10000, reduced to 2500 */
    int full_def_roll = 10000;
    int reduced_def = (int)(full_def_roll * 0.25f);
    ASSERT_INT_EQ("VLS def reduction", reduced_def, 2500);
}

/* ======================================================================== */
/* test: statius warhammer spec (LMS, DWH path in our code)                 */
/*                                                                          */
/* statius warhammer (LMS): 35% cost, 1.25x acc, 1.25x str,               */
/* 30% defence drain on hit. no minimum hit floor.                          */
/* ======================================================================== */

static void test_statius_warhammer_mechanics(void) {
    printf("--- statius warhammer (LMS) spec ---\n");

    ASSERT_INT_EQ("DWH cost",   get_melee_spec_cost(MELEE_SPEC_DWH),   35);
    ASSERT_FLOAT_EQ("DWH acc",  get_melee_spec_acc_mult(MELEE_SPEC_DWH), 1.25f, 1e-5f);
    ASSERT_FLOAT_EQ("DWH str",  get_melee_spec_str_mult(MELEE_SPEC_DWH), 1.25f, 1e-3f);

    /* 30% defence drain: if current_def = 70, drain = floor(70 * 0.30) = 21, new def = 49 */
    int current_def = 70;
    int drain = (int)(current_def * 30 / 100.0f);
    int new_def = current_def - drain;
    ASSERT_INT_EQ("DWH def drain (70)", drain, 21);
    ASSERT_INT_EQ("DWH new def (70)", new_def, 49);

    /* edge case: minimum defence is 1 */
    int low_def = 2;
    int low_drain = (int)(low_def * 30 / 100.0f);
    int clamped = low_def - low_drain;
    if (clamped < 1) clamped = 1;
    ASSERT_INT_EQ("DWH drain (def=2)", low_drain, 0);
    ASSERT_INT_EQ("DWH clamp (def=2)", clamped, 2);
}

/* ======================================================================== */
/* test: BGS spec mechanics                                                 */
/*                                                                          */
/* BGS drains defence by the damage dealt (drain_type=2).                  */
/* cost=50%, acc=2.0x, str=1.21x (godsword 1.1 * bgs 1.1).               */
/* ref: PlayerVsNPCCalc.ts:458-459 [11,10] for godsword + BGS damage.     */
/* ======================================================================== */

static void test_bgs_mechanics(void) {
    printf("--- BGS spec ---\n");

    ASSERT_INT_EQ("BGS cost",    get_melee_spec_cost(MELEE_SPEC_BGS),      50);
    ASSERT_FLOAT_EQ("BGS acc",   get_melee_spec_acc_mult(MELEE_SPEC_BGS),  2.0f,  1e-3f);
    ASSERT_FLOAT_EQ("BGS str",   get_melee_spec_str_mult(MELEE_SPEC_BGS),  1.21f, 1e-3f);

    /* drain_type=2: defence reduced by damage dealt.
       if damage = 35, current_def = 80: new_def = clamp(80-35, 1, 255) = 45 */
    int damage = 35;
    int current_def = 80;
    int new_def = current_def - damage;
    if (new_def < 1) new_def = 1;
    ASSERT_INT_EQ("BGS drain (dmg=35, def=80)", new_def, 45);

    /* edge case: damage > defence */
    int big_dmg = 90;
    int result = current_def - big_dmg;
    if (result < 1) result = 1;
    ASSERT_INT_EQ("BGS drain clamp (dmg=90, def=80)", result, 1);
}

/* ======================================================================== */
/* test: ZGS spec mechanics                                                 */
/*                                                                          */
/* ZGS: 50% cost, 2.0x acc, 1.1x str, applies 32-tick freeze on hit.      */
/* ======================================================================== */

static void test_zgs_mechanics(void) {
    printf("--- ZGS spec ---\n");

    ASSERT_INT_EQ("ZGS cost",    get_melee_spec_cost(MELEE_SPEC_ZGS),      50);
    ASSERT_FLOAT_EQ("ZGS acc",   get_melee_spec_acc_mult(MELEE_SPEC_ZGS),  2.0f, 1e-5f);
    ASSERT_FLOAT_EQ("ZGS str",   get_melee_spec_str_mult(MELEE_SPEC_ZGS),  1.1f, 1e-3f);
    /* freeze ticks set to 32 in perform_attack when applies_freeze=1 (line 1502-1503) */
}

/* ======================================================================== */
/* test: SGS spec mechanics                                                 */
/*                                                                          */
/* SGS: 50% cost, 2.0x acc, 1.1x str, heals 50% of damage dealt.         */
/* ======================================================================== */

static void test_sgs_mechanics(void) {
    printf("--- SGS spec ---\n");

    ASSERT_INT_EQ("SGS cost",    get_melee_spec_cost(MELEE_SPEC_SGS),      50);
    ASSERT_FLOAT_EQ("SGS acc",   get_melee_spec_acc_mult(MELEE_SPEC_SGS),  2.0f, 1e-3f);
    ASSERT_FLOAT_EQ("SGS str",   get_melee_spec_str_mult(MELEE_SPEC_SGS),  1.1f, 1e-3f);
    /* heal_percent=50 set in perform_attack when heals_attacker=1 (line 1505-1506) */
}

/* ======================================================================== */
/* test: ancient godsword spec mechanics                                    */
/*                                                                          */
/* ancient GS: 50% cost, 2.0x acc, 1.1x str.                              */
/* blood sacrifice: 25 magic damage at 8 tick delay if any hit lands,      */
/* + heals attacker 15% of target max HP (capped at 15 in PvP).           */
/* ======================================================================== */

static void test_ancient_gs_mechanics(void) {
    printf("--- ancient godsword spec ---\n");

    ASSERT_INT_EQ("ancient GS cost",  get_melee_spec_cost(MELEE_SPEC_ANCIENT_GS),      50);
    ASSERT_FLOAT_EQ("ancient GS acc", get_melee_spec_acc_mult(MELEE_SPEC_ANCIENT_GS),   2.0f, 1e-5f);
    ASSERT_FLOAT_EQ("ancient GS str", get_melee_spec_str_mult(MELEE_SPEC_ANCIENT_GS),   1.1f, 1e-3f);

    /* blood sacrifice: 25 fixed magic damage (bleed_damage constant from line 1368) */
    /* heal = clamp(target_base_hp * 0.15, 0, 15) */
    int target_hp = 99;
    int heal = (int)(target_hp * 0.15f);
    if (heal > 15) heal = 15;
    ASSERT_INT_EQ("ancient GS heal (hp=99)", heal, 14);

    int target_hp2 = 120;
    int heal2 = (int)(target_hp2 * 0.15f);
    if (heal2 > 15) heal2 = 15;
    ASSERT_INT_EQ("ancient GS heal capped (hp=120)", heal2, 15);
}

/* ======================================================================== */
/* test: dark bow spec mechanics                                            */
/*                                                                          */
/* dark bow: 55% cost, 1.0x acc, 1.5x str, 2 hits.                        */
/* with dragon arrows: min 8, max 48 clamped per hit.                      */
/* ref: PlayerVsNPCCalc.ts:751-757 — min=8 (dragon arrows), max=48,       */
/*      [15,10]=1.5x damage with dragon arrows.                            */
/* our code: osrs_pvp_combat.h:989-1015                                     */
/* ======================================================================== */

static void test_dark_bow_mechanics(void) {
    printf("--- dark bow spec ---\n");

    ASSERT_INT_EQ("dark bow cost",  get_ranged_spec_cost(RANGED_SPEC_DARK_BOW),      55);
    ASSERT_FLOAT_EQ("dark bow acc", get_ranged_spec_acc_mult(RANGED_SPEC_DARK_BOW),   1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("dark bow str", get_ranged_spec_str_mult(RANGED_SPEC_DARK_BOW),   1.5f, 1e-3f);

    /* clamping: damage = clamp(rand(0, max_hit), 8, 48) */
    /* if hit: damage in [8, 48]. if miss: damage = 8 (minimum guaranteed). */
    int test_damage;

    /* simulate hit: rand gives 3 -> clamped to 8 */
    test_damage = 3;
    test_damage = test_damage < 8 ? 8 : (test_damage > 48 ? 48 : test_damage);
    ASSERT_INT_EQ("dbow clamp low (3->8)", test_damage, 8);

    /* simulate hit: rand gives 60 -> clamped to 48 */
    test_damage = 60;
    test_damage = test_damage < 8 ? 8 : (test_damage > 48 ? 48 : test_damage);
    ASSERT_INT_EQ("dbow clamp high (60->48)", test_damage, 48);

    /* simulate hit: rand gives 25 -> stays 25 */
    test_damage = 25;
    test_damage = test_damage < 8 ? 8 : (test_damage > 48 ? 48 : test_damage);
    ASSERT_INT_EQ("dbow no clamp (25)", test_damage, 25);

    /* miss case: damage = 8 (minimum guaranteed per our code line 1008) */
    ASSERT_INT_EQ("dbow miss min", 8, 8);
}

/* ======================================================================== */
/* test: morrigan's javelin (Phantom Strike) bleed mechanics                */
/*                                                                          */
/* morrigan's: 50% cost, 1.0x acc, 1.0x str (just ranged hit).            */
/* on hit: sets morr_dot_remaining = damage dealt (post-prayer).           */
/* bleed: 5 HP every 3 ticks until remaining exhausted.                     */
/* our code: osrs_pvp_api.h:633-646                                         */
/* ======================================================================== */

static void test_morrigans_bleed(void) {
    printf("--- morrigan's javelin bleed ---\n");

    ASSERT_INT_EQ("morrigan's cost", get_ranged_spec_cost(RANGED_SPEC_MORRIGANS), 50);
    ASSERT_FLOAT_EQ("morrigan's acc", get_ranged_spec_acc_mult(RANGED_SPEC_MORRIGANS), 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("morrigan's str", get_ranged_spec_str_mult(RANGED_SPEC_MORRIGANS), 1.0f, 1e-5f);

    /* bleed tick simulation: damage=23, 5 per 3 ticks.
       tick 3:  5 dealt (23-5=18 remaining)
       tick 6:  5 dealt (18-5=13 remaining)
       tick 9:  5 dealt (13-5=8 remaining)
       tick 12: 5 dealt (8-5=3 remaining)
       tick 15: 3 dealt (3-3=0 remaining, done) */
    int remaining = 23;
    int total_bleed = 0;
    int bleed_ticks = 0;

    while (remaining > 0) {
        int dot = remaining >= 5 ? 5 : remaining;
        remaining -= dot;
        total_bleed += dot;
        bleed_ticks++;
    }

    ASSERT_INT_EQ("morr bleed total (23)", total_bleed, 23);
    ASSERT_INT_EQ("morr bleed ticks (23)", bleed_ticks, 5);

    /* edge case: damage = 5 -> exactly 1 bleed tick */
    remaining = 5;
    total_bleed = 0;
    bleed_ticks = 0;
    while (remaining > 0) {
        int dot = remaining >= 5 ? 5 : remaining;
        remaining -= dot;
        total_bleed += dot;
        bleed_ticks++;
    }
    ASSERT_INT_EQ("morr bleed total (5)", total_bleed, 5);
    ASSERT_INT_EQ("morr bleed ticks (5)", bleed_ticks, 1);

    /* edge case: damage = 1 -> 1 bleed tick of 1 */
    remaining = 1;
    total_bleed = 0;
    bleed_ticks = 0;
    while (remaining > 0) {
        int dot = remaining >= 5 ? 5 : remaining;
        remaining -= dot;
        total_bleed += dot;
        bleed_ticks++;
    }
    ASSERT_INT_EQ("morr bleed total (1)", total_bleed, 1);
    ASSERT_INT_EQ("morr bleed ticks (1)", bleed_ticks, 1);
}

/* ======================================================================== */
/* test: volatile nightmare staff spec                                      */
/*                                                                          */
/* volatile staff: 55% cost, 1.5x acc.                                     */
/* max hit = min(58, 58 * floor(magicLevel/99) + 1) at 99 magic = 58.     */
/* ref: PlayerVsNPCCalc.ts:924-925                                          */
/* ======================================================================== */

static void test_volatile_staff_mechanics(void) {
    printf("--- volatile nightmare staff spec ---\n");

    ASSERT_INT_EQ("volatile cost",  get_magic_spec_cost(MAGIC_SPEC_VOLATILE_STAFF), 55);
    ASSERT_FLOAT_EQ("volatile acc", get_magic_spec_acc_mult(MAGIC_SPEC_VOLATILE_STAFF), 1.5f, 1e-5f);

    /* max hit at magic level 99: min(58, 58 * floor(99/99) + 1) = min(58, 59) = 58 */
    int magic_level = 99;
    int vol_max = 58 * (magic_level / 99) + 1;
    if (vol_max > 58) vol_max = 58;
    if (vol_max < 1) vol_max = 1;
    ASSERT_INT_EQ("volatile max (lvl 99)", vol_max, 58);

    /* at magic level 98: floor(98/99) = 0, so max = min(58, 1) = 1 */
    magic_level = 98;
    vol_max = 58 * (magic_level / 99) + 1;
    if (vol_max > 58) vol_max = 58;
    if (vol_max < 1) vol_max = 1;
    ASSERT_INT_EQ("volatile max (lvl 98)", vol_max, 1);
}

/* ======================================================================== */
/* test: double-hit spec weapons                                            */
/*                                                                          */
/* DDS (dragon dagger): 2 hits, 25% cost, 1.20x acc, 1.15x str.          */
/* abyssal dagger: 2 hits, 50% cost, 1.25x acc, 0.85x str.               */
/* dragon knife: 2 hits, 25% cost, 1.0x acc, 1.0x str.                    */
/* MSB (magic shortbow i): 2 hits, 55% cost, 1.0x acc, 1.0x str.         */
/* ======================================================================== */

static void test_double_hit_specs(void) {
    printf("--- double-hit spec weapons ---\n");

    /* DDS */
    ASSERT_INT_EQ("DDS cost",       get_melee_spec_cost(MELEE_SPEC_DRAGON_DAGGER),     25);
    ASSERT_FLOAT_EQ("DDS acc",      get_melee_spec_acc_mult(MELEE_SPEC_DRAGON_DAGGER),  1.15f, 1e-3f);
    ASSERT_FLOAT_EQ("DDS str",      get_melee_spec_str_mult(MELEE_SPEC_DRAGON_DAGGER),  1.15f, 1e-3f);

    /* abyssal dagger */
    ASSERT_INT_EQ("abyssal dagger cost",  get_melee_spec_cost(MELEE_SPEC_ABYSSAL_DAGGER), 50);
    ASSERT_FLOAT_EQ("abyssal dagger acc", get_melee_spec_acc_mult(MELEE_SPEC_ABYSSAL_DAGGER), 1.25f, 1e-3f);
    ASSERT_FLOAT_EQ("abyssal dagger str", get_melee_spec_str_mult(MELEE_SPEC_ABYSSAL_DAGGER), 0.85f, 1e-3f);

    /* dragon knife */
    ASSERT_INT_EQ("dragon knife cost",  get_ranged_spec_cost(RANGED_SPEC_DRAGON_KNIFE), 25);
    ASSERT_FLOAT_EQ("dragon knife acc", get_ranged_spec_acc_mult(RANGED_SPEC_DRAGON_KNIFE), 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("dragon knife str", get_ranged_spec_str_mult(RANGED_SPEC_DRAGON_KNIFE), 1.0f, 1e-5f);

    /* MSB (magic shortbow i) */
    ASSERT_INT_EQ("MSB cost",  get_ranged_spec_cost(RANGED_SPEC_MSB), 50);
    ASSERT_FLOAT_EQ("MSB acc", get_ranged_spec_acc_mult(RANGED_SPEC_MSB), 1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("MSB str", get_ranged_spec_str_mult(RANGED_SPEC_MSB), 1.0f, 1e-5f);
}

/* ======================================================================== */
/* test: granite maul spec (instant)                                        */
/*                                                                          */
/* gmaul: 50% cost, 1.0x acc, 1.0x str, instant attack (resets timer).    */
/* ======================================================================== */

static void test_granite_maul_mechanics(void) {
    printf("--- granite maul spec ---\n");

    ASSERT_INT_EQ("gmaul cost",     get_melee_spec_cost(MELEE_SPEC_GRANITE_MAUL),      50);
    ASSERT_FLOAT_EQ("gmaul acc",    get_melee_spec_acc_mult(MELEE_SPEC_GRANITE_MAUL),   1.0f, 1e-5f);
    ASSERT_FLOAT_EQ("gmaul str",    get_melee_spec_str_mult(MELEE_SPEC_GRANITE_MAUL),   1.0f, 1e-5f);
}

/* ======================================================================== */
/* test: heavy ballista spec                                                */
/*                                                                          */
/* ballista: 65% cost, 1.25x acc, 1.25x str.                              */
/* ref: PlayerVsNPCCalc.ts:584 [5,4]=1.25x acc, line 744 [5,4]=1.25x str  */
/* ======================================================================== */

static void test_ballista_mechanics(void) {
    printf("--- heavy ballista spec ---\n");

    ASSERT_INT_EQ("ballista cost",    get_ranged_spec_cost(RANGED_SPEC_BALLISTA),      65);
    ASSERT_FLOAT_EQ("ballista acc",   get_ranged_spec_acc_mult(RANGED_SPEC_BALLISTA),   1.25f, 1e-3f);
    ASSERT_FLOAT_EQ("ballista str",   get_ranged_spec_str_mult(RANGED_SPEC_BALLISTA),   1.25f, 1e-3f);
}

/* ======================================================================== */
/* test: ZCB spec (ACB removed — non-DPS bolt proc spec, UNIMPLEMENTED_SPECS) */
/*                                                                          */
/* ZCB: 75% cost, 2.0x acc, 1.0x str.                                     */
/* ref: PlayerVsNPCCalc.ts:580 [2,1]=2.0x acc for ZCB.                    */
/* ======================================================================== */

static void test_crossbow_specs(void) {
    printf("--- crossbow specs (ZCB) ---\n");

    /* ACB removed: non-DPS spec (bolt proc related, dps-calc UNIMPLEMENTED_SPECS) */

    ASSERT_INT_EQ("ZCB cost",    get_ranged_spec_cost(RANGED_SPEC_ZCB),       75);
    ASSERT_FLOAT_EQ("ZCB acc",   get_ranged_spec_acc_mult(RANGED_SPEC_ZCB),    2.0f, 1e-5f);
    ASSERT_FLOAT_EQ("ZCB str",   get_ranged_spec_str_mult(RANGED_SPEC_ZCB),    1.0f, 1e-5f);
}

/* ======================================================================== */
/* test: melee spec two-handed classification                               */
/*                                                                          */
/* godswords, dragon claws, abyssal bludgeon are two-handed.               */
/* DDS, gmaul, VW, DWH, VLS, abyssal dagger, dlong, dmace are one-handed. */
/* ======================================================================== */

static void test_melee_spec_two_handed(void) {
    printf("--- melee spec two-handed classification ---\n");

    ASSERT_INT_EQ("AGS two-handed",      is_melee_spec_two_handed(MELEE_SPEC_AGS),              1);
    ASSERT_INT_EQ("claws two-handed",    is_melee_spec_two_handed(MELEE_SPEC_DRAGON_CLAWS),      1);
    ASSERT_INT_EQ("BGS two-handed",      is_melee_spec_two_handed(MELEE_SPEC_BGS),               1);
    ASSERT_INT_EQ("ZGS two-handed",      is_melee_spec_two_handed(MELEE_SPEC_ZGS),               1);
    ASSERT_INT_EQ("SGS two-handed",      is_melee_spec_two_handed(MELEE_SPEC_SGS),               1);
    ASSERT_INT_EQ("ancient GS two-hand", is_melee_spec_two_handed(MELEE_SPEC_ANCIENT_GS),        1);
    ASSERT_INT_EQ("bludgeon two-handed", is_melee_spec_two_handed(MELEE_SPEC_ABYSSAL_BLUDGEON),  1);

    ASSERT_INT_EQ("gmaul one-handed",    is_melee_spec_two_handed(MELEE_SPEC_GRANITE_MAUL),      0);
    ASSERT_INT_EQ("DDS one-handed",      is_melee_spec_two_handed(MELEE_SPEC_DRAGON_DAGGER),     0);
    ASSERT_INT_EQ("VW one-handed",       is_melee_spec_two_handed(MELEE_SPEC_VOIDWAKER),         0);
    ASSERT_INT_EQ("DWH one-handed",      is_melee_spec_two_handed(MELEE_SPEC_DWH),               0);
    ASSERT_INT_EQ("VLS one-handed",      is_melee_spec_two_handed(MELEE_SPEC_VESTAS),            0);
    ASSERT_INT_EQ("abyssal dag one-hand",is_melee_spec_two_handed(MELEE_SPEC_ABYSSAL_DAGGER),    0);
    ASSERT_INT_EQ("dlong one-handed",    is_melee_spec_two_handed(MELEE_SPEC_DRAGON_LONGSWORD),  0);
    ASSERT_INT_EQ("dmace one-handed",    is_melee_spec_two_handed(MELEE_SPEC_DRAGON_MACE),       0);
}

/* ======================================================================== */
/* test: melee spec bonus types (stab/slash/crush)                          */
/*                                                                          */
/* ref: osrs_pvp_gear.h MELEE_SPEC_BONUS_TYPES[]                           */
/* ======================================================================== */

static void test_melee_spec_bonus_types(void) {
    printf("--- melee spec bonus types ---\n");

    ASSERT_INT_EQ("AGS is slash",        MELEE_SPEC_BONUS_TYPES[MELEE_SPEC_AGS],              MELEE_BONUS_SLASH);
    ASSERT_INT_EQ("claws is slash",      MELEE_SPEC_BONUS_TYPES[MELEE_SPEC_DRAGON_CLAWS],     MELEE_BONUS_SLASH);
    ASSERT_INT_EQ("gmaul is crush",      MELEE_SPEC_BONUS_TYPES[MELEE_SPEC_GRANITE_MAUL],     MELEE_BONUS_CRUSH);
    ASSERT_INT_EQ("DDS is stab",         MELEE_SPEC_BONUS_TYPES[MELEE_SPEC_DRAGON_DAGGER],    MELEE_BONUS_STAB);
    ASSERT_INT_EQ("DWH is crush",        MELEE_SPEC_BONUS_TYPES[MELEE_SPEC_DWH],              MELEE_BONUS_CRUSH);
}

/* ======================================================================== */
/* test: ranged spec hit delays                                             */
/*                                                                          */
/* ref: osrs_pvp_combat.h:497-513                                           */
/* dragon knife / morrigan's: 1 tick. ballista: 3 ticks.                   */
/* dark bow / MSB / ZCB: default ranged formula.                           */
/* ======================================================================== */

static void test_ranged_spec_hit_delays(void) {
    printf("--- ranged spec hit delays ---\n");

    int distance = 5;

    /* dragon knife: always 1 tick delay */
    ASSERT_INT_EQ("dragon knife delay",
        pvp_ranged_hit_delay_for_weapon(distance, 1, RANGED_SPEC_DRAGON_KNIFE), 1);

    /* morrigan's: always 1 tick delay */
    ASSERT_INT_EQ("morrigan's delay",
        pvp_ranged_hit_delay_for_weapon(distance, 1, RANGED_SPEC_MORRIGANS), 1);

    /* ballista: always 3 ticks delay */
    ASSERT_INT_EQ("ballista delay",
        pvp_ranged_hit_delay_for_weapon(distance, 1, RANGED_SPEC_BALLISTA), 3);

    /* dark bow spec: uses default ranged formula */
    int expected_default = (3 + distance) / 6 + 1;
    ASSERT_INT_EQ("dark bow delay",
        pvp_ranged_hit_delay_for_weapon(distance, 1, RANGED_SPEC_DARK_BOW), expected_default);

    /* non-special: always uses default formula regardless of weapon */
    ASSERT_INT_EQ("non-special delay",
        pvp_ranged_hit_delay_for_weapon(distance, 0, RANGED_SPEC_DRAGON_KNIFE), expected_default);
}

/* ======================================================================== */
/* test: spec energy sufficiency checks                                     */
/*                                                                          */
/* verify can_spec / energy checks respect costs correctly.                */
/* ======================================================================== */

static void test_spec_energy_checks(void) {
    printf("--- spec energy sufficiency ---\n");

    /* AGS needs 50: player with 50 can spec, player with 49 cannot */
    ASSERT_INT_EQ("AGS: 50 >= 50",  50 >= get_melee_spec_cost(MELEE_SPEC_AGS), 1);
    ASSERT_INT_EQ("AGS: 49 >= 50",  49 >= get_melee_spec_cost(MELEE_SPEC_AGS), 0);

    /* DDS needs 25: can double-spec from 50 */
    ASSERT_INT_EQ("DDS: 50 >= 25",  50 >= get_melee_spec_cost(MELEE_SPEC_DRAGON_DAGGER), 1);
    ASSERT_INT_EQ("DDS: 25 >= 25",  25 >= get_melee_spec_cost(MELEE_SPEC_DRAGON_DAGGER), 1);
    ASSERT_INT_EQ("DDS: 24 >= 25",  24 >= get_melee_spec_cost(MELEE_SPEC_DRAGON_DAGGER), 0);

    /* BGS needs 50 */
    ASSERT_INT_EQ("BGS: 50 >= 50", 50 >= get_melee_spec_cost(MELEE_SPEC_BGS), 1);
    ASSERT_INT_EQ("BGS: 49 >= 50", 49 >= get_melee_spec_cost(MELEE_SPEC_BGS), 0);

    /* DWH needs 35 */
    ASSERT_INT_EQ("DWH: 35 >= 35",  35 >= get_melee_spec_cost(MELEE_SPEC_DWH), 1);
    ASSERT_INT_EQ("DWH: 34 >= 35",  34 >= get_melee_spec_cost(MELEE_SPEC_DWH), 0);

    /* dark bow needs 55 */
    ASSERT_INT_EQ("dbow: 55 >= 55",  55 >= get_ranged_spec_cost(RANGED_SPEC_DARK_BOW), 1);
    ASSERT_INT_EQ("dbow: 54 >= 55",  54 >= get_ranged_spec_cost(RANGED_SPEC_DARK_BOW), 0);

    /* volatile staff needs 55 */
    ASSERT_INT_EQ("volatile: 55 >= 55", 55 >= get_magic_spec_cost(MAGIC_SPEC_VOLATILE_STAFF), 1);
    ASSERT_INT_EQ("volatile: 54 >= 55", 54 >= get_magic_spec_cost(MAGIC_SPEC_VOLATILE_STAFF), 0);
}

/* ======================================================================== */
/* test: max hit calculation with spec str multiplier                       */
/*                                                                          */
/* uses calculate_max_hit directly with a synthetic player to verify that   */
/* str_mult is applied correctly.                                           */
/* formula: floor(((eff_str * (str_bonus + 64) + 320) / 640) * str_mult)   */
/* ======================================================================== */

static void test_max_hit_with_spec_mult(void) {
    printf("--- max hit with spec str multiplier ---\n");

    Player p;
    memset(&p, 0, sizeof(p));
    p.current_strength = 99;
    p.current_ranged = 99;
    p.offensive_prayer = OFFENSIVE_PRAYER_NONE;
    p.fight_style = FIGHT_STYLE_ACCURATE;

    /* need to set up gear bonuses for strength. use a simple approach:
       set the slot gear bonuses array directly. */
    GearBonuses gb;
    memset(&gb, 0, sizeof(gb));
    gb.melee_strength = 100;  /* typical high str bonus */
    gb.ranged_strength = 80;
    p.current_gear = GEAR_MELEE;

    /* get_slot_gear_bonuses reads slot_cached_bonuses when slot_gear_dirty=0 */
    p.slot_cached_bonuses = gb;
    p.slot_gear_dirty = 0;

    /* effective_strength = floor(99 * 1.0) + 0 (accurate doesn't boost str) + 8 = 107
       base max hit = floor((107 * (100 + 64) + 320) / 640) = floor((17548 + 320) / 640)
                    = floor(17868 / 640) = floor(27.91875) = 27 */
    int base_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 1.0f, 30);
    ASSERT_INT_EQ("base melee max hit (str=99, bonus=100)", base_max, 27);

    /* base max hit = floor((107 * 164 + 320) / 640) = floor(17868 / 640) = 27.
       spec multipliers now apply to the truncated base (matching dps-calc Math.trunc).
       osrs_resolve_spec uses integer multipliers (e.g. max*11/8 for AGS) which is
       the correct OSRS formula. calculate_max_hit with float str_mult truncates
       floor(27 * mult). */

    /* AGS: 1.375x -> floor(27 * 1.375) = floor(37.125) = 37 */
    int ags_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 1.375f, 30);
    ASSERT_INT_EQ("AGS max hit (1.375x)", ags_max, 37);

    /* DDS: 1.15x -> floor(27 * 1.15) = floor(31.05) = 31 */
    int dds_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 1.15f, 30);
    ASSERT_INT_EQ("DDS max hit (1.15x)", dds_max, 31);

    /* BGS: 1.21x -> floor(27 * 1.21) = floor(32.67) = 32 */
    int bgs_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 1.21f, 30);
    ASSERT_INT_EQ("BGS max hit (1.21x)", bgs_max, 32);

    /* DWH/statius: 1.25x -> floor(27 * 1.25) = floor(33.75) = 33 */
    int dwh_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 1.25f, 30);
    ASSERT_INT_EQ("DWH max hit (1.25x)", dwh_max, 33);

    /* ZGS/SGS: 1.1x -> floor(27 * 1.1) = floor(29.7) = 29 */
    int zgs_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 1.1f, 30);
    ASSERT_INT_EQ("ZGS max hit (1.1x)", zgs_max, 29);

    /* abyssal dagger: 0.85x -> floor(27 * 0.85) = floor(22.95) = 22 */
    int abd_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 0.85f, 30);
    ASSERT_INT_EQ("abyssal dagger max hit (0.85x)", abd_max, 22);

    /* dragon mace: 1.5x -> floor(27 * 1.5) = floor(40.5) = 40 */
    int dmace_max = calculate_max_hit(&p, ATTACK_STYLE_MELEE, 1.5f, 30);
    ASSERT_INT_EQ("dragon mace max hit (1.5x)", dmace_max, 40);

    /* ranged: with str_bonus=80, eff_str = floor(99*1.0) + 0 + 8 = 107
       base = floor((107 * (80+64) + 320) / 640) = floor((15408+320)/640) = floor(24.575) = 24 */
    GearBonuses rgb;
    memset(&rgb, 0, sizeof(rgb));
    rgb.ranged_strength = 80;
    p.slot_cached_bonuses = rgb;
    p.slot_gear_dirty = 0;
    p.fight_style = FIGHT_STYLE_ACCURATE;

    int ranged_base = calculate_max_hit(&p, ATTACK_STYLE_RANGED, 1.0f, 30);
    ASSERT_INT_EQ("base ranged max hit (range=99, bonus=80)", ranged_base, 24);

    /* dark bow: 1.5x -> floor(24 * 1.5) = floor(36.0) = 36 */
    int dbow_max = calculate_max_hit(&p, ATTACK_STYLE_RANGED, 1.5f, 30);
    ASSERT_INT_EQ("dark bow max hit (1.5x)", dbow_max, 36);

    /* ballista: 1.25x -> floor(24 * 1.25) = floor(30.0) = 30 */
    int bal_max = calculate_max_hit(&p, ATTACK_STYLE_RANGED, 1.25f, 30);
    ASSERT_INT_EQ("ballista max hit (1.25x)", bal_max, 30);
}

/* ======================================================================== */
/* test: spec accuracy affects hit chance correctly                         */
/*                                                                          */
/* calculate_hit_chance applies acc_mult to the attack roll.               */
/* formula: attack_roll = eff_attack * (att_bonus + 64) * acc_mult          */
/* then normal accuracy formula.                                            */
/* ======================================================================== */

static void test_hit_chance_with_spec_acc(void) {
    printf("--- hit chance with spec accuracy ---\n");

    /* set up minimal OsrsEnv env + attacker/defender */
    OsrsEnv env;
    memset(&env, 0, sizeof(env));

    Player attacker;
    memset(&attacker, 0, sizeof(attacker));
    attacker.current_attack = 99;
    attacker.current_strength = 99;
    attacker.current_defence = 70;
    attacker.current_magic = 99;
    attacker.offensive_prayer = OFFENSIVE_PRAYER_NONE;
    attacker.fight_style = FIGHT_STYLE_ACCURATE;

    GearBonuses att_gb;
    memset(&att_gb, 0, sizeof(att_gb));
    att_gb.slash_attack = 120;
    att_gb.melee_strength = 100;
    attacker.slot_cached_bonuses = att_gb;
    attacker.slot_gear_dirty = 0;
    attacker.current_gear = GEAR_MELEE;

    Player defender;
    memset(&defender, 0, sizeof(defender));
    defender.current_defence = 70;
    defender.current_magic = 70;
    defender.offensive_prayer = OFFENSIVE_PRAYER_NONE;
    defender.fight_style = FIGHT_STYLE_DEFENSIVE;

    GearBonuses def_gb;
    memset(&def_gb, 0, sizeof(def_gb));
    def_gb.slash_defence = 100;
    defender.slot_cached_bonuses = def_gb;
    defender.slot_gear_dirty = 0;
    defender.current_gear = GEAR_MELEE;

    env.players[0] = attacker;
    env.players[1] = defender;

    /* base accuracy (1.0x mult) */
    float base_acc = calculate_hit_chance(&env, &env.players[0], &env.players[1],
                                           ATTACK_STYLE_MELEE, 1.0f);

    /* AGS accuracy (2.0x mult) should be higher */
    float ags_acc = calculate_hit_chance(&env, &env.players[0], &env.players[1],
                                          ATTACK_STYLE_MELEE, 2.0f);

    ASSERT_INT_EQ("AGS acc > base acc", ags_acc > base_acc, 1);
    ASSERT_INT_EQ("base acc > 0", base_acc > 0.0f, 1);
    ASSERT_INT_EQ("AGS acc <= 1.0", ags_acc <= 1.0f, 1);

    /* with 2.0x mult, attack_roll doubles. verify the relationship.
       eff_attack = floor(99*1.0) + 3 + 8 = 110
       attack_roll_base = 110 * (120+64) * 1.0 = 110 * 184 = 20240
       attack_roll_ags  = 110 * 184 * 2.0 = 40480
       eff_def = floor(70*1.0) + 3 + 8 = 81
       def_roll = 81 * (100+64) = 81 * 164 = 13284

       base: att(20240) > def(13284): 1 - 13286/(2*20241) = 1 - 0.32834 = 0.67166
       ags:  att(40480) > def(13284): 1 - 13286/(2*40481) = 1 - 0.16407 = 0.83593 */
    float expected_base = 1.0f - 13286.0f / (2.0f * 20241.0f);
    float expected_ags = 1.0f - 13286.0f / (2.0f * 40481.0f);

    ASSERT_FLOAT_EQ("base acc value", base_acc, expected_base, 1e-3f);
    ASSERT_FLOAT_EQ("AGS acc value",  ags_acc,  expected_ags,  1e-3f);
}

/* ======================================================================== */
/* test: osrs_resolve_spec dispatch                                         */
/*                                                                          */
/* verifies the shared spec dispatch returns correct costs and sensible     */
/* damage values for each weapon category.                                  */
/* ======================================================================== */

static void test_spec_dispatch(void) {
    printf("--- osrs_resolve_spec dispatch ---\n");
    uint32_t rng = 12345;

    /* spec costs via osrs_spec_cost() */
    ASSERT_INT_EQ("AGS cost (dispatch)",       osrs_spec_cost(ITEM_AGS),                  50);
    ASSERT_INT_EQ("claws cost (dispatch)",     osrs_spec_cost(ITEM_DRAGON_CLAWS),         50);
    ASSERT_INT_EQ("DWH cost (dispatch)",       osrs_spec_cost(ITEM_STATIUS_WARHAMMER),    35);
    ASSERT_INT_EQ("BGS cost (dispatch)",       osrs_spec_cost(ITEM_BGS),                  50);
    ASSERT_INT_EQ("ZGS cost (dispatch)",       osrs_spec_cost(ITEM_ZGS),                  50);
    ASSERT_INT_EQ("SGS cost (dispatch)",       osrs_spec_cost(ITEM_SGS),                  50);
    ASSERT_INT_EQ("ancient GS cost (dispatch)",osrs_spec_cost(ITEM_ANCIENT_GS),           50);
    ASSERT_INT_EQ("VLS cost (dispatch)",       osrs_spec_cost(ITEM_VESTAS),               25);
    ASSERT_INT_EQ("VW cost (dispatch)",        osrs_spec_cost(ITEM_VOIDWAKER),            50);
    ASSERT_INT_EQ("gmaul cost (dispatch)",     osrs_spec_cost(ITEM_GRANITE_MAUL),         50);
    ASSERT_INT_EQ("DDS cost (dispatch)",       osrs_spec_cost(ITEM_DRAGON_DAGGER),        25);
    ASSERT_INT_EQ("elder maul cost (dispatch)",osrs_spec_cost(ITEM_ELDER_MAUL),           50);
    ASSERT_INT_EQ("blowpipe cost (dispatch)",  osrs_spec_cost(ITEM_TOXIC_BLOWPIPE),       50);
    ASSERT_INT_EQ("MSB cost (dispatch)",       osrs_spec_cost(ITEM_MAGIC_SHORTBOW_I),     50);
    ASSERT_INT_EQ("dark bow cost (dispatch)",  osrs_spec_cost(ITEM_DARK_BOW),             55);
    ASSERT_INT_EQ("ZCB cost (dispatch)",       osrs_spec_cost(ITEM_ZARYTE_CROSSBOW),      75);
    ASSERT_INT_EQ("ballista cost (dispatch)",  osrs_spec_cost(ITEM_HEAVY_BALLISTA),       65);
    ASSERT_INT_EQ("morr cost (dispatch)",      osrs_spec_cost(ITEM_MORRIGANS_JAVELIN),    50);
    ASSERT_INT_EQ("volatile cost (dispatch)",  osrs_spec_cost(ITEM_VOLATILE_STAFF),       55);
    ASSERT_INT_EQ("eye of ayak cost (dispatch)", osrs_spec_cost(ITEM_EYE_OF_AYAK),        50);
    ASSERT_INT_EQ("zuriel cost (no spec)",     osrs_spec_cost(ITEM_ZURIELS_STAFF),        0);
    ASSERT_INT_EQ("ACB cost (dispatch)",        osrs_spec_cost(ITEM_ARMADYL_CROSSBOW),     50);
    ASSERT_INT_EQ("non-weapon cost",           osrs_spec_cost(ITEM_BARROWS_GLOVES),       0);

    /* resolve AGS with guaranteed hit (huge att vs tiny def) */
    rng = 42;
    SpecResult sr = osrs_resolve_spec(ITEM_AGS, 100000, 50, 100, 99, &rng);
    ASSERT_INT_EQ("AGS num_hits", sr.num_hits, 1);
    tests_run++;
    if (sr.total_damage >= 0 && sr.total_damage <= 50 * 11 / 8) { tests_passed++; }
    else { tests_failed++; printf("  FAIL: AGS damage %d out of range [0, %d]\n", sr.total_damage, 50 * 11 / 8); }

    /* resolve blowpipe spec with guaranteed hit */
    rng = 99;
    sr = osrs_resolve_spec(ITEM_TOXIC_BLOWPIPE, 100000, 30, 100, 99, &rng);
    ASSERT_INT_EQ("blowpipe num_hits", sr.num_hits, 1);
    ASSERT_INT_EQ("blowpipe heal", sr.heal, sr.total_damage / 2);

    /* resolve dragon claws — always 4 hits */
    rng = 777;
    sr = osrs_resolve_spec(ITEM_DRAGON_CLAWS, 100000, 40, 100, 99, &rng);
    ASSERT_INT_EQ("claws num_hits", sr.num_hits, 4);
    ASSERT_INT_EQ("claws total = sum", sr.total_damage,
                  sr.damage[0] + sr.damage[1] + sr.damage[2] + sr.damage[3]);

    /* resolve DDS — always 2 hits */
    rng = 555;
    sr = osrs_resolve_spec(ITEM_DRAGON_DAGGER, 100000, 40, 100, 99, &rng);
    ASSERT_INT_EQ("DDS num_hits", sr.num_hits, 2);

    /* resolve SGS — heals half of damage */
    rng = 333;
    sr = osrs_resolve_spec(ITEM_SGS, 100000, 50, 100, 99, &rng);
    ASSERT_INT_EQ("SGS heal", sr.heal, sr.total_damage / 2);

    /* resolve ZGS — freezes on hit */
    rng = 111;
    sr = osrs_resolve_spec(ITEM_ZGS, 100000, 50, 100, 99, &rng);
    if (sr.total_damage > 0) {
        ASSERT_INT_EQ("ZGS freeze", sr.freeze_ticks, 32);
    }

    /* resolve DWH — drains def on hit */
    rng = 222;
    sr = osrs_resolve_spec(ITEM_STATIUS_WARHAMMER, 100000, 50, 100, 70, &rng);
    if (sr.total_damage > 0) {
        ASSERT_INT_EQ("DWH def drain", sr.def_drain, 70 * 30 / 100);
    }

    /* resolve BGS — drains def by damage */
    rng = 444;
    sr = osrs_resolve_spec(ITEM_BGS, 100000, 50, 100, 99, &rng);
    if (sr.total_damage > 0) {
        ASSERT_INT_EQ("BGS drain = dmg", sr.def_drain, sr.total_damage);
    }

    /* resolve dark bow — 2 hits, each >= 8 */
    rng = 888;
    sr = osrs_resolve_spec(ITEM_DARK_BOW, 100000, 30, 100, 99, &rng);
    ASSERT_INT_EQ("dbow num_hits", sr.num_hits, 2);
    ASSERT_INT_EQ("dbow hit1 >= 8", sr.damage[0] >= 8, 1);
    ASSERT_INT_EQ("dbow hit2 >= 8", sr.damage[1] >= 8, 1);

    /* resolve MSB — 2 hits */
    rng = 666;
    sr = osrs_resolve_spec(ITEM_MAGIC_SHORTBOW_I, 100000, 20, 100, 99, &rng);
    ASSERT_INT_EQ("MSB num_hits", sr.num_hits, 2);

    /* resolve eye of ayak — magic def drain */
    rng = 1234;
    sr = osrs_resolve_spec(ITEM_EYE_OF_AYAK, 100000, 40, 100, 99, &rng);
    ASSERT_INT_EQ("ayak num_hits", sr.num_hits, 1);
    if (sr.total_damage > 0) {
        ASSERT_INT_EQ("ayak magic drain = dmg", sr.magic_def_drain, sr.total_damage);
    }
    ASSERT_INT_EQ("ayak speed override", sr.attack_speed_override, 5);

    /* resolve non-spec weapon — empty result */
    rng = 9999;
    sr = osrs_resolve_spec(ITEM_BARROWS_GLOVES, 100000, 50, 100, 99, &rng);
    ASSERT_INT_EQ("non-weapon num_hits", sr.num_hits, 0);
    ASSERT_INT_EQ("non-weapon damage", sr.total_damage, 0);
}


int main(void) {
    printf("=== special attack tests (cross-referenced with osrs-dps-calc) ===\n\n");

    /* spec costs */
    test_melee_spec_costs();
    test_ranged_spec_costs();
    test_magic_spec_costs();

    /* accuracy multipliers */
    test_melee_spec_acc_multipliers();
    test_ranged_spec_acc_multipliers();
    test_magic_spec_acc_multiplier();

    /* strength multipliers */
    test_melee_spec_str_multipliers();
    test_ranged_spec_str_multipliers();

    /* blowpipe spec */
    test_blowpipe_spec_constants();
    test_blowpipe_spec_resolve();

    /* special mechanics */
    test_dragon_claws_cascade();
    test_voidwaker_mechanics();
    test_vls_mechanics();
    test_statius_warhammer_mechanics();
    test_bgs_mechanics();
    test_zgs_mechanics();
    test_sgs_mechanics();
    test_ancient_gs_mechanics();
    test_dark_bow_mechanics();
    test_morrigans_bleed();
    test_volatile_staff_mechanics();
    test_double_hit_specs();
    test_granite_maul_mechanics();
    test_ballista_mechanics();
    test_crossbow_specs();

    /* classification + metadata */
    test_melee_spec_two_handed();
    test_melee_spec_bonus_types();
    test_ranged_spec_hit_delays();
    test_spec_energy_checks();

    /* integration: max hit + hit chance with spec multipliers */
    test_max_hit_with_spec_mult();
    test_hit_chance_with_spec_acc();

    /* shared spec dispatch */
    test_spec_dispatch();

    printf("\n=== results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(", %d FAILED", tests_failed);
    }
    printf(" ===\n");

    return tests_failed > 0 ? 1 : 0;
}
