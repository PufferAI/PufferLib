/**
 * @fileoverview Regression checks for OSRS visual animation export coverage.
 *
 * BUILD:
 *   cc -std=c11 -O0 -g -I. -o /tmp/test_osrs_visual_asset_exports \
 *       ocean/osrs/tests/test_osrs_visual_asset_exports.c -lm
 *   /tmp/test_osrs_visual_asset_exports
 */

#include <stdio.h>

#include "ocean/osrs/encounters/encounter_inferno.h"
#include "ocean/osrs/osrs_anim.h"

static int tests_run = 0;
static int tests_failed = 0;

static int has_anim(AnimCache* first, AnimCache* second, int seq_id) {
    return anim_get_sequence(first, (uint16_t)seq_id) ||
        anim_get_sequence(second, (uint16_t)seq_id);
}

#define ASSERT_ANIM_PRESENT(label, first, second, seq_id) do { \
    tests_run++; \
    if (!has_anim((first), (second), (seq_id))) { \
        tests_failed++; \
        printf("  FAIL: %s missing seq %d\n", (label), (seq_id)); \
    } \
} while (0)

int main(void) {
    AnimCache* equipment = anim_cache_load("data/equipment.anims");
    AnimCache* inferno = anim_cache_load("data/inferno.anims");
    AnimCache* zulrah = anim_cache_load("data/zulrah.anims");
    if (!equipment || !inferno || !zulrah) return 1;

    printf("--- inferno runtime animation export coverage ---\n");
    ASSERT_ANIM_PRESENT("jad magic attack", equipment, inferno, INF_ANIM_JALTOK_JAD_MAGIC_ATTACK);
    ASSERT_ANIM_PRESENT("jad ranged attack", equipment, inferno, INF_ANIM_JALTOK_JAD_RANGED_ATTACK);
    ASSERT_ANIM_PRESENT("jad melee attack", equipment, inferno, INF_ANIM_JALTOK_JAD_MELEE_ATTACK);
    ASSERT_ANIM_PRESENT("meleer dig down", equipment, inferno, INF_GEN_ANIM_MELEER_DIG_DOWN);
    ASSERT_ANIM_PRESENT("meleer dig up", equipment, inferno, INF_GEN_ANIM_MELEER_DIG_UP);

    printf("--- zulrah runtime animation export coverage ---\n");
    ASSERT_ANIM_PRESENT("zulrah attack", equipment, zulrah, ZULRAH_ANIM_ATTACK);
    ASSERT_ANIM_PRESENT("zulrah idle", equipment, zulrah, ZULRAH_ANIM_IDLE);
    ASSERT_ANIM_PRESENT("zulrah dive", equipment, zulrah, ZULRAH_ANIM_DIVE);
    ASSERT_ANIM_PRESENT("zulrah rise", equipment, zulrah, ZULRAH_ANIM_RISE);
    ASSERT_ANIM_PRESENT("snakeling idle", equipment, zulrah, SNAKELING_ANIM_IDLE);
    ASSERT_ANIM_PRESENT("snakeling melee attack", equipment, zulrah, SNAKELING_ANIM_MELEE);
    ASSERT_ANIM_PRESENT("snakeling magic attack", equipment, zulrah, SNAKELING_ANIM_MAGIC);
    ASSERT_ANIM_PRESENT("snakeling death", equipment, zulrah, SNAKELING_ANIM_DEATH);
    ASSERT_ANIM_PRESENT("snakeling walk", equipment, zulrah, SNAKELING_ANIM_WALK);

    if (tests_failed > 0) {
        printf("\n%d/%d animation export checks failed\n", tests_failed, tests_run);
        return 1;
    }
    printf("\n%d/%d animation export checks passed\n", tests_run, tests_run);
    return 0;
}
