#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "affine_lock_visible_targets.h"

#define EXPECT_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "EXPECT_TRUE failed at %s:%d: %s\n", \
            __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

#define EXPECT_EQ_U32(actual, expected) do { \
    uint32_t actual_value = (uint32_t)(actual); \
    uint32_t expected_value = (uint32_t)(expected); \
    if (actual_value != expected_value) { \
        fprintf(stderr, \
            "EXPECT_EQ_U32 failed at %s:%d: %s=%u expected %u\n", \
            __FILE__, __LINE__, #actual, actual_value, expected_value); \
        exit(1); \
    } \
} while (0)

#define EXPECT_EQ_U64(actual, expected) do { \
    uint64_t actual_value = (uint64_t)(actual); \
    uint64_t expected_value = (uint64_t)(expected); \
    if (actual_value != expected_value) { \
        fprintf(stderr, \
            "EXPECT_EQ_U64 failed at %s:%d: %s=%llu expected %llu\n", \
            __FILE__, __LINE__, #actual, \
            (unsigned long long)actual_value, \
            (unsigned long long)expected_value); \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr,
            "usage: %s TARGET_BIN EXPECTED_RECORD_COUNT "
            "EXPECTED_SAMPLE_COUNT EXPECTED_D16_COUNT\n",
            argv[0]);
        return 1;
    }

    char* end = NULL;
    unsigned long expected_record_count = strtoul(argv[2], &end, 10);
    EXPECT_TRUE(end != argv[2] && *end == '\0');
    unsigned long expected_sample_count = strtoul(argv[3], &end, 10);
    EXPECT_TRUE(end != argv[3] && *end == '\0');
    unsigned long expected_d16_count = strtoul(argv[4], &end, 10);
    EXPECT_TRUE(end != argv[4] && *end == '\0');

    VisibleTargetTable table;
    int rc = visible_targets_load(
        argv[1],
        VISIBLE_TARGET_8ACTION_V1_HASH,
        &table);
    if (rc != 0) {
        fprintf(stderr, "failed to load visible target table: %s\n", argv[1]);
        return 1;
    }

    EXPECT_EQ_U32(table.bits, 16);
    EXPECT_EQ_U32(table.num_actions, 8);
    EXPECT_EQ_U32(table.depth_count, 6);
    EXPECT_EQ_U32(table.record_size, 16);
    EXPECT_EQ_U32(table.record_count, expected_record_count);
    EXPECT_EQ_U64(
        table.action_set_hash,
        VISIBLE_TARGET_8ACTION_V1_HASH);

    const uint32_t expected_depths[6] = {2, 4, 5, 6, 8, 16};
    const uint64_t expected_exact_counts[6] = {
        2216496ull,
        34379722ull,
        115388932ull,
        331789220ull,
        1125374770ull,
        100548ull,
    };
    uint32_t first_record = 0;
    for (uint32_t i = 0; i < table.depth_count; i++) {
        EXPECT_EQ_U32(table.depths[i].depth, expected_depths[i]);
        EXPECT_EQ_U32(table.depths[i].first_record, first_record);
        uint32_t expected_stored_count = i == 5 ?
            (uint32_t)expected_d16_count : (uint32_t)expected_sample_count;
        EXPECT_EQ_U32(table.depths[i].stored_count, expected_stored_count);
        EXPECT_EQ_U64(table.depths[i].exact_pair_count, expected_exact_counts[i]);
        first_record += table.depths[i].stored_count;
    }

    for (uint32_t i = 0; i < table.record_count; i++) {
        const VisibleTargetRecord* record = &table.records[i];
        EXPECT_TRUE(record->solution_length == record->depth);
        EXPECT_TRUE(
            record->depth == 2 ||
            record->depth == 4 ||
            record->depth == 5 ||
            record->depth == 6 ||
            record->depth == 8 ||
            record->depth == 16);
        for (uint8_t step = 0; step < record->solution_length; step++) {
            uint8_t action = (record->packed_actions >> (3u * step)) & 7u;
            EXPECT_TRUE(action < table.num_actions);
        }
    }

    visible_targets_free(&table);
    return 0;
}
