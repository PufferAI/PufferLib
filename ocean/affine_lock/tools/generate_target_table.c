#define _POSIX_C_SOURCE 200809L
#define AFFINE_LOCK_NO_RENDER
#define AFFINE_LOCK_ENABLE_TRANSFORM_TABLE_HELPERS

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../affine_lock.h"

#define TARGET_STATE_COUNT (1 << AFFINE_LOCK_BITS)
#define TARGET_CHOICES 8

typedef struct TargetRecord {
    uint16_t transform_indices[TARGET_CHOICES];
    uint8_t target_distance;
    uint16_t target_count;
    uint8_t choice_count;
} TargetRecord;

typedef struct CandidateSet {
    uint16_t transform_indices[TARGET_CHOICES];
    uint32_t scores[TARGET_CHOICES];
    int choice_count;
    int target_count;
} CandidateSet;

static const int target_depths[] = {2, 4, 8, 16};
static const int target_depth_count =
    (int)(sizeof(target_depths) / sizeof(target_depths[0]));

static uint64_t mix_u64(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint32_t candidate_score(uint32_t state, int distance, uint32_t index) {
    uint64_t hash = 1469598103934665603ull;
    hash = mix_u64(hash, state);
    hash = mix_u64(hash, (uint64_t)distance);
    hash = mix_u64(hash, (uint64_t)index);
    return (uint32_t)(hash ^ (hash >> 32));
}

static void reset_candidate_set(CandidateSet* set) {
    memset(set, 0, sizeof(*set));
}

static void add_candidate(
        CandidateSet* set,
        uint32_t state,
        int distance,
        uint32_t transform_index) {
    set->target_count += 1;
    uint32_t score = candidate_score(state, distance, transform_index);
    if (set->choice_count < TARGET_CHOICES) {
        int slot = set->choice_count++;
        set->scores[slot] = score;
        set->transform_indices[slot] = (uint16_t)transform_index;
        return;
    }

    int worst_slot = 0;
    for (int i = 1; i < TARGET_CHOICES; i++) {
        if (set->scores[i] > set->scores[worst_slot]) {
            worst_slot = i;
        }
    }
    if (score < set->scores[worst_slot]) {
        set->scores[worst_slot] = score;
        set->transform_indices[worst_slot] = (uint16_t)transform_index;
    }
}

static void store_candidate_set(
        TargetRecord* record,
        const CandidateSet* set,
        int target_distance) {
    memset(record, 0, sizeof(*record));
    record->target_distance = (uint8_t)target_distance;
    record->target_count = (uint16_t)set->target_count;
    record->choice_count = (uint8_t)set->choice_count;
    for (int i = 0; i < set->choice_count; i++) {
        record->transform_indices[i] = set->transform_indices[i];
    }
}

static int target_depth_index(int distance) {
    for (int i = 0; i < target_depth_count; i++) {
        if (target_depths[i] == distance) {
            return i;
        }
    }
    return -1;
}

static void compute_records_for_state(
        const AffineLockShared* shared,
        uint32_t state,
        uint32_t* seen_generation,
        uint32_t generation,
        TargetRecord records[4]) {
    CandidateSet exact_sets[4];
    for (int i = 0; i < target_depth_count; i++) {
        reset_candidate_set(&exact_sets[i]);
    }
    int farthest_distance = 0;
    CandidateSet farthest_set;
    reset_candidate_set(&farthest_set);

    for (int distance = 0;
            distance <= AFFINE_LOCK_PRECOMPUTED_TRANSFORM_MAX_DISTANCE;
            distance++) {
        uint32_t shell_start =
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[distance];
        uint32_t shell_end =
            AFFINE_LOCK_PRECOMPUTED_TRANSFORM_SHELL_OFFSETS[distance + 1];
        for (uint32_t index = shell_start; index < shell_end; index++) {
            const AffineLockPrecomputedTransform* transform =
                &AFFINE_LOCK_PRECOMPUTED_TRANSFORMS[index];
            uint32_t target =
                affine_lock_apply_precomputed_transform(shared, state, transform);
            if (seen_generation[target] == generation) {
                continue;
            }
            seen_generation[target] = generation;

            int depth_index = target_depth_index(distance);
            if (depth_index >= 0) {
                add_candidate(&exact_sets[depth_index], state, distance, index);
            }

            if (distance > farthest_distance) {
                farthest_distance = distance;
                reset_candidate_set(&farthest_set);
                add_candidate(&farthest_set, state, distance, index);
            } else if (distance == farthest_distance && distance > 0) {
                add_candidate(&farthest_set, state, distance, index);
            }
        }
    }

    for (int i = 0; i < target_depth_count; i++) {
        if (exact_sets[i].target_count > 0) {
            store_candidate_set(&records[i], &exact_sets[i], target_depths[i]);
        } else {
            store_candidate_set(&records[i], &farthest_set, farthest_distance);
        }
    }
}

static uint64_t checksum_records(const TargetRecord* records, int record_count) {
    uint64_t hash = 1469598103934665603ull;
    for (int i = 0; i < target_depth_count; i++) {
        hash = mix_u64(hash, (uint64_t)target_depths[i]);
    }
    for (int i = 0; i < record_count; i++) {
        for (int choice = 0; choice < TARGET_CHOICES; choice++) {
            hash = mix_u64(hash, records[i].transform_indices[choice]);
        }
        hash = mix_u64(hash, records[i].target_distance);
        hash = mix_u64(hash, records[i].target_count);
        hash = mix_u64(hash, records[i].choice_count);
    }
    return hash;
}

static int write_header(const char* path, const TargetRecord* records) {
    FILE* file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }

    int record_count = TARGET_STATE_COUNT * target_depth_count;
    uint64_t checksum = checksum_records(records, record_count);

    fprintf(file, "#pragma once\n\n");
    fprintf(file,
        "/* Generated by ocean/affine_lock/tools/generate_target_table.c. */\n\n");
    fprintf(file, "#include <stdint.h>\n\n");
    fprintf(file, "#define AFFINE_LOCK_PRECOMPUTED_TARGET_STATE_COUNT %d\n",
        TARGET_STATE_COUNT);
    fprintf(file, "#define AFFINE_LOCK_PRECOMPUTED_TARGET_DEPTH_COUNT %d\n",
        target_depth_count);
    fprintf(file, "#define AFFINE_LOCK_PRECOMPUTED_TARGET_CHOICES %d\n",
        TARGET_CHOICES);
    fprintf(file, "#define AFFINE_LOCK_PRECOMPUTED_TARGET_CHECKSUM 0x%016llxull\n\n",
        (unsigned long long)checksum);
    fprintf(file, "typedef struct AffineLockPrecomputedTarget {\n");
    fprintf(file, "    uint16_t transform_indices[%d];\n", TARGET_CHOICES);
    fprintf(file, "    uint8_t target_distance;\n");
    fprintf(file, "    uint16_t target_count;\n");
    fprintf(file, "    uint8_t choice_count;\n");
    fprintf(file, "} AffineLockPrecomputedTarget;\n\n");
    fprintf(file, "static const uint8_t AFFINE_LOCK_PRECOMPUTED_TARGET_DEPTHS[%d] = {",
        target_depth_count);
    for (int i = 0; i < target_depth_count; i++) {
        fprintf(file, "%s%d", i == 0 ? "" : ", ", target_depths[i]);
    }
    fprintf(file, "};\n\n");
    fprintf(file,
        "static const AffineLockPrecomputedTarget\n"
        "AFFINE_LOCK_PRECOMPUTED_TARGETS[%d][%d] = {\n",
        TARGET_STATE_COUNT, target_depth_count);

    for (int state = 0; state < TARGET_STATE_COUNT; state++) {
        fprintf(file, "    {");
        for (int depth_index = 0; depth_index < target_depth_count;
                depth_index++) {
            const TargetRecord* record =
                &records[state * target_depth_count + depth_index];
            fprintf(file, "%s{{", depth_index == 0 ? "" : ", ");
            for (int choice = 0; choice < TARGET_CHOICES; choice++) {
                fprintf(file, "%s%u", choice == 0 ? "" : ", ",
                    (unsigned int)record->transform_indices[choice]);
            }
            fprintf(file, "}, %u, %u, %u}",
                (unsigned int)record->target_distance,
                (unsigned int)record->target_count,
                (unsigned int)record->choice_count);
        }
        fprintf(file, "},\n");
    }
    fprintf(file, "};\n");

    if (fclose(file) != 0) {
        fprintf(stderr, "failed to close %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

static int files_equal(const char* a, const char* b) {
    FILE* fa = fopen(a, "rb");
    FILE* fb = fopen(b, "rb");
    if (fa == NULL || fb == NULL) {
        if (fa != NULL) {
            fclose(fa);
        }
        if (fb != NULL) {
            fclose(fb);
        }
        return 0;
    }

    int equal = 1;
    while (1) {
        unsigned char ba[8192];
        unsigned char bb[8192];
        size_t ra = fread(ba, 1, sizeof(ba), fa);
        size_t rb = fread(bb, 1, sizeof(bb), fb);
        if (ra != rb || memcmp(ba, bb, ra) != 0) {
            equal = 0;
            break;
        }
        if (ra < sizeof(ba)) {
            break;
        }
    }

    fclose(fa);
    fclose(fb);
    return equal;
}

int main(int argc, char** argv) {
    int check = argc > 1 && strcmp(argv[1], "--check") == 0;
    const char* output_path =
        "ocean/affine_lock/generated/affine_lock_target_table.h";
    const char* temp_path = "/tmp/affine_lock_target_table_check.h";

    AffineLockShared shared;
    memset(&shared, 0, sizeof(shared));
    if (affine_lock_init_shared(&shared, AFFINE_LOCK_BITS, 2, 16, 2, 0) != 0 ||
            affine_lock_prepare_precomputed_transforms(&shared) != 0) {
        fprintf(stderr, "failed to initialize affine_lock shared state\n");
        affine_lock_free_shared(&shared);
        return 1;
    }

    int record_count = TARGET_STATE_COUNT * target_depth_count;
    TargetRecord* records =
        (TargetRecord*)calloc((size_t)record_count, sizeof(TargetRecord));
    uint32_t* seen_generation =
        (uint32_t*)calloc((size_t)shared.num_states, sizeof(uint32_t));
    if (records == NULL || seen_generation == NULL) {
        fprintf(stderr, "failed to allocate target table generator buffers\n");
        free(records);
        free(seen_generation);
        affine_lock_free_shared(&shared);
        return 1;
    }

    for (uint32_t state = 0; state < (uint32_t)shared.num_states; state++) {
        uint32_t generation = state + 1u;
        compute_records_for_state(
            &shared, state, seen_generation, generation,
            &records[state * target_depth_count]);
    }

    int rc = write_header(check ? temp_path : output_path, records);
    if (rc == 0 && check && !files_equal(temp_path, output_path)) {
        fprintf(stderr,
            "stale generated affine lock target table: %s\n"
            "run: ocean/affine_lock/tools/generate_target_table\n",
            output_path);
        rc = 1;
    }

    free(records);
    free(seen_generation);
    affine_lock_free_shared(&shared);
    return rc == 0 ? 0 : 1;
}
