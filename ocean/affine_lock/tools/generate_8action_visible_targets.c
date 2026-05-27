#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define BITS 16
#define STATE_COUNT (1u << BITS)
#define MAX_ACTIONS 8
#define TARGET_DEPTH_COUNT 6
#define MAX_DISTANCE 64
#define RECORD_SIZE 16
#define FORMAT_VERSION 1

static const int TARGET_DEPTHS[TARGET_DEPTH_COUNT] = {2, 4, 5, 6, 8, 16};
typedef enum ActionOp {
    ACTION_OP_SHIFT_LEFT = 0,
    ACTION_OP_SHIFT_RIGHT = 1,
    ACTION_OP_MIRROR = 2,
    ACTION_OP_INVERT_RIGHT_7 = 3,
    ACTION_OP_SWAP_ADJACENT_BITS = 4,
    ACTION_OP_SWAP_ADJACENT_PAIRS = 5,
    ACTION_OP_SWAP_NIBBLES_EACH_BYTE = 6,
    ACTION_OP_REVERSE_EACH_NIBBLE = 7,
    ACTION_OP_REVERSE_EACH_BYTE = 8,
} ActionOp;

typedef struct ActionSet {
    const char* name;
    int num_actions;
    int store_all_d16_by_default;
    // Stable salt for deterministic sampled-record selection.
    uint64_t candidate_score_seed;
    const char* default_bin;
    const char* default_json;
    ActionOp ops[MAX_ACTIONS];
    const char* names[MAX_ACTIONS];
} ActionSet;

static const ActionSet ACTION_SETS[] = {
    {
        "affine_lock_8action_v1",
        8,
        1,
        0x7b7ba09982ec5a9dull,
        "ocean/affine_lock/generated/affine_lock_8action_visible_targets.bin",
        "ocean/affine_lock/generated/affine_lock_8action_visible_targets.json",
        {
            ACTION_OP_SHIFT_LEFT,
            ACTION_OP_SHIFT_RIGHT,
            ACTION_OP_INVERT_RIGHT_7,
            ACTION_OP_SWAP_ADJACENT_BITS,
            ACTION_OP_SWAP_ADJACENT_PAIRS,
            ACTION_OP_SWAP_NIBBLES_EACH_BYTE,
            ACTION_OP_REVERSE_EACH_NIBBLE,
            ACTION_OP_REVERSE_EACH_BYTE,
        },
        {
            "shift_left",
            "shift_right",
            "invert_right_7",
            "swap_adjacent_bits",
            "swap_adjacent_pairs",
            "swap_nibbles_each_byte",
            "reverse_each_nibble",
            "reverse_each_byte",
        },
    },
    {
        // Generator-only alternate for future runtime experiments. Fewer
        // actions can make policy search easier while producing many more
        // exact depth-16 pairs than the committed 8-action training set.
        "affine_lock_4action_v1",
        4,
        0,
        0x8c4d9362024c02b8ull,
        "ocean/affine_lock/generated/affine_lock_4action_visible_targets.bin",
        "ocean/affine_lock/generated/affine_lock_4action_visible_targets.json",
        {
            ACTION_OP_SHIFT_RIGHT,
            ACTION_OP_MIRROR,
            ACTION_OP_INVERT_RIGHT_7,
            ACTION_OP_SWAP_ADJACENT_BITS,
        },
        {
            "shift_right",
            "mirror",
            "invert_right_7",
            "swap_adjacent_bits",
        },
    },
};

static const int ACTION_SET_COUNT =
    (int)(sizeof(ACTION_SETS) / sizeof(ACTION_SETS[0]));
static const ActionSet* ACTIVE_ACTION_SET = &ACTION_SETS[0];

typedef struct TargetRecord {
    uint16_t start;
    uint16_t target;
    uint64_t packed_actions;
    uint8_t solution_length;
    uint8_t depth;
    uint64_t score;
} TargetRecord;

typedef struct DepthSample {
    int depth;
    int store_all;
    uint64_t exact_count;
    uint32_t capacity;
    uint32_t count;
    TargetRecord* records;
} DepthSample;

typedef struct WorkerResult {
    DepthSample depths[TARGET_DEPTH_COUNT];
    uint64_t histogram[MAX_DISTANCE + 1];
    uint64_t disconnected_starts;
    int max_distance;
} WorkerResult;

typedef struct Options {
    const char* output_bin;
    const char* output_json;
    const ActionSet* action_set;
    uint32_t sample_per_depth;
    uint64_t sample_seed;
    int store_all_depths[TARGET_DEPTH_COUNT];
    int output_bin_explicit;
    int output_json_explicit;
} Options;

static uint16_t NEXT_STATE[STATE_COUNT][MAX_ACTIONS];
static uint64_t ACTIVE_SAMPLE_SEED = 0u;

static uint64_t mix_u64(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t mix_bytes(uint64_t hash, const char* text) {
    const unsigned char* ptr = (const unsigned char*)text;
    while (*ptr != '\0') {
        hash = mix_u64(hash, (uint64_t)*ptr);
        ptr++;
    }
    return hash;
}

static uint16_t shift_left(uint16_t state) {
    uint16_t first = state & 1u;
    return (uint16_t)((state >> 1) | (first << (BITS - 1)));
}

static uint16_t shift_right(uint16_t state) {
    uint16_t last = (uint16_t)((state >> (BITS - 1)) & 1u);
    return (uint16_t)(((state << 1) & 0xffffu) | last);
}

static uint16_t mirror_bits(uint16_t state) {
    uint16_t out = 0u;
    for (int bit = 0; bit < BITS; bit++) {
        if ((state & (1u << bit)) != 0u) {
            out |= (uint16_t)(1u << (BITS - 1 - bit));
        }
    }
    return out;
}

static uint16_t swap_adjacent_bits(uint16_t state) {
    return (uint16_t)(((state & 0x5555u) << 1) |
        ((state & 0xaaaau) >> 1));
}

static uint16_t swap_adjacent_pairs(uint16_t state) {
    return (uint16_t)(((state & 0x3333u) << 2) |
        ((state & 0xccccu) >> 2));
}

static uint16_t swap_nibbles_each_byte(uint16_t state) {
    return (uint16_t)(((state & 0x0f0fu) << 4) |
        ((state & 0xf0f0u) >> 4));
}

static uint16_t reverse_each_nibble(uint16_t state) {
    return swap_adjacent_pairs(swap_adjacent_bits(state));
}

static uint16_t reverse_each_byte(uint16_t state) {
    return swap_nibbles_each_byte(reverse_each_nibble(state));
}

static uint16_t apply_action_op(uint16_t state, ActionOp op) {
    switch (op) {
        case ACTION_OP_SHIFT_LEFT:
            return shift_left(state);
        case ACTION_OP_SHIFT_RIGHT:
            return shift_right(state);
        case ACTION_OP_MIRROR:
            return mirror_bits(state);
        case ACTION_OP_INVERT_RIGHT_7:
            return (uint16_t)(state ^ 0xfe00u);
        case ACTION_OP_SWAP_ADJACENT_BITS:
            return swap_adjacent_bits(state);
        case ACTION_OP_SWAP_ADJACENT_PAIRS:
            return swap_adjacent_pairs(state);
        case ACTION_OP_SWAP_NIBBLES_EACH_BYTE:
            return swap_nibbles_each_byte(state);
        case ACTION_OP_REVERSE_EACH_NIBBLE:
            return reverse_each_nibble(state);
        case ACTION_OP_REVERSE_EACH_BYTE:
            return reverse_each_byte(state);
        default:
            return state;
    }
}

static void build_next_state(void) {
    for (uint32_t state = 0; state < STATE_COUNT; state++) {
        for (int action = 0; action < ACTIVE_ACTION_SET->num_actions; action++) {
            NEXT_STATE[state][action] = apply_action_op(
                (uint16_t)state, ACTIVE_ACTION_SET->ops[action]);
        }
    }
}

static const ActionSet* action_set_by_name(const char* name) {
    for (int i = 0; i < ACTION_SET_COUNT; i++) {
        if (strcmp(ACTION_SETS[i].name, name) == 0) {
            return &ACTION_SETS[i];
        }
    }
    return NULL;
}

static int target_depth_index(int depth) {
    for (int i = 0; i < TARGET_DEPTH_COUNT; i++) {
        if (TARGET_DEPTHS[i] == depth) {
            return i;
        }
    }
    return -1;
}

static int record_worse(const TargetRecord* a, const TargetRecord* b) {
    if (a->score != b->score) {
        return a->score > b->score;
    }
    if (a->start != b->start) {
        return a->start > b->start;
    }
    if (a->target != b->target) {
        return a->target > b->target;
    }
    if (a->packed_actions != b->packed_actions) {
        return a->packed_actions > b->packed_actions;
    }
    return a->depth > b->depth;
}

static int record_better(const TargetRecord* a, const TargetRecord* b) {
    return record_worse(b, a);
}

static void heap_swap(TargetRecord* a, TargetRecord* b) {
    TargetRecord tmp = *a;
    *a = *b;
    *b = tmp;
}

static void heap_sift_up(TargetRecord* records, uint32_t index) {
    while (index > 0) {
        uint32_t parent = (index - 1u) / 2u;
        if (!record_worse(&records[index], &records[parent])) {
            break;
        }
        heap_swap(&records[index], &records[parent]);
        index = parent;
    }
}

static void heap_sift_down(TargetRecord* records, uint32_t count, uint32_t index) {
    while (1) {
        uint32_t left = 2u * index + 1u;
        uint32_t right = left + 1u;
        uint32_t worst = index;
        if (left < count && record_worse(&records[left], &records[worst])) {
            worst = left;
        }
        if (right < count && record_worse(&records[right], &records[worst])) {
            worst = right;
        }
        if (worst == index) {
            break;
        }
        heap_swap(&records[index], &records[worst]);
        index = worst;
    }
}

static int ensure_capacity(DepthSample* sample, uint32_t required) {
    if (required <= sample->capacity) {
        return 0;
    }
    uint32_t next_capacity = sample->capacity == 0 ? 1024u : sample->capacity;
    while (next_capacity < required) {
        if (next_capacity > UINT32_MAX / 2u) {
            return -1;
        }
        next_capacity *= 2u;
    }
    TargetRecord* next = (TargetRecord*)realloc(
        sample->records, (size_t)next_capacity * sizeof(TargetRecord));
    if (next == NULL) {
        return -1;
    }
    sample->records = next;
    sample->capacity = next_capacity;
    return 0;
}

static int add_record(DepthSample* sample, const TargetRecord* record) {
    if (sample->store_all) {
        if (ensure_capacity(sample, sample->count + 1u) != 0) {
            return -1;
        }
        sample->records[sample->count++] = *record;
        return 0;
    }

    if (sample->capacity == 0) {
        return 0;
    }
    if (sample->count < sample->capacity) {
        sample->records[sample->count] = *record;
        heap_sift_up(sample->records, sample->count);
        sample->count += 1u;
        return 0;
    }
    if (record_better(record, &sample->records[0])) {
        sample->records[0] = *record;
        heap_sift_down(sample->records, sample->count, 0);
    }
    return 0;
}

static uint64_t candidate_score(
        uint16_t start,
        uint16_t target,
        int depth,
        uint64_t packed_actions,
        int store_all) {
    uint64_t hash = ACTIVE_ACTION_SET->candidate_score_seed;
    // Store-all depths are complete sets, so keep their ordering stable across
    // sample seeds and only reseed the sampled pools.
    if (!store_all && ACTIVE_SAMPLE_SEED != 0u) {
        hash = mix_u64(hash, ACTIVE_SAMPLE_SEED);
    }
    hash = mix_u64(hash, start);
    hash = mix_u64(hash, target);
    hash = mix_u64(hash, (uint64_t)depth);
    hash = mix_u64(hash, packed_actions);
    return hash;
}

static uint64_t pack_solution(
        uint16_t start,
        uint16_t target,
        uint8_t solution_length,
        const uint16_t* parent,
        const uint8_t* parent_action) {
    uint8_t actions[MAX_DISTANCE];
    uint16_t state = target;
    for (int i = (int)solution_length - 1; i >= 0; i--) {
        actions[i] = parent_action[state];
        state = parent[state];
    }
    if (state != start) {
        fprintf(stderr, "failed to reconstruct path from %u to %u\n",
            (unsigned int)start, (unsigned int)target);
        exit(2);
    }

    uint64_t packed = 0u;
    for (uint8_t i = 0; i < solution_length; i++) {
        packed |= (uint64_t)(actions[i] & 7u) << (3u * i);
    }
    return packed;
}

static void init_worker_result(
        WorkerResult* result,
        const Options* options) {
    memset(result, 0, sizeof(*result));
    for (int i = 0; i < TARGET_DEPTH_COUNT; i++) {
        result->depths[i].depth = TARGET_DEPTHS[i];
        result->depths[i].store_all = options->store_all_depths[i];
        if (!result->depths[i].store_all && options->sample_per_depth > 0) {
            result->depths[i].capacity = options->sample_per_depth;
            result->depths[i].records = (TargetRecord*)calloc(
                options->sample_per_depth, sizeof(TargetRecord));
            if (result->depths[i].records == NULL) {
                fprintf(stderr, "failed to allocate target sampler\n");
                exit(2);
            }
        }
    }
}

static void free_worker_result(WorkerResult* result) {
    for (int i = 0; i < TARGET_DEPTH_COUNT; i++) {
        free(result->depths[i].records);
        result->depths[i].records = NULL;
        result->depths[i].capacity = 0;
        result->depths[i].count = 0;
    }
}

static void compute_worker_records(WorkerResult* result) {
    uint32_t* seen = (uint32_t*)calloc(STATE_COUNT, sizeof(uint32_t));
    uint16_t* queue = (uint16_t*)malloc(STATE_COUNT * sizeof(uint16_t));
    uint16_t* parent = (uint16_t*)malloc(STATE_COUNT * sizeof(uint16_t));
    uint8_t* parent_action = (uint8_t*)malloc(STATE_COUNT * sizeof(uint8_t));
    uint8_t* depth = (uint8_t*)malloc(STATE_COUNT * sizeof(uint8_t));
    if (seen == NULL || queue == NULL || parent == NULL ||
            parent_action == NULL || depth == NULL) {
        fprintf(stderr, "failed to allocate BFS buffers\n");
        exit(2);
    }

#ifdef _OPENMP
    #pragma omp for schedule(dynamic, 64)
#endif
    for (uint32_t start = 0; start < STATE_COUNT; start++) {
        uint32_t stamp = start + 1u;
        uint32_t head = 0;
        uint32_t tail = 0;
        seen[start] = stamp;
        parent[start] = (uint16_t)start;
        parent_action[start] = 0;
        depth[start] = 0;
        queue[tail++] = (uint16_t)start;
        result->histogram[0] += 1u;

        while (head < tail) {
            uint16_t state = queue[head++];
            uint8_t state_depth = depth[state];
            const uint16_t* row = NEXT_STATE[state];
            for (int action = 0; action < ACTIVE_ACTION_SET->num_actions; action++) {
                uint16_t next = row[action];
                if (seen[next] == stamp) {
                    continue;
                }
                uint8_t next_depth = (uint8_t)(state_depth + 1u);
                seen[next] = stamp;
                parent[next] = state;
                parent_action[next] = (uint8_t)action;
                depth[next] = next_depth;
                queue[tail++] = next;
                if (next_depth > MAX_DISTANCE) {
                    fprintf(stderr, "distance exceeded internal limit\n");
                    exit(2);
                }
                result->histogram[next_depth] += 1u;
                if ((int)next_depth > result->max_distance) {
                    result->max_distance = (int)next_depth;
                }

                int depth_index = target_depth_index((int)next_depth);
                if (depth_index < 0) {
                    continue;
                }
                DepthSample* sample = &result->depths[depth_index];
                sample->exact_count += 1u;
                uint64_t packed_actions = pack_solution(
                    (uint16_t)start, next, next_depth, parent, parent_action);
                TargetRecord record;
                memset(&record, 0, sizeof(record));
                record.start = (uint16_t)start;
                record.target = next;
                record.packed_actions = packed_actions;
                record.solution_length = next_depth;
                record.depth = next_depth;
                record.score = candidate_score(
                    (uint16_t)start, next, (int)next_depth, packed_actions,
                    sample->store_all);
                if (add_record(sample, &record) != 0) {
                    fprintf(stderr, "failed to store sampled target record\n");
                    exit(2);
                }
            }
        }

        if (tail != STATE_COUNT) {
            result->disconnected_starts += 1u;
        }
    }

    free(seen);
    free(queue);
    free(parent);
    free(parent_action);
    free(depth);
}

static int compare_records(const void* lhs, const void* rhs) {
    const TargetRecord* a = (const TargetRecord*)lhs;
    const TargetRecord* b = (const TargetRecord*)rhs;
    if (a->depth != b->depth) {
        return (int)a->depth - (int)b->depth;
    }
    if (a->score < b->score) {
        return -1;
    }
    if (a->score > b->score) {
        return 1;
    }
    if (a->start != b->start) {
        return (int)a->start - (int)b->start;
    }
    if (a->target != b->target) {
        return (int)a->target - (int)b->target;
    }
    if (a->packed_actions < b->packed_actions) {
        return -1;
    }
    if (a->packed_actions > b->packed_actions) {
        return 1;
    }
    return 0;
}

static void merge_results(
        WorkerResult* merged,
        WorkerResult* workers,
        int worker_count,
        const Options* options) {
    init_worker_result(merged, options);
    for (int worker_index = 0; worker_index < worker_count; worker_index++) {
        WorkerResult* worker = &workers[worker_index];
        merged->disconnected_starts += worker->disconnected_starts;
        if (worker->max_distance > merged->max_distance) {
            merged->max_distance = worker->max_distance;
        }
        for (int distance = 0; distance <= MAX_DISTANCE; distance++) {
            merged->histogram[distance] += worker->histogram[distance];
        }
        for (int depth_index = 0; depth_index < TARGET_DEPTH_COUNT; depth_index++) {
            DepthSample* dst = &merged->depths[depth_index];
            DepthSample* src = &worker->depths[depth_index];
            dst->exact_count += src->exact_count;
            for (uint32_t i = 0; i < src->count; i++) {
                if (add_record(dst, &src->records[i]) != 0) {
                    fprintf(stderr, "failed to merge sampled target records\n");
                    exit(2);
                }
            }
        }
    }

    for (int depth_index = 0; depth_index < TARGET_DEPTH_COUNT; depth_index++) {
        DepthSample* sample = &merged->depths[depth_index];
        qsort(sample->records, sample->count, sizeof(TargetRecord),
            compare_records);
    }
}

static uint64_t action_set_hash(void) {
    uint64_t hash = 1469598103934665603ull;
    hash = mix_bytes(hash, ACTIVE_ACTION_SET->name);
    hash = mix_u64(hash, BITS);
    hash = mix_u64(hash, ACTIVE_ACTION_SET->num_actions);
    hash = mix_u64(hash, 0xfe00u);
    for (int i = 0; i < ACTIVE_ACTION_SET->num_actions; i++) {
        hash = mix_u64(hash, (uint64_t)i);
        hash = mix_bytes(hash, ACTIVE_ACTION_SET->names[i]);
    }
    return hash;
}

static uint64_t checksum_records(const WorkerResult* result) {
    uint64_t hash = 1469598103934665603ull;
    hash = mix_u64(hash, action_set_hash());
    for (int depth_index = 0; depth_index < TARGET_DEPTH_COUNT; depth_index++) {
        const DepthSample* sample = &result->depths[depth_index];
        hash = mix_u64(hash, (uint64_t)sample->depth);
        hash = mix_u64(hash, sample->exact_count);
        hash = mix_u64(hash, sample->count);
        for (uint32_t i = 0; i < sample->count; i++) {
            const TargetRecord* record = &sample->records[i];
            hash = mix_u64(hash, record->start);
            hash = mix_u64(hash, record->target);
            hash = mix_u64(hash, record->packed_actions);
            hash = mix_u64(hash, record->solution_length);
            hash = mix_u64(hash, record->depth);
        }
    }
    return hash;
}

static int write_bytes(FILE* file, const void* data, size_t size) {
    return fwrite(data, 1, size, file) == size ? 0 : -1;
}

static int write_u16(FILE* file, uint16_t value) {
    unsigned char bytes[2] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
    };
    return write_bytes(file, bytes, sizeof(bytes));
}

static int write_u32(FILE* file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
        (unsigned char)((value >> 16) & 0xffu),
        (unsigned char)((value >> 24) & 0xffu),
    };
    return write_bytes(file, bytes, sizeof(bytes));
}

static int write_u64(FILE* file, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; i++) {
        bytes[i] = (unsigned char)((value >> (8 * i)) & 0xffu);
    }
    return write_bytes(file, bytes, sizeof(bytes));
}

static uint32_t total_record_count(const WorkerResult* result) {
    uint64_t count = 0;
    for (int depth_index = 0; depth_index < TARGET_DEPTH_COUNT; depth_index++) {
        count += result->depths[depth_index].count;
    }
    if (count > UINT32_MAX) {
        fprintf(stderr, "too many target records for binary format\n");
        exit(2);
    }
    return (uint32_t)count;
}

static uint32_t header_size(void) {
    return 52u + (uint32_t)TARGET_DEPTH_COUNT * 24u;
}

static int write_binary(const char* path, const WorkerResult* result) {
    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }

    const unsigned char magic[8] = {'A', 'L', '7', 'T', 'G', 'T', '1', '\0'};
    uint32_t record_count = total_record_count(result);
    uint64_t checksum = checksum_records(result);
    uint64_t set_hash = action_set_hash();
    int rc = 0;
    rc |= write_bytes(file, magic, sizeof(magic));
    rc |= write_u32(file, FORMAT_VERSION);
    rc |= write_u32(file, header_size());
    rc |= write_u32(file, RECORD_SIZE);
    rc |= write_u32(file, BITS);
    rc |= write_u32(file, (uint32_t)ACTIVE_ACTION_SET->num_actions);
    rc |= write_u32(file, TARGET_DEPTH_COUNT);
    rc |= write_u32(file, record_count);
    rc |= write_u64(file, checksum);
    rc |= write_u64(file, set_hash);

    uint32_t first_record = 0;
    for (int depth_index = 0; depth_index < TARGET_DEPTH_COUNT; depth_index++) {
        const DepthSample* sample = &result->depths[depth_index];
        rc |= write_u32(file, (uint32_t)sample->depth);
        rc |= write_u32(file, first_record);
        rc |= write_u32(file, sample->count);
        rc |= write_u32(file, 0u);
        rc |= write_u64(file, sample->exact_count);
        first_record += sample->count;
    }

    for (int depth_index = 0; depth_index < TARGET_DEPTH_COUNT; depth_index++) {
        const DepthSample* sample = &result->depths[depth_index];
        for (uint32_t i = 0; i < sample->count; i++) {
            const TargetRecord* record = &sample->records[i];
            rc |= write_u16(file, record->start);
            rc |= write_u16(file, record->target);
            rc |= write_u64(file, record->packed_actions);
            rc |= fputc(record->solution_length, file) == EOF ? -1 : 0;
            rc |= fputc(record->depth, file) == EOF ? -1 : 0;
            rc |= write_u16(file, 0u);
        }
    }

    if (fclose(file) != 0) {
        fprintf(stderr, "failed to close %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (rc != 0) {
        fprintf(stderr, "failed to write %s\n", path);
        return -1;
    }
    return 0;
}

static int write_json(const char* path, const WorkerResult* result,
        const Options* options) {
    FILE* file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }

    uint32_t record_count = total_record_count(result);
    uint64_t checksum = checksum_records(result);
    uint64_t set_hash = action_set_hash();

    fprintf(file, "{\n");
    fprintf(file, "  \"action_id_to_name\": [\n");
    for (int i = 0; i < ACTIVE_ACTION_SET->num_actions; i++) {
        fprintf(file, "    \"%s\"%s\n", ACTIVE_ACTION_SET->names[i],
            i == ACTIVE_ACTION_SET->num_actions - 1 ? "" : ",");
    }
    fprintf(file, "  ],\n");
    fprintf(file, "  \"action_set\": \"%s\",\n", ACTIVE_ACTION_SET->name);
    fprintf(file, "  \"action_set_hash\": \"0x%016llx\",\n",
        (unsigned long long)set_hash);
    fprintf(file, "  \"binary_path\": \"%s\",\n", options->output_bin);
    fprintf(file, "  \"bits\": %d,\n", BITS);
    fprintf(file, "  \"checksum\": \"0x%016llx\",\n",
        (unsigned long long)checksum);
    fprintf(file, "  \"depth_records\": [\n");
    uint32_t first_record = 0;
    for (int depth_index = 0; depth_index < TARGET_DEPTH_COUNT; depth_index++) {
        const DepthSample* sample = &result->depths[depth_index];
        fprintf(file,
            "    {\"depth\": %d, \"exact_pair_count\": %llu, "
            "\"first_record\": %u, \"stored_count\": %u}%s\n",
            sample->depth,
            (unsigned long long)sample->exact_count,
            first_record,
            sample->count,
            depth_index == TARGET_DEPTH_COUNT - 1 ? "" : ",");
        first_record += sample->count;
    }
    fprintf(file, "  ],\n");
    fprintf(file, "  \"depths\": [");
    for (int i = 0; i < TARGET_DEPTH_COUNT; i++) {
        fprintf(file, "%s%d", i == 0 ? "" : ", ", TARGET_DEPTHS[i]);
    }
    fprintf(file, "],\n");
    fprintf(file, "  \"disconnected_starts\": %llu,\n",
        (unsigned long long)result->disconnected_starts);
    fprintf(file, "  \"format\": \"affine_lock_visible_targets_bin\",\n");
    fprintf(file, "  \"header_size\": %u,\n", header_size());
    fprintf(file, "  \"max_distance\": %d,\n", result->max_distance);
    fprintf(file, "  \"num_actions\": %d,\n", ACTIVE_ACTION_SET->num_actions);
    fprintf(file, "  \"record_count\": %u,\n", record_count);
    fprintf(file, "  \"record_size\": %d,\n", RECORD_SIZE);
    fprintf(file, "  \"sample_per_depth\": %u,\n",
        options->sample_per_depth);
    fprintf(file, "  \"sample_seed\": %llu,\n",
        (unsigned long long)options->sample_seed);
    fprintf(file, "  \"stored_all_depths\": [");
    int wrote_depth = 0;
    for (int i = 0; i < TARGET_DEPTH_COUNT; i++) {
        if (!options->store_all_depths[i]) {
            continue;
        }
        fprintf(file, "%s%d", wrote_depth ? ", " : "", TARGET_DEPTHS[i]);
        wrote_depth = 1;
    }
    fprintf(file, "],\n");
    fprintf(file, "  \"version\": %d,\n", FORMAT_VERSION);
    fprintf(file, "  \"visible_distance_histogram\": {\n");
    int first = 1;
    for (int distance = 0; distance <= result->max_distance; distance++) {
        if (!first) {
            fprintf(file, ",\n");
        }
        fprintf(file, "    \"%d\": %llu", distance,
            (unsigned long long)result->histogram[distance]);
        first = 0;
    }
    fprintf(file, "\n  }\n");
    fprintf(file, "}\n");

    if (fclose(file) != 0) {
        fprintf(stderr, "failed to close %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

static int parse_uint32(const char* text, uint32_t* out) {
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX) {
        return -1;
    }
    *out = (uint32_t)value;
    return 0;
}

static int parse_uint64(const char* text, uint64_t* out) {
    char* end = NULL;
    errno = 0;
    if (text[0] == '-') {
        return -1;
    }
    unsigned long long value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }
    *out = (uint64_t)value;
    return 0;
}

static void print_usage(const char* program) {
    fprintf(stderr,
        "usage: %s [--action-set NAME] [--sample-per-depth N] "
        "[--sample-seed N] [--store-all-depth D] "
        "[--output-bin PATH] [--output-json PATH]\n",
        program);
    fprintf(stderr, "available action sets:");
    for (int i = 0; i < ACTION_SET_COUNT; i++) {
        fprintf(stderr, " %s", ACTION_SETS[i].name);
    }
    fprintf(stderr, "\n");
}

static int parse_args(int argc, char** argv, Options* options) {
    options->action_set = &ACTION_SETS[0];
    options->output_bin = NULL;
    options->output_json = NULL;
    options->sample_per_depth = 65536u;
    options->sample_seed = 0u;
    memset(options->store_all_depths, 0, sizeof(options->store_all_depths));
    options->output_bin_explicit = 0;
    options->output_json_explicit = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--action-set") == 0 && i + 1 < argc) {
            const ActionSet* action_set = action_set_by_name(argv[++i]);
            if (action_set == NULL) {
                fprintf(stderr, "unknown --action-set %s\n", argv[i]);
                return -1;
            }
            options->action_set = action_set;
        } else if (strcmp(argv[i], "--sample-per-depth") == 0 && i + 1 < argc) {
            if (parse_uint32(argv[++i], &options->sample_per_depth) != 0) {
                fprintf(stderr, "invalid --sample-per-depth value\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--sample-seed") == 0 && i + 1 < argc) {
            if (parse_uint64(argv[++i], &options->sample_seed) != 0) {
                fprintf(stderr, "invalid --sample-seed value\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--store-all-depth") == 0 && i + 1 < argc) {
            uint32_t depth = 0;
            if (parse_uint32(argv[++i], &depth) != 0) {
                fprintf(stderr, "invalid --store-all-depth value\n");
                return -1;
            }
            int depth_index = target_depth_index((int)depth);
            if (depth_index < 0) {
                fprintf(stderr, "unsupported --store-all-depth %u\n", depth);
                return -1;
            }
            options->store_all_depths[depth_index] = 1;
        } else if (strcmp(argv[i], "--output-bin") == 0 && i + 1 < argc) {
            options->output_bin = argv[++i];
            options->output_bin_explicit = 1;
        } else if (strcmp(argv[i], "--output-json") == 0 && i + 1 < argc) {
            options->output_json = argv[++i];
            options->output_json_explicit = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            print_usage(argv[0]);
            return -1;
        }
    }

    if (options->output_bin == NULL || !options->output_bin_explicit) {
        options->output_bin = options->action_set->default_bin;
    }
    if (options->output_json == NULL || !options->output_json_explicit) {
        options->output_json = options->action_set->default_json;
    }
    if (options->action_set->store_all_d16_by_default) {
        options->store_all_depths[target_depth_index(16)] = 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    Options options;
    if (parse_args(argc, argv, &options) != 0) {
        return 1;
    }

    ACTIVE_ACTION_SET = options.action_set;
    ACTIVE_SAMPLE_SEED = options.sample_seed;
    build_next_state();
    int worker_count = 1;
#ifdef _OPENMP
    worker_count = omp_get_max_threads();
#endif
    WorkerResult* workers =
        (WorkerResult*)calloc((size_t)worker_count, sizeof(WorkerResult));
    if (workers == NULL) {
        fprintf(stderr, "failed to allocate worker results\n");
        return 1;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int worker_index = 0;
#ifdef _OPENMP
        worker_index = omp_get_thread_num();
#endif
        init_worker_result(&workers[worker_index], &options);
        compute_worker_records(&workers[worker_index]);
    }

    WorkerResult merged;
    merge_results(&merged, workers, worker_count, &options);
    int rc = 0;
    if (write_binary(options.output_bin, &merged) != 0) {
        rc = 1;
    }
    if (write_json(options.output_json, &merged, &options) != 0) {
        rc = 1;
    }
    for (int i = 0; i < worker_count; i++) {
        free_worker_result(&workers[i]);
    }
    free(workers);
    free_worker_result(&merged);
    return rc == 0 ? 0 : 1;
}
