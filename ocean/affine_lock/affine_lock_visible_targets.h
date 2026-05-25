#pragma once

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AFFINE_LOCK_VISIBLE_TARGET_FORMAT_VERSION 1u
#define AFFINE_LOCK_VISIBLE_TARGET_RECORD_SIZE 16u
#define AFFINE_LOCK_VISIBLE_TARGET_8ACTION_V1_HASH 0x6e11e18fdafc0baaull

typedef struct AffineLockVisibleTargetDepth {
    uint32_t depth;
    uint32_t first_record;
    uint32_t stored_count;
    uint64_t exact_pair_count;
} AffineLockVisibleTargetDepth;

typedef struct AffineLockVisibleTargetRecord {
    uint16_t start;
    uint16_t target;
    uint64_t packed_actions;
    uint8_t solution_length;
    uint8_t depth;
} AffineLockVisibleTargetRecord;

typedef struct AffineLockVisibleTargetTable {
    uint32_t version;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t bits;
    uint32_t num_actions;
    uint32_t depth_count;
    uint32_t record_count;
    uint64_t checksum;
    uint64_t action_set_hash;
    AffineLockVisibleTargetDepth* depths;
    AffineLockVisibleTargetRecord* records;
} AffineLockVisibleTargetTable;

static uint64_t affine_lock_visible_targets_mix_u64(
        uint64_t hash,
        uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static void affine_lock_visible_targets_set_error(
        char* error,
        size_t error_size,
        const char* format,
        ...) {
    if (error == NULL || error_size == 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    vsnprintf(error, error_size, format, args);
    va_end(args);
}

static int affine_lock_visible_targets_read_exact(
        FILE* file,
        void* out,
        size_t size) {
    return fread(out, 1, size, file) == size ? 0 : -1;
}

static int affine_lock_visible_targets_read_u16(
        FILE* file,
        uint16_t* out) {
    unsigned char bytes[2];
    if (affine_lock_visible_targets_read_exact(file, bytes, sizeof(bytes)) != 0) {
        return -1;
    }
    *out = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
    return 0;
}

static int affine_lock_visible_targets_read_u32(
        FILE* file,
        uint32_t* out) {
    unsigned char bytes[4];
    if (affine_lock_visible_targets_read_exact(file, bytes, sizeof(bytes)) != 0) {
        return -1;
    }
    *out = (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
    return 0;
}

static int affine_lock_visible_targets_read_u64(
        FILE* file,
        uint64_t* out) {
    unsigned char bytes[8];
    if (affine_lock_visible_targets_read_exact(file, bytes, sizeof(bytes)) != 0) {
        return -1;
    }
    uint64_t value = 0;
    for (int i = 0; i < 8; i++) {
        value |= (uint64_t)bytes[i] << (8 * i);
    }
    *out = value;
    return 0;
}

static void affine_lock_visible_targets_free(
        AffineLockVisibleTargetTable* table) {
    if (table == NULL) {
        return;
    }
    free(table->depths);
    free(table->records);
    memset(table, 0, sizeof(*table));
}

static uint64_t affine_lock_visible_targets_checksum(
        const AffineLockVisibleTargetTable* table) {
    uint64_t hash = 1469598103934665603ull;
    hash = affine_lock_visible_targets_mix_u64(hash, table->action_set_hash);
    for (uint32_t depth_index = 0; depth_index < table->depth_count;
            depth_index++) {
        const AffineLockVisibleTargetDepth* depth = &table->depths[depth_index];
        hash = affine_lock_visible_targets_mix_u64(hash, depth->depth);
        hash = affine_lock_visible_targets_mix_u64(hash, depth->exact_pair_count);
        hash = affine_lock_visible_targets_mix_u64(hash, depth->stored_count);
        for (uint32_t i = 0; i < depth->stored_count; i++) {
            uint32_t record_index = depth->first_record + i;
            const AffineLockVisibleTargetRecord* record =
                &table->records[record_index];
            hash = affine_lock_visible_targets_mix_u64(hash, record->start);
            hash = affine_lock_visible_targets_mix_u64(hash, record->target);
            hash = affine_lock_visible_targets_mix_u64(
                hash, record->packed_actions);
            hash = affine_lock_visible_targets_mix_u64(
                hash, record->solution_length);
            hash = affine_lock_visible_targets_mix_u64(hash, record->depth);
        }
    }
    return hash;
}

static int affine_lock_visible_targets_load(
        const char* path,
        uint64_t expected_action_set_hash,
        AffineLockVisibleTargetTable* table,
        char* error,
        size_t error_size) {
    static const unsigned char expected_magic[8] = {
        'A', 'L', '7', 'T', 'G', 'T', '1', '\0'
    };
    memset(table, 0, sizeof(*table));

    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        affine_lock_visible_targets_set_error(
            error, error_size, "failed to open %s: %s", path, strerror(errno));
        return -1;
    }

    unsigned char magic[8];
    if (affine_lock_visible_targets_read_exact(file, magic, sizeof(magic)) != 0 ||
            affine_lock_visible_targets_read_u32(file, &table->version) != 0 ||
            affine_lock_visible_targets_read_u32(file, &table->header_size) != 0 ||
            affine_lock_visible_targets_read_u32(file, &table->record_size) != 0 ||
            affine_lock_visible_targets_read_u32(file, &table->bits) != 0 ||
            affine_lock_visible_targets_read_u32(file, &table->num_actions) != 0 ||
            affine_lock_visible_targets_read_u32(file, &table->depth_count) != 0 ||
            affine_lock_visible_targets_read_u32(file, &table->record_count) != 0 ||
            affine_lock_visible_targets_read_u64(file, &table->checksum) != 0 ||
            affine_lock_visible_targets_read_u64(file, &table->action_set_hash) != 0) {
        affine_lock_visible_targets_set_error(
            error, error_size, "truncated visible target header");
        fclose(file);
        return -1;
    }

    if (memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        affine_lock_visible_targets_set_error(
            error, error_size, "invalid visible target magic");
        fclose(file);
        return -1;
    }
    if (table->version != AFFINE_LOCK_VISIBLE_TARGET_FORMAT_VERSION ||
            table->record_size != AFFINE_LOCK_VISIBLE_TARGET_RECORD_SIZE ||
            table->bits != 16 ||
            table->num_actions == 0 ||
            table->num_actions > 8 ||
            table->depth_count == 0 ||
            table->depth_count > 16) {
        affine_lock_visible_targets_set_error(
            error, error_size, "unsupported visible target table header");
        fclose(file);
        return -1;
    }
    uint32_t expected_header_size = 52u + table->depth_count * 24u;
    if (table->header_size != expected_header_size) {
        affine_lock_visible_targets_set_error(
            error, error_size, "unexpected visible target header size");
        fclose(file);
        return -1;
    }
    if (expected_action_set_hash != 0 &&
            table->action_set_hash != expected_action_set_hash) {
        affine_lock_visible_targets_set_error(
            error, error_size, "visible target action set hash mismatch");
        fclose(file);
        return -1;
    }

    table->depths = (AffineLockVisibleTargetDepth*)calloc(
        table->depth_count, sizeof(AffineLockVisibleTargetDepth));
    table->records = (AffineLockVisibleTargetRecord*)calloc(
        table->record_count, sizeof(AffineLockVisibleTargetRecord));
    if (table->depths == NULL || table->records == NULL) {
        affine_lock_visible_targets_set_error(
            error, error_size, "failed to allocate visible target table");
        fclose(file);
        affine_lock_visible_targets_free(table);
        return -1;
    }

    uint64_t depth_record_total = 0;
    for (uint32_t i = 0; i < table->depth_count; i++) {
        AffineLockVisibleTargetDepth* depth = &table->depths[i];
        uint32_t reserved = 0;
        if (affine_lock_visible_targets_read_u32(file, &depth->depth) != 0 ||
                affine_lock_visible_targets_read_u32(
                    file, &depth->first_record) != 0 ||
                affine_lock_visible_targets_read_u32(
                    file, &depth->stored_count) != 0 ||
                affine_lock_visible_targets_read_u32(file, &reserved) != 0 ||
                affine_lock_visible_targets_read_u64(
                    file, &depth->exact_pair_count) != 0) {
            affine_lock_visible_targets_set_error(
                error, error_size, "truncated visible target depth table");
            fclose(file);
            affine_lock_visible_targets_free(table);
            return -1;
        }
        if (reserved != 0 ||
                depth->first_record > table->record_count ||
                depth->stored_count > table->record_count ||
                depth->first_record + depth->stored_count >
                    table->record_count) {
            affine_lock_visible_targets_set_error(
                error, error_size, "invalid visible target depth table");
            fclose(file);
            affine_lock_visible_targets_free(table);
            return -1;
        }
        depth_record_total += depth->stored_count;
    }
    if (depth_record_total != table->record_count) {
        affine_lock_visible_targets_set_error(
            error, error_size, "visible target depth counts do not sum");
        fclose(file);
        affine_lock_visible_targets_free(table);
        return -1;
    }

    for (uint32_t i = 0; i < table->record_count; i++) {
        AffineLockVisibleTargetRecord* record = &table->records[i];
        uint16_t reserved = 0;
        if (affine_lock_visible_targets_read_u16(file, &record->start) != 0 ||
                affine_lock_visible_targets_read_u16(file, &record->target) != 0 ||
                affine_lock_visible_targets_read_u64(
                    file, &record->packed_actions) != 0) {
            affine_lock_visible_targets_set_error(
                error, error_size, "truncated visible target record");
            fclose(file);
            affine_lock_visible_targets_free(table);
            return -1;
        }
        int solution_length = fgetc(file);
        int depth = fgetc(file);
        if (solution_length == EOF || depth == EOF ||
                affine_lock_visible_targets_read_u16(file, &reserved) != 0) {
            affine_lock_visible_targets_set_error(
                error, error_size, "truncated visible target record");
            fclose(file);
            affine_lock_visible_targets_free(table);
            return -1;
        }
        record->solution_length = (uint8_t)solution_length;
        record->depth = (uint8_t)depth;
        if (reserved != 0 || record->solution_length != record->depth) {
            affine_lock_visible_targets_set_error(
                error, error_size, "invalid visible target record");
            fclose(file);
            affine_lock_visible_targets_free(table);
            return -1;
        }
    }

    int extra = fgetc(file);
    if (extra != EOF) {
        affine_lock_visible_targets_set_error(
            error, error_size, "visible target file has trailing bytes");
        fclose(file);
        affine_lock_visible_targets_free(table);
        return -1;
    }
    fclose(file);

    uint64_t computed_checksum =
        affine_lock_visible_targets_checksum(table);
    if (computed_checksum != table->checksum) {
        affine_lock_visible_targets_set_error(
            error, error_size,
            "visible target checksum mismatch: got 0x%016llx expected 0x%016llx",
            (unsigned long long)computed_checksum,
            (unsigned long long)table->checksum);
        affine_lock_visible_targets_free(table);
        return -1;
    }
    return 0;
}
