#ifndef PUFFERLIB_OCEAN_BOXOBAN_PARSE_MAPS_H
#define PUFFERLIB_OCEAN_BOXOBAN_PARSE_MAPS_H

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define BOXOBAN_AGENT_CHAR '@'
#define BOXOBAN_WALL_CHAR '#'
#define BOXOBAN_BOX_CHAR '$'
#define BOXOBAN_TARGET_CHAR '.'
#define BOXOBAN_BOX_ON_TARGET_CHAR '*'
#define BOXOBAN_AGENT_ON_TARGET_CHAR '+'

#define BOXOBAN_EXPECTED_ROWS 10
#define BOXOBAN_EXPECTED_COLS 10
#define BOXOBAN_PUZZLE_OBS_BYTES (4 * BOXOBAN_EXPECTED_ROWS * BOXOBAN_EXPECTED_COLS)
#define BOXOBAN_PUZZLE_META_BYTES 5
#define BOXOBAN_PUZZLE_BYTES (BOXOBAN_PUZZLE_OBS_BYTES + BOXOBAN_PUZZLE_META_BYTES)

typedef struct {
    char rows[BOXOBAN_EXPECTED_ROWS][BOXOBAN_EXPECTED_COLS];
    int row_lengths[BOXOBAN_EXPECTED_ROWS];
    int row_count;
} BoxobanPuzzleDraft;

static int boxoban_is_blank_line(const char* line) {
    const unsigned char* p = (const unsigned char*)line;
    while (*p != '\0') {
        if (!isspace(*p)) {
            return 0;
        }
        p++;
    }
    return 1;
}

static int boxoban_validate_shape(const BoxobanPuzzleDraft* draft, char* reason, size_t reason_cap) {
    if (draft->row_count != BOXOBAN_EXPECTED_ROWS) {
        snprintf(reason, reason_cap, "expected %d rows, got %d", BOXOBAN_EXPECTED_ROWS, draft->row_count);
        return -1;
    }

    for (int r = 0; r < BOXOBAN_EXPECTED_ROWS; r++) {
        if (draft->row_lengths[r] != BOXOBAN_EXPECTED_COLS) {
            snprintf(reason, reason_cap, "row %d expected %d cols, got %d",
                r, BOXOBAN_EXPECTED_COLS, draft->row_lengths[r]);
            return -1;
        }
    }

    reason[0] = '\0';
    return 0;
}

static int boxoban_encode_and_write_puzzle(const BoxobanPuzzleDraft* draft, FILE* out, char* reason, size_t reason_cap) {
    uint8_t agent[BOXOBAN_EXPECTED_ROWS * BOXOBAN_EXPECTED_COLS] = {0};
    uint8_t walls[BOXOBAN_EXPECTED_ROWS * BOXOBAN_EXPECTED_COLS] = {0};
    uint8_t boxes[BOXOBAN_EXPECTED_ROWS * BOXOBAN_EXPECTED_COLS] = {0};
    uint8_t targets[BOXOBAN_EXPECTED_ROWS * BOXOBAN_EXPECTED_COLS] = {0};
    uint8_t meta[BOXOBAN_PUZZLE_META_BYTES] = {0};

    int agent_x = -1;
    int agent_y = -1;
    int n_boxes = 0;
    int n_targets = 0;
    int on_target = 0;

    int idx = 0;
    for (int r = 0; r < BOXOBAN_EXPECTED_ROWS; r++) {
        for (int c = 0; c < BOXOBAN_EXPECTED_COLS; c++, idx++) {
            char ch = draft->rows[r][c];

            int is_agent = (ch == BOXOBAN_AGENT_CHAR || ch == BOXOBAN_AGENT_ON_TARGET_CHAR);
            int is_wall = (ch == BOXOBAN_WALL_CHAR);
            int is_box = (ch == BOXOBAN_BOX_CHAR || ch == BOXOBAN_BOX_ON_TARGET_CHAR);
            int is_target = (ch == BOXOBAN_TARGET_CHAR || ch == BOXOBAN_BOX_ON_TARGET_CHAR || ch == BOXOBAN_AGENT_ON_TARGET_CHAR);

            if (is_agent) {
                if (agent_x != -1) {
                    snprintf(reason, reason_cap, "Puzzle has multiple agents");
                    return -1;
                }
                agent_x = c;
                agent_y = r;
            }

            n_boxes += is_box;
            n_targets += is_target;
            on_target += (is_box && is_target);

            agent[idx] = (uint8_t)is_agent;
            walls[idx] = (uint8_t)is_wall;
            boxes[idx] = (uint8_t)is_box;
            targets[idx] = (uint8_t)is_target;
        }
    }

    if (agent_x == -1) {
        snprintf(reason, reason_cap, "Puzzle has no agent");
        return -1;
    }

    meta[0] = (uint8_t)agent_x;
    meta[1] = (uint8_t)agent_y;
    meta[2] = (uint8_t)n_boxes;
    meta[3] = (uint8_t)n_targets;
    meta[4] = (uint8_t)on_target;

    if (fwrite(agent, 1, sizeof(agent), out) != sizeof(agent)) return -1;
    if (fwrite(walls, 1, sizeof(walls), out) != sizeof(walls)) return -1;
    if (fwrite(boxes, 1, sizeof(boxes), out) != sizeof(boxes)) return -1;
    if (fwrite(targets, 1, sizeof(targets), out) != sizeof(targets)) return -1;
    if (fwrite(meta, 1, sizeof(meta), out) != sizeof(meta)) return -1;

    reason[0] = '\0';
    return 0;
}

static int boxoban_finalize_puzzle(
    BoxobanPuzzleDraft* draft,
    FILE* out,
    const char* src_path,
    size_t* puzzle_idx,
    size_t* written_count
) {
    char reason[128];
    size_t idx = *puzzle_idx;
    (*puzzle_idx)++;

    if (boxoban_validate_shape(draft, reason, sizeof(reason)) != 0) {
        fprintf(stdout, "[Boxoban] Skipping malformed puzzle in %s puzzle#%zu: %s\n", src_path, idx, reason);
        draft->row_count = 0;
        return 0;
    }

    if (boxoban_encode_and_write_puzzle(draft, out, reason, sizeof(reason)) != 0) {
        if (reason[0] == '\0') {
            return -1;
        }
        fprintf(stdout, "[Boxoban] Skipping malformed puzzle in %s puzzle#%zu: %s\n", src_path, idx, reason);
        draft->row_count = 0;
        return 0;
    }

    (*written_count)++;
    draft->row_count = 0;
    return 0;
}

static int boxoban_write_bin_from_files(
    const char* const* files,
    size_t file_count,
    const char* out_path,
    int verbose,
    size_t* out_puzzle_count
) {
    FILE* out = fopen(out_path, "wb");
    if (out == NULL) {
        return -1;
    }

    size_t puzzle_count = 0;

    for (size_t file_idx = 0; file_idx < file_count; file_idx++) {
        const char* src_path = files[file_idx];
        FILE* in = fopen(src_path, "r");
        if (in == NULL) {
            fclose(out);
            return -1;
        }

        char* line = NULL;
        size_t line_cap = 0;
        ssize_t line_len;
        BoxobanPuzzleDraft draft;
        memset(&draft, 0, sizeof(draft));
        size_t puzzle_idx = 0;

        while ((line_len = getline(&line, &line_cap, in)) != -1) {
            if (line_len > 0 && line[line_len - 1] == '\n') {
                line[--line_len] = '\0';
            }

            if (line[0] == ';') {
                if (draft.row_count > 0) {
                    if (boxoban_finalize_puzzle(&draft, out, src_path, &puzzle_idx, &puzzle_count) != 0) {
                        free(line);
                        fclose(in);
                        fclose(out);
                        return -1;
                    }
                }
                continue;
            }

            if (boxoban_is_blank_line(line)) {
                continue;
            }

            if (draft.row_count < BOXOBAN_EXPECTED_ROWS) {
                int dst_row = draft.row_count;
                int copy_len = line_len < BOXOBAN_EXPECTED_COLS ? (int)line_len : BOXOBAN_EXPECTED_COLS;
                memcpy(draft.rows[dst_row], line, (size_t)copy_len);
                draft.row_lengths[dst_row] = (int)line_len;
                draft.row_count++;
            }

            if (draft.row_count == BOXOBAN_EXPECTED_ROWS) {
                if (boxoban_finalize_puzzle(&draft, out, src_path, &puzzle_idx, &puzzle_count) != 0) {
                    free(line);
                    fclose(in);
                    fclose(out);
                    return -1;
                }
            }
        }

        free(line);
        fclose(in);
    }

    if (fflush(out) != 0) {
        fclose(out);
        return -1;
    }

    long bytes_written = ftell(out);
    fclose(out);
    if (bytes_written < 0) {
        return -1;
    }

    size_t expected = puzzle_count * BOXOBAN_PUZZLE_BYTES;
    if ((size_t)bytes_written != expected) {
        fprintf(stderr, "Wrong output size: got %ld expected %zu\n", bytes_written, expected);
        return -1;
    }

    if (verbose) {
        fprintf(stdout, "Wrote %zu puzzles to %s\n", puzzle_count, out_path);
    }
    if (out_puzzle_count != NULL) {
        *out_puzzle_count = puzzle_count;
    }
    return 0;
}

#endif
