/**
 * @fileoverview OSRS animation runtime — loads .anims binary, applies vertex-group
 * transforms to model base geometry, re-expands into raylib mesh for rendering.
 *
 * OSRS animations use vertex-group-based transforms (not bones). Each vertex has a
 * skin label (group index). FrameBase defines transform slots with types + label arrays.
 * Each frame provides per-slot {dx,dy,dz} values. Transform types:
 *   0 = origin (compute centroid of referenced vertex groups → set pivot)
 *   1 = translate (add dx/dy/dz to all vertices in referenced groups)
 *   2 = rotate (euler Z-X-Y around pivot, raw*8 → 2048-entry sine table)
 *   3 = scale (relative to pivot, 128 = 1.0x identity)
 *   5 = alpha (face transparency, not used in our viewer)
 *
 * Binary format (.anims) produced by tools/cache_pipeline/export_animations.py:
 *   legacy header: uint32 magic ("MINA"), uint16 framebase_count,
 *                  uint16 sequence_count
 *   current header: char[4] magic ("ANM2"), uint16 version,
 *                   uint16 header_size, uint32 framebase_count,
 *                   uint32 sequence_count, uint32 sequence_frame_count,
 *                   uint32 flags
 *   framebases section, sequences section with inlined frame data.
 */

#include "fc_anim_loader.h"
#include "fc_assets.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ANIM_MAGIC 0x414E494D  /* legacy bytes "MINA" when read little-endian */
#define ANIM2_MAGIC 0x324D4E41 /* bytes "ANM2" when read little-endian */
#define ANIM2_MIN_VERSION 2
#define ANIM2_VERSION 3
#define ANIM2_HEADER_SIZE 24
#define ANIM_MAX_SLOTS 256
#define ANIM_SINE_COUNT 2048

/* ======================================================================== */
/* sine/cosine table (matches OSRS Rasterizer3D, fixed-point scale 65536)     */
/* ======================================================================== */

static int anim_sine[ANIM_SINE_COUNT];
static int anim_cosine[ANIM_SINE_COUNT];
static int anim_trig_initialized = 0;

static void anim_init_trig(void) {
    if (anim_trig_initialized) return;
    for (int i = 0; i < ANIM_SINE_COUNT; i++) {
        double angle = (double)i * (2.0 * 3.14159265358979323846 / ANIM_SINE_COUNT);
        anim_sine[i] = (int)(65536.0 * sin(angle));
        anim_cosine[i] = (int)(65536.0 * cos(angle));
    }
    anim_trig_initialized = 1;
}

/* ======================================================================== */
/* loading                                                                    */
/* ======================================================================== */

typedef struct {
    const uint8_t* p;
    size_t remaining;
    const char* path;
} AnimCursor;

static int anim_take(AnimCursor* c, void* dst, size_t size, const char* what) {
    if (!c || c->remaining < size) {
        fprintf(stderr, "%s: short read while loading %s\n",
                c && c->path ? c->path : "(anim)", what);
        return 0;
    }
    if (dst) memcpy(dst, c->p, size);
    c->p += size;
    c->remaining -= size;
    return 1;
}

static int anim_read_u8(AnimCursor* c, uint8_t* out, const char* what) {
    return anim_take(c, out, 1, what);
}

static int anim_read_u16(AnimCursor* c, uint16_t* out, const char* what) {
    uint8_t b[2];
    if (!anim_take(c, b, sizeof(b), what)) return 0;
    *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
    return 1;
}

static int anim_read_i16(AnimCursor* c, int16_t* out, const char* what) {
    uint16_t u = 0;
    if (!anim_read_u16(c, &u, what)) return 0;
    *out = (int16_t)u;
    return 1;
}

static int anim_read_i8(AnimCursor* c, int8_t* out, const char* what) {
    uint8_t u = 0;
    if (!anim_read_u8(c, &u, what)) return 0;
    *out = (int8_t)u;
    return 1;
}

static int anim_read_u32(AnimCursor* c, uint32_t* out, const char* what) {
    uint8_t b[4];
    if (!anim_take(c, b, sizeof(b), what)) return 0;
    *out = (uint32_t)b[0]
         | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
    return 1;
}

AnimCache* anim_cache_load(const char* path) {
    size_t size = 0;
    uint8_t* buf = fc_asset_read_all(path, &size);
    AnimCursor cur;
    uint32_t magic = 0;
    uint32_t base_count = 0;
    uint32_t seq_count = 0;
    uint16_t format_version = 1;
    AnimCache* cache;

    if (!buf) return NULL;
    cur.p = buf;
    cur.remaining = size;
    cur.path = path;

    if (!anim_read_u32(&cur, &magic, "anim magic")) {
        free(buf);
        return NULL;
    }
    if (magic == ANIM_MAGIC) {
        uint16_t legacy_base_count = 0;
        uint16_t legacy_seq_count = 0;
        if (!anim_read_u16(&cur, &legacy_base_count, "framebase count") ||
            !anim_read_u16(&cur, &legacy_seq_count, "sequence count")) {
            free(buf);
            return NULL;
        }
        base_count = legacy_base_count;
        seq_count = legacy_seq_count;
    } else if (magic == ANIM2_MAGIC) {
        uint16_t version = 0;
        uint16_t header_size = 0;
        uint32_t sequence_frame_count = 0;
        uint32_t flags = 0;
        (void)sequence_frame_count;
        (void)flags;
        if (!anim_read_u16(&cur, &version, "anim version") ||
            !anim_read_u16(&cur, &header_size, "anim header size") ||
            !anim_read_u32(&cur, &base_count, "framebase count") ||
            !anim_read_u32(&cur, &seq_count, "sequence count") ||
            !anim_read_u32(&cur, &sequence_frame_count, "sequence frame count") ||
            !anim_read_u32(&cur, &flags, "anim flags")) {
            free(buf);
            return NULL;
        }
        if (version < ANIM2_MIN_VERSION || version > ANIM2_VERSION ||
            header_size < ANIM2_HEADER_SIZE) {
            fprintf(stderr, "anim_cache_load: unsupported ANM2 v%u header %u\n",
                    version, header_size);
            free(buf);
            return NULL;
        }
        format_version = version;
        if (header_size > ANIM2_HEADER_SIZE &&
            !anim_take(&cur, NULL, header_size - ANIM2_HEADER_SIZE,
                       "anim header extension")) {
            free(buf);
            return NULL;
        }
    } else {
        fprintf(stderr, "anim_cache_load: bad magic 0x%08X\n", magic);
        free(buf);
        return NULL;
    }

    if (base_count > 65535u || seq_count > 65535u) {
        fprintf(stderr, "anim_cache_load: unreasonable counts %u/%u\n",
                base_count, seq_count);
        free(buf);
        return NULL;
    }

    cache = (AnimCache*)calloc(1, sizeof(AnimCache));
    if (!cache) {
        free(buf);
        return NULL;
    }
    cache->base_count = (int)base_count;
    cache->seq_count = (int)seq_count;

    /* load framebases */
    cache->bases = (AnimFrameBase*)calloc(cache->base_count, sizeof(AnimFrameBase));
    cache->base_ids = (uint16_t*)malloc(cache->base_count * sizeof(uint16_t));
    if (!cache->bases || !cache->base_ids) {
        free(buf);
        anim_cache_free(cache);
        return NULL;
    }

    for (int i = 0; i < cache->base_count; i++) {
        AnimFrameBase* fb = &cache->bases[i];
        if (!anim_read_u16(&cur, &fb->base_id, "framebase id")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        cache->base_ids[i] = fb->base_id;
        if (!anim_read_u8(&cur, &fb->slot_count, "framebase slot count")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }

        fb->types = (uint8_t*)malloc(fb->slot_count);
        for (int s = 0; s < fb->slot_count; s++) {
            if (!fb->types ||
                !anim_read_u8(&cur, &fb->types[s], "framebase slot type")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
        }

        fb->map_lengths = (uint8_t*)malloc(fb->slot_count);
        fb->frame_maps = (uint8_t**)malloc(fb->slot_count * sizeof(uint8_t*));
        if (!fb->map_lengths || !fb->frame_maps) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        for (int s = 0; s < fb->slot_count; s++) {
            uint8_t ml = 0;
            if (!anim_read_u8(&cur, &ml, "framebase map length")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            fb->map_lengths[s] = ml;
            fb->frame_maps[s] = (uint8_t*)malloc(ml);
            if (ml > 0 && !fb->frame_maps[s]) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            for (int j = 0; j < ml; j++) {
                if (!anim_read_u8(&cur, &fb->frame_maps[s][j],
                                  "framebase map label")) {
                    free(buf);
                    anim_cache_free(cache);
                    return NULL;
                }
            }
        }
    }

    /* load sequences */
    cache->sequences = (AnimSequence*)calloc(cache->seq_count, sizeof(AnimSequence));
    if (!cache->sequences) {
        free(buf);
        anim_cache_free(cache);
        return NULL;
    }
    for (int i = 0; i < cache->seq_count; i++) {
        AnimSequence* seq = &cache->sequences[i];
        if (!anim_read_u16(&cur, &seq->seq_id, "sequence id") ||
            !anim_read_u16(&cur, &seq->frame_count, "sequence frame count")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }

        if (!anim_read_u8(&cur, &seq->interleave_count, "sequence interleave count")) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        if (seq->interleave_count > 0) {
            seq->interleave_order = (uint8_t*)malloc(seq->interleave_count);
            if (!seq->interleave_order) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            for (int j = 0; j < seq->interleave_count; j++) {
                if (!anim_read_u8(&cur, &seq->interleave_order[j],
                                  "sequence interleave slot")) {
                    free(buf);
                    anim_cache_free(cache);
                    return NULL;
                }
            }
        }

        if (format_version >= 3) {
            if (!anim_read_i16(&cur, &seq->frame_step,
                               "sequence frame step") ||
                !anim_read_i8(&cur, &seq->preanim_move,
                              "sequence pre-animation movement") ||
                !anim_read_i8(&cur, &seq->postanim_move,
                              "sequence post-animation movement") ||
                !anim_read_u8(&cur, &seq->forced_priority,
                              "sequence forced priority") ||
                !anim_read_u8(&cur, &seq->max_loops,
                              "sequence max loops") ||
                !anim_read_i8(&cur, &seq->reply_mode,
                              "sequence reply mode") ||
                !anim_read_u8(&cur, &seq->stretches,
                              "sequence stretches")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            seq->walk_flag = seq->postanim_move;
        } else {
            uint8_t walk_flag = 0;
            if (!anim_read_u8(&cur, &walk_flag, "sequence walk flag")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }
            seq->walk_flag = (int8_t)walk_flag;
            seq->frame_step = -1;
            seq->preanim_move = seq->interleave_count > 0 ? 2 : 0;
            seq->postanim_move = seq->walk_flag >= 0
                ? seq->walk_flag
                : (seq->interleave_count > 0 ? 2 : 0);
            seq->forced_priority = 5;
            seq->max_loops = 99;
            seq->reply_mode = -1;
            seq->stretches = 0;
        }

        seq->frames = (AnimSequenceFrame*)calloc(seq->frame_count, sizeof(AnimSequenceFrame));
        if (!seq->frames) {
            free(buf);
            anim_cache_free(cache);
            return NULL;
        }
        for (int fi = 0; fi < seq->frame_count; fi++) {
            AnimSequenceFrame* sf = &seq->frames[fi];
            if (!anim_read_u16(&cur, &sf->delay, "sequence frame delay") ||
                !anim_read_u16(&cur, &sf->frame.framebase_id, "sequence framebase id") ||
                !anim_read_u8(&cur, &sf->frame.transform_count, "sequence transform count")) {
                free(buf);
                anim_cache_free(cache);
                return NULL;
            }

            if (sf->frame.transform_count > 0) {
                sf->frame.transforms = (AnimTransform*)malloc(
                    sf->frame.transform_count * sizeof(AnimTransform));
                if (!sf->frame.transforms) {
                    free(buf);
                    anim_cache_free(cache);
                    return NULL;
                }
                for (int t = 0; t < sf->frame.transform_count; t++) {
                    if (!anim_read_u8(&cur, &sf->frame.transforms[t].slot_index,
                                      "sequence transform slot") ||
                        !anim_read_i16(&cur, &sf->frame.transforms[t].dx,
                                       "sequence transform dx") ||
                        !anim_read_i16(&cur, &sf->frame.transforms[t].dy,
                                       "sequence transform dy") ||
                        !anim_read_i16(&cur, &sf->frame.transforms[t].dz,
                                       "sequence transform dz")) {
                        free(buf);
                        anim_cache_free(cache);
                        return NULL;
                    }
                }
            }
        }
    }

    free(buf);
    anim_init_trig();

    fprintf(stderr, "anim_cache_load: loaded %d framebases, %d sequences from %s\n",
            cache->base_count, cache->seq_count, path);
    return cache;
}

/* ======================================================================== */
/* lookup                                                                     */
/* ======================================================================== */

AnimSequence* anim_get_sequence(AnimCache* cache, uint16_t seq_id) {
    if (!cache) return NULL;
    for (int i = 0; i < cache->seq_count; i++) {
        if (cache->sequences[i].seq_id == seq_id) {
            return &cache->sequences[i];
        }
    }
    return NULL;
}

AnimFrameBase* anim_get_framebase(AnimCache* cache, uint16_t base_id) {
    if (!cache) return NULL;
    for (int i = 0; i < cache->base_count; i++) {
        if (cache->bases[i].base_id == base_id) {
            return &cache->bases[i];
        }
    }
    return NULL;
}

/* ======================================================================== */
/* per-model animation state                                                  */
/* ======================================================================== */

AnimModelState* anim_model_state_create(
    const uint8_t* vertex_skins,
    int base_vert_count
) {
    AnimModelState* state = (AnimModelState*)calloc(1, sizeof(AnimModelState));
    state->vert_count = base_vert_count;
    state->verts = (int16_t*)calloc(base_vert_count * 3, sizeof(int16_t));

    /* build vertex group lookup from skin labels */
    state->groups = (int**)calloc(ANIM_MAX_LABELS, sizeof(int*));
    state->group_counts = (int*)calloc(ANIM_MAX_LABELS, sizeof(int));

    /* first pass: count vertices per label */
    int label_counts[ANIM_MAX_LABELS] = {0};
    for (int v = 0; v < base_vert_count; v++) {
        uint8_t label = vertex_skins[v];
        label_counts[label]++;
    }

    /* allocate per-label arrays */
    for (int l = 0; l < ANIM_MAX_LABELS; l++) {
        if (label_counts[l] > 0) {
            state->groups[l] = (int*)malloc(label_counts[l] * sizeof(int));
            state->group_counts[l] = 0;
        }
    }

    /* second pass: fill vertex indices */
    for (int v = 0; v < base_vert_count; v++) {
        uint8_t label = vertex_skins[v];
        state->groups[label][state->group_counts[label]++] = v;
    }

    return state;
}

void anim_model_state_free(AnimModelState* state) {
    if (!state) return;
    free(state->verts);
    for (int l = 0; l < ANIM_MAX_LABELS; l++) {
        free(state->groups[l]);
    }
    free(state->groups);
    free(state->group_counts);
    free(state);
}

/* ======================================================================== */
/* transform application (mirrors OSRS Model.transform)                       */
/* ======================================================================== */

void anim_apply_frame(
    AnimModelState* state,
    const int16_t* base_verts_src,
    const AnimFrameData* frame,
    const AnimFrameBase* fb
) {
    /* reset to base pose */
    memcpy(state->verts, base_verts_src, state->vert_count * 3 * sizeof(int16_t));

    /* pivot point for rotate/scale */
    int pivot_x = 0, pivot_y = 0, pivot_z = 0;

    for (int t = 0; t < frame->transform_count; t++) {
        uint8_t slot_idx = frame->transforms[t].slot_index;
        if (slot_idx >= fb->slot_count) continue;

        int type = fb->types[slot_idx];
        int dx = frame->transforms[t].dx;
        int dy = frame->transforms[t].dy;
        int dz = frame->transforms[t].dz;

        uint8_t map_len = fb->map_lengths[slot_idx];
        const uint8_t* labels = fb->frame_maps[slot_idx];

        if (type == 0) {
            /* origin: compute centroid of referenced vertex groups */
            int count = 0;
            int sum_x = 0, sum_y = 0, sum_z = 0;
            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    sum_x += state->verts[v * 3];
                    sum_y += state->verts[v * 3 + 1];
                    sum_z += state->verts[v * 3 + 2];
                    count++;
                }
            }
            if (count > 0) {
                pivot_x = sum_x / count + dx;
                pivot_y = sum_y / count + dy;
                pivot_z = sum_z / count + dz;
            } else {
                pivot_x = dx;
                pivot_y = dy;
                pivot_z = dz;
            }
        } else if (type == 1) {
            /* translate: add dx/dy/dz to all vertices in referenced groups */
            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    state->verts[v * 3]     += (int16_t)dx;
                    state->verts[v * 3 + 1] += (int16_t)dy;
                    state->verts[v * 3 + 2] += (int16_t)dz;
                }
            }
        } else if (type == 2) {
            /* rotate: euler Z-X-Y around pivot.
             * raw value * 8 → index into 2048-entry sine table.
             * rotation order: Z first, then X, then Y. */
            int ax = (dx & 0xFF) * 8;
            int ay = (dy & 0xFF) * 8;
            int az = (dz & 0xFF) * 8;

            int sin_x = anim_sine[ax & 2047];
            int cos_x = anim_cosine[ax & 2047];
            int sin_y = anim_sine[ay & 2047];
            int cos_y = anim_cosine[ay & 2047];
            int sin_z = anim_sine[az & 2047];
            int cos_z = anim_cosine[az & 2047];

            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    int vx = state->verts[v * 3]     - pivot_x;
                    int vy = state->verts[v * 3 + 1] - pivot_y;
                    int vz = state->verts[v * 3 + 2] - pivot_z;

                    /* Z rotation */
                    int rx = (vx * cos_z + vy * sin_z) >> 16;
                    int ry = (vy * cos_z - vx * sin_z) >> 16;
                    vx = rx; vy = ry;

                    /* X rotation */
                    ry = (vy * cos_x - vz * sin_x) >> 16;
                    int rz = (vy * sin_x + vz * cos_x) >> 16;
                    vy = ry; vz = rz;

                    /* Y rotation */
                    rx = (vz * sin_y + vx * cos_y) >> 16;
                    rz = (vz * cos_y - vx * sin_y) >> 16;
                    vx = rx; vz = rz;

                    state->verts[v * 3]     = (int16_t)(vx + pivot_x);
                    state->verts[v * 3 + 1] = (int16_t)(vy + pivot_y);
                    state->verts[v * 3 + 2] = (int16_t)(vz + pivot_z);
                }
            }
        } else if (type == 3) {
            /* scale: relative to pivot, 128 = 1.0x identity */
            for (int m = 0; m < map_len; m++) {
                uint8_t label = labels[m];
                /* label is uint8_t, always < 256 = ANIM_MAX_LABELS */
                for (int vi = 0; vi < state->group_counts[label]; vi++) {
                    int v = state->groups[label][vi];
                    int vx = state->verts[v * 3]     - pivot_x;
                    int vy = state->verts[v * 3 + 1] - pivot_y;
                    int vz = state->verts[v * 3 + 2] - pivot_z;

                    vx = (vx * dx) / 128;
                    vy = (vy * dy) / 128;
                    vz = (vz * dz) / 128;

                    state->verts[v * 3]     = (int16_t)(vx + pivot_x);
                    state->verts[v * 3 + 1] = (int16_t)(vy + pivot_y);
                    state->verts[v * 3 + 2] = (int16_t)(vz + pivot_z);
                }
            }
        }
        /* type 5 (alpha) skipped — we don't use face transparency in the viewer */
    }
}

/* ======================================================================== */
/* two-track interleaved animation (matches OSRS Model.applyAnimationFrames)  */
/* ======================================================================== */

/**
 * Apply a single transform slot to the vertex state (extracted from anim_apply_frame
 * to allow per-slot interleave filtering).
 *
 * pivot_x/y/z are read/written through pointers — they persist across slots
 * within a pass, exactly like the reference's transformTempX/Y/Z.
 */
static void anim_apply_single_transform(
    AnimModelState* state,
    int type, const uint8_t* labels, uint8_t map_len,
    int dx, int dy, int dz,
    int* pivot_x, int* pivot_y, int* pivot_z
) {
    if (type == 0) {
        /* origin: compute centroid of referenced vertex groups */
        int count = 0, sx = 0, sy = 0, sz = 0;
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                sx += state->verts[v * 3];
                sy += state->verts[v * 3 + 1];
                sz += state->verts[v * 3 + 2];
                count++;
            }
        }
        if (count > 0) {
            *pivot_x = sx / count + dx;
            *pivot_y = sy / count + dy;
            *pivot_z = sz / count + dz;
        } else {
            *pivot_x = dx;
            *pivot_y = dy;
            *pivot_z = dz;
        }
    } else if (type == 1) {
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                state->verts[v * 3]     += (int16_t)dx;
                state->verts[v * 3 + 1] += (int16_t)dy;
                state->verts[v * 3 + 2] += (int16_t)dz;
            }
        }
    } else if (type == 2) {
        int ax = (dx & 0xFF) * 8, ay = (dy & 0xFF) * 8, az = (dz & 0xFF) * 8;
        int sin_x = anim_sine[ax & 2047], cos_x = anim_cosine[ax & 2047];
        int sin_y = anim_sine[ay & 2047], cos_y = anim_cosine[ay & 2047];
        int sin_z = anim_sine[az & 2047], cos_z = anim_cosine[az & 2047];
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                int vx = state->verts[v * 3]     - *pivot_x;
                int vy = state->verts[v * 3 + 1] - *pivot_y;
                int vz = state->verts[v * 3 + 2] - *pivot_z;
                int rx = (vx * cos_z + vy * sin_z) >> 16;
                int ry = (vy * cos_z - vx * sin_z) >> 16;
                vx = rx; vy = ry;
                ry = (vy * cos_x - vz * sin_x) >> 16;
                int rz = (vy * sin_x + vz * cos_x) >> 16;
                vy = ry; vz = rz;
                rx = (vz * sin_y + vx * cos_y) >> 16;
                rz = (vz * cos_y - vx * sin_y) >> 16;
                state->verts[v * 3]     = (int16_t)(rx + *pivot_x);
                state->verts[v * 3 + 1] = (int16_t)(vy + *pivot_y);
                state->verts[v * 3 + 2] = (int16_t)(rz + *pivot_z);
            }
        }
    } else if (type == 3) {
        for (int m = 0; m < map_len; m++) {
            uint8_t label = labels[m];
            for (int vi = 0; vi < state->group_counts[label]; vi++) {
                int v = state->groups[label][vi];
                int vx = state->verts[v * 3]     - *pivot_x;
                int vy = state->verts[v * 3 + 1] - *pivot_y;
                int vz = state->verts[v * 3 + 2] - *pivot_z;
                state->verts[v * 3]     = (int16_t)((vx * dx) / 128 + *pivot_x);
                state->verts[v * 3 + 1] = (int16_t)((vy * dy) / 128 + *pivot_y);
                state->verts[v * 3 + 2] = (int16_t)((vz * dz) / 128 + *pivot_z);
            }
        }
    }
}

/**
 * Apply two animation frames with body-part interleaving.
 *
 * Mirrors OSRS Model.applyAnimationFrames():
 *   - interleave_order lists framebase SLOT INDICES owned by SECONDARY (walk)
 *   - Pass 1: apply primary transforms for slots NOT in interleave_order
 *   - Pass 2: apply secondary transforms for slots IN interleave_order
 *   - Type-0 (pivot) transforms always execute in both passes
 *
 * CRITICAL: interleave_order contains framebase SLOT INDICES, not vertex labels!
 * The reference code (Model.java:1322-1343) walks both the frame's slot list and
 * the interleave_order simultaneously, comparing slot indices directly.
 *
 * Both passes operate on the same vertex state with independent pivot tracking,
 * exactly as the reference does with transformTempX/Y/Z reset between passes.
 */
static void anim_apply_frame_interleaved(
    AnimModelState* state,
    const int16_t* base_verts_src,
    const AnimFrameData* secondary_frame, const AnimFrameBase* secondary_fb,
    const AnimFrameData* primary_frame, const AnimFrameBase* primary_fb,
    const uint8_t* interleave_order, int interleave_count
) {
    /* reset to base pose */
    memcpy(state->verts, base_verts_src, state->vert_count * 3 * sizeof(int16_t));

    /* build boolean mask: interleave_order lists SLOT INDICES the SECONDARY owns.
       index by slot index (0-244 for our 245-slot framebase), NOT vertex labels. */
    uint8_t secondary_slot[256];
    memset(secondary_slot, 0, sizeof(secondary_slot));
    for (int i = 0; i < interleave_count; i++) {
        secondary_slot[interleave_order[i]] = 1;
    }

    /* pass 1: primary frame — apply transforms for slots NOT in interleave_order.
     * type-0 (pivot) always executes regardless of ownership.
     * matches reference: if (k1 != i1 || class18.types[k1] == 0) */
    int pivot_x = 0, pivot_y = 0, pivot_z = 0;
    for (int t = 0; t < primary_frame->transform_count; t++) {
        uint8_t slot_idx = primary_frame->transforms[t].slot_index;
        if (slot_idx >= primary_fb->slot_count) continue;

        int type = primary_fb->types[slot_idx];
        int in_interleave = secondary_slot[slot_idx];

        if (!in_interleave || type == 0) {
            anim_apply_single_transform(
                state, type,
                primary_fb->frame_maps[slot_idx],
                primary_fb->map_lengths[slot_idx],
                primary_frame->transforms[t].dx,
                primary_frame->transforms[t].dy,
                primary_frame->transforms[t].dz,
                &pivot_x, &pivot_y, &pivot_z);
        }
    }

    /* pass 2: secondary frame — apply transforms for slots IN interleave_order.
     * type-0 (pivot) always executes.
     * matches reference: if (i2 == i1 || class18.types[i2] == 0) */
    pivot_x = 0; pivot_y = 0; pivot_z = 0;
    for (int t = 0; t < secondary_frame->transform_count; t++) {
        uint8_t slot_idx = secondary_frame->transforms[t].slot_index;
        if (slot_idx >= secondary_fb->slot_count) continue;

        int type = secondary_fb->types[slot_idx];
        int in_interleave = secondary_slot[slot_idx];

        if (in_interleave || type == 0) {
            anim_apply_single_transform(
                state, type,
                secondary_fb->frame_maps[slot_idx],
                secondary_fb->map_lengths[slot_idx],
                secondary_frame->transforms[t].dx,
                secondary_frame->transforms[t].dy,
                secondary_frame->transforms[t].dz,
                &pivot_x, &pivot_y, &pivot_z);
        }
    }
}

/* Apply the current pose/action pair to one model state. The action owns the
 * full model unless its sequence supplies an OSRS interleave table, in which
 * case the pose supplies the interleaved transform slots. If the action frame
 * cannot be applied, fall back to the pose exactly as a single-track actor
 * would. Mesh upload remains the caller's responsibility because player
 * models are unique while same-type NPCs share one render mesh. */
int anim_mix_pose_action(
    AnimCache* cache,
    AnimModelState* state,
    const int16_t* base_verts,
    AnimSequence* pose,
    int pose_frame_index,
    AnimSequence* action,
    int action_frame_index
) {
    if (!cache || !state || !base_verts) return 0;

    if (action && action_frame_index >= 0 &&
        action_frame_index < action->frame_count) {
        AnimFrameData* action_frame =
            &action->frames[action_frame_index].frame;
        AnimFrameBase* action_base =
            anim_get_framebase(cache, action_frame->framebase_id);
        if (action_base && pose && pose_frame_index >= 0 &&
            pose_frame_index < pose->frame_count &&
            action->interleave_count > 0 && action->interleave_order) {
            AnimFrameData* pose_frame =
                &pose->frames[pose_frame_index].frame;
            AnimFrameBase* pose_base =
                anim_get_framebase(cache, pose_frame->framebase_id);
            if (pose_base) {
                anim_apply_frame_interleaved(
                    state, base_verts,
                    pose_frame, pose_base, action_frame, action_base,
                    action->interleave_order, action->interleave_count);
                return 1;
            }
        } else if (action_base) {
            anim_apply_frame(state, base_verts, action_frame, action_base);
            return 1;
        }
    }

    if (pose && pose_frame_index >= 0 &&
        pose_frame_index < pose->frame_count) {
        AnimFrameData* pose_frame = &pose->frames[pose_frame_index].frame;
        AnimFrameBase* pose_base =
            anim_get_framebase(cache, pose_frame->framebase_id);
        if (pose_base) {
            anim_apply_frame(state, base_verts, pose_frame, pose_base);
            return 1;
        }
    }
    return 0;
}

/* ======================================================================== */
/* mesh re-expansion (apply animated base verts → expanded rendering verts)   */
/* ======================================================================== */

/**
 * Re-expand animated base vertices into the raylib mesh's expanded vertex buffer.
 * This mirrors expand_model from the Python exporter but in-place, using
 * face_indices to map from base to expanded vertices.
 *
 * The mesh has face_count*3 expanded vertices. Each triplet (i*3, i*3+1, i*3+2)
 * corresponds to face_indices[i*3], face_indices[i*3+1], face_indices[i*3+2]
 * pointing into base_vertices.
 *
 * OSRS Y is negated for rendering (negative-up → positive-up).
 */
void anim_update_mesh(
    float* mesh_vertices,
    const AnimModelState* state,
    const uint16_t* face_indices,
    int face_count
) {
    for (int fi = 0; fi < face_count; fi++) {
        int a = face_indices[fi * 3];
        int b = face_indices[fi * 3 + 1];
        int c = face_indices[fi * 3 + 2];

        int vi = fi * 9; /* 3 verts * 3 coords */
        mesh_vertices[vi]     = (float)state->verts[a * 3];
        mesh_vertices[vi + 1] = (float)(-state->verts[a * 3 + 1]); /* negate Y */
        mesh_vertices[vi + 2] = (float)state->verts[a * 3 + 2];

        mesh_vertices[vi + 3] = (float)state->verts[b * 3];
        mesh_vertices[vi + 4] = (float)(-state->verts[b * 3 + 1]);
        mesh_vertices[vi + 5] = (float)state->verts[b * 3 + 2];

        mesh_vertices[vi + 6] = (float)state->verts[c * 3];
        mesh_vertices[vi + 7] = (float)(-state->verts[c * 3 + 1]);
        mesh_vertices[vi + 8] = (float)state->verts[c * 3 + 2];
    }
}

/* ======================================================================== */
/* cleanup                                                                    */
/* ======================================================================== */

void anim_cache_free(AnimCache* cache) {
    if (!cache) return;

    for (int i = 0; i < cache->base_count; i++) {
        AnimFrameBase* fb = &cache->bases[i];
        free(fb->types);
        free(fb->map_lengths);
        for (int s = 0; s < fb->slot_count; s++) {
            free(fb->frame_maps[s]);
        }
        free(fb->frame_maps);
    }
    free(cache->bases);
    free(cache->base_ids);

    for (int i = 0; i < cache->seq_count; i++) {
        AnimSequence* seq = &cache->sequences[i];
        free(seq->interleave_order);
        for (int fi = 0; fi < seq->frame_count; fi++) {
            free(seq->frames[fi].frame.transforms);
        }
        free(seq->frames);
    }
    free(cache->sequences);
    free(cache);
}
