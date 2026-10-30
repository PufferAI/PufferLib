/*
 * viewer.c — Fight Caves viewer: asset loading, the game interface, 3D rendering
 * and debug overlays, and the playable window and replay loop. It is compiled
 * only as part of binding.c, after the core and fc_apply_env_config.
 *
 * Controls:
 *   Space       — pause/resume              Right      — single-step one tick
 *   Q or Esc    — quit (Esc first closes an open menu)
 *   O or D      — toggle debug overlays     Shift+O    — cycle debug overlays
 *   G           — grid                      C          — collision overlay
 *   L           — toggle camera lock        Scroll     — zoom
 *   Right click — context menu              Right-drag — orbit camera
 *
 * Manual play (./fight_caves) adds:
 *   Left click  — walk to a tile or attack an NPC; use the side panel and minimap
 *   1/2/3       — protect from melee/missiles/magic
 *   F           — eat shark                 P          — drink prayer potion
 *   X           — toggle run                R          — reset episode
 *   4/5         — camera presets            F9         — toggle god mode
 *
 * Policy replay (./puffer eval, ./fight_caves eval) adds:
 *   1/2/4/0     — 1x/2x/4x/10x speed        Up/Down    — faster/slower
 *   Shift+4/5   — camera presets
 *
 * The console in the chat box lists NPC targets (click to attack), toggles
 * debug overlays and god mode, jumps to a wave, sets the tick rate, and shows
 * player state, observations, the action mask, rewards and the event log.
 */

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "fight_caves.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ===================== Asset loading ===================== */

/* Assets */

/* Reads a whole asset into a new buffer, or prints why it cannot and
 * returns NULL. Asset paths are relative to FC_ASSET_ROOT or, by default, to
 * the installed bundle under the working directory (the repository root),
 * the way the core finds its arena maps. */
unsigned char* fc_asset_read_all(const char* logical_path, size_t* out_size) {
    const char* root = getenv("FC_ASSET_ROOT");
    char path[1024];
    long size;
    unsigned char* data;
    size_t got;

    if (out_size) *out_size = 0;
    if (!root || !root[0]) root = "resources/fight_caves/viewer";
    snprintf(path, sizeof(path), "%s/%s", root, logical_path);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open viewer asset %s: %s\n", path,
                strerror(errno));
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: seek failed while reading asset\n", logical_path);
        fclose(f);
        return NULL;
    }
    size = ftell(f);
    if (size <= 0) {
        fprintf(stderr, "%s: empty or unreadable asset\n", logical_path);
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: seek failed while reading asset\n", logical_path);
        fclose(f);
        return NULL;
    }
    data = malloc((size_t)size);
    if (!data) {
        fprintf(stderr, "%s: out of memory while reading asset\n", logical_path);
        fclose(f);
        return NULL;
    }
    got = fread(data, 1, (size_t)size, f);
    fclose(f);
    if (got != (size_t)size) {
        fprintf(stderr, "%s: short read while reading asset\n", logical_path);
        free(data);
        return NULL;
    }
    if (out_size) *out_size = (size_t)size;
    return data;
}

/* Asset Reader */

/* Bounds-checked reader over a whole asset file. The first failure (a short
 * file, a bad header or an allocation failure) prints one message and clears
 * ok; every later read is a no-op that yields zeros or NULL, so a loader can
 * read a whole record and check ok once. Fields are copied as stored: the
 * asset files are little-endian like every platform the viewer supports. */
typedef struct {
    unsigned char* data;
    size_t size;
    size_t pos;
    int ok;
    const char* path;
} FcReader;

static int fc_reader_open(FcReader* r, const char* path) {
    r->path = path;
    r->pos = 0;
    r->data = fc_asset_read_all(path, &r->size);
    r->ok = r->data != NULL;
    return r->ok;
}

static void fc_reader_close(FcReader* r) {
    free(r->data);
    r->data = NULL;
}

static void fc_reader_fail(FcReader* r, const char* reason) {
    if (r->ok) fprintf(stderr, "%s: %s\n", r->path, reason);
    r->ok = 0;
}

static void fc_reader_require(FcReader* r, int condition, const char* reason) {
    if (!condition) fc_reader_fail(r, reason);
}

/* The next count * size bytes, or NULL once the reader has failed. */
static const unsigned char* fc_reader_take(FcReader* r, size_t count,
                                           size_t size) {
    if (r->ok && size > 0 && count > (r->size - r->pos) / size)
        fc_reader_fail(r, "file is truncated");
    if (!r->ok) return NULL;
    const unsigned char* bytes = r->data + r->pos;
    r->pos += count * size;
    return bytes;
}

static void fc_read(FcReader* r, void* dst, size_t count, size_t size) {
    const unsigned char* bytes = fc_reader_take(r, count, size);
    if (bytes) memcpy(dst, bytes, count * size);
    else memset(dst, 0, count * size);
}

static uint8_t fc_read_u8(FcReader* r) {
    uint8_t value;
    fc_read(r, &value, 1, sizeof(value));
    return value;
}

static uint16_t fc_read_u16(FcReader* r) {
    uint16_t value;
    fc_read(r, &value, 1, sizeof(value));
    return value;
}

static int16_t fc_read_i16(FcReader* r) {
    int16_t value;
    fc_read(r, &value, 1, sizeof(value));
    return value;
}

static uint32_t fc_read_u32(FcReader* r) {
    uint32_t value;
    fc_read(r, &value, 1, sizeof(value));
    return value;
}

static int32_t fc_read_i32(FcReader* r) {
    int32_t value;
    fc_read(r, &value, 1, sizeof(value));
    return value;
}

static float fc_read_f32(FcReader* r) {
    float value;
    fc_read(r, &value, 1, sizeof(value));
    return value;
}

static void fc_reader_skip(FcReader* r, size_t bytes) {
    fc_reader_take(r, bytes, 1);
}

static void fc_reader_seek(FcReader* r, size_t pos) {
    if (r->ok && pos > r->size) fc_reader_fail(r, "file is truncated");
    if (r->ok) r->pos = pos;
}

/* A zeroed array of count elements. NULL when count is 0 or the reader has
 * failed, so callers check r->ok rather than the pointer. */
static void* fc_alloc_array(FcReader* r, size_t count, size_t size) {
    if (!r->ok || count == 0 || size == 0) return NULL;
    void* array = calloc(count, size);
    if (!array) fc_reader_fail(r, "out of memory");
    return array;
}

/* Reads count elements into a new array; NULL as for fc_alloc_array. */
static void* fc_read_array(FcReader* r, size_t count, size_t size) {
    const unsigned char* bytes = fc_reader_take(r, count, size);
    void* array = bytes ? fc_alloc_array(r, count, size) : NULL;
    if (array) memcpy(array, bytes, count * size);
    return array;
}

/* Asset Raylib */

static const char* fc_asset_extension(const char* path, const char* fallback) {
    const char* dot = path ? strrchr(path, '.') : NULL;
    return dot && dot[0] ? dot : fallback;
}

Image fc_load_image_asset(const char* path) {
    Image empty = {0};
    size_t size = 0;
    unsigned char* bytes = fc_asset_read_all(path, &size);
    Image image;

    if (!bytes || size == 0) return empty;
    image = LoadImageFromMemory(fc_asset_extension(path, ".png"), bytes,
                                (int)size);
    free(bytes);
    return image;
}

Texture2D fc_load_texture_asset(const char* path) {
    Texture2D empty = {0};
    Image image = fc_load_image_asset(path);
    Texture2D texture;

    if (!image.data) return empty;
    texture = LoadTextureFromImage(image);
    UnloadImage(image);
    return texture;
}

Font fc_load_font_asset(const char* path, int font_size) {
    Font empty = {0};
    size_t size = 0;
    unsigned char* bytes = fc_asset_read_all(path, &size);
    Font font;

    if (!bytes || size == 0) return empty;
    font = LoadFontFromMemory(fc_asset_extension(path, ".ttf"), bytes,
                              (int)size, font_size, NULL, 95);
    free(bytes);
    return font.texture.id != 0 ? font : empty;
}

/* Animated Atlas */

typedef struct {
    uint32_t texture_id;
    uint16_t x;
    uint16_t y;
    uint16_t w;
    uint16_t h;
    uint8_t direction;
    uint8_t speed;
    uint16_t pad;
} FcTextureAnimRow;

/* The texture atlas shared by terrain objects and textured models. Its
 * animated cells scroll every frame: each one is rebuilt from base_pixels in
 * cell_pixels and uploaded on its own. */
typedef struct {
    Texture2D texture;
    unsigned char* base_pixels;
    unsigned char* cell_pixels;
    FcTextureAnimRow* anims;
    int width;
    int height;
    int anim_count;
    float anim_ticks;
} FcAnimatedAtlas;

#define FC_ATLAS_MAGIC 0x41544C53u
#define FC_TEXTURE_ANIM_MAGIC 0x4D4E4154u
#define FC_TEXTURE_ANIM_VERSION 1u

_Static_assert(sizeof(FcTextureAnimRow) == 16,
               "TANM rows must retain their 16-byte file layout");

/* Reads the atlas's animation table: which cells scroll, which way and how
 * fast. */
static int fc_animated_atlas_load_anims(FcAnimatedAtlas* atlas,
                                        const char* path) {
    FcReader r;

    if (!fc_reader_open(&r, path))
        return 0;
    uint32_t magic = fc_read_u32(&r);
    uint32_t version = fc_read_u32(&r);
    uint32_t count = fc_read_u32(&r);
    fc_reader_require(&r, magic == FC_TEXTURE_ANIM_MAGIC
                          && version == FC_TEXTURE_ANIM_VERSION,
                      "unsupported texture animation format");
    FcTextureAnimRow* rows = fc_read_array(&r, count, sizeof(*rows));
    size_t cell_size = 0;
    for (uint32_t i = 0; rows && i < count; i++) {
        size_t size = (size_t)rows[i].w * rows[i].h * 4;
        if (size > cell_size) cell_size = size;
    }
    atlas->cell_pixels = fc_alloc_array(&r, cell_size, 1);
    fc_reader_close(&r);
    if (!r.ok) {
        free(rows);
        return 0;
    }
    atlas->anims = rows;
    atlas->anim_count = (int)count;
    fprintf(stderr, "animated atlas: %d cells loaded from %s\n",
            atlas->anim_count, path);
    return 1;
}

int fc_animated_atlas_load(FcAnimatedAtlas* atlas, const char* atlas_path,
                           const char* anim_path) {
    FcAnimatedAtlas loaded = {0};
    FcReader r;
    Image image;

    if (!atlas || !atlas_path || atlas->texture.id > 0
            || !fc_reader_open(&r, atlas_path))
        return 0;
    uint32_t magic = fc_read_u32(&r);
    uint32_t width = fc_read_u32(&r);
    uint32_t height = fc_read_u32(&r);
    fc_reader_require(&r, magic == FC_ATLAS_MAGIC && width > 0 && height > 0,
                      "not a texture atlas");
    loaded.base_pixels = fc_read_array(&r, (size_t)width * height, 4);
    fc_reader_close(&r);
    if (!r.ok) return 0;

    image = (Image) {
        .data = loaded.base_pixels,
        .width = (int)width,
        .height = (int)height,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    };
    loaded.texture = LoadTextureFromImage(image);
    if (loaded.texture.id == 0) {
        free(loaded.base_pixels);
        return 0;
    }
    SetTextureFilter(loaded.texture, TEXTURE_FILTER_POINT);
    loaded.width = (int)width;
    loaded.height = (int)height;
    if (!fc_animated_atlas_load_anims(&loaded, anim_path)) {
        UnloadTexture(loaded.texture);
        free(loaded.base_pixels);
        return 0;
    }
    *atlas = loaded;
    fprintf(stderr, "animated atlas: %ux%u loaded from %s\n",
            width, height, atlas_path);
    return 1;
}

void fc_animated_atlas_update(FcAnimatedAtlas* atlas, float dt) {
    if (!atlas || atlas->texture.id == 0 || atlas->anim_count <= 0)
        return;

    atlas->anim_ticks += dt * 50.0f;
    for (int r = 0; r < atlas->anim_count; r++) {
        FcTextureAnimRow* row = &atlas->anims[r];
        unsigned char* cell = atlas->cell_pixels;
        int shift;
        if (row->w == 0 || row->h == 0
                || row->x + row->w > atlas->width
                || row->y + row->h > atlas->height
                || row->speed == 0)
            continue;
        shift = (int)(atlas->anim_ticks * (float)row->speed);
        if (row->direction == 1 || row->direction == 3) {
            int pad = row->pad;
            int center_h;
            if (pad * 2 >= row->h) pad = 0;
            center_h = row->h - pad * 2;
            shift %= center_h;
            if (row->direction == 1) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                int sy = (y - pad + shift) % center_h;
                if (sy < 0) sy += center_h;
                sy += pad;
                for (int x = 0; x < row->w; x++) {
                    size_t dst = ((size_t)y * row->w + x) * 4;
                    size_t src = ((size_t)(row->y + sy) * atlas->width
                                  + row->x + x) * 4;
                    memcpy(&cell[dst], &atlas->base_pixels[src], 4);
                }
            }
        } else if (row->direction == 2 || row->direction == 4) {
            shift %= row->w;
            if (row->direction == 2) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                for (int x = 0; x < row->w; x++) {
                    int sx = (x + shift) % row->w;
                    size_t dst;
                    size_t src;
                    if (sx < 0) sx += row->w;
                    dst = ((size_t)y * row->w + x) * 4;
                    src = ((size_t)(row->y + y) * atlas->width
                           + row->x + sx) * 4;
                    memcpy(&cell[dst], &atlas->base_pixels[src], 4);
                }
            }
        } else {
            continue;
        }
        Rectangle area = {row->x, row->y, row->w, row->h};
        UpdateTextureRec(atlas->texture, area, cell);
    }
}

void fc_animated_atlas_unload(FcAnimatedAtlas* atlas) {
    if (!atlas) return;
    if (atlas->texture.id > 0) UnloadTexture(atlas->texture);
    free(atlas->base_pixels);
    free(atlas->cell_pixels);
    free(atlas->anims);
    *atlas = (FcAnimatedAtlas) {0};
}

/* Anim Loader */
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
 * Binary format (.anims, ANM2 version 3):
 *   header: char[4] magic ("ANM2"), uint16 version (3), uint16 header_size
 *           (24), uint32 framebase_count, uint32 sequence_count,
 *           uint32 sequence_frame_count, uint32 flags
 *   framebases section, sequences section with inlined frame data.
 */

#define ANIM_MAX_LABELS 256

typedef struct {
    uint16_t base_id;
    uint8_t slot_count;
    uint8_t *types;
    uint8_t *map_lengths;
    uint8_t **frame_maps;
} AnimFrameBase;

typedef struct {
    uint8_t slot_index;
    int16_t dx;
    int16_t dy;
    int16_t dz;
} AnimTransform;

typedef struct {
    uint16_t framebase_id;
    uint8_t transform_count;
    AnimTransform *transforms;
} AnimFrameData;

typedef struct {
    uint16_t delay;
    AnimFrameData frame;
} AnimSequenceFrame;

typedef struct {
    uint16_t seq_id;
    uint16_t frame_count;
    uint8_t interleave_count;
    uint8_t *interleave_order;
    int16_t frame_step;
    int8_t postanim_move;
    AnimSequenceFrame *frames;
} AnimSequence;

typedef struct {
    AnimFrameBase *bases;
    int base_count;
    AnimSequence *sequences;
    int seq_count;
} AnimCache;

typedef struct {
    int16_t *verts;
    int vert_count;
    int **groups;
    int *group_counts;
} AnimModelState;

/* One sequence being played: its id, the frame shown and the seconds left on
 * that frame. A zeroed track plays nothing. */
typedef struct {
    uint16_t sequence;
    int frame;
    float timer;
} AnimTrack;

#define ANIM2_MAGIC 0x324D4E41 /* bytes "ANM2" when read little-endian */
#define ANIM2_VERSION 3
#define ANIM2_HEADER_SIZE 24
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

/* Counts are stored only once their arrays exist, so anim_cache_free can
 * release a partly loaded cache. */
static void anim_read_framebase(FcReader* r, AnimFrameBase* fb) {
    fb->base_id = fc_read_u16(r);
    uint8_t slot_count = fc_read_u8(r);
    fb->types = fc_read_array(r, slot_count, 1);
    fb->map_lengths = fc_alloc_array(r, slot_count, 1);
    fb->frame_maps = fc_alloc_array(r, slot_count, sizeof(uint8_t*));
    if (!r->ok) return;
    fb->slot_count = slot_count;
    for (int s = 0; s < slot_count; s++) {
        fb->map_lengths[s] = fc_read_u8(r);
        fb->frame_maps[s] = fc_read_array(r, fb->map_lengths[s], 1);
    }
}

static void anim_read_sequence(FcReader* r, AnimSequence* seq) {
    seq->seq_id = fc_read_u16(r);
    uint16_t frame_count = fc_read_u16(r);
    seq->interleave_count = fc_read_u8(r);
    seq->interleave_order = fc_read_array(r, seq->interleave_count, 1);
    seq->frame_step = fc_read_i16(r);
    fc_reader_skip(r, 1); /* pre-animation movement: unused */
    seq->postanim_move = (int8_t)fc_read_u8(r);
    fc_reader_skip(r, 4); /* priority, loops, reply mode, stretch: unused */

    seq->frames = fc_alloc_array(r, frame_count, sizeof(AnimSequenceFrame));
    if (!r->ok) return;
    seq->frame_count = frame_count;
    for (int fi = 0; fi < frame_count; fi++) {
        AnimSequenceFrame* sf = &seq->frames[fi];
        sf->delay = fc_read_u16(r);
        sf->frame.framebase_id = fc_read_u16(r);
        uint8_t transform_count = fc_read_u8(r);
        sf->frame.transforms =
            fc_alloc_array(r, transform_count, sizeof(AnimTransform));
        if (!r->ok) return;
        sf->frame.transform_count = transform_count;
        for (int t = 0; t < transform_count; t++) {
            AnimTransform* transform = &sf->frame.transforms[t];
            transform->slot_index = fc_read_u8(r);
            transform->dx = fc_read_i16(r);
            transform->dy = fc_read_i16(r);
            transform->dz = fc_read_i16(r);
        }
    }
}

AnimCache* anim_cache_load(const char* path) {
    AnimCache* cache = calloc(1, sizeof(AnimCache));
    FcReader r;
    if (!cache || !fc_reader_open(&r, path)) {
        free(cache);
        return NULL;
    }

    uint32_t magic = fc_read_u32(&r);
    fc_reader_require(&r, magic == ANIM2_MAGIC, "not an animation file");
    uint16_t version = fc_read_u16(&r);
    uint16_t header_size = fc_read_u16(&r);
    fc_reader_require(&r, version == ANIM2_VERSION
                          && header_size == ANIM2_HEADER_SIZE,
                      "unsupported ANM2 version or header size");
    uint32_t base_count = fc_read_u32(&r);
    uint32_t seq_count = fc_read_u32(&r);
    fc_reader_skip(&r, 8); /* total sequence frames and flags: unused */
    fc_reader_require(&r, base_count <= 65535u && seq_count <= 65535u,
                      "unreasonable framebase or sequence count");

    cache->bases = fc_alloc_array(&r, base_count, sizeof(AnimFrameBase));
    if (r.ok) cache->base_count = (int)base_count;
    for (int i = 0; r.ok && i < cache->base_count; i++)
        anim_read_framebase(&r, &cache->bases[i]);

    cache->sequences = fc_alloc_array(&r, seq_count, sizeof(AnimSequence));
    if (r.ok) cache->seq_count = (int)seq_count;
    for (int i = 0; r.ok && i < cache->seq_count; i++)
        anim_read_sequence(&r, &cache->sequences[i]);

    fc_reader_close(&r);
    if (!r.ok) {
        anim_cache_free(cache);
        return NULL;
    }
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

/* Seconds a sequence frame is shown: delays count 20 ms client cycles, and no
 * frame lasts less than one 60 Hz viewer frame. */
static float anim_frame_duration(const AnimSequence* sequence, int frame) {
    if (!sequence || frame < 0 || frame >= sequence->frame_count) return 0.016f;
    float duration = (float)sequence->frames[frame].delay * 0.02f;
    return duration < 0.016f ? 0.016f : duration;
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

/**
 * Apply one transform slot to the vertex state.
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

/* One pass over a frame's transforms in file order, starting from a zero
 * pivot. With an ownership mask (1 = slot owned by the secondary track), only
 * the slots whose owner matches `secondary` are applied; type-0 (pivot)
 * transforms always run. */
static void anim_apply_pass(AnimModelState* state, const AnimFrameData* frame,
                            const AnimFrameBase* fb,
                            const uint8_t* secondary_slot, int secondary) {
    int pivot_x = 0, pivot_y = 0, pivot_z = 0;
    for (int t = 0; t < frame->transform_count; t++) {
        const AnimTransform* transform = &frame->transforms[t];
        uint8_t slot = transform->slot_index;
        if (slot >= fb->slot_count) continue;
        int type = fb->types[slot];
        if (secondary_slot && secondary_slot[slot] != secondary && type != 0)
            continue;
        anim_apply_single_transform(state, type, fb->frame_maps[slot],
                                    fb->map_lengths[slot], transform->dx,
                                    transform->dy, transform->dz,
                                    &pivot_x, &pivot_y, &pivot_z);
    }
}

void anim_apply_frame(
    AnimModelState* state,
    const int16_t* base_verts_src,
    const AnimFrameData* frame,
    const AnimFrameBase* fb
) {
    /* reset to base pose */
    memcpy(state->verts, base_verts_src, state->vert_count * 3 * sizeof(int16_t));
    anim_apply_pass(state, frame, fb, NULL, 0);
}

/* ======================================================================== */
/* two-track interleaved animation (matches OSRS Model.applyAnimationFrames)  */
/* ======================================================================== */

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

    /* pass 1, primary frame: reference if (k1 != i1 || class18.types[k1] == 0) */
    anim_apply_pass(state, primary_frame, primary_fb, secondary_slot, 0);
    /* pass 2, secondary frame: reference if (i2 == i1 || class18.types[i2] == 0) */
    anim_apply_pass(state, secondary_frame, secondary_fb, secondary_slot, 1);
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

/* Models */
// Loads models from .models MDL2/MDL3 binary for Raylib rendering.
// Fight Caves raylib model loader.

typedef struct {
    uint8_t textured;
    uint16_t tex_a;
    uint16_t tex_b;
    uint16_t tex_c;
    float u_base;
    float v_base;
    float u_scale;
    float v_scale;
} ModelFaceUvInfo;

typedef struct {
    uint32_t model_id;
    Model model;
    float *rest_verts;
    int16_t *base_verts;
    uint8_t *vertex_skins;
    uint16_t *face_indices;
    ModelFaceUvInfo *face_uvs;
    int base_vert_count;
    int face_count;
} ModelEntry;

typedef struct {
    ModelEntry *entries;
    int count;
    int has_textures;
} ModelSet;

#define MDL2_MAGIC 0x4D444C32
#define MDL3_MAGIC 0x4D444C33
#define MUV1_MAGIC 0x3156554D

/* Sets hold at most a few dozen models, so a linear search is enough. */
ModelEntry *model_find(ModelSet *set, uint32_t id) {
    if (!set) return NULL;
    for (int i = 0; i < set->count; i++)
        if (set->entries[i].model_id == id) return &set->entries[i];
    return NULL;
}

/* Screen-space face picking at the drawn pose, with the client's 5px tolerance.
 * Returns camera depth for overlap ordering, or -1 for a miss. posed_vertices
 * uses animation/cache coordinates; NULL selects the asset's rest mesh. */
float models_pick_depth(const ModelEntry *entry, const int16_t *posed_vertices,
                         Vector3 position, float yaw_degrees, Camera3D camera,
                         Vector2 mouse, int screen_width, int screen_height) {
    if (!entry || !entry->rest_verts) return -1;
    /* Same model -> yaw -> translation order as DrawModelEx. Never read the
     * shared uploaded mesh: another NPC of this type may have a different pose. */
    Matrix transform = MatrixMultiply(entry->model.transform,
        MatrixMultiply(MatrixRotateY(yaw_degrees * DEG2RAD),
                       MatrixTranslate(position.x, position.y, position.z)));
    Vector3 forward = Vector3Normalize(Vector3Subtract(camera.target, camera.position));
    float nearest = FLT_MAX;
    for (int face = 0; face < entry->face_count; face++) {
        float min_x = FLT_MAX, min_y = FLT_MAX;
        float max_x = -FLT_MAX, max_y = -FLT_MAX, depth = 0;
        int visible = 1;
        for (int corner = 0; corner < 3; corner++) {
            int offset = (face * 3 + corner) * 3;
            Vector3 local;
            if (posed_vertices && entry->face_indices) {
                int vertex = entry->face_indices[face * 3 + corner] * 3;
                local = (Vector3){posed_vertices[vertex] / 128.0f,
                    -posed_vertices[vertex + 1] / 128.0f,
                    -posed_vertices[vertex + 2] / 128.0f};
            } else {
                local = (Vector3){entry->rest_verts[offset], entry->rest_verts[offset + 1],
                                   entry->rest_verts[offset + 2]};
            }
            Vector3 world = Vector3Transform(local, transform);
            float z = Vector3DotProduct(Vector3Subtract(world, camera.position), forward);
            if (z <= 0.01f) { visible = 0; break; }
            Vector2 screen = GetWorldToScreenEx(world, camera, screen_width, screen_height);
            min_x = fminf(min_x, screen.x); max_x = fmaxf(max_x, screen.x);
            min_y = fminf(min_y, screen.y); max_y = fmaxf(max_y, screen.y);
            depth += z / 3.0f;
        }
        /* RuneLite Perspective.calculate2DBounds uses projected face rectangles
         * padded by five pixels, not a ground-tile test or pixel-perfect triangles. */
        if (visible && mouse.x >= min_x - 5 && mouse.x <= max_x + 5 &&
            mouse.y >= min_y - 5 && mouse.y <= max_y + 5 && depth < nearest)
            nearest = depth;
    }
    return nearest == FLT_MAX ? -1 : nearest;
}

static float model_clamp_uv(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static void model_project_uvs_for_face(const int16_t *verts,
                                       int tri_a, int tri_b, int tri_c,
                                       int tex_a, int tex_b, int tex_c,
                                       float *u, float *v) {
    u[0] = 0.0f; u[1] = 1.0f; u[2] = 0.0f;
    v[0] = 0.0f; v[1] = 0.0f; v[2] = 1.0f;

    float v1x = (float)verts[tex_a * 3];
    float v1y = (float)verts[tex_a * 3 + 1];
    float v1z = (float)verts[tex_a * 3 + 2];
    float v2x = (float)verts[tex_b * 3] - v1x;
    float v2y = (float)verts[tex_b * 3 + 1] - v1y;
    float v2z = (float)verts[tex_b * 3 + 2] - v1z;
    float v3x = (float)verts[tex_c * 3] - v1x;
    float v3y = (float)verts[tex_c * 3 + 1] - v1y;
    float v3z = (float)verts[tex_c * 3 + 2] - v1z;
    float v4x = (float)verts[tri_a * 3] - v1x;
    float v4y = (float)verts[tri_a * 3 + 1] - v1y;
    float v4z = (float)verts[tri_a * 3 + 2] - v1z;
    float v5x = (float)verts[tri_b * 3] - v1x;
    float v5y = (float)verts[tri_b * 3 + 1] - v1y;
    float v5z = (float)verts[tri_b * 3 + 2] - v1z;
    float v6x = (float)verts[tri_c * 3] - v1x;
    float v6y = (float)verts[tri_c * 3 + 1] - v1y;
    float v6z = (float)verts[tri_c * 3 + 2] - v1z;

    float v7x = v2y * v3z - v2z * v3y;
    float v7y = v2z * v3x - v2x * v3z;
    float v7z = v2x * v3y - v2y * v3x;

    float v8x = v3y * v7z - v3z * v7y;
    float v8y = v3z * v7x - v3x * v7z;
    float v8z = v3x * v7y - v3y * v7x;
    float denom = v8x * v2x + v8y * v2y + v8z * v2z;
    if (fabsf(denom) < 1.0e-6f)
        return;
    float inv = 1.0f / denom;
    u[0] = (v8x * v4x + v8y * v4y + v8z * v4z) * inv;
    u[1] = (v8x * v5x + v8y * v5y + v8z * v5z) * inv;
    u[2] = (v8x * v6x + v8y * v6y + v8z * v6z) * inv;

    v8x = v2y * v7z - v2z * v7y;
    v8y = v2z * v7x - v2x * v7z;
    v8z = v2x * v7y - v2y * v7x;
    denom = v8x * v3x + v8y * v3y + v8z * v3z;
    if (fabsf(denom) < 1.0e-6f)
        return;
    inv = 1.0f / denom;
    v[0] = (v8x * v4x + v8y * v4y + v8z * v4z) * inv;
    v[1] = (v8x * v5x + v8y * v5y + v8z * v5z) * inv;
    v[2] = (v8x * v6x + v8y * v6y + v8z * v6z) * inv;
}

void models_recompute_texture_uvs_from_vertices(ModelEntry *entry,
                                                const int16_t *verts) {
    if (!entry || !entry->face_uvs || !verts)
        return;
    Mesh *mesh = &entry->model.meshes[0];
    if (!mesh->texcoords || !entry->face_indices)
        return;
    for (int fi = 0; fi < entry->face_count; fi++) {
        ModelFaceUvInfo *info = &entry->face_uvs[fi];
        if (!info->textured)
            continue;
        int tri_a = entry->face_indices[fi * 3];
        int tri_b = entry->face_indices[fi * 3 + 1];
        int tri_c = entry->face_indices[fi * 3 + 2];
        if (tri_a >= entry->base_vert_count || tri_b >= entry->base_vert_count
                || tri_c >= entry->base_vert_count
                || info->tex_a >= entry->base_vert_count
                || info->tex_b >= entry->base_vert_count
                || info->tex_c >= entry->base_vert_count)
            continue;
        float u[3], v[3];
        model_project_uvs_for_face(verts, tri_a, tri_b, tri_c,
                                   info->tex_a, info->tex_b, info->tex_c,
                                   u, v);
        for (int j = 0; j < 3; j++) {
            float cu = model_clamp_uv(u[j]);
            float cv = model_clamp_uv(v[j]);
            int out = (fi * 3 + j) * 2;
            mesh->texcoords[out] = info->u_base + cu * info->u_scale;
            mesh->texcoords[out + 1] = info->v_base + cv * info->v_scale;
        }
    }
    UpdateMeshBuffer(*mesh, 1, mesh->texcoords,
                     mesh->vertexCount * 2 * sizeof(float), 0);
}

void models_free(ModelSet *set) {
    if (!set) return;
    for (int i = 0; i < set->count; i++) {
        ModelEntry *entry = &set->entries[i];
        if (entry->model.meshes) UnloadModel(entry->model);
        free(entry->base_verts);
        free(entry->rest_verts);
        free(entry->vertex_skins);
        free(entry->face_indices);
        free(entry->face_uvs);
    }
    free(set->entries);
    free(set);
}

/* World tile that becomes the arena's (0, 0): the terrain, objects and object
 * animations are stored in world tiles and loaded in arena tiles. */
#define FC_WORLD_ORIGIN_X 2368
#define FC_WORLD_ORIGIN_Y 5056

static void fc_mesh_free_arrays(Mesh *mesh) {
    free(mesh->vertices);
    free(mesh->texcoords);
    free(mesh->normals);
    free(mesh->colors);
}

/* Uploads a triangle list read from an asset (three vertices per face) as a
 * one-mesh model. Flat face normals come from the vertices as read; arena
 * meshes are then moved from world tiles to arena tiles (Z is negated world
 * Y). Loaders allocate the zeroed normals array while reading, so a failed
 * allocation fails the load. */
static Model fc_mesh_upload(Mesh mesh, int world_to_arena) {
    const float *v = mesh.vertices;
    for (int i = 0; i < mesh.triangleCount; i++) {
        int base = i * 9;
        float e1x = v[base + 3] - v[base + 0];
        float e1y = v[base + 4] - v[base + 1];
        float e1z = v[base + 5] - v[base + 2];
        float e2x = v[base + 6] - v[base + 0];
        float e2y = v[base + 7] - v[base + 1];
        float e2z = v[base + 8] - v[base + 2];
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 0.0001f) { nx /= len; ny /= len; nz /= len; }
        for (int k = 0; k < 3; k++) {
            mesh.normals[base + k * 3 + 0] = nx;
            mesh.normals[base + k * 3 + 1] = ny;
            mesh.normals[base + k * 3 + 2] = nz;
        }
    }
    if (world_to_arena) {
        for (int i = 0; i < mesh.vertexCount; i++) {
            mesh.vertices[i * 3 + 0] -= (float)FC_WORLD_ORIGIN_X;
            mesh.vertices[i * 3 + 2] += (float)FC_WORLD_ORIGIN_Y;
        }
    }
    UploadMesh(&mesh, false);
    return LoadModelFromMesh(mesh);
}

/* Reads one model record. Textured (MDL3) records end with a MUV1 block
 * holding each face's UV projection, used to re-texture posed vertices.
 * Arrays go straight into the entry so models_free releases them on
 * failure. */
static void model_entry_read(FcReader *r, ModelEntry *entry, int has_textures,
                             Texture2D atlas_texture) {
    uint32_t id = fc_read_u32(r);
    int vertex_count = fc_read_u16(r);
    int face_count = fc_read_u16(r);
    int base_count = fc_read_u16(r);
    Mesh mesh = {.vertexCount = vertex_count, .triangleCount = face_count};
    mesh.vertices = fc_read_array(r, (size_t)vertex_count * 3, sizeof(float));
    mesh.colors = fc_read_array(r, (size_t)vertex_count * 4, 1);
    if (has_textures)
        mesh.texcoords = fc_read_array(r, (size_t)vertex_count * 2,
                                       sizeof(float));
    entry->base_verts = fc_read_array(r, (size_t)base_count * 3,
                                      sizeof(int16_t));
    entry->vertex_skins = fc_read_array(r, base_count, 1);
    entry->face_indices = fc_read_array(r, (size_t)face_count * 3,
                                        sizeof(uint16_t));
    fc_reader_skip(r, face_count); /* face render priorities: unused */
    if (has_textures) {
        uint32_t uv_magic = fc_read_u32(r);
        uint32_t uv_count = fc_read_u32(r);
        fc_reader_require(r, uv_magic == MUV1_MAGIC
                             && uv_count == (uint32_t)face_count,
                          "textured model without a MUV1 block");
        entry->face_uvs = fc_alloc_array(r, face_count,
                                         sizeof(ModelFaceUvInfo));
        for (int i = 0; entry->face_uvs && i < face_count; i++) {
            ModelFaceUvInfo *uv = &entry->face_uvs[i];
            uv->textured = fc_read_u8(r) != 0;
            fc_reader_skip(r, 3);
            uv->tex_a = fc_read_u16(r);
            uv->tex_b = fc_read_u16(r);
            uv->tex_c = fc_read_u16(r);
            uv->u_base = fc_read_f32(r);
            uv->v_base = fc_read_f32(r);
            uv->u_scale = fc_read_f32(r);
            uv->v_scale = fc_read_f32(r);
            fc_reader_skip(r, 4); /* V repeat margin: unused, V is clamped */
        }
    }
    entry->rest_verts = fc_alloc_array(r, (size_t)vertex_count * 3,
                                       sizeof(float));
    mesh.normals = fc_alloc_array(r, (size_t)vertex_count * 3, sizeof(float));
    if (!r->ok) {
        fc_mesh_free_arrays(&mesh);
        return;
    }

    /* OSRS units -> tile units, flip Z for Raylib */
    for (int i = 0; i < vertex_count; i++) {
        mesh.vertices[i * 3] /= 128.0f;
        mesh.vertices[i * 3 + 1] /= 128.0f;
        mesh.vertices[i * 3 + 2] /= -128.0f;
    }
    memcpy(entry->rest_verts, mesh.vertices,
           (size_t)vertex_count * 3 * sizeof(float));
    entry->model_id = id;
    entry->model = fc_mesh_upload(mesh, 0);
    if (has_textures)
        entry->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture =
            atlas_texture;
    entry->base_vert_count = base_count;
    entry->face_count = face_count;
    fprintf(stderr, "  model %u: %d tris, %d base verts\n", id, face_count,
            base_count);
}

ModelSet *models_load(const char *path, Texture2D atlas_texture) {
    ModelSet *set = calloc(1, sizeof(ModelSet));
    FcReader r;
    if (!set || !fc_reader_open(&r, path)) {
        free(set);
        return NULL;
    }

    uint32_t magic = fc_read_u32(&r);
    uint32_t count = fc_read_u32(&r);
    fc_reader_require(&r, magic == MDL2_MAGIC || magic == MDL3_MAGIC,
                      "not a model file");
    set->has_textures = magic == MDL3_MAGIC;
    fc_reader_require(&r, !set->has_textures || atlas_texture.id != 0,
                      "textured models need the shared atlas");
    uint32_t *offsets = fc_read_array(&r, count, sizeof(uint32_t));
    set->entries = fc_alloc_array(&r, count, sizeof(ModelEntry));
    if (r.ok) set->count = (int)count;
    for (int m = 0; r.ok && m < set->count; m++) {
        fc_reader_seek(&r, offsets[m]);
        model_entry_read(&r, &set->entries[m], set->has_textures,
                         atlas_texture);
    }
    free(offsets);
    fc_reader_close(&r);
    if (!r.ok) {
        models_free(set);
        return NULL;
    }
    fprintf(stderr, "models: loaded %d from %s\n", set->count, path);
    return set;
}

/* Npc Models */

/* Model id of each NPC type in fc_npcs.models. */
static const uint32_t NPC_MODEL_IDS[NPC_TYPE_COUNT] = {
    [NPC_TZ_KIH] = 3116, [NPC_TZ_KEK] = 3118, [NPC_TZ_KEK_SM] = 3120,
    [NPC_TOK_XIL] = 3121, [NPC_YT_MEJKOT] = 3123, [NPC_KET_ZEK] = 3125,
    [NPC_TZTOK_JAD] = 3127, [NPC_YT_HURKOT] = 3128,
};

ModelEntry *npc_model_find(ModelSet *set, int npc_type) {
    if (npc_type <= NPC_NONE || npc_type >= NPC_TYPE_COUNT) return NULL;
    return model_find(set, NPC_MODEL_IDS[npc_type]);
}

/* Model Animation */

/* Re-expands posed base vertices into the uploaded mesh, like the exporter's
 * expand_model: the mesh has three vertices per face, taken from base vertices
 * through face_indices, in the units models_load uses (tiles, with OSRS's
 * downward Y negated and Z flipped). */
void fc_model_animation_upload(ModelEntry *entry, AnimModelState *state) {
    if (!entry || !state) return;
    models_recompute_texture_uvs_from_vertices(entry, state->verts);
    Mesh *mesh = &entry->model.meshes[0];
    for (int i = 0; i < entry->face_count * 3; i++) {
        const int16_t *vertex = &state->verts[entry->face_indices[i] * 3];
        mesh->vertices[i * 3] = (float)vertex[0] / 128.0f;
        mesh->vertices[i * 3 + 1] = (float)(-vertex[1]) / 128.0f;
        mesh->vertices[i * 3 + 2] = (float)vertex[2] / -128.0f;
    }
    UpdateMeshBuffer(*mesh, 0, mesh->vertices,
                     entry->face_count * 9 * sizeof(float), 0);
}

/* Advances a track playing `sequence` by dt seconds. Switching sequences
 * restarts at frame 0. Past the last frame a looping track steps back by the
 * sequence's frame step (to frame 0 without one); a play-once track holds its
 * last frame. */
static void anim_track_step(AnimTrack *track, const AnimSequence *sequence,
                            float dt, int play_once) {
    if (track->sequence != sequence->seq_id) {
        track->sequence = sequence->seq_id;
        track->frame = 0;
        track->timer = anim_frame_duration(sequence, 0);
    }
    track->timer -= dt;
    while (track->timer <= 0.0f &&
           (!play_once || track->frame < sequence->frame_count - 1)) {
        track->frame++;
        if (track->frame >= sequence->frame_count) {
            if (sequence->frame_step > 0 &&
                sequence->frame_step <= sequence->frame_count) {
                track->frame -= sequence->frame_step;
            } else {
                track->frame = 0;
            }
        }
        track->timer += anim_frame_duration(sequence, track->frame);
    }
    if (play_once && track->timer <= 0.0f) track->timer = 0.016f;
}

/* Plays animation_id on a model of its own (spot animations, animated
 * objects). A new pose state starts phase_ticks frames into the loop. */
void fc_model_animation_update(ModelEntry *entry,
                               AnimCache *cache,
                               AnimModelState **state,
                               AnimTrack *track,
                               int animation_id,
                               float dt,
                               float phase_ticks) {
    if (!entry || !cache || animation_id < 0 ||
        !entry->vertex_skins || !state || !track) return;
    AnimSequence *sequence = anim_get_sequence(cache, (uint16_t)animation_id);
    if (!sequence || sequence->frame_count == 0) return;
    if (!*state || (*state)->vert_count != entry->base_vert_count) {
        if (*state) anim_model_state_free(*state);
        *state = anim_model_state_create(entry->vertex_skins,
                                         entry->base_vert_count);
        track->sequence = (uint16_t)animation_id;
        track->frame = (int)phase_ticks % sequence->frame_count;
        if (track->frame < 0) track->frame = 0;
        track->timer = anim_frame_duration(sequence, track->frame);
    }
    anim_track_step(track, sequence, dt, 0);
    AnimFrameData *frame = &sequence->frames[track->frame].frame;
    AnimFrameBase *framebase = anim_get_framebase(cache, frame->framebase_id);
    if (framebase) {
        anim_apply_frame(*state, entry->base_verts, frame, framebase);
        fc_model_animation_upload(entry, *state);
    }
}

/* Objects Loader */
/**
 * @fileoverview Loads placed map objects from .objects binary into a single raylib Model.
 *
 * Binary format (OBJ2): vertices + colors + texcoords (texture atlas support).
 *
 * The shared texture atlas becomes the model's diffuse texture. Vertex
 * colors are multiplied by the texture sample: textured faces use white
 * vertex color + real texture, non-textured faces use HSL vertex color +
 * white atlas pixel.
 */

typedef struct {
    Model model;
} ObjectMesh;

typedef struct {
    uint32_t model_id;
    uint32_t obj_id;
    int32_t animation_id;
    int32_t world_x;
    int32_t world_y;
    uint8_t plane;
    uint8_t obj_type;
    uint8_t rotation;
    uint8_t flags; /* unused: zero for every shipped placement */
    float pos_x;
    float pos_y;
    float pos_z;
    float phase_ticks;
} ObjectAnimPlacement;

typedef struct {
    ObjectAnimPlacement *rows;
    int count;
} ObjectAnimSet;

#define OBJ2_MAGIC 0x4F424A32  /* "OBJ2" with texcoords */
#define OANM_MAGIC 0x4D4E414F  /* "OANM" animated object placements */
#define OANM_VERSION 1
ObjectMesh* objects_load(const char* path, Texture2D atlas_texture) {
    ObjectMesh* om = (ObjectMesh*)calloc(1, sizeof(ObjectMesh));
    FcReader r;
    if (!om || !fc_reader_open(&r, path)) {
        free(om);
        return NULL;
    }

    uint32_t magic = fc_read_u32(&r);
    fc_reader_require(&r, magic == OBJ2_MAGIC, "not an OBJ2 objects file");
    uint32_t placement_count = fc_read_u32(&r);
    fc_reader_skip(&r, 8); /* minimum world x and y: unused */
    uint32_t total_verts = fc_read_u32(&r);

    Mesh mesh = { 0 };
    mesh.vertexCount = (int)total_verts;
    mesh.triangleCount = (int)(total_verts / 3);
    mesh.vertices = fc_read_array(&r, (size_t)total_verts * 3, sizeof(float));
    mesh.colors = fc_read_array(&r, (size_t)total_verts * 4, 1);
    mesh.texcoords = fc_read_array(&r, (size_t)total_verts * 2, sizeof(float));
    mesh.normals = fc_alloc_array(&r, (size_t)total_verts * 3, sizeof(float));
    fc_reader_close(&r);
    if (!r.ok) {
        fc_mesh_free_arrays(&mesh);
        free(om);
        return NULL;
    }
    fprintf(stderr, "objects_load: %u placements, %u verts\n",
            placement_count, total_verts);

    om->model = fc_mesh_upload(mesh, 1);
    om->model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = atlas_texture;
    return om;
}

void objects_free(ObjectMesh* om) {
    if (!om) return;
    UnloadModel(om->model);
    free(om);
}

_Static_assert(sizeof(ObjectAnimPlacement) == 40,
               "OANM rows must retain their 40-byte file layout");

void object_anims_free(ObjectAnimSet* set) {
    if (!set) return;
    free(set->rows);
    free(set);
}

ObjectAnimSet* object_anims_load(const char* path) {
    ObjectAnimSet* set = (ObjectAnimSet*)calloc(1, sizeof(*set));
    FcReader r;
    if (!set || !fc_reader_open(&r, path)) {
        free(set);
        return NULL;
    }

    uint32_t magic = fc_read_u32(&r);
    uint32_t version = fc_read_u32(&r);
    uint32_t count = fc_read_u32(&r);
    fc_reader_require(&r, magic == OANM_MAGIC && version == OANM_VERSION,
                      "unsupported object animation format");
    set->rows = fc_read_array(&r, count, sizeof(*set->rows));
    set->count = (int)count;
    fc_reader_close(&r);
    if (!r.ok) {
        object_anims_free(set);
        return NULL;
    }
    /* world tiles -> arena tiles, as for the meshes */
    for (int i = 0; i < set->count; i++) {
        ObjectAnimPlacement* row = &set->rows[i];
        row->pos_x -= (float)FC_WORLD_ORIGIN_X;
        row->pos_z += (float)FC_WORLD_ORIGIN_Y;
        row->world_x -= FC_WORLD_ORIGIN_X;
        row->world_y -= FC_WORLD_ORIGIN_Y;
    }
    fprintf(stderr, "object_anims_load: loaded %d animated placements from %s\n",
            set->count, path);
    return set;
}

/* Terrain Loader */
/**
 * @fileoverview Loads terrain mesh from .terrain binary into raylib Model.
 *
 * Binary format:
 *   magic: uint32 "TERR" (0x54455252)
 *   vertex_count: uint32
 *   region_count: uint32
 *   min_world_x: int32
 *   min_world_y: int32
 *   vertices: float32[vertex_count * 3]
 *   colors: uint8[vertex_count * 4]
 *   heightmap: int32 min_x, int32 min_y, uint32 width, uint32 height,
 *              float32[width * height]
 */

typedef struct {
    Model model;
    float *heightmap;
    int hm_min_x;
    int hm_min_y;
    int hm_width;
    int hm_height;
} TerrainMesh;

#define TERR_MAGIC 0x54455252

void terrain_free(TerrainMesh* tm) {
    if (!tm) return;
    if (tm->model.meshes) UnloadModel(tm->model);
    free(tm->heightmap);
    free(tm);
}

TerrainMesh* terrain_load(const char* path) {
    TerrainMesh* tm = (TerrainMesh*)calloc(1, sizeof(TerrainMesh));
    FcReader r;
    if (!tm || !fc_reader_open(&r, path)) {
        free(tm);
        return NULL;
    }

    uint32_t magic = fc_read_u32(&r);
    fc_reader_require(&r, magic == TERR_MAGIC, "not a terrain file");
    uint32_t vert_count = fc_read_u32(&r);
    uint32_t region_count = fc_read_u32(&r);
    int32_t min_wx = fc_read_i32(&r);
    int32_t min_wy = fc_read_i32(&r);

    Mesh mesh = { 0 };
    mesh.vertexCount = (int)vert_count;
    mesh.triangleCount = (int)(vert_count / 3);
    mesh.vertices = fc_read_array(&r, (size_t)vert_count * 3, sizeof(float));
    mesh.colors = fc_read_array(&r, (size_t)vert_count * 4, 1);
    mesh.normals = fc_alloc_array(&r, (size_t)vert_count * 3, sizeof(float));

    /* the heightmap follows the colors; its origin moves to arena tiles */
    tm->hm_min_x = fc_read_i32(&r) - FC_WORLD_ORIGIN_X;
    tm->hm_min_y = fc_read_i32(&r) - FC_WORLD_ORIGIN_Y;
    uint32_t hm_w = fc_read_u32(&r);
    uint32_t hm_h = fc_read_u32(&r);
    fc_reader_require(&r, hm_w > 0 && hm_h > 0 && hm_w <= 4096 && hm_h <= 4096,
                      "invalid terrain heightmap dimensions");
    tm->hm_width = (int)hm_w;
    tm->hm_height = (int)hm_h;
    tm->heightmap = fc_read_array(&r, (size_t)hm_w * hm_h, sizeof(float));
    fc_reader_close(&r);
    if (!r.ok) {
        fc_mesh_free_arrays(&mesh);
        terrain_free(tm);
        return NULL;
    }
    fprintf(stderr, "terrain_load: %u verts, %u regions, origin (%d, %d)\n",
            vert_count, region_count, min_wx, min_wy);

    tm->model = fc_mesh_upload(mesh, 1);
    fprintf(stderr, "terrain heightmap: %dx%d, arena origin (%d, %d)\n",
            tm->hm_width, tm->hm_height, tm->hm_min_x, tm->hm_min_y);
    return tm;
}

/* query terrain height at a world tile position (tile corner) */
float terrain_height_at(TerrainMesh* tm, int world_x, int world_y) {
    if (!tm) return -2.0f;
    int lx = world_x - tm->hm_min_x;
    int ly = world_y - tm->hm_min_y;
    if (lx < 0 || lx >= tm->hm_width || ly < 0 || ly >= tm->hm_height)
        return -2.0f;
    return tm->heightmap[lx + ly * tm->hm_width];
}

/* Spotanims */

typedef struct {
    uint32_t id;
    int32_t model_id;
    int32_t animation_id;
    uint32_t resize_xy;
    uint32_t resize_z;
    uint32_t rotation;
    int32_t brightness;
    int32_t shadow;
} SpotAnimDef;

typedef struct {
    SpotAnimDef *defs;
    int count;
} SpotAnimSet;

#define SPOTANIM_MAGIC 0x544F5053u
#define SPOTANIM_VERSION 1u

void spotanims_free(SpotAnimSet *set) {
    if (!set) return;
    free(set->defs);
    free(set);
}

_Static_assert(sizeof(SpotAnimDef) == 32,
               "spot animation rows must retain their 32-byte file layout");

SpotAnimSet *spotanims_load(const char *path) {
    SpotAnimSet *set = (SpotAnimSet *)calloc(1, sizeof(*set));
    FcReader r;
    if (!set || !fc_reader_open(&r, path)) {
        free(set);
        return NULL;
    }

    uint32_t magic = fc_read_u32(&r);
    uint32_t version = fc_read_u32(&r);
    uint32_t count = fc_read_u32(&r);
    fc_reader_require(&r, magic == SPOTANIM_MAGIC && version == SPOTANIM_VERSION,
                      "unsupported spot animation format");
    set->defs = fc_read_array(&r, count, sizeof(*set->defs));
    set->count = (int)count;
    fc_reader_close(&r);
    if (!r.ok) {
        spotanims_free(set);
        return NULL;
    }
    fprintf(stderr, "spotanims: loaded %d from %s\n", set->count, path);
    return set;
}

const SpotAnimDef *spotanim_find(const SpotAnimSet *set, int id) {
    if (!set || id < 0) return NULL;
    for (int i = 0; i < set->count; i++) {
        if ((int)set->defs[i].id == id) return &set->defs[i];
    }
    return NULL;
}

/* Player Appearance */

/* Model id given to the composed player model. */
#define FC_PLAYER_MODEL_ID 0xFC000001u

typedef struct {
    ModelSet *parts;
    ModelSet *model;
    struct { uint32_t item_id, hide_mask; } records[64];
    int record_count;
    int worn_ids[FC_EQUIPMENT_SLOTS];
} FcPlayerAppearance;

int fc_player_appearance_load(FcPlayerAppearance *appearance) {
    memset(appearance, 0, sizeof(*appearance));
    const char *path = "fc_player.parts";
    FcReader r;
    if (!fc_reader_open(&r, path)) return 0;
    uint32_t magic = fc_read_u32(&r);
    uint32_t count = fc_read_u32(&r);
    fc_reader_require(&r, magic == 0x31504346 && count > 0 && count <= 64,
                      "bad header");
    for (uint32_t i = 0; r.ok && i < count; i++) {
        uint32_t item_id = fc_read_u32(&r);
        uint32_t hide_mask = fc_read_u32(&r);
        fc_reader_require(&r, hide_mask <= 127, "bad hide mask");
        /* The map also covers items the simulation does not offer. */
        if (!fc_item_definition((int)item_id)) continue;
        appearance->records[appearance->record_count].item_id = item_id;
        appearance->records[appearance->record_count].hide_mask = hide_mask;
        appearance->record_count++;
    }
    fc_reader_require(&r, r.pos == r.size, "unexpected data after the last item");
    fc_reader_close(&r);
    if (!r.ok) { fprintf(stderr, "Invalid player appearance map: %s\n", path); return 0; }
    appearance->parts = models_load("fc_player.models", (Texture2D){0});
    if (!appearance->parts || appearance->parts->has_textures) return 0;
    for (int i = 0; i < 7; i++)
        if (!model_find(appearance->parts, 0xFC100000u + (uint32_t)i)) return 0;
    for (int i = 0; i < appearance->record_count; i++)
        if (!model_find(appearance->parts, appearance->records[i].item_id)) return 0;
    return 1;
}

static ModelSet *compose(ModelEntry *parts[], int count, uint32_t id) {
    int vertices = 0, faces = 0;
    for (int i = 0; i < count; i++) {
        vertices += parts[i]->base_vert_count;
        faces += parts[i]->face_count;
    }
    if (vertices <= 0 || vertices > UINT16_MAX || faces <= 0) return NULL;
    ModelSet *set = calloc(1, sizeof(*set));
    if (!set) return NULL;
    set->entries = calloc(1, sizeof(*set->entries));
    if (!set->entries) { free(set); return NULL; }
    set->count = 1;
    ModelEntry *out = set->entries;
    out->model_id = id;
    out->base_vert_count = vertices;
    out->face_count = faces;
    out->base_verts = malloc((size_t)vertices * 3 * sizeof(int16_t));
    out->vertex_skins = malloc((size_t)vertices);
    out->face_indices = malloc((size_t)faces * 3 * sizeof(uint16_t));
    out->rest_verts = malloc((size_t)faces * 9 * sizeof(float));
    Mesh mesh = {.vertexCount = faces * 3, .triangleCount = faces};
    mesh.vertices = malloc((size_t)faces * 9 * sizeof(float));
    mesh.normals = malloc((size_t)faces * 9 * sizeof(float));
    mesh.colors = malloc((size_t)faces * 12);
    if (!out->base_verts || !out->vertex_skins || !out->face_indices ||
        !out->rest_verts || !mesh.vertices || !mesh.normals || !mesh.colors) {
        free(mesh.vertices); free(mesh.normals); free(mesh.colors);
        models_free(set);
        return NULL;
    }
    int vertex_offset = 0, face_offset = 0;
    for (int i = 0; i < count; i++) {
        const ModelEntry *part = parts[i];
        const Mesh *source = &part->model.meshes[0];
        memcpy(out->base_verts + vertex_offset * 3, part->base_verts,
               (size_t)part->base_vert_count * 3 * sizeof(int16_t));
        memcpy(out->vertex_skins + vertex_offset, part->vertex_skins,
               (size_t)part->base_vert_count);
        for (int f = 0; f < part->face_count * 3; f++)
            out->face_indices[face_offset * 3 + f] =
                (uint16_t)(part->face_indices[f] + vertex_offset);
        memcpy(mesh.vertices + face_offset * 9, part->rest_verts,
               (size_t)part->face_count * 9 * sizeof(float));
        memcpy(mesh.normals + face_offset * 9, source->normals,
               (size_t)part->face_count * 9 * sizeof(float));
        memcpy(mesh.colors + face_offset * 12, source->colors,
               (size_t)part->face_count * 12);
        vertex_offset += part->base_vert_count;
        face_offset += part->face_count;
    }
    memcpy(out->rest_verts, mesh.vertices, (size_t)faces * 9 * sizeof(float));
    UploadMesh(&mesh, true);
    out->model = LoadModelFromMesh(mesh);
    return set;
}

/* 1 = rebuilt, 0 = unchanged, -1 = missing/invalid asset or allocation failure. */
int fc_player_appearance_sync(FcPlayerAppearance *appearance,
                               const FcPlayer *player, uint32_t model_id) {
    int ids[FC_EQUIPMENT_SLOTS];
    for (int i = 0; i < FC_EQUIPMENT_SLOTS; i++) ids[i] = player->equipment[i].item_id;
    if (appearance->model && appearance->model->entries[0].model_id == model_id &&
        memcmp(ids, appearance->worn_ids, sizeof(ids)) == 0) return 0;
    if (!appearance->parts) return -1;
    ModelEntry *selected[7 + FC_EQUIPMENT_SLOTS];
    int count = 0;
    uint32_t hidden = 0;
    for (int slot = 0; slot < FC_EQUIPMENT_SLOTS; slot++) {
        if (!ids[slot] || slot == FC_EQUIP_SLOT_AMMO || slot == FC_EQUIP_SLOT_RING) continue;
        int record = 0;
        while (record < appearance->record_count &&
               appearance->records[record].item_id != (uint32_t)ids[slot]) record++;
        if (record == appearance->record_count) return -1;
        hidden |= appearance->records[record].hide_mask;
    }
    /* Identity kits precede equipped models, matching client composition. */
    for (int body = 0; body < 7; body++)
        if (!(hidden & (1u << body)))
            selected[count++] = model_find(appearance->parts, 0xFC100000u + (uint32_t)body);
    for (int slot = 0; slot < FC_EQUIPMENT_SLOTS; slot++)
        if (ids[slot] && slot != FC_EQUIP_SLOT_AMMO && slot != FC_EQUIP_SLOT_RING)
            selected[count++] = model_find(appearance->parts, (uint32_t)ids[slot]);
    for (int i = 0; i < count; i++) if (!selected[i]) return -1;
    ModelSet *model = compose(selected, count, model_id);
    if (!model) return -1;
    models_free(appearance->model);
    appearance->model = model;
    memcpy(appearance->worn_ids, ids, sizeof(ids));
    return 1;
}

void fc_player_appearance_free(FcPlayerAppearance *appearance) {
    models_free(appearance->model);
    models_free(appearance->parts);
    memset(appearance, 0, sizeof(*appearance));
}

/* ===================== Game interface ===================== */

/* Ui Reference */

/*
 * OSRS UI reference constants copied from the osrs-dumps interface
 * definitions (toplevel_osrs_stretch, orbs, inventory and
 * combat_interface). These mirror what Void/RSMod open
 * server-side; the viewer should use these as layout authority instead of
 * hand-tuned gameframe coordinates.
 */

#define FC_OSRS_MAP_CONTAINER_W 211.0f
#define FC_OSRS_MAP_CONTAINER_H 207.0f
#define FC_OSRS_MINIMAP_X 53.0f
#define FC_OSRS_MINIMAP_Y 8.0f
#define FC_OSRS_MINIMAP_W 152.0f
#define FC_OSRS_MINIMAP_H 152.0f
#define FC_OSRS_MAP_SURROUND_X 29.0f
#define FC_OSRS_MAP_SURROUND_Y 0.0f
#define FC_OSRS_MAP_SURROUND_W 182.0f
#define FC_OSRS_MAP_SURROUND_H 166.0f
#define FC_OSRS_COMPASS_X 34.0f
#define FC_OSRS_COMPASS_Y 5.0f
#define FC_OSRS_COMPASS_W 35.0f
#define FC_OSRS_COMPASS_H 35.0f

#define FC_OSRS_ORBS_X 0.0f
#define FC_OSRS_ORBS_Y 10.0f
#define FC_OSRS_HP_X 0.0f
#define FC_OSRS_HP_Y 37.0f
#define FC_OSRS_PRAYER_X 0.0f
#define FC_OSRS_PRAYER_Y 71.0f
#define FC_OSRS_RUN_X 10.0f
#define FC_OSRS_RUN_Y 103.0f
#define FC_OSRS_WORLDMAP_X 177.0f
#define FC_OSRS_WORLDMAP_Y 137.0f

#define FC_OSRS_CHAT_W 519.0f
#define FC_OSRS_CHAT_H 165.0f
#define FC_OSRS_SIDE_MENU_W 241.0f
#define FC_OSRS_SIDE_MENU_H 335.0f
#define FC_OSRS_SIDE_CONTENT_X 25.0f
#define FC_OSRS_SIDE_CONTENT_Y 37.0f
#define FC_OSRS_SIDE_CONTENT_W 190.0f
#define FC_OSRS_SIDE_CONTENT_H 261.0f
#define FC_OSRS_SIDE_TOP_Y 0.0f
#define FC_OSRS_SIDE_BOTTOM_Y 298.0f

#define FC_OSRS_INVENTORY_SLOT_X 14.0f
#define FC_OSRS_INVENTORY_SLOT_Y 8.0f
#define FC_OSRS_INVENTORY_SLOT_STEP_X 42.0f
#define FC_OSRS_INVENTORY_SLOT_STEP_Y 36.0f
#define FC_OSRS_INVENTORY_SLOT_W 32.0f
#define FC_OSRS_INVENTORY_SLOT_H 32.0f

typedef struct FcUiStoneRef {
    int logical_tab;              /* FcUiTab, or -1 for an inert stone */
    const char *stone_asset;
    const char *icon_asset;
    Rectangle rect;
    Rectangle icon_rect;
} FcUiStoneRef;

static const FcUiStoneRef FC_OSRS_SIDE_STONES[] = {
    {0, "side_stone_highlights_0", "side_icon_combat",    {0,   0, 38, 36}, {4,   0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_stats",    {38,  0, 33, 36}, {38,  0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_quests",   {71,  0, 38, 36}, {71,  0, 33, 36}},
    {1, "side_stone_highlights_4", "side_icon_inventory", {104, 0, 33, 36}, {104, 0, 33, 36}},
    {2, "side_stone_highlights_4", "side_icon_equipment", {137, 0, 33, 36}, {137, 0, 33, 36}},
    {3, "side_stone_highlights_4", "side_icon_prayer",    {170, 0, 33, 36}, {170, 0, 33, 36}},
    {-1, "side_stone_highlights_1", "side_icon_magic",    {203, 0, 38, 36}, {204, 0, 33, 36}},
    {-1, "side_stone_highlights_2", "side_icon_clan",     {0,   0, 38, 36}, {4,   0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_friends",  {38,  0, 33, 36}, {38,  0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_grouping", {71,  0, 33, 36}, {71,  0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_logout",   {104, 0, 33, 36}, {104, 0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_options",  {137, 0, 33, 36}, {137, 0, 33, 36}},
    {-1, "side_stone_highlights_4", "side_icon_emotes",   {170, 0, 33, 36}, {170, 0, 33, 36}},
    {-1, "side_stone_highlights_3", "side_icon_music",    {203, 0, 38, 36}, {204, 0, 33, 36}},
};

/* Button, icon and label rectangles of a weapon's three combat styles. */
typedef struct FcUiCombatStyleRef {
    Rectangle rect;
    Rectangle icon_rect;
    Rectangle text_rect;
} FcUiCombatStyleRef;

static const FcUiCombatStyleRef FC_OSRS_COMBAT_STYLES[] = {
    {{20,  46, 68, 47}, {37,  51, 34, 24}, {20,  67, 68, 13}},
    {{102, 46, 68, 47}, {119, 51, 34, 24}, {102, 76, 68, 13}},
    {{20,  99, 68, 47}, {37, 104, 34, 24}, {20, 129, 68, 13}},
};

static const Rectangle FC_OSRS_COMBAT_HEADER = {10, 0, 170, 44};
static const Rectangle FC_OSRS_COMBAT_TITLE = {10, 6, 170, 14};
static const Rectangle FC_OSRS_COMBAT_LEVEL = {10, 26, 170, 12};
static const Rectangle FC_OSRS_COMBAT_CATEGORY = {0, 233, 190, 28};

/* Ui Assets */

/* Sprites loaded from data/sprites/ui/<name>.png. The viewer refuses to
 * start unless every one of them loads. */
static const char *const g_ui_asset_names[] = {
    "tradebacking_dark",
    "chatbox_bg",
    "main_stones_bottom",
    "side_stone_highlights_0",
    "side_stone_highlights_1",
    "side_stone_highlights_2",
    "side_stone_highlights_3",
    "side_stone_highlights_4",
    "osrs_stretch_side_topbottom_0",
    "osrs_stretch_side_topbottom_1",
    "osrs_stretch_side_columns_0",
    "osrs_stretch_side_columns_1",
    "compass",
    "osrs_stretch_mapsurround",
    "side_icon_combat",
    "side_icon_stats",
    "side_icon_quests",
    "side_icon_inventory",
    "side_icon_equipment",
    "side_icon_prayer",
    "side_icon_magic",
    "side_icon_clan",
    "side_icon_friends",
    "side_icon_options",
    "side_icon_logout",
    "side_icon_emotes",
    "side_icon_music",
    "side_icon_grouping",
    "orb_frame_0",
    "orb_frame_1",
    "orb_filler_0",
    "orb_filler_1",
    "orb_filler_4",
    "orb_filler_5",
    "orb_filler_6",
    "orb_icon_0",
    "orb_icon_1",
    "orb_icon_2",
    "orb_icon_3",
    "ring_30",
    "worldmap_icon_0",
    "chat_tab_button_0",
    "wornicons_0",
    "wornicons_1",
    "wornicons_2",
    "wornicons_3",
    "wornicons_4",
    "wornicons_5",
    "wornicons_6",
    "wornicons_7",
    "wornicons_8",
    "wornicons_9",
    "wornicons_10",
    "wornicons_11",
    "miscgraphics_2",
    "miscgraphics_3",
    "combatboxes_0",
    "combatboxes_1",
    "combaticons_14",
    "combaticons_15",
    "combaticons_16",
    "combaticons2_15",
    "combaticons2_16",
    "combaticons2_17",
};

#define FC_UI_ASSET_COUNT \
    ((int)(sizeof(g_ui_asset_names) / sizeof(g_ui_asset_names[0])))

typedef struct FcUiAssets {
    Texture2D textures[FC_UI_ASSET_COUNT];
    Font font;
    Font small_font;
    int missing_count;
} FcUiAssets;

static int find_asset_index(const char *name) {
    for (int i = 0; i < FC_UI_ASSET_COUNT; i++) {
        if (strcmp(g_ui_asset_names[i], name) == 0)
            return i;
    }
    return -1;
}

static Texture2D load_compass_texture(const char *path) {
    /* The OSRS client clips the rotating compass through sprite 1179 instead
     * of drawing sprite 169 as an opaque square. Pre-applying that circular
     * mask is equivalent because its shape is invariant under rotation. */
    Image compass = fc_load_image_asset(path);
    Image mask = fc_load_image_asset("data/sprites/ui/resize_compass_mask.png");
    Texture2D texture = {0};

    if (!compass.data || !mask.data) {
        if (compass.data) UnloadImage(compass);
        if (mask.data) UnloadImage(mask);
        return texture;
    }

    ImageResizeNN(&mask, compass.width, compass.height);
    ImageFormat(&compass, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    ImageFormat(&mask, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Color *compass_pixels = compass.data;
    const Color *mask_pixels = mask.data;
    int pixel_count = compass.width * compass.height;
    for (int i = 0; i < pixel_count; i++) {
        if (mask_pixels[i].a != 0)
            compass_pixels[i].a = 0;
    }

    texture = LoadTextureFromImage(compass);
    UnloadImage(mask);
    UnloadImage(compass);
    return texture;
}

static Font load_ui_font(const char *path, int size) {
    Font font = fc_load_font_asset(path, size);
    if (font.texture.id != 0) {
        SetTextureFilter(font.texture, TEXTURE_FILTER_POINT);
        fprintf(stderr, "ui_assets: loaded font %s\n", path);
    }
    return font;
}

void fc_ui_assets_load(FcUiAssets *assets) {
    memset(assets, 0, sizeof(*assets));
    for (int i = 0; i < FC_UI_ASSET_COUNT; i++) {
        const char *name = g_ui_asset_names[i];
        char path[256];
        snprintf(path, sizeof(path), "data/sprites/ui/%s.png", name);
        Texture2D tex = strcmp(name, "compass") == 0
            ? load_compass_texture(path)
            : fc_load_texture_asset(path);
        if (tex.id == 0) {
            assets->missing_count++;
            fprintf(stderr, "ui_assets: failed to load sprite %s (%s)\n", name, path);
            continue;
        }
        SetTextureFilter(tex, TEXTURE_FILTER_POINT);
        assets->textures[i] = tex;
    }
    fprintf(stderr, "ui_assets: loaded %d/%d sprites\n",
            FC_UI_ASSET_COUNT - assets->missing_count, FC_UI_ASSET_COUNT);
    assets->font = load_ui_font("data/fonts/runescape.ttf", 14);
    assets->small_font = load_ui_font("data/fonts/runescape_small.ttf", 12);
}

void fc_ui_assets_unload(FcUiAssets *assets) {
    for (int i = 0; i < FC_UI_ASSET_COUNT; i++) {
        if (assets->textures[i].id != 0)
            UnloadTexture(assets->textures[i]);
    }
    if (assets->font.texture.id != 0)
        UnloadFont(assets->font);
    if (assets->small_font.texture.id != 0)
        UnloadFont(assets->small_font);
    memset(assets, 0, sizeof(*assets));
}

const Texture2D *fc_ui_asset(const FcUiAssets *assets, const char *name) {
    int index = find_asset_index(name);
    return index >= 0 ? &assets->textures[index] : NULL;
}

void fc_ui_draw_asset(const FcUiAssets *assets, const char *name,
                      Rectangle dst, Color tint) {
    const Texture2D *tex = fc_ui_asset(assets, name);
    if (!tex)
        return;
    Rectangle src = {0, 0, (float)tex->width, (float)tex->height};
    DrawTexturePro(*tex, src, dst, (Vector2){0, 0}, 0.0f, tint);
}

Font fc_ui_font_for_size(const FcUiAssets *assets, float size) {
    return size <= 12.0f ? assets->small_font : assets->font;
}

/* Text with the client's one-pixel black drop shadow, drawn exactly at position. */
void fc_ui_draw_font_shadow(Font font, const char *text, Vector2 position,
                            float size, Color color) {
    DrawTextEx(font, text, (Vector2){position.x + 1.0f, position.y + 1.0f},
               size, 0.0f, BLACK);
    DrawTextEx(font, text, position, size, 0.0f, color);
}

/* UI text in the font for its size, snapped to whole pixels. */
void fc_ui_draw_text_shadow(const FcUiAssets *assets, const char *text,
                            float x, float y, float size, Color color) {
    if (!text || !text[0])
        return;
    if (size < 12.0f)
        size = 12.0f;
    Font font = fc_ui_font_for_size(assets, size);
    x = (float)((int)(x + 0.5f));
    y = (float)((int)(y + 0.5f));
    fc_ui_draw_font_shadow(font, text, (Vector2){x, y}, size, color);
}

/* Osrs Text */

#define FC_OSRS_MENU_FONT_SIZE 16.0f
#define FC_OSRS_MENU_FONT_ASSET "data/fonts/runescape_bold.ttf"

static Font g_menu_font;
static Font g_text_font;    /* the UI's small font, owned by FcUiAssets */

void fc_osrs_text_shutdown(void) {
    if (g_menu_font.texture.id != 0)
        UnloadFont(g_menu_font);
    g_menu_font = (Font){0};
    g_text_font = (Font){0};
}

static int init_menu_font(void) {
    g_menu_font = fc_load_font_asset(FC_OSRS_MENU_FONT_ASSET,
                                    (int)FC_OSRS_MENU_FONT_SIZE);
    if (g_menu_font.texture.id == 0) return 0;
    /* RuneC ui_assets.c:runec_make_font_texture_crisp. Rasterize at the
     * displayed size, then threshold alpha and use point sampling. */
    Image image = LoadImageFromTexture(g_menu_font.texture);
    if (!image.data) return 0;
    ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Color *pixels = image.data;
    for (int i = 0; i < image.width * image.height; i++)
        pixels[i].a = pixels[i].a >= 160 ? 255 : 0;
    Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);
    if (texture.id == 0) return 0;
    UnloadTexture(g_menu_font.texture);
    g_menu_font.texture = texture;
    SetTextureFilter(g_menu_font.texture, TEXTURE_FILTER_POINT);
    return 1;
}

Font fc_osrs_menu_font(void) {
    return g_menu_font;
}

/* Loads the menu font. Console, debug and toast text is drawn in text_font
 * at the size it was loaded at. */
int fc_osrs_text_init(Font text_font) {
    fc_osrs_text_shutdown();
    if (!init_menu_font()) {
        fprintf(stderr, "error: failed to load required menu font %s\n",
                FC_OSRS_MENU_FONT_ASSET);
        fc_osrs_text_shutdown();
        return 0;
    }
    g_text_font = text_font;
    return 1;
}

void fc_osrs_draw_text(const char* text, int x, int y, Color color) {
    DrawTextEx(g_text_font, text, (Vector2){(float)x, (float)y},
               (float)g_text_font.baseSize, 0.0f, color);
}

Vector2 fc_osrs_measure_text(const char* text) {
    return MeasureTextEx(g_text_font, text, (float)g_text_font.baseSize, 0.0f);
}

/* Minimap */

#define FC_MINIMAP_DISPLAY_SIZE 152
#define FC_MINIMAP_DISPLAY_CENTER 76.0f
#define FC_MINIMAP_DISPLAY_RADIUS 75.0f
#define FC_MINIMAP_DISPLAY_PIXELS_PER_TILE 3.5f
#define FC_MINIMAP_SCENE_SIZE 512
#define FC_MINIMAP_SCENE_PIXELS_PER_TILE 4.0f
#define FC_MINIMAP_SCENE_BORDER_TILES 32.0f

typedef struct FcMinimapScene {
    Color* pixels;
    int ready;
} FcMinimapScene;

void fc_minimap_scene_free(FcMinimapScene* scene) {
    if (!scene) return;
    free(scene->pixels);
    memset(scene, 0, sizeof(*scene));
}

int fc_minimap_scene_load_pixels(FcMinimapScene* scene,
                                 const Color* pixels, int width, int height) {
    if (!scene || !pixels || width != FC_MINIMAP_SCENE_SIZE ||
        height != FC_MINIMAP_SCENE_SIZE) {
        return 0;
    }

    size_t count = (size_t)width * (size_t)height;
    Color* copy = (Color*)malloc(count * sizeof(*copy));
    if (!copy) return 0;
    memcpy(copy, pixels, count * sizeof(*copy));

    fc_minimap_scene_free(scene);
    scene->pixels = copy;
    scene->ready = 1;
    return 1;
}

/* Rotates an offset by the camera yaw, given the yaw's sine and cosine. */
static Vector2 minimap_rotate(float dx, float dy, float sine, float cosine) {
    return (Vector2){
        dx * cosine + dy * sine,
        -dx * sine + dy * cosine,
    };
}

Vector2 fc_minimap_rotate_offset(float dx, float dy, float camera_yaw) {
    return minimap_rotate(dx, dy, sinf(camera_yaw), cosf(camera_yaw));
}

static Color sample_source(const FcMinimapScene* scene, float x, float y) {
    const Color blank = {0, 0, 0, 0};
    if (!scene || !scene->ready || !scene->pixels || x < 0.0f || y < 0.0f ||
        x >= (float)FC_MINIMAP_SCENE_SIZE ||
        y >= (float)FC_MINIMAP_SCENE_SIZE) {
        return blank;
    }

    int source_x = (int)floorf(x);
    int source_y = (int)floorf(y);
    return scene->pixels[source_x + source_y * FC_MINIMAP_SCENE_SIZE];
}

void fc_minimap_render(const FcMinimapScene* scene, float player_x,
                       float player_y, float camera_yaw, Color* output) {
    if (!output) return;
    const Color blank = {0, 0, 0, 0};
    float sine = sinf(camera_yaw);
    float cosine = cosf(camera_yaw);
    float player_source_x = 0.0f;
    float player_source_y = 0.0f;
    float source_scale =
        FC_MINIMAP_SCENE_PIXELS_PER_TILE /
        FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
    if (scene && scene->ready) {
        player_source_x =
            (player_x + FC_MINIMAP_SCENE_BORDER_TILES) *
            FC_MINIMAP_SCENE_PIXELS_PER_TILE;
        player_source_y =
            (float)FC_MINIMAP_SCENE_SIZE -
            (player_y + FC_MINIMAP_SCENE_BORDER_TILES) *
            FC_MINIMAP_SCENE_PIXELS_PER_TILE;
    }

    for (int y = 0; y < FC_MINIMAP_DISPLAY_SIZE; y++) {
        for (int x = 0; x < FC_MINIMAP_DISPLAY_SIZE; x++) {
            float screen_x = (float)x - FC_MINIMAP_DISPLAY_CENTER;
            float screen_y = (float)y - FC_MINIMAP_DISPLAY_CENTER;
            int output_index = x + y * FC_MINIMAP_DISPLAY_SIZE;
            if (screen_x * screen_x + screen_y * screen_y >
                FC_MINIMAP_DISPLAY_RADIUS * FC_MINIMAP_DISPLAY_RADIUS) {
                output[output_index] = blank;
                continue;
            }

            Vector2 offset = minimap_rotate(screen_x, screen_y, sine, cosine);
            float source_x = player_source_x + offset.x * source_scale;
            float source_y = player_source_y + offset.y * source_scale;
            output[output_index] = sample_source(scene, source_x, source_y);
        }
    }
}

int fc_minimap_click_to_tile(float map_x, float map_y, float player_x,
                             float player_y, float camera_yaw,
                             int* tile_x, int* tile_y) {
    if (!tile_x || !tile_y) return 0;
    float screen_x = map_x - FC_MINIMAP_DISPLAY_CENTER;
    float screen_y = map_y - FC_MINIMAP_DISPLAY_CENTER;
    if (screen_x * screen_x + screen_y * screen_y >
        FC_MINIMAP_DISPLAY_RADIUS * FC_MINIMAP_DISPLAY_RADIUS) {
        return 0;
    }

    /* Scene raster rows grow south, tile y grows north. */
    Vector2 offset = fc_minimap_rotate_offset(screen_x, screen_y, camera_yaw);
    float dx = offset.x / FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
    float dy = -offset.y / FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
    int x = (int)floorf(player_x + dx);
    int y = (int)floorf(player_y + dy);
    if (x < 0 || y < 0 || x >= FC_ARENA_WIDTH || y >= FC_ARENA_HEIGHT)
        return 0;
    *tile_x = x;
    *tile_y = y;
    return 1;
}

/* Context Menu */

/* RuneC's client-style geometry, shared by menu drawing and hit testing. */
#define FC_MENU_HEADER_HEIGHT 24
#define FC_MENU_ROW_HEIGHT 19
#define FC_MENU_DISMISS_MARGIN 10

typedef struct { int x, y, width, height; } FcMenuLayout;
typedef struct { const char *name; int level; } FcNpcMenuInfo;

FcNpcMenuInfo fc_menu_npc_info(int npc_type) {
    /* Fight Cave NPC display levels, not attack-skill levels or Inferno healers. */
    static const FcNpcMenuInfo npcs[] = {
        {"", 0}, {"Tz-Kih", 22}, {"Tz-Kek", 45}, {"Tz-Kek", 22},
        {"Tok-Xil", 90}, {"Yt-MejKot", 180}, {"Ket-Zek", 360},
        {"TzTok-Jad", 702}, {"Yt-HurKot", 108}
    };
    if (npc_type < 0 || npc_type >= (int)(sizeof(npcs) / sizeof(npcs[0]))) npc_type = 0;
    return npcs[npc_type];
}

uint32_t fc_menu_level_color(int player_level, int npc_level) {
    /* Client3 entry/client.c:getCombatLevelColorTag and defines.h RGB ramps. */
    int difference = player_level - npc_level;
    if (difference < -9) return 0xff0000;
    if (difference < -6) return 0xff3000;
    if (difference < -3) return 0xff7000;
    if (difference < 0) return 0xffb000;
    if (difference > 9) return 0x00ff00;
    if (difference > 6) return 0x40ff00;
    if (difference > 3) return 0x80ff00;
    if (difference > 0) return 0xc0ff00;
    return 0xffff00;
}

FcMenuLayout fc_menu_layout(int x, int y, int screen_width, int screen_height,
                             int text_width, int rows) {
    if (rows < 0) rows = 0;
    FcMenuLayout menu = {0, 0, text_width + 12,
                         FC_MENU_HEADER_HEIGHT + rows * FC_MENU_ROW_HEIGHT + 4};
    if (menu.width < 1) menu.width = 1;
    if (screen_width > 0 && menu.width > screen_width) menu.width = screen_width;
    menu.x = x - menu.width / 2;
    menu.y = y;
    if (menu.x + menu.width > screen_width) menu.x = screen_width - menu.width;
    if (menu.y + menu.height > screen_height) menu.y = screen_height - menu.height;
    if (menu.x < 0) menu.x = 0;
    if (menu.y < 0) menu.y = 0;
    return menu;
}

int fc_menu_contains(FcMenuLayout menu, int x, int y, int margin) {
    return x >= menu.x - margin && x < menu.x + menu.width + margin &&
           y >= menu.y - margin && y < menu.y + menu.height + margin;
}

int fc_menu_action_at(FcMenuLayout menu, int rows, int x, int y) {
    if (!fc_menu_contains(menu, x, y, 0)) return -1;
    int offset = y - menu.y - FC_MENU_HEADER_HEIGHT;
    if (offset < 0) return -1;
    int row = offset / FC_MENU_ROW_HEIGHT;
    return row < rows ? row : -1;
}

/* Ui */

#define FC_UI_CHAT_INPUT_MAX 128
#define FC_UI_CONTEXT_ACTIONS 5
#define FC_UI_MINIMAP_DOTS 256
#define FC_UI_ITEM_ICON_CACHE 512
#define FC_UI_COMBAT_STYLE_COUNT 3

typedef enum FcUiTab {
    FC_UI_TAB_NONE = -1,
    FC_UI_TAB_COMBAT = 0,
    FC_UI_TAB_INVENTORY,
    FC_UI_TAB_EQUIPMENT,
    FC_UI_TAB_PRAYER,
    FC_UI_TAB_COUNT
} FcUiTab;

typedef enum FcUiIntentKind {
    FC_UI_INTENT_NONE = 0,
    FC_UI_INTENT_TAB,
    FC_UI_INTENT_INVENTORY_SLOT,
    FC_UI_INTENT_EQUIPMENT_SLOT,
    FC_UI_INTENT_PRAYER_SLOT,
    FC_UI_INTENT_QUICK_PRAYER_TOGGLE,
    FC_UI_INTENT_MINIMAP_CLICK,
    FC_UI_INTENT_RUN_TOGGLE,
    FC_UI_INTENT_WORLD_ACTION,
    FC_UI_INTENT_INVENTORY_ACTION,
    FC_UI_INTENT_EQUIPMENT_ACTION,
    FC_UI_INTENT_INVENTORY_DRAG
} FcUiIntentKind;

typedef struct FcUiIntent {
    FcUiIntentKind kind;
    int primary;
    int secondary;
    Vector2 position;
    char text[FC_UI_CHAT_INPUT_MAX];
} FcUiIntent;

typedef struct FcUiSlot {
    uint32_t item_id;
    int quantity;
    char label[24];
    int enabled;
    const char *action;           /* supplied by the inventory owner */
} FcUiSlot;

typedef enum FcUiMinimapDotKind {
    FC_UI_MINIMAP_DOT_NPC = 0,
    FC_UI_MINIMAP_DOT_PLAYER,
    FC_UI_MINIMAP_DOT_DESTINATION
} FcUiMinimapDotKind;

typedef struct FcUiMinimapDot {
    float dx;
    float dy;
    FcUiMinimapDotKind kind;
} FcUiMinimapDot;

typedef struct FcUiItemIcon {
    uint32_t item_id;
    Texture2D texture;
} FcUiItemIcon;

typedef struct FcUiCombatStyleDef {
    const char *label;
    const char *mode;
    const char *icon_asset;
} FcUiCombatStyleDef;

/* A weapon category's combat styles. The tab only shows them: the active
 * style is the one the simulation attacks with. */
typedef struct FcUiCombatProfile {
    int weapon_category;
    int active_style;
    FcUiCombatStyleDef styles[FC_UI_COMBAT_STYLE_COUNT];
} FcUiCombatProfile;

typedef enum FcUiContextSourceKind {
    FC_UI_CONTEXT_NONE = 0,
    FC_UI_CONTEXT_INVENTORY,
    FC_UI_CONTEXT_EQUIPMENT,
    FC_UI_CONTEXT_PRAYER,
    FC_UI_CONTEXT_WORLD
} FcUiContextSourceKind;

typedef struct FcUiDragState {
    int active;
    FcUiContextSourceKind source_kind;
    int source_slot;
    Vector2 start;
} FcUiDragState;

typedef struct FcUiState {
    FcUiTab active_tab;
    FcUiIntent last_intent;
    float tab_press_timer[FC_UI_TAB_COUNT];

    FcUiSlot inventory[FC_INVENTORY_SLOTS];
    FcUiSlot equipment[FC_EQUIPMENT_SLOTS];
    char combat_weapon_name[64];
    const FcUiCombatProfile *combat_profile;

    int hitpoints;
    int hitpoints_max;
    int prayer_points;
    int prayer_points_max;
    int run_energy;
    int run_enabled;
    int combat_level;

    int context_open;
    Vector2 context_pos;
    Color context_target_color;
    int context_combat_level;
    char context_title[48];
    char context_actions[FC_UI_CONTEXT_ACTIONS][32];
    int context_action_count;
    FcUiContextSourceKind context_source_kind;
    int context_source_slot;
    uint32_t context_source_item_id;

    FcUiDragState drag;

    FcUiMinimapDot minimap_dots[FC_UI_MINIMAP_DOTS];
    int minimap_dot_count;
    float minimap_rotation;
    Texture2D minimap_texture;
    FcUiItemIcon item_icons[FC_UI_ITEM_ICON_CACHE];
    int item_icon_count;

    FcUiAssets assets;
} FcUiState;

#define OSRS_ORANGE ((Color){255, 152, 31, 255})
#define OSRS_YELLOW ((Color){255, 255, 0, 255})
#define OSRS_GREEN  ((Color){0, 255, 0, 255})
#define OSRS_RED    ((Color){255, 40, 25, 255})
#define OSRS_TAB_PRESS_SECONDS 0.12f

typedef struct FcUiLayout {
    Rectangle chat;
    Rectangle chat_messages;
    Rectangle chat_controls;
    Rectangle map;
    Rectangle minimap;
    Rectangle compass;
    Rectangle worldmap_button;
    Rectangle hp_orb;
    Rectangle prayer_orb;
    Rectangle run_orb;
    Rectangle side;
    Rectangle side_content;
    Rectangle tab[FC_UI_TAB_COUNT];
} FcUiLayout;

static const Rectangle g_equipment_offsets[FC_EQUIPMENT_SLOTS] = {
    {77, 4, 36, 36},
    {36, 43, 36, 36},
    {77, 43, 36, 36},
    {21, 82, 36, 36},
    {77, 82, 36, 36},
    {133, 82, 36, 36},
    {-1000, -1000, 0, 0},
    {77, 122, 36, 36},
    {-1000, -1000, 0, 0},
    {21, 162, 36, 36},
    {77, 162, 36, 36},
    {-1000, -1000, 0, 0},
    {133, 162, 36, 36},
    {133, 43, 36, 36},
};

static const char *g_worn_icon_names[FC_EQUIPMENT_SLOTS] = {
    "wornicons_0", "wornicons_1", "wornicons_2", "wornicons_3",
    "wornicons_4", "wornicons_5", "wornicons_10", "wornicons_6",
    NULL, "wornicons_7", "wornicons_8", NULL, "wornicons_9", "wornicons_11",
};

/*
 * Exact b237 combat_interface_weapon_category DB rows from the local
 * Joshua-F dump for the weapon categories the player can have
 * (FcLoadout.combat_style_profile, or 0 when unarmed), so weapon tabs use the
 * same labels/icons as OSRS. Core has no style choice: it punches unarmed and
 * always fires the bow on Rapid.
 */
static const FcUiCombatProfile g_combat_profiles[] = {
    {0, 0, {    /* unarmed: OSRS category 0 */
        {"Punch", "Accurate", "combaticons_14"},
        {"Kick", "Aggressive", "combaticons_15"},
        {"Block", "Defensive", "combaticons_16"},
    }},
    {25, 1, {   /* bow: OSRS category 3 */
        {"Accurate", "Accurate", "combaticons2_15"},
        {"Rapid", "Rapid", "combaticons2_16"},
        {"Longrange", "Longrange", "combaticons2_17"},
    }},
};

static void copy_text(char *dst, size_t cap, const char *src) {
    if (cap == 0)
        return;
    if (!src)
        src = "";
    snprintf(dst, cap, "%s", src);
}

static const FcUiCombatProfile *combat_profile_for_weapon_category(int category) {
    int count = (int)(sizeof(g_combat_profiles) / sizeof(g_combat_profiles[0]));
    for (int i = 0; i < count; i++) {
        if (g_combat_profiles[i].weapon_category == category)
            return &g_combat_profiles[i];
    }
    return &g_combat_profiles[0];
}

void fc_ui_set_combat_weapon_name(FcUiState *ui, const char *name) {
    if (!ui)
        return;
    copy_text(ui->combat_weapon_name, sizeof(ui->combat_weapon_name),
              name && name[0] ? name : "Unarmed");
}

static float fmin2(float a, float b) {
    return a < b ? a : b;
}

static void ui_layout(int screen_w, int screen_h, FcUiLayout *out) {
    memset(out, 0, sizeof(*out));

    out->chat = (Rectangle){0, (float)screen_h - FC_OSRS_CHAT_H,
                            FC_OSRS_CHAT_W, FC_OSRS_CHAT_H};
    out->chat_messages = (Rectangle){7, out->chat.y + 7, 506, 126};
    out->chat_controls = (Rectangle){0, out->chat.y + 142, 519, 23};

    out->map = (Rectangle){(float)screen_w - FC_OSRS_MAP_CONTAINER_W, 0,
                           FC_OSRS_MAP_CONTAINER_W, FC_OSRS_MAP_CONTAINER_H};
    out->minimap = (Rectangle){out->map.x + FC_OSRS_MINIMAP_X,
                               out->map.y + FC_OSRS_MINIMAP_Y,
                               FC_OSRS_MINIMAP_W, FC_OSRS_MINIMAP_H};
    out->compass = (Rectangle){out->map.x + FC_OSRS_COMPASS_X,
                               out->map.y + FC_OSRS_COMPASS_Y,
                               FC_OSRS_COMPASS_W, FC_OSRS_COMPASS_H};
    out->hp_orb = (Rectangle){out->map.x + FC_OSRS_ORBS_X + FC_OSRS_HP_X,
                              out->map.y + FC_OSRS_ORBS_Y + FC_OSRS_HP_Y, 57, 34};
    out->prayer_orb = (Rectangle){out->map.x + FC_OSRS_ORBS_X + FC_OSRS_PRAYER_X,
                                  out->map.y + FC_OSRS_ORBS_Y + FC_OSRS_PRAYER_Y, 57, 34};
    out->run_orb = (Rectangle){out->map.x + FC_OSRS_ORBS_X + FC_OSRS_RUN_X,
                               out->map.y + FC_OSRS_ORBS_Y + FC_OSRS_RUN_Y, 57, 34};
    out->worldmap_button = (Rectangle){out->map.x + FC_OSRS_ORBS_X + FC_OSRS_WORLDMAP_X,
                                       out->map.y + FC_OSRS_ORBS_Y + FC_OSRS_WORLDMAP_Y,
                                       30, 30};

    out->side = (Rectangle){(float)screen_w - FC_OSRS_SIDE_MENU_W,
                            (float)screen_h - FC_OSRS_SIDE_MENU_H,
                            FC_OSRS_SIDE_MENU_W, FC_OSRS_SIDE_MENU_H};
    out->side_content = (Rectangle){out->side.x + FC_OSRS_SIDE_CONTENT_X,
                                    out->side.y + FC_OSRS_SIDE_CONTENT_Y,
                                    FC_OSRS_SIDE_CONTENT_W,
                                    FC_OSRS_SIDE_CONTENT_H};

    for (int i = 0; i < (int)(sizeof(FC_OSRS_SIDE_STONES) / sizeof(FC_OSRS_SIDE_STONES[0])); i++) {
        const FcUiStoneRef *ref = &FC_OSRS_SIDE_STONES[i];
        if (ref->logical_tab < 0 || ref->logical_tab >= FC_UI_TAB_COUNT)
            continue;
        float row_y = out->side.y + (i < 7 ? FC_OSRS_SIDE_TOP_Y : FC_OSRS_SIDE_BOTTOM_Y);
        out->tab[ref->logical_tab] =
            (Rectangle){out->side.x + ref->rect.x, row_y + ref->rect.y,
                        ref->rect.width, ref->rect.height};
    }
}

static int mouse_over_ui(const FcUiLayout *layout, Vector2 mouse) {
    return CheckCollisionPointRec(mouse, layout->chat)
        || CheckCollisionPointRec(mouse, layout->side)
        || CheckCollisionPointRec(mouse, layout->map);
}

static void clear_intent(FcUiState *ui) {
    memset(&ui->last_intent, 0, sizeof(ui->last_intent));
}

/* Records this frame's UI action; any text is set separately. */
static void set_intent(FcUiState *ui, FcUiIntentKind kind,
                       int primary, int secondary, Vector2 position) {
    ui->last_intent.kind = kind;
    ui->last_intent.primary = primary;
    ui->last_intent.secondary = secondary;
    ui->last_intent.position = position;
}

static void set_context(FcUiState *ui, Vector2 pos, const char *title,
                        const char **actions, int action_count) {
    ui->context_open = 1;
    ui->context_pos = pos;
    ui->context_target_color = OSRS_ORANGE;
    ui->context_combat_level = 0;
    ui->drag.active = 0;
    copy_text(ui->context_title, sizeof(ui->context_title), title);
    ui->context_source_kind = FC_UI_CONTEXT_NONE;
    ui->context_source_slot = -1;
    ui->context_source_item_id = 0;
    if (action_count > FC_UI_CONTEXT_ACTIONS)
        action_count = FC_UI_CONTEXT_ACTIONS;
    if (action_count < 0) action_count = 0;
    ui->context_action_count = action_count;
    for (int i = 0; i < action_count; i++) {
        copy_text(ui->context_actions[i], sizeof(ui->context_actions[i]), actions[i]);
    }
}

static void set_context_source(FcUiState *ui,
                               FcUiContextSourceKind source_kind,
                               int source_slot,
                               uint32_t source_item_id) {
    ui->context_source_kind = source_kind;
    ui->context_source_slot = source_slot;
    ui->context_source_item_id = source_item_id;
}

void fc_ui_close_context(FcUiState *ui) {
    ui->context_open = 0;
    ui->context_source_kind = FC_UI_CONTEXT_NONE;
    ui->context_source_slot = -1;
    ui->context_source_item_id = 0;
}

void fc_ui_open_world_context(FcUiState *ui, Vector2 pos, int npc_type, int can_walk) {
    FcNpcMenuInfo npc = fc_menu_npc_info(npc_type);
    const char *actions[4];
    int count = 0;
    if (npc.level) actions[count++] = "Attack";
    if (can_walk) actions[count++] = "Walk here";
    if (npc.level) actions[count++] = "Examine";
    actions[count++] = "Cancel";
    set_context(ui, pos, npc.name, actions, count);
    ui->context_combat_level = npc.level;
    ui->context_source_kind = FC_UI_CONTEXT_WORLD;
    ui->context_target_color = OSRS_YELLOW;
}

/* Menu for one of the viewer's prayer buttons; picking the verb reports
 * FC_UI_INTENT_PRAYER_SLOT with this slot. */
void fc_ui_open_prayer_context(FcUiState *ui, Vector2 pos, int slot,
                               const char *name, int active) {
    const char *actions[] = {active ? "Deactivate" : "Activate", "Cancel"};
    set_context(ui, pos, name, actions, 2);
    set_context_source(ui, FC_UI_CONTEXT_PRAYER, slot, 0);
}

static int context_has_target(const FcUiState *ui, const char *action) {
    return ui->context_title[0] && strcmp(action, "Cancel") && strcmp(action, "Walk here");
}

static void context_level_suffix(const FcUiState *ui, char suffix[32]) {
    suffix[0] = '\0';
    if (ui->context_combat_level > 0)
        snprintf(suffix, 32, " (level-%d)", ui->context_combat_level);
}

static FcMenuLayout context_layout(const FcUiState *ui) {
    Font font = fc_osrs_menu_font();
    float width = MeasureTextEx(font, "Choose Option", FC_OSRS_MENU_FONT_SIZE, 0).x;
    char suffix[32];
    context_level_suffix(ui, suffix);
    for (int i = 0; i < ui->context_action_count; i++) {
        char text[128];
        const char *action = ui->context_actions[i];
        snprintf(text, sizeof(text), "%s%s%s%s", action,
                 context_has_target(ui, action) ? " " : "",
                 context_has_target(ui, action) ? ui->context_title : "",
                 context_has_target(ui, action) ? suffix : "");
        float row_width = MeasureTextEx(font, text, FC_OSRS_MENU_FONT_SIZE, 0).x;
        if (row_width > width) width = row_width;
    }
    return fc_menu_layout((int)ui->context_pos.x, (int)ui->context_pos.y,
        GetScreenWidth(), GetScreenHeight(), (int)ceilf(width), ui->context_action_count);
}

void fc_ui_init(FcUiState *ui) {
    memset(ui, 0, sizeof(*ui));
    ui->active_tab = FC_UI_TAB_INVENTORY;
    ui->combat_profile = &g_combat_profiles[0];
    ui->context_source_slot = -1;
    ui->drag.source_slot = -1;

    fc_ui_assets_load(&ui->assets);
    Image minimap = GenImageColor(FC_MINIMAP_DISPLAY_SIZE, FC_MINIMAP_DISPLAY_SIZE, BLANK);
    ui->minimap_texture = LoadTextureFromImage(minimap);
    UnloadImage(minimap);
    if (ui->minimap_texture.id != 0)
        SetTextureFilter(ui->minimap_texture, TEXTURE_FILTER_POINT);
}

void fc_ui_shutdown(FcUiState *ui) {
    if (ui->minimap_texture.id != 0)
        UnloadTexture(ui->minimap_texture);
    for (int i = 0; i < ui->item_icon_count; i++)
        UnloadTexture(ui->item_icons[i].texture);
    ui->item_icon_count = 0;
    fc_ui_assets_unload(&ui->assets);
}

void fc_ui_add_minimap_dot(FcUiState *ui, float dx, float dy,
                           FcUiMinimapDotKind kind) {
    if (ui->minimap_dot_count >= FC_UI_MINIMAP_DOTS)
        return;
    ui->minimap_dots[ui->minimap_dot_count++] =
        (FcUiMinimapDot){dx, dy, kind};
}

void fc_ui_set_item_icon(FcUiState *ui, uint32_t item_id, Texture2D texture) {
    if (!ui || item_id == 0 || texture.id == 0)
        return;
    for (int i = 0; i < ui->item_icon_count; i++) {
        if (ui->item_icons[i].item_id == item_id) {
            UnloadTexture(ui->item_icons[i].texture);
            ui->item_icons[i].texture = texture;
            return;
        }
    }
    if (ui->item_icon_count >= FC_UI_ITEM_ICON_CACHE) {
        UnloadTexture(texture);
        return;
    }
    ui->item_icons[ui->item_icon_count++] = (FcUiItemIcon){item_id, texture};
}

static int handle_context_click(FcUiState *ui, Vector2 mouse) {
    if (!ui->context_open)
        return 0;

    FcMenuLayout menu = context_layout(ui);
    if (!fc_menu_contains(menu, (int)mouse.x, (int)mouse.y, 0)) {
        fc_ui_close_context(ui);
        return 1; /* Dismissal must not click through into the scene/UI. */
    }

    for (int i = 0; i < ui->context_action_count; i++) {
        if (fc_menu_action_at(menu, ui->context_action_count, (int)mouse.x, (int)mouse.y) == i) {
            const char *action = ui->context_actions[i];
            if (strcmp(action, "Cancel") == 0) {
                fc_ui_close_context(ui);
                return 1;
            }
            /* Do not act on a different item if the original slot changed
             * while its menu was open (consumption, ammo depletion, etc.). */
            if ((ui->context_source_kind == FC_UI_CONTEXT_INVENTORY &&
                 ui->inventory[ui->context_source_slot].item_id != ui->context_source_item_id) ||
                (ui->context_source_kind == FC_UI_CONTEXT_EQUIPMENT &&
                 ui->equipment[ui->context_source_slot].item_id != ui->context_source_item_id)) {
                fc_ui_close_context(ui);
                return 1;
            }
            int slot = ui->context_source_slot;
            if (ui->context_source_kind == FC_UI_CONTEXT_WORLD) {
                set_intent(ui, FC_UI_INTENT_WORLD_ACTION, i, 0, mouse);
            } else if (ui->context_source_kind == FC_UI_CONTEXT_INVENTORY) {
                set_intent(ui, FC_UI_INTENT_INVENTORY_ACTION, slot, i, mouse);
            } else if (ui->context_source_kind == FC_UI_CONTEXT_EQUIPMENT) {
                set_intent(ui, FC_UI_INTENT_EQUIPMENT_ACTION, slot, i, mouse);
            } else if (ui->context_source_kind == FC_UI_CONTEXT_PRAYER) {
                /* Prayer menus offer Activate or Deactivate, then Cancel. */
                set_intent(ui, FC_UI_INTENT_PRAYER_SLOT, slot,
                           strcmp(action, "Deactivate") == 0 ? -1 : 1, mouse);
                copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
                          ui->context_title);
            }
            if (!ui->last_intent.text[0])
                copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
                          action);
            fc_ui_close_context(ui);
            return 1;
        }
    }

    return 1;
}

static Rectangle side_ref_rect(const FcUiLayout *layout, Rectangle ref) {
    return (Rectangle){layout->side_content.x + ref.x, layout->side_content.y + ref.y,
                       ref.width, ref.height};
}

/* A grid of equal cells in the side panel content area; first is the first
 * cell relative to that area. */
typedef struct FcUiGrid {
    int count;
    int cols;
    Rectangle first;
    float step_x;
    float step_y;
} FcUiGrid;

static const FcUiGrid g_inventory_grid = {
    FC_INVENTORY_SLOTS, 4,
    {FC_OSRS_INVENTORY_SLOT_X, FC_OSRS_INVENTORY_SLOT_Y,
     FC_OSRS_INVENTORY_SLOT_W, FC_OSRS_INVENTORY_SLOT_H},
    FC_OSRS_INVENTORY_SLOT_STEP_X, FC_OSRS_INVENTORY_SLOT_STEP_Y,
};

static Rectangle grid_cell_rect(const FcUiLayout *layout,
                                const FcUiGrid *grid, int index) {
    int col = index % grid->cols;
    int row = index / grid->cols;
    return (Rectangle){
        layout->side_content.x + grid->first.x + col * grid->step_x,
        layout->side_content.y + grid->first.y + row * grid->step_y,
        grid->first.width,
        grid->first.height,
    };
}

static int grid_index_at(const FcUiLayout *layout, const FcUiGrid *grid,
                         Vector2 mouse) {
    for (int i = 0; i < grid->count; i++) {
        if (CheckCollisionPointRec(mouse, grid_cell_rect(layout, grid, i)))
            return i;
    }
    return -1;
}

static Rectangle equip_slot_rect(const FcUiLayout *layout, int slot) {
    if (slot < 0 || slot >= FC_EQUIPMENT_SLOTS)
        return (Rectangle){0, 0, 0, 0};
    Rectangle off = g_equipment_offsets[slot];
    if (off.width <= 0 || off.height <= 0)
        return off;
    return side_ref_rect(layout, off);
}

static int equipment_slot_at(const FcUiLayout *layout, Vector2 mouse) {
    for (int i = 0; i < FC_EQUIPMENT_SLOTS; i++) {
        Rectangle r = equip_slot_rect(layout, i);
        if (r.width > 0 && CheckCollisionPointRec(mouse, r))
            return i;
    }
    return -1;
}

static void update_tab_press_timers(FcUiState *ui, float dt) {
    for (int i = 0; i < FC_UI_TAB_COUNT; i++) {
        if (ui->tab_press_timer[i] <= 0.0f)
            continue;
        ui->tab_press_timer[i] -= dt;
        if (ui->tab_press_timer[i] < 0.0f)
            ui->tab_press_timer[i] = 0.0f;
    }
}

static int handle_drag_release(FcUiState *ui,
                               const FcUiLayout *layout,
                               Vector2 mouse) {
    if (!ui->drag.active || !IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
        return 0;

    FcUiDragState drag = ui->drag;
    ui->drag.active = 0;
    ui->drag.source_kind = FC_UI_CONTEXT_NONE;
    ui->drag.source_slot = -1;

    if (drag.source_kind == FC_UI_CONTEXT_INVENTORY) {
        int target = grid_index_at(layout, &g_inventory_grid, mouse);
        float dx = mouse.x - drag.start.x;
        float dy = mouse.y - drag.start.y;
        int moved = fabsf(dx) > 3.0f || fabsf(dy) > 3.0f;
        if (target >= 0 && moved)
            set_intent(ui, FC_UI_INTENT_INVENTORY_DRAG, drag.source_slot, target, mouse);
        else
            set_intent(ui, FC_UI_INTENT_INVENTORY_SLOT, drag.source_slot, 0, mouse);
        return 1;
    }

    if (drag.source_kind == FC_UI_CONTEXT_EQUIPMENT) {
        set_intent(ui, FC_UI_INTENT_EQUIPMENT_SLOT, drag.source_slot, 0, mouse);
        return 1;
    }
    return 0;
}

static int handle_tab_click(FcUiState *ui,
                            const FcUiLayout *layout,
                            Vector2 mouse) {
    for (int i = 0; i < FC_UI_TAB_COUNT; i++) {
        if (!CheckCollisionPointRec(mouse, layout->tab[i]))
            continue;
        ui->active_tab = (FcUiTab)i;
        ui->tab_press_timer[i] = OSRS_TAB_PRESS_SECONDS;
        set_intent(ui, FC_UI_INTENT_TAB, i, 0, mouse);
        return 1;
    }
    return 0;
}

static int handle_orb_or_minimap_click(FcUiState *ui,
                                       const FcUiLayout *layout,
                                       Vector2 mouse) {
    Rectangle run_button = {
        layout->run_orb.x + 3,
        layout->run_orb.y + 5,
        50,
        26,
    };
    if (CheckCollisionPointRec(mouse, run_button)) {
        set_intent(ui, FC_UI_INTENT_RUN_TOGGLE, 0, 0, mouse);
        return 1;
    }
    if (CheckCollisionPointRec(mouse, layout->prayer_orb)) {
        set_intent(ui, FC_UI_INTENT_QUICK_PRAYER_TOGGLE, 1, 0, mouse);
        copy_text(ui->last_intent.text, sizeof(ui->last_intent.text),
                  "Quick-prayer");
        return 1;
    }

    Vector2 center = {
        layout->minimap.x + layout->minimap.width * 0.5f,
        layout->minimap.y + layout->minimap.height * 0.5f,
    };
    float dx = mouse.x - center.x;
    float dy = mouse.y - center.y;
    if (!CheckCollisionPointRec(mouse, layout->minimap) ||
        dx * dx + dy * dy > FC_MINIMAP_DISPLAY_RADIUS * FC_MINIMAP_DISPLAY_RADIUS)
        return 0;
    set_intent(ui, FC_UI_INTENT_MINIMAP_CLICK,
               (int)(mouse.x - layout->minimap.x),
               (int)(mouse.y - layout->minimap.y), mouse);
    return 1;
}

static int handle_inventory_click(FcUiState *ui,
                                  const FcUiLayout *layout,
                                  Vector2 mouse) {
    int slot = grid_index_at(layout, &g_inventory_grid, mouse);
    if (slot < 0)
        return 0;
    ui->drag.active = 1;
    ui->drag.source_kind = FC_UI_CONTEXT_INVENTORY;
    ui->drag.source_slot = slot;
    ui->drag.start = mouse;
    return 1;
}

static int handle_equipment_click(FcUiState *ui,
                                  const FcUiLayout *layout,
                                  Vector2 mouse) {
    int slot = equipment_slot_at(layout, mouse);
    if (slot < 0)
        return 0;
    ui->drag.active = 1;
    ui->drag.source_kind = FC_UI_CONTEXT_EQUIPMENT;
    ui->drag.source_slot = slot;
    ui->drag.start = mouse;
    return 1;
}

static int handle_active_tab_click(FcUiState *ui,
                                   const FcUiLayout *layout,
                                   Vector2 mouse) {
    switch (ui->active_tab) {
    case FC_UI_TAB_INVENTORY:
        return handle_inventory_click(ui, layout, mouse);
    case FC_UI_TAB_EQUIPMENT:
        return handle_equipment_click(ui, layout, mouse);
    default:
        return 0;
    }
}

static int handle_primary_click(FcUiState *ui,
                                const FcUiLayout *layout,
                                Vector2 mouse) {
    if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        return 0;
    if (handle_tab_click(ui, layout, mouse) ||
        handle_orb_or_minimap_click(ui, layout, mouse) ||
        handle_active_tab_click(ui, layout, mouse))
        return 1;
    return 0;
}

static int handle_context_menu_open(FcUiState *ui,
                                    const FcUiLayout *layout,
                                    Vector2 mouse) {
    if (!IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) ||
        !mouse_over_ui(layout, mouse))
        return 0;

    if (ui->active_tab == FC_UI_TAB_INVENTORY) {
        int slot = grid_index_at(layout, &g_inventory_grid, mouse);
        if (slot >= 0) {
            const char *actions[] = {ui->inventory[slot].action ?
                ui->inventory[slot].action : "Use", "Examine", "Cancel"};
            static const char *empty_actions[] = {"Cancel"};
            const char *title = ui->inventory[slot].enabled
                ? ui->inventory[slot].label : "Empty inventory slot";
            if (ui->inventory[slot].enabled) {
                set_context(ui, mouse, title, actions, 3);
                set_context_source(ui, FC_UI_CONTEXT_INVENTORY, slot,
                                   ui->inventory[slot].item_id);
            } else {
                set_context(ui, mouse, title, empty_actions, 1);
            }
            return 1;
        }
    }
    if (ui->active_tab == FC_UI_TAB_EQUIPMENT) {
        int slot = equipment_slot_at(layout, mouse);
        if (slot >= 0) {
            static const char *actions[] = {"Remove", "Examine", "Cancel"};
            static const char *empty_actions[] = {"Cancel"};
            if (ui->equipment[slot].enabled) {
                set_context(ui, mouse, ui->equipment[slot].label, actions, 3);
                set_context_source(ui, FC_UI_CONTEXT_EQUIPMENT, slot,
                                   ui->equipment[slot].item_id);
            } else {
                set_context(ui, mouse, "", empty_actions, 1);
            }
            return 1;
        }
    }

    static const char *actions[] = {"Cancel"};
    set_context(ui, mouse, "", actions, 1);
    return 1;
}

int fc_ui_handle_input(FcUiState *ui, int screen_w, int screen_h) {
    FcUiLayout layout;
    ui_layout(screen_w, screen_h, &layout);
    Vector2 mouse = GetMousePosition();
    clear_intent(ui);
    update_tab_press_timers(ui, GetFrameTime());

    if (ui->context_open) {
        if (IsKeyPressed(KEY_ESCAPE)) {
            fc_ui_close_context(ui);
        } else if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            handle_context_click(ui, mouse);
        } else if (!fc_menu_contains(context_layout(ui), (int)mouse.x, (int)mouse.y,
                                      FC_MENU_DISMISS_MARGIN)) {
            fc_ui_close_context(ui);
        }
        return 1;
    }

    if (handle_drag_release(ui, &layout, mouse) ||
        handle_primary_click(ui, &layout, mouse) ||
        handle_context_menu_open(ui, &layout, mouse))
        return 1;

    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && !ui->context_open)
        return 0;
    return mouse_over_ui(&layout, mouse);
}

static void draw_text_shadow(const FcUiState *ui, const char *text,
                             float x, float y, float size, Color color) {
    fc_ui_draw_text_shadow(&ui->assets, text, x, y, size, color);
}

static void draw_centered_text(const FcUiState *ui, const char *text,
                               Rectangle rect, float size, Color color) {
    if (size < 12.0f)
        size = 12.0f;
    Font font = fc_ui_font_for_size(&ui->assets, size);
    Vector2 m = MeasureTextEx(font, text, size, 0);
    draw_text_shadow(ui, text, rect.x + (rect.width - m.x) * 0.5f,
                     rect.y + (rect.height - m.y) * 0.5f, size, color);
}

static void draw_asset_centered(const FcUiState *ui, const char *name,
                                Rectangle rect, float max_w, float max_h, Color tint) {
    const Texture2D *tex = fc_ui_asset(&ui->assets, name);
    if (!tex)
        return;
    float scale = fmin2(max_w / (float)tex->width, max_h / (float)tex->height);
    if (scale > 1.0f)
        scale = 1.0f;
    Rectangle dst = {
        rect.x + (rect.width - tex->width * scale) * 0.5f,
        rect.y + (rect.height - tex->height * scale) * 0.5f,
        tex->width * scale,
        tex->height * scale,
    };
    fc_ui_draw_asset(&ui->assets, name, dst, tint);
}

static void draw_asset_tiled(const FcUiState *ui, const char *name,
                             Rectangle dst, Color tint) {
    const Texture2D *tex = fc_ui_asset(&ui->assets, name);
    if (!tex)
        return;
    for (float y = dst.y; y < dst.y + dst.height; y += (float)tex->height) {
        for (float x = dst.x; x < dst.x + dst.width; x += (float)tex->width) {
            float w = fmin2((float)tex->width, dst.x + dst.width - x);
            float h = fmin2((float)tex->height, dst.y + dst.height - y);
            DrawTexturePro(*tex, (Rectangle){0, 0, w, h},
                           (Rectangle){x, y, w, h}, (Vector2){0, 0}, 0, tint);
        }
    }
}

static void draw_side_chrome(const FcUiState *ui, const FcUiLayout *layout) {
    Rectangle backing = {layout->side.x + 20, layout->side.y + 27, 200, 281};
    draw_asset_tiled(ui, "tradebacking_dark", backing, WHITE);

    fc_ui_draw_asset(&ui->assets, "osrs_stretch_side_topbottom_0",
                     (Rectangle){layout->side.x, layout->side.y + FC_OSRS_SIDE_TOP_Y,
                                 241, 37}, WHITE);
    fc_ui_draw_asset(&ui->assets, "osrs_stretch_side_topbottom_1",
                     (Rectangle){layout->side.x, layout->side.y + FC_OSRS_SIDE_BOTTOM_Y,
                                 241, 37}, WHITE);
    fc_ui_draw_asset(&ui->assets, "osrs_stretch_side_columns_0",
                     (Rectangle){layout->side.x + 2, layout->side.y + 37, 26, 261}, WHITE);
    fc_ui_draw_asset(&ui->assets, "osrs_stretch_side_columns_1",
                     (Rectangle){layout->side.x + 212, layout->side.y + 37, 26, 261}, WHITE);

    for (int i = 0; i < (int)(sizeof(FC_OSRS_SIDE_STONES) / sizeof(FC_OSRS_SIDE_STONES[0])); i++) {
        const FcUiStoneRef *ref = &FC_OSRS_SIDE_STONES[i];
        if (ref->logical_tab != ui->active_tab)
            continue;
        float row_y = layout->side.y + (i < 7 ? FC_OSRS_SIDE_TOP_Y : FC_OSRS_SIDE_BOTTOM_Y);
        float pressed = ui->tab_press_timer[ui->active_tab] > 0.0f ? 1.0f : 0.0f;
        Rectangle stone = {layout->side.x + ref->rect.x,
                           row_y + ref->rect.y + pressed,
                           ref->rect.width, ref->rect.height};
        draw_asset_tiled(ui, ref->stone_asset, stone, WHITE);
        DrawRectangleRec(stone, (Color){145, 22, 18, pressed > 0.0f ? 72 : 44});
        break;
    }

    for (int i = 0; i < (int)(sizeof(FC_OSRS_SIDE_STONES) / sizeof(FC_OSRS_SIDE_STONES[0])); i++) {
        const FcUiStoneRef *ref = &FC_OSRS_SIDE_STONES[i];
        float row_y = layout->side.y + (i < 7 ? FC_OSRS_SIDE_TOP_Y : FC_OSRS_SIDE_BOTTOM_Y);
        float pressed = ref->logical_tab == ui->active_tab
            && ui->tab_press_timer[ui->active_tab] > 0.0f ? 1.0f : 0.0f;
        Rectangle icon = {layout->side.x + ref->icon_rect.x, row_y + ref->icon_rect.y,
                          ref->icon_rect.width, ref->icon_rect.height};
        icon.y += pressed;
        fc_ui_draw_asset(&ui->assets, ref->icon_asset, icon, WHITE);
    }
}

static void draw_orb(const FcUiState *ui, Rectangle rect, const char *filler,
                     const char *icon, int value, int max_value) {
    fc_ui_draw_asset(&ui->assets, "orb_frame_0", rect, WHITE);
    Rectangle fill = {rect.x + 27, rect.y + 4, 26, 26};
    fc_ui_draw_asset(&ui->assets, "orb_filler_0", fill, WHITE);
    fc_ui_draw_asset(&ui->assets, filler, fill, WHITE);
    draw_asset_centered(ui, icon, fill, 22, 22, WHITE);

    char text[16];
    snprintf(text, sizeof(text), "%d", value);
    Color text_color = max_value > 0 && value < max_value / 3 ? OSRS_RED : OSRS_GREEN;
    draw_centered_text(ui, text, (Rectangle){rect.x + 3, rect.y + 14, 24, 13}, 12, text_color);
}

static Color orb_value_color(int value, int max_value) {
    if (max_value <= 0)
        max_value = 1;
    if (value < 0)
        value = 0;
    if (value > max_value)
        value = max_value;

    int half = max_value / 2;
    if (half <= 0)
        return value >= max_value ? (Color){0, 255, 0, 255}
                                  : (Color){255, 0, 0, 255};
    if (value > half) {
        int red = 255 - 255 * (value - half) / half;
        return (Color){(unsigned char)red, 255, 0, 255};
    }
    int green = 255 * value / half;
    return (Color){255, (unsigned char)green, 0, 255};
}

static void draw_run_orb(const FcUiState *ui, Rectangle rect) {
    Rectangle button = {rect.x + 3, rect.y + 5, 50, 26};
    const char *frame = CheckCollisionPointRec(GetMousePosition(), button)
        ? "orb_frame_1" : "orb_frame_0";
    fc_ui_draw_asset(&ui->assets, frame, rect, WHITE);

    Rectangle fill = {rect.x + 27, rect.y + 4, 26, 26};
    const char *filler = ui->run_enabled ? "orb_filler_6" : "orb_filler_5";
    const char *icon = ui->run_enabled ? "orb_icon_3" : "orb_icon_2";
    fc_ui_draw_asset(&ui->assets, filler, fill,
                     (Color){255, 255, 255, 230});

    int energy = ui->run_energy;
    if (energy < 0) energy = 0;
    if (energy > 100) energy = 100;
    int empty_height = 26 * (100 - energy) / 100;
    const Texture2D *empty = fc_ui_asset(&ui->assets, "orb_filler_0");
    if (empty_height > 0) {
        float source_height = (float)empty->height * (float)empty_height /
                              fill.height;
        DrawTexturePro(*empty,
                       (Rectangle){0, 0, (float)empty->width, source_height},
                       (Rectangle){fill.x, fill.y, fill.width,
                                   (float)empty_height},
                       (Vector2){0, 0}, 0.0f, WHITE);
    }
    draw_asset_centered(ui, icon, fill, 26, 26, WHITE);

    char text[16];
    snprintf(text, sizeof(text), "%d", energy);
    draw_centered_text(ui, text,
                       (Rectangle){rect.x + 3, rect.y + 14, 24, 13},
                       12, orb_value_color(energy, 100));
}

static void draw_minimap(const FcUiState *ui, const FcUiLayout *layout) {
    Vector2 center = {layout->minimap.x + layout->minimap.width * 0.5f,
                      layout->minimap.y + layout->minimap.height * 0.5f};
    DrawTexturePro(ui->minimap_texture,
                   (Rectangle){0, 0, FC_MINIMAP_DISPLAY_SIZE, FC_MINIMAP_DISPLAY_SIZE},
                   layout->minimap, (Vector2){0, 0}, 0, WHITE);

    for (int i = 0; i < ui->minimap_dot_count; i++) {
        const FcUiMinimapDot *dot = &ui->minimap_dots[i];
        float px = center.x +
            dot->dx * FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
        float py = center.y -
            dot->dy * FC_MINIMAP_DISPLAY_PIXELS_PER_TILE;
        float dx = px - center.x;
        float dy = py - center.y;
        if (dx * dx + dy * dy > 68.0f * 68.0f)
            continue;
        Color c = OSRS_YELLOW;
        float radius = 2.0f;
        if (dot->kind == FC_UI_MINIMAP_DOT_PLAYER) {
            c = WHITE;
            radius = 3.0f;
        } else if (dot->kind == FC_UI_MINIMAP_DOT_DESTINATION) {
            c = OSRS_RED;
            radius = 3.0f;
        }
        DrawCircle((int)px, (int)py, radius, c);
    }

    const Texture2D *compass = fc_ui_asset(&ui->assets, "compass");
    Rectangle source = {0, 0, (float)compass->width, (float)compass->height};
    Rectangle destination = {
        layout->compass.x + layout->compass.width * 0.5f,
        layout->compass.y + layout->compass.height * 0.5f,
        layout->compass.width,
        layout->compass.height,
    };
    Vector2 origin = {layout->compass.width * 0.5f,
                      layout->compass.height * 0.5f};
    DrawTexturePro(*compass, source, destination, origin,
                   ui->minimap_rotation * (180.0f / 3.14159265358979323846f),
                   WHITE);

    Rectangle cover = {layout->map.x + FC_OSRS_MAP_SURROUND_X,
                       layout->map.y + FC_OSRS_MAP_SURROUND_Y,
                       FC_OSRS_MAP_SURROUND_W, FC_OSRS_MAP_SURROUND_H};
    fc_ui_draw_asset(&ui->assets, "osrs_stretch_mapsurround", cover, WHITE);

    draw_orb(ui, layout->hp_orb, "orb_filler_1", "orb_icon_0",
             ui->hitpoints, ui->hitpoints_max);
    draw_orb(ui, layout->prayer_orb, "orb_filler_4", "orb_icon_1",
             ui->prayer_points, ui->prayer_points_max);
    draw_run_orb(ui, layout->run_orb);

    fc_ui_draw_asset(&ui->assets, "ring_30", layout->worldmap_button, WHITE);
    draw_asset_centered(ui, "worldmap_icon_0", layout->worldmap_button, 22, 22, WHITE);
}

static void draw_chat_panel_chrome(const FcUiState *ui,
                                   const FcUiLayout *layout) {
    fc_ui_draw_asset(&ui->assets, "chatbox_bg", layout->chat, WHITE);
    DrawRectangleRec(layout->chat_messages, (Color){0, 0, 0, 104});
    fc_ui_draw_asset(&ui->assets, "main_stones_bottom",
                     layout->chat_controls, WHITE);
}

static Color stack_text_color(int quantity) {
    if (quantity >= 10000000)
        return OSRS_GREEN;
    if (quantity >= 100000)
        return WHITE;
    return OSRS_YELLOW;
}

static void format_stack_quantity(int quantity, char *dst, size_t cap) {
    if (quantity >= 10000000) {
        snprintf(dst, cap, "%dM", quantity / 1000000);
    } else if (quantity >= 100000) {
        snprintf(dst, cap, "%dK", quantity / 1000);
    } else {
        snprintf(dst, cap, "%d", quantity);
    }
}

static const Texture2D *ui_item_icon_texture(const FcUiState *ui,
                                             uint32_t item_id) {
    for (int i = 0; i < ui->item_icon_count; i++) {
        if (ui->item_icons[i].item_id == item_id)
            return &ui->item_icons[i].texture;
    }
    return NULL;
}

static void draw_inventory_item(const FcUiState *ui, const FcUiSlot *slot, Rectangle r) {
    const Texture2D *icon = ui_item_icon_texture(ui, slot->item_id);
    if (!icon)
        return;
    float sx = r.width / (float)icon->width;
    float sy = r.height / (float)icon->height;
    float scale = sx < sy ? sx : sy;
    Rectangle dst = {
        r.x + (r.width - (float)icon->width * scale) * 0.5f,
        r.y + (r.height - (float)icon->height * scale) * 0.5f,
        (float)icon->width * scale,
        (float)icon->height * scale
    };
    DrawTexturePro(*icon,
                   (Rectangle){0, 0, (float)icon->width, (float)icon->height},
                   dst, (Vector2){0, 0}, 0.0f, WHITE);
}

/* An item icon with its stack size in the top-left corner. */
static void draw_slot_item(const FcUiState *ui, const FcUiSlot *slot, Rectangle r) {
    draw_inventory_item(ui, slot, r);
    if (slot->quantity > 1) {
        char q[16];
        format_stack_quantity(slot->quantity, q, sizeof(q));
        draw_text_shadow(ui, q, r.x + 1, r.y - 1, 10, stack_text_color(slot->quantity));
    }
}

static void draw_inventory(const FcUiState *ui, const FcUiLayout *layout) {
    for (int i = 0; i < FC_INVENTORY_SLOTS; i++) {
        if (ui->inventory[i].enabled)
            draw_slot_item(ui, &ui->inventory[i], grid_cell_rect(layout, &g_inventory_grid, i));
    }
}

static void draw_equipment(const FcUiState *ui, const FcUiLayout *layout) {
    draw_asset_tiled(ui, "miscgraphics_2",
                     side_ref_rect(layout, (Rectangle){77, 39, 36, 124}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_2",
                     side_ref_rect(layout, (Rectangle){21, 118, 36, 45}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_2",
                     side_ref_rect(layout, (Rectangle){133, 118, 36, 45}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_3",
                     side_ref_rect(layout, (Rectangle){56, 81, 78, 36}), WHITE);
    draw_asset_tiled(ui, "miscgraphics_3",
                     side_ref_rect(layout, (Rectangle){71, 42, 48, 36}), WHITE);

    for (int i = 0; i < FC_EQUIPMENT_SLOTS; i++) {
        Rectangle r = equip_slot_rect(layout, i);
        if (r.width <= 0)
            continue;
        if (ui->equipment[i].enabled)
            draw_slot_item(ui, &ui->equipment[i], r);
        else if (g_worn_icon_names[i])
            draw_asset_centered(ui, g_worn_icon_names[i], r, 28, 28, (Color){190, 178, 150, 175});
    }
}

static void draw_combat_box(const FcUiState *ui, Rectangle r, int selected) {
    const char *asset = selected ? "combatboxes_1" : "combatboxes_0";
    fc_ui_draw_asset(&ui->assets, asset, r, WHITE);
    if (selected)
        DrawRectangleRec(r, (Color){120, 27, 20, 54});
}

static void draw_combat(const FcUiState *ui, const FcUiLayout *layout) {
    Rectangle header = side_ref_rect(layout, FC_OSRS_COMBAT_HEADER);
    const char *weapon_name = ui->combat_weapon_name[0]
        ? ui->combat_weapon_name : "Unarmed";
    draw_centered_text(ui, weapon_name, side_ref_rect(layout, FC_OSRS_COMBAT_TITLE),
                       13, OSRS_ORANGE);
    char level[32];
    snprintf(level, sizeof(level), "Combat Lvl: %d", ui->combat_level);
    draw_centered_text(ui, level, side_ref_rect(layout, FC_OSRS_COMBAT_LEVEL),
                       12, OSRS_ORANGE);
    DrawLineEx((Vector2){header.x + 10, header.y + 42},
               (Vector2){header.x + header.width - 10, header.y + 42}, 1,
               (Color){75, 64, 45, 180});

    const FcUiCombatProfile *profile = ui->combat_profile;
    for (int i = 0; i < FC_UI_COMBAT_STYLE_COUNT; i++) {
        const FcUiCombatStyleDef *option = &profile->styles[i];
        const FcUiCombatStyleRef *style = &FC_OSRS_COMBAT_STYLES[i];
        int active = i == profile->active_style;
        Rectangle button = side_ref_rect(layout, style->rect);
        draw_combat_box(ui, button, active);
        draw_asset_centered(ui, option->icon_asset, side_ref_rect(layout, style->icon_rect),
                            34, 24, WHITE);
        draw_centered_text(ui, option->label, side_ref_rect(layout, style->text_rect),
                           10, active ? OSRS_YELLOW : OSRS_ORANGE);
    }

    char category[64];
    snprintf(category, sizeof(category), "Attack style: %s",
             profile->styles[profile->active_style].mode);
    draw_centered_text(ui, category, side_ref_rect(layout, FC_OSRS_COMBAT_CATEGORY),
                       12, OSRS_ORANGE);
}

static void draw_side(FcUiState *ui, const FcUiLayout *layout) {
    draw_side_chrome(ui, layout);

    switch (ui->active_tab) {
    case FC_UI_TAB_INVENTORY:
        draw_inventory(ui, layout);
        break;
    case FC_UI_TAB_EQUIPMENT:
        draw_equipment(ui, layout);
        break;
    case FC_UI_TAB_PRAYER:
        /* The viewer draws its protection-prayer tab over the content. */
        break;
    case FC_UI_TAB_COMBAT:
        draw_combat(ui, layout);
        break;
    default:
        break;
    }
}

void fc_ui_draw_context(const FcUiState *ui) {
    if (!ui->context_open)
        return;
    FcMenuLayout menu = context_layout(ui);
    Color brown = {93, 84, 71, 255};
    DrawRectangle(menu.x, menu.y, menu.width, menu.height, brown);
    DrawRectangle(menu.x + 1, menu.y + 1, menu.width - 2, FC_MENU_HEADER_HEIGHT - 3, BLACK);
    DrawRectangleLinesEx((Rectangle){menu.x + 1, menu.y + FC_MENU_HEADER_HEIGHT - 1,
                         menu.width - 2, menu.height - FC_MENU_HEADER_HEIGHT}, 1, BLACK);
    Font font = fc_osrs_menu_font();
    DrawTextEx(font, "Choose Option", (Vector2){menu.x + 4, menu.y + 2},
               FC_OSRS_MENU_FONT_SIZE, 0, brown);
    Vector2 mouse = GetMousePosition();
    int hovered = fc_menu_action_at(menu, ui->context_action_count, (int)mouse.x, (int)mouse.y);
    BeginScissorMode(menu.x, menu.y, menu.width, menu.height);
    for (int i = 0; i < ui->context_action_count; i++) {
        float x = menu.x + 4;
        float y = menu.y + FC_MENU_HEADER_HEIGHT + i * FC_MENU_ROW_HEIGHT + 1;
        const char *action = ui->context_actions[i];
        fc_ui_draw_font_shadow(font, action, (Vector2){x, y}, FC_OSRS_MENU_FONT_SIZE,
                               hovered == i ? OSRS_YELLOW : WHITE);
        if (context_has_target(ui, action)) {
            char target[52];
            snprintf(target, sizeof(target), " %s", ui->context_title);
            x += MeasureTextEx(font, action, FC_OSRS_MENU_FONT_SIZE, 0).x;
            fc_ui_draw_font_shadow(font, target, (Vector2){x, y}, FC_OSRS_MENU_FONT_SIZE,
                                   ui->context_target_color);
            if (ui->context_combat_level > 0) {
                char suffix[32];
                context_level_suffix(ui, suffix);
                x += MeasureTextEx(font, target, FC_OSRS_MENU_FONT_SIZE, 0).x;
                Color color = GetColor((fc_menu_level_color(ui->combat_level,
                    ui->context_combat_level) << 8) | 0xffu);
                fc_ui_draw_font_shadow(font, suffix, (Vector2){x, y},
                                       FC_OSRS_MENU_FONT_SIZE, color);
            }
        }
    }
    EndScissorMode();
}

/* The inventory item being dragged, under the cursor. */
static void draw_dragged_item(const FcUiState *ui) {
    if (ui->drag.active && ui->drag.source_kind == FC_UI_CONTEXT_INVENTORY
            && ui->drag.source_slot >= 0
            && ui->drag.source_slot < FC_INVENTORY_SLOTS
            && ui->inventory[ui->drag.source_slot].enabled) {
        Vector2 mouse = GetMousePosition();
        Rectangle r = {mouse.x - 16, mouse.y - 16,
                       FC_OSRS_INVENTORY_SLOT_W,
                       FC_OSRS_INVENTORY_SLOT_H};
        DrawRectangleRec(r, (Color){0, 0, 0, 80});
        draw_inventory_item(ui, &ui->inventory[ui->drag.source_slot], r);
    }
}

void fc_ui_draw(FcUiState *ui, int screen_w, int screen_h) {
    FcUiLayout layout;
    ui_layout(screen_w, screen_h, &layout);

    draw_chat_panel_chrome(ui, &layout);
    draw_minimap(ui, &layout);
    draw_side(ui, &layout);
    draw_dragged_item(ui);
}

Rectangle fc_ui_chat_panel_rect(int screen_w, int screen_h) {
    FcUiLayout layout;
    ui_layout(screen_w, screen_h, &layout);
    return layout.chat;
}

Rectangle fc_ui_side_content_rect(int screen_w, int screen_h) {
    FcUiLayout layout;
    ui_layout(screen_w, screen_h, &layout);
    return layout.side_content;
}

/* ===================== 3D rendering and debug overlays ===================== */

/* Actor Visual */

#define FC_VISUAL_LOCAL_UNITS 128.0f
#define FC_VISUAL_CLIENT_TICK_SECONDS 0.02f
#define FC_VISUAL_PATH_CAPACITY 10
#define FC_VISUAL_ACTIVE_PATH_MAX 9

typedef enum {
    FC_VISUAL_LOCOMOTION_IDLE = 0,
    FC_VISUAL_LOCOMOTION_TURN,
    FC_VISUAL_LOCOMOTION_WALK_FORWARD,
    FC_VISUAL_LOCOMOTION_WALK_BACK,
    FC_VISUAL_LOCOMOTION_WALK_LEFT,
    FC_VISUAL_LOCOMOTION_WALK_RIGHT,
    FC_VISUAL_LOCOMOTION_RUN,
} FcVisualLocomotion;

typedef enum {
    FC_VISUAL_TARGET_NONE = 0,
    FC_VISUAL_TARGET_PLAYER,
    FC_VISUAL_TARGET_NPC,
} FcVisualTargetKind;

typedef struct {
    int active;
    int size;
    int server_tile_x;
    int server_tile_y;

    /* Persistent client-local position, in 1/128-tile RuneScape units. */
    float local_x;
    float local_y;
    float previous_local_x;
    float previous_local_y;

    int path_x[FC_VISUAL_PATH_CAPACITY];
    int path_y[FC_VISUAL_PATH_CAPACITY];
    unsigned char path_running[FC_VISUAL_PATH_CAPACITY];
    int path_count;

    float yaw_degrees;
    float desired_yaw_degrees;
    FcVisualTargetKind target_kind;
    int target_slot;
    int movement_blocked;
    int moving;
    FcVisualLocomotion locomotion;
} FcVisualActor;

typedef struct {
    FcVisualActor player;
    FcVisualActor npcs[FC_MAX_NPCS];
    float client_tick_accumulator;
    float render_alpha;
} FcVisualScene;

typedef struct {
    float x;
    float y;
    float yaw_degrees;
    FcVisualLocomotion locomotion;
} FcVisualPose;

#define FC_VISUAL_WALK_UNITS_PER_TICK 4.0f
#define FC_VISUAL_TURN_DEGREES_PER_TICK 5.625f

static float normalize_degrees(float degrees) {
    while (degrees >= 180.0f) degrees -= 360.0f;
    while (degrees < -180.0f) degrees += 360.0f;
    return degrees;
}

static float face_angle(float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    if (fabsf(dx) < 0.0001f && fabsf(dy) < 0.0001f) return 0.0f;
    return normalize_degrees(atan2f(dx, -dy) * (180.0f / 3.14159265358979323846f));
}

static void reset_actor(FcVisualActor* actor, int tile_x, int tile_y,
                        int size, float yaw_degrees) {
    if (!actor) return;
    memset(actor, 0, sizeof(*actor));
    actor->active = 1;
    actor->size = size > 0 ? size : 1;
    actor->server_tile_x = tile_x;
    actor->server_tile_y = tile_y;
    actor->local_x = (float)tile_x * FC_VISUAL_LOCAL_UNITS +
                     (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);
    actor->local_y = (float)tile_y * FC_VISUAL_LOCAL_UNITS +
                     (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);
    actor->previous_local_x = actor->local_x;
    actor->previous_local_y = actor->local_y;
    actor->yaw_degrees = normalize_degrees(yaw_degrees);
    actor->desired_yaw_degrees = actor->yaw_degrees;
    actor->target_kind = FC_VISUAL_TARGET_NONE;
    actor->target_slot = -1;
    actor->locomotion = FC_VISUAL_LOCOMOTION_IDLE;
}

void fc_visual_scene_init(FcVisualScene* scene) {
    if (!scene) return;
    memset(scene, 0, sizeof(*scene));
    scene->player.target_slot = -1;
    for (int i = 0; i < FC_MAX_NPCS; i++) scene->npcs[i].target_slot = -1;
    scene->render_alpha = 1.0f;
}

void fc_visual_scene_reset_player(FcVisualScene* scene, int tile_x, int tile_y,
                                  int size, float yaw_degrees) {
    if (!scene) return;
    reset_actor(&scene->player, tile_x, tile_y, size, yaw_degrees);
    scene->client_tick_accumulator = 0.0f;
    scene->render_alpha = 1.0f;
}

void fc_visual_scene_reset_npc(FcVisualScene* scene, int slot, int tile_x,
                               int tile_y, int size, float yaw_degrees) {
    if (!scene || slot < 0 || slot >= FC_MAX_NPCS) return;
    reset_actor(&scene->npcs[slot], tile_x, tile_y, size, yaw_degrees);
}

void fc_visual_scene_deactivate_npc(FcVisualScene* scene, int slot) {
    if (!scene || slot < 0 || slot >= FC_MAX_NPCS) return;
    memset(&scene->npcs[slot], 0, sizeof(scene->npcs[slot]));
    scene->npcs[slot].target_slot = -1;
}

static void pop_path_front(FcVisualActor* actor) {
    if (!actor || actor->path_count <= 0) return;
    actor->path_count--;
    if (actor->path_count > 0) {
        memmove(actor->path_x, actor->path_x + 1,
                actor->path_count * sizeof(actor->path_x[0]));
        memmove(actor->path_y, actor->path_y + 1,
                actor->path_count * sizeof(actor->path_y[0]));
        memmove(actor->path_running, actor->path_running + 1,
                actor->path_count * sizeof(actor->path_running[0]));
    }
}

void fc_visual_actor_enqueue_tile(FcVisualActor* actor, int tile_x, int tile_y,
                                  int running) {
    if (!actor || !actor->active) return;
    actor->server_tile_x = tile_x;
    actor->server_tile_y = tile_y;

    if (actor->path_count > 0) {
        int last = actor->path_count - 1;
        if (actor->path_x[last] == tile_x && actor->path_y[last] == tile_y) {
            actor->path_running[last] = running ? 1u : 0u;
            return;
        }
    }

    /* Native actor queues store ten entries but keep at most nine active
     * route points. A new server step drops the oldest visual waypoint. */
    if (actor->path_count >= FC_VISUAL_ACTIVE_PATH_MAX) pop_path_front(actor);

    int next = actor->path_count++;
    actor->path_x[next] = tile_x;
    actor->path_y[next] = tile_y;
    actor->path_running[next] = running ? 1u : 0u;
}

void fc_visual_actor_enqueue_transition(FcVisualActor* actor, int from_x,
                                        int from_y, int to_x, int to_y,
                                        int running) {
    if (!actor || !actor->active) return;
    int x = from_x;
    int y = from_y;
    int dx = to_x - from_x;
    int dy = to_y - from_y;
    int steps = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    int sx = (dx > 0) - (dx < 0);
    int sy = (dy > 0) - (dy < 0);

    /* Large transitions are resets/teleports, not ordinary route updates. */
    if (steps > FC_VISUAL_PATH_CAPACITY) {
        reset_actor(actor, to_x, to_y, actor->size, actor->yaw_degrees);
        return;
    }

    for (int i = 0; i < steps; i++) {
        if (x != to_x) x += sx;
        if (y != to_y) y += sy;
        fc_visual_actor_enqueue_tile(actor, x, y, running);
    }
    if (steps == 0) {
        actor->server_tile_x = to_x;
        actor->server_tile_y = to_y;
    }
}

void fc_visual_actor_set_target(FcVisualActor* actor,
                                FcVisualTargetKind target_kind,
                                int target_slot) {
    if (!actor) return;
    actor->target_kind = target_kind;
    actor->target_slot = target_slot;
}

void fc_visual_actor_set_movement_blocked(FcVisualActor* actor, int blocked) {
    if (!actor) return;
    actor->movement_blocked = blocked ? 1 : 0;
}

static const FcVisualActor* target_actor(const FcVisualScene* scene,
                                         const FcVisualActor* actor) {
    if (!scene || !actor) return NULL;
    if (actor->target_kind == FC_VISUAL_TARGET_PLAYER)
        return scene->player.active ? &scene->player : NULL;
    if (actor->target_kind == FC_VISUAL_TARGET_NPC &&
        actor->target_slot >= 0 && actor->target_slot < FC_MAX_NPCS &&
        scene->npcs[actor->target_slot].active)
        return &scene->npcs[actor->target_slot];
    return NULL;
}

static void move_toward(float* value, float destination, float speed) {
    if (*value < destination) {
        *value += speed;
        if (*value > destination) *value = destination;
    } else if (*value > destination) {
        *value -= speed;
        if (*value < destination) *value = destination;
    }
}

static FcVisualLocomotion directional_locomotion(float movement_yaw,
                                                  float actor_yaw,
                                                  int fast_movement) {
    float relative = normalize_degrees(movement_yaw - actor_yaw);
    if (relative >= -45.0f && relative <= 45.0f) {
        return fast_movement ? FC_VISUAL_LOCOMOTION_RUN
                             : FC_VISUAL_LOCOMOTION_WALK_FORWARD;
    }
    if (relative > 45.0f && relative < 135.0f)
        return FC_VISUAL_LOCOMOTION_WALK_RIGHT;
    if (relative < -45.0f && relative > -135.0f)
        return FC_VISUAL_LOCOMOTION_WALK_LEFT;
    return FC_VISUAL_LOCOMOTION_WALK_BACK;
}

static void update_actor_movement(FcVisualActor* actor) {
    actor->moving = 0;
    actor->locomotion = FC_VISUAL_LOCOMOTION_IDLE;
    if (!actor->active || actor->path_count <= 0 || actor->movement_blocked)
        return;

    float dst_x = (float)actor->path_x[0] * FC_VISUAL_LOCAL_UNITS +
                  (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);
    float dst_y = (float)actor->path_y[0] * FC_VISUAL_LOCAL_UNITS +
                  (float)actor->size * (FC_VISUAL_LOCAL_UNITS * 0.5f);

    /* The native client snaps to a queued waypoint when local prediction is
     * more than two tiles out of sync, rather than gliding across the gap. */
    if (fabsf(actor->local_x - dst_x) > 256.0f ||
        fabsf(actor->local_y - dst_y) > 256.0f) {
        actor->local_x = dst_x;
        actor->local_y = dst_y;
        actor->previous_local_x = dst_x;
        actor->previous_local_y = dst_y;
        return;
    }

    float movement_yaw = face_angle(actor->local_x, actor->local_y, dst_x, dst_y);
    int running = actor->path_running[0] != 0;
    float speed = FC_VISUAL_WALK_UNITS_PER_TICK;
    if (actor->target_kind == FC_VISUAL_TARGET_NONE &&
        fabsf(normalize_degrees(movement_yaw - actor->yaw_degrees)) > 0.01f)
        speed = 2.0f;
    if (actor->path_count > 2) speed = 6.0f;
    if (actor->path_count > 3) speed = 8.0f;
    if (running) speed *= 2.0f;

    actor->desired_yaw_degrees = movement_yaw;
    actor->locomotion = directional_locomotion(
        movement_yaw, actor->yaw_degrees, running || speed >= 8.0f);
    actor->moving = 1;
    move_toward(&actor->local_x, dst_x, speed);
    move_toward(&actor->local_y, dst_y, speed);
    if (fabsf(actor->local_x - dst_x) < 0.001f &&
        fabsf(actor->local_y - dst_y) < 0.001f)
        pop_path_front(actor);
}

static void update_actor_facing(const FcVisualScene* scene,
                                FcVisualActor* actor) {
    if (!actor->active) return;
    const FcVisualActor* target = target_actor(scene, actor);
    if (target) {
        actor->desired_yaw_degrees = face_angle(
            actor->local_x, actor->local_y, target->local_x, target->local_y);
    }

    float delta = normalize_degrees(actor->desired_yaw_degrees -
                                    actor->yaw_degrees);
    if (fabsf(delta) <= FC_VISUAL_TURN_DEGREES_PER_TICK) {
        actor->yaw_degrees = actor->desired_yaw_degrees;
    } else {
        actor->yaw_degrees = normalize_degrees(
            actor->yaw_degrees +
            (delta > 0.0f ? FC_VISUAL_TURN_DEGREES_PER_TICK
                          : -FC_VISUAL_TURN_DEGREES_PER_TICK));
    }

    if (!actor->moving && fabsf(delta) > 0.01f)
        actor->locomotion = FC_VISUAL_LOCOMOTION_TURN;
}

static void update_client_tick(FcVisualScene* scene) {
    scene->player.previous_local_x = scene->player.local_x;
    scene->player.previous_local_y = scene->player.local_y;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        scene->npcs[i].previous_local_x = scene->npcs[i].local_x;
        scene->npcs[i].previous_local_y = scene->npcs[i].local_y;
    }

    update_actor_movement(&scene->player);
    for (int i = 0; i < FC_MAX_NPCS; i++)
        update_actor_movement(&scene->npcs[i]);

    update_actor_facing(scene, &scene->player);
    for (int i = 0; i < FC_MAX_NPCS; i++)
        update_actor_facing(scene, &scene->npcs[i]);
}

void fc_visual_scene_update(FcVisualScene* scene, float elapsed_seconds) {
    if (!scene || elapsed_seconds <= 0.0f) return;
    scene->client_tick_accumulator += elapsed_seconds;
    /* Avoid an unbounded catch-up loop after a debugger stop or window drag. */
    if (scene->client_tick_accumulator > 0.25f)
        scene->client_tick_accumulator = 0.25f;
    while (scene->client_tick_accumulator >= FC_VISUAL_CLIENT_TICK_SECONDS) {
        update_client_tick(scene);
        scene->client_tick_accumulator -= FC_VISUAL_CLIENT_TICK_SECONDS;
    }
    scene->render_alpha = scene->client_tick_accumulator /
                          FC_VISUAL_CLIENT_TICK_SECONDS;
}

FcVisualPose fc_visual_actor_pose(const FcVisualScene* scene,
                                  const FcVisualActor* actor) {
    FcVisualPose pose = {0};
    if (!scene || !actor || !actor->active) return pose;
    float alpha = scene->render_alpha; /* 0..1 by construction */
    float local_x = actor->previous_local_x +
                    (actor->local_x - actor->previous_local_x) * alpha;
    float local_y = actor->previous_local_y +
                    (actor->local_y - actor->previous_local_y) * alpha;
    pose.x = local_x / FC_VISUAL_LOCAL_UNITS;
    pose.y = local_y / FC_VISUAL_LOCAL_UNITS;
    pose.yaw_degrees = actor->yaw_degrees;
    pose.locomotion = actor->locomotion;
    return pose;
}

FcVisualPose fc_visual_scene_player_pose(const FcVisualScene* scene) {
    return scene ? fc_visual_actor_pose(scene, &scene->player)
                 : (FcVisualPose){0};
}

FcVisualPose fc_visual_scene_npc_pose(const FcVisualScene* scene, int slot) {
    if (!scene || slot < 0 || slot >= FC_MAX_NPCS)
        return (FcVisualPose){0};
    return fc_visual_actor_pose(scene, &scene->npcs[slot]);
}

/* Actor Animation */

/* How an attack's projectile looks and flies, in client units: spot
 * animation ids (0 = none), heights in 1/128 tile above the ground, and
 * client cycles (30 per game tick). */
typedef struct {
    uint32_t travel_spot;
    uint32_t launch_spot;
    uint32_t impact_spot;
    float start_height;
    float end_height;
    float launch_cycle;       /* when it leaves the source */
    float angle;              /* launch elevation, 1/256 turn */
    float length_adjustment;  /* flight cycles at any distance */
    float progress;           /* start offset toward the target, 1/128 tile */
    float step_multiplier;    /* flight cycles per tile of distance */
    int fixed_target;         /* lands where the target stood at launch */
} FcProjectileProfile;

typedef struct {
    uint16_t idle_anim;
    uint16_t walk_anim;
    uint16_t walk_back_anim;
    uint16_t walk_left_anim;
    uint16_t walk_right_anim;
    uint16_t turn_anim;
    uint16_t run_anim;
    uint16_t attack_anim;
    FcProjectileProfile projectile;
} FcPlayerVisualProfile;

/* One NPC slot: its posed vertices, the pose (idle/walk) and action
 * (attack/death) tracks, what its last attack started, and its server tile
 * at the start of the current tick. */
typedef struct {
    AnimModelState *state;
    AnimTrack pose;
    AnimTrack action;
    int attack_style;
    float attack_timer;
    float prayer_indicator_timer;
    int prayer_lock_tick;
    int previous_x;
    int previous_y;
    int previous_active;
} FcNpcAnimation;

typedef struct {
    FcVisualScene scene;
    AnimModelState *player_state;
    AnimTrack player_pose;
    AnimTrack player_action;
    uint16_t player_lock_sequence;
    float player_lock_timer;
    int player_attack_target;
    float prayer_flick_timer;
    FcNpcAnimation npcs[FC_MAX_NPCS];
} FcActorAnimation;

#define POLICY_REPLAY_BASE_TPS (5.0f / 3.0f)

#define PLAYER_ANIM_HUMAN_IDLE 808
#define PLAYER_ANIM_HUMAN_WALK 819
#define PLAYER_ANIM_HUMAN_WALK_BACK 820
#define PLAYER_ANIM_HUMAN_WALK_RIGHT 821
#define PLAYER_ANIM_HUMAN_WALK_LEFT 822
#define PLAYER_ANIM_HUMAN_TURN 823
#define PLAYER_ANIM_HUMAN_RUN 824
#define PLAYER_ANIM_BOW_ATTACK 426
#define PLAYER_ANIM_EAT 829
#define PLAYER_ANIM_DEATH 836

#define JAD_ANIM_RANGED 2652
#define JAD_ANIM_MELEE 2655
#define JAD_ANIM_MAGIC 2656

/* The Twisted bow's animations. Projectile columns follow
 * FcProjectileProfile. */
static const FcPlayerVisualProfile TWISTED_BOW_VISUALS = {
    PLAYER_ANIM_HUMAN_IDLE, PLAYER_ANIM_HUMAN_WALK,
    PLAYER_ANIM_HUMAN_WALK_BACK, PLAYER_ANIM_HUMAN_WALK_LEFT,
    PLAYER_ANIM_HUMAN_WALK_RIGHT, PLAYER_ANIM_HUMAN_TURN,
    PLAYER_ANIM_HUMAN_RUN, PLAYER_ANIM_BOW_ATTACK,
    {1120, 1116, 0, 163.0f, 146.0f, 41.0f, 15.0f, 5.0f, 11.0f, 5.0f, 0},
};

typedef struct {
    uint16_t idle;
    uint16_t walk;
    uint16_t attack;
    uint16_t death;
} FcNpcAnims;

static const FcNpcAnims NPC_ANIMS[NPC_TYPE_COUNT] = {
    [NPC_TZ_KIH] = {2618, 2619, 2621, 2620},
    [NPC_TZ_KEK] = {2624, 2623, 2625, 2627},
    [NPC_TZ_KEK_SM] = {2624, 2623, 2625, 2627},
    [NPC_TOK_XIL] = {2631, 2632, 2628, 2630},
    [NPC_YT_MEJKOT] = {2636, 2634, 2637, 2638},
    [NPC_KET_ZEK] = {2642, 2643, 2644, 2646},
    [NPC_TZTOK_JAD] = {2650, 2651, 2655, 2654},
    [NPC_YT_HURKOT] = {2636, 2634, 2637, 2638},
};

/* The sequences of an NPC type; NPC_NONE (and any unknown type) has none. */
static const FcNpcAnims *npc_anims(int npc_type) {
    if (npc_type <= NPC_NONE || npc_type >= NPC_TYPE_COUNT)
        npc_type = NPC_NONE;
    return &NPC_ANIMS[npc_type];
}

/* The wielded weapon's animations: the Twisted bow's, or unarmed (no
 * projectile). */
const FcPlayerVisualProfile *fc_player_visual_profile(const FcPlayer *player) {
    static const FcPlayerVisualProfile unarmed = {
        .idle_anim=808, .walk_anim=819, .walk_back_anim=820,
        .walk_left_anim=822, .walk_right_anim=821, .turn_anim=823,
        .run_anim=824, .attack_anim=422,
    };
    return player->weapon_kind == FC_WEAPON_TWISTED_BOW
        ? &TWISTED_BOW_VISUALS : &unarmed;
}

ModelEntry *fc_actor_player_model_entry(ModelSet *player_models) {
    if (!player_models) return NULL;
    return model_find(player_models, FC_PLAYER_MODEL_ID);
}


void fc_actor_animation_upload_npc(FcActorAnimation *animation,
                                   int npc_slot,
                                   ModelEntry *entry) {
    if (!animation || npc_slot < 0 || npc_slot >= FC_MAX_NPCS) return;
    fc_model_animation_upload(entry, animation->npcs[npc_slot].state);
}

/* Plays `desired` on an actor track. Returns its sequence, or NULL when there
 * is nothing to play. */
static AnimSequence *advance_track(AnimCache *cache, uint16_t desired,
                                   AnimTrack *track, float dt, int play_once) {
    if (!cache || desired == 0) return NULL;
    AnimSequence *sequence = anim_get_sequence(cache, desired);
    if (!sequence || sequence->frame_count == 0) return NULL;
    anim_track_step(track, sequence, dt, play_once);
    return sequence;
}

static float sequence_duration(const AnimSequence *sequence) {
    if (!sequence || sequence->frame_count == 0) return 0.45f;
    float total = 0.0f;
    for (int i = 0; i < sequence->frame_count; i++)
        total += anim_frame_duration(sequence, i);
    return total < 0.35f ? 0.35f : total;
}

static uint16_t npc_attack_sequence(int npc_type, int attack_style) {
    if (npc_type == NPC_TZTOK_JAD) {
        if (attack_style == ATTACK_MAGIC) return JAD_ANIM_MAGIC;
        if (attack_style == ATTACK_RANGED) return JAD_ANIM_RANGED;
        if (attack_style == ATTACK_MELEE) return JAD_ANIM_MELEE;
    }
    return npc_anims(npc_type)->attack;
}

/* Counts a running timer down by dt, stopping at zero. Returns 1 on the call
 * that runs it out. */
static int tick_down(float *timer, float dt) {
    if (*timer > 0.0f) {
        *timer -= dt;
        if (*timer <= 0.0f) {
            *timer = 0.0f;
            return 1;
        }
    }
    return 0;
}

float fc_actor_animation_scaled_dt(float tps, float dt) {
    if (tps <= 0.0f) return dt;
    float scale = tps / POLICY_REPLAY_BASE_TPS;
    if (scale < 0.05f) scale = 0.05f;
    if (scale > 36.0f) scale = 36.0f;
    return dt * scale;
}

float fc_actor_animation_scaled_duration(float tps, float seconds) {
    float scale = tps > 0.0f ? tps / POLICY_REPLAY_BASE_TPS : 1.0f;
    if (scale < 0.05f) scale = 0.05f;
    if (scale > 36.0f) scale = 36.0f;
    seconds /= scale;
    return seconds < 0.05f ? 0.05f : seconds;
}

static int player_lock_active(const FcActorAnimation *animation) {
    return animation && animation->player_lock_sequence != 0 &&
           animation->player_lock_timer > 0.0f;
}

static uint16_t player_action_sequence(const FcActorAnimation *animation,
                                       const FcState *state) {
    if (!animation || !state) return 0;
    if (state->terminal == TERMINAL_PLAYER_DEATH) return PLAYER_ANIM_DEATH;
    if (state->player.food_eaten_this_tick) return PLAYER_ANIM_EAT;
    return player_lock_active(animation) ? animation->player_lock_sequence : 0;
}

/* The death or attack sequence an NPC plays over its pose, or 0. A death
 * waits while a projectile still carries damage to the NPC. */
static uint16_t npc_action_sequence(const FcActorAnimation *animation,
                                    const FcState *state, int slot,
                                    const unsigned char deferred_deaths[FC_MAX_NPCS]) {
    const FcNpc *npc = &state->npcs[slot];
    const FcNpcAnimation *anim = &animation->npcs[slot];
    if ((npc->is_dead || npc->died_this_tick) &&
        (!deferred_deaths || !deferred_deaths[slot]))
        return npc_anims(npc->npc_type)->death;
    if (anim->attack_timer > 0.0f)
        return npc_attack_sequence(npc->npc_type, anim->attack_style);
    return 0;
}

static int sequence_blocks_movement(AnimCache *cache, uint16_t sequence_id) {
    if (!cache || sequence_id == 0) return 0;
    AnimSequence *sequence = anim_get_sequence(cache, sequence_id);
    return sequence && sequence->postanim_move == 0;
}

static void recreate_player_state(FcActorAnimation *animation,
                                  ModelEntry *entry) {
    if (!animation || !entry || !entry->vertex_skins) return;
    if (animation->player_state &&
        animation->player_state->vert_count == entry->base_vert_count) return;
    if (animation->player_state) anim_model_state_free(animation->player_state);
    animation->player_state = anim_model_state_create(entry->vertex_skins,
                                                       entry->base_vert_count);
    fprintf(stderr, "Player animation state created (%d base verts, model %u)\n",
            entry->base_vert_count, entry->model_id);
}

void fc_actor_animation_reset(FcActorAnimation *animation,
                              const FcState *state,
                              ModelSet *player_models) {
    if (!animation || !state) return;
    fc_visual_scene_init(&animation->scene);
    fc_visual_scene_reset_player(&animation->scene, state->player.x,
                                 state->player.y, 1,
                                 state->player.facing_angle);
    const FcPlayerVisualProfile *profile =
        fc_player_visual_profile(&state->player);
    animation->player_pose = (AnimTrack){profile->idle_anim, 0, 0.0f};
    animation->player_action = (AnimTrack){0};
    animation->player_lock_sequence = 0;
    animation->player_lock_timer = 0.0f;
    animation->player_attack_target = -1;
    animation->prayer_flick_timer = 0.0f;
    recreate_player_state(animation,
                          fc_actor_player_model_entry(player_models));
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        FcNpcAnimation *anim = &animation->npcs[i];
        if (npc->active || npc->died_this_tick) {
            fc_visual_scene_reset_npc(&animation->scene, i, npc->x, npc->y,
                                      npc->size, 0.0f);
        }
        if (anim->state) anim_model_state_free(anim->state);
        memset(anim, 0, sizeof(*anim));
        anim->prayer_lock_tick = -1;
        anim->previous_x = npc->x;
        anim->previous_y = npc->y;
        anim->previous_active = npc->active;
    }
}

void fc_actor_animation_shutdown(FcActorAnimation *animation) {
    if (!animation) return;
    if (animation->player_state) anim_model_state_free(animation->player_state);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (animation->npcs[i].state)
            anim_model_state_free(animation->npcs[i].state);
    }
    memset(animation, 0, sizeof(*animation));
}

void fc_actor_animation_capture_tick_start(FcActorAnimation *animation,
                                           const FcState *state) {
    if (!animation || !state) return;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        animation->npcs[i].previous_x = state->npcs[i].x;
        animation->npcs[i].previous_y = state->npcs[i].y;
        animation->npcs[i].previous_active = state->npcs[i].active;
    }
}

void fc_actor_animation_ingest_tick(FcActorAnimation *animation,
                                    const FcState *state,
                                    const FcRenderEvents *events) {
    if (!animation || !state || !events) return;
    FcVisualActor *player = &animation->scene.player;
    if (!player->active) {
        fc_visual_scene_reset_player(&animation->scene,
            events->player_move_start_x, events->player_move_start_y, 1,
            state->player.facing_angle);
    }
    int waypoint_count = events->player_move_waypoint_count;
    int running = waypoint_count > 1;
    for (int i = 0; i < waypoint_count; i++) {
        fc_visual_actor_enqueue_tile(player, events->player_move_waypoint_x[i],
                                     events->player_move_waypoint_y[i], running);
    }
    if (waypoint_count == 0 &&
        (player->server_tile_x != state->player.x ||
         player->server_tile_y != state->player.y)) {
        fc_visual_actor_enqueue_transition(player, player->server_tile_x,
            player->server_tile_y, state->player.x, state->player.y,
            state->player.is_running);
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        const FcNpcAnimation *anim = &animation->npcs[i];
        FcVisualActor *visual = &animation->scene.npcs[i];
        if (npc->active && !anim->previous_active) {
            fc_visual_scene_reset_npc(&animation->scene, i, npc->x, npc->y,
                                      npc->size, 0.0f);
        } else if ((npc->active || npc->died_this_tick) && visual->active) {
            fc_visual_actor_enqueue_transition(visual,
                anim->previous_x, anim->previous_y, npc->x, npc->y, 0);
        } else if (!npc->active && !npc->died_this_tick) {
            fc_visual_scene_deactivate_npc(&animation->scene, i);
        }
    }
}

void fc_actor_animation_ingest_events(FcActorAnimation *animation,
                                      const FcRenderEvents *events,
                                      AnimCache *cache,
                                      const FcPlayerVisualProfile *profile,
                                      float tps) {
    if (!animation || !events) return;
    if (events->player_attack_fired) {
        AnimSequence *sequence = cache
            ? anim_get_sequence(cache, profile->attack_anim) : NULL;
        animation->player_lock_sequence = profile->attack_anim;
        animation->player_lock_timer = fc_actor_animation_scaled_duration(
            tps, sequence_duration(sequence));
        animation->player_attack_target =
            events->player_attack_target_npc_slot;
        animation->player_action = (AnimTrack){0};
    }
    if (events->prayer_flick_performed) {
        animation->prayer_flick_timer =
            fc_actor_animation_scaled_duration(tps, 0.10f);
    }
    for (int i = 0; i < events->npc_attack_count; i++) {
        const FcRenderNpcAttack *attack = &events->npc_attacks[i];
        int slot = attack->npc_slot;
        if (slot < 0 || slot >= FC_MAX_NPCS ||
            attack->attack_style == ATTACK_NONE) continue;
        FcNpcAnimation *anim = &animation->npcs[slot];
        anim->attack_style = attack->attack_style;
        anim->attack_timer = 1.15f;
        anim->prayer_indicator_timer = 0.30f;
        anim->prayer_lock_tick = attack->prayer_lock_tick;
    }
}

static void update_targets(FcActorAnimation *animation, const FcState *state) {
    int target = player_lock_active(animation)
        ? animation->player_attack_target : state->player.attack_target_idx;
    if (target >= 0 && target < FC_MAX_NPCS &&
        animation->scene.npcs[target].active) {
        fc_visual_actor_set_target(&animation->scene.player,
                                   FC_VISUAL_TARGET_NPC, target);
    } else {
        fc_visual_actor_set_target(&animation->scene.player,
                                   FC_VISUAL_TARGET_NONE, -1);
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcVisualActor *actor = &animation->scene.npcs[i];
        if (actor->active && state->npcs[i].active) {
            int heal_target = state->npcs[i].heal_target_idx;
            if (heal_target >= 0 && heal_target < FC_MAX_NPCS &&
                heal_target != i && animation->scene.npcs[heal_target].active) {
                fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_NPC,
                                           heal_target);
            } else if (heal_target == i) {
                fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_NONE, -1);
            } else {
                fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_PLAYER, 0);
            }
        } else {
            fc_visual_actor_set_target(actor, FC_VISUAL_TARGET_NONE, -1);
        }
    }
}

void fc_actor_animation_update_scene(FcActorAnimation *animation,
                                     const FcState *state,
                                     AnimCache *cache,
                                     float tps,
                                     float dt,
                                     int advance_scene,
                                     const unsigned char deferred_deaths[FC_MAX_NPCS]) {
    if (!animation || !state) return;
    update_targets(animation, state);
    fc_visual_actor_set_movement_blocked(&animation->scene.player,
        sequence_blocks_movement(cache,
            player_action_sequence(animation, state)));
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        fc_visual_actor_set_movement_blocked(&animation->scene.npcs[i],
            sequence_blocks_movement(cache,
                npc_action_sequence(animation, state, i, deferred_deaths)));
    }
    float visual_dt = fc_actor_animation_scaled_dt(tps, dt);
    if (advance_scene) fc_visual_scene_update(&animation->scene, visual_dt);
    if (tick_down(&animation->player_lock_timer, dt)) {
        animation->player_lock_sequence = 0;
        animation->player_attack_target = -1;
        animation->player_action = (AnimTrack){0};
    }
    tick_down(&animation->prayer_flick_timer, dt);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpcAnimation *anim = &animation->npcs[i];
        if (anim->prayer_lock_tick >= 0 &&
            state->tick >= anim->prayer_lock_tick) {
            anim->prayer_indicator_timer = 0.0f;
            anim->prayer_lock_tick = -1;
        }
        tick_down(&anim->prayer_indicator_timer, dt);
    }
}

static uint16_t player_pose_sequence(const FcPlayerVisualProfile *profile,
                                     FcVisualLocomotion locomotion) {
    switch (locomotion) {
        case FC_VISUAL_LOCOMOTION_TURN: return profile->turn_anim;
        case FC_VISUAL_LOCOMOTION_WALK_BACK: return profile->walk_back_anim;
        case FC_VISUAL_LOCOMOTION_WALK_LEFT: return profile->walk_left_anim;
        case FC_VISUAL_LOCOMOTION_WALK_RIGHT: return profile->walk_right_anim;
        case FC_VISUAL_LOCOMOTION_RUN: return profile->run_anim;
        case FC_VISUAL_LOCOMOTION_WALK_FORWARD: return profile->walk_anim;
        default: return profile->idle_anim;
    }
}

void fc_actor_animation_update_models(FcActorAnimation *animation,
                                      const FcState *state,
                                      ModelSet *player_models,
                                      ModelSet *npc_models,
                                      AnimCache *cache,
                                      float tps,
                                      float dt,
                                      const unsigned char deferred_deaths[FC_MAX_NPCS]) {
    if (!animation || !state || !cache) return;
    float anim_dt = fc_actor_animation_scaled_dt(tps, dt);
    ModelEntry *player_entry = fc_actor_player_model_entry(player_models);
    recreate_player_state(animation, player_entry);
    if (animation->player_state && player_entry) {
        const FcPlayerVisualProfile *profile =
            fc_player_visual_profile(&state->player);
        FcVisualPose pose = fc_visual_scene_player_pose(&animation->scene);
        uint16_t pose_sequence = player_pose_sequence(profile, pose.locomotion);
        uint16_t action_sequence = player_action_sequence(animation, state);
        AnimSequence *pose_track = advance_track(cache, pose_sequence,
            &animation->player_pose, anim_dt, 0);
        AnimSequence *action_track = NULL;
        if (action_sequence != 0) {
            action_track = advance_track(cache, action_sequence,
                &animation->player_action, anim_dt,
                player_lock_active(animation));
        } else {
            animation->player_action = (AnimTrack){0};
        }
        if (anim_mix_pose_action(cache, animation->player_state,
                player_entry->base_verts, pose_track,
                animation->player_pose.frame, action_track,
                animation->player_action.frame)) {
            fc_model_animation_upload(player_entry, animation->player_state);
        }
    }
    if (!npc_models) return;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        FcNpcAnimation *anim = &animation->npcs[i];
        if (!npc->active && !npc->died_this_tick) {
            if (anim->state) {
                anim_model_state_free(anim->state);
                anim->state = NULL;
            }
            anim->attack_style = ATTACK_NONE;
            anim->attack_timer = 0.0f;
            anim->prayer_indicator_timer = 0.0f;
            anim->prayer_lock_tick = -1;
            continue;
        }
        ModelEntry *entry = npc_model_find(npc_models, npc->npc_type);
        if (!entry || !entry->vertex_skins) continue;
        const FcNpcAnims *anims = npc_anims(npc->npc_type);
        if (!anim->state || anim->state->vert_count != entry->base_vert_count) {
            if (anim->state) anim_model_state_free(anim->state);
            anim->state = anim_model_state_create(entry->vertex_skins,
                                                  entry->base_vert_count);
            anim->pose = (AnimTrack){anims->idle, 0, 0.0f};
            anim->action = (AnimTrack){0};
        }
        uint16_t pose_sequence = animation->scene.npcs[i].moving
            ? anims->walk : anims->idle;
        uint16_t action_sequence =
            npc_action_sequence(animation, state, i, deferred_deaths);
        AnimSequence *pose_track = advance_track(cache, pose_sequence,
            &anim->pose, anim_dt, 0);
        AnimSequence *action_track = NULL;
        if (action_sequence != 0) {
            action_track = advance_track(cache, action_sequence,
                &anim->action, anim_dt, 1);
        } else {
            anim->action = (AnimTrack){0};
        }
        anim_mix_pose_action(cache, anim->state, entry->base_verts,
            pose_track, anim->pose.frame, action_track, anim->action.frame);
        if (tick_down(&anim->attack_timer, anim_dt))
            anim->attack_style = ATTACK_NONE;
    }
}

int fc_actor_animation_render_prayer(const FcActorAnimation *animation,
                                     const FcState *state) {
    if (!animation || !state || animation->prayer_flick_timer > 0.0f)
        return PRAYER_NONE;
    return state->player.prayer;
}

int fc_actor_animation_prayer_window_active(const FcActorAnimation *animation,
                                            int npc_slot,
                                            int current_tick) {
    if (!animation || npc_slot < 0 || npc_slot >= FC_MAX_NPCS) return 0;
    const FcNpcAnimation *anim = &animation->npcs[npc_slot];
    return anim->prayer_indicator_timer > 0.0f ||
        (anim->prayer_lock_tick >= 0 && current_tick < anim->prayer_lock_tick);
}

int fc_actor_animation_previous_npc_active(const FcActorAnimation *animation,
                                           int npc_slot) {
    return animation && npc_slot >= 0 && npc_slot < FC_MAX_NPCS
        ? animation->npcs[npc_slot].previous_active : 0;
}

/* Click Feedback */

#define FC_CLICK_CROSS_FRAME_COUNT 4
#define FC_CLICK_CROSS_FRAME_SECONDS 0.10f

typedef enum {
    FC_CLICK_CROSS_NONE = 0,
    FC_CLICK_CROSS_MOVE,
    FC_CLICK_CROSS_INTERACTION,
} FcClickCrossKind;

typedef struct {
    int destination_active;
    int destination_x;
    int destination_y;

    int preview_pending;
    int preview_route_x[FC_MAX_ROUTE];
    int preview_route_y[FC_MAX_ROUTE];
    int preview_route_len;

    FcClickCrossKind cross_kind;
    float cross_screen_x;
    float cross_screen_y;
    float cross_elapsed;
} FcClickFeedback;

void fc_click_feedback_reset(FcClickFeedback* feedback) {
    if (!feedback) return;
    memset(feedback, 0, sizeof(*feedback));
    feedback->destination_x = -1;
    feedback->destination_y = -1;
}

static void start_cross(FcClickFeedback* feedback, FcClickCrossKind kind,
                        float screen_x, float screen_y) {
    feedback->cross_kind = kind;
    feedback->cross_screen_x = screen_x;
    feedback->cross_screen_y = screen_y;
    feedback->cross_elapsed = 0.0f;
}

/* Build a read-only preview with the same move-near pathfinder used by
 * fc_step(). The simulation state itself is never modified. */
void fc_click_feedback_select_move(FcClickFeedback* feedback,
                                   const FcState* state,
                                   int tile_x, int tile_y,
                                   float screen_x, float screen_y) {
    if (!feedback || !state ||
        tile_x < 0 || tile_x >= FC_ARENA_WIDTH ||
        tile_y < 0 || tile_y >= FC_ARENA_HEIGHT) {
        return;
    }

    feedback->destination_active = 1;
    feedback->destination_x = tile_x;
    feedback->destination_y = tile_y;
    feedback->preview_pending = 1;
    feedback->preview_route_len = fc_pathfind_bfs_move_near(
        state->player.x, state->player.y, tile_x, tile_y,
        state->walkable, state->movement_flags,
        feedback->preview_route_x, feedback->preview_route_y, FC_MAX_ROUTE);
    start_cross(feedback, FC_CLICK_CROSS_MOVE, screen_x, screen_y);
}

void fc_click_feedback_select_interaction(FcClickFeedback* feedback,
                                          float screen_x, float screen_y) {
    if (!feedback) return;
    feedback->destination_active = 0;
    feedback->destination_x = -1;
    feedback->destination_y = -1;
    feedback->preview_pending = 0;
    feedback->preview_route_len = 0;
    start_cross(feedback, FC_CLICK_CROSS_INTERACTION, screen_x, screen_y);
}

/* Clear a completed or cancelled authoritative destination. */
void fc_click_feedback_sync(FcClickFeedback* feedback,
                            const FcState* state) {
    if (!feedback || !state || !feedback->destination_active ||
        feedback->preview_pending) {
        return;
    }
    if (state->player.route_idx >= state->player.route_len) {
        feedback->destination_active = 0;
        feedback->destination_x = -1;
        feedback->destination_y = -1;
    }
}

/* Hand the preview over to the route produced by the authoritative tick. */
void fc_click_feedback_accept_move_tick(FcClickFeedback* feedback,
                                        const FcState* state) {
    if (!feedback || !state || !feedback->preview_pending) return;
    feedback->preview_pending = 0;
    feedback->preview_route_len = 0;
    fc_click_feedback_sync(feedback, state);
}

void fc_click_feedback_update(FcClickFeedback* feedback,
                              float elapsed_seconds) {
    if (!feedback || feedback->cross_kind == FC_CLICK_CROSS_NONE ||
        elapsed_seconds <= 0.0f) {
        return;
    }
    feedback->cross_elapsed += elapsed_seconds;
    if (feedback->cross_elapsed >=
        FC_CLICK_CROSS_FRAME_COUNT * FC_CLICK_CROSS_FRAME_SECONDS) {
        feedback->cross_kind = FC_CLICK_CROSS_NONE;
        feedback->cross_elapsed = 0.0f;
    }
}

int fc_click_feedback_cross_frame(const FcClickFeedback* feedback) {
    if (!feedback || feedback->cross_kind == FC_CLICK_CROSS_NONE) return -1;
    int frame = (int)(feedback->cross_elapsed / FC_CLICK_CROSS_FRAME_SECONDS);
    if (frame < 0) frame = 0;
    if (frame >= FC_CLICK_CROSS_FRAME_COUNT)
        frame = FC_CLICK_CROSS_FRAME_COUNT - 1;
    return frame;
}

/* Return the immediate preview while the click is buffered, then the live
 * core route after the next simulation tick accepts it. */
int fc_click_feedback_route(const FcClickFeedback* feedback,
                            const FcState* state,
                            const int** out_x, const int** out_y,
                            int* out_start, int* out_len) {
    if (!feedback || !state || !out_x || !out_y || !out_start || !out_len ||
        !feedback->destination_active) {
        return 0;
    }

    if (feedback->preview_pending) {
        *out_x = feedback->preview_route_x;
        *out_y = feedback->preview_route_y;
        *out_start = 0;
        *out_len = feedback->preview_route_len;
        return feedback->preview_route_len > 0;
    }

    *out_x = state->player.route_x;
    *out_y = state->player.route_y;
    *out_start = state->player.route_idx;
    *out_len = state->player.route_len;
    return state->player.route_idx < state->player.route_len;
}

/* Combat Presentation */

typedef struct FcCombatPresentation FcCombatPresentation;

typedef struct {
    const FcState *state;
    const FcRenderEvents *events;
    const FcVisualScene *scene;
    TerrainMesh *terrain;
    AnimCache *anim_cache;
    const FcPlayerVisualProfile *player_profile;
    float tps;
} FcCombatPresentationContext;

typedef struct {
    FcCombatPresentationContext presentation;
    const FcRenderEntity *entities;
    int entity_count;
    ModelSet *player_models;
    ModelSet *npc_models;
    const FcUiAssets *ui_assets;
    Camera3D camera;
} FcCombatPresentationDrawContext;

#define MAX_HITSPLATS 32
#define MAX_PROJECTILES 16
#define MAX_VISUAL_EFFECTS 32
#define OSRS_HITSPLAT_SECONDS 1.0f
#define OSRS_HEALTHBAR_SECONDS 6.0f

#define PROJ_JAD_MAGIC_LAUNCH 439
#define PROJ_TOK_XIL_SPINE 443
#define PROJ_TOK_XIL_IMPACT 444
#define PROJ_KET_ZEK_FIRE 445
#define PROJ_KET_ZEK_IMPACT 446
#define PROJ_JAD_MAGIC_TRAVEL 448
#define PROJ_JAD_MAGIC_IMPACT 157
#define PROJ_JAD_RANGED_IMPACT 451
#define PROJ_SPOTANIM_MODEL_BASE 0xA2000000u

typedef enum {
    HITSPLAT_DAMAGE = 0,
    HITSPLAT_HEAL = 1,
    HITSPLAT_PRAYER_DRAIN = 2,
} HitsplatKind;

typedef struct {
    int active;
    float world_x;
    float world_y;
    float world_z;
    FcVisualTargetKind actor_kind;
    int actor_slot;
    int overlay_slot;
    int damage;
    int kind;
    float seconds_left;
} Hitsplat;

typedef struct {
    int active;
    float src_x;
    float src_y;
    float src_z;
    float dst_x;
    float dst_y;
    float dst_z;
    float x;
    float y;
    float z;
    float velocity_x;
    float velocity_z;
    float total_time;
    float elapsed;
    float launch_delay;
    int launched;
    FcVisualTargetKind source_kind;
    int source_slot;
    float source_y_offset;
    FcVisualTargetKind target_kind;
    int target_slot;
    float target_y_offset;
    int track_target;
    int attack_style;
    int launch_tick;
    int has_deferred_hitsplat;
    FcVisualTargetKind hitsplat_actor_kind;
    int hitsplat_actor_slot;
    float hitsplat_world_x;
    float hitsplat_world_y;
    float hitsplat_world_z;
    int hitsplat_damage;
    uint32_t spot_id;
    uint32_t impact_spot_id;
    float projectile_angle;
    float projectile_progress;
    AnimModelState *anim_state;
    AnimTrack anim;
} VisualProjectile;

typedef struct {
    int active;
    float x;
    float y;
    float z;
    float total_time;
    float elapsed;
    uint32_t spot_id;
    float yaw_degrees;
    int attached;
    FcVisualTargetKind attached_kind;
    int attached_slot;
    float attached_y_offset;
    FcVisualTargetKind face_kind;
    int face_slot;
    AnimModelState *anim_state;
    AnimTrack anim;
} VisualEffect;

/* Actor overhead sprites: hitsplats and the 30-pixel health bar. */
typedef enum {
    OVERLAY_HITSPLAT_ZERO,
    OVERLAY_HITSPLAT_DAMAGE,
    OVERLAY_HITSPLAT_HEAL,
    OVERLAY_HITSPLAT_PRAYER_DRAIN,
    OVERLAY_HEALTHBAR_FULL,
    OVERLAY_HEALTHBAR_EMPTY,
    OVERLAY_SPRITE_COUNT
} FcOverlaySprite;

static const char *const OVERLAY_SPRITE_PATHS[OVERLAY_SPRITE_COUNT] = {
    "data/sprites/ui/hitsplat_zero.png",
    "data/sprites/ui/hitsplat_damage.png",
    "data/sprites/ui/hitsplat_heal.png",
    "data/sprites/ui/hitsplat_prayer_drain.png",
    "data/sprites/ui/healthbar_full_30.png",
    "data/sprites/ui/healthbar_empty_30.png",
};

struct FcCombatPresentation {
    Hitsplat hitsplats[MAX_HITSPLATS];
    float player_healthbar_timer;
    float npc_healthbar_timers[FC_MAX_NPCS];
    VisualProjectile projectiles[MAX_PROJECTILES];
    VisualEffect effects[MAX_VISUAL_EFFECTS];
    ModelSet *projectile_models;
    SpotAnimSet *spotanims;
    Texture2D overlay_sprites[OVERLAY_SPRITE_COUNT];
};

/* Actor ground height: slightly above the terrain, so actors stand on the
 * flattened cracks. */
static float ground_height(TerrainMesh *terrain, int tile_x, int tile_y) {
    return terrain ? terrain_height_at(terrain, tile_x, tile_y) + 0.1f : 0.0f;
}

/* Ground height between tile centres, interpolated bilinearly. */
static float smooth_ground_height(TerrainMesh *terrain,
                                  float tile_x, float tile_y) {
    int x0 = (int)floorf(tile_x);
    int y0 = (int)floorf(tile_y);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 >= FC_ARENA_WIDTH) x0 = FC_ARENA_WIDTH - 1;
    if (y0 >= FC_ARENA_HEIGHT) y0 = FC_ARENA_HEIGHT - 1;
    int x1 = x0 + 1 < FC_ARENA_WIDTH ? x0 + 1 : x0;
    int y1 = y0 + 1 < FC_ARENA_HEIGHT ? y0 + 1 : y0;
    float tx = tile_x - floorf(tile_x);
    float ty = tile_y - floorf(tile_y);
    float h00 = ground_height(terrain, x0, y0);
    float h10 = ground_height(terrain, x1, y0);
    float h01 = ground_height(terrain, x0, y1);
    float h11 = ground_height(terrain, x1, y1);
    float h0 = h00 + (h10 - h00) * tx;
    float h1 = h01 + (h11 - h01) * tx;
    return h0 + (h1 - h0) * ty;
}

static void free_projectile(VisualProjectile *projectile) {
    if (!projectile) return;
    if (projectile->anim_state)
        anim_model_state_free(projectile->anim_state);
    memset(projectile, 0, sizeof(*projectile));
}

static void free_effect(VisualEffect *effect) {
    if (!effect) return;
    if (effect->anim_state) anim_model_state_free(effect->anim_state);
    memset(effect, 0, sizeof(*effect));
}

static void clear_visuals(FcCombatPresentation *presentation) {
    for (int i = 0; i < MAX_PROJECTILES; i++)
        free_projectile(&presentation->projectiles[i]);
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++)
        free_effect(&presentation->effects[i]);
}

FcCombatPresentation *fc_combat_presentation_create(Texture2D shared_atlas) {
    FcCombatPresentation *presentation = calloc(1, sizeof(*presentation));
    if (!presentation) return NULL;
    presentation->projectile_models = models_load("fc_projectiles.models",
                                                  shared_atlas);
    presentation->spotanims = spotanims_load("fc_spotanims.bin");
    int loaded = 0;
    for (int i = 0; i < OVERLAY_SPRITE_COUNT; i++) {
        Texture2D *sprite = &presentation->overlay_sprites[i];
        *sprite = fc_load_texture_asset(OVERLAY_SPRITE_PATHS[i]);
        if (sprite->id > 0) {
            SetTextureFilter(*sprite, TEXTURE_FILTER_POINT);
            loaded++;
        }
    }
    fprintf(stderr, "Actor overhead sprites loaded: %d/%d\n", loaded,
            OVERLAY_SPRITE_COUNT);
    return presentation;
}

int fc_combat_presentation_ready(
        const FcCombatPresentation *presentation) {
    if (!presentation || !presentation->projectile_models ||
        !presentation->spotanims) {
        return 0;
    }
    for (int i = 0; i < OVERLAY_SPRITE_COUNT; i++) {
        if (presentation->overlay_sprites[i].id == 0) return 0;
    }
    return 1;
}

void fc_combat_presentation_destroy(FcCombatPresentation *presentation) {
    if (!presentation) return;
    clear_visuals(presentation);
    for (int i = 0; i < OVERLAY_SPRITE_COUNT; i++) {
        if (presentation->overlay_sprites[i].id > 0)
            UnloadTexture(presentation->overlay_sprites[i]);
    }
    if (presentation->spotanims) spotanims_free(presentation->spotanims);
    if (presentation->projectile_models)
        models_free(presentation->projectile_models);
    free(presentation);
}

void fc_combat_presentation_reset(FcCombatPresentation *presentation) {
    if (!presentation) return;
    clear_visuals(presentation);
    memset(presentation->hitsplats, 0, sizeof(presentation->hitsplats));
    presentation->player_healthbar_timer = 0.0f;
    memset(presentation->npc_healthbar_timers, 0,
           sizeof(presentation->npc_healthbar_timers));
}

void fc_combat_presentation_clear_npc_healthbar(
    FcCombatPresentation *presentation, int npc_slot) {
    if (!presentation || npc_slot < 0 || npc_slot >= FC_MAX_NPCS) return;
    presentation->npc_healthbar_timers[npc_slot] = 0.0f;
}

static int actor_world_point(const FcCombatPresentationContext *context,
                             FcVisualTargetKind kind, int slot,
                             float *x, float *y, float *z) {
    if (!context || !context->scene || !x || !y || !z) return 0;
    FcVisualPose pose;
    float height;
    if (kind == FC_VISUAL_TARGET_PLAYER && context->scene->player.active) {
        pose = fc_visual_scene_player_pose(context->scene);
        height = 1.5f;
    } else if (kind == FC_VISUAL_TARGET_NPC && slot >= 0 &&
               slot < FC_MAX_NPCS && context->scene->npcs[slot].active) {
        pose = fc_visual_scene_npc_pose(context->scene, slot);
        height = 1.0f + (float)context->scene->npcs[slot].size * 0.3f;
    } else {
        return 0;
    }
    *x = pose.x;
    *z = -pose.y;
    *y = smooth_ground_height(context->terrain, pose.x, pose.y) + height;
    return 1;
}

static int tile_distance(int source_x, int source_y,
                         int target_x, int target_y) {
    int dx = abs(target_x - source_x);
    int dy = abs(target_y - source_y);
    return dx > dy ? dx : dy;
}

static int clamp_int(int value, int lo, int hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

static float animation_client_cycles(const AnimSequence *sequence) {
    if (!sequence || sequence->frame_count == 0) return 0.0f;
    float total = 0.0f;
    for (int i = 0; i < sequence->frame_count; i++)
        total += sequence->frames[i].delay > 0 ? sequence->frames[i].delay : 1;
    return total;
}

/* Starts a spot animation at a point. It shows until its animation ends (30
 * client cycles without one) or the client stops retaining it after
 * retained_cycles, whichever is sooner, so short spot animations never loop. */
static VisualEffect *spawn_effect(FcCombatPresentation *presentation,
                                  const FcCombatPresentationContext *context,
                                  uint32_t spot_id, float x, float y, float z,
                                  float retained_cycles, float yaw_degrees) {
    if (spot_id == 0) return NULL;
    const SpotAnimDef *spot = spotanim_find(presentation->spotanims,
                                            (int)spot_id);
    float cycles = 0.0f;
    if (spot && spot->animation_id >= 0) {
        cycles = animation_client_cycles(anim_get_sequence(
            context->anim_cache, (uint16_t)spot->animation_id));
    }
    if (cycles <= 0.0f) cycles = 30.0f;
    if (cycles > retained_cycles) cycles = retained_cycles;
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++) {
        if (!presentation->effects[i].active) {
            VisualEffect *effect = &presentation->effects[i];
            memset(effect, 0, sizeof(*effect));
            effect->active = 1;
            effect->x = x;
            effect->y = y;
            effect->z = z;
            effect->total_time = cycles / (30.0f * context->tps);
            effect->spot_id = spot_id;
            effect->yaw_degrees = yaw_degrees;
            return effect;
        }
    }
    return NULL;
}

/* Fires a projectile from a source tile at a target footprint, timed in
 * client cycles (30 per game tick): it leaves at launch_cycle and flies
 * through end_cycle, length_adjustment + step_multiplier per tile later. Each
 * end then follows its actor (the target only when tracked), keeping the
 * profile's height as an offset. Also starts the launch spot animation. */
static void launch_projectile(FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    const FcProjectileProfile *profile, int attack_style,
    FcVisualTargetKind source_kind, int source_slot,
    int source_x, int source_y,
    FcVisualTargetKind target_kind, int target_slot,
    int target_x, int target_y, int target_size) {
    if (!profile->travel_spot && !profile->launch_spot && !profile->impact_spot)
        return; /* nothing to show, e.g. unarmed */
    VisualProjectile *projectile = NULL;
    for (int i = 0; i < MAX_PROJECTILES && !projectile; i++) {
        if (!presentation->projectiles[i].active)
            projectile = &presentation->projectiles[i];
    }
    if (!projectile) return;
    free_projectile(projectile);
    projectile->active = 1;
    projectile->spot_id = profile->travel_spot;
    projectile->impact_spot_id = profile->impact_spot;
    projectile->source_kind = source_kind;
    projectile->source_slot = source_slot;
    projectile->target_kind = target_kind;
    projectile->target_slot = target_slot;
    projectile->track_target = !profile->fixed_target;
    projectile->attack_style = attack_style;
    projectile->launch_tick = context->state->tick;
    projectile->projectile_angle = profile->angle;
    projectile->projectile_progress = profile->progress;

    float end_cycle = profile->launch_cycle + profile->length_adjustment +
        profile->step_multiplier *
        (float)tile_distance(source_x, source_y, target_x, target_y);
    if (end_cycle <= profile->launch_cycle)
        end_cycle = profile->launch_cycle + 1.0f;
    float seconds_per_cycle = 1.0f / (30.0f * context->tps);
    float flight = (end_cycle + 1.0f - profile->launch_cycle) *
                   seconds_per_cycle;
    projectile->launch_delay = profile->launch_cycle * seconds_per_cycle;
    projectile->total_time = projectile->launch_delay + flight;

    projectile->src_x = (float)source_x + 0.5f;
    projectile->src_y = ground_height(context->terrain, source_x, source_y) +
                        profile->start_height / 128.0f;
    projectile->src_z = -((float)source_y + 0.5f);
    projectile->dst_x = (float)target_x + (float)target_size * 0.5f;
    projectile->dst_y = ground_height(context->terrain, target_x, target_y) +
                        profile->end_height / 128.0f;
    projectile->dst_z = -((float)target_y + (float)target_size * 0.5f);
    float x;
    float y;
    float z;
    if (actor_world_point(context, source_kind, source_slot, &x, &y, &z)) {
        projectile->source_y_offset = projectile->src_y - y;
        projectile->src_x = x;
        projectile->src_y = y + projectile->source_y_offset;
        projectile->src_z = z;
    }
    if (projectile->track_target &&
        actor_world_point(context, target_kind, target_slot, &x, &y, &z)) {
        projectile->target_y_offset = projectile->dst_y - y;
        projectile->dst_x = x;
        projectile->dst_y = y + projectile->target_y_offset;
        projectile->dst_z = z;
    }
    projectile->x = projectile->src_x;
    projectile->y = projectile->src_y;
    projectile->z = projectile->src_z;
    if (profile->launch_spot != 0) {
        float retain = profile->launch_cycle > 30.0f
            ? profile->launch_cycle : 30.0f;
        float yaw = atan2f(projectile->dst_x - projectile->src_x,
                           projectile->dst_z - projectile->src_z) * RAD2DEG;
        VisualEffect *effect = spawn_effect(presentation, context,
            profile->launch_spot, projectile->src_x, projectile->src_y,
            projectile->src_z, retain, yaw);
        if (effect) {
            effect->attached = 1;
            effect->attached_kind = source_kind;
            effect->attached_slot = source_slot;
            effect->face_kind = target_kind;
            effect->face_slot = target_slot;
            if (actor_world_point(context, source_kind, source_slot,
                                  &x, &y, &z))
                effect->attached_y_offset = effect->y - y;
        }
    }
}

static void show_healthbar(FcCombatPresentation *presentation,
                           FcVisualTargetKind kind, int slot) {
    if (kind == FC_VISUAL_TARGET_PLAYER) {
        presentation->player_healthbar_timer = OSRS_HEALTHBAR_SECONDS;
    } else if (kind == FC_VISUAL_TARGET_NPC &&
               slot >= 0 && slot < FC_MAX_NPCS) {
        presentation->npc_healthbar_timers[slot] = OSRS_HEALTHBAR_SECONDS;
    }
}

static int next_overlay_slot(const FcCombatPresentation *presentation,
                             FcVisualTargetKind kind, int actor_slot) {
    unsigned int used = 0;
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        const Hitsplat *hit = &presentation->hitsplats[i];
        if (hit->active && hit->actor_kind == kind &&
            hit->actor_slot == actor_slot &&
            hit->overlay_slot >= 0 && hit->overlay_slot < 4)
            used |= 1u << hit->overlay_slot;
    }
    for (int i = 0; i < 4; i++)
        if ((used & (1u << i)) == 0) return i;
    return 0;
}

static void spawn_status_splat(FcCombatPresentation *presentation,
    FcVisualTargetKind kind, int actor_slot,
    float x, float y, float z, int damage, HitsplatKind splat_kind) {
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        if (!presentation->hitsplats[i].active) {
            Hitsplat *hit = &presentation->hitsplats[i];
            hit->active = 1;
            hit->world_x = x;
            hit->world_y = y;
            hit->world_z = z;
            hit->actor_kind = kind;
            hit->actor_slot = actor_slot;
            hit->overlay_slot = next_overlay_slot(presentation, kind, actor_slot);
            hit->damage = damage;
            hit->kind = splat_kind;
            hit->seconds_left = OSRS_HITSPLAT_SECONDS;
            if (splat_kind != HITSPLAT_PRAYER_DRAIN)
                show_healthbar(presentation, kind, actor_slot);
            return;
        }
    }
}

static void spawn_hitsplat(FcCombatPresentation *presentation,
    FcVisualTargetKind kind, int actor_slot,
    float x, float y, float z, int damage) {
    spawn_status_splat(presentation, kind, actor_slot, x, y, z,
                       damage, HITSPLAT_DAMAGE);
}

static int defer_hitsplat(FcCombatPresentation *presentation,
                          const FcCombatPresentationContext *context,
                          const FcRenderHit *hit,
                          FcVisualTargetKind kind, int actor_slot,
                          float x, float y, float z) {
    if (!hit || hit->attack_style == ATTACK_MELEE) return 0;
    FcVisualTargetKind source_kind;
    FcVisualTargetKind target_kind;
    int source_slot;
    int target_slot;
    if (hit->target_entity_type == ENTITY_PLAYER) {
        if (hit->source_npc_slot < 0) return 0;
        source_kind = FC_VISUAL_TARGET_NPC;
        source_slot = hit->source_npc_slot;
        target_kind = FC_VISUAL_TARGET_PLAYER;
        target_slot = 0;
    } else if (hit->target_entity_type == ENTITY_NPC) {
        if (hit->target_npc_slot < 0) return 0;
        source_kind = FC_VISUAL_TARGET_PLAYER;
        source_slot = 0;
        target_kind = FC_VISUAL_TARGET_NPC;
        target_slot = hit->target_npc_slot;
    } else {
        return 0;
    }
    VisualProjectile *match = NULL;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        VisualProjectile *projectile = &presentation->projectiles[i];
        if (!projectile->active || projectile->has_deferred_hitsplat ||
            projectile->launch_tick >= context->state->tick ||
            projectile->source_kind != source_kind ||
            projectile->source_slot != source_slot ||
            projectile->target_kind != target_kind ||
            projectile->target_slot != target_slot ||
            projectile->attack_style != hit->attack_style) continue;
        if (!match || projectile->elapsed > match->elapsed) match = projectile;
    }
    if (!match) return 0;
    match->has_deferred_hitsplat = 1;
    match->hitsplat_actor_kind = kind;
    match->hitsplat_actor_slot = actor_slot;
    match->hitsplat_world_x = x;
    match->hitsplat_world_y = y;
    match->hitsplat_world_z = z;
    match->hitsplat_damage = hit->damage;
    return 1;
}

static int deferred_damage(const FcCombatPresentation *presentation,
                           FcVisualTargetKind kind, int actor_slot) {
    int damage = 0;
    if (!presentation) return 0;
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        const VisualProjectile *projectile = &presentation->projectiles[i];
        if (projectile->active && projectile->has_deferred_hitsplat &&
            projectile->hitsplat_actor_kind == kind &&
            projectile->hitsplat_actor_slot == actor_slot)
            damage += projectile->hitsplat_damage;
    }
    return damage;
}

int fc_combat_presentation_npc_death_deferred(
    const FcCombatPresentation *presentation,
    const FcState *state,
    int npc_slot) {
    return presentation && state && npc_slot >= 0 && npc_slot < FC_MAX_NPCS &&
        state->npcs[npc_slot].is_dead &&
        deferred_damage(presentation, FC_VISUAL_TARGET_NPC, npc_slot) > 0;
}

void fc_combat_presentation_deferred_deaths(
    const FcCombatPresentation *presentation,
    const FcState *state,
    unsigned char deferred_deaths[FC_MAX_NPCS]) {
    if (!deferred_deaths) return;
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        deferred_deaths[i] = (unsigned char)
            fc_combat_presentation_npc_death_deferred(presentation, state, i);
    }
}

static void ingest_player_attack(FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context) {
    const FcRenderEvents *events = context->events;
    launch_projectile(presentation, context,
        &context->player_profile->projectile, ATTACK_RANGED,
        FC_VISUAL_TARGET_PLAYER, 0,
        events->player_attack_source_x, events->player_attack_source_y,
        FC_VISUAL_TARGET_NPC, events->player_attack_target_npc_slot,
        events->player_attack_target_x, events->player_attack_target_y,
        events->player_attack_target_size);
}

/* NPC projectiles by attacker type and attack style; melee attacks have
 * none. Profile columns follow FcProjectileProfile. */
static const struct {
    int npc_type;
    int attack_style;
    FcProjectileProfile profile;
} NPC_PROJECTILES[] = {
    {NPC_TOK_XIL, ATTACK_RANGED,
     {PROJ_TOK_XIL_SPINE, 0, PROJ_TOK_XIL_IMPACT,
      296.0f, 40.0f, 32.0f, 16.0f, 0.0f, 0.0f, 5.0f, 0}},
    {NPC_KET_ZEK, ATTACK_MAGIC,
     {PROJ_KET_ZEK_FIRE, 0, PROJ_KET_ZEK_IMPACT,
      192.0f, 40.0f, 28.0f, 16.0f, 8.0f, 0.0f, 8.0f, 0}},
    {NPC_TZTOK_JAD, ATTACK_MAGIC,
     {PROJ_JAD_MAGIC_TRAVEL, PROJ_JAD_MAGIC_LAUNCH, PROJ_JAD_MAGIC_IMPACT,
      172.0f, 124.0f, 41.0f, 16.0f, 0.0f, 64.0f, 5.0f, 0}},
    /* Jad's ranged attack has no flight model: only its impact shows, on
     * the player's tile at launch, when the flight ends at cycle 60. */
    {NPC_TZTOK_JAD, ATTACK_RANGED,
     {0, 0, PROJ_JAD_RANGED_IMPACT,
      768.0f, 52.0f, 0.0f, 0.0f, 60.0f, 0.0f, 0.0f, 1}},
};

static void ingest_npc_attack(FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    const FcRenderNpcAttack *attack) {
    if (!attack->hit_queued) return;
    const FcProjectileProfile *profile = NULL;
    int rows = (int)(sizeof(NPC_PROJECTILES) / sizeof(NPC_PROJECTILES[0]));
    for (int i = 0; i < rows && !profile; i++) {
        if (NPC_PROJECTILES[i].npc_type == attack->npc_type &&
            NPC_PROJECTILES[i].attack_style == attack->attack_style)
            profile = &NPC_PROJECTILES[i].profile;
    }
    if (!profile) return;
    /* Large NPCs fire from their footprint tile nearest the target. */
    int last_x = attack->source_x + attack->source_size - 1;
    int last_y = attack->source_y + attack->source_size - 1;
    launch_projectile(presentation, context, profile, attack->attack_style,
        FC_VISUAL_TARGET_NPC, attack->npc_slot,
        clamp_int(attack->target_x, attack->source_x, last_x),
        clamp_int(attack->target_y, attack->source_y, last_y),
        FC_VISUAL_TARGET_PLAYER, 0, attack->target_x, attack->target_y, 1);
}

void fc_combat_presentation_ingest_tick(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context) {
    if (!presentation || !context || !context->state || !context->events ||
        !context->player_profile) return;
    const FcState *state = context->state;
    float player_ground = ground_height(context->terrain,
        clamp_int(state->player.x, 0, FC_ARENA_WIDTH - 1),
        clamp_int(state->player.y, 0, FC_ARENA_HEIGHT - 1));
    float player_world_x = (float)state->player.x + 0.5f;
    float player_world_z = -((float)state->player.y + 0.5f);
    if (state->tz_kih_prayer_drain_this_tick > 0) {
        spawn_status_splat(presentation, FC_VISUAL_TARGET_PLAYER, 0,
            player_world_x + 0.3f, player_ground + 3.0f, player_world_z,
            state->tz_kih_prayer_drain_this_tick, HITSPLAT_PRAYER_DRAIN);
    }
    if (context->events->player_attack_fired)
        ingest_player_attack(presentation, context);
    for (int i = 0; i < context->events->hit_count; i++) {
        const FcRenderHit *hit = &context->events->hits[i];
        if (hit->target_entity_type == ENTITY_PLAYER) {
            if (!defer_hitsplat(presentation, context, hit,
                    FC_VISUAL_TARGET_PLAYER, 0, player_world_x,
                    player_ground + 2.5f, player_world_z)) {
                spawn_hitsplat(presentation, FC_VISUAL_TARGET_PLAYER, 0,
                    player_world_x, player_ground + 2.5f,
                    player_world_z, hit->damage);
            }
        } else if (hit->target_entity_type == ENTITY_NPC &&
                   hit->target_npc_slot >= 0 &&
                   hit->target_npc_slot < FC_MAX_NPCS) {
            const FcNpc *target = &state->npcs[hit->target_npc_slot];
            float ground = ground_height(context->terrain, target->x, target->y);
            float x = (float)target->x + (float)target->size * 0.5f;
            float z = -((float)target->y + (float)target->size * 0.5f);
            float y = ground + 1.0f + (float)target->size * 0.5f;
            if (!defer_hitsplat(presentation, context, hit,
                    FC_VISUAL_TARGET_NPC, hit->target_npc_slot, x, y, z)) {
                spawn_hitsplat(presentation, FC_VISUAL_TARGET_NPC,
                    hit->target_npc_slot, x, y, z, hit->damage);
            }
        }
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc *npc = &state->npcs[i];
        if (npc->healing_received_this_tick > 0) {
            float ground = ground_height(context->terrain, npc->x, npc->y);
            float x = (float)npc->x + (float)npc->size * 0.5f;
            float z = -((float)npc->y + (float)npc->size * 0.5f);
            float y = ground + 1.4f + (float)npc->size * 0.5f;
            spawn_status_splat(presentation, FC_VISUAL_TARGET_NPC, i,
                x, y, z, npc->healing_received_this_tick, HITSPLAT_HEAL);
        }
    }
    for (int i = 0; i < context->events->npc_attack_count; i++) {
        const FcRenderNpcAttack *attack = &context->events->npc_attacks[i];
        if (attack->npc_slot >= 0 && attack->npc_slot < FC_MAX_NPCS)
            ingest_npc_attack(presentation, context, attack);
    }
}

static void refresh_actor_points(const FcCombatPresentationContext *context,
                                 VisualProjectile *projectile,
                                 int refresh_source) {
    float x;
    float y;
    float z;
    if (refresh_source && actor_world_point(context, projectile->source_kind,
            projectile->source_slot, &x, &y, &z)) {
        projectile->src_x = x;
        projectile->src_y = y + projectile->source_y_offset;
        projectile->src_z = z;
    }
    if (projectile->track_target && actor_world_point(context,
            projectile->target_kind, projectile->target_slot, &x, &y, &z)) {
        projectile->dst_x = x;
        projectile->dst_y = y + projectile->target_y_offset;
        projectile->dst_z = z;
    }
}

static int update_projectile(const FcCombatPresentationContext *context,
                             VisualProjectile *projectile, float dt) {
    float end = projectile->elapsed + dt;
    if (end > projectile->total_time) end = projectile->total_time;
    refresh_actor_points(context, projectile, !projectile->launched);
    if (end < projectile->launch_delay) {
        projectile->x = projectile->src_x;
        projectile->y = projectile->src_y;
        projectile->z = projectile->src_z;
        projectile->elapsed = end;
        return 0;
    }
    projectile->launched = 1;
    float duration = projectile->total_time - projectile->launch_delay;
    if (duration < 0.001f) duration = 0.001f;
    /* Sample the client projectile curve at an absolute point t of the
     * flight (0..duration): the start moves `progress` toward the target,
     * leaves at `angle` and accelerates to the target height. The ends may
     * move every frame, reproducing the client's homing without
     * frame-rate-dependent integration. */
    float dx = projectile->dst_x - projectile->src_x;
    float dz = projectile->dst_z - projectile->src_z;
    float horizontal = sqrtf(dx * dx + dz * dz);
    float direction_x = 0.0f;
    float direction_z = 1.0f;
    if (horizontal > 0.00001f) {
        direction_x = dx / horizontal;
        direction_z = dz / horizontal;
    }
    float progress = projectile->projectile_progress / 128.0f;
    float source_x = projectile->src_x + direction_x * progress;
    float source_z = projectile->src_z + direction_z * progress;
    float velocity_x = (projectile->dst_x - source_x) / duration;
    float velocity_z = (projectile->dst_z - source_z) / duration;
    float horizontal_speed = sqrtf(
        velocity_x * velocity_x + velocity_z * velocity_z);
    float velocity_y = horizontal_speed *
        tanf(projectile->projectile_angle * (3.14159265358979323846f / 128.0f));
    float acceleration_y = 2.0f *
        (projectile->dst_y - projectile->src_y - velocity_y * duration) /
        (duration * duration);
    float t = end - projectile->launch_delay;
    projectile->x = source_x + velocity_x * t;
    projectile->y = projectile->src_y + velocity_y * t +
                    0.5f * acceleration_y * t * t;
    projectile->z = source_z + velocity_z * t;
    projectile->velocity_x = velocity_x;
    projectile->velocity_z = velocity_z;
    projectile->elapsed = end;
    if (end >= projectile->total_time) {
        projectile->x = projectile->dst_x;
        projectile->y = projectile->dst_y;
        projectile->z = projectile->dst_z;
        return 1;
    }
    return 0;
}

void fc_combat_presentation_update(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    float dt) {
    if (!presentation || !context || dt <= 0.0f) return;
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        if (presentation->hitsplats[i].active) {
            presentation->hitsplats[i].seconds_left -= dt;
            if (presentation->hitsplats[i].seconds_left <= 0.0f)
                presentation->hitsplats[i].active = 0;
        }
    }
    tick_down(&presentation->player_healthbar_timer, dt);
    for (int i = 0; i < FC_MAX_NPCS; i++)
        tick_down(&presentation->npc_healthbar_timers[i], dt);
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        VisualProjectile *projectile = &presentation->projectiles[i];
        if (!projectile->active || !update_projectile(context, projectile, dt))
            continue;
        spawn_effect(presentation, context, projectile->impact_spot_id,
            projectile->x, projectile->y, projectile->z, 90.0f, 0.0f);
        if (projectile->has_deferred_hitsplat) {
            spawn_hitsplat(presentation, projectile->hitsplat_actor_kind,
                projectile->hitsplat_actor_slot, projectile->hitsplat_world_x,
                projectile->hitsplat_world_y, projectile->hitsplat_world_z,
                projectile->hitsplat_damage);
        }
        free_projectile(projectile);
    }
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++) {
        VisualEffect *effect = &presentation->effects[i];
        if (effect->active) {
            effect->elapsed += dt;
            if (effect->elapsed >= effect->total_time) free_effect(effect);
        }
    }
}

/* A spot animation's definition and model. Both ship for every spot id the
 * projectile profiles name; the viewer requires those files at startup. */
static ModelEntry *spot_model(const FcCombatPresentation *presentation,
                              uint32_t spot_id, const SpotAnimDef **spot) {
    *spot = spotanim_find(presentation->spotanims, (int)spot_id);
    if (!*spot) return NULL;
    return model_find(presentation->projectile_models,
                      PROJ_SPOTANIM_MODEL_BASE + spot_id);
}

/* Draws a spot animation's model at its current animation frame, resized by
 * its definition and turned by axis/angle (degrees). */
static void draw_spot_model(const FcCombatPresentationContext *context,
                            ModelEntry *entry, const SpotAnimDef *spot,
                            AnimModelState **anim_state, AnimTrack *anim,
                            Vector3 position, Vector3 axis, float angle,
                            float dt) {
    float scale_xy = spot->resize_xy > 0
        ? (float)spot->resize_xy / 128.0f : 1.0f;
    float scale_z = spot->resize_z > 0
        ? (float)spot->resize_z / 128.0f : 1.0f;
    if (spot->animation_id >= 0) {
        fc_model_animation_update(entry, context->anim_cache, anim_state,
            anim, spot->animation_id, dt, 0.0f);
    }
    rlDisableBackfaceCulling();
    DrawModelEx(entry->model, position, axis, angle,
                (Vector3){scale_xy, scale_z, scale_xy}, WHITE);
    rlEnableBackfaceCulling();
}

void fc_combat_presentation_draw_world(
    FcCombatPresentation *presentation,
    const FcCombatPresentationContext *context,
    float dt) {
    if (!presentation || !context) return;
    float animation_dt = fc_actor_animation_scaled_dt(context->tps, dt);
    for (int i = 0; i < MAX_PROJECTILES; i++) {
        VisualProjectile *projectile = &presentation->projectiles[i];
        if (!projectile->active || projectile->spot_id == 0 ||
            !projectile->launched) continue;
        const SpotAnimDef *spot = NULL;
        ModelEntry *entry = spot_model(presentation, projectile->spot_id,
                                       &spot);
        if (!entry) continue;
        /* Face along the flight. The quaternion's axis and angle round
         * slightly differently from a plain yaw; keep them. */
        float angle = atan2f(projectile->velocity_x,
                             projectile->velocity_z) * RAD2DEG;
        angle += (float)spot->rotation;
        Quaternion yaw = QuaternionFromAxisAngle(
            (Vector3){0, 1, 0}, angle * DEG2RAD);
        Vector3 axis = {0, 1, 0};
        float rotation_angle = 0.0f;
        QuaternionToAxisAngle(yaw, &axis, &rotation_angle);
        draw_spot_model(context, entry, spot, &projectile->anim_state,
            &projectile->anim,
            (Vector3){projectile->x, projectile->y, projectile->z},
            axis, rotation_angle * RAD2DEG, animation_dt);
    }
    for (int i = 0; i < MAX_VISUAL_EFFECTS; i++) {
        VisualEffect *effect = &presentation->effects[i];
        if (!effect->active) continue;
        const SpotAnimDef *spot = NULL;
        ModelEntry *entry = spot_model(presentation, effect->spot_id, &spot);
        if (!entry) continue;
        float x = effect->x;
        float y = effect->y;
        float z = effect->z;
        if (effect->attached) {
            float actor_x;
            float actor_y;
            float actor_z;
            if (actor_world_point(context, effect->attached_kind,
                    effect->attached_slot, &actor_x, &actor_y, &actor_z)) {
                x = actor_x;
                y = actor_y + effect->attached_y_offset;
                z = actor_z;
            }
        }
        float yaw = effect->yaw_degrees;
        if (effect->face_kind != FC_VISUAL_TARGET_NONE) {
            float target_x;
            float target_y;
            float target_z;
            if (actor_world_point(context, effect->face_kind,
                    effect->face_slot, &target_x, &target_y, &target_z)) {
                (void)target_y;
                yaw = atan2f(target_x - x, target_z - z) * RAD2DEG;
            }
        }
        draw_spot_model(context, entry, spot, &effect->anim_state,
            &effect->anim, (Vector3){x, y, z}, (Vector3){0, 1, 0},
            yaw + (float)spot->rotation, animation_dt);
    }
}

static const FcRenderEntity *find_actor(
    const FcCombatPresentationDrawContext *context,
    int actor_kind, int actor_slot) {
    for (int i = 0; i < context->entity_count; i++) {
        const FcRenderEntity *entity = &context->entities[i];
        if (actor_kind == FC_VISUAL_TARGET_PLAYER &&
            entity->entity_type == ENTITY_PLAYER) return entity;
        if (actor_kind == FC_VISUAL_TARGET_NPC &&
            entity->entity_type == ENTITY_NPC &&
            entity->npc_slot == actor_slot) return entity;
    }
    return NULL;
}

static FcVisualPose entity_pose(const FcVisualScene *scene,
                                const FcRenderEntity *entity) {
    return entity->entity_type == ENTITY_PLAYER
        ? fc_visual_scene_player_pose(scene)
        : fc_visual_scene_npc_pose(scene, entity->npc_slot);
}

static float entity_model_top(const FcCombatPresentationDrawContext *context,
                              const FcRenderEntity *entity) {
    Model *model = NULL;
    if (entity->entity_type == ENTITY_PLAYER) {
        ModelEntry *entry = fc_actor_player_model_entry(
            context->player_models);
        if (entry) model = &entry->model;
    } else if (context->npc_models) {
        ModelEntry *entry = npc_model_find(context->npc_models,
                                           entity->npc_type);
        if (entry) model = &entry->model;
    }
    if (model) {
        BoundingBox bounds = GetModelBoundingBox(*model);
        if (bounds.max.y > 0.1f && bounds.max.y < 20.0f) return bounds.max.y;
    }
    return entity->entity_type == ENTITY_PLAYER
        ? 2.0f : 1.3f + (float)entity->size * 0.5f;
}

static Vector3 overlay_anchor(const FcCombatPresentationDrawContext *context,
                              const FcRenderEntity *entity,
                              float height_fraction, float extra_height) {
    FcVisualPose pose = entity_pose(context->presentation.scene, entity);
    return (Vector3){
        pose.x,
        smooth_ground_height(context->presentation.terrain, pose.x, pose.y) +
            entity_model_top(context, entity) * height_fraction + extra_height,
        -pose.y,
    };
}

static Texture2D hitsplat_texture(const FcCombatPresentation *presentation,
                                  const Hitsplat *hit) {
    const Texture2D *sprites = presentation->overlay_sprites;
    if (hit->kind == HITSPLAT_HEAL) return sprites[OVERLAY_HITSPLAT_HEAL];
    if (hit->kind == HITSPLAT_PRAYER_DRAIN)
        return sprites[OVERLAY_HITSPLAT_PRAYER_DRAIN];
    return hit->damage > 0 ? sprites[OVERLAY_HITSPLAT_DAMAGE]
                           : sprites[OVERLAY_HITSPLAT_ZERO];
}

void fc_combat_presentation_draw_healthbars(
    const FcCombatPresentation *presentation,
    const FcCombatPresentationDrawContext *context) {
    if (!presentation || !context || !context->entities ||
        !context->presentation.scene) return;
    for (int i = 0; i < context->entity_count; i++) {
        const FcRenderEntity *entity = &context->entities[i];
        float timer = entity->entity_type == ENTITY_PLAYER
            ? presentation->player_healthbar_timer
            : entity->npc_slot >= 0 && entity->npc_slot < FC_MAX_NPCS
                ? presentation->npc_healthbar_timers[entity->npc_slot] : 0.0f;
        if (timer <= 0.0f || entity->max_hp <= 0) continue;
        Vector2 screen = GetWorldToScreen(
            overlay_anchor(context, entity, 1.0f, 0.12f), context->camera);
        if (screen.x < -40.0f || screen.x > GetScreenWidth() + 40.0f ||
            screen.y < -20.0f || screen.y > GetScreenHeight() + 20.0f) continue;
        FcVisualTargetKind kind = entity->entity_type == ENTITY_PLAYER
            ? FC_VISUAL_TARGET_PLAYER : FC_VISUAL_TARGET_NPC;
        int slot = entity->entity_type == ENTITY_PLAYER ? 0 : entity->npc_slot;
        int visible_hp = entity->current_hp + deferred_damage(presentation,
                                                               kind, slot);
        if (visible_hp > entity->max_hp) visible_hp = entity->max_hp;
        int fill = visible_hp * 30 / entity->max_hp;
        if (fill < 0) fill = 0;
        if (fill > 30) fill = 30;
        int x = (int)roundf(screen.x) - 15;
        int y = (int)roundf(screen.y) - 3;
        DrawTexture(presentation->overlay_sprites[OVERLAY_HEALTHBAR_EMPTY],
                    x, y, WHITE);
        if (fill > 0) {
            Rectangle source = {0.0f, 0.0f, (float)fill, 5.0f};
            DrawTextureRec(presentation->overlay_sprites[OVERLAY_HEALTHBAR_FULL],
                           source, (Vector2){(float)x, (float)y}, WHITE);
        }
    }
}

void fc_combat_presentation_draw_hitsplats(
    const FcCombatPresentation *presentation,
    const FcCombatPresentationDrawContext *context) {
    if (!presentation || !context || !context->entities ||
        !context->presentation.scene) return;
    static const int slot_x[4] = {0, 0, -15, 15};
    static const int slot_y[4] = {0, -20, -10, -10};
    for (int i = 0; i < MAX_HITSPLATS; i++) {
        const Hitsplat *hit = &presentation->hitsplats[i];
        if (!hit->active) continue;
        Vector3 world = {hit->world_x, hit->world_y, hit->world_z};
        const FcRenderEntity *entity = find_actor(context, hit->actor_kind,
                                                   hit->actor_slot);
        if (entity) world = overlay_anchor(context, entity, 0.5f, 0.0f);
        Vector2 screen = GetWorldToScreen(world, context->camera);
        if (screen.x < -50.0f || screen.x > GetScreenWidth() + 50.0f ||
            screen.y < -50.0f || screen.y > GetScreenHeight() + 50.0f) continue;
        int slot = hit->overlay_slot;
        if (slot < 0 || slot >= 4) slot = 0;
        int center_x = (int)roundf(screen.x) + slot_x[slot];
        int center_y = (int)roundf(screen.y) + slot_y[slot];
        Texture2D texture = hitsplat_texture(presentation, hit);
        if (texture.id > 0)
            DrawTexture(texture, center_x - 12, center_y - 12, WHITE);
        int value = hit->kind == HITSPLAT_DAMAGE
            ? hit->damage / 10 : (hit->damage + 9) / 10;
        char text[16];
        snprintf(text, sizeof(text), "%d", value);
        const float font_size = 12.0f;
        Font font = fc_ui_font_for_size(context->ui_assets, font_size);
        Vector2 measured = MeasureTextEx(font, text, font_size, 0.0f);
        float text_x = floorf((float)center_x - 1.0f - measured.x * 0.5f);
        float text_y = floorf((float)center_y - 6.0f);
        fc_ui_draw_font_shadow(font, text, (Vector2){text_x, text_y},
                               font_size, WHITE);
    }
}

/* Debug Overlay */
/*
 * Viewer debug tooling and overlays.
 *
 * All functions are read-only — they never modify FcState or ViewerState.
 *
 * Two overlay entry points are called from the viewer loop below:
 *   debug_overlay_3d()     — inside BeginMode3D/EndMode3D (collision tiles)
 *   debug_overlay_screen() — after EndMode3D (LOS, path and range overlays)
 *
 * O or D toggles all overlays; Shift+O cycles through them one at a time.
 */

#define DBG_COLLISION (1 << 0)
#define DBG_LOS (1 << 1)
#define DBG_PATH (1 << 2)
#define DBG_RANGE (1 << 3)
#define DBG_ALL (DBG_COLLISION | DBG_LOS | DBG_PATH | DBG_RANGE)

/* ======================================================================== */
/* Event log ring buffer                                                    */
/* ======================================================================== */

#define DBG_LOG_MAX_ENTRIES  128
#define DBG_LOG_MAX_MSG      80

typedef struct {
    char entries[DBG_LOG_MAX_ENTRIES][DBG_LOG_MAX_MSG];
    int tick[DBG_LOG_MAX_ENTRIES];      /* game tick when event occurred */
    Color color[DBG_LOG_MAX_ENTRIES];   /* display color per entry */
    int head;                           /* next write position */
    int count;                          /* total entries (capped at MAX) */
} DbgEventLog;

static DbgEventLog g_dbg_log = {0};

void dbg_log_clear(void) {
    g_dbg_log.head = 0;
    g_dbg_log.count = 0;
}

static void dbg_log_event(int game_tick, Color c, const char* fmt, ...) {
    int idx = g_dbg_log.head;
    g_dbg_log.tick[idx] = game_tick;
    g_dbg_log.color[idx] = c;

    va_list args;
    va_start(args, fmt);
    vsnprintf(g_dbg_log.entries[idx], DBG_LOG_MAX_MSG, fmt, args);
    va_end(args);

    g_dbg_log.head = (g_dbg_log.head + 1) % DBG_LOG_MAX_ENTRIES;
    if (g_dbg_log.count < DBG_LOG_MAX_ENTRIES) g_dbg_log.count++;
}

/* Short NPC names for the event log and the console's target list. */
static const char* npc_short_name(int npc_type) {
    static const char* const names[NPC_TYPE_COUNT] = {
        "?", "Tz-Kih", "Tz-Kek", "Kek-Sm", "Tok-Xil",
        "MejKot", "Ket-Zek", "Jad", "HurKot"
    };
    return npc_type > NPC_NONE && npc_type < NPC_TYPE_COUNT
        ? names[npc_type] : "?";
}

/* Call once per tick to auto-generate events from state changes.
 * Reads current state and emits relevant log entries. */
void dbg_log_tick(const FcState* state) {
    int t = state->tick;
    const FcPlayer* p = &state->player;

    /* Player took damage */
    if (p->damage_taken_this_tick > 0) {
        dbg_log_event(t, CLITERAL(Color){255,120,120,255},
                      "Player took %d damage", p->damage_taken_this_tick / 10);
    }
    if (state->tz_kih_prayer_drain_this_tick > 0) {
        dbg_log_event(t, CLITERAL(Color){100,190,255,255},
                      "Tz-Kih drained %d Prayer",
                      (state->tz_kih_prayer_drain_this_tick + 9) / 10);
    }

    /* Player ate food */
    if (p->food_eaten_this_tick) {
        dbg_log_event(t, CLITERAL(Color){180,255,180,255},
                      "Ate shark (+20 HP) [%d left]", p->sharks_remaining);
    }

    /* Player drank potion */
    if (p->potion_used_this_tick) {
        dbg_log_event(t, CLITERAL(Color){120,180,255,255},
                      "Drank ppot (+17 pray) [%d doses left]", p->prayer_doses_remaining);
    }

    /* Player switched prayer */
    if (p->prayer_changed_this_tick) {
        static const char* pray_names[] = {"OFF","Prot Melee","Prot Range","Prot Magic"};
        dbg_log_event(t, CLITERAL(Color){255,255,100,255},
                      "Prayer -> %s", pray_names[p->prayer]);
    }

    /* Player attack attempt */
    if (state->attack_attempt_this_tick && p->attack_target_idx >= 0) {
        const FcNpc* tgt = &state->npcs[p->attack_target_idx];
        dbg_log_event(t, CLITERAL(Color){200,200,200,255},
                      "Attacked %s [%d]", npc_short_name(tgt->npc_type),
                      p->attack_target_idx);
    }

    /* NPC events */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];

        const char* nm = npc_short_name(n->npc_type);

        /* NPC damage taken */
        if (n->damage_taken_this_tick > 0) {
            dbg_log_event(t, CLITERAL(Color){255,200,100,255},
                          "%s[%d] took %d dmg (%d HP left)",
                          nm, i, n->damage_taken_this_tick / 10, n->current_hp / 10);
        }
        if (n->healing_received_this_tick > 0) {
            dbg_log_event(t, CLITERAL(Color){120,255,140,255},
                          "%s[%d] healed %d HP",
                          nm, i, (n->healing_received_this_tick + 9) / 10);
        }

        /* NPC died */
        if (n->died_this_tick) {
            dbg_log_event(t, CLITERAL(Color){255,80,80,255},
                          "%s[%d] KILLED", nm, i);
        }

    }

    /* Movement — log new route destination (walk-to-tile or click-to-move) */
    {
        static int prev_route_dest_x = -1, prev_route_dest_y = -1;
        if (p->route_len > 0 && p->route_idx < p->route_len) {
            int dx = p->route_x[p->route_len - 1];
            int dy = p->route_y[p->route_len - 1];
            if (dx != prev_route_dest_x || dy != prev_route_dest_y) {
                dbg_log_event(t, CLITERAL(Color){180,220,255,255},
                              "Move to tile (%d,%d) [%d steps]", dx, dy, p->route_len - p->route_idx);
                prev_route_dest_x = dx;
                prev_route_dest_y = dy;
            }
        } else {
            prev_route_dest_x = -1;
            prev_route_dest_y = -1;
        }
    }

    /* Wave events */
    if (state->wave_just_cleared) {
        dbg_log_event(t, CLITERAL(Color){100,255,255,255},
                      "Wave %d CLEARED - advancing", state->current_wave - 1);
    }

    /* Jad killed */
    if (state->jad_killed) {
        dbg_log_event(t, CLITERAL(Color){255,215,0,255}, "*** JAD DEFEATED ***");
    }

    /* Player death */
    if (state->terminal == TERMINAL_PLAYER_DEATH && p->current_hp <= 0) {
        dbg_log_event(t, CLITERAL(Color){255,0,0,255}, "*** PLAYER DIED ***");
    }

    /* Cave complete */
    if (state->terminal == TERMINAL_CAVE_COMPLETE) {
        dbg_log_event(t, CLITERAL(Color){0,255,0,255}, "*** CAVE COMPLETE - FIRE CAPE ***");
    }
}

/* Colors */
#define DBG_COL_WALK     CLITERAL(Color){  30, 180,  30, 40 }
#define DBG_COL_BLOCK    CLITERAL(Color){ 200,  30,  30, 60 }
#define DBG_COL_LOS_OK   CLITERAL(Color){  30, 255,  30, 200 }
#define DBG_COL_LOS_FAIL CLITERAL(Color){ 255,  30,  30, 200 }
#define DBG_COL_PATH     CLITERAL(Color){ 255, 255,   0, 160 }
#define DBG_COL_RANGE    CLITERAL(Color){ 100, 200, 255, 120 }
#define DBG_COL_LABEL    CLITERAL(Color){ 200, 200, 200, 255 }
#define DBG_COL_VALUE    CLITERAL(Color){ 255, 255, 100, 255 }
#define DBG_COL_GOOD     CLITERAL(Color){ 100, 255, 100, 255 }
#define DBG_COL_BAD      CLITERAL(Color){ 255, 100, 100, 255 }
#define DBG_COL_DIM      CLITERAL(Color){ 120, 120, 120, 255 }

static const char* dbg_basename(const char* path) {
    const char* slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static Color dbg_reward_color(float value) {
    if (value > 0.0001f) return DBG_COL_GOOD;
    if (value < -0.0001f) return DBG_COL_BAD;
    return DBG_COL_DIM;
}

/* ======================================================================== */
/* A. Collision / LOS / Path / Range overlays (3D)                          */
/* ======================================================================== */

/* Draw walkable/blocked tile overlay */
static void dbg_draw_collision(const FcState* state) {
    for (int tx = 0; tx < FC_ARENA_WIDTH; tx++) {
        for (int ty = 0; ty < FC_ARENA_HEIGHT; ty++) {
            Color c = state->walkable[tx][ty] ? DBG_COL_WALK : DBG_COL_BLOCK;
            DrawCube((Vector3){tx + 0.5f, 0.02f, -(ty + 0.5f)}, 0.9f, 0.02f, 0.9f, c);
        }
    }
}

/* Master 3D overlay — called inside BeginMode3D/EndMode3D.
 * Only collision tiles use 3D (they work with depth test disabled). */
void debug_overlay_3d(const FcState* state, int dbg_flags) {
    rlDisableDepthTest();
    if (dbg_flags & DBG_COLLISION) dbg_draw_collision(state);
    rlEnableDepthTest();
}

/* ======================================================================== */
/* 2D screen-space overlays (LOS, path, range) — drawn after EndMode3D     */
/* Uses GetWorldToScreen() projection so nothing can occlude them.          */
/* ======================================================================== */

static Vector2 dbg_tile_to_screen(int tx, int ty, Camera3D cam) {
    Vector3 w = { tx + 0.5f, 0.5f, -(ty + 0.5f) };
    return GetWorldToScreen(w, cam);
}

/* LOS rays — 2D projected lines from player to NPCs */
static void dbg_draw_los_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    Vector2 ps = dbg_tile_to_screen(p->x, p->y, cam);

    for (int i = 0; i < FC_MAX_NPCS; i++) {
        const FcNpc* n = &state->npcs[i];
        if (!n->active || n->is_dead) continue;

        int ncx = n->x + n->size / 2;
        int ncy = n->y + n->size / 2;
        Vector2 ns = dbg_tile_to_screen(ncx, ncy, cam);

        int has_los = fc_has_los_between_areas(
            p->x, p->y, 1, n->x, n->y, n->size, state->los_flags);
        Color c = has_los ? DBG_COL_LOS_OK : DBG_COL_LOS_FAIL;
        DrawLineEx(ps, ns, 3.0f, c);
        DrawCircleV(ns, 6.0f, c);
    }
}

/* Path visualization — 2D projected dots and lines along route */
static void dbg_draw_path_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    if (p->route_idx >= p->route_len) return;

    Vector2 prev = {0};
    for (int i = p->route_idx; i < p->route_len; i++) {
        Vector2 s = dbg_tile_to_screen(p->route_x[i], p->route_y[i], cam);
        DrawCircleV(s, 4.0f, DBG_COL_PATH);
        if (i > p->route_idx) {
            DrawLineEx(prev, s, 2.0f, DBG_COL_PATH);
        }
        prev = s;
    }

    /* Destination marker — larger circle */
    if (p->route_len > 0) {
        Vector2 dest = dbg_tile_to_screen(
            p->route_x[p->route_len - 1], p->route_y[p->route_len - 1], cam);
        DrawCircleV(dest, 8.0f, DBG_COL_PATH);
        DrawCircleLinesV(dest, 12.0f, DBG_COL_PATH);
    }
}

/* Attack range ring — 2D projected Chebyshev boundary at the wielded
 * weapon's range */
static void dbg_draw_range_2d(const FcState* state, Camera3D cam) {
    const FcPlayer* p = &state->player;
    int range = p->weapon_range;

    for (int dx = -range; dx <= range; dx++) {
        for (int dy = -range; dy <= range; dy++) {
            int dist = (abs(dx) > abs(dy)) ? abs(dx) : abs(dy);
            if (dist == range) {
                int tx = p->x + dx;
                int ty = p->y + dy;
                if (tx >= 0 && tx < FC_ARENA_WIDTH && ty >= 0 && ty < FC_ARENA_HEIGHT) {
                    Vector2 s = dbg_tile_to_screen(tx, ty, cam);
                    DrawCircleV(s, 3.0f, DBG_COL_RANGE);
                }
            }
        }
    }
}

/* Master 2D overlays for LOS/path/range — called after EndMode3D */
void debug_overlay_screen(const FcState* state, Camera3D cam, int dbg_flags) {
    if (dbg_flags & DBG_LOS)   dbg_draw_los_2d(state, cam);
    if (dbg_flags & DBG_PATH)  dbg_draw_path_2d(state, cam);
    if (dbg_flags & DBG_RANGE) dbg_draw_range_2d(state, cam);
}

void dbg_draw_prayer_window_indicator(Vector3 world_anchor, Camera3D cam) {
    Vector2 screen = GetWorldToScreen(world_anchor, cam);
    /* Keep the marker visibly separate from the centered overhead HP bar. */
    screen.x += 24.0f;
    screen.y -= 12.0f;
    int screen_width = GetScreenWidth();
    int screen_height = GetScreenHeight();
    if (screen.x < -20.0f || screen.x > (float)screen_width + 20.0f ||
        screen.y < -20.0f || screen.y > (float)screen_height + 20.0f) {
        return;
    }

    DrawCircleV(screen, 15.0f, CLITERAL(Color){70, 255, 100, 70});
    DrawCircleV(screen, 9.0f, CLITERAL(Color){70, 255, 100, 250});
    DrawCircleLines((int)screen.x, (int)screen.y, 10.0f, WHITE);
}

/* ======================================================================== */
/* Compact debug tabs (usable in any viewer panel)                          */
/* ======================================================================== */

/* Draws one formatted line of debug text at (x, *y) and moves *y down by
 * advance. */
static void dbg_print_line(int x, int* y, int advance, Color color,
                           const char* fmt, ...) {
    char text[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    fc_osrs_draw_text(text, x, *y, color);
    *y += advance;
}

/* Writes one '0'/'1' per action of an action head into bits, using
 * fc_write_mask's layout (heads in FC_ACTION_DIMS order), and returns how
 * many of them are valid. */
static int dbg_mask_bits(const float* mask, int head, char* bits) {
    int start = 0;
    for (int h = 0; h < head; h++) start += FC_ACTION_DIMS[h];
    int valid = 0;
    for (int i = 0; i < FC_ACTION_DIMS[head]; i++) {
        bits[i] = mask[start + i] > 0.5f ? '1' : '0';
        valid += bits[i] == '1';
    }
    bits[FC_ACTION_DIMS[head]] = '\0';
    return valid;
}

/* Draw one debug tab's contents from (x, by); the console clips and scrolls
 * them. dbg_tab: 0=player, 1=obs, 2=mask, 3=reward, 4=log.
 * Returns end Y position. */
int dbg_draw_panel_tabs(const FcState* state,
                        const FcRewardBreakdown* reward_breakdown,
                        const FcRewardRuntime* reward_runtime,
                        int reward_config_loaded,
                        const char* reward_config_path,
                        int x, int by, int dbg_tab) {
    int lh = 14;

    if (dbg_tab == 0) {
        /* Player state */
        const FcPlayer* p = &state->player;
        static const char* pray_str[] = { "OFF", "Melee", "Range", "Magic" };

        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "Pos:(%d,%d) Face:%.0f",
                       p->x, p->y, p->facing_angle);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "HP:%d/%d Pray:%d/%d",
                       p->current_hp/10, p->max_hp/10,
                       p->current_prayer/10, p->max_prayer/10);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "Tmr atk:%d fd:%d pot:%d",
                       p->attack_timer, p->food_timer, p->potion_timer);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL,
                       "Prayer:%s Drain:%d Bonus:+%d", pray_str[p->prayer],
                       p->prayer_drain_counter, p->prayer_bonus);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "Shark:%d Dose:%d Ammo:%d",
                       p->sharks_remaining, p->prayer_doses_remaining,
                       p->ammo_count);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL,
                       "Tgt:%d Appr:%d Route:%d/%d", p->attack_target_idx,
                       p->approach_target, p->route_idx, p->route_len);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "Hits:%d Run:%d Regen:%d",
                       p->num_pending_hits, p->is_running, p->hp_regen_counter);

        /* Active NPCs compact list */
        by += 2;
        static const char* npc_names[] = {"?","Kih","Kek","KSm","Xil","Mej","Zek","Jad","Hur"};
        for (int i = 0; i < FC_MAX_NPCS; i++) {
            const FcNpc* n = &state->npcs[i];
            if (!n->active) continue;
            const char* nm = n->npc_type > NPC_NONE && n->npc_type < NPC_TYPE_COUNT
                ? npc_names[n->npc_type] : "?";
            dbg_print_line(x, &by, lh - 2,
                           n->is_dead ? DBG_COL_BAD : DBG_COL_DIM,
                           "[%d]%s hp:%d atk:%d/%d", i, nm, n->current_hp/10,
                           n->attack_timer, n->attack_speed);
        }

    } else if (dbg_tab == 1) {
        /* Observation values */
        float obs[FC_POLICY_OBS_SIZE];
        fc_write_obs(state, obs);
        int mbase = FC_OBS_META_START;

        dbg_print_line(x, &by, lh, DBG_COL_LABEL,
                       "HP:%.2f Pray:%.2f X:%.2f Y:%.2f",
                       obs[FC_OBS_PLAYER_HP], obs[FC_OBS_PLAYER_PRAYER],
                       obs[FC_OBS_PLAYER_X], obs[FC_OBS_PLAYER_Y]);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL,
                       "Atk:%.2f Run:%.2f Sharks:%.2f Dose:%.2f",
                       obs[FC_OBS_PLAYER_ATK_TIMER],
                       obs[FC_OBS_PLAYER_RUN_ENERGY],
                       obs[FC_OBS_PLAYER_SHARKS],
                       obs[FC_OBS_PLAYER_DOSES]);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "Pray M%.0f R%.0f G%.0f T%.2f",
                       obs[FC_OBS_PLAYER_PRAY_MEL],
                       obs[FC_OBS_PLAYER_PRAY_RNG],
                       obs[FC_OBS_PLAYER_PRAY_MAG],
                       obs[FC_OBS_PLAYER_TARGET]);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "PrayDDL M%.2f R%.2f G%.2f",
                       obs[FC_OBS_PLAYER_PRAY_DDL_MEL],
                       obs[FC_OBS_PLAYER_PRAY_DDL_RNG],
                       obs[FC_OBS_PLAYER_PRAY_DDL_MAG]);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "Meta W%.2f Rot%.2f Rem%.2f",
                       obs[mbase + FC_OBS_META_WAVE],
                       obs[mbase + FC_OBS_META_ROTATION],
                       obs[mbase + FC_OBS_META_REMAINING]);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "Drain:%.2f Dmg:%.2f Clear:%.0f",
                       obs[mbase + FC_OBS_META_PRAY_DRAIN],
                       obs[mbase + FC_OBS_META_DMG_T_TICK],
                       obs[mbase + FC_OBS_META_WAVE_CLR]);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL,
                       "Prog C%.2f W%.2f Rem%.2f NoP%.2f",
                       obs[mbase + FC_OBS_META_CAVE_PROG],
                       obs[mbase + FC_OBS_META_WAVE_PROG],
                       obs[mbase + FC_OBS_META_WORK_REM],
                       obs[mbase + FC_OBS_META_NO_PROG]);

        by += 2;
        dbg_print_line(x, &by, lh - 1, DBG_COL_DIM, "NPC slots:");
        for (int s = 0; s < FC_OBS_NPC_SLOTS; s++) {
            int base = FC_OBS_NPC_START + s * FC_OBS_NPC_STRIDE;
            if (obs[base + FC_NPC_VALID] < 0.5f) continue;
            char tele = '-';
            if (obs[base + FC_NPC_TELE_MELEE]  > 0.5f) tele = 'M';
            else if (obs[base + FC_NPC_TELE_RANGED] > 0.5f) tele = 'R';
            else if (obs[base + FC_NPC_TELE_MAGIC]  > 0.5f) tele = 'A';
            dbg_print_line(x, &by, lh - 2, DBG_COL_LABEL,
                           "[%d] hp:%.2f d:%.2f tele:%c los:%.0f pd:%.2f ddl:%.0f/%.2f",
                           s,
                           obs[base + FC_NPC_HP],
                           obs[base + FC_NPC_DISTANCE],
                           tele,
                           obs[base + FC_NPC_LOS],
                           obs[base + FC_NPC_PENDING_STYLE],
                           obs[base + FC_NPC_PENDING_PRAYER_WINDOW],
                           obs[base + FC_NPC_PENDING_PRAYER_DEADLINE]);
        }

    } else if (dbg_tab == 2) {
        /* Action mask, one bit per action. Heads 0-6 are move, attack,
         * prayer, eat, drink and the two move-target coordinates. */
        static const char* labels[] = { "MOVE:", "ATK:", "PRAY:" };
        float mask[FC_ACTION_MASK_SIZE];
        char bits[FC_ACTION_MASK_SIZE + 1];
        char drink_bits[FC_ACTION_MASK_SIZE + 1];
        fc_write_mask(state, mask);

        for (int head = 0; head < 3; head++) {
            fc_osrs_draw_text(labels[head], x, by, DBG_COL_DIM);
            dbg_mask_bits(mask, head, bits);
            dbg_print_line(x + 30, &by, lh, DBG_COL_LABEL, " %s", bits);
        }
        dbg_mask_bits(mask, 3, bits);
        dbg_mask_bits(mask, 4, drink_bits);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "EAT:%s  DRINK:%s",
                       bits, drink_bits);
        int valid_x = dbg_mask_bits(mask, 5, bits);
        int valid_y = dbg_mask_bits(mask, 6, bits);
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "TGT_X:%d/%d TGT_Y:%d/%d",
                       valid_x, FC_ACTION_DIMS[5], valid_y, FC_ACTION_DIMS[6]);

    } else if (dbg_tab == 3) {
        const FcRewardBreakdown* b = reward_breakdown;
        const char* cfg_name = dbg_basename(reward_config_path);

        dbg_print_line(x, &by, lh + 2, DBG_COL_VALUE,
                       "Training reward parity");
        dbg_print_line(x, &by, lh, DBG_COL_LABEL, "cfg:%s%s", cfg_name,
                       reward_config_loaded ? "" : " (defaults)");
        dbg_print_line(x, &by, lh + 2, dbg_reward_color(b->total),
                       "total:%+.4f noatk_t:%d", b->total,
                       reward_runtime->ticks_since_attack);

        {
            struct {
                const char* name;
                float value;
            } terms[] = {
                {"progress", b->progress},
                {"dmg_taken", b->damage_taken},
                {"complete", b->cave_complete},
                {"death", b->player_death},
                {"pray_lost", b->prayer_lost},
                {"no_progress", b->no_progress},
                {"no_attack", b->no_attack},
                {"npc_heal", b->npc_heal},
                {"invalid", b->invalid_action},
                {"tick_pen", b->tick_penalty},
            };
            int term_count = (int)(sizeof(terms) / sizeof(terms[0]));

            for (int i = 0; i < term_count; i++) {
                dbg_print_line(x, &by, lh, dbg_reward_color(terms[i].value),
                               "%-12s %+.4f", terms[i].name, terms[i].value);
            }
        }
    } else if (dbg_tab == 4) {
        /* Event log, most recent at top */
        if (g_dbg_log.count == 0)
            dbg_print_line(x, &by, lh, DBG_COL_DIM, "No events yet");
        for (int i = 0; i < g_dbg_log.count; i++) {
            int idx = (g_dbg_log.head - 1 - i + DBG_LOG_MAX_ENTRIES * 2) % DBG_LOG_MAX_ENTRIES;
            dbg_print_line(x, &by, lh - 2, g_dbg_log.color[idx],
                           "t%d %s", g_dbg_log.tick[idx],
                           g_dbg_log.entries[idx]);
        }
    }

    return by;
}

/* ===================== Viewer window and main loop ===================== */

#define DEFAULT_WINDOW_W 1244
#define DEFAULT_WINDOW_H 1064
#define MAX_TPS         60.0f
#define MIN_TPS         0.25f

/* Colors */
#define COL_BG          CLITERAL(Color){ 80, 80, 85, 255 }
#define COL_PANEL       CLITERAL(Color){ 62, 53, 41, 255 }
#define COL_PANEL_BORDER CLITERAL(Color){ 42, 36, 28, 255 }
#define COL_TEXT_YELLOW CLITERAL(Color){ 255, 255,  0, 255 }
#define COL_TEXT_WHITE  CLITERAL(Color){ 255, 255, 255, 255 }
#define COL_TEXT_SHADOW CLITERAL(Color){   0,   0,   0, 255 }
#define COL_TEXT_DIM    CLITERAL(Color){ 130, 130, 140, 255 }
#define COL_TEXT_GREEN  CLITERAL(Color){ 100, 255, 100, 255 }
#define COL_HP_GREEN    CLITERAL(Color){  30, 255,  30, 255 }
#define COL_HP_RED      CLITERAL(Color){ 120,   0,   0, 255 }
#define COL_PRAY_BLUE   CLITERAL(Color){  50, 120, 210, 255 }
#define COL_GRID        CLITERAL(Color){  30,  30,  30, 80  }

#define FC_REWARD_CONFIG_PATH_MAX 256
typedef struct {
    AnimModelState* anim_state;
    AnimTrack anim;
} ObjectAnimRuntime;

/* The protection prayers the simulator models, in Prayer tab order. Each row
 * drives its hotkey, tab button, menu and overhead icon; sprite is the
 * prayeron_N / prayeroff_N frame. */
typedef struct {
    int prayer;           /* PRAYER_PROTECT_* */
    int action;           /* FC_PRAYER_* request that switches it on */
    int key;
    const char* name;     /* menu title */
    const char* label;    /* tab button text */
    Color color;
    int sprite;
} FcProtectionPrayer;

static const FcProtectionPrayer FC_PROTECTION_PRAYERS[] = {
    {PRAYER_PROTECT_MELEE, FC_PRAYER_MELEE, KEY_ONE, "Protect from Melee",
     "Prot. Melee", COL_TEXT_YELLOW, 14},
    {PRAYER_PROTECT_RANGE, FC_PRAYER_RANGE, KEY_TWO, "Protect from Missiles",
     "Prot. Range", COL_TEXT_GREEN, 13},
    {PRAYER_PROTECT_MAGIC, FC_PRAYER_MAGIC, KEY_THREE, "Protect from Magic",
     "Prot. Magic", COL_PRAY_BLUE, 12},
};
#define FC_PROTECTION_PRAYER_COUNT \
    ((int)(sizeof(FC_PROTECTION_PRAYERS) / sizeof(FC_PROTECTION_PRAYERS[0])))

/* Viewer state */
typedef struct ViewerState {
    FcState state;
    /* CPU snapshots only: worker-thread stepping must never call Raylib. */
    FcState pending_state, reset_state;
    FcRewardRuntime pending_reward_runtime;
    FcRewardBreakdown pending_reward_breakdown;
    int pending_actions[FC_NUM_ACTION_HEADS];
    int pending_frame;
    FcRenderEvents render_events;
    FcActorAnimation actor_animation;
    FcCombatPresentation* combat_presentation;
    FcUiState ui;
    FcRenderEntity entities[FC_MAX_RENDER_ENTITIES];
    int entity_count;
    int paused, step_once;
    float tps;
    float tick_acc;
    int show_grid, show_collision;
    Camera3D camera;
    float cam_yaw, cam_pitch, cam_dist;
    int camera_locked;
    int actions[FC_NUM_ACTION_HEADS];
    /* Terrain + Objects + NPC models */
    TerrainMesh* terrain;
    FcMinimapScene minimap_scene;
    ObjectMesh* objects;
    ObjectAnimSet* object_anims;
    FcAnimatedAtlas atlas;
    ModelSet* object_anim_models;
    ObjectAnimRuntime* object_anim_runtimes;
    int object_anim_runtime_count;
    ModelSet* npc_models;
    FcPlayerAppearance appearance;
    char item_message[96];
    float item_message_seconds;
    /* Animation cache (shared by player + all NPCs) */
    AnimCache* anim_cache;
    /* Buffered key inputs (captured every frame, consumed on tick) */
    int pending_prayer, pending_eat, pending_drink;
    int pending_attack_npc;
    int pending_tile_x, pending_tile_y;
    FcClickFeedback click_feedback;
    int context_npc_slot, context_npc_spawn_index;
    int context_tile_x, context_tile_y;
    Vector2 scene_right_start;
    int scene_right_tracking, scene_right_dragged;
    int console_tab;              /* controls, player, obs, mask, reward, log */
    int console_wave_dropdown_open;
    int console_scroll[5];        /* player/obs/mask/reward/log vertical offsets */
    int console_content_height[5];
    /* prayeron/prayeroff sprites per FC_PROTECTION_PRAYERS row, for the
     * Prayer tab and (on) the overhead icon. */
    Texture2D prayer_on[FC_PROTECTION_PRAYER_COUNT];
    Texture2D prayer_off[FC_PROTECTION_PRAYER_COUNT];
    Texture2D click_cross_tex[FC_CLICK_CROSS_FRAME_COUNT * 2];
    /* Debug overlay — toggled with O key */
    int dbg_flags;       /* bitmask of DBG_* flags */
    /* Debug toggles */
    int godmode;        /* 1 = player can't die */
    int policy_replay;    /* 1 = read-only presentation of native policy transitions */
    int start_wave;     /* 0 = wave 1 (default), >0 = skip to this wave on reset */
    int initial_sharks;
    int initial_prayer_doses;
    FcRewardParams reward_params;
    FcRewardRuntime reward_runtime;
    FcRewardBreakdown reward_breakdown;
    int reward_breakdown_tick;
    int reward_config_loaded;
    char reward_config_path[FC_REWARD_CONFIG_PATH_MAX];
} ViewerState;

static void viewer_trace_log_to_stderr(int log_level, const char* text,
                                       va_list args) {
    (void)log_level;
    vfprintf(stderr, text, args);
    fputc('\n', stderr);
}

/* The FC_PROTECTION_PRAYERS row of a PRAYER_PROTECT_* value, or -1. */
static int protection_prayer_index(int prayer) {
    for (int i = 0; i < FC_PROTECTION_PRAYER_COUNT; i++) {
        if (FC_PROTECTION_PRAYERS[i].prayer == prayer)
            return i;
    }
    return -1;
}

/* The prayer request that toggles this protection prayer. */
static int protection_prayer_toggle(const FcPlayer* p,
                                    const FcProtectionPrayer* row) {
    return p->prayer == row->prayer ? FC_PRAYER_OFF : row->action;
}

static void queue_viewer_prayer_button(ViewerState* v,
                                       const FcProtectionPrayer* row) {
    if (!v) return;
    FcPlayer* p = &v->state.player;
    if (p->current_prayer <= 0)
        return;
    v->pending_prayer = protection_prayer_toggle(p, row);
}

static int load_ui_item_icon(FcUiState* ui, uint32_t item_id) {
    if (!ui || item_id == 0) return 0;
    char path[128];
    snprintf(path, sizeof(path), "data/sprites/items/item_%u.png", item_id);
    Texture2D tex = fc_load_texture_asset(path);
    if (tex.id == 0) return 0;
    SetTextureFilter(tex, TEXTURE_FILTER_POINT);
    fc_ui_set_item_icon(ui, item_id, tex);
    return 1;
}

/* Icons for every item the simulation knows: the loadout and the supplies. */
static int load_fc_ui_item_icons(ViewerState* v) {
    if (!v) return 0;
    int ready = 1;
    for (int i = 0; i < (int)(sizeof(ITEMS) / sizeof(ITEMS[0])); i++)
        ready &= load_ui_item_icon(&v->ui, (uint32_t)ITEMS[i].id);
    return ready;
}

static void sync_item_slots(FcUiSlot *slots, const FcItemStack *items, int count) {
    for (int i = 0; i < count; i++) {
        FcUiSlot *slot = &slots[i];
        memset(slot, 0, sizeof(*slot));
        const FcItemDef *item = fc_item_definition(items[i].item_id);
        if (!item) continue;
        if (items[i].quantity > 0) {
            slot->item_id = (uint32_t)item->id;
            slot->quantity = items[i].quantity;
            snprintf(slot->label, sizeof(slot->label), "%s", item->name);
            slot->enabled = 1;
        }
        slot->action = item->slot == FC_EQUIP_SLOT_WEAPON ? "Wield" :
            item->slot >= 0 ? "Wear" : item->id == FC_ITEM_ID_SHARK ? "Eat" :
            item->id == FC_ITEM_ID_VIAL ? "Use" : "Drink";
    }
}

static void sync_fc_ui_items(ViewerState* v) {
    sync_item_slots(v->ui.inventory, v->state.player.inventory, FC_INVENTORY_SLOTS);
    sync_item_slots(v->ui.equipment, v->state.player.equipment, FC_EQUIPMENT_SLOTS);
}

/* The OSRS combat level: a quarter of Defence + Hitpoints + Prayer / 2, plus
 * 13/40 of the largest of Attack + Strength, 3/2 Ranged and 3/2 Magic. */
static int player_combat_level(const FcPlayer* p) {
    int best = p->attack_level + p->strength_level;
    if (p->ranged_level * 3 / 2 > best) best = p->ranged_level * 3 / 2;
    if (p->magic_level * 3 / 2 > best) best = p->magic_level * 3 / 2;
    int base = p->defence_level + p->max_hp / 10 + p->prayer_level / 2;
    return (10 * base + 13 * best) / 40;
}

static void sync_fc_ui_status(ViewerState* v) {
    if (!v) return;
    FcPlayer* p = &v->state.player;
    v->ui.hitpoints = p->current_hp > 0 ? (p->current_hp + 9) / 10 : 0;
    v->ui.hitpoints_max = p->max_hp > 0 ? (p->max_hp + 9) / 10 : 0;
    v->ui.prayer_points = p->current_prayer > 0 ? (p->current_prayer + 9) / 10 : 0;
    v->ui.prayer_points_max = p->max_prayer > 0 ? (p->max_prayer + 9) / 10 : 0;
    v->ui.run_energy = p->run_energy / 100;
    if (v->ui.run_energy < 0) v->ui.run_energy = 0;
    if (v->ui.run_energy > 100) v->ui.run_energy = 100;
    v->ui.run_enabled = p->is_running != 0;
    v->ui.combat_level = player_combat_level(p);
    const FcItemDef *weapon = fc_item_definition(p->equipment[FC_EQUIP_SLOT_WEAPON].item_id);
    fc_ui_set_combat_weapon_name(&v->ui, weapon ? weapon->name : "Unarmed");
    v->ui.combat_profile = combat_profile_for_weapon_category(
        weapon ? FC_LOADOUT.combat_style_profile : 0);
}

/* Where the minimap centres the player: the drawn position, or the tile
 * centre before the actor scene has placed the player. */
static Vector2 minimap_player_position(const ViewerState* v) {
    const FcPlayer* p = &v->state.player;
    if (!v->actor_animation.scene.player.active)
        return (Vector2){(float)p->x + 0.5f, (float)p->y + 0.5f};
    FcVisualPose pose = fc_visual_scene_player_pose(&v->actor_animation.scene);
    return (Vector2){pose.x, pose.y};
}

/* Only drawing reads the minimap, so each frame renders it once, just
 * before the interface is drawn. */
static void sync_fc_ui_minimap(ViewerState* v) {
    FcPlayer* p = &v->state.player;
    Vector2 player = minimap_player_position(v);
    Color pixels[FC_MINIMAP_DISPLAY_SIZE * FC_MINIMAP_DISPLAY_SIZE];
    fc_minimap_render(&v->minimap_scene, player.x, player.y, v->cam_yaw,
                      pixels);
    UpdateTexture(v->ui.minimap_texture, pixels);
    v->ui.minimap_rotation = v->cam_yaw;

    v->ui.minimap_dot_count = 0;
    fc_ui_add_minimap_dot(&v->ui, 0.0f, 0.0f, FC_UI_MINIMAP_DOT_PLAYER);
    /* The destination flag: the clicked tile, else the end of the route. */
    int has_destination = 0, tx = 0, ty = 0;
    if (v->click_feedback.destination_active) {
        has_destination = 1;
        tx = v->click_feedback.destination_x;
        ty = v->click_feedback.destination_y;
    } else if (p->route_idx < p->route_len && p->route_len > 0) {
        has_destination = 1;
        tx = p->route_x[p->route_len - 1];
        ty = p->route_y[p->route_len - 1];
    }
    if (has_destination) {
        Vector2 offset = fc_minimap_rotate_offset(
            (float)tx + 0.5f - player.x,
            (float)ty + 0.5f - player.y, v->cam_yaw);
        fc_ui_add_minimap_dot(&v->ui, offset.x, offset.y,
                              FC_UI_MINIMAP_DOT_DESTINATION);
    }
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* n = &v->state.npcs[i];
        if (!n->active || n->is_dead) continue;
        FcVisualPose npc_pose =
            fc_visual_scene_npc_pose(&v->actor_animation.scene, i);
        float npc_x = v->actor_animation.scene.npcs[i].active
            ? npc_pose.x : (float)n->x + (float)n->size * 0.5f;
        float npc_y = v->actor_animation.scene.npcs[i].active
            ? npc_pose.y : (float)n->y + (float)n->size * 0.5f;
        Vector2 offset = fc_minimap_rotate_offset(
            npc_x - player.x, npc_y - player.y, v->cam_yaw);
        fc_ui_add_minimap_dot(&v->ui, offset.x, offset.y,
                              FC_UI_MINIMAP_DOT_NPC);
    }
}

static void sync_fc_ui(ViewerState* v) {
    sync_fc_ui_items(v);
    sync_fc_ui_status(v);
}

static void queue_player_tile_request(ViewerState* v, int tx, int ty,
                                      float screen_x, float screen_y) {
    if (!v || v->policy_replay || tx < 0 || tx >= FC_ARENA_WIDTH ||
        ty < 0 || ty >= FC_ARENA_HEIGHT)
        return;
    v->pending_tile_x = tx;
    v->pending_tile_y = ty;
    v->pending_attack_npc = -1;
    fc_click_feedback_select_move(&v->click_feedback, &v->state, tx, ty,
                                  screen_x, screen_y);
}

static void queue_player_attack_request(ViewerState* v, int npc_idx,
                                        float screen_x, float screen_y) {
    if (!v || v->policy_replay || npc_idx < 0 || npc_idx >= FC_MAX_NPCS) return;
    v->pending_attack_npc = npc_idx;
    v->pending_tile_x = -1;
    v->pending_tile_y = -1;
    fc_click_feedback_select_interaction(&v->click_feedback,
                                         screen_x, screen_y);
}

static void show_item_result(ViewerState *v, FcItemResult result) {
    snprintf(v->item_message, sizeof(v->item_message), "%s", fc_item_result_message(result));
    v->item_message_seconds = result == FC_ITEM_OK ? 0.0f : 4.0f;
}

static void use_inventory_slot(ViewerState *v, int slot) {
    if (v->policy_replay || slot < 0 || slot >= FC_INVENTORY_SLOTS) return;
    const FcItemDef *item = fc_item_definition(v->state.player.inventory[slot].item_id);
    if (!item) return;
    FcItemResult result;
    if (item->slot >= 0) {
        result = fc_equip_item(&v->state, slot);
        if (result == FC_ITEM_OK) v->pending_attack_npc = -1;
    } else {
        result = fc_select_consumable(&v->state, slot);
        if (result == FC_ITEM_OK) {
            if (item->id == FC_ITEM_ID_SHARK) v->pending_eat = FC_EAT_SHARK;
            else v->pending_drink = FC_DRINK_PRAYER_POT;
        }
    }
    show_item_result(v, result);
}

static void handle_fc_ui_intent(ViewerState* v) {
    if (!v) return;
    FcUiIntent* intent = &v->ui.last_intent;
    FcPlayer* p = &v->state.player;
    switch (intent->kind) {
        case FC_UI_INTENT_WORLD_ACTION: {
            int slot = v->context_npc_slot;
            const FcNpc *npc = slot >= 0 && slot < FC_MAX_NPCS ? &v->state.npcs[slot] : NULL;
            int same_npc = npc && npc->active && !npc->is_dead &&
                npc->spawn_index == v->context_npc_spawn_index;
            if (strcmp(intent->text, "Examine") == 0 && same_npc) {
                snprintf(v->item_message, sizeof(v->item_message), "%s", fc_menu_npc_info(npc->npc_type).name);
                v->item_message_seconds = 4.0f;
            } else if (!v->policy_replay && v->state.terminal == TERMINAL_NONE) {
                if (strcmp(intent->text, "Attack") == 0 && same_npc)
                    queue_player_attack_request(v, slot, intent->position.x, intent->position.y);
                else if (strcmp(intent->text, "Walk here") == 0)
                    queue_player_tile_request(v, v->context_tile_x, v->context_tile_y,
                                               intent->position.x, intent->position.y);
            }
            break;
        }
        case FC_UI_INTENT_INVENTORY_SLOT:
            use_inventory_slot(v, intent->primary);
            break;
        case FC_UI_INTENT_INVENTORY_ACTION:
            /* The menu offers the slot's left-click verb, then Examine. */
            if (strcmp(intent->text, "Examine") == 0) {
                snprintf(v->item_message, sizeof(v->item_message), "%s",
                         v->ui.inventory[intent->primary].label);
                v->item_message_seconds = 4.0f;
            } else {
                use_inventory_slot(v, intent->primary);
            }
            break;
        case FC_UI_INTENT_EQUIPMENT_ACTION:
            if (strcmp(intent->text, "Examine") == 0) {
                snprintf(v->item_message, sizeof(v->item_message), "%s",
                         v->ui.equipment[intent->primary].label);
                v->item_message_seconds = 4.0f;
            }
            if (strcmp(intent->text, "Remove") != 0) break;
            /* fall through */
        case FC_UI_INTENT_EQUIPMENT_SLOT:
            if (!v->policy_replay)
                show_item_result(v, fc_unequip_item(&v->state, intent->primary));
            break;
        case FC_UI_INTENT_INVENTORY_DRAG:
            if (!v->policy_replay)
                show_item_result(v, fc_inventory_swap(&v->state, intent->primary, intent->secondary));
            break;
        case FC_UI_INTENT_PRAYER_SLOT: {
            int row = intent->primary;
            if (row < 0 || row >= FC_PROTECTION_PRAYER_COUNT ||
                p->current_prayer <= 0)
                break;
            int action = protection_prayer_toggle(p, &FC_PROTECTION_PRAYERS[row]);
            /* A context-menu verb is explicit, not a toggle of whatever
             * prayer happens to be active when the option is selected. */
            if (intent->secondary == 1 && action == FC_PRAYER_OFF) action = 0;
            if (intent->secondary == -1 && action != FC_PRAYER_OFF) action = 0;
            if (action && !v->policy_replay) v->pending_prayer = action;
            break;
        }
        case FC_UI_INTENT_RUN_TOGGLE:
            if (!v->policy_replay) {
                fc_request_set_running(&v->state, !p->is_running);
                v->ui.run_enabled = p->is_running != 0;
            }
            break;
        case FC_UI_INTENT_MINIMAP_CLICK: {
            Vector2 player = minimap_player_position(v);
            int tile_x = -1;
            int tile_y = -1;
            Vector2 mouse = GetMousePosition();
            if (fc_minimap_click_to_tile(
                    (float)intent->primary, (float)intent->secondary,
                    player.x, player.y, v->cam_yaw, &tile_x, &tile_y)) {
                queue_player_tile_request(v, tile_x, tile_y,
                                          mouse.x, mouse.y);
            }
            break;
        }
        default:
            break;
    }
}

static void text_s(const char* t, int x, int y, Color c) {
    fc_osrs_draw_text(t, x+1, y+1, COL_TEXT_SHADOW);
    fc_osrs_draw_text(t, x, y, c);
}

static int float_near(float a, float b) {
    return fabsf(a - b) < 0.0001f;
}

/* Manual play applies the [env] section of config/fight_caves.ini, found
 * relative to the working directory like the trainer's config, the way
 * puf_init does; without the file it keeps the defaults. */
static void load_reward_params(ViewerState* v) {
    const char* config_path = "config/fight_caves.ini";
    FILE* config = fopen(config_path, "r");
    int found = config != NULL;
    Ini ini = {0};
    if (config) {
        fclose(config);
        puf_ini_load_file(&ini, config_path);
    }
    /* A config without an [env] section also keeps the defaults. */
    fc_apply_env_config(puf_ini_section(&ini, "env", 1), &v->reward_params,
                        &v->initial_sharks, &v->initial_prayer_doses);
    puf_ini_free(&ini);
    v->reward_config_loaded = found;
    snprintf(v->reward_config_path, sizeof(v->reward_config_path), "%s",
             found ? config_path : "defaults");
}

/* Restart reward tracking from the current state. */
static void reset_reward_tracking(ViewerState* v) {
    memset(&v->reward_breakdown, 0, sizeof(v->reward_breakdown));
    v->reward_breakdown_tick = -1;
    fc_reward_runtime_begin_episode(&v->reward_runtime, &v->state);
}

static void update_reward_breakdown(ViewerState* v) {
    if (v->reward_breakdown_tick == v->state.tick) return;
    v->reward_breakdown = fc_reward_compute_breakdown(
        &v->state, &v->reward_params, &v->reward_runtime);
    fc_reward_sync_progress_state(&v->state, &v->reward_runtime);
    v->reward_breakdown_tick = v->state.tick;
}

static const float MANUAL_TPS_PRESETS[] = {
    0.25f, 0.50f, POLICY_REPLAY_BASE_TPS, 4.0f, 10.0f, 15.0f, 30.0f, 60.0f
};
static const char* MANUAL_TPS_LABELS[] = {
    "0.25", "0.5", "5/3", "4", "10", "15", "30", "60"
};
#define NUM_MANUAL_TPS_PRESETS ((int)(sizeof(MANUAL_TPS_PRESETS) / sizeof(MANUAL_TPS_PRESETS[0])))

/* Policy replay speeds as multiples of POLICY_REPLAY_BASE_TPS. Keys 1, 2, 4
 * and 0 pick REPLAY_SPEEDS[0..3]; Up and Down step through them. */
static const int REPLAY_SPEEDS[] = {1, 2, 4, 10};
#define REPLAY_SPEED_COUNT ((int)(sizeof(REPLAY_SPEEDS) / sizeof(REPLAY_SPEEDS[0])))

static float replay_speed_tps(int index) {
    return (float)(POLICY_REPLAY_BASE_TPS * REPLAY_SPEEDS[index]);
}

/* The REPLAY_SPEEDS index of the current tick rate; any other rate (a
 * console preset) counts as 1x. */
static int replay_speed_index(const ViewerState* v) {
    for (int i = 0; i < REPLAY_SPEED_COUNT; i++) {
        if (float_near(v->tps, replay_speed_tps(i))) return i;
    }
    return 0;
}

static void set_viewer_tps(ViewerState* v, float tps) {
    if (tps < MIN_TPS) tps = MIN_TPS;
    if (tps > MAX_TPS) tps = MAX_TPS;
    v->tps = tps;
    if (v->tick_acc >= 1.0f)
        v->tick_acc = fmodf(v->tick_acc, 1.0f);
}

/* Switch to REPLAY_SPEEDS[index], clamped to the table. */
static void set_policy_replay_speed(ViewerState* v, int index) {
    if (index < 0) index = 0;
    if (index > REPLAY_SPEED_COUNT - 1) index = REPLAY_SPEED_COUNT - 1;
    set_viewer_tps(v, replay_speed_tps(index));
    fprintf(stderr, "[eval] Replay speed set to %dx (%.2f TPS)\n",
            REPLAY_SPEEDS[index], v->tps);
}

static void toggle_debug_overlay(ViewerState* v) {
    if (!v) return;
    v->dbg_flags = v->dbg_flags ? 0 : DBG_ALL;
}

static void toggle_godmode(ViewerState* v) {
    if (!v || v->policy_replay) return;
    v->godmode = !v->godmode;
    fprintf(stderr, "GODMODE: %s\n", v->godmode ? "ON" : "OFF");
}

static void sync_player_appearance(ViewerState *v) {
    int changed = fc_player_appearance_sync(&v->appearance, &v->state.player,
                                            FC_PLAYER_MODEL_ID);
    if (changed < 0) {
        fprintf(stderr, "Cannot compose player appearance from local equipment assets.\n");
        exit(EXIT_FAILURE);
    }
    if (changed && v->actor_animation.player_state) {
        anim_model_state_free(v->actor_animation.player_state);
        v->actor_animation.player_state = NULL;
        /* Keep pose/action clocks: a gear change isn't an animation restart. */
    }
}

/* Rebuild what is drawn from the state: render entities and events, combat
 * effects, actor animation, click feedback and the debug event log. */
static void reset_scene_presentation(ViewerState* v) {
    fc_fill_render_entities(&v->state, v->entities, &v->entity_count);
    fc_fill_render_events(&v->state, &v->render_events);
    fc_combat_presentation_reset(v->combat_presentation);
    fc_actor_animation_reset(&v->actor_animation, &v->state,
                             v->appearance.model);
    fc_click_feedback_reset(&v->click_feedback);
    dbg_log_clear();
}

/* A new episode or replay state: also drops buffered input and messages. */
static void fc_viewer_reset_presentation(ViewerState* v) {
    sync_player_appearance(v);
    v->item_message_seconds = 0.0f;
    memset(v->actions, 0, sizeof(v->actions));
    v->pending_prayer = 0;
    v->pending_eat = 0;
    v->pending_drink = 0;
    v->pending_attack_npc = -1;
    v->pending_tile_x = -1;
    v->pending_tile_y = -1;
    reset_scene_presentation(v);
}

/* Replace the NPCs with a fresh spawn of `wave`, restore HP and Prayer, and
 * clear the terminal and Jad-healer flags (already clear after fc_reset). */
static void restart_at_wave(FcState* state, int wave) {
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        state->npcs[i].active = 0;
        state->npcs[i].is_dead = 0;
    }
    state->npcs_remaining = 0;
    state->current_wave = wave;
    fc_wave_spawn(state, wave);
    state->player.current_hp = state->player.max_hp;
    state->player.current_prayer = state->player.max_prayer;
    state->terminal = TERMINAL_NONE;
    state->jad_healers_spawned = 0;
}

static void reset_ep(ViewerState* v) {
    fc_ui_close_context(&v->ui);
    v->scene_right_tracking = v->scene_right_dragged = 0;
    load_reward_params(v);
    fc_reset(&v->state, (uint32_t)GetRandomValue(1, 999999));
    if (v->start_wave > 1 && v->start_wave <= FC_NUM_WAVES)
        restart_at_wave(&v->state, v->start_wave);
    fc_set_initial_supplies(&v->state, v->initial_sharks, v->initial_prayer_doses);
    reset_reward_tracking(v);
    fc_viewer_reset_presentation(v);
}

static void viewer_jump_to_wave(ViewerState* v, int wave) {
    if (!v || v->policy_replay) return;
    fc_ui_close_context(&v->ui);
    if (wave < 1) wave = 1;
    if (wave > FC_NUM_WAVES) wave = FC_NUM_WAVES;
    restart_at_wave(&v->state, wave);
    reset_reward_tracking(v);
    reset_scene_presentation(v);
}

/* ======================================================================== */
/* Human input → action heads                                                */
/* ======================================================================== */

/* Raycast from mouse position to find the tile coordinate on the ground plane */
static int raycast_to_tile(ViewerState* v, int* out_x, int* out_y) {
    Ray ray = GetScreenToWorldRay(GetMousePosition(), v->camera);
    /* Intersect with the ground plane at the arena centre's height */
    float gy = ground_height(v->terrain, 32, 32);
    if (fabsf(ray.direction.y) < 0.001f) return 0;  /* ray parallel to ground */
    float t = (gy - ray.position.y) / ray.direction.y;
    if (t < 0) return 0;  /* behind camera */
    float wx = ray.position.x + ray.direction.x * t;
    float wz = ray.position.z + ray.direction.z * t;
    /* Convert to tile coords: X = world X, tile Y = -world Z */
    int tx = (int)floorf(wx);
    int ty = (int)floorf(-wz);
    if (tx < 0 || tx >= FC_ARENA_WIDTH || ty < 0 || ty >= FC_ARENA_HEIGHT) return 0;
    *out_x = tx;
    *out_y = ty;
    return 1;
}

/* Pick what was drawn, including the individual animation and interpolated
 * position. Core state only decides whether the picked NPC is still targetable. */
static int find_clicked_npc_idx(ViewerState* v) {
    int best = -1;
    float best_depth = INFINITY;
    Vector2 mouse = GetMousePosition();
    for (int i = 0; i < v->entity_count; i++) {
        const FcRenderEntity *entity = &v->entities[i];
        int slot = entity->npc_slot;
        if (entity->entity_type == ENTITY_PLAYER || slot < 0 || slot >= FC_MAX_NPCS) continue;
        const FcNpc *n = &v->state.npcs[slot];
        if (!n->active || n->is_dead) continue;
        FcVisualPose pose = fc_visual_scene_npc_pose(&v->actor_animation.scene, slot);
        Vector3 position = {
            pose.x, smooth_ground_height(v->terrain, pose.x, pose.y), -pose.y};
        const ModelEntry *model = npc_model_find(v->npc_models,
                                                 entity->npc_type);
        const AnimModelState *animation = v->actor_animation.npcs[slot].state;
        float depth = models_pick_depth(model, animation ? animation->verts : NULL,
            position, pose.yaw_degrees, v->camera, mouse, GetScreenWidth(), GetScreenHeight());
        if (depth >= 0 && depth < best_depth) { best_depth = depth; best = slot; }
    }
    return best;
}

static void open_scene_context_menu(ViewerState *v) {
    int tx = -1, ty = -1;
    int can_walk = raycast_to_tile(v, &tx, &ty);
    int slot = find_clicked_npc_idx(v);
    if (slot < 0 && !can_walk) return;
    v->context_npc_slot = slot;
    v->context_npc_spawn_index = slot >= 0 ? v->state.npcs[slot].spawn_index : -1;
    v->context_tile_x = tx;
    v->context_tile_y = ty;
    fc_ui_open_world_context(&v->ui, GetMousePosition(),
        slot >= 0 ? v->state.npcs[slot].npc_type : 0, can_walk);
}

/* Called EVERY FRAME to capture clicks (which only fire once at 60fps).
 * Buffers authoritative actions and starts presentation-only feedback. */
static void process_human_clicks(ViewerState* v, int ui_capture) {
    FcPlayer* p = &v->state.player;

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
        !IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && !IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
        if (ui_capture) return;
        Vector2 mpos = GetMousePosition();
        int tx = -1;
        int ty = -1;
        int rc = raycast_to_tile(v, &tx, &ty);
        fprintf(stderr, "CLICK mouse=(%.0f,%.0f) raycast=%d tile=(%d,%d) player=(%d,%d)",
                mpos.x, mpos.y, rc, tx, ty, p->x, p->y);
        int npc_idx = find_clicked_npc_idx(v);
        if (npc_idx >= 0) {
            queue_player_attack_request(v, npc_idx, mpos.x, mpos.y);
            fprintf(stderr, " → ATTACK npc_idx=%d\n", npc_idx);
        } else if (rc) {
            int walkable = v->state.walkable[tx][ty];
            fprintf(stderr, " walkable=%d", walkable);
            queue_player_tile_request(v, tx, ty, mpos.x, mpos.y);
            fprintf(stderr, " → MOVE%s\n", walkable ? "" : "-NEAR");
        } else {
            fprintf(stderr, " → MISS (raycast failed)\n");
        }
    }
}

/* Called EVERY FRAME for key presses. Buffers actions for next tick. */
static void process_human_keys(ViewerState* v) {
    FcPlayer* p = &v->state.player;
    for (int i = 0; i < FC_PROTECTION_PRAYER_COUNT; i++) {
        if (IsKeyPressed(FC_PROTECTION_PRAYERS[i].key))
            v->pending_prayer = protection_prayer_toggle(p, &FC_PROTECTION_PRAYERS[i]);
    }
    if (IsKeyPressed(KEY_F))     v->pending_eat = FC_EAT_SHARK;
    if (IsKeyPressed(KEY_P))     v->pending_drink = FC_DRINK_PRAYER_POT;
    if (IsKeyPressed(KEY_X))
        fc_request_set_running(&v->state, !p->is_running);

    /* --- Debug toggles (testing only) --- */
    /* F9: toggle godmode (player can't die) */
    if (IsKeyPressed(KEY_F9)) toggle_godmode(v);
}

/* Called on TICK frames: build action array from buffered inputs. */
static void build_human_actions(ViewerState* v) {
    memset(v->actions, 0, sizeof(v->actions));
    v->actions[0] = FC_MOVE_IDLE;
    v->actions[1] = FC_ATTACK_NONE;
    if (v->pending_attack_npc >= 0) {
        int visible[FC_VISIBLE_NPCS];
        int count = fc_visible_npc_indices(&v->state, visible);
        for (int slot = 0; slot < count; slot++) {
            if (visible[slot] == v->pending_attack_npc) {
                v->actions[1] = slot + 1;
                break;
            }
        }
    }
    if (v->pending_tile_x >= 0 && v->pending_tile_y >= 0) {
        v->actions[5] = v->pending_tile_x + 1;
        v->actions[6] = v->pending_tile_y + 1;
    }
    /* Buffered prayer/eat/drink */
    v->actions[2] = v->pending_prayer;
    v->actions[3] = v->pending_eat;
    v->actions[4] = v->pending_drink;
    /* Clear buffers */
    v->pending_prayer = 0;
    v->pending_eat = 0;
    v->pending_drink = 0;
    v->pending_attack_npc = -1;
    v->pending_tile_x = -1;
    v->pending_tile_y = -1;
}

static Vector2 click_route_point_to_screen(ViewerState* v,
                                           float tile_x, float tile_y) {
    Vector3 world = {
        tile_x,
        smooth_ground_height(v->terrain, tile_x, tile_y) + 0.08f,
        -tile_y,
    };
    return GetWorldToScreen(world, v->camera);
}

static void draw_click_destination_3d(ViewerState* v) {
    if (!v || !v->click_feedback.destination_active) return;
    int tx = v->click_feedback.destination_x;
    int ty = v->click_feedback.destination_y;
    if (tx < 0 || tx >= FC_ARENA_WIDTH ||
        ty < 0 || ty >= FC_ARENA_HEIGHT) {
        return;
    }

    float y = ground_height(v->terrain, tx, ty) + 0.025f;
    Vector3 center = {(float)tx + 0.5f, y, -((float)ty + 0.5f)};
    Color fill = {255, 215, 0, 70};
    Color edge = {255, 235, 70, 230};
    DrawCube(center, 0.92f, 0.025f, 0.92f, fill);
    DrawCubeWires(center, 0.92f, 0.025f, 0.92f, edge);
}

static void draw_click_route_2d(ViewerState* v) {
    if (!v) return;
    const int* route_x = NULL;
    const int* route_y = NULL;
    int start = 0;
    int len = 0;
    if (!fc_click_feedback_route(&v->click_feedback, &v->state,
                                 &route_x, &route_y, &start, &len)) {
        return;
    }

    FcVisualPose player =
        fc_visual_scene_player_pose(&v->actor_animation.scene);
    Vector2 previous = click_route_point_to_screen(v, player.x, player.y);
    Color line = {255, 220, 30, 210};
    Color point = {255, 245, 120, 240};
    for (int i = start; i < len; i++) {
        Vector2 next = click_route_point_to_screen(
            v, (float)route_x[i] + 0.5f, (float)route_y[i] + 0.5f);
        DrawLineEx(previous, next, 2.0f, line);
        DrawCircleV(next, 2.5f, point);
        previous = next;
    }
}

static void draw_click_cross(ViewerState* v) {
    if (!v) return;
    int frame = fc_click_feedback_cross_frame(&v->click_feedback);
    if (frame < 0) return;
    int base = v->click_feedback.cross_kind == FC_CLICK_CROSS_INTERACTION
        ? FC_CLICK_CROSS_FRAME_COUNT : 0;
    Texture2D texture = v->click_cross_tex[base + frame];
    if (texture.id == 0) return;
    DrawTexture(texture,
                (int)roundf(v->click_feedback.cross_screen_x) -
                    texture.width / 2,
                (int)roundf(v->click_feedback.cross_screen_y) -
                    texture.height / 2,
                WHITE);
}

static Vector3 camera_follow_target(const ViewerState* v) {
    float cx = FC_ARENA_WIDTH * 0.5f;
    float cy = -(FC_ARENA_HEIGHT * 0.5f);
    if (v->entity_count > 0) {
        FcVisualPose pose = entity_pose(&v->actor_animation.scene, &v->entities[0]);
        cx = pose.x;
        cy = -pose.y;
    }
    return (Vector3){cx, 0.5f, cy};
}

static void draw_npc_prayer_window_indicators(ViewerState* v) {
    if (!v || !v->dbg_flags) return;

    for (int i = 0; i < v->entity_count; i++) {
        const FcRenderEntity* entity = &v->entities[i];
        int npc_idx = entity->npc_slot;
        if (entity->entity_type != ENTITY_NPC || entity->is_dead ||
            npc_idx < 0 || npc_idx >= FC_MAX_NPCS ||
            !fc_actor_animation_prayer_window_active(
                &v->actor_animation, npc_idx, v->state.tick)) {
            continue;
        }

        FcVisualPose pose = entity_pose(&v->actor_animation.scene, entity);
        Vector3 anchor = {
            pose.x,
            smooth_ground_height(v->terrain, pose.x, pose.y) +
                2.2f + (float)entity->size * 0.8f,
            -pose.y
        };
        dbg_draw_prayer_window_indicator(anchor, v->camera);
    }
}

/* ======================================================================== */
/* Scene drawing                                                             */
/* ======================================================================== */

static void draw_animated_objects(ViewerState* v) {
    if (!v || !v->object_anims ||
        !v->object_anim_models || !v->object_anim_runtimes)
        return;

    float dt = GetFrameTime();
    rlDisableBackfaceCulling();
    for (int i = 0; i < v->object_anims->count; i++) {
        ObjectAnimPlacement* row = &v->object_anims->rows[i];
        ModelEntry* entry = model_find(v->object_anim_models, row->model_id);
        if (!entry) continue;

        if (row->animation_id >= 0 && v->anim_cache) {
            ObjectAnimRuntime* rt = &v->object_anim_runtimes[i];
            fc_model_animation_update(entry, v->anim_cache, &rt->anim_state,
                                      &rt->anim, row->animation_id, dt,
                                      row->phase_ticks);
        }

        DrawModelEx(entry->model,
                    (Vector3){row->pos_x, row->pos_y, row->pos_z},
                    (Vector3){0, 1, 0}, 0.0f,
                    (Vector3){1, 1, 1}, WHITE);
    }
    rlEnableBackfaceCulling();
}

static void draw_actor_footprint(ViewerState* v,
                                 const FcRenderEntity* entity) {
    if (!v || !entity || entity->size <= 0) {
        return;
    }
    if (entity->is_dead &&
        (entity->entity_type != ENTITY_NPC ||
         !fc_combat_presentation_npc_death_deferred(
             v->combat_presentation, &v->state, entity->npc_slot))) {
        return;
    }

    const int is_player = entity->entity_type == ENTITY_PLAYER;
    const Color fill = is_player
        ? CLITERAL(Color){50, 220, 100, 72}
        : CLITERAL(Color){50, 140, 255, 62};
    const Color outline = is_player
        ? CLITERAL(Color){80, 255, 130, 225}
        : CLITERAL(Color){80, 180, 255, 210};
    TerrainMesh* terrain = v->terrain;
    for (int offset_x = 0; offset_x < entity->size; offset_x++) {
        for (int offset_y = 0; offset_y < entity->size; offset_y++) {
            float tile_x = (float)(entity->x + offset_x) + 0.5f;
            float tile_y = (float)(entity->y + offset_y) + 0.5f;
            float height = smooth_ground_height(terrain, tile_x, tile_y) + 0.035f;
            Vector3 center = {tile_x, height, -tile_y};
            DrawCube(center, 0.94f, 0.02f, 0.94f, fill);
        }
    }

    float min_x = (float)entity->x + 0.03f;
    float max_x = (float)(entity->x + entity->size) - 0.03f;
    float min_y = (float)entity->y + 0.03f;
    float max_y = (float)(entity->y + entity->size) - 0.03f;
    Vector3 northwest = {
        min_x, smooth_ground_height(terrain, min_x, min_y) + 0.055f, -min_y};
    Vector3 northeast = {
        max_x, smooth_ground_height(terrain, max_x, min_y) + 0.055f, -min_y};
    Vector3 southeast = {
        max_x, smooth_ground_height(terrain, max_x, max_y) + 0.055f, -max_y};
    Vector3 southwest = {
        min_x, smooth_ground_height(terrain, min_x, max_y) + 0.055f, -max_y};
    DrawLine3D(northwest, northeast, outline);
    DrawLine3D(northeast, southeast, outline);
    DrawLine3D(southeast, southwest, outline);
    DrawLine3D(southwest, northwest, outline);
}

/* What the combat presentation reads from the viewer, as of now. */
static FcCombatPresentationContext combat_presentation_context(ViewerState* v) {
    return (FcCombatPresentationContext){
        .state = &v->state,
        .events = &v->render_events,
        .scene = &v->actor_animation.scene,
        .terrain = v->terrain,
        .anim_cache = v->anim_cache,
        .player_profile = fc_player_visual_profile(&v->state.player),
        .tps = v->tps,
    };
}

static void draw_scene(ViewerState* v) {
    if (v->camera_locked) {
        v->camera.target = camera_follow_target(v);
    }
    v->camera.position = (Vector3){
        v->camera.target.x + v->cam_dist*cosf(v->cam_pitch)*sinf(v->cam_yaw),
        v->cam_dist*sinf(v->cam_pitch),
        v->camera.target.z + v->cam_dist*cosf(v->cam_pitch)*cosf(v->cam_yaw) };
    BeginMode3D(v->camera);

    /* Terrain + objects */
    if (v->terrain) {
        rlDisableBackfaceCulling();
        DrawModel(v->terrain->model, (Vector3){0,0,0}, 1.0f, WHITE);
        rlEnableBackfaceCulling();
    }
    if (v->objects) {
        rlDisableBackfaceCulling();
        DrawModel(v->objects->model, (Vector3){0,0,0}, 1.0f, WHITE);
        rlEnableBackfaceCulling();
    }
    draw_animated_objects(v);

    /* Grid overlay */
    if (v->show_grid) {
        for (int x = 0; x <= FC_ARENA_WIDTH; x++)
            DrawLine3D((Vector3){(float)x,0.01f,0}, (Vector3){(float)x,0.01f,-(float)FC_ARENA_HEIGHT}, COL_GRID);
        for (int z = 0; z <= FC_ARENA_HEIGHT; z++)
            DrawLine3D((Vector3){0,0.01f,-(float)z}, (Vector3){(float)FC_ARENA_WIDTH,0.01f,-(float)z}, COL_GRID);
    }

    /* Movement feedback is immediate, even though the selected action remains
     * buffered until the next authoritative simulation tick. */
    draw_click_destination_3d(v);

    /* Entities */
    for (int i = 0; i < v->entity_count; i++) {
        FcRenderEntity* e = &v->entities[i];

        FcVisualPose pose = entity_pose(&v->actor_animation.scene, e);
        float ex = pose.x;
        float ey = -pose.y;

        /* Sample terrain continuously along the interpolated movement path. */
        float gy = smooth_ground_height(v->terrain, pose.x, pose.y);

        draw_actor_footprint(v, e);
        /* sync_player_appearance composes the player model every frame and
         * startup requires every NPC type's model. Facing comes from the
         * client-style actor runtime. */
        ModelEntry* model;
        if (e->entity_type == ENTITY_PLAYER) {
            model = fc_actor_player_model_entry(v->appearance.model);
        } else {
            model = npc_model_find(v->npc_models, e->npc_type);
            if (e->npc_slot >= 0 && e->npc_slot < FC_MAX_NPCS &&
                v->actor_animation.npcs[e->npc_slot].state) {
                /* Same-type NPCs share an asset mesh. Upload this actor's
                 * transformed vertices immediately before its draw call. */
                fc_actor_animation_upload_npc(
                    &v->actor_animation, e->npc_slot, model);
            }
        }
        rlDisableBackfaceCulling();
        DrawModelEx(model->model, (Vector3){ex, gy, ey}, (Vector3){0,1,0},
                    pose.yaw_degrees, (Vector3){1,1,1}, WHITE);
        rlEnableBackfaceCulling();
    }

    FcCombatPresentationContext combat_context = combat_presentation_context(v);
    fc_combat_presentation_draw_world(v->combat_presentation,
                                      &combat_context, GetFrameTime());
    /* Debug overlays — 3D collision tiles (before EndMode3D). C shows the
     * collision tiles on their own. */
    int overlay_3d_flags =
        v->dbg_flags | (v->show_collision ? DBG_COLLISION : 0);
    if (overlay_3d_flags) debug_overlay_3d(&v->state, overlay_3d_flags);

    EndMode3D();

    FcCombatPresentationDrawContext combat_draw_context = {
        .presentation = combat_context,
        .entities = v->entities,
        .entity_count = v->entity_count,
        .player_models = v->appearance.model,
        .npc_models = v->npc_models,
        .ui_assets = &v->ui.assets,
        .camera = v->camera,
    };
    /* Native client actor overheads are fixed-size screen-space sprites. */
    fc_combat_presentation_draw_healthbars(v->combat_presentation,
                                            &combat_draw_context);

    /* Debug overlays — 2D screen-space (LOS, path, range — after EndMode3D) */
    if (v->dbg_flags) {
        int debug_flags = v->dbg_flags;
        if (v->click_feedback.destination_active)
            debug_flags &= ~DBG_PATH;
        debug_overlay_screen(&v->state, v->camera, debug_flags);
        draw_npc_prayer_window_indicators(v);
    }

    draw_click_route_2d(v);

    fc_combat_presentation_draw_hitsplats(v->combat_presentation,
                                          &combat_draw_context);

    /* Prayer overhead icon — 2D projected from player head position */
    int prayer_row = protection_prayer_index(fc_actor_animation_render_prayer(
        &v->actor_animation, &v->state));
    if (v->entity_count > 0 && prayer_row >= 0) {
        FcVisualPose pose = entity_pose(&v->actor_animation.scene, &v->entities[0]);
        float p_gy = smooth_ground_height(v->terrain, pose.x, pose.y);
        Vector3 head_pos = {pose.x, p_gy + 3.0f, -pose.y};
        Vector2 scr = GetWorldToScreen(head_pos, v->camera);
        int px = (int)scr.x, py = (int)scr.y;

        /* Scale sprite to ~28x28 pixels and center on projected position */
        Texture2D tex = v->prayer_on[prayer_row];
        float scale = 28.0f / (float)tex.width;
        int dw = (int)(tex.width * scale);
        int dh = (int)(tex.height * scale);
        DrawTextureEx(tex, (Vector2){(float)(px - dw/2), (float)(py - dh/2)},
                      0.0f, scale, WHITE);
    }
}

/* ======================================================================== */
/* UI drawing                                                                */
/* ======================================================================== */

#define FC_CONSOLE_TAB_COUNT 6
#define FC_CONSOLE_NPC_ROWS 8
#define FC_CONSOLE_NPC_ROW_H 13
#define FC_CONSOLE_TPS_COLS 4

static Rectangle fc_console_panel_rect(void) {
    return fc_ui_chat_panel_rect(GetScreenWidth(), GetScreenHeight());
}

static Rectangle fc_console_body_rect(Rectangle panel) {
    return (Rectangle){panel.x + 7.0f, panel.y + 7.0f,
                       panel.width - 13.0f, panel.height - 39.0f};
}

static Rectangle fc_console_tab_rect(Rectangle panel, int index) {
    float width = (panel.width - 6.0f) / (float)FC_CONSOLE_TAB_COUNT;
    return (Rectangle){panel.x + 3.0f + width * (float)index,
                       panel.y + panel.height - 23.0f,
                       width, 21.0f};
}

static Rectangle fc_console_debug_button_rect(Rectangle body) {
    float right_x = body.x + 286.0f;
    float width = body.x + body.width - right_x;
    return (Rectangle){right_x, body.y + 15.0f,
                       (width - 4.0f) * 0.5f, 18.0f};
}

static Rectangle fc_console_god_button_rect(Rectangle body) {
    Rectangle debug = fc_console_debug_button_rect(body);
    return (Rectangle){debug.x + debug.width + 4.0f, debug.y,
                       debug.width, debug.height};
}

static Rectangle fc_console_wave_button_rect(Rectangle body) {
    return (Rectangle){body.x + 286.0f, body.y + 37.0f,
                       body.width - 286.0f, 20.0f};
}

static Rectangle fc_console_tps_button_rect(Rectangle body,
                                            int index) {
    float box_x = body.x + 286.0f;
    float box_w = body.width - 286.0f;
    const float gap = 3.0f;
    float button_w = (box_w - gap * (FC_CONSOLE_TPS_COLS - 1)) /
                     FC_CONSOLE_TPS_COLS;
    int row = index / FC_CONSOLE_TPS_COLS;
    int col = index % FC_CONSOLE_TPS_COLS;
    return (Rectangle){box_x + col * (button_w + gap),
                       body.y + 78.0f + row * 20.0f,
                       button_w, 17.0f};
}

static Rectangle fc_console_target_row_rect(Rectangle body,
                                            int row) {
    return (Rectangle){body.x + 1.0f,
                       body.y + 16.0f + row * FC_CONSOLE_NPC_ROW_H,
                       277.0f, FC_CONSOLE_NPC_ROW_H};
}

static Rectangle fc_console_wave_cell_rect(Rectangle body,
                                           int wave) {
    const int columns = 9;
    const float gap = 1.0f;
    float cell_w = (body.width - 6.0f - gap * (columns - 1)) / columns;
    int index = wave - 1;
    int row = index / columns;
    int col = index % columns;
    return (Rectangle){body.x + 3.0f + col * (cell_w + gap),
                       body.y + 18.0f + row * 15.0f,
                       cell_w, 14.0f};
}

static void draw_fc_console_button(Rectangle rect, const char* label,
                                   int selected) {
    int hovered = CheckCollisionPointRec(GetMousePosition(), rect);
    Color background = selected ? CLITERAL(Color){82, 73, 61, 245}
        : (hovered ? CLITERAL(Color){72, 63, 51, 245}
                   : CLITERAL(Color){42, 36, 28, 235});
    Color text = selected ? COL_TEXT_YELLOW : COL_TEXT_WHITE;
    DrawRectangleRec(rect, background);
    DrawRectangleLinesEx(rect, 1, COL_PANEL_BORDER);
    Vector2 size = fc_osrs_measure_text(label);
    fc_osrs_draw_text(label, (int)(rect.x + (rect.width - size.x) * 0.5f),
                      (int)(rect.y + (rect.height - size.y) * 0.5f), text);
}

static void draw_fc_console_wave_grid(ViewerState* v, Rectangle body) {
    Vector2 mouse = GetMousePosition();
    DrawRectangleRec(body, CLITERAL(Color){18, 16, 13, 252});
    DrawRectangleLinesEx(body, 1, COL_PANEL_BORDER);
    fc_osrs_draw_text("Select Wave", (int)body.x + 4, (int)body.y + 3, COL_TEXT_YELLOW);
    fc_osrs_draw_text("click current selection to close",
                      (int)(body.x + body.width) - 155, (int)body.y + 4, COL_TEXT_DIM);
    for (int wave = 1; wave <= FC_NUM_WAVES; wave++) {
        Rectangle cell = fc_console_wave_cell_rect(body, wave);
        int current = wave == v->state.current_wave;
        int hovered = CheckCollisionPointRec(mouse, cell);
        Color bg = current ? CLITERAL(Color){60, 80, 40, 250}
            : (hovered ? CLITERAL(Color){72, 63, 51, 245}
                       : CLITERAL(Color){35, 30, 24, 240});
        DrawRectangleRec(cell, bg);
        DrawRectangleLinesEx(cell, 1, COL_PANEL_BORDER);
        char label[8];
        snprintf(label, sizeof(label), "%d", wave);
        fc_osrs_draw_text(label,
                          (int)(cell.x + (cell.width - fc_osrs_measure_text(label).x) * 0.5f),
                          (int)cell.y + 3, current ? COL_TEXT_YELLOW : COL_TEXT_WHITE);
    }
}

/* The NPC slots the console's target rows list: living NPCs in slot order. */
static int console_target_slots(const FcState* state,
                                int slots[FC_CONSOLE_NPC_ROWS]) {
    int count = 0;
    for (int ni = 0; ni < FC_MAX_NPCS && count < FC_CONSOLE_NPC_ROWS; ni++) {
        const FcNpc* npc = &state->npcs[ni];
        if (npc->active && !npc->is_dead) slots[count++] = ni;
    }
    return count;
}

/* Keep a diagnostics tab's scroll offset within its content; returns the
 * largest offset. */
static int clamp_console_scroll(ViewerState* v, int tab, int visible_height) {
    int max_scroll = v->console_content_height[tab] - visible_height;
    if (max_scroll < 0) max_scroll = 0;
    if (v->console_scroll[tab] < 0) v->console_scroll[tab] = 0;
    if (v->console_scroll[tab] > max_scroll) v->console_scroll[tab] = max_scroll;
    return max_scroll;
}

static void draw_fc_console_controls(ViewerState* v, Rectangle body) {
    Vector2 mouse = GetMousePosition();
    int x = (int)body.x + 4;
    char text[128];

    fc_osrs_draw_text("NPC Targets", x, (int)body.y + 3, COL_TEXT_YELLOW);
    int slots[FC_CONSOLE_NPC_ROWS];
    int shown = console_target_slots(&v->state, slots);
    for (int i = 0; i < shown; i++) {
        int ni = slots[i];
        FcNpc* npc = &v->state.npcs[ni];
        Rectangle row = fc_console_target_row_rect(body, i);
        int selected = ni == v->state.player.attack_target_idx;
        int hovered = CheckCollisionPointRec(mouse, row);
        if (selected || hovered) {
            DrawRectangleRec(row, selected
                ? CLITERAL(Color){82, 73, 61, 205}
                : CLITERAL(Color){52, 45, 35, 190});
        }
        snprintf(text, sizeof(text), "%s%s[%d] d:%d",
                 selected ? ">" : " ", npc_short_name(npc->npc_type), ni,
                 fc_distance_to_npc(v->state.player.x,
                                    v->state.player.y, npc));
        fc_osrs_draw_text(text, (int)row.x + 2, (int)row.y + 3,
                          selected ? COL_TEXT_YELLOW : COL_TEXT_WHITE);

        int bar_x = (int)row.x + 116;
        int bar_w = 116;
        float hp = npc->max_hp > 0
            ? (float)npc->current_hp / (float)npc->max_hp : 0.0f;
        if (hp < 0.0f) hp = 0.0f;
        if (hp > 1.0f) hp = 1.0f;
        DrawRectangle(bar_x, (int)row.y + 3, bar_w, 7, COL_HP_RED);
        DrawRectangle(bar_x, (int)row.y + 3,
                      (int)((float)bar_w * hp), 7, COL_HP_GREEN);
        snprintf(text, sizeof(text), "%d", npc->current_hp / 10);
        fc_osrs_draw_text(text, bar_x + bar_w + 4, (int)row.y + 2, COL_TEXT_DIM);
    }
    if (shown == 0)
        fc_osrs_draw_text("No NPCs alive", x + 2, (int)body.y + 20, COL_TEXT_DIM);

    int right_x = (int)body.x + 286;
    DrawLine(right_x - 5, (int)body.y + 2,
             right_x - 5, (int)(body.y + body.height) - 2,
             COL_PANEL_BORDER);
    fc_osrs_draw_text("Viewer Controls", right_x, (int)body.y + 3, COL_TEXT_YELLOW);

    char debug_label[24];
    snprintf(debug_label, sizeof(debug_label), "Debug: %s",
             v->dbg_flags ? "ON" : "OFF");
    draw_fc_console_button(fc_console_debug_button_rect(body),
                           debug_label, v->dbg_flags != 0);
    char god_label[24];
    snprintf(god_label, sizeof(god_label), "God: %s",
             v->godmode ? "ON" : "OFF");
    draw_fc_console_button(fc_console_god_button_rect(body),
                           god_label, v->godmode);

    Rectangle wave = fc_console_wave_button_rect(body);
    snprintf(text, sizeof(text), "Jump to Wave: %d  v",
             v->state.current_wave);
    draw_fc_console_button(wave, text, 0);

    fc_osrs_draw_text(v->policy_replay ? "Replay TPS" : "TPS Presets",
                      right_x, (int)body.y + 62, COL_TEXT_DIM);
    for (int i = 0; i < NUM_MANUAL_TPS_PRESETS; i++) {
        draw_fc_console_button(fc_console_tps_button_rect(body, i),
                               MANUAL_TPS_LABELS[i],
                               float_near(v->tps, MANUAL_TPS_PRESETS[i]));
    }

    if (v->console_wave_dropdown_open)
        draw_fc_console_wave_grid(v, body);
}

/* Console tabs 1-5 show debug tabs 0-4, scrolled by the mouse wheel. */
static void draw_fc_console_diagnostics(ViewerState* v, Rectangle body) {
    int debug_tab = v->console_tab - 1;
    int available_height = (int)body.height - 6;
    clamp_console_scroll(v, debug_tab, available_height);

    int content_y = (int)body.y + 3 - v->console_scroll[debug_tab];
    BeginScissorMode((int)body.x, (int)body.y,
                     (int)body.width, (int)body.height);
    int end_y = dbg_draw_panel_tabs(
        &v->state,
        &v->reward_breakdown, &v->reward_runtime,
        v->reward_config_loaded, v->reward_config_path,
        (int)body.x + 4, content_y, debug_tab);
    EndScissorMode();

    int height = end_y - content_y;
    v->console_content_height[debug_tab] = height > 0 ? height : 0;
    int max_scroll = clamp_console_scroll(v, debug_tab, available_height);
    if (max_scroll > 0) {
        int track_x = (int)(body.x + body.width) - 6;
        int track_y = (int)body.y + 3;
        int track_h = available_height;
        DrawRectangle(track_x, track_y, 4, track_h,
                      CLITERAL(Color){30, 26, 20, 230});
        float visible_fraction =
            (float)available_height / (float)height;
        int thumb_h = (int)((float)track_h * visible_fraction);
        if (thumb_h < 12) thumb_h = 12;
        float scroll_fraction = max_scroll > 0
            ? (float)v->console_scroll[debug_tab] / (float)max_scroll
            : 0.0f;
        int thumb_y = track_y +
            (int)((float)(track_h - thumb_h) * scroll_fraction);
        DrawRectangle(track_x, thumb_y, 4, thumb_h,
                      CLITERAL(Color){120, 110, 90, 255});
    }
}

static void draw_fc_console(ViewerState* v) {
    static const char* labels[FC_CONSOLE_TAB_COUNT] = {
        "Controls", "Player", "Obs", "Mask", "Reward", "Log"
    };
    Rectangle panel = fc_console_panel_rect();
    Rectangle body = fc_console_body_rect(panel);
    DrawRectangleRec(body, CLITERAL(Color){0, 0, 0, 76});
    if (v->console_tab == 0)
        draw_fc_console_controls(v, body);
    else
        draw_fc_console_diagnostics(v, body);

    for (int tab = 0; tab < FC_CONSOLE_TAB_COUNT; tab++) {
        Rectangle rect = fc_console_tab_rect(panel, tab);
        fc_ui_draw_asset(&v->ui.assets, "chat_tab_button_0", rect, WHITE);
        int selected = tab == v->console_tab;
        int hovered = CheckCollisionPointRec(GetMousePosition(), rect);
        DrawRectangleRec(rect, selected
            ? CLITERAL(Color){82, 73, 61, 96}
            : (hovered ? CLITERAL(Color){90, 78, 62, 72}
                       : CLITERAL(Color){0, 0, 0, 18}));
        if (selected)
            DrawLine((int)rect.x + 2, (int)(rect.y + rect.height) - 2,
                     (int)(rect.x + rect.width) - 2,
                     (int)(rect.y + rect.height) - 2,
                     COL_TEXT_YELLOW);
        fc_osrs_draw_text(labels[tab],
                          (int)(rect.x + (rect.width - fc_osrs_measure_text(labels[tab]).x) * 0.5f),
                          (int)rect.y + 7, selected ? COL_TEXT_YELLOW : COL_TEXT_WHITE);
    }
}

static int process_fc_console_input(ViewerState* v) {
    if (!v) return 0;
    Rectangle panel = fc_console_panel_rect();
    Vector2 mouse = GetMousePosition();
    if (!CheckCollisionPointRec(mouse, panel))
        return 0;

    int clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
    if (clicked) {
        for (int tab = 0; tab < FC_CONSOLE_TAB_COUNT; tab++) {
            if (!CheckCollisionPointRec(mouse,
                                        fc_console_tab_rect(panel, tab)))
                continue;
            v->console_tab = tab;
            v->console_wave_dropdown_open = 0;
            return 1;
        }
    }

    Rectangle body = fc_console_body_rect(panel);
    if (v->console_tab == 0 && clicked) {
        if (v->console_wave_dropdown_open) {
            for (int wave = 1; wave <= FC_NUM_WAVES; wave++) {
                if (!CheckCollisionPointRec(
                        mouse, fc_console_wave_cell_rect(body, wave)))
                    continue;
                if (wave != v->state.current_wave)
                    viewer_jump_to_wave(v, wave);
                v->console_wave_dropdown_open = 0;
                return 1;
            }
            v->console_wave_dropdown_open = 0;
            return 1;
        }

        if (CheckCollisionPointRec(mouse,
                                   fc_console_debug_button_rect(body))) {
            toggle_debug_overlay(v);
            return 1;
        }
        if (CheckCollisionPointRec(mouse,
                                   fc_console_god_button_rect(body))) {
            toggle_godmode(v);
            return 1;
        }
        if (CheckCollisionPointRec(mouse,
                                   fc_console_wave_button_rect(body))) {
            v->console_wave_dropdown_open = 1;
            return 1;
        }
        for (int i = 0; i < NUM_MANUAL_TPS_PRESETS; i++) {
            if (!CheckCollisionPointRec(
                    mouse, fc_console_tps_button_rect(body, i)))
                continue;
            set_viewer_tps(v, MANUAL_TPS_PRESETS[i]);
            return 1;
        }

        int slots[FC_CONSOLE_NPC_ROWS];
        int shown = console_target_slots(&v->state, slots);
        for (int i = 0; i < shown; i++) {
            if (CheckCollisionPointRec(
                    mouse, fc_console_target_row_rect(body, i))) {
                queue_player_attack_request(v, slots[i], mouse.x, mouse.y);
                fprintf(stderr, "CONSOLE CLICK -> ATTACK npc_idx=%d\n", slots[i]);
                return 1;
            }
        }
    }

    if (v->console_tab >= 1 && CheckCollisionPointRec(mouse, body)) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            int index = v->console_tab - 1;
            v->console_scroll[index] -= (int)wheel * 24;
            clamp_console_scroll(v, index, (int)body.height - 6);
        }
    }
    return 1;
}

static void draw_tex_fit(Texture2D tex, int dx, int dy, int dw, int dh,
                         Color tint) {
    if (tex.id == 0) return;
    Rectangle src = {0, 0, (float)tex.width, (float)tex.height};
    float sx = (float)dw / (float)tex.width;
    float sy = (float)dh / (float)tex.height;
    float scale = sx < sy ? sx : sy;
    int rw = (int)(tex.width * scale);
    int rh = (int)(tex.height * scale);
    Rectangle dst = {
        (float)(dx + (dw - rw) / 2),
        (float)(dy + (dh - rh) / 2),
        (float)rw,
        (float)rh,
    };
    DrawTexturePro(tex, src, dst, (Vector2){0, 0}, 0, tint);
}

static Rectangle fc_prayer_button_rect(Rectangle content, int index) {
    const int btn_h = 34;
    const int gap = 3;
    int x = (int)content.x + 8;
    int y = (int)content.y + 8 + 34 + index * (btn_h + gap);
    int w = (int)content.width - 16;
    if (w < 120) w = 120;
    return (Rectangle){(float)x, (float)y, (float)w, (float)btn_h};
}

static void draw_fc_prayer_tab(ViewerState* v, Rectangle content) {
    BeginScissorMode((int)content.x, (int)content.y,
                     (int)content.width, (int)content.height);

    DrawRectangleRec(content, COL_PANEL);

    FcPlayer* p = &v->state.player;
    int rendered_prayer = fc_actor_animation_render_prayer(
        &v->actor_animation, &v->state);
    int x = (int)content.x + 8;
    int by = (int)content.y + 8;
    int right = (int)(content.x + content.width) - 4;
    char b[64];

    snprintf(b, sizeof(b), "Prayer: %d / %d",
             p->current_prayer / 10, p->max_prayer / 10);
    text_s(b, x, by, COL_PRAY_BLUE);
    by += 16;

    if (rendered_prayer != PRAYER_NONE) {
        int resistance = 60 + 2 * p->prayer_bonus;
        snprintf(b, sizeof(b), "Drain rate: 12 / %d resist", resistance);
        text_s(b, x, by, COL_TEXT_DIM);
    } else {
        text_s("No prayer active", x, by, COL_TEXT_DIM);
    }
    by += 14;

    DrawLine((int)content.x + 4, by - 2, right, by - 2, COL_PANEL_BORDER);

    int no_points = (p->current_prayer <= 0);
    Vector2 mouse = GetMousePosition();
    const Color slot_empty = CLITERAL(Color){30, 26, 20, 255};
    const Color pray_active = CLITERAL(Color){60, 120, 200, 200};
    const Color tab_hover = CLITERAL(Color){72, 63, 51, 255};
    const Color pray_button = CLITERAL(Color){50, 44, 36, 255};
    for (int i = 0; i < FC_PROTECTION_PRAYER_COUNT; i++) {
        const FcProtectionPrayer* row = &FC_PROTECTION_PRAYERS[i];
        Rectangle br = fc_prayer_button_rect(content, i);
        int is_active = (rendered_prayer == row->prayer);
        int hovered = CheckCollisionPointRec(mouse, br);

        Color bg;
        if (no_points) {
            bg = slot_empty;
        } else if (is_active) {
            bg = pray_active;
        } else if (hovered) {
            bg = tab_hover;
        } else {
            bg = pray_button;
        }
        DrawRectangleRec(br, bg);
        DrawRectangleLinesEx(br, is_active ? 2 : 1,
                             is_active ? row->color : COL_PANEL_BORDER);

        Texture2D icon = is_active ? v->prayer_on[i] : v->prayer_off[i];
        Color icon_tint = no_points ? CLITERAL(Color){80,80,80,255} : WHITE;
        draw_tex_fit(icon, (int)br.x + 4, (int)br.y + 2, 30, 30, icon_tint);

        Color tc = no_points ? COL_TEXT_DIM
            : (is_active ? COL_TEXT_WHITE : row->color);
        text_s(row->label, (int)br.x + 38, (int)br.y + 7, tc);

        snprintf(b, sizeof(b), "[%d]", i + 1);
        text_s(b, (int)br.x + 38, (int)br.y + 21, COL_TEXT_DIM);
        if (is_active) {
            text_s("ACTIVE", (int)(br.x + br.width) - 44,
                   (int)br.y + 21, COL_TEXT_WHITE);
        }
    }

    by = (int)fc_prayer_button_rect(
        content, FC_PROTECTION_PRAYER_COUNT - 1).y + 34 + 9;
    snprintf(b, sizeof(b), "Prayer bonus: +%d", p->prayer_bonus);
    text_s(b, x, by, COL_TEXT_DIM);

    EndScissorMode();
}

static void draw_fc_side_overrides(ViewerState* v) {
    Rectangle content = fc_ui_side_content_rect(GetScreenWidth(), GetScreenHeight());
    if (v->ui.active_tab == FC_UI_TAB_PRAYER)
        draw_fc_prayer_tab(v, content);
}

static int process_fc_prayer_click(ViewerState* v) {
    if (!v || v->ui.active_tab != FC_UI_TAB_PRAYER)
        return 0;
    int left_click = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
    int right_click = IsMouseButtonPressed(MOUSE_BUTTON_RIGHT);
    if (!left_click && !right_click)
        return 0;

    Rectangle content = fc_ui_side_content_rect(GetScreenWidth(), GetScreenHeight());
    Vector2 mouse = GetMousePosition();
    if (!CheckCollisionPointRec(mouse, content))
        return 0;

    for (int i = 0; i < FC_PROTECTION_PRAYER_COUNT; i++) {
        const FcProtectionPrayer* row = &FC_PROTECTION_PRAYERS[i];
        Rectangle r = fc_prayer_button_rect(content, i);
        if (!CheckCollisionPointRec(mouse, r))
            continue;
        if (right_click) {
            int active = fc_actor_animation_render_prayer(
                &v->actor_animation, &v->state) == row->prayer;
            fc_ui_open_prayer_context(&v->ui, mouse, i, row->name, active);
        } else {
            queue_viewer_prayer_button(v, row);
        }
        return 1;
    }

    return 1;
}
/* ======================================================================== */
/* Main                                                                      */
/* ======================================================================== */

/* The same viewer serves the standalone game, optional CPU compatibility
 * replay, and Puffer's puf_render(). No graphics are allocated by training. */
static void fc_viewer_destroy(ViewerState* v) {
    if (!v) return;
    for (int i = 0; i < FC_PROTECTION_PRAYER_COUNT; i++) {
        if (v->prayer_on[i].id > 0) UnloadTexture(v->prayer_on[i]);
        if (v->prayer_off[i].id > 0) UnloadTexture(v->prayer_off[i]);
    }
    for (int i = 0; i < FC_CLICK_CROSS_FRAME_COUNT * 2; i++) {
        if (v->click_cross_tex[i].id > 0)
            UnloadTexture(v->click_cross_tex[i]);
    }
    fc_combat_presentation_destroy(v->combat_presentation);
    fc_actor_animation_shutdown(&v->actor_animation);
    if (v->object_anim_runtimes) {
        for (int i = 0; i < v->object_anim_runtime_count; i++) {
            if (v->object_anim_runtimes[i].anim_state)
                anim_model_state_free(v->object_anim_runtimes[i].anim_state);
        }
        free(v->object_anim_runtimes);
    }
    if (v->anim_cache) anim_cache_free(v->anim_cache);
    fc_player_appearance_free(&v->appearance);
    models_free(v->npc_models);
    models_free(v->object_anim_models);
    fc_animated_atlas_unload(&v->atlas);
    if (v->object_anims) object_anims_free(v->object_anims);
    objects_free(v->objects);
    fc_minimap_scene_free(&v->minimap_scene);
    terrain_free(v->terrain);
    fc_osrs_text_shutdown();
    fc_ui_shutdown(&v->ui);
    CloseWindow();
    fc_destroy(&v->state);
    free(v);
}

static ViewerState* fc_viewer_create(int replay) {
#if defined(__linux__)
    /* Raylib's X11 build can crash inside InitWindow without DISPLAY. */
    const char* display = getenv("DISPLAY");
    if (!display || !display[0]) {
        fprintf(stderr, "error: no graphical DISPLAY is configured; provide X11 "
                "display access, or use --headless for checkpoint evaluation\n");
        return NULL;
    }
#endif
    fprintf(stderr, "=== Fight Caves Viewer ===\n");
    /* Keep graphical diagnostics off the native evaluator's stdout. */
    if (replay) {
        SetTraceLogCallback(viewer_trace_log_to_stderr);
        SetTraceLogLevel(LOG_WARNING);
    }
    SetConfigFlags(FLAG_WINDOW_RESIZABLE|FLAG_MSAA_4X_HINT);
    InitWindow(DEFAULT_WINDOW_W, DEFAULT_WINDOW_H, replay
               ? "Fight Caves RL — Checkpoint Viewer"
               : "Fight Caves RL — Playable Viewer");
    SetExitKey(KEY_NULL); /* Escape first dismisses menus; handled below. */
    if (!IsWindowReady()) {
        fprintf(stderr,
                "error: viewer window initialization failed; verify the "
                "graphical display and OpenGL driver\n");
        return NULL;
    }
    SetTargetFPS(60);

    ViewerState* v = calloc(1, sizeof(*v));
    if (!v) {
        fprintf(stderr, "error: cannot allocate Fight Caves viewer state\n");
        CloseWindow();
        return NULL;
    }
    fc_init(&v->state);
    fc_ui_init(&v->ui);
    if (!fc_osrs_text_init(v->ui.assets.small_font)) {
        fprintf(stderr,
                "error: required OSRS viewer fonts failed to load\n");
        fc_viewer_destroy(v);
        return NULL;
    }
    int item_icons_ready = load_fc_ui_item_icons(v);
    v->paused = 1; v->tps = POLICY_REPLAY_BASE_TPS;
    v->cam_yaw = 0; v->cam_pitch = 0.8f; v->cam_dist = 30;
    v->camera_locked = 1;
    v->camera.up = (Vector3){0,1,0}; v->camera.fovy = 32;
    v->camera.projection = CAMERA_PERSPECTIVE;
    v->camera.target = (Vector3){FC_ARENA_WIDTH * 0.5f, 0.5f, -(FC_ARENA_HEIGHT * 0.5f)};

    /* The terrain mesh is just the ground surface, in the original cache
     * colors; the red/black lava pattern comes from the objects mesh. */
    v->terrain = terrain_load("fightcaves.terrain");
    /* OSRS rasterizes the current 104x104 scene from cache terrain and
     * locations into a 512x512 minimap. This asset contains the Fight Caves
     * mapsquare centered in that same scene format; runtime only crops and
     * rotates it around the player. */
    Image minimap_image = fc_load_image_asset("fightcaves.minimap.png");
    if (minimap_image.data) {
        Color* minimap_pixels = LoadImageColors(minimap_image);
        if (!minimap_pixels || !fc_minimap_scene_load_pixels(
                &v->minimap_scene, minimap_pixels,
                minimap_image.width, minimap_image.height)) {
            fprintf(stderr, "error: Fight Caves minimap failed to load\n");
        } else {
            fprintf(stderr,
                    "minimap: loaded cache scene raster %dx%d\n",
                    minimap_image.width, minimap_image.height);
        }
        UnloadImageColors(minimap_pixels);
        UnloadImage(minimap_image);
    } else {
        fprintf(stderr, "error: missing fightcaves.minimap.png\n");
    }
    if (!fc_animated_atlas_load(&v->atlas, "fightcaves.atlas",
                                "fightcaves.tanim"))
        fprintf(stderr, "error: texture atlas failed to load\n");
    v->objects = objects_load("fightcaves.objects", v->atlas.texture);
    v->object_anims = object_anims_load("fightcaves.oanim");
    v->object_anim_models = models_load("fightcaves.object_anim.models",
                                        v->atlas.texture);
    if (v->object_anims && v->object_anims->count > 0) {
        v->object_anim_runtimes = (ObjectAnimRuntime*)calloc(
            (size_t)v->object_anims->count, sizeof(*v->object_anim_runtimes));
        if (v->object_anim_runtimes)
            v->object_anim_runtime_count = v->object_anims->count;
    }
    if (!v->terrain) v->show_grid = 1;

    /* NPC models: the scene draws and picks every NPC type with its own. */
    v->npc_models = models_load("fc_npcs.models", (Texture2D){0});
    if (!v->npc_models) fprintf(stderr, "error: NPC models failed to load\n");
    int npc_models_ready = 1;
    for (int type = NPC_NONE + 1; type < NPC_TYPE_COUNT; type++)
        if (!npc_model_find(v->npc_models, type)) npc_models_ready = 0;

    /* Load composable player body and equipment models. */
    if (!fc_player_appearance_load(&v->appearance)) {
        fprintf(stderr, "Required player appearance assets are missing or invalid.\n");
        fc_viewer_destroy(v);
        return NULL;
    }

    /* Load the animation cache shared by actor and combat presentation. */
    v->anim_cache = anim_cache_load("fc_all.anims");
    v->combat_presentation = fc_combat_presentation_create(v->atlas.texture);
    if (!v->combat_presentation)
        fprintf(stderr, "error: combat presentation initialization failed\n");

    /* Protection prayer sprites for the Prayer tab and the overhead icon. */
    int prayer_icons_loaded = 0;
    for (int i = 0; i < FC_PROTECTION_PRAYER_COUNT; i++) {
        char path[64];
        snprintf(path, sizeof(path), "data/sprites/ui/prayeron_%d.png",
                 FC_PROTECTION_PRAYERS[i].sprite);
        v->prayer_on[i] = fc_load_texture_asset(path);
        snprintf(path, sizeof(path), "data/sprites/ui/prayeroff_%d.png",
                 FC_PROTECTION_PRAYERS[i].sprite);
        v->prayer_off[i] = fc_load_texture_asset(path);
        prayer_icons_loaded += (v->prayer_on[i].id > 0) + (v->prayer_off[i].id > 0);
    }
    fprintf(stderr, "Prayer icons loaded: %d/%d\n", prayer_icons_loaded,
            2 * FC_PROTECTION_PRAYER_COUNT);

    /* Native b237 click crosses: frames 0-3 are movement (yellow), frames
     * 4-7 are interaction (red). RuneC advances one frame every 100 ms. */
    int click_cross_loaded = 0;
    {
        for (int i = 0; i < FC_CLICK_CROSS_FRAME_COUNT * 2; i++) {
            char path[64];
            snprintf(path, sizeof(path),
                     "data/sprites/ui/cross_%d.png", i);
            v->click_cross_tex[i] = fc_load_texture_asset(path);
            if (v->click_cross_tex[i].id > 0) {
                SetTextureFilter(v->click_cross_tex[i], TEXTURE_FILTER_POINT);
                click_cross_loaded++;
            }
        }
        fprintf(stderr, "Click cross sprites loaded: %d/8\n",
                click_cross_loaded);
        if (click_cross_loaded != FC_CLICK_CROSS_FRAME_COUNT * 2)
            fprintf(stderr, "error: required click cross sprites failed to load\n");
    }

    int required_resources_ready = 1;
#define REQUIRE_VIEWER_RESOURCE(condition, description) do {                 \
        if (!(condition)) {                                                  \
            fprintf(stderr, "error: required viewer resource failed: %s\n", \
                    description);                                            \
            required_resources_ready = 0;                                    \
        }                                                                    \
    } while (0)
    REQUIRE_VIEWER_RESOURCE(v->ui.assets.missing_count == 0,
                            "UI sprites");
    REQUIRE_VIEWER_RESOURCE(v->ui.assets.font.texture.id > 0 &&
                            v->ui.assets.small_font.texture.id > 0,
                            "UI fonts");
    REQUIRE_VIEWER_RESOURCE(v->ui.minimap_texture.id > 0,
                            "minimap render texture");
    REQUIRE_VIEWER_RESOURCE(item_icons_ready, "Fight Caves item icons");
    REQUIRE_VIEWER_RESOURCE(v->terrain, "terrain mesh");
    REQUIRE_VIEWER_RESOURCE(v->minimap_scene.ready, "minimap scene raster");
    REQUIRE_VIEWER_RESOURCE(v->atlas.texture.id > 0, "texture atlas");
    REQUIRE_VIEWER_RESOURCE(v->objects, "textured terrain objects");
    REQUIRE_VIEWER_RESOURCE(v->object_anims, "object animation placements");
    REQUIRE_VIEWER_RESOURCE(v->object_anim_models, "animated object models");
    REQUIRE_VIEWER_RESOURCE(!v->object_anims || v->object_anims->count == 0 ||
                            v->object_anim_runtimes,
                            "object animation runtime allocation");
    REQUIRE_VIEWER_RESOURCE(npc_models_ready, "NPC models");
    REQUIRE_VIEWER_RESOURCE(v->appearance.parts,
                            "player equipment and body models");
    REQUIRE_VIEWER_RESOURCE(v->anim_cache, "actor animation data");
    REQUIRE_VIEWER_RESOURCE(
        fc_combat_presentation_ready(v->combat_presentation),
        "projectile, spot-animation, healthbar, and hitsplat data");
    REQUIRE_VIEWER_RESOURCE(prayer_icons_loaded == 2 * FC_PROTECTION_PRAYER_COUNT,
                            "Prayer tab and overhead icons");
    REQUIRE_VIEWER_RESOURCE(
        click_cross_loaded == FC_CLICK_CROSS_FRAME_COUNT * 2,
        "click cross sprites");
#undef REQUIRE_VIEWER_RESOURCE
    if (!required_resources_ready) {
        fprintf(stderr,
                "error: viewer startup aborted instead of using reduced "
                "graphics; reinstall assets with: "
                "python3 ocean/fight_caves/tools.py setup --force\n");
        fc_viewer_destroy(v);
        return NULL;
    }

    v->policy_replay = replay;
    v->paused = !replay;
    return v;
}

static void fc_viewer_ingest_tick(ViewerState* v) {
    fc_fill_render_events(&v->state, &v->render_events);
    fc_actor_animation_ingest_tick(&v->actor_animation, &v->state,
                                   &v->render_events);
    fc_actor_animation_ingest_events(
        &v->actor_animation, &v->render_events, v->anim_cache,
        fc_player_visual_profile(&v->state.player), v->tps);

    /* Debug event log — record events from this tick */
    dbg_log_tick(&v->state);

    /* Snap prev positions for newly spawned NPCs so they don't fly.
     * An NPC that wasn't active last tick but is now = new spawn. */
    for (int ni = 0; ni < FC_MAX_NPCS; ni++) {
        if (v->state.npcs[ni].active &&
            !fc_actor_animation_previous_npc_active(
                &v->actor_animation, ni)) {
            fc_combat_presentation_clear_npc_healthbar(
                v->combat_presentation, ni);
        }
    }

    fc_fill_render_entities(&v->state, v->entities, &v->entity_count);

    FcCombatPresentationContext combat_context = combat_presentation_context(v);
    fc_combat_presentation_ingest_tick(v->combat_presentation,
                                        &combat_context);
}

/* Return 1 when a simulation tick is due, 0 for another display frame,
 * and -1 when the window closes. Policy replay never steps its state copy. */
static int fc_viewer_frame(ViewerState* v) {
    if (WindowShouldClose()) return -1;
    int ui_capture = 0;
    /* Age the previous click before capturing this frame's input. A newly
     * clicked cross must start at frame zero, even after a slow frame. */
    fc_click_feedback_update(&v->click_feedback, GetFrameTime());

    /* Global keys (always active) */
    if (IsKeyPressed(KEY_Q)) return -1;
    if (IsKeyPressed(KEY_ESCAPE) && !v->ui.context_open) return -1;
    if (IsKeyPressed(KEY_SPACE)) v->paused = !v->paused;
    if (IsKeyPressed(KEY_RIGHT)) v->step_once = 1;
    if (v->policy_replay) {
        if (IsKeyPressed(KEY_ONE)) set_policy_replay_speed(v, 0);
        if (IsKeyPressed(KEY_TWO)) set_policy_replay_speed(v, 1);
        if (!IsKeyDown(KEY_LEFT_SHIFT) && !IsKeyDown(KEY_RIGHT_SHIFT) &&
            IsKeyPressed(KEY_FOUR)) set_policy_replay_speed(v, 2);
        if (IsKeyPressed(KEY_ZERO)) set_policy_replay_speed(v, 3);
        if (IsKeyPressed(KEY_UP))
            set_policy_replay_speed(v, replay_speed_index(v) + 1);
        if (IsKeyPressed(KEY_DOWN))
            set_policy_replay_speed(v, replay_speed_index(v) - 1);
    }
    if (!v->policy_replay && IsKeyPressed(KEY_R)) reset_ep(v);
    if (IsKeyPressed(KEY_L)) {
        if (v->camera_locked) {
            v->camera.target = camera_follow_target(v);
        }
        v->camera_locked = !v->camera_locked;
    }

    /* Toggle keys */
    if (IsKeyPressed(KEY_G)) v->show_grid = !v->show_grid;
    if (IsKeyPressed(KEY_C)) v->show_collision = !v->show_collision;
    /* O: cycle debug overlay modes. O=all on/off, Shift+O=cycle sub-modes */
    if (IsKeyPressed(KEY_O)) {
        if (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) {
            /* Cycle through individual modes */
            if (v->dbg_flags == 0) v->dbg_flags = DBG_COLLISION;
            else if (v->dbg_flags == DBG_COLLISION) v->dbg_flags = DBG_LOS;
            else if (v->dbg_flags == DBG_LOS) v->dbg_flags = DBG_PATH | DBG_RANGE;
            else v->dbg_flags = 0;
        } else {
            /* Toggle all on/off */
            toggle_debug_overlay(v);
        }
    }
    /* D: the same toggle as O */
    if (IsKeyPressed(KEY_D)) toggle_debug_overlay(v);
    /* Camera presets */
    if ((!v->policy_replay && IsKeyPressed(KEY_FOUR)) ||
        (v->policy_replay &&
         (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) &&
         IsKeyPressed(KEY_FOUR))) {
        v->cam_yaw=0; v->cam_pitch=1.35f; v->cam_dist=120;
    }
    if ((!v->policy_replay && IsKeyPressed(KEY_FIVE)) ||
        (v->policy_replay &&
         (IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT)) &&
         IsKeyPressed(KEY_FIVE))) {
        v->cam_yaw=0; v->cam_pitch=0.6f; v->cam_dist=50;
    }

    sync_fc_ui(v);
    if (v->ui.context_open) {
        ui_capture = fc_ui_handle_input(&v->ui, GetScreenWidth(), GetScreenHeight());
        handle_fc_ui_intent(v);
    } else {
        ui_capture = process_fc_prayer_click(v);
        if (!ui_capture)
            ui_capture = process_fc_console_input(v);
        if (!ui_capture) {
            ui_capture = fc_ui_handle_input(&v->ui, GetScreenWidth(), GetScreenHeight());
            handle_fc_ui_intent(v);
        } else {
            v->ui.last_intent.kind = FC_UI_INTENT_NONE;
        }
    }

    /* RuneC: open on right press; a real drag dismisses the menu and
     * retains the viewer's existing camera gesture. No game action fires. */
    if (!ui_capture && IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
        v->scene_right_tracking = 1;
        v->scene_right_dragged = 0;
        v->scene_right_start = GetMousePosition();
        open_scene_context_menu(v);
        ui_capture = 1;
    }
    if (v->scene_right_tracking && IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        Vector2 mouse = GetMousePosition();
        float dx = mouse.x - v->scene_right_start.x;
        float dy = mouse.y - v->scene_right_start.y;
        if (dx * dx + dy * dy > 9.0f) v->scene_right_dragged = 1;
        if (v->scene_right_dragged) {
            fc_ui_close_context(&v->ui);
            Vector2 d = GetMouseDelta();
            v->cam_yaw += d.x*0.005f; v->cam_pitch -= d.y*0.005f;
            if (v->cam_pitch < 0.1f) v->cam_pitch = 0.1f;
            if (v->cam_pitch > 1.4f) v->cam_pitch = 1.4f;
        }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_RIGHT)) {
        v->scene_right_tracking = v->scene_right_dragged = 0;
    }
    float wh = GetMouseWheelMove();
    if (!ui_capture && wh != 0) {
        v->cam_dist *= (wh > 0) ? (1.0f/1.15f) : 1.15f;
        if (v->cam_dist < 5) v->cam_dist = 5;
        if (v->cam_dist > 300) v->cam_dist = 300;
    }

    /* Tick processing */
    int tick = 0;
    if (!v->paused) {
        v->tick_acc += GetFrameTime() * (float)v->tps;
        if (v->tick_acc >= 1.0f) {
            v->tick_acc = fmodf(v->tick_acc, 1.0f);
            tick = 1;
        }
    }
    if (v->step_once) { tick = 1; v->step_once = 0; }

    /* Capture clicks and key presses EVERY frame (60fps).
     * These set routes/targets/buffers on the player struct.
     * The tick loop reads them when the next tick fires. */
    if (!v->policy_replay && v->state.terminal == TERMINAL_NONE) {
        process_human_clicks(v, ui_capture);
        process_human_keys(v);
    }

    if (!v->policy_replay && tick && v->state.terminal == TERMINAL_NONE) {
        build_human_actions(v);

        fc_actor_animation_capture_tick_start(&v->actor_animation,
                                              &v->state);

        /* Step simulation */
        fc_step(&v->state, v->actions);
        if (v->actions[5] > 0 && v->actions[6] > 0)
            fc_click_feedback_accept_move_tick(&v->click_feedback,
                                               &v->state);
        fc_click_feedback_sync(&v->click_feedback, &v->state);

        /* Playable-viewer test aid only. The simulator has already
         * resolved the hit; keep the local session alive at one HP. */
        if (v->godmode &&
            v->state.terminal == TERMINAL_PLAYER_DEATH) {
            v->state.player.current_hp = 10;
            v->state.terminal = TERMINAL_NONE;
        }
        update_reward_breakdown(v);
        fc_viewer_ingest_tick(v);

        if (v->state.terminal != TERMINAL_NONE) v->paused = 1;
    }

    float frame_dt = GetFrameTime();
    sync_player_appearance(v);
    if (v->item_message_seconds > 0) v->item_message_seconds -= frame_dt;
    FcCombatPresentationContext combat_context = combat_presentation_context(v);
    unsigned char deferred_deaths[FC_MAX_NPCS];
    fc_combat_presentation_deferred_deaths(
        v->combat_presentation, &v->state, deferred_deaths);
    fc_actor_animation_update_scene(
        &v->actor_animation, &v->state, v->anim_cache, v->tps, frame_dt,
        !v->paused || v->policy_replay, deferred_deaths);
    fc_animated_atlas_update(&v->atlas, frame_dt);
    fc_combat_presentation_update(v->combat_presentation,
                                  &combat_context, frame_dt);
    fc_combat_presentation_deferred_deaths(
        v->combat_presentation, &v->state, deferred_deaths);
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        if (!v->state.npcs[i].active && !v->state.npcs[i].died_this_tick)
            fc_combat_presentation_clear_npc_healthbar(
                v->combat_presentation, i);
    }
    fc_actor_animation_update_models(
        &v->actor_animation, &v->state, v->appearance.model, v->npc_models,
        v->anim_cache, v->tps, frame_dt, deferred_deaths);
    /* Draw */
    BeginDrawing();
    ClearBackground(COL_BG);
    draw_scene(v);
    sync_fc_ui(v);
    sync_fc_ui_minimap(v);
    fc_ui_draw(&v->ui, GetScreenWidth(), GetScreenHeight());
    draw_fc_side_overrides(v);
    draw_fc_console(v);
    draw_click_cross(v);
    if (v->item_message_seconds > 0) {
        DrawRectangle(8, GetScreenHeight() - 34, 490, 26, (Color){20, 16, 12, 240});
        text_s(v->item_message, 16, GetScreenHeight() - 29, YELLOW);
    }
    /* Menus must cover the console and prayer overrides, not sit behind them. */
    fc_ui_draw_context(&v->ui);

    EndDrawing();
    return tick;
}

/* Consume the latest authoritative transition, including a terminal snapshot
 * saved before Puffer's same-step autoreset. State is copied, never advanced. */
static void fc_viewer_present_pending(ViewerState* v) {
    if (!v->pending_frame) return;
    if (v->pending_state.rng_seed != v->state.rng_seed ||
        v->pending_state.tick <= v->state.tick) {
        v->state = v->reset_state;
        fc_viewer_reset_presentation(v);
    }
    fc_actor_animation_capture_tick_start(&v->actor_animation, &v->state);
    v->state = v->pending_state;
    v->reward_runtime = v->pending_reward_runtime;
    v->reward_breakdown = v->pending_reward_breakdown;
    v->reward_breakdown_tick = v->state.tick;
    memcpy(v->actions, v->pending_actions, sizeof(v->actions));
    fc_viewer_ingest_tick(v);
    v->pending_frame = 0;
}

int fc_viewer_main(int argc, char** argv) {
    const char* screenshot_path = NULL;
    int start_wave = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc) {
            screenshot_path = argv[++i];
        } else if (strcmp(argv[i], "--start-wave") == 0 && i + 1 < argc) {
            char* end;
            long wave = strtol(argv[++i], &end, 10);
            if (*end || wave < 1 || wave > 63) {
                fprintf(stderr, "--start-wave requires a wave from 1 to 63.\n");
                return EXIT_FAILURE;
            }
            start_wave = (int)wave;
        } else {
            int help = strcmp(argv[i], "--help") == 0;
            fprintf(help ? stdout : stderr,
                "Usage: %s [--start-wave 1..63] [--screenshot FILE]\n"
                "For policy replay, use the native trainer: ./puffer eval CHECKPOINT\n",
                argv[0]);
            return help ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    ViewerState* v = fc_viewer_create(0);
    if (!v) return EXIT_FAILURE;
    v->start_wave = start_wave;
    reset_ep(v);
    int frame_count = 0;
    while (fc_viewer_frame(v) >= 0) {
        if (screenshot_path && ++frame_count == 6) {
            TakeScreenshot(screenshot_path);
            break;
        }
    }
    fc_viewer_destroy(v);
    return EXIT_SUCCESS;
}
