#ifndef OSRS_MODELS_H
#define OSRS_MODELS_H

#if __has_include("raylib.h")
#include "raylib.h"
#elif __has_include("raylib-5.5_macos/include/raylib.h")
#include "raylib-5.5_macos/include/raylib.h"
#else
#error "raylib.h not found"
#endif
#include "osrs_assets.h"
#include "osrs_asset_raylib.h"
#include "osrs_binary_io.h"
#include "osrs_types.h"
#include "osrs_items.h"
#include "data/item_models.h"
#include "data/player_models.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MDL4_MAGIC 0x4D444C34
#define TANM_MAGIC 0x4D4E4154
#define TANM_VERSION 1
#define MODEL_CACHE_DENSE_INDEX_LIMIT 0x100000u

typedef struct {
    uint32_t texture_id;
    uint16_t x, y, w, h;
    uint8_t direction;
    uint8_t speed;
    uint16_t pad;
} ModelTextureAnimRow;

typedef struct {
    uint32_t model_id;
    Mesh mesh;
    Model model;

    int16_t*  base_vertices;
    uint8_t*  vertex_skins;
    uint16_t* face_indices;
    uint8_t*  face_priorities;
    uint8_t*  base_face_alphas;
    uint8_t*  face_alpha_labels;
    uint16_t  base_vert_count;

} OsrsModel;

typedef struct {
    OsrsModel* models;
    int* index_by_id;
    int count;
    size_t index_limit;
    Texture2D atlas_texture;
    unsigned char* atlas_base_pixels;
    unsigned char* atlas_pixels;
    ModelTextureAnimRow* texture_anims;
    int atlas_width;
    int atlas_height;
    int texture_anim_count;
    float texture_anim_ticks;
    int has_atlas;
} ModelCache;

typedef enum {
    OSRS_MODEL_APPEND_OK,
    OSRS_MODEL_APPEND_EMPTY,
    OSRS_MODEL_APPEND_BASE_VERTEX_OVERFLOW,
    OSRS_MODEL_APPEND_FACE_OVERFLOW,
} OsrsModelAppendResult;

static const char* osrs_model_append_result_name(OsrsModelAppendResult result) {
    switch (result) {
        case OSRS_MODEL_APPEND_OK:
            return "ok";
        case OSRS_MODEL_APPEND_EMPTY:
            return "empty model";
        case OSRS_MODEL_APPEND_BASE_VERTEX_OVERFLOW:
            return "base vertex overflow";
        case OSRS_MODEL_APPEND_FACE_OVERFLOW:
            return "face overflow";
    }
    return "unknown append result";
}

static OsrsModelAppendResult osrs_model_append_check(
    int current_base_vert_count,
    int current_face_count,
    const OsrsModel* model,
    int base_vert_capacity,
    int face_capacity
) {
    if (!model || !model->base_vertices || model->base_vert_count == 0) {
        return OSRS_MODEL_APPEND_EMPTY;
    }
    if (current_base_vert_count < 0 || current_face_count < 0 ||
            base_vert_capacity < 0 || face_capacity < 0 ||
            model->mesh.triangleCount < 0) {
        fprintf(stderr, "osrs_model_append_check: invalid composite geometry counts\n");
        abort();
    }

    int model_base_vert_count = (int)model->base_vert_count;
    int model_face_count = model->mesh.triangleCount;
    if (model_base_vert_count > base_vert_capacity ||
            current_base_vert_count > base_vert_capacity - model_base_vert_count) {
        return OSRS_MODEL_APPEND_BASE_VERTEX_OVERFLOW;
    }
    if (model_face_count > face_capacity ||
            current_face_count > face_capacity - model_face_count) {
        return OSRS_MODEL_APPEND_FACE_OVERFLOW;
    }
    return OSRS_MODEL_APPEND_OK;
}

static int model_cache_companion_path(
    char* out,
    size_t cap,
    const char* path,
    const char* suffix
) {
    if (!out || cap == 0 || !path || !suffix) return 0;
    const char* dot = strrchr(path, '.');
    size_t stem_len = dot ? (size_t)(dot - path) : strlen(path);
    int n = snprintf(out, cap, "%.*s%s", (int)stem_len, path, suffix);
    return n > 0 && (size_t)n < cap;
}

static void model_cache_load_texture_anims(ModelCache* cache, const char* model_path) {
    if (!cache || !model_path) return;

    char tanm_path[1024];
    if (!model_cache_companion_path(tanm_path, sizeof(tanm_path), model_path, ".tanim"))
        return;

    FILE* f = osrs_asset_fopen(tanm_path, "rb");
    if (!f) return;

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    osrs_read_exact(f, &magic, sizeof(magic), 1, tanm_path, "tanim magic");
    osrs_read_exact(f, &version, sizeof(version), 1, tanm_path, "tanim version");
    osrs_read_exact(f, &count, sizeof(count), 1, tanm_path, "tanim count");
    if (magic != TANM_MAGIC || version != TANM_VERSION) {
        fprintf(stderr, "model_cache_load: bad texture anim file %s\n", tanm_path);
        abort();
    }

    cache->texture_anims = (ModelTextureAnimRow*)osrs_calloc_or_abort(
        count, sizeof(ModelTextureAnimRow), "model texture animation rows");
    cache->texture_anim_count = (int)count;

    for (uint32_t i = 0; i < count; i++) {
        ModelTextureAnimRow* row = &cache->texture_anims[i];
        osrs_read_exact(f, &row->texture_id, sizeof(row->texture_id), 1,
            tanm_path, "tanim texture id");
        osrs_read_exact(f, &row->x, sizeof(row->x), 1, tanm_path, "tanim x");
        osrs_read_exact(f, &row->y, sizeof(row->y), 1, tanm_path, "tanim y");
        osrs_read_exact(f, &row->w, sizeof(row->w), 1, tanm_path, "tanim width");
        osrs_read_exact(f, &row->h, sizeof(row->h), 1, tanm_path, "tanim height");
        osrs_read_exact(f, &row->direction, sizeof(row->direction), 1,
            tanm_path, "tanim direction");
        osrs_read_exact(f, &row->speed, sizeof(row->speed), 1, tanm_path, "tanim speed");
        osrs_read_exact(f, &row->pad, sizeof(row->pad), 1, tanm_path, "tanim pad");
    }
    fclose(f);

    fprintf(stderr, "model_cache_load: loaded %d texture anim rows from %s\n",
        cache->texture_anim_count, tanm_path);
}

static void model_cache_update_texture_anims(ModelCache* cache, float dt) {
    if (!cache || !cache->atlas_pixels || !cache->atlas_base_pixels ||
            cache->atlas_texture.id <= 0 || cache->texture_anim_count <= 0) {
        return;
    }

    cache->texture_anim_ticks += dt * 50.0f;
    size_t total = (size_t)cache->atlas_width * (size_t)cache->atlas_height * 4;
    memcpy(cache->atlas_pixels, cache->atlas_base_pixels, total);

    for (int r = 0; r < cache->texture_anim_count; r++) {
        ModelTextureAnimRow* row = &cache->texture_anims[r];
        if (row->w == 0 || row->h == 0 || row->speed == 0) continue;
        if ((int)row->x + (int)row->w > cache->atlas_width ||
                (int)row->y + (int)row->h > cache->atlas_height) {
            continue;
        }

        int shift = (int)(cache->texture_anim_ticks * (float)row->speed);
        if (row->direction == 1 || row->direction == 3) {
            int pad = row->pad;
            if (pad * 2 >= row->h) pad = 0;
            int center_h = row->h - pad * 2;
            if (center_h <= 0) center_h = row->h;
            shift %= center_h;
            if (row->direction == 1) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                int sy = (y - pad + shift) % center_h;
                if (sy < 0) sy += center_h;
                sy += pad;
                for (int x = 0; x < row->w; x++) {
                    size_t dst = ((size_t)(row->y + y) * (size_t)cache->atlas_width +
                        (size_t)(row->x + x)) * 4;
                    size_t src = ((size_t)(row->y + sy) * (size_t)cache->atlas_width +
                        (size_t)(row->x + x)) * 4;
                    memcpy(&cache->atlas_pixels[dst], &cache->atlas_base_pixels[src], 4);
                }
            }
        } else if (row->direction == 2 || row->direction == 4) {
            shift %= row->w;
            if (row->direction == 2) shift = -shift;
            for (int y = 0; y < row->h; y++) {
                for (int x = 0; x < row->w; x++) {
                    int sx = (x + shift) % row->w;
                    if (sx < 0) sx += row->w;
                    size_t dst = ((size_t)(row->y + y) * (size_t)cache->atlas_width +
                        (size_t)(row->x + x)) * 4;
                    size_t src = ((size_t)(row->y + y) * (size_t)cache->atlas_width +
                        (size_t)(row->x + sx)) * 4;
                    memcpy(&cache->atlas_pixels[dst], &cache->atlas_base_pixels[src], 4);
                }
            }
        }
    }

    UpdateTexture(cache->atlas_texture, cache->atlas_pixels);
}

static Texture2D model_cache_load_atlas(ModelCache* cache, const char* model_path) {
    char atlas_path[1024];
    if (!model_cache_companion_path(atlas_path, sizeof(atlas_path), model_path, ".atlas")) {
        return (Texture2D){0};
    }
    if (!osrs_asset_exists(atlas_path)) return (Texture2D){0};

    Image image = osrs_asset_load_atlas_image(atlas_path);
    Texture2D texture = LoadTextureFromImage(image);
    if (texture.id > 0) SetTextureFilter(texture, TEXTURE_FILTER_POINT);
    if (texture.id > 0 && cache) {
        size_t pixel_count = (size_t)image.width * (size_t)image.height * 4;
        cache->atlas_width = image.width;
        cache->atlas_height = image.height;
        cache->atlas_base_pixels = (unsigned char*)osrs_malloc_or_abort(
            pixel_count, "model atlas base pixels");
        cache->atlas_pixels = (unsigned char*)osrs_malloc_or_abort(
            pixel_count, "model atlas working pixels");
        memcpy(cache->atlas_base_pixels, image.data, pixel_count);
        memcpy(cache->atlas_pixels, image.data, pixel_count);
        model_cache_load_texture_anims(cache, model_path);
    }
    fprintf(stderr, "model_cache_load: loaded atlas %dx%d from %s\n",
        image.width, image.height, atlas_path);
    UnloadImage(image);
    return texture;
}

static size_t model_cache_index_limit_or_abort(
    FILE* f,
    const uint32_t* offsets,
    uint32_t count,
    const char* path
) {
    size_t index_limit = 0;
    for (uint32_t i = 0; i < count; i++) {
        osrs_seek_or_abort(f, (long)offsets[i], path);
        uint32_t model_id = 0;
        osrs_read_exact(f, &model_id, sizeof(model_id), 1, path, "model id");
        if (model_id < MODEL_CACHE_DENSE_INDEX_LIMIT) {
            size_t candidate = (size_t)model_id + 1;
            if (candidate > index_limit) index_limit = candidate;
        }
    }
    return index_limit;
}

static void model_cache_init_index(ModelCache* cache, size_t limit) {
    if (limit == 0) return;
    if (limit > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "model_cache_load: model id index too large for %zu slots\n", limit);
        abort();
    }

    cache->index_by_id = (int*)osrs_malloc_or_abort(
        limit * sizeof(int), "model id index");
    cache->index_limit = limit;
    for (size_t i = 0; i < limit; i++) {
        cache->index_by_id[i] = -1;
    }
}

static void model_cache_set_index(ModelCache* cache, uint32_t model_id, int index) {
    if (!cache->index_by_id || (size_t)model_id >= cache->index_limit) return;
    cache->index_by_id[model_id] = index;
}

static ModelCache* model_cache_load(const char* path) {
    FILE* f = osrs_asset_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "model_cache_load: cannot open %s\n", path);
        return NULL;
    }

    uint32_t magic, count;
    osrs_read_exact(f, &magic, 4, 1, path, "magic");
    osrs_read_exact(f, &count, 4, 1, path, "model count");

    if (magic != MDL4_MAGIC) {
        fprintf(stderr, "model_cache_load: bad magic 0x%08X (expected MDL4) in %s\n",
                magic, path);
        abort();
    }

    uint32_t* offsets = (uint32_t*)osrs_malloc_or_abort(
        count * sizeof(uint32_t), "model offsets");
    osrs_read_exact(f, offsets, 4, count, path, "model offsets");

    if (count > (uint32_t)INT_MAX) {
        fprintf(stderr, "model_cache_load: too many models: %u\n", count);
        abort();
    }

    ModelCache* cache = (ModelCache*)osrs_calloc_or_abort(
        1, sizeof(ModelCache), "model cache");
    cache->models = (OsrsModel*)osrs_calloc_or_abort(
        count, sizeof(OsrsModel), "model entries");
    cache->count = (int)count;
    if (count > 0) {
        size_t index_limit = model_cache_index_limit_or_abort(f, offsets, count, path);
        model_cache_init_index(cache, index_limit);
    }
    cache->atlas_texture = model_cache_load_atlas(cache, path);
    cache->has_atlas = cache->atlas_texture.id > 0;
    if (!cache->has_atlas) {
        fprintf(stderr, "model_cache_load: MDL4 model set requires a sibling .atlas file: %s\n", path);
        abort();
    }

    for (uint32_t i = 0; i < count; i++) {
        osrs_seek_or_abort(f, (long)offsets[i], path);

        uint32_t model_id;
        uint16_t vert_count, face_count, base_vert_count;
        osrs_read_exact(f, &model_id, 4, 1, path, "model id");
        osrs_read_exact(f, &vert_count, 2, 1, path, "expanded vertex count");
        osrs_read_exact(f, &face_count, 2, 1, path, "face count");
        osrs_read_exact(f, &base_vert_count, 2, 1, path, "base vertex count");

        cache->models[i].model_id = model_id;
        cache->models[i].base_vert_count = base_vert_count;
        model_cache_set_index(cache, model_id, (int)i);

        Mesh mesh = { 0 };
        mesh.vertexCount = vert_count;
        mesh.triangleCount = face_count;

        mesh.vertices = (float*)RL_MALLOC(vert_count * 3 * sizeof(float));
        mesh.colors = (unsigned char*)RL_MALLOC(vert_count * 4);
        mesh.texcoords = (float*)RL_MALLOC(vert_count * 2 * sizeof(float));
        if (!mesh.vertices || !mesh.colors || !mesh.texcoords) {
            fprintf(stderr, "model_cache_load: raylib mesh allocation failed for model %u\n",
                model_id);
            abort();
        }

        osrs_read_exact(f, mesh.vertices, sizeof(float), vert_count * 3, path, "expanded vertices");
        osrs_read_exact(f, mesh.colors, 1, vert_count * 4, path, "vertex colors");
        osrs_read_exact(f, mesh.texcoords, sizeof(float), vert_count * 2, path, "texcoords");

        cache->models[i].base_vertices = (int16_t*)osrs_malloc_or_abort(
            base_vert_count * 3 * sizeof(int16_t), "model base vertices");
        osrs_read_exact(f, cache->models[i].base_vertices, sizeof(int16_t),
            base_vert_count * 3, path, "base vertices");

        cache->models[i].vertex_skins = (uint8_t*)osrs_malloc_or_abort(
            base_vert_count, "model vertex skins");
        osrs_read_exact(f, cache->models[i].vertex_skins, 1,
            base_vert_count, path, "vertex skins");

        cache->models[i].face_indices = (uint16_t*)osrs_malloc_or_abort(
            face_count * 3 * sizeof(uint16_t), "model face indices");
        osrs_read_exact(f, cache->models[i].face_indices, sizeof(uint16_t),
            face_count * 3, path, "face indices");

        cache->models[i].face_priorities = (uint8_t*)osrs_malloc_or_abort(
            face_count, "model face priorities");
        osrs_read_exact(f, cache->models[i].face_priorities, 1,
            face_count, path, "face priorities");

        cache->models[i].base_face_alphas = (uint8_t*)osrs_malloc_or_abort(
            face_count, "model base face alphas");
        osrs_read_exact(f, cache->models[i].base_face_alphas, 1,
            face_count, path, "base face alphas");
        cache->models[i].face_alpha_labels = (uint8_t*)osrs_malloc_or_abort(
            face_count, "model face alpha labels");
        osrs_read_exact(f, cache->models[i].face_alpha_labels, 1,
            face_count, path, "face alpha labels");

        for (uint16_t fp = 0; fp < face_count; fp++) {
            if (cache->models[i].base_face_alphas[fp] == 0 &&
                    mesh.colors[(fp * 3) * 4 + 3] == 0)
                cache->models[i].base_face_alphas[fp] = 255;
        }

        UploadMesh(&mesh, false);
        cache->models[i].mesh = mesh;
        cache->models[i].model = LoadModelFromMesh(mesh);
        if (cache->has_atlas) {
            cache->models[i].model.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture =
                cache->atlas_texture;
        }
    }

    free(offsets);
    fclose(f);

    fprintf(stderr, "model_cache_load: loaded %d models from %s\n", cache->count, path);
    return cache;
}

static OsrsModel* model_cache_get(ModelCache* cache, uint32_t model_id) {
    if (!cache) return NULL;
    if (cache->index_by_id && (size_t)model_id < cache->index_limit) {
        int idx = cache->index_by_id[model_id];
        if (idx >= 0 && idx < cache->count && cache->models[idx].model_id == model_id) {
            return &cache->models[idx];
        }
    }
    for (int i = 0; i < cache->count; i++) {
        if (cache->models[i].model_id == model_id) {
            return &cache->models[i];
        }
    }
    return NULL;
}

typedef struct {
    uint32_t hide_body_mask;
    uint32_t body_model_ids[BODY_PART_COUNT];
    uint32_t item_model_ids[NUM_GEAR_SLOTS];
    uint8_t body_visible[BODY_PART_COUNT];
    uint8_t item_visible[NUM_GEAR_SLOTS];
} OsrsPlayerAppearance;

#define OSRS_VISIBLE_EQUIP_SLOT_COUNT 9

static const int OSRS_VISIBLE_EQUIP_SLOTS[OSRS_VISIBLE_EQUIP_SLOT_COUNT] = {
    GEAR_SLOT_HEAD,
    GEAR_SLOT_CAPE,
    GEAR_SLOT_NECK,
    GEAR_SLOT_WEAPON,
    GEAR_SLOT_SHIELD,
    GEAR_SLOT_BODY,
    GEAR_SLOT_LEGS,
    GEAR_SLOT_HANDS,
    GEAR_SLOT_FEET,
};

static const ItemModelMapping* item_model_mapping_for_item(uint16_t item_id) {
    for (int i = 0; i < ITEM_MODEL_COUNT; i++) {
        if (ITEM_MODEL_MAP[i].item_id == item_id) {
            return &ITEM_MODEL_MAP[i];
        }
    }
    return NULL;
}

static uint32_t item_to_wield_model(uint16_t item_id) {
    const ItemModelMapping* mapping = item_model_mapping_for_item(item_id);
    if (!mapping) return ITEM_RENDER_MODEL_MISSING;
    return mapping->wield_model;
}

static uint32_t item_hide_body_mask(uint16_t item_id) {
    const ItemModelMapping* mapping = item_model_mapping_for_item(item_id);
    if (!mapping) return 0;
    return mapping->hide_body_mask;
}

static inline uint32_t item_render_equip_slot(uint16_t item_id) {
    const ItemModelMapping* mapping = item_model_mapping_for_item(item_id);
    if (!mapping) return ITEM_RENDER_MODEL_MISSING;
    return mapping->equip_slot;
}

static uint32_t item_render_flags(uint16_t item_id) {
    const ItemModelMapping* mapping = item_model_mapping_for_item(item_id);
    if (!mapping) return 0;
    return mapping->render_flags;
}

static uint32_t item_render_ready_anim(uint16_t item_id) {
    const ItemModelMapping* mapping = item_model_mapping_for_item(item_id);
    if (!mapping) return ITEM_RENDER_MODEL_MISSING;
    return mapping->ready_anim_id;
}

static uint32_t item_render_walk_anim(uint16_t item_id) {
    const ItemModelMapping* mapping = item_model_mapping_for_item(item_id);
    if (!mapping) return ITEM_RENDER_MODEL_MISSING;
    return mapping->walk_anim_id;
}

static uint32_t item_render_run_anim(uint16_t item_id) {
    const ItemModelMapping* mapping = item_model_mapping_for_item(item_id);
    if (!mapping) return ITEM_RENDER_MODEL_MISSING;
    return mapping->run_anim_id;
}

static int item_render_is_two_handed(uint16_t item_id) {
    return (item_render_flags(item_id) & ITEM_RENDER_FLAG_TWO_HANDED) != 0;
}

static OsrsPlayerAppearance osrs_resolve_player_appearance(
    const uint8_t equipped[NUM_GEAR_SLOTS]
) {
    OsrsPlayerAppearance out;
    memset(&out, 0, sizeof(out));

    for (int bp = 0; bp < BODY_PART_COUNT; bp++) {
        out.body_model_ids[bp] = ITEM_RENDER_MODEL_MISSING;
    }
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        out.item_model_ids[slot] = ITEM_RENDER_MODEL_MISSING;
    }

    uint8_t weapon_index = equipped[GEAR_SLOT_WEAPON];
    int suppress_shield = 0;
    if (weapon_index < NUM_ITEMS) {
        uint16_t weapon_item_id = ITEM_DATABASE[weapon_index].item_id;
        suppress_shield = item_is_two_handed(weapon_index) ||
            item_render_is_two_handed(weapon_item_id);
    }

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        if (slot == GEAR_SLOT_SHIELD && suppress_shield) continue;
        uint8_t db_idx = equipped[slot];
        if (db_idx >= NUM_ITEMS) continue;
        out.hide_body_mask |= item_hide_body_mask(ITEM_DATABASE[db_idx].item_id);
    }

    for (int bp = 0; bp < BODY_PART_COUNT; bp++) {
        uint32_t model_id = DEFAULT_BODY_MODELS[bp];
        out.body_model_ids[bp] = model_id;
        out.body_visible[bp] = ((out.hide_body_mask & (1u << bp)) == 0) &&
            model_id != ITEM_RENDER_MODEL_MISSING;
    }

    for (int i = 0; i < OSRS_VISIBLE_EQUIP_SLOT_COUNT; i++) {
        int slot = OSRS_VISIBLE_EQUIP_SLOTS[i];
        if (slot == GEAR_SLOT_SHIELD && suppress_shield) continue;
        uint8_t db_idx = equipped[slot];
        if (db_idx >= NUM_ITEMS) continue;
        uint16_t item_id = ITEM_DATABASE[db_idx].item_id;
        uint32_t model_id = item_to_wield_model(item_id);
        out.item_model_ids[slot] = model_id;
        out.item_visible[slot] = model_id != ITEM_RENDER_MODEL_MISSING;
    }

    return out;
}

static void model_cache_free(ModelCache* cache) {
    if (!cache) return;
    for (int i = 0; i < cache->count; i++) {
        UnloadModel(cache->models[i].model);
        free(cache->models[i].base_vertices);
        free(cache->models[i].vertex_skins);
        free(cache->models[i].face_indices);
        free(cache->models[i].face_priorities);
        free(cache->models[i].base_face_alphas);
        free(cache->models[i].face_alpha_labels);
    }
    if (cache->atlas_texture.id > 0) UnloadTexture(cache->atlas_texture);
    free(cache->atlas_base_pixels);
    free(cache->atlas_pixels);
    free(cache->texture_anims);
    free(cache->index_by_id);
    free(cache->models);
    free(cache);
}

#endif
